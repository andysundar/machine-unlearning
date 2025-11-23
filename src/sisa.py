# src/sisa.py
import os
import time
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .models import SmallCNN, SmallMLP
from .utils import set_seed, device_pick, CSVLogger, ensure_dir

# lock default to float32 (avoids NNPack dtype mismatch on CPU)
torch.set_default_dtype(torch.float32)

# ----------------------- model helpers -----------------------

def build_model(model_name: str, num_classes: int, in_shape=None) -> nn.Module:
    if model_name.lower() in ("cnn", "smallcnn"):
        return SmallCNN(num_classes=num_classes)
    elif model_name.lower() in ("mlp", "smallmlp"):
        in_dim = (in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape) or 600
        return SmallMLP(in_dim=in_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name={model_name}")

def _eval_acc(model: nn.Module, dataset, batch_size=512, device=None) -> float:
    device = device or device_pick()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y if torch.is_tensor(y) else torch.tensor(y)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            y = y.cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return (correct / total) if total else 0.0

def _train_one_slice(model: nn.Module, train_subset, valset, testset,
                     epochs: int, batch_size: int, lr: float,
                     device, logger: Optional[CSVLogger], k: int, s: int) -> None:
    model.train()
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb if torch.is_tensor(yb) else torch.tensor(yb)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = nn.functional.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        dt = time.perf_counter() - t0
        acc_val = _eval_acc(model, valset, batch_size=batch_size, device=device)
        acc_test = _eval_acc(model, testset, batch_size=batch_size, device=device)
        if logger:
            logger.log({
                "phase":"train", "k":k, "s":s, "epoch_s":e+1,
                "n_slice": len(train_subset),
                "acc_val": f"{acc_val:.6f}", "acc_test": f"{acc_test:.6f}",
                "time_s": f"{dt:.6f}", "note": ""
            })

# ----------------------- SISA core -----------------------

def make_shards_slices(n: int, K: int, S: int, seed: int) -> List[List[List[int]]]:
    """
    Returns layout[K][S] -> indices per (shard, slice), chronological within each shard.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    shards = []
    shard_sizes = [(n + i) // K for i in range(K)]  # near-even split
    offs = 0
    for k in range(K):
        size = shard_sizes[k]
        shard_idx = perm[offs:offs+size]
        offs += size
        # chronological slicing inside shard
        slice_sizes = [(size + i) // S for i in range(S)]
        so = 0
        shard_slices = []
        for s in range(S):
            ss = slice_sizes[s]
            shard_slices.append(shard_idx[so:so+ss])
            so += ss
        shards.append(shard_slices)
    return shards

def checkpoint_path(ckpt_dir: str, k: int, s: int) -> str:
    return os.path.join(ckpt_dir, f"ckpt_k{k}_s{s}.pt")

def save_ckpt(model: nn.Module, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save({"state_dict": model.state_dict()}, path)

def load_ckpt(model: nn.Module, path: str, device) -> nn.Module:
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd["state_dict"])
    return model

def aggregate_models(models: List[nn.Module]) -> nn.Module:
    """Simple weight-wise average across models."""
    base = models[0]
    with torch.no_grad():
        for name, p in base.state_dict().items():
            stack = torch.stack([m.state_dict()[name].to(p.device) for m in models], dim=0)
            p.copy_(stack.mean(dim=0))
    return base

def impacted_slices(layout: List[List[List[int]]], delete_indices: List[int]) -> List[Tuple[int,int]]:
    impacted = []
    delset = set(delete_indices)
    for k, shard in enumerate(layout):
        for s, idxs in enumerate(shard):
            if delset.intersection(idxs):
                impacted.append((k, s))
    return impacted

def sisa_train(dataset, valset, testset, model_name: str,
               K: int, S: int, batch_size: int, epochs_per_slice: int, lr: float,
               seed: int, log_path: Optional[str], checkpoint_dir: Optional[str]):
    """
    Train SISA: sequentially per shard, carrying weights forward within a shard.
    Returns (aggregated_model, layout)
    """
    device = device_pick()
    set_seed(seed)
    n = len(dataset)
    layout = make_shards_slices(n, K, S, seed)
    logger = CSVLogger(log_path, fieldnames=[
        "phase","k","s","epoch_s","n_slice","acc_val","acc_test","time_s","note"
    ]) if log_path else None

    # track last model of each shard for aggregation
    shard_final_models = []

    for k in range(K):
        # fresh model at start of shard
        m = build_model(model_name, num_classes=getattr(dataset, 'classes', None) or getattr(testset.dataset if hasattr(testset,'dataset') else testset, 'classes', None) or 10,
                        in_shape=getattr(dataset, 'shape', None))
        m = m.to(device).float()

        for s in range(S):
            slice_idx = layout[k][s]
            if not slice_idx:
                continue
            sub = Subset(dataset, slice_idx)
            _train_one_slice(m, sub, valset, testset, epochs_per_slice, batch_size, lr, device, logger, k, s)
            # save checkpoint after this slice
            if checkpoint_dir:
                save_ckpt(m, checkpoint_path(checkpoint_dir, k, s))

        # store final shard model (deep copy to CPU for aggregation safety)
        shard_final_models.append(m)

    # aggregate
    agg = aggregate_models(shard_final_models)
    # final baseline log
    if logger:
        logger.log({"phase":"aggregate","k":"-","s":"-","epoch_s":"-","n_slice":"-",
                    "acc_val":f"{_eval_acc(agg, valset, batch_size, device):.6f}",
                    "acc_test":f"{_eval_acc(agg, testset, batch_size, device):.6f}",
                    "time_s":"0.0","note":"baseline-aggregate"})
        logger.close()
    return agg, layout

def sisa_unlearn(dataset, valset, testset, model_name: str,
                 K: int, S: int, layout: List[List[List[int]]], delete_indices: List[int],
                 batch_size: int, epochs_per_slice: int, lr: float,
                 seed: int, log_path: Optional[str], checkpoint_dir: Optional[str]):
    """
    Selective retraining: for each impacted shard, resume from the checkpoint BEFORE
    the earliest impacted slice, then replay from that slice onward with deletions removed.
    Returns (aggregated_model, retrained_slices_count)
    """
    device = device_pick()
    set_seed(seed)
    logger = CSVLogger(log_path, fieldnames=[
        "phase","k","s","epoch_s","n_slice","acc_val","acc_test","time_s","note"
    ]) if log_path else None

    impact = impacted_slices(layout, delete_indices)
    # earliest impacted slice per shard
    first_impacted = {}
    for k, s in impact:
        first_impacted[k] = min(s, first_impacted.get(k, s))

    retrained_slices = 0
    shard_final_models = []

    for k in range(K):
        # if shard not impacted: reuse its final checkpoint (s = S-1) if available,
        # else retrain from scratch on whole shard (rare path)
        if k not in first_impacted:
            # reuse last checkpoint
            m = build_model(model_name, num_classes=getattr(dataset,'classes',None) or 10,
                            in_shape=getattr(dataset,'shape',None)).to(device).float()
            last_ckpt = checkpoint_path(checkpoint_dir, k, S-1) if checkpoint_dir else None
            if last_ckpt and os.path.exists(last_ckpt):
                load_ckpt(m, last_ckpt, device)
            else:
                # train from scratch on the whole shard if no ckpt (safety)
                for s in range(S):
                    idx = layout[k][s]
                    if not idx: continue
                    _train_one_slice(m, Subset(dataset, idx), valset, testset,
                                     epochs_per_slice, batch_size, lr, device, logger, k, s)
            shard_final_models.append(m)
            continue

        # impacted: resume from s0-1 (or fresh if s0==0), then replay s0..S-1 with deletions removed
        s0 = first_impacted[k]
        m = build_model(model_name, num_classes=getattr(dataset,'classes',None) or 10,
                        in_shape=getattr(dataset,'shape',None)).to(device).float()
        if s0 > 0 and checkpoint_dir:
            prev_ckpt = checkpoint_path(checkpoint_dir, k, s0-1)
            if os.path.exists(prev_ckpt):
                load_ckpt(m, prev_ckpt, device)

        for s in range(s0, S):
            # filter out deleted indices from this slice
            slice_idx = [i for i in layout[k][s] if i not in delete_indices]
            if not slice_idx:
                continue
            sub = Subset(dataset, slice_idx)
            _train_one_slice(m, sub, valset, testset, epochs_per_slice, batch_size, lr, device, logger, k, s)
            retrained_slices += 1
            if checkpoint_dir:
                save_ckpt(m, checkpoint_path(checkpoint_dir, k, s))
        shard_final_models.append(m)

    agg = aggregate_models(shard_final_models)
    if logger:
        logger.log({"phase":"aggregate","k":"-","s":"-","epoch_s":"-","n_slice":"-",
                    "acc_val":f"{_eval_acc(agg, valset, batch_size, device):.6f}",
                    "acc_test":f"{_eval_acc(agg, testset, batch_size, device):.6f}",
                    "time_s":"0.0","note":"unlearn-aggregate"})
        logger.close()
    return agg, retrained_slices
