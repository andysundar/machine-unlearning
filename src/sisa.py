# sisa.py (patched)
import os, time, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .models import SmallCNN, SmallMLP
from .utils import set_seed, device_pick, CSVLogger, ensure_dir

# Keep CPU safe
torch.set_default_dtype(torch.float32)

# ----------------------- helpers -----------------------
def infer_in_shape(ds):
    """Infer input shape from the first sample of a dataset."""
    x0, _ = ds[0]
    if torch.is_tensor(x0):
        return tuple(x0.shape)
    try:
        return (int(np.prod(x0.shape)),)
    except Exception:
        return None

def build_model(model_name: str, num_classes: int, in_shape=None) -> nn.Module:
    """Clean model factory (no device/dtype kwargs)."""
    name = (model_name or "").lower()
    if name in ("cnn", "smallcnn"):
        return SmallCNN(num_classes=num_classes)
    if name in ("mlp", "smallmlp"):
        if isinstance(in_shape, int):
            in_dim = in_shape
        elif isinstance(in_shape, (tuple, list)):
            in_dim = int(np.prod(in_shape))
        elif in_shape is None:
            in_dim = 600
        else:
            try:
                in_dim = int(in_shape)
            except Exception:
                in_dim = 600
        return SmallMLP(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model_name={model_name!r}")

@torch.no_grad()
def _eval_acc(model, dataset, batch_size=512, device=None):
    device = device or device_pick()
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total=0; correct=0
    for x,y in loader:
        x = x.to(device).float()
        y = y if torch.is_tensor(y) else torch.tensor(y)
        pred = model(x).argmax(dim=1).cpu()
        y = y.cpu()
        total += y.numel(); correct += (pred==y).sum().item()
    return (correct/total) if total else 0.0

def _train_one_slice(model, train_subset, valset, testset, epochs, batch_size, lr, device, logger, k, s):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    for e in range(epochs):
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = (yb if torch.is_tensor(yb) else torch.tensor(yb)).to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        dt = time.perf_counter() - t0
        acc_val = _eval_acc(model, valset, batch_size, device)
        acc_test = _eval_acc(model, testset, batch_size, device)
        if logger:
            logger.log({"phase":"train","k":k,"s":s,"epoch_s":e+1,"n_slice":len(train_subset),
                        "acc_val":f"{acc_val:.6f}","acc_test":f"{acc_test:.6f}","time_s":f"{dt:.6f}","note":""})

# ----------------------- layout & ckpt -----------------------
def make_shards_slices(n, K, S, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    shards=[]; offs=0
    for k in range(K):
        size = (n + k) // K
        shard_idx = perm[offs:offs+size]; offs += size
        shard_slices=[]; so=0
        for s in range(S):
            ss = (size + s) // S
            shard_slices.append(shard_idx[so:so+ss]); so += ss
        shards.append(shard_slices)
    return shards

def checkpoint_path(ckpt_dir, k, s): return os.path.join(ckpt_dir, f"ckpt_k{k}_s{s}.pt")
def save_ckpt(model, path): ensure_dir(os.path.dirname(path)); torch.save({"state_dict":model.state_dict()}, path)
def load_ckpt(model, path, device): sd=torch.load(path,map_location=device); model.load_state_dict(sd["state_dict"]); return model

def aggregate_models(models):
    base = models[0]
    with torch.no_grad():
        for name, p in base.state_dict().items():
            stack = torch.stack([m.state_dict()[name].to(p.device) for m in models], dim=0)
            p.copy_(stack.mean(dim=0))
    return base

def impacted_slices(layout, delete_indices):
    delset=set(delete_indices); imp=[]
    for k, shard in enumerate(layout):
        for s, idx in enumerate(shard):
            if delset.intersection(idx): imp.append((k,s))
    return imp

# ----------------------- SISA train & unlearn -----------------------
def sisa_train(dataset, valset, testset, model_name, K, S, batch_size, epochs_per_slice, lr, seed, log_path, checkpoint_dir):
    device = device_pick(); set_seed(seed)
    n=len(dataset); layout = make_shards_slices(n,K,S,seed)
    logger = CSVLogger(log_path, fieldnames=["phase","k","s","epoch_s","n_slice","acc_val","acc_test","time_s","note"]) if log_path else None

    # robust shape & class inference once
    in_shape = getattr(dataset, 'shape', None) or infer_in_shape(dataset)
    num_classes = getattr(dataset, 'classes', None) or 10

    shard_final=[]

    for k in range(K):
        m = build_model(model_name, num_classes=num_classes, in_shape=in_shape).to(device).float()
        for s in range(S):
            idx = layout[k][s]
            if not idx: continue
            _train_one_slice(m, Subset(dataset, idx), valset, testset, epochs_per_slice, batch_size, lr, device, logger, k, s)
            if checkpoint_dir: save_ckpt(m, checkpoint_path(checkpoint_dir,k,s))
        shard_final.append(m)

    agg = aggregate_models(shard_final)
    if logger:
        logger.log({"phase":"aggregate","k":"-","s":"-","epoch_s":"-","n_slice":"-",
                    "acc_val":f"{_eval_acc(agg,valset,batch_size,device):.6f}",
                    "acc_test":f"{_eval_acc(agg,testset,batch_size,device):.6f}",
                    "time_s":"0.0","note":"baseline-aggregate"})
        logger.close()
    return agg, layout

def sisa_unlearn(dataset, valset, testset, model_name, K, S, layout, delete_indices, batch_size, epochs_per_slice, lr, seed, log_path, checkpoint_dir):
    device = device_pick(); set_seed(seed)
    logger = CSVLogger(log_path, fieldnames=["phase","k","s","epoch_s","n_slice","acc_val","acc_test","time_s","note"]) if log_path else None

    # same robust inference used here
    in_shape = getattr(dataset, 'shape', None) or infer_in_shape(dataset)
    num_classes = getattr(dataset, 'classes', None) or 10

    impact = impacted_slices(layout, delete_indices)
    first_imp = {}
    for k,s in impact: first_imp[k] = min(s, first_imp.get(k,s))
    retrained=0; shard_final=[]

    for k in range(K):
        if k not in first_imp:
            m = build_model(model_name, num_classes=num_classes, in_shape=in_shape).to(device).float()
            last = checkpoint_path(checkpoint_dir,k,S-1) if checkpoint_dir else None
            if last and os.path.exists(last): load_ckpt(m,last,device); m = m.to(device).float()
            else:
                for s in range(S):
                    idx = layout[k][s]
                    if not idx: continue
                    _train_one_slice(m, Subset(dataset, idx), valset, testset, epochs_per_slice, batch_size, lr, device, logger, k, s)
            shard_final.append(m); continue

        s0 = first_imp[k]
        m = build_model(model_name, num_classes=num_classes, in_shape=in_shape).to(device).float()
        if s0>0 and checkpoint_dir:
            prev = checkpoint_path(checkpoint_dir,k,s0-1)
            if os.path.exists(prev): load_ckpt(m,prev,device); m = m.to(device).float()

        for s in range(s0, S):
            idx = [i for i in layout[k][s] if i not in delete_indices]
            if not idx: continue
            _train_one_slice(m, Subset(dataset, idx), valset, testset, epochs_per_slice, batch_size, lr, device, logger, k, s)
            retrained += 1
            if checkpoint_dir: save_ckpt(m, checkpoint_path(checkpoint_dir,k,s))
        shard_final.append(m)

    agg = aggregate_models(shard_final)
    if logger:
        logger.log({"phase":"aggregate","k":"-","s":"-","epoch_s":"-","n_slice":"-",
                    "acc_val":f"{_eval_acc(agg,valset,batch_size,device):.6f}",
                    "acc_test":f"{_eval_acc(agg,testset,batch_size,device):.6f}",
                    "time_s":"0.0","note":"unlearn-aggregate"})
        logger.close()
    return agg, retrained
