import argparse, os, time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from .datasets import get_cifar10, get_purchase_synth, split_train_val
from .sisa import sisa_train
from .utils import ensure_dir, set_seed
from .mia import confidence_scores, mia_auc

def _eval_metrics(model, testset, batch_size=512, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ys, ps = [], []
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y if torch.is_tensor(y) else torch.tensor(y)
            p = model(x).argmax(dim=1).cpu()
            ys.append(y.cpu()); ps.append(p)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    acc = float((y_true == y_pred).mean())
    f1  = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1

def _mia_auc_for(model, trainset_subset, testset_subset, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_loader  = DataLoader(trainset_subset, batch_size=512, shuffle=False)
    out_loader = DataLoader(testset_subset,  batch_size=512, shuffle=False)
    in_scores  = confidence_scores(model, in_loader, device)
    out_scores = confidence_scores(model, out_loader, device)
    return float(mia_auc(in_scores, out_scores))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","purchase_synth"], default="purchase_synth")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--delete_fracs", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out_dir", default="results/logs")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Load full datasets
    if args.dataset == "cifar10":
        train_full, _, _ = get_cifar10(args.data_root, train=True, download=True)
        testset, _, _   = get_cifar10(args.data_root, train=False, download=True)
        model_name = "cnn"
    else:
        train_full, _, _ = get_purchase_synth(args.data_root, train=True)
        testset, _, _    = get_purchase_synth(args.data_root, train=False)
        model_name = "mlp"

    # We will select deletions from the training split used for SISA baseline
    trainset, valset = split_train_val(train_full, val_fraction=0.1, seed=args.seed)
    n = len(trainset)
    base_indices = np.arange(n)

    rng = np.random.default_rng(args.seed + 9001)

    for frac in args.delete_fracs:
        dcnt = max(1, int(round(n * frac)))
        delete_idx_local = rng.choice(n, size=dcnt, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[delete_idx_local] = False
        keep_indices = base_indices[keep_mask].tolist()

        # Train a fresh SISA model from scratch on the remaining data
        print(f"[Naive] Full retrain after deletion: {frac*100:.1f}% (kept {len(keep_indices)} of {n})")
        t0 = time.perf_counter()
        train_kept = Subset(trainset, keep_indices)

        # fresh run, fresh ckpts per percentage
        ckpt_dir = os.path.join(args.out_dir, f"ckpts_naive_{int(frac*100)}pct")
        ensure_dir(ckpt_dir)

        naive_model, _ = sisa_train(
            dataset=train_kept, valset=valset, testset=testset, model_name=model_name,
            K=args.K, S=args.S, batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
            seed=args.seed, log_path=None, checkpoint_dir=ckpt_dir
        )
        time_elapsed = time.perf_counter() - t0

        # Evaluate
        acc_test, f1_test = _eval_metrics(naive_model, testset)
        # MIA: use membership = kept-train subset
        mia_val = _mia_auc_for(
            naive_model,
            Subset(train_kept, range(min(5000, len(train_kept)))),
            Subset(testset,    range(min(5000, len(testset))))
        )

        print(f"[Naive] %del={frac*100:.1f}  time={time_elapsed:.1f}s  acc={acc_test:.4f}  f1={f1_test:.4f}  mia={mia_val:.6f}")

        # Log ONE aggregate row per deletion %
        csv_path = os.path.join(args.out_dir, f"naive_{int(frac*100)}pct.csv")
        row = {
            "phase": "aggregate",
            "percent_deleted": frac * 100.0,
            "time_s": time_elapsed,
            "acc_test": acc_test,
            "f1_test": f1_test,
            "mia_auc": mia_val,
        }
        # header if file doesn't exist
        hdr = not os.path.exists(csv_path)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=hdr)

if __name__ == "__main__":
    main()
