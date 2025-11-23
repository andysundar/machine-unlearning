# src/run_experiments.py
import argparse, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

from .datasets import get_cifar10, get_purchase_synth, split_train_val
from .sisa import sisa_train, sisa_unlearn
from .utils import ensure_dir, set_seed
from .mia import confidence_scores, mia_auc

# dtype safety for CPU NNPack
torch.set_default_dtype(torch.float32)

# ---------------------- Eval helpers ----------------------

def eval_metrics(model, testset, batch_size=512, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else
                        (torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                         else torch.device("cpu")))
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y if torch.is_tensor(y) else torch.tensor(y)
            p = model(x).argmax(dim=1).cpu()
            ys.append(y.cpu()); preds.append(p)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1

def mia_auc_for(model, trainset_subset, testset_subset, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else
                        (torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                         else torch.device("cpu")))
    in_loader  = DataLoader(trainset_subset, batch_size=512, shuffle=False)
    out_loader = DataLoader(testset_subset,  batch_size=512, shuffle=False)
    in_scores  = confidence_scores(model, in_loader, device)
    out_scores = confidence_scores(model, out_loader, device)
    return float(mia_auc(in_scores, out_scores))

# ---------------------- Deletion modes ----------------------

def _select_delete_indices(layout, n, frac, mode, few_shards_k, rng):
    dcnt = max(1, int(round(n * frac)))
    if mode == "uniform" or layout is None:
        return rng.choice(n, size=dcnt, replace=False).tolist()

    K = len(layout)
    S = len(layout[0]) if K > 0 else 0

    if mode == "last_slice":
        candidates = []
        for k in range(K):
            if S == 0: continue
            candidates.extend(layout[k][S-1])
        if len(candidates) < dcnt:
            chosen = rng.choice(candidates, size=dcnt, replace=True).tolist()
            return list(dict.fromkeys(chosen))[:dcnt]
        return rng.choice(candidates, size=dcnt, replace=False).tolist()

    if mode == "few_shards":
        ks = min(max(1, few_shards_k if few_shards_k is not None else max(1, K//3)), K)
        affected = rng.choice(K, size=ks, replace=False)
        candidates = []
        for k in affected:
            for s in range(S):
                candidates.extend(layout[k][s])
        if len(candidates) < dcnt:
            chosen = rng.choice(candidates, size=dcnt, replace=True).tolist()
            return list(dict.fromkeys(chosen))[:dcnt]
        return rng.choice(candidates, size=dcnt, replace=False).tolist()

    return rng.choice(n, size=dcnt, replace=False).tolist()

# ---------------------- Plot helpers ----------------------

def fig1_acc_vs_deleted(df, out_png, subtitle=None):
    import matplotlib.pyplot as plt
    plt.figure()

    # X values are the deletion %s present in SISA-unlearned (if none, fall back to any >0 in df)
    xs = sorted({float(x) for x in df["percent_deleted"].unique() if float(x) > 0.0})
    if not xs:
        plt.xlabel("% deleted"); plt.ylabel("Test accuracy")
        t = "Fig-1: Test accuracy vs % deleted"
        if subtitle: t += f"\n{subtitle}"
        plt.title(t); plt.savefig(out_png, bbox_inches="tight"); plt.close(); return

    # 1) Baseline: horizontal line at baseline acc across xs
    base = df[df["method"]=="SISA-baseline"]
    if not base.empty:
        base_acc = float(base.iloc[0]["accuracy"])
        plt.plot(xs, [base_acc]*len(xs), marker="o", label="SISA-baseline")

    # 2) Naive (prefer true, else sim)
    naive_true = df[df["method"]=="Naive-delete"].sort_values("percent_deleted")
    if not naive_true.empty:
        plt.plot(naive_true["percent_deleted"], naive_true["accuracy"], marker="o", label="Naive-delete")
    else:
        naive_sim = df[df["method"]=="Naive-delete-sim"].sort_values("percent_deleted")
        if not naive_sim.empty:
            plt.plot(naive_sim["percent_deleted"], naive_sim["accuracy"], marker="o", label="Naive-delete-sim")

    # 3) SISA
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if not sisa.empty:
        plt.plot(sisa["percent_deleted"], sisa["accuracy"], marker="o", label="SISA-unlearned")

    plt.xlabel("% deleted"); plt.ylabel("Test accuracy")
    title = "Fig-1: Test accuracy vs % deleted"
    if subtitle: title += f"\n{subtitle}"
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


def fig2_time_saved_vs_naive(df, out_png, subtitle=None):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure()

    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    xs = sisa["percent_deleted"].tolist()
    if not xs:
        plt.xlabel("% deleted"); plt.ylabel("Time saved (s)")
        t = "Fig-2: Time saved vs Naive-delete"
        if subtitle: t += f"\n{subtitle}"
        plt.title(t); plt.savefig(out_png, bbox_inches="tight"); plt.close(); return

    # Prefer true naive times; else use baseline time as the “full retrain” time
    naive_df = df[df["method"]=="Naive-delete"].set_index("percent_deleted")
    if not naive_df.empty:
        full_times = [float(naive_df.loc[x, "time"]) if x in naive_df.index else np.nan for x in xs]
        # fill missing with baseline time
        base = df[df["method"]=="SISA-baseline"]
        baseline_time = float(base.iloc[0]["time"]) if not base.empty else max(sisa["time"].tolist())
        full_times = [baseline_time if (isinstance(t, float) and np.isnan(t)) else t for t in full_times]
    else:
        base = df[df["method"]=="SISA-baseline"]
        baseline_time = float(base.iloc[0]["time"]) if not base.empty else max(sisa["time"].tolist())
        full_times = [baseline_time for _ in xs]

    sisa_times = sisa["time"].astype(float).tolist()
    time_saved = [max(ft - st, 0.0) for ft, st in zip(full_times, sisa_times)]

    # Plot: SISA time-saved curve and a zero line for naive
    plt.plot(xs, time_saved, marker="o", linewidth=2, label="SISA-unlearned")
    plt.plot(xs, [0.0]*len(xs), marker="o", linestyle="--", linewidth=2, label="Naive-delete")

    plt.xlabel("% deleted"); plt.ylabel("Time saved (s)")
    t = "Fig-2: Time saved vs Naive-delete"
    if subtitle: t += f"\n{subtitle}"
    plt.title(t); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def fig3_mia_auc_grouped(df, out_png, subtitle=None):
    xs = sorted([x for x in df["percent_deleted"].unique().tolist() if x != 0.0])
    methods = ["SISA-baseline", "SISA-unlearned"]
    if (df["method"]=="Naive-delete").any():
        methods.append("Naive-delete")
    elif (df["method"]=="Naive-delete-sim").any():
        methods.append("Naive-delete-sim")

    width = 0.8 / max(len(methods), 1)
    x0 = np.arange(len(xs))

    plt.figure()
    for i, m in enumerate(methods):
        sub = (df[df["method"] == m]
               .set_index("percent_deleted")["mia_auc"]
               .reindex(xs))
        ys = sub.astype(float).fillna(0.0).values
        xpos = x0 + i*width
        plt.bar(xpos, ys, width=width, label=m)

    plt.xticks(x0 + width*(len(methods)-1)/2, [str(x) for x in xs])
    plt.xlabel("% deleted"); plt.ylabel("MIA AUC")
    t = "Fig-3: MIA AUC comparison"
    if subtitle: t += f"\n{subtitle}"
    plt.title(t)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# ---------------------- Report ----------------------

def build_pdf_report(df, cfg, paths):
    out_pdf = paths["out_pdf"]
    ensure_dir(os.path.dirname(out_pdf))

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(name="H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=12)
    h2 = ParagraphStyle(name="H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=6)
    body = styles["BodyText"]
    center_small = ParagraphStyle(name="CenterSmall", parent=styles["BodyText"], alignment=TA_CENTER, leading=10, fontSize=9)

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    flow = []

    flow.append(Paragraph("Machine Unlearning with SISA", h1))
    desc = ("SISA baseline, selective retraining with deletions, and a confidence-based membership inference check. "
            "We report accuracy/F1, time saved vs Naive, MIA AUC, and retrained-slices fraction.")
    flow.append(Paragraph(desc, body))
    cfg_line = (
        f"Dataset={cfg['dataset']} | K={cfg['K']} S={cfg['S']} | epochs/slice={cfg['epochs']} | "
        f"batch={cfg['batch_size']} | seed={cfg['seed']} | deletion={cfg.get('deletion_mode','uniform')}"
        f"{(' | few_shards_k='+str(cfg.get('few_shards_k')) if cfg.get('few_shards_k') is not None else '')}"
        f"{(' | true_naive' if cfg.get('run_true_naive') else '')}"
    )
    flow.append(Spacer(1, 8))
    flow.append(Paragraph("<b>Run configuration</b>", styles["Heading3"]))
    flow.append(Paragraph(cfg_line, body))

    base = df[df["method"]=="SISA-baseline"].head(1)
    if not base.empty:
        b = base.iloc[0]
        bl = f"Baseline: acc={b['accuracy']:.4f}, F1={b['f1']:.4f}, time={b['time']:.1f}s, MIA AUC={b['mia_auc']:.6f}"
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(bl, body))

    flow.append(PageBreak())

    flow.append(Paragraph("Fig-1: Test accuracy vs % deleted", h2))
    if os.path.exists(paths["fig1"]):
        flow.append(Image(paths["fig1"], width=14*cm, height=10*cm))
    flow.append(PageBreak())

    flow.append(Paragraph("Fig-2: Time saved vs Naive-delete", h2))
    if os.path.exists(paths["fig2"]):
        flow.append(Image(paths["fig2"], width=14*cm, height=10*cm))
    flow.append(PageBreak())

    flow.append(Paragraph("Fig-3: MIA AUC comparison", h2))
    if os.path.exists(paths["fig3"]):
        flow.append(Image(paths["fig3"], width=14*cm, height=8*cm))
        flow.append(Spacer(1, 10))

    flow.append(Paragraph("Table-1: Metrics summary", h2))

    show = df.copy().sort_values(["method","percent_deleted"])
    show["percent_deleted"] = show["percent_deleted"].map(lambda v: f"{float(v):.1f}")
    show["accuracy"] = show["accuracy"].map(lambda v: f"{float(v):.4f}")
    show["f1"]       = show["f1"].map(lambda v: f"{float(v):.4f}")
    show["time"]     = show["time"].map(lambda v: f"{float(v):.1f}")
    show["mia_auc"]  = show["mia_auc"].map(lambda v: f"{float(v):.6f}")
    show["retrained_slices_frac"] = show["retrained_slices_frac"].map(lambda v: f"{float(v):.2f}")

    cols = ["method","percent_deleted","accuracy","f1","time","mia_auc","retrained_slices_frac"]
    header = [
        Paragraph("method", center_small),
        Paragraph("percent<br/>deleted", center_small),
        Paragraph("accuracy", center_small),
        Paragraph("f1", center_small),
        Paragraph("time (s)", center_small),
        Paragraph("MIA AUC", center_small),
        Paragraph("retrained slices frac", center_small),
    ]
    data = [header] + [ [show[c].iloc[i] for c in cols] for i in range(len(show)) ]

    table = Table(
        data,
        colWidths=[4*cm, 2.3*cm, 2*cm, 2*cm, 2.7*cm, 3*cm, 3.8*cm],
        repeatRows=1
    )
    table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.75, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEADING", (0,0), (-1,0), 10),
    ]))
    flow.append(table)

    doc.build(flow)
    print(f"Saved report to {out_pdf}")

# ---------------------- Main driver ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","purchase_synth"], default="purchase_synth")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--delete_fracs", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--run_true_naive", action="store_true",
                    help="Also perform true Naive-delete full retrain per deletion %. Slower.")
    ap.add_argument("--include_naive_sim", action="store_true", default=True,
                    help="Append Naive-delete-sim rows per deletion % (baseline metrics & time).")
    ap.add_argument("--deletion_mode", choices=["uniform","last_slice","few_shards"], default="uniform",
                    help="How to select deletion indices across slices/shards.")
    ap.add_argument("--few_shards_k", type=int, default=None,
                    help="Number of shards to target when deletion_mode=few_shards.")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir); ensure_dir(os.path.join(args.out_dir, "logs")); ensure_dir(os.path.join(args.out_dir, "plots"))
    logs_dir = os.path.join(args.out_dir, "logs")
    ckpt_dir = os.path.join(args.out_dir, "ckpts")

    # Load data
    if args.dataset == "cifar10":
        train_full, _, _ = get_cifar10(args.data_root, train=True, download=True)
        testset, _, _   = get_cifar10(args.data_root, train=False, download=True)
        model_name = "cnn"
    else:
        train_full, _, _ = get_purchase_synth(args.data_root, train=True)
        testset, _, _    = get_purchase_synth(args.data_root, train=False)
        model_name = "mlp"

    # Split
    trainset, valset = split_train_val(train_full, val_fraction=0.1, seed=args.seed)

    # -------- Baseline --------
    print("running baseline SISA train...")
    t0 = time.perf_counter()
    model, layout = sisa_train(
        dataset=trainset, valset=valset, testset=testset,
        model_name=model_name, K=args.K, S=args.S,
        batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
        seed=args.seed, log_path=os.path.join(logs_dir, "baseline.csv"),
        checkpoint_dir=ckpt_dir
    )
    t_baseline = time.perf_counter() - t0
    acc_b, f1_b = eval_metrics(model, testset)
    mia_b = mia_auc_for(model,
                        Subset(trainset, range(min(5000, len(trainset)))),
                        Subset(testset, range(min(5000, len(testset)))))

    print(f"SISA baseline training done. Total time: {t_baseline:.1f}s")
    print(f"Eval Acc {acc_b:.4f} F1 {f1_b:.6f}")
    print(f"MIA AUC: {mia_b:.6f}")

    rows = [{
        "method":"SISA-baseline", "percent_deleted":0.0,
        "accuracy":acc_b, "f1":f1_b, "time":t_baseline,
        "mia_auc":mia_b, "retrained_slices_frac":0.00
    }]

    # -------- Deletions --------
    n = len(trainset)
    rng = np.random.default_rng(args.seed+2025)
    for frac in args.delete_fracs:
        delete_idx = _select_delete_indices(layout, n, frac, args.deletion_mode, args.few_shards_k, rng)
        print(f"Running deletion scenario: {frac*100:.1f}%")

        t0 = time.perf_counter()
        model_u, retrained = sisa_unlearn(
            dataset=trainset, valset=valset, testset=testset, model_name=model_name,
            K=args.K, S=args.S, layout=layout, delete_indices=delete_idx,
            batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
            seed=args.seed, log_path=os.path.join(logs_dir, f"unlearn_{int(frac*100)}pct.csv"),
            checkpoint_dir=ckpt_dir
        )
        t_un = time.perf_counter() - t0

        acc_u, f1_u = eval_metrics(model_u, testset)
        mia_u = mia_auc_for(model_u,
                            Subset(trainset, range(min(5000, len(trainset)))),
                            Subset(testset, range(min(5000, len(testset)))))

        print(f"Eval Acc {acc_u:.4f} F1 {f1_u:.6f}")
        print(f"MIA AUC: {mia_u:.6f}")

        rows.append({
            "method":"SISA-unlearned", "percent_deleted":frac*100,
            "accuracy":acc_u, "f1":f1_u, "time":t_un,
            "mia_auc":mia_u, "retrained_slices_frac":retrained/(args.K*args.S)
        })

        # Simulated Naive (full retrain ~= baseline cost & metrics)
        if args.include_naive_sim:
            rows.append({
                "method": "Naive-delete-sim",
                "percent_deleted": frac*100,
                "accuracy": acc_b,
                "f1": f1_b,
                "time": t_baseline,
                "mia_auc": mia_b,
                "retrained_slices_frac": 1.00
            })

        # True Naive (optional): full retrain on kept data
        if args.run_true_naive:
            base_indices = np.arange(n)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[delete_idx] = False
            keep_indices = base_indices[keep_mask].tolist()

            train_kept = Subset(trainset, keep_indices)
            ckpt_dir_naive = os.path.join(ckpt_dir, f"naive_{int(frac*100)}pct")
            ensure_dir(ckpt_dir_naive)

            print(f"[Naive] Full retrain after deletion {frac*100:.1f}% (kept {len(keep_indices)}/{n})")
            t0n = time.perf_counter()
            naive_model, _ = sisa_train(
                dataset=train_kept, valset=valset, testset=testset, model_name=model_name,
                K=args.K, S=args.S, batch_size=args.batch_size, epochs_per_slice=args.epochs,
                lr=args.lr, seed=args.seed, log_path=None, checkpoint_dir=ckpt_dir_naive
            )
            t_naive = time.perf_counter() - t0n

            acc_n, f1_n = eval_metrics(naive_model, testset)
            mia_n = mia_auc_for(
                naive_model,
                Subset(train_kept, range(min(5000, len(train_kept)))),
                Subset(testset,    range(min(5000, len(testset))))
            )

            rows.append({
                "method": "Naive-delete",
                "percent_deleted": frac*100,
                "accuracy": acc_n,
                "f1": f1_n,
                "time": t_naive,
                "mia_auc": mia_n,
                "retrained_slices_frac": 1.00
            })

    # Summary CSV
    df = pd.DataFrame(rows, columns=["method","percent_deleted","accuracy","f1","time","mia_auc","retrained_slices_frac"])
    out_csv = os.path.join(args.out_dir, "results_summary.csv")
    ensure_dir(args.out_dir)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")

    # Figures
    fig1 = os.path.join(args.out_dir, "plots", "fig1_acc_vs_deleted.png")
    fig2 = os.path.join(args.out_dir, "plots", "fig2_time_saved.png")
    fig3 = os.path.join(args.out_dir, "plots", "fig3_mia_auc.png")
    ensure_dir(os.path.dirname(fig1))

    subtitle = f"mode={args.deletion_mode}" + (f", few_shards_k={args.few_shards_k}" if args.deletion_mode=='few_shards' and args.few_shards_k else "") + (" + true_naive" if args.run_true_naive else "")
    fig1_acc_vs_deleted(df, fig1, subtitle=subtitle)
    fig2_time_saved_vs_naive(df, fig2, subtitle=subtitle)
    fig3_mia_auc_grouped(df, fig3, subtitle=subtitle)

    # Report (4 pages)
    build_pdf_report(
        df,
        cfg=dict(dataset=args.dataset, K=args.K, S=args.S, epochs=args.epochs,
                 batch_size=args.batch_size, seed=args.seed,
                 deletion_mode=args.deletion_mode, few_shards_k=args.few_shards_k,
                 run_true_naive=args.run_true_naive),
        paths=dict(out_pdf=os.path.join(args.out_dir, "report.pdf"),
                   fig1=fig1, fig2=fig2, fig3=fig3)
    )

    print("\nExperiments complete. Summary:")
    print(df)

if __name__ == "__main__":
    main()
