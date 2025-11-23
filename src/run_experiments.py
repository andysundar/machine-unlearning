# run_experiments.py (ready-to-drop)
import argparse, os, time, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from .datasets import get_cifar10, get_purchase_synth, split_train_val
from .sisa import sisa_train, sisa_unlearn
from .utils import ensure_dir, set_seed
from .mia import confidence_scores, mia_auc

torch.set_default_dtype(torch.float32)

# --------------------- metrics helpers ---------------------
def _pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def eval_metrics(model, testset, batch_size=512, device=None):
    device = device or _pick_device()
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y if torch.is_tensor(y) else torch.tensor(y)
            p = model(x).argmax(dim=1).cpu()
            ys.append(y.cpu()); preds.append(p)
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(preds).numpy()
    acc = float((y_true==y_pred).mean())
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = float("nan")
    return acc, f1

def mia_auc_for(model, trainset_subset, testset_subset, device=None):
    device = device or _pick_device()
    in_loader  = DataLoader(trainset_subset, batch_size=512, shuffle=False)
    out_loader = DataLoader(testset_subset,  batch_size=512, shuffle=False)
    in_scores  = confidence_scores(model, in_loader, device)
    out_scores = confidence_scores(model, out_loader, device)
    return float(mia_auc(in_scores, out_scores))

# --------------------- core figures ---------------------
def fig1_acc_vs_deleted(df, out_png):
    plt.figure()
    xs = sorted({float(x) for x in df["percent_deleted"].dropna().unique() if float(x) > 0.0})

    base = df[df["method"]=="SISA-baseline"]
    if not base.empty and xs:
        base_acc = float(base.iloc[0]["accuracy"])
        plt.plot(xs, [base_acc]*len(xs), linestyle="--", marker="s", markersize=7, linewidth=2, label="SISA-baseline")
        plt.scatter([0.0], [base_acc], marker="s", s=60, label="Baseline (0%)", zorder=4)

    naive_true = df[df["method"]=="Naive-delete"].sort_values("percent_deleted")
    naive_sim  = df[df["method"]=="Naive-delete-sim"].sort_values("percent_deleted")
    if not naive_true.empty:
        plt.plot(naive_true["percent_deleted"], naive_true["accuracy"], marker="^", markersize=7, linewidth=2, label="Naive-delete")
    elif not naive_sim.empty:
        plt.plot(naive_sim["percent_deleted"], naive_sim["accuracy"], marker="^", markersize=7, linewidth=2, label="Naive-delete-sim")

    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if not sisa.empty:
        plt.plot(sisa["percent_deleted"], sisa["accuracy"], marker="o", markersize=7, linewidth=2, label="SISA-unlearned")

    acc_vals = df["accuracy"].dropna().astype(float).tolist()
    if acc_vals:
        amin, amax = float(min(acc_vals)), float(max(acc_vals))
        pad = 0.05 * max(1e-6, (amax - amin) if amax > amin else (amax + amin + 1e-6))
        plt.ylim(amin - pad, amax + pad)

    plt.xlabel("% deleted"); plt.ylabel("Test accuracy")
    plt.title("Fig-1: Test accuracy vs % deleted")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def fig2_time_saved_vs_naive(df, out_png):
    plt.figure()
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    xs = sisa["percent_deleted"].astype(float).tolist()
    if not xs:
        plt.xlabel("% deleted"); plt.ylabel("Time saved (s)")
        plt.title("Fig-2: Time saved vs Naive-delete")
        plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    naive_df = df[df["method"]=="Naive-delete"].set_index("percent_deleted")
    base = df[df["method"]=="SISA-baseline"]
    baseline_time = float(base.iloc[0]["time"]) if not base.empty else max(sisa["time"].astype(float).tolist())
    if not naive_df.empty:
        full_times = [float(naive_df.loc[x,"time"]) if x in naive_df.index else baseline_time for x in xs]
    else:
        full_times = [baseline_time for _ in xs]
    sisa_times = sisa["time"].astype(float).tolist()
    time_saved = [max(ft - st, 0.0) for ft, st in zip(full_times, sisa_times)]
    plt.plot(xs, time_saved, marker="o", markersize=7, linewidth=2, label="SISA-unlearned")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.5, label="Naive-delete")
    ymin, ymax = min(time_saved + [0.0]), max(time_saved + [0.0])
    pad = 0.05 * (ymax - ymin + 1e-6)
    plt.ylim(ymin - pad, ymax + pad)
    plt.xlabel("% deleted"); plt.ylabel("Time saved (s)")
    plt.title("Fig-2: Time saved vs Naive-delete")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def fig3_mia_auc_grouped(df, out_png):
    xs = sorted({float(x) for x in df["percent_deleted"].dropna().unique() if float(x) > 0.0})
    methods = ["SISA-baseline", "SISA-unlearned"]
    if (df["method"]=="Naive-delete").any(): methods.append("Naive-delete")
    elif (df["method"]=="Naive-delete-sim").any(): methods.append("Naive-delete-sim")
    width = 0.8/max(len(methods),1); x0 = np.arange(len(xs))
    plt.figure()
    for i,m in enumerate(methods):
        sub = (df[df["method"]==m].set_index("percent_deleted")["mia_auc"]).reindex(xs)
        ys = sub.astype(float).fillna(0.0).values; xpos = x0 + i*width
        plt.bar(xpos, ys, width=width, label=m)
    plt.xticks(x0 + width*(len(methods)-1)/2, [str(x) for x in xs])
    plt.xlabel("% deleted"); plt.ylabel("MIA AUC")
    plt.title("Fig-3: MIA AUC comparison")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

# --------------------- extra figures (7) ---------------------
def plot_acc_vs_delete(df, out_png):  # 1
    return fig1_acc_vs_deleted(df, out_png)

def plot_mia(df, out_png):            # 2
    return fig3_mia_auc_grouped(df, out_png)

def plot_time_comparison(df, out_png):  # 3
    plt.figure()
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if sisa.empty:
        plt.xlabel("Deletion (%)"); plt.ylabel("Time (s)"); plt.title("SISA vs Naive Retraining Time Comparison")
        plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    xs = sisa["percent_deleted"].astype(float).tolist()
    plt.plot(xs, sisa["time"].astype(float).tolist(), marker="o", linewidth=2, label="SISA-unlearned")
    naive_true = df[df["method"]=="Naive-delete"].sort_values("percent_deleted")
    naive_sim  = df[df["method"]=="Naive-delete-sim"].sort_values("percent_deleted")
    if not naive_true.empty:
        plt.plot(naive_true["percent_deleted"].astype(float).tolist(), naive_true["time"].astype(float).tolist(), marker="^", linewidth=2, label="Naive-delete")
    elif not naive_sim.empty:
        plt.plot(naive_sim["percent_deleted"].astype(float).tolist(), naive_sim["time"].astype(float).tolist(), marker="^", linewidth=2, label="Naive-delete-sim")
    plt.xlabel("Deletion (%)"); plt.ylabel("Time (s)"); plt.title("SISA vs Naive Retraining Time Comparison")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_slices_retrained(df, out_png, K=None, S=None):  # 4
    plt.figure()
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if sisa.empty:
        plt.xlabel("Deletion (%)"); plt.ylabel("Slices retrained"); plt.title("Slices Retrained per Scenario")
        plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    xs = sisa["percent_deleted"].astype(float).tolist()
    fracs = sisa["retrained_slices_frac"].astype(float).tolist()
    if K and S:
        ys = [f * (K*S) for f in fracs]; ylabel = "Slices retrained (#)"
    else:
        ys = fracs; ylabel = "Slices retrained (fraction)"
    plt.bar([str(x) for x in xs], ys)
    plt.xlabel("Deletion (%)"); plt.ylabel(ylabel); plt.title("Slices Retrained per Scenario")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_efficiency(df, out_png):  # 5
    plt.figure()
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if sisa.empty:
        plt.xlabel("Deletion (%)"); plt.title("SISA Efficiency Metrics")
        plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    xs = sisa["percent_deleted"].astype(float).tolist()
    naive_df = df[df["method"]=="Naive-delete"].set_index("percent_deleted")
    base = df[df["method"]=="SISA-baseline"]
    baseline_time = float(base.iloc[0]["time"]) if not base.empty else max(sisa["time"].astype(float).tolist())
    if not naive_df.empty:
        full_times = [float(naive_df.loc[x, "time"]) if x in naive_df.index else baseline_time for x in xs]
    else:
        full_times = [baseline_time for _ in xs]
    sisa_times = sisa["time"].astype(float).tolist()
    time_saved = [max(ft - st, 0.0) for ft, st in zip(full_times, sisa_times)]
    frac = sisa["retrained_slices_frac"].astype(float).tolist()
    ax1 = plt.gca()
    ax1.plot(xs, time_saved, marker="o", linewidth=2, label="Time saved (s)")
    ax1.set_xlabel("Deletion (%)"); ax1.set_ylabel("Time saved (s)")
    ax2 = ax1.twinx()
    ax2.plot(xs, frac, marker="^", linewidth=2, label="Slices retrained (frac)")
    ax2.set_ylabel("Slices retrained (fraction)")
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    plt.legend(lines, labels, loc="best")
    plt.title("SISA Efficiency Metrics")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_time_distribution_across_shards(logs_dir, out_png):  # 6
    shard_time = {}
    for fp in glob.glob(os.path.join(logs_dir, "*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        required = {"phase","k","s","epoch_s","time_s"}
        if not required.issubset(df.columns):
            continue
        df = df[df["phase"] == "train"]
        df["k_num"] = pd.to_numeric(df["k"], errors="coerce")
        df["time_s_num"] = pd.to_numeric(df["time_s"], errors="coerce")
        df = df.dropna(subset=["k_num", "time_s_num"])
        if df.empty: continue
        g = df.groupby("k_num")["time_s_num"].sum()
        for k, t in g.items():
            k_int = int(k)
            shard_time[k_int] = shard_time.get(k_int, 0.0) + float(t)
    if not shard_time:
        plt.figure(); plt.text(0.5, 0.5, "No shard-wise time found in logs", ha="center")
        plt.axis("off"); plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    ks = sorted(shard_time.keys()); ts = [shard_time[k] for k in ks]
    plt.figure()
    plt.bar([str(k) for k in ks], ts)
    plt.xlabel("Shard k"); plt.ylabel("Total training time (s)")
    plt.title("Training Time Distribution Across Shards")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_model_acc_after_unlearning(df, out_png):  # 7
    plt.figure()
    sisa = df[df["method"]=="SISA-unlearned"].sort_values("percent_deleted")
    if not sisa.empty:
        plt.plot(sisa["percent_deleted"], sisa["accuracy"], marker="o", linewidth=2, label="SISA-unlearned")
    naive_true = df[df["method"]=="Naive-delete"].sort_values("percent_deleted")
    naive_sim  = df[df["method"]=="Naive-delete-sim"].sort_values("percent_deleted")
    if not naive_true.empty:
        plt.plot(naive_true["percent_deleted"], naive_true["accuracy"], marker="^", linewidth=2, label="Naive-delete")
    elif not naive_sim.empty:
        plt.plot(naive_sim["percent_deleted"], naive_sim["accuracy"], marker="^", linewidth=2, label="Naive-delete-sim")
    base = df[df["method"]=="SISA-baseline"]
    if not base.empty:
        base_acc = float(base.iloc[0]["accuracy"])
        xs = sorted({float(x) for x in df["percent_deleted"].dropna().unique() if float(x) > 0.0})
        if xs:
            plt.plot(xs, [base_acc]*len(xs), linestyle="--", marker="s", markersize=6, label="Baseline")
            plt.scatter([0.0], [base_acc], marker="s", s=60, label="Baseline (0%)", zorder=4)
    plt.xlabel("Deletion (%)"); plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy After Unlearning")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

# --------------------- report ---------------------
def build_pdf_report(df, cfg, paths):
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]; h2 = styles["Heading2"]; body = styles["BodyText"]
    doc = SimpleDocTemplate(paths["out_pdf"], pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    flow = []

    # Page 1: Title + config + baseline metrics
    flow.append(Paragraph("Machine Unlearning with SISA", h1))
    cfg_line = (f"Dataset={cfg['dataset']} | K={cfg['K']} S={cfg['S']} | epochs/slice={cfg['epochs']} | "
                f"batch={cfg['batch_size']} | seed={cfg['seed']} | deletion={cfg.get('deletion_mode','uniform')}"
                f"{(' | few_shards_k='+str(cfg.get('few_shards_k')) if cfg.get('few_shards_k') is not None else '')}"
                f"{(' | true_naive' if cfg.get('run_true_naive') else '')}")
    flow.append(Paragraph(cfg_line, body))
    base = df[df["method"]=="SISA-baseline"].head(1)
    if not base.empty:
        b = base.iloc[0]
        flow.append(Paragraph(f"Baseline: acc={float(b['accuracy']):.4f}, F1={float(b['f1']):.4f}, time={float(b['time']):.1f}s, MIA AUC={float(b['mia_auc']):.6f}", body))
    flow.append(PageBreak())

    # Pages 2-4: original 3 figs
    if os.path.exists(paths["fig1"]):
        flow.append(Paragraph("Fig-1: Test accuracy vs % deleted", h2)); flow.append(Image(paths["fig1"], width=14*cm, height=10*cm)); flow.append(PageBreak())
    if os.path.exists(paths["fig2"]):
        flow.append(Paragraph("Fig-2: Time saved vs Naive-delete", h2)); flow.append(Image(paths["fig2"], width=14*cm, height=10*cm)); flow.append(PageBreak())
    if os.path.exists(paths["fig3"]):
        flow.append(Paragraph("Fig-3: MIA AUC comparison", h2)); flow.append(Image(paths["fig3"], width=14*cm, height=9*cm)); flow.append(PageBreak())

    # Extra figures if present
    extra_map = [
        ("fig_g1", "Test Accuracy vs Deletion Percentage"),
        ("fig_g2", "Membership Inference Attack Results"),
        ("fig_g3", "SISA vs Naive Retraining Time Comparison"),
        ("fig_g4", "Slices Retrained per Scenario"),
        ("fig_g5", "SISA Efficiency Metrics"),
        ("fig_g6", "Training Time Distribution Across Shards"),
        ("fig_g7", "Model Accuracy After Unlearning"),
    ]
    for key, title in extra_map:
        pth = paths.get(key)
        if pth and os.path.exists(pth):
            flow.append(Paragraph(title, h2))
            flow.append(Image(pth, width=14*cm, height=9.5*cm))
            flow.append(PageBreak())

    # Final page: Table-1
    show = df.copy().sort_values(["method","percent_deleted"])
    fmt_map = {
        "percent_deleted": ":.1f",
        "accuracy":        ":.4f",
        "f1":              ":.4f",
        "time":            ":.1f",
        "mia_auc":         ":.6f",
        "retrained_slices_frac": ":.2f",
    }
    for c, fmt in fmt_map.items():
        if c in show.columns:
            s = pd.to_numeric(show[c], errors="coerce")
            spec = fmt[1:] if fmt.startswith(":") else fmt
            show[c] = s.apply(lambda v: "NaN" if pd.isna(v) else format(float(v), spec))

    cols = ["method","percent_deleted","accuracy","f1","time","mia_auc","retrained_slices_frac"]
    header = ["method","percent\ndeleted","accuracy","f1","time (s)","MIA AUC","retrained slices frac"]
    data = [header] + [[show[c].iloc[i] for c in cols] for i in range(len(show))]
    table = Table(data, colWidths=[4*cm,2.3*cm,2*cm,2*cm,2.7*cm,3*cm,3.8*cm], repeatRows=1)
    table.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.75,colors.black),("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    flow.append(KeepTogether([Spacer(1,8), table]))
    doc.build(flow)

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","purchase_synth"], default="purchase_synth")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--delete_fracs", type=float, nargs="+", default=[0.01,0.05])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--run_true_naive", action="store_true")
    ap.add_argument("--include_naive_sim", action="store_true", default=True)
    ap.add_argument("--deletion_mode", choices=["uniform","last_slice","few_shards"], default="uniform")
    ap.add_argument("--few_shards_k", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir); ensure_dir(os.path.join(args.out_dir,"logs")); ensure_dir(os.path.join(args.out_dir,"plots"))
    logs_dir = os.path.join(args.out_dir,"logs"); ckpt_dir = os.path.join(args.out_dir,"ckpts")

    # Load dataset
    if args.dataset == "cifar10":
        train_full, _, _ = get_cifar10(args.data_root, train=True, download=True)
        testset,  _, _  = get_cifar10(args.data_root, train=False, download=True)
        model_name = "cnn"
    else:
        train_full, _, _ = get_purchase_synth(args.data_root, train=True)
        testset,  _, _   = get_purchase_synth(args.data_root, train=False)
        model_name = "mlp"

    trainset, valset = split_train_val(train_full, val_fraction=0.1, seed=args.seed)

    # Baseline
    print("running baseline SISA train...")
    t0 = time.perf_counter()
    model, layout = sisa_train(dataset=trainset, valset=valset, testset=testset, model_name=model_name,
                               K=args.K, S=args.S, batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
                               seed=args.seed, log_path=os.path.join(logs_dir,"baseline.csv"), checkpoint_dir=ckpt_dir)
    t_baseline = time.perf_counter() - t0
    acc_b, f1_b = eval_metrics(model, testset)
    mia_b = mia_auc_for(model, Subset(trainset, range(min(5000,len(trainset)))),
                               Subset(testset, range(min(5000,len(testset)))))

    rows = [dict(method="SISA-baseline", percent_deleted=0.0, accuracy=acc_b, f1=f1_b, time=t_baseline, mia_auc=mia_b, retrained_slices_frac=0.0)]

    # Deletions
    n = len(trainset); rng = np.random.default_rng(args.seed+2025)
    for frac in args.delete_fracs:
        delete_cnt = max(1, int(round(n*frac)))
        delete_idx = rng.choice(n, size=delete_cnt, replace=False).tolist()
        print(f"Running deletion scenario: {frac*100:.1f}%")

        t0 = time.perf_counter()
        model_u, retrained = sisa_unlearn(dataset=trainset, valset=valset, testset=testset, model_name=model_name,
                                          K=args.K, S=args.S, layout=layout, delete_indices=delete_idx,
                                          batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
                                          seed=args.seed, log_path=os.path.join(logs_dir,f"unlearn_{int(frac*100)}pct.csv"), checkpoint_dir=ckpt_dir)
        t_un = time.perf_counter() - t0
        acc_u, f1_u = eval_metrics(model_u, testset)
        mia_u = mia_auc_for(model_u, Subset(trainset, range(min(5000,len(trainset)))),
                                     Subset(testset, range(min(5000,len(testset)))))
        rows.append(dict(method="SISA-unlearned", percent_deleted=frac*100, accuracy=acc_u, f1=f1_u, time=t_un,
                         mia_auc=mia_u, retrained_slices_frac=retrained/(args.K*args.S)))

        if args.include_naive_sim:
            rows.append(dict(method="Naive-delete-sim", percent_deleted=frac*100, accuracy=acc_b, f1=f1_b, time=t_baseline, mia_auc=mia_b, retrained_slices_frac=1.0))

        if args.run_true_naive:
            base_indices = np.arange(n); keep_mask = np.ones(n, dtype=bool); keep_mask[delete_idx]=False
            keep_indices = base_indices[keep_mask].tolist()
            train_kept = Subset(trainset, keep_indices)
            ckpt_dir_naive = os.path.join(ckpt_dir, f"naive_{int(frac*100)}pct"); ensure_dir(ckpt_dir_naive)
            print(f"[Naive] Full retrain after deletion {frac*100:.1f}% (kept {len(keep_indices)}/{n})")
            t0n = time.perf_counter()
            naive_model, _ = sisa_train(dataset=train_kept, valset=valset, testset=testset, model_name=model_name,
                                        K=args.K, S=args.S, batch_size=args.batch_size, epochs_per_slice=args.epochs, lr=args.lr,
                                        seed=args.seed, log_path=None, checkpoint_dir=ckpt_dir_naive)
            t_naive = time.perf_counter() - t0n
            acc_n, f1_n = eval_metrics(naive_model, testset)
            mia_n = mia_auc_for(naive_model, Subset(train_kept, range(min(5000,len(train_kept)))),
                                           Subset(testset,    range(min(5000,len(testset)))))
            rows.append(dict(method="Naive-delete", percent_deleted=frac*100, accuracy=acc_n, f1=f1_n, time=t_naive, mia_auc=mia_n, retrained_slices_frac=1.0))

    # Summary DF
    df = pd.DataFrame(rows, columns=["method","percent_deleted","accuracy","f1","time","mia_auc","retrained_slices_frac"])
    for _c in ("percent_deleted","accuracy","f1","time","mia_auc","retrained_slices_frac"):
        if _c in df.columns: df[_c] = pd.to_numeric(df[_c], errors="coerce")

    out_csv = os.path.join(args.out_dir,"results_summary.csv"); ensure_dir(args.out_dir); df.to_csv(out_csv, index=False)

    # Core figs
    fig1 = os.path.join(args.out_dir,"plots","fig1_acc_vs_deleted.png")
    fig2 = os.path.join(args.out_dir,"plots","fig2_time_saved.png")
    fig3 = os.path.join(args.out_dir,"plots","fig3_mia_auc.png")
    ensure_dir(os.path.dirname(fig1))
    fig1_acc_vs_deleted(df, fig1); fig2_time_saved_vs_naive(df, fig2); fig3_mia_auc_grouped(df, fig3)

    # Extra figs
    extra_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(extra_dir)
    print(f"[extra-plots] Writing to: {extra_dir}")

    def _run_plot(fn, *a, **kw):
        out_arg = kw.get("out_png", None)
        if out_arg is None:
            str_args = [x for x in a if isinstance(x, str)]
            out_arg = str_args[-1] if str_args else f"{fn.__name__}.png"
        name = os.path.basename(out_arg)
        try:
            fn(*a, **kw)
            print(f"[extra-plots] OK  -> {name}")
        except Exception as e:
            print(f"[extra-plots] FAIL -> {name}: {e}")

    _run_plot(plot_acc_vs_delete, df, out_png=os.path.join(extra_dir, "g1_acc_vs_delete.png"))
    _run_plot(plot_mia, df, out_png=os.path.join(extra_dir, "g2_mia_auc.png"))
    _run_plot(plot_time_comparison, df, out_png=os.path.join(extra_dir, "g3_time_sisa_vs_naive.png"))
    _run_plot(plot_slices_retrained, df, K=args.K, S=args.S, out_png=os.path.join(extra_dir, "g4_slices_retrained.png"))
    _run_plot(plot_efficiency, df, out_png=os.path.join(extra_dir, "g5_sisa_efficiency.png"))
    _run_plot(plot_time_distribution_across_shards, os.path.join(args.out_dir, "logs"), out_png=os.path.join(extra_dir, "g6_time_by_shard.png"))
    _run_plot(plot_model_acc_after_unlearning, df, out_png=os.path.join(extra_dir, "g7_acc_after_unlearning.png"))

    # PDF report (includes extra figs if present)
    build_pdf_report(
        df,
        cfg=dict(dataset=args.dataset,K=args.K,S=args.S,epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,
                 deletion_mode=args.deletion_mode,few_shards_k=args.few_shards_k,run_true_naive=args.run_true_naive),
        paths=dict(
            out_pdf=os.path.join(args.out_dir,"report.pdf"),
            fig1=fig1, fig2=fig2, fig3=fig3,
            fig_g1=os.path.join(extra_dir, "g1_acc_vs_delete.png"),
            fig_g2=os.path.join(extra_dir, "g2_mia_auc.png"),
            fig_g3=os.path.join(extra_dir, "g3_time_sisa_vs_naive.png"),
            fig_g4=os.path.join(extra_dir, "g4_slices_retrained.png"),
            fig_g5=os.path.join(extra_dir, "g5_sisa_efficiency.png"),
            fig_g6=os.path.join(extra_dir, "g6_time_by_shard.png"),
            fig_g7=os.path.join(extra_dir, "g7_acc_after_unlearning.png"),
        )
    )

    print("Saved summary to", out_csv)
    print("Experiments complete. Summary:")
    print(df)

if __name__ == "__main__":
    main()
