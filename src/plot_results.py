
import argparse, os, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="results/results_summary.csv")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    for m in df["method"].unique():
        sub = df[df["method"]==m].sort_values("percent_deleted")
        if sub.empty: continue
        plt.plot(sub["percent_deleted"], sub["accuracy"], marker="o", label=m)
    plt.xlabel("% deleted"); plt.ylabel("Test accuracy"); plt.title("Fig-1")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fig1.png")); plt.close()
if __name__ == "__main__":
    main()
