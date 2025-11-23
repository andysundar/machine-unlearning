# SISA Machine Unlearning (PyTorch)

A compact, reproducible implementation of **SISA** (Sharded, Isolated, Sliced, Aggregated) training with:
- selective retraining to honor deletion requests,
- time/accuracy/privacy (MIA) metrics,
- optional **true Naive** full-retrain baseline,
- flexible deletion modes (uniform / last-slice / few-shards),
- 4‑page PDF report with figures & table.

---

## 1) Quick start

### Option A: pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate sisa-unlearning
```

> **macOS SSL note:** If CIFAR‑10 download fails with a certificate error, we already set up `certifi` in `src/datasets.py`. If it still appears, install the certs (e.g., `Install Certificates.command` on macOS) or manually place the dataset in `./data`.

---

## 2) Run the full experiment & report

### Purchase (synthetic, fast)
```bash
python -m src.run_experiments   --dataset purchase_synth   --K 10 --S 2 --epochs 3   --delete_fracs 0.01 0.05   --out_dir results
```

### CIFAR‑10 (match classroom settings closer)
```bash
python -m src.run_experiments   --dataset cifar10 --data_root ./data   --K 10 --S 3 --epochs 10 --batch_size 128   --delete_fracs 0.01 0.05   --deletion_mode last_slice   --run_true_naive   --out_dir results_cifar10
```

Outputs:
- `results*/results_summary.csv`
- `results*/plots/fig1_acc_vs_deleted.png`
- `results*/plots/fig2_time_saved.png` (Time saved vs Naive; uses **true** Naive when available)
- `results*/plots/fig3_mia_auc.png`
- `results*/report.pdf` (4 pages)
  - **Page 1**: Title + full config (now includes `deletion` and `few_shards_k`) + baseline metrics
  - **Page 2**: Fig‑1 (accuracy vs % deleted)
  - **Page 3**: Fig‑2 (time saved vs Naive)
  - **Page 4**: Fig‑3 (MIA AUC) + Table‑1 (metrics summary)

---

## 3) Deletion modes

Choose how deletion indices are drawn (`--deletion_mode`):

- `uniform` (default): random across all shards & slices (often retrains many slices).
- `last_slice`: deletes only from the **latest slice** in each shard (typical “recent user deletion”), minimizing replay.
- `few_shards`: deletes concentrated in **K' shards** (set with `--few_shards_k`), leaving others untouched.

Examples:
```bash
# last-slice deletions
--deletion_mode last_slice

# few-shards deletions (affect 3 shards only)
--deletion_mode few_shards --few_shards_k 3
```

---

## 4) About baselines

- **SISA-baseline**: one clean SISA train (0% deleted); `retrained_slices_frac = 0.0` (no unlearning).
- **SISA-unlearned**: selective replay of only impacted slices; the fraction should be ≪ 1.0 for last-slice or few-shards modes.
- **Naive-delete-sim**: simulated full retrain, uses baseline metrics/time (always present unless disabled).
- **Naive-delete (true)**: full retrain after each deletion %, enabled with `--run_true_naive`.

Tip: When comparing time saved, Fig‑2 prefers **true Naive** if present; otherwise it falls back to baseline time.

---

## 5) Code map

- `src/run_experiments.py` — end-to-end driver (training, unlearning, optional true Naive, plots, PDF).
- `src/sisa.py` — SISA training, selective unlearning, impacted slice detection.
- `src/datasets.py` — CIFAR‑10 with standard transforms + synthetic Purchase dataset.
- `src/models.py` — Small CNN (CIFAR‑10) and MLP (Purchase).
- `src/mia.py` — confidence‑based MIA AUC utility.
- `src/utils.py` — seeds, dirs, and a tiny CSV logger.
- `src/train.py` — baseline-only helper (optional).
- `src/unlearn.py` — one-off selective unlearning demo (optional).
- `src/naive_delete.py` — standalone true Naive retrainer (optional if you use `--run_true_naive`).

---

## 6) Troubleshooting

- **Very low accuracy (~1%)** on CIFAR‑10: double-check dataset path/normalize and that `model_name="cnn"` is used for CIFAR‑10.
- **SISA-unlearned retrained fraction ~1.0** at small % deletes: this happens for uniform deletions. Try `--deletion_mode last_slice` or `--deletion_mode few_shards` (and/or increase `S`).
- **SSL errors on download**: `certifi` is included; on macOS you may still need to run the certificate installer included with Python.

---

## 7) Reproducibility

All major random sources are seeded (`--seed`). Log files live in `results*/logs/`. The summary CSV is the single source of truth for the report and plots.
