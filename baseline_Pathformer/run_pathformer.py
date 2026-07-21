"""
Orchestrator: preprocess pan_survival_5yr-style data into Pathformer format,
then train Pathformer on all K stratified CV folds and aggregate the results.
Default is 5-fold CV with a per-fold 72/8/20 train/val/test split.

Usage:
    python run_pathformer.py \
        --cnv_file      data_tcga/pan_survival_5yr/cnv_data.csv \
        --mutation_file data_tcga/pan_survival_5yr/mutation_data.csv \
        --label_file    data_tcga/pan_survival_5yr/patient_labels.csv \
        --pathway_file  data_tcga/pan_survival_5yr/filtered_pathways.csv \
        --adjacency_file data_tcga/pan_survival_5yr/adjacency_matrix.csv \
        --data_dir   data_pan_survival_5yr \
        --save_dir   results_pan_survival_5yr \
        [--skip_preprocessing] [--folds 1 2 3 ...] [--python /path/to/python]

The aligned hyperparameters (200 epochs / batch 16 / lr 1e-4 / AdamW wd 5e-4 /
patience 25 / min_epochs 50 / dropout 0.2) are fixed in Pathformer_train_2mod.py.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnv_file")
    ap.add_argument("--mutation_file")
    ap.add_argument("--label_file")
    ap.add_argument("--pathway_file")
    ap.add_argument("--adjacency_file")
    ap.add_argument("--label_col", default="response")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="Which 1-indexed folds to train (default: 1..n_folds)")
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override default batch_size (16) in Pathformer_train_2mod.py")
    args = ap.parse_args()

    if not args.skip_preprocessing:
        for f in [args.cnv_file, args.mutation_file, args.label_file,
                  args.pathway_file, args.adjacency_file]:
            if f is None or not os.path.exists(f):
                sys.exit(f"missing input: {f}")
        run([
            args.python, "-u", os.path.join(HERE, "pathformer_preprocess.py"),
            "--cnv_file", args.cnv_file,
            "--mutation_file", args.mutation_file,
            "--label_file", args.label_file,
            "--pathway_file", args.pathway_file,
            "--adjacency_file", args.adjacency_file,
            "--label_col", args.label_col,
            "--output_dir", args.data_dir,
            "--seed", str(args.seed),
            "--n_folds", str(args.n_folds),
        ])

    folds = args.folds if args.folds else list(range(1, args.n_folds + 1))
    os.makedirs(args.save_dir, exist_ok=True)
    for k in folds:
        cmd = [
            args.python, "-u", os.path.join(HERE, "Pathformer_train_2mod.py"),
            "--data_dir", args.data_dir,
            "--save_dir", args.save_dir,
            "--fold", str(k),
            "--seed", str(args.seed),
        ]
        if args.save_model:
            cmd.append("--save_model")
        if args.batch_size is not None:
            cmd += ["--batch_size", str(args.batch_size)]
        run(cmd)

    # Aggregate per-fold results.
    fold_results = []
    for k in folds:
        p = os.path.join(args.save_dir, f"fold{k}_result.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            fold_results.append(json.load(f))

    if fold_results:
        metric_names = list(fold_results[0]["test_metrics"].keys())
        summary = {"per_fold": [], "mean": {}, "std": {}}
        for r in fold_results:
            summary["per_fold"].append({"fold": r["fold"], **r["test_metrics"]})
        for m in metric_names:
            vals = [r["test_metrics"][m] for r in fold_results]
            summary["mean"][m] = float(np.mean(vals))
            summary["std"][m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.save_dir, f"{args.n_folds}fold_summary_{ts}.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n=== Summary across folds ===")
        for m in metric_names:
            print(f"  {m}: {summary['mean'][m]:.4f} ± {summary['std'][m]:.4f}")
        print("wrote", out)


if __name__ == "__main__":
    main()
