"""Orchestrator for the 3-modality BRCA PAM50 Pathformer baseline.

Preprocesses the shared (cnv, mutation, expression) data into Pathformer format,
trains Pathformer on all 5 shared CV folds, then aggregates per-fold per-class
probabilities into the shared multiclass metrics JSON.

Run from the repo root in the `base` conda env:

    python baseline_Pathformer/run_pathformer_3mod.py \
        --data_dir baseline_Pathformer/data_brca_pam50 \
        --save_dir baseline_Pathformer/results/brca_pam50

Hyperparameters match the original (200 epochs / batch 16 / lr 1e-4 /
AdamW wd 5e-4 / patience 25 / min_epochs 50 / dropout 0.2).
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from multiclass_metrics import fold_metrics, write_metrics_json  # noqa: E402

N_CLASSES = 5
N_FOLDS = 5
METRICS_JSON = os.path.join(HERE, "results", "brca_pam50", "Pathformer_metrics.json")


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data_brca_pam50"))
    ap.add_argument("--save_dir", default=os.path.join(HERE, "results", "brca_pam50"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="Which 1-indexed folds to train (default: 1..5)")
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--skip_training", action="store_true",
                    help="Only aggregate existing per-fold result JSONs.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    if not args.skip_preprocessing:
        run([
            args.python, "-u", os.path.join(HERE, "pathformer_preprocess_3mod.py"),
            "--output_dir", args.data_dir,
        ])

    folds = args.folds if args.folds else list(range(1, N_FOLDS + 1))
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.skip_training:
        for k in folds:
            cmd = [
                args.python, "-u", os.path.join(HERE, "Pathformer_train_3mod.py"),
                "--data_dir", args.data_dir,
                "--save_dir", args.save_dir,
                "--fold", str(k),
                "--seed", str(args.seed),
            ]
            if args.save_model:
                cmd.append("--save_model")
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.batch_size is not None:
                cmd += ["--batch_size", str(args.batch_size)]
            run(cmd)

    # Aggregate per-fold per-class probabilities into shared metrics JSON.
    per_fold = []
    for k in folds:
        p = os.path.join(args.save_dir, f"fold{k}_result.json")
        if not os.path.exists(p):
            print(f"WARNING: missing {p}; skipping fold {k}")
            continue
        with open(p) as f:
            r = json.load(f)
        y_true = r["y_true"]
        y_prob = r["y_prob"]
        per_fold.append(fold_metrics(y_true, y_prob, N_CLASSES))

    if per_fold:
        write_metrics_json(METRICS_JSON, "Pathformer", "brca_pam50", N_CLASSES, per_fold)
        print(f"\nWrote shared metrics JSON ({len(per_fold)} folds) -> {METRICS_JSON}")
    else:
        print("No per-fold results found; metrics JSON not written.")


if __name__ == "__main__":
    main()
