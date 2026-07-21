"""Convert P-NET's native per-fold testing CSVs into the shared metrics JSON.

P-NET's training harness writes results in its own format (scores.csv, per-fold
`*_testing_fold_K.csv` with columns: index, pred, pred_score_0..C-1, y) rather
than calling multiclass_metrics.write_metrics_json like the other baselines.
This reads those per-fold CSVs and emits baseline_pnet/results/brca_pam50/
pnet_metrics.json so pnet appears in the consolidated table.

    python baseline_pnet/pnet_collect_metrics.py \
        --run_dir baseline_pnet/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh_multiclass_brca3mod
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from multiclass_metrics import fold_metrics, write_metrics_json  # noqa: E402

N_CLASSES = 5
DEFAULT_OUT = os.path.join(HERE, "results", "brca_pam50", "pnet_metrics.json")


def fold_index(path):
    m = re.search(r"testing_fold_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="P-NET run dir containing *_testing_fold_K.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.run_dir, "*_testing_fold_*.csv")),
                  key=fold_index)
    if not csvs:
        sys.exit(f"No *_testing_fold_*.csv found in {args.run_dir}")

    per_fold = []
    for c in csvs:
        df = pd.read_csv(c, index_col=0)
        prob_cols = sorted(
            [col for col in df.columns if col.startswith("pred_score_")],
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        y_true = df["y"].to_numpy()
        y_prob = df[prob_cols].to_numpy()
        per_fold.append(fold_metrics(y_true, y_prob, N_CLASSES))
        print(f"fold {fold_index(c)}: acc={per_fold[-1]['accuracy']:.4f} "
              f"auc={per_fold[-1]['auc']:.4f} f1_macro={per_fold[-1]['f1_macro']:.4f}")

    write_metrics_json(args.out, "pnet", "brca_pam50", N_CLASSES, per_fold)
    print(f"\nWrote {len(per_fold)} folds -> {args.out}")


if __name__ == "__main__":
    main()
