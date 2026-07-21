#!/usr/bin/env python
"""
Collect a CV run's outputs into results/<dataset>/:
    pnet_cv10_metrics.txt   — mean ± std summary + per-fold table
    pnet_cv10_y_scores.csv  — concatenated per-fold test predictions

Usage:
    python results/_collect_cv_results.py <dataset_short_name> [--model P-net_<ID>]

If --model is omitted, the script picks the most-recent <ID> by mtime in the
CV output dir.
"""
import argparse, glob, os, re, sys
from datetime import datetime
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIR_BINARY = os.path.join(REPO_ROOT, "_logs/p1000/pnet/crossvalidation_average_reg_10_tanh")
CV_DIR_MULTICLASS = os.path.join(REPO_ROOT, "_logs/p1000/pnet/crossvalidation_average_reg_10_tanh_multiclass")


def find_model_name(cv_dir, model_hint=None):
    csvs = sorted(glob.glob(os.path.join(cv_dir, "*_scores.csv")),
                  key=os.path.getmtime, reverse=True)
    if not csvs:
        sys.exit("no *_scores.csv in %s — re-run training so the new "
                 "CV pipeline writes the summary CSV" % cv_dir)
    if model_hint:
        for p in csvs:
            if model_hint in os.path.basename(p):
                return os.path.basename(p).replace("_scores.csv", "")
        sys.exit("no scores.csv matching %s" % model_hint)
    return os.path.basename(csvs[0]).replace("_scores.csv", "")


def grep_elapsed(cv_dir):
    log = os.path.join(cv_dir, "log.log")
    if not os.path.exists(log):
        return "?"
    with open(log) as f:
        lines = [l for l in f if "Elapsed Time" in l]
    return lines[-1].strip().split("Elapsed Time:")[-1].strip() if lines else "?"


def build_metrics(cv_dir, model, out_path, dataset):
    csv_path = os.path.join(cv_dir, f"{model}_scores.csv")
    df = pd.read_csv(csv_path, index_col=0)
    fold_rows = [i for i in df.index if str(i).startswith("fold_")]
    scores = df.loc[fold_rows]
    smean = df.loc["mean"]
    sstd = df.loc["std"]
    metric_order = ["accuracy", "precision", "auc", "f1", "f1_macro", "aupr", "recall"]
    metric_order = [m for m in metric_order if m in smean.index]

    elapsed = grep_elapsed(cv_dir)
    n_folds = len(scores)
    lines = [
        f"P-NET {n_folds}-fold cross-validation — {dataset} (mutation + CNV)",
        f"Run: crossvalidation_average_reg_10_tanh    Elapsed: {elapsed}",
        f"Source log: _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Mean ± Std (across {n_folds} folds)",
        "============================",
    ]
    for m in metric_order:
        lines.append(f"{m:<10s} : {smean[m]:.3f} ± {sstd[m]:.3f}")
    lines += ["", "Per-fold scores", "==============="]
    header = "fold  " + "  ".join(f"{m:>9s}" for m in metric_order)
    lines.append(header)
    for i, (label, row) in enumerate(scores.iterrows()):
        cells = "  ".join(f"{row[m]:9.6f}" for m in metric_order)
        lines.append(f"{i:>3d}   {cells}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


def build_yscores(cv_dir, model, out_path):
    pattern = os.path.join(cv_dir, f"{model}_testing_fold_*.csv")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"fold_(\d+)\.csv$", p).group(1)))
    if not files:
        sys.exit("no per-fold testing CSVs at " + pattern)
    parts = []
    for p in files:
        i = int(re.search(r"fold_(\d+)\.csv$", p).group(1))
        df = pd.read_csv(p)
        df = df.rename(columns={df.columns[0]: "sample_id"})
        df.insert(0, "fold", i)
        score_cols = [c for c in df.columns if c == "pred_score" or c.startswith("pred_score_")]
        keep = ["fold", "sample_id", "y", "pred"] + score_cols
        parts.append(df[keep])
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows from {len(files)} folds, "
          f"{len(score_cols)} score column(s))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder name under results/, e.g. pancancer")
    ap.add_argument("--model", default=None,
                    help="model id substring (default: most recent in CV dir)")
    ap.add_argument("--label", default=None,
                    help="dataset label in metrics.txt header (default: dataset)")
    ap.add_argument("--multiclass", action="store_true",
                    help="read from the multiclass CV output directory")
    ap.add_argument("--cv-dir", default=None,
                    help="explicit CV output directory (overrides --multiclass)")
    args = ap.parse_args()

    if args.cv_dir is not None:
        cv_dir = args.cv_dir
    elif args.multiclass:
        cv_dir = CV_DIR_MULTICLASS
    else:
        cv_dir = CV_DIR_BINARY

    out_dir = os.path.join(REPO_ROOT, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    model = find_model_name(cv_dir, args.model)
    print("using model:", model)
    print("reading from:", cv_dir)
    build_metrics(cv_dir, model, os.path.join(out_dir, "pnet_cv10_metrics.txt"),
                  args.label or args.dataset)
    build_yscores(cv_dir, model, os.path.join(out_dir, "pnet_cv10_y_scores.csv"))


if __name__ == "__main__":
    main()
