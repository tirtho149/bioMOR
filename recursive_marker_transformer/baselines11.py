# ============================================================================
# SMART -- external / classical baselines on the 11-dataset roster.
# Copyright (c) 2026 The SMART Authors. PROPRIETARY AND CONFIDENTIAL. See LICENSE.
# ============================================================================
"""Strong non-transformer baselines on the SAME 11 splits SMART uses, to calibrate the
magnitude of SMART's gains (reviewer request; cf. simple linear pipelines rivalling
foundation models). Methods: a linear ANOVA->PCA->logistic pipeline, Random Forest, and a
NearestCentroid marker-style classifier -- all on each dataset's stratified train/test
split, multi-seed.

    python -m recursive_marker_transformer.baselines11 --seeds 0 1 2
-> results_baselines11/<dataset>/<method>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .bio_learned_genomap import DATASETS as SC_DATASETS, load_genomap
from .bio_redesign_curated import _cap_genes, load_pnet
from .singlecell import _make_splits

ROOT = Path(__file__).resolve().parents[1]
PN_COHORTS = ["prostate", "blca", "stad"]


def _methods(F, K):
    k = int(min(500, max(10, F - 1)))
    npca = int(min(50, k - 1, max(2, K * 3)))
    return {
        "Linear (ANOVA->PCA->LogReg)": Pipeline([
            ("sc", StandardScaler()),
            ("anova", SelectKBest(f_classif, k=k)),
            ("pca", PCA(n_components=npca, random_state=0)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("anova", SelectKBest(f_classif, k=k)),
            ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                          class_weight="balanced_subsample", n_jobs=-1,
                                          random_state=0)),
        ]),
        "NearestCentroid": Pipeline([
            ("sc", StandardScaler()),
            ("anova", SelectKBest(f_classif, k=k)),
            ("nc", NearestCentroid()),
        ]),
    }


def _load(name):
    if name in SC_DATASETS:
        X, y = load_genomap(name)
        return X.astype(np.float32), y.astype(int)
    X, y, genes, classes = load_pnet(name)
    X, _ = _cap_genes(X, genes, 3000)
    return X.astype(np.float32), y.astype(int)


def run_dataset(name, seeds, out):
    X, y = _load(name)
    F, K = X.shape[1], int(y.max() + 1)
    out_dir = out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[base11] {name} N={len(y)} F={F} K={K}", flush=True)
    for seed in seeds:
        tr, va, te = _make_splits(y, None, seed)
        trv = np.concatenate([tr, va])                 # baselines have no val need
        for mname, clf in _methods(F, K).items():
            p = out_dir / f"{mname.split()[0].lower()}_s{seed}.json"
            try:
                clf.fit(X[trv], y[trv])
                yp = clf.predict(X[te])
                r = {"dataset": name, "method": mname, "seed": seed,
                     "test_macro_f1": 100 * f1_score(y[te], yp, average="macro"),
                     "test_accuracy": 100 * accuracy_score(y[te], yp),
                     "n_features": F, "n_classes": K, "n_samples": int(len(y))}
                p.write_text(json.dumps(r, indent=1))
                print(f"  [{mname} s{seed}] F1={r['test_macro_f1']:.2f} acc={r['test_accuracy']:.2f}", flush=True)
            except Exception as e:
                print(f"  [{mname} s{seed}] FAILED: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(SC_DATASETS) + PN_COHORTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=ROOT / "results_baselines11")
    args = ap.parse_args()
    for name in args.datasets:
        try:
            run_dataset(name, args.seeds, args.out)
        except Exception as e:
            print(f"[base11] {name} SKIP: {e}", flush=True)


if __name__ == "__main__":
    main()
