# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#
# Shared cross-validation split — ONE definition of the protocol so every
# training entry point (single-cell, bioMoR, multi-omics pathway, bioMoR-pnet)
# uses byte-identical folds. This is what makes cross-variant comparisons paired.
# ============================================================================
"""Unified 5-fold CV split.

Protocol (fixed for the whole paper table):
  * StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
      -> each fold holds out 1/n_folds of the data as TEST (20% at n_folds=5),
         the remaining 80% is TRAIN.
  * Within each fold's train, hold out `val_frac` of TRAIN as VALIDATION
      (stratified, same seed) -> per fold: 72% train / 8% val / 20% test at
      n_folds=5, val_frac=0.10.

Because the split depends only on (y, n_folds, seed, val_frac) and every dataset
is loaded deterministically, the folds are identical across every model variant.
Falls back to non-stratified KFold only when a class is too rare to stratify.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

SEED = 42
N_FOLDS = 5
VAL_FRAC = 0.10


def cv_folds(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED,
             val_frac: float = VAL_FRAC) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return [(train_idx, val_idx, test_idx), ...] for each of `n_folds` folds."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    stratify_outer = counts.min() >= n_folds
    if stratify_outer:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer = splitter.split(np.zeros(len(y)))

    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_all, te in outer:
        _, cnt = np.unique(y[tr_all], return_counts=True)
        strat = y[tr_all] if cnt.min() >= 2 else None
        tr, va = train_test_split(tr_all, test_size=val_frac,
                                  random_state=seed, stratify=strat)
        folds.append((np.asarray(tr), np.asarray(va), np.asarray(te)))
    return folds


def summarize(values) -> dict:
    """mean/std (population, ddof=0) over the per-fold metric list."""
    v = np.asarray(list(values), dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
            "folds": [float(x) for x in v]}
