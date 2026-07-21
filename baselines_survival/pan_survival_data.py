"""Shared data contract for the 2-modality pan-cancer survival baseline runs.

Guarantees every survival baseline trains on the same patients, the same
(time, event) labels, and the same 5-fold stratified splits (seed 42,
val = 10% of trainval -> 72/8/20 overall), stratified on the event indicator.

Data: data_tcga/pan_survival_cox (CNV + mutation only; no expression).
Survival labels: survival.csv with columns id, OS, OS.time (endpoint OS).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Set SURV_3MODAL=1 to add gene expression as a third modality (pan_survival_3mod,
# 8454 patients x 17940 common genes). Default (unset) keeps the original
# 2-modality CNV+mutation contract (pan_survival_cox) so prior results reproduce.
_3MOD = os.environ.get("SURV_3MODAL", "0") == "1"

if _3MOD:
    DATA_DIR = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/pan_survival_3mod"
    MODALITIES = ["cnv_data.csv", "mutation_data.csv", "expression_data.csv"]
    MODALITY_NAMES = ["cnv", "mutation", "expression"]
else:
    DATA_DIR = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/pan_survival_cox"
    MODALITIES = ["cnv_data.csv", "mutation_data.csv"]
    MODALITY_NAMES = ["cnv", "mutation"]

SURVIVAL_FILE = "survival.csv"
ENDPOINT = "OS"
SEED = 42
N_FOLDS = 5
VAL_FRACTION = 0.10  # of the trainval remainder -> 72/8/20 overall

# ---------------------------------------------------------------------------
# Shared training hyperparameters -- imported by EVERY baseline so the survival
# comparison is apples-to-apples: identical folds (make_folds), learning rate,
# weight decay, optimizer (AdamW), minibatch size, epoch budget, and early-
# stopping rule (patience on validation C-index, checked once per epoch, with a
# minimum number of epochs before stopping is allowed).
#
# NOTE: MOGONET (transductive full-graph GCN) and MOGAT (two-stage GAT->head)
# keep their native training loops per design; they still consume THESE lr /
# weight-decay / early-stopping constants so every tunable knob is shared.
# ---------------------------------------------------------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
LR = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 4 if SMOKE else 200
MIN_EPOCHS = 1 if SMOKE else 50
PATIENCE = 25  # early stop after this many epochs with no val-C-index improvement


def load_modalities(data_dir=DATA_DIR, endpoint=ENDPOINT):
    """Return (mods, time, event, patients, genes).

    mods:  dict name -> (N, G) float32 ndarray, rows aligned to `patients`.
    time:  (N,) float32 survival durations (> 0).
    event: (N,) int event indicator (1 = event observed, 0 = censored).
    patients: list[str].   genes: list[str] (shared across modalities).
    """
    frames = {}
    for fname, name in zip(MODALITIES, MODALITY_NAMES):
        df = pd.read_csv(f"{data_dir}/{fname}", index_col=0)
        df.index = df.index.astype(str)
        frames[name] = df

    # Common gene columns across ALL modalities (stable original order from cnv).
    common_genes = set.intersection(*[set(frames[n].columns) for n in MODALITY_NAMES])
    genes = [g for g in frames["cnv"].columns if g in common_genes]
    if not genes:
        raise ValueError("No overlapping genes across modalities.")
    for name in MODALITY_NAMES:
        frames[name] = frames[name][genes]

    # Survival labels for the endpoint.
    surv = pd.read_csv(f"{data_dir}/{SURVIVAL_FILE}")
    id_col = surv.columns[0]
    surv[id_col] = surv[id_col].astype(str)
    surv = surv.set_index(id_col)
    time_col, event_col = f"{endpoint}.time", endpoint
    sub = surv[[event_col, time_col]].dropna()
    sub = sub[sub[time_col] > 0]
    sub[event_col] = sub[event_col].astype(int)
    sub[time_col] = sub[time_col].astype(float)

    common = set(sub.index)
    for df in frames.values():
        common &= set(df.index)
    patients = sorted(common)

    mods = {name: frames[name].loc[patients].to_numpy(dtype=np.float32)
            for name in frames}

    # Impute missing values (gene expression has NaNs; CNV/mutation do not) with
    # the per-gene cohort mean, so no NaN reaches StandardScaler.fit -- an
    # unhandled NaN there poisons the whole graph/attention models to all-NaN
    # risk. A fully-NaN gene column falls back to 0.
    for name, arr in mods.items():
        if np.isnan(arr).any():
            col_mean = np.nanmean(arr, axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            idx = np.where(np.isnan(arr))
            arr[idx] = np.take(col_mean, idx[1])
            mods[name] = arr

    time = sub.loc[patients, time_col].to_numpy(dtype=np.float32)
    event = sub.loc[patients, event_col].to_numpy(dtype=int)
    return mods, time, event, patients, genes


def make_folds(event, n_folds=N_FOLDS, seed=SEED, val_fraction=VAL_FRACTION):
    """Return list of (train_idx, val_idx, test_idx) ndarrays for each fold.

    StratifiedKFold on the event indicator so each fold has a comparable
    event rate; val carved from the train portion, also stratified on event.
    """
    event = np.asarray(event)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_val_idx, test_idx in skf.split(np.arange(len(event)), event):
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_fraction,
            stratify=event[train_val_idx], random_state=seed,
        )
        folds.append((np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)))
    return folds


if __name__ == "__main__":
    mods, time, event, patients, genes = load_modalities()
    print(f"patients={len(patients)} genes={len(genes)} "
          f"events={int(event.sum())} ({event.mean()*100:.1f}%) "
          f"median_time={np.median(time):.0f}d")
    for name, arr in mods.items():
        print(f"  {name}: {arr.shape}")
    folds = make_folds(event)
    for i, (tr, va, te) in enumerate(folds, 1):
        print(f"  fold {i}: train={len(tr)} val={len(va)} test={len(te)} "
              f"test_events={int(event[te].sum())}")
