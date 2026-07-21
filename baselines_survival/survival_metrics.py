"""Shared survival metric definitions for the 2-modality pan-cancer baselines.

Every survival baseline imports these so the Cox loss and C-index are computed
identically everywhere. Metric: Harrell's concordance index (higher is better;
higher predicted risk should mean shorter survival).
"""
import json
import os

import numpy as np

# NOTE: torch is imported lazily inside cox_partial_loglik_loss so that the
# Keras-env baselines (PathCNN) can import the numpy-based C-index helpers
# without requiring torch in that environment.


def cox_partial_loglik_loss(risk, time, event, eps: float = 1e-8):
    """Negative Cox partial log-likelihood (Breslow ties) within the batch.

    risk:  (B,) predicted log-relative-hazard.
    time:  (B,) durations.   event: (B,) 1=event, 0=censored.
    Returns a scalar tensor; if the batch has no events the loss is 0.
    """
    import torch
    if event.sum() < 1:
        return torch.zeros((), device=risk.device, requires_grad=True)
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order].float()
    log_risk_cum = torch.logcumsumexp(r, dim=0)
    loss_per_event = -(r - log_risk_cum) * e
    return loss_per_event.sum() / (e.sum() + eps)


def harrell_c_index(risk, time, event) -> float:
    """Harrell's concordance index. Higher risk => shorter survival."""
    risk = np.asarray(risk).reshape(-1)
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(int)
    t_diff = time[None, :] - time[:, None]
    valid = (event[:, None] == 1) & (t_diff > 0)
    risk_diff = risk[None, :] - risk[:, None]
    concordant = (risk_diff < 0).astype(float) + 0.5 * (risk_diff == 0).astype(float)
    num = float((valid * concordant).sum())
    den = float(valid.sum())
    return num / den if den > 0 else float("nan")


def fold_metrics(risk, time, event):
    """Per-fold survival metrics dict."""
    return {
        "c_index": harrell_c_index(risk, time, event),
        "n": int(np.asarray(time).size),
        "events": int(np.asarray(event).sum()),
    }


def write_risk_scores(path, patients, fold_records):
    """Write pooled per-test-patient risk scores for post-hoc analysis
    (e.g. within-cancer-type C-index).

    patients:      array of sample IDs indexable by each fold's test_idx.
    fold_records:  list of dicts, one per fold, with keys:
                   test_idx, risk, time, event.
    Output CSV columns: Sample, time, event, risk, fold.
    """
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    patients = np.asarray(patients)
    rows = []
    for fi, rec in enumerate(fold_records, 1):
        ti = np.asarray(rec["test_idx"]).reshape(-1)
        rows.append(pd.DataFrame({
            "Sample": patients[ti],
            "time": np.asarray(rec["time"]).reshape(-1),
            "event": np.asarray(rec["event"]).reshape(-1).astype(int),
            "risk": np.asarray(rec["risk"]).reshape(-1),
            "fold": fi,
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(path, index=False)
    return path


def write_metrics_json(path, baseline, dataset, per_fold):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c_vals = np.array([m["c_index"] for m in per_fold if not np.isnan(m["c_index"])])
    payload = {
        "baseline": baseline,
        "dataset": dataset,
        "metric": "c_index",
        "c_index_mean": float(c_vals.mean()) if len(c_vals) else float("nan"),
        "c_index_std": float(c_vals.std()) if len(c_vals) else float("nan"),
        "folds": [dict(fold=i, **m) for i, m in enumerate(per_fold, 1)],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
