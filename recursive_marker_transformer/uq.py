# ============================================================================
# SMART -- log-probability-driven uncertainty quantification (UQ).
#
# From the model's test-set softmax probabilities we compute the standard
# calibration / log-prob UQ metrics: negative log-likelihood (NLL), expected
# calibration error (ECE, 15 equal-width confidence bins), Brier score, mean
# confidence, and the AUROC with which max-probability separates correct from
# incorrect predictions (confidence as a failure detector). All are label-free of
# the architecture's claims, so they fairly test whether the biological prior or
# adaptive depth buy any *uncertainty* quality over a vanilla transformer.
#
#   python -m recursive_marker_transformer.uq_sweep      # produces results_uq/
# ============================================================================
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_probs(model, loader, device, head):
    """Run a multiclass head and return (y_true (N,), probs (N,C))."""
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logit = model(xb.to(device))["logits"][head]
        ps.append(torch.softmax(logit, dim=-1).cpu().numpy())
        ys.append(yb[head].numpy() if hasattr(yb[head], "numpy") else np.asarray(yb[head]))
    return np.concatenate(ys), np.concatenate(ps)


def uq_metrics(y_true, probs, n_bins: int = 15):
    """Log-prob / calibration UQ metrics from softmax probabilities.
    Returns dict with nll, ece, brier, conf, auroc (all floats). Lower is better
    for nll/ece/brier; higher is better for conf-AUROC."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    N, C = p.shape
    idx = np.arange(N)

    nll = float(-np.log(p[idx, y_true]).mean())
    onehot = np.zeros_like(p)
    onehot[idx, y_true] = 1.0
    brier = float(((p - onehot) ** 2).sum(1).mean())

    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y_true).astype(np.float64)
    # ECE: |accuracy - confidence| weighted by bin mass
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * (m.mean())
    # AUROC of confidence as a correctness detector (skip if one class only)
    auroc = float("nan")
    if 0 < correct.sum() < N:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(correct, conf))
        except Exception:
            pass
    return {"nll": nll, "ece": float(ece), "brier": brier,
            "conf": float(conf.mean()), "auroc": auroc}
