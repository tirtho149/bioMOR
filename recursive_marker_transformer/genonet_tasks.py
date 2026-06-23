# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Run SMART (MoR) on the genomap genoNet classification tasks.

genomap's genoNet is a small CNN that classifies samples from constructed
genomap images. This module keeps the *tasks* (the BIO5 phenotype labels in
``data/tcga/unified_bio5.csv``: 2738 samples x 20530 genes) but swaps genoNet
for the headline SMART configuration (cross-attention marker router + expert
choice Mixture-of-Recursions), run directly on the raw, full gene-expression
vector (all 20530 genes -- "all data"). One results JSON is written per task.

Tasks (all classification, 0 NaN in the unified table):
    cancer_type      4-class  (breast / head_neck / lung / thyroid)
    pathologic_stage 4-class
    pathologic_T     4-class  (tumour size)
    pathologic_N     4-class  (lymph-node)
    os_binary        2-class  (overall survival)
    tumor_status     2-class  (with-tumour / tumour-free)

Fully reproducible: fixed seed, seeded stratified split (80/20, with a 15%
validation slice carved from train), train-split z-scoring, no network.

Usage:
    python -m recursive_marker_transformer.genonet_tasks                  # all tasks
    python -m recursive_marker_transformer.genonet_tasks --tasks cancer_type --epochs 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import RMTConfig
from .losses import RMTLoss
from .model import RecursiveMarkerTransformer
from .train import _class_weights, evaluate, resolve_device

# label columns in unified_bio5.csv (everything else is a gene)
META_COLS = ["cancer_type", "cancer_name", "os_binary", "pathologic_stage",
             "pathologic_T", "pathologic_N", "tumor_status"]
# classification tasks to run. cancer_type/cancer_name are dropped: cohort
# detection is near-linearly-separable from bulk expression (a saturated
# sanity-check, not an informative clinical task), so the genoNet benchmark here
# is the five clinical / pathology prediction tasks.
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N",
         "os_binary", "tumor_status"]


def _load_unified(csv: Path):
    """Return (X float32 [N,G], labels DataFrame, gene_names list)."""
    df = pd.read_csv(csv, index_col=0)
    meta = [c for c in META_COLS if c in df.columns]
    gene_cols = [c for c in df.columns if c not in META_COLS]
    X = df[gene_cols].values.astype(np.float32)
    return X, df[meta].copy(), gene_cols


class _DictLoader:
    """Wrap a DataLoader so it yields (x, {head: y}) like the TCGA loaders."""
    def __init__(self, X, y, idx, bs, shuffle, head):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        self.dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        self.head = head

    def __iter__(self):
        for xb, yb in self.dl:
            yield xb, {self.head: yb}

    def __len__(self):
        return len(self.dl)


def run_task(task: str, X: np.ndarray, labels: pd.DataFrame,
             base: RMTConfig, out_dir: Path) -> dict:
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    device = resolve_device(base.device)
    dtypes = {task: "multiclass"}

    y_raw = labels[task].values
    uniq = np.unique(y_raw)
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    G, K = X.shape[1], int(y.max() + 1)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=base.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.15, random_state=base.seed, stratify=y[tr])

    # z-score on the train split only.
    Xs = X.copy()
    mu = Xs[tr].mean(0, keepdims=True)
    sd = Xs[tr].std(0, keepdims=True) + 1e-6
    Xs = (Xs - mu) / sd

    cfg = replace(base, heads=(task,), n_hvg=None, n_markers=min(base.n_markers, G))
    print(f"\n########## {task}: N={len(y)} G={G} K={K} "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) device={device} ##########",
          flush=True)

    dl_tr = _DictLoader(Xs, y, tr, cfg.batch_size, True, task)
    dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False, task)
    dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False, task)

    model = RecursiveMarkerTransformer(cfg, G, {task: K}, dtypes).to(device)
    model.set_gene_variance(torch.from_numpy(Xs[tr].var(0).astype(np.float32)))

    # Biology-informed router: genomap gene-gene-interaction centrality prior,
    # built (label-free) on the train split of this task.
    if getattr(cfg, "gene_interaction", None) not in (None, "none"):
        from .interaction import build_interaction
        inter = build_interaction(Xs[tr], G, mode=cfg.gene_interaction,
                                  knn=cfg.interaction_knn, seed=cfg.seed)
        model.set_gene_interaction(inter.centrality)
        print(f"  gene_interaction={cfg.gene_interaction} prior installed "
              f"(beta0={cfg.router_prior_beta}, anneal={cfg.router_prior_anneal})",
              flush=True)

    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    criterion = RMTLoss(cfg, dtypes, {task: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        model.set_anneal(ep / max(cfg.epochs - 1, 1))
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = {h: v.to(device) for h, v in yb.items()}
            out = model(xb)
            loss = criterion(out, yb)["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        yt, yp = evaluate(model, dl_va, device, dtypes)[task]
        vf1 = f1_score(yt, yp, average="macro")
        print(f"  epoch {ep+1:2d}/{cfg.epochs}  val_macroF1={vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stop @ epoch {ep+1}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    yt, yp = evaluate(model, dl_te, device, dtypes)[task]

    # Realised per-token recursion depth + token-aware FLOP saving on the test
    # set (identical estimator to train.run, so cohort and phenotype tasks are
    # directly comparable). Early-exited tokens skip the quadratic attention of
    # later steps, so the FLOP saving exceeds the linear depth ratio.
    from .train import _depth_stats
    M = min(cfg.n_markers, G)
    mean_slot_depth, _midx, active = _depth_stats(model, dl_te, device, cfg)

    def _step_flops(a):    # attention O(a^2) + FFN O(a)
        return 4.0 * a * a * cfg.d_model + 4.0 * a * cfg.d_model * cfg.d_ff

    flops_nominal = cfg.recursion_depth * _step_flops(M)
    flops_eff = float(sum(_step_flops(float(active[t])) for t in range(cfg.recursion_depth)))
    saving = flops_eff / flops_nominal if flops_nominal else 1.0

    res = {
        "task": task,
        "n_samples": int(len(y)), "n_genes": int(G), "n_classes": int(K),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "mean_recursion_depth": float(mean_slot_depth.mean()),
        "compute_saving_ratio": saving,
        "config": cfg.as_dict(),
        "heads": {task: {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
            "per_class": classification_report(yt, yp, zero_division=0, output_dict=True),
        }},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{task}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    h = res["heads"][task]
    print(f"  [TEST] acc={h['accuracy']*100:.1f} macroF1={h['macro_f1']*100:.1f} "
          f"weightedF1={h['weighted_f1']*100:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_genonet"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_markers", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    print(f"[genonet] loading {args.csv} ...", flush=True)
    X, labels, gene_cols = _load_unified(args.csv)
    print(f"[genonet] X={X.shape}  genes={len(gene_cols)}  tasks={args.tasks}", flush=True)

    base = RMTConfig(
        heads=("cancer_type",), n_hvg=None, batch_size=args.batch_size,
        d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
        marker_mode="router", recursion_mode="expert", recursion_depth=4,
        epochs=args.epochs, patience=args.patience, lr=args.lr, device=args.device,
    )
    summary = []
    for task in args.tasks:
        r = run_task(task, X, labels, base, args.out)
        h = r["heads"][task]
        summary.append((task, r["n_classes"], h["accuracy"], h["macro_f1"], h["weighted_f1"]))
    print("\n==== SMART (MoR) on genoNet tasks -- all 20530 genes ====", flush=True)
    print(f"  {'task':18s} {'K':>2s}  {'acc':>6s} {'macroF1':>8s} {'wF1':>6s}")
    for n, k, a, f, w in summary:
        print(f"  {n:18s} {k:2d}  {a*100:6.1f} {f*100:8.1f} {w*100:6.1f}")


if __name__ == "__main__":
    main()
