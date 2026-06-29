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

"""SMART on the TCGA PANCAN multimodal / molecular-subtype benchmark.

Two new pan-cancer evaluations built from the UCSC Xena PANCAN release
(``data/pancan/``, produced by ``new data/build_pancan.py``):

  * subtype classification -- predict the Thorsson pan-cancer immune subtype
    (6 classes) and the curated TCGA molecular subtype (``Subtype_Selected``,
    long tail trimmed) from gene expression alone; and

  * multimodal fusion -- the same headline SMART configuration fed an increasing
    set of gene-aligned assays (expression -> +copy-number -> +mutation), where
    each gene token fuses its C channels at the value projection. The
    expression-only run is the clean ablation baseline for the fused runs.

All experiments share one sample set (samples with expression + CNV + mutation +
an immune-subtype call) and one 18k shared-gene set, so every row is directly
comparable. Reproducible: fixed seed, seeded stratified split, per-channel
train-split z-scoring, no network.

Usage:
    python -m recursive_marker_transformer.pancan_tasks \
        --task immune_subtype --channels expr expr_cnv expr_cnv_mut --device cuda
    python -m recursive_marker_transformer.pancan_tasks \
        --task molecular_subtype --channels expr --out results_pancan_subtype
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
from .train import _class_weights, _depth_stats, evaluate, resolve_device

# channel-set name -> ordered modality keys
CHANNEL_SETS = {
    "expr":          ["expr"],
    "expr_cnv":      ["expr", "cnv"],
    "expr_cnv_mut":  ["expr", "cnv", "mut"],
}
_MODALITY_FILE = {"expr": "mm_expr.npy", "cnv": "mm_cnv.npy", "mut": "mm_mut.npy"}


def _load_pancan(root: Path):
    """Return (mats dict[name -> (N,G) float32], labels DataFrame, genes list)."""
    mats = {k: np.load(root / f) for k, f in _MODALITY_FILE.items()}
    labels = pd.read_csv(root / "labels.csv")
    genes = (root / "genes.txt").read_text().split()
    return mats, labels, genes


class _DictLoader:
    """Yield (x, {head: y}) like the TCGA loaders. x may be (B,G) or (B,G,C)."""
    def __init__(self, X, y, idx, bs, shuffle, head):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        self.dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        self.head = head

    def __iter__(self):
        for xb, yb in self.dl:
            yield xb, {self.head: yb}

    def __len__(self):
        return len(self.dl)


def _stack_channels(mats: dict, names: list[str]) -> np.ndarray:
    """Stack selected modalities into (N, G, C) float32 (C==1 stays (N,G))."""
    chans = [mats[n] for n in names]
    if len(chans) == 1:
        return chans[0].astype(np.float32, copy=False)
    return np.stack(chans, axis=-1).astype(np.float32)


def _zscore_train(X: np.ndarray, tr: np.ndarray):
    """Per-(gene, channel) z-scoring on the train rows. Handles (N,G) and (N,G,C)."""
    mu = X[tr].mean(0, keepdims=True)
    sd = X[tr].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def _fit_eval(task, X, y, tr, va, te, cfg, G, K, dtypes, device):
    """Train on tr (z-scored on its own stats), early-stop on va, eval on te."""
    Xs = _zscore_train(X, tr)
    dl_tr = _DictLoader(Xs, y, tr, cfg.batch_size, True, task)
    dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False, task)
    dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False, task)

    model = RecursiveMarkerTransformer(cfg, G, {task: K}, dtypes).to(device)
    # variance prior uses the expression channel (channel 0 when multimodal)
    expr_tr = Xs[tr] if Xs.ndim == 2 else Xs[tr, :, 0]
    model.set_gene_variance(torch.from_numpy(expr_tr.var(0).astype(np.float32)))

    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    criterion = RMTLoss(cfg, dtypes, {task: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear LR warmup (1%->100% over first ~10% of epochs) then cosine; prevents the
    # wide-model collapse-to-majority-class seen without warmup.
    _warm = max(1, round(0.1 * cfg.epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=_warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - _warm))],
        milestones=[_warm])

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        model.set_anneal(ep / max(cfg.epochs - 1, 1))
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = {h: v.to(device) for h, v in yb.items()}
            loss = criterion(model(xb), yb)["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        yt, yp = evaluate(model, dl_va, device, dtypes)[task]
        vf1 = f1_score(yt, yp, average="macro")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    yt, yp = evaluate(model, dl_te, device, dtypes)[task]
    return yt, yp, model, dl_te


def run(task, channel_set, mats, labels, base, out_dir, device):
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    names = CHANNEL_SETS[channel_set]
    dtypes = {task: "multiclass"}

    # drop samples without a label for this task
    keep = labels[task].notna().values
    X = _stack_channels(mats, names)[keep]
    y_raw = labels[task].values[keep]
    uniq = sorted(set(y_raw))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    G = X.shape[1]
    K = int(y.max() + 1)
    C = 1 if X.ndim == 2 else X.shape[2]

    cfg = replace(base, heads=(task,), n_hvg=None, n_channels=C,
                  n_markers=min(base.n_markers, G))

    idx = np.arange(len(y))
    _, cnt = np.unique(y, return_counts=True)
    strat = y if cnt.min() >= 2 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=base.seed, stratify=strat)
    _, cnt = np.unique(y[tr], return_counts=True)
    strat = y[tr] if cnt.min() >= 2 else None
    tr, va = train_test_split(tr, test_size=0.15, random_state=base.seed, stratify=strat)

    print(f"\n########## {task} [{channel_set}] N={len(y)} G={G} C={C} K={K} "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) device={device} ##########",
          flush=True)

    yt, yp, model, dl_te = _fit_eval(task, X, y, tr, va, te, cfg, G, K, dtypes, device)

    M = min(cfg.n_markers, G)
    mean_slot_depth, _midx, active = _depth_stats(model, dl_te, device, cfg)

    def _step_flops(a):
        return 4.0 * a * a * cfg.d_model + 4.0 * a * cfg.d_model * cfg.d_ff
    flops_nominal = cfg.recursion_depth * _step_flops(M)
    flops_eff = float(sum(_step_flops(float(active[t])) for t in range(cfg.recursion_depth)))
    saving = flops_eff / flops_nominal if flops_nominal else 1.0

    res = {
        "task": task, "channel_set": channel_set, "channels": names, "n_channels": C,
        "n_samples": int(len(y)), "n_genes": int(G), "n_classes": int(K),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "class_names": [str(u) for u in uniq],
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "mean_recursion_depth": float(mean_slot_depth.mean()),
        "compute_saving_ratio": saving,
        "config": cfg.as_dict(),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "per_class": classification_report(yt, yp, zero_division=0, output_dict=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{task}__{channel_set}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"  [TEST] {task} [{channel_set}] acc={res['accuracy']*100:.1f} "
          f"macroF1={res['macro_f1']*100:.1f} weightedF1={res['weighted_f1']*100:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pancan"))
    ap.add_argument("--out", type=Path, default=Path("results_pancan"))
    ap.add_argument("--task", type=str, default="immune_subtype",
                    choices=["immune_subtype", "molecular_subtype"])
    ap.add_argument("--channels", nargs="+", default=["expr"],
                    choices=list(CHANNEL_SETS.keys()))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_markers", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[pancan] loading {args.data} ...", flush=True)
    mats, labels, genes = _load_pancan(args.data)
    print(f"[pancan] mats={ {k: v.shape for k, v in mats.items()} }  "
          f"samples={len(labels)}  genes={len(genes)}", flush=True)

    base = RMTConfig(
        heads=(args.task,), n_hvg=None, batch_size=args.batch_size,
        d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
        marker_mode="router", recursion_mode="expert", recursion_depth=4,
        share_weights=True, seed=args.seed, epochs=args.epochs,
        patience=args.patience, lr=args.lr, device=args.device,
    )
    for cs in args.channels:
        out = args.out / f"{args.task}__{cs}.json"
        if out.exists():
            print(f"[pancan] [skip] {args.task} [{cs}] (done)", flush=True)
            continue
        run(args.task, cs, mats, labels, base, args.out, device)
    print(f"\n[pancan] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
