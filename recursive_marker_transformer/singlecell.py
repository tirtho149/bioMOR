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

"""Run SMART on the converted single-cell datasets (genomap capsule 6967747).

Each dataset lives in ``data/singlecell/<name>/`` as produced by
``tools/convert_capsule_to_csv.py`` (expression.csv.gz + labels.csv + optional
split.csv). This module trains the headline SMART configuration (cross-attention
marker router + expert-choice Mixture-of-Recursions) on every dataset and writes
one results JSON per dataset, which ``make_paper.py`` renders into the single-cell
generalisation table.

Fully reproducible: fixed seed, deterministic split (the dataset's own split.csv
when present, otherwise a seeded stratified split), no network.

Usage:
    python -m recursive_marker_transformer.singlecell                 # all datasets
    python -m recursive_marker_transformer.singlecell --datasets pancreas --epochs 6
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

HEAD = "cell_type"
_DTYPES = {HEAD: "multiclass"}


def _load_dataset(d: Path):
    """Return (X float32 [N,F], y int64 [N] 0-based, split str[N] | None)."""
    X = pd.read_csv(d / "expression.csv.gz", index_col="cell_id").values.astype(np.float32)
    y_raw = pd.read_csv(d / "labels.csv", index_col="cell_id")["label"].values.astype(np.int64)
    uniq = np.unique(y_raw)
    remap = {v: i for i, v in enumerate(uniq)}             # contiguous 0..K-1
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    split = None
    if (d / "split.csv").exists():
        split = pd.read_csv(d / "split.csv", index_col="cell_id")["split"].values.astype(str)
    return X, y, split


def _make_splits(y, split, seed):
    """Genomap-paper protocol (Islam & Xing, Nat. Commun. 2023): use the capsule's
    shipped train/test split when present (this is the paper's exact split, e.g.
    Tabula Muris 38,407/16,458 = 70/30; pancreas = the integration split); otherwise
    a stratified 70/30 train/test split, "a standard practice" per the paper."""
    idx = np.arange(len(y))
    if split is not None and (split == "train").any() and (split == "test").any():
        tr, te = idx[split == "train"], idx[split == "test"]
    else:
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    # carve a small validation slice out of train (stratified) for early stopping;
    # the test set is untouched and remains the paper's exact 30% hold-out.
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=y[tr])
    return tr, va, te


class _DictLoader:
    """Wrap a DataLoader so it yields (x, {HEAD: y}) like the TCGA loaders."""
    def __init__(self, X, y, idx, bs, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        self.dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)

    def __iter__(self):
        for xb, yb in self.dl:
            yield xb, {HEAD: yb}

    def __len__(self):
        return len(self.dl)


def _fit_eval(Xs_full, y, tr, va, te, cfg, F, K, device):
    """Train on `tr` (z-scored on its own stats), early-stop on `va`, eval on `te`.
    Returns (y_true, y_pred, model). Shared by the single-split and CV paths."""
    mu = Xs_full[tr].mean(0, keepdims=True)
    sd = Xs_full[tr].std(0, keepdims=True) + 1e-6
    Xs = (Xs_full - mu) / sd

    dl_tr = _DictLoader(Xs, y, tr, cfg.batch_size, True)
    dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False)
    dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False)

    model = RecursiveMarkerTransformer(cfg, F, {HEAD: K}, _DTYPES).to(device)
    model.set_gene_variance(torch.from_numpy(Xs[tr].var(0).astype(np.float32)))
    # Biology-informed router: build the genomap gene-gene-interaction centrality
    # prior on the train split (expression only, label-free -> leakage-safe) and
    # install it on the depth router. coexpr = genomap correlation graph; random =
    # degree-matched control; none = original SMART router.
    if getattr(cfg, "gene_interaction", None) not in (None, "none"):
        from .interaction import build_interaction
        inter = build_interaction(Xs[tr].astype(np.float32, copy=False), F,
                                  mode=cfg.gene_interaction, knn=cfg.interaction_knn,
                                  seed=cfg.seed)
        model.set_gene_interaction(inter.centrality)
        print(f"  [bio-router] gene_interaction={cfg.gene_interaction} "
              f"beta0={cfg.router_prior_beta} anneal={cfg.router_prior_anneal} "
              f"knn={cfg.interaction_knn} prior installed", flush=True)
    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    criterion = RMTLoss(cfg, _DTYPES, {HEAD: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear LR warmup (1%->100% over the first ~10% of epochs) then cosine anneal.
    # Without warmup, wide models (d_model 192/384) hit full LR at step 0 and collapse
    # to majority-class prediction; warmup stabilises training across all widths/archs.
    warm = max(1, round(0.1 * cfg.epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - warm))],
        milestones=[warm])

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
            if not torch.isfinite(loss):     # skip non-finite steps instead of poisoning weights
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        yt, yp = evaluate(model, dl_va, device, _DTYPES)[HEAD]
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

    yt, yp = evaluate(model, dl_te, device, _DTYPES)[HEAD]
    return yt, yp, model


def run_dataset_cv(name: str, data_root: Path, base: RMTConfig, out_dir: Path,
                   folds: int = 5) -> dict:
    """Stratified k-fold CV on a single-cell dataset; macro-F1 / accuracy reported
    as mean +/- std over folds (identical protocol to the TCGA cohort CV)."""
    from sklearn.model_selection import StratifiedKFold
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    device = resolve_device(base.device)

    X, y, _ = _load_dataset(data_root / name)
    F, K = X.shape[1], int(y.max() + 1)
    cfg = replace(base, heads=(HEAD,), n_markers=min(base.n_markers, F))
    Xf = X.astype(np.float32, copy=False)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=base.seed)
    print(f"\n########## CV {name}: N={len(y)} F={F} K={K} folds={folds} device={device} "
          f"##########", flush=True)

    fold_metrics, model = [], None
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        _, cnt = np.unique(y[tr_all], return_counts=True)
        strat = y[tr_all] if cnt.min() >= 2 else None
        tr, va = train_test_split(tr_all, test_size=0.15,
                                  random_state=base.seed + fi, stratify=strat)
        yt, yp, model = _fit_eval(Xf, y, tr, va, te, cfg, F, K, device)
        m = {"fold": fi, "n_test": int(len(te)),
             "accuracy": float(accuracy_score(yt, yp)),
             "macro_f1": float(f1_score(yt, yp, average="macro")),
             "weighted_f1": float(f1_score(yt, yp, average="weighted"))}
        fold_metrics.append(m)
        print(f"  fold {fi+1}/{folds}: acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f} (test {len(te)})", flush=True)

    def _ms(key):
        v = np.array([m[key] for m in fold_metrics], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=0))}

    res = {
        "dataset": name, "n_samples": int(len(y)), "n_features": int(F),
        "n_classes": int(K), "cv_folds": folds,
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "config": cfg.as_dict(),
        "fold_metrics": fold_metrics,
        "accuracy": _ms("accuracy"),
        "macro_f1": _ms("macro_f1"),
        "weighted_f1": _ms("weighted_f1"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"  [CV] {name}: acc={res['accuracy']['mean']:.4f}+/-{res['accuracy']['std']:.4f} "
          f"macroF1={res['macro_f1']['mean']:.4f}+/-{res['macro_f1']['std']:.4f}", flush=True)
    return res


def run_dataset(name: str, data_root: Path, base: RMTConfig, out_dir: Path) -> dict:
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    device = resolve_device(base.device)

    X, y, split = _load_dataset(data_root / name)
    F, K = X.shape[1], int(y.max() + 1)
    tr, va, te = _make_splits(y, split, base.seed)

    cfg = replace(base, heads=(HEAD,), n_markers=min(base.n_markers, F))
    print(f"\n########## {name}: N={len(y)} F={F} K={K} "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) device={device} ##########")

    yt, yp, model = _fit_eval(X.astype(np.float32, copy=False), y, tr, va, te, cfg, F, K, device)
    res = {
        "dataset": name,
        "n_samples": int(len(y)), "n_features": int(F), "n_classes": int(K),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "config": cfg.as_dict(),
        "heads": {HEAD: {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
            "per_class": classification_report(yt, yp, zero_division=0, output_dict=True),
        }},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    h = res["heads"][HEAD]
    print(f"  [TEST] acc={h['accuracy']*100:.1f} macroF1={h['macro_f1']*100:.1f} "
          f"weightedF1={h['weighted_f1']*100:.1f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--out", type=Path, default=Path("results_singlecell"))
    ap.add_argument("--datasets", nargs="*",
                    default=["tabula_muris", "common_class", "prototype", "pancreas"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_markers", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--recursion_mode", type=str, default="expert",
                    choices=["fixed", "expert", "token"],
                    help="expert=shared early-exit (default), token=MoR token routing")
    ap.add_argument("--recursion_depth", type=int, default=4,
                    help="K recursion steps (1 = no recursion, single pass)")
    ap.add_argument("--marker_mode", type=str, default="router",
                    choices=["router", "concrete", "random", "variance"],
                    help="router=cross-attention selector (default); random/variance=baselines")
    ap.add_argument("--share_weights", dest="share_weights", action="store_true", default=True)
    ap.add_argument("--no_share_weights", dest="share_weights", action="store_false",
                    help="untie the recursion blocks (Independent stack)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--share_strategy", type=str, default="cycle",
                    choices=["cycle", "sequence", "middle_cycle", "middle_sequence"],
                    help="MoR parameter-sharing scheme over the K recursion steps")
    ap.add_argument("--n_unique_blocks", type=int, default=None,
                    help="# distinct blocks; None -> 1 if shared else K")
    ap.add_argument("--step_cache", action="store_true",
                    help="reuse step-1 attention K/V across recursions (KV-reuse analogue)")
    ap.add_argument("--router_type", type=str, default="linear", choices=["linear", "mlp"])
    ap.add_argument("--router_z_coeff", type=float, default=1e-3)
    ap.add_argument("--router_balance_coeff", type=float, default=0.1)
    ap.add_argument("--router_alpha", type=float, default=1.0)
    ap.add_argument("--router_temp", type=float, default=1.0)
    ap.add_argument("--cv_folds", type=int, default=0,
                    help="if >0, stratified k-fold CV reporting mean+/-std")
    args = ap.parse_args()

    base = RMTConfig(
        heads=(HEAD,), n_hvg=None, batch_size=args.batch_size,
        d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
        marker_mode=args.marker_mode, recursion_mode=args.recursion_mode,
        recursion_depth=args.recursion_depth,
        share_weights=args.share_weights, seed=args.seed,
        share_strategy=args.share_strategy, n_unique_blocks=args.n_unique_blocks,
        step_cache=args.step_cache, router_type=args.router_type,
        router_z_coeff=args.router_z_coeff, router_balance_coeff=args.router_balance_coeff,
        router_alpha=args.router_alpha, router_temp=args.router_temp,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device,
    )
    print(f"[singlecell] variant: recursion_mode={args.recursion_mode} "
          f"share_weights={args.share_weights} cv_folds={args.cv_folds}", flush=True)
    summary = []
    for name in args.datasets:
        if not (args.data / name).exists():
            print(f"[skip] {name}: not found under {args.data}")
            continue
        if args.cv_folds > 0:
            r = run_dataset_cv(name, args.data, base, args.out, folds=args.cv_folds)
            summary.append((name, r["accuracy"]["mean"], r["accuracy"]["std"],
                            r["macro_f1"]["mean"], r["macro_f1"]["std"]))
        else:
            r = run_dataset(name, args.data, base, args.out)
            h = r["heads"][HEAD]
            summary.append((name, h["accuracy"], 0.0, h["macro_f1"], 0.0))
    tag = f"{args.cv_folds}-fold CV (mean+/-std)" if args.cv_folds > 0 else "single split"
    print(f"\n==== SMART single-cell summary [{tag}] ====")
    for n, a, asd, f, fsd in summary:
        print(f"  {n:14s} acc={a*100:5.1f}+/-{asd*100:3.1f}  macroF1={f*100:5.1f}+/-{fsd*100:3.1f}")


if __name__ == "__main__":
    main()
