# ============================================================================
# SMART -- learned-graph bio-router on the genomap single-cell suite.
# Copyright (c) 2026 The SMART Authors. PROPRIETARY AND CONFIDENTIAL. See LICENSE.
# ============================================================================

"""Run the bio-router modes (none / coexpr / random / LEARNED) on the collaborator
genomap single-cell datasets (genomap_data/, materialised .npy/.mat).

The learned data-driven graph needs only expression (no gene symbols), so unlike the
curated Reactome net it runs on the anonymised single-cell suite. This is the
"does the learned graph generalise beyond P-NET?" sweep.

    python -m recursive_marker_transformer.bio_learned_genomap \
        --dataset Spleen --modes none coexpr random learned --seeds 0 1 2 --epochs 60
-> results_learned_genomap/<dataset>/<mode>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .singlecell import HEAD, _fit_eval, _make_splits
from .train import resolve_device

ROOT = Path(__file__).resolve().parents[1]
GD = ROOT / "genomap_data"

# dataset -> (kind, path-parts). Supervised cell-recognition sets only (Retinal /
# Trajectory are unsupervised and excluded).
DATASETS = {
    "Lung":       ("ischaemic", "Ischaemic/Lung"),
    "Oesophagus": ("ischaemic", "Ischaemic/Oesophagus"),
    "Spleen":     ("ischaemic", "Ischaemic/Spleen"),
    "Tcell":      ("tcell",     "Tcell/Elyahu2019_SCP490"),
    "Baron":      ("pancreas",  "Pancreas/Baron"),
    "Muraro":     ("pancreas",  "Pancreas/Muraro"),
    "Segerstolpe":("pancreas",  "Pancreas/Segerstolpe"),
    "Wang":       ("pancreas",  "Pancreas/Wang"),
    "Xin":        ("pancreas",  "Pancreas/Xin"),
}


def _encode(y_raw):
    uniq = np.unique(y_raw)
    remap = {v: i for i, v in enumerate(uniq)}
    return np.array([remap[v] for v in y_raw], dtype=np.int64)


def load_genomap(dataset: str):
    """Return (X float32 [N,G], y int64 [N])."""
    kind, rel = DATASETS[dataset]
    d = GD / rel
    if kind in ("ischaemic", "tcell"):
        stem = dataset.lower() if kind == "ischaemic" else "tcell"
        X = np.load(d / f"{stem}_data.npy").astype(np.float32)
        gt = sio.loadmat(d / f"GT_{stem}.mat")["GT"].ravel()
        y = _encode(gt)
    else:  # pancreas: data{DS}X.mat holds BOTH the (N,G) matrix and the (N,1) GT
        mats = [p for p in d.glob("*.mat") if p.name.startswith("data")]
        m = sio.loadmat(mats[0])
        arrs = {k: v for k, v in m.items() if not k.startswith("__")}
        Xk = max(arrs, key=lambda k: arrs[k].size)                 # the 2D data matrix
        X = np.asarray(arrs[Xk], dtype=np.float32)
        gk = [k for k in arrs if arrs[k].shape[0] == X.shape[0] and k != Xk]
        y = _encode(np.asarray(arrs[gk[0]]).ravel())
    return X, y


def load_gene_symbols(dataset: str):
    """Gene symbols for `dataset` if the source stored them, else None. Only Tcell (mouse)
    ships a gene-name array; the pancreas/ischaemic suites are anonymized (no symbols), so
    an external gene network cannot be mapped onto them."""
    kind, rel = DATASETS[dataset]
    d = GD / rel
    p = d / "tcell_gene_names.npy"
    if kind == "tcell" and p.exists():
        names = np.load(p, allow_pickle=True)
        # the data matrix is HVG-subset; align the full gene-name list to the kept columns
        hvg = d / "tcell_hvg_idx.npy"
        if hvg.exists():
            idx = np.load(hvg)
            names = names[idx]
        return [str(g) for g in names]
    return None


def _cfg(mode: str, K: int, seed: int, epochs: int, n_markers: int = 128) -> RMTConfig:
    base = dict(
        heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
        n_markers=n_markers, marker_mode="router", recursion_mode="expert",
        recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
        patience=12, lr=1e-3, weight_decay=1e-5, device="cuda",
        gene_interaction=(mode if mode in ("coexpr", "random", "aggnet") else "none"),
    )
    cfg = RMTConfig(**base)
    if mode in ("coexpr", "random"):
        cfg.bio_graph_prop = True; cfg.bio_prop_lambda_init = 0.3; cfg.bio_prop_hops = 1
        cfg.bio_prior_gate = True; cfg.bio_prior_learnable = True; cfg.bio_beta_init = 0.5
        cfg.bio_depth_laplacian = 0.01; cfg.bio_centrality = "ppr"
        cfg.router_prior_anneal = False
    elif mode == "aggnet":
        # aggregated external network (STRING+KEGG[+Reactome]) smoothing prior; needs
        # gene symbols (Tcell only). Same smoothing knobs as coexpr for a fair comparison.
        cfg.bio_graph_prop = True; cfg.bio_prop_lambda_init = 0.3; cfg.bio_prop_hops = 1
        cfg.aggnet_species = "mouse"
    elif mode == "learned":
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
    elif mode == "learned_aggnet":
        # learned graph warm-started AND annealed-anchored to the AGGREGATED external
        # network (STRING+KEGG[+Reactome]) instead of noisy co-expression: clean biology
        # as the starting structure + a standing pull, refined end-to-end. The strongest
        # honest test of "biology helps if you give the learned graph a real network."
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"
        cfg.bio_init_scale = 0.05; cfg.bio_init_rand = 0.005
        cfg.bio_learned_anchor = True; cfg.bio_anchor_lambda = 0.5; cfg.bio_anchor_floor = 0.2
        cfg.aggnet_species = "mouse"
    # --- C1 confound factorial: isolate input SMOOTHING from depth ROUTING ---
    elif mode in ("smooth_coexpr", "smooth_random"):
        # SMOOTHING ONLY: propagate x along the fixed graph; NO routing prior.
        cfg.gene_interaction = "coexpr" if mode == "smooth_coexpr" else "random"
        cfg.bio_graph_prop = True; cfg.bio_prop_lambda_init = 0.3; cfg.bio_prop_hops = 1
        cfg.bio_prior_gate = False; cfg.router_prior_beta = 0.0
        cfg.bio_depth_laplacian = 0.0; cfg.bio_centrality = "ppr"; cfg.router_prior_anneal = False
    elif mode in ("route_coexpr", "route_random"):
        # ROUTING ONLY: fixed centrality prior on the depth router; NO input smoothing.
        cfg.gene_interaction = "coexpr" if mode == "route_coexpr" else "random"
        cfg.bio_graph_prop = False; cfg.bio_prior_gate = True; cfg.bio_prior_learnable = True
        cfg.bio_beta_init = 0.5; cfg.bio_depth_laplacian = 0.0
        cfg.bio_centrality = "ppr"; cfg.router_prior_anneal = False
    elif mode == "learned_bio":
        # learned graph, IDENTICAL to `learned` except gene_embed is warm-started from
        # the co-expression graph (degenerate/NaN graphs fall back to random init).
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"
        cfg.bio_init_scale = 0.01; cfg.bio_init_rand = 0.01   # original weak warm-start (unchanged)
    elif mode == "learned_anchor":
        # learned graph, warm-started from biology AND held near it early by an annealed
        # ||A_learned - A_bio||^2 penalty (lambda: 0.5 -> 0 over training), with a larger
        # biological init footprint. The fix for "learned_bio == random-init learned":
        # keeps the warm-start from being overwritten before it can shape marker choice.
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"
        cfg.bio_learned_anchor = True; cfg.bio_anchor_lambda = 0.5; cfg.bio_anchor_rank = 16
        cfg.bio_init_scale = 0.05; cfg.bio_init_rand = 0.005
    elif mode == "learned_bigbio":
        # ABLATION: larger biological init footprint (0.05 vs 0.005) but NO anchor penalty.
        # Isolates whether any learned_anchor gain is the anchor or merely a stronger init.
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"
        cfg.bio_init_scale = 0.05; cfg.bio_init_rand = 0.005
    elif mode == "learned_fused":
        # graph comes from BIOLOGY + LEARNING: co-expression interaction matrix kept as a
        # persistent, learnably-gated propagation term alongside the learned graph.
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"; cfg.bio_learned_fuse = True; cfg.bio_fuse_source = "coexpr"
    elif mode == "learned_fused_rand":
        # control: fuse a degree-matched RANDOM graph instead of the biological one, to
        # prove any gain from learned_fused is biology, not just the extra mechanism.
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "random"; cfg.bio_learned_fuse = True; cfg.bio_fuse_source = "random"
    return cfg


def run_cell(X, y, dataset, mode, K, seed, epochs, device, n_markers=128, overrides=None):
    F, C = X.shape[1], int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, None, seed)
    cfg = _cfg(mode, K, seed, epochs, n_markers=n_markers); cfg.n_markers = min(cfg.n_markers, F)
    for k, v in (overrides or {}).items():        # tuning-sweep hyperparameter overrides
        setattr(cfg, k, v)
    need_symbols = getattr(cfg, "gene_interaction", None) == "aggnet" or mode == "learned_aggnet"
    gene_symbols = load_gene_symbols(dataset) if need_symbols else None
    bio_op = None
    if mode == "learned_aggnet":
        if gene_symbols is not None and len(gene_symbols) == F:
            from .bio_network import load_aggregated_adjacency
            bio_op = load_aggregated_adjacency(
                list(gene_symbols), species=getattr(cfg, "aggnet_species", "mouse")).astype(np.float32)
            print(f"  [learned_aggnet] warm-start/anchor from aggregated network ({F} genes)", flush=True)
        else:
            print(f"  [learned_aggnet] no gene symbols for {dataset} -> random init (no biology)", flush=True)
    yt, yp, model = _fit_eval(X.astype(np.float32), y, tr, va, te, cfg, F, C, device,
                              bio_op=bio_op, gene_symbols=gene_symbols)
    out = {"dataset": dataset, "mode": mode, "K": K, "seed": seed, "n_markers": cfg.n_markers,
           "test_macro_f1": 100 * f1_score(yt, yp, average="macro"),
           "test_accuracy": 100 * accuracy_score(yt, yp),
           "val_macro_f1": getattr(model, "_val_f1", None),   # for validation-based config selection
           "n_features": F, "n_classes": C, "n_samples": int(len(y))}
    if mode in ("learned", "learned_bio", "learned_anchor", "learned_aggnet", "learned_bigbio", "learned_fused", "learned_fused_rand"):
        with torch.no_grad():
            out["learned_lambda"] = float(torch.sigmoid(model.bio_prop_logit))
            out["bio_anchor_have"] = bool(getattr(model, "_bio_anchor_have", False))
            if hasattr(model, "bio_fuse_gate"):
                out["fuse_gate"] = float(torch.sigmoid(model.bio_fuse_gate))
    return out


def run_cell_cv(X, y, dataset, mode, K, epochs, device, n_markers=128,
                overrides=None, n_folds=5):
    """5-fold CV variant of run_cell: identical shared folds (cv.cv_folds, seed 42,
    20% test / 10%-of-train val), fresh training per fold, macro-F1 mean +/- SD."""
    from .cv import cv_folds, summarize, SEED, VAL_FRAC
    F, C = X.shape[1], int(y.max() + 1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _cfg(mode, K, SEED, epochs, n_markers=n_markers); cfg.n_markers = min(cfg.n_markers, F)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    need_symbols = getattr(cfg, "gene_interaction", None) == "aggnet" or mode == "learned_aggnet"
    gene_symbols = load_gene_symbols(dataset) if need_symbols else None
    bio_op = None
    if mode == "learned_aggnet":
        if gene_symbols is not None and len(gene_symbols) == F:
            from .bio_network import load_aggregated_adjacency
            bio_op = load_aggregated_adjacency(
                list(gene_symbols), species=getattr(cfg, "aggnet_species", "mouse")).astype(np.float32)
        else:
            print(f"  [learned_aggnet] no gene symbols for {dataset} -> random init", flush=True)

    Xf = X.astype(np.float32)
    fold_f1, fold_acc, model = [], [], None
    for fi, (tr, va, te) in enumerate(cv_folds(y, n_folds=n_folds, seed=SEED, val_frac=VAL_FRAC)):
        yt, yp, model = _fit_eval(Xf, y, tr, va, te, cfg, F, C, device,
                                  bio_op=bio_op, gene_symbols=gene_symbols)
        f1 = 100.0 * f1_score(yt, yp, average="macro")
        fold_f1.append(f1); fold_acc.append(100.0 * accuracy_score(yt, yp))
        print(f"  fold {fi+1}/{n_folds}: macroF1={f1:.2f} (test {len(te)})", flush=True)
    out = {"dataset": dataset, "mode": mode, "K": K, "n_markers": cfg.n_markers,
           "n_features": F, "n_classes": C, "n_samples": int(len(y)),
           "n_folds": n_folds, "seed": SEED, "val_frac": VAL_FRAC,
           "cv_macro_f1": summarize(fold_f1), "cv_accuracy": summarize(fold_acc),
           "config": cfg.as_dict() if hasattr(cfg, "as_dict") else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--modes", nargs="*", default=["none", "coexpr", "random", "learned"])
    ap.add_argument("--cv_folds", type=int, default=0,
                    help="if >0, run unified k-fold CV (mean+/-SD) instead of per-seed single split")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_markers", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results_learned_genomap")
    ap.add_argument("--force", action="store_true")
    # anchor tuning overrides (applied to any anchor mode); None = leave mode default
    ap.add_argument("--anchor_lambda", type=float, default=None)
    ap.add_argument("--anchor_floor", type=float, default=None)
    ap.add_argument("--anchor_rank", type=int, default=None)
    ap.add_argument("--init_scale", type=float, default=None)
    ap.add_argument("--recursion_mode", default=None,
                    help="override router type (e.g. 'token' for token-choice bioMoR)")
    ap.add_argument("--step_cache", action="store_true", help="enable KV-cache (needs fixed/token mode)")
    ap.add_argument("--share_strategy", default=None, help="e.g. 'middle_cycle' for M-Cyc share")
    ap.add_argument("--patience", type=int, default=None, help="early-stop patience override")
    ap.add_argument("--d_model", type=int, default=None, help="width override (d_ff set to 2*d_model)")
    args = ap.parse_args()
    overrides = {}
    if args.recursion_mode: overrides["recursion_mode"] = args.recursion_mode
    if args.step_cache: overrides["step_cache"] = True
    if args.share_strategy: overrides["share_strategy"] = args.share_strategy
    if args.patience is not None: overrides["patience"] = args.patience
    if args.d_model is not None: overrides["d_model"] = args.d_model; overrides["d_ff"] = 2 * args.d_model
    if args.anchor_lambda is not None: overrides["bio_anchor_lambda"] = args.anchor_lambda
    if args.anchor_floor is not None: overrides["bio_anchor_floor"] = args.anchor_floor
    if args.anchor_rank is not None: overrides["bio_anchor_rank"] = args.anchor_rank
    if args.init_scale is not None: overrides["bio_init_scale"] = args.init_scale
    device = resolve_device(args.device)
    X, y = load_genomap(args.dataset)
    out_dir = args.out / args.dataset; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[genomap] {args.dataset} N={len(y)} G={X.shape[1]} C={int(y.max()+1)}", flush=True)

    summary = []
    if args.cv_folds > 0:
        for mode in args.modes:
            p = out_dir / f"{mode}_cv.json"
            if p.exists() and not args.force:
                summary.append(json.loads(p.read_text())); continue
            print(f"\n##### genomap CV {args.dataset} mode={mode} folds={args.cv_folds} #####", flush=True)
            r = run_cell_cv(X, y, args.dataset, mode, args.K, args.epochs, device,
                            n_markers=args.n_markers, overrides=overrides, n_folds=args.cv_folds)
            p.write_text(json.dumps(r, indent=1))
            print(f"  [{mode} CV] macroF1={r['cv_macro_f1']['mean']:.2f}+/-{r['cv_macro_f1']['std']:.2f}",
                  flush=True)
            summary.append(r)
        return
    for mode in args.modes:
        for seed in args.seeds:
            p = out_dir / f"{mode}_s{seed}.json"
            if p.exists() and not args.force:
                summary.append(json.loads(p.read_text())); continue
            print(f"\n##### genomap {args.dataset} mode={mode} seed={seed} #####", flush=True)
            r = run_cell(X, y, args.dataset, mode, args.K, seed, args.epochs, device,
                         n_markers=args.n_markers, overrides=overrides)
            p.write_text(json.dumps(r, indent=1))
            print(f"  [{mode} s{seed}] F1={r['test_macro_f1']:.2f} acc={r['test_accuracy']:.2f}"
                  + (f" lam={r.get('learned_lambda'):.3f}" if "learned_lambda" in r else ""),
                  flush=True)
            summary.append(r)

    def _mean(m):
        v = [s["test_macro_f1"] for s in summary if s["mode"] == m]; return sum(v)/len(v) if v else float("nan")
    print(f"\n==== {args.dataset} summary (mean macro-F1) ====")
    print("  " + "  ".join(f"{m}={_mean(m):.2f}" for m in args.modes))
    if "learned" in args.modes and "none" in args.modes:
        print(f"  learned-none={_mean('learned')-_mean('none'):+.2f}", flush=True)


if __name__ == "__main__":
    main()
