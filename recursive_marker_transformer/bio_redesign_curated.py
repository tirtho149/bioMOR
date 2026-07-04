# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL. See LICENSE.
# ============================================================================

"""CURATED-NETWORK bio-router falsification on symbol-bearing cohorts.

The co-expression redesign (bio_redesign.py) provably ties its degree-matched
shuffle on single-cell data: a shuffle of a |corr| covariation graph reproduces
its statistics (FACT B). This runner swaps the graph SOURCE for a CURATED Reactome
gene-gene network -- gene pairs co-occurring in a Reactome pathway -- whose edges
encode literature biology a degree-preserving shuffle cannot recover. It runs the
IDENTICAL redesigned mechanism (Fix A propagation / Fix B gate / Fix D learnable
beta / Fix E Laplacian, label-aware PPR centrality) on:

  * ``curated`` -- the real Reactome gene-gene graph;
  * ``random``  -- that graph under a random gene RELABELLING (degree/weight/
     spectrum identical, biological identity destroyed -- the exact FACT-B control);
  * ``none``    -- no graph (plain SMART router).

Cohorts (need gene symbols; single-cell sets are anonymised):
  * P-NET  : prostate / blca / stad / brca  (Reactome-native, mutation channel);
  * TCGA   : breast / lung / head_neck / thyroid (expression; pathologic stage/T/N).

Success (pre-registered, BIO_ROUTER_REDESIGN.txt §6): pooled paired
mean(curated) - mean(random) >= +2.0 macro-F1, one-sided Wilcoxon p<0.05, AND
curated >= none.

    python -m recursive_marker_transformer.bio_redesign_curated \
        --family tcga --cohort lung --task pathologic_stage \
        --modes none curated random --seeds 0 1 2 --epochs 60
-> results_bio_curated/<family>/<cohort>__<task>/<mode>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .interaction import _reactome_membership, build_reactome_falsification
from .singlecell import HEAD, _fit_eval, _make_splits
from .train import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Global Reactome membership (identical 1268-pathway list across cohorts); used to
# build the gene-gene co-membership graph for ANY symbol list.
REACTOME_CSV = DATA / "brca" / "filtered_pathways.csv"

PNET_COHORTS = ["prostate", "blca", "stad", "brca"]
TCGA_COHORTS = ["breast", "lung", "head_neck", "thyroid"]
TCGA_TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary"]


# --------------------------------------------------------------------------- #
#  Cohort loaders -> (X float32 [N,G], y int64 [N], gene_symbols list, classes) #
# --------------------------------------------------------------------------- #
def load_tcga(cohort: str, task: str):
    import pandas as pd
    gf = DATA / "tcga" / f"{cohort}_genes.csv"
    lf = DATA / "tcga" / f"{cohort}_labels.csv"
    if not gf.exists() or not lf.exists():
        return None
    lab = pd.read_csv(lf, index_col=0)
    if task not in lab.columns:
        return None
    X = pd.read_csv(gf, index_col=0)
    yv = lab[task]
    common = [s for s in X.index if s in yv.index and pd.notna(yv.loc[s])]
    if len(common) < 60:
        return None
    Xv = X.loc[common].values.astype(np.float32)
    yr = yv.loc[common].astype(str).values
    genes = [str(g) for g in X.columns]
    # drop classes with < 10 samples, then require >=2 classes / >=60 samples
    cls, cnt = np.unique(yr, return_counts=True)
    keep = set(cls[cnt >= 10])
    m = np.array([v in keep for v in yr])
    if m.sum() < 60 or len(keep) < 2:
        return None
    Xv, yr = Xv[m], yr[m]
    classes = sorted(set(yr))
    enc = {c: i for i, c in enumerate(classes)}
    y = np.array([enc[v] for v in yr], dtype=np.int64)
    # NaNs in expression -> column means
    if np.isnan(Xv).any():
        col = np.nanmean(Xv, axis=0)
        col = np.where(np.isnan(col), 0.0, col)
        Xv = np.where(np.isnan(Xv), col, Xv).astype(np.float32)
    return Xv, y, genes, classes


def load_pnet(cohort: str):
    from .pathway_data import load_cohort
    coh = load_cohort(cohort, channels="mut", root=DATA)     # single channel -> X is 2D
    X = coh.X
    if X.ndim == 3:
        X = X[..., 0]
    return X.astype(np.float32), coh.y, [str(g) for g in coh.genes], coh.classes


def _cap_genes(X, genes, cap: int):
    """Keep <=cap genes: top-variance among Reactome-covered genes first (so the
    graph carries signal), padded with top-variance uncovered genes. Applied
    identically to all three modes, so `none` sees the same X."""
    if X.shape[1] <= cap:
        return X, genes
    P = _reactome_membership(genes, REACTOME_CSV)
    covered = P.sum(1) > 0
    order = np.argsort(-X.var(0))
    sel = [i for i in order if covered[i]][:cap]
    if len(sel) < cap:
        sel += [i for i in order if not covered[i]][: cap - len(sel)]
    sel = np.array(sorted(sel))
    return X[:, sel], [genes[i] for i in sel]


# --------------------------------------------------------------------------- #
def _cfg(mode: str, K: int, seed: int, epochs: int, n_classes: int) -> RMTConfig:
    # curated/random use a FIXED external graph (gene_interaction=curated); learned
    # builds its own graph and needs no external prior (gene_interaction=none).
    fixed = mode in ("curated", "random")
    base = dict(
        heads=(HEAD,), n_hvg=None, batch_size=64, d_model=96, d_ff=192,
        n_markers=128, marker_mode="router", recursion_mode="expert",
        recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
        patience=12, lr=1e-3, weight_decay=1e-5, device="cuda",
        gene_interaction=("curated" if fixed else "none"),
    )
    cfg = RMTConfig(**base)
    if fixed:
        # the redesign fixes (all ON), matching bio_redesign.py
        cfg.bio_graph_prop = True
        cfg.bio_prop_lambda_init = 0.3
        cfg.bio_prop_hops = 1
        cfg.bio_prior_gate = True
        cfg.bio_prior_learnable = True
        cfg.bio_beta_init = 0.5
        cfg.bio_depth_laplacian = 0.01
        cfg.bio_centrality = "ppr"
        cfg.router_prior_anneal = False
    elif mode == "learned":
        # DATA-DRIVEN learned low-rank gene graph (no fixed prior, no external graph)
        cfg.bio_learned_graph = True
        cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2
        cfg.bio_prop_hops = 1
    elif mode == "learned_bio":
        # learned graph, IDENTICAL to `learned` except gene_embed is warm-started from
        # the curated Reactome graph (degenerate graphs fall back to random init).
        cfg.bio_learned_graph = True
        cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2
        cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio"
    elif mode in ("learned_fused", "learned_fused_rand"):
        # graph comes from BIOLOGY + LEARNING: curated Reactome interaction matrix (or a
        # random control) kept as a persistent, learnably-gated propagation term.
        cfg.bio_learned_graph = True
        cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2
        cfg.bio_prop_hops = 1
        cfg.bio_learned_init = "bio" if mode == "learned_fused" else "random"
        cfg.bio_learned_fuse = True
    # --- C1 confound factorial: isolate input SMOOTHING from depth ROUTING ---
    elif mode in ("smooth_curated", "smooth_random"):
        cfg.bio_graph_prop = True; cfg.bio_prop_lambda_init = 0.3; cfg.bio_prop_hops = 1
        cfg.bio_prior_gate = False; cfg.router_prior_beta = 0.0
        cfg.bio_depth_laplacian = 0.0; cfg.bio_centrality = "ppr"; cfg.router_prior_anneal = False
    elif mode in ("route_curated", "route_random"):
        cfg.bio_graph_prop = False; cfg.bio_prior_gate = True; cfg.bio_prior_learnable = True
        cfg.bio_beta_init = 0.5; cfg.bio_depth_laplacian = 0.0
        cfg.bio_centrality = "ppr"; cfg.router_prior_anneal = False
    return cfg


def run_cell(X, y, genes, classes, family, cohort, task, mode, K, seed, epochs,
             device, graphs):
    F, C = X.shape[1], len(classes)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, None, seed)
    cfg = _cfg(mode, K, seed, epochs, C)
    cfg.n_markers = min(cfg.n_markers, F)
    # curated/random install a fixed external graph; none/learned install nothing
    # (learned builds its own graph inside the model). learned_bio installs no fixed
    # prior but warm-starts the learned graph from the curated Reactome operator.
    inter = graphs[mode] if mode in ("curated", "random") else None
    if mode in ("smooth_curated", "route_curated"):
        inter = graphs["curated"]           # install graph; activation via cfg flags
    elif mode in ("smooth_random", "route_random"):
        inter = graphs["random"]
    bio_op = None
    if mode in ("learned_bio", "learned_fused"):
        bio_op = graphs["curated"].operator
    elif mode == "learned_fused_rand":
        bio_op = graphs["random"].operator
    yt, yp, model = _fit_eval(X.astype(np.float32), y, tr, va, te, cfg, F, C, device,
                              inter=inter, bio_op=bio_op)
    out = {
        "family": family, "cohort": cohort, "task": task, "mode": mode,
        "K": K, "seed": seed,
        "test_macro_f1": 100 * f1_score(yt, yp, average="macro"),
        "test_accuracy": 100 * accuracy_score(yt, yp),
        "n_features": F, "n_classes": C, "n_samples": int(len(y)),
    }
    if mode != "none":
        with torch.no_grad():
            out["learned_lambda"] = float(torch.sigmoid(model.bio_prop_logit))
            out["learned_beta"] = float(torch.nn.functional.softplus(model.bio_beta))
            if hasattr(model, "bio_fuse_gate"):
                out["fuse_gate"] = float(torch.sigmoid(model.bio_fuse_gate))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["tcga", "pnet"], required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--task", default="response",
                    help="TCGA label column; ignored for pnet (uses built-in response)")
    ap.add_argument("--modes", nargs="*", default=["none", "curated", "random"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cap_genes", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results_bio_curated")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.device)

    loaded = (load_tcga(args.cohort, args.task) if args.family == "tcga"
              else load_pnet(args.cohort))
    if loaded is None:
        raise SystemExit(f"[curated] SKIP {args.family}/{args.cohort}/{args.task} "
                         f"(missing / too few labelled samples)")
    X, y, genes, classes = loaded
    X, genes = _cap_genes(X, genes, args.cap_genes)
    task = args.task if args.family == "tcga" else "response"
    tag = f"{args.cohort}__{task}"
    out_dir = args.out / args.family / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[curated] {args.family}/{tag}  N={len(y)} G={X.shape[1]} C={len(classes)}",
          flush=True)

    summary = []
    for seed in args.seeds:
        # Build the curated + matched-random graphs ONCE per seed on the TRAIN split
        # (leakage-safe: labels only via train-fold PPR seeds), shared by all modes.
        tr, _, _ = _make_splits(y, None, seed)
        graphs, diag = build_reactome_falsification(
            X[tr], genes, REACTOME_CSV, y_train=y[tr], knn=16, seed=seed,
            centrality="ppr", deconfound_pc=0)
        if seed == args.seeds[0]:
            print(f"  [graph] {json.dumps(diag)}", flush=True)
        for mode in args.modes:
            p = out_dir / f"{mode}_s{seed}.json"
            if p.exists() and not args.force:
                summary.append(json.loads(p.read_text())); continue
            print(f"\n##### curated {tag} mode={mode} seed={seed} #####", flush=True)
            r = run_cell(X, y, genes, classes, args.family, args.cohort, task, mode,
                         args.K, seed, args.epochs, device, graphs)
            r["graph_diag"] = diag
            p.write_text(json.dumps(r, indent=1))
            lb = (f" lam={r.get('learned_lambda'):.3f} beta={r.get('learned_beta'):.3f}"
                  if "learned_lambda" in r else "")
            print(f"  [{mode} s{seed}] F1={r['test_macro_f1']:.2f} "
                  f"acc={r['test_accuracy']:.2f}{lb}", flush=True)
            summary.append(r)

    def _mean(m):
        v = [s["test_macro_f1"] for s in summary if s["mode"] == m]
        return sum(v) / len(v) if v else float("nan")
    mn, mc, mr = _mean("none"), _mean("curated"), _mean("random")
    print(f"\n==== {tag} CURATED summary (mean macro-F1) ====")
    print(f"  none={mn:.2f}  curated={mc:.2f}  random={mr:.2f}")
    print(f"  curated-none={mc-mn:+.2f}  curated-random={mc-mr:+.2f}  "
          f"(target: curated-random >= +2.0)", flush=True)


if __name__ == "__main__":
    main()
