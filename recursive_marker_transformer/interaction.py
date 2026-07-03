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

"""Genomap gene-gene interaction graph -> biological prior for the MoR router.

genomap (Islam & Xing 2023) identifies gene-gene *interactions* as the pairwise
correlation between genes across samples -- ``createInteractionMatrix`` in
``genomap/genomap/genomap.py`` is literally
``sklearn.pairwise_distances(data.T, metric='correlation')`` -- and feeds that
interaction matrix to optimal transport. We reuse only that interaction-
identification step.

From the training split we build the gene-gene co-expression graph (|Pearson
correlation|, sparsified to each gene's top-k neighbours, symmetrised), then read a
**network-centrality prior** off it:

    pi = z-score( eigenvector_centrality(W) )

Hub genes -- those central in the co-expression network -- receive a larger prior.
This prior is injected as an *annealed additive bias* on the expert-choice depth
router (``router.py``):

    r_tilde_m = (w_r^T h_m) / tau   +   beta_t * pi_m

so a gene that is a co-expression hub is nudged to survive deeper into the
recursion funnel early in training, with ``beta_t`` decaying to 0 so the data-driven
term takes over (the same warm-start logic as the marker-router temperature anneal).

Crucially the graph is built from **expression alone, with no labels**, so the prior
injects biological network structure without leaking the cohort labels -- unlike a
curated-marker prior. Two controls make the ablation honest:

* ``mode="coexpr"`` -- the real genomap correlation graph (the proposed component).
* ``mode="random"`` -- a degree-matched random graph (same sparsity, shuffled
  edges); if co-expression structure matters, ``coexpr`` must beat ``random``.
* ``mode="none"``   -- no prior (the original SMART router).

Correlation is accumulated by streaming sufficient statistics over the loader, so no
full expression matrix is materialised -- only the ``N x N`` accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class Interaction:
    centrality: torch.Tensor              # (N,) z-scored network-centrality prior
    operator: Optional[torch.Tensor]      # (N, N) GCN-normalised propagation op (or None)
    mode: str
    laplacian: Optional[torch.Tensor] = None   # (N, N) scaled combinatorial Laplacian (or None)


def collect_X(loader) -> np.ndarray:
    """Materialise a loader's feature matrix (n_samples, n_genes) as float32.
    Used to compute the gene-gene interaction matrix; the train splits here are
    small enough (a few thousand samples) that this is cheap."""
    chunks = [xb.numpy() if hasattr(xb, "numpy") else np.asarray(xb)
              for xb, _yb in loader]
    return np.concatenate(chunks, axis=0).astype(np.float32)


# Path to the bundled genomap source (the package's own genomap.py holds
# createInteractionMatrix). We load that function directly so the interaction
# matrix comes from genomap itself, not a reimplementation.
_GENOMAP_SRC = Path(__file__).resolve().parents[1] / "genomap" / "genomap" / "genomap.py"


def _load_genomap_create_interaction():
    """Return genomap's own ``createInteractionMatrix`` function.

    genomap's ``genomap.py`` does ``from genomap.genomapOPT import ...`` at module
    top, which needs POT (``ot``) that we do not have and do not need just to build
    the interaction matrix. We therefore stub that optimal-transport module in
    ``sys.modules`` and exec genomap's source file, so the *actual* genomap
    function is loaded without dragging in the OT stack."""
    import importlib.util
    import sys
    import types
    for name, attrs in [("genomap", []),
                        ("genomap.genomapOPT",
                         ["create_space_distributions", "gromov_wasserstein_adjusted_norm"])]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            for a in attrs:
                setattr(m, a, None)            # unused by createInteractionMatrix
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("genomap._genomap_src", _GENOMAP_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.createInteractionMatrix


def genomap_interaction(X: np.ndarray) -> np.ndarray:
    """(n_samples, n_genes) -> (N, N) genomap gene-gene interaction matrix, i.e.
    the correlation *distance* ``createInteractionMatrix(X, metric='correlation')``
    from genomap (Islam & Xing 2023). If genomap's source cannot be loaded we fall
    back to the identical ``sklearn.pairwise_distances`` call genomap itself uses."""
    try:
        create = _load_genomap_create_interaction()
        return np.asarray(create(X, metric="correlation"), dtype=np.float32)
    except Exception as e:                     # robust fallback = genomap's own one-liner
        import sklearn.metrics as mpd
        print(f"[interaction] genomap source load failed ({e}); using identical "
              f"sklearn.pairwise_distances(metric='correlation')", flush=True)
        return mpd.pairwise_distances(X.T, metric="correlation").astype(np.float32)


def _sparse_affinity(aff: Optional[np.ndarray], n_genes: int, mode: str,
                     knn: int, seed: int) -> np.ndarray:
    """Symmetric gene-gene affinity W (N, N), top-k per row, no self-loops.
    ``aff`` is the (nonnegative) interaction-strength matrix for ``coexpr``."""
    k = min(knn, n_genes - 1)
    if mode == "random":
        rng = np.random.default_rng(seed)
        W = np.zeros((n_genes, n_genes), dtype=np.float32)
        for i in range(n_genes):
            j = rng.choice(n_genes, size=k, replace=False)
            W[i, j] = 1.0
    elif mode == "coexpr":
        if aff is None:
            raise ValueError("coexpr mode needs the interaction affinity matrix")
        aff = np.array(aff, dtype=np.float32, copy=True)
        np.fill_diagonal(aff, 0.0)
        W = np.zeros_like(aff)
        idx = np.argpartition(-aff, kth=k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n_genes), idx.shape[1])
        W[rows, idx.ravel()] = aff[rows, idx.ravel()]
    else:
        raise ValueError(f"Unknown gene_interaction mode: {mode!r}")
    return np.maximum(W, W.T)                                   # undirected


def _eigenvector_centrality(W: np.ndarray, iters: int = 200) -> np.ndarray:
    """Leading eigenvector (power iteration) of the affinity -> z-scored prior."""
    n = W.shape[0]
    v = np.ones(n, dtype=np.float64) / np.sqrt(n)
    Wd = W.astype(np.float64)
    for _ in range(iters):
        v_new = Wd @ v
        nrm = np.linalg.norm(v_new)
        if nrm < 1e-12:
            break
        v = v_new / nrm
    c = np.abs(v)
    c = (c - c.mean()) / (c.std() + 1e-8)                       # z-score
    return c.astype(np.float32)


def _gcn_operator(W: np.ndarray) -> np.ndarray:
    """Self-loops + symmetric normalisation D^{-1/2}(W+I)D^{-1/2} (for smoothing)."""
    A = W.copy()
    np.fill_diagonal(A, 1.0)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-8, None))
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# BIO-ROUTER REDESIGN helpers (see BIO_ROUTER_REDESIGN.txt Fix C / A / E).
# ---------------------------------------------------------------------------
def _deconfound(X: np.ndarray, n_pc: int) -> np.ndarray:
    """Regress out the top-`n_pc` principal components (library-size / housekeeping
    technical axes) from the gene-by-sample matrix, so the co-expression graph is
    not dominated by the depth/ribosomal hairball (Fix C1)."""
    if n_pc <= 0:
        return X
    Xc = X - X.mean(0, keepdims=True)
    # top PCs of the sample space via SVD on the centred matrix.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_pc, Vt.shape[0])
    proj = (U[:, :k] * S[:k]) @ Vt[:k]        # rank-k technical reconstruction
    return (Xc - proj).astype(np.float32)


def _precision_affinity(X: np.ndarray, eps: float = 0.15) -> np.ndarray:
    """Partial-correlation (direct-interaction) affinity |p_ij| from the shrunk
    precision matrix (Fix C2). Marginal |corr| links everything to everything; the
    precision matrix keeps edges that survive conditioning on all other genes."""
    R = np.corrcoef(X.T).astype(np.float64)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    Rs = (1.0 - eps) * R + eps * np.eye(R.shape[0])         # ridge shrinkage -> invertible
    Theta = np.linalg.inv(Rs)
    d = np.sqrt(np.clip(np.diag(Theta), 1e-8, None))
    P = -Theta / np.outer(d, d)                            # partial correlations
    aff = np.abs(P).astype(np.float32)
    np.fill_diagonal(aff, 0.0)
    return aff


def _seeded_ppr(W: np.ndarray, seeds: np.ndarray, alpha: float = 0.85,
                iters: int = 100) -> np.ndarray:
    """Personalized PageRank restarted at `seeds` (label-free high-variance genes),
    so 'importance' points at task-relevant modules instead of housekeeping hubs
    (Fix C3). Power iteration on the row-stochastic transition, z-scored."""
    n = W.shape[0]
    deg = W.sum(1, keepdims=True)
    Pt = (W / np.clip(deg, 1e-8, None)).T.astype(np.float64)   # transition, walker on columns
    s = np.zeros(n, dtype=np.float64)
    s[seeds] = 1.0 / max(1, len(seeds))
    pi = s.copy()
    for _ in range(iters):
        pi = alpha * (Pt @ pi) + (1.0 - alpha) * s
    pi = pi / (pi.sum() + 1e-12)
    c = (pi - pi.mean()) / (pi.std() + 1e-8)
    return c.astype(np.float32)


def _scaled_laplacian(W: np.ndarray) -> np.ndarray:
    """Combinatorial Laplacian L = diag(deg) - W, scaled by mean degree so the
    depth-smoothness penalty d^T L d = sum_ij W_ij (d_i-d_j)^2 is O(1) (Fix E)."""
    deg = W.sum(1)
    L = np.diag(deg) - W
    return (L / (deg.mean() + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# CURATED-NETWORK falsification (BIO_ROUTER_REDESIGN.txt, symbol-bearing cohorts).
#
# The co-expression |corr| graph provably TIES its degree-matched shuffle (FACT B):
# a shuffle of a covariation graph reproduces the same global statistics. A CURATED
# network -- gene pairs that co-occur in a Reactome pathway -- encodes literature
# edges a degree-preserving shuffle cannot recover, so `curated` vs `random` has a
# mechanism to separate. Requires gene SYMBOLS (single-cell sets are anonymised);
# runs on the P-NET / TCGA cohorts, the low-signal stage/T/N regime where a real
# inductive bias is most defensible.
# ---------------------------------------------------------------------------
def _reactome_membership(gene_symbols: list, pathways_csv, gene_col: str = "Genes"
                         ) -> np.ndarray:
    """(G, M) binary gene->Reactome-pathway membership for `gene_symbols`, read from
    a ``filtered_pathways.csv`` (Pathway_ID, Pathway_Name, Genes[comma-separated])."""
    import pandas as pd
    from .pathway_data import fix_symbol
    pw = pd.read_csv(pathways_csv)
    gidx = {g: i for i, g in enumerate(gene_symbols)}
    G, M = len(gene_symbols), len(pw)
    P = np.zeros((G, M), dtype=np.float32)
    for j, gene_str in enumerate(pw[gene_col].fillna("")):
        for g in (x.strip() for x in str(gene_str).split(",")):
            g = fix_symbol(g)
            i = gidx.get(g)
            if i is not None:
                P[i, j] = 1.0
    return P


def _label_aware_seeds(X: np.ndarray, y: np.ndarray, n_seeds: int) -> np.ndarray:
    """Top between-class-variance genes (ANOVA-F statistic), TRAIN FOLD ONLY so it is
    leakage-safe (baselines use train labels the same way). Replaces the label-free
    top-variance PPR seeds, which in bulk/scRNA track library size / housekeeping --
    exactly the non-discriminative hubs FM-3 warns about. Points 'importance' at the
    discriminative modules instead."""
    X = np.asarray(X, dtype=np.float64)
    classes = np.unique(y)
    n, g = X.shape
    grand = X.mean(0)
    bss = np.zeros(g); wss = np.zeros(g)
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(0)
        bss += len(Xc) * (mc - grand) ** 2
        wss += ((Xc - mc) ** 2).sum(0)
    f = (bss / max(len(classes) - 1, 1)) / (wss / max(n - len(classes), 1) + 1e-8)
    return np.argsort(-f)[: max(1, n_seeds)]


def build_reactome_falsification(X_train: np.ndarray, gene_symbols: list, pathways_csv,
                                 y_train: Optional[np.ndarray] = None, knn: int = 16,
                                 seed: int = 42, centrality: str = "ppr",
                                 deconfound_pc: int = 0):
    """Build the {none, curated, random} Interaction set for the falsification test.

    * ``curated`` -- Reactome gene-gene co-membership graph over ``gene_symbols``
      (W = P Pᵀ, top-``knn`` per row); genes in no pathway are isolated nodes.
    * ``random``  -- the SAME graph under a random gene RELABELLING (node-identity
      permutation). Its degree sequence, edge-weight multiset and spectrum are
      IDENTICAL to ``curated`` by construction; it differs ONLY in which genes the
      real edges connect. This is the exact FACT-B control -- any ``curated`` >
      ``random`` gap is attributable to real biological edge identity, nothing else.
    * ``none``    -- no graph.

    Returns ``(dict[mode -> Interaction | None], diagnostics dict)``.
    """
    G = len(gene_symbols)
    P = _reactome_membership(gene_symbols, pathways_csv)         # (G, M)
    W = (P @ P.T).astype(np.float32)                            # co-membership counts
    np.fill_diagonal(W, 0.0)
    W = _sparse_affinity(W, G, "coexpr", knn, seed)            # top-k sparsify (weighted)

    Xc = np.asarray(X_train, dtype=np.float32)
    if deconfound_pc:
        Xc = _deconfound(Xc, deconfound_pc)                    # Fix C1 (housekeeping PCs)
    if centrality == "ppr":
        n_seed = max(5, G // 20)
        seeds = (_label_aware_seeds(Xc, y_train, n_seed) if y_train is not None
                 else np.argsort(-Xc.var(0))[:n_seed])
        pi = _seeded_ppr(W, seeds)
    else:
        pi = _eigenvector_centrality(W)
    op = _gcn_operator(W)
    lap = _scaled_laplacian(W)

    curated = Interaction(
        centrality=torch.from_numpy(pi.astype(np.float32)),
        operator=torch.from_numpy(op.astype(np.float32)), mode="curated",
        laplacian=torch.from_numpy(lap.astype(np.float32)))

    # Degree/weight/spectrum-matched control: permute node identities. random node i
    # inherits curated node perm[i]'s connectivity and centrality, so gene i is mixed
    # with biologically UNRELATED genes while every graph statistic is preserved.
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(G)
    rand = Interaction(
        centrality=torch.from_numpy(pi[perm].astype(np.float32)),
        operator=torch.from_numpy(op[np.ix_(perm, perm)].astype(np.float32)), mode="random",
        laplacian=torch.from_numpy(lap[np.ix_(perm, perm)].astype(np.float32)))

    deg = (W > 0).sum(1)
    covered = P.sum(1) > 0
    diag = {
        "n_genes": int(G),
        "n_genes_in_reactome": int(covered.sum()),
        "reactome_coverage_frac": float(covered.mean()),
        "mean_degree": float(deg.mean()),
        "isolated_frac": float((deg == 0).mean()),
        # FM-3 confound check: does centrality track the housekeeping/library axis?
        "corr_pi_pc1": _corr_pi_pc1(pi, np.asarray(X_train, dtype=np.float32)),
    }
    return {"none": None, "curated": curated, "random": rand}, diag


def _corr_pi_pc1(pi: np.ndarray, X: np.ndarray) -> float:
    """|corr(centrality, per-gene loading on the top expression PC)| -- the FM-3
    diagnostic. ~0 means centrality is NOT just the housekeeping/library-size axis."""
    Xz = X - X.mean(0, keepdims=True)
    # top right singular vector loading per gene (sign-free)
    try:
        _, _, vt = np.linalg.svd(Xz, full_matrices=False)
        load = np.abs(vt[0])
    except np.linalg.LinAlgError:
        return float("nan")
    a = pi - pi.mean(); b = load - load.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(abs((a @ b) / denom))


def build_interaction_v2(source, n_genes: int, mode: str = "coexpr", knn: int = 16,
                         seed: int = 42, deconfound_pc: int = 0, precision: bool = False,
                         centrality: str = "eigcent") -> Optional[Interaction]:
    """Redesigned prior (BIO_ROUTER_REDESIGN.txt): returns the centrality prior AND
    the propagation operator S (Fix A) AND the graph Laplacian L (Fix E) for the
    requested mode. `coexpr`/`random` build the real / degree-matched graph so the
    falsification test runs the SAME mechanism on both; `precision`+`ppr`+`deconfound`
    turn on the Fix-C improvements to the statistic itself."""
    if mode in (None, "none"):
        return None
    aff = None
    if mode == "coexpr":
        if source is None:
            raise ValueError("coexpr mode needs the train feature matrix")
        X = source if isinstance(source, np.ndarray) else (
            source.numpy() if hasattr(source, "numpy") else collect_X(source))
        X = np.asarray(X, dtype=np.float32)
        X = _deconfound(X, deconfound_pc)                  # Fix C1
        if precision:                                      # Fix C2
            aff = _precision_affinity(X)
        else:
            d = genomap_interaction(X)
            aff = np.abs(1.0 - d).astype(np.float32)
    W = _sparse_affinity(aff, n_genes, mode, knn, seed)
    if centrality == "ppr" and mode == "coexpr":           # Fix C3 (needs a real graph)
        var = np.asarray(X).var(0)
        seeds = np.argsort(-var)[: max(5, n_genes // 20)]  # top-variance genes as label-free seeds
        pi = torch.from_numpy(_seeded_ppr(W, seeds))
    else:
        pi = torch.from_numpy(_eigenvector_centrality(W))
    op = torch.from_numpy(_gcn_operator(W))                # Fix A operator (always built)
    lap = torch.from_numpy(_scaled_laplacian(W))           # Fix E Laplacian (always built)
    return Interaction(centrality=pi, operator=op, mode=mode, laplacian=lap)


def build_interaction(source, n_genes: int, mode: str = "coexpr", knn: int = 16,
                      seed: int = 42, want_operator: bool = False) -> Optional[Interaction]:
    """Build the centrality prior (and optionally the propagation operator) for the
    requested mode, or ``None`` when disabled.

    ``source`` is the train feature matrix (ndarray/tensor) or a DataLoader yielding
    it; it is only consulted for the real correlation graph (``coexpr``)."""
    if mode in (None, "none"):              # CLI may coerce "none" -> None
        return None
    aff = None
    if mode == "coexpr":
        if source is None:
            raise ValueError("coexpr mode needs the train feature matrix")
        X = source if isinstance(source, np.ndarray) else (
            source.numpy() if hasattr(source, "numpy") else collect_X(source))
        # genomap interaction matrix = correlation DISTANCE (1 - corr); the
        # interaction *strength* (edge weight) is |corr| = |1 - distance|.
        d = genomap_interaction(np.asarray(X, dtype=np.float32))
        aff = np.abs(1.0 - d).astype(np.float32)
    W = _sparse_affinity(aff, n_genes, mode, knn, seed)
    pi = torch.from_numpy(_eigenvector_centrality(W))
    op = torch.from_numpy(_gcn_operator(W)) if want_operator else None
    return Interaction(centrality=pi, operator=op, mode=mode)
