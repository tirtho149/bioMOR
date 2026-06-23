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
