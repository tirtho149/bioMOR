# ============================================================================
# bioMoR: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The bioMoR Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Configuration for the Recursive Marker Transformer (RMT).

A single dataclass holds every knob the model and training loop need. Every
ablation in the plan is reachable by overriding fields here (see ``ablate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Tuple


@dataclass
class RMTConfig:
    # ---- data ----------------------------------------------------------
    heads: Tuple[str, ...] = ("cancer_type",)   # genomic_dataloader phenotype heads
    cohorts: Optional[Tuple[str, ...]] = None    # None = all 4 (keeps cancer_type contiguous)
    n_hvg: Optional[int] = 4000                  # top-variance gene pre-filter; None = all ~20.5k
    n_channels: int = 1                          # per-gene input channels: 1=expression only,
                                                 # >1 = multimodal (expr / CNV / mutation), each
                                                 # gene-aligned; fused at the value projection
    batch_size: int = 64
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ---- model geometry ------------------------------------------------
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # ---- marker / compression ------------------------------------------
    n_markers: int = 1000                        # M; clamped to <= n_genes at build time
    marker_mode: str = "learnable"               # "learnable"|"random"|"variance"|
                                                 # "concrete"|"router"|"pathway".
                                                 # "pathway" = fixed Reactome gene->
                                                 # pathway tokens (M set by membership,
                                                 # n_markers ignored)
    pathway_pool: str = "mean"                   # pathway-token pooling: "mean"
                                                 # (dense assays, e.g. CNV/expr) |
                                                 # "sum" (burden; sparse binary mutation)
    compress_mode: str = "aggregate"             # "aggregate": fold non-markers into clusters;
                                                 # "drop": use ONLY selected marker genes
    recursive_marker_refine: bool = True         # re-score markers after each recursion
    peak_init: bool = True                        # peaked router init (vs uniform-mush start)
    anneal_markers: bool = True                   # anneal selector temperature (vs constant hot)

    # ---- recursion -----------------------------------------------------
    recursion_depth: int = 4                     # K
    share_weights: bool = True                   # True: one block x K (RMT); False: K blocks
    share_strategy: str = "cycle"                # block-tying scheme over the K steps
                                                 # (MoR Table 1): cycle|sequence|
                                                 # middle_cycle|middle_sequence
    n_unique_blocks: Optional[int] = None        # # distinct blocks; None -> 1 if
                                                 # share_weights else K (independent)
    step_cache: bool = False                     # reuse step-1 attention K/V across the
                                                 # K recursions (set-encoder analogue of
                                                 # MoR recursion-wise KV cache)
    adaptive_depth: bool = False                 # legacy soft-halting analogue (Ablation 5)
    marker_ffn: bool = False                     # dedicated FFN for markers (Ablation 7)

    # ---- Mixture-of-Recursions routing ---------------------------------
    recursion_mode: str = "fixed"               # "fixed" | "expert" | "token"
    router_capacity: Optional[Tuple[float, ...]] = None  # per-step funnel; None=taper to 0.5
    router_alpha: float = 1.0                    # router gate scale; 1.0 lets selected
                                                 # tokens take the FULL block update (0.1
                                                 # caps updates at 10% -> starves pooling)
    router_temp: float = 1.0                     # logits / temp before routing
    router_type: str = "linear"                 # "linear" | "mlp"
    router_z_coeff: float = 1e-3                 # weight of router z-loss
    router_balance_coeff: float = 0.1            # weight of token-choice balancing loss

    # ---- biology-informed router (genomap gene-gene interaction prior) -----
    gene_interaction: str = "none"               # "none"|"coexpr"|"random"|"reactome"
                                                 # coexpr = genomap correlation graph;
                                                 # random = degree-matched control;
                                                 # reactome = curated pathway-hierarchy
                                                 # centrality (per pathway token)
    interaction_knn: int = 16                    # k nearest co-expression neighbours
    router_prior_beta: float = 1.0               # beta_0: additive centrality-bias
                                                 # strength on the depth-router logits
    router_prior_anneal: bool = True             # decay beta_t -> 0 over training
                                                 # (warm-start prior; data takes over)

    # ---- BIO-ROUTER REDESIGN (see BIO_ROUTER_REDESIGN.txt) ----------------
    # Fix A: sample-conditional graph propagation on the input expression, using
    #        the co-expression GCN operator S = D^-1/2 (W+I) D^-1/2.
    bio_graph_prop: bool = False                 # smooth x along the gene graph pre-selection
    bio_prop_lambda_init: float = 0.3            # initial mix; lambda is LEARNABLE (sigmoid)
    bio_prop_hops: int = 1                       # # propagation hops (S^k x)
    # Fix B: gate the per-gene prior by the token state (FiLM) instead of a fixed bias.
    bio_prior_gate: bool = False
    # Fix C: de-confound technical axes + direct-interaction graph + seeded score.
    bio_deconfound_pc: int = 0                   # # top PCs to regress out of X (housekeeping)
    bio_precision: bool = False                  # partial-correlation graph vs marginal |corr|
    bio_centrality: str = "eigcent"              # "eigcent" | "ppr" (seeded personalized PageRank)
    # Fix D: persistent LEARNABLE prior strength beta (softplus), NOT annealed to 0.
    bio_prior_learnable: bool = False
    bio_beta_init: float = 0.5                   # init for the learnable beta (pre-softplus)
    # Fix E: graph-Laplacian depth-smoothness penalty (co-regulated genes share depth).
    bio_depth_laplacian: float = 0.0             # gamma; 0 disables the penalty
    # DATA-DRIVEN alternative to the fixed prior: a LEARNED low-rank gene-gene graph
    # (synthetic correlation) trained end-to-end by the task loss. Affinity
    # A = normalize(E) normalize(E)^T, propagated in low rank as
    # x <- (1-lam) x + lam (x E~)(E~^T). Fixes the fixed prior's three failure modes:
    # label-free, annealed-away, and not task-shaped.
    bio_learned_graph: bool = False              # enable the learned gene graph
    bio_learned_rank: int = 16                   # r: gene-embedding rank (graph is rank-r)
    bio_learned_init: str = "random"             # "random" | "bio": warm-start the learned
                                                 # gene embedding from the biological graph
                                                 # (co-expression / curated Reactome operator,
                                                 # top-r eigenvectors) instead of randn.
    bio_learned_fuse: bool = False               # keep the fixed biological interaction matrix
                                                 # as a PERSISTENT propagation term, learnably
                                                 # mixed (gate g) with the learned graph, so the
                                                 # graph comes from biology AND is learned.
    bio_fuse_source: str = "coexpr"              # which biological graph to fuse: "coexpr" |
                                                 # "random" (degree-matched control) | "reactome"
    # ANCHORED warm-start: keep the learned graph near the biological graph EARLY, then
    # let go. Adds an annealed Frobenius penalty ||A_learned - A_bio||^2 (computed in
    # low rank) so the bio warm-start is not immediately overwritten by the task gradient
    # -- the fix for "learned_bio matches random-init learned within noise". lambda decays
    # linearly to 0 over training; the bio init is also given a larger footprint than the
    # random baseline (bio_init_scale vs bio_init_rand). Degenerate (NaN) graphs disable
    # the anchor and fall back to the plain random-init learned graph.
    aggnet_species: str = "human"                # species for the aggregated STRING+KEGG+Reactome
                                                 # gene-gene network (gene_interaction="aggnet"):
                                                 # "human" (P-NET) | "mouse" (Tcell single-cell)
    bio_learned_anchor: bool = False             # enable the annealed bio-graph anchor penalty
    bio_anchor_lambda: float = 0.5               # lambda_0; annealed toward lambda_0*floor over epochs
    bio_anchor_floor: float = 0.0                # persistent fraction of lambda_0 kept at end of
                                                 # training (0 = fully release biology; >0 = keep a
                                                 # standing pull so a genuinely-useful bio graph
                                                 # survives to convergence)
    bio_anchor_rank: int = 16                    # rank of the stored bio target factor B (A_bio=BB^T)
    bio_anchor_source: str = "coexpr"            # biological graph to anchor to: "coexpr" | "reactome"
    bio_init_scale: float = 0.05                 # magnitude of the biological init component
    bio_init_rand: float = 0.005                 # magnitude of the matched random init component

    # ---- pathway-hierarchy attention bias (Reactome pathway->pathway graph) ----
    pathway_attn_bias: bool = False              # bias self-attention so a pathway
                                                 # token attends to its Reactome
                                                 # neighbours (needs full-token modes:
                                                 # recursion_mode in {fixed, token})
    pathway_attn_lambda: float = 2.0             # additive bias on attention logits
                                                 # for hierarchy-adjacent pathway pairs
    # SINGLE-CELL analogue of pathway_attn_bias: bias marker-token self-attention along the
    # LEARNED gene sub-graph over the selected markers (Eg[markers] Eg[markers]^T), so a marker
    # attends to its biological neighbours. Un-gates the attention-bias mechanism for
    # marker_mode!='pathway' (which has no provided Reactome graph). Requires bio_learned_graph.
    gene_attn_bias: bool = False
    gene_attn_lambda: float = 2.0                # additive bias magnitude on marker attention
    gene_attn_topk: int = 15                     # keep top-k biological neighbours per marker
    # ---- PATHWAY-SPACE learned graph (multi-omics analogue of the single-cell
    # gene learned graph). The interaction graph is the PROVIDED Reactome
    # pathway->pathway adjacency_matrix.csv -- NEVER a co-expression graph computed
    # on the sparse mut/CNV data. Warm-starts a learnable pathway embedding from the
    # provided adjacency's eigenmodes, propagates the pathway tokens along it, and
    # optionally fuses the fixed provided graph -- exactly like the gene path, but in
    # pathway-token space. Requires marker_mode='pathway'.
    pathway_learned_graph: bool = False
    pathway_learned_rank: int = 16               # rank r of the learned pathway graph
    pathway_prop_lambda_init: float = 0.2        # initial token-smoothing mix (learnable)
    pathway_learned_init: str = "bio"            # "bio": warm-start from adjacency_matrix.csv
                                                 # eigenmodes | "random"
    pathway_learned_fuse: bool = False           # also fuse the fixed provided adjacency
    # MONOTONE-SAFE MERGE (exploratory): inject the embedding-site pathway biology as a
    # LayerScale-style ZERO-INIT residual (cluster += gamma * prop, gamma init 0) instead of
    # the convex-mix (1-lam)cluster+lam*prop that REPLACES 20% of the clean signal at init
    # (lam init 0.2) and corrupts cohorts where smoothing hurts (e.g. prostate 73.6->56.2).
    # With gamma=0 at init, "both" == router-only exactly and biology can only ADD -> the
    # merged embedding+router is monotone-safe and can win-or-tie, never collapse.
    pathway_prop_residual: bool = False
    # L2 penalty pulling the embedding-residual scale gamma toward 0, so the embedding site
    # stays OFF unless it earns generalization benefit (prevents gamma drifting into harm on
    # cohorts where smoothing hurts, e.g. prostate). 0 = off; ~1-5 enforces monotone-safety.
    pathway_prop_gamma_reg: float = 0.0
    # COMPLEMENTARY embedding residual (exploratory): instead of pure low-pass smoothing
    # (which over-smooths and hurts, e.g. prostate), inject a learnable ZERO-INIT residual
    # over [neighbour-mean, high-freq contrast] = cluster += MLP([prop, cluster-prop]) with
    # the output layer zero-initialised. Router adjusts DEPTH; this adds a complementary
    # biological feature on token VALUES -> 'both' can strictly beat router, not just tie.
    pathway_prop_complement: bool = False
    # USE THE PROVIDED ADJACENCY DIRECTLY: drive BOTH the router and the embedding residual
    # from the fixed GCN-normalised Reactome operator (adjacency_matrix.csv) instead of the
    # learnable low-rank cosine graph. Removes the shared-parameter drift that let the
    # embedding objective corrupt the router's graph (the prostate coupling). Combine with
    # --pathway_learned_graph (to build/install the operator) + --pathway_prop_residual.
    pathway_fixed_graph: bool = False
    # REDESIGNED bio-router: zero-init graph-conv residual on the depth-router logits so
    # routing depth can depend on a token's biological neighbourhood (learned, bounded,
    # starts as a no-op). Replaces the harmful static centrality prior as router-site biology.
    bio_graph_router: bool = False
    # Decouple the two injection SITES that share one learned graph: propagation smooths the
    # token EMBEDDING; bio_graph_router feeds the ROUTER. Turn propagation off (keep the
    # warm-started graph for the router only) to get a clean router-only condition.
    bio_learned_prop: bool = True                # gene learned-graph input smoothing on/off
    pathway_learned_prop: bool = True            # pathway learned-graph token smoothing on/off

    # ---- loss weights --------------------------------------------------
    lambda_marker: float = 0.1
    gamma_diversity: float = 0.05
    beta_compression: float = 0.01

    # ---- optimisation --------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 30
    patience: int = 8                            # early-stop on val macro-F1
    seed: int = 42
    device: str = "auto"                         # "auto" | "cpu" | "cuda" | "mps"

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_overrides(cls, **kw) -> "RMTConfig":
        """Build a config, coercing string overrides to the field's type."""
        valid = {f.name: f for f in fields(cls)}
        clean = {}
        for k, v in kw.items():
            if k not in valid:
                raise ValueError(f"Unknown config field: {k}")
            clean[k] = _coerce(v, valid[k])
        return cls(**clean)


_TRUE = {"1", "true", "yes", "y", "t"}


def _coerce(v, f):
    """Coerce a CLI string to the dataclass field's declared type."""
    if not isinstance(v, str):
        return v
    ann = str(f.type)
    if v.lower() in {"none", "null"}:
        return None
    if "bool" in ann:
        return v.lower() in _TRUE
    if "Tuple" in ann:                       # check before scalar int/float
        parts = [s for s in v.replace(" ", "").split(",") if s]
        if "float" in ann:
            return tuple(float(s) for s in parts)
        if "int" in ann:
            return tuple(int(s) for s in parts)
        return tuple(parts)
    if "int" in ann and "Optional" not in ann:
        return int(v)
    if "Optional[int]" in ann or ("int" in ann and "Optional" in ann):
        return int(v)
    if "float" in ann:
        return float(v)
    return v
