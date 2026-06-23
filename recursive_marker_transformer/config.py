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
    marker_mode: str = "learnable"               # "learnable" | "random" | "variance"
    compress_mode: str = "aggregate"             # "aggregate": fold non-markers into clusters;
                                                 # "drop": use ONLY selected marker genes
    recursive_marker_refine: bool = True         # re-score markers after each recursion
    peak_init: bool = True                        # peaked router init (vs uniform-mush start)
    anneal_markers: bool = True                   # anneal selector temperature (vs constant hot)

    # ---- recursion -----------------------------------------------------
    recursion_depth: int = 4                     # K
    share_weights: bool = True                   # True: one block x K (RMT); False: K blocks
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
    gene_interaction: str = "none"               # "none" | "coexpr" | "random"
                                                 # coexpr = genomap correlation graph;
                                                 # random = degree-matched control
    interaction_knn: int = 16                    # k nearest co-expression neighbours
    router_prior_beta: float = 1.0               # beta_0: additive centrality-bias
                                                 # strength on the depth-router logits
    router_prior_anneal: bool = True             # decay beta_t -> 0 over training
                                                 # (warm-start prior; data takes over)

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
