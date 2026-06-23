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

"""Stage 4: parameter-shared recursive transformer.

Parameter efficiency is architectural here: instead of K independent layers we
instantiate **one** transformer block and apply it K times. This mirrors the
weight-sharing mechanism in ``mixture_of_recursions``
(``model/sharing_strategy/llama.py``), but for a small tabular classifier the
explicit "call one module K times" loop (WSRT-style) is exactly equivalent in
parameter count and far simpler than tensor-aliasing across a ModuleList.

Set ``share_weights=False`` to get K independent blocks -- the Ablation-3
baseline that isolates the parameter saving.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .router import ExpertChoiceRouter, TokenChoiceRouter


class SharedTransformerBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN. Optionally a separate FFN that
    only marker tokens receive (Ablation 7)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 marker_ffn: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.marker_ffn = None
        if marker_ffn:
            self.norm3 = nn.LayerNorm(d_model)
            self.marker_ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        if self.marker_ffn is not None:
            x = x + self.drop(self.marker_ffn(self.norm3(x)))
        return x


class RecursiveStack(nn.Module):
    """Apply a transformer block ``depth`` times, optionally with MoR routing.

    ``refine_fn`` (optional) is called between iterations with the current
    tokens and must return a per-token gate (B, M); tokens are multiplied by it,
    realising the recursive marker-refinement feedback loop.

    ``recursion_mode`` selects how depth is allocated across the M marker tokens:

    * ``"fixed"``   -- every token gets all ``K`` iterations (the original
      weight-shared recursion). ``adaptive_depth`` keeps the legacy soft-halting
      analogue for the old Ablation-5 baseline.
    * ``"expert"``  -- MoR expert-choice: a capacity funnel keeps a top-k of
      tokens at each step (``router.py:ExpertChoiceRouter``).
    * ``"token"``   -- MoR token-choice: each token picks one depth up front
      (``router.py:TokenChoiceRouter``).

    ``forward`` returns ``(tokens, route_info)`` where ``route_info`` carries the
    per-token recursion depth (an intrinsic importance signal) and the raw,
    unweighted router losses (z-loss, load-balancing) for the objective.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 depth: int, share_weights: bool = True,
                 adaptive_depth: bool = False, marker_ffn: bool = False,
                 recursion_mode: str = "fixed",
                 router_capacity: Optional[Tuple[float, ...]] = None,
                 router_alpha: float = 0.1, router_temp: float = 1.0,
                 router_type: str = "linear"):
        super().__init__()
        self.depth = depth
        self.share_weights = share_weights
        self.adaptive_depth = adaptive_depth
        self.recursion_mode = recursion_mode
        if share_weights:
            self.block = SharedTransformerBlock(d_model, n_heads, d_ff, dropout, marker_ffn)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                SharedTransformerBlock(d_model, n_heads, d_ff, dropout, marker_ffn)
                for _ in range(depth)
            ])
        self.halt = nn.Linear(d_model, 1) if adaptive_depth else None

        # MoR router operates over the M marker tokens, reusing the shared block.
        self.router = None
        if recursion_mode == "expert":
            self.router = ExpertChoiceRouter(
                d_model, depth, router_capacity or (), alpha=router_alpha,
                temp=router_temp, router_type=router_type)
        elif recursion_mode == "token":
            self.router = TokenChoiceRouter(
                d_model, depth, alpha=router_alpha, temp=router_temp,
                router_type=router_type)
        elif recursion_mode != "fixed":
            raise ValueError(f"Unknown recursion_mode: {recursion_mode!r}")

    def _block(self, t: int) -> SharedTransformerBlock:
        return self.block if self.share_weights else self.blocks[t]

    def forward(
        self,
        tokens: torch.Tensor,                               # (B, M, d)
        refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        prior: Optional[torch.Tensor] = None,               # (M,) biological prior
        prior_weight: float = 0.0,                          # beta_t (annealed)
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.router is not None:
            # Pass the step-indexed block picker: _block(t) is the shared block
            # when share_weights, else the t-th independent block -- so routing
            # composes with both weight-sharing and the independent ablation.
            return self.router(tokens, lambda t, x: self._block(t)(x), refine_fn,
                               prior=prior, prior_weight=prior_weight)

        remaining = torch.ones(tokens.shape[:2], device=tokens.device)  # (B, M)
        for t in range(self.depth):
            updated = self._block(t)(tokens)
            if self.adaptive_depth:
                # Soft per-token halting: spend less update on "done" tokens.
                halt = torch.sigmoid(self.halt(tokens)).squeeze(-1)     # (B, M)
                step = remaining * (1.0 - halt)
                tokens = tokens + step.unsqueeze(-1) * (updated - tokens)
                remaining = remaining * halt
            else:
                tokens = updated
            if refine_fn is not None and t < self.depth - 1:
                gate = refine_fn(tokens)                                # (B, M)
                tokens = tokens * gate.unsqueeze(-1)
        info: Dict[str, object] = {
            "depth_per_token": torch.full(tokens.shape[:2], float(self.depth),
                                          device=tokens.device),
            "z_loss": tokens.new_zeros(()),
            "balance_loss": tokens.new_zeros(()),
            "capacity": None,
        }
        return tokens, info
