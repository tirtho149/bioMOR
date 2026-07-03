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

    def forward(self, x: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        # attn_bias (M, M): additive bias on the attention logits so each pathway
        # token attends preferentially to its Reactome-hierarchy neighbours. None =
        # ordinary full self-attention. Broadcasts over batch and heads.
        # kv (B, M, d): step-cache -- use these cached (step-0) states as the
        # key/value source instead of the current tokens, so attention is not
        # recomputed against the evolving hidden states (KV-reuse analogue). None =
        # ordinary self-attention where query=key=value=h.
        if kv is None:
            a, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_bias)
        else:
            hk = self.norm1(kv)
            a, _ = self.attn(h, hk, hk, need_weights=False, attn_mask=attn_bias)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        if self.marker_ffn is not None:
            x = x + self.drop(self.marker_ffn(self.norm3(x)))
        return x


def block_assignment(strategy: str, depth: int, n_unique: int) -> list:
    """Map each recursion step t in [0, depth) to a block index in [0, n_unique),
    realising MoR's parameter-sharing schemes (Table 1) for the recursive stack:

      * ``cycle``           -- t mod n_unique          (n_unique=1 => fully shared RMT)
      * ``sequence``        -- consecutive repeats of each unique block
      * ``middle_cycle``    -- first & last steps get their own block, interior steps
                               cycle the shared middle blocks (MoR's safest scheme)
      * ``middle_sequence`` -- first & last unique, interior in sequence

    n_unique is clamped to [1, depth]; ``cycle`` with n_unique=depth == independent.
    """
    n_unique = max(1, min(n_unique, depth))
    if strategy == "cycle":
        return [t % n_unique for t in range(depth)]
    if strategy == "sequence":
        per = -(-depth // n_unique)                         # ceil(depth/n_unique)
        return [min(t // per, n_unique - 1) for t in range(depth)]
    if strategy in ("middle_cycle", "middle_sequence"):
        if n_unique < 3 or depth < 3:                       # not enough to reserve ends
            return [t % n_unique for t in range(depth)]
        mids = n_unique - 2                                 # interior unique blocks
        assign = [0] * depth
        assign[-1] = n_unique - 1
        interior = list(range(1, depth - 1))
        if strategy == "middle_cycle":
            for j, t in enumerate(interior):
                assign[t] = 1 + (j % mids)
        else:                                               # middle_sequence
            per = -(-len(interior) // mids)
            for j, t in enumerate(interior):
                assign[t] = 1 + min(j // per, mids - 1)
        return assign
    raise ValueError(f"Unknown share_strategy: {strategy!r}")


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
                 router_type: str = "linear",
                 share_strategy: str = "cycle", n_unique_blocks: Optional[int] = None,
                 step_cache: bool = False, bio_prior_gate: bool = False):
        super().__init__()
        self.depth = depth
        self.share_weights = share_weights
        self.adaptive_depth = adaptive_depth
        self.recursion_mode = recursion_mode
        self.step_cache = step_cache
        # Parameter-sharing scheme: n_unique distinct blocks, assigned to the K steps
        # by `share_strategy`. Defaults reproduce the old behaviour exactly: shared =>
        # 1 block (cycle), not-shared => K blocks (independent).
        n_unique = n_unique_blocks if n_unique_blocks is not None else (1 if share_weights else depth)
        self.assign = block_assignment(share_strategy, depth, n_unique)
        n_blocks = max(self.assign) + 1
        self.blocks = nn.ModuleList([
            SharedTransformerBlock(d_model, n_heads, d_ff, dropout, marker_ffn)
            for _ in range(n_blocks)
        ])
        self.block = self.blocks[0] if n_blocks == 1 else None
        self.halt = nn.Linear(d_model, 1) if adaptive_depth else None

        # MoR router operates over the M marker tokens, reusing the shared block.
        self.router = None
        if recursion_mode == "expert":
            self.router = ExpertChoiceRouter(
                d_model, depth, router_capacity or (), alpha=router_alpha,
                temp=router_temp, router_type=router_type, prior_gate=bio_prior_gate)
        elif recursion_mode == "token":
            self.router = TokenChoiceRouter(
                d_model, depth, alpha=router_alpha, temp=router_temp,
                router_type=router_type)
        elif recursion_mode != "fixed":
            raise ValueError(f"Unknown recursion_mode: {recursion_mode!r}")

    def _block(self, t: int) -> SharedTransformerBlock:
        return self.blocks[self.assign[t]]

    def forward(
        self,
        tokens: torch.Tensor,                               # (B, M, d)
        refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        prior: Optional[torch.Tensor] = None,               # (M,) biological prior
        prior_weight: float = 0.0,                          # beta_t (annealed)
        attn_bias: Optional[torch.Tensor] = None,           # (M,M) hierarchy bias
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.router is not None:
            # Pass the step-indexed block picker: _block(t) is the shared block
            # when share_weights, else the t-th independent block -- so routing
            # composes with both weight-sharing and the independent ablation.
            # The (M,M) attention bias is only valid when the block sees the FULL
            # token set (token-choice / fixed). Expert-choice gathers a per-step
            # top-k subset, so a fixed (M,M) mask cannot apply -- drop it there.
            if attn_bias is not None and self.recursion_mode != "token":
                raise ValueError("pathway_attn_bias needs recursion_mode in "
                                 "{token, fixed}; expert-choice gathers token "
                                 "subsets incompatible with a fixed (M,M) mask")
            if self.step_cache and self.recursion_mode != "token":
                raise ValueError("step_cache needs recursion_mode in {token, fixed}; "
                                 "expert-choice gathers per-step token subsets")
            kv0 = tokens if self.step_cache else None
            block = (lambda t, x: self._block(t)(x, attn_bias, kv0))
            return self.router(tokens, block, refine_fn,
                               prior=prior, prior_weight=prior_weight)

        remaining = torch.ones(tokens.shape[:2], device=tokens.device)  # (B, M)
        kv0 = tokens if self.step_cache else None                       # step-0 cache
        for t in range(self.depth):
            updated = self._block(t)(tokens, attn_bias, kv0)
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
