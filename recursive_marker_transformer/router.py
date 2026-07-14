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

"""Mixture-of-Recursions (MoR) routing for the marker token stream.

This is the core of the adaptive-computation half of the model. Instead of
running the shared transformer block ``K`` times over *every* marker token
(fixed depth), a **router** assigns each token its own recursion depth, exactly
as in Bae et al. 2025 (arXiv:2507.10524). Because genes are a non-causal *set*
(not an autoregressive sequence) we drop MoR's causal-order machinery
(``torch.sort`` over selected positions, the causal auxiliary predictor) and the
KV-cache sharing -- none of it applies to a one-shot set classifier -- and route
with the true top-k directly at both train and eval.

Two regimes, mirroring the paper:

* ``ExpertChoiceRouter`` (headline) -- each recursion step keeps a *capacity*
  top-k of the currently-active tokens; survivors funnel into the next step
  (``1.0 -> 0.5 -> 0.25 -> ...`` of M). A token's **survival depth** -- the
  number of steps it was selected for -- is an intrinsic, compute-allocation
  based importance score ("a gene that survives to loop 4 is a disease gene").

* ``TokenChoiceRouter`` (ablation) -- each token picks *one* depth up front
  (top-1 over ``{1..K}``); its chosen depth is its importance. Load is not
  balanced by construction, so a Switch-style balancing loss is added.

Both routers reuse the *same* shared block via the ``block_fn`` callback, so
weight sharing (the parameter-efficiency claim) is preserved.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# block_fn(step_index, tokens) -> tokens. Indexing the step lets the caller
# return the shared block (weight sharing) or a per-step block (independent).
BlockFn = Callable[[int, torch.Tensor], torch.Tensor]
RouteInfo = Dict[str, object]


def _make_router(router_type: str, d_model: int, out_dim: int) -> nn.Module:
    """Tiny routing head (cf. ``mor_model/util.py`` LinearRouter / MLPRouter)."""
    if router_type == "linear":
        return nn.Linear(d_model, out_dim, bias=False)
    if router_type == "mlp":
        return nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim),
        )
    raise ValueError(f"Unknown router_type: {router_type!r}")


import os
_CAP_FLOOR = float(os.environ.get("RMT_CAP_FLOOR", "0.75"))

def default_capacity(depth: int) -> Tuple[float, ...]:
    """Funnel schedule: keep all tokens at step 0, then taper to a floor
    (default ``1.0, 0.75, 0.75, 0.75, ...`` at floor 0.75). A gentle funnel suits a
    *pooling* classifier -- the patient vector averages all M markers, so freezing too
    many tokens too early starves the pooled representation. The old 0.5 floor gave
    ``1.0, 0.75, 0.5, 0.5`` at K=4, whose repeated deep 0.5-step over-pruned the compact
    marker panel and made K=4 the weakest depth; a 0.75 floor removes that penalty.
    Override with env RMT_CAP_FLOOR."""
    return tuple(max(_CAP_FLOOR, 1.0 - 0.25 * t) for t in range(depth))


class ExpertChoiceRouter(nn.Module):
    """Per-step expert-choice routing with a decreasing-capacity funnel."""

    def __init__(self, d_model: int, depth: int, capacity: Tuple[float, ...],
                 alpha: float = 0.1, temp: float = 1.0, router_type: str = "linear",
                 prior_gate: bool = False):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.temp = temp
        # capacity given relative to the full marker set M; one entry per step.
        cap = list(capacity) if capacity else list(default_capacity(depth))
        if len(cap) < depth:                       # pad by repeating the last cap
            cap = cap + [cap[-1]] * (depth - len(cap))
        self.capacity = cap[:depth]
        # One scalar router per recursion step (the "expert" at that depth).
        self.routers = nn.ModuleList(
            [_make_router(router_type, d_model, 1) for _ in range(depth)])
        # Fix B: FiLM gate g_phi(h_m) that makes the biological prior SAMPLE- and
        # STATE-conditional instead of a fixed additive bias. sigmoid-bounded so the
        # model can learn to fully trust (1) or ignore (0) the prior per token.
        self.prior_gate = None
        if prior_gate:
            self.prior_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1))

    def forward(self, tokens: torch.Tensor, block_fn: BlockFn,
                refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                prior: Optional[torch.Tensor] = None, prior_weight: float = 0.0,
                ) -> Tuple[torch.Tensor, RouteInfo]:
        B, M, _ = tokens.shape
        device = tokens.device
        # cand_mask[b, j] = token j is still in the funnel at this step.
        cand_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        depth_count = torch.zeros(B, M, device=device)     # survival depth (importance)
        soft_depth = tokens.new_zeros(B, M)                # differentiable depth proxy (Fix E)
        z_loss = tokens.new_zeros(())
        prior_on = prior is not None and (
            prior_weight if isinstance(prior_weight, float) else True)

        for t in range(self.depth):
            k = max(1, int(round(self.capacity[t] * M)))
            k = min(k, int(cand_mask.sum(dim=1).min().item()))  # never exceed candidates
            k = max(1, k)

            logits = self.routers[t](tokens / self.temp).squeeze(-1)   # (B, M)
            # Biology-informed routing. OLD: fixed annealed additive bias
            # (sample-independent, provably ties the random control). REDESIGN:
            # the prior enters through a FiLM gate g_phi(h_m) (Fix B, sample- and
            # state-conditional) scaled by a persistent LEARNABLE beta (Fix D, the
            # tensor prior_weight), so the router can trust real structure and gate
            # out shuffled structure. prior is (M,) over the selected markers.
            if prior_on:
                bias = prior.unsqueeze(0)                              # (1, M)
                if self.prior_gate is not None:
                    g = torch.sigmoid(self.prior_gate(tokens).squeeze(-1))  # (B, M)
                    bias = g * bias
                logits = logits + prior_weight * bias
            # Expected survival at this step (differentiable in the router weights);
            # summed over steps it is a soft per-token recursion depth for Fix E.
            soft_depth = soft_depth + torch.sigmoid(logits) * cand_mask.float()
            z_loss = z_loss + (logits ** 2).mean()
            # Restrict selection to current candidates.
            masked = logits.masked_fill(~cand_mask, float("-inf"))
            weights, sel = torch.topk(masked, k, dim=1)               # (B, k)

            gate = (torch.sigmoid(weights) * self.alpha).unsqueeze(-1)  # (B, k, 1)
            idx = sel.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])    # (B, k, d)
            gathered = torch.gather(tokens, 1, idx)                     # (B, k, d)
            processed = block_fn(t, gathered)                         # shared block on k
            # Weighted residual: router gate scales how much this step moves the token.
            updated = gathered + gate * (processed - gathered)

            tokens = torch.scatter(tokens, 1, idx, updated)           # frozen if unselected
            # Survivors at step t are the candidates for step t+1 (the funnel).
            new_cand = torch.zeros_like(cand_mask)
            new_cand.scatter_(1, sel, True)
            cand_mask = new_cand
            depth_count.scatter_add_(1, sel, torch.ones_like(sel, dtype=tokens.dtype))

            if refine_fn is not None and t < self.depth - 1:
                g = refine_fn(tokens)                                  # (B, M)
                tokens = tokens * g.unsqueeze(-1)

        info: RouteInfo = {
            "depth_per_token": depth_count,         # (B, M) in [0, depth]
            "soft_depth_per_token": soft_depth,     # (B, M) differentiable proxy (Fix E)
            "z_loss": z_loss / self.depth,
            "balance_loss": tokens.new_zeros(()),   # balanced by construction
            "capacity": tuple(self.capacity),
        }
        return tokens, info


class TokenChoiceRouter(nn.Module):
    """Each token picks one depth up front (top-1 over K), then rides to it."""

    def __init__(self, d_model: int, depth: int, alpha: float = 1.0,
                 temp: float = 1.0, router_type: str = "linear"):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.temp = temp
        self.router = _make_router(router_type, d_model, depth)

    def forward(self, tokens: torch.Tensor, block_fn: BlockFn,
                refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                prior: Optional[torch.Tensor] = None, prior_weight: float = 0.0,
                ) -> Tuple[torch.Tensor, RouteInfo]:
        B, M, _ = tokens.shape
        logits = self.router(tokens / self.temp)                       # (B, M, depth)
        # Biology-informed routing: bias every depth logit by the per-marker prior.
        if prior is not None and prior_weight != 0.0:
            logits = logits + prior_weight * prior.view(1, M, 1)
        probs = F.softmax(logits, dim=-1) * self.alpha
        chosen = probs.argmax(dim=-1)                                   # (B, M) in [0, depth-1]
        gate = torch.gather(probs, -1, chosen.unsqueeze(-1))           # (B, M, 1)

        for t in range(self.depth):
            active = (chosen >= t).unsqueeze(-1)                        # (B, M, 1)
            processed = block_fn(t, tokens)                           # shared block
            updated = tokens + gate * (processed - tokens)
            tokens = torch.where(active, updated, tokens)
            if refine_fn is not None and t < self.depth - 1:
                g = refine_fn(tokens)
                tokens = tokens * g.unsqueeze(-1)

        # Switch-style load-balancing over depths (mirror token_choice_router.py).
        P_i = probs.mean(dim=(0, 1))                                    # (depth,) mean prob
        frac = torch.bincount(chosen.reshape(-1), minlength=self.depth).float() / (B * M)
        balance_loss = self.depth * (P_i * frac).sum()
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        info: RouteInfo = {
            "depth_per_token": (chosen + 1).to(tokens.dtype),          # (B, M) in [1, depth]
            "z_loss": z_loss,
            "balance_loss": balance_loss,
            "capacity": None,
        }
        return tokens, info
