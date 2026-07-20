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

"""The Recursive Marker Transformer.

Pipeline (proposal Stages 1-5):

    gene expression
      -> GeneEmbedding              (B, N, d)
      -> MarkerModule.select        top-K marker genes (global, interpretable)
      -> MarkerModule.aggregate     compress to (B, M, d) marker-anchored tokens
      -> RecursiveStack (K x shared) with recursive marker refinement
      -> mean-pool markers -> per-head linear classifier
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import RMTConfig
from .embedding import GeneEmbedding
from .marker import ConcreteSelector, MarkerModule, PathwayPooler, SlotRouter
from .recursion import RecursiveStack


class RecursiveMarkerTransformer(nn.Module):
    def __init__(self, cfg: RMTConfig, n_genes: int, head_n_classes: Dict[str, int],
                 head_dtypes: Dict[str, str],
                 pathway: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes
        self.head_dtypes = head_dtypes

        self.embed = GeneEmbedding(n_genes, cfg.d_model, cfg.dropout,
                                   n_channels=getattr(cfg, "n_channels", 1))
        self.marker = MarkerModule(cfg.d_model, n_genes, cfg.n_markers, cfg.marker_mode)
        # Soft selectors produce M marker tokens directly with all-gene gradient.
        if cfg.marker_mode == "concrete":
            self.selector = ConcreteSelector(n_genes, cfg.n_markers)
        elif cfg.marker_mode == "router":
            self.selector = SlotRouter(n_genes, cfg.n_markers, cfg.d_model)
            if getattr(cfg, "peak_init", True):
                self._peak_init_router()
        elif cfg.marker_mode == "pathway":
            # Reactome gene->pathway membership pooling (M = #pathways, fixed by
            # biology). M tokens = curated pathways, each pooling its member genes.
            if pathway is None:
                raise ValueError("marker_mode='pathway' requires the (N_genes, "
                                 "M_pathways) membership matrix")
            self.selector = PathwayPooler(pathway, pool=getattr(cfg, "pathway_pool", "mean"))
        else:
            self.selector = None
        self.stack = RecursiveStack(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            depth=cfg.recursion_depth, share_weights=cfg.share_weights,
            adaptive_depth=cfg.adaptive_depth, marker_ffn=cfg.marker_ffn,
            recursion_mode=cfg.recursion_mode, router_capacity=cfg.router_capacity,
            router_alpha=cfg.router_alpha, router_temp=cfg.router_temp,
            router_type=cfg.router_type,
            share_strategy=getattr(cfg, "share_strategy", "cycle"),
            n_unique_blocks=getattr(cfg, "n_unique_blocks", None),
            step_cache=getattr(cfg, "step_cache", False),
            bio_prior_gate=getattr(cfg, "bio_prior_gate", False),   # Fix B
        )
        # One classifier per head; binary heads use a single logit.
        self.classifiers = nn.ModuleDict({
            h: nn.Linear(cfg.d_model, 1 if head_dtypes[h] == "binary" else head_n_classes[h])
            for h in cfg.heads
        })
        # Auxiliary head for the marker-sufficiency loss (primary head only).
        self._primary = cfg.heads[0]
        prim_out = 1 if head_dtypes[self._primary] == "binary" else head_n_classes[self._primary]
        self.aux_marker_head = nn.Linear(cfg.d_model, prim_out)

        self.register_buffer("gene_variance", torch.zeros(n_genes), persistent=False)
        # Biology-informed router: per-gene network-centrality prior (genomap
        # gene-gene interaction graph). Zero unless set; beta_t is the annealed
        # additive-bias strength, updated by set_anneal.
        self.register_buffer("gene_centrality", torch.zeros(n_genes), persistent=False)
        self._use_prior = getattr(cfg, "gene_interaction", None) not in (None, "none")
        self._prior_beta = float(getattr(cfg, "router_prior_beta", 0.0))
        # Per-token prior for pathway tokens: Reactome pathway-hierarchy centrality
        # (M,), one entry per pathway token. Used in place of the per-gene
        # gene_centrality when the tokens ARE pathways (marker_mode='pathway').
        M_tok = self.selector.n_markers if cfg.marker_mode == "pathway" else 1
        self.register_buffer("token_prior", torch.zeros(M_tok), persistent=False)
        self._use_token_prior = False
        # Pathway-hierarchy attention bias (Reactome pathway->pathway graph): an
        # additive (M,M) bias on the self-attention logits so a pathway token
        # attends to its hierarchy neighbours. Zero unless installed.
        self.register_buffer("pathway_attn", torch.zeros(M_tok, M_tok),
                             persistent=False)
        self._use_attn_bias = False
        self._attn_lambda = float(getattr(cfg, "pathway_attn_lambda", 0.0))

        # ---- BIO-ROUTER REDESIGN state (BIO_ROUTER_REDESIGN.txt) ----------
        # Fix A: gene-graph propagation operator S = D^-1/2 (W+I) D^-1/2 (N,N) and a
        # LEARNABLE mix lambda (sigmoid-bounded) that smooths input expression along
        # the co-expression graph BEFORE marker selection (sample-conditional).
        self.register_buffer("bio_operator", torch.zeros(n_genes, n_genes),
                             persistent=False)
        # Fix E: gene-graph Laplacian L (N,N) for the depth-smoothness penalty.
        self.register_buffer("bio_laplacian", torch.zeros(n_genes, n_genes),
                             persistent=False)
        self._bio_prop = bool(getattr(cfg, "bio_graph_prop", False))
        self._bio_hops = int(getattr(cfg, "bio_prop_hops", 1))
        self._bio_lap_coeff = float(getattr(cfg, "bio_depth_laplacian", 0.0))
        self._bio_have_graph = False
        import math as _math
        lam0 = float(getattr(cfg, "bio_prop_lambda_init", 0.3))
        lam0 = min(max(lam0, 1e-3), 1.0 - 1e-3)
        self.bio_prop_logit = nn.Parameter(                     # sigmoid(.) = lambda
            torch.tensor(_math.log(lam0 / (1.0 - lam0)), dtype=torch.float32))
        # Fix D: persistent LEARNABLE prior strength beta = softplus(bio_beta), NOT
        # annealed to zero, so biology can survive to convergence if data wants it.
        self._bio_prior_learnable = bool(getattr(cfg, "bio_prior_learnable", False))
        self.bio_beta = nn.Parameter(
            torch.tensor(float(getattr(cfg, "bio_beta_init", 0.5)), dtype=torch.float32))
        # DATA-DRIVEN learned gene graph: a per-gene embedding whose cosine-similarity
        # IS the (rank-r) gene-gene affinity, trained end-to-end by the task loss.
        self._bio_learned = bool(getattr(cfg, "bio_learned_graph", False))
        if self._bio_learned:
            r = int(getattr(cfg, "bio_learned_rank", 16))
            self.gene_embed = nn.Parameter(torch.randn(n_genes, r) * 0.01)
        # FUSION: learnably mix the fixed biological interaction matrix (installed via
        # set_bio_graph) with the learned graph. sigmoid(gate) in [0,1]; gate<0 starts
        # biology-light so a pathological graph is easy to down-weight.
        self._bio_fuse = bool(getattr(cfg, "bio_learned_fuse", False))
        if self._bio_learned and self._bio_fuse:
            self.bio_fuse_gate = nn.Parameter(torch.tensor([-1.0]))
        # ANCHORED warm-start: an annealed penalty pulling the learned graph toward the
        # biological graph so the warm-start survives past initialisation. We store a
        # low-rank normalised factor B (G x rb) with A_bio ~= B B^T; the penalty is the
        # Frobenius distance ||norm(E)norm(E)^T - B B^T||^2, evaluated in r x rb space.
        self._bio_anchor = bool(getattr(cfg, "bio_learned_anchor", False))
        self._bio_anchor_have = False
        self._anneal_progress = 0.0
        if self._bio_learned and self._bio_anchor:
            rb = int(getattr(cfg, "bio_anchor_rank", 16))
            self.register_buffer("bio_anchor_factor", torch.zeros(n_genes, rb),
                                 persistent=False)

        # PATHWAY-SPACE learned graph (multi-omics): the single-cell learned-graph
        # mechanism, but the tokens are pathways and the graph is the PROVIDED Reactome
        # pathway->pathway adjacency (adjacency_matrix.csv), warm-started + propagated +
        # fused exactly like the gene path -- NEVER a co-expression graph computed on the
        # sparse omics. Installed by set_pathway_graph(); operates on the (B, M, d) tokens.
        self._pw_learned = (bool(getattr(cfg, "pathway_learned_graph", False))
                            and cfg.marker_mode == "pathway" and self.selector is not None)
        if self._pw_learned:
            M_pw = self.selector.n_markers
            rp = int(getattr(cfg, "pathway_learned_rank", 16))
            self.pathway_embed = nn.Parameter(torch.randn(M_pw, rp) * 0.01)
            _lp = min(max(float(getattr(cfg, "pathway_prop_lambda_init", 0.2)),
                          1e-3), 1.0 - 1e-3)
            self.pathway_prop_logit = nn.Parameter(
                torch.tensor(_math.log(_lp / (1.0 - _lp)), dtype=torch.float32))
            self.register_buffer("pathway_operator", torch.zeros(M_pw, M_pw),
                                 persistent=False)
            self._pw_have_graph = False
            self._pw_fuse = bool(getattr(cfg, "pathway_learned_fuse", False))
            if self._pw_fuse:
                self.pathway_fuse_gate = nn.Parameter(torch.tensor([-1.0]))
            # USE PROVIDED ADJACENCY DIRECTLY: fixed GCN-normalised operator for both sites.
            self._pw_fixed_graph = bool(getattr(cfg, "pathway_fixed_graph", False))
            # MONOTONE-SAFE MERGE: LayerScale-style zero-init residual for the embedding site.
            self._pw_prop_residual = bool(getattr(cfg, "pathway_prop_residual", False))
            self._pw_prop_complement = bool(getattr(cfg, "pathway_prop_complement", False))
            if self._pw_prop_residual:
                self.pathway_prop_gamma = nn.Parameter(torch.zeros(1))   # gamma=0 -> no-op at init
            if self._pw_prop_complement:
                _d = int(cfg.d_model)
                self.pathway_prop_mlp = nn.Sequential(
                    nn.Linear(2 * _d, _d), nn.GELU(), nn.Linear(_d, _d))
                nn.init.zeros_(self.pathway_prop_mlp[-1].weight)         # zero-init output -> no-op
                nn.init.zeros_(self.pathway_prop_mlp[-1].bias)

        # REDESIGNED bio-router: zero-init graph-conv residual on the depth-router logits
        # (router.py). The router message-passes over the (M,M) biological token graph so
        # routing depth can depend on a token's neighbourhood -- learned, bounded, starts
        # as a no-op. Replaces the harmful static centrality prior as the router-site biology.
        self._bio_graph_router = bool(getattr(cfg, "bio_graph_router", False))
        # single-cell attention-bias (analogue of pathway_attn_bias for marker_mode!='pathway'):
        # bias marker self-attention along the learned gene sub-graph over selected markers.
        self._gene_attn = (bool(getattr(cfg, "gene_attn_bias", False))
                           and cfg.marker_mode != "pathway")
        self._gene_attn_lambda = float(getattr(cfg, "gene_attn_lambda", 2.0))
        self._gene_attn_topk = int(getattr(cfg, "gene_attn_topk", 15))
        # site decoupling: keep the warm-started graph but optionally skip its input smoothing
        self._bio_learned_prop = bool(getattr(cfg, "bio_learned_prop", True))
        self._pw_learned_prop = bool(getattr(cfg, "pathway_learned_prop", True))

    def set_gene_variance(self, variance: torch.Tensor) -> None:
        self.gene_variance.copy_(variance.to(self.gene_variance))

    def init_gene_embed_from_operator(self, operator: torch.Tensor) -> bool:
        """Warm-start the LEARNED gene graph from a biological affinity operator
        (co-expression S or curated Reactome co-membership). We take the operator's
        informative low-frequency eigenmodes (Laplacian-eigenmap style) to seed the
        gene embedding, so the learned cosine graph A = normalize(E)normalize(E)^T
        starts near the biological graph's structure and is then refined end-to-end.

        Robustness safeguards:
        (1) DROP the leading eigenvector -- for a smoothing / row-stochastic operator it
            is the trivial ~uniform mode; seeding it makes every gene embed alike;
        (2) add a matched random baseline (randn*0.01) so the embedding stays full-rank
            and trainable;
        (3) GUARD: some co-expression graphs are all-NaN (zero-variance genes) or empty;
            seeding from them injects NaNs into the logits and collapses the model, so we
            reject a degenerate operator and KEEP the random init.

        Returns True if the biological warm-start was applied, False if it fell back to
        the random init. No-op / False unless bio_learned_graph is enabled."""
        if not getattr(self, "_bio_learned", False):
            return False
        with torch.no_grad():
            W = operator.detach().cpu().float()
            if W.dim() != 2 or W.shape[0] != W.shape[1] or W.shape[0] != self.gene_embed.shape[0]:
                return False
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            if float(W.abs().sum()) == 0.0:                        # degenerate / empty graph
                return False
            W = 0.5 * (W + W.t())                                  # symmetrise
            r = int(self.gene_embed.shape[1])
            evals, evecs = torch.linalg.eigh(W)                   # ascending
            order = torch.argsort(evals, descending=True)
            # skip the leading (trivial, over-smoothing) mode; take the next r modes
            idx = order[1:r + 1] if order.numel() > r else order[:r]
            E = evecs[:, idx] * evals[idx].clamp(min=0.0).sqrt().unsqueeze(0)
            if not torch.isfinite(E).all() or float(E.std()) < 1e-8:
                return False
            # Give the biological structure a larger footprint than the matched random
            # baseline so the warm-start actually shapes the initial graph (the old code
            # used equal 0.01/0.01, which made bio-init nearly indistinguishable from
            # random). Scales are configurable; defaults 0.05 (bio) vs 0.005 (random).
            s_bio = float(getattr(self.cfg, "bio_init_scale", 0.05))
            s_rand = float(getattr(self.cfg, "bio_init_rand", 0.005))
            E = E / (E.std() + 1e-6) * s_bio                      # biological structure
            E = E + torch.randn_like(E) * s_rand                  # matched random baseline
            self.gene_embed.copy_(E.to(self.gene_embed))
            return True

    def set_bio_anchor(self, operator: torch.Tensor) -> bool:
        """Store a low-rank normalised factor B of the biological graph so the anchor
        penalty can pull the learned cosine graph toward A_bio = B B^T. Rejects a
        degenerate (NaN / empty) operator, disabling the anchor for that dataset (it then
        behaves as the plain random-init learned graph). Returns True if installed."""
        if not (self._bio_learned and self._bio_anchor):
            return False
        with torch.no_grad():
            W = operator.detach().cpu().float()
            if W.dim() != 2 or W.shape[0] != W.shape[1] or W.shape[0] != self.gene_embed.shape[0]:
                return False
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            if float(W.abs().sum()) == 0.0:
                return False
            W = 0.5 * (W + W.t())
            rb = int(self.bio_anchor_factor.shape[1])
            evals, evecs = torch.linalg.eigh(W)
            order = torch.argsort(evals, descending=True)
            idx = order[1:rb + 1] if order.numel() > rb else order[:rb]   # drop trivial mode
            B = evecs[:, idx] * evals[idx].clamp(min=0.0).sqrt().unsqueeze(0)
            if not torch.isfinite(B).all() or float(B.std()) < 1e-8:
                return False
            B = nn.functional.normalize(B, dim=1)                 # so A_bio=BB^T is a cosine graph
            self.bio_anchor_factor.copy_(B.to(self.bio_anchor_factor))
            self._bio_anchor_have = True
            return True

    def bio_anchor_loss(self) -> torch.Tensor:
        """Annealed Frobenius distance between the learned cosine graph A=norm(E)norm(E)^T
        and the biological target A_bio=B B^T, computed in low rank (never materialising
        the G x G matrices):  ||A - A_bio||_F^2 = ||E^T E||^2 - 2||E^T B||^2 + ||B^T B||^2.
        Weight decays linearly to 0 over training so biology guides early, data decides
        late. Zero unless an anchor factor was installed for this dataset."""
        # bio_prop_logit is always registered; gene_embed exists ONLY when the learned
        # graph is enabled, so derive the device from the former and guard before touching
        # gene_embed (models without the learned graph, e.g. pathway P-NET, have none).
        dev = self.bio_prop_logit.device
        if not (getattr(self, "_bio_anchor", False) and getattr(self, "_bio_anchor_have", False)
                and hasattr(self, "gene_embed")):
            return torch.zeros((), device=dev)
        lam0 = float(getattr(self.cfg, "bio_anchor_lambda", 0.5))
        floor = float(getattr(self.cfg, "bio_anchor_floor", 0.0))
        # decay from lam0 toward lam0*floor; floor>0 keeps a standing pull to convergence
        lam = lam0 * (floor + (1.0 - floor) * (1.0 - self._anneal_progress))
        if lam <= 0.0:
            return torch.zeros((), device=dev)
        En = nn.functional.normalize(self.gene_embed, dim=1)      # (G, r)
        B = self.bio_anchor_factor.to(dev)                        # (G, rb)
        ete = En.t() @ En                                         # (r, r)
        etb = En.t() @ B                                          # (r, rb)
        btb = B.t() @ B                                           # (rb, rb)
        frob = (ete * ete).sum() - 2.0 * (etb * etb).sum() + (btb * btb).sum()
        return lam * frob.clamp(min=0.0)

    def set_gene_interaction(self, centrality: torch.Tensor) -> None:
        """Install the genomap gene-gene-interaction centrality prior (N,)."""
        self.gene_centrality.copy_(centrality.to(self.gene_centrality))
        self._use_prior = True

    def set_bio_graph(self, operator: Optional[torch.Tensor],
                      laplacian: Optional[torch.Tensor]) -> None:
        """Install the co-expression propagation operator S (Fix A) and Laplacian L
        (Fix E). Enables graph-propagation / depth-smoothness for this model."""
        if operator is not None:
            self.bio_operator.copy_(operator.to(self.bio_operator))
        if laplacian is not None:
            self.bio_laplacian.copy_(laplacian.to(self.bio_laplacian))
        self._bio_have_graph = True

    def set_token_prior(self, prior: torch.Tensor) -> None:
        """Install a per-pathway-token prior (M,) -- e.g. Reactome pathway-graph
        eigenvector centrality. Added (annealed by beta_t) to the depth-router
        logits exactly like the gene prior, but indexed by token, not gene."""
        self.token_prior.copy_(prior.to(self.token_prior))
        self._use_token_prior = True

    def set_pathway_adjacency(self, adjacency: torch.Tensor) -> None:
        """Install the Reactome pathway->pathway hierarchy graph (M, M) as an
        additive attention bias: lambda on hierarchy-adjacent pathway pairs, 0
        elsewhere (self-attention on the diagonal is left unbiased). Pathway
        tokens then attend preferentially along the curated Reactome graph."""
        A = adjacency.to(self.pathway_attn).float()
        A = torch.maximum(A, A.t())
        A.fill_diagonal_(0.0)
        self.pathway_attn.copy_(self._attn_lambda * (A > 0).float())
        self._use_attn_bias = True

    def set_pathway_graph(self, adjacency: torch.Tensor) -> bool:
        """Install the PROVIDED Reactome pathway->pathway adjacency (M, M) as the fixed
        pathway graph AND warm-start the learnable pathway embedding from its
        low-frequency eigenmodes. Pathway-space twin of init_gene_embed_from_operator:
        the graph is loaded from adjacency_matrix.csv, never computed from the sparse
        omics. Returns True if installed (no-op / False unless pathway_learned_graph)."""
        if not getattr(self, "_pw_learned", False):
            return False
        with torch.no_grad():
            A = torch.nan_to_num(adjacency.detach().cpu().float(),
                                 nan=0.0, posinf=0.0, neginf=0.0)
            if A.dim() != 2 or A.shape[0] != A.shape[1] or A.shape[0] != self.pathway_embed.shape[0]:
                return False
            A = 0.5 * (A + A.t())
            M = A.shape[0]
            # GCN-normalised operator S = D^-1/2 (A+I) D^-1/2 for propagation / fuse
            Ai = A + torch.eye(M)
            dinv = Ai.sum(1).clamp_min(1e-8).rsqrt()
            op = Ai * dinv[:, None] * dinv[None, :]
            self.pathway_operator.copy_(op.to(self.pathway_operator))
            self._pw_have_graph = True
            # warm-start pathway_embed from the operator's top-r eigenmodes (drop the
            # trivial leading mode), + a small matched-random baseline (Laplacian-eigenmap)
            if (getattr(self.cfg, "pathway_learned_init", "bio") == "bio"
                    and float(A.abs().sum()) > 0.0):
                r = int(self.pathway_embed.shape[1])
                evals, evecs = torch.linalg.eigh(op)
                order = torch.argsort(evals, descending=True)
                idx = order[1:r + 1] if order.numel() > r else order[:r]
                E = evecs[:, idx] * evals[idx].clamp(min=0.0).sqrt().unsqueeze(0)
                if torch.isfinite(E).all() and float(E.std()) > 1e-8:
                    E = E / (E.std() + 1e-6) * 0.05
                    E = E + torch.randn_like(E) * 0.005
                    self.pathway_embed.copy_(E.to(self.pathway_embed))
        return True

    @torch.no_grad()
    def _peak_init_router(self):
        """Point each router query at a distinct gene's key so attention starts
        peaked (random-selection quality) rather than uniform mush -- the same
        fix that lets Concrete learn in a small epoch budget."""
        ident = self.embed.gene_identity()                      # (N, d)
        k = self.selector.key(ident)                            # (N, d)
        kn = nn.functional.normalize(k, dim=1)
        genes = torch.randperm(self.n_genes)[: self.selector.n_markers]
        # Strong peak so each slot starts ~one-hot on a distinct gene; the exact
        # constant is uncritical as long as the softmax over N genes is peaked.
        self.selector.queries.copy_(kn[genes] * 60.0)

    def set_anneal(self, progress: float) -> None:
        """Advance the explore->exploit schedules. (1) the selector temperature
        (gated by anneal_markers); (2) the biological-prior strength beta_t, which
        decays linearly to 0 over training when router_prior_anneal is on (a
        warm-start prior that hands off to the data-driven router), else stays at
        beta_0."""
        progress = min(1.0, max(0.0, float(progress)))
        self._anneal_progress = progress                        # drives the bio-anchor decay
        if getattr(self.cfg, "anneal_markers", True) and \
                self.selector is not None and hasattr(self.selector, "set_progress"):
            self.selector.set_progress(progress)
        if getattr(self.cfg, "router_prior_anneal", True):
            self._prior_beta = float(getattr(self.cfg, "router_prior_beta", 0.0)) * (1.0 - progress)
        else:
            self._prior_beta = float(getattr(self.cfg, "router_prior_beta", 0.0))

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        gene_identity = self.embed.gene_identity()              # (N, d)

        # Fix A: sample-conditional graph propagation. Smooth the input expression
        # along the co-expression graph, x <- (1-lam) x + lam (x S), BEFORE marker
        # selection, so each gene token carries its module's signal (denoising with
        # the REAL graph; noise-mixing with a shuffled one -- the falsifiable gap).
        if self._bio_prop and self._bio_have_graph and x.dim() == 2:
            lam = torch.sigmoid(self.bio_prop_logit)
            xs = x
            for _ in range(max(1, self._bio_hops)):
                xs = (1.0 - lam) * xs + lam * (xs @ self.bio_operator)
            x = xs
        elif self._bio_prop and self._bio_have_graph and x.dim() == 3:
            # Multi-modal (B,G,C): smooth EACH omics channel along the same gene-gene
            # graph, so mutation and copy-number both denoise along the biological network
            # (this is what lets a fixed aggregated network act as a prior on P-NET too).
            lam = torch.sigmoid(self.bio_prop_logit)
            xs = x
            for _ in range(max(1, self._bio_hops)):
                xs = (1.0 - lam) * xs + lam * torch.einsum("bgc,gh->bhc", xs, self.bio_operator)
            x = xs
        # DATA-DRIVEN alternative (bio_learned_graph): propagate x along the LEARNED
        # low-rank synthetic-correlation graph A = E~ E~^T. Computed as (x E~) E~^T so
        # it never materialises the G x G matrix. Magnitude is renormalised per sample
        # so propagation mixes information without rescaling x (training stability);
        # the learnable lam controls how much to trust the learned graph.
        elif self._bio_learned and self._bio_learned_prop and x.dim() == 2:
            lam = torch.sigmoid(self.bio_prop_logit)
            En = nn.functional.normalize(self.gene_embed, dim=1)     # (G, r) unit rows
            fuse = self._bio_fuse and self._bio_have_graph
            g = torch.sigmoid(self.bio_fuse_gate) if fuse else None
            xs = x
            for _ in range(max(1, self._bio_hops)):
                prop = (xs @ En) @ En.t()                            # (B, G) learned graph
                prop = prop * (xs.norm(dim=1, keepdim=True)
                               / (prop.norm(dim=1, keepdim=True) + 1e-6))
                if fuse:
                    # persistent biological interaction matrix, learnably mixed in
                    bprop = xs @ self.bio_operator                  # (B, G) fixed bio graph
                    bprop = bprop * (xs.norm(dim=1, keepdim=True)
                                     / (bprop.norm(dim=1, keepdim=True) + 1e-6))
                    prop = (1.0 - g) * prop + g * bprop
                xs = (1.0 - lam) * xs + lam * prop
            x = xs

        if self.selector is not None:
            # Soft selection (Concrete or cross-attention router): soft (train) /
            # hard (eval) selection over ALL genes -> M marker tokens directly, so
            # gradients reach every gene (the property hard top-k routing lacks).
            w = self.selector.weights(gene_identity)           # (M, N)
            sel_ident = w @ gene_identity                      # (M, d)
            if x.dim() == 3:
                # Multimodal: pool each gene-aligned channel by the same marker
                # weights, then fuse the C channels at the shared value projection.
                sel_value = torch.einsum("bnc,mn->bmc", x, w)  # (B, M, C)
                cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value)
            else:
                sel_value = x @ w.t()                          # (B, M) selected expression
                cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value.unsqueeze(-1))
            marker_idx = self.selector.selected_indices(gene_identity)
            scores = w.max(dim=0).values                       # (N,) per-gene max selection weight
            marker_ident = nn.functional.normalize(sel_ident, dim=1)
        else:
            tokens = self.embed(x)                             # (B, N, d)
            marker_idx, scores, init_gate = self.marker.select(gene_identity, self.gene_variance)
            if self.cfg.compress_mode == "drop":
                # Strict selection: use ONLY the selected marker genes (no folding).
                gate_b = init_gate.unsqueeze(0).expand(x.shape[0], -1)   # (B, M)
                cluster = tokens[:, marker_idx, :] * gate_b.unsqueeze(-1)
            else:
                cluster = self.marker.aggregate(tokens, gene_identity, marker_idx, init_gate)
            marker_ident = nn.functional.normalize(gene_identity[marker_idx], dim=1)

        # PATHWAY-SPACE graph propagation (multi-omics): smooth the pathway tokens along
        # the LEARNED pathway graph A = norm(E)norm(E)^T (E warm-started from the PROVIDED
        # Reactome adjacency_matrix.csv), optionally fused with the fixed provided graph.
        # Exactly the single-cell x-propagation, but on the (B, M, d) pathway tokens using
        # the provided graph -- never a co-expression graph computed on the sparse omics.
        if (getattr(self, "_pw_learned", False) and getattr(self, "_pw_have_graph", False)
                and self._pw_learned_prop):
            lam = torch.sigmoid(self.pathway_prop_logit)
            if getattr(self, "_pw_fixed_graph", False):
                # provided GCN-normalised Reactome operator, used DIRECTLY (no learned graph)
                prop = torch.einsum("mn,bnd->bmd", self.pathway_operator, cluster)
            else:
                En = nn.functional.normalize(self.pathway_embed, dim=1)   # (M, r)
                z = torch.einsum("mr,bmd->brd", En, cluster)             # E^T cluster
                prop = torch.einsum("mr,brd->bmd", En, z)                # (E E^T) cluster
            prop = prop * (cluster.norm(dim=(1, 2), keepdim=True)
                           / (prop.norm(dim=(1, 2), keepdim=True) + 1e-6))
            if getattr(self, "_pw_fuse", False):
                g = torch.sigmoid(self.pathway_fuse_gate)
                pf = torch.einsum("mn,bnd->bmd", self.pathway_operator, cluster)
                pf = pf * (cluster.norm(dim=(1, 2), keepdim=True)
                           / (pf.norm(dim=(1, 2), keepdim=True) + 1e-6))
                prop = (1.0 - g) * prop + g * pf
            if getattr(self, "_pw_prop_complement", False):
                # COMPLEMENTARY residual: learnable zero-init MLP over [neighbour-mean,
                # high-freq contrast]. No-op at init (output layer zeroed) so 'both' starts
                # == router-only, but can add complementary biological signal -> strict win.
                feat = self.pathway_prop_mlp(torch.cat([prop, cluster - prop], dim=-1))
                cluster = cluster + feat
            elif getattr(self, "_pw_prop_residual", False):
                # LayerScale-style ZERO-INIT residual: cluster unchanged at init (gamma=0),
                # so "both" starts == router-only and the embedding biology can only ADD.
                cluster = cluster + self.pathway_prop_gamma * prop
            else:
                cluster = (1.0 - lam) * cluster + lam * prop

        # Marker-sufficiency aux loss: probe whether the (pre-recursion) marker
        # tokens alone are task-sufficient, which trains selection toward
        # discriminative genes.
        aux_logits = self.aux_marker_head(cluster.mean(dim=1))

        # Biology-informed router: gather the centrality prior for the selected
        # markers and pass it (with the annealed strength beta_t) to the depth
        # router. For pathway tokens the prior is per-pathway (token_prior);
        # otherwise it is the per-gene centrality gathered at the selected markers
        # (marker_idx is the per-slot arg-max gene). Either way prior is (M,).
        if self._use_token_prior:
            prior = self.token_prior
            prior_weight = self._prior_beta
        elif self._use_prior:
            prior = self.gene_centrality[marker_idx]
            # Fix D: persistent learnable beta = softplus(bio_beta) (a tensor, so
            # gradient flows and it need not anneal to 0), else the old annealed float.
            prior_weight = (nn.functional.softplus(self.bio_beta)
                            if self._bio_prior_learnable else self._prior_beta)
        else:
            prior, prior_weight = None, 0.0

        refine_fn = self.marker.refine_gate if self.cfg.recursive_marker_refine else None
        attn_bias = self.pathway_attn if self._use_attn_bias else None
        # SINGLE-CELL attention bias: additive (M,M) bias on marker self-attention along the
        # learned gene sub-graph (top-k biological neighbours per marker). Analogue of the
        # pathway attention bias, but the graph is the LEARNED gene graph over selected markers
        # (single-cell has no provided Reactome graph). Requires the learned gene graph.
        if getattr(self, "_gene_attn", False) and self._bio_learned and hasattr(self, "gene_embed"):
            Eg = nn.functional.normalize(self.gene_embed[marker_idx], dim=1)   # (M, r)
            S = (Eg @ Eg.t())                                                  # (M, M) cosine
            M_ = S.shape[0]
            kk = min(self._gene_attn_topk, M_ - 1)
            S = S - torch.eye(M_, device=S.device) * 2.0                       # drop self
            nbr = S.topk(kk, dim=1).indices
            B = torch.zeros(M_, M_, device=S.device)
            B.scatter_(1, nbr, 1.0)
            attn_bias = self._gene_attn_lambda * B                             # (M, M) additive
        # REDESIGNED bio-router: build the (M,M) row-normalised token graph so the depth
        # router can message-pass over the biological neighbourhood (zero-init residual in
        # router.py, so it can only learn to help). Source: the learned pathway graph
        # (pathway mode) or the learned gene sub-graph over the selected markers (else).
        token_graph = None
        if getattr(self, "_bio_graph_router", False):
            A = None
            if getattr(self, "_pw_fixed_graph", False) and getattr(self, "_pw_have_graph", False):
                # provided Reactome operator directly -> router graph cannot be corrupted by
                # the embedding objective (decouples the two sites; fixes prostate coupling)
                A = self.pathway_operator
            elif getattr(self, "_pw_learned", False):
                Ep = nn.functional.normalize(self.pathway_embed, dim=1)
                A = Ep @ Ep.t()
            elif getattr(self, "_pw_have_graph", False):
                A = self.pathway_operator
            elif self._bio_learned and hasattr(self, "gene_embed"):
                Eg = nn.functional.normalize(self.gene_embed[marker_idx], dim=1)
                A = Eg @ Eg.t()
            if A is not None:
                A = A.clamp(min=0.0)
                token_graph = A / (A.sum(-1, keepdim=True) + 1e-6)      # row-stochastic
        h, route_info = self.stack(cluster, refine_fn, prior=prior,
                                   prior_weight=prior_weight,
                                   attn_bias=attn_bias, token_graph=token_graph)
        pooled = h.mean(dim=1)                                  # (B, d)

        logits = {head: clf(pooled) for head, clf in self.classifiers.items()}

        # Fix E: graph-Laplacian depth-smoothness -- co-regulated genes (adjacent in
        # the co-expression graph) should get similar recursion depth. Penalise
        # d^T L d over the selected markers using the soft (differentiable) depth.
        bio_lap = h.new_zeros(())
        if (self._bio_lap_coeff > 0.0 and self._bio_have_graph
                and "soft_depth_per_token" in route_info):
            d_soft = route_info["soft_depth_per_token"]         # (B, M)
            Lsub = self.bio_laplacian[marker_idx][:, marker_idx]  # (M, M)
            # sum_ij L_ij d_i d_j, averaged over batch and markers.
            bio_lap = torch.einsum("bi,ij,bj->b", d_soft, Lsub, d_soft).mean() / d_soft.shape[1]

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "scores": scores,
            "marker_idx": marker_idx,
            "marker_ident": marker_ident,
            # MoR routing: per-marker-token recursion depth (importance signal)
            # and the raw router losses (z-loss, load-balancing) for the objective.
            "recursion_depth_per_token": route_info["depth_per_token"],
            "router_z_loss": route_info["z_loss"],
            "router_balance_loss": route_info["balance_loss"],
            "bio_lap_loss": bio_lap,                             # Fix E penalty
            "bio_anchor_loss": self.bio_anchor_loss(),           # annealed warm-start anchor
        }

    def transformer_param_count(self) -> int:
        """Parameters in the recursive transformer stack only (the quantity the
        weight-sharing claim is about)."""
        return sum(p.numel() for p in self.stack.parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
