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

"""SMART on the Reactome / P-NET pathway-informed multi-omics cohorts.

Predict the per-patient ``response`` (binary for prostate/blca/stad; the 5-class
subtype for brca) from somatic mutation and/or copy-number, where the M tokens
are **Reactome pathways** (``marker_mode='pathway'``): each token pools its member
genes through the fixed gene->pathway membership, and -- optionally -- the depth
router is biased by the curated pathway-hierarchy centrality
(``gene_interaction='reactome'``). This is the curated-prior counterpart to the
data-driven co-expression router, whose ablation (none/coexpr/random) left
``coexpr ~= none`` open.

Ablation axes (all reachable from the CLI):
  * ``--marker_mode  pathway`` (proposed) vs ``router`` (learned marker tokens);
  * ``--gene_interaction reactome`` (curated prior) vs ``none``/``coexpr``/``random``;
  * ``--channels mut|cnv|mut_cnv`` (modality).

Usage:
    python -m recursive_marker_transformer.pathway_tasks \
        --task prostate --channels mut_cnv --marker_mode pathway \
        --gene_interaction reactome --device cuda
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import RMTConfig
from .losses import RMTLoss
from .model import RecursiveMarkerTransformer
from .pathway_data import CHANNEL_SETS, load_cohort, load_pan_meta
from .train import _class_weights, _depth_stats, evaluate, resolve_device

# pancancer-meta tasks: pathways/labels ship without omics, so expression is
# joined from the Xena PANCAN matrix. task -> (cohort dir, label column).
PANMETA = {
    "panmeta_response": ("pancancer_meta_pri", "response"),         # primary vs metastatic
    "panmeta_subtype":  ("pancancer_meta_pri", "primary_disease"),  # 32-class cancer type
}
TASKS = ["prostate", "blca", "stad", "brca"] + list(PANMETA)


class _DictLoader:
    """Yield (x, {head: y}); x is (B,G) or (B,G,C)."""
    def __init__(self, X, y, idx, bs, shuffle, head):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        self.dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        self.head = head

    def __iter__(self):
        for xb, yb in self.dl:
            yield xb, {self.head: yb}

    def __len__(self):
        return len(self.dl)


def _result_tag(task: str, channel_set: str, cfg) -> str:
    """Stable filename tag encoding every ablation axis that varies a run."""
    t = f"{task}__{channel_set}__{cfg.marker_mode}__{cfg.gene_interaction}"
    if getattr(cfg, "marker_mode", "") == "pathway" and getattr(cfg, "pathway_pool", "mean") != "mean":
        t += f"__{cfg.pathway_pool}"
    if cfg.recursion_mode != "expert":
        t += f"__{cfg.recursion_mode}"
    if getattr(cfg, "pathway_attn_bias", False):
        t += "__attnbias"
    if not cfg.share_weights:
        t += "__indep"
    if getattr(cfg, "n_unique_blocks", None) is not None or getattr(cfg, "share_strategy", "cycle") != "cycle":
        t += f"__{cfg.share_strategy}{cfg.n_unique_blocks or ''}"
    if getattr(cfg, "step_cache", False):
        t += "__stepcache"
    if cfg.recursion_depth != 4:
        t += f"__k{cfg.recursion_depth}"
    return t


def _zscore_train(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Per-(gene, channel) z-scoring on the train rows. Handles (N,G) and (N,G,C)."""
    mu = X[tr].mean(0, keepdims=True)
    sd = X[tr].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def _fit_eval(task, coh, X, y, tr, va, te, cfg, G, K, dtypes, device, init_block=None):
    Xs = _zscore_train(X, tr)
    dl_tr = _DictLoader(Xs, y, tr, cfg.batch_size, True, task)
    dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False, task)
    dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False, task)

    pathway = torch.from_numpy(coh.P) if cfg.marker_mode == "pathway" else None
    model = RecursiveMarkerTransformer(cfg, G, {task: K}, dtypes, pathway=pathway).to(device)
    if init_block is not None:                       # warm-start: seed the shared block
        model.stack.blocks[0].load_state_dict(init_block)
    # variance prior uses the mutation/first channel
    expr_tr = Xs[tr] if Xs.ndim == 2 else Xs[tr, :, 0]
    model.set_gene_variance(torch.from_numpy(expr_tr.var(0).astype(np.float32)))
    # curated Reactome pathway-hierarchy prior on the depth router (per token)
    if cfg.gene_interaction == "reactome" and cfg.marker_mode == "pathway":
        model.set_token_prior(torch.from_numpy(coh.centrality))
    # Reactome pathway->pathway hierarchy as an attention bias between pathway tokens
    if getattr(cfg, "pathway_attn_bias", False) and cfg.marker_mode == "pathway":
        model.set_pathway_adjacency(torch.from_numpy(coh.adjacency))

    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    criterion = RMTLoss(cfg, dtypes, {task: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear LR warmup (1%->100% over first ~10% of epochs) then cosine; prevents the
    # wide-model collapse-to-majority-class seen without warmup.
    _warm = max(1, round(0.1 * cfg.epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=_warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - _warm))],
        milestones=[_warm])

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        model.set_anneal(ep / max(cfg.epochs - 1, 1))
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = {h: v.to(device) for h, v in yb.items()}
            loss = criterion(model(xb), yb)["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        yt, yp = evaluate(model, dl_va, device, dtypes)[task]
        vf1 = f1_score(yt, yp, average="macro")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    yt, yp = evaluate(model, dl_te, device, dtypes)[task]
    return yt, yp, model, dl_te


def run(task, channel_set, base, out_dir, device, min_genes=5):
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    dtypes = {task: "multiclass"}

    if task in PANMETA:
        cohort_dir, label = PANMETA[task]
        coh = load_pan_meta(label=label, cohort=cohort_dir, min_genes=min_genes)
    else:
        coh = load_cohort(task, channels=channel_set, min_genes=min_genes)
    X, y = coh.X, coh.y
    G = X.shape[1]
    K = int(y.max() + 1)
    C = 1 if X.ndim == 2 else X.shape[2]
    M = len(coh.pathways)

    cfg = replace(base, heads=(task,), n_hvg=None, n_channels=C)

    idx = np.arange(len(y))
    _, cnt = np.unique(y, return_counts=True)
    strat = y if cnt.min() >= 2 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=base.seed, stratify=strat)
    _, cnt = np.unique(y[tr], return_counts=True)
    strat = y[tr] if cnt.min() >= 2 else None
    tr, va = train_test_split(tr, test_size=0.15, random_state=base.seed, stratify=strat)

    print(f"\n########## {task} [{channel_set}] mode={cfg.marker_mode} "
          f"prior={cfg.gene_interaction} N={len(y)} G={G} C={C} M={M} K={K} "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) device={device} ##########",
          flush=True)

    yt, yp, model, dl_te = _fit_eval(task, coh, X, y, tr, va, te, cfg, G, K, dtypes, device)

    mean_slot_depth, _midx, active = _depth_stats(model, dl_te, device, cfg)

    def _step_flops(a):
        return 4.0 * a * a * cfg.d_model + 4.0 * a * cfg.d_model * cfg.d_ff
    flops_nominal = cfg.recursion_depth * _step_flops(M)
    flops_eff = float(sum(_step_flops(float(active[t])) for t in range(cfg.recursion_depth)))
    saving = flops_eff / flops_nominal if flops_nominal else 1.0

    res = {
        "task": task, "channel_set": channel_set, "channels": coh.channels, "n_channels": C,
        "marker_mode": cfg.marker_mode, "gene_interaction": cfg.gene_interaction,
        "recursion_mode": cfg.recursion_mode, "pathway_attn_bias": bool(cfg.pathway_attn_bias),
        "n_samples": int(len(y)), "n_genes": int(G), "n_pathways": int(M), "n_classes": int(K),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "class_names": [str(u) for u in coh.classes],
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "mean_recursion_depth": float(mean_slot_depth.mean()),
        "compute_saving_ratio": saving,
        "config": cfg.as_dict(),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "per_class": classification_report(yt, yp, zero_division=0, output_dict=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _result_tag(task, channel_set, cfg)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"  [TEST] {tag} acc={res['accuracy']*100:.1f} "
          f"macroF1={res['macro_f1']*100:.1f} weightedF1={res['weighted_f1']*100:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_pathway"))
    ap.add_argument("--task", type=str, default="prostate", choices=TASKS)
    ap.add_argument("--channels", nargs="+", default=["mut_cnv"],
                    choices=list(CHANNEL_SETS.keys()))
    ap.add_argument("--marker_mode", type=str, default="pathway",
                    choices=["pathway", "router", "learnable", "concrete"])
    ap.add_argument("--pathway_pool", type=str, default="mean", choices=["mean", "sum"],
                    help="'sum' (burden) for sparse binary mutation, 'mean' for dense assays")
    ap.add_argument("--gene_interaction", type=str, default="reactome",
                    choices=["reactome", "none", "coexpr", "random"])
    ap.add_argument("--recursion_mode", type=str, default="expert",
                    choices=["expert", "token", "fixed"])
    ap.add_argument("--pathway_attn_bias", action="store_true",
                    help="bias attention by the Reactome pathway hierarchy "
                         "(forces recursion_mode=token, the full-token regime)")
    ap.add_argument("--pathway_attn_lambda", type=float, default=2.0)
    # ---- 14-table mechanisms (mirror singlecell.py) ----
    ap.add_argument("--recursion_depth", type=int, default=4)
    ap.add_argument("--no_share_weights", dest="share_weights", action="store_false",
                    default=True, help="untie recursion blocks (independent stack)")
    ap.add_argument("--share_strategy", type=str, default="cycle",
                    choices=["cycle", "sequence", "middle_cycle", "middle_sequence"])
    ap.add_argument("--n_unique_blocks", type=int, default=None)
    ap.add_argument("--step_cache", action="store_true")
    ap.add_argument("--router_type", type=str, default="linear", choices=["linear", "mlp"])
    ap.add_argument("--router_temp", type=float, default=1.0)
    ap.add_argument("--min_genes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_markers", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    # The (M,M) attention bias needs the full token set every step; expert-choice
    # gathers a top-k subset, so force the full-token token-choice router when bias
    # is on (keeps adaptive per-token depth, makes the with/without arms apples-to-apples).
    rec_mode = args.recursion_mode
    if args.pathway_attn_bias and rec_mode == "expert":
        rec_mode = "token"
        print("[pathway] pathway_attn_bias on -> recursion_mode=token", flush=True)
    base = RMTConfig(
        heads=(args.task,), n_hvg=None, batch_size=args.batch_size,
        d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
        marker_mode=args.marker_mode, recursion_mode=rec_mode,
        recursion_depth=args.recursion_depth, share_weights=args.share_weights,
        share_strategy=args.share_strategy, n_unique_blocks=args.n_unique_blocks,
        step_cache=args.step_cache, router_type=args.router_type,
        router_temp=args.router_temp, seed=args.seed, epochs=args.epochs,
        patience=args.patience, lr=args.lr, device=args.device,
        gene_interaction=args.gene_interaction, pathway_pool=args.pathway_pool,
        pathway_attn_bias=args.pathway_attn_bias,
        pathway_attn_lambda=args.pathway_attn_lambda,
    )

    for cs in args.channels:
        tag = _result_tag(args.task, cs, base)
        out = args.out / f"{tag}.json"
        if out.exists():
            print(f"[pathway] [skip] {tag} (done)", flush=True)
            continue
        run(args.task, cs, base, args.out, device, min_genes=args.min_genes)
    print(f"\n[pathway] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
