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

"""Train / evaluate the Recursive Marker Transformer on bundled bulk TCGA data.

Usage:
    python -m recursive_marker_transformer.train [key=value ...]
    e.g.  python -m recursive_marker_transformer.train epochs=2 n_hvg=1000

Reports test accuracy + macro-F1 per head, transformer vs total parameter
counts (the weight-sharing claim), and an approximate FLOP estimate. Writes the
top markers (by learned importance) to ``markers_top.csv`` for the biology
ablation.
"""

from __future__ import annotations

import csv
import sys
import time
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import RMTConfig
from .data import build_data
from .losses import RMTLoss
from .model import RecursiveMarkerTransformer


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collect_train_targets(loader, heads):
    acc = {h: [] for h in heads}
    for _xb, yb in loader:
        for h in heads:
            acc[h].append(yb[h])
    return {h: torch.cat(v) for h, v in acc.items()}


def _class_weights(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets.long(), minlength=n_classes).float()
    w = counts.sum() / (n_classes * counts.clamp(min=1))
    return w


@torch.no_grad()
def evaluate(model, loader, device, head_dtypes) -> Dict[str, tuple]:
    model.eval()
    preds = {h: [] for h in head_dtypes}
    trues = {h: [] for h in head_dtypes}
    for xb, yb in loader:
        out = model(xb.to(device))
        for h, logit in out["logits"].items():
            if head_dtypes[h] == "binary":
                p = (torch.sigmoid(logit.squeeze(-1)) > 0.5).long().cpu()
            else:
                p = logit.argmax(-1).cpu()
            preds[h].append(p)
            trues[h].append(yb[h])
    return {h: (torch.cat(trues[h]).numpy(), torch.cat(preds[h]).numpy()) for h in head_dtypes}


@torch.no_grad()
def _depth_stats(model, loader, device, cfg):
    """Aggregate per-token recursion depth over a loader.

    Returns ``(mean_slot_depth (M,), marker_idx (M,), active_per_step (K,))``:
    the mean recursion depth of each marker slot (its intrinsic importance) and
    the mean number of tokens still active at each recursion step (for the
    token-count-aware FLOP estimate). Marker selection is batch-independent at
    eval, so ``marker_idx`` is taken once.
    """
    model.eval()
    K = cfg.recursion_depth
    slot_sum = None
    marker_idx = None
    active_per_step = torch.zeros(K)
    n = 0
    for xb, _yb in loader:
        out = model(xb.to(device))
        d = out["recursion_depth_per_token"].detach().cpu()        # (B, M)
        if slot_sum is None:
            slot_sum = d.sum(dim=0)
            marker_idx = out["marker_idx"].detach().cpu()
        else:
            slot_sum = slot_sum + d.sum(dim=0)
        n += d.shape[0]
        for t in range(K):
            active_per_step[t] += (d > t).float().sum()
    mean_slot_depth = slot_sum / max(n, 1)
    active_per_step = active_per_step / max(n, 1)
    return mean_slot_depth, marker_idx, active_per_step


# Global raw TCGA cancer_type codes -> human cohort label (LaTeX-safe, no ``&``).
# genomic_dataloader.build_loaders indexes cohorts by enumerate(sorted(cohorts));
# for the four loaded cohorts that is breast=0, head_neck=1, lung=2, thyroid=3
# (there is NO prostate cohort in this dataset). The 5-cohort build_unified.py uses
# a different order (3=prostate, 4=thyroid) and must not be used for this loader.
_CANCER_RAW_NAMES = {
    0: "Breast (BRCA)", 1: "Head--neck (HNSC)", 2: "Lung (LUNG)",
    3: "Thyroid (THCA)",
}


def _class_names(head, data):
    """contiguous-class-index (as str) -> cohort label, for the cancer_type head.
    Inverts the train-fit label map (raw code -> contiguous idx). Returns None for
    heads without a known naming."""
    if head != "cancer_type":
        return None
    lm = getattr(data, "label_maps", {}).get(head)
    if not lm:
        return None
    return {str(idx): _CANCER_RAW_NAMES.get(int(raw), f"cohort {raw}")
            for raw, idx in lm.items()}


def run(cfg: RMTConfig, markers_path: str = "markers_top.csv",
        return_internals: bool = False):
    """Train + evaluate the model.

    Returns the results ``dict`` by default. When ``return_internals=True`` it
    returns ``(results, internals)`` where ``internals`` exposes the trained
    ``model``, the ``data`` bundle, the selected ``marker_idx`` and the per-slot
    ``mean_slot_depth`` so callers (e.g. the token-reduction validation) can
    inspect the learned selection without re-training.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[rmt] device={device} | config={cfg.as_dict()}")

    t0 = time.time()
    data = build_data(cfg)
    print(f"[rmt] data ready in {time.time()-t0:.1f}s | n_genes={data.n_genes} "
          f"| heads={dict(data.head_n_classes)}")

    train_targets = _collect_train_targets(data.loaders["train"], cfg.heads)

    model = RecursiveMarkerTransformer(
        cfg, data.n_genes, data.head_n_classes, data.head_dtypes).to(device)
    model.set_gene_variance(torch.from_numpy(data.gene_variance))

    # Biology-informed router: build the genomap gene-gene-interaction centrality
    # prior on the train split (label-free) and install it on the model.
    if getattr(cfg, "gene_interaction", None) not in (None, "none"):
        from .interaction import build_interaction
        t1 = time.time()
        inter = build_interaction(data.loaders["train"], data.n_genes,
                                  mode=cfg.gene_interaction, knn=cfg.interaction_knn,
                                  seed=cfg.seed)
        model.set_gene_interaction(inter.centrality)
        print(f"[rmt] gene_interaction={cfg.gene_interaction} prior built in "
              f"{time.time()-t1:.1f}s (beta0={cfg.router_prior_beta}, "
              f"anneal={cfg.router_prior_anneal})")

    # Class weights for multiclass heads (inverse frequency on train split).
    class_weights = {}
    for h in cfg.heads:
        if data.head_dtypes[h] == "multiclass":
            class_weights[h] = _class_weights(train_targets[h], data.head_n_classes[h]).to(device)
    criterion = RMTLoss(cfg, data.head_dtypes, class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear LR warmup (1%->100% over first ~10% of epochs) then cosine; prevents the
    # wide-model collapse-to-majority-class seen without warmup.
    _warm = max(1, round(0.1 * cfg.epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=_warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - _warm))],
        milestones=[_warm])

    primary = cfg.heads[0]
    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        model.set_anneal(epoch / max(cfg.epochs - 1, 1))   # Concrete temperature schedule
        agg = {}
        for xb, yb in data.loaders["train"]:
            xb = xb.to(device)
            yb = {h: v.to(device) for h, v in yb.items()}
            out = model(xb)
            losses = criterion(out, yb)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        sched.step()

        val = evaluate(model, data.loaders["val"], device, data.head_dtypes)
        yt, yp = val[primary]
        val_f1 = f1_score(yt, yp, average="macro")
        n_batches = len(data.loaders["train"])
        print(f"[rmt] epoch {epoch+1:2d}/{cfg.epochs} | "
              + " ".join(f"{k}={agg[k]/n_batches:.4f}" for k in ("total", "task", "marker", "diversity"))
              + f" | val_{primary}_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[rmt] early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test report ---------------------------------------------------
    test = evaluate(model, data.loaders["test"], device, data.head_dtypes)
    results = {"config": cfg.as_dict(), "heads": {}}
    print("\n[rmt] ===== TEST =====")
    for h in cfg.heads:
        yt, yp = test[h]
        acc = accuracy_score(yt, yp)
        macro = f1_score(yt, yp, average="macro")
        weighted = f1_score(yt, yp, average="weighted")
        # Per-category (per-class) report for EVERY head, separately.
        per_class = classification_report(yt, yp, zero_division=0, output_dict=True)
        results["heads"][h] = {
            "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
            "per_class": per_class,
        }
        # Persist contiguous-index -> human label so the paper tables read by
        # cohort name, not "0/1/2/3" (LaTeX-safe strings; no ``&``).
        cn = _class_names(h, data)
        if cn:
            results["heads"][h]["class_names"] = cn
        print(f"\n  [head: {h}] accuracy={acc:.4f} macro_f1={macro:.4f} weighted_f1={weighted:.4f}")
        print(classification_report(yt, yp, zero_division=0))

    # ---- efficiency numbers -------------------------------------------
    tf_params = model.transformer_param_count()
    total_params = model.total_param_count()
    M = min(cfg.n_markers, data.n_genes)

    # Per-token recursion depth on the test set -> token-aware FLOPs + the
    # recursion-depth importance ranking (MoR interpretability).
    mean_slot_depth, marker_idx, active = _depth_stats(
        model, data.loaders["test"], device, cfg)

    def _step_flops(a):    # attention is O(a^2); FFN is O(a)
        return 4.0 * a * a * cfg.d_model + 4.0 * a * cfg.d_model * cfg.d_ff

    flops_nominal = cfg.recursion_depth * _step_flops(M)           # fixed-depth baseline
    flops_eff = float(sum(_step_flops(float(active[t])) for t in range(cfg.recursion_depth)))
    saving = flops_eff / flops_nominal if flops_nominal else 1.0

    results["transformer_params"] = tf_params
    results["total_params"] = total_params
    results["approx_flops_per_sample"] = int(round(flops_eff))
    results["approx_flops_nominal"] = int(round(flops_nominal))
    results["compute_saving_ratio"] = saving
    results["mean_recursion_depth"] = float(mean_slot_depth.mean())
    results["active_tokens_per_step"] = [float(a) for a in active]
    print(f"\n[rmt] transformer params={tf_params:,} | total params={total_params:,}")
    print(f"[rmt] mode={cfg.recursion_mode} | mean recursion depth={results['mean_recursion_depth']:.2f}"
          f" | active/step={[round(float(a),1) for a in active]}")
    print(f"[rmt] effective FLOPs/sample={results['approx_flops_per_sample']:,} "
          f"(={saving:.2f}x of fixed-depth {flops_nominal:,.0f})")
    print(f"[rmt] share_weights={cfg.share_weights} -> "
          f"{'shared block x K' if cfg.share_weights else 'K independent blocks'}")

    # gene -> mean recursion depth (only the M selected markers have a depth).
    gene_depth = {int(g): float(d) for g, d in zip(marker_idx.tolist(),
                                                   mean_slot_depth.tolist())}
    _save_markers(model, data, cfg, path=markers_path, gene_depth=gene_depth)
    if return_internals:
        internals = {
            "model": model,
            "data": data,
            "marker_idx": marker_idx,
            "mean_slot_depth": mean_slot_depth,
            "device": device,
        }
        return results, internals
    return results


def _save_markers(model, data, cfg, path="markers_top.csv", top=200, gene_depth=None):
    gene_depth = gene_depth or {}
    model.eval()
    with torch.no_grad():
        ident = model.embed.gene_identity()
        if getattr(model, "selector", None) is not None:
            # per-gene max selection weight across the M selectors (concrete/router)
            scores = model.selector.weights(ident).max(dim=0).values
        elif cfg.marker_mode == "learnable" and model.marker.head is not None:
            scores = model.marker.head(ident)
        elif cfg.marker_mode == "variance":
            scores = model.gene_variance
        else:
            scores = torch.zeros(data.n_genes)
    order = torch.argsort(scores, descending=True)[:top].cpu().numpy()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # recursion_depth: mean MoR depth this gene's marker slot received
        # (blank if the gene was not selected as a marker).
        w.writerow(["rank", "gene", "importance", "recursion_depth"])
        for r, i in enumerate(order):
            depth = gene_depth.get(int(i), "")
            depth = f"{depth:.3f}" if depth != "" else ""
            w.writerow([r + 1, data.gene_names[i], float(scores[i]), depth])
    print(f"[rmt] wrote top-{top} markers -> {path}")


def _parse_overrides(argv):
    kw = {}
    for tok in argv:
        if "=" not in tok:
            raise SystemExit(f"Expected key=value, got: {tok!r}")
        k, v = tok.split("=", 1)
        kw[k] = v
    return kw


def main():
    cfg = RMTConfig.from_overrides(**_parse_overrides(sys.argv[1:]))
    run(cfg)


if __name__ == "__main__":
    main()
