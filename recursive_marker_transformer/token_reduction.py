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

"""Token Reduction Validation for SMART.

Quantifies (and qualitatively validates) how effectively the learned marker
router shrinks the input token space (the ~N gene tokens) to the M marker tokens
that the recursive stack actually attends over, while preserving the most
informative features.

The artifact written by this module has four components, mirroring the
validation protocol:

1. ``token_reduction_summary``  -- original / retained / removed token counts,
   reduction ratio + percentage, and the O(N^2)->O(M^2) attention-cost factor.
2. ``token_selection``          -- indices and gene identifiers of the retained
   tokens (and the count / a sample of the discarded ones).
3. ``feature_importance_ranking`` -- a ranked feature list in BOTH descending
   (most->least important) and ascending (least->most important) views; each
   entry is ``{rank, feature, score}``. The full ranking over all N genes is
   also written to ``token_reduction_ranking.csv``.
4. ``selection_metadata``       -- the configuration of the reduction process
   (method, top-k, selection rule / threshold, temperature schedule, and the
   relevant model hyperparameters).

It additionally reports ranking-quality diagnostics: the share of total
importance mass captured by the retained tokens, the mean importance of retained
vs discarded tokens, and the rank correlation between a gene's selection
importance and the compute (MoR recursion depth) it is allocated -- the
qualitative "feature importance consistency" check.

Usage
-----
    python -m recursive_marker_transformer.token_reduction \
        --config results/main.json --out results --top 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import fields

import numpy as np
import torch

from .config import RMTConfig
from .train import run


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def _cfg_from_json(path: str) -> RMTConfig:
    """Reconstruct the exact headline config persisted by ``train.run``."""
    with open(path) as f:
        blob = json.load(f)
    raw = blob.get("config", blob)
    valid = {f.name for f in fields(RMTConfig)}
    kw = {k: v for k, v in raw.items() if k in valid}
    for seq in ("heads", "cohorts"):
        if isinstance(kw.get(seq), list):
            kw[seq] = tuple(kw[seq])
    return RMTConfig(**kw)


def _headline_cfg() -> RMTConfig:
    """The headline cohort config (used when no JSON is supplied)."""
    return RMTConfig(
        heads=("cancer_type",), n_hvg=4000, d_model=128, d_ff=256,
        n_markers=256, marker_mode="router", compress_mode="aggregate",
        recursive_marker_refine=True, recursion_depth=4, share_weights=True,
        recursion_mode="expert", epochs=25, patience=25, seed=42,
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _per_gene_importance(model, n_genes: int) -> np.ndarray:
    """Continuous per-gene importance = max marker-query affinity over the M
    query slots.

    At eval the router collapses each slot to a one-hot arg-max, so its *weights*
    are a binary selected/not mask. For a graded ranking over all N genes we use
    the pre-hardening affinity (query.key) instead, which is continuous and does
    not saturate. Falls back to the selection weight / marker head if no router.
    """
    model.eval()
    with torch.no_grad():
        ident = model.embed.gene_identity()
        sel = getattr(model, "selector", None)
        if sel is not None and hasattr(sel, "_logits"):
            scores = sel._logits(ident).max(dim=0).values          # continuous affinity
        elif sel is not None:
            scores = sel.weights(ident).float().max(dim=0).values
        elif getattr(model, "marker", None) is not None and getattr(model.marker, "head", None) is not None:
            scores = model.marker.head(ident)
        elif getattr(model, "gene_variance", None) is not None:
            scores = model.gene_variance
        else:
            scores = torch.zeros(n_genes)
    return scores.detach().cpu().numpy().astype(float)[:n_genes]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (scipy if present, else a numpy fallback)."""
    if len(a) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(a, b).correlation
        return float(rho)
    except Exception:
        ar = np.argsort(np.argsort(a)).astype(float)
        br = np.argsort(np.argsort(b)).astype(float)
        ar -= ar.mean(); br -= br.mean()
        denom = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
        return float((ar * br).sum() / denom) if denom else float("nan")


# --------------------------------------------------------------------------- #
# build the validation artifact
# --------------------------------------------------------------------------- #
def build_validation(results: dict, internals: dict, top: int = 50) -> tuple[dict, list]:
    model = internals["model"]
    data = internals["data"]
    marker_idx = internals["marker_idx"].detach().cpu().numpy().astype(int)
    slot_depth = internals["mean_slot_depth"].detach().cpu().numpy().astype(float)
    cfg = model.cfg if hasattr(model, "cfg") else None

    N = int(data.n_genes)
    genes = list(data.gene_names)
    scores = _per_gene_importance(model, N)

    # ---- retained vs discarded -------------------------------------------- #
    M_slots = int(len(marker_idx))                       # marker tokens the stack attends over
    retained = sorted(set(int(i) for i in marker_idx))   # distinct genes selected
    retained_set = set(retained)
    discarded = [i for i in range(N) if i not in retained_set]
    n_ret_genes = len(retained)
    removed = N - n_ret_genes

    # gene -> recursion depth (only selected slots have a depth)
    gene_depth = {}
    for g, d in zip(marker_idx.tolist(), slot_depth.tolist()):
        gene_depth.setdefault(int(g), []).append(float(d))
    gene_depth = {g: float(np.mean(v)) for g, v in gene_depth.items()}

    # ---- 1. summary ------------------------------------------------------- #
    summary = {
        "original_tokens": N,
        "marker_tokens": M_slots,
        "retained_tokens": n_ret_genes,
        "removed_tokens": removed,
        "retention_ratio": round(M_slots / N, 6),
        "reduction_ratio": round(removed / N, 6),
        "reduction_percentage": round(100.0 * removed / N, 4),
        "attention_cost_factor": round((N / max(M_slots, 1)) ** 2, 2),  # O(N^2)/O(M^2)
    }

    # ---- 2. token selection ----------------------------------------------- #
    selection = {
        "retained_indices": retained,
        "retained_features": [genes[i] for i in retained],
        "n_retained_features": n_ret_genes,
        "n_discarded_features": removed,
        "discarded_indices_sample": discarded[:50],
    }

    # ---- 3. feature importance ranking ------------------------------------ #
    order_desc = np.argsort(-scores, kind="stable")
    full_ranking = [
        {"rank": r + 1, "feature": genes[i], "score": round(float(scores[i]), 6),
         "selected": int(i in retained_set),
         "recursion_depth": round(gene_depth[int(i)], 4) if int(i) in gene_depth else None}
        for r, i in enumerate(order_desc)
    ]
    descending = [{"rank": e["rank"], "feature": e["feature"], "score": e["score"]}
                  for e in full_ranking[:top]]
    ascending = [{"rank": j + 1, "feature": full_ranking[-(j + 1)]["feature"],
                  "score": full_ranking[-(j + 1)]["score"]}
                 for j in range(min(top, len(full_ranking)))]
    ranking = {"top_k": top, "descending": descending, "ascending": ascending,
               "full_ranking_csv": "token_reduction_ranking.csv"}

    # ---- 4. selection metadata -------------------------------------------- #
    cd = cfg.as_dict() if cfg is not None else results.get("config", {})
    metadata = {
        "method": "cross-attention marker router (recursive marker selection)",
        "marker_mode": cd.get("marker_mode"),
        "compress_mode": cd.get("compress_mode"),
        "recursive_marker_refine": cd.get("recursive_marker_refine"),
        "importance_score": "max marker-query affinity (query.key) over the M slots (pre-hardening, continuous)",
        "top_k_markers": cd.get("n_markers"),
        "n_query_slots": M_slots,
        "selection_rule": "hard arg-max gene per query slot at eval (soft over all genes during training)",
        "temperature_anneal": "Concrete/Gumbel temperature annealed high->low across epochs",
        "threshold": "none (top-k by slot arg-max; no score cutoff)",
        "n_hvg": cd.get("n_hvg"),
        "d_model": cd.get("d_model"),
        "recursion_depth": cd.get("recursion_depth"),
        "recursion_mode": cd.get("recursion_mode"),
        "share_weights": cd.get("share_weights"),
        "seed": cd.get("seed"),
    }

    # ---- ranking quality / consistency ------------------------------------ #
    total_mass = float(scores.sum())
    ret_mass = float(scores[retained].sum()) if retained else 0.0
    ret_scores = scores[retained] if retained else np.array([0.0])
    dis_scores = scores[discarded] if discarded else np.array([0.0])
    sel_genes = sorted(gene_depth.keys())
    consistency = {
        "importance_mass_retained": round(ret_mass / total_mass, 6) if total_mass else None,
        "mean_importance_retained": round(float(ret_scores.mean()), 6),
        "mean_importance_discarded": round(float(dis_scores.mean()), 6),
        "spearman_importance_vs_recursion_depth": round(
            _spearman(np.array([scores[g] for g in sel_genes]),
                      np.array([gene_depth[g] for g in sel_genes])), 4) if len(sel_genes) > 1 else None,
        "note": ("ascending and descending views are exact reverses of one ranking; "
                 "the depth correlation checks that higher-importance markers receive "
                 "more recursive compute."),
    }

    head = list(results.get("heads", {}).values())
    acc = head[0].get("accuracy") if head else None
    macro = head[0].get("macro_f1") if head else None

    artifact = {
        "task": cd.get("heads", ["cancer_type"])[0] if isinstance(cd.get("heads"), list) else "cancer_type",
        "test_accuracy": acc,
        "test_macro_f1": macro,
        "token_reduction_summary": summary,
        "token_selection": selection,
        "feature_importance_ranking": ranking,
        "selection_metadata": metadata,
        "ranking_consistency": consistency,
    }
    return artifact, full_ranking


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Token Reduction Validation for SMART")
    ap.add_argument("--config", default="results/main.json",
                    help="headline result JSON to read the exact config from "
                         "(falls back to the built-in headline config)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--top", type=int, default=50,
                    help="how many entries to keep in the JSON ascending/descending views")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.config and os.path.exists(args.config):
        cfg = _cfg_from_json(args.config)
        print(f"[token-reduction] config from {args.config}")
    else:
        cfg = _headline_cfg()
        print("[token-reduction] using built-in headline config")
    if args.device:
        from dataclasses import replace
        cfg = replace(cfg, device=args.device)

    results, internals = run(
        cfg, markers_path=os.path.join(args.out, "markers_token_reduction.csv"),
        return_internals=True)

    artifact, full_ranking = build_validation(results, internals, top=args.top)

    json_path = os.path.join(args.out, "token_reduction.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)

    csv_path = os.path.join(args.out, "token_reduction_ranking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "score", "selected", "recursion_depth"])
        for e in full_ranking:
            w.writerow([e["rank"], e["feature"], e["score"], e["selected"],
                        "" if e["recursion_depth"] is None else e["recursion_depth"]])

    s = artifact["token_reduction_summary"]
    c = artifact["ranking_consistency"]
    print("\n[token-reduction] ===== SUMMARY =====")
    print(f"  original tokens (genes) : {s['original_tokens']}")
    print(f"  marker tokens (kept)    : {s['marker_tokens']}  "
          f"({n_pct(s['retention_ratio'])} of input)")
    print(f"  removed tokens          : {s['removed_tokens']}  "
          f"({s['reduction_percentage']}% reduction)")
    print(f"  attention cost factor   : {s['attention_cost_factor']}x  (O(N^2)/O(M^2))")
    print(f"  importance mass retained: {n_pct(c['importance_mass_retained'])}")
    print(f"  mean importance ret/dis : {c['mean_importance_retained']} / {c['mean_importance_discarded']}")
    print(f"  rho(importance, depth)  : {c['spearman_importance_vs_recursion_depth']}")
    print(f"[token-reduction] wrote {json_path}")
    print(f"[token-reduction] wrote {csv_path}")


def n_pct(x):
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


if __name__ == "__main__":
    main()
