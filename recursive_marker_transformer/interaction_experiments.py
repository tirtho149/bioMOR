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

"""Biology-informed-router ablation: does the genomap gene-gene interaction prior
on the depth router help, and is it the *real* co-expression structure that matters?

For each task we sweep the routing prior over three modes -- ``none`` (baseline
SMART router), ``coexpr`` (genomap correlation-graph centrality prior), and
``random`` (degree-matched random-graph control) -- across several seeds, and report
macro-F1 mean+/-std. ``coexpr`` beating ``random`` (which approximates ``none``) is
the evidence that biological co-expression structure, not "any bias", drives any
gain. Per the honest framing, the prior is expected to matter most on the
low-signal hard phenotype tasks (stage / T / N), not the near-saturated cohort task.

Resumable: one JSON per (task, mode, seed) under ``results_interaction/``.

    python -m recursive_marker_transformer.interaction_experiments \
        --tasks cohort pathologic_stage pathologic_T pathologic_N \
        --modes none coexpr random --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import RMTConfig

# headline cohort config (matches results/main.json scale)
_COHORT = dict(heads=("cancer_type",), n_hvg=4000, d_model=128, d_ff=256,
               n_markers=256, marker_mode="router", recursion_mode="expert",
               recursion_depth=4, epochs=25, patience=25)

# hard genoNet phenotype tasks (all genes) -- the low-signal regime
_HARD_TASKS = ("pathologic_stage", "pathologic_T", "pathologic_N",
               "os_binary", "tumor_status")
_GENONET = dict(heads=("cancer_type",), n_hvg=None, batch_size=32, d_model=128,
                d_ff=256, n_markers=256, marker_mode="router",
                recursion_mode="expert", recursion_depth=4, epochs=40, patience=8)


def _run_cohort(mode, seed, beta, knn, anneal):
    from .train import run
    cfg = RMTConfig(**_COHORT, seed=seed, gene_interaction=mode,
                    router_prior_beta=beta, interaction_knn=knn,
                    router_prior_anneal=anneal)
    r = run(cfg, markers_path="/dev/null")
    h = r["heads"]["cancer_type"]
    return {"macro_f1": h["macro_f1"], "accuracy": h["accuracy"],
            "transformer_params": r.get("transformer_params"),
            "mean_recursion_depth": r.get("mean_recursion_depth")}


def _run_hard(task, X, labels, mode, seed, beta, knn, anneal, scratch):
    from .genonet_tasks import run_task
    base = RMTConfig(**_GENONET, seed=seed, lr=3e-4, gene_interaction=mode,
                     router_prior_beta=beta, interaction_knn=knn,
                     router_prior_anneal=anneal)
    r = run_task(task, X, labels, base, scratch)
    h = r["heads"][task]
    return {"macro_f1": h["macro_f1"], "accuracy": h["accuracy"],
            "n_classes": r["n_classes"],
            "transformer_params": r.get("transformer_params"),
            "mean_recursion_depth": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*",
                    default=["cohort", "pathologic_stage", "pathologic_T", "pathologic_N"])
    ap.add_argument("--modes", nargs="*", default=["none", "coexpr", "random"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--anneal", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_interaction"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "_scratch"

    # Load the unified table once if any hard task is requested.
    X = labels = None
    if any(t != "cohort" for t in args.tasks):
        from .genonet_tasks import _load_unified
        print(f"[inter] loading {args.csv} ...", flush=True)
        X, labels, _ = _load_unified(args.csv)
        print(f"[inter] X={X.shape}", flush=True)

    for task in args.tasks:
        for mode in args.modes:
            for s in args.seeds:
                f = args.out / f"{task}__{mode}__seed{s}.json"
                if f.exists():
                    print(f"[skip] {f.name}", flush=True)
                    continue
                print(f"\n######### task={task} mode={mode} seed={s} "
                      f"beta={args.beta} anneal={args.anneal} #########", flush=True)
                if task == "cohort":
                    rec = _run_cohort(mode, s, args.beta, args.knn, args.anneal)
                else:
                    rec = _run_hard(task, X, labels, mode, s, args.beta, args.knn,
                                    args.anneal, scratch)
                rec.update({"task": task, "mode": mode, "seed": s,
                            "beta": args.beta, "knn": args.knn, "anneal": args.anneal})
                f.write_text(json.dumps(rec, indent=1, default=float))
                print(f"[done] {f.name}  macroF1={rec['macro_f1']*100:.2f} "
                      f"acc={rec['accuracy']*100:.2f}", flush=True)

    print("\n[inter] done -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
