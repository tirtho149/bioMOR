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

"""Early-stopping recursion vs fixed depth, at a deep cap (K=8).

Instead of running every marker token for a fixed K=8 passes, the expert-choice
Mixture-of-Recursions router lets each token *early-exit* the recursion: the router
decides which tokens survive each step, so a token's realised recursion depth is
learned and most tokens stop early. This runner contrasts the two at the same cap:

  * ``fixed8``  -- recursion_mode="fixed",  K=8 : every token runs all 8 passes
                   (realised mean depth = 8).
  * ``early8``  -- recursion_mode="expert", K=8 : per-token early-exit
                   (realised mean depth << 8, far fewer FLOPs).

We report macro-F1, the realised mean recursion depth, and the compute-saving ratio,
across the cohort task and the hard phenotype tasks, over several seeds. Resumable:
one JSON per (task, mode, seed) under ``results_earlystop/``.

    python -m recursive_marker_transformer.earlystop_experiments \
        --tasks cohort pathologic_stage pathologic_T pathologic_N --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RMTConfig

_MODES = {
    "fixed8": dict(recursion_mode="fixed",  recursion_depth=8),
    "early8": dict(recursion_mode="expert", recursion_depth=8),
}
_COHORT = dict(heads=("cancer_type",), n_hvg=4000, d_model=128, d_ff=256,
               n_markers=256, marker_mode="router", epochs=25, patience=25)
_GENONET = dict(heads=("cancer_type",), n_hvg=None, batch_size=32, d_model=128,
                d_ff=256, n_markers=256, marker_mode="router", epochs=40, patience=8)


def _run_cohort(mode, seed):
    from .train import run
    cfg = RMTConfig(**_COHORT, seed=seed, **_MODES[mode])
    r = run(cfg, markers_path="/dev/null")
    h = r["heads"]["cancer_type"]
    return {"macro_f1": h["macro_f1"], "accuracy": h["accuracy"],
            "mean_recursion_depth": r.get("mean_recursion_depth"),
            "compute_saving_ratio": r.get("compute_saving_ratio")}


def _run_hard(task, X, labels, mode, seed, scratch):
    from .genonet_tasks import run_task
    base = RMTConfig(**_GENONET, seed=seed, lr=3e-4, **_MODES[mode])
    r = run_task(task, X, labels, base, scratch)
    h = r["heads"][task]
    return {"macro_f1": h["macro_f1"], "accuracy": h["accuracy"],
            "n_classes": r["n_classes"],
            "mean_recursion_depth": r.get("mean_recursion_depth"),
            "compute_saving_ratio": r.get("compute_saving_ratio")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*",
                    default=["cohort", "pathologic_stage", "pathologic_T", "pathologic_N"])
    ap.add_argument("--modes", nargs="*", default=["fixed8", "early8"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_earlystop"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "_scratch"

    X = labels = None
    if any(t != "cohort" for t in args.tasks):
        from .genonet_tasks import _load_unified
        print(f"[earlystop] loading {args.csv} ...", flush=True)
        X, labels, _ = _load_unified(args.csv)
        print(f"[earlystop] X={X.shape}", flush=True)

    for task in args.tasks:
        for mode in args.modes:
            for s in args.seeds:
                f = args.out / f"{task}__{mode}__seed{s}.json"
                if f.exists():
                    print(f"[skip] {f.name}", flush=True)
                    continue
                print(f"\n######### task={task} mode={mode} seed={s} #########", flush=True)
                rec = (_run_cohort(mode, s) if task == "cohort"
                       else _run_hard(task, X, labels, mode, s, scratch))
                rec.update({"task": task, "mode": mode, "seed": s})
                f.write_text(json.dumps(rec, indent=1, default=float))
                d = rec.get("mean_recursion_depth")
                print(f"[done] {f.name}  macroF1={rec['macro_f1']*100:.2f} "
                      f"depth={d if d is None else round(d,2)}", flush=True)

    print("\n[earlystop] done -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
