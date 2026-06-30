# ============================================================================
# SMART -- MoR Table 9 analogue (WARM-START / uptraining) on the pathway/P-NET
# multi-omics cohorts. The single-cell version lives in warmstart.py; this is the
# bulk-omics twin so the warm-start table reports the P-NET cohorts alongside the
# genomap single-cell datasets.
#
# Protocol (mirrors warmstart.py): train a fixed-depth SMART, initialise the shared
# recursive block of an expert-choice MoR SMART from those weights and continue-train
# -- vs training the MoR model from scratch. Reports the uptraining gain.
#     python -m recursive_marker_transformer.pathway_warmstart --tasks prostate blca stad panmeta_subtype
# ============================================================================
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .config import RMTConfig
from .pathway_data import load_cohort, load_pan_meta
from .pathway_tasks import PANMETA, _fit_eval
from .train import resolve_device


def _splits(y, seed):
    idx = np.arange(len(y))
    _, cnt = np.unique(y, return_counts=True)
    strat = y if cnt.min() >= 2 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    _, cnt = np.unique(y[tr], return_counts=True)
    strat = y[tr] if cnt.min() >= 2 else None
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=strat)
    return tr, va, te


# default modality per cohort (matches run_pathway.sbatch headline config)
_CHANNELS = {"prostate": "mut_cnv", "blca": "mut_cnv", "stad": "mut_cnv",
             "brca": "mut", "panmeta_response": "expr", "panmeta_subtype": "expr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_pathway_warmstart"))
    ap.add_argument("--tasks", nargs="*",
                    default=["prostate", "blca", "stad", "panmeta_subtype"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_markers", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_genes", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        cs = _CHANNELS.get(task, "mut_cnv")
        bs = 128 if cs == "expr" else args.batch_size
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if task in PANMETA:
            cohort_dir, label = PANMETA[task]
            coh = load_pan_meta(label=label, cohort=cohort_dir, min_genes=args.min_genes)
        else:
            coh = load_cohort(task, channels=cs, min_genes=args.min_genes)
        X, y = coh.X, coh.y
        G = X.shape[1]
        K = int(y.max() + 1)
        C = 1 if X.ndim == 2 else X.shape[2]
        tr, va, te = _splits(y, args.seed)
        dtypes = {task: "multiclass"}

        # pathway tokens + Reactome prior = the headline cohort configuration
        common = dict(heads=(task,), n_hvg=None, n_channels=C, batch_size=bs,
                      d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
                      marker_mode="pathway", gene_interaction="reactome",
                      pathway_pool=("sum" if task == "brca" else "mean"),
                      recursion_depth=4, share_weights=True, seed=args.seed,
                      epochs=args.epochs, patience=args.patience, lr=args.lr,
                      weight_decay=1e-5, device=args.device)
        fixed_cfg = replace(RMTConfig(recursion_mode="fixed", **common))
        mor_cfg = replace(RMTConfig(recursion_mode="expert", **common))

        print(f"\n########## warmstart {task} [{cs}]: G={G} C={C} K={K} ##########", flush=True)

        yt, yp, fixed_model, _ = _fit_eval(task, coh, X, y, tr, va, te, fixed_cfg, G, K, dtypes, device)
        f1_fixed = f1_score(yt, yp, average="macro")
        init = {k: v.detach().cpu().clone()
                for k, v in fixed_model.stack.blocks[0].state_dict().items()}

        yt, yp, _, _ = _fit_eval(task, coh, X, y, tr, va, te, mor_cfg, G, K, dtypes, device)
        f1_scratch = f1_score(yt, yp, average="macro")

        yt, yp, _, _ = _fit_eval(task, coh, X, y, tr, va, te, mor_cfg, G, K, dtypes, device,
                                 init_block=init)
        f1_warm = f1_score(yt, yp, average="macro")

        res = {"task": task, "channel_set": cs,
               "fixed_source_macro_f1": 100 * f1_fixed,
               "mor_from_scratch_macro_f1": 100 * f1_scratch,
               "mor_warm_start_macro_f1": 100 * f1_warm,
               "warm_start_gain": 100 * (f1_warm - f1_scratch)}
        (args.out / f"{task}.json").write_text(json.dumps(res, indent=1))
        print(f"  [warmstart] {task}: fixed={100*f1_fixed:.1f} scratch={100*f1_scratch:.1f} "
              f"warm={100*f1_warm:.1f} gain={100*(f1_warm-f1_scratch):+.1f}", flush=True)


if __name__ == "__main__":
    main()
