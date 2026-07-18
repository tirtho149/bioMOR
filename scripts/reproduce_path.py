"""Reproduce the PATH paper's evaluation protocol/metric on our cohorts.

PATH reports POSITIVE-CLASS (progression: metastatic / late-stage) precision,
recall and F1 -- NOT macro-F1 -- over 5-fold CV x 2 seeds (10 folds). Our Table-2
harness reports macro-F1, which is a different (harder) number on these imbalanced
cohorts. This script runs OUR model under PATH's exact protocol and prints both the
positive-class metrics (to compare against PATH's baseline bars) and macro-F1.

Usage:
  python reproduce_path.py --task blca --arm vanilla --seeds 42 43 --device cuda
"""
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path (script lives in scripts/)
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.pathway_tasks import _fit_eval, load_cohort, load_pan_meta, PANMETA
from recursive_marker_transformer.cv import cv_folds, VAL_FRAC
from recursive_marker_transformer.train import resolve_device

# Table-2 ladder rows (multi-omics general block). marker_mode=pathway, reactome prior
# throughout; rows differ by recursion mode / depth / weight sharing.
_P = dict(marker_mode="pathway", gene_interaction="reactome")
ARMS = {
    # Vanilla = independent (untied) 4-block stack
    "vanilla":   dict(_P, recursion_mode="expert", share_weights=False, K=4),
    # Recursive = weight-shared, fixed depth
    "fixed_k2":  dict(_P, recursion_mode="fixed",  share_weights=True,  K=2),
    "fixed_k3":  dict(_P, recursion_mode="fixed",  share_weights=True,  K=3),
    "fixed_k4":  dict(_P, recursion_mode="fixed",  share_weights=True,  K=4),
    # MoR-general expert-choice routing
    "expert_k2": dict(_P, recursion_mode="expert", share_weights=True,  K=2),
    "expert_k3": dict(_P, recursion_mode="expert", share_weights=True,  K=3),
    "expert_k4": dict(_P, recursion_mode="expert", share_weights=True,  K=4),
    # MoR-general token-choice routing
    "token_k2":  dict(_P, recursion_mode="token",  share_weights=True,  K=2),
    "token_k3":  dict(_P, recursion_mode="token",  share_weights=True,  K=3),
    "token_k4":  dict(_P, recursion_mode="token",  share_weights=True,  K=4),
    # aliases
    "biomor":    dict(_P, recursion_mode="expert", share_weights=True,  K=4),
    "token":     dict(_P, recursion_mode="token",  share_weights=True,  K=4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", default="vanilla", choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--channels", default="mut_cnv")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--pathway_pool", default="mean", choices=["mean", "sum"])
    ap.add_argument("--pos_class", type=int, default=None,
                    help="index of the progression/positive class; default=minority")
    ap.add_argument("--select", choices=["macro", "pos"], default="macro",
                    help="val model-selection metric: 'macro' (Table-2 default) or "
                         "'pos' (positive-class F1, aligned to PATH's reported metric)")
    ap.add_argument("--path_protocol", action="store_true",
                    help="PATH's exact eval (binary): AUROC model selection + "
                         "validation-tuned decision threshold maximising positive-class F1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/repro"))
    args = ap.parse_args()

    out_json = args.out / f"{args.task}__{args.arm}.json"
    if out_json.exists():                      # skip-on-exist -> safe to fan out over both partitions
        print(f"[repro] [skip] {out_json} already done", flush=True); return

    import torch
    device = resolve_device(args.device)
    a = ARMS[args.arm]
    if args.task in PANMETA:
        cohort_dir, label = PANMETA[args.task]
        coh = load_pan_meta(label=label, cohort=cohort_dir)
    else:
        coh = load_cohort(args.task, channels=args.channels)
    X, y = coh.X, coh.y
    G = X.shape[1]; K = int(y.max() + 1); C = 1 if X.ndim == 2 else X.shape[2]
    dtypes = {args.task: "multiclass"}

    # positive = the progression class. Default: the MINORITY class (metastatic/late
    # is the rare, clinically-important label PATH highlights).
    _, cnt = np.unique(y, return_counts=True)
    pos = args.pos_class if args.pos_class is not None else int(np.argmin(cnt))
    print(f"[repro] task={args.task} arm={args.arm} N={len(y)} G={G} C={C} K={K} "
          f"pos_class={pos} class_counts={dict(enumerate(cnt.tolist()))} "
          f"seeds={args.seeds} folds={args.cv_folds}", flush=True)

    per_pos, per_maj, macro, accs = [], [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = replace(RMTConfig(), heads=(args.task,), n_hvg=None, n_channels=C,
                      d_model=args.d_model, d_ff=2 * args.d_model, marker_mode=a["marker_mode"],
                      recursion_mode=a["recursion_mode"], gene_interaction=a["gene_interaction"],
                      share_weights=a["share_weights"], recursion_depth=a.get("K", 4),
                      pathway_pool=args.pathway_pool,
                      dropout=args.dropout, weight_decay=args.weight_decay, lr=3e-4,
                      epochs=args.epochs, patience=args.patience, batch_size=32, seed=seed)
        for fi, (tr, va, te) in enumerate(cv_folds(y, n_folds=args.cv_folds, seed=seed, val_frac=VAL_FRAC)):
            # selection metric: macro (Table-2 default) or positive-class F1 (PATH-aligned);
            # --path_protocol adds AUROC selection + validation-tuned threshold (PATH's recipe)
            yt, yp, *_ = _fit_eval(args.task, coh, X, y, tr, va, te, cfg, G, K, dtypes, device,
                                   sel_pos=(pos if (args.select == "pos" or args.path_protocol) else None),
                                   path_protocol=args.path_protocol)
            p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=list(range(K)),
                                                         average=None, zero_division=0)
            per_pos.append((p[pos], r[pos], f[pos]))
            maj = int(np.argmax(cnt)); per_maj.append((p[maj], r[maj], f[maj]))
            macro.append(f1_score(yt, yp, average="macro"))
            accs.append(accuracy_score(yt, yp))
            print(f"  seed{seed} fold{fi+1}: pos(F1={f[pos]:.3f} P={p[pos]:.3f} R={r[pos]:.3f}) "
                  f"macroF1={macro[-1]:.3f}", flush=True)

    def ms(a, i): v = np.array([x[i] for x in a]); return v.mean(), v.std()
    res = {
        "task": args.task, "arm": args.arm, "pos_class": pos, "n_folds": len(macro),
        "pos_f1": ms(per_pos, 2), "pos_precision": ms(per_pos, 0), "pos_recall": ms(per_pos, 1),
        "maj_f1": ms(per_maj, 2),
        "macro_f1": (float(np.mean(macro)), float(np.std(macro))),
        "accuracy": (float(np.mean(accs)), float(np.std(accs))),
    }
    print("\n================ PATH-protocol result ================")
    print(f"  POSITIVE-class (idx {pos}) F1 = {res['pos_f1'][0]*100:.1f} ± {res['pos_f1'][1]*100:.1f}"
          f"  (P={res['pos_precision'][0]:.2f} R={res['pos_recall'][0]:.2f})")
    print(f"  majority-class  F1 = {res['maj_f1'][0]*100:.1f}")
    print(f"  macro-F1           = {res['macro_f1'][0]*100:.1f} ± {res['macro_f1'][1]*100:.1f}")
    print(f"  accuracy           = {res['accuracy'][0]*100:.1f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}__{args.arm}.json").write_text(json.dumps(res, indent=1, default=list))


if __name__ == "__main__":
    main()
