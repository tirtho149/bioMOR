# ============================================================================
# SMART -- formal recursion-depth selection ("early-stopping over depth").
#
# We sweep the (fixed) recursion depth K over a grid up to 100 and select the best
# depth the way early stopping selects training length: by a held-out VALIDATION
# criterion, never the test set. To avoid chasing validation noise we use the formal
# one-standard-error rule -- choose the SMALLEST depth whose mean validation macro-F1
# is within one standard error of the best depth's. This yields a parsimonious K*
# (the knee of the depth curve) and its test performance, per dataset and pooled.
#
#   python -m recursive_marker_transformer.depth_sweep --datasets pancreas
#   python -m recursive_marker_transformer.depth_sweep --report      # selection table
# results_depthsweep/<dataset>/K<k>_s<seed>.json
# ============================================================================
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .singlecell import _DictLoader, _DTYPES, _load_dataset, _make_splits, _fit_eval, HEAD
from .train import evaluate, resolve_device

ROOT = Path(__file__).resolve().parents[1]
# depth grid spanning 1..100 (0 recursions == a single pass is K=1); dense low, sparse high
GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]
SC = ["tabula_muris", "pancreas", "common_class", "prototype", "baron",
      "segerstolpe", "lung", "oesophagus", "spleen", "tcell"]


def _one(ds, data, K, seed, epochs, device):
    X, y, split = _load_dataset(data / ds)
    F, C = X.shape[1], int(y.max() + 1)
    Xf = X.astype(np.float32, copy=False)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, split, seed)
    cfg = RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
                    n_markers=128, marker_mode="router", recursion_mode="fixed",
                    recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
                    patience=12, lr=1e-3, weight_decay=1e-5, device="cuda")
    yt, yp, model = _fit_eval(Xf, y, tr, va, te, cfg, F, C, device)
    test_f1 = f1_score(yt, yp, average="macro"); test_acc = accuracy_score(yt, yp)
    # validation macro-F1 of the SELECTED (best-val) model -> the selection signal
    mu = Xf[tr].mean(0, keepdims=True); sd = Xf[tr].std(0, keepdims=True) + 1e-6
    Xs = (Xf - mu) / sd
    yv, pv = evaluate(model, _DictLoader(Xs, y, va, 128, False), device, _DTYPES)[HEAD]
    val_f1 = f1_score(yv, pv, average="macro")
    return {"dataset": ds, "K": K, "seed": seed,
            "val_macro_f1": 100 * val_f1, "test_macro_f1": 100 * test_f1,
            "test_accuracy": 100 * test_acc}


COH = {"prostate": "mut_cnv", "blca": "mut_cnv", "stad": "mut_cnv", "panmeta_subtype": "expr"}


def _one_coh(task, K, seed, epochs, device):
    """One cohort run at fixed depth K -> val/test macro-F1 (Reactome pathway tokens)."""
    from .pathway_data import load_cohort, load_pan_meta
    from .pathway_tasks import (PANMETA, _fit_eval as pw_fit, _zscore_train, _DictLoader as PWLoader)
    from .pathway_warmstart import _splits as pw_splits
    chan = COH[task]
    bs = 128 if chan == "expr" else 32
    if task in PANMETA:
        cohort_dir, label = PANMETA[task]
        coh = load_pan_meta(label=label, cohort=cohort_dir, min_genes=5)
    else:
        coh = load_cohort(task, channels=chan, min_genes=5)
    X, y = coh.X, coh.y
    G, C = X.shape[1], (1 if X.ndim == 2 else X.shape[2])
    Kc = int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = pw_splits(y, seed)
    cfg = RMTConfig(heads=(task,), n_hvg=None, n_channels=C, batch_size=bs, d_model=128,
                    d_ff=256, n_markers=256, marker_mode="pathway", recursion_mode="fixed",
                    recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
                    patience=8, lr=3e-4, weight_decay=1e-5, device="cuda",
                    gene_interaction="reactome",
                    pathway_pool=("sum" if task == "brca" else "mean"))
    dtypes = {task: "multiclass"}
    yt, yp, model, dl_te = pw_fit(task, coh, X, y, tr, va, te, cfg, G, Kc, dtypes, device)
    test_f1 = f1_score(yt, yp, average="macro"); test_acc = accuracy_score(yt, yp)
    Xs = _zscore_train(X, tr)
    yv, pv = evaluate(model, PWLoader(Xs, y, va, bs, False, task), device, dtypes)[task]
    val_f1 = f1_score(yv, pv, average="macro")
    return {"dataset": task, "K": K, "seed": seed, "val_macro_f1": 100 * val_f1,
            "test_macro_f1": 100 * test_f1, "test_accuracy": 100 * test_acc}


def run(args):
    device = resolve_device(args.device)
    out = ROOT / "results_depthsweep"
    for ds in args.datasets:
        is_coh = ds in COH
        if not is_coh and not (args.data / ds).exists():
            print(f"[depth] skip {ds}"); continue
        for K in (args.grid or GRID):
            for seed in args.seeds:
                p = out / ds / f"K{K}_s{seed}.json"
                if p.exists() and not args.force:
                    continue
                print(f"\n##### depthsweep {ds} K={K} seed={seed} #####", flush=True)
                r = _one_coh(ds, K, seed, args.epochs, device) if is_coh \
                    else _one(ds, args.data, K, seed, args.epochs, device)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(r, indent=1))
                print(f"  [depth] {ds} K={K} s{seed}: val={r['val_macro_f1']:.1f} "
                      f"test={r['test_macro_f1']:.1f}", flush=True)


# ---------------------------------------------------------------- formal selection
def _curve(ds):
    """K -> (val_mean, val_sem, test_mean, n) over seeds for one dataset."""
    out = {}
    for K in GRID:
        vs, ts = [], []
        for f in glob.glob(str(ROOT / "results_depthsweep" / ds / f"K{K}_s*.json")):
            d = json.loads(Path(f).read_text())
            vs.append(d["val_macro_f1"]); ts.append(d["test_macro_f1"])
        if vs:
            sem = (st.pstdev(vs) / math.sqrt(len(vs))) if len(vs) > 1 else 0.0
            out[K] = (st.mean(vs), sem, st.mean(ts), len(vs))
    return out


def select_best(ds):
    """One-standard-error rule on validation macro-F1: smallest K within 1 SEM of the
    best K. Returns (K_star, K_argmax, val_at_star, test_at_star) or None."""
    c = _curve(ds)
    if not c:
        return None
    k_arg = max(c, key=lambda k: c[k][0])
    best_val, best_sem = c[k_arg][0], c[k_arg][1]
    thresh = best_val - best_sem
    k_star = min(k for k in c if c[k][0] >= thresh)
    return {"K_star": k_star, "K_argmax": k_arg,
            "val_star": c[k_star][0], "test_star": c[k_star][2],
            "val_argmax": c[k_arg][0], "test_argmax": c[k_arg][2]}


def report():
    rows = []
    for ds in SC:
        s = select_best(ds)
        if s:
            rows.append((ds, s))
    print("dataset       K*  Kbest  val@K*  test@K*  test@Kbest")
    for ds, s in rows:
        print(f"{ds:13s} {s['K_star']:3d} {s['K_argmax']:5d}  {s['val_star']:6.1f}  "
              f"{s['test_star']:6.1f}   {s['test_argmax']:6.1f}")
    if rows:
        ks = [s['K_star'] for _, s in rows]
        print(f"\nformal one-SE depth K* : median={int(st.median(ks))}, "
              f"range {min(ks)}-{max(ks)} across {len(rows)} datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=SC)
    ap.add_argument("--grid", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="print the selection table only")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        run(args)


if __name__ == "__main__":
    main()
