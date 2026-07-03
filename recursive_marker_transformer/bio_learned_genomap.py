# ============================================================================
# SMART -- learned-graph bio-router on the genomap single-cell suite.
# Copyright (c) 2026 The SMART Authors. PROPRIETARY AND CONFIDENTIAL. See LICENSE.
# ============================================================================

"""Run the bio-router modes (none / coexpr / random / LEARNED) on the collaborator
genomap single-cell datasets (genomap_data/, materialised .npy/.mat).

The learned data-driven graph needs only expression (no gene symbols), so unlike the
curated Reactome net it runs on the anonymised single-cell suite. This is the
"does the learned graph generalise beyond P-NET?" sweep.

    python -m recursive_marker_transformer.bio_learned_genomap \
        --dataset Spleen --modes none coexpr random learned --seeds 0 1 2 --epochs 60
-> results_learned_genomap/<dataset>/<mode>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .singlecell import HEAD, _fit_eval, _make_splits
from .train import resolve_device

ROOT = Path(__file__).resolve().parents[1]
GD = ROOT / "genomap_data"

# dataset -> (kind, path-parts). Supervised cell-recognition sets only (Retinal /
# Trajectory are unsupervised and excluded).
DATASETS = {
    "Lung":       ("ischaemic", "Ischaemic/Lung"),
    "Oesophagus": ("ischaemic", "Ischaemic/Oesophagus"),
    "Spleen":     ("ischaemic", "Ischaemic/Spleen"),
    "Tcell":      ("tcell",     "Tcell/Elyahu2019_SCP490"),
    "Baron":      ("pancreas",  "Pancreas/Baron"),
    "Muraro":     ("pancreas",  "Pancreas/Muraro"),
    "Segerstolpe":("pancreas",  "Pancreas/Segerstolpe"),
    "Wang":       ("pancreas",  "Pancreas/Wang"),
    "Xin":        ("pancreas",  "Pancreas/Xin"),
}


def _encode(y_raw):
    uniq = np.unique(y_raw)
    remap = {v: i for i, v in enumerate(uniq)}
    return np.array([remap[v] for v in y_raw], dtype=np.int64)


def load_genomap(dataset: str):
    """Return (X float32 [N,G], y int64 [N])."""
    kind, rel = DATASETS[dataset]
    d = GD / rel
    if kind in ("ischaemic", "tcell"):
        stem = dataset.lower() if kind == "ischaemic" else "tcell"
        X = np.load(d / f"{stem}_data.npy").astype(np.float32)
        gt = sio.loadmat(d / f"GT_{stem}.mat")["GT"].ravel()
        y = _encode(gt)
    else:  # pancreas: data{DS}X.mat holds BOTH the (N,G) matrix and the (N,1) GT
        mats = [p for p in d.glob("*.mat") if p.name.startswith("data")]
        m = sio.loadmat(mats[0])
        arrs = {k: v for k, v in m.items() if not k.startswith("__")}
        Xk = max(arrs, key=lambda k: arrs[k].size)                 # the 2D data matrix
        X = np.asarray(arrs[Xk], dtype=np.float32)
        gk = [k for k in arrs if arrs[k].shape[0] == X.shape[0] and k != Xk]
        y = _encode(np.asarray(arrs[gk[0]]).ravel())
    return X, y


def _cfg(mode: str, K: int, seed: int, epochs: int) -> RMTConfig:
    base = dict(
        heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
        n_markers=128, marker_mode="router", recursion_mode="expert",
        recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
        patience=12, lr=1e-3, weight_decay=1e-5, device="cuda",
        gene_interaction=(mode if mode in ("coexpr", "random") else "none"),
    )
    cfg = RMTConfig(**base)
    if mode in ("coexpr", "random"):
        cfg.bio_graph_prop = True; cfg.bio_prop_lambda_init = 0.3; cfg.bio_prop_hops = 1
        cfg.bio_prior_gate = True; cfg.bio_prior_learnable = True; cfg.bio_beta_init = 0.5
        cfg.bio_depth_laplacian = 0.01; cfg.bio_centrality = "ppr"
        cfg.router_prior_anneal = False
    elif mode == "learned":
        cfg.bio_learned_graph = True; cfg.bio_learned_rank = 16
        cfg.bio_prop_lambda_init = 0.2; cfg.bio_prop_hops = 1
    return cfg


def run_cell(X, y, dataset, mode, K, seed, epochs, device):
    F, C = X.shape[1], int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, None, seed)
    cfg = _cfg(mode, K, seed, epochs); cfg.n_markers = min(cfg.n_markers, F)
    yt, yp, model = _fit_eval(X.astype(np.float32), y, tr, va, te, cfg, F, C, device)
    out = {"dataset": dataset, "mode": mode, "K": K, "seed": seed,
           "test_macro_f1": 100 * f1_score(yt, yp, average="macro"),
           "test_accuracy": 100 * accuracy_score(yt, yp),
           "n_features": F, "n_classes": C, "n_samples": int(len(y))}
    if mode == "learned":
        with torch.no_grad():
            out["learned_lambda"] = float(torch.sigmoid(model.bio_prop_logit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--modes", nargs="*", default=["none", "coexpr", "random", "learned"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results_learned_genomap")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.device)
    X, y = load_genomap(args.dataset)
    out_dir = args.out / args.dataset; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[genomap] {args.dataset} N={len(y)} G={X.shape[1]} C={int(y.max()+1)}", flush=True)

    summary = []
    for mode in args.modes:
        for seed in args.seeds:
            p = out_dir / f"{mode}_s{seed}.json"
            if p.exists() and not args.force:
                summary.append(json.loads(p.read_text())); continue
            print(f"\n##### genomap {args.dataset} mode={mode} seed={seed} #####", flush=True)
            r = run_cell(X, y, args.dataset, mode, args.K, seed, args.epochs, device)
            p.write_text(json.dumps(r, indent=1))
            print(f"  [{mode} s{seed}] F1={r['test_macro_f1']:.2f} acc={r['test_accuracy']:.2f}"
                  + (f" lam={r.get('learned_lambda'):.3f}" if "learned_lambda" in r else ""),
                  flush=True)
            summary.append(r)

    def _mean(m):
        v = [s["test_macro_f1"] for s in summary if s["mode"] == m]; return sum(v)/len(v) if v else float("nan")
    print(f"\n==== {args.dataset} summary (mean macro-F1) ====")
    print("  " + "  ".join(f"{m}={_mean(m):.2f}" for m in args.modes))
    if "learned" in args.modes and "none" in args.modes:
        print(f"  learned-none={_mean('learned')-_mean('none'):+.2f}", flush=True)


if __name__ == "__main__":
    main()
