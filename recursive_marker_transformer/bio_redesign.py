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

"""BIO-ROUTER REDESIGN ablation (BIO_ROUTER_REDESIGN.txt).

The original bio-router (an annealed additive centrality bias) neither beat the
plain router (`none`) nor separated from a degree-matched `random` graph. This
runner evaluates the redesigned router, in which biology enters as:

  Fix A  sample-conditional graph PROPAGATION of the input expression (learnable lambda),
  Fix B  a FiLM-GATED prior (state-conditional, not a fixed bias),
  Fix C  a de-confounded PRECISION graph with seeded personalized-PageRank centrality,
  Fix D  a persistent LEARNABLE beta (softplus, not annealed to 0),
  Fix E  a graph-Laplacian DEPTH-SMOOTHNESS penalty (co-regulated genes share depth).

The falsification test: run the IDENTICAL redesigned mechanism on the real
co-expression graph (`coexpr`) and on a degree-matched shuffled graph (`random`).
Success = coexpr > random ~ none (the real edge set, not just its degree, matters).

    python -m recursive_marker_transformer.bio_redesign \
        --dataset segerstolpe --modes none coexpr random --seeds 0 1 --epochs 60
-> results_bio_redesign/<dataset>/<mode>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .singlecell import HEAD, _fit_eval, _load_dataset, _make_splits
from .train import resolve_device

ROOT = Path(__file__).resolve().parents[1]


def _cfg(mode: str, K: int, seed: int, epochs: int, prop: bool = True) -> RMTConfig:
    """Base SMART config (matches the k4 arch sweep) + the redesign fixes when a
    biological graph is used. `none` = plain router (all fixes off). `prop=False`
    turns OFF Fix A (graph propagation) to isolate the router-side biology
    (Fixes B/C/D/E) -- the over-smoothing ablation."""
    base = dict(
        heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
        n_markers=128, marker_mode="router", recursion_mode="expert",
        recursion_depth=K, share_weights=True, seed=seed, epochs=epochs,
        patience=12, lr=1e-3, weight_decay=1e-5, device="cuda",
        gene_interaction=mode,
    )
    if mode in (None, "none"):
        return RMTConfig(**base)
    base.update(
        bio_graph_prop=prop, bio_prop_lambda_init=0.3, bio_prop_hops=1,   # Fix A (toggle)
        bio_prior_gate=True,                                              # Fix B
        bio_deconfound_pc=1, bio_precision=True, bio_centrality="ppr",    # Fix C
        bio_prior_learnable=True, bio_beta_init=0.5,                      # Fix D
        bio_depth_laplacian=0.01,                                        # Fix E
        router_prior_anneal=False,
    )
    return RMTConfig(**base)


def run(ds: str, mode: str, K: int, seed: int, epochs: int, data_dir: Path, device,
        prop: bool = True) -> dict:
    X, y, split = _load_dataset(data_dir / ds)
    F, C = X.shape[1], int(y.max() + 1)
    Xf = X.astype(np.float32, copy=False)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, split, seed)
    cfg = _cfg(mode, K, seed, epochs, prop=prop)
    cfg.n_markers = min(cfg.n_markers, F)
    yt, yp, model = _fit_eval(Xf, y, tr, va, te, cfg, F, C, device)
    out = {
        "dataset": ds, "mode": mode, "K": K, "seed": seed,
        "test_macro_f1": 100 * f1_score(yt, yp, average="macro"),
        "test_accuracy": 100 * accuracy_score(yt, yp),
        "transformer_params": model.transformer_param_count(),
        "total_params": model.total_param_count(),
        "n_features": F, "n_classes": C,
    }
    # Diagnostics: did the model KEEP the graph? (Fix A lambda, Fix D beta)
    if mode not in (None, "none"):
        with torch.no_grad():
            out["learned_lambda"] = float(torch.sigmoid(model.bio_prop_logit))
            out["learned_beta"] = float(torch.nn.functional.softplus(model.bio_beta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="segerstolpe")
    ap.add_argument("--modes", nargs="*", default=["none", "coexpr", "random"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results_bio_redesign")
    ap.add_argument("--prop", choices=["on", "off"], default="on",
                    help="Fix A graph propagation; 'off' isolates router-side biology")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prop = args.prop == "on"

    device = resolve_device(args.device)
    if not (args.data / args.dataset).exists():
        raise SystemExit(f"dataset not found: {args.data / args.dataset}")
    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for mode in args.modes:
        for seed in args.seeds:
            p = out_dir / f"{mode}_s{seed}.json"
            if p.exists() and not args.force:
                print(f"[bio_redesign] skip {args.dataset}/{mode} s{seed} (exists)", flush=True)
                summary.append(json.loads(p.read_text())); continue
            print(f"\n##### bio_redesign {args.dataset} mode={mode} K={args.K} seed={seed} "
                  f"prop={args.prop} #####", flush=True)
            r = run(args.dataset, mode, args.K, seed, args.epochs, args.data, device, prop=prop)
            p.write_text(json.dumps(r, indent=1))
            print(f"  [{mode} s{seed}] F1={r['test_macro_f1']:.2f} acc={r['test_accuracy']:.2f}"
                  + (f" lam={r.get('learned_lambda'):.3f} beta={r.get('learned_beta'):.3f}"
                     if 'learned_lambda' in r else ""), flush=True)
            summary.append(r)

    # Aggregate mean F1 per mode + the falsification deltas.
    def _mean(mode):
        v = [s["test_macro_f1"] for s in summary if s["mode"] == mode]
        return sum(v) / len(v) if v else float("nan")
    mn, mc, mr = _mean("none"), _mean("coexpr"), _mean("random")
    print(f"\n==== {args.dataset} REDESIGN summary (mean macro-F1) ====")
    print(f"  none={mn:.2f}  coexpr={mc:.2f}  random={mr:.2f}")
    print(f"  coexpr-none={mc-mn:+.2f}   coexpr-random={mc-mr:+.2f}  "
          f"(target: coexpr-random >= +2.0)", flush=True)


if __name__ == "__main__":
    main()
