# ============================================================================
# SMART -- curated-net falsification aggregator.
# Copyright (c) 2026 The SMART Authors. PROPRIETARY AND CONFIDENTIAL. See LICENSE.
# ============================================================================

"""Aggregate results_bio_curated/ into the pre-registered falsification verdict.

Pairs are formed WITHIN (family, cohort, task, seed): curated/random/none share the
exact split there, so the paired difference cancels split noise. Reports, per cohort
and pooled:

  * mean paired  curated-random  and  curated-none  (macro-F1 points),
  * one-sided Wilcoxon signed-rank p (H1: curated > random),
  * a 95% bootstrap CI on the mean paired curated-random,
  * learned lambda / beta and the corr(pi, PC1) confound diagnostic.

Success (BIO_ROUTER_REDESIGN.txt §6): pooled mean(curated-random) >= +2.0,
one-sided p < 0.05, AND curated >= none.

    python -m recursive_marker_transformer.bio_curated_stats --results results_bio_curated
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def _load(results: Path):
    # key = (family, cohort, task, seed) -> {mode: f1}, plus diag accumulators
    cells = defaultdict(dict)
    lam, beta, corr = [], [], []
    for f in results.rglob("*.json"):
        d = json.loads(f.read_text())
        key = (d["family"], d["cohort"], d["task"], d["seed"])
        cells[key][d["mode"]] = d["test_macro_f1"]
        if "learned_lambda" in d:
            lam.append(d["learned_lambda"]); beta.append(d["learned_beta"])
        if isinstance(d.get("graph_diag"), dict) and "corr_pi_pc1" in d["graph_diag"]:
            corr.append(d["graph_diag"]["corr_pi_pc1"])
    return cells, lam, beta, corr


def _boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _wilcoxon_greater(d):
    d = np.asarray(d, float)
    nz = d[d != 0]
    if len(nz) < 6:
        return float("nan")
    try:
        return float(wilcoxon(nz, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def _report(name, cr, cn):
    cr, cn = np.asarray(cr, float), np.asarray(cn, float)
    if len(cr) == 0:
        print(f"  {name:28s}  (no complete pairs)"); return
    lo, hi = _boot_ci(cr)
    p = _wilcoxon_greater(cr)
    print(f"  {name:28s}  n={len(cr):3d}  "
          f"cur-rand={cr.mean():+5.2f} [{lo:+.2f},{hi:+.2f}]  "
          f"p(1-sided)={p:.4g}  cur-none={cn.mean():+5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results_bio_curated")
    args = ap.parse_args()
    cells, lam, beta, corr = _load(args.results)

    # per-cohort and pooled paired deltas
    per = defaultdict(lambda: ([], []))     # (family,cohort,task) -> (cur-rand, cur-none)
    pool_cr, pool_cn = [], []
    for (fam, coh, task, seed), m in cells.items():
        if "curated" not in m:
            continue
        if "random" in m:
            per[(fam, coh, task)][0].append(m["curated"] - m["random"])
            pool_cr.append(m["curated"] - m["random"])
        if "none" in m:
            per[(fam, coh, task)][1].append(m["curated"] - m["none"])
            pool_cn.append(m["curated"] - m["none"])

    print("=" * 78)
    print("CURATED-NET FALSIFICATION  (paired within family/cohort/task/seed)")
    print("=" * 78)
    for key in sorted(per):
        cr, cn = per[key]
        _report("/".join(key), cr, cn)

    print("-" * 78)
    _report("POOLED", pool_cr, pool_cn)
    if lam:
        print(f"\n  learned lambda: mean={np.mean(lam):.3f} (kept graph if >0)")
        print(f"  learned beta  : mean={np.mean(beta):.3f} (persistent prior if >0)")
    if corr:
        print(f"  corr(pi, PC1) : mean={np.mean(corr):.3f} "
              f"(~0 => centrality is NOT the housekeeping axis)")

    # verdict
    if pool_cr:
        cr = np.asarray(pool_cr, float)
        lo, _ = _boot_ci(cr)
        p = _wilcoxon_greater(cr)
        ok = (cr.mean() >= 2.0) and (p < 0.05) and (np.mean(pool_cn) >= 0.0)
        print("\n  VERDICT (curated biology):", "PASS ✅" if ok else "FAIL ❌",
              f"(need cur-rand>=+2.0 [{cr.mean():+.2f}], p<0.05 [{p:.4g}], "
              f"cur-none>=0 [{np.mean(pool_cn):+.2f}])")

    # DATA-DRIVEN learned graph: learned - none (vs plain router) and learned - curated
    # (vs the best biology graph). This is the "does the alternative win?" panel.
    ln, lc, lr = [], [], []
    per_ln = defaultdict(list)
    for (fam, coh, task, seed), m in cells.items():
        if "learned" not in m:
            continue
        if "none" in m:
            ln.append(m["learned"] - m["none"]); per_ln[(fam, coh, task)].append(m["learned"] - m["none"])
        if "curated" in m:
            lc.append(m["learned"] - m["curated"])
        if "random" in m:
            lr.append(m["learned"] - m["random"])
    if ln:
        print("\n" + "=" * 78)
        print("DATA-DRIVEN LEARNED GRAPH  (learned vs plain router / vs biology)")
        print("=" * 78)
        for key in sorted(per_ln):
            d = np.asarray(per_ln[key], float); lo, hi = _boot_ci(d)
            print(f"  {'/'.join(key):28s}  n={len(d):3d}  "
                  f"learned-none={d.mean():+5.2f} [{lo:+.2f},{hi:+.2f}]  "
                  f"p(1-sided)={_wilcoxon_greater(d):.4g}")
        ln = np.asarray(ln, float); lo, hi = _boot_ci(ln); p = _wilcoxon_greater(ln)
        print("-" * 78)
        print(f"  {'POOLED learned-none':28s}  n={len(ln):3d}  "
              f"delta={ln.mean():+5.2f} [{lo:+.2f},{hi:+.2f}]  p(1-sided)={p:.4g}")
        if lc:
            print(f"  {'POOLED learned-curated':28s}  n={len(lc):3d}  delta={np.mean(lc):+5.2f}")
        if lr:
            print(f"  {'POOLED learned-random':28s}  n={len(lr):3d}  delta={np.mean(lr):+5.2f}")
        win = (ln.mean() >= 2.0) and (p < 0.05) and (np.mean(lc) >= 0 if lc else True)
        print("\n  LEARNED-GRAPH VERDICT:", "WINS ✅" if win else "no clear win ❌",
              f"(learned-none={ln.mean():+.2f} p={p:.4g}; learned-curated={np.mean(lc) if lc else float('nan'):+.2f})")


if __name__ == "__main__":
    main()
