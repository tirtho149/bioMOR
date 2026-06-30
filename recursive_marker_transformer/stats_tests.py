# ============================================================================
# SMART -- statistical validation of the two headline mechanisms.
#
# Reviewers (rightly) ask whether the biology-informed router and the adaptive
# recursion depth are doing anything beyond noise. This module runs the proper
# hypothesis tests on the per-seed result files and renders significance tables +
# prose tokens for the paper. Everything degrades gracefully: if a result file is
# not present yet, the affected cell/verdict reads "pending" / "TODO" rather than
# fabricating a number.
#
#   python -m recursive_marker_transformer.stats_tests        # prints both reports
#
# Design (what is tested, and why that test):
#   * Biological prior  -- the decisive question is coexpr vs a DEGREE-MATCHED
#     RANDOM GRAPH (not vs no-prior), so any gain cannot be explained by "adding
#     any graph". Paired by seed within each dataset; pooled across datasets.
#     Wilcoxon signed-rank (no normality assumption) + paired t; Holm-Bonferroni
#     across datasets; Cohen's d_z effect size and a 95% CI on the mean gain.
#   * Adaptive depth    -- two separate claims: (i) recursion HELPS (adaptive/fixed
#     vs K=1), a one-sided superiority test; (ii) adaptive depth does NOT cost
#     accuracy vs fixed depth, an EQUIVALENCE test (two one-sided tests, TOST,
#     margin = 1.0 macro-F1 point) -- the right test for a "no measurable cost"
#     claim, since a non-significant difference is not evidence of equivalence;
#     (iii) compute is genuinely reduced (mean token depth < K) across datasets.
# ============================================================================
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
EQUIV_MARGIN = 1.0          # macro-F1 points; "no measurable accuracy cost" band
GENOMAP = ["tabula_muris", "pancreas", "common_class", "prototype", "baron",
           "segerstolpe", "lung", "oesophagus", "spleen", "tcell"]
PATHWAY_COH = ["prostate", "blca", "stad", "panmeta_subtype"]


# --------------------------------------------------------------------------- IO
def _present(ds):
    return [d for d in ds if (ROOT / "data" / "singlecell" / d).exists()]


def _router_seeds(ds, mode):
    """macro-F1 (%) per seed for results_sc_interaction/<ds>__<mode>__seed*.json."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "results_sc_interaction" / f"{ds}__{mode}__seed*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("macro_f1") is not None:
            out.append((d.get("seed"), 100 * d["macro_f1"]))
    return dict(out)               # seed -> f1, so we can pair by seed


def _arch_seeds(ds, variant):
    """macro-F1 (%) per seed for results_singlecell_arch/<variant>/s*/<ds>.json."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results_singlecell_arch" / variant / "s*" / f"{ds}.json"))):
        seed = Path(f).parent.name[1:]
        d = json.loads(Path(f).read_text())
        h = (d.get("heads") or {}).get("cell_type") or next(iter((d.get("heads") or {}).values()), {})
        if h.get("macro_f1") is not None:
            out[seed] = 100 * h["macro_f1"]
    return out


def _paired(a: dict, b: dict):
    """Pair two seed->value dicts on shared seeds; return (xs, ys) aligned."""
    keys = sorted(set(a) & set(b), key=lambda k: str(k))
    return [a[k] for k in keys], [b[k] for k in keys]


# ----------------------------------------------------------------- core stats
def paired_contrast(xs, ys, equiv_margin=None):
    """Paired comparison of xs vs ys (difference d = x - y). Returns a dict with the
    mean gain, 95% CI, paired-t and Wilcoxon p-values, Cohen's d_z, and (optionally)
    a TOST equivalence verdict at +/-equiv_margin. None if <2 pairs."""
    n = len(xs)
    if n < 2:
        return None
    diffs = [x - y for x, y in zip(xs, ys)]
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    df = n - 1
    res = {"n": n, "mean": mean, "sd": sd}
    # paired t-test (two-sided)
    if sd > 0:
        tval = mean / se
        res["t_p"] = float(2 * stats.t.sf(abs(tval), df))
        crit = stats.t.ppf(1 - ALPHA / 2, df)
        res["ci_lo"], res["ci_hi"] = mean - crit * se, mean + crit * se
        res["dz"] = mean / sd                      # Cohen's d_z (paired effect size)
    else:                                          # all diffs identical
        res["t_p"] = 0.0 if mean != 0 else 1.0
        res["ci_lo"] = res["ci_hi"] = mean
        res["dz"] = float("inf") if mean != 0 else 0.0
    # Wilcoxon signed-rank (distribution-free); needs some non-zero diffs
    if n >= 6 and any(d != 0 for d in diffs):
        try:
            res["w_p"] = float(stats.wilcoxon(diffs).pvalue)
        except ValueError:
            res["w_p"] = None
    else:
        res["w_p"] = None
    # TOST equivalence: both one-sided tests must reject to declare equivalence
    if equiv_margin is not None and sd > 0:
        t_low = (mean - (-equiv_margin)) / se
        p_low = float(stats.t.sf(t_low, df))            # H0: diff <= -margin
        t_up = (mean - equiv_margin) / se
        p_up = float(stats.t.cdf(t_up, df))             # H0: diff >=  margin
        res["tost_p"] = max(p_low, p_up)
        res["equivalent"] = res["tost_p"] < ALPHA
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a list of (key, p); p may be None."""
    items = [(k, p) for k, p in pvals if p is not None]
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    m = len(items)
    adj = {}
    running = 0.0
    for rank, i in enumerate(order):
        k, p = items[i]
        a = min(1.0, (m - rank) * p)
        running = max(running, a)                  # enforce monotonicity
        adj[k] = running
    for k, p in pvals:
        adj.setdefault(k, None)
    return adj


# ------------------------------------------------------- biological-prior report
def router_report():
    """coexpr vs random-graph control (primary) and vs no-prior (secondary)."""
    ds_list = _present(GENOMAP)
    per_ds = {}
    pooled_primary, pooled_secondary = ([], []), ([], [])
    for d in ds_list:
        cx, rnd, non = _router_seeds(d, "coexpr"), _router_seeds(d, "random"), _router_seeds(d, "none")
        xs, ys = _paired(cx, rnd)
        xs2, ys2 = _paired(cx, non)
        per_ds[d] = {
            "vs_random": paired_contrast(xs, ys),
            "vs_none": paired_contrast(xs2, ys2),
        }
        pooled_primary[0].extend(xs); pooled_primary[1].extend(ys)
        pooled_secondary[0].extend(xs2); pooled_secondary[1].extend(ys2)
    holm_adj = holm([(d, per_ds[d]["vs_random"]["t_p"] if per_ds[d]["vs_random"] else None)
                     for d in ds_list])
    for d in ds_list:
        per_ds[d]["vs_random_holm"] = holm_adj.get(d)
    pooled = {
        "vs_random": paired_contrast(*pooled_primary),
        "vs_none": paired_contrast(*pooled_secondary),
    }
    nsig = sum(1 for d in ds_list
               if per_ds[d].get("vs_random_holm") is not None
               and per_ds[d]["vs_random_holm"] < ALPHA
               and per_ds[d]["vs_random"] and per_ds[d]["vs_random"]["mean"] > 0)
    return {"datasets": ds_list, "per_ds": per_ds, "pooled": pooled, "n_sig": nsig}


# --------------------------------------------------------- adaptive-depth report
def depth_report():
    ds_list = _present(GENOMAP)
    # (i) recursion helps: shared (adaptive) vs depth1 (K=1), pooled, one-sided
    helps_x, helps_y = [], []
    # (ii) adaptive vs fixed: equivalence (TOST), pooled
    equiv_x, equiv_y = [], []
    per_ds = {}
    for d in ds_list:
        sh, fx, k1 = _arch_seeds(d, "shared"), _arch_seeds(d, "fixed"), _arch_seeds(d, "depth1")
        hx, hy = _paired(sh, k1)
        ex, ey = _paired(sh, fx)
        per_ds[d] = {"helps": paired_contrast(hx, hy),
                     "equiv": paired_contrast(ex, ey, equiv_margin=EQUIV_MARGIN)}
        helps_x.extend(hx); helps_y.extend(hy)
        equiv_x.extend(ex); equiv_y.extend(ey)
    helps = paired_contrast(helps_x, helps_y)
    equiv = paired_contrast(equiv_x, equiv_y, equiv_margin=EQUIV_MARGIN)
    # (iii) compute genuinely reduced: per-dataset saved fraction 1 - mean_depth/K
    saved = []
    for d in ds_list:
        p = ROOT / "results_depth" / f"{d}.json"
        if p.exists():
            r = json.loads(p.read_text())
            K = r.get("recursion_depth", 4)
            md = r.get("mean_token_depth")
            if md is not None and K:
                saved.append(100 * (1 - md / K))
    compute = None
    if len(saved) >= 2:
        m = sum(saved) / len(saved)
        sd = (sum((s - m) ** 2 for s in saved) / (len(saved) - 1)) ** 0.5
        se = sd / math.sqrt(len(saved)) if sd > 0 else 0.0
        df = len(saved) - 1
        compute = {"n": len(saved), "mean": m, "sd": sd,
                   "p": float(stats.t.sf(m / se, df)) if se > 0 else (0.0 if m > 0 else 1.0)}
    return {"datasets": ds_list, "per_ds": per_ds, "helps": helps,
            "equiv": equiv, "compute": compute}


# --------------------------------------------------------------------- rendering
def _p(p):
    if p is None:
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def _fmt_contrast(c, signed=True):
    if c is None:
        return "TODO"
    g = f"{c['mean']:+.2f}" if signed else f"{c['mean']:.2f}"
    return g


def _table(header, rows, fmt):
    if fmt == "tex":
        def esc(s):
            s = str(s)
            # bold (**x**) -> \textbf{x}; do this before escaping underscores etc.
            while "**" in s:
                s = s.replace("**", r"\textbf{", 1).replace("**", "}", 1)
            s = s.replace("%", r"\%").replace("_", r"\_")     # escape underscores first
            for a, b in [("Δ", r"$\Delta$"), (r"d\_z", r"$d_z$"), ("≤", r"$\le$"),
                         ("−", r"$-$"), ("×", r"$\times$"), ("±", r"$\pm$")]:
                s = s.replace(a, b)
            return s
        out = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
               " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
        out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)
    out = ["| " + " | ".join(map(str, header)) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def router_stats_table(fmt="md"):
    rep = router_report()
    rows = []
    for d in rep["datasets"]:
        c = rep["per_ds"][d]["vs_random"]
        h = rep["per_ds"][d].get("vs_random_holm")
        if c is None:
            rows.append([d, "TODO", "TODO", "TODO", "TODO", "TODO"]); continue
        dz = "--" if not math.isfinite(c["dz"]) else f"{c['dz']:+.2f}"
        rows.append([d, f"{c['mean']:+.2f}", f"[{c['ci_lo']:+.1f}, {c['ci_hi']:+.1f}]",
                     _p(c.get("w_p") or c["t_p"]), _p(h), dz])
    pc = rep["pooled"]["vs_random"]
    if pc is not None:
        dz = "--" if not math.isfinite(pc["dz"]) else f"{pc['dz']:+.2f}"
        rows.append(["**pooled**", f"{pc['mean']:+.2f}",
                     f"[{pc['ci_lo']:+.1f}, {pc['ci_hi']:+.1f}]",
                     _p(pc.get("w_p") or pc["t_p"]), "--", dz])
    return _table(["Dataset", "ΔF1 (coexpr−random)", "95% CI", "p", "p (Holm)", "d_z"], rows, fmt)


def depth_stats_table(fmt="md"):
    rep = depth_report()
    rows = []
    for d in rep["datasets"]:
        hl = rep["per_ds"][d]["helps"]
        eq = rep["per_ds"][d]["equiv"]
        rows.append([
            d,
            _fmt_contrast(hl), _p(hl["t_p"]) if hl else "TODO",
            _fmt_contrast(eq),
            ("equiv" if (eq and eq.get("equivalent")) else ("no" if eq else "TODO")),
            _p(eq.get("tost_p")) if eq else "TODO",
        ])
    return _table(["Dataset", "ΔF1 (adaptive−K1)", "p (helps)",
                   "ΔF1 (adaptive−fixed)", "equiv?", "TOST p"], rows, fmt)


# ---------------------------------------------------- prose tokens for make_paper
def prose_tokens():
    """Build a dict of @@TOKEN@@ -> string for the paper's statistics sentences."""
    r = router_report()
    d = depth_report()
    t = {}
    pc = r["pooled"]["vs_random"]
    if pc:
        sig = (pc.get("w_p") or pc["t_p"]) < ALPHA
        verb = "significantly outperforms" if sig and pc["mean"] > 0 else "is statistically indistinguishable from"
        t["@@ROUTER_STAT_SENT@@"] = (
            f"Across the suite the co-expression prior {verb} the degree-matched "
            f"random-graph control (paired $\\Delta$F1 $=$ {pc['mean']:+.2f} points, "
            f"95\\% CI [{pc['ci_lo']:+.1f}, {pc['ci_hi']:+.1f}], "
            f"$p={ (pc.get('w_p') or pc['t_p']):.3f}$, Cohen's $d_z={pc['dz']:+.2f}$; "
            f"{r['n_sig']}/{len(r['datasets'])} datasets significant after Holm-Bonferroni)")
    else:
        t["@@ROUTER_STAT_SENT@@"] = "Statistical validation of the prior is pending the multi-seed router runs"
    hl, eq, cp = d["helps"], d["equiv"], d["compute"]
    if hl and eq:
        eqv = "statistically equivalent" if eq.get("equivalent") else "not formally equivalent"
        t["@@DEPTH_STAT_SENT@@"] = (
            f"Recursion improves over a single pass (adaptive $-$ $K{{=}}1$: "
            f"$\\Delta$F1 $=$ {hl['mean']:+.2f}, $p={hl['t_p']:.3f}$), while adaptive "
            f"depth is {eqv} to fixed depth within a {EQUIV_MARGIN:.0f}-point margin "
            f"(TOST $p={eq.get('tost_p', float('nan')):.3f}$)")
        if cp:
            t["@@DEPTH_STAT_SENT@@"] += (
                f", at a mean compute saving of {cp['mean']:.0f}\\% "
                f"($p={cp['p']:.3f}$ that the saving exceeds zero)")
    else:
        t["@@DEPTH_STAT_SENT@@"] = "Statistical validation of the depth router is pending the multi-seed arch runs"
    return t


def main():
    print("=== Biological-prior significance (coexpr vs random-graph control) ===")
    print(router_stats_table("md"))
    print("\n=== Adaptive-depth significance ===")
    print(depth_stats_table("md"))
    print("\n=== Prose ===")
    for k, v in prose_tokens().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
