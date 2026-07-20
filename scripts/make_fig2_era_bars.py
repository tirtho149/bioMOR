#!/usr/bin/env python
"""Figure 2 -- biology ablation (None / E / R / A / ERA) as a two-row grouped bar chart.
Row 1 = single-cell (8 datasets), Row 2 = multi-omics (6 cohorts). 5 bars per group,
y-axis cut at 50 to zoom the differences. Bars whose mean falls below the 50 floor are
annotated in red at the baseline so no information is lost. Reads results/cv5/fig2_{sc,mo}.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC_FLOOR = 40   # y-axis cut at 40 on both rows so small ERA differences are easy to see
MO_FLOOR = 40   # (bars below 40 -- a few Bladder cells -- are annotated in red at the baseline)
CONDS = ["N", "E", "R", "A", "ERA"]
CLABEL = {"N": "None", "E": "E (embed)", "R": "R (router)", "A": "A (attn-bias)", "ERA": "ERA (full)"}
COLORS = {"N": "#9e9e9e", "E": "#4c9be8", "R": "#f0a848", "A": "#8e6fcf", "ERA": "#2ca25f"}

# nice display names / order
SC_ORDER = ["Segerstolpe", "Lung", "Oesophagus", "Baron", "Muraro", "Tcell", "Spleen", "Xin"]
SC_DISP = {"Tcell": "T-cell", "Segerstolpe": "Segerst.", "Oesophagus": "Oesoph."}
MO_ORDER = ["prostate", "blca", "stad", "pan_meta_pri", "panmeta_response", "pan_meta_pri_3modal"]
MO_DISP = {"prostate": "Prostate", "blca": "Bladder", "stad": "Stomach",
           "pan_meta_pri": "Pan-MC", "panmeta_response": "Pan-Ex", "pan_meta_pri_3modal": "Pan-3M"}


def meanstd(f):
    v = json.load(open(f)).get("cv_macro_f1")
    if isinstance(v, dict):
        return v.get("mean"), v.get("std", 0.0)
    return None, None


def load_sc():
    d = {}
    for f in glob.glob(os.path.join(ROOT, "results/cv5/fig2_sc/*/*_cv.json")):
        ds = f.split(os.sep)[-2]
        cond = os.path.basename(f).replace("_cv.json", "")
        d.setdefault(ds, {})[cond] = meanstd(f)
    return d


def load_mo():
    d = {}
    for f in glob.glob(os.path.join(ROOT, "results/cv5/fig2_mo/*/*/*_cv.json")):
        coh = f.split(os.sep)[-3]
        cond = f.split(os.sep)[-2]
        d.setdefault(coh, {})[cond] = meanstd(f)
    return d


def draw_row(ax, data, order, disp, title, floor):
    n = len(order)
    w = 0.16
    x = np.arange(n)
    for j, cond in enumerate(CONDS):
        means = [(data.get(k, {}).get(cond) or (np.nan, 0))[0] for k in order]
        stds = [(data.get(k, {}).get(cond) or (np.nan, 0))[1] for k in order]
        off = (j - 2) * w
        bars = ax.bar(x + off, means, w, yerr=stds, capsize=2,
                      color=COLORS[cond], edgecolor="black", linewidth=0.4,
                      error_kw=dict(lw=0.7, alpha=0.6), label=CLABEL[cond])
        for xi, m, s in zip(x + off, means, stds):
            if np.isnan(m):
                continue
            if m >= floor:
                ax.text(xi, m + s + 0.4, f"{m:.0f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)
            else:  # still below this row's floor -> annotate value in red at the baseline
                ax.text(xi, floor + 0.4, f"{m:.0f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90, color="#d62728", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([disp.get(k, k) for k in order], fontsize=9)
    ax.set_ylim(floor, 100)
    ax.set_ylabel("Macro-F1 (5-fold)")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    # visual cue that the axis is broken at the floor
    ax.text(-0.006, floor, "≈", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=11)


def main():
    sc, mo = load_sc(), load_mo()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2))
    draw_row(ax1, sc, SC_ORDER, SC_DISP, "(a) Single-cell", SC_FLOOR)
    draw_row(ax2, mo, MO_ORDER, MO_DISP, "(b) Multi-omics", MO_FLOOR)
    handles = [Patch(facecolor=COLORS[c], edgecolor="black", label=CLABEL[c]) for c in CONDS]
    ax1.legend(handles=handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.28), frameon=False, fontsize=9)
    fig.suptitle("Figure 2 — Biology ablation (None → E → R → A → ERA); y-axis cut at 40",
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ROOT, "paper", "figs")   # paper/figs is the tree the Overleaf bridge syncs
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig2_era_bars.{ext}"), dpi=600, bbox_inches="tight")
    print("wrote figs/fig2_era_bars.{pdf,png}")
    # report clipped (below-floor) cells
    for name, d, order, floor in [("SC", sc, SC_ORDER, SC_FLOOR), ("MO", mo, MO_ORDER, MO_FLOOR)]:
        for k in order:
            for c in CONDS:
                v = d.get(k, {}).get(c)
                if v and v[0] < floor:
                    print(f"  clipped ({name}, floor={floor}): {k} {c} = {v[0]:.1f}")


if __name__ == "__main__":
    main()
