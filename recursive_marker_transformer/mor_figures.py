# ============================================================================
# SMART -- figures for the MoR-table reproduction on the genomap suite.
# Generates the genomap analogues of MoR Figures 3 (scaling) and 5 (recursion-depth
# / token-count analysis), plus a parameter-efficiency panel, into paper/figs/.
#     python -m recursive_marker_transformer.mor_figures
# ============================================================================
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
GENOMAP = ["tabula_muris", "pancreas", "common_class", "prototype", "baron", "segerstolpe"]
SIZES = [48, 96, 192, 384]
# MoR-paper palette (Bae et al. 2025, Fig. 3): Vanilla=green, Recursive=blue, MoR=orange
COL = {"vanilla": "#2CA02C", "recursive": "#1F77B4", "mor": "#FF7F0E"}
ARCHS = [("vanilla", "Vanilla"), ("recursive", "Recursive"), ("mor", "MoR (SMART)")]
plt.rcParams.update({"font.size": 10, "axes.facecolor": "white",
                     "axes.edgecolor": "#444444", "axes.grid": True,
                     "grid.color": "#DDDDDD", "grid.linewidth": 0.6})


def _present(ds):
    return [d for d in ds if (ROOT / "data" / "singlecell" / d).exists()]


def _f1(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    h = d.get("heads", {})
    h = h.get("cell_type") or (next(iter(h.values())) if h else None)
    return 100 * h["macro_f1"] if h else None


def fig_scaling():
    """MoR Fig 3 analogue: macro-F1 vs model size for Vanilla/Recursive/MoR."""
    ds = _present(GENOMAP)
    n = len(ds)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, d in enumerate(ds):
        ax = axes[i // cols][i % cols]
        for a, lab in ARCHS:
            ys = [_f1(ROOT / f"results_scaling/{a}_d{D}" / f"{d}.json") for D in SIZES]
            xs = [s for s, y in zip(SIZES, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            if yy:
                ax.plot(xs, yy, marker="o", color=COL[a], linewidth=2, label=lab)
        ax.set_title(d); ax.set_xlabel("d_model"); ax.set_ylabel("macro-F1")
        ax.grid(alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Fig 3 analogue: scaling of MoR vs Recursive vs Vanilla (genomap)")
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_scaling.png", dpi=140); plt.close(fig)
    return "fig_scaling.png"


def _arch_f1(variant, d):
    import statistics as st
    xs = [json.loads(Path(f).read_text())["heads"]["cell_type"]["macro_f1"] * 100
          for f in glob.glob(str(ROOT / f"results_singlecell_arch/{variant}/s*/{d}.json"))]
    return st.mean(xs) if xs else None


def fig_depth():
    """Adaptive-depth result, per dataset: macro-F1 under no recursion (K=1),
    fixed-depth recursion, and adaptive routed depth. (The raw active-fraction funnel
    is set by the fixed capacity schedule and is identical across datasets, so we
    instead show the dataset-varying accuracy each regime achieves.)"""
    import numpy as np
    ds = _present(GENOMAP)
    regimes = [("depth1", "K=1 (no recursion)", "#9E9E9E"),
               ("fixed", "fixed depth", "#1F77B4"),
               ("shared", "adaptive depth (MoR)", "#FF7F0E")]
    vals = {r: [_arch_f1(r, d) for d in ds] for r, _, _ in regimes}
    if not any(any(v is not None for v in vv) for vv in vals.values()):
        return None
    x = np.arange(len(ds)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (r, lab, col) in enumerate(regimes):
        ys = [v if v is not None else 0 for v in vals[r]]
        ax.bar(x + (i - 1) * w, ys, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=20, ha="right")
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 100)
    ax.set_title("Adaptive recursion depth across the genomap suite")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_depth.png", dpi=140); plt.close(fig)
    return "fig_depth.png"


def fig_param_efficiency():
    """Parameter reduction: macro-F1 vs transformer params (recursive vs independent)."""
    ds = _present(GENOMAP)
    fig, ax = plt.subplots(figsize=(6, 4))
    got = False
    for a, lab in [("recursive", "Recursive (shared)"), ("vanilla", "Vanilla (independent)")]:
        xs, ys = [], []
        for D in SIZES:
            for d in ds:
                p = ROOT / f"results_scaling/{a}_d{D}" / f"{d}.json"
                if p.exists():
                    j = json.loads(p.read_text())
                    tp = j.get("transformer_params")
                    f1 = _f1(p)
                    if tp and f1 is not None:
                        xs.append(tp); ys.append(f1); got = True
        if xs:
            ax.scatter(xs, ys, label=lab, alpha=0.8, s=40, color=COL[a])
    if not got:
        plt.close(fig); return None
    ax.set_xscale("log"); ax.set_xlabel("transformer params (log)"); ax.set_ylabel("macro-F1")
    ax.set_title("Parameter efficiency: shared recursion vs independent")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_param_efficiency.png", dpi=140); plt.close(fig)
    return "fig_param_efficiency.png"


def main():
    made = [f for f in (fig_scaling(), fig_depth(), fig_param_efficiency()) if f]
    print(f"[mor_figures] wrote {len(made)} figures -> {FIGS}: {made}")
    return made


if __name__ == "__main__":
    main()
