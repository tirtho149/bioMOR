"""Single-column, colour-blind-safe bar chart for the bio-router ablation.

Four router variants (data-driven only / static biology / learned graph / bioMoR) on Baron
(single-cell) and BLCA (multi-omics), read from the completed 5-fold CV results. Okabe-Ito
palette (colour-blind safe, high contrast). Writes paper/figs/biorouter_bars.pdf.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def rd(f):
    d = json.load(open(f))["cv_macro_f1"]
    return d["mean"], d["std"]


PATHS = {
    "Baron":    "results/cv5/biorouter_ablation/Baron/{}_cv.json",
    "Prostate": "results/cv5/biorouter_ablation/pnet/prostate__response/{}_cv.json",
}
MODES = {  # (biology only [fixed graph], data only [learned graph], biology+data [bioMoR])
    "Baron":    ["coexpr", "learned", "learned_bio"],
    "Prostate": ["curated", "learned", "learned_bio"],
}
VARIANTS = ["Biology only", "Data only", "Biology + data (bioMoR)"]
COLORS = ["#E69F00", "#56B4E9", "#009E73"]  # Okabe-Ito: orange / sky-blue / green

groups = list(PATHS)
x = np.arange(len(groups))
w = 0.25

Y0 = 40  # truncated y-axis so the differences are visible (bars start at 40)
fig, ax = plt.subplots(figsize=(3.35, 2.45))
for i, (v, c) in enumerate(zip(VARIANTS, COLORS)):
    means, stds = [], []
    for g in groups:
        m, s = rd(ROOT / PATHS[g].format(MODES[g][i]))
        means.append(m); stds.append(s)
    xs = x + (i - 1) * w
    ax.bar(xs, means, w, yerr=stds, capsize=2, color=c,
           edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.6), label=v)
    for xi, m, s in zip(xs, means, stds):        # label ABOVE the error-bar cap (no overlap)
        ax.text(xi, m + s + 0.9, f"{m:.0f}", ha="center", va="bottom", fontsize=5.2)

ax.set_xticks(x)
ax.set_xticklabels(["Baron (single-cell)", "Prostate (multi-omics)"])
ax.set_ylabel("macro-F1 (\%)".replace("\\", ""))
ax.set_ylim(Y0, 92)
ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.2)
fig.tight_layout()
figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
fig.savefig(figs / "biorouter_bars.pdf", bbox_inches="tight", dpi=600)
print("wrote", figs / "biorouter_bars.pdf")
for g in groups:
    print(g, {v: round(rd(ROOT / PATHS[g].format(MODES[g][i]))[0], 1) for i, v in enumerate(VARIANTS)})
