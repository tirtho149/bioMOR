"""Generates paper/figs/pareto_efficiency.pdf (fig:pareto in the paper).

Pareto training-cost figure for ONE dataset (Baron): y = Macro F1, x = compute.
We do NOT yet have measured GPU-hours in the result JSONs, so this prototype uses the
analytical relative FLOPs (per forward pass) we already compute -- a hardware-independent
proxy. Swap the x-values for measured GPU-hours once a timed run exists (see plan).

Points: Vanilla, general Recursive, Mixture-of-Recursions (MoR), and bioMoR, each at its
depth variants. Colour = architecture (Okabe-Ito, colour-blind safe); marker size = params.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import build_cv5_tex as B

BARON = 0  # Baron is data column 0

def arch_of(label, variant, kind):
    if kind.startswith("biomor"): return "bioMoR"
    if label.startswith("Vanilla") or variant == "independent": return "Vanilla"
    if label.startswith("Recursive") or (variant or "").startswith("fixed"): return "Recursive"
    return "MoR (general)"

COL = {"Vanilla": "#999999", "Recursive": "#E69F00",
       "MoR (general)": "#56B4E9", "bioMoR": "#009E73"}

pts = []
for s in B.SPECS:
    if s is None: continue
    label, variant, mode, K, typ, param, kind = s
    sc, pn, pan = B.row_vals(kind, variant, K, mode)
    v = (sc + pn + pan)[BARON]
    if v is None: continue          # skip not-yet-run (token k2/k3 placeholders)
    f1 = v[0]
    flops = B.flops_rel(mode, K) or 1.0
    p = 300 if param == "300K" else 75      # K-params
    pts.append(dict(arch=arch_of(label, variant, kind), f1=f1, cost=flops,
                    params=p, depth=K, mode=mode))

# dedupe to ONE bubble per (architecture, depth): keep the best-F1 variant so the
# expert/token near-duplicates don't pile on top of each other.
_bestby = {}
for p in pts:
    k = (p["arch"], p["depth"])
    if k not in _bestby or p["f1"] > _bestby[k]["f1"]:
        _bestby[k] = p
pts = list(_bestby.values())

# Pareto frontier: maximise F1, minimise cost -> upper-left non-dominated set
def dominated(a, all_):
    return any((b["cost"] <= a["cost"] and b["f1"] >= a["f1"] and
                (b["cost"] < a["cost"] or b["f1"] > a["f1"])) for b in all_ if b is not a)
front = sorted([p for p in pts if not dominated(p, pts)], key=lambda d: d["cost"])

from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(6.0, 4.2))

xmin, xmax = 0.33, 1.10
ymin, ymax = 55, 90
best = max(pts, key=lambda d: d["f1"])          # headline bioMoR point
van = max((p for p in pts if p["arch"] == "Vanilla"), key=lambda d: d["cost"])
DARK = {"Vanilla": "#5a5a5a", "Recursive": "#8a6100",
        "MoR (general)": "#1f6f9e", "bioMoR": "#00694f"}

# --- shaded "winning" region: high accuracy AND low compute (upper-left of Vanilla) ---
ax.axhspan(van["f1"], ymax, xmin=0, xmax=(van["cost"] - xmin) / (xmax - xmin),
           color="#009E73", alpha=0.06, zorder=0)

# --- Pareto frontier line ---
fx = [p["cost"] for p in front]; fy = [p["f1"] for p in front]
ax.plot(fx, fy, color="#333333", lw=1.4, ls="--", alpha=0.6, zorder=1)

# --- bubbles: area proportional to parameters, K label inside each ---
for p in sorted(pts, key=lambda d: -d["params"]):   # big bubbles first (drawn behind)
    area = 180 + p["params"] * 1.7          # Vanilla 300K -> larger; shared 75K -> small
    ax.scatter(p["cost"], p["f1"], s=area, c=COL[p["arch"]], alpha=0.6,
               edgecolors=DARK[p["arch"]], linewidths=1.5, zorder=3)
    ax.text(p["cost"], p["f1"], f"K{p['depth']}", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=DARK[p["arch"]], zorder=4)

# single callout for the headline win, in the empty middle-right
gain = best["f1"] - van["f1"]; speed = van["cost"] / best["cost"]
ax.annotate(f"bioMoR: +{gain:.0f} macro-F1\nat {speed:.1f}$\\times$ less compute",
            xy=(best["cost"] + 0.03, best["f1"]), xytext=(0.70, 78.5),
            fontsize=10, fontweight="bold", color="#00694f", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#eafaf3", ec="#009E73", lw=1.3),
            arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=1.7))

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xlabel("Relative compute  (FLOPs / forward pass, $\\times$Vanilla)", fontsize=11)
ax.set_ylabel("Macro-F1 (\\%)".replace("\\", ""), fontsize=11)
ax.set_title("Accuracy vs.\\ compute on Baron".replace("\\", ""), fontsize=12, fontweight="bold")
ax.grid(True, ls=":", lw=0.5, alpha=0.5)
# color legend BELOW the axes (no in-plot overlap); bubble size note alongside
order = ["Vanilla", "Recursive", "MoR (general)", "bioMoR"]
handles = [Line2D([0], [0], marker="o", ls="", ms=10, mfc=COL[a], mec=DARK[a],
                  mew=1.4, alpha=0.7, label=a) for a in order]
handles.append(Line2D([0], [0], ls="--", color="#333333", lw=1.4, label="Pareto frontier"))
ax.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
          loc="upper center", bbox_to_anchor=(0.5, -0.16), handletextpad=0.4,
          columnspacing=1.3)
ax.text(0.985, 0.03, "bubble size $\\propto$ parameters", transform=ax.transAxes,
        fontsize=8, color="#555555", ha="right", style="italic")
fig.tight_layout()
import os as _os
_figs = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "paper", "figs")
_os.makedirs(_figs, exist_ok=True)
fig.savefig(_os.path.join(_figs, "pareto_efficiency.pdf"), bbox_inches="tight", dpi=600)
print("wrote paper/figs/pareto_efficiency.pdf")
print(f"{'arch':16s} {'depth':5s} {'cost(FLOPs)':11s} {'F1':6s} {'onFrontier'}")
for p in sorted(pts, key=lambda d: (d["arch"], d["cost"])):
    print(f"{p['arch']:16s} K{p['depth']:<4d} {p['cost']:<11.3f} {p['f1']:<6.1f} {'*' if p in front else ''}")
