"""Figure 2 real-data right panels: per-pathway recursion depth on PROSTATE (multi-omics),
expert-choice vs token-choice, in the SAME dot-matrix style as the schematic right panel
(rows = Reactome pathways, columns = recursion steps t=1..4; filled dot = still recursing,
open = exited; d_m = exit depth). Same pathways in both panels so the routing contrast is
direct. -> paper/figs/fig2_depth.pdf"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/depth/prostate_panels.json"))
names = d["pathways"]; K = d["recursion_depth"]
et = np.array(d["expert"]["depths"]); tt = np.array(d["token"]["depths"])
ACC = "#009E73"       # accentA green (recurses); open = exited

def short(s, n=26): return s if len(s) <= n else s[:n-1] + "…"

# pick 10 pathways spanning depths 1..4 with maximum expert-vs-token contrast, so both
# dot-matrix panels show visible depth variance (a staircase) rather than all-deep rows.
te_r = np.clip(np.round(et).astype(int), 1, K)
tt_r = np.clip(np.round(tt).astype(int), 1, K)
quota = {4: 3, 3: 3, 2: 2, 1: 2}                 # 10 rows spanning all depths
sel = []
for L in (4, 3, 2, 1):
    cand = [i for i in range(len(names)) if tt_r[i] == L and i not in sel]
    cand.sort(key=lambda i: (-abs(int(te_r[i]) - int(tt_r[i])), len(names[i])))  # prefer contrast + shorter names
    sel += cand[:quota[L]]
sel = sorted(sel[:10], key=lambda i: (-tt[i], -et[i]))   # deepest token at top
row_names = [short(names[i]) for i in sel]
nrow = len(sel)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4), sharey=True)
for ax, mode, dep, title in [(axes[0], "expert", et, "Expert-choice"),
                             (axes[1], "token", tt, "Token-choice")]:
    for t in range(1, K + 1):                    # faint vertical step guides
        ax.axvline(t, color="#e9e9e9", lw=6, zorder=0)
    for r, i in enumerate(sel):
        y = nrow - 1 - r
        dm = int(round(dep[i]))
        if r % 2 == 0:                            # subtle alternating row band
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
        # recursion-depth bar: t=1 -> exit depth (the "run" of reused f_theta)
        ax.plot([1, dm], [y, y], color=ACC, lw=5.5, alpha=0.28,
                solid_capstyle="round", zorder=1)
        for t in range(1, K + 1):
            on = t <= dm
            ax.scatter(t, y, s=115, facecolor=ACC if on else "white",
                       edgecolor=ACC if on else "#bbbbbb",
                       linewidths=1.4, zorder=3)
        ax.text(K + 0.55, y, f"$d_m{{=}}{dm}$", va="center", ha="left",
                fontsize=8.5, color="#00694f", fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#00694f", pad=8)
    ax.set_xlim(0.4, K + 1.6); ax.set_ylim(-0.6, nrow - 0.4)
    ax.set_xticks(range(1, K + 1)); ax.set_xticklabels([f"$t{{=}}{t}$" for t in range(1, K + 1)], fontsize=8)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(row_names[::-1], fontsize=7.2)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("recursion step ($f_\\theta$ reused each step)", fontsize=8)

# shared legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", ls="", mfc=ACC, mec=ACC, ms=8, label="recurses"),
       Line2D([0], [0], marker="o", ls="", mfc="white", mec="#999999", ms=8, label="exited ($d_m$ reached)")]
fig.legend(handles=leg, fontsize=8.5, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.07), columnspacing=1.6)
fig.suptitle("Per-pathway recursion depth $d_m$ on Prostate (multi-omics)",
             fontsize=10.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig("paper/figs/fig2_depth.pdf", bbox_inches="tight", dpi=600)
print("wrote paper/figs/fig2_depth.pdf")
