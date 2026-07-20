"""Baron accuracy-vs-cost POINT diagram (replaces the two-panel training-cost curves in the
main paper; the full curves move to the supplementary appendix).

For each architecture we mark TWO operating points on a single (cost, macro-F1) plane:
  * best-loss  point (o): the checkpoint with the lowest training loss  -> (cost, macro-F1 there)
  * best-acc   point (*): the checkpoint with the highest val macro-F1  -> (cost, best macro-F1)
A thin connector shows the loss->accuracy operating-point shift. Y axis = macro-F1 (the accuracy
metric; the early-stopping/test metric); X axis = cumulative A100 GPU cost (s). Reads the same
results/cv5/curves/baron_cost_cv5.json the curves use, so it is in parity with the paper.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/cv5/curves/baron_cost_cv5.json"
ORDER = ["Vanilla", "Recursive", "MoR", "bioMoR"]
COLOR = {"Vanilla": "#7f7f7f", "Recursive": "#ff7f0e", "MoR": "#2ca02c", "bioMoR": "#1f77b4"}


def main():
    if not SRC.exists():
        print(f"[pointdiagram] {SRC} not found"); return
    d = json.load(open(SRC))

    # label placement offsets (dx in seconds, dy in F1 pts, ha) tuned so text clears the markers
    LAB = {"Vanilla": (1.2, 1.6, "left"), "Recursive": (-1.2, -2.6, "right"),
           "MoR": (1.4, -2.4, "left"), "bioMoR": (1.4, 1.4, "left")}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xs, ys = [], []
    for name in ORDER:
        if name not in d:
            continue
        agg = [e for e in d[name]["agg"] if e.get("sec_mean") is not None]
        if not agg:
            continue
        bl = min(agg, key=lambda e: e["loss_mean"])       # best training loss checkpoint
        ba = max(agg, key=lambda e: e["val_f1_mean"])     # best val macro-F1 checkpoint
        test = d[name].get("test_f1_mean")
        c = COLOR[name]
        xb, yb = bl["sec_mean"], bl["val_f1_mean"]
        xa, ya = ba["sec_mean"], ba["val_f1_mean"]
        xs += [xb, xa]; ys += [yb, ya]
        ax.plot([xb, xa], [yb, ya], color=c, lw=1.1, alpha=0.5, zorder=1)
        ax.scatter(xb, yb, facecolors="white", edgecolors=c, s=46, lw=1.7, marker="o", zorder=3)
        ax.scatter(xa, ya, color=c, s=115, marker="*", edgecolors="white", lw=0.5, zorder=4)
        dx, dy, ha = LAB.get(name, (1.2, 1.4, "left"))
        ax.annotate(f"{name} ({test:.1f})", (xa, ya), (xa + dx, ya + dy), ha=ha, va="center",
                    fontsize=7.4, fontweight="bold", color=c, zorder=5)

    # single compact marker-shape key placed in the empty upper-right; NOT over any point
    from matplotlib.lines import Line2D
    key = [Line2D([0], [0], marker="o", mfc="white", mec="#444", ls="none", ms=6.5, label="best loss"),
           Line2D([0], [0], marker="*", color="#444", ls="none", ms=10, label="best accuracy")]
    ax.legend(handles=key, loc="upper right", fontsize=7, frameon=True, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.3)

    ax.set_xlabel("cumulative A100 GPU cost (s)", fontsize=8.5)
    ax.set_ylabel("macro-F1", fontsize=8.5)
    xlo, xhi = min(xs), max(xs); ylo, yhi = min(ys), max(ys)
    ax.set_xlim(xlo - 0.06 * (xhi - xlo), xhi + 0.12 * (xhi - xlo))
    ax.set_ylim(ylo - 0.10 * (yhi - ylo), yhi + 0.16 * (yhi - ylo))   # headroom for top star
    ax.tick_params(labelsize=7.5)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"baron_pointdiagram.{ext}", bbox_inches="tight", dpi=600)
    print("[pointdiagram] wrote paper/figs/baron_pointdiagram.{pdf,png}")
    for name in ORDER:
        if name in d:
            agg = d[name]["agg"]
            bl = min(agg, key=lambda e: e["loss_mean"]); ba = max(agg, key=lambda e: e["val_f1_mean"])
            print(f"  {name:10} best-loss=(cost {bl['sec_mean']:.0f}s, F1 {bl['val_f1_mean']:.1f})  "
                  f"best-acc=(cost {ba['sec_mean']:.0f}s, F1 {ba['val_f1_mean']:.1f})  test {d[name]['test_f1_mean']:.1f}")


if __name__ == "__main__":
    main()
