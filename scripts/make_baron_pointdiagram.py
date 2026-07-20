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

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for name in ORDER:
        if name not in d:
            continue
        agg = [e for e in d[name]["agg"] if e.get("sec_mean") is not None]
        if not agg:
            continue
        # best-loss checkpoint (min training loss) and best-accuracy checkpoint (max val F1)
        bl = min(agg, key=lambda e: e["loss_mean"])
        ba = max(agg, key=lambda e: e["val_f1_mean"])
        test = d[name].get("test_f1_mean")
        c = COLOR[name]
        # connector between the two operating points of this architecture
        ax.plot([bl["sec_mean"], ba["sec_mean"]], [bl["val_f1_mean"], ba["val_f1_mean"]],
                color=c, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter(bl["sec_mean"], bl["val_f1_mean"], facecolors="none", edgecolors=c,
                   s=52, lw=1.6, marker="o", zorder=3)
        ax.scatter(ba["sec_mean"], ba["val_f1_mean"], color=c, s=90, marker="*", zorder=3,
                   label=f"{name} (test {test:.1f})" if test is not None else name)

    # marker legend (best-loss vs best-accuracy), separate from the colour/arch legend
    from matplotlib.lines import Line2D
    shape_leg = [Line2D([0], [0], marker="o", mfc="none", mec="#333", ls="none", ms=7, label="best loss"),
                 Line2D([0], [0], marker="*", color="#333", ls="none", ms=10, label="best accuracy")]
    leg1 = ax.legend(loc="lower right", fontsize=6.6, frameon=True, title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=shape_leg, loc="upper left", fontsize=6.8, frameon=True)

    ax.set_xlabel("cumulative A100 GPU cost (s)", fontsize=8.5)
    ax.set_ylabel("macro-F1", fontsize=8.5)
    ax.set_xscale("log")
    ax.margins(y=0.14)                       # headroom so the top (bioMoR) star isn't clipped
    ax.tick_params(labelsize=7.5)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
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
