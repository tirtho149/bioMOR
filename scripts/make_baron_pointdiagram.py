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

    # label offset (dx sec, dy F1, ha, va) so the model name clears its markers
    LAB = {"Vanilla": (0.8, 2.1, "left", "bottom"), "Recursive": (0.8, 2.1, "left", "bottom"),
           "MoR": (1.0, -2.3, "left", "top"), "bioMoR": (0.8, 2.1, "left", "bottom")}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xs = []
    for name in ORDER:
        if name not in d:
            continue
        agg = [e for e in d[name]["agg"] if e.get("sec_mean") is not None]
        if not agg:
            continue
        # cost (x) at the best-loss and best-accuracy checkpoints; y = the model's TEST accuracy
        xb = min(agg, key=lambda e: e["loss_mean"])["sec_mean"]       # cost at best training loss
        xa = max(agg, key=lambda e: e["val_f1_mean"])["sec_mean"]     # cost at best val-F1 (early stop)
        yt = d[name].get("test_f1_mean")
        ysd = d[name].get("test_f1_sd", 0.0)
        c = COLOR[name]
        xs += [xb, xa]
        # test-accuracy band (+/-1 SD) across the two cost operating points
        ax.fill_between([min(xb, xa), max(xb, xa)], yt - ysd, yt + ysd, color=c, alpha=0.10, zorder=0)
        ax.plot([xb, xa], [yt, yt], color=c, lw=1.1, alpha=0.55, zorder=1)
        ax.scatter(xb, yt, facecolors="white", edgecolors=c, s=46, lw=1.7, marker="o", zorder=3)
        ax.scatter(xa, yt, color=c, s=120, marker="*", edgecolors="white", lw=0.5, zorder=4)
        dx, dy, ha, va = LAB.get(name, (0.8, 2.1, "left", "bottom"))
        ax.annotate(f"{name} ({yt:.1f})", (max(xb, xa), yt), (max(xb, xa) + dx, yt + dy),
                    ha=ha, va=va, fontsize=7.4, fontweight="bold", color=c, zorder=5)

    from matplotlib.lines import Line2D
    key = [Line2D([0], [0], marker="o", mfc="white", mec="#444", ls="none", ms=6.5, label="best loss"),
           Line2D([0], [0], marker="*", color="#444", ls="none", ms=10, label="best accuracy")]
    ax.legend(handles=key, loc="center right", fontsize=7, frameon=True, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.3, title="checkpoint (cost)", title_fontsize=6.8)

    ax.set_xlabel("cumulative A100 GPU cost (s)", fontsize=8.5)
    ax.set_ylabel("test macro-F1", fontsize=8.5)
    xlo, xhi = min(xs), max(xs)
    ax.set_xlim(xlo - 0.05 * (xhi - xlo), xhi + 0.16 * (xhi - xlo))
    ax.margins(y=0.20)
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
