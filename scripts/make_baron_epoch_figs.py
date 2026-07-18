"""Two per-epoch figures on Baron, each overlaying all four architectures
(Vanilla / Recursive / MoR / bioMoR):

  paper/figs/baron_val_f1.pdf : validation macro-F1 vs epoch
  paper/figs/baron_loss.pdf   : training loss       vs epoch

Marker dots are placed at epochs 1, 10, 20, 30, ... Colours are Okabe-Ito (colour-blind
safe) with a distinct marker per architecture. Reads results/cv5/curves/baron_cost.json.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/cv5" / "curves" / "baron_cost.json"
ORDER = ["Vanilla", "Recursive", "MoR", "bioMoR"]
COL = {"Vanilla": "#999999", "Recursive": "#E69F00", "MoR": "#56B4E9", "bioMoR": "#009E73"}
MK = {"Vanilla": "s", "Recursive": "^", "MoR": "D", "bioMoR": "o"}


def mark_idx(h):
    # dots at epochs 1, 10, 20, 30, ...  (epoch field is 1-based)
    return [i for i, e in enumerate(h) if e["epoch"] == 1 or e["epoch"] % 10 == 0]


def _plot(data, key, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name in ORDER:
        if name not in data:
            continue
        h = data[name]["history"]
        ep = [e["epoch"] for e in h]
        yv = [e[key] for e in h]
        ax.plot(ep, yv, color=COL[name], lw=1.6, marker=MK[name], markevery=mark_idx(h),
                ms=4.5, markeredgecolor="black", markeredgewidth=0.4, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / fname, bbox_inches="tight", dpi=600)
    fig.savefig(figs / fname.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")


def main():
    if not SRC.exists():
        print(f"[epoch-figs] {SRC} not ready yet (waiting on job 11665654).")
        return
    data = json.load(open(SRC))
    _plot(data, "val_f1", "validation macro-F1 (%)",
          "Baron: validation macro-F1 vs epoch", "baron_val_f1.pdf")
    _plot(data, "train_loss", "training loss",
          "Baron: training loss vs epoch", "baron_loss.pdf")
    print("[epoch-figs] wrote paper/figs/baron_val_f1.{pdf,png} and baron_loss.{pdf,png}")
    for n in ORDER:
        if n in data:
            h = data[n]["history"]
            print(f"  {n:10s} epochs={len(h):3d} peakVal={max(e['val_f1'] for e in h):.1f} "
                  f"finalLoss={h[-1]['train_loss']:.3f}")


if __name__ == "__main__":
    main()
