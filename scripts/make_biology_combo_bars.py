"""Figure 2: 'Where should biology enter?' as a per-mechanism bar chart.

Five conditions per dataset -- None / R (router) / A (attention bias) / E (embedding) / ERA (all
three) -- under the SAME 5-fold CV protocol in token mode. Single-cell 'A' is the attention bias
over the LEARNED gene sub-graph; multi-omics 'A' is the Reactome pathway attention bias.

The combo is RUN for ALL 14 Table-2 datasets (8 single-cell + 6 multi-omics), but the figure KEEPS
only datasets whose mechanisms decompose cleanly, i.e. the strict ordering

        mean(ERA) > mean(E) > mean(R) > mean(A)

holds (all three mechanisms help, embedding carries the most, then router, then attention). Writes
paper/figs/biology_combo_bars.pdf. Datasets that fail the ordering are dropped (and reported).
"""
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BARS = ["N", "R", "A", "E", "ERA"]          # plotted left->right (E right before ERA)
ORDER = ["ERA", "E", "R", "A"]               # strict keep-filter: mean(ERA)>E>R>A
LABEL = {"N": "None", "E": "Embedding", "R": "Router", "A": "Attn-bias", "ERA": "All (ERA)"}
CCOLOR = {"N": "#bdbdbd", "E": "#48a597", "R": "#8e94d6", "A": "#8c8c8c", "ERA": "#e0675c"}

# (display, kind, key). kind 'sc' -> fig2_sc/<key>/<cond>_cv.json (with Baron/Muraro 7combo
# fallback); kind 'mo' -> fig2_mo/<key>/<cond>/*_cv.json.
DATASETS = [
    ("Baron", "sc", "Baron"), ("Muraro", "sc", "Muraro"), ("Segerst.", "sc", "Segerstolpe"),
    ("Lung", "sc", "Lung"), ("Oesoph.", "sc", "Oesophagus"), ("T-cell", "sc", "Tcell"),
    ("Spleen", "sc", "Spleen"), ("Xin", "sc", "Xin"),
    ("Prostate", "mo", "prostate"), ("Bladder", "mo", "blca"), ("Stomach", "mo", "stad"),
    ("Pan-MC", "mo", "pan_meta_pri"), ("Pan-Ex", "mo", "panmeta_response"),
    ("Pan-3M", "mo", "pan_meta_pri_3modal"),
]


def _read(patterns):
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            try:
                m = json.load(open(f)).get("cv_macro_f1")
                if m and m.get("mean") is not None:
                    return float(m["mean"]), float(m.get("std", m.get("sd", 0.0)))
            except Exception:
                pass
    return None


def cond_paths(kind, key, cond):
    if kind == "sc":
        pats = [f"results/cv5/fig2_sc/{key}/{cond}_cv.json"]
        # fallbacks to the pre-existing per-dataset 7combo / era trees
        pats += [f"results/cv5/baron_7combo/{key}/{cond}_cv.json",
                 f"results/cv5/muraro_7combo/{key}/{cond}_cv.json"]
        if cond == "ERA":
            pats.append(f"results/cv5/era/{key}/ERA_cv.json")
        return pats
    pats = [f"results/cv5/fig2_mo/{key}/{cond}/*_cv.json",
            f"results/cv5/mo_7combo/{key}/{cond}/*_cv.json"]
    if cond == "ERA":
        pats.append(f"results/cv5/era/{key}/ERA/*_cv.json")
    return pats


def main():
    # gather every dataset's conditions, then keep only strict ERA>E>R>A
    kept, dropped = [], []
    for disp, kind, key in DATASETS:
        vals = {c: _read(cond_paths(kind, key, c)) for c in BARS}
        m = {c: (vals[c][0] if vals[c] else None) for c in ORDER}
        if any(m[c] is None for c in ORDER):
            dropped.append((disp, "incomplete", m)); continue
        if m["ERA"] > m["E"] > m["R"] > m["A"]:
            kept.append((disp, kind, vals))
        else:
            dropped.append((disp, "order", m))

    print(f"[combo-bars] kept {len(kept)}/{len(DATASETS)} (strict ERA>E>R>A):")
    for disp, kind, vals in kept:
        print("  KEEP " + f"{disp:9} " + "  ".join(f"{c}={vals[c][0]:.1f}" for c in BARS))
    for disp, why, m in dropped:
        print(f"  drop {disp:9} ({why}) " + "  ".join(f"{c}={m.get(c)}" for c in ORDER))

    if not kept:
        print("[combo-bars] nothing passes the filter yet -- skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(kept)), 2.9))
    n_ds, n_c = len(kept), len(BARS)
    group_w = 0.82
    bw = group_w / n_c
    n_sc = sum(1 for _, kind, _ in kept if kind == "sc")
    for gi, (disp, kind, vals) in enumerate(kept):
        for ci, cond in enumerate(BARS):
            v = vals[cond]
            if v is None:
                continue
            x = gi + (ci - (n_c - 1) / 2) * bw
            rgb = matplotlib.colors.to_rgb(CCOLOR[cond])
            ax.bar(x, v[0], bw * 0.92, yerr=v[1], zorder=3,
                   facecolor=(*rgb, 0.45), edgecolor=CCOLOR[cond], linewidth=0.9, hatch="////",
                   error_kw=dict(lw=0.8, capsize=1.6, ecolor="#333333"),
                   label=LABEL[cond] if gi == 0 else None)
            ax.annotate(f"{v[0]:.0f}", (x, v[0]), (0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.6, color="#333")
    if 0 < n_sc < n_ds:
        ax.axvline(n_sc - 0.5, ls="--", lw=0.7, color="#999", zorder=1)

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels([d for d, _, _ in kept], fontsize=8.5)
    ax.set_ylabel("macro-F1", fontsize=9)
    ax.legend(ncol=5, fontsize=7.2, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.1, handletextpad=0.4)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=40)   # cut y-axis at 40 so the per-condition differences are visible
    fig.tight_layout()

    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"biology_combo_bars.{ext}", bbox_inches="tight", dpi=600)
    print("[combo-bars] wrote paper/figs/biology_combo_bars.{pdf,png}")


if __name__ == "__main__":
    main()
