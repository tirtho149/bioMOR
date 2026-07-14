"""bioMoR model-SIZE scaling study — 5-FOLD CV edition.
Macro-F1 vs transformer parameters, width sweep d_model in {96,136,192,272,352} at
100/15, for Vanilla / MoR-general / bioMoR, single-cell + multi-omics. Every point is
the mean over datasets of the per-dataset 5-fold CV mean; error bar = mean within-dataset
fold SD. Data-driven from results_cv5/scaling_*. Renders partial as jobs land."""
import json, glob, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
WIDTHS = [96, 136, 192, 272, 352]

def _load(f):
    try: return json.load(open(f))
    except: return None

def _cv(d):
    if not isinstance(d, dict): return None
    m = d.get("cv_macro_f1")
    return (m["mean"], m["std"]) if m and m.get("mean") is not None else None

def _agg(files):
    """(mean-over-datasets, mean-fold-SD) from a list of CV jsons."""
    ms, ss = [], []
    for f in files:
        t = _cv(_load(f))
        if t: ms.append(t[0]); ss.append(t[1])
    if not ms: return (None, None)
    return (round(float(np.mean(ms)), 2), round(float(np.mean(ss)), 2))

def _params(files):
    ps = [_load(f).get("transformer_params") for f in files if _load(f)]
    ps = [p for p in ps if p]
    return st.mean(ps) if ps else None

def gen_series(base, model):
    xs, ys, es = [], [], []
    for w in WIDTHS:
        files = glob.glob(f"{ROOT}/results_cv5/{base}/{model}_d{w}/*.json")
        y, e = _agg(files); x = _params(files)
        if y is not None and x is not None:
            xs.append(x); ys.append(y); es.append(e or 0)
    return xs, ys, es

def bio_series(base, base_gen_for_params):
    xs, ys, es = [], [], []
    for w in WIDTHS:
        files = (glob.glob(f"{ROOT}/results_cv5/{base}/d{w}/*/learned_cv.json") +
                 glob.glob(f"{ROOT}/results_cv5/{base}/d{w}/pnet/*/learned_cv.json"))
        y, e = _agg(files)
        x = _params(glob.glob(f"{ROOT}/results_cv5/{base_gen_for_params}/morgen_d{w}/*.json"))
        if y is not None and x is not None:
            xs.append(x); ys.append(y); es.append(e or 0)
    return xs, ys, es

STYLE = {'vanilla': ('#888888','o','Vanilla (independent, N_R=4)'),
         'morgen':  ('#1f77b4','s','MoR-general (shared, N_R=4)'),
         'biomor':  ('#d62728','^','bioMoR (learned graph, N_R=4)')}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
DOMAINS = [("Single-cell (8 datasets)", 'scaling_sc', 'scaling_sc_biomor'),
           ("Multi-omics (P-NET, 3 cohorts)", 'scaling_mo', 'scaling_mo_biomor')]
for ax, (title, bg, bb) in zip(axes, DOMAINS):
    any_pt = False
    for model in ['vanilla', 'morgen', 'biomor']:
        c, mk, lab = STYLE[model]
        if model == 'biomor':
            xs, ys, es = bio_series(bb, bg)
        else:
            xs, ys, es = gen_series(bg, model)
        if xs:
            any_pt = True
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]
            ax.errorbar([x/1000 for x in xs], ys, yerr=es, marker=mk, color=c, label=lab,
                        lw=2, markersize=7, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel("Recursion-stack parameters (K, log scale)")
    ax.set_ylabel("Macro-F1 (5-fold CV mean)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    if any_pt: ax.legend(fontsize=9, loc='lower right')
    else: ax.text(0.5,0.5,"(runs pending)", ha='center', va='center', transform=ax.transAxes, style='italic', color='#999')

fig.suptitle("bioMoR scaling: macro-F1 vs model size (width sweep d_model 96→352) — 5-fold CV (seed 42, 100 ep / patience 15)",
             fontsize=12.5, fontweight='bold')
fig.text(0.5, 0.005,
         "Same 5-fold split protocol as the ladder table (StratifiedKFold-5 seed 42, 20% test / 10%-of-train val). "
         "Error bars = mean within-dataset fold SD. Vanilla stacks N_R=4 independent blocks (4× params); "
         "MoR-general & bioMoR reuse 1 shared block. Left-shifted curve = same accuracy at fewer params.",
         ha='center', fontsize=8, style='italic')
plt.tight_layout(rect=[0,0.03,1,0.96])
out = f"{ROOT}/biomor_scaling_figure_cv5.png"
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white'); print("wrote", out)
