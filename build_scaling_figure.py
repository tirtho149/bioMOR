"""bioMoR scaling study: macro-F1 vs transformer parameters, width sweep d_model in
{96,136,192,272,352} at 100/15, for Vanilla / MoR-general / bioMoR, single-cell + multi-omics.
Data-driven from results_scaling_sc{,_biomor}/ + results_scaling_mo{,_biomor}/. Renders partial
as jobs land. x = recursion-stack params (Vanilla=4xblock, MoR-general/bioMoR=1xblock, shared)."""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

WIDTHS = [96, 136, 192, 272, 352]
def pct(v): return None if v is None else (v*100 if v <= 1 else v)
def _m(xs): xs=[x for x in xs if x is not None]; return (round(st.mean(xs),2), (round(st.pstdev(xs),2) if len(xs)>1 else 0.0)) if xs else (None,None)
def _load(f):
    try: return json.load(open(f))
    except: return None
def _hf1(d):
    if not isinstance(d,dict): return None
    if 'heads' in d:
        try: return d['heads']['cell_type']['macro_f1']*100
        except: return None
    if 'macro_f1' in d: return d['macro_f1']*100
    if 'test_macro_f1' in d: return pct(d['test_macro_f1'])
    return None

def gen_f1(base, model, w):     # vanilla/morgen from singlecell or pathway_tasks jsons
    fs=glob.glob(f'results_{base}/{model}_d{w}/s*/*.json')
    return _m([_hf1(_load(f)) for f in fs])
def gen_params(base, model, w):
    ps=[_load(f).get('transformer_params') for f in glob.glob(f'results_{base}/{model}_d{w}/s*/*.json') if _load(f)]
    ps=[p for p in ps if p]; return st.mean(ps) if ps else None
def bio_f1(base, w):            # bioMoR from learned-graph jsons (no transformer_params stored)
    fs=glob.glob(f'results_{base}/d{w}/*/learned_s*.json') + glob.glob(f'results_{base}/d{w}/pnet/*/learned_s*.json')
    return _m([pct(_load(f).get('test_macro_f1')) for f in fs if _load(f)])

def series(base_gen, base_bio, model):
    """Return (xs, ys, es) across widths for one model in one domain."""
    xs, ys, es = [], [], []
    for w in WIDTHS:
        if model == 'biomor':
            y,e = bio_f1(base_bio, w); x = gen_params(base_gen, 'morgen', w)   # same shared block as morgen
        else:
            y,e = gen_f1(base_gen, model, w); x = gen_params(base_gen, model, w)
        if y is not None and x is not None:
            xs.append(x); ys.append(y); es.append(e or 0)
    return xs, ys, es

STYLE = {'vanilla': ('#888888','o','Vanilla (independent, N_R=4)'),
         'morgen':  ('#1f77b4','s','MoR-general (shared, N_R=4)'),
         'biomor':  ('#d62728','^','bioMoR (learned graph, N_R=4)')}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
for ax, (title, bg, bb) in zip(axes, [("Single-cell (8 datasets)", 'scaling_sc', 'scaling_sc_biomor'),
                                       ("Multi-omics (P-NET, 3 cohorts)", 'scaling_mo', 'scaling_mo_biomor')]):
    any_pt=False
    for model in ['vanilla','morgen','biomor']:
        c,mk,lab = STYLE[model]
        xs,ys,es = series(bg, bb, model)
        if xs:
            any_pt=True
            order=sorted(range(len(xs)), key=lambda i: xs[i])
            xs=[xs[i] for i in order]; ys=[ys[i] for i in order]; es=[es[i] for i in order]
            ax.errorbar([x/1000 for x in xs], ys, yerr=es, marker=mk, color=c, label=lab,
                        lw=2, markersize=7, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel("Recursion-stack parameters (K, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    if any_pt: ax.legend(fontsize=9, loc='lower right')
    else: ax.text(0.5,0.5,"(runs pending)", ha='center', va='center', transform=ax.transAxes, style='italic', color='#999')

fig.suptitle("bioMoR scaling: macro-F1 vs model size (width sweep d_model 96→352, 100 epochs / patience 15, 3 seeds)",
             fontsize=13, fontweight='bold')
fig.text(0.5, 0.005,
         "Vanilla stacks N_R=4 independent blocks (4× params); MoR-general & bioMoR reuse 1 shared block. "
         "Left-shifted curve = same accuracy at fewer params. bioMoR x-position uses the shared-block size (= MoR-general).",
         ha='center', fontsize=8, style='italic')
plt.tight_layout(rect=[0,0.03,1,0.96])
out="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/biomor_scaling_figure.png"
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white'); print("wrote", out)
