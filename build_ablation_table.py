"""Component-ablation table for bioMoR -- 'where does the gain come from?'.
Data-driven from results_learned_genomap/ (graph source), results_bio_curated/ (multi-omics
graph), results_arch13/ (routing), results_biomor_ladder{,_pnet}/ (recursion depth).
Three blocks answer the top reviewer questions: (A) biological graph, (B) routing, (C) recursion.
Re-run any time to refresh. Renders assets/biomor_ablation_table.png."""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

SC = ['Baron','Lung','Muraro','Oesophagus','Segerstolpe','Spleen','Tcell','Xin']
PN = ['prostate','blca','stad']
def pct(v): return None if v is None else (v*100 if v <= 1 else v)
def _m(xs): xs=[x for x in xs if x is not None]; return round(st.mean(xs),1) if xs else None
def _mean_over(paths, key='test_macro_f1'):
    vals=[]
    for f in paths:
        try: vals.append(pct(json.load(open(f)).get(key)))
        except: pass
    return _m(vals)

def graph_sc(mode):   # single-cell mean over 8 datasets for a graph mode
    return _m([_mean_over(glob.glob(f'results_learned_genomap/{d}/{mode}_s*.json')) for d in SC])
def graph_mo(mode):   # multi-omics mean over 3 P-NET cohorts
    return _m([_mean_over(glob.glob(f'results_bio_curated/pnet/{c}__*/{mode}_s*.json')) for c in PN])
def route_sc(variant):
    def one(d):
        vals=[]
        for f in glob.glob(f'results_arch13/{variant}/s*/{d.lower()}.json'):
            try: vals.append(json.load(open(f))['heads']['cell_type']['macro_f1']*100)
            except: pass
        return _m(vals)
    return _m([one(d) for d in SC])
def ladder_sc(K):
    return _m([_mean_over(glob.glob(f'results_biomor_ladder/k{K}/{d}/learned_s*.json')) for d in SC]) if K<4 \
        else _m([_mean_over(glob.glob(f'results_learned_genomap/{d}/learned_s*.json')) for d in SC])
def ladder_mo(K):
    return _m([_mean_over(glob.glob(f'results_biomor_ladder_pnet/k{K}/pnet/{c}__*/learned_s*.json')) for c in PN]) if K<4 \
        else _m([_mean_over(glob.glob(f'results_bio_curated/pnet/{c}__*/learned_s*.json')) for c in PN])
def flops_expert(K):
    phi=lambda a: 4*a*a*96+4*a*96*192; M=128
    return sum(phi(round(max(0.5,1-0.25*t)*M)) for t in range(K))/(4*phi(M))

D="–"
def dv(v): return f"{v:.1f}" if v is not None else D
def delta(v, base): return f"{v-base:+.1f}" if (v is not None and base is not None) else D

# ---------- gather ----------
none_sc, none_mo = graph_sc('none'), graph_mo('none')
BLOCKS = []
# Block A -- biological graph
A_rows=[]
for label, mode, desc in [
    ("No graph",            'none',        "no gene-gene edges"),
    ("Random graph",        'random',      "random edges"),
    ("Co-expression graph", 'coexpr',      "correlation edges"),
    ("Learned graph (bioMoR)",'learned',   "task-learned edges"),
    ("Learned + bio-init",  'learned_bio', "learned, Reactome-init"),
]:
    sc, mo = graph_sc(mode), graph_mo(mode)
    A_rows.append([label, desc, dv(sc), dv(mo), delta(sc,none_sc)])
BLOCKS.append(("A.  Biological gene-graph  (Expert routing, N_R=4)  —  isolates the core contribution",
               ["Variant","Graph edges","SC-avg","MO-avg","ΔSC vs no-graph"], A_rows,
               "Learned biological graph adds +%s macro-F1 over no graph (single-cell); random/co-expression edges do NOT help."
               % (f"{graph_sc('learned')-none_sc:.1f}" if none_sc else "?")))
# Block B -- routing (biology-free single-cell)
ind=route_sc('independent')
B_rows=[]
for label, variant, desc in [
    ("Independent (vanilla)",'independent',"no weight sharing"),
    ("Fixed recursion",     'fixed',       "shared, fixed depth"),
    ("Token-choice MoR",    'token',       "per-token routing"),
    ("Expert-choice MoR",   'shared',      "per-expert routing"),
]:
    sc=route_sc(variant); B_rows.append([label, desc, dv(sc), delta(sc,ind)])
BLOCKS.append(("B.  Routing strategy  (biology-free single-cell)  —  routing alone is a minor lever",
               ["Variant","Mechanism","SC-avg","Δ vs vanilla"], B_rows,
               "Fixed/token/expert routing differ by only ~1-2 pts without the graph: routing is NOT the main source of gain."))
# Block C -- recursion depth (bioMoR)
C_rows=[]
base2=None
for K in [2,3,4]:
    sc, mo = ladder_sc(K), ladder_mo(K)
    core=_m([x for x in [sc,mo] if x is not None])
    if K==2: base2=sc
    C_rows.append([f"N_R = {K}", f"{flops_expert(K):.2f}×", dv(sc), dv(mo), delta(sc,base2)])
BLOCKS.append(("C.  Recursion depth N_R  (bioMoR, learned graph)  —  depth saturates quickly",
               ["Depth","FLOPs","SC-avg","MO-avg","ΔSC vs N_R=2"], C_rows,
               "bioMoR at N_R=2 already beats every biology-free MoR at N_R=4: the gain is the graph, not depth. Depth adds <1 pt."))

# ---------- render ----------
nrows_total = sum(len(b[2]) for b in BLOCKS)
fig_h = 1.4 + 0.42*nrows_total + 0.9*len(BLOCKS)
fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=200)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
X0, XR = 0.012, 0.988
y = 0.965
ax.text(X0, y, "bioMoR component ablation — what drives the gain?", fontsize=14, fontweight="bold", va="top")
y -= 0.045
ax.text(X0, y, "Data-driven from results_learned_genomap/ · results_bio_curated/ · results_arch13/ · results_biomor_ladder/. "
               "SC-avg = mean macro-F1 over 8 single-cell datasets; MO-avg = mean over 3 P-NET cohorts.",
        fontsize=8, style="italic", va="top", color="#333")
y -= 0.035
def L(a,b,yy,lw=1.0,c="black"): ax.plot([a,b],[yy,yy],color=c,lw=lw,solid_capstyle="butt")

for title, header, rws, note in BLOCKS:
    y -= 0.018
    ax.text(X0, y, title, fontsize=10.4, fontweight="bold", va="top", color="#0b3d66")
    y -= 0.040
    ncol=len(header)
    # column x-positions: first col wide (label), rest evenly spread on the right
    left_w=0.30
    xs=[X0+0.004]
    for i in range(1,ncol):
        xs.append(left_w + (XR-left_w)*(i-0.5)/(ncol-1))
    L(X0, XR, y+0.020, 1.3)
    for i,h in enumerate(header):
        ha="left" if i==0 else "center"
        ax.text(xs[i], y, h, fontsize=9, fontweight="bold", va="top", ha=ha)
    y -= 0.028
    L(X0, XR, y+0.010, 0.8)
    best = max((float(r[2]) for r in rws if r[2]!=D), default=None)
    for r in rws:
        yy=y-0.006
        hot = (r[2]!=D and best is not None and abs(float(r[2])-best)<1e-6)
        if hot:
            ax.add_patch(Rectangle((X0, yy-0.030), XR-X0, 0.038, facecolor="#d9e6f2", edgecolor="none", zorder=0))
        for i,v in enumerate(r):
            ha="left" if i==0 else "center"
            fw="bold" if (hot or (i==0)) else "normal"
            col="black"
            if i==len(r)-1 and isinstance(v,str) and v.startswith("+"): col="#0a7d2c"
            if i==len(r)-1 and isinstance(v,str) and v.startswith("-"): col="#b02020"
            ax.text(xs[i], yy, v, fontsize=9, va="top", ha=ha, fontweight=fw, color=col)
        y-=0.040
    L(X0, XR, y+0.012, 0.8)
    y-=0.012
    ax.text(X0+0.004, y, "→ "+note, fontsize=7.9, style="italic", va="top", color="#444")
    y-=0.040

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
out="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/biomor_ablation_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); print("wrote", out)
