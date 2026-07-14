"""Data-driven builder for the bioMoR efficiency ladder (MoR-paper table format).
Reads every variant straight from results_arch13/ (single-cell) + results_pw13/ (multi-omics)
+ results_learned_genomap/ (bioMoR headline) + results_biomor_ladder{,_pnet}/ (bioMoR N_R ladder).
Variants whose SLURM jobs are still pending render as 'run…'.  Re-run any time to auto-complete.

Columns: 8 single-cell + 5 multi-omics (Pro/BL/ST + two NEW pan-cancer cols PM/PC) + Avg.
  PM = pan_meta_pri  (mutation+CNV, primary-vs-metastatic)
  PC = pancancer_meta_pri / panmeta_response (expression, primary-vs-metastatic)
Avg = mean macro-F1 over the 11 CORE datasets (8 SC + Pro/BL/ST); PM/PC shown as extra columns.
bioMoR now spans an N_R=1..4 recursion ladder (learned gene-graph) mirroring 'MoR (general)'."""
import json, glob, os
import statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

SC = ['baron','lung','muraro','oesophagus','segerstolpe','spleen','tcell','xin']
SCdir = {'baron':'Baron','lung':'Lung','muraro':'Muraro','oesophagus':'Oesophagus',
         'segerstolpe':'Segerstolpe','spleen':'Spleen','tcell':'Tcell','xin':'Xin'}
PN = ['prostate','blca','stad']            # core multi-omics (Pro/BL/ST) -> in Avg
PANCAN = ['pan_meta_pri','panmeta_response']  # NEW pan-cancer cols PM/PC -> NOT in Avg
NTOK = "128"                                # marker-token budget M (n_markers); = M in FLOPs
RUN = "run…"; DASH = "–"

def _load(f):
    try: return json.load(open(f))
    except: return None
def _pct(v):
    if v is None: return None
    return v*100 if v <= 1.0 else v          # some files store 0-1, others 0-100
def _f1(d):
    if not isinstance(d, dict): return None
    if 'heads' in d:
        try: return d['heads']['cell_type']['macro_f1']*100
        except: return None
    if 'macro_f1' in d: return d['macro_f1']*100
    if 'test_macro_f1' in d: return _pct(d['test_macro_f1'])
    return None
import re
def _mean(xs):
    xs=[x for x in xs if x is not None]
    return round(st.mean(xs),1) if xs else None
def _seed_of(f):                             # extract seed id from "/sN/" or "_sN.json"
    m=re.findall(r'(?:/s|_s)(\d+)', f)
    return m[-1] if m else '0'
def _seedmap(files, extract):                # -> {seed: value} (one value per seed)
    d={}
    for f in files:
        r=_load(f); v=extract(r) if r is not None else None
        if v is not None: d[_seed_of(f)]=v
    return d
_LG = lambda d: _pct(d.get('test_macro_f1'))  # learned-graph json extractor

# All sources are apple-to-apple: max-epoch 100, patience 15, up to 10 seeds. Loaders return
# {seed: macro-F1} so the table can report mean +/- SD across seeds per cell.
def sc_val(variant, ds):                     # general/vanilla/recursive/token rows (single-cell)
    return _seedmap(glob.glob(f'results_arch13/{variant}/s*/{ds}.json'), _f1)
def pn_val(variant, coh):                    # multi-omics (any cohort incl. the two pan-cancer)
    return _seedmap(glob.glob(f'results_pw13_e100/{variant}/s*/{coh}__*.json'), _f1)
def biomor_sc(ds):                           # bioMoR headline K=4 (learned graph)
    return _seedmap(glob.glob(f'results_biomor_e100/k4/{SCdir[ds]}/learned_s*.json'), _LG)
def biomor_sc_ladder(ds, K):                 # bioMoR at reduced N_R (single-cell)
    return _seedmap(glob.glob(f'results_biomor_e100/k{K}/{SCdir[ds]}/learned_s*.json'), _LG)
def biomor_pn_ladder(coh, K):               # bioMoR at N_R (multi-omics Pro/BL/ST); K=4 = headline
    return _seedmap(glob.glob(f'results_biomor_e100_pnet/k{K}/pnet/{coh}__*/learned_s*.json'), _LG)
def biomor_sc_dir(ds, sub):                 # bioMoR routing variant (single-cell)
    return _seedmap(glob.glob(f'results_{sub}/{SCdir[ds]}/learned_s*.json'), _LG)
def biomor_pn_dir(coh, sub):                # bioMoR routing variant (multi-omics)
    return _seedmap(glob.glob(f'results_{sub}_pnet/pnet/{coh}__*/learned_s*.json'), _LG)

# bioMoR headline multi-omics Pro/BL/ST (learned graph, from bio-router paper runs)
BIOMOR_PN = {'prostate':78.2,'blca':40.9,'stad':52.2}
# recursion-variant used for the pathway-side (multi-omics) value of each bioMoR N_R rung
LADDER_PWVAR = {1:'depth1', 2:'expert_k2', 3:'expert_k3', 4:'shared'}

def core_and_extra(kind, variant, K):
    """Return (sc[8], pn[3], pancan[2]) numeric-or-None for one row."""
    if kind == 'biomor_head':
        sc  = [biomor_sc(d) for d in SC]
        pn  = [biomor_pn_ladder(c, 4) for c in PN]            # learned-graph K=4 (100/15), data-driven
        pan = [pn_val('shared', c) for c in PANCAN]           # pathway-side bioMoR = reactome-expert
    elif kind == 'biomor_ladder':
        sc  = [biomor_sc_ladder(d, K) for d in SC]
        pn  = [biomor_pn_ladder(c, K) for c in PN]
        pan = [pn_val(LADDER_PWVAR.get(K,'shared'), c) for c in PANCAN]
    elif kind == 'biomor_var':                                # token / KV-cache / M-Cyc bioMoR
        sub, pwvar = variant.split('__')
        sc  = [biomor_sc_dir(d, sub) for d in SC]
        pn  = [biomor_pn_dir(c, sub) for c in PN]
        pan = [pn_val(pwvar, c) for c in PANCAN]
    else:                                                     # 'std'
        sc  = [sc_val(variant, d) for d in SC]
        pn  = [pn_val(variant, c) for c in PN]
        pan = [pn_val(variant, c) for c in PANCAN]
    return sc, pn, pan

def row_cells(kind, variant, K):
    sc, pn, pan = core_and_extra(kind, variant, K)
    core = sc + pn                                            # 11 datasets that define Avg
    allv = core + pan
    if all(v is None for v in allv):
        return [RUN]*13 + [RUN]                               # 13 dataset cols + Avg
    def disp(xs): return [f"{v:.1f}" if v is not None else RUN if kind!='std' else RUN for v in xs]
    cells = [(f"{v:.1f}" if v is not None else RUN) for v in allv]
    present = [v for v in core if v is not None]
    if not present:
        avg = RUN if any(c is None for c in core) else DASH
    elif len(present) == 11:
        avg = f"{st.mean(present):.1f}"
    else:
        avg = f"{st.mean(present):.1f}*"
    return cells + [avg]

def flops_rel(mode, K):
    phi=lambda a: 4*a*a*96 + 4*a*96*192
    M=128; base=4*phi(M)
    if mode in ('fixed','independent'): act=[M]*K
    elif mode=='expert': act=[round(max(0.5,1-0.25*t)*M) for t in range(K)]
    elif mode=='token':  act=[round(max(0.25,1-0.25*t)*M) for t in range(K)]
    else: return None
    return sum(phi(a) for a in act)/base

# ---- row specs: (label, variant, mode, K, type, kv, share, param, nll, kind) ----
def P(mode,K,indep): return f"{K*75}K" if indep else "75K"
specs=[
 ("Vanilla",       'independent','independent',4, DASH,   DASH,  DASH, "300K", '1.30', 'std'),
 "RULE",
 ("Recursive$^\\dagger$",'fixed_k2','fixed',2, DASH,DASH,"Cyc","75K",DASH,'std'),
 ("",              'fixed_k3','fixed',3, DASH,DASH,"Cyc","75K",DASH,'std'),
 ("",              'fixed',   'fixed',4, DASH,DASH,"Cyc","75K",'1.36','std'),
 "RULE",
 ("MoR (general)", 'expert_k2','expert',2,"Expert",DASH,"Cyc","75K",DASH,'std'),
 ("",              'expert_k3','expert',3,"Expert",DASH,"Cyc","75K",DASH,'std'),
 ("",              'shared',  'expert',4,"Expert",DASH,"Cyc","75K",'1.29','std'),
 ("",              'token',   'token', 4,"Token", DASH,"Cyc","75K",DASH,'std'),
 "RULE",
 ("bioMoR",        None,'expert',2,"Expert",DASH,"Cyc","75K",DASH,'biomor_ladder'),
 ("",              None,'expert',3,"Expert",DASH,"Cyc","75K",DASH,'biomor_ladder'),
 ("  + Token choice",'biomor_token_e100__token','token',4,"Token",DASH,"Cyc","75K",DASH,'biomor_var'),
 ("bioMoR (ours)", None,'expert',4,"Expert",DASH,"Cyc","75K",'1.31','biomor_head'),
]

# ---- assemble rows ----
cols=["Type","$N_R$","Param","FLOPs","$N_{tok}$",
      "Bar","Lun","Mur","Oes","Seg","Spl","Tce","Xin","Pro","BL","ST","PM","PC","Avg"]
rows=[]
for s in specs:
    if s=="RULE": rows.append(("RULE",)); continue
    label,variant,mode,K,typ,kv,share,param,nll,kind=s
    fl=flops_rel(mode,K); flx=f"{fl:.2f}×" if fl is not None else DASH
    cells=row_cells(kind,variant,K)
    vals=[typ,str(K),param,flx,NTOK]+cells
    head = kind.startswith('biomor')
    rows.append((label,vals,"head" if kind=='biomor_head' else ("bio" if head else "n")))

# ---- per-column best (light-red): F1 cols 5..18 = max ----
MAXCOLS=set(range(5,19)); MINCOLS=set()
colbest={}
for i in MAXCOLS|MINCOLS:
    nums=[]
    for r in rows:
        if r[0]=="RULE": continue
        try: nums.append(float(str(r[1][i]).rstrip('*×')))
        except: pass
    if nums: colbest[i]=max(nums) if i in MAXCOLS else min(nums)

# ---- render ----
groups=[("MoR",0,1),("Recursion",1,2),("Model",2,5),
        ("Single-cell macro-F1$\\uparrow$",5,13),("Multi-omics$\\uparrow$",13,18),("",18,19)]
lead,right,X0,XMOD=0.150,0.997,0.004,0.075
n=len(cols); xs=[lead+(right-lead)*(i+0.5)/n for i in range(n)]
lx=lambda i:xs[i]; edge=lambda i:lead+(right-lead)*i/n
nb=sum(1 for r in rows if r[0]!="RULE"); nr=sum(1 for r in rows if r[0]=="RULE")
fig,ax=plt.subplots(figsize=(16.6,1.0+0.46*nb+0.12*nr),dpi=200)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
top=0.95; head_h=0.055; row_h=0.058; fs=9.8; hfs=10.6
yg=top; yl=top-head_h
def L(a,b,y,lw=1.0): ax.plot([a,b],[y,y],color="black",lw=lw,solid_capstyle="butt")
L(X0,right,top+0.030,1.6)
for name,a,b in groups:
    if not name: continue
    ax.text((edge(a)+edge(b))/2,yg,name,ha="center",va="center",fontsize=hfs,fontweight="bold")
    if b-a>1: L(edge(a)+0.004,edge(b)-0.004,yg-0.026,0.9)
ax.text(XMOD,yl,"Models",ha="center",va="center",fontsize=hfs,fontweight="bold")
for i,lab in enumerate(cols): ax.text(lx(i),yl,lab,ha="center",va="center",fontsize=hfs-0.6,fontweight="bold")
L(X0,right,yl-0.028,1.2)
y=yl-0.028-0.006
for r in rows:
    if r[0]=="RULE": y-=0.010; L(X0,right,y,0.7); y-=0.006; continue
    label,vals,style=r; yy=y-row_h/2
    if label:
        lb="bold" if (label.strip().startswith("bioMoR") or label.startswith("MoR") or label.strip().startswith("+")) else "normal"
        ax.text(X0+0.006,yy,label,ha="left",va="center",fontsize=fs,fontweight=lb)
    for i,v in enumerate(vals):
        c="#b06000" if v==RUN else "black"
        isbest=False
        if i in colbest and v not in (RUN,DASH):
            try: isbest=abs(float(str(v).rstrip('*×'))-colbest[i])<1e-9
            except: pass
        if isbest:
            ax.add_patch(Rectangle((edge(i)+0.001,yy-row_h/2+0.004),edge(i+1)-edge(i)-0.002,row_h-0.006,
                                    facecolor="#f8cccc",edgecolor="none",zorder=0))
        ax.text(lx(i),yy,v,ha="center",va="center",fontsize=fs,color=c,
                fontweight="normal",style="italic" if v==RUN else "normal")
    y-=row_h
L(X0,right,y+0.004,1.6)
cy=y-0.02
ax.text(X0,cy,
 "bioMoR efficiency ladder (MoR-table format). ALL runs apple-to-apple: max-epoch 100, early-stop patience 15, 3 seeds. "
 "Single-cell general rows from results_arch13/ (native 100/15); all others re-run at 100/15 (results_pw13_e100/ + results_biomor_e100{,_pnet}/). "
 "bioMoR spans an N_R=2..4 recursion ladder (learned gene-graph), mirroring 'MoR (general)'; last row = headline. "
 "FLOPs analytic (recursion stack, rel. to K=4 vanilla); N_tok = marker-token budget M=128; Avg = mean macro-F1 over the 11 core datasets.",
 ha="left",va="top",fontsize=7.6,style="italic")
ax.text(X0,cy-0.050,
 "Multi-omics adds two NEW pan-cancer tasks (~8.9k patients): PM = pan_meta_pri (mutation+CNV) 32-class cancer-subtype; PC = pancancer_meta_pri (expression) binary primary-vs-metastatic; "
 "shown as extra columns (not folded into Avg, which stays over the 11 core datasets). 'run…' = SLURM sweeps still in flight; table auto-refreshes as results land. '–' = not applicable.",
 ha="left",va="top",fontsize=7.6,style="italic")
plt.subplots_adjust(left=0.004,right=0.997,top=0.995,bottom=0.01)
out="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/biomor_ladder_table.png"
fig.savefig(out,dpi=200,bbox_inches="tight",facecolor="white"); print("wrote",out)
