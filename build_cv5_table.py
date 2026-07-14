"""bioMoR efficiency ladder — 5-FOLD CV edition (mean +/- SD in every cell).

Reads ONLY results_cv5/ (the unified 5-fold CV re-run: seed 42, 20% test,
10%-of-train val, epochs=100 patience=15). Every cell is macro-F1 mean +/- SD
over the 5 identical folds. Variants whose SLURM tasks are still running render
as 'run…'. Re-run any time to auto-complete.

Layout mirrors biomor_ladder_table.png: 8 single-cell + 5 multi-omics (Pro/BL/ST
core + PM/PC extra) + Avg. Avg = mean over the 11 core datasets (8 SC + Pro/BL/ST).
"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
SC = ['baron','lung','muraro','oesophagus','segerstolpe','spleen','tcell','xin']
SCdir = {'baron':'Baron','lung':'Lung','muraro':'Muraro','oesophagus':'Oesophagus',
         'segerstolpe':'Segerstolpe','spleen':'Spleen','tcell':'Tcell','xin':'Xin'}
PN = ['prostate','blca','stad']                 # core multi-omics -> in Avg
PANCAN = ['pan_meta_pri','panmeta_response']     # extra cols PM/PC -> NOT in Avg
NTOK = "128"; RUN = "run…"; DASH = "–"

# ---- unified CV-json reader: returns (mean, std) in percent, or None ----
def _read(path_glob):
    fs = sorted(glob.glob(os.path.join(ROOT, path_glob)))
    for f in fs:
        try:
            d = json.load(open(f))
            m = d.get("cv_macro_f1")
            if m and m.get("mean") is not None:
                return (float(m["mean"]), float(m["std"]))
        except Exception:
            pass
    return None

def sc_gen(variant, ds):     return _read(f"results_cv5/sc/{variant}/{ds}.json")
def sc_biomor(ds, K):        return _read(f"results_cv5/biomor_sc/k{K}/{SCdir[ds]}/learned_cv.json")
def sc_biomor_tok(ds):       return _read(f"results_cv5/biomor_sc_token/{SCdir[ds]}/learned_cv.json")
def mo_gen(variant, task):   return _read(f"results_cv5/mo/{variant}/{task}__*_cv.json")
def mo_biomor(coh, K):       return _read(f"results_cv5/biomor_mo/k{K}/pnet/{coh}__response/learned_cv.json")
def mo_biomor_tok(coh):      return _read(f"results_cv5/biomor_mo_token/pnet/{coh}__response/learned_cv.json")

# pathway-side (PM/PC) variant reused by each bioMoR rung (mirrors the legacy table)
LADDER_PWVAR = {2:'expert_k2', 3:'expert_k3', 4:'shared'}

def core_and_extra(kind, variant, K):
    """Return (sc[8], pn[3], pancan[2]) each a (mean,std) tuple or None."""
    if kind == 'biomor_head':          # bioMoR (ours) K=4
        sc  = [sc_biomor(d, 4) for d in SC]
        pn  = [mo_biomor(c, 4) for c in PN]
        pan = [mo_gen('shared', c) for c in PANCAN]
    elif kind == 'biomor_ladder':      # bioMoR K=2/3
        sc  = [sc_biomor(d, K) for d in SC]
        pn  = [mo_biomor(c, K) for c in PN]
        pan = [mo_gen(LADDER_PWVAR[K], c) for c in PANCAN]
    elif kind == 'biomor_token':       # + Token choice (K=4 token)
        sc  = [sc_biomor_tok(d) for d in SC]
        pn  = [mo_biomor_tok(c) for c in PN]
        pan = [mo_gen('token', c) for c in PANCAN]
    else:                              # 'std' general rows
        sc  = [sc_gen(variant, d) for d in SC]
        pn  = [mo_gen(variant, c) for c in PN]
        pan = [mo_gen(variant, c) for c in PANCAN]
    return sc, pn, pan

def _fmt(t):
    return RUN if t is None else f"{t[0]:.1f}±{t[1]:.1f}"

def row_cells(kind, variant, K):
    sc, pn, pan = core_and_extra(kind, variant, K)
    core = sc + pn
    allv = core + pan
    cells = [_fmt(t) for t in allv]
    present = [t for t in core if t is not None]
    if len(present) == 11:
        avg_mean = float(np.mean([t[0] for t in present]))
        avg_sd   = float(np.mean([t[1] for t in present]))  # mean within-dataset fold SD
        avg = f"{avg_mean:.1f}±{avg_sd:.1f}"
    elif present:
        avg = f"{np.mean([t[0] for t in present]):.1f}*"
    else:
        avg = RUN
    return cells + [avg], allv

def flops_rel(mode, K):
    phi = lambda a: 4*a*a*96 + 4*a*96*192
    M = 128; base = 4*phi(M)
    if mode in ('fixed','independent'): act=[M]*K
    elif mode=='expert': act=[round(max(0.5,1-0.25*t)*M) for t in range(K)]
    elif mode=='token':  act=[round(max(0.25,1-0.25*t)*M) for t in range(K)]
    else: return None
    return sum(phi(a) for a in act)/base

def P(indep, K): return f"{K*75}K" if indep else "75K"

# ---- row specs: (label, variant, mode, K, type, param, kind) ----
specs=[
 ("Vanilla",          'independent','independent',4,DASH,   "300K",'std'),
 "RULE",
 ("Recursive$^\\dagger$",'fixed_k2','fixed',2,DASH,"75K",'std'),
 ("",                 'fixed_k3','fixed',3,DASH,"75K",'std'),
 ("",                 'fixed',   'fixed',4,DASH,"75K",'std'),
 "RULE",
 ("MoR (general)",    'expert_k2','expert',2,"Expert","75K",'std'),
 ("",                 'expert_k3','expert',3,"Expert","75K",'std'),
 ("",                 'shared',  'expert',4,"Expert","75K",'std'),
 ("",                 'token',   'token', 4,"Token", "75K",'std'),
 "RULE",
 ("bioMoR",           None,'expert',2,"Expert","75K",'biomor_ladder'),
 ("",                 None,'expert',3,"Expert","75K",'biomor_ladder'),
 ("  + Token choice", None,'token', 4,"Token", "75K",'biomor_token'),
 ("bioMoR (ours)",    None,'expert',4,"Expert","75K",'biomor_head'),
]

cols=["Type","$N_R$","Param","FLOPs","$N_{tok}$",
      "Bar","Lun","Mur","Oes","Seg","Spl","Tce","Xin","Pro","BL","ST","PM","PC","Avg"]
rows=[]; rowvals=[]
for s in specs:
    if s=="RULE": rows.append(("RULE",)); rowvals.append(None); continue
    label,variant,mode,K,typ,param,kind=s
    fl=flops_rel(mode,K); flx=f"{fl:.2f}×" if fl is not None else DASH
    cells,allv=row_cells(kind,variant,K)
    vals=[typ,str(K),param,flx,NTOK]+cells
    head=kind.startswith('biomor')
    rows.append((label,vals,"head" if kind=='biomor_head' else ("bio" if head else "n")))
    rowvals.append(allv)

# ---- per-column best (light-red) on the MEAN: F1 cols 5..18 ----
MAXCOLS=set(range(5,19))
colbest={}
for i in MAXCOLS:
    nums=[]
    for label,allv in zip([r[0] for r in rows],rowvals):
        if label=="RULE" or allv is None: continue
        t=allv[i-5] if i-5 < len(allv) else None
        if t is not None: nums.append(t[0])
    if nums: colbest[i]=max(nums)

# ---- render ----
groups=[("MoR",0,1),("Recursion",1,2),("Model",2,5),
        ("Single-cell macro-F1$\\uparrow$",5,13),("Multi-omics$\\uparrow$",13,18),("",18,19)]
lead,right,X0,XMOD=0.135,0.998,0.004,0.068
n=len(cols); xs=[lead+(right-lead)*(i+0.5)/n for i in range(n)]
lx=lambda i:xs[i]; edge=lambda i:lead+(right-lead)*i/n
nb=sum(1 for r in rows if r[0]!="RULE"); nr=sum(1 for r in rows if r[0]=="RULE")
fig,ax=plt.subplots(figsize=(21.0,1.0+0.46*nb+0.12*nr),dpi=200)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
top=0.95; head_h=0.055; row_h=0.058; fs=8.6; hfs=10.0
yg=top; yl=top-head_h
def L(a,b,y,lw=1.0): ax.plot([a,b],[y,y],color="black",lw=lw,solid_capstyle="butt")
L(X0,right,top+0.030,1.6)
for name,a,b in groups:
    if not name: continue
    ax.text((edge(a)+edge(b))/2,yg,name,ha="center",va="center",fontsize=hfs,fontweight="bold")
    if b-a>1: L(edge(a)+0.004,edge(b)-0.004,yg-0.026,0.9)
ax.text(XMOD,yl,"Models",ha="center",va="center",fontsize=hfs,fontweight="bold")
for i,lab in enumerate(cols): ax.text(lx(i),yl,lab,ha="center",va="center",fontsize=hfs-0.8,fontweight="bold")
L(X0,right,yl-0.028,1.2)
y=yl-0.028-0.006
for r in rows:
    if r[0]=="RULE": y-=0.010; L(X0,right,y,0.7); y-=0.006; continue
    label,vals,style=r; yy=y-row_h/2
    if label:
        lb="bold" if (label.strip().startswith("bioMoR") or label.startswith("MoR") or label.strip().startswith("+")) else "normal"
        ax.text(X0+0.004,yy,label,ha="left",va="center",fontsize=fs+0.3,fontweight=lb)
    for i,v in enumerate(vals):
        c="#b06000" if v==RUN else "black"
        isbest=False
        if i in colbest and v not in (RUN,DASH):
            try: isbest=abs(float(str(v).split('±')[0].rstrip('*×'))-colbest[i])<1e-9
            except: pass
        if isbest:
            ax.add_patch(Rectangle((edge(i)+0.001,yy-row_h/2+0.004),edge(i+1)-edge(i)-0.002,row_h-0.006,
                                    facecolor="#f8cccc",edgecolor="none",zorder=0))
        ax.text(lx(i),yy,v,ha="center",va="center",fontsize=fs,color=c,
                style="italic" if v==RUN else "normal")
    y-=row_h
L(X0,right,y+0.004,1.6)
cy=y-0.02
ax.text(X0,cy,
 "bioMoR efficiency ladder — 5-FOLD CROSS-VALIDATION (mean ± SD over 5 folds in every cell). "
 "One split protocol for ALL rows: StratifiedKFold(5, seed=42) -> 20% test / 80% train per fold; "
 "10% of train held out as validation; fresh training each fold at max-epoch 100, early-stop patience 15. "
 "Folds are identical across every variant (paired comparison). FLOPs analytic (recursion stack, rel. to K=4 vanilla); "
 "N_tok = marker-token budget M=128; Avg = mean macro-F1 over the 11 core datasets (8 SC + Pro/BL/ST), ± = mean within-dataset fold SD.",
 ha="left",va="top",fontsize=7.2,style="italic")
ax.text(X0,cy-0.050,
 "Multi-omics: PM = pan_meta_pri (mutation+CNV) 32-class subtype; PC = panmeta_response (expression) binary primary-vs-metastatic; "
 "extra columns (not folded into Avg). 'run…' = CV tasks still in flight; table auto-refreshes as results_cv5/ lands. '–' = not applicable.",
 ha="left",va="top",fontsize=7.2,style="italic")
plt.subplots_adjust(left=0.004,right=0.998,top=0.995,bottom=0.01)
out=os.path.join(ROOT,"biomor_ladder_table_cv5.png")
fig.savefig(out,dpi=200,bbox_inches="tight",facecolor="white"); print("wrote",out)
