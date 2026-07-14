"""Emit NATIVE LaTeX table fragments (booktabs) for the three CV5 result sets, read
straight from results_cv5/. \\input-ed by paper/cv5_placeholders.tex and regenerated
every refresh cycle so the compiled PDF updates in real time. Missing cells render as
'run...'. Writes paper/cv5_main_table.tex, cv5_scaling_table.tex, cv5_ablation_table.tex."""
import glob, json, os
import numpy as np

ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
PAP = os.path.join(ROOT, "paper")
SC = ['baron','lung','muraro','oesophagus','segerstolpe','spleen','tcell','xin']
SCdir = {'baron':'Baron','lung':'Lung','muraro':'Muraro','oesophagus':'Oesophagus',
         'segerstolpe':'Segerstolpe','spleen':'Spleen','tcell':'Tcell','xin':'Xin'}
PN = ['prostate','blca','stad']
PANCAN = ['pan_meta_pri','panmeta_response']
RUN = r"\emph{\scriptsize run\ldots}"; DASH = "--"

def _read(pg):
    for f in sorted(glob.glob(os.path.join(ROOT, pg))):
        try:
            d = json.load(open(f)); m = d.get("cv_macro_f1")
            if m and m.get("mean") is not None: return (float(m["mean"]), float(m["std"]))
        except Exception: pass
    return None

def cell(t, bold=False):
    if t is None: return RUN
    s = f"{t[0]:.1f}{{\\scriptsize$\\pm${t[1]:.1f}}}"
    return f"\\textbf{{{s}}}" if bold else s

# ---------- main ladder ----------
def sc_gen(v,ds):  return _read(f"results_cv5/sc/{v}/{ds}.json")
def sc_bio(ds,K):  return _read(f"results_cv5/biomor_sc/k{K}/{SCdir[ds]}/learned_cv.json")
def sc_biotok(ds): return _read(f"results_cv5/biomor_sc_token/{SCdir[ds]}/learned_cv.json")
def mo_gen(v,t):   return _read(f"results_cv5/mo/{v}/{t}__*_cv.json")
def mo_bio(c,K):   return _read(f"results_cv5/biomor_mo/k{K}/pnet/{c}__response/learned_cv.json")
def mo_biotok(c):  return _read(f"results_cv5/biomor_mo_token/pnet/{c}__response/learned_cv.json")
LPW = {2:'expert_k2',3:'expert_k3',4:'shared'}

def row_vals(kind, variant, K):
    if kind=='biomor_head':
        sc=[sc_bio(d,4) for d in SC]; pn=[mo_bio(c,4) for c in PN]; pan=[mo_gen('shared',c) for c in PANCAN]
    elif kind=='biomor_ladder':
        sc=[sc_bio(d,K) for d in SC]; pn=[mo_bio(c,K) for c in PN]; pan=[mo_gen(LPW[K],c) for c in PANCAN]
    elif kind=='biomor_token':
        sc=[sc_biotok(d) for d in SC]; pn=[mo_biotok(c) for c in PN]; pan=[mo_gen('token',c) for c in PANCAN]
    else:
        sc=[sc_gen(variant,d) for d in SC]; pn=[mo_gen(variant,c) for c in PN]; pan=[mo_gen(variant,c) for c in PANCAN]
    return sc, pn, pan

def flops_rel(mode,K):
    phi=lambda a:4*a*a*96+4*a*96*192; M=128; base=4*phi(M)
    if mode in ('fixed','independent'): act=[M]*K
    elif mode=='expert': act=[round(max(0.75,1-0.25*t)*M) for t in range(K)]
    elif mode=='token':  act=[round(max(0.25,1-0.25*t)*M) for t in range(K)]
    else: return None
    return sum(phi(a) for a in act)/base

SPECS=[  # (label,variant,mode,K,type,param,kind)
 ("Vanilla",'independent','independent',4,"--","300K",'std'),None,
 ("Recursive",'fixed_k2','fixed',2,"--","75K",'std'),
 ("",'fixed_k3','fixed',3,"--","75K",'std'),
 ("",'fixed','fixed',4,"--","75K",'std'),None,
 ("MoR (general)",'expert_k2','expert',2,"Expert","75K",'std'),
 ("",'expert_k3','expert',3,"Expert","75K",'std'),
 ("",'shared','expert',4,"Expert","75K",'std'),
 ("",'token','token',4,"Token","75K",'std'),None,
 ("bioMoR",None,'expert',2,"Expert","75K",'biomor_ladder'),
 ("",None,'expert',3,"Expert","75K",'biomor_ladder'),
 ("",None,'expert',4,"Expert","75K",'biomor_head'),
 ("\\quad + Token choice",None,'token',4,"Token","75K",'biomor_token'),
]

def emit_main():
    DS = ['Bar','Lun','Mur','Oes','Seg','Spl','Tce','Xin','Pro','BL','ST','PM','PC']
    # best (max mean) per data column for bolding
    grid=[]
    for s in SPECS:
        if s is None: grid.append(None); continue
        label,variant,mode,K,typ,param,kind=s
        sc,pn,pan=row_vals(kind,variant,K); grid.append((s, sc+pn+pan))
    best=[None]*13
    for i in range(13):
        vals=[g[1][i][0] for g in grid if g and g[1][i] is not None]
        best[i]=max(vals) if vals else None
    L=[]
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{l l c c c " + "c "*8 + "c "*5 + "c}")
    L.append(r"\toprule")
    L.append(r" & & & & & \multicolumn{8}{c}{Single-cell macro-F1 $\uparrow$} & \multicolumn{5}{c}{Multi-omics macro-F1 $\uparrow$} & \\")
    L.append(r"\cmidrule(lr){6-13}\cmidrule(lr){14-18}")
    L.append(r"Model & Type & $N_R$ & Param & FLOPs & " + " & ".join(DS) + r" & Avg \\")
    L.append(r"\midrule")
    for g in grid:
        if g is None: L.append(r"\midrule"); continue
        (label,variant,mode,K,typ,param,kind), allv = g
        fl=flops_rel(mode,K); flx=f"{fl:.2f}$\\times$" if fl is not None else "--"
        core=[v for v in allv[:11] if v is not None]
        avg = f"{np.mean([v[0] for v in core]):.1f}{{\\scriptsize$\\pm${np.mean([v[1] for v in core]):.1f}}}" if len(core)==11 else (f"{np.mean([v[0] for v in core]):.1f}*" if core else RUN)
        cells=[cell(allv[i], bold=(allv[i] is not None and best[i] is not None and abs(allv[i][0]-best[i])<1e-9)) for i in range(13)]
        head = kind.startswith('biomor')
        lab = f"\\textbf{{{label}}}" if (label and (label.startswith('bioMoR') or label.startswith('MoR') or '+' in label)) else label
        L.append(" & ".join([lab, typ, str(K), param, flx]+cells+[avg]) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    open(os.path.join(PAP,"cv5_main_table.tex"),"w").write("\n".join(L)+"\n")

# ---------- model-size scaling (native table) ----------
WIDTHS=[96,136,192,272,352]
def _agg(globs):
    ms,ss=[],[]
    for g in globs:
        for f in glob.glob(g):
            try:
                d=json.load(open(f)); m=d.get("cv_macro_f1")
                if m and m.get("mean") is not None: ms.append(m["mean"]); ss.append(m["std"])
            except Exception: pass
    return (float(np.mean(ms)),float(np.mean(ss))) if ms else None
def sc_scale(model,w):
    if model=='biomor': return _agg([f"{ROOT}/results_cv5/scaling_sc_biomor/d{w}/*/learned_cv.json"])
    return _agg([f"{ROOT}/results_cv5/scaling_sc/{model}_d{w}/*.json"])
def mo_scale(model,w):
    if model=='biomor': return _agg([f"{ROOT}/results_cv5/scaling_mo_biomor/d{w}/pnet/*/learned_cv.json"])
    return _agg([f"{ROOT}/results_cv5/scaling_mo/{model}_d{w}/*_cv.json"])

def emit_scaling():
    L=[r"\resizebox{\columnwidth}{!}{%",
       r"\begin{tabular}{c ccc ccc}",
       r"\toprule",
       r" & \multicolumn{3}{c}{Single-cell} & \multicolumn{3}{c}{Multi-omics (P-NET)} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
       r"$d_{\text{model}}$ & Vanilla & MoR-gen. & bioMoR & Vanilla & MoR-gen. & bioMoR \\",
       r"\midrule"]
    for w in WIDTHS:
        row=[str(w)]+[cell(sc_scale(m,w)) for m in ('vanilla','morgen','biomor')]+[cell(mo_scale(m,w)) for m in ('vanilla','morgen','biomor')]
        L.append(" & ".join(row)+r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(PAP,"cv5_scaling_table.tex"),"w").write("\n".join(L)+"\n")

# ---------- six-family ablation (native table) ----------
def cond_sc(name): return _agg([f"{ROOT}/results_cv5/ablation/{name}/*.json",
                                f"{ROOT}/results_cv5/ablation/{name}/*/*_cv.json"])
def cond_mo(name): return _agg([f"{ROOT}/results_cv5/ablation/{name}/pnet/*/*_cv.json",
                                f"{ROOT}/results_cv5/ablation/{name}/*_cv.json"])
BSC_GEN=_agg([f"{ROOT}/results_cv5/sc/shared/*.json"])
BSC_BIO=_agg([f"{ROOT}/results_cv5/biomor_sc/k4/*/learned_cv.json"])
BMO_BIO=_agg([f"{ROOT}/results_cv5/biomor_mo/k4/pnet/*/learned_cv.json"])
BMO_GEN=_agg([f"{ROOT}/results_cv5/mo/shared/prostate__*_cv.json",
              f"{ROOT}/results_cv5/mo/shared/blca__*_cv.json",
              f"{ROOT}/results_cv5/mo/shared/stad__*_cv.json"])
def dlt(t,b):
    if t is None or b is None: return DASH if t is None else ""
    d=t[0]-b[0]; c="1A7F37" if d>=0 else "B00020"
    return f"\\textcolor[HTML]{{{c}}}{{{d:+.1f}}}"

def emit_ablation():
    R=[]
    def add(fam,label,sc,mo,bsc,bmo,hdr=False): R.append((fam,label,sc,mo,bsc,bmo,hdr))
    add("1. Biological graph source","bioMoR learned (baseline)",BSC_BIO,BMO_BIO,None,None,True)
    add("","\\quad none (no graph)",cond_sc("f1_sc_none"),cond_mo("f1_mo_none"),BSC_BIO,BMO_BIO)
    add("","\\quad coexpr / curated",cond_sc("f1_sc_coexpr"),cond_mo("f1_mo_curated"),BSC_BIO,BMO_BIO)
    add("","\\quad random (deg.-matched)",cond_sc("f1_sc_random"),cond_mo("f1_mo_random"),BSC_BIO,BMO_BIO)
    add("2. Routing internals","MoR-general shared (baseline)",BSC_GEN,None,None,None,True)
    add("","\\quad share: sequence",cond_sc("f2_share_seq"),None,BSC_GEN,None)
    add("","\\quad share: middle-cycle",cond_sc("f2_share_midcyc"),None,BSC_GEN,None)
    add("","\\quad share: middle-seq.",cond_sc("f2_share_midseq"),None,BSC_GEN,None)
    add("","\\quad router: MLP",cond_sc("f2_router_mlp"),None,BSC_GEN,None)
    add("","\\quad no load-balance loss",cond_sc("f2_nobalance"),None,BSC_GEN,None)
    add("3. Marker selection / budget","router markers, $M{=}128$ (baseline)",BSC_GEN,None,None,None,True)
    add("","\\quad markers: random",cond_sc("f3_marker_random"),None,BSC_GEN,None)
    add("","\\quad markers: variance",cond_sc("f3_marker_var"),None,BSC_GEN,None)
    add("","\\quad budget $M{=}64$",cond_sc("f3_M64"),None,BSC_GEN,None)
    add("","\\quad budget $M{=}256$",cond_sc("f3_M256"),None,BSC_GEN,None)
    add("4. Recursion depth","$N_R{=}4$ (baseline)",BSC_GEN,None,None,None,True)
    add("","\\quad $N_R{=}1$ (no recursion)",cond_sc("f4_K1"),None,BSC_GEN,None)
    add("","\\quad $N_R{=}6$",cond_sc("f4_K6"),None,BSC_GEN,None)
    add("","\\quad $N_R{=}8$",cond_sc("f4_K8"),None,BSC_GEN,None)
    add("5. Learned-graph refinement","learned (baseline)",BSC_BIO,None,None,None,True)
    add("","\\quad warm-start from curated",cond_sc("f5_learned_bio"),None,BSC_BIO,None)
    add("","\\quad fuse coexpr graph",cond_sc("f5_learned_fused"),None,BSC_BIO,None)
    add("","\\quad fuse RANDOM (control)",cond_sc("f5_learned_fused_rand"),None,BSC_BIO,None)
    add("6. Input modality (multi-omics)","mut+CNV (baseline)",None,BMO_GEN,None,None,True)
    add("","\\quad CNV only",None,cond_mo("f6_mo_cnv"),None,BMO_GEN)
    add("","\\quad mutation only",None,cond_mo("f6_mo_mut"),None,BMO_GEN)
    L=[r"\resizebox{\columnwidth}{!}{%",
       r"\begin{tabular}{l cc cc}",
       r"\toprule",
       r"Condition & SC macro-F1 & $\Delta$SC & MO macro-F1 & $\Delta$MO \\",
       r"\midrule"]
    for fam,label,sc,mo,bsc,bmo,hdr in R:
        if fam: L.append(r"\addlinespace[1pt]\multicolumn{5}{l}{\textbf{"+fam+r"}}\\")
        sctxt=cell(sc,bold=hdr) if sc is not None else DASH
        motxt=cell(mo,bold=hdr) if mo is not None else DASH
        dsc = "" if hdr else (dlt(sc,bsc) if sc is not None else DASH)
        dmo = "" if hdr else (dlt(mo,bmo) if mo is not None else DASH)
        lab = f"\\emph{{{label}}}" if hdr else label
        L.append(" & ".join([lab,sctxt,dsc,motxt,dmo])+r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(PAP,"cv5_ablation_table.tex"),"w").write("\n".join(L)+"\n")

# ---------- Table 5: bioMoR vs non-transformer baselines (native) ----------
BL_SC = [('Baron','Baron'),('Lung','Lung'),('Muraro','Muraro'),('Oesophagus','Oes.'),
         ('Segerstolpe','Seger.'),('Spleen','Spleen'),('Tcell','T-cell'),('Xin','Xin')]
BL_MO = [('prostate','Prostate'),('blca','BLCA'),('stad','STAD')]
def _base(ds, meth): return _read(f"results_cv5/baselines/{ds}/{meth}_cv.json")
def _biomor_ds(ds, is_sc):
    return (_read(f"results_cv5/biomor_sc/k4/{ds}/learned_cv.json") if is_sc
            else _read(f"results_cv5/biomor_mo/k4/pnet/{ds}__response/learned_cv.json"))

def emit_baselines():
    L=[r"\resizebox{\columnwidth}{!}{%",
       r"\begin{tabular}{l cccc}",
       r"\toprule",
       r"Dataset & Linear & Random Forest & Nearest Centroid & bioMoR \\",
       r"\midrule",
       r"\multicolumn{5}{l}{\emph{Single-cell (genomap)}} \\"]
    colmeans=[[],[],[],[]]
    def row(ds,lab,is_sc):
        cells=[_base(ds,'linear'),_base(ds,'random'),_base(ds,'nearestcentroid'),_biomor_ds(ds,is_sc)]
        for i,t in enumerate(cells):
            if t is not None: colmeans[i].append(t[0])
        vals=[c[0] for c in cells if c is not None]
        best=max(vals) if vals else None
        txt=[cell(t, bold=(t is not None and best is not None and abs(t[0]-best)<1e-9)) for t in cells]
        L.append(f"\\quad {lab} & "+" & ".join(txt)+r" \\")
    for ds,lab in BL_SC: row(ds,lab,True)
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{l}{\emph{Multi-omics (Reactome/P-NET)}} \\")
    for ds,lab in BL_MO: row(ds,lab,False)
    L.append(r"\midrule")
    mean=[f"\\textbf{{{np.mean(c):.1f}}}" if c else RUN for c in colmeans]
    L.append(r"\textbf{Mean} & "+" & ".join(mean)+r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(PAP,"cv5_baselines_table.tex"),"w").write("\n".join(L)+"\n")

if __name__ == "__main__":
    emit_main(); emit_scaling(); emit_ablation(); emit_baselines()
    print("wrote paper/cv5_{main,scaling,ablation,baselines}_table.tex")
