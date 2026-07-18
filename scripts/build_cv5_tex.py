"""Emit NATIVE LaTeX table fragments (booktabs) for the three CV5 result sets, read
straight from results/cv5/. \\input-ed by paper/cv5_placeholders.tex and regenerated
every refresh cycle so the compiled PDF updates in real time. Missing cells render as
'run...'. Writes paper/cv5_main_table.tex, cv5_scaling_table.tex, cv5_ablation_table.tex."""
import glob, json, os
import numpy as np

ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
PAP = os.path.join(ROOT, "paper")
SC = ['segerstolpe','lung','oesophagus','baron','muraro','tcell','spleen','xin']
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
def sc_gen(v,ds):  return _read(f"results/cv5/sc/{v}/{ds}.json")
# CANONICAL bioMoR = bio_both (interaction at BOTH sites). K4-expert headline lives in
# biomor_canonical/; the rest of the ladder in biomor_ladder/<cfg>/. Multi-omics bioMoR
# uses the PROVIDED Reactome pathway graph (inject_mo/both = K4, biomor_ladder_mo/<cfg>).
def sc_bio(ds,K):
    if K == 4: return _read(f"results/cv5/biomor_canonical/{SCdir[ds]}/bio_both_cv.json")
    return _read(f"results/cv5/biomor_ladder/expert_k{K}/{SCdir[ds]}/bio_both_cv.json")
def sc_biotok(ds): return _read(f"results/cv5/biomor_ladder/token_k4/{SCdir[ds]}/bio_both_cv.json")
def mo_gen(v,t):   return _read(f"results/cv5/mo/{v}/{t}__*_cv.json")
def mo_bio(c,K):
    if K == 4: return _read(f"results/cv5/inject_mo/both/{c}__*_cv.json")
    return _read(f"results/cv5/biomor_ladder_mo/expert_k{K}/{c}__*_cv.json")
def mo_biotok(c):  return _read(f"results/cv5/biomor_ladder_mo/token_k4/{c}__*_cv.json")
def sc_biotok_k(ds,K): return _read(f"results/cv5/biomor_ladder/token_k{K}/{SCdir[ds]}/bio_both_cv.json")
def mo_biotok_k(c,K):  return _read(f"results/cv5/biomor_ladder_mo/token_k{K}/{c}__*_cv.json")
LPW = {2:'expert_k2',3:'expert_k3',4:'shared'}

def mo_3m(kind, variant, mode, K):
    """Tri-modal (mut+CNV+expr) pan-cancer 3M macro-F1, read from the SAME unified 5-fold/
    seed-42 results/cv5 trees as every other data column (was previously the 10-fold
    reproduce_path.py protocol in results/repro/ -> not apple-to-apple). 3M is now treated
    as an ordinary multi-omics cohort: baseline rows -> results/cv5/mo/<variant> (like
    mo_gen); bioMoR rows -> inject_mo/both (K4) + biomor_ladder_mo/<mode>_k<K> (like mo_bio)."""
    C = 'pan_meta_pri_3modal'
    if kind and kind.startswith('biomor'):
        # mirror mo_bio/mo_biotok: only the EXPERT-K4 headline reuses inject_mo/both
        # (which is the expert depth-4 run). The token-K4 row must read its own token_k4
        # arm, NOT inject_mo/both -- else the two K=4 rows print an identical 3M cell.
        if kind == 'biomor_head': return _read(f"results/cv5/inject_mo/both/{C}__*_cv.json")
        return _read(f"results/cv5/biomor_ladder_mo/{mode}_k{K}/{C}__*_cv.json")
    return _read(f"results/cv5/mo/{variant}/{C}__*_cv.json")

def row_vals(kind, variant, K, mode):
    if kind=='biomor_head':
        sc=[sc_bio(d,4) for d in SC]; pn=[mo_bio(c,4) for c in PN]; pan=[mo_bio(c,4) for c in PANCAN]
    elif kind=='biomor_ladder':
        sc=[sc_bio(d,K) for d in SC]; pn=[mo_bio(c,K) for c in PN]; pan=[mo_bio(c,K) for c in PANCAN]
    elif kind=='biomor_token':
        sc=[sc_biotok(d) for d in SC]; pn=[mo_biotok(c) for c in PN]; pan=[mo_biotok(c) for c in PANCAN]
    elif kind=='biomor_tokenk':
        sc=[sc_biotok_k(d,K) for d in SC]; pn=[mo_biotok_k(c,K) for c in PN]; pan=[mo_biotok_k(c,K) for c in PANCAN]
    else:
        sc=[sc_gen(variant,d) for d in SC]; pn=[mo_gen(variant,c) for c in PN]; pan=[mo_gen(variant,c) for c in PANCAN]
    pan = pan + [mo_3m(kind, variant, mode, K)]      # append tri-modal 3M column (unified 5-fold)
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
 ("",'token_k2','token',2,"Token","75K",'std'),
 ("",'token_k3','token',3,"Token","75K",'std'),
 ("",'token','token',4,"Token","75K",'std'),None,
 ("bioMoR",None,'expert',2,"Expert","75K",'biomor_ladder'),
 ("",None,'expert',3,"Expert","75K",'biomor_ladder'),
 ("",None,'expert',4,"Expert","75K",'biomor_head'),
 ("\\quad + Token ($N_R{=}2$)",None,'token',2,"Token","75K",'biomor_tokenk'),
 ("\\quad + Token ($N_R{=}3$)",None,'token',3,"Token","75K",'biomor_tokenk'),
 ("\\quad + Token choice",None,'token',4,"Token","75K",'biomor_token'),
]

def emit_main():
    DS = ['Seg','Lun','Oes','Bar','Mur','Tce','Spl','Xin','Pro','BL','ST','PM','PC','3M']
    NC = len(DS)   # 14 data columns (8 single-cell + 6 multi-omics incl. tri-modal 3M)
    # best (max mean) per data column for bolding
    grid=[]
    for s in SPECS:
        if s is None: grid.append(None); continue
        label,variant,mode,K,typ,param,kind=s
        sc,pn,pan=row_vals(kind,variant,K,mode); grid.append((s, sc+pn+pan))
    best=[None]*NC
    for i in range(NC):
        vals=[g[1][i][0] for g in grid if g and g[1][i] is not None]
        best[i]=max(vals) if vals else None
    L=[]
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{l l c c c " + "c "*8 + "c "*6 + "c}")
    L.append(r"\toprule")
    L.append(r" & & & & & \multicolumn{8}{c}{Single-cell macro-F1 $\uparrow$} & \multicolumn{6}{c}{Multi-omics macro-F1 $\uparrow$} & \\")
    L.append(r"\cmidrule(lr){6-13}\cmidrule(lr){14-19}")
    L.append(r"Model & Type & $N_R$ & Param & FLOPs & " + " & ".join(DS) + r" & Avg \\")
    L.append(r"\midrule")
    for g in grid:
        if g is None: L.append(r"\midrule"); continue
        (label,variant,mode,K,typ,param,kind), allv = g
        fl=flops_rel(mode,K); flx=f"{fl:.2f}$\\times$" if fl is not None else "--"
        core=[v for v in allv[:11] if v is not None]   # 8 SC + Pro/BL/ST
        avg = f"{np.mean([v[0] for v in core]):.1f}{{\\scriptsize$\\pm${np.mean([v[1] for v in core]):.1f}}}" if len(core)==11 else (f"{np.mean([v[0] for v in core]):.1f}*" if core else RUN)
        cells=[cell(allv[i], bold=(allv[i] is not None and best[i] is not None and abs(allv[i][0]-best[i])<1e-9)) for i in range(NC)]
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
    if model=='biomor': return _agg([f"{ROOT}/results/cv5/scaling_sc_biomor/d{w}/*/learned_cv.json"])
    return _agg([f"{ROOT}/results/cv5/scaling_sc/{model}_d{w}/*.json"])
def mo_scale(model,w):
    if model=='biomor': return _agg([f"{ROOT}/results/cv5/scaling_mo_biomor/d{w}/pnet/*/learned_cv.json"])
    return _agg([f"{ROOT}/results/cv5/scaling_mo/{model}_d{w}/*_cv.json"])

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
def cond_sc(name): return _agg([f"{ROOT}/results/cv5/ablation/{name}/*.json",
                                f"{ROOT}/results/cv5/ablation/{name}/*/*_cv.json"])
def cond_mo(name): return _agg([f"{ROOT}/results/cv5/ablation/{name}/pnet/*/*_cv.json",
                                f"{ROOT}/results/cv5/ablation/{name}/*_cv.json"])
BSC_GEN=_agg([f"{ROOT}/results/cv5/sc/shared/*.json"])
BSC_BIO=_agg([f"{ROOT}/results/cv5/biomor_sc/k4/*/learned_cv.json"])
BMO_BIO=_agg([f"{ROOT}/results/cv5/biomor_mo/k4/pnet/*/learned_cv.json"])
BMO_GEN=_agg([f"{ROOT}/results/cv5/mo/shared/prostate__*_cv.json",
              f"{ROOT}/results/cv5/mo/shared/blca__*_cv.json",
              f"{ROOT}/results/cv5/mo/shared/stad__*_cv.json"])
def dlt(t,b):
    if t is None or b is None: return DASH if t is None else ""
    d=t[0]-b[0]; c="1A7F37" if d>=0 else "B00020"
    return f"\\textcolor[HTML]{{{c}}}{{{d:+.1f}}}"

def emit_ablation():
    R=[]
    def add(fam,label,sc,mo,bsc,bmo,hdr=False): R.append((fam,label,sc,mo,bsc,bmo,hdr))
    # Family 1 = the biological interaction graph: SOURCE (none/curated/random) AND
    # REFINEMENT (warm-start/fuse) share the SAME bioMoR-learned baseline, so they are one
    # family with a single baseline row (was split into families 1 and 5 -> duplicate 74.0).
    add("1. Biological interaction graph","bioMoR learned (baseline)",BSC_BIO,BMO_BIO,None,None,True)
    add("","\\quad none (no graph)",cond_sc("f1_sc_none"),cond_mo("f1_mo_none"),BSC_BIO,BMO_BIO)
    add("","\\quad coexpr / curated (fixed)",cond_sc("f1_sc_coexpr"),cond_mo("f1_mo_curated"),BSC_BIO,BMO_BIO)
    add("","\\quad random (deg.-matched)",cond_sc("f1_sc_random"),cond_mo("f1_mo_random"),BSC_BIO,BMO_BIO)
    add("","\\quad warm-start from curated",cond_sc("f5_learned_bio"),None,BSC_BIO,None)
    add("","\\quad fuse coexpr graph",cond_sc("f5_learned_fused"),None,BSC_BIO,None)
    add("","\\quad fuse RANDOM (control)",cond_sc("f5_learned_fused_rand"),None,BSC_BIO,None)
    # Family 2 = recursion & routing internals: routing / markers / depth all ablate from the
    # SAME MoR-general baseline, so one baseline row (was families 2,3,4 -> triplicate 63.2).
    add("2. Recursion \\& routing (MoR-general)","$N_R{=}4$, $M{=}128$ shared (baseline)",BSC_GEN,None,None,None,True)
    add("","\\quad share: sequence",cond_sc("f2_share_seq"),None,BSC_GEN,None)
    add("","\\quad share: middle-cycle",cond_sc("f2_share_midcyc"),None,BSC_GEN,None)
    add("","\\quad share: middle-seq.",cond_sc("f2_share_midseq"),None,BSC_GEN,None)
    add("","\\quad router: MLP",cond_sc("f2_router_mlp"),None,BSC_GEN,None)
    add("","\\quad no load-balance loss",cond_sc("f2_nobalance"),None,BSC_GEN,None)
    add("","\\quad markers: random",cond_sc("f3_marker_random"),None,BSC_GEN,None)
    add("","\\quad markers: variance",cond_sc("f3_marker_var"),None,BSC_GEN,None)
    add("","\\quad budget $M{=}64$",cond_sc("f3_M64"),None,BSC_GEN,None)
    add("","\\quad budget $M{=}256$",cond_sc("f3_M256"),None,BSC_GEN,None)
    add("","\\quad $N_R{=}1$ (no recursion)",cond_sc("f4_K1"),None,BSC_GEN,None)
    add("","\\quad $N_R{=}6$",cond_sc("f4_K6"),None,BSC_GEN,None)
    add("","\\quad $N_R{=}8$",cond_sc("f4_K8"),None,BSC_GEN,None)
    add("3. Input modality (multi-omics)","mut+CNV (baseline)",None,BMO_GEN,None,None,True)
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
BL_SC = [('Lung','Lung','sc')]
BL_MO = [('pan_meta_pri','Pan-cancer (PM)','pancan')]
def _base(ds, meth): return _read(f"results/cv5/baselines/{ds}/{meth}_cv.json")
def _biomor_ds(ds, kind):
    # headline bioMoR = token-choice (main-table Avg headline 68.2; see abstract/intro)
    if kind=='sc':   return _read(f"results/cv5/biomor_sc_token/{ds}/learned_cv.json")
    if kind=='pnet': return _read(f"results/cv5/biomor_mo_token/pnet/{ds}__response/learned_cv.json")
    return _read(f"results/cv5/mo/token/{ds}__*_cv.json")   # pancan token-choice

def emit_baselines():
    L=[r"\resizebox{\columnwidth}{!}{%",
       r"\begin{tabular}{l cccc}",
       r"\toprule",
       r"Dataset & Linear & Random Forest & Nearest Centroid & bioMoR \\",
       r"\midrule",
       r"\multicolumn{5}{l}{\emph{Single-cell (genomap)}} \\"]
    colmeans=[[],[],[],[]]
    def row(ds,lab,kind):
        cells=[_base(ds,'linear'),_base(ds,'random'),_base(ds,'nearestcentroid'),_biomor_ds(ds,kind)]
        for i,t in enumerate(cells):
            if t is not None: colmeans[i].append(t[0])
        vals=[c[0] for c in cells if c is not None]
        best=max(vals) if vals else None
        txt=[cell(t, bold=(t is not None and best is not None and abs(t[0]-best)<1e-9)) for t in cells]
        L.append(f"\\quad {lab} & "+" & ".join(txt)+r" \\")
    for ds,lab,kind in BL_SC: row(ds,lab,kind)
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{l}{\emph{Multi-omics (Reactome/P-NET \& pan-cancer)}} \\")
    for ds,lab,kind in BL_MO: row(ds,lab,kind)
    L.append(r"\midrule")
    mean=[f"\\textbf{{{np.mean(c):.1f}}}" if c else RUN for c in colmeans]
    L.append(r"\textbf{Mean} & "+" & ".join(mean)+r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(PAP,"cv5_baselines_table.tex"),"w").write("\n".join(L)+"\n")

# ---------- focused bio-router ablation: 4 datasets x 4 router variants ----------
def _br_sc(ds,mode):  return _read(f"results/cv5/biorouter_ablation/{ds}/{mode}_cv.json")
def _br_mo(coh,mode): return _read(f"results/cv5/biorouter_ablation/pnet/{coh}__response/{mode}_cv.json")
def emit_biorouter():
    L=[r"\resizebox{\columnwidth}{!}{%",
       r"\begin{tabular}{l cccc}",
       r"\toprule",
       r"Dataset & Data-driven & Static biology & Learned graph & bioMoR \\",
       r" & router only & only (fixed) & (end-to-end) & (data$+$bio) \\",
       r"\midrule"]
    rows=[("Baron",_br_sc,"Baron","coexpr"),("Segerstolpe",_br_sc,"Segerstolpe","coexpr"),
          ("STAD",_br_mo,"stad","curated"),("BLCA",_br_mo,"blca","curated")]
    for lab,fn,key,statmode in rows:
        cells=[fn(key,"none"),fn(key,statmode),fn(key,"learned"),fn(key,"learned_bio")]
        vals=[c[0] for c in cells if c is not None]
        best=max(vals) if vals else None
        txt=[cell(c,bold=(c is not None and best is not None and abs(c[0]-best)<1e-9)) for c in cells]
        L.append(f"{lab} & "+" & ".join(txt)+r" \\")
    L += [r"\bottomrule",r"\end{tabular}}"]
    open(os.path.join(PAP,"biorouter_ablation_table.tex"),"w").write("\n".join(L)+"\n")

if __name__ == "__main__":
    emit_main(); emit_scaling(); emit_ablation(); emit_baselines()
    print("wrote paper/cv5_{main,scaling,ablation,baselines}_table.tex")
