"""Emit paper/cv5_posf1_table.tex: threshold-tuned positive-class F1 (PATH-comparable)
for the multi-omics binary cohorts, full Table-2 ladder. Reads results/repro/ladder/.
Cells: pos-F1 (macro-F1 in parens). 'run...' where a cell is still computing."""
import glob, json, os
ROOT="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
ARMS=[("vanilla","Vanilla"),("fixed_k2","Recursive $K{=}2$"),("fixed_k3","Recursive $K{=}3$"),
      ("fixed_k4","Recursive $K{=}4$"),("expert_k2","MoR-Expert $K{=}2$"),("expert_k3","MoR-Expert $K{=}3$"),
      ("expert_k4","MoR-Expert $K{=}4$"),("token_k2","Token-choice $K{=}2$"),
      ("token_k3","Token-choice $K{=}3$"),("token_k4","Token-choice $K{=}4$")]
COH=[("prostate","Prostate"),("blca","BLCA"),("stad","STAD"),("pan_meta_pri","PM"),("panmeta_response","PC"),("pan_meta_pri_3modal","3M")]
RUN=r"\emph{\scriptsize run\ldots}"
def cell(t,a):
    f=f"{ROOT}/results/repro/ladder/{t}_{a}/{t}__{a}.json"
    if not os.path.exists(f): return RUN
    try:
        d=json.load(open(f))
        p,ps=d["pos_f1"][0]*100,d["pos_f1"][1]*100      # positive-class F1: mean, SD over 10 folds
        return f"{p:.1f}{{\\scriptsize$\\pm${ps:.1f}}}"
    except Exception: return RUN
L=[r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "c"*len(COH) + "}", r"\toprule"]
L.append("Model & "+" & ".join(c for _,c in COH)+r" \\")
L.append(r"\midrule")
for a,lab in ARMS:
    L.append(lab+" & "+" & ".join(cell(t,a) for t,_ in COH)+r" \\")
L.append(r"\bottomrule"); L.append(r"\end{tabular}}")
open(f"{ROOT}/paper/cv5_posf1_table.tex","w").write("\n".join(L)+"\n")
print("wrote paper/cv5_posf1_table.tex ; done cells:",
      sum(1 for t,_ in COH for a,_ in ARMS if os.path.exists(f"{ROOT}/results/repro/ladder/{t}_{a}/{t}__{a}.json")),"/50")
