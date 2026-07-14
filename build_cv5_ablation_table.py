"""Six-family ablation table — 5-FOLD CV (mean ± SD). Reads results_cv5/ablation/*
plus the main results_cv5/ baselines. Each cell = macro-F1 averaged over the domain's
datasets (SC=8, MO=3 P-NET); ± = mean within-dataset fold SD. Δ = vs the family baseline.
Renders biomor_ablation_table_cv5.png."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
RUN, DASH = "run…", "–"

def _cv(f):
    try:
        d = json.load(open(f)); m = d.get("cv_macro_f1")
        return (m["mean"], m["std"]) if m and m.get("mean") is not None else None
    except Exception: return None

def agg(globs):
    """mean-over-datasets of per-dataset CV means, and mean of per-dataset fold SDs."""
    ms, ss = [], []
    seen = set()
    for g in globs:
        for f in glob.glob(g):
            if f in seen: continue
            seen.add(f)
            t = _cv(f)
            if t: ms.append(t[0]); ss.append(t[1])
    if not ms: return None
    return (float(np.mean(ms)), float(np.mean(ss)), len(ms))

def cond_sc(name):  # singlecell OR bio_learned condition dir
    return agg([f"{ROOT}/results_cv5/ablation/{name}/*.json",
                f"{ROOT}/results_cv5/ablation/{name}/*/*_cv.json"])
def cond_mo(name):  # bio_redesign OR pathway condition dir
    return agg([f"{ROOT}/results_cv5/ablation/{name}/pnet/*/*_cv.json",
                f"{ROOT}/results_cv5/ablation/{name}/*_cv.json"])

# ---- baselines from the MAIN CV5 sweep ----
BASE_SC_GEN  = agg([f"{ROOT}/results_cv5/sc/shared/*.json"])                       # MoR-general shared K4
BASE_SC_BIO  = agg([f"{ROOT}/results_cv5/biomor_sc/k4/*/learned_cv.json"])         # bioMoR learned K4
BASE_MO_BIO  = agg([f"{ROOT}/results_cv5/biomor_mo/k4/pnet/*/learned_cv.json"])    # bioMoR learned K4 (P-NET)
BASE_MO_GEN  = agg([f"{ROOT}/results_cv5/mo/shared/prostate__*_cv.json",
                    f"{ROOT}/results_cv5/mo/shared/blca__*_cv.json",
                    f"{ROOT}/results_cv5/mo/shared/stad__*_cv.json"])              # pathway shared mut_cnv

def fmt(t): return RUN if t is None else f"{t[0]:.1f}±{t[1]:.1f}"
def delta(t, base):
    if t is None or base is None: return DASH if t is None else ""
    return f"{t[0]-base[0]:+.1f}"

# rows: (family_label, condition_label, sc_tuple, mo_tuple, baseline_for_delta)
# 'domain' picks which of sc/mo is populated; the other is DASH.
ROWS = []
def add(fam, label, sc=None, mo=None, base_sc=None, base_mo=None, header=False):
    ROWS.append(dict(fam=fam, label=label, sc=sc, mo=mo, bsc=base_sc, bmo=base_mo, header=header))

# Family 1 — graph source
add("1. Biological graph source", "bioMoR learned (baseline)", BASE_SC_BIO, BASE_MO_BIO, header=True)
add("", "none (no graph)",        cond_sc("f1_sc_none"),   cond_mo("f1_mo_none"),   BASE_SC_BIO, BASE_MO_BIO)
add("", "coexpr / curated",       cond_sc("f1_sc_coexpr"), cond_mo("f1_mo_curated"),BASE_SC_BIO, BASE_MO_BIO)
add("", "random (degree-matched)",cond_sc("f1_sc_random"), cond_mo("f1_mo_random"), BASE_SC_BIO, BASE_MO_BIO)
# Family 2 — routing internals
add("2. Routing internals", "MoR-general shared (baseline)", BASE_SC_GEN, None, header=True)
add("", "share: sequence",        cond_sc("f2_share_seq"),   None, BASE_SC_GEN)
add("", "share: middle-cycle",    cond_sc("f2_share_midcyc"),None, BASE_SC_GEN)
add("", "share: middle-sequence", cond_sc("f2_share_midseq"),None, BASE_SC_GEN)
add("", "router: MLP",            cond_sc("f2_router_mlp"),  None, BASE_SC_GEN)
add("", "no load-balance loss",   cond_sc("f2_nobalance"),   None, BASE_SC_GEN)
# Family 3 — marker selection
add("3. Marker selection / budget", "router markers, M=128 (baseline)", BASE_SC_GEN, None, header=True)
add("", "markers: random",        cond_sc("f3_marker_random"),None, BASE_SC_GEN)
add("", "markers: variance",      cond_sc("f3_marker_var"),  None, BASE_SC_GEN)
add("", "budget M=64",            cond_sc("f3_M64"),         None, BASE_SC_GEN)
add("", "budget M=256",           cond_sc("f3_M256"),        None, BASE_SC_GEN)
# Family 4 — recursion depth
add("4. Recursion depth", "N_R=4 (baseline)", BASE_SC_GEN, None, header=True)
add("", "N_R=1 (no recursion)",   cond_sc("f4_K1"), None, BASE_SC_GEN)
add("", "N_R=6",                  cond_sc("f4_K6"), None, BASE_SC_GEN)
add("", "N_R=8",                  cond_sc("f4_K8"), None, BASE_SC_GEN)
# Family 5 — learned-graph refinements
add("5. Learned-graph refinement", "learned (baseline)", BASE_SC_BIO, None, header=True)
add("", "warm-start from curated",cond_sc("f5_learned_bio"),        None, BASE_SC_BIO)
add("", "fuse coexpr graph",      cond_sc("f5_learned_fused"),      None, BASE_SC_BIO)
add("", "fuse RANDOM (control)",  cond_sc("f5_learned_fused_rand"), None, BASE_SC_BIO)
# Family 6 — input modality (multi-omics)
add("6. Input modality (multi-omics)", "mut+CNV (baseline)", None, BASE_MO_GEN, header=True)
add("", "CNV only",        None, cond_mo("f6_mo_cnv"), None, BASE_MO_GEN)
add("", "mutation only",          None, cond_mo("f6_mo_mut"),  None, BASE_MO_GEN)

# ---- render ----
cols = ["Ablation condition", "SC macro-F1", "ΔSC", "MO macro-F1", "ΔMO"]
n = len(ROWS)
fig, ax = plt.subplots(figsize=(11.5, 1.1 + 0.42*n), dpi=200)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
X = [0.02, 0.46, 0.60, 0.73, 0.92]   # column x-anchors
XL = 0.02
top = 0.95; row_h = 0.92/(n+2)
def L(a,b,y,lw=1.0): ax.plot([a,b],[y,y],color="black",lw=lw,solid_capstyle="butt")
L(0.01,0.99,top+0.5*row_h,1.6)
# header
ax.text(X[0],top,cols[0],ha="left",va="center",fontsize=10.5,fontweight="bold")
for i,c in enumerate(cols[1:],1):
    ax.text(X[i],top,c,ha="center",va="center",fontsize=10.5,fontweight="bold")
L(0.01,0.99,top-0.5*row_h,1.2)
y = top - row_h
for r in ROWS:
    yy = y
    if r["fam"]:
        L(0.01,0.99,yy+0.5*row_h,0.7)
        ax.text(X[0]-0.005,yy,r["fam"],ha="left",va="center",fontsize=9.6,fontweight="bold",color="#333")
        y -= row_h; yy = y
    sc, mo = r["sc"], r["mo"]
    lab_fw = "bold" if r["header"] else "normal"
    ax.text(X[0]+ (0.0 if r["header"] else 0.02), yy, r["label"], ha="left", va="center",
            fontsize=9.2, fontweight=lab_fw, style="italic" if r["header"] else "normal")
    def put(xi, t, base, is_delta=False, dbase=None):
        if is_delta:
            s = delta(t, dbase) if t is not None else DASH
            col = "#1a7f37" if (s not in (DASH,"") and s.startswith("+")) else ("#b00020" if s.startswith("-") else "#666")
            ax.text(X[xi], yy, s, ha="center", va="center", fontsize=8.8, color=col)
        else:
            s = fmt(t) if t is not None else DASH
            c = "#b06000" if s==RUN else "black"
            ax.text(X[xi], yy, s, ha="center", va="center", fontsize=8.8, color=c,
                    style="italic" if s==RUN else "normal",
                    fontweight="bold" if r["header"] else "normal")
    put(1, sc, None); put(2, sc, None, is_delta=not r["header"], dbase=r["bsc"])
    put(3, mo, None); put(4, mo, None, is_delta=not r["header"], dbase=r["bmo"])
    y -= row_h
L(0.01,0.99,y+0.5*row_h,1.6)
ax.text(0.01, y-0.2*row_h,
 "SMART/bioMoR component ablations under the SAME 5-fold CV protocol as the ladder table "
 "(StratifiedKFold-5 seed 42, 20% test / 10%-of-train val, epochs=100 patience=15). "
 "SC = mean macro-F1 over 8 single-cell datasets; MO = mean over 3 P-NET cohorts; ± = mean within-dataset fold SD. "
 "Δ = change vs the italic family baseline. 'run…' = CV tasks still in flight.",
 ha="left", va="top", fontsize=6.8, style="italic")
plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.02)
out = f"{ROOT}/biomor_ablation_table_cv5.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); print("wrote", out)
