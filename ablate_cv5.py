#!/usr/bin/env python
"""Six ablation families, ALL under the identical 5-fold CV protocol used by the
ladder table (cv.py: StratifiedKFold-5 seed 42, 20% test / 10%-of-train val,
epochs=100 patience=15, mean±SD). Each job = one (condition × dataset); dispatched
by array index to the already-validated per-entry-point `--cv_folds 5` CLIs.

  python ablate_cv5.py --list            # print job count / manifest
  python ablate_cv5.py --index N         # run job N (used by the sbatch array)

Families:
  1 graph source      2 routing internals   3 marker selection
  4 recursion depth   5 learned-graph refinements   6 input-modality robustness
Outputs -> results_cv5/ablation/<condition>/... (unified cv_macro_f1 key).
"""
import argparse, glob, os, subprocess, sys

ROOT = "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
PY = sys.executable
SC_LC  = ["baron","lung","muraro","oesophagus","segerstolpe","spleen","tcell","xin"]
SC_CAP = ["Baron","Lung","Muraro","Oesophagus","Segerstolpe","Spleen","Tcell","Xin"]
MO     = ["prostate","blca","stad"]

# ---- per-runner base flags (everything shared with the main CV5 sweep) ----
SC_BASE = ("--data data/singlecell --cv_folds 5 --seed 42 --epochs 100 --d_model 96 "
           "--n_markers 128 --batch_size 128 --lr 0.001 --weight_decay 0.00001 "
           "--patience 15 --device cuda --recursion_mode expert").split()
BIO_BASE  = "--cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 --device cuda".split()
BIOR_BASE = ("--family pnet --task response --cv_folds 5 --K 4 --epochs 100 --patience 15 "
             "--cap_genes 3000 --device cuda").split()
PATH_BASE = ("--marker_mode pathway --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 "
             "--patience 15 --batch_size 32 --cv_folds 5 --device cuda "
             "--gene_interaction reactome --recursion_mode expert").split()

# condition = (family, name, runner, extra_flags)  runner in {sc,bio,bior,path}
CONDITIONS = [
 # -- Family 1: biological graph SOURCE --
 (1,"f1_sc_none",   "bio", ["--modes","none"]),
 (1,"f1_sc_coexpr", "bio", ["--modes","coexpr"]),
 (1,"f1_sc_random", "bio", ["--modes","random"]),
 (1,"f1_mo_none",   "bior",["--modes","none"]),
 (1,"f1_mo_curated","bior",["--modes","curated"]),
 (1,"f1_mo_random", "bior",["--modes","random"]),
 # -- Family 2: routing internals (general shared-expert model) --
 (2,"f2_share_seq",   "sc",["--share_strategy","sequence"]),
 (2,"f2_share_midcyc","sc",["--share_strategy","middle_cycle"]),
 (2,"f2_share_midseq","sc",["--share_strategy","middle_sequence"]),
 (2,"f2_router_mlp",  "sc",["--router_type","mlp"]),
 (2,"f2_nobalance",   "sc",["--router_balance_coeff","0"]),
 # -- Family 3: marker selection / token budget --
 (3,"f3_marker_random","sc",["--marker_mode","random"]),
 (3,"f3_marker_var",   "sc",["--marker_mode","variance"]),
 (3,"f3_M64",          "sc",["--n_markers","64"]),
 (3,"f3_M256",         "sc",["--n_markers","256"]),
 # -- Family 4: recursion depth beyond the ladder --
 (4,"f4_K1","sc",["--recursion_depth","1"]),
 (4,"f4_K6","sc",["--recursion_depth","6"]),
 (4,"f4_K8","sc",["--recursion_depth","8"]),
 # -- Family 5: learned-graph refinements --
 (5,"f5_learned_bio",       "bio",["--modes","learned_bio"]),
 (5,"f5_learned_fused",     "bio",["--modes","learned_fused"]),
 (5,"f5_learned_fused_rand","bio",["--modes","learned_fused_rand"]),
 # -- Family 6: input-modality robustness (multi-omics) --
 (6,"f6_mo_cnv","path",["--channels","cnv"]),
 (6,"f6_mo_mut","path",["--channels","mut"]),
]

def _out(name): return os.path.join(ROOT, "results_cv5", "ablation", name)

def _expected(runner, name, ds):
    o = _out(name)
    if runner == "sc":   return [os.path.join(o, f"{ds}.json")]
    if runner == "bio":  return [os.path.join(o, ds, "*_cv.json")]
    if runner == "bior": return [os.path.join(o, "pnet", f"{ds}__response", "*_cv.json")]
    if runner == "path": return [os.path.join(o, f"{ds}__*_cv.json")]

def _cmd(runner, name, ds, extra):
    o = _out(name)
    if runner == "sc":
        return [PY,"-m","recursive_marker_transformer.singlecell","--out",o,
                "--datasets",ds] + SC_BASE + extra
    if runner == "bio":
        return [PY,"-m","recursive_marker_transformer.bio_learned_genomap","--out",o,
                "--dataset",ds] + BIO_BASE + extra
    if runner == "bior":
        return [PY,"-m","recursive_marker_transformer.bio_redesign_curated","--out",o,
                "--cohort",ds] + BIOR_BASE + extra
    if runner == "path":
        return [PY,"-m","recursive_marker_transformer.pathway_tasks","--out",o,
                "--task",ds] + PATH_BASE + extra

def build_jobs():
    jobs = []
    for fam, name, runner, extra in CONDITIONS:
        dss = SC_CAP if runner == "bio" else (SC_LC if runner == "sc" else MO)
        for ds in dss:
            jobs.append((fam, name, runner, ds, extra))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = build_jobs()
    if a.list or a.index is None:
        print(f"total jobs: {len(jobs)}")
        by = {}
        for fam,name,runner,ds,_ in jobs: by[name]=by.get(name,0)+1
        for name,c in by.items(): print(f"  {name:22s} x{c}")
        return
    fam, name, runner, ds, extra = jobs[a.index]
    exp = _expected(runner, name, ds)
    if not a.force and all(glob.glob(p) for p in exp):
        print(f"[skip] {name}/{ds} (done)"); return
    cmd = _cmd(runner, name, ds, extra)
    print(f"[ablate] fam={fam} {name} ds={ds}\n  {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.call(cmd, cwd=ROOT))

if __name__ == "__main__":
    main()
