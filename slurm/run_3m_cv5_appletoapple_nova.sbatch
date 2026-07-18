#!/usr/bin/env bash
#SBATCH --job-name=3m-cv5-a2a
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/3m-cv5-a2a-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/3m-cv5-a2a-%A_%a.out
# Make Table 2's 3M column apple-to-apple. The old 3M numbers came from scripts/reproduce_path.py
# with --seeds 42 43 (=> 10 folds) in results/repro/, a DIFFERENT protocol than the rest of
# the table (unified 5-fold, seed 42, 10% val, pathway_tasks). This re-runs 3M
# (pan_meta_pri_3modal, mut+CNV+expr) as a NORMAL cohort through pathway_tasks at 5-fold/seed42:
#   idx 0-9  = baseline ladder arms  -> results/cv5/mo/<variant>   (mirrors run_cv5_mo)
#   idx 10-14= bioMoR-both ladder arms -> results/cv5/biomor_ladder_mo/<N> (mirrors run_biomorboth_ladder_mo)
# K=4 expert bioMoR-both 3M already exists (results/cv5/inject_mo/both). NOVA: --account=mech-ai --qos=normal.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
TASK=pan_meta_pri_3modal; CH=mut_cnv_expr
COMMON="--task $TASK --channels $CH --marker_mode pathway --n_markers 256 --d_model 128 \
 --epochs 100 --lr 3e-4 --patience 15 --batch_size 32 --cv_folds 5 --device cuda"
i=$SLURM_ARRAY_TASK_ID
# ---- baseline ladder arms (general MoR, provided reactome graph) ----
BVAR=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token_k2 token_k3 token)
declare -a BFLAG
BFLAG[0]="--gene_interaction reactome --recursion_mode expert --no_share_weights"
BFLAG[1]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 2"
BFLAG[2]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 3"
BFLAG[3]="--gene_interaction reactome --recursion_mode fixed"
BFLAG[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 2"
BFLAG[5]="--gene_interaction reactome --recursion_mode expert --recursion_depth 3"
BFLAG[6]="--gene_interaction reactome --recursion_mode expert"
BFLAG[7]="--gene_interaction reactome --recursion_mode token  --recursion_depth 2"
BFLAG[8]="--gene_interaction reactome --recursion_mode token  --recursion_depth 3"
BFLAG[9]="--gene_interaction reactome --recursion_mode token"
# ---- bioMoR-both ladder arms (learned graph on BOTH sites) ----
BIO="--gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_pool sum"
MVAR=(expert_k2 expert_k3 token_k2 token_k3 token_k4)
declare -a MFLAG
MFLAG[0]="--recursion_mode expert --recursion_depth 2"
MFLAG[1]="--recursion_mode expert --recursion_depth 3"
MFLAG[2]="--recursion_mode token  --recursion_depth 2"
MFLAG[3]="--recursion_mode token  --recursion_depth 3"
MFLAG[4]="--recursion_mode token  --recursion_depth 4"
if [ "$i" -lt 10 ]; then
  V=${BVAR[$i]}; F="${BFLAG[$i]}"; OUT="results/cv5/mo/$V"
  echo "[3m-a2a] BASELINE variant=$V flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
  $PY -m recursive_marker_transformer.pathway_tasks $COMMON $F --out "$OUT"
else
  j=$(( i - 10 )); V=${MVAR[$j]}; F="${MFLAG[$j]}"; OUT="results/cv5/biomor_ladder_mo/$V"
  echo "[3m-a2a] BIOMOR arm=$V flags='$BIO $F' host=$(hostname) $(date)"; nvidia-smi -L || true
  $PY -m recursive_marker_transformer.pathway_tasks $COMMON $BIO $F --out "$OUT"
fi
echo "[3m-a2a] DONE idx=$i rc=$? $(date)"
