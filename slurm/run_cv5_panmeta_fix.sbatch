#!/usr/bin/env bash
#SBATCH --job-name=cv5-panmeta-fix
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-9%10
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-panmeta-fix-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-panmeta-fix-%A_%a.out
# ============================================================================
# Table 2 PM-column fix. pan_meta_pri was trained on the WRONG label column
# (primary_disease, 32-class) instead of the binary `response` (metastatic vs
# primary). load_cohort now selects `response` per data.md. Re-run pan_meta_pri
# (mut+cnv) across ALL 10 Table-2 variants under the IDENTICAL 5-fold CV
# protocol (seed 42, epochs=100, patience=15) so the PM cells are apple-to-apple
# with every other cell. Output: results/cv5/mo/<variant>/<tag>_cv.json
# ============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

VARIANTS=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token token_k2 token_k3)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert --no_share_weights"
FLAGS[1]="--recursion_mode fixed  --recursion_depth 2"
FLAGS[2]="--recursion_mode fixed  --recursion_depth 3"
FLAGS[3]="--recursion_mode fixed"
FLAGS[4]="--recursion_mode expert --recursion_depth 2"
FLAGS[5]="--recursion_mode expert --recursion_depth 3"
FLAGS[6]="--recursion_mode expert"
FLAGS[7]="--recursion_mode token"
FLAGS[8]="--recursion_mode token  --recursion_depth 2"
FLAGS[9]="--recursion_mode token  --recursion_depth 3"

I=$SLURM_ARRAY_TASK_ID
V=${VARIANTS[$I]}; F="${FLAGS[$I]}"
echo "[panmeta-fix] variant=$V flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task pan_meta_pri --channels mut_cnv --marker_mode pathway \
    --gene_interaction reactome \
    --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results/cv5/mo/$V" $F
echo "[panmeta-fix] DONE rc=$? variant=$V $(date)"
