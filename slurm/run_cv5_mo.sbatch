#!/usr/bin/env bash
#SBATCH --job-name=cv5-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-39%40
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-mo-%A_%a.out
# 5-FOLD CV, seed 42, epochs=100 patience=15. Multi-omics GENERAL rows (Pro/BL/ST/PM/PC).
# 8 variants x 5 tasks = 40. variant = idx/5 ; task = idx%5.
# tasks: prostate,blca,stad,pan_meta_pri -> mut_cnv ; panmeta_response -> expr.
# Output: results_cv5/mo/<variant>/<tag>_cv.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
VARIANTS=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode expert --no_share_weights"
FLAGS[1]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 2"
FLAGS[2]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 3"
FLAGS[3]="--gene_interaction reactome --recursion_mode fixed"
FLAGS[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 2"
FLAGS[5]="--gene_interaction reactome --recursion_mode expert --recursion_depth 3"
FLAGS[6]="--gene_interaction reactome --recursion_mode expert"
FLAGS[7]="--gene_interaction reactome --recursion_mode token"
TASKS=(prostate blca stad pan_meta_pri panmeta_response)
CHANS=(mut_cnv mut_cnv mut_cnv mut_cnv expr)

IDX=$SLURM_ARRAY_TASK_ID
V=${VARIANTS[$(( IDX / 5 ))]}; F="${FLAGS[$(( IDX / 5 ))]}"
T=${TASKS[$(( IDX % 5 ))]}; CH=${CHANS[$(( IDX % 5 ))]}
echo "[cv5-mo] variant=$V task=$T chan=$CH flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$T" --channels "$CH" --marker_mode pathway \
    --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results_cv5/mo/$V" $F
echo "[cv5-mo] DONE rc=$? variant=$V task=$T $(date)"
