#!/usr/bin/env bash
#SBATCH --job-name=mo-poolsum
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-poolsum-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-poolsum-%A_%a.out
# Signal lever (NOT capacity): mutation-burden pooling (pathway_pool=sum) for the
# sparse binary mutation matrix, current d128, same 5-fold CV protocol.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COH=(prostate blca stad)
T=${COH[$SLURM_ARRAY_TASK_ID]}
echo "[poolsum] cohort=$T host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$T" --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --recursion_mode expert --pathway_pool sum \
    --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results/cv5/mo_poolsum"
echo "[poolsum] DONE rc=$? cohort=$T $(date)"
