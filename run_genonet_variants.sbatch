#!/usr/bin/env bash
#SBATCH --job-name=smart-genonet-var
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genonet-var-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genonet-var-%j.out
# =============================================================================
# Run the Independent (untied) and MoR (token-routing) model variants on ALL
# six BIO5 TCGA tasks, to fill in the per-task 3-model comparison (Table 4).
# Shared-recursive results already exist in results_genonet/.
#   sbatch run_genonet_variants.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[var] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[var] torch',torch.__version__,'cuda',torch.cuda.is_available())"

COMMON="--csv data/tcga/unified_bio5.csv --epochs 40 --d_model 128 --n_markers 256 \
        --batch_size 32 --lr 3e-4 --patience 8 --device cuda"

echo "==================== INDEPENDENT (untied blocks) ===================="
$PY -m recursive_marker_transformer.genonet_tasks $COMMON \
    --out results_genonet_independent --recursion_mode expert --no_share_weights
RC1=$?
echo "[var] independent rc=$RC1 date=$(date)"

echo "==================== MoR (token routing) ===================="
$PY -m recursive_marker_transformer.genonet_tasks $COMMON \
    --out results_genonet_mor --recursion_mode token
RC2=$?
echo "[var] mor rc=$RC2 date=$(date)"

echo "[var] DONE independent=$RC1 mor=$RC2 date=$(date)"
