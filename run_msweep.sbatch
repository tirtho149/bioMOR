#!/usr/bin/env bash
# Token-reduction validation: accuracy vs number of marker/pathway tokens M on the
# genomap datasets (the O(M^2)-vs-O(N^2) claim -- how few tokens suffice).
# Array: one M per task. out: results_msweep/M<n>/<dataset>.json
#SBATCH --job-name=smart-msweep
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/msweep-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/msweep-%A_%a.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
M=(16 32 64 128 256); MM=${M[$SLURM_ARRAY_TASK_ID]}
echo "[msweep] M=$MM host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.singlecell --data data/singlecell \
    --out "results_msweep/M$MM" \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe \
    --epochs 80 --d_model 96 --n_markers $MM --batch_size 128 --lr 0.001 \
    --weight_decay 0.00001 --patience 12 --device cuda --recursion_mode expert --seed 0
echo "[msweep] M=$MM DONE rc=$? date=$(date)"
