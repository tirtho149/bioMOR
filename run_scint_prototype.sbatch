#!/usr/bin/env bash
# Complete the biology-router ablation for prototype (preempted mid-run). 100 epochs
# (prototype is saturated, converges fast) x none/coexpr/random x 3 seeds.
#SBATCH --job-name=smart-scint-pp
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scintpp-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scintpp-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
echo "[scint-pp] host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.sc_interaction --data data/singlecell --out results_sc_interaction \
    --datasets prototype --modes none coexpr random --seeds 0 1 2 \
    --epochs 100 --d_model 96 --n_markers 128 --batch_size 256 --lr 0.001 \
    --weight_decay 0.00001 --patience 12 --device cuda
echo "[scint-pp] DONE rc=$? date=$(date)"
