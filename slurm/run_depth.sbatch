#!/usr/bin/env bash
# MoR Table 14 / Fig 5: per-marker recursion-depth export. results_depth/<ds>.json
#SBATCH --job-name=smart-depth
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depth-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depth-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
echo "[depth] host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.depth_viz --data data/singlecell --out results_depth \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell --epochs 60 --d_model 96 --device cuda
echo "[depth] DONE rc=$? date=$(date)"
