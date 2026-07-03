#!/usr/bin/env bash
#SBATCH --job-name=smart-bioredesign
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bioredesign-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bioredesign-%j.out
# BIO-ROUTER REDESIGN (BIO_ROUTER_REDESIGN.txt): all 5 fixes ON. Falsification test
# on ONE dataset -- run the identical redesigned mechanism on the real co-expression
# graph (coexpr) vs a degree-matched shuffled graph (random) vs plain router (none).
# Success = coexpr > random ~ none.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
DATASET="${DATASET:-pancreas}"
echo "[bioredesign] start $(date) dataset=$DATASET"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign \
    --dataset "$DATASET" --modes none coexpr random --seeds 0 1 2 \
    --K 4 --epochs 60 --device cuda
echo "[bioredesign] DONE rc=$? $(date)"
