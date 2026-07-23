#!/bin/bash
#SBATCH --job-name=tosica_smoke
#SBATCH --account=mech-ai-scavenger --qos=scavenger --partition=scavenger
#SBATCH --gres=gpu:1 --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8 --mem=64G --time=02:00:00 --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/tosica_smoke_%j.out
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export BIOMOR_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines
python sc_baselines/TOSICA/scripts/tosica_cv.py --dataset Xin --folds 1 --epochs 3
echo "SMOKE DONE"
