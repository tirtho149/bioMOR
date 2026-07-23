#!/bin/bash
#SBATCH --job-name=sct_smoke
#SBATCH --account=mech-ai-scavenger --qos=scavenger --partition=scavenger
#SBATCH --gres=gpu:1 --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8 --mem=64G --time=02:00:00 --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/sct_smoke_%j.out
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venvs_baselines/scTransSort/bin/activate
export BIOMOR_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export TF_CPP_MIN_LOG_LEVEL=2
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines
python sc_baselines/scTransSort/scripts/sctranssort_cv.py --dataset Xin --folds 1 --epochs 3
echo "SMOKE DONE"
