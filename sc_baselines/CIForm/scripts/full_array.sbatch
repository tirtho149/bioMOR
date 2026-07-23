#!/bin/bash
#SBATCH --job-name=ciform_full
#SBATCH --account=mech-ai-scavenger --qos=scavenger --partition=scavenger
#SBATCH --gres=gpu:1 --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8 --mem=96G --time=08:00:00 --requeue
#SBATCH --array=0-6
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/ciform_full_%A_%a.out
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export BIOMOR_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines
DS=(Baron Lung Oesophagus Segerstolpe Spleen Tcell Xin)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "=== CIForm full run dataset=$D ==="
python sc_baselines/CIForm/scripts/ciform_cv.py --dataset "$D" --folds 5 --epochs 20
echo "FULL DONE $D"
