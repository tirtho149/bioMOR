#!/usr/bin/env bash
#SBATCH --job-name=smart-base11
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/base11-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/base11-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1
echo "[base11] $(date) host=$(hostname)"
$PY -m recursive_marker_transformer.baselines11 \
    --datasets Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Xin prostate blca stad \
    --seeds 0 1 2
echo "[base11] DONE rc=$? $(date)"
