#!/usr/bin/env bash
#SBATCH --job-name=smart-c1-sc
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-sc-%A_%a.out
# C1 confound factorial (single-cell): isolate SMOOTHING from ROUTING. Runs the 4 new
# factorial modes (none/learned already exist, 10 seeds). Output merges into
# results_learned_genomap/<Ds>/<mode>_s<seed>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[c1-sc] $DS $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
    --modes smooth_coexpr smooth_random route_coexpr route_random \
    --seeds 0 1 2 3 4 --K 4 --epochs 60 --device cuda
echo "[c1-sc] DONE rc=$? $(date)"
