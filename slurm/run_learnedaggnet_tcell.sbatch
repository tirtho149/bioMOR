#!/usr/bin/env bash
#SBATCH --job-name=smart-lagg
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lagg-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lagg-%j.out
# Does warm-starting/anchoring the LEARNED graph from the clean aggregated network
# (learned_aggnet) beat the co-expression warm-start (learned_bio) and plain learned?
# The strongest honest test of "learned graph + real biology" on Tcell SC.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[lagg] $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset Tcell \
    --modes learned_aggnet learned_bio --seeds 0 1 2 3 4 --K 4 --epochs 50 \
    --device cuda --out results_aggnet_sc
echo "[lagg] DONE rc=$? $(date)"
