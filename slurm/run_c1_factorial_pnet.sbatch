#!/usr/bin/env bash
#SBATCH --job-name=smart-c1-pnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-pnet-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-pnet-%A_%a.out
# C1 confound factorial (P-NET): smoothing vs routing with the curated Reactome graph.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad)
COHORT="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
echo "[c1-pnet] $COHORT $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort "$COHORT" --task response \
    --modes smooth_curated smooth_random route_curated route_random \
    --seeds 0 1 2 3 4 --K 4 --epochs 60 --cap_genes 3000 --device cuda
echo "[c1-pnet] DONE rc=$? $(date)"
