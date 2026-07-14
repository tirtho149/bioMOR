#!/usr/bin/env bash
#SBATCH --job-name=biomor-ladder-pnet
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%9
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-ladder-pnet-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-ladder-pnet-%A_%a.out
# bioMoR recursion ladder on multi-omics P-NET: learned gene-graph at K=1/2/3 (K=4 done).
# 3 cohorts x 3 K = 9 tasks; idx -> K=1+idx/3, cohort=idx%3. mode=learned, seeds 0 1 2.
# Writes to results_biomor_ladder_pnet/k<K>/pnet/<cohort>__response/learned_s<seed>.json.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
K=$(( 1 + IDX / 3 ))
COHORT="${COHORTS[$(( IDX % 3 ))]}"
echo "[biomor-ladder-pnet] start $(date) cohort=$COHORT K=$K"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort "$COHORT" --task response \
    --modes learned --seeds 0 1 2 \
    --K "$K" --epochs 60 --cap_genes 3000 --device cuda \
    --out "$PROJ/results_biomor_ladder_pnet/k${K}"
echo "[biomor-ladder-pnet] DONE rc=$? $(date) cohort=$COHORT K=$K"
