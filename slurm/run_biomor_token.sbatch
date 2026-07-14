#!/usr/bin/env bash
#SBATCH --job-name=biomor-token
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-11%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-token-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-token-%A_%a.out
# Token-choice bioMoR: learned gene-graph + TOKEN routing at K=4 (headline uses expert-choice).
# Tasks 0-8 = single-cell datasets; tasks 9-11 = P-NET cohorts. mode=learned, seeds 0 1 2.
# SC -> results_biomor_token/<dataset>/learned_s*.json ; PN -> results_biomor_token_pnet/pnet/<coh>__response/learned_s*.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
echo "[biomor-token] start $(date) idx=$IDX host=$(hostname)"; nvidia-smi -L || true
if [ "$IDX" -lt 9 ]; then
  DS="${DATASETS[$IDX]}"
  $PY -m recursive_marker_transformer.bio_learned_genomap \
      --dataset "$DS" --modes learned --seeds 0 1 2 --K 4 --epochs 60 --device cuda \
      --recursion_mode token --out "$PROJ/results_biomor_token"
else
  COHORT="${COHORTS[$(( IDX - 9 ))]}"
  $PY -m recursive_marker_transformer.bio_redesign_curated \
      --family pnet --cohort "$COHORT" --task response \
      --modes learned --seeds 0 1 2 --K 4 --epochs 60 --cap_genes 3000 --device cuda \
      --recursion_mode token --out "$PROJ/results_biomor_token_pnet"
fi
echo "[biomor-token] DONE rc=$? $(date) idx=$IDX"
