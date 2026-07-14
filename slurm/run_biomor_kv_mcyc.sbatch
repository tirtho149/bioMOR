#!/usr/bin/env bash
#SBATCH --job-name=biomor-kv-mcyc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-23%16
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-kvmcyc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-kvmcyc-%A_%a.out
# bioMoR + KV-cache (learned graph + fixed routing + step_cache) and
# bioMoR + M-Cyc share (learned graph + expert + middle_cycle), mirroring the general rows.
# idx 0-11 = KV-cache (0-8 SC, 9-11 PN); idx 12-23 = M-Cyc (12-20 SC, 21-23 PN). seeds 0 1 2, K=4.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt 12 ]; then VARIANT=kvcache; J=$IDX;      SCFLAGS="--recursion_mode fixed --step_cache"; PNFLAGS="--recursion_mode fixed --step_cache";
else                       VARIANT=mcyc;    J=$((IDX-12)); SCFLAGS="--share_strategy middle_cycle";      PNFLAGS="--share_strategy middle_cycle"; fi
echo "[biomor-$VARIANT] start $(date) idx=$IDX J=$J host=$(hostname)"; nvidia-smi -L || true
if [ "$J" -lt 9 ]; then
  DS="${DATASETS[$J]}"
  $PY -m recursive_marker_transformer.bio_learned_genomap \
      --dataset "$DS" --modes learned --seeds 0 1 2 --K 4 --epochs 60 --device cuda \
      $SCFLAGS --out "$PROJ/results_biomor_${VARIANT}"
else
  COHORT="${COHORTS[$(( J - 9 ))]}"
  $PY -m recursive_marker_transformer.bio_redesign_curated \
      --family pnet --cohort "$COHORT" --task response \
      --modes learned --seeds 0 1 2 --K 4 --epochs 60 --cap_genes 3000 --device cuda \
      $PNFLAGS --out "$PROJ/results_biomor_${VARIANT}_pnet"
fi
echo "[biomor-$VARIANT] DONE rc=$? $(date) idx=$IDX"
