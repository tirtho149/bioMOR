#!/usr/bin/env bash
#SBATCH --job-name=e100-biomor-pn
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-11%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-biomor-pn-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-biomor-pn-%A_%a.out
# APPLE-TO-APPLE re-run: bioMoR multi-omics (P-NET, learned graph) at epochs=100 patience=15.
# 3 cohorts x {learned K2, learned K3, learned K4, token K4} = 12 tasks. seeds 0 1 2.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
CFG=$(( IDX / 3 )); COHORT="${COHORTS[$(( IDX % 3 ))]}"
case $CFG in
  0) K=2; OUT="$PROJ/results_biomor_e100_pnet/k2"; EXTRA="" ;;
  1) K=3; OUT="$PROJ/results_biomor_e100_pnet/k3"; EXTRA="" ;;
  2) K=4; OUT="$PROJ/results_biomor_e100_pnet/k4"; EXTRA="" ;;
  3) K=4; OUT="$PROJ/results_biomor_token_e100_pnet"; EXTRA="--recursion_mode token" ;;
esac
echo "[e100-biomor-pn] $(date) coh=$COHORT K=$K cfg=$CFG out=$OUT host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort "$COHORT" --task response \
    --modes learned --seeds 0 1 2 3 4 5 6 7 8 9 --K "$K" \
    --epochs 100 --patience 15 --cap_genes 3000 --device cuda $EXTRA --out "$OUT"
echo "[e100-biomor-pn] DONE rc=$? $(date) coh=$COHORT K=$K cfg=$CFG"
