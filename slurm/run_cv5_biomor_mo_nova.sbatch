#!/usr/bin/env bash
#SBATCH --job-name=cv5-biomor-mo
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-11%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biomor-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biomor-mo-%A_%a.out
# 5-FOLD CV, seed 42, epochs=100 patience=15. bioMoR multi-omics (P-NET, learned graph).
# 3 cohorts x {learned K2, learned K3, learned K4, token K4} = 12. cfg=idx/3 ; coh=idx%3.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
CFG=$(( IDX / 3 )); COHORT="${COHORTS[$(( IDX % 3 ))]}"
case $CFG in
  0) K=2; OUT="$PROJ/results/cv5/biomor_mo/k2"; EXTRA="" ;;
  1) K=3; OUT="$PROJ/results/cv5/biomor_mo/k3"; EXTRA="" ;;
  2) K=4; OUT="$PROJ/results/cv5/biomor_mo/k4"; EXTRA="" ;;
  3) K=4; OUT="$PROJ/results/cv5/biomor_mo_token"; EXTRA="--recursion_mode token" ;;
esac
echo "[cv5-biomor-mo] $(date) coh=$COHORT K=$K cfg=$CFG out=$OUT host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort "$COHORT" --task response \
    --modes learned --cv_folds 5 --K "$K" \
    --epochs 100 --patience 15 --cap_genes 3000 --device cuda $EXTRA --out "$OUT"
echo "[cv5-biomor-mo] DONE rc=$? $(date) coh=$COHORT K=$K cfg=$CFG"
