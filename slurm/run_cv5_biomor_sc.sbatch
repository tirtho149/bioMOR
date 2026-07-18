#!/usr/bin/env bash
#SBATCH --job-name=cv5-biomor-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-31%32
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biomor-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biomor-sc-%A_%a.out
# 5-FOLD CV, seed 42, epochs=100 patience=15. bioMoR single-cell (learned graph).
# 8 datasets x {learned K2, learned K3, learned K4, token K4} = 32 tasks.
# cfg = idx/8 (0:K2 1:K3 2:K4 3:tokK4) ; ds = idx%8.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Baron Lung Muraro Oesophagus Segerstolpe Spleen Tcell Xin)
IDX=$SLURM_ARRAY_TASK_ID
CFG=$(( IDX / 8 )); DS="${DATASETS[$(( IDX % 8 ))]}"
case $CFG in
  0) K=2; OUT="$PROJ/results/cv5/biomor_sc/k2"; EXTRA="" ;;
  1) K=3; OUT="$PROJ/results/cv5/biomor_sc/k3"; EXTRA="" ;;
  2) K=4; OUT="$PROJ/results/cv5/biomor_sc/k4"; EXTRA="" ;;
  3) K=4; OUT="$PROJ/results/cv5/biomor_sc_token"; EXTRA="--recursion_mode token" ;;
esac
echo "[cv5-biomor-sc] $(date) ds=$DS K=$K cfg=$CFG out=$OUT host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned --cv_folds 5 --K "$K" \
    --epochs 100 --patience 15 --n_markers 128 --device cuda $EXTRA --out "$OUT"
echo "[cv5-biomor-sc] DONE rc=$? $(date) ds=$DS K=$K cfg=$CFG"
