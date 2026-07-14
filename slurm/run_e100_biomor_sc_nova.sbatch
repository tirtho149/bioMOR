#!/usr/bin/env bash
#SBATCH --job-name=e100-biomor-sc
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

#SBATCH --array=0-35%24
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-biomor-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-biomor-sc-%A_%a.out
# APPLE-TO-APPLE re-run: bioMoR single-cell (learned graph) at epochs=100 patience=15.
# 9 datasets x {learned K2, learned K3, learned K4, token K4} = 36 tasks. seeds 0 1 2.
# cfg = idx/9 (0:lrnK2 1:lrnK3 2:lrnK4 3:tokK4); ds = idx%9.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
IDX=$SLURM_ARRAY_TASK_ID
CFG=$(( IDX / 9 )); DS="${DATASETS[$(( IDX % 9 ))]}"
case $CFG in
  0) K=2; OUT="$PROJ/results_biomor_e100/k2"; EXTRA="" ;;
  1) K=3; OUT="$PROJ/results_biomor_e100/k3"; EXTRA="" ;;
  2) K=4; OUT="$PROJ/results_biomor_e100/k4"; EXTRA="" ;;
  3) K=4; OUT="$PROJ/results_biomor_token_e100"; EXTRA="--recursion_mode token" ;;
esac
echo "[e100-biomor-sc] $(date) ds=$DS K=$K cfg=$CFG out=$OUT host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned --seeds 0 1 2 3 4 5 6 7 8 9 --K "$K" \
    --epochs 100 --patience 15 --device cuda $EXTRA --out "$OUT"
echo "[e100-biomor-sc] DONE rc=$? $(date) ds=$DS K=$K cfg=$CFG"
