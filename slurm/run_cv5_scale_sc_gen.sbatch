#!/usr/bin/env bash
#SBATCH --job-name=cv5-scale-sc-gen
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-79%64
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-sc-gen-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-sc-gen-%A_%a.out
# 5-FOLD CV model-SIZE sweep (seed 42, 100/15). Single-cell Vanilla + MoR-general.
# 2 models x 5 widths x 8 datasets = 80. model=idx/40 ; w=(idx%40)/8 ; ds=(idx%40)%8.
# Output: results/cv5/scaling_sc/<model>_d<w>/<ds>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
DATASETS=(baron lung muraro oesophagus segerstolpe spleen tcell xin)
IDX=$SLURM_ARRAY_TASK_ID
if [ $(( IDX / 40 )) -eq 0 ]; then MODEL=vanilla; SHARE="--no_share_weights"; else MODEL=morgen; SHARE="--share_weights"; fi
REM=$(( IDX % 40 )); W=${WIDTHS[$(( REM / 8 ))]}; DS=${DATASETS[$(( REM % 8 ))]}
echo "[cv5-scale-sc-gen] model=$MODEL d=$W ds=$DS $(date) host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results/cv5/scaling_sc/${MODEL}_d${W}" \
    --datasets "$DS" --cv_folds 5 --seed 42 \
    --epochs 100 --d_model "$W" --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
    --recursion_mode expert --recursion_depth 4 $SHARE
echo "[cv5-scale-sc-gen] DONE rc=$? $(date) model=$MODEL d=$W ds=$DS"
