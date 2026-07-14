#!/usr/bin/env bash
#SBATCH --job-name=scale-sc-bio
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-44%20
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-sc-bio-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-sc-bio-%A_%a.out
# Scaling study, single-cell bioMoR (learned graph) at widths {96,136,192,272,352}, 100/15, 3 seeds.
# 5 widths x 9 datasets = 45 tasks. w=WIDTHS[idx/9]; ds=DATASETS[idx%9].
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
IDX=$SLURM_ARRAY_TASK_ID
W=${WIDTHS[$(( IDX / 9 ))]}; DS="${DATASETS[$(( IDX % 9 ))]}"
echo "[scale-sc-bio] ds=$DS d_model=$W $(date) host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned --seeds 0 1 2 --K 4 \
    --epochs 100 --patience 15 --d_model "$W" --device cuda \
    --out "$PROJ/results_scaling_sc_biomor/d${W}"
echo "[scale-sc-bio] DONE rc=$? $(date) ds=$DS d=$W"
