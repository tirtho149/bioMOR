#!/usr/bin/env bash
# Biology-informed router ablation (none/coexpr/random) on the 4 genomap datasets not
# yet covered, so the headline ablation spans all 6. out: results_sc_interaction/<ds>__<mode>__seed<s>.json
#SBATCH --job-name=smart-scint-x
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scintx-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scintx-%A_%a.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
DS=(common_class prototype baron segerstolpe); D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[scint-x] dataset=$D host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.sc_interaction --data data/singlecell --out results_sc_interaction \
    --datasets $D --modes none coexpr random --seeds 0 1 2 \
    --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 --lr 0.001 \
    --weight_decay 0.00001 --patience 15 --device cuda
echo "[scint-x] $D DONE rc=$? date=$(date)"
