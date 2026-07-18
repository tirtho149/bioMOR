#!/usr/bin/env bash
#SBATCH --job-name=ladder-3modal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/ladder-3modal-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/ladder-3modal-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=8
ARM=(vanilla fixed_k2 fixed_k3 fixed_k4 expert_k2 expert_k3 expert_k4 token_k2 token_k3 token_k4)
arm=${ARM[$SLURM_ARRAY_TASK_ID]}
echo "[3modal-ladder] arm=$arm $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task pan_meta_pri_3modal --arm "$arm" --channels mut_cnv_expr \
    --pos_class 1 --path_protocol --seeds 42 43 --cv_folds 5 --d_model 128 \
    --epochs 100 --patience 15 --device cuda --out "results/repro/ladder/pan_meta_pri_3modal_${arm}"
echo "[3modal-ladder] DONE rc=$? arm=$arm $(date)"
