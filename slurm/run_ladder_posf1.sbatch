#!/usr/bin/env bash
#SBATCH --job-name=ladder-posf1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-49
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/ladder-posf1-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/ladder-posf1-%A_%a.out
# Table-2 FULL LADDER x 5 binary cohorts, PATH protocol (AUROC selection + val-tuned
# threshold): threshold-calibrated positive-class F1 AND macro-F1. skip-on-exist =>
# safe to run this same array on BOTH nova and scavenger simultaneously.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COH=(prostate blca stad pan_meta_pri panmeta_response)
CH=(mut_cnv mut_cnv mut_cnv mut_cnv expr)
ARM=(vanilla fixed_k2 fixed_k3 fixed_k4 expert_k2 expert_k3 expert_k4 token_k2 token_k3 token_k4)
i=$SLURM_ARRAY_TASK_ID
c=$(( i / 10 )); a=$(( i % 10 ))
t=${COH[$c]}; ch=${CH[$c]}; arm=${ARM[$a]}
echo "[ladder] task=$t chan=$ch arm=$arm host=$(hostname) $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task "$t" --arm "$arm" --channels "$ch" \
    --pos_class 1 --path_protocol --seeds 42 43 --cv_folds 5 --d_model 128 \
    --epochs 100 --patience 15 --device cuda --out "results/repro/ladder/${t}_${arm}"
echo "[ladder] DONE rc=$? task=$t arm=$arm $(date)"
