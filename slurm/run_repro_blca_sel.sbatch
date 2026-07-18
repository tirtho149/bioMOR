#!/usr/bin/env bash
#SBATCH --job-name=repro-blca-sel
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-sel-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-sel-%A_%a.out
# Same Table-2 splits/CV/model, but SELECT on positive-class F1 (PATH's reported metric).
# Expect stable pos-F1 ~0.75-0.81 matching PATH baselines. arm x d_model.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
ARMS=(biomor biomor vanilla vanilla); DMS=(32 128 32 128)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}; DM=${DMS[$SLURM_ARRAY_TASK_ID]}
echo "[repro-blca-sel] arm=$A d=$DM select=pos $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task blca --arm "$A" --channels mut_cnv \
    --pos_class 1 --select pos --seeds 42 43 --cv_folds 5 --d_model "$DM" \
    --epochs 100 --patience 15 --device cuda --out "results/repro/sel_${A}_d${DM}"
echo "[repro-blca-sel] DONE rc=$? $(date)"
