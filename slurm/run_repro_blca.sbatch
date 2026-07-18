#!/usr/bin/env bash
#SBATCH --job-name=repro-blca
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-%A_%a.out
# Reproduce PATH's protocol/metric (positive-class F1, progression=label 1, 5-fold x 2-seed)
# with OUR transformer. Target = PATH's blca baselines: F1 0.69-0.81 (MOGAT .79, MOGONET .77,
# eiCNN .75, PathCNN .69, PATH .81). arm 0=vanilla(untied transformer), 1=biomor.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
ARMS=(vanilla biomor)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "[repro-blca] arm=$A host=$(hostname) $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task blca --arm "$A" --channels mut_cnv \
    --pos_class 1 --seeds 42 43 --cv_folds 5 --d_model 128 \
    --epochs 100 --patience 15 --device cuda --out results/repro
echo "[repro-blca] DONE rc=$? arm=$A $(date)"
