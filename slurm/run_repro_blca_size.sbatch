#!/usr/bin/env bash
#SBATCH --job-name=repro-blca-sz
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-sz-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-blca-sz-%A_%a.out
# Table-2-EXACT protocol (macro-F1 val selection, 5-fold x 2-seed) at varying model
# size; REPORT positive-class(label 1) F1 to compare to PATH blca baselines
# (F1: PATH .81 MOGAT .79 MOGONET .77 eiCNN .75 PathCNN .69). arm x d_model grid.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
ARMS=(vanilla vanilla vanilla vanilla biomor biomor biomor biomor)
DMS=(16 32 64 128 16 32 64 128)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}; DM=${DMS[$SLURM_ARRAY_TASK_ID]}
echo "[repro-blca-sz] arm=$A d_model=$DM host=$(hostname) $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task blca --arm "$A" --channels mut_cnv \
    --pos_class 1 --seeds 42 43 --cv_folds 5 --d_model "$DM" \
    --epochs 100 --patience 15 --device cuda --out "results/repro/sz_${A}_d${DM}"
echo "[repro-blca-sz] DONE rc=$? arm=$A d=$DM $(date)"
