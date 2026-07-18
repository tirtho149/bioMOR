#!/usr/bin/env bash
#SBATCH --job-name=blca-reg-scav
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/blca-reg-scav-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/blca-reg-scav-%A_%a.out
# ============================================================================
# PROOF ON ONE DATASET (blca, the worst overfitter: bioMoR 43.7 < linear 46 /
# nearestcentroid 51.5). Same 5-fold CV protocol (seed 42, epochs=100,
# patience=15, marker_mode=pathway, gene_interaction=reactome, recursion_mode
# =expert/shared K4). idx 0 = CURRENT config (d128,do0.1,wd0.01) reference;
# 1-7 = smaller/more-regularized. If a small config beats the baseline, roll the
# winner out to ALL cohorts x ALL Table-2 rows.
# Output: results/cv5/mo_reg_sweep/<cfgname>/<tag>_cv.json
# ============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# (d_model, dropout, weight_decay)
DM=(128 64 32 32 16 16 8  32)
DO=(0.1 0.5 0.5 0.3 0.5 0.3 0.5 0.5)
WD=(0.01 0.1 0.1 0.1 0.1 0.1 0.1 0.3)

I=$SLURM_ARRAY_TASK_ID
dm=${DM[$I]}; do=${DO[$I]}; wd=${WD[$I]}
NAME="d${dm}_do${do}_wd${wd}"
echo "[blca-proof] cfg=$NAME host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task blca --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --recursion_mode expert \
    --n_markers 256 --d_model "$dm" --dropout "$do" --weight_decay "$wd" \
    --epochs 100 --lr 3e-4 --patience 15 --batch_size 32 --cv_folds 5 \
    --device cuda --out "results/cv5/mo_reg_sweep/$NAME"
echo "[blca-proof] DONE rc=$? cfg=$NAME $(date)"
