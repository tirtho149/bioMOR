#!/usr/bin/env bash
#SBATCH --job-name=mo-reg-sweep
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-23%24
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-reg-sweep-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-reg-sweep-%A_%a.out
# ============================================================================
# Fight overfitting on the small P-NET cohorts (blca/stad/prostate lose to a
# linear baseline at d_model=128). Sweep a SMALLER, more-regularized transformer
# under the IDENTICAL 5-fold CV protocol (seed 42, epochs=100, patience=15,
# marker_mode=pathway, gene_interaction=reactome, recursion_mode=expert/shared K4).
# Only capacity+regularization change (d_model, dropout, weight_decay) -> the
# split/metric/schedule protocol is untouched.
# 8 configs x 3 cohorts = 24.  cfg = idx/3 ; cohort = idx%3.
# Output: results/cv5/mo_reg_sweep/<cfgname>/<tag>_cv.json
# ============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# (d_model, dropout, weight_decay)
DM=(16 16 32 32 32 64 64 64)
DO=(0.5 0.3 0.5 0.3 0.5 0.5 0.3 0.5)
WD=(0.1 0.1 0.1 0.1 0.3 0.1 0.1 0.3)
COH=(prostate blca stad)

I=$SLURM_ARRAY_TASK_ID
C=$(( I / 3 )); T=${COH[$(( I % 3 ))]}
dm=${DM[$C]}; do=${DO[$C]}; wd=${WD[$C]}
NAME="d${dm}_do${do}_wd${wd}"
echo "[reg-sweep] cohort=$T cfg=$NAME host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$T" --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --recursion_mode expert \
    --n_markers 256 --d_model "$dm" --dropout "$do" --weight_decay "$wd" \
    --epochs 100 --lr 3e-4 --patience 15 --batch_size 32 --cv_folds 5 \
    --device cuda --out "results/cv5/mo_reg_sweep/$NAME"
echo "[reg-sweep] DONE rc=$? cohort=$T cfg=$NAME $(date)"
