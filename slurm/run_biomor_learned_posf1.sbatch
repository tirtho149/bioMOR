#!/usr/bin/env bash
#SBATCH --job-name=biomor-lrn-pf1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-17
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-lrn-pf1-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-lrn-pf1-%A_%a.out
# bioMoR LEARNED-graph rows for Table 3 pos-F1 (threshold-tuned, added to run_cell_cv).
# 3 cohorts x {expert K2/3/4, token K2/3/4} = 18. skip via --force omitted (writes cv json).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=8
COH=(prostate blca stad)
CFG_K=(2 3 4 2 3 4); CFG_M=("" "" "" "--recursion_mode token" "--recursion_mode token" "--recursion_mode token")
CFG_D=(expert_k2 expert_k3 expert_k4 token_k2 token_k3 token_k4)
i=$SLURM_ARRAY_TASK_ID
c=${COH[$(( i / 6 ))]}; j=$(( i % 6 ))
K=${CFG_K[$j]}; M=${CFG_M[$j]}; D=${CFG_D[$j]}
echo "[biomor-lrn] cohort=$c cfg=$D K=$K $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort "$c" \
    --task response --modes learned --cv_folds 5 --K "$K" $M \
    --epochs 100 --patience 15 --cap_genes 3000 --device cuda \
    --out "results/repro/biomor_learned/$D"
echo "[biomor-lrn] DONE rc=$? cohort=$c cfg=$D $(date)"
