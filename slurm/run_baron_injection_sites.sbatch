#!/usr/bin/env bash
#SBATCH --job-name=baron-inject
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-inject-%A.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-inject-%A.out
# Injection-site ablation on Baron (single-cell), SAME 5-fold CV (seed 42) as the
# existing biorouter_ablation results so learned_bio is reused (cached), NOT re-run.
#   learned_bio  = interaction in gene-EMBEDDING space only (bar A)
#   route_coexpr = interaction in depth-ROUTER only          (bar B)
#   bio_both     = interaction in BOTH (bioMoR)              (bar C)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
OUT="$PROJ/results/cv5/biorouter_ablation"
echo "[baron-inject] host=$(hostname) $(date)"; nvidia-smi -L || true

"$PY" -m recursive_marker_transformer.bio_learned_genomap --dataset Baron \
   --modes learned_bio route_coexpr bio_both \
   --cv_folds 5 --K 4 --epochs 100 --patience 15 \
   --n_markers 128 --device cuda --out "$OUT"

echo "[baron-inject] DONE rc=$? $(date)"
