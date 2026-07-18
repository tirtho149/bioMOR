#!/usr/bin/env bash
#SBATCH --job-name=bmboth-ladder-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-39%20
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-sc-%A_%a.out
# Canonical bioMoR (bio_both) across the efficiency ladder for Table 2, single-cell.
# 5 (K,routing) configs x 8 datasets; identical protocol (d96,e100,p15,m128), CV5 seed42.
# expert_k4 headline = run_canonical_biomor_sc (results/cv5/biomor_canonical).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DS=(Segerstolpe Lung Oesophagus Baron Muraro Tcell Spleen Xin)
CFG_K=(2 3 2 3 4); CFG_R=(expert expert token token token); CFG_N=(expert_k2 expert_k3 token_k2 token_k3 token_k4)
i=$SLURM_ARRAY_TASK_ID
D=${DS[$(( i % 8 ))]}; ci=$(( i / 8 ))
K=${CFG_K[$ci]}; R=${CFG_R[$ci]}; N=${CFG_N[$ci]}
echo "[bmboth-sc] D=$D cfg=$N host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$D" \
    --modes bio_both --cv_folds 5 --K "$K" --recursion_mode "$R" \
    --epochs 100 --patience 15 --n_markers 128 --device cuda \
    --out "results/cv5/biomor_ladder/$N"
echo "[bmboth-sc] DONE D=$D cfg=$N rc=$? $(date)"
