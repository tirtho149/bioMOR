#!/usr/bin/env bash
#SBATCH --job-name=inject-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-sc-%A_%a.out
# Injection-site ablation (new Table 3), single-cell: the 3 non-both conditions.
# 'bio_both' (=both) comes from run_canonical_biomor_sc.sbatch into the same dir.
#   none         = no biology            route_coexpr = router-only
#   learned_bio  = embedding-only
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DS=(Segerstolpe Lung Oesophagus Baron Muraro Tcell Spleen Xin)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[inject-sc] dataset=$D host=$(hostname) $(date)"; nvidia-smi -L || true
# Redesigned 3-way injection ablation: none / route_graph (router-only, zero-init
# graph-conv router) / learned_bio (embedding-only). bio_both (=both) from the canonical job.
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$D" \
    --modes none learned_bio route_graph bio_both --cv_folds 5 --K 4 --epochs 100 --patience 15 \
    --n_markers 128 --device cuda --out "results/cv5/biomor_canonical"
echo "[inject-sc] DONE dataset=$D rc=$? $(date)"
