#!/usr/bin/env bash
#SBATCH --job-name=inject-3m
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-3m-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-3m-%A_%a.out
# 3M = pan_meta_pri_3modal (tri-modal mut+CNV+expr) injection ablation, redesigned,
# provided pathway graph. 4 conditions. High mem for the 770MB expression matrix.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COMMON="--task pan_meta_pri_3modal --channels mut_cnv_expr --marker_mode pathway \
 --recursion_mode expert --recursion_depth 4 --pathway_pool sum --n_markers 256 --d_model 128 \
 --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 --cv_folds 5 --device cuda"
case $SLURM_ARRAY_TASK_ID in
  0) NAME=none;   FLAGS="--gene_interaction none" ;;
  1) NAME=router; FLAGS="--gene_interaction none --pathway_learned_graph --no_pathway_prop --bio_graph_router" ;;
  2) NAME=embed;  FLAGS="--gene_interaction none --pathway_learned_graph --pathway_learned_fuse" ;;
  3) NAME=both;   FLAGS="--gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router" ;;
esac
echo "[inject-3m] cond=$NAME host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks $COMMON $FLAGS --out "results/cv5/inject_mo/$NAME"
echo "[inject-3m] DONE cond=$NAME rc=$? $(date)"
