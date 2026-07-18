#!/usr/bin/env bash
#SBATCH --job-name=inject-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-15%16
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/inject-mo-%A_%a.out
# Injection-site ablation on MULTI-OMICS, IDENTICAL protocol to the pathway ladder
# (marker_mode=pathway, recursion_mode=expert, K=4, m=256, d=128, batch=32, lr=3e-4,
#  epochs=100, patience=15, 5-fold CV seed 42), pathway_pool=sum for sparse mutation.
# Interaction graph = the PROVIDED adjacency_matrix.csv (pathway space), never co-expression.
#   cond 0 none  : no biology
#   cond 1 router: reactome centrality prior on the depth router (router site)
#   cond 2 embed : learned pathway graph warm-start+prop+fuse from adjacency_matrix.csv (embedding site)
#   cond 3 both  : router + embed  (= canonical bioMoR, Table-2 MO)
# idx = cohort*4 + cond. Each cond writes to its own subdir (results/cv5/inject_mo/<cond>).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(prostate blca stad brca)
i=$SLURM_ARRAY_TASK_ID
coh=${COHS[$(( i / 4 ))]}; cond=$(( i % 4 ))
CH=mut_cnv; [ "$coh" = brca ] && CH=mut

COMMON="--task $coh --channels $CH --marker_mode pathway --recursion_mode expert \
 --recursion_depth 4 --pathway_pool sum --n_markers 256 --d_model 128 \
 --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 --cv_folds 5 --device cuda"

# REDESIGNED injection sites (provided pathway graph; NO static centrality prior):
#   router = graph feeds ONLY the zero-init graph-conv router (no token smoothing)
#   embed  = graph smooths the pathway tokens (fuse), router off
#   both   = both
case $cond in
  0) NAME=none;   FLAGS="--gene_interaction none" ;;
  1) NAME=router; FLAGS="--gene_interaction none --pathway_learned_graph --no_pathway_prop --bio_graph_router" ;;
  2) NAME=embed;  FLAGS="--gene_interaction none --pathway_learned_graph --pathway_learned_fuse" ;;
  3) NAME=both;   FLAGS="--gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router" ;;
esac
echo "[inject-mo] coh=$coh cond=$NAME host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks $COMMON $FLAGS \
    --out "results/cv5/inject_mo/$NAME"
echo "[inject-mo] DONE coh=$coh cond=$NAME rc=$? $(date)"
