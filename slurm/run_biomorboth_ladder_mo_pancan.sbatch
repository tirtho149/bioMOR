#!/usr/bin/env bash
#SBATCH --job-name=bmboth-ladder-mo-pancan
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-pancan-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-pancan-%A_%a.out
# Fill the MISSING PM/PC (pancan) cells of Table 2's bioMoR ladder. The main
# ladder-MO run (run_biomorboth_ladder_mo.sbatch) only covered the P-NET cohorts
# (prostate/blca/stad/brca); the two pancan cohorts were never run at K=2,3, so
# scripts/build_cv5_tex.py renders them as 'run...'. K=4 already exists via inject_mo/both.
# 2 cohorts x 4 (K,routing) configs = 8 tasks. Same protocol as run_biomorboth_ladder_mo.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# PM = pan_meta_pri (mut+CNV), PC = panmeta_response (expression)
COHS=(pan_meta_pri panmeta_response); CHS=(mut_cnv expr)
CFG_K=(2 3 2 3); CFG_R=(expert expert token token); CFG_N=(expert_k2 expert_k3 token_k2 token_k3)
i=$SLURM_ARRAY_TASK_ID
ci=$(( i % 4 )); cj=$(( i / 4 ))
coh=${COHS[$cj]}; CH=${CHS[$cj]}
K=${CFG_K[$ci]}; R=${CFG_R[$ci]}; N=${CFG_N[$ci]}
echo "[bmboth-mo-pancan] coh=$coh ch=$CH cfg=$N host=$(hostname) $(date)"; nvidia-smi -L || true
# REDESIGNED both = provided pathway graph on BOTH sites (embedding smoothing + fuse +
# zero-init graph-conv router), NO static centrality prior -- identical to inject_mo/both.
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --recursion_mode "$R" \
    --recursion_depth "$K" --gene_interaction none \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_pool sum \
    --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 \
    --cv_folds 5 --device cuda --out "results/cv5/biomor_ladder_mo/$N"
echo "[bmboth-mo-pancan] DONE coh=$coh cfg=$N rc=$? $(date)"
