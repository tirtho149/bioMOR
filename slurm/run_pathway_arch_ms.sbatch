#!/usr/bin/env bash
#SBATCH --job-name=smart-pw-archms
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pw-archms-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pw-archms-%A_%a.out
# =============================================================================
# Multi-seed (3-seed) P-NET cohort completion of the six-configuration grid, so the
# consolidated standalone table has proper mean+/-std for the cohorts (no single-seed
# placeholders) and the bio(reactome)/none router columns are filled.
# One array task per configuration; each runs all 4 cohorts x 3 seeds.
# Output: results_pathway_ms/<variant>/s<seed>/<tag>.json
#   sbatch run_pathway_arch_ms.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[pw-archms] host=$(hostname) date=$(date)"; nvidia-smi -L || true

# variant -> extra flags (gene_interaction + recursion mode/sharing)
NAMES=(bio none adaptive fixed depth1 independent)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode expert"          # bio = reactome prior
FLAGS[1]="--gene_interaction none --recursion_mode expert"              # general router (no prior)
FLAGS[2]="--gene_interaction reactome --recursion_mode expert"          # adaptive MoR (= headline)
FLAGS[3]="--gene_interaction reactome --recursion_mode fixed"           # fixed depth
FLAGS[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 1"  # K=1
FLAGS[5]="--gene_interaction reactome --recursion_mode expert --no_share_weights"   # vanilla/independent

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[pw-archms] variant=$V flags='$F'"

# (cohort, channels)
COHORTS=("prostate mut_cnv" "blca mut_cnv" "stad mut_cnv" "panmeta_subtype expr")
for SEED in 0 1 2; do
  for CH in "${COHORTS[@]}"; do
    set -- $CH; TASK=$1; CHAN=$2
    BS=32; [ "$CHAN" = "expr" ] && BS=128
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels $CHAN --marker_mode pathway \
        --n_markers 256 --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
        --batch_size $BS --seed $SEED --device cuda \
        --out results_pathway_ms/$V/s$SEED $F
  done
done
echo "[pw-archms] variant=$V DONE rc=$? date=$(date)"
