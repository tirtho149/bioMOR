#!/usr/bin/env bash
#SBATCH --job-name=repro-all
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --array=0-11%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-all-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/repro-all-%A_%a.out
# EXACT Table-2 protocol: macro-F1 val selection, argmax@0.5, 5-fold x 2-seed, d128.
# NO threshold tuning / NO AUROC selection. Just report positive-class F1 AND macro-F1.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
TASK=(prostate prostate blca blca stad stad pan_meta_pri pan_meta_pri panmeta_response panmeta_response brca brca)
CHAN=(mut_cnv mut_cnv mut_cnv mut_cnv mut_cnv mut_cnv mut_cnv mut_cnv expr expr mut_cnv mut_cnv)
ARM=(vanilla biomor vanilla biomor vanilla biomor vanilla biomor vanilla biomor vanilla biomor)
POS=(1 1 1 1 1 1 1 1 1 1 0 0)
i=$SLURM_ARRAY_TASK_ID
t=${TASK[$i]}; ch=${CHAN[$i]}; a=${ARM[$i]}; p=${POS[$i]}
echo "[repro-all] task=$t chan=$ch arm=$a pos=$p (Table-2 protocol) $(date)"; nvidia-smi -L || true
$PY scripts/reproduce_path.py --task "$t" --arm "$a" --channels "$ch" \
    --pos_class "$p" --select macro --seeds 42 43 --cv_folds 5 --d_model 128 \
    --epochs 100 --patience 15 --device cuda --out "results/repro/all_${t}_${a}"
echo "[repro-all] DONE rc=$? task=$t arm=$a $(date)"
