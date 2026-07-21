#!/bin/bash
#SBATCH --job-name=surv3_smoke
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_3mod_smoke.log
#SBATCH --error=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_3mod_smoke.log
bash /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/run_survival_3mod_smoke.sh
