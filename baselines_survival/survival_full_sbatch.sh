#!/bin/bash
#SBATCH --job-name=surv_full
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_full.log
#SBATCH --error=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_full.log

bash /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/run_survival_full.sh
