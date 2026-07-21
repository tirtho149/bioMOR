#!/bin/bash
#SBATCH --job-name=surv_risk
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_risk.log
#SBATCH --error=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/survival_risk.log

bash /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/run_survival_risk.sh
