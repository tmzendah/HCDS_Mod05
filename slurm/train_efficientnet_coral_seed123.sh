#!/bin/bash
#SBATCH --job-name=efficientnet_coral_s123
#SBATCH --output=logs/train_efficientnet_coral_seed123_%j.log
#SBATCH --error=logs/train_efficientnet_coral_seed123_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=YOUR-SLURM-ACCOUNT

# Activate environment
source ~/.bashrc
conda activate knee_oa

# Move to project root
cd ~/knee-oa-kl-grading

# Run training
python src/train.py \
    --arch       efficientnet \
    --loss       coral \
    --seed       123 \
    --data_dir   /rds/user/$USER/hpc-work/data/knee_oa \
    --output_dir results \
    --epochs     30
