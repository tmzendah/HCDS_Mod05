#!/bin/bash
#SBATCH --job-name=resnet50_ce_s42
#SBATCH --output=logs/train_resnet50_ce_seed42_%j.log
#SBATCH --error=logs/train_resnet50_ce_seed42_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=COMPUTERLAB-SL2-GPU

# Activate environment
source ~/.bashrc
conda activate knee_oa

# Move to project root
cd ~/knee-oa-kl-grading

# Run training
python src/train.py \
    --arch       resnet50 \
    --loss       ce \
    --seed       42 \
    --data_dir   /rds/user/tm922/hpc-work/data/knee_oa \
    --output_dir results \
    --epochs     30
