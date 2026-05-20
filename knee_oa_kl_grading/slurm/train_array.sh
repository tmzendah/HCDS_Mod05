#!/bin/bash
#SBATCH --job-name=mrkr_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-ampere

MODELS=("resnet50" "densenet121" "efficientnet_b0")
SEEDS=(42 123 456)

IDX=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$((IDX / 3))
SEED_IDX=$((IDX % 3))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

source ~/.bashrc
conda activate OAIKaggle
cd ~/mrkr_klg

python code/03_train.py --model $MODEL --seed $SEED --epochs 30
