#!/bin/bash
#SBATCH --job-name=mrkr_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-ampere

source ~/.bashrc
conda activate OAIKaggle
cd ~/mrkr_klg
python code/04_evaluate.py
