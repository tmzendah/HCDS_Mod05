#!/bin/bash
#SBATCH --job-name=mrkr_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=icelake

source ~/.bashrc
conda activate OAIKaggle
cd ~/mrkr_klg

# Test mode (small subset)
python code/02_preprocess_parallel.py --test --workers 32

# When ready, remove --test and run full:
# python code/02_preprocess_parallel.py --workers 32
