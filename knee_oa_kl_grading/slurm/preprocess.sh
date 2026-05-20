#!/bin/bash
#SBATCH --job-name=mrkr_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=icelake    # or skylake, cpu – check CSD3 partitions

# Load environment
source ~/.bashrc
conda activate OAIKaggle

# Run preprocessing (full dataset)
python code/02_preprocess_parallel.py --workers 32

# Optional: test mode
# python code/02_preprocess_parallel.py --test --workers 32
