#!/bin/bash
#SBATCH --job-name=mrkr_eda
#SBATCH --output=logs/eda_%j.out
#SBATCH --error=logs/eda_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=icelake

source ~/.bashrc
conda activate OAIKaggle
cd ~/mrkr_klg
python code/01_eda_original.py --sample_size 200
