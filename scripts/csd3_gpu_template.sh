#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CSD3 HPC SLURM GPU Job Template
# University of Cambridge — Wilkes3 / Ampere A100
#
# CRITICAL LESSONS LEARNED (saves you hours of debugging):
#
# 1. ENVIRONMENT: Always use OAIKaggle (Python 3.11) for GPU jobs.
#    conda environments built with Python 3.10 cause "Illegal instruction"
#    errors on ampere nodes due to CPU microarchitecture mismatch.
#
# 2. CONDA INIT: Always use eval method, not source method:
#    CORRECT:   eval "$(~/miniforge3/bin/conda shell.bash hook)"
#    INCORRECT: source /home/tm922/miniforge3/etc/profile.d/conda.sh
#
# 3. CUDA MODULE: Always load cuda/11.8 (matches PyTorch cu118 in OAIKaggle)
#    module load cuda/11.8
#
# 4. PYTORCH: OAIKaggle has torch 2.7.1+cu118 with Python 3.11 — works on A100
#
# 5. TEST FIRST: Always run a GPU smoke test (15 min job) before submitting
#    full training runs. Catches errors before wasting hours in the queue.
#
# 6. PARTITION: Use ampere for GPU jobs. Check availability first:
#    sinfo -p ampere
#    If ampere is busy (estimated wait >4hrs), check queue:
#    squeue -p ampere | wc -l
#
# 7. TIME: Set realistic time limits. GPU jobs that exceed the limit are
#    killed with no output saved. Better to overestimate.
#    Typical times on A100:
#      - Smoke test (50 images, 2 epochs): 15 minutes
#      - Full training (10K images, 30 epochs, 512px): 1-2 hours
#      - Full training (20K images, 30 epochs, 512px): 2-4 hours
#
# 8. ACCOUNT: Use TORABI-SL3-GPU for GPU jobs, TORABI-SL3-CPU for CPU jobs
#    Check your balance: mybalance
#
# 9. EXTEND TIME: You cannot extend a running job on CSD3.
#    Set time generously upfront.
#
# 10. ILLEGAL INSTRUCTION: If you see this error, the Python binary is
#     incompatible with the node CPU. Fix = use OAIKaggle environment.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=job_name          # change this
#SBATCH --output=/home/tm922/mrkr_klg/logs/%j_%x.out
#SBATCH --error=/home/tm922/mrkr_klg/logs/%j_%x.err
#SBATCH --time=02:00:00              # HH:MM:SS — set generously
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU

# ── Environment setup ─────────────────────────────────────────────────────────
# CRITICAL: Use eval method and OAIKaggle environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate OAIKaggle

# CRITICAL: Load CUDA 11.8 to match PyTorch cu118
module load cuda/11.8

# ── Confirm GPU is available ──────────────────────────────────────────────────
echo "============================================================"
echo "  Job: $SLURM_JOB_NAME"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "============================================================"

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

python3 -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── Your job goes here ────────────────────────────────────────────────────────
# Replace this section with your actual training command

python3 /path/to/your/script.py \
    --arg1 value1 \
    --arg2 value2

EXIT_CODE=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Job complete"
echo "  Exit code: $EXIT_CODE"
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
