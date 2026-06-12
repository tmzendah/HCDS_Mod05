#!/bin/bash
# ─────────────────────────────────────────────────────────
# submit_all.sh
# Submits all 12 training jobs to the CSD3 GPU queue.
#
# Usage:
#   cd ~/knee-oa-kl-grading
#   bash scripts/slurm/submit_all.sh
#
# Monitor jobs:
#   squeue -u $USER
#
# Check logs:
#   tail -f logs/train_resnet50_ce_seed42_*.log
#
# Cancel all your jobs if needed:
#   scancel -u $USER
# ─────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

echo "========================================"
echo "  Submitting 12 training jobs"
echo "  Project: $PROJECT_DIR"
echo "========================================"
echo ""

SUBMITTED=0

for script in "$SCRIPT_DIR"/train_*.sh; do
    name=$(basename "$script" .sh)
    job_id=$(sbatch "$script" | awk '{print $NF}')
    echo "  submitted $name  ->  job $job_id"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "  Total submitted: $SUBMITTED / 12"
echo ""
echo "  Monitor with:  squeue -u $USER"
echo "  Cancel all:    scancel -u $USER"
echo "========================================"
