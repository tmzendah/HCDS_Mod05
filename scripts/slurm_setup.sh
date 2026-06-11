#!/bin/bash
#SBATCH --job-name=mrkr_setup
#SBATCH --output=/home/tm922/mrkr_klg/logs/setup_%j.out
#SBATCH --error=/home/tm922/mrkr_klg/logs/setup_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=icelake
#SBATCH --account=MLMI-tm922-SL2-CPU

# ─────────────────────────────────────────────────────────────────────────────
# MRKR Setup Pipeline
# Runs three stages in sequence:
#   Stage 1 — Filter metadata + merge demographics → mrkr_selected_v2.csv
#   Stage 2 — Download DICOMs from S3 for selected images
#   Stage 3 — Convert DICOMs to PNG → mrkr_png_manifest.csv
#
# Submit:
#   sbatch ~/mrkr_klg/code/slurm_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error

echo "============================================================"
echo "  MRKR Setup Pipeline"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Started : $(date)"
echo "============================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/tm922/mrkr_klg
CODE_DIR=$PROJECT_DIR/code
DATA_DIR=/rds/user/tm922/hpc-work/data/mrkr
PNG_DIR=/rds/user/tm922/hpc-work/data/mrkr_png
LOG_DIR=$PROJECT_DIR/logs

mkdir -p $LOG_DIR $PNG_DIR

# ── Environment ───────────────────────────────────────────────────────────────
source /home/tm922/miniforge3/etc/profile.d/conda.sh
conda activate mrkr_env

# Confirm AWS CLI available
echo ""
echo "AWS CLI:"
aws --version
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Filter and sample
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Stage 1 / 3  Filter metadata + merge demographics"
echo "  Started: $(date)"
echo "============================================================"

python3 $CODE_DIR/mrkr_filter_v2.py \
    --metadata_csv     $DATA_DIR/MRKR_image_metadata.csv \
    --demographics_csv $DATA_DIR/MRKR_demographics.csv \
    --output_csv       $DATA_DIR/mrkr_selected_v2.csv \
    --n_per_grade      2000 \
    --seed             42

echo ""
echo "  Stage 1 complete: $(date)"
echo "  Output: $DATA_DIR/mrkr_selected_v2.csv"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Download DICOMs from S3
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Stage 2 / 3  Download DICOMs from S3"
echo "  Started: $(date)"
echo "============================================================"

python3 - << 'PYEOF'
import pandas as pd
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_path   = '/rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv'
output_dir = '/rds/user/tm922/hpc-work/data/mrkr/images'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
print(f"Images to download: {len(df):,}")

def download_file(row):
    src = f"s3://emory-mrkr-dataset/images/{row['dicom_path']}"
    dst = os.path.join(output_dir, row['dicom_path'])
    if os.path.exists(dst):
        return (True, 'skipped')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    result = subprocess.run(
        ['aws', 's3', 'cp', src, dst,
         '--region', 'us-east-1', '--quiet'],
        capture_output=True
    )
    return (result.returncode == 0, 'downloaded' if result.returncode == 0 else 'failed')

completed = 0
skipped   = 0
failed    = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_file, row): i
               for i, row in df.iterrows()}
    for future in as_completed(futures):
        success, status = future.result()
        if status == 'skipped':
            skipped += 1
        elif success:
            completed += 1
        else:
            failed += 1
        total = completed + skipped + failed
        if total % 500 == 0:
            print(f"  Progress: {total}/{len(df)} | "
                  f"Downloaded: {completed} | "
                  f"Skipped: {skipped} | "
                  f"Failed: {failed}")

print(f"\nDownload complete:")
print(f"  Downloaded : {completed:,}")
print(f"  Skipped    : {skipped:,}  (already existed)")
print(f"  Failed     : {failed:,}")
PYEOF

echo ""
echo "  Stage 2 complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Convert DICOMs to PNG
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Stage 3 / 3  Convert DICOMs to PNG"
echo "  Started: $(date)"
echo "============================================================"

python3 $CODE_DIR/mrkr_dicom_to_png.py \
    --input_csv   $DATA_DIR/mrkr_selected_v2.csv \
    --dicom_root  $DATA_DIR/images \
    --output_dir  $PNG_DIR \
    --num_workers 8

echo ""
echo "  Stage 3 complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline complete"
echo "  Finished: $(date)"
echo ""
echo "  Outputs:"
echo "    CSV      : $DATA_DIR/mrkr_selected_v2.csv"
echo "    PNGs     : $PNG_DIR/images/"
echo "    Manifest : $PNG_DIR/mrkr_png_manifest.csv"
echo ""
echo "  Next: run EDA then submit training job"
echo "============================================================"
