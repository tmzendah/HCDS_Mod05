#!/bin/bash
#SBATCH --job-name=mrkr_v3
#SBATCH --output=/home/tm922/mrkr_klg/logs/v3_%j.out
#SBATCH --error=/home/tm922/mrkr_klg/logs/v3_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU

# ─────────────────────────────────────────────────────────────────────────────
# Additional 10K MRKR images — separate from current 10K training dataset
# Patients in mrkr_selected_v2.csv are excluded — no overlap guaranteed
#
# Output:
#   /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v3.csv
#   /rds/user/tm922/hpc-work/data/mrkr_png_v3/images/*.png
#   /rds/user/tm922/hpc-work/data/mrkr_png_v3/mrkr_png_manifest_v3.csv
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo "  MRKR Additional Dataset Pipeline (v3)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Started : $(date)"
echo "============================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/tm922/mrkr_klg
CODE_DIR=$PROJECT_DIR/code
DATA_DIR=/rds/user/tm922/hpc-work/data/mrkr
PNG_DIR=/rds/user/tm922/hpc-work/data/mrkr_png_v3
DICOM_DIR=$DATA_DIR/images_v3

mkdir -p $PNG_DIR $DICOM_DIR

# ── Environment ───────────────────────────────────────────────────────────────
source /home/tm922/miniforge3/etc/profile.d/conda.sh
conda activate mrkr_env

echo "AWS CLI: $(aws --version)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Filter metadata excluding existing patients
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Stage 1 / 3  Filter metadata (excluding v2 patients)"
echo "  Started: $(date)"
echo "============================================================"

python3 $CODE_DIR/mrkr_filter_v3.py \
    --metadata_csv     $DATA_DIR/MRKR_image_metadata.csv \
    --demographics_csv $DATA_DIR/MRKR_demographics.csv \
    --exclude_csv      $DATA_DIR/mrkr_selected_v2.csv \
    --output_csv       $DATA_DIR/mrkr_selected_v3.csv \
    --n_per_grade      4000 \
    --seed             99

echo ""
echo "  Stage 1 complete: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Download DICOMs
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

csv_path   = '/rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v3.csv'
output_dir = '/rds/user/tm922/hpc-work/data/mrkr/images_v3'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
print(f"Images to download: {len(df):,}")
print(f"Label dist: {df['label'].value_counts().sort_index().to_dict()}")

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
    return (result.returncode == 0,
            'downloaded' if result.returncode == 0 else 'failed')

completed = skipped = failed = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_file, row): i
               for i, row in df.iterrows()}
    for future in as_completed(futures):
        success, status = future.result()
        if status == 'skipped':   skipped += 1
        elif success:             completed += 1
        else:                     failed += 1
        total = completed + skipped + failed
        if total % 500 == 0:
            print(f"  Progress: {total}/{len(df)} | "
                  f"Downloaded: {completed} | "
                  f"Skipped: {skipped} | "
                  f"Failed: {failed}")

print(f"\nDownload complete:")
print(f"  Downloaded : {completed:,}")
print(f"  Skipped    : {skipped:,}")
print(f"  Failed     : {failed:,}")
PYEOF

echo ""
echo "  Stage 2 complete: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Convert DICOMs to PNG
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Stage 3 / 3  Convert DICOMs to PNG"
echo "  Started: $(date)"
echo "============================================================"

python3 $CODE_DIR/mrkr_dicom_to_png.py \
    --input_csv   $DATA_DIR/mrkr_selected_v3.csv \
    --dicom_root  $DICOM_DIR \
    --output_dir  $PNG_DIR \
    --num_workers 8

# Rename manifest to v3
mv $PNG_DIR/mrkr_png_manifest.csv $PNG_DIR/mrkr_png_manifest_v3.csv 2>/dev/null || true

echo ""
echo "  Stage 3 complete: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo ""
echo "  Outputs:"
echo "    CSV      : $DATA_DIR/mrkr_selected_v3.csv"
echo "    PNGs     : $PNG_DIR/images/"
echo "    Manifest : $PNG_DIR/mrkr_png_manifest_v3.csv"
echo ""
echo "  No patient overlap with mrkr_selected_v2.csv guaranteed."
echo "  Ready to merge with v2 for 20K training when needed."
echo "============================================================"
