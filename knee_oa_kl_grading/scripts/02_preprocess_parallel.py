#!/usr/bin/env python3
"""
02_preprocess_parallel.py - Parallel bilateral cropping, CLAHE, resizing.

Input:
  /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/images/

Output:
  /rds/user/tm922/hpc-work/data/mrkr_cropped/images/ (PNG files)
  /rds/user/tm922/hpc-work/data/mrkr_cropped/mrkr_cropped_manifest.csv

Usage (on HPC login node or as SLURM job):
  conda activate OAIKaggle
  python code/02_preprocess_parallel.py

To test with first 100 images only:
  python code/02_preprocess_parallel.py --test
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

# =============================================================================
# CONFIGURATION – based on 00_setup.py
# =============================================================================
RDS_BASE = "/rds/user/tm922/hpc-work"
INPUT_CSV = os.path.join(RDS_BASE, "data", "mrkr", "mrkr_selected_v2.csv")
INPUT_IMAGE_DIR = os.path.join(RDS_BASE, "data", "mrkr_png_v2", "images")
OUTPUT_IMAGE_DIR = os.path.join(RDS_BASE, "data", "mrkr_cropped", "images")
OUTPUT_MANIFEST = os.path.join(RDS_BASE, "data", "mrkr_cropped", "mrkr_cropped_manifest.csv")

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Output image size (square) – common base size
OUTPUT_SIZE = (512, 512)

# Ensure output directory exists
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# =============================================================================
# Helper functions (must be at top level for multiprocessing)
# =============================================================================
def crop_bilateral(image, side, flip_flag):
    """
    Crop left or right half from a bilateral AP knee X-ray.

    Parameters:
        image: PIL Image (full bilateral)
        side: 'L' or 'R' (knee_side metadata)
        flip_flag: 0 or 1 (horizontal_flip metadata)
    Returns:
        Cropped PIL Image (half of original)
    """
    width, height = image.size
    half = width // 2

    # Flip logic confirmed:
    # flip_flag == 0: normal orientation → left knee on left half, right knee on right half
    # flip_flag == 1: image flipped → left knee on right half, right knee on left half
    if flip_flag == 0:
        if side == 'L':
            box = (0, 0, half, height)
        else:  # 'R'
            box = (half, 0, width, height)
    else:  # flip_flag == 1
        if side == 'L':
            box = (half, 0, width, height)
        else:  # 'R'
            box = (0, 0, half, height)

    return image.crop(box)

def apply_clahe(pil_img, clip_limit=2.0, grid_size=(8,8)):
    """Apply CLAHE to a PIL image (grayscale)."""
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    img_np = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced)

def resize_image(pil_img, target_size):
    """Resize PIL image to target size (width, height)."""
    return pil_img.resize(target_size, Image.Resampling.LANCZOS)

def process_one_row(row, output_dir):
    """Process a single row from the DataFrame – for parallel execution."""
    img_filename = row['png_path']
    img_path = os.path.join(INPUT_IMAGE_DIR, img_filename)

    if not os.path.exists(img_path):
        return None, f"Missing: {img_path}"

    try:
        pil_img = Image.open(img_path).convert('L')
    except Exception as e:
        return None, f"Error loading {img_path}: {e}"

    # Crop
    side = row['knee_side']
    flip = float(row.get('horizontal_flip', 0))
    cropped = crop_bilateral(pil_img, side, flip)

    # CLAHE
    clahe_img = apply_clahe(cropped, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE)

    # Resize
    resized = resize_image(clahe_img, OUTPUT_SIZE)

    # Output filename: patientid_side_kl.png
    patient_id = row['empi_anon']
    kl = int(row['kl_grade'])
    output_filename = f"{patient_id}_{side}_KL{kl}.png"
    output_path = os.path.join(output_dir, output_filename)

    resized.save(output_path, "PNG")

    # Return record for manifest
    record = {
        "cropped_path": output_path,
        "kl_grade": kl,
        "patient_id": patient_id,
        "side": side,
        "original_image": img_filename
    }
    return record, None

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Process only first 100 images')
    parser.add_argument('--workers', type=int, default=None, help='Number of CPU cores (default: all available)')
    args = parser.parse_args()

    print("=" * 60)
    print("MRKR Preprocessing (Parallel): Bilateral cropping + CLAHE + resizing")
    print("=" * 60)

    # 1. Load metadata
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Check required columns
    required_cols = ['png_path', 'knee_side', 'horizontal_flip', 'kl_grade', 'empi_anon']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    # Convert horizontal_flip to float if needed
    df['horizontal_flip'] = df['horizontal_flip'].astype(float)

    # Test mode
    if args.test:
        df = df.head(100)
        print("TEST MODE: processing first 100 rows only")

    # Determine number of workers
    if args.workers:
        n_workers = args.workers
    else:
        n_workers = os.cpu_count()
    print(f"Using {n_workers} CPU workers")

    # 2. Process in parallel
    results = []
    errors = []

    # Use functools.partial to fix output_dir argument
    process_func = functools.partial(process_one_row, output_dir=OUTPUT_IMAGE_DIR)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_func, row): idx for idx, row in df.iterrows()}

        with tqdm(total=len(futures), desc="Processing images") as pbar:
            for future in as_completed(futures):
                record, error = future.result()
                if record:
                    results.append(record)
                elif error:
                    errors.append(error)
                pbar.update(1)

    # 3. Save manifest
    if results:
        manifest_df = pd.DataFrame(results)
        manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
        print(f"\nSaved manifest to {OUTPUT_MANIFEST}")
        print(f"Successfully processed: {len(results)} images")
        if errors:
            print(f"Errors: {len(errors)}")
            with open(OUTPUT_MANIFEST.replace('.csv', '_errors.txt'), 'w') as f:
                f.write("\n".join(errors[:20]))  # first 20 errors
    else:
        print("No images processed successfully. Check errors above.")

    # Quick stats
    if results:
        print("\nKL grade distribution in cropped set:")
        print(pd.Series([r['kl_grade'] for r in results]).value_counts().sort_index())

if __name__ == "__main__":
    main()
