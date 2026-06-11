"""MRKR Preprocessing Pipeline — Crop + CLAHE

Applies two preprocessing steps to all 10,000 MRKR PNGs:

Step 1 — Rule-based bilateral crop
  Uses horizontal_flip and knee_side metadata to correctly
  separate each knee from bilateral radiographs.

  Logic:
    flip=0, knee_side=L → left  50% of image
    flip=0, knee_side=R → right 50% of image
    flip=1, knee_side=L → right 50% of image (flipped orientation)
    flip=1, knee_side=R → left  50% of image (flipped orientation)

Step 2 — CLAHE contrast enhancement
  Contrast Limited Adaptive Histogram Equalisation applied to
  each cropped image to improve visibility of subtle joint space
  features relevant to KL 1 and KL 2 grading.
  Parameters: clip_limit=2.0, tile_grid_size=(8,8)
  Justified by Yaylu et al. (2025) who used identical parameters
  for knee OA KL grade classification.

Output
------
  <output_dir>/
    images/
      {original_filename}.png   -- one cropped+enhanced PNG per row
    mrkr_cropped_manifest.csv   -- updated manifest with new png_path
    preprocessing_report.json   -- counts and crop statistics

Usage
-----
  python mrkr_preprocess.py \
      --manifest   /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv \
      --img_root   /rds/user/tm922/hpc-work/data/mrkr_png_v2 \
      --output_dir /rds/user/tm922/hpc-work/data/mrkr_cropped \
      --num_workers 8

Notes
-----
- Idempotent: skips files that already exist
- All images in this dataset are bilateral (laterality == B)
- Unilateral logic included as safety fallback but not expected
- CLAHE applied in grayscale then converted back to RGB
"""

import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Crop logic
# ─────────────────────────────────────────────────────────────────────────────

def get_crop_box(w, h, laterality, knee_side, horizontal_flip):
    """Calculate crop box based on MRKR clinical metadata.

    Parameters
    ----------
    w, h           : image width and height in pixels
    laterality     : 'B', 'L', or 'R'
    knee_side      : 'L' or 'R' — which knee this row represents
    horizontal_flip: 0 or 1

    Returns
    -------
    (x0, y0, x1, y1) crop box for PIL Image.crop()
    """
    flipped = bool(float(horizontal_flip))

    if laterality == "B":
        # Determine which half contains this knee
        # Standard (flip=0): L knee on LEFT, R knee on RIGHT
        # Flipped  (flip=1): L knee on RIGHT, R knee on LEFT
        take_left_half = (
            (knee_side == "L" and not flipped) or
            (knee_side == "R" and flipped)
        )
        if take_left_half:
            return (0, 0, w // 2, h)
        else:
            return (w // 2, 0, w, h)
    else:
        # Unilateral fallback — centre crop removing 10% each side
        margin_w = int(w * 0.10)
        margin_h = int(h * 0.10)
        return (margin_w, margin_h, w - margin_w, h - margin_h)


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img_rgb: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to an RGB image.

    Converts to grayscale, applies CLAHE, converts back to RGB.
    This is the standard approach for radiograph preprocessing
    (Yaylu et al., 2025; clip_limit=2.0, tile_grid_size=(8,8)).

    Parameters
    ----------
    img_rgb        : numpy array (H, W, 3) uint8
    clip_limit     : CLAHE clip limit (default 2.0)
    tile_grid_size : CLAHE tile grid size (default 8x8)

    Returns
    -------
    numpy array (H, W, 3) uint8 with enhanced contrast
    """
    # Convert RGB to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(gray)

    # Convert back to RGB (3-channel) for ImageNet pretrained models
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Process single image
# ─────────────────────────────────────────────────────────────────────────────

def process_one(args_tuple):
    """Process one image: crop + CLAHE + save.

    Returns (status, filename, message)
    """
    row, img_root, output_images_dir = args_tuple

    png_path    = str(row["png_path"])
    filename    = os.path.basename(png_path)
    out_path    = os.path.join(output_images_dir, filename)

    # Skip if already done
    if os.path.exists(out_path):
        return ("skipped", filename, out_path)

    src_path = os.path.join(img_root, png_path)
    if not os.path.exists(src_path):
        return ("failed", filename, f"Source not found: {src_path}")

    try:
        # Load image
        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        # Step 1: Crop
        crop_box = get_crop_box(
            w, h,
            str(row["laterality"]),
            str(row["knee_side"]),
            row.get("horizontal_flip", 0),
        )
        cropped = img.crop(crop_box)

        # Step 2: CLAHE
        img_array   = np.array(cropped, dtype=np.uint8)
        enhanced    = apply_clahe(img_array)
        result_img  = Image.fromarray(enhanced)

        # Save
        result_img.save(out_path, format="PNG")
        return ("success", filename, out_path)

    except Exception as e:
        return ("failed", filename, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MRKR preprocessing: bilateral crop + CLAHE"
    )
    p.add_argument("--manifest",     type=str, required=True,
                   help="Path to mrkr_png_manifest.csv")
    p.add_argument("--img_root",     type=str, required=True,
                   help="Root directory of source PNGs")
    p.add_argument("--output_dir",   type=str, required=True,
                   help="Output directory for cropped+CLAHE PNGs")
    p.add_argument("--num_workers",  type=int, default=8,
                   help="Parallel workers (0=single process)")
    p.add_argument("--clip_limit",   type=float, default=2.0,
                   help="CLAHE clip limit (default 2.0)")
    p.add_argument("--tile_grid",    type=int, default=8,
                   help="CLAHE tile grid size (default 8)")
    return p.parse_args()


def main():
    args = parse_args()

    output_images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    print("=" * 60)
    print("MRKR — Preprocessing Pipeline")
    print("  Step 1: Bilateral crop (flip-corrected)")
    print("  Step 2: CLAHE contrast enhancement")
    print("=" * 60)
    print(f"  Manifest    : {args.manifest}")
    print(f"  Source root : {args.img_root}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Workers     : {args.num_workers}")
    print(f"  CLAHE       : clip_limit={args.clip_limit}, "
          f"tile_grid={args.tile_grid}x{args.tile_grid}")

    # Load manifest
    df = pd.read_csv(args.manifest)
    print(f"\n  Total rows  : {len(df):,}")
    print(f"  Laterality  : {df['laterality'].value_counts().to_dict()}")
    print(f"  Flip counts : {df['horizontal_flip'].value_counts().to_dict()}")
    print(f"  Label dist  : "
          f"{df['label'].value_counts().sort_index().to_dict()}")

    # Build task list
    rows  = df.to_dict("records")
    tasks = [
        (row, args.img_root, output_images_dir)
        for row in rows
    ]

    # Process
    results    = {"success": 0, "skipped": 0, "failed": 0}
    failed_log = []

    if args.num_workers > 0:
        with mp.Pool(processes=args.num_workers) as pool:
            for status, fname, msg in tqdm(
                pool.imap(process_one, tasks),
                total=len(tasks),
                desc="Processing",
            ):
                results[status] += 1
                if status == "failed":
                    failed_log.append({"file": fname, "error": msg})
    else:
        for task in tqdm(tasks, desc="Processing"):
            status, fname, msg = process_one(task)
            results[status] += 1
            if status == "failed":
                failed_log.append({"file": fname, "error": msg})

    # Build updated manifest — same columns, updated png_path
    # png_path stays the same filename, just points to new location
    df["png_path"] = df["png_path"].apply(
        lambda p: f"images/{os.path.basename(p)}"
    )

    manifest_path = os.path.join(args.output_dir,
                                 "mrkr_cropped_manifest.csv")
    df.to_csv(manifest_path, index=False)

    # Save report
    report = {
        "total":        len(df),
        "success":      results["success"],
        "skipped":      results["skipped"],
        "failed":       results["failed"],
        "clahe_params": {
            "clip_limit":     args.clip_limit,
            "tile_grid_size": args.tile_grid,
        },
        "failed_cases": failed_log[:20],
    }
    report_path = os.path.join(args.output_dir,
                               "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Processed  : {results['success']:,}")
    print(f"  Skipped    : {results['skipped']:,}  (already existed)")
    print(f"  Failed     : {results['failed']:,}")
    if failed_log:
        print(f"  First failure: {failed_log[0]}")
    print(f"\n  Manifest   -> {manifest_path}")
    print(f"  Report     -> {report_path}")
    print("\n" + "=" * 60)
    print("  Use these paths for training:")
    print(f"    --data_csv  {manifest_path}")
    print(f"    --img_root  {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
