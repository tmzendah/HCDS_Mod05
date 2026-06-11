"""Convert MRKR DICOMs to PNG using selected_images.csv

Works with the exact column structure of selected_images.csv:
  empi_anon, KLG, age_at_exam, weight_bearing, laterality, dicom_path

Output
------
  <output_dir>/
    images/
      {empi_anon}_{dicom_stem}.png  -- one PNG per DICOM
    mrkr_png_manifest.csv           -- updated CSV with png_path column added
    conversion_report.json          -- success / skipped / failed counts

Usage
-----
  python mrkr_dicom_to_png.py \
      --input_csv   /rds/user/tm922/hpc-work/data/mrkr/selected_images.csv \
      --dicom_root  /rds/user/tm922/hpc-work/data/mrkr/images \
      --output_dir  /rds/user/tm922/hpc-work/data/mrkr_png \
      --num_workers 8

Requirements
------------
  pip install pydicom pylibjpeg pylibjpeg-libjpeg
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
    import pylibjpeg  # noqa: F401 — registers JPEG Lossless codec
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}\n"
        "Install: pip install pydicom pylibjpeg pylibjpeg-libjpeg"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_png_name(row: dict) -> str:
    """Derive a unique PNG filename from empi_anon + last part of dicom_path."""
    stem = Path(str(row["dicom_path"])).stem
    return f"{row['empi_anon']}_{stem}.png"


def convert_one(
    row: dict,
    dicom_root: Path,
    output_images_dir: Path,
) -> Tuple[str, str]:
    """Convert one DICOM to PNG. Returns (status, png_path_or_error)."""
    png_name = make_png_name(row)
    png_path = output_images_dir / png_name

    # Skip if already done — safe to rerun after interruption
    if png_path.exists():
        return ("skipped", str(png_path))

    dcm_full = dicom_root / str(row["dicom_path"])
    if not dcm_full.exists():
        return ("failed", f"DICOM not found: {dcm_full}")

    try:
        dcm = pydicom.dcmread(str(dcm_full))
        arr = dcm.pixel_array.astype(np.float32)

        # Normalise to [0, 1]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

        # Inversion correction via DICOM tag — more reliable than metadata flag
        # MONOCHROME1: bright = air, dark = bone — invert for standard view
        # MONOCHROME2: bright = bone — standard radiographic convention
        photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            arr = 1.0 - arr

        # Convert to uint8 RGB (3-channel for ImageNet pretrained models)
        arr_uint8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8).convert("RGB")
        img.save(str(png_path), format="PNG")

        return ("success", str(png_path))

    except Exception as e:
        return ("failed", str(e))


def worker(args: Tuple) -> Tuple[str, str, str]:
    """Multiprocessing wrapper."""
    row, dicom_root, output_images_dir = args
    status, msg = convert_one(row, Path(dicom_root), Path(output_images_dir))
    return (status, make_png_name(row), msg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",   type=str, required=True,
                   help="Path to selected_images.csv")
    p.add_argument("--dicom_root",  type=str, required=True,
                   help="Root directory containing DICOM files")
    p.add_argument("--output_dir",  type=str, required=True,
                   help="Output directory for PNGs and manifest")
    p.add_argument("--num_workers", type=int, default=8,
                   help="Parallel workers (0 = single process)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dicom_root        = Path(args.dicom_root)
    output_dir        = Path(args.output_dir)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MRKR  —  DICOM to PNG Conversion")
    print("=" * 60)
    print(f"  Input CSV  : {args.input_csv}")
    print(f"  DICOM root : {dicom_root}")
    print(f"  Output dir : {output_dir}")
    print(f"  Workers    : {args.num_workers}")

    df = pd.read_csv(args.input_csv)
    print(f"\n  Total rows : {len(df):,}")
    print(f"  KLG dist   : {df['KLG'].value_counts().sort_index().to_dict()}")

    rows  = df.to_dict("records")
    tasks = [(row, str(dicom_root), str(output_images_dir)) for row in rows]

    results    = {"success": 0, "skipped": 0, "failed": 0}
    png_map    = {}
    failed_log = []

    if args.num_workers > 0:
        with mp.Pool(processes=args.num_workers) as pool:
            for status, png_name, msg in tqdm(
                pool.imap(worker, tasks),
                total=len(tasks),
                desc="Converting",
            ):
                results[status] += 1
                if status in ("success", "skipped"):
                    png_map[png_name] = f"images/{png_name}"
                else:
                    failed_log.append({"png_name": png_name, "error": msg})
    else:
        for row in tqdm(rows, desc="Converting"):
            status, msg = convert_one(row, dicom_root, output_images_dir)
            png_name = make_png_name(row)
            results[status] += 1
            if status in ("success", "skipped"):
                png_map[png_name] = f"images/{png_name}"
            else:
                failed_log.append({"png_name": png_name, "error": msg})

    # Build manifest — same as input with png_path added, KLG renamed to label
    df["png_name"] = df.apply(make_png_name, axis=1)
    df["png_path"] = df["png_name"].map(png_map)
    df = df.drop(columns=["png_name"])
    df = df.rename(columns={"KLG": "label"})

    manifest_path = output_dir / "mrkr_png_manifest.csv"
    df.to_csv(manifest_path, index=False)

    report = {
        "total":        len(df),
        "success":      results["success"],
        "skipped":      results["skipped"],
        "failed":       results["failed"],
        "output_dir":   str(output_images_dir),
        "manifest":     str(manifest_path),
        "failed_cases": failed_log[:20],
    }
    with (output_dir / "conversion_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Converted  : {results['success']:,}")
    print(f"  Skipped    : {results['skipped']:,}  (already existed)")
    print(f"  Failed     : {results['failed']:,}")
    if failed_log:
        print(f"  First failure: {failed_log[0]}")
    print(f"\n  Manifest   -> {manifest_path}")
    print(f"  Columns    : {list(df.columns)}")
    print("\n" + "=" * 60)
    print("  Use these paths in training:")
    print(f"    --data_csv  {manifest_path}")
    print(f"    --img_root  {output_images_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
