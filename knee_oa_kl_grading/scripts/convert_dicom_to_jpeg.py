"""Convert MRKR DICOM files to JPEG for model training."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def convert_dicom(dicom_path: Path, output_path: Path, size: int = 512) -> None:
    try:
        import pydicom
        dcm = pydicom.dcmread(str(dicom_path))
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8)).resize((size, size))
        img.save(str(output_path))
    except Exception as e:
        print(f"Failed {dicom_path}: {e}")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_files = list(input_dir.rglob("*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files")

    for dcm_path in dicom_files:
        rel = dcm_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_dicom(dcm_path, out_path, size=args.size)

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM files to JPEG")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    main(parser.parse_args())
