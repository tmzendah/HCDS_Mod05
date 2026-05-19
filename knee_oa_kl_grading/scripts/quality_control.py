"""Quality control checks on converted radiograph images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def assess_image(img_path: Path) -> dict:
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        return {
            "path": str(img_path),
            "mean_intensity": float(arr.mean()),
            "std_intensity": float(arr.std()),
            "min_intensity": float(arr.min()),
            "max_intensity": float(arr.max()),
            "is_blank": arr.std() < 5.0,
            "is_clipped": arr.max() == 255 and arr.mean() > 200,
            "status": "ok",
        }
    except Exception as e:
        return {"path": str(img_path), "status": f"error: {e}"}


def main(args):
    image_dir = Path(args.image_dir)
    images = list(image_dir.rglob("*.png"))
    print(f"Assessing {len(images)} images...")

    records = [assess_image(p) for p in images]
    df = pd.DataFrame(records)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"QC report saved to {out}")
    print(f"Blank images: {df['is_blank'].sum()}")
    print(f"Clipped images: {df['is_clipped'].sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image quality control")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", default="reports/qc_reports/qc_report.csv")
    main(parser.parse_args())
