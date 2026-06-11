"""Audit rule-based knee crop on a sample of MRKR images.

Saves a grid of original vs cropped images for each stratum:
  - Bilateral / Left knee
  - Bilateral / Right knee
  - Unilateral Left
  - Unilateral Right

Crop logic accounts for horizontal_flip flag from MRKR metadata:
  - Bilateral flip=0: L knee on LEFT half,  R knee on RIGHT half
  - Bilateral flip=1: L knee on RIGHT half, R knee on LEFT half
  - Unilateral:       Centre crop removing 10% from each edge

Usage:
  python mrkr_crop_audit.py \
      --manifest      /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv \
      --img_root      /rds/user/tm922/hpc-work/data/mrkr_png_v2 \
      --output_dir    /home/tm922/mrkr_klg/results/crop_audit_v2 \
      --n_per_stratum 5

Outputs:
  crop_audit_v2/
    audit_bilateral_L.png    -- bilateral left knee crops
    audit_bilateral_R.png    -- bilateral right knee crops
    audit_unilateral_L.png   -- unilateral left crops
    audit_unilateral_R.png   -- unilateral right crops
    audit_summary.json       -- crop dimensions per stratum
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Crop logic
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_crop(img, laterality, knee_side, horizontal_flip=0):
    """Apply rule-based crop using MRKR clinical metadata.

    Bilateral images (laterality == B):
      Standard orientation (horizontal_flip == 0):
        Left knee  → left  50% of image
        Right knee → right 50% of image
      Flipped orientation (horizontal_flip == 1):
        Left knee  → right 50% of image
        Right knee → left  50% of image

    Unilateral images (laterality == L or R):
      Centre crop removing 10% from each edge.
      Removes scanner annotation borders and soft tissue margins
      while keeping the joint centred.

    Parameters
    ----------
    img            : PIL Image (RGB)
    laterality     : 'B', 'L', or 'R' from MRKR metadata
    knee_side      : 'L' or 'R' — which knee this row represents
    horizontal_flip: 0 or 1 from MRKR metadata

    Returns
    -------
    (cropped PIL Image, crop_box tuple)
    """
    w, h = img.size
    flipped = bool(float(horizontal_flip))

    if laterality == "B":
        # Determine which half contains this knee
        # Standard: L=left half, R=right half
        # Flipped:  L=right half, R=left half
        take_left_half = (knee_side == "L" and not flipped) or \
                         (knee_side == "R" and flipped)

        if take_left_half:
            crop_box = (0, 0, w // 2, h)
        else:
            crop_box = (w // 2, 0, w, h)
    else:
        # Unilateral — centre crop removing 10% each side
        margin_w = int(w * 0.10)
        margin_h = int(h * 0.10)
        crop_box = (margin_w, margin_h, w - margin_w, h - margin_h)

    return img.crop(crop_box), crop_box


# ─────────────────────────────────────────────────────────────────────────────
# Audit grid for one stratum
# ─────────────────────────────────────────────────────────────────────────────

def audit_stratum(df_stratum, img_root, n, stratum_name, output_dir):
    """Generate audit grid for one stratum of images."""

    sample = df_stratum.sample(min(n, len(df_stratum)), random_state=42)

    fig, axes = plt.subplots(len(sample), 2,
                             figsize=(10, 4 * len(sample)))
    if len(sample) == 1:
        axes = axes.reshape(1, -1)

    crop_dims = []

    for i, (_, row) in enumerate(sample.iterrows()):
        img_path = os.path.join(img_root, str(row["png_path"]))

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] Cannot open {img_path}: {e}")
            continue

        h_flip = float(row.get("horizontal_flip", 0))

        cropped, crop_box = rule_based_crop(
            img,
            str(row["laterality"]),
            str(row["knee_side"]),
            h_flip,
        )

        crop_dims.append({
            "original_size":   list(img.size),
            "crop_box":        list(crop_box),
            "crop_size":       list(cropped.size),
            "laterality":      str(row["laterality"]),
            "knee_side":       str(row["knee_side"]),
            "horizontal_flip": h_flip,
            "label":           int(row["label"]),
        })

        # ── Original with red crop box overlay ───────────────────────────────
        axes[i, 0].imshow(np.array(img), cmap="gray")
        x0, y0, x1, y1 = crop_box
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(
            f"Original {img.size[0]}x{img.size[1]}\n"
            f"KL={row['label']} | lat={row['laterality']} | "
            f"side={row['knee_side']} | flip={int(h_flip)}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        # ── Cropped result ────────────────────────────────────────────────────
        axes[i, 1].imshow(np.array(cropped), cmap="gray")
        axes[i, 1].set_title(
            f"Cropped {cropped.size[0]}x{cropped.size[1]}\n"
            f"Box: {crop_box}",
            fontsize=9,
        )
        axes[i, 1].axis("off")

    fig.suptitle(
        f"Crop Audit v2 — {stratum_name}  (n={len(sample)})\n"
        f"flip=0: L=left half, R=right half  |  "
        f"flip=1: L=right half, R=left half",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"audit_{stratum_name}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> audit_{stratum_name}.png")

    return crop_dims


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",      type=str, required=True)
    p.add_argument("--img_root",      type=str, required=True)
    p.add_argument("--output_dir",    type=str,
                   default="/home/tm922/mrkr_klg/results/crop_audit_v2")
    p.add_argument("--n_per_stratum", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MRKR — Rule-Based Crop Audit v2 (with flip correction)")
    print("=" * 60)

    df = pd.read_csv(args.manifest)
    print(f"  Manifest rows : {len(df):,}")

    for col in ["png_path", "laterality", "knee_side",
                "label", "horizontal_flip"]:
        if col not in df.columns:
            print(f"  [ERROR] Missing column: {col}")
            return

    print(f"\n  Stratum counts:")
    bil_L = df[(df["laterality"] == "B") & (df["knee_side"] == "L")]
    bil_R = df[(df["laterality"] == "B") & (df["knee_side"] == "R")]
    uni_L = df[df["laterality"] == "L"]
    uni_R = df[df["laterality"] == "R"]

    print(f"  Bilateral  L : {len(bil_L):,}  "
          f"(flip=0: {len(bil_L[bil_L['horizontal_flip']==0]):,}  "
          f"flip=1: {len(bil_L[bil_L['horizontal_flip']==1]):,})")
    print(f"  Bilateral  R : {len(bil_R):,}  "
          f"(flip=0: {len(bil_R[bil_R['horizontal_flip']==0]):,}  "
          f"flip=1: {len(bil_R[bil_R['horizontal_flip']==1]):,})")
    print(f"  Unilateral L : {len(uni_L):,}")
    print(f"  Unilateral R : {len(uni_R):,}")

    summary = {}

    # Audit each stratum — sample includes both flip=0 and flip=1 cases
    for label, subset in [
        ("bilateral_L", bil_L),
        ("bilateral_R", bil_R),
        ("unilateral_L", uni_L),
        ("unilateral_R", uni_R),
    ]:
        section = label.replace("_", " ").title()
        print(f"\n[{list(['bilateral_L','bilateral_R','unilateral_L','unilateral_R']).index(label)+1}/4] {section}")
        if len(subset) == 0:
            print("  [SKIP] No images found")
            continue
        dims = audit_stratum(
            subset, args.img_root,
            args.n_per_stratum, label, args.output_dir
        )
        summary[label] = dims

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "audit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary -> {summary_path}")

    print("\n" + "=" * 60)
    print("  Audit complete.")
    print(f"  Images in: {args.output_dir}")
    print("  Check red box captures the correct knee joint.")
    print("  flip value shown in title for each image.")
    print("=" * 60)


if __name__ == "__main__":
    main()
