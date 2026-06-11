#!/usr/bin/env python3
"""
mrkr_eda_v2.py - Exploratory Data Analysis with image quality assessment.

Merges standard EDA with image quality metrics (sharpness, rotation,
collimation, exposure) and Spearman correlation analysis between
quality metrics and KL grade.

Input:
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/images/

Output (saved to ~/mrkr_klg/eda/):
  class_distribution.png
  demographics_breakdown.png
  brightness_histogram.png
  sample_images_per_kl.png
  crop_audit/                    -- cropped halves for visual inspection
  quality_metrics_distribution.png
  quality_metrics_by_kl.png
  quality_kl_correlation.png
  eda_summary.txt

Usage:
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/mrkr_eda_v2.py --sample_size 100 --quality_sample 500
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RDS_BASE    = "/rds/user/tm922/hpc-work"
MANIFEST    = os.path.join(RDS_BASE, "data", "mrkr_png_v2", "mrkr_png_manifest.csv")
IMAGE_DIR   = os.path.join(RDS_BASE, "data", "mrkr_png_v2", "images")
PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
EDA_DIR     = os.path.join(PROJECT_DIR, "eda")
AUDIT_DIR   = os.path.join(EDA_DIR, "crop_audit")

os.makedirs(EDA_DIR,   exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, name):
    path = os.path.join(EDA_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def resolve_img_path(png_path):
    """Handle both 'images/xxx.png' and bare filename."""
    p = os.path.join(IMAGE_DIR, png_path) if not png_path.startswith("/") else png_path
    if not os.path.exists(p):
        # try stripping leading 'images/' prefix
        bare = os.path.basename(png_path)
        p2   = os.path.join(IMAGE_DIR, bare)
        if os.path.exists(p2):
            return p2
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Crop logic (same as mrkr_preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────

def crop_bilateral(image, side, flip_flag):
    """Extract single knee from bilateral AP radiograph."""
    width, height = image.size
    half = width // 2
    flip = int(float(flip_flag))
    if flip == 0:
        box = (0, 0, half, height) if side == "L" else (half, 0, width, height)
    else:
        box = (half, 0, width, height) if side == "L" else (0, 0, half, height)
    return image.crop(box)


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics (inline — no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_metrics(img_array):
    """
    Compute image quality metrics on a grayscale numpy array (uint8).

    Returns
    -------
    dict with keys:
      brightness              -- mean pixel value (0–255)
      noise                   -- std pixel value
      sharpness_laplacian_var -- Laplacian variance (higher = sharper)
      rotation_score          -- 0.0 (upright) to 1.0 (rotated)
      collimation_score       -- fraction of non-zero border pixels (0=clean)
      underexposed            -- bool: mean brightness < 50
      overexposed             -- bool: mean brightness > 200
      anatomical_truncated    -- bool: any image border is >90% non-zero
      possible_rotation       -- bool: rotation_score > 0.15
      blurred                 -- bool: sharpness_laplacian_var < 100
      poor_collimation        -- bool: collimation_score > 0.3
    """
    metrics = {}

    arr = img_array.astype(np.float32)

    # Brightness and noise
    metrics["brightness"] = float(arr.mean())
    metrics["noise"]      = float(arr.std())

    # Sharpness via Laplacian variance
    # Approximate Laplacian without cv2 using numpy
    lap = (
        -arr[:-2, 1:-1]
        - arr[2:,  1:-1]
        - arr[1:-1, :-2]
        - arr[1:-1, 2:]
        + 4 * arr[1:-1, 1:-1]
    )
    metrics["sharpness_laplacian_var"] = float(lap.var())

    # Rotation score: compare horizontal vs vertical gradient energy
    # A well-positioned knee X-ray has strong horizontal edges (joint space)
    grad_h = np.abs(np.diff(arr, axis=0)).mean()
    grad_v = np.abs(np.diff(arr, axis=1)).mean()
    total  = grad_h + grad_v + 1e-8
    metrics["rotation_score"] = float(abs(grad_h - grad_v) / total)

    # Collimation score: fraction of border pixels that are non-zero
    h, w = arr.shape
    border_pixels = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    metrics["collimation_score"] = float((border_pixels > 5).mean())

    # Boolean flags
    metrics["underexposed"]       = bool(metrics["brightness"] < 50)
    metrics["overexposed"]        = bool(metrics["brightness"] > 200)
    metrics["possible_rotation"]  = bool(metrics["rotation_score"] > 0.15)
    metrics["blurred"]            = bool(metrics["sharpness_laplacian_var"] < 100)
    metrics["poor_collimation"]   = bool(metrics["collimation_score"] > 0.3)

    # Anatomical truncation: any border row/col >90% non-zero
    borders = [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    metrics["anatomical_truncated"] = bool(
        any((b > 5).mean() > 0.9 for b in borders)
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df):
    print("\n[1] Class distribution")
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"KL {k}" for k in counts.index],
                  counts.values,
                  color=sns.color_palette("muted", len(counts)))
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                str(count), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("KL Grade")
    ax.set_ylabel("Count")
    ax.set_title("MRKR Working Set — KL Grade Distribution")
    save_fig(fig, "class_distribution.png")
    for k, v in counts.items():
        print(f"  KL{k}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Demographics breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_demographics(df):
    print("\n[2] Demographics breakdown")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sex
    sex_counts = df["sex"].value_counts()
    axes[0].bar(sex_counts.index, sex_counts.values,
                color=sns.color_palette("muted", len(sex_counts)))
    axes[0].set_title("Sex distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(sex_counts.values):
        axes[0].text(i, v + sex_counts.max() * 0.01,
                     str(v), ha="center", fontsize=9)

    # Race (top 4 + other)
    race_counts = df["race"].value_counts()
    top4  = race_counts.head(4)
    other = race_counts.iloc[4:].sum()
    race_plot = pd.concat([top4, pd.Series({"Other/Unknown": other})])
    axes[1].barh(race_plot.index[::-1], race_plot.values[::-1],
                 color=sns.color_palette("muted", len(race_plot)))
    axes[1].set_title("Race distribution")
    axes[1].set_xlabel("Count")

    # Weightbearing
    wb_counts = df["weight_bearing"].map({1: "Weightbearing", 0: "Non-WB"}).value_counts()
    axes[2].bar(wb_counts.index, wb_counts.values,
                color=sns.color_palette("muted", len(wb_counts)))
    axes[2].set_title("Weightbearing status")
    axes[2].set_ylabel("Count")
    for i, v in enumerate(wb_counts.values):
        axes[2].text(i, v + wb_counts.max() * 0.01,
                     str(v), ha="center", fontsize=9)

    save_fig(fig, "demographics_breakdown.png")
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    print(f"  Race (top 4): {dict(race_counts.head(4))}")
    print(f"  Weightbearing: {df['weight_bearing'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Brightness histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_brightness(df, n_sample=200):
    print(f"\n[3] Brightness distribution (sample n={n_sample})")
    brightness_vals = []
    for _, row in df.head(n_sample).iterrows():
        p = resolve_img_path(row["png_path"])
        if os.path.exists(p):
            img = Image.open(p).convert("L")
            brightness_vals.append(float(np.array(img).mean()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(brightness_vals, bins=30, edgecolor="black",
            color=sns.color_palette("muted")[0])
    ax.set_xlabel("Mean pixel brightness (0–255)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Brightness distribution — bilateral images (n={len(brightness_vals)})")
    save_fig(fig, "brightness_histogram.png")
    print(f"  Mean={np.mean(brightness_vals):.1f}  Std={np.std(brightness_vals):.1f}")
    return brightness_vals


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sample images per KL grade
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_images(df):
    print("\n[4] Sample images per KL grade")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for grade in range(5):
        row  = df[df["label"] == grade].iloc[0]
        p    = resolve_img_path(row["png_path"])
        if os.path.exists(p):
            img = Image.open(p).convert("L")
            axes[grade].imshow(img, cmap="gray")
        axes[grade].set_title(f"KL{grade}\nflip={int(float(row['horizontal_flip']))}")
        axes[grade].axis("off")
    save_fig(fig, "sample_images_per_kl.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Crop audit
# ─────────────────────────────────────────────────────────────────────────────

def run_crop_audit(df, n_sample=100):
    print(f"\n[5] Crop audit (n={n_sample})")
    sample = df.sample(n=min(n_sample, len(df)), random_state=42)
    saved  = 0
    for _, row in sample.iterrows():
        p = resolve_img_path(row["png_path"])
        if not os.path.exists(p):
            continue
        img     = Image.open(p).convert("L")
        side    = row["knee_side"]
        flip    = row["horizontal_flip"]
        cropped = crop_bilateral(img, side, flip)
        name    = f"{row['empi_anon']}_{side}_flip{int(float(flip))}_KL{row['label']}.png"
        cropped.save(os.path.join(AUDIT_DIR, name))
        saved += 1
    print(f"  Saved {saved} cropped images → {AUDIT_DIR}")
    print(f"  Review these images to confirm crop logic before running preprocessing.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_batch(df, n_sample=500):
    print(f"\n[6] Quality metrics (sample n={n_sample})")
    sample   = df.sample(n=min(n_sample, len(df)), random_state=123)
    records  = []
    failures = 0

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="  quality"):
        p = resolve_img_path(row["png_path"])
        if not os.path.exists(p):
            failures += 1
            continue
        img     = Image.open(p).convert("L")
        arr     = np.array(img)
        metrics = compute_quality_metrics(arr)
        metrics["label"]    = int(row["label"])
        metrics["png_path"] = row["png_path"]
        records.append(metrics)

    if failures:
        print(f"  {failures} images not found — skipped")

    return pd.DataFrame(records)


def plot_quality_distribution(qdf):
    cols = ["brightness", "noise", "sharpness_laplacian_var",
            "rotation_score", "collimation_score"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, col in zip(axes, cols):
        vals = qdf[col].dropna()
        ax.hist(vals, bins=30, edgecolor="black",
                color=sns.color_palette("muted")[0])
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.set_title(f"{col.replace('_', ' ').title()}\n(n={len(vals)})")
    save_fig(fig, "quality_metrics_distribution.png")


def plot_quality_by_kl(qdf):
    cols = ["brightness", "sharpness_laplacian_var",
            "rotation_score", "collimation_score"]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    for ax, col in zip(axes, cols):
        data = [qdf[qdf["label"] == g][col].dropna().values
                for g in range(5)]
        ax.boxplot(data, labels=[f"KL{g}" for g in range(5)])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("KL Grade")
        ax.set_ylabel(col.replace("_", " ").title())
    save_fig(fig, "quality_metrics_by_kl.png")


def plot_quality_correlation(qdf):
    """
    Spearman correlation heatmap between quality metrics and KL grade.

    Spearman is used because KL grade is ordinal not continuous.
    Strong correlations (|rho| > 0.3) suggest the metric may be a
    confound — the model could learn image artefacts rather than pathology.
    """
    corr_cols = ["brightness", "noise", "sharpness_laplacian_var",
                 "rotation_score", "collimation_score", "label"]
    sub = qdf[corr_cols].dropna()
    if len(sub) < 10:
        print("  Skipping correlation — too few samples")
        return pd.DataFrame()

    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("_", " ").title() for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val   = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Quality Metrics vs KL Grade\n(Spearman Correlation)", fontsize=11)
    save_fig(fig, "quality_kl_correlation.png")

    # Print notable correlations with KL grade
    kl_corr = corr["label"].drop("label")
    print("\n  Spearman correlations with KL grade:")
    for metric, rho in kl_corr.items():
        flag = " *** potential confound" if abs(rho) > 0.3 else (
               " *  notable"             if abs(rho) > 0.2 else "")
        print(f"    {metric:30s}: ρ = {rho:+.3f}{flag}")

    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary text file
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(df, brightness_vals, qdf, corr):
    path = os.path.join(EDA_DIR, "eda_summary.txt")
    with open(path, "w") as f:
        f.write("MRKR Dataset EDA Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Manifest        : {MANIFEST}\n")
        f.write(f"Total images    : {len(df)}\n")
        f.write(f"KL distribution : {df['label'].value_counts().sort_index().to_dict()}\n")
        f.write(f"\nDemographics:\n")
        f.write(f"  Sex           : {df['sex'].value_counts().to_dict()}\n")
        f.write(f"  Race          : {df['race'].value_counts().head(4).to_dict()}\n")
        f.write(f"  Weightbearing : {df['weight_bearing'].value_counts().to_dict()}\n")
        f.write(f"\nBrightness (sample n={len(brightness_vals)}):\n")
        f.write(f"  Mean={np.mean(brightness_vals):.1f}  Std={np.std(brightness_vals):.1f}\n")
        f.write(f"  CLAHE recommended: yes\n")

        if not qdf.empty:
            f.write(f"\nQuality Metrics (sample n={len(qdf)}):\n")
            for col in ["brightness", "noise", "sharpness_laplacian_var",
                        "rotation_score", "collimation_score"]:
                vals = qdf[col].dropna()
                f.write(f"  {col:30s}: mean={vals.mean():.3f}  std={vals.std():.3f}\n")

            f.write(f"\nFlagged images:\n")
            for flag in ["underexposed", "overexposed", "blurred",
                         "poor_collimation", "anatomical_truncated", "possible_rotation"]:
                if flag in qdf.columns:
                    n = int(qdf[flag].sum())
                    t = int(qdf[flag].notna().sum())
                    f.write(f"  {flag:30s}: {n}/{t} ({100*n/t:.1f}%)\n")

            if not corr.empty:
                f.write(f"\nSpearman correlation with KL grade:\n")
                kl_corr = corr["label"].drop("label")
                for metric, rho in kl_corr.items():
                    flag = " ***" if abs(rho) > 0.3 else (" *" if abs(rho) > 0.2 else "")
                    f.write(f"  {metric:30s}: ρ = {rho:+.3f}{flag}\n")
                f.write("\n  * |ρ|>0.2 notable   *** |ρ|>0.3 potential confound\n")

    print(f"\n  Summary → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_size",    type=int, default=100,
                   help="Images for crop audit (default 100)")
    p.add_argument("--quality_sample", type=int, default=500,
                   help="Images for quality metrics (default 500)")
    p.add_argument("--brightness_n",   type=int, default=200,
                   help="Images for brightness histogram (default 200)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  MRKR EDA — with quality assessment")
    print("=" * 60)
    print(f"  Manifest : {MANIFEST}")
    print(f"  Image dir: {IMAGE_DIR}")
    print(f"  EDA dir  : {EDA_DIR}")

    # Load manifest
    if not os.path.exists(MANIFEST):
        print(f"ERROR: Manifest not found: {MANIFEST}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    print(f"\n  Loaded {len(df):,} rows")

    # Verify required columns
    required = ["label", "empi_anon", "knee_side",
                "horizontal_flip", "png_path",
                "sex", "race", "weight_bearing"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        sys.exit(1)

    df["horizontal_flip"] = df["horizontal_flip"].astype(float)

    # Run all sections
    plot_class_distribution(df)
    plot_demographics(df)
    brightness_vals = plot_brightness(df, n_sample=args.brightness_n)
    plot_sample_images(df)
    run_crop_audit(df, n_sample=args.sample_size)

    qdf  = compute_quality_batch(df, n_sample=args.quality_sample)
    corr = pd.DataFrame()

    if not qdf.empty:
        plot_quality_distribution(qdf)
        plot_quality_by_kl(qdf)
        corr = plot_quality_correlation(qdf)

        # Print flag summary
        print("\n  Flagged image counts:")
        for flag in ["underexposed", "overexposed", "blurred",
                     "poor_collimation", "anatomical_truncated", "possible_rotation"]:
            if flag in qdf.columns:
                n = int(qdf[flag].sum())
                t = int(qdf[flag].notna().sum())
                print(f"    {flag:30s}: {n}/{t} ({100*n/t:.1f}%)")

    write_summary(df, brightness_vals, qdf, corr)

    print("\n" + "=" * 60)
    print(f"  EDA complete. All outputs in: {EDA_DIR}")
    print(f"  NEXT: Review crop_audit/ images before running preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
