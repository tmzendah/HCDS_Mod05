#!/usr/bin/env python3
"""
01_eda_with_quality.py - Exploratory Data Analysis with quality metric assessment.

Integrates image quality assessment (rotation, collimation, sharpness, exposure,
anatomical completeness) into the standard EDA pipeline on bilateral MRKR PNGs.

Input:
  /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/images/

Output (saved to ~/mrkr_klg/eda/):
  - class_distribution.png
  - brightness_histogram.png
  - sample_images_per_kl.png
  - crop_audit/ (directory with cropped halves for visual inspection)
  - quality_metrics_distribution.png
  - quality_metrics_by_kl.png
  - quality_kl_correlation.png
  - eda_summary.txt

Usage:
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/01_eda_with_quality.py --sample_size 100 --quality_sample 500
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import quality_utils as qu

# =============================================================================
# Paths (from 00_setup)
# =============================================================================
RDS_BASE = "/rds/user/tm922/hpc-work"
INPUT_CSV = os.path.join(RDS_BASE, "data", "mrkr", "mrkr_selected_v2.csv")
IMAGE_DIR = os.path.join(RDS_BASE, "data", "mrkr_png_v2", "images")

PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
EDA_DIR = os.path.join(PROJECT_DIR, "eda")
CROP_AUDIT_DIR = os.path.join(EDA_DIR, "crop_audit")

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(CROP_AUDIT_DIR, exist_ok=True)


# =============================================================================
# Cropping logic (same as in preprocessing – used here for visual audit)
# =============================================================================
def crop_bilateral(image, side, flip_flag):
    width, height = image.size
    half = width // 2
    if flip_flag == 0:
        if side == 'L':
            box = (0, 0, half, height)
        else:
            box = (half, 0, width, height)
    else:  # flip_flag == 1
        if side == 'L':
            box = (half, 0, width, height)
        else:
            box = (0, 0, half, height)
    return image.crop(box)


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_quality_metrics_distribution(metrics_df, save_dir):
    """Plot histograms of continuous quality metrics."""
    continuous_cols = [
        "brightness", "noise", "sharpness_laplacian_var",
        "rotation_score", "collimation_score"
    ]
    n_cols = len(continuous_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    for ax, col in zip(axes, continuous_cols):
        values = metrics_df[col].dropna()
        ax.hist(values, bins=30, edgecolor="black")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.set_title(f"{col.replace('_', ' ').title()}\n(n={len(values)})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quality_metrics_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved quality_metrics_distribution.png")


def plot_quality_metrics_by_kl(metrics_df, save_dir):
    """Box plots of key quality metrics stratified by KL grade."""
    metrics_to_plot = ["brightness", "sharpness_laplacian_var", "rotation_score", "collimation_score"]
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, metric in zip(axes, metrics_to_plot):
        data = [metrics_df[metrics_df["kl_grade"] == g][metric].dropna().values for g in range(5)]
        bp = ax.boxplot(data, labels=[f"KL{g}" for g in range(5)])
        ax.set_title(f"{metric.replace('_', ' ').title()} by KL Grade")
        ax.set_xlabel("KL Grade")
        ax.set_ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quality_metrics_by_kl.png"), dpi=150)
    plt.close()
    print(f"  Saved quality_metrics_by_kl.png")


def plot_quality_kl_correlation(metrics_df, save_dir):
    """Computes and plots a Spearman correlation heatmap between continuous
    quality metrics and KL grade.

    Spearman is used (rather than Pearson) because KL grade is ordinal.
    Strong correlations (|rho| > 0.3) suggest a quality metric may be a
    confound — the model could learn the artifact rather than pathology.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns: brightness, noise, sharpness_laplacian_var,
        rotation_score, collimation_score, kl_grade.
    save_dir : str
        Directory to save the output PNG.

    Returns
    -------
    pd.DataFrame
        The computed Spearman correlation matrix (for use in the summary).
    """
    # Columns to correlate (continuous quality metrics + ordinal KL grade)
    corr_cols = [
        "brightness",
        "noise",
        "sharpness_laplacian_var",
        "rotation_score",
        "collimation_score",
        "kl_grade",
    ]
    sub = metrics_df[corr_cols].dropna()
    if len(sub) < 10:
        print("  Skipping correlation plot: too few samples after dropping NaNs.")
        return pd.DataFrame()

    corr_matrix = sub.corr(method="spearman")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    labels = [c.replace("_", " ").title() for c in corr_matrix.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell with the correlation value
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Quality Metrics vs KL Grade\n(Spearman Correlation)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quality_kl_correlation.png"), dpi=150)
    plt.close()
    print(f"  Saved quality_kl_correlation.png")

    # Print notable correlations involving KL grade
    kl_corr = corr_matrix["kl_grade"].drop("kl_grade")
    notable = kl_corr[kl_corr.abs() > 0.2]
    if len(notable):
        print("  Notable correlations with KL grade (|ρ| > 0.2):")
        for metric, rho in notable.items():
            direction = "positive" if rho > 0 else "negative"
            print(f"    {metric:30s}: ρ = {rho:+.3f} ({direction})")
    else:
        print("  No quality metric shows notable correlation with KL grade (all |ρ| ≤ 0.2).")

    return corr_matrix


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of images to sample for detailed audit')
    parser.add_argument('--quality_sample', type=int, default=500,
                        help='Number of images to sample for quality metrics')
    args = parser.parse_args()

    print("=" * 60)
    print("EDA on original bilateral MRKR PNGs (with quality assessment)")
    print("=" * 60)
    ver = qu.cv2_version()
    if ver:
        print(f"OpenCV version: {ver}")
    else:
        print("WARNING: OpenCV not available — quality metrics requiring cv2 will be skipped.")

    # 1. Load metadata
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Required columns
    required = ['png_path', 'knee_side', 'horizontal_flip', 'kl_grade', 'empi_anon']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}'")
            sys.exit(1)

    # Convert horizontal_flip to float (it may be string)
    df['horizontal_flip'] = df['horizontal_flip'].astype(float)

    # 2. Class distribution (should be balanced 2000 each)
    kl_counts = df['kl_grade'].value_counts().sort_index()
    print("\nKL grade distribution in working set:")
    for k in range(5):
        print(f"  KL{k}: {kl_counts.get(k, 0)}")
    plt.figure()
    plt.bar(kl_counts.index, kl_counts.values)
    plt.xlabel("KL Grade")
    plt.ylabel("Count")
    plt.title("MRKR Working Set Class Distribution")
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"))
    plt.close()

    # 3. Image size analysis (sample first 500 images)
    sizes = []
    valid_paths = []
    for idx, row in df.head(500).iterrows():
        img_path = os.path.join(IMAGE_DIR, row['png_path'])
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                sizes.append(img.size)
                valid_paths.append(img_path)
    widths, heights = zip(*sizes)
    print(f"\nImage size stats (sample of {len(sizes)}):")
    print(f"  Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

    # 4. Brightness distribution (sample)
    brightness_vals = []
    for p in valid_paths[:200]:
        img = Image.open(p).convert('L')
        brightness_vals.append(np.mean(np.array(img)))
    plt.figure()
    plt.hist(brightness_vals, bins=30, edgecolor='black')
    plt.xlabel("Mean brightness (0–255)")
    plt.ylabel("Frequency")
    plt.title("Brightness distribution of original bilateral images")
    plt.savefig(os.path.join(EDA_DIR, "brightness_histogram.png"))
    plt.close()
    print(f"  Brightness: mean={np.mean(brightness_vals):.1f}, std={np.std(brightness_vals):.1f}")

    # 5. Sample one image per KL grade (original bilateral)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for grade in range(5):
        sample_row = df[df['kl_grade'] == grade].iloc[0]
        img_path = os.path.join(IMAGE_DIR, sample_row['png_path'])
        img = Image.open(img_path).convert('L')
        axes[grade].imshow(img, cmap='gray')
        axes[grade].set_title(f"KL{grade} (flip={sample_row['horizontal_flip']})")
        axes[grade].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "sample_images_per_kl.png"))
    plt.close()

    # 6. Crop audit: for a sample of images, save the cropped halves
    sample_df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
    print(f"\nGenerating crop audit for {len(sample_df)} images...")
    audit_records = []
    for idx, row in sample_df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['png_path'])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert('L')
        side = row['knee_side']
        flip = row['horizontal_flip']
        cropped = crop_bilateral(img, side, flip)
        # Save cropped image for visual check
        audit_name = f"{row['empi_anon']}_{side}_flip{int(flip)}_KL{row['kl_grade']}.png"
        cropped.save(os.path.join(CROP_AUDIT_DIR, audit_name))
        audit_records.append({
            'patient': row['empi_anon'],
            'side': side,
            'flip': flip,
            'kl': row['kl_grade'],
            'saved_as': audit_name
        })
    print(f"  Saved {len(audit_records)} cropped images to {CROP_AUDIT_DIR}")

    # =========================================================================
    # 7. Quality assessment on a larger sample
    # =========================================================================
    print(f"\nComputing quality metrics on up to {args.quality_sample} images...")
    quality_sample_df = df.sample(n=min(args.quality_sample, len(df)), random_state=123)
    quality_records = []
    quality_failures = 0
    for idx, row in quality_sample_df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['png_path'])
        if not os.path.exists(img_path):
            quality_failures += 1
            continue
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        metrics = qu.compute_quality_metrics(img_array)
        metrics["kl_grade"] = row["kl_grade"]
        metrics["png_path"] = row["png_path"]
        quality_records.append(metrics)

    if quality_records:
        qdf = pd.DataFrame(quality_records)
        print(f"  Computed quality metrics for {len(qdf)} images ({quality_failures} missing files)")

        # Summary statistics
        print("\n  --- Quality Metric Summary ---")
        for col in ["brightness", "noise", "sharpness_laplacian_var", "rotation_score", "collimation_score"]:
            vals = qdf[col].dropna()
            if len(vals):
                print(f"  {col:30s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                      f"min={vals.min():.3f}, max={vals.max():.3f}")

        # Proportion of flagged images
        flag_cols = ["underexposed", "overexposed", "anatomical_truncated", "possible_rotation"]
        if "blurred" in qdf.columns:
            flag_cols += ["blurred", "poor_collimation"]
        for flag_col in flag_cols:
            if flag_col in qdf.columns:
                flagged = qdf[flag_col].sum()
                total = qdf[flag_col].notna().sum()
                if total > 0:
                    print(f"  {flag_col:30s}: {flagged}/{total} ({100 * flagged / total:.1f}%)")

        # Quality by KL grade
        print("\n  --- Quality Metrics by KL Grade ---")
        for g in range(5):
            subset = qdf[qdf["kl_grade"] == g]
            if len(subset) == 0:
                continue
            avg_brightness = subset["brightness"].mean()
            avg_sharpness = subset["sharpness_laplacian_var"].mean()
            avg_rotation = subset["rotation_score"].mean()
            n_blurred = subset["blurred"].sum() if "blurred" in subset.columns else 0
            n_trunc = subset["anatomical_truncated"].sum()
            print(f"    KL{g} (n={len(subset)}): brightness={avg_brightness:.1f}, "
                  f"sharpness={avg_sharpness:.1f}, rotation={avg_rotation:.3f}, "
                  f"blurred={n_blurred}, truncated={n_trunc}")

        # Plot quality distributions
        plot_quality_metrics_distribution(qdf, EDA_DIR)
        plot_quality_metrics_by_kl(qdf, EDA_DIR)

        # Correlation heatmap: quality metrics vs KL grade
        print("\n  --- Quality–KL Correlation Analysis ---")
        corr_matrix = plot_quality_kl_correlation(qdf, EDA_DIR)
    else:
        print("  No quality metrics computed.")
        qdf = pd.DataFrame()

    # =========================================================================
    # 8. Write summary text file
    # =========================================================================
    summary_path = os.path.join(EDA_DIR, "eda_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("MRKR Dataset EDA Summary (with quality assessment)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total images in working set: {len(df)}\n")
        f.write(f"KL distribution: {dict(kl_counts)}\n")
        f.write(f"Image width range: {min(widths)}–{max(widths)} pixels\n")
        f.write(f"Image height range: {min(heights)}–{max(heights)} pixels\n")
        f.write(f"Mean brightness (sample): {np.mean(brightness_vals):.1f}\n")
        f.write(f"CLAHE recommended: yes (brightness variation present)\n")
        f.write(f"Cropping logic audit images saved to: {CROP_AUDIT_DIR}\n")
        f.write("\n--- Quality Metrics ---\n")

        if not qdf.empty:
            for col in ["brightness", "noise", "sharpness_laplacian_var",
                         "rotation_score", "collimation_score"]:
                vals = qdf[col].dropna()
                if len(vals):
                    f.write(f"{col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                            f"range=[{vals.min():.3f}, {vals.max():.3f}]\n")
            flag_cols = ["underexposed", "overexposed", "anatomical_truncated", "possible_rotation"]
            if "blurred" in qdf.columns:
                flag_cols += ["blurred", "poor_collimation"]
            for flag_col in flag_cols:
                if flag_col in qdf.columns:
                    flagged = qdf[flag_col].sum()
                    total = qdf[flag_col].notna().sum()
                    if total > 0:
                        f.write(f"{flag_col}: {flagged}/{total} ({100 * flagged / total:.1f}%)\n")

            # Correlation summary
            if not corr_matrix.empty:
                f.write("\n--- Spearman Correlation with KL Grade ---\n")
                kl_corr = corr_matrix["kl_grade"].drop("kl_grade")
                for metric, rho in kl_corr.items():
                    flag = " ***" if abs(rho) > 0.3 else (" *" if abs(rho) > 0.2 else "")
                    f.write(f"{metric:30s}: ρ = {rho:+.3f}{flag}\n")
                f.write("\n  *  |ρ| > 0.2 (notable)     *** |ρ| > 0.3 (potential confound)\n")
        else:
            f.write("Quality metrics not computed (cv2 unavailable or no images found).\n")

    print(f"\nSummary saved to {summary_path}")
    print("\nEDA complete. Review the crop_audit folder to confirm cropping logic before running preprocessing.")


if __name__ == "__main__":
    main()
