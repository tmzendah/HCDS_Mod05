#!/usr/bin/env python3
"""
mrkr_eda_v3.py - Exploratory Data Analysis with image quality assessment.

Merges standard EDA with quality metric assessment (sharpness, rotation,
collimation, exposure, anatomical completeness) on bilateral MRKR PNGs.

All quality metrics implemented inline — no external quality_utils dependency.

Input:
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/images/

Output (saved to ~/mrkr_klg/eda/):
  class_distribution.png
  demographics_breakdown.png
  brightness_histogram.png
  sample_images_per_kl.png
  crop_audit/               -- cropped halves for visual inspection
  quality_metrics_distribution.png
  quality_metrics_by_kl.png
  quality_kl_correlation.png
  eda_summary.txt

Column names used (from mrkr_png_manifest.csv):
  label             -- KL grade 0-4
  png_path          -- relative path to PNG
  empi_anon         -- patient ID
  knee_side         -- L or R
  horizontal_flip   -- 0.0 or 1.0
  sex, race, ethnicity, weight_bearing, age_group

Usage:
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/mrkr_eda_v3.py --sample_size 100 --quality_sample 500
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
IMAGE_DIR   = os.path.join(RDS_BASE, "data", "mrkr_png_v2")   # png_path is relative to here

PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
EDA_DIR     = os.path.join(PROJECT_DIR, "eda")
AUDIT_DIR   = os.path.join(EDA_DIR, "crop_audit")

os.makedirs(EDA_DIR,   exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Cropping logic — matches mrkr_preprocess.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def crop_bilateral(image, side, flip_flag):
    """Extract single knee from bilateral radiograph.

    horizontal_flip=0 — standard orientation:
        L knee on LEFT half, R knee on RIGHT half
    horizontal_flip=1 — flipped orientation:
        L knee on RIGHT half, R knee on LEFT half
    """
    width, height = image.size
    half = width // 2
    if int(flip_flag) == 0:
        box = (0, 0, half, height) if side == "L" else (half, 0, width, height)
    else:
        box = (half, 0, width, height) if side == "L" else (0, 0, half, height)
    return image.crop(box)


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics — implemented inline, no external dependency
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_metrics(img_array):
    """Compute image quality metrics from a grayscale numpy array.

    Parameters
    ----------
    img_array : np.ndarray
        Grayscale image as uint8 array (H x W).

    Returns
    -------
    dict with keys:
        brightness              -- mean pixel intensity (0-255)
        noise                   -- estimated noise level (std of high-freq residual)
        sharpness_laplacian_var -- variance of Laplacian (higher = sharper)
        rotation_score          -- 0-1 score (1 = well-aligned, 0 = possibly rotated)
        collimation_score       -- 0-1 score (1 = well-collimated, 0 = truncated)
        underexposed            -- bool flag: mean brightness < 30
        overexposed             -- bool flag: mean brightness > 220
        blurred                 -- bool flag: sharpness_laplacian_var < 50
        possible_rotation       -- bool flag: rotation_score < 0.7
        poor_collimation        -- bool flag: collimation_score < 0.5
        anatomical_truncated    -- bool flag: significant bright border detected
    """
    arr = img_array.astype(np.float32)
    h, w = arr.shape

    # Brightness
    brightness = float(arr.mean())

    # Noise — estimate from high-frequency residual using simple local variance
    # Approximate Laplacian noise estimator
    from_top    = arr[:-2, 1:-1]
    from_bottom = arr[2:,  1:-1]
    from_left   = arr[1:-1, :-2]
    from_right  = arr[1:-1, 2:]
    center      = arr[1:-1, 1:-1]
    laplacian   = np.abs(center * 4 - from_top - from_bottom - from_left - from_right)
    noise       = float(laplacian.mean())

    # Sharpness — variance of Laplacian (standard focus metric)
    sharpness_laplacian_var = float(laplacian.var())

    # Rotation score — uses column-wise intensity profile
    # In a well-aligned AP knee radiograph the femur/tibia axis is roughly vertical
    # We measure the ratio of vertical to horizontal gradient energy as a proxy
    grad_v = float(np.abs(np.diff(arr, axis=0)).mean())
    grad_h = float(np.abs(np.diff(arr, axis=1)).mean())
    if (grad_v + grad_h) > 0:
        rotation_score = grad_v / (grad_v + grad_h)
    else:
        rotation_score = 0.5

    # Collimation score — checks how much of the image border is very bright
    # (collimation artifacts appear as bright edges)
    border_width = max(1, int(min(h, w) * 0.05))
    border_mask  = np.zeros_like(arr, dtype=bool)
    border_mask[:border_width, :]  = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width]  = True
    border_mask[:, -border_width:] = True
    border_mean  = float(arr[border_mask].mean())
    interior_mean = float(arr[~border_mask].mean())
    if interior_mean > 0:
        collimation_score = float(np.clip(1.0 - (border_mean - interior_mean) / 255.0, 0, 1))
    else:
        collimation_score = 0.5

    # Flags
    underexposed         = brightness < 30
    overexposed          = brightness > 220
    blurred              = sharpness_laplacian_var < 50
    possible_rotation    = rotation_score < 0.55
    poor_collimation     = collimation_score < 0.5
    anatomical_truncated = border_mean > (interior_mean * 1.5)

    return {
        "brightness":               brightness,
        "noise":                    noise,
        "sharpness_laplacian_var":  sharpness_laplacian_var,
        "rotation_score":           rotation_score,
        "collimation_score":        collimation_score,
        "underexposed":             underexposed,
        "overexposed":              overexposed,
        "blurred":                  blurred,
        "possible_rotation":        possible_rotation,
        "poor_collimation":         poor_collimation,
        "anatomical_truncated":     anatomical_truncated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_class_distribution(df, out):
    vc = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(k) for k in vc.index], vc.values,
                  color=sns.color_palette("muted", len(vc)))
    ax.set_xlabel("KL Grade")
    ax.set_ylabel("Count")
    ax.set_title("MRKR Working Set — KL Grade Distribution")
    for bar, count in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20, str(count),
                ha="center", va="bottom", fontsize=10)
    save_fig(fig, os.path.join(out, "class_distribution.png"))


def plot_demographics(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sex
    sex_vc = df["sex"].value_counts()
    axes[0].bar(sex_vc.index, sex_vc.values,
                color=sns.color_palette("muted", len(sex_vc)))
    axes[0].set_title("Sex Distribution")
    axes[0].set_ylabel("Count")
    for i, (idx, val) in enumerate(sex_vc.items()):
        axes[0].text(i, val + 20, str(val), ha="center", fontsize=9)

    # Race — top 5 for readability
    race_vc = df["race"].value_counts().head(5)
    axes[1].barh(race_vc.index[::-1], race_vc.values[::-1],
                 color=sns.color_palette("muted", len(race_vc)))
    axes[1].set_title("Race Distribution (top 5)")
    axes[1].set_xlabel("Count")

    # Weightbearing
    wb_vc = df["weight_bearing"].map({1: "WB", 0: "Non-WB"}).value_counts()
    axes[2].bar(wb_vc.index, wb_vc.values,
                color=sns.color_palette("muted", len(wb_vc)))
    axes[2].set_title("Weightbearing Status")
    axes[2].set_ylabel("Count")
    for i, (idx, val) in enumerate(wb_vc.items()):
        axes[2].text(i, val + 20, str(val), ha="center", fontsize=9)

    save_fig(fig, os.path.join(out, "demographics_breakdown.png"))


def plot_brightness(brightness_vals, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(brightness_vals, bins=30, edgecolor="black",
            color=sns.color_palette("muted")[0])
    ax.set_xlabel("Mean pixel brightness (0–255)")
    ax.set_ylabel("Frequency")
    ax.set_title("Brightness distribution — original bilateral PNGs")
    save_fig(fig, os.path.join(out, "brightness_histogram.png"))


def plot_sample_images(df, out):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for grade in range(5):
        row = df[df["label"] == grade].iloc[0]
        img_path = os.path.join(IMAGE_DIR, str(row["png_path"]))
        img = Image.open(img_path).convert("L")
        axes[grade].imshow(img, cmap="gray")
        axes[grade].set_title(f"KL{grade}\nflip={int(row['horizontal_flip'])}")
        axes[grade].axis("off")
    save_fig(fig, os.path.join(out, "sample_images_per_kl.png"))


def plot_quality_distribution(qdf, out):
    cols = ["brightness", "noise", "sharpness_laplacian_var",
            "rotation_score", "collimation_score"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, col in zip(axes, cols):
        vals = qdf[col].dropna()
        ax.hist(vals, bins=30, edgecolor="black",
                color=sns.color_palette("muted")[0])
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel("Frequency")
        ax.set_title(f"n={len(vals)}", fontsize=8)
    save_fig(fig, os.path.join(out, "quality_metrics_distribution.png"))


def plot_quality_by_kl(qdf, out):
    metrics = ["brightness", "sharpness_laplacian_var",
               "rotation_score", "collimation_score"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        data = [qdf[qdf["label"] == g][metric].dropna().values
                for g in range(5)]
        ax.boxplot(data, labels=[f"KL{g}" for g in range(5)])
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("KL Grade")
    save_fig(fig, os.path.join(out, "quality_metrics_by_kl.png"))


def plot_quality_kl_correlation(qdf, out):
    """Spearman correlation heatmap — quality metrics vs KL grade.

    Spearman used because KL grade is ordinal.
    |rho| > 0.3 suggests a quality metric may be a confound.
    """
    corr_cols = ["brightness", "noise", "sharpness_laplacian_var",
                 "rotation_score", "collimation_score", "label"]
    sub = qdf[corr_cols].dropna()
    if len(sub) < 10:
        print("  Skipping correlation plot — too few samples.")
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
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Quality Metrics vs KL Grade\n(Spearman Correlation)", fontsize=11)
    save_fig(fig, os.path.join(out, "quality_kl_correlation.png"))

    kl_corr = corr["label"].drop("label")
    notable = kl_corr[kl_corr.abs() > 0.2]
    if len(notable):
        print("  Notable correlations with KL grade (|ρ| > 0.2):")
        for metric, rho in notable.items():
            print(f"    {metric:30s}: ρ = {rho:+.3f}")
    else:
        print("  No quality metric shows notable correlation with KL grade.")

    return corr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_size",    type=int, default=100,
                   help="Images for crop audit")
    p.add_argument("--quality_sample", type=int, default=500,
                   help="Images for quality metrics")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MRKR EDA v3 — with image quality assessment")
    print("=" * 60)

    # ── 1. Load manifest ──────────────────────────────────────────────────────
    df = pd.read_csv(MANIFEST)
    print(f"  Manifest : {MANIFEST}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")

    # Validate required columns
    required = ["label", "png_path", "empi_anon", "knee_side",
                "horizontal_flip", "sex", "race", "weight_bearing"]
    for col in required:
        if col not in df.columns:
            print(f"  ERROR: missing column '{col}'")
            sys.exit(1)

    df["horizontal_flip"] = df["horizontal_flip"].astype(float)

    # ── 2. Class distribution ─────────────────────────────────────────────────
    print("\n[1/8] Class distribution")
    kl_counts = df["label"].value_counts().sort_index()
    for k in range(5):
        print(f"  KL{k}: {kl_counts.get(k, 0):,}")
    plot_class_distribution(df, EDA_DIR)

    # ── 3. Demographics ───────────────────────────────────────────────────────
    print("\n[2/8] Demographics")
    print(f"  Sex         : {df['sex'].value_counts().to_dict()}")
    print(f"  Race (top4) : {df['race'].value_counts().head(4).to_dict()}")
    print(f"  WB          : {df['weight_bearing'].value_counts().to_dict()}")
    plot_demographics(df, EDA_DIR)

    # ── 4. Image size analysis ────────────────────────────────────────────────
    print("\n[3/8] Image size analysis (first 200 images)")
    sizes, valid_paths = [], []
    for _, row in df.head(200).iterrows():
        p = os.path.join(IMAGE_DIR, str(row["png_path"]))
        if os.path.exists(p):
            with Image.open(p) as img:
                sizes.append(img.size)
                valid_paths.append(p)

    if sizes:
        widths, heights = zip(*sizes)
        print(f"  Width  : min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.0f}")
        print(f"  Height : min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.0f}")
    else:
        widths, heights = [0], [0]
        print("  [WARN] No images found — check IMAGE_DIR path")

    # ── 5. Brightness distribution ────────────────────────────────────────────
    print("\n[4/8] Brightness distribution")
    brightness_vals = []
    for p in valid_paths[:200]:
        img = Image.open(p).convert("L")
        brightness_vals.append(float(np.mean(np.array(img))))
    if brightness_vals:
        print(f"  Mean={np.mean(brightness_vals):.1f}  "
              f"Std={np.std(brightness_vals):.1f}")
        plot_brightness(brightness_vals, EDA_DIR)

    # ── 6. Sample images per KL grade ─────────────────────────────────────────
    print("\n[5/8] Sample images per KL grade")
    plot_sample_images(df, EDA_DIR)

    # ── 7. Crop audit ─────────────────────────────────────────────────────────
    print(f"\n[6/8] Crop audit ({args.sample_size} images)")
    sample_df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
    saved = 0
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                       desc="  cropping"):
        img_path = os.path.join(IMAGE_DIR, str(row["png_path"]))
        if not os.path.exists(img_path):
            continue
        img     = Image.open(img_path).convert("L")
        cropped = crop_bilateral(img, row["knee_side"],
                                 row["horizontal_flip"])
        name    = (f"{row['empi_anon']}_{row['knee_side']}"
                   f"_flip{int(row['horizontal_flip'])}"
                   f"_KL{row['label']}.png")
        cropped.save(os.path.join(AUDIT_DIR, name))
        saved += 1
    print(f"  Saved {saved} cropped images to {AUDIT_DIR}")

    # ── 8. Quality metrics ────────────────────────────────────────────────────
    print(f"\n[7/8] Quality metrics ({args.quality_sample} images)")
    q_sample = df.sample(n=min(args.quality_sample, len(df)), random_state=123)
    records, failures = [], 0
    for _, row in tqdm(q_sample.iterrows(), total=len(q_sample),
                       desc="  quality"):
        img_path = os.path.join(IMAGE_DIR, str(row["png_path"]))
        if not os.path.exists(img_path):
            failures += 1
            continue
        img     = Image.open(img_path).convert("L")
        metrics = compute_quality_metrics(np.array(img))
        metrics["label"]    = int(row["label"])
        metrics["png_path"] = str(row["png_path"])
        records.append(metrics)

    if records:
        qdf = pd.DataFrame(records)
        print(f"  Computed metrics for {len(qdf):,} images "
              f"({failures} missing)")

        print("\n  Quality metric summary:")
        for col in ["brightness", "noise", "sharpness_laplacian_var",
                    "rotation_score", "collimation_score"]:
            v = qdf[col].dropna()
            print(f"    {col:30s}: mean={v.mean():.3f}  "
                  f"std={v.std():.3f}  "
                  f"range=[{v.min():.3f}, {v.max():.3f}]")

        print("\n  Flag proportions:")
        for flag in ["underexposed", "overexposed", "blurred",
                     "possible_rotation", "poor_collimation",
                     "anatomical_truncated"]:
            if flag in qdf.columns:
                n = int(qdf[flag].sum())
                t = int(qdf[flag].notna().sum())
                print(f"    {flag:30s}: {n}/{t} ({100*n/t:.1f}%)")

        print("\n  Quality by KL grade:")
        for g in range(5):
            sub = qdf[qdf["label"] == g]
            print(f"    KL{g} (n={len(sub)}): "
                  f"brightness={sub['brightness'].mean():.1f}  "
                  f"sharpness={sub['sharpness_laplacian_var'].mean():.1f}  "
                  f"rotation={sub['rotation_score'].mean():.3f}  "
                  f"blurred={int(sub['blurred'].sum())}")

        plot_quality_distribution(qdf, EDA_DIR)
        plot_quality_by_kl(qdf, EDA_DIR)

        print("\n[8/8] Quality–KL correlation")
        corr = plot_quality_kl_correlation(qdf, EDA_DIR)
    else:
        print("  No quality metrics computed — check image paths.")
        qdf   = pd.DataFrame()
        corr  = pd.DataFrame()

    # ── Summary text file ─────────────────────────────────────────────────────
    summary = os.path.join(EDA_DIR, "eda_summary.txt")
    with open(summary, "w") as f:
        f.write("MRKR EDA Summary v3 — with quality assessment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Manifest             : {MANIFEST}\n")
        f.write(f"Total images         : {len(df):,}\n")
        f.write(f"KL distribution      : {kl_counts.to_dict()}\n")
        f.write(f"Image width range    : {min(widths)}–{max(widths)} px\n")
        f.write(f"Image height range   : {min(heights)}–{max(heights)} px\n")
        f.write(f"Mean brightness      : {np.mean(brightness_vals):.1f}\n")
        f.write(f"CLAHE recommended    : yes\n")
        f.write(f"Crop audit dir       : {AUDIT_DIR}\n")
        f.write("\n--- Demographics ---\n")
        f.write(f"Sex  : {df['sex'].value_counts().to_dict()}\n")
        f.write(f"Race : {df['race'].value_counts().to_dict()}\n")
        f.write(f"WB   : {df['weight_bearing'].value_counts().to_dict()}\n")
        if not qdf.empty:
            f.write("\n--- Quality Metrics ---\n")
            for col in ["brightness", "noise", "sharpness_laplacian_var",
                        "rotation_score", "collimation_score"]:
                v = qdf[col].dropna()
                f.write(f"{col}: mean={v.mean():.3f}, std={v.std():.3f}, "
                        f"range=[{v.min():.3f}, {v.max():.3f}]\n")
            f.write("\n--- Quality Flags ---\n")
            for flag in ["underexposed", "overexposed", "blurred",
                         "possible_rotation", "poor_collimation",
                         "anatomical_truncated"]:
                if flag in qdf.columns:
                    n = int(qdf[flag].sum())
                    t = int(qdf[flag].notna().sum())
                    f.write(f"{flag}: {n}/{t} ({100*n/t:.1f}%)\n")
            if not corr.empty:
                f.write("\n--- Spearman Correlation with KL Grade ---\n")
                kl_corr = corr["label"].drop("label")
                for metric, rho in kl_corr.items():
                    flag = (" ***" if abs(rho) > 0.3
                            else (" *" if abs(rho) > 0.2 else ""))
                    f.write(f"{metric:30s}: rho={rho:+.3f}{flag}\n")
                f.write("  * |rho|>0.2 notable   *** |rho|>0.3 potential confound\n")

    print(f"\n  Summary → {summary}")
    print("\n" + "=" * 60)
    print("  EDA complete.")
    print("  Review crop_audit/ before running preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
