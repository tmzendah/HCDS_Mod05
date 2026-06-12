"""
01_eda.py
Exploratory Data Analysis — Knee OA KL Grading Dataset

Radiographer-informed EDA. Focuses on:
  - Class distribution and imbalance across splits
  - Visual inspection of images per KL grade
  - Exposure quality check (pixel-based EXI proxy)
  - Artifact detection (metal implants, overlays)

Note on exposure: True EXI (IEC 62494-1) requires DICOM metadata.
The Kaggle dataset provides JPEG only — DICOM tags are unavailable.
Mean pixel intensity is used as a proxy:
  - Mean < 0.20  -> likely underexposed
  - Mean > 0.75  -> likely overexposed
  - Max  > 0.98 in >5% of pixels -> potential metal artifact

Usage
-----
    python scripts/01_eda.py \
        --data_dir /rds/user/tm922/hpc-work/data/knee_oa \
        --output_dir results/eda \
        --n_exposure_sample 50
"""

import argparse
import sys
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from tqdm import tqdm

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

KL_COLOURS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4"]
SPLITS     = ["train", "val", "test"]
GRADES     = [0, 1, 2, 3, 4]
DPI        = 150

KL_LABELS = {
    0: "KL 0 - Normal\nNo features of OA",
    1: "KL 1 - Doubtful\nPossible osteophyte",
    2: "KL 2 - Mild\nDefinite osteophyte\nPossible JSN",
    3: "KL 3 - Moderate\nModerate JSN\nSclerosis",
    4: "KL 4 - Severe\nLarge osteophyte\nMarked JSN",
}

UNDEREXPOSED_THRESHOLD = 0.20
OVEREXPOSED_THRESHOLD  = 0.75
ARTIFACT_THRESHOLD     = 0.98


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.name}")


def count_images(data_dir):
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for grade in GRADES:
            grade_dir = data_dir / split / str(grade)
            counts[split][grade] = (
                len([f for f in grade_dir.iterdir() if f.is_file()])
                if grade_dir.exists() else 0
            )
    return counts


def get_image_paths(data_dir, split, grade):
    grade_dir = data_dir / split / str(grade)
    if not grade_dir.exists():
        return []
    return [f for f in sorted(grade_dir.iterdir()) if f.is_file()]


def load_gray(path):
    return np.array(
        Image.open(path).convert("L"), dtype=np.float32
    ) / 255.0


def plot_class_distribution(counts, out):
    print("\n[1/6] Class distribution - counts")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("KL Grade Distribution Across Splits", fontsize=14, y=1.02)
    for ax, split in zip(axes, SPLITS):
        grade_counts = [counts[split][g] for g in GRADES]
        bars = ax.bar([f"KL {g}" for g in GRADES], grade_counts,
                      color=KL_COLOURS, edgecolor="white", linewidth=0.8)
        for bar, count in zip(bars, grade_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(grade_counts) * 0.02,
                    str(count), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(f"{split.capitalize()} split  (n={sum(grade_counts):,})")
        ax.set_xlabel("KL Grade")
        ax.set_ylabel("Number of images")
        ax.set_ylim(0, max(grade_counts) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "01_class_distribution.png")


def plot_class_proportions(counts, out):
    print("\n[2/6] Class proportions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("KL Grade Proportions Across Splits (%)", fontsize=14, y=1.02)
    for ax, split in zip(axes, SPLITS):
        grade_counts = np.array([counts[split][g] for g in GRADES], dtype=float)
        total = grade_counts.sum()
        props = (grade_counts / total * 100) if total > 0 else grade_counts
        bars = ax.bar([f"KL {g}" for g in GRADES], props,
                      color=KL_COLOURS, edgecolor="white", linewidth=0.8)
        for bar, prop in zip(bars, props):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{prop:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(f"{split.capitalize()} split")
        ax.set_xlabel("KL Grade")
        ax.set_ylabel("Proportion (%)")
        ax.set_ylim(0, max(props) * 1.20)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "02_class_proportions.png")


def plot_imbalance(counts, out):
    print("\n[3/6] Imbalance ratio")
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Class Imbalance - Majority vs Minority Grade per Split\n"
        "(ratio > 3 considered significant in medical imaging)",
        fontsize=13, y=1.05)
    for ax, split in zip(axes, SPLITS):
        grade_counts = {g: counts[split][g] for g in GRADES}
        max_grade = max(grade_counts, key=grade_counts.get)
        min_grade = min((g for g in GRADES if grade_counts[g] > 0),
                        key=grade_counts.get)
        ratio = (grade_counts[max_grade] / grade_counts[min_grade]
                 if grade_counts[min_grade] > 0 else float("inf"))
        colours = ["#d65f5f" if g == max_grade
                   else "#6acc65" if g == min_grade
                   else "#aec6e8" for g in GRADES]
        ax.bar([f"KL {g}" for g in GRADES],
               [grade_counts[g] for g in GRADES],
               color=colours, edgecolor="white")
        ax.set_title(f"{split.capitalize()}\nRatio {ratio:.1f}:1  "
                     f"(KL{max_grade} / KL{min_grade})")
        ax.set_xlabel("KL Grade")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            Patch(facecolor="#d65f5f", label=f"Majority  KL{max_grade}"),
            Patch(facecolor="#6acc65", label=f"Minority  KL{min_grade}"),
        ], fontsize=8, loc="upper right")
    save(fig, out / "03_imbalance_ratio.png")


def plot_sample_images(data_dir, out, n_per_grade=4):
    print("\n[4/6] Sample images per KL grade")
    split    = "train"
    n_grades = len(GRADES)
    fig = plt.figure(figsize=(n_per_grade * 3.5 + 2, n_grades * 3.5 + 1))
    fig.suptitle(
        "Representative Knee Radiographs per KL Grade  (train split, seed=42)\n"
        "JSN = Joint Space Narrowing",
        fontsize=12, y=1.01)
    outer = gridspec.GridSpec(n_grades, 2,
                               width_ratios=[1, n_per_grade],
                               hspace=0.35, wspace=0.05)
    for row, grade in enumerate(GRADES):
        ax_label = fig.add_subplot(outer[row, 0])
        ax_label.text(0.5, 0.5, KL_LABELS[grade],
                      transform=ax_label.transAxes,
                      ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      color=KL_COLOURS[grade],
                      multialignment="center",
                      bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="#f7f7f7",
                                edgecolor=KL_COLOURS[grade],
                                linewidth=1.5))
        ax_label.axis("off")
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_per_grade, subplot_spec=outer[row, 1], wspace=0.04)
        paths = get_image_paths(data_dir, split, grade)
        random.seed(42)
        sample = random.sample(paths, min(n_per_grade, len(paths)))
        for col, img_path in enumerate(sample):
            ax = fig.add_subplot(inner[col])
            try:
                img = Image.open(img_path).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}",
                        transform=ax.transAxes,
                        ha="center", va="center", fontsize=7)
            ax.axis("off")
    save(fig, out / "04_sample_images.png")


def plot_exposure_quality(data_dir, out, n_sample=50):
    print(f"\n[5/6] Exposure quality check  (n={n_sample} per grade)")
    split        = "train"
    all_stats    = []
    underexposed = []
    overexposed  = []
    for grade in GRADES:
        paths = get_image_paths(data_dir, split, grade)
        random.seed(42)
        sample = random.sample(paths, min(n_sample, len(paths)))
        for p in tqdm(sample, desc=f"  KL {grade}", leave=False):
            try:
                arr   = load_gray(p)
                m     = float(arr.mean())
                s     = float(arr.std())
                all_stats.append({"grade": grade, "path": p,
                                  "mean": m, "std": s})
                if m < UNDEREXPOSED_THRESHOLD:
                    underexposed.append({"grade": grade, "path": p, "mean": m})
                elif m > OVEREXPOSED_THRESHOLD:
                    overexposed.append({"grade": grade, "path": p, "mean": m})
            except Exception:
                continue
    grade_means = {g: [] for g in GRADES}
    grade_stds  = {g: [] for g in GRADES}
    for s in all_stats:
        grade_means[s["grade"]].append(s["mean"])
        grade_stds[s["grade"]].append(s["std"])
    avg_means = [np.mean(grade_means[g]) if grade_means[g] else 0 for g in GRADES]
    avg_stds  = [np.mean(grade_stds[g])  if grade_stds[g]  else 0 for g in GRADES]
    n_under   = len(underexposed)
    n_over    = len(overexposed)
    print(f"  Underexposed flagged (mean < {UNDEREXPOSED_THRESHOLD}): {n_under}")
    print(f"  Overexposed  flagged (mean > {OVEREXPOSED_THRESHOLD}):  {n_over}")
    fig      = plt.figure(figsize=(15, 10))
    gs_main  = gridspec.GridSpec(2, 1, hspace=0.5)
    ax_top   = fig.add_subplot(gs_main[0])
    x        = np.arange(len(GRADES))
    bars     = ax_top.bar(x, avg_means, color=KL_COLOURS, edgecolor="white",
                          yerr=avg_stds, capsize=4)
    ax_top.axhline(UNDEREXPOSED_THRESHOLD, color="blue", linestyle="--",
                   linewidth=1.5,
                   label=f"Underexposed threshold ({UNDEREXPOSED_THRESHOLD})")
    ax_top.axhline(OVEREXPOSED_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.5,
                   label=f"Overexposed threshold ({OVEREXPOSED_THRESHOLD})")
    ax_top.fill_between([-0.5, len(GRADES) - 0.5], 0,
                         UNDEREXPOSED_THRESHOLD, alpha=0.08, color="blue")
    ax_top.fill_between([-0.5, len(GRADES) - 0.5], OVEREXPOSED_THRESHOLD,
                         1.0, alpha=0.08, color="red")
    for bar, val in zip(bars, avg_means):
        ax_top.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"KL {g}" for g in GRADES])
    ax_top.set_ylabel("Mean pixel intensity (0-1)")
    ax_top.set_ylim(0, 1.0)
    ax_top.set_title(
        f"Exposure Quality per KL Grade  ({n_sample} images per grade)\n"
        f"Proxy for EXI - true EXI unavailable (DICOM metadata stripped)\n"
        f"Flagged: {n_under} underexposed  |  {n_over} overexposed")
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.spines[["top", "right"]].set_visible(False)
    examples = []
    if underexposed:
        examples += sorted(underexposed, key=lambda x: x["mean"])[:3]
    if overexposed:
        examples += sorted(overexposed, key=lambda x: x["mean"],
                           reverse=True)[:3]
    if examples:
        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, len(examples), subplot_spec=gs_main[1], wspace=0.1)
        for i, ex in enumerate(examples):
            ax = fig.add_subplot(gs_bottom[i])
            try:
                img = Image.open(ex["path"]).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")
            except Exception:
                pass
            status = ("UNDEREXPOSED" if ex["mean"] < UNDEREXPOSED_THRESHOLD
                      else "OVEREXPOSED")
            colour = "blue" if status == "UNDEREXPOSED" else "red"
            ax.set_title(f"KL {ex['grade']}  [{status}]\nmean={ex['mean']:.3f}",
                         fontsize=9, color=colour, fontweight="bold")
            ax.axis("off")
    else:
        ax_b = fig.add_subplot(gs_main[1])
        ax_b.text(0.5, 0.5, "No exposure outliers detected.",
                  transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=12, color="green", fontweight="bold")
        ax_b.axis("off")
    save(fig, out / "05_exposure_quality.png")


def plot_artifact_detection(data_dir, out, n_sample=50):
    print(f"\n[6/6] Artifact detection  (n={n_sample} per grade)")
    split   = "train"
    flagged = []
    all_max = {g: [] for g in GRADES}
    for grade in GRADES:
        paths = get_image_paths(data_dir, split, grade)
        random.seed(42)
        sample = random.sample(paths, min(n_sample, len(paths)))
        for p in tqdm(sample, desc=f"  KL {grade}", leave=False):
            try:
                arr        = load_gray(p)
                max_val    = float(arr.max())
                bright_pct = float((arr > ARTIFACT_THRESHOLD).mean())
                all_max[grade].append(max_val)
                if bright_pct > 0.05:
                    flagged.append({"grade": grade, "path": p,
                                    "bright_pct": bright_pct,
                                    "max_val": max_val})
            except Exception:
                continue
    print(f"  Artifact candidates flagged: {len(flagged)}")
    fig     = plt.figure(figsize=(15, 10))
    gs_main = gridspec.GridSpec(2, 1, hspace=0.5)
    ax_top  = fig.add_subplot(gs_main[0])
    avg_max = [np.mean(all_max[g]) if all_max[g] else 0 for g in GRADES]
    bars    = ax_top.bar([f"KL {g}" for g in GRADES], avg_max,
                         color=KL_COLOURS, edgecolor="white")
    ax_top.axhline(ARTIFACT_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.5,
                   label=f"Artifact threshold (>{ARTIFACT_THRESHOLD} in >5% pixels)")
    for bar, val in zip(bars, avg_max):
        ax_top.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax_top.set_ylabel("Mean maximum pixel value (0-1)")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(
        f"Artifact Detection per KL Grade  ({n_sample} images per grade)\n"
        f"High max pixel values in >5% pixels may indicate metal implants "
        f"or text overlays\nFlagged candidates: {len(flagged)}")
    ax_top.legend(fontsize=9)
    ax_top.spines[["top", "right"]].set_visible(False)
    top_flagged = sorted(flagged, key=lambda x: x["bright_pct"],
                         reverse=True)[:6]
    if top_flagged:
        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, len(top_flagged), subplot_spec=gs_main[1], wspace=0.1)
        for i, ex in enumerate(top_flagged):
            ax = fig.add_subplot(gs_bottom[i])
            try:
                img = Image.open(ex["path"]).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")
            except Exception:
                pass
            ax.set_title(
                f"KL {ex['grade']}\nbright={ex['bright_pct']*100:.1f}%\n"
                f"max={ex['max_val']:.3f}",
                fontsize=8, color="red", fontweight="bold")
            ax.axis("off")
    else:
        ax_b = fig.add_subplot(gs_main[1])
        ax_b.text(0.5, 0.5, "No artifact candidates detected.",
                  transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=12, color="green", fontweight="bold")
        ax_b.axis("off")
    save(fig, out / "06_artifact_detection.png")


def parse_args():
    p = argparse.ArgumentParser(
        description="Radiographer-informed EDA for Knee OA KL grading dataset"
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/eda")
    p.add_argument("--n_exposure_sample", type=int, default=50)
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)

    print("=" * 60)
    print("  Knee OA KL Grading - EDA")
    print("=" * 60)
    print(f"  Data dir        : {data_dir}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Exposure sample : {args.n_exposure_sample} per grade")

    if not data_dir.exists():
        print(f"\n  ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    for split in SPLITS:
        if not (data_dir / split).exists():
            print(f"\n  ERROR: missing split: {data_dir / split}")
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Counting images...")
    counts = count_images(data_dir)
    for split in SPLITS:
        total     = sum(counts[split].values())
        per_grade = "  ".join(f"KL{g}:{counts[split][g]}" for g in GRADES)
        print(f"  {split:6s}: {total:,} total  |  {per_grade}")

    plot_class_distribution(counts, out_dir)
    plot_class_proportions(counts, out_dir)
    plot_imbalance(counts, out_dir)
    plot_sample_images(data_dir, out_dir)
    plot_exposure_quality(data_dir, out_dir, n_sample=args.n_exposure_sample)
    plot_artifact_detection(data_dir, out_dir, n_sample=args.n_exposure_sample)

    print("\n" + "=" * 60)
    print(f"  Done. Plots saved to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
