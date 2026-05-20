#!/usr/bin/env python3
"""
02_filter_by_quality.py - Annotate all images with quality metrics (no exclusion by default).

Runs the full quality assessment pipeline on ALL images in the dataset,
adds quality flag columns to the metadata CSV, and generates diagnostic
plots. **No images are excluded** — the flags are added for analysis and
discussion, not for filtering.

Use --exclude True to create a clean subset CSV if desired.

Quality assessments are run on **raw** images (before CLAHE) so that
acquisition-level artifacts are detected rather than hidden by enhancement.

Input:
  /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv
  /rds/user/tm922/hpc-work/data/mrkr_png_v2/images/

Output (saved to ~/mrkr_klg/filtered/):
  - full_quality_annotated.csv      (original CSV with all quality columns + flags)
  - quality_report.txt              (detailed text summary)
  - filtering_sunburst.png          (how many images trigger each flag)
  - filter_impact_by_kl.png         (flag rates per KL grade)
  - failed_examples_grid.png        (grid of flagged images with reasons)
  - clean_metadata.csv              (only if --exclude True; subset with no flags)

Usage:
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/02_filter_by_quality.py                          # annotate only (default)
  python code/02_filter_by_quality.py --exclude True           # also create clean subset
  python code/02_filter_by_quality.py --blur_threshold 150     # custom thresholds
  python code/02_filter_by_quality.py --max_images 5000        # limit for testing
  python code/02_filter_by_quality.py --num_workers 8          # parallel processing
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import quality_utils as qu

# =============================================================================
# Paths
# =============================================================================
RDS_BASE = "/rds/user/tm922/hpc-work"
INPUT_CSV = os.path.join(RDS_BASE, "data", "mrkr", "mrkr_selected_v2.csv")
IMAGE_DIR = os.path.join(RDS_BASE, "data", "mrkr_png_v2", "images")

PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
FILTERED_DIR = os.path.join(PROJECT_DIR, "filtered")

os.makedirs(FILTERED_DIR, exist_ok=True)


# =============================================================================
# Single-image processing (designed for parallel mapping)
# =============================================================================

def process_one_image(args):
    """Load an image, compute quality metrics, return a dict with results.

    This is a standalone function (not a method) so it can be pickled for
    parallel processing with ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (img_path, row_dict, blur_threshold, rotation_threshold,
         collimation_threshold, min_brightness, max_brightness)

    Returns
    -------
    dict or None
        Dictionary with all quality info plus row data, or None if image
        file was not found.
    """
    (img_path, row, blur_thr, rot_thr, coll_thr,
     min_bright, max_bright) = args

    if not os.path.exists(img_path):
        return None

    try:
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
    except Exception:
        return None

    metrics = qu.compute_quality_metrics(
        img_array,
        blur_threshold=blur_thr,
        rotation_threshold=rot_thr,
        collimation_threshold=coll_thr,
        min_brightness=min_bright,
        max_brightness=max_bright,
    )

    # Determine pass/fail
    metrics["quality_pass"] = qu.compute_overall_pass(metrics)

    # Attach original row identifiers
    result = {
        "png_path": row.get("png_path", ""),
        "empi_anon": row.get("empi_anon", ""),
        "knee_side": row.get("knee_side", ""),
        "horizontal_flip": row.get("horizontal_flip", ""),
        "kl_grade": row.get("kl_grade", ""),
    }
    result.update(metrics)
    return result


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_filtering_failure_breakdown(results_df, save_dir):
    """Horizontal bar chart showing how many images fail each quality check."""
    flag_cols = ["blurred", "poor_collimation", "possible_rotation",
                 "underexposed", "overexposed", "anatomical_truncated"]
    # Only use flags that actually exist in the DataFrame
    flag_cols = [c for c in flag_cols if c in results_df.columns]

    if not flag_cols:
        return

    failed = results_df[~results_df["quality_pass"]]
    n_total = len(results_df)
    n_failed = len(failed)
    n_passed = n_total - n_failed

    # Count how many images fail each check
    fail_counts = {}
    for col in flag_cols:
        cnt = failed[col].sum()
        if cnt > 0:
            # Human-readable label
            label = col.replace("_", " ").title()
            fail_counts[label] = int(cnt)

    if not fail_counts:
        print("  No failures to plot (all images passed).")
        return

    # Simple horizontal bar chart showing failure reasons
    labels = list(fail_counts.keys())
    counts = list(fail_counts.values())
    sorted_idx = np.argsort(counts)[::-1]
    labels = [labels[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4 + 1)))
    bars = ax.barh(labels, counts, color="coral", edgecolor="black")
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / n_failed if n_failed > 0 else 0
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of images flagged")
    ax.set_title(f"Images Flagged by Quality Checks (n={n_failed} flagged out of {n_total})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "filtering_sunburst.png"), dpi=150)
    plt.close()
    print(f"  Saved filtering_sunburst.png")


def plot_filter_impact_by_kl(orig_kl_counts, passed_kl_counts, save_dir):
    """Compare class distribution before and after filtering."""
    grades = sorted(set(list(orig_kl_counts.keys()) + list(passed_kl_counts.keys())))
    x = np.arange(len(grades))
    width = 0.35

    orig_vals = [orig_kl_counts.get(g, 0) for g in grades]
    passed_vals = [passed_kl_counts.get(g, 0) for g in grades]
    drop_pcts = [(o - p) / o * 100 if o > 0 else 0 for o, p in zip(orig_vals, passed_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart: before vs after
    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, orig_vals, width, label="Before filtering", color="steelblue")
    bars2 = ax1.bar(x + width / 2, passed_vals, width, label="After filtering", color="seagreen")
    ax1.set_xlabel("KL Grade")
    ax1.set_ylabel("Count")
    ax1.set_title("Images With All Flags Clear vs Any Flag Set")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"KL{g}" for g in grades])
    ax1.legend()

    # Annotate counts on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(orig_vals) * 0.01,
                 str(int(bar.get_height())), ha="center", fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(orig_vals) * 0.01,
                 str(int(bar.get_height())), ha="center", fontsize=8)

    # Drop rate per KL
    ax2 = axes[1]
    colors = ["crimson" if p > 10 else ("orange" if p > 5 else "seagreen") for p in drop_pcts]
    ax2.bar(x, drop_pcts, color=colors, edgecolor="black")
    ax2.axhline(y=10, color="crimson", linestyle="--", alpha=0.5, label="10% threshold")
    ax2.set_xlabel("KL Grade")
    ax2.set_ylabel("Drop rate (%)")
    ax2.set_title("Percentage of Images Flagged per KL Grade")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"KL{g}" for g in grades])
    ax2.legend()

    for xi, pct in zip(x, drop_pcts):
        ax2.text(xi, pct + 0.5, f"{pct:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "filter_impact_by_kl.png"), dpi=150)
    plt.close()
    print(f"  Saved filter_impact_by_kl.png")


def plot_failed_examples(results_df, image_dir, save_dir, n_examples=12):
    """Show a grid of example images that failed quality checks, labeled with why."""
    failed = results_df[~results_df["quality_pass"]]
    if len(failed) == 0:
        print("  No failed images to preview.")
        return

    failed_sample = failed.sample(n=min(n_examples, len(failed)), random_state=42)
    n = len(failed_sample)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (idx, row) in enumerate(failed_sample.iterrows()):
        img_path = os.path.join(image_dir, row["png_path"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('L')
            axes[i].imshow(img, cmap="gray", aspect="auto")
        else:
            axes[i].text(0.5, 0.5, "Missing", ha="center", va="center")

        # Build failure reason string
        reasons = []
        if row.get("blurred", False):
            reasons.append(f"Blur({row['sharpness_laplacian_var']:.0f})")
        if row.get("poor_collimation", False):
            reasons.append(f"Collim({row['collimation_score']:.2f})")
        if row.get("possible_rotation", False):
            reasons.append(f"Rotate({row['rotation_score']:.2f})")
        if row.get("underexposed", False):
            reasons.append("Underexp")
        if row.get("overexposed", False):
            reasons.append("Overexp")
        if row.get("anatomical_truncated", False):
            reasons.append("Truncated")

        kl = row.get("kl_grade", "?")
        axes[i].set_title(f"KL{kl} | {' '.join(reasons)}", fontsize=7)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Examples of Images Flagged by Quality Filters", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "failed_examples_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved failed_examples_grid.png ({len(failed_sample)} examples)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Annotate knee X-ray images with quality metrics for analysis and discussion."
    )
    # Thresholds
    parser.add_argument("--blur_threshold", type=float, default=100,
                        help="Laplacian variance below this = blurry (default: 100)")
    parser.add_argument("--rotation_threshold", type=float, default=0.1,
                        help="Rotation score above this = rotated (default: 0.1)")
    parser.add_argument("--collimation_threshold", type=float, default=0.6,
                        help="Collimation score above this = poor collimation (default: 0.6)")
    parser.add_argument("--min_brightness", type=float, default=30,
                        help="Mean brightness below this = underexposed (default: 30)")
    parser.add_argument("--max_brightness", type=float, default=220,
                        help="Mean brightness above this = overexposed (default: 220)")
    parser.add_argument("--exclude", type=bool, default=False,
                        help="If True, create clean_metadata.csv excluding flagged images (default: False)")
    parser.add_argument("--max_fail_flags", type=int, default=0,
                        help="Max flags tolerated when --exclude True (0 = any flag = fail, -1 = all must pass)")

    # Runtime
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit processing to this many images (for testing)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    print("=" * 60)
    print("02_filter_by_quality.py - Image quality annotation (no exclusion by default)")
    print("=" * 60)
    ver = qu.cv2_version()
    if ver:
        print(f"OpenCV version: {ver}")
    else:
        print("WARNING: OpenCV not available — blur, collimation, and "
              "sharpness metrics will be NaN and those checks skipped.")

    print(f"\nThresholds:")
    print(f"  Blur threshold:           {args.blur_threshold}")
    print(f"  Rotation threshold:       {args.rotation_threshold}")
    print(f"  Collimation threshold:    {args.collimation_threshold}")
    print(f"  Min brightness:           {args.min_brightness}")
    print(f"  Max brightness:           {args.max_brightness}")
    print(f"  Exclude flagged images:   {args.exclude}")
    if args.exclude:
        print(f"  Max fail flags allowed:   {args.max_fail_flags}")

    # 1. Load metadata
    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} rows from {INPUT_CSV}")

    # Validate columns
    required = ["png_path", "kl_grade", "empi_anon", "knee_side", "horizontal_flip"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Optionally limit for testing
    if args.max_images is not None and args.max_images < len(df):
        df = df.sample(n=args.max_images, random_state=args.seed)
        print(f"  Subsampled to {len(df)} images for testing (--max_images).")

    # 2. Process all images
    print(f"\nProcessing {len(df)} images...")
    t_start = datetime.now()

    # Prepare arguments for each image
    process_args = []
    for idx, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["png_path"])
        process_args.append((
            img_path, row.to_dict(),
            args.blur_threshold, args.rotation_threshold,
            args.collimation_threshold, args.min_brightness,
            args.max_brightness,
        ))

    results = []
    if args.num_workers and args.num_workers > 1:
        print(f"  Using {args.num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one_image, pa): pa for pa in process_args}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                res = future.result()
                if res is not None:
                    results.append(res)
                if done_count % 500 == 0:
                    elapsed = (datetime.now() - t_start).total_seconds()
                    print(f"    {done_count}/{len(process_args)} done ({elapsed:.0f}s)")
    else:
        print("  Processing sequentially (use --num_workers N for parallel)...")
        for i, pa in enumerate(process_args):
            res = process_one_image(pa)
            if res is not None:
                results.append(res)
            if (i + 1) % 500 == 0:
                elapsed = (datetime.now() - t_start).total_seconds()
                print(f"    {i + 1}/{len(process_args)} done ({elapsed:.0f}s)")

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"  Completed {len(results)} images in {elapsed:.1f}s "
          f"({len(results) / max(elapsed, 0.1):.0f} img/s)")

    if not results:
        print("ERROR: No images were successfully processed. Check IMAGE_DIR path.")
        sys.exit(1)

    # 3. Build results DataFrame
    results_df = pd.DataFrame(results)

    # Ensure KL grade is int
    results_df["kl_grade"] = results_df["kl_grade"].astype(int)

    # 4. Summary statistics
    n_total = len(results_df)
    n_passed = results_df["quality_pass"].sum()
    n_failed = n_total - n_passed
    pass_rate = 100 * n_passed / n_total
    fail_rate = 100 * n_failed / n_total

    # Define flag columns once (used in both console output and report)
    _flag_col_names = ["blurred", "poor_collimation", "possible_rotation",
                       "underexposed", "overexposed", "anatomical_truncated"]
    available_flags = [c for c in _flag_col_names if c in results_df.columns]

    print(f"\n{'=' * 60}")
    print(f"FILTER SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total images processed:  {n_total}")
    print(f"  Passed quality check:    {n_passed} ({pass_rate:.1f}%)")
    print(f"  Failed quality check:    {n_failed} ({fail_rate:.1f}%)")

    if n_failed > 0:
        print(f"\n  Failure breakdown:")
        for col in available_flags:
            cnt = results_df[col].sum()
            pct = 100 * cnt / n_failed
            if cnt > 0:
                print(f"    {col:30s}: {int(cnt):5d} ({pct:5.1f}% of failed)")

    # 5. Per-KL breakdown
    print(f"\n  Per-KL grade impact:")
    orig_kl_counts = results_df["kl_grade"].value_counts().sort_index()
    passed_df = results_df[results_df["quality_pass"]]
    passed_kl_counts = passed_df["kl_grade"].value_counts().sort_index()
    for g in sorted(set(list(orig_kl_counts.keys()) + list(passed_kl_counts.keys()))):
        orig_n = orig_kl_counts.get(g, 0)
        passed_n = passed_kl_counts.get(g, 0)
        drop_n = orig_n - passed_n
        drop_pct = 100 * drop_n / orig_n if orig_n > 0 else 0
        print(f"    KL{g}: {passed_n}/{orig_n} passed ({drop_pct:.1f}% dropped)")

    # 6. Save annotated CSV (all images with quality columns)
    annot_csv_path = os.path.join(FILTERED_DIR, "full_quality_annotated.csv")
    results_df.to_csv(annot_csv_path, index=False)
    print(f"\n  Saved annotated CSV: {annot_csv_path}")

    # 7. Optionally save clean subset CSV (only if --exclude True)
    if args.exclude:
        clean_csv_path = os.path.join(FILTERED_DIR, "clean_metadata.csv")
        passed_df.to_csv(clean_csv_path, index=False)
        print(f"  Saved clean CSV (excluded {n_failed} flagged images): {clean_csv_path}")
    else:
        print(f"  Note: --exclude not set — all {n_total} images kept. Quality flags are in the annotated CSV.")

    # 8. Generate plots
    print(f"\n  Generating diagnostic plots...")
    plot_filtering_failure_breakdown(results_df, FILTERED_DIR)
    plot_filter_impact_by_kl(orig_kl_counts.to_dict(), passed_kl_counts.to_dict(), FILTERED_DIR)
    if n_failed > 0:
        plot_failed_examples(results_df, IMAGE_DIR, FILTERED_DIR, n_examples=12)

    # 9. Write filter report
    report_path = os.path.join(FILTERED_DIR, "quality_report.txt")
    with open(report_path, "w") as f:
        f.write("MRKR Dataset Quality Annotation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Thresholds:\n")
        f.write(f"  Blur threshold (Laplacian var <):       {args.blur_threshold}\n")
        f.write(f"  Rotation threshold (>):                 {args.rotation_threshold}\n")
        f.write(f"  Collimation threshold (>):              {args.collimation_threshold}\n")
        f.write(f"  Min brightness:                         {args.min_brightness}\n")
        f.write(f"  Max brightness:                         {args.max_brightness}\n")
        f.write(f"  Max fail flags tolerated:               {args.max_fail_flags}\n\n")

        f.write(f"Total images processed: {n_total}\n")
        f.write(f"Passed (all flags clear): {n_passed} ({pass_rate:.1f}%)\n")
        f.write(f"Failed (any flag set):    {n_failed} ({fail_rate:.1f}%)\n\n")

        if n_failed > 0:
            f.write("Flag breakdown (among failed images):\n")
            for col in available_flags:
                cnt = results_df[col].sum()
                pct = 100 * cnt / n_failed
                if cnt > 0:
                    f.write(f"  {col:30s}: {int(cnt):5d} ({pct:5.1f}% of failed)\n")
            f.write("\n")

        f.write("Per-KL breakdown:\n")
        for g in sorted(set(list(orig_kl_counts.keys()) + list(passed_kl_counts.keys()))):
            orig_n = orig_kl_counts.get(g, 0)
            passed_n = passed_kl_counts.get(g, 0)
            drop_n = orig_n - passed_n
            drop_pct = 100 * drop_n / orig_n if orig_n > 0 else 0
            f.write(f"  KL{g}: {passed_n}/{orig_n} passed ({drop_pct:.1f}% dropped)\n")

        f.write(f"\nFull annotated CSV: {annot_csv_path}\n")
        if args.exclude:
            f.write(f"Clean CSV (excluded): {clean_csv_path}\n")
        f.write(f"OpenCV available: {qu.cv2_available()}\n")
        if not qu.cv2_available():
            f.write("  NOTE: Blur and collimation metrics were not computed.\n")
            f.write("  Install opencv-python for full quality assessment.\n")

    print(f"  Saved report:        {report_path}")
    print(f"\n{'=' * 60}")
    print("Annotation complete.")
    print(f"{'=' * 60}")
    print(f"Next step: use the annotated CSV for your CLAHE preprocessing pipeline.")
    if args.exclude:
        print(f"  Clean (excluded) CSV: {clean_csv_path}")
    print(f"  Review {os.path.join(FILTERED_DIR, 'failed_examples_grid.png')} to inspect flagged images.")


if __name__ == "__main__":
    main()
