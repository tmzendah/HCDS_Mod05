"""Exploratory data analysis for the MRKR knee radiograph dataset.

Produces
--------
eda/
  summary.json           -- high-level dataset stats
  label_distribution.png -- KL grade counts + proportions
  metadata_heatmap.png   -- missing-value heatmap
  acq_laterality.png     -- acquisition type × laterality × KL grade
  image_stats.csv        -- per-image: resolution, aspect ratio, file size
  image_resolution.png   -- scatter of width vs height
  pixel_stats.png        -- per-class mean / std of pixel intensity
  grade1_focus.png       -- grade 0 vs 1 vs 2 breakdown across metadata strata
  correlation.png        -- encoded metadata correlation matrix

Usage
-----
    python mrkr_eda.py \
        --data_csv data/metadata.csv \
        --img_root data/images \
        --output_dir eda \
        --n_pixel_sample 200   # images to sample for pixel stats (slow step)

Notes
-----
- Script is read-only: no data is modified.
- Pixel stats are sampled to avoid long runtimes on CPU.
- All plots saved as PNG at 150 Dpi (readable, not huge).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on HPC login nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

DPI = 150


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def resolve_path(rel: str, root: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ──────────────────────────────────────────────
# 1. CSV summary
# ──────────────────────────────────────────────

def summarise_csv(df: pd.DataFrame, out: Path) -> dict:
    print("\n[1/7] CSV summary")

    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_image_paths": int(df["image_path"].duplicated().sum())
        if "image_path" in df.columns else "N/A",
    }

    if "label" in df.columns:
        vc = df["label"].value_counts().sort_index()
        summary["label_counts"] = {str(k): int(v) for k, v in vc.items()}
        summary["label_proportions"] = {
            str(k): round(float(v) / len(df), 4) for k, v in vc.items()
        }
        summary["n_classes"] = int(df["label"].nunique())
        summary["imbalance_ratio"] = round(
            float(vc.max()) / float(vc.min()) if vc.min() > 0 else float("inf"), 2
        )

    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"  rows          : {summary['n_rows']}")
    print(f"  columns       : {summary['columns']}")
    print(f"  missing values: {summary['missing_per_column']}")
    print(f"  duplicate rows: {summary['duplicate_rows']}")
    if "label_counts" in summary:
        print(f"  KL distribution: {summary['label_counts']}")
        print(f"  imbalance ratio (max/min class): {summary['imbalance_ratio']}")
    return summary


# ──────────────────────────────────────────────
# 2. Label distribution
# ──────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, out: Path) -> None:
    print("\n[2/7] Label distribution")

    if "label" not in df.columns:
        print("  [SKIP] no 'label' column")
        return

    vc = df["label"].value_counts().sort_index()
    labels = [str(x) for x in vc.index]
    counts = vc.values
    props  = counts / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Counts
    bars = axes[0].bar(labels, counts, color=sns.color_palette("muted", len(labels)))
    axes[0].set_xlabel("KL Grade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("KL Grade — sample counts")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + counts.max() * 0.01,
                     str(count), ha="center", va="bottom", fontsize=10)

    # Proportions
    axes[1].bar(labels, props * 100, color=sns.color_palette("muted", len(labels)))
    axes[1].set_xlabel("KL Grade")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("KL Grade — class proportions")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    save_fig(fig, out / "label_distribution.png")


# ──────────────────────────────────────────────
# 3. Missing value heatmap
# ──────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame, out: Path) -> None:
    print("\n[3/7] Missing value heatmap")

    miss = df.isnull()
    if not miss.any().any():
        print("  No missing values found — skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns)), 4))
    sns.heatmap(
        miss.T, cbar=False, yticklabels=True, xticklabels=False,
        cmap=["#d9f0d3", "#d73027"], ax=ax
    )
    ax.set_title("Missing value map (red = missing)")
    ax.set_xlabel("Samples")
    save_fig(fig, out / "metadata_heatmap.png")


# ──────────────────────────────────────────────
# 4. Metadata cross-tabs (laterality, acq type)
# ──────────────────────────────────────────────

CANDIDATE_META_COLS = [
    # common MRKR column names — script will use whichever exist
    "laterality", "side", "acquisition_type", "view", "position",
    "weightbearing", "ap_view", "study_type",
]


def detect_meta_columns(df: pd.DataFrame) -> List[str]:
    """Return columns likely to be clinically meaningful metadata."""
    found = [c for c in CANDIDATE_META_COLS if c in df.columns]
    # Also include any object/category column that isn't image_path or label
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if c not in ("image_path", "label") and c not in found:
            n_unique = df[c].nunique()
            if 2 <= n_unique <= 20:   # looks like a categorical variable
                found.append(c)
    return found


def plot_metadata_crosstabs(df: pd.DataFrame, out: Path) -> None:
    print("\n[4/7] Metadata cross-tabs")

    if "label" not in df.columns:
        print("  [SKIP] no 'label' column")
        return

    meta_cols = detect_meta_columns(df)
    if not meta_cols:
        print("  No metadata columns detected beyond image_path/label.")
        return

    print(f"  Detected metadata columns: {meta_cols}")

    # One subplot per metadata column
    n = len(meta_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, col in zip(axes[0], meta_cols):
        ct = pd.crosstab(df[col], df["label"])
        ct.plot(kind="bar", stacked=True, ax=ax,
                colormap="tab10", legend=(col == meta_cols[-1]))
        ax.set_title(f"{col} × KL grade")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    save_fig(fig, out / "acq_laterality.png")

    # Grade 1 focus: for each metadata col, what proportion of grade-1
    # examples come from each category?
    print("\n  Grade-1 breakdown by metadata column:")
    g1 = df[df["label"].astype(str) == "1"] if "1" in df["label"].astype(str).values \
        else df[df["label"] == 1]

    if len(g1) == 0:
        print("  No grade-1 samples found — check label encoding.")
        return

    grade1_stats = {}
    for col in meta_cols:
        vc = g1[col].value_counts(dropna=False)
        grade1_stats[col] = {str(k): int(v) for k, v in vc.items()}
        print(f"    {col}: {grade1_stats[col]}")

    with (out / "grade1_metadata_breakdown.json").open("w") as f:
        json.dump(grade1_stats, f, indent=2)

    # Grade 1 focus plot
    if len(meta_cols) >= 2:
        _plot_grade1_focus(df, meta_cols, out)


def _plot_grade1_focus(df: pd.DataFrame, meta_cols: List[str], out: Path) -> None:
    """Stacked bar: grade 0 / 1 / 2 counts per metadata stratum."""
    col = meta_cols[0]   # use the first detected column (likely laterality)
    sub = df[df["label"].isin([0, 1, 2])].copy()
    if len(sub) == 0:
        return

    ct = pd.crosstab(sub[col], sub["label"])
    ct.columns = [f"KL {c}" for c in ct.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=False, ax=ax, colormap="Set2")
    ax.set_title(f"KL grades 0–2 by {col}\n(focus on hard boundary region)")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, out / "grade1_focus.png")


# ──────────────────────────────────────────────
# 5. Image file audit
# ──────────────────────────────────────────────

def audit_images(df: pd.DataFrame, img_root: str, out: Path) -> pd.DataFrame:
    print("\n[5/7] Image file audit")

    root = Path(img_root)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="auditing images"):
        rel  = row.get("image_path", "")
        path = resolve_path(str(rel), root)
        rec  = {"image_path": str(rel), "exists": path.exists()}

        if rec["exists"]:
            try:
                rec["file_size_kb"] = round(path.stat().st_size / 1024, 1)
                with Image.open(path) as img:
                    rec["width"]  = img.width
                    rec["height"] = img.height
                    rec["mode"]   = img.mode
                    rec["format"] = img.format
            except Exception as e:
                rec["error"] = str(e)

        if "label" in row:
            rec["label"] = row["label"]

        records.append(rec)

    stats_df = pd.DataFrame(records)
    stats_df.to_csv(out / "image_stats.csv", index=False)

    n_missing = (~stats_df["exists"]).sum()
    print(f"  Total images  : {len(stats_df)}")
    print(f"  Missing files : {n_missing}")
    if n_missing > 0:
        missing_paths = stats_df.loc[~stats_df["exists"], "image_path"].tolist()
        print(f"  First 5 missing: {missing_paths[:5]}")

    if "error" in stats_df.columns:
        n_err = stats_df["error"].notna().sum()
        if n_err:
            print(f"  Corrupt/unreadable files: {n_err}")

    return stats_df


# ──────────────────────────────────────────────
# 6. Image resolution distribution
# ──────────────────────────────────────────────

def plot_resolutions(stats_df: pd.DataFrame, out: Path) -> None:
    print("\n[6/7] Image resolution distribution")

    if "width" not in stats_df.columns:
        print("  [SKIP] no resolution data (image audit may have failed)")
        return

    sub = stats_df.dropna(subset=["width", "height"]).copy()
    sub["aspect_ratio"] = sub["width"] / sub["height"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Width vs height scatter
    axes[0].scatter(sub["width"], sub["height"], alpha=0.3, s=10,
                    c=sub.get("label", pd.Series([0]*len(sub))),
                    cmap="tab10")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image dimensions")

    # Aspect ratio histogram
    axes[1].hist(sub["aspect_ratio"], bins=40, color="#4878d0", edgecolor="white")
    axes[1].axvline(1.0, color="red", linestyle="--", label="square")
    axes[1].set_xlabel("Aspect ratio (w/h)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Aspect ratio distribution")
    axes[1].legend()

    # File size histogram
    if "file_size_kb" in sub.columns:
        axes[2].hist(sub["file_size_kb"], bins=40, color="#ee854a", edgecolor="white")
        axes[2].set_xlabel("File size (KB)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("File size distribution")

    save_fig(fig, out / "image_resolution.png")

    print(f"  Width  : min={sub['width'].min()}, max={sub['width'].max()}, "
          f"median={sub['width'].median():.0f}")
    print(f"  Height : min={sub['height'].min()}, max={sub['height'].max()}, "
          f"median={sub['height'].median():.0f}")
    print(f"  Aspect : min={sub['aspect_ratio'].min():.2f}, "
          f"max={sub['aspect_ratio'].max():.2f}, "
          f"median={sub['aspect_ratio'].median():.2f}")
    print(f"  Modes  : {sub['mode'].value_counts().to_dict() if 'mode' in sub.columns else 'N/A'}")


# ──────────────────────────────────────────────
# 7. Pixel statistics (sampled)
# ──────────────────────────────────────────────

def compute_pixel_stats(
    df: pd.DataFrame, img_root: str, out: Path, n_sample: int = 200
) -> None:
    print(f"\n[7/7] Pixel statistics (sampling up to {n_sample} images per class)")

    if "label" not in df.columns or "image_path" not in df.columns:
        print("  [SKIP]")
        return

    root   = Path(img_root)
    labels = sorted(df["label"].unique())
    results = {}

    for lbl in labels:
        sub = df[df["label"] == lbl]
        sample = sub.sample(min(n_sample, len(sub)), random_state=42)
        means, stds = [], []

        for _, row in tqdm(sample.iterrows(), total=len(sample),
                           desc=f"  KL {lbl}", leave=False):
            p = resolve_path(str(row["image_path"]), root)
            if not p.exists():
                continue
            try:
                img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                means.append(float(img.mean()))
                stds.append(float(img.std()))
            except Exception:
                continue

        if means:
            results[str(lbl)] = {
                "mean_pixel": round(np.mean(means), 4),
                "std_pixel":  round(np.mean(stds),  4),
                "n_sampled":  len(means),
            }

    with (out / "pixel_stats.json").open("w") as f:
        json.dump(results, f, indent=2)

    if not results:
        print("  No pixel stats computed — check image paths.")
        return

    klasses  = list(results.keys())
    means    = [results[k]["mean_pixel"] for k in klasses]
    stds_val = [results[k]["std_pixel"]  for k in klasses]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(klasses))
    ax.bar(x - 0.2, means, width=0.35, label="Mean pixel intensity", color="#4878d0")
    ax.bar(x + 0.2, stds_val, width=0.35, label="Std pixel intensity", color="#ee854a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"KL {k}" for k in klasses])
    ax.set_ylabel("Pixel value (normalised 0–1)")
    ax.set_title("Per-class pixel intensity (grayscale, sampled)")
    ax.legend()
    save_fig(fig, out / "pixel_stats.png")

    print("\n  Per-class pixel stats:")
    for k, v in results.items():
        print(f"    KL {k}: mean={v['mean_pixel']:.4f}  std={v['std_pixel']:.4f}  "
              f"(n={v['n_sampled']})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",       type=str, required=True)
    p.add_argument("--img_root",       type=str, required=True)
    p.add_argument("--output_dir",     type=str, default="eda")
    p.add_argument("--n_pixel_sample", type=int, default=200,
                   help="Images per class to sample for pixel stats (default 200)")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    out   = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MRKR — Exploratory Data Analysis")
    print("=" * 60)

    df = pd.read_csv(args.data_csv)

    summarise_csv(df, out)
    plot_label_distribution(df, out)
    plot_missing_heatmap(df, out)
    plot_metadata_crosstabs(df, out)
    stats_df = audit_images(df, args.img_root, out)
    plot_resolutions(stats_df, out)
    compute_pixel_stats(df, args.img_root, out, args.n_pixel_sample)

    print("\n" + "=" * 60)
    print(f"EDA complete. All outputs in: {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
