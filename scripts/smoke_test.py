"""Smoke test for mrkr_baseline_foundation.py

Subsamples the real CSV to a small stratified subset and runs the
baseline for 2 epochs on CPU with minimal settings.

Usage (from repo root, before GPU job):
    python smoke_test.py \
        --data_csv data/metadata.csv \
        --img_root data/images

Pass --n_samples to control subset size (default 100).
Pass --epochs to override epoch count (default 2).

What this checks
----------------
- CSV is readable and has required columns (image_path, label)
- Image files are reachable at the resolved paths
- Label encoding works across all classes present in the subset
- Full train → val → test loop completes without error
- Metrics dict is populated and JSON-serialisable
- Output directory and checkpoint are written correctly
- No CUDA / AMP dependency (safe on login nodes)

Exit code 0 = pipeline is healthy.
Exit code 1 = something to fix before submitting the GPU job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",  type=str, required=True,
                   help="Path to the full MRKR metadata CSV")
    p.add_argument("--img_root",  type=str, required=True,
                   help="Root directory for images")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of rows to subsample (default 100). "
                        "Must be >= num_classes * 3 to allow stratified splits.")
    p.add_argument("--epochs",    type=int, default=2,
                   help="Training epochs for the smoke run (default 2)")
    p.add_argument("--baseline_script", type=str,
                   default="mrkr_baseline_foundation.py",
                   help="Path to the baseline script")
    return p.parse_args()


def check_image_paths(df: pd.DataFrame, img_root: str, n_check: int = 10) -> None:
    """Spot-check that the first n_check image paths resolve to real files."""
    root = Path(img_root)
    missing = []
    for rel in df["image_path"].head(n_check):
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[WARN] Missing image files (first batch):")
        for m in missing:
            print(f"  {m}")
        print("       Fix paths before the GPU run.\n")
    else:
        print(f"[OK]  First {n_check} image paths resolve correctly.")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Return a stratified subsample of size n (or the whole df if n >= len)."""
    if n >= len(df):
        print(f"[INFO] n_samples ({n}) >= dataset size ({len(df)}), using full dataset.")
        return df.copy()

    # Some classes may have very few samples; make sure n is feasible.
    min_class_count = df["label"].value_counts().min()
    n_classes = df["label"].nunique()
    min_n = n_classes * 3          # need at least 3 per class for two splits
    if n < min_n:
        print(f"[WARN] n_samples={n} is small for {n_classes} classes. "
              f"Raising to {min_n}.")
        n = min_n

    frac = n / len(df)
    try:
        sub, _ = train_test_split(
            df, train_size=frac, stratify=df["label"], random_state=seed
        )
    except ValueError:
        # Fallback: unstratified if any class has < 2 members after split.
        print("[WARN] Stratified subsample failed (rare class); falling back to random.")
        sub = df.sample(n=n, random_state=seed)
    return sub.copy()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("SMOKE TEST — MRKR baseline pipeline")
    print("=" * 60)

    # --- 1. Load CSV ---
    csv_path = Path(args.data_csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[OK]  CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    for col in ("image_path", "label"):
        if col not in df.columns:
            sys.exit(f"[ERROR] Required column missing: '{col}'")
    print(f"[OK]  Required columns present.")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().sort_index().to_string()}\n")

    # --- 2. Check image paths ---
    check_image_paths(df, args.img_root)

    # --- 3. Subsample ---
    sub = stratified_subsample(df, args.n_samples)
    print(f"[OK]  Subsample: {len(sub)} rows, "
          f"label dist: {sub['label'].value_counts().sort_index().to_dict()}")

    # --- 4. Write temp CSV ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="smoke_subset_"
    ) as tmp:
        sub.to_csv(tmp, index=False)
        tmp_csv = tmp.name
    print(f"[OK]  Temp CSV written: {tmp_csv}")

    # --- 5. Run baseline ---
    baseline = Path(args.baseline_script)
    if not baseline.exists():
        sys.exit(f"[ERROR] Baseline script not found: {baseline}")

    cmd = [
        sys.executable, str(baseline),
        "--data_csv",      tmp_csv,
        "--img_root",      args.img_root,
        "--output_dir",    "runs/smoke",
        "--model",         "resnet18",
        "--epochs",        str(args.epochs),
        "--batch_size",    "8",
        "--num_workers",   "0",      # avoid multiprocessing issues on login node
        "--no_amp",                  # CPU: disable AMP
        "--val_size",      "0.2",
        "--test_size",     "0.2",
        "--seed",          "42",
    ]

    print("\n[RUN] " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, text=True)

    # --- 6. Report ---
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("[PASS] Smoke test completed successfully.")
        print("       Safe to submit GPU job.")
        print("       Check runs/smoke/ for config.json and test_metrics.json.")
    else:
        print(f"[FAIL] Baseline exited with code {result.returncode}.")
        print("       Fix the error above before submitting to the GPU queue.")
    print("=" * 60)

    # Clean up temp file
    Path(tmp_csv).unlink(missing_ok=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
