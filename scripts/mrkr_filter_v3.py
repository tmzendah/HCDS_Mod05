"""Filter and sample additional 4,000 images per grade from MRKR.

Excludes patients already present in mrkr_selected_v2.csv to ensure
no patient overlap between the two datasets.

Output: mrkr_selected_v3.csv — 20,000 rows, 4,000 per grade
        (completely separate patients from mrkr_selected_v2.csv)

Usage:
  python mrkr_filter_v3.py \
      --metadata_csv     /rds/user/tm922/hpc-work/data/mrkr/MRKR_image_metadata.csv \
      --demographics_csv /rds/user/tm922/hpc-work/data/mrkr/MRKR_demographics.csv \
      --exclude_csv      /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv \
      --output_csv       /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v3.csv \
      --n_per_grade      4000 \
      --seed             42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bin_age(age) -> str:
    if pd.isna(age):
        return "unknown"
    age = int(age)
    if age < 40:   return "<40"
    elif age < 50: return "40-49"
    elif age < 60: return "50-59"
    elif age < 70: return "60-69"
    else:          return "70+"


def section(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv",     type=str, required=True)
    p.add_argument("--demographics_csv", type=str, required=True)
    p.add_argument("--exclude_csv",      type=str, required=True,
                   help="CSV of already downloaded patients to exclude")
    p.add_argument("--output_csv",       type=str, required=True)
    p.add_argument("--n_per_grade",      type=int, default=4000)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("MRKR  —  Filter and Sample  (v3 — additional dataset)")
    print("=" * 60)

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    section("1 / 5  Load metadata")
    df = pd.read_csv(args.metadata_csv, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # ── 2. Load exclusion list ────────────────────────────────────────────────
    section("2 / 5  Load exclusion list")
    exclude_df = pd.read_csv(args.exclude_csv)
    exclude_patients = set(exclude_df["empi_anon"].unique())
    print(f"  Patients to exclude: {len(exclude_patients):,}")

    # Exclude already downloaded patients
    before = len(df)
    df = df[~df["empi_anon"].isin(exclude_patients)].copy()
    print(f"  Rows after exclusion: {len(df):,}  (removed {before - len(df):,})")

    # ── 3. Load and merge demographics ────────────────────────────────────────
    section("3 / 5  Merge demographics")
    demo = pd.read_csv(args.demographics_csv, low_memory=False)
    demo = demo[~demo["empi_anon"].isin(exclude_patients)]
    print(f"  Demographics rows after exclusion: {len(demo):,}")

    df = df.merge(
        demo[["empi_anon", "sex", "race", "ethnicity"]],
        on="empi_anon",
        how="left",
    )
    print(f"  After merge: {len(df):,} rows")

    # ── 4. Filter cascade ─────────────────────────────────────────────────────
    section("4 / 5  Filter cascade")

    cascade = {"00_raw_after_exclusion": len(df)}

    df = df[df["view_position"] == "F"].copy()
    cascade["01_frontal"] = len(df)
    print(f"  Frontal only          : {len(df):,}")

    df = df[df["laterality"] != "-1"].copy()
    cascade["02_known_laterality"] = len(df)
    print(f"  Known laterality      : {len(df):,}")

    df = df[df["arthroplasty"] == "0"].copy()
    cascade["03_no_arthroplasty"] = len(df)
    print(f"  No arthroplasty       : {len(df):,}")

    # Derive label
    uni_R = df[df["laterality"] == "R"].copy()
    uni_R["label"] = uni_R["R_KLG_inference"]
    uni_R["knee_side"] = "R"

    uni_L = df[df["laterality"] == "L"].copy()
    uni_L["label"] = uni_L["L_KLG_inference"]
    uni_L["knee_side"] = "L"

    bil = df[df["laterality"] == "B"].copy()
    bil_R = bil.copy()
    bil_R["label"] = bil_R["R_KLG_inference"]
    bil_R["knee_side"] = "R"
    bil_L = bil.copy()
    bil_L["label"] = bil_L["L_KLG_inference"]
    bil_L["knee_side"] = "L"

    df = pd.concat([uni_R, uni_L, bil_R, bil_L], ignore_index=True)
    cascade["04_bilateral_expansion"] = len(df)
    print(f"  After expansion       : {len(df):,}")

    df = df[df["label"].notna()].copy()
    cascade["05_label_not_nan"] = len(df)
    print(f"  Label not NaN         : {len(df):,}")

    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2, 3, 4])].copy()
    cascade["06_valid_klg"] = len(df)
    print(f"  Valid KLG 0-4         : {len(df):,}")

    print(f"\n  Available per grade after exclusion:")
    print(df["label"].value_counts().sort_index().to_string())

    # ── 5. Sample and save ────────────────────────────────────────────────────
    section("5 / 5  Sample and save")
    print(f"  Sampling up to {args.n_per_grade:,} per grade (seed={args.seed})")

    grades = []
    for grade in [0, 1, 2, 3, 4]:
        available = df[df["label"] == grade]
        n = min(args.n_per_grade, len(available))
        if n < args.n_per_grade:
            print(f"  [WARN] Grade {grade}: only {len(available):,} available, "
                  f"sampling all {n:,}")
        else:
            print(f"  Grade {grade}: {len(available):,} available -> {n:,}")
        grades.append(available.sample(n=n, random_state=args.seed))

    df_sample = pd.concat(grades, ignore_index=True)
    print(f"\n  Total sampled: {len(df_sample):,}")

    # Add derived columns
    df_sample["age_group"] = df_sample["age_at_exam"].apply(bin_age)
    df_sample["weight_bearing"] = df_sample["weight_bearing"].map(
        {1.0: 1, 0.0: 0, 1: 1, 0: 0, True: 1, False: 0}
    )

    keep_cols = [
        "empi_anon", "label", "age_at_exam", "age_group",
        "sex", "race", "ethnicity", "weight_bearing",
        "knee_side", "laterality", "dicom_path",
        "inverted", "horizontal_flip",
        "img_height", "img_width",
        "L_KLG_inference", "R_KLG_inference",
    ]
    keep_cols = [c for c in keep_cols if c in df_sample.columns]
    df_final  = df_sample[keep_cols].copy()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    # Save cascade
    cascade_path = output_path.parent / "filter_cascade_v3.json"
    with cascade_path.open("w") as f:
        json.dump(cascade, f, indent=2)

    print(f"\n  Label distribution:")
    print(df_final["label"].value_counts().sort_index().to_string())
    print(f"\n  Weightbearing breakdown:")
    print(df_final["weight_bearing"].value_counts(dropna=False).to_string())
    print(f"\n  Sex breakdown:")
    print(df_final["sex"].value_counts(dropna=False).head(5).to_string())
    print(f"\n  Saved -> {output_path}  ({len(df_final):,} rows)")
    print(f"  Cascade -> {cascade_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
