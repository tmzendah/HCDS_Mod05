"""Filter and sample MRKR metadata for KL grading experiments.

Produces a balanced dataset of 2,000 images per KL grade including:
  - All acquisition types (weightbearing AND non-weightbearing)
  - Demographics: sex, race, ethnicity merged from MRKR_demographics.csv
  - Age groups for subgroup analysis
  - All metadata needed for post-hoc fairness analysis

Subgroup analyses enabled
--------------------------
  - Performance by weightbearing status (WB vs non-WB)
  - Performance by sex (Male / Female)
  - Performance by race
  - Performance by ethnicity
  - Performance by age group (decade bins)
  - Performance by laterality (left vs right knee)
  - Grade 1 detection across all strata

Output
------
  mrkr_selected_v2.csv  -- 10,000 rows, 2,000 per grade

  Columns:
    empi_anon, label, age_at_exam, age_group,
    sex, race, ethnicity,
    weight_bearing, knee_side, laterality,
    dicom_path, inverted, horizontal_flip,
    img_height, img_width,
    L_KLG_inference, R_KLG_inference

Usage
-----
  python mrkr_filter_v2.py \
      --metadata_csv     /rds/user/tm922/hpc-work/data/mrkr/MRKR_image_metadata.csv \
      --demographics_csv /rds/user/tm922/hpc-work/data/mrkr/MRKR_demographics.csv \
      --output_csv       /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv \
      --n_per_grade      2000 \
      --seed             42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bin_age(age) -> str:
    """Bin age into decade groups for subgroup analysis."""
    if pd.isna(age):
        return "unknown"
    age = int(age)
    if age < 40:
        return "<40"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70+"


def section(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv",     type=str, required=True,
                   help="Path to MRKR_image_metadata.csv")
    p.add_argument("--demographics_csv", type=str, required=True,
                   help="Path to MRKR_demographics.csv")
    p.add_argument("--output_csv",       type=str, required=True,
                   help="Output CSV path")
    p.add_argument("--n_per_grade",      type=int, default=2000,
                   help="Images to sample per KL grade (default 2000)")
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("MRKR  —  Filter and Sample  (v2)")
    print("=" * 60)

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    section("1 / 5  Load metadata")
    df = pd.read_csv(args.metadata_csv, low_memory=False)
    print(f"  Raw rows          : {len(df):,}")
    print(f"  Columns           : {list(df.columns)}")

    # ── 2. Load and merge demographics ────────────────────────────────────────
    section("2 / 5  Load and merge demographics")
    demo = pd.read_csv(args.demographics_csv, low_memory=False)
    print(f"  Demographics rows : {len(demo):,}")
    print(f"  Demographics cols : {list(demo.columns)}")

    for col in ["sex", "race", "ethnicity"]:
        if col in demo.columns:
            print(f"\n  {col} distribution:")
            print(f"{demo[col].value_counts(dropna=False).head(10).to_string()}")
        else:
            print(f"  [WARN] Column '{col}' not found in demographics")

    # Merge on empi_anon — left join keeps all images
    df = df.merge(
        demo[["empi_anon", "sex", "race", "ethnicity"]],
        on="empi_anon",
        how="left",
    )
    print(f"\n  After merge       : {len(df):,} rows")
    print(f"  Sex missing       : {df['sex'].isna().sum():,}")
    print(f"  Race missing      : {df['race'].isna().sum():,}")
    print(f"  Ethnicity missing : {df['ethnicity'].isna().sum():,}")

    # ── 3. Filter cascade ─────────────────────────────────────────────────────
    section("3 / 5  Filter cascade")

    cascade = {"00_raw": len(df)}

    # Frontal views only
    df = df[df["view_position"] == "F"].copy()
    cascade["01_frontal"] = len(df)
    print(f"  Frontal only          : {len(df):,}")

    # Known laterality
    df = df[df["laterality"] != "-1"].copy()
    cascade["02_known_laterality"] = len(df)
    print(f"  Known laterality      : {len(df):,}")

    # No arthroplasty
    df = df[df["arthroplasty"] == "0"].copy()
    cascade["03_no_arthroplasty"] = len(df)
    print(f"  No arthroplasty       : {len(df):,}")

    # Derive label from laterality
    uni_R = df[df["laterality"] == "R"].copy()
    uni_R["label"]     = uni_R["R_KLG_inference"]
    uni_R["knee_side"] = "R"

    uni_L = df[df["laterality"] == "L"].copy()
    uni_L["label"]     = uni_L["L_KLG_inference"]
    uni_L["knee_side"] = "L"

    bil   = df[df["laterality"] == "B"].copy()
    bil_R = bil.copy()
    bil_R["label"]     = bil_R["R_KLG_inference"]
    bil_R["knee_side"] = "R"
    bil_L = bil.copy()
    bil_L["label"]     = bil_L["L_KLG_inference"]
    bil_L["knee_side"] = "L"

    df = pd.concat([uni_R, uni_L, bil_R, bil_L], ignore_index=True)
    cascade["04_bilateral_expansion"] = len(df)
    print(f"  After expansion       : {len(df):,}")

    # Drop NaN labels
    df = df[df["label"].notna()].copy()
    cascade["05_label_not_nan"] = len(df)
    print(f"  Label not NaN         : {len(df):,}")

    # Valid KLG range 0-4
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2, 3, 4])].copy()
    cascade["06_valid_klg"] = len(df)
    print(f"  Valid KLG 0-4         : {len(df):,}")

    print(f"\n  KLG distribution before sampling:")
    print(df["label"].value_counts().sort_index().to_string())
    print(f"\n  Weightbearing breakdown:")
    print(df["weight_bearing"].value_counts(dropna=False).to_string())
    print(f"\n  Sex breakdown:")
    print(df["sex"].value_counts(dropna=False).head(8).to_string())
    print(f"\n  Race breakdown:")
    print(df["race"].value_counts(dropna=False).head(8).to_string())

    # ── 4. Stratified sample 2000 per grade ───────────────────────────────────
    section("4 / 5  Stratified sample")
    print(f"  Sampling up to {args.n_per_grade:,} per grade")
    print(f"  Includes ALL acquisition types (WB + non-WB)")
    print(f"  Random sample within each grade (seed={args.seed})")

    grades = []
    for grade in [0, 1, 2, 3, 4]:
        available = df[df["label"] == grade]
        n = min(args.n_per_grade, len(available))
        if n < args.n_per_grade:
            print(f"  [WARN] Grade {grade}: only {len(available):,} available, "
                  f"sampling all {n:,}")
        else:
            print(f"  Grade {grade}: {len(available):,} available "
                  f"-> sampling {n:,}")
        sampled = available.sample(n=n, random_state=args.seed)
        grades.append(sampled)

    df_sample = pd.concat(grades, ignore_index=True)
    print(f"\n  Total sampled: {len(df_sample):,}")

    # ── 5. Add derived columns and save ───────────────────────────────────────
    section("5 / 5  Finalise and save")

    # Age group bins
    df_sample["age_group"] = df_sample["age_at_exam"].apply(bin_age)

    # Clean weightbearing to binary int
    df_sample["weight_bearing"] = df_sample["weight_bearing"].map(
        {1.0: 1, 0.0: 0, 1: 1, 0: 0,
         True: 1, False: 0,
         "1.0": 1, "0.0": 0}
    )

    # Final column selection
    keep_cols = [
        "empi_anon",
        "label",
        "age_at_exam",
        "age_group",
        "sex",
        "race",
        "ethnicity",
        "weight_bearing",
        "knee_side",
        "laterality",
        "dicom_path",
        "inverted",
        "horizontal_flip",
        "img_height",
        "img_width",
        "L_KLG_inference",
        "R_KLG_inference",
    ]

    keep_cols = [c for c in keep_cols if c in df_sample.columns]
    df_final  = df_sample[keep_cols].copy()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    # Summary
    print(f"  Final columns     : {list(df_final.columns)}")
    print(f"\n  Label distribution:")
    print(df_final["label"].value_counts().sort_index().to_string())
    print(f"\n  Weightbearing breakdown:")
    print(df_final["weight_bearing"].value_counts(dropna=False).to_string())
    print(f"\n  Age group breakdown:")
    print(df_final["age_group"].value_counts().sort_index().to_string())
    print(f"\n  Sex breakdown:")
    print(df_final["sex"].value_counts(dropna=False).head(8).to_string())
    print(f"\n  Race breakdown:")
    print(df_final["race"].value_counts(dropna=False).head(8).to_string())
    print(f"\n  Ethnicity breakdown:")
    print(df_final["ethnicity"].value_counts(dropna=False).head(8).to_string())

    cascade_path = output_path.parent / "filter_cascade_v2.json"
    with cascade_path.open("w") as f:
        json.dump(cascade, f, indent=2)

    print(f"\n  Saved CSV     -> {output_path}  ({len(df_final):,} rows)")
    print(f"  Saved cascade -> {cascade_path}")
    print("\n" + "=" * 60)
    print("  Next steps:")
    print("  1. Submit SLURM download job for these DICOMs")
    print("  2. Convert DICOMs to PNG via SLURM")
    print("  3. Rerun EDA on this dataset")
    print("  4. Train baseline")
    print("=" * 60)


if __name__ == "__main__":
    main()
