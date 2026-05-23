#!/usr/bin/env python3
"""
00_setup.py - Create project structure and check dependencies.

Run this first on the HPC before any other scripts.

Project: Automated KL Grading of Knee OA from MRKR Dataset
Author : tm922
HPC    : Cambridge CSD3 (NVIDIA A100-SXM4-80GB)
Env    : conda activate OAIKaggle (Python 3.11, torch 2.7.1+cu118)

Directory structure created:
  ~/MLOAIProject/
    code/          -- Python scripts and SLURM jobs
    data/          -- Small local files (smoke CSVs, manifests)
    eda/           -- EDA outputs (PNGs, JSONs)
    logs/          -- SLURM job logs
    notebooks/     -- Jupyter notebooks
    results/       -- Figures and evaluation outputs
      crop_audit/  -- Bilateral crop audit images
      figures/     -- Final report figures
    runs/          -- Training run outputs (per model per seed)

RDS data paths (large files, not in repo):
  /rds/user/tm922/hpc-work/data/
    mrkr/                        -- MRKR metadata CSVs
      MRKR_image_metadata.csv    -- Full 503K metadata
      MRKR_demographics.csv      -- Patient demographics
      mrkr_selected_v2.csv       -- 10K working set
    mrkr_png_v2/                 -- Original 10K bilateral PNGs
      images/                    -- PNG files
      mrkr_png_manifest.csv      -- Manifest with png_path column
    mrkr_cropped/                -- Cropped + CLAHE PNGs (training data)
      images/                    -- Cropped PNG files
      mrkr_cropped_manifest.csv  -- Manifest for training
    mrkr_png_v3/                 -- Additional 20K PNGs (future use)
      images/
      mrkr_png_manifest_v3.csv
    knee_oa/                     -- OAI/Kaggle external validation dataset
      train/                     -- Pre-defined train split
      val/                       -- Pre-defined validation split
      test/                      -- Pre-defined test split

Usage:
  conda activate OAIKaggle
  cd ~/MLOAIProject
  python3 code/00_setup.py
"""

import os
import sys
import subprocess


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

HOME          = os.path.expanduser("~")
PROJECT_DIR   = os.path.join(HOME, "MLOAIProject")
RDS_BASE      = "/rds/user/tm922/hpc-work"
RDS_DATA      = os.path.join(RDS_BASE, "data")

# Key data paths — verify these exist
EXPECTED_PATHS = {
    "MRKR metadata CSV":        os.path.join(RDS_DATA, "mrkr", "MRKR_image_metadata.csv"),
    "MRKR demographics CSV":    os.path.join(RDS_DATA, "mrkr", "MRKR_demographics.csv"),
    "MRKR working set CSV":     os.path.join(RDS_DATA, "mrkr", "working_set.csv"),
    "MRKR PNG manifest":        os.path.join(RDS_DATA, "mrkr_png_v2", "mrkr_png_manifest.csv"),
    "MRKR cropped manifest":    os.path.join(RDS_DATA, "mrkr_cropped", "mrkr_cropped_manifest.csv"),
    "OAI/Kaggle train dir":     os.path.join(RDS_DATA, "knee_oa", "train"),
    "OAI/Kaggle val dir":       os.path.join(RDS_DATA, "knee_oa", "val"),
    "OAI/Kaggle test dir":      os.path.join(RDS_DATA, "knee_oa", "test"),
}

# Project directories to create under ~/mrkr_klg
PROJECT_DIRS = [
    "code",
    "data",
    "eda",
    "logs",
    "notebooks",
    "results/crop_audit",
    "results/figures",
    "runs",
]

# Required Python packages
REQUIRED_PACKAGES = {
    "torch":          "torch",
    "torchvision":    "torchvision",
    "pandas":         "pandas",
    "numpy":          "numpy",
    "scikit-learn":   "sklearn",
    "matplotlib":     "matplotlib",
    "seaborn":        "seaborn",
    "tqdm":           "tqdm",
    "pillow":         "PIL",
    "pydicom":        "pydicom",
    "opencv-python":  "cv2",
    "scikit-image":   "skimage",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Create project directories
# ─────────────────────────────────────────────────────────────────────────────

def create_directories():
    section("1 / 4  Creating project directories")
    for d in PROJECT_DIRS:
        full_path = os.path.join(PROJECT_DIR, d)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✓ {full_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Check required packages
# ─────────────────────────────────────────────────────────────────────────────

def check_packages():
    section("2 / 4  Checking required packages")
    missing = []
    for pkg_name, import_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            missing.append(pkg_name)
            print(f"  ✗ {pkg_name}  (missing)")

    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify RDS data paths
# ─────────────────────────────────────────────────────────────────────────────

def check_data_paths():
    section("3 / 4  Verifying RDS data paths")
    missing = []
    for label, path in EXPECTED_PATHS.items():
        if os.path.exists(path):
            print(f"  ✓ {label}")
            print(f"      {path}")
        else:
            missing.append((label, path))
            print(f"  ✗ {label}  (NOT FOUND)")
            print(f"      {path}")

    if missing:
        print(f"\n  {len(missing)} path(s) missing — check RDS mount and paths above.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Check GPU and environment
# ─────────────────────────────────────────────────────────────────────────────

def check_environment():
    section("4 / 4  Checking environment")

    # Python version
    print(f"  Python  : {sys.version.split()[0]}")

    # Conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"  Env     : {conda_env}")
    if conda_env != "OAIKaggle":
        print(f"  [WARN] Expected OAIKaggle environment — got '{conda_env}'")
        print(f"         Run: conda activate OAIKaggle")

    # PyTorch and CUDA
    try:
        import torch
        print(f"  PyTorch : {torch.__version__}")
        print(f"  CUDA    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU     : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        else:
            print(f"  [INFO]  GPU not visible from login node — expected on GPU job nodes")
    except ImportError:
        print(f"  [ERROR] PyTorch not installed")

    # Disk usage on RDS
    try:
        result = subprocess.run(
            ["du", "-sh", RDS_DATA],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"  RDS data dir size: {size}")
    except Exception:
        pass

    # Count training images
    cropped_dir = os.path.join(RDS_DATA, "mrkr_cropped", "images")
    if os.path.exists(cropped_dir):
        n_cropped = len([f for f in os.listdir(cropped_dir) if f.endswith(".png")])
        print(f"  Cropped PNGs (training data): {n_cropped:,}")

    original_dir = os.path.join(RDS_DATA, "mrkr_png_v2", "images")
    if os.path.exists(original_dir):
        n_orig = len([f for f in os.listdir(original_dir) if f.endswith(".png")])
        print(f"  Original PNGs (bilateral):    {n_orig:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MRKR KL Grading — Project Setup")
    print("  Cambridge CSD3 HPC")
    print("=" * 60)
    print(f"  Project dir : {PROJECT_DIR}")
    print(f"  RDS data    : {RDS_DATA}")

    create_directories()
    pkg_ok  = check_packages()
    data_ok = check_data_paths()
    check_environment()

    print("\n" + "=" * 60)
    if pkg_ok and data_ok:
        print("  Setup complete. All checks passed.")
        print("  Next step: python3 code/mrkr_train_v3.py --help")
    else:
        print("  Setup incomplete — fix issues above before proceeding.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
