"""
src/dataset.py
Data loading, preprocessing, and augmentation pipeline.

Pipeline (applied in this order)
----------------------------------
1. Load image from disk as grayscale (mode L)
2. Apply CLAHE (contrast enhancement)
3. Convert to 3-channel RGB (required for ImageNet pretrained weights)
4. Apply augmentation (train split only):
     - Random horizontal flip
     - Random rotation +-10 degrees
5. Convert to tensor
6. Normalise with ImageNet stats

CLAHE settings (IEC 62494-1 / Hassan et al. 2024):
    clipLimit=2.0, tileGridSize=(8,8)

Normalisation stats (ImageNet):
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]

Note on class imbalance
-----------------------
Class imbalance (KL0:KL4 ratio ~13:1 in train split) is acknowledged
as a dataset characteristic. It is not corrected via oversampling or
weighted sampling in order to keep all four experimental conditions
(2 architectures x 2 loss functions) cleanly comparable.
The imbalance is reported in results and discussed as a limitation.
Reference: Hassan et al. 2024 (SMOTE+CLAHE) for future work direction.

Usage
-----
    from src.dataset import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir = "/path/to/your/data/knee_oa",
        batch_size = 32,
        seed = 42,
    )
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────
NUM_CLASSES   = 5
GRADES        = [0, 1, 2, 3, 4]
SPLITS        = ["train", "val", "test"]

# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# CLAHE settings (Hassan et al. 2024; standard medical imaging)
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_SIZE   = (8, 8)


# ─────────────────────────────────────────────────────────
# CLAHE transform
# ─────────────────────────────────────────────────────────

class ApplyCLAHE:
    """
    Custom transform: applies CLAHE to a PIL Image.

    Why CLAHE?
    ----------
    Contrast Limited Adaptive Histogram Equalisation enhances local
    contrast in small image regions (tiles) independently, improving
    visibility of subtle structural features such as early joint space
    narrowing and osteophyte margins without globally altering brightness.
    This is particularly important for KL1/KL2 discrimination where
    contrast differences at joint margins are clinically significant.

    Applied to ALL splits (train, val, test) equally — it is a
    preprocessing step, not an augmentation.

    Input : PIL Image (any mode)
    Output: PIL Image (RGB)
    """

    def __init__(self,
                 clip_limit: float = CLAHE_CLIP_LIMIT,
                 tile_size: tuple  = CLAHE_TILE_SIZE):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL to numpy grayscale
        gray = np.array(img.convert("L"), dtype=np.uint8)

        # Apply CLAHE
        enhanced = self.clahe.apply(gray)

        # Stack to 3-channel RGB (required for ImageNet pretrained weights)
        # All three channels are identical — grayscale encoded as RGB
        rgb = np.stack([enhanced, enhanced, enhanced], axis=2)

        return Image.fromarray(rgb, mode="RGB")


# ─────────────────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Training transforms:
        CLAHE → random horizontal flip → random rotation +-10 deg
        → tensor → ImageNet normalisation

    Augmentation rationale:
        - Horizontal flip: knee radiographs are laterally symmetric;
          flipping creates valid additional examples without distortion
        - Rotation +-10 deg: simulates minor patient positioning variation,
          consistent with clinical acquisition variability observed in EDA
        - No brightness/contrast jitter: CLAHE already handles contrast;
          additional jitter would conflict with its normalising effect
    """
    return transforms.Compose([
        ApplyCLAHE(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms() -> transforms.Compose:
    """
    Validation and test transforms:
        CLAHE → tensor → ImageNet normalisation

    No augmentation — deterministic pipeline for reproducible evaluation.
    CLAHE still applied to match training distribution.
    """
    return transforms.Compose([
        ApplyCLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────

class KneeOADataset(Dataset):
    """
    PyTorch Dataset for the Kaggle Knee OA dataset.

    Expects ImageFolder-style directory structure:
        data_dir/
            split/
                0/  <- KL grade 0 images
                1/  <- KL grade 1 images
                2/  <- KL grade 2 images
                3/  <- KL grade 3 images
                4/  <- KL grade 4 images

    Args:
        data_dir  : path to dataset root
        split     : 'train', 'val', or 'test'
        transform : torchvision transform pipeline
    """

    # Supported image extensions
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self,
                 data_dir:  str,
                 split:     str,
                 transform: transforms.Compose = None):

        self.data_dir  = Path(data_dir)
        self.split     = split
        self.transform = transform

        # Validate split
        if split not in SPLITS:
            raise ValueError(
                f"split must be one of {SPLITS}, got '{split}'"
            )

        # Validate directory
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}"
            )

        # Build image list: [(path, label), ...]
        self.samples = self._load_samples(split_dir)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Check directory structure and file extensions."
            )

        # Class counts for reporting
        self.class_counts = self._count_classes()

    def _load_samples(self, split_dir: Path) -> list:
        """Scan grade folders and collect (image_path, label) pairs."""
        samples = []
        for grade in GRADES:
            grade_dir = split_dir / str(grade)
            if not grade_dir.exists():
                print(f"  Warning: grade folder not found: {grade_dir}")
                continue
            for img_path in sorted(grade_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    samples.append((img_path, grade))
        return samples

    def _count_classes(self) -> dict:
        """Return count of samples per KL grade."""
        counts = {g: 0 for g in GRADES}
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image : tensor of shape [3, H, W], normalised
            label : integer KL grade (0-4)
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {img_path}\nError: {e}"
            )

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        counts_str = "  ".join(
            f"KL{g}:{self.class_counts[g]}" for g in GRADES
        )
        return (
            f"KneeOADataset(split={self.split}, "
            f"n={len(self.samples)},  {counts_str})"
        )


# ─────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:   str,
    batch_size: int = 32,
    seed:       int = 42,
    num_workers: int = 4,
) -> tuple:
    """
    Create train, val, and test DataLoaders.

    Args:
        data_dir   : path to dataset root (contains train/, val/, test/)
        batch_size : images per batch (default 32, per Momenpour et al. 2025)
        seed       : random seed for reproducibility
        num_workers: parallel data loading workers (default 4 for A100)

    Returns:
        (train_loader, val_loader, test_loader)

    Design decisions:
        - shuffle=True for train only (val/test must be deterministic)
        - pin_memory=True for faster GPU transfer on A100
        - persistent_workers=True reduces worker startup overhead
          across epochs
        - drop_last=False ensures all test images are evaluated
    """

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Build datasets
    train_dataset = KneeOADataset(
        data_dir  = data_dir,
        split     = "train",
        transform = get_train_transforms(),
    )
    val_dataset = KneeOADataset(
        data_dir  = data_dir,
        split     = "val",
        transform = get_val_test_transforms(),
    )
    test_dataset = KneeOADataset(
        data_dir  = data_dir,
        split     = "test",
        transform = get_val_test_transforms(),
    )

    # Worker seed function for reproducibility across seeds
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size        = batch_size,
        shuffle           = True,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= True,
        worker_init_fn    = worker_init_fn,
        drop_last         = False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= True,
        drop_last         = False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= True,
        drop_last         = False,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────
# Quick verification (run directly to test pipeline)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    args = p.parse_args()

    print("=" * 60)
    print("  dataset.py — pipeline verification")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir   = args.data_dir,
        batch_size = 32,
        seed       = 42,
    )

    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        print(f"\n  {name}: {loader.dataset}")
        images, labels = next(iter(loader))
        print(f"    batch shape : {images.shape}")
        print(f"    dtype       : {images.dtype}")
        print(f"    pixel range : [{images.min():.3f}, {images.max():.3f}]")
        print(f"    label range : [{labels.min()}, {labels.max()}]")

    print("\n  Pipeline verified successfully.")
    print("=" * 60)
