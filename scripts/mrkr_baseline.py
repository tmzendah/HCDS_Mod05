"""MRKR baseline training script — PNG loading version

Architectures supported: resnet50, densenet121, efficientnet_b0

Design decisions from EDA:
  - img_size=512        : median DICOM resolution 3028x2539; 224px loses
                          joint space detail critical for KL 0/1/2
  - PNG loading         : DICOMs converted once; PIL loading removes
                          pydicom dependency from training pipeline
  - Balanced dataset    : 2,000 per grade sampled at data level;
                          no WeightedRandomSampler or class weights needed
  - Patient-level split : bilateral expansion means empi_anon-level split
                          prevents data leakage across partitions
  - Subgroup metadata   : sex, race, ethnicity, weightbearing, age_group
                          retained in split CSVs for post-hoc fairness

References:
  ResNet50    — Tiulpin et al. (2019) Scientific Reports
  DenseNet121 — Gour et al. (2023) Scientific Reports
  EfficientNet— Dipto & Goni (2024) IEEE PEEIACON

Usage:
  python mrkr_baseline.py \
      --data_csv   /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv \
      --img_root   /rds/user/tm922/hpc-work/data/mrkr_png_v2/images \
      --output_dir runs/resnet50 \
      --model      resnet50 \
      --epochs     30 \
      --batch_size 32 \
      --img_size   512
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data_csv: str
    img_root: str
    output_dir: str      = "runs/baseline"
    model: str           = "resnet50"
    num_classes: int     = 5
    img_size: int        = 512
    batch_size: int      = 32
    epochs: int          = 30
    lr: float            = 1e-4
    weight_decay: float  = 1e-4
    num_workers: int     = 4
    seed: int            = 42
    val_size: float      = 0.15
    test_size: float     = 0.15
    early_stop_patience: int = 7
    use_amp: bool        = True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — PNG loading
# ─────────────────────────────────────────────────────────────────────────────

class MRKRDataset(Dataset):
    """MRKR knee radiograph dataset — loads PNGs with PIL.

    Expects a manifest CSV with columns:
      png_path  : relative path from img_root to PNG file
      label     : integer KL grade 0-4
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_root: str,
        transform=None,
    ):
        self.df       = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.img_root / str(row["png_path"])
        image    = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        return model

    raise ValueError(
        f"Unsupported model: {name}. "
        "Choose from: resnet50, densenet121, efficientnet_b0"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(img_size: int):
    """
    512px: preserves fine joint space detail lost at 224px.
    Conservative augmentation: knee radiographs have fixed orientation;
    aggressive transforms risk distorting diagnostic joint space geometry.
    Horizontal flip included: MRKR contains both left and right knees.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tfms, eval_tfms


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level split
# ─────────────────────────────────────────────────────────────────────────────

def patient_stratified_split(
    df: pd.DataFrame,
    seed: int,
    val_size: float,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at patient (empi_anon) level to prevent data leakage.

    Bilateral expansion creates two rows per bilateral image.
    Row-level split would place both knees from same patient in
    train and test — leaking information. Patient-level split prevents this.
    """
    patient_grade = (
        df.groupby("empi_anon")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    patient_grade.columns = ["empi_anon", "majority_label"]

    train_pats, temp_pats = train_test_split(
        patient_grade,
        test_size=(val_size + test_size),
        stratify=patient_grade["majority_label"],
        random_state=seed,
    )

    rel_test = test_size / (val_size + test_size)
    val_pats, test_pats = train_test_split(
        temp_pats,
        test_size=rel_test,
        stratify=temp_pats["majority_label"],
        random_state=seed,
    )

    train_df = df[df["empi_anon"].isin(train_pats["empi_anon"])].copy()
    val_df   = df[df["empi_anon"].isin(val_pats["empi_anon"])].copy()
    test_df  = df[df["empi_anon"].isin(test_pats["empi_anon"])].copy()

    print(f"  Patient-level split (70/15/15):")
    print(f"    Train : {len(train_pats):,} patients, {len(train_df):,} images")
    print(f"    Val   : {len(val_pats):,} patients,  {len(val_df):,} images")
    print(f"    Test  : {len(test_pats):,} patients,  {len(test_df):,} images")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in loader:
        images  = images.to(device)
        targets = targets.to(device)
        logits  = model(images)
        probs   = torch.softmax(logits, dim=1)
        preds   = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy())

    y_true   = np.array(all_targets)
    y_pred   = np.array(all_preds)
    prob_arr = np.vstack(all_probs)

    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred,
                                            average="macro",
                                            zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred,
                                            average="weighted",
                                            zero_division=0)),
        "mae_grade":         float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
        "preds":   y_pred.tolist(),
        "targets": y_true.tolist(),
        "probs":   prob_arr.tolist(),
    }

    # Grade-1 one-vs-rest — key clinical metric
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true != 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred != 1)).sum())
    tn = int(((y_true != 1) & (y_pred != 1)).sum())
    precision   = tp / (tp + fp) if (tp + fp) else 0.0
    recall      = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)

    metrics["grade1_one_vs_rest"] = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "specificity": round(specificity, 4),
        "f1":          round(f1, 4),
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="  train", leave=False):
        images  = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        raise TypeError(f"Not serialisable: {type(o)}")

    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_convert)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",    type=str, required=True,
                   help="Path to mrkr_png_manifest.csv")
    p.add_argument("--img_root",    type=str, required=True,
                   help="Root directory containing PNG images")
    p.add_argument("--output_dir",  type=str, default="runs/baseline")
    p.add_argument("--model",       type=str, default="resnet50",
                   choices=["resnet50", "densenet121", "efficientnet_b0"])
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--val_size",    type=float, default=0.15)
    p.add_argument("--test_size",   type=float, default=0.15)
    p.add_argument("--early_stop_patience", type=int, default=7)
    p.add_argument("--no_amp",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config(
        data_csv=args.data_csv,
        img_root=args.img_root,
        output_dir=args.output_dir,
        model=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        early_stop_patience=args.early_stop_patience,
        use_amp=not args.no_amp,
    )

    seed_everything(cfg.seed)

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), outdir / "config.json")

    print("=" * 60)
    print(f"MRKR  —  Training  [{cfg.model.upper()}]")
    print("=" * 60)
    print(f"  img_size    : {cfg.img_size}px")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  epochs      : {cfg.epochs}")
    print(f"  lr          : {cfg.lr}")
    print(f"  seed        : {cfg.seed}")

    # Load manifest
    df = pd.read_csv(cfg.data_csv)
    print(f"\n  Manifest loaded : {len(df):,} rows")
    print(f"  Label dist      : "
          f"{df['label'].value_counts().sort_index().to_dict()}")

    # Patient-level split
    train_df, val_df, test_df = patient_stratified_split(
        df, cfg.seed, cfg.val_size, cfg.test_size
    )

    # Save splits — include all metadata for subgroup analysis
    train_df.to_csv(outdir / "train_split.csv", index=False)
    val_df.to_csv(outdir / "val_split.csv",     index=False)
    test_df.to_csv(outdir / "test_split.csv",   index=False)
    print(f"\n  Splits saved to {outdir}/")

    # Transforms
    train_tfms, eval_tfms = build_transforms(cfg.img_size)

    # Datasets
    train_ds = MRKRDataset(train_df, cfg.img_root, train_tfms)
    val_ds   = MRKRDataset(val_df,   cfg.img_root, eval_tfms)
    test_ds  = MRKRDataset(test_df,  cfg.img_root, eval_tfms)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    model  = build_model(cfg.model, cfg.num_classes).to(device)

    # Loss — standard cross entropy (data is balanced by design)
    criterion = nn.CrossEntropyLoss()

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if (cfg.use_amp and device.type == "cuda") else None
    )

    # Training loop
    best_val  = -1.0
    best_path = outdir / "best_model.pt"
    patience  = 0
    history: List[Dict] = []

    print(f"\n{'─'*60}")
    print(f"  Training — up to {cfg.epochs} epochs  "
          f"(early stop patience={cfg.early_stop_patience})")
    print(f"{'─'*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss  = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, val_loader, device)
        val_score   = float(val_metrics["balanced_accuracy"])
        scheduler.step(val_score)

        row = {
            "epoch":             epoch,
            "train_loss":        round(train_loss, 4),
            "val_accuracy":      round(val_metrics["accuracy"], 4),
            "val_balanced_acc":  round(val_metrics["balanced_accuracy"], 4),
            "val_macro_f1":      round(val_metrics["macro_f1"], 4),
            "val_mae_grade":     round(val_metrics["mae_grade"], 4),
            "val_grade1_recall": round(
                val_metrics["grade1_one_vs_rest"]["recall"], 4),
            "val_grade1_f1":     round(
                val_metrics["grade1_one_vs_rest"]["f1"], 4),
        }
        history.append(row)
        save_json({"history": history}, outdir / "history.json")

        print(
            f"  Ep {epoch:03d} | "
            f"loss={row['train_loss']:.4f} | "
            f"bacc={row['val_balanced_acc']:.4f} | "
            f"f1={row['val_macro_f1']:.4f} | "
            f"g1_recall={row['val_grade1_recall']:.4f} | "
            f"g1_f1={row['val_grade1_f1']:.4f}"
        )

        if val_score > best_val:
            best_val = val_score
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config":           asdict(cfg),
                "best_val_balanced_accuracy": best_val,
                "epoch":            epoch,
            }, best_path)
            print(f"         ✓ best ({best_val:.4f}) — saved")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # Test set evaluation
    print(f"\n{'─'*60}")
    print("  Test set evaluation (best checkpoint)")
    print(f"{'─'*60}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    save_json(test_metrics, outdir / "test_metrics.json")

    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
    print(f"  MAE (grade)      : {test_metrics['mae_grade']:.4f}")
    print(f"  Grade-1 recall   : "
          f"{test_metrics['grade1_one_vs_rest']['recall']:.4f}")
    print(f"  Grade-1 F1       : "
          f"{test_metrics['grade1_one_vs_rest']['f1']:.4f}")
    print(f"\n  Results saved → {outdir}/test_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
