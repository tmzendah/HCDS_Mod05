"""MRKR Training Script v3 — Ordinal Loss + Architecture-Specific Sizes

Key design decisions:
  - Ordinal loss: penalises predictions by grade distance
    e.g. predicting KL4 for true KL0 penalised 4x more than KL1 for KL0
  - Architecture-specific image sizes (Gour et al. 2023):
      ResNet50        -> 448px
      DenseNet121     -> 384px
      EfficientNet-B0 -> 456px
  - Augmentation matches Kaggle training minus RandomResizedCrop:
      RandomHorizontalFlip(p=0.5)
      RandomRotation(degrees=10)
      ColorJitter(brightness=0.2, contrast=0.2)
  - lr=1e-4 standard for ImageNet fine-tuning
  - Patient-level 70/15/15 split prevents data leakage
  - Three seeds 42, 123, 456 for stability reporting

References:
  - Ordinal loss: Frank & Hall (2001), adapted for deep learning
  - Architecture sizes: Gour et al. (2023) Scientific Reports
  - CLAHE: Yaylu et al. (2025)

Usage:
  python mrkr_train_v3.py \
      --data_csv   /rds/user/tm922/hpc-work/data/mrkr_cropped/mrkr_cropped_manifest.csv \
      --img_root   /rds/user/tm922/hpc-work/data/mrkr_cropped \
      --output_dir /home/tm922/mrkr_klg/runs/v3_resnet50_seed42 \
      --model      resnet50 \
      --seed       42 \
      --epochs     30
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-specific image sizes (Gour et al. 2023)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_IMG_SIZES = {
    "resnet50":        448,
    "densenet121":     384,
    "efficientnet_b0": 456,
}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal Loss
# ─────────────────────────────────────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    """Ordinal cross-entropy loss for KL grade classification.

    Penalises predictions proportionally to grade distance.
    Normalised so loss scale matches standard CrossEntropy (~1.5-2.0).

    Formula:
        OrdinalLoss = mean over batch of:
            -sum_k( |k-y| * log(p_k) ) / sum_k( |k-y| )

    Weight matrix W[i][j] = |i - j| where i=true grade, j=predicted grade.

    Reference: Frank & Hall (2001), adapted for deep learning.
    """

    def __init__(self, num_classes=5):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        W = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                W[i][j] = abs(i - j)
        self.register_buffer("W", W)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        weights = self.W[targets]
        log_probs = torch.log(probs + 1e-8)
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        loss = -(weights * log_probs / weight_sum).sum(dim=1).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MRKRDataset(Dataset):
    """MRKR knee radiograph dataset."""

    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, str(row["png_path"]))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name, num_classes):
    name = name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(img_size):
    """Augmentation matches Kaggle training minus RandomResizedCrop."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, eval_tfms


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level split
# ─────────────────────────────────────────────────────────────────────────────

def patient_stratified_split(df, seed, val_size=0.15, test_size=0.15):
    """Split at patient level to prevent data leakage."""
    patient_grade = (
        df.groupby("empi_anon")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    patient_grade.columns = ["empi_anon", "majority_label"]

    train_pats, temp = train_test_split(
        patient_grade,
        test_size=(val_size + test_size),
        stratify=patient_grade["majority_label"],
        random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    val_pats, test_pats = train_test_split(
        temp,
        test_size=rel_test,
        stratify=temp["majority_label"],
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
# Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.max(1)[1]
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.max(1)[1]
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true != 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred != 1)).sum())
    tn = int(((y_true != 1) & (y_pred != 1)).sum())
    g1_prec = tp / (tp + fp) if (tp + fp) else 0.0
    g1_rec  = tp / (tp + fn) if (tp + fn) else 0.0
    g1_f1   = (2 * g1_prec * g1_rec / (g1_prec + g1_rec)
               if (g1_prec + g1_rec) else 0.0)

    metrics = {
        "loss":              running_loss / len(loader),
        "accuracy":          float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae_grade":         float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "preds":             y_pred.tolist(),
        "targets":           y_true.tolist(),
        "probs":             all_probs,
        "grade1_one_vs_rest": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(g1_prec, 4),
            "recall":    round(g1_rec,  4),
            "f1":        round(g1_f1,   4),
        },
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serialisable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_convert)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",     required=True)
    p.add_argument("--img_root",     required=True)
    p.add_argument("--output_dir",   default="runs/v3_baseline")
    p.add_argument("--model",        default="resnet50",
                   choices=["resnet50", "densenet121", "efficientnet_b0"])
    p.add_argument("--num_classes",  type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--val_size",     type=float, default=0.15)
    p.add_argument("--test_size",    type=float, default=0.15)
    p.add_argument("--patience",     type=int,   default=7)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    img_size = MODEL_IMG_SIZES.get(args.model, 224)

    print("=" * 60)
    print(f"MRKR Training v3 — {args.model.upper()}")
    print(f"  Loss       : Ordinal (distance-penalised, normalised)")
    print(f"  Image size : {img_size}px")
    print(f"  lr         : {args.lr}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Patience   : {args.patience}")
    print(f"  Seed       : {args.seed}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    df = pd.read_csv(args.data_csv)
    print(f"\n  Manifest : {len(df):,} rows")
    print(f"  Labels   : {df['label'].value_counts().sort_index().to_dict()}")

    train_df, val_df, test_df = patient_stratified_split(
        df, args.seed, args.val_size, args.test_size)

    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir,   "val_split.csv"),   index=False)
    test_df.to_csv(os.path.join(args.output_dir,  "test_split.csv"),  index=False)

    train_tfms, eval_tfms = build_transforms(img_size)

    train_ds = MRKRDataset(train_df, args.img_root, train_tfms)
    val_ds   = MRKRDataset(val_df,   args.img_root, eval_tfms)
    test_ds  = MRKRDataset(test_df,  args.img_root, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    model     = build_model(args.model, args.num_classes).to(device)
    criterion = OrdinalLoss(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_val  = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")
    patience  = 0
    history   = []

    print(f"\n  Training up to {args.epochs} epochs (patience={args.patience})")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_score   = val_metrics["balanced_accuracy"]
        scheduler.step(val_score)

        row = {
            "epoch":            epoch,
            "train_loss":       round(train_loss, 4),
            "train_acc":        round(train_acc, 2),
            "val_loss":         round(val_metrics["loss"], 4),
            "val_balanced_acc": round(val_score, 4),
            "val_macro_f1":     round(val_metrics["macro_f1"], 4),
            "val_mae":          round(val_metrics["mae_grade"], 4),
            "val_g1_recall":    round(val_metrics["grade1_one_vs_rest"]["recall"], 4),
            "val_g1_f1":        round(val_metrics["grade1_one_vs_rest"]["f1"], 4),
        }
        history.append(row)
        save_json({"history": history},
                  os.path.join(args.output_dir, "history.json"))

        print(f"  Ep {epoch:03d} | loss={row['train_loss']:.4f} | "
              f"bacc={row['val_balanced_acc']:.4f} | "
              f"f1={row['val_macro_f1']:.4f} | "
              f"g1_recall={row['val_g1_recall']:.4f}")

        if val_score > best_val:
            best_val = val_score
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "best_val_bacc":    best_val,
                "model_name":       args.model,
                "img_size":         img_size,
            }, best_path)
            print(f"         checkpoint saved (bacc={best_val:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n{'─'*60}")
    print("  Test set evaluation (best checkpoint)")
    print(f"{'─'*60}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
    print(f"  MAE grade        : {test_metrics['mae_grade']:.4f}")
    print(f"  Grade-1 recall   : {test_metrics['grade1_one_vs_rest']['recall']:.4f}")
    print(f"  Grade-1 F1       : {test_metrics['grade1_one_vs_rest']['f1']:.4f}")
    print(f"\n  Results -> {args.output_dir}/test_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
