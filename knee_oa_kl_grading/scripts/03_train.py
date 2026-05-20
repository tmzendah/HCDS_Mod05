#!/usr/bin/env python3
"""
03_train.py - Train a single CNN model on MRKR cropped dataset (70/15/15 patient split).

Usage (single run on GPU node):
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/03_train.py --model resnet50 --seed 42

For all combos, use SLURM array job (provided separately).
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Paths (based on 00_setup)
# ============================================================================
RDS_BASE = "/rds/user/tm922/hpc-work"
MANIFEST_CSV = os.path.join(RDS_BASE, "data", "mrkr_cropped", "mrkr_cropped_manifest.csv")
PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
TEST_PATIENTS_CSV = os.path.join(RUNS_DIR, "test_patients.csv")  # save test set for later
os.makedirs(RUNS_DIR, exist_ok=True)

# Model input sizes (from literature, for MRKR dataset)
MODEL_SIZES = {
    'resnet50': 448,
    'densenet121': 384,
    'efficientnet_b0': 456
}

# ImageNet normalisation (standard for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Dataset with on-the-fly resizing & augmentation
# ============================================================================
class KneeOADataset(Dataset):
    def __init__(self, df, target_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.augment = augment

        # Base transform: resize → tensor → normalise
        self.base_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        # Mild augmentation for training only (standard for knee OA)
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['cropped_path']
        label = int(row['kl_grade'])

        # Open image as grayscale, then convert to RGB by channel duplication
        img = Image.open(img_path).convert('L')
        img = img.convert('RGB')   # standard practice for pretrained models

        if self.augment and self.aug_transform:
            img = self.aug_transform(img)

        img = self.base_transform(img)
        return img, label


# ============================================================================
# Ordinal loss (penalises distance between predicted and true grade)
# ============================================================================
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # Softmax applied internally to convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        batch_size = targets.size(0)
        total_loss = 0.0
        for i in range(batch_size):
            y = targets[i]
            k = torch.arange(self.num_classes, device=logits.device)
            distances = torch.abs(k - y.float())
            weighted_log_probs = distances * torch.log(probs[i] + 1e-8)
            loss_i = -weighted_log_probs.sum() / distances.sum()
            total_loss += loss_i
        return total_loss / batch_size


# ============================================================================
# Model factory (head replacement, full fine-tuning)
# ============================================================================
def get_model(model_name, num_classes=5):
    """Load pretrained model and replace classification head."""
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    # All layers are trainable (full fine-tuning) – no freezing
    return model


# ============================================================================
# Training utilities
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, kappa, bal_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, kappa, bal_acc, all_preds, all_labels


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed (e.g., 42, 123, 456)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Seed: {args.seed}")

    # 1. Load manifest
    if not os.path.exists(MANIFEST_CSV):
        print(f"ERROR: Manifest not found: {MANIFEST_CSV}")
        sys.exit(1)
    df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded {len(df)} cropped images")

    # 2. Patient-level split: 70% train, 15% val, 15% test
    patients = df['patient_id'].unique()
    train_patients, temp_patients = train_test_split(
        patients, test_size=0.3, random_state=args.seed, stratify=None
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=args.seed, stratify=None
    )

    train_df = df[df['patient_id'].isin(train_patients)]
    val_df   = df[df['patient_id'].isin(val_patients)]
    test_df  = df[df['patient_id'].isin(test_patients)]

    print(f"Train patients: {len(train_patients)}, images: {len(train_df)}")
    print(f"Val patients:   {len(val_patients)}, images: {len(val_df)}")
    print(f"Test patients:  {len(test_patients)}, images: {len(test_df)}")

    # Save test patients CSV for evaluation script (only once, from first seed)
    if args.seed == 42 and not os.path.exists(TEST_PATIENTS_CSV):
        pd.Series(test_patients).to_csv(TEST_PATIENTS_CSV, index=False, header=['patient_id'])
        print(f"Saved test patient IDs to {TEST_PATIENTS_CSV}")

    # 3. Datasets and loaders
    target_size = MODEL_SIZES[args.model]
    train_dataset = KneeOADataset(train_df, target_size, augment=True)
    val_dataset   = KneeOADataset(val_df,   target_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)

    # 4. Model, loss, optimizer, scheduler
    model = get_model(args.model, num_classes=5).to(device)
    criterion = OrdinalLoss(num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=3)

    # 5. Training loop (saves best checkpoint by validation Kappa)
    best_val_kappa = -1.0
    best_epoch = 0
    history = {'train_loss': [], 'train_kappa': [], 'val_loss': [], 'val_kappa': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_kappa, train_bal_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_kappa, val_bal_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(val_kappa)

        print(f"Train: Loss={train_loss:.4f}, Kappa={train_kappa:.4f}, BalAcc={train_bal_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Kappa={val_kappa:.4f}, BalAcc={val_bal_acc:.4f}")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_bal_acc': val_bal_acc,
                'seed': args.seed
            }
            torch.save(checkpoint, os.path.join(RUNS_DIR, f"{args.model}_seed{args.seed}_best.pth"))
            print(f"  -> Saved new best model (Kappa={val_kappa:.4f})")

    # 6. Save training history
    history_path = os.path.join(RUNS_DIR, f"{args.model}_seed{args.seed}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed. Best val Kappa: {best_val_kappa:.4f} at epoch {best_epoch}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
