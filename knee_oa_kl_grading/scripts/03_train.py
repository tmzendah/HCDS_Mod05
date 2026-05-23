#!/usr/bin/env python3
"""
03_train.py - Train a single CNN model on MRKR cropped dataset (70/15/15 patient split).
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
# Paths
# ============================================================================
MANIFEST_CSV = "/rds/user/tm922/hpc-work/data/mrkr_cropped/mrkr_cropped_manifest.csv"
PROJECT_DIR = os.path.expanduser("~/MLOAIProject")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
TEST_PATIENTS_CSV = os.path.join(RUNS_DIR, "test_patients.csv")
os.makedirs(RUNS_DIR, exist_ok=True)

# Model input sizes
MODEL_SIZES = {
    'resnet50': 448,
    'densenet121': 384,
    'efficientnet_b0': 456
}

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Dataset
# ============================================================================
class KneeOADataset(Dataset):
    def __init__(self, df, target_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
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
        img = Image.open(img_path).convert('L').convert('RGB')
        if self.augment and self.aug_transform:
            img = self.aug_transform(img)
        img = self.base_transform(img)
        return img, label

# ============================================================================
# Ordinal Loss
# ============================================================================
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
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
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Model: {args.model}, Seed: {args.seed}")

    # Load manifest
    if not os.path.exists(MANIFEST_CSV):
        print(f"ERROR: {MANIFEST_CSV} not found.")
        sys.exit(1)
    df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded {len(df)} images")

    # Patient-level split: 70 train, 15 val, 15 test
    patients = df['patient_id'].unique()
    train_pat, temp_pat = train_test_split(patients, test_size=0.3, random_state=args.seed)
    val_pat, test_pat = train_test_split(temp_pat, test_size=0.5, random_state=args.seed)

    train_df = df[df['patient_id'].isin(train_pat)]
    val_df   = df[df['patient_id'].isin(val_pat)]
    test_df  = df[df['patient_id'].isin(test_pat)]

    print(f"Train: {len(train_df)} images, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save test patients for evaluation (only once from seed 42)
    if args.seed == 42 and not os.path.exists(TEST_PATIENTS_CSV):
        pd.Series(test_pat).to_csv(TEST_PATIENTS_CSV, index=False, header=['patient_id'])

    # Data loaders
    target_size = MODEL_SIZES[args.model]
    train_dataset = KneeOADataset(train_df, target_size, augment=True)
    val_dataset   = KneeOADataset(val_df, target_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = get_model(args.model).to(device)
    criterion = OrdinalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_kappa = -1.0
    best_epoch = 0
    history = {'train_loss': [], 'train_kappa': [], 'val_loss': [], 'val_kappa': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_kappa, train_bal = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_kappa, val_bal, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(val_kappa)

        print(f"Epoch {epoch}/{args.epochs}: Train Loss={train_loss:.4f} Kappa={train_kappa:.4f} | Val Loss={val_loss:.4f} Kappa={val_kappa:.4f}")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_bal_acc': val_bal,
                'seed': args.seed
            }, os.path.join(RUNS_DIR, f"{args.model}_seed{args.seed}_best.pth"))
            print(f"  -> Best model saved (Kappa={val_kappa:.4f})")

    # Save history
    with open(os.path.join(RUNS_DIR, f"{args.model}_seed{args.seed}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training completed. Best val Kappa: {best_val_kappa:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
