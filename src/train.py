"""
src/train.py
Main training script for the 2x2 KL grading experiment.

Experiment design:
------------------
    2 architectures x 2 loss functions x 3 seeds = 12 training runs

    Architectures : resnet50, efficientnet
    Loss functions: ce, coral
    Seeds         : 42, 123, 456

Training configuration:
-----------------------
    Optimiser       : Adam, lr=0.0001, weight_decay=1e-4
    Scheduler       : ReduceLROnPlateau, factor=0.1, patience=3
    Early stopping  : patience=10 (monitors validation loss)
    Max epochs      : 50
    Batch size      : 32

Overfitting controls:
---------------------
    1. Early stopping (patience=7)
       Stops training if val loss does not improve for 7 epochs.
       Best checkpoint saved automatically.

    2. ReduceLROnPlateau (patience=3, factor=0.1)
       Reduces lr by 10x if val loss does not improve for 3 epochs.
       Gives model a chance to escape plateau before early stopping.

    3. Weight decay 1e-4 (L2 regularisation)
       Applied via Adam optimiser. Penalises large weights,
       discourages overfitting without architecture changes.
       Standard for fine-tuning pretrained CNNs in medical imaging.

Outputs (per run):
------------------
    results/checkpoints/{arch}_{loss}_seed{seed}.pth  -- best model
    results/metrics/{arch}_{loss}_seed{seed}_history.json  -- training history
    results/metrics/{arch}_{loss}_seed{seed}_config.json   -- run config

Usage
-----
    python src/train.py \
        --arch resnet50 \
        --loss ce \
        --seed 42 \
        --data_dir /path/to/your/data/knee_oa \
        --output_dir results \
        --epochs 30

Smoke test (CPU, 2 epochs, small data):
    python src/train.py \
        --arch resnet50 \
        --loss ce \
        --seed 42 \
        --data_dir /path/to/your/data/knee_oa \
        --output_dir results \
        --epochs 2 \
        --smoke_test
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score

from dataset import get_dataloaders
from losses  import get_loss_function, coral_predict
from models  import get_model


# ─────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fix all random seeds for full reproducibility.
    Must be called before any model, dataloader, or tensor creation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic CUDA operations
    # Small performance cost but required for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


# ─────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────

def get_predictions(outputs: torch.Tensor,
                    loss_name: str) -> torch.Tensor:
    """
    Convert model outputs to predicted KL grades.

    CE loss  : argmax of 5-class softmax logits
    CORAL    : count of rank boundaries with sigmoid > 0.5

    Args:
        outputs   : raw model outputs [batch, output_size]
        loss_name : 'ce' or 'coral'

    Returns:
        predicted grades [batch], integer values in {0,1,2,3,4}
    """
    if loss_name == "ce":
        return outputs.argmax(dim=1)
    elif loss_name == "coral":
        return coral_predict(outputs)


# ─────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    loss_name: str,
) -> dict:
    """
    Run one training epoch.

    Returns:
        dict with 'loss' and 'balanced_acc'
    """
    model.train()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    n_batches   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping -- prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        preds = get_predictions(outputs.detach(), loss_name)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1

    avg_loss    = total_loss / n_batches
    bal_acc     = balanced_accuracy_score(all_labels, all_preds)

    return {"loss": avg_loss, "balanced_acc": bal_acc}


# ─────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────

def validate(
    model,
    loader,
    criterion,
    device,
    loss_name: str,
) -> dict:
    """
    Run one validation epoch.

    Returns:
        dict with 'loss' and 'balanced_acc'
    """
    model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    n_batches   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds = get_predictions(outputs, loss_name)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            n_batches += 1

    avg_loss = total_loss / n_batches
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)

    return {"loss": avg_loss, "balanced_acc": bal_acc}


# ─────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────

def train(args) -> None:
    """
    Full training pipeline for one experimental configuration.
    """

    # ── Setup ──────────────────────────────────────────────
    run_name = f"{args.arch}_{args.loss}_seed{args.seed}"
    print("=" * 60)
    print(f"  Training run: {run_name}")
    print("=" * 60)
    print(f"  Architecture : {args.arch}")
    print(f"  Loss         : {args.loss}")
    print(f"  Seed         : {args.seed}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  ES patience  : {args.es_patience}")
    print(f"  LR patience  : {args.lr_patience}")
    print(f"  Smoke test   : {args.smoke_test}")

    # Fix seeds
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Output directories
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    metrics_dir    = Path(args.output_dir) / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────
    print("\n  Loading data...")
    num_workers = 0 if args.smoke_test else 4
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        seed        = args.seed,
        num_workers = num_workers,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # Smoke test -- use only first 5 batches per split
    if args.smoke_test:
        from torch.utils.data import Subset
        train_loader.dataset.samples = (
            train_loader.dataset.samples[:5 * args.batch_size]
        )
        val_loader.dataset.samples = (
            val_loader.dataset.samples[:5 * args.batch_size]
        )
        print("  [SMOKE TEST] Using 5 batches per split")

    # ── Model ──────────────────────────────────────────────
    print("\n  Building model...")
    model = get_model(args.arch, args.loss)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss, optimiser, scheduler ─────────────────────────
    criterion = get_loss_function(args.loss)
    criterion = criterion.to(device) if hasattr(criterion, "to") else criterion

    optimizer = Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode      = "min",      # monitor validation loss (lower is better)
        factor    = 0.1,        # reduce lr by 10x
        patience  = args.lr_patience,
        min_lr    = 1e-7,
    )

    # ── Training loop ──────────────────────────────────────
    history = {
        "train_loss":     [],
        "train_bal_acc":  [],
        "val_loss":       [],
        "val_bal_acc":    [],
        "lr":             [],
    }

    best_val_loss    = float("inf")
    es_counter       = 0
    best_epoch       = 0
    checkpoint_path  = checkpoint_dir / f"{run_name}.pth"

    print(f"\n  Starting training...")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train BAcc':>10}  "
          f"{'Val Loss':>10}  {'Val BAcc':>10}  {'LR':>10}  {'':>6}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, args.loss
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion,
            device, args.loss
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_bal_acc"].append(train_metrics["balanced_acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_bal_acc"].append(val_metrics["balanced_acc"])
        history["lr"].append(current_lr)

        # Check improvement
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch    = epoch
            es_counter    = 0
            torch.save({
                "epoch":      epoch,
                "arch":       args.arch,
                "loss_name":  args.loss,
                "seed":       args.seed,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   best_val_loss,
                "val_bal_acc": val_metrics["balanced_acc"],
            }, checkpoint_path)
            flag = "  SAVED"
        else:
            es_counter += 1
            flag = f"  patience {es_counter}/{args.es_patience}"

        # Print epoch summary
        print(
            f"  {epoch:>6}  "
            f"{train_metrics['loss']:>10.4f}  "
            f"{train_metrics['balanced_acc']:>10.4f}  "
            f"{val_metrics['loss']:>10.4f}  "
            f"{val_metrics['balanced_acc']:>10.4f}  "
            f"{current_lr:>10.2e}  "
            f"{flag}"
        )

        # Early stopping
        if es_counter >= args.es_patience:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed/60:.1f} minutes.")
    print(f"  Best val loss  : {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint     : {checkpoint_path}")

    # ── Save history and config ─────────────────────────────
    history_path = metrics_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved  : {history_path}")

    config = {
        "run_name":     run_name,
        "arch":         args.arch,
        "loss":         args.loss,
        "seed":         args.seed,
        "epochs_run":   epoch,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "es_patience":  args.es_patience,
        "lr_patience":  args.lr_patience,
        "data_dir":     str(args.data_dir),
    }
    config_path = metrics_dir / f"{run_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved   : {config_path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train one configuration of the 2x2 KL grading experiment"
    )

    # Required
    p.add_argument("--arch",      type=str, required=True,
                   choices=["resnet50", "efficientnet"],
                   help="Model architecture")
    p.add_argument("--loss",      type=str, required=True,
                   choices=["ce", "coral"],
                   help="Loss function")
    p.add_argument("--seed",      type=int, required=True,
                   choices=[42, 123, 456],
                   help="Random seed")
    p.add_argument("--data_dir",  type=str, required=True,
                   help="Path to dataset root (train/ val/ test/)")
    p.add_argument("--output_dir",type=str, required=True,
                   help="Where to save checkpoints and metrics")

    # Optional with defaults
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--es_patience",  type=int,   default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--lr_patience",  type=int,   default=3,
                   help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--smoke_test",   action="store_true",
                   help="Run 2 epochs on 5 batches to verify pipeline")

    return p.parse_args()


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
