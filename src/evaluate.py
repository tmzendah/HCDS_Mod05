"""
src/evaluate.py
Full evaluation suite for all trained model checkpoints.

Metrics computed (on test set):
--------------------------------
Primary outcome:
    - KL1 recall (sensitivity) -- core research question

Field standard:
    - Quadratic Weighted Kappa (QWK)

Overall performance:
    - Balanced accuracy
    - Macro AUC (one-vs-rest)

Per-grade (KL0-KL4):
    - Recall (sensitivity)
    - AUC (one-vs-rest)

Stability (computed from JSON outputs by results_analysis notebook):
    - Mean +- SD across 3 seeds per configuration
    - Wilcoxon signed-rank test CE vs CORAL per architecture

Outputs:
---------
    results/metrics/{run_name}_eval.json   -- full results per run
    results/metrics/summary.csv            -- all runs as one table

Usage
-----
    # Evaluate all checkpoints
    python src/evaluate.py \
        --data_dir /rds/user/tm922/hpc-work/data/knee_oa \
        --results_dir results

    # Evaluate one specific checkpoint
    python src/evaluate.py \
        --data_dir /rds/user/tm922/hpc-work/data/knee_oa \
        --results_dir results \
        --checkpoint results/checkpoints/resnet50_ce_seed42.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders
from losses  import coral_predict
from models  import get_model

NUM_CLASSES = 5
GRADES      = [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────
# QWK
# ─────────────────────────────────────────────────────────

def quadratic_weighted_kappa(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              n_classes: int = 5) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK is the primary metric in KL grading literature.
    It penalises distant grade errors more than adjacent ones,
    reflecting the ordinal nature of the KL scale.

    Range: -1 (perfect disagreement) to 1 (perfect agreement)
    Interpretation:
        > 0.81 = almost perfect
        0.61-0.80 = substantial
        0.41-0.60 = moderate
        < 0.40 = poor

    Args:
        y_true    : true KL grades, shape [n]
        y_pred    : predicted KL grades, shape [n]
        n_classes : number of classes (default 5)

    Returns:
        float QWK score
    """
    # Weight matrix -- quadratic penalties
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2

    # Confusion matrix (normalised)
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(n_classes)))
    cm = cm.astype(float)

    # Expected matrix
    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    expected  = np.outer(hist_true, hist_pred) / cm.sum()

    # QWK
    numerator   = (weights * cm).sum()
    denominator = (weights * expected).sum()

    if denominator == 0:
        return 0.0

    return float(1.0 - numerator / denominator)


# ─────────────────────────────────────────────────────────
# Per-grade AUC
# ─────────────────────────────────────────────────────────

def compute_per_grade_auc(y_true: np.ndarray,
                           y_prob: np.ndarray) -> dict:
    """
    Compute one-vs-rest AUC for each KL grade.

    Args:
        y_true : true labels, shape [n]
        y_prob : predicted probabilities, shape [n, 5]

    Returns:
        dict mapping grade to AUC score
    """
    y_bin = label_binarize(y_true, classes=GRADES)
    aucs  = {}

    for grade in GRADES:
        try:
            auc = roc_auc_score(y_bin[:, grade], y_prob[:, grade])
            aucs[grade] = float(auc)
        except ValueError:
            # Grade not present in test set
            aucs[grade] = None

    return aucs


# ─────────────────────────────────────────────────────────
# Get probabilities from model outputs
# ─────────────────────────────────────────────────────────

def get_probabilities(outputs: torch.Tensor,
                      loss_name: str) -> np.ndarray:
    """
    Convert raw model outputs to class probabilities.

    CE   : softmax over 5 logits -> [n, 5]
    CORAL: convert 4 boundary probabilities to grade probabilities -> [n, 5]

    For CORAL, grade probabilities are derived as:
        P(grade=0) = 1 - P(grade>0)
        P(grade=k) = P(grade>k-1) - P(grade>k)  for k=1,2,3
        P(grade=4) = P(grade>3)

    Args:
        outputs   : raw model outputs
        loss_name : 'ce' or 'coral'

    Returns:
        numpy array of shape [n, 5]
    """
    if loss_name == "ce":
        probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    elif loss_name == "coral":
        # Sigmoid probabilities for each rank boundary
        # shape: [n, 4]
        rank_probs = torch.sigmoid(outputs).cpu().numpy()

        n = rank_probs.shape[0]
        grade_probs = np.zeros((n, NUM_CLASSES))

        # P(grade=0) = 1 - P(grade>0)
        grade_probs[:, 0] = 1.0 - rank_probs[:, 0]

        # P(grade=k) = P(grade>k-1) - P(grade>k)
        for k in range(1, NUM_CLASSES - 1):
            grade_probs[:, k] = rank_probs[:, k-1] - rank_probs[:, k]

        # P(grade=4) = P(grade>3)
        grade_probs[:, 4] = rank_probs[:, 3]

        # Clip to [0,1] to handle floating point edge cases
        grade_probs = np.clip(grade_probs, 0, 1)

        return grade_probs


def get_predictions_numpy(outputs: torch.Tensor,
                           loss_name: str) -> np.ndarray:
    """Convert raw outputs to integer grade predictions."""
    if loss_name == "ce":
        return outputs.argmax(dim=1).cpu().numpy()
    elif loss_name == "coral":
        return coral_predict(outputs).cpu().numpy()


# ─────────────────────────────────────────────────────────
# Evaluate one checkpoint
# ─────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: str,
    metrics_dir: Path,
    device: torch.device,
) -> dict:
    """
    Load a checkpoint, run on test set, compute all metrics.

    Args:
        checkpoint_path : path to .pth file
        data_dir        : path to dataset root
        metrics_dir     : where to save results
        device          : torch device

    Returns:
        dict of all metrics
    """
    # ── Load checkpoint ────────────────────────────────────
    print(f"\n  Evaluating: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    arch      = checkpoint["arch"]
    loss_name = checkpoint["loss_name"]
    seed      = checkpoint["seed"]
    run_name  = f"{arch}_{loss_name}_seed{seed}"

    print(f"    arch={arch}  loss={loss_name}  seed={seed}")
    print(f"    Best val loss: {checkpoint['val_loss']:.4f} "
          f"at epoch {checkpoint['epoch']}")

    # ── Build model and load weights ───────────────────────
    model = get_model(arch, loss_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Load test data ─────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = 32,
        seed        = seed,
        num_workers = 4,
    )

    # ── Run inference ──────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            preds = get_predictions_numpy(outputs, loss_name)
            probs = get_probabilities(outputs, loss_name)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    # ── Compute metrics ────────────────────────────────────

    # Overall
    qwk      = quadratic_weighted_kappa(all_labels, all_preds)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)

    # Macro AUC
    try:
        macro_auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_auc = None

    # Per-grade recall
    per_grade_recall = recall_score(
        all_labels, all_preds,
        labels=GRADES, average=None,
        zero_division=0
    )

    # Per-grade AUC
    per_grade_auc = compute_per_grade_auc(all_labels, all_probs)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=GRADES)

    # ── Assemble results dict ──────────────────────────────
    results = {
        "run_name":       run_name,
        "arch":           arch,
        "loss":           loss_name,
        "seed":           seed,
        "n_test":         int(len(all_labels)),

        # Primary outcome
        "kl1_recall":     float(per_grade_recall[1]),

        # Overall metrics
        "qwk":            float(qwk),
        "balanced_acc":   float(bal_acc),
        "macro_auc":      float(macro_auc) if macro_auc else None,

        # Per-grade recall
        "recall_kl0":     float(per_grade_recall[0]),
        "recall_kl1":     float(per_grade_recall[1]),
        "recall_kl2":     float(per_grade_recall[2]),
        "recall_kl3":     float(per_grade_recall[3]),
        "recall_kl4":     float(per_grade_recall[4]),

        # Per-grade AUC
        "auc_kl0":        per_grade_auc[0],
        "auc_kl1":        per_grade_auc[1],
        "auc_kl2":        per_grade_auc[2],
        "auc_kl3":        per_grade_auc[3],
        "auc_kl4":        per_grade_auc[4],

        # Confusion matrix (for figures)
        "confusion_matrix": cm.tolist(),

        # Training info
        "best_epoch":     int(checkpoint["epoch"]),
        "best_val_loss":  float(checkpoint["val_loss"]),
    }

    # ── Print summary ──────────────────────────────────────
    print(f"\n    {'Metric':<20} {'Value':>10}")
    print(f"    {'-'*32}")
    print(f"    {'QWK':<20} {results['qwk']:>10.4f}")
    print(f"    {'Balanced Acc':<20} {results['balanced_acc']:>10.4f}")
    print(f"    {'Macro AUC':<20} {results['macro_auc']:>10.4f}")
    print(f"    {'KL1 Recall':<20} {results['kl1_recall']:>10.4f}  <- PRIMARY")
    print(f"    {'KL1 AUC':<20} {results['auc_kl1']:>10.4f}")
    print(f"\n    Per-grade recall:")
    for g in GRADES:
        marker = "  <- KL1" if g == 1 else ""
        print(f"      KL{g}: {per_grade_recall[g]:.4f}{marker}")

    # ── Save JSON ──────────────────────────────────────────
    json_path = metrics_dir / f"{run_name}_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved -> {json_path.name}")

    return results


# ─────────────────────────────────────────────────────────
# Save CSV summary
# ─────────────────────────────────────────────────────────

def save_summary_csv(all_results: list, metrics_dir: Path) -> None:
    """
    Save all evaluation results as a single CSV summary table.
    One row per run, all metrics as columns.
    Sorted by architecture, loss, seed.
    """
    # Flatten for CSV -- exclude confusion matrix
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "confusion_matrix"}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["arch", "loss", "seed"]).reset_index(drop=True)

    csv_path = metrics_dir / "summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Summary CSV saved -> {csv_path}")
    print(f"  Shape: {df.shape[0]} runs x {df.shape[1]} metrics")
    print(f"\n  Preview:")
    print(df[["run_name", "qwk", "balanced_acc",
              "macro_auc", "kl1_recall"]].to_string(index=False))


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate all trained checkpoints on the test set"
    )
    p.add_argument("--data_dir",    type=str, required=True,
                   help="Path to dataset root")
    p.add_argument("--results_dir", type=str, required=True,
                   help="Results directory (contains checkpoints/ and metrics/)")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Evaluate one specific checkpoint (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    results_dir    = Path(args.results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    metrics_dir    = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  evaluate.py -- Test set evaluation")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Results dir  : {results_dir}")

    # ── Find checkpoints ───────────────────────────────────
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print(f"\n  ERROR: No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        print(f"\n  Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(f"    {cp.name}")

    # ── Evaluate each checkpoint ───────────────────────────
    all_results = []
    for cp in checkpoints:
        try:
            result = evaluate_checkpoint(
                checkpoint_path = cp,
                data_dir        = args.data_dir,
                metrics_dir     = metrics_dir,
                device          = device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR evaluating {cp.name}: {e}")
            continue

    # ── Save summary CSV ───────────────────────────────────
    if len(all_results) > 1:
        save_summary_csv(all_results, metrics_dir)

    print("\n" + "=" * 60)
    print(f"  Evaluation complete. {len(all_results)} run(s) evaluated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
