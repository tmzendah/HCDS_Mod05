#!/usr/bin/env python3
"""
04_evaluate.py - Evaluate trained models on MRKR test set and external OAI/Kaggle test set.

Generates:
  - Performance tables with bootstrap 95% CIs (QWK, balanced accuracy, MAE, macro F1)
  - Confusion matrices and pairwise misclassification counts
  - McNemar statistical tests between models (Bonferroni p<0.017)
  - Grad-CAM heatmaps for sample test images
  - Subgroup fairness analysis (using MRKR demographics)

Usage:
  conda activate OAIKaggle
  cd ~/mrkr_klg
  python code/04_evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (cohen_kappa_score, balanced_accuracy_score,
                             mean_absolute_error, f1_score, confusion_matrix)
from scipy.stats import chi2_contingency
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Paths (from 00_setup)
# ============================================================================
RDS_BASE = "/rds/user/tm922/hpc-work"
PROJECT_DIR = os.path.expanduser("~/mrkr_klg")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
TEST_PATIENTS_CSV = os.path.join(RUNS_DIR, "test_patients.csv")
MANIFEST_CSV = os.path.join(RDS_BASE, "data", "mrkr_cropped", "mrkr_cropped_manifest.csv")
MRKR_DEMOGRAPHICS = os.path.join(RDS_BASE, "data", "mrkr", "MRKR_demographics.csv")
OAI_TEST_DIR = os.path.join(RDS_BASE, "data", "knee_oa", "test")
OAI_LABELS_CSV = os.path.join(RDS_BASE, "data", "knee_oa", "test_labels.csv")  # columns: image_path, kl_grade
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_SIZES = {'resnet50': 448, 'densenet121': 384, 'efficientnet_b0': 456}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Dataset (no augmentation)
# ============================================================================
class KneeOADataset(Dataset):
    def __init__(self, df, target_size):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['cropped_path']
        label = int(row['kl_grade'])
        img = Image.open(img_path).convert('L').convert('RGB')
        img = self.transform(img)
        return img, label, img_path


# ============================================================================
# Model loader
# ============================================================================
def load_model(model_name, checkpoint_path):
    """Load model architecture and best checkpoint weights."""
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, 5)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Bootstrap confidence intervals
# ============================================================================
def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def _bal_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def _mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.95):
    """
    Bootstrap 95% CI for a metric.
    Returns: (observed, ci_lower, ci_upper, bootstrap_mean, bootstrap_std)
    """
    observed = metric_func(y_true, y_pred)
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            scores.append(metric_func(y_true[idx], y_pred[idx]))
        except Exception:
            scores.append(np.nan)
    scores = np.array(scores)
    lower = np.nanpercentile(scores, (1 - alpha) / 2 * 100)
    upper = np.nanpercentile(scores, (1 - (1 - alpha) / 2) * 100)
    return observed, lower, upper, np.nanmean(scores), np.nanstd(scores)


# ============================================================================
# Evaluation functions
# ============================================================================
def evaluate_model(model, dataloader):
    """Run inference; return predictions, labels, and image paths."""
    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    return np.array(all_preds), np.array(all_labels), all_paths


def compute_metrics(y_true, y_pred, n_bootstrap=1000):
    """Return all metrics with bootstrap CIs."""
    metrics = {}
    for name, func in [('kappa', _qwk), ('balanced_accuracy', _bal_acc),
                       ('mae', _mae), ('macro_f1', _macro_f1)]:
        obs, lo, hi, bmean, bstd = bootstrap_ci(y_true, y_pred, func, n_bootstrap)
        metrics[name] = {'value': obs, 'ci_lower': lo, 'ci_upper': hi,
                         'bootstrap_mean': bmean, 'bootstrap_std': bstd}
    errors = y_true != y_pred
    distances = np.abs(y_true[errors] - y_pred[errors])
    metrics['confusion_matrix']    = confusion_matrix(y_true, y_pred, labels=range(5))
    metrics['adjacent_errors']     = int(np.sum(distances == 1))
    metrics['non_adjacent_errors'] = int(np.sum(distances >= 2))
    metrics['total_errors']        = int(errors.sum())
    return metrics


def plot_confusion_matrix(cm, model_name, dataset_name, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted KL Grade')
    plt.ylabel('True KL Grade')
    plt.title(f'{model_name} — {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# McNemar test
# ============================================================================
def mcnemar_test(y_true, preds_m1, preds_m2):
    """McNemar's test for paired predictions."""
    both_correct  = np.sum((preds_m1 == y_true) & (preds_m2 == y_true))
    both_wrong    = np.sum((preds_m1 != y_true) & (preds_m2 != y_true))
    only_m1_wrong = np.sum((preds_m1 != y_true) & (preds_m2 == y_true))
    only_m2_wrong = np.sum((preds_m1 == y_true) & (preds_m2 != y_true))
    table = np.array([[both_correct, only_m2_wrong],
                      [only_m1_wrong, both_wrong]])
    return chi2_contingency(table, correction=True).pvalue


# ============================================================================
# Grad-CAM (hook-based)
# ============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o))
        target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        grads = self.gradients.detach().cpu().numpy()[0]
        acts  = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.maximum(np.einsum('c,chw->hw', weights, acts), 0)
        cam = cam / (cam.max() + 1e-8)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        return zoom(cam, (h / cam.shape[0], w / cam.shape[1]))


def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return model.layer4[-1]
    elif model_name == 'densenet121':
        return model.features[-1]
    elif model_name == 'efficientnet_b0':
        return model.features[-1]
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Evaluation with Bootstrap 95% CIs")
    print("=" * 60)

    # 1. Load MRKR test set
    if not os.path.exists(TEST_PATIENTS_CSV):
        print(f"ERROR: {TEST_PATIENTS_CSV} not found. Run training with seed=42 first.")
        sys.exit(1)
    test_patients = pd.read_csv(TEST_PATIENTS_CSV)['patient_id'].values
    df_full  = pd.read_csv(MANIFEST_CSV)
    test_df  = df_full[df_full['patient_id'].isin(test_patients)]
    print(f"MRKR test set: {len(test_df)} images")

    # 2. Load OAI/Kaggle test set
    if not os.path.exists(OAI_LABELS_CSV):
        print(f"ERROR: {OAI_LABELS_CSV} not found.")
        sys.exit(1)
    oai_df = pd.read_csv(OAI_LABELS_CSV)
    oai_df['cropped_path'] = oai_df['image_path'].apply(
        lambda x: os.path.join(OAI_TEST_DIR, x))
    oai_df = oai_df[['cropped_path', 'kl_grade']]
    print(f"OAI/Kaggle test set: {len(oai_df)} images")

    # 3. Evaluate each model
    model_names = ['resnet50', 'densenet121', 'efficientnet_b0']
    seed = 42
    results_mrkr  = {}
    results_oai   = {}
    predictions_mrkr = {}
    y_true_ref    = None
    all_rows      = []

    for model_name in model_names:
        ckpt_path = os.path.join(RUNS_DIR, f"{model_name}_seed{seed}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path} — skipping.")
            continue

        print(f"\nEvaluating {model_name}...")
        model = load_model(model_name, ckpt_path)
        target_size = MODEL_SIZES[model_name]

        # MRKR test
        loader = DataLoader(KneeOADataset(test_df, target_size),
                            batch_size=64, shuffle=False, num_workers=4)
        y_pred, y_true, _ = evaluate_model(model, loader)
        metrics = compute_metrics(y_true, y_pred)
        results_mrkr[model_name] = metrics
        predictions_mrkr[model_name] = y_pred
        y_true_ref = y_true

        plot_confusion_matrix(metrics['confusion_matrix'], model_name, 'MRKR Test',
                              os.path.join(FIGURES_DIR, f'{model_name}_MRKR_cm.png'))

        # OAI/Kaggle test
        oai_loader = DataLoader(KneeOADataset(oai_df, target_size),
                                batch_size=64, shuffle=False, num_workers=4)
        y_pred_oai, y_true_oai, _ = evaluate_model(model, oai_loader)
        metrics_oai = compute_metrics(y_true_oai, y_pred_oai)
        results_oai[model_name] = metrics_oai

        plot_confusion_matrix(metrics_oai['confusion_matrix'], model_name, 'OAI/Kaggle Test',
                              os.path.join(FIGURES_DIR, f'{model_name}_OAI_cm.png'))

        print(f"  MRKR: QWK={metrics['kappa']['value']:.3f} "
              f"[{metrics['kappa']['ci_lower']:.3f}–{metrics['kappa']['ci_upper']:.3f}], "
              f"BalAcc={metrics['balanced_accuracy']['value']:.3f}, "
              f"MAE={metrics['mae']['value']:.3f}")
        print(f"  OAI:  QWK={metrics_oai['kappa']['value']:.3f} "
              f"[{metrics_oai['kappa']['ci_lower']:.3f}–{metrics_oai['kappa']['ci_upper']:.3f}], "
              f"BalAcc={metrics_oai['balanced_accuracy']['value']:.3f}, "
              f"MAE={metrics_oai['mae']['value']:.3f}")

        # Collect rows for output CSV
        for dataset, m in [('MRKR_test', metrics), ('OAI_test', metrics_oai)]:
            for metric_name in ['kappa', 'balanced_accuracy', 'mae', 'macro_f1']:
                all_rows.append({
                    'model': model_name, 'dataset': dataset, 'metric': metric_name,
                    'value':           m[metric_name]['value'],
                    'ci_lower':        m[metric_name]['ci_lower'],
                    'ci_upper':        m[metric_name]['ci_upper'],
                    'bootstrap_mean':  m[metric_name]['bootstrap_mean'],
                    'bootstrap_std':   m[metric_name]['bootstrap_std'],
                })
            for metric_name in ['adjacent_errors', 'non_adjacent_errors']:
                all_rows.append({
                    'model': model_name, 'dataset': dataset, 'metric': metric_name,
                    'value': m[metric_name],
                    'ci_lower': np.nan, 'ci_upper': np.nan,
                    'bootstrap_mean': np.nan, 'bootstrap_std': np.nan,
                })

    # 4. McNemar pairwise tests on MRKR test set
    if len(predictions_mrkr) >= 2:
        print("\nMcNemar's test (Bonferroni threshold p<0.017):")
        m_names = list(predictions_mrkr.keys())
        for i in range(len(m_names)):
            for j in range(i + 1, len(m_names)):
                pval = mcnemar_test(y_true_ref,
                                    predictions_mrkr[m_names[i]],
                                    predictions_mrkr[m_names[j]])
                sig = " *" if pval < 0.017 else ""
                print(f"  {m_names[i]} vs {m_names[j]}: p={pval:.4f}{sig}")

    # 5. Grad-CAM on 5 MRKR test images
    print("\nGenerating Grad-CAM heatmaps (5 images)...")
    best_model_name = max(results_mrkr, key=lambda m: results_mrkr[m]['kappa']['value'],
                          default='resnet50')
    ckpt_path = os.path.join(RUNS_DIR, f"{best_model_name}_seed{seed}_best.pth")
    if os.path.exists(ckpt_path):
        model = load_model(best_model_name, ckpt_path)
        grad_cam = GradCAM(model, get_target_layer(model, best_model_name))
        sample_ds = KneeOADataset(test_df.head(5), MODEL_SIZES[best_model_name])
        for idx, (img_tensor, label, img_path) in enumerate(sample_ds):
            img_tensor = img_tensor.unsqueeze(0).to(device)
            cam = grad_cam.generate(img_tensor)
            orig = Image.open(img_path).convert('L').resize(
                (MODEL_SIZES[best_model_name], MODEL_SIZES[best_model_name]))
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title(f'True KL={label}')
            axes[0].axis('off')
            axes[1].imshow(orig, cmap='gray')
            axes[1].imshow(cam, cmap='jet', alpha=0.5)
            axes[1].set_title('Grad-CAM')
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'gradcam_{idx}.png'), dpi=150)
            plt.close()
        print("Grad-CAM images saved.")

    # 6. Subgroup fairness analysis
    if os.path.exists(MRKR_DEMOGRAPHICS):
        demo = pd.read_csv(MRKR_DEMOGRAPHICS)
        best = max(results_mrkr, key=lambda m: results_mrkr[m]['kappa']['value'])
        test_with_demo = test_df.merge(demo, left_on='patient_id', right_on='empi_anon', how='left')
        test_with_demo['pred'] = predictions_mrkr[best]
        test_with_demo['age_group'] = pd.cut(
            test_with_demo['age'], bins=[0, 40, 50, 60, 70, 200],
            labels=['<40', '40-49', '50-59', '60-69', '70+']
        )
        subgroups = {
            'sex': 'sex', 'race': 'race',
            'weightbearing': 'weightbearing_status',
            'laterality': 'side', 'age_group': 'age_group',
        }
        fairness_rows = []
        for group_name, col in subgroups.items():
            for val in test_with_demo[col].dropna().unique():
                mask = test_with_demo[col] == val
                y_t = test_with_demo.loc[mask, 'kl_grade'].values
                y_p = test_with_demo.loc[mask, 'pred'].values
                if len(y_t) < 10:
                    continue
                grade1_mask = y_t == 1
                grade1_recall = (
                    (y_p[grade1_mask] == 1).sum() / grade1_mask.sum()
                    if grade1_mask.sum() > 0 else np.nan
                )
                fairness_rows.append({
                    'subgroup': group_name, 'value': val, 'n': len(y_t),
                    'balanced_accuracy': balanced_accuracy_score(y_t, y_p),
                    'grade1_recall': grade1_recall,
                })
        fairness_df = pd.DataFrame(fairness_rows)
        fairness_path = os.path.join(RESULTS_DIR, 'fairness_analysis.csv')
        fairness_df.to_csv(fairness_path, index=False)
        print(f"Fairness analysis saved to {fairness_path}")
    else:
        print(f"Demographics file not found: {MRKR_DEMOGRAPHICS} — skipping fairness analysis.")

    # 7. Save results table
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results_with_ci.csv')
    pd.DataFrame(all_rows).to_csv(results_path, index=False)
    print(f"\nResults with bootstrap CIs saved to {results_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
