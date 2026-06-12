"""
src/gradcam_audit.py
Grad-CAM attention audit for KL1 misclassification cases.

Purpose:
--------
Generates Grad-CAM heatmaps for KL1 misclassified cases across all
four model configurations (best seed per configuration).

Research question addressed:
-----------------------------
Does the training objective (categorical CE vs ordinal CORAL) change
where the model attends during KL1 misclassification?

Clinically plausible attention = activation on:
    - Medial joint space
    - Osteophyte margins
    - Subchondral bone
    - Tibial plateau

Clinically implausible attention = activation on:
    - Femoral shaft
    - Soft tissue margins
    - Image borders
    - Background regions

Audit approach:
---------------
1. Load best checkpoint per configuration (lowest val loss across seeds)
2. Run inference on full test set, identify all KL1 misclassifications
3. Find cases misclassified by ALL models (hardest cases) and
   cases misclassified by AT LEAST 2 models (shared failures)
4. Generate Grad-CAM overlays for all four models on the same images
5. Save side-by-side comparison figures for clinical audit
6. Compute quantitative localisation score (% activation in centre crop)

Outputs:
--------
    results/gradcam/
        shared_misclassified/     -- cases misclassified by 2+ models
            case_001_comparison.png
            case_002_comparison.png
            ...
        all_cases/                -- all KL1 misclassifications per model
            resnet50_ce/
            resnet50_coral/
            efficientnet_ce/
            efficientnet_coral/
        audit_scores.json         -- quantitative localisation scores
        audit_summary.png         -- summary figure for report

Usage
-----
    python src/gradcam_audit.py \
        --data_dir   /path/to/your/data/knee_oa \
        --results_dir results \
        --output_dir  results/gradcam
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from dataset import get_val_test_transforms, KneeOADataset
from losses  import coral_predict
from models  import get_model

# ── Constants ─────────────────────────────────────────────
NUM_CLASSES  = 5
GRADES       = [0, 1, 2, 3, 4]
KL1_LABEL    = 1
DPI          = 150

# Centre crop ratio for quantitative localisation score
# Joint space is typically in the central 50% of knee radiograph
CENTRE_CROP_RATIO = 0.5

KL_GRADE_NAMES = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe"
}

CONFIG_COLOURS = {
    "resnet50_ce":        "#4878d0",
    "resnet50_coral":     "#ee854a",
    "efficientnet_ce":    "#6acc65",
    "efficientnet_coral": "#d65f5f",
}


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def save(fig, path, dpi=DPI):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_target_layer(model, arch: str):
    """
    Returns the target layer for Grad-CAM.

    For ResNet50: last convolutional layer (layer4[-1])
    For EfficientNet-B0: last convolutional block (features[-1])

    The target layer should be the last layer before global
    average pooling — this gives the highest resolution feature
    maps while still being semantically meaningful.
    """
    if arch == "resnet50":
        return [model.layer4[-1]]
    elif arch == "efficientnet":
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown arch: {arch}")


def load_image_for_display(img_path: Path) -> np.ndarray:
    """
    Load image as float32 RGB array in [0,1] for display.
    Used as background for Grad-CAM overlay.
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def compute_localisation_score(cam: np.ndarray) -> float:
    """
    Quantitative attention localisation score.

    Measures what percentage of total Grad-CAM activation mass
    falls within the central crop of the image.

    The central crop approximates the joint space region in
    knee radiographs, which are pre-centred in the Kaggle dataset.

    Args:
        cam : Grad-CAM heatmap, shape [H, W], values in [0,1]

    Returns:
        float: proportion of activation in centre crop (0-1)
    """
    h, w = cam.shape
    h_start = int(h * (1 - CENTRE_CROP_RATIO) / 2)
    h_end   = int(h * (1 + CENTRE_CROP_RATIO) / 2)
    w_start = int(w * (1 - CENTRE_CROP_RATIO) / 2)
    w_end   = int(w * (1 + CENTRE_CROP_RATIO) / 2)

    centre_activation = cam[h_start:h_end, w_start:w_end].sum()
    total_activation  = cam.sum()

    if total_activation == 0:
        return 0.0

    return float(centre_activation / total_activation)


# ─────────────────────────────────────────────────────────
# Find best checkpoint per configuration
# ─────────────────────────────────────────────────────────

def find_best_checkpoints(results_dir: Path) -> dict:
    """
    For each of the 4 configurations, find the checkpoint
    with the lowest validation loss across seeds.

    Returns:
        dict mapping config_name -> checkpoint_path
    """
    checkpoint_dir = results_dir / "checkpoints"
    configs = [
        ("resnet50",     "ce"),
        ("resnet50",     "coral"),
        ("efficientnet", "ce"),
        ("efficientnet", "coral"),
    ]

    best = {}

    for arch, loss in configs:
        config_name = f"{arch}_{loss}"
        best_loss   = float("inf")
        best_path   = None

        for seed in [42, 123, 456]:
            cp_path = checkpoint_dir / f"{arch}_{loss}_seed{seed}.pth"
            if not cp_path.exists():
                continue
            cp = torch.load(cp_path, map_location="cpu")
            if cp["val_loss"] < best_loss:
                best_loss = cp["val_loss"]
                best_path = cp_path

        if best_path is not None:
            best[config_name] = best_path
            print(f"  {config_name}: {best_path.name} "
                  f"(val_loss={best_loss:.4f})")
        else:
            print(f"  {config_name}: NO CHECKPOINT FOUND")

    return best


# ─────────────────────────────────────────────────────────
# Run inference and collect KL1 misclassifications
# ─────────────────────────────────────────────────────────

def get_kl1_misclassifications(
    checkpoint_path: Path,
    data_dir: str,
    device: torch.device,
) -> dict:
    """
    Run inference on test set, return all KL1 misclassification cases.
    """
    cp        = torch.load(checkpoint_path, map_location=device)
    arch      = cp["arch"]
    loss_name = cp["loss_name"]

    model = get_model(arch, loss_name)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dataset = KneeOADataset(
        data_dir  = data_dir,
        split     = "test",
        transform = get_val_test_transforms(),
    )

    misclassified_indices = []
    misclassified_paths   = []
    predicted_labels      = []
    true_labels           = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)),
                        desc=f"  inference {arch}_{loss_name}",
                        leave=False):
            image, label = test_dataset[idx]

            if label != KL1_LABEL:
                continue

            image_tensor = image.unsqueeze(0).to(device)
            output       = model(image_tensor)

            if loss_name == "ce":
                pred = output.argmax(dim=1).item()
            else:
                pred = coral_predict(output).item()

            if pred != KL1_LABEL:
                img_path, _ = test_dataset.samples[idx]
                misclassified_indices.append(idx)
                misclassified_paths.append(img_path)
                predicted_labels.append(pred)
                true_labels.append(label)

    return {
        "arch":                  arch,
        "loss_name":             loss_name,
        "misclassified_indices": misclassified_indices,
        "misclassified_paths":   misclassified_paths,
        "predicted_labels":      predicted_labels,
        "true_labels":           true_labels,
        "n_misclassified":       len(misclassified_indices),
        "n_kl1_total":           sum(1 for _, l in test_dataset.samples
                                     if l == KL1_LABEL),
    }


# ─────────────────────────────────────────────────────────
# Generate Grad-CAM for one image and one model
# ─────────────────────────────────────────────────────────

def generate_gradcam(
    model,
    arch: str,
    loss_name: str,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for one image.
    """
    target_layers = get_target_layer(model, arch)

    # For CORAL, target boundary 0 = P(grade>0), most relevant for KL1
    if loss_name == "coral":
        targets = [ClassifierOutputTarget(0)]
    else:
        targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets
        )

    return grayscale_cam[0]   # shape [H, W]


# ─────────────────────────────────────────────────────────
# Generate comparison figure for shared cases
# ─────────────────────────────────────────────────────────

def generate_comparison_figure(
    img_path: Path,
    cams: dict,
    predictions: dict,
    case_idx: int,
    out_path: Path,
) -> None:
    """
    Generate side-by-side Grad-CAM comparison across all 4 configurations.
    """
    configs = list(cams.keys())
    n_cols  = len(configs) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.5, 4))
    fig.suptitle(
        f"KL1 Misclassification Case {case_idx + 1:03d}\n"
        f"True grade: KL1 (Doubtful OA) — All models misclassified",
        fontsize=11, y=1.02
    )

    orig_img = load_image_for_display(img_path)

    axes[0].imshow(orig_img, cmap="gray", aspect="auto")
    axes[0].set_title("Original\n(CLAHE)", fontsize=9)
    axes[0].axis("off")

    for col, config in enumerate(configs, start=1):
        cam    = cams[config]
        pred   = predictions[config]
        colour = CONFIG_COLOURS.get(config, "black")

        overlay = show_cam_on_image(
            orig_img, cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.5
        )

        axes[col].imshow(overlay, aspect="auto")

        arch, loss = config.rsplit("_", 1)
        axes[col].set_title(
            f"{arch.upper()}\n{loss.upper()} loss\n"
            f"Predicted: KL{pred} ({KL_GRADE_NAMES[pred]})",
            fontsize=8, color=colour, fontweight="bold"
        )
        axes[col].axis("off")

        # Draw dashed box showing centre crop region
        h, w    = cam.shape
        h_start = int(h * (1 - CENTRE_CROP_RATIO) / 2)
        h_end   = int(h * (1 + CENTRE_CROP_RATIO) / 2)
        w_start = int(w * (1 - CENTRE_CROP_RATIO) / 2)
        w_end   = int(w * (1 + CENTRE_CROP_RATIO) / 2)

        rect = plt.Rectangle(
            (w_start, h_start),
            w_end - w_start, h_end - h_start,
            linewidth=1.5, edgecolor="white",
            facecolor="none", linestyle="--"
        )
        axes[col].add_patch(rect)

    plt.tight_layout()
    save(fig, out_path)


# ─────────────────────────────────────────────────────────
# Main audit pipeline
# ─────────────────────────────────────────────────────────

def run_audit(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Grad-CAM Audit — KL1 Misclassification Analysis")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Output dir  : {args.output_dir}")

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find best checkpoints
    print("\n  Finding best checkpoint per configuration...")
    best_checkpoints = find_best_checkpoints(results_dir)

    if len(best_checkpoints) < 4:
        print(f"\n  WARNING: Only {len(best_checkpoints)}/4 "
              f"configurations found. Proceeding with available.")

    # Step 2: Get KL1 misclassifications per model
    print("\n  Running inference to find KL1 misclassifications...")
    misclassification_results = {}

    for config_name, cp_path in best_checkpoints.items():
        result = get_kl1_misclassifications(
            checkpoint_path = cp_path,
            data_dir        = args.data_dir,
            device          = device,
        )
        misclassification_results[config_name] = result

        recall = 1 - result["n_misclassified"] / result["n_kl1_total"]
        print(f"  {config_name}: {result['n_misclassified']} misclassified "
              f"/ {result['n_kl1_total']} KL1 total "
              f"(recall={recall:.3f})")

    # Step 3: Find shared misclassification cases
    print("\n  Finding shared misclassification cases...")

    misclassified_sets = {
        config: set(str(p) for p in result["misclassified_paths"])
        for config, result in misclassification_results.items()
    }

    if misclassified_sets:
        all_models_set = set.intersection(*misclassified_sets.values())
    else:
        all_models_set = set()

    path_counts = Counter()
    for paths in misclassified_sets.values():
        for p in paths:
            path_counts[p] += 1
    shared_2plus = {p for p, c in path_counts.items() if c >= 2}

    print(f"  Misclassified by all 4 models : {len(all_models_set)}")
    print(f"  Misclassified by 2+ models    : {len(shared_2plus)}")

    if len(all_models_set) >= 5:
        audit_cases = sorted(all_models_set)[:20]
        case_type   = "all_models"
        print(f"  Using all-model cases for comparison figures")
    else:
        audit_cases = sorted(shared_2plus)[:20]
        case_type   = "shared_2plus"
        print(f"  Using 2+ model cases for comparison figures")

    print(f"  Audit cases selected: {len(audit_cases)}")

    # Step 4: Load models for Grad-CAM
    print("\n  Loading models for Grad-CAM generation...")
    models_dict = {}

    for config_name, cp_path in best_checkpoints.items():
        cp    = torch.load(cp_path, map_location=device)
        arch  = cp["arch"]
        loss  = cp["loss_name"]
        model = get_model(arch, loss)
        model.load_state_dict(cp["model_state_dict"])
        model = model.to(device)
        model.eval()
        models_dict[config_name] = {
            "model": model,
            "arch":  arch,
            "loss":  loss,
        }
        print(f"  Loaded: {config_name}")

    # Step 5: Generate Grad-CAM for audit cases
    print(f"\n  Generating Grad-CAM for {len(audit_cases)} audit cases...")

    transform    = get_val_test_transforms()
    audit_scores = []
    shared_dir   = output_dir / "shared_misclassified"
    shared_dir.mkdir(parents=True, exist_ok=True)

    for case_idx, img_path_str in enumerate(
        tqdm(audit_cases, desc="  generating CAMs")
    ):
        img_path = Path(img_path_str)

        try:
            pil_img      = Image.open(img_path)
            image_tensor = transform(pil_img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            continue

        cams        = {}
        predictions = {}
        loc_scores  = {}

        for config_name, model_info in models_dict.items():
            model     = model_info["model"]
            arch      = model_info["arch"]
            loss_name = model_info["loss"]

            try:
                cam = generate_gradcam(
                    model        = model,
                    arch         = arch,
                    loss_name    = loss_name,
                    image_tensor = image_tensor,
                    target_class = KL1_LABEL,
                    device       = device,
                )
                cams[config_name] = cam

                with torch.no_grad():
                    output = model(image_tensor)
                if loss_name == "ce":
                    pred = output.argmax(dim=1).item()
                else:
                    pred = coral_predict(output).item()
                predictions[config_name] = pred

                loc_scores[config_name] = compute_localisation_score(cam)

            except Exception as e:
                print(f"  CAM error {config_name}: {e}")
                continue

        if not cams:
            continue

        fig_path = shared_dir / f"case_{case_idx+1:03d}_comparison.png"
        generate_comparison_figure(
            img_path    = img_path,
            cams        = cams,
            predictions = predictions,
            case_idx    = case_idx,
            out_path    = fig_path,
        )

        audit_scores.append({
            "case_idx":            case_idx + 1,
            "image_path":          img_path_str,
            "case_type":           case_type,
            "predictions":         predictions,
            "localisation_scores": loc_scores,
        })

    # Step 6: Save audit scores
    print("\n  Saving audit scores...")

    scores_path = output_dir / "audit_scores.json"
    with open(scores_path, "w") as f:
        json.dump({
            "n_audit_cases":     len(audit_scores),
            "case_type":         case_type,
            "n_all_model_cases": len(all_models_set),
            "n_shared_2plus":    len(shared_2plus),
            "centre_crop_ratio": CENTRE_CROP_RATIO,
            "misclassification_summary": {
                config: {
                    "n_misclassified": r["n_misclassified"],
                    "n_kl1_total":     r["n_kl1_total"],
                    "kl1_recall":      round(
                        1 - r["n_misclassified"] / r["n_kl1_total"], 4
                    ) if r["n_kl1_total"] > 0 else 0,
                }
                for config, r in misclassification_results.items()
            },
            "cases": audit_scores,
        }, f, indent=2)

    print(f"  Saved -> {scores_path}")

    # Step 7: Localisation score summary
    print("\n  Localisation score summary:")
    print(f"  (% activation in central {CENTRE_CROP_RATIO*100:.0f}% crop)")

    config_scores = {config: [] for config in models_dict.keys()}
    for case in audit_scores:
        for config, score in case["localisation_scores"].items():
            config_scores[config].append(score)

    for config, scores in config_scores.items():
        if scores:
            print(f"  {config:30s}: "
                  f"mean={np.mean(scores):.3f}  "
                  f"std={np.std(scores):.3f}  "
                  f"n={len(scores)}")

    print("\n" + "=" * 60)
    print(f"  Audit complete.")
    print(f"  Comparison figures: {shared_dir}")
    print(f"  Audit scores      : {scores_path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grad-CAM audit for KL1 misclassification cases"
    )
    p.add_argument("--data_dir",    type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--output_dir",  type=str,
                   default="results/gradcam")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(args)
