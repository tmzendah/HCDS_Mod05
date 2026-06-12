"""
src/models.py
Model definitions for the 2x2 KL grading experiment.

Two architectures compared:
----------------------------
1. ResNet50    -- most cited baseline in KL grading literature
                  (Tiulpin et al. 2018, Pi et al. 2023)
                  Deep residual network, 25.6M parameters
                  Strong feature extraction via skip connections

2. EfficientNet-B0 -- used on this exact Kaggle dataset
                      (Momenpour et al. 2025, 82.07% accuracy)
                      Compound scaling of depth/width/resolution
                      7.8M parameters -- more efficient than ResNet50

Both architectures:
-------------------
- Pretrained on ImageNet (transfer learning)
- Final classifier replaced for KL grading task
- Output head depends on loss function:
    CE loss   -> 5 outputs (one per KL grade, softmax at inference)
    CORAL loss -> 4 outputs (one per rank boundary, sigmoid at inference)
- All layers fine-tuned during training (full fine-tuning)

Why full fine-tuning?
---------------------
Knee radiographs differ substantially from natural ImageNet images.
Fine-tuning all layers allows low-level features (edges, textures)
to adapt to the radiographic domain, not just the final classifier.
This is consistent with Momenpour et al. 2025 and Tiulpin et al. 2018.

Reference:
    He et al. (2016). Deep Residual Learning for Image Recognition.
    Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs.
    Momenpour et al. (2025). Diagnostics.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)

# ── Constants ─────────────────────────────────────────────
NUM_CLASSES  = 5   # KL grades 0-4
NUM_RANKS    = 4   # CORAL rank boundaries (num_classes - 1)

SUPPORTED_ARCHS  = ["resnet50", "efficientnet"]
SUPPORTED_LOSSES = ["ce", "coral"]


# ─────────────────────────────────────────────────────────
# Output size helper
# ─────────────────────────────────────────────────────────

def get_output_size(loss_name: str) -> int:
    """
    Returns the number of output neurons required for each loss.

    CE loss   -> 5 (one logit per KL grade)
    CORAL loss -> 4 (one logit per rank boundary)

    Args:
        loss_name : 'ce' or 'coral'

    Returns:
        integer output size
    """
    if loss_name == "ce":
        return NUM_CLASSES
    elif loss_name == "coral":
        return NUM_RANKS
    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Choose from {SUPPORTED_LOSSES}."
        )


# ─────────────────────────────────────────────────────────
# ResNet50
# ─────────────────────────────────────────────────────────

def build_resnet50(loss_name: str) -> nn.Module:
    """
    Build ResNet50 with ImageNet pretrained weights.
    Final fully connected layer replaced for KL grading.

    Architecture:
        Input  : [batch, 3, 224, 224]
        Backbone: ResNet50 (pretrained, all layers trainable)
        Classifier: Linear(2048 -> output_size)
        Output : [batch, output_size]
                 output_size = 5 for CE, 4 for CORAL

    Args:
        loss_name : 'ce' or 'coral' -- determines output size

    Returns:
        nn.Module (ResNet50 with replaced classifier)
    """
    output_size = get_output_size(loss_name)

    # Load pretrained weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace final fully connected layer
    # Original: Linear(2048, 1000) for ImageNet 1000 classes
    # Replaced: Linear(2048, output_size) for KL grading
    in_features = model.fc.in_features   # 2048 for ResNet50
    model.fc = nn.Linear(in_features, output_size)

    return model


# ─────────────────────────────────────────────────────────
# EfficientNet-B0
# ─────────────────────────────────────────────────────────

def build_efficientnet_b0(loss_name: str) -> nn.Module:
    """
    Build EfficientNet-B0 with ImageNet pretrained weights.
    Final classifier layer replaced for KL grading.

    Architecture:
        Input  : [batch, 3, 224, 224]
        Backbone: EfficientNet-B0 (pretrained, all layers trainable)
        Classifier: Dropout(0.2) -> Linear(1280 -> output_size)
        Output : [batch, output_size]
                 output_size = 5 for CE, 4 for CORAL

    Note on dropout:
        EfficientNet-B0 includes dropout (p=0.2) before the final
        linear layer by default. This is kept as-is -- consistent
        with Momenpour et al. 2025 on this exact dataset.

    Args:
        loss_name : 'ce' or 'coral' -- determines output size

    Returns:
        nn.Module (EfficientNet-B0 with replaced classifier)
    """
    output_size = get_output_size(loss_name)

    # Load pretrained weights
    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # Replace final classifier
    # EfficientNet classifier is Sequential(Dropout, Linear)
    # Original: Linear(1280, 1000) for ImageNet 1000 classes
    # Replaced: Linear(1280, output_size) for KL grading
    in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
    model.classifier[1] = nn.Linear(in_features, output_size)

    return model


# ─────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────

def get_model(arch: str, loss_name: str) -> nn.Module:
    """
    Factory function -- returns the correct model for a given
    architecture and loss function combination.

    Used by train.py so model choice is driven by command-line args.

    Args:
        arch      : 'resnet50' or 'efficientnet'
        loss_name : 'ce' or 'coral'

    Returns:
        nn.Module with correct output head for the loss function

    Raises:
        ValueError if arch or loss_name is not recognised
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from {SUPPORTED_ARCHS}."
        )
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Choose from {SUPPORTED_LOSSES}."
        )

    if arch == "resnet50":
        return build_resnet50(loss_name)
    elif arch == "efficientnet":
        return build_efficientnet_b0(loss_name)


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model : nn.Module

    Returns:
        dict with 'total' and 'trainable' parameter counts
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────
# Verification (run directly to test model builds)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  models.py -- model build verification")
    print("=" * 60)

    # Test all four combinations
    configs = [
        ("resnet50",    "ce"),
        ("resnet50",    "coral"),
        ("efficientnet","ce"),
        ("efficientnet","coral"),
    ]

    # Dummy batch matching dataloader output
    dummy_input = torch.randn(4, 3, 224, 224)

    for arch, loss in configs:
        print(f"\n  {arch} + {loss}")
        model  = get_model(arch, loss)
        params = count_parameters(model)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        expected_out = 5 if loss == "ce" else 4
        print(f"    Output shape    : {list(output.shape)}")
        print(f"    Expected shape  : [4, {expected_out}]")
        print(f"    Total params    : {params['total']:,}")
        print(f"    Trainable params: {params['trainable']:,}")

        assert output.shape == (4, expected_out), (
            f"Output shape mismatch: "
            f"got {output.shape}, expected (4, {expected_out})"
        )
        print(f"    PASSED")

    print("\n  All four configurations verified.")
    print("=" * 60)
