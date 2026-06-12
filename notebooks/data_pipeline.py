"""
notebooks/data_pipeline.py
Knee Osteoarthritis Dataset with KL Severity Grading

Purpose:
- Load images from folder structure using PyTorch ImageFolder
- Apply clinically justified preprocessing and augmentation
- Create data loaders for training, validation, and testing
- Handle class imbalance with weighted sampling and loss weighting

This script does not train the model — it builds and verifies
the data pipeline that the training script will use.

Date: April 2026
"""

# ── Step 1: Import tools ──────────────────────────────────────────────────────
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ── Step 2: Paths and settings ────────────────────────────────────────────────
data_dir = "/path/to/your/data/knee_oa"

# Paths for each split
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")
test_dir  = os.path.join(data_dir, "test")

# ResNet50 standard input size — consistent with published literature
IMAGE_SIZE = 224

# Batch size — how many images the GPU processes at once
# 32 is a safe starting point for the A100
BATCH_SIZE = 32

# Number of parallel workers loading data in the background
# 4 is standard for HPC environments
NUM_WORKERS = 4

# Grade labels for reference
grade_labels = {
    0: "Grade 0 — Normal",
    1: "Grade 1 — Doubtful",
    2: "Grade 2 — Mild",
    3: "Grade 3 — Moderate",
    4: "Grade 4 — Severe"
}


# ── Step 3: Define transforms ─────────────────────────────────────────────────
# ImageNet normalisation values — standard for ResNet pretrained on ImageNet
# Mean and std calculated across millions of ImageNet images
# We use these even for X-rays because our model starts from ImageNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training transform — includes augmentation ────────────────────────────────
# Augmentation is ONLY applied during training, never val or test
# Each transformation is clinically justified:

train_transform = transforms.Compose([

    # Resize to ResNet50 input size
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Convert greyscale to 3-channel
    # ResNet50 expects 3 channels (RGB) but X-rays are greyscale (1 channel)
    # This duplicates the greyscale channel three times — standard practice
    transforms.Grayscale(num_output_channels=3),

    # Horizontal flip — left vs right knee, both clinically equivalent
    # p=0.5 means 50% chance of flipping each image
    transforms.RandomHorizontalFlip(p=0.5),

    # Rotation ±10° — simulates real positioning variability in clinical practice
    # Patients are never perfectly positioned — this is clinically realistic
    transforms.RandomRotation(degrees=10),

    # Brightness and contrast variation — simulates exposure differences
    # between scanners, departments, and patient body habitus
    # Directly justified by underexposed grade 4 images found in Phase 1
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # Slight zoom — simulates different patient sizes and collimation
    # scale=(0.85, 1.0) means between 85% and 100% of original size
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.85, 1.0)
    ),

    # Convert PIL image to PyTorch tensor (scales pixels to 0-1 range)
    transforms.ToTensor(),

    # Normalise using ImageNet mean and std
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ── Validation and test transform — NO augmentation ───────────────────────────
# Consistent, deterministic preprocessing only
# Must be identical for val and test — ensures fair, reproducible evaluation

val_test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ── Step 4: Load datasets using ImageFolder ───────────────────────────────────
# ImageFolder automatically reads folder names as class labels
# Folder "0" → class 0, folder "1" → class 1, etc.

print("Loading datasets...")

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=val_test_transform)

print(f"  Training images:   {len(train_dataset)}")
print(f"  Validation images: {len(val_dataset)}")
print(f"  Test images:       {len(test_dataset)}")
print(f"  Classes detected:  {train_dataset.classes}")
print()


# ── Step 5: Handle class imbalance with weighted sampling ─────────────────────
# Problem: Grade 0 has 2,286 images, Grade 4 has only 173
# Without correction, the model sees Grade 0 ~13x more than Grade 4
# Solution: WeightedRandomSampler gives rare classes a higher chance of being selected

# Count images per class in training set
class_counts = np.zeros(len(train_dataset.classes))
for _, label in train_dataset.samples:
    class_counts[label] += 1

print("Class distribution in training set:")
for i, count in enumerate(class_counts):
    print(f"  {grade_labels[i]}: {int(count)} images")
print()

# Calculate weight for each class — inverse of frequency
# Rare classes get higher weight, common classes get lower weight
class_weights = 1.0 / class_counts
print("Class weights (inverse frequency):")
for i, weight in enumerate(class_weights):
    print(f"  {grade_labels[i]}: {weight:.6f}")
print()

# Assign weight to every individual sample in the training set
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sample_weights = torch.DoubleTensor(sample_weights)

# Create the weighted sampler
# replacement=True allows the same image to be sampled multiple times
# num_samples = length of training set (one full pass per epoch)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ── Step 6: Create class weights for loss function ────────────────────────────
# Second layer of imbalance correction — penalise the model more heavily
# for getting rare classes wrong during training
# These weights are passed to the loss function (CrossEntropyLoss)

loss_weights = torch.FloatTensor(class_weights)
loss_weights = loss_weights / loss_weights.sum()  # normalise to sum to 1

print("Loss weights (for CrossEntropyLoss):")
for i, weight in enumerate(loss_weights):
    print(f"  {grade_labels[i]}: {weight:.4f}")
print()


# ── Step 7: Create DataLoaders ────────────────────────────────────────────────
# DataLoader wraps the dataset and handles batching and parallel loading
# Training uses the weighted sampler — shuffle=False because sampler handles order
# Val and test use shuffle=False — consistent evaluation order

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,          # weighted sampling for class imbalance
    num_workers=NUM_WORKERS,
    pin_memory=True           # speeds up GPU data transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("DataLoaders created:")
print(f"  Training batches:   {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches:       {len(test_loader)}")
print()


# ── Step 8: Verify a batch ────────────────────────────────────────────────────
# Load one batch and check shapes and label distribution
# This confirms the pipeline works before we attach a model to it

print("Verifying one training batch...")
images, labels = next(iter(train_loader))

print(f"  Batch image tensor shape: {images.shape}")
print(f"  Expected:                 torch.Size([{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}])")
print(f"  Batch label tensor shape: {labels.shape}")
print(f"  Labels in this batch:     {labels.tolist()}")
print()

# Check pixel value range after normalisation
print(f"  Pixel value range after normalisation:")
print(f"    Min: {images.min():.3f}")
print(f"    Max: {images.max():.3f}")
print(f"    Mean: {images.mean():.3f}")
print()


# ── Step 9: Visualise one augmented batch ─────────────────────────────────────
# Show what images look like AFTER augmentation is applied
# Compare with Phase 1 to see the effect of transforms

print("Saving augmented batch visualisation...")

# Denormalise images for display
# Reverse the normalisation so pixel values are back in 0-1 range
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
images_display = images * std + mean
images_display = images_display.clamp(0, 1)

# Show first 10 images from the batch
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Phase 2 — Augmented Training Batch Sample", fontsize=12, fontweight="bold")

for i, ax in enumerate(axes.flat):
    if i < len(images_display):
        # Convert from (3, H, W) tensor to (H, W, 3) numpy for display
        img_np = images_display[i].permute(1, 2, 0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.set_title(grade_labels[labels[i].item()], fontsize=8)
        ax.axis("off")

plt.tight_layout()
output_path = "/path/to/your/outputs/phase2_augmented_batch.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"  Saved to: {output_path}")
print()


# ── Step 10: Save pipeline components for use in training ─────────────────────
# Save the loss weights so the training script can load them
weights_path = "/path/to/your/outputs/loss_weights.pt"
torch.save(loss_weights, weights_path)
print(f"Loss weights saved to: {weights_path}")
print()


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 2 Complete")
print("=" * 60)
print(f"""
Pipeline verified and ready for training.

Settings confirmed:
  Image size:    {IMAGE_SIZE} x {IMAGE_SIZE} pixels
  Batch size:    {BATCH_SIZE}
  Augmentation:  Horizontal flip, rotation ±10 degrees,
                 brightness/contrast variation, random crop
  Imbalance:     Weighted sampler + weighted loss function
  CLAHE:         Not applied (baseline — add later if needed)

Next step: Phase 3 — Model training script
""")
