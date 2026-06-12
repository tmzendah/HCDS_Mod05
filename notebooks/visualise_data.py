"""
notebooks/visualise_data.py
Knee Osteoarthritis Dataset with KL Severity Grading

Purpose:
- Visually inspect images from each KL grade
- Confirm all images are knees (no wrong body parts)
- Check image quality across grades
- Understand what the model will be learning from

Author: tm922
Date: April 2026
"""

# ── Step 1: Import the tools we need ─────────────────────────────────────────
import os                          # navigates folders and finds files
import random                      # randomly selects images so we see variety
import numpy as np                 # handles images as grids of numbers
import matplotlib.pyplot as plt    # displays images visually
from PIL import Image              # opens image files from disk


# ── Step 2: Define where our data lives ──────────────────────────────────────
data_dir = "/rds/user/tm922/hpc-work/data/knee_oa"

# Which split to inspect — change to "val" or "test" to check those too
split = "train"

# The five KL grades — folder names match these exactly
grades = ["0", "1", "2", "3", "4"]

# Human-readable labels for the plot titles
grade_labels = {
    "0": "Grade 0 — Normal",
    "1": "Grade 1 — Doubtful",
    "2": "Grade 2 — Mild",
    "3": "Grade 3 — Moderate",
    "4": "Grade 4 — Severe"
}

# How many images to show per grade
images_per_grade = 5


# ── Step 3: Count images per grade ───────────────────────────────────────────
print("=" * 60)
print(f"Dataset: Knee OA | Split: {split}")
print("=" * 60)
print("\nImage counts per grade:")
print("-" * 30)

total = 0
counts = {}
for grade in grades:
    grade_path = os.path.join(data_dir, split, grade)
    count = len(os.listdir(grade_path))
    counts[grade] = count
    total += count
    print(f"  Grade {grade} ({grade_labels[grade]}): {count} images")

print("-" * 30)
print(f"  Total: {total} images")
print()

max_count = max(counts.values())
min_count = min(counts.values())
print(f"Class imbalance ratio: {max_count/min_count:.1f}:1")
print(f"(Grade 0 has {max_count/min_count:.1f}x more images than Grade 4)")
print()


# ── Step 4: Visualise sample images from each grade ──────────────────────────
print("Generating visualisation grid...")

fig, axes = plt.subplots(
    nrows=len(grades),
    ncols=images_per_grade,
    figsize=(15, 12)
)

fig.suptitle(
    f"Knee OA Dataset — KL Grade Visual Inspection ({split} split)",
    fontsize=14,
    fontweight="bold",
    y=1.02
)

for row_idx, grade in enumerate(grades):
    grade_path = os.path.join(data_dir, split, grade)
    all_images = os.listdir(grade_path)
    selected = random.sample(all_images, min(images_per_grade, len(all_images)))

    for col_idx, filename in enumerate(selected):
        img_path = os.path.join(grade_path, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        ax = axes[row_idx, col_idx]
        ax.imshow(img_array, cmap="gray")
        ax.axis("off")

        if col_idx == 0:
            ax.set_title(
                grade_labels[grade],
                fontsize=9,
                fontweight="bold",
                loc="left",
                pad=4
            )

        ax.set_xlabel(
            filename[:15] + "..." if len(filename) > 15 else filename,
            fontsize=6,
            labelpad=2
        )

plt.tight_layout()

output_path = "/rds/user/tm922/hpc-work/outputs/phase1_visual_check.png"
os.makedirs("/rds/user/tm922/hpc-work/outputs", exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Visualisation saved to: {output_path}")
print()


# ── Step 5: Image property check ─────────────────────────────────────────────
print("=" * 60)
print("Image Property Check")
print("=" * 60)

for grade in grades:
    grade_path = os.path.join(data_dir, split, grade)
    all_images = os.listdir(grade_path)
    sample = all_images[:10]
    sizes = []
    modes = []

    for filename in sample:
        img_path = os.path.join(grade_path, filename)
        img = Image.open(img_path)
        sizes.append(img.size)
        modes.append(img.mode)

    unique_sizes = set(sizes)
    unique_modes = set(modes)

    print(f"\nGrade {grade}:")
    print(f"  Image sizes (sample of 10): {unique_sizes}")
    print(f"  Image modes: {unique_modes}")

    if "RGB" in unique_modes:
        print(f"  WARNING: Some images are RGB not greyscale!")
    else:
        print(f"  All greyscale — correct for X-rays")


# ── Step 6: Final checklist ───────────────────────────────────────────────────
print()
print("=" * 60)
print("Phase 1 Complete — Clinical checks for saved image:")
print("=" * 60)
print("""
Use your radiographer expertise to check:
  1. Are all images clearly knee X-rays?
  2. Are grade 0 images visually normal?
  3. Do grade 4 images show clear severe disease?
  4. Can you see visual progression from grade 0 to grade 4?
  5. Are there any obviously mislabelled images?
  6. Are any images blank, corrupted, or poor quality?
  7. Are images AP weight-bearing or could some be supine?
  8. Is positioning consistent or are there rotation errors?

Note any problem filenames — we can remove them before training.
""")
