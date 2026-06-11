# HCDS Module 05 — Does Loss Function Choice Affect Early Osteoarthritis Detection?

**Talita Mzendah | MSt Healthcare Data Science | University of Cambridge**

---

## Overview

This project investigates whether loss function choice affects early knee osteoarthritis (OA) detection in automated Kellgren-Lawrence (KL) grading. A 2×2 factorial design compares two training objectives (categorical cross-entropy and CORAL ordinal loss) across two CNN architectures (ResNet50 and EfficientNet-B0), trained across three random seeds on the Kaggle Knee OA Dataset (n=8,260 radiographs).

---

## Repository structure

```
HCDS_Mod05/
├── README.md
├── environment.yml
├── requirements.txt
├── LICENSE
│
├── configs/                        ← Per-experiment YAML configs
│   ├── resnet50_ce.yaml
│   ├── resnet50_coral.yaml
│   ├── efficientnet_b0_ce.yaml
│   └── efficientnet_b0_coral.yaml
│
├── src/                            ← Modular Python source
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── gradcam.py
│   └── utils.py
│
├── notebooks/                      ← Analysis notebooks
│   ├── 01_dataset_check_and_eda.ipynb
│   ├── 02_training_summary.ipynb
│   └── 03_results_figures.ipynb
│
├── scripts/                        ← Experiment runner scripts
│   ├── run_all_experiments.sh
│   ├── run_resnet50_ce.sh
│   ├── run_resnet50_coral.sh
│   ├── run_efficientnet_ce.sh
│   └── run_efficientnet_coral.sh
│
├── results/                        ← Outputs (metrics, figures, Grad-CAM)
│   ├── metrics_summary.csv
│   ├── per_seed_results.csv
│   ├── confusion_matrices/
│   ├── figures/
│   └── gradcam_examples/
│
├── reports/                        ← Report source and figures
│   ├── report.qmd
│   ├── references.bib
│   ├── vancouver.csl
│   ├── cambridge_logo.png
│   └── report_figures/
│
└── docs/                           ← Supporting documentation
    ├── dataset_access.md
    ├── reproducibility.md
    └── model_cards.md
```

---

## Dataset

The Kaggle Knee Osteoarthritis Dataset with Severity Grading (n=8,260 radiographs, KL grades 0–4) is not included in this repository. See `docs/dataset_access.md` for download instructions.

---

## Environment setup

```bash
conda env create -f environment.yml
conda activate knee_oa
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Rendering the report

```bash
cd reports/
quarto render report.qmd
```
