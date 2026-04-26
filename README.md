# Knee OA KL Grading — HCDS Module 05

**Grade-Specific Performance Failure and Clinical Attention Audit in Automated Knee Osteoarthritis Grading: A Radiographer-Led Explainability Study**

Talita Mzendah | MSt Healthcare Data Science | University of Cambridge

---

## Project overview

This project trains a ResNet50 convolutional neural network to classify knee osteoarthritis severity from plain radiographs using the Kellgren-Lawrence (KL) grading system. It evaluates grade-specific performance and uses Grad-CAM to audit the clinical plausibility of model attention patterns.

## Repository structure

```
├── report/
│   ├── report.qmd           ← Main Quarto document
│   ├── references.bib       ← BibTeX references
│   └── vancouver.csl        ← Citation style
│
├── scripts/
│   ├── config.py            ← All settings in one place
│   ├── visualise_data.py    ← Phase 1 — data exploration
│   ├── data_pipeline.py     ← Phase 2 — data loading
│   ├── train.py             ← Phase 3 — model training
│   ├── evaluate.py          ← Phase 4 — metrics per grade
│   └── gradcam.py           ← Phase 4 — attention maps
│
├── outputs/
│   ├── figures/             ← Generated plots and images
│   └── metrics/             ← CSV files with results
│
├── supplementary/           ← Development logs and QA documents
│
├── requirements.txt         ← Python package versions
└── environment.yml          ← Conda environment export
```

## Dataset

The Knee Osteoarthritis Dataset with Severity Grading is publicly available on Kaggle:
<https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity>

Data files are not included in this repository.

## Requirements

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

Or via conda:
```bash
conda env create -f environment.yml
conda activate knee-oa
```
