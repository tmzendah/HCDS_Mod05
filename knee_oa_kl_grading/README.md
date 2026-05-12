# Grade-Specific Performance Failure and Clinical Attention Audit in Automated Knee Osteoarthritis Grading

Talita Mzendah | MSt Healthcare Data Science | University of Cambridge

---

## Abstract
[Paste your final abstract here once complete]

---

## Project overview

This project trains ResNet50 and EfficientNet-B0 convolutional neural networks to classify knee osteoarthritis severity from plain radiographs using the Kellgren-Lawrence (KL) grading system. It evaluates grade-specific performance failures and uses Grad-CAM to audit the clinical plausibility of model attention patterns.

---

## Repository structure

```
knee_oa_kl_grading/
├── src/                     ← Python source code
│   ├── dataset.py           ← DataLoader, augmentation, CLAHE pipeline
│   ├── models.py            ← get_model() for ResNet50 + EfficientNet-B0
│   ├── train.py             ← Training loop, early stopping, seed control
│   ├── evaluate.py          ← Per-grade metrics, confusion matrix, AUC
│   ├── gradcam.py           ← Grad-CAM generation and grid figure
│   └── utils.py             ← Shared helpers, class weights calculator
│
├── experiments/             ← One config file per experiment
│   ├── baseline_resnet50.yaml
│   ├── baseline_efficientnet.yaml
│   ├── exp1_clahe.yaml
│   ├── exp2_grade1_weight.yaml
│   ├── exp3_ordinal_loss.yaml
│   └── exp4_tta.yaml
│
├── report/
│   ├── report.qmd           ← Main Quarto document
│   ├── references.bib       ← BibTeX references
│   ├── vancouver.csl        ← Citation style
│   └── plos2015.bst         ← Bibliography style
│
├── notebooks/               ← Exploratory work
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_debug.ipynb
│   └── 03_gradcam_exploration.ipynb
│
├── slurm/                   ← CSD3 HPC job scripts
│   ├── train_resnet50.sh
│   ├── train_efficientnet.sh
│   └── run_experiments.sh
│
├── outputs/
│   ├── figures/             ← Generated plots (committed)
│   ├── results/             ← CSV metrics (committed)
│   └── checkpoints/         ← Model weights — not committed
│
├── data/                    ← Not committed — see Dataset section below
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Dataset

The Kaggle Knee Osteoarthritis Dataset with Severity Grading (CC BY 4.0) is available at:  
<https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity>

Download and place in `data/raw/` maintaining the `train/val/test` folder structure:

```
data/raw/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
└── test/
```

Data files are not included in this repository.

---

## Reproduction Steps

### 1. Set up environment
```bash
conda env create -f environment.yml
conda activate knee_oa
```

### 2. Download dataset
See Dataset section above. Place downloaded files in `data/raw/`.

### 3. Run baseline training
```bash
python src/train.py --config experiments/baseline_resnet50.yaml
python src/train.py --config experiments/baseline_efficientnet.yaml
```

### 4. Run improvement experiments
```bash
python src/train.py --config experiments/exp1_clahe.yaml
python src/train.py --config experiments/exp2_grade1_weight.yaml
python src/train.py --config experiments/exp3_ordinal_loss.yaml
python src/train.py --config experiments/exp4_tta.yaml
```

### 5. Generate figures and evaluation
```bash
python src/evaluate.py
python src/gradcam.py
```

### 6. Render report
```bash
quarto render report/report.qmd
```

---

## Library Versions

| Package | Version |
|---|---|
| torch | 2.1.0 |
| torchvision | 0.16.0 |
| numpy | 1.24.0 |
| scikit-learn | 1.3.0 |
| matplotlib | 3.7.2 |
| opencv-python | 4.8.0 |
| grad-cam | 1.4.8 |
| quarto | >=1.3 |
