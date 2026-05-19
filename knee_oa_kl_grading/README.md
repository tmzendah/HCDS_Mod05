# Can CNNs Trained on DL-Inferred KL Pseudo-Labels from MRKR Classify Knee OA Severity and Generalise to Radiologist-Labelled OAI/Kaggle Radiographs, Particularly for Early OA Grades?

Talita Mzendah | MSt Healthcare Data Science | University of Cambridge

---

## Project overview

This project trains ResNet50, DenseNet121, and EfficientNet-B0 convolutional neural networks to classify knee osteoarthritis severity from plain radiographs using the Kellgren-Lawrence (KL) grading system. Models are trained on the MRKR dataset using DL-inferred KL pseudo-labels and evaluated against radiologist-graded labels from the OAI/Kaggle dataset, with a focus on early-grade performance and subgroup fairness.

---

## Repository structure

```
knee_oa_kl_grading/
├── scripts/                 ← Production Python scripts
│   ├── convert_dicom_to_png.py
│   ├── quality_control.py
│   ├── flag_poor_quality.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── external_validate.py
│   └── utils.py
│
├── config/                  ← Experiment configuration
│   ├── config.yaml
│   └── model_configs.yaml
│
├── notebooks/               ← Sequential analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_dicom_conversion.ipynb
│   ├── 03_quality_control.ipynb
│   ├── 04_data_preparation.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_evaluation_analysis.ipynb
│
├── report/
│   ├── report.qmd           ← Main Quarto document
│   ├── references.bib       ← BibTeX references
│   └── vancouver.csl        ← Citation style
│
├── reports/                 ← Generated outputs
│   ├── final_results/       ← Figures and CSV metrics
│   ├── qc_reports/          ← Quality control reports
│   └── flagged_images/      ← Poor-quality image log
│
├── slurm/                   ← CSD3 HPC job scripts
│   ├── train_resnet50.sh
│   ├── train_efficientnet.sh
│   └── run_experiments.sh
│
├── tests/                   ← Unit tests
│   ├── test_data_loading.py
│   ├── test_model_architecture.py
│   └── test_metrics.py
│
├── docker/                  ← Docker support (optional)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── checkpoints/             ← Model weights — not committed
├── data/                    ← Not committed — see Dataset section
├── environment.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## Datasets

Data files are **not included** in this repository. Full download instructions, expected folder structures, and AWS/Kaggle CLI commands are in [`data/README.md`](data/README.md).

| Dataset | Role | Source | Licence |
|---|---|---|---|
| MRKR (Emory) | Primary training | [AWS Open Data](https://registry.opendata.aws/mrkr) | CC-BY-SA |
| OAI/Kaggle | External validation | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) | CC BY 4.0 |

**Ethics:** Both datasets are publicly available and fully anonymised. No identifiable patient data are included and no additional ethical approval was required.

---

## Reproduction steps

### 1. Set up environment
```bash
conda env create -f environment.yml
conda activate knee_oa
pip install -e .
```

### 2. Download datasets
Follow the instructions in [`data/README.md`](data/README.md). Place MRKR DICOMs in `data/raw/mrkr/dicoms/` and OAI/Kaggle images in `data/raw/kaggle_oai/`.

### 3. Convert DICOMs to PNG
```bash
python scripts/convert_dicom_to_png.py --input_dir data/raw/mrkr_dicoms --output_dir data/processed --size 512
```

### 4. Run quality control
```bash
python scripts/quality_control.py --image_dir data/processed --output reports/qc_reports/qc_report.csv
python scripts/flag_poor_quality.py --qc_report reports/qc_reports/qc_report.csv
```

### 5. Prepare data and train models
```bash
python scripts/prepare_data.py --config config/config.yaml
python scripts/train_model.py --config config/config.yaml --model resnet50
python scripts/train_model.py --config config/config.yaml --model densenet121
python scripts/train_model.py --config config/config.yaml --model efficientnet_b0
```

### 6. External validation
```bash
python scripts/external_validate.py --config config/config.yaml
```

### 7. Render report
```bash
quarto render report/report.qmd
```

**On CSD3 HPC:** use the scripts in `slurm/` instead of running locally.

---

## Library versions

| Package | Version |
|---|---|
| torch | 2.1.0 |
| torchvision | 0.16.0 |
| numpy | 1.24.0 |
| scikit-learn | 1.3.0 |
| matplotlib | 3.7.2 |
| opencv-python | 4.8.0 |
| grad-cam | 1.4.8 |
| quarto | ≥1.3 |
