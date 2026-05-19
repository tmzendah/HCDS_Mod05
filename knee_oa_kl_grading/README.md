# Can CNNs Trained on DL-Inferred KL Pseudo-Labels from MRKR Classify Knee OA Severity and Generalise to Radiologist-Labelled OAI/Kaggle Radiographs, Particularly for Early OA Grades?

Talita Mzendah | MSt Healthcare Data Science | University of Cambridge

---

## Abstract

[Paste your final abstract here once complete]

---

## Project overview

This project trains ResNet50, DenseNet121, and EfficientNet-B0 convolutional neural networks to classify knee osteoarthritis severity from plain radiographs using the Kellgren-Lawrence (KL) grading system. Models are trained on the MRKR dataset using DL-inferred KL pseudo-labels and evaluated against radiologist-graded labels from the OAI/Kaggle dataset, with a focus on early-grade performance and subgroup fairness.

---

## Repository structure

```
knee_oa_kl_grading/
├── scripts/                 ← Production Python scripts
│   ├── convert_dicom_to_jpeg.py
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

### Primary training dataset — MRKR

The Emory Knee Radiograph (MRKR) dataset comprises 503,261 knee radiographs from 83,011 patients. Approximately 40% of patients are African American, making it one of the most demographically diverse knee imaging datasets available. Images are provided in DICOM format alongside clinical metadata including patient-reported pain scores, diagnostic and procedural codes, image laterality, view type, and presence of hardware. KL grades in MRKR are DL-inferred rather than radiologist-assigned.

- **Access:** [AWS Open Data Registry](https://registry.opendata.aws/mrkr) — accessed 17 May 2026
- **Documentation:** <https://github.com/Emory-HITI/MRKR>
- **Licence:** CC-BY-SA
- **Acknowledgement:** MD.ai provided assistance with image de-identification.

### External validation dataset — OAI/Kaggle

The Kaggle Knee Osteoarthritis Dataset with Severity Grading provides radiologist-assigned KL grades and is used for external validation only.

- **Access:** <https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity>
- **Licence:** CC BY 4.0

**Ethics:** Both datasets are publicly available and fully anonymised. No identifiable patient data are included and no additional ethical approval was required.

Data files are not included in this repository. Place downloaded files in `data/raw/`.

---

## Reproduction steps

### 1. Set up environment
```bash
conda env create -f environment.yml
conda activate knee_oa
pip install -e .
```

### 2. Download datasets
See Dataset section above. Place MRKR DICOMs in `data/raw/mrkr_dicoms/` and OAI/Kaggle images in `data/raw/kaggle_oai/`.

### 3. Convert DICOMs to JPEG
```bash
python scripts/convert_dicom_to_jpeg.py --input_dir data/raw/mrkr_dicoms --output_dir data/processed --size 512
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
