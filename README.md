# Does Loss Function Choice Affect Early Osteoarthritis Detection?

**Talita Mzendah | MSt Healthcare Data Science | University of Cambridge**

---

## Project summary

Knee OA grading using the Kellgren-Lawrence scale is ordinal, yet most automated systems use categorical cross-entropy (CE) which ignores grade order. This study tested whether CORAL ordinal loss improves KL Grade 1 detection compared with CE across two CNN architectures (ResNet50 and EfficientNet-B0), trained on 8,260 knee radiographs across three random seeds. CORAL improved KL1 recall from 0.127 to 0.341 for ResNet50 and from 0.190 to 0.354 for EfficientNet-B0, with EfficientNet-B0+CORAL achieving the best overall QWK of 0.809 ± 0.009. Despite this, all configurations remained below the meta-analytic KL1 benchmark of 0.64, suggesting that image-only models are insufficient without clinical metadata such as weight-bearing status and patient age.

---

## Research question

> Does CORAL ordinal loss improve KL Grade 1 recall compared with categorical cross-entropy across CNN architectures for automated knee OA grading?

---

## Dataset

**Kaggle Knee OA Dataset**

- Source: [https://www.kaggle.com/datasets/tommyngx/kneeoa](https://www.kaggle.com/datasets/tommyngx/kneeoa)
- Original data: Chen, Pingjun (2018). Knee Osteoarthritis Severity Grading Dataset. Mendeley Data, V1. doi: 10.17632/56rmx5bjcr.1. Organised from the Osteoarthritis Initiative (OAI).
- Licence: CC BY 4.0
- Size: 8,260 anteroposterior knee radiographs, KL grades 0–4
- Images are supplied as **preprocessed 224×224 single-knee radiographs** — no additional resizing or cropping is required
- Already organised into predefined `train/`, `val/`, and `test/` folders (n=5,778 / 826 / 1,656; approximately 70/10/20)

**Class distribution:**

| KL Grade | Description | Train (n=5,778) | Validation (n=826) | Test (n=1,656) |
|---|---|---|---|---|
| KL0 | Normal | 2,286 (39.6%) | 328 (39.7%) | 639 (38.6%) |
| KL1 | Doubtful | 1,046 (18.1%) | 153 (18.5%) | 296 (17.9%) |
| KL2 | Mild | 1,516 (26.2%) | 212 (25.7%) | 447 (27.0%) |
| KL3 | Moderate | 757 (13.1%) | 106 (12.8%) | 223 (13.5%) |
| KL4 | Severe | 173 (3.0%) | 27 (3.3%) | 51 (3.1%) |

> **Note:** The dataset contains no patient-level metadata (age, sex, weight-bearing status, body mass index). This precludes subgroup-level fairness analysis and is a recognised limitation for clinical deployment.

---

## Experimental design

```
2 architectures × 2 loss functions × 3 seeds = 12 training runs
```

| Configuration | Architecture | Loss | Seeds |
|---|---|---|---|
| ResNet50 + CE | ResNet50 (23.5M params) | Cross-entropy | 42, 123, 456 |
| ResNet50 + CORAL | ResNet50 (23.5M params) | CORAL ordinal | 42, 123, 456 |
| EfficientNet-B0 + CE | EfficientNet-B0 (4M params) | Cross-entropy | 42, 123, 456 |
| EfficientNet-B0 + CORAL | EfficientNet-B0 (4M params) | CORAL ordinal | 42, 123, 456 |

Both architectures were initialised with ImageNet pretrained weights. Differential learning rates were applied: backbone at lr=1×10⁻⁵, classification head at lr=1×10⁻⁴.

---

## Primary metrics

| Metric | Description |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | Field-standard metric; penalises distant grade errors proportionally. Clinical benchmark: 0.81. |
| **KL1 recall** | Primary clinical outcome; measures detection of doubtful OA. Meta-analytic benchmark: 0.64. |

**Key results:**

| Configuration | QWK (mean ± SD) | KL1 Recall (mean ± SD) |
|---|---|---|
| ResNet50 + CE | 0.782 ± 0.003 | 0.127 ± 0.011 |
| ResNet50 + CORAL | 0.800 ± 0.004 | 0.341 ± 0.019 |
| EfficientNet-B0 + CE | 0.787 ± 0.002 | 0.190 ± 0.017 |
| EfficientNet-B0 + CORAL | **0.809 ± 0.007** | **0.354 ± 0.009** |

---

## Repository structure

```
configs/    # One YAML per experimental configuration (what was compared)
src/        # Training, evaluation, Grad-CAM, model and loss definitions (how it was implemented)
slurm/      # 12 SLURM scripts + submit_all.sh for HPC submission (how to reproduce on GPU cluster)
results/    # Figures used in the report (what was found)
docs/       # Reproducibility guide and model cards (how to understand and rerun)
notebooks/  # Exploratory data analysis and visualisation
reports/    # Quarto report source (.qmd), references, and rendered output
```

---

## Reproducing this work

Full step-by-step instructions — environment setup, dataset download, training, evaluation, figures, and expected outputs — are in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Hardware

Training was performed on the Cambridge Service for Data-Driven Discovery (CSD3) HPC:

- **GPU:** NVIDIA A100-SXM4-80GB
- **Cluster:** CSD3 (University of Cambridge)
- All 12 training runs completed within the standard CSD3 GPU allocation

---

## Environment

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| CUDA toolkit | 11.8 |
| numpy | 2.4.3 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scipy | 1.17.1 |
| Pillow | 12.1.1 |
| opencv-python | 4.13.0.92 |
| grad-cam | 1.5.5 |
| tqdm | 4.67.3 |
| PyYAML | 6.0.3 |
| kaggle | 2.0.1 |

Full specification in `environment.yml`.

---

## Citation and licence

**Dataset:** Kaggle Knee Osteoarthritis Dataset with Severity Grading — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Code:** This repository is licensed under the [MIT Licence](LICENSE).

If you use this work, please cite:

> Mzendah, T. (2026). *Does Loss Function Choice Affect Early Osteoarthritis Detection? Comparing Cross-Entropy and CORAL Ordinal Loss for Automated Kellgren-Lawrence Grading.* MSt Healthcare Data Science, University of Cambridge.
