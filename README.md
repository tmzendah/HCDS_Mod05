# Does Loss Function Choice Affect Early Osteoarthritis Detection?

**Talita Mzendah | MSt Healthcare Data Science | University of Cambridge**

---

## Project summary

Knee osteoarthritis (OA) is a leading cause of musculoskeletal disability globally, with radiographic severity graded using the Kellgren-Lawrence (KL) scale. Automated grading using convolutional neural networks (CNNs) has shown promise, yet KL Grade 1 (doubtful OA) remains poorly detected across architectures, with a meta-analytic sensitivity of only 0.64. Because KL grading is ordinal, existing ordinal-regression literature suggests that training objectives preserving grade order may be better aligned with the task than categorical cross-entropy (CE), which treats KL grades as unordered classes. However, it remains clinically important to quantify whether this advantage improves the specific early-OA failure mode of KL1 detection, rather than only overall grading performance.

This study compared CE and CORAL ordinal loss across two CNN architectures (ResNet50 and EfficientNet-B0) on the Kaggle Knee Osteoarthritis Dataset (n=8,260 anteroposterior radiographs, KL grades 0–4) across three random seeds, yielding 12 training runs. Primary outcomes were KL1 recall and Quadratic Weighted Kappa (QWK). Grad-CAM explainability analysis was performed on shared misclassification cases, with heatmaps inspected by the author, a qualified MSK radiographer, to identify recurring patterns in model attention and error behaviour. CORAL loss consistently improved KL1 recall across both architectures: ResNet50+CORAL achieved 0.341 ± 0.023 versus 0.127 ± 0.013 for ResNet50+CE, and EfficientNet+CORAL achieved 0.354 ± 0.011 versus 0.190 ± 0.021 for EfficientNet+CE. EfficientNet-B0+CORAL achieved the highest mean QWK of 0.809 ± 0.009, approaching the 0.81 "almost perfect agreement" threshold. Despite these improvements, all configurations remained below the meta-analytic KL1 benchmark. These findings support CORAL as a more clinically appropriate objective than CE for ordinal KL grading, but also show that loss-function optimisation alone is insufficient for reliable early OA detection. The persistent KL1 performance gap points to a broader limitation of image-only models: borderline radiographic appearances may require acquisition and patient context, including weight-bearing status, age and body habitus, to support clinically meaningful interpretation.

---

## Research question

> Does CORAL ordinal loss improve KL Grade 1 recall compared with categorical cross-entropy across CNN architectures for automated knee OA grading?

---

## Dataset

**Kaggle Knee Osteoarthritis Dataset with Severity Grading**

- Source: [https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
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
| ResNet50 + CORAL | 0.800 ± 0.004 | 0.341 ± 0.019 (+168%) |
| EfficientNet-B0 + CE | 0.787 ± 0.002 | 0.190 ± 0.017 |
| EfficientNet-B0 + CORAL | **0.809 ± 0.007** | **0.354 ± 0.009** (+86%) |

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
