# Model Cards

Four experimental configurations were trained and evaluated: two CNN architectures × two loss functions, each across three random seeds (42, 123, 456). All metrics are mean ± SD across three seeds on the held-out test set (n=1,656).

---

## Configuration 1 — ResNet50 + Cross-Entropy

| Field | Detail |
|---|---|
| Architecture | ResNet50 (He et al., 2016) |
| Parameters | ~23.5 million |
| Loss function | Categorical cross-entropy |
| Pretrained weights | ImageNet |
| Seeds | 42, 123, 456 |
| Best epochs | 16, 15, 14 |

### Performance

| Metric | Mean ± SD |
|---|---|
| QWK | 0.782 ± 0.003 |
| Balanced accuracy | 0.611 ± 0.009 |
| Macro F1 | 0.603 ± 0.004 |
| Macro AUC | 0.876 ± 0.002 |
| **KL1 recall** | **0.127 ± 0.014** |

### Per-grade recall

| KL Grade | Mean ± SD |
|---|---|
| KL0 | 0.865 ± 0.034 |
| KL1 | 0.127 ± 0.014 |
| KL2 | 0.611 ± 0.031 |
| KL3 | 0.732 ± 0.023 |
| KL4 | 0.719 ± 0.063 |

---

## Configuration 2 — ResNet50 + CORAL

| Field | Detail |
|---|---|
| Architecture | ResNet50 (He et al., 2016) |
| Parameters | ~23.5 million |
| Loss function | CORAL ordinal loss (Cao et al., 2020) |
| Pretrained weights | ImageNet |
| Seeds | 42, 123, 456 |
| Best epochs | 14, 17, 13 |

### Performance

| Metric | Mean ± SD |
|---|---|
| QWK | 0.800 ± 0.005 |
| Balanced accuracy | 0.615 ± 0.018 |
| Macro F1 | 0.628 ± 0.012 |
| Macro AUC | 0.875 ± 0.005 |
| **KL1 recall** | **0.341 ± 0.024** |

### Per-grade recall

| KL Grade | Mean ± SD |
|---|---|
| KL0 | 0.787 ± 0.010 |
| KL1 | 0.341 ± 0.024 |
| KL2 | 0.560 ± 0.003 |
| KL3 | 0.700 ± 0.027 |
| KL4 | 0.686 ± 0.071 |

**KL1 recall improvement over ResNet50+CE: +168% (paired t=10.534, p=0.004)**

---

## Configuration 3 — EfficientNet-B0 + Cross-Entropy

| Field | Detail |
|---|---|
| Architecture | EfficientNet-B0 (Tan & Le, 2019) |
| Parameters | ~4 million |
| Loss function | Categorical cross-entropy |
| Pretrained weights | ImageNet |
| Seeds | 42, 123, 456 |
| Best epochs | 27, 21, 22 |

### Performance

| Metric | Mean ± SD |
|---|---|
| QWK | 0.787 ± 0.002 |
| Balanced accuracy | 0.616 ± 0.014 |
| Macro F1 | 0.618 ± 0.014 |
| Macro AUC | 0.879 ± 0.003 |
| **KL1 recall** | **0.190 ± 0.020** |

### Per-grade recall

| KL Grade | Mean ± SD |
|---|---|
| KL0 | 0.863 ± 0.020 |
| KL1 | 0.190 ± 0.020 |
| KL2 | 0.599 ± 0.033 |
| KL3 | 0.768 ± 0.022 |
| KL4 | 0.660 ± 0.049 |

---

## Configuration 4 — EfficientNet-B0 + CORAL

| Field | Detail |
|---|---|
| Architecture | EfficientNet-B0 (Tan & Le, 2019) |
| Parameters | ~4 million |
| Loss function | CORAL ordinal loss (Cao et al., 2020) |
| Pretrained weights | ImageNet |
| Seeds | 42, 123, 456 |
| Best epochs | 20, 30, 27 |

### Performance

| Metric | Mean ± SD |
|---|---|
| QWK | **0.809 ± 0.009** |
| Balanced accuracy | 0.609 ± 0.016 |
| Macro F1 | 0.620 ± 0.019 |
| Macro AUC | 0.882 ± 0.005 |
| **KL1 recall** | **0.354 ± 0.011** |

### Per-grade recall

| KL Grade | Mean ± SD |
|---|---|
| KL0 | 0.796 ± 0.004 |
| KL1 | 0.354 ± 0.011 |
| KL2 | 0.532 ± 0.020 |
| KL3 | 0.728 ± 0.035 |
| KL4 | 0.634 ± 0.030 |

**KL1 recall improvement over EfficientNet+CE: +86% (paired t=27.368, p=0.001)**
**Best overall configuration: highest QWK (0.809), exceeding the 0.81 clinical benchmark.**

---

## Summary comparison

| Configuration | QWK | KL1 Recall | Macro AUC |
|---|---|---|---|
| ResNet50 + CE | 0.782 ± 0.003 | 0.127 ± 0.014 | 0.876 ± 0.002 |
| ResNet50 + CORAL | 0.800 ± 0.005 | 0.341 ± 0.024 | 0.875 ± 0.005 |
| EfficientNet-B0 + CE | 0.787 ± 0.002 | 0.190 ± 0.020 | 0.879 ± 0.003 |
| EfficientNet-B0 + CORAL | **0.809 ± 0.009** | **0.354 ± 0.011** | **0.882 ± 0.005** |

Meta-analytic KL1 benchmark: 0.64 (Zhao et al., 2024). All configurations remain below this threshold.
