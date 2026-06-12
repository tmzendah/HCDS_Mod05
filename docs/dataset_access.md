# Dataset Access

## Kaggle Knee Osteoarthritis Dataset with Severity Grading

| Field | Detail |
|---|---|
| Source | Kaggle / Mendeley Data |
| Author | Chen, Pingjun |
| Year | 2018 |
| DOI | 10.17632/56rmx5bjcr.1 |
| Licence | CC BY 4.0 |
| Images | 8,260 anteroposterior knee radiographs |
| Labels | Kellgren-Lawrence grades 0–4 (radiologist-assigned) |
| Image size | 224 × 224 px (pre-processed, single-knee crops) |
| Format | PNG |

### Split

| Partition | n | Notes |
|---|---|---|
| Train | 5,778 | Pre-defined; retained without modification |
| Validation | 826 | Pre-defined |
| Test | 1,656 | Held out; used for all reported metrics |

The original split was preserved to avoid data leakage risk (no patient-level metadata available to verify patient separation across partitions) and to maintain comparability with published benchmarks on this dataset.

### Class distribution

| KL Grade | Description | Train | Validation | Test |
|---|---|---|---|---|
| 0 | Normal | 2,286 (39.6%) | 328 (39.7%) | 639 (38.6%) |
| 1 | Doubtful | 1,046 (18.1%) | 153 (18.5%) | 296 (17.9%) |
| 2 | Mild | 1,516 (26.2%) | 212 (25.7%) | 447 (27.0%) |
| 3 | Moderate | 757 (13.1%) | 106 (12.8%) | 223 (13.5%) |
| 4 | Severe | 173 (3.0%) | 27 (3.3%) | 51 (3.1%) |

### Download instructions

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
3. Run:

```bash
kaggle datasets download -d shashwatwork/knee-osteoarthritis-dataset-with-severity
unzip knee-osteoarthritis-dataset-with-severity.zip -d data/
```

Expected structure after unzip:

```
data/
  train/   0/  1/  2/  3/  4/
  val/     0/  1/  2/  3/  4/
  test/    0/  1/  2/  3/  4/
```

Update `data_dir` in each config file under `configs/` to point to your local `data/` path.

### Metadata availability

The dataset contains **no patient-level metadata** (age, sex, weight-bearing status, body mass index). This precludes subgroup-level fairness analysis and is a recognised limitation for clinical deployment.

### Citation

```
Chen, P. (2018). Knee Osteoarthritis Severity Grading Dataset.
Mendeley Data, v1. https://doi.org/10.17632/56rmx5bjcr.1
```
