# Data

Data files are not included in this repository. Download each dataset from the sources below and place the files as described.

---

## MRKR — Primary training dataset

**Source:** AWS Open Data Registry  
**URL:** https://registry.opendata.aws/mrkr  
**Licence:** CC-BY-SA  
**Size:** ~503,261 radiographs (large download — use AWS CLI)

```bash
aws s3 sync s3://emory-mrkr/ data/raw/mrkr/ --no-sign-request
```

After download, expected structure:

```
data/raw/mrkr/
└── dicoms/
    └── [patient_id]/
        └── [study]/
            └── *.dcm
```

Convert DICOMs to JPEG before training:

```bash
python scripts/convert_dicom_to_jpeg.py \
  --input_dir data/raw/mrkr/dicoms \
  --output_dir data/processed \
  --size 512
```

---

## OAI/Kaggle — External validation dataset

**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity  
**Licence:** CC BY 4.0  
**Size:** ~8,000 images (manageable download)

```bash
# Requires a Kaggle account and kaggle CLI configured
kaggle datasets download shashwatwork/knee-osteoarthritis-dataset-with-severity \
  -p data/raw/kaggle_oai --unzip
```

After download, expected structure:

```
data/raw/kaggle_oai/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
└── test/
```

---

## Processed data

`data/processed/` is populated by `scripts/convert_dicom_to_jpeg.py` and `scripts/prepare_data.py`. Do not commit these files.

## Metadata

`data/metadata/` contains CSV files generated during data preparation (split manifests, filter cascade logs). These are also not committed — they are reproduced by running `scripts/prepare_data.py`.
