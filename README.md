# NIDS MLOps: Network Intrusion Detection with Versioned Data

Machine-learning pipeline foundation for detecting and classifying network attacks on the **CIC-IDS2017** dataset, with **reproducible data versioning** (Git + DVC) and an interactive **Streamlit** app that creates a new tagged dataset version with one click.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![DVC](https://img.shields.io/badge/DVC-data%20versioning-945dd6) ![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b) ![Domain](https://img.shields.io/badge/domain-cybersecurity%20%2B%20ML-black)

## Problem

Classify network flows as **BENIGN** or as one of 14 attack types (DoS/DDoS, Port Scan, Brute Force, Botnet, Web Attacks, Infiltration, Heartbleed). Intrusion detection is a core cybersecurity task, and ML can analyse traffic volumes that rules alone cannot.

## Dataset

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity): 2,830,743 labeled network flows over 5 days, 78 flow features plus a label, 15 classes.

Data issues found and handled in this project:

| Issue | Handling |
|---|---|
| 1,358 NaN and 4,376 infinite values (`Flow Bytes/s`, `Flow Packets/s`) | infinities converted to NaN, then imputed (median / mean / zero / ffill / drop) |
| Duplicate column `Fwd Header Length.1`, constant columns | removed during cleaning |
| Broken character in `Web Attack ...` labels | normalised to `Web Attack - ...` |
| Extreme class imbalance (2.27M BENIGN vs 11 Heartbleed) | stratified sampling and stratified train/test split |

The version-controlled dataset is a **stratified sample** (about 61.6k rows, at least 300 per class where available) so the app stays fast. `make_dataset.py` rebuilds it from the full CSVs.

## How data versioning works

```
Streamlit app  ->  version_data()  ->  dvc add      (data -> DVC cache)
                                   ->  git commit   (pointer + params -> Git)
                                   ->  git tag vX.Y (readable version)
                                   ->  dvc push     (data -> remote storage)
```

Git stores only a small `dataset.csv.dvc` pointer and a `versions/vX.Y.json` file with the exact parameters that produced the version. The data itself lives in DVC, so checking out a tag restores the exact code and data together.

| Version | Transformation (created from the app) |
|---|---|
| v1.0 | initial stratified sample, 61,623 rows x 79 columns |
| v1.1 | median imputation, multiclass target, 80/20 split |
| v1.2 | mean imputation, binary BENIGN/ATTACK target, 70/30 split |
| v1.3 | zero fill, multiclass target, 80/20 split |

## Project structure

```
params.yaml                       configuration (split ratios, imputation, seed)
app/app.py                        Streamlit app: explore, transform, save as new version
src/data/make_dataset.py          build the stratified dataset from the 8 CIC-IDS2017 CSVs
src/data/preprocess.py            clean / impute / encode / select columns / split
src/data/split.py                 emit train.csv and test.csv
src/versioning/version_data.py    dvc add -> git commit -> git tag -> dvc push
versions/vX.Y.json                parameters that produced each version
dvc.yaml                          reproducible preprocess + split pipeline
```

## Quick start

```bash
git clone https://github.com/lyna-benabderahmen/nids-mlops.git
cd nids-mlops
pip install -r requirements.txt

# the dataset lives in a DVC remote, not in GitHub:
dvc remote modify localremote url /path/to/your/dvc-storage
dvc pull

python3 -m streamlit run app/app.py
```

Rebuild the dataset from the raw CSVs:

```bash
python3 src/data/make_dataset.py path/to/MachineLearningCVE
```

Run the pipeline: `dvc repro`. Load an old version: `dvc get . data/dataset.csv --rev v1.1`.

## The app

- **Data before cleaning**: shape, preview, dtypes, null and infinite counts
- **Visualise**: class distribution and feature histograms with takeaways
- **Create a version**: four transformations (target encoding, missing values, column selection, train/test split) with live impact, then **Save as new version**
- **Version history**: browse every tag, see its parameters, load any past version from DVC

## Roadmap

- [x] Data cleaning, versioning and reproducible pipeline
- [ ] Baseline models (Logistic Regression, Decision Tree), then Random Forest and XGBoost
- [ ] Evaluation per attack class: precision, recall, F1, confusion matrix
- [ ] Experiment tracking with MLflow and model registry
- [ ] Prediction API and monitoring dashboard for alerts
- [ ] CI/CD with data validation checks

## Author

**Lyna Ben Abderahmen**, Applied Computer Science / AI student at Pristini School of AI (Sousse, Tunisia). Interests: machine learning, deep learning and cybersecurity. Built as an MLOps course project.
