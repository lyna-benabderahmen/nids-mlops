# NIDS MLOps - Network Intrusion Detection (CIC-IDS2017)

Git tracks code + metadata, DVC tracks data, a Streamlit app creates tagged dataset versions.

```
params.yaml                 config (split ratios, imputation, data_version)
app/app.py                  Streamlit UI -> "Save as new version"
src/data/make_dataset.py    builds data/dataset.csv from the 8 CIC-IDS2017 CSVs (stratified sample)
src/data/preprocess.py      clean / impute / encode / select / split
src/data/split.py           train.csv / test.csv
src/versioning/version_data.py   dvc add -> git commit -> git tag -> dvc push
versions/vX.Y.json          parameters that produced each version
dvc.yaml                    preprocess + split pipeline
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc remote modify localremote url /path/to/your/dvc-storage   # keep it outside the repo
dvc pull                       # fetch data/dataset.csv
streamlit run app/app.py
```
Rebuild the dataset from the raw CSVs: `python src/data/make_dataset.py path/to/MachineLearningCVE`
