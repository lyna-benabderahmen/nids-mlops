"""Cleaning + transformation functions (shared by the DVC pipeline and the Streamlit app)."""
import numpy as np
import pandas as pd

LABEL = "Label"


def clean(df: pd.DataFrame, drop_duplicates=True, clip_infinite=True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if clip_infinite:
        df = df.replace([np.inf, -np.inf], np.nan)
    if drop_duplicates:
        df = df.drop_duplicates()
        dup_cols = df.T.duplicated()          # e.g. 'Fwd Header Length.1'
        df = df.loc[:, ~dup_cols.values]
    return df


def handle_missing(df: pd.DataFrame, strategy="median") -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes("number").columns
    if strategy == "drop":
        return df.dropna()
    if strategy == "mean":
        df[num] = df[num].fillna(df[num].mean())
    elif strategy == "median":
        df[num] = df[num].fillna(df[num].median())
    elif strategy == "ffill":
        df[num] = df[num].ffill().bfill()
    elif strategy == "zero":
        df[num] = df[num].fillna(0)
    return df


def encode_label(df: pd.DataFrame, mode="multiclass") -> pd.DataFrame:
    """mode: 'none' | 'binary' (BENIGN=0 / ATTACK=1) | 'multiclass' (integer codes)."""
    df = df.copy()
    if mode == "binary":
        df["target"] = (df[LABEL] != "BENIGN").astype(int)
    elif mode == "multiclass":
        classes = sorted(df[LABEL].unique())
        df["target"] = df[LABEL].map({c: i for i, c in enumerate(classes)})
    return df


def select_columns(df: pd.DataFrame, drop_constant=True, keep=None) -> pd.DataFrame:
    df = df.copy()
    if drop_constant:
        const = [c for c in df.columns if c != LABEL and df[c].nunique(dropna=False) <= 1]
        df = df.drop(columns=const)
    if keep:
        cols = [c for c in keep if c in df.columns]
        extra = [c for c in (LABEL, "target") if c in df.columns]
        df = df[cols + [c for c in extra if c not in cols]]
    return df


def add_split(df: pd.DataFrame, test_size=0.2, seed=42, stratify=True) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    df = df.copy()
    if test_size <= 0:
        df["split"] = "train"
        return df
    strat = df[LABEL] if stratify else None
    # classes with a single sample cannot be stratified: put them in train
    if stratify:
        vc = df[LABEL].value_counts()
        rare = df[LABEL].isin(vc[vc < 2].index)
        idx_tr, idx_te = train_test_split(df.index[~rare], test_size=test_size,
                                          random_state=seed, stratify=df.loc[~rare, LABEL])
        df["split"] = "train"
        df.loc[idx_te, "split"] = "test"
        return df
    idx_tr, idx_te = train_test_split(df.index, test_size=test_size, random_state=seed)
    df["split"] = "train"
    df.loc[idx_te, "split"] = "test"
    return df


def transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the four transformations configured in the app / params."""
    df = clean(df)
    df = handle_missing(df, cfg.get("missing", "median"))
    df = encode_label(df, cfg.get("encoding", "multiclass"))
    df = select_columns(df, cfg.get("drop_constant", True), cfg.get("columns"))
    df = add_split(df, cfg.get("test_size", 0.2), cfg.get("seed", 42), cfg.get("stratify", True))
    return df.reset_index(drop=True)


if __name__ == "__main__":      # DVC stage entry point
    import yaml
    P = yaml.safe_load(open("params.yaml"))
    raw = pd.read_csv(P["data"]["raw_file"])
    out = transform(raw, {"missing": P["preprocess"]["missing_strategy"],
                          "encoding": "multiclass", "test_size": 0.0})
    out.to_csv(f"{P['data']['processed_dir']}/train_dataset.csv", index=False)
    print("processed:", out.shape)
