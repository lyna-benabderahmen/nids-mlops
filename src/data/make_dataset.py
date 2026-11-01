"""Build the stratified dataset tracked by DVC from the 8 CIC-IDS2017 daily CSVs.

Memory-friendly: each daily file is loaded and sampled on its own, then the
samples are concatenated (the full merged data is ~2.8M rows).
"""
import glob, os
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fix_label(s: pd.Series) -> pd.Series:
    # Source files contain a broken character in 'Web Attack <?> ...' labels
    return s.str.replace(r"Web Attack\W+", "Web Attack - ", regex=True).str.strip()


def main(src_dir: str):
    p = load_params()["data"]
    total_rows = 2_830_743            # known size of CIC-IDS2017 MachineLearningCSV
    frac = p["sample_size"] / total_rows
    parts, seen = [], 0
    for f in sorted(glob.glob(os.path.join(src_dir, "*.csv"))):
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = fix_label(df["Label"])
        seen += len(df)
        for _, g in df.groupby(p["label_column"]):
            n = min(len(g), max(int(len(g) * frac), 300))   # keep rare classes
            parts.append(g.sample(n=n, random_state=p["random_state"]))
        del df
    out = (pd.concat(parts).sample(frac=1, random_state=p["random_state"])
             .reset_index(drop=True))
    os.makedirs(os.path.dirname(p["raw_file"]), exist_ok=True)
    out.to_csv(p["raw_file"], index=False)
    print(f"source={seen:,} rows -> tracked dataset={len(out):,} rows -> {p['raw_file']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
