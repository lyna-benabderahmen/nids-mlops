"""Emit train.csv / test.csv from the processed dataset (DVC stage)."""
import yaml
import pandas as pd
from preprocess import add_split

if __name__ == "__main__":
    P = yaml.safe_load(open("params.yaml"))
    d = P["data"]["processed_dir"]
    df = pd.read_csv(f"{d}/train_dataset.csv")
    df = add_split(df, P["split"]["test_size"], P["split"]["random_state"], P["split"]["stratify"])
    df[df.split == "train"].drop(columns="split").to_csv(f"{d}/train.csv", index=False)
    df[df.split == "test"].drop(columns="split").to_csv(f"{d}/test.csv", index=False)
    print(df.split.value_counts().to_dict())
