"""NIDS data dashboard + automatic data versioning (Git + DVC).

Run from anywhere:  streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # data path independent of launch dir
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "data"))

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.data.preprocess import transform, LABEL
from src.versioning.version_data import version_data, list_versions, load_version

st.set_page_config(page_title="NIDS Data Versioning", layout="wide")
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


@st.cache_data(show_spinner=False)
def load_base() -> pd.DataFrame:
    """The tracked dataset at tag v1.0: every new version is derived from it,
    so each version is reproducible from its recorded parameters."""
    return load_version("v1.0")


@st.cache_data(show_spinner=False)
def load_tag(tag: str) -> pd.DataFrame:
    return load_version(tag)


def next_tag(versions) -> str:
    nums = [int(v["tag"].split(".")[1]) for v in versions if v["tag"].startswith("v1.")]
    return f"v1.{max(nums, default=-1) + 1}"


st.title("Network Intrusion Detection - dataset versioning")
st.caption("Data goes to DVC, metadata to Git, every version gets a readable tag.")

versions = list_versions()
base = load_base()

tab_raw, tab_viz, tab_make, tab_hist = st.tabs(
    ["Data before cleaning", "Visualise", "Create a version", "Version history"])

# ---------------------------------------------------------------- PART 1
with tab_raw:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(base):,}")
    c2.metric("Columns", base.shape[1])
    num = base.drop(columns=LABEL)
    c3.metric("NaN / inf cells", f"{int(num.isna().sum().sum()):,} / {int(np.isinf(num.to_numpy('float64')).sum()):,}")
    st.dataframe(base.head(20), width="stretch")
    left, right = st.columns(2)
    left.write("**Dtypes**")
    left.dataframe(base.dtypes.astype(str).rename("dtype"), height=250)
    right.write("**Null counts (top 10)**")
    right.dataframe(base.isna().sum().sort_values(ascending=False).head(10).rename("nulls"))

# ---------------------------------------------------------------- PART 2
with tab_viz:
    counts = base[LABEL].value_counts()
    st.bar_chart(counts)
    st.caption(f"Takeaway: {counts.iloc[0] / len(base):.0%} of flows are {counts.index[0]}; "
               f"{(counts < 100).sum()} classes have fewer than 100 samples, so per-class "
               "metrics matter more than accuracy.")
    col = st.selectbox("Feature to inspect", [c for c in base.columns if c != LABEL], index=1)
    clipped = base[col].replace([np.inf, -np.inf], np.nan).dropna()
    hist, edges = np.histogram(np.log10(clipped.clip(lower=0) + 1), bins=30)
    st.bar_chart(pd.Series(hist, index=np.round(edges[:-1], 2), name="count"))
    st.caption("Takeaway: network features are heavy-tailed, so a log scale is shown; "
               "scaling or tree-based models are needed downstream.")

# ---------------------------------------------------------------- PART 3 / Task 3
with tab_make:
    st.subheader("Transformations (applied to the v1.0 base)")
    a, b = st.columns(2)
    with a:
        encoding = st.selectbox("1. Target encoding", ["multiclass", "binary", "none"],
                                help="multiclass = integer per attack type; binary = BENIGN/ATTACK")
        missing = st.selectbox("2. Missing / infinite values", ["median", "mean", "zero", "ffill", "drop"])
    with b:
        feats = [c for c in base.columns if c != LABEL]
        drop_const = st.checkbox("3. Drop constant columns", value=True)
        keep = st.multiselect("3. Columns to keep (empty = all)", feats, default=[])
        test_size = st.slider("4. Test split", 0.0, 0.5, 0.2, 0.05)
    cfg = {"encoding": encoding, "missing": missing, "drop_constant": drop_const,
           "columns": keep or None, "test_size": test_size, "seed": PARAMS["split"]["random_state"],
           "stratify": True}

    out = transform(base, cfg)                     # in memory only until saved
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(out):,}", f"{len(out) - len(base):,}")
    m2.metric("Columns", out.shape[1], out.shape[1] - base.shape[1])
    m3.metric("Missing cells", int(out.isna().sum().sum()))
    m4.metric("Test rows", int((out["split"] == "test").sum()))
    st.dataframe(out.head(10), width="stretch")

    st.divider()
    tag = st.text_input("Version tag", value=next_tag(versions))
    message = st.text_input("Commit message", value=f"{missing} imputation, {encoding} target")
    if st.button("Save as new version", type="primary"):
        try:
            rec = version_data(out, tag, message, cfg)
            load_tag.clear()
            st.success(f"Saved {rec['tag']}: {rec['rows']:,} rows -> DVC, metadata -> Git.")
        except Exception as e:                      # tag exists, dvc/git error ...
            st.error(str(e))

# ---------------------------------------------------------------- history
with tab_hist:
    versions = list_versions()
    st.dataframe(pd.DataFrame([{"tag": v["tag"], "message": v.get("message", ""),
                                "rows": v.get("rows"), "columns": v.get("columns"),
                                "created": v.get("created")} for v in versions]),
                 width="stretch")
    pick = st.selectbox("Load a past version", [v["tag"] for v in versions][::-1])
    rec = next(v for v in versions if v["tag"] == pick)
    st.write("**Parameters that produced it**")
    st.json(rec.get("params") or {"note": "initial dataset, no transformations"})
    if st.button("Load this version"):
        df = load_tag(pick)
        st.write(f"{pick}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        st.dataframe(df.head(20), width="stretch")
