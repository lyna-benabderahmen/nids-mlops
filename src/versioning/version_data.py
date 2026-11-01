"""Automatic data versioning: data -> DVC, metadata -> Git, readable tag.

Called by the Streamlit app, but also usable from the command line.
"""
import json, subprocess, datetime, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # independent of launch directory
DATA = ROOT / "data" / "dataset.csv"
META = ROOT / "versions"


def run(cmd, check=True):
    if cmd[0] == "dvc":
        cmd = [sys.executable, "-m", "dvc"] + cmd[1:]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr or r.stdout}")
    return r


def version_data(df, tag: str, message: str, params: dict) -> dict:
    """Write df to data/dataset.csv, dvc add, git commit + tag, dvc push. Returns the record."""
    if run(["git", "tag", "-l", tag]).stdout.strip():
        raise ValueError(f"Tag {tag} already exists")
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
    META.mkdir(exist_ok=True)
    record = {"tag": tag, "message": message, "rows": int(len(df)),
              "columns": int(df.shape[1]), "params": params,
              "created": datetime.datetime.now().isoformat(timespec="seconds")}
    (META / f"{tag}.json").write_text(json.dumps(record, indent=2))
    run(["dvc", "add", "data/dataset.csv"])                      # data -> DVC cache
    run(["git", "add", "data/dataset.csv.dvc", "data/.gitignore", f"versions/{tag}.json"])
    run(["git", "commit", "-m", f"{tag}: {message}"])            # metadata -> Git
    run(["git", "tag", "-a", tag, "-m", message])                # readable tag
    run(["dvc", "push"])                                         # data -> remote
    return record


def list_versions():
    out = run(["git", "tag", "-l", "v*", "--sort=version:refname"]).stdout.split()
    recs = []
    for t in out:
        r = run(["git", "show", f"{t}:versions/{t}.json"], check=False)
        recs.append(json.loads(r.stdout) if r.returncode == 0
                    else {"tag": t, "message": "initial dataset", "params": {}})
    return recs


def load_version(tag: str):
    """Load any past dataset version straight from DVC storage (cache or remote)."""
    import pandas as pd, io
    tmp = ROOT / ".tmp_version.csv"
    run(["dvc", "get", ".", "data/dataset.csv", "--rev", tag, "-o", str(tmp), "--force"])
    df = pd.read_csv(tmp)
    tmp.unlink(missing_ok=True)
    return df
