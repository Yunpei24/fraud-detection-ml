import os
import yaml
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def build(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    src = cfg["dataset"]["source"]
    out = cfg["dataset"]["output"]
    interim_dir = cfg["dataset"]["interim_dir"]
    processed_dir = cfg["dataset"]["processed_dir"]
    drop_cols = cfg["dataset"].get("drop_columns", [])
    scale_cols = cfg["dataset"].get("scale_columns", [])

    Path(interim_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df.to_parquet(out, index=False)
    print(f"Saved processed dataset to {out} with shape {df.shape}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build(args.config)
