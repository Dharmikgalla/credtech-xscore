import os
import pandas as pd
import numpy as np

from .config import cfg
from .utils import load_csv, save_csv, timer

def create_labels(horizon_days: int = 30, threshold: float = -0.10):
    with timer("Labels: create proxy distress"):
        feat = load_csv(os.path.join(cfg.processed_dir, "features.csv")).reset_index()
        feat = feat.sort_values(["ticker","date"])
        # Future 30d return (based on price column carried through features build)
        feat["fwd_ret_30d"] = feat.groupby("ticker")["price"].pct_change(periods=horizon_days).shift(-horizon_days)
        feat["distress"] = (feat["fwd_ret_30d"] <= threshold).astype(int)
        # Drop rows with NaNs in label
        labeled = feat.dropna(subset=["distress"]).copy()
        save_csv(labeled.set_index(["date","ticker"]), os.path.join(cfg.processed_dir, "labeled.csv"))
        return labeled

def main():
    create_labels()
    print("Labels created.")

if __name__ == "__main__":
    main()
