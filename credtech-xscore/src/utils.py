import os, json, logging, time
from contextlib import contextmanager
from typing import Any, Dict
import pandas as pd

# Basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=True)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def timer(msg: str):
    @contextmanager
    def _timer():
        logging.info(f"▶ {msg}")
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            logging.info(f"✔ {msg} in {dt:.2f}s")
    return _timer()

def robust_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left"):
    # Convenience wrapper that keeps index and sorts
    out = left.merge(right, on=on, how=how)
    out = out.sort_values(on)
    out = out.set_index(on)
    return out
