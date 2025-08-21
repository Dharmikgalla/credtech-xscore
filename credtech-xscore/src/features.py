import os, logging
import pandas as pd
import numpy as np

from .config import cfg
from .utils import load_csv, save_csv, ensure_dir, timer

PROC_DIR = cfg.processed_dir

def _rolling_max_drawdown(series: pd.Series, window: int = 90):
    roll_max = series.rolling(window=window, min_periods=1).max()
    dd = series / roll_max - 1.0
    return dd

def build_features():
    with timer("Features: prices -> tech factors"):
        prices = load_csv(os.path.join(cfg.raw_dir, "prices.csv"))
        prices = prices.ffill().dropna(how="all")
        prices = prices.reset_index().rename(columns={"Date":"date"})
        prices["date"] = pd.to_datetime(prices["date"])
        long = prices.melt(id_vars=["date"], var_name="ticker", value_name="price")

        long = long.sort_values(["ticker","date"]).dropna()
        long["ret_7d"] = long.groupby("ticker")["price"].pct_change(7)
        long["ret_30d"] = long.groupby("ticker")["price"].pct_change(30)
        long["vol_30d"] = long.groupby("ticker")["ret_7d"].rolling(30, min_periods=5).std().reset_index(level=0, drop=True)
        long["mom_30d"] = long.groupby("ticker")["price"].transform(lambda s: s / s.shift(30) - 1.0)
        long["mdd_90d"] = long.groupby("ticker")["price"].transform(lambda s: _rolling_max_drawdown(s, 90))
        tech = long.dropna().copy()

    with timer("Features: macro join"):
        macro_path = os.path.join(cfg.raw_dir, f"worldbank_{cfg.default_country}.csv")
        if os.path.exists(macro_path):
            macro = load_csv(macro_path).reset_index().rename(columns={"index":"date"})
            macro["date"] = pd.to_datetime(macro["date"])
            # Convert annual to monthly, then daily via forward-fill
            macro_m = macro.set_index("date").resample("MS").ffill()
            macro_d = macro_m.resample("D").ffill().reset_index()
        else:
            logging.warning("Macro file missing; filling with NaNs.")
            macro_d = pd.DataFrame({"date": tech["date"].unique()})
        feat = tech.merge(macro_d, on="date", how="left")

    with timer("Features: news sentiment aggregation"):
        news_path = os.path.join(cfg.raw_dir, "news.csv")
        if os.path.exists(news_path) and os.path.getsize(news_path) > 0:
            news = load_csv(news_path).reset_index().rename(columns={"index":"published_utc"})
            news["date"] = pd.to_datetime(news["date"])
            agg = news.groupby(["ticker","date"]).agg(
                news_sent_mean=("sentiment","mean"),
                news_sent_count=("sentiment","count"),
                news_sent_neg_rate=("sentiment", lambda x: (np.array(x) < -0.2).mean() if len(x)>0 else np.nan),
            ).reset_index()
            feat = feat.merge(agg, on=["ticker","date"], how="left")
        else:
            feat["news_sent_mean"] = np.nan
            feat["news_sent_count"] = 0
            feat["news_sent_neg_rate"] = np.nan

    feat = feat.sort_values(["ticker","date"]).reset_index(drop=True)
    ensure_dir(PROC_DIR)
    save_csv(feat.set_index(["date","ticker"]), os.path.join(PROC_DIR, "features.csv"))
    return feat

def main():
    build_features()
    print("Features built.")

if __name__ == "__main__":
    main()
