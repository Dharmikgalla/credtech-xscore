import os, math, datetime as dt, logging, re
from typing import List
import pandas as pd
import numpy as np
import requests
import feedparser
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from .config import cfg
from .utils import ensure_dir, save_csv, timer

RAW_DIR = cfg.raw_dir

def _cache_path(name: str):
    return os.path.join(RAW_DIR, name)

def ingest_prices(tickers: List[str], start: str) -> pd.DataFrame:
    with timer("Ingest: Yahoo Finance prices"):
      df = yf.download(tickers, start=start, progress=False)["Adj Close"]
      if isinstance(df, pd.Series):
          df = df.to_frame(name=tickers[0])
      df = df.sort_index()
      save_csv(df, _cache_path("prices.csv"))
      return df

def _wb_fetch(country: str, indicator: str) -> pd.DataFrame:
    # World Bank API returns yearly values. We'll monthly-forward-fill later for features.
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=10000"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list) or len(js) < 2:
        raise RuntimeError("Unexpected World Bank response")
    rows = js[1]
    df = pd.DataFrame([{ "date": int(x["date"]), "value": x["value"] } for x in rows if x["value"] is not None])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def ingest_worldbank(country: str) -> pd.DataFrame:
    with timer("Ingest: World Bank macro (GDP growth, Inflation)"):
        indicators = {
            "gdp_growth": "NY.GDP.MKTP.KD.ZG",
            "inflation": "FP.CPI.TOTL.ZG",
        }
        frames = []
        for k, ind in indicators.items():
            try:
                df = _wb_fetch(country, ind)
                df["indicator"] = k
                frames.append(df)
            except Exception as e:
                logging.warning(f"World Bank fetch failed for {k}: {e}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        pivot = df.pivot_table(index="date", columns="indicator", values="value")
        pivot.index = pd.to_datetime(pivot.index, format="%Y")
        save_csv(pivot, _cache_path(f"worldbank_{country}.csv"))
        return pivot

def _news_query_for_ticker(t: str) -> str:
    # Clean ticker-like suffixes (e.g., .NS) to form a brand/company query
    base = re.sub(r"\..*$", "", t)
    return base

def ingest_news(tickers: List[str], lookback_days: int, lang: str, region: str) -> pd.DataFrame:
    with timer("Ingest: Google News RSS headlines + VADER sentiment"):
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()

        since = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days)
        rows = []
        for t in tickers:
            q = _news_query_for_ticker(t)
            # Google News RSS query
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl={lang}&gl={region}&ceid={region}:{lang}"
            feed = feedparser.parse(url)
            for e in feed.entries[:200]:
                # Parse date if available
                pub = None
                for key in ["published_parsed", "updated_parsed"]:
                    if getattr(e, key, None):
                        pub = pd.Timestamp(dt.datetime(*getattr(e, key)[:6]), tz="UTC")
                        break
                title = getattr(e, "title", "")
                if not pub:
                    continue
                if pub < since.tz_localize("UTC"):
                    continue
                score = sia.polarity_scores(title)["compound"]
                rows.append({
                    "ticker": t,
                    "query": q,
                    "title": title,
                    "published_utc": pub,
                    "sentiment": score,
                    "link": getattr(e, "link", ""),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            logging.warning("No news pulled (RSS may be throttled).")
            save_csv(pd.DataFrame(columns=["ticker","published_utc","title","sentiment","link"]).set_index(pd.Index([])), _cache_path("news.csv"))
            return df

        df["date"] = df["published_utc"].dt.tz_convert("UTC").dt.date
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("published_utc")
        save_csv(df.set_index("published_utc"), _cache_path("news.csv"))
        return df

def main():
    ensure_dir(RAW_DIR)
    prices = ingest_prices(cfg.tickers, cfg.start_date)
    macro = ingest_worldbank(cfg.default_country)
    news = ingest_news(cfg.tickers, cfg.news_lookback_days, cfg.google_news_lang, cfg.google_news_region)
    print("Done ingest:",
          f"prices={prices.shape if isinstance(prices, pd.DataFrame) else 'NA'}",
          f"macro={macro.shape if isinstance(macro, pd.DataFrame) else 'NA'}",
          f"news={news.shape if isinstance(news, pd.DataFrame) else 'NA'}")

if __name__ == "__main__":
    main()
