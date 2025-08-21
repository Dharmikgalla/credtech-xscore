import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: os.getenv("TICKERS", "RELIANCE.NS,TCS.NS,HDFCBANK.NS").split(","))
    default_country: str = os.getenv("DEFAULT_COUNTRY", "IN")
    start_date: str = os.getenv("START_DATE", "2018-01-01")
    google_news_lang: str = os.getenv("GOOGLE_NEWS_LANG", "en")
    google_news_region: str = os.getenv("GOOGLE_NEWS_REGION", "IN")
    news_lookback_days: int = int(os.getenv("NEWS_LOOKBACK_DAYS", "90"))

    # Paths
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    scores_dir: str = "data/scores"
    reports_dir: str = "data/reports"
    models_dir: str = "models"

cfg = Config()
