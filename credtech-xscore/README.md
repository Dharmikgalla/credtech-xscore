# credtech-xscore

Real-time, explainable credit risk scoring (toy but end-to-end) for listed issuers.
Ingests **Yahoo Finance** (prices), **World Bank** (macro indicators), and **Google News RSS** (unstructured headlines), 
engineers features, derives proxy distress labels, trains a **logistic regression** model, produces per-issuer time-series scores,
and serves an **analyst dashboard** with Streamlit.

> ⚠️ This is a reference implementation for demos/hackathons. It is intentionally lightweight and avoids paid APIs.
> It **does require internet access** at runtime to fetch public data (yfinance, World Bank API, Google News RSS).

---

## Quickstart (Local)

```bash
# 1) Clone (or unzip) and cd
cd credtech-xscore

# 2) Create env and install deps
python3 -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure
cp .env.example .env
# Optionally edit .env to change tickers, dates, etc.

# 4) Run end-to-end pipeline
make pipeline          # or: python -m src.pipeline

# 5) Launch dashboard
make app               # or: streamlit run app/app.py
```

The pipeline will produce:
- `models/model.pkl` (trained logistic regression)
- `data/scores/<TICKER>.csv` (daily probability-of-distress style score)
- `data/reports/metrics.json` (AUROC, F1, etc.), `coefficients.csv`, and PNG plots
- `data/raw/*.csv` caches of pulled sources

---

## Docker

```bash
# Build image
make docker-build
# Run pipeline inside container (writes to /workspace mounted volume)
make docker-run-pipeline
# Run Streamlit app (mapped to localhost:8501)
make docker-run-app
```

---

## Repo Layout

```
credtech-xscore/
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── .env.example
├── data/
│   ├── raw/          # cached raw pulls
│   ├── processed/    # featuresets, labels, train/test splits
│   ├── scores/       # time series scores per issuer
│   └── reports/      # metrics.json, coefficients, plots
├── models/
│   └── model.pkl
├── app/
│   └── app.py        # Streamlit analyst dashboard
└── src/
    ├── __init__.py
    ├── config.py     # config defaults
    ├── utils.py      # helpers
    ├── ingest.py     # yfinance + World Bank + Google News RSS
    ├── features.py   # feature engineering
    ├── labels.py     # proxy distress labels
    ├── train.py      # train logistic regression
    └── pipeline.py   # end-to-end orchestration
```

---

## How it works (high level)

1. **Ingest**
   - **Prices** via `yfinance` for each ticker (`Adj Close` daily).
   - **Macro** via World Bank REST (`GDP growth` = `NY.GDP.MKTP.KD.ZG`, `Inflation` = `FP.CPI.TOTL.ZG`) by `DEFAULT_COUNTRY`.
   - **News** via Google News RSS. Headlines are scored with **NLTK VADER** sentiment and aggregated per day/issuer.

2. **Feature engineering**
   - Returns (7d/30d), rolling volatility (30d), 90d max drawdown, 30d momentum.
   - Macro is forward-filled monthly → daily.
   - News sentiment aggregated daily (mean, count, %negative).

3. **Labels (proxy)**  
   Distress = 1 if **30d forward return ≤ -10%**; else 0.

4. **Model & Explainability**
   - `LogisticRegression` on standardized features; time-based split for validation.
   - Coefficient-based **per-feature contributions** (`x_i * coef_i`) for “why” at each scoring date (no external SHAP dependency).

5. **Scoring & Dashboard**
   - Daily probability scores written under `data/scores/`.
   - Streamlit shows **score trends**, **latest explanation**, **feature effects**, and **recent headlines** per issuer.

---

## Configuration

Edit `.env` (or override via CLI env vars):

- `TICKERS` – comma-separated tickers (default: `RELIANCE.NS,TCS.NS,HDFCBANK.NS`).
- `DEFAULT_COUNTRY` – World Bank country (ISO-2 or ISO-3), default `IN`.
- `START_DATE` – data start (default `2018-01-01`).
- `NEWS_LOOKBACK_DAYS` – how many days of headlines to fetch each run (default `90`).
- `GOOGLE_NEWS_LANG` – language code, default `en`.
- `GOOGLE_NEWS_REGION` – e.g. `IN`, `US` (affects RSS edition), default `IN`.

---

## Repro/MLOps notes

- Deterministic seeds for model split.
- Caching raw pulls to `data/raw/` to avoid refetching.
- Idempotent pipeline stages; each can be invoked directly (`python -m src.ingest`, etc.).
- Error-tolerant ingestion: missing sources won’t crash the run; warnings are logged.

---

## License

MIT (include attribution if you build on this).
