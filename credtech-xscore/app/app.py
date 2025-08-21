import os, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from src.config import cfg

st.set_page_config(page_title="CredTech XScore", layout="wide")

st.title("CredTech XScore — Explainable Credit Intelligence (Demo)")

# Sidebar
tickers = cfg.tickers
ticker = st.sidebar.selectbox("Issuer (Ticker)", tickers)

# Load data
scores_path = os.path.join(cfg.scores_dir, f"{ticker}.csv")
if not os.path.exists(scores_path):
    st.warning("No scores yet. Run the pipeline first (make pipeline).")
    st.stop()

df = pd.read_csv(scores_path, parse_dates=["date"])
df = df.sort_values("date")

# Charts
st.subheader(f"Probability of distress — {ticker}")
fig = px.line(df, x="date", y="score", title="Daily score (0-1)")
st.plotly_chart(fig, use_container_width=True)

# Show contributions for latest date
latest = df.iloc[-1:]
st.subheader("Latest explanation")
contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
contrib = latest[contrib_cols].T.reset_index()
contrib.columns = ["feature","contribution"]
contrib["feature"] = contrib["feature"].str.replace("contrib_","", regex=False)
contrib = contrib.sort_values("contribution", key=lambda s: s.abs(), ascending=False)
st.dataframe(contrib, use_container_width=True)

# Recent news
news_path = os.path.join(cfg.raw_dir, "news.csv")
st.subheader("Recent headlines (last pull)")
if os.path.exists(news_path):
    news = pd.read_csv(news_path, parse_dates=["published_utc"])
    news = news[news["ticker"] == ticker].sort_values("published_utc", ascending=False).head(50)
    if len(news) == 0:
        st.info("No recent headlines ingested for this ticker.")
    else:
        news["published_utc"] = news["published_utc"].dt.tz_localize("UTC")
        st.dataframe(news[["published_utc","title","sentiment","link"]], use_container_width=True)
else:
    st.info("Run ingest to fetch headlines.")

# Metrics panel
metrics_path = os.path.join(cfg.reports_dir, "metrics.json")
st.subheader("Model metrics (holdout)")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    cols = st.columns(5)
    cols[0].metric("AUC", f"{metrics.get('auc', float('nan')):.3f}")
    cols[1].metric("F1", f"{metrics.get('f1', float('nan')):.3f}")
    cols[2].metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
    cols[3].metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
    cols[4].metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.3f}")
else:
    st.info("Train the model to see metrics.")
