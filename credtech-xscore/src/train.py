import os, json, logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import pickle

from .config import cfg
from .utils import load_csv, save_csv, save_json, ensure_dir, timer

FEATURE_COLS = [
    "ret_7d","ret_30d","vol_30d","mom_30d","mdd_90d",
    "gdp_growth","inflation","news_sent_mean","news_sent_count","news_sent_neg_rate"
]

def train_model(out_model_path: str = os.path.join(cfg.models_dir, "model.pkl")):
    with timer("Train: logistic regression on proxy labels"):
        df = load_csv(os.path.join(cfg.processed_dir, "labeled.csv")).reset_index()
        df = df.dropna(subset=FEATURE_COLS + ["distress"])
        df = df.sort_values("date")

        X = df[FEATURE_COLS].values
        y = df["distress"].values

        # Time-based split (80/20)
        split_idx = int(0.8 * len(df))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        dates_test = df["date"].iloc[split_idx:].values
        tickers_test = df["ticker"].iloc[split_idx:].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        pipe.fit(X_train, y_train)

        # Evaluate
        proba = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba) if len(np.unique(y_test))>1 else float("nan")
        pred = (proba >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        acc = accuracy_score(y_test, pred)

        metrics = {
            "auc": auc,
            "f1": f1,
            "precision": p,
            "recall": r,
            "accuracy": acc,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        ensure_dir(cfg.reports_dir)
        save_json(metrics, os.path.join(cfg.reports_dir, "metrics.json"))

        # Feature coefficients
        scaler = pipe.named_steps["scaler"]
        clf = pipe.named_steps["clf"]
        coefs = clf.coef_[0] / (scaler.scale_ + 1e-12)  # rescale to original feature scale for contributions
        coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": coefs}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
        save_csv(coef_df.set_index("feature"), os.path.join(cfg.reports_dir, "coefficients.csv"))

        # Plots
        plt.figure()
        coef_df.sort_values("coef").plot(kind="barh", y="coef", x="feature", legend=False)
        plt.title("Logistic Regression Coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.reports_dir, "coefficients.png"))
        plt.close()

        # Save model
        ensure_dir(cfg.models_dir)
        with open(out_model_path, "wb") as f:
            pickle.dump({"pipeline": pipe, "feature_names": FEATURE_COLS, "coef_rescaled": coefs}, f)

        # Also save test scores for dashboard
        out = pd.DataFrame({
            "date": dates_test,
            "ticker": tickers_test,
            "y_true": y_test,
            "score": proba,
        })
        out = out.sort_values(["ticker","date"])
        save_csv(out.set_index(["date","ticker"]), os.path.join(cfg.processed_dir, "test_scores.csv"))

        return out, metrics, coef_df

def score_full_series():
    with timer("Score: full historical series per issuer"):
        # Load features (including last unlabeled tails)
        feat = load_csv(os.path.join(cfg.processed_dir, "features.csv")).reset_index()
        # Load model
        import pickle
        with open(os.path.join(cfg.models_dir, "model.pkl"), "rb") as f:
            obj = pickle.load(f)
        pipe = obj["pipeline"]
        feature_names = obj["feature_names"]
        coefs = obj["coef_rescaled"]

        # Compute scores per date/ticker
        X = feat[feature_names].fillna(0.0).values
        proba = pipe.predict_proba(X)[:,1]
        feat["score"] = proba

        # Contribution breakdown (approx) = x_i * coef_i (on original feature scale)
        X_orig = feat[feature_names].fillna(0.0).to_numpy()
        contrib = X_orig * coefs.reshape(1, -1)
        contrib_df = pd.DataFrame(contrib, columns=[f"contrib_{c}" for c in feature_names])
        out = pd.concat([feat[["date","ticker","price"]], contrib_df, feat[["score"]]], axis=1)

        for t, g in out.groupby("ticker"):
            path = os.path.join(cfg.scores_dir, f"{t}.csv")
            save_csv(g.set_index("date"), path)

        # Save latest explanation snapshot
        latest = out.sort_values("date").groupby("ticker").tail(1)
        latest_out = latest[["ticker","score"] + [c for c in contrib_df.columns]]
        save_csv(latest_out.set_index("ticker"), os.path.join(cfg.reports_dir, "latest_explanations.csv"))

def main():
    train_model()
    score_full_series()
    print("Training + scoring complete.")

if __name__ == "__main__":
    main()
