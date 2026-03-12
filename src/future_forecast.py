import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ────────────────────────── CONFIG ──────────────────────────
BASE_DIR   = "E:/bitcoin-time-series-forecasting"
CSV_INPUT  = f"{BASE_DIR}/data/processed/predictions_improved.csv"
CSV_DAILY  = f"{BASE_DIR}/data/processed/future_dynamic_predictions_daily.csv"
CSV_MONTHLY= f"{BASE_DIR}/data/processed/future_dynamic_predictions_monthly.csv"

WINDOW       = 180
FUTURE_DAYS  = 60
USE_LOG      = True
RETURN_SCALE = 2.0  # amplify predicted returns for realistic movement

# ────────────────────────── LOAD DATA ──────────────────────────
df = pd.read_csv(CSV_INPUT)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date").dropna(subset=["ensemble"])
prices = df["ensemble"].values.astype(float)
series = np.log1p(prices) if USE_LOG else prices

# log-returns
log_returns = np.diff(series)
print(f"Loaded {len(prices)} days, computed {len(log_returns)} log-returns")

# ────────────────────────── FEATURE FUNCTION ──────────────────────────
def make_features(hist, log_returns_hist):
    vals = []
    # Lags of log-price
    vals.append(hist[-1])
    vals.append(hist[-2] if len(hist) >= 2 else hist[-1])
    vals.append(hist[-7] if len(hist) >= 7 else hist[-1])
    # Lagged returns
    vals.append(log_returns_hist[-1] if len(log_returns_hist) > 0 else 0)
    vals.append(np.mean(log_returns_hist[-5:]) if len(log_returns_hist) >= 5 else 0)
    vals.append(np.mean(log_returns_hist[-14:]) if len(log_returns_hist) >= 14 else 0)
    # Volatility
    vals.append(np.std(log_returns_hist[-14:]) if len(log_returns_hist) >= 14 else 0)
    # Trend
    vals.append(hist[-1] - hist[-30] if len(hist) >= 30 else 0)
    return np.array(vals)

# ────────────────────────── TRAINING ──────────────────────────
X_raw, y = [], []

for i in range(WINDOW, len(series)-1):
    hist = series[i-WINDOW:i+1]
    feats = make_features(hist[:-1], log_returns[i-WINDOW:i])
    X_raw.append(feats)
    y.append(log_returns[i])

X_raw = np.array(X_raw)
y = np.array(y)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

ridge = RidgeCV(alphas=[1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1], cv=5, scoring="r2")
ridge.fit(X, y)
print(f"→ Best alpha: {ridge.alpha_:.6f}, Train R²: {r2_score(y, ridge.predict(X)):.4f}")

# ────────────────────────── RECURSIVE FORECAST ──────────────────────────
history_log = series.copy()
history_returns = log_returns.copy()
future_log_returns = []

for day in range(FUTURE_DAYS):
    feats = make_features(history_log, history_returns)
    feats_sc = scaler.transform(feats.reshape(1, -1))
    pred_ret = ridge.predict(feats_sc)[0] * RETURN_SCALE  # amplify returns
    future_log_returns.append(pred_ret)
    # update series
    next_log_price = history_log[-1] + pred_ret
    history_log = np.append(history_log, next_log_price)
    history_returns = np.append(history_returns, pred_ret)

# back to price
future_preds = np.expm1(history_log[-FUTURE_DAYS:]) if USE_LOG else history_log[-FUTURE_DAYS:]

# dates
future_dates = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq="D")

# ────────────────────────── SAVE ──────────────────────────
daily_df = pd.DataFrame({"date": future_dates, "predicted_price": future_preds})
daily_df.to_csv(CSV_DAILY, index=False)
monthly_df = daily_df.resample("ME", on="date").mean(numeric_only=True).reset_index()
monthly_df.to_csv(CSV_MONTHLY, index=False)
print(f"→ Daily saved → {CSV_DAILY}")
print(f"→ Monthly saved → {CSV_MONTHLY}")

# ────────────────────────── PLOT ──────────────────────────
plt.figure(figsize=(14,7))
plt.plot(df["date"], prices, color="#1f77b4", lw=1.3, label="Historical")
plt.plot(future_dates, future_preds, color="#d62728", lw=1.9, ls="--", label=f"Forecast ({FUTURE_DAYS}d)")

plt.axvline(df["date"].max(), color="0.5", ls=":", lw=1.2, alpha=0.8, label="Last data")
plt.title("Bitcoin Forecast")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(loc="upper left"); plt.grid(ls="--", alpha=0.25); plt.xticks(rotation=40)
plt.tight_layout(); plt.show()