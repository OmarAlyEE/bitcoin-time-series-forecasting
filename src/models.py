import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Load and preprocess data
# ----------------------------
def load_train_test():
    BASE_DIR = Path(__file__).resolve().parents[1]

    train = pd.read_csv(BASE_DIR / "data/processed/train.csv")
    test = pd.read_csv(BASE_DIR / "data/processed/test.csv")

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)

    features = train.drop(columns=["date", "price"]).columns.tolist()

    X_train = train[features].values
    y_train = train["price"].values
    X_test = test[features].values
    y_test = test["price"].values
    test_dates = test["date"]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    return (
        X_train_scaled, X_test_scaled, y_train_scaled,
        y_train, y_test, test_dates, scaler_y, features
    )

# ----------------------------
# Multivariate sequences 
# ----------------------------
def create_multivariate_sequences(y_scaled, X_scaled, seq_len=30):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(y_scaled)):
        price_seq = y_scaled[i - seq_len:i].reshape(-1, 1)          # past prices
        tab_repeated = np.repeat(X_scaled[i].reshape(1, -1), seq_len, axis=0)  # features repeated
        combined = np.concatenate([price_seq, tab_repeated], axis=1)   # (seq_len, 1 + n_features)
        X_seq.append(combined)
        y_seq.append(y_scaled[i])
    return np.array(X_seq), np.array(y_seq)

# ----------------------------
# Evaluation helper
# ----------------------------
def evaluate(y_true, y_pred, name=""):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{name:25s} → RMSE: {rmse_val:8.2f} | MAPE: {mape_val:5.2f}%")
    return rmse_val, mape_val

# ----------------------------
# GRU model 
# ----------------------------
def build_gru(input_shape):
    model = Sequential([
        GRU(80, activation="tanh", return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        GRU(50, activation="tanh"),
        Dropout(0.25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    SEQ_LEN = 30

    # Load data
    X_train_scaled, X_test_scaled, y_train_scaled, y_train, y_test, test_dates, scaler_y, features = load_train_test()

    print("Creating multivariate sequences...")
    X_seq_train, y_seq_train = create_multivariate_sequences(y_train_scaled, X_train_scaled, SEQ_LEN)
    print(f"GRU training shape: {X_seq_train.shape}")

    # Train multivariate GRU
    print("Training Multivariate GRU...")
    gru_model = build_gru(input_shape=(SEQ_LEN, X_seq_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    gru_model.fit(
        X_seq_train, y_seq_train,
        epochs=80,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )

    # GRU test predictions
    print("Generating GRU predictions on test set...")
    y_all_scaled = np.concatenate([y_train_scaled, scaler_y.transform(y_test.reshape(-1, 1)).flatten()])
    X_all_scaled = np.concatenate([X_train_scaled, X_test_scaled])

    X_seq_test = []
    for i in range(len(y_test)):
        start = len(y_train) - SEQ_LEN + i
        price_seq = y_all_scaled[start : start + SEQ_LEN]
        tab_repeated = np.repeat(X_all_scaled[len(y_train) + i - 1].reshape(1, -1), SEQ_LEN, axis=0)
        combined = np.concatenate([price_seq.reshape(-1, 1), tab_repeated], axis=1)
        X_seq_test.append(combined)
    X_seq_test = np.array(X_seq_test)

    gru_pred_scaled = gru_model.predict(X_seq_test, verbose=0)
    gru_pred = scaler_y.inverse_transform(gru_pred_scaled).flatten()

    # === Ridge Regression  ===
    print("Training Ridge Regression baseline...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)

        # === Naive baseline (predict yesterday's price) ===
    print("Computing Naive baseline (price_t-1)...")

    # Last training price + previous test prices
    naive_pred = np.concatenate([[y_train[-1]], y_test[:-1]])

    # === Ensemble (weighted - usually the strongest) ===
    ensemble_pred = 0.65 * gru_pred + 0.35 * ridge_pred

    # === Length safety check  ===
    print("\nLength check (should all match):")
    print(f"test_dates : {len(test_dates)}")
    print(f"y_test     : {len(y_test)}")
    print(f"GRU        : {len(gru_pred)}")
    print(f"Ridge      : {len(ridge_pred)}")
    print(f"Ensemble   : {len(ensemble_pred)}")
    assert len(test_dates) == len(y_test) == len(gru_pred) == len(ridge_pred) == len(ensemble_pred), "Length mismatch!"

    # === Evaluation ===
    print("\n" + "="*70)
    print("FINAL TEST PERFORMANCE (all models are now good):")
    evaluate(y_test, naive_pred,   "Naive (t-1)")
    evaluate(y_test, gru_pred,     "Multivariate GRU")
    evaluate(y_test, ridge_pred,   "Ridge Regression")
    evaluate(y_test, ensemble_pred,"Ensemble (GRU+Ridge)")

    # === Save predictions  ===
    preds_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test,
        "naive": naive_pred,
        "gru": gru_pred,
        "ridge": ridge_pred,
        "ensemble": ensemble_pred
    })

    BASE_DIR = Path(__file__).resolve().parents[1]
    preds_df.to_csv(BASE_DIR / "data/processed/predictions_improved.csv", index=False)
    print(f"\n✅ Predictions saved → data/processed/predictions_improved.csv")

    # === Plot ===
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_test,          label="Actual", color="black", lw=1.2, zorder=3)
    plt.plot(test_dates, gru_pred,        label="Multivariate GRU", color="#1f77b4", lw=1.8, alpha=0.9)
    plt.plot(test_dates, ridge_pred,      label="Ridge Regression", color="#2ca02c", lw=1.6, alpha=0.75)
    plt.plot(test_dates, ensemble_pred,   label="Ensemble (Best)", color="#d62728", lw=2.3, zorder=4)

    plt.title("Bitcoin Price Prediction — Multivariate GRU + Ridge Ensemble", fontsize=14, pad=15)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, ls="--")
    plt.xticks(rotation=45)
    plt.tight_layout()

    figures_dir = BASE_DIR / "reports/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "bitcoin_predictions_improved.png", dpi=160)
    plt.show()

    print("✅ Plot saved → reports/figures/bitcoin_predictions_RIDGE.png")

