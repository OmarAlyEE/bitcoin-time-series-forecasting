import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


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
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_train,
        y_test,
        test_dates,
        scaler_y,
        features,
    )


# ----------------------------
# Create GRU sequences
# ----------------------------
def create_sequences(y, X_features, seq_len=30):

    X_seq = []
    X_tab = []
    y_seq = []

    for i in range(seq_len, len(y)):

        X_seq.append(y[i - seq_len : i])
        X_tab.append(X_features[i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(X_tab), np.array(y_seq)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return rmse, mape


# ----------------------------
# GRU model
# ----------------------------
def train_gru(X_seq, y_seq):

    model = Sequential()

    model.add(GRU(64, activation="tanh", input_shape=(X_seq.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_seq,
        y_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    return model


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    SEQ_LEN = 30

    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_train,
        y_test,
        test_dates,
        scaler_y,
        features,
    ) = load_train_test()

    # ----------------------------
    # GRU training sequences
    # ----------------------------

    X_seq_train, X_tab_train, y_seq_train = create_sequences(
        y_train_scaled, X_train_scaled, seq_len=SEQ_LEN
    )

    X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], X_seq_train.shape[1], 1))

    print(f"GRU training data shape: {X_seq_train.shape}")

    print("Training GRU...")

    gru_model = train_gru(X_seq_train, y_seq_train)

    # ----------------------------
    # Train predictions
    # ----------------------------

    y_gru_train_pred = gru_model.predict(X_seq_train)

    y_gru_train_pred_inv = scaler_y.inverse_transform(y_gru_train_pred)

    residuals = y_train[SEQ_LEN:] - y_gru_train_pred_inv.flatten()

    # ----------------------------
    # Train XGBoost on residuals
    # ----------------------------

    print("Training XGBoost on GRU residuals...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(X_tab_train, residuals)

    # ----------------------------
    # Prepare test sequences
    # ----------------------------

    y_all = np.concatenate([y_train, y_test])

    y_all_scaled = scaler_y.transform(y_all.reshape(-1, 1))

    X_all = np.concatenate([X_train_scaled, X_test_scaled])

    X_seq_all, X_tab_all, _ = create_sequences(
        y_all_scaled,
        X_all,
        seq_len=SEQ_LEN,
    )

    # Select only test portion

    test_start = len(y_train) - SEQ_LEN

    X_seq_test = X_seq_all[test_start:]
    X_tab_test = X_tab_all[test_start:]

    X_seq_test = X_seq_test.reshape((X_seq_test.shape[0], SEQ_LEN, 1))

    # ----------------------------
    # GRU predictions
    # ----------------------------

    y_gru_test_pred = gru_model.predict(X_seq_test)

    y_gru_test_pred_inv = scaler_y.inverse_transform(y_gru_test_pred)

    # ----------------------------
    # XGBoost residual prediction
    # ----------------------------

    y_xgb_resid_pred = xgb_model.predict(X_tab_test)

    # ----------------------------
    # Final hybrid prediction
    # ----------------------------

    y_hybrid_pred = y_gru_test_pred_inv.flatten() + y_xgb_resid_pred

    # Align ground truth

    y_true_aligned = y_test[: len(y_hybrid_pred)]

    # ----------------------------
    # Evaluation
    # ----------------------------

    rmse, mape = evaluate(y_true_aligned, y_hybrid_pred)

    print(f"Hybrid GRU + XGBoost → RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # ----------------------------
    # Save predictions
    # ----------------------------

    pred_len = len(y_hybrid_pred)

    preds_df = pd.DataFrame(
        {
            "date": test_dates[:pred_len],
            "y_true": y_true_aligned,
            "gru": y_gru_test_pred_inv.flatten()[:pred_len],
            "xgb_residual": y_xgb_resid_pred[:pred_len],
            "hybrid": y_hybrid_pred[:pred_len],
        }
    )

    BASE_DIR = Path(__file__).resolve().parents[1]

    preds_df.to_csv(
        BASE_DIR / "data/processed/predictions_gru_xgb.csv",
        index=False,
    )

    # ----------------------------
    # Plot predictions
    # ----------------------------

    plt.figure(figsize=(14, 7))

    plt.plot(preds_df["date"], preds_df["y_true"], label="Actual", color="black")

    plt.plot(preds_df["date"], preds_df["gru"], label="GRU", alpha=0.8)

    plt.plot(preds_df["date"], preds_df["hybrid"], label="Hybrid GRU + XGBoost", alpha=0.8)

    plt.xlabel("Date")
    plt.ylabel("Bitcoin Price (USD)")
    plt.title("Bitcoin Price Prediction — GRU + XGBoost Hybrid")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xticks(rotation=60)

    plt.tight_layout()

    figures_dir = BASE_DIR / "reports/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(figures_dir / "gru_xgb_predictions.png", dpi=140)

    plt.show()
