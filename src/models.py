import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from pathlib import Path

def load_train_test():
    BASE_DIR = Path(__file__).resolve().parents[1]
    train = pd.read_csv(BASE_DIR / "data/processed/train.csv")
    test = pd.read_csv(BASE_DIR / "data/processed/test.csv")
    
    X_train = train.drop(columns=["date", "price"])
    y_train = train["price"]
    
    X_test = test.drop(columns=["date", "price"])
    y_test = test["price"]
    
    return X_train, X_test, y_train, y_test

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # classic sqrt(MSE)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mape

def train_arima(y_train, y_test, order=(5,1,0)):
    # Fit ARIMA on training price series
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    
    # Forecast length of test set
    predictions = model_fit.forecast(steps=len(y_test))
    
    rmse, mape = evaluate(y_test, predictions)
    print(f"ARIMA RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return predictions


def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    
    rmse, mape = evaluate(y_test, predictions)
    print(f"Random Forest RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return predictions

def train_xgboost(X_train, X_test, y_train, y_test):
    xgbr = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgbr.fit(X_train, y_train)
    
    predictions = xgbr.predict(X_test)
    
    rmse, mape = evaluate(y_test, predictions)
    print(f"XGBoost RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return predictions


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test()
    
    # ARIMA
    print("Training ARIMA...")
    arima_preds = train_arima(y_train, y_test)
    
    # Random Forest
    print("Training Random Forest...")
    rf_preds = train_random_forest(X_train, X_test, y_train, y_test)
    
    # XGBoost
    print("Training XGBoost...")
    xgb_preds = train_xgboost(X_train, X_test, y_train, y_test)


# Load test data for dates
test_df = pd.read_csv("../data/processed/test.csv")

dates = test_df["date"].iloc[-len(y_test):].reset_index(drop=True)

# Create predictions DataFrame
preds_df = pd.DataFrame({
    "date": dates,
    "y_true": y_test.reset_index(drop=True),
    "arima": pd.Series(arima_preds),
    "rf": pd.Series(rf_preds),
    "xgb": pd.Series(xgb_preds)
})

# Save predictions
preds_df.to_csv("../data/processed/predictions.csv", index=False)

