# Bitcoin Time Series Forecasting

Bitcoin price forecasting using time-series models and machine learning.

This project explores different approaches for predicting Bitcoin prices using historical market data.  
Multiple models are implemented and compared, including a naive baseline, a regularized linear model, a deep learning model, and an ensemble approach.

The goal is to analyze how different modeling techniques perform on financial time-series data.

---

## Dataset

The Bitcoin historical data used in this project was sourced from **CoinGecko**:

https://www.coingecko.com/en/coins/bitcoin/historical_data

The dataset includes the following columns:

| Column | Description |
|------|------|
| `snapped_at` | Timestamp of the data point |
| `price` | Bitcoin price in USD |
| `market_cap` | Market capitalization |
| `total_volume` | Total trading volume |

> Note: The CSV file included in this repository is a snapshot of the data downloaded from CoinGecko.

---

## Project Objective

The objective of this project is to forecast the **next-day Bitcoin price** using historical price and market data.

Several modeling approaches were compared to understand which models perform best on financial time-series data.

---

## Models Compared

### 1. Naive Baseline (Persistence Model)

The naive persistence model predicts the next value using the previous value:

y_hat(t) = y(t-1)


This simple baseline is commonly used in time-series forecasting.

---

### 2. Ridge Regression

Ridge Regression is a linear model with **L2 regularization** that helps prevent overfitting and stabilize regression coefficients.

The Ridge regression solution can be written as:

beta = (X^T X + lambda I)^(-1) X^T y


The model uses engineered features such as:

- Lagged price values (previous prices)
- Rolling statistics (moving averages and volatility)
- Market indicators such as trading volume

---

### 3. Multivariate GRU (Deep Learning Model)

A **Gated Recurrent Unit (GRU)** neural network was implemented to capture sequential patterns in the time-series data.

The model uses:

- multivariate input features
- sliding time windows
- dropout regularization
- early stopping during training

---

### 4. Ensemble Model (GRU + Ridge)

An ensemble model was created by combining predictions from the GRU and Ridge models to test whether combining models improves forecasting performance.

---

## Evaluation Metrics

Two standard forecasting metrics were used.

### Root Mean Squared Error (RMSE)

RMSE measures the magnitude of prediction errors.
RMSE = sqrt((1/n) * Σ(yi - y_hat_i)^2)


---

### Mean Absolute Percentage Error (MAPE)

MAPE measures prediction error as a percentage.

MAPE = (100/n) * Σ |(yi - y_hat_i) / yi|


Lower values indicate better performance.

---

## Model Performance

| Model | RMSE | MAPE |
|------|------|------|
| Naive Baseline | 1903 | 1.76% |
| Multivariate GRU | 17718 | 12.58% |
| Ridge Regression | **811** | **0.93%** |
| Ensemble (GRU + Ridge) | 11285 | 8.00% |

The **Ridge Regression model achieved the best performance** across both evaluation metrics.

---

## Interpretation of Results

### Strong Performance of the Naive Baseline

The naive persistence model performs reasonably well because financial time series often exhibit **strong short-term autocorrelation**, meaning the price at time *t* is usually close to the price at time *t − 1*.

---

### Why Ridge Regression Performs Best

Ridge Regression performs well because it captures **linear relationships between lagged features and the target price** while using regularization to prevent overfitting.

The engineered features provide strong signals for short-term price prediction, allowing the model to outperform the naive baseline.

---

### Why the GRU Model Performs Worse

Although recurrent neural networks such as GRU are powerful for sequential data, they often require:

- very large datasets
- careful hyperparameter tuning
- longer training time

For structured financial datasets with strong linear relationships, simpler models such as regularized regression often perform better.

---

### Ensemble Model Performance

The ensemble model combining GRU and Ridge predictions did not outperform the Ridge model.

This happens because ensemble methods improve performance **only when the individual models contribute complementary predictive information**.

---

## Key Takeaways

This project highlights several important insights about machine learning for financial time-series forecasting:

- Baseline models are important benchmarks
- Feature engineering significantly improves model performance
- Regularized linear models can perform extremely well on tabular financial data
- Deep learning models are not always superior for structured datasets
- Comparing multiple models is essential when building forecasting systems

In this experiment, **Ridge Regression provided the most accurate forecasts for next-day Bitcoin prices**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

