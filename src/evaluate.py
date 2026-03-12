import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def evaluate_models(df, true_col="actual"):
    """
    Automatically evaluate all numeric prediction columns against the true column.
    """
    y_true = df[true_col].values
    results = []

    # Find all prediction columns (exclude date and actual/true column)
    pred_cols = [
        col for col in df.columns
        if col not in [true_col, "date"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in pred_cols:
        y_pred = df[col].values
        results.append({
            "Model": col.replace("_", " ").title(),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # ────────────────────────────────────────────────
    #  Change this line to match your latest output file
    # ────────────────────────────────────────────────
    PREDICTIONS_FILE = "../data/processed/predictions_improved.csv"

    try:
        preds_df = pd.read_csv(PREDICTIONS_FILE)
        print(f"Loaded predictions from: {PREDICTIONS_FILE}")
        print("Columns found:", list(preds_df.columns))
    except FileNotFoundError:
        print(f"Error: File not found → {PREDICTIONS_FILE}")
        print("Make sure you ran the prediction script first.")
        exit(1)

    # Evaluate all prediction columns automatically
    results_df = evaluate_models(preds_df, true_col="actual")

    # ─── Pretty print ───
    print("\n" + "=" * 65)
    print(" MODEL EVALUATION ".center(65, "="))
    print(results_df.round(2).to_string(index=False))
    print("=" * 65)

    # Highlight best models
    if not results_df.empty:
        best_rmse_row = results_df.loc[results_df["RMSE"].idxmin()]
        best_mape_row = results_df.loc[results_df["MAPE"].idxmin()]

        print(f"Best RMSE  → {best_rmse_row['Model']:<18} ({best_rmse_row['RMSE']:.2f})")
        print(f"Best MAPE  → {best_mape_row['Model']:<18} ({best_mape_row['MAPE']:.2f}%)")
        print()

    # Save results
    results_df.to_csv("../data/processed/model_evaluation.csv", index=False)
    print("Evaluation results saved to: ../data/processed/model_evaluation.csv")
