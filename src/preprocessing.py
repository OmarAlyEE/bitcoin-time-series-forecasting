import pandas as pd
from pathlib import Path
from src.data_loader import load_data

def preprocess_data():

    # Load raw dataset
    df = load_data()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Keep only useful columns
    df = df[['date', 'price', 'market_cap', 'total_volume']]

    # Create daily returns
    df['returns'] = df['price'].pct_change()

    # Drop first NaN caused by pct_change
    df = df.dropna()

    # Normalize price (optional but useful for ML models)
    df['price_scaled'] = (df['price'] - df['price'].mean()) / df['price'].std()

    return df


def save_processed_data(df):

    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "processed" / "btc_processed.csv"

    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    df = preprocess_data()

    save_processed_data(df)

    print(df.head())
    print("Processed data saved.")
