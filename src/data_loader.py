import pandas as pd
from pathlib import Path

def load_data(filename="btc_price.csv"):
    """
    Load CoinGecko BTC-USD data from data/raw.
    """
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "raw" / filename

    df = pd.read_csv(file_path)

    # Rename columns to lowercase and remove spaces
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Convert 'snapped_at' to datetime and rename to 'date'
    df['date'] = pd.to_datetime(df['snapped_at'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Keep only relevant columns for forecasting
    df = df[['date', 'price', 'market_cap', 'total_volume']]

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.info())