import pandas as pd
from pathlib import Path


def load_processed_data():

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "btc_processed.csv"

    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def create_lag_features(df):

    df["lag_1"] = df["price"].shift(1)
    df["lag_3"] = df["price"].shift(3)
    df["lag_7"] = df["price"].shift(7)
    df["lag_14"] = df["price"].shift(14)

    return df

def create_rolling_features(df):

    df["rolling_mean_7"] = df["price"].rolling(window=7).mean()
    df["rolling_mean_30"] = df["price"].rolling(window=30).mean()

    df["rolling_std_7"] = df["price"].rolling(window=7).std()
    df["rolling_std_30"] = df["price"].rolling(window=30).std()

    return df

def create_time_features(df):

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week

    return df

def clean_features(df):

    df = df.dropna()

    return df

def save_features(df):

    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "processed" / "btc_features.csv"

    df.to_csv(output_path, index=False)



if __name__ == "__main__":

    df = load_processed_data()

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_time_features(df)

    df = clean_features(df)

    save_features(df)

    print(df.head())
