import pandas as pd

def load_data():

    df = pd.read_csv("../data/processed/btc_features.csv")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    return df

def split_data(df):

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    return train, test

def create_xy(train, test):

    X_train = train.drop(columns=["date", "price"])
    y_train = train["price"]

    X_test = test.drop(columns=["date", "price"])
    y_test = test["price"]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()

    train, test = split_data(df)

    X_train, X_test, y_train, y_test = create_xy(train, test)

    print("Train size:", len(train))
    print("Test size:", len(test))

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

train.to_csv("../data/processed/train.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)
