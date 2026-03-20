import pandas as pd
import os

def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    return df

def preprocess(df):
    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)
    return df

def save_clean_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("✅ Cleaned data saved at data/processed/cleaned_data.csv")


if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    save_clean_data(df)