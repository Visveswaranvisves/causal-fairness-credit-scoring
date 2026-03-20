import pandas as pd

def load_data():
    df = pd.read_csv("data/german_credit.csv")
    return df

def preprocess(df):
    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)
    return df