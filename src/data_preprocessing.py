import pandas as pd

def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    return df

def preprocess(df):
    df = pd.get_dummies(df, drop_first=True)
    return df