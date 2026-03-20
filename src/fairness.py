import pandas as pd

def check_gender_bias(df):
    male = df[df['gender'] == 1]['target'].mean()
    female = df[df['gender'] == 0]['target'].mean()

    print("Male approval rate:", male)
    print("Female approval rate:", female)