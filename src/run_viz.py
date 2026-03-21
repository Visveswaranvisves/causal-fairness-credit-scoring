# src/run_viz.py
import sys
import os
import pandas as pd
import joblib

# Fix paths
sys.path.append(os.path.dirname(_file_))
while not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")

from sklearn.model_selection import train_test_split
from fairness import check_gender_bias, plot_bias

# Load data
df = pd.read_csv("data/german_credit_data.csv")
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

sex_col = df["Sex"].copy()
y = df["Risk"]
X = df.drop(columns=["Risk", "Sex", "Age_group"], errors="ignore")
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex_col, test_size=0.2, random_state=42, stratify=y
)

baseline = joblib.load("models/logistic_model.pkl")
y_pred = baseline.predict(X_test)

male, female = check_gender_bias(
    pd.DataFrame({"Sex": sex_test.values, "target": (y_pred == "good").astype(int)})
)
plot_bias(male, female)

print("Done.")