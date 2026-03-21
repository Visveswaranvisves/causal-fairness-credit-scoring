import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_prepare():
    df = pd.read_csv("data/german_credit_data.csv")

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    df["target"] = (df["Credit amount"] > 2000).astype(int)
    return df

def train_fair_model(df):
    sex_col = df["Sex"].copy()

    y = df["target"]
    X = df.drop("target", axis=1)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sex_train = sex_col.loc[X_train.index]
    sex_test  = sex_col.loc[X_test.index]

    # Reweighting
    weights = np.ones(len(X_train))
    weights[sex_train == "female"] = 1.5

    # Baseline
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    # Fair model
    fair = LogisticRegression(max_iter=1000)
    fair.fit(X_train, y_train, sample_weight=weights)
    fair_acc = accuracy_score(y_test, fair.predict(X_test))

    print(f"Baseline Accuracy : {baseline_acc:.4f}")
    print(f"Fair Accuracy     : {fair_acc:.4f}")

    return baseline, fair, X_test, y_test, sex_test

import mlflow

mlflow.set_experiment("credit-fairness")

with mlflow.start_run(run_name="baseline"):
    mlflow.log_metric("accuracy", baseline_acc)
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("debiased", False)

with mlflow.start_run(run_name="fair_model"):
    mlflow.log_metric("accuracy", fair_acc)
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("debiased", True)
    mlflow.log_param("female_weight", 1.5)

print("Experiments logged to MLflow!")

def save_models(baseline, fair):
    os.makedirs("models", exist_ok=True)
    joblib.dump(baseline, "models/logistic_model.pkl")
    joblib.dump(fair, "models/fair_model.pkl")
    print("Models saved!")

if __name__ == "__main__":
    df = load_and_prepare()
    baseline, fair, X_test, y_test, sex_test = train_fair_model(df)
    save_models(baseline, fair)