import pandas as pd
import numpy as np
import joblib
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_and_prepare():
    while not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.chdir("..")

    df = pd.read_csv("data/german_credit_data.csv")

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    return df


def train_fair_model(df):
    sex_col = df["Sex"].copy()

    y = df["Risk"]
    X = df.drop(columns=["Risk", "Sex", "Age_group"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X, y, sex_col, test_size=0.2, random_state=42, stratify=y
    )

    # Fill NaN
    X_train = X_train.fillna(X_train.median())
    X_test  = X_test.fillna(X_train.median())

    # Reweighting females
    weights = np.ones(len(X_train))
    weights[sex_train.values == "female"] = 1.5

    # Baseline model
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    # Fair model
    fair = LogisticRegression(max_iter=1000, random_state=42)
    fair.fit(X_train, y_train, sample_weight=weights)
    fair_acc = accuracy_score(y_test, fair.predict(X_test))

    print(f"Baseline Accuracy : {baseline_acc:.4f}")
    print(f"Fair Accuracy     : {fair_acc:.4f}")

    return baseline, fair, X_test, y_test, sex_test, baseline_acc, fair_acc


def log_to_mlflow(baseline_acc, fair_acc):
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
    baseline, fair, X_test, y_test, sex_test, baseline_acc, fair_acc = train_fair_model(df)
    log_to_mlflow(baseline_acc, fair_acc)
    save_models(baseline, fair)