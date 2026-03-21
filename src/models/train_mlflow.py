import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# ── Fix working directory ──────────────────────────────────────────
while not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")

# ── Load & Prepare ─────────────────────────────────────────────────
df = pd.read_csv("data/german_credit_data.csv")
sex_col = df["Sex"].copy()
y = df["Risk"]
X = df.drop(columns=["Risk", "Sex", "Age_group"], errors="ignore")
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X_encoded, y, sex_col,
    test_size=0.2,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# ── MLflow: Logistic Regression Run ───────────────────────────────
mlflow.set_experiment("credit-fairness")

with mlflow.start_run(run_name="logistic_regression"):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    # Fairness metrics
    results = pd.DataFrame({
        "prediction": lr.predict(X_test),
        "actual": y_test.values,
        "Sex": sex_test.values
    })
    parity = results.groupby("Sex")["prediction"].apply(
        lambda x: (x == "good").mean()
    )
    parity_gap = abs(parity["male"] - parity["female"])

    acc_by_sex = {}
    for group in results["Sex"].unique():
        subset = results[results["Sex"] == group]
        acc_by_sex[group] = accuracy_score(subset["actual"], subset["prediction"])

    # Log to MLflow
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", lr_acc)
    mlflow.log_metric("parity_gap", parity_gap)
    mlflow.log_metric("accuracy_male",   acc_by_sex.get("male", 0))
    mlflow.log_metric("accuracy_female", acc_by_sex.get("female", 0))

    mlflow.sklearn.log_model(lr, "logistic_model")

    print(f" LR Run logged — Accuracy: {lr_acc:.4f} | Parity Gap: {parity_gap:.4f}")

# ── MLflow: Random Forest Run ──────────────────────────────────────
with mlflow.start_run(run_name="random_forest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    results_rf = pd.DataFrame({
        "prediction": rf.predict(X_test),
        "actual": y_test.values,
        "Sex": sex_test.values
    })
    parity_rf = results_rf.groupby("Sex")["prediction"].apply(
        lambda x: (x == "good").mean()
    )
    parity_gap_rf = abs(parity_rf["male"] - parity_rf["female"])

    acc_by_sex_rf = {}
    for group in results_rf["Sex"].unique():
        subset = results_rf[results_rf["Sex"] == group]
        acc_by_sex_rf[group] = accuracy_score(subset["actual"], subset["prediction"])

    mlflow.log_param("model", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", rf_acc)
    mlflow.log_metric("parity_gap", parity_gap_rf)
    mlflow.log_metric("accuracy_male",   acc_by_sex_rf.get("male", 0))
    mlflow.log_metric("accuracy_female", acc_by_sex_rf.get("female", 0))

    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(f" RF Run logged — Accuracy: {rf_acc:.4f} | Parity Gap: {parity_gap_rf:.4f}")

print("\n All runs logged to MLflow.")
print("Run `mlflow ui` in terminal to view dashboard.")
