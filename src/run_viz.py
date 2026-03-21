# src/run_viz.py
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fix working directory
while not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")
print("Working directory:", os.getcwd())

# Load data
df = pd.read_csv("data/german_credit_data.csv")
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Prepare features
sex_col = df["Sex"].copy()
y       = df["Risk"]
X       = df.drop(columns=["Risk", "Sex", "Age_group"], errors="ignore")
X_encoded = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X_encoded, y, sex_col,
    test_size=0.2, random_state=42, stratify=y
)

# Load models and preprocessors
imputer  = joblib.load("models/imputer.pkl")
scaler   = joblib.load("models/scaler.pkl")
baseline = joblib.load("models/logistic_model.pkl")

# Preprocess test data
X_test_proc = scaler.transform(imputer.transform(X_test))

# Predict
y_pred = baseline.predict(X_test_proc)

# ── Fairness Evaluation ────────────────────────────────────────────
results = pd.DataFrame({
    "prediction": y_pred,
    "actual":     y_test.values,
    "Sex":        sex_test.values
})

print("\n" + "="*50)
print("FAIRNESS EVALUATION")
print("="*50)

# Demographic Parity
parity = results.groupby("Sex")["prediction"].apply(
    lambda x: (x == "good").mean()
)
gap = abs(parity["male"] - parity["female"])
print("\nDemographic Parity:")
print(parity)
print(f"Parity Gap: {gap:.4f}")

# Accuracy by Sex
print("\nAccuracy by Sex:")
for group in results["Sex"].unique():
    subset = results[results["Sex"] == group]
    acc = accuracy_score(subset["actual"], subset["prediction"])
    print(f"  {group}: {acc:.4f}")

# ── Plot 1: Demographic Parity ─────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6, 4))
parity.plot(kind="bar", color=["salmon", "steelblue"], edgecolor="black")
plt.title("Demographic Parity by Sex")
plt.ylabel("Approval Rate")
plt.xlabel("Sex")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/demographic_parity.png", dpi=150)
plt.show()
print("Saved: outputs/demographic_parity.png")

# ── Plot 2: Accuracy by Sex ────────────────────────────────────────
acc_by_sex = {}
for group in results["Sex"].unique():
    subset = results[results["Sex"] == group]
    acc_by_sex[group] = accuracy_score(subset["actual"], subset["prediction"])

plt.figure(figsize=(6, 4))
pd.Series(acc_by_sex).plot(
    kind="bar", color=["salmon", "steelblue"], edgecolor="black"
)
plt.title("Accuracy by Sex")
plt.ylabel("Accuracy")
plt.xlabel("Sex")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/accuracy_by_sex.png", dpi=150)
plt.show()
print("Saved: outputs/accuracy_by_sex.png")

# ── Plot 3: Prediction Distribution ───────────────────────────────
plt.figure(figsize=(8, 4))
for i, group in enumerate(["male", "female"]):
    subset = results[results["Sex"] == group]
    counts = subset["prediction"].value_counts()
    plt.subplot(1, 2, i+1)
    counts.plot(kind="bar", color=["steelblue", "salmon"], edgecolor="black")
    plt.title(f"Predictions — {group.capitalize()}")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("outputs/prediction_distribution.png", dpi=150)
plt.show()
print("Saved: outputs/prediction_distribution.png")

print("\nAll charts saved to outputs/")

