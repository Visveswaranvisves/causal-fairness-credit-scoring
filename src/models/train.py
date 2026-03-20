import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Fix working directory ──────────────────────────────────────────
while not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")

print("Working directory:", os.getcwd())

# ── Load Data ──────────────────────────────────────────────────────
df = pd.read_csv("data/german_credit_data.csv")
print("Data loaded:", df.shape)

# ── Prepare Features ───────────────────────────────────────────────
sex_col = df["Sex"].copy()
y = df["Risk"]
X = df.drop(columns=["Risk", "Sex", "Age_group"], errors="ignore")
X_encoded = pd.get_dummies(X, drop_first=True)

# ── Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X_encoded, y, sex_col,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── Impute ─────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# ── Train Models ───────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────
lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("=" * 40)
print(f"Logistic Regression Accuracy : {lr_acc:.4f}")
print(f"Random Forest Accuracy       : {rf_acc:.4f}")
print("=" * 40)

# ── Fairness ───────────────────────────────────────────────────────
results = pd.DataFrame({
    "prediction": lr.predict(X_test),
    "actual":     y_test.values,
    "Sex":        sex_test.values
})

print("\nDemographic Parity by Sex:")
parity = results.groupby("Sex")["prediction"].apply(
    lambda x: (x == "good").mean()
)
print(parity)
print(f"Parity Gap: {abs(parity['male'] - parity['female']):.4f}")

print("\nAccuracy by Sex:")
for group in results["Sex"].unique():
    subset = results[results["Sex"] == group]
    acc = accuracy_score(subset["actual"], subset["prediction"])
    print(f"  {group}: {acc:.4f}")

# ── Save Models ────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(lr,      "models/logistic_model.pkl")
joblib.dump(rf,      "models/random_forest.pkl")
joblib.dump(imputer, "models/imputer.pkl")
print("\n✅ Models saved.")