import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, preprocess
# Import functions
from src.data_preprocessing import load_data, preprocess

# 🔹 Step 1: Load data
df = load_data()

print("Columns BEFORE processing:", df.columns)

# 🔹 Step 2: Drop unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# 🔹 Step 3: Create TARGET (since dataset has none)
# You can adjust logic later
df["target"] = (df["Credit amount"] > 2000).astype(int)

# 🔹 Step 4: Split features and target
y = df["target"]
X = df.drop("target", axis=1)

# 🔹 Step 5: Preprocess features only
X = preprocess(X)

# 🔹 Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 7: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔹 Step 8: Predictions
y_pred = model.predict(X_test)

# 🔹 Step 9: Evaluation
accuracy = accuracy_score(y_test, y_pred)

print(" Model trained successfully!")
print(f"Accuracy: {accuracy:.4f}")