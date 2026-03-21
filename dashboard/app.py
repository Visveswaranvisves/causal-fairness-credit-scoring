import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Credit Fairness Dashboard", layout="wide")
st.title("⚖️ Causal Fairness-Aware Credit Scoring")
st.markdown("Predict credit risk and see fairness analysis in real time.")

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/fair_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    df["target"] = (df["Credit amount"] > 2000).astype(int)
    return df

model = load_model()
df    = load_data()

# ── Sidebar Inputs ───────────────────────────────────────────
st.sidebar.header("Enter Applicant Details")

age            = st.sidebar.slider("Age", 18, 75, 30)
credit_amount  = st.sidebar.number_input("Credit Amount (€)", 100, 20000, 2500)
duration       = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)
sex            = st.sidebar.selectbox("Sex", ["male", "female"])
job            = st.sidebar.selectbox("Job Level", [0, 1, 2, 3])
housing        = st.sidebar.selectbox("Housing", ["own", "free", "rent"])
saving_account = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
checking       = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich"])
purpose        = st.sidebar.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "education", "business"])

# ── Build input row ──────────────────────────────────────────
input_dict = {
    "Age":              age,
    "Sex":              sex,
    "Job":              job,
    "Housing":          housing,
    "Saving accounts":  saving_account,
    "Checking account": checking,
    "Credit amount":    credit_amount,
    "Duration":         duration,
    "Purpose":          purpose,
}

input_df = pd.DataFrame([input_dict])

# ── Encode to match training ──────────────────────────────────
full_df    = pd.concat([df.drop("target", axis=1), input_df], ignore_index=True)
full_enc   = pd.get_dummies(full_df, drop_first=True)
input_enc  = full_enc.iloc[[-1]]

# Align columns with model
train_enc = pd.get_dummies(df.drop("target", axis=1), drop_first=True)
input_enc = input_enc.reindex(columns=train_enc.columns, fill_value=0)

# ── Predict ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button("Run Prediction"):
        prediction = model.predict(input_enc)[0]
        probability = model.predict_proba(input_enc)[0][1]

        if prediction == 1:
            st.success(f"✅ Credit Approved (High Amount Risk: {probability:.1%})")
        else:
            st.error(f"❌ Credit Denied (Risk Score: {probability:.1%})")

        st.metric("Risk Score", f"{probability:.2%}")

with col2:
    st.subheader("Dataset Fairness Summary")
    train_enc_full = train_enc.copy()
    train_enc_full["target"] = df["target"].values
    train_enc_full["Sex"]    = df["Sex"].values

    male_rate   = train_enc_full[train_enc_full["Sex"] == "male"]["target"].mean()
    female_rate = train_enc_full[train_enc_full["Sex"] == "female"]["target"].mean()
    gap         = abs(male_rate - female_rate)

    st.metric("Male Approval Rate",   f"{male_rate:.1%}")
    st.metric("Female Approval Rate", f"{female_rate:.1%}")
    st.metric("Parity Gap",           f"{gap:.4f}", delta=f"{gap:.4f}" if gap > 0.03 else "Low ✅")

# ── Dataset overview ─────────────────────────────────────────
st.subheader("Dataset Overview")
st.dataframe(df.head(10))