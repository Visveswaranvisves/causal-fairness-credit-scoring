import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Causal Fairness Credit Scoring",
    page_icon="",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title(" Causal Fairness-Aware Credit Scoring")
st.markdown("An end-to-end Responsible AI system — prediction, fairness "
            "measurement, and causal bias mitigation.")
st.divider()

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    paths  = {
        "Fair (Reweighted)":     "models/fair_model.pkl",
        "Causal 1 (No Sex)":     "models/causal_model_1.pkl",
        "Causal 2 (No Proxies)": "models/causal_model_2.pkl",
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    df["target"] = (df["Credit amount"] > 2000).astype(int)
    return df

models = load_models()
df     = load_data()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Live Prediction", " Fairness Analysis", " Model Comparison"])

# ════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Credit Risk")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        age            = st.slider("Age", 18, 75, 30)
        credit_amount  = st.number_input("Credit Amount (€)", 100, 20000, 2500)
        duration       = st.slider("Loan Duration (months)", 6, 72, 24)
        sex            = st.selectbox("Sex", ["male", "female"])
        job            = st.selectbox("Job Level", [0, 1, 2, 3])
        housing        = st.selectbox("Housing", ["own", "free", "rent"])
        saving_account = st.selectbox("Saving Accounts",
                                      ["little", "moderate", "quite rich", "rich"])
        checking       = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        purpose        = st.selectbox("Purpose",
                                      ["car", "furniture/equipment",
                                       "radio/TV", "education", "business"])
        model_choice   = st.selectbox("Model to use", list(models.keys()))

    with col_result:
        if st.button("Run Prediction", type="primary"):
            input_dict = {
                "Age": age, "Sex": sex, "Job": job,
                "Housing": housing, "Saving accounts": saving_account,
                "Checking account": checking, "Credit amount": credit_amount,
                "Duration": duration, "Purpose": purpose
            }
            input_df  = pd.DataFrame([input_dict])
            full_data = pd.concat([df.drop("target", axis=1), input_df],
                                  ignore_index=True)
            full_enc  = pd.get_dummies(full_data, drop_first=True)
            input_enc = full_enc.iloc[[-1]]

            train_enc = pd.get_dummies(df.drop("target", axis=1), drop_first=True)
            input_enc = input_enc.reindex(columns=train_enc.columns, fill_value=0)

            selected = models[model_choice]

            # Drop proxy cols for causal models
            if "Causal 1" in model_choice:
                drop_cols = [c for c in input_enc.columns if "Sex" in c]
                input_enc = input_enc.drop(columns=drop_cols, errors="ignore")
            elif "Causal 2" in model_choice:
                proxy_kw  = ["Sex", "Job", "Housing", "Saving"]
                drop_cols = [c for c in input_enc.columns
                             if any(kw.lower() in c.lower() for kw in proxy_kw)]
                input_enc = input_enc.drop(columns=drop_cols, errors="ignore")

            try:
                pred  = selected.predict(input_enc)[0]
                prob  = selected.predict_proba(input_enc)[0][1]
                st.metric("Risk Score", f"{prob:.2%}")
                if pred == 1:
                    st.success(" High Credit Amount Profile — Likely Approved")
                else:
                    st.error(" Low Credit Amount Profile — May Be Denied")

                st.info(f"Model used: **{model_choice}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ════════════════════════════════════════════════════════
# TAB 2 — Fairness Analysis
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Fairness Metrics Across Gender Groups")

    male_rate   = df[df["Sex"] == "male"]["target"].mean()
    female_rate = df[df["Sex"] == "female"]["target"].mean()
    gap         = abs(male_rate - female_rate)

    m1, m2, m3 = st.columns(3)
    m1.metric("Male Approval Rate",   f"{male_rate:.1%}")
    m2.metric("Female Approval Rate", f"{female_rate:.1%}")
    m3.metric("Parity Gap", f"{gap:.4f}",
              delta="High " if gap > 0.05 else "Low ")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if os.path.exists("outputs/demographic_parity.png"):
            st.image("outputs/demographic_parity.png", use_column_width=True)
    with chart_col2:
        if os.path.exists("outputs/equal_opportunity.png"):
            st.image("outputs/equal_opportunity.png", use_column_width=True)

    if os.path.exists("outputs/bias_summary.png"):
        st.image("outputs/bias_summary.png", use_column_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("All Models — Side by Side")

    if os.path.exists("outputs/final_comparison.csv"):
        comp = pd.read_csv("outputs/final_comparison.csv")
        st.dataframe(comp.style.highlight_min(
            subset=["Parity Gap", "EO Gap"], color="lightgreen"
        ).highlight_max(
            subset=["Accuracy"], color="lightblue"
        ), use_container_width=True)

        best_fair = comp.loc[comp["Parity Gap"].idxmin(), "Model"]
        best_acc  = comp.loc[comp["Accuracy"].idxmax(),   "Model"]
        st.success(f" Fairest model: **{best_fair}**")
        st.info(   f" Most accurate: **{best_acc}**")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if os.path.exists("outputs/final_comparison.png"):
            st.image("outputs/final_comparison.png", use_column_width=True)
    with img_col2:
        if os.path.exists("outputs/tradeoff_scatter.png"):
            st.image("outputs/tradeoff_scatter.png", use_column_width=True)
            st.caption("Green arrow shows ideal direction — lower gap, "
                       "higher accuracy.")

    if os.path.exists("outputs/bias_reduction.png"):
        st.image("outputs/bias_reduction.png", use_column_width=True)