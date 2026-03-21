import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(
    page_title="Causal Fairness Credit Scoring",
    page_icon="",
    layout="wide"
)

# Fix working directory
while not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")

# Header
st.title("Causal Fairness-Aware Credit Scoring")
st.markdown("An end-to-end Responsible AI system — prediction, fairness "
            "measurement, and causal bias mitigation.")
st.divider()

# Feature columns used during training
FEATURE_COLS = ['Age', 'Job', 'Housing', 'Saving accounts',
                'Checking account', 'Credit amount', 'Duration', 'Purpose']

# Load assets
@st.cache_resource
def load_assets():
    assets = {}

    for key, path in {
        "model":      "models/logistic_model.pkl",
        "imputer":    "models/imputer.pkl",
        "scaler":     "models/scaler.pkl",
        "causal1":    "models/causal_model_1.pkl",
        "imputer_c1": "models/imputer_c1.pkl",
        "scaler_c1":  "models/scaler_c1.pkl",
        "causal2":    "models/causal_model_2.pkl",
        "imputer_c2": "models/imputer_c2.pkl",
        "scaler_c2":  "models/scaler_c2.pkl",
    }.items():
        if os.path.exists(path):
            assets[key] = joblib.load(path)

    if os.path.exists("models/fair_thresholds.json"):
        with open("models/fair_thresholds.json") as f:
            assets["thresholds"] = json.load(f)

    return assets

@st.cache_data
def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    return df

assets = load_assets()
df     = load_data()


def prepare_input(input_dict, drop_cols=None):
    """Prepare input for prediction, aligned with training columns."""
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

    # Combine with reference data to ensure all dummy categories appear
    ref_df   = df[FEATURE_COLS].copy()
    combined = pd.concat([ref_df, input_df], ignore_index=True)
    combined_enc = pd.get_dummies(combined, drop_first=True)

    # Take only the last row (the actual input)
    input_enc = combined_enc.iloc[[-1]].copy()

    if drop_cols:
        input_enc = input_enc.drop(columns=drop_cols, errors="ignore")

    return input_enc


def predict(input_enc, model, imputer, scaler, sex=None, thresholds=None):
    """Run prediction and return label + probability."""
    X = scaler.transform(imputer.transform(input_enc))

    classes   = list(model.classes_)
    good_idx  = classes.index("good") if "good" in classes else 1
    prob_good = model.predict_proba(X)[0][good_idx]

    if thresholds and sex:
        threshold = thresholds.get(sex, 0.5)
    else:
        threshold = 0.5

    pred = "good" if prob_good >= threshold else "bad"
    return pred, prob_good


# Tabs
tab1, tab2, tab3 = st.tabs([
    "Live Prediction",
    "Fairness Analysis",
    "Model Comparison"
])

# ════════════════════════════════════════
# TAB 1 — Live Prediction
# ════════════════════════════════════════
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
        checking       = st.selectbox("Checking Account",
                                      ["little", "moderate", "rich"])
        purpose        = st.selectbox("Purpose",
                                      ["car", "furniture/equipment",
                                       "radio/TV", "education", "business"])
        model_choice   = st.selectbox("Model to use", [
            "Baseline",
            "Fair (Threshold)",
            "Causal 1 (No Sex)",
            "Causal 2 (No Proxies)"
        ])

    with col_result:
        if st.button("Run Prediction", type="primary"):
            input_dict = {
                "Age":              age,
                "Job":              job,
                "Housing":          housing,
                "Saving accounts":  saving_account,
                "Checking account": checking,
                "Credit amount":    credit_amount,
                "Duration":         duration,
                "Purpose":          purpose
            }

            try:
                model   = assets.get("model")
                imputer = assets.get("imputer")
                scaler  = assets.get("scaler")

                if model_choice == "Baseline":
                    input_enc = prepare_input(input_dict)
                    pred, prob = predict(input_enc, model, imputer, scaler)

                elif model_choice == "Fair (Threshold)":
                    input_enc  = prepare_input(input_dict)
                    thresholds = assets.get(
                        "thresholds", {"female": 0.45, "male": 0.50}
                    )
                    pred, prob = predict(
                        input_enc, model, imputer, scaler,
                        sex=sex, thresholds=thresholds
                    )

                elif model_choice == "Causal 1 (No Sex)":
                    input_enc = prepare_input(input_dict, drop_cols=["Sex_male"])
                    m = assets.get("causal1", model)
                    i = assets.get("imputer_c1", imputer)
                    s = assets.get("scaler_c1",  scaler)
                    pred, prob = predict(input_enc, m, i, s)

                elif model_choice == "Causal 2 (No Proxies)":
                    drop_cols = ["Sex_male", "Purpose_car", "Purpose_education"]
                    input_enc = prepare_input(input_dict, drop_cols=drop_cols)
                    m = assets.get("causal2", model)
                    i = assets.get("imputer_c2", imputer)
                    s = assets.get("scaler_c2",  scaler)
                    pred, prob = predict(input_enc, m, i, s)

                # Display results
                st.metric("Approval Probability", f"{prob:.2%}")

                if pred == "good":
                    st.success("APPROVED — Good credit risk")
                else:
                    st.error("DENIED — High credit risk")

                st.info(f"Model used: **{model_choice}**")

                # Show fairness note for fair model
                if model_choice == "Fair (Threshold)":
                    thresh = assets.get(
                        "thresholds", {"female": 0.45, "male": 0.50}
                    )
                    st.caption(
                        f"Fair threshold applied: "
                        f"female={thresh['female']}, male={thresh['male']}"
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

# ════════════════════════════════════════
# TAB 2 — Fairness Analysis
# ════════════════════════════════════════
with tab2:
    st.subheader("Fairness Metrics Across Gender Groups")

    male_rate   = (df[df["Sex"] == "male"]["Risk"] == "good").mean()
    female_rate = (df[df["Sex"] == "female"]["Risk"] == "good").mean()
    gap         = abs(male_rate - female_rate)

    m1, m2, m3 = st.columns(3)
    m1.metric("Male Approval Rate",   f"{male_rate:.1%}")
    m2.metric("Female Approval Rate", f"{female_rate:.1%}")
    m3.metric(
        "Parity Gap", f"{gap:.4f}",
        delta="High" if gap > 0.05 else "Low"
    )

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        for path in ["outputs/parity_comparison.png",
                     "outputs/demographic_parity.png"]:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                break

    with chart_col2:
        for path in ["outputs/accuracy_fairness_tradeoff.png",
                     "outputs/fairness_by_sex.png"]:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                break

    if os.path.exists("outputs/shap_gender_comparison.png"):
        st.subheader("SHAP Feature Importance by Gender")
        st.image("outputs/shap_gender_comparison.png",
                 use_container_width=True)
    elif os.path.exists("outputs/shap_importance.png"):
        st.subheader("SHAP Feature Importance")
        st.image("outputs/shap_importance.png", use_container_width=True)

# ════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════
with tab3:
    st.subheader("All Models — Side by Side")

    csv_path = "outputs/model_comparison.csv"
    if os.path.exists(csv_path):
        comp = pd.read_csv(csv_path)
        st.dataframe(
            comp.style.highlight_min(
                subset=["Parity Gap", "EO Gap"], color="lightgreen"
            ).highlight_max(
                subset=["Accuracy"], color="lightblue"
            ),
            use_container_width=True
        )

        best_fair = comp.loc[comp["Parity Gap"].idxmin(), "Model"]
        best_acc  = comp.loc[comp["Accuracy"].idxmax(),   "Model"]
        st.success(f"Fairest model: **{best_fair}**")
        st.info(   f"Most accurate: **{best_acc}**")
    else:
        st.warning(
            "Run day6_causal_fairness_final.ipynb first "
            "to generate comparison data."
        )

    if os.path.exists("outputs/model_comparison.png"):
        st.image("outputs/model_comparison.png", use_container_width=True)

    # Summary insight
    st.divider()
    st.subheader("Key Findings")
    st.markdown("""
    - **Baseline** — 69% accuracy, demographic parity gap of 0.038
    - **Fair (Threshold)** — 25% bias reduction with only 1% accuracy drop
    - **Causal 1 (No Sex)** — Removing Sex column had no effect;
      bias comes from correlated proxy features
    - **Causal 2 (No Proxies)** — Removing proxies increased bias,
      showing those features were partially corrective

    > *Simply removing sensitive attributes does not guarantee fairness.
    Threshold adjustment proved most effective for this dataset.*
    """)
