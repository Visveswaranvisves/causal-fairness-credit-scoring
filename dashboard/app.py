import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(
    page_title="Causal Fairness Credit Scoring",
    page_icon="◈",
    layout="wide"
)

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.markdown('\n<style>\n.stApp{background:#f4f7fb;color:#16243a}\n[data-testid="stHeader"]{background:rgba(244,247,251,.96)}\n.block-container{padding-top:2.4rem;padding-bottom:2rem;max-width:1440px}\nh1,h2,h3{letter-spacing:-.035em;color:#16243a}\nh3{font-size:1.35rem!important}\n[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2e8f0}\n[data-testid="stSidebar"] .block-container{padding-top:2rem}\n[data-testid="stMetric"]{background:#fff;border:1px solid #e0e8f1;border-radius:16px;padding:19px 20px;box-shadow:0 4px 16px #172b4d05;min-height:118px}\n[data-testid="stMetricLabel"]{color:#5b6b82;font-size:.8rem}\n[data-testid="stMetricValue"]{font-size:clamp(1.35rem,2vw,2rem);font-weight:650;letter-spacing:-.04em}\n[data-testid="stCaptionContainer"]{color:#607087}\n[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:16px}\n[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]{background:white;border:1px solid #e0e8f1;border-radius:16px;overflow:hidden}\n[data-baseweb="tab-list"]{gap:8px;background:#e9eef5;border-radius:12px;padding:5px;margin-bottom:22px}\nbutton[data-baseweb="tab"]{padding:10px 20px;border-radius:9px;height:44px}\nbutton[data-baseweb="tab"][aria-selected="true"]{background:white;color:#167e8b;box-shadow:0 2px 6px #172b4d10}\n[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none}\n.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#167e8b;border:0;border-radius:10px;min-height:46px;font-weight:650}\n.stButton>button,.stDownloadButton>button{border-radius:10px}\n[data-testid="stFileUploaderDropzone"]{border:1px dashed #91abc6;border-radius:16px;background:#edf4fb;padding:28px}\n[data-testid="stAlert"]{border-radius:12px}\n[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}\n[data-testid="stSidebar"] [role="radiogroup"] label{padding:10px 12px;border-radius:10px}\n[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#e8f3f2;font-weight:600}\n.brand{display:flex;align-items:center;gap:10px;font-weight:750;font-size:21px;letter-spacing:-.7px;margin-bottom:22px}\n.brand-mark{background:#167e8b;color:white;display:inline-flex;align-items:center;justify-content:center;width:35px;height:35px;border-radius:10px;font-size:17px}\n.hero{background:linear-gradient(110deg,#112339,#203e57);color:white;border-radius:22px;padding:32px 36px;margin:8px 0 28px;position:relative;overflow:hidden;border:1px solid #314d66}\n.hero h1{color:white!important;font-size:clamp(2rem,3.5vw,3rem);font-weight:720;margin:8px 0 12px;padding:0;max-width:850px}\n.hero p{color:#c6d6e7;max-width:750px;line-height:1.6;margin:0}\n.eyebrow{color:#8de0cf;font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase}\n.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}\n.chip{font-size:11px;padding:6px 11px;border:1px solid #536b80;border-radius:6px;color:#e2edf7}\n.empty{border:1px dashed #c6d3e0;background:#f9fbfd;border-radius:16px;padding:48px 26px;text-align:center;color:#66788f;min-height:235px}\n.empty strong{display:block;color:#243c55;font-size:19px;margin:14px 0 9px}\n.section-label{font-size:11px;font-weight:750;letter-spacing:1.5px;color:#167e8b;text-transform:uppercase;margin:4px 0 10px}\n.footer{border-top:1px solid #dce5ef;padding-top:18px;margin-top:36px;font-size:11px;color:#6a7b90;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}\n@media(max-width:768px){.block-container{padding:1.5rem 1rem}.hero{padding:25px 22px}.hero h1{font-size:2rem} [data-testid="stMetric"]{padding:15px;min-height:100px}[data-testid="stMetricValue"]{font-size:1.6rem}button[data-baseweb="tab"]{padding:8px 12px;font-size:12px}}\n</style>\n', unsafe_allow_html=True)
st.markdown('<div class="brand"><span class="brand-mark">C</span>Credit Scoring / Fairness Lab</div><div class="hero"><div class="eyebrow">Responsible AI workspace</div><h1>Understand risk. Examine fairness.</h1><p>Explore credit-risk estimates and compare how modelling choices affect performance and demographic parity.</p><div class="chips"><span class="chip">4 MODEL OPTIONS</span><span class="chip">GERMAN CREDIT DATA</span><span class="chip">RESEARCH DEMO</span></div></div>', unsafe_allow_html=True)
st.caption('Portfolio research only. Outputs are model estimates, not lending decisions.')

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
    st.subheader("Credit risk workspace")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown('<div class="section-label">01 / Applicant profile</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 18, 75, 30)
            sex = st.selectbox("Sex", ["male", "female"])
            housing = st.selectbox("Housing", ["own", "free", "rent"])
            saving_account = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        with right:
            duration = st.slider("Loan Duration (months)", 6, 72, 24)
            job = st.selectbox("Job Level", [0, 1, 2, 3])
            credit_amount = st.number_input("Credit Amount (€)", 100, 20000, 2500)
            checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "education", "business"])
        st.markdown('<div class="section-label">02 / Model selection</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Model to use", ["Baseline", "Fair (Threshold)", "Causal 1 (No Sex)", "Causal 2 (No Proxies)"])

    with col_result:
        st.markdown('<div class="section-label">03 / Model output</div>', unsafe_allow_html=True)
        st.caption("Adjust the applicant profile, choose a model, then run the estimate.")
        if st.button("Run Prediction", type="primary", use_container_width=True):
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
                st.metric("Good-credit model score", f"{prob:.2%}")
                st.progress(float(prob))
                st.caption("Model score; not a calibrated approval probability.")

                if pred == "good":
                    st.success("Model classification: good credit risk")
                else:
                    st.error("Model classification: high credit risk")

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
        else:
            st.markdown('<div class="empty"><span style="font-size:36px">◈</span><strong>Your estimate appears here</strong>Compare the same profile across models to explore how the output changes.</div>', unsafe_allow_html=True)


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
                     "models/fairness_by_sex.png"]:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                break

    if os.path.exists("notebooks/outputs/shap_gender_comparison.png"):
        st.subheader("SHAP Feature Importance by Gender")
        st.image("notebooks/outputs/shap_gender_comparison.png",
                 use_container_width=True)
    elif os.path.exists("notebooks/outputs/shap_importance.png"):
        st.subheader("SHAP Feature Importance")
        st.image("notebooks/outputs/shap_importance.png", use_container_width=True)

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

st.markdown('<div class="footer"><span>Credit Scoring · Built by Visveswaran V</span><span>Causal Fairness Credit Scoring · Research portfolio</span></div>', unsafe_allow_html=True)
