from pathlib import Path
from html import escape
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(page_title='Credit Scoring | Risk & Fairness',page_icon='◉',layout='wide')
os.chdir(Path(__file__).resolve().parents[1])
st.markdown('''<style>
.stApp{background:#eeede7;color:#202720}.block-container{max-width:1440px;padding:2.4rem 2.4rem 3rem}
[data-testid="stHeader"]{background:#eeede7}[data-testid="stSidebar"]{background:#eeede7;border-right:1px solid #dedfd6;width:230px!important}
h1,h2,h3{color:#202720;letter-spacing:-.04em}h1{font-size:2.15rem!important;font-weight:550!important}h3{font-size:1.12rem!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{background:#fff;border-radius:22px;border-color:transparent!important;padding:12px}
[data-testid="stMetric"]{background:white;border-radius:20px;padding:20px 22px;min-height:112px}
[data-testid="stMetricLabel"]{color:#737c72;font-size:12px}[data-testid="stMetricValue"]{font-size:1.8rem;font-weight:650;letter-spacing:-.04em}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:12px 15px;border-radius:30px;margin-bottom:7px;background:#ffffff70}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#253e34;color:white}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#385e4c;color:white;border:0;border-radius:12px;min-height:46px}
[data-testid="stForm"]{border:0;padding:0}[data-testid="stCaptionContainer"]{color:#707b70}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:35px;font-weight:650;font-size:17px}.sun{font-size:40px;color:#42664e;line-height:1}
.kicker{color:#778377;font-size:10px;letter-spacing:1.8px;text-transform:uppercase;font-weight:700}.subtle{color:#798176;font-size:12px}.gauge-card{text-align:center;padding:8px 10px 0}.gauge-card svg{max-width:260px;width:100%;display:block;margin:auto}
.pill{display:inline-block;background:#e7eee8;color:#41634f;padding:5px 10px;border-radius:20px;font-size:11px}.wallet{border-radius:18px;padding:24px;background:linear-gradient(120deg,#4e725e,#829c8a);color:white;min-height:146px;margin:8px 0 18px}.wallet .subtle{color:#e0e9e0}.wallet strong{display:block;font-size:26px;margin:15px 0 6px;letter-spacing:-1px}.empty{padding:32px 15px;text-align:center;color:#798176;font-size:13px}.divider{height:1px;background:#e8ebe4;margin:20px 0}.bottom{font-size:11px;color:#7e867b;margin-top:25px}
@media(max-width:768px){.block-container{padding:1.6rem 1rem}h1{font-size:1.8rem!important}[data-testid="stMetric"]{padding:15px}[data-testid="stMetricValue"]{font-size:1.5rem}}
</style>''',unsafe_allow_html=True)
FEATURE_COLS = ['Age','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose']
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
assets=load_assets()
df=load_data()
with st.sidebar:
 st.markdown('<div class="brand"><span class="sun">✺</span><span>Credit Scoring<br><span class="subtle">Risk & fairness workspace</span></span></div>',unsafe_allow_html=True)
 page=st.radio('Workspace',['Assessment','Fairness audit','Model comparison'],label_visibility='collapsed')
 st.divider()
 st.caption('GERMAN CREDIT DATASET')
 st.write(f'{len(df):,} reference records')
 st.caption('Portfolio research · No lending decisions')
 st.markdown('<div class="bottom">Visveswaran V<br>Responsible AI research</div>',unsafe_allow_html=True)

st.markdown('<div class="kicker">Credit intelligence / '+escape(page)+'</div>',unsafe_allow_html=True)
st.title({'Assessment':'A clearer view of credit risk.','Fairness audit':'Look beyond the headline score.','Model comparison':'Compare the trade-offs.'}[page])
st.caption({'Assessment':'Build an applicant profile and examine how each model evaluates it.','Fairness audit':'Reference-label distributions and saved model diagnostics, side by side.','Model comparison':'Saved evaluation results from the project’s research pipeline.'}[page])

if page=='Assessment':
 top=st.columns(4)
 top[0].metric('Reference applicants',f'{len(df):,}')
 top[1].metric('Model options','4')
 top[2].metric('Reference good-credit rate',f'{df.Risk.eq("good").mean():.1%}')
 top[3].metric('Prediction target','Credit risk')
 st.write('')
 left,right=st.columns([1.6,1],gap='medium')
 with left,st.container(border=True):
  st.subheader('Applicant details')
  st.caption('Inputs are used for this assessment only.')
  with st.form('applicant_form'):
   a,b=st.columns(2)
   age=a.number_input('Age',18,75,30)
   sex=b.selectbox('Sex',['male','female'])
   amount=a.number_input('Credit amount (€)',100,20000,2500)
   duration=b.number_input('Duration (months)',6,72,24)
   job=a.selectbox('Job level',[0,1,2,3])
   housing=b.selectbox('Housing',['own','free','rent'])
   saving=a.selectbox('Saving accounts',['little','moderate','quite rich','rich'])
   checking=b.selectbox('Checking account',['little','moderate','rich'])
   purpose=a.selectbox('Purpose',['car','furniture/equipment','radio/TV','education','business'])
   choice=b.selectbox('Model',['Baseline','Fair (Threshold)','Causal 1 (No Sex)','Causal 2 (No Proxies)'])
   submit=st.form_submit_button('Calculate assessment →',type='primary',use_container_width=True)
  if submit:
   try:
    inputs={'Age':age,'Job':job,'Housing':housing,'Saving accounts':saving,'Checking account':checking,'Credit amount':amount,'Duration':duration,'Purpose':purpose}
    keys=('model','imputer','scaler');drops=None
    if choice=='Causal 1 (No Sex)':keys=('causal1','imputer_c1','scaler_c1');drops=['Sex_male']
    if choice=='Causal 2 (No Proxies)':keys=('causal2','imputer_c2','scaler_c2');drops=['Sex_male','Purpose_car','Purpose_education']
    thresholds=assets.get('thresholds',{'female':.45,'male':.50}) if choice=='Fair (Threshold)' else None
    label,score=predict(prepare_input(inputs,drops),*(assets[k] for k in keys),sex=sex,thresholds=thresholds)
    st.session_state.assessment={'label':label,'score':float(score),'model':choice,'amount':amount,'duration':duration,'threshold':thresholds.get(sex,.5) if thresholds else .5}
   except Exception as exc:
    st.session_state.pop('assessment',None)
    st.error(f'Assessment failed: {exc}')
 with right:
  with st.container(border=True):
   st.subheader('Assessment summary')
   r=st.session_state.get('assessment')
   value=f'{r["score"]:.1%}' if r else '—'
   length=314*(r['score'] if r else 0)
   st.markdown(f'<div class="gauge-card"><svg viewBox="0 0 260 170" role="img" aria-label="Good-credit model score {value}"><path d="M30 130 A100 100 0 0 1 230 130" fill="none" stroke="#e8eee8" stroke-width="24"/><path d="M30 130 A100 100 0 0 1 230 130" fill="none" stroke="#729782" stroke-width="24" stroke-dasharray="{length} 314"/><text x="130" y="115" text-anchor="middle" font-family="Arial" font-size="38" fill="#263d30">{value}</text><text x="130" y="151" text-anchor="middle" font-family="Arial" font-size="11" fill="#748170">GOOD-CREDIT MODEL SCORE</text></svg></div>',unsafe_allow_html=True)
   if r:
    st.markdown(f'<span class="pill">{escape(r["label"].title())} credit risk · Model classification</span>',unsafe_allow_html=True)
    st.caption(f'{r["model"]} · Threshold {r["threshold"]:.2f}')
    st.caption('Shows the last submitted profile. Recalculate after editing inputs.')
   else:st.markdown('<div class="empty">Complete the profile and calculate<br>to view the model estimate.</div>',unsafe_allow_html=True)
  with st.container(border=True):
   st.subheader('Credit request')
   amount_text=f'€ {r["amount"]:,.0f}' if r else 'Awaiting assessment'
   term=f'{r["duration"]} month term' if r else 'Submit the form to create a summary'
   st.markdown(f'<div class="wallet"><span class="subtle">RESEARCH ASSESSMENT</span><strong>{amount_text}</strong><span>{term}</span></div>',unsafe_allow_html=True)
   st.caption('Model scores are not calibrated approval probabilities. This demo does not approve or deny credit.')
else:
 if page=='Fairness audit':
  male=df.loc[df.Sex.eq('male'),'Risk'].eq('good').mean();female=df.loc[df.Sex.eq('female'),'Risk'].eq('good').mean()
  a,b,c=st.columns(3);a.metric('Male reference good-credit rate',f'{male:.1%}');b.metric('Female reference good-credit rate',f'{female:.1%}');c.metric('Absolute reference-label gap',f'{abs(male-female):.4f}')
  st.caption('These rates describe dataset labels, not model approval decisions or a live fairness audit.')
  a,b=st.columns(2)
  for col,path,title in [(a,'outputs/demographic_parity.png','Saved demographic parity'),(b,'models/fairness_by_sex.png','Saved group comparison')]:
   with col,st.container(border=True):
    st.subheader(title)
    if Path(path).exists():st.image(path,use_container_width=True)
  with st.expander('Feature attribution by demographic group'):
   path=Path('notebooks/outputs/shap_gender_comparison.png')
   if path.exists():st.image(str(path),use_container_width=True)
 else:
  comparison=pd.read_csv('outputs/model_comparison.csv')
  a,b,c=st.columns(3)
  a.metric('Highest saved accuracy',f'{comparison.Accuracy.max():.1%}')
  b.metric('Smallest saved parity gap',f'{comparison["Parity Gap"].min():.4f}')
  c.metric('Models evaluated',str(len(comparison)))
  with st.container(border=True):
   st.subheader('Performance & fairness')
   st.dataframe(comparison,hide_index=True,use_container_width=True)
   st.caption('Higher accuracy and lower gaps are preferred on these metrics. The table contains saved evaluation results; it does not update with applicant inputs.')
   if Path('outputs/model_comparison.png').exists():st.image('outputs/model_comparison.png',use_container_width=True)
   st.download_button('Download comparison',comparison.to_csv(index=False),'credit_model_comparison.csv','text/csv')
st.markdown('<div class="bottom">Causal Fairness-Aware Credit Scoring · Research portfolio · German Credit Dataset</div>',unsafe_allow_html=True)
