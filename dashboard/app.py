from pathlib import Path
from html import escape
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from PIL import Image, ImageDraw


def make_tab_icon():
    """Browser-tab icon: dark green tile with a credit card (theme colours)."""
    try:
        s = 256
        img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=60, fill='#253e34')
        d.rounded_rectangle((28, 62, 228, 196), radius=24, fill='#e7eee8')
        d.rectangle((28, 94, 228, 120), fill='#729782')
        d.rounded_rectangle((48, 140, 94, 172), radius=8, fill='#385e4c')
        d.rounded_rectangle((110, 150, 190, 166), radius=8, fill='#b9cdbf')
        return img
    except Exception:
        return '💳'


st.set_page_config(page_title='Credit Scoring | Risk & Fairness',page_icon=make_tab_icon(),layout='wide')
os.chdir(Path(__file__).resolve().parents[1])

# ---------- Finance illustrations (inline SVG, theme colours only) ----------
G_DARK = '#253e34'
G_MID = '#385e4c'
G_SOFT = '#729782'
G_LIGHT = '#829c8a'
G_PALE = '#e7eee8'

HERO_ART_SVG = (
    '<svg viewBox="0 0 260 170" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    f'<g transform="rotate(-9 110 90)"><rect x="34" y="28" width="170" height="106" rx="14" fill="{G_MID}" stroke="{G_SOFT}" stroke-width="2"/>'
    f'<rect x="34" y="52" width="170" height="16" fill="{G_SOFT}" opacity=".55"/></g>'
    f'<rect x="60" y="40" width="172" height="108" rx="14" fill="{G_LIGHT}"/>'
    f'<rect x="78" y="64" width="36" height="28" rx="6" fill="{G_PALE}"/>'
    f'<path d="M78 78 H114 M96 64 V92" stroke="{G_LIGHT}" stroke-width="1.6"/>'
    '<g stroke="#fff" stroke-width="2.4" opacity=".9"><path d="M196 64 q8 10 0 20"/><path d="M204 58 q13 16 0 32"/><path d="M212 52 q18 22 0 44"/></g>'
    '<g fill="#fff" opacity=".85"><rect x="78" y="112" width="28" height="6" rx="3"/><rect x="112" y="112" width="28" height="6" rx="3"/><rect x="146" y="112" width="28" height="6" rx="3"/><rect x="180" y="112" width="28" height="6" rx="3"/></g>'
    '<rect x="78" y="128" width="54" height="5" rx="2.5" fill="#fff" opacity=".5"/>'
    f'<circle cx="222" cy="140" r="26" fill="{G_PALE}" stroke="{G_SOFT}" stroke-width="3"/>'
    f'<circle cx="222" cy="140" r="19" stroke="{G_SOFT}" stroke-width="1.6" stroke-dasharray="3 3"/>'
    f'<text x="222" y="150" text-anchor="middle" font-family="Arial" font-size="28" font-weight="700" fill="{G_MID}">€</text>'
    '</svg>'
)

TREND_SVG = (
    '<svg viewBox="0 0 420 44" preserveAspectRatio="none" width="100%" height="44" xmlns="http://www.w3.org/2000/svg" fill="none">'
    '<path d="M0 38 L40 32 L80 35 L120 24 L160 28 L200 17 L240 22 L280 10 L320 15 L420 4 L420 44 L0 44 Z" fill="#a8c9b4" opacity=".14"/>'
    '<path d="M0 38 L40 32 L80 35 L120 24 L160 28 L200 17 L240 22 L280 10 L320 15 L420 4" stroke="#a8c9b4" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
    '</svg>'
)

BRAND_SVG = (
    '<svg viewBox="0 0 40 40" width="36" height="36" xmlns="http://www.w3.org/2000/svg" fill="none">'
    f'<rect x="1" y="1" width="38" height="38" rx="12" fill="{G_DARK}"/>'
    f'<rect x="7" y="12" width="26" height="17" rx="4" fill="{G_PALE}"/>'
    f'<rect x="7" y="16" width="26" height="4" fill="{G_SOFT}"/>'
    f'<rect x="10" y="23" width="7" height="3.5" rx="1.5" fill="{G_MID}"/>'
    '</svg>'
)

SIDE_CARD_SVG = (
    '<svg viewBox="0 0 64 44" width="56" height="38" xmlns="http://www.w3.org/2000/svg" fill="none">'
    f'<rect x="1" y="1" width="62" height="42" rx="8" fill="{G_SOFT}"/>'
    f'<rect x="8" y="11" width="14" height="11" rx="3" fill="{G_PALE}"/>'
    '<g stroke="#fff" stroke-width="1.8" stroke-linecap="round"><path d="M43 12 q4 5 0 10"/><path d="M48 9 q7 8 0 16"/></g>'
    '<rect x="8" y="31" width="30" height="3.5" rx="1.75" fill="#fff" opacity=".7"/>'
    '</svg>'
)

WALLET_ART_SVG = (
    '<svg viewBox="0 0 76 46" width="68" height="41" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round">'
    '<rect x="1" y="6" width="32" height="26" rx="7" fill="#fff" opacity=".88"/>'
    f'<path d="M1 19 H33 M17 6 V32" stroke="{G_LIGHT}" stroke-width="1.6"/>'
    '<g stroke="#fff" stroke-width="2.4" opacity=".95"><path d="M46 12 q6 8 0 16"/><path d="M54 7 q10 13 0 26"/><path d="M62 2 q14 18 0 38"/></g>'
    '</svg>'
)

def icon(path_markup):
    return (f'<svg viewBox="0 0 48 48" width="30" height="30" xmlns="http://www.w3.org/2000/svg" fill="none" '
            f'stroke="{G_MID}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">{path_markup}</svg>')

ICON_SCALE = icon('<path d="M24 8 V40 M14 40 H34"/><path d="M10 14 H38"/><path d="M10 14 L4 28 a6 6 0 0 0 12 0 Z"/><path d="M38 14 L32 28 a6 6 0 0 0 12 0 Z"/>')
ICON_SHIELD = icon('<path d="M24 5 L39 11 V23 c0 9 -6 16 -15 20 c-9 -4 -15 -11 -15 -20 V11 Z"/><path d="M17 24 l5 5 l9 -10"/>')
ICON_BARS = icon('<path d="M6 42 H42"/><rect x="9" y="26" width="7" height="16" rx="1.5"/><rect x="20.5" y="16" width="7" height="26" rx="1.5"/><rect x="32" y="6" width="7" height="36" rx="1.5"/>')
ICON_NODES = icon('<circle cx="10" cy="24" r="5"/><circle cx="38" cy="10" r="5"/><circle cx="38" cy="38" r="5"/><path d="M15 22 L33 12 M15 26 L33 36 M38 15 V33"/>')
ICON_COINS = icon('<ellipse cx="22" cy="14" rx="14" ry="6"/><path d="M8 14 V26 c0 3.3 6.3 6 14 6 s14 -2.7 14 -6 V14"/><path d="M8 26 V36 c0 3.3 6.3 6 14 6 s14 -2.7 14 -6 V32"/>')
ICON_MAGNIFY = icon('<circle cx="21" cy="21" r="13"/><path d="M31 31 L42 42"/><path d="M15 25 V20 M21 25 V15 M27 25 V22"/>')


def hero(tag, headline, text, chips):
    chip_html = ''.join(f'<span class="chip">{escape(c)}</span>' for c in chips)
    return (
        '<div class="hero"><div class="hero-copy">'
        f'<span class="hero-tag">{escape(tag)}</span>'
        f'<div class="hero-title">{escape(headline)}</div>'
        f'<div class="hero-text">{escape(text)}</div>'
        f'<div class="trend">{TREND_SVG}</div>'
        f'<div class="chip-row">{chip_html}</div>'
        '</div>'
        f'<div class="hero-art">{HERO_ART_SVG}</div>'
        '</div>'
    )


def ctx_card(svg, title, body):
    return (f'<div class="ctx-card"><div class="ctx-icon">{svg}</div>'
            f'<div class="ctx-title">{escape(title)}</div><div class="ctx-body">{escape(body)}</div></div>')


st.markdown('''<style>
.stApp{background:#eeede7;color:#202720}.block-container{max-width:1440px;padding:2.4rem 2.4rem 3rem}
[data-testid="stHeader"]{background:#eeede7}[data-testid="stSidebar"]{background:#eeede7;border-right:1px solid #dedfd6;width:230px!important}
h1,h2,h3{color:#202720;letter-spacing:-.04em}h1{font-size:2.15rem!important;font-weight:550!important}h3{font-size:1.12rem!important}
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

/* Scope surfaces to named cards, never to Streamlit's structural wrappers. */
[class*="st-key-panel_"]{background:#fff;border:1px solid #e0e2dc!important;border-radius:20px;padding:20px!important;box-sizing:border-box;min-width:0}
[data-testid="stMetric"]{box-sizing:border-box;min-width:0;width:100%}
[data-testid="stSidebarUserContent"]{padding:1.5rem 1rem}
[data-testid="stCaptionContainer"] p{font-size:13px;line-height:1.5}
[data-testid="stMainBlockContainer"]{width:100%;max-width:1600px;padding:2.8rem 2rem 2rem}
[data-testid="stHorizontalBlock"]{min-width:0}
[data-testid="stColumn"]{min-width:0}
@media(max-width:768px){[data-testid="stMainBlockContainer"]{padding:2rem 1rem}[class*="st-key-panel_"]{padding:16px!important}}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#fff!important}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:14px}
[data-testid="stMetric"]{padding:16px 18px;min-height:94px;border:1px solid #e1e4dc;border-radius:16px}
[data-testid="stMetricValue"]{font-size:1.65rem}
[data-testid="stForm"] [data-testid="stVerticalBlock"]{gap:.65rem}
.brand{margin-bottom:24px;font-size:17px;gap:9px}.brand .subtle{font-size:11px}
.empty{padding:16px 8px;color:#586554}.subtle{color:#596556}
.gauge-card svg{max-width:225px}.wallet{min-height:120px;padding:20px;margin:4px 0 8px}
h1{font-size:2rem!important}.bottom{color:#586554}

/* ===== Spacing fixes ===== */
/* Streamlit gives inner elements a fixed pixel width; inside a padded card that pushes them past the right edge. */
[class*="st-key-panel_"] [data-testid="stElementContainer"],
[class*="st-key-panel_"] [data-testid="stElementContainer"]>div,
[class*="st-key-panel_"] [data-testid="stForm"],
[class*="st-key-panel_"] [data-testid="stImage"],
[class*="st-key-panel_"] [data-testid="stDataFrame"],
[class*="st-key-panel_"] [data-testid="stMarkdownContainer"]{width:100%!important;max-width:100%!important;box-sizing:border-box;min-width:0}
[class*="st-key-panel_"]{padding:22px 22px 18px!important;gap:.85rem;overflow-x:clip}
[class*="st-key-panel_"] h3{margin:0;padding:0 0 .1rem}
[class*="st-key-panel_"] [data-testid="stImage"] img{max-width:100%;height:auto;border-radius:12px}
.wallet{width:100%;box-sizing:border-box;margin:2px 0 4px}
.gauge-card{width:100%;box-sizing:border-box;padding:4px 10px 0}
h1{padding:.2rem 0 .1rem!important;line-height:1.15!important}
[data-testid="stMainBlockContainer"]{padding-bottom:4rem}
[data-testid="stMainBlockContainer"]>div>[data-testid="stVerticalBlock"]{gap:1rem}
[data-testid="stHorizontalBlock"]{gap:1rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}
[data-testid="stCaptionContainer"]{color:#5d695c}
.bottom{margin-top:18px;padding-bottom:6px}
/* Let a card that is alone in its column fill the column so side-by-side cards end at the same line. */
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]:has(>[data-testid="stLayoutWrapper"]:only-child){height:100%}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]>[data-testid="stLayoutWrapper"]:only-child{flex:1 1 auto}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]>[data-testid="stLayoutWrapper"]:only-child>[class*="st-key-panel_"]{height:100%}

/* ===== Finance elements ===== */
.brand svg{flex:none}
.side-card{background:#253e34;border-radius:20px;padding:16px;margin-top:18px;color:#fff}
.side-card .side-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.side-card .side-top span{font-size:10px;letter-spacing:.08em;font-weight:650;background:#e7eee8;color:#253e34;padding:5px 10px;border-radius:20px}
.side-card strong{display:block;font-size:15px;letter-spacing:-.02em}
.side-card p{font-size:12px;line-height:1.5;color:#c9d8ce;margin:6px 0 0}
.hero{display:flex;align-items:stretch;justify-content:space-between;gap:22px;background:#253e34;border-radius:24px;padding:24px 28px;margin:4px 0 6px;box-sizing:border-box;width:100%}
.hero-copy{flex:1;min-width:0;display:flex;flex-direction:column}
.hero-tag{font-size:11px;letter-spacing:.12em;color:#a8c9b4;font-weight:600}
.hero-title{font-size:26px;line-height:1.2;letter-spacing:-.04em;color:#fff;font-weight:550;margin:8px 0 6px}
.hero-text{font-size:13px;line-height:1.6;color:#c9d8ce;max-width:560px}
.trend{margin:12px 0 10px;max-width:560px}
.chip-row{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:12px;color:#253e34;background:#e7eee8;border-radius:20px;padding:5px 12px;font-weight:550}
.chip:nth-child(even){background:#a8c9b4}
.hero-art{flex:none;width:240px;height:156px;align-self:center}
.wallet{position:relative;overflow:hidden}
.wallet-art{position:absolute;right:20px;top:18px}
.ctx-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:4px 0 4px}
.ctx-card{background:#fff;border:1px solid #e0e2dc;border-radius:20px;padding:20px;box-sizing:border-box;min-width:0}
.ctx-icon{width:50px;height:50px;border-radius:16px;background:#e7eee8;display:grid;place-items:center;margin-bottom:14px}
.ctx-title{font-size:15px;font-weight:600;color:#202720;letter-spacing:-.02em;margin-bottom:6px}
.ctx-body{font-size:13px;line-height:1.55;color:#586554}
@media(max-width:1100px){.ctx-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:768px){.hero{flex-direction:column;padding:20px}.hero-art{display:none}.ctx-grid{grid-template-columns:1fr}.hero-title{font-size:21px}}

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
 st.markdown(f'<div class="brand">{BRAND_SVG}<span>Credit Scoring<br><span class="subtle">Risk & fairness workspace</span></span></div>',unsafe_allow_html=True)
 page=st.radio('Workspace',['Assessment','Fairness audit','Model comparison'],label_visibility='collapsed')
 st.divider()
 st.caption('GERMAN CREDIT DATASET')
 st.write(f'{len(df):,} reference records')
 st.caption('Portfolio research · No lending decisions')
 st.markdown(f'<div class="side-card"><div class="side-top">{SIDE_CARD_SVG}<span>RESEARCH</span></div><strong>Risk, made fair.</strong><p>Compare models, then look beyond the headline score.</p></div>',unsafe_allow_html=True)
 st.markdown('<div class="bottom">Visveswaran V<br>Responsible AI research</div>',unsafe_allow_html=True)

st.markdown('<div class="kicker">Credit intelligence / '+escape(page)+'</div>',unsafe_allow_html=True)
st.title({'Assessment':'A clearer view of credit risk.','Fairness audit':'Look beyond the headline score.','Model comparison':'Compare the trade-offs.'}[page])
st.caption({'Assessment':'Build an applicant profile and examine how each model evaluates it.','Fairness audit':'Reference-label distributions and saved model diagnostics, side by side.','Model comparison':'Saved evaluation results from the project’s research pipeline.'}[page])
HERO={'Assessment':('CREDIT RISK · RESPONSIBLE AI','Credit scoring, examined for fairness.','Compare baseline, threshold-adjusted and causal models on the same applicant profile.'),
 'Fairness audit':('FAIRNESS AUDIT · GERMAN CREDIT DATA','Fair lending starts with measurement.','Reference-label rates and saved diagnostics show where outcomes differ between groups.'),
 'Model comparison':('MODEL COMPARISON · RESEARCH PIPELINE','Accuracy and fairness, side by side.','Saved results show the trade-off between predictive performance and parity.')}[page]
st.markdown(hero(*HERO,['German Credit data',f'{len(df):,} records','Causal inference']),unsafe_allow_html=True)

if page=='Assessment':
 top=st.columns(4)
 top[0].metric('Reference applicants',f'{len(df):,}')
 top[1].metric('Model options','4')
 top[2].metric('Reference good-credit rate',f'{df.Risk.eq("good").mean():.1%}')
 top[3].metric('Prediction target','Credit risk')
 left,right=st.columns([1.6,1],gap='medium')
 with left,st.container(border=False,key="panel_1"):
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
  with st.container(border=False,key="panel_2"):
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
  with st.container(border=False,key="panel_3"):
   st.subheader('Credit request')
   amount_text=f'€ {r["amount"]:,.0f}' if r else 'Awaiting assessment'
   term=f'{r["duration"]} month term' if r else 'Submit the form to create a summary'
   st.markdown(f'<div class="wallet"><div class="wallet-art">{WALLET_ART_SVG}</div><span class="subtle">RESEARCH ASSESSMENT</span><strong>{amount_text}</strong><span>{term}</span></div>',unsafe_allow_html=True)
   st.caption('Model scores are not calibrated approval probabilities. This demo does not approve or deny credit.')
else:
 if page=='Fairness audit':
  male=df.loc[df.Sex.eq('male'),'Risk'].eq('good').mean();female=df.loc[df.Sex.eq('female'),'Risk'].eq('good').mean()
  a,b,c=st.columns(3);a.metric('Male reference good-credit rate',f'{male:.1%}');b.metric('Female reference good-credit rate',f'{female:.1%}');c.metric('Absolute reference-label gap',f'{abs(male-female):.4f}')
  st.caption('These rates describe dataset labels, not model approval decisions or a live fairness audit.')
  st.markdown('<div class="ctx-grid">'
   +ctx_card(ICON_SCALE,'Four fairness metrics','Demographic parity, equal opportunity, equalized odds and counterfactual fairness.')
   +ctx_card(ICON_NODES,'Causal graph','Shows how sex, age and finances relate, so proxies for sex can be found.')
   +ctx_card(ICON_MAGNIFY,'SHAP explanations','Feature attribution shows what pushes each model score up or down.')
   +ctx_card(ICON_SHIELD,'Responsible lending','Research only. No credit is approved or denied here.')
   +'</div>',unsafe_allow_html=True)
  a,b=st.columns(2)
  for col,path,title in [(a,'outputs/demographic_parity.png','Saved demographic parity'),(b,'models/fairness_by_sex.png','Saved group comparison')]:
   with col,st.container(border=False,key="panel_"+title):
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
  st.markdown('<div class="ctx-grid">'
   +ctx_card(ICON_BARS,'Baseline','Standard logistic model trained on all applicant features.')
   +ctx_card(ICON_COINS,'Fair (Threshold)','Uses group-specific decision thresholds to narrow outcome gaps.')
   +ctx_card(ICON_NODES,'Causal 1 · No Sex','Removes the sex feature before the model is trained.')
   +ctx_card(ICON_SHIELD,'Causal 2 · No Proxies','Also removes features that act as proxies for sex.')
   +'</div>',unsafe_allow_html=True)
  with st.container(border=False,key="panel_5"):
   st.subheader('Performance & fairness')
   st.dataframe(comparison,hide_index=True,use_container_width=True)
   st.caption('Higher accuracy and lower gaps are preferred on these metrics. The table contains saved evaluation results; it does not update with applicant inputs.')
   if Path('outputs/model_comparison.png').exists():st.image('outputs/model_comparison.png',use_container_width=True)
   st.download_button('Download comparison',comparison.to_csv(index=False),'credit_model_comparison.csv','text/csv')
st.markdown('<div class="bottom">Causal Fairness-Aware Credit Scoring · Research portfolio · German Credit Dataset</div>',unsafe_allow_html=True)
