# Causal Fairness-Aware Credit Scoring

> A Responsible AI system that predicts credit risk while measuring,
> analyzing, and mitigating demographic bias using causal inference techniques.


## Problem Statement

Credit scoring models trained on historical data inherit societal biases.
A model may systematically disadvantage female applicants — not from explicit
discrimination, but because training data reflects past inequality.

This project answers three questions:
1. **Does bias exist** in a standard credit scoring model?
2. **How much bias** exists, measured across multiple fairness metrics?
3. **Can we reduce it** without destroying model performance?


## Project Highlights

| Feature | Detail |
|---|---|
| Dataset | German Credit Dataset (1000 samples) |
| Sensitive Attributes | Sex, Age |
| Fairness Metrics | Demographic Parity, Equal Opportunity, Predictive Parity |
| Debiasing Techniques | Reweighting, Causal Feature Removal, Proxy Removal |
| Explainability | SHAP feature importance + gender-wise comparison |
| Deployment | Streamlit dashboard with live prediction + fairness tabs |


## Key Results

| Model | Accuracy | Parity Gap | Bias Reduction |
|---|---|---|---|
| Baseline | ~0.70 | 0.0381 | — |
| Fair (Reweighted) | ~0.68 | 0.0286 | ~25%  |
| Causal 1 (No Sex) | ~0.67 | lower | ~35%  |
| Causal 2 (No Proxies) | ~0.65 | lowest | ~45% |

> Fairness improved significantly with only a marginal accuracy trade-off —
> demonstrating that responsible AI is practical, not just theoretical.

---

## What Makes This Different

Most ML projects stop at accuracy. This project goes further:

- **Measures bias** using industry-standard fairness metrics
- **Traces root causes** using causal graph reasoning
- **Removes indirect bias** by identifying proxy features
- **Verifies fairness** using counterfactual analysis
- **Explains decisions** using SHAP values per demographic group

---

## Project Structure
```
causal-fairness-credit-scoring/
│
├── data/
│   └── german_credit_data.csv       # Raw dataset
│
├── notebooks/
│   ├── day1_eda.ipynb               # Exploratory data analysis
│   ├── day2_model.ipynb             # Baseline model training
│   ├── day3_fairness.ipynb          # Reweighting debiasing
│   ├── day4_shap.ipynb              # SHAP explainability
│   ├── day5_fairness_metrics.ipynb  # Fairness metric audit
│   ├── day6_causal_fairness.ipynb   # Causal modeling
│   └── day7_final_analysis.ipynb    # Final comparison + story
│
├── src/
│   ├── data_preprocessing.py        # Data loading and cleaning
│   ├── fairness_metrics.py          # Demographic parity, equal opportunity
│   ├── fairness_viz.py              # Fairness charts
│   ├── comparison_viz.py            # Model comparison charts
│   └── final_viz.py                 # Final polished visualizations
│
├── models/
│   ├── logistic_model.pkl           # Baseline model
│   ├── fair_model.pkl               # Reweighted fair model
│   ├── causal_model_1.pkl           # No-Sex causal model
│   ├── causal_model_2.pkl           # No-proxy causal model
│   └── feature_columns.pkl          # Saved feature list
│
├── outputs/
│   ├── final_comparison.csv         # All metrics table
│   ├── shap_importance.png
│   ├── shap_gender_gap.png
│   ├── demographic_parity.png
│   ├── equal_opportunity.png
│   ├── bias_summary.png
│   ├── final_comparison.png
│   ├── tradeoff_scatter.png
│   └── bias_reduction.png
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## Setup and Run

### 1. Clone the repo
```bash
git clone https://github.com/visveswaranvisves/causal-fairness-credit-scoring.git

cd causal-fairness-credit-scoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
# Generate all outputs and charts
python src/final_viz.py

# Launch the dashboard
streamlit run dashboard/app.py
```

### 4. Run individual notebooks

Open any notebook in `notebooks/` in order from day1 to day7.
Each notebook is self-contained and runs independently.

---

## Fairness Metrics Explained

**Demographic Parity** — Are approval rates equal across gender groups?
A gap above 0.05 indicates systemic bias.

**Equal Opportunity** — Are True Positive Rates equal?
This checks whether creditworthy applicants from all groups are approved equally.

**Counterfactual Fairness** — Would a prediction change if only gender changed?
A low change rate means decisions are driven by financial factors, not demographics.

---

## Causal Reasoning

Standard fairness techniques remove statistical correlations.
Causal fairness goes further — it removes the *causal path* from gender to prediction.

We modeled: `Gender → Income → Credit Approval` (indirect, acceptable)
vs `Gender → Credit Approval` (direct, unfair)

By removing both direct and proxy-mediated paths, we achieved stronger
fairness guarantees at a modest accuracy cost.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML & Fairness | scikit-learn, pandas, numpy |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |

---

## Team



Visveswaran V | Core DS — EDA, fairness metrics, causal modeling, SHAP |
Kowsalya P | MLOps — pipeline, visualizations, dashboard, MLflow |

---

## Interview Talking Points

> "We built a credit scoring system and found that the baseline model
> had a demographic parity gap of 0.038. We applied three techniques —
> reweighting, direct feature removal, and proxy removal — reducing bias
> by up to 45% while keeping accuracy above 0.65. We validated results
> using counterfactual analysis and explained predictions using SHAP."

---

## License