import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# ── Metric 1: Demographic Parity ─────────────────────────────
def demographic_parity(y_pred, sensitive_feature):
    """
    Checks if approval rates are equal across groups.
    Ideal: rates should be close to equal.
    """
    y_pred = np.array([str(p) for p in y_pred])
    sensitive_feature = np.array([str(s) for s in sensitive_feature])
    
    df = pd.DataFrame({
        "prediction": y_pred,
        "group":      sensitive_feature
    })
    rates = df.groupby("group")["prediction"].apply(
        lambda x: (x == "good").mean()
    )
    gap = abs(rates.max() - rates.min())

    print("\n===== DEMOGRAPHIC PARITY =====")
    for group, rate in rates.items():
        print(f"  {group}: {rate:.4f}")
    print(f"  Gap: {gap:.4f}")

    return rates, gap


# ── Metric 2: Equal Opportunity ───────────────────────────────
def equal_opportunity(y_true, y_pred, sensitive_feature):
    y_true = np.array([str(v) for v in y_true])
    y_pred = np.array([str(p) for p in y_pred])
    sensitive_feature = np.array([str(s) for s in sensitive_feature])
    
    df = pd.DataFrame({
        "actual":     y_true,
        "prediction": y_pred,
        "group":      sensitive_feature
    })
    tpr = {}
    for group in df["group"].unique():
        subset   = df[df["group"] == group]
        positive = subset[subset["actual"] == "good"]
        tpr[group] = (positive["prediction"] == "good").mean() if len(positive) > 0 else 0.0
    gap = abs(max(tpr.values()) - min(tpr.values()))
    return tpr, gap

# ── Metric 3: Predictive Parity ───────────────────────────────
def predictive_parity(y_true, y_pred, sensitive_feature):
    """
    Checks if precision (PPV) is equal across groups.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group":  sensitive_feature
    })

    results = {}
    print("\n===== PREDICTIVE PARITY (Precision) =====")

    for group in df["group"].unique():
        subset = df[df["group"] == group]
        try:
            tn, fp, fn, tp = confusion_matrix(
                subset["y_true"], subset["y_pred"]
            ).ravel()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        except ValueError:
            ppv = 0
        results[group] = ppv
        print(f"  {group}: Precision = {ppv:.4f}")

    return results


# ── Summary Report ────────────────────────────────────────────
def fairness_report(y_true, y_pred, sensitive_feature, group_name="Sex"):
    print(f"\n{'='*50}")
    print(f"  FAIRNESS AUDIT REPORT — Sensitive: {group_name}")
    print(f"{'='*50}")

    dp_rates, dp_gap = demographic_parity(y_pred, sensitive_feature)
    eo_results, eo_gap = equal_opportunity(y_true, y_pred, sensitive_feature)
    pp_results = predictive_parity(y_true, y_pred, sensitive_feature)

    print(f"\n===== SUMMARY =====")
    print(f"  Demographic Parity Gap : {dp_gap:.4f}")
    print(f"  Equal Opportunity Gap  : {eo_gap:.4f}")

    overall_bias = "HIGH ⚠️" if (dp_gap > 0.05 or eo_gap > 0.05) else "LOW ✅"
    print(f"  Overall Bias Level     : {overall_bias}")

    return {
        "dp_rates":   dp_rates,
        "dp_gap":     dp_gap,
        "eo_results": eo_results,
        "eo_gap":     eo_gap,
        "pp_results": pp_results
    }