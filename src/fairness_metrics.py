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
    df = pd.DataFrame({
        "prediction": y_pred,
        "group":      sensitive_feature
    })
    rates = df.groupby("group")["prediction"].mean()
    gap   = abs(rates.max() - rates.min())

    print("\n===== DEMOGRAPHIC PARITY =====")
    for group, rate in rates.items():
        print(f"  {group}: {rate:.4f}")
    print(f"  Parity Gap: {gap:.4f}")
    if gap > 0.05:
        print("  ⚠️  BIAS DETECTED — gap exceeds 0.05 threshold")
    else:
        print("  ✅  Gap within acceptable range")

    return rates, gap


# ── Metric 2: Equal Opportunity ───────────────────────────────
def equal_opportunity(y_true, y_pred, sensitive_feature):
    """
    Checks if True Positive Rate (TPR) is equal across groups.
    Ideal: TPR should be similar for all groups.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group":  sensitive_feature
    })

    results = {}
    print("\n===== EQUAL OPPORTUNITY (True Positive Rate) =====")

    for group in df["group"].unique():
        subset = df[df["group"] == group]
        try:
            tn, fp, fn, tp = confusion_matrix(
                subset["y_true"], subset["y_pred"]
            ).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        except ValueError:
            tpr = 0
        results[group] = tpr
        print(f"  {group}: TPR = {tpr:.4f}")

    tpr_values = list(results.values())
    gap = abs(max(tpr_values) - min(tpr_values))
    print(f"  TPR Gap: {gap:.4f}")
    if gap > 0.05:
        print("  ⚠️  BIAS DETECTED — TPR gap exceeds 0.05 threshold")
    else:
        print("  ✅  TPR within acceptable range")

    return results, gap


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