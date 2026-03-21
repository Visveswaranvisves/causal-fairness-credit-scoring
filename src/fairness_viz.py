import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)


def plot_demographic_parity(dp_rates_baseline, dp_rates_fair, save=True):
    groups = dp_rates_baseline.index.tolist()
    x      = np.arange(len(groups))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, dp_rates_baseline.values, width,
           label="Baseline", color="tomato",    alpha=0.85)
    ax.bar(x + width/2, dp_rates_fair.values,     width,
           label="Fair Model", color="steelblue", alpha=0.85)

    ax.axhline(y=dp_rates_baseline.mean(), color="red",
               linestyle="--", linewidth=1, label="Baseline Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("Approval Rate")
    ax.set_title("Demographic Parity: Baseline vs Fair Model")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save:
        plt.savefig("outputs/demographic_parity.png", dpi=150)
        print("Saved: outputs/demographic_parity.png")
    plt.show()


def plot_equal_opportunity(eo_baseline, eo_fair, save=True):
    groups   = list(eo_baseline.keys())
    base_tpr = [eo_baseline[g] for g in groups]
    fair_tpr = [eo_fair[g]     for g in groups]
    x        = np.arange(len(groups))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, base_tpr, width,
           label="Baseline",   color="tomato",    alpha=0.85)
    ax.bar(x + width/2, fair_tpr, width,
           label="Fair Model", color="steelblue", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Equal Opportunity: Baseline vs Fair Model")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save:
        plt.savefig("outputs/equal_opportunity.png", dpi=150)
        print("Saved: outputs/equal_opportunity.png")
    plt.show()


def plot_bias_summary(dp_gap_base, dp_gap_fair, eo_gap_base, eo_gap_fair, save=True):
    metrics  = ["Demographic Parity Gap", "Equal Opportunity Gap"]
    baseline = [dp_gap_base, eo_gap_base]
    fair     = [dp_gap_fair, eo_gap_fair]
    x        = np.arange(len(metrics))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, baseline, width,
                   label="Baseline",   color="tomato",    alpha=0.85)
    bars2 = ax.bar(x + width/2, fair,    width,
                   label="Fair Model", color="steelblue", alpha=0.85)

    # Add value labels on bars
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0.05, color="orange", linestyle="--",
               linewidth=1.5, label="Fairness Threshold (0.05)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Gap (lower = fairer)")
    ax.set_title("Bias Summary: Before vs After Debiasing")
    ax.legend()
    ax.set_ylim(0, 0.3)
    plt.tight_layout()

    if save:
        plt.savefig("outputs/bias_summary.png", dpi=150)
        print("Saved: outputs/bias_summary.png")
    plt.show()