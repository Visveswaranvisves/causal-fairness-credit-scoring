import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)


def plot_model_comparison(results_df, save=True):
    """
    results_df must have columns: Model, Accuracy, Parity Gap, EO Gap
    """
    models   = results_df["Model"].tolist()
    accuracy = results_df["Accuracy"].tolist()
    parity   = results_df["Parity Gap"].tolist()
    eo_gap   = results_df["EO Gap"].tolist()

    x     = np.arange(len(models))
    width = 0.25
    colors = ["tomato", "steelblue", "mediumseagreen", "mediumpurple"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison: Accuracy vs Fairness Trade-off", fontsize=14)

    # Accuracy
    bars = axes[0].bar(x, accuracy, color=colors, alpha=0.85)
    axes[0].set_title("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0.5, 0.85)
    axes[0].set_ylabel("Accuracy")
    for bar, val in zip(bars, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Parity Gap
    bars = axes[1].bar(x, parity, color=colors, alpha=0.85)
    axes[1].axhline(y=0.05, color="orange", linestyle="--",
                    linewidth=1.5, label="Threshold (0.05)")
    axes[1].set_title("Demographic Parity Gap")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylim(0, 0.15)
    axes[1].set_ylabel("Gap (lower = fairer)")
    axes[1].legend(fontsize=8)
    for bar, val in zip(bars, parity):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # EO Gap
    bars = axes[2].bar(x, eo_gap, color=colors, alpha=0.85)
    axes[2].axhline(y=0.05, color="orange", linestyle="--",
                    linewidth=1.5, label="Threshold (0.05)")
    axes[2].set_title("Equal Opportunity Gap")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylim(0, 0.2)
    axes[2].set_ylabel("Gap (lower = fairer)")
    axes[2].legend(fontsize=8)
    for bar, val in zip(bars, eo_gap):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig("outputs/model_comparison.png", dpi=150)
        print("Saved: outputs/model_comparison.png")
    plt.show()


def plot_fairness_accuracy_tradeoff(results_df, save=True):
    """
    Scatter plot showing the fairness vs accuracy trade-off.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tomato", "steelblue", "mediumseagreen", "mediumpurple"]

    for i, row in results_df.iterrows():
        ax.scatter(row["Parity Gap"], row["Accuracy"],
                   color=colors[i], s=150, zorder=5)
        ax.annotate(row["Model"],
                    xy=(row["Parity Gap"], row["Accuracy"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=9)

    ax.axvline(x=0.05, color="orange", linestyle="--",
               linewidth=1.5, label="Fairness Threshold (0.05)")
    ax.set_xlabel("Demographic Parity Gap (lower = fairer)", fontsize=11)
    ax.set_ylabel("Accuracy (higher = better)",              fontsize=11)
    ax.set_title("Fairness vs Accuracy Trade-off",           fontsize=13)
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig("outputs/fairness_accuracy_tradeoff.png", dpi=150)
        print("Saved: outputs/fairness_accuracy_tradeoff.png")
    plt.show()