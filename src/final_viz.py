import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("outputs", exist_ok=True)

COLORS = {
    "Baseline":              "tomato",
    "Fair (Reweighted)":     "steelblue",
    "Causal 1 (No Sex)":     "mediumseagreen",
    "Causal 2 (No Proxies)": "mediumpurple"
}


def plot_full_comparison(df, save=True):
    models   = df["Model"].tolist()
    colors   = [COLORS.get(m, "gray") for m in models]
    x        = np.arange(len(models))
    width    = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Final Model Comparison — Accuracy vs Fairness",
                 fontsize=15, fontweight="bold", y=1.02)

    metrics = [
        ("Accuracy",    "Accuracy",    (0.50, 0.85), None),
        ("Parity Gap",  "Parity Gap",  (0.00, 0.15), 0.05),
        ("EO Gap",      "EO Gap",      (0.00, 0.20), 0.05),
    ]

    for ax, (title, col, ylim, threshold) in zip(axes, metrics):
        bars = ax.bar(x, df[col], color=colors, alpha=0.85, width=0.5)
        if threshold:
            ax.axhline(y=threshold, color="orange", linestyle="--",
                       linewidth=1.5, label=f"Threshold ({threshold})")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(ylim)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ylim[1] * 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig("outputs/final_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved: outputs/final_comparison.png")
    plt.show()


def plot_tradeoff_scatter(df, save=True):
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        color = COLORS.get(row["Model"], "gray")
        ax.scatter(row["Parity Gap"], row["Accuracy"],
                   color=color, s=200, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(row["Model"],
                    xy=(row["Parity Gap"], row["Accuracy"]),
                    xytext=(8, 5), textcoords="offset points", fontsize=9)

    # Draw arrow from baseline toward ideal
    ax.annotate("", xy=(df["Parity Gap"].min() - 0.002, df["Accuracy"].max()),
                xytext=(df.iloc[0]["Parity Gap"], df.iloc[0]["Accuracy"]),
                arrowprops=dict(arrowstyle="->", color="green",
                                lw=2, connectionstyle="arc3,rad=0.2"))
    ax.text(df["Parity Gap"].min() - 0.002, df["Accuracy"].max() + 0.003,
            "Ideal direction", color="green", fontsize=9)

    ax.axvline(x=0.05, color="orange", linestyle="--",
               linewidth=1.5, label="Fairness threshold (0.05)")
    ax.set_xlabel("Demographic Parity Gap (↓ fairer)", fontsize=11)
    ax.set_ylabel("Accuracy (↑ better)",               fontsize=11)
    ax.set_title("Fairness vs Accuracy Trade-off",      fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        plt.savefig("outputs/tradeoff_scatter.png", dpi=150, bbox_inches="tight")
        print("Saved: outputs/tradeoff_scatter.png")
    plt.show()


def plot_bias_reduction_bars(df, save=True):
    models    = df["Model"].tolist()[1:]  # skip baseline
    reduction = df["Bias Reduction %"].tolist()[1:]
    acc_drop  = df["Accuracy Drop %"].tolist()[1:]
    colors    = [COLORS.get(m, "gray") for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, reduction, width,
                   label="Bias Reduction %", color="mediumseagreen", alpha=0.85)
    bars2 = ax.bar(x + width/2, acc_drop,  width,
                   label="Accuracy Drop %",  color="tomato",        alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Bias Reduction vs Accuracy Cost per Technique",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(reduction + acc_drop) + 10)
    plt.tight_layout()

    if save:
        plt.savefig("outputs/bias_reduction.png", dpi=150, bbox_inches="tight")
        print("Saved: outputs/bias_reduction.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("outputs/final_comparison.csv")
    print(df)
    plot_full_comparison(df)
    plot_tradeoff_scatter(df)
    plot_bias_reduction_bars(df)
    print("\nAll final charts saved ")