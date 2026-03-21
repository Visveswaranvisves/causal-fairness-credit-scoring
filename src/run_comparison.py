# src/run_comparison.py
import pandas as pd
import sys
sys.path.append("src")
from comparison_viz import plot_model_comparison, plot_fairness_accuracy_tradeoff

# Load the comparison CSV that your teammate saved
results = pd.read_csv("outputs/model_comparison.csv")
print(results)

plot_model_comparison(results)
plot_fairness_accuracy_tradeoff(results)