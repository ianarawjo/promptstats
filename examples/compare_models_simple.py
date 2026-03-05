"""Simple synthetic example for promptstats.compare_models.

Usage:
    python examples/compare_models_simple.py
"""

import numpy as np

import promptstats as pstats


rng = np.random.default_rng(7)

# Synthetic per-input scores for three models across two prompt templates.
# model_A and model_B are similarly strong; model_C is clearly weaker.
# Replace with your own data or generation process as desired.
n_inputs = 120
scores_dict = {
    "model_A": {
        "template_1": rng.normal(loc=0.81, scale=0.07, size=n_inputs),
        "template_2": rng.normal(loc=0.82, scale=0.06, size=n_inputs),
        "template_3": rng.normal(loc=0.79, scale=0.08, size=n_inputs),
    },
    "model_B": {
        "template_1": rng.normal(loc=0.80, scale=0.07, size=n_inputs),
        "template_2": rng.normal(loc=0.83, scale=0.07, size=n_inputs),
        "template_3": rng.normal(loc=0.78, scale=0.08, size=n_inputs),
    },
    "model_C": {
        "template_1": rng.normal(loc=0.70, scale=0.10, size=n_inputs),
        "template_2": rng.normal(loc=0.72, scale=0.10, size=n_inputs),
        "template_3": rng.normal(loc=0.68, scale=0.10, size=n_inputs),
    },
}

report = pstats.compare_models(
    scores_dict,
    statistic="mean",
    correction="holm",
    n_bootstrap=2_000,
    rng=np.random.default_rng(42),
)

print("Quick summary:")
print(" ", report.quick_summary())
print()

print("Detailed summary:")
report.summary()
print()