"""Simple synthetic example for promptstats.compare_prompts.

Usage:
    python examples/compare_prompts_simple.py
"""

import numpy as np

import promptstats as pstats


rng = np.random.default_rng(7)

# Synthetic per-input scores for three prompt variants.
# A and B are similarly strong; C is clearly weaker.
# Replace with your own data or generation process as desired.
n_inputs = 120
scores_dict = {
    # These are normal distributions around the mean (loc) and standard deviation (scale)
    "prompt_a": rng.normal(loc=0.80, scale=0.08, size=n_inputs),
    "prompt_b": rng.normal(loc=0.81, scale=0.06, size=n_inputs),
    "prompt_c": rng.normal(loc=0.70, scale=0.12, size=n_inputs),
}

report = pstats.compare_prompts(
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

