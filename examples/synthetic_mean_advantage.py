"""Test and demo for promptstats: generates synthetic data and produces the
mean advantage plot with dual uncertainty bands."""

import numpy as np

import promptstats as pstats

# --- Synthetic benchmark data ---
# 6 templates, 100 inputs, designed to illustrate different behaviors
rng = np.random.default_rng(42)
M = 100  # inputs

scores = np.zeros((6, M))

# Template A: "The Reliable Winner" — high mean, low variance
scores[0] = rng.normal(loc=8.0, scale=0.8, size=M)

# Template B: "The Close Second" — slightly lower mean, also low variance
scores[1] = rng.normal(loc=7.5, scale=0.9, size=M)

# Template C: "The Volatile Genius" — high mean but huge variance
# Sometimes brilliant (10), sometimes terrible (3)
scores[2] = np.where(
    rng.random(M) > 0.3,
    rng.normal(loc=9.0, scale=0.5, size=M),
    rng.normal(loc=4.0, scale=1.0, size=M),
)

# Template D: "The Mediocre but Steady" — average score, very tight
scores[3] = rng.normal(loc=6.5, scale=0.4, size=M)

# Template E: "The Underperformer" — clearly below average
scores[4] = rng.normal(loc=5.0, scale=1.2, size=M)

# Template F: "Barely Different from Average" — right at the mean, moderate variance
scores[5] = rng.normal(loc=6.8, scale=1.0, size=M)

# Clip to reasonable range
scores = np.clip(scores, 0, 10)

labels = [
    "A: Reliable Winner",
    "B: Close Second",
    "C: Volatile Genius",
    "D: Steady Mediocre",
    "E: Underperformer",
    "F: Near Average",
]
input_labels = [f"input_{i:03d}" for i in range(M)]

# --- Create BenchmarkResult ---
result = pstats.BenchmarkResult(
    scores=scores,
    template_labels=labels,
    input_labels=input_labels,
)

print(f"Created BenchmarkResult: {result.n_templates} templates × {result.n_inputs} inputs")
print()

# --- Robustness metrics ---
rob = pstats.robustness_metrics(scores, labels, failure_threshold=4.0)
print("=== Robustness Summary ===")
print(rob.summary_table().to_string())
print()

# --- Pairwise differences (A vs C as an interesting pair) ---
diff_ac = pstats.pairwise_differences(
    scores, 0, 2, labels[0], labels[2], method="auto", rng=rng,
)
print(f"=== Pairwise: {diff_ac.template_a} vs {diff_ac.template_b} ===")
print(f"  Mean diff: {diff_ac.mean_diff:+.3f}")
print(f"  95% CI:    [{diff_ac.ci_low:+.3f}, {diff_ac.ci_high:+.3f}]")
print(f"  p-value:   {diff_ac.p_value:.4f}")
print(f"  Effect size (Cohen's d): {diff_ac.effect_size:.3f}")
print(f"  Significant: {diff_ac.significant}")
print()

# --- Bootstrap ranking ---
ranks = pstats.bootstrap_ranks(scores, labels, n_bootstrap=10_000, rng=rng)
print("=== Bootstrap Rank Probabilities ===")
print(f"{'Template':<25s} {'P(Best)':>8s} {'E[Rank]':>8s}")
for i, label in enumerate(labels):
    print(f"  {label:<23s} {ranks.p_best[i]:>7.1%} {ranks.expected_ranks[i]:>7.2f}")
print()

# --- Mean advantage (the key computation for the plot) ---
adv = pstats.bootstrap_point_advantage(
    scores, labels, reference="grand_mean", n_bootstrap=10_000, rng=rng,
)
print("=== Mean Advantage over Grand Mean ===")
print(f"{'Template':<25s} {'Mean':>7s} {'CI Low':>8s} {'CI High':>8s} {'Spread Lo':>10s} {'Spread Hi':>10s}")
for i in range(len(labels)):
    print(
        f"  {labels[i]:<23s} "
        f"{adv.point_advantages[i]:>+6.3f} "
        f"{adv.bootstrap_ci_low[i]:>+7.3f} "
        f"{adv.bootstrap_ci_high[i]:>+7.3f} "
        f"{adv.spread_low[i]:>+9.3f} "
        f"{adv.spread_high[i]:>+9.3f}"
    )
print()

# --- Generate the plot ---
fig = pstats.plot_point_advantage(result, reference="grand_mean", rng=np.random.default_rng(42))
fig.savefig("mean_advantage_plot.png", dpi=150, bbox_inches="tight")
print("Saved: mean_advantage_plot.png")

# Also generate a version comparing against a specific baseline
fig2 = pstats.plot_point_advantage(
    result,
    reference="A: Reliable Winner",
    title="Mean Advantage over 'A: Reliable Winner'",
    rng=np.random.default_rng(42),
)
fig2.savefig("mean_advantage_vs_baseline.png", dpi=150, bbox_inches="tight")
print("Saved: mean_advantage_vs_baseline.png")

print("\nDone!")
