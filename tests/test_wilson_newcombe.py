"""Tests for Wilson and Newcombe score intervals for binary eval data.

Covers:
  - wilson_ci / wilson_ci_1d in resampling.py
  - newcombe_paired_ci in resampling.py
  - _mcnemar_p in paired.py
  - pairwise_differences with method='wilson'
  - bootstrap_point_advantage with method='wilson'
  - is_binary_scores detection in resampling.py
  - analyze() auto-detecting binary data and using Wilson/Newcombe
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from promptstats.core.resampling import (
    is_binary_scores,
    wilson_ci,
    wilson_ci_1d,
    newcombe_paired_ci,
)
from promptstats.core.paired import (
    _mcnemar_p,
    pairwise_differences,
    all_pairwise,
)
from promptstats.core.ranking import bootstrap_point_advantage
from promptstats.core.types import BenchmarkResult
from promptstats.core.router import analyze


# ---------------------------------------------------------------------------
# is_binary_scores
# ---------------------------------------------------------------------------

def test_is_binary_scores_true_for_zeros_and_ones():
    scores = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    assert is_binary_scores(scores) is True


def test_is_binary_scores_false_for_continuous():
    scores = np.array([[0.5, 0.8, 0.3], [0.1, 0.9, 0.6]])
    assert is_binary_scores(scores) is False


def test_is_binary_scores_false_for_mixed():
    scores = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 1.0]])
    assert is_binary_scores(scores) is False


def test_is_binary_scores_3d():
    scores = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 1]]], dtype=float)
    assert is_binary_scores(scores) is True


def test_is_binary_scores_empty_returns_false():
    assert is_binary_scores(np.array([])) is False


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_ci_exact_boundary_cases():
    # All successes: p_hat = 1.0, CI upper should be at most 1.0
    lo, hi = wilson_ci(10, 10, alpha=0.05)
    assert lo > 0.6
    assert hi <= 1.0

    # No successes: p_hat = 0.0, CI lower should be 0.0
    lo, hi = wilson_ci(0, 10, alpha=0.05)
    assert lo == 0.0
    assert hi < 0.4

    # n = 0 → (0.0, 0.0)
    assert wilson_ci(0, 0, 0.05) == (0.0, 0.0)


def test_wilson_ci_midpoint():
    # 5/10 successes, 95% CI should bracket 0.5
    lo, hi = wilson_ci(5, 10, alpha=0.05)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_narrower_with_larger_n():
    lo_small, hi_small = wilson_ci(5, 10, alpha=0.05)
    lo_large, hi_large = wilson_ci(50, 100, alpha=0.05)
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_large < width_small


def test_wilson_ci_1d_matches_wilson_ci():
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    successes = int(values.sum())
    n = len(values)
    alpha = 0.05

    lo1, hi1 = wilson_ci_1d(values, alpha)
    lo2, hi2 = wilson_ci(successes, n, alpha)
    assert lo1 == lo2
    assert hi1 == hi2


# ---------------------------------------------------------------------------
# newcombe_paired_ci
# ---------------------------------------------------------------------------

def test_newcombe_paired_ci_no_discordant_pairs():
    # Identical arrays → m = 0 → (0.0, 0.0)
    a = np.array([1.0, 1.0, 0.0, 1.0])
    b = np.array([1.0, 1.0, 0.0, 1.0])
    lo, hi = newcombe_paired_ci(a, b, alpha=0.05)
    assert lo == 0.0
    assert hi == 0.0


def test_newcombe_paired_ci_all_discordant_a_wins():
    # A always wins discordant pairs → CI should be positive
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    lo, hi = newcombe_paired_ci(a, b, alpha=0.05)
    # Difference = 1.0 exactly; CI should be entirely positive
    assert lo > 0.0
    assert hi <= 1.0


def test_newcombe_paired_ci_symmetric():
    # CI(A - B) = -CI(B - A) (reversed endpoints)
    a = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    alpha = 0.05
    lo_ab, hi_ab = newcombe_paired_ci(a, b, alpha)
    lo_ba, hi_ba = newcombe_paired_ci(b, a, alpha)
    np.testing.assert_allclose(lo_ab, -hi_ba, atol=1e-10)
    np.testing.assert_allclose(hi_ab, -lo_ba, atol=1e-10)


def test_newcombe_paired_ci_covers_true_diff():
    # p_a = 0.8, p_b = 0.5, true diff = 0.3
    rng = np.random.default_rng(42)
    n = 100
    a = rng.binomial(1, 0.8, size=n).astype(float)
    b = rng.binomial(1, 0.5, size=n).astype(float)
    lo, hi = newcombe_paired_ci(a, b, alpha=0.05)
    true_diff = 0.3
    assert lo < true_diff < hi, f"95% CI [{lo:.3f}, {hi:.3f}] did not cover true diff {true_diff}"


# ---------------------------------------------------------------------------
# _mcnemar_p
# ---------------------------------------------------------------------------

def test_mcnemar_p_no_discordant_pairs():
    a = np.array([1.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 1.0])
    assert _mcnemar_p(a, b) == 1.0


def test_mcnemar_p_extreme_case_a_always_wins():
    # n10 = 10, n01 = 0 → p ~ 2 * Binom(0, 0.5)^10 very small
    a = np.ones(10)
    b = np.zeros(10)
    p = _mcnemar_p(a, b)
    assert p < 0.01


def test_mcnemar_p_balanced_discordant_pairs():
    # n10 = 5, n01 = 5 → not significant
    a = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
    b = np.array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
    p = _mcnemar_p(a, b)
    assert p > 0.9  # very balanced, no signal


def test_mcnemar_p_bounded():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.7, size=30).astype(float)
    b = rng.binomial(1, 0.5, size=30).astype(float)
    p = _mcnemar_p(a, b)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# pairwise_differences with method='newcombe'
# ---------------------------------------------------------------------------

def test_pairwise_differences_newcombe_uses_newcombe():
    rng = np.random.default_rng(7)
    n_templates = 2
    m_inputs = 40
    scores = np.zeros((n_templates, m_inputs))
    scores[0] = rng.binomial(1, 0.8, m_inputs)  # template A
    scores[1] = rng.binomial(1, 0.5, m_inputs)  # template B

    result = pairwise_differences(
        scores, 0, 1, "A", "B", method="newcombe", ci=0.95,
    )
    assert result.test_method == "newcombe (mcnemar p-value)"
    assert result.ci_low <= result.point_diff <= result.ci_high
    assert 0.0 <= result.p_value <= 1.0


def test_pairwise_differences_newcombe_no_difference():
    # Identical templates → CI should contain 0, p should be 1.0
    a = np.array([1., 0., 1., 1., 0., 0., 1., 0.])
    b = a.copy()
    scores = np.stack([a, b])
    result = pairwise_differences(scores, 0, 1, "A", "B", method="newcombe", ci=0.95)
    assert result.point_diff == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.p_value == 1.0


def test_pairwise_differences_newcombe_seeded_falls_back_to_smooth():
    # When R >= 3, newcombe should fall back to smooth_bootstrap
    rng = np.random.default_rng(9)
    scores = rng.binomial(1, 0.7, size=(2, 20, 5)).astype(float)  # (N, M, R=5)
    result = pairwise_differences(
        scores, 0, 1, "A", "B", method="newcombe", ci=0.95,
        rng=np.random.default_rng(9),
    )
    # Should use smooth bootstrap, not newcombe
    assert "newcombe" not in result.test_method
    assert "smooth" in result.test_method


# ---------------------------------------------------------------------------
# bootstrap_point_advantage with method='wilson'
# ---------------------------------------------------------------------------

def test_bootstrap_point_advantage_wilson_grand_mean():
    rng = np.random.default_rng(11)
    n_templates = 3
    m_inputs = 30
    scores = np.zeros((n_templates, m_inputs))
    for i in range(n_templates):
        scores[i] = rng.binomial(1, 0.5 + 0.1 * i, m_inputs)

    result = bootstrap_point_advantage(
        scores, ["A", "B", "C"],
        reference="grand_mean", method="wilson", ci=0.95,
    )
    assert result.n_bootstrap == 0  # Wilson, no bootstrap
    assert len(result.point_advantages) == 3
    assert np.all(result.bootstrap_ci_low <= result.bootstrap_ci_high)


def test_bootstrap_point_advantage_wilson_specific_reference():
    rng = np.random.default_rng(13)
    scores = np.zeros((3, 20))
    scores[0] = rng.binomial(1, 0.6, 20)
    scores[1] = rng.binomial(1, 0.8, 20)
    scores[2] = rng.binomial(1, 0.5, 20)

    result = bootstrap_point_advantage(
        scores, ["A", "B", "C"],
        reference="A", method="wilson", ci=0.95,
    )
    # Reference template A has zero advantage vs itself
    idx_a = result.labels.index("A")
    assert result.bootstrap_ci_low[idx_a] == 0.0
    assert result.bootstrap_ci_high[idx_a] == 0.0


# ---------------------------------------------------------------------------
# analyze() auto-detection of binary data
# ---------------------------------------------------------------------------

def _make_benchmark(scores: np.ndarray, labels: list[str]) -> BenchmarkResult:
    """Helper to construct a BenchmarkResult with auto-generated input labels."""
    m = scores.shape[1]
    input_labels = [f"q{i}" for i in range(m)]
    return BenchmarkResult(
        scores=scores,
        template_labels=labels,
        input_labels=input_labels,
    )


def test_analyze_auto_detects_binary_and_uses_wilson():
    rng = np.random.default_rng(42)
    n_templates = 3
    m_inputs = 50
    scores = np.zeros((n_templates, m_inputs))
    for i in range(n_templates):
        scores[i] = rng.binomial(1, 0.5 + 0.1 * i, m_inputs)

    result_obj = _make_benchmark(scores, ["low", "mid", "high"])
    bundle = analyze(result_obj, method="auto", rng=np.random.default_rng(42))

    # Pairwise comparisons should use Newcombe
    pair = bundle.pairwise.get("low", "mid")
    assert "newcombe" in pair.test_method

    # Advantage CIs should have n_bootstrap=0 (Wilson, no bootstrap)
    assert bundle.point_advantage.n_bootstrap == 0


def test_analyze_non_binary_still_uses_smooth_bootstrap():
    rng = np.random.default_rng(77)
    scores = rng.uniform(0, 1, size=(2, 30))

    result_obj = _make_benchmark(scores, ["A", "B"])
    bundle = analyze(result_obj, method="auto", rng=np.random.default_rng(77))

    pair = bundle.pairwise.get("A", "B")
    assert "smooth" in pair.test_method
    assert bundle.point_advantage.n_bootstrap > 0


def test_analyze_explicit_bootstrap_method_overrides_binary_detection():
    # If the user explicitly sets method='smooth_bootstrap', use it even for binary data
    rng = np.random.default_rng(99)
    scores = rng.binomial(1, 0.6, size=(2, 40)).astype(float)

    result_obj = _make_benchmark(scores, ["A", "B"])
    bundle = analyze(result_obj, method="smooth_bootstrap", rng=np.random.default_rng(99))

    pair = bundle.pairwise.get("A", "B")
    assert "smooth" in pair.test_method


# ---------------------------------------------------------------------------
# stress/property tests for Wilson/Newcombe correctness
# ---------------------------------------------------------------------------

def _make_pairs_from_counts(n10: int, n01: int, n11: int, n00: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct paired binary arrays from discordant/concordant counts."""
    a = np.array([1.0] * n10 + [0.0] * n01 + [1.0] * n11 + [0.0] * n00)
    b = np.array([0.0] * n10 + [1.0] * n01 + [1.0] * n11 + [0.0] * n00)
    return a, b


@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05, 0.01])
def test_wilson_ci_matches_scipy_reference_grid(alpha: float):
    # Deterministic grid check against SciPy's Wilson implementation.
    for n in [1, 2, 3, 5, 10, 20, 50, 200]:
        for k in range(n + 1):
            lo, hi = wilson_ci(k, n, alpha=alpha)
            ref = stats.binomtest(k, n).proportion_ci(
                confidence_level=1.0 - alpha,
                method="wilson",
            )
            np.testing.assert_allclose(lo, ref.low, atol=1e-12)
            np.testing.assert_allclose(hi, ref.high, atol=1e-12)


def test_wilson_ci_interval_width_monotone_in_confidence():
    # Higher confidence (lower alpha) must not produce narrower intervals.
    for n in [5, 10, 30, 100]:
        for k in range(n + 1):
            lo_99, hi_99 = wilson_ci(k, n, alpha=0.01)
            lo_95, hi_95 = wilson_ci(k, n, alpha=0.05)
            width_99 = hi_99 - lo_99
            width_95 = hi_95 - lo_95
            assert width_99 >= width_95


def test_newcombe_matches_count_formula_exhaustive_small_n():
    # Exhaustive over all discordant-count combinations for small n.
    alpha = 0.05
    for n in range(1, 16):
        for n10 in range(n + 1):
            for n01 in range(n + 1 - n10):
                m = n10 + n01
                concordant = n - m
                n11 = concordant // 2
                n00 = concordant - n11

                a, b = _make_pairs_from_counts(n10, n01, n11, n00)
                lo, hi = newcombe_paired_ci(a, b, alpha=alpha)

                if m == 0:
                    assert (lo, hi) == (0.0, 0.0)
                    continue

                t_lo, t_hi = wilson_ci(n10, m, alpha)
                expected_lo = (m / n) * (2.0 * t_lo - 1.0)
                expected_hi = (m / n) * (2.0 * t_hi - 1.0)
                np.testing.assert_allclose(lo, expected_lo, atol=1e-12)
                np.testing.assert_allclose(hi, expected_hi, atol=1e-12)


def test_newcombe_invariant_to_pair_order_and_concordant_mix():
    # CI should depend only on n10, n01, and n (not order or n11/n00 split).
    n10, n01, n11, n00 = 9, 5, 7, 11
    alpha = 0.05

    a1, b1 = _make_pairs_from_counts(n10, n01, n11, n00)
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(a1))
    a2 = a1[perm]
    b2 = b1[perm]

    # Alternate concordant allocation with same n and discordant counts.
    n = n10 + n01 + n11 + n00
    n11_alt = 0
    n00_alt = n - (n10 + n01)
    a3, b3 = _make_pairs_from_counts(n10, n01, n11_alt, n00_alt)

    lo1, hi1 = newcombe_paired_ci(a1, b1, alpha)
    lo2, hi2 = newcombe_paired_ci(a2, b2, alpha)
    lo3, hi3 = newcombe_paired_ci(a3, b3, alpha)

    np.testing.assert_allclose([lo1, hi1], [lo2, hi2], atol=1e-12)
    np.testing.assert_allclose([lo1, hi1], [lo3, hi3], atol=1e-12)

