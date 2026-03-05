import numpy as np
import pytest

import promptstats as ps
from promptstats.compare import CompareReport, EntityStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_compare_prompts_requires_dict():
    with pytest.raises(TypeError, match="dict"):
        ps.compare_prompts([[1, 0, 1], [1, 1, 1]])


def test_compare_prompts_requires_at_least_two_prompts():
    with pytest.raises(ValueError, match="at least 2"):
        ps.compare_prompts({"only_one": [1, 0, 1]})


def test_compare_prompts_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same number of inputs"):
        ps.compare_prompts(
            {"a": [1, 0, 1], "b": [1, 1]},
            rng=_rng(),
        )


def test_compare_prompts_mixed_1d_2d_raises():
    with pytest.raises(ValueError, match="mix of 1-D and 2-D"):
        ps.compare_prompts(
            {"a": [1, 0, 1], "b": [[1, 0], [0, 1], [1, 1]]},
            rng=_rng(),
        )


def test_compare_prompts_mismatched_run_counts():
    with pytest.raises(ValueError, match="same number of runs"):
        ps.compare_prompts(
            {
                "a": [[1, 0, 1], [0, 1, 1]],   # R=3
                "b": [[1, 0], [0, 1]],           # R=2
            },
            rng=_rng(),
        )


def test_compare_prompts_wrong_ndim():
    with pytest.raises(ValueError, match="dimensions"):
        ps.compare_prompts(
            {"a": np.ones((3, 2, 2)), "b": np.ones((3, 2, 2))},
            rng=_rng(),
        )


# ---------------------------------------------------------------------------
# Return type and attribute shapes
# ---------------------------------------------------------------------------

def test_compare_prompts_returns_report():
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [1, 1, 1, 1, 0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert isinstance(report, CompareReport)


def test_report_has_expected_attributes():
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [1, 1, 1, 1, 0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert set(report.labels) == {"a", "b"}
    assert set(report.means.keys()) == {"a", "b"}
    assert set(report.prompt_stats.keys()) == {"a", "b"}
    assert set(report.pairwise_p_values.keys()) == {("a", "b")}
    assert set(report.pairwise_p_values[("a", "b")].keys()) == {"p_boot", "p_wilcoxon"}
    assert isinstance(report.p_best, float)
    assert 0.0 <= report.p_best <= 1.0
    assert report.winners in (None, ["a"], ["b"])
    assert isinstance(report.significant, bool)
    assert isinstance(report.quick_summary(), str) and len(report.quick_summary()) > 0
    assert isinstance(report.full_analysis, ps.AnalysisBundle)


def test_report_means_are_correct():
    report = ps.compare_prompts(
        {"a": [0.0, 1.0, 0.0], "b": [1.0, 1.0, 1.0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert report.means["a"] == pytest.approx(1 / 3, abs=1e-9)
    assert report.means["b"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# prompt_stats: PromptStats per template
# ---------------------------------------------------------------------------

def test_prompt_stats_fields_present():
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.85, 0.92, 0.75, 0.88] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    for label in ("a", "b"):
        s = report.prompt_stats[label]
        assert isinstance(s, EntityStats)
        assert isinstance(s.mean, float)
        assert isinstance(s.median, float)
        assert isinstance(s.std, float)
        assert isinstance(s.ci_low, float)
        assert isinstance(s.ci_high, float)
        # CI should straddle the mean
        assert s.ci_low <= s.mean <= s.ci_high


def test_prompt_stats_ci_ordering():
    """ci_low < ci_high for all templates."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    for label in report.labels:
        s = report.prompt_stats[label]
        assert s.ci_low < s.ci_high


def test_pairwise_p_values_populated_for_two_way():
    """2-way comparison: pairwise p-values should be populated in tuple-keyed dict."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    pair = report.pairwise_p_values[("a", "b")]
    assert pair["p_boot"] is not None
    assert 0.0 <= pair["p_boot"] <= 1.0


def test_pairwise_p_values_match_pairwise_matrix():
    """pairwise_p_values p_boot should equal PairwiseMatrix p_value."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    pair_p = report.pairwise.get("a", "b").p_value
    assert report.pairwise_p_values[("a", "b")]["p_boot"] == pytest.approx(pair_p, abs=1e-12)


def test_pairwise_p_values_present_for_all_nway_pairs():
    """N-way comparison: pairwise_p_values should contain all pair tuples."""
    report = ps.compare_prompts(
        {
            "a": [0.8, 0.9, 0.7] * 5,
            "b": [0.6, 0.5, 0.65] * 5,
            "c": [0.7, 0.75, 0.68] * 5,
        },
        n_bootstrap=500,
        rng=_rng(),
    )
    assert set(report.pairwise_p_values.keys()) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_get_pairwise_p_values_works_both_directions():
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    ab = report.get_pairwise_p_values("a", "b")
    ba = report.get_pairwise_p_values("b", "a")
    assert ab["p_boot"] == pytest.approx(ba["p_boot"], abs=1e-12)
    if ab["p_wilcoxon"] is None:
        assert ba["p_wilcoxon"] is None
    else:
        assert ab["p_wilcoxon"] == pytest.approx(ba["p_wilcoxon"], abs=1e-12)


# ---------------------------------------------------------------------------
# Two-prompt comparisons
# ---------------------------------------------------------------------------

def test_significant_difference_detected():
    """Large mean difference on many inputs should yield a significant result."""
    rng = _rng(42)
    a_scores = rng.normal(loc=0.5, scale=0.1, size=100)
    b_scores = rng.normal(loc=0.8, scale=0.1, size=100)

    report = ps.compare_prompts(
        {"a": a_scores, "b": b_scores},
        n_bootstrap=2_000,
        rng=_rng(7),
    )
    assert report.winners == ["b"]
    assert report.significant is True
    assert report.p_best < 0.05


def test_no_significant_difference_detected():
    """Identical scores should yield no winners (all tied)."""
    scores = [0.8, 0.7, 0.9, 0.6, 0.8]
    report = ps.compare_prompts(
        {"a": scores, "b": scores},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert report.winners is None
    assert report.significant is False


def test_alpha_controls_winner_threshold():
    """With a very strict alpha the same data should not produce a winner."""
    rng = _rng(42)
    a = rng.normal(0.5, 0.1, 50)
    b = rng.normal(0.55, 0.1, 50)  # small difference

    report_strict = ps.compare_prompts(
        {"a": a, "b": b},
        alpha=0.001,
        n_bootstrap=1_000,
        rng=_rng(1),
    )
    assert report_strict.p_best > report_strict.alpha or report_strict.winners is None


# ---------------------------------------------------------------------------
# N-way comparisons (N > 2)
# ---------------------------------------------------------------------------

def test_three_way_comparison_returns_report():
    report = ps.compare_prompts(
        {
            "zero-shot": [0.80, 0.90, 0.70, 0.85, 0.75] * 4,
            "few-shot":  [0.75, 0.88, 0.65, 0.80, 0.70] * 4,
            "cot":       [0.82, 0.91, 0.73, 0.87, 0.78] * 4,
        },
        n_bootstrap=500,
        rng=_rng(),
    )
    assert len(report.labels) == 3
    assert len(report.pairwise.results) == 3


def test_three_way_single_winner_has_highest_mean():
    """When a single winner is declared it must be the highest-mean prompt."""
    rng = _rng(99)
    a = rng.normal(0.5, 0.05, 200)
    b = rng.normal(0.6, 0.05, 200)
    c = rng.normal(0.4, 0.05, 200)

    report = ps.compare_prompts(
        {"a": a, "b": b, "c": c},
        n_bootstrap=2_000,
        rng=_rng(3),
    )
    if report.winners is not None and len(report.winners) == 1:
        assert report.winners[0] == max(report.means, key=report.means.get)


def test_winners_can_include_multiple_top_prompts():
    """Top prompts can tie with each other while beating lower prompts."""
    rng = _rng(123)
    a = rng.normal(0.80, 0.02, 160)
    b = rng.normal(0.80, 0.02, 160)
    c = rng.normal(0.60, 0.02, 160)

    report = ps.compare_prompts(
        {"a": a, "b": b, "c": c},
        n_bootstrap=2_000,
        rng=_rng(8),
    )

    assert report.winners is not None
    assert set(report.winners) == {"a", "b"}


# ---------------------------------------------------------------------------
# Multi-run (nested bootstrap)
# ---------------------------------------------------------------------------

def test_multirun_1d_equivalent_shape():
    """1-D and (M, 1) inputs should produce the same means."""
    scores_1d = {"a": [0.8, 0.7, 0.9], "b": [0.85, 0.75, 0.95]}
    scores_2d = {"a": [[0.8], [0.7], [0.9]], "b": [[0.85], [0.75], [0.95]]}

    r1 = ps.compare_prompts(scores_1d, n_bootstrap=200, rng=_rng())
    r2 = ps.compare_prompts(scores_2d, n_bootstrap=200, rng=_rng())

    assert r1.means == pytest.approx(r2.means, abs=1e-9)


def test_multirun_nested_bootstrap_activates():
    """R >= 3 runs should engage the nested bootstrap path (smoke test)."""
    rng = _rng(5)
    a = rng.normal(0.5, 0.1, (20, 3))
    b = rng.normal(0.7, 0.1, (20, 3))

    report = ps.compare_prompts(
        {"a": a, "b": b},
        n_bootstrap=500,
        rng=_rng(6),
    )
    assert report.full_analysis.benchmark.n_runs == 3
    assert isinstance(report.p_best, float)


def test_multirun_mean_matches_flattened():
    """Reported mean should equal the mean of per-input cell means."""
    a_runs = np.array([[0.8, 0.9, 0.7], [0.6, 0.5, 0.7]])  # (2 inputs, 3 runs)
    b_runs = np.array([[0.9, 0.8, 0.85], [0.7, 0.65, 0.72]])

    report = ps.compare_prompts(
        {"a": a_runs, "b": b_runs},
        n_bootstrap=200,
        rng=_rng(),
    )
    expected_a = float(np.mean(a_runs.mean(axis=1)))
    expected_b = float(np.mean(b_runs.mean(axis=1)))
    assert report.means["a"] == pytest.approx(expected_a, abs=1e-9)
    assert report.means["b"] == pytest.approx(expected_b, abs=1e-9)


# ---------------------------------------------------------------------------
# Summary and print (smoke tests)
# ---------------------------------------------------------------------------

def test_summary_is_nonempty_string():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1, 0], "b": [1, 1, 1, 0, 1]},
        n_bootstrap=200,
        rng=_rng(),
    )
    assert isinstance(report.quick_summary(), str)
    assert len(report.quick_summary()) > 10


def test_summary_mentions_selected_statistic_and_correction():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1, 0], "b": [1, 1, 1, 0, 1]},
        statistic="median",
        correction="bonferroni",
        n_bootstrap=200,
        rng=_rng(),
    )
    assert "Δmedian" in report.quick_summary() or "median=" in report.quick_summary()
    assert "bonferroni-corrected" in report.quick_summary()


def test_summary_delegates_to_analysis_summary(capsys):
    """summary() should call print_analysis_summary and produce its standard output."""
    report = ps.compare_prompts(
        {"baseline": [0.8, 0.7, 0.9, 0.85] * 5, "new": [0.82, 0.75, 0.91, 0.87] * 5},
        n_bootstrap=200,
        rng=_rng(),
    )
    report.summary()
    captured = capsys.readouterr()
    # print_analysis_summary always emits these sections
    assert "Robustness" in captured.out
    assert "Rank Probabilities" in captured.out


def test_print_is_alias_for_summary(capsys):
    report = ps.compare_prompts(
        {"baseline": [0.8, 0.7, 0.9, 0.85] * 5, "new": [0.82, 0.75, 0.91, 0.87] * 5},
        n_bootstrap=200,
        rng=_rng(),
    )
    report.print()
    captured = capsys.readouterr()
    assert "Robustness" in captured.out
    assert "Rank Probabilities" in captured.out


# ---------------------------------------------------------------------------
# Pairwise access
# ---------------------------------------------------------------------------

def test_pairwise_get_works_both_directions():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1], "b": [0, 1, 0, 1]},
        n_bootstrap=200,
        rng=_rng(),
    )
    ab = report.pairwise.get("a", "b")
    ba = report.pairwise.get("b", "a")
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-12)
    assert ab.point_diff == pytest.approx(-ba.point_diff, abs=1e-12)
