"""Tests for the LMM (Linear Mixed Model) analysis path.

All tests in this module require pymer4 and a working R installation with
lme4 and emmeans.  They are automatically skipped when those dependencies
are absent so the standard CI suite (bootstrap-only) is never broken.

How to run just these tests once R is available:

    pytest tests/test_lmm.py -v
"""

import warnings

import numpy as np
import pytest

import promptstats as ps
from promptstats import BenchmarkResult, analyze
from promptstats.core.types import MultiModelBenchmark

# ---------------------------------------------------------------------------
# Module-level skip guard: skip everything if pymer4 / R is not present.
# ---------------------------------------------------------------------------

pymer4 = pytest.importorskip(
    "pymer4",
    reason="pymer4 not installed (pip install pymer4; needs R + lme4 + emmeans)",
)

# Use _require_pymer4() rather than a bare import so that the pandas2ri
# activation fix is exercised here too — and so that a missing R / lme4
# installation produces a clean skip rather than a confusing ImportError.
try:
    from promptstats.core.mixed_effects import _require_pymer4
    _require_pymer4()
except Exception:
    pytest.skip("pymer4 installed but R/lme4 not reachable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    rng,
    n_templates: int = 3,
    n_inputs: int = 20,
    template_effects=None,
    sigma_input: float = 0.4,
    sigma_resid: float = 0.2,
) -> BenchmarkResult:
    """Synthetic benchmark drawn from the exact LMM generative model.

    score[t, i] = intercept + template_effect[t] + input_effect[i] + resid[t, i]

    This makes the LMM correctly specified, so we can test that estimates
    recover the true parameters within reasonable tolerance.
    """
    if template_effects is None:
        # Default: template 0 is best (+0.5), template 2 is worst (−0.3)
        template_effects = np.array([0.5, 0.0, -0.3])

    assert len(template_effects) == n_templates

    intercept = 5.0
    input_effects = rng.normal(0.0, sigma_input, size=n_inputs)
    resid = rng.normal(0.0, sigma_resid, size=(n_templates, n_inputs))

    scores = (
        intercept
        + template_effects[:, None]
        + input_effects[None, :]
        + resid
    )

    labels = [f"T{i}" for i in range(n_templates)]
    inputs = [f"inp_{j:03d}" for j in range(n_inputs)]
    return BenchmarkResult(scores=scores, template_labels=labels, input_labels=inputs)




def _make_result_with_missing(
    rng,
    n_templates: int = 3,
    n_inputs: int = 25,
    missing_fraction: float = 0.10,
    template_effects=None,
    sigma_input: float = 0.4,
    sigma_resid: float = 0.2,
) -> BenchmarkResult:
    """Like _make_result but with a random fraction of NaN cells (MCAR)."""
    result = _make_result(
        rng,
        n_templates=n_templates,
        n_inputs=n_inputs,
        template_effects=template_effects,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
    )
    scores = result.scores.copy()
    n_total = n_templates * n_inputs
    n_missing = max(1, int(round(n_total * missing_fraction)))
    flat_idx = rng.choice(n_total, size=n_missing, replace=False)
    scores.ravel()[flat_idx] = np.nan
    return BenchmarkResult(
        scores=scores,
        template_labels=result.template_labels,
        input_labels=result.input_labels,
    )


def _make_full_result_parametrized(
    rng,
    n_templates: int,
    n_inputs: int,
    n_runs: int,
    sigma_input: float = 0.45,
    sigma_run: float = 0.05,
    sigma_resid: float = 0.15,
) -> BenchmarkResult:
    """Synthetic complete-data benchmark for bootstrap-vs-LMM parity checks."""
    template_effects = np.linspace(0.6, -0.6, n_templates)
    intercept = 5.0
    input_effects = rng.normal(0.0, sigma_input, size=n_inputs)

    if n_runs <= 1:
        resid = rng.normal(0.0, sigma_resid, size=(n_templates, n_inputs))
        scores = (
            intercept
            + template_effects[:, None]
            + input_effects[None, :]
            + resid
        )
    else:
        run_effects = rng.normal(0.0, sigma_run, size=n_runs)
        resid = rng.normal(0.0, sigma_resid, size=(n_templates, n_inputs, n_runs))
        scores = (
            intercept
            + template_effects[:, None, None]
            + input_effects[None, :, None]
            + run_effects[None, None, :]
            + resid
        )

    return BenchmarkResult(
        scores=scores,
        template_labels=[f"T{i}" for i in range(n_templates)],
        input_labels=[f"inp_{j:03d}" for j in range(n_inputs)],
    )


def _pairwise_mean_diffs(bundle: ps.AnalysisBundle) -> np.ndarray:
    """Return pairwise mean differences in a deterministic label order."""
    labels = list(bundle.pairwise.labels)
    diffs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            diffs.append(bundle.pairwise.get(labels[i], labels[j]).point_diff)
    return np.asarray(diffs, dtype=float)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_lmm_analyze_returns_analysis_bundle():
    """analyze(method='lmm') should return an AnalysisBundle with lmm_info set."""
    result = _make_result(np.random.default_rng(0))
    bundle = analyze(result, method="lmm", n_bootstrap=500, rng=np.random.default_rng(1))

    assert isinstance(bundle, ps.AnalysisBundle)
    assert bundle.lmm_info is not None
    assert isinstance(bundle.lmm_info, ps.LMMInfo)


def test_lmm_info_fields_are_finite():
    """LMMInfo variance components and ICC should all be non-negative and finite."""
    result = _make_result(np.random.default_rng(2))
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(3))
    info = bundle.lmm_info

    assert np.isfinite(info.icc)
    assert 0.0 <= info.icc <= 1.0
    assert info.sigma_input >= 0.0
    assert info.sigma_resid >= 0.0
    assert np.isfinite(info.sigma_input)
    assert np.isfinite(info.sigma_resid)
    assert info.n_obs == result.n_templates * result.n_inputs
    assert info.converged


def test_lmm_icc_high_when_input_variance_dominates():
    """ICC should be high (>0.6) when between-input variance >> residual variance."""
    rng = np.random.default_rng(10)
    result = _make_result(rng, sigma_input=1.5, sigma_resid=0.1)
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(11))
    assert bundle.lmm_info.icc > 0.6, f"ICC={bundle.lmm_info.icc:.3f} unexpectedly low"


def test_lmm_icc_low_when_residual_variance_dominates():
    """ICC should be low (<0.4) when residual variance >> between-input variance."""
    rng = np.random.default_rng(20)
    result = _make_result(rng, sigma_input=0.05, sigma_resid=1.0)
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(21))
    assert bundle.lmm_info.icc < 0.4, f"ICC={bundle.lmm_info.icc:.3f} unexpectedly high"


# ---------------------------------------------------------------------------
# Result-type checks (shapes and labels)
# ---------------------------------------------------------------------------

def test_lmm_result_shapes_match_benchmark():
    """All per-template arrays in the bundle should have length == n_templates."""
    n_templates, n_inputs = 4, 30
    result = _make_result(
        np.random.default_rng(30),
        n_templates=n_templates,
        n_inputs=n_inputs,
        template_effects=np.array([0.6, 0.2, -0.1, -0.4]),
    )
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(31))

    N = n_templates
    assert len(bundle.rank_dist.p_best) == N
    assert len(bundle.rank_dist.expected_ranks) == N
    assert bundle.rank_dist.rank_probs.shape == (N, N)
    assert len(bundle.point_advantage.point_advantages) == N
    assert len(bundle.point_advantage.bootstrap_ci_low) == N
    assert len(bundle.point_advantage.bootstrap_ci_high) == N
    assert len(bundle.point_advantage.spread_low) == N
    assert len(bundle.point_advantage.spread_high) == N
    assert len(bundle.robustness.mean) == N


def test_lmm_labels_preserved():
    """Template labels should be identical between input and all result fields."""
    labels = ["Alpha", "Beta", "Gamma"]
    result = _make_result(
        np.random.default_rng(40),
        n_templates=3,
        template_effects=np.array([0.3, 0.0, -0.3]),
    )
    # Override labels
    result = BenchmarkResult(
        scores=result.scores,
        template_labels=labels,
        input_labels=result.input_labels,
    )
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(41))

    assert bundle.rank_dist.labels == labels
    assert bundle.point_advantage.labels == labels
    assert bundle.robustness.labels == labels
    assert list(bundle.pairwise.labels) == labels


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_templates,n_inputs,n_runs",
    [
        (2, 20, 1),
        (3, 24, 1),
        (3, 24, 4),
        (5, 32, 4),
    ],
)
def test_bootstrap_and_lmm_are_similar_on_complete_data(
    n_templates: int,
    n_inputs: int,
    n_runs: int,
):
    """Bootstrap and LMM should agree closely on complete (no-NaN) designs."""
    data_rng = np.random.default_rng(6000 + 100 * n_templates + 10 * n_inputs + n_runs)
    result = _make_full_result_parametrized(
        data_rng,
        n_templates=n_templates,
        n_inputs=n_inputs,
        n_runs=n_runs,
    )

    bootstrap_bundle = analyze(
        result,
        method="bootstrap",
        reference="grand_mean",
        n_bootstrap=600,
        rng=np.random.default_rng(7000 + n_templates + n_inputs + n_runs),
    )
    lmm_bundle = analyze(
        result,
        method="lmm",
        reference="grand_mean",
        n_bootstrap=600,
        rng=np.random.default_rng(8000 + n_templates + n_inputs + n_runs),
    )

    np.testing.assert_allclose(
        bootstrap_bundle.point_advantage.point_advantages,
        lmm_bundle.point_advantage.point_advantages,
        atol=0.10,
    )
    np.testing.assert_allclose(
        _pairwise_mean_diffs(bootstrap_bundle),
        _pairwise_mean_diffs(lmm_bundle),
        atol=0.10,
    )

    np.testing.assert_allclose(
        bootstrap_bundle.robustness.mean,
        lmm_bundle.robustness.mean,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bootstrap_bundle.robustness.std,
        lmm_bundle.robustness.std,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bootstrap_bundle.robustness.iqr,
        lmm_bundle.robustness.iqr,
        atol=1e-12,
    )

    assert int(np.argmax(bootstrap_bundle.rank_dist.p_best)) == int(
        np.argmax(lmm_bundle.rank_dist.p_best)
    )
    max_rank_gap = float(
        np.max(
            np.abs(
                bootstrap_bundle.rank_dist.expected_ranks
                - lmm_bundle.rank_dist.expected_ranks
            )
        )
    )
    assert max_rank_gap <= 1.0, f"Expected-rank gap too large: {max_rank_gap:.3f}"

    if n_runs >= 3:
        assert bootstrap_bundle.seed_variance is not None
        assert lmm_bundle.seed_variance is not None
        np.testing.assert_allclose(
            bootstrap_bundle.seed_variance.instability,
            lmm_bundle.seed_variance.instability,
            atol=1e-12,
        )

def test_lmm_mean_advantages_sum_to_zero_for_grand_mean_reference():
    """Mean advantages relative to the grand mean must sum to zero (by construction)."""
    result = _make_result(np.random.default_rng(50))
    bundle = analyze(
        result, method="lmm", reference="grand_mean",
        n_bootstrap=300, rng=np.random.default_rng(51),
    )
    total = float(bundle.point_advantage.point_advantages.sum())
    assert abs(total) < 1e-8, f"mean_advantages sum = {total:.2e}, expected 0"


def test_lmm_reference_template_has_zero_advantage():
    """When reference is a specific template, that template's advantage is exactly 0."""
    labels = ["T0", "T1", "T2"]
    result = _make_result(
        np.random.default_rng(60),
        n_templates=3,
        template_effects=np.array([0.4, 0.0, -0.3]),
    )
    result = BenchmarkResult(
        scores=result.scores, template_labels=labels, input_labels=result.input_labels
    )
    bundle = analyze(
        result, method="lmm", reference="T1",
        n_bootstrap=200, rng=np.random.default_rng(61),
    )
    ref_idx = labels.index("T1")
    assert abs(float(bundle.point_advantage.point_advantages[ref_idx])) < 1e-8


def test_lmm_recovers_best_template():
    """P(best) should be highest for the template with the largest effect.

    With a 0.5-point advantage and 30 inputs the signal is clear enough that
    the LMM reliably identifies the correct best template.
    """
    rng = np.random.default_rng(70)
    template_effects = np.array([0.5, 0.0, -0.3])
    result = _make_result(
        rng,
        n_templates=3,
        n_inputs=30,
        template_effects=template_effects,
        sigma_input=0.4,
        sigma_resid=0.15,
    )
    bundle = analyze(result, method="lmm", n_bootstrap=500, rng=np.random.default_rng(71))
    best_idx = int(np.argmax(bundle.rank_dist.p_best))
    assert best_idx == 0, (
        f"Expected T0 to have highest P(best) but got {bundle.rank_dist.labels[best_idx]}"
    )


def test_lmm_mean_advantage_sign_matches_effect_direction():
    """Templates with positive effects should have positive mean advantages."""
    rng = np.random.default_rng(80)
    template_effects = np.array([0.6, 0.0, -0.4])
    result = _make_result(
        rng, n_templates=3, n_inputs=40,
        template_effects=template_effects,
        sigma_resid=0.1,
    )
    bundle = analyze(
        result, method="lmm", reference="grand_mean",
        n_bootstrap=200, rng=np.random.default_rng(81),
    )
    adv = bundle.point_advantage.point_advantages
    # T0 best → positive advantage
    assert adv[0] > 0, f"T0 advantage {adv[0]:.4f} should be positive"
    # T2 worst → negative advantage
    assert adv[2] < 0, f"T2 advantage {adv[2]:.4f} should be negative"


def test_lmm_ci_contains_zero_for_near_equal_templates():
    """When two templates have the same true effect, the pairwise CI should include 0."""
    rng = np.random.default_rng(90)
    result = _make_result(
        rng,
        n_templates=2,
        n_inputs=25,
        template_effects=np.array([0.0, 0.0]),  # truly equal
        sigma_resid=0.3,
    )
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(91))
    pair = bundle.pairwise.get("T0", "T1")
    assert pair.ci_low <= 0.0 <= pair.ci_high, (
        f"CI [{pair.ci_low:.4f}, {pair.ci_high:.4f}] should straddle 0 for equal templates"
    )


def test_lmm_rank_probs_rows_sum_to_one():
    """Each template's rank probabilities must sum to exactly 1.0."""
    result = _make_result(np.random.default_rng(100))
    bundle = analyze(result, method="lmm", n_bootstrap=500, rng=np.random.default_rng(101))
    row_sums = bundle.rank_dist.rank_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_lmm_rank_probs_cols_sum_to_one():
    """Each rank's total probability across templates must sum to exactly 1.0."""
    result = _make_result(np.random.default_rng(110))
    bundle = analyze(result, method="lmm", n_bootstrap=500, rng=np.random.default_rng(111))
    col_sums = bundle.rank_dist.rank_probs.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-6)


def test_lmm_n_bootstrap_zero_signals_wald():
    """n_bootstrap=0 on MeanAdvantageResult signals parametric Wald, not bootstrap."""
    result = _make_result(np.random.default_rng(120))
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(121))
    assert bundle.point_advantage.n_bootstrap == 0, (
        "LMM mean_advantage should carry n_bootstrap=0 to indicate Wald (not bootstrap) CIs"
    )


# ---------------------------------------------------------------------------
# Missing data (NaN cells)
# ---------------------------------------------------------------------------

def test_lmm_accepts_nan_cells():
    """analyze(method='lmm') should succeed when scores contain NaN cells."""
    result = _make_result_with_missing(
        np.random.default_rng(200), n_inputs=30, missing_fraction=0.10
    )
    assert result.has_missing
    # Should not raise
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(201))
    assert isinstance(bundle, ps.AnalysisBundle)


def test_lmm_nan_emits_warning():
    """analyze(method='lmm') should emit a UserWarning when NaN cells are present."""
    result = _make_result_with_missing(
        np.random.default_rng(210), n_inputs=25, missing_fraction=0.15
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(211))

    mar_warnings = [w for w in caught if "MAR" in str(w.message) or "missing" in str(w.message).lower()]
    assert len(mar_warnings) >= 1, "Expected at least one missing-data UserWarning"


def test_lmm_nan_result_shapes_are_correct():
    """Result arrays should still have shape (n_templates,) with NaN cells present."""
    n_templates, n_inputs = 3, 30
    result = _make_result_with_missing(
        np.random.default_rng(220),
        n_templates=n_templates,
        n_inputs=n_inputs,
        missing_fraction=0.12,
    )
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(221))

    N = n_templates
    assert len(bundle.rank_dist.p_best) == N
    assert len(bundle.point_advantage.point_advantages) == N
    assert len(bundle.robustness.mean) == N
    assert bundle.rank_dist.rank_probs.shape == (N, N)


def test_lmm_nan_advantages_sum_to_approx_zero():
    """Grand-mean advantages should still sum to ~0 with a small fraction of NaN cells."""
    result = _make_result_with_missing(
        np.random.default_rng(230), n_inputs=30, missing_fraction=0.08
    )
    bundle = analyze(
        result, method="lmm", reference="grand_mean",
        n_bootstrap=300, rng=np.random.default_rng(231),
    )
    # With missing data the LMM mean advantages are based on the fitted model,
    # which is not constrained to sum to exactly zero, but should be close.
    total = float(bundle.point_advantage.point_advantages.sum())
    assert abs(total) < 0.5, (
        f"mean_advantages sum = {total:.4f} is too large; missing data may bias estimates"
    )


def test_lmm_nan_rank_probs_sum_to_one():
    """Row and column sums of rank_probs should be 1.0 even with missing data."""
    result = _make_result_with_missing(
        np.random.default_rng(240), n_inputs=25, missing_fraction=0.10
    )
    bundle = analyze(result, method="lmm", n_bootstrap=400, rng=np.random.default_rng(241))
    np.testing.assert_allclose(bundle.rank_dist.rank_probs.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(bundle.rank_dist.rank_probs.sum(axis=0), 1.0, atol=1e-6)


def test_lmm_nan_robustness_metrics_are_finite():
    """Robustness metrics should all be finite when some cells are NaN."""
    result = _make_result_with_missing(
        np.random.default_rng(250), n_inputs=30, missing_fraction=0.10
    )
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(251))
    rob = bundle.robustness
    assert np.all(np.isfinite(rob.mean)), "robustness.mean has non-finite values"
    assert np.all(np.isfinite(rob.std)),  "robustness.std has non-finite values"
    assert np.all(np.isfinite(rob.iqr)),  "robustness.iqr has non-finite values"


def test_lmm_nan_pairwise_n_inputs_is_reduced():
    """PairedDiffResult.n_inputs should equal the number of complete pairs, not M."""
    n_inputs = 40
    result = _make_result_with_missing(
        np.random.default_rng(260),
        n_templates=2,
        n_inputs=n_inputs,
        missing_fraction=0.20,
        template_effects=np.array([0.3, -0.3]),
    )
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(261))
    pair = bundle.pairwise.get("T0", "T1")
    # With ~20% missing and 2 templates, complete pairs should be less than M
    assert pair.n_inputs < n_inputs, (
        f"n_inputs={pair.n_inputs} should be < {n_inputs} with 20% missing data"
    )
    assert pair.n_inputs >= 1


def test_lmm_nan_high_missing_fraction_still_runs():
    """LMM should still produce results with up to 30% missing cells."""
    result = _make_result_with_missing(
        np.random.default_rng(270),
        n_inputs=40,
        missing_fraction=0.30,
    )
    # Should not raise (though results may be less reliable)
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(271))
    assert isinstance(bundle, ps.AnalysisBundle)
    assert bundle.lmm_info is not None


# ---------------------------------------------------------------------------
# Bootstrap path should reject NaN
# ---------------------------------------------------------------------------

def test_bootstrap_rejects_nan_cells():
    """analyze() with default bootstrap method should raise ValueError on NaN scores."""
    result = _make_result_with_missing(np.random.default_rng(300), n_inputs=20)
    assert result.has_missing
    with pytest.raises(ValueError, match="NaN"):
        analyze(result)  # default method='auto' → bootstrap


def test_bootstrap_method_auto_rejects_nan():
    """method='auto' should also reject NaN since it routes to bootstrap."""
    result = _make_result_with_missing(np.random.default_rng(310), n_inputs=20)
    with pytest.raises(ValueError, match="NaN"):
        analyze(result, method="auto")


def test_bootstrap_method_bootstrap_rejects_nan():
    """method='bootstrap' should explicitly reject NaN."""
    result = _make_result_with_missing(np.random.default_rng(320), n_inputs=20)
    with pytest.raises(ValueError, match="NaN"):
        analyze(result, method="bootstrap")


# ---------------------------------------------------------------------------
# Seeded (multi-run) data
# ---------------------------------------------------------------------------

def test_lmm_with_seeded_scores_populates_seed_variance():
    """When the benchmark carries R>=3 runs, seed_variance should be populated."""
    rng = np.random.default_rng(400)
    N, M, R = 3, 25, 4
    template_effects = np.array([0.4, 0.0, -0.3])
    intercept = 5.0
    input_effects = rng.normal(0, 0.4, size=M)

    scores = np.empty((N, M, R))
    for r in range(R):
        scores[:, :, r] = (
            intercept
            + template_effects[:, None]
            + input_effects[None, :]
            + rng.normal(0, 0.2, size=(N, M))
        )

    result = BenchmarkResult(
        scores=scores,
        template_labels=[f"T{i}" for i in range(N)],
        input_labels=[f"inp_{j:03d}" for j in range(M)],
    )
    assert result.is_seeded
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(401))

    assert bundle.seed_variance is not None
    assert len(bundle.seed_variance.instability) == N


def test_lmm_with_seeded_scores_shapes():
    """LMM on seeded data: all result arrays should have correct shapes."""
    rng = np.random.default_rng(410)
    N, M, R = 3, 20, 5
    scores = rng.normal(5.0, 0.5, size=(N, M, R))
    result = BenchmarkResult(
        scores=scores,
        template_labels=[f"T{i}" for i in range(N)],
        input_labels=[f"inp_{j}" for j in range(M)],
    )
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(411))

    assert bundle.point_advantage.per_input_advantages.shape == (N, M)
    assert bundle.robustness.mean.shape == (N,)
    assert bundle.rank_dist.rank_probs.shape == (N, N)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_lmm_two_templates():
    """LMM with exactly 2 templates should work and produce a single pairwise result."""
    rng = np.random.default_rng(500)
    result = _make_result(
        rng, n_templates=2, n_inputs=25,
        template_effects=np.array([0.3, -0.3]),
    )
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(501))
    assert len(bundle.rank_dist.p_best) == 2
    # Exactly one pairwise comparison (T0 vs T1)
    assert len(bundle.pairwise.results) == 1


def test_lmm_five_templates():
    """LMM with 5 templates should produce (5 choose 2) = 10 pairwise comparisons."""
    rng = np.random.default_rng(510)
    result = _make_result(
        rng, n_templates=5, n_inputs=30,
        template_effects=np.array([0.6, 0.3, 0.0, -0.2, -0.5]),
    )
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(511))
    assert len(bundle.pairwise.results) == 10  # C(5,2)
    assert bundle.rank_dist.rank_probs.shape == (5, 5)


def test_lmm_ci_bounds_are_ordered():
    """CI low should always be <= mean, and mean <= CI high, for all templates."""
    result = _make_result(np.random.default_rng(520), n_inputs=25)
    bundle = analyze(result, method="lmm", n_bootstrap=300, rng=np.random.default_rng(521))
    ma = bundle.point_advantage
    for i, label in enumerate(ma.labels):
        lo = float(ma.bootstrap_ci_low[i])
        mid = float(ma.point_advantages[i])
        hi = float(ma.bootstrap_ci_high[i])
        assert lo <= mid <= hi, (
            f"Template '{label}': CI low={lo:.4f} mean={mid:.4f} high={hi:.4f} "
            "violates lo <= mean <= hi"
        )


def test_lmm_spread_bounds_are_ordered():
    """Spread low should always be <= spread high for all templates."""
    result = _make_result(np.random.default_rng(530), n_inputs=30)
    bundle = analyze(result, method="lmm", n_bootstrap=200, rng=np.random.default_rng(531))
    ma = bundle.point_advantage
    for i, label in enumerate(ma.labels):
        lo = float(ma.spread_low[i])
        hi = float(ma.spread_high[i])
        assert lo <= hi, f"Template '{label}': spread_low={lo:.4f} > spread_high={hi:.4f}"
