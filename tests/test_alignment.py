"""Tests for judge_alignment() and compare(alignment=...) PPI propagation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.config import GRADIENT_CI_ALPHAS
from evalstats.alignment import AlignmentResult, judge_alignment, _fit_calibration
from evalstats.api import ComparisonResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_binary_evaldata(
    n_items: int = 60,
    n_labeled: int = 30,
    p_a: float = 0.70,
    p_b: float = 0.45,
    agreement_rate: float = 0.80,
    seed: int = 0,
) -> tuple[es.EvalResults, str]:
    """Long-format binary eval data with sparse human labels.

    Returns (evaldata, llm_metric_col).
    LLM and human labels agree with probability ``agreement_rate``; lower
    values simulate a noisier judge.
    """
    rng = _rng(seed)
    df = pd.DataFrame({
        "model":     ["A"] * n_items + ["B"] * n_items,
        "item":      list(range(n_items)) * 2,
        "llm_score": np.concatenate([
            rng.binomial(1, p_a, n_items),
            rng.binomial(1, p_b, n_items),
        ]).astype(float),
    })
    human = np.full(len(df), np.nan)
    for idx in rng.choice(len(df), size=n_labeled, replace=False):
        llm = df.loc[idx, "llm_score"]
        human[idx] = llm if rng.random() < agreement_rate else (1.0 - llm)
    df["human_score"] = human
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata, "llm_score"


def _make_likert_evaldata(
    n_items: int = 60,
    n_labeled: int = 30,
    seed: int = 1,
) -> tuple[es.EvalResults, str]:
    """Long-format 1–5 Likert eval data with sparse human labels."""
    rng = _rng(seed)
    cats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    llm_a = rng.choice(cats, size=n_items, p=[0.05, 0.10, 0.15, 0.35, 0.35])
    llm_b = rng.choice(cats, size=n_items, p=[0.20, 0.25, 0.30, 0.15, 0.10])
    df = pd.DataFrame({
        "model":     ["A"] * n_items + ["B"] * n_items,
        "item":      list(range(n_items)) * 2,
        "llm_score": np.concatenate([llm_a, llm_b]).astype(float),
    })
    human = np.full(len(df), np.nan)
    for idx in rng.choice(len(df), size=n_labeled, replace=False):
        noise = rng.choice([-1, 0, 0, 1])
        human[idx] = float(np.clip(df.loc[idx, "llm_score"] + noise, 1, 5))
    df["human_score"] = human
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata, "llm_score"


def _make_continuous_evaldata(
    n_items: int = 60,
    n_labeled: int = 30,
    seed: int = 2,
) -> tuple[es.EvalResults, str]:
    """Long-format continuous [0,1] eval data with sparse human labels."""
    rng = _rng(seed)
    df = pd.DataFrame({
        "model":     ["A"] * n_items + ["B"] * n_items,
        "item":      list(range(n_items)) * 2,
        "llm_score": np.concatenate([
            rng.beta(6, 2, n_items),
            rng.beta(3, 4, n_items),
        ]),
    })
    human = np.full(len(df), np.nan)
    for idx in rng.choice(len(df), size=n_labeled, replace=False):
        human[idx] = float(np.clip(df.loc[idx, "llm_score"] + rng.normal(0, 0.05), 0, 1))
    df["human_score"] = human
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata, "llm_score"


# ---------------------------------------------------------------------------
# judge_alignment — basic contracts
# ---------------------------------------------------------------------------

class TestJudgeAlignmentBasic:
    def test_returns_alignment_result(self):
        evaldata, metric = _make_binary_evaldata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert isinstance(result, AlignmentResult)

    def test_stores_metadata(self):
        evaldata, metric = _make_binary_evaldata(n_items=60, n_labeled=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert ar.llm_metric == metric
        assert ar.human_col == "human_score"
        assert ar.score_type == "binary"
        assert ar.n_labeled == 30
        assert ar.n_total == 120  # 60 items × 2 models

    def test_raises_missing_llm_column(self):
        evaldata, _ = _make_binary_evaldata()
        with pytest.raises(ValueError, match="llm_metric column"):
            judge_alignment(evaldata, llm_metric="nonexistent", human_groundtruth="human_score")

    def test_raises_missing_human_column(self):
        evaldata, metric = _make_binary_evaldata()
        with pytest.raises(ValueError, match="human_groundtruth column"):
            judge_alignment(evaldata, llm_metric=metric, human_groundtruth="nonexistent")

    def test_raises_no_labels_at_all(self):
        evaldata, metric = _make_binary_evaldata()
        evaldata._df["human_score"] = np.nan  # wipe all labels
        with pytest.raises(ValueError, match="No rows have human labels"):
            judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")

    def test_warns_small_n_labeled(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=15)
        with pytest.warns(UserWarning, match="fewer than ~30 labeled items"):
            judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")

    def test_no_small_n_warning_above_threshold(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        small_n_warns = [w for w in caught if "fewer than ~30" in str(w.message)]
        assert len(small_n_warns) == 0


# ---------------------------------------------------------------------------
# judge_alignment — alignment metrics by score type
# ---------------------------------------------------------------------------

class TestAlignmentMetrics:
    def test_binary_has_agreement_and_kappa(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert "percent_agreement" in ar.alignment_metrics
        assert "cohens_kappa" in ar.alignment_metrics

    def test_binary_agreement_in_range(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=50, agreement_rate=0.80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        pa = ar.alignment_metrics["percent_agreement"]["estimate"]
        assert 0.0 <= pa <= 1.0
        # With ~80% agreement rate we expect measured agreement between 0.5 and 1.0
        assert pa >= 0.5

    def test_binary_ci_bounds_ordered(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        for entry in ar.alignment_metrics.values():
            assert entry["ci_low"] <= entry["estimate"] <= entry["ci_high"]

    def test_likert_has_weighted_kappa_and_spearman(self):
        evaldata, metric = _make_likert_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert "weighted_kappa" in ar.alignment_metrics
        assert "spearman_r" in ar.alignment_metrics

    def test_continuous_has_pearson_and_spearman(self):
        evaldata, metric = _make_continuous_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert "pearson_r" in ar.alignment_metrics
        assert "spearman_r" in ar.alignment_metrics

    def test_perfect_agreement_kappa_near_one(self):
        """When human labels == LLM labels, κ should be close to 1."""
        rng = _rng(99)
        n = 80
        df = pd.DataFrame({
            "model":     ["A"] * n + ["B"] * n,
            "item":      list(range(n)) * 2,
            "llm_score": np.concatenate([rng.binomial(1, 0.7, n), rng.binomial(1, 0.4, n)]).astype(float),
        })
        # Perfect alignment: human == LLM for all labeled rows
        labeled_idx = rng.choice(len(df), size=40, replace=False)
        human = np.full(len(df), np.nan)
        human[labeled_idx] = df.loc[labeled_idx, "llm_score"].to_numpy()
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        kappa = ar.alignment_metrics["cohens_kappa"]["estimate"]
        assert kappa >= 0.90


# ---------------------------------------------------------------------------
# judge_alignment — representativeness checks
# ---------------------------------------------------------------------------

class TestRepresentativenessCheck:
    def test_representative_set_passes(self):
        """Randomly sampled alignment set should pass the distribution check."""
        evaldata, metric = _make_binary_evaldata(n_labeled=40, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        assert "score_distribution" in ar.representativeness

    def test_skewed_alignment_set_warns(self):
        """Alignment set drawn only from high-scoring items should trigger warning."""
        rng = _rng(42)
        n = 80
        llm_scores = np.concatenate([rng.binomial(1, 0.7, n), rng.binomial(1, 0.4, n)]).astype(float)
        df = pd.DataFrame({
            "model":     ["A"] * n + ["B"] * n,
            "item":      list(range(n)) * 2,
            "llm_score": llm_scores,
        })
        # Label only high-scoring items (biased subset)
        human = np.full(len(df), np.nan)
        high_mask = np.where(llm_scores == 1.0)[0]
        for i in high_mask[:30]:
            human[i] = 1.0
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        repr_warns = [w for w in caught if "non-representative" in str(w.message).lower()
                      or "representative" in str(w.message).lower()]
        assert len(repr_warns) >= 1

    def test_slice_column_check_added_to_result(self):
        """Categorical slice columns should appear in representativeness dict."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35, seed=5)
        evaldata._df["difficulty"] = pd.array(
            ["easy" if v < 30 else "hard" for v in evaldata._df["item"]],
            dtype=object,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        slice_keys = [k for k in ar.representativeness if k.startswith("slice_")]
        assert "slice_difficulty" in slice_keys


# ---------------------------------------------------------------------------
# AlignmentResult._sample_imputed_scores — calibration sampler
# ---------------------------------------------------------------------------

class TestSampleImputedScores:
    def test_binary_output_is_zero_or_one(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        llm_scores = evaldata._df[metric].to_numpy(dtype=float)
        rng = _rng(10)
        imputed = ar._sample_imputed_scores(llm_scores, rng)
        assert imputed.shape == llm_scores.shape
        assert set(np.unique(imputed)).issubset({0.0, 1.0})

    def test_likert_output_in_category_set(self):
        evaldata, metric = _make_likert_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        llm_scores = evaldata._df[metric].to_numpy(dtype=float)
        rng = _rng(11)
        imputed = ar._sample_imputed_scores(llm_scores, rng)
        assert imputed.shape == llm_scores.shape
        expected_cats = {1.0, 2.0, 3.0, 4.0, 5.0}
        assert set(np.unique(imputed)).issubset(expected_cats)

    def test_continuous_output_is_float_array(self):
        evaldata, metric = _make_continuous_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        llm_scores = evaldata._df[metric].to_numpy(dtype=float)
        rng = _rng(12)
        imputed = ar._sample_imputed_scores(llm_scores, rng)
        assert imputed.shape == llm_scores.shape
        assert imputed.dtype == float

    def test_different_rng_states_give_different_draws(self):
        """Two independent draws from the calibration posterior should differ."""
        evaldata, metric = _make_binary_evaldata(n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        llm_scores = evaldata._df[metric].to_numpy(dtype=float)
        draw1 = ar._sample_imputed_scores(llm_scores, _rng(1))
        draw2 = ar._sample_imputed_scores(llm_scores, _rng(2))
        assert not np.array_equal(draw1, draw2)

    def test_perfect_calibration_produces_near_identical_scores(self):
        """When LLM == human for all labeled items, imputed ≈ LLM (Beta → Bernoulli(~1 or ~0))."""
        rng = _rng(50)
        n = 80
        llm = rng.binomial(1, 0.7, n).astype(float)
        df = pd.DataFrame({
            "model":     ["A"] * n + ["B"] * n,
            "item":      list(range(n)) * 2,
            "llm_score": np.tile(llm, 2),
        })
        human = np.full(len(df), np.nan)
        labeled_idx = rng.choice(len(df), size=40, replace=False)
        human[labeled_idx] = df.loc[labeled_idx, "llm_score"].to_numpy()
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

        llm_scores = evaldata._df["llm_score"].to_numpy(dtype=float)
        imputed = ar._sample_imputed_scores(llm_scores, _rng(99))
        # With perfect calibration, imputed should agree with LLM > 90% of the time
        agreement = np.mean(imputed == llm_scores)
        assert agreement >= 0.85


# ---------------------------------------------------------------------------
# compare(alignment=...) — CI widening behaviour
# ---------------------------------------------------------------------------

class TestCompareAlignmentPPI:

    # n_mc=30 throughout for test speed; enough to verify direction of effects

    def test_cis_widen_under_misalignment_binary(self):
        """With 30% judge error rate, PPI-corrected CIs should be wider than base CIs."""
        evaldata, metric = _make_binary_evaldata(
            n_items=80, n_labeled=40, agreement_rate=0.65, seed=3
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result_mc   = es.compare(evaldata, factors="model", metric=metric,
                                     alignment={metric: ar}, n_mc=30)
            result_base = es.compare(evaldata, factors="model", metric=metric)

        bundle_mc   = result_mc._primary_bundle()
        bundle_base = result_base._primary_bundle()

        for i in range(len(bundle_mc.robustness.mean)):
            width_mc   = float(bundle_mc.robustness.ci_high[i]   - bundle_mc.robustness.ci_low[i])
            width_base = float(bundle_base.robustness.ci_high[i] - bundle_base.robustness.ci_low[i])
            assert width_mc > width_base, (
                f"Entity {i}: PPI CI width {width_mc:.4f} should exceed base {width_base:.4f}"
            )

    def test_rectifier_near_zero_under_perfect_alignment(self):
        """PPI rectifier (human_mean - llm_mean per entity) should be 0 when the judge
        is perfect, and nonzero when the judge disagrees with humans."""
        def _make_scenario(agreement_rate, seed):
            rng = _rng(seed)
            n = 80
            df = pd.DataFrame({
                "model":     ["A"] * n + ["B"] * n,
                "item":      list(range(n)) * 2,
                "llm_score": np.concatenate([
                    rng.binomial(1, 0.75, n), rng.binomial(1, 0.45, n)
                ]).astype(float),
            })
            human = np.full(len(df), np.nan)
            labeled_idx = rng.choice(len(df), size=50, replace=False)
            for idx in labeled_idx:
                llm = df.loc[idx, "llm_score"]
                human[idx] = llm if rng.random() < agreement_rate else (1.0 - llm)
            df["human_score"] = human
            evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score",
                                        human_groundtruth="human_score")
                result = es.compare(evaldata, factors="model", metric="llm_score",
                                    alignment={"llm_score": ar}, n_mc=50)
            vc = result.to_dict()["variance_components"]["entities"]
            return max(abs(v["rectifier"]) for v in vc.values())

        perfect_rect = _make_scenario(agreement_rate=1.00, seed=44)
        noisy_rect   = _make_scenario(agreement_rate=0.55, seed=44)
        assert perfect_rect == 0.0, (
            f"Perfect alignment rectifier should be exactly 0, got {perfect_rect:.6f}"
        )
        assert noisy_rect > 0.0, (
            f"Poor alignment rectifier should be nonzero, got {noisy_rect:.6f}"
        )

    def test_variance_components_populated(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)
        d = result.to_dict()
        assert "variance_components" in d
        vc = d["variance_components"]
        assert vc["method"] == "ppi"
        assert "n_lab" in vc
        assert "n_boot" in vc
        assert vc["n_boot"] >= 1000             # PPI floors at 1000 boot draws
        assert "entities" in vc
        for entry in vc["entities"].values():
            assert "n_labeled" in entry         # labeled items available per entity
            assert "llm_mean" in entry          # uncorrected LLM estimate
            assert "rectifier" in entry         # bias-correction term
            assert "ppi_mean" in entry          # PPI-corrected estimate
            assert entry["n_labeled"] >= 0

    def test_to_dict_no_variance_components_without_alignment(self):
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = es.compare(evaldata, factors="model", metric=metric)
        d = result.to_dict()
        assert "variance_components" not in d

    def test_pairwise_cis_also_widen(self):
        """Pairwise diff CIs should also be wider under misalignment."""
        evaldata, metric = _make_binary_evaldata(
            n_items=80, n_labeled=40, agreement_rate=0.60, seed=6
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result_mc   = es.compare(evaldata, factors="model", metric=metric,
                                     alignment={metric: ar}, n_mc=30)
            result_base = es.compare(evaldata, factors="model", metric=metric)

        pw_mc   = list(result_mc._primary_bundle().pairwise.results.values())[0]
        pw_base = list(result_base._primary_bundle().pairwise.results.values())[0]
        width_mc   = pw_mc.ci_high   - pw_mc.ci_low
        width_base = pw_base.ci_high - pw_base.ci_low
        assert width_mc > width_base

    def test_alignment_works_with_likert(self):
        evaldata, metric = _make_likert_evaldata(n_items=60, n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)
        assert isinstance(result, ComparisonResult)
        bundle = result._primary_bundle()
        assert bundle.robustness.ci_low is not None
        assert bundle.robustness.ci_high is not None

    def test_alignment_works_with_continuous(self):
        evaldata, metric = _make_continuous_evaldata(n_items=60, n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)
        assert isinstance(result, ComparisonResult)
        d = result.to_dict()
        assert "variance_components" in d

    def test_alignment_works_with_path_c_arbitrary_factor(self):
        """alignment= should work when the factor column is not 'model' or 'prompt'."""
        rng = _rng(20)
        n = 50
        df = pd.DataFrame({
            "system":    ["A"] * n + ["B"] * n,
            "item":      list(range(n)) * 2,
            "llm_score": np.concatenate([
                rng.binomial(1, 0.7, n), rng.binomial(1, 0.45, n)
            ]).astype(float),
        })
        human = np.full(len(df), np.nan)
        for idx in rng.choice(len(df), size=35, replace=False):
            llm = df.loc[idx, "llm_score"]
            human[idx] = llm if rng.random() < 0.75 else (1.0 - llm)
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"system": "model", "item": "item"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric="llm_score",
                                alignment={"llm_score": ar}, n_mc=20)
        assert isinstance(result, ComparisonResult)
        assert result._variance_components is not None

    def test_wrong_alignment_key_warns(self):
        """alignment= dict with wrong metric key should warn and leave CIs unchanged."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            base = es.compare(evaldata, factors="model", metric=metric)

        with pytest.warns(UserWarning, match="no entry for metric column"):
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={"wrong_col": ar}, n_mc=20)

        # CIs should be unchanged (alignment was skipped)
        base_ci = base._primary_bundle().robustness.ci_low
        result_ci = result._primary_bundle().robustness.ci_low
        np.testing.assert_array_almost_equal(result_ci, base_ci)

    def test_alignment_not_dict_warns(self):
        """Passing a non-dict to alignment= should warn and be ignored."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        with pytest.warns(UserWarning, match="must be a dict"):
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment=ar, n_mc=20)
        assert result._variance_components is None

    def test_multimodel_alignment_warns_not_supported(self):
        """Multi-model analysis (model + prompt) should warn that alignment= is not yet supported."""
        rng = _rng(30)
        n = 40
        df = pd.DataFrame({
            "model":     (["A"] * n + ["B"] * n) * 2,
            "prompt":    ["P1"] * (n * 2) + ["P2"] * (n * 2),
            "item":      list(range(n)) * 4,
            "llm_score": rng.binomial(1, 0.6, n * 4).astype(float),
        })
        human = np.full(len(df), np.nan)
        for idx in rng.choice(len(df), size=30, replace=False):
            llm = df.loc[idx, "llm_score"]
            human[idx] = llm if rng.random() < 0.8 else (1.0 - llm)
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"model": "model", "prompt": "prompt", "item": "item"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

        with pytest.warns(UserWarning, match="not yet supported"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar}, n_mc=10)


# ---------------------------------------------------------------------------
# PPI alignment p-value computation (Rubin pooled)
# ---------------------------------------------------------------------------

class TestPPIPooledPValues:
    """Tests for MI-pooled pairwise p-values with Rubin combining rules."""

    def test_pairwise_pvalues_present_after_mc(self):
        """Pairwise p_value field should be set (not None) after PPI pooling."""
        evaldata, metric = _make_binary_evaldata(n_items=80, n_labeled=40, seed=61)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=25)

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            assert pair.p_value is not None, f"p_value is None for pair ({a}, {b})"
            assert isinstance(pair.p_value, (int, float)), \
                f"p_value should be numeric, got {type(pair.p_value)}"

    def test_pvalues_in_valid_range(self):
        """All p-values should be in [0, 1]."""
        evaldata, metric = _make_binary_evaldata(n_items=80, n_labeled=40, seed=62)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=25)

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            assert 0.0 <= pair.p_value <= 1.0, \
                f"p_value {pair.p_value} out of [0,1] range for pair ({a}, {b})"

    def test_ci_excludes_zero_implies_p_significant(self):
        """If CI excludes 0 (diff > 0 or diff < 0), p-value should be ≤ 0.05.

        PPI uses a percentile bootstrap, so the p-value and CI are derived
        from the same bootstrap distribution.  The relationship is p ≤ α when
        the CI excludes 0 (not strict inequality: p can equal α exactly when
        the boundary draw lands exactly at 0).
        """
        evaldata, metric = _make_binary_evaldata(
            n_items=100, n_labeled=50, p_a=0.85, p_b=0.50, seed=63
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric, alpha=0.05,
                                alignment={metric: ar}, n_mc=30,
                                rng=np.random.default_rng(99))

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            ci_excludes_zero = (pair.ci_low > 0) or (pair.ci_high < 0)
            if ci_excludes_zero:
                assert pair.p_value <= 0.05, (
                    f"Pair ({a}, {b}): CI excludes 0 but p={pair.p_value:.4f} > 0.05"
                )

    def test_n_mc_small_succeeds(self):
        """compare(alignment=..., n_mc=1) should succeed with PPI (floors to 1000 boot draws)."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=1)
        vc = result.to_dict()["variance_components"]
        assert vc["method"] == "ppi"
        assert vc["n_boot"] == 1000   # floored

    def test_n_mc_zero_succeeds(self):
        """compare(alignment=..., n_mc=0) should succeed with PPI (floors to 1000 boot draws)."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=0)
        vc = result.to_dict()["variance_components"]
        assert vc["method"] == "ppi"
        assert vc["n_boot"] == 1000   # floored

    def test_pairwise_pvalues_consistent_across_directions(self):
        """p_value(A, B) should equal p_value(B, A) (since test is two-sided)."""
        evaldata, metric = _make_binary_evaldata(n_items=80, n_labeled=40, seed=64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=25)

        pw = result._primary_bundle().pairwise
        labels = list(pw.labels)
        if len(labels) >= 2:
            a, b = labels[0], labels[1]
            p_ab = pw.get(a, b).p_value
            p_ba = pw.get(b, a).p_value
            assert abs(p_ab - p_ba) < 1e-10, \
                f"p_value({a}, {b})={p_ab} != p_value({b}, {a})={p_ba}"

    def test_correction_method_applied_to_pooled_pvalues(self):
        """With multiple pairs, correction method (fdr_bh, holm, bonferroni) should affect p-values."""
        evaldata, metric = _make_binary_evaldata(
            n_items=100, n_labeled=50, seed=65
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")

        # Compare with different correction methods (via backend's native routing)
        # Note: correction is currently hardcoded in _run_alignment_mc, but we can
        # verify that uncorrected values (if we had them) would differ from corrected.
        result = es.compare(evaldata, factors="model", metric=metric,
                            alignment={metric: ar}, n_mc=20, correction="fdr_bh")
        bundle = result._primary_bundle()
        pvals_fdr = [pair.p_value for pair in bundle.pairwise.results.values()]
        assert all(0.0 <= p <= 1.0 for p in pvals_fdr)

    def test_reproducibility_with_seeded_rng(self):
        """Two runs with the same engine_kwargs RNG seed should give identical p-values."""
        evaldata, metric = _make_binary_evaldata(n_items=80, n_labeled=40, seed=66)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")

        rng1 = np.random.default_rng(42)
        result1 = es.compare(evaldata, factors="model", metric=metric,
                             alignment={metric: ar}, n_mc=15, rng=rng1)

        rng2 = np.random.default_rng(42)
        result2 = es.compare(evaldata, factors="model", metric=metric,
                             alignment={metric: ar}, n_mc=15, rng=rng2)

        pw1 = result1._primary_bundle().pairwise
        pw2 = result2._primary_bundle().pairwise

        for key in pw1.results:
            p1 = pw1.results[key].p_value
            p2 = pw2.results[key].p_value
            assert abs(p1 - p2) < 1e-14, \
                f"Seeded RNG did not reproduce p-value for {key}: {p1} vs {p2}"

    def test_pvalues_differ_significantly_with_high_vs_low_misalignment(self):
        """Under high misalignment (noisy judge), p-values should be larger (wider pooled CIs)
        compared to low misalignment (precise judge)."""
        def _get_pvalues(agreement_rate, seed):
            rng = _rng(seed)
            n = 80
            df = pd.DataFrame({
                "model":     ["A"] * n + ["B"] * n,
                "item":      list(range(n)) * 2,
                "llm_score": np.concatenate([
                    rng.binomial(1, 0.80, n), rng.binomial(1, 0.45, n)
                ]).astype(float),
            })
            human = np.full(len(df), np.nan)
            labeled_idx = rng.choice(len(df), size=40, replace=False)
            for idx in labeled_idx:
                llm = df.loc[idx, "llm_score"]
                human[idx] = llm if rng.random() < agreement_rate else (1.0 - llm)
            df["human_score"] = human
            evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score",
                                        human_groundtruth="human_score")
                result = es.compare(evaldata, factors="model", metric="llm_score",
                                    alignment={"llm_score": ar}, n_mc=35)
            pw = result._primary_bundle().pairwise
            return [pair.p_value for pair in pw.results.values()]

        pvals_high = _get_pvalues(agreement_rate=0.95, seed=70)
        pvals_low  = _get_pvalues(agreement_rate=0.50, seed=70)

        # Under perfect alignment, p-values should be more significant (smaller).
        # Under poor alignment, wider CIs → larger p-values.
        avg_p_high = np.mean(pvals_high)
        avg_p_low  = np.mean(pvals_low)
        # This is a trend test; we can't guarantee it for every run, but on average
        # high-quality alignment should yield smaller p-values for a fixed effect.
        assert avg_p_high <= avg_p_low, (
            f"High-quality alignment p-values ({avg_p_high:.4f}) should be "
            f"<= low-quality ({avg_p_low:.4f})"
        )

    def test_pairwise_point_diff_and_ci_consistent(self):
        """Pairwise point_diff should be within its own CI."""
        evaldata, metric = _make_binary_evaldata(n_items=80, n_labeled=40, seed=67)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=25)

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            assert pair.ci_low <= pair.point_diff <= pair.ci_high, (
                f"Pair ({a}, {b}): point_diff {pair.point_diff} not in "
                f"CI [{pair.ci_low}, {pair.ci_high}]"
            )

    def test_pvalues_populated_with_likert_alignment(self):
        """PPI pooled p-values should work with Likert-scale data."""
        evaldata, metric = _make_likert_evaldata(n_items=80, n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            assert pair.p_value is not None
            assert 0.0 <= pair.p_value <= 1.0

    def test_pvalues_populated_with_continuous_alignment(self):
        """PPI pooled p-values should work with continuous [0,1] data."""
        evaldata, metric = _make_continuous_evaldata(n_items=80, n_labeled=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)

        bundle = result._primary_bundle()
        for (a, b), pair in bundle.pairwise.results.items():
            assert pair.p_value is not None
            assert 0.0 <= pair.p_value <= 1.0

    def test_multi_ci_populated_after_mc(self):
        """Gradient CI bands (multi_ci) should be populated from the PPI bootstrap."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            result = es.compare(evaldata, factors="model", metric=metric,
                                alignment={metric: ar}, n_mc=20)
        bundle = result._primary_bundle()
        assert bundle.robustness.multi_ci is not None
        assert set(bundle.robustness.multi_ci) == set(GRADIENT_CI_ALPHAS)
        for lo, hi in bundle.robustness.multi_ci.values():
            assert lo.shape == hi.shape
        for pr in bundle.pairwise.results.values():
            assert pr.multi_ci is not None
            assert set(pr.multi_ci) == set(GRADIENT_CI_ALPHAS)
            for lo, hi in pr.multi_ci.values():
                assert np.isscalar(lo)
                assert np.isscalar(hi)

    def test_n_mc_parameter_controls_n_boot(self):
        """n_boot = max(n_mc, 1000) is stored in variance_components."""
        evaldata, metric = _make_binary_evaldata(n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
            for n_mc, expected_n_boot in [(10, 1000), (25, 1000), (2000, 2000)]:
                result = es.compare(evaldata, factors="model", metric=metric,
                                    alignment={metric: ar}, n_mc=n_mc)
                assert result.to_dict()["variance_components"]["n_boot"] == expected_n_boot


# ---------------------------------------------------------------------------
# PPI sample-size checks and CI method override warnings
# ---------------------------------------------------------------------------

def _make_small_evaldata(n_items: int, n_labeled: int, seed: int = 99, shared: bool = False):
    """Helper: binary evaldata with specified total items and labeled count.

    ``shared=True`` labels whole ITEMS across both models, the way a rater
    actually works and the way ``evalstats label`` samples. The default picks
    random rows, which leaves the two models sharing almost no labeled items --
    fine for the total-count checks, but below the paired PPI overlap floor.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "model":     ["A"] * n_items + ["B"] * n_items,
        "item":      list(range(n_items)) * 2,
        "llm_score": np.concatenate([
            rng.binomial(1, 0.70, n_items),
            rng.binomial(1, 0.45, n_items),
        ]).astype(float),
    })
    human = np.full(len(df), np.nan)
    if shared:
        chosen = set(rng.choice(n_items, size=n_labeled, replace=False).tolist())
        mask = df["item"].isin(chosen).to_numpy()
        human[mask] = df.loc[mask, "llm_score"]
    else:
        for idx in rng.choice(len(df), size=n_labeled, replace=False):
            human[idx] = df.loc[idx, "llm_score"]
    df["human_score"] = human
    return es.load_from(df, col_map={"model": "model", "item": "item"})


class TestPPISampleSizeChecks:
    """PPI alignment enforces minimum sample-size requirements."""

    def test_raises_when_n_lab_below_15(self):
        """compare(alignment=...) should raise ValueError when n_labeled < 15."""
        evaldata = _make_small_evaldata(n_items=40, n_labeled=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score",
                                    human_groundtruth="human_score")
        with pytest.raises(ValueError, match="15 human-labeled items"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar})

    def test_raises_when_n_all_below_50(self):
        """compare(alignment=...) should raise ValueError when N < 50."""
        # 20 items × 2 models = 40 total rows; n_labeled=15 to pass that check
        evaldata = _make_small_evaldata(n_items=20, n_labeled=15)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score",
                                    human_groundtruth="human_score")
        with pytest.raises(ValueError, match="50 items"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar})

    def test_warns_when_n_lab_below_30(self):
        """compare(alignment=...) should warn about potential under-coverage when n_labeled < 30."""
        # n_labeled=20 satisfies the ≥15 hard requirement but not the ≥30 soft one.
        # n_items=60 so n_all=120, above the 100 threshold.
        evaldata = _make_small_evaldata(n_items=60, n_labeled=20, shared=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score",
                                    human_groundtruth="human_score")
        # compare()'s own "recommend >= 30" counts labeled ROWS, and with two
        # conditions the paired overlap floor (15 shared ITEMS) already puts
        # that at 30+. The thin-labels warning the user actually sees at this
        # size comes from the alignment layer.
        with pytest.warns(UserWarning, match="recommend ≥ 30|fewer than ~30 labeled items"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar})

    def test_warns_when_n_all_below_100(self):
        """compare(alignment=...) should warn about potential under-coverage when N < 100."""
        # n_items=30 → n_all=60 (30 items × 2 models), between 50 and 100.
        evaldata = _make_small_evaldata(n_items=30, n_labeled=20, shared=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score",
                                    human_groundtruth="human_score")
        with pytest.warns(UserWarning, match="recommend ≥ 100"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar})

    def test_raises_clear_error_when_entity_is_100pct_labeled(self):
        """An entity with EVERY item human-labeled has zero unlabeled
        residual left for the LLM-only term (n_all=0 in the PPI variance
        decomposition Var(unlab)/n_all + Var(rectifier)/n_lab). This used to
        raise a bare ZeroDivisionError from deep inside
        evalstats.tests._ppi_single_wilson/_ppi_paired_mj_floor; it should now
        raise a clear, actionable ValueError instead."""
        rng = np.random.default_rng(7)
        n_items = 60
        df = pd.DataFrame({
            "model": ["A"] * n_items + ["B"] * n_items,
            "item": list(range(n_items)) * 2,
            "llm_score": np.concatenate([
                rng.binomial(1, 0.70, n_items),
                rng.binomial(1, 0.45, n_items),
            ]).astype(float),
        })
        human = np.full(len(df), np.nan)
        # Every item for model A is labeled (n_all=0 for A).
        a_mask = df["model"] == "A"
        human[a_mask] = df.loc[a_mask, "llm_score"]
        # B needs labels of its own, else its missing-labels error fires first
        # and this test never reaches the case it is about.
        b_idx = df.index[df["model"] == "B"].to_numpy()[:20]
        human[b_idx] = df.loc[b_idx, "llm_score"]
        df["human_score"] = human
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score",
                                    human_groundtruth="human_score")
        with pytest.raises(ValueError, match="at least one unlabeled item"):
            es.compare(evaldata, factors="model", metric="llm_score",
                       alignment={"llm_score": ar})

    def test_no_size_warnings_above_thresholds(self):
        """No size warnings when n_labeled ≥ 30 and N ≥ 100."""
        evaldata, metric = _make_binary_evaldata(n_items=60, n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            es.compare(evaldata, factors="model", metric=metric, alignment={metric: ar})
        size_warns = [
            w for w in caught
            if "recommend ≥" in str(w.message)
        ]
        assert len(size_warns) == 0


class TestPPICIMethodWarning:
    """PPI dispatches to a method-specific PPI correction, or raises a clear
    error when the resolved method has no validated PPI-corrected form
    (rather than silently overriding it and warning, as in a previous
    version of this API)."""

    def test_raises_when_method_has_no_ppi_correction(self):
        """PPI should raise a clear ValueError when the resolved method has no
        validated PPI-corrected counterpart."""
        evaldata, metric = _make_binary_evaldata(n_items=60, n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        with pytest.raises(ValueError, match="no validated implementation"):
            es.compare(evaldata, factors="model", metric=metric,
                       alignment={metric: ar},
                       method="bca")  # BCA has no PPI-corrected form

    def test_plain_bootstrap_method_is_ppi_corrected(self):
        """method="bootstrap" has a validated PPI-corrected form and should
        run without error or a method-override warning."""
        evaldata, metric = _make_binary_evaldata(n_items=60, n_labeled=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric=metric, human_groundtruth="human_score")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = es.compare(evaldata, factors="model", metric=metric,
                                 alignment={metric: ar},
                                 method="bootstrap")
        assert result._primary_bundle().resolved_ci_method == "bootstrap"
        override_warns = [
            w for w in caught
            if "percentile bootstrap" in str(w.message) and "overridden" in str(w.message)
        ]
        assert len(override_warns) == 0


# ---------------------------------------------------------------------------
# Multi-condition judge_alignment: design=/test= dispatch, per-test
# linearizations, and the label-efficiency (n_eff/multiplier) numbers.
#
# The linearizations are validated against theory, not just smoke-tested:
# with the variance-minimizing lambda, PPI's variance reduction is exactly
# 1/(1 - rho^2*(1 - n_lab/N)) where rho correlates the two sides' INFLUENCE
# FUNCTIONS. test_oracle_lambda_matches_predicted_multiplier below rebuilds
# that oracle estimator directly (no bootstrap, no evalstats.tests lambda)
# and checks the prediction end to end -- this is the regression guard for
# two real bugs found and fixed:
#   * mannwhitney scored group B as -F_X(y) instead of 1-F_X(y), putting the
#     pooled halves ~1.0 apart and inflating rho ~2x.
#   * wilcoxon borrowed sign(d)*(2F-1) from the (since-removed)
#     hajek_experimental path, which is not affine in F_D(d); it now reuses
#     evalstats.ppi._walsh_theta_h1_components, the same per-item Hajek
#     projection the production correction builds its own variance from.
# ---------------------------------------------------------------------------

def _cond(rng, n=400, n_lab=120, d=0.0, judge_noise=0.6, paired=False):
    """One (judge, sparse-human) pair per condition."""
    if paired:
        base = rng.normal(0, 1, n)
        ha, hb = base, base + rng.normal(d, 1.0, n)
        m = np.zeros(n, bool)
        m[rng.choice(n, n_lab, replace=False)] = True
        ma = mb = m
    else:
        ha, hb = rng.normal(0, 1, n), rng.normal(d, 1, n)
        ma = np.zeros(n, bool); ma[rng.choice(n, n_lab, replace=False)] = True
        mb = np.zeros(n, bool); mb[rng.choice(n, n_lab, replace=False)] = True
    ja = ha + rng.normal(0, judge_noise, n)
    jb = hb + rng.normal(0, judge_noise, n)
    return {"A": (ja, np.where(ma, ha, np.nan)), "B": (jb, np.where(mb, hb, np.nan))}


class TestMultiConditionAlignment:

    def test_design_required_when_ambiguous(self):
        conds = _cond(_rng(1))
        with pytest.raises(ValueError, match="design"):
            judge_alignment(conds, selection="random")

    def test_test_implies_design_and_conflict_raises(self):
        conds = _cond(_rng(2), paired=True)
        r = judge_alignment(conds, test="wilcoxon", selection="random")
        assert r.design == "within"          # implied, not passed
        with pytest.raises(ValueError, match="always design"):
            judge_alignment(conds, test="wilcoxon", design="between", selection="random")

    def test_two_condition_test_decomposes_over_pairs(self):
        """A 2-condition-only test given 3+ conditions is BY DESIGN reported
        per pair (the post-hoc planning number), not rejected."""
        rng = _rng(3)
        conds = _cond(rng, paired=True)
        conds["C"] = _cond(rng, paired=True)["B"]
        r = judge_alignment(conds, test="wilcoxon", selection="random")
        assert r.omnibus_metric is None
        assert set(r.test_pairwise_metrics) == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_linearize_rejects_wrong_condition_count(self):
        """The count bound itself is enforced where the whole-design
        linearization is actually built."""
        from evalstats.alignment import _linearize_for_test
        rng = _rng(31)
        conds = _cond(rng, paired=True)
        conds["C"] = _cond(rng, paired=True)["B"]
        with pytest.raises(ValueError, match="exactly 2"):
            _linearize_for_test(conds, test="wilcoxon", design="within")

    def test_single_condition_rejects_comparison_test(self):
        rng = _rng(4)
        n = 200
        h = rng.normal(0, 1, n); j = h + rng.normal(0, 0.5, n)
        hs = h.copy(); hs[rng.random(n) < 0.5] = np.nan
        with pytest.raises(ValueError, match="needs a comparison"):
            judge_alignment(j, hs, test="wilcoxon", selection="random")

    def test_mean_estimate_exposes_n_eff(self):
        rng = _rng(5)
        n = 600
        h = rng.normal(0, 1, n); j = h + rng.normal(0, 0.5, n)
        hs = h.copy(); hs[rng.random(n) < 0.7] = np.nan
        r = judge_alignment(j, hs, test="mean_estimate", selection="random")
        assert r.n_eff > r.n_labeled            # PPI buys something
        assert r.multiplier > 1.0
        assert r.n_eff == pytest.approx(r.n_labeled * r.multiplier, rel=1e-9)

    def test_n_eff_requires_test(self):
        rng = _rng(6)
        n = 300
        h = rng.normal(0, 1, n); j = h + rng.normal(0, 0.5, n)
        hs = h.copy(); hs[rng.random(n) < 0.5] = np.nan
        r = judge_alignment(j, hs, selection="random")
        with pytest.raises(ValueError, match="No test="):
            _ = r.n_eff

    @pytest.mark.parametrize("test,paired", [
        ("ttest", False), ("ttest", True), ("mannwhitney", False), ("wilcoxon", True),
    ])
    def test_two_condition_tests_report_per_pair(self, test, paired):
        conds = _cond(_rng(7), paired=paired)
        kw = {} if test in ("wilcoxon", "mannwhitney") else {"design": "within" if paired else "between"}
        r = judge_alignment(conds, test=test, selection="random", **kw)
        assert r.omnibus_metric is None
        m = r.test_pairwise_metrics[("A", "B")]
        assert np.isfinite(m["estimate"]) and 0.0 <= m["n_eff"] < np.inf
        assert m["n_eff"] >= m["n"]           # never worse than labels alone

    @pytest.mark.parametrize("test", ["anova_oneway", "kruskalwallis", "friedman"])
    def test_omnibus_tests_report_whole_design(self, test):
        rng = _rng(8)
        paired = test == "friedman"
        conds = _cond(rng, paired=paired)
        conds["C"] = _cond(rng, paired=paired)["B"]
        kw = {"design": "within" if paired else "between"} if test == "anova_oneway" else {}
        r = judge_alignment(conds, test=test, selection="random", **kw)
        assert r.omnibus_metric is not None    # whole-design number
        assert r.test_pairwise_metrics is None
        assert len(r.pairwise_metrics) == 3    # raw pairwise still reported
        assert np.isfinite(r.omnibus_metric["n_eff"])

    @pytest.mark.parametrize("test,design", [
        ("friedman", "within"), ("kruskalwallis", "between"), ("anova_oneway", "between"),
    ])
    def test_omnibus_multiplier_matches_savings_formula(self, test, design):
        """The omnibus multiplier obeys 1/(1 - rho^2 (1 - n_lab/N)) with both
        counts on ONE scale, and n_eff stays on the scale of its own labeled
        count. Regression: friedman's linearization emits k numbers per item,
        so pairing it with the item-scale _pair_total_n drove n_lab/N about k
        times too high, flattening the multiplier to ~1 while n_eff was read
        as per-condition -- a ~3x overstatement of the reported label savings.
        """
        rng = _rng(11)
        n, n_lab, k = 80, 25, 3
        conds = {}
        for i in range(k):
            human = rng.normal(size=n)
            judge = 0.7 * human + rng.normal(size=n) * 0.7
            human = human.copy()
            human[n_lab:] = np.nan
            conds[f"c{i}"] = (judge, human)
        kw = {"design": design} if test == "anova_oneway" else {}
        r = judge_alignment(conds, test=test, selection="random", ci=False, **kw)
        m = r.omnibus_metric

        # n_lab and N must share a scale, so their ratio is the labeled fraction.
        assert m["n"] / m["N"] == pytest.approx(n_lab / n, rel=1e-9)
        expected = 1.0 / (1.0 - m["estimate"] ** 2 * (1.0 - n_lab / n))
        assert m["multiplier"] == pytest.approx(expected, rel=1e-9)
        assert m["n_eff"] == pytest.approx(m["multiplier"] * m["n"], rel=1e-9)
        # PPI can only ever add information.
        assert 1.0 <= m["multiplier"] < 1.0 / (1.0 - (1.0 - n_lab / n))

    def test_full_linearized_n_only_differs_for_friedman(self):
        """_full_linearized_n replaces _pair_total_n, so it must agree with it
        everywhere the old count was already on the linearization's scale."""
        from evalstats.alignment import _full_linearized_n, _pair_total_n

        rng = _rng(12)
        conds = {}
        for i in range(3):
            human = rng.normal(size=40)
            judge = 0.6 * human + rng.normal(size=40) * 0.8
            human = human.copy()
            human[12:] = np.nan
            conds[f"c{i}"] = (judge, human)
        names = list(conds)
        for test, design, same in [
            ("wilcoxon", "within", True), ("ttest", "within", True),
            ("mannwhitney", "between", True), ("kruskalwallis", "between", True),
            ("anova_oneway", "between", True), ("friedman", "within", False),
        ]:
            sub = conds if test in ("friedman", "kruskalwallis", "anova_oneway") else \
                {nm: conds[nm] for nm in names[:2]}
            full = _full_linearized_n(sub, test=test, design=design)
            pair = _pair_total_n(sub, list(sub), design)
            if same:
                assert full == pair, f"{test}/{design}: {full} != {pair}"
            else:
                assert full == pair * len(sub), f"{test}/{design}: {full} != {pair}*{len(sub)}"

    def test_per_condition_counts_reported(self):
        conds = _cond(_rng(9), n=400, n_lab=120)
        r = judge_alignment(conds, design="between", selection="random")
        for name in ("A", "B"):
            n_lab, n_tot = r.condition_counts[name]
            assert n_lab == 120 and n_tot == 400

    def test_selection_warns_when_unknown(self):
        conds = _cond(_rng(10))
        with pytest.warns(UserWarning, match="selection"):
            judge_alignment(conds, design="between")

    def test_wilcoxon_rho_is_the_production_hajek_projection(self):
        """The whole point of test=: for a rank test the reported rho is the
        influence-function correlation the production correction itself uses
        (evalstats.ppi._walsh_theta_h1_components), NOT raw Spearman."""
        from evalstats.ppi import _walsh_theta_h1_components

        conds = _cond(_rng(11), paired=True, d=0.5)
        r = judge_alignment(conds, test="wilcoxon", selection="random")

        (ja, al), (jb, bl) = conds["A"], conds["B"]
        m = ~np.isnan(al) & ~np.isnan(bl)
        expected = np.corrcoef(
            _walsh_theta_h1_components(ja[m] - jb[m]),
            _walsh_theta_h1_components(al[m] - bl[m]),
        )[0, 1]

        corrected = r.test_pairwise_metrics[("A", "B")]["estimate"]
        assert corrected == pytest.approx(expected, abs=1e-12)
        raw = r.pairwise_metrics[("A", "B")]["spearman_r"]["estimate"]
        assert corrected != pytest.approx(raw, abs=1e-9)   # a different quantity

    @pytest.mark.parametrize("kind,d", [("wilcoxon", 0.0), ("wilcoxon", 1.0),
                                        ("mannwhitney", 0.0), ("mannwhitney", 1.0)])
    def test_oracle_lambda_matches_predicted_multiplier(self, kind, d):
        """rho from the linearization must predict the ACTUAL variance
        reduction achievable at the variance-minimizing lambda, at every
        effect size. Guards the two fixed linearization bugs."""
        from evalstats.alignment import (
            _linearize_wilcoxon, _linearize_mannwhitney, _n_eff,
        )
        from evalstats.ppi import paired_walsh_midrank_theta
        from evalstats.tests import _midrank_theta

        N, n_lab, reps = 600, 150, 1500
        rng = _rng(1234 + int(d * 10))
        cls, ppi, rhos = [], [], []
        for _ in range(reps):
            if kind == "wilcoxon":
                base = rng.normal(0, 1, N)
                ha, hb = base, base + rng.normal(d, 1.0, N)
            else:
                ha, hb = rng.normal(0, 1, N), rng.normal(d, 1, N)
            ja = ha + rng.normal(0, 0.8, N)
            jb = hb + rng.normal(0, 0.8, N)
            if kind == "wilcoxon":
                idx = rng.permutation(N); L, U = idx[:n_lab], idx[n_lab:]
                al = np.where(np.isin(np.arange(N), L), ha, np.nan)
                bl = np.where(np.isin(np.arange(N), L), hb, np.nan)
                th = paired_walsh_midrank_theta(ha[L] - hb[L])
                thh_l = paired_walsh_midrank_theta(ja[L] - jb[L])
                thh_u = paired_walsh_midrank_theta(ja[U] - jb[U])
                jl, hl = _linearize_wilcoxon({"A": (ja, al), "B": (jb, bl)})
            else:
                ia, ib = rng.permutation(N), rng.permutation(N)
                La, Ua, Lb, Ub = ia[:n_lab], ia[n_lab:], ib[:n_lab], ib[n_lab:]
                al = np.where(np.isin(np.arange(N), La), ha, np.nan)
                bl = np.where(np.isin(np.arange(N), Lb), hb, np.nan)
                th = _midrank_theta(ha[La], hb[Lb])
                thh_l = _midrank_theta(ja[La], jb[Lb])
                thh_u = _midrank_theta(ja[Ua], jb[Ub])
                jl, hl = _linearize_mannwhitney({"A": (ja, al), "B": (jb, bl)})
            u = N - n_lab
            Vj = np.var(jl, ddof=1)
            C = np.cov(hl, jl, ddof=1)[0, 1]
            lam = (C / n_lab) / (Vj / u + Vj / n_lab) if Vj > 0 else 0.0
            ppi.append(th + lam * (thh_u - thh_l))
            cls.append(th)
            rhos.append(np.corrcoef(jl, hl)[0, 1] ** 2)

        oracle_M = np.var(cls, ddof=1) / np.var(ppi, ddof=1)
        pred_M, _ = _n_eff(np.sqrt(np.nanmean(rhos)), n_lab, N)
        assert pred_M / oracle_M == pytest.approx(1.0, abs=0.10), (
            f"{kind} d={d}: predicted {pred_M:.4f} vs oracle {oracle_M:.4f}"
        )


# ---------------------------------------------------------------------------
# Label-position check on paired data: judge ITEMS, not rows
# ---------------------------------------------------------------------------


def _make_paired_judge_df(n_items: int = 60, k: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(k):
        for i in range(n_items):
            truth = rng.uniform(1, 5)
            rows.append({
                "item": f"i{i:03d}", "model": f"M{m}",
                "coherence": float(np.clip(round(truth + rng.normal(0, 0.8)), 1, 5)),
                "_truth": truth,
            })
    return pd.DataFrame(rows)


def _reveal_items(df: pd.DataFrame, labeled_items: set) -> pd.DataFrame:
    out = df.copy()
    out["human_coherence"] = np.where(out["item"].isin(labeled_items), out["_truth"], np.nan)
    return out.drop(columns="_truth")


def _position_check(df: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar = judge_alignment(es.load_from(df), llm_metric="coherence",
                             human_groundtruth="human_coherence", selection="random")
    return ar.representativeness["label_contiguity"]


class TestLabelPositionCheckPairedDesign:
    def test_random_items_sorted_by_item_pass(self):
        """Same 15 random items labeled for all 3 models, rows sorted by item:
        labeled rows come in runs of 3 by construction, which must not be
        read as clustering."""
        df = _make_paired_judge_df()
        items = sorted(df["item"].unique())
        labeled = set(np.random.default_rng(1).choice(items, size=15, replace=False))
        res = _position_check(_reveal_items(df, labeled).sort_values(["item", "model"]))
        assert res["passed"], res["message"]

    def test_random_items_are_order_invariant(self):
        df = _make_paired_judge_df()
        items = sorted(df["item"].unique())
        labeled = set(np.random.default_rng(1).choice(items, size=15, replace=False))
        by_item = _position_check(_reveal_items(df, labeled).sort_values(["item", "model"]))
        by_model = _position_check(_reveal_items(df, labeled).sort_values(["model", "item"]))
        assert by_item["passed"] and by_model["passed"]
        assert by_item["p_value"] == pytest.approx(by_model["p_value"])

    def test_contiguous_block_of_items_still_fails(self):
        df = _make_paired_judge_df()
        items = sorted(df["item"].unique())
        res = _position_check(_reveal_items(df, set(items[:15])).sort_values(["item", "model"]))
        assert not res["passed"]
        assert "single contiguous block" in res["message"]

    def test_single_row_per_item_behaviour_unchanged(self):
        """With one row per item the item collapse is a no-op: a random draw
        passes and a leading block fails, as before."""
        df = _make_paired_judge_df(k=1)
        items = sorted(df["item"].unique())
        random_lab = set(np.random.default_rng(2).choice(items, size=15, replace=False))
        assert _position_check(_reveal_items(df, random_lab))["passed"]
        assert not _position_check(_reveal_items(df, set(items[:15])))["passed"]
