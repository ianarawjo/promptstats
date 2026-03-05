import numpy as np
import pytest

import promptstats as ps
from promptstats.compare import CompareReport, EntityStats


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_compare_models_requires_dict():
    with pytest.raises(TypeError, match="dict"):
        ps.compare_models([np.ones((2, 10)), np.ones((2, 10))])


def test_compare_models_requires_at_least_two_models():
    with pytest.raises(ValueError, match="at least 2"):
        ps.compare_models({"only_one": np.ones((2, 10))})


def test_compare_models_flat_3d_raises():
    with pytest.raises(ValueError, match=r"Expected 1-D \(M inputs\) or 2-D \(M inputs, R runs\)"):
        ps.compare_models(
            {
                "m1": np.ones((2, 10, 3)),
                "m2": np.ones((2, 10, 3)),
            },
            rng=_rng(),
        )


def test_compare_models_mismatched_template_counts_raises():
    with pytest.raises(ValueError, match="same template keys"):
        ps.compare_models(
            {
                "m1": {"t1": np.ones(10), "t2": np.ones(10)},
                "m2": {"t1": np.ones(10), "t2": np.ones(10), "t3": np.ones(10)},
            },
            rng=_rng(),
        )


def test_compare_models_mismatched_run_counts_raises():
    with pytest.raises(ValueError, match="same number of runs"):
        ps.compare_models(
            {
                "m1": {
                    "t1": np.ones((10, 3)),
                    "t2": np.ones((10, 3)),
                },
                "m2": {
                    "t1": np.ones((10, 2)),
                    "t2": np.ones((10, 2)),
                },
            },
            rng=_rng(),
        )


def test_compare_models_template_labels_length_check():
    with pytest.raises(ValueError, match="template_labels length"):
        ps.compare_models(
            {
                "m1": {"t1": np.ones(10), "t2": np.ones(10)},
                "m2": {"t1": np.ones(10), "t2": np.ones(10)},
            },
            template_labels=["only_one"],
            rng=_rng(),
        )


def test_compare_models_returns_report_and_fields():
    report = ps.compare_models(
        {
            "model_a": {
                "t1": [0.80, 0.82, 0.78, 0.81, 0.79],
                "t2": [0.77, 0.79, 0.76, 0.78, 0.75],
            },
            "model_b": {
                "t1": [0.86, 0.88, 0.84, 0.87, 0.85],
                "t2": [0.83, 0.84, 0.82, 0.85, 0.81],
            },
        },
        n_bootstrap=400,
        rng=_rng(5),
    )

    assert isinstance(report, CompareReport)
    assert set(report.labels) == {"model_a", "model_b"}
    assert set(report.model_stats.keys()) == {"model_a", "model_b"}
    assert isinstance(report.model_stats["model_a"], EntityStats)
    assert set(report.pairwise_p_values.keys()) == {("model_a", "model_b")}
    assert isinstance(report.full_analysis, ps.MultiModelBundle)
    assert isinstance(report.quick_summary(), str)
    assert report.winner in (None, "model_a", "model_b")


def test_compare_models_detects_clear_winner():
    rng = _rng(11)
    m1 = rng.normal(loc=0.55, scale=0.05, size=120)
    m2 = rng.normal(loc=0.72, scale=0.05, size=120)

    report = ps.compare_models(
        {"weaker": m1, "stronger": m2},
        n_bootstrap=1200,
        rng=_rng(12),
    )

    assert report.winners == ["stronger"]
    assert report.significant is True
    assert report.p_best < 0.05


def test_compare_models_pairwise_lookup_direction_agnostic():
    report = ps.compare_models(
        {
            "m1": {
                "t1": [0.7, 0.72, 0.71, 0.69, 0.70],
                "t2": [0.65, 0.66, 0.64, 0.67, 0.66],
            },
            "m2": {
                "t1": [0.75, 0.76, 0.74, 0.77, 0.75],
                "t2": [0.70, 0.71, 0.72, 0.69, 0.70],
            },
        },
        n_bootstrap=400,
        rng=_rng(2),
    )

    p12 = report.get_pairwise_p_values("m1", "m2")
    p21 = report.get_pairwise_p_values("m2", "m1")
    assert p12["p_boot"] == pytest.approx(p21["p_boot"], abs=1e-12)


def test_compare_models_accepts_flat_1d_per_model_scores():
    report = ps.compare_models(
        {
            "m1": np.array([0.70, 0.72, 0.69, 0.71, 0.70]),
            "m2": np.array([0.76, 0.77, 0.75, 0.78, 0.76]),
        },
        n_bootstrap=300,
        rng=_rng(21),
    )

    assert isinstance(report, CompareReport)
    assert set(report.labels) == {"m1", "m2"}


def test_compare_models_accepts_flat_nested_arrays_for_runs():
    report = ps.compare_models(
        {
            "m1": np.array([
                [0.70, 0.71, 0.69],
                [0.72, 0.73, 0.71],
                [0.68, 0.69, 0.67],
                [0.71, 0.72, 0.70],
            ]),
            "m2": np.array([
                [0.76, 0.77, 0.75],
                [0.78, 0.79, 0.77],
                [0.74, 0.75, 0.73],
                [0.77, 0.78, 0.76],
            ]),
        },
        template_labels=["single_template"],
        n_bootstrap=300,
        rng=_rng(22),
    )

    assert isinstance(report, CompareReport)
    assert report.full_analysis.benchmark.template_labels == ["single_template"]


def test_compare_models_accepts_nested_template_dicts():
    report = ps.compare_models(
        {
            "m1": {
                "t_a": [0.70, 0.71, 0.69, 0.72],
                "t_b": [0.68, 0.69, 0.67, 0.70],
            },
            "m2": {
                "t_a": [0.76, 0.77, 0.75, 0.78],
                "t_b": [0.74, 0.75, 0.73, 0.76],
            },
        },
        n_bootstrap=300,
        rng=_rng(23),
    )

    assert isinstance(report, CompareReport)
    assert report.full_analysis.benchmark.template_labels == ["t_a", "t_b"]


def test_compare_models_accepts_nested_template_dicts_with_runs():
    report = ps.compare_models(
        {
            "m1": {
                "t_a": [[0.70, 0.71], [0.72, 0.73], [0.69, 0.70]],
                "t_b": [[0.68, 0.69], [0.70, 0.71], [0.67, 0.68]],
            },
            "m2": {
                "t_a": [[0.76, 0.77], [0.78, 0.79], [0.75, 0.76]],
                "t_b": [[0.74, 0.75], [0.76, 0.77], [0.73, 0.74]],
            },
        },
        n_bootstrap=300,
        rng=_rng(24),
    )

    assert isinstance(report, CompareReport)
    assert report.full_analysis.benchmark.template_labels == ["t_a", "t_b"]


def test_compare_models_nested_template_dict_keys_must_match():
    with pytest.raises(ValueError, match="same template keys"):
        ps.compare_models(
            {
                "m1": {
                    "t_a": [0.70, 0.71, 0.69, 0.72],
                    "t_b": [0.68, 0.69, 0.67, 0.70],
                },
                "m2": {
                    "t_a": [0.76, 0.77, 0.75, 0.78],
                    "t_c": [0.74, 0.75, 0.73, 0.76],
                },
            },
            rng=_rng(25),
        )
