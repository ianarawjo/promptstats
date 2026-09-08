"""Tests for evalstats.core.pareto and compare(secondary_metric=...) propagation."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.core.pareto import (
    ParetoBootstrapResult,
    pareto_bootstrap,
    classify_pareto_status,
    orient_higher_is_better,
)
from evalstats.core.summary import _pareto_sorted_labels


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# orient_higher_is_better
# ---------------------------------------------------------------------------

def test_orient_max_is_noop():
    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(orient_higher_is_better(a, "max"), a)


def test_orient_min_negates():
    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(orient_higher_is_better(a, "min"), -a)


def test_orient_bad_direction_raises():
    with pytest.raises(ValueError):
        orient_higher_is_better(np.array([1.0]), "lower")


# ---------------------------------------------------------------------------
# pareto_bootstrap: core engine
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pareto_bootstrap(
            np.zeros((2, 10)), np.zeros((2, 11)), ["a", "b"],
            n_bootstrap=100, rng=_rng(),
        )


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError):
        pareto_bootstrap(
            np.zeros((2, 10)), np.zeros((2, 10)), ["a", "b", "c"],
            n_bootstrap=100, rng=_rng(),
        )


def test_clear_dominance_detected():
    """A dominates B on both axes with a large, unambiguous margin."""
    rng = _rng(1)
    M = 200
    primary = np.stack([
        rng.normal(0.85, 0.05, M),  # A: high
        rng.normal(0.60, 0.05, M),  # B: low
    ])
    secondary = np.stack([
        rng.normal(0.80, 0.05, M),  # A: high (already oriented so higher=better)
        rng.normal(0.50, 0.05, M),  # B: low
    ])
    result = pareto_bootstrap(primary, secondary, ["A", "B"], n_bootstrap=3000, rng=_rng(2))
    assert result.p_frontier[0] > 0.99   # A almost always on frontier
    assert result.p_frontier[1] < 0.01   # B almost never
    assert result.p_dominated_by[1, 0] > 0.99  # A dominates B with high confidence
    assert result.p_dominated_by[0, 1] < 0.01

    statuses = classify_pareto_status(result)
    assert statuses["A"].status == "frontier"
    assert statuses["B"].status == "dominated"
    assert statuses["B"].dominated_by == ["A"]


def test_genuine_tradeoff_both_on_frontier():
    """A wins on primary, B wins on secondary -- neither dominates."""
    rng = _rng(3)
    M = 200
    primary = np.stack([
        rng.normal(0.85, 0.05, M),  # A: better primary
        rng.normal(0.60, 0.05, M),  # B: worse primary
    ])
    secondary = np.stack([
        rng.normal(0.30, 0.05, M),  # A: worse secondary
        rng.normal(0.80, 0.05, M),  # B: better secondary
    ])
    result = pareto_bootstrap(primary, secondary, ["A", "B"], n_bootstrap=3000, rng=_rng(4))
    statuses = classify_pareto_status(result)
    assert statuses["A"].status == "frontier"
    assert statuses["B"].status == "frontier"
    assert result.p_dominated_by[0, 1] < 0.05
    assert result.p_dominated_by[1, 0] < 0.05


def test_flat_rate_secondary_metric_does_not_crash():
    """A secondary metric with zero per-item variance (e.g. a flat $/token
    rate) shouldn't cause numerical issues -- unlike a covariance-matrix
    reconstruction approach, plain resampling degrades gracefully to a
    constant on that axis."""
    rng = _rng(5)
    M = 100
    primary = np.stack([
        rng.normal(0.80, 0.10, M),
        rng.normal(0.60, 0.10, M),
    ])
    secondary = np.stack([
        np.full(M, 2.0),
        np.full(M, 2.0),
    ])
    result = pareto_bootstrap(primary, secondary, ["A", "B"], n_bootstrap=2000, rng=_rng(6))
    assert np.all(np.isfinite(result.p_frontier))
    assert np.all(np.isfinite(result.p_dominated_by))
    statuses = classify_pareto_status(result)
    assert statuses["A"].status == "frontier"
    assert statuses["B"].status == "dominated"


def test_correlated_metrics_preserve_correlation():
    """When both metrics move together across items (harder items are both
    slower and less accurate), the joint bootstrap should reflect the
    resulting positive dominance relationship, not treat the axes as
    independent."""
    rng = _rng(7)
    M = 200
    item_difficulty = rng.normal(0, 1, M)  # shared across both metrics
    # A is uniformly better on both correlated axes.
    primary = np.stack([
        0.8 - 0.1 * item_difficulty + rng.normal(0, 0.05, M),
        0.6 - 0.1 * item_difficulty + rng.normal(0, 0.05, M),
    ])
    secondary = np.stack([
        0.7 - 0.1 * item_difficulty + rng.normal(0, 0.05, M),
        0.5 - 0.1 * item_difficulty + rng.normal(0, 0.05, M),
    ])
    result = pareto_bootstrap(primary, secondary, ["A", "B"], n_bootstrap=2000, rng=_rng(8))
    statuses = classify_pareto_status(result)
    assert statuses["A"].status == "frontier"
    assert statuses["B"].status == "dominated"


# ---------------------------------------------------------------------------
# classify_pareto_status: three-state logic (manual ParetoBootstrapResult)
# ---------------------------------------------------------------------------

def _manual_result(p_dominated_by, point_primary, point_secondary, labels=("A", "B", "C")):
    n = len(labels)
    return ParetoBootstrapResult(
        labels=list(labels),
        point_primary=np.array(point_primary),
        point_secondary=np.array(point_secondary),
        p_frontier=np.zeros(n),
        p_dominated_by=np.array(p_dominated_by),
        n_bootstrap=2000,
    )


def test_ambiguous_when_point_dominated_but_not_confident():
    # B's point estimate is dominated by A on both axes, but bootstrap
    # confidence (30%) doesn't clear the Bonferroni threshold.
    result = _manual_result(
        p_dominated_by=[
            [0.0, 0.0, 0.05],
            [0.30, 0.0, 0.02],
            [0.0, 0.0, 0.0],
        ],
        point_primary=[0.75, 0.70, 0.60],
        point_secondary=[0.50, 0.40, 0.65],
    )
    statuses = classify_pareto_status(result, alpha=0.05)
    assert statuses["B"].status == "ambiguous"
    assert statuses["B"].ambiguous_vs == ["A"]
    assert statuses["A"].status == "frontier"
    assert statuses["C"].status == "frontier"


def test_dominated_when_confident():
    result = _manual_result(
        p_dominated_by=[
            [0.0, 0.0, 0.05],
            [0.99, 0.0, 0.02],  # clears the ~0.975 Bonferroni threshold at N=3
            [0.0, 0.0, 0.0],
        ],
        point_primary=[0.75, 0.70, 0.60],
        point_secondary=[0.50, 0.40, 0.65],
    )
    statuses = classify_pareto_status(result, alpha=0.05)
    assert statuses["B"].status == "dominated"
    assert statuses["B"].dominated_by == ["A"]


def test_frontier_when_point_estimate_not_dominated():
    # A > B > C on primary, but C > B > A on secondary -- a genuine 3-way
    # tradeoff where no entity's point estimate dominates another's.
    result = _manual_result(
        p_dominated_by=np.zeros((3, 3)),
        point_primary=[0.75, 0.60, 0.50],
        point_secondary=[0.30, 0.80, 0.90],
    )
    statuses = classify_pareto_status(result, alpha=0.05)
    for label in ["A", "B", "C"]:
        assert statuses[label].status == "frontier"


# ---------------------------------------------------------------------------
# compare(secondary_metric=...) integration
# ---------------------------------------------------------------------------

def _make_evaldata(models, acc, lat, n_items=150, seed=0, missing_cell=None):
    rng = _rng(seed)
    rows = []
    for m in models:
        for i in range(n_items):
            rows.append({
                "model": m,
                "item": f"q{i}",
                "score": float(np.clip(rng.normal(acc[m], 0.08), 0, 1)),
                "latency_ms": float(np.clip(rng.normal(lat[m], 0.4), 0.05, None)),
            })
    df = pd.DataFrame(rows)
    if missing_cell is not None:
        model, item = missing_cell
        df.loc[(df["model"] == model) & (df["item"] == item), "latency_ms"] = np.nan
    return es.load_from(df, col_map={"model": "model", "item": "item"})


_MODELS = ["gpt-4o", "claude-sonnet", "llama-70b"]
_ACC = {"gpt-4o": 0.85, "claude-sonnet": 0.75, "llama-70b": 0.70}
_LAT = {"gpt-4o": 2.0, "claude-sonnet": 3.0, "llama-70b": 0.6}  # llama much faster (tradeoff vs gpt-4o)


def test_compare_secondary_end_to_end():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=10)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(11),
    )
    statuses = result.pareto_status
    assert statuses["gpt-4o"].status == "frontier"
    assert statuses["llama-70b"].status == "frontier"       # genuine tradeoff
    assert statuses["claude-sonnet"].status == "dominated"  # worse on both axes
    assert statuses["claude-sonnet"].dominated_by == ["gpt-4o"]

    probs = result.pareto_frontier_probability
    assert probs["gpt-4o"] > 0.9
    assert probs["claude-sonnet"] < 0.1


def test_compare_no_secondary_leaves_pareto_none():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=12)
    result = es.compare(evaldata, factors="model", metric="score", rng=_rng(13))
    assert result.pareto_status is None
    assert result.pareto_frontier_probability is None


def test_compare_secondary_bad_direction_raises():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=14)
    with pytest.raises(ValueError):
        es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric={"latency_ms": "lower"}, rng=_rng(15),
        )


def test_compare_secondary_n_way_not_implemented():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=16)
    with pytest.raises(NotImplementedError):
        es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric={"latency_ms": "min", "score": "max"}, rng=_rng(17),
        )


def test_compare_secondary_non_dict_warns_and_ignores():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=18)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric="latency_ms", rng=_rng(19),
        )
        assert any("secondary_metric=" in str(x.message) for x in w)
    assert result.pareto_status is None


def test_compare_secondary_missing_column_raises():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=20)
    with pytest.raises(es.EvalLoadError):
        es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric={"nonexistent_col": "min"}, rng=_rng(21),
        )


def test_compare_secondary_incomplete_design_raises():
    evaldata = _make_evaldata(
        _MODELS, _ACC, _LAT, seed=22, missing_cell=("llama-70b", "q0"),
    )
    with pytest.raises(ValueError, match="missing"):
        es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric={"latency_ms": "min"}, rng=_rng(23),
        )


def test_compare_secondary_warns_for_multi_model():
    rng = _rng(24)
    rows = []
    for m in ["gpt-4o", "claude-sonnet"]:
        for p in ["p1", "p2"]:
            for i in range(60):
                rows.append({
                    "model": m, "prompt": p, "item": f"q{i}",
                    "score": float(np.clip(rng.normal(0.7, 0.1), 0, 1)),
                    "latency_ms": float(rng.normal(2.0, 0.3)),
                })
    df = pd.DataFrame(rows)
    evaldata = es.load_from(df, col_map={"model": "model", "prompt": "prompt", "item": "item"})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = es.compare(
            evaldata, factors="model", metric="score",
            secondary_metric={"latency_ms": "min"}, rng=_rng(25),
        )
        assert any("multi-model" in str(x.message) for x in w)
    assert result.pareto_status is None


def test_to_dict_includes_pareto():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=26)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(27), show_rank_probabilities=True,
    )
    d = result.to_dict()
    assert d["pareto"]["secondary_metric"] == "latency_ms"
    assert d["pareto"]["direction"] == "min"
    assert d["pareto"]["entities"]["claude-sonnet"]["status"] == "dominated"
    assert "p_pareto_optimal" in d["pareto"]["entities"]["gpt-4o"]


def test_to_dict_omits_p_pareto_optimal_by_default():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=28)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(29),
    )
    d = result.to_dict()
    assert "p_pareto_optimal" not in d["pareto"]["entities"]["gpt-4o"]


def test_to_frame_includes_pareto():
    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=30)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(31),
    )
    frames = result.to_frame()
    assert "pareto" in frames
    assert set(frames["pareto"]["entity"]) == set(_MODELS)


def test_summary_prints_pareto_section():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=32)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(33),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    assert "Pareto Front" in out
    assert "Best trade-off" in out
    assert "Worse than" in out


def test_summary_pareto_shows_probability_only_when_requested():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=34)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(35),
    )

    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        result.summary()
    assert "P(Pareto-optimal)" not in buf1.getvalue()

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        result.summary(show_rank_probabilities=True)
    assert "P(Pareto-optimal)" in buf2.getvalue()


# ---------------------------------------------------------------------------
# Section ordering, sorting, and richer statistics (design follow-up)
# ---------------------------------------------------------------------------

def test_pareto_sorted_labels_groups_by_status_then_primary_mean():
    labels = ["A", "B", "C", "D"]
    result = ParetoBootstrapResult(
        labels=labels,
        point_primary=np.array([0.60, 0.90, 0.70, 0.50]),  # B > C > A > D
        point_secondary=np.zeros(4),
        p_frontier=np.zeros(4),
        p_dominated_by=np.zeros((4, 4)),
        n_bootstrap=1000,
    )
    statuses = {
        "A": type("S", (), {"status": "frontier"})(),
        "B": type("S", (), {"status": "dominated"})(),
        "C": type("S", (), {"status": "frontier"})(),
        "D": type("S", (), {"status": "ambiguous"})(),
    }
    pareto = {"result": result, "statuses": statuses}
    # frontier group (C, A by primary desc), then ambiguous (D), then dominated (B).
    assert _pareto_sorted_labels(pareto) == ["C", "A", "D", "B"]


def test_pareto_front_precedes_executive_summary():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=36)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(37),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    pareto_idx = out.index("Pareto Front")
    exec_idx = out.index("Executive Summary")
    assert pareto_idx < exec_idx


def test_pareto_table_shows_secondary_metric_statistics():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=38)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(39),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    assert "latency_ms" in out
    assert "95% CI" in out
    # Secondary metric's calibrated CI is actually computed and stored.
    sec_rob = result._pareto["secondary_robustness"]
    assert sec_rob.ci_low is not None and sec_rob.ci_high is not None
    for lo, hi in zip(sec_rob.ci_low, sec_rob.ci_high):
        assert lo <= hi


def test_executive_summary_has_pareto_column():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=40)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(41),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    exec_section = out[out.index("Executive Summary"):]
    assert "Trade-off" in exec_section.splitlines()[1]  # header row has the column
    # claude-sonnet is dominated by gpt-4o in this fixture -- its row should say so.
    for line in exec_section.splitlines():
        if line.strip().startswith("claude-sonnet"):
            assert "Worse than" in line


def test_executive_summary_no_pareto_column_without_secondary():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=42)
    result = es.compare(evaldata, factors="model", metric="score", rng=_rng(43))
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    exec_section = out[out.index("Executive Summary"):]
    assert "Trade-off" not in exec_section.splitlines()[1]


def test_executive_summary_verdict_header_scoped_to_metric_when_pareto_shown():
    """"Verdict" alone would read as the final word once a second axis
    (Trade-off) exists in the same row -- it should be relabeled to make
    clear it's scoped to the primary metric only. Without secondary_metric=,
    the plain "Verdict" header is unambiguous and should stay as-is."""
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=60)
    with_secondary = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(61),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        with_secondary.summary()
    out = buf.getvalue()
    header_line = out[out.index("Executive Summary"):].splitlines()[1]
    assert "On score" in header_line
    assert "Verdict" not in header_line

    without_secondary = es.compare(evaldata, factors="model", metric="score", rng=_rng(63))
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        without_secondary.summary()
    out2 = buf2.getvalue()
    header_line2 = out2[out2.index("Executive Summary"):].splitlines()[1]
    assert "Verdict" in header_line2
    assert "On score" not in header_line2


def test_executive_summary_tradeoff_header_names_secondary_metric():
    """The Trade-off column header should name the secondary metric too
    (paired with "On {metric}"), so the two headers alone state both axes
    without needing the Pareto section above -- and truncate a long
    secondary column name rather than blowing out the table width."""
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=64)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(65),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    header_line = buf.getvalue()[buf.getvalue().index("Executive Summary"):].splitlines()[1]
    assert "Trade-off vs latency_ms" in header_line

    # Long secondary column name gets truncated, not left to stretch the table.
    long_col = "average_response_latency_milliseconds_p99"
    rng2 = _rng(66)
    rows = [
        {
            "model": m, "item": f"q{i}",
            "score": float(np.clip(rng2.normal(_ACC[m], 0.08), 0, 1)),
            long_col: float(np.clip(rng2.normal(_LAT[m], 0.4), 0.05, None)),
        }
        for m in _MODELS for i in range(150)
    ]
    evaldata2 = es.load_from(pd.DataFrame(rows), col_map={"model": "model", "item": "item"})
    result2 = es.compare(
        evaldata2, factors="model", metric="score",
        secondary_metric={long_col: "min"}, rng=_rng(67),
    )
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        result2.summary()
    header_line2 = buf2.getvalue()[buf2.getvalue().index("Executive Summary"):].splitlines()[1]
    assert long_col not in header_line2
    assert "Trade-off vs average" in header_line2


def test_pareto_status_phrase_merges_status_and_detail():
    from evalstats.core.summary import _pareto_status_phrase
    from evalstats.core.pareto import ParetoStatus

    frontier = ParetoStatus(label="A", status="frontier")
    dominated = ParetoStatus(label="B", status="dominated", dominated_by=["A"])
    ambiguous = ParetoStatus(label="C", status="ambiguous", ambiguous_vs=["A"])

    assert _pareto_status_phrase(frontier) == "★ Best trade-off"
    assert _pareto_status_phrase(dominated) == "× Worse than A on both"
    assert _pareto_status_phrase(ambiguous, verbose=True) == "◌ Unclear vs A (not confirmed)"
    # Executive Summary's narrower column drops the "(not confirmed)" qualifier.
    assert _pareto_status_phrase(ambiguous, verbose=False) == "◌ Unclear vs A"


def test_pareto_table_has_single_merged_status_column():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=44)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(45),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    pareto_section = out[out.index("Pareto Front"):out.index("Executive Summary")]
    assert "Detail" not in pareto_section
    assert "Worse than gpt-4o on both" in pareto_section


def test_pareto_callout_names_frontier_alternatives():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=46)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(47),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    exec_idx = out.index("Executive Summary")
    next_idx = out.index("What to do next")
    between = out[exec_idx:next_idx]
    assert "leads on score" in between
    # llama-70b and gpt-4o-mini are both genuine tradeoffs against gpt-4o in
    # this fixture (see test_compare_secondary_end_to_end); which ones clear
    # the bootstrap confidence threshold at this particular seed can vary,
    # so just check at least one frontier alternative is actually named.
    assert "llama-70b" in between or "gpt-4o-mini" in between
    assert "competitive trade-off" in between  # singular or plural


def test_pareto_callout_absent_without_secondary():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=48)
    result = es.compare(evaldata, factors="model", metric="score", rng=_rng(49))
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    assert "leads on" not in out


# ---------------------------------------------------------------------------
# Scatterplot + glyph/definition line (design follow-up: visual trade-off view)
# ---------------------------------------------------------------------------

def test_pareto_status_glyphs_are_distinct():
    from evalstats.core.summary import _pareto_status_glyph

    glyphs = {_pareto_status_glyph(s) for s in ("frontier", "dominated", "ambiguous")}
    assert len(glyphs) == 3


def test_join_names_capped_truncates_long_dominator_lists():
    from evalstats.core.summary import _join_names_capped

    assert _join_names_capped(["A"]) == "A"
    assert _join_names_capped(["A", "B"]) == "A, B"
    # Entities dominated by many others would otherwise blow up the Status
    # column (and the whole table) with a name list as wide as the table.
    assert _join_names_capped(["A", "B", "C"]) == "A, B and 1 more"
    assert _join_names_capped(["A", "B", "C", "D", "E"]) == "A, B and 3 more"


def test_pareto_table_entity_column_capped_for_long_names():
    import io
    from contextlib import redirect_stdout

    models = [
        "gpt-4o-2024-11-20-high-reasoning-effort-and-then-some-more-text",
        "claude-opus-4-5-extended-thinking-mode-with-a-very-long-suffix",
    ]
    acc = {models[0]: 0.85, models[1]: 0.70}
    lat = {models[0]: 2.0, models[1]: 0.5}
    evaldata = _make_evaldata(models, acc, lat, seed=54)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(55),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    pareto_section = out[out.index("Pareto Front"):out.index("Executive Summary")]
    # Table rows now lead with a single-char "#" marker column (matching the
    # scatterplot's per-entity numbering) before the entity name -- strip it
    # off before checking where the (truncated) name starts.
    def _drop_marker(line: str) -> str:
        return re.sub(r"^\S\s+", "", line.strip())

    # No line in the table should run away to the full untruncated name length.
    table_lines = [
        l for l in pareto_section.splitlines()
        if _drop_marker(l).startswith(models[0][:10]) or _drop_marker(l).startswith(models[1][:10])
    ]
    assert table_lines
    for line in table_lines:
        # +40 budget, plus the leading "#  " marker column's own width.
        assert len(line) < len(models[0]) + 40 + 3


def test_pareto_scatter_flags_near_degenerate_axis():
    import io
    from contextlib import redirect_stdout

    # All entities share the ~same latency (tight noise, same true mean) --
    # the secondary axis is essentially flat, and the scatterplot should say
    # so rather than silently stretching sampling noise to fill the plot
    # width. Built directly (not via _make_evaldata) because that helper's
    # fixed noise std isn't tight enough relative to a mean of 1.0 to land
    # inside the "near-degenerate" threshold.
    rng = _rng(58)
    models = ["a", "b", "c", "d", "e"]
    acc = {"a": 0.60, "b": 0.68, "c": 0.75, "d": 0.82, "e": 0.90}
    rows = []
    for m in models:
        for i in range(150):
            rows.append({
                "model": m,
                "item": f"q{i}",
                "score": float(np.clip(rng.normal(acc[m], 0.08), 0, 1)),
                "latency_ms": float(rng.normal(1.0, 0.01)),
            })
    evaldata = es.load_from(pd.DataFrame(rows), col_map={"model": "model", "item": "item"})
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(59),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    pareto_section = out[out.index("Pareto Front"):out.index("Executive Summary")]
    assert "barely varies" in pareto_section
    assert "latency_ms" in pareto_section


def test_pareto_section_has_definition_line_and_scatterplot():
    import io
    from contextlib import redirect_stdout

    evaldata = _make_evaldata(_MODELS, _ACC, _LAT, seed=50)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(51),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    pareto_section = out[out.index("Pareto Front"):out.index("Executive Summary")]
    assert "best trade-off" in pareto_section
    # Scatterplot axis box (└...┘) and a numbered legend mapping back to names.
    assert "└" in pareto_section and "┘" in pareto_section
    assert "1=" in pareto_section
    for label in _MODELS:
        assert label in pareto_section


def test_pareto_scatter_handles_two_entities():
    """A minimal 2-entity case shouldn't crash the scatterplot renderer
    (degenerate axis ranges, single frontier/dominated pair)."""
    import io
    from contextlib import redirect_stdout

    models = _MODELS[:2]
    evaldata = _make_evaldata(models, {k: _ACC[k] for k in models}, {k: _LAT[k] for k in models}, seed=52)
    result = es.compare(
        evaldata, factors="model", metric="score",
        secondary_metric={"latency_ms": "min"}, rng=_rng(53),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        result.summary()
    out = buf.getvalue()
    assert "Pareto Front" in out
