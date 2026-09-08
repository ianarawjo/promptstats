"""Tests for the between-subjects comparison engine:
evalstats.core.unpaired.compare_unpaired(), GroupComparisonResult, and
compare(design=...) routing in evalstats/api.py.
"""
from __future__ import annotations

import io
import warnings as warnings_lib
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.core.unpaired import compare_unpaired, GroupComparisonResult, SYNTHETIC_ITEM_COL
from evalstats.alignment import judge_alignment


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_unpaired_df(
    group_means: dict[str, float],
    n_per_group: int | dict[str, int] = 40,
    std: float = 0.15,
    seed: int = 0,
    item_col: str | None = "item",
) -> pd.DataFrame:
    """Disjoint-item long-format continuous data, one independent cohort per group."""
    rng = _rng(seed)
    rows = []
    for g, mean in group_means.items():
        n = n_per_group[g] if isinstance(n_per_group, dict) else n_per_group
        for i in range(n):
            row = {"model": g, "score": float(np.clip(rng.normal(mean, std), 0, 1))}
            if item_col is not None:
                row[item_col] = f"{g}_{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def _make_unpaired_binary_df(
    group_p: dict[str, float], n_per_group: int = 40, seed: int = 1,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows = []
    for g, p in group_p.items():
        for i in range(n_per_group):
            rows.append({"model": g, "item": f"{g}_{i}", "score": float(rng.binomial(1, p))})
    return pd.DataFrame(rows)


def _make_unpaired_with_alignment(
    group_means: dict[str, float], n_per_group: int = 60, n_labeled_per_group: int = 20, seed: int = 2,
):
    """Disjoint-item continuous data with a sparse human_score column, for PPI tests."""
    rng = _rng(seed)
    rows = []
    for g, mean in group_means.items():
        for i in range(n_per_group):
            rows.append({"model": g, "item": f"{g}_{i}", "llm_score": float(np.clip(rng.normal(mean, 0.15), 0, 1))})
    df = pd.DataFrame(rows)
    human = np.full(len(df), np.nan)
    for g in group_means:
        idx = df.index[df["model"] == g].to_numpy()
        chosen = rng.choice(idx, size=min(n_labeled_per_group, len(idx)), replace=False)
        for j in chosen:
            human[j] = float(np.clip(df.loc[j, "llm_score"] + rng.normal(0, 0.05), 0, 1))
    df["human_score"] = human
    return df


# ─────────────────────────────────────────────────────────────────────────────
# compare_unpaired() -- direct engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareUnpairedBasics:
    def test_k2_continuous_rank_based_no_omnibus(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert isinstance(r, GroupComparisonResult)
        assert r.family == "rank_based"
        assert r.score_type == "continuous"
        assert len(r.groups) == 2
        assert r.n_pairs == 1
        assert r.omnibus_test_name is None
        assert r.ci_correction == "none"
        assert r.pvalue_correction == "none"
        assert len(r.pairwise) == 1
        pair = r.pairwise[0]
        # Every unpaired family reports a mean difference -- "rank_based"
        # names the tests (Kruskal-Wallis/Mann-Whitney), not the estimand.
        assert pair.estimand == "mean_diff"
        assert pair.null_value == 0.0
        means = {g.label: g.mean for g in r.groups}
        assert pair.point_estimate == pytest.approx(
            means[pair.label_a] - means[pair.label_b], abs=1e-9)
        # B has a clearly higher mean, so the interval should exclude 0.
        assert pair.significant

    def test_k3_continuous_has_omnibus_and_corrections(self):
        df = _make_unpaired_df({"A": 0.3, "B": 0.5, "C": 0.7})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert len(r.groups) == 3
        assert r.n_pairs == 3
        assert r.omnibus_test_name == "Kruskal-Wallis test"
        assert r.omnibus_statistic is not None
        assert r.omnibus_p_value is not None
        assert r.ci_correction == "bonferroni"
        # Shaffer, not Holm: identical FWER for an all-pairwise family (both
        # divide by m at step 1) but strictly more powerful from step 2 on.
        assert r.pvalue_correction == "shaffer"
        assert len(r.pairwise) == 3
        # Widely separated means -> omnibus should reject at alpha=0.05.
        assert r.omnibus_p_value < 0.05

    def test_binary_family_uses_anova_and_ttest(self):
        df = _make_unpaired_binary_df({"A": 0.3, "B": 0.5, "C": 0.8})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert r.score_type == "binary"
        assert r.family == "binary_proportion"
        assert r.omnibus_test_name == "One-way ANOVA (independent)"
        for pair in r.pairwise:
            assert pair.estimand == "mean_diff"
            assert pair.null_value == 0.0

    def test_unbalanced_group_sizes(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6}, n_per_group={"A": 15, "B": 55})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        n_a = next(g.n for g in r.groups if g.label == "A")
        n_b = next(g.n for g in r.groups if g.label == "B")
        assert n_a == 15
        assert n_b == 55

    def test_synthetic_item_column_fallback(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6}, item_col=None)
        assert "item" not in df.columns
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert r.item_col_synthetic is True
        assert r.item_col == SYNTHETIC_ITEM_COL

    def test_explicit_item_col_not_in_data_raises(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="model", metric_col="score", item_col="nonexistent")

    def test_unknown_factor_col_raises(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="nonexistent", metric_col="score")

    def test_single_group_raises(self):
        df = _make_unpaired_df({"A": 0.4})
        with pytest.raises(ValueError, match="at least 2 groups"):
            compare_unpaired(df, factor_col="model", metric_col="score")


class TestCompareUnpairedNaNAndPPIGuards:
    """Regression tests for bugs found by an independent integration review
    of the between-subjects engine (2026-08-15): NaN handling in the metric
    column, and PPI label-sanitization bypass.
    """

    def test_nan_scores_dropped_with_warning_not_poisoning_result(self):
        rng = _rng(10)
        rows = []
        for g, mean in [("A", 0.4), ("B", 0.6), ("C", 0.5)]:
            for i in range(30):
                score = float(np.clip(rng.normal(mean, 0.15), 0, 1))
                if rng.random() < 0.1:
                    score = float("nan")
                rows.append({"group": g, "item": f"{g}_{i}", "score": score})
        df = pd.DataFrame(rows)
        assert df["score"].isna().sum() > 0
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="group", metric_col="score", n_boot=300, rng=10)
        for g in r.groups:
            assert not np.isnan(g.mean)
            assert not np.isnan(g.ci_low) and not np.isnan(g.ci_high)
        assert any(g.n < 30 for g in r.groups)  # some rows were dropped somewhere
        assert not np.isnan(r.omnibus_p_value)

    def test_nan_scores_dont_crash_binary_family(self):
        rng = _rng(11)
        rows = []
        for g, p in [("A", 0.3), ("B", 0.6)]:
            for i in range(30):
                score = float(rng.binomial(1, p))
                if rng.random() < 0.1:
                    score = float("nan")
                rows.append({"group": g, "item": f"{g}_{i}", "score": score})
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="group", metric_col="score", n_boot=300, rng=11)
        assert r.score_type == "binary"

    def test_all_nan_group_raises_clear_error(self):
        rows = [{"group": "A", "item": f"A_{i}", "score": float("nan")} for i in range(10)]
        rows += [{"group": "B", "item": f"B_{i}", "score": 0.5} for i in range(10)]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="no valid"):
            compare_unpaired(df, factor_col="group", metric_col="score")

    def test_ppi_zero_labeled_group_raises_clear_error(self):
        df = _make_unpaired_with_alignment({"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=20)
        # Wipe out B's labels entirely after generation -- zero-labeled group.
        df = df.copy()
        df.loc[df["model"] == "B", "human_score"] = np.nan
        assert df.loc[df["model"] == "B", "human_score"].notna().sum() == 0
        assert df.loc[df["model"] == "A", "human_score"].notna().sum() > 0
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        with pytest.raises(ValueError, match="zero labeled"):
            compare_unpaired(df, factor_col="model", metric_col="llm_score", alignment={"llm_score": ar})

    def test_ppi_too_few_total_labels_raises_clear_error(self):
        df = _make_unpaired_with_alignment({"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=2)
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        with pytest.raises(ValueError, match="At least 15 human labels"):
            compare_unpaired(df, factor_col="model", metric_col="llm_score", alignment={"llm_score": ar})

    def test_score_range_threaded_through_and_suppresses_autodetect_warning(self):
        rng = _rng(12)
        rows = []
        for g, mean in [("A", 2.0), ("B", 3.5)]:
            for i in range(30):
                rows.append({"group": g, "item": f"{g}_{i}",
                             "score": float(np.clip(rng.normal(mean, 1.0), 1, 5))})
        df = pd.DataFrame(rows)
        with warnings_lib.catch_warnings(record=True) as caught:
            warnings_lib.simplefilter("always")
            r = compare_unpaired(df, factor_col="group", metric_col="score", score_range=(1, 5))
        assert not any("score_range" in str(w.message) for w in caught)
        assert r.groups[0].method != "t_interval"  # bounds-agnostic fallback shouldn't fire

    def test_method_override_raises_via_compare(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="method='bca'"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired", method="bca")

    def test_numeric_pairwise_estimand_is_mean_but_test_is_mannwhitney(self):
        """The numeric family reports a mean difference with a Welch interval,
        while its p-value still comes from Mann-Whitney U -- the post-hoc that
        follows the Kruskal-Wallis omnibus, and the test this project's PPI
        work validates. Both halves are asserted against the public/scipy
        references so neither can drift silently.

        Because those are different estimands, ``mean_test_p`` carries the
        interval's own (Welch) p-value alongside; assert it is present and
        distinct from the headline p.
        """
        from scipy.stats import mannwhitneyu, ttest_ind
        from evalstats.core.unpaired import _numeric_pairwise_uncorrected

        rng = _rng(99)
        x = rng.normal(0.4, 0.15, 40)
        y = rng.normal(0.6, 0.15, 35)
        out = _numeric_pairwise_uncorrected([x, y], alpha=0.05)

        assert out["point"][0] == pytest.approx(np.mean(x) - np.mean(y))
        welch = ttest_ind(x, y, equal_var=False)
        ci = welch.confidence_interval(confidence_level=0.95)
        assert out["ci_lo"][0] == pytest.approx(float(ci.low))
        assert out["ci_hi"][0] == pytest.approx(float(ci.high))
        assert out["pair_p"][0] == pytest.approx(
            float(mannwhitneyu(x, y, alternative="two-sided").pvalue))
        assert out["mean_test_p"][0] == pytest.approx(float(welch.pvalue))

    def test_judged_secondary_metric_is_refused_not_silently_uncorrected(self):
        """PPI reaches the primary metric only -- pareto_bootstrap_unpaired
        takes no labels and the secondary metric's CIs are computed without
        them. An alignment entry for the secondary column would therefore be
        accepted and silently dropped, reporting uncorrected frontier
        probabilities as if corrected. Assert it raises instead, and that a
        judge-free secondary (cost) still works alongside a corrected primary.
        """
        df = _make_unpaired_with_alignment({"A": 0.4, "B": 0.6, "C": 0.5}, n_per_group=40)
        df = df.copy()
        rng = _rng(11)
        df["cost"] = rng.normal(10, 2, len(df))
        df["quality2"] = df["llm_score"]
        df["human2"] = df["human_score"]
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            a1 = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            a2 = judge_alignment(evaldata, llm_metric="quality2", human_groundtruth="human2")

            with pytest.raises(ValueError, match="secondary metric is not supported"):
                compare_unpaired(
                    df, factor_col="model", metric_col="llm_score",
                    alignment={"llm_score": a1, "quality2": a2},
                    secondary_metric={"quality2": "max"}, n_boot=100, rng=1,
                )

            # A judge-free secondary is the supported case and must still run.
            r = compare_unpaired(
                df, factor_col="model", metric_col="llm_score",
                alignment={"llm_score": a1}, secondary_metric={"cost": "min"},
                n_boot=100, rng=1,
            )
        assert r.ppi_applied
        assert r.pareto is not None

    def test_routing_table_family_drives_dispatch(self):
        from evalstats.config import resolve_auto_unpaired_methods
        for score_type in ["binary", "continuous", "likert"]:
            family, omnibus_method, pairwise_method = resolve_auto_unpaired_methods(score_type)
            assert family in ("binary_proportion", "rank_based")
            if score_type == "binary":
                assert family == "binary_proportion"
                assert omnibus_method == "anova_oneway"
            else:
                assert family == "rank_based"
                assert omnibus_method == "kruskalwallis"


class TestGroupComparisonResultReporting:
    def _result(self) -> GroupComparisonResult:
        df = _make_unpaired_df({"A": 0.3, "B": 0.5, "C": 0.7})
        return compare_unpaired(df, factor_col="model", metric_col="score")

    def test_summary_runs_without_error(self):
        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Shape: BetweenGroups(" in out
        assert "Kruskal-Wallis" in out

    def test_plot_not_implemented(self):
        r = self._result()
        with pytest.raises(NotImplementedError):
            r.plot()

    def test_executive_summary_and_critical_difference_bands_present(self):
        """compare(design="unpaired") now shows an executive summary
        leaderboard and critical-difference rank bands, matching the
        paired path -- reused via _GroupComparisonResultAsBundle/
        _GroupDiffResultsAsPairwiseMatrix rather than reimplemented.
        """
        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Executive Summary (Group leaderboard)" in out
        assert "Grp" in out and "Verdict" in out
        assert "#1" in out
        assert "Statistically indistinguishable rank bands" in out

    def test_critical_difference_bands_use_mean_order_not_factor_order(self):
        # C has the highest mean (0.7) but is defined last in the factor
        # column order -- the CD bands / executive summary must rank by
        # mean (best first), not by group/factor-level order.
        df = _make_unpaired_df({"A": 0.3, "B": 0.7, "C": 0.5})
        r = compare_unpaired(df, factor_col="model", metric_col="score", n_boot=800, rng=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        exec_section = out.split("Executive Summary")[1]
        # B (highest mean) must be the first data row after the header.
        b_line_idx = exec_section.find("\n  B ")
        a_line_idx = exec_section.find("\n  A ")
        c_line_idx = exec_section.find("\n  C ")
        assert 0 < b_line_idx < a_line_idx
        assert 0 < b_line_idx < c_line_idx

    def test_executive_summary_k2_still_works(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.7})
        r = compare_unpaired(df, factor_col="model", metric_col="score", n_boot=500, rng=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Executive Summary" in out
        assert "#1" in out and "#2" in out

    def test_executive_summary_shows_pareto_tradeoff_column_when_present(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.85, "C": 0.4}, seed=41)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=800, rng=41)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Trade-off" in out
        assert "On score" in out  # verdict column relabeled once Pareto is present

    def test_pairwise_table_uses_shared_print_pairwise_section(self):
        """The pairwise comparison table is rendered by the SAME function
        the paired path uses (core.summary._print_pairwise_section), not a
        parallel reimplementation -- print_group_comparison_summary itself
        lives in core/summary.py alongside it (no separate
        core/summary_unpaired.py module). This changed the unpaired table's
        format: an interval-plot bar per pair (previously text-only), the
        estimand shown as a signed difference (Δ, or Δp for binary),
        and p-values with significance stars -- replacing the old verbal
        "Verdict: significant (A < B)" column, which doesn't exist in the
        shared renderer.
        """
        from evalstats.core.summary import print_group_comparison_summary
        assert print_group_comparison_summary.__module__ == "evalstats.core.summary"

        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "effect: Left - Right" in out  # shared axis/legend line
        assert "Δ" in out  # mean difference, null already 0 -- no shift applied
        assert "θ" not in out  # the dominance estimand is gone entirely
        # Old per-row verbal verdict cell ("significant (A < B)" / "not
        # significant") is gone -- replaced by the shared table's numeric
        # CI + p + stars. The unrelated footer sentence ("Verdict reflects
        # the ...-corrected CI...") is intentionally still present.
        assert "significant (" not in out

    def test_pairwise_table_primary_column_is_the_mean_difference(self):
        """The numeric family's primary column *is* the mean difference, so
        there is no secondary Δmean column any more -- that column existed
        only to put the old dominance estimand back on the metric's own
        scale. Assert the printed value in the primary column matches the
        marginal means' difference, and that neither family prints a
        redundant second copy of it.
        """
        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Δmean" not in out

        means = {g.label: g.mean for g in r.groups}
        pair = r.pairwise[0]
        expected = means[pair.label_a] - means[pair.label_b]
        assert pair.point_estimate == pytest.approx(expected, abs=1e-9)
        row_line = next(
            line for line in out.splitlines()
            if line.strip().startswith(pair.label_a) and pair.label_b in line
        )
        assert f"{expected:.3f}" in row_line or f"{expected:.2f}" in row_line

        df_bin = _make_unpaired_binary_df({"A": 0.3, "B": 0.6})
        r_bin = compare_unpaired(df_bin, factor_col="model", metric_col="score")
        buf_bin = io.StringIO()
        with redirect_stdout(buf_bin):
            r_bin.summary()
        assert "Δmean" not in buf_bin.getvalue()

    def test_means_table_uses_shared_print_mean_advantage(self):
        """The per-group means table is rendered by the SAME function the
        paired path uses (core.summary._print_mean_advantage), not a
        parallel reimplementation. A change to that shared function's
        section header renders identically for both paths; assert the
        literal header text here as a tripwire against that sharing
        silently regressing back into two independent implementations.
        """
        from evalstats.core.summary import print_group_comparison_summary, _print_mean_advantage
        assert _print_mean_advantage.__module__ == "evalstats.core.summary"

        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        # Exact text _print_mean_advantage prints -- same string the paired
        # path's own summary shows for its equivalent section.
        assert "--- Mean Performance (" in out

    def test_to_dict_shape(self):
        r = self._result()
        d = r.to_dict()
        assert d["design"] == "unpaired"
        assert set(d["groups"].keys()) == {"A", "B", "C"}
        assert d["omnibus"]["test_name"] == "Kruskal-Wallis test"
        assert len(d["pairwise"]) == 3

    def test_to_frame_shape(self):
        r = self._result()
        frame = r.to_frame()
        assert len(frame) == 3
        assert {"a", "b", "point_estimate", "ci_low", "ci_high", "p_value", "significant"} <= set(frame.columns)

    def test_groups_to_frame_shape(self):
        r = self._result()
        frame = r.groups_to_frame()
        assert len(frame) == 3
        assert frame.index.name == "label"
        assert set(frame.index) == {"A", "B", "C"}

    def test_labels_property(self):
        r = self._result()
        assert set(r.labels) == {"A", "B", "C"}


class TestCompareUnpairedWithPPI:
    def test_ppi_applied_and_single_alignment_banner(self):
        df = _make_unpaired_with_alignment({"A": 0.35, "B": 0.65})
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = compare_unpaired(
            df, factor_col="model", metric_col="llm_score",
            alignment={"llm_score": ar},
        )
        assert r.ppi_applied is True
        assert r.alignment_result is ar

        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        # Exactly one alignment report should be printed, not a duplicate/stale second one.
        assert out.count("PPI-CORRECTED") == 1

    def test_ppi_corrects_the_marginal_group_mean_not_just_pairwise(self):
        """Regression test for a bug where GroupStat.mean (the "Mean
        Performance" table, and the Delta-mean pairwise column derived from
        it) was ALWAYS computed from raw judge scores via
        _compute_group_stats -> robustness_metrics, which has no PPI/
        alignment parameter at all -- so alignment= silently had zero
        effect on the marginal mean, even though the pairwise Delta-theta/
        Delta-p WAS genuinely corrected. Found by comparing compare(...)
        with and without alignment= on real biased-judge data and noticing
        the "PPI-corrected" group means were bit-for-bit identical to the
        uncorrected ones.

        Here: a judge with a deliberate, systematic downward bias (true - 3
        clipped) for the "A" group only -- the correction should move A's
        mean substantially toward its true value, while leaving the
        well-calibrated "B" group's mean roughly where it was.
        """
        rng = _rng(9)
        rows = []
        for i in range(60):
            true = int(np.clip(round(rng.normal(3.5, 0.9)), 1, 5))
            judge = int(np.clip(true - 3, 1, 5))  # systematic downward bias
            rows.append({"model": "A", "item": f"A_{i}", "llm_score": judge,
                         "human_score": true if i < 20 else np.nan})
        for i in range(60):
            true = int(np.clip(round(rng.normal(3.5, 0.9)), 1, 5))
            judge = int(np.clip(round(true + rng.normal(0, 0.3)), 1, 5))  # well-calibrated
            rows.append({"model": "B", "item": f"B_{i}", "llm_score": judge,
                         "human_score": true if i < 20 else np.nan})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score", selection="random")

        r_ppi = compare_unpaired(df, factor_col="model", metric_col="llm_score",
                                  alignment={"llm_score": ar}, score_range=(1, 5), rng=10)
        r_raw = compare_unpaired(df, factor_col="model", metric_col="llm_score",
                                  score_range=(1, 5), rng=10)
        mean_ppi = {g.label: g.mean for g in r_ppi.groups}
        mean_raw = {g.label: g.mean for g in r_raw.groups}

        # The correction must actually move the biased group's mean --
        # not be silently identical to the raw estimate.
        assert mean_ppi["A"] != pytest.approx(mean_raw["A"], abs=1e-9)
        # And move it in the right direction: corrected should be higher
        # than raw (raw underestimates A due to the downward judge bias),
        # substantially closer to A's true mean (~3.5) than raw is.
        assert mean_ppi["A"] > mean_raw["A"] + 0.5
        # The well-calibrated group's correction should be much smaller.
        assert abs(mean_ppi["B"] - mean_raw["B"]) < abs(mean_ppi["A"] - mean_raw["A"])

    def test_ppi_k2_pairwise_survives_degenerate_covariance_seeds(self):
        """Regression test for a ZeroDivisionError in
        evalstats.tests._ppi_kruskal_wallis_pairwise (found via
        simulations/investigate_unpaired_battle_test.py's crash grid): at
        k=2 there is exactly one pair, so the pairwise Wald covariance is a
        1x1 matrix that can come back numerically rank-0 (all bootstrap
        replicates ~identical) for some data/seed combinations -- 6 of 192
        battle-test grid cells hit this (continuous/grade score types,
        k=2, ppi=True, specific seeds). `_ppi_kruskal_wallis_pairwise` now
        guards `df == 0` explicitly instead of dividing by `nu * df`.
        Sweep several seeds here since the failure is seed-dependent and a
        single fixed seed previously happened not to trigger it.
        """
        for seed in range(6):
            df = _make_unpaired_with_alignment(
                {"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=8, seed=seed,
            )
            evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
            with warnings_lib.catch_warnings():
                warnings_lib.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            r = compare_unpaired(
                df, factor_col="model", metric_col="llm_score",
                alignment={"llm_score": ar}, n_boot=400, rng=seed,
            )
            assert 0.0 <= r.pairwise[0].p_value <= 1.0
            assert 0.0 <= r.pairwise[0].raw_p_value <= 1.0

    def test_ppi_three_groups_omnibus_and_pairwise_work(self):
        df = _make_unpaired_with_alignment({"A": 0.3, "B": 0.5, "C": 0.7}, n_per_group=60, n_labeled_per_group=20)
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = compare_unpaired(
            df, factor_col="model", metric_col="llm_score",
            alignment={"llm_score": ar},
        )
        assert r.ppi_applied is True
        assert r.omnibus_test_name == "Kruskal-Wallis test"
        assert r.omnibus_corrected_p_value is not None
        assert len(r.pairwise) == 3


def _make_unpaired_pareto_df(
    score_means: dict[str, float],
    n_per_group: int | dict[str, int] = 50,
    seed: int = 30,
    cost_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Disjoint-item data with a primary ("score", higher=better) and a
    secondary ("cost", lower=better) metric, row-aligned within each group.
    cost_means defaults to the same value (200) for every group -- tests
    that care about a specific dominance pattern pass cost_means explicitly
    (or override df["cost"] afterward).
    """
    rng = _rng(seed)
    rows = []
    for g, score_mean in score_means.items():
        cost_mean = cost_means[g] if cost_means else 200.0
        n = n_per_group[g] if isinstance(n_per_group, dict) else n_per_group
        for i in range(n):
            rows.append({
                "model": g, "item": f"{g}_{i}",
                "score": float(np.clip(rng.normal(score_mean, 0.08), 0, 1)),
                "cost": float(rng.normal(cost_mean, 15)),
            })
    return pd.DataFrame(rows)


class TestCompareUnpairedPareto:
    def test_clear_dominator_is_frontier_others_dominated(self):
        # B has both the best score AND the lowest cost -- unambiguous dominator.
        df = _make_unpaired_pareto_df({"A": 0.6, "B": 0.85, "C": 0.5}, n_per_group={"A": 50, "B": 50, "C": 50})
        # cost means chosen so B < A < C on cost too (B dominates both on both axes)
        df.loc[df["model"] == "A", "cost"] = _rng(31).normal(200, 15, (df["model"] == "A").sum())
        df.loc[df["model"] == "B", "cost"] = _rng(32).normal(150, 15, (df["model"] == "B").sum())
        df.loc[df["model"] == "C", "cost"] = _rng(33).normal(260, 15, (df["model"] == "C").sum())
        r = compare_unpaired(
            df, factor_col="model", metric_col="score",
            secondary_metric={"cost": "min"}, n_boot=800, rng=30,
        )
        assert r.pareto is not None
        assert r.pareto_status["B"].status == "frontier"
        assert r.pareto_status["A"].status == "dominated"
        assert r.pareto_status["C"].status == "dominated"
        assert "B" in r.pareto_status["A"].dominated_by
        assert r.pareto_frontier_probability["B"] == pytest.approx(1.0)
        assert r.pareto_frontier_probability["A"] == pytest.approx(0.0)

    def test_k2_pareto_works(self):
        df = _make_unpaired_pareto_df({"A": 0.6, "B": 0.6}, n_per_group=50)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=30)
        assert r.pareto is not None
        assert set(r.pareto_status.keys()) == {"A", "B"}

    def test_unbalanced_groups_pareto_works(self):
        df = _make_unpaired_pareto_df(
            {"A": 0.5, "B": 0.7, "C": 0.6}, n_per_group={"A": 15, "B": 60, "C": 30}, seed=34,
        )
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=34)
        assert r.pareto is not None
        assert len(r.pareto["result"].labels) == 3

    def test_max_direction(self):
        # secondary metric where higher is also better (e.g. throughput).
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=35)
        df = df.rename(columns={"cost": "throughput"})
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"throughput": "max"}, n_boot=500, rng=35)
        assert r.pareto is not None
        assert r.pareto["direction"] == "max"

    def test_malformed_secondary_metric_raises(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7})
        with pytest.raises(ValueError, match="exactly one entry"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min", "extra": "max"})
        with pytest.raises(ValueError, match="min.*or.*max"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "sideways"})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"nonexistent_col": "min"})

    def test_row_level_nan_in_either_metric_drops_jointly(self):
        rng = _rng(36)
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, n_per_group=40, seed=36)
        # NaN the cost for a few rows in A, and score for a few rows in B --
        # both should be dropped from BOTH arrays to preserve row alignment.
        idx_a = df.index[df["model"] == "A"][:3]
        idx_b = df.index[df["model"] == "B"][:2]
        df.loc[idx_a, "cost"] = np.nan
        df.loc[idx_b, "score"] = np.nan
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="model", metric_col="score",
                                  secondary_metric={"cost": "min"}, n_boot=500, rng=36)
        assert r.pareto is not None
        a_group = r._group("A")
        b_group = r._group("B")
        assert a_group.n == 37  # 40 - 3
        assert b_group.n == 38  # 40 - 2

    def test_to_dict_includes_pareto(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=37)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=37)
        d = r.to_dict()
        assert "pareto" in d
        assert d["pareto"]["secondary_metric"] == "cost"
        assert d["pareto"]["direction"] == "min"
        assert set(d["pareto"]["groups"].keys()) == {"A", "B"}
        for entry in d["pareto"]["groups"].values():
            assert "status" in entry and "p_pareto_optimal" in entry

    def test_to_dict_omits_pareto_when_not_requested(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=38)
        r = compare_unpaired(df, factor_col="model", metric_col="score", n_boot=500, rng=38)
        assert r.pareto is None
        assert r.pareto_status is None
        assert r.pareto_frontier_probability is None
        assert "pareto" not in r.to_dict()

    def test_summary_prints_pareto_section_using_shared_paired_renderer(self):
        """The Pareto section is rendered by the SAME function the paired
        path uses (core.summary._print_pareto_section), including its ASCII
        scatterplot -- see evalstats.core.unpaired._GroupStatsAsRobustness.
        """
        from evalstats.core.summary import _print_pareto_section
        assert _print_pareto_section.__module__ == "evalstats.core.summary"

        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.8, "C": 0.4}, seed=39)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=800, rng=39)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Trade-off" in out
        assert "Pareto Front" in out

    def test_design_unpaired_via_compare_with_secondary_metric(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.8}, seed=40)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        secondary_metric={"cost": "min"}, rng=40)
        assert isinstance(r, GroupComparisonResult)
        assert r.pareto is not None
        assert r.pareto_status["B"].status == "frontier"


# ─────────────────────────────────────────────────────────────────────────────
# compare(design=...) routing -- api.py integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPValuesOmnibusToggles:
    """p_values=/omnibus= default to False on the unpaired path, same as
    compare()'s own default on the paired path -- see api.py's design=
    docstring. Verifies both the default (unset) hides them exactly like
    the paired path does, and that explicit True actually shows them.
    """

    def _df(self):
        rng = _rng(50)
        rows = []
        for g, mean in [("A", 0.3), ("B", 0.5), ("C", 0.7)]:
            for i in range(30):
                rows.append({"model": g, "item": f"{g}_{i}",
                             "score": float(np.clip(rng.normal(mean, 0.15), 0, 1))})
        return pd.DataFrame(rows)

    def test_default_hides_both(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired", rng=1)
        assert r.show_p_values is False
        assert r.omnibus_test_name is None
        assert r.omnibus_statistic is None
        assert r.omnibus_p_value is None
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Omnibus Test" not in out
        # pairwise table is untouched
        assert len(r.pairwise) == 3

    def test_p_values_true_shows_column(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        p_values=True, rng=1)
        assert r.show_p_values is True
        # underlying values are computed either way and accessible programmatically
        assert all(p.p_value is not None for p in r.pairwise)
        assert "p_value" in r.to_frame().columns
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "  p" in out or "p " in out

    def test_omnibus_true_runs_and_shows_it(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        omnibus=True, rng=1)
        assert r.omnibus_test_name is not None
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        assert "Omnibus Test" in buf.getvalue()

    def test_p_values_false_still_computes_data_when_explicit(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        p_values=False, rng=1)
        assert r.show_p_values is False
        assert all(p.p_value is not None for p in r.pairwise)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Verdict reflects" not in out  # p-correction footnote suppressed

    def test_paired_path_p_values_omnibus_unaffected_by_none_default(self):
        # compare()'s own p_values=/omnibus= defaults changed from False to
        # None (a sentinel distinguishing "unset" from "explicitly False")
        # -- both are falsy, so paired-path behavior must be identical.
        rng = _rng(51)
        rows = []
        for m in ["A", "B", "C"]:
            for i in range(20):
                rows.append({"model": m, "item": i,
                             "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r_default = es.compare(evaldata, factors="model", metric="score")
        r_explicit_false = es.compare(evaldata, factors="model", metric="score",
                                       p_values=False, omnibus=False)
        assert r_default.to_dict() == r_explicit_false.to_dict()

    def test_unpaired_path_p_values_omnibus_default_matches_paired(self):
        # Both paths now share the same default (False) -- unset on the
        # unpaired path is a real no-op relative to explicit False, exactly
        # like the paired path's own p_values=/omnibus= defaults.
        evaldata = es.load_from(self._df())
        r_default = es.compare(evaldata, factors="model", metric="score",
                                design="unpaired", rng=1)
        r_explicit_false = es.compare(evaldata, factors="model", metric="score",
                                       design="unpaired", p_values=False, omnibus=False, rng=1)
        assert r_default.to_dict() == r_explicit_false.to_dict()


class TestCompareDesignRouting:
    def test_design_auto_on_paired_data_is_unchanged(self):
        rows = []
        rng = _rng(3)
        for m in ["A", "B"]:
            for i in range(30):
                rows.append({"model": m, "item": i, "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score")
        from evalstats.api import ComparisonResult
        assert isinstance(r, ComparisonResult)

    def test_design_auto_raises_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="between-subjects"):
            es.compare(evaldata, factors="model", metric="score")

    def test_design_unpaired_dispatches_to_group_comparison_result(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired")
        assert isinstance(r, GroupComparisonResult)

    def test_design_paired_forces_old_path_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        # Forcing the paired path on genuinely disjoint items must still hit the
        # existing (pre-existing, unchanged) has_missing crash -- not a new error.
        with pytest.raises(ValueError, match="NaN"):
            es.compare(evaldata, factors="model", metric="score", design="paired")

    def test_design_unpaired_not_supported_for_factorial(self):
        rng = _rng(4)
        rows = []
        for m in ["A", "B"]:
            for p in ["p1", "p2"]:
                for i in range(20):
                    rows.append({"model": m, "prompt": p, "item": f"{m}_{p}_{i}",
                                 "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="not supported"):
            es.compare(evaldata, factors=["model", "prompt"], metric="score", design="unpaired")

    def test_design_unpaired_not_supported_for_lmm(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="not supported"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired", method="lmm")

    def test_design_auto_exempt_for_lmm_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        # method="lmm" tolerates disjoint items natively -- design="auto" must not
        # raise the between-subjects ValueError for this call.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = es.compare(evaldata, factors="model", metric="score", method="lmm")
        from evalstats.api import ComparisonResult
        assert isinstance(r, ComparisonResult)

    def test_design_unpaired_with_secondary_metric_runs_pareto(self):
        # secondary_metric= is supported under design="unpaired" -- see
        # TestCompareUnpairedPareto for the full engine-level coverage; this
        # just confirms compare()'s own dispatch threads it through.
        rng = _rng(6)
        rows = []
        for m, mean in [("A", 0.4), ("B", 0.7)]:
            for i in range(30):
                rows.append({"model": m, "item": f"{m}_{i}",
                             "score": float(np.clip(rng.normal(mean, 0.15), 0, 1)),
                             "latency_ms": float(rng.normal(100, 10))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        secondary_metric={"latency_ms": "min"}, rng=6)
        assert r.pareto is not None

    def test_design_unpaired_with_multirun_data_not_supported(self):
        rng = _rng(8)
        rows = []
        for m in ["A", "B"]:
            for i in range(20):
                item_noise = rng.normal(0, 0.1)
                for run in range(3):
                    rows.append({
                        "model": m, "item": f"{m}_{i}", "run": run,
                        "score": float(np.clip(0.5 + (0.15 if m == "B" else 0.0) + item_noise + rng.normal(0, 0.05), 0, 1)),
                    })
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="multi-run"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired")

        # Single-run (R=1) slice of the same data should work fine -- the guard
        # only fires when run_col genuinely has >1 distinct value.
        df_single_run = df[df["run"] == 0].drop(columns=["run"])
        evaldata_single = es.load_from(df_single_run)
        r = es.compare(evaldata_single, factors="model", metric="score", design="unpaired")
        assert isinstance(r, GroupComparisonResult)

    def test_design_unpaired_with_alignment_end_to_end(self):
        df = _make_unpaired_with_alignment({"A": 0.35, "B": 0.65})
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = es.compare(evaldata, factors="model", metric="llm_score", design="unpaired",
                        alignment={"llm_score": ar})
        assert isinstance(r, GroupComparisonResult)
        assert r.ppi_applied is True
