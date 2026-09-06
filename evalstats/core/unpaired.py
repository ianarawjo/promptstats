"""Between-subjects (unpaired) comparison engine.

Sibling to ``core/paired.py``, not a branch inside it: the paired path's
entire machinery (``all_pairwise``, ``PairwiseMatrix``, ``PairedDiffResult``)
is built around item-matched differences (``per_input_diffs``), which has no
meaning for genuinely disjoint groups (e.g. different, unrelated reviewers
per app). This module implements the corresponding between-subjects
statistics as its own self-contained path, dispatched to from
``evalstats.api.compare()`` when ``design="unpaired"`` (or auto-detected),
and reuses the existing PPI-corrected test machinery in ``evalstats.tests``
as its statistical engine rather than reimplementing anything.

Two test families (see ``config.AUTO_UNPAIRED_METHOD_TABLE`` for the
decision and full rationale):

Every family reports the same estimand -- the **mean difference** between
two groups (a difference of proportions on binary data, which is the mean
of a 0/1 variable). Only the interval construction and the accompanying
test differ:

* **binary** data -- ``anova_oneway`` (omnibus, k>=3 only) + pairwise
  Agresti-Caffo intervals on Δp, with Welch's t-test supplying the p-value.
* **continuous / likert / grade** data -- ``kruskalwallis`` (omnibus,
  k>=3 only) + pairwise Welch t-intervals on the mean difference, with
  Mann-Whitney U -- Kruskal-Wallis's own post-hoc -- supplying the p-value.

Because Mann-Whitney tests θ=P(a>b) against 1/2 rather than the mean
difference the interval covers, the two can disagree; each pair therefore
also carries ``mean_test_p``, the interval's own p-value. This mirrors the
paired path, which has always reported mean differences and carried its
rank test (``PairedDiffResult.wilcoxon_p``) alongside.

At k=2 there is only one possible comparison, so there is no separate
omnibus test and no multiple-comparison correction to apply (Bonferroni/
Holm are no-ops at a family size of 1) -- the single pairwise result *is*
the answer. At k>=3, pairwise CIs get a Bonferroni correction and pairwise
p-values get a Shaffer correction, two independent axes mirroring how the
paired path separates its own simultaneous-CI and p-value-correction
machinery (see ``core/paired.py``'s ``_simultaneous_cis_router`` and
``correct_pvalues``).
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats as _scipy_stats
import pandas as pd

from evalstats.config import resolve_auto_unpaired_methods, get_alpha_ci
from evalstats.core.stats_utils import correct_pvalues
from evalstats.loader import _CANONICAL_ALIASES, _find_col, _detect_score_type

if TYPE_CHECKING:
    from evalstats.alignment import AlignmentResult

# Same literal value as evalstats.labeling.SYNTHETIC_ITEM_COL -- kept as an
# independent local constant rather than imported, since core/ modules
# shouldn't depend on the CLI-facing labeling module (wrong layering
# direction); it's a small, deliberate duplication of one string constant.
SYNTHETIC_ITEM_COL = "_row_item"


# ─────────────────────────────────────────────────────────────────────────────
# Result objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroupStat:
    """Descriptive stats + calibrated marginal CI for one group.

    Computed independently per group (own auto-detected data kind, own N,
    own CI method) via the same machinery ``evalstats.quick.summarize``
    uses -- there is no rectangular-design requirement here, so unbalanced
    group sizes are handled naturally.
    """
    label: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    method: str
    multi_ci: Optional[dict] = None  # {alpha: (lo, hi)} gradient CI bands


class _GroupStatsAsRobustness:
    """Minimal ``RobustnessResult``-compatible view over a ``list[GroupStat]``.

    Exists so ``core.summary._print_pareto_section`` -- built for the
    paired path's ``RobustnessResult`` (per-entity ``.labels``/``.mean``/
    ``.ci_low``/``.ci_high`` arrays) -- can render the between-subjects
    Pareto section unmodified: it only ever reads those four attributes, so
    this adapter is all that's needed to reuse the whole section (including
    the ASCII scatterplot) rather than reimplementing it.
    """

    def __init__(self, stats: list[GroupStat]):
        self.labels = [s.label for s in stats]
        self.mean = np.array([s.mean for s in stats])
        self.ci_low = np.array([s.ci_low for s in stats])
        self.ci_high = np.array([s.ci_high for s in stats])


@dataclass
class GroupDiffResult:
    """One pairwise between-subjects comparison.

    Not ``PairedDiffResult`` -- there is no ``per_input_diffs`` here, since
    the two groups' items have no correspondence to difference.
    """
    label_a: str
    label_b: str
    estimand: str           # always "mean_diff" -- see compare_unpaired's note
    null_value: float       # always 0.0 (mean_diff's null)
    point_estimate: float
    ci_low: float
    ci_high: float          # Bonferroni-corrected at k>=3; nominal alpha at k=2
    p_value: float          # Shaffer-corrected at k>=3; raw at k=2
    raw_p_value: float      # always uncorrected, for transparency
    n_a: int
    n_b: int
    mean_test_p: Optional[float] = None
    """Uncorrected p-value from the test that MATCHES ``ci_low``/``ci_high``
    (Welch's t, or its PPI-corrected form). ``p_value`` is the headline test,
    which for the rank_based family is Mann-Whitney U -- a different estimand
    from the reported mean difference, so it can disagree with the interval.
    This field makes the mean-difference decision inspectable alongside it.
    None for the binary family, where ``p_value`` already is the Welch p."""

    rho2: Optional[float] = None
    """Judge-human alignment governing THIS pair's PPI variance reduction --
    the squared correlation of the two influence functions, for the test
    actually run on this pair. Not the raw kappa/Spearman from the alignment
    report: those are score-level, this is test-specific. None when the
    comparison is uncorrected, or when the alignment call could not supply it."""

    n_eff: Optional[float] = None
    """Effective human-label count PER CONDITION for this pair: how many
    hand-labeled items per condition would have matched this pair's precision.

    judge_alignment returns n_eff against the TOTAL item count a correlation
    spans (``_pair_total_n`` sums across conditions for design="between"), so
    the stored value here is that total divided by the 2 conditions the pair
    spans. The omnibus figure divides by k instead. Getting that divisor wrong
    silently inflates the number by a factor of k/2, which is why the two are
    computed in one place (:func:`_ppi_label_efficiency`) rather than at each
    call site.

    This is the ORACLE bound, the efficiency available at the variance-
    minimizing lambda; the shipped test may realize less (see
    AlignmentResult's note on _attach_savings)."""

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= self.null_value <= self.ci_high)


class _GroupDiffResultsAsPairwiseMatrix:
    """Minimal ``PairwiseMatrix``-compatible view over a ``list[GroupDiffResult]``.

    Exists so ``core.summary``'s critical-difference-band and executive-
    summary machinery (``_critical_difference_groups``/
    ``_assign_significance_groups``/``_print_executive_summary``) -- built
    for the paired path's ``PairwiseMatrix`` (a ``.get(a, b)`` lookup plus
    ``.simultaneous_ci_method``) -- can render the between-subjects case
    unmodified. Those functions only ever read ``.get(a, b).point_diff``/
    ``.ci_low``/``.ci_high``, and check ``.simultaneous_ci_method is not
    None`` to decide whether significance is CI-exclusion-based rather than
    a p-value threshold -- the only branch reached here, since this
    engine's own ``ci_correction`` already *is* a simultaneous-CI scheme
    (Bonferroni), so ``simultaneous_ci_method`` is set to a matching
    sentinel and the p-value-threshold branch never fires.
    ``point_diff``/``ci_low``/``ci_high`` are the same null-shifted
    quantities the pairwise table itself displays (Δ/Δp -- the null is 0 for
    every family, so the shift is a no-op), so "CI excludes zero" means
    exactly what it already means there.
    """

    def __init__(self, pairwise: list["GroupDiffResult"]):
        self._by_pair: dict[tuple[str, str], tuple[float, float, float]] = {}
        for p in pairwise:
            self._by_pair[(p.label_a, p.label_b)] = (
                p.point_estimate - p.null_value, p.ci_low - p.null_value, p.ci_high - p.null_value,
            )
        self.simultaneous_ci_method = "bonferroni"  # any non-None sentinel -- see docstring

    def get(self, a: str, b: str):
        from types import SimpleNamespace
        if (a, b) in self._by_pair:
            point_diff, ci_low, ci_high = self._by_pair[(a, b)]
            return SimpleNamespace(point_diff=point_diff, ci_low=ci_low, ci_high=ci_high)
        if (b, a) in self._by_pair:
            point_diff, ci_low, ci_high = self._by_pair[(b, a)]
            return SimpleNamespace(point_diff=-point_diff, ci_low=-ci_high, ci_high=-ci_low)
        raise KeyError(f"no comparison found for ({a}, {b})")


@dataclass
class GroupComparisonResult:
    """Result of a between-subjects ``compare(design="unpaired")`` call.

    Deliberately a narrower reporting surface than ``ComparisonResult``
    (no forest-plot brackets) -- per-group means with gradient CIs, a
    pairwise comparison table (with critical-difference rank bands), the
    omnibus test when k>=3, an executive summary leaderboard, and a
    Pareto-front section when ``secondary_metric=`` was passed.
    """
    factor_col: str
    metric_col: str
    item_col: str
    item_col_synthetic: bool
    score_type: str          # "binary" | "continuous" | "likert" | "grade"
    family: str              # "binary_proportion" | "rank_based"
    groups: list[GroupStat]
    pairwise: list[GroupDiffResult]
    omnibus_test_name: Optional[str]     # None at k=2 -- no separate omnibus test
    omnibus_statistic: Optional[float]
    omnibus_p_value: Optional[float]              # uncorrected
    omnibus_corrected_p_value: Optional[float]     # PPI-corrected, when alignment given
    alpha: float
    n_pairs: int
    ci_correction: str        # "bonferroni" or "none" (k=2, single comparison)
    pvalue_correction: str    # "shaffer" or "none" (k=2, single comparison)
    ppi_applied: bool
    alignment_result: Optional["AlignmentResult"] = None
    omnibus_rho2: Optional[float] = None
    """Whole-design judge-human alignment for the omnibus test, when one ran
    and PPI is applied. Not decomposable into the pairwise values: the omnibus
    correlation is defined across all conditions at once."""
    omnibus_n_eff: Optional[float] = None
    """Effective human labels PER CONDITION for the omnibus test (the total
    judge_alignment returns, divided by the k conditions it spans)."""
    n_lab_per_condition: Optional[float] = None
    """Mean human labels actually collected per condition, for the
    "N_eff against what you collected" comparison the summary prints."""
    marginal_n_eff: Optional[list] = None
    """Per-group effective label count for the MARGINAL mean CIs, in `groups`
    order. A group's marginal mean spans only itself, so this needs no
    per-condition division. None unless every group produced one."""
    show_p_values: bool = True
    pareto: Optional[dict] = None

    # ── convenience accessors ──────────────────────────────────────────────

    @property
    def labels(self) -> list[str]:
        return [g.label for g in self.groups]

    @property
    def pareto_status(self) -> Optional[dict]:
        """Per-group three-state Pareto classification, or ``None``.

        Populated only when ``compare(design="unpaired", secondary_metric=...)``
        was passed. Mirrors :attr:`~evalstats.api.ComparisonResult.pareto_status`
        exactly -- keys are group labels, values are
        :class:`~evalstats.core.pareto.ParetoStatus`.
        """
        return self.pareto["statuses"] if self.pareto is not None else None

    @property
    def pareto_frontier_probability(self) -> Optional[dict]:
        """Per-group ``P(group is Pareto-optimal)``, or ``None``.

        Populated only when ``compare(design="unpaired", secondary_metric=...)``
        was passed. Mirrors
        :attr:`~evalstats.api.ComparisonResult.pareto_frontier_probability`.
        """
        if self.pareto is None:
            return None
        result = self.pareto["result"]
        return dict(zip(result.labels, result.p_frontier.tolist()))

    def _group(self, label: str) -> GroupStat:
        for g in self.groups:
            if g.label == label:
                return g
        raise KeyError(f"no group {label!r}; available: {self.labels}")

    def _pair(self, label_a: str, label_b: str) -> GroupDiffResult:
        for p in self.pairwise:
            if {p.label_a, p.label_b} == {label_a, label_b}:
                return p
        raise KeyError(f"no pairwise result for ({label_a!r}, {label_b!r})")

    # ── reporting ───────────────────────────────────────────────────────────

    def summary(self, *, verbose: bool = False) -> None:
        """Print the comparison report.

        ``verbose=True`` adds the qualifications behind the label-efficiency
        columns: that N_eff is the best case at the variance-minimizing lambda,
        and that a rank-based rho^2 is tied to this dataset's effect size.
        """
        from evalstats.core.summary import print_group_comparison_summary
        print_group_comparison_summary(self, verbose=verbose)

    def plot(self, **kwargs):
        raise NotImplementedError(
            "GroupComparisonResult.plot() is not implemented yet -- between-"
            "subjects plotting is scoped for a later phase. Use .summary() "
            "or .to_frame() in the meantime."
        )

    def to_dict(self) -> dict:
        return {
            "design": "unpaired",
            "factor_col": self.factor_col,
            "metric_col": self.metric_col,
            "item_col": self.item_col,
            "item_col_synthetic": self.item_col_synthetic,
            "score_type": self.score_type,
            "family": self.family,
            "alpha": self.alpha,
            "ppi_applied": self.ppi_applied,
            "groups": {
                g.label: {
                    "n": g.n, "mean": g.mean, "ci_low": g.ci_low,
                    "ci_high": g.ci_high, "method": g.method,
                }
                for g in self.groups
            },
            "omnibus": (
                None if self.omnibus_test_name is None else {
                    "test_name": self.omnibus_test_name,
                    "statistic": self.omnibus_statistic,
                    "p_value": self.omnibus_p_value,
                    "corrected_p_value": self.omnibus_corrected_p_value,
                }
            ),
            "pairwise": [
                {
                    "a": p.label_a, "b": p.label_b, "estimand": p.estimand,
                    "point_estimate": p.point_estimate,
                    "ci_low": p.ci_low, "ci_high": p.ci_high,
                    "p_value": p.p_value, "raw_p_value": p.raw_p_value,
                    "mean_test_p": p.mean_test_p,
                    "significant": p.significant,
                    "n_a": p.n_a, "n_b": p.n_b,
                }
                for p in self.pairwise
            ],
            "ci_correction": self.ci_correction,
            "pvalue_correction": self.pvalue_correction,
            **self._pareto_to_dict(),
        }

    def _pareto_to_dict(self) -> dict:
        if self.pareto is None:
            return {}
        pareto_groups: dict[str, dict] = {}
        p_frontier = self.pareto_frontier_probability
        for label, st in self.pareto["statuses"].items():
            entry: dict = {"status": st.status, "p_pareto_optimal": float(p_frontier[label])}
            if st.dominated_by:
                entry["dominated_by"] = list(st.dominated_by)
            if st.ambiguous_vs:
                entry["ambiguous_vs"] = list(st.ambiguous_vs)
            pareto_groups[str(label)] = entry
        return {
            "pareto": {
                "secondary_metric": self.pareto["secondary_metric"],
                "direction": self.pareto["direction"],
                "groups": pareto_groups,
            }
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per pairwise comparison."""
        rows = [
            {
                "a": p.label_a, "b": p.label_b, "estimand": p.estimand,
                "point_estimate": p.point_estimate,
                "ci_low": p.ci_low, "ci_high": p.ci_high,
                "p_value": p.p_value, "raw_p_value": p.raw_p_value,
                "mean_test_p": p.mean_test_p,
                "significant": p.significant,
                "n_a": p.n_a, "n_b": p.n_b,
            }
            for p in self.pairwise
        ]
        return pd.DataFrame(rows)

    def groups_to_frame(self) -> pd.DataFrame:
        """One row per group (descriptive stats)."""
        rows = [
            {"label": g.label, "n": g.n, "mean": g.mean,
             "ci_low": g.ci_low, "ci_high": g.ci_high, "method": g.method}
            for g in self.groups
        ]
        return pd.DataFrame(rows).set_index("label")


class _GroupComparisonResultAsBundle:
    """Minimal ``AnalysisBundle``-compatible view over a
    ``GroupComparisonResult``, so ``core.summary._print_executive_summary``
    (built for the paired path) can render the between-subjects executive
    summary leaderboard unmodified. It only ever reads ``.labels``,
    ``.robustness.{mean,ci_low,ci_high}``, ``.pairwise`` (a
    ``PairwiseMatrix``-compatible lookup), ``.seed_variance`` (always
    ``None`` here -- no run/seed axis exists for between-subjects data, by
    construction: ``design="unpaired"`` refuses multi-run data outright),
    and ``.resolved_ci_method`` (only to decide the "Wilson-flat CI" column
    header).
    """

    def __init__(self, result: "GroupComparisonResult"):
        self.labels = list(result.labels)
        self.robustness = _GroupStatsAsRobustness(result.groups)
        self.pairwise = _GroupDiffResultsAsPairwiseMatrix(result.pairwise)
        self.seed_variance = None
        self.resolved_ci_method = result.groups[0].method if result.groups else None


# ─────────────────────────────────────────────────────────────────────────────
# FWER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bonferroni_alpha(alpha: float, n_pairs: int) -> float:
    """Bonferroni-adjusted alpha for a family of n_pairs comparisons.

    Deliberately Bonferroni, not Šidák: Šidák's exactness needs (near-)
    independence between the pairwise statistics, which is unverified for
    this bootstrap's dependence structure (pairs sharing a group are
    correlated). Bonferroni's union bound holds regardless.
    """
    return alpha if n_pairs <= 1 else alpha / n_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise engines -- one PPI + one non-PPI function per family. Both take a
# list of group arrays (any k>=2 -- Bonferroni/Holm no-op at n_pairs=1, so
# the k=2 case doesn't need separate code) and return a dict with "pairs",
# a point-estimate array, "ci_lo"/"ci_hi", and "pair_p" (uncorrected).
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_pairwise_ppi(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], alpha: float, n_boot: int, rng,
) -> dict:
    """PPI-corrected mean difference mean(a) - mean(b) for every pair, with a
    PPI-corrected Mann-Whitney U p-value alongside.

    The interval comes from :func:`evalstats.tests._ppi_two_sample_t_interval`
    -- the closed-form independent-groups mean-difference correction, i.e. the
    same construction the binary family uses, which is a mean of a 0/1 variable
    and needs no separate machinery. It is scale-agnostic, so a 1-5 Likert
    outcome needs no ``score_range`` for the pairwise interval to be on the
    right scale (the *marginal* group CIs do use the range -- see
    ``_compute_group_stats``).

    ``pair_p`` stays the PPI-corrected Mann-Whitney U p-value -- the post-hoc
    that follows the Kruskal-Wallis omnibus above it, and the one this
    project's PPI work validates. It tests theta = P_mid(a>b) against 1/2,
    which is NOT the estimand the interval covers, so ``mean_test_p`` carries
    the interval's own p-value for comparison.
    """
    from evalstats.tests import (
        _ppi_two_sample, _ppi_two_sample_t_interval, _p_x_gt_y_midrank,
    )

    def _auc_shifted(xa, ya):
        return _p_x_gt_y_midrank(xa, ya) - 0.5

    rng = np.random.default_rng(rng)
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs)); ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs)); pair_p = np.empty(len(pairs))
    mean_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        t = _ppi_two_sample_t_interval(
            groups[a], groups[b], groups_lab[a], groups_lab[b], alpha,
        )
        point[idx] = float(t.estimate)
        ci_lo[idx] = float(t.ci_low)
        ci_hi[idx] = float(t.ci_high)
        mean_p[idx] = float(t.p_value)
        u = _ppi_two_sample(groups[a], groups[b], groups_lab[a], groups_lab[b],
                            _auc_shifted, alpha, n_boot, rng)
        pair_p[idx] = 1.0 if u.p_value is None else float(u.p_value)
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pair_p": pair_p, "mean_test_p": mean_p}


def _numeric_pairwise_uncorrected(groups: list[np.ndarray], alpha: float) -> dict:
    """Non-PPI analog: Welch's t-interval on the mean difference, with a
    Mann-Whitney U p-value alongside.

    Welch rather than Student throughout: between-subjects groups routinely
    differ in both size and variance, and Welch costs almost nothing when the
    variances happen to match (Delacre, Lakens & Leys 2017; Ruxton 2006). It
    is also what ``cases/ci_unpaired.py`` measured as the safe default across
    both continuous and Likert real corpora -- ``mover_logit_t`` scores
    slightly better on Likert alone, but Welch is one method for all numeric
    data and does not degrade on ceiling-saturated continuous shapes, where
    MOVER constructions fall to ~0.72 coverage.
    """
    from scipy.stats import ttest_ind
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs)); ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs)); pair_p = np.empty(len(pairs))
    mean_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        r = ttest_ind(groups[a], groups[b], equal_var=False)
        ci = r.confidence_interval(confidence_level=1.0 - alpha)
        point[idx] = float(np.mean(groups[a]) - np.mean(groups[b]))
        ci_lo[idx] = float(ci.low)
        ci_hi[idx] = float(ci.high)
        mean_p[idx] = float(r.pvalue)
        # Post-hoc after the Kruskal-Wallis omnibus = the Mann-Whitney U test,
        # which is what a reader expects there and can reproduce in scipy.
        pair_p[idx] = float(_scipy_stats.mannwhitneyu(
            groups[a], groups[b], alternative="two-sided").pvalue)
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pair_p": pair_p, "mean_test_p": mean_p}


def _binary_pairwise_ppi(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], alpha: float, power_tune: bool = True,
) -> dict:
    """Δp = mean(a) - mean(b) for every pair, PPI-corrected via the exact
    closed-form construction ``ttest()`` itself uses for the independent,
    labeled case (``_ppi_two_sample_t_interval``) -- called directly
    (bypassing the public wrapper) the same way the rank-based family calls
    ``_ppi_kruskal_wallis_pairwise`` directly, so both families are handled
    consistently and neither changes ``evalstats.tests``' public contract.
    """
    from evalstats.tests import _ppi_two_sample_t_interval
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs))
    ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs))
    pair_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        res = _ppi_two_sample_t_interval(
            groups[a], groups[b], groups_lab[a], groups_lab[b], alpha, power_tune=power_tune,
        )
        point[idx] = res.estimate
        ci_lo[idx] = res.ci_low
        ci_hi[idx] = res.ci_high
        pair_p[idx] = res.p_value
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pair_p": pair_p, "mean_test_p": None}


def _agresti_caffo_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Agresti & Caffo (2000) interval for a difference of two independent
    proportions, The American Statistician 54(4):280-288.

    Add one success and one failure to EACH arm, then apply the plain Wald
    formula to the adjusted counts -- the two-sample analogue of the
    Agresti-Coull single-proportion adjustment, and a one-line change to Wald
    that removes most of Wald's small-sample undercoverage.

    Replaces a Welch t-interval on the raw 0/1 scores. Exact coverage
    enumerated over every (k_A, k_B) table (simulations/harness/cases/
    ci_unpaired.py's exact mode, no Monte Carlo) puts Welch's worst case at
    0.641 -- reached at p near 0 or 1, which is where binary eval data
    actually sits -- against 0.930 here, at a *narrower* mean width (0.472 vs
    0.511) and half the runtime. Agresti-Min is the only method in that
    comparison never dipping below nominal, but costs ~600x the time and
    roughly half the power, so it is not the default.
    """
    a_bin = (np.asarray(a, dtype=float) >= 0.5).astype(float)
    b_bin = (np.asarray(b, dtype=float) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa = (float(np.sum(a_bin)) + 1.0) / (na + 2.0)
    pb = (float(np.sum(b_bin)) + 1.0) / (nb + 2.0)
    se = float(np.sqrt(pa * (1.0 - pa) / (na + 2.0) + pb * (1.0 - pb) / (nb + 2.0)))
    z = float(_scipy_stats.norm.ppf(1.0 - alpha / 2.0))
    d = pa - pb
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _binary_pairwise_uncorrected(groups: list[np.ndarray], alpha: float) -> dict:
    """Non-PPI analog: Agresti-Caffo interval per pair, with Welch's t-test
    supplying the p-value.

    The point estimate stays the raw difference of proportions -- the +1/+1
    adjustment is a variance-stabilising device for the *interval*, and
    reporting the shrunken proportion as the effect would misstate the
    observed difference. The interval is therefore very slightly off-centre
    relative to the point estimate, which is expected and standard.
    """
    from scipy.stats import ttest_ind
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs))
    ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs))
    pair_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        r = ttest_ind(groups[a], groups[b], equal_var=False)
        point[idx] = float(np.mean(groups[a]) - np.mean(groups[b]))
        ci_lo[idx], ci_hi[idx] = _agresti_caffo_ci(groups[a], groups[b], alpha)
        pair_p[idx] = float(r.pvalue)
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pair_p": pair_p, "mean_test_p": None}


# ─────────────────────────────────────────────────────────────────────────────
# Per-group descriptive stats
# ─────────────────────────────────────────────────────────────────────────────

def _compute_group_stats(
    labels: list[str], arrays: list[np.ndarray], *, alpha: float, n_bootstrap: int, rng,
    score_range: Optional[tuple[float, float]] = None,
    lab_arrays: Optional[list[np.ndarray]] = None,
) -> list[GroupStat]:
    """Per-group mean + calibrated marginal CI (with gradient multi_ci
    bands), computed independently per group -- the exact same building
    block ``evalstats.quick.summarize`` uses internally, called directly
    here (with multi_ci=True, which summarize()'s own public signature
    doesn't expose) rather than through that quick-primitive wrapper.

    ``lab_arrays``, when given (PPI alignment is active), makes this a PPI-
    corrected marginal mean per group instead of a raw one -- mirroring
    ``evalstats.api._run_alignment_ppi``'s own single-sample correction
    exactly (same ``resolve_auto_robustness_method`` -> data kind ->
    ``resolve_ppi_auto_methods`` -> ``_ppi_robustness_dispatch`` chain, with
    the resolved score range forwarded to the dispatch for the
    scale-dependent methods, same ``GRADIENT_CI_ALPHAS`` sweep for the
    gradient bands), just applied per between-subjects group instead of per
    paired-path entity. Every group is
    guaranteed to have at least one label by this point -- the caller
    (``compare_unpaired``) already validates that and raises before this is
    ever reached otherwise -- so there is no paired-path-style "entity has
    zero labels, keep its uncorrected estimate" fallback needed here.
    """
    from evalstats.core.router import resolve_auto_robustness_method
    from evalstats.core.variance import robustness_metrics

    ppi_applied = lab_arrays is not None
    if ppi_applied:
        from evalstats.config import resolve_ppi_auto_methods, GRADIENT_CI_ALPHAS
        # A single import here, not per-call to compare_unpaired -- avoids the
        # api.py <-> core/unpaired.py circular import (api.py imports
        # compare_unpaired from this module at module scope).
        from evalstats.api import _ppi_robustness_dispatch

        # Resolve the data kind through the router -- the SAME resolution the
        # non-PPI branch below already uses -- rather than a local
        # binary/bounded_01/unbounded test. That local test had no "likert"
        # branch and ignored score_range, so discrete/ordinal data on e.g. a
        # 1-5 scale fell through to "unbounded" and silently took
        # ppi_t_interval, leaving PPI_AUTO_METHOD_TABLE's "likert" row
        # (ppi_logit_t) unreachable here. Resolved ONCE on the pooled scores,
        # not per group, so every group gets the same method and the group
        # CIs stay comparable to each other.
        #
        # ppi_score_range must then be forwarded to the dispatch: ppi_logit_t
        # is scale-DEPENDENT, and defaulting its bounds to (0, 1) on, say, a
        # 1-5 scale returns a CI on the wrong scale entirely (0% coverage,
        # not a subtle miscalibration). See evalstats.api's matching fix.
        pooled = np.concatenate(arrays).reshape(1, -1)
        _, _, ppi_score_range, data_kind = resolve_auto_robustness_method(
            pooled, score_range=score_range, stacklevel=4,
        )
        _, ppi_robustness_method = resolve_ppi_auto_methods(data_kind)

    out = []
    for i, (label, arr) in enumerate(zip(labels, arrays)):
        if ppi_applied:
            lab_arr = lab_arrays[i]
            res = _ppi_robustness_dispatch(ppi_robustness_method, arr, lab_arr, alpha, n_bootstrap, rng, ppi_score_range)
            multi_ci = {}
            for a in GRADIENT_CI_ALPHAS:
                g = _ppi_robustness_dispatch(ppi_robustness_method, arr, lab_arr, a, n_bootstrap, rng, ppi_score_range)
                multi_ci[a] = (float(g.ci_low), float(g.ci_high))
            out.append(GroupStat(
                label=label, n=int(arr.size), mean=float(res.estimate), std=float(np.std(arr)),
                ci_low=float(res.ci_low), ci_high=float(res.ci_high),
                method=ppi_robustness_method, multi_ci=multi_ci,
            ))
            continue

        a2d = arr.reshape(1, -1)
        _, robustness_method, resolved_score_range, _ = resolve_auto_robustness_method(
            a2d, score_range=score_range, stacklevel=4,
        )
        rob = robustness_metrics(
            a2d, ["_"],
            n_bootstrap=n_bootstrap, rng=rng, alpha=alpha,
            statistic="mean", marginal_method=robustness_method,
            multi_ci=True, score_range=resolved_score_range,
        )
        multi_ci = (
            {a: (float(lo[0]), float(hi[0])) for a, (lo, hi) in rob.multi_ci.items()}
            if rob.multi_ci is not None else None
        )
        out.append(GroupStat(
            label=label, n=int(arr.size), mean=float(rob.mean[0]), std=float(rob.std[0]),
            ci_low=float(rob.ci_low[0]) if rob.ci_low is not None else float("nan"),
            ci_high=float(rob.ci_high[0]) if rob.ci_high is not None else float("nan"),
            method=robustness_method, multi_ci=multi_ci,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_EFFICIENCY_TESTS = {
    # family -> (omnibus test, pairwise test), in judge_alignment's vocabulary.
    # These MUST track what compare_unpaired actually runs a few hundred lines
    # below: anova_oneway/Welch t for binary, Kruskal-Wallis/Mann-Whitney for
    # the rank_based family. A mismatch would report the efficiency of a test
    # the user never ran, which is worse than reporting nothing.
    "binary_proportion": ("anova_oneway", "ttest"),
    "rank_based": ("kruskalwallis", "mannwhitney"),
}


def _ppi_label_efficiency(labels, group_arrays, group_lab_arrays, family):
    """Judge-human alignment and effective label count for the tests just run.

    Returns ``(omnibus_rho2, omnibus_n_eff, {(a, b): (rho2, n_eff)})``, with
    every n_eff expressed PER CONDITION. Any element may be None: this is a
    reporting extra, so a failure here must never take down a comparison that
    otherwise succeeded.

    Why judge_alignment is called again here rather than reusing the caller's
    AlignmentResult: that one is score-level (kappa, Pearson, Spearman on the
    raw scores) and carries no n_eff at all unless the caller happened to pass
    ``test=``, which the documented workflow does not. The number that governs
    a PPI variance reduction is the correlation of the two INFLUENCE functions
    for the specific test, so it has to be requested per test. Form 3 takes the
    same (judge, human) arrays already in hand.

    n_eff arrives as a total over the conditions a correlation spans (see
    ``_pair_total_n``), so it is divided by k for the omnibus and by 2 for each
    pair. That divisor is the whole reason this lives in one function.
    """
    tests = _EFFICIENCY_TESTS.get(family)
    if tests is None or len(labels) < 2:
        return None, None, {}
    omnibus_test, pairwise_test = tests
    conds = {
        str(lbl): (np.asarray(g, dtype=float), np.asarray(lab, dtype=float))
        for lbl, g, lab in zip(labels, group_arrays, group_lab_arrays)
    }
    k = len(conds)

    def _call(test):
        from evalstats.alignment import judge_alignment
        # Suppressed for the same reason the omnibus call above is: this
        # constructs its own AlignmentResult with no selection=, and letting it
        # print would drop a second, worse-disclosed alignment report into the
        # middle of ours.
        with contextlib.redirect_stdout(io.StringIO()):
            return judge_alignment(conds, design="between", test=test,
                                   selection="random", ci=False)

    om_rho2 = om_neff = None
    if k >= 3:
        try:
            m = _call(omnibus_test).omnibus_metric
            if m is not None:
                om_rho2 = float(m["estimate"]) ** 2
                om_neff = float(m["n_eff"]) / k
        except Exception:
            pass

    pairs = {}
    try:
        pm = _call(pairwise_test).test_pairwise_metrics or {}
        for (a, b), m in pm.items():
            pairs[(str(a), str(b))] = (float(m["estimate"]) ** 2, float(m["n_eff"]) / 2)
    except Exception:
        pairs = {}
    return om_rho2, om_neff, pairs


def compare_unpaired(
    df: pd.DataFrame,
    *,
    factor_col: str,
    metric_col: str,
    item_col: Optional[str] = None,
    alignment: Optional[dict] = None,
    alpha: Optional[float] = None,
    n_boot: int = 2000,
    rng=None,
    score_range: Optional[tuple[float, float]] = None,
    p_values: bool = True,
    omnibus: bool = True,
    secondary_metric: Optional[dict] = None,
) -> GroupComparisonResult:
    """Between-subjects comparison engine -- see module docstring.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with at least ``factor_col`` and ``metric_col``.
    factor_col : str
        Column identifying which group each row belongs to.
    metric_col : str
        Numeric score column to compare.
    item_col : str, optional
        Row/item identifier column. When not given, auto-detected via the
        same canonical aliases ``load_from()`` uses; when none of those
        match either, a synthetic positional id is used (each row is its
        own item) -- between-subjects data commonly has no natural item id
        at all (e.g. just group + rating, no reviewer id).
    alignment : dict, optional
        ``{metric_col: AlignmentResult}``, matching ``compare()``'s own
        ``alignment=`` convention exactly -- the caller has already run
        ``judge_alignment()``; this just consumes the result (splitting its
        human-label column by group) and displays it inline.
    alpha : float, optional
        Significance level. Defaults to :func:`evalstats.get_alpha_ci`.
    n_boot : int
        Bootstrap resamples for the rank-based family's pairwise CIs
        (unused by the binary family, which is closed-form).
    rng : optional
        Seed or ``np.random.Generator``.
    score_range : (float, float), optional
        Explicit metric bounds (e.g. ``(1, 5)`` for a Likert scale), passed
        through to the per-group marginal CI's auto-method resolution
        (matches ``compare()``'s own ``score_range=`` engine kwarg). When
        ``None`` (default), bounds are auto-detected per group, same as
        the paired path's own default.
    p_values : bool
        Whether ``.summary()`` prints the pairwise table's p-value column
        (and the p-value-correction footnote at k>=3). Defaults to
        ``True`` here (an unpaired-specific default, deliberately not
        ``compare()``'s own ``False`` -- p-values are core to reading this
        narrower report, not an opt-in extra). The underlying
        ``GroupDiffResult.p_value``/``raw_p_value`` fields are always
        computed and available via ``.to_dict()``/``.to_frame()``
        regardless of this flag; it only controls console display.
    omnibus : bool
        Whether the omnibus test (Kruskal-Wallis/ANOVA, k>=3 only) is run
        at all. Defaults to ``True`` here (again, not ``compare()``'s own
        ``False``). When ``False``, ``omnibus_test_name`` and friends stay
        ``None`` even at k>=3 -- unlike ``p_values``, this skips the
        *computation*, not just the display, mirroring the paired path's
        own ``if omnibus and len(labels) >= 3:`` gate.
    secondary_metric : dict, optional
        ``{column_name: "min" | "max"}``, matching ``compare()``'s own
        ``secondary_metric=`` convention exactly. Runs an uncertainty-aware
        Pareto-front analysis between ``metric_col`` and this second column,
        via :func:`~evalstats.core.pareto.pareto_bootstrap_unpaired` -- a
        per-group joint bootstrap (each group's own rows resampled
        together, preserving the row-level primary/secondary correlation),
        not the paired path's shared-item-index bootstrap (there's no
        shared item pool to preserve correlation across between disjoint
        groups). Populates :attr:`GroupComparisonResult.pareto`/
        ``.pareto_status``/``.pareto_frontier_probability``.

    Returns
    -------
    GroupComparisonResult
    """
    if factor_col not in df.columns:
        raise ValueError(f"factor_col {factor_col!r} not found in data.")
    if metric_col not in df.columns:
        raise ValueError(f"metric_col {metric_col!r} not found in data.")

    secondary_col = None
    secondary_direction = None
    if secondary_metric is not None:
        if not isinstance(secondary_metric, dict) or len(secondary_metric) != 1:
            raise ValueError(
                "secondary_metric= must be a dict with exactly one entry, "
                "e.g. secondary_metric={'latency_ms': 'min'}."
            )
        (secondary_col, secondary_direction), = secondary_metric.items()
        if secondary_direction not in ("min", "max"):
            raise ValueError(
                f"secondary_metric={{{secondary_col!r}: {secondary_direction!r}}} -- "
                "direction must be 'min' or 'max'."
            )
        if secondary_col not in df.columns:
            raise ValueError(f"secondary_metric column {secondary_col!r} not found in data.")

    if alpha is None:
        alpha = get_alpha_ci()
    rng = np.random.default_rng(rng)

    resolved_item = item_col or _find_col(df, _CANONICAL_ALIASES["item"])
    item_synthetic = resolved_item is None
    if item_synthetic:
        resolved_item = SYNTHETIC_ITEM_COL
    elif resolved_item not in df.columns:
        raise ValueError(f"item_col {resolved_item!r} not found in data.")

    groups_df = dict(tuple(df.groupby(factor_col, sort=False)))
    labels = [str(k) for k in groups_df.keys()]
    if len(labels) < 2:
        raise ValueError(
            f"factor_col {factor_col!r} has only {len(labels)} distinct value(s) -- "
            "need at least 2 groups to compare."
        )

    score_type = _detect_score_type(df[metric_col].dropna())
    family, _, _ = resolve_auto_unpaired_methods(score_type)

    # A judged SECONDARY metric is not supported and must not fail quietly.
    # PPI reaches the primary metric only: the Pareto joint bootstrap
    # (core.pareto.pareto_bootstrap_unpaired) takes no labels, and the
    # secondary metric's own marginal CIs are computed without them -- so an
    # alignment entry for the secondary column would be accepted and then
    # silently ignored, reporting uncorrected frontier probabilities as if
    # they were corrected. Refuse instead. The common case (a cost/latency
    # secondary, which has no judge and needs no correction) is unaffected.
    if alignment is not None and secondary_col and secondary_col in alignment:
        raise ValueError(
            f"secondary_metric={secondary_col!r} also has a judge-alignment "
            f"entry, but PPI correction of a secondary metric is not supported: "
            f"the Pareto frontier bootstrap and the secondary metric's CIs are "
            f"computed on raw scores, so the correction would be silently "
            f"dropped. Pass alignment for the primary metric "
            f"({metric_col!r}) only -- a secondary metric measured without a "
            f"judge (cost, latency, length) needs no correction and works as-is."
        )

    ppi_applied = alignment is not None and metric_col in alignment
    alignment_result = alignment[metric_col] if ppi_applied else None
    human_col = None
    if ppi_applied:
        human_col = alignment_result.human_col
        if human_col not in df.columns:
            raise ValueError(
                f"alignment result's human_groundtruth column {human_col!r} "
                "not found in data."
            )

    # Build group_arrays and (if PPI) group_lab_arrays from the SAME per-group
    # slice so a dropped NaN-score row drops its label in lockstep -- keeping
    # positional alignment between the two, which the PPI machinery below
    # requires. Drop NaN per group with a warning (don't silently produce a
    # NaN-poisoned CI/omnibus stat, or crash deep inside a closed-form CI
    # helper) -- matches evalstats.quick._clean_1d's own drop-and-warn
    # convention for flat, unstructured score lists. Unlike the paired path's
    # hard-error on missing cells (which protects item *alignment*, a concern
    # that doesn't exist here -- there's no cross-group pairing to break).
    # When secondary_metric= is given, a row is only usable if BOTH metrics
    # are present -- the row-level (primary, secondary) pairing is exactly
    # what the Pareto joint bootstrap needs preserved, so both arrays must
    # drop the same rows in lockstep, not be cleaned independently.
    import warnings as _warnings
    group_arrays: list[np.ndarray] = []
    group_lab_arrays: Optional[list[np.ndarray]] = [] if ppi_applied else None
    secondary_arrays: Optional[list[np.ndarray]] = [] if secondary_col else None
    for lbl, key in zip(labels, groups_df.keys()):
        sub = groups_df[key]
        scores = sub[metric_col].to_numpy(dtype=float)
        is_nan = np.isnan(scores)
        sec_scores = None
        if secondary_col:
            sec_scores = sub[secondary_col].to_numpy(dtype=float)
            is_nan = is_nan | np.isnan(sec_scores)
        n_missing = int(is_nan.sum())
        if n_missing > 0:
            _warnings.warn(
                f"group {lbl!r}: dropped {n_missing} NaN (missing) value(s) out of "
                f"{scores.size}; computed from the remaining {scores.size - n_missing}.",
                UserWarning, stacklevel=4,
            )
            scores = scores[~is_nan]
        if scores.size == 0:
            raise ValueError(f"group {lbl!r} has no valid (non-NaN) scores.")
        group_arrays.append(scores)
        if ppi_applied:
            group_lab_arrays.append(sub[human_col].to_numpy(dtype=float)[~is_nan])
        if secondary_col:
            secondary_arrays.append(sec_scores[~is_nan])

    if ppi_applied:
        zero_labeled = [
            lbl for lbl, labs in zip(labels, group_lab_arrays) if np.all(np.isnan(labs))
        ]
        if zero_labeled:
            raise ValueError(
                f"Group(s) {zero_labeled!r} have zero labeled items. Every group "
                "needs at least one human label for PPI correction -- otherwise "
                "its rectifier term is undefined and any comparison touching it "
                "degenerates to a point estimate at the null with no real signal."
            )
        # Same minimum-label-count enforcement (>=15 total human labels,
        # warn below 30) every other PPI caller in evalstats.tests goes
        # through -- unpaired.py calls the *private* pairwise engines below
        # directly (bypassing the public ttest()/kruskalwallis()/etc.
        # wrappers, which do this themselves), so it must sanitize here or
        # the correction can silently run on too few labels (a near-zero-
        # width, spuriously confident CI) or crash on a zero-labeled group.
        from evalstats.tests import _sanitize_multigroup_ppi_labels
        group_lab_arrays = _sanitize_multigroup_ppi_labels(
            group_arrays, group_lab_arrays, repeated=False,
            test_label="between-subjects comparison",
        )

    group_stats = _compute_group_stats(
        labels, group_arrays, alpha=alpha, n_bootstrap=n_boot, rng=rng,
        score_range=score_range,
        lab_arrays=group_lab_arrays if ppi_applied else None,
    )

    k = len(labels)
    n_pairs = k * (k - 1) // 2
    ci_alpha = _bonferroni_alpha(alpha, n_pairs)

    # ── Omnibus test (k>=3 only -- at k=2 there's nothing to protect against,
    # the single pairwise comparison already answers the whole question) ────
    #
    # Suppressed stdout: every evalstats.tests function with labels given
    # unconditionally prints its own alignment report via _run_alignment_
    # report -- NOT gated by print_result (that only gates the TestResult's
    # own .summary()). That internal report re-runs judge_alignment() from
    # scratch with no selection= (always "unknown"), which would print a
    # second, worse, differently-labeled alignment report right in the
    # middle of ours -- we already print the caller's real, correctly-
    # disclosed AlignmentResult once via the PPI banner above. Left as-is
    # in evalstats.tests itself (existing, validated, widely-used behavior
    # for direct callers of that module -- not something to change here);
    # suppressed only at this call site.
    omnibus_test_name = omnibus_statistic = omnibus_p_value = omnibus_corrected_p_value = None
    if k >= 3 and omnibus:
        with contextlib.redirect_stdout(io.StringIO()) if ppi_applied else contextlib.nullcontext():
            if family == "binary_proportion":
                from evalstats.tests import anova_oneway
                om = anova_oneway(
                    *group_arrays, groups_lab=group_lab_arrays if ppi_applied else None,
                    alpha=alpha, n_boot=n_boot, rng=rng, print_result=False,
                )
                omnibus_test_name = "One-way ANOVA (independent)"
            else:
                from evalstats.tests import kruskalwallis
                om = kruskalwallis(
                    *group_arrays, groups_lab=group_lab_arrays if ppi_applied else None,
                    alpha=alpha, n_boot=n_boot, rng=rng, print_result=False,
                )
                omnibus_test_name = "Kruskal-Wallis test"
        omnibus_statistic = float(om.statistic)
        omnibus_p_value = float(om.p_value)
        omnibus_corrected_p_value = (
            float(om.corrected_p_value) if om.corrected_p_value is not None else None
        )

    # ── Pairwise table (all k>=2 -- Bonferroni/Holm no-op at n_pairs=1) ─────
    #
    # ONE estimand for every family: the mean difference (a difference of
    # proportions for binary data, which is the same thing on a 0/1 variable).
    # The rank_based family previously reported theta = P(a>b) + .5 P(a=b)
    # here, because no validated mean-difference post-hoc existed for
    # between-subjects data; simulations/harness/cases/ci_unpaired.py is that
    # validation, so the reason no longer holds. Reporting a dominance
    # probability also made this the only surface in evalstats stated in
    # something other than a mean -- the paired path has always reported mean
    # differences and carried its rank test (Wilcoxon) alongside as a
    # supplementary p-value. This mirrors that arrangement exactly: means are
    # the estimand, the rank test is still run and still reported.
    if family == "binary_proportion":
        pw = (
            _binary_pairwise_ppi(group_arrays, group_lab_arrays, ci_alpha)
            if ppi_applied else _binary_pairwise_uncorrected(group_arrays, ci_alpha)
        )
    else:
        pw = (
            _numeric_pairwise_ppi(group_arrays, group_lab_arrays, ci_alpha, n_boot, rng)
            if ppi_applied else _numeric_pairwise_uncorrected(group_arrays, ci_alpha)
        )
    estimand, null_value = "mean_diff", 0.0

    raw_p = np.asarray(pw["pair_p"], dtype=float)
    # Shaffer's modified step-down rather than plain Holm. For an ALL-PAIRWISE
    # family the two are identical at step 1 (Shaffer's first divisor is the
    # largest achievable true-null count, which is m -- all groups equal), so
    # the family-wise error rate is provably the same; measured identical to
    # 4 decimals across k in {3,4,5} x likert/normal x 4000 reps. Shaffer is
    # then strictly more powerful from step 2 on, because pairwise equality is
    # transitive and not every remaining true-null count is achievable
    # (measured +0.03 to +0.13 extra rejections per family under alternatives
    # at k=4). Free power at identical FWER, so there is no reason to prefer
    # Holm here. Needs n_groups to derive the divisor sequence.
    corrected_p = (
        correct_pvalues(raw_p, method="shaffer", n_groups=k)
        if n_pairs > 1 else raw_p.copy()
    )

    # Judge-human alignment and effective label count for the tests just run.
    # Reporting only: never allowed to fail the comparison, and skipped
    # entirely when the metric is not judge-corrected (there is no PPI variance
    # reduction to describe).
    _om_rho2 = _om_neff = _n_lab_per_cond = None
    _pair_eff = {}
    _marginal_neff = None
    if ppi_applied:
        _om_rho2, _om_neff, _pair_eff = _ppi_label_efficiency(
            labels, group_arrays, group_lab_arrays, family)
        # Marginal means are PPI-corrected here too (see _compute_group_stats'
        # lab_arrays), and their estimand is a plain mean, so each group's own
        # Pearson r^2 governs. One number per group, spanning that group only.
        from evalstats.alignment import _marginal_efficiency
        _marginal_neff = [
            _marginal_efficiency(g, lab)[1]
            for g, lab in zip(group_arrays, group_lab_arrays)
        ]
        if any(v is None for v in _marginal_neff):
            _marginal_neff = None
        _counts = [int(np.count_nonzero(~np.isnan(lab))) for lab in group_lab_arrays]
        _n_lab_per_cond = float(np.mean(_counts)) if _counts else None

    def _eff_for(a, b):
        """Pair efficiency, tolerating either key order from judge_alignment."""
        return _pair_eff.get((a, b)) or _pair_eff.get((b, a)) or (None, None)

    pairwise = []
    for idx, (i, j) in enumerate(pw["pairs"]):
        _r2, _ne = _eff_for(str(labels[i]), str(labels[j]))
        pairwise.append(GroupDiffResult(
            label_a=labels[i], label_b=labels[j], estimand=estimand, null_value=null_value,
            point_estimate=float(pw["point"][idx]),
            ci_low=float(pw["ci_lo"][idx]), ci_high=float(pw["ci_hi"][idx]),
            p_value=float(corrected_p[idx]), raw_p_value=float(raw_p[idx]),
            n_a=int(group_arrays[i].size), n_b=int(group_arrays[j].size),
            mean_test_p=(None if pw["mean_test_p"] is None
                         else float(pw["mean_test_p"][idx])),
            rho2=_r2, n_eff=_ne,
        ))

    pareto_dict = None
    if secondary_col:
        from evalstats.core.pareto import (
            pareto_bootstrap_unpaired, classify_pareto_status, orient_higher_is_better,
        )
        secondary_oriented = [orient_higher_is_better(arr, secondary_direction) for arr in secondary_arrays]
        pareto_result = pareto_bootstrap_unpaired(
            group_arrays, secondary_oriented, labels, n_bootstrap=n_boot, rng=rng,
        )
        secondary_group_stats = _compute_group_stats(
            labels, secondary_arrays, alpha=alpha, n_bootstrap=n_boot, rng=rng,
        )
        pareto_dict = {
            "secondary_metric": secondary_col,
            "direction": secondary_direction,
            "result": pareto_result,
            "statuses": classify_pareto_status(pareto_result, alpha=alpha),
            # Reuses core.summary._print_pareto_section's display unmodified
            # (see _GroupStatsAsRobustness) -- same keys the paired path's
            # own pareto dict uses.
            "primary_robustness": _GroupStatsAsRobustness(group_stats),
            "secondary_robustness": _GroupStatsAsRobustness(secondary_group_stats),
        }

    return GroupComparisonResult(
        factor_col=factor_col, metric_col=metric_col,
        item_col=resolved_item, item_col_synthetic=item_synthetic,
        score_type=score_type, family=family,
        groups=group_stats, pairwise=pairwise,
        omnibus_test_name=omnibus_test_name, omnibus_statistic=omnibus_statistic,
        omnibus_p_value=omnibus_p_value, omnibus_corrected_p_value=omnibus_corrected_p_value,
        alpha=alpha, n_pairs=n_pairs,
        ci_correction="bonferroni" if n_pairs > 1 else "none",
        pvalue_correction="shaffer" if n_pairs > 1 else "none",
        ppi_applied=ppi_applied, alignment_result=alignment_result,
        omnibus_rho2=_om_rho2, omnibus_n_eff=_om_neff,
        n_lab_per_condition=_n_lab_per_cond,
        marginal_n_eff=_marginal_neff,
        show_p_values=p_values, pareto=pareto_dict,
    )
