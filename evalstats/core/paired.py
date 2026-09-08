"""Paired statistical comparisons between templates.

All comparisons are paired by input, since every template is evaluated on the
same benchmark set. This eliminates input-level variance and dramatically
increases statistical power compared to unpaired tests.

When the score array includes a run axis (R >= 3), pairwise comparisons use
a two-level (nested) bootstrap that resamples both inputs and within-cell
runs, so that seed variance is correctly propagated into confidence intervals
rather than being silently discarded.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
from scipy.stats import rankdata, studentized_range

from ..tests import (
    ttest as _es_ttest,
    wilcoxon as _es_wilcoxon,
    friedman as _es_friedman,
    _mcnemar_p,
    _mcnemar_midp_p,
    _paired_sign_test_p,
    _paired_signflip_pvalue,
)
from .resampling import (
    _logit_t_alpha_crit_batch,
    _nig_alpha_crit_batch,
    bca_interval_1d,
    bayes_bootstrap_means_1d,
    bayes_bootstrap_diffs_nested,
    smooth_bootstrap_means_1d,
    smooth_bootstrap_diffs_nested,
    bootstrap_diffs_nested,
    bootstrap_means_1d,
    bootstrap_t_ci_1d,
    bootstrap_t_ci_nested,
    resolve_resampling_method,
    newcombe_mover_paired_ci,
    mj_floor_paired_ci,
    tango_scc_paired_ci,
    bonett_price_paired_ci_from_diffs,
    mj_floor_paired_ci_from_diffs,
    mj_floor_paired_ci_multirun_cluster,
    bonett_price_paired_ci,
    bonett_price_paired_ci_multirun_shrunk,
    t_interval_ci_1d,
    logit_t_ci_1d,
    nig_ci_1d,
    bayes_paired_diff_ci,
    binary_routing_applies,
    degenerate_sample_ci,
    is_binary_scores,
    is_lopsided_binary,
    _stat,
    _nested_cell_mean_diffs,
    _reduce_rows,
    _weighted_medians_rows,
)
from .stats_utils import correct_pvalues, rescaled_ci
from ..config import (
    get_alpha_ci, GRADIENT_CI_ALPHAS, MAX_T_AUTO_METHOD,
    resolve_auto_simultaneous_ci_method, resolve_auto_pvalue_correction_method,
)


BAYES_BINARY_LARGE_N_THRESHOLD = 200

_NIG_PAIRED_DIFF_B0 = 0.0625 / 4
"""nig_ci_1d's default b0=0.0625 (prior mean of sigma^2, i.e. prior
sigma~=0.25) is calibrated for a single-sample rescale onto [lo, hi] --
see that function's own docstring: "weak knowledge that scores live in
[0, 1]". A PAIRED diff instead gets rescaled onto [-(hi-lo), hi-lo]
(needed so a zero diff maps to 0.5, nig's own prior centre) -- twice as
wide a span as the single-sample case. Reusing b0=0.0625 unchanged on a
paired diff implies 2^2=4x the intended prior variance in real diff units
(variance scales with the square of a linear rescale factor), producing
persistent, substantial over-coverage that isn't a deliberate safety
margin, just an unpropagated rescale-span change. This restores NIG's
effective prior to the same absolute variance the single-sample case
already uses correctly -- verified via simulation
(simulations/harness/cases/ci_paired.py): on likert paired diffs,
coverage went from 0.983 (n=10, default b0) to 0.946 (n=10, this
correction), 23% narrower for the same validity, holding across n=10-500
and on continuous data too."""


def _warn_bayes_binary_large_n(n_inputs: int, *, stacklevel: int = 4) -> None:
    """Warn when bayes_binary pairwise CI is used beyond its calibrated range."""
    if n_inputs < BAYES_BINARY_LARGE_N_THRESHOLD:
        return

    warnings.warn(
        "method='bayes_binary' was requested for pairwise binary comparison "
        f"with N={n_inputs} inputs. Simulations indicate this importance-"
        "sampling-based CI becomes dangerously overconfident at larger N "
        "(roughly ~10% at N=500 and ~20% at N=1000). "
        "Use method='newcombe' (or method='auto') for calibrated pairwise "
        "intervals at this sample size.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _rank_biserial(diffs: np.ndarray) -> float:
    """Rank biserial correlation for paired differences.

    Computed from the signed-rank decomposition of ``diffs``: rank the absolute
    values of non-zero differences, then return (R+ - R-) / (R+ + R-), where
    R+ and R- are the sums of ranks for positive and negative differences
    respectively.  Returns 0.0 when all differences are zero.

    Interpretation guidelines (Kerby, 2014): small ≈ 0.1, medium ≈ 0.3,
    large ≈ 0.5.  Range is [-1, 1].
    """
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    r_plus = float(np.sum(ranks[nonzero > 0]))
    r_minus = float(np.sum(ranks[nonzero < 0]))
    total = r_plus + r_minus
    return (r_plus - r_minus) / total if total > 0 else 0.0


def _compute_agreement_mcc(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[float, tuple[int, int, int, int]]:
    """Compute pairwise agreement MCC and confusion counts for two binary arrays.

    Treats ``values_a`` and ``values_b`` as binary vectors (thresholded at 0.5)
    and computes the Matthews Correlation Coefficient measuring how correlated
    their pass/fail patterns are — independent of which model is "better."

    MCC is symmetric: MCC(a, b) == MCC(b, a).  Range is [-1, 1]:
      +1 = identical pass/fail patterns
       0 = uncorrelated (independent errors)
      -1 = perfectly opposite patterns

    Returns
    -------
    (mcc, (n11, n10, n01, n00))
        n11 = both pass, n10 = A passes B fails, n01 = A fails B passes,
        n00 = both fail.
    """
    a = (np.asarray(values_a) >= 0.5).astype(int)
    b = (np.asarray(values_b) >= 0.5).astype(int)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    # MCC: TP=n11, TN=n00, FP=n01, FN=n10 (treating a as reference, b as prediction).
    # Symmetric: swapping a↔b swaps FP↔FN, giving the same value.
    tp, tn, fp, fn = n11, n00, n01, n10
    denom_sq = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom_sq == 0.0:
        mcc = 0.0
    else:
        mcc = float((tp * tn - fp * fn) / (denom_sq ** 0.5))
    return mcc, (n11, n10, n01, n00)


@dataclass
class PairedDiffResult:
    """Result of a paired comparison between two templates."""

    template_a: str
    template_b: str
    point_diff: float       # point estimate under the chosen statistic
    std_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    test_method: str
    n_inputs: int
    per_input_diffs: np.ndarray  # shape (M,) — per-input cell-mean differences
    n_runs: int = 1              # R used; 1 means no seed dimension
    statistic: str = "mean"      # 'mean' or 'median'
    wilcoxon_p: Optional[float] = None  # Wilcoxon signed-rank p-value (two-sided, on per_input_diffs)
    agreement_mcc: Optional[float] = None  # pass/fail pattern correlation (binary data only)
    binary_confusion: Optional[tuple[int, int, int, int]] = None  # (n11, n10, n01, n00)
    multi_ci: Optional[dict[float, tuple[float, float]]] = None  # {alpha: (lo, hi)} gradient bands

    @property
    def rank_biserial(self) -> float:
        """Rank biserial correlation for paired differences.

        Computed from ``per_input_diffs`` via the signed-rank decomposition:
        rank absolute non-zero differences, then return (R+ − R−) / (R+ + R−).
        Range is [−1, 1].  Interpretation guidelines (Kerby, 2014):
        small ≈ 0.1, medium ≈ 0.3, large ≈ 0.5.
        """
        return _rank_biserial(self.per_input_diffs)

    @property
    def effect_size(self) -> float:
        """Alias for ``rank_biserial``."""
        return self.rank_biserial

    def summary(self, *, alpha: Optional[float] = None, correction: str = "") -> None:
        """Print a focused summary for this pairwise comparison.

        Displays the gap, an ASCII interval plot of the confidence interval,
        and a plain-language verdict.

        Parameters
        ----------
        alpha : float
            Significance threshold (default 0.01).
        correction : str
            Name of the multiple-comparisons correction applied, shown in the
            header when provided.

        Examples
        --------
        >>> pair = report.pairwise.get("Model A", "Model B")
        >>> pair.summary()
        """
        if alpha is None:
            alpha = get_alpha_ci()
        from .summary import print_pairwise_summary
        print_pairwise_summary(self, alpha=alpha, correction=correction)


@dataclass
class FriedmanResult:
    """Friedman omnibus test + Nemenyi pairwise post-hoc.

    The Friedman test is a non-parametric alternative to repeated-measures
    ANOVA.  It ranks treatments within each block (input) and tests whether
    any treatment's average rank differs from the others.

    The Nemenyi post-hoc uses the Studentized range distribution to compare
    all pairs of average ranks simultaneously (FWER-controlled at the family
    level — no additional correction needed).
    """

    statistic: float                          # Friedman χ² statistic (uncorrected LLM-only, always)
    df: int                                   # degrees of freedom = k - 1
    p_value: float                            # omnibus p-value (uncorrected, from the raw LLM scores)
    nemenyi_p: dict[tuple[str, str], float]  # upper-triangle pairwise p-values
    avg_ranks: dict[str, float]              # mean rank per template (1 = best)
    n_inputs: int                             # N blocks
    n_templates: int                          # k treatments
    corrected_p_value: Optional[float] = None  # PPI-corrected omnibus p, when alignment= was passed

    def get_nemenyi_p(self, a: str, b: str) -> Optional[float]:
        """Return Nemenyi p for a pair regardless of storage order."""
        if (a, b) in self.nemenyi_p:
            return self.nemenyi_p[(a, b)]
        if (b, a) in self.nemenyi_p:
            return self.nemenyi_p[(b, a)]
        return None


def friedman_nemenyi(scores: np.ndarray, labels: list[str]) -> FriedmanResult:
    """Friedman omnibus test + Nemenyi pairwise post-hoc (scipy only).
    NOTE: This function is verified to match R's friedman.test and 
    PMCMRplus::frdAllPairsNemenyiTest on a reference matrix in the tests/.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(k, N)`` — k templates × N inputs.  If 3-D ``(k, N, R)``,
        cell means are taken over runs before ranking.
    labels : list[str]
        Template labels, length k.

    Returns
    -------
    FriedmanResult
    """
    scores = np.asarray(scores)
    if scores.ndim not in (2, 3):
        raise ValueError("scores must have shape (k, N) or (k, N, R)")

    if scores.ndim == 3:
        scores = scores.mean(axis=2)  # (k, N) cell means

    k, N = scores.shape

    if len(labels) != k:
        raise ValueError(f"labels length ({len(labels)}) must match number of templates ({k})")
    if k < 3:
        raise ValueError("Friedman test requires at least 3 templates (k >= 3)")
    if N < 1:
        raise ValueError("scores must include at least one input (N >= 1)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")

    # Friedman omnibus test — delegates to evalstats.tests.friedman (uncorrected
    # path) so the scipy call has a single implementation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _res = _es_friedman(*[scores[i] for i in range(k)], print_result=False)
    stat, p_val = _res.statistic, _res.p_value
    if not (np.isfinite(stat) and np.isfinite(p_val)):
        # Degenerate case (e.g., all treatments tied for every input).
        stat, p_val = 0.0, 1.0

    # Average ranks: rank across k treatments within each input, then average.
    # rank_matrix[i, j] = rank of template i for input j.
    rank_matrix = np.apply_along_axis(rankdata, 0, -scores)  # (k, N)
    avg_ranks = rank_matrix.mean(axis=1)  # (k,)

    # Nemenyi post-hoc: compare pairs via the Studentized range distribution.
    # Standard error of average-rank differences under H0.
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    nemenyi_p: dict[tuple[str, str], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(avg_ranks[i] - avg_ranks[j]) / se
            # Convert to Studentized range statistic (factor sqrt(2) per Demšar 2006).
            p = float(studentized_range.sf(q * np.sqrt(2), k, np.inf))
            nemenyi_p[(labels[i], labels[j])] = p

    avg_ranks_dict = {labels[i]: float(avg_ranks[i]) for i in range(k)}

    return FriedmanResult(
        statistic=float(stat),
        df=k - 1,
        p_value=float(p_val),
        nemenyi_p=nemenyi_p,
        avg_ranks=avg_ranks_dict,
        n_inputs=N,
        n_templates=k,
    )


@dataclass
class PairwiseMatrix:
    """Results of all pairwise comparisons."""

    labels: list[str]
    results: dict[tuple[str, str], PairedDiffResult]
    correction_method: str
    friedman: Optional[FriedmanResult] = None
    simultaneous_ci: bool = True
    simultaneous_ci_method: Optional[str] = None  # 'max_t' or 'bonferroni'; None if not applied

    def get(self, a: str, b: str) -> PairedDiffResult:
        """Get the comparison result for templates a vs b."""
        if (a, b) in self.results:
            return self.results[(a, b)]
        if (b, a) in self.results:
            r = self.results[(b, a)]
            # Flip confusion counts: swap n10 ↔ n01 (A and B are exchanged).
            # agreement_mcc is symmetric so it stays the same.
            flipped_conf: Optional[tuple[int, int, int, int]] = None
            if r.binary_confusion is not None:
                n11, n10, n01, n00 = r.binary_confusion
                flipped_conf = (n11, n01, n10, n00)
            flipped_multi_ci: Optional[dict[float, tuple[float, float]]] = None
            if r.multi_ci is not None:
                flipped_multi_ci = {a_: (-hi, -lo) for a_, (lo, hi) in r.multi_ci.items()}
            # Flip the result
            return PairedDiffResult(
                template_a=a,
                template_b=b,
                point_diff=-r.point_diff,
                std_diff=r.std_diff,
                ci_low=-r.ci_high,
                ci_high=-r.ci_low,
                p_value=r.p_value,
                test_method=r.test_method,
                n_inputs=r.n_inputs,
                per_input_diffs=-r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=r.wilcoxon_p,  # two-sided, so p is the same when flipping direction
                agreement_mcc=r.agreement_mcc,
                binary_confusion=flipped_conf,
                multi_ci=flipped_multi_ci,
            )
        raise KeyError(f"No comparison found for ({a}, {b})")

    def summary(self, a: str, b: str, *, alpha: Optional[float] = None) -> None:
        """Print a focused summary for the comparison between `a` and `b`.

        Retrieves the pairwise result via ``get(a, b)``, then delegates to
        ``PairedDiffResult.summary()``, automatically passing the correction
        method stored on this matrix.

        Parameters
        ----------
        a, b : str
            Entity labels.  The direction is always ``a − b``.
        alpha : float
            Significance threshold (default 0.01).

        Examples
        --------
        >>> report.pairwise.summary("Model A", "Model B")
        """
        if alpha is None:
            alpha = get_alpha_ci()
        pair = self.get(a, b)
        pair.summary(alpha=alpha, correction=self.correction_method)

    def point_diff_matrix(self) -> np.ndarray:
        """Return NxN matrix of point-estimate differences (mean or median)."""
        n = len(self.labels)
        mat = np.zeros((n, n))
        for i, a in enumerate(self.labels):
            for j, b in enumerate(self.labels):
                if i != j:
                    mat[i, j] = self.get(a, b).point_diff
        return mat


def _paired_t_pvalue(
    values_a: np.ndarray, values_b: np.ndarray, diffs: np.ndarray,
) -> float:
    """Paired t-test p-value, with an exact sign-test floor on zero-variance
    differences.

    Delegates to :func:`evalstats.tests.ttest`'s uncorrected paired path so
    the scipy call has a single implementation, then guards the one input
    where that p-value is not just imprecise but degenerate: a **constant
    non-zero** difference vector (every ``a_i - b_i`` identical, e.g. arm A
    scores 0.9 on every item and arm B scores 0.8). There ``s = 0``, so
    ``t = d/(s/sqrt(M))`` diverges and scipy returns exactly ``0.0`` --
    certainty that the effect is non-zero, obtained from a sample that
    contains no variance estimate at all.

    That number is indefensible on its own terms, and it also sits badly
    next to the companion interval, which on this same input is now the
    deliberately wide :func:`~evalstats.core.resampling.degenerate_sample_ci`
    bound (see :func:`_bonferroni_simultaneous_cis`) and can straddle 0. The
    two are not actually in conflict -- they answer different questions: an
    all-same-sign difference vector *does* rule out a null symmetric about
    0, while still leaving the *mean* unbounded away from 0, because the
    unobserved tail mass the CI has to allow for could sit anywhere in the
    metric's range. But that reading only survives if the p-value is a real
    number from a stated test rather than a divide-by-zero artifact.

    The replacement is :func:`~evalstats.tests._paired_sign_test_p`, the
    exact two-sided sign test, which on M identical non-zero differences is
    ``binomtest(M, M, 0.5) = 2 * 0.5**M``. This is the strongest claim the
    data supports without a variance estimate -- it uses only the signs,
    which is all a zero-spread sample actually pins down -- and it is not a
    new convention here: the binary/Tango and ``sign_test`` paths already
    report exactly this number on the same input (2**-29 at M=30), so this
    makes the continuous paths agree with them instead of reporting 0.

    Applied as ``max()`` rather than a straight substitution, so it can only
    ever widen the p-value, and only on the degenerate branch -- a genuinely
    tiny t-test p-value from data that *does* have spread is left alone.
    """
    t_result = _es_ttest(values_a, values_b, paired=True, print_result=False)
    p_value = float(t_result.p_value) if np.isfinite(t_result.p_value) else 1.0
    if len(diffs) >= 1 and float(np.ptp(diffs)) == 0.0:
        # Zero-variance differences. (_paired_sign_test_p itself returns 1.0
        # for the all-zero case, which is the right answer there too.)
        return max(p_value, _paired_sign_test_p(diffs))
    return p_value


def pairwise_differences(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str = "A",
    label_b: str = "B",
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "mj_floor", "tango", "bayes_binary", "permutation", "sign_test", "t_interval", "logit_t", "nig"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    multi_ci: bool = False,
    compute_wilcoxon: bool = True,
    score_range: Optional[tuple[float, float]] = None,
) -> PairedDiffResult:
    """Compute paired differences between two templates.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` a two-level nested bootstrap is used so that seed
        variance contributes to the confidence interval.  ``R = 1`` or
        ``R = 2`` fall back to the standard (non-seeded) path.
    idx_a, idx_b : int
        Indices of the two templates to compare.
    label_a, label_b : str
        Human-readable labels for the templates.
    method : str
        Statistical method: ``'auto'`` (default), ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'`` (Bayesian bootstrap), ``'smooth_bootstrap'``
        (smoothed bootstrap via Gaussian KDE), ``'bootstrap_t'``
        (studentized bootstrap-t CI), ``'newcombe'`` for paired
        binary (0/1) data using Newcombe CI + McNemar mid-p p-value,
        ``'mj_floor'`` for paired binary (0/1) data using the floored
        May & Johnson (1997) score CI + McNemar mid-p p-value,
        ``'tango'`` for paired binary (0/1) data using the exact Tango
        (1998) score CI + McNemar mid-p p-value (single-run only), or
        ``'bayes_binary'`` for paired binary (0/1) data using the
        Dirichlet-multinomial Bayesian model (Bowyer et al. 2025).
        Requires binary data; raises ValueError otherwise.
        ``'permutation'`` computes a paired sign-flip randomization p-value
        and reports a percentile-bootstrap CI for the paired effect size.
        ``'sign_test'`` computes an exact two-sided paired sign-test p-value
        (ties dropped) and reports a percentile-bootstrap CI for the paired
        effect size.
        ``'auto'`` selects ``'smooth_bootstrap'`` for non-binary data.
    ci : float
        Confidence level for the interval (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'mean'`` (default) or
        ``'median'``.
    compute_wilcoxon : bool
        Whether to compute the supplementary Wilcoxon signed-rank p-value
        (default ``True``, matching prior behavior). Set ``False`` to skip
        it -- e.g. for callers that never read ``PairedDiffResult.wilcoxon_p``
        and are calling this at high volume (Monte Carlo simulations), where
        the scipy call is pure overhead.

    Returns
    -------
    PairedDiffResult
    """
    if rng is None:
        rng = np.random.default_rng()

    def _seeded_fallback(seed_method: str) -> PairedDiffResult:
        return _pairwise_diffs_seeded(
            scores, idx_a, idx_b, label_a, label_b,
            method=seed_method, ci=ci, n_bootstrap=n_bootstrap,
            rng=rng, statistic=statistic, multi_ci=multi_ci,
            compute_wilcoxon=compute_wilcoxon,
        )

    def _paired_stats(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        diffs = values_a - values_b
        m = len(diffs)
        point_d = _stat(diffs, statistic)
        std_d = float(np.std(diffs, ddof=1))
        return diffs, m, point_d, std_d

    def _percentile_ci(boot_stats: np.ndarray, alpha_val: float) -> tuple[float, float]:
        ci_low = float(np.percentile(boot_stats, 100 * alpha_val / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha_val / 2)))
        return ci_low, ci_high

    def _bootstrap_tail_pvalue(boot_centered_stats: np.ndarray, point: float) -> float:
        extreme_count = np.sum(np.abs(boot_centered_stats) >= abs(point))
        return float((extreme_count + 1) / (n_bootstrap + 1))

    def _bootstrap_t_tail_pvalue_1d(values: np.ndarray, observed_stat: float) -> float:
        """Two-sided bootstrap-t p-value for 1-D paired differences.

        Uses studentized pivots ``t* = (theta* - theta_hat) / se*`` and compares
        against ``|t_obs| = |theta_hat| / se_obs`` for the null ``theta = 0``.
        Falls back to centered-bootstrap tail p-value when studentization is
        unstable or undefined.
        """
        n = len(values)
        centered_values = values - observed_stat

        def _fallback_centered_tail_pvalue() -> float:
            centered_boot = bootstrap_means_1d(
                centered_values, n_bootstrap=n_bootstrap, rng=rng, statistic="mean",
            )
            return _bootstrap_tail_pvalue(centered_boot, observed_stat)

        if n < 2:
            # Degenerate case: no variance estimate is available for studentization.
            return 1.0

        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = values[idx]                                # (B, n)
        boot_stats = samples.mean(axis=1)                    # (B,)
        boot_ses = np.std(samples, ddof=1, axis=1) / np.sqrt(n)

        se_obs = float(np.std(values, ddof=1)) / np.sqrt(n)
        if se_obs <= 0.0 or not np.isfinite(se_obs):
            return _fallback_centered_tail_pvalue()

        valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
        if not np.any(valid):
            return _fallback_centered_tail_pvalue()
        se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
        tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
        if tiny_frac > 0.05:
            return _fallback_centered_tail_pvalue()
        valid = valid & (boot_ses >= se_floor)
        if not np.any(valid):
            return _fallback_centered_tail_pvalue()

        t_stats = (boot_stats[valid] - observed_stat) / boot_ses[valid]
        t_obs = abs(observed_stat) / se_obs
        extreme_count = int(np.sum(np.abs(t_stats) >= t_obs))
        return float((extreme_count + 1) / (len(t_stats) + 1))

    def _build_result(
        *,
        diffs: np.ndarray,
        point_d: float,
        std_d: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        test_name: str,
        values_a: Optional[np.ndarray] = None,
        values_b: Optional[np.ndarray] = None,
        multi_ci_dict: Optional[dict[float, tuple[float, float]]] = None,
    ) -> PairedDiffResult:
        agr_mcc: Optional[float] = None
        bin_conf: Optional[tuple[int, int, int, int]] = None
        if (
            values_a is not None
            and values_b is not None
            and is_binary_scores(np.stack([values_a, values_b]))
        ):
            agr_mcc, bin_conf = _compute_agreement_mcc(values_a, values_b)

        # Two-sided Wilcoxon signed-rank p-value, reported alongside whatever
        # primary method was chosen. Calls evalstats.tests.wilcoxon directly
        # (uncorrected path) rather than a local reimplementation, so the
        # scipy call has a single home. That function raises when all paired
        # differences are zero (matching plain scipy); caught here since a
        # supplementary stat should degrade to None rather than crash the
        # whole comparison.
        wa = values_a if values_a is not None else diffs
        wb = values_b if values_b is not None else np.zeros_like(diffs)
        wilcoxon_p: Optional[float] = None
        if compute_wilcoxon and int(np.sum((wa - wb) != 0)) >= 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wilcoxon_p = float(_es_wilcoxon(wa, wb, print_result=False).p_value)
            except ValueError:
                wilcoxon_p = None

        return PairedDiffResult(
            template_a=label_a,
            template_b=label_b,
            point_diff=point_d,
            std_diff=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_method=test_name,
            n_inputs=len(diffs),
            per_input_diffs=diffs,
            n_runs=1,
            statistic=statistic,
            wilcoxon_p=wilcoxon_p,
            agreement_mcc=agr_mcc,
            binary_confusion=bin_conf,
            multi_ci=multi_ci_dict,
        )

    # ------------------------------------------------------------------ #
    # Bayesian binary path (Dirichlet-multinomial paired model)           #
    # ------------------------------------------------------------------ #
    if method == "bayes_binary":
        # When R >= 3 the per-run cell means are not binary values;
        # fall back to smooth bootstrap for the seeded nested path.
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback("smooth_bootstrap")
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        if not is_binary_scores(flat):
            raise ValueError(
                "method='bayes_binary' requires binary (0/1) data, but "
                "non-binary values were found in the score array. "
                "Use is_binary_scores() to check before calling."
            )
        diffs, m, point_d, std_d = _paired_stats(values_a, values_b)
        _warn_bayes_binary_large_n(m)
        alpha_val = 1.0 - ci
        ci_low, ci_high, prob_a_greater = bayes_paired_diff_ci(
            values_a, values_b, alpha_val, num_samples=n_bootstrap, rng=rng,
        )
        # Two-sided Bayesian p-value: posterior mass on the wrong side × 2
        p_value = float(2.0 * min(prob_a_greater, 1.0 - prob_a_greater))
        p_value = max(1.0 / (n_bootstrap + 1), p_value)
        mci: Optional[dict[float, tuple[float, float]]] = None
        if multi_ci:
            mci = {}
            for _a in GRADIENT_CI_ALPHAS:
                _lo, _hi, _ = bayes_paired_diff_ci(values_a, values_b, _a, num_samples=n_bootstrap, rng=rng)
                mci[_a] = (_lo, _hi)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name=f"bayes binary (n={n_bootstrap})",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Newcombe path for paired binary (0/1) data                         #
    # ------------------------------------------------------------------ #
    if method == "newcombe":
        # When R >= 3 the cell means are proportions, not binary values.
        # Fall back to smooth bootstrap for the seeded nested path.
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback("smooth_bootstrap")
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = newcombe_mover_paired_ci(values_a, values_b, alpha_val)
        p_value = _mcnemar_midp_p(values_a, values_b)
        mci = {_a: newcombe_mover_paired_ci(values_a, values_b, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="newcombe (mcnemar_midp p-value)",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    if method == "mj_floor":
        multirun = scores.ndim == 3 and scores.shape[2] > 1
        if multirun:
            # Multi-run: use the effective-N variant (mj_floor_er,
            # "ER-Tango" in the paper's decision tree / appendix), which estimates
            # an effective number of runs to account for within-item correlation
            # and reduces exactly to the standard Tango CI when n_runs == 1.
            values_a_full = scores[idx_a]   # (M, R)
            values_b_full = scores[idx_b]   # (M, R)
            values_a = values_a_full[:, 0]  # for _paired_stats / mcnemar (single-run view)
            values_b = values_b_full[:, 0]
        else:
            flat = scores.mean(axis=2) if scores.ndim == 3 else scores
            values_a = flat[idx_a]
            values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci

        if multirun:
            # Cluster (plain item-level variance), NOT the effective-runs variant:
            # its Kish R_eff term cancels exactly when the max() does not clamp and
            # inflates variance up to 2.8x when it does, so it was inert in the
            # high-ICC regime real eval data occupies and conservative elsewhere.
            ci_low, ci_high = mj_floor_paired_ci_multirun_cluster(values_a_full, values_b_full, alpha_val)
            if multi_ci:
                mci = {_a: mj_floor_paired_ci_multirun_cluster(values_a_full, values_b_full, _a) for _a in GRADIENT_CI_ALPHAS}
            else:
                mci = None
        else:
            ci_low, ci_high = mj_floor_paired_ci(values_a, values_b, alpha_val)
            mci = {_a: mj_floor_paired_ci(values_a, values_b, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        p_value = _mcnemar_midp_p(values_a, values_b)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="mj_floor cluster" if multirun else "mj_floor",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Bonett-Price path for paired binary (0/1) data                       #
    # ------------------------------------------------------------------ #
    if method == "bonett_price":
        multirun = scores.ndim == 3 and scores.shape[2] >= 3
        _flat_check = scores.mean(axis=2) if scores.ndim == 3 else scores
        if not is_binary_scores(scores if multirun else _flat_check):
            raise ValueError(
                "method='bonett_price' requires binary (0/1) data, but the scores "
                "array contains non-binary values. Use is_binary_scores() to check "
                "before calling, or choose a different method."
            )
        if multirun:
            values_a_full = scores[idx_a]
            values_b_full = scores[idx_b]
            values_a = values_a_full[:, 0]
            values_b = values_b_full[:, 0]
        else:
            flat = scores.mean(axis=2) if scores.ndim == 3 else scores
            values_a = flat[idx_a]
            values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        if multirun:
            # Multi-run default is the Laplace-shrunk-magnitude variant: the
            # +/-1 pseudo-items of the _cluster form are the largest possible
            # item-level discordance, which is right at R=1 but several times
            # heavier than a real discordant item once runs are averaged.
            # _shrunk reduces to bonett_price_paired_ci at R=1 bit-for-bit.
            ci_low, ci_high = bonett_price_paired_ci_multirun_shrunk(
                values_a_full, values_b_full, alpha_val
            )
            mci = ({_a: bonett_price_paired_ci_multirun_shrunk(values_a_full, values_b_full, _a)
                    for _a in GRADIENT_CI_ALPHAS} if multi_ci else None)
        else:
            ci_low, ci_high = bonett_price_paired_ci(values_a, values_b, alpha_val)
            mci = ({_a: bonett_price_paired_ci(values_a, values_b, _a)
                    for _a in GRADIENT_CI_ALPHAS} if multi_ci else None)
        p_value = _mcnemar_midp_p(values_a, values_b)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="bonett_price shrunk" if multirun else "bonett_price",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    if method == "tango":
        # The genuine Tango (1998) asymptotic score interval, obtained in
        # closed form via Chang et al. (2024)'s quartic with the continuity
        # correction set to zero. Validated against the published limits in
        # Fagerland, Lydersen & Laake (2014), Table V.
        #
        # NOTE: before 2026-08-24 this name dispatched to what is now
        # ``mj_floor`` -- a May & Johnson construction that is NOT Tango's
        # interval. See mj_floor_paired_ci's docstring.
        if scores.ndim == 3 and scores.shape[2] > 1:
            raise NotImplementedError(
                "method='tango' (the exact Tango score interval) has no "
                "multi-run form. Use method='mj_floor' for multi-run paired "
                "binary data, which dispatches to the effective-runs variant."
            )
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = tango_scc_paired_ci(values_a, values_b, alpha_val, c=0.0)
        mci = ({_a: tango_scc_paired_ci(values_a, values_b, _a, c=0.0)
                for _a in GRADIENT_CI_ALPHAS} if multi_ci else None)
        p_value = _mcnemar_midp_p(values_a, values_b)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="tango score (exact)",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired sign test path                                               #
    # ------------------------------------------------------------------ #
    if method in {"sign_test", "permutation"}:
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback(method)
        if scores.ndim == 3:
            scores = scores.mean(axis=2)

        _va_st = scores[idx_a]
        _vb_st = scores[idx_b]
        diffs, _, point_d, std_d = _paired_stats(_va_st, _vb_st)
        alpha = 1.0 - ci

        boot_stats = bootstrap_means_1d(
            diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats, alpha)
        mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None

        if method == "sign_test":
            p_value = _paired_sign_test_p(diffs)
            test_name = f"paired sign test + bootstrap ci (n={n_bootstrap})"
        else:
            p_value = _paired_signflip_pvalue(
                diffs, statistic=statistic, n_samples=n_bootstrap, rng=rng,
            )
            test_name = f"paired permutation + bootstrap ci (n={n_bootstrap})"

        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name=test_name,
            values_a=_va_st,
            values_b=_vb_st,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired t-interval path                                              #
    # ------------------------------------------------------------------ #
    if method == "t_interval":
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = t_interval_ci_1d(diffs, alpha_val)
        p_value = _paired_t_pvalue(values_a, values_b, diffs)
        mci = {_a: t_interval_ci_1d(diffs, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="paired t-interval",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired logit-t path (bounded [0, 1] numeric data)                   #
    # ------------------------------------------------------------------ #
    if method == "logit_t":
        # `flat` collapses the run axis to per-input cell means, so this is
        # "logit-t on run mean differences" for seeded (R >= 3) benchmarks
        # and plain logit-t for single-run ones -- no separate nested variant
        # needed. A paired difference of two [lo, hi] scores spans
        # [-(hi-lo), hi-lo], not [lo, hi] itself, so it's rescaled onto
        # [0, 1] using the diff span (not score_range directly) before the
        # logit transform -- rescaled_ci maps a zero diff to logit_t_ci_1d's
        # own centre of 0.5. Defaults to a [0, 1] native scale (diff span
        # [-1, 1]) when no score_range is given.
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        diff_span = (score_range[1] - score_range[0]) if score_range is not None else 1.0
        diff_lo, diff_hi = -diff_span, diff_span
        ci_low, ci_high = rescaled_ci(logit_t_ci_1d, diffs, alpha_val, diff_lo, diff_hi)
        p_value = _paired_t_pvalue(values_a, values_b, diffs)
        mci = (
            {_a: rescaled_ci(logit_t_ci_1d, diffs, _a, diff_lo, diff_hi) for _a in GRADIENT_CI_ALPHAS}
            if multi_ci else None
        )
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="paired logit-t",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired NIG path (discrete/ordinal bounded data, e.g. Likert)        #
    # ------------------------------------------------------------------ #
    if method == "nig":
        # Same rescale structure as the logit_t path above (paired diff of
        # two [lo, hi] scores spans [-(hi-lo), hi-lo]), but with the prior
        # variance corrected for that wider span -- see
        # _NIG_PAIRED_DIFF_B0's docstring. Recommended over logit_t
        # specifically for discrete/ordinal data (a Likert scale, an
        # integer percentage grade): see config.AUTO_ANALYZE_METHOD_TABLE's
        # "likert" row for the full rationale.
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        diff_span = (score_range[1] - score_range[0]) if score_range is not None else 1.0
        diff_lo, diff_hi = -diff_span, diff_span
        _nig_paired = functools.partial(nig_ci_1d, b0=_NIG_PAIRED_DIFF_B0)
        ci_low, ci_high = rescaled_ci(_nig_paired, diffs, alpha_val, diff_lo, diff_hi)
        p_value = _paired_t_pvalue(values_a, values_b, diffs)
        mci = (
            {_a: rescaled_ci(_nig_paired, diffs, _a, diff_lo, diff_hi) for _a in GRADIENT_CI_ALPHAS}
            if multi_ci else None
        )
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="paired NIG",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Route: seeded (R >= 3) vs. standard (2-D or R < 3)                 #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        R = scores.shape[2]
        if R >= 3:
            return _seeded_fallback(method)
        # R == 1 or R == 2: collapse to 2-D (warning already issued during validation)
        scores = scores.mean(axis=2)

    # ------------------------------------------------------------------ #
    # Standard (non-seeded) path                                          #
    # ------------------------------------------------------------------ #
    _va_std = scores[idx_a]
    _vb_std = scores[idx_b]
    diffs = _va_std - _vb_std
    m = len(diffs)
    point_d = _stat(diffs, statistic)
    std_d = float(np.std(diffs, ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, m)

    mci: Optional[dict[float, tuple[float, float]]] = None

    if resolved_method == "bootstrap":
        centered_diffs = diffs - point_d
        boot_centered_stats = np.empty(n_bootstrap)
        if statistic == "median":
            for b in range(n_bootstrap):
                idx = rng.choice(m, size=m, replace=True)
                boot_centered_stats[b] = np.median(centered_diffs[idx])
        else:
            for b in range(n_bootstrap):
                idx = rng.choice(m, size=m, replace=True)
                boot_centered_stats[b] = np.mean(centered_diffs[idx])
        boot_stats = boot_centered_stats + point_d
        ci_low, ci_high = _percentile_ci(boot_stats, alpha)
        p_value = _bootstrap_tail_pvalue(boot_centered_stats, point_d)
        test_name = f"bootstrap (n={n_bootstrap})"
        if multi_ci:
            mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

    elif resolved_method in {"bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"}:
        samplers = {
            "bca": bootstrap_means_1d,
            "bayes_bootstrap": bayes_bootstrap_means_1d,
            "smooth_bootstrap": smooth_bootstrap_means_1d,
            "bootstrap_t": bootstrap_means_1d,
        }
        sampler = samplers[resolved_method]

        if resolved_method == "bootstrap_t":
            ci_low, ci_high = bootstrap_t_ci_1d(
                diffs,
                point_d,
                n_bootstrap,
                alpha,
                rng,
                statistic=statistic,
            )
            if multi_ci:
                mci = {
                    _a: bootstrap_t_ci_1d(
                        diffs,
                        point_d,
                        n_bootstrap,
                        _a,
                        rng,
                        statistic=statistic,
                    )
                    for _a in GRADIENT_CI_ALPHAS
                }
        else:
            boot_stats = sampler(
                diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )

        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(
                diffs, point_d, boot_stats, alpha, statistic=statistic,
            )
            if multi_ci:
                mci = {_a: bca_interval_1d(diffs, point_d, boot_stats, _a, statistic=statistic) for _a in GRADIENT_CI_ALPHAS}
        elif resolved_method != "bootstrap_t":
            ci_low, ci_high = _percentile_ci(boot_stats, alpha)
            if multi_ci:
                mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

        if resolved_method == "bootstrap_t" and statistic == "mean":
            p_value = _bootstrap_t_tail_pvalue_1d(diffs, point_d)
        else:
            centered_diffs = diffs - point_d
            boot_centered_stats = sampler(
                centered_diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )
            p_value = _bootstrap_tail_pvalue(boot_centered_stats, point_d)

        test_labels = {
            "bca": "bca bootstrap",
            "bayes_bootstrap": "bayesian bootstrap",
            "smooth_bootstrap": "smooth bootstrap",
            "bootstrap_t": "bootstrap-t",
        }
        test_name = f"{test_labels[resolved_method]} (n={n_bootstrap})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    return _build_result(
        diffs=diffs,
        point_d=point_d,
        std_d=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_name=test_name,
        values_a=_va_std,
        values_b=_vb_std,
        multi_ci_dict=mci,
    )


def _pairwise_diffs_seeded(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str,
    label_b: str,
    *,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "permutation", "sign_test"],
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    multi_ci: bool = False,
    compute_wilcoxon: bool = True,
) -> PairedDiffResult:
    """Seeded paired comparison using a two-level nested bootstrap.

    ``scores`` has shape ``(N, M, R)`` with R >= 3.

    Point estimates are computed from per-input cell means (averaged over
    runs).  The bootstrap resamples both inputs and within-cell runs so that
    seed variance is propagated into the CI.  For BCa, the jackknife
    acceleration is estimated at the input level (leaving one input out at a
    time), which is the correct primary sampling unit.
    """
    M, R = scores.shape[1], scores.shape[2]
    scores_a = scores[idx_a]   # (M, R)
    scores_b = scores[idx_b]   # (M, R)

    # Point estimates from cell means (within-cell aggregation always uses mean).
    cell_means_a = scores_a.mean(axis=1)    # (M,)
    cell_means_b = scores_b.mean(axis=1)    # (M,)
    cell_diffs = cell_means_a - cell_means_b  # (M,)

    point_d = _stat(cell_diffs, statistic)
    std_d = float(cell_diffs.std(ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, M)

    def _percentile_ci(boot_stats: np.ndarray) -> tuple[float, float]:
        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    def _percentile_ci_alpha(boot_stats: np.ndarray, a: float) -> tuple[float, float]:
        return (
            float(np.percentile(boot_stats, 100 * a / 2)),
            float(np.percentile(boot_stats, 100 * (1 - a / 2))),
        )

    def _bootstrap_tail_pvalue(boot_stats: np.ndarray) -> float:
        boot_centered = boot_stats - point_d
        extreme_count = np.sum(np.abs(boot_centered) >= abs(point_d))
        return float((extreme_count + 1) / (n_bootstrap + 1))

    def _bootstrap_t_tail_pvalue_nested(diff_scores: np.ndarray) -> float:
        """Two-sided bootstrap-t p-value for seeded paired differences.

        ``diff_scores`` has shape ``(M, R)``. Studentization is performed using
        bootstrap replicate SE over resampled input-level cell means.
        """
        m_inputs, n_runs = diff_scores.shape
        cell_means_obs = diff_scores.mean(axis=1)
        se_obs = float(np.std(cell_means_obs, ddof=1)) / np.sqrt(m_inputs)

        if se_obs <= 0.0 or not np.isfinite(se_obs):
            boot_stats_fallback = bootstrap_diffs_nested(
                scores_a, scores_b, n_bootstrap, rng, statistic="mean",
            )
            return _bootstrap_tail_pvalue(boot_stats_fallback)

        input_idx = rng.integers(0, m_inputs, size=(n_bootstrap, m_inputs))
        run_idx = rng.integers(0, n_runs, size=(n_bootstrap, m_inputs, n_runs))

        selected = diff_scores[input_idx]  # (B, M, R)
        b_rng = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]
        m_rng = np.arange(m_inputs)[np.newaxis, :, np.newaxis]
        resampled = selected[b_rng, m_rng, run_idx]  # (B, M, R)
        cell_means_boot = resampled.mean(axis=2)  # (B, M)

        boot_stats = cell_means_boot.mean(axis=1)  # (B,)
        boot_ses = np.std(cell_means_boot, ddof=1, axis=1) / np.sqrt(m_inputs)

        valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
        if not np.any(valid):
            return _bootstrap_tail_pvalue(boot_stats)
        se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
        tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
        if tiny_frac > 0.05:
            return _bootstrap_tail_pvalue(boot_stats)
        valid = valid & (boot_ses >= se_floor)
        if not np.any(valid):
            return _bootstrap_tail_pvalue(boot_stats)

        t_stats = (boot_stats[valid] - point_d) / boot_ses[valid]
        t_obs = abs(point_d) / se_obs
        extreme_count = int(np.sum(np.abs(t_stats) >= t_obs))
        return float((extreme_count + 1) / (len(t_stats) + 1))

    mci_seeded: Optional[dict[float, tuple[float, float]]] = None

    if method == "permutation":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        if multi_ci:
            mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
        p_value = _paired_signflip_pvalue(
            cell_diffs, statistic=statistic, n_samples=n_bootstrap, rng=rng,
        )
        test_name = f"nested paired permutation + bootstrap ci (n={n_bootstrap}, R={R})"

    elif method == "sign_test":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        if multi_ci:
            mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
        p_value = _paired_sign_test_p(cell_diffs)
        test_name = f"nested paired sign test + bootstrap ci (n={n_bootstrap}, R={R})"

    elif resolved_method in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"}:
        samplers = {
            "bootstrap": bootstrap_diffs_nested,
            "bca": bootstrap_diffs_nested,
            "bayes_bootstrap": bayes_bootstrap_diffs_nested,
            "smooth_bootstrap": smooth_bootstrap_diffs_nested,
            "bootstrap_t": bootstrap_diffs_nested,
        }

        if resolved_method == "bootstrap_t" and statistic == "mean":
            # bootstrap_t_ci_nested/_bootstrap_t_tail_pvalue_nested draw their
            # own studentized resamples; the plain sampler is not needed here.
            diff_scores = scores_a - scores_b  # (M, R)
            ci_low, ci_high = bootstrap_t_ci_nested(
                diff_scores,
                point_d,
                n_bootstrap,
                alpha,
                rng,
            )
            if multi_ci:
                mci_seeded = {
                    _a: bootstrap_t_ci_nested(
                        diff_scores,
                        point_d,
                        n_bootstrap,
                        _a,
                        rng,
                    )
                    for _a in GRADIENT_CI_ALPHAS
                }
            p_value = _bootstrap_t_tail_pvalue_nested(diff_scores)
        else:
            boot_stats = samplers[resolved_method](
                scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
            )

            if resolved_method == "bootstrap_t":
                # statistic == "median": studentization isn't implemented for
                # median, so fall back to plain percentile bootstrap.
                warnings.warn(
                    "nested bootstrap-t studentization is implemented for "
                    "'mean'; falling back to percentile bootstrap for "
                    "'median'.",
                    UserWarning,
                    stacklevel=3,
                )
                ci_low, ci_high = _percentile_ci(boot_stats)
                if multi_ci:
                    mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
            elif resolved_method == "bca":
                # BCa: jackknife over inputs (the outer sampling unit) using cell_diffs.
                ci_low, ci_high = bca_interval_1d(
                    cell_diffs, point_d, boot_stats, alpha, statistic=statistic,
                )
                if multi_ci:
                    mci_seeded = {_a: bca_interval_1d(cell_diffs, point_d, boot_stats, _a, statistic=statistic) for _a in GRADIENT_CI_ALPHAS}
            else:
                ci_low, ci_high = _percentile_ci(boot_stats)
                if multi_ci:
                    mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

            p_value = _bootstrap_tail_pvalue(boot_stats)

        test_labels = {
            "bootstrap": "nested bootstrap",
            "bca": "nested bca bootstrap",
            "bayes_bootstrap": "nested bayesian bootstrap",
            "smooth_bootstrap": "nested smooth bootstrap",
            "bootstrap_t": "nested bootstrap-t",
        }
        test_name = f"{test_labels[resolved_method]} (n={n_bootstrap}, R={R})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    # Two-sided Wilcoxon signed-rank p-value on cell means, reported alongside
    # whatever primary (nested) method was chosen. See the non-seeded path in
    # pairwise_differences for why the ValueError guard is needed here.
    wilcoxon_p: Optional[float] = None
    if compute_wilcoxon and int(np.sum(cell_diffs != 0)) >= 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wilcoxon_p = float(_es_wilcoxon(cell_means_a, cell_means_b, print_result=False).p_value)
        except ValueError:
            wilcoxon_p = None

    # Agreement MCC for seeded binary data: use per-input majority vote.
    agr_mcc: Optional[float] = None
    bin_conf: Optional[tuple[int, int, int, int]] = None
    if is_binary_scores(scores_a) and is_binary_scores(scores_b):
        majority_a = (cell_means_a >= 0.5).astype(float)
        majority_b = (cell_means_b >= 0.5).astype(float)
        agr_mcc, bin_conf = _compute_agreement_mcc(majority_a, majority_b)

    return PairedDiffResult(
        template_a=label_a,
        template_b=label_b,
        point_diff=point_d,
        std_diff=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_method=test_name,
        n_inputs=M,
        per_input_diffs=cell_diffs,
        n_runs=R,
        statistic=statistic,
        wilcoxon_p=wilcoxon_p,
        agreement_mcc=agr_mcc,
        binary_confusion=bin_conf,
        multi_ci=mci_seeded,
    )


def _apply_max_t_cis(
    boot_stats: np.ndarray,
    point_ests: np.ndarray,
    pairs: list,
    ci: float,
) -> tuple[dict, dict]:
    """Apply the studentized max-T critical value to a pre-built bootstrap matrix.

    This is the shared computation used by both the standard resampling path
    and the pre-computed bootstrap path (e.g. PPI) in
    :func:`_max_stat_simultaneous_cis`.

    Parameters
    ----------
    boot_stats : np.ndarray, shape (B, k)
        Bootstrap distribution of pairwise diffs, one column per pair.
    point_ests : np.ndarray, shape (k,)
        Observed pairwise point estimates.
    pairs : list[tuple[str, str]]
        Pair labels in the same order as columns of *boot_stats*.
    ci : float
        Simultaneous confidence level (e.g. 0.95).

    Returns
    -------
    tuple[dict, dict]
        ``(sim_cis, max_t_pvalues)``.
    """
    se = np.std(boot_stats, axis=0, ddof=1)  # (k,)
    valid = se > 1e-12

    if not np.any(valid):
        return {}, {}

    se_safe = np.where(valid, se, 1.0)
    T = (boot_stats - point_ests[np.newaxis, :]) / se_safe[np.newaxis, :]  # (B, k)
    M_b = np.max(np.abs(T[:, valid]), axis=1)  # (B,)

    return _max_t_cis_from_null_dist(M_b, se, valid, point_ests, pairs, ci)


def _max_t_cis_from_null_dist(
    M_b: np.ndarray,
    se: np.ndarray,
    valid: np.ndarray,
    point_ests: np.ndarray,
    pairs: list,
    ci: float,
) -> tuple[dict, dict]:
    """Build simultaneous CIs/p-values from an already-computed max-|T| null
    distribution and per-pair SEs -- the final step shared by
    :func:`_apply_max_t_cis` (plain bootstrap SE) and the studentized
    bootstrap-t branch of :func:`_max_stat_simultaneous_cis` (per-replicate
    SE), which differ only in how ``M_b``/``se`` were computed upstream, not
    in how a critical value and per-pair CI/p-value are built from them.

    Parameters
    ----------
    M_b : np.ndarray, shape (B,)
        Bootstrap draws of the max standardized statistic across pairs.
    se : np.ndarray, shape (k,)
        Per-pair standard error used both to build the CI half-width and to
        studentize the observed point estimate for the p-value.
    valid : np.ndarray, shape (k,), bool
        Which pairs have a usable (non-degenerate) SE.
    point_ests : np.ndarray, shape (k,)
        Observed pairwise point estimates.
    pairs : list[tuple[str, str]]
        Pair labels in the same order as *se*/*point_ests*.
    ci : float
        Simultaneous confidence level (e.g. 0.95).
    """
    c = float(np.quantile(M_b, ci))
    B_total = len(M_b)

    sim_cis: dict = {}
    max_t_pvalues: dict = {}
    for p_idx, pair in enumerate(pairs):
        if valid[p_idx]:
            half = c * se[p_idx]
            sim_cis[pair] = (
                float(point_ests[p_idx] - half),
                float(point_ests[p_idx] + half),
            )
            t_obs = abs(float(point_ests[p_idx])) / float(se[p_idx])
            extreme = int(np.sum(M_b >= t_obs))
            max_t_pvalues[pair] = float((extreme + 1) / (B_total + 1))
        else:
            sim_cis[pair] = (float(point_ests[p_idx]), float(point_ests[p_idx]))
            max_t_pvalues[pair] = 1.0

    return sim_cis, max_t_pvalues


def _max_stat_simultaneous_cis(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[str],
    method: str,
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    *,
    precomputed_boot_stats: Optional[np.ndarray] = None,
    precomputed_point_ests: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    """Compute simultaneous CIs via the studentized bootstrap max-T method.

    Uses shared resamples across all pairs so that the joint distribution of
    the max standardized statistic naturally accounts for correlations between
    comparisons (unlike Bonferroni, which assumes independence).

    For each bootstrap replicate *b* and each pair *(i, j)*, the standardized
    statistic is::

        T_ij^b = (θ̂_ij^b − θ̂_ij) / SE_ij

    where SE_ij = std({θ̂_ij^b}) over all B replicates.  The simultaneous
    critical value *c* is the (1−α) quantile of::

        M^b = max_{(i,j)} |T_ij^b|

    and each simultaneous CI is [θ̂_ij − c·SE_ij, θ̂_ij + c·SE_ij].

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M)`` or ``(N, M, R)``.  When ``R >= 3`` the seeded
        nested bootstrap is used; otherwise scores are collapsed to 2-D.
        Ignored when *precomputed_boot_stats* is supplied.
    pairs : list[tuple[str, str]]
        All pairs for which simultaneous CIs should be computed, in the
        canonical (label_a, label_b) storage order.
    labels : list[str]
        Template labels — used to map names to row indices in *scores*.
        Ignored when *precomputed_boot_stats* is supplied.
    method : str
        Bootstrap variant.  Supported: ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'bootstrap_t'``, ``'auto'``
        (treated as ``'smooth_bootstrap'``), ``'permutation'``,
        ``'sign_test'``.  Methods that do not use bootstrap resampling
        for CIs (``'newcombe'``, ``'mj_floor'``, ``'tango'``, ``'bayes_binary'``,
        ``'lmm'``) are not supported; an empty dict is returned for these.
        Ignored when *precomputed_boot_stats* is supplied.
    ci : float
        Desired simultaneous confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap replicates.  Ignored when *precomputed_boot_stats*
        is supplied.
    rng : np.random.Generator
        Ignored when *precomputed_boot_stats* is supplied.
    statistic : str
        ``'mean'`` or ``'median'``.  Ignored when *precomputed_boot_stats*
        is supplied.
    precomputed_boot_stats : np.ndarray, shape (B, k), optional
        Pre-computed bootstrap distribution of pairwise diffs, one column
        per pair in *pairs* order.  When provided the resampling block is
        skipped entirely and the max-T statistic is derived directly from
        this matrix.  Requires *precomputed_point_ests*.
    precomputed_point_ests : np.ndarray, shape (k,), optional
        Observed pairwise point estimates corresponding to each column of
        *precomputed_boot_stats*.  Required when *precomputed_boot_stats*
        is supplied.

    Returns
    -------
    tuple[dict[tuple[str, str], tuple[float, float]], dict[tuple[str, str], float]]
        ``(sim_cis, max_t_pvalues)`` where *sim_cis* maps each pair to its
        ``(ci_low, ci_high)`` simultaneous CI.  Returns ``({}, {})`` for
        unsupported methods or degenerate inputs.
    """
    if len(pairs) == 0:
        return {}, {}

    # ── Pre-computed bootstrap path (e.g. PPI correction) ────────────────────
    # When the caller already has a joint bootstrap distribution (one draw per
    # row, one pair per column), skip all resampling and run the shared max-T
    # computation directly.
    if precomputed_boot_stats is not None:
        if precomputed_point_ests is None:
            raise ValueError(
                "precomputed_point_ests must be provided together with "
                "precomputed_boot_stats"
            )
        return _apply_max_t_cis(
            np.asarray(precomputed_boot_stats, dtype=float),
            np.asarray(precomputed_point_ests, dtype=float),
            pairs,
            ci,
        )

    # ── Standard path: resample from raw scores ───────────────────────────────
    _BOOTSTRAP_COMPATIBLE = {
        "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
        "permutation", "sign_test", "auto",
    }
    # Resolve 'auto' to its concrete method
    if method == "auto":
        method = MAX_T_AUTO_METHOD

    if method not in _BOOTSTRAP_COMPATIBLE:
        return {}, {}

    k = len(pairs)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    pair_indices = [(label_to_idx[a], label_to_idx[b]) for (a, b) in pairs]

    seeded = scores.ndim == 3 and scores.shape[2] >= 3

    # ------------------------------------------------------------------
    # Seeded path  (N, M, R) with R >= 3
    # ------------------------------------------------------------------
    if seeded:
        M, R = scores.shape[1], scores.shape[2]

        # Point estimates: statistic of per-input cell-mean differences.
        point_ests = np.array([
            _stat(scores[i].mean(axis=1) - scores[j].mean(axis=1), statistic)
            for (i, j) in pair_indices
        ])

        boot_stats_cols: list[np.ndarray] = []

        if method == "bayes_bootstrap":
            # Shared inner run-resample indices and shared Dirichlet weights.
            run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))  # (B, M, R)
            exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))
            outer_weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)  # (B, M)
            for (i, j) in pair_indices:
                diffs = _nested_cell_mean_diffs(
                    scores[i], scores[j], run_idx,
                )  # (B, M) — no outer resampling; Dirichlet weights applied below
                if statistic == "mean":
                    boot_stats_cols.append(
                        (outer_weights * diffs).sum(axis=1)
                    )
                else:
                    boot_stats_cols.append(
                        _weighted_medians_rows(diffs, outer_weights)
                    )
        else:
            # Shared outer input indices and inner run indices.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))  # (B, M)
            run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))  # (B, M, R)
            for (i, j) in pair_indices:
                if method == "smooth_bootstrap":
                    from scipy.stats import gaussian_kde
                    cell_diffs = scores[i].mean(axis=1) - scores[j].mean(axis=1)
                    std_val = float(np.std(cell_diffs, ddof=1)) if M > 1 else 0.0
                    h = 0.0
                    if M >= 2 and np.isfinite(std_val) and std_val > 0:
                        try:
                            h = float(gaussian_kde(cell_diffs).factor * std_val)
                        except np.linalg.LinAlgError:
                            pass
                    diffs = _nested_cell_mean_diffs(
                        scores[i], scores[j], run_idx, input_idx,
                    )  # (B, M)
                    if h > 0.0:
                        diffs = diffs + rng.normal(0.0, h, size=(n_bootstrap, M))
                else:
                    # bootstrap, bca, permutation, sign_test
                    diffs = _nested_cell_mean_diffs(
                        scores[i], scores[j], run_idx, input_idx,
                    )  # (B, M)
                boot_stats_cols.append(_reduce_rows(diffs, statistic))  # (B,)

        boot_stats = np.column_stack(boot_stats_cols)  # (B, k)

    # ------------------------------------------------------------------
    # Non-seeded path  (N, M) or (N, M, R) with R < 3 collapsed to 2-D
    # ------------------------------------------------------------------
    else:
        def _batch_resample(
            diffs_mat: np.ndarray,
            input_idx: np.ndarray,
            statistic: str,
            batch_size: int = 128,
            bandwidths: Optional[np.ndarray] = None,
            noise_rng: Optional[np.random.Generator] = None,
        ) -> np.ndarray:
            """Memory-efficient joint resampling for Max-T statistics.

            Processes bootstrap resamples in batches so that only a slice of
            shape (batch, M, k) is live at once rather than the full (B, M, k).
            When ``bandwidths`` and ``noise_rng`` are supplied, KDE noise is
            added per-batch before aggregation (smooth bootstrap path).
            """
            M_mat = diffs_mat.T  # (M, k) — transposed for cache-friendly row access
            B, M = input_idx.shape
            k = diffs_mat.shape[0]
            out = np.empty((B, k), dtype=diffs_mat.dtype)

            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                batch = end - start
                # (batch, M, k)
                chunk = M_mat[input_idx[start:end]]
                if bandwidths is not None and noise_rng is not None:
                    chunk = chunk + (
                        noise_rng.normal(0.0, 1.0, size=(batch, M, k))
                        * bandwidths[np.newaxis, np.newaxis, :]
                    )
                if statistic == "mean":
                    out[start:end] = chunk.mean(axis=1)
                else:
                    out[start:end] = np.median(chunk, axis=1)

            return out

        scores_2d = scores.mean(axis=2) if scores.ndim == 3 else scores  # (N, M)
        M = scores_2d.shape[1]

        # Per-pair diffs stacked: (k, M).
        # diffs_mat[:, input_idx] uses numpy fancy indexing to produce
        # shape (k, B, M), then .mean(axis=2).T → (B, k).
        diffs_mat = np.stack(
            [scores_2d[i] - scores_2d[j] for (i, j) in pair_indices],
            axis=0,
        )  # (k, M)

        if statistic == "mean":
            point_ests = diffs_mat.mean(axis=1)  # (k,)
        else:
            point_ests = np.median(diffs_mat, axis=1)  # (k,)

        if method == "bayes_bootstrap":
            # Shared Dirichlet weights over the M inputs.
            exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))
            weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)  # (B, M)
            if statistic == "mean":
                # (B, M) @ (M, k) → (B, k)
                boot_stats = weights @ diffs_mat.T
            else:
                boot_stats = np.empty((n_bootstrap, k))
                for p_idx in range(k):
                    vals = np.broadcast_to(diffs_mat[p_idx], (n_bootstrap, M))
                    boot_stats[:, p_idx] = _weighted_medians_rows(
                        np.ascontiguousarray(vals), weights,
                    )

        elif method == "smooth_bootstrap":
            from scipy.stats import gaussian_kde
            # Per-pair KDE bandwidth; shared input indices.
            bandwidths = np.zeros(k)
            for p_idx in range(k):
                d = diffs_mat[p_idx]
                std_val = float(np.std(d, ddof=1)) if M > 1 else 0.0
                if M >= 2 and np.isfinite(std_val) and std_val > 0:
                    try:
                        bandwidths[p_idx] = float(gaussian_kde(d).factor * std_val)
                    except np.linalg.LinAlgError:
                        pass

            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            boot_stats = _batch_resample(
                diffs_mat, input_idx, statistic,
                bandwidths=bandwidths, noise_rng=rng,
            )  # (B, k)

        elif method == "bootstrap_t" and statistic == "mean":
            # Studentized max-T: per-bootstrap-sample SE eliminates the
            # anti-conservative bias of plain pivots, which underestimate
            # SE by sqrt((n-1)/n).  The studentized pivot T_b = (d_b - d_obs)/se_b
            # and observed t_obs = |d_obs|/se_obs both follow approximately
            # t_{M-1}, so the Romano-Wolf guarantee holds.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            obs_se = np.std(diffs_mat, axis=1, ddof=1) / np.sqrt(M)  # (k,)

            M_mat_T = diffs_mat.T  # (M, k) — transposed for cache-friendly access
            batch_sz = 128
            bmeans_rows: list[np.ndarray] = []
            bses_rows: list[np.ndarray] = []
            for _s in range(0, n_bootstrap, batch_sz):
                _e = min(_s + batch_sz, n_bootstrap)
                chunk = M_mat_T[input_idx[_s:_e]]  # (batch, M, k)
                bmeans_rows.append(chunk.mean(axis=1))
                bses_rows.append(chunk.std(axis=1, ddof=1) / np.sqrt(M))
            boot_means_b = np.concatenate(bmeans_rows, axis=0)  # (B, k)
            boot_ses_b = np.concatenate(bses_rows, axis=0)  # (B, k)

            se_b_safe = np.where(boot_ses_b > 1e-12, boot_ses_b, 1.0)
            T_stud = (boot_means_b - point_ests) / se_b_safe  # (B, k)

            se_valid_b = obs_se > 1e-12
            if not np.any(se_valid_b):
                return {}, {}

            M_b_stud = np.max(np.abs(T_stud[:, se_valid_b]), axis=1)  # (B,)
            return _max_t_cis_from_null_dist(
                M_b_stud, obs_se, se_valid_b, point_ests, pairs, ci,
            )

        else:
            # bootstrap, bca, permutation, sign_test, bootstrap_t+median —
            # shared integer indices, plain (non-studentized) pivots.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            # _batch_resample already computes the per-pair statistic: (B, k)
            boot_stats = _batch_resample(diffs_mat, input_idx, statistic)  # (B, k)

    return _apply_max_t_cis(boot_stats, point_ests, pairs, ci)


def _degenerate_pair_ci(
    point_diff: float,
    M: int,
    alpha: float,
    diff_bounds: Optional[tuple[float, float]],
) -> tuple[float, float]:
    """CI for a pair whose paired differences carry no variance information.

    Covers both degenerate inputs :func:`_bonferroni_simultaneous_cis` can
    hit: ``M < 2`` (a single paired observation, or none) and a constant
    difference vector (``se`` numerically 0). In both, every variance-driven
    construction -- the t-interval, the delta method, any resampling scheme
    -- collapses to the zero-width interval ``(point_diff, point_diff)``,
    which asserts the effect is *exactly* ``point_diff`` with certainty and
    covers the truth with probability 0 unless the difference really is a
    point mass. See :func:`~evalstats.core.resampling.degenerate_sample_ci`
    for the bound used instead and why it is the honest answer.

    Marginal coverage on a paired DGP that reaches this branch often
    (difference = +0.1 with probability p, -0.5 otherwise, so the truth is
    *near* the atom but not at it; 4000 reps, diff bounds [-1, 1],
    nominal 95%)::

        n    p     P(degenerate)   coverage before   coverage after
        10   0.90      0.36             0.641            0.999
        10   0.99      0.90             0.099            1.000
        20   0.97      0.55             0.453            1.000
        30   0.90      0.05             0.951            0.998
        30   0.99      0.73             0.267            1.000

    -- i.e. the old branch missed on *every* degenerate rep (the truth is
    never exactly the atom), so coverage tracked ``1 - P(degenerate)`` and
    collapsed as the branch fired more often. The new bound is conservative
    rather than nominal (~100%, the price
    :func:`~evalstats.core.resampling.degenerate_sample_ci` documents), and
    is only ever paid on samples that would otherwise have been reported
    with false certainty.

    *diff_bounds* is the support of a single paired difference, ``(-(hi-lo),
    hi-lo)`` for a metric ranging over ``[lo, hi]`` -- not the metric's own
    bounds. The router resolves it per data kind (see
    :func:`_simultaneous_cis_router`).

    When *diff_bounds* is ``None`` -- the unbounded data kind, i.e. no
    ``score_range`` and non-binary scores -- the result is
    ``(-inf, +inf)``. That is not a punt: for a distribution with unbounded
    support, no finite confidence interval for the mean has guaranteed
    coverage over all distributions (Bahadur-Savage), and a zero-variance
    sample is exactly the case where nothing else is left to lean on. An
    infinite interval says "this tells you nothing about the mean", which is
    true; the zero-width one said the opposite. Callers who want a finite
    answer here should pass ``score_range`` -- the interval then narrows to
    the ``degenerate_sample_ci`` bound, whose width is roughly
    ``ln(2/alpha) * 2*(hi-lo)/M``. Emits a ``UserWarning`` saying so, since
    an ``inf`` bound appearing in a report deserves an explanation.
    """
    if diff_bounds is None:
        warnings.warn(
            "Simultaneous CI: a pair's per-input differences have zero "
            "variance (all identical) and the data has no known bounds "
            "(non-binary scores, no score_range given), so its mean cannot "
            "be bounded at any confidence level and the interval is "
            "reported as (-inf, +inf). Pass score_range=(min, max) to get "
            "the finite conservative interval instead.",
            UserWarning,
            stacklevel=3,
        )
        return (float("-inf"), float("inf"))
    lo, hi = float(diff_bounds[0]), float(diff_bounds[1])
    value = float(min(max(point_diff, lo), hi)) if np.isfinite(point_diff) else lo
    return degenerate_sample_ci(value, M, alpha, lo, hi)


def _bonferroni_simultaneous_cis(
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    ci: float,
    diff_bounds: Optional[tuple[float, float]] = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Bonferroni-corrected simultaneous CIs via per-pair paired t-intervals.

    Each CI is recomputed at the Bonferroni-adjusted confidence level
    ``1 − (1−ci)/k`` (where *k* = number of pairs) using the
    ``per_input_diffs`` already stored in each :class:`PairedDiffResult`.
    This makes the result independent of the original CI method, so it
    works as a universal fallback for non-bootstrap methods such as
    ``'newcombe'``, ``'mj_floor'``, ``'tango'``, and ``'bayes_binary'``.

    It is also the *only* construction that runs for a **single pair**
    (k=1): :func:`_simultaneous_cis_router` gates Sidak/boot on
    ``len(pairs) > 1``, so a two-arm comparison lands here unconditionally.
    That makes the degenerate branches below load-bearing for the most
    common shape of comparison there is, not just an edge case in a large
    family -- they used to return ``(point_diff, point_diff)``, so
    ``compare()`` on two arms with a constant offset (arm A ≡ 0.9, arm B ≡
    0.8) reported a zero-width CI at exactly the point estimate, and it
    *overrode* the underlying method's own correct interval on the same
    result object. They now delegate to :func:`_degenerate_pair_ci`.

    Parameters
    ----------
    diff_bounds : tuple[float, float], optional
        Support of a single paired difference, ``(-(hi-lo), hi-lo)`` for a
        metric over ``[lo, hi]``. Used *only* on the degenerate branches;
        the ordinary t-interval path ignores it. ``None`` (the default)
        means no bounds are known -- see :func:`_degenerate_pair_ci` for
        what that yields and why.

    Returns
    -------
    dict[tuple[str, str], tuple[float, float]]
        Maps each pair to its ``(ci_low, ci_high)`` simultaneous CI.
        Returns an empty dict when *pairs* is empty.
    """
    from scipy import stats as _scipy_stats

    k = len(pairs)
    if k == 0:
        return {}

    alpha_adj = (1.0 - ci) / k  # per-comparison alpha after Bonferroni

    sim_cis: dict[tuple[str, str], tuple[float, float]] = {}
    for pair in pairs:
        r = results[pair]
        diffs = r.per_input_diffs
        M = len(diffs)
        if M < 2:
            sim_cis[pair] = _degenerate_pair_ci(
                float(r.point_diff), M, alpha_adj, diff_bounds,
            )
            continue
        se = float(np.std(diffs, ddof=1)) / np.sqrt(M)
        if se < 1e-12:
            sim_cis[pair] = _degenerate_pair_ci(
                float(r.point_diff), M, alpha_adj, diff_bounds,
            )
            continue
        t_crit = float(_scipy_stats.t.ppf(1.0 - alpha_adj / 2.0, df=M - 1))
        half = t_crit * se
        sim_cis[pair] = (float(r.point_diff - half), float(r.point_diff + half))

    return sim_cis


def _sidak_simultaneous_cis(
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    ci: float,
    ci_func: "Callable[[np.ndarray, float], tuple[float, float]]",
) -> dict[tuple[str, str], tuple[float, float]]:
    """Sidak-adjusted simultaneous CIs built from an arbitrary alpha-
    parameterized per-pair CI formula, instead of falling back to
    :func:`_bonferroni_simultaneous_cis`'s generic paired t-interval.

    This is agnostic to which CI construction it widens: *ci_func* is any
    callable ``(diffs, alpha) -> (ci_low, ci_high)`` -- e.g.
    :func:`~evalstats.core.resampling.mj_floor_paired_ci_from_diffs` for binary
    paired data, but equally ``newcombe_mover_paired_ci``, ``t_interval_ci_1d``, or
    any other closed-form interval that accepts a significance level.

    Each pair's CI is *ci_func* evaluated at the Sidak-adjusted per-
    comparison level ``alpha_adj = 1 - (1 - alpha)**(1/k)`` (where *k* =
    number of pairs) -- slightly less conservative than Bonferroni's
    ``alpha/k`` while remaining a closed-form, distribution-agnostic bound
    (no independence assumption beyond what Sidak itself requires). Reuses
    ``per_input_diffs`` already stored in each :class:`PairedDiffResult`, so
    it works regardless of which *method* produced the raw pairwise matrix
    (the caller is responsible for passing a *ci_func* that's actually valid
    for that data -- e.g. a binary-only formula for genuinely binary
    ``per_input_diffs``).

    Returns
    -------
    dict[tuple[str, str], tuple[float, float]]
        Maps each pair to its ``(ci_low, ci_high)`` simultaneous CI.
        Returns an empty dict when *pairs* is empty.
    """
    k = len(pairs)
    if k == 0:
        return {}

    alpha_fam = 1.0 - ci
    alpha_adj = 1.0 - (1.0 - alpha_fam) ** (1.0 / k)  # Sidak-adjusted per-comparison alpha

    return {pair: ci_func(results[pair].per_input_diffs, alpha_adj) for pair in pairs}


def _joint_bootstrap_critical_value(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[str],
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    *,
    statistic: Literal["mean", "median"] = "mean",
    batch_size: int = 128,
) -> Optional[float]:
    """Joint bootstrap critical value for scaling a marginal, alpha-
    parameterized CI to hold simultaneously across *pairs*.

    Resamples the M paired inputs (one shared draw of row indices per
    replicate, applied to every pair at once) so the joint distribution of
    the max standardized statistic accounts for correlation between
    comparisons -- the same mechanism :func:`_max_stat_simultaneous_cis`
    uses, but returning the raw critical value *c* instead of building a
    symmetric Wald CI from it, so a caller can substitute *c* into any
    alpha-parameterized closed-form CI formula instead (see
    :func:`_joint_bootstrap_scaled_simultaneous_cis`). Not tied to any
    particular estimand -- *statistic* controls whether each bootstrap
    replicate's per-pair value is the mean or median of the (collapsed,
    non-seeded) per-input differences, matching
    :func:`_max_stat_simultaneous_cis`'s non-seeded path.

    Returns ``None`` when there are no pairs or every pair is degenerate
    (zero bootstrap variance).
    """
    k = len(pairs)
    if k == 0:
        return None

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    flat = scores.mean(axis=2) if scores.ndim == 3 else scores  # (N, M)

    pair_indices = [(label_to_idx[a], label_to_idx[b]) for (a, b) in pairs]
    diffs_mat = np.stack([flat[i] - flat[j] for (i, j) in pair_indices], axis=0)  # (k, M)
    M = diffs_mat.shape[1]
    point_ests = diffs_mat.mean(axis=1) if statistic == "mean" else np.median(diffs_mat, axis=1)  # (k,)

    input_idx = rng.integers(0, M, size=(n_bootstrap, M))  # (B, M)
    diffs_by_input = diffs_mat.T  # (M, k) -- cache-friendly row access per resample

    boot_stats = np.empty((n_bootstrap, k))
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        chunk = diffs_by_input[input_idx[start:end]]  # (batch, M, k)
        boot_stats[start:end] = chunk.mean(axis=1) if statistic == "mean" else np.median(chunk, axis=1)

    se = np.std(boot_stats, axis=0, ddof=1)
    valid = se > 1e-12
    if not np.any(valid):
        return None

    se_safe = np.where(valid, se, 1.0)
    T = (boot_stats - point_ests[np.newaxis, :]) / se_safe[np.newaxis, :]
    M_b = np.max(np.abs(T[:, valid]), axis=1)
    return float(np.quantile(M_b, ci))


#: Resample cap for _calibrated_joint_critical_value -- see its use there.
_CALIBRATED_JOINT_MAX_RESAMPLES = 1500


def _two_sided_alpha_to_z(a: float) -> float:
    """z such that 2*(1-Phi(z)) == a -- the inverse of the alpha_eff step in
    _calibrated_joint_simultaneous_cis, so an exactly-calibrated alpha survives
    the round trip through that conversion unchanged."""
    from scipy import stats as _st
    return float(_st.norm.ppf(a / 2.0))


def _calibrated_joint_critical_value(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[str],
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    ci_func: "Callable[[np.ndarray, float], tuple[float, float]]",
    *,
    statistic: Literal["mean", "median"] = "mean",
    alpha_ref: float = 0.05,
) -> Optional[float]:
    """Joint critical value studentized by *ci_func's own* centre and scale.

    :func:`_joint_bootstrap_critical_value` standardizes each replicate by the
    BOOTSTRAP standard error of the point estimate, then
    :func:`_joint_bootstrap_scaled_simultaneous_cis` converts the resulting
    *c* to ``alpha_eff = 2(1-Phi(c))`` and evaluates ``ci_func`` there. That
    composition is only exact when ``ci_func(., a)`` has coverage exactly
    ``1-a``. When the formula is marginally conservative -- Bonett-Price's
    Laplace adjustment measures ``1-a+delta`` with delta up to +4.3pp at
    n=10, decaying to ~+0.2pp by n=100 -- the simultaneous interval inherits
    that conservatism on top of the multiplicity widening.

    This variant removes that assumption by reading the centre and scale off
    ``ci_func`` itself on every replicate::

        lo, hi = ci_func(resampled_diffs_r, alpha_ref)
        m = (lo + hi) / 2                       # the formula's own centre
        s = (hi - lo) / (2 z_{alpha_ref/2})     # the formula's own scale
        z_r = |theta_r - m| / s

    so the returned quantile of ``max_r z_r`` is calibrated against the
    construction's actual finite-sample behaviour, including any centre shift
    (Bonett-Price shrinks the point estimate by n/(n+2), which the bootstrap-SE
    route ignores entirely). Stays method-agnostic: ``ci_func`` is only ever
    called as ``(diffs, alpha)``.

    Costs ``n_bootstrap * k`` calls to *ci_func* -- linear, not the nested
    ``B^2`` a naive recalibration would need, because for an interval whose
    half-width is proportional to the normal quantile the level at which a
    replicate just covers has a closed form.

    Returns ``None`` when there are no pairs or every pair is degenerate.
    """
    from scipy import stats as _scipy_stats

    k = len(pairs)
    if k == 0:
        return None

    z_ref = float(_scipy_stats.norm.ppf(1.0 - alpha_ref / 2.0))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    flat = scores.mean(axis=2) if scores.ndim == 3 else scores  # (N, M)
    pair_indices = [(label_to_idx[a], label_to_idx[b]) for (a, b) in pairs]
    diffs_mat = np.stack([flat[i] - flat[j] for (i, j) in pair_indices], axis=0)  # (k, M)
    M = diffs_mat.shape[1]
    point_ests = diffs_mat.mean(axis=1) if statistic == "mean" else np.median(diffs_mat, axis=1)

    # Same degeneracy rule as _joint_bootstrap_critical_value: a pair with no
    # spread contributes an unbounded standardized deviation and would other-
    # wise dominate every replicate's max.
    spread = np.ptp(diffs_mat, axis=1)
    valid = spread > 1e-12
    if not np.any(valid):
        return None

    # The calibration only needs a (1-alpha) quantile of a max over k pairs,
    # which stabilizes well before the resample count `boot` uses for its SE.
    # Capped because this loop costs n_cal * k calls into ci_func (Python-level,
    # since ci_func is an arbitrary callable), vs `boot`'s fully vectorized
    # resample -- uncapped at n_bootstrap=5000 it runs ~100x slower than boot
    # for no measurable gain in the quantile.
    n_cal = int(min(n_bootstrap, _CALIBRATED_JOINT_MAX_RESAMPLES))
    input_idx = rng.integers(0, M, size=(n_cal, M))

    # Fast path: a formula may publish `centre_scale_batch`, evaluating its own
    # centre and scale over a whole (n_cal, M) resample matrix in one numpy
    # call instead of n_cal scalar calls per pair. Two wins, not one:
    #   - ~485x faster on the inner loop (this is otherwise 1500 * k Python
    #     calls; at k=20 that is 285k of them);
    #   - EXACT, where the fallback is not. The fallback recovers scale as
    #     (hi - lo) / (2 z_ref), which understates it whenever the interval
    #     clipped at its bounds -- exactly the sparse small-n case this
    #     calibration exists for.
    # EXACT path: a formula may publish `alpha_crit_batch`, giving the level at
    # which each resample's interval just covers a target. That is the quantity
    # this calibration actually wants, and it needs no reference-distribution
    # assumption of ours: the formula answers in its own parameterization
    # (Bonett-Price normal on the difference scale, NIG t at df=2*a_n, logit-t t
    # at df=n-1 on the LOGIT scale). Joint coverage at a' is P(a' <= min_r
    # alpha_crit), so alpha* is the alpha-quantile of those per-replicate minima.
    # The centre/scale route below cannot express the logit-t case at all -- it
    # assumes symmetry on the difference scale -- so it is an approximation
    # there, not just a slower path.
    acrit = getattr(ci_func, "alpha_crit_batch", None)
    if acrit is not None:
        a_min = np.ones(n_cal, dtype=float)
        for r in range(k):
            if not valid[r]:
                continue
            a_r = np.asarray(acrit(diffs_mat[r][input_idx], float(point_ests[r])), dtype=float)
            np.minimum(a_min, a_r, out=a_min)
        alpha_star = float(np.quantile(a_min, 1.0 - ci))
        alpha_star = min(max(alpha_star, 1e-12), 1.0 - 1e-12)
        return -float(_two_sided_alpha_to_z(alpha_star))

    batch = getattr(ci_func, "centre_scale_batch", None)
    if batch is not None:
        z_max = np.zeros(n_cal)
        for r in range(k):
            if not valid[r]:
                continue
            centre, scale = batch(diffs_mat[r][input_idx], alpha_ref)
            centre = np.asarray(centre, dtype=float)
            scale = np.asarray(scale, dtype=float)
            ok = np.isfinite(centre) & np.isfinite(scale) & (scale > 1e-12)
            if not np.any(ok):
                continue
            z_r = np.zeros(n_cal)
            z_r[ok] = np.abs(point_ests[r] - centre[ok]) / scale[ok]
            np.maximum(z_max, z_r, out=z_max)
    else:
        z_max = np.empty(n_cal)
        for b in range(n_cal):
            idx = input_idx[b]
            worst = 0.0
            for r in range(k):
                if not valid[r]:
                    continue
                lo, hi = ci_func(diffs_mat[r][idx], alpha_ref)
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                scale = (hi - lo) / (2.0 * z_ref)
                if not np.isfinite(scale) or scale <= 1e-12:
                    continue
                centre = 0.5 * (lo + hi)
                worst = max(worst, abs(point_ests[r] - centre) / scale)
            z_max[b] = worst
    if not np.any(z_max > 0.0):
        return None
    return float(np.quantile(z_max, ci))


def _calibrated_joint_simultaneous_cis(
    scores: np.ndarray,
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    labels: list[str],
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    ci_func: "Callable[[np.ndarray, float], tuple[float, float]]",
    *,
    statistic: Literal["mean", "median"] = "mean",
) -> dict[tuple[str, str], tuple[float, float]]:
    """``boot``, but with the joint level calibrated against *ci_func's* own
    finite-sample behaviour rather than the nominal normal quantile -- see
    :func:`_calibrated_joint_critical_value`. Same output contract as
    :func:`_joint_bootstrap_scaled_simultaneous_cis`.
    """
    from scipy import stats as _scipy_stats

    if not pairs:
        return {}
    c = _calibrated_joint_critical_value(
        scores=scores, pairs=pairs, labels=labels, ci=ci, n_bootstrap=n_bootstrap,
        rng=rng, ci_func=ci_func, statistic=statistic,
    )
    if c is None:
        return {}
    alpha_eff = float(2.0 * (1.0 - _scipy_stats.norm.cdf(c)))
    alpha_eff = min(max(alpha_eff, 1e-9), 1.0 - 1e-9)
    return {pair: ci_func(results[pair].per_input_diffs, alpha_eff) for pair in pairs}


def _joint_bootstrap_scaled_simultaneous_cis(
    scores: np.ndarray,
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    labels: list[str],
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    ci_func: "Callable[[np.ndarray, float], tuple[float, float]]",
    *,
    statistic: Literal["mean", "median"] = "mean",
) -> dict[tuple[str, str], tuple[float, float]]:
    """Simultaneous CIs built by scaling an arbitrary alpha-parameterized
    per-pair CI formula with a joint bootstrap critical value, instead of
    Sidak/Bonferroni's independence-assuming adjustment.

    Like :func:`_sidak_simultaneous_cis`, this is agnostic to which CI
    construction it widens -- *ci_func* is any callable
    ``(diffs, alpha) -> (ci_low, ci_high)``. Bootstraps the paired dataset
    to get the joint distribution of the standardized pairwise statistics
    (:func:`_joint_bootstrap_critical_value`), takes the max over all
    k(k-1)/2 pairs per replicate, and uses the resulting
    ``(1-alpha)``-quantile critical value *c* in place of the marginal
    normal quantile ``z_{alpha/2}`` inside *ci_func* (most closed-form score
    intervals -- e.g. ``mj_floor_paired_ci_from_diffs`` -- derive ``z`` from
    ``alpha`` internally, so translating *c* back to an equivalent
    ``alpha_eff = 2*(1 - Phi(c))`` and evaluating *ci_func* at that level is
    equivalent to substituting *c* for *z* directly). This keeps the
    resulting interval shaped like whatever *ci_func* produces (e.g. an
    asymmetric score interval) rather than falling back to a symmetric Wald
    interval around the point estimate, while still accounting for the
    correlation between comparisons that Sidak/Bonferroni cannot.

    Returns
    -------
    dict[tuple[str, str], tuple[float, float]]
        Maps each pair to its ``(ci_low, ci_high)`` simultaneous CI.
        Returns an empty dict when there are no pairs, or when the joint
        bootstrap critical value is degenerate (all pairs zero-variance).
    """
    from scipy import stats as _scipy_stats

    k = len(pairs)
    if k == 0:
        return {}

    c = _joint_bootstrap_critical_value(
        scores=scores, pairs=pairs, labels=labels, ci=ci, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
    )
    if c is None:
        return {}

    alpha_eff = float(2.0 * (1.0 - _scipy_stats.norm.cdf(c)))
    alpha_eff = min(max(alpha_eff, 1e-9), 1.0 - 1e-9)

    return {pair: ci_func(results[pair].per_input_diffs, alpha_eff) for pair in pairs}


def romano_wolf_stepdown_pvalues(
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    rng: "np.random.Generator",
    *,
    statistic: Literal["mean", "median"] = "mean",
    batch_size: int = 256,
) -> dict[tuple[str, str], float]:
    """Romano & Wolf (2005)'s bootstrap step-down FWER-adjusted p-values.

    Unlike single-step max-T (one joint critical value for every pair, see
    :func:`_max_stat_simultaneous_cis`) or Sidak/Bonferroni (independence-
    assuming, closed-form), the step-down refinement here recomputes the max
    only over pairs not yet rejected at each step -- starting from the pair
    with the largest observed studentized statistic and working down -- which
    strictly dominates single-step corrections in power for the same strong
    FWER guarantee. This is exactly the "recover power lost to a
    correlation-blind correction" case repeated-measures eval designs create,
    since shared items make every pair's diffs correlated. Per
    fig:fwer-decision-tree, this is the recommended p-value-correction
    procedure for k>=3 comparisons at N>=30 (see
    :func:`~evalstats.config.resolve_auto_pvalue_correction_method`).

    Ported from (and structurally identical to -- same shared-index,
    per-replicate-studentized bootstrap-t construction, same step-down
    reduction) the implementation in
    ``simulations/harness/cases/pvalues.py``'s
    ``_bootstrap_t_matrix``/``_stepdown_max_t_pvalues``, used to generate
    this method's own validation numbers. Omits that version's Monte-Carlo-
    throughput micro-optimizations (a BLAS-matmul resampling trick, RNG-draw
    sharing with other corrections computed in the same simulation sweep) --
    unnecessary at :func:`compare`'s scale -- in favor of a plain gather-
    based resample, which the harness's own docstring notes its optimized
    form was independently verified bit-close against. Only the "mean"/
    per-item-bootstrap construction the auto-routing table selects is
    implemented here, not the harness's permutation-based Westfall-Young
    variant.

    Parameters
    ----------
    results : dict[tuple[str, str], PairedDiffResult]
        Already-computed pairwise results (as built by :func:`all_pairwise`);
        only ``per_input_diffs`` is used, so this works regardless of which
        *method* produced the raw pairwise matrix.
    pairs : list[tuple[str, str]]
        All pairs to jointly step down over, in the canonical
        ``(label_a, label_b)`` storage order.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
    statistic : {"mean", "median"}
        Central-tendency statistic each pair's point estimate uses.
    batch_size : int
        Bootstrap replicates processed per chunk (bounds peak memory for
        large ``k * n_bootstrap * n_items`` -- mirrors the chunking already
        used elsewhere in this module, e.g. :func:`_ppi_bootstrap_t_joint_stats`).

    Returns
    -------
    dict[tuple[str, str], float]
        Maps each pair to its Romano-Wolf FWER-adjusted p-value, monotonized
        via a running max along the testing order (the same reformulation
        Holm's own adjusted p-values use) so they are directly comparable to
        alpha. Returns an empty dict when there are no pairs.
    """
    k = len(pairs)
    if k == 0:
        return {}

    diffs_mat = np.stack([results[pair].per_input_diffs for pair in pairs], axis=0)  # (k, m)
    m = diffs_mat.shape[1]

    means = diffs_mat.mean(axis=1) if statistic == "mean" else np.median(diffs_mat, axis=1)
    ses = diffs_mat.std(axis=1, ddof=1) / np.sqrt(m)
    ses_safe = np.where(ses > 1e-12, ses, 1.0)
    t_obs = np.abs(means) / ses_safe

    b_means = np.empty((k, n_bootstrap))
    b_ses_safe = np.empty((k, n_bootstrap))
    start = 0
    while start < n_bootstrap:
        stop = min(start + batch_size, n_bootstrap)
        b = stop - start
        idx = rng.integers(0, m, size=(b, m))  # shared across all pairs
        resampled = diffs_mat[:, idx]  # (k, b, m)
        chunk_means = resampled.mean(axis=2) if statistic == "mean" else np.median(resampled, axis=2)
        chunk_ses = resampled.std(axis=2, ddof=1) / np.sqrt(m)
        b_means[:, start:stop] = chunk_means
        b_ses_safe[:, start:stop] = np.where(chunk_ses > 1e-12, chunk_ses, 1.0)
        start = stop

    t_abs = np.abs((b_means - means[:, np.newaxis]) / b_ses_safe)  # (k, B)

    order = np.argsort(-t_obs)  # descending observed |t|: tested first
    t_abs_sorted = t_abs[order]  # (k, B)
    # suffix_max[step] = max over pairs tested at or after `step` -- the
    # step-down "remaining hypotheses" set, per bootstrap draw.
    suffix_max = np.maximum.accumulate(t_abs_sorted[::-1], axis=0)[::-1]  # (k, B)
    t_obs_sorted = t_obs[order]
    extreme_counts = (suffix_max >= t_obs_sorted[:, np.newaxis]).sum(axis=1)  # (k,)
    raw_step_p_sorted = (extreme_counts + 1) / (n_bootstrap + 1)
    adjusted_sorted = np.minimum(np.maximum.accumulate(raw_step_p_sorted), 1.0)

    adjusted = np.empty(k)
    adjusted[order] = adjusted_sorted
    return {pair: float(adjusted[i]) for i, pair in enumerate(pairs)}


# Methods for which _max_stat_simultaneous_cis can produce bootstrap CIs.
_SIMULTANEOUS_CI_BOOTSTRAP_METHODS = {
    "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
    "permutation", "sign_test", "auto",
}


def canonical_pairwise_ci_func(data_kind: str, diff_bounds, method: Optional[str] = None):
    """The alpha-parameterized per-pair CI formula evalstats reports for
    *data_kind*, as a ``(diffs, alpha) -> (lo, hi)`` callable.

    SINGLE SOURCE OF TRUTH for "which interval does a pairwise difference
    get?". The simultaneous-CI constructions (Sidak, joint-bootstrap
    scaling) must widen the SAME formula the non-simultaneous pairwise path
    would otherwise show, or the two disagree about what a comparison's
    interval even is. Simulation harnesses must call this rather than
    re-listing the formulas, which is how they drift: cases/pvalues.py's
    ``_canonical_ci_func`` had Likert on logit-t after Likert gained its own
    NIG row, and binary on mj_floor after binary moved to Bonett-Price, so
    the published simultaneous-CI numbers were measured on formulas the
    library no longer used for those data kinds.

    Returns ``None`` for "unbounded", whose construction needs a degenerate
    -sample fallback the caller supplies (there are no bounds to fall back
    on -- see the router's else-branch).
    """
    # PREFER the already-resolved pairwise method. The main path resolves
    # method="auto" ONCE (router.analyze -> config.resolve_auto_analyze_methods)
    # and hands the concrete name down; keying off it here means the
    # simultaneous CI widens the very interval the pairwise row reports,
    # instead of a second, independently-derived opinion about this data.
    # data_kind is only the fallback, for resampling methods (bootstrap, bca,
    # ...) that have no closed form to widen.
    _bounded = None
    if diff_bounds is not None:
        _lo_b, _hi_b = diff_bounds

        def _bounded(fn):
            def ci_func(diffs, alpha, _lo=_lo_b, _hi=_hi_b, _f=fn):
                return rescaled_ci(_f, diffs, alpha, _lo, _hi)
            return ci_func

    def _attach(f, provider):
        try:
            f.alpha_crit_batch = provider
        except AttributeError:
            pass
        return f

    if method == "bonett_price":
        return bonett_price_paired_ci_from_diffs
    if method in ("mj_floor", "tango"):
        return mj_floor_paired_ci_from_diffs
    if method == "nig" and _bounded is not None:
        return _attach(_bounded(functools.partial(nig_ci_1d, b0=_NIG_PAIRED_DIFF_B0)),
                       functools.partial(_nig_alpha_crit_batch, b0=_NIG_PAIRED_DIFF_B0,
                                         lo=_lo_b, hi=_hi_b))
    if method == "logit_t" and _bounded is not None:
        return _attach(_bounded(logit_t_ci_1d),
                       functools.partial(_logit_t_alpha_crit_batch, lo=_lo_b, hi=_hi_b))

    if data_kind == "binary":
        return bonett_price_paired_ci_from_diffs
    if data_kind in ("bounded_01", "likert") and _bounded is not None:
        if data_kind == "likert":
            return _attach(_bounded(functools.partial(nig_ci_1d, b0=_NIG_PAIRED_DIFF_B0)),
                           functools.partial(_nig_alpha_crit_batch, b0=_NIG_PAIRED_DIFF_B0,
                                             lo=_lo_b, hi=_hi_b))
        return _attach(_bounded(logit_t_ci_1d),
                       functools.partial(_logit_t_alpha_crit_batch, lo=_lo_b, hi=_hi_b))
    return None


def _simultaneous_cis_router(
    scores: np.ndarray,
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    labels: list[str],
    method: str,
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    statistic: str,
    *,
    prefer: str = "auto",
    score_range: Optional[tuple[float, float]] = None,
    eval_type: Optional[Literal["likert", "continuous"]] = None,
) -> tuple[dict[tuple[str, str], tuple[float, float]], str, dict]:
    """Route simultaneous CI computation to the requested construction.

    ``prefer="auto"`` (default) follows fig:fwer-decision-tree: Sidak for
    small N (binary N<50, numeric N<30) or a lopsided binary split
    regardless of N (see :func:`~evalstats.config.resolve_auto_simultaneous_ci_method`),
    else joint bootstrap with an effective alpha (``"boot"``,
    :func:`_joint_bootstrap_scaled_simultaneous_cis`). Both widen whichever
    canonical closed-form pairwise CI formula the data resolves to -- Bonett-Price
    for binary data, logit-t for any bounded numeric range (*score_range*),
    plain t-interval as the bounds-agnostic fallback for everything else --
    the same per-data-kind formula :data:`~evalstats.config.AUTO_ANALYZE_METHOD_TABLE`
    already uses for the *non*-simultaneous pairwise CI on genuinely
    continuous data, so Sidak/boot always widen the formula that would
    otherwise have been shown, regardless of which resampling *method*
    (bootstrap, bca, ...) the point estimate itself used.

    ``eval_type="likert"`` (or ``method="nig"`` passed directly) widens
    NIG instead of logit-t for bounded numeric data here, matching
    :func:`pairwise_differences`'s own ``method="nig"`` path and
    :data:`~evalstats.config.AUTO_ANALYZE_METHOD_TABLE`'s "likert" row.
    This used to be logit-t-only regardless of ``eval_type`` (the k>=3
    construction had only ever been tested with NIG's OLD, buggy prior,
    before ``_NIG_PAIRED_DIFF_B0`` fixed it) -- a real compare_e2e
    overnight sweep surfaced exactly the failure that scoping predicted:
    family-wise coverage for likert data collapsing to 10-26% at n=15,
    k=10 (vs. 93-99% for continuous at the same n, k), because Sidak's
    shrinking per-pair alpha_adj as k grows drives logit-t straight into
    the same paired-diff rounding-cancellation failure mode NIG was built
    to fix in the first place -- this router just hadn't been made to use
    the fix. See :data:`~evalstats.config.AUTO_ANALYZE_METHOD_TABLE`'s
    "likert" row for the validation numbers (single-run and nested/multi-
    run alike -- unlike the pairwise *method* table, this router doesn't
    vary by seeded= at all, since Sidak/boot here widen whatever
    ``results[pair].per_input_diffs`` already is, computed identically
    for single- and multi-run data upstream).

    Historical note: this used to default unconditionally to Bonferroni
    (with the studentized bootstrap max-T method as the sole opt-in
    alternative via ``prefer="max_t"``) -- max-T's ``bootstrap_t``
    studentization has a documented instability at small N combined with
    many simultaneous comparisons (resampling just N points to re-estimate
    a per-replicate SE gets noisy, and taking a max over k(k-1)/2 pairs
    multiplies the chances of hitting a near-zero denominator on any one
    replicate), which is why it stayed opt-in rather than becoming this
    table's default. Sidak/boot are unrelated to that instability -- they
    don't restudentize per replicate -- and are validated in
    ``simulations/harness/cases/pvalues.py`` (``--mode simultaneous_ci``,
    ``CORR_SIDAK``/``CORR_BOOT``), which imports
    :func:`_sidak_simultaneous_cis`/:func:`_joint_bootstrap_scaled_simultaneous_cis`
    directly to generate the paper's fig:fwer-decision-tree numbers.

    Pass ``prefer="max_t"`` to opt into the (separate, tree-unrelated)
    studentized bootstrap construction for bootstrap-compatible methods, or
    ``prefer="bonferroni"``/``"sidak"``/``"boot"`` to force one specific
    construction directly. Any requested construction falls back to
    Bonferroni if it returns an empty/degenerate result (e.g. max-T on a
    non-bootstrap-compatible *method*, or a joint bootstrap with zero
    variance on every pair).

    Returns
    -------
    tuple[dict, str, dict]
        ``(cis, method_used, max_t_pvalues)`` where *method_used* is one of
        ``'sidak'``, ``'boot'``, ``'max_t'``, or ``'bonferroni'`` (fallback).
        *max_t_pvalues* maps each pair to its max-T p-value only when
        *method_used* is ``'max_t'``; empty dict otherwise.
    """
    # Resolve the data kind once, up front, rather than inside the Sidak/boot
    # branch: every route can end at the Bonferroni fallback (max-T returning
    # empty, a joint bootstrap that degenerates, prefer="bonferroni", or a
    # single pair, which skips Sidak/boot by construction), and that fallback
    # needs the diff bounds too -- they are what lets it produce a real
    # interval instead of a zero-width one on a constant difference vector.
    # See _degenerate_pair_ci.
    #
    # Same explicit-score_range-wins rule as the main router uses -- see
    # resampling.binary_routing_applies. The warning is emitted there, at
    # the routing decision, not repeated per pair.
    is_binary = binary_routing_applies(scores, score_range)
    if is_binary:
        data_kind = "binary"
    elif score_range is not None:
        # eval_type="likert" (explicit or auto-resolved upstream via
        # detect_quantization_step()) or an explicit method="nig" call
        # both route to the NIG-widened branch below -- see this
        # function's docstring for why (validated fix for logit-t's
        # paired-diff rounding-cancellation failure mode, which gets
        # worse, not better, as Sidak's alpha_adj shrinks with k).
        data_kind = "likert" if (eval_type == "likert" or method == "nig") else "bounded_01"
    else:
        data_kind = "unbounded"

    # Support of a single paired difference: two scores in [lo, hi] differ by
    # at most hi-lo in either direction, so the diff spans [-(hi-lo), hi-lo]
    # -- the same widened span the logit_t/NIG paths rescale onto. Binary
    # data is [0, 1] whether or not a score_range was passed.
    if data_kind == "binary":
        diff_bounds: Optional[tuple[float, float]] = (-1.0, 1.0)
    elif score_range is not None:
        _span = float(score_range[1]) - float(score_range[0])
        diff_bounds = (-_span, _span)
    else:
        diff_bounds = None

    if prefer == "max_t" and method in _SIMULTANEOUS_CI_BOOTSTRAP_METHODS:
        cis, max_t_pvalues = _max_stat_simultaneous_cis(
            scores=scores,
            pairs=pairs,
            labels=labels,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            rng=rng,
            statistic=statistic,
        )
        if cis:
            return cis, "max_t", max_t_pvalues

    elif prefer in ("auto", "sidak", "boot", "boot_cal") and len(pairs) > 1:
        # fig:fwer-decision-tree's Sidak/boot branch is explicitly scoped to
        # "Family of comparisons (k>=3)" -- with a single pair (k=2, one
        # comparison), there's no family to control FWER across, and
        # Bonferroni/Sidak's adjustment is already an exact no-op at k=1
        # (alpha_adj = 1-(1-alpha)**(1/1) = alpha). "boot" doesn't degenerate
        # as cleanly (it studentizes via a joint bootstrap max, then
        # translates the resulting critical value back through the normal
        # CDF to get alpha_eff -- a step that can materially misestimate
        # alpha_eff, and therefore over- or under-widen the interval, for
        # non-normal single-pair resampling distributions, e.g. a heavily
        # outlier-contaminated median), so route k=1 through the plain
        # Bonferroni fallback below rather than attempting the k>=3-only
        # constructions at all.
        resolved = prefer
        if prefer == "auto":
            n_items = scores.shape[1]
            lopsided = is_binary and is_lopsided_binary(scores)
            resolved = resolve_auto_simultaneous_ci_method(
                data_kind, n_items, lopsided_binary=lopsided,
            )

        ci_func = canonical_pairwise_ci_func(data_kind, diff_bounds, method)
        if ci_func is None:
            # Unbounded: t_interval_ci_1d still returns (mean, mean) on a
            # constant difference vector -- the marginal contract that
            # degenerate_sample_ci deliberately left alone, since with no
            # bounds there is nothing for it to fall back on. Left as-is,
            # that reintroduces exactly the zero-width interval this router's
            # Bonferroni fallback now refuses to emit, on any family where
            # sidak/boot succeed (k>=3 with at least one non-degenerate pair
            # to carry the joint bootstrap). Route the degenerate case
            # through the same _degenerate_pair_ci the fallback uses so both
            # branches give one answer, without changing t_interval_ci_1d
            # itself or any marginal path that depends on it.
            def ci_func(diffs, alpha):
                M = len(diffs)
                if M < 2 or float(np.ptp(diffs)) == 0.0:
                    return _degenerate_pair_ci(
                        float(np.mean(diffs)) if M else 0.0, M, alpha, None,
                    )
                return t_interval_ci_1d(diffs, alpha)

        if resolved == "sidak":
            cis = _sidak_simultaneous_cis(results=results, pairs=pairs, ci=ci, ci_func=ci_func)
            if cis:
                return cis, "sidak", {}
        elif resolved == "boot_cal":
            cis = _calibrated_joint_simultaneous_cis(
                scores=scores, results=results, pairs=pairs, labels=labels,
                ci=ci, n_bootstrap=n_bootstrap, rng=rng, ci_func=ci_func, statistic=statistic,
            )
            if cis:
                return cis, "boot_cal", {}
        elif resolved == "boot":
            cis = _joint_bootstrap_scaled_simultaneous_cis(
                scores=scores, results=results, pairs=pairs, labels=labels,
                ci=ci, n_bootstrap=n_bootstrap, rng=rng, ci_func=ci_func, statistic=statistic,
            )
            if cis:
                return cis, "boot", {}

    # Fallback (and prefer="bonferroni"): Bonferroni t-intervals work for any
    # method. diff_bounds only affects its zero-variance branch, where it is
    # the difference between a real interval and a zero-width one.
    cis = _bonferroni_simultaneous_cis(
        results=results, pairs=pairs, ci=ci, diff_bounds=diff_bounds,
    )
    return cis, "bonferroni", {}


def all_pairwise(
    scores: np.ndarray,
    labels: list[str],
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "mj_floor", "tango", "bayes_binary", "permutation", "sign_test", "t_interval", "logit_t", "nig"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"] = "auto",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    simultaneous_ci: bool = True,
    omnibus: bool = False,
    multi_ci: bool = False,
    compute_wilcoxon: bool = True,
    score_range: Optional[tuple[float, float]] = None,
    prefer: str = "auto",
    eval_type: Optional[Literal["likert", "continuous"]] = None,
) -> PairwiseMatrix:
    """Compute all pairwise comparisons with multiple comparisons correction.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` each comparison uses the nested bootstrap.
    labels : list[str]
        Template labels.
    method : str
        Statistical test method.
    ci : float
        Confidence level.
    n_bootstrap : int
        Number of bootstrap resamples.
    correction : str
        p-value correction across the k(k-1)/2 pairwise comparisons.
        ``'auto'`` (default) follows fig:fwer-decision-tree: Shaffer's
        modified step-down Holm procedure for N < 30 (or a lopsided binary
        split regardless of N -- see
        :func:`~evalstats.config.resolve_auto_pvalue_correction_method`),
        else Romano-Wolf bootstrap step-down. Explicit alternatives:
        ``'shaffer'``, ``'romano_wolf'``, ``'holm'``, ``'bonferroni'``,
        ``'fdr_bh'`` (Benjamini-Hochberg FDR control, not FWER -- use when
        the false discovery rate rather than the family-wise error rate is
        the actual target), ``'hochberg'``, or ``'none'``.

        ``'romano_wolf'`` needs genuine per-pair resampling (see
        :func:`romano_wolf_stepdown_pvalues`) and so only corrects the
        primary bootstrap-derived p-value; the companion Wilcoxon p-value
        (``PairedDiffResult.wilcoxon_p``) falls back to Shaffer's for that
        one field, since there's no validated Romano-Wolf-on-Wilcoxon
        construction.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'mean'`` (default) or
        ``'median'``.
    simultaneous_ci : bool
        When ``True``, replace individual pairwise CIs with simultaneous
        (family-wise) CIs. Routes through :func:`_simultaneous_cis_router`
        with ``prefer="auto"``, which follows fig:fwer-decision-tree: Sidak
        for small N (binary N<50, numeric N<30) or a lopsided binary split
        regardless of N, else joint bootstrap with an effective alpha
        (``'boot'``) -- see that function's docstring for the full
        rationale and the (separate, tree-unrelated) studentized bootstrap
        max-T alternative available via a direct ``prefer="max_t"`` call.

        The method actually used is recorded in
        :attr:`PairwiseMatrix.simultaneous_ci_method` (``'sidak'``,
        ``'boot'``, ``'max_t'``, or ``'bonferroni'`` as the ultimate
        fallback).
    prefer : str
        Passed through to :func:`_simultaneous_cis_router`'s ``prefer=``
        knob to force a specific simultaneous-CI construction instead of
        the ``"auto"`` (default) table lookup: ``"sidak"``, ``"boot"``,
        ``"max_t"``, or ``"bonferroni"``.
    eval_type : "likert", "continuous", or None
        ``"likert"`` (explicit, or resolved via ``method="auto"``'s own
        quantization auto-detection) routes BOTH the per-pair CI (via
        :func:`pairwise_differences`'s ``method="nig"`` path) AND the
        ``simultaneous_ci=True`` (default) k>=3 Sidak/joint-bootstrap-
        widened construction (:func:`_simultaneous_cis_router`) through
        NIG instead of logit-t -- validated for single-run and nested/
        multi-run pairwise data alike (see
        ``config.AUTO_ANALYZE_METHOD_TABLE``'s "likert" row). Continuous
        (or unspecified) bounded numeric data keeps logit-t throughout.
    omnibus : bool
        When ``True``, run the Friedman omnibus test (with Nemenyi post-hoc)
        alongside the pairwise comparisons.  Requires k ≥ 3.  Defaults to
        ``False`` — the Friedman test is a NHST procedure that may not be
        desirable in estimation-focused workflows.  The result is stored in
        :attr:`PairwiseMatrix.friedman`.
    compute_wilcoxon : bool
        Forwarded to each :func:`pairwise_differences` call (default
        ``True``). Set ``False`` to skip the supplementary Wilcoxon
        signed-rank p-value for every pair -- e.g. for high-volume Monte
        Carlo callers that never read ``PairedDiffResult.wilcoxon_p``.

    Returns
    -------
    PairwiseMatrix
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(labels)
    results = {}
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            result = pairwise_differences(
                scores, i, j, labels[i], labels[j],
                method=method, ci=ci, n_bootstrap=n_bootstrap, rng=rng,
                statistic=statistic, multi_ci=multi_ci, compute_wilcoxon=compute_wilcoxon,
                score_range=score_range,
            )
            results[(labels[i], labels[j])] = result
            pairs.append((labels[i], labels[j]))

    # Apply multiple comparisons correction to bootstrap p-values (and Wilcoxon if available).
    resolved_correction = correction
    if correction == "auto":
        _is_binary = is_binary_scores(scores)
        _lopsided = _is_binary and is_lopsided_binary(scores)
        resolved_correction = resolve_auto_pvalue_correction_method(
            scores.shape[1], lopsided_binary=_lopsided,
        )

    if resolved_correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        wsr_pairs = [p for p in pairs if results[p].wilcoxon_p is not None]
        wsr_pvals = (
            np.array([results[p].wilcoxon_p for p in wsr_pairs], dtype=float)
            if len(wsr_pairs) > 1 else None
        )
        # Shaffer's needs the *complete* n_groups*(n_groups-1)/2 all-pairs
        # set -- the companion Wilcoxon field can be a strict subset of that
        # (e.g. undefined for a pair with zero-variance identical diffs),
        # in which case Shaffer's own count check would raise. Holm has no
        # such requirement and is still FWER-valid for any subset, so it's
        # the safe fallback specifically for that partial set.
        _wsr_is_complete = len(wsr_pairs) == n * (n - 1) // 2
        _wsr_method = lambda m: m if (m != "shaffer" or _wsr_is_complete) else "holm"

        if resolved_correction == "romano_wolf":
            # Needs genuine per-pair resampling (see
            # romano_wolf_stepdown_pvalues), so only the primary bootstrap-
            # derived p-value gets the validated step-down correction. The
            # companion Wilcoxon p-value falls back to Shaffer's for that
            # one field -- there's no validated Romano-Wolf-on-Wilcoxon
            # construction (see all_pairwise's correction= docstring).
            _rw_adj = romano_wolf_stepdown_pvalues(
                results, pairs, n_bootstrap, rng, statistic=statistic,
            )
            adjusted = np.array([_rw_adj[p] for p in pairs])
            wsr_adj_map = (
                dict(zip(wsr_pairs, correct_pvalues(wsr_pvals, _wsr_method("shaffer"), n_groups=n)))
                if wsr_pvals is not None
                else {p: results[p].wilcoxon_p for p in wsr_pairs}
            )
        else:
            adjusted = correct_pvalues(p_values, resolved_correction, n_groups=n)
            wsr_adj_map = (
                dict(zip(wsr_pairs, correct_pvalues(wsr_pvals, _wsr_method(resolved_correction), n_groups=n)))
                if wsr_pvals is not None
                else {p: results[p].wilcoxon_p for p in wsr_pairs}
            )

        for pair, adj_p in zip(pairs, adjusted):
            r = results[pair]
            adj_wsr = wsr_adj_map.get(pair, r.wilcoxon_p)
            results[pair] = PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                point_diff=r.point_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=r.test_method,
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=float(adj_wsr) if adj_wsr is not None else None,
                agreement_mcc=r.agreement_mcc,
                binary_confusion=r.binary_confusion,
                multi_ci=r.multi_ci,
            )

    # Simultaneous CIs: Sidak/joint-bootstrap by default, per
    # fig:fwer-decision-tree (see _simultaneous_cis_router's docstring).
    applied_simultaneous_ci = False
    applied_simultaneous_ci_method: Optional[str] = None
    if simultaneous_ci and len(pairs) > 0:
        sim_cis, sim_method, sim_pvalues = _simultaneous_cis_router(
            scores=scores,
            results=results,
            pairs=pairs,
            labels=labels,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            rng=rng,
            statistic=statistic,
            score_range=score_range,
            prefer=prefer,
            eval_type=eval_type,
        )
        if sim_cis:
            applied_simultaneous_ci = True
            applied_simultaneous_ci_method = sim_method
            ci_label = {
                "max_t": "simultaneous CIs computed with max-T",
                "sidak": "simultaneous CIs computed with Sidak's procedure",
                "boot": "simultaneous CIs computed with a joint bootstrap (effective alpha)",
            }.get(sim_method, "simultaneous CIs computed with Bonferroni")
            for pair, (ci_low, ci_high) in sim_cis.items():
                r = results[pair]
                results[pair] = PairedDiffResult(
                    template_a=r.template_a,
                    template_b=r.template_b,
                    point_diff=r.point_diff,
                    std_diff=r.std_diff,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=(
                        sim_pvalues.get(pair, r.p_value)
                        if sim_method == "max_t" and method in {
                            "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto",
                        }
                        else r.p_value
                    ),
                    test_method=r.test_method,
                    n_inputs=r.n_inputs,
                    per_input_diffs=r.per_input_diffs,
                    n_runs=r.n_runs,
                    statistic=r.statistic,
                    wilcoxon_p=r.wilcoxon_p,
                    agreement_mcc=r.agreement_mcc,
                    binary_confusion=r.binary_confusion,
                    multi_ci=r.multi_ci,
                )

    # Friedman omnibus + Nemenyi post-hoc (only when explicitly requested).
    friedman: Optional[FriedmanResult] = None
    if omnibus and len(labels) >= 3:
        try:
            friedman = friedman_nemenyi(scores, labels)
        except Exception:
            pass

    return PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method=resolved_correction,
        friedman=friedman,
        simultaneous_ci=applied_simultaneous_ci,
        simultaneous_ci_method=applied_simultaneous_ci_method,
    )


def vs_baseline(
    scores: np.ndarray,
    labels: list[str],
    baseline: str,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "mj_floor", "tango", "bayes_binary", "permutation", "sign_test", "t_interval", "logit_t", "nig"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    score_range: Optional[tuple[float, float]] = None,
) -> list[PairedDiffResult]:
    """Compare all templates against a designated baseline.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
    labels : list[str]
        Template labels.
    baseline : str
        Label of the baseline template.
    method, ci, n_bootstrap, correction, rng :
        Same as ``all_pairwise``.
    statistic : str
        Point-estimate and bootstrap statistic: ``'mean'`` (default) or
        ``'median'``.

    Returns
    -------
    list[PairedDiffResult]
        One result per non-baseline template.
    """
    if rng is None:
        rng = np.random.default_rng()

    baseline_idx = labels.index(baseline)
    results = []

    for i, label in enumerate(labels):
        if i == baseline_idx:
            continue
        result = pairwise_differences(
            scores, i, baseline_idx, label, baseline,
            method=method, ci=ci, n_bootstrap=n_bootstrap, rng=rng,
            statistic=statistic, score_range=score_range,
        )
        results.append(result)

    # Apply correction to bootstrap p-values (and Wilcoxon if available).
    if correction != "none" and len(results) > 1:
        p_values = np.array([r.p_value for r in results])
        adjusted = correct_pvalues(p_values, correction)

        # Correct Wilcoxon p-values independently.
        wsr_results = [r for r in results if r.wilcoxon_p is not None]
        if len(wsr_results) > 1:
            wsr_pvals = np.array([r.wilcoxon_p for r in wsr_results], dtype=float)
            wsr_adj_vals = correct_pvalues(wsr_pvals, correction)
            wsr_adj_map = {
                (r.template_a, r.template_b): float(v)
                for r, v in zip(wsr_results, wsr_adj_vals)
            }
        else:
            wsr_adj_map = {
                (r.template_a, r.template_b): r.wilcoxon_p for r in wsr_results
            }

        results = [
            PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                point_diff=r.point_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=wsr_adj_map.get((r.template_a, r.template_b), r.wilcoxon_p),
                agreement_mcc=r.agreement_mcc,
                binary_confusion=r.binary_confusion,
                multi_ci=r.multi_ci,
            )
            for r, adj_p in zip(results, adjusted)
        ]

    return results

