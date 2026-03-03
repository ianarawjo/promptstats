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

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .resampling import bca_interval_1d, bootstrap_diffs_nested, bootstrap_means_1d, resolve_resampling_method, _stat
from .stats_utils import correct_pvalues


def _wilcoxon_signed_rank_p(diffs: np.ndarray) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value for per-input paired differences.

    Uses ``zero_method='wilcox'`` (discards zero differences before ranking),
    which is the most common convention.  Returns ``None`` if the test cannot
    be computed (all differences are zero, or fewer than one non-zero pair).
    """
    from scipy.stats import wilcoxon  # scipy is a core dep; import here to keep top-level clean

    if int(np.sum(diffs != 0)) < 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return None


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

    @property
    def effect_size(self) -> float:
        """Paired effect size: point_diff / std_diff."""
        if self.std_diff == 0:
            return float("inf") if self.point_diff != 0 else 0.0
        return self.point_diff / self.std_diff


@dataclass
class PairwiseMatrix:
    """Results of all pairwise comparisons."""

    labels: list[str]
    results: dict[tuple[str, str], PairedDiffResult]
    correction_method: str

    def get(self, a: str, b: str) -> PairedDiffResult:
        """Get the comparison result for templates a vs b."""
        if (a, b) in self.results:
            return self.results[(a, b)]
        if (b, a) in self.results:
            r = self.results[(b, a)]
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
            )
        raise KeyError(f"No comparison found for ({a}, {b})")

    def point_diff_matrix(self) -> np.ndarray:
        """Return NxN matrix of point-estimate differences (mean or median)."""
        n = len(self.labels)
        mat = np.zeros((n, n))
        for i, a in enumerate(self.labels):
            for j, b in enumerate(self.labels):
                if i != j:
                    mat[i, j] = self.get(a, b).point_diff
        return mat


def pairwise_differences(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str = "A",
    label_b: str = "B",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "median",
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
        Statistical method: ``'auto'`` (default), ``'bootstrap'``, or
        ``'bca'``.  ``'auto'`` selects BCa for 15 ≤ M ≤ 200.
    ci : float
        Confidence level for the interval (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'median'`` (default) or
        ``'mean'``.  Median is preferred for LLM score distributions, which
        are frequently non-normal.

    Returns
    -------
    PairedDiffResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Route: seeded (R >= 3) vs. standard (2-D or R < 3)                 #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        R = scores.shape[2]
        if R >= 3:
            return _pairwise_diffs_seeded(
                scores, idx_a, idx_b, label_a, label_b,
                method=method, ci=ci, n_bootstrap=n_bootstrap, rng=rng,
                statistic=statistic,
            )
        # R == 1 or R == 2: collapse to 2-D (warning already issued during validation)
        scores = scores.mean(axis=2)

    # ------------------------------------------------------------------ #
    # Standard (non-seeded) path                                          #
    # ------------------------------------------------------------------ #
    diffs = scores[idx_a] - scores[idx_b]
    m = len(diffs)
    point_d = _stat(diffs, statistic)
    std_d = float(np.std(diffs, ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, m)

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
        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        extreme_count = np.sum(np.abs(boot_centered_stats) >= abs(point_d))
        p_value = float((extreme_count + 1) / (n_bootstrap + 1))
        test_name = f"bootstrap (n={n_bootstrap})"

    elif resolved_method == "bca":
        boot_stats = bootstrap_means_1d(
            diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        ci_low, ci_high = bca_interval_1d(
            diffs, point_d, boot_stats, alpha, statistic=statistic,
        )
        centered_diffs = diffs - point_d
        boot_centered_stats = bootstrap_means_1d(
            centered_diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        extreme_count = np.sum(np.abs(boot_centered_stats) >= abs(point_d))
        p_value = float((extreme_count + 1) / (n_bootstrap + 1))
        test_name = f"bca bootstrap (n={n_bootstrap})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    wilcoxon_p = _wilcoxon_signed_rank_p(diffs)

    return PairedDiffResult(
        template_a=label_a,
        template_b=label_b,
        point_diff=point_d,
        std_diff=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_method=test_name,
        n_inputs=m,
        per_input_diffs=diffs,
        n_runs=1,
        statistic=statistic,
        wilcoxon_p=wilcoxon_p,
    )


def _pairwise_diffs_seeded(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str,
    label_b: str,
    *,
    method: Literal["bootstrap", "bca", "auto"],
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
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

    # Nested bootstrap replicates of the statistic of paired cell-mean differences.
    boot_stats = bootstrap_diffs_nested(
        scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
    )

    if resolved_method == "bootstrap":
        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        # Null-centred: shift bootstrap distribution to have statistic = 0.
        boot_centered = boot_stats - point_d
        extreme_count = np.sum(np.abs(boot_centered) >= abs(point_d))
        p_value = float((extreme_count + 1) / (n_bootstrap + 1))
        test_name = f"nested bootstrap (n={n_bootstrap}, R={R})"

    elif resolved_method == "bca":
        # BCa: jackknife over inputs (the outer sampling unit) using cell_diffs.
        ci_low, ci_high = bca_interval_1d(
            cell_diffs, point_d, boot_stats, alpha, statistic=statistic,
        )
        boot_centered = boot_stats - point_d
        extreme_count = np.sum(np.abs(boot_centered) >= abs(point_d))
        p_value = float((extreme_count + 1) / (n_bootstrap + 1))
        test_name = f"nested bca bootstrap (n={n_bootstrap}, R={R})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    wilcoxon_p = _wilcoxon_signed_rank_p(cell_diffs)

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
    )


def all_pairwise(
    scores: np.ndarray,
    labels: list[str],
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "median",
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
        Multiple comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'median'`` (default) or
        ``'mean'``.

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
                statistic=statistic,
            )
            results[(labels[i], labels[j])] = result
            pairs.append((labels[i], labels[j]))

    # Apply multiple comparisons correction to bootstrap p-values (and Wilcoxon if available).
    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)

        # Correct Wilcoxon p-values independently (only for pairs where the test ran).
        wsr_pairs = [p for p in pairs if results[p].wilcoxon_p is not None]
        if len(wsr_pairs) > 1:
            wsr_pvals = np.array([results[p].wilcoxon_p for p in wsr_pairs], dtype=float)
            wsr_adj_map = dict(zip(wsr_pairs, correct_pvalues(wsr_pvals, correction)))
        else:
            wsr_adj_map = {p: results[p].wilcoxon_p for p in wsr_pairs}

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
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=float(adj_wsr) if adj_wsr is not None else None,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


def vs_baseline(
    scores: np.ndarray,
    labels: list[str],
    baseline: str,
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "median",
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
        Point-estimate and bootstrap statistic: ``'median'`` (default) or
        ``'mean'``.

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
            statistic=statistic,
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
            )
            for r, adj_p in zip(results, adjusted)
        ]

    return results

