"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy import stats


def _stat(values: np.ndarray, statistic: Literal["mean", "median"]) -> float:
    """Apply *statistic* to a 1-D array and return a Python float."""
    if statistic == "median":
        return float(np.median(values))
    return float(np.mean(values))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of *values* given *weights* (must sum to 1)."""
    sorted_idx = np.argsort(values)
    cumsum = np.cumsum(weights[sorted_idx])
    idx = int(np.searchsorted(cumsum, 0.5))
    return float(values[sorted_idx[min(idx, len(values) - 1)]])


def _weighted_medians_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted medians for 2-D *values* and matching *weights*."""
    sorted_idx = np.argsort(values, axis=1)
    sorted_vals = np.take_along_axis(values, sorted_idx, axis=1)
    sorted_w = np.take_along_axis(weights, sorted_idx, axis=1)
    cumsum_w = np.cumsum(sorted_w, axis=1)
    med_idx = np.argmax(cumsum_w >= 0.5, axis=1)
    row_idx = np.arange(values.shape[0])
    return sorted_vals[row_idx, med_idx]


def _percentile_interval(boot_stats: np.ndarray, alpha: float) -> tuple[float, float]:
    """Equal-tailed percentile interval from bootstrap replicates."""
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


def _reduce_rows(values: np.ndarray, statistic: Literal["mean", "median"]) -> np.ndarray:
    """Reduce 2-D rows via mean or median."""
    if statistic == "median":
        return np.median(values, axis=1)
    return values.mean(axis=1)


def _warn_smooth_bootstrap_fallback(function_name: str, reason: str) -> None:
    """Warn that a smooth-bootstrap path fell back to plain bootstrap."""
    warnings.warn(
        f"{function_name} falling back to plain bootstrap; no KDE smoothing applied. Reason: {reason}.",
        UserWarning,
        stacklevel=2,
    )


def _nested_cell_mean_diffs(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    run_idx: np.ndarray,
    input_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Compute bootstrap-wise per-input diffs of inner-resampled cell means.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Arrays of shape ``(M, R)``.
    run_idx : np.ndarray
        Inner resample run indices with shape ``(B, M, R)``.
    input_idx : np.ndarray, optional
        Optional outer input resample indices with shape ``(B, M)``.
        If omitted, no outer input resampling is performed.

    Returns
    -------
    np.ndarray
        Shape ``(B, M)`` containing paired cell-mean differences.
    """
    n_bootstrap, M, _ = run_idx.shape
    if input_idx is None:
        m_range = np.arange(M)[np.newaxis, :, np.newaxis]         # (1, M, 1)
        resampled_a = scores_a[m_range, run_idx]                  # (B, M, R)
        resampled_b = scores_b[m_range, run_idx]                  # (B, M, R)
    else:
        sel_a = scores_a[input_idx]                               # (B, M, R)
        sel_b = scores_b[input_idx]                               # (B, M, R)
        b_range = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]  # (B, 1, 1)
        m_range = np.arange(M)[np.newaxis, :, np.newaxis]         # (1, M, 1)
        resampled_a = sel_a[b_range, m_range, run_idx]            # (B, M, R)
        resampled_b = sel_b[b_range, m_range, run_idx]            # (B, M, R)
    return resampled_a.mean(axis=2) - resampled_b.mean(axis=2)    # (B, M)


def _inner_resample_cell_means(
    scores: np.ndarray,
    run_idx: np.ndarray,
    input_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Inner-resample per-input cell means for scores of shape ``(N, M, R)``."""
    _, M, _ = scores.shape
    selected = scores if input_idx is None else scores[:, input_idx, :]
    m_range = np.arange(M)[:, np.newaxis]                         # (M, 1)
    resampled = selected[:, m_range, run_idx]                     # (N, M, R)
    return resampled.mean(axis=2)                                 # (N, M)


def is_binary_scores(scores: np.ndarray) -> bool:
    """Return True if all finite values in *scores* are exactly 0 or 1.

    Used to auto-detect binary evaluation data so that :func:`analyze` can
    switch to Wilson score intervals (single-sample) and Newcombe score
    intervals (pairwise) rather than the default smooth bootstrap.

    Parameters
    ----------
    scores : np.ndarray
        Any-shape score array.

    Returns
    -------
    bool
    """
    flat = scores.ravel()
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return False
    return bool(np.all((finite == 0.0) | (finite == 1.0)))


def wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes (observations equal to 1).
    n : int
        Total number of trials.
    alpha : float
        Significance level (1 − confidence level).  E.g. 0.05 for a 95% CI.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval clamped to [0, 1].
    """
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt(
        (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))
    )
    return (max(0.0, float(center - radius)), min(1.0, float(center + radius)))


def wilson_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Wilson score CI for a 1-D binary (0/1) array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of binary observations (should contain only 0s and 1s).
    alpha : float
        Significance level (1 − confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    n = len(values)
    successes = int(np.round(np.sum(values)))
    return wilson_ci(successes, n, alpha)


def newcombe_paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Newcombe score CI for the paired binary difference p(A=1) − p(B=1).

    Uses the discordant-pairs formulation (Newcombe 1998, *Stat Med*).
    Let n10 = number of inputs where A=1, B=0, and n01 = A=0, B=1.
    A Wilson score interval is computed for theta = n10 / (n10 + n01)
    (proportion of discordant pairs where A wins), then transformed to
    the difference scale::

        d_low  = (m / n) * (2 * theta_low  − 1)
        d_high = (m / n) * (2 * theta_high − 1)

    where m = n10 + n01 is the number of discordant pairs and n is the
    total number of paired inputs.

    Returns (0.0, 0.0) when m == 0 (no discordant pairs, perfect agreement).

    Parameters
    ----------
    values_a, values_b : np.ndarray
        1-D arrays of equal length.  Values are thresholded at 0.5 to
        determine binary membership (accommodates float representations).
    alpha : float
        Significance level (1 − confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) − p(B=1).
    """
    n = len(values_a)
    if n <= 0:
        return (0.0, 0.0)
    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)
    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    m = n10 + n01
    if m == 0:
        return (0.0, 0.0)
    theta_low, theta_high = wilson_ci(n10, m, alpha)
    scale = m / n
    return (
        float(scale * (2.0 * theta_low - 1.0)),
        float(scale * (2.0 * theta_high - 1.0)),
    )


def resolve_resampling_method(
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto"],
    sample_size: int,
    *,
    bca_min_n: int = 15,
    bca_max_n: int = 200,
) -> Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"]:
    """Resolve ``method='auto'`` to a concrete bootstrap method.

    ``method='auto'`` always resolves to ``'smooth_bootstrap'``.
    ``sample_size`` and BCa threshold arguments are retained for API
    compatibility.
    ``'bayes_bootstrap'`` and ``'smooth_bootstrap'`` are passed through unchanged.
    """
    _ = (sample_size, bca_min_n, bca_max_n)
    if method == "auto":
        return "smooth_bootstrap"
    return method  # type: ignore[return-value]


def bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Generate bootstrap replicates of the sample statistic for 1-D values.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    m = len(values)
    boot_stats = np.empty(n_bootstrap)
    if statistic == "median":
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_stats[b] = np.median(values[idx])
    else:
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_stats[b] = np.mean(values[idx])
    return boot_stats


def bayes_bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bayesian bootstrap replicates for 1-D values.

    Implements the Bayesian bootstrap (Rubin 1981) as used by Banks (1988)
    "Histospline smoothing the Bayesian bootstrap."  Rather than drawing
    integer-valued multinomial counts (as in the standard bootstrap), each
    replicate draws continuous Dirichlet(1,...,1) weights via normalised
    Exp(1) variates.  This gives smoother coverage—especially at small
    sample sizes—because it explores the full simplex of weight assignments
    rather than just the lattice of integer multiples of 1/n.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    n_bootstrap : int
        Number of Bayesian bootstrap replicates to draw.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.  For ``'mean'``, replicates are
        Dirichlet-weighted means; for ``'median'``, weighted medians.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    n = len(values)
    # Draw (n_bootstrap, n) Exp(1) variates; normalise rows → Dirichlet(1,...,1).
    exp_mat = rng.exponential(1.0, size=(n_bootstrap, n))          # (B, n)
    weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)         # (B, n)

    if statistic == "mean":
        return weights @ values                                     # (B,)

    row_values = np.broadcast_to(values, (n_bootstrap, n))
    return _weighted_medians_rows(row_values, weights)


def bayes_bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bayesian nested bootstrap replicates of paired cell-mean differences.

    Outer level: Dirichlet(1,...,1_M) weights over the M inputs.
    Inner level: standard uniform resample of R runs within each input.

    Using Dirichlet outer weights (rather than multinomial resampling) gives
    smoother bootstrap distributions for small M—the primary motivation for
    Bayesian bootstrap over the standard nested bootstrap.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)``.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    M, R = scores_a.shape

    # Inner resample: which R runs for each (bootstrap, input) pair.
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))         # (B, M, R)
    # Gather inner-resampled runs from all M original inputs.
    diffs = _nested_cell_mean_diffs(scores_a, scores_b, run_idx)   # (B, M)

    # Outer Dirichlet weights for the M inputs.
    exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))          # (B, M)
    outer_weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)   # (B, M)

    if statistic == "mean":
        return (outer_weights * diffs).sum(axis=1)                 # (B,)

    return _weighted_medians_rows(diffs, outer_weights)


def bayes_bootstrap_resample_cell_means_once(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One Bayesian bootstrap nested resample of per-input cell means.

    Inner level resamples R runs uniformly; outer level returns Dirichlet
    weights for the M inputs (rather than resampling them with replacement).

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M, R)``.
    rng : np.random.Generator

    Returns
    -------
    cell_means : np.ndarray
        Shape ``(N, M)`` — inner-resampled cell means for all M inputs.
    outer_weights : np.ndarray
        Shape ``(M,)`` — Dirichlet(1,...,1) weights summing to 1.
    """
    N, M, R = scores.shape
    run_idx = rng.integers(0, R, size=(M, R))                      # (M, R)
    cell_means = _inner_resample_cell_means(scores, run_idx)       # (N, M)

    exp_samp = rng.exponential(1.0, size=M)
    outer_weights = exp_samp / exp_samp.sum()                      # (M,)
    return cell_means, outer_weights


def smooth_bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Smoothed bootstrap replicates for 1-D values using Gaussian KDE.

    Each replicate resamples n observations with replacement from *values*
    and adds i.i.d. Gaussian noise with standard deviation equal to the KDE
    bandwidth (Scott's rule via ``scipy.stats.gaussian_kde``).  This smooths
    the discrete empirical distribution, which can improve coverage for
    continuous data—especially at small sample sizes.

    Falls back to the plain percentile bootstrap if ``std(values) == 0``
    or ``n < 2`` (KDE is degenerate).

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    n_bootstrap : int
        Number of smoothed bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    from scipy.stats import gaussian_kde

    n = len(values)
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or not np.isfinite(std_val) or std_val <= 0.0:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_means_1d",
            f"n={n}, sample std={std_val:.6g}",
        )
        return bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)

    try:
        h = float(gaussian_kde(values).factor * std_val)
    except np.linalg.LinAlgError as exc:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_means_1d",
            f"KDE failed with {exc.__class__.__name__}: {exc}",
        )
        return bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    noise = rng.normal(0.0, h, size=(n_bootstrap, n))
    samples = values[idx] + noise          # (B, n)
    if statistic == "median":
        return np.median(samples, axis=1)  # (B,)
    return samples.mean(axis=1)            # (B,)


def smooth_bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Smoothed nested bootstrap replicates of paired cell-mean differences.

    KDE bandwidth is estimated from the M per-input cell-mean differences.
    The outer level resamples M inputs with replacement; the inner level
    resamples R runs; Gaussian noise with std = KDE bandwidth is then added
    to each resampled cell-mean difference.

    Falls back to ``bootstrap_diffs_nested`` if ``std(cell_diffs) == 0``
    or ``M < 2``.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)``.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    from scipy.stats import gaussian_kde

    M, R = scores_a.shape
    cell_diffs = scores_a.mean(axis=1) - scores_b.mean(axis=1)   # (M,)
    std_val = float(np.std(cell_diffs, ddof=1)) if M > 1 else 0.0
    if M < 2 or not np.isfinite(std_val) or std_val <= 0.0:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_diffs_nested",
            f"M={M}, std(cell_diffs)={std_val:.6g}",
        )
        return bootstrap_diffs_nested(scores_a, scores_b, n_bootstrap, rng, statistic=statistic)

    try:
        h = float(gaussian_kde(cell_diffs).factor * std_val)
    except np.linalg.LinAlgError as exc:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_diffs_nested",
            f"KDE failed with {exc.__class__.__name__}: {exc}",
        )
        return bootstrap_diffs_nested(scores_a, scores_b, n_bootstrap, rng, statistic=statistic)

    input_idx = rng.integers(0, M, size=(n_bootstrap, M))         # (B, M)
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))        # (B, M, R)
    diffs = _nested_cell_mean_diffs(scores_a, scores_b, run_idx, input_idx)   # (B, M)
    diffs += rng.normal(0.0, h, size=(n_bootstrap, M))
    return _reduce_rows(diffs, statistic)


def smooth_bootstrap_resample_cell_means_once(
    scores: np.ndarray,
    bandwidths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One smoothed nested resample of per-input cell means.

    Inner level resamples R runs uniformly; outer level resamples M inputs
    with replacement. Gaussian noise with std = ``bandwidths[i]`` is then
    added to each resampled cell mean for template *i*.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M, R)``.
    bandwidths : np.ndarray
        Shape ``(N,)`` — per-template KDE bandwidths.  Zero entries skip
        smoothing for that template (degenerate case).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(N, M)`` — smoothed resampled cell means.
    """
    N, M, R = scores.shape
    input_idx = rng.integers(0, M, size=M)      # (M,)
    run_idx = rng.integers(0, R, size=(M, R))   # (M, R)

    cell_means = _inner_resample_cell_means(scores, run_idx, input_idx)  # (N, M)

    for i in range(N):
        if bandwidths[i] > 0.0:
            cell_means[i] += rng.normal(0.0, bandwidths[i], size=M)
    return cell_means


def bootstrap_ci_1d(
    values: np.ndarray,
    observed_stat: float,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Bootstrap, BCa, Bayesian bootstrap, or smoothed bootstrap CI for a 1-D array.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    if method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
        return _percentile_interval(boot_stats, alpha)
    if method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
        return _percentile_interval(boot_stats, alpha)
    boot_stats = bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
    if method == "bca":
        return bca_interval_1d(values, observed_stat, boot_stats, alpha, statistic=statistic)
    return _percentile_interval(boot_stats, alpha)


def bca_interval_1d(
    values: np.ndarray,
    observed_stat: float,
    boot_stats: np.ndarray,
    alpha: float,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Compute BCa confidence interval for a statistic of 1-D values.

    The jackknife acceleration estimate uses *statistic* for the
    leave-one-out estimates, matching the bootstrap statistic being corrected.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    b = len(boot_stats)
    less_count = np.sum(boot_stats < observed_stat)
    prop_less = (less_count + 0.5) / (b + 1)
    z0 = stats.norm.ppf(prop_less)

    m = len(values)
    jackknife_stats = np.empty(m)
    for i in range(m):
        jackknife_stats[i] = _stat(np.delete(values, i), statistic)
    # The acceleration uses the mean of jackknife estimates (standard BCa formula).
    jack_mean = np.mean(jackknife_stats)
    d = jack_mean - jackknife_stats
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    accel = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    def adjusted_prob(z_alpha: float) -> float:
        denom_term = 1 - accel * (z0 + z_alpha)
        if denom_term == 0:
            return 0.5
        z_adj = z0 + (z0 + z_alpha) / denom_term
        p = stats.norm.cdf(z_adj)
        return float(np.clip(p, 0.0, 1.0))

    p_low = adjusted_prob(z_alpha_low)
    p_high = adjusted_prob(z_alpha_high)

    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))
    return ci_low, ci_high


def bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bootstrap replicates of ``statistic(cell_mean_a − cell_mean_b)`` via
    two-level (nested) resampling over inputs then runs.

    Both inputs must share the same shape ``(M, R)`` where M is the number
    of benchmark inputs and R is the number of repeated runs per input.

    On each bootstrap iteration the outer level resamples M inputs with
    replacement; the inner level independently resamples R runs for each
    selected input.  This propagates both input-sampling uncertainty and
    within-cell seed variance into the resulting distribution.

    The cell-level aggregation over R runs always uses the mean (collapsing
    repeated runs to a stable cell estimate).  The *statistic* parameter
    controls the across-inputs aggregation: ``'mean'`` or ``'median'``.

    The implementation is fully vectorised across bootstrap iterations.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)`` for the two templates.
    n_bootstrap : int
        Number of bootstrap replicates to generate.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        Across-inputs aggregator: ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.  Each entry is the statistic of paired
        cell-mean differences for one bootstrap resample.
    """
    M, R = scores_a.shape

    # Outer resample: which M inputs to use for each bootstrap iteration.
    # Shape (n_bootstrap, M).
    input_idx = rng.integers(0, M, size=(n_bootstrap, M))

    # Inner resample: which R runs to use for each (bootstrap, input) pair.
    # Shape (n_bootstrap, M, R).
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))

    diffs = _nested_cell_mean_diffs(scores_a, scores_b, run_idx, input_idx)  # (B, M)
    return _reduce_rows(diffs, statistic)                                  # (B,)


def nested_resample_cell_means_once(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One nested resample of per-input cell means for ``scores`` of shape ``(N, M, R)``.

    Outer level resamples inputs; inner level resamples runs within each
    selected input. Returns resampled cell means of shape ``(N, M)``.
    """
    N, M, R = scores.shape
    input_idx = rng.integers(0, M, size=M)      # (M,)
    run_idx = rng.integers(0, R, size=(M, R))   # (M, R)

    return _inner_resample_cell_means(scores, run_idx, input_idx)  # (N, M)
