"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats


def _stat(values: np.ndarray, statistic: Literal["mean", "median"]) -> float:
    """Apply *statistic* to a 1-D array and return a Python float."""
    if statistic == "median":
        return float(np.median(values))
    return float(np.mean(values))


def resolve_resampling_method(
    method: Literal["bootstrap", "bca", "auto"],
    sample_size: int,
    *,
    bca_min_n: int = 15,
    bca_max_n: int = 200,
) -> Literal["bootstrap", "bca"]:
    """Resolve ``method='auto'`` to a concrete bootstrap method.

    Uses BCa for moderate sample sizes where acceleration/bias correction is
    typically beneficial, and percentile bootstrap otherwise.
    """
    if method == "auto":
        return "bca" if bca_min_n <= sample_size <= bca_max_n else "bootstrap"
    return method


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


def bootstrap_ci_1d(
    values: np.ndarray,
    observed_stat: float,
    method: Literal["bootstrap", "bca"],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Bootstrap or BCa CI for the chosen statistic of a 1-D array.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    boot_stats = bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
    if method == "bca":
        return bca_interval_1d(values, observed_stat, boot_stats, alpha, statistic=statistic)
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


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

    # Gather inputs: scores_a[input_idx] broadcasts over the (B, M) index
    # into axis 0 of scores_a (shape M, R), giving shape (B, M, R).
    sel_a = scores_a[input_idx]   # (B, M, R)
    sel_b = scores_b[input_idx]   # (B, M, R)

    # Gather runs: for each (b, k), pick the R run indices in run_idx[b, k].
    # sel_a[b, k, run_idx[b, k, r]] for all b, k, r.
    b_range = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]  # (B, 1, 1)
    m_range = np.arange(M)[np.newaxis, :, np.newaxis]            # (1, M, 1)
    resampled_a = sel_a[b_range, m_range, run_idx]               # (B, M, R)
    resampled_b = sel_b[b_range, m_range, run_idx]               # (B, M, R)

    # Cell means (always mean over R runs) and per-input paired differences.
    diffs = resampled_a.mean(axis=2) - resampled_b.mean(axis=2)  # (B, M)
    if statistic == "median":
        return np.median(diffs, axis=1)                          # (B,)
    return diffs.mean(axis=1)                                    # (B,)


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

    sel = scores[:, input_idx, :]               # (N, M, R)
    m_range = np.arange(M)[:, np.newaxis]       # (M, 1)
    resampled = sel[:, m_range, run_idx]        # (N, M, R)
    return resampled.mean(axis=2)               # (N, M)
