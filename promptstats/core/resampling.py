"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap replicates of the sample mean for 1-D values."""
    m = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(m, size=m, replace=True)
        boot_means[b] = np.mean(values[idx])
    return boot_means


def bca_interval_1d(
    values: np.ndarray,
    observed_mean: float,
    boot_means: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Compute BCa confidence interval for the mean of 1-D values."""
    b = len(boot_means)
    less_count = np.sum(boot_means < observed_mean)
    prop_less = (less_count + 0.5) / (b + 1)
    z0 = stats.norm.ppf(prop_less)

    m = len(values)
    jackknife_means = np.empty(m)
    for i in range(m):
        jackknife_means[i] = np.mean(np.delete(values, i))
    jack_mean = np.mean(jackknife_means)
    d = jack_mean - jackknife_means
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

    ci_low = float(np.percentile(boot_means, 100 * p_low))
    ci_high = float(np.percentile(boot_means, 100 * p_high))
    return ci_low, ci_high


def bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap replicates of ``mean(cell_mean_a − cell_mean_b)`` via
    two-level (nested) resampling over inputs then runs.

    Both inputs must share the same shape ``(M, R)`` where M is the number
    of benchmark inputs and R is the number of repeated runs per input.

    On each bootstrap iteration the outer level resamples M inputs with
    replacement; the inner level independently resamples R runs for each
    selected input.  This propagates both input-sampling uncertainty and
    within-cell seed variance into the resulting distribution.

    The implementation is fully vectorised across bootstrap iterations.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)`` for the two templates.
    n_bootstrap : int
        Number of bootstrap replicates to generate.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.  Each entry is the mean paired cell-mean
        difference for one bootstrap resample.
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

    # Cell means and per-input paired differences.
    diffs = resampled_a.mean(axis=2) - resampled_b.mean(axis=2)  # (B, M)
    return diffs.mean(axis=1)                                     # (B,)
