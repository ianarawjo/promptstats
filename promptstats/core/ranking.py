"""Bootstrap ranking analysis for prompt templates.

Provides rank distributions and mean advantage calculations that respect
the paired structure of benchmark data (same inputs across all templates).

When the score array includes a runs axis (R >= 3), all bootstrap functions
use a two-level (nested) resample: inputs are resampled in the outer level,
and runs within each selected input are resampled in the inner level.  This
correctly propagates seed variance into rank and CI estimates instead of
treating per-run cell means as fixed observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .resampling import (
    _stat,
    _weighted_median,
    bca_interval_1d,
    bayes_bootstrap_means_1d,
    bayes_bootstrap_resample_cell_means_once,
    smooth_bootstrap_means_1d,
    smooth_bootstrap_resample_cell_means_once,
    bootstrap_means_1d,
    nested_resample_cell_means_once,
    resolve_resampling_method,
    wilson_ci_1d,
    newcombe_paired_ci,
)


def _accumulate_tie_aware_rank_mass(rank_counts: np.ndarray, agg: np.ndarray) -> None:
    """Accumulate one bootstrap draw of rank mass with fair tie handling.

    For each tie block of size ``t`` occupying ranks ``[r, r+t-1]``, each tied
    template receives ``1/t`` mass at each occupied rank. This removes the
    deterministic first-index tie bias introduced by ``np.argsort``.
    """
    order = np.argsort(-agg, kind="mergesort")
    sorted_scores = agg[order]

    start = 0
    n_templates = len(order)
    while start < n_templates:
        end = start + 1
        while end < n_templates and sorted_scores[end] == sorted_scores[start]:
            end += 1

        tie_indices = order[start:end]
        tie_size = end - start
        share = 1.0 / tie_size
        rank_counts[tie_indices, start:end] += share
        start = end


@dataclass
class RankDistribution:
    """Bootstrap distribution over template rankings.

    Attributes
    ----------
    labels : list[str]
        Template labels.
    rank_probs : np.ndarray
        Shape (N_templates, N_templates). Entry [i, r] is the probability
        that template i achieves rank r (0-indexed, 0 = best).
    expected_ranks : np.ndarray
        Shape (N_templates,). Expected rank for each template (1-indexed).
    p_best : np.ndarray
        Shape (N_templates,). Probability each template is ranked first.
    n_bootstrap : int
        Number of bootstrap iterations used.
    """

    labels: list[str]
    rank_probs: np.ndarray
    expected_ranks: np.ndarray
    p_best: np.ndarray
    n_bootstrap: int


@dataclass
class PointAdvantageResult:
    """Point advantage of each template over a reference, with uncertainty.

    This is the core data structure for the advantage plot. It separates
    two kinds of uncertainty:

    - **Epistemic (CI on the point estimate)**: Would shrink with more
      benchmark inputs.  Captured by bootstrap_ci_low/high.
    - **Intrinsic (score spread)**: A property of the template, won't shrink
      with more data. Captured by spread_low/high (percentiles of per-input
      advantages).

    Attributes
    ----------
    labels : list[str]
        Template labels.
    point_advantages : np.ndarray
        Shape (N,). Point-estimate advantage over reference for each template
        (mean or median depending on ``statistic``).
    bootstrap_ci_low : np.ndarray
        Shape (N,). Lower bound of bootstrap CI on the point advantage.
    bootstrap_ci_high : np.ndarray
        Shape (N,). Upper bound of bootstrap CI on the point advantage.
    spread_low : np.ndarray
        Shape (N,). Lower percentile of per-input advantage distribution.
    spread_high : np.ndarray
        Shape (N,). Upper percentile of per-input advantage distribution.
    reference : str
        Description of the reference used (e.g., 'grand_mean' or a template label).
        Note: The grand reference is always the per-input mean across templates 
        regardless of statistic; see `bootstrap_point_advantage` for rationale.
    per_input_advantages : np.ndarray
        Shape (N, M). Raw per-input cell-mean advantages for each template.
    n_bootstrap : int
        Number of bootstrap iterations used.
    spread_percentiles : tuple[float, float]
        The percentiles used for spread_low/high (e.g., (10, 90)).
    statistic : str
        The central-tendency statistic used: ``'mean'`` or ``'median'``.
    """

    labels: list[str]
    point_advantages: np.ndarray
    bootstrap_ci_low: np.ndarray
    bootstrap_ci_high: np.ndarray
    spread_low: np.ndarray
    spread_high: np.ndarray
    reference: str
    per_input_advantages: np.ndarray
    n_bootstrap: int
    spread_percentiles: tuple[float, float]
    statistic: str = "mean"


def bootstrap_ranks(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Compute bootstrap distribution over template rankings.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` a two-level nested bootstrap is used.
        When ``R < 3`` (or 2-D input) the standard single-level resample
        is used.
    labels : list[str]
        Template labels.
    n_bootstrap : int
        Number of bootstrap iterations.
    method : str
        Resampling method for API consistency: ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, or ``'auto'``.  Rank
        distributions use multinomial (``'bootstrap'``/``'bca'``),
        Dirichlet (``'bayes_bootstrap'``), or smoothed KDE
        (``'smooth_bootstrap'``) outer weights. ``'auto'`` resolves to
        ``'smooth_bootstrap'``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Statistic used to aggregate scores across inputs when determining
        template rankings per bootstrap resample: ``'median'`` (default)
        or ``'mean'``.

    Returns
    -------
    RankDistribution
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto"}:
        raise ValueError(f"Unknown method: {method}")

    m_inputs = scores.shape[1]
    resolved_method = resolve_resampling_method(method, m_inputs)

    # ------------------------------------------------------------------ #
    # Seeded path (R >= 3)                                                #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3 and scores.shape[2] >= 3:
        if resolved_method == "bayes_bootstrap":
            return _bayes_bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "smooth_bootstrap":
            return _smooth_bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)
        return _bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)

    # ------------------------------------------------------------------ #
    # Standard path (2-D or R < 3)                                        #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        scores = scores.mean(axis=2)  # collapse small run axis

    n_templates, m_inputs = scores.shape
    rank_counts = np.zeros((n_templates, n_templates), dtype=float)

    if resolved_method == "bayes_bootstrap":
        # Dirichlet-weighted aggregation per template instead of
        # multinomial input resampling.
        exp_mat = rng.exponential(1.0, size=(n_bootstrap, m_inputs))   # (B, M)
        weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)         # (B, M)
        if statistic == "median":
            for b in range(n_bootstrap):
                agg = np.array([_weighted_median(scores[t], weights[b]) for t in range(n_templates)])
                _accumulate_tie_aware_rank_mass(rank_counts, agg)
        else:
            for b in range(n_bootstrap):
                agg = scores @ weights[b]                               # (N,)
                _accumulate_tie_aware_rank_mass(rank_counts, agg)
    elif resolved_method == "smooth_bootstrap":
        # KDE-smoothed resample: resample inputs with replacement + add Gaussian noise.
        from scipy.stats import gaussian_kde as _kde
        # Compute per-template bandwidths from the M cell means.
        bws = np.zeros(n_templates)
        for t in range(n_templates):
            std_t = float(np.std(scores[t], ddof=1)) if m_inputs > 1 else 0.0
            if std_t > 0.0 and m_inputs >= 2:
                bws[t] = float(_kde(scores[t]).factor * std_t)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, m_inputs, size=m_inputs)
            samples = scores[:, idx].copy()                            # (N, M)
            for t in range(n_templates):
                if bws[t] > 0.0:
                    samples[t] += rng.normal(0.0, bws[t], size=m_inputs)
            if statistic == "median":
                agg = np.median(samples, axis=1)
            else:
                agg = samples.mean(axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)
    elif statistic == "median":
        for _ in range(n_bootstrap):
            idx = rng.choice(m_inputs, size=m_inputs, replace=True)
            agg = np.median(scores[:, idx], axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)
    else:
        for _ in range(n_bootstrap):
            idx = rng.choice(m_inputs, size=m_inputs, replace=True)
            agg = scores[:, idx].mean(axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, n_templates + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Rank distribution via nested bootstrap for ``scores`` of shape ``(N, M, R)``."""
    N, _, _ = scores.shape
    rank_counts = np.zeros((N, N), dtype=float)

    for _ in range(n_bootstrap):
        boot_cell_means = nested_resample_cell_means_once(scores, rng)  # (N, M)
        if statistic == "median":
            agg = np.median(boot_cell_means, axis=1)                   # (N,)
        else:
            agg = boot_cell_means.mean(axis=1)                         # (N,)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _bayes_bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Bayesian bootstrap rank distribution via nested bootstrap for ``scores`` of shape ``(N, M, R)``.

    Inner level resamples R runs uniformly; outer level uses Dirichlet(1,...,1_M)
    weights instead of multinomial input resampling.
    """
    N, M, _ = scores.shape
    rank_counts = np.zeros((N, N), dtype=float)

    for _ in range(n_bootstrap):
        cell_means, w = bayes_bootstrap_resample_cell_means_once(scores, rng)  # (N, M), (M,)
        if statistic == "median":
            agg = np.array([_weighted_median(cell_means[t], w) for t in range(N)])
        else:
            agg = cell_means @ w                                        # (N,)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _smooth_bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Smoothed bootstrap rank distribution for ``scores`` of shape ``(N, M, R)``.

    Inner level resamples R runs uniformly; outer level resamples M inputs
    with replacement; Gaussian KDE noise is added to each resampled cell mean.
    """
    N, M, _ = scores.shape
    cell_means = scores.mean(axis=2)   # (N, M) — original cell means for bandwidth estimation

    from scipy.stats import gaussian_kde as _kde
    bws = np.zeros(N)
    for t in range(N):
        std_t = float(np.std(cell_means[t], ddof=1)) if M > 1 else 0.0
        if std_t > 0.0 and M >= 2:
            bws[t] = float(_kde(cell_means[t]).factor * std_t)

    rank_counts = np.zeros((N, N), dtype=float)
    for _ in range(n_bootstrap):
        boot_cell_means = smooth_bootstrap_resample_cell_means_once(scores, bws, rng)  # (N, M)
        if statistic == "median":
            agg = np.median(boot_cell_means, axis=1)
        else:
            agg = boot_cell_means.mean(axis=1)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_point_advantage(
    scores: np.ndarray,
    labels: list[str],
    reference: str = "grand_mean",
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    spread_percentiles: tuple[float, float] = (10, 90),
    rng: Optional[np.random.Generator] = None,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "wilson"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
) -> PointAdvantageResult:
    """Compute point advantage over a reference with dual uncertainty bands.

    For each template, computes its per-input advantage over a reference
    (either the grand median/mean across all templates, or a specific baseline
    template). Point estimates use per-input cell means aggregated by
    *statistic* across inputs.

    When ``scores`` has shape ``(N, M, R)`` with ``R >= 3``, the bootstrap
    CI is computed via two-level nested resampling so that seed variance
    inflates the CI correctly.  The BCa jackknife uses the cell-mean
    advantages (outer level), which is the correct primary sampling unit.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(N, M)`` or ``(N, M, R)``.
    labels : list[str]
        Template labels.
    reference : str
        ``'grand_mean'`` (default) or a specific template label.  When
        ``statistic='median'`` the grand reference is still the mean.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (default 0.95).
    method : str
        CI method: ``'auto'``, ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, or ``'smooth_bootstrap'``. ``'auto'``
        resolves to ``'smooth_bootstrap'``.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band (default ``(10, 90)``).
    rng : np.random.Generator, optional
        Random number generator.
    statistic : str
        Point-estimate and bootstrap statistic: ``'mean'`` (default) or
        ``'mean'``.

    Returns
    -------
    PointAdvantageResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Wilson path for binary (0/1) data                                   #
    # ------------------------------------------------------------------ #
    if method == "wilson":
        # When R >= 3 the per-run cell means are no longer binary values;
        # fall back to smooth bootstrap for the seeded nested path.
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _smooth_bootstrap_point_advantage_seeded(
                scores, labels,
                reference=reference, n_bootstrap=n_bootstrap,
                ci=ci, spread_percentiles=spread_percentiles,
                rng=rng, statistic=statistic,
            )
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        return _wilson_point_advantage(
            flat, labels,
            reference=reference,
            alpha=1.0 - ci,
            spread_percentiles=spread_percentiles,
        )

    # ------------------------------------------------------------------ #
    # Seeded path (R >= 3)                                                #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3 and scores.shape[2] >= 3:
        resolved_method = resolve_resampling_method(method, scores.shape[1])
        if resolved_method == "bayes_bootstrap":
            return _bayes_bootstrap_point_advantage_seeded(
                scores, labels,
                reference=reference,
                n_bootstrap=n_bootstrap,
                ci=ci,
                spread_percentiles=spread_percentiles,
                rng=rng,
                statistic=statistic,
            )
        if resolved_method == "smooth_bootstrap":
            return _smooth_bootstrap_point_advantage_seeded(
                scores, labels,
                reference=reference,
                n_bootstrap=n_bootstrap,
                ci=ci,
                spread_percentiles=spread_percentiles,
                rng=rng,
                statistic=statistic,
            )
        return _bootstrap_point_advantage_seeded(
            scores, labels,
            reference=reference,
            n_bootstrap=n_bootstrap,
            ci=ci,
            spread_percentiles=spread_percentiles,
            rng=rng,
            method=resolved_method,
            statistic=statistic,
        )

    # ------------------------------------------------------------------ #
    # Standard path (2-D or R < 3)                                        #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        scores = scores.mean(axis=2)

    n_templates, m_inputs = scores.shape
    alpha = 1 - ci
    resolved_method = resolve_resampling_method(method, m_inputs)

    if reference == "grand_mean":
        # Grand reference is always the per-input mean across templates.
        # Using median here (axis=0 over N templates) would cause the middle
        # template to have identically-zero advantages with an odd template
        # count, which is degenerate.  The mean is a stable, non-degenerate
        # reference regardless of the across-inputs statistic.
        ref_scores = scores.mean(axis=0)
        ref_label = "grand_mean"
    else:
        ref_idx = labels.index(reference)
        ref_scores = scores[ref_idx]
        ref_label = reference

    advantages = scores - ref_scores[np.newaxis, :]           # (N, M)
    # Point advantage: aggregate across inputs with chosen statistic.
    if statistic == "median":
        point_adv = np.median(advantages, axis=1)             # (N,)
    else:
        point_adv = advantages.mean(axis=1)                   # (N,)
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    ci_low = np.empty(n_templates)
    ci_high = np.empty(n_templates)

    for i in range(n_templates):
        vals = advantages[i]
        if resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_means_1d(
                vals, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )
            ci_low[i] = np.percentile(boot_stats, 100 * alpha / 2)
            ci_high[i] = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        elif resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_means_1d(
                vals, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )
            ci_low[i] = np.percentile(boot_stats, 100 * alpha / 2)
            ci_high[i] = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        else:
            boot_stats = bootstrap_means_1d(
                vals, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )
            if resolved_method == "bootstrap":
                ci_low[i] = np.percentile(boot_stats, 100 * alpha / 2)
                ci_high[i] = np.percentile(boot_stats, 100 * (1 - alpha / 2))
            elif resolved_method == "bca":
                low, high = bca_interval_1d(
                    vals, float(point_adv[i]), boot_stats, alpha, statistic=statistic,
                )
                ci_low[i] = low
                ci_high[i] = high
            else:
                raise ValueError(f"Unknown method: {method}")

    return PointAdvantageResult(
        labels=labels,
        point_advantages=point_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles,
        statistic=statistic,
    )


def _bootstrap_point_advantage_seeded(
    scores: np.ndarray,
    labels: list[str],
    *,
    reference: str,
    n_bootstrap: int,
    ci: float,
    spread_percentiles: tuple[float, float],
    rng: np.random.Generator,
    method: Literal["bootstrap", "bca"],
    statistic: Literal["mean", "median"],
) -> PointAdvantageResult:
    """Point advantage with nested bootstrap for ``scores`` of shape ``(N, M, R)``."""
    N, M, _ = scores.shape
    alpha = 1 - ci

    # ---- Point estimates from cell means --------------------------------
    # Within-cell aggregation always uses mean (collapsing R runs per cell).
    cell_means = scores.mean(axis=2)   # (N, M)

    if reference == "grand_mean":
        # Grand reference is always the per-input mean across templates.
        # See bootstrap_point_advantage for the rationale.
        ref_scores = cell_means.mean(axis=0)   # (M,)
        ref_label = "grand_mean"
        ref_idx = None
    else:
        ref_idx = labels.index(reference)
        ref_scores = cell_means[ref_idx]                 # (M,)
        ref_label = reference

    advantages = cell_means - ref_scores[np.newaxis, :]        # (N, M)
    if statistic == "median":
        point_adv = np.median(advantages, axis=1)              # (N,)
    else:
        point_adv = advantages.mean(axis=1)                    # (N,)
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    # ---- Nested bootstrap replicates of point advantages ----------------
    # Shape (n_bootstrap, N): for each iteration, the point advantage of
    # every template after resampling inputs and runs.
    boot_point_advs = np.empty((n_bootstrap, N))

    for b in range(n_bootstrap):
        boot_cell_means = nested_resample_cell_means_once(scores, rng)  # (N, M)

        if ref_idx is None:
            # Grand reference is always per-input mean (see bootstrap_point_advantage).
            boot_ref = boot_cell_means.mean(axis=0)             # (M,)
        else:
            boot_ref = boot_cell_means[ref_idx]                 # (M,)

        boot_adv = boot_cell_means - boot_ref[np.newaxis, :]    # (N, M)
        if statistic == "median":
            boot_point_advs[b] = np.median(boot_adv, axis=1)   # (N,)
        else:
            boot_point_advs[b] = boot_adv.mean(axis=1)         # (N,)

    # ---- CIs per template -----------------------------------------------
    ci_low = np.empty(N)
    ci_high = np.empty(N)

    for i in range(N):
        boot_i = boot_point_advs[:, i]      # (B,)
        if method == "bootstrap":
            ci_low[i] = np.percentile(boot_i, 100 * alpha / 2)
            ci_high[i] = np.percentile(boot_i, 100 * (1 - alpha / 2))
        elif method == "bca":
            # Jackknife over inputs (outer sampling unit) using cell-point advantages.
            ci_low[i], ci_high[i] = bca_interval_1d(
                advantages[i], float(point_adv[i]), boot_i, alpha,
                statistic=statistic,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return PointAdvantageResult(
        labels=labels,
        point_advantages=point_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles,
        statistic=statistic,
    )


def _bayes_bootstrap_point_advantage_seeded(
    scores: np.ndarray,
    labels: list[str],
    *,
    reference: str,
    n_bootstrap: int,
    ci: float,
    spread_percentiles: tuple[float, float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
) -> PointAdvantageResult:
    """Bayesian bootstrap point advantage with nested bootstrap for ``scores`` of shape ``(N, M, R)``.

    Uses Dirichlet outer weights over the M inputs; inner level resamples
    R runs uniformly for each input.
    """
    N, M, _ = scores.shape
    alpha = 1 - ci

    cell_means = scores.mean(axis=2)   # (N, M) point-estimate cell means

    if reference == "grand_mean":
        ref_scores = cell_means.mean(axis=0)   # (M,)
        ref_label = "grand_mean"
        ref_idx = None
    else:
        ref_idx = labels.index(reference)
        ref_scores = cell_means[ref_idx]
        ref_label = reference

    advantages = cell_means - ref_scores[np.newaxis, :]        # (N, M)
    if statistic == "median":
        point_adv = np.median(advantages, axis=1)
    else:
        point_adv = advantages.mean(axis=1)
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    boot_point_advs = np.empty((n_bootstrap, N))

    for b in range(n_bootstrap):
        boot_cell_means, w = bayes_bootstrap_resample_cell_means_once(scores, rng)  # (N, M), (M,)

        if ref_idx is None:
            boot_ref = boot_cell_means.mean(axis=0)         # (M,) grand mean ref
        else:
            boot_ref = boot_cell_means[ref_idx]             # (M,)

        boot_adv = boot_cell_means - boot_ref[np.newaxis, :]  # (N, M)
        if statistic == "median":
            boot_point_advs[b] = np.array([_weighted_median(boot_adv[t], w) for t in range(N)])
        else:
            boot_point_advs[b] = boot_adv @ w              # (N,) Dirichlet-weighted mean

    ci_low = np.empty(N)
    ci_high = np.empty(N)
    for i in range(N):
        boot_i = boot_point_advs[:, i]
        ci_low[i] = np.percentile(boot_i, 100 * alpha / 2)
        ci_high[i] = np.percentile(boot_i, 100 * (1 - alpha / 2))

    return PointAdvantageResult(
        labels=labels,
        point_advantages=point_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles,
        statistic=statistic,
    )


def _smooth_bootstrap_point_advantage_seeded(
    scores: np.ndarray,
    labels: list[str],
    *,
    reference: str,
    n_bootstrap: int,
    ci: float,
    spread_percentiles: tuple[float, float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
) -> PointAdvantageResult:
    """Smoothed bootstrap point advantage for ``scores`` of shape ``(N, M, R)``.

    Inner level resamples R runs uniformly; outer level resamples M inputs
    with replacement; Gaussian KDE noise is added to each resampled cell mean.
    """
    from scipy.stats import gaussian_kde as _kde

    N, M, _ = scores.shape
    alpha = 1 - ci

    cell_means = scores.mean(axis=2)   # (N, M)

    if reference == "grand_mean":
        ref_scores = cell_means.mean(axis=0)   # (M,)
        ref_label = "grand_mean"
        ref_idx = None
    else:
        ref_idx = labels.index(reference)
        ref_scores = cell_means[ref_idx]
        ref_label = reference

    advantages = cell_means - ref_scores[np.newaxis, :]        # (N, M)
    if statistic == "median":
        point_adv = np.median(advantages, axis=1)
    else:
        point_adv = advantages.mean(axis=1)
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    # Per-template KDE bandwidths estimated from original cell means.
    bws = np.zeros(N)
    for t in range(N):
        std_t = float(np.std(cell_means[t], ddof=1)) if M > 1 else 0.0
        if std_t > 0.0 and M >= 2:
            bws[t] = float(_kde(cell_means[t]).factor * std_t)

    boot_point_advs = np.empty((n_bootstrap, N))

    for b in range(n_bootstrap):
        boot_cell_means = smooth_bootstrap_resample_cell_means_once(scores, bws, rng)  # (N, M)

        if ref_idx is None:
            boot_ref = boot_cell_means.mean(axis=0)             # (M,)
        else:
            boot_ref = boot_cell_means[ref_idx]                 # (M,)

        boot_adv = boot_cell_means - boot_ref[np.newaxis, :]    # (N, M)
        if statistic == "median":
            boot_point_advs[b] = np.median(boot_adv, axis=1)
        else:
            boot_point_advs[b] = boot_adv.mean(axis=1)

    ci_low = np.empty(N)
    ci_high = np.empty(N)
    for i in range(N):
        boot_i = boot_point_advs[:, i]
        ci_low[i] = np.percentile(boot_i, 100 * alpha / 2)
        ci_high[i] = np.percentile(boot_i, 100 * (1 - alpha / 2))

    return PointAdvantageResult(
        labels=labels,
        point_advantages=point_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles,
        statistic=statistic,
    )


def _wilson_point_advantage(
    scores: np.ndarray,
    labels: list[str],
    *,
    reference: str,
    alpha: float,
    spread_percentiles: tuple[float, float],
) -> PointAdvantageResult:
    """Wilson score CIs for per-template binary success rates, shifted by reference.

    For the grand-mean reference, computes a Wilson CI on each template's raw
    success proportion and shifts it by the grand-mean proportion.

    For a specific template reference, uses the Newcombe paired score interval
    between each template and the reference, equivalent to the CI on p_i − p_ref.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M)`` with binary (0/1) per-input scores.
    labels : list[str]
        Template labels.
    reference : str
        ``'grand_mean'`` or a template label.
    alpha : float
        Significance level (1 − confidence level).
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band.

    Returns
    -------
    PointAdvantageResult
        ``n_bootstrap`` is 0 (no bootstrap was used).
    """
    n_templates, m_inputs = scores.shape

    if reference == "grand_mean":
        ref_scores = scores.mean(axis=0)   # (M,) per-input grand mean
        ref_label = "grand_mean"
        ref_idx = None
        grand_mean_p = float(ref_scores.mean())
    else:
        ref_idx = labels.index(reference)
        ref_scores = scores[ref_idx]       # (M,) raw binary scores for reference template
        ref_label = reference

    advantages = scores - ref_scores[np.newaxis, :]   # (N, M)
    point_adv = advantages.mean(axis=1)                # (N,) — always mean for proportions
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    ci_low = np.empty(n_templates)
    ci_high = np.empty(n_templates)

    if reference == "grand_mean":
        for i in range(n_templates):
            w_low, w_high = wilson_ci_1d(scores[i], alpha)
            ci_low[i] = w_low - grand_mean_p
            ci_high[i] = w_high - grand_mean_p
    else:
        for i in range(n_templates):
            if i == ref_idx:
                ci_low[i] = 0.0
                ci_high[i] = 0.0
            else:
                ci_low[i], ci_high[i] = newcombe_paired_ci(scores[i], ref_scores, alpha)

    return PointAdvantageResult(
        labels=labels,
        point_advantages=point_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=0,
        spread_percentiles=spread_percentiles,
        statistic="mean",
    )
