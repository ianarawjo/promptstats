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
    _weighted_median,
    bayes_bootstrap_resample_cell_means_once,
    smooth_bootstrap_resample_cell_means_once,
    nested_resample_cell_means_once,
    resolve_resampling_method,
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


class LazyRankDistribution(RankDistribution):
    """A ``RankDistribution`` whose bootstrap is deferred until the rank
    arrays are actually read.

    ``labels`` and ``n_bootstrap`` answer immediately -- a lot of code
    (``core/summary.py`` especially) reads ``rank_dist.labels`` purely as the
    canonical label list and must not pay for a rank bootstrap to get it.
    Reading ``rank_probs``/``expected_ranks``/``p_best`` runs the bootstrap
    once and caches it, so P(Best)/E[Rank] cost is paid only by callers that
    actually want those numbers -- matching the opt-in story
    ``ResultReport._show_rank_probabilities`` already tells for the *output*.

    The generator state is snapshotted at construction rather than the live
    ``rng`` being held, so a deferred bootstrap draws exactly what an eager
    one would have drawn. Note the parent ``rng`` is NOT advanced when the
    ranks go uncomputed, so downstream draws differ from the old
    always-compute behaviour; the rank distribution itself is unchanged.
    """

    def __init__(self, labels, n_bootstrap, compute, rng=None):
        self.labels = list(labels)
        self.n_bootstrap = n_bootstrap
        self._compute = compute
        self._resolved: Optional[RankDistribution] = None
        self._state = None
        if rng is not None:
            try:
                self._state = (type(rng.bit_generator), rng.bit_generator.state)
            except Exception:
                self._state = None

    def _resolve(self) -> RankDistribution:
        if self._resolved is None:
            rng = None
            if self._state is not None:
                bg_type, state = self._state
                rng = np.random.Generator(bg_type())
                rng.bit_generator.state = state
            self._resolved = self._compute(rng)
        return self._resolved

    @property
    def computed(self) -> bool:
        """True once the bootstrap has actually run."""
        return self._resolved is not None

    @property
    def rank_probs(self) -> np.ndarray:
        return self._resolve().rank_probs

    @property
    def expected_ranks(self) -> np.ndarray:
        return self._resolve().expected_ranks

    @property
    def p_best(self) -> np.ndarray:
        return self._resolve().p_best

    def __repr__(self) -> str:
        if self._resolved is None:
            return (f"LazyRankDistribution(labels={self.labels!r}, "
                    f"n_bootstrap={self.n_bootstrap}, computed=False)")
        return repr(self._resolved)


def bootstrap_ranks(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "bayes_binary", "permutation"] = "auto",
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
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'bootstrap_t'``, or ``'auto'``.  Rank
        distributions use multinomial (``'bootstrap'``/``'bca'``),
        multinomial (``'bootstrap_t'``),
        Dirichlet (``'bayes_bootstrap'``), or smoothed KDE
        (``'smooth_bootstrap'``) outer weights. ``'auto'`` resolves to
        ``'smooth_bootstrap'``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Statistic used to aggregate scores across inputs when determining
        template rankings per bootstrap resample: ``'mean'`` (default)
        or ``'median'``.

    Returns
    -------
    RankDistribution
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "bayes_binary", "permutation"}:
        raise ValueError(f"Unknown method: {method}")

    # Rank distribution does not use a special Bayesian binary model;
    # treat bayes_binary as smooth_bootstrap for ranking purposes.
    # Permutation is a p-value method for pairwise tests; rank distributions
    # still use bootstrap-style resampling.
    if method == "bayes_binary":
        effective_method = "smooth_bootstrap"
    elif method == "permutation":
        effective_method = "bootstrap"
    elif method == "bootstrap_t":
        effective_method = "bootstrap"
    else:
        effective_method = method
    m_inputs = scores.shape[1]
    resolved_method = resolve_resampling_method(effective_method, m_inputs)

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


def ppi_bootstrap_ranks(
    scores_2d: np.ndarray,
    lab_matrix: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> RankDistribution:
    """PPI-aware rank distribution: jointly resamples items across entities.

    ``bootstrap_ranks()`` ranks entities from the raw (uncorrected) LLM
    judge scores, so its ``P(Best)``/``E[Rank]`` output does not reflect a
    PPI alignment correction applied elsewhere (see
    ``evalstats.api._run_alignment_ppi``). This function recomputes the
    rank distribution using the PPI estimator itself: on each bootstrap
    draw it resamples item positions once (shared across all entities, to
    preserve the paired correlation from the benchmark's item-aligned
    design) from the full item pool, resamples item positions once from the
    human-labeled subset, forms each entity's PPI point estimate for that
    draw (``mean(LLM on resampled pool) + rectifier``), and accumulates
    tie-aware rank mass exactly like the base bootstrap.

    Parameters
    ----------
    scores_2d : np.ndarray, shape (n_entities, n_items)
        Item-aligned LLM judge scores (NaN for incomplete-design cells).
    lab_matrix : np.ndarray, shape (n_entities, n_items)
        Item-aligned human labels; NaN where an (entity, item) cell has no
        human label.
    labels : list[str]
        Entity labels, in the same order as the rows of ``scores_2d``.
    n_bootstrap : int
        Number of bootstrap iterations.
    rng : np.random.Generator

    Returns
    -------
    RankDistribution
    """
    n_entities, n_items = scores_2d.shape
    labeled_item_mask = ~np.all(np.isnan(lab_matrix), axis=0)
    labeled_item_positions = np.where(labeled_item_mask)[0]
    n_lab_items = len(labeled_item_positions)
    # unlabeled_item_positions must be DISJOINT from labeled_item_positions
    # -- idx_all below is resampled independently of idx_lab, which is only
    # valid for genuinely disjoint samples (see evalstats.ppi.correct's
    # docstring for why resampling an overlapping "unlab" pool independently
    # of the labeled subset silently miscalibrates the bootstrap).
    unlabeled_item_positions = np.where(~labeled_item_mask)[0]
    n_unlab_items = len(unlabeled_item_positions)
    if n_unlab_items == 0:
        raise ValueError(
            "ppi_bootstrap_ranks: every item is labeled -- PPI has no "
            "unlabeled pool left to extrapolate the correction to. With "
            "100% human labels, rank entities directly on the human scores "
            "instead of PPI."
        )

    # Fallback for entities with zero human labels anywhere: keep them at
    # their uncorrected LLM-only mean on every draw (rectifier == 0).
    fallback_mean = np.nanmean(scores_2d, axis=1)

    rank_counts = np.zeros((n_entities, n_entities), dtype=float)

    with np.errstate(invalid="ignore"):
        for _ in range(n_bootstrap):
            idx_all = unlabeled_item_positions[rng.integers(0, n_unlab_items, n_unlab_items)]
            f_unlab = np.nanmean(scores_2d[:, idx_all], axis=1)

            agg = np.where(np.isnan(f_unlab), fallback_mean, f_unlab)

            if n_lab_items > 0:
                idx_lab = labeled_item_positions[rng.integers(0, n_lab_items, n_lab_items)]
                lab_vals = lab_matrix[:, idx_lab]
                llm_at_lab = scores_2d[:, idx_lab]
                valid = ~np.isnan(lab_vals) & ~np.isnan(llm_at_lab)
                n_valid = valid.sum(axis=1)
                lab_sum = np.where(valid, lab_vals, 0.0).sum(axis=1)
                llm_sum = np.where(valid, llm_at_lab, 0.0).sum(axis=1)
                has_lab = n_valid > 0
                rectifier = np.zeros(n_entities)
                rectifier[has_lab] = (
                    lab_sum[has_lab] / n_valid[has_lab] - llm_sum[has_lab] / n_valid[has_lab]
                )
                agg = agg + rectifier

            _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, n_entities + 1)).sum(axis=1)
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
