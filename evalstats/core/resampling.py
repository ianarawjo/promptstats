"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from scipy import stats
from scipy.special import betaln

from ..config import BOOTSTRAP_AUTO_MIN_N


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
    input_idx: Optional[np.ndarray] = None,
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
    input_idx: Optional[np.ndarray] = None,
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


def binary_routing_applies(
    scores: np.ndarray,
    score_range: Optional[tuple[float, float]] = None,
    *,
    stacklevel: int = 3,
) -> bool:
    """Whether all-{0, 1} *scores* should route to the binary CI methods.

    :func:`is_binary_scores` answers a question about the *sample*: are these
    values all 0 or 1? Routing needs the question about the *population*: is
    this metric Bernoulli? Those come apart whenever the caller has declared a
    ``score_range`` wider than [0, 1] -- a 1-5 Likert scale where every
    response landed on the floor, or a 0-100 grade where this particular
    sample happens to contain only 0s and 1s. Treating those as Bernoulli
    repeats the mistake :func:`degenerate_sample_ci` exists to avoid: reading
    the sample's observed support as the population's support, when the
    caller has explicitly said the metric ranges wider. The consequences are
    concrete -- an all-floor 1-5 Likert sample used to get Wilson's
    ``[0.886, 1.0]``, whose lower bound sits *below* the scale's minimum and
    which opens downward from data that can only go up.

    So an explicitly passed ``score_range`` wins: it is a direct statement
    about the metric, while binary detection is an inference from the values
    that happened to be sampled. A ``score_range`` of exactly (0, 1) agrees
    with the detection and changes nothing, and passing nothing at all leaves
    auto-detection fully in charge -- the overwhelmingly common case, which
    behaves exactly as before.

    Parameters
    ----------
    scores : np.ndarray
        Any-shape score array.
    score_range : (float, float), optional
        The metric's declared bounds, or None if the caller didn't say.
    stacklevel : int
        Passed through to ``warnings.warn`` so the override is reported at
        the user's own call site.

    Returns
    -------
    bool
        True to use the binary methods (Wilson/Newcombe/mj_floor), False to fall
        through to the bounds-aware continuous/Likert routing.
    """
    if not is_binary_scores(scores):
        return False
    if score_range is None:
        return True
    lo, hi = float(score_range[0]), float(score_range[1])
    if lo == 0.0 and hi == 1.0:
        return True
    warnings.warn(
        f"All scores are 0 or 1, which would normally auto-detect as binary "
        f"data, but score_range={score_range} was given explicitly -- so this "
        "is being treated as bounded numeric data on that scale, not as a "
        "Bernoulli metric. The explicit range wins because a sample "
        "containing only 0s and 1s doesn't establish that the metric can't "
        "take other values (e.g. every response landing on a Likert scale's "
        "floor, or a 0-100 grade where this sample happened to score only 0 "
        "or 1); the binary methods would treat that unseen headroom as "
        "impossible. Drop score_range (or pass score_range=(0, 1)) if the "
        "metric really is binary.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return False


def detect_quantization_step(scores: np.ndarray) -> Optional[float]:
    """Detect whether *scores* sit on a consistent quantization grid (e.g.
    integer-valued Likert responses, or a percentage grade rounded to whole
    points), returning the grid step -- or ``None`` if no consistent grid is
    found (the data looks genuinely continuous).

    Used to auto-detect discrete/ordinal bounded data so :func:`analyze` can
    route to NIG (calibrated for this case) instead of logit-t. Takes the
    SMALLEST observed gap between distinct values as a candidate step, then
    verifies every other gap is (within tolerance) an integer multiple of
    it -- a GCD-style check, not a "does the most common gap recur >= N
    times" frequency threshold, which is blind exactly where this matters
    most: a small, peaked/boundary-heavy sample can collapse to just 2-3
    distinct values, too few for any gap to recur several times even when
    the grid (e.g. step=1) is completely unambiguous.

    False-positive risk on genuinely continuous data is close to zero:
    demanding EVERY gap (not just the most common one) independently land
    within tolerance of an integer multiple of the candidate step has
    vanishing probability by chance (verified empirically down to n=6
    pooled values, 0% false-positive rate up to n=1000). Ported from
    simulations/harness/cases/ci_paired.py's ``_detect_dither_halfwidth``,
    which found the same regression this guards against: a frequency-based
    predecessor of this check went blind on small, peaked Likert samples.

    Parameters
    ----------
    scores : np.ndarray
        Any-shape score array (raw values, not yet rescaled).

    Returns
    -------
    float or None
        The detected step, or ``None`` if the data doesn't look quantized.
    """
    flat = scores.ravel()
    finite = flat[np.isfinite(flat)]
    uniq = np.unique(finite)
    if uniq.size < 2:
        return None
    gaps = np.diff(uniq)
    gaps = gaps[gaps > 1e-9]
    if gaps.size == 0:
        return None
    step = float(np.min(gaps))
    ratios = gaps / step
    residuals = np.abs(ratios - np.round(ratios))
    if np.max(residuals) > 0.05:
        return None
    return step


def is_lopsided_binary(scores: np.ndarray, threshold: int = 5) -> bool:
    """Return True if any compared group has fewer than *threshold* observed
    instances of its rarer binary outcome (e.g. only 2 ones out of 40).

    Small-sample binary comparisons with a heavily skewed 0/1 split are the
    regime where resampling-based FWER corrections (joint bootstrap,
    Romano-Wolf step-down) can misbehave -- too few of the rarer outcome
    makes the bootstrap's resampled distribution degenerate or lumpy -- while
    Sidak's closed-form, independence-based adjustment stays reliable. Used
    to force the small-N branch of the FWER auto-routing tables regardless
    of the overall sample size N or number of comparisons k (see
    fig:fwer-decision-tree).

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(N_groups, M)`` or ``(N_groups, M, R)`` --
        one row per compared entity/group. Assumed already known to be
        binary (call :func:`is_binary_scores` first); non-binary values are
        ignored rather than raising.
    threshold : int
        Minimum required observed count of the rarer outcome per group
        (default 5, matching the "<5 expected observations" rule of thumb).

    Returns
    -------
    bool
        True if ANY group has ``min(n_ones, n_zeros) < threshold``.
    """
    n_groups = scores.shape[0]
    for i in range(n_groups):
        flat = scores[i].ravel()
        finite = flat[np.isfinite(flat)]
        if len(finite) == 0:
            continue
        n_ones = int(np.sum(finite == 1.0))
        n_zeros = int(np.sum(finite == 0.0))
        if min(n_ones, n_zeros) < threshold:
            return True
    return False


def is_bounded_01_scores(scores: np.ndarray) -> bool:
    """Return True if all finite values in *scores* lie within [0, 1].

    Used to auto-detect continuous [0, 1] evaluation data (e.g. normalised
    accuracy, ROUGE, similarity scores) so that :func:`analyze` can switch to
    the NIG credible interval for single-sample marginal CIs.  Call
    :func:`is_binary_scores` first; if that returns True the data is binary
    and NIG is not appropriate.

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
    return bool(np.all(finite >= 0.0) and np.all(finite <= 1.0))


def resolve_score_bounds(
    scores: np.ndarray,
    score_range: Optional[tuple[float, float]] = None,
    *,
    stacklevel: int = 2,
) -> Optional[tuple[float, float]]:
    """Resolve the ``[lo, hi]`` bounds used to rescale numeric data onto
    ``[0, 1]`` for bounds-dependent methods (``logit_t``, ``nig``).

    Only call this for data that has already been confirmed non-binary
    (see :func:`is_binary_scores`) -- binary data has exact, unambiguous
    bounds and never needs a ``score_range``.

    Resolution order:

    1. ``score_range`` explicit — used as-is, after checking every finite
       value in *scores* actually falls within it (raises ``ValueError``
       otherwise; a declared range that the data violates is a user error,
       not something to silently paper over). No warning: this is an
       informed, explicit choice.
    2. All finite values already lie in ``[0, 1]`` (see
       :func:`is_bounded_01_scores`) — returns ``(0.0, 1.0)`` exactly. This
       is the common case (accuracy, ROUGE, similarity scores, ...) and
       needs no approximation, but a ``UserWarning`` is still emitted
       announcing the auto-detected range and the method it selects, since
       it's still an inference from the data rather than something the
       caller stated.
    3. Otherwise — returns ``None``. There is no reliable way to infer a
       finite range for data that falls outside ``[0, 1]`` without the
       caller's input (the sample's own min/max is *not* a safe substitute
       for the metric's true theoretical range -- e.g. a 1-5 Likert scale
       sampled only between 2 and 4). Callers should fall back to a
       bounds-agnostic method (e.g. ``t_interval``) in this case, and are
       expected to emit their own ``UserWarning`` recommending an explicit
       ``score_range`` (this function doesn't warn here itself, since the
       right fallback method/message differs by call site -- ``auto``
       routing silently downgrades, while an explicit ``method='logit_t'``
       request should instead raise).

    Parameters
    ----------
    scores : np.ndarray
        Any-shape score array, already known to be non-binary.
    score_range : tuple[float, float], optional
        User-declared ``(lo, hi)`` bounds for the eval metric, e.g.
        ``(0, 1)`` for normalised accuracy or ``(1, 5)`` for a Likert scale.
    stacklevel : int, optional
        Passed through to ``warnings.warn`` so the auto-detection warning
        points at the caller's caller (default assumes one intermediate
        frame, e.g. ``_analyze_single``).

    Returns
    -------
    (lo, hi) : tuple[float, float], or None
        ``None`` when no reliable range could be established.

    Raises
    ------
    ValueError
        If ``score_range`` is given but some value in *scores* falls
        outside it, or if ``score_range`` itself is degenerate (``lo >= hi``).
    """
    flat = scores.ravel()
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        raise ValueError("resolve_score_bounds requires at least one finite value.")

    if score_range is not None:
        lo, hi = float(score_range[0]), float(score_range[1])
        if lo >= hi:
            raise ValueError(f"score_range must satisfy lo < hi; got {score_range!r}.")
        if np.any(finite < lo) or np.any(finite > hi):
            bad_lo = float(np.min(finite))
            bad_hi = float(np.max(finite))
            raise ValueError(
                f"score_range={score_range!r} was given, but the data ranges "
                f"from {bad_lo:g} to {bad_hi:g}, which falls outside it. Either "
                "the declared range is wrong, or the data contains values "
                "that shouldn't be there."
            )
        return lo, hi

    if bool(np.all(finite >= 0.0) and np.all(finite <= 1.0)):
        warnings.warn(
            "Numeric evaluation data was auto-detected as [0, 1]-bounded "
            "(e.g. normalised accuracy, ROUGE) with no explicit score_range "
            "given, so evalstats is using method='logit_t' with score_range="
            "(0, 1). If this metric's true range isn't actually [0, 1], pass "
            "score_range=(true_min, true_max) explicitly to avoid a "
            "miscalibrated CI.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return 0.0, 1.0

    return None


def wald_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wald (normal-approximation) confidence interval for a binomial proportion.

    ``p̂ ± z_{α/2} · sqrt(p̂(1−p̂)/n)``, clamped to [0, 1].

    Known to under-cover near p=0 or p=1 and over-cover near p=0.5; included
    as the standard baseline that more accurate methods (Wilson, BCa, …) are
    compared against.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    alpha : float
        Significance level.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval clamped to [0, 1].
    """
    if n <= 0:
        return (0.0, 0.0)
    elif not (0 <= successes <= n):
        raise ValueError("successes must be in [0, n]")
    p_hat = successes / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    radius = z * np.sqrt(p_hat * (1.0 - p_hat) / n)
    return (max(0.0, float(p_hat - radius)), min(1.0, float(p_hat + radius)))


def wald_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Wald CI for a 1-D binary (0/1) array."""
    n = len(values)
    successes = int(np.sum(values))
    return wald_ci(successes, n, alpha)


def clopper_pearson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson 'exact' confidence interval for a binomial proportion.

    Uses the Beta-distribution quantile inversion::

        lo = Beta(α/2;  k,   n−k+1)
        hi = Beta(1−α/2; k+1, n−k)

    Guarantees at least nominal coverage for all p and n (conservative).

    Parameters
    ----------
    successes : int
        Number of successes (k).
    n : int
        Total number of trials.
    alpha : float
        Significance level.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval in [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    k = int(successes)
    lo = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)) if k < n else 1.0
    return (lo, hi)


def mj_floor_paired_ci_flat(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Score CI treating multi-run data as a single-run flat baseline.

    When ``values_a`` / ``values_b`` are 2-D arrays of shape ``(N, R)``,
    only the **first run** (column 0) is used.  This is the honest
    single-run baseline: one (A, B) binary pair per input, exactly as if
    each input had been run once.  Concatenating all N*R observations as
    independent pairs would inflate *n* and cause severe under-coverage;
    this function avoids that by keeping the input as the unit of analysis.

    If 1-D arrays are passed the call is forwarded directly to
    :func:`mj_floor_paired_ci` unchanged.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Either 1-D arrays of length N, or 2-D arrays of shape (N, R).
    alpha : float
        Significance level.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.ndim == 2:
        a = a[:, 0]
    if b.ndim == 2:
        b = b[:, 0]
    return mj_floor_paired_ci(a, b, alpha)


def mj_floor_paired_ci_mean(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """DO NOT USE for multi-run coverage. Kept for reference only.

    Thresholding the run mean at 0.5 changes the estimand: it targets
    E[1{mean_a >= 0.5}] - E[1{mean_b >= 0.5}], a majority-vote difference,
    not the run-and-item-averaged difference E[a] - E[b] the other paired
    methods estimate. Measured consequence on multi-run data: mean coverage
    .843, MinCov 0.000, 1550 of 4536 cells below .90, and it gets WORSE with
    more runs (.877 at R=2 down to .803 at R=20). The same pathology holds
    for the Newcombe and Bonett-Price analogues, so it is the reduction that
    is broken, not the interval. Use
    :func:`bonett_price_paired_ci_multirun_cluster` for multi-run data.

    Original description follows.
Heuristic score CI using per-item run means for multi-run inputs.

    When ``values_a`` / ``values_b`` are 2-D arrays of shape ``(N, R)``,
    each item is first reduced to its run mean (shape ``(N,)``), then
    :func:`mj_floor_paired_ci` is applied.

    This is intentionally a pragmatic variant, not a strict score
    interval derivation: :func:`mj_floor_paired_ci` was derived for paired
    Bernoulli observations, while run means live in ``[0, 1]`` and are
    thresholded at 0.5 inside :func:`mj_floor_paired_ci`.

    If 1-D arrays are passed the call is forwarded directly to
    :func:`mj_floor_paired_ci` unchanged.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Either 1-D arrays of length N, or 2-D arrays of shape (N, R).
    alpha : float
        Significance level.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.ndim == 2:
        a = np.mean(a, axis=1)
    if b.ndim == 2:
        b = np.mean(b, axis=1)
    return mj_floor_paired_ci(a, b, alpha)


def bonett_price_paired_ci_flat(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bonett-Price CI treating multi-run data as a single-run flat baseline.

    The Bonett-Price counterpart of :func:`mj_floor_paired_ci_flat`, and the
    honest single-run reference the multi-run variants have to beat: when
    ``values_a`` / ``values_b`` are 2-D ``(N, R)`` arrays only the **first
    run** (column 0) is used, i.e. exactly the data you would have had if
    each input were run once. Flattening all ``N*R`` observations into one
    long vector of "independent" pairs instead would inflate ``n`` to ``N*R``
    while the real information stays at the item scale, and under-cover badly.

    If 1-D arrays are passed the call is forwarded to
    :func:`bonett_price_paired_ci` unchanged.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.ndim == 2:
        a = a[:, 0]
    if b.ndim == 2:
        b = b[:, 0]
    return bonett_price_paired_ci(a, b, alpha)


def bonett_price_paired_ci_mean(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Heuristic Bonett-Price CI using per-item run means for multi-run inputs.

    The Bonett-Price counterpart of :func:`mj_floor_paired_ci_mean`: each item
    is reduced to its run mean (shape ``(N,)``) and then thresholded at 0.5
    inside :func:`bonett_price_paired_ci`, i.e. every item is scored by
    majority vote across its runs.

    Deliberately a pragmatic baseline, not a derivation. Two things are wrong
    with it and both are worth stating, because they are the reason the
    ``multirun`` variants below exist:

    1. It changes the ESTIMAND. Thresholding the run means estimates
       ``p(majority-vote A = 1) - p(majority-vote B = 1)``, not the
       run-and-item-averaged ``p(A=1) - p(B=1)`` that the multi-run variants
       (and the harness's ``true_diff``) target. Majority voting is a
       different, less noisy system than the one being evaluated, so the two
       estimands only coincide when the per-item run distributions are
       symmetric about the threshold.
    2. It discards the within-item run spread entirely, so a knife-edge item
       (half its runs 1, half 0) is recorded with the same confidence as a
       deterministic one.

    If 1-D arrays are passed the call is forwarded to
    :func:`bonett_price_paired_ci` unchanged.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.ndim == 2:
        a = np.mean(a, axis=1)
    if b.ndim == 2:
        b = np.mean(b, axis=1)
    return bonett_price_paired_ci(a, b, alpha)


def clopper_pearson_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson CI for a 1-D binary (0/1) array."""
    n = len(values)
    successes = int(np.sum(values))
    return clopper_pearson_ci(successes, n, alpha)


def jeffreys_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Jeffreys interval for a binomial proportion.

    Uses the equal-tailed posterior interval under the Jeffreys prior
    ``Beta(1/2, 1/2)``:

    ``lo = Beta^{-1}(alpha/2; k+1/2, n-k+1/2)``
    ``hi = Beta^{-1}(1-alpha/2; k+1/2, n-k+1/2)``

    This interval is often better calibrated than Wald near boundaries while
    remaining less conservative than Clopper-Pearson.

    Parameters
    ----------
    successes : int
        Number of successes (k).
    n : int
        Total number of trials.
    alpha : float
        Significance level.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval in [0, 1]. Returns (0.0, 1.0) when n <= 0.
    """
    if n <= 0:
        return (0.0, 1.0)
    k = int(successes)
    lo = float(stats.beta.ppf(alpha / 2.0, k + 0.5, n - k + 0.5))
    hi = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 0.5, n - k + 0.5))
    return (max(0.0, lo), min(1.0, hi))


def jeffreys_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Jeffreys interval for a 1-D binary (0/1) array."""
    n = len(values)
    successes = int(np.sum(values))
    return jeffreys_ci(successes, n, alpha)


def t_interval_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Student's t confidence interval for the mean of a 1-D array.

    ``x̄ ± t_{n−1, α/2} · s/√n``

    Valid for approximately normal data; converges to the correct interval
    by CLT for large n regardless of distribution.  This is the standard
    frequentist baseline for continuous-score data.

    Returns a degenerate point interval ``(x̄, x̄)`` when n ≤ 1 or s = 0.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    alpha : float
        Significance level (1 − confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    n = len(values)
    if n <= 1:
        mean = float(np.mean(values)) if n == 1 else 0.0
        return (mean, mean)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    if se <= 0.0 or not np.isfinite(se):
        return (mean, mean)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return (mean - t_crit * se, mean + t_crit * se)


def beta_ci_1d(
    values: np.ndarray,
    alpha: float,
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Parametric-bootstrap Beta CI for the mean of bounded [0, 1] data.

    Fits a Beta(a, b) distribution via **method of moments** (matching the
    sample mean and variance), then estimates the sampling distribution of the
    mean by drawing ``n_bootstrap`` synthetic samples of the same size from
    the fitted distribution and taking equal-tailed percentiles.

    Using method-of-moments rather than MLE ensures the fitted Beta variance
    always equals the empirical sample variance, preventing coverage collapse
    for misspecified or zero-inflated distributions at large n.

    Falls back to :func:`t_interval_ci_1d` when the MOM fit is degenerate
    (e.g. all values equal, or sample variance ≥ x̄(1−x̄)).

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed scores in [0, 1].
    alpha : float
        Significance level (1 − confidence level).
    n_bootstrap : int
        Number of parametric bootstrap replicates (default 2000).
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval clamped to [0, 1].
    """
    n = len(values)
    if n <= 0:
        return (0.0, 1.0)
    vals = np.asarray(values, dtype=float)
    x_bar = float(np.mean(vals))
    s2 = float(np.var(vals, ddof=1))
    if n > 1 and float(np.ptp(vals)) == 0.0:
        # Constant sample: the MOM fit is undefined and the t-interval this
        # used to fall back to is itself zero-width here. Same binomial
        # worst-case bound logit_t_ci_1d uses -- see degenerate_sample_ci.
        return degenerate_sample_ci(float(vals[0]), n, alpha)
    if s2 <= 0.0 or not np.isfinite(s2) or x_bar <= 0.0 or x_bar >= 1.0:
        return t_interval_ci_1d(vals, alpha)
    # Method-of-moments: concentration κ = a+b from mean and variance
    # σ² = μ(1−μ)/(κ+1)  →  κ = μ(1−μ)/σ² − 1
    conc = x_bar * (1.0 - x_bar) / s2 - 1.0
    if conc <= 0.0:
        return t_interval_ci_1d(vals, alpha)
    a = x_bar * conc
    b = (1.0 - x_bar) * conc
    if rng is None:
        rng = np.random.default_rng()
    boot_means = rng.beta(a, b, size=(n_bootstrap, n)).mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (max(0.0, lo), min(1.0, hi))


_LOGIT_T_BOUNDARY_EPS = 1e-9
"""Tolerance for treating an out-of-[0,1] value as floating-point rounding
noise (e.g. 1.0000000000000004 from an upstream `score * scale` rescale)
rather than genuinely bad data -- see logit_t_ci_1d's docstring."""


def degenerate_sample_ci(
    value: float, n: int, alpha: float, lo: float = 0.0, hi: float = 1.0,
) -> tuple[float, float]:
    """Conservative CI for E[X] when all *n* observed values are identical.

    A zero-variance sample carries no information about spread, so every
    variance-driven interval (the delta method, the t-interval, any
    resampling scheme) degenerates to zero width and covers the truth with
    probability 0 whenever the population isn't genuinely a point mass. That
    isn't a rounding artifact -- it's the honest answer to the wrong
    question. The right question is what the sample *does* pin down.

    Treat "X == value" as a Bernoulli success. Observing n successes out of n
    gives the exact (Clopper-Pearson) lower confidence bound

        p = P(X = value) >= (alpha/2) ** (1/n)

    and the remaining 1-p of the mass is unconstrained within the metric's
    known bounds [lo, hi]. So

        E[X]  in  [p*value + (1-p)*lo,  p*value + (1-p)*hi]

    covers E[X] for *any* configuration of that unseen mass whenever the
    bound on p holds: the interval's endpoints are attained exactly at the
    worst cases (all remaining mass at lo, or all at hi), and both endpoints
    move monotonically in p (the gap between the truth and the endpoint is
    (p - p_lo)*(value - lo) >= 0 at the bottom and (p - p_lo)*(value - hi)
    <= 0 at the top). And when the bound fails -- true p < p_lo -- a
    degenerate sample only arises with probability p**n < alpha/2 in the
    first place, so the branch contributes at most alpha/2 to the overall
    miss rate.

    For all-successes binary data (value == hi == 1, lo == 0) this reduces to
    exactly the two-sided Clopper-Pearson interval [(alpha/2)**(1/n), 1],
    which is the answer the binary methods already give -- so the bounded
    continuous path and the binary path agree at the boundary instead of
    disagreeing by the full width of the interval.

    The price is conservatism: width is (1-p)*(hi-lo), roughly
    ln(2/alpha)*(hi-lo)/n -- about 0.15 at n=25 on a [0,1] metric, shrinking
    like 1/n. That is the correct price for a sample that shows no spread at
    all, and it is only ever paid on samples that would otherwise have been
    reported with false certainty.

    Parameters
    ----------
    value : float
        The single value every observation took, assumed within [lo, hi].
    n : int
        Number of observations (>= 1).
    alpha : float
        Significance level (1 - confidence level).
    lo, hi : float
        The metric's known bounds. Callers working on a rescaled [0, 1] axis
        (see ``stats_utils.rescaled_ci``) should leave these at the default
        and let the wrapper map the result back.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval clamped to [lo, hi].
    """
    if n < 1:
        return (lo, hi)
    p = float(alpha / 2.0) ** (1.0 / n)
    ci_low = p * value + (1.0 - p) * lo
    ci_high = p * value + (1.0 - p) * hi
    return (max(lo, ci_low), min(hi, ci_high))


def logit_t_ci_1d(values: np.ndarray, alpha: float, order: int = 1) -> tuple[float, float]:
    """Logit-transform t-interval (delta method) for [0, 1]-bounded data.

    Applies the delta method to obtain a CI for the arithmetic mean E[X]:

    1. Compute the sample mean x̄ and its standard error SE = s/√n.
    2. Map to the logit scale: g = log(x̄/(1−x̄)).
    3. If ``order >= 2``, bias-correct the logit-scale point estimate for
       the transform's own curvature (see below) -- default ``order=1``
       skips this and uses the plain first-order delta method.
    4. Propagate uncertainty: SE_logit ≈ SE / (x̄(1−x̄)).
    5. Form a t-interval on the logit scale: g ± t_{n−1} · SE_logit.
    6. Back-transform via the sigmoid to recover bounds on [0, 1].

    This targets E[X] directly (not E[logit(X)]), and the asymmetric
    back-transformed interval is better calibrated than a symmetric t-interval
    for skewed or boundary-hugging distributions.

    Values within ``_LOGIT_T_BOUNDARY_EPS`` of 0 or 1 but technically outside
    are treated as floating-point rounding noise: clipped to [0, 1] with a
    ``UserWarning``, not rejected. Anything further outside still raises
    ``ValueError`` -- that's a real data problem (e.g. forgetting to rescale
    a non-[0,1] metric), not rounding, and should surface loudly rather than
    be silently "fixed". This distinction was added after a real incident
    (2026-07-27): one item in a real OpenEval corpus (grok-4/truthfulqa) had
    value 1.0000000000000004 from upstream score*scale rescaling, which the
    unconditional raise rejected outright -- and the calling harness
    (cases/ci_single.py) wrapped every CI computation in a blanket
    ``except Exception: ci_low = ci_high = obs_mean``, silently turning that
    single rejected sample into a zero-width, essentially-never-covering
    interval. Since that item's without-replacement inclusion probability
    scaled with n/corpus_size, more reps got silently corrupted as n grew,
    producing a coverage curve that fell from ~90% (n=15) to ~32% (n=500) --
    which read exactly like a genuine, worsening-with-n calibration failure.
    It wasn't: once real_data.py's builders were fixed to clip their own
    rescaled output (the actual right place to fix upstream data hygiene),
    the plain order=1 delta method covered fine (92-99% across the same n
    range, no collapse) -- there was no real logit-transform weakness on
    that data after all. This eps-tolerant clip+warn is a second line of
    defense so a similar rounding artifact from a different, not-yet-audited
    caller degrades gracefully (loud warning, still-valid interval) instead
    of silently masquerading as a statistical finding again.

    order : optional 2nd-order bias correction. A first-order delta method
    linearizes g=logit around x̄ and ignores g's own curvature; in principle,
    for a true mean very close to a boundary (small x̄, large g''(x̄)) that
    ignored curvature could become a non-negligible, slowly-vanishing bias.
    The standard Taylor correction, E[g(X̄)] ≈ g(μ) + ½g''(μ)Var(X̄), is
    available via ``order=2`` (subtracts ½g''(x̄)·SE² from the logit-scale
    point estimate before forming the interval; g''(x) = -1/x² + 1/(1-x)²).
    In the same investigation above, once the real data-hygiene bug was
    fixed, order=2 tracked order=1 almost exactly on that (non-skewed)
    dataset -- but a later, dedicated investigation (2026-08-04, see
    simulations/investigate_logit_t_boundary.py and the harness's
    logit_t_boundary_investigation memory) tested order=2 on genuinely
    boundary-hugging, right-skewed single-sample data (e.g. BLEU-style
    automated metrics) and found a real, consistent coverage improvement at
    small n, at negligible width cost on well-behaved data -- so "no
    measurable benefit" does NOT generalize; it was specific to that one
    non-skewed case. ``order=1`` (cheaper, one fewer term, easier to audit)
    is this function's default, and stays so everywhere in this project
    (simulations/harness/cases/ci_single.py briefly defaulted its LOGIT_T
    method to order=2 over the above finding, then reverted it the same day
    -- the gain was real but too modest to justify re-running the paper's
    simulations/rewriting results over; see LOGIT_T_2ND's Method-registry
    comment in simulations/harness/methods.py for the opt-in comparison
    variant kept for whenever that tradeoff gets revisited). order=2 was
    also confirmed to NOT help cases/ci_paired.py's use of logit_t, since
    paired diffs get rescaled to center near 0.5 regardless of marginal
    skew, where order=2's boundary-only correction never activates. A
    3rd-order term (correcting for g'''(x̄) and the sample's third central
    moment) was tried in both the original (misdiagnosed) and the 2026-08-04
    genuinely-
    skewed investigations, and gave no additional improvement over 2nd-order
    in either -- the noisy third-moment estimate at small n cancels out any
    theoretical gain -- so it isn't offered as an option.

    A **zero-variance sample** (every value identical, which includes the
    all-0s and all-1s boundary cases) is handed to
    :func:`degenerate_sample_ci` rather than being reported as the zero-width
    interval the delta method implies. This matters on saturated metrics: on
    a one-inflated DGP at 95% inflation, 36% of n=20 samples come out
    constant, and before this fallback existed those reps dragged marginal
    coverage from a nominal 95% down to 60% -- while coverage *conditional*
    on a non-degenerate sample stayed at 94.6%. The transform was never the
    problem; the zero-width branch was, and it is shared with
    ``t_interval``/``beta``/the bootstrap methods rather than being specific
    to logit-t. See ``simulations/harness/scenarios/synthetic.py``'s
    ``cont-{zero,one}-inflated-extreme`` shapes, added so the suite actually
    reaches the regime where this fires (the pre-existing 70%-inflation
    shapes produce a constant sample <3% of the time even at n=10, which is
    why routine sweeps gave a clean bill of health here).

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed scores in [0, 1] (values within
        ``_LOGIT_T_BOUNDARY_EPS`` of the boundary are clipped, not rejected).
    alpha : float
        Significance level (1 − confidence level).
    order : int
        1 (default) for the plain first-order delta method, or 2 for the
        curvature-corrected variant -- see above.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Interval clamped to [0, 1].
    """
    n = len(values)
    if n <= 1:
        mean = float(np.mean(values)) if n == 1 else 0.0
        return (mean, mean)
    vals = np.asarray(values, dtype=float)
    if np.any(vals < -_LOGIT_T_BOUNDARY_EPS) or np.any(vals > 1.0 + _LOGIT_T_BOUNDARY_EPS):
        raise ValueError("logit_t_ci_1d requires all values in [0, 1]")
    out_of_range = (vals < 0.0) | (vals > 1.0)
    if np.any(out_of_range):
        warnings.warn(
            f"logit_t_ci_1d: {int(np.sum(out_of_range))} value(s) fractionally "
            f"outside [0, 1] (within {_LOGIT_T_BOUNDARY_EPS:g}, consistent with "
            "floating-point rounding) clipped to [0, 1].",
            UserWarning, stacklevel=2,
        )
        vals = np.clip(vals, 0.0, 1.0)
    x_bar = float(np.mean(vals))
    se = float(np.std(vals, ddof=1)) / np.sqrt(n)
    if float(np.ptp(vals)) == 0.0:
        # Zero-variance sample (all values identical -- including the all-0s
        # and all-1s boundary cases). The delta method has nothing to
        # propagate here and would report a zero-width interval; hand off to
        # the binomial worst-case bound instead. See degenerate_sample_ci.
        return degenerate_sample_ci(float(vals[0]), n, alpha)
    if se <= 0.0 or not np.isfinite(se) or x_bar <= 0.0 or x_bar >= 1.0:
        # Only reachable now for non-finite input (NaN/inf), since within
        # [0, 1] both x_bar == 0 and x_bar == 1 imply a constant sample.
        return (x_bar, x_bar)
    # Delta method: SE of logit(x̄) ≈ SE(x̄) / (x̄(1−x̄))
    logit_mean = float(np.log(x_bar / (1.0 - x_bar)))
    if order >= 2:
        # 2nd-order bias correction: g''(x) = -1/x² + 1/(1-x)² for g=logit(x)
        # (equivalently (2x-1)/(x²(1-x)²)); subtract ½g''(x̄)·SE² per the
        # standard delta-method Taylor bias correction -- see docstring.
        g2 = -1.0 / x_bar**2 + 1.0 / (1.0 - x_bar) ** 2
        logit_mean -= 0.5 * g2 * se**2
    se_logit = se / (x_bar * (1.0 - x_bar))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    lo = float(1.0 / (1.0 + np.exp(-(logit_mean - t_crit * se_logit))))
    hi = float(1.0 / (1.0 + np.exp(-(logit_mean + t_crit * se_logit))))
    return (max(0.0, lo), min(1.0, hi))


def nig_ci_1d(
    values: np.ndarray,
    alpha: float,
    m0: float = 0.5,
    k0: float = 1.0,
    a0: float = 2.0,
    b0: float = 0.0625,
) -> tuple[float, float]:
    """Normal-Inverse-Gamma Bayesian credible interval for continuous data.

    Places a NIG(m₀, κ₀, α₀, β₀) conjugate prior on (μ, σ²) and returns an
    equal-tailed (1−α) credible interval for μ.  The marginal posterior is::

        μ | data  ~  t(2αₙ,  mₙ,  √(βₙ / (αₙ κₙ)))

    with posterior hyperparameters updated analytically.

    Default prior encodes weak knowledge that scores live in [0, 1]: centre
    m₀=0.5, worth κ₀=1 pseudo-observation, prior variance of σ² centred at
    b₀/(a₀−1)=0.0625 (i.e. σ≈0.25).  With κ₀→0 and α₀→−½ this recovers
    the frequentist t-interval exactly.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed scores.
    alpha : float
        Significance level (1 − confidence level).
    m0, k0, a0, b0 : float
        Prior hyperparameters (mean, strength, shape, rate).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        scale = float(np.sqrt(b0 * (k0 + 1.0) / (a0 * k0)))
        lo = float(stats.t.ppf(alpha / 2.0, df=2.0 * a0, loc=m0, scale=scale))
        hi = float(stats.t.ppf(1.0 - alpha / 2.0, df=2.0 * a0, loc=m0, scale=scale))
        return (lo, hi)
    x_bar = float(np.mean(vals))
    ss = float(np.sum((vals - x_bar) ** 2))
    # Posterior hyperparameter updates
    kn = k0 + n
    mn = (k0 * m0 + n * x_bar) / kn
    an = a0 + n / 2.0
    bn = b0 + 0.5 * ss + (k0 * n) / (2.0 * kn) * (x_bar - m0) ** 2
    scale = float(np.sqrt(bn / (an * kn)))
    if scale <= 0.0 or not np.isfinite(scale):
        return (mn, mn)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=2.0 * an))
    return (mn - t_crit * scale, mn + t_crit * scale)


def nig_ci_nested(
    values: np.ndarray,   # shape (N, R) or (N,) fallback
    alpha: float,
    m0: float = 0.5,
    k0: float = 1.0,
    a0: float = 2.0,
    b0: float = 0.0625,
) -> tuple[float, float]:
    """
    Normal-Inverse-Gamma CI for the grand mean of multi-run data.

    Supports:
      - (N,)    -> falls back to standard NIG (``nig_ci_1d``)
      - (N, R)  -> per-item means (NaN-robust, so unbalanced run counts are
                   fine), then standard NIG on those means

    Model:
        X_ir ~ Normal(theta_i, sigma_run^2)
        theta_i ~ Normal(mu, sigma_item^2)

    So ``Var(item_mean_i) = sigma_item^2 + sigma_run^2/R_i`` -- which is
    exactly what the empirical variance of the *observed* item means already
    estimates directly, no further correction needed. An earlier version
    computed ``Var(item_means)`` as if it were an estimate of
    ``sigma_item^2`` alone (calling it ``s2_item``) and then added
    ``sigma_run^2/R`` back on top as an "inflation" -- silently double-
    counting the run-noise contribution that ``Var(item_means)`` already
    included. Verified empirically: constructing data with ``sigma_item^2 =
    0`` by design (so the correct effective variance is exactly
    ``sigma_run^2/R``), the old formula came out ~2x too large. Subtracting
    then re-adding the same run-noise term is a no-op by construction, so
    the correct effective variance is just ``Var(item_means)`` itself --
    which also sidesteps the unbalanced-R case entirely, since it never
    needs a single scalar R at all.
    """
    vals = np.asarray(values, dtype=float)

    if vals.ndim == 1:
        return nig_ci_1d(vals, alpha, m0, k0, a0, b0)

    if vals.ndim != 2:
        raise ValueError("values must be (N,) or (N, R)")

    item_means = np.nanmean(vals, axis=1)
    return nig_ci_1d(item_means, alpha, m0, k0, a0, b0)


def el_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical-likelihood confidence interval for the mean (Owen 1988/1990).

    Constructs the profile empirical-likelihood ratio CI::

        EL-CI = {θ : −2 log R(θ) ≤ χ²(1, 1−α)}

    where R(θ) = max ∏ nᵢpᵢ s.t. Σpᵢ=1, Σpᵢxᵢ=θ.  The Lagrange multiplier
    λ(θ) is found via root-finding; the CI bounds are located by binary search
    exploiting the convexity of −2 log R.

    EL is nonparametric and Bartlett-correctable (coverage error O(n⁻²) vs
    O(n⁻¹) for bootstrap), making it attractive for small samples with skewed
    or bounded distributions.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    alpha : float
        Significance level (1 − confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    from scipy.optimize import brentq as _brentq

    n = len(values)
    if n <= 1:
        mean = float(np.mean(values)) if n == 1 else 0.0
        return (mean, mean)
    vals = np.asarray(values, dtype=float)
    x_bar = float(np.mean(vals))
    x_min, x_max = float(np.min(vals)), float(np.max(vals))
    if x_min == x_max:
        return (x_min, x_max)

    crit = float(stats.chi2.ppf(1.0 - alpha, df=1))

    def neg2logR(theta: float) -> float:
        if theta <= x_min or theta >= x_max:
            return np.inf
        d = vals - theta
        d_pos, d_neg = d[d > 0], d[d < 0]
        # Feasible range: all (1 + lambda*d_i) > 0
        lam_lo = (-1.0 / d_pos.max() + 1e-12) if len(d_pos) else -1e15
        lam_hi = (-1.0 / d_neg.min() - 1e-12) if len(d_neg) else  1e15
        if lam_lo >= lam_hi:
            return np.inf
        def _constraint(l: float) -> float:
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = d / (1.0 + l * d)
            s = float(np.sum(terms))
            # constraint is monotonically decreasing: +∞ near lam_lo, −∞ near lam_hi
            return s if np.isfinite(s) else np.sign((lam_lo + lam_hi) / 2 - l) * 1e15

        try:
            lam = _brentq(_constraint, lam_lo, lam_hi, xtol=1e-12, rtol=1e-8, maxiter=500)
        except ValueError:
            return np.inf
        log_terms = np.log(1.0 + lam * d)
        if not np.all(np.isfinite(log_terms)):
            return np.inf
        return float(2.0 * np.sum(log_terms))

    def excess(theta: float) -> float:
        return neg2logR(theta) - crit

    span_l = x_bar - x_min
    span_r = x_max - x_bar
    try:
        lo = _brentq(
            excess,
            x_min + 1e-8 * span_l,
            x_bar - 1e-10 * span_l,
            xtol=1e-8, maxiter=200,
        )
    except (ValueError, Exception):
        lo = x_min
    try:
        hi = _brentq(
            excess,
            x_bar + 1e-10 * span_r,
            x_max - 1e-8 * span_r,
            xtol=1e-8, maxiter=200,
        )
    except (ValueError, Exception):
        hi = x_max

    return (float(lo), float(hi))


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


def _wilson_neff(p_hat: float, n_eff: float, alpha: float, z: float | None = None) -> tuple[float, float]:
    """Wilson score interval parameterised by an effective sample size.

    Applies the standard Wilson formula with *n_eff* substituted for n.
    ``z`` overrides the normal quantile (e.g. with a t-quantile when the
    variance behind ``n_eff`` is itself estimated from few clusters).
    """
    if n_eff <= 0.0 or not np.isfinite(n_eff):
        return (0.0, 1.0)
    if z is None:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom  = 1.0 + z2 / n_eff
    center = (p_hat + z2 / (2.0 * n_eff)) / denom
    radius = (z / denom) * np.sqrt(
        p_hat * (1.0 - p_hat) / n_eff + z2 / (4.0 * n_eff * n_eff)
    )
    return (max(0.0, float(center - radius)), min(1.0, float(center + radius)))


def wilson_nested_de(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Wilson score CI for multi-run binary data with design-effect correction.

    Estimates the intraclass correlation (ICC) via a one-way ANOVA decomposition
    of the per-item cell means, then computes an effective sample size::

        D      = 1 + (R − 1) · ICĈ
        n_eff  = n · R / D

    where ``D`` is the design effect.  When ICC = 0 (iid runs) ``n_eff = n·R``;
    when ICC = 1 (all variance is between items) ``n_eff = n``.  The standard
    Wilson formula is then applied with ``n_eff``.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` — binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    if n < 2:
        return _wilson_neff(p_hat, float(n * R), alpha)

    s2   = float(np.var(cell_means, ddof=1))
    MS_B = R * s2                          # between-item mean square
    MS_W = p_hat * (1.0 - p_hat)          # within-item variance (binomial approx)

    if R > 1 and MS_W > 0.0 and (MS_B + (R - 1) * MS_W) > 0.0:
        icc = (MS_B - MS_W) / (MS_B + (R - 1) * MS_W)
        icc = float(np.clip(icc, 0.0, 1.0))
    else:
        icc = 0.0

    D     = 1.0 + (R - 1) * icc
    n_eff = n * R / D
    return _wilson_neff(p_hat, n_eff, alpha)


def wilson_nested_od(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Wilson score CI for multi-run binary data with overdispersion plug-in.

    Uses the sample variance of per-item cell means to estimate the total
    variance of the grand mean, then derives an effective sample size by
    expressing it as a scaled binomial variance::

        phi   = s² / (p̂(1 − p̂))          (overdispersion factor)
        n_eff = n / phi = n · p̂(1 − p̂) / s²

    This preserves the Wilson property that the interval width depends on
    the parameter *p* rather than solely on p̂.  Reduces to standard Wilson
    on the full *n·R* observations when the runs within each item are iid.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` — binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)

    if n < 2 or p_var <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha)

    s2 = float(np.var(cell_means, ddof=1))
    if s2 <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha)

    n_eff = n * p_var / s2
    return _wilson_neff(p_hat, n_eff, alpha)


def wilson_nested_od_bc(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """``wilson_nested_od`` with a bias correction for the reciprocal-variance plug-in.

    ``wilson_nested_od``'s ``n_eff = n * p_var / s2`` is biased upward: ``s2``
    is unbiased for the true between-item variance, but it enters through
    ``1/s2``, and ``1/x`` is convex, so ``E[1/s2] > 1/sigma**2`` (Jensen's
    inequality) -- worse at small ``n``, vanishing as ``n`` grows. Treating
    ``s2 ~ sigma**2 * chi2_v / v`` with ``v = n - 1`` gives
    ``E[1/s2] = 1/sigma**2 * v / (v - 2)`` for ``v > 2``, so multiplying
    ``n_eff`` by ``(v - 2) / v = (n - 3) / (n - 1)`` removes that leading-order
    bias (requires ``n > 3``; falls back to the plain ``n * R`` Wilson
    interval otherwise, same as ``wilson_nested_od``'s own small-``n`` guard).

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)

    if n <= 3 or p_var <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha)

    s2 = float(np.var(cell_means, ddof=1))
    if s2 <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha)

    n_eff = n * p_var / s2
    n_eff_bc = n_eff * (n - 3) / (n - 1)
    return _wilson_neff(p_hat, n_eff_bc, alpha)


def wilson_nested_od_t(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """``wilson_nested_od_bc`` plus a t-quantile and a hard cap on ``n_eff``.

    Two additional guards on top of the Jensen bias correction:

    1. ``n_eff`` is clipped to ``n * R``: the overdispersion plug-in
       ``n * p_var / s2`` is unbounded above when ``s2`` happens to be small,
       yielding more effective observations than actual observations and a
       spuriously tight interval. ``n * R`` (iid runs, ICC = 0) is the
       information-theoretic ceiling.
    2. The normal quantile is replaced by a t-quantile with ``n - 1`` degrees
       of freedom: ``n_eff`` is built from a between-item variance estimated
       on only ``n`` clusters, and plugging an estimated variance into a
       z-interval is exactly the error the t-interval exists to fix. The two
       quantiles converge as ``n`` grows.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(n - 1, 1)))

    if n <= 3 or p_var <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha, z=t_crit)

    s2 = float(np.var(cell_means, ddof=1))
    if s2 <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha, z=t_crit)

    n_eff = n * p_var / s2 * (n - 3) / (n - 1)
    n_eff = min(n_eff, float(n * R))
    return _wilson_neff(p_hat, n_eff, alpha, z=t_crit)


def jeffreys_nested_od(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Jeffreys credible interval for multi-run binary data, design-effect corrected.

    Reuses ``wilson_nested_od_t``'s bias-corrected, capped overdispersion
    plug-in ``n_eff``, but feeds it into a Jeffreys(1/2, 1/2) Beta posterior
    (``jeffreys_ci``'s construction) instead of a Wilson/normal-approximation
    formula: pseudo-counts ``s_eff = n_eff * p_hat``, ``f_eff = n_eff * (1 -
    p_hat)`` go to ``Beta(s_eff + 1/2, f_eff + 1/2)``.

    Motivation: ``wilson_nested_od``/``_bc``/``_t`` all still undercover at
    extreme ``p_hat``, and Jeffreys generally has good *average* coverage
    across p. In practice this variant did NOT fix the boundary undercoverage
    (empirically worse: verified against real iid Bernoulli(150, 0.98) data
    with no clustering at all, Jeffreys' own coverage there is ~0.92 vs.
    Wilson's ~0.97 -- a genuine, well-documented (Brown, Cai & DasGupta 2001)
    coverage oscillation with p, not an artifact of the design-effect
    plug-in). Kept as a reference/comparison point; see
    ``clopper_pearson_nested_od`` for the variant that actually guarantees
    coverage.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)

    if n <= 3 or p_var <= 0.0:
        n_eff = float(n * R)
    else:
        s2 = float(np.var(cell_means, ddof=1))
        if s2 <= 0.0:
            n_eff = float(n * R)
        else:
            n_eff = n * p_var / s2 * (n - 3) / (n - 1)
            n_eff = min(n_eff, float(n * R))

    if n_eff <= 0.0 or not np.isfinite(n_eff):
        return (0.0, 1.0)

    s_eff = n_eff * p_hat
    f_eff = n_eff - s_eff
    lo = float(stats.beta.ppf(alpha / 2.0, s_eff + 0.5, f_eff + 0.5))
    hi = float(stats.beta.ppf(1.0 - alpha / 2.0, s_eff + 0.5, f_eff + 0.5))
    return (max(0.0, lo), min(1.0, hi))


def clopper_pearson_nested_od(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Clopper-Pearson exact interval for multi-run binary data, design-effect corrected.

    Same bias-corrected, capped overdispersion plug-in ``n_eff`` as
    ``wilson_nested_od_t``/``jeffreys_nested_od``, but fed into the
    Clopper-Pearson exact tail-inversion formula (continuous generalisation
    of ``clopper_pearson_ci_1d``'s integer-count Beta-quantile construction)
    instead of a Wilson score or Jeffreys posterior.

    Motivation: neither Wilson nor Jeffreys is uniformly better across ``p``
    -- both have genuine, well-documented (Brown, Cai & DasGupta 2001)
    coverage dips below nominal at specific (n, p) combinations, purely from
    the discreteness of the binomial, with no clustering involved (verified
    directly against iid data: Jeffreys drops to ~0.92 at n=150, p=0.98;
    Wilson drops to ~0.94 at n=150, p=0.95). Clopper-Pearson is the one
    classical construction guaranteed to never undercover for any p, at any
    n -- exact tail-probability inversion rather than an approximation. The
    price is conservatism (wider intervals on average) rather than
    efficiency, which is the trade this variant is testing.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)

    if n <= 3 or p_var <= 0.0:
        n_eff = float(n * R)
    else:
        s2 = float(np.var(cell_means, ddof=1))
        if s2 <= 0.0:
            n_eff = float(n * R)
        else:
            n_eff = n * p_var / s2 * (n - 3) / (n - 1)
            n_eff = min(n_eff, float(n * R))

    if n_eff <= 0.0 or not np.isfinite(n_eff):
        return (0.0, 1.0)

    s_eff = n_eff * p_hat
    f_eff = n_eff - s_eff
    lo = 0.0 if s_eff <= 0.0 else float(stats.beta.ppf(alpha / 2.0, s_eff, f_eff + 1.0))
    hi = 1.0 if f_eff <= 0.0 else float(stats.beta.ppf(1.0 - alpha / 2.0, s_eff + 1.0, f_eff))
    return (max(0.0, lo), min(1.0, hi))


def beta_binomial_bayes_nested(
    scores: np.ndarray,
    alpha: float,
    n_p: int = 300,
    n_icc: int = 60,
) -> tuple[float, float]:
    """Genuine hierarchical Bayesian credible interval for multi-run binary data.

    Model: item ``i`` has a latent success probability ``p_i ~ Beta(p*kappa,
    (1-p)*kappa)`` (population mean ``p``, concentration ``kappa``); observed
    per-item successes ``k_i ~ Binomial(R, p_i)``, giving the marginal
    ``k_i ~ BetaBinomial(R, p*kappa, (1-p)*kappa)`` -- this exactly matches
    ``simulations/harness/scenarios/synthetic.py``'s ``sample_group_truth``
    binary branch, which draws item pass-probabilities the same way, with
    ``kappa = 1/ICC - 1``. Unlike ``wilson_nested_bb`` (which despite its
    name fits this same Beta-Binomial *form* by method-of-moments, collapses
    it to a single point-estimate effective sample size, and then still runs
    the Wilson score formula), this computes the actual joint posterior over
    ``(p, kappa)`` on a grid, numerically marginalizes out ``kappa``, and
    returns the equal-tailed credible interval of the resulting marginal
    posterior for ``p`` -- so uncertainty in the design effect/ICC itself
    (which is what a point-estimate n_eff plug-in silently ignores) is
    propagated into the width of the interval, rather than assumed away.

    Priors: Jeffreys ``Beta(1/2, 1/2)`` on ``p`` (matches ``jeffreys_ci``
    elsewhere in this module); **uniform on ICC in (0, 1)**, gridded directly
    in ICC space and mapped to ``kappa = 1/ICC - 1``. An earlier version
    gridded log-uniformly in kappa directly, which implies a prior density on
    ICC of ``prop 1/(ICC*(1-ICC))`` -- U-shaped, piling mass at ICC near 0
    ("no clustering"), which biased the posterior toward overconfident
    (too-narrow) intervals in exactly the same direction as the point-
    estimate n_eff methods' failure mode. Gridding uniformly in the natural,
    bounded ICC parameterization avoids that.

    Exploits the fact that each per-item count ``k_i`` only takes ``R + 1``
    distinct values: the log-likelihood only needs their histogram, not a
    per-item term, so cost is independent of ``n`` (dominated by the
    ``n_p * n_icc`` grid, not by the number of items).

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.
    n_p, n_icc : int
        Grid resolution for ``p`` and ``ICC``.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    k = scores.sum(axis=1).astype(int)
    counts = np.bincount(k, minlength=R + 1).astype(float)

    p_grid = np.linspace(1e-4, 1.0 - 1e-4, n_p)
    icc_grid = np.linspace(1e-4, 1.0 - 1e-4, n_icc)
    kappa_grid = (1.0 - icc_grid) / icc_grid
    p_col = p_grid[:, None]
    kappa_row = kappa_grid[None, :]
    a = p_col * kappa_row
    b = (1.0 - p_col) * kappa_row

    log_like = -n * betaln(a, b)
    for r in range(R + 1):
        if counts[r] == 0.0:
            continue
        log_like += counts[r] * betaln(r + a, R - r + b)

    log_prior_p = -0.5 * np.log(p_col) - 0.5 * np.log(1.0 - p_col)
    log_post = log_like + log_prior_p
    log_post -= log_post.max()
    post = np.exp(log_post)

    marg_p = post.sum(axis=1)
    total = marg_p.sum()
    if total <= 0.0 or not np.isfinite(total):
        return (0.0, 1.0)
    marg_p /= total
    cdf = np.cumsum(marg_p)
    lo = float(np.interp(alpha / 2.0, cdf, p_grid))
    hi = float(np.interp(1.0 - alpha / 2.0, cdf, p_grid))
    return (max(0.0, lo), min(1.0, hi))


def beta_binomial_bayes_robust_nested(
    scores: np.ndarray,
    alpha: float,
    n_p: int = 300,
    n_icc: int = 60,
    gamma: float = 0.001,
) -> tuple[float, float]:
    """Berger-Boos robustified version of ``beta_binomial_bayes_nested``.

    Same hierarchical Beta-Binomial model and joint ``(p, ICC)`` grid, but the
    nuisance parameter (ICC) is eliminated by *restricted maximization*
    instead of integration (Berger & Boos 1994): build a ``1 - gamma``
    profile-likelihood confidence set ``K`` for ICC, form the conditional
    ``1 - (alpha - gamma)`` credible interval for ``p`` at each ICC in ``K``,
    and report the union (min of lowers, max of uppers).

    Why: marginalizing ICC (as ``beta_binomial_bayes_nested`` does) quietly
    lets the *prior* decide the interval width whenever the data cannot
    identify ICC -- which happens exactly in the hardest regime, e.g. very
    rare successes at near-1 ICC, where a typical sample yields a degenerate
    count histogram (every item 0-of-R or R-of-R, no interior counts) whose
    likelihood is nearly flat in ICC. All-zero data justifies an upper bound
    ~``1-(alpha/2)^(1/n)`` if items are deterministic (ICC=1) but
    ~``1-(alpha/2)^(1/(nR))`` if runs are iid (ICC=0); integrating over an
    unidentified ICC lands in between and undercovers. The union pays the
    ICC=1 worst-case width *only* when the data genuinely cannot rule it
    out; when the histogram has interior counts, the profile-likelihood set
    ``K`` collapses and the interval reverts to the efficient marginal one.

    Total error budget is ``alpha``: ``gamma`` spent on the ICC confidence
    set, ``alpha - gamma`` on the conditional intervals. The conditional
    pieces are Bayesian posterior slices rather than exact tail inversions,
    so the frequentist guarantee is approximate, not exact.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` -- binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.
    n_p, n_icc : int
        Grid resolution for ``p`` and ``ICC``.
    gamma : float
        Error budget spent on the ICC profile-likelihood confidence set.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    k = scores.sum(axis=1).astype(int)
    counts = np.bincount(k, minlength=R + 1).astype(float)

    p_grid = np.linspace(1e-4, 1.0 - 1e-4, n_p)
    icc_grid = np.linspace(1e-4, 1.0 - 1e-4, n_icc)
    kappa_grid = (1.0 - icc_grid) / icc_grid
    p_col = p_grid[:, None]
    kappa_row = kappa_grid[None, :]
    a = p_col * kappa_row
    b = (1.0 - p_col) * kappa_row

    log_like = -n * betaln(a, b)
    for r in range(R + 1):
        if counts[r] == 0.0:
            continue
        log_like += counts[r] * betaln(r + a, R - r + b)

    # Profile likelihood over ICC: max over p within each ICC column, then
    # keep columns within the chi-square(1) cutoff for a 1-gamma set.
    profile = log_like.max(axis=0)
    cutoff = 0.5 * float(stats.chi2.ppf(1.0 - gamma, df=1))
    in_set = profile >= (profile.max() - cutoff)

    log_prior_p = -0.5 * np.log(p_col) - 0.5 * np.log(1.0 - p_col)
    log_post = log_like + log_prior_p
    log_post -= log_post.max()
    post = np.exp(log_post)

    alpha_eff = max(alpha - gamma, 1e-6)
    lo_best, hi_best = 1.0, 0.0
    for j in np.flatnonzero(in_set):
        col = post[:, j]
        total = col.sum()
        if total <= 0.0 or not np.isfinite(total):
            continue
        cdf = np.cumsum(col) / total
        lo_j = float(np.interp(alpha_eff / 2.0, cdf, p_grid))
        hi_j = float(np.interp(1.0 - alpha_eff / 2.0, cdf, p_grid))
        lo_best = min(lo_best, lo_j)
        hi_best = max(hi_best, hi_j)

    if hi_best < lo_best:  # every column degenerate -- give up gracefully
        return (0.0, 1.0)
    return (max(0.0, lo_best), min(1.0, hi_best))


def wilson_nested_bb(
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Wilson-style CI for multi-run binary data via Beta-Binomial model.

    Fits a Beta-Binomial(R, α, β) marginal model to the item-level success
    counts using method-of-moments, yielding a concentration parameter κ = α + β.
    The implied effective sample size is::

        n_eff = n · R · (κ + 1) / (R + κ)

    which interpolates from ``n·R`` (κ → ∞, iid runs) down to ``n``
    (κ → 0, maximum clustering) as clustering increases.  The standard Wilson
    formula is then applied with this ``n_eff``.

    Falls back to ``n_eff = n·R`` when no overdispersion is detected
    (s² ≤ p̂(1 − p̂)/R) and to ``n_eff = n`` for maximum overdispersion
    (s² ≥ p̂(1 − p̂)).

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(n, R)`` — binary (0/1) per-item per-run scores.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)`` clamped to [0, 1].
    """
    n, R = scores.shape
    cell_means = scores.mean(axis=1)
    p_hat = float(cell_means.mean())
    p_var = p_hat * (1.0 - p_hat)

    if n < 2 or p_var <= 0.0:
        return _wilson_neff(p_hat, float(n * R), alpha)

    s2     = float(np.var(cell_means, ddof=1))
    s2_min = p_var / R            # iid lower bound on Var(cell_mean)

    if s2 <= s2_min:
        # No detected overdispersion — treat as fully iid
        n_eff = float(n * R)
    elif s2 >= p_var:
        # Maximum overdispersion (kappa → 0) — one effective obs per item
        n_eff = float(n)
    else:
        # Method-of-moments: solve s² = p*(1-p)*(R + κ) / (R*(κ + 1)) for κ
        kappa = R * (p_var - s2) / (s2 * R - p_var)
        kappa = max(kappa, 1e-8)
        n_eff = n * R * (kappa + 1.0) / (R + kappa)

    return _wilson_neff(p_hat, n_eff, alpha)


def _paired_binary_cells(values_a, values_b, fname: str) -> tuple[int, int, int, int]:
    """Return the 2x2 paired-binary cell counts (n11, n10, n01, n00).

    Shared input handling for the paired binary interval methods. Values are
    thresholded at 0.5 (accommodates float representations).
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError(f"{fname} expects 1-D input arrays.")
    if values_a.shape != values_b.shape:
        raise ValueError(f"{fname} expects arrays with equal shape.")
    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)
    return (
        int(np.sum((a_bin == 1) & (b_bin == 1))),
        int(np.sum((a_bin == 1) & (b_bin == 0))),
        int(np.sum((a_bin == 0) & (b_bin == 1))),
        int(np.sum((a_bin == 0) & (b_bin == 0))),
    )


def bonett_price_paired_ci(
    values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float]:
    """Bonett-Price Laplace-adjusted Wald CI for the paired binary difference.

    Fagerland, Lydersen & Laake (2014) eq. (16) -- their *prime* recommendation
    for a CI on the difference between paired proportions, on the grounds that
    it is conservative, performs very well, and is trivial to compute.

    Applies a Laplace (add-one) adjustment to the discordant cells before
    forming a Wald interval::

        p12 = (n10 + 1) / (n + 2),  p21 = (n01 + 1) / (n + 2)
        (p12 - p21) +/- z * sqrt[ (p12 + p21 - (p12 - p21)^2) / (n + 2) ]

    Unlike the plain Wald interval it never produces a zero-width interval,
    since the add-one adjustment keeps the variance term strictly positive.
    Limits are truncated to [-1, 1].

    Reproduces Fagerland et al.'s Table V to the three decimals published.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1).
    """
    _, n10, n01, _ = _paired_binary_cells(values_a, values_b, "bonett_price_paired_ci")
    n = len(np.asarray(values_a))
    if n <= 0:
        return (0.0, 0.0)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p12 = (n10 + 1.0) / (n + 2.0)
    p21 = (n01 + 1.0) / (n + 2.0)
    diff = p12 - p21
    se = float(np.sqrt(max(p12 + p21 - diff * diff, 0.0) / (n + 2.0)))
    return (
        float(np.clip(diff - z * se, -1.0, 1.0)),
        float(np.clip(diff + z * se, -1.0, 1.0)),
    )



def newcombe_mover_paired_ci(
    values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe square-and-add (MOVER Wilson score) CI for the paired difference.

    Newcombe (1998) method 10, as presented in Fagerland, Lydersen & Laake
    (2014) eqs. (19)-(22) -- one of their three recommended intervals.

    This is the "square-and-add"/MOVER construction: separate Wilson score
    intervals are computed for the two *marginal* proportions p(A=1) and
    p(B=1), then combined with a correlation correction::

        L = d - sqrt[ (pA - l1)^2 + (u2 - pB)^2 - 2*phi*(pA - l1)*(u2 - pB) ]
        U = d + sqrt[ (pB - l2)^2 + (u1 - pA)^2 - 2*phi*(pB - l2)*(u1 - pA) ]

    where phi is estimated from A = n11*n00 - n10*n01 as (A - n/2)/sqrt(...)
    if A > n/2, 0 if 0 <= A <= n/2, and A/sqrt(...) if A < 0; phi is set to 0
    when any marginal sum is zero.

    This is the only Newcombe interval in evalstats. An earlier
    discordant-pairs formulation (a Wilson interval on n10/(n10+n01)
    rescaled to the difference scale) was removed on 2026-08-24: it is a
    different, poorly-covering method that is NOT the one Fagerland et al.
    recommend under the name "Newcombe".

    Reproduces Fagerland et al.'s Table V to the three decimals published.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1).
    """
    n11, n10, n01, n00 = _paired_binary_cells(
        values_a, values_b, "newcombe_mover_paired_ci"
    )
    n = n11 + n10 + n01 + n00
    if n <= 0:
        return (0.0, 0.0)
    n_a = n11 + n10          # successes for A (row margin)
    n_b = n11 + n01          # successes for B (column margin)
    l1, u1 = wilson_ci(n_a, n, alpha)
    l2, u2 = wilson_ci(n_b, n, alpha)
    p_a = n_a / n
    p_b = n_b / n

    margins = (n_a, n - n_a, n_b, n - n_b)
    if any(mg == 0 for mg in margins):
        phi = 0.0
    else:
        det = n11 * n00 - n10 * n01
        denom = float(np.sqrt(float(n_a) * (n - n_a) * n_b * (n - n_b)))
        if det > n / 2.0:
            phi = (det - n / 2.0) / denom
        elif det < 0.0:
            phi = det / denom
        else:
            phi = 0.0

    d = p_a - p_b
    lo_term = (p_a - l1) ** 2 + (u2 - p_b) ** 2 - 2.0 * phi * (p_a - l1) * (u2 - p_b)
    hi_term = (p_b - l2) ** 2 + (u1 - p_a) ** 2 - 2.0 * phi * (p_b - l2) * (u1 - p_a)
    return (
        float(np.clip(d - np.sqrt(max(lo_term, 0.0)), -1.0, 1.0)),
        float(np.clip(d + np.sqrt(max(hi_term, 0.0)), -1.0, 1.0)),
    )


def _clustered_paired_cells(values_a, values_b, fname):
    """Per-item 2x2 cell counts (a_k, b_k, c_k, d_k) for (n_items, n_runs) data.

    Maps the clustered matched-pair layout of Yang, Sun & Hardin (2012) onto an
    eval sweep: the ITEM is the cluster and each RUN is a unit within it, so
    cluster sizes are equal (n_k = R for every k). That equality matters --
    Eliasziw & Donner's n_c and Yang's differ for unequal clusters but both
    collapse to exactly R here, so the estimator is unambiguous for our design.
    """
    va = np.asarray(values_a)
    vb = np.asarray(values_b)
    if va.shape != vb.shape:
        raise ValueError(f"{fname} expects arrays with equal shape (n_items, n_runs).")
    if va.ndim != 2:
        raise ValueError(f"{fname} expects 2-D arrays (n_items, n_runs).")
    a_bin = (va >= 0.5).astype(int)
    b_bin = (vb >= 0.5).astype(int)
    a_k = np.sum((a_bin == 1) & (b_bin == 1), axis=1).astype(float)
    b_k = np.sum((a_bin == 1) & (b_bin == 0), axis=1).astype(float)
    c_k = np.sum((a_bin == 0) & (b_bin == 1), axis=1).astype(float)
    d_k = np.sum((a_bin == 0) & (b_bin == 0), axis=1).astype(float)
    return a_k, b_k, c_k, d_k


def _eliasziw_inflation_factor(a_k, b_k, c_k, d_k):
    """Variance inflation factor 1 + (n_c - 1) * rho_hat.

    Eliasziw & Donner (1991), as presented in Yang, Sun & Hardin (2012) sec 2.1:
    rho_tilde comes from an ANOVA decomposition into between- and within-cluster
    mean squares, then rho_hat rescales it using the discordant probabilities.
    With equal cluster sizes n_c = R exactly.

    Returns 1.0 (no inflation) for most degenerate cases, following the paper's
    Remark 1: if rho falls outside [-1, 1] or cannot be computed because only
    one type of discordant pair is present, the factor is set to 1.

    ONE DELIBERATE DEVIATION from Remark 1. When the ANOVA denominator is
    exactly 0 -- both mean squares vanish, so rho is literally 0/0 and the data
    carry NO information about within-cluster correlation -- this returns
    ``n_c`` (equivalently rho_hat = 1, full clustering) rather than 1.

    Remark 1's fallback of 1 asserts INDEPENDENCE of all ``N = K * n_c``
    units, which manufactures precision from absent data. That is harmless at
    the cluster sizes Yang et al. study (their example averages 2.4 units per
    cluster) but severe at the cluster sizes multi-run evals produce: on an
    all-concordant table with K = 15 items and n_c = 20 runs, the fallback of 1
    gives a 95% interval of width 0.025 -- claiming +/-1.3% precision from data
    containing no disagreements at all -- and the width shrinks further as runs
    are added. Returning ``n_c`` instead makes the interval reduce to the
    single-run score on ``K`` items, which is the correct answer when nothing
    discordant was observed, and makes its width invariant to the number of
    runs (verified: 0.40777 at n_c = 1, 3 and 20).

    This fires ONLY on the exact 0/0 branch; wherever rho is estimable the
    factor is bit-identical to Remark 1's. Measured effect on a 54-cell sweep:
    MinCov .6976 -> .9160 and cells below .93 coverage 6 -> 1, with cells where
    rho is estimable unchanged to the digit.
    """
    n_k = a_k + b_k + c_k + d_k
    K = len(n_k)
    N = float(n_k.sum())
    if K < 2 or N <= 0:
        return 1.0
    n_bar = N / K
    n_c = float((n_k ** 2).sum() / N)
    p = np.array([a_k.sum(), b_k.sum(), c_k.sum(), d_k.sum()], dtype=float) / N
    cells = np.stack([a_k, b_k, c_k, d_k], axis=1)
    expect = np.outer(n_k, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        bms = float((((cells - expect) ** 2).sum(axis=1) / n_k).sum() / K)
        wms_num = (cells * (n_k[:, None] - cells)).sum(axis=1) / n_k
    if n_bar <= 1.0:
        return 1.0
    wms = float(wms_num.sum() / (K * (n_bar - 1.0)))
    n_0 = n_bar - float(((n_k - n_bar) ** 2).sum()) / (K * (K - 1) * n_bar)
    denom = bms + (n_0 - 1.0) * wms
    if not np.isfinite(denom) or denom == 0.0:
        # rho is 0/0 -- no information about clustering. Assume full
        # clustering rather than independence; see the docstring's deviation
        # note. rho_hat = 1 gives factor = 1 + (n_c - 1) * 1 = n_c.
        return float(n_c) if np.isfinite(n_c) and n_c > 0 else 1.0
    rho_tilde = (bms - wms) / denom
    if not np.isfinite(rho_tilde) or rho_tilde <= 0.0:
        return 1.0
    q = (1.0 - rho_tilde) / rho_tilde
    rho_hat = 1.0 / (1.0 + p[1] * q + p[2] * q)
    if not np.isfinite(rho_hat) or not (-1.0 <= rho_hat <= 1.0):
        return 1.0
    factor = 1.0 + (n_c - 1.0) * rho_hat
    return float(factor) if np.isfinite(factor) and factor > 0 else 1.0


def clustered_score_paired_ci(
    values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float]:
    """Yang, Sun & Hardin (2012) score CI for clustered matched-pair binary data.

    Their X^2_Score: Tango's score statistic with the variance multiplied by the
    Eliasziw-Donner inflation factor, inverted by solving the same quartic used
    for the unclustered case. Concretely it is :func:`tango_scc_paired_ci` with
    ``z^2`` replaced by ``z^2 * (1 + (n_c - 1) * rho_hat)``, which is why no new
    solver is needed.

    Validated against Yang et al.'s published worked example (their Table II,
    PET/SPECT data): reproduces the reported CI (-0.03829, 0.29140) exactly, and
    reduces exactly to ``tango_scc_paired_ci(..., c=0)`` when the inflation
    factor is 1.

    Unlike our earlier effective-runs correction, the design effect here
    multiplies a variance built from POOLED run-level counts, which genuinely
    understates uncertainty under clustering -- the level at which the
    correction is meant to act.
    """
    a_k, b_k, c_k, d_k = _clustered_paired_cells(
        values_a, values_b, "clustered_score_paired_ci"
    )
    n_total = float((a_k + b_k + c_k + d_k).sum())
    if n_total <= 0:
        return (0.0, 0.0)
    z2 = float(stats.norm.ppf(1.0 - alpha / 2.0)) ** 2
    z2_eff = z2 * _eliasziw_inflation_factor(a_k, b_k, c_k, d_k)
    b_tot, c_tot = float(b_k.sum()), float(c_k.sum())
    upper = _tango_scc_real_roots_in_range(
        _tango_scc_quartic_coeffs(b_tot, c_tot, n_total, z2_eff, 0.0)
    )
    d_hat = (b_tot - c_tot) / n_total
    hi = float(upper[-1]) if len(upper) else d_hat
    lo = float(upper[0]) if len(upper) else d_hat
    lo, hi = (max(-1.0, min(lo, hi)), min(1.0, max(lo, hi)))
    if c_tot == 0.0 and b_tot == n_total:
        hi = 1.0
    elif b_tot == 0.0 and c_tot == n_total:
        lo = -1.0
    return (lo, hi)


def modified_obuchowski_paired_ci(
    values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float]:
    """Yang et al. (2010) modified-Obuchowski CI for clustered matched-pair data.

    As given in Yang, Sun & Hardin (2012) sec 2.4::

        (1/N) sum(b_k - c_k)  +/-  z * (1/N) * sqrt(
            K / (2 (K-1)) * sum[ ((b_k-c_k) - mean_k(b-c))^2
                               + ((b_k-c_k) - (n_k/N) sum(b-c))^2 ] )

    Cluster-level and assumption-free about the within-cluster correlation
    structure: no ICC is estimated at all. Yang et al. (2012) recommend this
    over Obuchowski's and Durkalski's variants on power grounds for larger
    numbers of clusters. Reproduces the reference R implementation
    (``clust.bin.pair``) exactly, but has no small-sample adjustment (at R=1
    it's bit-identical to the unregularised Wald interval on item differences,
    and returns a zero-width interval at zero discordance); kept as a citable
    negative result, not a recommended method (see
    ``simulations/harness/methods.py``).
    """
    a_k, b_k, c_k, d_k = _clustered_paired_cells(
        values_a, values_b, "modified_obuchowski_paired_ci"
    )
    n_k = a_k + b_k + c_k + d_k
    K = len(n_k)
    N = float(n_k.sum())
    if N <= 0:
        return (0.0, 0.0)
    s_k = b_k - c_k
    s_tot = float(s_k.sum())
    d_hat = s_tot / N
    if K < 2:
        return (max(-1.0, d_hat), min(1.0, d_hat))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    term = ((s_k - s_tot / K) ** 2 + (s_k - (n_k / N) * s_tot) ** 2).sum()
    var = (K / (2.0 * (K - 1.0))) * float(term)
    radius = z * np.sqrt(max(var, 0.0)) / N
    return (max(-1.0, d_hat - radius), min(1.0, d_hat + radius))


def _mj_discordance_floor(discordance_rate: float, floor: float = 0.25) -> float:
    """Floored discordance term for the May & Johnson score interval.

    The closed-form solution of the score inversion carries an additive
    ``z^2 * S_hat`` inside the discriminant, where ``S_hat`` is the observed
    discordance rate (n10+n01)/n. Left unfloored that term vanishes when no
    pairs disagree, collapsing the interval to zero width -- the degeneracy
    Tango's 2000 letter to the editor (Statist. Med. 19(1):133-139)
    criticised in the Quesenberry-Hurst / May & Johnson construction, along
    with its anticonservatism at low discordance.

    Flooring ``S_hat`` at 1/4 removes that failure while never SHRINKING the
    score interval's variance term, so the result is never narrower than the
    published interval. Measured effect (see
    simulations/papers/pairwise_binary_rerun_plan.md): at n=15 with 10%
    discordance, unfloored May & Johnson covers 0.719 at its worst over the
    true difference against a nominal 0.95 (0.787 at delta=0.04), while the
    floored interval stays at or above 0.987. NOTE the coverage gap is
    invisible exactly at delta=0, where the degenerate zero-width interval
    still "contains" a true difference of zero.
    On real eval corpora the floor lifts worst-case coverage
    from 0.721 to 0.789 at n=10 (single-run) and 0.902 to 0.925 at n=50
    (multi-run), while leaving low-asymmetry corpora untouched.
    """
    return max(float(discordance_rate), floor)


def mj_floor_paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
    floor: float = 0.25,
) -> tuple[float, float]:
    """Closed-form paired-binary CI for p(A=1) - p(B=1).

    This is NOT Tango (1998)'s own interval, despite the name it carries
    throughout the codebase and paper. Tango's interval inverts a score test
    through the constrained MLE and is solved iteratively (secant method),
    which is why it is not used as a fast default here.

    What this actually computes is a Wilson-regularized Quesenberry-Hurst-style
    interval. With m = n10 + n01 and d = n10 - n01::

        d / (n + z^2)  +/-  z/(n + z^2) * sqrt( m - d^2/n + z^2/4 )

    That is the centre of May & Johnson (1997), "Confidence intervals for
    differences in correlated binary proportions" (Statistics in Medicine
    16(18):2127-2136), equation 11 -- their adaptation of Quesenberry-Hurst --
    with their variance term ``z^2 * m / n`` replaced by the constant
    ``z^2 / 4`` from Wilson's one-sample score interval. Writing S_hat = m/n
    for the observed discordance rate, their additive term is ``z^2 * S_hat``
    and ours is ``z^2 / 4``, so they coincide exactly at S_hat = 1/4. Below
    that this interval is the WIDER of the two (conservative), above it the
    narrower. Paired eval comparisons sit well below 1/4 -- competing models
    agree on most items -- so the substitution is conservative in the regime
    it is used in.

    Note ``1/4`` is not a max-variance bound here: Var(A_i - B_i) at delta=0
    is S in [0, 1], so its maximum is 1, not 1/4. The constant amounts to
    imputing a fixed 25% discordance rate in a term of order z^2, which only
    matters when m is small.

    That substitution is deliberate: the published Quesenberry-Hurst form
    collapses to a ZERO-WIDTH interval when no pairs disagree (m = 0). Tango's
    2000 letter to the editor (Statist. Med. 19(1):133-139) criticised exactly
    that degeneracy, and the anticonservatism, of Quesenberry-Hurst. Sparse
    discordance is common in small eval sets, so the constant is what makes
    this usable here.

    Note also that inverting the score test for this variance function returns
    May & Johnson's interval exactly -- so this is NOT a score interval; it
    freezes the variance at the observed d_hat rather than solving at the
    hypothesised delta.

    Consequence worth knowing: like the other closed-form members of this
    family, this runs somewhat NARROWER than Tango's exact score interval --
    by ~0.005-0.018 in absolute width at n=100-200, and it stays finite at
    zero discordance where May-Johnson gives width 0. If you want the exact
    interval in closed form, call :func:`tango_scc_paired_ci` with ``c=0.0``;
    that implements Chang et al. (2024)'s quartic solution and agrees with a
    direct numerical inversion of the score equation to ~5e-4.

    Let:

    * ``n10`` be the count of pairs with ``A=1, B=0``
    * ``n01`` be the count of pairs with ``A=0, B=1``
    * ``n`` be the total number of pairs

    and ``d_hat = (n10 - n01) / n``. The interval is:

    ``(center +/- radius)`` where::

        center = d_hat / (1 + z^2 / n)
        radius = z / (1 + z^2 / n) * sqrt(
            (n10 + n01) / n^2
            - (n10 - n01)^2 / n^3
            + z^2 / (4 n^2)
        )

    This is a score-type interval for the paired risk difference; it
    remains non-degenerate even when there are no discordant pairs.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        1-D arrays of equal length. Values are thresholded at 0.5.
    alpha : float
        Significance level (1 - confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].

    Raises
    ------
    ValueError
        If inputs are not 1-D arrays of equal length.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("mj_floor_paired_ci expects 1-D input arrays.")
    if values_a.shape != values_b.shape:
        raise ValueError("mj_floor_paired_ci expects arrays with equal shape.")

    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)
    return mj_floor_paired_ci_from_diffs(a_bin - b_bin, alpha, floor)


def bonett_price_paired_ci_from_diffs(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """:func:`bonett_price_paired_ci` from a-minus-b diffs.

    The Bonett-Price interval depends on the raw pairs only through the two
    discordant counts and n (see that function: p12/p21 are built from n10,
    n01 and n), and ``diffs`` in ``{-1, 0, 1}`` determines all three -- so
    rebuilding a representative pair of binary arrays and delegating gives
    bit-identical output while keeping ONE copy of the formula.

    Exists so simultaneous-CI constructions (Sidak, joint-bootstrap scaling)
    can widen the SAME interval the non-simultaneous pairwise path reports
    for binary data, reusing each comparison's stored ``per_input_diffs``.
    Before this, the simultaneous path had no diffs-based Bonett-Price to
    call and widened ``mj_floor`` instead, so the simultaneous and pairwise
    CIs for binary data were built from two different formulas.
    """
    d = np.asarray(diffs).ravel()
    values_a = (d == 1).astype(float)
    values_b = (d == -1).astype(float)
    return bonett_price_paired_ci(values_a, values_b, alpha)


def _bonett_price_centre_scale_batch(diffs_2d: np.ndarray, alpha: float = 0.05):
    """Vectorized (centre, scale) for :func:`bonett_price_paired_ci_from_diffs`
    over a WHOLE matrix of resampled difference vectors at once.

    ``diffs_2d`` is ``(B, M)`` -- B resamples of the same pair's per-item
    differences. Returns ``(centre, scale)``, each shape ``(B,)``, such that
    the interval at level *a* is ``centre +/- z_{a/2} * scale`` (before the
    formula's clip to [-1, 1]).

    Exists for :func:`~evalstats.core.paired._calibrated_joint_critical_value`,
    whose calibration needs the construction's own centre and scale on every
    resample. Calling the scalar formula B times per pair is ~485x slower and,
    worse, recovers the scale as ``(hi - lo) / (2 z)`` -- which is WRONG
    whenever the interval clipped at +/-1, precisely the sparse small-n case
    the calibration is for. This computes both analytically from the counts,
    so it is exact and never sees the clip.

    Bonett-Price depends on the data only through ``(n10, n01, n)``, so the
    whole batch reduces to two count reductions along the item axis. The
    ``alpha`` argument is accepted (and ignored) because centre and scale are
    alpha-free for this Wald-form interval -- the signature matches the
    protocol so other formulas can supply an alpha-dependent version.
    """
    d = np.asarray(diffs_2d)
    n = d.shape[1]
    if n == 0:
        z = np.zeros(d.shape[0])
        return z, z
    n10 = (d == 1).sum(axis=1)
    n01 = (d == -1).sum(axis=1)
    p12 = (n10 + 1.0) / (n + 2.0)
    p21 = (n01 + 1.0) / (n + 2.0)
    centre = p12 - p21
    scale = np.sqrt(np.maximum(p12 + p21 - centre * centre, 0.0) / (n + 2.0))
    return centre, scale


def _alpha_crit_symmetric(target, centre, scale, sf):
    """Level at which a symmetric interval ``centre +/- q(alpha/2)*scale`` just
    covers *target*, given the reference distribution's survival function *sf*.

    Covering means ``|target-centre| <= q(alpha/2)*scale``; since ``q`` falls as
    alpha rises, the crossing level is ``2*sf(|target-centre|/scale)``.
    """
    centre = np.asarray(centre, dtype=float)
    scale = np.asarray(scale, dtype=float)
    out = np.ones(centre.shape, dtype=float)
    ok = np.isfinite(centre) & np.isfinite(scale) & (scale > 1e-12)
    if np.any(ok):
        z = np.abs(target - centre[ok]) / scale[ok]
        out[ok] = np.clip(2.0 * sf(z), 1e-12, 1.0)
    return out


def _bonett_price_alpha_crit_batch(diffs_2d, target):
    """alpha_crit for Bonett-Price -- symmetric on the difference scale, normal
    reference (see :func:`bonett_price_paired_ci_from_diffs`)."""
    centre, scale = _bonett_price_centre_scale_batch(diffs_2d)
    return _alpha_crit_symmetric(target, centre, scale, stats.norm.sf)


def _logit_t_alpha_crit_batch(values_2d, target, lo=0.0, hi=1.0):
    """alpha_crit for :func:`logit_t_ci_1d` (optionally through
    :func:`~evalstats.core.stats_utils.rescaled_ci` bounds *lo*/*hi*).

    logit-t is symmetric on the LOGIT scale with a t reference, not on the
    value scale -- so the crossing level is computed there and the target is
    mapped through the same transform. Degenerate rows (zero-variance
    resamples, which the scalar path hands to ``degenerate_sample_ci``) return
    1.0, i.e. they never bind the joint minimum, matching that path's skip.
    """
    span = float(hi - lo)
    v = (np.asarray(values_2d, dtype=float) - lo) / span
    y = (float(target) - lo) / span
    n = v.shape[1]
    out = np.ones(v.shape[0], dtype=float)
    if n <= 1:
        return out
    x_bar = v.mean(axis=1)
    se = v.std(axis=1, ddof=1) / np.sqrt(n)
    ok = (np.ptp(v, axis=1) > 0.0) & (se > 0.0) & np.isfinite(se) & (x_bar > 0.0) & (x_bar < 1.0)
    ok &= (y > 0.0) & (y < 1.0)
    if not np.any(ok):
        return out
    g = np.log(x_bar[ok] / (1.0 - x_bar[ok]))
    se_g = se[ok] / (x_bar[ok] * (1.0 - x_bar[ok]))
    z = np.abs(np.log(y / (1.0 - y)) - g) / se_g
    out[ok] = np.clip(2.0 * stats.t.sf(z, df=n - 1), 1e-12, 1.0)
    return out


def _nig_alpha_crit_batch(values_2d, target, b0=0.0625, m0=0.5, k0=1.0, a0=2.0, lo=0.0, hi=1.0):
    """alpha_crit for :func:`nig_ci_1d` -- symmetric on the (rescaled) value
    scale with a t reference at ``df = 2*a_n``. Mirrors that function's
    posterior update exactly; see it for the parameterization."""
    span = float(hi - lo)
    v = (np.asarray(values_2d, dtype=float) - lo) / span
    y = (float(target) - lo) / span
    n = v.shape[1]
    out = np.ones(v.shape[0], dtype=float)
    if n <= 0:
        return out
    xbar = v.mean(axis=1)
    ss = ((v - xbar[:, None]) ** 2).sum(axis=1)
    kn = k0 + n
    mn = (k0 * m0 + n * xbar) / kn
    an = a0 + n / 2.0
    bn = b0 + 0.5 * ss + (k0 * n * (xbar - m0) ** 2) / (2.0 * kn)
    scale = np.sqrt(np.maximum(bn / (an * kn), 0.0))
    return _alpha_crit_symmetric(y, mn, scale, lambda z: stats.t.sf(z, df=2.0 * an))


#: Optional fast path consumed by
#: evalstats.core.paired._calibrated_joint_critical_value. Formulas without
#: one fall back to per-resample scalar calls there.
bonett_price_paired_ci_from_diffs.centre_scale_batch = _bonett_price_centre_scale_batch
bonett_price_paired_ci_from_diffs.alpha_crit_batch = _bonett_price_alpha_crit_batch


def mj_floor_paired_ci_from_diffs(diffs: np.ndarray, alpha: float, floor: float = 0.25) -> tuple[float, float]:
    """Floored May & Johnson score CI for the paired binary difference,
    from a-minus-b diffs.

    Same closed-form score interval as :func:`mj_floor_paired_ci`, but takes
    the already-computed per-pair difference ``a_bin - b_bin`` (values in
    ``{-1, 0, 1}``) directly instead of the two raw ``values_a``/``values_b``
    arrays. Concordant pairs (``diff == 0``, whether both 1 or both 0) don't
    distinguish ``n10``/``n01`` counts either way, so ``diffs`` alone is
    sufficient -- this lets simultaneous-CI constructions (Sidak, joint-
    bootstrap scaling) reuse a paired comparison's already-stored
    ``per_input_diffs`` at an adjusted alpha without re-deriving the raw
    binary arrays.

    Parameters
    ----------
    diffs : np.ndarray
        1-D array of per-pair differences in ``{-1, 0, 1}`` (``a_bin - b_bin``).
    alpha : float
        Significance level (1 - confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].
    """
    diffs = np.asarray(diffs)
    n = int(len(diffs))
    if n <= 0:
        return (0.0, 0.0)

    n10 = int(np.sum(diffs > 0))
    n01 = int(np.sum(diffs < 0))

    d_hat = float((n10 - n01) / n)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n

    s_hat = _mj_discordance_floor((n10 + n01) / n, floor)
    radicand = (
        (n10 + n01) / (n * n)
        - ((n10 - n01) ** 2) / (n**3)
        + z2 * s_hat / (n * n)
    )
    radius = (z / denom) * float(np.sqrt(max(radicand, 0.0)))
    center = d_hat / denom

    lo = max(-1.0, float(center - radius))
    hi = min(1.0, float(center + radius))
    return (lo, hi)


def _tango_scc_quartic_coeffs(n12: float, n21: float, N: float, z2: float, correction: float) -> list[float]:
    """Quartic coefficients [a4..a0] for one branch (+c or -c) of the SCC interval.

    Directly implements Chang et al. (2024) Eqs. 4-5's G, H, I and a4..a0,
    with ``correction`` standing in for ``+c`` (upper-limit branch) or ``-c``
    (lower-limit branch, per their instruction to replace c with -c in H/I).
    """
    z4 = z2 * z2
    G = N * N + z2 * N
    H = 0.5 * z2 * (2.0 * N - n12 + n21) - 2.0 * N * (n12 - n21 + correction) - z2 * N
    I = (n12 - n21 + correction) ** 2 - 0.5 * z2 * (n12 + n21)

    a4 = G * G
    a3 = 2.0 * G * H
    a2 = H * H + 2.0 * G * I - 0.25 * z4 * ((2.0 * N - n12 + n21) ** 2 - 8.0 * N * n21)
    a1 = 2.0 * H * I - 0.25 * z4 * (8.0 * N * n21 - 2.0 * (n12 + n21) * (2.0 * N - n12 + n21))
    a0 = I * I - 0.25 * z4 * (n12 + n21) ** 2
    return [a4, a3, a2, a1, a0]


def _tango_scc_real_roots_in_range(coeffs: list[float]) -> np.ndarray:
    roots = np.roots(coeffs)
    roots = roots[np.abs(roots.imag) < 1e-8].real
    return np.sort(roots[(roots >= -1.0 - 1e-9) & (roots <= 1.0 + 1e-9)])


def mj_unfloored_paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """May & Johnson (1997) eq. 11 as published, with NO discordance floor.

    Provided as the comparison baseline that shows why the floor exists: this
    is the literal published interval, which degenerates to zero width when
    no pairs disagree and under-covers badly at low discordance (worst-case
    0.719 against a nominal 0.95 at n=15, S=0.10, over the true difference).
    Use :func:`mj_floor_paired_ci` in
    practice.
    """
    return mj_floor_paired_ci(values_a, values_b, alpha, floor=0.0)


def tango_scc_paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
    c: float = 0.125,
) -> tuple[float, float]:
    """Continuity-corrected Tango score CI ("SCC" interval) for p(A=1) - p(B=1).

    Implements the closed-form quartic solution from Chang, Liu, Hou, Yan &
    Shan (2024), "Continuity corrected score confidence interval for the
    difference in proportions in paired data" (J. Applied Statistics
    51(1):139-152, Eqs. 4-5), which adds a continuity correction ``c`` to
    Tango (1998)'s score test statistic and derives a non-iterative (Ferrari's
    method) solution rather than the secant-method iteration Tango's own
    interval traditionally uses.

    ``n12``/``n21`` follow the paper's Table 1 convention (row=test A,
    column=test B): ``n12`` = A-response/B-no-response (this function's
    ``n10``), ``n21`` = A-no-response/B-response (``n01``). Note: despite the
    paper's prose defining ``Delta = p21 - p12`` (i.e. ``p(B=1) - p(A=1)``),
    their actual working equations (4)-(5) solve for ``Delta = p12 - p21``
    (this function's ``p(A=1) - p(B=1)`` estimand) -- confirmed empirically
    (the roots bracket the observed point estimate only under that reading,
    never the stated one) rather than by further algebraic derivation, since
    reproducing their Eq. 2 (the constrained-MLE quadratic feeding into this
    quartic) from scratch turned out to disagree with what's printed and
    produced a degenerate (divide-by-zero) statistic at the point estimate in
    basic sanity checks. So this function's output needs no sign flip.

    ``c=0.125`` is the paper's recommended "SCC-S" (small-correction)
    variant -- found in their simulations to best balance coverage and width
    against the plain (uncorrected) Tango score interval
    (:func:`mj_floor_paired_ci`, a separate, simpler large-sample approximation
    that does not use this quartic's constrained-MLE derivation).
    ``c=0.25`` and ``c=0.5`` are their "SCC-M"/"SCC-L" variants.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        1-D arrays of equal length. Values are thresholded at 0.5.
    alpha : float
        Significance level (1 - confidence level).
    c : float
        Continuity correction (default 0.125, the paper's SCC-S).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].

    Raises
    ------
    ValueError
        If inputs are not 1-D arrays of equal length.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("tango_scc_paired_ci expects 1-D input arrays.")
    if values_a.shape != values_b.shape:
        raise ValueError("tango_scc_paired_ci expects arrays with equal shape.")

    n = int(len(values_a))
    if n <= 0:
        return (0.0, 0.0)

    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)
    n12 = float(np.sum((a_bin == 1) & (b_bin == 0)))
    n21 = float(np.sum((a_bin == 0) & (b_bin == 1)))
    N = float(n)
    d_hat = (n12 - n21) / N

    z2 = float(stats.norm.ppf(1.0 - alpha / 2.0)) ** 2

    upper_roots = _tango_scc_real_roots_in_range(_tango_scc_quartic_coeffs(n12, n21, N, z2, c))
    lower_roots = _tango_scc_real_roots_in_range(_tango_scc_quartic_coeffs(n12, n21, N, z2, -c))

    ci_high = float(upper_roots[-1]) if len(upper_roots) else d_hat
    ci_low = float(lower_roots[0]) if len(lower_roots) else d_hat

    ci_low, ci_high = (max(-1.0, min(ci_low, ci_high)), min(1.0, max(ci_low, ci_high)))

    # Yang, Sun & Hardin (2012) Remark 1. When every pair is discordant in
    # the same direction the score statistic is 0/0 at delta = +/-1, so the
    # quartic loses the corresponding root and the interval comes back
    # EXCLUDING the point estimate d_hat = +/-1. Tango's interval is defined
    # to take the boundary there. Without this the interval is also
    # asymmetric under swapping a and b, since the root is recovered in one
    # orientation but not the other. Matches Fagerland et al.'s reference
    # implementation (R package contingencytables).
    if n21 == 0.0 and n12 == N:
        ci_high = 1.0
    elif n12 == 0.0 and n21 == N:
        ci_low = -1.0

    return (ci_low, ci_high)


def mj_floor_paired_ci_multirun_cluster(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Cluster-robust Tango-style CI for paired binary difference.

    This is what ``pairwise_differences(method='mj_floor')`` dispatches to for
    multi-run data (see :func:`~evalstats.core.paired.pairwise_differences`).
    NOTE: this method is not Tango's own interval despite the name it carried
    before 2026-08-24 (see :func:`mj_floor_paired_ci`).

    The additive discordance term in the discriminant is floored at 1/4 (see
    :func:`_mj_discordance_floor`); this is the multi-run analogue of the
    single-run floor in :func:`mj_floor_paired_ci`, using the mean per-item
    discordance mass as S_hat.

    Treats each item as the unit of analysis, using the variance of per-item
    paired differences directly rather than a within/between decomposition.

    Reduces exactly to mj_floor_paired_ci when n_runs == 1.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if values_a.shape != values_b.shape:
        raise ValueError("Inputs must have equal shape (n_items, n_runs).")
    if values_a.ndim != 2:
        raise ValueError("Expected 2-D arrays (n_items, n_runs).")

    n_items, n_runs = values_a.shape
    if n_items <= 0:
        return (0.0, 0.0)

    if n_runs == 1:
        return mj_floor_paired_ci(values_a[:, 0], values_b[:, 0], alpha)

    # --- binarize ---
    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)

    # --- per-item discordance rates ---
    d10_i = np.mean((a_bin == 1) & (b_bin == 0), axis=1)
    d01_i = np.mean((a_bin == 0) & (b_bin == 1), axis=1)

    delta_i = d10_i - d01_i

    # --- point estimate ---
    d_hat = float(np.mean(delta_i))

    # --- cluster variance ---
    if n_items > 1:
        var_delta = float(np.var(delta_i, ddof=1))
    else:
        var_delta = 0.0

    # --- Tango shrinkage ---
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n_items

    # --- variance (cluster-robust) ---
    s_hat = _mj_discordance_floor(float(np.mean(d10_i + d01_i)))
    radicand = var_delta / n_items + z2 * s_hat / (n_items * n_items)

    radius = (z / denom) * float(np.sqrt(max(radicand, 0.0)))
    center = d_hat / denom

    lo = max(-1.0, float(center - radius))
    hi = min(1.0, float(center + radius))

    return (lo, hi)


def mj_floor_paired_ci_multirun_effective(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Correlation-aware multi-run score CI using effective sample size.

    The additive discordance term in the discriminant is floored at 1/4 (see
    :func:`_mj_discordance_floor`); this is the multi-run analogue of the
    single-run floor in :func:`mj_floor_paired_ci`, using the mean per-item
    discordance mass as S_hat. NOTE: this method is NOT Tango's interval
    despite the name it carried before 2026-08-24.

    Not the method ``pairwise_differences(method='mj_floor')`` dispatches to for
    multi-run data -- that's :func:`mj_floor_paired_ci_multirun_cluster`. This
    variant remains available as an alternative/comparison point (see
    ``simulations/harness``), not as a routed default.

    Adjusts within-item variance using an estimated effective number of runs
    to account for correlation between runs.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if values_a.shape != values_b.shape:
        raise ValueError("Inputs must have equal shape (n_items, n_runs).")
    if values_a.ndim != 2:
        raise ValueError("Expected 2-D arrays (n_items, n_runs).")

    n_items, n_runs = values_a.shape
    if n_items <= 0:
        return (0.0, 0.0)

    if n_runs == 1:
        return mj_floor_paired_ci(values_a[:, 0], values_b[:, 0], alpha)

    # --- binarize ---
    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)

    # --- per-item discordance ---
    d10_i = np.mean((a_bin == 1) & (b_bin == 0), axis=1)
    d01_i = np.mean((a_bin == 0) & (b_bin == 1), axis=1)

    delta_i = d10_i - d01_i
    u_i = d10_i + d01_i

    d_hat = float(np.mean(delta_i))

    if n_items > 1:
        var_delta = float(np.var(delta_i, ddof=1))
    else:
        var_delta = 0.0

    # --- within variance ---
    within_i = np.maximum(u_i - delta_i * delta_i, 0.0)
    within_bar = float(np.mean(within_i))

    # --- estimate effective number of runs ---
    # rho ≈ fraction of variance attributable to shared signal
    if var_delta > 0:
        rho = max(0.0, min(1.0, 1.0 - (within_bar / (var_delta * n_runs + 1e-12))))
    else:
        rho = 0.0

    R_eff = n_runs / (1.0 + (n_runs - 1.0) * rho)

    # --- adjusted variance decomposition ---
    between_latent = max(var_delta - within_bar / R_eff, 0.0)

    between = between_latent / n_items
    within = within_bar / (n_items * R_eff)

    # --- Tango shrinkage ---
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n_items

    s_hat = _mj_discordance_floor(float(np.mean(u_i)))
    radicand = between + within + z2 * s_hat / (n_items * n_items)

    radius = (z / denom) * float(np.sqrt(max(radicand, 0.0)))
    center = d_hat / denom

    lo = max(-1.0, float(center - radius))
    hi = min(1.0, float(center + radius))

    return (lo, hi)


def mj_floor_paired_ci_multirun_moments(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Multi-run Tango-style CI using a cluster moments decomposition.

    The additive discordance term in the discriminant is floored at 1/4 (see
    :func:`_mj_discordance_floor`); this is the multi-run analogue of the
    single-run floor in :func:`mj_floor_paired_ci`, using the mean per-item
    discordance mass as S_hat. NOTE: this method is NOT Tango's interval
    despite the name it carried before 2026-08-24.

    Not the method ``pairwise_differences(method='mj_floor')`` dispatches to
    for multi-run data -- that's :func:`mj_floor_paired_ci_multirun_cluster`.
    This variant remains available as an alternative/comparison point (see
    ``simulations/harness``), not as a routed default.

    This variant estimates the paired risk-difference uncertainty via
    item-level moments of paired run differences:

    - between-item term: ``Var(delta_i) / n_items``
    - within-item term:  ``E[u_i - delta_i^2] / (n_items * n_runs)``

    where ``delta_i`` is the per-item mean paired difference across runs and
    ``u_i`` is the per-item discordance mass. It remains score-shrunk using
    the same denominator and reverts exactly to :func:`mj_floor_paired_ci` when
    ``n_runs == 1``.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Arrays of shape (n_items, n_runs). Values thresholded at 0.5.
        Runs are assumed to be paired across A and B.
    alpha : float
        Significance level (1 - confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if values_a.shape != values_b.shape:
        raise ValueError("Inputs must have equal shape (n_items, n_runs).")
    if values_a.ndim != 2:
        raise ValueError("Expected 2-D arrays (n_items, n_runs).")

    n_items, n_runs = values_a.shape
    if n_items <= 0:
        return (0.0, 0.0)

    # Exact reduction to the original paired Tango interval for single-run data.
    if n_runs == 1:
        return mj_floor_paired_ci(values_a[:, 0], values_b[:, 0], alpha)

    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)

    d10_i = np.mean((a_bin == 1) & (b_bin == 0), axis=1)
    d01_i = np.mean((a_bin == 0) & (b_bin == 1), axis=1)

    delta_i = d10_i - d01_i
    u_i = d10_i + d01_i

    d_hat = float(np.mean(delta_i))

    if n_items > 1:
        var_delta = float(np.var(delta_i, ddof=1))
    else:
        var_delta = 0.0

    within_i = np.maximum(u_i - delta_i * delta_i, 0.0)
    within_bar = float(np.mean(within_i))

    # Decompose observed item-level variance into latent between-item variance
    # plus within-item Monte Carlo noise; this avoids adding within variance twice.
    between_latent = max(var_delta - within_bar / n_runs, 0.0)
    between = between_latent / n_items
    within = within_bar / (n_items * n_runs)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n_items

    s_hat = _mj_discordance_floor(float(np.mean(u_i)))
    radicand = between + within + z2 * s_hat / (n_items * n_items)

    radius = (z / denom) * float(np.sqrt(max(radicand, 0.0)))
    center = d_hat / denom

    lo = max(-1.0, float(center - radius))
    hi = min(1.0, float(center + radius))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Multi-run Bonett-Price
#
# THE DERIVATION, once, so the three variants below can just cite it.
#
# Write D_i = A_i - B_i in {-1, 0, +1} for the single-run per-item difference.
# Then sum_i D_i = n10 - n01 and sum_i D_i^2 = n10 + n01 (squaring a value in
# {-1,0,1} is the same as taking its absolute value), so the published
# Bonett & Price (2012) limits
#
#     p12 = (n10 + 1)/(n + 2),  p21 = (n01 + 1)/(n + 2)
#     (p12 - p21) +/- z * sqrt[ (p12 + p21 - (p12 - p21)^2) / (n + 2) ]
#
# can be rewritten with no reference to the 2x2 table at all:
#
#     p12 - p21           = (sum_i D_i) / (n + 2)
#     p12 + p21           = (sum_i D_i^2 + 2) / (n + 2)
#     variance term       = mean(D^2) - mean(D)^2, both means over n + 2
#
# i.e. Bonett-Price is the plain Wald interval on the mean of D, computed on
# the sample augmented by two pseudo-items, one with D = +1 and one with
# D = -1 -- with the ddof=0 plug-in variance and the divisor n + 2 used
# consistently for the mean, the variance and the standard error (verified
# numerically against :func:`bonett_price_paired_ci` to 2e-16 over a grid of
# n and alpha; see tests/test_bonett_price_multirun.py). The Laplace
# adjustment is the two pseudo-items: they cancel in the numerator of the
# point estimate, shrinking it toward 0 by n/(n+2), and they contribute 2 to
# the second moment, which is what keeps the variance term strictly positive
# when no pairs disagree.
#
# This reading settles whether the pseudo-counts should be scaled by the
# number of runs R -- they should not. The pseudo-observations are ITEMS,
# not runs: two extra items, each perfectly concordant across all R of its
# own runs (delta = +1 and -1, so within-item variance 0). Scaling them to R
# pseudo-runs (pseudo-items with delta = +-1/R) makes the whole
# regularisation vanish as R grows, collapsing the interval toward zero
# width at zero observed discordance -- exactly the degeneracy the Laplace
# adjustment exists to prevent. Item-level heterogeneity is bounded by N,
# not by N*R: more runs per item tell you nothing about items never sampled.
# See tests/test_bonett_price_multirun.py's
# ``test_per_run_laplace_scaling_degenerates`` for the regression guard.
#
# So, for (N, R) data, work at the item scale throughout:
#
#     delta_i = mean_r (A_ir - B_ir)   in [-1, 1]   per-item mean difference
#     u_i     = mean_r |A_ir - B_ir|   in [0, 1]    per-item discordance mass
#     w_i     = u_i - delta_i^2        >= 0         per-item within-run variance
#
# augment with delta = +1 and delta = -1 (both with u = 1, hence w = 0), and
# the three augmented moments are
#
#     delta~ = (sum_i delta_i) / (N + 2)
#     m2~    = (sum_i delta_i^2 + 2) / (N + 2)
#     V~     = m2~ - delta~^2          item-level total variance
#     w~     = (sum_i w_i) / (N + 2)   mean within-item variance
#
# with the interval delta~ +/- z * sqrt(V~ / (N + 2)). At R = 1 every
# delta_i is in {-1,0,1} so delta_i^2 = u_i, giving m2~ = p12 + p21 and
# w_i = 0 -- every variant below reduces to :func:`bonett_price_paired_ci`
# EXACTLY, not just asymptotically, and no special-casing of R == 1 is
# needed anywhere.
#
# WHY NO EXPLICIT BETWEEN-RUN CORRELATION TERM. V~ is already the right
# quantity: items are the sampling unit and are iid, and whatever
# correlation the R runs of item i have with each other only affects
# Var(delta_i), which is exactly what the item-level spread measures.
# Concretely, under the usual one-way random-effects decomposition
#
#     Var(delta_i) = sigma_B^2 + sigma_W^2/R * (1 + (R-1)*rho)
#
# the design-effect factor is *inside* the quantity being estimated, so
# estimating Var(delta_i) directly needs no rho at all: applying Kish's
# R_eff = R/(1 + (R-1)*rho) explicitly -- pool all N*R runs for a run-level
# variance B~ = u~ - delta~^2, estimate the run-level ICC
# rho = 1 - sigma_W^2/B~ with the unbiased within estimate
# sigma_W^2 = R/(R-1) * w~, then inflate by the design effect -- gives
# B~ * (1 + (R-1)*rho) / R == V~ identically, for every input (verified in
# tests/test_bonett_price_multirun.py). The design-effect correction and the
# item-level variance are the same estimator written two ways.
#
# So the three variants below differ ONLY in a floor applied to V~, mirroring
# what mj_floor's three multi-run variants turn out to differ by (their
# ``max(var - w/R', 0) + w/R'`` construction is algebraically
# ``max(var, w/R')``):
#
#     cluster    V~                      no floor -- the derivation as-is
#     moments    max(V~, w~/R)           floor at the within-item term alone
#     effective  max(V~, w~/R_eff)       same, with Kish's R_eff <= R
#
# ---------------------------------------------------------------------------


def _bp_item_moments(
    values_a: np.ndarray, values_b: np.ndarray, fname: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item ``(delta_i, u_i)`` from ``(n_items, n_runs)`` binary matrices.

    ``delta_i`` is the per-item mean of ``A_ir - B_ir`` and ``u_i`` the
    per-item mean of ``|A_ir - B_ir|`` (the discordance mass). Values are
    thresholded at 0.5. See the derivation block above.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.shape != b.shape:
        raise ValueError(f"{fname} expects arrays with equal shape (n_items, n_runs).")
    if a.ndim != 2:
        raise ValueError(f"{fname} expects 2-D arrays (n_items, n_runs).")
    if a.shape[1] < 1:
        # Caught explicitly: the per-item means below would be all-NaN and the
        # w~/R floors would divide by zero, so an item with no runs at all
        # would surface as a ZeroDivisionError from deep inside the variance
        # rather than as the input error it is.
        raise ValueError(f"{fname} expects at least one run per item.")
    d = (a >= 0.5).astype(np.int8) - (b >= 0.5).astype(np.int8)
    return np.mean(d, axis=1, dtype=float), np.mean(np.abs(d), axis=1, dtype=float)


def _bonett_price_augmented_interval(
    delta_i: np.ndarray, alpha: float, var_floor: float = 0.0,
    pseudo_m2: float = 1.0,
) -> tuple[float, float]:
    """Wald interval on ``mean(delta_i)`` over the ``+/-1``-augmented item sample.

    The shared core of every Bonett-Price variant in this module, single-run
    included: see the derivation block above for why the two pseudo-items are
    the Laplace adjustment. ``var_floor`` is a lower bound on the augmented
    item-level variance ``V~``, used by the ``moments`` and ``effective``
    variants; leave it at 0 for the plain (``cluster``) interval.
    """
    n = int(np.asarray(delta_i).shape[0])
    if n <= 0:
        return (0.0, 0.0)
    delta_i = np.asarray(delta_i, dtype=float)
    n_aug = n + 2.0
    delta_t = float(np.sum(delta_i)) / n_aug        # pseudo-items cancel: +1 - 1 = 0
    m2_t = (float(np.sum(delta_i * delta_i)) + 2.0 * pseudo_m2) / n_aug  # pseudo-items add m2 each
    var_t = max(m2_t - delta_t * delta_t, float(var_floor), 0.0)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = float(np.sqrt(var_t / n_aug))
    return (
        float(np.clip(delta_t - z * se, -1.0, 1.0)),
        float(np.clip(delta_t + z * se, -1.0, 1.0)),
    )


def bonett_price_paired_ci_multirun_cluster(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Multi-run Bonett-Price CI, item-clustered -- the derivation with no floor.

    The Bonett-Price counterpart of
    :func:`mj_floor_paired_ci_multirun_cluster`, and the most defensible of
    the three: it is the single-run interval's own construction carried over
    unchanged, with the item as the unit of analysis and no extra modelling.

        delta~ = (sum_i delta_i) / (N + 2)
        V~     = (sum_i delta_i^2 + 2) / (N + 2) - delta~^2
        CI     = delta~ +/- z * sqrt( V~ / (N + 2) )

    where ``delta_i = mean_r (A_ir - B_ir)``. See the derivation block above
    the private helpers in this module for the full argument, in particular
    for why the ``+1/+2`` pseudo-counts stay on the ITEM scale and why no
    between-run correlation term is needed (``V~`` already contains it, and
    a correctly-specified Kish design effect provably reduces to ``V~``).

    Reduces to :func:`bonett_price_paired_ci` EXACTLY at ``n_runs == 1``, by
    construction rather than by a special case: at R = 1 each ``delta_i`` is
    in ``{-1, 0, 1}``, so ``sum_i delta_i = n10 - n01`` and
    ``sum_i delta_i^2 = n10 + n01``.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Arrays of shape ``(n_items, n_runs)``, thresholded at 0.5. Runs are
        assumed paired across A and B.
    alpha : float
        Significance level (1 - confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].
    """
    delta_i, _ = _bp_item_moments(
        values_a, values_b, "bonett_price_paired_ci_multirun_cluster"
    )
    return _bonett_price_augmented_interval(delta_i, alpha)


def bonett_price_paired_ci_multirun_shrunk(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Multi-run Bonett-Price with the pseudo-item magnitude Laplace-shrunk.
    The shipped default for multi-run Bonett-Price data (see
    :func:`~evalstats.core.paired.pairwise_differences`).

    :func:`bonett_price_paired_ci_multirun_cluster` pins its two pseudo-items
    at ``delta = +/-1``, the largest possible item-level discordance. At
    ``R = 1`` that's exactly right (every discordant item has
    ``delta_i^2 = 1``), but at ``R > 1`` a discordant item's ``delta_i^2``
    shrinks toward its squared per-item rate while the pseudo-mass stays at
    2, so the pseudo-items become disproportionately heavy and the floor
    overcorrects.

    This variant shrinks the pseudo-item magnitude the same way Bonett-Price
    already shrinks the discordance rate::

        m2 = (sum_i delta_i^2 + 2) / (sum_i u_i + 2),   u_i = mean_r |A_ir - B_ir|

    Each pseudo-item then carries ``delta^2 = m2`` (as ``+/-sqrt(m2)``, so
    they still cancel in the mean). ``sum_i u_i`` is the effective count of
    fully-discordant items, so an item discordant on 1 of 20 runs contributes
    0.05 rather than 1 -- a plain count of discordant items instead collapses
    ``m2`` toward 0 when items flip sign across runs, undoing the correction.

    ``m2`` is a shrinkage estimator: ``m2 = w * (sum delta_i^2 / sum u_i) +
    (1 - w) * 1`` with ``w = sum u_i / (sum u_i + 2)``, i.e. the observed
    mean squared discordance magnitude shrunk toward the R=1 reference value
    of 1. (An earlier variant used the data term alone, ``w = 1``: it
    undercovered badly wherever discordance was sparse, since the quantity
    it shrinks toward zero has nothing holding it up.)

    Properties: reduces to :func:`bonett_price_paired_ci` bit-for-bit at
    R=1 (``m2 = 1`` exactly, since ``delta_i^2 = u_i`` there); ``m2 = 1`` at
    zero discordance too, matching the ``+/-1`` construction; replication-
    invariant (R identical copies of one run leave the interval unchanged);
    antisymmetric under swapping the two arms.

    On a 300-cell calibration sweep (5 discordance shapes x n in 20..100 x
    R in 2..20 x five run-consistency mixtures), this is narrower on
    average (94% of the ``+/-1`` construction's width) but with a slightly
    worse worst case (MinCov .9296 vs .9444 for
    :func:`bonett_price_paired_ci_multirun_cluster`)."""
    delta_i, u_i = _bp_item_moments(
        values_a, values_b, "bonett_price_paired_ci_multirun_shrunk"
    )
    if delta_i.shape[0] == 0:
        return (0.0, 0.0)
    sq_sum = float(np.sum(delta_i * delta_i))
    u_sum = float(np.sum(u_i))
    m2 = (sq_sum + 2.0) / (u_sum + 2.0)
    return _bonett_price_augmented_interval(delta_i, alpha, pseudo_m2=m2)


def resolve_resampling_method(
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto"],
    sample_size: int,
) -> Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"]:
    """Resolve ``method='auto'`` to a concrete bootstrap method.

    ``method='auto'`` resolves to ``'bootstrap'`` when
    ``sample_size >= config.BOOTSTRAP_AUTO_MIN_N`` (plain bootstrap is simpler
    and at least as accurate at that scale) and ``'bootstrap_t'`` otherwise.
    ``'bayes_bootstrap'`` and ``'smooth_bootstrap'`` are passed through unchanged.
    """
    if method == "auto":
        return "bootstrap" if sample_size >= BOOTSTRAP_AUTO_MIN_N else "bootstrap_t"
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
    boot_stats = np.empty(n_bootstrap, dtype=float)

    chunk_size = max(1, min(n_bootstrap, 4096, max(1, int(1_000_000 // max(m, 1)))))

    start = 0
    while start < n_bootstrap:
        stop = min(start + chunk_size, n_bootstrap)
        idx = rng.integers(0, m, size=(stop - start, m))
        samples = values[idx]
        if statistic == "median":
            boot_stats[start:stop] = np.median(samples, axis=1)
        else:
            boot_stats[start:stop] = samples.mean(axis=1)
        start = stop

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


def bootstrap_t_ci_1d(
    values: np.ndarray,
    observed_stat: float,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Bootstrap-t (studentized bootstrap) CI for a 1-D array.

    Inverts the distribution of the studentized pivot
    ``t* = (θ̂* − θ̂) / SE*`` rather than ``θ̂*`` directly, giving
    second-order accuracy — the same theoretical order as BCa but via a
    different route.  The CI is::

        [θ̂ − t*_{1−α/2} · SE,  θ̂ − t*_{α/2} · SE]

    For ``statistic='mean'``, SE = ``std(sample) / sqrt(n)`` (both observed
    and bootstrap).

    For ``statistic='median'``, this routine falls back to percentile bootstrap and emits a
    warning.  A proper median bootstrap-t requires a replicate-wise median SE
    estimator, which is not implemented here.

    Falls back to the plain percentile bootstrap when the observed SE is zero
    (degenerate sample), when no stable studentized pivots are available, or
    when too many bootstrap replicates have numerically tiny ``SE*`` values.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    observed_stat : float
        The statistic computed on the original sample (mean or median).
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (1 − confidence level).
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    if statistic == "median":
        warnings.warn(
            "bootstrap_t_ci_1d: bootstrap-t studentization is implemented for "
            "'mean'; falling back to percentile bootstrap for 'median'.",
            UserWarning,
            stacklevel=3,
        )
        boot_stats = bootstrap_means_1d(values, n_bootstrap, rng, statistic="median")
        return _percentile_interval(boot_stats, alpha)

    n = len(values)

    # ── Generate bootstrap samples ────────────────────────────────────────
    chunk_size = max(1, min(n_bootstrap, 4096, max(1, int(1_000_000 // max(n, 1)))))
    boot_stats = np.empty(n_bootstrap, dtype=float)
    boot_ses   = np.empty(n_bootstrap, dtype=float)

    start = 0
    while start < n_bootstrap:
        stop = min(start + chunk_size, n_bootstrap)
        idx = rng.integers(0, n, size=(stop - start, n))
        samples = values[idx]                                   # (chunk, n)
        boot_stats[start:stop] = samples.mean(axis=1)
        boot_ses[start:stop] = np.std(samples, ddof=1, axis=1) / np.sqrt(n)
        start = stop

    # ── Observed SE ───────────────────────────────────────────────────────
    se_obs = float(np.std(values, ddof=1)) / np.sqrt(n)

    if se_obs <= 0.0 or not np.isfinite(se_obs):
        return _percentile_interval(boot_stats, alpha)

    # ── Studentized pivots ────────────────────────────────────────────────
    valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
    if not np.any(valid):
        return _percentile_interval(boot_stats, alpha)
    se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
    tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
    if tiny_frac > 0.05:
        return _percentile_interval(boot_stats, alpha)
    valid = valid & (boot_ses >= se_floor)
    if not np.any(valid):
        return _percentile_interval(boot_stats, alpha)
    t_stats = (boot_stats[valid] - observed_stat) / boot_ses[valid]

    t_lo = float(np.percentile(t_stats, 100.0 * alpha / 2))
    t_hi = float(np.percentile(t_stats, 100.0 * (1.0 - alpha / 2)))
    return (
        float(observed_stat - t_hi * se_obs),
        float(observed_stat - t_lo * se_obs),
    )


def bootstrap_ci_1d(
    values: np.ndarray,
    observed_stat: float,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Bootstrap, BCa, Bayesian bootstrap, smoothed bootstrap, or bootstrap-t CI for a 1-D array.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    if method == "bootstrap_t":
        return bootstrap_t_ci_1d(values, observed_stat, n_bootstrap, alpha, rng, statistic=statistic)
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


def bootstrap_means_nested(
    scores: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Nested bootstrap replicates of the statistic for single-sample multi-run data.

    Outer level resamples M inputs with replacement; inner level resamples R
    runs within each selected input.  Propagates both input-sampling uncertainty
    and within-cell run variance.  Reduces to a standard percentile bootstrap
    when ``R = 1``.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(M, R)`` — per-input per-run scores.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    M, R = scores.shape
    input_idx = rng.integers(0, M, size=(n_bootstrap, M))           # (B, M)
    run_idx   = rng.integers(0, R, size=(n_bootstrap, M, R))        # (B, M, R)

    selected   = scores[input_idx]                                   # (B, M, R)
    b_rng = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]       # (B, 1, 1)
    m_rng = np.arange(M)[np.newaxis, :, np.newaxis]                 # (1, M, 1)
    resampled  = selected[b_rng, m_rng, run_idx]                    # (B, M, R)
    cell_means = resampled.mean(axis=2)                              # (B, M)
    return _reduce_rows(cell_means, statistic)                       # (B,)


def bayes_bootstrap_means_nested(
    scores: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bayesian nested bootstrap replicates for single-sample multi-run data.

    Outer level assigns Dirichlet(1,...,1_M) weights over M inputs; inner
    level resamples R runs uniformly within each input.  Dirichlet outer
    weights give smoother distributions than multinomial resampling at small M.
    Reduces to a standard Bayesian bootstrap when ``R = 1``.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(M, R)``.
    n_bootstrap : int
    rng : np.random.Generator
    statistic : str

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    M, R = scores.shape
    run_idx    = rng.integers(0, R, size=(n_bootstrap, M, R))       # (B, M, R)
    m_rng      = np.arange(M)[np.newaxis, :, np.newaxis]            # (1, M, 1)
    resampled  = scores[m_rng, run_idx]                             # (B, M, R)
    cell_means = resampled.mean(axis=2)                             # (B, M)

    exp_mat       = rng.exponential(1.0, size=(n_bootstrap, M))     # (B, M)
    outer_weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)    # (B, M)

    if statistic == "mean":
        return (outer_weights * cell_means).sum(axis=1)             # (B,)
    return _weighted_medians_rows(cell_means, outer_weights)


def smooth_bootstrap_means_nested(
    scores: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Smoothed nested bootstrap replicates for single-sample multi-run data.

    Outer level resamples M inputs with replacement; inner level resamples R
    runs within each selected input.  Gaussian noise with std = KDE bandwidth
    is then added to each resampled cell mean.  Falls back to
    ``bootstrap_means_nested`` if ``std(cell_means) == 0`` or ``M < 2``.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(M, R)`` — per-input per-run scores.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    from scipy.stats import gaussian_kde

    M, R = scores.shape
    cell_means_obs = scores.mean(axis=1)                            # (M,)
    std_val = float(np.std(cell_means_obs, ddof=1)) if M > 1 else 0.0
    if M < 2 or not np.isfinite(std_val) or std_val <= 0.0:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_means_nested",
            f"M={M}, std(cell_means)={std_val:.6g}",
        )
        return bootstrap_means_nested(scores, n_bootstrap, rng, statistic=statistic)

    try:
        h = float(gaussian_kde(cell_means_obs).factor * std_val)
    except np.linalg.LinAlgError as exc:
        _warn_smooth_bootstrap_fallback(
            "smooth_bootstrap_means_nested",
            f"KDE failed with {exc.__class__.__name__}: {exc}",
        )
        return bootstrap_means_nested(scores, n_bootstrap, rng, statistic=statistic)

    input_idx = rng.integers(0, M, size=(n_bootstrap, M))           # (B, M)
    run_idx   = rng.integers(0, R, size=(n_bootstrap, M, R))        # (B, M, R)

    selected        = scores[input_idx]                              # (B, M, R)
    b_rng           = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]   # (B, 1, 1)
    m_rng           = np.arange(M)[np.newaxis, :, np.newaxis]             # (1, M, 1)
    resampled       = selected[b_rng, m_rng, run_idx]               # (B, M, R)
    cell_means_boot = resampled.mean(axis=2)                        # (B, M)
    cell_means_boot += rng.normal(0.0, h, size=(n_bootstrap, M))
    return _reduce_rows(cell_means_boot, statistic)                  # (B,)


def bootstrap_t_ci_nested(
    scores: np.ndarray,
    observed_stat: float,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap-t (studentized) CI for single-sample multi-run data using nested resampling.

    Outer level resamples M inputs with replacement; inner level resamples R
    runs within each selected input.  The studentized pivot is

        t* = (θ̂* − θ̂) / SE*

    where SE* = std(resampled cell means) / sqrt(M) for each replicate and
    SE_obs uses the original cell means.  The CI is::

        [θ̂ − t*_{1−α/2} · SE_obs,  θ̂ − t*_{α/2} · SE_obs]

    Falls back to percentile interval from ``bootstrap_means_nested`` when
    ``SE_obs`` is zero, no stable pivots are produced, or studentization is
    numerically unstable due to too many tiny ``SE*`` replicates.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(M, R)`` — per-input per-run scores.
    observed_stat : float
        Grand mean of the observed cell means.
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level.
    rng : np.random.Generator

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)``.
    """
    M, R = scores.shape
    cell_means_obs = scores.mean(axis=1)                            # (M,)
    se_obs = float(np.std(cell_means_obs, ddof=1)) / np.sqrt(M)

    input_idx = rng.integers(0, M, size=(n_bootstrap, M))          # (B, M)
    run_idx   = rng.integers(0, R, size=(n_bootstrap, M, R))       # (B, M, R)

    selected        = scores[input_idx]                             # (B, M, R)
    b_rng           = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]
    m_rng           = np.arange(M)[np.newaxis, :, np.newaxis]
    resampled       = selected[b_rng, m_rng, run_idx]              # (B, M, R)
    cell_means_boot = resampled.mean(axis=2)                       # (B, M)

    boot_stats = cell_means_boot.mean(axis=1)                      # (B,)
    boot_ses   = np.std(cell_means_boot, ddof=1, axis=1) / np.sqrt(M)  # (B,)

    if se_obs <= 0.0 or not np.isfinite(se_obs):
        return _percentile_interval(boot_stats, alpha)

    valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
    if not np.any(valid):
        return _percentile_interval(boot_stats, alpha)
    se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
    tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
    if tiny_frac > 0.05:
        return _percentile_interval(boot_stats, alpha)
    valid = valid & (boot_ses >= se_floor)
    if not np.any(valid):
        return _percentile_interval(boot_stats, alpha)

    t_stats = (boot_stats[valid] - observed_stat) / boot_ses[valid]
    t_lo = float(np.percentile(t_stats, 100.0 * alpha / 2))
    t_hi = float(np.percentile(t_stats, 100.0 * (1.0 - alpha / 2)))
    return (
        float(observed_stat - t_hi * se_obs),
        float(observed_stat - t_lo * se_obs),
    )


def bayes_binary_ci_1d(
    values: np.ndarray,
    alpha: float,
    prior: tuple = (1, 1),
) -> tuple[float, float]:
    """Bayesian credible interval for binary (0/1) data using a Beta posterior.

    Places a Beta(prior[0], prior[1]) prior on the Bernoulli success probability
    and returns the equal-tailed ``(1 - alpha)`` credible interval.  With the
    default uniform prior (1, 1), this is equivalent to the HDI of
    Beta(a + 1, n - a + 1).

    Parameters
    ----------
    values : np.ndarray
        1-D array of binary observations (0 or 1).
    alpha : float
        Significance level (1 − confidence level).
    prior : tuple
        Beta prior parameters (default: (1, 1), i.e. uniform).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Credible interval clamped to [0, 1].
    """
    from scipy.stats import beta as _beta_dist

    n = len(values)
    a = int(np.round(np.sum(values)))
    b = n - a
    dist = _beta_dist(a + prior[0], b + prior[1])
    lo, hi = dist.interval(1.0 - alpha)
    return float(lo), float(hi)


def bayes_paired_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
    num_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Bayesian credible interval for the paired binary difference p(A=1) − p(B=1).

    Implements the paired Dirichlet-multinomial model from Bowyer et al. (2025),
    "Don't use the CLT in LLM evals with fewer than a few hundred datapoints"
    (``evalstats/core/bayes_evals.py``).  Uses importance sampling over a
    bivariate Gaussian model to obtain the full posterior over
    ``theta_A - theta_B``, accounting for within-question correlation.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        1-D binary arrays of equal length.  Values are thresholded at 0.5
        to determine binary membership.
    alpha : float
        Significance level (1 − confidence level).
    num_samples : int
        Number of importance samples (default 10,000).
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    (ci_low, ci_high, prob_a_greater) : tuple[float, float, float]
        Equal-tailed credible interval on p(A=1) − p(B=1), and the posterior
        probability P(theta_A > theta_B).
    """
    from .bayes_evals import binorm_cdf
    from scipy.stats import norm as _norm

    if rng is None:
        rng = np.random.default_rng()

    a_bin = (values_a >= 0.5).astype(float)
    b_bin = (values_b >= 0.5).astype(float)

    # 2×2 contingency table
    S = float(np.sum(a_bin * b_bin))              # A=1, B=1
    T = float(np.sum(a_bin * (1.0 - b_bin)))      # A=1, B=0
    U = float(np.sum((1.0 - a_bin) * b_bin))      # A=0, B=1
    V = float(np.sum((1.0 - a_bin) * (1.0 - b_bin)))  # A=0, B=0

    # Sample from the prior: uniform on (0,1) for theta, Beta(4,2)-shifted for rho
    theta_As = rng.beta(1.0, 1.0, size=num_samples)
    theta_Bs = rng.beta(1.0, 1.0, size=num_samples)
    rhos = 2.0 * rng.beta(4.0, 2.0, size=num_samples) - 1.0
    rhos = np.clip(rhos, -1.0 + 1e-20, 1.0 - 1e-20)

    diff = theta_As - theta_Bs

    # Bivariate normal probit parameterisation
    mu_A = _norm.ppf(theta_As)
    mu_B = _norm.ppf(theta_Bs)

    theta_V = binorm_cdf(0, 0, mu_A, mu_B, 1, 1, rhos)
    theta_S = theta_As + theta_Bs + theta_V - 1.0
    theta_T = 1.0 - theta_Bs - theta_V
    theta_U = 1.0 - theta_As - theta_V

    # Log-likelihood (= log-weights since prior = proposal)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = (
            S * np.log(theta_S)
            + T * np.log(theta_T)
            + U * np.log(theta_U)
            + V * np.log(theta_V)
        )

    max_log_w = np.nanmax(log_w)
    weights = np.exp(log_w - max_log_w)
    weights[np.isnan(weights)] = 0.0
    w_sum = float(weights.sum())
    if w_sum == 0.0:
        weights = np.ones(num_samples, dtype=float) / num_samples
    else:
        weights /= w_sum

    indices = rng.choice(num_samples, size=num_samples, replace=True, p=weights)
    diff_post = diff[indices]

    ci_low = float(np.percentile(diff_post, 100.0 * alpha / 2))
    ci_high = float(np.percentile(diff_post, 100.0 * (1.0 - alpha / 2)))
    prob_a_greater = float((diff_post > 0).mean())

    return ci_low, ci_high, prob_a_greater


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
