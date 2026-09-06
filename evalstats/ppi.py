"""evalstats.ppi — Prediction-Powered Inference (PPI) for arbitrary estimators.

Access via ``import evalstats as es; es.ppi.correct(...)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import expit as _sigmoid
from scipy.stats import norm as _norm_dist
from scipy.stats import t as _t_dist

from .core.resampling import _LOGIT_T_BOUNDARY_EPS

_MIN_LAB_RECOMMENDED = 30
"""Below this many labeled items, the percentile bootstrap is known to
undercover -- see :func:`correct`'s ``backend`` parameter. Shared with the
PPI-alignment warning threshold in ``evalstats/api.py`` and
``evalstats/alignment.py``."""

_POWER_TUNE_SHRINKAGE_C = 20.0
"""Pseudo-count controlling how much :func:`correct`'s power-tuning weight
lambda gets shrunk toward an adaptive target as ``n_lab`` shrinks -- used
identically by the bootstrap path (below) and the analytic-mean backend
(:func:`_analytic_mean_point_se`, via :func:`_adaptive_shrink_lambda`). The
target is estimated from the data (confidently-informative judge -> target
near 1, confidently-uninformative judge -> target near 0) rather than fixed
at 1, since a raw lambda estimated from only ``n_lab`` points has no reason
to be presumed close to 1 -- see the ``power_tune`` parameter docstring."""


def _adaptive_shrink_lambda(
    lam_raw: float, lam_replicates: Optional[np.ndarray], n_lab: int,
) -> float:
    """Shrink a raw power-tuning weight toward an adaptive target instead of
    a fixed target of 1. Shared by every ``power_tune`` implementation in
    this module (``correct()``'s bootstrap path, :func:`_analytic_mean_point_se`,
    :func:`_analytic_walsh_theta_correct`, and others), so they share one
    implementation rather than each reimplementing the arithmetic.

    ``lam_replicates`` is an array of independent raw-lambda estimates from
    resampling the labeled pair (see :func:`_bootstrap_batch_lambda_replicates`
    / :func:`_analytic_mean_lambda_replicates`). Pass ``None`` when a
    degenerate-labeled-sample guard already fired upstream -- this shrinks
    toward a target of 1, matching every backend's degenerate fallback."""
    target = 1.0 if lam_replicates is None else 1.0 - float(np.mean(lam_replicates < 0.5))
    w = n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
    return w * lam_raw + (1.0 - w) * target


def _lambda_var_inflation(r_term: float, lam_replicates: Optional[np.ndarray]) -> float:
    """Delta-method variance-inflation term for a power-tuned PPI estimate of
    the form ``f_lab + lambda_hat * r_term``: the ``r_term**2 * Var(lambda_hat)``
    piece of Var(lambda_hat * r_term), using ``lam_replicates`` as the
    Var(lambda_hat) estimate. Returns 0.0 when ``lam_replicates`` is ``None``
    or too small to estimate a variance from (an already-fixed lambda, or
    the degenerate-labeled-sample guard fired upstream).

    Every power_tune site's variance/CI construction otherwise plugs in the
    adaptively-chosen lambda as if it were a known constant, ignoring that
    it was estimated from the same sample it then corrects. Callers with an
    explicit variance formula (:func:`_analytic_mean_point_se`,
    :func:`_analytic_walsh_theta_correct`) add this directly to their
    variance; callers building a percentile-bootstrap CI (``correct()``'s
    bootstrap path, ``evalstats.tests._ppi_paired_bayes_bootstrap``)
    convolve in independent noise of this variance instead, since lambda is
    held fixed across every replicate there.

    Only inflates uncertainty when lambda estimation is itself uncertain,
    not whenever n_lab is small -- a confidently-poor judge (tight lambda
    replicates near 0) isn't penalized, only a genuinely ambiguous one. In
    ``evalstats.api._ppi_bootstrap_t_joint_stats``'s Romano-Wolf step-down
    construction, ``r_term`` must be held fixed at its observed value
    (not resampled per replicate) for this to hold, matching how lambda
    itself is held fixed across replicates there."""
    if lam_replicates is None or len(lam_replicates) <= 1:
        return 0.0
    var_lam_hat = float(np.var(lam_replicates, ddof=1))
    return r_term * r_term * var_lam_hat


def _shrunk_lambda_variance(lam_raw: float, var_lam_raw: float, w: float) -> float:
    """Closed-form delta-method estimate of Var(shrunk lambda) --
    Var(w*lam_raw + (1-w)*target) -- accounting for the adaptive shrinkage
    target's own sampling uncertainty, without a nested bootstrap. Used by
    :func:`evalstats.tests._ppi_friedman_f_stat`/`_ppi_anova_repeated_f_stat`
    in place of a naive ``Var(lam_raw)`` plug-in (an implicit ``w=1``
    assumption that ignores the target's uncertainty).

    :func:`_adaptive_shrink_lambda`'s ``target`` is
    ``1 - mean(lam_replicates < 0.5)``, which approximates
    ``P(lam_raw_boot >= 0.5)`` under the bootstrap distribution of
    ``lam_raw``, i.e. ``Phi((lam_raw - 0.5) / sigma)`` with
    ``sigma = sqrt(var_lam_raw)`` and ``Phi`` the standard normal CDF.
    Treating ``sigma`` as fixed, the shrunk lambda is
    ``H(lam_raw) = w*lam_raw + (1-w)*h(lam_raw)`` with
    ``H'(lam_raw) = w + (1-w)*phi(z)/sigma`` (``phi`` = standard normal PDF,
    ``z = (lam_raw - 0.5)/sigma``), giving

        Var(lam) ~= H'(lam_raw)^2 * Var(lam_raw) = [w*sigma + (1-w)*phi(z)]^2

    This replaces (not adds to) a raw ``Var(lam_raw)`` plug-in. At ``w=1``
    (no shrinkage) it reduces to ``Var(lam_raw)`` exactly, matching the
    un-shrunk case."""
    if var_lam_raw <= 0.0:
        return 0.0
    sigma = float(np.sqrt(var_lam_raw))
    z = (lam_raw - 0.5) / sigma
    phi_z = float(_norm_dist.pdf(z))
    return float((w * sigma + (1.0 - w) * phi_z) ** 2)


def _bootstrap_batch_lambda_replicates(
    b_lab: np.ndarray, b_hat_lab: np.ndarray, b_unlab: np.ndarray,
) -> np.ndarray:
    """Turn an existing ``(n_boot,)`` bootstrap draw (each element the
    per-replicate estimator value, e.g. a bootstrap-resample MEAN -- not
    the full resampled data) into an array of independent raw-lambda
    replicates for :func:`_adaptive_shrink_lambda`, by splitting it into
    batches and recomputing the covariance/variance ratio within each
    batch. A single element of ``b_lab``/``b_hat_lab`` can't yield a
    covariance on its own (it's already reduced to one number per
    replicate), so this pools ``n_boot // n_batches`` of them per batch --
    unlike :func:`_analytic_mean_lambda_replicates`/:func:`
    _walsh_theta_lambda_replicates`, which resample the full labeled-pair
    ARRAY per draw and so can compute one ratio straight from each draw
    with no pooling needed. Used by ``correct()``'s bootstrap path and
    ``evalstats.tests._ppi_paired_bayes_bootstrap`` (whose Dirichlet-
    weighted ``b1_*`` draws have the identical shape/meaning for this
    purpose, just resampled differently upstream)."""
    n_boot = len(b_lab)
    n_batches = max(5, min(30, n_boot // 50))
    batch_size = n_boot // n_batches
    lam_batches = np.empty(n_batches)
    for k in range(n_batches):
        sl = slice(k * batch_size, (k + 1) * batch_size)
        d = float(np.var(b_unlab[sl] - b_hat_lab[sl], ddof=1))
        if d > 1e-12:
            lb = float(np.cov(b_lab[sl], b_hat_lab[sl], ddof=1)[0, 1] / d)
            lam_batches[k] = min(max(lb, 0.0), 1.0)
        else:
            lam_batches[k] = 1.0
    return lam_batches


_LABEL_SHIFT_SHRINKAGE_K = 3.0
"""Pseudo-count controlling how aggressively :func:`_analytic_mean_point_se`
(when ``label_shift_robust=True``) blends its power-tuned lambda back
toward 1.0 (full rectifier / no power tuning) in response to a detected
labeled-vs-unlabeled judge-score distribution shift -- see that function's
``label_shift_robust`` parameter docstring for the mechanism this addresses
(label-selection MNAR's "restriction of range" attenuation of the
power-tuning ratio). Small enough that a strongly-detected shift blends
most of the way to lambda=1; large enough that MCAR/weak-judge scenarios
(pure null noise in the shift statistic) are barely perturbed."""


_ANALYTIC_TARGET_SEED = 0
"""Fixed internal seed for :func:`_analytic_mean_lambda_replicates`/
:func:`_walsh_theta_lambda_replicates`'s micro-bootstrap -- purely an
implementation detail for cheaply approximating a shrinkage target (see
:func:`_adaptive_shrink_lambda`), not a source of reported Monte Carlo
uncertainty a caller would need to control. Fixed (not threaded through
from callers) so every caller -- there are several, across ``evalstats/``
-- stays fully deterministic (same inputs -> same outputs) without a
signature change."""


def _analytic_mean_lambda_replicates(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, var_unlab: float, n_lab: int,
    n_boot: int = 800,
) -> np.ndarray:
    """Raw-lambda replicates for :func:`_adaptive_shrink_lambda`, for a
    MEAN-based rectifier (plain sample covariance/variance) -- used by
    :func:`_analytic_mean_point_se` and (per-pair, in a loop)
    ``evalstats.api._ppi_bootstrap_t_joint_stats``, since both use the
    identical mean-based ratio.

    Since ``var_unlab`` is already a closed form from the large unlabeled
    sample (no resampling needed there), only the small (Y_lab, Y_hat_lab)
    PAIRED sample needs to be resampled -- cheap regardless of n_lab.
    Unlike :func:`_bootstrap_batch_lambda_replicates`'s ``b1`` arrays
    (which store only a bootstrap MEAN per draw, so a group of them has to
    be pooled before a single ratio can be computed at all), each resample
    here is the FULL (Y_lab, Y_hat_lab) pair array, so the exact same
    closed-form ratio the point estimate itself uses can be recomputed
    directly per draw -- a standard bootstrap-the-statistic distribution,
    no batching needed. n_boot=800 pairs are cheap regardless of n_lab, so
    this stays fast even at the small n_lab (~15-30) these callers target."""
    rng = np.random.default_rng(_ANALYTIC_TARGET_SEED)
    idx = rng.integers(0, n_lab, size=(n_boot, n_lab))
    Yl_b = Y_lab[idx]
    Yh_b = Y_hat_lab[idx]
    var_hat_lab_b = Yh_b.var(axis=1, ddof=1) / n_lab
    cov_b = ((Yl_b - Yl_b.mean(axis=1, keepdims=True)) * (Yh_b - Yh_b.mean(axis=1, keepdims=True))).sum(axis=1) / (n_lab - 1) / n_lab
    denom_b = var_unlab + var_hat_lab_b
    return np.where(denom_b > 1e-12, np.clip(cov_b / np.maximum(denom_b, 1e-300), 0.0, 1.0), 1.0)


def _label_shift_blend_weight(
    f_hat_lab: float, f_unlab: float, var_hat_lab: float, var_unlab: float, k: float,
) -> float:
    """How much to trust the power-tuned lambda vs. fall back toward 1.0
    (full rectifier), based on whether the labeled subsample's judge-score
    distribution detectably differs from the unlabeled subsample's -- see
    :func:`_analytic_mean_point_se`'s ``label_shift_robust`` docstring.
    Returns ``w_rep`` such that the final lambda is
    ``w_rep * lam_power_tuned + (1 - w_rep) * 1.0``: 1.0 means "no detected
    shift, trust power-tuning fully" and 0.0 means "strongly detected
    shift, fall back to the full-rectifier estimator entirely."

    ``z_shift**2`` is approximately chi2(1)-distributed under a true null of
    no labeled/unlabeled shift, so ``excess = max(0, z_shift**2 - 1)`` is a
    null-centered "excess evidence" statistic, fed through the same
    pseudo-count shrinkage-blend shape :func:`_adaptive_shrink_lambda` uses
    elsewhere (``w = excess / (excess + k)``)."""
    var_shift = var_hat_lab + var_unlab
    if var_shift <= 1e-12:
        return 1.0
    z_shift = abs(f_hat_lab - f_unlab) / np.sqrt(var_shift)
    excess = max(0.0, z_shift * z_shift - 1.0)
    return 1.0 - excess / (excess + k)


def _label_shift_blended_lambda_replicates(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, f_unlab: float, var_unlab: float,
    n_lab: int, w_shrink: float, target: float, k: float, n_boot: int = 800,
) -> np.ndarray:
    """Bootstrap replicates of the full ``label_shift_robust`` lambda chain
    (raw ratio -> adaptive shrink -> label-shift blend), for a variance
    estimate that captures the shift-blend's own sampling noise, not just
    the raw ratio's -- the blend weight is itself a noisy function of the
    same small labeled sample, so holding it fixed at its observed value
    under-covers. ``f_unlab`` (the fixed unlabeled-pool mean) and the
    adaptive-shrink parameters ``(w_shrink, target)`` are held fixed at
    their already-computed values, matching :func:`_lambda_var_inflation`'s
    "hold other pieces fixed across replicates" convention -- only the
    (Y_lab, Y_hat_lab) pair is resampled, exactly as in
    :func:`_analytic_mean_lambda_replicates`."""
    rng = np.random.default_rng(_ANALYTIC_TARGET_SEED)
    idx = rng.integers(0, n_lab, size=(n_boot, n_lab))
    Yl_b = Y_lab[idx]
    Yh_b = Y_hat_lab[idx]
    mean_hat_lab_b = Yh_b.mean(axis=1)
    var_hat_lab_b = Yh_b.var(axis=1, ddof=1) / n_lab
    cov_b = ((Yl_b - Yl_b.mean(axis=1, keepdims=True)) * (Yh_b - Yh_b.mean(axis=1, keepdims=True))).sum(axis=1) / (n_lab - 1) / n_lab
    denom_b = var_unlab + var_hat_lab_b
    lam_raw_b = np.where(denom_b > 1e-12, np.clip(cov_b / np.maximum(denom_b, 1e-300), 0.0, 1.0), 1.0)
    lam_power_tuned_b = w_shrink * lam_raw_b + (1.0 - w_shrink) * target

    var_shift_b = var_unlab + var_hat_lab_b
    z_shift_b = np.abs(mean_hat_lab_b - f_unlab) / np.sqrt(np.maximum(var_shift_b, 1e-300))
    excess_b = np.maximum(0.0, z_shift_b * z_shift_b - 1.0)
    w_rep_b = 1.0 - excess_b / (excess_b + k)
    return w_rep_b * lam_power_tuned_b + (1.0 - w_rep_b) * 1.0


@dataclass
class PPIResult:
    """Result returned by :func:`correct`.

    Attributes
    ----------
    estimate : float
        PPI-corrected point estimate.
    ci_low, ci_high : float
        Lower and upper bounds of the bootstrap confidence interval.
    alpha : float
        Significance level used (e.g. 0.05 for a 95 % CI).
    llm_estimate : float
        Uncorrected LLM-only estimate ``f(Ŷ_unlab, X_unlab)``.
    human_estimate : float
        Human-only estimate on the labeled subset ``f(Y_lab, X_lab)``.
    rectifier : float
        Bias-correction term ``human_estimate − f(Ŷ_lab, X_lab)``.
        Positive values mean the LLM overestimates; negative means it underestimates.
    p_value : float or None
        Two-sided bootstrap p-value for H₀: θ = 0.
        None when *compute_pvalue=False* was passed to :func:`correct`.
    """

    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    llm_estimate: float
    human_estimate: float
    rectifier: float
    p_value: Optional[float]
    lam: Optional[float] = None
    """The power-tuning weight actually used (see :func:`correct`'s
    ``power_tune`` parameter) -- ``None`` when ``power_tune=False`` (the
    default), in which case ``lam`` is implicitly 1.0 (today's fixed "full
    rectifier" PPI estimator). When ``power_tune=True``, this is the
    bootstrap-estimated PPI++ weight, clipped to ``[0, 1]``: 0 means the
    correction contributed nothing (falls back to the classical
    labels-only estimate, ``human_estimate``); 1 means the full rectifier
    was worth applying (identical to ``power_tune=False``'s estimate)."""

    def __repr__(self) -> str:
        ci_pct = int(round((1 - self.alpha) * 100))
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"PPIResult(estimate={self.estimate:.4f}, "
            f"{ci_pct}%CI=[{self.ci_low:.4f}, {self.ci_high:.4f}]"
            f"{p_str})"
        )

    def summary(self) -> None:
        """Print a human-readable summary of the PPI result."""
        ci_pct = int(round((1 - self.alpha) * 100))
        w = 26
        print(f"{'PPI corrected estimate':<{w}}: {self.estimate:.4f}")
        print(f"{f'{ci_pct}% CI':<{w}}: [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        if self.p_value is not None:
            print(f"{'p-value (H₀: θ=0)':<{w}}: {self.p_value:.4f}")
        print(f"{'LLM-only estimate':<{w}}: {self.llm_estimate:.4f}")
        print(f"{'Human-only estimate':<{w}}: {self.human_estimate:.4f}")
        print(f"{'Rectifier (δ)':<{w}}: {self.rectifier:+.4f}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call(func: Callable, Y: np.ndarray, X: Optional[np.ndarray]) -> float:
    """Invoke estimator with (Y,) or (Y, X) depending on whether X is supplied."""
    if X is None:
        return float(func(Y))
    return float(func(Y, X))


_MEDIAN_TIE_JITTER_DIVISOR = 20.0
"""``correct()``'s smoothed-bootstrap jitter std is (min positive gap
between distinct values in the data) / this divisor -- small enough to only
break exact ties, large enough to stop resamples from repeatedly landing on
the identical median. See :func:`_tie_jitter_scale`."""


def _tie_jitter_scale(arr: np.ndarray) -> float:
    """Std-dev of the Gaussian jitter ``correct()`` adds before each
    bootstrap resample's median, when ``estimator_func``/``rectifier_func``
    is ``np.median``.

    Percentile-bootstrapping a median on data with substantial exact ties
    is a known-bad combination (the "smoothed bootstrap" literature, e.g.
    Efron; Hall & DiCiccio-Romano): with enough repeated values, most
    resamples' median lands on the same repeated value, collapsing the
    bootstrap distribution toward a near-constant and making the CI/p-value
    severely too conservative. Adding noise far below the data's real
    resolution before each resample's median restores a non-degenerate
    bootstrap distribution without perturbing genuine order structure.

    Returns 0.0 (no jitter) when the array has fewer than 2 distinct
    values, or when every gap between consecutive distinct values is
    effectively zero (already degenerate; jittering wouldn't help).
    """
    uniq = np.unique(arr)
    if len(uniq) < 2:
        return 0.0
    gaps = np.diff(uniq)
    positive_gaps = gaps[gaps > 1e-15]
    if positive_gaps.size == 0:
        return 0.0
    return float(np.min(positive_gaps)) / _MEDIAN_TIE_JITTER_DIVISOR


def _walsh_theta_row(d: np.ndarray) -> float:
    """Exact O(n log n) computation of the one-sample midrank-sign
    statistic ``P_mid(Walsh_ij > 0)`` for a single array of paired
    differences ``d``, where ``Walsh_ij = (d_i + d_j) / 2`` for ``i <= j``
    (``n*(n+1)/2`` pairs total, including self-pairs ``i == j``). This is
    the Hodges-Lehmann one-sample location estimator's own construction
    (its point estimate is the median of the Walsh averages; this is the
    analogous midrank-tie-corrected sign statistic of those same averages).

    Computed via a sort + vectorized ``searchsorted`` rather than the naive
    O(n^2) pairwise enumeration, which is too slow to call thousands of
    times per :func:`correct` invocation at realistic corpus sizes."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return 0.5
    ds = np.sort(d)
    neg = -ds
    idx_right = np.searchsorted(ds, neg, side="right")
    idx_left = np.searchsorted(ds, neg, side="left")
    i_arr = np.arange(n)
    hi = np.maximum(idx_right, i_arr)
    lo = np.maximum(idx_left, i_arr)
    count_gt = n - hi
    count_eq = np.maximum(0, idx_right - lo)
    total_pairs = n * (n + 1) // 2
    return float((count_gt.sum() + 0.5 * count_eq.sum()) / total_pairs)


def paired_walsh_midrank_theta(d: np.ndarray) -> float:
    """``_walsh_theta_row(d) - 0.5``, shifted to be 0 under H0 (D symmetric
    about 0). This is :func:`correct`'s default estimand for the Wilcoxon
    signed-rank family (``wilcoxon()``, via
    ``evalstats.tests._ppi_paired_arrays``).

    Counting signs of all pairwise Walsh averages ``(d_i + d_j) / 2``,
    rather than a per-item sign proportion or the paired median, avoids the
    power collapse a median-based estimand suffers under heavy ties (the
    population median of a paired difference can stay locked at exactly 0
    even under a real, classical-Wilcoxon-detectable shift). It is
    asymptotically equivalent to the Wilcoxon signed-rank statistic itself.

    An earlier version of this pipeline inflated Type-I error at small
    ``n_lab`` (~15) on data with an extreme tie rate; the current
    implementation holds nominal calibration in that regime. See
    ``simulations/harness/cases/pvalues.py``/``ppi_real.py``'s WILCOXON
    blocks for the calibration checks behind this.

    Registered in :func:`correct`'s ``_fast_batch`` dispatch (via
    :func:`_walsh_theta_batch`) so bootstrap replicates go through the same
    vectorized-resampling fast path ``np.mean``/``np.median`` use, instead
    of the slow per-replicate Python loop arbitrary ``estimator_func``
    callers fall into -- required for this to be practical at realistic
    corpus sizes and ``n_boot``."""
    d = np.asarray(d)
    if len(d) == 0:
        return 0.0
    return _walsh_theta_row(d) - 0.5


def _walsh_theta_batch(arr: np.ndarray) -> np.ndarray:
    """Row-wise :func:`paired_walsh_midrank_theta` over a ``(m, n)``
    bootstrap-replicate array, for :func:`correct`'s ``_fast_batch``
    dispatch. Still a Python-level loop over the ``m`` replicates (no known
    way to vectorize ``_walsh_theta_row``'s per-row sort + searchsorted
    across rows without materializing a worse O(m * n^2) intermediate), but
    each row's O(n log n) call is fast enough that looping ``m`` ~2000-4000
    times is practical.

    Returns the shifted (-0.5) value, matching
    ``paired_walsh_midrank_theta`` exactly -- not the raw
    ``_walsh_theta_row`` value. ``correct()`` looks this batch function up
    by the shifted function's own ``id()``, so an unshifted return here
    would offset every bootstrap replicate by +0.5 relative to the point
    estimate, pushing the whole bootstrap distribution away from 0."""
    return np.array([_walsh_theta_row(row) - 0.5 for row in arr])


def _walsh_theta_h1_components(d: np.ndarray) -> np.ndarray:
    """Per-item empirical Hajek-projection ("structural component") values
    for :func:`paired_walsh_midrank_theta`'s underlying pairwise kernel
    ``h(d_i,d_j) = 1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}`` -- the same
    "structural components" construction DeLong et al. (1988) use to get
    the variance (and, for two correlated samples, covariance) of an AUC
    estimator, applied here to the analogous one-sample Walsh-average
    U-statistic instead of DeLong's two-sample Mann-Whitney U-statistic.

    ``h1_hat(d_i) = (1/n) * sum_j [1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}]``, with
    ``j`` ranging over all ``n`` items (including ``j=i``) -- the plug-in
    estimate of the first-order Hajek projection ``h1(d) = E_D2[h(d, D2)]``,
    evaluated at each observed ``d_i``. This is a "leave-in" (not
    leave-one-out) projection, which does not matter for a variance
    estimate (only the point estimate needs the sharper leave-one-out
    correction, and this function is never used for the point estimate --
    :func:`paired_walsh_midrank_theta`, via :func:`_walsh_theta_row`,
    handles that separately).

    Used by :func:`_walsh_theta_analytic_variance` (single-sample variance,
    ``Var(U) ~= (4/n)*Var(h1_hat)``, the standard degree-2 U-statistic
    asymptotic variance) and directly by
    :func:`_analytic_walsh_theta_correct` (paired covariance: for two
    structural-component arrays computed on the same n items,
    ``Var(A-B) ~= (4/n)*Var(h1_hat_A - h1_hat_B)``, folding the
    variance-of-A, variance-of-B, and covariance(A,B) terms into one pass).

    Same O(n log n) sort + vectorized ``searchsorted`` construction as
    :func:`_walsh_theta_row`, just without that function's ``i<=j``
    upper-triangular restriction (every item is compared against all n
    others here, not just itself-and-later)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return np.empty(0, dtype=float)
    ds = np.sort(d)
    neg = -d
    idx_right = np.searchsorted(ds, neg, side="right")
    idx_left = np.searchsorted(ds, neg, side="left")
    count_gt = n - idx_right
    count_eq = idx_right - idx_left
    return (count_gt + 0.5 * count_eq) / n


def _walsh_theta_analytic_variance(d: np.ndarray) -> float:
    """Analytic (no-bootstrap) variance estimate of
    ``paired_walsh_midrank_theta(d)`` for a single sample ``d``, via the
    standard degree-2 U-statistic asymptotic variance
    ``Var(U_n) ~= (m^2/n)*zeta_1`` (``m=2`` for a pairwise kernel), with
    ``zeta_1 = Var(h1(D))`` estimated by the empirical plug-in structural
    components from :func:`_walsh_theta_h1_components`. See that
    function's docstring for the numerical validation against a Monte
    Carlo oracle."""
    n = len(d)
    if n < 2:
        return 0.0
    h1 = _walsh_theta_h1_components(d)
    return 4.0 * float(np.var(h1, ddof=1)) / n


def _walsh_theta_lambda_replicates(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, var_unlab: float, n_lab: int,
    n_boot: int = 800,
) -> np.ndarray:
    """Raw-lambda replicates for :func:`_adaptive_shrink_lambda`, for the
    Walsh-theta rectifier -- same idea as
    :func:`_analytic_mean_lambda_replicates` (see that function's
    docstring for the shared rationale), just re-evaluating the
    Hajek-projection cov/var ratio (not the mean's plain sample cov/var)
    per resample of the (Y_lab, Y_hat_lab) pair.

    ``_walsh_theta_h1_components``'s O(n log n) sort+searchsorted isn't
    vectorizable across a batch dimension (see :func:`_walsh_theta_batch`'s
    docstring) -- stays a Python loop over ``n_boot`` draws, same tradeoff
    that function already accepts. Fine here: n_lab is small (this
    backend's whole point is being fast at the small n_lab where it's
    preferred), so n_boot=800 tiny sorts is cheap in aggregate.

    CALLER GUARD (referenced from every site that calls a
    ``_..._lambda_replicates`` helper). Callers must skip this and pass
    ``lam_replicates=None`` when the labeled pair is degenerate:
    ``n_lab <= 1 or var_hat_lab < 1e-12 or var_lab < var_hat_lab * 1e-6``.
    The ``var_hat_lab < 1e-12`` clause needs to be its OWN absolute check,
    not merely the relative ``var_lab < var_hat_lab * 1e-6`` one: when
    ``Y_hat_lab`` is EXACTLY tied (all Walsh comparisons agree -- plausible
    at small n_lab on real, heavily-tied Likert-like data) the covariance is
    identically 0 regardless of ``Y_lab``'s own spread, and the relative
    test cannot fire because ``0 < 0`` is False. Letting a spuriously
    "confident" lambda through there was confirmed (2026-08-15) to drive
    real-data Type-I as high as 0.515 -- see
    results_why_ppi_shrink_1_over_0.md's real-data wilcoxon addendum."""
    rng = np.random.default_rng(_ANALYTIC_TARGET_SEED)
    idx = rng.integers(0, n_lab, size=(n_boot, n_lab))
    lam_b = np.empty(n_boot)
    for b in range(n_boot):
        h1_lab_b = _walsh_theta_h1_components(Y_lab[idx[b]])
        h1_hat_lab_b = _walsh_theta_h1_components(Y_hat_lab[idx[b]])
        var_hat_lab_b = 4.0 * float(np.var(h1_hat_lab_b, ddof=1)) / n_lab
        denom_b = var_unlab + var_hat_lab_b
        if denom_b > 1e-12:
            cov_b = 4.0 * float(np.cov(h1_lab_b, h1_hat_lab_b, ddof=1)[0, 1]) / n_lab
            lam_b[b] = min(max(cov_b / denom_b, 0.0), 1.0)
        else:
            lam_b[b] = 1.0
    return lam_b


_WALSH_SIGNFLIP_B = 200
"""Sign-flip draws used by :func:`_walsh_theta_signflip_null_var`. 200 is
enough for a variance (not a tail quantile) and keeps the cost well below
the two :func:`_walsh_theta_lambda_replicates` calls the cross-fitted
construction it replaced already paid."""


def _walsh_theta_signflip_null_var(Y_lab: np.ndarray, n_boot: int = _WALSH_SIGNFLIP_B) -> Optional[float]:
    """Var(theta) under H0, obtained by SIGN-FLIP randomization -- the
    score-test counterpart to :func:`_walsh_theta_analytic_variance`'s
    Wald (evaluate-at-the-estimate) variance.

    Under H0 the paired differences are symmetric about 0, so flipping the
    sign of any subset of them leaves the null distribution unchanged. The
    variance of ``paired_walsh_midrank_theta`` across sign flips is
    therefore its null variance, computed CONDITIONAL on the observed
    ``|Y_lab|`` -- which is exactly the classical randomization reference
    for a signed-rank statistic, and is valid regardless of ties.

    Why not the textbook closed form. Under H0 (and no ties) the Walsh
    count IS the Wilcoxon signed-rank statistic, giving
    ``Var(theta) = (2n+1) / (6n(n+1))`` exactly. That matches simulation on
    continuous data (n=20: 0.016270 vs. 0.016105 measured) but is badly
    wrong under the heavy ties real judge data carries -- on appstore's
    88%-tied judge differences it reads 0.021528 against a true 0.006156,
    3.5x too large. Sign-flipping handles ties and discreteness exactly, so
    it is used instead.

    Returns ``None`` when every labeled difference is exactly 0: sign
    flipping cannot move a vector of zeros, so no null variance is
    recoverable and the caller must fall back to the Wald estimate.

    Deterministic (fixed :data:`_ANALYTIC_TARGET_SEED`), matching every
    other source of internal randomness in this backend.
    """
    d = np.asarray(Y_lab, dtype=float)
    n = len(d)
    if n < 2 or not np.any(d != 0.0):
        return None
    rng = np.random.default_rng(_ANALYTIC_TARGET_SEED)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
    return float(np.var([paired_walsh_midrank_theta(d * flips[b]) for b in range(n_boot)], ddof=1))


def _cross_fit_satterthwaite_df(vA: float, dfA: float, vB: float, dfB: float) -> float:
    """Welch-Satterthwaite effective df for combining two fold-level
    variance estimates from :func:`_analytic_walsh_theta_correct`'s
    cross-fitting. Unlike a single-sample delta-method inflation term
    (tried and empirically rejected for this problem -- see
    ``simulations/out/results_why_ppi_shrink_1_over_0.md``'s Wilcoxon
    power-tuning addendum), ``vA``/``vB`` here really ARE independent:
    each fold's point estimate uses the OTHER fold's lambda, so this
    combination rests on a valid, not merely convenient, assumption."""
    total = vA + vB
    if total <= 0.0 or vA <= 0.0 or vB <= 0.0:
        return max(dfA, dfB, 1.0)
    return (total * total) / (vA * vA / dfA + vB * vB / dfB)


def _analytic_walsh_theta_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool,
) -> "PPIResult":
    """Closed-form (no-bootstrap) PPI correction for
    ``estimator_func=paired_walsh_midrank_theta`` -- the Walsh-average
    midrank-sign analogue of :func:`_analytic_mean_correct`'s role for
    ``np.mean``. Same two-term (rectifier + unlabeled-sample) variance
    decomposition and power-tuning shrinkage as that function; only the
    per-term variance/covariance estimator differs (Hajek-projection
    structural components -- :func:`_walsh_theta_h1_components` /
    :func:`_walsh_theta_analytic_variance` -- instead of the mean's plain
    sample variance/covariance).

    Used at every ``n_lab`` under ``backend="auto"`` (see
    :data:`_ANALYTIC_ALWAYS_PREFERRED`), not just below the usual n_lab=30
    cutoff: it dominates the percentile bootstrap on power for this
    estimand across the full n_lab range.

    ``power_tune=True`` evaluates the human term's variance under H0 (a
    score construction) rather than at the observed estimate (a Wald
    construction) -- the same reason this package prefers Wilson over Wald
    for binary proportions and Tango for paired binary. ``theta`` is
    proportion-like on [-0.5, 0.5], so its Wald variance is maximal at
    theta=0 and collapses toward the boundaries, mechanically shrinking the
    SE wherever ``|theta_hat|`` is large and inflating a two-sided test in
    both tails; a null variance is a constant with respect to ``theta_hat``
    and removes that coupling. Under H0 the Walsh count is the Wilcoxon
    signed-rank statistic, whose null law is distribution-free;
    :func:`_walsh_theta_signflip_null_var` obtains it by sign-flip
    randomization (exact under ties). The estimated correlation is kept and
    only the human-side variance is rescaled -- replacing ``var_lab`` alone
    would violate the quadratic form's Cauchy-Schwarz consistency (see the
    inline comment at the substitution).

    ``power_tune=False`` stays on the plain Wald variance: at fixed
    lambda=1 that path is long-validated (including under MNAR) and serves
    as the harness's classical reference baseline.

    Degrees of freedom for the Student-t interval: ``n_lab - 1``, matching
    :func:`_analytic_mean_correct`'s choice."""
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    f_unlab = paired_walsh_midrank_theta(Y_hat_unlab)
    var_unlab = _walsh_theta_analytic_variance(Y_hat_unlab) if n_all > 1 else 0.0

    f_lab = paired_walsh_midrank_theta(Y_lab)
    f_hat_lab = paired_walsh_midrank_theta(Y_hat_lab)
    rectifier = f_lab - f_hat_lab

    var_lab = _walsh_theta_analytic_variance(Y_lab) if n_lab > 1 else 0.0
    var_hat_lab = _walsh_theta_analytic_variance(Y_hat_lab) if n_lab > 1 else 0.0

    if n_lab > 1:
        h1_lab = _walsh_theta_h1_components(Y_lab)
        h1_hat_lab = _walsh_theta_h1_components(Y_hat_lab)
        cov_lab_hatlab = 4.0 * float(np.cov(h1_lab, h1_hat_lab, ddof=1)[0, 1]) / n_lab
    else:
        cov_lab_hatlab = 0.0

    lam = 1.0
    lam_replicates = None
    if power_tune:
        denom = var_unlab + var_hat_lab
        if denom > 1e-12:
            lam_raw = min(max(cov_lab_hatlab / denom, 0.0), 1.0)
        else:
            lam_raw = 1.0  # degenerate variance -- fall back, don't divide by ~0.

        # Adaptive shrinkage -- see _adaptive_shrink_lambda's docstring for
        # the shared rationale, and _walsh_theta_lambda_replicates for this
        # estimand's version of the replicate-generation step (including why
        # var_hat_lab needs an absolute floor check of its own).
        if n_lab <= 1 or var_hat_lab < 1e-12 or var_lab < var_hat_lab * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _walsh_theta_lambda_replicates(Y_lab, Y_hat_lab, var_unlab, n_lab)
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)

    estimate = f_lab + lam * (f_unlab - f_hat_lab)

    # SCORE-TYPE variance for the human term (power_tune only) -- see this
    # function's docstring for the full rationale. var_lab is the Wald
    # (evaluate-at-the-estimate) variance, and because Var(theta) DEPENDS on
    # theta for this proportion-like estimand, using it couples se to
    # |estimate| and inflates a two-sided test. Substituting the H0 variance
    # breaks that coupling.
    #
    # The substitution must be COHERENT. var_lab + lam^2*D - 2*lam*cov is the
    # variance of an actual linear combination, so it is non-negative only
    # because cov^2 <= var_lab*var_hat_lab (Cauchy-Schwarz) holds for the
    # Wald pair. Swapping var_lab alone breaks that: measured 9.0% of reps
    # went negative, clamped to ~0 se, and produced spurious rejections
    # (Type-I 0.122). So keep the ESTIMATED CORRELATION and rescale only the
    # human side:
    #     rho      = cov / sqrt(var_lab * var_hat_lab)
    #     cov_used = rho * sqrt(var_null * var_hat_lab)
    # The result is a quadratic in lam with discriminant
    # 4*var_null*(rho^2*var_hat_lab - D) <= 0, since D = var_unlab +
    # var_hat_lab >= var_hat_lab >= rho^2*var_hat_lab -- provably non-negative.
    var_lab_used, cov_used = var_lab, cov_lab_hatlab
    if power_tune:
        var_null = _walsh_theta_signflip_null_var(Y_lab)
        if var_null is not None and var_lab > 1e-15 and var_hat_lab > 1e-15:
            rho = float(np.clip(cov_lab_hatlab / np.sqrt(var_lab * var_hat_lab), -1.0, 1.0))
            var_lab_used = var_null
            cov_used = rho * float(np.sqrt(var_null * var_hat_lab))

    var_estimate = max(
        var_lab_used + lam * lam * (var_unlab + var_hat_lab) - 2.0 * lam * cov_used, 0.0
    )
    if power_tune:
        var_estimate += _lambda_var_inflation(f_unlab - f_hat_lab, lam_replicates)
    se = float(np.sqrt(var_estimate))
    df = max(n_lab - 1, 1)

    if se <= 0.0:
        ci_low = ci_high = estimate
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
    else:
        t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
        ci_low, ci_high = estimate - t_crit * se, estimate + t_crit * se
        p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(estimate) / se, df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=(lam if power_tune else None),
    )


def _analytic_mean_point_se(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray, power_tune: bool,
    label_shift_robust: bool = False,
) -> tuple[float, float, float, float, float, Optional[float], int]:
    """Shared closed-form point-estimate/SE/df computation for a PPI mean
    correction -- factored out of :func:`_analytic_mean_correct` so
    :func:`_analytic_logit_t_correct` can reuse the identical point-
    estimate/variance derivation (only the CI construction differs: a
    plain t-interval on the raw scale in ``_analytic_mean_correct`` vs. a
    delta-method logit-scale transform in ``_analytic_logit_t_correct``).
    See ``_analytic_mean_correct``'s docstring for the closed-form
    lambda*/variance derivation this implements.

    ``label_shift_robust`` (default False, preserving every existing
    caller's behavior unchanged) additionally blends the power-tuned
    lambda back toward 1.0 (full rectifier) in proportion to detected
    evidence of a labeled-vs-unlabeled judge-score distribution shift --
    see :func:`_label_shift_blend_weight`'s docstring for the mechanism.
    Fixes a bias/undercoverage failure specific to a single-arm mean
    estimand under label-selection MNAR (missingness correlated with an
    item's own true value, not just an observed covariate); see
    ``simulations/out/results_why_ppi_shrink_1_over_0.md`` for the
    calibration behind it.

    Root cause: the labeled subsample's dynamic range on the (unobserved-
    for-the-unlabeled-side) truth variable gets restricted by MNAR
    selection -- a classical "restriction of range" attenuation that pulls
    the power-tuning ratio ``lam_raw = Cov(Y_lab, Y_hat_lab) / (Var(Y_unlab)
    + Var(Y_hat_lab))`` (and the adaptive-shrinkage target, resampled from
    the same restricted sample) toward 0 even when the judge is a
    genuinely good predictor on the full population. For a two-group
    comparison (see ``_pooled_two_group_lambda``'s docstring) this same
    per-item selection mechanism biases both groups' point estimates
    roughly equally, so it mostly cancels in the difference; a single-arm
    estimand has no second group to cancel against, so the point estimate
    collapses toward the raw, badly-biased human-labels-only mean
    (``f_lab``) as lambda shrinks. This fix is deliberately scoped to
    single-arm callers only (``_ppi_single_t_interval``/``_ppi_single_logit_t``)
    rather than this function's paired-difference callers, where it isn't
    needed and wasn't validated.

    Does not achieve full nominal coverage under label-selection MNAR --
    non-ignorable (outcome-dependent) missingness is not, in general, fully
    identifiable without further assumptions.

    Returns (estimate, se, f_unlab, f_lab, rectifier, lam_or_None, df).
    """
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)
    if n_all == 0:
        raise ValueError(
            "PPI correction requires at least one unlabeled item, got n_all=0 "
            "(every item in this comparison is human-labeled, leaving no "
            "unlabeled residual for the LLM-only term). If every item is "
            "labeled, use the human labels directly instead of PPI correction."
        )

    f_unlab = float(np.mean(Y_hat_unlab))
    f_lab = float(np.mean(Y_lab))
    f_hat_lab = float(np.mean(Y_hat_lab))
    rectifier = f_lab - f_hat_lab

    var_unlab = float(np.var(Y_hat_unlab, ddof=1)) / n_all if n_all > 1 else 0.0
    var_lab = float(np.var(Y_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    cov_lab_hatlab = float(np.cov(Y_lab, Y_hat_lab, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0

    lam = 1.0
    if power_tune:
        denom = var_unlab + var_hat_lab
        if denom > 1e-12:
            lam_raw = min(max(cov_lab_hatlab / denom, 0.0), 1.0)
        else:
            lam_raw = 1.0  # degenerate variance -- fall back, don't divide by ~0.

        # Adaptive shrinkage (see _adaptive_shrink_lambda's docstring for
        # the shared rationale, and _analytic_mean_lambda_replicates for
        # this estimand's version of the replicate-generation step). Falls
        # back to target=1 when Y_lab itself is near-degenerate: a
        # near-constant labeled sample can't reveal covariance no matter
        # how it's resampled, so a "confidently near 0" reading there is a
        # resampling artifact, not evidence -- the same reasoning the
        # degenerate `denom<=1e-12` fallback above already uses.
        raw_var_lab = var_lab * n_lab
        raw_var_hat_lab = var_hat_lab * n_lab
        # raw_var_hat_lab < 1e-12 is its own trigger (not just relative to
        # raw_var_lab) -- see _walsh_theta_lambda_replicates' CALLER GUARD note for why an
        # exactly-degenerate Y_hat_lab needs the same fallback as an
        # exactly-degenerate Y_lab: either side being ~0 makes cov_lab_hatlab
        # trivially ~0 too, so a relative-only check can miss it.
        if n_lab <= 1 or raw_var_hat_lab < 1e-12 or raw_var_lab < raw_var_hat_lab * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _analytic_mean_lambda_replicates(Y_lab, Y_hat_lab, var_unlab, n_lab)
        lam_power_tuned = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)

        if label_shift_robust:
            w_rep = _label_shift_blend_weight(f_hat_lab, f_unlab, var_hat_lab, var_unlab, _LABEL_SHIFT_SHRINKAGE_K)
            lam = w_rep * lam_power_tuned + (1.0 - w_rep) * 1.0
        else:
            lam = lam_power_tuned

    estimate = f_lab + lam * (f_unlab - f_hat_lab)
    var_estimate = max(var_lab + lam * lam * (var_unlab + var_hat_lab) - 2.0 * lam * cov_lab_hatlab, 0.0)
    if power_tune and lam_replicates is not None and len(lam_replicates) > 1:
        if label_shift_robust:
            # Full-chain lambda-uncertainty inflation: bootstrap the ENTIRE
            # raw-ratio -> adaptive-shrink -> label-shift-blend pipeline
            # (not just the raw ratio), since a first-order "hold the
            # blend weight fixed" approximation was found (empirically) to
            # under-cover -- see _label_shift_blended_lambda_replicates's
            # docstring.
            w_shrink = n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
            target = 1.0 - float(np.mean(lam_replicates < 0.5))
            lam_blend_replicates = _label_shift_blended_lambda_replicates(
                Y_lab, Y_hat_lab, f_unlab, var_unlab, n_lab, w_shrink, target, _LABEL_SHIFT_SHRINKAGE_K,
            )
            var_estimate += _lambda_var_inflation(f_unlab - f_hat_lab, lam_blend_replicates)
        else:
            var_estimate += _lambda_var_inflation(f_unlab - f_hat_lab, lam_replicates)
    se = float(np.sqrt(var_estimate))
    df = max(n_lab - 1, 1)

    return estimate, se, f_unlab, f_lab, rectifier, (lam if power_tune else None), df


def _pooled_two_group_lambda(
    Y_lab_a: np.ndarray, Y_hat_lab_a: np.ndarray, Y_hat_unlab_a: np.ndarray,
    Y_lab_b: np.ndarray, Y_hat_lab_b: np.ndarray, Y_hat_unlab_b: np.ndarray,
) -> tuple[float, float]:
    """Single lambda estimated from the pooled (both groups') labeled and
    unlabeled data, instead of each group independently estimating its own
    -- see :func:`evalstats.tests._ppi_two_sample_t_interval`'s docstring
    for why this replaced per-group estimation there. Per-group lambda is
    fine under MCAR, but under MNAR (label selection correlated with an
    item's own value) it can distort the two groups' labeled subsamples
    asymmetrically, with no data for a per-group lambda to average that
    distortion out over. Pooling before estimating lambda restores that
    averaging -- the same "single global rectifier" pattern
    :func:`_ppi_two_sample` already uses via ``correct()``'s general
    bootstrap path, just computed in closed form here.

    Returns ``(lam, var_lam)``. ``var_lam`` is ``Var(lam_raw)`` from the
    pooled bootstrap replicates -- the caller needs it to build the joint
    lambda-uncertainty inflation term, since lambda is now shared between
    the two groups' point estimates: its contribution to
    ``Var(est_a - est_b)`` uses ``(r_a - r_b)**2 * var_lam`` (the
    *difference* of each group's rectifier term), not each group's term
    squared independently as :func:`_lambda_var_inflation` computes for a
    single estimator, since lambda noise is perfectly correlated between
    the two groups here."""
    # Each group is centred on its own mean before pooling: lambda* is
    # defined by within-group moments, and pooling the raw (uncentred)
    # values would let between-group separation drag lam_raw toward
    # n_all/(n_all + n_lab) as the groups separate. Centring leaves the
    # MNAR protection intact -- under MNAR the covariance signal is already
    # crushed, so there's no between-group inflation to remove.
    def _c(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x - x.mean() if x.size else x

    Y_lab = np.concatenate([_c(Y_lab_a), _c(Y_lab_b)])
    Y_hat_lab = np.concatenate([_c(Y_hat_lab_a), _c(Y_hat_lab_b)])
    Y_hat_unlab = np.concatenate([_c(Y_hat_unlab_a), _c(Y_hat_unlab_b)])
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    var_unlab = float(np.var(Y_hat_unlab, ddof=1)) / n_all if n_all > 1 else 0.0
    var_lab = float(np.var(Y_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    cov_lab_hatlab = float(np.cov(Y_lab, Y_hat_lab, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0

    denom = var_unlab + var_hat_lab
    lam_raw = min(max(cov_lab_hatlab / denom, 0.0), 1.0) if denom > 1e-12 else 1.0

    raw_var_lab = var_lab * n_lab
    raw_var_hat_lab = var_hat_lab * n_lab
    # See _walsh_theta_lambda_replicates' CALLER GUARD note for why
    # raw_var_hat_lab itself needs an absolute floor check too.
    if n_lab <= 1 or raw_var_hat_lab < 1e-12 or raw_var_lab < raw_var_hat_lab * 1e-6:
        lam_replicates = None
    else:
        lam_replicates = _analytic_mean_lambda_replicates(Y_lab, Y_hat_lab, var_unlab, n_lab)
    lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)
    var_lam = (
        float(np.var(lam_replicates, ddof=1))
        if lam_replicates is not None and len(lam_replicates) > 1
        else 0.0
    )
    return lam, var_lam


def _pooled_k_group_lambda(
    Y_lab_groups: list[np.ndarray], Y_hat_lab_groups: list[np.ndarray], Y_hat_unlab_groups: list[np.ndarray],
) -> tuple[float, float]:
    """Same idea as :func:`_pooled_two_group_lambda` (single lambda from
    all groups' pooled labeled/unlabeled data instead of each group
    independently estimating its own), generalized from 2 to k groups --
    kept as a separate function rather than widening that function's
    signature, since it has exactly one existing caller already validated
    at k=2.

    Motivated by :func:`evalstats.tests._ppi_anova_independent_f_stat`'s
    Type-I inflation under ``power_tune=True``: each group there
    independently estimates its own lambda to minimize that group's own
    reported variance using the same finite labeled sample's noisy
    moments, which is a textbook argmin-then-evaluate-at-the-argmin
    downward bias in the reported variance -- distinct from lambda's own
    sampling uncertainty (which :func:`_lambda_var_inflation` already
    corrects for). Pooling increases the effective sample lambda is
    estimated from (all groups' labeled data combined), which shrinks the
    bias. This is a different mechanism from
    :func:`_pooled_two_group_lambda`'s MNAR motivation (an
    asymmetric-distortion-cancellation argument that doesn't apply to
    ANOVA's grand-mean-centered estimand) -- just more data to estimate
    lambda from.

    Returns ``(lam, var_lam)`` -- same shape as
    :func:`_pooled_two_group_lambda`, consumed the same way by
    :func:`_analytic_mean_point_se_given_lambda` per group plus a joint
    lambda-uncertainty term built from each group's own ``r_term``."""
    # Centre each group before pooling, for the same reason
    # _pooled_two_group_lambda does: pooling uncentred values lets
    # between-group spread inflate cov_lab_hatlab more than denom, drifting
    # lam_raw upward with effect size. Centring makes lambda effect-invariant
    # for continuous/likert data; for binary, lambda still drifts with effect
    # size, correctly -- Var(Y) = p(1-p) genuinely depends on the base rate,
    # so the variance-minimising lambda really does move as group means
    # approach a boundary. See
    # simulations/investigate_pooled_k_group_lambda_{all_eval_types,binary_typeI}.py
    # for the calibration sweep.
    def _c(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x - x.mean() if x.size else x

    Y_lab = np.concatenate([_c(g) for g in Y_lab_groups])
    Y_hat_lab = np.concatenate([_c(g) for g in Y_hat_lab_groups])
    Y_hat_unlab = np.concatenate([_c(g) for g in Y_hat_unlab_groups])
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    var_unlab = float(np.var(Y_hat_unlab, ddof=1)) / n_all if n_all > 1 else 0.0
    var_lab = float(np.var(Y_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    cov_lab_hatlab = float(np.cov(Y_lab, Y_hat_lab, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0

    denom = var_unlab + var_hat_lab
    lam_raw = min(max(cov_lab_hatlab / denom, 0.0), 1.0) if denom > 1e-12 else 1.0

    raw_var_lab = var_lab * n_lab
    raw_var_hat_lab = var_hat_lab * n_lab
    # See _walsh_theta_lambda_replicates' CALLER GUARD note for why
    # raw_var_hat_lab itself needs an absolute floor check too.
    if n_lab <= 1 or raw_var_hat_lab < 1e-12 or raw_var_lab < raw_var_hat_lab * 1e-6:
        lam_replicates = None
    else:
        lam_replicates = _analytic_mean_lambda_replicates(Y_lab, Y_hat_lab, var_unlab, n_lab)
    lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)
    var_lam = (
        float(np.var(lam_replicates, ddof=1))
        if lam_replicates is not None and len(lam_replicates) > 1
        else 0.0
    )
    return lam, var_lam


def _analytic_mean_point_se_given_lambda(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray, lam: float,
) -> tuple[float, float, float, float, float, float, int]:
    """Same point-estimate/variance construction as
    :func:`_analytic_mean_point_se`, but takes ``lam`` as GIVEN (already
    estimated elsewhere -- e.g. pooled across two groups by
    :func:`_pooled_two_group_lambda`) instead of estimating its own from
    this group's data alone.

    Deliberately does NOT add lambda's own estimation-uncertainty
    inflation (:func:`_lambda_var_inflation`'s term): a caller sharing
    one lambda across multiple groups needs to add that jointly, using
    all the groups' rectifier terms together, not once per group -- see
    :func:`_pooled_two_group_lambda`'s docstring.

    Returns ``(estimate, var_estimate, f_unlab, f_lab, rectifier, r_term, df)``
    -- ``r_term = f_unlab - f_hat_lab`` (the lambda-uncertainty caller
    needs this per group to build the joint inflation term).
    """
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)
    if n_all == 0:
        raise ValueError(
            "PPI correction requires at least one unlabeled item, got n_all=0 "
            "(every item in this comparison is human-labeled, leaving no "
            "unlabeled residual for the LLM-only term). If every item is "
            "labeled, use the human labels directly instead of PPI correction."
        )

    f_unlab = float(np.mean(Y_hat_unlab))
    f_lab = float(np.mean(Y_lab))
    f_hat_lab = float(np.mean(Y_hat_lab))
    rectifier = f_lab - f_hat_lab
    r_term = f_unlab - f_hat_lab

    var_unlab = float(np.var(Y_hat_unlab, ddof=1)) / n_all if n_all > 1 else 0.0
    var_lab = float(np.var(Y_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    cov_lab_hatlab = float(np.cov(Y_lab, Y_hat_lab, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0

    estimate = f_lab + lam * r_term
    var_estimate = max(var_lab + lam * lam * (var_unlab + var_hat_lab) - 2.0 * lam * cov_lab_hatlab, 0.0)
    df = max(n_lab - 1, 1)

    return estimate, var_estimate, f_unlab, f_lab, rectifier, r_term, df


def _analytic_mean_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool, label_shift_robust: bool = False,
) -> "PPIResult":
    """Closed-form (delta-method) PPI correction for
    ``estimator_func=np.mean`` -- no bootstrap resampling at all. See
    ``correct()``'s ``backend`` parameter for when this replaces the
    percentile bootstrap.

    The percentile bootstrap needs roughly n_lab >= 50 on noisy/discrete
    real data before Type-I error settles near nominal alpha; this
    closed-form path reaches the same target by n_lab ~= 25-30, since it
    plugs sample variances directly into a known distributional form
    (Student's t, df = n_lab - 1, since the labeled-subset term is the
    variance bottleneck) instead of approximating a sampling distribution
    from a small empirical resample.

    Point estimate is identical to the bootstrap path's (same f_unlab/
    f_lab/f_hat_lab/rectifier definitions); only the variance/CI/p-value
    construction differs. The shared point-estimate/variance/df
    computation lives in :func:`_analytic_mean_point_se` (this function is
    a thin wrapper around it, building a plain t-interval).

    power_tune's lambda* has a closed form here too (the original PPI++
    derivation for a mean/OLS-type estimand, Angelopoulos/Duchi/Zrnic
    2023): minimizing Var(F_lab + lambda*(F_unlab - F_hat_lab)) over
    lambda, where F_unlab is independent of (F_lab, F_hat_lab) by the
    disjointness requirement, gives

        lambda* = Cov(F_lab, F_hat_lab) / [Var(F_unlab) + Var(F_hat_lab)]

    with F_lab/F_hat_lab/F_unlab the sample means (not raw items) -- i.e.
    Var(F_hat_lab) = var(Y_hat_lab, ddof=1)/n_lab, Cov(F_lab, F_hat_lab) =
    cov(Y_lab, Y_hat_lab, ddof=1)/n_lab (paired at the item level,
    cross-item covariance is 0 under i.i.d. sampling), Var(F_unlab) =
    var(Y_hat_unlab, ddof=1)/n_all. ``correct()``'s bootstrap path
    estimates this same quantity by resampling, as a general-purpose
    stand-in that works for arbitrary ``estimator_func``; for the mean
    specifically this closed form is exact and needs no extra bootstrap
    pass. No small-n_lab shrinkage is applied (unlike the bootstrap path's
    ``_POWER_TUNE_SHRINKAGE_C``) -- that shrinkage exists specifically to
    compensate for the bootstrap's own small-sample weakness, which this
    path doesn't have.

    ``label_shift_robust`` (default False) is passed straight through to
    :func:`_analytic_mean_point_se` -- see its docstring.
    """
    estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
        Y_lab, Y_hat_lab, Y_hat_unlab, power_tune, label_shift_robust=label_shift_robust,
    )

    if se <= 0.0:
        ci_low = ci_high = estimate
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
    else:
        t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
        ci_low, ci_high = estimate - t_crit * se, estimate + t_crit * se
        p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(estimate) / se, df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=lam,
    )


def _analytic_logit_t_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool, lo: float = 0.0, hi: float = 1.0,
    label_shift_robust: bool = False,
) -> "PPIResult":
    """Closed-form PPI correction for a [lo, hi]-bounded mean estimand, CI
    constructed on the logit scale -- the PPI analogue of
    ``evalstats.core.resampling.logit_t_ci_1d``. Built the same way
    ``_analytic_mean_correct`` is: identical point-estimate/variance
    derivation (delegated to ``_analytic_mean_point_se``, shared with
    ``_analytic_mean_correct`` -- the two differ only in how the CI is
    built from (estimate, se, df)), then a delta-method logit-scale
    t-interval instead of a plain one:

      1. Rescale (estimate, se) linearly onto [0, 1]: scaled = (x - lo) /
         (hi - lo). Valid because a linear rescale commutes with taking a
         sample mean/SE, so this does NOT require recomputing the point
         estimate/variance on rescaled arrays from scratch.
      2. logit(scaled_estimate), SE_logit = scaled_se / (scaled*(1-scaled))
         -- literally logit_t_ci_1d's own delta-method step, applied to
         PPI's corrected estimate/SE instead of a plain sample mean/SE.
      3. t-interval on the logit scale, df = max(n_lab-1, 1) -- the SAME
         df convention _analytic_mean_correct's plain t-interval uses
         (the labeled-subset term is this estimator's variance
         bottleneck either way).
      4. Back-transform via sigmoid, rescale to [lo, hi].

    Unlike logit_t_ci_1d's raw per-item values (guaranteed within [0, 1]
    once each item itself is, by that function's own stricter raw-value
    range check), PPI's corrected ESTIMATE is f_lab + lam*(f_unlab -
    f_hat_lab) -- a signed combination NOT itself constrained to [lo, hi]
    -- and CAN legitimately land outside it for a small/noisy labeled
    subset (the same phenomenon _ppi_single_wilson's docstring notes for
    its own p_hat_for_wilson clip). So instead of logit_t_ci_1d's
    raise-on-out-of-range (a real raw-data-hygiene bug there), the
    rescaled estimate here is clipped into [_LOGIT_T_BOUNDARY_EPS, 1 -
    _LOGIT_T_BOUNDARY_EPS] before the logit transform when it lands at or
    outside [0, 1] (with a UserWarning) -- reusing _ppi_single_wilson's
    established "clip before feeding a bounded-domain formula" precedent
    rather than inventing new logic. The returned ``estimate`` is left
    UN-clipped (same convention as _ppi_single_wilson: the reported point
    estimate is always the true PPI estimate; only the value fed
    internally to the bounded-domain formula is clipped). The final CI is
    clamped to [lo, hi] (matching logit_t_ci_1d's own guarantee).

    p_value uses estimate/se on the raw scale -- numerically identical to
    what ``_analytic_mean_correct`` would report on the same inputs. This
    matches ``evalstats/core/paired.py``'s classical (non-PPI)
    ``method="logit_t"`` branch, which also returns the plain paired-t-test
    p-value regardless of the logit CI construction -- only the CI's shape
    differs, never the significance test itself.

    ``label_shift_robust`` (default False) is passed straight through to
    :func:`_analytic_mean_point_se` -- see its docstring.
    """
    estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
        Y_lab, Y_hat_lab, Y_hat_unlab, power_tune, label_shift_robust=label_shift_robust,
    )

    if se <= 0.0 or not np.isfinite(se):
        ci_low = ci_high = float(np.clip(estimate, lo, hi))
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
        return PPIResult(
            estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
            p_value=p_value, lam=lam,
        )

    span = hi - lo
    scaled_est = (estimate - lo) / span
    scaled_se = se / span

    if scaled_est <= 0.0 or scaled_est >= 1.0:
        warnings.warn(
            f"_analytic_logit_t_correct: PPI-corrected estimate {estimate:.6g} is outside "
            f"[{lo:g}, {hi:g}] -- clipping toward the boundary for the logit transform. "
            "Expected occasionally for a small/noisy labeled subset (the correction term "
            "isn't itself constrained to [lo, hi] even though every individual observation "
            "is); see this function's docstring.",
            UserWarning, stacklevel=2,
        )
    clipped = float(np.clip(scaled_est, _LOGIT_T_BOUNDARY_EPS, 1.0 - _LOGIT_T_BOUNDARY_EPS))

    logit_mean = float(np.log(clipped / (1.0 - clipped)))
    se_logit = scaled_se / (clipped * (1.0 - clipped))
    t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
    # scipy.special.expit (not a raw 1/(1+exp(-x))) -- numerically stable
    # for large |logit_mean +/- t_crit*se_logit|, which a clipped-but-still-
    # near-boundary estimate can produce (se_logit blows up as clipped
    # approaches 0 or 1); a raw exp() here can overflow before the ratio
    # collapses to the correct 0.0/1.0 limit.
    lo_s = float(_sigmoid(logit_mean - t_crit * se_logit))
    hi_s = float(_sigmoid(logit_mean + t_crit * se_logit))
    ci_low = float(np.clip(lo_s * span + lo, lo, hi))
    ci_high = float(np.clip(hi_s * span + lo, lo, hi))

    t_obs = estimate / se
    p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(t_obs), df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=lam,
    )


_ANALYTIC_BACKENDS = {
    id(np.mean): _analytic_mean_correct,
    id(paired_walsh_midrank_theta): _analytic_walsh_theta_correct,
}
"""Maps ``estimator_func``'s identity (``id()``) to its closed-form
analytic corrector, for :func:`correct`'s ``backend`` dispatch. Extending
this to a new estimand requires both a corrector function matching
``_analytic_mean_correct``'s signature (``Y_lab, Y_hat_lab, Y_hat_unlab,
alpha, power_tune -> PPIResult``) and an entry here -- ``estimator_func``
and ``rectifier_func`` must resolve to the same entry, since the two-term
variance decomposition assumes the rectifier and the main estimand share
one variance/covariance model."""

_ANALYTIC_ALWAYS_PREFERRED = {id(paired_walsh_midrank_theta)}
"""``estimator_func`` identities that :func:`correct`'s ``backend="auto"``
routes to the analytic backend at every ``n_lab`` -- not only below
``_MIN_LAB_RECOMMENDED``, the threshold ``np.mean``'s entry in
:data:`_ANALYTIC_BACKENDS` still uses. ``paired_walsh_midrank_theta``
belongs here because its analytic backend beats the percentile bootstrap
on power across the full n_lab range with no calibration cost -- see
:func:`_analytic_walsh_theta_correct`'s docstring. ``np.mean`` is
deliberately not added here: its own n_lab<30 gate is independently
validated (see :func:`_analytic_mean_correct`) and this set exists so one
estimand's preference doesn't force a review of the shared
``_MIN_LAB_RECOMMENDED`` threshold."""


def resolve_arrays(
    df,
    *,
    metric_col: str,
    group_col: str,
    alignment_result,
):
    """Extract PPI arrays from a DataFrame and an AlignmentResult.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str
        Column of LLM scores (present for all rows).
    group_col : str
        Column of group labels (factor / condition).
    alignment_result : AlignmentResult
        From :func:`~evalstats.alignment.judge_alignment`.
        Its ``human_col`` attribute identifies the sparse human-label column.

    Returns
    -------
    tuple
        ``(Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab)`` as numpy arrays,
        ready to pass directly to :func:`correct`. ``Y_hat_unlab``/
        ``X_unlab`` EXCLUDE the labeled rows (disjoint from ``Y_lab``/
        ``Y_hat_lab``/``X_lab``) -- see :func:`correct`'s docstring for why
        that disjointness is required for its bootstrap to be valid.
    """
    human_col = alignment_result.human_col
    labeled_mask = df[human_col].notna()
    unlabeled = df.loc[~labeled_mask]
    Y_hat_unlab = unlabeled[metric_col].to_numpy(dtype=float)
    X_unlab     = unlabeled[group_col].to_numpy()
    Y_lab       = df.loc[labeled_mask, human_col].to_numpy(dtype=float)
    Y_hat_lab   = df.loc[labeled_mask, metric_col].to_numpy(dtype=float)
    X_lab       = df.loc[labeled_mask, group_col].to_numpy()
    return Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab


# ── Public API ────────────────────────────────────────────────────────────────

def correct(
    estimator_func: Callable,
    *,
    Y_lab,
    Y_hat_lab,
    Y_hat_unlab,
    X_lab=None,
    X_unlab=None,
    alpha: float = 0.05,
    n_boot: int = 1000,
    rng=None,
    compute_pvalue: bool = True,
    rectifier_func: Optional[Callable] = None,
    power_tune: bool = True,
    backend: str = "auto",
) -> PPIResult:
    """Correct any scalar estimator for LLM judge measurement error using PPI.

    Given a large LLM-scored dataset and a small human-annotated subset,
    this function returns a bias-corrected estimate and bootstrap CI.

    The PPI corrected estimator is:

    .. code-block:: text

        θ̂_PPI = f(Ŷ_unlab, X_unlab)    [LLM on full unlabeled set]
               + f(Y_lab,   X_lab)       [human on labeled subset]
               − f(Ŷ_lab,  X_lab)       [LLM on labeled subset]

    The last two terms form the *rectifier*: the signed difference between
    what the human and the LLM said about the same items.  When the LLM is
    unbiased the rectifier is near zero; when it is biased the rectifier
    shifts the estimate toward the truth.

    This is the ORIGINAL prediction-powered inference estimator (Angelopoulos
    et al. 2023) -- equivalently, a *power-tuning* weight λ fixed at 1 in the
    generalized family θ̂_λ = f(Y_lab, X_lab) + λ·[f(Ŷ_unlab, X_unlab) −
    f(Ŷ_lab, X_lab)] (λ=0 recovers the classical, labels-only estimator;
    λ=1 recovers the formula above). See *power_tune* below for the
    variance-minimizing choice of λ (PPI++, Angelopoulos/Duchi/Zrnic 2023),
    which is never less efficient than the classical estimator by
    construction -- fixing λ=1 unconditionally (this function's behavior
    when *power_tune=False*) has no such guarantee: a sufficiently
    uninformative or noisy LLM can make the λ=1 estimator strictly WORSE
    (wider CI, lower power) than just running the classical test on
    *Y_lab* alone, since the rectifier's own bootstrap variance is paid in
    full regardless of whether the LLM earns it back.

    A percentile bootstrap CI is computed by independently resampling the
    unlabeled set (size N) and the labeled set (size n_lab) on each draw
    and recomputing the PPI estimator. This independent resampling is only
    valid because *Y_hat_unlab* and the labeled set are assumed to be
    genuinely DISJOINT samples (the original prediction-powered inference
    setup, Angelopoulos et al. 2023) — disjoint samples are independent by
    construction, so bootstrapping them separately is exact.

    **Callers must exclude the labeled items from *Y_hat_unlab*/*X_unlab*.**
    A common mistake: if your data is one score column plus a sparse human-
    label overlay (label human-reviewed a subset of everything you already
    LLM-scored), *Y_hat_unlab* is NOT "every item's LLM score" — it's only
    the LLM scores for the items that do NOT also appear in *Y_lab*. Passing
    the full (overlapping) array instead silently breaks the independence
    this function's bootstrap relies on: the two terms then share items, so
    resampling them separately ignores their true covariance and produces a
    CI/p-value that drifts from nominal coverage as a function of n_lab
    (confirmed via simulation — see ``simulations/harness/cases/pvalues.py
    --mode ppi``'s N x N_lab calibration grid). There is no parameter here
    to opt into a "shared" mode; correctness depends entirely on the caller
    constructing a genuinely disjoint *Y_hat_unlab* up front.

    **The labeled subset (*Y_lab*) must be selected independently of its own
    outcome value — ideally a uniform random sample of the full dataset.**
    The rectifier's validity relies on the labeled items being representative
    of the population it's correcting; if which items get human-labeled is
    itself influenced by the (true or LLM-judged) score — e.g. "always
    double-check the borderline/highest-scoring responses" — this is
    literally MNAR (missing-not-at-random) selection on the outcome, and the
    correction can stay miscalibrated by a large, non-vanishing amount
    regardless of how large *n_lab* is. This is not a small-sample artifact
    that more labels fixes: confirmed via simulation to persist (30–65%
    false-positive rate against a 5% nominal target) from n_lab=15 up through
    n_lab=300 out of N=400, non-monotonically, under a labeling process that
    preferentially selects high-scoring items (see
    ``simulations/harness/cases/pvalues.py --mode ppi``'s MNAR-labeling
    sweep). A stratified/local-rectifier mitigation was prototyped and does
    reduce the worst cells substantially at large n_lab (≳80), but only at
    the cost of measurably worse calibration on the common, correctly-random-
    sampled case — not worth the trade given the fix that actually works for
    free: sample which items to label uniformly at random. There is
    currently no supported way to opt into a stratified/propensity-adjusted
    rectifier for non-random labeling; random sampling of the labeled subset
    is a hard requirement for this function's calibration guarantee, not a
    recommendation to weigh against convenience.

    Parameters
    ----------
    estimator_func : callable
        ``f(Y) → float`` or ``f(Y, X) → float``.  Receives numpy arrays;
        must return a scalar.  X is forwarded only when *X_lab* / *X_unlab*
        are supplied.
    Y_lab : array-like, shape (n_lab,)
        Human-annotated scores for the labeled subset.
    Y_hat_lab : array-like, shape (n_lab,)
        LLM scores for the same items as *Y_lab* (paired, same order).
    Y_hat_unlab : array-like, shape (N,)
        LLM scores for the UNLABELED dataset only — i.e. every item that
        does NOT also appear in *Y_lab*/*Y_hat_lab*. Must be disjoint from
        the labeled set; see the warning above.
    X_lab : array-like, shape (n_lab, ...), optional
        Covariates / condition labels for the labeled subset.
        When provided, passed as the second argument to *estimator_func*,
        indexed consistently with Y.  Requires *X_unlab* to also be given.
    X_unlab : array-like, shape (N, ...), optional
        Covariates / condition labels for the full dataset.
        Required when *X_lab* is provided.
    alpha : float
        Significance level; ``1 − alpha`` gives the CI width (default 0.05).
    n_boot : int
        Bootstrap resamples (default 1000).
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.
    compute_pvalue : bool
        Compute a two-sided p-value for H₀: θ = 0 (default True).
    rectifier_func : callable, optional
        Alternative estimator used for the rectifier terms ``f(Y_lab)`` and
        ``f(Ŷ_lab)`` only.  When *None* (default), *estimator_func* is used
        for all three terms.  Providing a different function (e.g. ``np.mean``
        when *estimator_func* is ``np.median``) can improve bootstrap
        calibration for non-smooth estimands like the median.
    power_tune : bool
        When *True* (the default), use PPI++'s variance-minimizing power-
        tuning weight λ instead of fixing λ=1 (the original, 2023 PPI
        estimator -- pass *False* to reproduce it exactly). λ̂ is estimated
        from a bootstrap replicate draw -- λ̂ = Ĉov(b_lab, b_hat_lab) /
        V̂ar(b_unlab − b_hat_lab), where ``b_unlab``/``b_lab``/``b_hat_lab``
        are per-replicate ``f(Ŷ_unlab)``/``f(Y_lab)``/``f(Ŷ_hat_lab)`` draws
        -- rather than a closed-form gradient derivation specific to one
        model family (the PPI++ paper derives these for OLS/logistic/
        quantile regression; this bootstrap-plug-in version is a
        general-purpose stand-in that works for *any* ``estimator_func``,
        generalizing the same variance-minimization argument). A second,
        independent bootstrap draw then builds the percentile CI at that
        fixed λ̂ -- estimating λ̂ from the same draw used for the CI
        measurably undercovers, since λ̂ ends up partly optimized against
        noise specific to that one draw ("double dipping"); the split
        removes that circularity at the cost of one extra bootstrap pass.
        λ̂ is clipped to ``[0, 1]``, then shrunk by an n_lab-dependent
        amount (see ``_POWER_TUNE_SHRINKAGE_C``) toward an adaptive target
        estimated from the same bootstrap draw, rather than a fixed target
        of 1: the target is ``1 - P(λ̂ < 0.5)``, so confidently-informative
        data pulls it toward 1 (the original fixed behavior),
        confidently-uninformative data pulls it toward 0 (the classical
        labels-only estimate, ``human_estimate``), and ambiguous data
        lands near 0.5 -- an empirical refinement of the published PPI++
        derivation, see ``simulations/out/results_why_ppi_shrink_1_over_0.md``.
        Falls back to a target of 1 when ``Y_lab`` itself is near-degenerate
        (its own bootstrap variance ≈0, so it can't reveal covariance no
        matter how it's resampled), and to λ=1 unchanged if the bootstrap
        variance in λ̂'s own denominator is degenerate. ``PPIResult.lam``
        reports the (shrunk) value actually used.

        The reported CI/SE also account for λ̂ being estimated rather than
        fixed (see :func:`_lambda_var_inflation`) -- treating a data-driven
        λ̂ as a known constant would understate variance, worst at small
        n_lab.

        ``kruskal``/``anova``/``friedman``/``bootstrap_t``/``lmm*`` and the
        MNAR-experimental rectifiers use bespoke bootstrap/closed-form code
        of their own and are unaffected by ``power_tune``. For
        kruskal/anova/friedman this is deliberate: power-tuning does not
        transfer to their variance-like, quadratic-form estimand, whose
        λ=0 endpoint is the raw, judge-biased estimate rather than a safe
        classical fallback -- see ``simulations/harness/README.md``'s
        "PPI++ power-tuning" section. ``mj_floor`` is not on this list:
        ``evalstats.tests._ppi_paired_mj_floor`` delegates its point
        estimate, variance, and λ directly to :func:`_analytic_mean_point_se`,
        so it already gets the same adaptive-target shrinkage as every
        other caller of that function.
    backend : {"auto", "bootstrap", "analytic"}
        How to build the CI/p-value. "bootstrap" is the percentile-
        resampling method described above, unconditionally. "analytic" is
        a closed-form (delta-method / Hajek-projection, depending on the
        estimand) alternative for ``estimator_func`` registered in
        :data:`_ANALYTIC_BACKENDS` (``np.mean`` -- see
        :func:`_analytic_mean_correct` -- and
        :func:`paired_walsh_midrank_theta` -- see
        :func:`_analytic_walsh_theta_correct`) with no covariates
        (``X_lab``/``X_unlab`` both None) and ``rectifier_func`` matching
        ``estimator_func`` (or ``None``); raises ``ValueError`` if
        requested for anything else. "auto" (the default) uses "analytic"
        when it's applicable and either ``n_lab < 30`` (below which the
        percentile bootstrap is known to undercover) or ``estimator_func``
        is in :data:`_ANALYTIC_ALWAYS_PREFERRED` (currently just
        ``paired_walsh_midrank_theta``, whose analytic backend beats the
        bootstrap on power at every n_lab -- see that constant's
        docstring), otherwise falls back to "bootstrap" and, if
        ``n_lab < 30`` there too, emits a ``UserWarning`` instead of
        silently returning an under-covering interval.

        On noisy/discrete real data, the percentile bootstrap needs
        n_lab >~ 50 before Type-I error settles near nominal alpha, while
        an applicable analytic path reaches the same target by
        n_lab ~= 25-30, since it plugs sample variances directly into a
        known (Student's t) distributional form instead of approximating a
        sampling distribution from a small empirical resample.

    Returns
    -------
    PPIResult

    Raises
    ------
    ValueError
        If inputs are malformed (invalid ``alpha``/``n_boot``, inconsistent
        lengths, empty arrays, non-finite values), or if exactly one of
        *X_lab* / *X_unlab* is supplied.

    Examples
    --------
    >>> import numpy as np
    >>> import evalstats as es
    >>>
    >>> def mean_diff(Y, X):
    ...     \"\"\"Mean score under condition A minus condition B.\"\"\"
    ...     return float(Y[X == "A"].mean() - Y[X == "B"].mean())
    >>>
    >>> result = es.ppi.correct(
    ...     estimator_func=mean_diff,
    ...     Y_lab=gold_df["human_score"].values,
    ...     Y_hat_lab=gold_df["llm_score"].values,
    ...     Y_hat_unlab=large_df["llm_score"].values,
    ...     X_lab=gold_df["condition"].values,
    ...     X_unlab=large_df["condition"].values,
    ...     alpha=0.05,
    ... )
    >>> result.summary()
    """
    rng = np.random.default_rng(rng)

    # Validate scalar control parameters early for clearer errors.
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as e:
        raise ValueError(f"alpha must be a finite float in (0, 1); got {alpha!r}.") from e
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}.")
    if int(n_boot) <= 0:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}.")
    n_boot = int(n_boot)

    # ── Coerce inputs ─────────────────────────────────────────────────────────
    Y_lab       = np.asarray(Y_lab,       dtype=float)
    Y_hat_lab   = np.asarray(Y_hat_lab,   dtype=float)
    Y_hat_unlab = np.asarray(Y_hat_unlab, dtype=float)

    if Y_lab.ndim == 0 or Y_hat_lab.ndim == 0 or Y_hat_unlab.ndim == 0:
        raise ValueError(
            "Y_lab, Y_hat_lab, and Y_hat_unlab must be at least 1-D arrays."
        )
    if Y_lab.ndim != Y_hat_lab.ndim:
        raise ValueError(
            f"Y_lab and Y_hat_lab must have the same ndim "
            f"(got {Y_lab.ndim} vs {Y_hat_lab.ndim})."
        )
    if Y_hat_unlab.ndim != Y_hat_lab.ndim:
        raise ValueError(
            f"Y_hat_unlab and Y_hat_lab must have the same ndim "
            f"(got {Y_hat_unlab.ndim} vs {Y_hat_lab.ndim})."
        )
    if Y_lab.shape[1:] != Y_hat_lab.shape[1:]:
        raise ValueError(
            f"Y_lab and Y_hat_lab must have matching trailing shape "
            f"(got {Y_lab.shape[1:]} vs {Y_hat_lab.shape[1:]})."
        )
    if Y_hat_unlab.shape[1:] != Y_hat_lab.shape[1:]:
        raise ValueError(
            f"Y_hat_unlab and Y_hat_lab must have matching trailing shape "
            f"(got {Y_hat_unlab.shape[1:]} vs {Y_hat_lab.shape[1:]})."
        )

    if X_lab is not None:
        X_lab = np.asarray(X_lab)
    if X_unlab is not None:
        X_unlab = np.asarray(X_unlab)

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(Y_lab) != len(Y_hat_lab):
        raise ValueError(
            f"Y_lab and Y_hat_lab must have the same length "
            f"(got {len(Y_lab)} vs {len(Y_hat_lab)})"
        )
    if (X_lab is None) != (X_unlab is None):
        raise ValueError(
            "Provide both X_lab and X_unlab, or neither. "
            "Exactly one was supplied."
        )
    if X_lab is not None and len(X_lab) != len(Y_lab):
        raise ValueError(
            f"X_lab must have the same length as Y_lab "
            f"(got {len(X_lab)} vs {len(Y_lab)})"
        )
    if X_unlab is not None and len(X_unlab) != len(Y_hat_unlab):
        raise ValueError(
            f"X_unlab must have the same length as Y_hat_unlab "
            f"(got {len(X_unlab)} vs {len(Y_hat_unlab)})"
        )

    if len(Y_lab) == 0:
        raise ValueError("Y_lab and Y_hat_lab must be non-empty.")
    if len(Y_hat_unlab) == 0:
        raise ValueError(
            "Y_hat_unlab must be non-empty. This usually means every item is "
            "already labeled -- PPI has no unlabeled pool left to extrapolate "
            "the correction to. With 100% human labels, just run a classical "
            "test directly on Y_lab instead of PPI."
        )

    if not np.all(np.isfinite(Y_lab)):
        raise ValueError("Y_lab contains non-finite values (NaN/inf).")
    if not np.all(np.isfinite(Y_hat_lab)):
        raise ValueError("Y_hat_lab contains non-finite values (NaN/inf).")
    if not np.all(np.isfinite(Y_hat_unlab)):
        raise ValueError("Y_hat_unlab contains non-finite values (NaN/inf).")

    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    _rect_fn = rectifier_func if rectifier_func is not None else estimator_func

    # ── backend dispatch (auto/bootstrap/analytic) ────────────────────────────
    if backend not in ("auto", "bootstrap", "analytic"):
        raise ValueError(f"backend must be 'auto', 'bootstrap', or 'analytic'; got {backend!r}.")
    _analytic_fn = _ANALYTIC_BACKENDS.get(id(estimator_func))
    _analytic_available = (
        _analytic_fn is not None and id(_rect_fn) == id(estimator_func)
        and X_lab is None and X_unlab is None
    )
    if backend == "analytic" and not _analytic_available:
        raise ValueError(
            "backend='analytic' requires estimator_func to be one of the registered analytic "
            "estimands (np.mean, or evalstats.ppi.paired_walsh_midrank_theta -- see "
            "_ANALYTIC_BACKENDS), with rectifier_func matching estimator_func (or None) and no "
            f"covariates; got estimator_func={estimator_func!r}, rectifier_func={rectifier_func!r}, "
            f"X_lab={'given' if X_lab is not None else None}."
        )
    use_analytic = backend == "analytic" or (
        backend == "auto" and _analytic_available and (
            n_lab < _MIN_LAB_RECOMMENDED or id(estimator_func) in _ANALYTIC_ALWAYS_PREFERRED
        )
    )
    if not use_analytic and n_lab < _MIN_LAB_RECOMMENDED:
        warnings.warn(
            f"PPI bootstrap CI/p-value with only {n_lab} labeled items (recommend >= "
            f"{_MIN_LAB_RECOMMENDED}) is known to undercover -- Type-I error above nominal "
            "alpha should be expected."
            + ("" if _analytic_available else " No closed-form 'analytic' backend is available "
               "for this estimator_func; consider collecting more labels."),
            UserWarning, stacklevel=2,
        )
    if use_analytic:
        return _analytic_fn(Y_lab, Y_hat_lab, Y_hat_unlab, alpha, power_tune)

    # ── Point estimate (lambda=1 terms; combined into `estimate` below) ──────
    f_unlab   = _call(estimator_func, Y_hat_unlab, X_unlab)
    f_lab     = _call(_rect_fn,       Y_lab,       X_lab)
    f_hat_lab = _call(_rect_fn,       Y_hat_lab,   X_lab)
    rectifier = f_lab - f_hat_lab

    # ── Bootstrap replicates ──────────────────────────────────────────────────
    # Fast path: when there are no covariates and both functions are one of
    # the built-ins actually used by this codebase's internal PPI dispatch
    # (np.mean / np.median / paired_walsh_midrank_theta), the whole
    # bootstrap batches over an added replicate axis instead of a Python
    # loop with n_boot scalar calls each -- this matters most for
    # np.median (which re-sorts on every call) and paired_walsh_midrank_
    # theta (an O(n log n) sort+searchsorted per call -- see that
    # function's docstring for why it NEEDS this fast path to be practical
    # at all, unlike the O(n^2) construction it replaced). Falls back to
    # the general per-replicate loop for arbitrary user-supplied estimator
    # functions or when X_lab/X_unlab are provided.
    #
    # Factored into a helper (drawing ONE full set of n_boot replicates per
    # call) because power_tune needs TWO independent draws -- see below.
    _fast_batch = {
        id(np.mean): lambda a: a.mean(axis=1),
        id(np.median): lambda a: np.median(a, axis=1),
        id(paired_walsh_midrank_theta): _walsh_theta_batch,
    }
    fast_est = _fast_batch.get(id(estimator_func)) if X_unlab is None else None
    fast_rect = _fast_batch.get(id(_rect_fn)) if X_lab is None else None

    # Smoothed-bootstrap jitter (see _tie_jitter_scale). Applied
    # unconditionally, not just for np.median: _tie_jitter_scale is
    # self-scaling from the DATA alone (min gap between distinct values /
    # 20), so it's already a near-zero no-op on high-resolution continuous
    # data and only becomes meaningfully sized on coarse/discrete data
    # (e.g. binary {0,1} scores) -- no need to special-case which
    # estimator_func is in use. Originally gated to np.median only
    # (bootstrapping a median under ties is a classically degenerate
    # combination -- most resamples land on the identical repeated value),
    # but a mean-type estimator on a near-boundary binary proportion has an
    # analogous (if less severe) failure mode: a percentile bootstrap of a
    # discrete, boundary-adjacent proportion is skewed/undercovers. Confirmed
    # via simulation this helps ttest's covariate-based construction
    # (_ppi_two_sample, which never reaches the analytic backend and so
    # never got jitter before this) on binary scenarios where differential
    # judge bias pushes one group's score distribution toward 0 or 1 -- see
    # simulations/out/results_why_ppi_shrink_1_over_0.md's ttest-binary
    # addendum. 0.0 disables jitter (np.random.normal(0, 0, ...) is exactly
    # a no-op, so this is safe to add unconditionally below).
    _jitter_unlab = _tie_jitter_scale(Y_hat_unlab)
    _jitter_labpair = _tie_jitter_scale(np.concatenate([Y_lab, Y_hat_lab]))

    def _draw_replicates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_unlab_arr = np.empty(n_boot)
        b_lab_arr = np.empty(n_boot)
        b_hat_lab_arr = np.empty(n_boot)
        if fast_est is not None and fast_rect is not None:
            chunk_size = max(1, min(n_boot, 4096, max(1, int(2_000_000 // max(n_all, n_lab, 1)))))
            start = 0
            while start < n_boot:
                stop = min(start + chunk_size, n_boot)
                m = stop - start
                idx_all = rng.integers(0, n_all, size=(m, n_all))
                idx_lab = rng.integers(0, n_lab, size=(m, n_lab))
                resampled_unlab = Y_hat_unlab[idx_all]
                resampled_lab = Y_lab[idx_lab]
                resampled_hat_lab = Y_hat_lab[idx_lab]
                if _jitter_unlab > 0.0:
                    resampled_unlab = resampled_unlab + rng.normal(0.0, _jitter_unlab, size=resampled_unlab.shape)
                if _jitter_labpair > 0.0:
                    resampled_lab = resampled_lab + rng.normal(0.0, _jitter_labpair, size=resampled_lab.shape)
                    resampled_hat_lab = resampled_hat_lab + rng.normal(0.0, _jitter_labpair, size=resampled_hat_lab.shape)
                b_unlab_arr[start:stop]   = fast_est(resampled_unlab)
                b_lab_arr[start:stop]     = fast_rect(resampled_lab)
                b_hat_lab_arr[start:stop] = fast_rect(resampled_hat_lab)
                start = stop
        else:
            for b in range(n_boot):
                idx_all = rng.integers(0, n_all, n_all)
                idx_lab = rng.integers(0, n_lab, n_lab)
                Xa_b = X_unlab[idx_all] if X_unlab is not None else None
                Xl_b = X_lab[idx_lab]   if X_lab   is not None else None
                Yl = Y_hat_unlab[idx_all]
                Ya = Y_lab[idx_lab]
                Yb = Y_hat_lab[idx_lab]
                if _jitter_unlab > 0.0:
                    Yl = Yl + rng.normal(0.0, _jitter_unlab, size=Yl.shape)
                if _jitter_labpair > 0.0:
                    # ONE draw, shared by both halves of the labeled pair.
                    # Ya (human label) and Yb (judge score) are the SAME items
                    # in the same order, and the rectifier is a DIFFERENCE
                    # between statistics computed on them, so jittering them
                    # independently destroys the item-level coupling the
                    # rectifier depends on. For a mid-rank estimand that is
                    # catastrophic and SCALE-INVARIANT: any non-zero
                    # independent noise resolves a tied pair's 0.5 to 0-or-1,
                    # so shrinking the jitter does not help (verified: at
                    # 5e-6, 10,000x smaller, Likert Type-I is still 0.000).
                    # Measured on Likert/good-judge cells: independent draws
                    # give Type-I 0.011 and a bootstrap SE 2.3x the true
                    # sampling SD; one shared draw restores both.
                    # Mean-type rectifiers are unaffected by the bug (variance
                    # scales with s^2, no tie-resolution step) which is why
                    # ttest/paired_t never showed it -- and they keep their
                    # smoothing here, since each half is still jittered.
                    _j = rng.normal(0.0, _jitter_labpair, size=Ya.shape)
                    Ya = Ya + _j
                    Yb = Yb + _j
                b_unlab_arr[b]   = _call(estimator_func, Yl, Xa_b)
                b_lab_arr[b]     = _call(_rect_fn,       Ya, Xl_b)
                b_hat_lab_arr[b] = _call(_rect_fn,       Yb, Xl_b)
        return b_unlab_arr, b_lab_arr, b_hat_lab_arr

    # ── Power tuning (PPI++, Angelopoulos/Duchi/Zrnic 2023) ──────────────────
    # lambda* minimizes Var(f_lab + lambda*(f_unlab - f_hat_lab)); since the
    # unlabeled replicate is drawn from an independent resample of the
    # (disjoint) unlabeled set, Cov(b_lab, b_unlab) = 0 in the bootstrap
    # distribution, which reduces the minimizer to a single covariance/
    # variance ratio -- see correct()'s power_tune parameter docstring for
    # the full derivation.
    #
    # Two independent bootstrap draws are used when power_tune=True: one
    # (b1_*) only to estimate lambda, a second, fresh one (b2_*) only to
    # build the percentile CI at that now-fixed lambda. Estimating lambda
    # and building its CI from the same replicates measurably undercovers
    # nominal coverage, since lambda ends up partly optimized away noise
    # specific to that one bootstrap draw ("double dipping"); splitting the
    # two draws removes that circularity at the cost of one extra bootstrap
    # pass (still cheap via the fast-batch path above). power_tune=False
    # needs only one draw.
    #
    # lambda is then shrunk by an n_lab-dependent amount toward an ADAPTIVE
    # target (see _POWER_TUNE_SHRINKAGE_C) rather than a fixed target of 1.
    # Some shrinkage is needed regardless of target: without it,
    # power_tune=True's Type-I error runs measurably worse than
    # power_tune=False's baseline at small n_lab, since the percentile
    # bootstrap CI of a small sample is itself mildly anti-conservative,
    # and vanilla PPI's fixed λ=1 masks that by always blending in the
    # large, well-behaved unlabeled-sample bootstrap -- power-tuning, by
    # correctly identifying an uninformative judge and shrinking λ toward
    # 0, leans more on that same small-sample bootstrap, unmasking its
    # pre-existing weakness. But shrinking specifically TOWARD 1
    # conflates "λ̂ is imprecise because n_lab is small" with "the true λ
    # is probably close to 1" -- there's no reason the second should
    # follow from the first, and a fixed shrink-to-1 target costs real
    # power against a genuinely uninformative judge (confirmed via
    # simulation) with no matching Type-I benefit in that regime. This is
    # not part of the published PPI++ derivation -- it's an empirical
    # patch for a bootstrap-construction limitation this codebase already
    # had, layered on top; see simulations/out/
    # results_why_ppi_shrink_1_over_0.md for the investigation behind the
    # adaptive-target version below.
    lam: Optional[float] = None
    if power_tune:
        b1_unlab, b1_lab, b1_hat_lab = _draw_replicates()
        denom = float(np.var(b1_unlab - b1_hat_lab, ddof=1))
        if denom > 1e-12:
            lam_raw = float(np.cov(b1_lab, b1_hat_lab, ddof=1)[0, 1] / denom)
            lam_raw = min(max(lam_raw, 0.0), 1.0)
        else:
            lam_raw = 1.0  # degenerate bootstrap variance -- fall back, don't divide by ~0.

        # Adaptive shrinkage -- see _adaptive_shrink_lambda's docstring for
        # the shared rationale, and _bootstrap_batch_lambda_replicates for
        # how the SAME b1 draw already computed above gets turned into
        # replicate lambda estimates (no extra bootstrap draws). Falls
        # back to target=1 when Y_lab itself is near-degenerate (a
        # near-constant labeled sample can never reveal covariance no
        # matter how it's resampled, so a "confidently near 0" signal
        # there is an artifact, not evidence) -- the analytic backends'
        # degenerate guards mirror this same logic.
        var_lab = float(np.var(Y_lab, ddof=1)) if n_lab > 1 else 0.0
        var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) if n_lab > 1 else 0.0
        # var_hat_lab < 1e-12 is its own trigger too -- see
        # _walsh_theta_lambda_replicates' CALLER GUARD note for why an
        # exactly-degenerate Y_hat_lab needs the same fallback as an
        # exactly-degenerate Y_lab (either side being ~0 makes the raw
        # covariance ratio trivially ~0, not genuinely informative).
        if n_lab <= 1 or var_hat_lab < 1e-12 or var_lab < var_hat_lab * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _bootstrap_batch_lambda_replicates(b1_lab, b1_hat_lab, b1_unlab)
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)
        b2_unlab, b2_lab, b2_hat_lab = _draw_replicates()
        estimate = f_lab + lam * (f_unlab - f_hat_lab)
        boots = b2_lab + lam * (b2_unlab - b2_hat_lab)
        # See _lambda_var_inflation's docstring: `lam` is a single point
        # value (estimated once from b1) applied uniformly across every b2
        # replicate, so `boots`' spread reflects zero uncertainty from
        # lambda's own estimation -- convolve it back in as independent
        # noise, rather than re-deriving lambda per b2 replicate (a full
        # nested bootstrap).
        extra_var = _lambda_var_inflation(f_unlab - f_hat_lab, lam_replicates)
        if extra_var > 0.0:
            boots = boots + rng.normal(0.0, np.sqrt(extra_var), size=boots.shape)
    else:
        b_unlab_arr, b_lab_arr, b_hat_lab_arr = _draw_replicates()
        estimate = f_unlab + (f_lab - f_hat_lab)
        boots = b_unlab_arr + (b_lab_arr - b_hat_lab_arr)

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    # ── p-value ───────────────────────────────────────────────────────────────
    p_value: Optional[float] = None
    if compute_pvalue:
        # Proportion of bootstrap draws on each side of 0; two-sided.
        p_value = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
        p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=float(estimate),
        ci_low=lo,
        ci_high=hi,
        alpha=alpha,
        llm_estimate=float(f_unlab),
        human_estimate=float(f_lab),
        rectifier=float(rectifier),
        p_value=p_value,
        lam=lam,
    )
