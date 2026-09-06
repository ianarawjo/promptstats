"""ci_paired case: paired-difference CI coverage, synthetic and/or real benchmark data.

One engine operating over ``scenarios.CIPairSource`` objects, so the same
CI-method dispatch and report format covers synthetic paired distributions
(``scenarios/synthetic.py``) and real shared-item OpenEval / Inspect AI
corpora (``scenarios/real_data.py``) alike.

Methods compared
-----------------
bootstrap, bca, bayes_bootstrap, smooth_bootstrap, bootstrap_t (all eval
types, statistic=mean or median); t_interval, logit_t, nig, el (non-binary,
statistic=mean only); newcombe_mover, mj_floor, tango_scc, bayes_indep_comp,
bayes_paired_comp (binary, statistic=mean only). tango_scc is the
continuity-corrected "SCC-S" (c=0.125) score interval from Chang et al.
(2024, J. Applied Statistics 51(1):139-152) -- see
evalstats.core.resampling.tango_scc_paired_ci's docstring for the closed-form
quartic derivation (verified against R's PropCIs::scoreci.mp reference
implementation) and a note on a sign-convention discrepancy found between
the paper's prose and its working equations.

Two variants were tried and abandoned after simulation, kept here as notes
so the same dead ends aren't re-explored:
- tango_hybrid: plain mj_floor, switching to tango_scc only when the
  observed discordant pairs looked imbalanced. Its worst-case coverage
  barely improved on plain mj_floor's, because the residual failures were
  samples that *looked* balanced by chance despite a lopsided true
  generating process -- unrescuable by any method conditioning on the
  observed discordant split (confirmed bayes_paired_comp fails the same way
  on that same subset).
- wilson_discordant: treats the discordant count m=n10+n01 as the effective
  sample size and puts a Wilson interval on the discordant split q=n10/m,
  transformed back via d=m*(2q-1)/n. This under-covers catastrophically
  (down to ~0.40-0.65) on exactly the lopsided-discordance scenarios it was
  meant to fix, because it conditions on m without propagating m's own
  binomial sampling variance -- worst exactly when q_hat sits near a
  boundary (0 or 1), i.e. when discordance is sparse or lopsided.

Known exceptions (see simulations/harness/README.md):
- Flat (non-nested) mode's real-data pairs (openeval/inspect/wmt_da_paired/
  real) only support R=1 (single run per item); multi-run real pairs need
  --nested-mode instead (see below).

--nested-mode (ported from simulations/sim_compare_boot_nested.py, pairwise
phase) sweeps multi-run paired scenarios, reporting flat (cell-mean
reduction) and "*_nested"/"*_flat"/"*_multirun_*" CI methods side by side.
Supports synthetic data (parameterised by run_noise_frac) and real
multi-run data from Inspect AI logs (--data-source inspect, via
build_real_pair_sources_nested; see nested_real_official_args()). Always
uses statistic="mean" (no median support). There is no separate ci_nested
case -- see cases/ci_single.py's --nested-mode for the single-sample
analogue.
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import itertools
import multiprocessing as _mp
import os
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        bootstrap_t_ci_1d,
        bca_interval_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        bootstrap_means_1d,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
        resolve_resampling_method,
        t_interval_ci_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        el_ci_1d,
        mj_floor_paired_ci,
        tango_scc_paired_ci,
        mj_unfloored_paired_ci,
        newcombe_mover_paired_ci,
        bonett_price_paired_ci,
        bonett_price_paired_ci_flat,
        bonett_price_paired_ci_multirun_cluster,
        bonett_price_paired_ci_multirun_shrunk,
        mj_floor_paired_ci_flat,
        mj_floor_paired_ci_multirun_effective,
        mj_floor_paired_ci_multirun_cluster,
        clustered_score_paired_ci,
        mj_floor_paired_ci_multirun_moments,
        bayes_paired_diff_ci,
    )
    from evalstats.core.stats_utils import interval_score, rescaled_ci

from ..latex_tables import (
    booktabs_table,
    coverage_cell,
    escape_latex,
    mark_best_and_runnerup,
    report_eval_type_group,
)
from ..scenarios import CIPairSource, EVAL_TYPES, DEFAULT_EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    RUN_NOISE_FRACS_DEFAULT,
    build_pair_sources,
)
from ..scenarios.real_data import DEFAULT_INSPECT_CSV, PAIR_SOURCES as REAL_PAIR_SOURCES, build_real_pair_sources, build_real_pair_sources_nested
from ..methods import (
    BOOTSTRAP_METHODS as METHODS,
    BAYES_BOOTSTRAP,
    T_INTERVAL,
    LOGIT_T,
    NIG,
    EL,
    MJ_FLOOR,
    TANGO_SCC,
    TANGO_EXACT,
    MJ_UNFLOORED,
    BONETT_PRICE,
    NEWCOMBE_MOVER,
    BAYES_PAIR_INDEP,
    BAYES_PAIR_PAIRED,
    WALD_PAIR_INDEP,
    PAIRWISE_EXTRA_METHODS,
    LOGIT_T_DITHER,
    SMOOTH_BOOTSTRAP_DITHER,
    DITHER_EXTRA_METHODS,
    PAIR_DIFF_NESTED_METHODS,
    BOOTSTRAP_DIFF_NESTED,
    BAYES_DIFF_NESTED,
    SMOOTH_DIFF_NESTED,
    BINARY_PAIR_FLAT_METHODS,
    MJ_FLOOR_FLAT,
    NEWCOMBE_FLAT,
    BONETT_PRICE_FLAT,
    BINARY_PAIR_NESTED_METHODS,
    BINARY_PAIR_NESTED_OFFICIAL,
    MJ_FLOOR_CLUSTER,
    CLUSTERED_SCORE,
    BONETT_PRICE_CLUSTER,
    BONETT_PRICE_SHRUNK,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "ci_paired"

DATA_SOURCES = ["synthetic"] + REAL_PAIR_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]


@dataclass
class SimResult:
    """Aggregated outcome for one (source, eval_type, n, method) cell, summed
    over n_reps repetitions."""

    source: str  # "synthetic" | "openeval" | "inspect" | "real"
    label: str
    eval_type: str
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_score: float = 0.0
    """Sum of interval_score() (see evalstats.core.stats_utils) across n_reps."""
    total_pen_under: float = 0.0
    """Sum of the (2/alpha)*(lo - y) penalty for y BELOW the interval.

    Bracher, Ray, Gneiting & Reich (2021) decompose the interval score into
    the interval width (sharpness) and the penalty for observations outside
    the interval (calibration), splitting the latter into over- and
    underprediction to expose systematic bias. Kept separate from
    total_score because the mean score is ~90% width, so a method can
    under-cover badly and still post the best score -- the penalty term is
    what tracks calibration (it reproduces the MinCov ordering exactly)."""
    total_pen_over: float = 0.0
    """Sum of the (2/alpha)*(y - hi) penalty for y ABOVE the interval."""
    rejects: int = 0
    """Count of reps whose CI EXCLUDED zero, i.e. the decision "these differ".

    On null rows (delta = 0) this is the Type I error count -- identical to
    n_reps - covered there, kept as its own counter so the same field also
    gives POWER on the non-null rows, where coverage is about containing the
    true delta rather than excluding zero. evalstats' users act on this
    decision (directly, and through the simultaneous-CI/FWER path, which
    widens these intervals and decides from them), so it is reported
    alongside coverage rather than left implicit."""
    total_time: float = 0.0
    total_time_sq: float = 0.0
    is_null: bool = False
    model_a: str | None = None
    model_b: str | None = None
    benchmark_id: str | None = None
    corpus_size: int | None = None
    true_diff: float | None = None
    run_noise_frac: float = 0.0
    """f_run for --nested-mode rows (scenarios.synthetic.build_pair_sources' run_noise_fracs); 0.0 otherwise."""
    runs: int = 1
    """Runs per input for --nested-mode rows; 1 (flat) otherwise."""


def _stat(values: np.ndarray, statistic: str = "mean") -> float:
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    from scipy.stats import norm

    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _bayes_indep_comp_ci(a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Independent Beta-posteriors CI for paired binary difference p(A=1)-p(B=1)."""
    a_bin = (a >= 0.5).astype(float)
    b_bin = (b >= 0.5).astype(float)
    post_a = rng.beta(float(np.sum(a_bin)) + 1.0, float(a_bin.shape[0] - np.sum(a_bin)) + 1.0, size=num_samples)
    post_b = rng.beta(float(np.sum(b_bin)) + 1.0, float(b_bin.shape[0] - np.sum(b_bin)) + 1.0, size=num_samples)
    diff = post_a - post_b
    return float(np.percentile(diff, 100.0 * alpha / 2.0)), float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0)))


def _wald_indep_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Naive independent-samples Wald CI for paired binary difference p(A=1)-p(B=1).

    Computes p_A and p_B as if drawn from two INDEPENDENT samples
    (variance = p_A(1-p_A)/n + p_B(1-p_B)/n, no covariance term) and forms
    a plain normal-approximation interval on their difference -- ignoring
    that A and B are measured on the same items entirely. This is the
    textbook "wrong way" to compare matched/paired binary outcomes -- the
    frequentist analog of bayes_indep_comp's identical independence
    assumption (draw separate posteriors for p_A, p_B, subtract). Unlike
    mj_floor (which uses the discordant-pair structure)
    or even a plain paired t-interval on the per-item differences, it makes
    no use of which items overlap between A and B at all.

    When A and B are positively correlated (the common case for paired
    model/item comparisons -- easy items tend to be easy for both models),
    ignoring that correlation means overestimating the variance, so the
    dominant failure mode here is typically excess width/conservatism
    rather than undercoverage -- still a real reason to avoid it (wasted
    power to detect real differences), just a different one from the
    small-n/boundary undercoverage story that motivates avoiding `wald`/
    `wald_flat` for a single proportion.

    Parameters
    ----------
    a, b : np.ndarray
        1-D arrays of paired binary (0/1) observations, same length.
    alpha : float
        Significance level (1 - confidence level).

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        CI on p(A=1) - p(B=1), clamped to [-1, 1].
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    n = a_bin.shape[0]
    if n <= 0:
        return (0.0, 0.0)
    p_a = float(np.mean(a_bin))
    p_b = float(np.mean(b_bin))
    se = float(np.sqrt(p_a * (1.0 - p_a) / n + p_b * (1.0 - p_b) / n))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d = p_a - p_b
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _bayes_paired_comp_ci(a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Paired Bayesian CI for p(A=1)-p(B=1) -- thin wrapper around evalstats'
    production ``bayes_paired_diff_ci`` (the bivariate-normal latent model
    from bayes_evals.py), which is also what ``analyze(method="bayes_binary")``
    calls for small-N binary pairwise comparisons. Delegating here (instead of
    hand-duplicating the algorithm, as this used to) means the calibration
    sweep actually exercises the shipped code, not a frozen copy of it.
    """
    ci_low, ci_high, _prob_a_greater = bayes_paired_diff_ci(
        (a >= 0.5).astype(float), (b >= 0.5).astype(float), alpha,
        num_samples=num_samples, rng=rng,
    )
    return ci_low, ci_high


def _pairwise_ci(
    a: np.ndarray, b: np.ndarray, method: str, n_bootstrap: int, alpha: float,
    rng: np.random.Generator, *, statistic: str = "mean",
) -> tuple[float, float]:
    """Compute CI for paired mean/median difference A-B using evalstats logic."""
    n_inputs, runs = a.shape
    resolved_method = resolve_resampling_method(method, n_inputs)

    if runs >= 3:
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        observed = _stat(cell_diffs, statistic=statistic)

        if resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        if resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        if resolved_method == "bootstrap_t":
            return bootstrap_t_ci_1d(cell_diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)
        boot_stats = bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "bca":
            return bca_interval_1d(cell_diffs, observed, boot_stats, alpha, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    diffs = a.mean(axis=1) - b.mean(axis=1)
    observed = _stat(diffs, statistic=statistic)

    if resolved_method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    if resolved_method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    if resolved_method == "bootstrap_t":
        return bootstrap_t_ci_1d(diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)

    boot_stats = bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
    if resolved_method == "bca":
        return bca_interval_1d(diffs, observed, boot_stats, alpha, statistic=statistic)
    return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))


_NIG_PAIRED_DIFF_B0 = 0.0625 / 4
"""nig_ci_1d's default b0=0.0625 (prior mean of sigma^2, i.e. prior
sigma~=0.25) is calibrated for ci_single.py's own rescale span
[scale_lo, scale_hi] -- see that function's docstring: "weak knowledge
that scores live in [0, 1]". ci_paired.py instead rescales paired diffs
onto [-diff_span, diff_span] = [-(scale_hi-scale_lo), (scale_hi-scale_lo)]
(needed so a zero diff maps to 0.5, nig's own prior centre) -- TWICE as
wide a span as ci_single's own [scale_lo, scale_hi]. Reusing b0=0.0625
unchanged there implies 2^2=4x the prior variance in real diff units
(variance scales with the square of a linear rescale factor) versus what
ci_single already uses for a raw score on the same eval type, causing
persistent, substantial over-coverage that isn't a deliberate safety
margin, just an unpropagated rescale-span change. Verified directly on
likert paired diffs: coverage 0.983 (n=10, default b0) vs 0.946 (n=10,
this correction) -- the corrected version is 23% NARROWER for the same
validity, and the same ~20-30% narrowing (with coverage moving from
badly over- to essentially exactly at nominal) holds at n=30, n=100, and
on continuous data too. This restores NIG's effective prior to match
ci_single.py's own calibration point; it is not a new invented value,
just correctly propagated through the wider diff rescale."""


def _detect_dither_halfwidth(pooled: np.ndarray) -> float:
    """Auto-detect a rounding/quantization grid step from pooled raw arm
    values (both arms, one rep) and return half that step -- the dither
    half-width needed to reconstruct the pre-quantization variance that a
    paired diff of two highly-correlated arms can lose to rounding
    cancellation (see LOGIT_T_DITHER's docstring). Data-driven rather than
    eval_type-driven: labeled "continuous" data that's actually coarse
    (e.g. a judge that only emits a handful of distinct values) gets
    detected and dithered correctly; genuinely continuous data (no
    consistent grid) returns 0.0, meaning "don't dither" -- unlike a
    hardcoded width, this can't apply a jitter mismatched to the data's
    real resolution and reintroduce the boundary-clipping bias that broke
    continuous coverage when a flat +-0.5 was tried there (see
    add_dither_extras's comment).

    Takes the SMALLEST observed gap between distinct pooled values as the
    candidate step, then verifies every other gap is (within tolerance) an
    integer multiple of it -- a GCD-style check, not a "does the dominant
    gap recur >= N times" frequency threshold. The frequency-threshold
    version this replaced was blind exactly where dithering matters most:
    at small N with a peaked/near-boundary distribution (e.g. a likert
    "near-floor" shape at n=10), pooled values often collapse to just 2-3
    distinct integers -- too few gap observations for any gap to recur 3+
    times even though the grid (step=1) is completely unambiguous. This
    was found via a real regression: at n=10, icc=0.95, the near-floor and
    near-ceiling likert shapes' logit_t_dither coverage was numerically
    IDENTICAL to plain logit_t's (0.763, 0.767) -- i.e. dithering silently
    never activated on exactly the scenarios with the worst collapse.
    Requiring ALL gaps (not just the most common one) to line up on the
    candidate grid is deliberately strict: for genuinely continuous data,
    the smallest of many gaps is essentially arbitrary, and demanding every
    other gap independently land within tolerance of an integer multiple of
    it has vanishing false-positive probability (each gap has only a small
    chance of matching by coincidence, and they must ALL match)."""
    uniq = np.unique(pooled)
    if uniq.size < 2:
        return 0.0
    gaps = np.diff(uniq)
    gaps = gaps[gaps > 1e-9]
    if gaps.size == 0:
        return 0.0
    step = float(np.min(gaps))
    ratios = gaps / step
    residuals = np.abs(ratios - np.round(ratios))
    if np.max(residuals) > 0.05:
        return 0.0
    return step / 2.0


def _debiased_dither(x: np.ndarray, half: float, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Add U(-half, half) jitter to x, clip to [lo, hi], then subtract the
    closed-form bias that clipping introduces near the boundaries.

    Naive clip(x + jitter, lo, hi) is not mean-preserving for x within
    `half` of a boundary: jitter that would push x below lo (or above hi)
    piles up exactly at the boundary instead of continuing past it, pulling
    E[clipped] toward the interior. This doesn't average away with N (it's
    a fixed per-item shift, not noise), and because two paired arms
    generally have different boundary-mass compositions, the bias doesn't
    cancel in the arms' difference either -- it contaminates the estimated
    diff directly.

    Derivation: for x with distance d = x - lo from the lower bound
    (d < half means boundary-adjacent) and jitter j ~ U(-half, half),
    E[max(x+j, lo)] - x = E[max(j, -d)] = (half - d)^2 / (4*half) for
    d < half (0 otherwise) -- a standard truncated-uniform expectation.
    The upper-bound case is the mirror image. Subtracting these exactly
    recenters the expectation back on x regardless of boundary proximity.
    Reflection or rejection-resampling are worse alternatives here: for
    jitter straddling a hard boundary symmetrically, both fold the entire
    out-of-bounds half onto an exact duplicate of the in-bounds half
    (rather than restoring symmetry around x), roughly doubling the bias
    plain clipping would leave.

    The correction is re-clipped to [lo, hi] rather than left unclipped:
    pair_diffs_dither/cell_diffs_dither are per-item arrays passed directly
    into logit_t_ci_1d, which raises ValueError on inputs meaningfully
    outside [0, 1] after rescaling, and with n independent items the
    chance that at least one exceeds tolerance grows with n. Re-clipping
    trades away some of the bias correction for boundary-adjacent items in
    exchange for guaranteed validity -- a value exactly at a hard boundary
    can't be made simultaneously unbiased and contained in [lo, hi] by any
    deterministic remapping of a jitter that spans past that boundary;
    containment is the non-negotiable constraint here since violating it
    corrupts the whole interval for that rep, rather than degrading
    gracefully.
    """
    if half <= 0:
        return x
    jitter = rng.uniform(-half, half, size=x.shape)
    raw = np.clip(x + jitter, lo, hi)
    d_lo = x - lo
    d_hi = hi - x
    bias_lo = np.where(d_lo < half, (half - d_lo) ** 2 / (4 * half), 0.0)
    bias_hi = np.where(d_hi < half, (half - d_hi) ** 2 / (4 * half), 0.0)
    return np.clip(raw - bias_lo + bias_hi, lo, hi)


def _run_cell(
    source_obj: CIPairSource, n: int, n_reps: int, n_bootstrap: int, bayes_n: int,
    alpha: float, runs: int, statistic: str, seed, method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    """Run all reps for one (source, n) cell -- pairwise estimand.

    ``method_names``, if given, restricts computation (not just reporting) to
    methods whose ``.name`` is in the set -- e.g. ``{"mj_floor",
    "tango_scc", "bayes_paired_comp"}`` skips the bootstrap family, newcombe,
    and bayes_indep_comp entirely, which matters because bayes_indep_comp/
    bayes_paired_comp (importance sampling) are ~40-70x slower per call than
    mj_floor/tango_scc's closed-form/quartic evaluation. ``None`` (default)
    computes every applicable method, matching prior behavior.
    """
    rng = np.random.default_rng(seed)

    def _want(method_name: str) -> bool:
        return method_names is None or method_name in method_names

    active_bootstrap_methods = [m for m in METHODS if _want(m.name)]
    active_pairwise_extras = [m for m in PAIRWISE_EXTRA_METHODS if _want(m.name)]
    add_pairwise_extras = statistic == "mean" and source_obj.eval_type != "binary" and bool(active_pairwise_extras)
    active_dither_extras = [m for m in DITHER_EXTRA_METHODS if _want(m.name)]
    # Non-binary; the actual jitter width is auto-detected per rep from the
    # data itself (_detect_dither_halfwidth), not hardcoded. The motivating
    # mechanism (rounding-driven diff cancellation between two paired,
    # highly-correlated arms) needs a quantization grid to undo -- a fixed
    # +-0.5 (right for likert's integer rounding) was tried on continuous's
    # [0,1] scale too and was WRONG there (half the entire range), causing
    # heavy boundary clipping and a bias that got worse with N (coverage
    # 0.936 -> 0.800, n=10 -> n=100, nested screening). Detecting the grid
    # from the data instead of assuming one from eval_type fixes that AND
    # generalizes: labeled-"continuous" data that's actually coarse (a judge
    # emitting only a handful of distinct values) gets dithered correctly,
    # while genuinely continuous data detects no grid and the dither variant
    # safely reduces to its base method (identical CI, no bias introduced).
    add_dither_extras = (
        statistic == "mean" and source_obj.eval_type != "binary" and bool(active_dither_extras)
    )
    add_mj_floor = source_obj.eval_type == "binary" and statistic == "mean" and _want(MJ_FLOOR.name)
    add_tango_scc = source_obj.eval_type == "binary" and statistic == "mean" and _want(TANGO_SCC.name)
    add_tango_exact = source_obj.eval_type == "binary" and statistic == "mean" and _want(TANGO_EXACT.name)
    add_mj_unfloored = source_obj.eval_type == "binary" and statistic == "mean" and _want(MJ_UNFLOORED.name)
    add_bonett_price = source_obj.eval_type == "binary" and statistic == "mean" and _want(BONETT_PRICE.name)
    add_newcombe_mover = source_obj.eval_type == "binary" and statistic == "mean" and _want(NEWCOMBE_MOVER.name)
    add_bayes_indep = source_obj.eval_type == "binary" and statistic == "mean" and _want(BAYES_PAIR_INDEP.name)
    add_bayes_paired = source_obj.eval_type == "binary" and statistic == "mean" and _want(BAYES_PAIR_PAIRED.name)
    add_wald_indep = source_obj.eval_type == "binary" and statistic == "mean" and _want(WALD_PAIR_INDEP.name)

    active_methods = list(active_bootstrap_methods)
    if add_pairwise_extras:
        active_methods += active_pairwise_extras
    if add_dither_extras:
        active_methods += active_dither_extras
    if add_mj_floor:
        active_methods.append(MJ_FLOOR)
    if add_tango_scc:
        active_methods.append(TANGO_SCC)
    if add_tango_exact:
        active_methods.append(TANGO_EXACT)
    if add_mj_unfloored:
        active_methods.append(MJ_UNFLOORED)
    if add_bonett_price:
        active_methods.append(BONETT_PRICE)
    if add_newcombe_mover:
        active_methods.append(NEWCOMBE_MOVER)
    if add_bayes_indep:
        active_methods.append(BAYES_PAIR_INDEP)
    if add_bayes_paired:
        active_methods.append(BAYES_PAIR_PAIRED)
    if add_wald_indep:
        active_methods.append(WALD_PAIR_INDEP)

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_score: dict = {m: 0.0 for m in active_methods}
    total_pen_under: dict = {m: 0.0 for m in active_methods}
    total_pen_over: dict = {m: 0.0 for m in active_methods}
    rejects: dict = {m: 0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}
    true_diff = source_obj.true_diff

    def _record(method, ci_low: float, ci_high: float) -> None:
        if ci_low <= true_diff <= ci_high:
            covered[method] += 1
        total_w[method] += ci_high - ci_low
        total_score[method] += interval_score(ci_low, ci_high, true_diff, alpha)
        if true_diff < ci_low:
            total_pen_under[method] += (2.0 / alpha) * (ci_low - true_diff)
        elif true_diff > ci_high:
            total_pen_over[method] += (2.0 / alpha) * (true_diff - ci_high)
        if ci_low > 0.0 or ci_high < 0.0:
            rejects[method] += 1

    for _rep in range(n_reps):
        a, b = source_obj.generate_pair(rng, n, runs)

        for method in active_bootstrap_methods:
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = _pairwise_ci(
                        a, b, method=method.name, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng, statistic=statistic,
                    )
            except Exception:
                obs = _stat(a.mean(axis=1) - b.mean(axis=1), statistic=statistic)
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            _record(method, ci_low, ci_high)

        if add_pairwise_extras:
            pair_diffs = a.mean(axis=1) - b.mean(axis=1)
            obs = float(np.mean(pair_diffs))
            # A paired difference of two [scale_lo, scale_hi] values ranges
            # over [-span, span], naturally centred at 0 -- not [scale_lo,
            # scale_hi] itself. nig_ci_1d's default prior assumes its input
            # is centred at the scale's midpoint, and logit_t_ci_1d requires
            # values strictly in [0, 1], so both need diff_lo/diff_hi (not
            # scale_lo/scale_hi) to correctly re-centre the diff at 0 (which
            # rescaled_ci maps to 0.5 -- logit_t's own centre point).
            _scale_lo, _scale_hi = EVAL_TYPE_SCALE_BOUNDS[source_obj.eval_type]
            diff_span = _scale_hi - _scale_lo
            diff_lo, diff_hi = -diff_span, diff_span
            _nig_paired = functools.partial(nig_ci_1d, b0=_NIG_PAIRED_DIFF_B0)
            _extra_fns = dict(zip(PAIRWISE_EXTRA_METHODS, (t_interval_ci_1d, logit_t_ci_1d, _nig_paired, el_ci_1d)))
            for method in active_pairwise_extras:
                fn = _extra_fns[method]
                _t0 = time.perf_counter()
                try:
                    if method is NIG or method is LOGIT_T:
                        ci_low, ci_high = rescaled_ci(fn, pair_diffs, alpha, diff_lo, diff_hi)
                    else:
                        ci_low, ci_high = fn(pair_diffs, alpha)
                except Exception:
                    ci_low = ci_high = obs
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        if add_dither_extras:
            # Independent U(-half, +half) jitter per arm (not on the diff
            # directly), clip back to the scale, then subtract the exact
            # boundary-clipping bias (see _debiased_dither's docstring) --
            # half is detected per rep from the data's own quantization
            # grid (0.0 -- i.e. no jitter -- if none is detected). See
            # LOGIT_T_DITHER's docstring for why this specifically targets
            # the paired-diff rounding-cancellation pathology, not just
            # "add some noise."
            _scale_lo, _scale_hi = EVAL_TYPE_SCALE_BOUNDS[source_obj.eval_type]
            _half = _detect_dither_halfwidth(np.concatenate([a.ravel(), b.ravel()]))
            a_dither = _debiased_dither(a, _half, _scale_lo, _scale_hi, rng)
            b_dither = _debiased_dither(b, _half, _scale_lo, _scale_hi, rng)
            pair_diffs_dither = a_dither.mean(axis=1) - b_dither.mean(axis=1)
            obs_dither = float(np.mean(pair_diffs_dither))
            diff_span_dither = _scale_hi - _scale_lo
            diff_lo_dither, diff_hi_dither = -diff_span_dither, diff_span_dither
            for method in active_dither_extras:
                _t0 = time.perf_counter()
                try:
                    if method is LOGIT_T_DITHER:
                        ci_low, ci_high = rescaled_ci(
                            logit_t_ci_1d, pair_diffs_dither, alpha, diff_lo_dither, diff_hi_dither,
                        )
                    else:  # SMOOTH_BOOTSTRAP_DITHER
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            boot_stats = smooth_bootstrap_means_1d(pair_diffs_dither, n_bootstrap, rng, statistic=statistic)
                        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
                except Exception:
                    ci_low = ci_high = obs_dither
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        if add_mj_floor:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = mj_floor_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[MJ_FLOOR] += _el
            total_t_sq[MJ_FLOOR] += _el * _el
            _record(MJ_FLOOR, ci_low, ci_high)

        if add_tango_scc:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_scc_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[TANGO_SCC] += _el
            total_t_sq[TANGO_SCC] += _el * _el
            _record(TANGO_SCC, ci_low, ci_high)

        if add_tango_exact:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_scc_paired_ci(a[:, 0], b[:, 0], alpha, c=0.0)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[TANGO_EXACT] += _el
            total_t_sq[TANGO_EXACT] += _el * _el
            _record(TANGO_EXACT, ci_low, ci_high)

        if add_mj_unfloored:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = mj_unfloored_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[MJ_UNFLOORED] += _el
            total_t_sq[MJ_UNFLOORED] += _el * _el
            _record(MJ_UNFLOORED, ci_low, ci_high)

        if add_bonett_price:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = bonett_price_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[BONETT_PRICE] += _el
            total_t_sq[BONETT_PRICE] += _el * _el
            _record(BONETT_PRICE, ci_low, ci_high)

        if add_newcombe_mover:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = newcombe_mover_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[NEWCOMBE_MOVER] += _el
            total_t_sq[NEWCOMBE_MOVER] += _el * _el
            _record(NEWCOMBE_MOVER, ci_low, ci_high)

        if add_bayes_indep:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_INDEP] += _el
            total_t_sq[BAYES_PAIR_INDEP] += _el * _el
            _record(BAYES_PAIR_INDEP, ci_low, ci_high)

        if add_bayes_paired:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_paired_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_PAIRED] += _el
            total_t_sq[BAYES_PAIR_PAIRED] += _el * _el
            _record(BAYES_PAIR_PAIRED, ci_low, ci_high)

        if add_wald_indep:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _wald_indep_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[WALD_PAIR_INDEP] += _el
            total_t_sq[WALD_PAIR_INDEP] += _el * _el
            _record(WALD_PAIR_INDEP, ci_low, ci_high)

    return [
        SimResult(
            source=source_obj.source, label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_score=total_score[method],
            total_pen_under=total_pen_under[method],
            total_pen_over=total_pen_over[method],
            rejects=rejects[method],
            total_time=total_t[method], total_time_sq=total_t_sq[method],
            is_null=source_obj.is_null, model_a=source_obj.model_a, model_b=source_obj.model_b,
            benchmark_id=source_obj.benchmark_id, corpus_size=source_obj.max_n,
            true_diff=(source_obj.true_diff if source_obj.source != "synthetic" else None),
        )
        for method in active_methods
    ]


class _ProgressReporter:
    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        if self.mode == "off":
            return
        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.2:
            return
        self.last_print = now
        if self.mode == "cell":
            pct = 100.0 * min(step, self.total) / self.total
            print(f"\r  [{step:>7d}/{self.total:<7d}] {pct:6.2f}%  {detail:<55s}", end="", flush=True)
            if is_final:
                print()
            return
        frac = min(step, self.total) / self.total
        filled = int(28 * frac)
        bar = "█" * filled + "░" * (28 - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = step / elapsed
        eta_sec = max(self.total - step, 0) / max(rate, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        prefix = f"{self.label}: " if self.label else ""
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}  {detail[:40]:<40s}",
            end="", flush=True,
        )
        if is_final:
            print()


_CELL_SOURCES: list = []  # fork-inherited worker state for run_simulation
_NESTED_CELL_SOURCES: list = []  # fork-inherited worker state for run_nested_pairwise_simulation


def _run_cell_worker(args: tuple) -> list[SimResult]:
    sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed, method_names = args
    return _run_cell(_CELL_SOURCES[sc_idx], n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed, method_names)


def _run_nested_cell_worker(args: tuple) -> list[SimResult]:
    sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed, skip_bootstrap_binary, method_names = args
    return _run_nested_pairwise_cell(
        _NESTED_CELL_SOURCES[sc_idx], n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed,
        skip_bootstrap_binary, method_names,
    )


def run_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], n_reps: int, n_bootstrap: int,
    bayes_n: int, alpha: float, runs: int, statistic: str,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    """Run the flat (non-nested) pairwise CI simulation over every
    (source, sample size) cell, sequentially or across `n_workers` fork
    processes, and return the concatenated per-method SimResults.

    Cells where `n` meets or exceeds a source's corpus size (`max_n`) are
    skipped, with a warning printed for each.
    """
    global _CELL_SOURCES
    _CELL_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes if s.max_n is None or n < s.max_n]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed, method_names)
                 for (sc_idx, n), seed in zip(cells, child_seeds)]

    skipped = [(s, n) for s in sources for n in sample_sizes if not (s.max_n is None or n < s.max_n)]
    for s, n in skipped:
        print(f"  Warning: sample size n={n} >= corpus size {s.max_n} for {s.label}. Skipping.")

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="ci_paired")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Nested mode (--nested-mode): flat vs. nested pairwise CI methods on
# multi-run data, ported from sim_compare_boot_nested.py's
# _run_pairwise_multirun_cell. Mean-statistic only (matches the legacy
# script -- nested mode never supported --statistic median).
# ---------------------------------------------------------------------------


def _run_nested_pairwise_cell(
    source_obj: CIPairSource, n: int, runs: int, n_reps: int, n_bootstrap: int, bayes_n: int, alpha: float, seed,
    skip_bootstrap_binary: bool = False,
    method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    """Run all reps for one (source, n, runs) cell -- flat-vs-nested pairwise mean-diff estimand.

    ``method_names``, if given, restricts computation (not just reporting) to
    methods whose ``.name`` is in the set -- mirrors ``_run_cell``'s
    ``method_names`` (see its docstring). ``None`` (default) computes every
    applicable method, matching prior behavior.
    """
    rng = np.random.default_rng(seed)
    is_binary = source_obj.eval_type == "binary"
    true_diff = source_obj.true_diff

    def _want(method_name: str) -> bool:
        return method_names is None or method_name in method_names

    # skip_bootstrap_binary: mirrors ci_single.py's _run_nested_cell -- the
    # bootstrap family (flat cell-diff resampling and full-matrix nested
    # resampling alike) underperforms the dedicated binary pairwise methods
    # (mj_floor_*/newcombe_flat/bayes_pair_*) on binary data, so skip it there
    # to save compute and avoid diluting the bootstrap family's own
    # Score/Width average with its own binary underperformance in the
    # overall-summary and LaTeX output.
    run_bootstrap = not (skip_bootstrap_binary and is_binary)

    active_methods = [m for m in [T_INTERVAL] if _want(m.name)]
    if run_bootstrap:
        active_methods += [m for m in METHODS if _want(m.name)]
        active_methods += [m for m in PAIR_DIFF_NESTED_METHODS if _want(m.name)]
    if is_binary:
        active_methods += [m for m in BINARY_PAIR_FLAT_METHODS if _want(m.name)]
        # Default to the official subset; an explicit --methods can still name
        # anything in the full list (e.g. bonett_price_cluster as an ablation).
        _nested_pool = (BINARY_PAIR_NESTED_METHODS if method_names is not None
                        else BINARY_PAIR_NESTED_OFFICIAL)
        active_methods += [m for m in _nested_pool if _want(m.name)]
    else:
        active_methods += [m for m in (LOGIT_T, NIG, EL) if _want(m.name)]
        active_methods += [m for m in DITHER_EXTRA_METHODS if _want(m.name)]

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_score: dict = {m: 0.0 for m in active_methods}
    total_pen_under: dict = {m: 0.0 for m in active_methods}
    total_pen_over: dict = {m: 0.0 for m in active_methods}
    rejects: dict = {m: 0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}

    def _record(method, ci_low: float, ci_high: float) -> None:
        if ci_low <= true_diff <= ci_high:
            covered[method] += 1
        total_w[method] += ci_high - ci_low
        total_score[method] += interval_score(ci_low, ci_high, true_diff, alpha)
        if true_diff < ci_low:
            total_pen_under[method] += (2.0 / alpha) * (ci_low - true_diff)
        elif true_diff > ci_high:
            total_pen_over[method] += (2.0 / alpha) * (true_diff - ci_high)
        if ci_low > 0.0 or ci_high < 0.0:
            rejects[method] += 1

    for _rep in range(n_reps):
        a, b = source_obj.generate_pair(rng, n, runs)
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        obs_diff = float(np.mean(cell_diffs))

        # -- Cell-mean diff bootstrap family (flat) --
        if run_bootstrap:
            for method in METHODS:
                if not _want(method.name):
                    continue
                n_draws = bayes_n if method is BAYES_BOOTSTRAP else n_bootstrap
                _t0 = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        ci_low, ci_high = bootstrap_ci_1d(cell_diffs, obs_diff, method=method.name, n_bootstrap=n_draws, alpha=alpha, rng=rng)
                except Exception:
                    ci_low = ci_high = obs_diff
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        # -- t-interval on cell-mean diffs --
        if _want(T_INTERVAL.name):
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = t_interval_ci_1d(cell_diffs, alpha)
            except Exception:
                ci_low = ci_high = obs_diff
            _el = time.perf_counter() - _t0
            total_t[T_INTERVAL] += _el
            total_t_sq[T_INTERVAL] += _el * _el
            _record(T_INTERVAL, ci_low, ci_high)

        # -- logit_t/nig/el on cell-mean diffs -- mirrors _run_cell's
        # PAIRWISE_EXTRA_METHODS treatment above: logit_t/nig assume a
        # [0, 1] scale, so rescale onto diff_lo/diff_hi = [-span, span]
        # first (a zero diff maps to 0.5, logit_t/nig's own centre point).
        # el_ci_1d is nonparametric and needs no rescale. There is no
        # hierarchical (full N x R matrix) variant of these three for
        # paired diffs -- unlike the bootstrap family below, they only
        # ever see the cell-mean reduction, same as t_interval above.
        if not is_binary:
            _scale_lo, _scale_hi = EVAL_TYPE_SCALE_BOUNDS[source_obj.eval_type]
            diff_span = _scale_hi - _scale_lo
            diff_lo, diff_hi = -diff_span, diff_span
            _nig_paired = functools.partial(nig_ci_1d, b0=_NIG_PAIRED_DIFF_B0)
            for method, fn in zip((LOGIT_T, NIG, EL), (logit_t_ci_1d, _nig_paired, el_ci_1d)):
                if not _want(method.name):
                    continue
                _t0 = time.perf_counter()
                try:
                    if fn is el_ci_1d:
                        ci_low, ci_high = fn(cell_diffs, alpha)
                    else:
                        ci_low, ci_high = rescaled_ci(fn, cell_diffs, alpha, diff_lo, diff_hi)
                except Exception:
                    ci_low = ci_high = obs_diff
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        # -- logit_t_dither/smooth_bootstrap_dither on cell-mean diffs,
        # non-binary -- same fix as _run_cell's flat-mode add_dither_extras
        # block, see LOGIT_T_DITHER's and _detect_dither_halfwidth's
        # docstrings: the jitter width is auto-detected per rep from the
        # data's own quantization grid (0.0, i.e. no jitter, if none is
        # found), not a hardcoded +-0.5 -- a fixed width calibrated to
        # likert's integer rounding was tried on continuous's own scale too
        # and caused a bias that got WORSE with N (coverage 0.936 -> 0.800,
        # n=10 -> n=100). Like logit_t/nig/el above, these have no full-N-x-R
        # nested variant -- they operate on the same cell-mean-reduced
        # diffs, just computed from independently dithered a/b first.
        if not is_binary:
            _half = _detect_dither_halfwidth(np.concatenate([a.ravel(), b.ravel()]))
            a_dither = _debiased_dither(a, _half, _scale_lo, _scale_hi, rng)
            b_dither = _debiased_dither(b, _half, _scale_lo, _scale_hi, rng)
            cell_diffs_dither = a_dither.mean(axis=1) - b_dither.mean(axis=1)
            obs_diff_dither = float(np.mean(cell_diffs_dither))
            for method in (LOGIT_T_DITHER, SMOOTH_BOOTSTRAP_DITHER):
                if not _want(method.name):
                    continue
                _t0 = time.perf_counter()
                try:
                    if method is LOGIT_T_DITHER:
                        ci_low, ci_high = rescaled_ci(logit_t_ci_1d, cell_diffs_dither, alpha, diff_lo, diff_hi)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            boot_stats = smooth_bootstrap_means_1d(cell_diffs_dither, n_bootstrap, rng, statistic="mean")
                        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
                except Exception:
                    ci_low = ci_high = obs_diff_dither
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        # -- Nested pairwise diff methods (full N x R pair matrices) --
        if run_bootstrap:
            for method, fn in [
                (BOOTSTRAP_DIFF_NESTED, bootstrap_diffs_nested),
                (BAYES_DIFF_NESTED, bayes_bootstrap_diffs_nested),
                (SMOOTH_DIFF_NESTED, smooth_bootstrap_diffs_nested),
            ]:
                if not _want(method.name):
                    continue
                n_draws = bayes_n if method is BAYES_DIFF_NESTED else n_bootstrap
                _t0 = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*falling back to plain bootstrap; no KDE smoothing applied.*",
                            category=UserWarning,
                        )
                        boot_stats = fn(a, b, n_draws, rng)
                    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
                except Exception:
                    ci_low = ci_high = obs_diff
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        # -- Binary pairwise methods --
        if is_binary:
            a0, b0 = a[:, 0], b[:, 0]  # first run only (flat iid baseline)

            if _want(MJ_FLOOR_FLAT.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = mj_floor_paired_ci_flat(a, b, alpha)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[MJ_FLOOR_FLAT] += _el
                total_t_sq[MJ_FLOOR_FLAT] += _el * _el
                _record(MJ_FLOOR_FLAT, ci_low, ci_high)

            if _want(NEWCOMBE_FLAT.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = newcombe_mover_paired_ci(a0, b0, alpha)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[NEWCOMBE_FLAT] += _el
                total_t_sq[NEWCOMBE_FLAT] += _el * _el
                _record(NEWCOMBE_FLAT, ci_low, ci_high)

            if _want(BONETT_PRICE_FLAT.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = bonett_price_paired_ci_flat(a, b, alpha)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[BONETT_PRICE_FLAT] += _el
                total_t_sq[BONETT_PRICE_FLAT] += _el * _el
                _record(BONETT_PRICE_FLAT, ci_low, ci_high)

            if _want(BAYES_PAIR_INDEP.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _bayes_indep_comp_ci(a0, b0, alpha, n_bootstrap, rng)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[BAYES_PAIR_INDEP] += _el
                total_t_sq[BAYES_PAIR_INDEP] += _el * _el
                _record(BAYES_PAIR_INDEP, ci_low, ci_high)

            if _want(BAYES_PAIR_PAIRED.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _bayes_paired_comp_ci(a0, b0, alpha, n_bootstrap, rng)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[BAYES_PAIR_PAIRED] += _el
                total_t_sq[BAYES_PAIR_PAIRED] += _el * _el
                _record(BAYES_PAIR_PAIRED, ci_low, ci_high)

            if _want(WALD_PAIR_INDEP.name):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _wald_indep_ci(a0, b0, alpha)
                except Exception:
                    ci_low = ci_high = float(np.mean(a0 - b0))
                _el = time.perf_counter() - _t0
                total_t[WALD_PAIR_INDEP] += _el
                total_t_sq[WALD_PAIR_INDEP] += _el * _el
                _record(WALD_PAIR_INDEP, ci_low, ci_high)

            for method, fn in [
                (MJ_FLOOR_CLUSTER, mj_floor_paired_ci_multirun_cluster),
                (CLUSTERED_SCORE, clustered_score_paired_ci),
                (BONETT_PRICE_CLUSTER, bonett_price_paired_ci_multirun_cluster),
                (BONETT_PRICE_SHRUNK, bonett_price_paired_ci_multirun_shrunk),
            ]:
                # `covered` is keyed by active_methods, which defaults to
                # BINARY_PAIR_NESTED_OFFICIAL -- so this also skips methods that
                # are selectable but not in the default set.
                if not _want(method.name) or method not in covered:
                    continue
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(a, b, alpha)
                except Exception:
                    ci_low = ci_high = obs_diff
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

    return [
        SimResult(
            source="synthetic", label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_score=total_score[method],
            total_pen_under=total_pen_under[method],
            total_pen_over=total_pen_over[method],
            rejects=rejects[method],
            total_time=total_t[method], total_time_sq=total_t_sq[method],
            is_null=source_obj.is_null, run_noise_frac=source_obj.run_noise_frac, runs=runs,
        )
        for method in active_methods
    ]


def run_nested_pairwise_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    bayes_n: int, alpha: float, progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    skip_bootstrap_binary: bool = False,
    method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    """Run the --nested-mode flat-vs-nested pairwise CI simulation over
    every (source, sample size) cell at a fixed `runs` per input,
    sequentially or across `n_workers` fork processes, and return the
    concatenated per-method SimResults.
    """
    global _NESTED_CELL_SOURCES
    _NESTED_CELL_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [
        (sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed, skip_bootstrap_binary, method_names)
        for (sc_idx, n), seed in zip(cells, child_seeds)
    ]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label=f"ci_paired-nested[runs={runs}]")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_nested_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_nested_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    if cov < target - tol:
        return "v"
    if cov > target + tol:
        return "^"
    return " "


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _time_stats(subset: list[SimResult]) -> tuple[float, float]:
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = sum(r.total_time for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    return avg * 1000.0, float(np.sqrt(var / total_reps)) * 1000.0


def _headline_cov_width_score(
    per_n_vals: dict[tuple[str, int], list[tuple[float, float, float, float]]],
    m: str,
    sizes_present: list[int],
) -> tuple[float, float, float, float]:
    """Headline (Cov, Width, Score) for method `m`: average per n first (one
    number per n, unweighted across whatever sources contributed at that n),
    then average those per-n numbers across n -- rather than pooling every
    (source, n) cell into one flat list, which implicitly weights each n by
    how many sources have data there. This matters most for Score: a method
    that under-covers only at small n should have that penalty show up in
    the headline average, not get diluted by unrelated large-n cells."""
    per_n_means = []
    for n in sizes_present:
        vals = per_n_vals.get((m, n))
        if vals:
            per_n_means.append((
                float(np.mean([v[0] for v in vals])),
                float(np.mean([v[1] for v in vals])),
                float(np.mean([v[2] for v in vals])),
                float(np.mean([v[3] for v in vals])),
            ))
    if not per_n_means:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean([c for c, _, _, _ in per_n_means])),
        float(np.mean([w for _, w, _, _ in per_n_means])),
        float(np.mean([s for _, _, s, _ in per_n_means])),
        float(np.mean([q for _, _, _, q in per_n_means])),
    )


def _decision_rates(results: list[SimResult]) -> tuple[dict, dict]:
    """(type1, power) keyed by (eval_type, method).

    Type I is the reject rate on null rows (delta = 0); power is the reject
    rate on the alternative rows, averaged per scenario first so the two
    swept effect sizes (d=0.20, d=0.40) and every p/icc combination weigh
    equally rather than by how many cells each happens to contribute.
    """
    t1_acc: dict = defaultdict(lambda: [0, 0])
    pw_cells: dict = defaultdict(list)
    for r in results:
        key = (r.eval_type, r.method)
        if r.is_null:
            acc = t1_acc[key]
            acc[0] += r.rejects
            acc[1] += r.n_reps
        else:
            pw_cells[(r.eval_type, r.method, r.label)].append((r.rejects, r.n_reps))
    type1 = {k: (v[0] / v[1]) if v[1] else float("nan") for k, v in t1_acc.items()}
    by_method: dict = defaultdict(list)
    for (et, m, _label), cells in pw_cells.items():
        c = sum(x[0] for x in cells); n = sum(x[1] for x in cells)
        if n:
            by_method[(et, m)].append(c / n)
    power = {k: float(np.mean(v)) for k, v in by_method.items() if v}
    return type1, power


def _print_overall_summary_table(
    title: str,
    eval_types: list[str],
    results: list[SimResult],
    agg: dict[tuple, list[tuple[float, float, float]]],
    agg_counts: dict[tuple, tuple[int, int]],
    target: float,
    sizes_present: list[int],
    type1: dict | None = None,
    power: dict | None = None,
) -> None:
    """Print one OVERALL SUMMARY table, aggregated only over `eval_types`.

    No-ops (prints nothing) if none of `eval_types` are present in `results`,
    so callers can unconditionally request e.g. a binary-only table even when
    a given run has no binary data. `results` should already be non-null
    (i.e. `is_null=False`) since `agg`/`agg_counts` are built from those.
    """
    present_methods = {r.method for r in results if r.eval_type in eval_types}
    if not present_methods:
        return
    method_labels = [m.name for m in order_present_methods(present_methods)]

    per_n_vals: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    per_n_counts: dict[tuple[str, int], tuple[int, int]] = defaultdict(lambda: (0, 0))
    min_cov: dict[str, float] = defaultdict(lambda: float("inf"))
    for (et, m, n), vals in agg.items():
        if et not in eval_types:
            continue
        per_n_vals[(m, n)].extend(vals)
        c, t = agg_counts[(et, m, n)]
        c_prev, t_prev = all_counts[m]
        all_counts[m] = (c_prev + c, t_prev + t)
        c_prev_n, t_prev_n = per_n_counts[(m, n)]
        per_n_counts[(m, n)] = (c_prev_n + c, t_prev_n + t)
        min_cov[m] = min(min_cov[m], min(v[0] for v in vals))

    n_cols_hdr = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")
    print(f"  MinCov = worst per-scenario coverage seen for that method (not an average) --\n"
          f"  flags methods whose good mean coverage hides an unreliable scenario/n cell.")
    print(f"  TypeI = P(CI excludes 0) on null cells (target alpha); Power = the same rate\n"
          f"  on the alternative cells, averaged over scenarios. evalstats users act on this\n"
          f"  decision, directly and through the simultaneous-CI/FWER path.")
    print(f"  Score = Width + Penalty, reported separately because Score is ~90% Width,\n"
          f"  so a too-narrow method can post the best Score while under-covering.\n"
          f"  The two are one-sided in OPPOSITE directions: Width penalises intervals\n"
          f"  that are too WIDE, Penalty ((2/alpha) x mean miss distance) those that are\n"
          f"  too NARROW. Neither means 'calibration' on its own -- Penalty falls\n"
          f"  monotonically to 0 as an interval is widened, and a perfectly calibrated\n"
          f"  interval still carries a large Penalty (it misses alpha of the time by\n"
          f"  construction). Read Width, Penalty and Cov/MinCov together.")
    print(f"\n  {'Method':<20}  {'Cov':>6}  {'MinCov':>7}  {'Band95':>13}  {'Width':>8}  {'Penalty':>8}  {'Score':>8}  {'TypeI':>7}  {'Power':>7}  {'Time(ms)':>14}{n_cols_hdr}")
    _et_key = eval_types[0] if len(eval_types) == 1 else None
    for m in method_labels:
        mc, mw, ms, mp = _headline_cov_width_score(per_n_vals, m, sizes_present)
        c_tot, t_tot = all_counts[m]
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats(
            [r for r in results if r.method == m and r.eval_type in eval_types]
        )
        time_str = f"{avg_ms:.3f}+-{se_ms:.3f}" if np.isfinite(avg_ms) else "-"
        worst = min_cov[m]
        worst_str = f"{worst:.3f}{_cov_marker(worst, target)}" if np.isfinite(worst) else "-"
        n_cols_vals = ""
        for n in sizes_present:
            c_n, t_n = per_n_counts.get((m, n), (0, 0))
            cov_n = c_n / t_n if t_n > 0 else float("nan")
            n_cols_vals += f"  {cov_n:>5.3f}{_cov_marker(cov_n, target)} " if np.isfinite(cov_n) else f"  {'  -':>7}"
        t1s = f"{type1[(_et_key, m)]:.3f}" if type1 and (_et_key, m) in type1 else "-"
        pws = f"{power[(_et_key, m)]:.3f}" if power and (_et_key, m) in power else "-"
        print(f"  {m:<20}  {mc:>5.3f}{_cov_marker(mc, target)}  {worst_str:>7}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {mp:>8.4f}  {ms:>8.4f}  {t1s:>7}  {pws:>7}  {time_str:>14}{n_cols_vals}")


def print_report(results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, statistic: str) -> None:
    """Print the full text report: per-(eval_type, method, n) coverage
    grid, one OVERALL SUMMARY table per eval-type group (coverage/width/
    score/decision-rate), and, if any null-scenario rows are present, a
    Type-I error table.
    """
    target = 1.0 - alpha
    type1_map, power_map = _decision_rates(results)
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_labels = [m.name for m in order_present_methods(present_methods)]

    agg: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in non_null:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        score = r.total_score / r.n_reps
        penalty = (r.total_pen_under + r.total_pen_over) / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width, score, penalty))
        c_prev, t_prev = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

    def mean_cov(et, m, n):
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    sep = "=" * 72
    print(f"\n{sep}\n  CI_PAIRED COVERAGE -- SIMULATION RESULTS\n"
          f"  Estimand: paired template difference ({statistic})\n"
          f"  Nominal coverage: {target:.0%}   |   reps/cell: {n_reps}\n"
          f"  v = under-covered   ^ = over-conservative\n"
          f"  Score = interval score (width + (2/alpha)*miss-distance; lower is better --\n"
          f"  see evalstats.core.stats_utils.interval_score)\n{sep}")

    for et in eval_types_present:
        print(f"\n  [{et}]")
        hdr = f"    {'Method':<20}" + "".join(f"  n={n:<6}" for n in sample_sizes)
        print(hdr)
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sample_sizes:
                cov = mean_cov(et, m, n)
                row += "  " + (" " * 7 if np.isnan(cov) else f"{cov:.3f}{_cov_marker(cov, target)}".ljust(8))
            print(row)

    # Split into per-eval-type OVERALL SUMMARY tables -- binary, continuous
    # [0,1], likert, and grades all separate -- since these data types are
    # answered by very different method families/scales and a pooled table
    # obscures which methods actually perform best for which type. Likert
    # and grades used to be pooled together as one "numeric" table; kept
    # separate now since likert was found (2026-08-11) to have materially
    # different small-N paired-diff behavior than continuous/grades (see
    # LOGIT_T_DITHER's docstring) -- pooling would hide exactly that,
    # and concretely mixes likert's 1-5-scale widths with grades' 0-100-
    # scale widths, an even more obviously incomparable pair. Official runs
    # never include grades (see official_args()), so in practice this only
    # ever prints 3 tables there.
    sizes_present = sorted({r.n for r in non_null})
    _print_overall_summary_table(
        "OVERALL SUMMARY -- BINARY (averaged across sources)",
        ["binary"], non_null, agg, agg_counts, target, sizes_present, type1_map, power_map,
    )
    _print_overall_summary_table(
        "OVERALL SUMMARY -- CONTINUOUS [0,1] (averaged across sources)",
        ["continuous"], non_null, agg, agg_counts, target, sizes_present, type1_map, power_map,
    )
    _print_overall_summary_table(
        "OVERALL SUMMARY -- LIKERT (averaged across sources)",
        ["likert"], non_null, agg, agg_counts, target, sizes_present, type1_map, power_map,
    )
    _print_overall_summary_table(
        "OVERALL SUMMARY -- GRADES (averaged across sources)",
        ["grades"], non_null, agg, agg_counts, target, sizes_present, type1_map, power_map,
    )

    null_results = [r for r in results if r.is_null]
    if null_results:
        print(f"\n{'-'*72}\n  TYPE I ERROR RATE (null scenarios: delta = 0)\n"
              f"  Empirical P(CI excludes 0) -- target = alpha = {alpha:.2f}\n{'-'*72}")
        null_agg: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for r in null_results:
            c_prev, t_prev = null_agg[(r.eval_type, r.method, r.n)]
            null_agg[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)
        present_null_methods = {r.method for r in null_results}
        null_method_labels = [m for m in method_labels if m in present_null_methods]
        null_eval_types = [et for et in EVAL_TYPES if any(r.eval_type == et for r in null_results)]
        for et in null_eval_types:
            print(f"\n  {et}  (type I error = 1 - null coverage; target ~ {alpha:.2f})")
            hdr = f"    {'Method':<20}" + "".join(f"  n={n:<6}" for n in sample_sizes)
            print(hdr)
            for m in null_method_labels:
                row = f"    {m:<20}"
                for n in sample_sizes:
                    c_tot, t_tot = null_agg.get((et, m, n), (0, 0))
                    if t_tot <= 0:
                        row += "  " + " " * 7
                    else:
                        t1e = 1.0 - c_tot / t_tot
                        marker = "v" if t1e > alpha + 0.04 else ("^" if t1e < alpha - 0.04 else " ")
                        row += f"  {t1e:.3f}{marker}".ljust(9)
                print(row)
    print()


#: Fine-grained eval-type block label; see latex_tables for why the
#: coarse `eval_type_group` isn't used for these tables.
_report_eval_type_group = report_eval_type_group


def latex_overall_summary(results: list[SimResult], alpha: float, n_reps: int) -> str:
    """LaTeX booktabs version of print_report's OVERALL SUMMARY block
    (non-null rows only), plus one coverage column per sample size actually
    swept, appended to the right -- the aggregate "Coverage" column
    collapses across n and can hide miscalibration that only shows up at
    small or large sample sizes.

    Methods that ran on more than one eval type get one row PER eval type --
    "<method> (bin)"/"<method> (cont)"/"<method> (lik)" -- each computed
    from only that type's own data, rather than one row averaging across
    incomparable scales/regimes (see _report_eval_type_group's docstring for
    why this is now a 3-way split, not the 2-way binary/numeric split an
    earlier version of this function used).

    Within each eval-type block (separated by a midrule), the Score column's
    best value is bold and the runner-up is underlined -- see
    latex_tables.mark_best_and_runnerup. Coverage cells (aggregate and
    per-n) are shaded by latex_tables.coverage_cell to flag miscalibration
    at a glance."""
    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    method_labels = [m.name for m in order_present_methods({r.method for r in non_null})]
    sizes_present = sorted({r.n for r in non_null})

    # Decision rates, keyed by (report group, method) to match the row blocks.
    _grouped = [
        SimResult(**{**vars(r), "eval_type": _report_eval_type_group(r.eval_type)})
        for r in results
    ]
    g_type1, g_power = _decision_rates(_grouped)

    agg: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in non_null:
        g = _report_eval_type_group(r.eval_type)
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        score = r.total_score / r.n_reps
        agg[(g, r.method, r.n)].append((cov, width, score, (r.total_pen_under + r.total_pen_over) / r.n_reps))
        c_prev, t_prev = agg_counts[(g, r.method, r.n)]
        agg_counts[(g, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

    method_groups: dict[str, set[str]] = defaultdict(set)
    for (g, m, _n) in agg:
        method_groups[m].add(g)

    group_order = ["bin", "cont", "lik", "grades"]
    groups_present = sorted(
        {g for methods in method_groups.values() for g in methods},
        key=lambda g: group_order.index(g) if g in group_order else len(group_order),
    )

    rows = []
    rule_before = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        group_start = len(rows)
        score_vals: list[float] = []
        penalty_vals: list[float] = []
        for m in method_labels:
            if g not in method_groups[m]:
                continue
            multi_group = len(method_groups[m]) > 1
            per_n_vals: dict[tuple[str, int], list[tuple[float, float, float, float]]] = defaultdict(list)
            all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
            per_n_counts: dict[tuple[str, int], tuple[int, int]] = defaultdict(lambda: (0, 0))
            for n in sizes_present:
                vals = agg.get((g, m, n))
                if vals:
                    per_n_vals[(m, n)] = list(vals)
                c, t = agg_counts.get((g, m, n), (0, 0))
                c_prev, t_prev = all_counts[m]
                all_counts[m] = (c_prev + c, t_prev + t)
                per_n_counts[(m, n)] = (c, t)

            mc, mw, ms, mp = _headline_cov_width_score(per_n_vals, m, sizes_present)
            # Worst single (scenario, n) coverage -- the tail that the headline
            # Cov averages away. Same quantity as the printed table's MinCov.
            _worst = [v[0] for vals in per_n_vals.values() for v in vals]
            mmin = min(_worst) if _worst else float("nan")
            avg_ms, _ = _time_stats(
                [r for r in non_null if r.method == m and _report_eval_type_group(r.eval_type) == g]
            )
            time_str = f"{avg_ms:.3f}" if np.isfinite(avg_ms) else "-"
            label = f"{escape_latex(m)} ({g})" if multi_group else escape_latex(m)
            row = [
                label,
                coverage_cell(mc, target),
                coverage_cell(mmin, target) if np.isfinite(mmin) else "-",
                f"{mw:.4f}" if np.isfinite(mw) else "-",
                f"{mp:.4f}" if np.isfinite(mp) else "-",
                f"{ms:.4f}" if np.isfinite(ms) else "-",
                f"{g_type1[(g, m)]:.3f}" if (g, m) in g_type1 else "-",
                f"{g_power[(g, m)]:.3f}" if (g, m) in g_power else "-",
                time_str,
                g,
            ]
            for n in sizes_present:
                c_n, t_n = per_n_counts.get((m, n), (0, 0))
                cov_n = c_n / t_n if t_n > 0 else float("nan")
                row.append(coverage_cell(cov_n, target))
            rows.append(row)
            score_vals.append(ms)
            penalty_vals.append(mp)

        # Mark the best/runner-up in BOTH Penalty and Score. Marking Score
        # alone bolds whichever method is narrowest, which is how a
        # badly-calibrated method ends up looking like the winner.
        for col, vals in ((5, score_vals), (4, penalty_vals)):
            decorated = mark_best_and_runnerup([r[col] for r in rows[group_start:]], vals)
            for i, cell in enumerate(decorated):
                rows[group_start + i][col] = cell

    return booktabs_table(
        caption=(
            f"ci\\_paired: overall CI coverage summary (nominal {target*100:.0f}\\%, reps/cell={n_reps}). "
            "MinCov is the worst coverage over any single (scenario, $n$) cell -- the tail the ""headline Cov averages away. ""Score is the interval score, decomposed as Width + Pen(alty), where Pen is ""$\\frac{2}{\\alpha}\\times$the mean miss-distance \\citep{bracher2021evaluating}. ""Score is dominated by Width, so a method can be narrowest -- and so score best -- ""while covering worst. The two components are one-sided in opposite directions: ""Width penalises intervals that are too wide, Penalty those that are too narrow. ""Neither is a calibration measure on its own: Penalty decreases monotonically to ""zero as an interval is widened, and a perfectly calibrated interval still carries ""a substantial Penalty, since it misses $\\alpha$ of the time by construction. ""Type-I is the rate at which the interval excludes zero on the null scenarios ""(target $\\alpha$); Power is that rate on the alternative scenarios, averaged over ""scenarios. These are the decisions users act on, directly and through the ""simultaneous-CI/FWER path, which widens these same intervals. "
            "Methods tested on more than one eval type are reported as one row per type "
            "(bin/cont/lik), so no row averages across incomparable scales. Rows are grouped by "
            "eval type (all bin, then all cont, then all lik) so methods are comparable within a block."
        ),
        label="tab:ci_paired_overall",
        columns=["Method", "Cov", "MinCov", "Width", "Pen $\\downarrow$", "Score $\\downarrow$",
                 "Type-I", "Power $\\uparrow$", "Time (ms)", "Type"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
        rule_before=rule_before,
    )


def save_results_artifacts(
    *, results: list[SimResult], alpha: float, sample_sizes: list[int], n_reps: int,
    statistic: str, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    """Write one row per (source, eval_type, n, method) cell to
    `<run_stem>_results.csv`, and print_report()'s text (plus a LaTeX
    booktabs table appended if `latex=True`) to `<run_stem>_summary.log`,
    both under `out_dir`. Returns the two written paths.
    """
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "model_a", "model_b", "benchmark_id", "label", "eval_type", "n", "method", "n_reps",
            "covered", "total_width", "coverage", "mean_width", "total_score", "mean_score",
            "mean_penalty", "mean_pen_under", "mean_pen_over",
            "rejects", "reject_rate",
            "total_time", "total_time_sq", "mcse", "band95_low", "band95_high",
            "avg_time_ms", "se_time_ms", "is_null", "corpus_size", "true_diff", "run_noise_frac", "runs",
        ])
        for r in results:
            coverage = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            mean_score = r.total_score / r.n_reps
            mean_pen_under = r.total_pen_under / r.n_reps
            mean_pen_over = r.total_pen_over / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms = _time_stats([r])
            writer.writerow([
                r.source, r.model_a or "", r.model_b or "", r.benchmark_id or "", r.label, r.eval_type, r.n,
                r.method, r.n_reps, r.covered, f"{r.total_width:.8f}", f"{coverage:.8f}", f"{mean_width:.8f}",
                f"{r.total_score:.8f}", f"{mean_score:.8f}",
                f"{mean_pen_under + mean_pen_over:.8f}",
                f"{mean_pen_under:.8f}", f"{mean_pen_over:.8f}",
                r.rejects, f"{r.rejects / r.n_reps:.8f}",
                f"{r.total_time:.10f}", f"{r.total_time_sq:.10f}",
                f"{mcse:.8f}", f"{lo:.8f}", f"{hi:.8f}",
                f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                f"{se_ms:.6f}" if np.isfinite(se_ms) else "",
                r.is_null, r.corpus_size if r.corpus_size is not None else "",
                f"{r.true_diff:.8f}" if r.true_diff is not None else "",
                f"{r.run_noise_frac:.6f}", r.runs,
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, sample_sizes=sample_sizes, alpha=alpha, n_reps=n_reps, statistic=statistic)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_overall_summary(results, alpha=alpha, n_reps=n_reps)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_coverage_vs_n_plot(*, results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, out_path: str) -> str:
    """Coverage vs. sample size line plots -- one subplot per eval type, all methods overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "coverage": r.covered / r.n_reps}
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    label_level = df.groupby(["eval_type", "label", "method", "n"], as_index=False).agg(coverage=("coverage", "mean"))
    agg = label_level.groupby(["eval_type", "method", "n"], as_index=False).agg(
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"), coverage_count=("coverage", "count"),
    )
    palette = {m.name: m.color for m in method_objs}

    fig, axes = plt.subplots(1, max(len(eval_types_present), 1), figsize=(5.5 * max(len(eval_types_present), 1), 5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [name for name in method_names if name in et_agg["method"].values]
        if et_agg.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(data=et_agg, x="n", y="coverage_mean", hue="method", hue_order=et_methods,
                     palette=palette, marker=None, linewidth=1.0, alpha=0.70, ax=ax)
        for method, sub in et_agg.groupby("method"):
            if sub["coverage_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = get_method_color(str(method))
            se = sub["coverage_std"] / np.sqrt(sub["coverage_count"])
            ax.errorbar(sub["n"], sub["coverage_mean"], yerr=se, fmt="none", color=color, elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["coverage_mean"], s=28, color=color, edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(f"Coverage vs. Sample Size\nci_paired | reps={n_reps} | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_width_vs_n_plot(*, results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, out_path: str) -> str:
    """Mean CI width vs. sample size line plots -- one subplot per eval type, all methods overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "width": r.total_width / r.n_reps}
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    label_level = df.groupby(["eval_type", "label", "method", "n"], as_index=False).agg(width=("width", "mean"))
    agg = label_level.groupby(["eval_type", "method", "n"], as_index=False).agg(
        width_mean=("width", "mean"), width_std=("width", "std"), width_count=("width", "count"),
    )
    palette = {m.name: m.color for m in method_objs}

    fig, axes = plt.subplots(1, max(len(eval_types_present), 1), figsize=(5.5 * max(len(eval_types_present), 1), 5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [name for name in method_names if name in et_agg["method"].values]
        if et_agg.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(data=et_agg, x="n", y="width_mean", hue="method", hue_order=et_methods,
                     palette=palette, marker=None, linewidth=1.0, alpha=0.70, ax=ax)
        for method, sub in et_agg.groupby("method"):
            if sub["width_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = get_method_color(str(method))
            se = sub["width_std"] / np.sqrt(sub["width_count"])
            ax.errorbar(sub["n"], sub["width_mean"], yerr=se, fmt="none", color=color, elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["width_mean"], s=28, color=color, edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Mean CI width" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(f"CI Width vs. Sample Size\nci_paired | reps={n_reps} | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_reliability_violin_plot(*, results: list[SimResult], alpha: float, n_reps: int, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario coverage and interval
    score, one dot per (label, method) -- i.e. per data-generating scenario, averaged
    over sample sizes and reps but NOT over scenarios. Exposes the spread the OVERALL
    SUMMARY table's mean hides: a method with good average coverage can still have a
    long undercoverage tail on specific scenarios, which a single mean cannot reveal."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]
    palette = {m.name: m.color for m in method_objs}

    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "method": r.method,
            "coverage": r.covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    scenario_level = df.groupby(["eval_type", "label", "method"], as_index=False).agg(
        coverage=("coverage", "mean"), score=("score", "mean"),
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        et_df = scenario_level[scenario_level["eval_type"] == et]
        et_methods = [name for name in method_names if name in et_df["method"].values]
        for row_idx, (metric, ylabel) in enumerate([("coverage", "Coverage per scenario"), ("score", "Interval score per scenario")]):
            ax = axes[row_idx][col_idx]
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            sns.violinplot(
                data=et_df, x="method", y=metric, order=et_methods, hue="method",
                hue_order=et_methods, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="method", y=metric, order=et_methods, hue="method",
                hue_order=et_methods, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if metric == "coverage":
                ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\nci_paired | reps={n_reps} | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_by_n_violin_plot(
    *, results: list[SimResult], alpha: float, n_reps: int, out_dir: str, run_stem: str,
) -> list[str]:
    """Grouped violin plots of per-scenario coverage and interval score vs.
    sample size n -- one column per eval type present in ``results``, one
    violin per method at each n within a column (dodged side by side); each
    dot is one scenario's (label) mean coverage/score at that n and method.

    Originally built (and hardcoded) for comparing mj_floor vs.
    tango_scc vs. bayes_paired_comp across N on binary data only; generalized
    to any eval type/method combination so it also works for e.g. logit_t vs.
    another continuous/likert method (via --eval-types/--methods -- see
    ``add_arguments``'s ``--by-n-violin-plot`` help and
    ``discordant_comparison_args`` below for a worked binary example).

    One violin per method at each n (dodged side by side); each dot is one
    scenario's (label) mean coverage/score at that n and method -- this
    exists to make small-N coverage failures, and the resulting case for
    method choice as a function of N, visible directly in a figure rather
    than only in a results table.

    Not part of --official-tests: this is meant to be run deliberately on
    demand, since (a) it isn't a general-purpose calibration check like the
    other plots in this file, and (b) it's cheapest with --methods scoped
    down to just the 2-3 methods being compared (bayes_paired_comp's
    importance sampling is ~40-70x slower per call than mj_floor/tango_scc's
    closed-form/quartic paths -- see ``--methods``' help).

    Returns
    -------
    list[str]
        Paths to the two saved PNGs (coverage, then score).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]
    palette = {m.name: m.color for m in method_objs}

    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n,
            "coverage": r.covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    if df.empty:
        raise ValueError(
            "save_by_n_violin_plot: no non-null results found for the requested methods "
            f"{sorted(method_names)}. Check --eval-types and --methods produced overlapping "
            "data (--include-null rows are excluded from this plot)."
        )

    ns = sorted(df["n"].unique())
    n_order = [str(n) for n in ns]
    df["n_label"] = df["n"].astype(str)

    out_paths: list[str] = []
    for metric, ylabel, fname_suffix in [
        ("coverage", "Coverage per scenario", "by_n_violin_coverage"),
        ("score", "Interval score per scenario", "by_n_violin_score"),
    ]:
        n_cols = max(len(eval_types_present), 1)
        fig, axes = plt.subplots(1, n_cols, figsize=((1.1 * len(ns) + 2.5) * n_cols, 5.5), squeeze=False)
        for col_idx, et in enumerate(eval_types_present):
            ax = axes[0][col_idx]
            et_df = df[df["eval_type"] == et]
            et_methods = [m for m in method_names if m in et_df["method"].values]
            sns.violinplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="method", hue_order=et_methods,
                palette=palette, cut=0, inner="quartile", linewidth=0.8, dodge=True, alpha=0.35, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="method", hue_order=et_methods,
                palette=palette, size=4, alpha=0.6, dodge=True, jitter=0.15,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )

            if metric == "coverage":
                ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)

            handles, _ = ax.get_legend_handles_labels()
            method_handles = handles[:len(et_methods)]
            ax.legend(
                handles=method_handles, title="Method", fontsize=8, title_fontsize=9,
                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
            )

            ax.set_xlabel("Sample size (n)")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper())

        fig.suptitle(f"{ylabel} vs. Sample Size\n{run_stem} | reps={n_reps} | alpha={alpha}", fontsize=12)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
            fig.tight_layout()

        out_path = str(Path(out_dir) / f"{run_stem}_{fname_suffix}.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def save_cost_plot(*, results: list[SimResult], alpha: float, n_reps: int, out_path: str) -> str:
    """Scatter plot: x = mean CI time (log ms), y = coverage; one subplot per eval type."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]

    def _label_indices(ns: list) -> set:
        if len(ns) <= 2:
            return set(range(len(ns)))
        return {0, len(ns) // 2, len(ns) - 1}

    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11.0, 4.5 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        et_results = [r for r in non_null if r.eval_type == et]
        sample_sizes = sorted({r.n for r in et_results})

        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)

        legend_handles = []
        all_xs: list[float] = []
        for m in method_objs:
            color = m.color
            m_results = [r for r in et_results if r.method == m.name]
            if not m_results:
                continue
            points = []
            for n in sample_sizes:
                subset = [r for r in m_results if r.n == n]
                if not subset:
                    continue
                avg_ms, se_ms = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms <= 0:
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                points.append((n, avg_ms, cov, 1.96 * se_ms))
            if not points:
                continue

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            all_xs.extend(xs)
            ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.errorbar(xs, ys, xerr=[p[3] for p in points], fmt="o", color=color,
                        markersize=6, markeredgewidth=0.7, markeredgecolor="white",
                        elinewidth=0.9, capsize=2.5, capthick=0.9, alpha=0.90, zorder=3)

            for i, (n, x, y, _) in enumerate(points):
                if i in _label_indices(points):
                    ax.annotate(f"n={n}", xy=(x, y), xytext=(0, 4), textcoords="offset points",
                                fontsize=6.5, ha="center", va="bottom", color=color, alpha=0.85)

            legend_handles.append(plt.Line2D([0], [0], marker="o", color=color, markerfacecolor=color, markersize=7, label=m.name, linewidth=1.5))

        ax.set_xscale("log")
        ax.set_xlabel("Mean CI time (ms) -- log scale", fontsize=9.5)
        ax.set_ylabel("Coverage rate", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.set_ylim(max(0.0, target - 0.20), min(1.01, target + 0.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.tick_params(labelsize=8.5)
        if all_xs:
            import matplotlib.ticker as _ticker
            lo_ms, hi_ms = min(all_xs), max(all_xs)
            n_ticks = max(4, min(8, len(all_xs) // 2))
            tick_locs = np.logspace(np.log10(max(lo_ms, 1e-9)), np.log10(max(hi_ms, 1e-8)), n_ticks)
            ax.xaxis.set_major_locator(_ticker.FixedLocator(tick_locs))
            ax.xaxis.set_major_formatter(_ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
            ax.xaxis.set_minor_locator(_ticker.NullLocator())

        if not et_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        if legend_handles:
            ax.legend(handles=legend_handles, title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0.0, fontsize=7.5, title_fontsize=8, framealpha=0.85, ncol=1)

    fig.suptitle(
        f"Cost x Coverage Trade-off\n"
        f"ci_paired | x = mean CI compute time | y = empirical coverage | target = {target:.0%} | reps={n_reps}",
        fontsize=10.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_coverage_vs_run_noise_plot(*, results: list[SimResult], alpha: float, n_reps: int, out_path: str) -> str | None:
    """Coverage vs. run_noise_frac -- key --nested-mode plot for when nested/flat matters."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    run_noise_fracs = sorted({r.run_noise_frac for r in non_null})
    if len(run_noise_fracs) < 2:
        print(f"Skipped coverage-vs-run-noise plot (only one f_run value): {out_path}")
        return None

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.0, 4.0 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", zorder=1)

        for m in method_objs:
            covs = []
            for f in run_noise_fracs:
                subset = [r for r in non_null if r.eval_type == et and r.method == m.name and r.run_noise_frac == f]
                if not subset:
                    covs.append(float("nan"))
                    continue
                c_tot = sum(r.covered for r in subset)
                t_tot = sum(r.n_reps for r in subset)
                covs.append(c_tot / t_tot if t_tot > 0 else float("nan"))

            xs = [f for f, c in zip(run_noise_fracs, covs) if not np.isnan(c)]
            ys = [c for c in covs if not np.isnan(c)]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=m.color, linewidth=1.4, label=m.name, markersize=5, alpha=0.85)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(max(0.0, target - 0.25), min(1.01, target + 0.12))
        ax.set_xlabel("Run noise fraction  f_run = var_run / (var_input + var_run)", fontsize=9.5)
        ax.set_ylabel("Empirical coverage", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.legend(fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.85)

    runs_val = non_null[0].runs if non_null else "?"
    fig.suptitle(
        f"Coverage vs. Run Noise Fraction\n"
        f"ci_paired (nested mode) | runs={runs_val} | alpha={alpha} | reps={n_reps}\n"
        f"Averaged across all shapes and sample sizes",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ci_paired's CLI arguments onto `parser`: data-source/
    scenario selection, eval-type/method filters, sweep sizes and effect
    sizes, --nested-mode and its sub-options, and output/plotting controls.
    """
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                         help="'synthetic' (default), or a real-data source: " + ", ".join(REAL_PAIR_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="Synthetic scenario breadth (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES,
                         default=list(DEFAULT_EVAL_TYPES), metavar="TYPE",
                         help="Default matches the official presets (no 'grades'); pass "
                              "--eval-types grades explicitly to include it.")
    parser.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                         help="Restrict to these CI methods only, by Method.name (e.g. mj_floor "
                              "tango_scc bayes_paired_comp). Skips *computing* (not just reporting) any "
                              "method not listed -- the way to cut runtime when bayes_indep_comp/"
                              "bayes_paired_comp (importance sampling, ~40-70x slower per call than "
                              "mj_floor/tango_scc's closed-form) aren't needed. Ignored in --nested-mode, "
                              "which doesn't yet support per-method filtering. Default: compute every "
                              "method applicable to each source's eval type.")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID", help="Real-data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME", help="Real-data: model names to filter to")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--min-pair-size", type=int, default=50)
    parser.add_argument("--inspect-csv", default=None,
                         help=f"Path to CSV from collect_inspect_benchmarks.py "
                              f"(used by --data-source inspect/real; defaults to {DEFAULT_INSPECT_CSV!r})")
    parser.add_argument("--runs", type=int, default=1, metavar="R",
                         help="Flat mode: runs per input. R>=3 activates the nested-bootstrap-diffs "
                              "path in _pairwise_ci (real-data sources force runs=1, so this only "
                              "engages for synthetic data; default: 1). In --nested-mode this is the "
                              "fallback when --runs-sweep isn't given; nested mode itself supports "
                              "both synthetic and real (inspect) data.")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N")
    parser.add_argument("--bayes-n", type=int, default=2000, metavar="N")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50], metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC",
                         help="Flat mode: ICC sweep for build_pair_sources (default: 0.05 0.20 0.40 0.60 0.80). "
                              "Nested mode: optional additional ICC sweep, converted via f_run = 1 - ICC.")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.3], metavar="D")
    parser.add_argument("--include-null", action="store_true", default=False)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs overall-summary table to the saved summary .log file.")
    parser.add_argument("--by-n-violin-plot", action="store_true", default=False,
                         help="Also save grouped violin plots of per-scenario coverage and interval "
                              "score vs. sample size -- one violin per method at each n (one column "
                              "per eval type present). Originally built for comparing mj_floor vs. "
                              "tango_scc vs. bayes_paired_comp across N on binary data (see "
                              "discordant_comparison_args() for that invocation), but works for any "
                              "eval type/method set, e.g. logit_t vs. another continuous method via "
                              "--eval-types continuous --methods logit_t <other>; not part "
                              "of --official-tests, and cheapest combined with --methods to skip the "
                              "methods you aren't plotting. Ignored in --nested-mode.")
    parser.add_argument("--nested-mode", action="store_true", default=False,
                         help="Multi-run flat-vs-nested pairwise CI comparison (ported from "
                              "sim_compare_boot_nested.py). Supports --data-source synthetic or "
                              "inspect (real multi-run data); statistic=mean only.")
    parser.add_argument("--runs-sweep", type=int, nargs="+", default=None, metavar="R",
                         help="Nested mode: sweep multiple R values, overrides --runs")
    parser.add_argument("--run-noise-fracs", type=float, nargs="+", default=RUN_NOISE_FRACS_DEFAULT, metavar="F",
                         help="Nested mode: f_run = var_run / (var_input + var_run) values to sweep "
                              "(--icc-values is also accepted in nested mode, converted via f_run = 1 - ICC)")
    parser.add_argument("--heteroscedastic", action="store_true", default=False,
                         help="Nested mode: run noise scales with input value (mimics real LLM eval variability)")
    parser.add_argument("--no-bootstrap-binary", action="store_true", default=False,
                         help="Nested mode: skip the bootstrap-family methods (bootstrap/bca/bayes_bootstrap/"
                              "smooth_bootstrap/bootstrap_t, flat and nested) on binary data -- they underperform "
                              "the dedicated binary pairwise methods there, and including them dilutes the "
                              "bootstrap family's Score/Width average in the overall-summary and LaTeX output with "
                              "their own binary underperformance.")
    parser.add_argument("--pairwise-noise-grid", action="store_true", default=False,
                         help="Nested mode: use full Cartesian grid of run-noise fractions across models "
                              "(all f_A, f_B combinations) instead of matched f_A=f_B pairs")
    parser.add_argument("--pairwise-noise-grid-max", type=int, default=None, metavar="K",
                         help="Nested mode: optional cap on number of (f_A, f_B) combinations")
    parser.add_argument("--pairwise-noise-grid-seed", type=int, default=42, metavar="N")
    parser.add_argument("--cross-item-rho", type=float, default=0.7, metavar="RHO",
                         help="Nested mode: Gaussian-copula correlation between A's and B's item-level latent scores")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential).")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset, mirroring sim_compare_boot.py --official-test
    (pairwise phase). Synthetic data.

    Excludes "grades" from eval_types: "continuous" already covers the
    [0, 1]-scale case well (grades is just continuous rescaled to 0-100),
    while "likert" is kept as a genuinely distinct limiting case (integer-
    valued, few levels). Dropping grades cuts a third eval type out of the
    official sweep's runtime for no real loss of coverage.

    icc_values matched to nested_official_args()'s range (was stale here --
    this preset never received the 2026-07-14 reweighting nested_official_args()
    got, see that docstring for the full writeup: measuring actual per-item
    ICC on 48 real (model, benchmark) corpora gave mean 0.739, median 0.748,
    IQR [0.644, 0.873], i.e. concentrated well above this preset's old cap of
    0.80, not spread evenly across [0, 1]. Concretely surfaced by checking
    whether logit_t_dither's likert numbers here could be justified against
    plain logit_t: at this preset's old max icc=0.80, logit_t showed no
    coverage degradation at all (0.9463 at n=10) -- the pairwise battery
    literally couldn't reach the regime (icc -> 1, small N) where the
    rounding-cancellation pathology dithering fixes actually bites, so it
    was untestable here by construction, not merely untested."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=["binary", "continuous", "likert"],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        seed=base_seed, icc_values=[0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        no_bootstrap_binary=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def real_official_args(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for real data sources (requires network/HF access)."""
    return argparse.Namespace(
        data_source="real", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        seed=base_seed, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        no_bootstrap_binary=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def nested_real_official_args(base_seed: int = 45) -> argparse.Namespace:
    """Official-test preset for nested-mode real (inspect) data.
    Requires simulations/out/inspect_benchmarks.csv (produced by
    collect_inspect_benchmarks.py --runs 5 ...). Tests paired-diff CI
    coverage using the collected per-item runs -- most items have all
    five, but some have fewer (collection failures/timeouts); items with
    fewer than `runs=5` requested get the missing columns bootstrap-
    resampled from their own real runs (see
    build_inspect_corpora_multirun). `runs` below is unused in
    --nested-mode (runs_sweep takes precedence, see run()'s `runs_list =
    args.runs_sweep if args.runs_sweep else [args.runs]`); set to 5 to
    match runs_sweep for clarity rather than left dangling."""
    return argparse.Namespace(
        data_source="inspect", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=5, statistic="mean", reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 20, 30, 50, 75, 100],
        seed=base_seed, icc_values=None, cohens_d_values=[0.2, 0.4], include_null=False,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=True, runs_sweep=[5], run_noise_fracs=[0.0], heteroscedastic=False,
        no_bootstrap_binary=True,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """All official-test variants for this case, as (label, args) pairs."""
    return [
        ("synthetic", official_args(base_seed)),
        ("real data", real_official_args(base_seed)),
        ("nested / synthetic", nested_official_args()),
        ("nested / real data (inspect)", nested_real_official_args()),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast sanity-check preset for --quick-test: same shape catalog as
    official_args() but with reps/sizes/bootstrap_n cut down for a quick pass
    that confirms the pipeline (incl. --latex output) still works.
    ``data_source="real"`` (or 'openeval'/'inspect') swaps in
    build_real_pair_sources() instead of the synthetic shape catalog --
    --quick-test calls this twice per case (synthetic, then real) so the
    real-data path doesn't go unexercised between --official-tests runs."""
    return argparse.Namespace(
        data_source=data_source, scenario_suite="standard", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=3, bootstrap_n=200, bayes_n=200, alpha=0.05,
        sizes=[10, 30, 50],
        seed=base_seed, icc_values=[0.20], cohens_d_values=[0.3], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        no_bootstrap_binary=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=1,
    )


def nested_official_args(base_seed: int = 44) -> argparse.Namespace:
    """Canonical --nested-mode official preset, mirroring sim_compare_boot_nested.py's
    --official-test (pairwise-estimand phase). Not wired into --official-tests (the
    harness runs one preset per case); invoke manually:
    python -m simulations.harness.cli ci_paired --nested-mode --runs-sweep 5 --reps 500
      --bootstrap-n 10000 --bayes-n 10000 --scenario-suite expanded
      --icc-values 0.01 0.3 0.5 0.65 0.75 0.85 0.95
      --cohens-d-values 0.2 0.4 --include-null --heteroscedastic
      --no-bootstrap-binary --sizes 10 20 30 50 75 100 --seed 44

    Excludes "grades" from eval_types -- see official_args()'s docstring.

    no_bootstrap_binary=True: mirrors ci_single.py's nested_official_args --
    the bootstrap-family methods underperform the dedicated binary pairwise
    methods (mj_floor_*/newcombe_flat/bayes_pair_*) on binary data, so skip
    computing them there entirely rather than waste compute and dilute the
    bootstrap family's own Score/Width average with binary underperformance.

    icc_values reweighted 2026-07-14, mirroring the identical change and
    reasoning in ci_single.py's nested_official_args (see that docstring for
    the full writeup): the previous sweep (icc_values=[0.05, 0.30, 0.50] +
    run_noise_fracs=[0.01, 0.1, 0.3, 0.5], combining to ICC in {0.05, 0.3,
    0.5, 0.7, 0.9, 0.99}) spread ~evenly across the full [0, 1] ICC range.
    Measuring actual per-item ICC on all 48 real (model, benchmark) corpora in
    simulations/out/inspect_benchmarks.csv gave mean 0.739, median 0.748, IQR
    [0.644, 0.873] -- concentrated in the upper half, not uniform. This
    dataset is single-sample (no paired A/B structure), but the ICC being
    measured is a property of how consistent an item's score is across
    repeated runs of one model -- the same thing --icc-values controls here
    for each of the two compared arms (pairwise_noise_grid=False below means
    matched f_A=f_B, i.e. both arms get the same ICC in a given scenario) --
    so the same empirical distribution applies. icc_values now cluster around
    the real 25th/50th/75th percentile and near-max (0.65, 0.75, 0.85, 0.95),
    while keeping 0.3 and 0.5 as a hedge (other real domains -- e.g. long
    chain-of-thought/agentic tasks -- plausibly have genuinely higher
    run-to-run variance than what's sampled in this dataset) and 0.01 as an
    explicit extreme stress test (matching the one real corpus that did have
    near-zero ICC: a model performing at random chance on a 4-option
    benchmark, where correctness genuinely isn't tied to item identity).
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=["binary", "continuous", "likert"],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=5, statistic="mean", reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 20, 30, 50, 75, 100],
        seed=base_seed, icc_values=[0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=True, runs_sweep=[5], run_noise_fracs=[], heteroscedastic=True,
        no_bootstrap_binary=True,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def discordant_comparison_args(base_seed: int = 46) -> argparse.Namespace:
    """mj_floor vs. tango_scc vs. bayes_paired_comp across N=10..125, for
    the coverage/interval-score violin plots (--by-n-violin-plot). Not wired
    into --official-tests (this exists to make a specific method-choice
    argument visible in a figure, not as a general calibration check);
    invoke manually:

    python -m simulations.harness.cli ci_paired --data-source synthetic
      --scenario-suite expanded --eval-types binary
      --methods mj_floor tango_scc bayes_paired_comp
      --reps 300 --bootstrap-n 10000 --bayes-n 10000 --alpha 0.05
      --sizes 10 15 20 30 40 50 60 70 80 90 100 110 125
      --icc-values 0.05 0.20 0.40 0.60 0.80 --cohens-d-values 0.2 0.4
      --include-null --seed 46 --by-n-violin-plot
      --progress bar --save-results save --out-dir simulations/out --latex

    --methods scopes computation to just the three methods being compared
    (skipping the bootstrap family, newcombe, and bayes_indep_comp entirely
    -- see --methods' help), which matters because bayes_paired_comp's
    importance sampling is ~40-70x slower per call than mj_floor/tango_scc's
    closed-form/quartic paths.
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=["binary"],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125],
        methods=["mj_floor", "tango_scc", "bayes_paired_comp"],
        seed=base_seed, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="off", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        no_bootstrap_binary=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1), by_n_violin_plot=True,
    )


def run(args: argparse.Namespace) -> CaseResult:
    """Harness entry point: build CIPairSources for `args.data_source`
    (synthetic, or real via build_real_pair_sources/build_real_pair_sources_nested),
    run the flat or --nested-mode simulation, print the report, and save
    CSV/log/plot artifacts as requested. Returns a CaseResult with status,
    output paths, and headline coverage metrics.
    """
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        nested_mode = getattr(args, "nested_mode", False)

        if nested_mode:
            if args.data_source not in ("synthetic", "inspect"):
                raise ValueError("--nested-mode supports --data-source synthetic or inspect.")

            bayes_n = args.bayes_n if args.bayes_n is not None else args.bootstrap_n
            runs_list = args.runs_sweep if args.runs_sweep else [args.runs]

            if args.data_source == "inspect":
                csv_path = getattr(args, "inspect_csv", None) or DEFAULT_INSPECT_CSV
                sources = build_real_pair_sources_nested(
                    csv_path, models=args.models, benchmarks=args.benchmarks,
                    min_pair_size=args.min_pair_size,
                )
                print(f"\nci_paired simulation (nested mode, inspect) -- runs={runs_list}")
            else:
                run_noise_fracs = list(args.run_noise_fracs)
                if args.icc_values:
                    icc_as_run_noise = [float(np.clip(1.0 - icc, 0.0, 1.0)) for icc in args.icc_values]
                    run_noise_fracs = sorted(set(run_noise_fracs + icc_as_run_noise))
                print(f"\nci_paired simulation (nested mode) -- runs={runs_list}, run_noise_fracs={run_noise_fracs}")
                sources = build_pair_sources(
                    suite=args.scenario_suite, cohens_d_values=args.cohens_d_values,
                    include_null=args.include_null, run_noise_fracs=run_noise_fracs, heteroscedastic=args.heteroscedastic,
                    pairwise_noise_grid=args.pairwise_noise_grid, pairwise_noise_grid_max=args.pairwise_noise_grid_max,
                    pairwise_noise_grid_seed=args.pairwise_noise_grid_seed, cross_item_rho=args.cross_item_rho,
                )

            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CIPairSources left after filtering.")
            method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}"
                  + (f", methods={sorted(method_names)}" if method_names else ""))

            results: list[SimResult] = []
            n_workers = getattr(args, "workers", 1)
            for r_val in runs_list:
                results.extend(run_nested_pairwise_simulation(
                    sources, sample_sizes=args.sizes, runs=r_val, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                    bayes_n=bayes_n, alpha=args.alpha, progress_mode=args.progress, seed=args.seed,
                    n_workers=n_workers,
                    skip_bootstrap_binary=getattr(args, "no_bootstrap_binary", False),
                    method_names=method_names,
                ))
            print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps, statistic="mean")

            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_stem = f"ci_paired_nested_{args.data_source}_runs{'-'.join(str(r) for r in runs_list)}_reps{args.reps}_{stamp}"
            output_paths: list[str] = []

            if args.save_results == "save":
                output_paths += save_results_artifacts(
                    results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                    statistic="mean", out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False),
                )

            if args.plots == "save":
                cov_path = save_coverage_vs_n_plot(
                    results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_n.png"),
                )
                width_path = save_width_vs_n_plot(
                    results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_width_vs_n.png"),
                )
                cost_path = save_cost_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_cost_coverage.png"),
                )
                reliability_path = save_reliability_violin_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths += [cov_path, width_path, cost_path, reliability_path]
                run_noise_path = save_coverage_vs_run_noise_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_run_noise.png"),
                )
                if run_noise_path:
                    output_paths.append(run_noise_path)
                print(f"Saved plots: {output_paths[-4:] if run_noise_path else output_paths[-3:]}")

            if getattr(args, "by_n_violin_plot", False):
                violin_paths = save_by_n_violin_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_dir=plots_dir, run_stem=run_stem,
                )
                output_paths += violin_paths
                print(f"Saved violin plots: {', '.join(violin_paths)}")

            non_null = [r for r in results if not r.is_null]
            overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
            return CaseResult(
                case_name=CASE_NAME, status="ok", output_paths=output_paths,
                key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
                duration_s=time.time() - t0,
            )

        print(f"\nci_paired simulation -- data_source={args.data_source}, statistic={args.statistic}")
        if args.data_source == "synthetic":
            icc_values = args.icc_values if args.icc_values is not None else [0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
            sources = build_pair_sources(
                suite=args.scenario_suite, icc_values=icc_values,
                cohens_d_values=args.cohens_d_values, include_null=args.include_null,
            )
        else:
            runs = args.runs
            if runs != 1:
                print("  Warning: real-data sources only support --runs 1 in this pass; forcing runs=1.")
                runs = 1
            args = argparse.Namespace(**{**vars(args), "runs": runs})
            sources = build_real_pair_sources(
                args.data_source, benchmarks=args.benchmarks, models=args.models,
                hf_token=args.hf_token, cache_dir=args.cache_dir, min_pair_size=args.min_pair_size,
                inspect_csv=args.inspect_csv, include_null=args.include_null,
            )

        if args.eval_types:
            requested = set(args.eval_types)
            sources = [s for s in sources if s.eval_type in requested]
        if not sources:
            raise ValueError("No CIPairSources left after filtering.")

        method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
        print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}, runs={args.runs}"
              + (f", methods={sorted(method_names)}" if method_names else ""))

        results = run_simulation(
            sources, sample_sizes=args.sizes, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
            bayes_n=args.bayes_n, alpha=args.alpha, runs=args.runs, statistic=args.statistic,
            progress_mode=args.progress, seed=args.seed, n_workers=getattr(args, "workers", 1),
            method_names=method_names,
        )
        print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps, statistic=args.statistic)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"ci_paired_{args.data_source}_stat{args.statistic}_reps{args.reps}_{stamp}"
        output_paths: list[str] = []

        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                statistic=args.statistic, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False),
            )

        if args.plots == "save":
            cov_path = save_coverage_vs_n_plot(
                results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_n.png"),
            )
            width_path = save_width_vs_n_plot(
                results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_width_vs_n.png"),
            )
            cost_path = save_cost_plot(
                results=results, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_cost_coverage.png"),
            )
            reliability_path = save_reliability_violin_plot(
                results=results, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
            )
            output_paths += [cov_path, width_path, cost_path, reliability_path]
            print(f"Saved plots: {cov_path}, {width_path}, {cost_path}, {reliability_path}")

        if getattr(args, "by_n_violin_plot", False):
            violin_paths = save_by_n_violin_plot(
                results=results, alpha=args.alpha, n_reps=args.reps,
                out_dir=plots_dir, run_stem=run_stem,
            )
            output_paths += violin_paths
            print(f"Saved violin plots: {', '.join(violin_paths)}")

        non_null = [r for r in results if not r.is_null]
        overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
            duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
