"""ci_unpaired case: between-subjects (unpaired) pairwise CI coverage.

The missing validation behind ``compare(design="unpaired")``. ``ci_paired.py``
answers "which CI method for a paired difference?"; nothing yet answers it for
two *independent* groups, which is why the paper currently has to disclaim
pairwise CIs for between-subjects designs.

What the shipped code does today (evalstats/config.py's
AUTO_UNPAIRED_METHOD_TABLE, evalstats/core/unpaired.py):

  binary                  -> Welch's t-interval on the raw 0/1 values, i.e.
                             a linear-probability-model patch, explicitly
                             flagged in config.py as "a deliberate patch, not
                             a clean solution".
  continuous/likert/grade -> a percentile bootstrap on theta_ab, the
                             stochastic-dominance probability from the
                             Mann-Whitney / Kruskal-Wallis path.

This case measures ONE estimand: the mean difference mean(A) - mean(B) (which
on binary data is the proportion difference). That is the estimand every other
recommendation in this project is stated in, including the paired path's, and
the one a reader of those recommendations expects.

theta is deliberately out of scope. It was implemented here and removed: it is
a different quantity, not a different method for the same quantity, so its
coverage and width are not comparable with anything else in the table, and
carrying it cost ~65% of the sweep's runtime. If the shipped theta path needs
calibrating, that is its own case with its own estimand, not a second axis
bolted onto this one.

Data generation
---------------
Reuses ``scenarios.CIPairSource`` unchanged, but *breaks the pairing*: arm A
comes from one ``generate_pair`` call and arm B from a second, independent
one, so the two groups share no items. Two consequences worth stating
explicitly:

  1. ``source.true_diff`` remains exactly correct. It is
     E[mean(a)] - E[mean(b)], a difference of marginals, and expectation is
     linear -- whether a and b were drawn jointly or independently does not
     change it. This holds for the hand-built 2x2 binary scenarios too
     (true_diff = p10 - p01 = p_A - p_B).
  2. Real corpora (``scenarios/real_unpaired.py``) draw each arm from a fixed
     pool with replacement, so the pool mean is exactly the coverage target.

Unequal group sizes are the norm in between-subjects work, so ``--size-ratios``
sweeps n_B / n_A (default 1.0; e.g. ``--size-ratios 1.0 2.0``).

Method slate
------------
Deliberately a *starting* slate, not a settled recommendation:

  all eval types : bootstrap, bca, bayes_bootstrap, smooth_bootstrap,
                   bootstrap_t (two-sample forms), welch_t, student_t,
                   mover_t, mover_logit_t
  non-binary     : mover_nig (NIG is invalid on a proportion -- it returns
                   limits above 1)
  binary only    : wald_unpaired (naive baseline), agresti_caffo,
                   newcombe_hybrid, miettinen_nurminen, agresti_min

The mover_* family builds a CI for each arm with a shipped one-sample method
and combines them by MOVER (Zou & Donner 2008). Newcombe's hybrid score IS
that construction with Wilson arms, which is what lets the paired path's own
recommendations (logit_t, nig) transfer here directly. mover_t is the control
that separates what the combination rule buys from what the arm buys.

Known limitations of this first pass (deliberate, to keep the engine small):
runs=1 only (no multi-run / nested variants), statistic=mean only.

Run:
    python -m simulations.harness.cli ci_unpaired --quick
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as _mp
import os
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.stats as stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_means_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        t_interval_ci_1d,
    )
    from evalstats.core.stats_utils import interval_score, rescaled_ci

from ..scenarios import CIPairSource, EVAL_TYPES, DEFAULT_EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import SCENARIO_SUITES, build_pair_sources
from ..scenarios.real_unpaired import (
    REAL_UNPAIRED_DATASETS, DEFAULT_DATA_DIR as REAL_DEFAULT_DATA_DIR,
    build_real_unpaired_sources,
)
from ..latex_tables import (
    booktabs_table, coverage_cell, escape_latex,
    mark_best_and_runnerup, report_eval_type_group,
)
from ..methods import (
    BOOTSTRAP_METHODS,
    BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T,
    WELCH_T, STUDENT_T, MOVER_T, MOVER_LOGIT_T, MOVER_NIG,
    WALD_UNPAIRED, AGRESTI_CAFFO, NEWCOMBE_HYBRID, MIETTINEN_NURMINEN, BAYES_BETA_INDEP,
    AGRESTI_MIN,
    UNPAIRED_MEAN_EXTRA_METHODS, UNPAIRED_BINARY_METHODS,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "ci_unpaired"

DATA_SOURCES = ["synthetic", "real"]
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]

_AGRESTI_MIN_MAX_N = 30
"""Largest group size at which agresti_min is run in the Monte Carlo sweep.

Not an arbitrary budget cap. Fagerland, Lydersen & Laake recommend the
Agresti-Min exact unconditional interval specifically "for small sample sizes
(less than 30 in each sample)"; above that they prefer the Newcombe hybrid
score, and note the asymptotic intervals all behave well there. So running the
exact interval at n=100 measures it outside the range anyone recommends it
for, and it is not cheap: 192 ms per interval at n=100 against 0.1 ms for
Welch, which made it 85% of the entire binary sweep. Exact-coverage mode
(--exact-coverage) has no such limit and is the right tool if you want its
behaviour at larger n -- it is both faster and free of Monte Carlo error.
Raise with --agresti-min-max-n."""

_BINARY_INELIGIBLE = (MOVER_NIG,)
"""MOVER arms that are not valid on binary data, so they are not run there --
matching ci_paired, which gates its own nig off binary for the same reason.

Not a convention but a validity failure, verified directly on 0/1 samples at
n=10: NIG returns [0.852, 1.0571] for a 10/10 sample -- an upper limit above
1 for a proportion.

mover_logit_t is deliberately NOT gated: the logit transform respects [0, 1]
by construction, and on binary it is the strongest method in the exact table
(0.954 minimum coverage, holding nominal), so excluding it would discard a
real result rather than avoid a spurious one."""

@dataclass
class SimResult:
    """One (source, n_a, n_b, method) cell's aggregated outcome over n_reps
    Monte Carlo draws: coverage, width, interval-score components, rejection
    rate and timing, summed rather than averaged so cells can be pooled
    before dividing."""
    source: str
    label: str
    eval_type: str
    n_a: int
    n_b: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_score: float = 0.0
    """Sum of interval_score() (see evalstats.core.stats_utils) across n_reps."""
    total_pen_under: float = 0.0
    """Sum of the (2/alpha)*(lo - target) penalty for target BELOW the interval.

    Bracher, Ray, Gneiting & Reich (2021) decompose the interval score into
    width (sharpness) and the penalty for the true value falling outside the
    interval (calibration), splitting the latter into over- and
    underprediction to expose systematic bias. Kept separate from
    total_score because the mean score is ~90% width, so a method can
    under-cover badly and still post the best score -- the penalty term is
    what tracks calibration."""
    total_pen_over: float = 0.0
    """Sum of the (2/alpha)*(target - hi) penalty for target ABOVE the interval."""
    rejects: int = 0
    """Reps whose CI excluded zero: Type I error on is_null rows, power elsewhere."""
    total_time: float = 0.0
    """Sum of wall-clock seconds spent computing this method's interval, across n_reps."""
    total_time_sq: float = 0.0
    is_null: bool = False
    """Whether this source's true_diff is exactly zero (a Type I / calibration row)."""
    true_value: float = 0.0
    """The coverage target for this source: source.true_diff, mean(A) - mean(B)."""
    base_n: int = 0
    """The sweep's size parameter for this cell, before the imbalance ratio.

    Not the same as n_a once the ratio may be applied to EITHER arm: a cell
    can be (20, 80) or (80, 20) and both come from base_n=20. Per-n report
    columns and the vs-n plots group on this, so a column headed n=20 means
    "base size 20" rather than mixing base-20-scaled-up with base-80."""
    scale_span: float = 1.0
    """Width of this source's measurement scale (hi - lo).

    Width, Penalty and Score are all homogeneous of degree 1 in the scale's
    units, so dividing by this is exactly equivalent to computing them on the
    [0, 1]-rescaled data -- which is what makes them poolable across sources.
    Synthetic sources of one eval type all share a scale, so this was moot
    until the real corpora arrived: SocSci210's likert outcomes alone span
    1-3, 1-4, 1-5, 1-6, 1-7, 1-9 and 0-5, and averaging raw widths over those
    let one wide-scale outcome dominate the mean. That artefact reversed the
    apparent likert ranking, making the best method look near-worst."""
    icc: float = 0.0
    cohens_d: float = 0.0


# ---------------------------------------------------------------------------
# Delta-mean / Delta-p intervals for two INDEPENDENT samples
# ---------------------------------------------------------------------------
# Local to this case on purpose: none of these are shipped by evalstats today,
# and which of them (if any) should be is exactly what this sweep is for.
# Promote the winners into evalstats.core.resampling once the slate settles.


def _welch_t_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Welch's unequal-variance t-interval on mean(a) - mean(b).

    This is the shipped behavior for binary score types
    (evalstats.core.unpaired._binary_pairwise_uncorrected), reached via
    scipy's ttest_ind(equal_var=False).confidence_interval.
    """
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        d = float(np.mean(a) - np.mean(b))
        return d, d
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = va / na + vb / nb
    d = float(np.mean(a) - np.mean(b))
    if se2 <= 0.0:
        # Both arms constant -- no variance to estimate from, so a
        # variance-based interval has nothing to say and collapses to a
        # point. NOT a harness bug: on binary data at small n and extreme p
        # this is common and real (measured: 17% of draws at n=5, 4.7% at
        # n=10, 1.25% at n=20, none by n=30), and it is a genuine part of
        # why the t-based methods bottom out near 0.64 exact coverage while
        # the dedicated binary intervals -- which handle a degenerate sample
        # rather than dividing by its variance -- never do this at all.
        return d, d
    se = float(np.sqrt(se2))
    # Welch-Satterthwaite degrees of freedom.
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return d - t * se, d + t * se


def _student_t_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Pooled-variance (equal-variance) two-sample t-interval."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        d = float(np.mean(a) - np.mean(b))
        return d, d
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    df = na + nb - 2
    sp2 = ((na - 1) * va + (nb - 1) * vb) / df
    se = float(np.sqrt(sp2 * (1.0 / na + 1.0 / nb)))
    d = float(np.mean(a) - np.mean(b))
    if se <= 0.0:
        return d, d
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return d - t * se, d + t * se


def _wald_unpaired_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Naive normal-approximation Wald interval for p_A - p_B.

    The textbook bad baseline for two independent proportions: it degenerates
    to zero width when either arm is at 0 or 1, and under-covers badly at
    small n or extreme p. Kept for the same reason ci_single keeps `wald` --
    a floor to measure the real methods against.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa, pb = float(np.mean(a_bin)), float(np.mean(b_bin))
    se = float(np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d = pa - pb
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _agresti_caffo_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Agresti & Caffo (2000), "Simple and effective confidence intervals for
    proportions and differences of proportions result from adding two
    successes and two failures", The American Statistician 54(4):280-288.

    Add one success and one failure to EACH arm, then apply the plain Wald
    formula to the adjusted counts. Motivated as the two-sample analogue of
    the Agresti-Coull single-proportion adjustment; the appeal is that it is
    a one-line change to Wald that fixes most of Wald's small-sample
    undercoverage.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa = (float(np.sum(a_bin)) + 1.0) / (na + 2.0)
    pb = (float(np.sum(b_bin)) + 1.0) / (nb + 2.0)
    se = float(np.sqrt(pa * (1 - pa) / (na + 2.0) + pb * (1 - pb) / (nb + 2.0)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d = pa - pb
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _wilson_bounds(successes: float, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a single proportion (helper for MOVER)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, float(centre - radius)), min(1.0, float(centre + radius))


def _mover_combine(
    ta: float, tb: float, arm_a: tuple[float, float], arm_b: tuple[float, float],
) -> tuple[float, float]:
    """MOVER ("method of variance estimates recovery") combination of two
    independent one-sample intervals into an interval for their difference.

    Zou & Donner (2008), "Construction of confidence limits about effect
    measures: a general approach", Statistics in Medicine 27(10):1693-1702.
    Given point estimates ta, tb and separate intervals (la, ua), (lb, ub):

        lower = (ta - tb) - sqrt( (ta - la)^2 + (ub - tb)^2 )
        upper = (ta - tb) + sqrt( (ua - ta)^2 + (tb - lb)^2 )

    Each arm contributes its variance estimate *at the tail that matters*
    for that endpoint, rather than a single symmetric SE at the point
    estimate -- which is what lets a skewed or boundary-respecting one-sample
    interval carry its good behavior through to the difference.

    This is the general form of Newcombe's hybrid score interval (which is
    exactly this with Wilson arms), so it lets the unpaired path reuse the
    very same one-sample methods the paired path already recommends.

    KNOWN STRUCTURAL LIMIT -- read before promoting any mover_* method.
    MOVER assembles the difference interval out of the two MARGINAL
    intervals, so it inherits whatever miscalibration those marginals carry
    and can never benefit from error that cancels in the subtraction. On a
    saturating shape (e.g. cont-one-inflated-extreme at high icc), both arms'
    means are severely skewed and every one-sample interval covers badly
    there, but the ARMS' DIFFERENCE is far milder because the two skews
    partly cancel -- so welch_t, which targets the difference directly,
    covers correctly while mover_logit_t inherits both bad marginals and
    under-covers.

    So MOVER is the right construction exactly when the marginals are
    well-calibrated -- which is why newcombe_hybrid works so well on binary
    (Wilson marginals are excellent) and why mover_logit_t is fine on likert
    and ordinary continuous data but not on ceiling-saturated continuous.
    This is a property of the construction, not a bug in it, and it is not
    fixable by a better arm method: even an ORACLE marginal (the exact 95%
    interval built from each arm's true sampling distribution) still
    under-covers here while being wider than welch_t, because assembling
    from marginals prices in each arm's full skew, while the difference's
    own skew is milder from cancellation that MOVER cannot reach.

    Candidate fixes tried and rejected: combining degenerate arms by interval
    arithmetic instead of in quadrature; logit_t order=2 (its documented
    small-n boundary gain does not transfer to this failure mode); switching
    the arm to NIG only when the sample looks saturated. None moved coverage.
    NIG arms throughout does fix it, but that is simply mover_nig, at over
    2x welch_t's width.

    Affects both boundaries (one-inflated and zero-inflated saturating
    shapes), not one. Likert is unaffected because no likert shape saturates
    enough to yield a constant sample, so the regime does not arise there.
    """
    d = ta - tb
    lo = d - float(np.sqrt((ta - arm_a[0]) ** 2 + (arm_b[1] - tb) ** 2))
    hi = d + float(np.sqrt((arm_a[1] - ta) ** 2 + (tb - arm_b[0]) ** 2))
    return lo, hi


def _newcombe_hybrid_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Newcombe (1998) method 10, the "hybrid score" / square-and-add
    interval, Statistics in Medicine 17(8):873-890 ("Interval estimation for
    the difference between independent proportions: comparison of eleven
    methods").

    Build a Wilson interval (l_i, u_i) separately for each arm, then combine:

        lower = (pa - pb) - sqrt( (pa - la)^2 + (ub - pb)^2 )
        upper = (pa - pb) + sqrt( (ua - pa)^2 + (pb - lb)^2 )

    This is the MOVER (method of variance estimates recovery) construction:
    it takes each arm's variance estimate *at the relevant tail* rather than
    at the point estimate, which is what keeps it honest as either arm
    approaches a boundary. Same family as the paired ``newcombe_mover``
    already in the harness.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = float(np.sum(a_bin)), float(np.sum(b_bin))
    pa, pb = ka / na, kb / nb
    lo, hi = _mover_combine(pa, pb, _wilson_bounds(ka, na, alpha), _wilson_bounds(kb, nb, alpha))
    return max(-1.0, lo), min(1.0, hi)


def _mover_one_sample_ci(
    values: np.ndarray, alpha: float, fn, bounds: tuple[float, float],
) -> tuple[float, float]:
    """One arm's mean CI via a shipped [0, 1]-domain method, rescaled onto
    the eval type's own scale.

    Note this is a *cleaner* use of these methods than the paired path gets.
    ci_paired has to rescale onto [-span, span] so a zero difference lands at
    the methods' own centre, which is what forced its b0/4 correction to
    nig's prior (see ci_paired._NIG_PAIRED_DIFF_B0). Here each arm is a plain
    mean on the original [lo, hi] scale -- exactly ci_single's usage -- so
    the shipped priors apply unmodified.
    """
    return rescaled_ci(fn, values, alpha, bounds[0], bounds[1])


def _mover_t_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with plain t-interval arms -- the CONTROL for the MOVER family.

    Holds the arm interval fixed at an ordinary t-interval and varies only
    the combination rule, so comparing this against welch_t isolates what
    MOVER's square-and-add buys, and comparing mover_logit_t against THIS
    isolates what the logit arm buys. Without it, mover_logit_t vs welch_t
    confounds the two changes.

    ``bounds`` is accepted and unused (a t-interval needs no rescaling);
    it keeps the signature uniform across the MOVER family.
    """
    ci_a = t_interval_ci_1d(a, alpha)
    ci_b = t_interval_ci_1d(b, alpha)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _mover_logit_t_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with logit-t arms -- the unpaired sibling of the paired path's
    ``logit_t`` recommendation for bounded_01 data."""
    ci_a = _mover_one_sample_ci(a, alpha, logit_t_ci_1d, bounds)
    ci_b = _mover_one_sample_ci(b, alpha, logit_t_ci_1d, bounds)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _mover_nig_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with NIG arms -- the unpaired sibling of the paired path's
    ``nig`` recommendation for likert data."""
    ci_a = _mover_one_sample_ci(a, alpha, nig_ci_1d, bounds)
    ci_b = _mover_one_sample_ci(b, alpha, nig_ci_1d, bounds)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _fm_constrained_mle(pa: float, pb: float, na: int, nb: int, delta: float) -> tuple[float, float]:
    """Farrington & Manning (1990) closed-form constrained MLEs of (p_A, p_B)
    subject to p_A - p_B = delta, used by the Miettinen-Nurminen score
    interval. Statistics in Medicine 9(12):1447-1454.

    Solves the cubic that the constrained likelihood equations reduce to. The
    trigonometric branch below is the standard three-real-roots form; the
    guarded fallbacks handle the degenerate cases (u == 0, |v/u^3| > 1) that
    arise at the boundaries, where the cubic has a repeated root.
    """
    theta = nb / na
    aa = 1.0 + theta
    bb = -(1.0 + theta + pa + theta * pb + delta * (theta + 2.0))
    cc = delta * delta + delta * (2.0 * pa + theta + 1.0) + pa + theta * pb
    dd = -pa * delta * (1.0 + delta)

    v = bb**3 / (27.0 * aa**3) - bb * cc / (6.0 * aa**2) + dd / (2.0 * aa)
    inner = bb**2 / (9.0 * aa**2) - cc / (3.0 * aa)
    if inner <= 0.0:
        p1 = float(np.clip(pa, 0.0, 1.0))
        return p1, float(np.clip(p1 - delta, 0.0, 1.0))
    u = float(np.sign(v) if v != 0 else 1.0) * np.sqrt(inner)
    if u == 0.0:
        p1 = float(np.clip(-bb / (3.0 * aa), 0.0, 1.0))
        return p1, float(np.clip(p1 - delta, 0.0, 1.0))
    ratio = float(np.clip(v / u**3, -1.0, 1.0))
    w = (np.pi + np.arccos(ratio)) / 3.0
    p1 = float(np.clip(2.0 * u * np.cos(w) - bb / (3.0 * aa), 0.0, 1.0))
    p2 = float(np.clip(p1 - delta, 0.0, 1.0))
    return p1, p2


def _miettinen_nurminen_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Miettinen & Nurminen (1985) score interval for p_A - p_B, Statistics in
    Medicine 4(2):213-226.

    Inverts the score test: the interval is the set of delta for which the
    score statistic

        Z(delta) = (pa - pb - delta) / sqrt( V(delta) )
        V(delta) = ( p1~(1-p1~)/na + p2~(1-p2~)/nb ) * N/(N-1)

    (with p1~, p2~ the constrained MLEs at that delta, and the N/(N-1) term
    Miettinen-Nurminen's small-sample variance correction) satisfies
    |Z| <= z_{1-alpha/2}. Z is monotone decreasing in delta, so each endpoint
    is a single root, found here by bisection rather than the closed-form
    quartic -- slower per call but far harder to get subtly wrong, and the
    per-call cost is still negligible next to any bootstrap.

    Widely reported as the best-performing interval in the two-independent-
    proportions comparisons (it is what statsmodels calls method="score").
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa, pb = float(np.mean(a_bin)), float(np.mean(b_bin))
    n_tot = na + nb
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d_hat = pa - pb

    def _score(delta: float) -> float:
        p1, p2 = _fm_constrained_mle(pa, pb, na, nb, delta)
        var = (p1 * (1 - p1) / na + p2 * (1 - p2) / nb) * (n_tot / (n_tot - 1.0))
        if var <= 1e-14:
            # Degenerate constrained variance -- both arms pinned to a
            # boundary, e.g. 0/8 vs 8/8 evaluated at delta = -1. The score is
            # 0/0 exactly at delta = d_hat, and that case must return 0 (not
            # +-inf): d_hat is always inside its own interval, and reporting
            # +-inf there collapses the interval to the single point d_hat.
            # Away from d_hat the numerator is nonzero over a vanishing
            # variance, so delta really is decisively excluded.
            if abs(d_hat - delta) <= 1e-12:
                return 0.0
            return float(np.inf) if d_hat > delta else float(-np.inf)
        return (d_hat - delta) / float(np.sqrt(var))

    def _bisect(target: float, lo: float, hi: float) -> float:
        # _score is decreasing in delta; find where it crosses `target`.
        f_lo, f_hi = _score(lo) - target, _score(hi) - target
        if f_lo <= 0:
            return lo
        if f_hi >= 0:
            return hi
        # Tolerance-based rather than a fixed iteration count: the bracket
        # starts at width <= 2, so 1e-11 is reached in ~38 halvings, and
        # capping at 60 bounds the worst case. (A flat 80 iterations cost
        # ~1 ms/call, which made this the most expensive method in the
        # sweep by 20x for precision nobody uses.)
        for _ in range(60):
            if hi - lo < 1e-11:
                break
            mid = 0.5 * (lo + hi)
            if _score(mid) - target > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    eps = 1e-9
    lower = _bisect(z, -1.0 + eps, d_hat)
    upper = _bisect(-z, d_hat, 1.0 - eps)
    return max(-1.0, lower), min(1.0, upper)


def _fm_constrained_mle_vec(
    pa: np.ndarray, pb: np.ndarray, na: int, nb: int, delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``_fm_constrained_mle`` over arrays of observed proportions.

    Same cubic, same branch; exists because Agresti-Min needs the constrained
    MLE at every one of the (na+1)*(nb+1) possible tables for each candidate
    delta, and doing that with the scalar version is ~100x slower.
    """
    theta = nb / na
    aa = 1.0 + theta
    bb = -(1.0 + theta + pa + theta * pb + delta * (theta + 2.0))
    cc = delta * delta + delta * (2.0 * pa + theta + 1.0) + pa + theta * pb
    dd = -pa * delta * (1.0 + delta)

    v = bb**3 / (27.0 * aa**3) - bb * cc / (6.0 * aa**2) + dd / (2.0 * aa)
    inner = bb**2 / (9.0 * aa**2) - cc / (3.0 * aa)
    inner_safe = np.where(inner > 0, inner, 1.0)
    sign = np.where(v != 0, np.sign(v), 1.0)
    u = sign * np.sqrt(inner_safe)
    u_safe = np.where(np.abs(u) < 1e-300, 1.0, u)
    ratio = np.clip(v / u_safe**3, -1.0, 1.0)
    w = (np.pi + np.arccos(ratio)) / 3.0
    p1 = 2.0 * u * np.cos(w) - bb / (3.0 * aa)
    # Degenerate branches, matching the scalar version's fallbacks.
    p1 = np.where(inner > 0, p1, pa)
    p1 = np.where(np.abs(u) < 1e-300, -bb / (3.0 * aa), p1)
    p1 = np.clip(p1, 0.0, 1.0)
    return p1, np.clip(p1 - delta, 0.0, 1.0)


def _mn_score_all_tables(na: int, nb: int, delta: float) -> np.ndarray:
    """|Miettinen-Nurminen score statistic| at ``delta`` for every possible
    2x2 table, as an (na+1, nb+1) array indexed by (successes in A,
    successes in B). The test statistic Agresti-Min inverts."""
    ka = np.arange(na + 1)[:, None] / na
    kb = np.arange(nb + 1)[None, :] / nb
    pa = np.broadcast_to(ka, (na + 1, nb + 1))
    pb = np.broadcast_to(kb, (na + 1, nb + 1))
    p1, p2 = _fm_constrained_mle_vec(pa, pb, na, nb, delta)
    n_tot = na + nb
    var = (p1 * (1 - p1) / na + p2 * (1 - p2) / nb) * (n_tot / (n_tot - 1.0))
    num = pa - pb - delta
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 1e-14, num / np.sqrt(np.where(var > 1e-14, var, 1.0)), 0.0)
    # var == 0 happens only when both constrained MLEs sit on a boundary; the
    # table is then either exactly consistent with delta (num == 0, Z = 0) or
    # impossible under it (Z infinite).
    z = np.where((var <= 1e-14) & (np.abs(num) > 1e-12), np.inf, z)
    return np.abs(z)


_AGRESTI_MIN_GRID = 80
_AGRESTI_MIN_SCAN = 161
"""Delta values scanned to locate the OUTERMOST crossing of R = alpha before
refining. R is not monotone in delta, so a bracket has to be found by scan
rather than assumed; 161 points is a step of 0.0125, which resolves the
~0.02-wide discrepancies plain bisection was observed to miss. A gap in
{delta : R > alpha} narrower than one step could still be stepped over --
the interval would then be slightly short, so this is approximately rather
than provably exact."""
_AGRESTI_MIN_GAMMA = 1e-4
"""Berger-Boos confidence level for restricting the nuisance-parameter sup.
Berger & Boos (1994) show that taking the supremum over a 100(1-gamma)%
confidence set for the nuisance parameter and then adding gamma back keeps
the test valid while removing the far-tail nuisance values that otherwise
dominate the sup and make the interval needlessly wide."""


def _agresti_min_ci(
    a: np.ndarray, b: np.ndarray, alpha: float,
    n_grid: int = _AGRESTI_MIN_GRID, gamma: float = _AGRESTI_MIN_GAMMA,
) -> tuple[float, float]:
    """Agresti & Min (2001) exact unconditional confidence interval for
    p_A - p_B, inverting the Miettinen-Nurminen score test.

    Agresti & Min, "On small-sample confidence intervals for parameters in
    discrete distributions", Biometrics 57(3):963-971. This is Fagerland,
    Lydersen & Laake's *prime recommendation for small samples* (their
    Section 8.1 / Table 7): unlike the asymptotic score intervals it never
    dips below nominal coverage, and it behaves better than they do when a
    proportion sits near 0 or 1.

    For a candidate delta, the exact unconditional p-value is

        R(delta) = sup_{p1} P( |Z(X | delta)| >= |Z(x_obs | delta)| )

    where X ranges over all (na+1)(nb+1) possible tables, each weighted by
    Binom(na, p1) x Binom(nb, p1 - delta), and the supremum eliminates the
    nuisance parameter p1. The interval is {delta : R(delta) > alpha}.

    Two practical notes, both standard:

    * The sup is taken over a Berger-Boos 100(1-gamma)% confidence set for
      p1 rather than all of [0, 1], with gamma added back to R.
    * R is not guaranteed monotone in delta, so bisecting each side of the
      point estimate (as here, and as the standard implementations do) finds
      *a* crossing rather than provably the outermost one. In exchange the
      whole interval costs ~40 evaluations instead of a full grid sweep.

    This is by far the most expensive method in the slate -- roughly 100x a
    Welch interval -- because each evaluation rebuilds the score statistic
    for every possible table. That cost is the reason it is not in most
    software (Fagerland et al.'s Table 8 lists it only in StatXact).
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = int(np.sum(a_bin)), int(np.sum(b_bin))
    d_hat = ka / na - kb / nb

    # Berger-Boos confidence set for the nuisance parameter p1, from arm A.
    if gamma > 0:
        bb_lo = stats.beta.ppf(gamma / 2, ka, na - ka + 1) if ka > 0 else 0.0
        bb_hi = stats.beta.ppf(1 - gamma / 2, ka + 1, na - ka) if ka < na else 1.0
        bb_lo = 0.0 if not np.isfinite(bb_lo) else float(bb_lo)
        bb_hi = 1.0 if not np.isfinite(bb_hi) else float(bb_hi)
    else:
        bb_lo, bb_hi = 0.0, 1.0

    def _r(delta: float) -> float:
        z_tab = _mn_score_all_tables(na, nb, delta)
        z_obs = z_tab[ka, kb]
        if not np.isfinite(z_obs):
            return 0.0
        mask = (z_tab >= z_obs - 1e-9).astype(float)
        lo_p1 = max(bb_lo, max(0.0, delta))
        hi_p1 = min(bb_hi, min(1.0, 1.0 + delta))
        if hi_p1 < lo_p1:
            return 0.0
        p1s = np.linspace(lo_p1, hi_p1, n_grid)
        p2s = np.clip(p1s - delta, 0.0, 1.0)
        pmf_a = stats.binom.pmf(np.arange(na + 1)[:, None], na, p1s[None, :])
        pmf_b = stats.binom.pmf(np.arange(nb + 1)[:, None], nb, p2s[None, :])
        probs = np.einsum("ig,ij,jg->g", pmf_a, mask, pmf_b)
        return float(np.max(probs)) + gamma

    def _refine(inside: float, outside: float) -> float:
        """Bisect a bracket that is already known to straddle the OUTERMOST
        crossing. Returns ``outside``, the last delta known to be excluded,
        so any residual search error widens the interval rather than
        narrowing it."""
        for _ in range(40):
            if abs(outside - inside) < 1e-6:
                break
            mid = 0.5 * (inside + outside)
            if _r(mid) > alpha:
                inside = mid
            else:
                outside = mid
        return outside

    # R(delta) is NOT monotone, so bisecting straight out from d_hat finds
    # *a* crossing rather than the outermost one, and returns an interval
    # that is too short. Measured directly: at n_A=n_B=20 plain bisection
    # under-covered (exact minimum 0.9490 < 0.95) because on tables like
    # k=(20,0) and k=(18,1) it stopped 0.02 short of the true limit -- and
    # those are exactly the tables carrying the mass at the worst (p_A, p_B).
    # So scan a grid first, take the outermost delta that is still inside,
    # and only then refine within that one cell.
    eps = 1e-9
    grid = np.linspace(-1.0 + eps, 1.0 - eps, _AGRESTI_MIN_SCAN)
    inside_flags = np.array([_r(d) > alpha for d in grid])
    idx = np.flatnonzero(inside_flags)
    if idx.size == 0:
        return d_hat, d_hat  # degenerate; d_hat is always inside in exact arithmetic
    i_lo, i_hi = int(idx[0]), int(idx[-1])
    lower = -1.0 if i_lo == 0 else _refine(float(grid[i_lo]), float(grid[i_lo - 1]))
    upper = 1.0 if i_hi == grid.size - 1 else _refine(float(grid[i_hi]), float(grid[i_hi + 1]))
    return max(-1.0, min(lower, d_hat)), min(1.0, max(upper, d_hat))


def _bayes_beta_indep_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Independent Jeffreys Beta(1/2, 1/2) posteriors on p_A and p_B, sampled
    and subtracted.

    Legitimate here in a way it is not for paired data: the two groups really
    are independent, so the independence the model assumes is a fact of the
    design rather than an error. (Contrast ci_paired's ``bayes_indep_comp``,
    where the same construction is deliberately the *wrong* answer.)
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = float(np.sum(a_bin)), float(np.sum(b_bin))
    post_a = rng.beta(ka + 0.5, na - ka + 0.5, size=num_samples)
    post_b = rng.beta(kb + 0.5, nb - kb + 0.5, size=num_samples)
    diff = post_a - post_b
    return (float(np.percentile(diff, 100.0 * alpha / 2.0)),
            float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0))))


# ---------------------------------------------------------------------------
# Two-sample bootstrap family
# ---------------------------------------------------------------------------


def _two_sample_boot_diffs(
    a: np.ndarray, b: np.ndarray, method: str, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of mean(a) - mean(b) under independent resampling.

    Because the arms are independent, resampling each one separately and
    subtracting the resampled means is exactly the two-sample bootstrap -- so
    this delegates to the shipped single-sample resamplers rather than
    reimplementing them, and any fix to those propagates here.
    """
    if method == "bayes_bootstrap":
        fn = bayes_bootstrap_means_1d
    elif method == "smooth_bootstrap":
        fn = smooth_bootstrap_means_1d
    else:
        fn = bootstrap_means_1d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ma = fn(a, n_boot, rng, statistic="mean")
        mb = fn(b, n_boot, rng, statistic="mean")
    return np.asarray(ma) - np.asarray(mb)


def _two_sample_bca_ci(
    a: np.ndarray, b: np.ndarray, boot_diffs: np.ndarray, alpha: float,
) -> tuple[float, float]:
    """BCa for a two-sample difference of means.

    Bias correction z0 from the bootstrap distribution as usual; the
    acceleration comes from a jackknife that deletes one observation at a
    time from the *pooled* set of observations across both arms (deleting
    from A, then from B), which is the standard two-sample extension.
    """
    observed = float(np.mean(a) - np.mean(b))
    boot = np.asarray(boot_diffs, dtype=float)
    n_boot = boot.size
    prop = float(np.mean(boot < observed))
    if prop <= 0.0 or prop >= 1.0 or n_boot == 0:
        return (float(np.percentile(boot, 100 * alpha / 2)),
                float(np.percentile(boot, 100 * (1 - alpha / 2))))
    z0 = float(stats.norm.ppf(prop))

    na, nb = a.size, b.size
    sum_a, sum_b = float(np.sum(a)), float(np.sum(b))
    jack = np.empty(na + nb, dtype=float)
    if na > 1:
        jack[:na] = (sum_a - a) / (na - 1) - sum_b / nb
    else:
        jack[:na] = observed
    if nb > 1:
        jack[na:] = sum_a / na - (sum_b - b) / (nb - 1)
    else:
        jack[na:] = observed
    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    denom = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    acc = float(np.sum(diffs**3)) / denom if denom > 0 else 0.0

    def _adj(q: float) -> float:
        zq = float(stats.norm.ppf(q))
        num = z0 + zq
        den = 1.0 - acc * num
        if abs(den) < 1e-12:
            return q
        return float(stats.norm.cdf(z0 + num / den))

    lo_q = float(np.clip(_adj(alpha / 2), 1e-6, 1 - 1e-6))
    hi_q = float(np.clip(_adj(1 - alpha / 2), 1e-6, 1 - 1e-6))
    return float(np.percentile(boot, 100 * lo_q)), float(np.percentile(boot, 100 * hi_q))


def _two_sample_bootstrap_t_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator,
) -> tuple[float, float]:
    """Studentized (bootstrap-t) interval for a two-sample mean difference.

    Each bootstrap replicate is studentized by its OWN Welch standard error,
    so the resulting quantiles are of a pivotal quantity rather than of the
    raw difference -- the reason bootstrap-t usually beats the percentile
    bootstrap at small n.
    """
    na, nb = a.size, b.size
    observed = float(np.mean(a) - np.mean(b))
    if na < 2 or nb < 2:
        return observed, observed
    se_obs = float(np.sqrt(np.var(a, ddof=1) / na + np.var(b, ddof=1) / nb))
    if se_obs <= 0:
        return observed, observed

    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    sa, sb = a[idx_a], b[idx_b]
    ma, mb = sa.mean(axis=1), sb.mean(axis=1)
    va, vb = sa.var(axis=1, ddof=1), sb.var(axis=1, ddof=1)
    se_boot = np.sqrt(va / na + vb / nb)
    # Replicates with zero resampled variance carry no information about the
    # pivot's tails; drop them rather than letting them become +-inf.
    ok = se_boot > 0
    if not np.any(ok):
        return observed, observed
    t_stats = (ma[ok] - mb[ok] - observed) / se_boot[ok]
    t_lo = float(np.percentile(t_stats, 100 * (1 - alpha / 2)))
    t_hi = float(np.percentile(t_stats, 100 * (alpha / 2)))
    return observed - t_lo * se_obs, observed - t_hi * se_obs


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _draw_unpaired(
    source: CIPairSource, rng: np.random.Generator, n_a: int, n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two INDEPENDENT groups from a paired source.

    Arm A is taken from one generate_pair call and arm B from a second,
    so no item is shared. Each arm keeps its own marginal distribution
    (including the effect shift on B), which is what makes source.true_diff
    still the right coverage target.
    """
    a, _ = source.generate_pair(rng, n_a, 1)
    _, b = source.generate_pair(rng, n_b, 1)
    return a[:, 0], b[:, 0]


def _run_cell(
    source: CIPairSource, n_a: int, n_b: int, n_reps: int, n_bootstrap: int, bayes_n: int,
    alpha: float, seed,
    method_names: frozenset[str] | None = None,
    agresti_min_max_n: int = _AGRESTI_MIN_MAX_N,
    base_n: int = 0,
) -> list[SimResult]:
    """Run all reps for one (source, n_a, n_b) cell.

    ``method_names``, if given, restricts computation (not just reporting) to
    methods whose ``.name`` is in the set, which matters because the slate
    spans a wide cost range -- e.g. agresti_min is ~100x a Welch interval
    (see ``_agresti_min_ci``) -- so filtering a sweep down to a few methods
    should skip the others' work entirely rather than compute and discard
    it. ``None`` (default) runs every applicable method for the source's
    eval type, matching prior behavior.
    """
    rng = np.random.default_rng(seed)
    is_binary = source.eval_type == "binary"
    scale_bounds = source.scale_bounds or EVAL_TYPE_SCALE_BOUNDS[source.eval_type]

    def _want(m) -> bool:
        return method_names is None or m.name in method_names

    mean_methods = [m for m in BOOTSTRAP_METHODS if _want(m)]
    mean_methods += [m for m in UNPAIRED_MEAN_EXTRA_METHODS
                     if _want(m) and not (is_binary and m in _BINARY_INELIGIBLE)]
    if is_binary:
        mean_methods += [
            m for m in UNPAIRED_BINARY_METHODS
            if _want(m) and not (m is AGRESTI_MIN and max(n_a, n_b) > agresti_min_max_n)
        ]

    all_methods = list(mean_methods)
    covered = {k: 0 for k in all_methods}
    total_w = {k: 0.0 for k in all_methods}
    total_score = {k: 0.0 for k in all_methods}
    pen_under = {k: 0.0 for k in all_methods}
    pen_over = {k: 0.0 for k in all_methods}
    rejects = {k: 0 for k in all_methods}
    total_t = {k: 0.0 for k in all_methods}
    total_t_sq = {k: 0.0 for k in all_methods}

    target = float(source.true_diff)
    scale_span = float(scale_bounds[1] - scale_bounds[0]) or 1.0

    def _record(key, lo: float, hi: float, elapsed: float) -> None:
        if lo <= target <= hi:
            covered[key] += 1
        total_w[key] += hi - lo
        total_score[key] += interval_score(lo, hi, target, alpha)
        if target < lo:
            pen_under[key] += (2.0 / alpha) * (lo - target)
        elif target > hi:
            pen_over[key] += (2.0 / alpha) * (target - hi)
        if lo > 0.0 or hi < 0.0:
            rejects[key] += 1
        total_t[key] += elapsed
        total_t_sq[key] += elapsed * elapsed

    for _rep in range(n_reps):
        a, b = _draw_unpaired(source, rng, n_a, n_b)
        obs_mean = float(np.mean(a) - np.mean(b))

        # --- mean-difference family ---------------------------------------
        # bootstrap and bca share one set of resampled differences (they are
        # the same draws, read off differently). That shared draw is timed
        # once and its cost added to BOTH consumers -- charging it only to
        # whichever method happened to run first would make the other look
        # ~20x cheaper than it is, and the cost/coverage tradeoff is one of
        # the things this sweep is meant to inform.
        shared_boot: np.ndarray | None = None
        shared_boot_t = 0.0
        if any(m in (BOOTSTRAP, BCA) for m in mean_methods):
            _t0 = time.perf_counter()
            shared_boot = _two_sample_boot_diffs(a, b, "bootstrap", n_bootstrap, rng)
            shared_boot_t = time.perf_counter() - _t0

        for method in mean_methods:
            key = method
            extra_t = shared_boot_t if method in (BOOTSTRAP, BCA) else 0.0
            t0 = time.perf_counter()
            try:
                if method is BOOTSTRAP:
                    lo = float(np.percentile(shared_boot, 100 * alpha / 2))
                    hi = float(np.percentile(shared_boot, 100 * (1 - alpha / 2)))
                elif method is BCA:
                    lo, hi = _two_sample_bca_ci(a, b, shared_boot, alpha)
                elif method is BAYES_BOOTSTRAP:
                    bd = _two_sample_boot_diffs(a, b, "bayes_bootstrap", n_bootstrap, rng)
                    lo = float(np.percentile(bd, 100 * alpha / 2))
                    hi = float(np.percentile(bd, 100 * (1 - alpha / 2)))
                elif method is SMOOTH_BOOTSTRAP:
                    bd = _two_sample_boot_diffs(a, b, "smooth_bootstrap", n_bootstrap, rng)
                    lo = float(np.percentile(bd, 100 * alpha / 2))
                    hi = float(np.percentile(bd, 100 * (1 - alpha / 2)))
                elif method is BOOTSTRAP_T:
                    lo, hi = _two_sample_bootstrap_t_ci(a, b, n_bootstrap, alpha, rng)
                elif method is WELCH_T:
                    lo, hi = _welch_t_ci(a, b, alpha)
                elif method is STUDENT_T:
                    lo, hi = _student_t_ci(a, b, alpha)
                elif method is MOVER_T:
                    lo, hi = _mover_t_ci(a, b, alpha, scale_bounds)
                elif method is MOVER_LOGIT_T:
                    lo, hi = _mover_logit_t_ci(a, b, alpha, scale_bounds)
                elif method is MOVER_NIG:
                    lo, hi = _mover_nig_ci(a, b, alpha, scale_bounds)
                elif method is WALD_UNPAIRED:
                    lo, hi = _wald_unpaired_ci(a, b, alpha)
                elif method is AGRESTI_CAFFO:
                    lo, hi = _agresti_caffo_ci(a, b, alpha)
                elif method is NEWCOMBE_HYBRID:
                    lo, hi = _newcombe_hybrid_ci(a, b, alpha)
                elif method is MIETTINEN_NURMINEN:
                    lo, hi = _miettinen_nurminen_ci(a, b, alpha)
                elif method is AGRESTI_MIN:
                    lo, hi = _agresti_min_ci(a, b, alpha)
                elif method is BAYES_BETA_INDEP:
                    lo, hi = _bayes_beta_indep_ci(a, b, alpha, bayes_n, rng)
                else:
                    raise AssertionError(f"unhandled mean-family method {method.name!r}")
            except Exception:
                lo = hi = obs_mean
            _record(key, lo, hi, time.perf_counter() - t0 + extra_t)

    out: list[SimResult] = []
    for method in all_methods:
        out.append(SimResult(
            source=source.source, label=source.label, eval_type=source.eval_type,
            n_a=n_a, n_b=n_b, base_n=base_n or n_a, method=method.name, n_reps=n_reps,
            covered=covered[method], total_width=total_w[method], total_score=total_score[method],
            total_pen_under=pen_under[method], total_pen_over=pen_over[method],
            rejects=rejects[method], total_time=total_t[method], total_time_sq=total_t_sq[method],
            is_null=source.is_null, true_value=target, scale_span=scale_span,
            icc=source.icc, cohens_d=source.cohens_d,
        ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
        eta_sec = max(self.total - step, 0) / max(step / elapsed, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        prefix = f"{self.label}: " if self.label else ""
        print(f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
              f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}  {detail[:40]:<40s}", end="", flush=True)
        if is_final:
            print()


_CELL_SOURCES: list = []


def _run_cell_worker(args: tuple) -> list[SimResult]:
    (sc_idx, n_a, n_b, base, n_reps, n_bootstrap, bayes_n, alpha, seed,
     method_names, agresti_min_max_n) = args
    return _run_cell(
        _CELL_SOURCES[sc_idx], n_a, n_b, n_reps, n_bootstrap, bayes_n, alpha,
        seed, method_names, agresti_min_max_n, base_n=base,
    )


def run_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], size_ratios: list[float],
    n_reps: int, n_bootstrap: int, bayes_n: int, alpha: float,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    method_names: frozenset[str] | None = None,
    agresti_min_max_n: int = _AGRESTI_MIN_MAX_N,
) -> list[SimResult]:
    """Run the full Monte Carlo sweep: every source x sample size x size
    ratio cell, each producing one ``SimResult`` per applicable method.

    Builds the cell list (applying ``size_ratios`` to a randomly-chosen arm
    per cell, see below), binds a reproducible child seed to each cell via
    ``np.random.SeedSequence``, shuffles execution order for a representative
    ETA, then runs cells serially or across ``n_workers`` processes.
    """
    global _CELL_SOURCES
    _CELL_SOURCES = list(sources)

    # Which arm carries the imbalance is randomised per cell, seeded so the
    # sweep stays reproducible. Arm B is the effect-carrying one, so applying
    # the ratio to it always -- as this did -- means the larger group is
    # always the shifted one, and on a bounded scale shifting toward a
    # boundary changes the variance, so only one pairing of (size, variance)
    # was ever tested. Scaling the OTHER arm up instead of scaling B down
    # keeps both groups at or above the base size, which a ratio below 1
    # would not: at base 10 it would put a group at 5, under the N=15 floor
    # below which evalstats refuses to report at all.
    _dir_rng = np.random.default_rng(seed ^ 0x5A17)
    cells = []
    for i, s in enumerate(sources):
        for n in sample_sizes:
            for ratio in size_ratios:
                scaled = max(2, int(round(n * ratio)))
                if ratio == 1.0 or _dir_rng.random() < 0.5:
                    cells.append((i, n, n, scaled))       # arm B larger
                else:
                    cells.append((i, n, scaled, n))       # arm A larger
    cells = [(i, na, nb, base) for (i, base, na, nb) in
             ((c[0], c[1], c[2], c[3]) for c in cells)]
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [
        (sc_idx, n_a, n_b, base, n_reps, n_bootstrap, bayes_n, alpha, cseed,
         method_names, agresti_min_max_n)
        for (sc_idx, n_a, n_b, base), cseed in zip(cells, child_seeds)
    ]
    # Shuffle the EXECUTION order (seeds are already bound to their cell above,
    # so results are unchanged and still reproducible). build_pair_sources emits
    # all binary sources first, and binary is by far the most expensive eval
    # type, so running in natural order makes every early ETA an extrapolation
    # from the worst case -- a 1.7 h run reported itself as 15 h. A shuffled
    # order makes the completed prefix a representative sample of the whole.
    np.random.default_rng(seed).shuffle(args_list)

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="ci_unpaired")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_cell_worker(a))
            sc_idx, n_a, n_b, _base = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n_a}/{n_b}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Exact coverage (binary only) -- no Monte Carlo
# ---------------------------------------------------------------------------
# For a binary outcome every interval is a function of the two success counts
# alone, and (k_A, k_B) has a known joint distribution. So coverage can be
# summed exactly over all (n_A+1)(n_B+1) tables instead of estimated by
# simulation: no reps, no seeds, no Monte Carlo band. This is also how the
# two-independent-proportions literature reports coverage (Fagerland,
# Lydersen & Laake's figures are exact curves, not simulations), so it makes
# our binary numbers directly comparable to theirs.
#
# It matters practically: at 200-300 reps the Monte Carlo band on a coverage
# estimate is about +-0.03, which is the same size as the entire spread
# between the good binary methods. The sweep's `minCov` column is dominated
# by that noise -- it reported agresti_min dipping below agresti_caffo, which
# exact enumeration shows cannot happen.


@dataclass
class ExactResult:
    """One (method, n_a, n_b) cell's exactly-enumerated coverage: the
    worst-case and mean coverage over a (p_A, p_B) grid, the grid point
    where the worst case occurs, and the mean interval width -- all computed
    by exact enumeration over every possible table, not Monte Carlo."""
    method: str
    n_a: int
    n_b: int
    min_coverage: float
    mean_coverage: float
    worst_p_a: float
    worst_p_b: float
    mean_width: float


_EXACT_BINARY_METHODS = (
    WALD_UNPAIRED, AGRESTI_CAFFO, NEWCOMBE_HYBRID, MIETTINEN_NURMINEN,
    AGRESTI_MIN, WELCH_T, STUDENT_T, MOVER_T, MOVER_LOGIT_T,
)
"""Deterministic binary methods only, i.e. every method whose interval is a
function of (k_A, k_B) alone -- verified deterministic across repeated calls,
which is what enumeration requires. bayes_beta_indep and the bootstrap family
are excluded because they are randomised, so no finite table of intervals
represents them and exact enumeration does not apply to them at all."""


def _exact_binary_ci_table(method, n_a: int, n_b: int, alpha: float) -> dict:
    """Every interval this method produces at (n_a, n_b), keyed by the two
    success counts. Computed once and reused across the whole (p_a, p_b)
    grid -- the interval does not depend on p."""
    fns = {
        WALD_UNPAIRED: _wald_unpaired_ci, AGRESTI_CAFFO: _agresti_caffo_ci,
        NEWCOMBE_HYBRID: _newcombe_hybrid_ci, MIETTINEN_NURMINEN: _miettinen_nurminen_ci,
        AGRESTI_MIN: _agresti_min_ci, WELCH_T: _welch_t_ci, STUDENT_T: _student_t_ci,
    }
    out = {}
    for ka in range(n_a + 1):
        a = np.r_[np.ones(ka), np.zeros(n_a - ka)]
        for kb in range(n_b + 1):
            b = np.r_[np.ones(kb), np.zeros(n_b - kb)]
            try:
                mover = {MOVER_T: _mover_t_ci, MOVER_LOGIT_T: _mover_logit_t_ci,
                         MOVER_NIG: _mover_nig_ci}.get(method)
                if mover is not None:
                    out[(ka, kb)] = mover(a, b, alpha, (0.0, 1.0))
                else:
                    out[(ka, kb)] = fns[method](a, b, alpha)
            except Exception:
                d = ka / n_a - kb / n_b
                out[(ka, kb)] = (d, d)
    return out


def run_exact_coverage(
    size_pairs: list[tuple[int, int]], p_grid: np.ndarray, alpha: float,
    method_names: frozenset[str] | None = None, progress_mode: str = "bar",
) -> list[ExactResult]:
    methods = [m for m in _EXACT_BINARY_METHODS
               if method_names is None or m.name in method_names]
    results: list[ExactResult] = []
    reporter = _ProgressReporter(len(size_pairs) * len(methods),
                                 mode=progress_mode, label="ci_unpaired/exact")
    step = 0
    for n_a, n_b in size_pairs:
        pmf_a = {p: stats.binom.pmf(np.arange(n_a + 1), n_a, p) for p in p_grid}
        pmf_b = {p: stats.binom.pmf(np.arange(n_b + 1), n_b, p) for p in p_grid}
        for method in methods:
            table = _exact_binary_ci_table(method, n_a, n_b, alpha)
            los = np.array([[table[(ka, kb)][0] for kb in range(n_b + 1)] for ka in range(n_a + 1)])
            his = np.array([[table[(ka, kb)][1] for kb in range(n_b + 1)] for ka in range(n_a + 1)])
            widths = his - los
            best = (1.0, 0.0, 0.0)
            covs, wmeans = [], []
            for pa in p_grid:
                wa = pmf_a[pa]
                for pb in p_grid:
                    wb = pmf_b[pb]
                    w = np.outer(wa, wb)
                    cov = float(np.sum(w * ((los <= pa - pb) & (pa - pb <= his))))
                    covs.append(cov)
                    wmeans.append(float(np.sum(w * widths)))
                    if cov < best[0]:
                        best = (cov, float(pa), float(pb))
            results.append(ExactResult(
                method=method.name, n_a=n_a, n_b=n_b,
                min_coverage=best[0], mean_coverage=float(np.mean(covs)),
                worst_p_a=best[1], worst_p_b=best[2], mean_width=float(np.mean(wmeans)),
            ))
            step += 1
            reporter.update(step, detail=f"{method.name} n={n_a}/{n_b}")
    reporter.update(len(size_pairs) * len(methods), detail="done")
    return results


def print_exact_report(results: list[ExactResult], alpha: float, n_grid_points: int) -> None:
    target = 1.0 - alpha
    print()
    print("=" * 100)
    print(f"EXACT binary coverage -- enumerated over all tables, {n_grid_points}x{n_grid_points} "
          f"(p_A, p_B) grid. No Monte Carlo.")
    print("=" * 100)
    by_size: dict = defaultdict(list)
    for r in results:
        by_size[(r.n_a, r.n_b)].append(r)
    for (n_a, n_b) in sorted(by_size):
        print(f"\n  n_A={n_a}, n_B={n_b}   (target {target:.2f})")
        print(f"  {'method':<22s} {'min cov':>9s} {'mean cov':>9s} {'width':>8s}  "
              f"{'worst at (p_A,p_B)':>20s}   verdict")
        rows = sorted(by_size[(n_a, n_b)], key=lambda r: -r.min_coverage)
        for r in rows:
            verdict = "holds nominal" if r.min_coverage >= target - 1e-9 else f"dips {r.min_coverage:.4f}"
            print(f"  {r.method:<22s} {r.min_coverage:9.4f} {r.mean_coverage:9.4f} "
                  f"{r.mean_width:8.4f}  {'(' + format(r.worst_p_a, '.2f') + ', ' + format(r.worst_p_b, '.2f') + ')':>20s}   {verdict}")


def save_exact_csv(results: list[ExactResult], out_dir: str, run_stem: str) -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_stem}.csv"
    rows = [asdict(r) for r in results]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved exact-coverage CSV: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    """Same glyphs ci_paired uses, so the two cases' logs read alike."""
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
    per_n_vals: dict, m: str, sizes_present: list[int],
) -> tuple[float, float, float, float]:
    """Headline (Cov, Width, Score, Penalty) for method `m`: average per n
    first, then average those across n -- rather than pooling every
    (scenario, n) cell into one flat list, which implicitly weights each n by
    how many scenarios have data there. Matters most for Score: a method that
    under-covers only at small n should have that show up in the headline
    rather than be diluted by unrelated large-n cells. Same construction as
    ci_paired's function of the same name."""
    per_n_means = []
    for n in sizes_present:
        vals = per_n_vals.get((m, n))
        if vals:
            per_n_means.append(tuple(float(np.mean([v[i] for v in vals])) for i in range(4)))
    if not per_n_means:
        return (float("nan"),) * 4
    return tuple(float(np.mean([pm[i] for pm in per_n_means])) for i in range(4))


def _decision_rates(results: list[SimResult]) -> tuple[dict, dict]:
    """(type1, power) keyed by (eval_type, method).

    Type I is the reject rate on null cells; power is the reject rate on the
    alternative cells, averaged per scenario first so every effect size and
    shape weighs equally rather than by how many cells each contributes.
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
        c = sum(x[0] for x in cells)
        n = sum(x[1] for x in cells)
        if n:
            by_method[(et, m)].append(c / n)
    power = {k: float(np.mean(v)) for k, v in by_method.items() if v}
    return type1, power


def _print_overall_summary_table(
    title: str, eval_types: list[str], results: list[SimResult], agg: dict,
    agg_counts: dict, target: float, sizes_present: list[int],
    type1: dict | None = None, power: dict | None = None,
) -> None:
    """One OVERALL SUMMARY table, aggregated over `eval_types`. No-ops if none
    of them are present, so callers can request a table unconditionally."""
    present_methods = {r.method for r in results if r.eval_type in eval_types}
    if not present_methods:
        return
    method_labels = [m.name for m in order_present_methods(present_methods)]

    per_n_vals: dict = defaultdict(list)
    all_counts: dict = defaultdict(lambda: (0, 0))
    per_n_counts: dict = defaultdict(lambda: (0, 0))
    min_cov: dict = defaultdict(lambda: float("inf"))
    for (et, m, n), vals in agg.items():
        if et not in eval_types:
            continue
        per_n_vals[(m, n)].extend(vals)
        c, t = agg_counts[(et, m, n)]
        cp, tp = all_counts[m]
        all_counts[m] = (cp + c, tp + t)
        cpn, tpn = per_n_counts[(m, n)]
        per_n_counts[(m, n)] = (cpn + c, tpn + t)
        min_cov[m] = min(min_cov[m], min(v[0] for v in vals))

    n_cols_hdr = "".join(f"  {'n=' + str(n):>7}" for n in sizes_present)
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")
    print("  MinCov = worst per-scenario coverage seen for that method (not an average) --\n"
          "  flags methods whose good mean coverage hides an unreliable scenario/n cell.")
    print("  TypeI = P(CI excludes 0) on null cells (target alpha); Power = the same rate\n"
          "  on the alternative cells, averaged over scenarios.")
    print("  Width, Penalty and Score are expressed as a FRACTION OF EACH SOURCE'S OWN\n"
          "  SCALE. They are homogeneous of degree 1 in the scale's units, so this is\n"
          "  exactly their value on the [0,1]-rescaled data -- and it is what makes them\n"
          "  poolable at all, since real corpora mix 1-5, 1-7, 0-10 ... outcomes within\n"
          "  one eval type. Coverage, MinCov and TypeI are already scale-free.")
    print("  Score = Width + Penalty, reported separately because Score is ~90% Width,\n"
          "  so a too-narrow method can post the best Score while under-covering.\n"
          "  The two are one-sided in OPPOSITE directions: Width penalises intervals\n"
          "  that are too WIDE, Penalty ((2/alpha) x mean miss distance) those that are\n"
          "  too NARROW. Neither means 'calibration' on its own. Read Width, Penalty\n"
          "  and Cov/MinCov together.")
    print(f"\n  {'Method':<20}  {'Cov':>6}  {'MinCov':>7}  {'Band95':>13}  {'Width':>8}  "
          f"{'Penalty':>8}  {'Score':>8}  {'TypeI':>7}  {'Power':>7}  {'Time(ms)':>14}{n_cols_hdr}")
    _et_key = eval_types[0] if len(eval_types) == 1 else None
    for m in method_labels:
        mc, mw, ms, mp = _headline_cov_width_score(per_n_vals, m, sizes_present)
        c_tot, t_tot = all_counts[m]
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats(
            [r for r in results if r.method == m and r.eval_type in eval_types])
        time_str = f"{avg_ms:.3f}+-{se_ms:.3f}" if np.isfinite(avg_ms) else "-"
        worst = min_cov[m]
        worst_str = f"{worst:.3f}{_cov_marker(worst, target)}" if np.isfinite(worst) else "-"
        n_cols_vals = ""
        for n in sizes_present:
            c_n, t_n = per_n_counts.get((m, n), (0, 0))
            cov_n = c_n / t_n if t_n > 0 else float("nan")
            n_cols_vals += (f"  {cov_n:>5.3f}{_cov_marker(cov_n, target)} "
                            if np.isfinite(cov_n) else f"  {'  -':>7}")
        t1s = f"{type1[(_et_key, m)]:.3f}" if type1 and (_et_key, m) in type1 else "-"
        pws = f"{power[(_et_key, m)]:.3f}" if power and (_et_key, m) in power else "-"
        print(f"  {m:<20}  {mc:>5.3f}{_cov_marker(mc, target)}  {worst_str:>7}  "
              f"{f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {mp:>8.4f}  {ms:>8.4f}  "
              f"{t1s:>7}  {pws:>7}  {time_str:>14}{n_cols_vals}")


def print_report(results: list[SimResult], alpha: float, sample_sizes: list[int]) -> None:
    """Print the console report: a per-(eval type, n) coverage grid followed
    by one OVERALL SUMMARY table per eval type (coverage, width, score,
    penalty, Type I/power, timing). Aggregates non-null cells per scenario
    per n before averaging, matching ``_headline_cov_width_score``."""
    target = 1.0 - alpha
    n_reps = results[0].n_reps if results else 0
    type1_map, power_map = _decision_rates(results)
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    method_labels = [m.name for m in order_present_methods({r.method for r in non_null})]

    agg: dict = defaultdict(list)
    agg_counts: dict = defaultdict(lambda: (0, 0))
    for r in non_null:
        n = max(r.n_reps, 1)
        sp = r.scale_span or 1.0
        agg[(r.eval_type, r.method, r.base_n or r.n_a)].append((
            r.covered / n, r.total_width / n / sp, r.total_score / n / sp,
            (r.total_pen_under + r.total_pen_over) / n / sp,
        ))
        cp, tp = agg_counts[(r.eval_type, r.method, r.base_n or r.n_a)]
        agg_counts[(r.eval_type, r.method, r.base_n or r.n_a)] = (cp + r.covered, tp + r.n_reps)

    def mean_cov(et, m, n):
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    sep = "=" * 72
    print(f"\n{sep}\n  CI_UNPAIRED COVERAGE -- SIMULATION RESULTS\n"
          f"  Estimand: between-subjects mean difference mean(A) - mean(B)\n"
          f"  Nominal coverage: {target:.0%}   |   reps/cell: {n_reps}\n"
          f"  v = under-covered   ^ = over-conservative\n"
          f"  Score = interval score (width + (2/alpha)*miss-distance; lower is better --\n"
          f"  see evalstats.core.stats_utils.interval_score)\n{sep}")

    sizes_present = sorted({(r.base_n or r.n_a) for r in non_null})
    for et in eval_types_present:
        print(f"\n  [{et}]")
        print(f"    {'Method':<20}" + "".join(f"  n={n:<6}" for n in sizes_present))
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sizes_present:
                cov = mean_cov(et, m, n)
                row += "  " + (" " * 7 if np.isnan(cov)
                               else f"{cov:.3f}{_cov_marker(cov, target)}".ljust(8))
            print(row)

    # One OVERALL SUMMARY per eval type rather than a pooled table: these data
    # types are answered by different method families on different scales
    # (a binary Delta-p width and a likert 1-5 width are not comparable), and
    # only binary runs the dedicated proportion intervals at all.
    for et in eval_types_present:
        _print_overall_summary_table(
            f"OVERALL SUMMARY -- {et.upper()}", [et], non_null, agg, agg_counts,
            target, sizes_present, type1=type1_map, power=power_map,
        )


def latex_summary(results: list[SimResult], alpha: float) -> str:
    """Booktabs summary, mirroring ci_paired.latex_overall_summary.

    Rows are blocked by eval type (bin/cont/lik) with a midrule between
    blocks, and per-n coverage columns are appended on the right because the
    aggregate column can hide miscalibration that only shows up at one end of
    the size range.
    """
    target = 1.0 - alpha
    rows_in = [r for r in results if not r.is_null]
    if not rows_in:
        return ""
    method_labels = [m.name for m in order_present_methods({r.method for r in rows_in})]
    sizes_present = sorted({(r.base_n or r.n_a) for r in rows_in})

    agg: dict = defaultdict(list)
    counts: dict = defaultdict(lambda: (0, 0))
    per_n: dict = defaultdict(lambda: (0, 0))
    for r in rows_in:
        g = report_eval_type_group(r.eval_type)
        n = max(r.n_reps, 1)
        sp = r.scale_span or 1.0
        agg[(g, r.method)].append((
            r.covered / n, r.total_width / n / sp, r.total_score / n / sp,
            (r.total_pen_under + r.total_pen_over) / n / sp,
        ))
        c, t = counts[(g, r.method)]
        counts[(g, r.method)] = (c + r.covered, t + r.n_reps)
        c2, t2 = per_n[(g, r.method, r.base_n or r.n_a)]
        per_n[(g, r.method, r.base_n or r.n_a)] = (c2 + r.covered, t2 + r.n_reps)

    # Type-I error lives on the null rows, which are excluded from `rows_in`.
    null_rate: dict = defaultdict(lambda: (0, 0))
    for r in results:
        if not r.is_null:
            continue
        g = report_eval_type_group(r.eval_type)
        c, t = null_rate[(g, r.method)]
        null_rate[(g, r.method)] = (c + r.rejects, t + r.n_reps)

    groups_present = [g for g in ("bin", "cont", "lik", "grades")
                      if any(k[0] == g for k in agg)]
    columns = (["Method", "Cov", "MinCov", "Width", r"Pen $\downarrow$", r"Score $\downarrow$",
                "TypeI"] + [f"$n{{=}}{n}$" for n in sizes_present])
    rows: list[list[str]] = []
    rule_before: set[int] = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        score_vals: list[float] = []
        for m in method_labels:
            if (g, m) not in agg:
                continue
            vals = agg[(g, m)]
            covs = np.array([v[0] for v in vals])
            width = float(np.mean([v[1] for v in vals]))
            score = float(np.mean([v[2] for v in vals]))
            pen = float(np.mean([v[3] for v in vals]))
            c, t = counts[(g, m)]
            pooled = c / t if t else float("nan")
            rc, rt = null_rate[(g, m)]
            t1 = rc / rt if rt else float("nan")
            cells = [
                f"{escape_latex(m)} ({g})",
                coverage_cell(pooled, target),
                coverage_cell(float(covs.min()), target),
                f"{width:.4f}", f"{pen:.4f}", f"{score:.4f}",
                f"{t1:.3f}" if np.isfinite(t1) else "--",
            ]
            for n in sizes_present:
                c2, t2 = per_n[(g, m, n)]
                cells.append(coverage_cell(c2 / t2, target) if t2 else "--")
            rows.append(cells)
            score_vals.append(score)
        block = [row[5] for row in rows[block_start:]]
        marked = mark_best_and_runnerup(block, score_vals, higher_is_better=False)
        for i, cell in enumerate(marked):
            rows[block_start + i][5] = cell

    return booktabs_table(
        caption=(r"Between-subjects (unpaired) pairwise CI calibration for the mean "
                 r"difference $\bar{A}-\bar{B}$. "
                 f"Non-null cells only; TypeI is the rejection rate on null cells. "
                 f"Coverage cells are shaded when they fall outside the acceptable band "
                 f"around {target:.2f}. Best Score per block in bold, runner-up underlined. "
                 r"Width, Pen and Score are expressed as a fraction of each source's own "
                 r"measurement scale, which is exactly their value on the $[0,1]$-rescaled "
                 r"data and is what makes them poolable across outcomes measured on "
                 r"different scales."),
        label="tab:ci_unpaired_overall",
        columns=columns, rows=rows, rule_before=rule_before,
    )


def latex_exact_summary(results: list, alpha: float) -> str:
    """Booktabs table for the exact binary coverage mode."""
    if not results:
        return ""
    target = 1.0 - alpha
    by_size = sorted({(r.n_a, r.n_b) for r in results})
    methods = [m.name for m in order_present_methods({r.method for r in results})]
    lookup = {(r.method, r.n_a, r.n_b): r for r in results}
    columns = ["Method"] + [f"$n{{=}}{a}/{b}$" for a, b in by_size] + ["Width"]
    rows = []
    for m in methods:
        cells = [escape_latex(m)]
        widths = []
        for a, b in by_size:
            r = lookup.get((m, a, b))
            cells.append(coverage_cell(r.min_coverage, target) if r else "--")
            if r:
                widths.append(r.mean_width)
        cells.append(f"{np.mean(widths):.3f}" if widths else "--")
        rows.append(cells)
    return booktabs_table(
        caption=(f"Exact MINIMUM coverage for a difference of two independent proportions, "
                 f"enumerated over every $(k_A, k_B)$ table and a grid of $(p_A, p_B)$. "
                 f"No Monte Carlo, so these are exact values rather than estimates with a "
                 f"simulation band, and directly comparable to the published coverage curves "
                 f"in the two-independent-proportions literature. Target {target:.2f}."),
        label="tab:ci-unpaired-exact",
        columns=columns, rows=rows,
    )


def save_results_artifacts(
    results: list[SimResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    """Write the per-cell results CSV (``{run_stem}_results.csv``, one row per
    ``SimResult`` with derived rates/means added) and the console report,
    captured to ``{run_stem}_summary.log``; appends LaTeX tables to the log
    when ``latex`` is set. Returns the list of written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{run_stem}_results.csv"
    rows = []
    for r in results:
        n = max(r.n_reps, 1)
        _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
        avg_ms, se_ms = _time_stats([r])
        d = asdict(r)
        d.update({
            "coverage": r.covered / n, "mean_width": r.total_width / n,
            "mean_score": r.total_score / n,
            "mean_penalty": (r.total_pen_under + r.total_pen_over) / n,
            "reject_rate": r.rejects / n, "mcse": mcse,
            "band95_low": lo, "band95_high": hi,
            "avg_time_ms": avg_ms, "se_time_ms": se_ms,
        })
        rows.append(d)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary_path = out / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, alpha=alpha, sample_sizes=sorted({r.n_a for r in results}))
    text = buf.getvalue()
    if latex:
        text += "\n% --- LaTeX tables (--latex) ---\n"
        text += latex_summary(results, alpha=alpha)
    summary_path.write_text(text, encoding="utf-8")
    print(f"\nSaved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def _plot_frame(results: list[SimResult]) -> "pd.DataFrame":
    """Per-cell frame for the plot helpers, one row per (scenario, method, n)."""
    import pandas as pd
    return pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "method": r.method,
            "n": (r.base_n or r.n_a), "coverage": r.covered / max(r.n_reps, 1),
            # Normalised by the source's own scale for the same reason the
            # tables are -- see SimResult.scale_span. A width-vs-n plot that
            # mixed 1-5 and 1-9 outcomes would show the scale, not the method.
            "width": r.total_width / max(r.n_reps, 1) / (r.scale_span or 1.0),
            "score": r.total_score / max(r.n_reps, 1) / (r.scale_span or 1.0),
        }
        for r in results if not r.is_null
    ])


def _plot_setup(results: list[SimResult]):
    """(eval types present, ordered Method objects, name list, colour palette).

    Shared so every plot in this case orders and colours methods the same way
    ci_paired does -- these figures sit next to each other in the appendix.
    """
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    method_objs = order_present_methods({r.method for r in non_null})
    names = [m.name for m in method_objs]
    return eval_types_present, method_objs, names, {m.name: m.color for m in method_objs}


def _finish(fig, out_path: str, suptitle: str) -> str:
    fig.suptitle(suptitle, fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


def _vs_n_plot(
    results: list[SimResult], alpha: float, out_path: str, *,
    metric: str, ylabel: str, title: str, n_reps: int, target_line: bool,
) -> str:
    """Shared engine for coverage-vs-n and width-vs-n.

    Aggregates to SCENARIO level first (mean within each data-generating
    scenario) and only then across scenarios, so a shape with many icc/effect
    variants does not outvote one with few -- same two-step aggregation
    ci_paired uses. Error bars are the between-scenario standard error, i.e.
    disagreement across scenarios, not Monte Carlo noise within one.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ..methods import get_method_color

    eval_types_present, _objs, names, palette = _plot_setup(results)
    df = _plot_frame(results)
    df = df[df["method"].isin(names)]
    label_level = df.groupby(["eval_type", "label", "method", "n"], as_index=False).agg(v=(metric, "mean"))
    agg = label_level.groupby(["eval_type", "method", "n"], as_index=False).agg(
        v_mean=("v", "mean"), v_std=("v", "std"), v_count=("v", "count"),
    )
    target = 1.0 - alpha

    ncols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [n for n in names if n in et_agg["method"].values]
        if et_agg.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue
        sns.lineplot(data=et_agg, x="n", y="v_mean", hue="method", hue_order=et_methods,
                     palette=palette, marker=None, linewidth=1.0, alpha=0.70, ax=ax)
        for method, sub in et_agg.groupby("method"):
            sub = sub.sort_values("n")
            color = get_method_color(str(method))
            if not sub["v_std"].isna().all():
                se = sub["v_std"] / np.sqrt(sub["v_count"])
                ax.errorbar(sub["n"], sub["v_mean"], yerr=se, fmt="none", color=color,
                            elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["v_mean"], s=28, color=color, edgecolors="white",
                       linewidths=0.6, alpha=0.85, zorder=3)
        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        if target_line:
            ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Group size (n per group)")
        ax.set_ylabel(ylabel if col_idx == 0 else "")
        ax.set_title(et.upper())
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # One shared legend beneath the panels rather than one inside each. With
    # 16 methods over 3 panels an in-axes legend covered the region where the
    # binary curves separate, which is the part a reader needs to see.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Method", loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=min(8, len(labels)),
                   fontsize=7.5, title_fontsize=8, frameon=False)
    return _finish(fig, out_path, f"{title}\nci_unpaired | reps={n_reps} | alpha={alpha}")


def save_coverage_vs_n_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Coverage vs. group size -- one subplot per eval type, all methods overlaid."""
    n_reps = results[0].n_reps if results else 0
    return _vs_n_plot(results, alpha, out_path, metric="coverage",
                      ylabel="Empirical coverage", title="Coverage vs. Group Size",
                      n_reps=n_reps, target_line=True)


def save_width_vs_n_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Mean CI width vs. group size -- the sharpness half of the tradeoff whose
    validity half the coverage plot shows. A method that covers only by being
    wide is visible here and nowhere else."""
    n_reps = results[0].n_reps if results else 0
    return _vs_n_plot(results, alpha, out_path, metric="width",
                      ylabel="Mean CI width (fraction of scale)", title="CI Width vs. Group Size",
                      n_reps=n_reps, target_line=False)


def save_cost_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Coverage against compute cost, one row per eval type, one line per method
    tracing it across group sizes. These methods span four orders of magnitude
    in runtime -- agresti_min's exact enumeration against a closed-form Wald --
    so "is the extra coverage worth the wait" is a question a reader will ask."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present, method_objs, _names, _pal = _plot_setup(results)

    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11.0, 4.5 * nrows),
                             squeeze=False, gridspec_kw={"hspace": 0.45})
    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        et_results = [r for r in non_null if r.eval_type == et]
        sizes = sorted({(r.base_n or r.n_a) for r in et_results})
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                   color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)
        for m in method_objs:
            m_results = [r for r in et_results if r.method == m.name]
            if not m_results:
                continue
            pts = []
            for n in sizes:
                subset = [r for r in m_results if (r.base_n or r.n_a) == n]
                if not subset:
                    continue
                avg_ms, _se = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms <= 0:
                    continue
                pts.append((avg_ms, float(np.mean([r.covered / max(r.n_reps, 1) for r in subset]))))
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, color=m.color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.scatter(xs, ys, s=30, color=m.color, edgecolors="white",
                       linewidths=0.6, alpha=0.9, zorder=3, label=m.name)
        ax.set_xscale("log")
        ax.set_xlabel("Mean time per interval (ms, log scale)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(et.upper())
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Method", loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=min(8, len(labels)),
                   fontsize=7, title_fontsize=8, frameon=False)
    n_reps = results[0].n_reps if results else 0
    return _finish(fig, out_path,
                   f"Coverage vs. Compute Cost (one point per group size)\n"
                   f"ci_unpaired | reps={n_reps} | alpha={alpha}")


def save_reliability_violin_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario coverage and
    interval score, one dot per (label, method) -- i.e. per data-generating
    scenario, averaged over group sizes and reps but NOT over scenarios.
    Exposes the spread the summary table's mean hides: a method with good
    average coverage can still have a long undercoverage tail on specific
    scenarios, which a single mean cannot reveal."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present, _objs, names, palette = _plot_setup(results)
    df = _plot_frame(results)
    df = df[df["method"].isin(names)]
    scenario_level = df.groupby(["eval_type", "label", "method"], as_index=False).agg(
        coverage=("coverage", "mean"), score=("score", "mean"),
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        et_df = scenario_level[scenario_level["eval_type"] == et]
        et_methods = [n for n in names if n in et_df["method"].values]
        for row_idx, (metric, ylabel) in enumerate(
            [("coverage", "Coverage per scenario"), ("score", "Interval score per scenario (scale-normalised)")]
        ):
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
            for lab in ax.get_xticklabels():
                lab.set_ha("right")
    n_reps = results[0].n_reps if results else 0
    return _finish(fig, out_path,
                   f"Cross-Scenario Reliability (one dot = one scenario)\n"
                   f"ci_unpaired | reps={n_reps} | alpha={alpha}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this case's CLI flags (data source, scenario/method
    selection, sweep sizes and ratios, Monte Carlo and exact-coverage
    controls, and output options) on the harness's per-case subparser."""
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                        help="'synthetic' (default) or 'real' -- human labels from the "
                             "judge-bias corpora and the App Store review corpus, split "
                             "into disjoint groups (see scenarios/real_unpaired.py).")
    parser.add_argument("--real-datasets", nargs="+", choices=REAL_UNPAIRED_DATASETS,
                        default=None, metavar="NAME",
                        help="Real data: restrict to these corpora (default: all available).")
    parser.add_argument("--data-dir", default=REAL_DEFAULT_DATA_DIR,
                        help=f"Real data: directory holding the corpus CSVs (default: {REAL_DEFAULT_DATA_DIR!r}).")
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES,
                        default=list(DEFAULT_EVAL_TYPES), metavar="TYPE")
    parser.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                        help="Restrict computation to these method names.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50], metavar="N",
                        help="Group-A sizes to sweep.")
    parser.add_argument("--size-ratios", type=float, nargs="+", default=[1.0], metavar="R",
                        help="n_B / n_A ratios (default 1.0). Unequal group sizes are the "
                             "norm between subjects, so e.g. --size-ratios 1.0 2.0.")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N")
    parser.add_argument("--bayes-n", type=int, default=2000, metavar="N")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.3], metavar="D")
    parser.add_argument("--include-null", action="store_true", default=False)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--agresti-min-max-n", type=int, default=_AGRESTI_MIN_MAX_N, metavar="N",
                        help=f"Largest group size at which agresti_min runs in the Monte Carlo "
                             f"sweep (default {_AGRESTI_MIN_MAX_N}). It is a small-sample "
                             f"recommendation and costs ~190 ms per interval at n=100, so above "
                             f"this it is skipped; use --exact-coverage for its behaviour at "
                             f"larger n. Set very high to disable the limit.")
    parser.add_argument("--latex", action="store_true", default=False,
                        help="Append booktabs LaTeX tables to the saved summary .log "
                             "and write a .tex for --exact-coverage.")
    parser.add_argument("--exact-coverage", action="store_true", default=False,
                        help="Binary only: compute coverage EXACTLY by enumerating every "
                             "possible pair of success counts, instead of estimating it by "
                             "simulation. No reps, no seed, no Monte Carlo error -- and it is "
                             "how the two-independent-proportions literature reports coverage, "
                             "so the numbers are directly comparable to Fagerland et al. "
                             "Ignores --reps/--sizes-as-pairs semantics: use --exact-sizes.")
    parser.add_argument("--exact-sizes", type=int, nargs="+", default=[10, 20, 30], metavar="N",
                        help="--exact-coverage: group-A sizes (combined with --size-ratios).")
    parser.add_argument("--exact-p-grid", type=int, default=19, metavar="K",
                        help="--exact-coverage: number of p values per axis, spread over "
                             "(0, 1) exclusive (default 19 -> 0.05..0.95 in steps of 0.05).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical preset -- mirrors ci_paired.official_args' breadth."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded",
        eval_types=["binary", "continuous", "likert"], methods=None,
        real_datasets=None, data_dir=REAL_DEFAULT_DATA_DIR,
        # sizes and bootstrap_n deliberately mirror ci_paired.official_args, so
        # the two cases' tables are directly comparable; size_ratios is the one
        # addition, since unequal group sizes only arise between subjects.
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100], size_ratios=[1.0, 2.0, 4.0],
        reps=300, bootstrap_n=10000, bayes_n=10000, alpha=0.05, seed=base_seed,
        icc_values=[0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95], cohens_d_values=[0.2, 0.4],
        include_null=True,
        exact_coverage=False, exact_sizes=[10, 20, 30], exact_p_grid=19,
        agresti_min_max_n=_AGRESTI_MIN_MAX_N, latex=True,
        progress="bar", plots="save", save_results="save",
        out_dir="simulations/out", plots_dir=None,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """Entries offered by ``--official-tests``' interactive menu.

    The synthetic preset deliberately mirrors ci_paired.official_args --
    same scenario suite, eval types, sizes, reps, bootstrap_n, bayes_n,
    alpha, icc and effect sweeps -- so the two cases' tables are directly
    comparable. size_ratios is the single addition, since unequal group
    sizes only arise between subjects, and ci_paired's statistic/runs
    options have no analogue here (this case is mean-only and flat-only).
    """
    real = argparse.Namespace(**{**vars(official_args(base_seed + 1)),
                                 "data_source": "real",
                                 # Real pools are fixed corpora, so the synthetic
                                 # shape/icc/effect sweep does not apply to them.
                                 "icc_values": None, "cohens_d_values": [0.3]})
    return [
        ("Between-subjects pairwise CIs, synthetic (mean difference)",
         official_args(base_seed)),
        ("Between-subjects pairwise CIs, real data incl. human subjects (mean difference)",
         real),
        ("Between-subjects binary CIs, EXACT coverage (no Monte Carlo)",
         argparse.Namespace(**{**vars(official_args(base_seed)), "exact_coverage": True})),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Small smoke-test preset -- runs in well under a minute.

    ``data_source`` is accepted and ignored: this preset always runs the
    synthetic scenario suite, so ``--quick-test``'s second (real-data) call
    is a no-op here rather than exercising ``build_real_unpaired_sources``.
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="standard",
        eval_types=["binary", "continuous", "likert"], methods=None,
        real_datasets=None, data_dir=REAL_DEFAULT_DATA_DIR,
        sizes=[10, 30], size_ratios=[1.0], reps=40, bootstrap_n=200, bayes_n=800,
        alpha=0.05, seed=base_seed, icc_values=[0.5], cohens_d_values=[0.3],
        include_null=True,
        exact_coverage=False, exact_sizes=[10, 20, 30], exact_p_grid=19,
        agresti_min_max_n=_AGRESTI_MIN_MAX_N, latex=True,
        progress="bar", plots="off", save_results="off",
        out_dir="simulations/out", plots_dir=None,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def run(args: argparse.Namespace) -> CaseResult:
    """Case entry point (the harness's per-case ``run`` contract).

    Two independent modes: ``--exact-coverage`` runs the binary-only exact
    enumeration path (``run_exact_coverage``) and returns early; otherwise
    builds sources (synthetic or real per ``args.data_source``), runs the
    Monte Carlo sweep (``run_simulation``), prints the console report, and
    optionally saves result artifacts and plots. Returns a ``CaseResult``
    with status, output paths and headline metrics; exceptions are caught
    and reported as an error status rather than propagated.
    """
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

        if getattr(args, "exact_coverage", False):
            k = args.exact_p_grid
            p_grid = np.linspace(0.0, 1.0, k + 2)[1:-1]
            size_pairs = [(n, max(2, int(round(n * r))))
                          for n in args.exact_sizes for r in args.size_ratios]
            method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
            print(f"\nci_unpaired EXACT binary coverage -- sizes={size_pairs}, "
                  f"{k}x{k} p-grid, alpha={args.alpha}")
            ex = run_exact_coverage(size_pairs, p_grid, args.alpha,
                                    method_names=method_names, progress_mode=args.progress)
            print_exact_report(ex, alpha=args.alpha, n_grid_points=k)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            paths = []
            if args.save_results == "save":
                paths.append(save_exact_csv(ex, args.out_dir, f"ci_unpaired_exact_{stamp}"))
                if getattr(args, "latex", False):
                    tex = Path(args.out_dir) / f"ci_unpaired_exact_{stamp}.tex"
                    tex.write_text(latex_exact_summary(ex, alpha=args.alpha), encoding="utf-8")
                    paths.append(str(tex))
                    print(f"Saved LaTeX table: {tex}")
            return CaseResult(
                case_name=CASE_NAME, status="ok", output_paths=paths,
                key_metrics={"n_results": len(ex),
                             "worst_min_coverage": min(r.min_coverage for r in ex)},
                duration_s=time.time() - t0,
            )

        print(f"\nci_unpaired simulation -- data_source={args.data_source}, "
              f"sizes={args.sizes}, size_ratios={args.size_ratios}")

        if args.data_source == "real":
            sources = build_real_unpaired_sources(
                data_dir=getattr(args, "data_dir", REAL_DEFAULT_DATA_DIR),
                datasets=getattr(args, "real_datasets", None),
                include_null=args.include_null,
            )
        else:
            icc_values = args.icc_values if args.icc_values is not None else [0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
            sources = build_pair_sources(
                suite=args.scenario_suite, icc_values=icc_values,
                cohens_d_values=args.cohens_d_values, include_null=args.include_null,
            )
        if args.eval_types:
            requested = set(args.eval_types)
            sources = [s for s in sources if s.eval_type in requested]
        if not sources:
            raise ValueError("No CIPairSources left after filtering.")

        method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
        n_cells = len(sources) * len(args.sizes) * len(args.size_ratios)
        print(f"  {len(sources)} sources, {n_cells} cells, reps={args.reps}, alpha={args.alpha}")

        results = run_simulation(
            sources, sample_sizes=args.sizes, size_ratios=list(args.size_ratios),
            n_reps=args.reps, n_bootstrap=args.bootstrap_n, bayes_n=args.bayes_n,
            alpha=args.alpha, progress_mode=args.progress, seed=args.seed,
            n_workers=getattr(args, "workers", 1), method_names=method_names,
            agresti_min_max_n=getattr(args, "agresti_min_max_n", _AGRESTI_MIN_MAX_N),
        )

        print_report(results, alpha=args.alpha, sample_sizes=args.sizes)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"ci_unpaired_{args.data_source}_reps{args.reps}_{stamp}"
        output_paths: list[str] = []
        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                latex=getattr(args, "latex", False),
            )
        if args.plots == "save":
            for suffix, fn in (
                ("coverage_vs_n", save_coverage_vs_n_plot),
                ("width_vs_n", save_width_vs_n_plot),
                ("cost_coverage", save_cost_plot),
                ("reliability_violin", save_reliability_violin_plot),
            ):
                output_paths.append(fn(
                    results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_{suffix}.png"),
                ))
            print(f"Saved plots: {output_paths[-4:]}")

        non_null = [r for r in results if not r.is_null]
        overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
            duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001 -- harness contract: report, don't crash the batch
        return CaseResult(
            case_name=CASE_NAME, status="error", duration_s=time.time() - t0, error=repr(exc),
        )
