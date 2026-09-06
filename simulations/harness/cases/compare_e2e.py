"""Black-box, end-to-end validation of ``evalstats.compare()`` itself.

Every other case in this harness (``ci_single``, ``ci_paired``, ``pvalues``,
``ppi_real``) validates evalstats' statistical methods by calling
``evalstats.tests``/``evalstats.core.paired`` functions directly on raw
numpy arrays. None of them ever call ``compare()`` -- the actual, real-user
entrypoint that resolves ``method="auto"``, ``simultaneous_ci=True``,
``correction="auto"`` (and, when ``alignment=`` is passed, the PPI-corrected
"sidak"/"boot" CI construction and "shaffer"/"romano_wolf" p-value
correction) into one specific concrete pipeline. This
case closes that gap: it builds a long-format DataFrame from a scenario,
calls ``compare()`` exactly as a real user would (only `factors=`, `metric=`,
`alignment=`, and `rng=` are ever set explicitly -- everything else is left
at compare()'s own defaults, on purpose), and checks the SAME properties
the rest of the harness checks for lower-level methods:

  - marginal (single-arm) CI coverage       -- ci_single's question
  - pairwise CI coverage                    -- ci_paired's question
  - Type-I error (family-wise, k>=3 arms)   -- pvalues --mode multiarm's question
  - power                                   -- pvalues --mode ppi's question

...across binary/continuous/likert data, a representative shape catalog,
a range of arm counts (k) and per-arm item counts (N), and -- the piece none
of the above ever combine -- WITH PPI alignment correction layered on top of
FWER control, at several human-label fractions, alongside a WITHOUT-PPI
baseline for direct comparison.

The two ppi_config levels answer DIFFERENT questions and are not a
before/after of the same scenario:

  ppi_config="none"    the TRUSTED-SCORES baseline. No judge bias, no PPI:
                       the ordinary use where someone analyses judge scores
                       they have no reason to distrust. Its estimand is
                       E[llm_score], and it is what shows whether evalstats
                       ITSELF is calibrated. Deliberately unbiased -- see
                       _run_cell's judge_biases for why biasing this row
                       makes its Type-I column measure judge bias rather
                       than calibration, and read as a library failure.
  ppi_config="frac=X"  the CORRECTION case. Differential judge bias IS
                       applied (see DEFAULT_JUDGE_BIAS_TYPE) and PPI has to
                       remove it. Its estimand is E[human label].

So "none" is not a weaker version of the PPI rows; power is not comparable
across the two (different judge, different estimand). The like-for-like
power comparisons live WITHIN each PPI row, against its own oracle and
subset-only reference arms (see CompareE2EResult's oracle_*/subset_*).

Coarse/breadth-first by design: the grid is
large (eval_type x shape x k x N x ppi_config x null/effect), run at a
modest ``reps`` per cell. The question this case answers is "does
compare()'s recommended path hold up broadly," not "get a tight estimate of
one specific scenario's rate" (that precision-focused job belongs to
pvalues.py's narrower, deeper sweeps).

Judge/human-label model: deliberately simple for this first pass (binary:
per-item bit-flip; continuous/likert: Gaussian noise + clip/round), NOT the
harness's much richer bias/noise-family model
(``scenarios/synthetic.py``'s ``generate_judge_bias_cell``, which builds
several different per-test-method array shapes at once and isn't set up to
hand back a long DataFrame). Reusing that richer model is a natural
follow-up once this scaffolding exists. Human labels reveal the TRUE
per-item value directly (the standard PPI framing -- the labeled subset is
gold-standard, unbiased for whatever ground truth compare()'s CIs are
checked against) on an MCAR-random, cross-arm-shared subset of items; the
LLM/judge score is a noisy (and, for binary data away from p=0.5,
systematically biased toward 0.5 by the symmetric bit-flip) view of that
same truth -- exactly the kind of imperfect-judge scenario PPI exists to
correct.

Usage:
    python -m simulations.harness.cli compare_e2e --reps 50
    python -m simulations.harness.cli compare_e2e --eval-types binary --k-values 3 5 --sizes 30 100
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import multiprocessing as _mp
import os
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.alignment import judge_alignment
from evalstats.core.stats_utils import interval_score

from ..latex_tables import booktabs_table, escape_latex
from ..scenarios import EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import (
    BINARY_SHAPES, CONTINUOUS_SHAPES, LIKERT_SHAPES, ShapeSpec, _jb_bias_magnitude,
    _jb_effect_magnitude, _jb_effect_magnitude_binary, _ppi_shape, _tier_shapes,
    sample_group_truth,
)
from . import CaseResult

CASE_NAME = "compare_e2e"

PROGRESS_MODES = ["bar", "cell", "off"]
SCENARIO_SUITES = ["standard", "expanded"]
PLOT_MODES = ["save", "off"]

# Fixed hue order/colors for ppi_config across every plot in this module, so
# "none" (raw, no alignment) is always the same color whether or not a given
# axis happens to include all four configs.
PPI_CONFIG_ORDER = ["none", "frac=0.10", "frac=0.20", "frac=0.40"]
PPI_CONFIG_COLORS = {
    "none": "0.4", "frac=0.10": "tab:blue", "frac=0.20": "tab:orange", "frac=0.40": "tab:red",
}

SHAPES_BY_EVAL_TYPE: dict[str, list[ShapeSpec]] = {
    "binary": BINARY_SHAPES,
    "continuous": CONTINUOUS_SHAPES,
    "likert": LIKERT_SHAPES,
}

# Per-arm-step effect size, expressed as a Cohen's-d-style fraction of each
# eval_type's OWN population standard deviation (arm i's true latent value
# gets shifted by i * this step, via sample_group_truth's own effects=
# parameter) -- the SAME standardized-severity convention synthetic.py's
# _jb_effect_magnitude/_jb_effect_magnitude_binary use for judge-bias power
# sweeps (see EVAL_TYPE_POPULATION_SD's docstring). A fixed raw-unit step is
# NOT comparable across eval types, since each has a different population SD.
# Binary must go through the *_binary wrapper, not _jb_effect_magnitude
# directly -- see that wrapper's docstring for why (its Beta(icc=1.0) truth
# model clips near {0,1}, attenuating an uncompensated shift).
EFFECT_FRAC_BY_EVAL_TYPE: dict[str, float] = {
    "continuous": 0.95, "likert": 0.22, "binary": 0.10,
}
"""Per-eval-type non-null effect size, calibrated so the ORACLE arm (every
item human-labelled -- the power ceiling) lands near 0.80 at the mid cell
(k=3, N=100):

    continuous 0.95 -> oracle ~0.83
    likert     0.22 -> oracle ~0.83
    binary     0.10 -> oracle ~0.75

Targeting the ORACLE (not PPI) is what makes the power rows readable: the
ceiling sits mid-range, so the power-vs-N curve rises across the grid
instead of pinning at 0 or 1, and PPI/subset-only have room to separate
BELOW it. A ceiling at 1.000 hides every difference the plot exists to show.

A single shared fraction cannot do this: standardizing to each eval type's
own population SD equalizes the effect in SD units but NOT its
detectability once the per-item noise this DGP has is in play, so each eval
type needs its own calibrated value."""


def _effect_frac_for(eval_type: str) -> float:
    """Non-null effect size for *eval_type* -- see EFFECT_FRAC_BY_EVAL_TYPE."""
    return EFFECT_FRAC_BY_EVAL_TYPE.get(eval_type, 0.50)


def _effect_step_for(eval_type: str, frac: float) -> float:
    """The NOMINAL per-arm step, before the k/icc corrections _cell_effect_step
    applies. Kept as its own function because the standalone
    investigate_joint_bootstrap_* scripts import it directly."""
    if eval_type == "binary":
        return _jb_effect_magnitude_binary(frac)
    return _jb_effect_magnitude(eval_type, frac)


MAX_EFFECT_SPAN_STEPS = 2
"""Cap on the arm-0..arm-(k-1) ramp, in units of the nominal per-arm step, so
the total spread is k-INVARIANT above k=3 instead of growing without bound.

`effects = arange(k) * step` with an uncapped step can push the top arms of
a large-k grid past the scale boundary (e.g. continuous's [0, 1] scale),
where they saturate (P(score=1.0) -> ~1) and a pair drawn from among them
becomes mostly exact zeros with a rare one-sided tail. No mean-based
interval is calibrated in that regime: the ramp drives one boundary-hugging
shape into saturation and the opposite-skewed shape away from it, so
coverage degrades asymmetrically between them -- a signature of DGP
saturation rather than a method defect, since no CI formula can care which
direction an effect points.

The cap is 2 steps = the k=3 span, so k=2 and k=3 keep their exact previous
effects (min() below is 1.0 there) and only k>3 is compressed, keeping the
ramp well inside the scale at every k.

Cost: reported POWER at k>3 changes (the extreme pair M0-vs-M(k-1) is now a
fixed standardized distance regardless of k, rather than growing with k),
which is the more meaningful power axis anyway -- it no longer pins at 1.000
for the sole reason that k is large."""

_ICC_SD_MC_N = 1_000_000
_ICC_SD_MC_SEED = 20260827
_ICC_SD_RATIO_SNAP = 0.005
"""Ratios this close to 1.0 are snapped to exactly 1.0. likert and binary are
STRUCTURALLY icc-invariant (their icc split decomposes a fixed total variance,
unlike continuous's, which adds to it), so their true ratio is 1.0 and
anything measured is Monte-Carlo noise -- at _ICC_SD_MC_N the relative SE of
an SD estimate is ~0.07%, so this 0.5% band is ~7 sigma of noise while
continuous's real deviations (+55% / -33%) are an order of magnitude outside
it. Without the snap those two eval types would pick up a spurious ~0.1% step
change at off-reference icc and stop being bit-identical to previous runs for
no scientific reason."""


@functools.lru_cache(maxsize=None)
def _representative_sd_at_icc(eval_type: str, icc: float) -> float:
    """EVAL_TYPE_POPULATION_SD's own methodology -- the representative shape's
    realized truth SD -- but measured AT `icc` instead of at icc=1.0."""
    rng = np.random.default_rng(_ICC_SD_MC_SEED)
    truth = sample_group_truth(_ppi_shape(eval_type), _ICC_SD_MC_N, 1, 1, icc, rng)[0, :, 0]
    return float(truth.std(ddof=0))


def _icc_sd_ratio(eval_type: str, icc: float) -> float:
    """Rescales the effect step so Cohen's d is CONSTANT across the icc sweep.

    `_jb_effect_magnitude` divides `frac` by a fixed EVAL_TYPE_POPULATION_SD
    measured once at icc=1.0. But continuous's `_group_noise_var` *adds*
    var_base*(1/icc - 1) on top of var_base, so its realized SD moves with
    icc, while likert and binary DECOMPOSE a fixed total and stay flat. With
    a fixed denominator and a moving realized SD, sweeping icc would silently
    sweep the standardized effect size too -- for continuous and ONLY
    continuous.

    Anchored at DEFAULT_ICC rather than at icc=1.0 on purpose: the ratio is
    exactly 1.0 at the reference, so every icc=DEFAULT_ICC cell -- the
    realistic point, and the value every PPI cell is pinned to -- keeps its
    previous effect step bit-for-bit, and EFFECT_FRAC_BY_EVAL_TYPE's
    oracle-power calibration survives unchanged. Only the off-reference icc
    values move. For binary/likert the ratio is ~1.0 by construction, so this
    is a no-op there."""
    if icc == DEFAULT_ICC:
        return 1.0
    ratio = _representative_sd_at_icc(eval_type, icc) / _representative_sd_at_icc(eval_type, DEFAULT_ICC)
    return 1.0 if abs(ratio - 1.0) < _ICC_SD_RATIO_SNAP else ratio


def _cell_effect_step(eval_type: str, frac: float, k: int, icc: float) -> float:
    """The per-arm effect step for one cell: the nominal step, held at constant
    Cohen's d across the icc sweep (_icc_sd_ratio) and capped so the total ramp
    does not outgrow the scale (MAX_EFFECT_SPAN_STEPS)."""
    step = _effect_step_for(eval_type, frac) * _icc_sd_ratio(eval_type, icc)
    return step * min(1.0, MAX_EFFECT_SPAN_STEPS / max(k - 1, 1))

DEFAULT_PPI_FRACS: tuple[Optional[float], ...] = (None, 0.10, 0.20, 0.40)
# k=2 is included deliberately as an UNCORRECTED baseline: with only one pair,
# Sidak's own alpha_adj = 1-(1-alpha)**(1/1) = alpha reduces to no adjustment
# at all, so k=2 cells report each pairwise CI's OWN calibration with no FWER
# widening confound. k>2 cells are where simultaneous_ci's FWER machinery
# actually engages -- see _aggregate_group, which uses k==2 rows for the
# "pairwise, uncorrected" metric and k>2 rows for "family-wise, corrected".
DEFAULT_K_VALUES = [2, 3, 5, 10]
DEFAULT_SIZES = [15, 30, 60, 100, 200, 400, 800, 1000]
"""Items per arm. The top end exists for the N/N_lab axis specifically: with
an absolute label budget (see _n_labeled_for) the ratio is N/n_lab, so
reaching 1000 puts a 30-label budget at a ratio of ~33. That is the regime
PPI is actually for -- gain comes from the UNLABELLED items -- and the trend
is what a fraction-based grid cannot show at all, since a fraction pins the
ratio to 1/frac for every N."""
DEFAULT_ICC = 0.20
"""Item-level reliability of the TRUTH generator, matching
scenarios.synthetic._ppi_power_baseline's own icc (the tier every PPI sweep
in cases/pvalues.py runs at). Below 1.0 so different arms produce genuinely
different per-item truth (a real per-item paired-difference signal to
estimate) rather than sharing one latent value shifted by a constant --
the latter degenerates any variance-based estimator (see
evalstats.api._JOINT_BOOT_SE_REL_FLOOR) and gives noiseless reference arms
spurious perfect power, independent of any real statistical effect."""

DEFAULT_ICC_VALUES: tuple[float, ...] = (0.05, 0.20, 0.60)
"""icc values swept on the no-PPI arm. ONE definition, read by both
official_args and the --icc-values argparse default, so a CLI run and the
official preset cannot disagree about what the sweep is.

0.20 is the realistic point: cross-model correlation measured on the real
corpora is 0.146 mean / 0.103 median, and icc=0.20 reproduces r=0.170. 0.05
is a stress point below that and 0.60 a robustness point above (higher than
any real corpus). Note this is the CROSS-ARM correlation -- different models
on shared items -- not the multi-run ICC (~0.68), which is a different axis
and is not exercised at runs=1."""

AGREEMENT_RATE_BY_EVAL_TYPE: dict[str, float] = {
    "binary": 0.92, "continuous": 0.40, "likert": 0.60,
}
"""Per-eval-type judge quality, calibrated so each lands near rho^2 ~ 0.64 --
the judge-alignment tier cases/pvalues.py's PPI sweeps use -- measured at
DEFAULT_ICC against each eval type's first shape:

    binary     0.92 -> rho^2 0.643      (flip probability 0.08)
    continuous 0.40 -> rho^2 0.653
    likert     0.60 -> rho^2 0.646

A single shared rate cannot do this: the same nominal "agreement" maps to a
very different rho per eval type, because _apply_judge_noise's noise is a
fraction of scale span for numeric data but a flip probability for binary."""

DEFAULT_JUDGE_BIAS_TYPE = "differential"
"""Judge bias applied across arms, mirroring _ppi_power_baseline's own
bias_type: "differential" biases ONLY arm 0, so a relative/paired comparison
sees a real judge-induced difference that PPI's rectifier has to remove.
"constant" biases every arm equally and "none" disables it."""

# Oracle/subset-only reference estimators (see CompareE2EResult) feed TRUTH
# values directly to compare(), with NO noise -- correct in principle (an
# "oracle" IS the ground truth), but a problem for CONTINUOUS data
# specifically: sample_group_truth's icc=1.0 applies `effects=` as a
# deterministic, per-item-IDENTICAL shift (see that function's own
# docstring: "icc=1.0 means no noise at all"), so for un-rounded continuous
# values the per-item PAIRED DIFFERENCE between any two arms is exactly
# constant across every item -- any variance-based CI (logit_t, smooth_bootstrap,
# and evalstats' own default construction) built from that collapses to a
# near-zero-width interval, giving spuriously ~100% power rather than a
# real small-vs-large-N story. NOT an issue for binary (each item's {0,1}
# realization is its own independent Bernoulli draw, not a deterministic
# shift of a shared latent value) or likert (rounding to the integer scale
# breaks the exact constancy for items near a rounding boundary) -- both
# already have genuine non-degenerate per-item diff variance under icc=1.0.
# Fix: apply a SMALL amount of realistic labeler noise (well below the LLM
# judge's own AGREEMENT_RATE_BY_EVAL_TYPE-level noise -- an "oracle" should
# still be near-perfect) when building oracle/subset's continuous scores
# specifically, just enough to break the exact-zero-variance degeneracy.
# Zero-mean (Gaussian) for continuous, so truth_means stays the correct,
# unbiased coverage-check reference.
ORACLE_NOISE_AGREEMENT_RATE = 0.99

# compare()'s own default is n_bootstrap=10_000 (evalstats/core/router.py's
# analyze()) -- this dominates per-call cost far more than n_mc (which floors
# at max(n_mc, 1000) for the PPI/Pareto path regardless of what's passed, so
# lowering IT does nothing; see this case's runtime-tuning notes). Lowering
# n_bootstrap is real savings, but NOT free below a certain point: pvalues.py's
# official_args_ppi (this repo's own prior investigation) found romano_wolf/
# westfall_young/"boot"'s FWER ran ~0.001-0.002 ABOVE nominal at
# bootstrap_n=500-2000 (Monte Carlo noise in the joint max-statistic's
# upper-tail quantile, not a correction-logic bug), resolved only at 5000.
# Since compare_e2e's whole job is validating compare()'s Type-I/FWER
# calibration (via correction="auto", which resolves to Romano-Wolf-family
# methods at k>=3), using a bootstrap count already shown to bias exactly that
# metric would confound the harness's own findings -- so 5000 (not a lower,
# faster-but-documented-biased value) is the default here, a real ~2x cut
# from compare()'s own 10_000 default while staying inside the range this
# codebase has already validated as safe for FWER checks specifically.
DEFAULT_BOOTSTRAP_N = 5000

# Reference-estimator cells (oracle/subset-only, see CompareE2EResult) run a
# SECOND compare() call every rep, roughly doubling per-cell cost -- since
# the "sandwich" story (row 7 of the hero plot) pools across shapes/n_items
# already, it doesn't need every k value to be informative. Restricting it to
# one representative k keeps the full n_items sweep (needed for the power
# curve's shape) while cutting the added cost back down to roughly the size
# of the ORIGINAL (pre-sandwich) grid instead of ~2x it.
REFERENCE_ESTIMATOR_K = 3

# Minimum sample-size requirements _run_alignment_ppi enforces (evalstats/api.py) --
# duplicated here (not imported, since these are compare()'s own internal
# constants, not part of its public API) purely to pre-filter PPI cells that
# are GUARANTEED to raise before wasting compute generating data for them.
_PPI_MIN_N_LAB = 15
_PPI_MIN_N_ALL = 50
_PPI_MAX_LAB_SHARE = 0.60
"""Upper bound on n_lab / n_items for a PPI cell.

PPI's entire premise is that MOST items are unlabelled and the judge supplies
the rest; labelling 60%+ of them is not a setting anyone would deploy, and at
the limit n_lab == n_items it is degenerate rather than merely unusual -- with
zero unlabelled items the rectifier has nothing to correct and the joint
bootstrap has no unlabelled term at all (_ppi_bootstrap_t_joint_stats returns
None on that branch). Those cells passed the n_lab>=15 / n_all>=50 floor while
being statistically empty: measured Type-I fell to ~0.000 against a nominal
0.05 at the first N where each fixed label budget became legal (n_lab == N),
i.e. a test that essentially never rejects, which then read on the calibration
plot as a large "conservative" excursion rather than as an excluded
configuration.

The fraction-based grid never exposed this because a fraction pins the share
by construction; an absolute label budget (see _n_labeled_for) sweeps N past
it, so the bound has to be stated explicitly."""

# One big, one-off Monte-Carlo draw used to estimate each shape's true mean
# numerically (works uniformly for "param" AND "custom" shapes, without
# needing each shape's closed form) -- computed once per (shape, k, effect)
# combination, not per rep.
_TRUE_MEAN_MC_N = 200_000


@dataclass
class CompareE2EResult:
    """Aggregated outcome for one (eval_type, shape, k, n_items, ppi_config,
    is_null) cell, summed/counted over `n_reps` replicates -- mirrors
    SimResult's aggregate-per-cell convention (ci_single.py), not one row
    per replicate, since this case's grid is large."""

    eval_type: str
    shape_label: str
    k: int
    n_items: int
    ppi_config: str  # "none" | "frac=0.10" | "frac=0.20" | "frac=0.40"
    is_null: bool
    n_reps: int
    icc: float = DEFAULT_ICC
    """The cell's intraclass correlation -- sample_group_truth's signal/noise
    split. Part of the cell KEY whenever --icc-values sweeps it: without it
    two cells differing only in icc are indistinguishable in the saved CSV and
    silently pool together in every downstream aggregation."""

    effect_step: float = 0.0
    """The per-arm effect step this cell actually ran (0.0 for null cells).

    Recorded because it is no longer a per-eval-type constant: _cell_effect_step
    varies it with k (MAX_EFFECT_SPAN_STEPS caps the total ramp) and with icc
    (_icc_sd_ratio holds Cohen's d fixed). Derivable from the other key columns,
    but only by re-running that logic -- storing it keeps the CSV self-describing,
    so a plot rebuilt from the CSV alone can label what effect size each cell
    was actually run at instead of assuming one number for the whole sweep."""
    n_errors: int = 0
    """Reps where compare() itself raised (e.g. a genuinely-degenerate draw) --
    excluded from all rate denominators below, which use n_reps - n_errors."""
    marginal_covered: int = 0
    marginal_total: int = 0
    """Total marginal-CI checks across all k arms and all successful reps
    (k * (n_reps - n_errors))."""
    marginal_width_sum: float = 0.0
    """Sum of (ci_high - ci_low) across the same checks as marginal_total --
    divide by marginal_total for mean width, the sharpness complement to
    marginal_covered/marginal_total's coverage. bundle.robustness's per-arm CI
    is NOT multiplicity-adjusted by k (see CompareE2EResult module notes), so
    unlike pairwise_covered below this needs no k-based split."""
    marginal_score_sum: float = 0.0
    """Sum of evalstats.core.stats_utils.interval_score(ci_low, ci_high,
    true_value, alpha) across the same checks as marginal_total -- the SAME
    metric ci_single.py/ci_paired.py report as "Score" (width + (2/alpha) *
    miss-distance when uncovered, lower is better), so this is
    directly comparable across the harness, not a compare_e2e-only number."""
    pairwise_covered: int = 0
    pairwise_total: int = 0
    """Total pairwise-CI checks across all C(k,2) pairs and all successful
    reps. IMPORTANT: only meaningful when filtered to k==2 rows (see
    _aggregate_group) -- at k>2, simultaneous_ci=True builds each pair's own
    CI at a much higher per-pair level (~1-(1-alpha)**(1/n_pairs), e.g.
    ~99.9% for k=10's 45 pairs) so the JOINT event holds, so pooling k==2 and
    k>2 rows here would blend a genuine per-pair calibration check with an
    artifact of FWER widening. k==2 rows report each pair's OWN calibration
    with no correction confound (Sidak's alpha_adj reduces to plain alpha
    when there's only 1 pair)."""
    pairwise_width_sum: float = 0.0
    pairwise_score_sum: float = 0.0
    """Sum of pairwise CI width / interval_score across the same checks as
    pairwise_total. Same k==2-vs-k>2 split applies: k==2 rows give the
    uncorrected per-pair width/score baseline; k>2 rows give the FWER-widened
    per-pair width/score -- the direct "what does simultaneous protection
    cost in width/score" comparison, on the SAME scale ci_paired.py uses."""
    family_covered: int = 0
    family_total: int = 0
    """Family-wise (simultaneous) coverage: a rep counts as 'covered' only if
    EVERY one of its C(k,2) pairwise CIs covers its true diff, not just one
    pair on average. This is the actual guarantee simultaneous_ci=True makes
    (P(all pairs covered) >= 1-alpha). IMPORTANT: only meaningful when
    filtered to k>2 rows (see _aggregate_group) -- at k==2 there is no
    "family" beyond the single pair, so it's identical to pairwise_covered
    and would just dilute this metric with a redundant case where the FWER
    machinery never actually engages. family_total == n_reps - n_errors (one
    check per successful rep, not one per pair)."""
    any_reject: int = 0
    """Type-I-relevant only: count of reps where ANY pairwise p-value < alpha
    (meaningful only when is_null=True)."""
    extreme_reject: int = 0
    """Power-relevant only: count of reps where the (arm 0, arm k-1) pair
    (the largest true gap by construction) has p-value < alpha (meaningful
    only when is_null=False)."""

    # Reference-estimator comparisons -- computed from the SAME synthetic
    # draw as the main result above, applying a different "estimator" to it
    # for an apples-to-apples power/Type-I comparison. Both use TRUTH values
    # directly as the score (no judge noise, no alignment= needed, since
    # there's no judge bias to correct when every point IS the ground truth).
    # oracle_* is attached to ppi_config=="none" cells only (as if EVERY item
    # were human-labeled -- the power CEILING, computed once per
    # (eval_type,shape,k,n_items,is_null) rather than redundantly once per
    # PPI frac swept, since it doesn't depend on frac at all). subset_* is
    # attached to ppi_config!="none" cells only (as if ONLY the n_labeled
    # human-labeled items existed -- the LLM-scored majority discarded
    # entirely -- the power FLOOR PPI should beat if it's earning its keep
    # over just throwing away the unlabeled data). See _run_truth_only_compare.
    oracle_marginal_covered: int = 0
    oracle_marginal_total: int = 0
    oracle_pairwise_covered: int = 0
    oracle_pairwise_total: int = 0
    oracle_family_covered: int = 0
    oracle_family_total: int = 0
    oracle_any_reject: int = 0
    oracle_extreme_reject: int = 0
    oracle_n_ok: int = 0
    """Successful oracle computations this cell -- the denominator for
    oracle_any_reject/oracle_extreme_reject (mirrors n_reps-n_errors for the
    main result, but tracked separately since oracle's own compare() call can
    fail independently of the main one)."""
    subset_marginal_covered: int = 0
    subset_marginal_total: int = 0
    subset_pairwise_covered: int = 0
    subset_pairwise_total: int = 0
    subset_family_covered: int = 0
    subset_family_total: int = 0
    subset_any_reject: int = 0
    subset_extreme_reject: int = 0
    subset_n_ok: int = 0
    """Successful subset-only computations this cell -- see oracle_n_ok."""


def _agreement_for(eval_type: str) -> float:
    """Judge quality for *eval_type* -- see AGREEMENT_RATE_BY_EVAL_TYPE."""
    return AGREEMENT_RATE_BY_EVAL_TYPE.get(eval_type, 0.60)


def _judge_biases_for(eval_type: str, k: int, bias_type: str = DEFAULT_JUDGE_BIAS_TYPE) -> np.ndarray:
    """Per-arm judge bias, mirroring scenarios.synthetic._jb_biases.

    "differential" biases arm 0 only; "constant" biases every arm equally
    (which a paired comparison should cancel out); "none" disables it.
    Magnitude comes from the same standardized helpers the PPI sweeps use:
    _jb_bias_magnitude for numeric scales (0.30 population SDs of the eval
    type's own truth distribution), and PPI_BINARY_BIAS_MAGNITUDES'
    "moderate" flip-probability skew for binary, whose bias is a
    one-directional flip rather than an additive offset."""
    mag = 0.10 if eval_type == "binary" else _jb_bias_magnitude(eval_type)
    if bias_type == "none":
        return np.zeros(k)
    if bias_type == "constant":
        return np.full(k, mag)
    if bias_type == "differential":
        b = np.zeros(k)
        b[0] = mag
        return b
    raise ValueError(f"Unknown bias_type: {bias_type!r}")


def _apply_judge_noise(
    truth: np.ndarray, eval_type: str, rng: np.random.Generator, agreement_rate: float,
    biases: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Judge model: noise at *agreement_rate*, plus an optional per-arm
    bias (see _judge_biases_for). Still far simpler than
    scenarios.synthetic.generate_judge_bias_cell's full bias/noise family
    (no slope miscalibration, no MNAR labeling), but no longer the
    unbiased near-perfect judge this case started with -- see
    AGREEMENT_RATE_BY_EVAL_TYPE and DEFAULT_JUDGE_BIAS_TYPE.

    *biases* is one value per arm, matching ``truth``'s first axis. For
    binary it is a one-directional flip probability (the biased arm's 0s
    get pushed to 1), since an additive offset is meaningless on {0, 1};
    for numeric scales it is an additive offset applied before the
    scale's own clipping/rounding, so boundary effects still apply.
    """
    if biases is None:
        biases = np.zeros(truth.shape[0])
    biases = np.asarray(biases, dtype=float).reshape(-1, 1)

    if eval_type == "binary":
        flip = rng.random(truth.shape) >= agreement_rate
        out = np.where(flip, 1.0 - truth, truth)
        # Differential bias: push this arm's 0s upward, the binary analogue
        # of an additive offset (see _jb_llm_binary's flip-probability skew).
        up = rng.random(truth.shape) < biases
        return np.where(up, 1.0, out)

    lo, hi = EVAL_TYPE_SCALE_BOUNDS[eval_type]
    span = hi - lo
    noise_sd = 0.5 * (1.0 - agreement_rate) * span
    noisy = truth + biases + rng.normal(0.0, noise_sd, size=truth.shape)
    noisy = np.clip(noisy, lo, hi)
    if eval_type == "likert":
        noisy = np.rint(noisy)
    return noisy


def _reference_means_for(
    shape: ShapeSpec, eval_type: str, k: int, effects: np.ndarray, agreement_rate: float,
    rng: np.random.Generator, icc: float = DEFAULT_ICC, biases: Optional[np.ndarray] = None,
    compute_llm_means: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically estimate each arm's TWO possible ground-truth references
    via one large draw -- works uniformly for "param" and "custom" shapes
    without hand-deriving each shape's closed form, and automatically
    captures any boundary-clipping bias _apply_judge_noise introduces (not
    just binary's symmetric-flip bias).

    Returns (truth_means, llm_means):
      truth_means -- E[human label] -- the PPI-corrected estimate's target
                     (human labels reveal truth directly in this generator).
      llm_means   -- E[llm score]   -- the RAW (no-PPI) estimate's target.
                     These genuinely differ whenever the judge-noise model
                     is biased relative to the shape's own true parameter
                     (e.g. binary's symmetric bit-flip pulls E[llm_score]
                     toward 0.5 for any true_p != 0.5) -- checking a no-PPI
                     cell's coverage against truth_means instead of this
                     would be comparing compare()'s CI to the wrong target
                     entirely, not a real coverage failure.
    """
    # icc/biases MUST match the cells' own generator -- these means are the
    # targets coverage is scored against, so a mismatch silently scores every
    # CI against the wrong truth (icc alone moves the arm-1-vs-2 gap from
    # -0.0180 at icc=1.0 to -0.0122 at icc=0.20).
    draw = sample_group_truth(shape, _TRUE_MEAN_MC_N, 1, k, icc, rng, effects=effects)[:, :, 0]  # (k, N)
    truth_means = draw.mean(axis=1)
    if not compute_llm_means:
        # Both arms now estimate truth_means (the no-PPI arm no longer applies
        # judge noise, so E[llm score] == E[truth] there). Skipping the second
        # 200k-item draw + judge-noise pass saves it per cell.
        return truth_means, truth_means
    llm_draw = _apply_judge_noise(draw, eval_type, rng, agreement_rate, biases)
    llm_means = llm_draw.mean(axis=1)
    return truth_means, llm_means


def _n_labeled_for(n_items: int, frac: float) -> int:
    """Labelled-item count for a PPI cell.

    ``frac`` carries two meanings, disambiguated by magnitude:
      < 1   a FRACTION of n_items (the original behaviour, e.g. 0.20)
      >= 1  an ABSOLUTE label count (e.g. 30), held FIXED as n_items varies

    The absolute form exists because PPI's whole value proposition is the
    N/N_lab ratio -- gain comes from the UNLABELLED items -- and a
    fraction-based grid cannot show that axis at all: it pins the ratio to
    1/frac for every N, so sweeping N moves both arms together and the
    comparison against the human-labels-only floor stays flat. Measured at
    fixed n_lab=60, PPI's power advantage over that floor grows +0.144 ->
    +0.300 as N/N_lab goes 2 -> 10, none of which is visible at a fixed
    frac=0.40 (ratio 2.5).

    It also lets the grid sit at label counts a practitioner would actually
    collect: N_lab ~ 30 is the usual lower bound for a human-labelled
    subset, and a fraction grid only hits it by coincidence at one N.
    """
    return int(round(frac)) if frac >= 1 else max(1, round(n_items * frac))


def _ppi_applicable(k: int, n_items: int, frac: float) -> bool:
    """Mirrors evalstats.api._run_alignment_ppi's own minimum-sample-size
    checks (n_lab >= 15, n_all >= 50) -- pre-filters cells GUARANTEED to
    raise, rather than generating data for them every rep only to hit the
    same ValueError deterministically."""
    n_lab = _n_labeled_for(n_items, frac)
    n_all = k * n_items
    # An absolute label count can also exceed the items available, and a
    # too-LARGE labelled share is excluded as well -- see _PPI_MAX_LAB_SHARE.
    return (n_lab >= _PPI_MIN_N_LAB and n_all >= _PPI_MIN_N_ALL
            and n_lab <= n_items * _PPI_MAX_LAB_SHARE)


def _build_dataframe(
    llm_scores: np.ndarray,  # (k, n_items)
    human_scores: Optional[np.ndarray],  # (k, n_items) or None; NaN where unlabeled
) -> pd.DataFrame:
    k, n_items = llm_scores.shape
    rows = []
    for i in range(k):
        model = f"M{i}"
        for j in range(n_items):
            row = {"model": model, "item": j, "score": float(llm_scores[i, j])}
            if human_scores is not None:
                h = human_scores[i, j]
                row["human_score"] = float(h) if np.isfinite(h) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _score_bundle(bundle, true_means: np.ndarray, k: int, alpha: float, is_null: bool) -> dict:
    """Given a compare() bundle and the reference true_means array (indexed
    by 'M{i}' labels), compute one rep's marginal/pairwise/family-wise
    coverage counts plus the Type-I/power indicator. Shared by the main
    llm/PPI-corrected result and the oracle/subset-only reference-estimator
    checks (_run_truth_only_compare) -- same scoring logic, different bundle
    and (always truth-based) reference means."""
    labels = list(bundle.benchmark.template_labels)
    rob = bundle.robustness
    label_to_true = {f"M{i}": true_means[i] for i in range(k)}

    marginal_total = marginal_covered = 0
    marginal_width_sum = 0.0
    marginal_score_sum = 0.0
    for i, lbl in enumerate(labels):
        lbl_s = str(lbl)
        ci_lo, ci_hi = rob.ci_low[i], rob.ci_high[i]
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            marginal_total += 1
            marginal_width_sum += ci_hi - ci_lo
            marginal_score_sum += interval_score(ci_lo, ci_hi, label_to_true[lbl_s], alpha)
            if ci_lo <= label_to_true[lbl_s] <= ci_hi:
                marginal_covered += 1

    pairwise_total = pairwise_covered = 0
    pairwise_width_sum = 0.0
    pairwise_score_sum = 0.0
    any_sig = False
    extreme_p = None
    all_pairs_covered = True
    for (a, b), pr in bundle.pairwise.results.items():
        true_diff = label_to_true[str(a)] - label_to_true[str(b)]
        pairwise_total += 1
        pairwise_width_sum += pr.ci_high - pr.ci_low
        pairwise_score_sum += interval_score(pr.ci_low, pr.ci_high, true_diff, alpha)
        pair_covered = pr.ci_low <= true_diff <= pr.ci_high
        if pair_covered:
            pairwise_covered += 1
        else:
            all_pairs_covered = False
        if pr.p_value is not None and pr.p_value < alpha:
            any_sig = True
        if {str(a), str(b)} == {"M0", f"M{k-1}"}:
            extreme_p = pr.p_value

    any_reject = 1 if (is_null and any_sig) else 0
    extreme_reject = 1 if (not is_null and extreme_p is not None and extreme_p < alpha) else 0

    return dict(
        marginal_covered=marginal_covered, marginal_total=marginal_total,
        marginal_width_sum=marginal_width_sum, marginal_score_sum=marginal_score_sum,
        pairwise_covered=pairwise_covered, pairwise_total=pairwise_total,
        pairwise_width_sum=pairwise_width_sum, pairwise_score_sum=pairwise_score_sum,
        family_covered=(1 if all_pairs_covered else 0), family_total=1,
        any_reject=any_reject, extreme_reject=extreme_reject,
    )


#: compare() kwargs for the rank-based pathway the PAPER reports: Friedman
#: omnibus, then Wilcoxon signed-rank pairwise (already ``pairwise_test="auto"``'s
#: pick for any k), then Shaffer as the FWER post-hoc. compare()'s own default
#: resolves ``correction="auto"`` to Romano-Wolf, which is the better method and
#: stays the default here -- but it is NOT the pathway the paper validates and
#: reports, so a reviewer cannot check the reported path against the oracle and
#: human-subset arms without this. Opt in with --classical-rank-path.
#:
#: Applied to EVERY arm (PPI, oracle, human-subset) or the comparison would be
#: apples-to-oranges: the reference arms would still be Romano-Wolf.
CLASSICAL_RANK_KWARGS = {"omnibus": True, "correction": "shaffer"}


def _run_truth_only_compare(
    scores: np.ndarray, rng: np.random.Generator, score_range, n_bootstrap: int,
    es_eval_type: Optional[str] = None, classical_rank: bool = False,
):
    """Run compare() directly on TRUTH values as the score (no judge noise,
    no alignment= needed -- there's no judge bias to correct when every
    point already IS the ground truth). Used for the two reference-estimator
    comparisons: 'oracle' (every item human-labeled, scores=full truth array)
    and 'subset-only' (only the labeled items, scores=truth[:, labeled_items],
    the LLM-scored majority discarded entirely). Returns the bundle, or None
    if compare() itself failed.

    es_eval_type : compare()'s own eval_type=("likert"|"continuous"|None)
        kwarg -- see _run_cell's docstring note on why this is passed
        explicitly rather than left to auto-detection."""
    df = _build_dataframe(scores, None)
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    kwargs = {"n_bootstrap": n_bootstrap}
    if classical_rank:
        kwargs.update(CLASSICAL_RANK_KWARGS)
    if score_range is not None:
        kwargs["score_range"] = score_range
    if es_eval_type is not None:
        kwargs["eval_type"] = es_eval_type
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr = es.compare(evaldata, factors="model", metric="score", rng=rng, **kwargs)
    return cr._primary_bundle()


def _run_cell(
    eval_type: str,
    shape: ShapeSpec,
    k: int,
    n_items: int,
    ppi_frac: Optional[float],
    is_null: bool,
    n_reps: int,
    alpha: float,
    seed,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    reference_estimator_k: Optional[int] = REFERENCE_ESTIMATOR_K,
    classical_rank: bool = False,
    icc: float = DEFAULT_ICC,
) -> CompareE2EResult:
    """Run all reps for one cell, aggregating coverage/Type-I/power counts.

    reference_estimator_k: only cells at this k value compute the
    oracle/subset-only reference estimators (see CompareE2EResult) -- roughly
    doubles cost per cell it applies to, so restricting it to one
    representative k keeps that cost from applying to the WHOLE grid. Pass
    None to compute it for every k (more complete, much slower)."""
    rng = np.random.default_rng(seed)
    cell_icc = float(icc)
    ppi_config = ("none" if ppi_frac is None else
                  (f"nlab={int(round(ppi_frac))}" if ppi_frac >= 1 else f"frac={ppi_frac:.2f}"))
    compute_reference = reference_estimator_k is None or k == reference_estimator_k

    effect_step = (0.0 if is_null else
                   _cell_effect_step(eval_type, _effect_frac_for(eval_type), k, cell_icc))
    effects = np.arange(k, dtype=float) * effect_step
    agreement_rate = _agreement_for(eval_type)
    # Judge bias applies to the PPI cells ONLY. The no-PPI ("none") cells are
    # this case's TRUSTED-SCORES baseline: the ordinary use where a user
    # analyses judge scores they have no reason to distrust, which is what
    # shows whether evalstats itself is calibrated. Biasing them instead
    # measures something else entirely and mislabels it:
    #
    #   a no-PPI cell's estimand is E[llm_score] (see the true_means line
    #   below). Under differential bias, arm 0's E[llm_score] genuinely
    #   differs from the others' EVEN WHEN is_null=True, because the bias is
    #   a real shift in the thing being estimated. The uncorrected test then
    #   correctly rejects, and the "Type-I error" column reports that as
    #   miscalibration -- it is not. It is power to detect judge bias, under
    #   a null that is not null for that estimand. Measured before this fix:
    #   likert "none" Type-I read 28.1% / 50.2% / 79.4% at N=50/100/200,
    #   rising with N exactly as a real effect does, while marginal coverage
    #   stayed at a healthy 95.5% -- the tell that nothing was actually
    #   miscalibrated.
    #
    # So: "none" rows answer "is evalstats calibrated on trustworthy
    # scores?", and the frac=X rows answer "does PPI recover calibration
    # when the judge IS biased?". Both are needed, and conflating them makes
    # the first unreadable.
    judge_biases = (
        _judge_biases_for(eval_type, k) if ppi_frac is not None else np.zeros(k)
    )
    truth_means, _ = _reference_means_for(
        shape, eval_type, k, effects, agreement_rate, rng,
        icc=cell_icc, biases=judge_biases, compute_llm_means=False,
    )
    # PPI cells estimate E[human label]; raw (no-PPI) cells can only ever
    # estimate E[llm_score] -- checking coverage against the wrong one of
    # these is not a real coverage failure, see _reference_means_for.
    # BOTH arms now target truth_means. PPI cells always did (they estimate
    # E[human label]). No-PPI cells used to target llm_means, because they
    # analysed judge-noised scores; with that noise layer gone they analyse
    # the truth draw directly, so the two targets coincide (measured max
    # |truth_means - llm_means| = 0.003 at zero bias).
    true_means = truth_means
    extreme_true_gap = true_means[-1] - true_means[0]

    n_labeled = _n_labeled_for(n_items, ppi_frac) if ppi_frac is not None else 0
    score_range = EVAL_TYPE_SCALE_BOUNDS[eval_type] if eval_type == "likert" else None
    # compare()'s own eval_type=("likert"|"continuous") kwarg -- narrower
    # than this file's own eval_type (which also has "binary"/"grades",
    # neither meaningful to compare()'s param). Passed explicitly rather
    # than left to compare()'s auto-detection (which would otherwise infer
    # the same thing from the data's own quantization grid) so this test's
    # intent is pinned down in the code, not implicit -- and so a future
    # reader isn't left wondering whether a compare() call quietly started
    # resolving to nig instead of logit_t because of a detection heuristic,
    # rather than a deliberate choice recorded here.
    es_eval_type = eval_type if eval_type in ("likert", "continuous") else None

    result = CompareE2EResult(
        eval_type=eval_type, shape_label=shape.label, k=k, n_items=n_items,
        ppi_config=ppi_config, is_null=is_null, n_reps=n_reps, icc=cell_icc,
        effect_step=float(effect_step),
    )

    for _rep in range(n_reps):
        truth = sample_group_truth(shape, n_items, 1, k, cell_icc, rng, effects=effects)[:, :, 0]  # (k, n_items)
        # Judge noise is applied ONLY on the PPI path, where it carries the
        # differential bias PPI exists to remove. The no-PPI arm analyses the
        # truth draw directly, which makes its DGP bit-for-bit identical to
        # scenarios.synthetic.build_multiarm_sources(effect_mode="ramp") --
        # the same builder cases/pvalues.py's multiarm and simultaneous_ci
        # sweeps use (asserted by tests/test_compare_e2e_dgp.py). Layering a
        # second rater-noise pass on top of sample_group_truth's own icc
        # noise put this arm at r(arm_i, arm_j)=0.37, below the lowest point
        # (0.49) ci_paired's official icc sweep ever validates -- so the CI
        # methods were being exercised outside the regime they were tuned in.
        llm_scores = (truth if ppi_frac is None
                      else _apply_judge_noise(truth, eval_type, rng, agreement_rate, judge_biases))

        human_scores = None
        labeled_items = None
        if ppi_frac is not None:
            human_scores = np.full_like(llm_scores, np.nan)
            labeled_items = rng.choice(n_items, size=n_labeled, replace=False)
            human_scores[:, labeled_items] = truth[:, labeled_items]

        df = _build_dataframe(llm_scores, human_scores)

        try:
            evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
            kwargs = {"n_bootstrap": n_bootstrap}
            if ppi_frac is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # ci=False: compare()'s PPI correction reads only the
                    # alignment POINT ESTIMATES, so the per-metric bootstrap
                    # CIs judge_alignment() computes by default are pure cost
                    # here -- verified output-identical, and ~72% of a PPI
                    # cell's runtime.
                    ar = judge_alignment(evaldata, llm_metric="score",
                                         human_groundtruth="human_score", ci=False)
                    kwargs["alignment"] = {"score": ar}
            if classical_rank:
                kwargs.update(CLASSICAL_RANK_KWARGS)
            if score_range is not None:
                kwargs["score_range"] = score_range
            if es_eval_type is not None:
                kwargs["eval_type"] = es_eval_type
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cr = es.compare(evaldata, factors="model", metric="score", rng=rng, **kwargs)
        except Exception:
            result.n_errors += 1
            continue

        bundle = cr._primary_bundle()
        if bundle is None:
            result.n_errors += 1
            continue

        sc = _score_bundle(bundle, true_means, k, alpha, is_null)
        result.marginal_covered += sc["marginal_covered"]
        result.marginal_total += sc["marginal_total"]
        result.marginal_width_sum += sc["marginal_width_sum"]
        result.marginal_score_sum += sc["marginal_score_sum"]
        result.pairwise_covered += sc["pairwise_covered"]
        result.pairwise_total += sc["pairwise_total"]
        result.pairwise_width_sum += sc["pairwise_width_sum"]
        result.pairwise_score_sum += sc["pairwise_score_sum"]
        result.family_covered += sc["family_covered"]
        result.family_total += sc["family_total"]
        result.any_reject += sc["any_reject"]
        result.extreme_reject += sc["extreme_reject"]

        # Reference-estimator comparisons -- see CompareE2EResult's docstring
        # notes on oracle_*/subset_*. Only computed for reference_estimator_k
        # (see _run_cell's docstring -- keeps this ~2x-per-cell cost from
        # applying to the whole grid). Failures here are independent of the
        # main result above (already recorded) and just don't increment
        # oracle_n_ok/subset_n_ok, the denominators _aggregate_group uses.
        if compute_reference and ppi_frac is None:
            try:
                # Continuous only: see ORACLE_NOISE_AGREEMENT_RATE's docstring
                # -- raw truth has an exactly-zero-variance paired diff under
                # icc=1.0's deterministic shift, degenerating any CI built
                # from it. Binary/likert don't need this (already
                # non-degenerate) and stay on raw truth.
                oracle_scores = (
                    _apply_judge_noise(truth, eval_type, rng, ORACLE_NOISE_AGREEMENT_RATE)
                    if eval_type == "continuous" else truth
                )
                oracle_bundle = _run_truth_only_compare(oracle_scores, rng, score_range, n_bootstrap, es_eval_type, classical_rank)
                if oracle_bundle is not None:
                    osc = _score_bundle(oracle_bundle, truth_means, k, alpha, is_null)
                    result.oracle_marginal_covered += osc["marginal_covered"]
                    result.oracle_marginal_total += osc["marginal_total"]
                    result.oracle_pairwise_covered += osc["pairwise_covered"]
                    result.oracle_pairwise_total += osc["pairwise_total"]
                    result.oracle_family_covered += osc["family_covered"]
                    result.oracle_family_total += osc["family_total"]
                    result.oracle_any_reject += osc["any_reject"]
                    result.oracle_extreme_reject += osc["extreme_reject"]
                    result.oracle_n_ok += 1
            except Exception:
                pass
        elif compute_reference:
            try:
                subset_truth = truth[:, labeled_items]
                subset_scores = (
                    _apply_judge_noise(subset_truth, eval_type, rng, ORACLE_NOISE_AGREEMENT_RATE)
                    if eval_type == "continuous" else subset_truth
                )
                subset_bundle = _run_truth_only_compare(subset_scores, rng, score_range, n_bootstrap, es_eval_type, classical_rank)
                if subset_bundle is not None:
                    ssc = _score_bundle(subset_bundle, truth_means, k, alpha, is_null)
                    result.subset_marginal_covered += ssc["marginal_covered"]
                    result.subset_marginal_total += ssc["marginal_total"]
                    result.subset_pairwise_covered += ssc["pairwise_covered"]
                    result.subset_pairwise_total += ssc["pairwise_total"]
                    result.subset_family_covered += ssc["family_covered"]
                    result.subset_family_total += ssc["family_total"]
                    result.subset_any_reject += ssc["any_reject"]
                    result.subset_extreme_reject += ssc["extreme_reject"]
                    result.subset_n_ok += 1
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Fork-pool parallelism (mirrors ci_single.py's _CELL_SOURCES/_run_cell_worker pattern)
# ---------------------------------------------------------------------------

_CELLS: list[dict] = []  # fork-inherited worker state for run_simulation


def _run_cell_worker(args: tuple) -> CompareE2EResult:
    idx, n_reps, alpha, seed, n_bootstrap, reference_estimator_k, classical_rank = args
    cell = _CELLS[idx]
    return _run_cell(
        cell["eval_type"], cell["shape"], cell["k"], cell["n_items"], cell["ppi_frac"], cell["is_null"],
        n_reps, alpha, seed, n_bootstrap=n_bootstrap, reference_estimator_k=reference_estimator_k,
        classical_rank=classical_rank, icc=cell.get("icc", DEFAULT_ICC),
    )


class _ProgressReporter:
    """Prints an in-place progress bar or one-line-per-cell status to stdout,
    rate-limited to avoid flooding the terminal on fast cells."""

    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        """Report progress at *step* out of ``self.total``; a no-op when mode is "off"."""
        if self.mode == "off":
            return
        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.5:
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


def build_cells(
    eval_types: list[str], scenario_suite: str, k_values: list[int], sizes: list[int],
    ppi_fracs: tuple[Optional[float], ...], icc_values: Optional[list[float]] = None,
) -> tuple[list[dict], list[str]]:
    """Enumerate every (eval_type, shape, k, n_items, ppi_config, is_null)
    cell, pre-filtering PPI configs that are guaranteed to fail evalstats'
    own minimum-sample-size checks (see _ppi_applicable). Returns
    (cells, skip_notes)."""
    cells: list[dict] = []
    skip_notes: list[str] = []
    skipped_ppi: dict[str, int] = {}
    for eval_type in eval_types:
        shapes = _tier_shapes(SHAPES_BY_EVAL_TYPE[eval_type], scenario_suite)
        for shape in shapes:
            for k in k_values:
                for n_items in sizes:
                    for ppi_frac in ppi_fracs:
                        if ppi_frac is not None and not _ppi_applicable(k, n_items, ppi_frac):
                            key = (f"k={k},n={n_items},"
                                   + (f"nlab={int(round(ppi_frac))}" if ppi_frac >= 1
                                      else f"frac={ppi_frac:.2f}"))
                            skipped_ppi[key] = skipped_ppi.get(key, 0) + 1
                            continue
                        # icc is swept on the NO-PPI arm only: there it is the
                        # sole noise knob, and a single value is exactly the
                        # blind spot that hid NIG's dispersion sensitivity. On
                        # the PPI arm judge noise dominates, so extra icc cells
                        # buy little for their cost.
                        cell_iccs = (icc_values or [DEFAULT_ICC]) if ppi_frac is None else [DEFAULT_ICC]
                        for cell_icc in cell_iccs:
                            for is_null in (True, False):
                                cells.append(dict(
                                    eval_type=eval_type, shape=shape, k=k, n_items=n_items,
                                    ppi_frac=ppi_frac, is_null=is_null, icc=float(cell_icc),
                                ))
    if skipped_ppi:
        skip_notes.append(
            "Skipped PPI configs below evalstats' own minimum-sample-size floor "
            "(n_lab>=15, n_all=k*n_items>=50) -- these would raise deterministically, "
            "not just occasionally: " + ", ".join(sorted(skipped_ppi))
        )
    return cells, skip_notes


def run_simulation(
    cells: list[dict], n_reps: int, alpha: float, seed: int = 42,
    progress_mode: str = "bar", n_workers: int = 1,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N, reference_estimator_k: Optional[int] = REFERENCE_ESTIMATOR_K,
    null_reps_mult: float = 1.0, classical_rank: bool = False,
) -> list[CompareE2EResult]:
    global _CELLS
    _CELLS = cells
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    # Null cells may run MORE reps than non-null ones (see --null-reps-mult).
    # Type-I is a null-only quantity, so every non-null cell contributes
    # nothing to it; scaling only the null side buys Type-I precision without
    # paying for power precision that is already sufficient.
    args_list = [
        (i, int(round(n_reps * (null_reps_mult if cells[i]["is_null"] else 1))),
         alpha, s, n_bootstrap, reference_estimator_k, classical_rank)
        for i, s in enumerate(child_seeds)
    ]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="compare_e2e")
    results: list[CompareE2EResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.append(_run_cell_worker(a))
            c = cells[i]
            reporter.update(i + 1, detail=f"{c['eval_type']} {c['shape'].label} k={c['k']} n={c['n_items']}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_cell_worker, args_list)):
                results.append(r)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mc_se(rate: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(np.sqrt(max(rate * (1.0 - rate), 0.0) / n))


def _aggregate_group(rows: list[CompareE2EResult]) -> dict:
    """Pool marginal/pairwise/family-wise coverage, marginal width, Type-I,
    and power counts across an arbitrary set of cells (e.g. one (eval_type,
    ppi_config, n_items) row for the full breakdown table, or one (eval_type,
    ppi_config) row pooling every shape/k/n_items for the compact
    paper-table/key-figure view).

    marg_cov is per-arm coverage (unaffected by k -- bundle.robustness's CI
    is never multiplicity-adjusted). pair_cov and fam_cov are DELIBERATELY
    NOT pooled over the same rows: pair_cov uses ONLY k==2 rows (the
    uncorrected baseline -- with one pair, Sidak's own alpha_adj reduces to
    plain alpha, so this is each pairwise CI's genuine own calibration with
    no FWER-widening confound), while fam_cov uses ONLY k>2 rows (where
    simultaneous_ci's FWER machinery actually engages -- P(ALL C(k,2) pairs
    simultaneously covered), the real guarantee simultaneous_ci=True makes).
    Pooling k==2 and k>2 together in one "pairwise coverage" number is what
    made the earlier version of this table/plot look artifactually
    over-conservative (~99%+) instead of showing the true story: uncorrected
    pairwise CIs are calibrated at the nominal rate, and FWER-corrected ones
    correctly protect the whole family at the nominal rate too -- two
    different, both-correct guarantees, not one miscalibrated metric."""
    null_rows = [r for r in rows if r.is_null]
    eff_rows = [r for r in rows if not r.is_null]
    k2_rows = [r for r in rows if r.k == 2]
    kgt2_rows = [r for r in rows if r.k > 2]
    marg_cov_den = sum(r.marginal_total for r in rows)
    pair_cov_den = sum(r.pairwise_total for r in k2_rows)
    fam_cov_den = sum(r.family_total for r in kgt2_rows)
    # Width/score are per-PAIR quantities (unlike family_covered/family_total,
    # the per-REP "ALL pairs held" event) -- k>2's per-pair width/score reuses
    # pairwise_width_sum/pairwise_total filtered to k>2 rows, giving the
    # direct "what does FWER widening cost in width/score" comparison against
    # k==2's own pairwise_width_sum/pairwise_total.
    fam_pair_den = sum(r.pairwise_total for r in kgt2_rows)
    type1_den = sum(r.n_reps - r.n_errors for r in null_rows)
    # 'power' is printed beside ref.pwr as a like-for-like comparison, so it
    # must be measured on the same cells. oracle_*/subset_* only exist at
    # reference_estimator_k, so pooling power over every k made the two
    # columns different populations -- the same mismatch the sandwich row of
    # save_calibration_plot had. Derived from the data rather than hardcoded
    # to 3, so --reference-k None (references at every k) is a no-op here.
    _ref_ks = {r.k for r in rows if r.oracle_n_ok > 0 or r.subset_n_ok > 0}
    power_rows = [r for r in eff_rows if not _ref_ks or r.k in _ref_ks]
    power_den = sum(r.n_reps - r.n_errors for r in power_rows)
    # Reference-estimator power/Type-I: oracle_n_ok is only nonzero on
    # ppi_config=="none" rows, subset_n_ok only nonzero on ppi_config!="none"
    # rows (see CompareE2EResult), so these are automatically NaN wherever
    # they don't apply -- no separate grouping needed.
    oracle_type1_den = sum(r.oracle_n_ok for r in null_rows)
    oracle_power_den = sum(r.oracle_n_ok for r in eff_rows)
    subset_type1_den = sum(r.subset_n_ok for r in null_rows)
    subset_power_den = sum(r.subset_n_ok for r in eff_rows)
    return dict(
        marg_cov=(sum(r.marginal_covered for r in rows) / marg_cov_den) if marg_cov_den else float("nan"),
        marg_width=(sum(r.marginal_width_sum for r in rows) / marg_cov_den) if marg_cov_den else float("nan"),
        marg_score=(sum(r.marginal_score_sum for r in rows) / marg_cov_den) if marg_cov_den else float("nan"),
        pair_cov=(sum(r.pairwise_covered for r in k2_rows) / pair_cov_den) if pair_cov_den else float("nan"),
        pair_width=(sum(r.pairwise_width_sum for r in k2_rows) / pair_cov_den) if pair_cov_den else float("nan"),
        pair_score=(sum(r.pairwise_score_sum for r in k2_rows) / pair_cov_den) if pair_cov_den else float("nan"),
        fam_cov=(sum(r.family_covered for r in kgt2_rows) / fam_cov_den) if fam_cov_den else float("nan"),
        fam_width=(sum(r.pairwise_width_sum for r in kgt2_rows) / fam_pair_den) if fam_pair_den else float("nan"),
        fam_score=(sum(r.pairwise_score_sum for r in kgt2_rows) / fam_pair_den) if fam_pair_den else float("nan"),
        type1=(sum(r.any_reject for r in null_rows) / type1_den) if type1_den else float("nan"),
        power=(sum(r.extreme_reject for r in power_rows) / power_den) if power_den else float("nan"),
        oracle_type1=(sum(r.oracle_any_reject for r in null_rows) / oracle_type1_den) if oracle_type1_den else float("nan"),
        oracle_power=(sum(r.oracle_extreme_reject for r in eff_rows) / oracle_power_den) if oracle_power_den else float("nan"),
        subset_type1=(sum(r.subset_any_reject for r in null_rows) / subset_type1_den) if subset_type1_den else float("nan"),
        subset_power=(sum(r.subset_extreme_reject for r in eff_rows) / subset_power_den) if subset_power_den else float("nan"),
        n_cells=len(rows), n_errors=sum(r.n_errors for r in rows),
    )


def overall_summary_rows(results: list[CompareE2EResult]) -> list[dict]:
    """One aggregated row per (eval_type, ppi_config), pooling every
    shape/k/n_items cell -- the compact "systematic good performance" view,
    shared by the printed KEY SUMMARY block, the LaTeX table, and the
    size-power key figure."""
    groups: dict[tuple, list[CompareE2EResult]] = {}
    for r in results:
        groups.setdefault((r.eval_type, r.ppi_config), []).append(r)
    out = []
    for (eval_type, ppi_config) in sorted(
        groups, key=lambda k: (k[0], PPI_CONFIG_ORDER.index(k[1]) if k[1] in PPI_CONFIG_ORDER else 99),
    ):
        agg = _aggregate_group(groups[(eval_type, ppi_config)])
        out.append(dict(eval_type=eval_type, ppi_config=ppi_config, **agg))
    return out


def print_key_summary(rows: list[dict], alpha: float) -> None:
    """The compact deliverable table: one row per (eval_type, ppi_config),
    pooled across every shape/k/n_items cell -- "does compare()'s recommended
    path hold up broadly" at a glance."""
    print(f"\n{'='*116}")
    print("compare_e2e -- KEY SUMMARY (pooled across all shapes/k/n_items)")
    print(f"{'='*116}")
    header = (
        f"{'eval_type':<11} {'ppi_config':<11} | {'marg.cov':>9} {'pair.cov':>9} {'fam.cov':>9} | "
        f"{'Type-I':>8} {'power':>8} {'ref.pwr':>9} | {'n_cells':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ref_power = r["oracle_power"] if r["ppi_config"] == "none" else r["subset_power"]
        print(
            f"{r['eval_type']:<11} {r['ppi_config']:<11} | "
            f"{r['marg_cov']*100:>8.1f}% {r['pair_cov']*100:>8.1f}% {r['fam_cov']*100:>8.1f}% | "
            f"{r['type1']*100:>7.1f}% {r['power']*100:>7.1f}% {ref_power*100:>8.1f}% | {r['n_cells']:>7}"
        )
    print()
    print(f"marg.cov/pair.cov/fam.cov target: {(1-alpha)*100:.0f}%.  Type-I target: <= {alpha*100:.0f}%.")
    print(
        "pair.cov uses ONLY k=2 cells -- an uncorrected baseline (Sidak's own alpha_adj reduces to "
        "plain alpha with 1 pair), so this is each pairwise CI's genuine own calibration. fam.cov uses "
        "ONLY k>2 cells -- P(ALL C(k,2) pairs simultaneously covered), the actual simultaneous_ci=True "
        "guarantee. Both should land near the target; they are not the same rows pooled two ways."
    )
    print(
        "ref.pwr is a REFERENCE-ESTIMATOR power comparison, same synthetic draw AND the same "
        "cells as 'power' (both are restricted to the k values where reference arms were "
        "measured, i.e. --reference-k): for "
        "ppi_config='none' it is 'oracle' power (as if EVERY item were human-labeled -- the ceiling); "
        "for ppi_config='frac=X' it is 'subset-only' power (as if ONLY the labeled X-fraction existed, "
        "LLM data discarded -- the floor PPI should beat). PPI's own 'power' column should sit between "
        "its row's ref.pwr (subset-only) and the 'none' row's ref.pwr (oracle)."
    )


def latex_key_summary_table(rows: list[dict], alpha: float, n_reps: int) -> str:
    """LaTeX booktabs version of print_key_summary -- the paper-ready table."""
    table_rows = []
    for r in rows:
        ref_power = r["oracle_power"] if r["ppi_config"] == "none" else r["subset_power"]
        table_rows.append([
            escape_latex(r["eval_type"]), escape_latex(r["ppi_config"]),
            f"{r['marg_cov']*100:.1f}", f"{r['pair_cov']*100:.1f}", f"{r['fam_cov']*100:.1f}",
            f"{r['type1']*100:.1f}", f"{r['power']*100:.1f}", f"{ref_power*100:.1f}", str(r["n_cells"]),
        ])
    rule_before = {i for i in range(1, len(rows)) if rows[i]["eval_type"] != rows[i - 1]["eval_type"]}
    return booktabs_table(
        caption=(
            f"\\texttt{{compare()}} black-box calibration, pooled across all shapes, arm counts (k), "
            f"and per-arm sample sizes (N) swept, reps/cell={n_reps}. Nominal CI coverage is "
            f"{(1-alpha)*100:.0f}\\%; Type-I target is $\\le {alpha*100:.0f}\\%$. "
            "``none'' is the raw (no-alignment) estimator; ``frac=$x$'' is PPI alignment correction "
            "with a fraction $x$ of items human-labeled, both under the same FWER-controlled "
            "simultaneous CIs and p-value correction. ``Pair.\\ Cov.'' uses only $k=2$ cells -- an "
            "uncorrected baseline, since Sidak's own per-comparison alpha reduces to the nominal alpha "
            "with a single pair -- so it reports each pairwise CI's genuine own calibration. "
            "``Fam.\\ Cov.'' uses only $k>2$ cells and is the actual family-wise guarantee "
            "(P(all $C(k,2)$ pairs simultaneously covered)) that simultaneous CIs are built to provide. "
            "Both columns should land near nominal; they are not the same underlying rows pooled two ways. "
            "``Ref.\\ Pwr.'' is a reference-estimator power comparison from the SAME synthetic draw: for "
            "``none'' it is ``oracle'' power (every item human-labeled, the ceiling); for ``frac=$x$'' it "
            "is ``subset-only'' power (only the labeled fraction used, LLM data discarded, the floor PPI "
            "should beat). PPI's own Power column should sit between its row's Ref.\\ Pwr.\\ and the "
            "``none'' row's Ref.\\ Pwr."
        ),
        label="tab:compare_e2e_key_summary",
        columns=["Eval type", "PPI config", "Marg. Cov. (\\%)", "Pair. Cov. (\\%)", "Fam. Cov. (\\%)",
                 "Type-I (\\%)", "Power (\\%)", "Ref. Pwr. (\\%)", "N cells"],
        rows=table_rows,
        rule_before=rule_before,
    )


def _sum_attr(rows: list[CompareE2EResult], attr: str) -> float:
    """attr == '_n_ok' sums (n_reps - n_errors) -- the denominator Type-I/
    power use, which isn't a single stored field -- otherwise sums the named
    field directly (e.g. 'marginal_total', 'family_covered')."""
    if attr == "_n_ok":
        return sum(r.n_reps - r.n_errors for r in rows)
    return sum(getattr(r, attr) for r in rows)


def _format_log_size_axis(ax, sizes) -> None:
    """Plain integer tick labels (15, 30, 60, ...) at exactly the n_items
    values swept, instead of log-scale's default "3x10^1"-style scientific
    notation -- these are small, meaningful sample sizes someone will read
    off directly, not physical quantities spanning many orders of magnitude
    where scientific notation is the natural choice."""
    import matplotlib.ticker as mticker

    sizes = sorted(set(sizes))
    if not sizes:
        return
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(n)) for n in sizes], fontsize=7, rotation=45 if len(sizes) > 5 else 0)
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def save_calibration_plot(*, results: list[CompareE2EResult], alpha: float, out_path: str) -> str:
    """THE hero figure: the whole calibration-and-power story in one image,
    one column per eval_type, one line per ppi_config, seven rows:

      1. Marginal CI coverage    -- per arm; bundle.robustness's CI is never
                                     multiplicity-adjusted by k.
      2. Marginal CI width       -- sharpness complement to row 1 (intervals
                                     shrink with N, not just "always cover").
      3. Pairwise CI coverage    -- k=2 ONLY: an uncorrected baseline (with a
                                     single pair, Sidak's own alpha_adj
                                     reduces to plain alpha), so this is each
                                     pairwise CI's genuine own calibration.
      4. Family-wise CI coverage -- k>2 ONLY: P(ALL C(k,2) pairs
                                     simultaneously covered), the actual
                                     guarantee simultaneous_ci=True makes.
      5. Type-I error (FWER)     -- null cells, any pairwise p < alpha.
      6. Power (extreme pair)    -- non-null cells, the largest-true-gap
                                     pair's p < alpha.
      7. Power: PPI vs. subset-only vs. oracle -- the "sandwich": PPI
                                     (solid, all N items, frac human-labeled)
                                     should sit between "subset-only" (same
                                     color, DASHED -- only the frac*N labeled
                                     items, LLM data discarded) and "oracle"
                                     (black, dotted -- every item
                                     human-labeled, the power ceiling).

    Rows 3/4 are DELIBERATELY not the same cells pooled two ways -- pooling
    k=2 and k>2 pairwise cells together (the original version of this plot)
    makes "pairwise coverage" look artifactually over-conservative (~99%+)
    instead of showing the true, cleaner story: uncorrected pairwise CIs are
    calibrated at the nominal rate (row 3), and FWER-corrected ones protect
    the whole family at the nominal rate too (row 4) -- two different, both
    CORRECT guarantees, not one miscalibrated metric. See _aggregate_group's
    docstring for the full reasoning. Row 7's oracle/subset-only lines are a
    DIFFERENT split of the data than rows 5/6's coverage-plot lines: oracle
    is measured on "none" cells (attached there since it doesn't depend on
    frac), subset-only on "frac=X" cells -- see CompareE2EResult and
    _run_truth_only_compare."""
    import matplotlib.pyplot as plt

    eval_types = sorted({r.eval_type for r in results})
    target = 1.0 - alpha

    row_specs = [
        dict(label="Marginal CI coverage\n(per arm)", pool=lambda rs: rs,
             num="marginal_covered", den="marginal_total", ylim=(0.5, 1.02), hline=target, has_se=True),
        dict(label="Marginal CI width\n(per arm, raw scale units)", pool=lambda rs: rs,
             num="marginal_width_sum", den="marginal_total", ylim=None, hline=None, has_se=False),
        dict(label="Pairwise CI coverage\n(k=2, UNCORRECTED baseline)", pool=lambda rs: [r for r in rs if r.k == 2],
             num="pairwise_covered", den="pairwise_total", ylim=(0.5, 1.02), hline=target, has_se=True),
        dict(label="Family-wise CI coverage\n(k>2, FWER-CORRECTED)", pool=lambda rs: [r for r in rs if r.k > 2],
             num="family_covered", den="family_total", ylim=(max(0.0, target - 0.15), 1.02), hline=target, has_se=True),
        dict(label="Type-I error\n(FWER, null cells)", pool=lambda rs: [r for r in rs if r.is_null],
             num="any_reject", den="_n_ok", ylim=(-0.02, max(0.20, alpha * 3)), hline=alpha, has_se=True),
        dict(label="Power\n(extreme pair, non-null cells)", pool=lambda rs: [r for r in rs if not r.is_null],
             num="extreme_reject", den="_n_ok", ylim=(-0.02, 1.02), hline=None, has_se=True),
    ]
    n_rows = len(row_specs) + 1  # +1 for the hand-drawn power-sandwich row
    sandwich_row = len(row_specs)
    fig, axes = plt.subplots(n_rows, len(eval_types), figsize=(5.0 * len(eval_types), 3.4 * n_rows), squeeze=False)

    for col, eval_type in enumerate(eval_types):
        et_results = [r for r in results if r.eval_type == eval_type]
        for row, spec in enumerate(row_specs):
            ax = axes[row][col]
            pooled = spec["pool"](et_results)
            all_sizes = sorted({r.n_items for r in pooled})
            for ppi_config in PPI_CONFIG_ORDER:
                cfg_results = [r for r in pooled if r.ppi_config == ppi_config]
                if not cfg_results:
                    continue
                sizes = sorted({r.n_items for r in cfg_results})
                ys, ses = [], []
                for n in sizes:
                    rows = [r for r in cfg_results if r.n_items == n]
                    num = _sum_attr(rows, spec["num"])
                    den = _sum_attr(rows, spec["den"])
                    rate = num / den if den else float("nan")
                    ys.append(rate)
                    ses.append(_mc_se(rate, den) if spec["has_se"] else float("nan"))
                color = PPI_CONFIG_COLORS[ppi_config]
                if spec["has_se"]:
                    ax.errorbar(sizes, ys, yerr=ses, marker="o", markersize=4, label=ppi_config, color=color)
                else:
                    ax.plot(sizes, ys, marker="o", markersize=4, label=ppi_config, color=color)
            if spec["hline"] is not None:
                ax.axhline(spec["hline"], linestyle="--", color="tab:cyan", linewidth=1,
                           label="target" if col == 0 and row == 0 else None)
            ax.set_xscale("log")
            _format_log_size_axis(ax, all_sizes)
            if spec["ylim"] is not None:
                ax.set_ylim(*spec["ylim"])
            if row == 0:
                ax.set_title(eval_type)
            if col == 0:
                ax.set_ylabel(spec["label"])
            if row == 0 and col == len(eval_types) - 1:
                ax.legend(fontsize=8, loc="lower right")

        # Row 7 (hand-drawn, not the generic num/den mechanism above): the
        # power sandwich. "none"/"frac=X" reuse the same extreme_reject/_n_ok
        # main-result fields as row 6; oracle/subset-only pull from the
        # dedicated oracle_*/subset_* fields instead.
        ax = axes[sandwich_row][col]
        # Restrict to the k values where the reference arms actually EXIST.
        # oracle_*/subset_* are only computed at reference_estimator_k (the
        # official preset passes --reference-k 3), but the PPI/none series
        # read extreme_reject, which every k has. Without this filter the
        # solid lines pool k=2..10 while the dashed/dotted ones they are
        # meant to be compared against are k=3 alone -- different
        # populations on the same axes, which is not the sandwich this row
        # claims to draw.
        #
        # This was masked for as long as the effect ramp grew with k: the
        # extra k=5/k=10 power inflated the pooled solid line above the
        # k=3-only dashed one, so the ordering looked right for the wrong
        # reason. Once MAX_EFFECT_SPAN_STEPS made the ramp k-invariant, the
        # pooled line fell BELOW its own baseline and the mismatch showed up
        # as "PPI is worse than using the labels alone" -- a plotting
        # artefact, not a result: at matched k the PPI gain is unchanged
        # (binary +11.82% before and after, bit-identical).
        _ref_ks = {r.k for r in et_results
                   if r.oracle_n_ok > 0 or r.subset_n_ok > 0}
        eff_results = [r for r in et_results
                       if not r.is_null and (not _ref_ks or r.k in _ref_ks)]
        sandwich_sizes: set[int] = set()

        def _power_series(rows_by_n_getter, num_attr: str, den_attr: str):
            cfg_rows = rows_by_n_getter(eff_results)
            sizes = sorted({r.n_items for r in cfg_rows})
            sandwich_sizes.update(sizes)
            ys, ses = [], []
            for n in sizes:
                rows = [r for r in cfg_rows if r.n_items == n]
                num = _sum_attr(rows, num_attr)
                den = _sum_attr(rows, den_attr)
                rate = num / den if den else float("nan")
                ys.append(rate)
                ses.append(_mc_se(rate, den))
            return sizes, ys, ses

        # "none" (raw, no PPI) and its "oracle" companion (every item labeled).
        none_sizes, none_ys, none_ses = _power_series(
            lambda rs: [r for r in rs if r.ppi_config == "none"], "extreme_reject", "_n_ok")
        if none_sizes:
            ax.errorbar(none_sizes, none_ys, yerr=none_ses, marker="o", markersize=4,
                        label="none", color=PPI_CONFIG_COLORS["none"])
        oracle_sizes, oracle_ys, oracle_ses = _power_series(
            lambda rs: [r for r in rs if r.ppi_config == "none"], "oracle_extreme_reject", "oracle_n_ok")
        if oracle_sizes:
            ax.errorbar(oracle_sizes, oracle_ys, yerr=oracle_ses, marker="^", markersize=4, linestyle=":",
                        label="oracle (all items labeled)", color="black")

        # Each PPI frac (solid) paired with its subset-only baseline (dashed,
        # same color) -- the direct "did PPI earn its keep over just using
        # the labeled subset alone" comparison.
        for ppi_config in PPI_CONFIG_ORDER:
            if ppi_config == "none":
                continue
            color = PPI_CONFIG_COLORS[ppi_config]
            ppi_sizes, ppi_ys, ppi_ses = _power_series(
                lambda rs, pc=ppi_config: [r for r in rs if r.ppi_config == pc], "extreme_reject", "_n_ok")
            if ppi_sizes:
                ax.errorbar(ppi_sizes, ppi_ys, yerr=ppi_ses, marker="o", markersize=4, label=ppi_config, color=color)
            sub_sizes, sub_ys, sub_ses = _power_series(
                lambda rs, pc=ppi_config: [r for r in rs if r.ppi_config == pc], "subset_extreme_reject", "subset_n_ok")
            if sub_sizes:
                ax.errorbar(sub_sizes, sub_ys, yerr=sub_ses, marker="s", markersize=4, linestyle="--",
                            label=f"subset only ({ppi_config})", color=color)

        ax.set_xscale("log")
        _format_log_size_axis(ax, sandwich_sizes)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("n_items (per arm)")
        if col == 0:
            ax.set_ylabel("Power: PPI vs. subset-only\nvs. oracle (non-null cells)")
        if col == len(eval_types) - 1:
            # Up to 8 entries (none, oracle, 3 fracs x2 each) -- placed OUTSIDE
            # the axes so it can never sit on top of this row's own data
            # (unlike an inside corner, which this row's few-lines-few-points
            # data can easily fall behind).
            ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    fig.suptitle(
        "Confidence-interval coverage and statistical power of compare(), across data types and sample sizes\n"
        "Columns: binary / continuous / Likert data. Color: fraction of items human-labeled under PPI alignment.",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        # rect reserves top margin for the two-line suptitle regardless of
        # aspect ratio (a single-eval_type run is much narrower/taller than
        # the full 3-column grid and would otherwise overlap the title).
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_report(results: list[CompareE2EResult], alpha: float) -> None:
    """The systematic summary table: rows grouped by (eval_type, ppi_config,
    n_items), columns = marginal/pairwise/family-wise CI coverage %, Type-I
    error %, power %, each averaged over shapes and k values present for that
    row (shape/k detail is in the saved CSV, not this top-level table)."""
    print(f"\n{'='*110}")
    print("compare_e2e — black-box compare() validation")
    print(f"{'='*110}")
    print(f"alpha={alpha}  |  {len(results)} cells total\n")

    groups: dict[tuple, list[CompareE2EResult]] = {}
    for r in results:
        key = (r.eval_type, r.ppi_config, r.n_items)
        groups.setdefault(key, []).append(r)

    header = (
        f"{'eval_type':<11} {'ppi_config':<11} {'n_items':>8} | "
        f"{'marg.cov':>9} {'pair.cov':>9} {'fam.cov':>9} | {'Type-I':>8} {'power':>8} | {'errs':>6}"
    )
    print(header)
    print("-" * len(header))

    for (eval_type, ppi_config, n_items) in sorted(groups, key=lambda k: (k[0], k[1] != "none", k[1], k[2])):
        rows = groups[(eval_type, ppi_config, n_items)]
        agg = _aggregate_group(rows)

        print(
            f"{eval_type:<11} {ppi_config:<11} {n_items:>8} | "
            f"{agg['marg_cov']*100:>8.1f}% {agg['pair_cov']*100:>8.1f}% {agg['fam_cov']*100:>8.1f}% | "
            f"{agg['type1']*100:>7.1f}% {agg['power']*100:>7.1f}% | {agg['n_errors']:>6}"
        )
    print()
    print("marg.cov/pair.cov/fam.cov target: (1-alpha)*100%.  Type-I target: <= alpha*100%.")
    print(
        "pair.cov uses ONLY k=2 cells -- an uncorrected baseline (Sidak's own alpha_adj reduces to "
        "plain alpha with 1 pair), so this is each pairwise CI's genuine own calibration. fam.cov uses "
        "ONLY k>2 cells -- P(ALL C(k,2) pairs simultaneously covered), the actual simultaneous_ci=True "
        "guarantee. Both should land near the target; they are not the same rows pooled two ways."
    )
    print("(Aggregated over all shapes/k-values present for each row; see the saved CSV for the full per-cell breakdown.)")


def save_results_artifacts(
    *, results: list[CompareE2EResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "eval_type", "shape_label", "k", "n_items", "ppi_config", "is_null", "icc", "effect_step",
            "n_reps", "n_errors",
            "marginal_covered", "marginal_total", "marginal_coverage", "marginal_mean_width", "marginal_mean_score",
            "pairwise_covered", "pairwise_total", "pairwise_coverage", "pairwise_mean_width", "pairwise_mean_score",
            "family_covered", "family_total", "family_coverage",
            "any_reject", "extreme_reject", "type1_rate", "power_rate",
            "oracle_n_ok", "oracle_type1_rate", "oracle_power_rate",
            "subset_n_ok", "subset_type1_rate", "subset_power_rate",
            # Raw sums/counts, so the CSV is a LOSSLESS serialization of
            # CompareE2EResult -- every rate column above is derivable from
            # these, but not the reverse (the reference-arm coverage counts
            # appear nowhere else, and reconstructing width/score sums from
            # the rounded means loses precision). Anything rebuilding results
            # to regenerate a plot should read these, not the rates.
            "marginal_width_sum", "marginal_score_sum",
            "pairwise_width_sum", "pairwise_score_sum",
            "oracle_marginal_covered", "oracle_marginal_total",
            "oracle_pairwise_covered", "oracle_pairwise_total",
            "oracle_family_covered", "oracle_family_total",
            "oracle_any_reject", "oracle_extreme_reject",
            "subset_marginal_covered", "subset_marginal_total",
            "subset_pairwise_covered", "subset_pairwise_total",
            "subset_family_covered", "subset_family_total",
            "subset_any_reject", "subset_extreme_reject",
        ])
        for r in results:
            n_ok = r.n_reps - r.n_errors
            marg_cov = r.marginal_covered / r.marginal_total if r.marginal_total else float("nan")
            marg_width = r.marginal_width_sum / r.marginal_total if r.marginal_total else float("nan")
            marg_score = r.marginal_score_sum / r.marginal_total if r.marginal_total else float("nan")
            pair_cov = r.pairwise_covered / r.pairwise_total if r.pairwise_total else float("nan")
            pair_width = r.pairwise_width_sum / r.pairwise_total if r.pairwise_total else float("nan")
            pair_score = r.pairwise_score_sum / r.pairwise_total if r.pairwise_total else float("nan")
            fam_cov = r.family_covered / r.family_total if r.family_total else float("nan")
            type1 = r.any_reject / n_ok if (r.is_null and n_ok) else float("nan")
            power = r.extreme_reject / n_ok if (not r.is_null and n_ok) else float("nan")
            oracle_type1 = r.oracle_any_reject / r.oracle_n_ok if (r.is_null and r.oracle_n_ok) else float("nan")
            oracle_power = r.oracle_extreme_reject / r.oracle_n_ok if (not r.is_null and r.oracle_n_ok) else float("nan")
            subset_type1 = r.subset_any_reject / r.subset_n_ok if (r.is_null and r.subset_n_ok) else float("nan")
            subset_power = r.subset_extreme_reject / r.subset_n_ok if (not r.is_null and r.subset_n_ok) else float("nan")
            writer.writerow([
                r.eval_type, r.shape_label, r.k, r.n_items, r.ppi_config, r.is_null, f"{r.icc:.4f}",
                # repr(float(...)) not a rounded format: effect_step is a DGP
                # input, so a run must be reproducible from it exactly.
                repr(float(r.effect_step)), r.n_reps, r.n_errors,
                r.marginal_covered, r.marginal_total, f"{marg_cov:.6f}", f"{marg_width:.6f}", f"{marg_score:.6f}",
                r.pairwise_covered, r.pairwise_total, f"{pair_cov:.6f}", f"{pair_width:.6f}", f"{pair_score:.6f}",
                r.family_covered, r.family_total, f"{fam_cov:.6f}",
                r.any_reject, r.extreme_reject, f"{type1:.6f}", f"{power:.6f}",
                r.oracle_n_ok, f"{oracle_type1:.6f}", f"{oracle_power:.6f}",
                r.subset_n_ok, f"{subset_type1:.6f}", f"{subset_power:.6f}",
                repr(float(r.marginal_width_sum)), repr(float(r.marginal_score_sum)),
                repr(float(r.pairwise_width_sum)), repr(float(r.pairwise_score_sum)),
                r.oracle_marginal_covered, r.oracle_marginal_total,
                r.oracle_pairwise_covered, r.oracle_pairwise_total,
                r.oracle_family_covered, r.oracle_family_total,
                r.oracle_any_reject, r.oracle_extreme_reject,
                r.subset_marginal_covered, r.subset_marginal_total,
                r.subset_pairwise_covered, r.subset_pairwise_total,
                r.subset_family_covered, r.subset_family_total,
                r.subset_any_reject, r.subset_extreme_reject,
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, alpha=alpha)
        key_rows = overall_summary_rows(results)
        print_key_summary(key_rows, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        n_reps = max((r.n_reps for r in results), default=0)
        summary_text += "\n% --- LaTeX key summary table (--latex) ---\n" + latex_key_summary_table(
            key_rows, alpha=alpha, n_reps=n_reps,
        )
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", choices=["synthetic", "real"], default="synthetic",
                         help="'synthetic' (default). 'real' is a no-op for this case -- "
                              "there is no ground truth on real judge-bias data to check "
                              "coverage/power against.")
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="standard",
                         help="Shape-catalog breadth per eval_type (standard=coarse, default).")
    parser.add_argument("--eval-types", nargs="+", choices=["binary", "continuous", "likert"],
                         default=["binary", "continuous", "likert"], metavar="TYPE")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES, metavar="K")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, metavar="N")
    parser.add_argument("--ppi-fracs", type=str, nargs="+", default=["none", "0.10", "0.20", "0.40"], metavar="FRAC",
                         help="'none' (no PPI) or a label fraction in (0, 1].")
    parser.add_argument("--icc-values", type=float, nargs="+", default=list(DEFAULT_ICC_VALUES),
                         metavar="ICC",
                         help="Sweep sample_group_truth's signal/noise split on the NO-PPI arm "
                              "(PPI cells stay at the default, where judge noise dominates). "
                              f"Default {list(DEFAULT_ICC_VALUES)}, matching official_args -- pass a "
                              f"single value (e.g. --icc-values {DEFAULT_ICC}) for the cheaper grid. "
                              "ci_paired's official sweep uses 0.05/0.20/0.40/0.60/0.80.")
    parser.add_argument("--reps", type=int, default=100, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N, metavar="N",
                         help=f"n_bootstrap passed to every compare() call (default {DEFAULT_BOOTSTRAP_N}, vs. "
                              "compare()'s own 10_000 default) -- the dominant per-call cost. NOT free to lower "
                              "further: this repo's own pvalues.py investigation found romano_wolf/boot's FWER "
                              "ran measurably hot at bootstrap_n=500-2000 (MC noise in the joint max-statistic's "
                              "quantile), which would confound this case's own Type-I findings.")
    parser.add_argument("--reference-k", type=int, default=REFERENCE_ESTIMATOR_K, metavar="K",
                         help="Only cells at this k value compute the oracle/subset-only reference estimators "
                              f"(row 7 of the hero plot) -- roughly doubles cost per cell it applies to, so "
                              f"restricting it to one k (default {REFERENCE_ESTIMATOR_K}) keeps that cost from "
                              "applying to the whole grid. Pass -1 to compute it for every k (slower, more complete).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--save-results", choices=["save", "off"], default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save",
                         help="Save the coverage-vs-n and size/power diagnostic figures (default) or skip them.")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs key-summary table (the paper table) to the saved summary .log file.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential).")
    parser.add_argument("--classical-rank-path", action="store_true", default=False,
                         help="Drive compare() down the rank-based pathway the paper reports "
                              "(Friedman omnibus + Wilcoxon pairwise + Shaffer FWER) instead of "
                              "letting correction=auto resolve to Romano-Wolf. Applied to the PPI, "
                              "oracle and human-subset arms alike so the comparison stays "
                              "like-for-like. Off by default: Romano-Wolf is the better method and "
                              "remains what compare() recommends.")
    parser.add_argument("--null-reps-mult", type=float, default=1.0, metavar="M",
                         help="Run null cells at M x --reps (default 1.0 = same as non-null). "
                              "Type-I error is a NULL-ONLY quantity, so non-null cells contribute "
                              "nothing to its precision, and it is by far the widest-banded metric "
                              "in this case: one decision per replicate rather than k CIs, null "
                              "cells only, and k>2 cells only, which together leave it a ~10x "
                              "smaller denominator than marginal coverage (1400 vs 14000 at "
                              "reps=200). Scaling only the null side buys that precision without "
                              "paying for power precision that is already ample -- M=4 quarters "
                              "the Type-I variance for ~2.5x runtime instead of the 4x a blanket "
                              "--reps increase would cost.")


def _parse_ppi_fracs(raw: list[str]) -> tuple[Optional[float], ...]:
    """"none", a fraction < 1 (e.g. 0.20), or an absolute label count >= 1
    (e.g. 30) held fixed across n_items -- see _n_labeled_for."""
    out: list[Optional[float]] = []
    for s in raw:
        if s.lower() == "none":
            out.append(None)
        else:
            out.append(float(s))
    return tuple(out)


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset. Full grid -- all of DEFAULT_SIZES
    (including 800), all of DEFAULT_PPI_FRACS' fractions (including 0.10),
    reps=100 -- breadth-over-depth given the grid's size. Two runtime knobs
    are still tuned down from their most-expensive possible values, reviewed
    and kept:

      - n_bootstrap=DEFAULT_BOOTSTRAP_N (5000, vs. compare()'s own 10_000
        default): picked deliberately, not just "as low as possible" --
        this repo's own prior investigation (pvalues.py) found romano_wolf/
        boot's FWER measurably biased at bootstrap_n=500-2000, which would
        confound this case's OWN Type-I findings, so 5000 stays inside the
        range already validated as safe rather than trading correctness for
        speed.
      - reference_estimator_k=REFERENCE_ESTIMATOR_K (3): the oracle/
        subset-only reference-estimator comparison (row 7 of the hero plot)
        only computes for one representative k instead of every k, since it
        roughly doubles cost per cell it applies to and the sandwich story
        doesn't need every k value to be informative -- see that constant's
        docstring. Pass reference_k=-1 (via --reference-k -1) for every k.

    All of --reps/--sizes/--k-values/--ppi-fracs/--bootstrap-n/--reference-k
    stay CLI-overridable in either direction (deeper or shallower) for a
    different pass later."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="standard",
        eval_types=["binary", "continuous", "likert"],
        k_values=DEFAULT_K_VALUES, sizes=DEFAULT_SIZES,
        ppi_fracs=["none", "0.10", "0.20", "0.40"],
        # Sweeps the no-PPI arm's signal/noise split across ci_paired's
        # low/mid/high (its official sweep is 0.05-0.80). A single icc is what
        # let the Likert NIG interval's dispersion sensitivity go unseen; three
        # points cost +70% cells against +139% for the full five.
        icc_values=list(DEFAULT_ICC_VALUES),
        reps=100, alpha=0.05, seed=base_seed, progress="bar", save_results="save",
        out_dir="simulations/out", plots="save", plots_dir=None, latex=True,
        workers=max(1, (os.cpu_count() or 2) - 1),
        bootstrap_n=DEFAULT_BOOTSTRAP_N, reference_k=REFERENCE_ESTIMATOR_K,
    )


def official_args_classical_rank(base_seed: int = 42) -> argparse.Namespace:
    """official_args, but down the rank-based pathway the paper reports
    (see CLASSICAL_RANK_KWARGS): Friedman omnibus, Wilcoxon pairwise,
    Shaffer FWER, on the PPI, oracle and human-subset arms alike.

    This is the official arm, not the plain one, because the paper
    validates and reports the rank-based tests -- so this is the
    configuration a reviewer can actually check against the oracle and
    human-subset references. compare()'s own recommendation is still
    Romano-Wolf (correction="auto"), which official_args below keeps
    available; the discrepancy is discussed in the paper rather than
    hidden."""
    args = official_args(base_seed)
    args.classical_rank_path = True
    return args


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    return [
        ("synthetic (Friedman/Wilcoxon/Shaffer -- the reported path)",
         official_args_classical_rank(base_seed)),
        ("synthetic (Romano-Wolf -- compare()'s own default)", official_args(base_seed)),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast pipeline-sanity preset for --quick-test. ``data_source="real"``
    is a documented no-op (see add_arguments) -- returned as-is so --quick-test's
    per-case real-data pass still completes (gracefully) rather than crashing."""
    return argparse.Namespace(
        data_source=data_source, scenario_suite="standard",
        eval_types=["binary"], k_values=[3], sizes=[30, 100],
        ppi_fracs=["none", "0.20"],
        reps=3, alpha=0.05, seed=base_seed, progress="bar", save_results="save",
        out_dir="simulations/out", plots="save", plots_dir=None, latex=False,
        workers=1, bootstrap_n=DEFAULT_BOOTSTRAP_N, reference_k=REFERENCE_ESTIMATOR_K,
    )


def run(args: argparse.Namespace) -> CaseResult:
    """Case entry point: build the cell grid from *args*, run the simulation,
    print/save reports and the calibration plot, and return a CaseResult
    summarizing overall marginal/pairwise coverage."""
    t0 = time.time()
    try:
        if getattr(args, "data_source", "synthetic") == "real":
            return CaseResult(
                case_name=CASE_NAME, status="ok",
                key_metrics={"note": "data_source='real' not applicable for this case -- "
                                      "no ground truth exists for real judge-bias data to "
                                      "check CI coverage/power against."},
                duration_s=time.time() - t0,
            )

        ppi_fracs = _parse_ppi_fracs(args.ppi_fracs)
        cells, skip_notes = build_cells(
            args.eval_types, args.scenario_suite, args.k_values, args.sizes, ppi_fracs,
            icc_values=getattr(args, "icc_values", None),
        )
        for note in skip_notes:
            print(f"  Note: {note}")
        print(f"\ncompare_e2e simulation -- {len(cells)} cells, reps={args.reps}, alpha={args.alpha}")

        reference_k = getattr(args, "reference_k", REFERENCE_ESTIMATOR_K)
        results = run_simulation(
            cells, n_reps=args.reps, alpha=args.alpha, seed=args.seed,
            null_reps_mult=getattr(args, "null_reps_mult", 1.0),
            progress_mode=args.progress, n_workers=getattr(args, "workers", 1),
            n_bootstrap=getattr(args, "bootstrap_n", DEFAULT_BOOTSTRAP_N),
            reference_estimator_k=(None if reference_k == -1 else reference_k),
            classical_rank=getattr(args, "classical_rank_path", False),
        )
        print_report(results, alpha=args.alpha)
        print_key_summary(overall_summary_rows(results), alpha=args.alpha)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"compare_e2e_reps{args.reps}_{stamp}"
        output_paths: list[str] = []
        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results=results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                latex=getattr(args, "latex", False),
            )

        if getattr(args, "plots", "save") == "save":
            plots_dir = getattr(args, "plots_dir", None) or str(Path(args.out_dir) / "plots")
            hero_path = save_calibration_plot(
                results=results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_calibration.png"),
            )
            output_paths += [hero_path]
            print(f"Saved plot: {hero_path}  (key figure)")

        overall_marg_cov = (
            sum(r.marginal_covered for r in results) / sum(r.marginal_total for r in results)
            if sum(r.marginal_total for r in results) else float("nan")
        )
        overall_pair_cov = (
            sum(r.pairwise_covered for r in results) / sum(r.pairwise_total for r in results)
            if sum(r.pairwise_total for r in results) else float("nan")
        )
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={
                "n_cells": len(results),
                "overall_marginal_coverage": overall_marg_cov,
                "overall_pairwise_coverage": overall_pair_cov,
            },
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
