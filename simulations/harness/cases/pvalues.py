"""pvalues case: p-value/rejection-decision calibration, non-PPI and PPI-corrected.

Consolidates two complementary benchmarks into one file with several modes
(``--mode {pairwise,multiarm,ppi,simultaneous_ci,pairwise_multiarm,all}``):

Non-PPI path (ported from ``simulations/sim_compare_pvalues.py``)
------------------------------------------------------------------
Given LLM-only scores (no human labels), which raw p-value/rejection
procedure is best calibrated?

- ``pairwise``: Type-I error and power for pairwise A-vs-B comparisons,
  via ``scenarios.synthetic.build_pair_sources``'s ICC x Cohen's-d grid
  (the same paired-difference scenario library ``cases/ci_paired.py`` uses)
  as the null/weak-alt/full-alt conditions.
- ``multiarm``: family-wise false-positive rate and best-arm selection power
  across p-value correction strategies (holm/bonferroni/fdr_bh/hochberg/
  shaffer/friedman_nemenyi/max_t/romano_wolf/westfall_young), via
  ``scenarios.synthetic.build_multiarm_sources``, sweeping the SAME shape
  catalog ``build_pair_sources`` uses, generalized to k arms. hochberg/
  shaffer are closed-form refinements of holm (see
  ``evalstats.core.stats_utils.correct_pvalues``); romano_wolf/
  westfall_young are step-down max-T procedures (see
  ``_stepdown_max_t_pvalues``) that refine max_t's single-step joint
  critical value by recomputing the max only over not-yet-rejected pairs at
  each step -- all four exist to recover power lost to holm/bonferroni when
  pairwise comparisons are positively correlated, which repeated-measures/
  shared-item designs (the same participants or evaluation items
  contributing to multiple comparisons) produce routinely.
- ``simultaneous_ci``: family-wise CI coverage and average per-comparison
  width for the three simultaneous-CI constructions with a well-established
  dual to multiarm's p-value corrections -- ``none`` (naive per-pair CI, no
  simultaneous adjustment -- the "why do you need any correction?"
  baseline), Bonferroni t-intervals, and max-T (single-step studentized
  bootstrap, what ``evalstats.core.paired.all_pairwise`` uses by default)
  -- forced side by side on the SAME draw with the SAME point-estimate
  method held fixed (bypassing ``all_pairwise``'s own automatic
  method-based routing for max-T/Bonferroni), on the identical k-arm
  sources ``multiarm`` uses. multiarm's other corrections (holm/fdr_bh/
  hochberg/shaffer/friedman_nemenyi/romano_wolf/westfall_young) have no
  such CI dual -- holm/fdr_bh/hochberg/shaffer are p-value-only
  adjustments, friedman_nemenyi operates on rank differences rather than
  the raw mean-difference scale a CI needs, and romano_wolf/westfall_young's
  step-down critical value varies per rejection step rather than being one
  fixed value a CI could be built from. This is the evidence for why max-T
  is the harness's default simultaneous-CI method: it should hit nominal
  coverage same as Bonferroni (unlike ``none``, which should visibly
  under-cover as k grows), so a narrower average width at matching coverage
  is what actually distinguishes it from Bonferroni.

PPI-corrected path (ported from ``simulations/sim_type_i_calibration.py``)
---------------------------------------------------------------------------
- ``ppi``: given LLM scores plus sparse (possibly MNAR) human labels, does
  PPI correction (``evalstats.tests``' internal machinery) fix the Type-I
  inflation that judge bias/miscalibration causes in the uncorrected
  (scipy-equivalent) version? Sweeps judge-bias parameters via
  ``scenarios.synthetic.build_judge_bias_sources`` /
  ``generate_judge_bias_cell``, one factor at a time from a fixed baseline,
  layered on top of ONE representative shape per eval type from the SAME
  catalog the other two modes use. Includes a ``noise_family.*`` factor
  (``JudgeBiasSource.noise_family="contaminated"``) checking the same
  question under "judge mostly right, occasionally catastrophically wrong"
  measurement error instead of the default symmetric Gaussian -- same total
  noise variance either way, just redistributed.

There is no separate ``ppi_calibration`` case: it was folded in here instead,
since both halves answer "is this statistical decision trustworthy" at
different levels of the stack (raw CI/p-value procedures vs. high-level
``evalstats.tests`` wrappers with PPI), sharing report/plot/CLI scaffolding
AND, as of the shape-catalog unification, the underlying truth-generating
process too (``scenarios.synthetic.sample_group_truth`` -- see that
module's docstring and the harness README's "Shared scenario library"
section).

Known exceptions (see simulations/harness/README.md):
- ``ppi`` mode's one-factor-at-a-time sweep covers ``eval_type`` in
  ``{continuous, likert, grades}`` in full (one representative shape per
  eval type rather than the full catalog ``multiarm`` sweeps -- judge-bias/
  noise/label-fraction/etc. parameters are PPI's actual axis of interest,
  not distribution shape); ``binary`` is supported only for the
  two-independent-groups/paired mean-based tests (``ttest``/``ttest_welch``/
  ``paired_t``/``bayes_bootstrap`` -- a proportion is just the mean of a
  0/1 variable, so PPI's rectifier applies unchanged), as a single
  baseline-settings scenario rather than swept across every other factor.
  ``bayes_bootstrap`` PPI-corrects the same paired-mean estimand as
  ``paired_t`` but via Dirichlet-weighted (Bayesian) bootstrap resampling
  instead of ``evalstats.ppi.correct``'s classical one (see
  ``evalstats.tests._ppi_paired_bayes_bootstrap``) -- kept as a validated
  alternative, not a recommended default (real-data testing found it
  underperforms; ``paired_t`` is the reasonable default for binary p-values).
  ``bootstrap_t`` PPI-corrects the SAME paired-mean estimand via a
  studentized-bootstrap pivot (see
  ``evalstats.tests._ppi_paired_bootstrap_t``), generalizing
  ``evalstats.core.resampling.bootstrap_t_ci_1d``'s per-replicate SE to
  PPI's two-term variance -- numeric (continuous/likert/grades) ONLY, not
  extended to binary, since its value is specifically for resampling-based
  CI estimation on numeric data at N>=50 (``ci_paired.py``), not pairwise
  binary p-values. ``mj_floor`` is the mirror image -- binary ONLY, not
  numeric -- PPI-correcting ``evalstats.core.resampling.mj_floor_paired_ci``'s
  score interval (see ``evalstats.tests._ppi_paired_mj_floor``): its variance
  term ``(n10+n01)/n^2 - (n10-n01)^2/n^3`` is exactly
  ``Var(diffs, ddof=0) / n``, so it generalizes to PPI's two-term variance
  by substituting an effective n (``n_eff = Var(unlabeled diffs) /
  V_hat_PPI``) into the SAME Wilson-style shrinkage formula -- fully
  closed-form, no bootstrap needed, and reduces EXACTLY to the original
  (uncorrected) formula when the "labeled" subset is the full sample with
  no judge error. ``mcnemar`` is intentionally NOT PPI-corrected here: its
  distinguishing feature is an EXACT small-sample binomial test on
  discordant-pair counts, and a PPI-corrected numerator is generally
  non-integer, breaking that exactness -- left as future work pending a
  firmer statistical basis rather than shipping an ad-hoc adaptation. The
  rank-based family (``mwu``/``wilcoxon``/``friedman``/``kruskal``) and
  ANOVA/LMM remain continuous/likert/grades-only: they assume a scale that
  doesn't hold up under binary's massive ties, and the judge-bias noise
  model used for those structures doesn't have a binary-compatible variant
  (yet) -- see ``scenarios.synthetic``'s
  ``_jb_llm_binary``/``_jb_llm_repeated_binary``.
- None of the three modes numerically matches its legacy script anymore
  (a deliberate trade: cross-mode truth-distribution consistency over
  per-mode legacy-script parity) -- verification is by sanity check
  (Type-I ~ alpha, power increasing with effect size/n) for all three.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import math
import multiprocessing as _mp


class _MWUResult:
    """Shim for the mwu call sites: exposes ``.p_value`` and ``.estimate``.

    NOTE the scale: _ppi_two_sample's ``.estimate`` is theta-0.5, while
    _ppi_mannwhitney_corrected returns theta on the P(X>Y) scale. This shim
    subtracts 0.5 so downstream consumers (e.g. the estimator-comparison
    variance ratio) keep the convention they were written against.

    They now go through evalstats.tests._ppi_mannwhitney_corrected -- the SAME
    code path mannwhitney() uses -- instead of calling _ppi_two_sample
    directly. Previously the harness measured a different variance
    construction from the one evalstats ships, so no sweep ever exercised the
    shipped test. The helper skips only the alignment report (~515ms/call),
    which the sweeps discard anyway.
    """
    __slots__ = ("p_value", "estimate")

    def __init__(self, p, theta=None):
        self.p_value = p
        self.estimate = float("nan") if theta is None else theta - 0.5

import time as _time
import os
import pathlib
import re
import threading
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.paired import (
        pairwise_differences, all_pairwise, friedman_nemenyi,
        _bonferroni_simultaneous_cis, _simultaneous_cis_router, _max_stat_simultaneous_cis,
    _calibrated_joint_simultaneous_cis,
        _sidak_simultaneous_cis, _joint_bootstrap_scaled_simultaneous_cis,
    )
    from evalstats.core.stats_utils import correct_pvalues, rescaled_ci
    from evalstats.core.resampling import (
        bayes_bootstrap_means_1d, mj_floor_paired_ci_from_diffs, logit_t_ci_1d,
    )
    from evalstats.tests import (
        _ppi_two_sample,
        _ppi_mannwhitney_corrected,
        _ppi_two_sample_t_interval,
        _ppi_paired_arrays,
        _ppi_paired_bayes_bootstrap,
        _ppi_paired_bootstrap_t,
        _ppi_paired_mj_floor,
        _ppi_paired_bonett_price,
        _ppi_single_wilson,
        _ppi_single_bootstrap_t,
        _ppi_single_t_interval,
        _ppi_paired_t_interval,
        _ppi_single_logit_t,
        _ppi_paired_logit_t,
        _p_x_gt_y_midrank,
        paired_walsh_midrank_theta,
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _ppi_anova_independent_ci,
        _ppi_anova_repeated_ci,
        _ppi_friedman_ci,
        _anova_between_variance_from_groups,
        _repeated_condition_variance,
        _friedman_rank_variance,
        _ppi_kruskal_wallis_pairwise,
        _ppi_kruskal_wallis_pairwise_mnar_experimental,
        _kw_candidate_from_pairwise,
        _ppi_kruskal_wallis_influence,
        _kw_rowsum_from_pairwise,
        _ppi_lmm_p_value,
        _kw_pairwise_thetas,
        _mcnemar_p,
        _mcnemar_midp_p,
    )
    from evalstats.core.mixed_effects import _fit_lmm_general, _get_fe_vcov_sm

from ..latex_tables import (
    booktabs_table,
    coverage_cell,
    error_rate_cell,
    escape_latex,
    eval_type_label,
    mark_best_and_runnerup,
    report_eval_type_group,
    sort_groups,
)
from ..scenarios import CIPairSource, MultiArmSource, JudgeBiasSource, EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS

#: Eval types these modes sweep unless --eval-types says otherwise. "grades"
#: is deliberately excluded: it is continuous rescaled onto a [0, 100] span
#: (see scenarios/synthetic.py), so it adds no distinct regime, and the
#: official tests have never reported it. Leaving it in the default was
#: actively harmful for the pooled plots -- sidak/boot have no canonical CI
#: for grades and so never ran there, while none/bonferroni/max_t did, which
#: meant the two groups of curves were averaged over different eval-type
#: mixes and their widths were not comparable at all.
DEFAULT_EVAL_TYPES = ["binary", "continuous", "likert"]
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    build_pair_sources,
    build_multiarm_sources,
    build_judge_bias_sources,
    build_judge_bias_sources_binary,
    build_ppi_power_sources,
    build_ppi_power_reinforcing_sources,
    build_ppi_power_nobias_sources,
    build_ppi_power_nlab_grid_reinforcing_sources,
    build_ppi_power_nlab_grid_opposing_sources,
    build_ppi_comparison_label_frac_sources,
    build_ppi_nlab_grid_sources,
    build_ppi_factorial_sources,
    build_ppi_label_efficiency_sources,
    build_ppi_label_efficiency_sources_binary,
    build_ppi_nformula_sources,
    build_ppi_nformula_sources_binary,
    PPI_LABEL_EFF_NOISE_LEVELS,
    PPI_LABEL_EFF_NOISE_LEVELS_BINARY,
    PPI_LABEL_EFF_NOISE_FAMILIES,
    PPI_LABEL_EFF_EFFECT_FRAC,
    PPI_LABEL_EFF_EFFECT_FRACS,
    PPI_LABEL_EFF_N,
    PPI_NFORMULA_N_VALUES,
    PPI_NFORMULA_NLAB_VALUES,
    PPI_NFORMULA_EFFECT_FRACS,
    _ppi_power_baseline,
    _ppi_power_baseline_binary,
    _jb_effect_magnitude,
    _jb_effect_magnitude_binary,
    _JB_MIN_LAB,
    build_ppi_power_sources_binary,
    build_ppi_power_reinforcing_sources_binary,
    build_ppi_power_nobias_sources_binary,
    build_ppi_comparison_label_frac_sources_binary,
    build_ppi_nlab_grid_sources_binary,
    build_ppi_factorial_sources_binary,
    PPI_BINARY_BIAS_MAGNITUDES,
    PPI_BINARY_NOISE_BASELINE,
    PPI_BINARY_NOISE_LEVELS,
    PPI_COMPARISON_MODERATE_EFFECT_FRAC,
    PPI_FACTORIAL_EFFECT_FRACS,
    PPI_FACTORIAL_N_VALUES,
    PPI_FACTORIAL_NLAB_VALUES,
    PPI_FACTORIAL_NOISE_LEVELS,
    PPI_FACTORIAL_NOISE_LEVELS_FAST,
    PPI_ALIGNMENT_HUMAN_NOISE_LEVELS,
    generate_judge_bias_cell,
    measure_judge_alignment,
    measure_human_human_alignment,
    estimate_judge_bias_gold_null_values,
    JUDGE_BIAS_LMM_FACTORIAL_FACTORS,
)
from ..scenarios.real_data import (
    DEFAULT_INSPECT_CSV, PAIR_SOURCES as REAL_PAIR_SOURCES, build_real_pair_sources, build_real_multiarm_sources,
)
from ..methods import (
    METHODS_BY_NAME,
    PAIRWISE_PVALUE_METHODS,
    MCNEMAR,
    MCNEMAR_MIDP,
    BOOTSTRAP,
    BOOTSTRAP_T,
    BCA,
    BAYES_BOOTSTRAP,
    SMOOTH_BOOTSTRAP,
    PERMUTATION,
    SIGN_TEST,
    BAYES_BINARY,
    WILCOXON,
    PAIRED_T,
    MJ_FLOOR,
    MJ_FLOOR_FIXED_LAMBDA,
    PPI_BONETT_PRICE,
    MULTIARM_CORRECTION_METHODS,
    SIMULTANEOUS_CI_METHODS,
    CORR_SIDAK,
    CORR_BOOT_CAL,
    CORR_BOOT,
    CANONICAL_SIMULTANEOUS_CI_METHODS,
    CORR_NONE,
    PPI_TEST_METHODS,
    PPI_OFFICIAL_TEST_METHODS,
    PPI_WILSON,
    PPI_BOOTSTRAP_T_SINGLE,
    PPI_T_INTERVAL,
    PPI_LOGIT_T,
    PPI_T_INTERVAL_SINGLE,
    PPI_LOGIT_T_SINGLE,
    TTEST,
    TTEST_WELCH,
    MWU,
    ANOVA_IND,
    ANOVA_REP,
    FRIEDMAN,
    KRUSKAL,
    KRUSKAL_ROWSUM,
    KRUSKAL_ROWSUM_LABELED,
    KRUSKAL_TWOPART,
    KRUSKAL_EIGENGAP,
    KRUSKAL_INFLUENCE,
    KRUSKAL_INFLUENCE_FLOOR,
    KRUSKAL_MNAR_EXPERIMENTAL,
    LMM,
    LMM_FACTORIAL,
    LMM_RUNS,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "pvalues"

MODES = ["pairwise", "multiarm", "ppi", "simultaneous_ci", "pairwise_multiarm", "all"]
DATA_SOURCES = ["synthetic"] + REAL_PAIR_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]
ALPHA_DEFAULT = 0.05

_BINARY_ONLY_PVAL_METHODS = {BAYES_BINARY.name, MCNEMAR.name, MCNEMAR_MIDP.name}
# NOTE on binary paired data sign_test and permutation are not merely similar
# to mcnemar (exact) -- they ARE it. The sign test drops ties, and on 0/1 data
# the non-tied differences are exactly the discordant pairs, giving the same
# Binomial(m, 1/2) reference; the sign-flip permutation test has the same
# reference up to Monte Carlo error. They are kept in the sweep because they
# are genuinely distinct on continuous/Likert data.

# Multiarm analogue of SIMULTANEOUS_CI_PLOT_METHODS below: `none`'s FWER is
# so far above nominal alpha (no correction at all) that plotting it on the
# same linear axis as every other correction (which all cluster near alpha)
# squashes the comparison save_multiarm_fwer_vs_k_plot /
# save_multiarm_fwer_vs_n_plot exist to show; `none` is still in the
# printed/logged report tables and the CSV.
MULTIARM_PLOT_METHODS = [m for m in MULTIARM_CORRECTION_METHODS if m.name != CORR_NONE.name]

# Every simultaneous-CI construction _run_simultaneous_ci_cell can produce:
# `none`/`bonferroni`/`max_t` (see SIMULTANEOUS_CI_METHODS) plus `sidak`/
# `boot` (see CANONICAL_SIMULTANEOUS_CI_METHODS' comment in methods.py).
# `none`/`bonferroni`/`sidak`/`boot` are all built on the scenario's
# eval-type-canonical CI method (Tango for binary, Logit-t for continuous/
# likert -- see _canonical_ci_func below), NOT
# --multiarm-method; `max_t` is the one exception, since it needs a
# bootstrap-compatible method to resample from and keeps using
# --multiarm-method (bootstrap_t by default) regardless of eval type. Report/
# plot functions filter this down to whichever names are actually present in
# a given results list, so a `grades`-only sweep (no canonical CI wired up --
# see _canonical_ci_func) simply never shows `sidak`/`boot`.
ALL_SIMULTANEOUS_CI_METHODS = SIMULTANEOUS_CI_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS

# `none` stays in _run_simultaneous_ci_cell's data collection and in
# print_simultaneous_ci_report's tables (it's the "why do you need any
# correction at all" baseline there), but is dropped from every
# simultaneous_ci *plot* -- it's so far below nominal family-wise coverage
# (even built on the canonical per-pair CI, which is well-calibrated
# per-comparison but never adjusted for multiplicity) that it squashes the
# Bonferroni-vs-max-T-vs-Sidak-vs-bootstrap comparison those plots exist to
# show.
SIMULTANEOUS_CI_PLOT_METHODS = [m for m in ALL_SIMULTANEOUS_CI_METHODS if m.name != CORR_NONE.name]


class _ProgressReporter:
    """One bar per phase. A check that runs several phases can declare how many
    up front (phase_plan), and each bar then says which phase it is and roughly
    how far through the whole check it is.

    Without that, a bar reading "24%" is 24% of one phase, and a check like
    run_ppi_label_efficiency_check -- which runs one phase per (eval type,
    noise family) -- looks nearly done when it has barely started. The overall
    figure assumes phases cost the same, which they do not exactly, so it is
    printed as an approximation ("~") and the per-phase bar stays the precise
    one.
    """

    _phase_i = 0
    _phase_n = 0

    @classmethod
    def phase_plan(cls, n_phases: int, *, resume: bool = False) -> None:
        """Declare how many phases the current check will run.

        resume=True keeps an already-declared plan of the same size counting
        instead of restarting it, for a phase loop that itself sits inside an
        outer loop -- the label-efficiency check runs its (eval type x noise
        family) groups once per effect-size arm, and without this each arm
        restarted the counter at 1, which is the same "this bar is the whole
        run" misreading the phase counter exists to prevent.
        """
        n = max(int(n_phases), 0)
        if resume and cls._phase_n == n and cls._phase_i:
            return
        cls._phase_n = n
        cls._phase_i = 0

    @classmethod
    def clear_phase_plan(cls) -> None:
        cls._phase_n = 0
        cls._phase_i = 0

    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0
        cls = type(self)
        if cls._phase_n:
            cls._phase_i += 1
            self.phase = (min(cls._phase_i, cls._phase_n), cls._phase_n)
        else:
            self.phase = None

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
        overall = ""
        if self.phase:
            i, n = self.phase
            prefix = f"{self.label} [phase {i}/{n}]: " if self.label else f"[phase {i}/{n}] "
            overall = f"  ~{100.0 * ((i - 1) + frac) / n:5.1f}% overall"
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}{overall}  {detail[:32]:<32s}",
            end="", flush=True,
        )
        if is_final:
            print()


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


# ---------------------------------------------------------------------------
# Pairwise mode (non-PPI): Type-I error and power across raw p-value
# procedures, ported from sim_compare_pvalues.py's pairwise phase. Reuses
# scenarios.synthetic.build_pair_sources -- its is_null rows are the "null"
# condition, its cohens_d_values sweep is the alt-condition power curve.
# Real-data sources (build_real_pair_sources) have no synthetic Cohen's d;
# their non-null condition is labeled "real" instead (see _run_pairwise_cell).
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    """One (eval_type, source, n, method) cell's Type-I/power outcome from
    the non-PPI pairwise sweep."""

    eval_type: str
    label: str
    n: int
    method: str
    condition: str  # "null" | f"d={cohens_d:.2f}" | "real" (real-data non-null)
    n_reps: int
    rejects: int
    p_sum: float
    cohens_d: float = 0.0


def _scenario_values(rows, numer, denom=lambda r: r.n_reps) -> list[float]:
    """Collapse `rows` to one value per scenario -- sum(numer)/sum(denom)
    within each (eval_type, label) -- so the bands treat the scenario as the
    unit of replication, which is what it is. Pooling every rep into one
    Bernoulli sample instead answers a much narrower question: how precisely
    THIS suite's average is pinned down, not how the method behaves.
    """
    acc: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        a = acc[(r.eval_type, r.label)]
        a[0] += numer(r)
        a[1] += denom(r)
    return [n / d for n, d in acc.values() if d > 0]


#: Which uncertainty band the line plots draw around each curve.
#:   "spread" -- 10th-90th percentile across scenarios (default)
#:   "ci"     -- 95% CI on the across-scenario mean
#:   "both"   -- spread outside, CI inside
#: One band by default: with a dozen methods on a panel, two translucent
#: fills per method stack into an unreadable wash.
#:
#: "ci" is the default the paper figures use. With 4-10 methods per panel the
#: percentile spread overlaps into mud, and the conditional detail it was
#: compensating for is already carried by the tables' per-n/per-k columns and
#: by the reliability violins. The CI band still widens honestly where
#: scenarios disagree -- it is a scenario-level standard error, not a per-rep
#: Monte Carlo one -- so a method that is unreliable at small n still shows a
#: visibly uncertain mean. Switch to "spread" when the distribution itself is
#: the point and no violin accompanies the figure.
BAND_STYLE = "ci"


def _scenario_bands(ax, xs, ys, per_scenario, *, color, z: float = 1.96,
                    style: str | None = None) -> list[float]:
    """Draw two bands around a curve of across-scenario averages.

    Inner (darker): a 95% CI on the mean, ``+- z * sd / sqrt(n_scenarios)``,
    with the scenario as the unit. It is inferential -- where the average
    plausibly sits -- and widens exactly where scenarios disagree, so a
    method that is unreliable at small n gets a visibly uncertain mean
    instead of the falsely-tight interval a per-rep Monte Carlo error gives.
    Centred on the plotted point rather than on the scenario mean: the two
    coincide under a balanced suite, and pinning the band to the drawn line
    avoids a visibly off-centre band that reads as a bug when they don't.

    Outer (lighter): the 10th-90th percentile of the scenarios themselves.
    This is descriptive, not inferential -- it makes no claim that the suite
    is a random sample of anything, which matters because the suite is
    purposively built to span regimes. It also does not shrink as reps or
    scenarios accumulate, so it cannot lull a reader into reading a tight
    mean as a consistent method. Percentiles rather than +-sd because these
    quantities are bounded (coverage at 1.0, rates at 0) and skew hard
    against the bound, where an sd band would run outside the range.

    Returns the finite band endpoints so callers can fit axis limits.
    """
    inner_lo, inner_hi, outer_lo, outer_hi = [], [], [], []
    for y, vals in zip(ys, per_scenario):
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2 or not np.isfinite(y):
            for acc in (inner_lo, inner_hi, outer_lo, outer_hi):
                acc.append(float("nan"))
            continue
        half = z * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
        inner_lo.append(y - half)
        inner_hi.append(y + half)
        outer_lo.append(float(np.percentile(vals, 10)))
        outer_hi.append(float(np.percentile(vals, 90)))
    style = style or BAND_STYLE
    shown: list[float] = []
    if style in ("spread", "both"):
        ax.fill_between(xs, outer_lo, outer_hi, color=color,
                        alpha=0.10 if style == "both" else 0.16,
                        linewidth=0, zorder=1)
        shown += outer_lo + outer_hi
    if style in ("ci", "both"):
        ax.fill_between(xs, inner_lo, inner_hi, color=color, alpha=0.22,
                        linewidth=0, zorder=2)
        shown += inner_lo + inner_hi
    return [v for v in shown if np.isfinite(v)]


def _width_scale(eval_type: str) -> float:
    """Span of `eval_type`'s natural outcome scale, for turning an absolute
    CI width into a fraction of that scale.

    A width of 1.2 means something completely different on Likert (a 1-5
    scale, so ~30% of the range) than on binary (0-1, so wider than the
    entire range). Any plot that pools eval types onto one width axis has to
    divide it out first, or the largest-scale type simply dominates the
    average. Uses the same EVAL_TYPE_SCALE_BOUNDS the simulation already
    applies to rescale data onto [0, 1] before calling CI methods, so the
    normalization matches what the estimators themselves see.
    """
    lo, hi = EVAL_TYPE_SCALE_BOUNDS.get(eval_type, (0.0, 1.0))
    span = hi - lo
    return span if span > 0 else 1.0


def _safe_wilcoxon_p(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value via scipy's default method="auto".

    Deliberately does NOT override method= for speed. scipy's "auto" does
    genuinely different (not just slower) work for small, tied/discrete-
    valued samples (binary 0/1 diffs, Likert-scale integer diffs, etc.) at
    roughly n<=13: it runs exhaustive permutation enumeration for a
    rigorously tie-corrected exact p-value, which is legitimately expensive
    (up to ~300ms/call at n=13 in scipy 1.17.x) but is the CORRECT p-value.
    Forcing method="exact" instead is much faster, but per scipy's own docs
    "method='exact' no longer calculates the exact p-value" once ties/zeros
    are present -- empirically this shifted individual p-values by up to
    0.125 and measurably changed small-n FWER calibration in this harness's
    own null-hypothesis checks, not just decision-level noise that would
    wash out. Given every pair/rep goes through this function as evalstats'
    canonical raw p-value (see _compute_multiarm_metrics), correctness at
    small n matters more than the wall-clock cost -- eat the cost rather
    than risk silently drifting already-reported simulation results.
    """
    if int(np.sum(diffs != 0)) < 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            w = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        p = float(w.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _safe_paired_t_p(diffs: np.ndarray) -> float:
    if len(diffs) <= 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            t = scipy_stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        p = float(t.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _pairwise_pvalue(a: np.ndarray, b: np.ndarray, method: str, n_bootstrap: int, rng: np.random.Generator, statistic: str) -> float:
    """Compute one method's p-value for paired A-vs-B data of shape (n, runs)."""
    diffs = a.mean(axis=1) - b.mean(axis=1)

    if method == WILCOXON.name:
        return _safe_wilcoxon_p(diffs)
    if method == PAIRED_T.name:
        return _safe_paired_t_p(diffs)

    if method in _BINARY_ONLY_PVAL_METHODS:
        aa = (a.mean(axis=1) >= 0.5).astype(float)
        bb = (b.mean(axis=1) >= 0.5).astype(float)
        if method == MCNEMAR.name:
            return _mcnemar_p(aa, bb)
        if method == MCNEMAR_MIDP.name:
            return _mcnemar_midp_p(aa, bb)
        scores = np.stack([aa, bb], axis=0)
    else:
        scores = np.stack([a[:, 0], b[:, 0]], axis=0) if a.shape[1] == 1 else np.stack([a, b], axis=0)

    result = pairwise_differences(
        scores=scores, idx_a=0, idx_b=1, label_a="A", label_b="B",
        method=method, ci=0.95, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
    )
    p = float(result.p_value)
    return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0


def _pairwise_methods_allowed(eval_type: str) -> list:
    return [m for m in PAIRWISE_PVALUE_METHODS if m.name not in _BINARY_ONLY_PVAL_METHODS or eval_type == "binary"]


def _run_pairwise_cell(
    source: CIPairSource, n: int, runs: int, n_reps: int, n_bootstrap: int, alpha: float, statistic: str, seed,
) -> list[PairwiseResult]:
    """Run n_reps replications of a paired 2-group test at one (source, n)
    cell, across every method allowed for the source's eval type. One
    PairwiseResult per method."""
    methods = _pairwise_methods_allowed(source.eval_type)
    if source.is_null:
        condition = "null"
    elif source.source != "synthetic":
        # Real A-vs-B pairs have a genuine (usually nonzero) true_diff, but no
        # synthetic Cohen's d -- label the power column "real" rather than the
        # misleading "d=0.00" (CIPairSource.cohens_d's default for real data).
        condition = "real"
    else:
        condition = f"d={source.cohens_d:.2f}"

    ss = np.random.SeedSequence(seed)
    data_rng = np.random.default_rng(ss.spawn(1)[0])
    method_rngs = {m.name: np.random.default_rng(s) for m, s in zip(methods, ss.spawn(len(methods)))}

    rejects: dict[str, int] = {m.name: 0 for m in methods}
    p_sums: dict[str, float] = {m.name: 0.0 for m in methods}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Real paired binary data frequently has many items where both
        # models score identically (diffs == 0), which triggers scipy's
        # nan_policy="omit" wrapper (_axis_nan_policy.py) to warn about
        # catastrophic cancellation in its internal moment calculation --
        # benign here (the p-value is still valid), same as the
        # friedmanchisquare/kruskal RuntimeWarning suppression below.
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_reps):
            a, b = source.generate_pair(data_rng, n, runs)
            for m in methods:
                p = _pairwise_pvalue(a, b, method=m.name, n_bootstrap=n_bootstrap, rng=method_rngs[m.name], statistic=statistic)
                p_sums[m.name] += p
                if p <= alpha:
                    rejects[m.name] += 1

    return [
        PairwiseResult(
            eval_type=source.eval_type, label=source.label, n=n, method=m.name, condition=condition,
            n_reps=n_reps, rejects=rejects[m.name], p_sum=p_sums[m.name], cohens_d=source.cohens_d,
        )
        for m in methods
    ]


_PAIRWISE_SOURCES: list = []  # fork-inherited worker state for run_pairwise_simulation
_MULTIARM_SOURCES: list = []  # fork-inherited worker state for run_multiarm_simulation


def _run_pairwise_cell_worker(args: tuple) -> list[PairwiseResult]:
    sc_idx, n, runs, n_reps, n_bootstrap, alpha, statistic, seed = args
    return _run_pairwise_cell(_PAIRWISE_SOURCES[sc_idx], n, runs, n_reps, n_bootstrap, alpha, statistic, seed)


def _run_multiarm_cell_worker(args: tuple) -> list[MultiArmResult]:
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, corrections = args
    return _run_multiarm_cell(
        _MULTIARM_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed,
        corrections=corrections,
    )


def _run_ppi_cell_worker(args: tuple) -> list:
    sc, active_tests, n_reps, n_boot, seed, progress_dict = args
    return _run_ppi_cell(sc, active_tests, n_reps, n_boot, seed, progress_dict=progress_dict, progress_key=sc.name)


def _run_ppi_effect_cell_worker(args: tuple) -> tuple:
    sc_idx, sc, active_tests, n_reps, n_boot, seed = args
    return (sc_idx, _run_ppi_effect_cell(sc, active_tests, n_reps, n_boot, seed))


def run_pairwise_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    alpha: float, statistic: str, progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[PairwiseResult]:
    """Sweep _run_pairwise_cell over every (source, sample size) cell,
    parallelized across n_workers, and flatten the per-cell PairwiseResult
    lists into one list."""
    global _PAIRWISE_SOURCES
    _PAIRWISE_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, n_reps, n_bootstrap, alpha, statistic, seed)
                 for (sc_idx, n), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-pairwise")
    results: list[PairwiseResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_pairwise_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_pairwise_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def print_pairwise_report(results: list[PairwiseResult], alpha: float) -> None:
    """Print the console Type-I-error/power report for a pairwise run,
    grouped by eval type and method."""
    _, _bradley_hi = bradley_bounds(alpha)
    print(f"\n{'='*78}\n  PVALUES (PAIRWISE, NON-PPI) -- TYPE I ERROR + POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]

    for et in eval_types_present:
        et_rows = [r for r in results if r.eval_type == et]
        # Per-eval-type, not global: some eval types (e.g. "binary", via
        # build_pair_sources' hand-picked asymmetric scenarios) have a
        # non-null condition literally labeled "d=0.00" alongside the real
        # "null" rows. Computing this list globally would leak that column
        # into every other eval type's table too, where no such non-null
        # d=0.00 rows exist -- showing a spurious all-nan "power(d=0.00)"
        # column instead of just omitting it.
        et_conditions = sorted({r.condition for r in et_rows if r.condition != "null"})
        print(f"\n  [{et}]")
        hdr = f"    {'Method':<14} {'typeI':>8}" + "".join(f"  power({c})".rjust(14) for c in et_conditions)
        print(hdr)
        for m in method_labels:
            m_rows = [r for r in et_rows if r.method == m]
            if not m_rows:
                continue
            null_rows = [r for r in m_rows if r.condition == "null"]
            c_tot = sum(r.rejects for r in null_rows)
            t_tot = sum(r.n_reps for r in null_rows)
            type1 = c_tot / t_tot if t_tot > 0 else float("nan")
            row = f"    {m:<14} {type1:>8.3f}"
            for c in et_conditions:
                c_rows = [r for r in m_rows if r.condition == c]
                cr = sum(r.rejects for r in c_rows)
                ct = sum(r.n_reps for r in c_rows)
                pw = cr / ct if ct > 0 else float("nan")
                row += f"  {pw:>12.3f}"
            print(row)

    conditions = sorted({r.condition for r in results if r.condition != "null"})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    n_cols = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    print(f"\n{'-'*72}\n  OVERALL SUMMARY (collapsed across eval types, sources, n)\n{'-'*72}")
    print(f"  MaxT1 = worst per-scenario Type-I error seen for that method (not an average) --\n"
          f"  flags methods whose good mean Type-I error hides an inflated scenario/n cell.")
    print(f"\n  {'Method':<20}  {'TypeI':>6}  {'MaxT1':>7}  {'Band95':>13}  {'MeanPow':>8}{n_cols}")
    for m in method_labels:
        m_rows = [r for r in results if r.method == m]
        if not m_rows:
            continue
        null_rows = [r for r in m_rows if r.condition == "null"]
        c_tot = sum(r.rejects for r in null_rows)
        t_tot = sum(r.n_reps for r in null_rows)
        type1 = c_tot / t_tot if t_tot > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        power_cells = []
        for c in conditions:
            c_rows = [r for r in m_rows if r.condition == c]
            cr = sum(r.rejects for r in c_rows)
            ct = sum(r.n_reps for r in c_rows)
            power_cells.append(cr / ct if ct > 0 else float("nan"))
        mean_power = float(np.mean([p for p in power_cells if np.isfinite(p)])) if power_cells else float("nan")
        marker = "*" if np.isfinite(type1) and type1 > _bradley_hi else " "
        per_label_t1 = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_t1[(r.eval_type, r.label)]
            acc[0] += r.rejects
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_t1.values() if t > 0]
        worst_t1 = max(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_t1:.3f}{'*' if np.isfinite(worst_t1) and worst_t1 > _bradley_hi else ' '}" if np.isfinite(worst_t1) else "-"
        n_type1 = ""
        for n in sizes_present:
            n_rows = [r for r in null_rows if r.n == n]
            c_n = sum(r.rejects for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            t1_n = c_n / t_n if t_n > 0 else float("nan")
            n_type1 += f"  {t1_n:>7.3f}" if np.isfinite(t1_n) else f"  {'  -':>7}"
        print(f"  {m:<20}  {type1:>5.3f}{marker}  {worst_str:>7}  {band:>13}  {mean_power:>8.3f}{n_type1}")
    print(f"  (* = TypeI above Bradley's liberal band, i.e. > 1.5*alpha = {_bradley_hi:.3f})")
    print()


def bradley_bounds(alpha: float) -> tuple[float, float]:
    """Bradley's (1978) "liberal" robustness criterion: a test counts as
    holding its nominal level when its empirical Type-I error / FWER falls
    within [0.5*alpha, 1.5*alpha] -- [0.025, 0.075] at the usual alpha=0.05.

    Used as the single definition of "acceptably calibrated" across this
    module's plain-text reports, plots, and LaTeX tables, so all three views
    of one run agree. It replaces an ad-hoc `alpha +- 0.02` band: numerically
    near-identical at alpha=0.05, but citable, and it scales with alpha
    instead of staying a fixed width that would be absurdly permissive at
    alpha=0.001 and impossibly strict at alpha=0.20.

    Bradley, J.V. (1978). Robustness? British Journal of Mathematical and
    Statistical Psychology, 31(2), 144-152.

    Rounded to kill binary-representation noise: `1.5 * 0.05` is
    0.07500000000000001, so an empirical rate of exactly 0.075 would land
    inside or outside the band depending on which side of that artifact it
    fell -- an arbitrary distinction at a threshold readers will check by
    hand.
    """
    return round(0.5 * alpha, 12), round(1.5 * alpha, 12)


def _power_ranking_values(
    powers: list[float], error_rates: list[float], alpha: float
) -> list[float]:
    """Blank out (as NaN) the power of any method that doesn't control its
    error rate, so `mark_best_and_runnerup` skips it.

    Power is only comparable between tests that hold their nominal level: an
    uncorrected procedure sitting at FWER 0.22 will "win" any power contest
    simply by rejecting more often, and bolding it in a paper table reads as
    an endorsement. Excluded rows still print their power -- they're just
    not eligible to be marked best.

    Only the UPPER half of `bradley_bounds` gates here. An anti-conservative
    test wins power by cheating, so it's disqualified; an over-conservative
    one is handicapped instead, and if it still takes the top power that is
    a real result worth marking rather than an artifact worth hiding.
    """
    _, upper = bradley_bounds(alpha)
    return [
        p if (np.isfinite(t1) and t1 <= upper) else float("nan")
        for p, t1 in zip(powers, error_rates)
    ]


def latex_pairwise_overall_summary(results: list[PairwiseResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-method Type-I error (with its 95%
    MC band) + mean power, plus one Type-I column per sample size actually
    swept, appended to the right -- the aggregate Type-I column collapses
    across n and can hide miscalibration that only shows up at small or
    large sample sizes.

    Methods that ran on more than one eval type get one row per type --
    "<method> (bin)"/"(cont)"/"(lik)" -- computed from only that type's own
    data, with rows grouped into midrule-separated blocks. This matches
    ci_single/ci_paired's layout so the whole paper reads one convention,
    and it stops a pooled row from hiding a type-specific miscalibration:
    a method can hold its nominal level on continuous data while running
    badly inflated on Likert, and a single averaged Type-I number reports
    neither. Power is ranked within a block, never across.
    """
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    conditions = sorted({r.condition for r in results if r.condition != "null"})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})

    method_groups: dict[str, set[str]] = defaultdict(set)
    for r in results:
        method_groups[r.method].add(report_eval_type_group(r.eval_type))
    groups_present = sort_groups({g for gs in method_groups.values() for g in gs})

    rows = []
    rule_before = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        powers, type1s = [], []
        for m in method_labels:
            if g not in method_groups[m]:
                continue
            m_rows = [r for r in results
                      if r.method == m and report_eval_type_group(r.eval_type) == g]
            null_rows = [r for r in m_rows if r.condition == "null"]
            c_tot = sum(r.rejects for r in null_rows)
            t_tot = sum(r.n_reps for r in null_rows)
            type1 = c_tot / t_tot if t_tot > 0 else float("nan")
            _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
            power_cells = []
            for c in conditions:
                c_rows = [r for r in m_rows if r.condition == c]
                cr = sum(r.rejects for r in c_rows)
                ct = sum(r.n_reps for r in c_rows)
                power_cells.append(cr / ct if ct > 0 else float("nan"))
            mean_power = float(np.mean([p for p in power_cells if np.isfinite(p)])) if power_cells else float("nan")
            label = f"{escape_latex(m)} ({g})" if len(method_groups[m]) > 1 else escape_latex(m)
            row = [
                label,
                error_rate_cell(type1, alpha),
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{mean_power:.3f}" if np.isfinite(mean_power) else "-",
                g,
            ]
            for n in sizes_present:
                n_rows = [r for r in null_rows if r.n == n]
                c_n = sum(r.rejects for r in n_rows)
                t_n = sum(r.n_reps for r in n_rows)
                type1_n = c_n / t_n if t_n > 0 else float("nan")
                row.append(error_rate_cell(type1_n, alpha))
            rows.append(row)
            powers.append(mean_power)
            type1s.append(type1)

        # Power is this table's "more is better, no nominal target" column,
        # the role Score plays in the CI tables, so it gets the best/
        # runner-up marks. Type-I error has a target and gets shading
        # instead -- bolding the lowest Type-I would reward the most
        # conservative method, not the best.
        POWER_COL = 3
        block = rows[block_start:]
        marked = mark_best_and_runnerup(
            [r[POWER_COL] for r in block],
            _power_ranking_values(powers, type1s, alpha),
            higher_is_better=True,
        )
        for row, cell in zip(block, marked):
            row[POWER_COL] = cell

    return booktabs_table(
        caption=f"pvalues (pairwise, non-PPI): Type-I error and mean power across conditions (nominal alpha={alpha}). "
                f"Methods tested on more than one eval type are reported as one row per type (bin/cont/lik), "
                f"grouped into blocks, so no row averages across eval types. "
                f"Type-I cells shade red when inflated above {alpha} and blue when conservative below it, "
                f"on the same scale as the coverage tables; best and runner-up mean power are bold and "
                f"underlined within each block, among methods holding their nominal level.",
        label="tab:pvalues_pairwise_overall",
        columns=["Method", "Type-I error", "95\\% MC band", "Mean power", "Type"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
        rule_before=rule_before,
    )


def save_results_artifacts_pairwise(*, results: list[PairwiseResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    """Write the pairwise run's results CSV (and LaTeX summary if
    `latex=True`) under out_dir. Returns the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_pairwise_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "method", "condition", "n_reps", "rejects", "reject_rate", "mean_p", "cohens_d"])
        for r in results:
            writer.writerow([
                r.eval_type, r.label, r.n, r.method, r.condition, r.n_reps, r.rejects,
                f"{r.rejects / r.n_reps:.8f}", f"{r.p_sum / r.n_reps:.8f}", f"{r.cohens_d:.6f}",
            ])
    summary_path = out_base / f"{run_stem}_pairwise_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_pairwise_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_pairwise_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def _save_pairwise_typeI_power_plot_one(
    *, results: list[PairwiseResult], alpha: float, out_path: str, eval_type: str,
) -> str:
    """Save a Type-I error (left) + power (right) plot for a single eval_type."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _ticker

    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    et_rows = [r for r in results if r.eval_type == eval_type]
    sample_sizes = sorted({r.n for r in results})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.0, 4.2), squeeze=False)
    ax_t1, ax_pw = axes[0][0], axes[0][1]
    ax_t1.axhline(alpha, color="black", linewidth=1.0, linestyle="--")
    ax_t1.axhspan(*bradley_bounds(alpha), color="#DDDDDD", alpha=0.4, zorder=0)

    for m in method_objs:
        m_rows = [r for r in et_rows if r.method == m.name]
        if not m_rows:
            continue
        null_rows = [r for r in m_rows if r.condition == "null"]
        xs, ys, scen = [], [], []
        for n in sample_sizes:
            subset = [r for r in null_rows if r.n == n]
            if not subset:
                continue
            c = sum(r.rejects for r in subset)
            t = sum(r.n_reps for r in subset)
            xs.append(n)
            ys.append(c / t if t > 0 else float("nan"))
            scen.append(_scenario_values(subset, lambda r: r.rejects))
        if xs:
            ax_t1.plot(xs, ys, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)
            _scenario_bands(ax_t1, xs, ys, scen, color=m.color)

        alt_rows = [r for r in m_rows if r.condition != "null"]
        xs2, ys2, scen2 = [], [], []
        for n in sample_sizes:
            subset = [r for r in alt_rows if r.n == n]
            if not subset:
                continue
            c = sum(r.rejects for r in subset)
            t = sum(r.n_reps for r in subset)
            xs2.append(n)
            ys2.append(c / t if t > 0 else float("nan"))
            scen2.append(_scenario_values(subset, lambda r: r.rejects))
        if xs2:
            ax_pw.plot(xs2, ys2, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)
            _scenario_bands(ax_pw, xs2, ys2, scen2, color=m.color)

    ax_t1.set_title(f"{eval_type}: Type-I error")
    ax_t1.set_xlabel("n")
    ax_t1.set_ylabel("Rejection rate (null)")
    ax_t1.set_xscale("log")
    ax_pw.set_title(f"{eval_type}: power (mean over alt conditions)")
    ax_pw.set_xlabel("n")
    ax_pw.set_ylabel("Rejection rate (alt)")
    ax_pw.set_xscale("log")
    # Legend outside, to the right of the rightmost panel. In-axes it sat on
    # top of the curves it was labelling -- with a dozen methods there is no
    # empty corner to put it in, and the Type-I panel's interesting region
    # (the inflated methods above alpha) is exactly where "upper right" lands.
    # bbox_inches="tight" at savefig grows the canvas to include it.
    _handles, _labels = ax_t1.get_legend_handles_labels()
    ax_pw.legend(_handles, _labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                 borderaxespad=0.0, fontsize=7)
    _loc = _ticker.FixedLocator(sample_sizes)
    _fmt = _ticker.FuncFormatter(lambda x, _: str(int(x)))
    _nul = _ticker.NullLocator()
    for _ax in (ax_t1, ax_pw):
        _ax.xaxis.set_major_locator(_loc)
        _ax.xaxis.set_major_formatter(_fmt)
        _ax.xaxis.set_minor_locator(_nul)
    # Ensure y=0 lines are visible even when all methods have zero Type-I error
    # (binary null under the shared-item model gives d_i=0 for all i, so all
    # tests correctly return p=1 -- FWER=0 is right, but visually looks blank).
    t1_lo, t1_hi = ax_t1.get_ylim()
    if t1_hi - t1_lo < 0.04:
        ax_t1.set_ylim(-0.005, max(t1_hi, alpha + 0.04))
    elif t1_lo > -0.003:
        ax_t1.set_ylim(-0.003, t1_hi)
    if eval_type == "binary":
        null_t1_vals = [
            r.rejects / r.n_reps
            for r in [r for r in et_rows if r.condition == "null"]
            if r.n_reps > 0
        ]
        if null_t1_vals and max(null_t1_vals) == 0.0:
            ax_t1.text(0.5, 0.25, "T1=0 (shared-item model:\nA≡B under null)", transform=ax_t1.transAxes,
                       ha="center", va="center", fontsize=7.5, color="#555555", style="italic")

    fig.suptitle(f"pvalues (pairwise, non-PPI): Type-I + Power [{eval_type}] | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pairwise_typeI_power_plot(*, results: list[PairwiseResult], alpha: float, out_path: str) -> list[str]:
    """Save one Type-I error + power plot per eval_type present in results.

    ``out_path`` is used as the base path; ``_{eval_type}`` is inserted before
    the file extension for each saved file.  Returns all saved paths.
    """
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    if not eval_types_present:
        return []
    base = Path(out_path)
    stem, suffix = base.stem, base.suffix or ".png"
    saved: list[str] = []
    for et in eval_types_present:
        et_path = str(base.parent / f"{stem}_{et}{suffix}")
        _save_pairwise_typeI_power_plot_one(results=results, alpha=alpha, out_path=et_path, eval_type=et)
        saved.append(et_path)
    return saved


def save_pairwise_reliability_violin_plot(*, results: list[PairwiseResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario Type-I error and
    power, one dot per (label, method) -- the pairwise-testing analogue of
    ci_single/ci_paired's reliability violin. Exposes the spread the OVERALL
    SUMMARY table's pooled Type-I error hides: a method with alpha-level Type-I
    error on average can still have scenario-specific inflation that pooling
    across labels masks."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]
    palette = {m.name: m.color for m in method_objs}

    null_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "typeI": r.rejects / r.n_reps}
        for r in results if r.condition == "null" and r.n_reps > 0 and r.method in method_names
    ])
    alt_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "power": r.rejects / r.n_reps}
        for r in results if r.condition != "null" and r.n_reps > 0 and r.method in method_names
    ])
    null_scenario = (
        null_df.groupby(["eval_type", "label", "method"], as_index=False).agg(typeI=("typeI", "mean"))
        if not null_df.empty else null_df
    )
    alt_scenario = (
        alt_df.groupby(["eval_type", "label", "method"], as_index=False).agg(power=("power", "mean"))
        if not alt_df.empty else alt_df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        for row_idx, (scenario_df, metric, ylabel, ref_line) in enumerate([
            (null_scenario, "typeI", "Type-I error per scenario", alpha),
            (alt_scenario, "power", "Power per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            et_df = scenario_df[scenario_df["eval_type"] == et] if not scenario_df.empty else scenario_df
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_methods = [name for name in method_names if name in et_df["method"].values]
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
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\npvalues pairwise | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Multi-arm mode (non-PPI): family-wise error rate and best-arm selection
# power across p-value correction strategies, ported from
# sim_compare_pvalues.py's multi-arm phase.
# ---------------------------------------------------------------------------


@dataclass
class MultiArmResult:
    """One (eval_type, source, n, k, correction) cell's FWER/best-arm-power
    outcome from the non-PPI multi-arm sweep."""

    eval_type: str
    label: str
    n: int
    k: int
    correction: str
    condition: str  # "null" | "alt"
    n_reps: int
    any_reject: int
    best_selected: int
    total_time: float = 0.0
    """Total wall-clock seconds for THIS correction's own computation, summed
    across all n_reps of this condition -- e.g. romano_wolf/westfall_young's
    own step-down resampling, max_t's own router call, etc. Does NOT include
    shared per-rep setup (score generation, Wilcoxon p-values) except for
    `none`, which that setup is attributed to (see _run_multiarm_cell /
    _compute_multiarm_metrics)."""


def _bootstrap_t_matrix(
    diffs_mat: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
    resample_mode: str, batch_size: int = 256,
    arm_scores: np.ndarray | None = None, pair_indices: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint resampling core shared by _stepdown_max_t_pvalues (Romano-Wolf/
    Westfall-Young step-down) and max_t's single-step critical value.

    Draws n_bootstrap joint resamples of the studentized per-pair statistic
    ``t_p = mean(diffs_p) / se(diffs_p)`` and returns ``(t_abs, t_obs)``:
    ``t_abs`` is ``(k_pairs, n_bootstrap)`` -- ``|T|`` per pair per replicate
    -- and ``t_obs`` is ``(k_pairs,)`` -- the observed ``|T|``. Two
    resample_mode options generate the joint null distribution differently:

    - ``"bootstrap"`` (Romano-Wolf / max_t): resample items/participants
      (rows of ``diffs_mat``, shared across pairs) with replacement, then
      studentize and recenter each bootstrap draw at its own pair's
      *observed* statistic -- the same nonparametric bootstrap-t null
      evalstats.core.paired's ``_max_stat_simultaneous_cis`` uses for
      max_t's single-step critical value (bootstrap_t branch): identical
      formula, so max_t's critical value is exactly ``quantile(t_abs.max(
      axis=0), ci)`` -- i.e. step 0 of the step-down procedure, before any
      rejections -- computed from this SAME matrix. See
      _compute_multiarm_metrics for where that sharing happens.
    - ``"permutation"`` (Westfall-Young): independently, per item/
      participant, draw a uniformly random relabeling of the ``k`` arms
      (``arm_scores``' rows) and recompute every pair's diff from the
      relabeled scores for that item -- the genuine label-permutation null,
      not merely a sign flip of the diffs. This is exact under
      exchangeability of the k arms for any number of arms: unlike sign-
      flipping the diff vector (which only coincides with a real relabeling
      when k=2, since the group of relabelings has k! elements but sign
      flips only give 2), permuting the raw per-item scores and re-deriving
      every pairwise diff is, by construction, always one of the k!
      relabelings, so the joint null distribution it draws from is the
      correct one for any k. Requires ``arm_scores`` (k arms x m items) and
      ``pair_indices`` (row-index pairs into ``arm_scores``, parallel to
      ``diffs_mat``'s rows).

    Both branches avoid ever materializing a ``(k_pairs, b, m)`` gathered
    array (the naive "resample then reduce" approach), which dominates
    memory/time at this harness's largest cells (k=20 -> 190 pairs, n up to
    500): "bootstrap" resamples the SAME item index across every pair, so
    each replicate's per-pair mean/variance is a fixed linear combination of
    the original per-item diffs (weighted by "how many times item i was
    drawn"), computable via a ``(b, m)`` counts matrix (one ``bincount``)
    and two BLAS matmuls (``diffs_mat @ counts.T`` for the mean,
    ``diffs_mat**2 @ counts.T`` for the second moment) instead of a
    ``diffs_mat[:, idx]`` gather. "permutation" similarly replaces the
    ``relabeled[:, :, pair_i] - relabeled[:, :, pair_j]`` fancy-index
    differencing with one matmul against a fixed signed ``(k_arms,
    k_pairs)`` pairing matrix, since taking a pairwise diff is itself a
    linear map of the relabeled per-item arm vector. Verified bit-close
    (``np.allclose``) against the original gather-based formulas; ~12-27x
    faster for "bootstrap" and ~2.3x faster for "permutation" at k=20/n=500,
    with no regression at small k/n.
    """
    k_pairs, m = diffs_mat.shape
    means = diffs_mat.mean(axis=1)
    ses = diffs_mat.std(axis=1, ddof=1) / np.sqrt(m)
    ses_safe = np.where(ses > 1e-12, ses, 1.0)
    t_obs = np.abs(means) / ses_safe
    # m is this harness's smallest swept sample size (always >= 10), so
    # m == 1 never happens in practice -- guarded only so this never raises
    # (matching np.std(ddof=1)'s own non-crashing nan-on-degenerate-df
    # behavior) instead of Python's ZeroDivisionError on m/(m-1).
    ddof1_factor = m / (m - 1) if m > 1 else 1.0

    if resample_mode == "permutation":
        if arm_scores is None or pair_indices is None:
            raise ValueError("'permutation' resample_mode requires arm_scores and pair_indices")
        k_arms = arm_scores.shape[0]
        arm_scores_t = arm_scores.T  # (m, k_arms)
        pair_i = np.array([p[0] for p in pair_indices])
        pair_j = np.array([p[1] for p in pair_indices])
        # Signed pairing matrix: diff_perm[..., p] = relabeled[..., pair_i[p]]
        # - relabeled[..., pair_j[p]] -- see docstring.
        pairing_matrix = np.zeros((k_arms, k_pairs))
        pairing_matrix[pair_i, np.arange(k_pairs)] = 1.0
        pairing_matrix[pair_j, np.arange(k_pairs)] = -1.0
    else:
        diffs_sq = diffs_mat**2  # hoisted -- fixed across every batch below

    t_abs_chunks: list[np.ndarray] = []
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        b = end - start
        if resample_mode == "bootstrap":
            idx = rng.integers(0, m, size=(b, m))
            # counts[draw, item] = how many times `item` was drawn in that
            # replicate -- see docstring for why this replaces the gather.
            flat_idx = idx + (np.arange(b)[:, None] * m)
            counts = np.bincount(flat_idx.ravel(), minlength=b * m).reshape(b, m).astype(diffs_mat.dtype)
            b_means = (diffs_mat @ counts.T) / m  # (k_pairs, b)
            sq_means = (diffs_sq @ counts.T) / m  # (k_pairs, b)
            var_unbiased = np.maximum(sq_means - b_means**2, 0.0) * ddof1_factor
            b_ses = np.sqrt(var_unbiased) / np.sqrt(m)
            b_ses_safe = np.where(b_ses > 1e-12, b_ses, 1.0)
            t_vals = (b_means - means[:, None]) / b_ses_safe
        else:  # "permutation" -- per-item random relabeling of the k arms
            # perm[b, j] is a uniformly random permutation of {0, ..., k_arms-1}
            # for bootstrap draw b, item j (argsort of iid uniforms is the
            # standard trick for batched random permutations).
            perm = np.argsort(rng.random(size=(b, m, k_arms)), axis=2)
            relabeled = np.take_along_axis(
                np.broadcast_to(arm_scores_t[None, :, :], (b, m, k_arms)), perm, axis=2,
            )  # (b, m, k_arms): item j's scores relabeled across arms
            diff_perm = (relabeled.reshape(b * m, k_arms) @ pairing_matrix).reshape(b, m, k_pairs)
            b_means = diff_perm.mean(axis=1).T  # (k_pairs, b)
            b_ses = (diff_perm.std(axis=1, ddof=1) / np.sqrt(m)).T
            b_ses_safe = np.where(b_ses > 1e-12, b_ses, 1.0)
            t_vals = b_means / b_ses_safe
        t_abs_chunks.append(np.abs(t_vals))
    t_abs = np.concatenate(t_abs_chunks, axis=1)  # (k_pairs, n_bootstrap)
    return t_abs, t_obs


def _single_step_max_t_pvalues(t_abs: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    """max_t's single-step FWER p-value per pair, from an already-computed
    _bootstrap_t_matrix() draw -- the max-over-all-pairs-per-replicate
    distribution (``t_abs.max(axis=0)``) IS max_t's joint null, identical to
    what evalstats.core.paired._apply_max_t_cis derives independently from
    its own separate resample. Letting _compute_multiarm_metrics reuse one
    shared draw for both max_t and romano_wolf (whose step-down procedure
    needs this same matrix anyway) avoids drawing and reducing a second
    n_bootstrap x k_pairs resample just to recompute the same statistic."""
    m_b = t_abs.max(axis=0)  # (n_bootstrap,) -- max over ALL pairs per replicate
    b_total = t_abs.shape[1]
    extreme = (m_b[np.newaxis, :] >= t_obs[:, np.newaxis]).sum(axis=1)  # (k_pairs,)
    return (extreme + 1) / (b_total + 1)


def _stepdown_max_t_pvalues(
    diffs_mat: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
    resample_mode: str, batch_size: int = 256,
    arm_scores: np.ndarray | None = None, pair_indices: list[tuple[int, int]] | None = None,
    precomputed: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Step-down max-|T| FWER p-values: Romano & Wolf (2005)'s bootstrap
    step-down, or its permutation analogue, Westfall & Young (1993)'s
    step-down min-P/max-T.

    Unlike single-step max-T (this harness's `max_t`), which uses ONE joint
    critical value for every pair, the step-down refinement here recomputes
    the max only over pairs not yet rejected at each step (starting from the
    pair with the largest observed |t|, working down), which strictly
    dominates single-step max-T in power for the same strong FWER guarantee
    -- exactly the "recover power lost to Holm/Bonferroni when comparisons
    are positively correlated" case repeated-measures designs create, since
    shared items/participants make every pair's diffs correlated.

    Returns one FWER-adjusted p-value per row of ``diffs_mat`` (same pair
    order), monotonized via a running max along the testing order (the same
    reformulation Holm's own adjusted p-values use) so they are directly
    comparable to alpha.

    Parameters
    ----------
    precomputed : tuple[np.ndarray, np.ndarray], optional
        ``(t_abs, t_obs)`` already computed by _bootstrap_t_matrix -- pass
        this to skip resampling entirely and reuse an existing draw (e.g.
        shared with max_t's single-step p-value; see
        _compute_multiarm_metrics). When ``None`` (default), resamples
        fresh via _bootstrap_t_matrix exactly as before this parameter
        existed -- passing nothing changes nothing.
    """
    if precomputed is not None:
        t_abs, t_obs = precomputed
        k_pairs = t_abs.shape[0]
    else:
        k_pairs = diffs_mat.shape[0]
        t_abs, t_obs = _bootstrap_t_matrix(
            diffs_mat, n_bootstrap, rng, resample_mode, batch_size, arm_scores, pair_indices,
        )
    b_total = t_abs.shape[1]

    order = np.argsort(-t_obs)  # descending observed |t|: tested first
    t_abs_sorted = t_abs[order]
    # suffix_max[step] = max over pairs tested at or after `step` -- the
    # step-down "remaining hypotheses" set, per bootstrap draw.
    suffix_max = np.maximum.accumulate(t_abs_sorted[::-1], axis=0)[::-1]

    # Both loops below are pure functions of `order` (testing sequence), so
    # they vectorize directly: compare/count and the running max are taken
    # along that sequence, then scattered back to original pair indices in
    # one assignment instead of a per-pair Python loop (k_pairs up to 190).
    t_obs_sorted = t_obs[order]
    extreme_counts = (suffix_max >= t_obs_sorted[:, None]).sum(axis=1)
    raw_step_p_sorted = (extreme_counts + 1) / (b_total + 1)
    adjusted_sorted = np.minimum(np.maximum.accumulate(raw_step_p_sorted), 1.0)

    adjusted = np.empty(k_pairs)
    adjusted[order] = adjusted_sorted
    return adjusted


def _compute_multiarm_metrics(
    *, scores: np.ndarray, labels: list[str], method: str, corrections: list[str],
    n_bootstrap: int, alpha: float, statistic: str, rng: np.random.Generator,
    eval_type: str | None = None,
) -> tuple[dict[str, tuple[bool, bool]], dict[str, float]]:
    """Compute (any_reject, best_selected) for every correction strategy.

    none/holm/bonferroni/fdr_bh/hochberg/shaffer correct the base paired
    p-value evalstats itself would report for the data type: McNemar mid-p on
    binary (all three binary branches of core/paired.py route there), Wilcoxon
    signed-rank otherwise, via _safe_wilcoxon_p on each pair's
    per_input_diffs -- NOT --multiarm-method's raw p-value (bootstrap_t by
    default). This branches per eval_type, like Tango/Logit-t in --mode
    simultaneous_ci: an earlier version used Wilcoxon for every type, which
    made the binary FWER rows describe a test the library never runs on
    binary data.
    per_input_diffs/point_diff are built directly from `scores` (a plain
    per-input difference and its mean/median -- no resampling involved), not
    via all_pairwise(method=method, ...): that used to run the *full*
    method-specific bootstrap per pair (e.g. bootstrap_t's two independent
    n_bootstrap-sized resamples, one for the CI and one for the p-value)
    purely to obtain per_input_diffs/point_diff, which need no resampling at
    all -- wasting O(pairs * n_bootstrap) draws every rep/condition/cell for
    results this function never read. hochberg/shaffer are closed-form
    reweightings of the same Wilcoxon p-values (see
    evalstats.core.stats_utils.correct_pvalues; shaffer additionally needs
    `n_groups=k`, the number of arms, to derive its all-pairwise divisor
    sequence).

    max_t/romano_wolf/westfall_young are the exception: they still need
    genuine resampling, since Wilcoxon has no joint max-T analogue. `max_t`
    resamples from `scores` directly via evalstats.core.paired's
    _max_stat_simultaneous_cis, called directly rather than through
    _simultaneous_cis_router -- the router's only other job is falling back
    to a Bonferroni CI on degenerate bootstrap results, but this harness
    never reads that CI (only whether the max-T draw succeeded), so there's
    nothing for the router's required `results` argument to feed; skipping
    it avoids building an unused dict[pair, PairedDiffResult] stand-in on
    every call just to satisfy that argument. For bootstrap-compatible
    methods the resulting p-values are *single-step* max-T FWER-controlled
    p-values -- each is the min p-value commensurate with the simultaneous
    CI that was reported to the user. For non-bootstrap methods
    (permutation) max_t falls back to the raw (marginal) p-value, matching
    what all_pairwise itself does when its router falls back to Bonferroni
    (it only widens the CI, not `.p_value`).

    `romano_wolf`/`westfall_young` are the genuine *step-down* max-T
    procedures (see _stepdown_max_t_pvalues) -- unlike max_t, they don't go
    through all_pairwise/_simultaneous_cis_router at all, since step-down
    needs the full per-bootstrap-draw statistic matrix (not just the single
    joint critical value the router returns) to recompute the max over
    shrinking "not yet rejected" subsets. They're built directly off the
    same method-invariant per_input_diffs hochberg/shaffer/etc. use.

    `boot` is the multiarm analogue of --mode simultaneous_ci's `boot`:
    unlike max_t/romano_wolf/westfall_young, it is NOT tied to
    --multiarm-method -- like none/holm/bonferroni/etc. it always widens the
    canonical Wilcoxon p-value, but using a joint bootstrap critical value
    (the max-over-all-pairs studentized-mean resample -- same construction
    max_t/romano_wolf use) rather than a fixed, correlation-blind factor.
    Concretely: the joint critical value is translated to an equivalent
    alpha_eff (mirroring evalstats.core.paired._joint_bootstrap_scaled_
    simultaneous_cis's z<->alpha translation for CIs), and raw_p is rescaled
    by alpha/alpha_eff -- the same "scale the raw p-value by the correction
    factor" pattern Bonferroni's own adjustment uses, just with a
    resampled, correlation-aware factor instead of a fixed k.

    `boot` always uses the exact "bootstrap" resample (mean-based item
    bootstrap, romano_wolf's own resample_mode regardless of
    --multiarm-method), so it and `romano_wolf` always share ONE draw when
    both are requested; `max_t` additionally joins that shared draw when
    its own construction happens to match (method="bootstrap_t"/
    statistic="mean", the defaults) -- see _bootstrap_t_matrix's docstring.

    `friedman_nemenyi` is unaffected either way -- already its own
    rank-based omnibus + post-hoc test, unrelated to `method`.

    Returns
    -------
    tuple[dict[str, tuple[bool, bool]], dict[str, float]]
        ``(results, timings)`` -- *timings* maps each correction to its own
        wall-clock seconds (so e.g. romano_wolf/westfall_young's genuine
        step-down resampling shows up as slower than none/holm/bonferroni's
        closed-form reweighting in the report's Time(ms) column, instead of
        every correction row displaying the same aggregate "whole call"
        time). The one-time shared setup (diffs_by_pair/point_diff_by_pair/
        raw_p, and stepdown_corrections' diffs_mat) is folded into `none`'s
        and the first stepdown correction's timing respectively, since it's
        not fairly attributable to any other single correction.
    """
    results: dict[str, tuple[bool, bool]] = {}
    timings: dict[str, float] = {}
    k = len(labels)
    pairs = [(labels[i], labels[j]) for i in range(k) for j in range(i + 1, k)]

    _STEPDOWN_RESAMPLE_MODE = {"romano_wolf": "bootstrap", "westfall_young": "permutation"}
    non_friedman_non_maxt = [
        c for c in corrections
        if c not in ("friedman_nemenyi", "max_t", "boot") and c not in _STEPDOWN_RESAMPLE_MODE
    ]
    include_max_t = "max_t" in corrections
    include_boot = "boot" in corrections
    stepdown_corrections = [c for c in _STEPDOWN_RESAMPLE_MODE if c in corrections]

    if non_friedman_non_maxt or include_max_t or include_boot or stepdown_corrections:
        # Plain per-input differences and their mean/median -- no resampling
        # needed, unlike the method-specific bootstrap all_pairwise(method=
        # method, ...) used to run here just to throw away everything but
        # these two quantities (see docstring above).
        _t_setup0 = time.perf_counter()
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores  # (k_arms, n)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        diffs_by_pair: dict[tuple[str, str], np.ndarray] = {}
        point_diff_by_pair: dict[tuple[str, str], float] = {}
        for pair in pairs:
            a, b = pair
            d = flat[label_to_idx[a]] - flat[label_to_idx[b]]
            diffs_by_pair[pair] = d
            point_diff_by_pair[pair] = float(d.mean()) if statistic == "mean" else float(np.median(d))

        # Base p-value per pair, matching what evalstats itself would report
        # for this data type. Wilcoxon everywhere was wrong for binary: the
        # library routes binary paired data to McNemar mid-p (all three binary
        # branches in core/paired.py), so a binary FWER row built on Wilcoxon
        # described a test users never get. The two are not interchangeable --
        # measured on the compare_e2e binary DGP, Wilcoxon rejects a strict
        # SUPERSET of mid-p's rejections (every discordant decision across
        # 12k paired draws x 6 cells went Wilcoxon-only, none the other way),
        # so it carries more power and more Type-I, crossing nominal by n=200
        # where mid-p stays under it.
        if eval_type == "binary":
            raw_p = np.array([
                _mcnemar_midp_p(flat[label_to_idx[a]], flat[label_to_idx[b]])
                for a, b in pairs
            ])
        else:
            raw_p = np.array([_safe_wilcoxon_p(diffs_by_pair[pair]) for pair in pairs])
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
        _setup_elapsed = time.perf_counter() - _t_setup0

        for correction in non_friedman_non_maxt:
            _t0 = time.perf_counter()
            if correction == "none":
                adj_p = raw_p
            elif correction == "shaffer":
                adj_p = correct_pvalues(raw_p, correction, n_groups=k)
            else:
                adj_p = correct_pvalues(raw_p, correction)
            has_any = bool(np.any(adj_p < alpha))
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                pair_idx = pair_to_idx.get((best, other))
                if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                    best_selected = False
                    break
            results[correction] = (has_any, best_selected)
            timings[correction] = time.perf_counter() - _t0
        if "none" in timings:
            # The one-time shared setup (diffs/point-diffs/Wilcoxon p-values)
            # is attributed to `none`, mirroring _run_simultaneous_ci_cell's
            # equivalent choice -- it's the row that most directly needs it,
            # not fairly split across every other correction.
            timings["none"] += _setup_elapsed

        # `boot` and `romano_wolf` always use the exact same fixed,
        # mean-based item-bootstrap construction (romano_wolf's
        # resample_mode is hardcoded to "bootstrap" regardless of
        # --multiarm-method/--statistic; `boot` -- evalstats' canonical-
        # Wilcoxon analogue of --mode simultaneous_ci's `boot` -- is
        # deliberately the same fixed construction for the same "canonical,
        # not --multiarm-method-tied" reason none/holm/bonferroni/etc. are),
        # so whenever both are requested they always share one draw, no
        # condition needed. `max_t` only joins that shared draw when its
        # OWN construction (tied to --multiarm-method) happens to be the
        # identical thing -- method="bootstrap_t" and statistic="mean" (the
        # CLI defaults) -- see _bootstrap_t_matrix's docstring. Falls back
        # to each computing its own resample independently (unchanged
        # behavior) whenever nothing needs to share.
        need_shared_matrix = include_boot or "romano_wolf" in stepdown_corrections
        max_t_matches_shared = method == "bootstrap_t" and statistic == "mean"
        share_max_t = include_max_t and need_shared_matrix and max_t_matches_shared
        diffs_mat = None
        pair_indices = None
        shared_t_abs = None
        shared_t_obs = None
        _shared_elapsed = 0.0
        _shared_owner = None  # which correction's timing bucket absorbs the shared resample
        if stepdown_corrections or include_boot:
            _t_stack0 = time.perf_counter()
            diffs_mat = np.stack([diffs_by_pair[pair] for pair in pairs], axis=0)
            pair_indices = [(label_to_idx[a], label_to_idx[b]) for a, b in pairs]
            _stack_elapsed = time.perf_counter() - _t_stack0
            if need_shared_matrix:
                _t_shared0 = time.perf_counter()
                shared_t_abs, shared_t_obs = _bootstrap_t_matrix(diffs_mat, n_bootstrap, rng, "bootstrap")
                _shared_elapsed = _stack_elapsed + (time.perf_counter() - _t_shared0)
                _shared_owner = "romano_wolf" if "romano_wolf" in stepdown_corrections else "boot"

        if include_max_t:
            _t0 = time.perf_counter()
            try:
                if share_max_t:
                    maxt_p = _single_step_max_t_pvalues(shared_t_abs, shared_t_obs)
                else:
                    # _max_stat_simultaneous_cis called directly (not via
                    # _simultaneous_cis_router) -- the router's only other
                    # job is falling back to _bonferroni_simultaneous_cis on
                    # degenerate bootstrap results, but this harness never
                    # reads the resulting CI values (`cis`), only whether
                    # the max-T draw succeeded (`sim_pvalues`) -- so there's
                    # nothing for a `results`/PairedDiffResult stand-in
                    # (formerly `results_stub`, built eagerly here on every
                    # call) to actually feed. Falling straight back to
                    # `raw_p` (unadjusted Wilcoxon) when the bootstrap
                    # returns empty matches what the router's own fallback
                    # would have produced as far as this harness could tell
                    # anyway (a safe placeholder, not a real Bonferroni CI
                    # the code ever used).
                    cis, max_t_pvalues = _max_stat_simultaneous_cis(
                        scores=scores, pairs=pairs, labels=labels, method=method,
                        ci=1.0 - alpha, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                    )
                    maxt_p = np.array([max_t_pvalues[pair] for pair in pairs]) if cis else raw_p
                has_any = bool(np.any(maxt_p < alpha))
                best = labels[0]
                best_selected = True
                for other in labels[1:]:
                    pair_idx = pair_to_idx.get((best, other))
                    if pair_idx is None or not (maxt_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                        best_selected = False
                        break
                results["max_t"] = (has_any, best_selected)
            except Exception:
                results["max_t"] = (False, False)
            # When shared, max_t's own incremental cost on top of the
            # already-computed matrix really is this small -- the resample
            # itself is charged to whichever of romano_wolf/boot is present
            # (see _shared_owner below), the correction(s) that intrinsically
            # need the full matrix regardless of whether max_t joins in.
            timings["max_t"] = time.perf_counter() - _t0

        if include_boot:
            _t0 = time.perf_counter()
            try:
                # Joint bootstrap critical value -- the max-over-all-pairs
                # studentized-mean distribution, the exact same construction
                # evalstats.core.paired._joint_bootstrap_critical_value uses
                # for --mode simultaneous_ci's `boot` -- translated to an
                # equivalent alpha (matching that construction's own
                # z<->alpha translation) and used to rescale the canonical
                # Wilcoxon p-value the same way Bonferroni rescales it with
                # a fixed, correlation-blind factor (alpha/k), except this
                # factor comes from a resampled joint null that DOES account
                # for correlation between comparisons.
                c = float(np.quantile(shared_t_abs.max(axis=0), 1.0 - alpha))
                alpha_eff = float(2.0 * (1.0 - scipy_stats.norm.cdf(c)))
                alpha_eff = min(max(alpha_eff, 1e-9), 1.0 - 1e-9)
                adj_p = np.minimum(raw_p * (alpha / alpha_eff), 1.0)
                has_any = bool(np.any(adj_p < alpha))
                best = labels[0]
                best_selected = True
                for other in labels[1:]:
                    pair_idx = pair_to_idx.get((best, other))
                    if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                        best_selected = False
                        break
                results["boot"] = (has_any, best_selected)
            except Exception:
                results["boot"] = (False, False)
            elapsed = time.perf_counter() - _t0
            if _shared_owner == "boot":
                elapsed += _shared_elapsed
            timings["boot"] = elapsed

        if stepdown_corrections:
            for i, correction in enumerate(stepdown_corrections):
                _t0 = time.perf_counter()
                try:
                    resample_mode = _STEPDOWN_RESAMPLE_MODE[correction]
                    if correction == "romano_wolf" and shared_t_abs is not None:
                        adj_p = _stepdown_max_t_pvalues(
                            diffs_mat, n_bootstrap, rng, resample_mode,
                            precomputed=(shared_t_abs, shared_t_obs),
                        )
                    else:
                        adj_p = _stepdown_max_t_pvalues(
                            diffs_mat, n_bootstrap, rng, resample_mode,
                            arm_scores=flat if resample_mode == "permutation" else None,
                            pair_indices=pair_indices if resample_mode == "permutation" else None,
                        )
                    has_any = bool(np.any(adj_p < alpha))
                    best = labels[0]
                    best_selected = True
                    for other in labels[1:]:
                        pair_idx = pair_to_idx.get((best, other))
                        if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                            best_selected = False
                            break
                    results[correction] = (has_any, best_selected)
                except Exception:
                    results[correction] = (False, False)
                elapsed = time.perf_counter() - _t0
                if correction == "romano_wolf" and _shared_owner == "romano_wolf":
                    # Includes the shared resample (see above) -- the one
                    # place its real cost is now charged.
                    elapsed += _shared_elapsed
                elif i == 0 and _shared_owner is None:
                    # np.stack's construction cost is shared setup for every
                    # stepdown correction; attributed to the first one rather
                    # than double-counted or arbitrarily split -- only when
                    # nothing else already absorbed it (_shared_owner is set
                    # whenever boot/romano_wolf triggered a shared matrix,
                    # which already includes this same stack cost).
                    elapsed += _stack_elapsed
                timings[correction] = elapsed

    if "friedman_nemenyi" in corrections:
        _t0 = time.perf_counter()
        try:
            fr = friedman_nemenyi(scores, labels)
            has_any = any(
                (p is not None and p < alpha) for (a, b) in pairs for p in [fr.get_nemenyi_p(a, b)]
            )
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                nem_p = fr.get_nemenyi_p(best, other)
                if nem_p is None:
                    best_selected = False
                    break
                if not (nem_p < alpha and fr.avg_ranks[best] < fr.avg_ranks[other]):
                    best_selected = False
                    break
            results["friedman_nemenyi"] = (has_any, best_selected)
        except Exception:
            results["friedman_nemenyi"] = (False, False)
        timings["friedman_nemenyi"] = time.perf_counter() - _t0

    return results, timings


def _run_multiarm_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed, corrections: list[str] | None = None,
) -> list[MultiArmResult]:
    """Run n_reps replications of a k-arm comparison at one (source, n, k)
    cell, across every requested multiple-comparisons correction. One
    MultiArmResult per correction."""
    labels = [f"arm_{i}" for i in range(k_arms)]
    if corrections is None:
        corrections = [m.name for m in MULTIARM_CORRECTION_METHODS]
    rng = np.random.default_rng(seed)

    agg_any: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    agg_best: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    # Per-(correction, condition), not a single per-condition total -- each
    # correction's own wall-clock cost, so e.g. romano_wolf/westfall_young's
    # genuine step-down resampling shows up as slower than none/holm/
    # bonferroni's closed-form reweighting in the report's Time(ms) column,
    # instead of every correction row displaying the same aggregate "whole
    # rep" time (see _compute_multiarm_metrics's per-correction timings).
    agg_time: dict[tuple[str, str], float] = {(c, cond): 0.0 for c in corrections for cond in ("null", "alt")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                # Score generation is shared setup, attributed to `none` --
                # not fairly attributable to any other single correction.
                _t_none0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                _scores_elapsed = time.perf_counter() - _t_none0
                metrics, timings = _compute_multiarm_metrics(
                    scores=scores, labels=labels, method=multiarm_method, corrections=corrections,
                    eval_type=source.eval_type,
                    n_bootstrap=n_bootstrap, alpha=alpha, statistic=statistic, rng=rng,
                )
                if "none" in timings:
                    timings["none"] += _scores_elapsed
                for correction in corrections:
                    any_reject, best_selected = metrics.get(correction, (False, False))
                    if any_reject:
                        agg_any[(correction, condition)] += 1
                    if best_selected:
                        agg_best[(correction, condition)] += 1
                    agg_time[(correction, condition)] += timings.get(correction, 0.0)

    return [
        MultiArmResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, correction=correction,
            condition=condition, n_reps=n_reps, any_reject=agg_any[(correction, condition)],
            best_selected=agg_best[(correction, condition)],
            total_time=agg_time[(correction, condition)],
        )
        for correction in corrections
        for condition in ("null", "alt")
    ]


def _multiarm_cell_feasible(s: MultiArmSource, n: int, k: int) -> bool:
    """Shared by run_multiarm_simulation and run_simultaneous_ci_simulation
    (both sweep the same MultiArmSource list over the same n x k grid)."""
    return (s.max_n is None or n < s.max_n) and (s.max_k is None or k <= s.max_k)


def _multiarm_style_cells(
    sources: list[MultiArmSource], sample_sizes: list[int], k_values: list[int],
) -> list[tuple[int, int, int]]:
    """(source_idx, n, k) cells feasible for every source, printing a skip
    warning (mirroring CISource.max_n's skip pattern) for infeasible ones."""
    cells = [(i, n, k) for i, s in enumerate(sources) for n in sample_sizes for k in k_values
             if _multiarm_cell_feasible(s, n, k)]
    skipped = [(s, n, k) for s in sources for n in sample_sizes for k in k_values
               if not _multiarm_cell_feasible(s, n, k)]
    for s, n, k in skipped:
        reason = f"n={n} >= corpus size {s.max_n}" if not (s.max_n is None or n < s.max_n) else f"k={k} > {s.max_k} real arms available"
        print(f"  Warning: {reason} for {s.label}. Skipping.")
    return cells


def run_multiarm_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_values: list[int], n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar",
    seed: int = 42, n_workers: int = 1, corrections: list[str] | None = None,
) -> list[MultiArmResult]:
    """Sweep _run_multiarm_cell over every (source, sample size, k) cell,
    parallelized across n_workers, and flatten the per-cell MultiArmResult
    lists into one list."""
    global _MULTIARM_SOURCES
    _MULTIARM_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, corrections)
                 for (sc_idx, n, k), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-multiarm")
    results: list[MultiArmResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_multiarm_cell_worker(a))
            sc_idx, n, k = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n} k={k}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_multiarm_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def _time_stats_multiarm(results: list[MultiArmResult]) -> tuple[float, float]:
    """Average ± SE of wall-clock time per rep in milliseconds across cells."""
    valid = [r for r in results if r.total_time > 0 and r.n_reps > 0]
    if not valid:
        return float("nan"), float("nan")
    per_rep_ms = [r.total_time * 1000.0 / r.n_reps for r in valid]
    avg = float(np.mean(per_rep_ms))
    se = float(np.std(per_rep_ms, ddof=1) / np.sqrt(len(per_rep_ms))) if len(per_rep_ms) > 1 else 0.0
    return avg, se


def print_multiarm_report(results: list[MultiArmResult], alpha: float) -> None:
    """Print the console FWER/best-arm-power report for a multi-arm run,
    grouped by eval type and k."""
    _, _bradley_hi = bradley_bounds(alpha)
    print(f"\n{'='*78}\n  PVALUES (MULTI-ARM, NON-PPI) -- FWER + BEST-ARM POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ks_present = sorted({r.k for r in results})
    for et in eval_types_present:
        for k in ks_present:
            subset = [r for r in results if r.eval_type == et and r.k == k]
            if not subset:
                continue
            print(f"\n  [{et}, k={k}]")
            print(f"    {'Correction':<20} {'FWER':>8} {'BestPower':>10}")
            for corr in corrections:
                c_rows = [r for r in subset if r.correction == corr]
                null_rows = [r for r in c_rows if r.condition == "null"]
                alt_rows = [r for r in c_rows if r.condition == "alt"]
                fwer_t = sum(r.n_reps for r in null_rows)
                fwer_c = sum(r.any_reject for r in null_rows)
                power_t = sum(r.n_reps for r in alt_rows)
                power_c = sum(r.best_selected for r in alt_rows)
                fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
                power = power_c / power_t if power_t > 0 else float("nan")
                print(f"    {corr:<20} {fwer:>8.3f} {power:>10.3f}")

    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    print(f"\n{'-'*72}\n  OVERALL SUMMARY (collapsed across eval types, sources, n, k)\n{'-'*72}")
    print(f"  MaxFWER = worst per-scenario FWER seen for that correction (not an average) --\n"
          f"  flags corrections whose good mean FWER hides an inflated scenario/n/k cell.")
    n_cols = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    k_cols = "".join(f"  {'k='+str(k):>6}" for k in ks_present)
    print(f"\n  {'Correction':<20}  {'FWER':>6}  {'MaxFWER':>8}  {'Band95':>13}  {'BestPow':>8}  {'Time(ms)':>14}{n_cols}{k_cols}")
    for corr in corrections:
        c_rows = [r for r in results if r.correction == corr]
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        fwer_c = sum(r.any_reject for r in null_rows)
        fwer_t = sum(r.n_reps for r in null_rows)
        power_c = sum(r.best_selected for r in alt_rows)
        power_t = sum(r.n_reps for r in alt_rows)
        fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
        power = power_c / power_t if power_t > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(fwer_c, fwer_t)
        avg_ms, se_ms = _time_stats_multiarm(null_rows)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        time_str = f"{avg_ms:.1f}+-{se_ms:.1f}" if np.isfinite(avg_ms) else "-"
        marker = "*" if np.isfinite(fwer) and fwer > _bradley_hi else " "
        per_label_fwer = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_fwer[(r.eval_type, r.label)]
            acc[0] += r.any_reject
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_fwer.values() if t > 0]
        worst_fwer = max(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_fwer:.3f}{'*' if np.isfinite(worst_fwer) and worst_fwer > _bradley_hi else ' '}" if np.isfinite(worst_fwer) else "-"
        n_fwer = ""
        for n in sizes_present:
            n_null = [r for r in null_rows if r.n == n]
            nc = sum(r.any_reject for r in n_null)
            nt = sum(r.n_reps for r in n_null)
            nf = nc / nt if nt > 0 else float("nan")
            n_fwer += f"  {nf:>7.3f}" if np.isfinite(nf) else f"  {'  -':>7}"
        k_fwer = ""
        for k in ks_present:
            k_null = [r for r in null_rows if r.k == k]
            kc = sum(r.any_reject for r in k_null)
            kt = sum(r.n_reps for r in k_null)
            kf = kc / kt if kt > 0 else float("nan")
            k_fwer += f"  {kf:>6.3f}" if np.isfinite(kf) else f"  {'  -':>6}"
        print(f"  {corr:<20}  {fwer:>5.3f}{marker}  {worst_str:>8}  {band:>13}  {power:>8.3f}  {time_str:>14}{n_fwer}{k_fwer}")
    print(f"  (* = FWER above Bradley's liberal band, i.e. > 1.5*alpha = {_bradley_hi:.3f})")


def latex_multiarm_overall_summary(results: list[MultiArmResult], alpha: float, *,
                                   include_uncorrected: bool = True) -> str:
    """LaTeX booktabs overall summary: per-correction FWER (with its 95% MC
    band) + best-arm power, plus one FWER column per sample size and per k
    value actually swept.

    As in `latex_pairwise_overall_summary`, corrections that ran on more
    than one eval type get one row per type in midrule-separated blocks
    rather than a single pooled row, matching the ci_single/ci_paired
    layout; power is ranked within a block.
    """
    # See latex_simultaneous_ci_overall_summary: `none` shades saturated red
    # across the row and only restates that correction is needed; the plots
    # already drop it (MULTIARM_PLOT_METHODS).
    pool = MULTIARM_CORRECTION_METHODS if include_uncorrected else MULTIARM_PLOT_METHODS
    corrections = [m.name for m in pool if m.name in {r.correction for r in results}]
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    ks_present = sorted({r.k for r in results if r.condition == "null"})

    corr_groups: dict[str, set[str]] = defaultdict(set)
    for r in results:
        if r.correction not in corrections:
            continue
        corr_groups[r.correction].add(report_eval_type_group(r.eval_type))
    groups_present = sort_groups({g for gs in corr_groups.values() for g in gs})

    rows = []
    rule_before = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        powers, fwers = [], []
        for corr in corrections:
            if g not in corr_groups[corr]:
                continue
            c_rows = [r for r in results
                      if r.correction == corr and report_eval_type_group(r.eval_type) == g]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
            power = power_c / power_t if power_t > 0 else float("nan")
            _, _, lo, hi = _mc_proportion_stats(fwer_c, fwer_t)
            avg_ms, se_ms = _time_stats_multiarm(null_rows)
            # No +- se: it is a fraction of a millisecond on every method and
            # eats a column's width for nothing (the CI tables drop it too).
            time_str = f"{avg_ms:.1f}" if np.isfinite(avg_ms) else "-"
            label = f"{escape_latex(corr)} ({g})" if len(corr_groups[corr]) > 1 else escape_latex(corr)
            row = [
                label,
                error_rate_cell(fwer, alpha),
                f"{power:.3f}" if np.isfinite(power) else "-",
                time_str,
                g,
            ]
            for n in sizes_present:
                n_rows = [r for r in null_rows if r.n == n]
                c_n = sum(r.any_reject for r in n_rows)
                t_n = sum(r.n_reps for r in n_rows)
                fwer_n = c_n / t_n if t_n > 0 else float("nan")
                row.append(error_rate_cell(fwer_n, alpha))
            for k in ks_present:
                k_rows = [r for r in null_rows if r.k == k]
                c_k = sum(r.any_reject for r in k_rows)
                t_k = sum(r.n_reps for r in k_rows)
                fwer_k = c_k / t_k if t_k > 0 else float("nan")
                row.append(error_rate_cell(fwer_k, alpha))
            rows.append(row)
            powers.append(power)
            fwers.append(fwer)

        # See the pairwise table: power is the marked column, FWER is
        # shaded, and a correction that doesn't hold its FWER can't win on
        # power.
        POWER_COL = 2
        block = rows[block_start:]
        marked = mark_best_and_runnerup(
            [r[POWER_COL] for r in block],
            _power_ranking_values(powers, fwers, alpha),
            higher_is_better=True,
        )
        for row, cell in zip(block, marked):
            row[POWER_COL] = cell

    return booktabs_table(
        caption=f"pvalues (multi-arm, non-PPI): FWER and best-arm selection power (nominal alpha={alpha}). "
                f"Corrections tested on more than one eval type are reported as one row per type "
                f"(bin/cont/lik), grouped into blocks, so no row averages across eval types. "
                f"Per-$n$ and per-$k$ FWER columns are collapsed across the other dimension only. "
                f"FWER cells shade red when inflated above {alpha} and blue when conservative below it, "
                f"on the same scale as the coverage tables; best and runner-up power are bold and "
                f"underlined within each block, among corrections holding their nominal level.",
        label="tab:pvalues_multiarm_overall",
        columns=["Correction", "FWER", "Best-arm power", "Time (ms)", "Type"]
                + [f"n={n}" for n in sizes_present]
                + [f"k={k}" for k in ks_present],
        rows=rows,
        rule_before=rule_before,
    )


def save_results_artifacts_multiarm(*, results: list[MultiArmResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    """Write the multi-arm run's results CSV (and LaTeX summary if
    `latex=True`) under out_dir. Returns the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_multiarm_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "k", "correction", "condition", "n_reps", "any_reject", "best_selected", "any_reject_rate", "best_selected_rate", "total_time_s", "time_ms_per_rep"])
        for r in results:
            time_ms = (r.total_time * 1000.0 / r.n_reps) if r.n_reps > 0 and r.total_time > 0 else float("nan")
            writer.writerow([
                r.eval_type, r.label, r.n, r.k, r.correction, r.condition, r.n_reps, r.any_reject, r.best_selected,
                f"{r.any_reject / r.n_reps:.8f}", f"{r.best_selected / r.n_reps:.8f}",
                f"{r.total_time:.6f}", f"{time_ms:.4f}" if not (time_ms != time_ms) else "",
            ])
    summary_path = out_base / f"{run_stem}_multiarm_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_multiarm_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_multiarm_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_multiarm_fwer_power_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER vs. best-arm-selection power, one point per correction strategy per eval type."""
    import matplotlib.pyplot as plt

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(5.0 * nrows, 5.0), squeeze=False)

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_rows = [r for r in results if r.eval_type == et]
        ax.axvline(alpha, color="black", linestyle="--", linewidth=1.0)
        powers: list[float] = []
        for m in MULTIARM_CORRECTION_METHODS:
            c_rows = [r for r in et_rows if r.correction == m.name]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            t1 = sum(r.n_reps for r in null_rows)
            c1 = sum(r.any_reject for r in null_rows)
            t2 = sum(r.n_reps for r in alt_rows)
            c2 = sum(r.best_selected for r in alt_rows)
            if t1 == 0 or t2 == 0:
                continue
            fwer = c1 / t1
            power = c2 / t2
            ax.scatter([fwer], [power], color=m.color, s=60, label=m.name, edgecolors="white", linewidths=0.6)
            powers.append(power)
        ax.set_xlabel("FWER (null)")
        ax.set_ylabel("Best-arm selection power (alt)")
        ax.set_title(f"eval type: {et}")
        ax.set_xlim(-0.02, max(0.3, alpha * 4))
        # Zoom to the actual power spread rather than a fixed [0, 1] -- power
        # can cluster near either end (uniformly low under a strict per-pair
        # rejection requirement, or uniformly high at large n), and a full
        # [0, 1] axis squashes that spread into an unreadable sliver either
        # way. No artificial 0.0 floor seed -- that would defeat the zoom
        # whenever power clusters near 1.0.
        if powers:
            pow_lo, pow_hi = min(powers), max(powers)
            pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
            ax.set_ylim(max(-0.02, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))
        else:
            ax.set_ylim(-0.02, 1.02)
    # One legend outside the rightmost facet rather than one per facet: every
    # facet plots the same correction strategies, so per-facet legends were
    # both redundant and sitting on top of the points they labelled.
    _handles, _labels = axes[0][0].get_legend_handles_labels()
    if _handles:
        axes[0][-1].legend(_handles, _labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                           borderaxespad=0.0, fontsize=7)

    fig.suptitle(f"Family-Wise Error Rate vs. Best-Arm Selection Power\nNominal alpha = {alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_fwer_vs_k_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER and best-arm power as a function of k (number of arms), one curve per
    correction method, collapsed across eval types and sample sizes -- the
    multiarm analogue of save_simultaneous_ci_coverage_width_vs_k_plot (same
    two-panel line-plot style: exact integer x-ticks pinned to the k values
    actually swept, FWER y-axis zoomed to the actual spread rather than a
    fixed [0, ...]). Only plots MULTIARM_PLOT_METHODS (every registered
    correction except `none` -- see that list's comment for why; `none` is
    still in the printed/logged report tables and the CSV).
    Only produced when more than one k value was swept; returns out_path
    unchanged (without writing) if all results share the same k."""
    import matplotlib.pyplot as plt

    ks_present = sorted({r.k for r in results})
    if len(ks_present) < 2:
        return out_path

    fig, (ax_fwer, ax_pow) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_fwer.axhline(alpha, color="black", linewidth=1.0, linestyle="--", label=f"α={alpha}")
    ax_fwer.axhspan(*bradley_bounds(alpha), color="#DDDDDD", alpha=0.4, zorder=0)

    all_fwer_vals: list[float] = [alpha]
    all_pow_vals: list[float] = []
    for m in MULTIARM_PLOT_METHODS:
        c_rows = [r for r in results if r.correction == m.name]
        if not c_rows:
            continue
        xs, ys_fwer, ys_pow = [], [], []
        scen_fwer, scen_pow = [], []
        for k in ks_present:
            k_rows = [r for r in c_rows if r.k == k]
            null_rows = [r for r in k_rows if r.condition == "null"]
            alt_rows = [r for r in k_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            if fwer_t == 0 or power_t == 0:
                continue
            xs.append(k)
            ys_fwer.append(fwer_c / fwer_t)
            ys_pow.append(power_c / power_t)
            scen_fwer.append(_scenario_values(null_rows, lambda r: r.any_reject))
            scen_pow.append(_scenario_values(alt_rows, lambda r: r.best_selected))
        if xs:
            ax_fwer.plot(xs, ys_fwer, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_pow.plot(xs, ys_pow, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            # Include the band endpoints in the y-limit inputs, not just
            # the point estimates, so the zoom below doesn't clip the band.
            all_fwer_vals.extend(_scenario_bands(ax_fwer, xs, ys_fwer, scen_fwer, color=m.color))
            all_pow_vals.extend(_scenario_bands(ax_pow, xs, ys_pow, scen_pow, color=m.color))
            all_fwer_vals.extend(ys_fwer)
            all_pow_vals.extend(ys_pow)

    ax_fwer.set_xlabel("k (number of arms)")
    ax_fwer.set_ylabel("FWER (null)")
    ax_fwer.set_title("FWER vs. number of arms")
    # Zoom to the actual FWER spread (plus the nominal alpha line) rather
    # than a fixed [0, ...] -- with `none` dropped from this plot
    # (MULTIARM_PLOT_METHODS), every remaining curve usually clusters near
    # alpha, and a floor of 0.0 squashes that spread into an unreadable
    # sliver at the bottom (see save_simultaneous_ci_coverage_width_vs_k_plot's
    # identical fix for coverage).
    fwer_lo, fwer_hi = min(all_fwer_vals), max(all_fwer_vals)
    fwer_pad = max(0.005, (fwer_hi - fwer_lo) * 0.15)
    ax_fwer.set_ylim(max(0.0, fwer_lo - fwer_pad), fwer_hi + fwer_pad)
    ax_fwer.set_xticks(ks_present)

    ax_pow.set_xlabel("k (number of arms)")
    ax_pow.set_ylabel("Best-arm selection power (alt)")
    ax_pow.set_title("Power vs. number of arms")
    # Zoom to the actual power spread rather than a fixed [0, 1] -- power is
    # often concentrated near one end (uniformly low, or uniformly high as
    # with best-arm selection at large n), and a full [0, 1] axis squashes
    # that spread the same way an unzoomed FWER axis would (see above). No
    # artificial 0.0 floor seed (unlike FWER's alpha-line seed) -- that would
    # defeat the zoom whenever power clusters near 1.0.
    if all_pow_vals:
        pow_lo, pow_hi = min(all_pow_vals), max(all_pow_vals)
        pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
        ax_pow.set_ylim(max(0.0, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))
    ax_pow.set_xticks(ks_present)

    # One shared legend for both panels (FWER's nominal-alpha line plus every
    # method, which both panels plot identically) instead of a separate
    # legend per panel, placed outside the axes to the right.
    handles, labels = ax_fwer.get_legend_handles_labels()
    ax_pow.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    fig.suptitle(
        "Family-Wise Error Rate and Best-Arm Selection Power vs. Number of Systems Compared\n"
        f"Nominal alpha = {alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_fwer_vs_n_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER and best-arm power as a function of n (sample size), one curve
    per correction method, collapsed across eval types and k -- the
    sample-size analogue of save_multiarm_fwer_vs_k_plot (same two-panel
    line-plot style: FWER y-axis zoomed to the actual spread rather than a
    fixed [0, ...]). X-axis is log-scaled, unlike the vs-k plot's linear
    one: n sweeps span an order of magnitude or more, so a linear axis
    crams the small-n tick labels into an unreadable overlapping cluster
    (see save_simultaneous_ci_coverage_width_vs_n_plot's identical fix).
    Only plots MULTIARM_PLOT_METHODS (every registered correction except
    `none` -- see that list's comment for why; `none` is still in the
    printed/logged report tables and the CSV).
    Only produced when more than one n value was swept; returns out_path
    unchanged (without writing) if all results share the same n."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    if len(sizes_present) < 2:
        return out_path

    fig, (ax_fwer, ax_pow) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_fwer.axhline(alpha, color="black", linewidth=1.0, linestyle="--", label=f"α={alpha}")
    ax_fwer.axhspan(*bradley_bounds(alpha), color="#DDDDDD", alpha=0.4, zorder=0)

    all_fwer_vals: list[float] = [alpha]
    all_pow_vals: list[float] = []
    for m in MULTIARM_PLOT_METHODS:
        c_rows = [r for r in results if r.correction == m.name]
        if not c_rows:
            continue
        xs, ys_fwer, ys_pow = [], [], []
        scen_fwer, scen_pow = [], []
        for n in sizes_present:
            n_rows = [r for r in c_rows if r.n == n]
            null_rows = [r for r in n_rows if r.condition == "null"]
            alt_rows = [r for r in n_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            if fwer_t == 0 or power_t == 0:
                continue
            xs.append(n)
            ys_fwer.append(fwer_c / fwer_t)
            ys_pow.append(power_c / power_t)
            scen_fwer.append(_scenario_values(null_rows, lambda r: r.any_reject))
            scen_pow.append(_scenario_values(alt_rows, lambda r: r.best_selected))
        if xs:
            ax_fwer.plot(xs, ys_fwer, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_pow.plot(xs, ys_pow, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            # Include the band endpoints in the y-limit inputs, not just
            # the point estimates, so the zoom below doesn't clip the band.
            all_fwer_vals.extend(_scenario_bands(ax_fwer, xs, ys_fwer, scen_fwer, color=m.color))
            all_pow_vals.extend(_scenario_bands(ax_pow, xs, ys_pow, scen_pow, color=m.color))
            all_fwer_vals.extend(ys_fwer)
            all_pow_vals.extend(ys_pow)

    ax_fwer.set_xlabel("n (sample size)")
    ax_fwer.set_ylabel("FWER (null)")
    ax_fwer.set_title("FWER vs. sample size")
    fwer_lo, fwer_hi = min(all_fwer_vals), max(all_fwer_vals)
    fwer_pad = max(0.005, (fwer_hi - fwer_lo) * 0.15)
    ax_fwer.set_ylim(max(0.0, fwer_lo - fwer_pad), fwer_hi + fwer_pad)

    ax_pow.set_xlabel("n (sample size)")
    ax_pow.set_ylabel("Best-arm selection power (alt)")
    ax_pow.set_title("Power vs. sample size")
    # Zoom to the actual power spread -- see save_multiarm_fwer_vs_k_plot's
    # identical fix (no artificial 0.0 floor seed).
    if all_pow_vals:
        pow_lo, pow_hi = min(all_pow_vals), max(all_pow_vals)
        pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
        ax_pow.set_ylim(max(0.0, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))

    # One shared legend for both panels, placed outside the axes to the
    # right -- see save_multiarm_fwer_vs_k_plot's identical fix.
    handles, labels = ax_fwer.get_legend_handles_labels()
    ax_pow.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    # Log-scale x-axis (see docstring) with exact tick labels at the swept
    # sizes instead of matplotlib's default log-scale power-of-ten ticks.
    for ax in (ax_fwer, ax_pow):
        ax.set_xscale("log")
        ax.set_xticks(sizes_present)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle(
        "Family-Wise Error Rate and Best-Arm Selection Power vs. Sample Size\n"
        f"Nominal alpha = {alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fwer_panel_axis(ax, xs, series, *, hline=None, band=None, ylabel="", xlabel=""):
    """One panel of the compact 1x4 FWER figures.

    Shared by save_multiarm_fwer_panels_plot and
    save_simultaneous_ci_panels_plot. `series` maps method name -> (y, sem).
    """
    import matplotlib.ticker as mticker
    for name, (y, e) in series.items():
        color = METHODS_BY_NAME[name].color if name in METHODS_BY_NAME else None
        ax.plot(xs, y, "-o", color=color, label=name)
        if e is not None:
            ax.fill_between(xs, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e),
                            color=color, alpha=0.18, linewidth=0)
    if band is not None:
        ax.axhspan(*band, color="0.90", zorder=0)
    if hline is not None:
        ax.axhline(hline, ls="--", lw=0.8, color="black")
    ax.set_xscale("log")
    ax.set_xticks(list(xs))
    # 1000 -> "1k": at four panels across the text width, 500 and 1000 collide.
    ax.get_xaxis().set_major_formatter(
        mticker.FuncFormatter(lambda v, _: (f"{v/1000:g}k" if v >= 1000 else f"{v:g}")))
    ax.get_xaxis().set_minor_locator(mticker.NullLocator())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _fwer_panels_figure(panels, methods, out_path):
    r"""Render a 1x4 panel row at ACM \textwidth with print-sized fonts.

    Drawn at its FINAL printed width (7in) so nothing is downscaled on
    \includegraphics -- the older two-panel plots were ~14.5in wide and shrank
    to ~0.38x in the paper, which is what made their labels unreadable.
    """
    import matplotlib.pyplot as plt
    with plt.rc_context({
        "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 2.6,
    }):
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
        for ax, kw in zip(axes, panels):
            _fwer_panel_axis(ax, **kw)
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 5
        nrows = -(-len(labels) // ncol)
        fig.tight_layout(rect=[0, 0.03 + 0.085 * nrows, 1, 1], w_pad=0.8)
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   handlelength=1.3, columnspacing=1.0, handletextpad=0.4,
                   borderaxespad=0.1, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out_path


def save_multiarm_fwer_panels_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """Compact 1x4 replacement for save_multiarm_fwer_vs_{n,k}_plot.

    FWER and best-arm power, each against n and against k, in one row with a
    single shared legend. This is the version the paper prints: the two
    separate two-panel plots carried three copies of the same legend between
    them and cost ~0.58 pages each; this costs ~0.20.

    "none" (uncorrected) is excluded: it runs at FWER ~0.45 and compresses
    every corrected method into an unreadable band -- the same reason the
    vs_n/vs_k plots drop it (see their note above).
    """
    rows = [r for r in results if r.correction != "none"]
    if not rows:
        return out_path
    methods = sorted({r.correction for r in rows})

    def agg(xattr, cond, num, den):
        xs = sorted({getattr(r, xattr) for r in rows})
        series = {}
        for m in methods:
            ys, es = [], []
            for x in xs:
                sel = [r for r in rows
                       if r.correction == m and getattr(r, xattr) == x and r.condition == cond]
                tot = sum(getattr(r, den) for r in sel)
                hit = sum(getattr(r, num) for r in sel)
                pr = hit / tot if tot else float("nan")
                ys.append(pr)
                es.append(math.sqrt(max(pr * (1 - pr), 0.0) / tot) if tot else 0.0)
            series[m] = (ys, es)
        return xs, series

    xs_n, s_n = agg("n", "null", "any_reject", "n_reps")
    xk_n, sk_n = agg("k", "null", "any_reject", "n_reps")
    xs_p, s_p = agg("n", "alt", "best_selected", "n_reps")
    xk_p, sk_p = agg("k", "alt", "best_selected", "n_reps")
    panels = [
        dict(xs=xs_n, series=s_n, hline=alpha, band=(alpha / 2, alpha * 1.5),
             ylabel="FWER (null)", xlabel="n (sample size)"),
        dict(xs=xk_n, series=sk_n, hline=alpha, band=(alpha / 2, alpha * 1.5),
             ylabel="FWER (null)", xlabel="k (arms)"),
        dict(xs=xs_p, series=s_p, ylabel="Best-arm power", xlabel="n (sample size)"),
        dict(xs=xk_p, series=sk_p, ylabel="Best-arm power", xlabel="k (arms)"),
    ]
    return _fwer_panels_figure(panels, methods, out_path)


def save_simultaneous_ci_panels_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Compact 1x4 replacement for save_simultaneous_ci_coverage_width_vs_{n,k}_plot."""
    rows = [r for r in results if r.ci_method != "none" and r.condition == "alt"]
    if not rows:
        return out_path
    methods = sorted({r.ci_method for r in rows})

    def agg(xattr, kind):
        xs = sorted({getattr(r, xattr) for r in rows})
        series = {}
        for m in methods:
            ys, es = [], []
            for x in xs:
                sel = [r for r in rows if r.ci_method == m and getattr(r, xattr) == x]
                tot = sum(r.n_reps for r in sel)
                if kind == "cov":
                    hit = sum(r.all_covered for r in sel)
                    pr = hit / tot if tot else float("nan")
                    ys.append(pr)
                    es.append(math.sqrt(max(pr * (1 - pr), 0.0) / tot) if tot else 0.0)
                else:
                    ys.append(sum(r.total_width for r in sel) / tot if tot else float("nan"))
                    es.append(0.0)
            series[m] = (ys, es)
        return xs, series

    xs_c, s_c = agg("n", "cov")
    xk_c, sk_c = agg("k", "cov")
    xs_w, s_w = agg("n", "width")
    xk_w, sk_w = agg("k", "width")
    tgt = 1 - alpha
    panels = [
        dict(xs=xs_c, series=s_c, hline=tgt, band=(tgt - 0.025, tgt + 0.025),
             ylabel="FW coverage", xlabel="n (sample size)"),
        dict(xs=xk_c, series=sk_c, hline=tgt, band=(tgt - 0.025, tgt + 0.025),
             ylabel="FW coverage", xlabel="k (arms)"),
        dict(xs=xs_w, series=s_w, ylabel="Avg. width", xlabel="n (sample size)"),
        dict(xs=xk_w, series=sk_w, ylabel="Avg. width", xlabel="k (arms)"),
    ]
    return _fwer_panels_figure(panels, methods, out_path)


def save_multiarm_reliability_violin_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario FWER and
    best-arm power, one dot per (label, correction) -- the multi-arm analogue
    of the pairwise reliability violin. Exposes the spread the OVERALL SUMMARY
    table's pooled FWER hides: a correction with alpha-level FWER on average
    can still have scenario-specific inflation that pooling across labels
    masks, collapsed across n and k the same way the headline table is."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    palette = {m.name: m.color for m in MULTIARM_CORRECTION_METHODS}

    null_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "correction": r.correction, "fwer": r.any_reject / r.n_reps}
        for r in results if r.condition == "null" and r.n_reps > 0
    ])
    alt_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "correction": r.correction, "power": r.best_selected / r.n_reps}
        for r in results if r.condition == "alt" and r.n_reps > 0
    ])
    null_scenario = (
        null_df.groupby(["eval_type", "label", "correction"], as_index=False).agg(fwer=("fwer", "mean"))
        if not null_df.empty else null_df
    )
    alt_scenario = (
        alt_df.groupby(["eval_type", "label", "correction"], as_index=False).agg(power=("power", "mean"))
        if not alt_df.empty else alt_df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        for row_idx, (scenario_df, metric, ylabel, ref_line) in enumerate([
            (null_scenario, "fwer", "FWER per scenario", alpha),
            (alt_scenario, "power", "Best-arm power per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            et_df = scenario_df[scenario_df["eval_type"] == et] if not scenario_df.empty else scenario_df
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_corrections = [name for name in corrections if name in et_df["correction"].values]
            sns.violinplot(
                data=et_df, x="correction", y=metric, order=et_corrections, hue="correction",
                hue_order=et_corrections, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="correction", y=metric, order=et_corrections, hue="correction",
                hue_order=et_corrections, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    # x-tick labels already name each correction, but a color-key legend
    # (matching the palette used across every other multiarm plot) makes it
    # easy to cross-reference colors against those plots without having to
    # read the rotated tick labels here. Built manually via mpatches (rather
    # than pulled from the violin/strip plots, which are legend=False --
    # seaborn's own hue legend duplicates each color once per subplot,
    # which is redundant here since every subplot shares the same palette).
    legend_handles = [mpatches.Patch(facecolor=palette[c], alpha=0.5, label=c) for c in corrections]
    axes[0][-1].legend(
        handles=legend_handles, title="Correction", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\npvalues multi-arm | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Simultaneous-CI mode (non-PPI): calibration check for simultaneous
# (family-wise) confidence intervals, built two different ways:
#
# 1. `none`/`bonferroni`/`max_t` (SIMULTANEOUS_CI_METHODS) -- the three of
#    multiarm's six p-value correction strategies that have an established
#    simultaneous-CI dual: `none` (naive per-pair CI, no adjustment -- the
#    uncorrected baseline), Bonferroni t-intervals, and max-T (studentized
#    bootstrap, Romano-Wolf) -- the two non-naive constructions
#    all_pairwise's own router (_simultaneous_cis_router) picks between
#    automatically based on whether `method` is bootstrap-compatible.
#    (holm/fdr_bh/friedman_nemenyi have no CI dual -- holm/fdr_bh are
#    p-value-only adjustments, friedman_nemenyi is on the rank scale -- so
#    they're multiarm-only.) `none`/`bonferroni` are built on the scenario's
#    eval-type-canonical CI method (see point 2); `max_t` is the exception,
#    kept on --multiarm-method (bootstrap_t by default) since it needs a
#    bootstrap-compatible method to resample from, and neither Tango nor
#    Logit-t is one.
#
# 2. `sidak`/`boot` (CANONICAL_SIMULTANEOUS_CI_METHODS) -- does adjusting
#    evalstats' actual production-default pairwise CI formula for
#    multiplicity (rather than max-T/Bonferroni's generic bootstrap_t-based
#    constructions) do better? _canonical_ci_func below maps
#    each eval type to its evalstats.config.AUTO_ANALYZE_METHOD_TABLE
#    default: Tango for binary (N>=50 row; small-N bayes_binary isn't
#    alpha-parameterized the same way, so isn't modeled here), Logit-t for
#    continuous/likert (both count as the "bounded_01" data_kind once this
#    harness's own EVAL_TYPE_SCALE_BOUNDS supplies the range Logit-t needs).
#    `grades` has no entry (out of scope -- not swept by default anyway; see
#    official_args()'s eval_types). `sidak` (closed-form Sidak-adjusted
#    per-comparison alpha) and `boot` (a joint bootstrap critical value
#    substituted for the canonical CI's marginal normal quantile, which
#    accounts for correlation between comparisons the way max-T does for a
#    generic statistic) are the two ways of widening it to hold
#    family-wise. Sidak/bootstrap-scaling themselves are NOT method-specific
#    -- see evalstats.core.paired's _sidak_simultaneous_cis /
#    _joint_bootstrap_scaled_simultaneous_cis, which take any
#    alpha-parameterized per-pair CI as a `ci_func` argument; the canonical
#    formula is just the ci_func passed in below.
#
# Both reuse the SAME k-arm MultiArmSource scenarios (synthetic and real)
# that --mode multiarm sweeps, since all these questions ("which p-value
# correction", "which CI construction", "does the canonical CI benefit from
# multiplicity adjustment") share the identical underlying k-arm generative
# model -- just a different measurement per rep (coverage + width of the
# constructed simultaneous CI, instead of reject/best-arm).
# ---------------------------------------------------------------------------


def save_multiarm_violin_vs_n_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """Grouped violin plots of FWER and best-arm power vs. sample size n,
    one violin per correction at each n (dodged side by side), faceted by
    eval type -- the multiarm analogue of
    save_simultaneous_ci_violin_vs_n_plot.

    save_multiarm_reliability_violin_plot already shows the per-scenario
    spread, but collapses n away, so a correction that is badly calibrated
    only at small n looks merely "wide" there. This plot separates the two:
    a correction whose violins march upward with n is converging, while one
    whose violins stay wide at every n is unreliable regardless of sample
    size -- a distinction the pooled violin cannot draw.

    Each violin pools every (scenario, k) cell at that n rather than
    averaging k away, since the small-n/large-k interaction is exactly what
    the FWER corrections differ on.

    Drops `none` (see MULTIARM_PLOT_METHODS): uncorrected FWER runs so far
    above nominal that it squashes the y-axis and hides the comparison
    between the corrections this plot exists to make. It remains in the
    report tables and the CSV.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    corrections = [m.name for m in MULTIARM_PLOT_METHODS if m.name in {r.correction for r in results}]
    palette = {m.name: m.color for m in MULTIARM_PLOT_METHODS}
    plot_names = {m.name for m in MULTIARM_PLOT_METHODS}

    rows = []
    for r in results:
        if r.n_reps <= 0 or r.correction not in plot_names:
            continue
        if r.condition == "null":
            rows.append({"eval_type": r.eval_type, "n": r.n, "correction": r.correction,
                         "metric": "fwer", "value": r.any_reject / r.n_reps})
        elif r.condition == "alt":
            rows.append({"eval_type": r.eval_type, "n": r.n, "correction": r.correction,
                         "metric": "power", "value": r.best_selected / r.n_reps})
    df = pd.DataFrame(rows)

    n_cols = max(len(eval_types_present), 1)
    if df.empty:
        fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
        for ax_row in axes:
            for ax in ax_row:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    ns_present = sorted(df["n"].unique())
    n_order = [str(n) for n in ns_present]
    df["n_label"] = df["n"].astype(str)

    # Width has to scale with the HUE count, not just the number of n groups:
    # this case plots ~10 corrections per group where the simultaneous-CI
    # analogue plots 4, and a fixed per-group width squeezes each violin into
    # an unreadable sliver. 0.30in per (correction x n) reproduces the
    # simultaneous-CI plot's proportions at its own 4-method width.
    col_width = max(1.3, 0.30 * len(corrections)) * len(ns_present) + 2.5
    fig, axes = plt.subplots(2, n_cols, figsize=(col_width * n_cols, 9.0), squeeze=False)
    legend_handles = [mpatches.Patch(facecolor=palette[m], alpha=0.5, label=m) for m in corrections]

    for col_idx, et in enumerate(eval_types_present):
        et_df = df[df["eval_type"] == et]
        for row_idx, (metric, ylabel, ref_line) in enumerate([
            ("fwer", "FWER (null)", alpha),
            ("power", "Best-arm selection power (alt)", None),
        ]):
            ax = axes[row_idx][col_idx]
            m_df = et_df[et_df["metric"] == metric]
            et_methods = [name for name in corrections if name in m_df["correction"].values]
            if m_df.empty or not et_methods:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            sns.violinplot(
                data=m_df, x="n_label", y="value", order=n_order, hue="correction",
                hue_order=et_methods, palette=palette, cut=0, inner="quartile",
                linewidth=0.7, dodge=True, alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=m_df, x="n_label", y="value", order=n_order, hue="correction",
                hue_order=et_methods, palette=palette, size=3, alpha=0.5, jitter=0.2,
                dodge=True, linewidth=0.3, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("n" if row_idx == 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")

    axes[0][-1].legend(
        handles=legend_handles, title="Correction", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )
    fig.suptitle(
        "Family-Wise Error Rate and Best-Arm Power vs. Sample Size\n"
        f"Nominal alpha = {alpha}; each violin pools all $k$ and scenarios at that $n$",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _canonical_ci_func(eval_type: str):
    """The alpha-parameterized ci_func for evalstats' canonical pairwise CI
    at this eval type, or ``None`` if there isn't one.

    DELEGATES to evalstats.core.paired.canonical_pairwise_ci_func -- the same
    call the library's own simultaneous-CI router makes -- so this harness
    always measures the formula that actually ships. It used to re-list the
    formulas here, which silently went stale twice: Likert stayed on logit-t
    after it gained its own NIG row, and binary stayed on mj_floor after
    binary moved to Bonett-Price. The published simultaneous-CI numbers for
    those two data kinds were therefore measured on intervals evalstats no
    longer reports.

    Returns ``None`` for "grades" and anything else with no bounded scale.
    """
    from evalstats.core.paired import canonical_pairwise_ci_func

    if eval_type == "binary":
        return canonical_pairwise_ci_func("binary", None)
    if eval_type in EVAL_TYPE_SCALE_BOUNDS:
        scale_lo, scale_hi = EVAL_TYPE_SCALE_BOUNDS[eval_type]
        diff_span = scale_hi - scale_lo
        data_kind = "likert" if eval_type == "likert" else "bounded_01"
        return canonical_pairwise_ci_func(data_kind, (-diff_span, diff_span))
    return None


@dataclass
class SimultaneousCIResult:
    """One (eval_type, source, n, k, CI method) cell's family-wise
    coverage/width outcome from the simultaneous-CI sweep."""

    eval_type: str
    label: str
    n: int
    k: int
    ci_method: str  # "none" | "bonferroni" | "max_t" | "sidak" | "boot" ("sidak"/"boot" absent when _canonical_ci_func(eval_type) is None, e.g. "grades")
    condition: str  # "null" | "alt"
    n_reps: int
    all_covered: int
    """Count of reps where EVERY one of the k(k-1)/2 pairwise simultaneous
    CIs simultaneously contained its true difference (the family-wise
    coverage event -- the CI-construction analogue of multiarm's
    any_reject/FWER, just measuring the opposite: a miss on ANY pair, not a
    false rejection on any pair)."""
    total_width: float
    """Sum, across reps, of that rep's MEAN CI width across all k(k-1)/2
    pairs -- dividing by n_reps gives the average per-comparison width,
    comparable across different k and n."""
    total_width_sq: float = 0.0
    """Sum of the SQUARES of the same per-rep mean widths, so the width
    curves can carry a Monte Carlo band like the coverage curves do.
    Coverage is a proportion and its MC error follows from the count alone;
    a mean width does not, and no standard error is recoverable from
    `total_width` by itself. Defaults to 0.0 so results CSVs written before
    this field existed still load -- plots treat a zero sum as "no variance
    recorded" and simply omit the band rather than drawing a zero-width
    one, which would falsely read as a perfectly-determined mean."""
    total_score: float = 0.0
    """Sum, across reps, of that rep's FAMILY-WISE interval score: mean CI
    width across all k(k-1)/2 pairs, plus (2/alpha) * the WORST pair's miss
    distance (0 iff all_covered). Deliberately not the mean of each pair's own
    interval_score() (see evalstats.core.stats_utils) -- that marginal,
    per-comparison version rewards `none` even when its family-wise coverage
    collapses, since each of its individual intervals is close to nominally
    calibrated on its own. Using the worst pair's miss distance ties the
    penalty to the same "did ANY pair miss" event that all_covered measures."""
    total_time: float = 0.0
    """Total wall-clock seconds for THIS method's own construction, summed
    across all n_reps of this condition -- e.g. bonferroni's own
    _bonferroni_simultaneous_cis() call, max_t's own _simultaneous_cis_router()
    call (which includes its bootstrap resampling), etc. Does NOT include
    shared per-rep setup (score generation, building matrix_raw) except for
    `none`, which that setup is attributed to (see _run_simultaneous_ci_cell)."""


def _run_simultaneous_ci_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed, ci_methods: list[str] | None = None,
) -> list[SimultaneousCIResult]:
    """Run n_reps replications of a k-arm simultaneous-CI sweep at one
    (source, n, k) cell, across every requested CI method. One
    SimultaneousCIResult per method."""
    labels = [f"arm_{i}" for i in range(k_arms)]
    pairs = [(labels[i], labels[j]) for i in range(k_arms) for j in range(i + 1, k_arms)]
    ci = 1.0 - alpha
    rng = np.random.default_rng(seed)

    # ci_func is the eval-type-canonical CI formula (Tango for binary,
    # Logit-t for continuous/likert; None for grades or anything else --
    # see _canonical_ci_func). When present, it replaces --multiarm-method
    # as the basis for `none` (and feeds `sidak`/`boot`, which don't exist
    # without a canonical formula to widen); `bonferroni` is unaffected
    # either way (_bonferroni_simultaneous_cis always builds its own
    # generic t-interval from per_input_diffs, never a per-method formula,
    # so it never depended on --multiarm-method to begin with); `max_t`
    # keeps using --multiarm-method (bootstrap_t by default) always, since
    # it needs a bootstrap-compatible method to resample from and neither
    # Tango nor Logit-t is one.
    ci_func = _canonical_ci_func(source.eval_type)
    has_canonical = ci_func is not None
    # Same diff span the canonical ci_func above is built on -- a difference of
    # two [lo, hi] scores ranges over [-(hi-lo), hi-lo]. Consumed only by
    # _bonferroni_simultaneous_cis' zero-variance branch, where it is the
    # difference between a conservative interval and an infinite one. Comes
    # from EVAL_TYPE_SCALE_BOUNDS rather than from has_canonical, so "grades"
    # (no canonical ci_func modeled here, but a known [0, 100] scale) still
    # gets a finite bound on that branch.
    _bonf_diff_bounds = None
    if source.eval_type in EVAL_TYPE_SCALE_BOUNDS:
        _s_lo, _s_hi = EVAL_TYPE_SCALE_BOUNDS[source.eval_type]
        _bonf_diff_bounds = (-(_s_hi - _s_lo), _s_hi - _s_lo)
    base_methods = [m.name for m in SIMULTANEOUS_CI_METHODS]
    canonical_methods = [m.name for m in CANONICAL_SIMULTANEOUS_CI_METHODS] if has_canonical else []
    all_methods = base_methods + canonical_methods
    if ci_methods is not None:
        requested = set(ci_methods)
        all_methods = [m for m in all_methods if m in requested]
    # `max_t`/`boot` each pay for their own independent bootstrap resample
    # (unlike --mode multiarm, they don't share one here -- see
    # _joint_bootstrap_critical_value vs _max_stat_simultaneous_cis), so
    # skipping either one when it's not requested is a real cost saving, not
    # just a bookkeeping nicety. `none`/`bonferroni`/`sidak` are all cheap
    # closed-form constructions regardless, but are gated the same way for
    # consistency -- a method absent from `all_methods` never appears in
    # the returned SimultaneousCIResult rows either way (see the loop over
    # `all_methods` below), so gating its computation here changes runtime,
    # not results.
    need = {m: (m in all_methods) for m in ("none", "bonferroni", "max_t", CORR_SIDAK.name, CORR_BOOT.name, CORR_BOOT_CAL.name)}
    agg_covered: dict[tuple[str, str], int] = {(m, cond): 0 for m in all_methods for cond in ("null", "alt")}
    agg_width: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}
    agg_width_sq: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}
    agg_score: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}
    # Per-(method, condition), not a single per-condition total -- each
    # construction's own wall-clock cost, so e.g. `boot`'s extra joint
    # bootstrap resampling actually shows up as slower than `sidak`'s
    # closed-form widening in the report's Time(ms) column, instead of every
    # method row displaying the same aggregate "whole rep" time.
    agg_time: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                # Score generation + matrix_raw + `none` share one timer,
                # attributed to `none` -- matrix_raw is the shared
                # prerequisite `none` (and, when there's no canonical
                # ci_func, only `none`) actually needs to exist; it's not
                # fairly attributable to any other single method.
                _t_none0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                true_means = source.true_means(k_arms, delta)

                # Raw per-pair results, for per_input_diffs (method-invariant:
                # always scores.mean(axis=2)[a] - scores.mean(axis=2)[b],
                # identical bit-for-bit regardless of which method computes
                # it -- verified against bootstrap_t's own seeded/non-seeded
                # construction) feeding Bonferroni's t-interval formula and,
                # when there's no canonical ci_func for this eval type,
                # `none` too. When a canonical ci_func DOES exist, built with
                # the cheapest closed-form method (t_interval -- no bootstrap
                # resampling at all) rather than --multiarm-method (bootstrap_t
                # by default): none/bonferroni/sidak/boot below never read
                # this matrix's own .ci_low/.ci_high in that case (only
                # per_input_diffs), and max_t's resampling reads `scores`
                # directly (see below), so paying for --multiarm-method's
                # expensive k(k-1)/2 independent nested double bootstrap here
                # would be pure waste. Falls back to --multiarm-method when
                # there's no canonical ci_func (e.g. "grades"), since `none`
                # genuinely needs that method's own CI there. Always built
                # (whenever anything at all is requested) since it's cheap
                # (t_interval, no bootstrap) and max_t's router below wants
                # it as a fallback safety net regardless of whether `none`
                # itself was requested.
                none_cis: dict = {}
                if any(need.values()):
                    matrix_raw = all_pairwise(
                        scores=scores, labels=labels, method=("t_interval" if has_canonical else multiarm_method), ci=ci,
                        n_bootstrap=n_bootstrap, correction="none", rng=rng, statistic=statistic,
                        simultaneous_ci=False,
                    )
                    if need["none"]:
                        if has_canonical:
                            # The canonical formula's own naive CI at the plain
                            # (unadjusted) alpha IS the "none" construction here --
                            # mathematically identical to what all_pairwise(method=
                            # "tango"/"logit_t") would compute for .ci_low/.ci_high,
                            # but derived directly from ci_func so continuous/likert
                            # get this harness's own EVAL_TYPE_SCALE_BOUNDS-derived
                            # diff span rather than all_pairwise's [0, 1]-diff-span
                            # default (which would silently mis-scale likert's
                            # [1, 5] range without an explicit score_range).
                            none_cis = {
                                pair: ci_func(matrix_raw.results[pair].per_input_diffs, alpha)
                                for pair in pairs
                            }
                        else:
                            none_cis = {pair: (matrix_raw.get(*pair).ci_low, matrix_raw.get(*pair).ci_high) for pair in pairs}
                if need["none"]:
                    agg_time[("none", condition)] += time.perf_counter() - _t_none0

                bonf_cis: dict = {}
                if need["bonferroni"]:
                    _t0 = time.perf_counter()
                    bonf_cis = _bonferroni_simultaneous_cis(
                        results=matrix_raw.results, pairs=pairs, ci=ci,
                        diff_bounds=_bonf_diff_bounds,
                    )
                    agg_time[("bonferroni", condition)] += time.perf_counter() - _t0

                # max-T: call _simultaneous_cis_router directly (the same
                # function all_pairwise(simultaneous_ci=True) would call
                # internally). Its actual max-T computation
                # (_max_stat_simultaneous_cis) resamples straight from
                # `scores` under --multiarm-method and never reads `results`
                # at all -- `results` is only consulted for the router's
                # rare Bonferroni-fallback safety net on degenerate data,
                # which matrix_raw.results above already supplies (cheaply,
                # when has_canonical). This is --multiarm-method's (bootstrap_t
                # by default) one remaining unavoidable cost in this cell:
                # a single shared max-T resample, not the k(k-1)/2
                # independent nested double bootstraps matrix_raw used to
                # pay for before this cheap-when-canonical rework. Skipped
                # entirely (its own independent bootstrap resample, the
                # single most expensive part of this cell alongside boot's)
                # when max_t isn't requested via --ci-methods.
                maxt_cis: dict = {}
                if need["max_t"]:
                    _t0 = time.perf_counter()
                    sim_cis, sim_method, _ = _simultaneous_cis_router(
                        scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                        method=multiarm_method, ci=ci, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                        prefer="max_t",
                    )
                    if sim_method == "max_t":
                        maxt_cis = sim_cis
                    agg_time[("max_t", condition)] += time.perf_counter() - _t0

                # sidak/boot: widen the canonical formula (ci_func) itself
                # for multiplicity, instead of Bonferroni/max-T's generic
                # bootstrap_t-based constructions. The widening machinery
                # (_sidak_simultaneous_cis / _joint_bootstrap_scaled_
                # simultaneous_cis) has no idea what "Tango" or "Logit-t"
                # is -- it's generic over any alpha-parameterized ci_func;
                # ci_func is just whichever canonical formula this
                # scenario's eval type resolves to. per_input_diffs is
                # method-agnostic, so matrix_raw.results -- built above
                # under multiarm_method -- is reused here rather than
                # re-running all_pairwise a second time.
                sidak_cis: dict = {}
                if has_canonical and need[CORR_SIDAK.name]:
                    _t0 = time.perf_counter()
                    sidak_cis = _sidak_simultaneous_cis(
                        results=matrix_raw.results, pairs=pairs, ci=ci, ci_func=ci_func,
                    )
                    agg_time[(CORR_SIDAK.name, condition)] += time.perf_counter() - _t0

                # boot: its own independent joint bootstrap resample (see
                # _joint_bootstrap_critical_value) -- unlike --mode multiarm,
                # this is NOT shared with max_t here, so skipping it when not
                # requested (and max_t IS requested, or vice versa) is a real
                # saving, not just symmetry with the gating above.
                boot_cis: dict = {}
                if has_canonical and need[CORR_BOOT.name]:
                    _t0 = time.perf_counter()
                    boot_cis = _joint_bootstrap_scaled_simultaneous_cis(
                        scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                        ci=ci, n_bootstrap=n_bootstrap, rng=rng, ci_func=ci_func, statistic=statistic,
                    )
                    agg_time[(CORR_BOOT.name, condition)] += time.perf_counter() - _t0

                # boot_cal: same joint-bootstrap idea as `boot`, but the
                # critical value is studentized by ci_func's OWN centre and
                # scale per replicate rather than by the bootstrap SE, so the
                # resulting level absorbs whatever finite-sample behaviour the
                # formula has (Bonett-Price is marginally conservative by up
                # to +4.3pp at n=10, which plain `boot` inherits). Its own
                # resample, like `boot`'s -- gated the same way.
                boot_cal_cis: dict = {}
                if has_canonical and need[CORR_BOOT_CAL.name]:
                    _t0 = time.perf_counter()
                    boot_cal_cis = _calibrated_joint_simultaneous_cis(
                        scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                        ci=ci, n_bootstrap=n_bootstrap, rng=rng, ci_func=ci_func, statistic=statistic,
                    )
                    agg_time[(CORR_BOOT_CAL.name, condition)] += time.perf_counter() - _t0

                for method_name, cis in (
                    ("none", none_cis), ("bonferroni", bonf_cis), ("max_t", maxt_cis),
                    (CORR_SIDAK.name, sidak_cis), (CORR_BOOT.name, boot_cis),
                    (CORR_BOOT_CAL.name, boot_cal_cis),
                ):
                    if not cis:
                        continue
                    widths: list[float] = []
                    miss_distances: list[float] = []
                    covered_all = True
                    for (label_a, label_b) in pairs:
                        idx_a, idx_b = labels.index(label_a), labels.index(label_b)
                        true_diff = true_means[idx_a] - true_means[idx_b]
                        lo, hi = cis[(label_a, label_b)]
                        widths.append(hi - lo)
                        if true_diff < lo:
                            miss_distances.append(lo - true_diff)
                        elif true_diff > hi:
                            miss_distances.append(true_diff - hi)
                        else:
                            miss_distances.append(0.0)
                        if not (lo <= true_diff <= hi):
                            covered_all = False
                    # Family-wise interval score: mean width (still a legitimate
                    # per-comparison cost) + (2/alpha) * the WORST pair's miss
                    # distance, not the mean of each pair's own miss distance.
                    # interval_score() alone is a marginal, per-comparison proper
                    # score -- averaging it pair-by-pair rewards "none" (whose
                    # individual pairs are each ~nominally calibrated on their
                    # own) even when its family-wise coverage collapses, since
                    # the penalty never triggers on the same "did ANY pair miss"
                    # event that all_covered measures. Using max ties the miss
                    # penalty to that exact event: it's 0 iff all_covered, and
                    # positive iff at least one pair missed, so a method that
                    # buys family-wise coverage by widening every interval is no
                    # longer penalized as if it were miscalibrated per-pair.
                    family_score = float(np.mean(widths)) + (2.0 / alpha) * (max(miss_distances) if miss_distances else 0.0)
                    _mean_width = float(np.mean(widths)) if widths else 0.0
                    agg_width[(method_name, condition)] += _mean_width
                    agg_width_sq[(method_name, condition)] += _mean_width ** 2
                    agg_score[(method_name, condition)] += family_score
                    if covered_all:
                        agg_covered[(method_name, condition)] += 1

    return [
        SimultaneousCIResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, ci_method=method_name,
            condition=condition, n_reps=n_reps, all_covered=agg_covered[(method_name, condition)],
            total_width=agg_width[(method_name, condition)],
            total_width_sq=agg_width_sq[(method_name, condition)],
            total_score=agg_score[(method_name, condition)],
            total_time=agg_time[(method_name, condition)],
        )
        for method_name in all_methods
        for condition in ("null", "alt")
    ]


_SIMULTANEOUS_CI_SOURCES: list = []  # fork-inherited worker state for run_simultaneous_ci_simulation


def _run_simultaneous_ci_cell_worker(args: tuple) -> list[SimultaneousCIResult]:
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, ci_methods = args
    return _run_simultaneous_ci_cell(
        _SIMULTANEOUS_CI_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed,
        ci_methods=ci_methods,
    )


def run_simultaneous_ci_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_values: list[int], n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar",
    seed: int = 42, n_workers: int = 1, ci_methods: list[str] | None = None,
) -> list[SimultaneousCIResult]:
    """Sweep _run_simultaneous_ci_cell over every (source, sample size, k)
    cell, parallelized across n_workers, and flatten the per-cell
    SimultaneousCIResult lists into one list."""
    global _SIMULTANEOUS_CI_SOURCES
    _SIMULTANEOUS_CI_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, ci_methods)
                 for (sc_idx, n, k), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-simultaneous_ci")
    results: list[SimultaneousCIResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_simultaneous_ci_cell_worker(a))
            sc_idx, n, k = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n} k={k}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_simultaneous_ci_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def _time_stats_simultaneous_ci(results: list[SimultaneousCIResult]) -> tuple[float, float]:
    """Average ± SE of wall-clock time per rep in milliseconds across cells."""
    valid = [r for r in results if r.total_time > 0 and r.n_reps > 0]
    if not valid:
        return float("nan"), float("nan")
    per_rep_ms = [r.total_time * 1000.0 / r.n_reps for r in valid]
    avg = float(np.mean(per_rep_ms))
    se = float(np.std(per_rep_ms, ddof=1) / np.sqrt(len(per_rep_ms))) if len(per_rep_ms) > 1 else 0.0
    return avg, se


def print_simultaneous_ci_report(results: list[SimultaneousCIResult], alpha: float) -> None:
    """Print the console family-wise-coverage report for a simultaneous-CI
    run, grouped by eval type and k."""
    target = 1.0 - alpha
    print(f"\n{'='*78}\n  PVALUES (SIMULTANEOUS CI) -- none vs. BONFERRONI vs. max-T vs. Tango variants\n"
          f"  Nominal family-wise coverage: {target:.0%}\n{'='*78}")
    ci_methods = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ks_present = sorted({r.k for r in results})

    for et in eval_types_present:
        for k in ks_present:
            subset = [r for r in results if r.eval_type == et and r.k == k]
            if not subset:
                continue
            print(f"\n  [{et}, k={k}]")
            print(f"    {'CI method':<12} {'Cov(null)':>10} {'Width(null)':>12} {'Score(null)':>12} {'Cov(alt)':>10} {'Width(alt)':>12} {'Score(alt)':>11}")
            for cm in ci_methods:
                c_rows = [r for r in subset if r.ci_method == cm]
                null_rows = [r for r in c_rows if r.condition == "null"]
                alt_rows = [r for r in c_rows if r.condition == "alt"]
                t_null = sum(r.n_reps for r in null_rows)
                c_null = sum(r.all_covered for r in null_rows)
                w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
                s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
                t_alt = sum(r.n_reps for r in alt_rows)
                c_alt = sum(r.all_covered for r in alt_rows)
                w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
                s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
                cov_null = c_null / t_null if t_null > 0 else float("nan")
                cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
                print(f"    {cm:<12} {cov_null:>10.3f} {w_null:>12.4f} {s_null:>12.4f} {cov_alt:>10.3f} {w_alt:>12.4f} {s_alt:>11.4f}")

    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY (collapsed across eval types, sources, n, k)", results, ci_methods, target,
    )
    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY -- LOW N (n <= 30)", [r for r in results if r.n <= 30], ci_methods, target,
    )
    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY -- HIGH N (n >= 30)", [r for r in results if r.n >= 30], ci_methods, target,
    )


def _print_simultaneous_overall_summary_table(
    title: str, results: list[SimultaneousCIResult], ci_methods: list[str], target: float,
) -> None:
    """One OVERALL SUMMARY table for print_simultaneous_ci_report, over
    whatever subset of `results` the caller passes in (e.g. all of them, or
    just the low-N / high-N slice -- see that function's low-N vs. high-N
    split, which exists because max-T's bootstrap_t studentization is
    well-behaved at large N but develops a random-denominator instability at
    small N with many simultaneous pairs; pooling both regimes into one
    table hides exactly that crossover)."""
    ks_present = sorted({r.k for r in results})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")
    print(f"  MinCov = worst per-scenario family-wise coverage seen for that CI method (not\n"
          f"  an average) -- flags methods whose good mean coverage hides an unreliable\n"
          f"  scenario/n/k cell.")
    n_cols = "".join(f"  {'n='+str(n):>9}" for n in sizes_present)
    k_cols = "".join(f"  {'k='+str(k):>8}" for k in ks_present)
    print(f"\n  {'CI method':<12}  {'Cov(null)':>9}  {'MinCov':>7}  {'Band95':>13}  {'Width(null)':>11}  {'Score(null)':>12}  "
          f"{'Cov(alt)':>8}  {'Width(alt)':>10}  {'Score(alt)':>11}  {'Time(ms)':>14}{n_cols}{k_cols}")
    for cm in ci_methods:
        c_rows = [r for r in results if r.ci_method == cm]
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        t_null = sum(r.n_reps for r in null_rows)
        c_null = sum(r.all_covered for r in null_rows)
        if t_null == 0:
            continue
        w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
        s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
        t_alt = sum(r.n_reps for r in alt_rows)
        c_alt = sum(r.all_covered for r in alt_rows)
        w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        cov_null = c_null / t_null if t_null > 0 else float("nan")
        cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        avg_ms, se_ms = _time_stats_simultaneous_ci(null_rows)
        time_str = f"{avg_ms:.1f}+-{se_ms:.1f}" if np.isfinite(avg_ms) else "-"
        marker = "*" if np.isfinite(cov_null) and abs(cov_null - target) > 0.02 else " "
        per_label_cov = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_cov[(r.eval_type, r.label)]
            acc[0] += r.all_covered
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_cov.values() if t > 0]
        worst_cov = min(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_cov:.3f}{'*' if np.isfinite(worst_cov) and abs(worst_cov - target) > 0.02 else ' '}" if np.isfinite(worst_cov) else "-"
        n_cells = ""
        for n in sizes_present:
            n_null = [r for r in null_rows if r.n == n]
            nc = sum(r.all_covered for r in n_null)
            nt = sum(r.n_reps for r in n_null)
            nf = nc / nt if nt > 0 else float("nan")
            n_cells += f"  {nf:>9.3f}" if np.isfinite(nf) else f"  {'  -':>9}"
        k_cells = ""
        for k in ks_present:
            k_null = [r for r in null_rows if r.k == k]
            kc = sum(r.all_covered for r in k_null)
            kt = sum(r.n_reps for r in k_null)
            kf = kc / kt if kt > 0 else float("nan")
            k_cells += f"  {kf:>8.3f}" if np.isfinite(kf) else f"  {'  -':>8}"
        print(f"  {cm:<12}  {cov_null:>8.3f}{marker}  {worst_str:>7}  {band:>13}  {w_null:>11.4f}  {s_null:>12.4f}  "
              f"{cov_alt:>8.3f}  {w_alt:>10.4f}  {s_alt:>11.4f}  {time_str:>14}{n_cells}{k_cells}")
    print(f"  (* = |coverage - nominal| > 0.02; narrower Width/Score at matching coverage is better)")
    print()


def latex_simultaneous_ci_overall_summary(
    results: list[SimultaneousCIResult], alpha: float, *,
    label_suffix: str = "", caption_suffix: str = "",
    condition: str | None = None, include_uncorrected: bool = True,
) -> str:
    """LaTeX booktabs overall summary: per-CI-method family-wise coverage
    (null, with its 95% MC band) + average width (null and alt), collapsed
    across eval types, plus one coverage column per sample size actually
    swept. `none` should visibly under-cover (no simultaneous adjustment at
    all, even though it's already built on evalstats' canonical per-eval-type
    CI -- see _canonical_ci_func); `bonferroni`/`max_t`/`sidak`/`boot` should
    all hit nominal coverage, so the tie-breaker between them is which gets
    there with a narrower average CI/interval score. *results* may be a
    filtered subset (e.g. n<=30 / n>=30 -- see latex_simultaneous_ci_full_report)
    with *label_suffix*/*caption_suffix* set so multiple calls in one
    document don't collide on \\label{}."""
    target = 1.0 - alpha
    # `include_uncorrected=False` drops the `none` baseline, matching what the
    # plots already do (SIMULTANEOUS_CI_PLOT_METHODS). It is so far below
    # nominal that its row shades saturated red across every column, which
    # dominates the table visually while only restating that some correction
    # is needed. Kept by default so the raw run log still carries the
    # baseline; the paper tables pass False.
    pool = ALL_SIMULTANEOUS_CI_METHODS if include_uncorrected else SIMULTANEOUS_CI_PLOT_METHODS
    ci_methods = [m.name for m in pool if m.name in {r.ci_method for r in results}]
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    ks_present = sorted({r.k for r in results if r.condition == "null"})

    method_groups: dict[str, set[str]] = defaultdict(set)
    for r in results:
        if r.ci_method not in ci_methods:
            continue
        method_groups[r.ci_method].add(report_eval_type_group(r.eval_type))
    groups_present = sort_groups({g for gs in method_groups.values() for g in gs})

    rows = []
    rule_before = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        scores = []
        for cm in ci_methods:
            if g not in method_groups[cm]:
                continue
            c_rows = [r for r in results
                      if r.ci_method == cm and report_eval_type_group(r.eval_type) == g]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
            s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
            t_alt = sum(r.n_reps for r in alt_rows)
            c_alt = sum(r.all_covered for r in alt_rows)
            w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
            s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
            cov_null = c_null / t_null if t_null > 0 else float("nan")
            cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
            _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
            label = f"{escape_latex(cm)} ({g})" if len(method_groups[cm]) > 1 else escape_latex(cm)
            row = [
                label,
                *( [coverage_cell(cov_null, target),
                    f"{w_null:.4f}" if np.isfinite(w_null) else "-",
                    f"{s_null:.4f}" if np.isfinite(s_null) else "-"]
                   if condition in (None, "null") else [] ),
                *( [coverage_cell(cov_alt, target),
                    f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
                    f"{s_alt:.4f}" if np.isfinite(s_alt) else "-"]
                   if condition in (None, "alt") else [] ),
                g,
            ]
            per_cond = alt_rows if condition == "alt" else null_rows
            for n in sizes_present:
                n_rows = [r for r in per_cond if r.n == n]
                t_n = sum(r.n_reps for r in n_rows)
                row.append(coverage_cell(
                    sum(r.all_covered for r in n_rows) / t_n if t_n > 0 else float("nan"), target))
            for k in ks_present:
                k_rows = [r for r in per_cond if r.k == k]
                t_k = sum(r.n_reps for r in k_rows)
                row.append(coverage_cell(
                    sum(r.all_covered for r in k_rows) / t_k if t_k > 0 else float("nan"), target))
            rows.append(row)
            scores.append(s_null)

        # This family measures coverage and width, so it takes the CI tables'
        # treatment wholesale: coverage shading plus best/runner-up on the
        # interval score, which already trades the two off (Gneiting &
        # Raftery). Ranked within the eval-type block, since widths and
        # scores live on different scales per type.
        SCORE_NULL_COL = 3 if condition is not None else 3
        block = rows[block_start:]
        marked = mark_best_and_runnerup([r[SCORE_NULL_COL] for r in block], scores)
        for row, cell in zip(block, marked):
            row[SCORE_NULL_COL] = cell

    return booktabs_table(
        caption=f"pvalues (simultaneous CI): family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score -- "
                f"{'none/' if include_uncorrected else ''}bonferroni/max\\_t (generic, "
                f"\\texttt{{--multiarm-method}}-based, bootstrap\\_t by default) vs. sidak/boot "
                f"(Sidak- and joint-bootstrap-scaled widenings of evalstats' canonical per-eval-type "
                f"CI: Tango for binary, Logit-t for continuous/likert){caption_suffix} "
                f"(nominal coverage={target:.0%}). Methods run on more than one eval type get one "
                f"row per type (bin/cont/lik), grouped into blocks, so no row averages across "
                f"types -- pooling hides that max\\_t's Cov(alt) is fine on continuous/likert but "
                f"collapses on binary, where its symmetric studentized interval is the wrong shape "
                f"for a difference of proportions with a real effect at small $n$.",
        label=f"tab:pvalues_simultaneous_ci_overall{label_suffix}",
        columns=["CI method"]
                + (["Cov(null)", "Width(null)", "Score(null)"] if condition in (None, "null") else [])
                + (["Cov(alt)", "Width(alt)", "Score(alt)"] if condition in (None, "alt") else [])
                + ["Type"]
                + [f"n={n}" for n in sizes_present]
                + [f"k={k}" for k in ks_present],
        rows=rows,
        rule_before=rule_before,
    )


def latex_simultaneous_ci_by_eval_type_summary(results: list[SimultaneousCIResult], alpha: float) -> str:
    """LaTeX booktabs summary faceted by eval type instead of collapsed
    across them: one row per (eval type, CI method), collapsed across n and
    k. Complements latex_simultaneous_ci_overall_summary -- `sidak`/`boot`
    widen a DIFFERENT canonical CI per eval type (Tango for binary, Logit-t
    for continuous/likert; see _canonical_ci_func), so this is the table
    that shows the effect holds for each formula separately rather than
    only in a pooled average that could be dominated by whichever eval type
    has the most scenarios/sizes swept."""
    target = 1.0 - alpha
    ci_methods = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]

    rows = []
    rule_before = set()
    for et in eval_types_present:
        et_results = [r for r in results if r.eval_type == et]
        et_methods = [cm for cm in ci_methods if any(r.ci_method == cm for r in et_results)]
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        block_scores = []
        for cm in et_methods:
            c_rows = [r for r in et_results if r.ci_method == cm]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
            s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
            t_alt = sum(r.n_reps for r in alt_rows)
            c_alt = sum(r.all_covered for r in alt_rows)
            w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
            s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
            cov_null = c_null / t_null if t_null > 0 else float("nan")
            cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
            _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
            rows.append([
                escape_latex(et), escape_latex(cm),
                coverage_cell(cov_null, target),
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{w_null:.4f}" if np.isfinite(w_null) else "-",
                f"{s_null:.4f}" if np.isfinite(s_null) else "-",
                coverage_cell(cov_alt, target),
                f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
                f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            ])
            block_scores.append(s_null)

        # Rank Score within each eval-type block, not across the whole
        # table: widths and scores live on different scales per eval type
        # (a Likert difference spans 4 points, a binary one spans 1), so a
        # global "best score" would just pick whichever eval type has the
        # narrowest scale.
        SCORE_NULL_COL = 5
        block = rows[block_start:]
        marked = mark_best_and_runnerup([r[SCORE_NULL_COL] for r in block], block_scores)
        for row, cell in zip(block, marked):
            row[SCORE_NULL_COL] = cell

    return booktabs_table(
        caption=f"pvalues (simultaneous CI): family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score, faceted by eval type "
                f"(nominal coverage={target:.0%}). Coverage cells shade red when below nominal and "
                f"blue when over-conservative; best and runner-up Score(null) are marked within "
                f"each eval-type block.",
        label="tab:pvalues_simultaneous_ci_by_eval_type",
        columns=["Eval type", "CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)"],
        rows=rows,
        rule_before=rule_before,
    )


def latex_simultaneous_ci_full_report(results: list[SimultaneousCIResult], alpha: float) -> str:
    """All simultaneous_ci LaTeX tables for this run, concatenated and ready
    to paste into a report: the pooled overall summary, the same summary
    split into low-N (n<=30) / high-N (n>=30) subsets (the mode's headline
    max-T-crossover finding -- see print_simultaneous_ci_report's LOW N /
    HIGH N split), and the by-eval-type facet (showing sidak/boot's effect
    holds separately for Tango (binary) and Logit-t (continuous/likert), not
    just in a pooled average)."""
    return "\n\n".join([
        latex_simultaneous_ci_overall_summary(results, alpha),
        latex_simultaneous_ci_overall_summary(
            [r for r in results if r.n <= 30], alpha,
            label_suffix="_lown", caption_suffix=", low-N ($n \\le 30$) subset",
        ),
        latex_simultaneous_ci_overall_summary(
            [r for r in results if r.n >= 30], alpha,
            label_suffix="_highn", caption_suffix=", high-N ($n \\ge 30$) subset",
        ),
        latex_simultaneous_ci_by_eval_type_summary(results, alpha),
    ])


def save_results_artifacts_simultaneous_ci(
    *, results: list[SimultaneousCIResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    """Write the simultaneous-CI run's results CSV (and LaTeX summary if
    `latex=True`) under out_dir. Returns the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_simultaneous_ci_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # width_sd is the per-rep SD behind avg_width. The other columns are
        # per-rep averages, from which no spread is recoverable -- without
        # this one, anything rebuilt from the CSV (rather than from the live
        # result objects) could not draw the width plots' Monte Carlo band.
        writer.writerow(["eval_type", "label", "n", "k", "ci_method", "condition", "n_reps", "all_covered", "coverage_rate", "avg_width", "width_sd", "avg_score", "total_time_s", "time_ms_per_rep"])
        for r in results:
            time_ms = (r.total_time * 1000.0 / r.n_reps) if r.n_reps > 0 and r.total_time > 0 else float("nan")
            mean_w = r.total_width / r.n_reps if r.n_reps > 0 else float("nan")
            width_sd = (
                math.sqrt(max(r.total_width_sq / r.n_reps - mean_w ** 2, 0.0))
                if r.n_reps > 0 and r.total_width_sq > 0 else float("nan")
            )
            writer.writerow([
                r.eval_type, r.label, r.n, r.k, r.ci_method, r.condition, r.n_reps, r.all_covered,
                f"{r.all_covered / r.n_reps:.8f}", f"{mean_w:.8f}",
                f"{width_sd:.8f}" if width_sd == width_sd else "",
                f"{r.total_score / r.n_reps:.8f}",
                f"{r.total_time:.6f}", f"{time_ms:.4f}" if not (time_ms != time_ms) else "",
            ])
    summary_path = out_base / f"{run_stem}_simultaneous_ci_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_simultaneous_ci_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX tables (--latex): overall, low-N, high-N, by-eval-type ---\n" + latex_simultaneous_ci_full_report(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_simultaneous_ci_coverage_width_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Coverage vs. width, one point per (scenario, CI method) per eval type
    (null condition) -- deliberately NOT one pooled dot per method, since a
    method's pooled-average width can look reasonable while individual
    scenario/n/k cells sit far from it (see max-T's random-denominator
    instability at small N + large k, evalstats.core.paired.
    _max_stat_simultaneous_cis's bootstrap_t branch): a single dot per
    method would average that away. Only plots Bonferroni and max-T
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none`, which sits so far below
    nominal coverage -- no simultaneous adjustment at all -- that it
    squashes the Bonferroni-vs-max-T comparison this plot exists to show;
    `none` is still in the printed/logged report tables and the CSV).
    Whichever cloud sits further left (narrower width) at matching
    coverage is the better default -- and stray points reveal exactly
    which scenarios don't follow that pattern."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(5.0 * nrows, 5.0), squeeze=False)

    plot_method_names = {m.name for m in SIMULTANEOUS_CI_PLOT_METHODS}
    null_rows_all = [
        r for r in results
        if r.condition == "null" and r.n_reps > 0 and r.ci_method in plot_method_names
    ]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "width": r.total_width / r.n_reps,
        }
        for r in null_rows_all
    ])
    scenario_level = (
        df.groupby(["eval_type", "label", "ci_method"], as_index=False).agg(
            coverage=("coverage", "mean"), width=("width", "mean"),
        )
        if not df.empty else df
    )

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
        et_df = scenario_level[scenario_level["eval_type"] == et] if not scenario_level.empty else scenario_level
        for m in SIMULTANEOUS_CI_PLOT_METHODS:
            m_df = et_df[et_df["ci_method"] == m.name] if not et_df.empty else et_df
            if m_df.empty:
                continue
            ax.scatter(
                m_df["width"], m_df["coverage"], color=m.color, s=34, label=m.name,
                edgecolors="white", linewidths=0.5, alpha=0.75,
            )
        ax.set_xlabel("Average per-comparison CI width (null)")
        ax.set_ylabel("Family-wise coverage (null)")
        ax.set_title(f"eval type: {et}")
        # Zoom to the actual coverage spread (plus the nominal line) rather
        # than a fixed [0, 1] -- with `none` dropped from this plot, every
        # remaining point usually clusters near nominal, and a full [0, 1]
        # axis squashes that spread into an unreadable sliver at the top.
        if not et_df.empty:
            cov_vals = et_df["coverage"].tolist() + [target]
            lo, hi = min(cov_vals), max(cov_vals)
            pad = max(0.01, (hi - lo) * 0.15)
            ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
        else:
            ax.set_ylim(0.0, 1.02)
    # One legend outside the rightmost facet (see save_multiarm_fwer_power_plot).
    _handles, _labels = axes[0][0].get_legend_handles_labels()
    if _handles:
        axes[0][-1].legend(_handles, _labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                           borderaxespad=0.0, fontsize=7)

    fig.suptitle(
        "Simultaneous Confidence Interval Calibration: Coverage vs. Width\n"
        f"One point per scenario, averaged across $n$ and $k$ (nominal coverage = {target:.0%})",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_coverage_width_vs_k_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Family-wise coverage and average width as a function of k (number of
    arms), one curve per CI method, collapsed across eval types and sample
    sizes -- mirrors save_multiarm_fwer_vs_k_plot. This is the direct
    picture of "pairwise comparisons grow as k(k-1)/2": Bonferroni's width
    should grow faster than max-T's, since Bonferroni's per-comparison
    budget (alpha/pairs) shrinks with the pair count while max-T's joint
    bootstrap doesn't pay that same tax. Only plots Bonferroni and max-T
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none`, whose coverage falling
    further below nominal as k grows is a different, already-obvious story
    that would squash this one on the same axes -- `none` is still in the
    printed/logged report tables and the CSV). Only produced when more
    than one k value was swept; returns out_path unchanged (without
    writing) if all results share the same k."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    ks_present = sorted({r.k for r in results})
    if len(ks_present) < 2:
        return out_path

    fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_cov.axhline(target, color="black", linewidth=1.0, linestyle="--", label=f"nominal={target:.0%}")

    all_cov_vals: list[float] = [target]
    for m in SIMULTANEOUS_CI_PLOT_METHODS:
        c_rows = [r for r in results if r.ci_method == m.name]
        if not c_rows:
            continue
        xs, ys_cov, ys_width = [], [], []
        scen_cov, scen_width = [], []
        for k in ks_present:
            k_rows = [r for r in c_rows if r.k == k]
            null_rows = [r for r in k_rows if r.condition == "null"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            # Normalize each row's width by its own eval type's scale span
            # before pooling -- see _width_scale. The squares divide by the
            # square of that span, so the band stays on the same axis.
            w_null = sum(r.total_width / _width_scale(r.eval_type) for r in null_rows)
            if t_null == 0:
                continue
            xs.append(k)
            ys_cov.append(c_null / t_null)
            ys_width.append(w_null / t_null)
            scen_cov.append(_scenario_values(null_rows, lambda r: r.all_covered))
            scen_width.append(_scenario_values(
                null_rows, lambda r: r.total_width / _width_scale(r.eval_type)))
        if xs:
            ax_cov.plot(xs, ys_cov, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_width.plot(xs, ys_width, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            # Band endpoints join the y-limit inputs so the zoom below fits them.
            all_cov_vals.extend(_scenario_bands(ax_cov, xs, ys_cov, scen_cov, color=m.color))
            _scenario_bands(ax_width, xs, ys_width, scen_width, color=m.color)
            all_cov_vals.extend(ys_cov)

    ax_cov.set_xlabel("k (number of arms)")
    ax_cov.set_ylabel("Family-wise coverage (null)")
    ax_cov.set_title("Coverage vs. number of arms")
    # Zoom to the actual coverage spread (plus the nominal line) rather than
    # a fixed [0, 1] -- with `none` dropped from this plot (SIMULTANEOUS_CI_
    # PLOT_METHODS), every remaining curve usually clusters near nominal, and
    # a full [0, 1] axis squashes that spread into an unreadable sliver at
    # the top (see save_simultaneous_ci_coverage_width_plot's identical fix).
    cov_lo, cov_hi = min(all_cov_vals), max(all_cov_vals)
    cov_pad = max(0.01, (cov_hi - cov_lo) * 0.15)
    ax_cov.set_ylim(max(0.0, cov_lo - cov_pad), min(1.02, cov_hi + cov_pad))
    ax_cov.set_xticks(ks_present)

    ax_width.set_xlabel("k (number of arms)")
    ax_width.set_ylabel("Avg per-comparison CI width (null),\nas a fraction of each eval type's scale")
    ax_width.set_title("Width vs. number of arms")
    ax_width.set_ylim(bottom=0.0)
    ax_width.set_xticks(ks_present)

    # One shared legend for both panels (coverage's nominal line plus every
    # method, which both panels plot identically) instead of a separate
    # legend per panel, placed outside the axes to the right.
    handles, labels = ax_cov.get_legend_handles_labels()
    ax_width.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    fig.suptitle(
        "Simultaneous Confidence Interval Calibration vs. Number of Systems Compared\n"
        f"Nominal coverage = {target:.0%}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_coverage_width_vs_n_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Family-wise coverage and average width as a function of n (sample
    size), one curve per CI method, collapsed across eval types and k --
    the sample-size analogue of save_simultaneous_ci_coverage_width_vs_k_plot
    (same two-panel line-plot style: exact x-ticks pinned to the sizes
    actually swept, coverage y-axis zoomed to the actual spread rather than
    a fixed [0, 1]). X-axis is log-scaled, unlike the vs-k plot's linear one:
    n sweeps span an order of magnitude or more (e.g. the official preset's
    15..500), so a linear axis crams the small-n tick labels into an
    unreadable overlapping cluster -- log-scale is the standard convention
    for coverage-vs-sample-size plots for exactly this reason, and still
    shows exact tick labels (via a ScalarFormatter override) rather than
    scientific notation. Complements save_simultaneous_ci_violin_vs_n_plot
    (full per-cell distribution, faceted by eval type) with the single
    pooled-mean curve per method that's easier to read at a glance across
    the whole n sweep. Only plots Bonferroni/max-T/sidak/boot
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none` -- see that plot's
    docstring for why; `none` is still in the printed/logged report tables
    and the CSV). Only produced when more than one n value was swept;
    returns out_path unchanged (without writing) if all results share the
    same n."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    target = 1.0 - alpha
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    if len(sizes_present) < 2:
        return out_path

    fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_cov.axhline(target, color="black", linewidth=1.0, linestyle="--", label=f"nominal={target:.0%}")

    all_cov_vals: list[float] = [target]
    for m in SIMULTANEOUS_CI_PLOT_METHODS:
        c_rows = [r for r in results if r.ci_method == m.name]
        if not c_rows:
            continue
        xs, ys_cov, ys_width = [], [], []
        scen_cov, scen_width = [], []
        for n in sizes_present:
            n_rows = [r for r in c_rows if r.n == n]
            null_rows = [r for r in n_rows if r.condition == "null"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            # Normalize each row's width by its own eval type's scale span
            # before pooling -- see _width_scale. The squares divide by the
            # square of that span, so the band stays on the same axis.
            w_null = sum(r.total_width / _width_scale(r.eval_type) for r in null_rows)
            if t_null == 0:
                continue
            xs.append(n)
            ys_cov.append(c_null / t_null)
            ys_width.append(w_null / t_null)
            scen_cov.append(_scenario_values(null_rows, lambda r: r.all_covered))
            scen_width.append(_scenario_values(
                null_rows, lambda r: r.total_width / _width_scale(r.eval_type)))
        if xs:
            ax_cov.plot(xs, ys_cov, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_width.plot(xs, ys_width, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            # Band endpoints join the y-limit inputs so the zoom below fits them.
            all_cov_vals.extend(_scenario_bands(ax_cov, xs, ys_cov, scen_cov, color=m.color))
            _scenario_bands(ax_width, xs, ys_width, scen_width, color=m.color)
            all_cov_vals.extend(ys_cov)

    ax_cov.set_xlabel("n (sample size)")
    ax_cov.set_ylabel("Family-wise coverage (null)")
    ax_cov.set_title("Coverage vs. sample size")
    # Zoom to the actual coverage spread (plus the nominal line) rather than
    # a fixed [0, 1] -- see save_simultaneous_ci_coverage_width_vs_k_plot's
    # identical fix.
    cov_lo, cov_hi = min(all_cov_vals), max(all_cov_vals)
    cov_pad = max(0.01, (cov_hi - cov_lo) * 0.15)
    ax_cov.set_ylim(max(0.0, cov_lo - cov_pad), min(1.02, cov_hi + cov_pad))

    ax_width.set_xlabel("n (sample size)")
    ax_width.set_ylabel("Avg per-comparison CI width (null),\nas a fraction of each eval type's scale")
    ax_width.set_title("Width vs. sample size")
    ax_width.set_ylim(bottom=0.0)

    # One shared legend for both panels, placed outside the axes to the
    # right -- see save_simultaneous_ci_coverage_width_vs_k_plot's identical
    # fix.
    handles, labels = ax_cov.get_legend_handles_labels()
    ax_width.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    # Log-scale x-axis (see docstring) with exact tick labels at the swept
    # sizes instead of matplotlib's default log-scale power-of-ten ticks.
    for ax in (ax_cov, ax_width):
        ax.set_xscale("log")
        ax.set_xticks(sizes_present)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle(
        "Simultaneous Confidence Interval Calibration vs. Sample Size\n"
        f"Nominal coverage = {target:.0%}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_null_vs_alt_coverage_plot(
    *, results: list[SimultaneousCIResult], alpha: float, out_path: str,
    omit: dict[str, list[str]] | None = None,
    omit_note: dict[str, str] | None = None,
) -> str:
    """Family-wise coverage vs. n under the null (top row) and under the
    alternative (bottom row), faceted by eval type, sharing a y-axis within
    each column so the two conditions are directly comparable for that type.

    Exists because the headline calibration figure
    (save_simultaneous_ci_coverage_width_vs_n_plot) plots null coverage
    only, and the overall table reports Cov(alt) collapsed across n -- so a
    method whose alternative-condition coverage falls apart looks merely
    slightly conservative in both.

    Faceting by eval type is load-bearing, not cosmetic: the effect this
    plot exists to show is binary-specific. max_t builds a symmetric
    studentized bootstrap interval (theta_hat +- c*SE), while sidak/boot
    widen the canonical per-type CI -- Tango, a score interval, for binary.
    A difference of proportions with a real effect at small n is skewed and
    boundary-constrained, exactly where symmetric Wald-type intervals lose
    to score intervals. On continuous and likert, where no boundary problem
    arises, max_t is fine. Pooling eval types averages the two and reports
    neither.

    ``omit`` maps an eval-type group to methods dropped from BOTH of that
    group's panels (default: max_t on binary). Dropping it from the alt
    panel alone does not work: the y-axis is shared down each column so the
    two conditions stay comparable, and max_t's null-panel band on binary
    reaches 0.72, which drags the alt panel's scale with it. Either way its
    collapse leaves the remaining methods -- the ones a reader is choosing
    between -- indistinguishable, which defeats the purpose of the panel.
    The omission is annotated in-panel rather than silent, with ``omit_note``
    supplying the text, so the number stays visible and the reader is
    pointed at the table that carries it in full.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _ticker

    if omit is None:
        omit = {"bin": ["max_t"]}
    if omit_note is None:
        omit_note = {
            "bin": ("max$\\_$t omitted: not built for binary data and severely\n"
                    "undercovers here (Cov(alt) = 0.86 overall, and still below\n"
                    "nominal at the largest $n$) -- see the accompanying table."),
        }

    target = 1.0 - alpha
    sizes_present = sorted({r.n for r in results})
    groups = sort_groups({report_eval_type_group(r.eval_type) for r in results})
    n_cols = max(len(groups), 1)
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(6.0 * n_cols, 8.4),
                             squeeze=False, sharey="col")

    for col, g in enumerate(groups):
        g_rows = [r for r in results if report_eval_type_group(r.eval_type) == g]
        for row, condition in enumerate(("null", "alt")):
            ax = axes[row][col]
            dropped = set(omit.get(g, []))
            ax.axhline(target, color="black", linewidth=1.0, linestyle="--")
            for m in SIMULTANEOUS_CI_PLOT_METHODS:
                if m.name in dropped:
                    continue
                rows_m = [r for r in g_rows if r.ci_method == m.name and r.condition == condition]
                if not rows_m:
                    continue
                xs, ys, scen = [], [], []
                for n in sizes_present:
                    n_rows = [r for r in rows_m if r.n == n]
                    t_n = sum(r.n_reps for r in n_rows)
                    if t_n == 0:
                        continue
                    xs.append(n)
                    ys.append(sum(r.all_covered for r in n_rows) / t_n)
                    scen.append(_scenario_values(n_rows, lambda r: r.all_covered))
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", color=m.color, markersize=5, linewidth=1.4,
                        alpha=0.85)
                _scenario_bands(ax, xs, ys, scen, color=m.color)
            if dropped and omit_note.get(g) and condition == "alt":
                ax.text(0.02, 0.03, omit_note[g], transform=ax.transAxes, fontsize=7.5,
                        va="bottom", ha="left", color="#444444", style="italic",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                  edgecolor="#BBBBBB", linewidth=0.6, alpha=0.9))
            ax.set_title(f"{g.upper()} -- {'null' if condition == 'null' else 'alternative'}",
                         fontsize=10.5)
            ax.set_xscale("log")
            ax.set_xticks(sizes_present)
            ax.get_xaxis().set_major_formatter(_ticker.FuncFormatter(lambda x, _: str(int(x))))
            ax.get_xaxis().set_minor_locator(_ticker.NullLocator())
            if row == 1:
                ax.set_xlabel("n (sample size)")
            if col == 0:
                ax.set_ylabel("Family-wise coverage")

    # Build the legend from every method drawn ANYWHERE in the figure, not
    # from one panel's handles: the top-left panel is a group that may omit a
    # method (see `omit`), which would silently drop it from the legend while
    # it is still plotted in the other facets -- an unlabelled line.
    from matplotlib.lines import Line2D
    present = [m for m in SIMULTANEOUS_CI_PLOT_METHODS
               if any(r.ci_method == m.name for r in results)]
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=1.0,
                      label=f"nominal={target:.0%}")]
    handles += [Line2D([], [], color=m.color, marker="o", markersize=5, linewidth=1.4,
                       alpha=0.85, label=m.name) for m in present]
    axes[0][-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                       borderaxespad=0.0, fontsize=8)
    # Describe whatever band was actually drawn -- hardcoding one description
    # mislabels the figure whenever BAND_STYLE is switched.
    band_desc = {
        "spread": "bands are the 10--90th percentile across scenarios",
        "ci": "bands are 95% CIs on the across-scenario mean",
        "both": "outer bands are the 10--90th percentile across scenarios, inner are 95% CIs on the mean",
    }.get(BAND_STYLE, "bands show across-scenario uncertainty")
    fig.suptitle(
        "Simultaneous CI Coverage: Null vs. Alternative, by Eval Type\n"
        f"Nominal = {target:.0%}; y-axis shared within each eval type; {band_desc}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_reliability_violin_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario family-wise
    coverage and average per-comparison interval score (null condition), one
    dot per (label, ci_method) -- the simultaneous-CI analogue of the
    pairwise/multi-arm reliability violins, and consistent with ci_single/
    ci_paired's reliability violin (coverage + interval score, not width).
    Exposes the spread the OVERALL SUMMARY table's pooled coverage hides: a
    method with nominal family-wise coverage on average can still miss badly
    on a specific scenario/k cell that pooling across labels masks. Only
    plots bonferroni/max_t/sidak/boot (`none` is dropped -- see
    SIMULTANEOUS_CI_PLOT_METHODS -- since it's so far below nominal
    coverage that it squashes the comparison this plot exists to show;
    it's still in the printed/logged report tables and the CSV)."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ci_methods = [m.name for m in SIMULTANEOUS_CI_PLOT_METHODS if m.name in {r.ci_method for r in results}]
    palette = {m.name: m.color for m in SIMULTANEOUS_CI_PLOT_METHODS}

    null_rows = [r for r in results if r.condition == "null" and r.n_reps > 0]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in null_rows
    ])
    scenario_level = (
        df.groupby(["eval_type", "label", "ci_method"], as_index=False).agg(
            coverage=("coverage", "mean"), score=("score", "mean"),
        )
        if not df.empty else df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        et_df = scenario_level[scenario_level["eval_type"] == et] if not scenario_level.empty else scenario_level
        for row_idx, (metric, ylabel, ref_line) in enumerate([
            ("coverage", "Family-wise coverage per scenario", target),
            ("score", "Interval score per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_methods = [name for name in ci_methods if name in et_df["ci_method"].values]
            sns.violinplot(
                data=et_df, x="ci_method", y=metric, order=et_methods, hue="ci_method",
                hue_order=et_methods, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="ci_method", y=metric, order=et_methods, hue="ci_method",
                hue_order=et_methods, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    # x-tick labels already name each method, but a color-key legend (see
    # save_multiarm_reliability_violin_plot's identical fix) makes it easy
    # to cross-reference colors against the other simultaneous_ci plots.
    legend_handles = [mpatches.Patch(facecolor=palette[m], alpha=0.5, label=m) for m in ci_methods]
    axes[0][-1].legend(
        handles=legend_handles, title="Simult. CI method", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        "Simultaneous Confidence Interval Reliability Across Evaluation Scenarios\n"
        f"One point per scenario, nominal coverage = {target:.0%}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_violin_vs_n_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Grouped violin plots of family-wise coverage and interval score vs.
    sample size n (null condition), one violin per CI method at each n
    (dodged side by side), faceted by eval type -- the Bonferroni/max-T
    analogue of ci_paired.py's --violin-plot (mj_floor vs. tango_scc vs.
    bayes_paired_comp vs. N).

    Each violin pools every (scenario, k) cell at that n rather than
    averaging k away: the small-N/large-k interaction is exactly what
    widens these violins and drags their tails at small n (max-T's
    random-denominator instability in the studentized-bootstrap-t branch of
    evalstats.core.paired._max_stat_simultaneous_cis -- resampling just n
    points to re-estimate a per-replicate SE gets noisy at small n, and
    taking a max over k(k-1)/2 simultaneous pairs multiplies the chances of
    hitting a near-zero denominator on any given replicate), so collapsing
    across k here would hide the very thing this plot exists to show.

    Only plots bonferroni/max_t/sidak/boot (`none` is dropped -- see
    SIMULTANEOUS_CI_PLOT_METHODS -- since it's so far below nominal
    coverage that it squashes the comparison this plot exists to show;
    it's still in the printed/logged report tables and the CSV).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ci_methods = [m.name for m in SIMULTANEOUS_CI_PLOT_METHODS if m.name in {r.ci_method for r in results}]
    palette = {m.name: m.color for m in SIMULTANEOUS_CI_PLOT_METHODS}

    null_rows = [r for r in results if r.condition == "null" and r.n_reps > 0]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "k": r.k, "n": r.n, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in null_rows
    ])

    n_cols = max(len(eval_types_present), 1)
    if df.empty:
        fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
        for ax_row in axes:
            for ax in ax_row:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    ns_present = sorted(df["n"].unique())
    n_order = [str(n) for n in ns_present]
    df["n_label"] = df["n"].astype(str)

    col_width = 1.3 * len(ns_present) + 2.5
    fig, axes = plt.subplots(2, n_cols, figsize=(col_width * n_cols, 9.0), squeeze=False)
    legend_handles = [mpatches.Patch(facecolor=palette[m], alpha=0.5, label=m) for m in ci_methods]

    for col_idx, et in enumerate(eval_types_present):
        et_df = df[df["eval_type"] == et]
        et_methods = [name for name in ci_methods if name in et_df["ci_method"].values]
        for row_idx, (metric, ylabel, ref_line) in enumerate([
            ("coverage", "Family-wise coverage", target),
            ("score", "Interval score", None),
        ]):
            ax = axes[row_idx][col_idx]
            if et_df.empty or not et_methods:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            sns.violinplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="ci_method", hue_order=et_methods,
                palette=palette, cut=0, inner="quartile", linewidth=0.7, dodge=True, alpha=0.35,
                legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="ci_method", hue_order=et_methods,
                palette=palette, size=3, alpha=0.5, jitter=0.2, dodge=True, linewidth=0.3,
                edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("n" if row_idx == 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")

    axes[0][-1].legend(
        handles=legend_handles, title="Simult. CI method", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        "Simultaneous Confidence Interval Coverage and Interval Score vs. Sample Size\n"
        f"Nominal coverage = {target:.0%}; each violin pools all $k$ and scenarios at that $n$",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode: Type-I error calibration for evalstats.tests' PPI-corrected
# wrappers under judge bias/miscalibration, ported from
# sim_type_i_calibration.py's _run_one. Calls evalstats.tests' internal PPI
# functions directly (the same functions back the public es.tests.* API) to
# skip judge_alignment overhead, exactly as the legacy script does.
# ---------------------------------------------------------------------------


@dataclass
class PPIResult:
    """One (source, test) cell's calibration outcome from the PPI-corrected
    sweep -- Type-I/coverage/power against the judge-bias source's true
    effect."""

    name: str
    tag: str
    test: str
    n_reps: int
    corrected_rejects: int
    uncorrected_rejects: int
    n_failed: int = 0
    n: int = 0
    """Group/condition-A sample size for this scenario (JudgeBiasSource.n).
    Only the 'sample_size' tag actually sweeps this (n=60/100/200/400);
    every other scenario uses the fixed baseline -- see
    latex_ppi_overall_summary's per-n columns, which are sourced from that
    one tag rather than every scenario (most of which share n=100)."""


@dataclass
class PPIEffectResult:
    """Bias and CI-coverage summary for one (scenario, test) cell's
    PPI-corrected point estimate -- complements PPIResult's Type-I check
    (does the p-value stay calibrated) with: is the estimate itself centered
    at the truth, and does its CI cover that truth at the nominal rate?
    Ported from sim_type_i_calibration.py's effect_results/_gold_null_values
    check; see run_ppi_effect_check."""
    name: str
    tag: str
    test: str
    n: int
    n_samples: int
    """Number of successful (non-failed) bootstrap draws this cell's stats are based on."""
    null_value: float
    """Monte Carlo gold-reference null value this estimate is compared against
    (not always 0 -- see estimate_judge_bias_gold_null_values)."""
    mean_bias: float
    """mean(estimate - null_value) across draws."""
    bias_z: float
    """mean_bias / SE(mean_bias) -- a |z| > 3 flags a real (not just noisy) bias."""
    coverage: float
    """Fraction of draws whose CI contains null_value."""
    mean_ci_width: float
    uncorrected_bias_z: float
    """Same z-score, but for the RAW (pre-PPI) LLM-only estimate -- contrast
    for how much PPI correction actually reduced bias."""


def _uncorrected_anova_independent_p_value(groups: list[np.ndarray]) -> float:
    return float(scipy_stats.f_oneway(*groups).pvalue)


def _uncorrected_anova_repeated_p_value(groups: list[np.ndarray]) -> float:
    from statsmodels.stats.anova import AnovaRM

    k = len(groups)
    n_subjects = len(groups[0])
    stacked = np.column_stack(groups)
    df_long = pd.DataFrame({
        "subject": np.repeat(np.arange(n_subjects), k),
        "condition": np.tile(np.arange(k), n_subjects),
        "score": stacked.reshape(-1),
    })
    rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
    return float(rm.anova_table.iloc[0]["Pr > F"])


def _uncorrected_friedman_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.friedmanchisquare(*groups).pvalue)


def _uncorrected_kruskal_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.kruskal(*groups).pvalue)


def _uncorrected_bayes_bootstrap_paired_p_value(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """LLM-only (uncorrected) Bayesian-bootstrap two-sided p-value for
    H0: mean(diffs) = 0 -- the same Dirichlet-weighted resampling
    evalstats.core.paired's 'bayes_bootstrap' method uses, applied directly
    (no PPI correction) as the baseline _ppi_paired_bayes_bootstrap's
    corrected version is compared against."""
    boots = bayes_bootstrap_means_1d(diffs, n_boot, rng, statistic="mean")
    p = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
    return min(max(p, 0.0), 1.0)


def _uncorrected_bootstrap_t_paired_p_value(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """LLM-only (uncorrected) studentized-bootstrap two-sided p-value for
    H0: mean(diffs) = 0 -- same pivot construction as
    evalstats.core.resampling.bootstrap_t_ci_1d (SE = std/sqrt(n) per
    replicate), applied directly (no PPI correction) as the baseline
    _ppi_paired_bootstrap_t's corrected version is compared against."""
    n = len(diffs)
    theta_hat = float(np.mean(diffs))
    se_hat = float(np.std(diffs, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    if not np.isfinite(se_hat) or se_hat <= 0.0:
        return 1.0
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = diffs[idx]
    boot_theta = samples.mean(axis=1)
    boot_se = np.std(samples, ddof=1, axis=1) / np.sqrt(n)
    valid = np.isfinite(boot_se) & (boot_se > 0.0)
    if not np.any(valid):
        return 1.0
    t_stats = (boot_theta[valid] - theta_hat) / boot_se[valid]
    t_obs = theta_hat / se_hat
    p = float(2.0 * min(np.mean(t_stats <= t_obs), np.mean(t_stats >= t_obs)))
    return min(max(p, 0.0), 1.0)


def _uncorrected_mj_floor_paired_p_value(diffs: np.ndarray) -> float:
    """LLM-only (uncorrected) two-sided p-value for H0: mean(diffs) = 0,
    using the SAME per-item variance evalstats.core.resampling.
    mj_floor_paired_ci's score interval is built from (V_hat = Var(diffs,
    ddof=0) / n, i.e. (n10+n01)/n^2 - (n10-n01)^2/n^3 for binary diffs) --
    applied directly (no PPI correction) as the baseline
    _ppi_paired_mj_floor's corrected version is compared against. Closed-form,
    no bootstrap needed."""
    n = len(diffs)
    d_hat = float(np.mean(diffs))
    v_hat = float(np.mean((diffs - d_hat) ** 2)) / n if n > 0 else 0.0
    if v_hat <= 0.0 or not np.isfinite(v_hat):
        return 1.0
    z_obs = d_hat / np.sqrt(v_hat)
    p = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_obs))))
    return min(max(p, 0.0), 1.0)


def _uncorrected_bonett_price_paired_p_value(diffs: np.ndarray) -> float:
    """LLM-only (uncorrected) two-sided p-value for H0: mean(diffs) = 0 built
    from the SAME shrunk-centre/regularized-SE pivot
    evalstats.tests._ppi_paired_bonett_price inverts, so this is the
    like-for-like uncorrected baseline for it -- exactly as
    _uncorrected_mj_floor_paired_p_value is for _ppi_paired_mj_floor.

    Bonett-Price's Laplace adjustment is a transform of (theta, V, n):
    kappa = n/(n+2), centre = kappa*theta, and the variance picks up an added
    pseudo-item regularization term 2*(1 + kappa*theta^2)/(n+2)^2 on top of
    kappa^2*V. Closed-form, no bootstrap needed."""
    n = len(diffs)
    if n <= 0:
        return 1.0
    theta = float(np.mean(diffs))
    v_hat = float(np.mean((diffs - theta) ** 2)) / n
    n_aug = n + 2.0
    kappa = n / n_aug
    centre = kappa * theta
    se = float(np.sqrt(max(kappa * kappa * v_hat
                           + 2.0 * (1.0 + kappa * theta * theta) / (n_aug * n_aug), 0.0)))
    if se <= 0.0 or not np.isfinite(se):
        return 1.0
    p = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(centre) / se)))
    return min(max(p, 0.0), 1.0)


def _lmm_wald_f_pvalue_from_fit(sm_result, k: int) -> float:
    """Wald-to-F omnibus p-value for template fixed effects, given an
    already-fitted MixedLM result (see _fit_lmm_general).

    Factored out of _uncorrected_lmm_p_value so callers that separately need
    the SAME LLM-only fit for the PPI correction (_ppi_lmm_p_value, via
    precomputed_fit=) can reuse it instead of fitting the identical model
    twice -- MixedLM's iterative MLE fit is by far the dominant cost of the
    ppi mode's lmm/lmm_factorial/lmm_runs tests (profiled at ~70% of total
    runtime), so this halves their cost.
    """
    beta = sm_result.fe_params.to_numpy()
    cov = _get_fe_vcov_sm(sm_result)
    df1 = k - 1
    df2 = float(sm_result.df_resid)
    beta_t, cov_t = beta[1:], cov[1:, 1:]
    wald = float(beta_t @ np.linalg.solve(cov_t, beta_t))
    f_stat = wald / df1
    return float(scipy_stats.f.sf(f_stat, df1, df2)) if f_stat > 0 else 1.0


def _uncorrected_lmm_p_value(groups: list[np.ndarray], factors=None) -> float:
    """Uncorrected (LLM-only) Wald F-test for score ~ <fixed factors> + (1|input),
    fit via statsmodels MixedLM (REML); _fit_lmm_general handles single-factor,
    multi-factor, and nested-run groups alike."""
    k = len(groups)
    template_labels = [f"T{i}" for i in range(k)]
    sm_result, _df_full, _x_row, _r = _fit_lmm_general(groups, template_labels, factors)
    return _lmm_wald_f_pvalue_from_fit(sm_result, k)


_ALPHA = ALPHA_DEFAULT

# Binary judge-bias data only supports the mean-based tests (a proportion is
# just the mean of a 0/1 variable, so PPI's rectifier applies unchanged --
# see scenarios.synthetic's binary judge-bias comment). The rank-based
# family (mwu/wilcoxon/friedman/kruskal) and ANOVA/LMM assume a scale that
# doesn't hold up under binary's massive ties, and generate_judge_bias_cell
# doesn't extend its additive noise/bias/slope judge model to a 0/1
# judgment for those structures either. PPI_WILSON is the single-arm
# analogue of MJ_FLOOR here -- same binary-only Wilson-style effective-n trick,
# just for a one-sample (not paired) proportion.
_PPI_BINARY_COMPATIBLE_TESTS = {
    TTEST.name, TTEST_WELCH.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, MJ_FLOOR.name,
    MJ_FLOOR_FIXED_LAMBDA.name, PPI_BONETT_PRICE.name, PPI_WILSON.name,
}

# The mirror-image restriction: tests whose estimand/formula is specific to
# paired/single-arm BINARY data (Tango's discordant-pair-rate score interval
# and PPI_WILSON's Wilson score interval, both with a continuity/shrinkage
# correction that only makes sense for a discrete proportion) and so should
# be excluded everywhere else, the same way BOOTSTRAP_T/BOOTSTRAP_T_SINGLE
# (numeric-only, see their Method-registry comments) are excluded FROM binary
# by simply never being added to _PPI_BINARY_COMPATIBLE_TESTS above.
_PPI_BINARY_ONLY_TESTS = {MJ_FLOOR.name, MJ_FLOOR_FIXED_LAMBDA.name, PPI_BONETT_PRICE.name, PPI_WILSON.name}

# ppi_wilson/bootstrap_t_single/t_interval_single/logit_t_single are
# single-ARM estimation methods (one group's mean, via cell.llm_a2/lab_a2)
# with no two-group/paired rejection decision to compute a Type-I error
# on -- unlike MJ_FLOOR/BOOTSTRAP_T, which are also two-/paired-group
# PAIRWISE_METHODS entries with a real Type-I concept. Excluded from
# _run_ppi_cell's Type-I sweep (see its use below) so they don't produce a
# fake "0/0 rejections, perfectly calibrated" row -- they're swept only by
# run_ppi_effect_check's bias/coverage pass, which is what they're
# actually for (see _PPI_EFFECT_TESTS). PPI_T_INTERVAL_SINGLE/
# PPI_LOGIT_T_SINGLE were missing from this set entirely (added after
# PPI_WILSON/PPI_BOOTSTRAP_T_SINGLE, per their own docstrings' "split out
# for the same reason" note, but never added here) -- caught by their
# Type-I row showing a literal, unconditional 0/n_reps in every scenario
# (both corrected AND uncorrected), not a real "perfectly calibrated"
# result: these estimands need single-sample data, so running them on
# this check's two-group cells degenerates instead of erroring.
_PPI_SINGLE_ARM_TESTS = {
    PPI_WILSON.name, PPI_BOOTSTRAP_T_SINGLE.name, PPI_T_INTERVAL_SINGLE.name, PPI_LOGIT_T_SINGLE.name,
}


def _ppi_effective_tests(sc: JudgeBiasSource, active_tests: list[str]) -> list[str]:
    """Restrict active_tests to what this scenario's eval_type actually
    supports: binary scenarios only run _PPI_BINARY_COMPATIBLE_TESTS;
    non-binary scenarios run everything except _PPI_BINARY_ONLY_TESTS."""
    if sc.eval_type == "binary":
        return [t for t in active_tests if t in _PPI_BINARY_COMPATIBLE_TESTS]
    return [t for t in active_tests if t not in _PPI_BINARY_ONLY_TESTS]


def _run_ppi_cell(
    sc: JudgeBiasSource, active_tests: list[str], n_reps: int, n_boot: int, seed,
    progress_dict=None, progress_key: str | None = None,
) -> list[PPIResult]:
    """Run all n_reps reps for one JudgeBiasSource.

    progress_dict / progress_key : optional
        When given, ``progress_dict[progress_key]`` is updated to ``(rep,
        n_reps)`` periodically (rate-limited to ~2/sec) as this cell runs.
        Lets a caller peek at a long-running cell's rep-level progress
        instead of it looking stalled until the whole cell returns -- some
        scenarios (large sample size, or hard-to-converge LMM fits) can
        take minutes on their own; see run_ppi_simulation's in-flight
        reporter thread. ``progress_dict`` may be a plain dict (serial
        mode) or a multiprocessing.Manager().dict() proxy (parallel mode)
        -- both support the same __setitem__ interface, so this function
        doesn't need to know which.
    """
    active_tests = [t for t in _ppi_effective_tests(sc, active_tests) if t not in _PPI_SINGLE_ARM_TESTS]
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in active_tests}
    uncorrected: dict[str, int] = {t: 0 for t in active_tests}
    failed: dict[str, int] = {t: 0 for t in active_tests}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    _last_progress_t = 0.0
    for _rep_i in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    # Closed-form (no-bootstrap) construction, not the
                    # general correct()-bootstrap _ppi_two_sample path --
                    # see _ppi_two_sample_t_interval's docstring for why:
                    # covariate-based estimators can never reach an
                    # analytic backend through correct()'s own dispatch,
                    # so ttest was stuck on the percentile bootstrap,
                    # which undercovers on near-boundary discrete (binary)
                    # proportions -- see this file's ttest-binary addendum.
                    r = _ppi_two_sample_t_interval(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA)
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST.name] += 1

            if TTEST_WELCH.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    # Closed-form construction -- see the matching TTEST
                    # block above for why (identical PPI-corrected
                    # construction; only the uncorrected reference test
                    # differs, equal_var=True vs False).
                    r = _ppi_two_sample_t_interval(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA)
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST_WELCH.name] += 1

            if MWU.name in active_tests:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(cell.llm_a2, cell.llm_b2, alternative="two-sided").pvalue)
                    uncorrected[MWU.name] += int(p_u < _ALPHA)
                    _e, _ci, _p, _rec, _lam = _ppi_mannwhitney_corrected(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA, n_boot, _rng_seed())
                    r = _MWUResult(_p)
                    corrected[MWU.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MWU.name] += 1





            if WILCOXON.name in active_tests:
                try:
                    # Deliberately left at scipy's default method="auto" --
                    # see _safe_wilcoxon_p's docstring: it's slower for
                    # small tied/discrete samples but computes a genuinely
                    # different (rigorously tie-corrected exact), not just
                    # slower, p-value than forcing method="exact" would.
                    p_u = float(scipy_stats.wilcoxon(cell.llm_x, cell.llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    # paired_walsh_midrank_theta (evalstats.ppi -- a Hodges-Lehmann
                    # Walsh-average midrank-sign statistic, matched estimator/
                    # rectifier), NOT np.median -- under heavy ties (e.g. likert's
                    # integer-rounded truth), the population MEDIAN of a paired
                    # difference can stay locked at exactly 0 even under a large,
                    # real, classical-Wilcoxon-detectable shift, which no bootstrap
                    # jitter fixes (it's the wrong estimand, not a resampling-
                    # degeneracy problem). A simpler per-item sign proportion fixed
                    # that but was itself found to have severely inflated Type-I
                    # error at small n_lab against real, heavily-tied judge-pair
                    # data -- this Walsh-average construction does NOT fix that
                    # second issue either (confirmed statistically equivalent to
                    # the sign proportion on the specific extreme-tie data that
                    # showed it) -- see paired_walsh_midrank_theta's docstring in
                    # evalstats/ppi.py for the full root-cause writeup; that
                    # inflation remains an open, documented limitation.
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, paired_walsh_midrank_theta, _ALPHA, n_boot, _rng_seed(), rectifier_func=paired_walsh_midrank_theta)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[WILCOXON.name] += 1

            if PAIRED_T.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_rel(cell.llm_x, cell.llm_y).pvalue)
                    uncorrected[PAIRED_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[PAIRED_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[PAIRED_T.name] += 1

            if BAYES_BOOTSTRAP.name in active_tests:
                try:
                    p_u = _uncorrected_bayes_bootstrap_paired_p_value(cell.llm_x - cell.llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BAYES_BOOTSTRAP.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bayes_bootstrap(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BAYES_BOOTSTRAP.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[BAYES_BOOTSTRAP.name] += 1

            if BOOTSTRAP_T.name in active_tests:
                try:
                    p_u = _uncorrected_bootstrap_t_paired_p_value(cell.llm_x - cell.llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BOOTSTRAP_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bootstrap_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BOOTSTRAP_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[BOOTSTRAP_T.name] += 1

            if MJ_FLOOR.name in active_tests:
                try:
                    p_u = _uncorrected_mj_floor_paired_p_value(cell.llm_x - cell.llm_y)
                    uncorrected[MJ_FLOOR.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_mj_floor(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    corrected[MJ_FLOOR.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MJ_FLOOR.name] += 1

            if MJ_FLOOR_FIXED_LAMBDA.name in active_tests:
                try:
                    p_u = _uncorrected_mj_floor_paired_p_value(cell.llm_x - cell.llm_y)
                    uncorrected[MJ_FLOOR_FIXED_LAMBDA.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_mj_floor(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, power_tune=False)
                    corrected[MJ_FLOOR_FIXED_LAMBDA.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MJ_FLOOR_FIXED_LAMBDA.name] += 1

            if PPI_BONETT_PRICE.name in active_tests:
                try:
                    p_u = _uncorrected_bonett_price_paired_p_value(cell.llm_x - cell.llm_y)
                    uncorrected[PPI_BONETT_PRICE.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bonett_price(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    corrected[PPI_BONETT_PRICE.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[PPI_BONETT_PRICE.name] += 1

            if PPI_T_INTERVAL.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_rel(cell.llm_x, cell.llm_y).pvalue)
                    uncorrected[PPI_T_INTERVAL.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_t_interval(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    corrected[PPI_T_INTERVAL.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[PPI_T_INTERVAL.name] += 1

            if PPI_LOGIT_T.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_rel(cell.llm_x, cell.llm_y).pvalue)
                    uncorrected[PPI_LOGIT_T.name] += int(p_u < _ALPHA)
                    _lo, _hi = EVAL_TYPE_SCALE_BOUNDS[sc.eval_type]
                    r = _ppi_paired_logit_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, lo=_lo, hi=_hi)
                    corrected[PPI_LOGIT_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[PPI_LOGIT_T.name] += 1

            if ANOVA_IND.name in active_tests:
                try:
                    groups_ind = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_ind_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_anova_independent_p_value(groups_ind)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups_ind, groups_ind_lab, k=len(groups_ind))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_IND.name] += 1

            if ANOVA_REP.name in active_tests:
                try:
                    groups_rep = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_rep_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_anova_repeated_p_value(groups_rep)
                    uncorrected[ANOVA_REP.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_repeated_p_value(groups_rep, groups_rep_lab, k=len(groups_rep))
                    corrected[ANOVA_REP.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_REP.name] += 1

            if FRIEDMAN.name in active_tests:
                try:
                    groups_fr = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_fr_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_friedman_p_value(groups_fr)
                    uncorrected[FRIEDMAN.name] += int(p_u < _ALPHA)
                    p = _ppi_friedman_p_value(groups_fr, groups_fr_lab, k=len(groups_fr))
                    corrected[FRIEDMAN.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[FRIEDMAN.name] += 1

            if KRUSKAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_influence(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL.name] += 1

            if KRUSKAL_MNAR_EXPERIMENTAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL_MNAR_EXPERIMENTAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise_mnar_experimental(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL_MNAR_EXPERIMENTAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL_MNAR_EXPERIMENTAL.name] += 1

            # Row-sum ("real Kruskal-Wallis") projection of the SAME corrected
            # pairwise vector KRUSKAL tests -- see
            # evalstats.tests._ppi_kruskal_wallis_rowsum. Its own bootstrap
            # draw on purpose: reusing KRUSKAL's would shift that method's rng
            # stream and silently change every existing kruskal number here.
            if any(_m.name in active_tests for _m in (
                    KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                    KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE)):
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    pw_r = _ppi_kruskal_wallis_influence(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    for _m, _w in ((KRUSKAL_ROWSUM, "full"), (KRUSKAL_ROWSUM_LABELED, "labeled")):
                        if _m.name in active_tests:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_rowsum_from_pairwise(pw_r, weights=_w)["wald_p"] < _ALPHA)
                    for _m, _c in ((KRUSKAL_TWOPART, "twopart"),
                                   (KRUSKAL_EIGENGAP, "eigengap")):
                        if _m.name in active_tests:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_candidate_from_pairwise(pw_r, _c, _ALPHA)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE.name in active_tests:
                        # Its own draw: it needs the raw groups (the covariance
                        # is built from per-item influence values, not from the
                        # bootstrap replicates), so it cannot share pw_r.
                        uncorrected[KRUSKAL_INFLUENCE.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups_kw, groups_kw_lab, _ALPHA, n_boot, _rng_seed()
                            )["wald_p"] < _ALPHA)
                except Exception:
                    for _m in (KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                               KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE):
                        if _m.name in active_tests:
                            failed[_m.name] += 1

            if LMM.name in active_tests:
                try:
                    groups_lmm = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_lmm_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    k = len(groups_lmm)
                    # Fit once, reuse for both the uncorrected Wald F-test and
                    # the PPI correction (which needs the identical LLM-only
                    # fit as its nuisance-parameter/reference point) -- see
                    # _lmm_wald_f_pvalue_from_fit's docstring.
                    fit = _fit_lmm_general(groups_lmm, [f"T{i}" for i in range(k)])
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_lmm, groups_lmm_lab, k=k, precomputed_fit=fit)
                    corrected[LMM.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM.name] += 1

            if LMM_FACTORIAL.name in active_tests:
                try:
                    groups_lf = [cell.llm_W, cell.llm_X, cell.llm_Y, cell.llm_Z]
                    groups_lf_lab = [cell.lab_W, cell.lab_X, cell.lab_Y, cell.lab_Z]
                    k = len(groups_lf)
                    fit = _fit_lmm_general(groups_lf, [f"T{i}" for i in range(k)], JUDGE_BIAS_LMM_FACTORIAL_FACTORS)
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM_FACTORIAL.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(
                        groups_lf, groups_lf_lab, k=k, factors=JUDGE_BIAS_LMM_FACTORIAL_FACTORS, precomputed_fit=fit,
                    )
                    corrected[LMM_FACTORIAL.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_FACTORIAL.name] += 1

            if LMM_RUNS.name in active_tests:
                try:
                    groups_runs = [cell.llm_A_runs, cell.llm_B_runs, cell.llm_C_runs]
                    groups_runs_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    k = len(groups_runs)
                    fit = _fit_lmm_general(groups_runs, [f"T{i}" for i in range(k)])
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM_RUNS.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_runs, groups_runs_lab, k=k, precomputed_fit=fit)
                    corrected[LMM_RUNS.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_RUNS.name] += 1

        if progress_dict is not None:
            _now = time.time()
            if _now - _last_progress_t >= 0.5 or _rep_i + 1 == n_reps:
                progress_dict[progress_key] = (_rep_i + 1, n_reps)
                _last_progress_t = _now

    return [
        PPIResult(
            name=sc.name, tag=sc.tag, test=t, n_reps=n_reps,
            corrected_rejects=corrected[t], uncorrected_rejects=uncorrected[t], n_failed=failed[t],
            n=sc.n,
        )
        for t in active_tests
    ]


def _ppi_in_flight_line(progress_dict, done_keys: set) -> str | None:
    """Format a one-line snapshot of currently in-progress (i.e. reporting
    rep-level progress but not yet returned) ppi cells, or None if there's
    nothing worth showing. Factored out of _run_in_flight_reporter so it's
    independently testable."""
    snapshot = dict(progress_dict)
    active = {name: rep_total for name, rep_total in snapshot.items() if name not in done_keys}
    if not active:
        return None
    parts = [
        f"{name}: {rep}/{total} ({100.0 * rep / total:.0f}%)"
        for name, (rep, total) in sorted(active.items())
    ]
    return "  [in-flight] " + "  |  ".join(parts)


def _run_in_flight_reporter(progress_dict, done_keys: set, done_lock, stop_event, interval: float = 8.0) -> None:
    """Background-thread body for run_ppi_simulation's parallel path.

    Some ppi scenarios (large sample size, or hard-to-converge LMM fits --
    see cases/pvalues.py module docstring / harness README) can take
    several minutes on their own; with imap_unordered, the main progress
    bar only advances when a WHOLE cell returns, so a long cell looks
    identical to a hang from the outside. This periodically prints a
    snapshot of every currently in-flight cell's rep-level progress
    (populated by _run_ppi_cell's progress_dict writes) so it's visible
    that work is still happening, and roughly how far along it is.
    """
    while not stop_event.wait(interval):
        with done_lock:
            done_snapshot = set(done_keys)
        line = _ppi_in_flight_line(progress_dict, done_snapshot)
        if line:
            print(f"\n{line}", flush=True)


def run_ppi_simulation(
    sources: list[JudgeBiasSource], active_tests: list[str], n_reps: int, n_boot: int,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[PPIResult]:
    """Sweep every requested PPI test over every JudgeBiasSource cell,
    parallelized across n_workers, and flatten the per-cell PPIResult lists
    into one list."""
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(sources))]

    reporter = _ProgressReporter(len(sources), mode=progress_mode, label="pvalues-ppi")
    results: list[PPIResult] = []

    if n_workers <= 1:
        for i, (sc, child_seed) in enumerate(zip(sources, child_seeds)):
            results.extend(_run_ppi_cell(sc, active_tests, n_reps, n_boot, child_seed))
            reporter.update(i + 1, detail=f"{sources[i].name}")
        reporter.update(len(sources), detail="done")
        return results

    # Parallel path: a shared Manager dict lets each worker report rep-level
    # progress while it's mid-cell (see _run_ppi_cell's progress_dict), and
    # a background thread periodically prints an in-flight snapshot -- see
    # _run_in_flight_reporter's docstring for why this matters here
    # specifically (long individual cells + imap_unordered's coarse
    # per-cell-only progress signal).
    #
    # ctx.Manager() (not the bare _mp.Manager()) -- a bare Manager() spawns
    # its server process using the platform's default start method
    # regardless of what context the Pool below explicitly requests, which
    # is "spawn" on macOS. Running this function's caller as a plain
    # top-level script (no `if __name__ == "__main__":` guard) can crash
    # under the bare Manager() -- a FileNotFoundError when spawn's
    # bootstrap tries to re-import a script with no real file backing it
    # (piped via stdin), or, given a real file, "An attempt has been made
    # to start a new process before the current process has finished its
    # bootstrapping phase" (spawn's safety check against the un-guarded
    # script re-executing the Manager()/Pool()-creating code on every
    # re-import). Pinning the Manager to the same fork context the Pool
    # already uses avoids the re-import step entirely (fork duplicates the
    # current process directly), matching run_ppi_comparison_simulation and
    # every other pool in this file, none of which hit this because none of
    # them also create a Manager.
    ctx = _mp.get_context("fork")
    manager = ctx.Manager()
    progress_dict = manager.dict()
    args_list = [
        (sc, active_tests, n_reps, n_boot, child_seed, progress_dict)
        for sc, child_seed in zip(sources, child_seeds)
    ]

    done_keys: set = set()
    done_lock = threading.Lock()
    stop_event = threading.Event()
    reporter_thread = None
    if progress_mode != "off":
        reporter_thread = threading.Thread(
            target=_run_in_flight_reporter,
            args=(progress_dict, done_keys, done_lock, stop_event),
            daemon=True,
        )
        reporter_thread.start()

    try:
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_ppi_cell_worker, args_list)):
                if cell_results:
                    with done_lock:
                        done_keys.add(cell_results[0].name)
                results.extend(cell_results)
                reporter.update(i + 1)
    finally:
        stop_event.set()
        if reporter_thread is not None:
            reporter_thread.join(timeout=1.0)

    reporter.update(len(sources), detail="done")
    return results


# ---------------------------------------------------------------------------
# PPI mode, effect-size calibration: bias and CI coverage of the PPI-
# corrected point estimate itself, complementing run_ppi_simulation's Type-I
# check (does the p-value stay calibrated). Ported from
# sim_type_i_calibration.py's effect_results/_gold_null_values check. lmm/
# lmm_factorial/lmm_runs are intentionally excluded (same as the legacy
# script): their headline estimand is a quadratic form in the fixed effects
# with no valid CI by design -- see es.tests.lmm()'s docstring.
# ---------------------------------------------------------------------------

_PPI_EFFECT_TESTS = (
    TTEST.name, TTEST_WELCH.name, MWU.name, WILCOXON.name, PAIRED_T.name, BAYES_BOOTSTRAP.name,
    BOOTSTRAP_T.name, MJ_FLOOR.name, MJ_FLOOR_FIXED_LAMBDA.name, PPI_BONETT_PRICE.name, ANOVA_IND.name, ANOVA_REP.name, FRIEDMAN.name, KRUSKAL.name, KRUSKAL_MNAR_EXPERIMENTAL.name,
    PPI_WILSON.name, PPI_BOOTSTRAP_T_SINGLE.name, PPI_T_INTERVAL.name, PPI_LOGIT_T.name, PPI_T_INTERVAL_SINGLE.name, PPI_LOGIT_T_SINGLE.name,
)

# bayes_bootstrap/bootstrap_t/mj_floor/ppi_wilson/bootstrap_t_single/
# ppi_t_interval/ppi_logit_t are excluded from the main ppi Type-I/effect
# plots and reported in a separate plot instead: they read differently to
# reviewers than the rest of PPI_TEST_METHODS (which are all textbook tests
# -- t-test, Wilcoxon, ANOVA, Friedman, Kruskal, LMM). These are bootstrap/
# CI-based constructions (Bayesian bootstrap, studentized bootstrap, Tango's/
# Wilson's score intervals, and now the closed-form logit-t/t-interval CIs)
# that would read as unfamiliar or confusing mixed in with the standard-
# methods plot -- mj_floor/ppi_wilson specifically are fundamentally CI
# constructions for binary paired/single-arm proportions (see
# evalstats.tests._ppi_paired_mj_floor/_ppi_single_wilson), not p-value tests
# in their own right, and (along with bootstrap_t_single) are restricted to
# a single binary scenario (_PPI_BINARY_ONLY_TESTS) rather than swept across
# the full catalog, so they'd look sparse/broken next to tests with ~44x
# more scenarios' worth of points. ppi_wilson/bootstrap_t_single also have
# NO CI-coverage analogue in the Type-I rejection sweep (unlike tango/
# bootstrap_t, which are also PAIRWISE_METHODS entries there) -- they only
# ever appear via this effect-check pass, since there's no single-arm
# rejection decision to test Type-I error on. ppi_t_interval/ppi_logit_t do
# have a Type-I analogue (see _run_ppi_cell) and run across the full
# non-binary catalog like BOOTSTRAP_T/PAIRED_T -- they're grouped here
# purely on "reads like a CI construction, not a textbook test" grounds,
# the same criterion already applied to tango/bootstrap_t.
_PPI_NONSTANDARD_TESTS = {
    BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name, MJ_FLOOR.name, MJ_FLOOR_FIXED_LAMBDA.name,
    PPI_BONETT_PRICE.name, PPI_WILSON.name,
    PPI_BOOTSTRAP_T_SINGLE.name, PPI_T_INTERVAL.name, PPI_LOGIT_T.name,
    PPI_T_INTERVAL_SINGLE.name, PPI_LOGIT_T_SINGLE.name,
}

_PPI_CI_COMPARISON_TESTS = {PPI_BONETT_PRICE.name, PPI_WILSON.name, PPI_LOGIT_T.name, PPI_T_INTERVAL.name}
"""The curated CI-coverage/width comparison methods for save_ppi_effect_plot's
ci_comparison=True figure -- replaces an older 5-method "nonstandard"
comparison ({bayes_bootstrap, bootstrap_t, tango, ppi_wilson,
bootstrap_t_single}, still available via _ppi_tests_present(nonstandard=True))
with exactly these four closed-form PPI-corrected CI methods: Bonett-Price (binary
paired), PPI Wilson (binary single), PPI logit-t and PPI t-interval (numeric
paired, the closed-form bounded/unbounded replacements for bootstrap_t's role
in this comparison -- see PPI_AUTO_METHOD_TABLE in evalstats/config.py).
bayes_bootstrap/bootstrap_t/bootstrap_t_single are NOT dropped from
validation -- they keep running (and being flagged for miscalibration) in
the main Type-I/power sweep and its own save_ppi_typeI_plot(nonstandard=True)
plot (_PPI_NONSTANDARD_TESTS, unchanged in role); they just no longer get
their own dedicated CI-coverage/width comparison figure here. Deliberately a
SEPARATE constant from _PPI_NONSTANDARD_TESTS, not a rename/reuse of it --
_PPI_NONSTANDARD_TESTS is shared by BOTH the Type-I plot and this effect/
coverage plot, and only the latter is meant to narrow to these four."""


def _ppi_tests_present(results, *, nonstandard: bool) -> list[str]:
    """Test names present in results, in PPI_TEST_METHODS' canonical order,
    filtered to the standard (textbook) subset or the nonstandard
    (bootstrap/CI-based) subset -- see _PPI_NONSTANDARD_TESTS."""
    present = {r.test for r in results}
    if nonstandard:
        return [m.name for m in PPI_TEST_METHODS if m.name in present and m.name in _PPI_NONSTANDARD_TESTS]
    return [m.name for m in PPI_TEST_METHODS if m.name in present and m.name not in _PPI_NONSTANDARD_TESTS]


def _ppi_ci_comparison_tests_present(results) -> list[str]:
    """Test names in _PPI_CI_COMPARISON_TESTS present in results, in
    PPI_TEST_METHODS' canonical order -- the curated-plot analogue of
    _ppi_tests_present(nonstandard=True), backed by _PPI_CI_COMPARISON_TESTS
    instead."""
    present = {r.test for r in results}
    return [m.name for m in PPI_TEST_METHODS if m.name in present and m.name in _PPI_CI_COMPARISON_TESTS]


def _run_ppi_effect_cell(
    sc: JudgeBiasSource, active_tests: list[str], n_reps: int, n_boot: int, seed,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Draw n_reps fresh replicates and capture each active effect-check
    test's PPI-corrected (estimate, ci_low, ci_high, llm_estimate) per rep.

    Runs as its OWN dedicated pass (with its own, typically much smaller,
    --effect-reps count) rather than piggybacking on run_ppi_simulation's
    Type-I sweep the way sim_type_i_calibration.py's _run_one does for its
    "free" tests -- this keeps _run_ppi_cell's Type-I return type/call site
    completely unchanged, at the cost of redrawing ttest/ttest_welch/mwu/
    wilcoxon/kruskal's bootstrap a second time (cheap at the smaller
    effect-reps count this is meant to run at). anova_ind/anova_rep/friedman
    use the same closed-form noncentral-F test-inversion CI functions
    (_ppi_anova_independent_ci/_ppi_anova_repeated_ci/_ppi_friedman_ci) that
    evalstats.tests.anova_oneway/friedman use for their corrected_estimate/
    corrected_ci, not a separate bootstrap-based scalar estimator, so
    anova_ind's bias-z/coverage checks are computed the same way the public
    API reports them. llm_estimate is recomputed here on the
    disjoint unlabeled-only complement, matching the convention documented
    in anova_oneway/friedman's rectifier comments.

    ppi_wilson/bootstrap_t_single/t_interval_single/logit_t_single are the
    single-arm robustness-CI methods PPI_AUTO_METHOD_TABLE routes to for
    marginal (not pairwise) alignment corrections -- unlike every other
    test here, they use only cell.llm_a2/lab_a2 (one group), not a
    two-group contrast; added so these methods get a genuine synthetic
    ground-truth coverage check instead of relying solely on
    cases/ppi_real.py's real-data check (see PPI_WILSON's Method-registry
    comment).
    """
    active_tests = _ppi_effective_tests(sc, active_tests)
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in active_tests:
                try:
                    # Closed-form construction -- see the Type-I sweep's
                    # matching TTEST block above for why.
                    r = _ppi_two_sample_t_interval(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA)
                    out[TTEST.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if TTEST_WELCH.name in active_tests:
                try:
                    # Closed-form construction -- see the Type-I sweep's
                    # matching TTEST_WELCH block for why.
                    r = _ppi_two_sample_t_interval(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA)
                    out[TTEST_WELCH.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if MWU.name in active_tests:
                try:
                    _e, _ci, _p, _rec, _lam = _ppi_mannwhitney_corrected(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA, n_boot, _rng_seed())
                    r = _MWUResult(_p)
                    out[MWU.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass





            if WILCOXON.name in active_tests:
                try:
                    # paired_walsh_midrank_theta (evalstats.ppi), matching
                    # _run_ppi_cell's and _ppi_comparison_pvalue's WILCOXON blocks
                    # (all switched from np.median, then from an intermediate
                    # per-item sign proportion -- see that function's docstring
                    # for the full history of why each was wrong/insufficient).
                    # estimate_judge_bias_gold_null_values' "wilcoxon" gold value
                    # was updated in lockstep to the matching true Walsh-average
                    # midrank-sign population quantity, so this stays an
                    # apples-to-apples bias/coverage comparison.
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, paired_walsh_midrank_theta, _ALPHA, n_boot, _rng_seed(), rectifier_func=paired_walsh_midrank_theta)
                    out[WILCOXON.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PAIRED_T.name in active_tests:
                try:
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    out[PAIRED_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in active_tests:
                try:
                    r = _ppi_paired_bayes_bootstrap(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    out[BAYES_BOOTSTRAP.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BOOTSTRAP_T.name in active_tests:
                try:
                    r = _ppi_paired_bootstrap_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    out[BOOTSTRAP_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if MJ_FLOOR.name in active_tests:
                try:
                    r = _ppi_paired_mj_floor(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    out[MJ_FLOOR.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if MJ_FLOOR_FIXED_LAMBDA.name in active_tests:
                try:
                    r = _ppi_paired_mj_floor(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, power_tune=False)
                    out[MJ_FLOOR_FIXED_LAMBDA.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_BONETT_PRICE.name in active_tests:
                try:
                    r = _ppi_paired_bonett_price(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    out[PPI_BONETT_PRICE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_T_INTERVAL.name in active_tests:
                try:
                    r = _ppi_paired_t_interval(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    out[PPI_T_INTERVAL.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_LOGIT_T.name in active_tests:
                try:
                    _lo, _hi = EVAL_TYPE_SCALE_BOUNDS[sc.eval_type]
                    r = _ppi_paired_logit_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, lo=_lo, hi=_hi)
                    out[PPI_LOGIT_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_WILSON.name in active_tests:
                try:
                    # Single-arm robustness CI (PPI_AUTO_METHOD_TABLE's binary
                    # marginal method) -- targets the same "a2" single-group
                    # mean estimand as TTEST/MWU's llm_a2/lab_a2 above, not a
                    # two-group contrast. Gold null: estimate_judge_bias_
                    # gold_null_values' "ppi_wilson" key (population mean of
                    # the a2 marginal).
                    r = _ppi_single_wilson(cell.llm_a2, cell.lab_a2, _ALPHA)
                    out[PPI_WILSON.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_BOOTSTRAP_T_SINGLE.name in active_tests:
                try:
                    # Single-arm robustness CI (PPI_AUTO_METHOD_TABLE's
                    # bounded_01/continuous marginal method) -- non-binary
                    # analogue of the PPI_WILSON block above, same a2 estimand.
                    r = _ppi_single_bootstrap_t(cell.llm_a2, cell.lab_a2, _ALPHA, n_boot, _rng_seed())
                    out[PPI_BOOTSTRAP_T_SINGLE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_T_INTERVAL_SINGLE.name in active_tests:
                try:
                    # Single-sample sibling of PPI_T_INTERVAL, closed-form
                    # analogue of PPI_BOOTSTRAP_T_SINGLE (identical a2
                    # estimand, no bootstrap resampling) -- see
                    # PPI_T_INTERVAL_SINGLE's Method-registry comment.
                    r = _ppi_single_t_interval(cell.llm_a2, cell.lab_a2, _ALPHA)
                    out[PPI_T_INTERVAL_SINGLE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_LOGIT_T_SINGLE.name in active_tests:
                try:
                    # [lo,hi]-bounded analogue of PPI_T_INTERVAL_SINGLE, same
                    # a2 estimand -- see PPI_LOGIT_T_SINGLE's Method-registry
                    # comment.
                    _lo, _hi = EVAL_TYPE_SCALE_BOUNDS[sc.eval_type]
                    r = _ppi_single_logit_t(cell.llm_a2, cell.lab_a2, _ALPHA, lo=_lo, hi=_hi)
                    out[PPI_LOGIT_T_SINGLE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if ANOVA_IND.name in active_tests:
                try:
                    groups_ai = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_ai_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    ci_result = _ppi_anova_independent_ci(groups_ai, groups_ai_lab, k=3, alpha=_ALPHA)
                    if ci_result is not None:
                        est, lo, hi = ci_result
                        masks = [~np.isnan(g_lab) for g_lab in groups_ai_lab]
                        groups_unlab = [g[~m] for g, m in zip(groups_ai, masks)]
                        llm_est = _anova_between_variance_from_groups(groups_unlab)
                        out[ANOVA_IND.name].append((est, lo, hi, llm_est))
                except Exception:
                    pass

            if ANOVA_REP.name in active_tests:
                try:
                    groups_ar = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_ar_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    ci_result = _ppi_anova_repeated_ci(groups_ar, groups_ar_lab, k=3, alpha=_ALPHA)
                    if ci_result is not None:
                        est, lo, hi = ci_result
                        labels_mat = np.column_stack(groups_ar_lab)
                        overlap = np.all(~np.isnan(labels_mat), axis=1)
                        llm_unlab_matrix = np.column_stack(groups_ar)[~overlap]
                        llm_est = _repeated_condition_variance(llm_unlab_matrix)
                        out[ANOVA_REP.name].append((est, lo, hi, llm_est))
                except Exception:
                    pass

            if FRIEDMAN.name in active_tests:
                try:
                    groups_fr = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_fr_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    ci_result = _ppi_friedman_ci(groups_fr, groups_fr_lab, k=3, alpha=_ALPHA)
                    if ci_result is not None:
                        est, lo, hi = ci_result
                        labels_mat = np.column_stack(groups_fr_lab)
                        overlap = np.all(~np.isnan(labels_mat), axis=1)
                        llm_unlab_matrix = np.column_stack(groups_fr)[~overlap]
                        llm_est = _friedman_rank_variance(llm_unlab_matrix)
                        out[FRIEDMAN.name].append((est, lo, hi, llm_est))
                except Exception:
                    pass

            if KRUSKAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw = _ppi_kruskal_wallis_influence(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                    out[KRUSKAL.name].append((
                        float(np.mean(pw["theta_hat"])), float(np.mean(pw["ci_lo"])),
                        float(np.mean(pw["ci_hi"])), float(np.mean(llm_theta)),
                    ))
                except Exception:
                    pass

            if KRUSKAL_MNAR_EXPERIMENTAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw = _ppi_kruskal_wallis_pairwise_mnar_experimental(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                    out[KRUSKAL_MNAR_EXPERIMENTAL.name].append((
                        float(np.mean(pw["theta_hat"])), float(np.mean(pw["ci_lo"])),
                        float(np.mean(pw["ci_hi"])), float(np.mean(llm_theta)),
                    ))
                except Exception:
                    pass

            # The row-sum projection changes only the omnibus WALD TEST, never
            # the corrected theta vector or its per-pair CIs -- so its
            # effect/CI row is the same estimator as KRUSKAL's (recomputed on
            # its own draw, hence Monte-Carlo-different but not
            # method-different). Emitted so --mode ppi's effect check has a
            # row for every active test rather than a hole.
            if any(_m.name in active_tests for _m in (
                    KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                    KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE)):
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw_r = _ppi_kruskal_wallis_influence(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw_r["pairs"])
                    row = (float(np.mean(pw_r["theta_hat"])), float(np.mean(pw_r["ci_lo"])),
                           float(np.mean(pw_r["ci_hi"])), float(np.mean(llm_theta)))
                    for _m in (KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                               KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE):
                        if _m.name in active_tests:
                            out[_m.name].append(row)
                except Exception:
                    pass

    return dict(out)


def _effect_cell_stats(
    samples: list[tuple[float, float, float, float]], null_val: float,
) -> tuple[float, float, float, float, int]:
    """(mean_bias, z, coverage_rate, mean_ci_width, n) for one (scenario,
    test) cell -- ported from sim_type_i_calibration.py's helper of the same
    name."""
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    estimates = np.array([s[0] for s in samples]) - null_val
    contains = np.array([(s[1] <= null_val <= s[2]) for s in samples])
    ci_widths = np.array([s[2] - s[1] for s in samples])
    bias_mean = float(estimates.mean())
    bias_se = float(estimates.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = bias_mean / bias_se if bias_se and bias_se > 0 else float("nan")
    coverage = float(contains.mean())
    mean_ci_width = float(ci_widths.mean())
    return bias_mean, z, coverage, mean_ci_width, n


def _uncorrected_bias_z(samples: list[tuple[float, float, float, float]], null_val: float) -> float:
    """z-score of the RAW (pre-PPI) LLM-only estimate's bias -- a contrast for
    how much PPI correction reduced bias. No CI exists for the raw estimate,
    so this is bias only, not coverage."""
    n = len(samples)
    if n < 2:
        return float("nan")
    raw = np.array([s[3] for s in samples]) - null_val
    se = float(raw.std(ddof=1) / np.sqrt(n))
    return float(raw.mean() / se) if se > 0 else float("nan")


def run_ppi_effect_check(
    sources: list[JudgeBiasSource], active_tests: list[str], n_reps: int, n_boot: int,
    gold_null_mc: int = 3000, progress_mode: str = "bar", seed: int = 44, n_workers: int = 1,
) -> list[PPIEffectResult]:
    """Bias and CI-coverage check for the PPI-corrected point estimate itself.

    Complements run_ppi_simulation's Type-I check ("does the p-value stay
    calibrated") with "is the estimate centered at the truth, and does its CI
    cover that truth at the nominal rate" -- ported from
    sim_type_i_calibration.py's second check.
    """
    effect_tests = [t for t in active_tests if t in _PPI_EFFECT_TESTS]
    if not effect_tests:
        return []
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(sources))]

    reporter = _ProgressReporter(len(sources), mode=progress_mode, label="pvalues-ppi-effect")
    results: list[PPIEffectResult] = []

    if n_workers > 1:
        args_list = [(i, sc, effect_tests, n_reps, n_boot, seed) for i, (sc, seed) in enumerate(zip(sources, child_seeds))]
        ctx = _mp.get_context("fork")
        gold_nulls = [estimate_judge_bias_gold_null_values(sc, n_mc=gold_null_mc, seed=int(child_seeds[i][0]))
                      for i, sc in enumerate(sources)]
        with ctx.Pool(n_workers) as pool:
            for i, (sc_idx, samples_by_test) in enumerate(pool.imap_unordered(_run_ppi_effect_cell_worker, args_list)):
                sc = sources[sc_idx]
                gold_null = gold_nulls[sc_idx]
                for t in effect_tests:
                    samples = samples_by_test.get(t, [])
                    null_val = gold_null.get(t, 0.0)
                    bias_mean, z, coverage, mean_width, n = _effect_cell_stats(samples, null_val)
                    if n == 0:
                        continue  # test not valid for this scenario's eval_type (e.g. binary) -- see _ppi_effective_tests
                    unc_z = _uncorrected_bias_z(samples, null_val)
                    results.append(PPIEffectResult(
                        name=sc.name, tag=sc.tag, test=t, n=sc.n, n_samples=n,
                        null_value=null_val, mean_bias=bias_mean, bias_z=z,
                        uncorrected_bias_z=unc_z, coverage=coverage, mean_ci_width=mean_width,
                    ))
                reporter.update(i + 1)
        reporter.update(len(sources), detail="done")
        return results

    for i, sc in enumerate(sources):
        gold_null = estimate_judge_bias_gold_null_values(sc, n_mc=gold_null_mc, seed=int(child_seeds[i][0]))
        samples_by_test = _run_ppi_effect_cell(sc, effect_tests, n_reps, n_boot, child_seeds[i])
        for t in effect_tests:
            samples = samples_by_test.get(t, [])
            null_val = gold_null.get(t, 0.0)
            bias_mean, z, coverage, mean_width, n = _effect_cell_stats(samples, null_val)
            if n == 0:
                continue  # test not valid for this scenario's eval_type (e.g. binary) -- see _ppi_effective_tests
            unc_z = _uncorrected_bias_z(samples, null_val)
            results.append(PPIEffectResult(
                name=sc.name, tag=sc.tag, test=t, n=sc.n, n_samples=n,
                null_value=null_val, mean_bias=bias_mean, bias_z=z,
                coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
            ))
        reporter.update(i + 1, detail=f"{sc.name}")
    reporter.update(len(sources), detail="done")
    return results


# ---------------------------------------------------------------------------
# PPI mode, estimator comparison: for ONE representative paired-mean estimand
# (paired_t -- generalizes to binary too since a proportion is just the mean
# of a 0/1 variable, and it's already documented elsewhere in this file as
# the "reasonable default" for that estimand), five ways of turning (sparse
# human labels + biased LLM-judge scores) into a hypothesis test, compared
# head to head on the SAME draws:
#   all_human      -- oracle: classical paired t-test on the FULL, dense
#                      ground truth (as if every item had a human label).
#   human_subset   -- classical paired t-test on ONLY the labeled subset's
#                      ground truth (small n_lab) -- the "why not just
#                      collect more human labels instead of trusting a
#                      correction" baseline a skeptical reviewer will ask
#                      about.
#   llm_only       -- classical paired t-test on the full LLM-judge scores,
#                      uncorrected (same number as run_ppi_cell's
#                      "uncorrected" arm, recomputed here for encapsulation).
#   llm_impute     -- classical paired t-test on the LLM-judge scores with
#                      labeled positions' values OVERWRITTEN by the true
#                      human label (a naive missing-data-imputation
#                      baseline with NO PPI rectifier) -- shows that simply
#                      "filling in what you know" is not the same as
#                      properly correcting for what you don't.
#   ppi            -- PPI-corrected (evalstats.tests._ppi_paired_arrays).
# Only paired_t's structure (cell.llm_x/llm_y/lab_x/lab_y/truth_x/truth_y) is
# used -- extending this same comparison to every PPI_TEST_METHODS estimand
# would multiply the plot count for no real gain in what it demonstrates;
# one clear, representative estimand is the point of this check.
# ---------------------------------------------------------------------------


@dataclass
class PPIComparisonResult:
    """One cell comparing PPI against the naive/human-only/judge-only
    baselines above, for a single representative estimand (paired_t)."""

    name: str
    tag: str  # "power" (vs. effect_size, reusing build_ppi_power_sources) | "compare_label_frac" (vs. label_frac)
    eval_type: str
    n: int
    n_reps: int
    effect_size: float
    """The eval-type-RELATIVE effect-size fraction (see
    build_ppi_power_sources/_jb_effect_magnitude), not JudgeBiasSource's raw
    absolute effect_size field -- the raw value is scaled by each eval
    type's own EVAL_TYPE_SCALE_BOUNDS span, so it isn't comparable across
    eval types (continuous vs. likert would show different x-axis values
    for "the same" relative effect). This field IS comparable across eval
    types; see _ppi_source_effect_frac."""
    label_frac: float
    n_lab: int
    """REALIZED labeled-item count (measured off the actual mask each
    replicate produces, not the nominal `n * label_frac`) -- see
    _JB_MIN_LAB: label_frac alone can be misleading once the floor binds
    (e.g. label_frac=0.05 and 0.10 both floor to n_lab=15 at n=100), so this
    is the field to plot/group by, not label_frac, whenever comparing
    across different n. For "independent"-mask structures (group/group3:
    ttest_welch, mwu, anova_ind, kruskal) this is the FIRST group's
    labeled count specifically -- see _run_ppi_comparison_cell's docstring
    for why every group is expected to match under this harness's scenario
    construction."""
    method: str = PAIRED_T.name
    """Which classical test this result is for -- see _COMPARISON_METHODS.
    Defaults to paired_t for backward compatibility; every
    _run_ppi_comparison_cell call now sets this explicitly to one of
    ttest_welch/paired_t/mwu/wilcoxon (never a pooled "average" tag --
    pooling across methods happens downstream, over a list of these, via
    pool_ppi_comparison_across_methods)."""
    rejects_all_human: int = 0
    rejects_human_subset: int = 0
    rejects_llm_only: int = 0
    rejects_llm_impute: int = 0
    rejects_ppi: int = 0
    n_failed: int = 0
    var_human_subset: float = float("nan")
    """Variance of the human-subset arm's POINT ESTIMATE across replicates.

    With var_ppi this yields a label-efficiency multiplier needing no power
    curve: the control-variate factor IS a variance ratio, and
    Var(classical)/Var(PPI) measures it directly -- no inversion, so no flat
    region, no clamping, no conditioning gate, and every cell reports.

    That matters because the power-curve route demonstrably breaks where the
    curve flattens. On binary's top tier (a 2% flip-rate judge) the inverted
    multiplier ran 1.24-1.37x the control-variate bound -- impossible -- while
    the direct variance ratio came in at 0.94x of it, i.e. sound. The excess
    was entirely inversion error, not the estimator.

    NaN when the arm produced too few finite estimates to take a variance."""
    var_ppi: float = float("nan")
    """Variance of the PPI estimator's point estimate. See var_human_subset."""
    n_est: int = 0
    """Replicates behind var_*: both arms must have produced a finite estimate
    in the SAME replicate, so this can sit below n_reps."""
    rho2_implied_se: float = float("nan")
    """Monte-Carlo standard error of the rho^2 implied by var_human_subset /
    var_ppi, from a paired bootstrap over the REPLICATE index (see
    _var_ratio_bootstrap_se).

    Here because a variance estimated from R replicates carries relative SE
    ~sqrt(2/R) -- ~10% at R=200 -- and the paired arm is worse than that: the
    paired difference D = truth_x - truth_y is heavier-tailed than either
    group's scores, so var_human_subset converges more slowly for a "pair"
    structure than a "group" one. Measured on the rho-drift cell at d=0,
    paired_t's implied rho^2 reads -17.6% against its own rho2_score at
    R=200, -3.8% at R=600 and +0.3% at R=1500, while ttest -- same cell, same
    draws -- moves only +4.0% / -3.3% / -0.5%. Without this column the R=200
    reading is indistinguishable from a real estimator defect, which is
    exactly the confusion the drift check's own control ran into.

    NaN when n_est is too small to bootstrap."""


def _var_ratio_bootstrap_se(est_hs, est_ppi, n_lab: int, n_total: int,
                            seed, n_boot: int = 2000) -> float:
    """Monte-Carlo SE of the implied rho^2, by PAIRED bootstrap over replicates.

    est_hs[i] and est_ppi[i] are the two arms' point estimates from the SAME
    replicate i, so the resample must draw replicate INDICES and keep the pair
    together -- the two variances are strongly positively correlated (they
    share a draw), and resampling them independently would overstate the SE of
    their ratio by treating that correlation as noise.

    Returns the SE of rho2 = (1 - 1/M) / (1 - n_lab/n_total), M = var_hs/var_ppi
    -- i.e. of the quantity RhoDriftPoint.rho2_implied reports, so it can be
    compared against a tolerance directly. NaN if there is too little to
    resample or the denominator degenerates."""
    a = np.asarray(est_hs, float)
    b = np.asarray(est_ppi, float)
    if a.size < 8 or a.size != b.size or not n_total or n_lab >= n_total:
        return float("nan")
    frac_unlab = 1.0 - n_lab / n_total
    if frac_unlab <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    # Chunked so the index array stays bounded: at n_reps=20000 a single
    # (n_boot, n) draw is 20000*2000 int64 = ~320 MB, and the drift sweep runs
    # one of these PER WORKER. Capping the batch at ~4M indices holds it near
    # 32 MB with no change to the estimator (the draws are still iid); only the
    # RNG consumption order differs, so SEs are not bit-comparable with values
    # produced before this was chunked.
    n = a.size
    per_batch = max(1, min(n_boot, int(4_000_000 // n)))
    parts = []
    drawn = 0
    while drawn < n_boot:
        k = min(per_batch, n_boot - drawn)
        idx = rng.integers(0, n, size=(k, n))
        va = a[idx].var(axis=1)
        vb = b[idx].var(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = va / vb
            parts.append((1.0 - 1.0 / M) / frac_unlab)
        drawn += k
    rho2 = np.concatenate(parts)
    rho2 = rho2[np.isfinite(rho2)]
    if rho2.size < 8:
        return float("nan")
    return float(np.std(rho2, ddof=1))


def _ppi_source_effect_frac(sc: JudgeBiasSource) -> float:
    """Eval-type-relative effect-size fraction for a comparison-sweep
    source -- see PPIComparisonResult.effect_size's docstring. Every tag
    besides "power" must be listed explicitly here (not defaulted to
    PPI_COMPARISON_MODERATE_EFFECT_FRAC): that fallback happens to be
    correct for "compare_label_frac"/"nlab_grid_power" (both built at
    exactly that fraction) but was silently WRONG for "nlab_grid"
    (build_ppi_nlab_grid_sources' effect_frac=0.0 calibration grid) before
    this was caught -- every nlab_grid PPIComparisonResult reported
    effect_size=0.20 instead of 0.0 in its CSV/log output, even though the
    underlying simulation itself used effect_size=0.0 correctly (this field
    is metadata only; JudgeBiasSource.effect_size, not this function, is
    what generate_judge_bias_cell actually reads)."""
    if sc.tag in ("power", "power_binary"):
        return _parse_ppi_power_name(sc.name)[1]
    if sc.tag in ("nlab_grid", "nlab_grid_binary", "irr_peak"):
        return 0.0
    if sc.tag in ("compare_label_frac", "nlab_grid_power", "complab_binary", "nlab_grid_power_binary"):
        return PPI_COMPARISON_MODERATE_EFFECT_FRAC
    if sc.tag in ("label_eff", "label_eff_binary"):
        return PPI_LABEL_EFF_EFFECT_FRAC
    if sc.tag in ("factorial", "factorial_binary"):
        m = re.search(r"\.es=([a-z]+)\.", sc.name)
        if not m:
            raise ValueError(f"_ppi_source_effect_frac: could not parse es label from {sc.name!r}")
        return PPI_FACTORIAL_EFFECT_FRACS[m.group(1)]
    if sc.tag in ("nformula", "nformula_binary", "rho_drift"):
        m = re.search(r"\.es=([\d.]+)$", sc.name)
        if not m:
            raise ValueError(f"_ppi_source_effect_frac: could not parse es frac from {sc.name!r}")
        return float(m.group(1))
    raise ValueError(f"_ppi_source_effect_frac: unrecognized tag {sc.tag!r}")


_COMPARISON_METHODS = (TTEST.name, TTEST_WELCH.name, PAIRED_T.name, MWU.name, WILCOXON.name)
"""The five classical two-sample/paired tests the PPI estimator-comparison
sweep (and everything downstream: N x N_lab grid, full factorial, the
null-effect bar chart) runs and, by default, averages across -- rather than
paired_t alone. ttest/ttest_welch/paired_t (mean-based) and mwu/wilcoxon
(rank-based) cover both the independent-two-group and paired structures, and
all five test the SAME two-group mean/location-shift question via different
classical machinery, so averaging their rejection rates is a coherent
summary of "does this hold across reasonable test choices." ttest (plain
Student's, equal_var=True) sits alongside ttest_welch (Welch's,
equal_var=False) -- both share the identical PPI correction (the
mean-difference estimator; PPI correction doesn't depend on the classical
equal-variance assumption at all), differing only in which classical
uncorrected test computes the "uncorrected" arm's p-value. Both are kept
so save_ppi_typeI_plot's 9-test roster (the main Type-I sweep,
build_judge_bias_sources) and _COMPARISON_METHODS + _COMPARISON_METHODS_
OMNIBUS (9 = 5 + 4) line up exactly, letting the factorial-sourced
Type-I-by-test violin plot (save_ppi_factorial_typeI_violin_plot) show the
same 9 tests the OFAT-sourced one does. Uses MWU (evalstats.tests.
_ppi_two_sample's single-global-rectifier midrank correction). A
per-group, per-score-bin local-rectifier alternative existed
(mwu_mnar_experimental and three variants): it fixed real MNAR-labeling
miscalibration MWU has, but cost real MCAR calibration doing so, and was
removed on 2026-08-21 after proving badly broken on binary data even under
MCAR -- see MWU's comment in methods.py. Given this project's stance that PPI
requires MCAR labeling and treats MNAR as a documented, out-of-scope
limitation, paying that MCAR cost for MNAR robustness is the wrong trade
here too -- same reasoning _COMPARISON_METHODS_OMNIBUS already applies to
kruskal vs. kruskal_mnar_experimental. Deliberately excludes the
omnibus/multi-group tests (anova_ind/anova_rep/friedman/kruskal/lmm*) and
the non-standard bootstrap-CI constructions (bayes_bootstrap/bootstrap_t/
mj_floor) -- those answer different questions (multi-group omnibus
effects, CI-based constructions), so folding them into the same "pooled
false-positive rate" would blend apples with oranges rather than checking
robustness across reasonable alternatives, the same way
build_ppi_factorial_sources/build_ppi_nlab_grid_sources' paired_t-only
scoping was never meant to claim the other PPI_TEST_METHODS behave
identically."""
_COMPARISON_METHODS_OMNIBUS = (ANOVA_IND.name, ANOVA_REP.name, FRIEDMAN.name, KRUSKAL.name)
"""The four omnibus/multi-group tests -- run alongside _COMPARISON_METHODS
against the same factorial sources (build_ppi_factorial_sources), using the
same 5-way (all_human/human_subset/llm_only/llm_impute/ppi) machinery, but
never pooled together with _COMPARISON_METHODS into one averaged rate: these
answer a genuinely different question (are the 3 groups/conditions
different at all, vs. _COMPARISON_METHODS' specific two-group location-shift
question) -- see _COMPARISON_METHODS' own docstring for why blending the two
would be apples-with-oranges. anova_ind/kruskal use the
independent-3-group structure (a3/b3/c3); anova_rep/friedman use the
repeated-3-group structure (A/B/C) -- see _COMPARISON_METHOD_STRUCTURE's
"group3"/"pair3" entries. Uses KRUSKAL (evalstats.tests.
_ppi_kruskal_wallis_pairwise's single-global-rectifier Wald test), not
KRUSKAL_MNAR_EXPERIMENTAL (evalstats.tests.
_ppi_kruskal_wallis_pairwise_mnar_experimental's per-group, per-score-bin
local rectifier) -- the same choice _COMPARISON_METHODS makes for MWU
(whose local-rectifier alternatives were removed outright), and for a
documented reason, not an oversight: the
local rectifier fixes the same combined bias x MNAR-labeling x coarse-scale
x large-N miscalibration MWU/kruskal's global rectifier both have, but
costs real MCAR calibration doing so in both cases -- a regression it
introduces, not one it inherits. A shrinkage/partial-pooling variant meant
to recover kruskal's MCAR cost without losing the MNAR fix made both worse
instead. Given this project's stance that PPI requires MCAR labeling and
treats MNAR as a documented, out-of-scope limitation rather than something
to actively correct for, paying an MCAR cost for MNAR robustness users are
already told not to rely on is the wrong trade -- see KRUSKAL/
KRUSKAL_MNAR_EXPERIMENTAL's Method docstrings in methods.py for the full
writeup. KRUSKAL_MNAR_EXPERIMENTAL remains selectable via --tests
kruskal_mnar_experimental for anyone deliberately studying the MNAR
question, just not part of this pooled/official comparison.
Pool these among THEMSELVES (pool_ppi_comparison_across_methods, or a
filtered subset of `results`) for their own "mean_of_4_omnibus" summary,
kept in its own report section/log rather than merged into the headline
_COMPARISON_METHODS one."""
_COMPARISON_METHOD_STRUCTURE = {
    TTEST.name: "group", TTEST_WELCH.name: "group", MWU.name: "group",
    PAIRED_T.name: "pair", WILCOXON.name: "pair",
    ANOVA_IND.name: "group3", KRUSKAL.name: "group3", KRUSKAL_MNAR_EXPERIMENTAL.name: "group3",
    KRUSKAL_INFLUENCE.name: "group3", KRUSKAL_INFLUENCE_FLOOR.name: "group3",
    ANOVA_REP.name: "pair3", FRIEDMAN.name: "pair3",
}
_COMPARISON_METHODS_LABEL = "ttest/ttest_welch/paired_t/mwu/wilcoxon"
_COMPARISON_METHODS_OMNIBUS_LABEL = "anova_ind/anova_rep/friedman/kruskal"
_COMPARISON_METHODS_BINARY = (TTEST_WELCH.name, PAIRED_T.name)
"""The 2 of _COMPARISON_METHODS' 4 pooled tests that are valid on binary's
heavily-tied 0/1 data (mwu/wilcoxon are rank-based and break down under
that many ties -- the same reason build_judge_bias_sources' _PPI_BINARY_
COMPATIBLE_TESTS excludes them). Run and pooled SEPARATELY from
_COMPARISON_METHODS via its own run_ppi_comparison_simulation call (never
blended into the same averaged rate -- pooling a 2-method and a 4-method
rate under one "false positive/power rate" figure would be apples-to-
oranges, and binary sources are never fed to a _COMPARISON_METHODS call in
the first place)."""
_COMPARISON_METHODS_BINARY_LABEL = "ttest_welch/paired_t"
POOLED_METHOD_LABEL = "mean_of_4"
"""PPIComparisonResult.method value for a row produced by
pool_ppi_comparison_across_methods -- distinguishes a pooled/averaged row
from a genuine single-method one (never a value _run_ppi_comparison_cell
itself produces). Used for both the _COMPARISON_METHODS pool and the
_COMPARISON_METHODS_OMNIBUS pool -- callers keep the two separate by never
pooling a `results` list that mixes both method sets together (see
_COMPARISON_METHODS_OMNIBUS' docstring)."""


def _classical_pvalue(a: np.ndarray, b: np.ndarray, method: str, structure: str) -> float:
    """The SAME classical-test call _run_ppi_cell uses for this method
    (ttest_ind/mannwhitneyu for "group", ttest_rel/wilcoxon for "pair"),
    factored out here so it can be reused identically for all_human,
    human_subset, llm_only, and llm_impute -- every arm of the comparison
    for a given method uses the SAME test, just on different input arrays,
    so the comparison is apples-to-apples per method (e.g. the "oracle"
    all_human/human_subset arms run Mann-Whitney on truth for the
    "mwu" method-rows, not always a t-test)."""
    if structure == "group":
        if method == TTEST.name:
            return float(scipy_stats.ttest_ind(a, b, equal_var=True).pvalue)
        if method == TTEST_WELCH.name:
            return float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
        return float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    if method == PAIRED_T.name:
        return float(scipy_stats.ttest_rel(a, b).pvalue)
    return float(scipy_stats.wilcoxon(a, b, alternative="two-sided").pvalue)


def _classical_point_estimate(a: np.ndarray, b: np.ndarray, method: str, structure: str) -> float:
    """The classical arm's POINT ESTIMATE of the same estimand the PPI arm
    targets, so their variances across replicates are a ratio of like for like.

    Mirrors _classical_pvalue's dispatch: mean difference for the t-tests,
    P(X>Y) midrank for Mann-Whitney, Walsh-average theta for Wilcoxon -- the
    functionals evalstats.ppi.correct is asked to correct in each case.

    Exists for the variance-route multiplier (see
    PPIComparisonResult.var_human_subset), which sidesteps the power-curve
    inversion entirely."""
    if structure == "group":
        if method in (TTEST.name, TTEST_WELCH.name):
            return float(np.mean(a) - np.mean(b))
        return float(_p_x_gt_y_midrank(a, b) - 0.5)
    if method == PAIRED_T.name:
        return float(np.mean(np.asarray(a) - np.asarray(b)))
    from evalstats.ppi import paired_walsh_midrank_theta
    return float(paired_walsh_midrank_theta(np.asarray(a) - np.asarray(b)))


def _ppi_comparison_pvalue(a: np.ndarray, b: np.ndarray, a_lab: np.ndarray, b_lab: np.ndarray, method: str, structure: str, n_boot: int, seed: int, power_tune: bool = True, return_result: bool = False):
    """The SAME PPI-corrected call _run_ppi_cell uses for this method
    (_ppi_two_sample_t_interval for ttest/ttest_welch, _ppi_two_sample for
    mwu, _ppi_paired_arrays for "pair" methods -- see _run_ppi_cell's
    ttest/ttest_welch/mwu/paired_t/wilcoxon blocks, which this mirrors
    exactly).

    power_tune : forwarded to _ppi_two_sample_t_interval/_ppi_two_sample/
    _ppi_paired_arrays as-is (see evalstats.ppi.correct's power_tune
    parameter). Default True matches the production default;
    --factorial-no-power-tune sets this False for a head-to-head
    comparison run -- see that flag's help text."""
    if structure == "group":
        if method in (TTEST.name, TTEST_WELCH.name):
            # Identical PPI correction for both -- PPI's mean-difference
            # estimator doesn't depend on the classical equal-variance
            # assumption at all; only _classical_pvalue's UNCORRECTED arm
            # differs between them (equal_var=True vs False). Closed-form
            # construction (see the matching _run_ppi_cell TTEST/
            # TTEST_WELCH blocks for why -- Addendum 29/31/32).
            _r = _ppi_two_sample_t_interval(a, b, a_lab, b_lab, _ALPHA, power_tune=power_tune)
            return _r if return_result else _r.p_value
        # MWU (global rectifier): the only midrank PPI correction --
        # see _COMPARISON_METHODS's docstring for why the local-rectifier
        # alternatives were removed rather than kept as options.
        # Routed through the SAME helper mannwhitney() uses, so this sweep
        # measures the shipped test rather than a parallel construction.
        _e, _ci, _p, _rec, _lam = _ppi_mannwhitney_corrected(
            a, b, a_lab, b_lab, _ALPHA, n_boot, seed, power_tune=power_tune)
        _r = _MWUResult(_p, _e)
        return _r if return_result else _r.p_value
    # paired_t: np.mean. wilcoxon: paired_walsh_midrank_theta (evalstats.ppi --
    # a Hodges-Lehmann Walsh-average midrank-sign statistic), NOT np.median --
    # see that function's docstring for why the median of a paired difference
    # is the WRONG estimand under heavy ties (e.g. likert's integer-rounded
    # truth), independent of any bootstrap-degeneracy jitter fix, and why a
    # simpler per-item sign proportion (tried first) also isn't safe (severely
    # inflated Type-I error at small n_lab against real, heavily-tied data).
    statistic = np.mean if method == PAIRED_T.name else paired_walsh_midrank_theta
    _r = _ppi_paired_arrays(a, b, a_lab, b_lab, statistic, _ALPHA, n_boot, seed, rectifier_func=statistic, power_tune=power_tune)
    return _r if return_result else _r.p_value


def _classical_pvalue_omnibus(groups: list[np.ndarray], method: str) -> float:
    """Omnibus counterpart to _classical_pvalue, for the 3-group
    _COMPARISON_METHODS_OMNIBUS methods -- the SAME uncorrected-p-value
    calls _run_ppi_cell's anova_ind/anova_rep/friedman/kruskal blocks use
    (_uncorrected_anova_independent_p_value etc.), reused identically here
    for all_human/human_subset/llm_only/llm_impute."""
    if method == ANOVA_IND.name:
        return _uncorrected_anova_independent_p_value(groups)
    if method == ANOVA_REP.name:
        return _uncorrected_anova_repeated_p_value(groups)
    if method == FRIEDMAN.name:
        return _uncorrected_friedman_p_value(groups)
    return _uncorrected_kruskal_p_value(groups)  # KRUSKAL.name / KRUSKAL_MNAR_EXPERIMENTAL.name (same uncorrected test)


def _classical_point_estimate_omnibus(groups: list[np.ndarray], method: str) -> float:
    """Omnibus counterpart to _classical_point_estimate: a SCALAR summary of
    the same estimand each omnibus correction targets, so the classical and
    PPI arms' variances are a ratio of like for like.

    The estimand per method, chosen to match what the shipped correction
    actually corrects rather than to be uniform:

      anova_ind  weighted between-group variance of the group means
                 (_anova_between_variance_from_groups)
      anova_rep  between-condition variance after removing subject means
                 (_repeated_condition_variance)
      friedman   condition variance of WITHIN-SUBJECT ranks
                 (_friedman_rank_variance) -- the rank analogue of anova_rep
      kruskal    mean squared pairwise dominance theta. Kruskal is the odd
                 one out: its correction estimates a VECTOR of pairwise
                 P_mid(a>b) values (see _kw_pairwise_thetas' docstring on why
                 global pooled ranks cannot be subset to the labeled items),
                 not a variance component, so the scalar summary is taken on
                 that vector instead.

    NOTE this defines a NEW estimand for the variance-route multiplier where
    none existed -- previously _run_ppi_comparison_cell recorded a p-value
    only for omnibus methods and left var_human_subset/var_ppi as NaN. The
    resulting multiplier is only meaningful if the same functional is read
    off both arms, which is why the PPI side
    (:func:`_ppi_point_estimate_omnibus`) recovers the SAME quantity rather
    than reusing the f-statistic directly."""
    from evalstats.tests import (
        _anova_between_variance_from_groups, _repeated_condition_variance,
        _friedman_rank_variance, _kw_pairwise_thetas,
    )
    if method == ANOVA_IND.name:
        return float(_anova_between_variance_from_groups(groups))
    if method in (ANOVA_REP.name, FRIEDMAN.name):
        # pair3: every condition must hold the SAME subjects in the same
        # order (_COMPARISON_CELL_FIELDS marks it "shared" masking), which is
        # what makes column_stack meaningful. Return NaN rather than raising
        # on a ragged input -- the caller treats NaN as "this replicate
        # contributes no paired point", which is the right outcome, whereas
        # an exception there would be swallowed and look like a pass.
        lens = {len(g) for g in groups}
        if len(lens) != 1 or lens == {0}:
            return float("nan")
        mat = np.column_stack(groups)
        return float(_repeated_condition_variance(mat) if method == ANOVA_REP.name
                     else _friedman_rank_variance(mat))
    k = len(groups)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    th = _kw_pairwise_thetas(groups, pairs)
    return float(np.mean(np.asarray(th, dtype=float) ** 2))


def _ppi_omnibus_pvalue_and_estimate(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], method: str,
    n_boot: int, seed: int, power_tune: bool = True,
) -> tuple[float | None, float]:
    """Both the corrected p-value and the matched point estimate from ONE
    fit, for the omnibus methods.

    Exists purely to avoid paying for the correction twice. The p-value and
    the point estimate come from the same underlying object in every case --
    the F-stat dict for anova_ind/anova_rep/friedman, the pairwise-Wald dict
    for kruskal -- so calling _ppi_comparison_pvalue_omnibus and
    _ppi_point_estimate_omnibus separately re-ran the whole correction. For
    kruskal that meant running _ppi_kruskal_wallis_pairwise's n_boot-resample
    bootstrap TWICE per replicate, which dominated the rho-drift sweep's
    runtime (measured: >10 min for a 50-rep sweep that should take ~2).

    Returns (p_value, point_estimate); either may be None/NaN when the
    correction declines to fit, matching how each route treated that before."""
    from evalstats.tests import (
        _ppi_kruskal_wallis_pairwise, _ppi_kruskal_wallis_pairwise_mnar_experimental,
        _ppi_kruskal_wallis_influence,
    )
    _KW_FNS = {
        KRUSKAL.name: _ppi_kruskal_wallis_influence,
        KRUSKAL_MNAR_EXPERIMENTAL.name: _ppi_kruskal_wallis_pairwise_mnar_experimental,
        # The influence variants share KRUSKAL's estimator and differ ONLY in
        # the Wald covariance, so they belong on this same deduplicated path.
        KRUSKAL_INFLUENCE.name: _ppi_kruskal_wallis_influence,
        KRUSKAL_INFLUENCE_FLOOR.name: functools.partial(
            _ppi_kruskal_wallis_influence, loo_group=False, floor_frac=0.5),
    }
    try:
        if method in _KW_FNS:
            # The only genuinely expensive duplicate: this runs an
            # n_boot-resample bootstrap. Call it ONCE and take both outputs
            # from the same dict -- "wald_p" is exactly the field
            # _ppi_comparison_pvalue_omnibus returns, so the p-value is
            # bit-identical to the un-deduplicated path.
            fn = _KW_FNS[method]
            pw = fn(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
            th = np.asarray(pw["theta_hat"], dtype=float)
            est = float(np.mean(th ** 2)) if th.size else float("nan")
            return pw["wald_p"], est

        # anova_ind / anova_rep / friedman: keep calling the SHIPPED p-value
        # function rather than re-deriving p from the f-stat dict. Those
        # functions do more than F.sf on the raw statistic (see
        # _ppi_anova_independent_p_value's docstring on the variance-inflation
        # rescaling), and silently substituting an equivalent-looking formula
        # would change simulation output. The extra _ppi_*_f_stat call this
        # costs is closed-form with no bootstrap, so it is cheap.
        p = _ppi_comparison_pvalue_omnibus(groups, groups_lab, method, n_boot, seed)
        est = _ppi_point_estimate_omnibus(groups, groups_lab, method, n_boot, seed,
                                          power_tune=power_tune)
        return p, est
    except Exception:
        return None, float("nan")


def _ppi_point_estimate_omnibus(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], method: str,
    n_boot: int, seed: int, power_tune: bool = True,
) -> float:
    """PPI-corrected counterpart of :func:`_classical_point_estimate_omnibus`,
    read off the SAME shipped corrections whose p-values
    _ppi_comparison_pvalue_omnibus uses -- so the variance ratio measures the
    shipped behaviour, not a re-implementation.

    For the three F-based methods the corrected between-condition variance is
    recovered from the returned dict as ``f_corr * dfn * denom / scale``:
    ``f_corr = (SS_condition/dfn) / denom`` by construction, so that product
    is ``SS_condition``, and ``scale`` is the same N (or n_subjects*k) the
    classical helpers divide by -- putting both arms on one variance scale.
    Verified numerically against _anova_between_variance_from_groups.

    Kruskal instead exposes its corrected estimand directly as ``theta_hat``,
    so the same mean-square summary is applied to that vector.

    Returns NaN when the correction declines to fit (the F-stat helpers can
    return None on a degenerate fit), matching how the p-value route treats
    that case."""
    from evalstats.tests import (
        _ppi_anova_independent_f_stat, _ppi_anova_repeated_f_stat,
        _ppi_friedman_f_stat, _ppi_kruskal_wallis_pairwise,
    )
    k = len(groups)
    try:
        if method == ANOVA_IND.name:
            d = _ppi_anova_independent_f_stat(groups, groups_lab, k=k, power_tune=power_tune)
        elif method == ANOVA_REP.name:
            d = _ppi_anova_repeated_f_stat(groups, groups_lab, k=k, power_tune=power_tune)
        elif method == FRIEDMAN.name:
            d = _ppi_friedman_f_stat(groups, groups_lab, k=k, power_tune=power_tune)
        else:
            pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA,
                                              n_boot=n_boot, rng=seed)
            th = np.asarray(pw["theta_hat"], dtype=float)
            return float(np.mean(th ** 2)) if th.size else float("nan")
        if not d:
            return float("nan")
        scale = float(d.get("scale", 0.0))
        if scale <= 0:
            return float("nan")
        return float(d["f_corr"]) * float(d["dfn"]) * float(d["denom"]) / scale
    except Exception:
        return float("nan")


def _ppi_comparison_pvalue_omnibus(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], method: str, n_boot: int, seed: int,
) -> float | None:
    """Omnibus counterpart to _ppi_comparison_pvalue -- the SAME
    PPI-corrected calls _run_ppi_cell's anova_ind/anova_rep/friedman/kruskal
    blocks use, reused identically here. May return None (anova_ind/
    anova_rep/friedman's PPI-corrected p-value functions can return None on
    a degenerate fit -- see their own docstrings); the caller must treat
    that as "not rejected," matching _run_ppi_cell's `p is not None and p <
    alpha` pattern."""
    k = len(groups)
    if method == ANOVA_IND.name:
        return _ppi_anova_independent_p_value(groups, groups_lab, k=k)
    if method == ANOVA_REP.name:
        return _ppi_anova_repeated_p_value(groups, groups_lab, k=k)
    if method == FRIEDMAN.name:
        return _ppi_friedman_p_value(groups, groups_lab, k=k)
    if method == KRUSKAL.name:
        pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
        return pw["wald_p"]
    if method == KRUSKAL_INFLUENCE.name:
        pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
        return pw["wald_p"]
    if method == KRUSKAL_INFLUENCE_FLOOR.name:
        pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed,
                                           loo_group=False, floor_frac=0.5)
        return pw["wald_p"]
    # KRUSKAL_MNAR_EXPERIMENTAL.name
    pw = _ppi_kruskal_wallis_pairwise_mnar_experimental(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
    return pw["wald_p"]


_COMPARISON_CELL_FIELDS = {
    "group": (("llm_a2", "llm_b2"), ("lab_a2", "lab_b2"), ("truth_a2", "truth_b2"), "independent"),
    "pair": (("llm_x", "llm_y"), ("lab_x", "lab_y"), ("truth_x", "truth_y"), "shared"),
    "group3": (("llm_a3", "llm_b3", "llm_c3"), ("lab_a3", "lab_b3", "lab_c3"), ("truth_a3", "truth_b3", "truth_c3"), "independent"),
    "pair3": (("llm_A", "llm_B", "llm_C"), ("lab_A", "lab_B", "lab_C"), ("truth_A", "truth_B", "truth_C"), "shared"),
}
"""Maps each _COMPARISON_METHOD_STRUCTURE value to the JudgeBiasCellData
field names it reads, and whether its labeling mask is "independent" (each
group masked separately, e.g. group/group3's _jb_labels_independent) or
"shared" (one mask reused across every group, e.g. pair/pair3's
_jb_labels_shared) -- see _run_ppi_comparison_cell."""


def _run_ppi_comparison_cell(sc: JudgeBiasSource, n_reps: int, n_boot: int, seed, method: str, power_tune: bool = True) -> PPIComparisonResult:
    """Runs the 5-way comparison (all_human/human_subset/llm_only/
    llm_impute/ppi) for ONE classical `method` (see _COMPARISON_METHODS/
    _COMPARISON_METHODS_OMNIBUS). Dispatches on
    _COMPARISON_METHOD_STRUCTURE[method] via _COMPARISON_CELL_FIELDS:
    "group"/"group3" methods (ttest_welch, mwu, anova_ind, kruskal) use
    generate_judge_bias_cell's independent-group structure (2 or 3 groups);
    "pair"/"pair3" methods (paired_t, wilcoxon, anova_rep, friedman) use its
    paired/repeated structure. generate_judge_bias_cell draws EVERY
    structure every replicate regardless of which this call's method needs
    (one rng stream, unused structures simply unused, not skipped -- keeps a
    given scenario/seed's draws identical regardless of which method is
    requested, the same reproducibility property --tests relies on in
    _run_ppi_cell).

    Each of the four arms is computed in its OWN try/except: a classical-
    test failure (e.g. wilcoxon raising on an all-zero-difference sample)
    just skips incrementing that arm for that replicate (same semantics as
    before this function supported rank-based tests, which never failed);
    only a PPI bootstrap-correction failure increments n_failed, preserving
    that field's original meaning. For "group3"/"pair3" methods, the
    PPI-corrected p-value can also be None on a degenerate fit (anova_ind/
    anova_rep/friedman -- see _ppi_comparison_pvalue_omnibus) -- treated as
    "not rejected," not a failure, matching _run_ppi_cell.

    n_lab (the realized labeled-item count) is the FIRST group's count --
    for "independent"-mask structures (group/group3) since every
    JudgeBiasSource this comparison-sweep machinery builds leaves n2/n3
    unset (so n2==n3==n) and applies the SAME label_frac to every group, so
    all groups are expected to match; for "shared"-mask structures
    (pair/pair3) every group shares one mask anyway, so the first group's
    count IS the shared count."""
    rng = np.random.default_rng(seed)
    rejects = {"all_human": 0, "human_subset": 0, "llm_only": 0, "llm_impute": 0, "ppi": 0}
    _est_hs: list = []   # human-subset point estimates, for the variance route
    _est_ppi: list = []  # PPI point estimates, paired with _est_hs by replicate
    n_failed = 0
    n_lab_realized = 0
    structure = _COMPARISON_METHOD_STRUCTURE[method]
    llm_fields, lab_fields, truth_fields, mask_kind = _COMPARISON_CELL_FIELDS[structure]
    is_omnibus = structure in ("group3", "pair3")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_reps):
            cell = generate_judge_bias_cell(sc, rng)
            llm_groups = [getattr(cell, f) for f in llm_fields]
            lab_groups = [getattr(cell, f) for f in lab_fields]
            truth_groups = [getattr(cell, f) for f in truth_fields]

            if mask_kind == "independent":
                masks = [~np.isnan(lab) for lab in lab_groups]
                subset_ok = all(int(m.sum()) >= 2 for m in masks)
            else:
                shared_mask = np.logical_and.reduce([~np.isnan(lab) for lab in lab_groups])
                masks = [shared_mask] * len(lab_groups)
                subset_ok = int(shared_mask.sum()) >= 2
            n_lab_realized = int(masks[0].sum())
            truth_subset_groups = [t[m] for t, m in zip(truth_groups, masks)]
            filled_groups = []
            for llm, lab, m in zip(llm_groups, lab_groups, masks):
                filled = llm.copy()
                filled[m] = lab[m]
                filled_groups.append(filled)

            if is_omnibus:
                classical = lambda groups: _classical_pvalue_omnibus(groups, method)  # noqa: E731
            else:
                classical = lambda groups: _classical_pvalue(groups[0], groups[1], method, structure)  # noqa: E731

            try:
                p_all_human = classical(truth_groups)
                rejects["all_human"] += int(p_all_human < _ALPHA)
            except Exception:
                pass

            try:
                p_llm_only = classical(llm_groups)
                rejects["llm_only"] += int(p_llm_only < _ALPHA)
            except Exception:
                pass

            try:
                p_llm_impute = classical(filled_groups)
                rejects["llm_impute"] += int(p_llm_impute < _ALPHA)
            except Exception:
                pass

            _e_hs = float("nan")
            if subset_ok:
                try:
                    p_human_subset = classical(truth_subset_groups)
                    rejects["human_subset"] += int(p_human_subset < _ALPHA)
                    if is_omnibus:
                        _e_hs = _classical_point_estimate_omnibus(truth_subset_groups, method)
                    else:
                        _e_hs = _classical_point_estimate(
                            truth_subset_groups[0], truth_subset_groups[1], method, structure)
                except Exception:
                    pass

            try:
                ppi_seed = int(rng.integers(0, 2 ** 31))
                if is_omnibus:
                    # ONE call for both -- see _ppi_omnibus_pvalue_and_estimate
                    # on why (kruskal's bootstrap was otherwise paid twice per
                    # replicate).
                    p_ppi, _e_ppi = _ppi_omnibus_pvalue_and_estimate(
                        llm_groups, lab_groups, method, n_boot, ppi_seed, power_tune=power_tune)
                    rejects["ppi"] += int(p_ppi is not None and p_ppi < _ALPHA)
                    # Same replicate-pairing rule as the two-group branch
                    # below: both arms must come from the SAME replicate or
                    # the variance ratio is a ratio of nothing.
                    if np.isfinite(_e_hs) and np.isfinite(_e_ppi):
                        _est_hs.append(_e_hs)
                        _est_ppi.append(_e_ppi)
                else:
                    _res = _ppi_comparison_pvalue(
                        llm_groups[0], llm_groups[1], lab_groups[0], lab_groups[1], method, structure, n_boot, ppi_seed,
                        power_tune=power_tune, return_result=True,
                    )
                    p_ppi = float(_res.p_value)
                    rejects["ppi"] += int(p_ppi < _ALPHA)
                    # Pair the two arms by REPLICATE: a variance ratio built
                    # from two differently-filtered sets of replicates is not a
                    # ratio of anything.
                    _e_ppi = float(getattr(_res, "estimate", float("nan")))
                    if np.isfinite(_e_hs) and np.isfinite(_e_ppi):
                        _est_hs.append(_e_hs)
                        _est_ppi.append(_e_ppi)
            except Exception:
                n_failed += 1

    return PPIComparisonResult(
        name=sc.name, tag=sc.tag, eval_type=sc.eval_type, n=sc.n, n_reps=n_reps,
        effect_size=_ppi_source_effect_frac(sc), label_frac=sc.label_frac, n_lab=n_lab_realized, method=method,
        rejects_all_human=rejects["all_human"], rejects_human_subset=rejects["human_subset"],
        rejects_llm_only=rejects["llm_only"], rejects_llm_impute=rejects["llm_impute"],
        rejects_ppi=rejects["ppi"], n_failed=n_failed,
        var_human_subset=float(np.var(_est_hs)) if len(_est_hs) > 2 else float("nan"),
        var_ppi=float(np.var(_est_ppi)) if len(_est_ppi) > 2 else float("nan"),
        n_est=len(_est_ppi),
        rho2_implied_se=_var_ratio_bootstrap_se(
            _est_hs, _est_ppi, n_lab_realized, sc.n, seed),
    )


def _run_ppi_comparison_cell_worker(args: tuple) -> PPIComparisonResult:
    sc, n_reps, n_boot, seed, method, power_tune = args
    return _run_ppi_comparison_cell(sc, n_reps, n_boot, seed, method, power_tune=power_tune)


def run_ppi_comparison_simulation(
    sources: list[JudgeBiasSource], n_reps: int, n_boot: int,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    methods: tuple = _COMPARISON_METHODS,
    power_tune: bool = True,
) -> list[PPIComparisonResult]:
    """Runs _run_ppi_comparison_cell for every (source, method) pair --
    len(sources) x len(methods) cells total -- returning a FLAT list (each
    PPIComparisonResult.method identifies which). Pool across methods with
    pool_ppi_comparison_across_methods for a single averaged row per
    scenario; group by .method for the per-method breakdown.

    n_workers=1 (the default) runs sequentially -- fine for the original
    ~24-scenario comparison grid (build_ppi_power_sources +
    build_ppi_comparison_label_frac_sources), where forking a worker pool
    would be pure overhead relative to the work itself. build_ppi_nlab_grid_
    sources (~44 scenarios), build_ppi_factorial_sources (~312), and now
    the x4 method sweep push this well past that point, so this supports
    the same fork-pool-over-sources pattern as run_ppi_simulation/
    run_multiarm_simulation (no in-cell progress-dict machinery, unlike
    run_ppi_simulation -- _run_ppi_comparison_cell is fast enough per
    (source, method) cell, seconds not minutes, that per-cell granularity
    isn't needed)."""
    cells = [(sc, m) for sc in sources for m in methods]
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-ppi-compare")
    results: list[PPIComparisonResult] = []
    if n_workers <= 1:
        for i, ((sc, m), child_seed) in enumerate(zip(cells, child_seeds)):
            results.append(_run_ppi_comparison_cell(sc, n_reps, n_boot, child_seed, m, power_tune=power_tune))
            reporter.update(i + 1, detail=f"{sc.name} [{m}]")
    else:
        args_list = [(sc, n_reps, n_boot, child_seed, m, power_tune) for (sc, m), child_seed in zip(cells, child_seeds)]
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_ppi_comparison_cell_worker, args_list)):
                results.append(result)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def pool_ppi_comparison_across_methods(results: list[PPIComparisonResult]) -> list[PPIComparisonResult]:
    """Pool PPIComparisonResult rows across _COMPARISON_METHODS, one output
    row per distinct scenario `name` (summing rejects/n_reps across
    whichever methods are present for that name -- equivalent to averaging
    each method's rate since every method shares the same n_reps per
    scenario). Output rows carry method=POOLED_METHOD_LABEL. This is the
    "average across ttest_welch/paired_t/mwu/wilcoxon" pooling requested for
    the headline figures (null-effect bar chart, 5-way comparison, N x
    N_lab grid, factorial slices) -- the per-method rows in `results`
    remain available (e.g. in the raw CSV) as the supplementary robustness
    breakdown, so a reviewer can check the average isn't hiding one method
    behaving badly."""
    by_name: dict[str, list[PPIComparisonResult]] = defaultdict(list)
    for r in results:
        by_name[r.name].append(r)

    pooled: list[PPIComparisonResult] = []
    for name, rows in by_name.items():
        r0 = rows[0]
        pooled.append(PPIComparisonResult(
            name=name, tag=r0.tag, eval_type=r0.eval_type, n=r0.n,
            n_reps=sum(r.n_reps for r in rows),
            effect_size=r0.effect_size, label_frac=r0.label_frac, n_lab=r0.n_lab,
            method=POOLED_METHOD_LABEL,
            rejects_all_human=sum(r.rejects_all_human for r in rows),
            rejects_human_subset=sum(r.rejects_human_subset for r in rows),
            rejects_llm_only=sum(r.rejects_llm_only for r in rows),
            rejects_llm_impute=sum(r.rejects_llm_impute for r in rows),
            rejects_ppi=sum(r.rejects_ppi for r in rows),
            n_failed=sum(r.n_failed for r in rows),
        ))
    return pooled


def _pool_ppi_comparison_rows(rows: list[PPIComparisonResult]) -> PPIComparisonResult | None:
    """Pool an arbitrary list of PPIComparisonResult rows -- possibly
    differing by method, N, N_lab, or anything else -- into ONE combined
    row: sums rejects/n_reps (equivalent to an unweighted average of each
    row's rate, since every row shares the same n_reps by construction in
    this harness), keeps the first row's descriptive metadata (name/tag/
    eval_type/n/n_lab/etc., which are display fields here, not recomputed
    as an average across whatever heterogeneous scenarios were pooled).
    Used for save_ppi_null_comparison_plot's continuous-eval-type panel,
    which pools ACROSS THE N x N_lab GRID on top of pool_ppi_comparison_
    across_methods' across-methods pooling -- two independent pooling
    axes, scenario and method, both folded into a single number/CI."""
    if not rows:
        return None
    r0 = rows[0]
    return PPIComparisonResult(
        name=r0.name, tag=r0.tag, eval_type=r0.eval_type, n=r0.n, n_reps=sum(r.n_reps for r in rows),
        effect_size=r0.effect_size, label_frac=r0.label_frac, n_lab=r0.n_lab, method=POOLED_METHOD_LABEL,
        rejects_all_human=sum(r.rejects_all_human for r in rows),
        rejects_human_subset=sum(r.rejects_human_subset for r in rows),
        rejects_llm_only=sum(r.rejects_llm_only for r in rows),
        rejects_llm_impute=sum(r.rejects_llm_impute for r in rows),
        rejects_ppi=sum(r.rejects_ppi for r in rows),
        n_failed=sum(r.n_failed for r in rows),
    )


def print_ppi_comparison_report(
    results: list[PPIComparisonResult], alpha: float,
    tags: list[tuple[str, str, str, str]] | None = None, label: str = "paired_t",
) -> None:
    """Five-way rejection-rate table: all_human / human_subset / llm_only /
    llm_impute / ppi, grouped by tag (vs. effect_size, tag="power"; vs.
    label_frac, tag="compare_label_frac") then eval_type.

    `tags` overrides the default (tag, x_field, x_label, x_fmt) list --
    e.g. the binary comparison sweep passes its own "power_binary"/
    "complab_binary" tags here, since binary sources are never tagged
    "power"/"compare_label_frac" (see build_ppi_power_sources_binary/
    build_ppi_comparison_label_frac_sources_binary). `label` names the
    estimand(s) in the header text only (default "paired_t", the original
    single-estimand comparison sweep)."""
    if not results:
        print("\n  (no PPI comparison results)")
        return
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- ESTIMATOR COMPARISON ({label})\n"
          f"  all_human=oracle full-N truth | human_subset=labeled-only truth | llm_only=uncorrected |\n"
          f"  llm_impute=LLM+label-overwrite, no PPI rectifier | ppi=PPI-corrected | alpha={alpha}\n{'='*96}")

    for tag, x_field, x_label, x_fmt in tags or [
        ("power", "effect_size", "es", "{:.2f}"), ("compare_label_frac", "n_lab", "nlab", "{:d}"),
    ]:
        tag_rows = [r for r in results if r.tag == tag]
        if not tag_rows:
            continue
        x_values = sorted({getattr(r, x_field) for r in tag_rows})
        eval_types = sorted({r.eval_type for r in tag_rows})
        print(f"\n  -- vs. {x_field} --")
        for et in eval_types:
            print(f"\n  [{et}]")
            print(f"    {'':<12}" + "".join((x_label + "=" + x_fmt.format(v)).rjust(11) for v in x_values))
            for col, label in [
                ("rejects_all_human", "all_human"), ("rejects_human_subset", "human_subset"),
                ("rejects_llm_only", "llm_only"), ("rejects_llm_impute", "llm_impute"), ("rejects_ppi", "ppi"),
            ]:
                row = f"    {label:<12}"
                for v in x_values:
                    r = next((r for r in tag_rows if r.eval_type == et and getattr(r, x_field) == v), None)
                    rate = getattr(r, col) / r.n_reps if r is not None and r.n_reps > 0 else float("nan")
                    row += f"  {rate:>9.3f}" if np.isfinite(rate) else f"  {'-':>9}"
                print(row)
    print()


def save_results_artifacts_ppi_comparison(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str,
    pooled_results: list[PPIComparisonResult] | None = None,
    tags: list[tuple[str, str, str, str]] | None = None, label: str = "paired_t",
) -> list[str]:
    """`results` is the RAW (per-method, len(sources)*len(methods) rows) data
    -- saved verbatim to the CSV (the per-method breakdown, for reviewers to
    check the pooled average isn't hiding one method behaving badly).

    The saved .log, however, must be built from POOLED data (one row per
    scenario), matching what run()'s own console output already prints via
    print_ppi_comparison_report(comparison_results_pooled, ...) -- pass that
    same pooled list as `pooled_results`. Calling print_ppi_comparison_report
    on the raw rows instead (as an earlier version of this function did) is
    NOT just a cosmetic difference: it doesn't affect the GLM-based factorial
    report's coefficients (grouped-binomial log-likelihood is additive over
    rows sharing the same covariates, so pooled vs. unpooled fits are
    numerically identical there), but THIS function's report picks single
    rows via `next(...)` lookups keyed on eval_type/x_field alone -- fed raw
    data, that silently returns whichever METHOD happens to appear first for
    a given cell instead of the 4-method-averaged rate, discarding the other
    3 methods' data entirely. `pooled_results=None` (the default) falls back
    to pooling `results` internally so old call sites don't silently regress,
    but new callers should pass the already-pooled list run() computes
    anyway, rather than pay to re-derive it here."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "method", "n", "n_reps", "effect_size", "label_frac", "n_lab",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.eval_type, r.method, r.n, r.n_reps, repr(float(r.effect_size)), f"{r.label_frac:.4f}", r.n_lab,
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_comparison_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_comparison_report(pooled_results, alpha=alpha, tags=tags, label=label)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


_PPI_COMPARISON_STYLE = {
    "all_human":    dict(color="#1b9e77", marker="o", ls="-",  label="all human (oracle)"),
    "ppi":          dict(color="#d95f02", marker="o", ls="-",  label="PPI-corrected"),
    "llm_impute":   dict(color="#7570b3", marker="s", ls="--", label="LLM + label overwrite (no PPI)"),
    "llm_only":     dict(color="#e7298a", marker="^", ls="--", label="LLM only (uncorrected)"),
    "human_subset": dict(color="#666666", marker="d", ls=":",  label="human subset only"),
}
_PPI_COMPARISON_COLS = [
    ("all_human", "rejects_all_human"), ("ppi", "rejects_ppi"), ("llm_impute", "rejects_llm_impute"),
    ("llm_only", "rejects_llm_only"), ("human_subset", "rejects_human_subset"),
]


def save_ppi_comparison_plot(
    *, results: list[PPIComparisonResult], alpha: float, out_path: str,
    results_binary: list[PPIComparisonResult] | None = None,
    nlab_pow_results: list[PPIComparisonResult] | None = None,
    nlab_pow_results_binary: list[PPIComparisonResult] | None = None,
    label: str = _COMPARISON_METHODS_LABEL,
) -> str:
    """The flagship 5-way estimator-comparison figure: rejection rate for
    all_human/human_subset/llm_only/llm_impute/ppi, one row per x-axis
    (effect_size, then label_frac), one column per eval type. The story this
    is built to show: human_subset and ppi should share all_human's Type-I
    error at effect_size=0 (all three are unbiased there) while llm_only/
    llm_impute are inflated; as effect_size grows, ppi's power curve should
    track much closer to all_human's than human_subset's flatter, small-N
    curve does -- i.e. PPI recovers most of all_human's power at a fraction
    of its labeling cost, which plain human-only subsetting cannot. The
    n_lab row shows the SAME story from the budget side: ppi's power should
    approach all_human's ceiling as N_lab grows, while human_subset's power
    stays low even at higher N_lab (still small relative to all_human's full
    N). Plotted against the REALIZED N_lab (PPIComparisonResult.n_lab), not
    the nominal label_frac -- see PPI_COMPARISON_LABEL_FRACS' docstring for
    why label_frac alone can be misleading once _JB_MIN_LAB's floor binds.

    results_binary : optional
        Binary's separate comparison sweep (own tags "power_binary"/
        "complab_binary", own 2-method pool -- see run()'s dedicated binary
        comparison block). When given, prepended as the LEFTMOST column --
        binary was previously silently absent from this figure entirely
        (computed and reported in text/CSV, never plotted), which reads to
        a reviewer as binary having been skipped rather than just shown
        elsewhere.

    nlab_pow_results : optional
        build_ppi_nlab_grid_sources' power grid (effect_frac=
        PPI_COMPARISON_MODERATE_EFFECT_FRAC), pre-pooled across methods --
        same data already used by save_ppi_null_comparison_plot's
        nlab_cal_results argument for the null case. When given for an
        eval_type, the n_lab row for that column is pooled (reps-weighted)
        ACROSS EVERY N in the grid at each N_lab, instead of the single
        fixed N=100 comparison_sources' 'compare_label_frac' sweep alone --
        this data was already being computed and printed/saved to CSV every
        run, just never fed into this plot. Falls back to the fixed-N=100
        sweep for any eval_type not covered by the grid (e.g. grades).

    nlab_pow_results_binary : optional
        build_ppi_nlab_grid_sources_binary's power grid, the binary
        analogue of ``nlab_pow_results`` -- pools binary's n_lab row across
        N the same way, instead of the fixed-N=100 results_binary sweep.
        Binary's effect_size row has no grid analogue (the grid only
        covers one fixed effect_frac per call, same limitation as the
        non-binary columns) and stays on the fixed-N=100 sweep.

    label : names which test(s) were pooled into `results`, shown in the
        figure's own suptitle (e.g. run()'s omnibus comparison call passes
        _COMPARISON_METHODS_OMNIBUS_LABEL here). Defaults to
        _COMPARISON_METHODS_LABEL, matching every other pooled-results
        report/save function's own `label` parameter/default in this
        module. Previously hardcoded as "(Paired-Mean Estimand)" -- not
        actually accurate even for the default two-group pool (mixes
        ttest/welch/paired_t's mean-difference estimand with mwu's
        dominance probability and wilcoxon's Walsh-average), so this
        replaces it with the actual pooled method list rather than a
        single estimand name that was never quite right."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No PPI comparison results to plot.")
    eval_types = sorted({r.eval_type for r in results})
    columns = (["binary"] if results_binary else []) + eval_types

    nlab_pow_by_et_nlab: dict[str, dict[int, PPIComparisonResult]] = {}
    if nlab_pow_results:
        for et in {r.eval_type for r in nlab_pow_results}:
            by_nlab = {}
            for nlab in sorted({r.n_lab for r in nlab_pow_results if r.eval_type == et}):
                pooled = _pool_ppi_comparison_rows(
                    [r for r in nlab_pow_results if r.eval_type == et and r.n_lab == nlab]
                )
                if pooled is not None:
                    by_nlab[nlab] = pooled
            if by_nlab:
                nlab_pow_by_et_nlab[et] = by_nlab

    nlab_pow_by_nlab_binary: dict[int, PPIComparisonResult] = {}
    if nlab_pow_results_binary:
        for nlab in sorted({r.n_lab for r in nlab_pow_results_binary}):
            pooled = _pool_ppi_comparison_rows([r for r in nlab_pow_results_binary if r.n_lab == nlab])
            if pooled is not None:
                nlab_pow_by_nlab_binary[nlab] = pooled

    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, len(columns), figsize=(4.8 * len(columns), 4.0 * n_rows), squeeze=False,
    )

    def _plot_row(ax, row_idx: int, col_idx: int, x_values: list, et_rows: dict, xlabel: str, fixed: str) -> None:
        ax.axhline(
            alpha, color="black", ls="--", lw=1.0, alpha=0.5,
            label=f"Nominal {_alpha_label(alpha)}" if row_idx == 0 and col_idx == 0 else None,
        )
        for key, rejects_field in _PPI_COMPARISON_COLS:
            style = _PPI_COMPARISON_STYLE[key]
            ys = [
                (getattr(et_rows[x], rejects_field) / et_rows[x].n_reps) if x in et_rows and et_rows[x].n_reps else float("nan")
                for x in x_values
            ]
            ax.plot(
                x_values, ys, color=style["color"], marker=style["marker"], linestyle=style["ls"],
                linewidth=1.8, markersize=5, label=style["label"] if row_idx == 0 and col_idx == 0 else None,
            )
        ax.set_ylim(-0.02, 1.02)
        if col_idx == 0:
            ax.set_ylabel("Rejection rate")
        ax.set_xlabel(f"{xlabel}\n({fixed})", fontsize=9)

    for col_idx, col in enumerate(columns):
        is_binary = col == "binary"
        et = col
        source = results_binary if is_binary else results
        methods_note = " (2 tests pooled)" if is_binary else ""

        # Row 0: vs effect_size -- always the fixed-N comparison_sources
        # sweep (no broader-N grid exists for this axis; a full effect_size
        # x N x N_lab grid would need a genuinely new sweep, not just a
        # different slice of already-collected data).
        ax0 = axes[0][col_idx]
        tag0 = "power_binary" if is_binary else "power"
        tag0_rows = [r for r in source if r.tag == tag0 and r.eval_type == et]
        x0_values = sorted({r.effect_size for r in tag0_rows})
        et0_rows = {r.effect_size: r for r in tag0_rows}
        _plot_row(ax0, 0, col_idx, x0_values, et0_rows, "Effect size", f"label budget fixed at N_lab/N = 20%{methods_note}")
        ax0.set_title("Binary" if is_binary else et.capitalize())

        # Row 1: vs n_lab -- pooled across the FULL N x N_lab grid when
        # available (nlab_pow_by_et_nlab), instead of the fixed-N=100 sweep.
        ax1 = axes[1][col_idx]
        if is_binary and nlab_pow_by_nlab_binary:
            et1_rows = nlab_pow_by_nlab_binary
            x1_values = sorted(et1_rows.keys())
            fixed1 = f"effect size fixed at {PPI_COMPARISON_MODERATE_EFFECT_FRAC:.0%}; pooled over N=30-400{methods_note}"
        elif is_binary:
            tag1_rows = [r for r in source if r.tag == "complab_binary" and r.eval_type == et]
            x1_values = sorted({r.n_lab for r in tag1_rows})
            et1_rows = {r.n_lab: r for r in tag1_rows}
            fixed1 = f"effect size fixed at {PPI_COMPARISON_MODERATE_EFFECT_FRAC:.0%}; N=100{methods_note}"
        elif et in nlab_pow_by_et_nlab:
            et1_rows = nlab_pow_by_et_nlab[et]
            x1_values = sorted(et1_rows.keys())
            fixed1 = f"effect size fixed at {PPI_COMPARISON_MODERATE_EFFECT_FRAC:.0%}; pooled over N=30-400"
        else:
            tag1_rows = [r for r in results if r.tag == "compare_label_frac" and r.eval_type == et]
            x1_values = sorted({r.n_lab for r in tag1_rows})
            et1_rows = {r.n_lab: r for r in tag1_rows}
            fixed1 = f"effect size fixed at {PPI_COMPARISON_MODERATE_EFFECT_FRAC:.0%}; N=100"
        _plot_row(ax1, 1, col_idx, x1_values, et1_rows, "N_lab (labeled items)", fixed1)

    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, borderaxespad=0.5)
    fig.suptitle(f"PPI-Corrected Estimator Comparison ({label})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        # rect right stays at 1 (not narrowed to make room for the legend) --
        # bbox_to_anchor=(1.0, ...) already puts the legend flush against the
        # rightmost subplot; savefig's bbox_inches="tight" grows the canvas
        # to include it. Narrowing rect's right edge below 1 here reserves
        # blank figure space between the subplots and the legend instead.
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_PPI_NULL_COMPARISON_ORDER = ["all_human", "human_subset", "ppi", "llm_impute", "llm_only"]
"""Deliberately NOT alphabetical or the same left-to-right order as
save_ppi_comparison_plot's legend: groups the three arms that SHOULD be
well-calibrated at the null (all_human, human_subset, ppi) together on the
left, then the two that are biased by construction (llm_impute, llm_only)
on the right -- the grouping itself is part of what makes the bar chart
read at a glance."""


def save_ppi_null_comparison_plot(
    *, results: list[PPIComparisonResult], alpha: float, out_path: str,
    nlab_cal_results: list[PPIComparisonResult] | None = None,
    results_binary: list[PPIComparisonResult] | None = None,
    nlab_cal_results_binary: list[PPIComparisonResult] | None = None,
) -> str:
    """Bar chart isolating JUST the null (no real effect) case from
    save_ppi_comparison_plot's line plot -- one bar per estimator arm, one
    panel per eval type. save_ppi_comparison_plot's effect_size=0 point
    carries the same numbers, but buried as one of several x-values on a
    line plot built to tell a POWER story -- easy to misread llm_only/
    llm_impute's high rejection rate at small real effect sizes as "more
    powerful than PPI" there, when it's actually inflated false positives
    from judge bias, not power (build_ppi_power_sources fixes bias
    direction to OPPOSE the injected effect; the observed uncorrected
    difference is `effect_size - bias_delta`, so llm_only/llm_impute
    already reject at ~100% at effect_size=0, before any real effect exists
    at all -- see save_ppi_power_direction_plot for the reinforcing-bias
    mirror image, where the same arms would instead overstate a real effect
    that IS present). This plot has no effect_size axis to be misread
    against: every bar here is, by construction, a false-positive rate.

    Every bar pools across _COMPARISON_METHODS (ttest_welch/paired_t/mwu/
    wilcoxon -- `results`/`nlab_cal_results` are expected to already be
    pool_ppi_comparison_across_methods output, one row per scenario, not
    the raw per-method rows). For continuous and likert specifically,
    passing `nlab_cal_results` (build_ppi_nlab_grid_sources' calibration
    grid, ALSO pre-pooled across methods, now itself crossing continuous/
    likert) pools a SECOND axis on top, per eval type: every N x N_lab cell
    in that eval type's slice of the grid, not just the single (N=100,
    N_lab=20) baseline scenario `results` alone would give -- a more robust
    version of this chart, averaging over 4 tests x ~22 (N, N_lab)
    conditions rather than one arbitrarily-chosen scenario.
    grades has no such sweep available (build_ppi_nlab_grid_sources
    deliberately excludes it as redundant with continuous), so it falls
    back to `results`' single scenario -- still pooled across the 4
    methods, just not across N/N_lab. Each panel's subtitle states which
    pooling applies.

    results_binary : optional
        Binary's separate null-effect comparison sweep (own "power_binary"
        tag, own 2-method pool -- ttest_welch/paired_t only, no mwu/
        wilcoxon). When given, prepended as the LEFTMOST panel -- binary was
        previously computed and reported in text/CSV but never plotted
        here.

    nlab_cal_results_binary : optional
        build_ppi_nlab_grid_sources_binary's calibration grid (effect_frac=
        0.0), pre-pooled across binary's 2 methods -- the binary analogue
        of ``nlab_cal_results``. When given, binary's panel is ALSO pooled
        across its full N x N_lab grid instead of the single (N=100,
        N_lab=20) scenario, matching continuous/likert's treatment. Falls
        back to the single-scenario pooling (same caveat as the grades
        fallback above) if omitted.

    Error bars are the 95% Wilson score interval for each bar's underlying
    binomial proportion (_ppi_wilson_interval, the same interval
    print_ppi_report's Type-I flagging already uses), computed on the
    POOLED rejects/n_reps -- i.e. treating every pooled replicate as one
    more independent Bernoulli draw at the same rate. That's exact for
    pooling across methods/conditions that are truly identically
    calibrated, and a standard, if slightly optimistic, simplification
    if there's real heterogeneity across the pooled methods/(N, N_lab)
    cells (the same simplification this file's pooled Type-I metrics
    already use, e.g. key_metrics["ppi_mean_corrected_type1"]) -- called
    out here rather than presented as more rigorous than it is."""
    import matplotlib.pyplot as plt

    null_rows = [r for r in results if r.tag == "power" and abs(r.effect_size) < 1e-9]
    if not null_rows:
        raise ValueError("No null-effect (effect_size=0) comparison results to plot.")
    null_rows_binary = [r for r in (results_binary or []) if r.tag == "power_binary" and abs(r.effect_size) < 1e-9]
    eval_types = sorted({r.eval_type for r in null_rows})
    columns = (["binary"] if null_rows_binary else []) + eval_types
    nlab_null_pool_by_et: dict[str, PPIComparisonResult | None] = {}
    if nlab_cal_results:
        for et in {r.eval_type for r in nlab_cal_results}:
            nlab_null_pool_by_et[et] = _pool_ppi_comparison_rows(
                [r for r in nlab_cal_results if r.tag == "nlab_grid" and r.eval_type == et]
            )
    nlab_null_pool_binary = (
        _pool_ppi_comparison_rows([r for r in nlab_cal_results_binary if r.tag == "nlab_grid_binary"])
        if nlab_cal_results_binary else None
    )

    fig, axes = plt.subplots(1, len(columns), figsize=(3.4 * len(columns), 4.9), squeeze=False)
    x = np.arange(len(_PPI_NULL_COMPARISON_ORDER))
    for col, et in enumerate(columns):
        ax = axes[0][col]
        if et == "binary" and nlab_null_pool_binary is not None:
            r = nlab_null_pool_binary
            subtitle = f"pooled: 2 tests x N x N_lab\n(n_reps={r.n_reps})"
        elif et == "binary":
            r = next(r for r in null_rows_binary if r.eval_type == "binary")
            subtitle = f"pooled: 2 tests\n(N={r.n}, N_lab={r.n_lab}, n_reps={r.n_reps})"
        elif nlab_null_pool_by_et.get(et) is not None:
            r = nlab_null_pool_by_et[et]
            subtitle = f"pooled: 4 tests x N x N_lab\n(n_reps={r.n_reps})"
        else:
            r = next(r for r in null_rows if r.eval_type == et)
            subtitle = f"pooled: 4 tests\n(N={r.n}, N_lab={r.n_lab}, n_reps={r.n_reps})"
        rejects = [getattr(r, f"rejects_{key}") for key in _PPI_NULL_COMPARISON_ORDER]
        rates = [k / r.n_reps if r.n_reps else float("nan") for k in rejects]
        ci_lo_hi = [_ppi_wilson_interval(k, r.n_reps) for k in rejects]
        yerr = [
            [max(0.0, rate - lo) for rate, (lo, _hi) in zip(rates, ci_lo_hi)],
            [max(0.0, hi - rate) for rate, (_lo, hi) in zip(rates, ci_lo_hi)],
        ]
        colors = [_PPI_COMPARISON_STYLE[key]["color"] for key in _PPI_NULL_COMPARISON_ORDER]
        ax.bar(x, rates, color=colors, width=0.65, zorder=2)
        ax.errorbar(
            x, rates, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4,
            label="95% Wilson CI" if col == 0 else None,
        )
        ax.axhline(
            alpha, color="black", ls="--", lw=1.2, zorder=3,
            label=f"Nominal {_alpha_label(alpha)}" if col == 0 else None,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_PPI_COMPARISON_STYLE[key]["label"] for key in _PPI_NULL_COMPARISON_ORDER],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0.0, 1.08)
        ax.set_title(f"{et.capitalize()}\n{subtitle}", fontsize=10)
        ax.set_ylabel("False positive rate" if col == 0 else "")
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=1)
        for xi, rate in zip(x, rates):
            if np.isfinite(rate):
                ax.text(xi, rate + 0.02, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, borderaxespad=0.5)
    fig.suptitle("False-Positive Rate Under the Null (No Real Effect)", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path



# ---------------------------------------------------------------------------
# Label-efficiency ("effective sample size") check: for a fixed labeling
# budget, how many labels would a HUMAN-ONLY test need to match PPI's power?
# Reported as an "equivalent N_lab" curve against a y=x "no benefit from the
# judge" reference -- the label-efficiency multiplier is the vertical (or
# horizontal) gap between the two, directly in units a reviewer cares about
# ("this many labels saved"), rather than a CI-width or rejection-rate
# comparison they'd have to translate themselves. Crossed with judge quality
# (build_ppi_label_efficiency_sources' llm_noise tiers) so the SAME figure
# also answers "does this benefit survive a worse judge."
# ---------------------------------------------------------------------------


@dataclass
class LabelEfficiencyPoint:
    """One label-efficiency measurement: a judge-quality tier x effect-size
    x eval-type cell, with the resulting PPI/classical equivalence."""

    eval_type: str
    judge_noise: float
    """The calibrated llm_noise value actually simulated -- kept for
    traceability/debugging, but NOT what the plot labels lines by (see
    alignment_value): the same noise means very different judge quality
    across eval types (continuous's Pearson r and likert's weighted kappa
    respond to noise very differently), so noise alone isn't a fair axis to
    compare eval-type panels against each other on."""
    alignment_metric: str
    """Which alignment metric this eval type is calibrated/labeled by --
    "pearson_r" (continuous), "weighted_kappa" (likert), "kappa" (binary) --
    see _LABEL_EFF_ALIGNMENT_METRIC and measure_judge_alignment."""
    alignment_target: float
    """The target alignment value _calibrate_noise_for_alignment's bisection
    aimed `judge_noise` at (e.g. 0.8/0.5/0.2) -- a round, reviewer-legible
    number chosen independent of eval type, unlike judge_noise itself."""
    alignment_value: float
    """The ACTUALLY achieved alignment metric at `judge_noise`, from a
    separate large-sample (measure_judge_alignment) measurement -- won't
    exactly equal alignment_target (MC noise in that measurement, plus
    bisection tolerance), so plots/tables should label by this, not the
    target, to avoid implying more precision than the calibration has."""
    n_lab: int
    """Realized N_lab (see PPIComparisonResult.n_lab's docstring) PPI actually
    used to achieve `ppi_power`."""
    ppi_power: float
    equiv_n_lab: float
    """The N_lab a human-only classical test (pooled across the SAME method
    family PPI's `ppi_power` was pooled across -- _COMPARISON_METHODS or
    _COMPARISON_METHODS_BINARY) would need to reach the SAME power, per
    _classical_pooled_power_curve/_equivalent_n_lab. Meaningless when
    `saturated` is True -- see that field's docstring; callers must check it
    rather than plotting/averaging equiv_n_lab unconditionally."""
    n_reps: int
    saturated: bool = False
    """True when `ppi_power` is at or above the classical reference curve's
    own ceiling (power_grid.max()) -- inverting a power at or past a flat
    curve's plateau is ill-posed (np.interp clamps to n_grid's upper edge
    instead of raising). Saturated points are still shown (as a lower-bound
    marker, not a real equivalent-N) but excluded from axis-limit
    computation."""
    effect_frac: float = PPI_LABEL_EFF_EFFECT_FRAC
    """Effect-size fraction this point was simulated at (see
    PPIComparisonResult.effect_size for the convention) -- the arm of
    PPI_LABEL_EFF_EFFECT_FRACS it came from. Defaulted to
    PPI_LABEL_EFF_EFFECT_FRAC since run_ppi_label_efficiency_check holds it
    fixed; run_ppi_nformula_check varies it instead, sweeping
    PPI_NFORMULA_EFFECT_FRACS. Kept as a CSV column so per-es curves can be
    checked separately from the pooled one, since the multiplier should be
    es-invariant (a property of judge quality, not effect size)."""
    mult_lo: float = float("nan")
    mult_hi: float = float("nan")
    """95% interval on the multiplier (equiv_n_lab / n_lab), from
    propagating ppi_power's binomial SE through the reference curve's local
    slope -- see _multiplier_ci. Reporting the multiplier without this
    overstates its precision: at effect_frac=0.15/n_lab=15 the interval can
    span [1.0, 5.3]. Covers binomial noise in ppi_power only; the reference
    curve's own MC error is addressed by smoothing
    (_smooth_monotone_power_curve) rather than by this interval."""
    rho2: float = float("nan")
    """Squared within-group Pearson correlation between judge score and human
    label, for the judge at this (eval_type, judge_noise) tier -- the SAME
    quantity for all three eval types, which is what lets one threshold cover
    them (see scenarios/synthetic._alignment_metric_dict's "rho2"). Recorded
    per point rather than only in the calibration csv so the measured
    `multiplier` and the theory it should follow sit on the same row."""
    predicted_mult: float = float("nan")
    """Control-variate prediction 1/(1 - rho2*(1 - n_lab/N)) from
    _ppi_predicted_savings -- the exact finite-pool form, not the asymptotic
    1/(1-rho2) (which overstates for a strong judge; see that function).
    Predicts the variance-scale saving, whereas `multiplier` is obtained by
    inverting a power curve and so saturates for strong judges -- expect
    predicted_mult >= multiplier at the top tiers rather than exact
    agreement."""
    predicted_mult_asymptotic: float = float("nan")
    """1/(1 - rho2), the large-unlabeled-pool limit. Carried alongside the
    exact form so a reader can see how far apart they are at this design
    point; not the headline number."""
    inversion_ratio: float = float("nan")
    """What this cell's human-subset arm inverts to, divided by its own n_lab.

    The human-subset arm is a classical test on exactly n_lab labeled items,
    so a faithful inversion returns n_lab and this is 1.00. It involves no
    judge scores at all, which is what makes it usable as a filter: it
    measures the reference curve's local conditioning, not the quantity being
    estimated. This is a gate (`well_conditioned`), not a divisor: dividing
    the multiplier by it removes no bias and injects its own spread into
    every number."""
    inversion_clamped: bool = False
    """Whether inversion_ratio came from a clamped inversion and so carries no
    information about conditioning.

    _equivalent_n_lab inverts with np.interp, which clamps to n_grid's
    endpoints instead of extrapolating. The human-subset arm at the smallest
    n_lab has power near alpha, at or below the reference curve's left edge,
    so its inversion pins to n_grid.min() -- returning a ratio of exactly
    1.000 regardless of how ill-conditioned the cell actually is. A clamped
    cell is therefore treated as unconditioned rather than trusted."""
    variance_multiplier: float = float("nan")
    """Label-efficiency multiplier measured as Var(classical)/Var(PPI) across
    replicates, with no power curve involved.

    The control-variate factor is a variance ratio by definition, so this
    measures it directly instead of inverting a power curve to recover it. It
    has no flat-curve regime, no clamping, and no conditioning gate.

    Use it to check `equiv_n_lab / n_lab`, not to replace it: the inverted
    multiplier is in the unit a practitioner acts on ("this many labels"),
    while this is the quantity the theory actually bounds.

    NaN on pooled-across-method rows: per-method estimands are on different
    scales (a mean difference and a Walsh theta are not commensurable), so
    their variances cannot be averaged -- only their ratios can."""
    noise_family: str = "gaussian"
    """Judge-error shape this cell was simulated under -- "gaussian" or
    "contaminated" (scenarios.synthetic.PPI_LABEL_EFF_NOISE_FAMILIES).

    Crossed with the judge-quality axis (alignment_value), not nested inside
    it: total judge-error variance is held identical across families, so a
    given alignment tier means the same thing in both and the two arms are
    directly comparable at matched rho^2. Exists because rank tests are
    sensitive to error shape and mean tests are not -- see
    notes/RANK_VS_PARAMETRIC_CROSSOVER.md."""

    @property
    def well_conditioned(self) -> bool:
        """Whether this cell's power-curve inversion is trustworthy enough to
        report, i.e. |inversion_ratio - 1| <= _INVERSION_DEV_TOL.

        Ill-conditioned cells are those where the reference power curve is
        flat (small effect size, small n_lab), so dn/dP is large and the
        binomial noise in a rejection rate maps to a huge swing in equivalent
        n. Filtering on them is the direct analogue of `saturated`, which
        excludes the opposite end of the same curve.

        Callers should require `not saturated and well_conditioned` before
        using `equiv_n_lab`. NaN (never measured) counts as conditioned so
        older results stay usable; a clamped inversion does NOT (see
        `inversion_clamped`)."""
        if self.inversion_clamped:
            return False
        return not np.isfinite(self.inversion_ratio) or abs(self.inversion_ratio - 1.0) <= _INVERSION_DEV_TOL

    n: int = PPI_LABEL_EFF_N
    """Total item count this point was simulated at. Defaulted to
    PPI_LABEL_EFF_N for backward compatibility with run_ppi_label_
    efficiency_check (which holds N fixed and never sets this explicitly)
    -- run_ppi_nformula_check is the only caller that varies it, sweeping
    PPI_NFORMULA_N_VALUES to test whether the label-efficiency multiplier
    needs an explicit N term (see that function's docstring)."""


_POWER_CURVE_CACHE_VERSION = 1
"""Bump this whenever anything that changes a reference curve's VALUES changes
but is not part of the cache key -- i.e. the data-generation path
(generate_judge_bias_cell, sample_group_truth, the eval type's shape/anchor
constants) or _classical_pooled_power_curve's own body. The key covers the
arguments; it cannot see module-level behaviour, so this constant is the
manual half of the invariant. Getting it wrong serves a stale curve silently,
which is exactly the class of bug the inversion self-consistency check exists
to catch -- but do not rely on that check to notice; bump the version."""

_POWER_CURVE_CACHE_DIR = pathlib.Path("simulations/out/.power_curve_cache")
"""Where cached reference curves live. Under simulations/out/, which is
gitignored, so cached curves never enter the repo. Delete the directory to
invalidate everything, or set PPI_NO_POWER_CURVE_CACHE=1 to bypass."""


def _classical_pooled_power_curve(
    eval_type: str, es: float, methods: tuple, n_values: np.ndarray, n_mc: int, seed: int,
) -> np.ndarray:
    """Disk-cached wrapper. The curve is a pure function of its arguments (a
    seeded Monte Carlo over ground truth only), so it is safe to memoize
    across runs -- and worth it: at ref_n_mc=10000 the twelve curves a
    label-efficiency sweep needs cost hours, and every later run, including
    the official tests, rebuilds exactly the same ones.

    Writes are atomic (temp file + rename), so parallel workers racing on the
    same key cannot serve a half-written array. A corrupt or unreadable entry
    is treated as a miss and recomputed rather than raising."""
    key_src = repr((
        _POWER_CURVE_CACHE_VERSION, eval_type, f"{float(es):.12g}", tuple(methods),
        np.asarray(n_values, dtype=float).tobytes(), int(n_mc), int(seed),
    ))
    key = hashlib.sha256(key_src.encode()).hexdigest()[:20]
    path = _POWER_CURVE_CACHE_DIR / f"{eval_type}_nmc{n_mc}_{key}.npy"
    use_cache = os.environ.get("PPI_NO_POWER_CURVE_CACHE", "") != "1"
    if use_cache and path.exists():
        try:
            cached = np.load(path)
            if cached.shape == np.asarray(n_values).shape:
                return cached
        except Exception:
            pass  # unreadable/corrupt -> recompute
    curve = _classical_pooled_power_curve_uncached(eval_type, es, methods, n_values, n_mc, seed)
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, curve)
            os.replace(tmp, path)
        except Exception:
            pass  # caching is an optimization; never fail the sweep over it
    return curve


def _classical_pooled_power_curve_uncached(
    eval_type: str, es: float, methods: tuple, n_values: np.ndarray, n_mc: int, seed: int,
) -> np.ndarray:
    """Pooled (mean-across-`methods`) classical-test power at effect size
    `es`, evaluated at every sample size in `n_values` -- the "how many
    labels alone would you need" reference curve LabelEfficiencyPoint's
    equiv_n_lab is read off of.

    Draws ONLY ground truth (generate_judge_bias_cell's truth_* fields),
    never LLM-judge scores: this reference is judge-quality-independent by
    construction, matching _run_ppi_comparison_cell's human_subset arm (a
    classical test on the labeled subset's TRUE ground truth) -- under MCAR
    labeling (this project's documented PPI scope, see MWU/KRUSKAL's Method
    docstrings in methods.py), a random n_lab-sized labeled subset of an iid
    truth pool is distributionally identical to an independent n_lab-sized
    draw, so a throwaway JudgeBiasSource built directly at n=n_lab (rather
    than reusing one of the label_frac sweep's own n=100 replicates and
    subsetting it) gives the exact same reference at ANY n, not just the
    handful the label_frac sweep happens to simulate -- avoiding any
    extrapolation risk in _equivalent_n_lab's inversion.

    One cell per rep feeds every method in `methods` (matching
    _run_ppi_comparison_cell's own "one draw, every arm" pattern) rather than
    redrawing per method. np.maximum.accumulate enforces monotonicity in n
    against MC noise -- power is monotonically non-decreasing in sample size
    by construction; without this, np.interp's inversion in _equivalent_n_lab
    could pick a slightly-too-small n at a local noise dip."""
    rng = np.random.default_rng(seed)
    powers = np.zeros(len(n_values))
    for i, n in enumerate(n_values):
        # generate_judge_bias_cell unconditionally requires n >= _JB_MIN_LAB
        # (it always draws a labeled subset internally, even though only
        # truth_* is read here) -- floor to that rather than n_grid's own
        # minimum, so a caller passing a smaller n_grid value still gets a
        # valid (if slightly right-shifted) reference point instead of a
        # crash.
        n_int = max(_JB_MIN_LAB, int(round(n)))
        sc = JudgeBiasSource(name="_labeleff_ref", tag="_ref", eval_type=eval_type, n=n_int, effect_size=es)
        rejects = {m: 0 for m in methods}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_mc):
                cell = generate_judge_bias_cell(sc, rng)
                for method in methods:
                    structure = _COMPARISON_METHOD_STRUCTURE[method]
                    a, b = (cell.truth_a2, cell.truth_b2) if structure == "group" else (cell.truth_x, cell.truth_y)
                    try:
                        p = _classical_pvalue(a, b, method, structure)
                        rejects[method] += int(p < _ALPHA)
                    except Exception:
                        pass
        powers[i] = float(np.mean([rejects[m] / n_mc for m in methods]))
    return np.maximum.accumulate(powers)


def _smooth_monotone_power_curve(n_grid: np.ndarray, power_grid: np.ndarray) -> np.ndarray:
    """Monotonize the classical reference curve with ISOTONIC REGRESSION, then
    break exact ties, so _equivalent_n_lab inverts it without bias.

    Two distinct defects have to be handled, and conflating them is how the
    previous version went wrong:

    1. NON-MONOTONE MC WIGGLE. Raw Monte Carlo power is not monotone in N, and
       np.interp would invert the wiggle as if it were signal. Isotonic
       regression (sklearn's PAVA) is the minimal fix: it returns the closest
       non-decreasing curve to the data and imposes NO shape of its own.

    2. EXACT TIES. At low ref_n_mc the raw curve ties across adjacent grid
       points (measured at effect_frac=0.15: power 0.070 at N=17.1, 19.4 AND
       22.1). np.interp resolves a tied plateau to its LEFT edge, biasing
       equiv_n_lab -- and so the multiplier -- downward exactly in the
       small-n_lab cells where the curve is flattest. Isotonic regression
       preserves ties (a tied block is already non-decreasing), so it does not
       address this on its own; a negligible strictly-increasing ramp is added
       afterwards so ties resolve through the middle of the plateau instead.

    THIS REPLACES A LOGISTIC-IN-LOG-N FIT, WHICH WAS BADLY BIASED. That fit
    imposed a parametric shape the real curve does not have: measured against
    a continuous reference curve at ref_n_mc=4000, it put power at N=15 at
    0.037 where the data said 0.087, and at N=208 at 0.727 where the data said
    0.595 -- far too steep. The consequence was a self-inconsistent inversion:
    the human-subset arm, which IS a classical test on exactly n_lab labeled
    items, inverted to 1.8x n_lab at n_lab=15 and 0.69x at n_lab=200, a 2.6x
    drift across the grid that multiplied straight into every reported
    multiplier and reversed its apparent trend in n_lab. Isotonic regression
    holds the same check to within +/-7% (see
    _check_inversion_self_consistency, which enforces it on every run).

    The reference curve's MC error is SHARED by every cell of an eval type (it
    is built once), so unlike ppi_power's binomial noise it is a systematic
    offset that raising --effect-reps cannot reduce -- monotonizing is still
    the right lever, just not a parametric one."""
    from sklearn.isotonic import IsotonicRegression

    p = np.asarray(power_grid, dtype=float)
    x = np.asarray(n_grid, dtype=float)
    if len(x) < 3 or not np.isfinite(p).all() or float(np.ptp(p)) < 1e-9:
        return np.maximum.accumulate(p)
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(np.log(x), p)
    # Break exact ties with a ramp far below MC resolution, so a tied plateau
    # inverts through its middle rather than collapsing to its left edge.
    # Headroom is reserved BEFORE adding the ramp: clipping to 1.0 afterwards
    # would flatten the ramp back into ties wherever the curve saturates.
    fitted = np.clip(fitted, 0.0, 1.0 - 1e-6)
    return fitted + np.arange(len(fitted), dtype=float) * 1e-9


_INVERSION_DEV_TOL = 0.25
"""How far a cell's human-subset arm may invert from its own n_lab and still
be reported (see LabelEfficiencyPoint.inversion_ratio/well_conditioned).

The inversion is unbiased (median exactly 1.000 across eval types), so this
tolerance only needs to remove variance, not bias. 0.25 roughly doubles
retention versus a tighter 0.15 gate at negligible attainment cost; past
0.30, paired_t's measured/predicted ratio drifts above 1.000, which is
impossible for a control variate, so that is the practical upper bound. See
notes/HOW_MULTIPLIERS_ARE_MEASURED.md for the full calibration sweep."""


def _check_inversion_self_consistency(
    n_lab_values, human_powers, n_grid: np.ndarray, power_grid: np.ndarray,
    label: str, tol: float = 0.15,
) -> float:
    """Assert the power-curve inversion is self-consistent, and warn if not.

    The human-subset arm IS a classical test on exactly n_lab labeled items, so
    feeding ITS rejection rate back through the same reference curve must
    return n_lab. Any systematic departure is a bias in the inversion itself,
    and it multiplies directly into every multiplier this check's caller goes
    on to report -- so it is checked on every run rather than trusted.

    This is free (the data is already in hand) and it is exactly the test that
    caught the logistic smoother: it returned ratios from 1.8 down to 0.69
    across the n_lab grid where a correct inversion returns ~1.0.

    Returns the worst |ratio - 1| seen. Warns rather than raises: a sweep that
    has already spent hours simulating should surface the problem, not discard
    the data."""
    ratios = []
    for n_lab, hp in zip(n_lab_values, human_powers):
        if not (n_lab and np.isfinite(hp)):
            continue
        inv = _equivalent_n_lab(hp, n_grid, power_grid)
        if np.isfinite(inv) and inv > 0:
            ratios.append(inv / float(n_lab))
    if not ratios:
        return float("nan")
    worst = float(np.max(np.abs(np.array(ratios) - 1.0)))
    if worst > tol:
        lo, hi = float(np.min(ratios)), float(np.max(ratios))
        print(f"  !! INVERSION NOT SELF-CONSISTENT [{label}]: the human-subset arm "
              f"inverts to {lo:.2f}x-{hi:.2f}x its own n_lab (want ~1.00). "
              f"Multipliers from this eval type are biased by roughly that factor "
              f"-- see _smooth_monotone_power_curve.")
    return worst


def _ppi_predicted_savings(rho2: float, n_lab: int, n_total: int) -> float:
    """Control-variate prediction of PPI's labeling-effort saving:

        saving = 1 / (1 - rho^2 * (1 - n_lab/n_total))

    i.e. how many times more human labels a human-only analysis would need to
    match this PPI analysis. `rho2` is the squared Pearson correlation between
    judge score and human label WITHIN a group (see scenarios/synthetic.
    _alignment_metric_dict's "rho2"), which is the same quantity for binary,
    likert and continuous data -- the property that lets one threshold serve
    all three.

    Derivation (labeled set of size n NESTED in n_total items, judge score f
    observed on all of them, so the labeled and full-sample means are
    correlated):

        theta_hat = lam*fbar_all + (Ybar_lab - lam*fbar_lab)
                  = Ybar_lab - lam*(1 - n/N)*(fbar_lab - fbar_unlab)
        Var       = (sigma_Y^2/n) * (1 - rho^2*(N-n)/N)      at lam* = rho*sY/sf

    against a human-only Var of sigma_Y^2/n, giving the ratio above.

    THE UNLABELED-FRACTION TERM IS NOT OPTIONAL. The asymptotic 1/(1-rho^2) is
    a reasonable approximation only for a mediocre judge; because the
    denominator is 1 - rho^2*k, sensitivity to k GROWS as the judge improves,
    which is the opposite of the usual intuition. Measured against empirical
    variance ratios: at rho^2=0.5 the asymptote is 2.00 vs 1.97 exact (fine),
    but at rho^2=0.90 it claims 10x against 8.4x, and at rho^2=0.99 it claims
    100x against a measured 40x. Report the exact form; use the asymptote only
    as intuition. An earlier N/(N+n_lab) variant of this correction, fitted at
    a single design point, is systematically too generous (43% high at
    n_lab/N=0.4) and should not be reused.

    Small n_lab is the FAVOURABLE end, worth stating for readers who will
    reach for the smallest labeled set the tool allows: at n_total=1000,
    n_lab=15 measured 39.7x where n_lab=400 measured 2.4x, since a small
    labeled set leverages a large unlabeled pool.

    Validated over a 48-cell (3 eval types x 4 noise x 4 bias) grid at 3000
    replicates per cell: R^2=0.9968 vs measured Var(human-subset)/Var(PPI),
    mean error -0.15%, max 5.5%, under ADAPTIVE (power-tuned) lambda."""
    if not np.isfinite(rho2) or n_total <= 0:
        return float("nan")
    k = max(0.0, 1.0 - float(n_lab) / float(n_total))
    denom = 1.0 - float(np.clip(rho2, 0.0, 1.0)) * k
    return float(1.0 / denom) if denom > 1e-9 else float("inf")


def _multiplier_ci(
    ppi_power: float, n_reps: int, n_lab: int, n_grid: np.ndarray, power_grid: np.ndarray,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """95% interval on equiv_n_lab / n_lab, by pushing ppi_power's binomial
    Wald interval through the same inversion the point estimate uses.

    The interval is wide because the inversion's gain is: dN/dP is 800-1250
    labels per unit power in the flat part of the curve, so an SE of 0.02 on
    ppi_power moves equiv_n_lab by +/-16-25 labels. That is real uncertainty,
    not a defect of this function -- it is why the effect-size sweep
    (PPI_LABEL_EFF_EFFECT_FRACS) matters more than raising reps: moving a
    cell into the curve's steep middle shrinks dN/dP, whereas reps only
    shrink SE as 1/sqrt(n)."""
    if n_lab <= 0 or n_reps <= 0 or not np.isfinite(ppi_power):
        return float("nan"), float("nan")
    se = math.sqrt(max(ppi_power * (1.0 - ppi_power), 0.0) / n_reps)
    lo_p = max(ppi_power - z * se, 0.0)
    hi_p = min(ppi_power + z * se, 1.0)
    return (_equivalent_n_lab(lo_p, n_grid, power_grid) / n_lab,
            _equivalent_n_lab(hi_p, n_grid, power_grid) / n_lab)


def _equivalent_n_lab(target_power: float, n_grid: np.ndarray, power_grid: np.ndarray) -> float:
    """Invert the classical reference curve (n_grid, power_grid; power_grid
    assumed non-decreasing, see _classical_pooled_power_curve) to find the
    sample size a human-only test would need to reach `target_power`.
    np.interp CLAMPS to n_grid's own endpoints rather than extrapolating --
    if `target_power` falls above power_grid.max() (PPI more powerful than
    ANY n this reference curve was evaluated at), the result silently
    saturates at n_grid.max() instead of reporting the true, larger
    equivalent count. Callers should size n_grid generously enough that
    this doesn't bind for the power levels actually observed -- flagged via
    LabelEfficiencyPoint.equiv_n_lab's own docstring rather than raising,
    since a saturated point is still informative (a lower bound on the
    label-efficiency gain) if plotted as-is."""
    return float(np.interp(target_power, power_grid, n_grid))


_LABEL_EFF_ALIGNMENT_TARGETS = (0.70, 0.60, 0.50, 0.40, 0.30, 0.20)
_LABEL_EFF_FIGURE_TITLES = os.environ.get("PPI_NO_FIGURE_TITLES", "") != "1"
#: Set at import from PPI_NO_FIGURE_TITLES, and overridable per run via
#: --no-figure-titles (applied in run()). The env var alone cannot serve
#: the harness: it is read once at import, so a preset that wants
#: publication figures has no way to ask for them -- which left the
#: figures a run emits subtly taller than the ones the paper prints.
"""Whether label-efficiency figures draw their own headline title.

Set PPI_NO_FIGURE_TITLES=1 for publication figures. Journal and conference
figures carry their content in the caption; an in-figure title duplicates it
and costs vertical space, which matters most for the multi-panel ones.

Also suppresses the in-figure footnote strip (the fig.text() line under each
axes explaining what the bands and points are). That is a subcaption, and a
figure with both a subcaption and a LaTeX caption makes the reader check two
places for one explanation -- so the flag moves that content into the caption
too. Anything suppressed here must be restated in the figure's LaTeX caption.

Panel labels (Binary/Continuous/Likert, the four design names in the lookup
grid) are NOT titles in this sense and are always drawn -- they identify axes
rather than restating the caption."""


_LABEL_EFF_PAYOFF_FLOOR = 0.40
"""Earliest rho^2 the "PPI starts to pay for itself" marker may sit at.

The marker's own rule -- the cheapest ROUND rho^2 where EVERY eval type clears
1.25x -- lands on 0.30 for the mean-test figures and 0.40 for the rank ones.
That is a real difference and it is reported honestly in the per-family
numbers, but it makes the headline of one figure disagree with the headline of
its neighbour, which is worse than useless in a paper where a reader takes away
a single number.

Pinning all of them to the STRICTER of the two is the conservative direction:
0.40 is where PPI pays off whatever design the reader runs, so the quoted
threshold is never optimistic for anyone. A mean-test user is told to wait
slightly longer than they strictly must; nobody is told to expect a saving
that will not materialise.

What is NOT overridden is the number on the label: the multiplier is still
interpolated from the measured curve AT 0.40, so the figure says something
true. Only the choice of which round value to draw attention to is editorial.

Set to None to let each figure report its own measured crossing."""


_LABEL_EFF_ALIGNMENT_TARGETS_BY_EVAL_TYPE = {
    "binary":     (0.72, 0.62, 0.51, 0.41, 0.30, 0.20),
    "continuous": (0.76, 0.64, 0.51, 0.39, 0.26, 0.14),
    "likert":     (0.80, 0.67, 0.55, 0.42, 0.30, 0.17),
}
"""Per-eval-type judge-quality ladders, replacing one shared set of targets.

The targets are SCORE-LEVEL Pearson rho^2, but the quantity a practitioner
looks up depends on their design, and the map from one to the other is
eval-type specific -- and, for likert, distinctly NON-LINEAR. Measured tier ->
paired rho^2 on the 60-rep screen:

    likert  0.37->0.180  0.49->0.268  0.61->0.393  0.72->0.552  0.84->0.800

That relation is convex: the gap between a likert judge's score correlation
and its paired-difference correlation collapses as the judge gets cleaner,
because differencing two discretised scores only destroys signal while there
is noise left to discretise. A first version of these ladders extrapolated a
LINEAR fit and asked tier 0.96 to reach paired rho^2 0.70; it delivered 0.944,
overshooting so far that four of likert's six tiers landed above any range a
reader needs. These come from a quadratic refit inside the measured range.

A shared 0.20-0.70 ladder therefore covers wildly different ranges of the axis
the lookup figures are actually drawn on: likert's paired rho^2 only reached
0.505 at the top tier while continuous's never fell below 0.251. The
within-subjects likert panel simply had no data above 0.53, and no continuous
panel had any below 0.25.

These ladders are each eval type's own 0.20-0.70 span on the PAIRED axis,
inverted through the fits above. Same six tiers per eval type, so the sweep
costs exactly what it did.

Likert needs a much cleaner judge (up to 0.96 score-level) to reach the same
paired rho^2, because differencing two discretised Likert scores destroys more
of the judge's signal than differencing two continuous ones. All six ends were
checked as reachable by _calibrate_noise_for_alignment before being adopted.

These must cover 0.20-0.70 on FOUR axes at once, because the lookup grid
draws one panel per (structure, correlation) pair and each maps from the tier
differently. Measured spans (calibrate the tier, read _method_rho2 -- no sweep
needed):

    eval         group-Pearson  paired-Pearson  group-Spearman  paired-Spearman
    binary          0.20-0.73      0.16-0.80          --               --
    continuous      0.14-0.76      0.18-0.82      0.13-0.72        0.16-0.77
    likert          0.20-0.86      0.10-0.73      0.20-0.86        0.09-0.70

Likert's group and paired axes sit ~0.18 apart, so no six-tier ladder covers
0.20-0.70 on both without overshooting one of them. Overshoot is harmless --
gaps are not -- so the ladders are set wide enough that every axis covers the
range, and some run past it.

Verify against the per-method CSV's rho2 column after a run anyway. Three
earlier versions of this constant were wrong in ways only a run exposed: the
first extrapolated a LINEAR fit and missed likert's top by 0.24 rho^2 (asking
0.96 to give 0.70, getting 0.944, which spiked likert's curve to 10x and
flattened every other series); the second fixed the top but left likert's
floor at 0.26, above the 0.20 the figures mark; the third covered the paired
axes but left MWU's group-Spearman panel short at both ends. Check all four
axes, not the one being looked at."""


_LABEL_EFF_NOMINAL_TIERS = (0.70, 0.60, 0.50, 0.40, 0.30, 0.20)
"""Round labels for the judge-quality tiers, by ladder POSITION.

Once each eval type calibrates to its own targets, `alignment_target` takes
3 x 6 = 18 distinct values and any legend keyed on it grows to 18 entries at
arbitrary spacing -- which is what happened. The ladders are built so position
k means the same judge-quality band in every eval type, so position is the
thing worth labelling, and labelling it in round 0.1 steps keeps the legend
readable and comparable across panels.

The achieved score-level value is not lost: it stays in `alignment_value` and
in the calibration CSV."""


def _nominal_tier(eval_type: str, target: float) -> float:
    """Map an eval type's own calibration target to its round ladder label.

    Falls back to the target itself for anything not on a known ladder, so
    callers outside the label-efficiency sweep are unaffected."""
    lad = _LABEL_EFF_ALIGNMENT_TARGETS_BY_EVAL_TYPE.get(eval_type)
    if not lad or len(lad) != len(_LABEL_EFF_NOMINAL_TIERS):
        return float(target)
    i = min(range(len(lad)), key=lambda k: abs(lad[k] - target))
    return float(_LABEL_EFF_NOMINAL_TIERS[i])
"""Round, reader-legible judge-quality targets the label-efficiency
check's noise axis is calibrated to hit, per eval type -- six points
spanning "substantial/almost perfect" down to "fair" on the Landis & Koch
(1977) kappa scale (also read loosely against Cohen 1988's "large"/"medium"/
"small" correlation bands for continuous's Pearson r -- see _kappa_band/
_corr_band), dense enough in the 0.3-0.8 range to show where the
label-efficiency multiplier approaches 1x (no benefit over human-only
testing) as well as where it's largest. Chosen so a reader can ask "how
would this look with a kappa=0.8 judge" and get a direct answer, rather
than an uninterpretable llm_noise dial that means something different in
every eval type."""
_LABEL_EFF_ALIGNMENT_METRIC = {
    "continuous": ("rho2", "ρ²"),
    "likert": ("rho2", "ρ²"),
    "binary": ("rho2", "ρ²"),
}
"""Which alignment metric (metric_name, display_symbol) each eval type's
judge-quality axis is calibrated/labeled by -- rho^2, the squared Pearson
correlation between judge score and human label, for ALL THREE eval types.

This deliberately does NOT follow _ALIGNMENT_VIEWS' per-eval-type choice
(kappa / quadratic weighted kappa / Pearson r), which this axis used
previously. Three reasons, in order of importance:

1. rho^2 is what actually predicts the label-efficiency multiplier. PPI++
   with tuned lambda is a control variate, so the saving is
   1/(1 - rho^2*(1 - n_lab/N)) -- see _ppi_predicted_savings. Measured over
   a 48-cell noise x bias grid, rho^2 collapses all three eval types onto
   one curve (pooled R^2=0.975, 1.07x spread at matched value) where
   ICC/CCC manage 0.703/1.58x and Krippendorff's alpha 0.553/1.87x.
2. A per-eval-type metric made the panels NOT directly comparable. Under
   the old choice a shared "IRR~=0.8" legend entry meant kappa=0.8 for
   binary, weighted kappa=0.8 for likert and r=0.8 for continuous, which
   realize rho^2 = 0.667 / 0.683 / 0.640 respectively -- close enough to
   look alignable while quietly asserting an equivalence that does not
   hold. On this axis a tier means the same judge quality in every panel.
3. It is the unit the rule of thumb is stated in, so the threshold reads
   directly off the axis instead of needing a per-eval-type conversion.

CONSUMERS MUST NOT RE-SQUARE. `alignment_value` is now already rho^2, not
rho -- anything deriving (1 - rho^2) from it wants `1 - alignment_value`,
NOT `1 - alignment_value**2` (see simulations/fit_nformula_rule_of_thumb.py,
updated alongside this)."""

_NFORMULA_ALIGNMENT_TARGETS = (0.70, 0.45, 0.20)
"""Reduced subset of _LABEL_EFF_ALIGNMENT_TARGETS for run_ppi_nformula_
check -- the same 3 points build_ppi_label_efficiency_sources' own
alignment axis used before this session widened it to 6 (see _LABEL_EFF_
ALIGNMENT_TARGETS' docstring), and label_efficiency_table.py's own
DEFAULT_ALIGNMENT_TARGETS. Kept at 3 (not the full 6) because run_ppi_
nformula_check already crosses N x N_lab x effect_frac x alignment_target
-- a genuinely 4-factor grid -- and each additional alignment target
multiplies every other axis's cell count, on top of alignment
calibration's own align_n_mc=20,000-draw cost per (eval_type, target)
pair (which does NOT get cheaper by reusing _LABEL_EFF_ALIGNMENT_TARGETS'
own calibration -- N/effect_frac vary independently of alignment, so a
separate calibration pass is still required here, just over fewer
targets)."""


def _calibrate_noise_for_alignment(
    eval_type: str, target: float, metric_name: str, base_kwargs: dict,
    n_mc: int = 20_000, seed: int = 0, lo: float = 0.005, hi: float = 10.0, iters: int = 16,
) -> tuple[float, float]:
    """Bisect llm_noise to hit a target alignment metric value (see
    measure_judge_alignment) for one eval type's judge model, holding every
    other JudgeBiasSource field in `base_kwargs` fixed (bias_type/bias_delta/
    icc/etc -- everything _ppi_power_baseline(_binary) sets except
    llm_noise, overridden each bisection step). `effect_size` is fixed at
    0.0 for the calibration draw regardless of what the real sweep uses --
    measure_judge_alignment always reads group A (see its docstring), which
    NEVER carries generate_judge_bias_cell's injected effect (only group B
    does), so alignment is effect-size-independent by construction and
    calibrating against the real es would be redundant work, not more
    accurate.

    Alignment is monotonically DECREASING in llm_noise (a noisier judge
    agrees with truth less), so bisection over [lo, hi] is well-posed as
    long as the target is actually reachable in that range -- not asserted
    here; a target outside [measure(hi), measure(lo)] just converges to
    whichever endpoint is closer, which callers should read as "unreachable
    at this bias severity," not a precise calibration. This genuinely
    happens, not just from a too-narrow [lo, hi]: likert's target=0.8 caps
    out around weighted_kappa~=0.71 even at
    noise->0, because _ppi_power_baseline's SEVERE bias_delta alone (a
    purely systematic, non-noise miscalibration) already costs kappa more
    than a "target 0.8" judge could have -- quadratic-weighted kappa
    penalizes a large systematic offset harder than Pearson r does (r is
    shift-invariant; kappa isn't). Callers must label by the ACHIEVED value
    (see below), never silently claim the nominal target was hit.

    Returns (calibrated_noise, achieved_metric_value, all_metrics) -- the
    achieved value is a FRESH measurement at the final calibrated noise, not
    interpolated from the bisection steps, since callers should label
    plots/tables by what was actually achieved (MC noise in any single
    n_mc-sample measurement means it won't land exactly on `target`), not the
    nominal target.

    `all_metrics` is the FULL alignment panel (every metric
    measure_judge_alignment computes for this eval type) from that same final
    measurement -- i.e. the other IRR statistics that the calibrated judge
    happens to realize at the noise level chosen to hit `target` on
    `metric_name`. It costs nothing extra to carry (the bisection already
    computed it and previously discarded all but one key) and is what lets
    the calibration CSV answer "would this judge-quality tier look the same
    under a different reliability statistic?" without a re-run."""
    def _measure_all(noise: float) -> dict:
        kw = dict(base_kwargs)
        kw["llm_noise"] = noise
        kw["eval_type"] = eval_type
        sc = JudgeBiasSource(name="_align_cal", tag="_ref", effect_size=0.0, **kw)
        return measure_judge_alignment(sc, n_mc=n_mc, seed=seed)

    # rho^2 is NOT monotone in llm_noise, so it cannot be bisected on
    # directly. For binary, llm_noise is a flip PROBABILITY (see scenarios/
    # synthetic._jb_llm_binary): past 0.5 the judge is systematically
    # INVERTED, and rho^2 -- which discards the sign -- climbs back toward 1.
    # Measured at the binary baseline: r = 0.850 at noise 0.01, 0.205 at 0.40,
    # -0.010 at 0.50, -0.855 at 1.0, -1.000 by 2.0, so rho^2 traces
    # 0.72 -> 0.04 -> 0.00 -> 0.73 -> 1.00. A bisection assuming monotone
    # decrease sails past the zero crossing and converges on a perfectly
    # ANTI-correlated judge reported as rho^2 = 1.0 -- which is exactly what
    # happened on the first run of the rho^2 axis, on every binary tier.
    #
    # Bisect on the SIGNED pearson_r against sqrt(target) instead: r IS
    # monotone decreasing across the whole noise range, so the search is
    # well-posed for every eval type without per-type bounds. The achieved
    # value returned below is still rho^2, read from the same final
    # measurement. (The old per-eval-type metrics did not hit this because
    # kappa/weighted kappa also go negative under inversion.)
    search_metric, search_target = metric_name, target
    if metric_name == "rho2":
        search_metric = "pearson_r"
        search_target = float(np.sqrt(max(0.0, target)))

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if float(_measure_all(mid)[search_metric]) > search_target:
            lo = mid
        else:
            hi = mid
    final_noise = (lo + hi) / 2.0
    final_metrics = _measure_all(final_noise)
    return final_noise, float(final_metrics[metric_name]), final_metrics


def run_ppi_label_efficiency_check(
    n_reps: int, n_boot: int, ref_n_mc: int = 3000, align_n_mc: int = 20_000, seed: int = 71,
    n_workers: int = 1, progress_mode: str = "bar",
) -> tuple[list[LabelEfficiencyPoint], list[PPIComparisonResult], list[tuple[str, float, str, float, float, dict]]]:
    """Runs the label-efficiency comparison sweep (continuous/likert via
    build_ppi_label_efficiency_sources + _COMPARISON_METHODS, binary via
    build_ppi_label_efficiency_sources_binary + _COMPARISON_METHODS_BINARY),
    pools each judge-quality tier's PPI rejection rate across its method
    family, and inverts it against that eval type's own classical reference
    curve (_classical_pooled_power_curve, built ONCE per eval type -- it does
    not depend on judge quality, see that function's docstring) to get
    equiv_n_lab.

    The judge-quality axis is expressed as ALIGNMENT (Pearson r / weighted
    kappa / kappa, per _LABEL_EFF_ALIGNMENT_METRIC), not raw llm_noise: the
    same noise value means very different judge quality across eval types,
    so noise alone isn't a fair axis to compare panels against. Before the
    comparison sweep runs, llm_noise is CALIBRATED per eval type
    (_calibrate_noise_for_alignment) to hit _LABEL_EFF_ALIGNMENT_TARGETS
    (0.8/0.5/0.2) -- this directly answers "how would this look with a
    kappa=0.8 judge" instead of leaving the reader to guess what a given
    noise value means.

    N is fixed at 100 throughout (via _ppi_power_baseline(_binary), inherited
    unchanged) -- see save_ppi_label_efficiency_plot, which states this
    explicitly so the figure doesn't leave N_lab's denominator implicit.

    Returns a 3-tuple:
      - one LabelEfficiencyPoint per (eval_type, alignment_target, n_lab)
        cell, pooled across each eval type's method family -- the
        save_ppi_label_efficiency_plot input (unchanged from before).
      - the RAW, per-method PPIComparisonResult rows underlying that
        pooling (every method x scenario cell, before pool_ppi_comparison_
        across_methods averages them away) -- computing this sweep is
        expensive (the alignment calibration alone runs align_n_mc=20,000
        MC draws per target per eval type), so callers should persist this
        alongside the pooled points rather than let it be silently
        discarded, the way it previously was: a "is one method dragging
        the pooled average down" question could only be answered before by
        re-running the whole sweep from scratch. See
        save_results_artifacts_ppi_label_efficiency_raw.
      - the noise -> (eval_type, alignment_metric, target, achieved)
        calibration lookup, as a flat list of tuples (eval_type, noise,
        alignment_metric, target, achieved, all_metrics) -- needed to map the
        raw rows' embedded noise value (in PPIComparisonResult.name) back to
        the alignment level it was calibrated to hit, without re-running
        _calibrate_noise_for_alignment. `all_metrics` is that tier's full
        realized IRR panel (see _CALIB_EXTRA_METRIC_COLUMNS), carried so the
        calibration csv can report every reliability statistic the judge
        achieved, not only the one it was tuned on."""
    results: list[LabelEfficiencyPoint] = []
    all_raw: list[PPIComparisonResult] = []
    # 7-tuple here (trailing noise_family); run_ppi_nformula_check still emits
    # the 6-tuple form, so the shared CSV writer below tolerates both.
    calib_rows: list[tuple[str, float, str, float, float, dict, str]] = []

    cont_likert_baselines = {et: _ppi_power_baseline(et) for et in ("continuous", "likert")}
    binary_baseline = _ppi_power_baseline_binary()

    # Calibrate llm_noise -> target alignment level, per eval type, BEFORE
    # building the comparison-sweep sources (which need the calibrated
    # noise values as input, not the other way around).
    # Calibration runs PER (eval_type, noise_family), not once per eval type.
    # llm_noise -> alignment is family-dependent: at matched total error
    # variance a contaminated judge concentrates its errors on a few items, so
    # the same llm_noise lands on a different Pearson r than the gaussian arm
    # does. Calibrating once and reusing would silently put the two arms on
    # different judge-quality tiers, which is precisely the confound this axis
    # exists to remove.
    noise_by_eval_type: dict[tuple[str, str], tuple[float, ...]] = {}
    calib_info: dict[tuple[str, str], dict[float, tuple[float, float, dict]]] = {}
    for et, baseline in cont_likert_baselines.items():
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[et]
        _targets = _LABEL_EFF_ALIGNMENT_TARGETS_BY_EVAL_TYPE.get(et, _LABEL_EFF_ALIGNMENT_TARGETS)
        for fam, nf, fam_kw in PPI_LABEL_EFF_NOISE_FAMILIES:
            fam_baseline = {**baseline, "noise_family": nf, **fam_kw}
            noises, info = [], {}
            for target in _targets:
                noise, achieved, panel = _calibrate_noise_for_alignment(
                    et, target, metric_name, fam_baseline, n_mc=align_n_mc, seed=seed)
                noises.append(noise)
                info[noise] = (target, achieved, panel)
            noise_by_eval_type[(et, fam)] = tuple(noises)
            calib_info[(et, fam)] = info

    # Binary calibrates on the gaussian arm only. Its contaminated arm is
    # implemented and produces different data, but statistically identical
    # results (phi 0.6296 vs 0.6287 at n=400k) -- see
    # build_ppi_label_efficiency_sources_binary for the derivation and for the
    # design that WOULD make binary shape-sensitive.
    metric_name_bin, _ = _LABEL_EFF_ALIGNMENT_METRIC["binary"]
    bin_noises_by_fam: dict[str, tuple[float, ...]] = {}
    for fam, nf, fam_kw in [f for f in PPI_LABEL_EFF_NOISE_FAMILIES if f[1] == "gaussian"]:
        fam_baseline = {**binary_baseline, "noise_family": nf, **fam_kw}
        noises, info = [], {}
        for target in _LABEL_EFF_ALIGNMENT_TARGETS_BY_EVAL_TYPE.get(
                "binary", _LABEL_EFF_ALIGNMENT_TARGETS):
            noise, achieved, panel = _calibrate_noise_for_alignment(
                "binary", target, metric_name_bin, fam_baseline, n_mc=align_n_mc, seed=seed,
            )
            noises.append(noise)
            info[noise] = (target, achieved, panel)
        bin_noises_by_fam[fam] = tuple(noises)
        calib_info[("binary", fam)] = info

    for (et, fam), info in calib_info.items():
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[et]
        for noise, (target, achieved, panel) in info.items():
            calib_rows.append((et, noise, metric_name, target, achieved, panel, fam))

    # Sweep PPI_LABEL_EFF_EFFECT_FRACS rather than a single effect size: one
    # es cannot keep the whole N_lab grid in the reference curve's steep
    # middle, and the multiplier's noise is dominated by that curve's local
    # slope (see PPI_LABEL_EFF_EFFECT_FRACS / _multiplier_ci). The arms
    # overlap deliberately -- the multiplier should be es-invariant, so
    # agreement across arms on shared n_lab cells is a robustness check.
    # Grid the classical reference curve is tabulated on. _equivalent_n_lab
    # inverts this curve with np.interp, which clamps at the endpoints, so
    # this cap is a hard ceiling on any reportable multiplier (multiplier =
    # equiv_n_lab / n_lab). A cap of 500 silently truncated binary's
    # best-performing tier (true multiplier ~4x, needing equiv ~800 at
    # n_lab=200) into looking worse than likert. 1500 gives headroom past
    # binary's ~800; the extra grid points keep low-end resolution despite
    # the wider span.
    n_grid = np.geomspace(float(_JB_MIN_LAB), 1500.0, 36)
    for effect_frac in PPI_LABEL_EFF_EFFECT_FRACS:
        cont_likert_sources = build_ppi_label_efficiency_sources(
            noise_by_eval_type=noise_by_eval_type, effect_frac=effect_frac,
        )
        binary_sources = build_ppi_label_efficiency_sources_binary(
            noise_levels=bin_noises_by_fam, effect_frac=effect_frac,
        )
        # One group per (eval_type, noise_family): each needs its own
        # reference curve lookup and its own calibration table, and grouping
        # them together would pool two different judge-error shapes into one
        # multiplier.
        groups = []
        for fam, _nf, _fam_kw in PPI_LABEL_EFF_NOISE_FAMILIES:
            for et, methods in (("continuous", _COMPARISON_METHODS),
                                ("likert", _COMPARISON_METHODS),
                                ("binary", _COMPARISON_METHODS_BINARY)):
                if et == "binary" and fam not in bin_noises_by_fam:
                    continue  # gaussian-only; see the calibration note above
                src = [x for x in (binary_sources if et == "binary" else cont_likert_sources)
                       if x.eval_type == et and x.noise_family == fam]
                groups.append((et, fam, src, methods,
                               rf"labeleff\.{et}\.fam={fam}\.noise=([\d.]+)\.lab=[\d.]+"))
        # One phase per non-empty group, per effect-size arm (this loop sits
        # inside `for effect_frac in PPI_LABEL_EFF_EFFECT_FRACS`), declared so
        # each bar says which phase it is -- a single bar here reads as the
        # whole check otherwise. resume=True so the count carries across arms.
        _ProgressReporter.phase_plan(
            sum(1 for g in groups if g[2]) * len(PPI_LABEL_EFF_EFFECT_FRACS),
            resume=True)
        for eval_type, noise_family, sources, methods, name_re in groups:
            if not sources:
                continue
            es = sources[0].effect_size
            # Smoothed, strictly-monotone reference curve: the raw MC curve
            # ties across adjacent grid points at ref_n_mc, and inverting a
            # tie biases equiv_n_lab downward exactly where the curve is
            # flattest -- see _smooth_monotone_power_curve.
            power_grid = _smooth_monotone_power_curve(
                n_grid, _classical_pooled_power_curve(eval_type, es, methods, n_grid, ref_n_mc, seed),
            )
            raw = run_ppi_comparison_simulation(
                sources, n_reps, n_boot, methods=methods, seed=seed, n_workers=n_workers,
                progress_mode=progress_mode,
            )
            all_raw.extend(raw)
            pooled = pool_ppi_comparison_across_methods(raw)
            # Free correctness check on the inversion this loop is about to use
            # (see _check_inversion_self_consistency). Runs before any
            # multiplier is derived, so a biased curve is reported at the point
            # it would start contaminating results.
            _check_inversion_self_consistency(
                [q.n_lab for q in pooled],
                [q.rejects_human_subset / q.n_reps if q.n_reps else float("nan") for q in pooled],
                n_grid, power_grid, f"{eval_type}/{noise_family} es={es:.4f}",
            )
            metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[eval_type]
            for r in pooled:
                m = re.match(name_re, r.name)
                if not m:
                    raise ValueError(f"run_ppi_label_efficiency_check: could not parse noise from {r.name!r}")
                noise = float(m.group(1))
                # The scenario name round-trips the calibrated noise through a
                # %.4f format, so an exact dict lookup can miss on precision --
                # match to the closest calibrated value instead.
                _cal = calib_info[(eval_type, noise_family)]
                closest_noise = min(_cal, key=lambda n: abs(n - noise))
                target, achieved, _panel = _cal[closest_noise]
                # The pooled multiplier averages across `methods`, so its
                # prediction must average the SAME methods' own correlations.
                # The calibration panel's rho2 is SCORE-level, which is right
                # only for group-structure tests -- paired tests (paired_t,
                # wilcoxon) operate on differences D = Y_x - Y_y, whose
                # correlation is a different number.
                #
                # It is usually smaller, so using score-level over-predicted
                # and looked harmless. Not always: binary's top tier is a 2%
                # flip-rate judge where difference-level rho^2 CROSSES ABOVE
                # score-level (0.775 vs 0.700), so the prediction came out too
                # LOW and the measured multiplier appeared to beat its own
                # control-variate bound by 1.37x -- an impossibility, and the
                # single most flaggable thing in the figure.
                #
                # Averaging the per-method predictions (rather than predicting
                # from an averaged rho^2) matches how the multiplier itself is
                # pooled. _method_rho2 is lru_cached, so this is one extra
                # measurement per (eval_type, noise, method, family), not per
                # cell.
                _m_r2 = [_method_rho2(eval_type, round(noise, 6), _m, noise_family)[0]
                         for _m in methods]
                _m_r2 = [v for v in _m_r2 if np.isfinite(v)]
                _r2 = float(np.mean(_m_r2)) if _m_r2 else float(_panel.get("rho2", float("nan")))
                ppi_power = r.rejects_ppi / r.n_reps if r.n_reps else float("nan")
                equiv = _equivalent_n_lab(ppi_power, n_grid, power_grid) if np.isfinite(ppi_power) else float("nan")
                saturated = bool(np.isfinite(ppi_power) and ppi_power >= power_grid.max() - 1e-9)
                lo, hi = _multiplier_ci(ppi_power, r.n_reps, r.n_lab, n_grid, power_grid)
                # Same curve, same inversion, but on the arm that uses no
                # judge scores -- so it measures this cell's conditioning
                # without touching what is being estimated. See
                # LabelEfficiencyPoint.inversion_ratio.
                _hp = r.rejects_human_subset / r.n_reps if r.n_reps else float("nan")
                _inv_h = _equivalent_n_lab(_hp, n_grid, power_grid) if np.isfinite(_hp) else float("nan")
                inv_ratio = _inv_h / r.n_lab if (r.n_lab and np.isfinite(_inv_h)) else float("nan")
                inv_clamped = bool(np.isfinite(_inv_h) and (
                    _inv_h <= n_grid.min() + 1e-9 or _inv_h >= n_grid.max() - 1e-9))
                results.append(LabelEfficiencyPoint(
                    eval_type=eval_type, judge_noise=noise, alignment_metric=metric_name,
                    alignment_target=_nominal_tier(eval_type, target), alignment_value=achieved,
                    n_lab=r.n_lab, ppi_power=ppi_power, equiv_n_lab=equiv, n_reps=r.n_reps,
                    saturated=saturated, effect_frac=effect_frac, mult_lo=lo, mult_hi=hi,
                    rho2=_r2,
                    predicted_mult=(float(np.mean([_ppi_predicted_savings(v, r.n_lab, r.n)
                                                   for v in _m_r2])) if _m_r2
                                    else _ppi_predicted_savings(_r2, r.n_lab, r.n)),
                    predicted_mult_asymptotic=(float(np.mean([_ppi_predicted_savings(v, 0, 1)
                                                              for v in _m_r2])) if _m_r2
                                               else _ppi_predicted_savings(_r2, 0, 1)),
                    inversion_ratio=inv_ratio, inversion_clamped=inv_clamped,
                    noise_family=noise_family,
                    variance_multiplier=(r.var_human_subset / r.var_ppi
                                         if getattr(r, "var_ppi", 0)
                                         and np.isfinite(r.var_ppi) else float("nan")),
                ))
    _ProgressReporter.clear_phase_plan()
    return results, all_raw, calib_rows


def run_ppi_nformula_check(
    n_reps: int, n_boot: int, ref_n_mc: int = 10_000, align_n_mc: int = 50_000, seed: int = 73,
    n_workers: int = 1, progress_mode: str = "bar",
) -> tuple[list[LabelEfficiencyPoint], list[PPIComparisonResult], list[tuple[str, float, str, float, float, dict]]]:
    """N x N_lab x effect_size x judge-quality label-efficiency sweep --
    extends run_ppi_label_efficiency_check (which holds N=PPI_LABEL_EFF_N
    and effect_size=PPI_LABEL_EFF_EFFECT_FRAC fixed) by also sweeping those
    two axes, via build_ppi_nformula_sources(_binary)/PPI_NFORMULA_N_VALUES/
    PPI_NFORMULA_NLAB_VALUES/PPI_NFORMULA_EFFECT_FRACS. Exists to derive
    (and check) a closed-form rule-of-thumb formula for the label-
    efficiency multiplier that includes N explicitly and holds across
    effect sizes -- the base asymptotic PPI++ formula N_lab' ~= N_lab /
    (1 - rho^2) drops N because it assumes N is large relative to N_lab,
    which run_ppi_label_efficiency_check's own fixed (N=1000, N_lab<=200,
    ratio>=5) design never tested outside of.

    ref_n_mc/align_n_mc default higher than run_ppi_label_efficiency_check's
    matching defaults (3000/20_000) -- deliberately, not an oversight: this
    check's whole output feeds a regression (fit_nformula_rule_of_thumb.py)
    whose coefficient standard errors are sensitive to per-cell noise in
    `multiplier` (a ratio-of-ratios inversion, worst for continuous's
    steep classical power curve), unlike the original label-efficiency
    check's per-cell table/plot use, which tolerates more per-cell noise.
    Both knobs are cheap to raise here regardless: each is evaluated only
    once per (eval_type, effect_frac) or (eval_type, target) combination
    -- 9 calibration draws, 9 reference curves -- not once per (N, N_lab)
    grid cell, so raising them doesn't scale with the 432-cell grid the
    way n_reps/n_boot do.

    Same alignment-calibration convention as run_ppi_label_efficiency_check
    (_calibrate_noise_for_alignment, over _NFORMULA_ALIGNMENT_TARGETS -- a
    reduced 3-point subset, see that constant's docstring for why), and the
    same classical reference-curve machinery (_classical_pooled_power_
    curve/_equivalent_n_lab) -- but now one reference curve per (eval_type,
    effect_frac) pair (PPI_NFORMULA_EFFECT_FRACS), not one per eval_type,
    since the reference itself depends on effect size. The reference curve
    does NOT depend on N (see _classical_pooled_power_curve's docstring --
    it's built from a synthetic n_grid unrelated to the real sweep's N/
    N_lab split), so each (eval_type, effect_frac) curve is correctly
    computed once and reused across every N value.

    Returns the same 3-tuple shape as run_ppi_label_efficiency_check --
    pooled LabelEfficiencyPoint rows (now with `n`/`effect_frac` varying
    instead of held at their default), raw per-method PPIComparisonResult
    rows, and the calibration lookup -- reusing that output schema (and
    save_results_artifacts_ppi_label_efficiency_raw for the raw CSV)
    rather than a parallel one, so rows from this sweep can be directly
    compared against/merged with the original label-efficiency check's:
    filtering to n=PPI_LABEL_EFF_N, effect_frac=PPI_LABEL_EFF_EFFECT_FRAC
    should closely reproduce (modulo n_lab/alignment-target subsetting and
    independent MC noise) that check's own results -- a useful sanity
    check in itself. Use save_results_artifacts_ppi_nformula (NOT save_
    results_artifacts_ppi_label_efficiency) for the pooled CSV, since the
    latter's writer doesn't emit the n/effect_frac columns this sweep
    needs to be interpretable."""
    results: list[LabelEfficiencyPoint] = []
    all_raw: list[PPIComparisonResult] = []
    calib_rows: list[tuple[str, float, str, float, float, dict]] = []

    cont_likert_baselines = {et: _ppi_power_baseline(et) for et in ("continuous", "likert")}
    binary_baseline = _ppi_power_baseline_binary()

    # Calibrate llm_noise -> target alignment level, per eval type -- same
    # as run_ppi_label_efficiency_check, and equally independent of N/
    # effect_frac here (see _calibrate_noise_for_alignment's docstring:
    # alignment is measured off group A, which never carries the injected
    # effect, and is unrelated to N/N_lab).
    noise_by_eval_type: dict[str, tuple[float, ...]] = {}
    calib_info: dict[str, dict[float, tuple[float, float]]] = {}
    for et, baseline in cont_likert_baselines.items():
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[et]
        noises, info = [], {}
        for target in _NFORMULA_ALIGNMENT_TARGETS:
            noise, achieved, panel = _calibrate_noise_for_alignment(et, target, metric_name, baseline, n_mc=align_n_mc, seed=seed)
            noises.append(noise)
            info[noise] = (target, achieved, panel)
        noise_by_eval_type[et] = tuple(noises)
        calib_info[et] = info

    metric_name_bin, _ = _LABEL_EFF_ALIGNMENT_METRIC["binary"]
    bin_noises, bin_info = [], {}
    for target in _NFORMULA_ALIGNMENT_TARGETS:
        noise, achieved, panel = _calibrate_noise_for_alignment(
            "binary", target, metric_name_bin, binary_baseline, n_mc=align_n_mc, seed=seed,
        )
        bin_noises.append(noise)
        bin_info[noise] = (target, achieved, panel)
    calib_info["binary"] = bin_info

    for et, info in calib_info.items():
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[et]
        for noise, (target, achieved, panel) in info.items():
            calib_rows.append((et, noise, metric_name, target, achieved, panel))

    cont_likert_sources = build_ppi_nformula_sources(noise_by_eval_type=noise_by_eval_type)
    groups = [
        ("continuous", [s for s in cont_likert_sources if s.eval_type == "continuous"],
         _COMPARISON_METHODS, r"nformula\.continuous\.noise=([\d.]+)\.n=\d+\.lab=\d+\.es=[\d.]+"),
        ("likert", [s for s in cont_likert_sources if s.eval_type == "likert"],
         _COMPARISON_METHODS, r"nformula\.likert\.noise=([\d.]+)\.n=\d+\.lab=\d+\.es=[\d.]+"),
        ("binary", build_ppi_nformula_sources_binary(noise_levels=tuple(bin_noises)),
         _COMPARISON_METHODS_BINARY, r"nformula\.binary\.noise=([\d.]+)\.n=\d+\.lab=\d+\.es=[\d.]+"),
    ]

    # One classical reference curve per (eval_type, effect_frac) -- NOT per
    # N (see docstring above) -- precomputed once and reused across every
    # N/alignment-target row at that (eval_type, effect_frac).
    # Grid cap of 1500 -- same reasoning as run_ppi_label_efficiency_check's
    # n_grid (a lower cap silently truncates binary's best-performing tier).
    n_grid = np.geomspace(float(_JB_MIN_LAB), 1500.0, 36)
    ref_curves: dict[tuple[str, float], np.ndarray] = {}
    for eval_type, _sources, methods, _name_re in groups:
        for frac in PPI_NFORMULA_EFFECT_FRACS:
            es = _jb_effect_magnitude_binary(frac) if eval_type == "binary" else _jb_effect_magnitude(eval_type, frac)
            ref_curves[(eval_type, frac)] = _classical_pooled_power_curve(eval_type, es, methods, n_grid, ref_n_mc, seed)

    for eval_type, sources, methods, name_re in groups:
        if not sources:
            continue
        raw = run_ppi_comparison_simulation(
            sources, n_reps, n_boot, methods=methods, seed=seed, n_workers=n_workers,
            progress_mode=progress_mode,
        )
        all_raw.extend(raw)
        pooled = pool_ppi_comparison_across_methods(raw)
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[eval_type]
        for r in pooled:
            m = re.match(name_re, r.name)
            if not m:
                raise ValueError(f"run_ppi_nformula_check: could not parse noise from {r.name!r}")
            noise = float(m.group(1))
            closest_noise = min(calib_info[eval_type], key=lambda n: abs(n - noise))
            target, achieved, _panel = calib_info[eval_type][closest_noise]
            closest_frac = min(PPI_NFORMULA_EFFECT_FRACS, key=lambda f: abs(f - r.effect_size))
            power_grid = ref_curves[(eval_type, closest_frac)]
            ppi_power = r.rejects_ppi / r.n_reps if r.n_reps else float("nan")
            equiv = _equivalent_n_lab(ppi_power, n_grid, power_grid) if np.isfinite(ppi_power) else float("nan")
            saturated = bool(np.isfinite(ppi_power) and ppi_power >= power_grid.max() - 1e-9)
            results.append(LabelEfficiencyPoint(
                eval_type=eval_type, judge_noise=noise, alignment_metric=metric_name,
                alignment_target=target, alignment_value=achieved,
                n_lab=r.n_lab, ppi_power=ppi_power, equiv_n_lab=equiv, n_reps=r.n_reps, saturated=saturated,
                n=r.n, effect_frac=r.effect_size,
            ))
    return results, all_raw, calib_rows


def print_ppi_label_efficiency_report(results: list[LabelEfficiencyPoint]) -> None:
    """N_lab / PPI power / equivalent human-only N_lab / multiplier table,
    grouped by eval_type then alignment_target -- the console/console-log
    counterpart of save_ppi_label_efficiency_plot. N is fixed at 100
    throughout (see run_ppi_label_efficiency_check's docstring)."""
    if not results:
        print("\n  (no label-efficiency results)")
        return
    print(
        f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- LABEL EFFICIENCY (effective sample size)\n"
        f"  N={PPI_LABEL_EFF_N} total items throughout; only N_lab (and its share of N) varies\n{'='*88}"
    )
    for et in sorted({r.eval_type for r in results}):
        print(f"\n  [{et}]")
        for target in sorted({r.alignment_target for r in results if r.eval_type == et}, reverse=True):
            rows = sorted(
                (r for r in results if r.eval_type == et and r.alignment_target == target),
                key=lambda r: r.n_lab,
            )
            metric = rows[0].alignment_metric
            achieved_vals = {r.alignment_value for r in rows}
            achieved_str = f"{sum(achieved_vals) / len(achieved_vals):.3f}" if achieved_vals else "n/a"
            print(f"    target {metric}={target:.2f}  (achieved ~{achieved_str}, noise={rows[0].judge_noise:.4f})")
            print(f"      {'N_lab':>8} {'ppi_power':>10} {'equiv_N_lab':>12} {'multiplier':>11}")
            for r in rows:
                mult = r.equiv_n_lab / r.n_lab if r.n_lab else float("nan")
                flag = "  (saturated, lower bound)" if r.saturated else ""
                print(f"      {r.n_lab:>8} {r.ppi_power:>10.3f} {r.equiv_n_lab:>12.1f} {mult:>10.2f}x{flag}")
    print()


PPI_RHO_DRIFT_EFFECT_FRACS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
"""Effect sizes (in population-SD units, per _jb_effect_magnitude) the
rho-drift check sweeps.

Deliberately MUCH wider than PPI_LABEL_EFF_EFFECT_FRACS (0.15-0.35). That
narrow band is right for its own purpose -- keeping the N_lab grid inside the
reference power curve's steep middle -- but it is exactly why the effect
dependence this check exists to measure went unnoticed: across 0.15-0.35 the
drift is ~0.3%, indistinguishable from Monte Carlo noise. It only becomes
visible past d ~ 0.5, and the interesting regime runs to d = 2 where the rank
statistics saturate. 0.0 is included as the anchor every named correlation is
implicitly calibrated at."""

PPI_RHO_DRIFT_ALIGNMENT_TARGET = 0.64
"""Single judge-quality tier (score-level rho^2) the drift sweep pins every
cell to, via _calibrate_noise_for_alignment.

The whole measurement rests on judge quality being HELD FIXED while the effect
moves -- a sweep that varied both would confound exactly what it is trying to
separate. One tier rather than _LABEL_EFF_ALIGNMENT_TARGETS' several, because
the drift is a property of the estimand rather than of judge quality: quality
sets how FAST rho falls (-32%/-59%/-75% at r = .95/.8/.6 for friedman), not
whether it does. 0.64 == r 0.8, the middle tier, chosen so the fall has room
to be visible without the judge being so good that rho^2 starts near its
ceiling."""


@dataclass
class RhoDriftPoint:
    """One (eval_type, method, effect_frac) cell of the rho-drift check."""
    eval_type: str
    method: str
    effect_frac: float
    judge_noise: float
    alignment_value: float
    """Achieved score-level rho^2 for the judge at this tier -- the quantity a
    named correlation recipe estimates, and which is constant by construction
    across every effect_frac in the sweep."""
    n: int
    n_lab: int
    n_reps: int
    variance_multiplier: float
    """Var(human-subset estimate) / Var(PPI estimate) over replicates, from
    PPIComparisonResult.var_human_subset / .var_ppi -- the same
    direct-variance route LabelEfficiencyPoint.variance_multiplier uses, with
    no power curve to invert."""
    rho2_implied: float
    """The rho^2 the measured multiplier implies, by inverting the N_eff
    formula: rho2 = (1 - 1/M) / (1 - n_lab/N). THIS is the quantity the
    label-efficiency formula actually needs. It is what drifts."""
    rho2_recipe: float
    """What this method's named-correlation recipe returns (_method_rho2, via
    _METHOD_CORR_KIND). NaN for methods with no entry -- currently the four
    omnibus tests, deliberately (see _METHOD_CORR_KIND's TODO)."""
    rho2_score: float
    """The structure-appropriate SCORE-level rho^2 measured ON THIS CELL, i.e.
    at this effect size: Corr(D, Dhat)^2 for "pair" methods, the within-group
    pooled correlation for "group" ones.

    This is the reference the control needs, and it is NOT constant across the
    sweep even though llm_noise is. _calibrate_noise_for_alignment pins
    alignment measured on the INDEPENDENT-GROUP scores, which does not pin
    Corr(D, Dhat) for a pair-structure method: in the bounded harness scenario
    the latter rises 0.707 -> 0.742 over d = 0 -> 2 while llm_noise is fixed.
    A mean-type method's rho MUST equal this quantity (its influence function
    is linear in the value), so "flat" was the wrong control -- "tracks
    rho2_score" is the right one, and paired_t passes it to within 1% at every
    effect while failing a flatness test by +5.3%."""
    n_eff_implied: float
    n_eff_recipe: float
    n_eff_error: float
    """n_eff_recipe / n_eff_implied - 1: the error a planner suffers by using
    the named recipe. NaN when the method has no recipe entry."""
    rho2_implied_se: float = float("nan")
    """Monte-Carlo SE of rho2_implied on THIS cell (paired bootstrap over
    replicates -- see PPIComparisonResult.rho2_implied_se, which this copies).

    The control reads against this rather than against a bare tolerance. A
    mean-type method's deviation from rho2_score is exact-zero in
    expectation, so any reading is |noise|, and at R=200 that noise is large
    enough (and, for the "pair" structures, skewed low enough) to look like a
    finding. Reporting the SE is what separates "the estimator is wrong" from
    "we did not run enough replicates"."""
    rho2_evalstats: float = float("nan")
    """What the SHIPPED LIBRARY returns for this method -- the test-specific
    linearization in evalstats.alignment (_linearize_for_test), i.e. the
    number judge_alignment(..., test=...) hands a user and builds its n_eff
    from.

    Deliberately distinct from rho2_recipe. rho2_recipe is THIS HARNESS's
    own named-correlation table (_METHOD_CORR_KIND: raw Spearman for the
    rank methods), which is effect-invariant by construction and therefore
    cannot track rho2_implied once a real effect exists. The library instead
    correlates each estimand's INFLUENCE FUNCTION -- Hajek projection for
    wilcoxon, empirical placements for mwu, identity for the mean-type ones
    -- which can track. Plotting both against rho2_implied is the point: it
    shows whether the number a user actually receives is the one the N_eff
    formula needs.

    NaN when evalstats.alignment has no linearization for this method, or
    the cell's structure doesn't supply the arrays it needs."""


_RHO_DRIFT_EVALSTATS_TEST = {
    TTEST.name:        ("ttest", "between"),
    TTEST_WELCH.name:  ("ttest", "between"),
    PAIRED_T.name:     ("ttest", "within"),
    MWU.name:          ("mannwhitney", "between"),
    WILCOXON.name:     ("wilcoxon", "within"),
    ANOVA_IND.name:    ("anova_oneway", "between"),
    ANOVA_REP.name:    ("anova_oneway", "within"),
    KRUSKAL.name:      ("kruskalwallis", "between"),
    KRUSKAL_INFLUENCE.name:       ("kruskalwallis", "between"),
    KRUSKAL_INFLUENCE_FLOOR.name: ("kruskalwallis", "between"),
    FRIEDMAN.name:     ("friedman", "within"),
}
"""Harness method name -> (evalstats.alignment test name, design) for
_rho_drift_evalstats_rho2. Maps this harness's own method vocabulary onto
judge_alignment's public `test=` values, so the drift plot can show what the
SHIPPED library would report for the same cell. anova_rep maps to
anova_oneway/"within" because that is exactly what judge_alignment calls a
repeated-measures one-way design (see _linearize_mean's within branch, which
double-centres at k>2)."""


def _rho_drift_evalstats_rho2(sc: JudgeBiasSource, method: str, seed: int,
                              n_mc: int = 40_000) -> float:
    """rho^2 as the SHIPPED library computes it -- evalstats.alignment's
    test-specific linearization -- measured on the same fresh draw
    _rho_drift_score_rho2 uses, at this cell's own effect size.

    Reads the structure-appropriate truth/llm arrays via
    _COMPARISON_CELL_FIELDS (the same map _run_ppi_comparison_cell uses),
    builds the {condition: (judge, human)} dict judge_alignment's
    multi-condition form expects, and takes Pearson r^2 of the linearized
    pair. Human arrays are passed dense (no NaN) because this is a
    large-sample measurement of the judge, not a labeled-subset estimate.

    Returns NaN rather than raising if the method has no mapping or the cell
    lacks the fields -- a missing line in one panel is a better failure than
    taking down the whole sweep."""
    mapped = _RHO_DRIFT_EVALSTATS_TEST.get(method)
    if mapped is None:
        return float("nan")
    test_name, design = mapped
    # _COMPARISON_METHOD_STRUCTURE maps method -> a plain STRING ("group",
    # "pair", "group3", "pair3"), unlike _METHOD_CORR_KIND's (structure, kind)
    # tuple -- and it is the only one of the two that covers the omnibus
    # methods (_METHOD_CORR_KIND has no entries for them; see its TODO). Read
    # the string map first so all nine methods resolve.
    structure = _COMPARISON_METHOD_STRUCTURE.get(method)
    if structure is None:
        structure = (_METHOD_CORR_KIND.get(method, (None, None)))[0]
    if structure in ("paired", "pair"):
        structure = "pair"
    if structure not in _COMPARISON_CELL_FIELDS:
        return float("nan")
    llm_fields, _lab_fields, truth_fields, _mask_kind = _COMPARISON_CELL_FIELDS[structure]

    try:
        from evalstats.alignment import _linearize_for_test
        from scipy.stats import pearsonr
        cell = generate_judge_bias_cell(replace(sc, n=n_mc), np.random.default_rng(seed))
        conditions = {}
        for i, (lf, tf) in enumerate(zip(llm_fields, truth_fields)):
            judge = np.asarray(getattr(cell, lf), dtype=float)
            human = np.asarray(getattr(cell, tf), dtype=float)
            conditions[chr(ord("A") + i)] = (judge, human)
        jl, hl = _linearize_for_test(conditions, test=test_name, design=design)[:2]
        if len(jl) < 3 or float(np.std(jl)) < 1e-12 or float(np.std(hl)) < 1e-12:
            return float("nan")
        return float(pearsonr(jl, hl).statistic) ** 2
    except Exception:
        return float("nan")


def _rho_drift_score_rho2(sc: JudgeBiasSource, method: str, seed: int,
                          n_mc: int = 40_000) -> float:
    """Structure-appropriate score-level rho^2 for `sc`'s judge, measured AT
    sc's own effect size on a large fresh draw.

    "pair" methods correlate the DIFFERENCES (D vs Dhat); "group" methods use
    the within-group-centred pooled correlation, the same quantity
    _method_rho2's group branch forms. Spearman for the rank methods, Pearson
    for the mean ones, per _METHOD_CORR_KIND. Measured per effect rather than
    once, because it is not effect-invariant in a bounded scenario -- see
    RhoDriftPoint.rho2_score."""
    from scipy.stats import pearsonr, spearmanr
    structure, kind = _METHOD_CORR_KIND.get(method, ("group", "pearson"))
    cell = generate_judge_bias_cell(replace(sc, n=n_mc), np.random.default_rng(seed))
    # NB _METHOD_CORR_KIND says "paired" where _COMPARISON_METHOD_STRUCTURE
    # says "pair" -- accept both, since silently taking the group branch for a
    # paired method returns an effect-invariant number and hides the very
    # movement this is here to measure.
    if structure in ("paired", "pair"):
        a = np.asarray(cell.truth_x, float) - np.asarray(cell.truth_y, float)
        b = np.asarray(cell.llm_x, float) - np.asarray(cell.llm_y, float)
    else:
        _a1 = np.asarray(cell.truth_a2, float); _b1 = np.asarray(cell.llm_a2, float)
        _a2 = np.asarray(getattr(cell, "truth_b2", _a1), float)
        _b2 = np.asarray(getattr(cell, "llm_b2", _b1), float)
        a = np.concatenate([_a1 - _a1.mean(), _a2 - _a2.mean()])
        b = np.concatenate([_b1 - _b1.mean(), _b2 - _b2.mean()])
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    r = (spearmanr(a, b).statistic if kind == "spearman" else pearsonr(a, b).statistic)
    return float(r) ** 2


def run_ppi_rho_drift_check(
    n_reps: int,
    n_boot: int,
    seed: int,
    effect_fracs: tuple[float, ...] = PPI_RHO_DRIFT_EFFECT_FRACS,
    n_lab_target: int = 100,
    eval_types: tuple[str, ...] = ("continuous",),
    align_n_mc: int = 20_000,
    n_workers: int = 1,
    progress_mode: str = "bar",
    only_methods: tuple[str, ...] | None = None,
    shape_label: str | None = None,
) -> tuple[list[RhoDriftPoint], list[tuple]]:
    """Is rho^2 a property of the judge, or of the judge and the design?

    Every label-efficiency number in this harness assumes the former:
    _method_rho2 builds its cell at effect_size=0.0 and caches on
    (eval_type, judge_noise, method), with no effect-size term. This check
    tests that assumption directly by holding judge quality pinned at
    PPI_RHO_DRIFT_ALIGNMENT_TARGET and sweeping the true effect, then
    inverting the measured multiplier back to the rho^2 it implies.

    The assumption holds exactly for mean-type estimands and fails for
    rank/dominance ones: PPI's variance reduction is 1 - rho^2 with rho
    correlating influence functions, and a mean's psi(y) = y - mu makes rho a
    plain Pearson correlation a location shift cannot move, while rank and
    dominance estimands have psi involving the CDF, whose shape changes as
    groups separate. Expect the mean-type methods (ttest, paired_t) to come
    back flat and the rank ones (mwu, wilcoxon) to fall -- a mean-type method
    showing drift is a bug in the measurement, not a finding, since its
    invariance is exact algebra and doubles as this check's own control.
    Spearman-based recipes (mwu, wilcoxon, kruskal) are shift-invariant, so
    they stand still while the true rho2 falls beneath them; friedman's
    recipe (mean per-participant Spearman on within-row ranks) is not even
    shift-invariant and moves the opposite direction, rising as the truth
    falls.

    Reports both rho2_recipe (the named recipe's value) and rho2_implied
    (inverted from the measured multiplier) -- the gap between them is the
    finding. See notes/omnibus_label_efficiency.html for the full measurement
    this check productionizes.

    Returns (points, calib_rows) -- calib_rows in the same shape
    run_ppi_label_efficiency_check emits, so it can reuse
    save_results_artifacts_ppi_label_efficiency_raw's calibration writer.
    """
    from simulations.harness.scenarios.synthetic import PPI_LABEL_EFF_N

    points: list[RhoDriftPoint] = []
    calib_rows: list[tuple] = []
    label_frac = n_lab_target / PPI_LABEL_EFF_N

    for et in eval_types:
        # shape_label goes into the baseline kwargs, so the calibration, the
        # cells, rho2_score and rho2_evalstats all draw from the SAME marginal.
        # _method_rho2 is the one that needs it passed explicitly (below): it
        # rebuilds its own baseline rather than receiving this one.
        baseline = (_ppi_power_baseline_binary() if et == "binary"
                    else _ppi_power_baseline(et))
        if shape_label is not None:
            baseline = {**baseline, "shape_label": shape_label}
        metric_name, _ = _LABEL_EFF_ALIGNMENT_METRIC[et]
        # One calibration per eval type -- alignment is measured off group A,
        # which never carries the injected effect, so it is independent of
        # effect_frac (see _calibrate_noise_for_alignment's docstring). That
        # independence is what lets one noise value serve every effect cell.
        noise, achieved, panel = _calibrate_noise_for_alignment(
            et, PPI_RHO_DRIFT_ALIGNMENT_TARGET, metric_name, baseline,
            n_mc=align_n_mc, seed=seed,
        )
        calib_rows.append((et, noise, metric_name, PPI_RHO_DRIFT_ALIGNMENT_TARGET,
                           achieved, panel, "gaussian"))

        # Now includes _COMPARISON_METHODS_OMNIBUS. The blocker this comment
        # used to describe -- _run_ppi_comparison_cell populating
        # var_human_subset/var_ppi for two-group structures only, so omnibus
        # rows came back silent NaN -- was removed by adding
        # _classical_point_estimate_omnibus / _ppi_point_estimate_omnibus,
        # which read a matched scalar functional off both arms (see those
        # functions for the per-method estimand and why kruskal differs).
        # _METHOD_CORR_KIND still has no omnibus entries, so rho2_recipe stays
        # NaN for these four and their panels show no dashed recipe line --
        # rho2_evalstats (what the shipped library reports) and rho2_score are
        # plotted for them regardless, which is the comparison that matters.
        methods = (_COMPARISON_METHODS_BINARY if et == "binary"
                   else _COMPARISON_METHODS + _COMPARISON_METHODS_OMNIBUS)
        # Opt-in narrowing. The four omnibus methods carry bootstraps the
        # others don't and dominate the runtime, so a figure that only needs
        # the two-group methods should not pay for them. Default (None) keeps
        # the full set, so --official-tests is unaffected.
        if only_methods:
            _want = tuple(only_methods)
            _unknown = [m for m in _want if m not in methods]
            if _unknown:
                raise ValueError(
                    f"run_ppi_rho_drift_check: unknown method(s) {_unknown} for "
                    f"eval_type={et!r}; available: {list(methods)}")
            methods = tuple(m for m in methods if m in _want)
        # Stable per-method offset: hash() on str is salted per process
        # (PYTHONHASHSEED), which would make this check irreproducible run to
        # run. The SAME offset is used at every effect_frac on purpose --
        # common random numbers across the sweep, so the drift is measured
        # against a shared draw rather than against independent noise.
        # ...and the SAME offset is used for every METHOD too, so that methods
        # sharing an estimand see the same draw.
        #
        # This used to hash the method name, giving each method its own seed.
        # That silently broke the control. ttest and ttest_welch target an
        # IDENTICAL estimand (_classical_point_estimate returns mean(a)-mean(b)
        # for both) through an IDENTICAL PPI call, so their variance ratio is
        # provably the same number -- yet on separate draws they read 2.1554 vs
        # 2.6765, a 24% spread, which propagated to a 16% spread in the implied
        # rho^2. A variance estimated from R replicates carries relative SE
        # ~sqrt(2/R), about 10% at R=200, so that spread is exactly sampling
        # error. With independent seeds the control could not tell "the
        # estimator is wrong" from "we did not run enough replicates", and it
        # failed reproducibly because the per-method seed was deterministic.
        #
        # Sharing one offset makes cross-method comparisons exact at any R:
        # methods that must agree now agree bit-for-bit, and any residual gap
        # is signal rather than draw noise.
        m_offs = {m: 0 for m in methods}
        # Build EVERY (effect_frac, method) cell up front and fan the whole
        # grid out at once, rather than one pool per effect_frac.
        #
        # Two bugs' worth of history here. n_workers used to be accepted and
        # then never used at all, so --workers 15 ran on one core. Fixing that
        # with a pool per effect_frac then hit a straggler problem: kruskal and
        # friedman carry bootstraps the other seven methods don't, so each
        # frac's pool sat blocked on those two while the rest idled --
        # measured 2 of 9 workers busy, i.e. effective parallelism ~2 out of a
        # possible 9. Pooling the full grid lets the slow cells from different
        # fracs overlap each other.
        #
        # Seeding is unaffected: each cell's seed is seed + m_off, a constant,
        # and no RNG object is shared or advanced across iterations -- so this
        # returns bit-identical results to the sequential loop (verified by
        # diffing workers=1 against workers=8).
        specs = []
        for frac in effect_fracs:
            # baseline already carries eval_type/n/label_frac/llm_noise --
            # override those four rather than passing them alongside it.
            kw = {**baseline, "n": PPI_LABEL_EFF_N, "label_frac": label_frac,
                  "llm_noise": noise}
            sc = JudgeBiasSource(
                name=f"rho_drift.{et}.es={frac}", tag="rho_drift",
                effect_size=_jb_effect_magnitude(et, frac), **kw,
            )
            for m in methods:
                specs.append((frac, sc, m))
        cell_args = [(sc, n_reps, n_boot, seed + m_offs[m], m, True)
                     for (_frac, sc, m) in specs]
        if n_workers > 1 and len(cell_args) > 1:
            ctx = _mp.get_context("fork")
            with ctx.Pool(min(n_workers, len(cell_args))) as pool:
                # imap, not map: map returns nothing until the WHOLE grid is
                # done, which at the official rep tiers is hours of blank
                # terminal. imap yields in submission order, so zip(specs, ...)
                # below is still correct, and each completion can be reported.
                cell_results = []
                _t0 = _time.time()
                _total = len(cell_args)
                for _i, _r in enumerate(
                        pool.imap(_run_ppi_comparison_cell_worker, cell_args), 1):
                    cell_results.append(_r)
                    if progress_mode != "off":
                        _el = _time.time() - _t0
                        _eta = _el / _i * (_total - _i)
                        print(f"      [{_i}/{_total}] {_r.method} "
                              f"d={_r.effect_size:g} done "
                              f"({_el/60:.1f} min elapsed, ~{_eta/60:.1f} min left)",
                              flush=True)
        else:
            cell_results = [_run_ppi_comparison_cell_worker(a) for a in cell_args]

        for (frac, sc, method), r in zip(specs, cell_results):
                m_off = m_offs[method]
                mult = (r.var_human_subset / r.var_ppi
                        if np.isfinite(r.var_human_subset) and r.var_ppi > 0
                        else float("nan"))
                frac_unlab = 1.0 - r.n_lab / sc.n if sc.n else float("nan")
                implied = ((1.0 - 1.0 / mult) / frac_unlab
                           if np.isfinite(mult) and mult > 0 and frac_unlab > 0
                           else float("nan"))
                recipe = (_method_rho2(et, noise, method,
                                       shape_label=shape_label)[0]
                          if method in _METHOD_CORR_KIND else float("nan"))
                score = _rho_drift_score_rho2(sc, method, seed + m_off)
                es_rho2 = _rho_drift_evalstats_rho2(sc, method, seed + m_off)
                ne_i = (_ppi_predicted_savings(implied, r.n_lab, sc.n) * r.n_lab
                        if np.isfinite(implied) else float("nan"))
                ne_r = (_ppi_predicted_savings(recipe, r.n_lab, sc.n) * r.n_lab
                        if np.isfinite(recipe) else float("nan"))
                points.append(RhoDriftPoint(
                    eval_type=et, method=method, effect_frac=frac,
                    judge_noise=noise, alignment_value=achieved,
                    n=sc.n, n_lab=r.n_lab, n_reps=n_reps,
                    variance_multiplier=mult, rho2_implied=implied,
                    rho2_recipe=recipe, rho2_score=score, rho2_evalstats=es_rho2,
                    rho2_implied_se=r.rho2_implied_se,
                    n_eff_implied=ne_i, n_eff_recipe=ne_r,
                    n_eff_error=(ne_r / ne_i - 1.0
                                 if np.isfinite(ne_i) and np.isfinite(ne_r) and ne_i > 0
                                 else float("nan")),
                ))
    return points, calib_rows


def print_ppi_rho_drift_report(points: list[RhoDriftPoint]) -> None:
    """Console counterpart of run_ppi_rho_drift_check.

    One block per eval type: rows are methods, columns are effect sizes,
    cells are rho2_implied. A correct effect-invariance assumption shows as a
    flat row; the drift column summarises first-to-last movement. The recipe
    columns follow, since the gap between implied and recipe is the point."""
    if not points:
        print("  (no rho-drift results)")
        return
    for et in sorted({p.eval_type for p in points}):
        sub = [p for p in points if p.eval_type == et]
        fracs = sorted({p.effect_frac for p in sub})
        methods = sorted({p.method for p in sub})
        align = sub[0].alignment_value
        n, n_lab = sub[0].n, sub[0].n_lab
        print(f"\n  eval_type={et}  judge rho^2={align:.3f} (held fixed)  "
              f"N={n}  N_lab={n_lab}  reps={sub[0].n_reps}")
        print(f"    {'method':<12}" + "".join(f"{'d=' + str(f):>9}" for f in fracs)
              + f"{'drift':>9}{'+-MC':>8}{'score@lo':>9}{'score@hi':>9}{'vs score':>10}"
              + f"{'recipe':>9}{'N_eff err':>11}")
        for m in methods:
            row = {p.effect_frac: p for p in sub if p.method == m}
            vals = [row[f].rho2_implied if f in row else float("nan") for f in fracs]
            first, last = vals[0], vals[-1]
            drift = (last / first - 1.0
                     if np.isfinite(first) and np.isfinite(last) and first > 0
                     else float("nan"))
            rec = row[fracs[0]].rho2_recipe if fracs[0] in row else float("nan")
            err = row[fracs[-1]].n_eff_error if fracs[-1] in row else float("nan")
            # MC error on the DRIFT itself, propagated from the two endpoints'
            # bootstrap SEs. Without it a drift number cannot be read: at
            # n_reps=200 the per-cell SE on rho2_implied is ~8% relative, so a
            # -12% drift is barely 1 sigma while a -25% one is ~3. Endpoints
            # are separate cells, so their errors are treated as independent.
            se0 = row[fracs[0]].rho2_implied_se if fracs[0] in row else float("nan")
            se1 = row[fracs[-1]].rho2_implied_se if fracs[-1] in row else float("nan")
            drift_se = float("nan")
            if (np.isfinite(se0) and np.isfinite(se1) and np.isfinite(first)
                    and np.isfinite(last) and first > 0 and last > 0):
                drift_se = abs(last / first) * float(np.hypot(se0 / first, se1 / last))
            sc0 = row[fracs[0]].rho2_score if fracs[0] in row else float("nan")
            sc1 = row[fracs[-1]].rho2_score if fracs[-1] in row else float("nan")
            track = (vals[-1] / sc1 - 1.0
                     if np.isfinite(sc1) and sc1 > 0 and np.isfinite(vals[-1])
                     else float("nan"))
            print(f"    {m:<12}" + "".join(f"{v:>9.4f}" for v in vals)
                  + f"{drift:>+8.1%}"
                  + (f"{drift_se:>8.1%}" if np.isfinite(drift_se) else f"{'--':>8}")
                  + f"{sc0:>9.4f}{sc1:>9.4f}"
                  + (f"{track:>+9.1%}" if np.isfinite(track) else f"{'--':>10}")
                  + (f"{rec:>9.4f}" if np.isfinite(rec) else f"{'--':>9}")
                  + (f"{err:>+10.1%}" if np.isfinite(err) else f"{'--':>11}"))
        print("\n    drift = rho^2 at the largest effect vs the smallest, with the judge")
        print("    unchanged. Flat is the assumption _method_rho2 makes; see")
        print("    _METHOD_CORR_KIND's standing caveat for which methods break it.")
        print("    recipe/N_eff err are blank for methods with no _METHOD_CORR_KIND entry.")
        print("    +-MC is the Monte-Carlo 1 sigma on drift (paired bootstrap over")
        print("    replicates). A drift smaller than ~2x it is not resolved by this run --")
        print("    raise --rho-drift-reps rather than reading it as a finding.")

        # CONTROL. ttest/ttest_welch/paired_t estimate means, whose influence
        # function is linear in the value, so their rho is a plain Pearson
        # correlation that a location shift cannot move -- their invariance is
        # exact algebra, not an empirical regularity. If they drift, the
        # measurement is picking up something other than the influence-function
        # structure it is trying to isolate, and the rank rows cannot be read as
        # that structure either. Surfaced rather than left for a reader to
        # notice, because a silently confounded drift number is worse than none.
        # The control is "tracks rho2_score", NOT "is flat". A mean-type
        # method's influence function is linear in the value, so its rho MUST
        # equal the structure-appropriate SCORE-level correlation -- but that
        # correlation is itself not effect-invariant in a bounded scenario
        # (see RhoDriftPoint.rho2_score). Testing flatness instead reports a
        # scenario-generator property as an estimator failure: paired_t drifts
        # +5.3% while tracking rho2_score to within 1% at every effect.
        ctrl = [m for m in methods if m in (TTEST.name, TTEST_WELCH.name, PAIRED_T.name)]
        # Read each deviation against its OWN Monte-Carlo SE, not against a
        # bare tolerance. A mean-type method's deviation from rho2_score is
        # exact-zero in expectation, so every reading here is noise -- and the
        # noise is not small: a variance from R replicates carries relative SE
        # ~sqrt(2/R), and the "pair" structures are worse still because
        # D = truth_x - truth_y is heavier-tailed than the group scores, so
        # var_human_subset converges more slowly. Same cell, same draws, d=0:
        # paired_t reads -17.6% at R=200, -3.8% at R=600, +0.3% at R=1500,
        # while ttest moves only +4.0% / -3.3% / -0.5%. Flagging the R=200
        # reading as an estimator defect is what this check used to do (see
        # the retired STATUS item 3 below), and it cost a long hunt for a bug
        # that was not there.
        drifts, sigmas = {}, {}
        for m in ctrl:
            row = {p.effect_frac: p for p in sub if p.method == m}
            usable = [row[f] for f in fracs
                      if f in row and np.isfinite(row[f].rho2_score)
                      and row[f].rho2_score > 0 and np.isfinite(row[f].rho2_implied)]
            devs = [abs(pt.rho2_implied / pt.rho2_score - 1.0) for pt in usable]
            # z = deviation in units of its own SE, so "5% at R=200" and "5% at
            # R=2000" are not treated as the same evidence.
            zs = [abs(pt.rho2_implied - pt.rho2_score) / pt.rho2_implied_se
                  for pt in usable
                  if np.isfinite(pt.rho2_implied_se) and pt.rho2_implied_se > 0]
            if devs:
                drifts[m] = max(devs)
            if zs:
                sigmas[m] = max(zs)
        if drifts:
            worst = max(drifts, key=lambda m: abs(drifts[m]))
            mag = abs(drifts[worst])
            # Relative SE of the worst method's deviation, for the report line.
            _rows = {p.effect_frac: p for p in sub if p.method == worst}
            _ses = [p.rho2_implied_se / p.rho2_score for p in _rows.values()
                    if np.isfinite(p.rho2_implied_se) and np.isfinite(p.rho2_score)
                    and p.rho2_score > 0]
            se_rel = float(np.median(_ses)) if _ses else float("nan")
            worst_z = max(sigmas.values()) if sigmas else float("nan")
            # FAIL only when the deviation exceeds BOTH the 5% tolerance and
            # 3 sigma of its own sampling error. Either test alone is wrong:
            # tolerance alone fails on noise at low R, sigma alone fails on a
            # trivially small but well-resolved offset at very high R.
            resolved = np.isfinite(worst_z) and worst_z > 3.0
            failed = mag > 0.05 and resolved
            if failed:
                verdict = "*** CONTROL FAILED ***"
            elif mag > 0.05:
                verdict = f"UNDERPOWERED (within {worst_z:.1f} sigma of 0 -- raise --rho-drift-reps)"
            else:
                verdict = "OK"
            print(f"\n    control (mean-type rho must EQUAL score@d, at every d): "
                  f"worst deviation = {mag:.1%} ({worst})  "
                  f"[MC SE ~{se_rel:.1%}, {worst_z:.1f} sigma]  {verdict}")
            if mag > 0.05 and not failed:
                # Solve sqrt(2/R) scaling for the R that would resolve 5%.
                _n_reps_seen = max((p.n_reps for p in sub), default=0)
                if np.isfinite(se_rel) and se_rel > 0 and _n_reps_seen:
                    need = int(np.ceil(_n_reps_seen * (se_rel / (0.05 / 3.0)) ** 2))
                    print(f"    The deviation is not resolved by the draw: at n_reps={_n_reps_seen} "
                          f"the MC SE is ~{se_rel:.1%},")
                    print(f"    so a real 5% offset could not be told from noise. Re-run with "
                          f"--rho-drift-reps {need}")
                    print("    before reading this as an estimator defect.")
            if failed:
                print("    The mean-type methods' invariance is exact algebra, so a drift this")
                print("    large means the measured numbers include an effect the influence-")
                print("    function account does not predict. Treat every row as confounded.")
                print()
                print("    STATUS (2026-08-21). Three distinct things have been found here;")
                print("    a failure is not necessarily the same failure twice.")
                print()
                print("    1. FIXED (estimator) -- evalstats.ppi._pooled_two_group_lambda")
                print("       pooled the two groups UNCENTERED, dragging lambda toward")
                print("       n_all/(n_all+n_lab) as they separated. It now centres first;")
                print("       ttest/ttest_welch went from -8%/-6% drift to bit-for-bit flat.")
                print()
                print("    2. NOT A BUG (scenario) -- paired_t's rise is REAL judge-quality")
                print("       change. Its estimator is bit-for-bit effect-invariant on an")
                print("       unbounded Gaussian DGP (rho^2 0.6059 at every d), while in the")
                print("       bounded harness scenario Corr(D, Dhat)^2 genuinely rises")
                print("       0.71 -> 0.75. _calibrate_noise_for_alignment pins alignment on")
                print("       the INDEPENDENT-GROUP scores, which does not pin Corr(D, Dhat).")
                print("       Hence the control compares against rho2_score measured at each")
                print("       effect, not against flatness -- flatness reported a scenario")
                print("       property as an estimator failure.")
                print()
                print("    3. RESOLVED (measurement, 2026-08-25) -- the residual LEVEL offset")
                print("       (paired_t ~7-10% below its own rho2_score at EVERY effect,")
                print("       d=0 included) was Monte-Carlo error in var_human_subset, not an")
                print("       estimator defect. It reproduced only at low n_reps: same cell,")
                print("       same draws, d=0, paired_t reads -17.6% at R=200, -3.8% at R=600")
                print("       and +0.3% at R=1500. That is also why the standalone measurement")
                print("       'disagreed' at ~1% -- it simply ran more replicates.")
                print("       The pair structures are hit hardest because")
                print("       D = truth_x - truth_y is heavier-tailed than the group scores,")
                print("       so its sample variance converges more slowly: over the same R")
                print("       sweep ttest moves only +4.0% / -3.3% / -0.5%. Driven directly,")
                print("       _ppi_paired_arrays matches 1/(1-rho^2(1-n_lab/N)) to within")
                print("       +/-0.8% at every effect, and an oracle optimal-lambda PPI on the")
                print("       same draws beats it by only ~1%. Hence rho2_implied_se and the")
                print("       sigma test above: this line can no longer fire on draw noise.")
                print()
                print("    While the control is red, the rank rows measure the SHIPPED")
                print("    estimator's realized multiplier -- a legitimate quantity, but not")
                print("    the influence-function drift the docstring describes.")


def save_ppi_rho_drift_plot(points: list[RhoDriftPoint], out_path: str) -> str:
    """The rho-drift figure: a GRID of small multiples, one panel per method
    (columns) per eval type (rows). Each panel carries exactly two lines --
    the rho^2 the N_eff formula NEEDS (solid, measured) against the rho^2 the
    named recipe RETURNS (dashed, flat by construction for a shift-invariant
    recipe) -- with the gap between them shaded, because that gap IS the error
    a planner suffers.

    Panels are per METHOD rather than the one-panel-per-eval-type layout the
    rest of this module uses (save_ppi_label_efficiency_plot etc.). That
    convention is right when a panel holds a few series; here it would put
    5 methods x 2 lines = 10 series in one axes and bury the only comparison
    that matters. Small multiples keep every panel at two lines, let the
    mean-vs-rank split read straight across the row, and stay legible when
    methods are added.

    Mean-type methods are labelled "(control)": their invariance is exact
    algebra, so their solid line MUST be flat, and the rank panels can only be
    read as influence-function drift once the control panels are. See
    print_ppi_rho_drift_report's control block for the current status."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ets = sorted({p.eval_type for p in points})
    methods = sorted({p.method for p in points})
    ctrl_names = {TTEST.name, TTEST_WELCH.name, PAIRED_T.name}
    nrow, ncol = len(ets), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.85 * nrow),
                             squeeze=False, sharey="row")
    NEED, REC, SCORE, EVAL = "#1B3A5C", "#C1553B", "#8A8F98", "#1E8A6E"
    for r, et in enumerate(ets):
        sub_et = [p for p in points if p.eval_type == et]
        fracs = sorted({p.effect_frac for p in sub_et})
        for c, m in enumerate(methods):
            ax = axes[r][c]
            row = {p.effect_frac: p for p in sub_et if p.method == m}
            if not row:
                ax.set_visible(False)
                continue
            need = [row[f].rho2_implied if f in row else float("nan") for f in fracs]
            rec = next((row[f].rho2_recipe for f in fracs
                        if f in row and np.isfinite(row[f].rho2_recipe)), float("nan"))
            if np.isfinite(rec):
                ax.fill_between(fracs, need, [rec] * len(fracs),
                                color=REC, alpha=0.13, lw=0)
                ax.plot(fracs, [rec] * len(fracs), "--", color=REC, lw=1.6,
                        label=r"harness recipe")
            # rho2_score: the CONTROL reference. A mean-type method's rho MUST
            # equal this (its influence function is linear in the value), and
            # this is NOT flat in the bounded harness scenario -- so "tracks
            # rho2_score", not "is flat", is what the control panels have to
            # be read against. See RhoDriftPoint.rho2_score.
            sco = [row[f].rho2_score if f in row else float("nan") for f in fracs]
            if any(np.isfinite(s) for s in sco):
                ax.plot(fracs, sco, ":", color=SCORE, lw=1.5, label=r"score-level $\rho^2$")
            # rho2_evalstats: what the SHIPPED library reports for this method.
            ev = [row[f].rho2_evalstats if f in row else float("nan") for f in fracs]
            if any(np.isfinite(e) for e in ev):
                ax.plot(fracs, ev, "-s", color=EVAL, lw=1.5, ms=3.0, alpha=0.9,
                        label=r"evalstats reports")
            ax.plot(fracs, need, "-o", color=NEED, lw=2.0, ms=3.4,
                    label=r"formula needs")
            is_ctrl = m in ctrl_names
            # NOT "must be flat": rho2_score is itself not flat in this
            # bounded scenario, and a mean-type method's rho must equal
            # rho2_score, not a constant. See RhoDriftPoint.rho2_score --
            # paired_t tracks it within 1% while failing flatness by +5.3%.
            ax.set_title(m + ("\n(control — must track score)" if is_ctrl else ""),
                         fontsize=8.5, color="#5A6570" if is_ctrl else "#14181C")
            ax.grid(axis="y", color="#E3E6E4", lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5)
            # rho^2 is bounded [0, 1] by definition, so clamp the view there.
            # Without this, ONE bad cell destroys every panel: the axes are
            # sharey="row", and rho2_implied is a ratio of two measured
            # variances that blows up (seen at -494) when reps are too few for
            # the denominator to be stable. Points outside the domain are a
            # measurement failure, not a finding -- so clip them, but SAY the
            # panel is clipped rather than silently dropping them off-screen.
            _off = sum(1 for v in need + sco + ev
                       if np.isfinite(v) and not (-0.02 <= v <= 1.02))
            ax.set_ylim(-0.02, 1.02)
            if _off:
                ax.text(0.98, 0.03, f"{_off} off-scale", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color=REC)
            if r == nrow - 1:
                ax.set_xlabel("effect size $d$", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{et}\n" + r"$\rho^2$", fontsize=8.5)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
    align = points[0].alignment_value
    fig.suptitle(r"Judge quality held fixed ($\rho^2$ = "
                 f"{align:.2f}) in every panel — only the true effect changes",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_results_artifacts_ppi_rho_drift(
    points: list[RhoDriftPoint], out_dir: str, run_stem: str,
) -> list[str]:
    """CSV + summary log for run_ppi_rho_drift_check, mirroring
    save_results_artifacts_ppi_nformula's shape (its own writer rather than a
    branch inside the label-efficiency one -- different row type, different
    columns)."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    csv_path = out_base / f"{run_stem}_ppi_rho_drift_results.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eval_type", "method", "effect_frac", "judge_noise",
                    "alignment_value", "n", "n_lab", "n_reps",
                    "variance_multiplier", "rho2_implied", "rho2_implied_se",
                    "rho2_recipe", "rho2_score",
                    "rho2_evalstats",
                    "n_eff_implied", "n_eff_recipe", "n_eff_error"])
        for p in points:
            w.writerow([p.eval_type, p.method, f"{p.effect_frac}", repr(p.judge_noise),
                        repr(p.alignment_value), p.n, p.n_lab, p.n_reps,
                        repr(p.variance_multiplier), repr(p.rho2_implied),
                        repr(p.rho2_implied_se),
                        repr(p.rho2_recipe), repr(p.rho2_score), repr(p.rho2_evalstats),
                        repr(p.n_eff_implied),
                        repr(p.n_eff_recipe), repr(p.n_eff_error)])
    written.append(str(csv_path))

    summary_path = out_base / f"{run_stem}_ppi_rho_drift_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_rho_drift_report(points)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    written.append(str(summary_path))
    for path in written:
        print(f"Saved results: {path}")
    return written


def print_ppi_nformula_report(results: list[LabelEfficiencyPoint]) -> None:
    """Console/log counterpart of run_ppi_nformula_check, analogous to
    print_ppi_label_efficiency_report -- but grouped by eval_type, THEN
    effect_frac, THEN N, then alignment_target, since (unlike the original
    label-efficiency check) neither N nor effect_frac is constant across
    `results` here. Reusing print_ppi_label_efficiency_report directly
    would be wrong: its per-target grouping silently assumes every row for
    a given (eval_type, target) shares the same N/effect_frac, which this
    sweep's rows don't."""
    if not results:
        print("\n  (no n-formula results)")
        return
    print(
        f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- LABEL EFFICIENCY N-FORMULA CHECK\n"
        f"  N, N_lab, and effect_frac all vary -- see PPI_NFORMULA_N_VALUES/"
        f"PPI_NFORMULA_NLAB_VALUES/PPI_NFORMULA_EFFECT_FRACS\n{'='*88}"
    )
    for et in sorted({r.eval_type for r in results}):
        print(f"\n  [{et}]")
        for frac in sorted({r.effect_frac for r in results if r.eval_type == et}):
            print(f"    effect_frac={frac:.3f}")
            for n in sorted({r.n for r in results if r.eval_type == et and r.effect_frac == frac}):
                print(f"      N={n}")
                for target in sorted(
                    {r.alignment_target for r in results if r.eval_type == et and r.effect_frac == frac and r.n == n},
                    reverse=True,
                ):
                    rows = sorted(
                        (r for r in results if r.eval_type == et and r.effect_frac == frac and r.n == n and r.alignment_target == target),
                        key=lambda r: r.n_lab,
                    )
                    metric = rows[0].alignment_metric
                    achieved_vals = {r.alignment_value for r in rows}
                    achieved_str = f"{sum(achieved_vals) / len(achieved_vals):.3f}" if achieved_vals else "n/a"
                    print(f"        target {metric}={target:.2f}  (achieved ~{achieved_str})")
                    print(f"          {'N_lab':>8} {'ppi_power':>10} {'equiv_N_lab':>12} {'multiplier':>11}")
                    for r in rows:
                        mult = r.equiv_n_lab / r.n_lab if r.n_lab else float("nan")
                        flag = "  (saturated, lower bound)" if r.saturated else ""
                        print(f"          {r.n_lab:>8} {r.ppi_power:>10.3f} {r.equiv_n_lab:>12.1f} {mult:>10.2f}x{flag}")
    print()


def save_results_artifacts_ppi_label_efficiency(
    *, results: list[LabelEfficiencyPoint], out_dir: str, run_stem: str,
) -> list[str]:
    """Write the label-efficiency run's results CSV under out_dir. Returns
    the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_label_efficiency_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            # effect_frac FIRST among the new columns: the sweep spans
            # PPI_LABEL_EFF_EFFECT_FRACS, so without it the arms are not
            # separable after the fact and the es-invariance check (which is
            # the point of sweeping) cannot be reproduced from the CSV.
            # noise_family right after eval_type: together they are the
            # grouping key every downstream analysis needs, and without it the
            # two judge-error-shape arms are indistinguishable in this file
            # except by cross-referencing judge_noise against the calibration
            # CSV (they calibrate to DIFFERENT llm_noise for the same tier).
            "eval_type", "noise_family", "effect_frac", "alignment_metric",
            "alignment_target", "alignment_value",
            "judge_noise", "n_lab", "n_reps", "ppi_power", "equiv_n_lab", "multiplier",
            "multiplier_lo", "multiplier_hi", "saturated",
            "rho2", "predicted_mult", "predicted_mult_asymptotic",
            "inversion_ratio", "inversion_clamped", "well_conditioned",
        ])
        for r in results:
            mult = r.equiv_n_lab / r.n_lab if r.n_lab else float("nan")
            writer.writerow([
                r.eval_type, r.noise_family, f"{r.effect_frac:.2f}", r.alignment_metric,
                f"{r.alignment_target:.2f}", f"{r.alignment_value:.4f}",
                f"{r.judge_noise:.4f}", r.n_lab, r.n_reps,
                f"{r.ppi_power:.6f}", f"{r.equiv_n_lab:.4f}", f"{mult:.4f}",
                f"{r.mult_lo:.4f}", f"{r.mult_hi:.4f}", r.saturated,
                f"{r.rho2:.4f}", f"{r.predicted_mult:.4f}", f"{r.predicted_mult_asymptotic:.4f}",
                f"{r.inversion_ratio:.4f}", r.inversion_clamped, r.well_conditioned,
            ])
    summary_path = out_base / f"{run_stem}_ppi_label_efficiency_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_label_efficiency_report(results)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_results_artifacts_ppi_nformula(
    *, results: list[LabelEfficiencyPoint], out_dir: str, run_stem: str,
) -> list[str]:
    """Pooled-CSV counterpart of save_results_artifacts_ppi_label_efficiency
    for run_ppi_nformula_check's output -- a SEPARATE function (not an
    extension of that one) so the original label-efficiency CSV's column
    set/writer stays untouched. Adds `n` and `effect_frac` columns (both
    fixed constants in the original sweep's CSV, so not worth adding
    there) since this sweep's whole point is that they vary. Use save_
    results_artifacts_ppi_label_efficiency_raw (unchanged, reused as-is)
    for the accompanying raw per-method/calibration CSVs -- PPIComparisonResult
    already carries `n`/`effect_size` per row, so nothing needed adding
    there."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_nformula_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "eval_type", "n", "effect_frac", "alignment_metric", "alignment_target", "alignment_value",
            "judge_noise", "n_lab", "n_reps", "ppi_power", "equiv_n_lab", "multiplier", "saturated",
        ])
        for r in results:
            mult = r.equiv_n_lab / r.n_lab if r.n_lab else float("nan")
            writer.writerow([
                r.eval_type, r.n, f"{r.effect_frac:.4f}", r.alignment_metric, f"{r.alignment_target:.2f}",
                f"{r.alignment_value:.4f}", f"{r.judge_noise:.4f}", r.n_lab, r.n_reps,
                f"{r.ppi_power:.6f}", f"{r.equiv_n_lab:.4f}", f"{mult:.4f}", r.saturated,
            ])
    summary_path = out_base / f"{run_stem}_ppi_nformula_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_nformula_report(results)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_results_artifacts_ppi_label_efficiency_raw(
    *, raw: list[PPIComparisonResult], calib_rows: list[tuple[str, float, str, float, float, dict]],
    out_dir: str, run_stem: str,
) -> list[str]:
    """Persists the RAW, per-method data run_ppi_label_efficiency_check
    computes but the pooled LabelEfficiencyPoint/save_results_artifacts_
    ppi_label_efficiency path discards -- this sweep is expensive (the
    alignment calibration alone runs align_n_mc=20,000 MC draws per target
    per eval type, on top of the comparison simulation itself), so "is one
    method dragging the pooled average down for eval type X" should be
    answerable from a saved CSV, not require re-running the whole check.

    effect_size is written at FULL PRECISION (repr), not rounded. It used to be
    formatted %.4f, which silently truncated e.g. 0.03015113445777636 to
    0.0302 -- enough to make any analysis that reads it back disagree with the
    sweep, and in particular to miss every reference-curve cache entry (those
    keys are built from the exact effect size), so a reader reconstructing
    results from this file would quietly rebuild every curve from scratch.

    Two CSVs: one row per (scenario, method) cell (same column shape as
    save_results_artifacts_ppi_comparison's raw CSV, for consistency with
    the other comparison-sweep raw exports elsewhere in this file), and a
    small calibration-lookup CSV mapping each embedded noise value (see
    PPIComparisonResult.name, e.g. "labeleff.continuous.noise=0.0909....")
    back to the alignment target/metric/achieved value it was calibrated
    to hit -- without this, the raw CSV's noise column is just a number,
    not "the noise level that hits weighted_kappa~=0.8"."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    raw_path = out_base / f"{run_stem}_ppi_label_efficiency_raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            # effect_frac is parsed back out of the scenario name (which
            # embeds ".es=<frac>") so the per-method rows stay separable by
            # sweep arm without having to re-derive it from effect_size,
            # whose absolute value differs per eval type.
            "name", "tag", "eval_type", "noise_family", "effect_frac", "method", "n", "n_reps",
            "effect_size", "label_frac", "n_lab", "var_human_subset", "var_ppi",
            "n_est", "variance_multiplier",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in raw:
            _m_es = re.search(r"\.es=([\d.]+)", r.name)
            # Same treatment as effect_frac above: recovered from the scenario
            # name rather than left implicit, so the arms stay separable
            # without every consumer having to re-parse the name themselves.
            _m_fam = re.search(r"\.fam=([a-z]+)\.", r.name)
            writer.writerow([
                r.name, r.tag, r.eval_type, (_m_fam.group(1) if _m_fam else "gaussian"),
                (_m_es.group(1) if _m_es else ""),
                r.method, r.n, r.n_reps, repr(float(r.effect_size)), f"{r.label_frac:.4f}", r.n_lab,
                repr(float(r.var_human_subset)), repr(float(r.var_ppi)), r.n_est,
                (repr(float(r.var_human_subset / r.var_ppi))
                 if getattr(r, "var_ppi", 0) and np.isfinite(r.var_ppi) else ""),
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    calib_path = out_base / f"{run_stem}_ppi_label_efficiency_calibration.csv"
    with calib_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["eval_type", "noise_family", "judge_noise", "alignment_metric",
             "alignment_target", "alignment_achieved"]
            + list(_CALIB_EXTRA_METRIC_COLUMNS)
        )
        for row in calib_rows:
            # run_ppi_label_efficiency_check appends a 7th element (the judge
            # noise_family it calibrated under); run_ppi_nformula_check does
            # not, and has no family axis. Default rather than require, so the
            # two callers can share this writer.
            et, noise, metric_name, target, achieved, panel = row[:6]
            fam = row[6] if len(row) > 6 else "gaussian"
            extra = []
            for col in _CALIB_EXTRA_METRIC_COLUMNS:
                v = panel.get(col)
                extra.append("" if v is None or not np.isfinite(v) else f"{float(v):.4f}")
            writer.writerow(
                [et, fam, f"{noise:.4f}", metric_name, f"{target:.2f}", f"{achieved:.4f}"] + extra
            )
    print(f"Saved results: {raw_path}")
    print(f"Saved results: {calib_path}")
    return [str(raw_path), str(calib_path)]


_CALIB_EXTRA_METRIC_COLUMNS = (
    "rho2",
    "pearson_r",
    "percent_agreement",
    "kappa",
    "weighted_kappa",
    "linear_weighted_kappa",
    "gwet_ac1",
    "pabak",
    "krippendorff_alpha",
    "spearman_r",
    "kendall_tau_b",
    "icc_21",
    "lin_ccc",
)
"""Extra inter-rater-reliability columns written to the label-efficiency
CALIBRATION csv (not the results csv): the full alignment panel each judge
tier actually realized at the llm_noise chosen to hit its nominal target on
the ONE primary metric (_LABEL_EFF_ALIGNMENT_METRIC -- kappa for binary,
weighted kappa for likert, Pearson r for continuous).

The point is to make the sweep's central claim falsifiable without a re-run.
The label-efficiency result is stated as a threshold in judge-human "IRR"
(see save_ppi_label_efficiency_threshold_plot), and the obvious reviewer
objection is that "IRR" there means a DIFFERENT statistic for each eval type,
so the apparent cross-type agreement could be an artifact of that choice.
With these columns a reader can re-read every tier under a common statistic
-- Krippendorff's alpha in particular is defined for all three types with
only its distance function changing (see scenarios/synthetic._alignment_
metric_dict) -- and check whether the tiers still line up.

The union of all eval types' panels; each row leaves blank whatever its own
type doesn't define (the chance-corrected categorical metrics need
categories, so continuous has no kappa/AC1/PABAK). Deliberately NOT added to
the per-cell results csv: these are properties of the calibrated JUDGE, fixed
within an (eval_type, target) tier, so repeating them on every method x
n_lab x es row would be pure duplication."""


_LABEL_EFF_MARKER_SHAPES = ("o", "s", "D", "P", "X", "*")
"""Per-alignment-target marker shapes for save_ppi_label_efficiency_plot,
cycled by index alongside (not instead of) the viridis color ramp -- a
colorblind/grayscale-print accessibility aid so lines stay distinguishable
by shape even where two adjacent targets' colors read as similar. "^"
(up-triangle) is deliberately excluded: it's reserved for the separate
"saturated" lower-bound overlay marker, and reusing it as a target's own
line marker would make that overlay ambiguous with the line's normal
markers at the same point. "*" renders visually smaller than the other
glyphs at equal markersize, hence _LABEL_EFF_MARKER_SIZE's per-shape bump."""
_LABEL_EFF_MARKER_SIZE = {"*": 9, "P": 6, "X": 6}
"""markersize overrides for _LABEL_EFF_MARKER_SHAPES entries that render
smaller/larger than "o" at the same nominal size; anything not listed here
falls back to the default markersize passed at the call site."""


_ANALYTIC_PLOT_SEED = 0
"""Fixed seed for the bootstrap CIs drawn inside plotting helpers, so a
re-render of the same results produces an identical figure."""


_LABEL_EFF_PANEL_TITLES = {
    "binary": "Binary",
    "continuous": "Continuous",
    "likert": "Likert",
}
"""Panel titles -- just the eval type now. They previously named each panel's
own alignment statistic ("Binary (Cohen's kappa)", "Continuous (Pearson r)",
"Likert (weighted kappa)") because the axis genuinely differed per panel and a
reader comparing them needed to know the numbers were not commensurable. Every
eval type is now calibrated on the SAME statistic, rho^2 (see
_LABEL_EFF_ALIGNMENT_METRIC), so naming a per-type metric here would assert a
difference that no longer exists -- and name the wrong statistic besides. The
shared axis label carries what the number is."""


def save_ppi_label_efficiency_invariance_plot(
    results: list[LabelEfficiencyPoint], out_path: str,
) -> str:
    """Effect-size INVARIANCE figure (appendix): multiplier on y, effect size
    on x, one line per rho^2 tier, one panel per eval type. See
    save_ppi_label_efficiency_invariance_pooled_plot for the pooled companion
    and why the two carry different claims.

    The claim this figure has to make is "the label-efficiency multiplier is a
    property of the JUDGE, not of the effect you happen to be testing", and
    the visual encoding is chosen so that claim needs no statistical setup
    from the reader: **flat lines mean invariance**. A reader who knows
    nothing about the reference-curve inversion can see the result.

    Why this figure exists at all: the multiplier is obtained by inverting a
    classical power curve, so it COULD in principle drift with effect size
    (the inversion is better conditioned in the curve's steep middle -- see
    PPI_LABEL_EFF_EFFECT_FRACS). Sweeping several effect sizes and showing
    the multiplier does not move is what licenses reporting a single pooled
    number in the main text.

    Medians across the N_lab grid, IQR/2 bars. Saturated points are dropped
    (their equiv_n_lab is clamped -- see LabelEfficiencyPoint.saturated)."""
    import matplotlib.pyplot as plt

    rows = [r for r in results if not r.saturated and r.well_conditioned and np.isfinite(r.equiv_n_lab)]
    if not rows:
        raise ValueError("No non-saturated label-efficiency results to plot.")
    eval_types = [et for et in ("binary", "continuous", "likert") if any(r.eval_type == et for r in rows)]
    tiers = sorted({r.alignment_target for r in rows})
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, len(eval_types), figsize=(4.4 * len(eval_types), 4.3), sharey=True,
                             squeeze=False)
    for ax, et in zip(axes[0], eval_types):
        fracs = sorted({r.effect_frac for r in rows if r.eval_type == et})
        for i, t in enumerate(tiers):
            med, err = [], []
            for ef in fracs:
                v = [r.equiv_n_lab / r.n_lab for r in rows
                     if r.eval_type == et and r.alignment_target == t and r.effect_frac == ef]
                med.append(float(np.median(v)) if v else np.nan)
                err.append(float(np.percentile(v, 75) - np.percentile(v, 25)) / 2 if len(v) > 2 else 0.0)
            ax.errorbar(fracs, med, yerr=err, marker="o", ms=4, lw=1.6, capsize=2.5,
                        color=cmap(i / max(len(tiers) - 1, 1)), label=f"{t:.1f}")
        ax.axhline(1.0, color="crimson", ls="--", lw=1.2, zorder=0)
        ax.set_title(_LABEL_EFF_PANEL_TITLES.get(et, et), fontsize=10)
        ax.set_xlabel("effect size (fraction of population SD)")
        ax.set_xticks(fracs)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("label-efficiency multiplier\n(equivalent human labels / actual labels)")
    axes[0][0].legend(title="judge–human\nagreement  ρ²", fontsize=8, title_fontsize=8,
                      loc="upper left", ncol=2)
    if _LABEL_EFF_FIGURE_TITLES:
        fig.suptitle("Label-efficiency multiplier is invariant to effect size (flat lines = invariance)",
                 fontsize=11, y=1.0)
    if _LABEL_EFF_FIGURE_TITLES:
        fig.text(0.5, -0.03, "Each line is one judge-quality tier; points are medians across the "
                 "$N_{lab}$ grid, bars are IQR/2. Saturated cells excluded.", ha="center", fontsize=8.5)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_label_efficiency_invariance_pooled_plot(
    results: list[LabelEfficiencyPoint], out_path: str,
) -> str:
    """Companion to save_ppi_label_efficiency_invariance_plot: the SAME
    effect-size invariance claim, but with all three eval types POOLED into
    one panel, one line per rho^2 tier.

    The per-eval-type version answers "does the multiplier drift with effect
    size?". This one additionally answers "do the three eval types agree at
    matched judge quality?" -- and it is only a legitimate figure to draw
    because the judge-quality axis is now rho^2 for every eval type (see
    _LABEL_EFF_ALIGNMENT_METRIC). Under the previous per-type metrics, a tier
    meant kappa=0.6 for binary and Pearson r=0.6 for continuous, which realize
    different rho^2, so pooling them would have averaged judges of genuinely
    different quality and the spread bars would have been meaningless.

    So the two figures carry different weight: flat lines here mean the
    multiplier depends on neither the effect size NOR the data type, only on
    rho^2 -- which is the single-number rule of thumb's whole premise. The
    error bars are the spread ACROSS eval types and the N_lab grid combined,
    so a tight bar is itself the cross-type agreement evidence rather than
    something a reader has to take on faith from a separate table.

    Keep both: this one is the headline, the per-type panels are what a
    reviewer asks for when they want to check the pooling was not hiding one
    badly-behaved arm.

    Medians with IQR/2 bars; saturated points dropped."""
    import matplotlib.pyplot as plt

    rows = [r for r in results if not r.saturated and r.well_conditioned and np.isfinite(r.equiv_n_lab) and r.n_lab]
    if not rows:
        raise ValueError("No non-saturated label-efficiency results to plot.")
    tiers = sorted({r.alignment_target for r in rows})
    fracs = sorted({r.effect_frac for r in rows})
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, t in enumerate(tiers):
        med, err = [], []
        for ef in fracs:
            v = [r.equiv_n_lab / r.n_lab for r in rows
                 if r.alignment_target == t and r.effect_frac == ef]
            med.append(float(np.median(v)) if v else np.nan)
            err.append(float(np.percentile(v, 75) - np.percentile(v, 25)) / 2 if len(v) > 2 else 0.0)
        ax.errorbar(fracs, med, yerr=err, marker="o", ms=5, lw=1.8, capsize=3,
                    color=cmap(i / max(len(tiers) - 1, 1)), label=f"{t:g}")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, zorder=0)
    ax.set_xlabel("effect size (fraction of population SD)")
    ax.set_ylabel("label-efficiency multiplier\n(equivalent human labels / actual labels)")
    ax.set_xticks(fracs)
    ax.grid(alpha=0.25)
    ax.legend(title="judge–human\nagreement  ρ²", fontsize=8.5, title_fontsize=8.5,
              loc="upper left", ncol=2)
    if _LABEL_EFF_FIGURE_TITLES:
        ax.set_title("Label efficiency depends on ρ² alone — not on effect size or data type",
                 fontsize=11)
    if _LABEL_EFF_FIGURE_TITLES:
        fig.text(0.5, -0.04, "All three eval types pooled, one line per ρ² tier. Points are medians, "
                 "bars are IQR/2 across\nboth eval types and the $N_{lab}$ grid — so a tight bar IS the "
                 "cross-type agreement. Saturated cells excluded.", ha="center", fontsize=8.5)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_label_efficiency_threshold_plot(
    results: list[LabelEfficiencyPoint], out_path: str, n_boot: int = 3000,
    corr_kind: str = "pearson",
) -> str:
    """"How good must the judge be?" figure: multiplier vs judge-human
    agreement, with the practically-useless region shaded.

    ONE TEST FAMILY PER FIGURE, and each plotted against ITS OWN correlation.
    `corr_kind` is "pearson" (mean-based tests), "spearman" (rank-based), or
    "mixed" -- every method pooled, each contributing its OWN correlation, so
    the x-axis is "whichever rho^2 governs your test". The per-family figures
    are the honest ones to act on; "mixed" is the single-number summary for a
    reader who has not yet chosen a test.

    This split is not cosmetic. The x-axis is the number a practitioner
    measures on a pilot set and looks up, so it has to be the number that
    actually governs THEIR test. A pooled figure labelled "squared Pearson
    correlation" whose y-axis averaged rank tests in with mean tests told a
    Wilcoxon user to read their threshold off the wrong statistic -- and the
    two differ substantially: at a judge whose score-level Pearson rho^2 is
    0.50, difference-level Spearman rho^2 ranges 0.47-0.61 depending on the
    shape of the judge's errors (see notes/WHICH_RHO_FOR_WHICH_TEST.md).

    x positions come from each cell's MEASURED rho^2 for that method, not from
    the calibration tier. The tiers are defined by score-level Pearson, which
    is the right x only for the group-structure mean tests; everything else
    sits somewhere else on the axis, and drawing it at the tier would put the
    point at a coordinate the practitioner would never measure.

    This is the figure a practitioner actually acts on -- it answers "is my
    judge good enough to be worth wiring up?" in the unit they care about
    (labels, hence money), not in power or p-values.

    The shaded band below 1.25x is deliberate. A multiplier can be
    STATISTICALLY above 1.0 while being practically pointless: at agreement
    0.4 the measured medians are 1.14x/1.01x/1.03x (binary/continuous/likert)
    -- a 1-14% label saving that no one would restructure a pipeline for. So
    the figure marks "distinguishable from 1.0" and "worth the trouble" as
    different thresholds, rather than letting a significance test stand in
    for a practical one.

    Bands are bootstrap CIs on the median, pooled across effect-size arms
    (licensed by save_ppi_label_efficiency_invariance_plot's result)."""
    import matplotlib.pyplot as plt

    rows = [r for r in results if not r.saturated and r.well_conditioned and np.isfinite(r.equiv_n_lab)]
    if not rows:
        raise ValueError("No non-saturated label-efficiency results to plot.")
    eval_types = [et for et in ("binary", "continuous", "likert") if any(r.eval_type == et for r in rows)]
    tiers = sorted({r.alignment_target for r in rows})
    # Tier -> the rho^2 a practitioner would actually MEASURE for this family,
    # averaged over the noise families present. Pooling the families here
    # matches the main-text figure's convention (see
    # save_ppi_label_efficiency_plots): the reported number is expected over
    # judge-error shapes rather than conditioned on one.
    # PER EVAL TYPE, not pooled. The same calibration tier realizes as very
    # different rho^2 across eval types -- at tier 0.50 the parametric figure
    # has binary at 0.476, continuous at 0.518 and likert at 0.379, a spread of
    # 0.139. Drawing all three at the pooled mean put likert's curve ~0.08 to
    # the RIGHT of where a likert user would measure their own judge, which on
    # a look-up figure is the error that actually misleads someone.
    _x_of = {}
    for t in tiers:
        for et in eval_types:
            v = [r.rho2 for r in rows
                 if r.alignment_target == t and r.eval_type == et and np.isfinite(r.rho2)]
            _x_of[(t, et)] = float(np.mean(v)) if v else float("nan")
    xs_plot = [v for v in _x_of.values() if np.isfinite(v)]
    if not xs_plot:
        raise ValueError("No finite rho^2 to place points on.")
    marks = {"binary": "o", "continuous": "s", "likert": "^"}
    cols = {"binary": "#2166ac", "continuous": "#1a9850", "likert": "#b2182b"}
    rng = np.random.default_rng(_ANALYTIC_PLOT_SEED)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    _xs_by_et: dict = {}
    ymax = 1.0
    for et in eval_types:
        med, lo, hi = [], [], []
        for t in tiers:
            v = np.array([r.equiv_n_lab / r.n_lab for r in rows
                          if r.eval_type == et and r.alignment_target == t])
            if not len(v):
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
            med.append(float(np.median(v)))
            lo.append(float(np.percentile(b, 2.5))); hi.append(float(np.percentile(b, 97.5)))
        ymax = max(ymax, float(np.nanmax(hi)))
        _xe = [_x_of[(t, et)] for t in tiers]
        _keep = [i for i, (x, m) in enumerate(zip(_xe, med)) if np.isfinite(x) and np.isfinite(m)]
        _xk = [_xe[i] for i in _keep]
        _mk = [med[i] for i in _keep]
        _xs_by_et[et] = (_xk, _mk)
        ax.plot(_xk, _mk, marker=marks.get(et, "o"), color=cols.get(et), lw=2, ms=6,
                label=_LABEL_EFF_PANEL_TITLES.get(et, et), zorder=3)
        ax.fill_between(_xk, [lo[i] for i in _keep], [hi[i] for i in _keep],
                        color=cols.get(et), alpha=0.18, zorder=2)

    ax.axhspan(0.95, 1.25, color="grey", alpha=0.16, zorder=0)
    # Label the shaded band in its EMPTY right half: every eval type has
    # climbed above 1.25x by the upper agreement tiers, so the band is clear
    # there, whereas the left half is exactly where the low-agreement points
    # sit and any label collides with them.
    ax.text(xs_plot[-1], 1.10, "not worth the trouble\n(<1.25× saving)  ",
            fontsize=8.5, color="#444", va="center", ha="right")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.3, zorder=1)

    # Markers sit at ROUND rho^2 values with the MEASURED multiplier read off
    # them -- deliberately not the other way around.
    #
    # An earlier version interpolated the curve against round multiplier
    # levels, which put the lines at rho^2 = 0.42 and 0.52. Statistically
    # fine, useless as a rule of thumb: the reader computes rho^2 and looks
    # up the consequence, so the MEMORABLE number has to be on the rho^2 axis.
    # "Below 0.4, not worth the trouble" is something someone repeats from
    # memory; "below 0.42" is not. (Hardcoding both numbers, the version before that,
    # went stale twice -- hence reading the multiplier from the data here.)
    pooled = {}
    for t in tiers:
        v = [r.equiv_n_lab / r.n_lab for r in rows if r.alignment_target == t and r.n_lab]
        if v:
            pooled[t] = float(np.median(v))
    WORTH_IT = 1.25  # matches the shaded band below
    # Annotation lines sit on ROUND rho^2 values, always starting at 0.20, with
    # the multiplier INTERPOLATED from the measured curve there.
    #
    # The measured tiers land off-round on this axis (a tier calibrated to
    # score-level Pearson 0.20 realizes as Spearman 0.18 for rank tests), and a
    # rule of thumb quoted as "below 0.18" is neither memorable nor honest about
    # its own precision. Snapping the LINES to round values while reading the
    # multiplier off the curve keeps the number quotable and still measured --
    # what moves is where we sample the curve, not what the curve says.
    def _at(x):
        """Median across eval types of their own curves at rho^2 = x.

        The quoted multiplier is the typical one; the MARKER position is set by
        _all_clear, i.e. the worst eval type. Those answer different questions
        -- "what will I get" versus "is it worth it for everyone" -- and the
        figure states both."""
        v = [float(np.interp(x, np.array(_xs_by_et[et][0]), np.array(_xs_by_et[et][1])))
             for et in _xs_by_et if len(_xs_by_et[et][0]) >= 2]
        return float(np.median(v)) if v else float("nan")
    # Round up so the grid COVERS the data: the rank panel's top tier realizes
    # at 0.67, and stopping at the last round value below it left the axis
    # ending at 0.6 with a visible stub of curve past the final gridline.
    _hi_round = float(np.ceil(max(xs_plot) * 10 - 1e-9) / 10)
    _rounds = [float(v) for v in np.round(np.arange(0.2, _hi_round + 1e-9, 0.1), 2)]
    # Annotations may only sit where the curve was MEASURED -- np.interp clamps
    # past the last point, so quoting a multiplier at 0.7 when the data stops at
    # 0.67 would silently reprint the 0.67 value under a rounder label.
    # ... and only where EVERY curve was measured, not merely one of them.
    # max(xs_plot) is the global max across eval types, so it let an annotation
    # sit past the end of the shorter curves -- np.interp clamps there, and
    # _at()/_all_clear() would then silently reuse a curve's last measured value
    # under a rounder label. That is the very failure the comment above warns
    # about, applied across eval types rather than within one: with binary
    # reaching rho^2 0.830 but continuous 0.763 and likert 0.739, the top marker
    # landed on 0.8, where two of the three curves are clamped and the quoted
    # "pooled" multiplier is really binary's alone.
    _meas_hi = min((max(_xs_by_et[et][0]) for et in _xs_by_et
                    if len(_xs_by_et[et][0]) >= 2), default=max(xs_plot))
    _rounds_meas = [v for v in _rounds if v <= _meas_hi + 1e-9]

    # Leftmost annotated line is always 0.20 -- the anchor the rule of thumb is
    # quoted against, whether or not the curve happens to cross 1.25x there.
    cut = _rounds[0] if _rounds else None
    if cut is not None:
        ax.axvline(cut, color="k", ls=":", lw=1.4, zorder=1)
        txt = f"ρ² < {cut:g}: judges not\nworth the trouble\n({_at(cut):.2f}× at {cut:g})"
        # Always to the RIGHT of the line, above the curves in its own x-span.
        #
        # The old rule chose left-or-right by comparing `cut` to min(xs_plot),
        # which broke once `cut` was pinned to 0.20: a panel whose lowest
        # measured point sits below 0.20 (parametric starts at 0.171) failed the
        # "near the left edge" test and got pushed LEFT into a gap ~0.06 wide,
        # where the text ran off the axis and through the y-label. There is
        # never meaningful room left of 0.20, because the axis starts there.
        _scan_c = np.linspace(cut, min(cut + 0.25, max(xs_plot)), 24)
        _under = [float(np.max(np.interp(_scan_c, np.array(_xs_by_et[et][0]),
                                         np.array(_xs_by_et[et][1]))))
                  for et in _xs_by_et if len(_xs_by_et[et][0]) >= 2]
        _y_cut = (max(_under) + 0.08 * (ymax - WORTH_IT)) if _under else WORTH_IT
        ax.text(cut + 0.012, min(_y_cut, ymax * 0.97), txt,
                fontsize=9, va="center", ha="left", color="#333")

    # Pay-off marker: the cheapest ROUND rho^2 whose interpolated multiplier
    # clears WORTH_IT. The reader wants a round number to aim at, not the exact
    # crossing point.
    # The pay-off marker requires EVERY eval type to clear WORTH_IT there, not
    # just the pooled median. A median can clear 1.25x while the weakest data
    # type is still at 1.16x, which would print a threshold that does not hold
    # for the reader who happens to have Likert data. "Worth it whatever your
    # data looks like" is the claim a rule of thumb should make.
    def _all_clear(x):
        vals = [float(np.interp(x, np.array(_xs_by_et[et][0]), np.array(_xs_by_et[et][1])))
                for et in _xs_by_et if len(_xs_by_et[et][0]) >= 2]
        return bool(vals) and min(vals) >= WORTH_IT
    _floor = _LABEL_EFF_PAYOFF_FLOOR
    past = [x for x in _rounds_meas
            if x > cut + 1e-9 and _all_clear(x)
            and (_floor is None or x >= _floor - 1e-9)] if cut is not None else []
    if past:
        g = min(past)
        ax.axvline(g, color="k", ls=":", lw=1.2, zorder=1)
        ax.text(g + 0.012, WORTH_IT + 0.72 * (ymax - WORTH_IT),
                f"ρ² ≈ {g:g}: PPI starts\nto pay for itself ({_at(g):.2f}×)",
                fontsize=9, va="center", color="#333")
        # Top of the ladder, for the "and if my judge is good?" reader. Only
        # when it is a distinct round value from the pay-off marker, and drawn
        # to the RIGHT of its line -- hence the right margin on xlim below.
        top = max(_rounds_meas)
        if top > g + 1e-9:
            ax.axvline(top, color="k", ls=":", lw=1.2, zorder=1)
            # Placed relative to the BAND ceiling, not as a fraction of ymax.
            # ymax varies a lot between these figures (the rank panel tops out
            # near 2.8, the pooled one near 4.6), and a fixed fraction put this
            # label inside the shaded band on the shorter ones, directly on top
            # of the band's own caption.
            # Above the HIGHEST curve at this x, not at a fixed height: the
            # curves fan out towards the strong-judge end, so any fixed
            # placement eventually runs through one of them.
            # Max over [top, right edge], not just AT top: the label extends
            # rightwards and the curves keep climbing under it, so clearing
            # them only at its anchor still let the steepest one cross the text.
            _scan = np.linspace(top, max(xs_plot), 24)
            _here = [float(np.max(np.interp(_scan, np.array(_xs_by_et[et][0]),
                                            np.array(_xs_by_et[et][1]))))
                     for et in _xs_by_et if len(_xs_by_et[et][0]) >= 2]
            _y_top = max(_here) + 0.06 * (ymax - WORTH_IT) if _here else WORTH_IT
            ax.text(top + 0.012, min(_y_top, ymax * 0.97),
                    f"ρ² ≈ {top:g}: substantial\nsavings ({_at(top):.2f}×)",
                    fontsize=9, va="center", color="#333")
    ax.set_xlabel(
        "judge–human agreement  ρ²  (squared Pearson correlation)" if corr_kind == "pearson"
        else "judge–human agreement  ρ²  (squared Spearman correlation, on paired differences)"
        if corr_kind == "spearman"
        else "judge–human agreement  ρ²  (Pearson for mean tests, Spearman for rank tests)")
    ax.set_ylabel("label-efficiency multiplier\n(equivalent human labels / actual labels)")
    if _LABEL_EFF_FIGURE_TITLES:
        ax.set_title("How good must an LLM judge be before PPI saves labeling effort?", fontsize=11)
    # Ticks on ROUND values, not on the measured tier positions. The data sits
    # where it was measured (which is why the markers are off-round), but a
    # reader looking up "my judge scores 0.4" needs 0.4 to be findable on the
    # axis. Gridlines at the same places make that lookup a straight read down.
    _lo_tick = 0.2
    _ticks = np.round(np.arange(_lo_tick, max(xs_plot) + 0.1001, 0.1), 2)
    ax.set_xticks(_ticks)
    for _t in _ticks:
        ax.axvline(_t, color="#bbb", lw=0.7, ls="-", alpha=0.55, zorder=0)
    # Right margin so the top-tier annotation has somewhere to sit that is not
    # on top of the strong-judge CI bands.
    ax.set_xlim(min(min(xs_plot), _lo_tick) - 0.035, max(xs_plot) + 0.135)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    if _LABEL_EFF_FIGURE_TITLES:
        fig.text(0.5, -0.04, "Bands are bootstrap 95% CIs on the median, pooled over effect sizes and the $N_{lab}$ grid.", ha="center", fontsize=8.5)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_label_efficiency_per_method_table(
    per_method_points: dict, out_dir: str, run_stem: str,
) -> str:
    """Per-method label-efficiency table: each method compared against ITS OWN
    classical power curve.

    This is the fair within-method comparison, and it is the one a reviewer
    should be shown. The pooled multiplier averages rejection rates across
    methods and then inverts a pooled curve, which conflates two different
    things: how much PPI buys for a given test, and how powerful that test was
    to begin with. Wilcoxon's smaller pooled gain, for instance, is partly just
    Wilcoxon being a lower-powered test on this data -- inverting PPI-Wilcoxon
    against CLASSICAL-Wilcoxon separates the two and asks only "how many human
    labels would a plain Wilcoxon have needed to match PPI-Wilcoxon?".

    It is also the diagnostic that explains binary's pooled outlier: paired_t
    has ~2x ttest_welch's baseline power on binary AND takes the largest PPI
    gain, so pooling them and inverting in the curve's steep region inflates
    the result. Per method, that inflation disappears.

    One row per (eval_type, method, rho^2 tier, n_lab)."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    path = out_base / f"{run_stem}_ppi_label_efficiency_per_method.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "noise_family", "method", "rho2_target", "rho2",
                         "rho2_pearson",
                         "rho2_spearman", "rank_penalty", "n_lab", "effect_frac",
                         "n_reps", "ppi_power", "equiv_n_lab", "multiplier",
                         "multiplier_lo", "multiplier_hi", "saturated", "predicted_mult",
                         "inversion_ratio", "inversion_clamped", "well_conditioned",
                         "variance_multiplier"])
        for key, pts in sorted(per_method_points.items()):
            eval_type, noise_family, method = key
            for r in sorted(pts, key=lambda q: (q.alignment_target, q.n_lab, q.effect_frac)):
                mult = r.equiv_n_lab / r.n_lab if r.n_lab else float("nan")
                # rank_penalty = rho2_pearson - rho2_spearman: how much of the
                # judge's linear signal a rank-based analysis cannot use. It is
                # the checkable diagnostic for the PPI-t-test vs PPI-Wilcoxon
                # gap, computable on a calibration set before any sweep runs.
                # SIGN FLIPS with noise_family -- negative (a rank BONUS) under
                # a contaminated judge. That reversal is the point of the
                # noise_family axis; see notes/RANK_VS_PARAMETRIC_CROSSOVER.md.
                _, _p2, _s2 = _method_rho2(eval_type, round(r.judge_noise, 6), method, noise_family)
                writer.writerow([
                    eval_type, noise_family, method, f"{r.alignment_target:.2f}", f"{r.rho2:.4f}",
                    f"{_p2:.4f}", f"{_s2:.4f}", f"{_p2 - _s2:.4f}", r.n_lab,
                    f"{r.effect_frac:.2f}", r.n_reps, f"{r.ppi_power:.6f}",
                    f"{r.equiv_n_lab:.4f}", f"{mult:.4f}",
                    f"{r.mult_lo:.4f}", f"{r.mult_hi:.4f}", r.saturated,
                    f"{r.predicted_mult:.4f}",
                    f"{r.inversion_ratio:.4f}", r.inversion_clamped, r.well_conditioned,
                    f"{r.variance_multiplier:.4f}",
                ])
    print(f"Saved results: {path}")
    return str(path)


_METHOD_CORR_KIND = {
    "ttest":       ("group",  "pearson"),
    "ttest_welch": ("group",  "pearson"),
    "mwu":         ("group",  "spearman"),
    "paired_t":    ("paired", "pearson"),
    "wilcoxon":    ("paired", "spearman"),
}
"""Which correlation governs each method's PPI variance reduction.

PPI++ is a control variate, so the variance reduction is 1 - rho^2 where rho
correlates the influence functions of the labeled estimator and the
judge-based rectifier. Two things vary by method:

STRUCTURE. A paired test's estimand is a function of the differences
D = Y_x - Y_y, so its control variate is Dhat = f_x - f_y and rho is
Pearson(D, Dhat), not the score-level correlation -- differencing two noisy
measurements changes the signal-to-noise ratio.

ESTIMAND. A mean-type test has an influence function linear in the values, so
Pearson is exact. A rank-type test (wilcoxon, mwu) has an influence function
that is a function of ranks -- for the signed-rank statistic the Hajek
projection is 1 - F_D(-d) - theta -- so the governing quantity is Spearman
instead, exact under H0 and first-order under local alternatives.

The four omnibus methods (anova_ind, anova_rep, friedman, kruskal) are
deliberately absent: no omnibus recipe was validated in this codebase when
this table was built. See notes/omnibus_label_efficiency.html for the
validated recipes (group/double structures, effect-size handling) needed to
add them.

Caveat on the entries already here: rho is not effect-invariant for the rank
methods (wilcoxon, mwu, kruskal, friedman), and _method_rho2 assumes it is
(builds its cell at effect_size=0.0, caches with no effect-size term). Mean
methods (ttest, paired_t, anova_rep) are exactly effect-invariant by
construction (Pearson on a linear influence function is location-invariant);
rank/dominance estimands' influence functions involve the CDF, whose shape
changes as groups separate, so their effect-invariant Spearman recipe
increasingly overstates rho^2 (and hence N_eff) as the true effect grows.
This is undetected within PPI_LABEL_EFF_EFFECT_FRACS' small-effect sweep
range; see notes/omnibus_label_efficiency.html for the full measurement and
for threading effect_size into _method_rho2's cache key as the fix."""


@functools.lru_cache(maxsize=None)
def _method_rho2(eval_type: str, judge_noise: float, method: str, noise_family: str = "gaussian",
                 n_mc: int = 60_000, seed: int = 3, shape_label: str | None = None) -> tuple:
    """(rho2 for `method`, pearson^2, spearman^2) on this judge's own scale.

    Returns all three so callers can also report the Pearson-minus-Spearman
    gap, which is the diagnostic for how much a rank-based analysis gives up
    relative to a mean-based one on the same judge.

    shape_label selects the truth marginal, matching JudgeBiasSource's field of
    the same name; None keeps the eval type's representative shape. It is part
    of the cache key, and it MUST be passed whenever the cells being predicted
    use a non-default shape. This argument did not exist before 2026-08-25, and
    its absence was a silent-wrong-number bug rather than a missing feature:
    the recipe was always built on _ppi_power_baseline(eval_type)'s default
    shape, so a sweep run under any other one compared a rho^2 from one DGP
    against measurements from a different DGP, with nothing in the output
    saying so. Under cont-near-center that misread ttest's recipe as 9% low and
    paired_t's as 12% low at d=0, where both are exact by construction.

    Cached: a sweep asks for the same (eval_type, judge_noise, method) on every
    n_lab and effect-size cell, and this draws n_mc rows each time."""
    from scipy.stats import pearsonr, spearmanr

    base = _ppi_power_baseline_binary() if eval_type == "binary" else _ppi_power_baseline(eval_type)
    kw = dict(base)
    kw["llm_noise"] = judge_noise
    if shape_label is not None:
        kw["shape_label"] = shape_label
    # Must match the cell being predicted: at matched total error variance a
    # contaminated judge yields a HIGHER Spearman than a gaussian one, so
    # reusing the gaussian correlation here would under-predict the
    # contaminated arm's rank-test multipliers by exactly the effect this axis
    # was added to measure.
    _fam_map = {lab: (nf, kws) for lab, nf, kws in PPI_LABEL_EFF_NOISE_FAMILIES}
    _nf, _kws = _fam_map.get(noise_family, (noise_family, {}))
    kw["noise_family"] = _nf
    kw.update(_kws)
    sc = JudgeBiasSource(name="_corr", tag="_ref", effect_size=0.0, **kw)
    cell = generate_judge_bias_cell(replace(sc, n=n_mc), np.random.default_rng(seed))
    structure, _ = _METHOD_CORR_KIND.get(method, ("group", "pearson"))
    if structure == "paired":
        a = np.asarray(cell.truth_x, dtype=float) - np.asarray(cell.truth_y, dtype=float)
        b = np.asarray(cell.llm_x, dtype=float) - np.asarray(cell.llm_y, dtype=float)
    else:
        # BOTH groups, each centred on its own mean, then concatenated.
        #
        # A two-sample estimand's influence function spans both groups, so its
        # control-variate correlation is the WITHIN-GROUP pooled one. Reading
        # group A alone was wrong whenever the judge's quality differs between
        # groups -- which is exactly what bias_type="differential" creates.
        #
        # It went unnoticed because for continuous and likert the differential
        # bias is an additive OFFSET, and Pearson is shift-invariant, so the two
        # groups' rho^2 agree to 4 decimal places. Binary's bias is a change in
        # FLIP PROBABILITY, which does move phi: at the cleanest tier group A
        # (biased) reads 0.712 while group B reads 0.923. Using A alone
        # under-predicted the bound by a factor of 1.244 at n_lab=200 -- almost
        # exactly the 1.24-1.40 "impossible" overshoot binary's top tier showed
        # in the group-structure methods, while its paired methods, which never
        # took this branch, sat at a healthy 0.94.
        #
        # Centring per group before pooling is what makes this the within-group
        # correlation rather than one inflated by the between-group difference.
        _a1 = np.asarray(cell.truth_a2, dtype=float)
        _b1 = np.asarray(cell.llm_a2, dtype=float)
        _a2 = np.asarray(getattr(cell, "truth_b2", _a1), dtype=float)
        _b2 = np.asarray(getattr(cell, "llm_b2", _b1), dtype=float)
        a = np.concatenate([_a1 - _a1.mean(), _a2 - _a2.mean()])
        b = np.concatenate([_b1 - _b1.mean(), _b2 - _b2.mean()])
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return (float("nan"), float("nan"), float("nan"))
    p2 = float(pearsonr(a, b).statistic) ** 2
    s2 = float(spearmanr(a, b).statistic) ** 2
    _, kind = _METHOD_CORR_KIND.get(method, ("group", "pearson"))
    return ((s2 if kind == "spearman" else p2), p2, s2)


def save_ppi_label_efficiency_plots_per_method(
    raw: list, calib_rows: list, out_path: str, ref_n_mc: int = 3000, seed: int = 71,
) -> tuple[list[str], dict]:
    """One set of label-efficiency figures PER METHOD, alongside the pooled set.

    Returns (plot paths, {(eval_type, method): points}) so the caller can feed
    the same points to save_ppi_label_efficiency_per_method_table without
    rebuilding any reference curves.

    The pooled multiplier averages rejection rates across methods and then
    inverts a pooled reference curve. That is a nonlinear composition, so it is
    only trustworthy when the methods it pools have comparable power -- and
    they do not. Measured on the 300-rep sweep:

      * binary's paired_t has ~2x the baseline power of ttest_welch
        (human-subset 0.59 vs 0.29) and takes the largest PPI gain in the study
        (0.580 -> 0.913 at rho^2=0.70). Pooling 0.913 with 0.601 and inverting
        in the curve's steep upper region produced binary's 4.13x at rho^2=0.70
        -- an artifact of averaging two very differently-powered tests, not a
        real cross-type difference.
      * the rank tests (mwu, wilcoxon) gain systematically less than the
        mean-based ones (+0.165 vs +0.216 at rho^2=0.70), so pooling them in
        understates what a mean-based analysis actually achieves, and by more
        at high judge quality.

    Per-method figures make both visible instead of averaged away. They are
    cheap -- the reference curves are disk-cached (see
    _classical_pooled_power_curve), so after the first run each method's curve
    is a file read -- and diagnostic: a method whose curve looks nothing like
    its siblings is the signal that pooling is hiding something.

    Every method should also clear the y=x line. A method sitting at or below
    it is not paying for itself over simply analysing the labeled subset.

    ref_n_mc MATCHES run_ppi_label_efficiency_check's default on purpose: these
    figures are read against the pooled ones, and curves built at a different
    Monte Carlo count are not comparable to them (and would miss the pooled
    run's cache entries). Measured on the 300-rep sweep, raising it to 10_000
    moved every pooled tier by under 2% and did not move the threshold at all,
    while costing ~3x -- the residual inversion error is dominated by
    conditioning at small effect size (worst deviation 0.158 at es=0.15 vs
    0.046 at es=0.35), where the power curve is flat and dn/dP is large, not by
    Monte Carlo noise. More samples cannot fix a flat curve."""
    import pathlib
    base = pathlib.Path(out_path)
    n_grid = np.geomspace(float(_JB_MIN_LAB), 1500.0, 36)
    # Keys carry noise_family: the two arms calibrate to DIFFERENT llm_noise
    # values for the same tier, so a family-blind nearest-noise match can
    # silently attribute a contaminated cell to a gaussian tier.
    _fam = lambda c: (c[6] if len(c) > 6 else "gaussian")
    tier_of = {(c[0], _fam(c), round(c[1], 4)): c[3] for c in calib_rows}
    val_of = {(c[0], _fam(c), round(c[1], 4)): c[4] for c in calib_rows}
    rho_of = {(c[0], _fam(c), round(c[1], 4)): float(c[5].get("rho2", float("nan"))) for c in calib_rows}
    by_method: dict = defaultdict(list)
    for r in raw:
        _fm = re.search(r"\.fam=([a-z]+)\.", r.name)
        by_method[(r.eval_type, _fm.group(1) if _fm else "gaussian", r.method)].append(r)
    paths: list[str] = []
    collected: dict = {}
    for (eval_type, noise_family, method), rows in sorted(by_method.items()):
        pts = []
        for r in rows:
            m = re.search(r"noise=(\d+\.\d+)", r.name)
            if not m:
                continue
            nz = float(m.group(1))
            mf = re.search(r"\.es=([\d.]+?)\.?$", r.name)
            _frac = float(mf.group(1)) if mf else float("nan")
            # Correlation matching THIS method's structure and estimand, not
            # the calibration panel's score-level rho^2 -- see _METHOD_CORR_KIND.
            _mr, _p2, _s2 = _method_rho2(eval_type, round(nz, 6), method, noise_family)
            keys = [k for k in tier_of if k[0] == eval_type and k[1] == noise_family]
            if not keys:
                continue
            k = min(keys, key=lambda q: abs(q[2] - nz))
            # PPIComparisonResult.effect_size is the eval-type-RELATIVE
            # FRACTION (see its docstring), not the absolute magnitude
            # _classical_pooled_power_curve needs -- the pooled path in
            # run_ppi_label_efficiency_check correctly uses
            # sources[0].effect_size, the JudgeBiasSource field. Passing the
            # fraction here built every per-method reference curve at the wrong
            # effect size, in a different DIRECTION per eval type: continuous's
            # true es is 0.018-0.042 so a 0.15 curve was far too powerful and
            # every inversion clamped to the grid minimum (97% clamped, 0% well
            # conditioned); likert's is 0.17-0.40 and binary's 0.13, so those
            # curves were too weak and their inversions overshot (median 2.88
            # and 1.62 against a target of 1.00). Pooled results were never
            # affected.
            es = (_jb_effect_magnitude_binary(_frac) if eval_type == "binary"
                  else _jb_effect_magnitude(eval_type, _frac))
            pg = _smooth_monotone_power_curve(
                n_grid, _classical_pooled_power_curve(eval_type, es, (method,), n_grid, ref_n_mc, seed))
            pw = r.rejects_ppi / r.n_reps if r.n_reps else float("nan")
            eq = _equivalent_n_lab(pw, n_grid, pg) if np.isfinite(pw) else float("nan")
            lo, hi = _multiplier_ci(pw, r.n_reps, r.n_lab, n_grid, pg)
            # Per-method conditioning gate, against THIS method's own curve --
            # see LabelEfficiencyPoint.inversion_ratio.
            _hp = r.rejects_human_subset / r.n_reps if r.n_reps else float("nan")
            _ih = _equivalent_n_lab(_hp, n_grid, pg) if np.isfinite(_hp) else float("nan")
            _ir = _ih / r.n_lab if (r.n_lab and np.isfinite(_ih)) else float("nan")
            _ic = bool(np.isfinite(_ih) and (_ih <= n_grid.min() + 1e-9 or _ih >= n_grid.max() - 1e-9))
            pts.append(LabelEfficiencyPoint(
                eval_type=eval_type, judge_noise=nz, alignment_metric="rho2",
                alignment_target=_nominal_tier(eval_type, tier_of[k]),
                alignment_value=val_of[k], n_lab=r.n_lab,
                n_reps=r.n_reps, ppi_power=pw, equiv_n_lab=eq,
                effect_frac=_frac, mult_lo=lo, mult_hi=hi,
                saturated=bool(np.isfinite(pw) and pw >= pg.max() - 1e-9),
                rho2=_mr, predicted_mult=_ppi_predicted_savings(_mr, r.n_lab, r.n),
                predicted_mult_asymptotic=_ppi_predicted_savings(_mr, 0, 1),
                inversion_ratio=_ir, inversion_clamped=_ic,
                noise_family=noise_family,
                variance_multiplier=(r.var_human_subset / r.var_ppi
                                     if getattr(r, "var_ppi", 0)
                                     and np.isfinite(r.var_ppi) else float("nan"))))
        if not pts:
            continue
        collected[(eval_type, noise_family, method)] = pts
        # Family in the filename: without it the two arms' figures collide and
        # the second silently overwrites the first.
        tag = f"{eval_type}_{noise_family}_{method}"
        try:
            paths.append(save_ppi_label_efficiency_plot(
                _pool_label_eff_across_es(pts), str(base.with_name(f"{base.stem}_bymethod_{tag}{base.suffix}"))))
        except Exception as exc:
            print(f"  (per-method plot skipped for {tag}: {exc})")
    print(f"Saved {len(paths)} per-method label-efficiency plots")
    return paths, collected


_PPI_ROBUSTNESS_CACHE_VERSION = 1
"""Bump to invalidate every cached robustness result below. The cache key
already covers every argument, so this is only for changes to the COMPUTATION
that the arguments cannot see (a different estimator, a changed DGP)."""
_PPI_ROBUSTNESS_CACHE_DIR = pathlib.Path("simulations/out/.ppi_robustness_cache")


def _robustness_cached(name: str, key_parts: tuple, compute):
    """Disk-memoize one robustness table.

    These checks are pure seeded Monte Carlo with no dependence on the sweep's
    own data, so they are safe to reuse across runs -- and worth it: together
    they cost ~35 min, which would otherwise be paid by every sweep including
    the official tests, to recompute a number that cannot have changed.

    Same discipline as _classical_pooled_power_curve: atomic temp+rename so
    parallel workers cannot serve a half-written file, an unreadable entry is
    a miss rather than an error, and PPI_NO_ROBUSTNESS_CACHE=1 bypasses."""
    key = hashlib.sha256(repr((_PPI_ROBUSTNESS_CACHE_VERSION, name) + key_parts).encode()).hexdigest()[:20]
    path = _PPI_ROBUSTNESS_CACHE_DIR / f"{name}_{key}.csv"
    use_cache = os.environ.get("PPI_NO_ROBUSTNESS_CACHE", "") != "1"
    if use_cache and path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    df = compute()
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp.csv")
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
        except Exception:
            pass  # caching is an optimization; never fail the sweep over it
    return df


def save_ppi_rho2_robustness_plots(
    out_path: str, reps: int = 1500, seed: int = 61,
) -> list[str]:
    """The two supplementary figures behind "the rule of thumb holds up".

    Both come from dedicated experiments rather than the sweep grid, because
    each needs something the grid cannot give:

      sufficiency -- pins Pearson rho^2 ANALYTICALLY across judge-error shapes
        (kappa = sqrt(1/target - 1) fixes it exactly whatever the shape, since
        Pearson sees only second moments) and asks whether the multiplier is a
        function of rho^2 alone. The grid's tiers are CALIBRATED, not pinned,
        so it cannot isolate shape this way. Answer: 0.987 +/- 0.028 for
        paired_t vs rho_P^2, 0.954 +/- 0.022 for wilcoxon vs rho_S^2 -- each
        family against its own correlation.

      crossover -- locates where PPI power swaps between rank-based and
        parametric tests as contamination moves rho_S^2 at pinned rho_P^2.

    See notes/RANK_VS_PARAMETRIC_CROSSOVER.md and
    notes/WHICH_RHO_FOR_WHICH_TEST.md. Results are disk-cached
    (_robustness_cached), so only the first sweep pays for them."""
    from simulations.investigate_rho2_sufficiency import run as _suff_run
    from simulations.investigate_rank_parametric_crossover import run as _cross_run
    from simulations.plot_rank_crossover_and_sufficiency import (
        plot_crossover as _plot_cross, plot_sufficiency as _plot_suff,
    )
    base = pathlib.Path(out_path)
    suff = _robustness_cached("sufficiency", (reps, 20, seed), lambda: _suff_run(reps, 20, seed))
    cross = _robustness_cached("crossover", (reps, 500, 17), lambda: _cross_run(reps, 500, 17))
    paths = [
        _plot_suff(suff, str(base.with_name(f"{base.stem}_rho2_sufficiency{base.suffix}"))),
        _plot_cross(cross, str(base.with_name(f"{base.stem}_rank_crossover{base.suffix}"))),
    ]
    return paths


_LOOKUP_PANELS = (
    (("group", "pearson"),   "Between-subjects $t$-test",   "Pearson on scores"),
    (("paired", "pearson"),  "Within-subjects paired $t$",  "Pearson on paired differences"),
    (("group", "spearman"),  "Mann–Whitney",                "Spearman on scores"),
    (("paired", "spearman"), "Wilcoxon signed-rank",        "Spearman on paired differences"),
)
"""The four (structure, correlation) combinations _METHOD_CORR_KIND maps
methods onto, each with the design a practitioner would recognise and the
statistic they must actually compute."""


def save_ppi_label_efficiency_lookup_grid(per_method_points: dict, out_path: str,
                                          compact: bool = False) -> str:
    """Practitioner lookup: one panel per experimental design, each stating the
    statistic to measure and reading the multiplier off it.

    The two-figure split (parametric vs rank) fixed WHICH CORRELATION, but not
    WHICH DATA it is computed on, and those are independent axes. A parametric
    panel pooling ttest/ttest_welch (correlate raw SCORES) with paired_t
    (correlate paired DIFFERENCES) puts two different measurements on one
    x-axis, and they diverge: mean gap 0.094 rho^2, up to 0.197 on likert,
    where the same judge reads 0.440 on scores and 0.261 on differences. A
    within-subjects likert user looking up 0.26 would land on a curve built
    partly from judges whose SCORES correlate at 0.44.

    Splitting on the full (structure, correlation) pair makes each panel
    unambiguous, so the caption can be an instruction rather than a caveat:
    find your design, compute the named statistic on a pilot set, read across.

    x is each eval type's OWN realized rho^2 -- see
    save_ppi_label_efficiency_threshold_plot for why drawing them at a pooled
    mean silently shifts whichever eval type sits furthest from it.

    Panels with no methods (binary has no rank tests) are annotated rather than
    left blank.

    compact=True lays the four panels in ONE ROW at the paper's printed width
    (7in) with a single shared legend, instead of a 2x2 grid at 11.4in that
    the paper then scales to 0.61x -- which is what makes the labels
    hard to read in print. Same data, same panels; only the arrangement and
    the type sizes differ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    WORTH_IT = 1.25
    marks = {"binary": "o", "continuous": "s", "likert": "^"}
    cols = {"binary": "#2166ac", "continuous": "#1a9850", "likert": "#b2182b"}

    # (structure, corr) -> eval_type -> tier -> [multipliers], plus realized rho^2
    cell: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rho: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (et, _fam, method), vals in per_method_points.items():
        kind = _METHOD_CORR_KIND.get(method)
        if kind is None:
            continue
        for r in vals:
            if r.saturated or not getattr(r, "well_conditioned", True) or not r.n_lab:
                continue
            cell[kind][et][r.alignment_target].append(r.equiv_n_lab / r.n_lab)
            if np.isfinite(r.rho2):
                rho[kind][et][r.alignment_target].append(r.rho2)
    if not cell:
        raise ValueError("save_ppi_label_efficiency_lookup_grid: no usable points")

    if compact:
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharey=False)
        axes = np.asarray(axes).reshape(1, 4)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), sharey=False)
    for ax, (kind, design, measure) in zip(axes.ravel(), _LOOKUP_PANELS):
        ymax, drew = 1.0, False
        for et in ("binary", "continuous", "likert"):
            tiers = sorted(t for t in cell[kind].get(et, {})
                           if cell[kind][et][t] and rho[kind][et].get(t))
            if len(tiers) < 2:
                continue
            xs = [float(np.mean(rho[kind][et][t])) for t in tiers]
            ys = [float(np.median(cell[kind][et][t])) for t in tiers]
            ax.plot(xs, ys, marker=marks[et], color=cols[et],
                    lw=1.1 if compact else 2, ms=2.8 if compact else 6,
                    label=_LABEL_EFF_PANEL_TITLES.get(et, et), zorder=3)
            ymax = max(ymax, max(ys)); drew = True
        if not drew:
            ax.text(0.5, 0.5, "no tests of this kind\non this data type",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axhspan(0.95, WORTH_IT, color="grey", alpha=0.16, zorder=0)
            for t in np.round(np.arange(0.1, 0.95, 0.1), 2):
                ax.axvline(t, color="#bbb", lw=0.7, alpha=0.55, zorder=0)
            ax.axhline(1.0, color="#c0392b", ls="--", lw=1.1, alpha=.8, zorder=1)
            ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
            if not compact:
                ax.legend(fontsize=8, loc="upper left")
        if compact:
            # the measure IS the point of this figure, so it stays on the panel
            ax.set_title(f"{design}\n{measure}", fontsize=6.5, linespacing=1.25)
            ax.tick_params(labelsize=6.0, length=2, pad=1.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        else:
            ax.set_title(f"{design}\nmeasure: {measure}", fontsize=10)
    if compact:
        for ax in axes[0]:
            ax.set_xlabel("judge–human agreement  ρ²", fontsize=6.5)
        axes[0][0].set_ylabel("label-efficiency\nmultiplier", fontsize=6.5)
    else:
        for ax in axes[1]:
            ax.set_xlabel("judge–human agreement  ρ²  (measured as named above)")
        for ax in axes[:, 0]:
            ax.set_ylabel("label-efficiency multiplier")
    # In compact mode both of these are dropped: at 7in they dwarf the panels
    # and collide with the shared legend, and the LaTeX caption already carries
    # the instruction and the band's meaning.
    if _LABEL_EFF_FIGURE_TITLES and not compact:
        fig.suptitle("Find your design, measure that statistic on a pilot set, read across",
                 fontsize=12)
    if _LABEL_EFF_FIGURE_TITLES and not compact:
        fig.text(0.5, 0.005, "Shaded band: savings under 1.25×, not worth restructuring a "
                 "pipeline for. Points sit at each data type's own measured ρ².",
                 ha="center", fontsize=8.5)
    if compact:
        h, l = [], []
        for ax in axes.ravel():
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.7)
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=6.5,
                   handlelength=1.3, columnspacing=1.2, handletextpad=0.4,
                   bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_label_efficiency_noise_family_plot(
    per_method_points: dict, out_path: str, compact: bool = False,
) -> str:
    """The robustness figure: does the rule of thumb survive a judge whose
    errors are NOT Gaussian?

    Laid out as eval_type (columns) x TEST FAMILY (rows), because the pooled
    view actively hides the result. Pooled across methods, contamination looks
    like it helps continuous (+0.03..+0.35) and hurts likert (-0.03..-0.24) --
    opposite directions, which reads as incoherent. Split by family the same
    effect appears in both:

        continuous   mean tests -0.07   rank tests +0.32
        likert       mean tests -0.25   rank tests -0.04

    i.e. contamination costs mean-based tests and spares rank-based ones
    everywhere; likert's overall drop is a discretisation cost (clipping and
    ties destroy information for every test) sitting on top of that. Averaging
    the two families together cancels the signal and leaves only a net sign
    that flips between eval types.

    Rows are keyed off _METHOD_CORR_KIND's correlation kind -- "pearson"
    methods use the values directly and so are the parametric row, "spearman"
    methods are functions of ranks -- rather than a hardcoded name list, so a
    newly added method lands in the right row automatically.

    Takes save_ppi_label_efficiency_plots_per_method's `collected` mapping,
    keyed (eval_type, noise_family, method), NOT the pooled
    LabelEfficiencyPoint list: the pooled points have already averaged the
    method axis away, which is exactly the axis this figure needs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    rows_spec = [("parametric (t-tests)", "pearson"), ("non-parametric (rank tests)", "spearman")]
    pts = defaultdict(list)
    for (et, fam, method), vals in per_method_points.items():
        kind = _METHOD_CORR_KIND.get(method, (None, "pearson"))[1]
        for r in vals:
            if not r.saturated and getattr(r, "well_conditioned", True) and r.n_lab:
                pts[(et, kind, fam)].append(r)
    if not pts:
        raise ValueError("save_ppi_label_efficiency_noise_family_plot: no usable points")
    ets = sorted({k[0] for k in pts}, key=lambda e: ("binary", "continuous", "likert").index(e)
                 if e in ("binary", "continuous", "likert") else 99)
    fams = sorted({k[2] for k in pts})
    if len(fams) < 2:
        raise ValueError(f"needs >=2 noise families, saw {fams}")

    # compact: the same rows x eval_types grid, drawn at the paper's printed
    # width (7in) with print-sized type, instead of 4.6in per column that
    # \includegraphics then scales down.
    _fs = ((7.0, 1.05 * len(rows_spec) + 0.85) if compact
           else (4.6 * len(ets), 7.4))
    fig, axes = plt.subplots(len(rows_spec), len(ets), figsize=_fs,
                             squeeze=False, sharex=True)
    style = {"gaussian": ("o-", "#3b76af"), "contaminated": ("s--", "#c0392b")}
    for ri, (row_label, kind) in enumerate(rows_spec):
        for ci, et in enumerate(ets):
            ax = axes[ri][ci]
            drew = False
            for fam in fams:
                agg = defaultdict(list)
                for r in pts.get((et, kind, fam), []):
                    agg[round(r.alignment_target, 3)].append(r.equiv_n_lab / r.n_lab)
                if not agg:
                    continue
                xs = sorted(agg)
                ys = [float(np.median(agg[x])) for x in xs]
                mk, col = style.get(fam, ("^:", "#61a05f"))
                ax.plot(xs, ys, mk, color=col, lw=1.1 if compact else 2.1,
                        ms=2.8 if compact else 6, label=f"{fam} judge")
                drew = True
            ax.axhline(1.0, color="grey", ls=":", lw=1)
            ax.grid(alpha=.25); ax.set_axisbelow(True)
            if compact:
                # Was gated on ri==0, so only the top row's ticks got sized
                # down to match the lookup grid; the bottom row silently fell
                # back to matplotlib's default (larger) tick label size.
                ax.tick_params(labelsize=6.0, length=2, pad=1.5)
                for _sp in ("top", "right"):
                    ax.spines[_sp].set_visible(False)
            if ri == 0:
                ax.set_title(et, fontsize=6.5 if compact else 11.5)
            if ri == len(rows_spec) - 1:
                ax.set_xlabel(r"judge quality tier  ($\rho^2$)" if compact
                              else r"judge quality tier  ($\rho^2$, score level)",
                              fontsize=6.5 if compact else None)
            if ci == 0:
                # Compact mode needs a shorter row label than row_label's full
                # "parametric (t-tests)" / "non-parametric (rank tests)" --
                # at this font size the two-line label bled into its neighbor.
                # Matches the "mean-based"/"rank-based" vocabulary used
                # elsewhere in the text (parametric == mean-based here).
                _short_row = {"parametric (t-tests)": "parametric",
                             "non-parametric (rank tests)": "rank-based"}.get(row_label, row_label)
                ax.set_ylabel(f"{_short_row} multiplier" if compact
                              else f"{row_label}\nlabel-efficiency multiplier",
                              fontsize=6.5 if compact else 9.5)
            if not drew:
                # binary has no rank row: _COMPARISON_METHODS_BINARY excludes
                # mwu/wilcoxon because ranks are uninformative on 0/1 data.
                # Say so, or an empty panel reads as a plotting failure.
                ax.text(*((0.42, 0.80) if compact else (0.5, 0.5)), "no rank tests on 0/1 data\n(ranks carry no information there)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=5.6 if compact else 8.5, color="#888", style="italic")
                # Strip the y scale. Matplotlib's default 0-1 range on an empty
                # panel reads as a multiplier axis running below 1.0, i.e. as
                # measurements showing PPI doing WORSE than labels alone --
                # the opposite of this figure's claim, asserted by an axis with
                # no data behind it. Keep the frame so the grid stays aligned.
                ax.set_yticks([])
                for side in ("left", "right", "top"):
                    ax.spines[side].set_visible(False)
                ax.grid(False)
            elif len({k[2] for k in pts if k[0] == et and k[1] == kind}) < 2:
                ax.text(*((0.40, 0.86) if compact else (0.5, 0.04)), "no error-shape axis\n(flip-probability judge)",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=5.4 if compact else 8, color="#777", style="italic")
    _seen, _h, _l = set(), [], []
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in _seen:
                    _seen.add(lab); _h.append(h); _l.append(lab)
    if _h:
        if compact:
            fig.legend(_h, _l, loc="lower center", ncol=len(_l), frameon=False,
                       fontsize=6.5, handlelength=1.3, columnspacing=1.2,
                       handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
        else:
            axes[0][0].legend(_h, _l, fontsize=9, loc="upper left")
    if _LABEL_EFF_FIGURE_TITLES:
        fig.suptitle("" if compact else "Does the rule of thumb survive a non-Gaussian judge?\n"
                 "same judge-quality tiers, two judge-error shapes, split by test family",
                 fontsize=11.5)
    if compact:
        # reserve room for the shared legend the compact branch adds below --
        # a 2-item single-row legend needs less than the lookup grid's 3-item
        # one, so this reserves less (was 0.11, oversized: visible gap above
        # the legend once the markers/lines were sized down to match).
        fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.5, w_pad=0.6)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_label_efficiency_plots(
    results: list[LabelEfficiencyPoint], out_path: str, square: bool = True,
) -> list[str]:
    """Emit the OVERALL label-efficiency figure (pooled across the effect-size
    sweep) plus ONE FIGURE PER effect size, as separate .png files.

    The per-es figures are not decoration: the multiplier is a property of
    judge quality and should be es-INVARIANT, so the sweep's whole value as a
    robustness check is that a reader can see the arms agree (or not) on the
    n_lab cells they share. Pooling alone would average that check away --
    an arm that disagrees would silently shift the pooled curve rather than
    announce itself.

    The pooled figure averages equiv_n_lab across arms per
    (eval_type, alignment_target, n_lab) cell, skipping saturated points
    (see LabelEfficiencyPoint.saturated -- a saturated equiv_n_lab is
    clamped to n_grid's edge and would drag any average it enters).

    Returns every path written, overall first."""
    # The headline figure is GAUSSIAN-ONLY, deliberately. Averaging the
    # noise-family arms into one multiplier would make the headline number
    # depend on an arbitrary 50/50 mix of judge-error regimes that corresponds
    # to no real population of judges. Keeping it to one stated regime means
    # the number has a definition; the family comparison gets its own figure
    # below, where the effect is shown rather than averaged away.
    # To blend instead, drop this filter -- now that binary carries a real
    # contaminated arm the blend is at least consistent across eval types.
    # Three factorial views of the same sweep:
    #   out_path              -- AVERAGED over judge-error shapes (main text)
    #   {stem}_fam_gaussian   -- gaussian judge only     (supplementary)
    #   {stem}_fam_contaminated -- contaminated judge only (supplementary)
    # The averaged one is the headline because it quantifies the multiplier a
    # practitioner should expect without assuming a judge-error shape; the
    # per-family ones are what license that average, by showing how much the
    # two regimes actually differ. See _pool_label_eff_across_es' docstring for
    # why the blend is opt-in rather than the pooling default.
    base = Path(out_path)
    paths = [save_ppi_label_efficiency_plot(
        _pool_label_eff_across_es(results, across_noise_families=True), out_path, square=square)]
    for fam in sorted({r.noise_family for r in results}):
        fam_rows = [r for r in results if r.noise_family == fam]
        if not fam_rows:
            continue
        try:
            paths.append(save_ppi_label_efficiency_plot(
                _pool_label_eff_across_es(fam_rows),
                str(base.with_name(f"{base.stem}_fam_{fam}{base.suffix}")), square=square))
        except Exception as exc:
            print(f"  (per-family figure skipped for {fam}: {type(exc).__name__}: {exc})")
    # The two analysis figures: es-invariance (which licenses pooling across
    # arms at all) and the practitioner-facing agreement threshold.
    # Supplementary robustness pair (disk-cached; only the first sweep pays).
    try:
        paths += save_ppi_rho2_robustness_plots(out_path)
    except Exception as exc:
        print(f"  (rho^2 robustness figures skipped: {type(exc).__name__}: {exc})")
    # The threshold figure now needs per-method points (one figure per test
    # family, each on its own correlation), so it is emitted from the
    # per-method block alongside the noise-family figure -- not here, where
    # only pooled points are available.
    fracs = sorted({r.effect_frac for r in results})
    if len(fracs) > 1:
        try:
            paths.append(save_ppi_label_efficiency_invariance_plot(
                results, str(base.with_name(f"{base.stem}_es_invariance{base.suffix}"))))
            paths.append(save_ppi_label_efficiency_invariance_pooled_plot(
                results, str(base.with_name(f"{base.stem}_es_invariance_pooled{base.suffix}"))))
        except ValueError:
            pass
        for frac in fracs:
            subset = [r for r in results if r.effect_frac == frac]
            if not subset:
                continue
            # readable suffix: foo_es0p35.png
            sub_path = base.with_name(f"{base.stem}_es{f'{frac:.2f}'.replace('.', 'p')}{base.suffix}")
            # Blended to match the headline: these arms exist to check that the
            # headline multiplier is effect-size invariant, so they have to be
            # the same quantity it is.
            paths.append(save_ppi_label_efficiency_plot(
                _pool_label_eff_across_es(subset, across_noise_families=True),
                str(sub_path), square=square))
    return paths


def _pool_label_eff_across_es(
    results: list[LabelEfficiencyPoint], *, across_noise_families: bool = False,
) -> list[LabelEfficiencyPoint]:
    """Average equiv_n_lab/ppi_power across the effect-size arms, per
    (eval_type, noise_family, alignment_target, n_lab). Saturated points are
    dropped first; a cell with nothing left stays saturated so the plot's own
    saturation handling still fires.

    `across_noise_families=True` additionally averages the judge-error-shape
    arms together, collapsing to (eval_type, alignment_target, n_lab). This is
    OPT-IN rather than the default because it silently blends two different
    judge-error regimes into one multiplier, and the resulting number depends
    on the mix of families the sweep happened to run (today an even split,
    which matches no measured population of real judges). It is the right
    default for a main-text figure that wants one multiplier per judge-quality
    tier averaged over error shapes; it is the wrong one for any comparison
    ACROSS eval types unless every eval type carries the same families -- which
    is why binary's contaminated arm had to be implemented for real rather than
    skipped (see scenarios.synthetic._contaminated_flip_probs)."""
    from collections import defaultdict
    buckets: dict[tuple, list[LabelEfficiencyPoint]] = defaultdict(list)
    for r in results:
        # noise_family is part of the key: pooling across it would average two
        # DIFFERENT judge-error regimes into one multiplier. Before binary had
        # a contaminated arm that was worse than untidy -- continuous/likert
        # blended two regimes while binary contributed only its clean one, so
        # cross-eval-type comparison silently flattered binary.
        _fam_key = "_all" if across_noise_families else r.noise_family
        buckets[(r.eval_type, _fam_key, r.alignment_target, r.n_lab)].append(r)
    pooled: list[LabelEfficiencyPoint] = []
    for (_et, _fam, _tgt, _nl), rows in buckets.items():
        usable = [r for r in rows if not r.saturated and r.well_conditioned and np.isfinite(r.equiv_n_lab)]
        src = usable or rows
        ref = src[0]
        # A blended point averages BOTH regimes, so it must not keep whichever
        # family happened to sort first in its bucket -- that label would claim
        # a gaussian-only measurement for a mixed one. Relabel explicitly.
        if across_noise_families and len({r.noise_family for r in src}) > 1:
            ref = replace(ref, noise_family="averaged")
        pooled.append(replace(
            ref,
            ppi_power=float(np.mean([r.ppi_power for r in src])),
            equiv_n_lab=float(np.mean([r.equiv_n_lab for r in src])),
            n_reps=int(sum(r.n_reps for r in src)),
            saturated=not usable,
            mult_lo=float(np.mean([r.mult_lo for r in src])),
            mult_hi=float(np.mean([r.mult_hi for r in src])),
            # The PREDICTION has to be averaged over the same arms the
            # measurement is. It does not vary across effect-size arms (it is a
            # function of rho^2, n_lab and N), so this was harmless while
            # pooling was effect-size only -- but it DOES vary across noise
            # families, by 8.5% on average and up to 0.38x, so inheriting
            # src[0]'s value drew the dashed line for whichever family happened
            # to sort first against a solid line averaging both.
            predicted_mult=float(np.mean([r.predicted_mult for r in src
                                          if np.isfinite(r.predicted_mult)] or [float("nan")])),
            predicted_mult_asymptotic=float(np.mean([r.predicted_mult_asymptotic for r in src
                                                     if np.isfinite(r.predicted_mult_asymptotic)]
                                                    or [float("nan")])),
            rho2=float(np.mean([r.rho2 for r in src if np.isfinite(r.rho2)] or [float("nan")])),
            variance_multiplier=float(np.mean([r.variance_multiplier for r in src
                                               if np.isfinite(r.variance_multiplier)]
                                              or [float("nan")])),
        ))
    return pooled


def save_ppi_label_efficiency_plot(results: list[LabelEfficiencyPoint], out_path: str, square: bool = True) -> str:
    """The flagship label-efficiency figure: one panel per eval type
    (binary, continuous, likert -- the standard panel order used
    throughout this harness's plots, see eval_types below), x=actual
    N_lab, y=equivalent human-only
    N_lab, one line per judge-QUALITY tier (calibrated to hit a target
    alignment level -- Pearson r for continuous, weighted kappa for likert,
    kappa for binary -- see _LABEL_EFF_ALIGNMENT_METRIC/_calibrate_noise_
    for_alignment), plus a y=x "no benefit from the judge" reference. Lines
    are labeled by alignment, not raw llm_noise, since the same noise value
    means very different judge quality across eval types -- alignment is
    the one axis a reader can compare panels against directly (a "kappa=0.8
    judge" means the same thing in every panel; a "noise=0.2 judge" does
    not). The target~0.7 tier is drawn bolder as a visual anchor (no shaded
    region to the diagonal, to avoid visual clutter); the other tiers show
    how that benefit moves with judge quality on the same axes. N (the
    total item count, fixed throughout -- see run_ppi_label_efficiency_check
    and PPI_LABEL_EFF_N's docstring) is left out of the title/axis text
    deliberately -- callers state it in the caption instead, so this figure
    carries no on-plot annotations or N callouts beyond the axis labels and
    legend.

One legend, shared across the whole figure (not one per panel) and
positioned to the right of the last panel. Each panel's lines ARE
calibrated to that eval type's own alignment metric (Pearson r for
continuous, weighted kappa for likert, kappa for binary), but the legend
labels by the generic "IRR~=<target>" (inter-rater reliability) instead
of the metric-specific symbol/achieved value -- since the label no longer
varies by panel, all three panels' entries for a given target collapse
into one shared legend line, rather than three near-duplicate,
metric-incompatible ones a reader had to mentally re-split by panel.

    ``square`` (default True): whether x and y share one axis max with
    ``ax.set_aspect("equal")``, so the y=x reference renders at a literal
    45 degrees. When True (the default), the shared max is driven by
    whichever of x (N_lab tested) or y (equiv_n_lab) is larger -- since
    the multiplier is consistently > 1x, that's almost always y, so most
    of the panel's width ends up spent on N_lab values nobody tested,
    compressing every real point toward the bottom-left corner. Pass
    ``square=False`` to trade the 45-degree diagonal for legibility at low
    N_lab instead: x caps at just the N_lab grid actually tested, y
    expands independently to fit equiv_n_lab, and the axes are unequal.
    The y=x reference line is exactly y=x in data coordinates either way
    -- with square=False it just won't render at a visual 45 degrees."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No label-efficiency results to plot.")
    eval_types = [et for et in ("binary", "continuous", "likert") if any(r.eval_type == et for r in results)]
    fig, axes = plt.subplots(
        1, len(eval_types), figsize=(4.4 * len(eval_types), 4.2), squeeze=False,
        gridspec_kw={"wspace": 0.15},
    )
    axes = axes[0]
    cmap = plt.cm.viridis
    # Collected across all panels (not per-panel) and deduped by label, so
    # the one shared legend has every category used anywhere (e.g.
    # "power saturated" even if only one panel happens to hit it) without
    # repeating a target's entry once per panel.
    legend_handles: dict[str, "plt.Artist"] = {}

    for col, et in enumerate(eval_types):
        ax = axes[col]
        et_rows = [r for r in results if r.eval_type == et]
        # Descending: highest alignment (best judge) plotted first/darkest.
        targets = sorted({r.alignment_target for r in et_rows}, reverse=True)
        # 0.7 is the visual "baseline" (fill + annotation) regardless of how
        # many OTHER targets surround it or where it sits in the sorted
        # list -- a deliberate, fixed choice (not derived from picking the
        # middle LIST POSITION, which silently drifted onto 0.6 when the
        # target set widened from 3 to 5 points, caught before it shipped).
        baseline_target = min(targets, key=lambda t: abs(t - 0.7))

        # Axis scale comes from NON-saturated points only -- a single
        # saturated cell's clamped-to-n_grid.max() equiv_n_lab must never be
        # allowed to dictate the panel scale (see
        # LabelEfficiencyPoint.saturated's docstring for the bug this
        # previously caused: one continuous cell's "500 labels" artifact
        # squashed every real point in that panel into an unreadable sliver).
        # Falls back to using every row's n_lab (never equiv_n_lab) if a
        # panel is saturated everywhere, which no current eval_type is.
        unsaturated = [r for r in et_rows if not r.saturated]
        x_data_max = max(r.n_lab for r in et_rows)
        if unsaturated:
            y_data_max = max(r.equiv_n_lab for r in unsaturated)
        else:
            y_data_max = x_data_max * 3.0

        if square:
            # x and y share one max (see this function's `square` docstring
            # section) so the y=x reference renders at a literal 45 degrees.
            x_max = y_max = max(x_data_max, y_data_max) * 1.15
        else:
            # x and y scale independently -- x caps at the actual N_lab
            # grid tested, y at the largest equiv_n_lab actually observed.
            x_max = x_data_max * 1.05
            y_max = y_data_max * 1.15

        no_benefit_line, = ax.plot(
            [0, x_max], [0, x_max], color="black", ls="--", lw=1.2, alpha=0.6,
            label="No benefit (y = x)", zorder=2,
        )
        legend_handles.setdefault("No benefit (y = x)", no_benefit_line)

        for i, target in enumerate(targets):
            rows = sorted((r for r in et_rows if r.alignment_target == target), key=lambda r: r.n_lab)
            xs = [r.n_lab for r in rows]
            # Saturated points are plotted as a lower-bound marker clipped
            # just inside the axis ceiling, never at their raw (meaningless)
            # equiv_n_lab value -- see LabelEfficiencyPoint.saturated.
            # Saturated points are pinned AT the axis ceiling, not just
            # below it. Drawing them at 0.97*y_max made them visually
            # indistinguishable from a real measurement slightly under the
            # highest true point -- a triangle at ~388 read as "tops out near
            # 390" when it actually means ">= 500, truly >= 800". Pinning to
            # the ceiling plus a caret marker says "runs off the top", which
            # is what a lower bound should look like.
            # UNUSABLE cells break the line instead of being drawn through.
            #
            # Pinning a saturated cell to y_max and letting the polyline run
            # through it manufactures a spike that no measurement supports: the
            # line dives to the ceiling and back for a cell whose value is not
            # known, and a reader cannot tell that excursion from a real
            # non-monotonicity. Binary's small-n_lab corner was unreadable for
            # exactly this reason -- the pooled cells there have every
            # constituent point filtered out (saturated = not usable), so the
            # spikes were drawn entirely from cells carrying no information.
            #
            # A NaN in the y-series makes matplotlib lift the pen, so the line
            # shows only the segments joining cells that were actually
            # measured. The markers are still drawn at the ceiling afterwards,
            # so "this cell exists and runs off the top" is still visible --
            # only the fictitious connecting segments are gone.
            #
            # Ill-conditioned cells are treated the same way: an inversion the
            # gate refuses to report should not anchor a line segment either.
            def _usable(r):
                return not r.saturated and getattr(r, "well_conditioned", True)
            ys = [r.equiv_n_lab if _usable(r) else float("nan") for r in rows]
            color = cmap(0.15 + 0.7 * i / max(1, len(targets) - 1))
            is_baseline = target == baseline_target
            marker = _LABEL_EFF_MARKER_SHAPES[i % len(_LABEL_EFF_MARKER_SHAPES)]
            # Labeled by the TARGET (a round, panel-independent number),
            # not each panel's own achieved alignment value -- the whole
            # point of one shared legend is that a given target's entry
            # means the same thing in every panel, which a per-panel
            # achieved value (e.g. r=0.83 here, weighted-kappa=0.79 there)
            # would undermine.
            line, = ax.plot(
                xs, ys, color=color, marker=marker,
                markersize=_LABEL_EFF_MARKER_SIZE.get(marker, 5), linewidth=2.0 if is_baseline else 1.4,
                label=f"ρ²~={target:.2f}",
                zorder=4,
            )
            legend_handles.setdefault(f"ρ²~={target:.2f}", line)

            # Control-variate prediction n_lab / (1 - rho^2*(1 - n_lab/N)),
            # drawn per tier in that tier's own colour (see
            # _ppi_predicted_savings). One SHARED legend entry rather than one
            # per tier -- it is the same theory curve in every case, and the
            # colour already says which tier it belongs to.
            #
            # It is expected to sit ABOVE the measured line at the strong-judge
            # tiers and converge at the weak ones: the prediction is on the
            # VARIANCE scale while equiv_n_lab comes from inverting a power
            # curve, which saturates. Divergence at the top is the power
            # ceiling, not a failure of the theory -- which is exactly why the
            # curve is worth drawing on the same axes.
            pred = [(r.n_lab, r.n_lab * r.predicted_mult) for r in rows
                    if np.isfinite(getattr(r, "predicted_mult", float("nan")))]
            # Points above the axis ceiling are DROPPED, not clamped to it:
            # clamping drew a flat run along the top edge that reads as a real
            # measurement topping out, when it means the prediction is off
            # scale. y_max is set from measured data, which saturates, so the
            # strong-judge predictions legitimately exceed it.
            pred = [q for q in pred if q[1] <= y_max]
            if pred:
                pline, = ax.plot(
                    [q[0] for q in pred], [q[1] for q in pred],
                    color=color, linestyle=(0, (1, 1.8)), linewidth=1.2, alpha=0.8, zorder=3,
                    label="Predicted from ρ²",
                )
                legend_handles.setdefault("Predicted from ρ²", pline)

            # y_max explicitly, NOT ys: ys now carries NaN at every unusable
            # cell so the connecting line breaks there (see above), and reading
            # the caret positions back out of it would place them all at NaN
            # and silently draw nothing.
            #
            # Covers ill-conditioned cells as well as saturated ones. Both are
            # "measured, but not reportable"; a reader needs to see that the
            # cell exists and why the line stops, and the distinction between
            # the two failure modes is in the CSV for anyone who needs it.
            # Unusable cells are NOT drawn. The line already breaks at them
            # (NaN in the y-series), so the gap is visible; adding a marker at
            # the axis ceiling on top of that put a symbol where no value was
            # measured, and readers consistently read it as a data point near
            # the top rather than as an absence. A broken line says "nothing
            # here" without asserting a magnitude.
            #
            # The count of omitted cells is reported by the caller rather than
            # drawn, so it can go in a caption where it can be explained -- see
            # the retention numbers in HOW_MULTIPLIERS_ARE_MEASURED.md.

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Num human labels used")
        ax.set_ylabel("Num human labels a classical test would need" if col == 0 else "")
        # pad clears the saturated caret, which is pinned at y_max with
        # clip_on=False and so projects ~half a marker height above the axes.
        ax.set_title(et.capitalize(), pad=10)
        if square:
            ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Label Efficiency: Human Labels a Classical Test Would Need to Match PPI's Power",
        fontsize=11, y=0.99,
    )
    # One shared legend, ordered "No benefit" -> IRR targets descending ->
    # any non-tier entries (e.g. the saturated marker) -- NOT plain insertion order (legend_handles fills
    # in whatever order panels happen to hit each category, so
    # "power saturated" can land mid-list if an early panel saturates on
    # its very first tier); explicitly sorted here instead.
    def _legend_sort_key(label: str) -> tuple[int, float]:
        if label == "No benefit (y = x)":
            return (0, 0.0)
        # Anything that isn't an "IRR~=<value>" tier entry sorts last. Matched
        # structurally rather than by exact string: this previously compared
        # against a hardcoded "power saturated" and raised IndexError the
        # moment that label's wording changed, since the fallthrough branch
        # assumes an "=" is present.
        if not label.startswith("ρ²"):
            return (2, 0.0)
        try:
            return (1, -float(label.rsplit("=", 1)[1]))  # descending IRR target
        except (IndexError, ValueError):
            return (2, 0.0)
    ordered_labels = sorted(legend_handles.keys(), key=_legend_sort_key)
    # Anchored to the RIGHTMOST axes' own transAxes (not a hand-picked
    # figure-fraction number, and not bbox_to_anchor=(1.0, ...), which
    # butts the legend's edge right up against the last panel with no
    # visible gap): a figure-fraction anchor has to be re-tuned any time
    # panel count/content changes tight_layout's actual axes width, and
    # over/under-shooting it either leaves a dead gap or overlaps the last
    # panel. Anchoring to the last axes' own coordinate system at 1.05
    # gives a fixed, panel-relative gap (5% of that axes' width) that's
    # stable regardless of the overall figure layout.
    axes[-1].legend(
        [legend_handles[l] for l in ordered_labels], ordered_labels,
        loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=8, borderaxespad=0.3, frameon=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_ppi_nlab_grid_report(
    results: list[PPIComparisonResult], alpha: float, header: str = "N x N_LAB GRID (calibration)",
) -> None:
    """N (columns) x N_lab (rows) grid table, one mini-table per arm
    (all_human / human_subset / ppi -- the three arms build_ppi_nlab_grid_
    sources' question is actually about; llm_only/llm_impute are omitted
    here since they aren't valid tests regardless of N/N_lab, so their rate
    doesn't answer a calibration-vs-(N,N_lab) question). Reading ACROSS a
    row (fixed N_lab, N varying) isolates N's effect; reading DOWN a column
    (fixed N, N_lab varying) isolates N_lab's effect -- directly separating
    "it's the ratio N_lab/N that matters" from "it's the absolute N_lab
    count that matters," which build_ppi_comparison_label_frac_sources'
    N=100-only sweep can't do on its own. One grid per eval type
    (build_ppi_nlab_grid_sources now crosses continuous/likert): grouping
    by `.eval_type` here isn't optional once more than one eval type is
    present -- (N, N_lab) pairs collide across eval types (each combination
    is generated for every eval type), so reading `results` without
    grouping would silently pick whichever eval type's row happened to come
    first for each cell."""
    if not results:
        print(f"\n  (no {header} results)")
        return
    eval_types = sorted({r.eval_type for r in results})
    print(f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- {header}\n"
          f"  Rows = N_lab (labeled items), columns = N (total items); nominal alpha={alpha}\n{'='*88}")
    for et in eval_types:
        et_results = [r for r in results if r.eval_type == et]
        n_values = sorted({r.n for r in et_results})
        nlab_values = sorted({r.n_lab for r in et_results})
        print(f"\n  === {et.capitalize()} ===")
        for label, rejects_field in [
            ("all_human", "rejects_all_human"), ("human_subset", "rejects_human_subset"), ("ppi", "rejects_ppi"),
        ]:
            print(f"\n  [{label}]")
            print(f"    {'N_lab \\ N':<10}" + "".join(f"n={n}".rjust(9) for n in n_values))
            for nlab in nlab_values:
                row = f"    {nlab:<10}"
                for n in n_values:
                    r = next((r for r in et_results if r.n == n and r.n_lab == nlab), None)
                    if r is None or r.n_reps == 0:
                        row += f"{'-':>9}"
                        continue
                    rate = getattr(r, rejects_field) / r.n_reps
                    row += f"{rate:>9.3f}"
                print(row)
    print()


def save_results_artifacts_ppi_nlab_grid(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str, header: str,
    pooled_results: list[PPIComparisonResult] | None = None,
) -> list[str]:
    """Same CSV shape as save_results_artifacts_ppi_comparison, but logs via
    print_ppi_nlab_grid_report instead -- that function's tag-based grouping
    (tag "power" / "compare_label_frac") doesn't match this grid's tags
    ("nlab_grid" / "nlab_grid_power"), so reusing it directly would produce
    an empty-looking log.

    `results` is the RAW (per-method) data, saved verbatim to the CSV.
    `pooled_results` (falls back to pooling `results` if omitted) feeds the
    saved .log instead -- print_ppi_nlab_grid_report's `next((r for r in
    et_results if r.n == n and r.n_lab == nlab), None)` cell lookup picks
    the first matching row for each (N, N_lab) cell, so fed raw data it
    silently reports whichever METHOD happens to appear first instead of
    the 4-method-averaged rate. See save_results_artifacts_ppi_comparison's
    docstring for the same issue there."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_nlab_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "method", "n", "n_lab", "n_reps", "effect_size",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.eval_type, r.method, r.n, r.n_lab, r.n_reps, repr(float(r.effect_size)),
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_nlab_grid_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_nlab_grid_report(pooled_results, alpha=alpha, header=header)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_nlab_grid_plot(
    *, calibration_results: list[PPIComparisonResult] | None, power_results: list[PPIComparisonResult] | None,
    alpha: float, out_path: str,
) -> str:
    """Heatmap(s) of the PPI-corrected rejection rate over the (N, N_lab)
    plane -- calibration (effect_size=0, diverging colormap centered on
    alpha so under/over-rejection are visually distinct) and power
    (moderate effect_size, sequential colormap), side by side when both are
    given. This is the direct visual answer to "is it the ratio N_lab/N or
    the absolute N_lab that drives calibration/power": scanning a ROW shows
    N's effect at fixed N_lab, scanning a COLUMN shows N_lab's effect at
    fixed N -- the line plots elsewhere in this module can't show this
    since they never vary N and N_lab independently (build_ppi_power_sources
    fixes N=100; build_ppi_comparison_label_frac_sources also fixes N=100
    and only varies the ratio).

    One ROW per eval type present across calibration_results/power_results
    (build_ppi_nlab_grid_sources now crosses continuous/likert), one COLUMN
    per panel (calibration and/or power) -- (N, N_lab) pairs collide across
    eval types the same way they do in print_ppi_nlab_grid_report, so each
    panel's grid is built from that eval type's rows only."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    panels = []
    if calibration_results:
        panels.append(("Type-I Error\n(no real effect)", calibration_results, "RdBu_r", alpha))
    if power_results:
        panels.append(("Power\n(moderate real effect)", power_results, "viridis", None))
    if not panels:
        raise ValueError("No N x N_lab grid results to plot.")
    eval_types = sorted({r.eval_type for _title, results, _cmap, _center in panels for r in results})

    fig, axes = plt.subplots(
        len(eval_types), len(panels), figsize=(6.0 * len(panels), 5.0 * len(eval_types)), squeeze=False,
    )
    for row, et in enumerate(eval_types):
        for col, (title, results, cmap, center) in enumerate(panels):
            ax = axes[row][col]
            et_results = [r for r in results if r.eval_type == et]
            n_values = sorted({r.n for r in et_results})
            nlab_values = sorted({r.n_lab for r in et_results})
            grid = np.full((len(nlab_values), len(n_values)), np.nan)
            for r in et_results:
                if r.n_reps == 0:
                    continue
                grid[nlab_values.index(r.n_lab), n_values.index(r.n)] = r.rejects_ppi / r.n_reps

            if center is not None:
                vmax = max(2.0 * center, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * center)
                im = ax.imshow(grid, origin="lower", cmap=cmap, norm=TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=vmax), aspect="auto")
            else:
                im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            for i in range(len(nlab_values)):
                for j in range(len(n_values)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                        )
            ax.set_xticks(range(len(n_values)))
            ax.set_xticklabels([str(n) for n in n_values])
            ax.set_yticks(range(len(nlab_values)))
            ax.set_yticklabels([str(nl) for nl in nlab_values])
            ax.set_xlabel("N (total items)")
            ax.set_ylabel("N_lab (labeled items)")
            ax.set_title(f"[{et.capitalize()}] {title}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PPI-Corrected Rejection Rate over N × N_lab (nominal {_alpha_label(alpha)})", y=1.02, fontsize=12)
    fig.text(0.5, -0.02, "Paired-mean estimand", ha="center", fontsize=8, color="#555555")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode, full factorial: build_ppi_factorial_sources' 7-factor cross
# (bias_magnitude x N x N_lab x label_mechanism x effect_size x
# bias_direction x llm_noise), analyzed three ways -- a pooled binomial GLM
# (which factors/interactions actually move the PPI-corrected rejection
# rate, with real coefficients/p-values, fit at the llm_noise=0.20 baseline
# -- see _PPI_FACTORIAL_FORMULA's docstring for why noise isn't itself a GLM
# term), a curated set of 2D heatmap slices (visual summary, also at the
# noise=0.20 baseline), and the judge-human alignment-
# bucketed false-positive-rate view (build_ppi_alignment_results_from_
# factorial/save_ppi_alignment_sweep_plot, further down this section), which
# is the one place llm_noise's other 10 levels get used. Reuses
# _run_ppi_comparison_cell/run_ppi_comparison_simulation unchanged -- this
# section is entirely new sources + new analysis, no new execution path.
# ---------------------------------------------------------------------------

_PPI_FACTORIAL_NAME_RE = re.compile(
    r"^fact\.(?P<et>[a-z]+)\.bm=(?P<bm>[a-z]+)\.n=(?P<n>\d+)\.nlab=(?P<nlab>\d+)\.lm=(?P<lm>[a-z_]+)\.es=(?P<es>[a-z]+)\.bd=(?P<bd>[a-z]+)\.noise=(?P<noise>[\d.]+)$"
)


def _parse_ppi_factorial_name(name: str) -> dict:
    m = _PPI_FACTORIAL_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized factorial scenario name: {name!r}")
    d = m.groupdict()
    d["n"] = int(d["n"])
    d["nlab"] = int(d["nlab"])
    d["noise"] = float(d["noise"])
    return d


def _ppi_factorial_dataframe(results: list[PPIComparisonResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = _parse_ppi_factorial_name(r.name)
        rows.append({
            **d, "method": r.method, "n_reps": r.n_reps,
            "rejects_ppi": r.rejects_ppi, "fails_ppi": r.n_reps - r.rejects_ppi,
            "rate_ppi": r.rejects_ppi / r.n_reps if r.n_reps else float("nan"),
            "rejects_all_human": r.rejects_all_human, "rejects_human_subset": r.rejects_human_subset,
        })
    return pd.DataFrame(rows)


_PPI_FACTORIAL_FORMULA = (
    "rejects_ppi + fails_ppi ~ "
    "C(bm, Treatment('none')) + C(n) + C(nlab) + C(lm, Treatment('mcar')) "
    "+ C(es, Treatment('null')) + C(bd, Treatment('opposing')) "
    "+ C(et, Treatment('continuous')) "
    "+ C(bm, Treatment('none')):C(es, Treatment('null')) "
    "+ C(bd, Treatment('opposing')):C(es, Treatment('null'))"
)
"""Grouped-binomial GLM formula (statsmodels/patsy's "successes + failures ~
..." syntax, the standard encoding for aggregate count data -- equivalent to
a per-replicate logistic regression here since there are no per-replicate
covariates beyond the factors themselves). Main effects for all seven
factors (the original six, plus eval_type now that build_ppi_factorial_
sources crosses continuous/likert), plus two theoretically-motivated 2-way
interactions: bias_magnitude:effect_size (does bias severity change how
power grows with effect size) and bias_direction:effect_size (does the
opposing/reinforcing asymmetry itself depend on effect size). N:N_lab is
deliberately NOT included as an interaction term here despite
build_ppi_nlab_grid_sources' finding that N_lab matters far more than N at
small N_lab -- that finding is a statement about MAGNITUDE (visible
directly in the heatmap), not really a linear-interaction question, and
the seven main effects plus two interactions already leave ample residual
df on ~624 cells (continuous+likert combined); adding more interaction
terms than the sample supports would just widen every coefficient's CI
without adding information. et is likewise a main effect only, not crossed
with the other six factors -- this treats "does the whole 6-factor picture
shift up/down for likert vs. continuous" as the question worth asking here,
not "does every individual factor interact differently with eval_type,"
which would need a fractional design to stay estimable. bm/bd/et are
Treatment-coded at their "no bias"/"opposing"/"continuous" reference levels
so every coefficient reads as "vs. no bias" / "vs. opposing" / "vs.
continuous," matching how the rest of the PPI mode's plots and reports are
already framed.

llm_noise (build_ppi_factorial_sources' 7th factor) is deliberately NOT a
term here, and `results` fed to this function should be pre-filtered to
noise=0.20 (the baseline every non-alignment factorial output already used
before llm_noise joined the source grid) -- adding it as an eighth main
effect would run into a real confound, not just added complexity: llm_noise
only varies away from 0.20 on es="null" cells (see build_ppi_factorial_
sources' docstring), so any non-baseline noise level implies es="null" with
perfect collinearity against the es term already in the formula, making the
two effects statistically inseparable. The full noise-swept es="null"
subset is exactly what feeds the separate alignment-bucketed view instead
(build_ppi_alignment_results_from_factorial), which bypasses this GLM
entirely and reports realized-alignment buckets directly rather than a
fitted noise coefficient."""


_PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS = {
    "bm": "none", "lm": "mcar", "es": "null", "bd": "opposing", "et": "continuous",
}
"""Every Treatment() reference level _PPI_FACTORIAL_FORMULA depends on --
checked up front by fit_ppi_factorial_model before handing `df` to patsy,
so a missing level fails with a clear, actionable message pointing at
which column/level is missing, rather than patsy's generic (and, for a
run that just spent 30-60 minutes of compute, very unwelcome) "specified
level 'x' not found" with no indication of why. This guards specifically
against PPI_FACTORIAL_NOISE_LEVELS losing its required 0.20 anchor point,
which would silently empty the es="null" rows out of the noise==0.20
baseline subset this formula is always fit on -- see that constant's
docstring."""


def fit_ppi_factorial_model(results: list[PPIComparisonResult]) -> tuple[str, pd.DataFrame]:
    """Fit _PPI_FACTORIAL_FORMULA and return (summary_text, raw dataframe).

    Caveat worth keeping in the write-up: at es="large" (or any stratum
    where the corrected rate saturates to ~0 or ~1 for every level of
    another factor), the GLM can show quasi-complete separation -- huge
    coefficients/standard errors for terms involving that stratum. This
    isn't a fitting bug; it's the correct signal that once power has
    saturated, that stratum carries no further information about which
    factor moved it there. statsmodels still converges and the OTHER
    (non-saturated) coefficients remain informative; treat any
    coefficient with a standard error orders of magnitude larger than its
    neighbors as "this stratum saturated," not as a real, enormous effect.

    Raises ValueError (not patsy's own cryptic "specified level ... not
    found") if `results` is missing any of _PPI_FACTORIAL_FORMULA_
    REFERENCE_LEVELS' required Treatment() reference levels -- almost
    always means `results` was filtered down to the wrong noise level (or
    otherwise mis-scoped) before reaching here, not a real data problem;
    see that constant's docstring."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = _ppi_factorial_dataframe(results)
    if df.empty:
        raise ValueError("fit_ppi_factorial_model: `results` produced an empty dataframe -- nothing to fit.")
    for col, required_level in _PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS.items():
        present = set(df[col].unique())
        if required_level not in present:
            raise ValueError(
                f"fit_ppi_factorial_model: column {col!r} is missing its required reference level "
                f"{required_level!r} (present: {sorted(present)}) -- `results` is almost certainly filtered "
                f"to the wrong scope (e.g. the wrong llm_noise baseline) before reaching this function. "
                f"See _PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS' docstring."
            )
    fit = smf.glm(formula=_PPI_FACTORIAL_FORMULA, data=df, family=sm.families.Binomial()).fit()
    return fit.summary().as_text(), df


def _print_ppi_factorial_lm_noise_table(null_rows: pd.DataFrame, alpha: float) -> None:
    """(label_mechanism x llm_noise) Type-I calibration breakdown, printed
    from the FULL noise-swept null-cell subset -- deliberately NOT the
    single baseline noise level fit_ppi_factorial_model/save_ppi_factorial_
    heatmap_plot are restricted to (see _PPI_FACTORIAL_FORMULA's docstring
    for why THAT restriction is real: llm_noise is collinear with es="null"
    once non-null cells, which only exist at the baseline noise, join the
    regression). That confound has no bearing on a null-cells-only table
    like this one, and restricting to baseline noise alone would hide the
    worst of the problem: mnar_strong looks mild to nonexistent at the
    baseline noise level but is far worse at low noise -- i.e. a more
    accurate-looking judge -- because low noise makes the judge's score
    track truth closely enough that MNAR-on-truth selection is effectively
    selecting on the judge's own score too, maximizing the rectifier's
    selection bias. mcar stays flat and near-nominal across the whole noise
    range."""
    if null_rows.empty:
        return
    lm_order = ["mcar", "mnar_mild", "mnar_strong"]
    lms = [lm for lm in lm_order if lm in set(null_rows["lm"])]
    noises = sorted(null_rows["noise"].unique())
    print(f"\n  Null-cell (Type-I) rejection rate by label_mechanism x llm_noise "
          f"(nominal alpha={alpha}; full noise sweep, NOT baseline-filtered):")
    print(f"    {'lm':<12}" + "".join(f"{n:>9.4f}" for n in noises) + f"{'mean':>9}")
    for lm in lms:
        lm_rows = null_rows[null_rows["lm"] == lm]
        cells = []
        for n in noises:
            sub = lm_rows[lm_rows["noise"] == n]
            reps = int(sub["n_reps"].sum())
            cells.append(int(sub["rejects_ppi"].sum()) / reps if reps else float("nan"))
        tot_reps = int(lm_rows["n_reps"].sum())
        row_mean = int(lm_rows["rejects_ppi"].sum()) / tot_reps if tot_reps else float("nan")
        cell_str = "".join(f"{c:>9.3f}" if np.isfinite(c) else f"{'--':>9}" for c in cells)
        print(f"    {lm:<12}{cell_str}{row_mean:>9.3f}")
    print()


def _print_ppi_factorial_method_lm_table(null_rows_raw: pd.DataFrame, alpha: float) -> None:
    """(method x label_mechanism) Type-I calibration breakdown, printed from
    the RAW per-method null-cell rows (never the method-pooled `results`
    the GLM/worst-cell scan use -- pool_ppi_comparison_across_methods
    combines every method's rejects/n_reps into one row per scenario before
    it ever reaches this report, so a per-method table is structurally
    impossible to recover from that pooled data; callers must pass the
    UNPOOLED per-method results separately -- see print_ppi_factorial_
    report's `raw_results_full` docstring).

    Exists because pooling methods together can hide the same way pooling
    label_mechanism/noise did: confirmed directly on a screening run where
    the omnibus family's pooled MCAR mean read 0.050 (looks perfectly
    calibrated) while the local-rectifier Kruskal variant (kruskal_mnar_
    experimental, at the time still the default under the name kruskal_corr)
    ALONE, unpooled, read 0.066 mean / 0.18 worst cell under that same
    MCAR-only slice -- anova_ind/anova_rep/friedman's good calibration was
    silently absorbing its elevation in the pooled view. This finding is
    what led to demoting that variant out of the default set (see KRUSKAL/
    KRUSKAL_MNAR_EXPERIMENTAL in methods.py) -- this table exists so a
    single miscalibrated method can't hide behind the rest again."""
    if null_rows_raw.empty:
        return
    methods = [m for m in null_rows_raw["method"].unique()]
    lm_order = ["mcar", "mnar_mild", "mnar_strong"]
    lms = [lm for lm in lm_order if lm in set(null_rows_raw["lm"])]
    print(f"\n  Null-cell (Type-I) rejection rate by method x label_mechanism "
          f"(nominal alpha={alpha}; unpooled per-method, full noise sweep):")
    print(f"    {'method':<16}" + "".join(f"{lm:>13}" for lm in lms))
    for method in methods:
        m_rows = null_rows_raw[null_rows_raw["method"] == method]
        cells = []
        for lm in lms:
            sub = m_rows[m_rows["lm"] == lm]
            reps = int(sub["n_reps"].sum())
            cells.append(int(sub["rejects_ppi"].sum()) / reps if reps else float("nan"))
        cell_str = "".join(f"{c:>13.3f}" if np.isfinite(c) else f"{'--':>13}" for c in cells)
        print(f"    {method:<16}{cell_str}")
    print()


def print_ppi_factorial_report(
    results: list[PPIComparisonResult], alpha: float, label: str = "paired_t",
    *, null_results_full: list[PPIComparisonResult] | None = None,
    raw_results_full: list[PPIComparisonResult] | None = None,
) -> None:
    """Regression summary (fit_ppi_factorial_model) plus two quotable
    headline numbers: the worst observed Type-I inflation (among es="null"
    cells) and the largest all_human-vs-ppi power gap (among non-null
    cells) -- the single-number "worst case across N x N_lab" summary
    figures, pulled directly from the factorial grid rather than eyeballed
    off a table.

    `label` names the estimand(s) `results` was pooled across in the header
    (default "paired_t", the original single-estimand factorial) -- pass
    _COMPARISON_METHODS_LABEL/_COMPARISON_METHODS_OMNIBUS_LABEL for the
    2-group/omnibus pooled reports respectively (see run()'s factorial_check
    block); this function itself is agnostic to which methods `results` was
    pooled across, it just needs a name for the header text.

    `null_results_full`, if given, should be the FULL pooled factorial
    results across every swept llm_noise level (unlike `results`, which
    must stay restricted to the single baseline noise level the GLM's
    identifiability depends on -- see fit_ppi_factorial_model's formula
    docstring). When given, the worst-Type-I-cell scan and the new
    label_mechanism x noise table below are computed from it instead of
    from `results`, so they aren't silently confined to whichever single
    noise level the GLM happens to require. Falls back to `results` (old
    behavior, baseline-only) when omitted.

    `raw_results_full`, if given, should be the RAW (never method-pooled),
    full-noise-range per-method results underlying `results`/
    `null_results_full` -- e.g. `[r for r in factorial_results_raw if
    r.method in _COMPARISON_METHODS]`, NOT `factorial_results`/
    `factorial_results_baseline` (both already pool_ppi_comparison_across_
    methods'd). When given, prints a (method x label_mechanism) table so a
    single well-calibrated method can't hide a miscalibrated one the pooled
    numbers above would otherwise average away -- see
    _print_ppi_factorial_method_lm_table's docstring for why this matters
    (confirmed to happen for real: kruskal_mnar_experimental vs. the rest of
    the omnibus family)."""
    if not results:
        print("\n  (no PPI factorial results)")
        return
    # The regression is a CROSS-eval-type contrast (et is a Treatment()
    # factor against a "continuous" reference, see
    # _PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS), so it is not merely
    # unfittable but meaningless when the run covers one eval type -- as
    # an --eval-types-restricted re-run does. Skip it and carry on to the
    # rest of the report rather than raising: the 2026-08-24 likert-only
    # re-run lost its alignment sweep to this, after the 11.7h factorial
    # it depends on had already finished and been written to disk.
    present_ets = {getattr(r, "eval_type", None) for r in results}
    ref_et = _PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS["et"]
    if ref_et not in present_ets:
        print(f"\n  (skipping the factorial regression: it contrasts eval types against "
              f"'{ref_et}', which this run does not cover -- present: "
              f"{'/'.join(sorted(str(e) for e in present_ets))}. Per-cell results, the "
              f"alignment sweep and all plots are unaffected.)")
        return
    summary_text, df = fit_ppi_factorial_model(results)
    eval_types = sorted(df["et"].unique())
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- FULL FACTORIAL "
          f"(bias_magnitude x N x N_lab x label_mechanism x effect_size x bias_direction x eval_type)\n"
          f"  {len(results)} cells, {'/'.join(eval_types)}/{label}; nominal alpha={alpha}\n{'='*96}\n")
    print(summary_text)

    null_df = _ppi_factorial_dataframe(null_results_full) if null_results_full is not None else df
    null_rows = null_df[null_df["es"] == "null"]
    if len(null_rows):
        worst = null_rows.loc[(null_rows["rate_ppi"] - alpha).abs().idxmax()]
        print(f"\n  Worst Type-I cell: rate={worst['rate_ppi']:.3f} (nominal alpha={alpha}) at "
              f"et={worst['et']} bm={worst['bm']} n={worst['n']} nlab={worst['nlab']} lm={worst['lm']} "
              f"bd={worst['bd']} noise={worst['noise']:.4f}")
        _print_ppi_factorial_lm_noise_table(null_rows, alpha)

    if raw_results_full is not None:
        raw_df = _ppi_factorial_dataframe(raw_results_full)
        raw_null_rows = raw_df[raw_df["es"] == "null"]
        _print_ppi_factorial_method_lm_table(raw_null_rows, alpha)

    nonnull_rows = df[df["es"] != "null"].copy()
    if len(nonnull_rows):
        nonnull_rows["power_gap"] = (nonnull_rows["rejects_all_human"] - nonnull_rows["rejects_ppi"]) / nonnull_rows["n_reps"]
        worst_gap = nonnull_rows.loc[nonnull_rows["power_gap"].idxmax()]
        print(f"  Largest all_human-vs-ppi power gap: {worst_gap['power_gap']:.3f} at "
              f"et={worst_gap['et']} bm={worst_gap['bm']} n={worst_gap['n']} nlab={worst_gap['nlab']} "
              f"lm={worst_gap['lm']} es={worst_gap['es']} bd={worst_gap['bd']}")
    print()


def save_results_artifacts_ppi_factorial(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str,
    pooled_results: list[PPIComparisonResult] | None = None, write_csv: bool = True, label: str = "paired_t",
    null_results_full: list[PPIComparisonResult] | None = None,
    raw_results_full: list[PPIComparisonResult] | None = None,
) -> list[str]:
    """`results` is the RAW (per-method) data, saved verbatim to the CSV
    (unless `write_csv=False` -- see below).

    `pooled_results` (falls back to pooling `results` if omitted) feeds
    the saved .log's GLM fit and headline numbers instead. The GLM
    coefficients themselves are numerically IDENTICAL either way (grouped-
    binomial log-likelihood is additive over rows sharing the same
    covariates), but print_ppi_factorial_report's two "worst cell" headline
    numbers are NOT: they pick the single most extreme row via `idxmax()`,
    and fed the raw per-method rows (4x as many, each nosier at 1/4 the
    pooled n_reps) that max is mechanically more extreme than the properly
    pooled one -- confirmed on a real official run: the raw-fed log claimed
    a "worst Type-I cell" of 0.445 (nominal alpha=0.05) where the correctly
    pooled figure for that same cell was 0.154, and a different "largest
    power gap" cell entirely (0.715 vs. the pooled 0.416). See
    save_results_artifacts_ppi_comparison's docstring for the same
    raw-vs-pooled issue in the other two saved logs.

    `write_csv=False` skips the CSV entirely, writing only the .log -- for a
    SECOND call against the SAME `run_stem` that should append another
    pooled summary (e.g. _COMPARISON_METHODS_OMNIBUS' own report) without
    re-writing (or worse, silently truncating to a different method subset)
    the CSV the first call already wrote for the combined raw data. `label`
    is forwarded to print_ppi_factorial_report's header text -- see its
    own docstring.

    `null_results_full`, if given, is forwarded to print_ppi_factorial_
    report's own `null_results_full` -- pass the FULL (every noise level)
    pooled results here, not the baseline-only `pooled_results`, so the
    saved .log's worst-cell/label_mechanism-x-noise numbers aren't silently
    confined to the GLM's single required noise level.

    `raw_results_full`, if given, is forwarded to print_ppi_factorial_
    report's own `raw_results_full` -- pass the methods-appropriate subset
    of `results` (RAW, never pool_ppi_comparison_across_methods'd) for
    WHICHEVER label/method family this specific call's `pooled_results` is
    scoped to (e.g. just the omnibus methods for the omnibus call), not the
    full combined `results` -- otherwise the printed per-method table would
    include methods this call isn't reporting on."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_factorial_results.csv"
    if write_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "name", "method", "et", "bm", "n", "nlab", "lm", "es", "bd", "noise", "n_reps",
                "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
            ])
            for r in results:
                d = _parse_ppi_factorial_name(r.name)
                writer.writerow([
                    r.name, r.method, d["et"], d["bm"], d["n"], d["nlab"], d["lm"], d["es"], d["bd"],
                    f"{d['noise']:.4f}", r.n_reps,
                    f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                    r.n_failed,
                ])
        print(f"Saved results: {csv_path}")
    summary_path = out_base / f"{run_stem}_ppi_factorial_summary.log"
    write_mode = "w" if write_csv else "a"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_factorial_report(
            pooled_results, alpha=alpha, label=label,
            null_results_full=null_results_full, raw_results_full=raw_results_full,
        )
    with summary_path.open(write_mode, encoding="utf-8") as handle:
        handle.write(buf.getvalue())
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)] if write_csv else [str(summary_path)]


def save_ppi_factorial_heatmap_plot(
    *, results: list[PPIComparisonResult], alpha: float, out_path: str, lm_fixed: str = "mcar",
) -> str:
    """Three flagship 2D heatmap slices through the 6D factorial cube, each
    fixing the other four factors at a moderate/representative level (bm=
    severe, n=200, nlab=30, lm=mcar, es=moderate, bd=opposing -- the same
    "severe" bias/moderate-effect severity used throughout the rest of
    this file's checks) so a reader can see two factors' effect on the
    PPI-corrected rate at a glance, the same way save_ppi_nlab_grid_plot
    does for N x N_lab alone. One ROW per eval type (build_ppi_factorial_
    sources now crosses continuous/likert), one COLUMN per slice, the same
    row-per-facet/column-per-slice convention save_ppi_nlab_grid_plot uses
    for its own eval-type faceting:
      1. N x N_lab (bm/lm/es/bd fixed) -- reproduces build_ppi_nlab_grid_
         sources' own heatmap as a consistency check, now inside the
         broader factorial's own data.
      2. bias_magnitude x label_mechanism (n/nlab/es/bd fixed) -- does a
         biased labeling PROCESS compound with judge bias severity. Always
         shows every label_mechanism level on its own axis regardless of
         `lm_fixed` -- this slice is the one place label_mechanism ISN'T
         fixed to a single value, so it needs no MNAR companion.
      3. effect_size x bias_direction (n/nlab/bm/lm fixed) -- the
         opposing/reinforcing asymmetry (save_ppi_power_direction_plot) at
         a fixed, moderate bias/label setting instead of the cross-eval-type
         line-plot framing.
    The pooled GLM (fit_ppi_factorial_model) is the rigorous backing for
    what these slices show; these are the visual/narrative complement.

    `lm_fixed` sets which label_mechanism level slices 1 and 3 fix (default
    "mcar", the expected-use-case/good-experimental-design view). Call this
    a second time with lm_fixed="mnar_strong" (see run()'s factorial_check
    block) for a separate companion figure showing the same two slices under
    the worst-case labeling mechanism -- kept as a distinct file rather than
    folded in, same rationale as the alignment sweep's MCAR/MNAR split (see
    PPIAlignmentSweepResult.lm's docstring)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No PPI factorial results to plot.")
    df = _ppi_factorial_dataframe(results)
    eval_types = sorted(df["et"].unique())

    _CATEGORICAL_FACTORS = ("bm", "lm", "es", "bd")
    slices = [
        ("n", "nlab", dict(bm="severe", lm=lm_fixed, es="moderate", bd="opposing")),
        ("bm", "lm", dict(n=200, nlab=30, es="moderate", bd="opposing")),
        ("es", "bd", dict(n=200, nlab=30, bm="severe", lm=lm_fixed)),
    ]
    order = {
        "bm": ["none", "moderate", "severe"], "lm": ["mcar", "mnar_mild", "mnar_strong"],
        "es": ["null", "moderate", "large"], "bd": ["opposing", "reinforcing"],
        "n": PPI_FACTORIAL_N_VALUES, "nlab": PPI_FACTORIAL_NLAB_VALUES,
    }

    fig, axes = plt.subplots(
        len(eval_types), len(slices), figsize=(6.0 * len(slices), 5.0 * len(eval_types)), squeeze=False,
    )
    for row, et in enumerate(eval_types):
        et_df = df[df["et"] == et]
        for col, (x_field, y_field, fixed) in enumerate(slices):
            ax = axes[row][col]
            sub = et_df
            for k, v in fixed.items():
                sub = sub[sub[k] == v]
            x_values = [v for v in order[x_field] if v in set(sub[x_field])]
            y_values = [v for v in order[y_field] if v in set(sub[y_field])]
            grid = np.full((len(y_values), len(x_values)), np.nan)
            for _, r in sub.iterrows():
                if r["n_reps"] == 0 or r[x_field] not in x_values or r[y_field] not in y_values:
                    continue
                grid[y_values.index(r[y_field]), x_values.index(r[x_field])] = r["rate_ppi"]

            vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
            im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
            for i in range(len(y_values)):
                for j in range(len(x_values)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                        )
            x_tick_labels = [_pretty_factorial_level(v) if x_field in _CATEGORICAL_FACTORS else str(v) for v in x_values]
            y_tick_labels = [_pretty_factorial_level(v) if y_field in _CATEGORICAL_FACTORS else str(v) for v in y_values]
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_tick_labels, rotation=20 if x_field in _CATEGORICAL_FACTORS else 0)
            ax.set_yticks(range(len(y_values)))
            ax.set_yticklabels(y_tick_labels)
            ax.set_xlabel(_PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field))
            ax.set_ylabel(_PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field))
            x_name = _PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field)
            y_name = _PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field)
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
            ax.set_title(f"[{et.capitalize()}] {x_name} × {y_name}\n({fixed_str})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    lm_suffix = "" if lm_fixed == "mcar" else f" [slices 1,3 at label_mechanism={lm_fixed}]"
    fig.suptitle(
        f"PPI-Corrected Rejection Rate: Full-Factorial Slices (nominal {_alpha_label(alpha)}){lm_suffix}",
        y=1.02, fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode, alignment view: false-positive rate (uncorrected vs PPI-corrected)
# as a function of REALIZED judge-human alignment -- derived from build_ppi_
# factorial_sources' own es="null" cells (which cross llm_noise x
# bias_magnitude, among the other factors) rather than a separate simulation
# run -- see scenarios.synthetic's build_ppi_factorial_sources/
# measure_judge_alignment for the design and the "percent aligned conflates
# noise and bias" motivation.
# ---------------------------------------------------------------------------

@dataclass
class PPIAlignmentSweepResult:
    name: str
    eval_type: str
    noise: float
    bias_label: str
    """One of PPI_FACTORIAL_BIAS_MAGNITUDES' keys (none/moderate/severe) --
    see _alignment_regime for how this maps to a bucketed-view regime:
    "none" -> no_bias, "severe" -> bias_present, "moderate" -> excluded
    from the two-regime bucketed print/plot views (still present in the
    raw per-cell CSV)."""
    lm: str
    """One of PPI_FACTORIAL_LABEL_MECHANISMS' keys (mcar/mnar_mild/
    mnar_strong) for the cell this alignment measurement was derived from.
    Not used by the bucketing itself (measure_judge_alignment doesn't
    depend on label_mechanism) -- carried through purely so callers can
    split the sweep into separate MCAR-only/MNAR-only views (see run()'s
    factorial_check block), since pooling every label_mechanism into one
    plot hides that MNAR's false-positive rate at a given alignment level
    can be much worse than MCAR's at that same level."""
    alignment_metrics: dict[str, float]
    """Raw (not rescaled) alignment metrics from measure_judge_alignment --
    e.g. {"pearson_r": ..., "spearman_r": ...} for continuous, or
    {"weighted_kappa": ..., "spearman_r": ..., "percent_agreement": ...} for
    likert. Which one to bucket/plot by is a presentational choice made by
    callers (see _ALIGNMENT_VIEWS), not baked in here -- e.g. likert gets
    reported/plotted against BOTH weighted_kappa and spearman_r, since some
    work recommends one over the other for Likert-scored judges."""
    n_reps: int
    rejects_llm_only: int
    """Uncorrected false-positive count -- this sweep's es=0.0 throughout, so
    rejects_llm_only/n_reps IS the uncorrected Type-I rate directly (no
    effect-size framing needed the way _COMPARISON_METHODS' other consumers
    have to guard against, e.g. save_ppi_null_comparison_plot's docstring)."""
    rejects_ppi: int
    n_failed: int


def _alignment_regime(bias_label: str) -> str:
    """The qualitative "why is this cell at this alignment level" split the
    whole sweep is built to make visible: "no_bias" (bias_label == "none" --
    whatever alignment level this cell landed at came purely from llm_noise)
    vs. "bias_present" (bias_label == "severe" ONLY). Returns "excluded" for
    "moderate" -- filtered out of print_ppi_alignment_sweep_report/save_ppi_
    alignment_sweep_plot's regime loops entirely (both only ever iterate
    ("no_bias", "bias_present")), not shown as a third regime.

    "bias_present" uses severe bias alone, not moderate+severe pooled
    together: pooling both magnitudes averages away the sharpest version of
    this view's own point -- the point being that a judge can read as
    near-perfectly aligned by IRR while still being catastrophically
    miscalibrated if used naively, which comes through far more sharply
    using severe alone than a pooled average would. Moderate-bias cells are
    not dropped from the underlying data (they still run as part of the
    full factorial grid, feed fit_ppi_factorial_model/
    save_ppi_factorial_heatmap_plot/save_results_artifacts_ppi_alignment_
    sweep's raw per-cell CSV as before) -- only excluded from this bucketed
    print/plot view's two-regime comparison."""
    if bias_label == "none":
        return "no_bias"
    if bias_label == "severe":
        return "bias_present"
    return "excluded"


_PPI_ALIGNMENT_REGIME_LABEL = {"no_bias": "no judge bias (noise only)", "bias_present": "bias present"}
_PPI_ALIGNMENT_BUCKET_WIDTH = 10


def _metric_pct(raw_value: float) -> float:
    """Rescale a correlation-or-kappa-like alignment metric (nominally
    -1..1, essentially always >= 0 for a judge at least weakly related to
    truth) to a 0-100 bucketing percentage, clipping at 0 on the rare
    below-chance/negative draw."""
    return float(np.clip(raw_value, 0.0, 1.0) * 100.0)


def _alignment_bucket(pct: float, width: int = _PPI_ALIGNMENT_BUCKET_WIDTH) -> tuple[int, str]:
    """(bucket_lo, label) for a 0-100 alignment percentage, in `width`-point
    buckets -- e.g. pct=73.2 -> (70, "70-80%") at width=10. Clamps into
    [0, 100-width] first so a pct of exactly 100.0 lands in the last bucket
    instead of spilling into a width-0 "100-110%" one."""
    lo = int(pct // width) * width
    lo = max(0, min(lo, 100 - width))
    return lo, f"{lo}-{lo + width}%"


def _kappa_band(x: float) -> str:
    """Landis & Koch (1977) benchmarks for kappa-type statistics -- same
    bands evalstats.alignment._interpret_kappa uses for the public alignment
    report, reused here so a bucket's qualitative label matches what a user
    would see calling judge_alignment() on the same kind of judge."""
    if x < 0:
        return "poor"
    if x <= 0.20:
        return "slight"
    if x <= 0.40:
        return "fair"
    if x <= 0.60:
        return "moderate"
    if x <= 0.80:
        return "substantial"
    return "almost perfect"


def _corr_band(x: float) -> str:
    """Cohen (1988) conventions for correlation-coefficient magnitude -- same
    bands evalstats.alignment._interpret_corr uses."""
    a = abs(x)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


def _icc_band(x: float) -> str:
    """Koo & Li (2016) benchmarks for ICC magnitude -- same bands
    evalstats.alignment._interpret_icc uses."""
    if x < 0.50:
        return "poor"
    if x < 0.75:
        return "moderate"
    if x < 0.90:
        return "good"
    return "excellent"


_ALIGNMENT_VIEWS = [
    ("continuous", "pearson_r", "Pearson r", _corr_band, "Cohen, 1988", "r"),
    ("continuous", "icc_21", "ICC(2,1)", _icc_band, "Koo & Li, 2016", "ICC"),
    ("likert", "weighted_kappa", "weighted κ", _kappa_band, "Landis & Koch, 1977", "κ"),
    ("likert", "spearman_r", "Spearman r", _corr_band, "Cohen, 1988", "ρ"),
    ("likert", "icc_21", "ICC(2,1)", _icc_band, "Koo & Li, 2016", "ICC"),
    ("binary", "kappa", "Cohen's κ", _kappa_band, "Landis & Koch, 1977", "κ"),
]
"""The (eval_type, metric, display_label, qualitative-band function,
citation, symbol) views the alignment sweep reports/plots -- one per
eval_type for the metric most commonly reported for that data type in
practice (Pearson r for continuous, weighted Cohen's kappa for likert),
PLUS a second view of likert bucketed by Spearman r, since some work
recommends rank correlation over weighted kappa for Likert-scored judges
(it doesn't require picking a tie-weighting scheme -- though empirically,
Spearman turned out MORE prone to masking a biased judge than weighted
kappa is, not less: at "large"/"almost perfect" alignment, Spearman's
bucket showed materially higher uncorrected false-positive rates than
kappa's -- both being rank/order-based to some degree, but kappa's
near-exact-match requirement is more bias-sensitive than pure rank
preservation is). Continuous and likert ALSO each get an ICC(2,1) view
(_icc_21/_icc_band) -- unlike Pearson r (invariant to any affine rescaling)
or weighted kappa (only bias-sensitive insofar as bias shifts items across
the fixed category grid), ICC(2,1) is absolute-agreement, not just
relative-agreement: a judge that's additively biased but otherwise
low-noise can read as well-aligned on the other views while ICC(2,1)
correctly marks it down. This makes ICC's bucket the sharpest version of
this whole sweep's point (see measure_judge_alignment's docstring) -- a
biased-but-precise judge should land in a LOWER ICC bucket even at a
region of the noise grid where Pearson r/weighted kappa would call it
"almost perfect". Binary gets (unweighted) Cohen's kappa only -- no
correlation-type view (on 2x2 binary data Pearson/Spearman/Kendall's
tau-b/phi/MCC are all essentially the same statistic, see measure_judge_
alignment's binary branch, so adding one would be redundant) and no ICC
view either, matching evalstats/alignment.py's own choice not to compute
ICC(2,1) for binary judges -- its variance-decomposition doesn't map
cleanly onto two nominal categories the way it does onto an ordinal/
numeric scale, and kappa already IS an absolute-agreement statistic on
binary data (unlike Pearson r on continuous/likert data), so there's no
"correlation looks great but absolute agreement doesn't" gap for ICC to
expose there. `symbol` is the
conventional single-character notation used in bucket subplot titles (e.g.
"κ=0.40-0.50"). Drives print_ppi_alignment_sweep_report/save_ppi_alignment_
sweep_plot/the human-human companion uniformly -- one call per entry -- so
all three stay in sync and none can silently drift out of step with the
others. NOTE: the human-human companion (run_human_human_alignment_sweep)
does not yet cover binary -- measure_human_human_alignment has no binary
branch -- so binary's alignment plot has no human-human reference bars;
this view still renders correctly (both consumer functions skip views with
no matching rows), it's just a known gap, not a crash."""


def build_ppi_alignment_results_from_factorial(
    factorial_sources: list[JudgeBiasSource], factorial_results: list[PPIComparisonResult],
    n_align_mc: int, seed: int = 0,
) -> list[PPIAlignmentSweepResult]:
    """Derives the judge-human alignment-bucketed view (_ALIGNMENT_VIEWS) from
    build_ppi_factorial_sources' own es="null" cells, rather than a separate
    simulation run against a separate, narrower source grid -- `factorial_results`
    should be the FULL pooled-across-_COMPARISON_METHODS factorial results,
    covering every llm_noise level (not the noise=0.20-only subset fed to
    fit_ppi_factorial_model/save_ppi_factorial_heatmap_plot -- see
    _PPI_FACTORIAL_FORMULA's docstring for why those two views need disjoint
    slices of the same data).

    Realized alignment (measure_judge_alignment) depends only on
    (eval_type, llm_noise, bias_delta, likert_max) -- not on N, N_lab,
    label_mechanism, or bias_direction (bias_direction is moot here anyway:
    build_ppi_factorial_sources skips bias_direction="reinforcing" whenever
    es="null") -- so this memoizes that measurement across the handful of
    distinct (eval_type, llm_noise, bias_delta, likert_max) combinations the
    null-effect subset actually contains (2 eval types x 11 noise levels x 3
    bias magnitudes = 66, at the default noise grid), instead of recomputing
    an identical large-sample calibration draw once per factorial cell
    (~1,584 of them at the default grid). This is also what gives each
    alignment bucket its richer N/N_lab/label_mechanism spread versus the
    original standalone sweep's one-baseline-value-each design: every
    es="null" cell sharing a (eval_type, llm_noise, bias_delta, likert_max)
    combo lands in the SAME bucket regardless of its own N/N_lab/
    label_mechanism, so a bucket now pools rejection counts across whichever
    of those combinations survive build_ppi_factorial_sources' n_lab>=n skip
    at that noise/bias/eval_type slice."""
    by_name = {sc.name: sc for sc in factorial_sources}
    align_cache: dict[tuple, dict] = {}
    results: list[PPIAlignmentSweepResult] = []
    for r in factorial_results:
        d = _parse_ppi_factorial_name(r.name)
        if d["es"] != "null":
            continue
        sc = by_name[r.name]
        key = (sc.eval_type, round(sc.llm_noise, 8), round(sc.bias_delta, 8), sc.likert_max)
        if key not in align_cache:
            align_cache[key] = measure_judge_alignment(sc, n_mc=n_align_mc, seed=seed + len(align_cache))
        results.append(PPIAlignmentSweepResult(
            name=r.name, eval_type=d["et"], noise=d["noise"], bias_label=d["bm"], lm=d["lm"],
            alignment_metrics=align_cache[key],
            n_reps=r.n_reps, rejects_llm_only=r.rejects_llm_only, rejects_ppi=r.rejects_ppi,
            n_failed=r.n_failed,
        ))
    return results


def print_ppi_alignment_sweep_report(results: list[PPIAlignmentSweepResult], alpha: float) -> None:
    """One table per _ALIGNMENT_VIEWS entry (uncorrected/PPI-corrected
    false-positive rate by alignment bucket x regime) -- the console/log-file
    counterpart to save_ppi_alignment_sweep_plot's bar charts, in text form.
    Each bucket's row also prints that bucket's qualitative interpretation
    band (Landis & Koch for kappa, Cohen for correlations), evaluated at the
    bucket's midpoint."""
    if not results:
        print("\n  (no PPI alignment-sweep results)")
        return
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- ALIGNMENT SWEEP "
          f"(false-positive rate vs. realized judge-human alignment)\n"
          f"  {len(results)} cells, {len({r.eval_type for r in results})} eval type(s); nominal alpha={alpha}\n{'='*96}\n\n"
          f"  READ THE WITHIN-BUCKET COMPARISON, not the across-bucket trend: within the 'bias present' regime,\n"
          f"  alignment here is driven almost entirely by judge NOISE, not bias (a pure additive bias barely\n"
          f"  moves these metrics) -- so higher buckets mostly mean lower noise, and lower noise makes the SAME\n"
          f"  fixed bias easier to detect, which is why 'bias present' rows can rise across buckets. That's not\n"
          f"  alignment causing miscalibration -- see measure_judge_alignment's docstring for the full mechanism.")
    for et, metric, display, band_fn, band_source, _symbol in _ALIGNMENT_VIEWS:
        et_rows = [r for r in results if r.eval_type == et and metric in r.alignment_metrics]
        if not et_rows:
            continue
        print(f"\n  [{et}, bucketed by {display} ({band_source} bands)]")
        print(f"    {'bucket':<10} {'band':<16} {'regime':<22} {'n_cells':>7} {'uncorrected':>12} {'ppi-corrected':>14}")
        buckets = sorted({_alignment_bucket(_metric_pct(r.alignment_metrics[metric])) for r in et_rows})
        for lo, label in buckets:
            band = band_fn((lo + _PPI_ALIGNMENT_BUCKET_WIDTH / 2) / 100.0)
            for regime in ("no_bias", "bias_present"):
                cells = [
                    r for r in et_rows
                    if _alignment_bucket(_metric_pct(r.alignment_metrics[metric])) == (lo, label)
                    and _alignment_regime(r.bias_label) == regime
                ]
                if not cells:
                    continue
                n_reps_tot = sum(c.n_reps for c in cells)
                unc_rate = sum(c.rejects_llm_only for c in cells) / n_reps_tot if n_reps_tot else float("nan")
                ppi_rate = sum(c.rejects_ppi for c in cells) / n_reps_tot if n_reps_tot else float("nan")
                print(f"    {label:<10} {band:<16} {_PPI_ALIGNMENT_REGIME_LABEL[regime]:<22} {len(cells):>7d} "
                      f"{unc_rate:>12.3f} {ppi_rate:>14.3f}")
    print()


def save_results_artifacts_ppi_alignment_sweep(
    *, results: list[PPIAlignmentSweepResult], alpha: float, out_dir: str, run_stem: str,
    human_human_rows: list[dict] | None = None,
) -> list[str]:
    """`human_human_rows` (run_human_human_alignment_sweep's output), if
    given, is appended as a trailer section to the SAME .log file -- see
    print_human_human_alignment_report's docstring for why it isn't its own
    section header. The CSV has one column per possible metric (blank where
    an eval type doesn't compute it) rather than a single "primary" column,
    so the raw data supports re-bucketing by any metric later without
    rerunning the simulation."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    # "kappa" (binary's own unweighted Cohen's kappa -- see measure_judge_
    # alignment's docstring) must stay in this column list -- it's binary's
    # primary alignment metric, and omitting it would silently break any
    # attempt to re-derive save_ppi_alignment_sweep_plot's binary/kappa view
    # from an already-saved CSV without rerunning the simulation.
    metric_cols = ["pearson_r", "spearman_r", "weighted_kappa", "kappa", "icc_21", "percent_agreement"]
    csv_path = out_base / f"{run_stem}_ppi_alignment_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "eval_type", "noise", "bias_label", "lm", "regime", *metric_cols,
            "n_reps", "rate_llm_only", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.eval_type, f"{r.noise:.4f}", r.bias_label, r.lm, _alignment_regime(r.bias_label),
                *[f"{r.alignment_metrics[c]:.4f}" if c in r.alignment_metrics else "" for c in metric_cols],
                r.n_reps,
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_alignment_sweep_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_alignment_sweep_report(results, alpha=alpha)
        if human_human_rows:
            print_human_human_alignment_report(human_human_rows)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    output_paths = [str(csv_path), str(summary_path)]
    if human_human_rows:
        output_paths.append(save_human_human_alignment_csv(rows=human_human_rows, out_dir=out_dir, run_stem=run_stem))
    return output_paths


def save_ppi_alignment_sweep_plot(
    *, results: list[PPIAlignmentSweepResult], eval_type: str, metric: str, display_label: str,
    band_fn, band_source: str, symbol: str, alpha: float, out_path: str,
) -> str:
    """Array of bar charts for ONE (eval_type, metric) view from
    _ALIGNMENT_VIEWS -- one COLUMN per alignment bucket present, each panel
    ALWAYS showing both (no_bias, bias_present) regimes x (uncorrected,
    PPI-corrected) arms (4 bars total), even when one regime has zero cells
    in that bucket -- drawn as a zero-height bar with "(n=0)" in its tick
    label, rather than omitted, so every panel has the SAME x-axis layout
    and bar width instead of the layout stretching/shrinking per panel based
    on which regimes happen to have data (visually misleading side by side).
    Every panel also shares the same fixed y-axis (0-1.05) for the same
    reason -- direct visual comparability across buckets AND across the
    other _ALIGNMENT_VIEWS figures this is called for.

    Each bucket's title is the metric's own notation over its range (e.g.
    "κ=0.40-0.50"), with its qualitative interpretation band underneath
    (band_fn, evaluated at the bucket midpoint -- e.g. "substantial" per
    Landis & Koch, 1977) -- publication-style notation rather than a raw
    percentage, so a reader isn't left to separately look up what the number
    means for this metric.

    Called once per _ALIGNMENT_VIEWS entry (see run()'s factorial_check
    block) -- deliberately separate figures rather than one combined grid,
    since continuous/likert use different metrics with different natural
    ranges and interpretation bands, and likert gets shown against two
    different metrics that deserve their own titles rather than sharing one.

    Error bars are the 95% Wilson score interval on each bar's pooled
    rejects/n_reps (same convention/caveat as save_ppi_null_comparison_plot:
    exact for a truly homogeneous pool, a standard mild simplification if the
    (noise, bias) cells landing in the same bucket/regime aren't perfectly
    identically calibrated). The within-bucket-vs-across-bucket reading
    caveat (see measure_judge_alignment's docstring) is deliberately left out
    of the figure itself -- that belongs in the surrounding write-up, not
    baked into the image."""
    import matplotlib.pyplot as plt

    et_rows = [r for r in results if r.eval_type == eval_type and metric in r.alignment_metrics]
    if not et_rows:
        raise ValueError(f"No PPI alignment-sweep results for eval_type={eval_type!r}, metric={metric!r}.")
    bar_width = 0.35
    group_gap = 0.25
    regimes = ("no_bias", "bias_present")
    arm_colors = {"llm_only": "#e7298a", "ppi": "#FFD400"}
    arm_edgecolors = {"llm_only": "none", "ppi": "#8a6d00"}
    arm_labels = {"llm_only": "uncorrected", "ppi": "PPI-corrected"}

    buckets = sorted({_alignment_bucket(_metric_pct(r.alignment_metrics[metric])) for r in et_rows})

    fig, axes = plt.subplots(1, len(buckets), figsize=(2.9 * len(buckets), 4.3), squeeze=False, sharey=True)
    for col, (lo, label) in enumerate(buckets):
        ax = axes[0][col]
        ax.axhline(
            alpha, color="black", ls="--", lw=1.0, alpha=0.6, zorder=1,
            label="nominal α" if col == 0 else None,
        )
        xticks, xticklabels = [], []
        for gi, regime in enumerate(regimes):  # ALWAYS both, even if empty
            cells = [
                r for r in et_rows
                if _alignment_bucket(_metric_pct(r.alignment_metrics[metric])) == (lo, label)
                and _alignment_regime(r.bias_label) == regime
            ]
            n_reps_tot = sum(c.n_reps for c in cells)
            for ai, arm in enumerate(("llm_only", "ppi")):
                x = gi * (2 * bar_width + group_gap) + ai * bar_width
                if n_reps_tot == 0:
                    continue  # nothing to draw; tick/slot still allocated below
                rejects_tot = sum(getattr(c, f"rejects_{arm}") for c in cells)
                rate = rejects_tot / n_reps_tot
                lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
                ax.bar(
                    x, rate, width=bar_width, color=arm_colors[arm], edgecolor=arm_edgecolors[arm],
                    linewidth=1.0, zorder=2,
                    label=arm_labels[arm] if (col == 0 and gi == 0) else None,
                )
                ax.errorbar(
                    x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4,
                )
            mid = gi * (2 * bar_width + group_gap) + bar_width / 2
            xticks.append(mid)
            xticklabels.append(f"{_PPI_ALIGNMENT_REGIME_LABEL[regime]}\n(n={len(cells)})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlim(-0.3, (2 * bar_width + group_gap) * len(regimes) - group_gap + 0.05)
        ax.set_ylim(0.0, 1.05)
        band = band_fn((lo + _PPI_ALIGNMENT_BUCKET_WIDTH / 2) / 100.0)
        ax.set_title(f"{symbol}={lo / 100:.2f}-{(lo + _PPI_ALIGNMENT_BUCKET_WIDTH) / 100:.2f}\n({band})", fontsize=10)
        if col == 0:
            ax.set_ylabel("False positive rate", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    fig.suptitle(f"{eval_type.capitalize()}: False-Positive Rate by Judge-Human Alignment ({display_label})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_human_human_alignment_sweep(n_mc: int = 20_000, seed: int = 42) -> list[dict]:
    """Companion measurement (not a hypothesis-test sweep -- see
    scenarios.synthetic.measure_human_human_alignment's docstring): for each
    eval_type x PPI_ALIGNMENT_HUMAN_NOISE_LEVELS combination, the realized
    alignment (every metric measure_human_human_alignment computes) between
    two independently-noisy synthetic human raters. Used as context alongside
    the main alignment sweep's judge-vs-human buckets, not merged into the
    same plot -- see save_human_human_alignment_plot."""
    rows = []
    for et in ("continuous", "likert"):
        for i, hn in enumerate(PPI_ALIGNMENT_HUMAN_NOISE_LEVELS):
            m = measure_human_human_alignment(et, hn, n_mc=n_mc, seed=seed + i)
            rows.append({"eval_type": et, "human_noise": hn, "metrics": m})
    return rows


def print_human_human_alignment_report(rows: list[dict]) -> None:
    """Text counterpart to save_human_human_alignment_plot -- printed as a
    trailer to print_ppi_alignment_sweep_report's own log, not a separate
    section header, since it's context for reading that report's buckets,
    not an independent result. One line per _ALIGNMENT_VIEWS entry, matching
    the main report's per-view breakdown."""
    if not rows:
        return
    print("  -- Human-human alignment range (context, NOT a claimed ceiling) --")
    for et, metric, display, _band_fn, _src, _symbol in _ALIGNMENT_VIEWS:
        et_rows = sorted(
            [r for r in rows if r["eval_type"] == et and metric in r["metrics"]], key=lambda r: r["human_noise"],
        )
        if not et_rows:
            continue
        vals = ", ".join(f"noise={r['human_noise']:.2f}: {_metric_pct(r['metrics'][metric]):.0f}%" for r in et_rows)
        print(f"    [{et}, {display}] {vals}")
    print()


def save_human_human_alignment_csv(*, rows: list[dict], out_dir: str, run_stem: str) -> str:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metric_cols = ["pearson_r", "spearman_r", "weighted_kappa", "icc_21", "percent_agreement"]
    csv_path = out_base / f"{run_stem}_human_human_alignment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "human_noise", *metric_cols])
        for r in rows:
            writer.writerow([
                r["eval_type"], f"{r['human_noise']:.4f}",
                *[f"{r['metrics'][c]:.4f}" if c in r["metrics"] else "" for c in metric_cols],
            ])
    print(f"Saved results: {csv_path}")
    return str(csv_path)


def save_human_human_alignment_plot(*, rows: list[dict], out_path: str) -> str:
    """Small companion figure: realized human-human alignment (%) across
    PPI_ALIGNMENT_HUMAN_NOISE_LEVELS, one panel per _ALIGNMENT_VIEWS entry
    (matching the main sweep's three views) -- a rough benchmark range for
    reading the main alignment sweep's buckets against (a judge landing well
    below where two independent humans typically land with each other is a
    materially different finding than one landing within that range).
    Deliberately a RANGE across several human_noise values, not one
    bar/number -- there's no canonical "true" human-human noise level to
    assert here, and presenting a single anchored value would repeat the
    same overfitting-to-one-number problem already avoided in the main
    sweep's design."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No human-human alignment rows to plot.")
    views = [
        (et, metric, display) for et, metric, display, _bf, _src, _symbol in _ALIGNMENT_VIEWS
        if any(r["eval_type"] == et and metric in r["metrics"] for r in rows)
    ]
    fig, axes = plt.subplots(1, len(views), figsize=(3.6 * len(views), 3.6), squeeze=False)
    for col, (et, metric, display) in enumerate(views):
        ax = axes[0][col]
        et_rows = sorted(
            [r for r in rows if r["eval_type"] == et and metric in r["metrics"]], key=lambda r: r["human_noise"],
        )
        x = np.arange(len(et_rows))
        pcts = [_metric_pct(r["metrics"][metric]) for r in et_rows]
        ax.bar(x, pcts, width=0.6, color="#4d4d4d", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"noise={r['human_noise']:.2f}" for r in et_rows], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"{et.capitalize()} ({display})", fontsize=10)
        ax.set_ylabel("Alignment %" if col == 0 else "")
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
        for xi, pct in zip(x, pcts):
            ax.text(xi, pct + 1.5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Human-Human Alignment Range (two independently-noisy synthetic raters)\n"
        "context for the judge-alignment sweep's buckets -- not a claimed ceiling",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode, binary factorial: build_ppi_factorial_sources_binary's 7-factor
# cross (bias_magnitude x N x N_lab x label_mechanism x effect_size x
# bias_direction x llm_noise), analyzed the same two ways the continuous/
# likert factorial is (a pooled binomial GLM, a curated set of 2D heatmap
# slices), plus the judge-human alignment-bucketed view -- see
# build_ppi_factorial_sources_binary's docstring for why binary crosses
# bias_magnitude/llm_noise as two independent factors (not one combined
# "severity" scale) and its own name/parser.
# ---------------------------------------------------------------------------

_PPI_FACTORIAL_BINARY_NAME_RE = re.compile(
    r"^fact\.binary\.bm=(?P<bm>[a-z]+)\.n=(?P<n>\d+)\.nlab=(?P<nlab>\d+)\.lm=(?P<lm>[a-z_]+)\.es=(?P<es>[a-z]+)\.bd=(?P<bd>[a-z]+)\.noise=(?P<noise>[\d.]+)$"
)


def _parse_ppi_factorial_binary_name(name: str) -> dict:
    m = _PPI_FACTORIAL_BINARY_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized binary factorial scenario name: {name!r}")
    d = m.groupdict()
    d["n"] = int(d["n"])
    d["nlab"] = int(d["nlab"])
    d["noise"] = float(d["noise"])
    return d


def _ppi_factorial_binary_dataframe(results: list[PPIComparisonResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = _parse_ppi_factorial_binary_name(r.name)
        rows.append({
            **d, "method": r.method, "n_reps": r.n_reps,
            "rejects_ppi": r.rejects_ppi, "fails_ppi": r.n_reps - r.rejects_ppi,
            "rate_ppi": r.rejects_ppi / r.n_reps if r.n_reps else float("nan"),
            "rejects_all_human": r.rejects_all_human, "rejects_human_subset": r.rejects_human_subset,
        })
    return pd.DataFrame(rows)


_PPI_FACTORIAL_BINARY_FORMULA = (
    "rejects_ppi + fails_ppi ~ "
    "C(bm, Treatment('none')) + C(n) + C(nlab) + C(lm, Treatment('mcar')) "
    "+ C(es, Treatment('null')) + C(bd, Treatment('opposing')) "
    "+ C(bm, Treatment('none')):C(es, Treatment('null')) "
    "+ C(bd, Treatment('opposing')):C(es, Treatment('null'))"
)
"""Binary analogue of _PPI_FACTORIAL_FORMULA -- no `et` term, since
build_ppi_factorial_sources_binary is always eval_type="binary" (nothing to
estimate a main effect for). llm_noise is deliberately NOT a term here, for
the exact same confound reason _PPI_FACTORIAL_FORMULA's docstring gives:
llm_noise only varies away from PPI_BINARY_NOISE_BASELINE on es="null"
cells, so any non-baseline noise level implies es="null" with perfect
collinearity against the es term already in the formula. `results` fed to
this function should be pre-filtered to noise=PPI_BINARY_NOISE_BASELINE
(see run()'s factorial_check_binary block) -- the full noise-swept
es="null" subset instead feeds build_ppi_alignment_results_from_factorial_
binary, which bypasses this GLM entirely."""


def fit_ppi_factorial_binary_model(results: list[PPIComparisonResult]) -> tuple[str, pd.DataFrame]:
    """Binary analogue of fit_ppi_factorial_model -- same quasi-complete-
    separation caveat at es="large" applies here (see that function's
    docstring)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = _ppi_factorial_binary_dataframe(results)
    fit = smf.glm(formula=_PPI_FACTORIAL_BINARY_FORMULA, data=df, family=sm.families.Binomial()).fit()
    return fit.summary().as_text(), df


def print_ppi_factorial_binary_report(
    results: list[PPIComparisonResult], alpha: float, label: str = _COMPARISON_METHODS_BINARY_LABEL,
    *, null_results_full: list[PPIComparisonResult] | None = None,
    raw_results_full: list[PPIComparisonResult] | None = None,
) -> None:
    """Binary analogue of print_ppi_factorial_report -- see its docstring
    for what `null_results_full`/`raw_results_full` do and why they're
    needed (the worst-cell/label_mechanism-x-noise numbers should NOT be
    confined to the single llm_noise=PPI_BINARY_NOISE_BASELINE level the
    GLM requires; the per-method table needs the RAW, never method-pooled,
    per-method rows)."""
    if not results:
        print("\n  (no PPI binary factorial results)")
        return
    summary_text, df = fit_ppi_factorial_binary_model(results)
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- BINARY FACTORIAL "
          f"(bias_magnitude x N x N_lab x label_mechanism x effect_size x bias_direction)\n"
          f"  {len(results)} cells, binary/{label}; nominal alpha={alpha}\n{'='*96}\n")
    print(summary_text)

    null_df = _ppi_factorial_binary_dataframe(null_results_full) if null_results_full is not None else df
    null_rows = null_df[null_df["es"] == "null"]
    if len(null_rows):
        worst = null_rows.loc[(null_rows["rate_ppi"] - alpha).abs().idxmax()]
        print(f"\n  Worst Type-I cell: rate={worst['rate_ppi']:.3f} (nominal alpha={alpha}) at "
              f"bm={worst['bm']} n={worst['n']} nlab={worst['nlab']} lm={worst['lm']} bd={worst['bd']} "
              f"noise={worst['noise']:.4f}")
        _print_ppi_factorial_lm_noise_table(null_rows, alpha)

    if raw_results_full is not None:
        raw_df = _ppi_factorial_binary_dataframe(raw_results_full)
        raw_null_rows = raw_df[raw_df["es"] == "null"]
        _print_ppi_factorial_method_lm_table(raw_null_rows, alpha)

    nonnull_rows = df[df["es"] != "null"].copy()
    if len(nonnull_rows):
        nonnull_rows["power_gap"] = (nonnull_rows["rejects_all_human"] - nonnull_rows["rejects_ppi"]) / nonnull_rows["n_reps"]
        worst_gap = nonnull_rows.loc[nonnull_rows["power_gap"].idxmax()]
        print(f"  Largest all_human-vs-ppi power gap: {worst_gap['power_gap']:.3f} at "
              f"bm={worst_gap['bm']} n={worst_gap['n']} nlab={worst_gap['nlab']} "
              f"lm={worst_gap['lm']} es={worst_gap['es']} bd={worst_gap['bd']}")
    print()


def save_results_artifacts_ppi_factorial_binary(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str,
    pooled_results: list[PPIComparisonResult] | None = None, label: str = _COMPARISON_METHODS_BINARY_LABEL,
    null_results_full: list[PPIComparisonResult] | None = None,
    raw_results_full: list[PPIComparisonResult] | None = None,
) -> list[str]:
    """Binary analogue of save_results_artifacts_ppi_factorial (including
    the `null_results_full`/`raw_results_full` passthroughs -- see that
    function's docstring)."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_factorial_binary_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "method", "bm", "n", "nlab", "lm", "es", "bd", "noise", "n_reps",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in results:
            d = _parse_ppi_factorial_binary_name(r.name)
            writer.writerow([
                r.name, r.method, d["bm"], d["n"], d["nlab"], d["lm"], d["es"], d["bd"], f"{d['noise']:.4f}", r.n_reps,
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    print(f"Saved results: {csv_path}")
    summary_path = out_base / f"{run_stem}_ppi_factorial_binary_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_factorial_binary_report(
            pooled_results, alpha=alpha, label=label,
            null_results_full=null_results_full, raw_results_full=raw_results_full,
        )
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_factorial_binary_heatmap_plot(
    *, results: list[PPIComparisonResult], alpha: float, out_path: str, lm_fixed: str = "mcar",
) -> str:
    """Binary analogue of save_ppi_factorial_heatmap_plot -- one ROW (always
    binary, no eval_type facet to cross), the same three slice idea: N x
    N_lab, bias_magnitude x label_mechanism, effect_size x bias_direction.
    `results` should be pre-filtered to noise=PPI_BINARY_NOISE_BASELINE (see
    run()'s factorial_check_binary block), the same way the continuous
    heatmap's caller filters to noise=0.20.

    `lm_fixed` -- see save_ppi_factorial_heatmap_plot's docstring; same
    "call twice, mcar then mnar_strong, separate files" pattern."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No PPI binary factorial results to plot.")
    df = _ppi_factorial_binary_dataframe(results)

    _CATEGORICAL_FACTORS = ("bm", "lm", "es", "bd")
    slices = [
        ("n", "nlab", dict(bm="severe", lm=lm_fixed, es="moderate", bd="opposing")),
        ("bm", "lm", dict(n=200, nlab=30, es="moderate", bd="opposing")),
        ("es", "bd", dict(n=200, nlab=30, bm="severe", lm=lm_fixed)),
    ]
    order = {
        "bm": list(PPI_BINARY_BIAS_MAGNITUDES.keys()), "lm": ["mcar", "mnar_mild", "mnar_strong"],
        "es": ["null", "moderate", "large"], "bd": ["opposing", "reinforcing"],
        "n": PPI_FACTORIAL_N_VALUES, "nlab": PPI_FACTORIAL_NLAB_VALUES,
    }

    fig, axes = plt.subplots(1, len(slices), figsize=(6.0 * len(slices), 5.0), squeeze=False)
    for col, (x_field, y_field, fixed) in enumerate(slices):
        ax = axes[0][col]
        sub = df
        for k, v in fixed.items():
            sub = sub[sub[k] == v]
        x_values = [v for v in order[x_field] if v in set(sub[x_field])]
        y_values = [v for v in order[y_field] if v in set(sub[y_field])]
        grid = np.full((len(y_values), len(x_values)), np.nan)
        for _, r in sub.iterrows():
            if r["n_reps"] == 0 or r[x_field] not in x_values or r[y_field] not in y_values:
                continue
            grid[y_values.index(r[y_field]), x_values.index(r[x_field])] = r["rate_ppi"]

        vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                        bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                    )
        x_tick_labels = [_pretty_factorial_level(v) if x_field in _CATEGORICAL_FACTORS else str(v) for v in x_values]
        y_tick_labels = [_pretty_factorial_level(v) if y_field in _CATEGORICAL_FACTORS else str(v) for v in y_values]
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_tick_labels, rotation=20 if x_field in _CATEGORICAL_FACTORS else 0)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel(_PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field))
        ax.set_ylabel(_PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field))
        x_name = _PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field)
        y_name = _PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field)
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
        ax.set_title(f"[Binary] {x_name} × {y_name}\n({fixed_str})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    lm_suffix = "" if lm_fixed == "mcar" else f" [slices 1,3 at label_mechanism={lm_fixed}]"
    fig.suptitle(
        f"PPI-Corrected Rejection Rate: Binary Factorial Slices (nominal {_alpha_label(alpha)}){lm_suffix}",
        y=1.02, fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_ppi_alignment_results_from_factorial_binary(
    factorial_sources: list[JudgeBiasSource], factorial_results: list[PPIComparisonResult],
    n_align_mc: int, seed: int = 0,
) -> list[PPIAlignmentSweepResult]:
    """Binary analogue of build_ppi_alignment_results_from_factorial --
    derives the alignment-bucketed view from build_ppi_factorial_sources_
    binary's own es="null" cells, which cross PPI_BINARY_NOISE_LEVELS x
    PPI_BINARY_BIAS_MAGNITUDES the same way the continuous/likert version's
    es="null" cells cross PPI_FACTORIAL_NOISE_LEVELS x PPI_FACTORIAL_BIAS_
    MAGNITUDES. Keyed the same way (eval_type, llm_noise, bias_delta,
    likert_max) for consistency, just with fewer distinct noise levels (9
    vs. 11) and one fewer bias level (3 vs. bm's role here). `bias_label` is
    the bias_magnitude label (none/moderate/severe) -- _alignment_regime
    checks for "none"/"severe" only (moderate excluded from the bucketed
    view, see that function's docstring), so this reuses that function
    unchanged."""
    by_name = {sc.name: sc for sc in factorial_sources}
    align_cache: dict[tuple, dict] = {}
    results: list[PPIAlignmentSweepResult] = []
    for r in factorial_results:
        d = _parse_ppi_factorial_binary_name(r.name)
        if d["es"] != "null":
            continue
        sc = by_name[r.name]
        key = (sc.eval_type, round(sc.llm_noise, 8), round(sc.bias_delta, 8), sc.likert_max)
        if key not in align_cache:
            align_cache[key] = measure_judge_alignment(sc, n_mc=n_align_mc, seed=seed + len(align_cache))
        results.append(PPIAlignmentSweepResult(
            name=r.name, eval_type="binary", noise=sc.llm_noise, bias_label=d["bm"], lm=d["lm"],
            alignment_metrics=align_cache[key],
            n_reps=r.n_reps, rejects_llm_only=r.rejects_llm_only, rejects_ppi=r.rejects_ppi,
            n_failed=r.n_failed,
        ))
    return results


def _ppi_wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate (ported from
    sim_type_i_calibration.py's ``_wilson_interval``)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return lo, hi


def _ppi_holm_rejections(pvals: list[tuple[tuple[str, str], float]], alpha: float = 0.05) -> set[tuple[str, str]]:
    """Rejected (scenario, test) cells under Holm-Bonferroni family-wise error
    control (ported from sim_type_i_calibration.py's ``_holm_rejections``)."""
    ordered = sorted(pvals, key=lambda x: x[1])
    m = len(ordered)
    rejected: set[tuple[str, str]] = set()
    for i, (cell, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        if p <= thresh:
            rejected.add(cell)
        else:
            break
    return rejected


def _fmt_ppi_rate(rate: float | None, flag2: float, flag3: float) -> str:
    if rate is None:
        return "  n/a  "
    s = f"{rate:.3f}"
    if rate > flag3:
        return s + "●●"
    if rate > flag2:
        return s + "● "
    return s + "  "


def print_ppi_report(results: list[PPIResult], alpha: float, regime: str = "") -> None:
    """Scenario x test calibration table, mirroring sim_type_i_calibration.py's
    ``_print_table``: tag-grouped scenario rows, one column per test, per-cell
    2-sigma/3-sigma inflation flags, a Wilson-CI miscalibration flag (dagger),
    a Holm-Bonferroni family-wise flag (double-dagger), and a SUMMARY section
    with flag counts plus per-test corrected/uncorrected max/mean/median --
    instead of one flat row per (scenario, test) cell and a single
    averaged-rate table.

    regime : str
        Optional label (e.g. "MCAR", "MNAR") appended to the header, used by
        run() to print separate MCAR/MNAR tables -- see JudgeBiasSource.
        label_mnar and the "MNAR is adversarial to PPI" discussion at that
        call site for why these are split rather than pooled."""
    if not results:
        print("\n  (no PPI results)")
        return

    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    n_reps = results[0].n_reps
    sigma = (alpha * (1 - alpha) / n_reps) ** 0.5 if n_reps > 0 else float("nan")
    flag2 = alpha + 2 * sigma
    flag3 = alpha + 3 * sigma

    width = 90
    bar = "-" * width
    dbar = "=" * width
    col_w = max(9, max((len(t) for t in tests), default=9) + 1)

    cell: dict[tuple[str, str], PPIResult] = {(r.name, r.test): r for r in results}
    scenario_order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in results:
        if r.name not in seen:
            seen.add(r.name)
            scenario_order.append((r.name, r.tag))
    name_w = max((len(name) for name, _tag in scenario_order), default=30) + 2

    tag_order: list[str] = []
    for _, tag in scenario_order:
        if tag not in tag_order:
            tag_order.append(tag)

    def rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.corrected_rejects / r.n_reps

    def uncorrected_rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.uncorrected_rejects / r.n_reps

    def wilson_outside(name: str, test: str) -> bool:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return False
        lo, hi = _ppi_wilson_interval(r.corrected_rejects, r.n_reps)
        return (alpha < lo) or (alpha > hi)

    print()
    print(dbar)
    print(f"  PVALUES (PPI-CORRECTED) -- TYPE I ERROR CALIBRATION{f' ({regime})' if regime else ''}")
    print(f"  n_reps={n_reps}  alpha={alpha}")
    print(f"  2σ flag (●): rate > {flag2:.3f}    3σ flag (●●): rate > {flag3:.3f}")
    print("  Wilson flag (†): 95% CI for rejection rate excludes alpha")
    print("  Holm flag (‡): exact binomial miscalibration survives family-wise correction")
    print(dbar)

    print()
    print(f"  {'Scenario':<{name_w}}" + "".join(f"{t:^{col_w}}" for t in tests))
    print(bar)

    pvals: list[tuple[tuple[str, str], float]] = []
    for name, _tag in scenario_order:
        for t in tests:
            r = cell.get((name, t))
            if r is not None and r.n_reps > 0:
                p = float(scipy_stats.binomtest(r.corrected_rejects, r.n_reps, alpha, alternative="two-sided").pvalue)
                pvals.append(((name, t), p))
    holm_bad = _ppi_holm_rejections(pvals, alpha=0.05)

    for tag in tag_order:
        print(f"\n[{tag}]")
        for name, sc_tag in scenario_order:
            if sc_tag != tag:
                continue
            row = f"  {name:<{name_w - 2}}"
            for t in tests:
                row += f" {_fmt_ppi_rate(rate_of(name, t), flag2, flag3):<{col_w - 1}}"
            n_failed_row = sum(cell[(name, t)].n_failed for t in tests if (name, t) in cell)
            if n_failed_row:
                row += f" ✗{n_failed_row}"
            if any(wilson_outside(name, t) for t in tests):
                row += "  †"
            if any((name, t) in holm_bad for t in tests):
                row += "‡"
            print(row)

    # -- Summary --------------------------------------------------------------
    print()
    print(bar)
    print("SUMMARY")
    print()

    n_scenarios = len(scenario_order)
    n_conditions = sum(1 for name, _tag in scenario_order for t in tests if (name, t) in cell)
    total_failed = sum(r.n_failed for r in results)

    all_corr = [rate_of(name, t) for name, _tag in scenario_order for t in tests]
    all_unc = [uncorrected_rate_of(name, t) for name, _tag in scenario_order for t in tests]

    flags2 = sum(1 for r in all_corr if r is not None and r > flag2)
    flags3 = sum(1 for r in all_corr if r is not None and r > flag3)
    wilson_miscal = sum(1 for name, _tag in scenario_order for t in tests if wilson_outside(name, t))
    nominal_miscal = sum(1 for _, p in pvals if p < 0.05)
    uncorrected_flags2 = sum(1 for r in all_unc if r is not None and r > flag2)
    uncorrected_flags3 = sum(1 for r in all_unc if r is not None and r > flag3)

    print(f"  Scenarios: {n_scenarios}  |  Tests: {len(tests)}  |  Conditions: {n_conditions}  |  Failed reps: {total_failed}")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {flags3}/{n_conditions}")
    print(f"  Wilson miscalibrated (alpha outside 95% CI): {wilson_miscal}/{n_conditions}")
    print(f"  Exact-binomial p<0.05 (corrected rates, unadjusted): {nominal_miscal}/{n_conditions}")
    print(f"  Holm-confirmed miscalibrated cells: {len(holm_bad)}/{n_conditions}")
    print()
    print("  Uncorrected aggregate")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {uncorrected_flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {uncorrected_flags3}/{n_conditions}")
    print()
    print(f"  {'Test':<14}  {'corr max':>9}  {'corr mean':>9}  {'corr med':>9}  {'unc max':>9}  {'unc mean':>9}  {'unc med':>9}")
    for t in tests:
        col_rates = [r for r in (rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        col_uncorrected = [r for r in (uncorrected_rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        if col_rates or col_uncorrected:
            corr_max = max(col_rates) if col_rates else float("nan")
            corr_mean = float(np.mean(col_rates)) if col_rates else float("nan")
            corr_median = float(np.median(col_rates)) if col_rates else float("nan")
            unc_max = max(col_uncorrected) if col_uncorrected else float("nan")
            unc_mean = float(np.mean(col_uncorrected)) if col_uncorrected else float("nan")
            unc_median = float(np.median(col_uncorrected)) if col_uncorrected else float("nan")
            print(
                f"  {t:<14}  {corr_max:>9.3f}  {corr_mean:>9.3f}  {corr_median:>9.3f}  "
                f"{unc_max:>9.3f}  {unc_mean:>9.3f}  {unc_median:>9.3f}"
            )
    print()


def latex_ppi_overall_summary(results: list[PPIResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-test corrected/uncorrected Type-I
    rate (each with its 95% MC band), averaged across scenarios, plus one
    corrected-rate column per sample size actually swept by the
    'sample_size' tag (n=60/100/200/400 -- the only scenarios that
    deliberately vary n; every other scenario shares the fixed baseline),
    appended to the right.

    PPIResult has no eval_type axis (scenarios are judge-bias/noise sweeps,
    not distribution shapes), so there's no "Eval types" column here.
    """
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    sizes_present = sorted({r.n for r in results if r.tag == "sample_size"})
    rows = []
    for t in tests:
        t_rows = [r for r in results if r.test == t]
        c_tot = sum(r.corrected_rejects for r in t_rows)
        u_tot = sum(r.uncorrected_rejects for r in t_rows)
        n_tot = sum(r.n_reps for r in t_rows)
        rate_c = c_tot / n_tot if n_tot > 0 else float("nan")
        rate_u = u_tot / n_tot if n_tot > 0 else float("nan")
        _, _, lo_c, hi_c = _mc_proportion_stats(c_tot, n_tot)
        _, _, lo_u, hi_u = _mc_proportion_stats(u_tot, n_tot)
        row = [
            escape_latex(t),
            f"{rate_c:.3f}" if np.isfinite(rate_c) else "-",
            f"${lo_c:.3f}\\text{{--}}{hi_c:.3f}$" if np.isfinite(lo_c) else "-",
            f"{rate_u:.3f}" if np.isfinite(rate_u) else "-",
            f"${lo_u:.3f}\\text{{--}}{hi_u:.3f}$" if np.isfinite(lo_u) else "-",
        ]
        for n in sizes_present:
            n_rows = [r for r in t_rows if r.tag == "sample_size" and r.n == n]
            c_n = sum(r.corrected_rejects for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            rate_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{rate_n:.3f}" if np.isfinite(rate_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"pvalues (PPI-corrected): corrected vs. uncorrected Type-I rate (nominal alpha={alpha}).",
        label="tab:pvalues_ppi_overall",
        columns=["Test", "Corrected", "95\\% MC band", "Uncorrected", "95\\% MC band"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def save_results_artifacts_ppi(*, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False, regime: str = "") -> list[str]:
    """Write the PPI run's results CSV (and LaTeX summary if `latex=True`)
    under out_dir. Returns the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "tag", "n", "test", "n_reps", "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate"])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.n, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_report(results, alpha=alpha, regime=regime)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_ppi_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


# ---------------------------------------------------------------------------
# Publication-facing label helpers for PPI mode's plots. Method.name
# (snake_case test identifiers) is fine for CSV columns and code but
# unreadable dropped cold into a figure, so every save_ppi_*_plot function
# below routes test names through _pretty_test. The factorial sweep's
# axis/tick labels go through _PPI_FACTORIAL_FACTOR_LABELS/
# _pretty_factorial_level the same way -- but its per-panel "what's held
# fixed" annotation deliberately stays in short raw-code form (bm=severe,
# not "Bias magnitude = Severe"): spelled out, four of those wrap across
# several lines and read as more cluttered, not less -- the terse form
# assumes the surrounding write-up defines what bm/n/nlab/lm/es/bd mean
# once, which is the same assumption the codes bm/lm/es/bd already require
# in cases/pvalues.py itself.
# ---------------------------------------------------------------------------

_PPI_PRETTY_TEST_NAMES: dict[str, str] = {
    TTEST.name: "t-test", TTEST_WELCH.name: "Welch's t-test",
    MWU.name: "Mann-Whitney U",
    WILCOXON.name: "Wilcoxon", PAIRED_T.name: "Paired t-test", BAYES_BOOTSTRAP.name: "Bayes bootstrap",
    BOOTSTRAP_T.name: "Bootstrap-t", MJ_FLOOR.name: "Tango score",
    MJ_FLOOR_FIXED_LAMBDA.name: "Tango score (fixed lambda)", ANOVA_IND.name: "ANOVA (indep.)",
    ANOVA_REP.name: "ANOVA (repeated)", FRIEDMAN.name: "Friedman",
    KRUSKAL.name: "Kruskal-Wallis", KRUSKAL_MNAR_EXPERIMENTAL.name: "Kruskal-Wallis (MNAR, experimental)",
    KRUSKAL_INFLUENCE.name: "Kruskal-Wallis (influence cov.)",
    KRUSKAL_INFLUENCE_FLOOR.name: "Kruskal-Wallis (influence cov. + floor)",
    LMM.name: "LMM", LMM_FACTORIAL.name: "LMM (factorial)", LMM_RUNS.name: "LMM (nested runs)",
    PPI_T_INTERVAL.name: "t-interval", PPI_LOGIT_T.name: "logit-t",
    PPI_WILSON.name: "Wilson", PPI_BONETT_PRICE.name: "Bonett-Price",
    PPI_BOOTSTRAP_T_SINGLE.name: "Bootstrap-t (single)",
    PPI_T_INTERVAL_SINGLE.name: "t-interval (single)", PPI_LOGIT_T_SINGLE.name: "logit-t (single)",
}


def _pretty_test(name: str) -> str:
    return _PPI_PRETTY_TEST_NAMES.get(name, name)


_PPI_FACTORIAL_FACTOR_LABELS: dict[str, str] = {
    "bm": "Bias magnitude", "n": "N (total items)", "nlab": "N_lab (labeled items)",
    "lm": "Label mechanism", "es": "Effect size", "bd": "Bias direction",
}
_PPI_FACTORIAL_LEVEL_LABELS: dict[str, str] = {
    "none": "None", "moderate": "Moderate", "severe": "Severe",
    "mcar": "MCAR", "mnar_mild": "MNAR (mild)", "mnar_strong": "MNAR (strong)",
    "null": "Null", "large": "Large", "opposing": "Opposing", "reinforcing": "Reinforcing",
}


def _pretty_factorial_level(value) -> str:
    return _PPI_FACTORIAL_LEVEL_LABELS.get(str(value), str(value))


def _alpha_label(alpha: float) -> str:
    return f"α = {alpha:g}"


def save_ppi_typeI_plot(*, results: list[PPIResult], alpha: float, out_path: str, nonstandard: bool = False, regime: str = "") -> str:
    """Grouped violin+strip of corrected vs. uncorrected Type-I rate, per
    test -- one gray violin (uncorrected) and one test-colored violin
    (corrected) side by side per test, each with its own jittered per-
    scenario dots overlaid. Replaces an earlier single-column jittered-
    scatter design (both corrected/uncorrected sharing one x per test,
    distinguished only by color/z-order) that made the SHAPE of each
    group's distribution hard to read once a test had more than a
    handful of scenarios -- the violin body shows that shape directly,
    while the dots keep the original "see every individual scenario,
    don't just trust an averaged rate" property that a plain box/violin
    alone would lose.

    Not drawn via seaborn's usual hue-dodge violin (see
    save_pairwise_reliability_violin_plot for that pattern elsewhere in
    this file): dodging by a 2-level "corrected/uncorrected" hue would
    force ONE color for every test's corrected violin, losing the
    per-test color coding get_method_color already gives every other PPI
    plot. Positioned by hand instead so "uncorrected" can stay uniformly
    gray while "corrected" keeps its test-specific color.

    nonstandard : bool
        When False (default), plots only the standard/textbook tests
        (excludes bayes_bootstrap/bootstrap_t/mj_floor). When True,
        plots ONLY those three bootstrap/CI-based methods instead -- see
        _PPI_NONSTANDARD_TESTS for why they're kept out of the main plot.
    """
    import matplotlib.pyplot as plt

    tests = _ppi_tests_present(results, nonstandard=nonstandard)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    rng = np.random.default_rng(0)
    unc_label_added = False
    all_rates: list[np.ndarray] = []

    violin_width = 0.30
    group_offset = 0.19

    def _violin_and_strip(x: float, values: np.ndarray, color: str, body_alpha: float, label: str | None) -> None:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        # A KDE-based violin body needs >= 3 points and some spread to be
        # well-defined (a constant-valued or 1-2-point group makes
        # matplotlib's internal gaussian_kde raise on a singular
        # bandwidth matrix) -- below that, the jittered dots alone still
        # communicate the group honestly, just without a shape to draw.
        if len(values) >= 3 and np.ptp(values) > 1e-9:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vp = ax.violinplot([values], positions=[x], widths=violin_width, showmedians=True, showextrema=False)
                body = vp["bodies"][0]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(body_alpha)
                vp["cmedians"].set_color(color)
                vp["cmedians"].set_linewidth(1.3)
                vp["cmedians"].set_alpha(0.9)
            except Exception:
                pass
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter, values, s=14, alpha=0.6, color=color,
            zorder=3, label=label, edgecolors="none",
        )

    for j, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t]
        rates_u = np.array([r.uncorrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        rates_c = np.array([r.corrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        all_rates.append(rates_u)
        all_rates.append(rates_c)

        _violin_and_strip(
            j - group_offset, rates_u, "#808080", 0.35,
            "Uncorrected (any test)" if not unc_label_added else None,
        )
        unc_label_added = True
        _violin_and_strip(j + group_offset, rates_c, get_method_color(t), 0.45, _pretty_test(t))

    ax.axhline(alpha, color="black", ls="--", lw=1.1, label=f"Nominal {_alpha_label(alpha)}")
    ax.set_xlim(-0.5, len(tests) - 0.5)
    scatter_max = np.nanmax(np.concatenate(all_rates)) if all_rates else float("nan")
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Observed rejection rate")
    ax.set_xlabel("Test")
    title_suffix = " -- Bootstrap/CI-Based Methods" if nonstandard else ""
    title_suffix += f" ({regime})" if regime else ""
    ax.set_title(
        f"PPI-Corrected Type-I Error, by Test{title_suffix}\n"
        "(gray = uncorrected, color = corrected; each dot: one judge-bias scenario)", fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    # Outside the axes (not "upper right" inside it, this plot's original
    # placement) -- with up to ~11 entries (uncorrected + one per test +
    # the nominal-alpha line), an inside legend routinely covered the
    # upper portion of whichever test's violin/dots happened to sit under
    # it. Same "outside right, vertically centered" convention already
    # used elsewhere in this file (see e.g. the power/width plots' legends).
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_factorial_typeI_violin_plot(
    *,
    two_group_results: list[PPIComparisonResult],
    omnibus_results: list[PPIComparisonResult],
    alpha: float,
    out_path: str,
    lm_filter: str | None = None,
) -> str:
    """Grouped violin+strip of corrected vs. uncorrected Type-I rate from the
    combined-factor factorial sweep's null cells -- ONE panel across all
    nine tests (the five _COMPARISON_METHODS two-group/paired tests, THEN
    the four _COMPARISON_METHODS_OMNIBUS tests, same order/columns
    save_ppi_typeI_plot's OFAT-sourced version uses), not two side-by-side
    panels. Same visual language as save_ppi_typeI_plot (gray = uncorrected,
    test-colored = corrected, one dot per scenario cell), scoped instead to
    PPIComparisonResult's (factorial sweep) fields and, via lm_filter, to a
    single labeling-mechanism regime.

    A single panel (not two side-by-side panels, two-group | omnibus) gives
    the factorial sweep's much denser scenario grid (every N x N_lab x
    bias x label_mechanism x noise combination, vs. build_judge_bias_
    sources' curated ~130-scenario OFAT catalog) the same single-panel,
    all-9-tests-together comparability save_ppi_typeI_plot already has, so
    the two plots read as directly comparable views of the same 9 tests
    rather than differently-shaped figures. Requires _COMPARISON_METHODS to
    include ttest alongside ttest_welch for this 9-test parity to hold --
    without it this panel would only ever show 8 of save_ppi_typeI_plot's 9
    tests.

    lm_filter exists because pooling MCAR and MNAR null cells into one
    violin hides that most of the mass sits right at nominal alpha under
    MCAR, with only a handful of MNAR cells (concentrated in mwu/kruskal,
    under mnar_strong/mnar_mild + opposing bias direction + low noise)
    producing a long right tail -- see run()'s factorial-check block for
    the two-call (mcar + mnar) convention this establishes, intended to
    make MCAR the headline figure and MNAR an explicit, separately-labeled
    stress test rather than a silent contributor to one pooled
    distribution: two separate saved figures, one per lm_filter value.

    lm_filter : {"mcar", "mnar", None}
        "mcar" keeps only MCAR-labeled null cells. "mnar" pools
        mnar_mild + mnar_strong. None keeps every labeling mechanism (the
        original, undifferentiated view) -- provided for completeness, not
        expected to be anyone's headline choice.
    omnibus_results : may be empty (e.g. factorial_omnibus=False) -- the
        four omnibus columns are simply omitted rather than drawn empty.
    """
    import matplotlib.pyplot as plt

    def _filter_null(results: list[PPIComparisonResult]) -> list[PPIComparisonResult]:
        out = []
        for r in results:
            d = _parse_ppi_factorial_name(r.name)
            if d["es"] != "null":
                continue
            if lm_filter == "mcar" and d["lm"] != "mcar":
                continue
            if lm_filter == "mnar" and d["lm"] not in ("mnar_mild", "mnar_strong"):
                continue
            out.append(r)
        return out

    have_omnibus = bool(omnibus_results)
    methods = list(_COMPARISON_METHODS) + (list(_COMPARISON_METHODS_OMNIBUS) if have_omnibus else [])
    results = _filter_null(two_group_results) + (_filter_null(omnibus_results) if have_omnibus else [])

    fig, ax = plt.subplots(figsize=(2.0 * len(methods), 5.0))
    rng = np.random.default_rng(0)
    violin_width = 0.30
    group_offset = 0.19
    unc_label_added = False
    all_rates: list[np.ndarray] = []

    def _violin_and_strip(x: float, values: np.ndarray, color: str, body_alpha: float, label: str | None) -> None:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        if len(values) >= 3 and np.ptp(values) > 1e-9:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vp = ax.violinplot([values], positions=[x], widths=violin_width, showmedians=True, showextrema=False)
                body = vp["bodies"][0]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(body_alpha)
                vp["cmedians"].set_color(color)
                vp["cmedians"].set_linewidth(1.3)
                vp["cmedians"].set_alpha(0.9)
            except Exception:
                pass
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter, values, s=12, alpha=0.55, color=color,
            zorder=3, label=label, edgecolors="none",
        )

    for j, m in enumerate(methods):
        m_rows = [r for r in results if r.method == m]
        rates_u = np.array([r.rejects_llm_only / r.n_reps if r.n_reps else float("nan") for r in m_rows])
        rates_c = np.array([r.rejects_ppi / r.n_reps if r.n_reps else float("nan") for r in m_rows])
        all_rates.append(rates_u)
        all_rates.append(rates_c)
        _violin_and_strip(
            j - group_offset, rates_u, "#808080", 0.35,
            "Uncorrected (any test)" if not unc_label_added else None,
        )
        unc_label_added = True
        _violin_and_strip(j + group_offset, rates_c, get_method_color(m), 0.45, _pretty_test(m))

    lm_note = {"mcar": " -- MCAR labeling", "mnar": " -- MNAR labeling (mild+strong)", None: ""}[lm_filter]
    ax.axhline(alpha, color="black", ls="--", lw=1.1, label=f"Nominal {_alpha_label(alpha)}")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    scatter_max = np.nanmax(np.concatenate(all_rates)) if all_rates else float("nan")
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([_pretty_test(m) for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Observed rejection rate")
    ax.set_xlabel("Test")
    ax.set_title(
        f"PPI-Corrected Type-I Error, by Test (Full Factorial Sweep){lm_note}\n"
        "(gray = uncorrected, color = corrected; each dot: one factorial scenario cell)", fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_PPI_POWER_NAME_RE = re.compile(r"^[a-z]+\.([a-z]+)\.es=([\d.]+)$")
"""Matches every power-family scenario name regardless of prefix -- "power."
(build_ppi_power_sources, bias opposing the effect), "powerrf."
(build_ppi_power_reinforcing_sources, bias reinforcing the effect), and
"powernb." (build_ppi_power_nobias_sources, bias_type="none") all share the
same "<prefix>.<eval_type>.es=<frac>" shape, so one parser/report/plot
pipeline serves all three variants."""


def _parse_ppi_power_name(name: str) -> tuple[str, float]:
    m = _PPI_POWER_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized power scenario name: {name!r}")
    return m.group(1), float(m.group(2))


def print_ppi_power_report(results: list[PPIResult], alpha: float, header: str = "POWER UNDER JUDGE BIAS") -> None:
    """Corrected vs. uncorrected rejection rate (POWER, not Type-I) as a
    real effect_size grows, per eval type -- the complement to
    print_ppi_report's null-only Type-I table (build_judge_bias_sources
    never sets effect_size above 0, so that table can only show whether
    PPI correction controls false positives, never whether it retains the
    power to detect a genuine difference under the SAME bias severity).
    es=0.00 doubles as a Type-I cross-check against build_judge_bias_sources'
    'eval_type.*' scenarios (same settings -- should read ~alpha here too).
    ``header`` distinguishes the bias-direction/no-bias variants (see
    _PPI_POWER_NAME_RE) when this same function is reused for them."""
    if not results:
        print("\n  (no PPI power results)")
        return
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    parsed = {r.name: _parse_ppi_power_name(r.name) for r in results}
    eval_types = sorted({et for et, _ in parsed.values()})
    es_values = sorted({es for _, es in parsed.values()})

    print(f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- {header}\n"
          f"  Same bias severity as build_judge_bias_sources' eval_type.* baseline; nominal alpha={alpha}\n"
          f"  es=0.00 column is a Type-I cross-check (should read ~alpha)\n{'='*88}")

    for et in eval_types:
        print(f"\n  [{et}]")
        hdr = f"    {'Test':<14}" + "".join(f"    c({es:.2f})".rjust(11) + f"  u({es:.2f})".rjust(11) for es in es_values)
        print(hdr)
        for t in tests:
            t_rows = [r for r in results if r.test == t]
            if not any(parsed[r.name][0] == et for r in t_rows):
                continue
            row = f"    {t:<14}"
            for es in es_values:
                cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                c_tot = sum(r.corrected_rejects for r in cell_rows)
                u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                n_tot = sum(r.n_reps for r in cell_rows)
                rc = c_tot / n_tot if n_tot > 0 else float("nan")
                ru = u_tot / n_tot if n_tot > 0 else float("nan")
                row += f"  {rc:>9.3f}  {ru:>9.3f}"
            print(row)
    print()


def save_results_artifacts_ppi_power(*, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str) -> list[str]:
    """Write the PPI power-sweep results CSV under out_dir. Returns the
    written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_power_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "effect_size", "n", "test", "n_reps",
            "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate",
        ])
        for r in results:
            et, es = _parse_ppi_power_name(r.name)
            writer.writerow([
                r.name, r.tag, et, f"{es:.4f}", r.n, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_power_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_power_report(results, alpha=alpha)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_power_plot(
    *, results: list[PPIResult], alpha: float, out_path: str, title_suffix: str = "",
) -> str:
    """Power curve (rejection rate vs. real effect_size) -- TWO rows (top:
    corrected, bottom: uncorrected), one column per eval type, rather than
    overlaying both in one set of axes: with up to 13 tests' worth of
    same-colored solid+dashed lines sharing one plot, superimposing
    corrected and uncorrected became unreadable. Uncorrected keeps its
    dashed linestyle in its own row, consistent with save_ppi_typeI_plot/
    save_ppi_power_direction_plot's convention.

    (An earlier version of this function also accepted an ``ideal_results``
    list -- build_ppi_power_nobias_sources' results, overlaid as a dotted
    "ideal" reference line on the corrected row. Removed on request as
    unneeded clutter; run() still computes the no-bias check before the
    main power plot, since it also feeds its own standalone
    ``..._power_vs_effect_size_nobias.png`` plot via a separate call.)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not results:
        raise ValueError("No PPI power results to plot.")
    tests = _ppi_tests_present(results, nonstandard=False)
    parsed = {r.name: _parse_ppi_power_name(r.name) for r in results}
    eval_types = sorted({et for et, _ in parsed.values()})
    es_values = sorted({es for _, es in parsed.values()})

    fig, axes = plt.subplots(2, len(eval_types), figsize=(4.6 * len(eval_types), 7.6), squeeze=False)
    for col, et in enumerate(eval_types):
        ax_c, ax_u = axes[0][col], axes[1][col]
        ax_c.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
        ax_u.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
        for t in tests:
            t_rows = [r for r in results if r.test == t]
            ys_c, ys_u = [], []
            for es in es_values:
                cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                c_tot = sum(r.corrected_rejects for r in cell_rows)
                u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                n_tot = sum(r.n_reps for r in cell_rows)
                ys_c.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                ys_u.append(u_tot / n_tot if n_tot > 0 else float("nan"))
            if not any(np.isfinite(ys_c)):
                continue
            color = get_method_color(t)
            ax_c.plot(es_values, ys_c, marker="o", color=color, linewidth=1.6, markersize=4, label=_pretty_test(t), zorder=2)
            ax_u.plot(es_values, ys_u, marker="x", color=color, linewidth=1.4, linestyle="--", markersize=4, zorder=2)

        ax_c.set_title(et.capitalize())
        ax_c.set_ylabel("Rejection rate\n(corrected)" if col == 0 else "")
        ax_c.set_ylim(-0.02, 1.02)
        ax_c.set_xticklabels([])

        ax_u.set_xlabel("Effect size")
        ax_u.set_ylabel("Rejection rate\n(uncorrected)" if col == 0 else "")
        ax_u.set_ylim(-0.02, 1.02)

    # Built directly from `tests` (not axes[0][0].get_legend_handles_labels())
    # -- a test whose ys_c happens to be all-NaN for the FIRST eval_type
    # column (e.g. a test that only applies to some eval types) gets `continue`d
    # before ever calling ax_c.plot(..., label=...) on axes[0][0] specifically,
    # even though it DOES get plotted (with a label) in a later column's axes.
    # Collecting from axes[0][0] alone silently dropped that test from the
    # legend despite its line being visible in the figure.
    handles = [Line2D([0], [0], color=get_method_color(t), marker="o", linewidth=1.6, markersize=4) for t in tests]
    labels = [_pretty_test(t) for t in tests]
    handles.append(Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6))
    labels.append(f"Nominal {_alpha_label(alpha)}")
    fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
    fig.suptitle(f"PPI-Corrected Power vs. Effect Size{title_suffix}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_power_direction_plot(
    *, opposing: list[PPIResult], reinforcing: list[PPIResult], alpha: float, out_path: str,
) -> str:
    """Power curve comparison: judge bias OPPOSING the injected real effect
    (build_ppi_power_sources) vs. REINFORCING it (build_ppi_power_reinforcing_
    sources) -- one row per direction, one column per eval type. Opposing
    bias produces the "cancellation dip" visible in save_ppi_power_plot's
    uncorrected line (bias and effect partially cancel as effect_size
    grows). Reinforcing bias instead pushes the uncorrected line ABOVE the
    corrected one with no dip at all -- arguably the more dangerous failure
    mode in practice, since nothing about the SHAPE of the uncorrected curve
    alone would flag it as wrong; it just quietly overstates the effect."""
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    row_titles = {"opposing": "Bias Opposes True Effect", "reinforcing": "Bias Reinforces True Effect"}
    rows = [(label, res) for label, res in [("opposing", opposing), ("reinforcing", reinforcing)] if res]
    if not rows:
        raise ValueError("No PPI power-direction results to plot.")
    all_results = [r for _, res in rows for r in res]
    tests = _ppi_tests_present(all_results, nonstandard=False)
    eval_types = sorted({_parse_ppi_power_name(r.name)[0] for r in all_results})

    fig, axes = plt.subplots(
        len(rows), len(eval_types), figsize=(5.2 * len(eval_types), 4.0 * len(rows)), squeeze=False,
    )
    for row_idx, (row_label, res) in enumerate(rows):
        parsed = {r.name: _parse_ppi_power_name(r.name) for r in res}
        es_values = sorted({es for _, es in parsed.values()})
        for col, et in enumerate(eval_types):
            ax = axes[row_idx][col]
            ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
            for t in tests:
                t_rows = [r for r in res if r.test == t]
                ys_c, ys_u = [], []
                for es in es_values:
                    cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                    c_tot = sum(r.corrected_rejects for r in cell_rows)
                    u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                    n_tot = sum(r.n_reps for r in cell_rows)
                    ys_c.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                    ys_u.append(u_tot / n_tot if n_tot > 0 else float("nan"))
                if not any(np.isfinite(ys_c)):
                    continue
                color = get_method_color(t)
                ax.plot(
                    es_values, ys_c, marker="o", color=color, linewidth=1.6,
                    label=_pretty_test(t) if row_idx == 0 else None, zorder=2,
                )
                ax.plot(es_values, ys_u, marker="x", color=color, linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)
            if row_idx == 0:
                ax.set_title(et.capitalize())
            if col == 0:
                ax.set_ylabel(f"{row_titles.get(row_label, row_label)}\nRejection rate")
            ax.set_xlabel("Effect size")
            ax.set_ylim(-0.02, 1.02)

    # Built directly from `tests` (not axes[0][0].get_legend_handles_labels())
    # -- same reason as save_ppi_power_plot: a test with no finite data in
    # row 0's FIRST eval_type column never gets a label registered on
    # axes[0][0], even though it's plotted (with a label) elsewhere.
    handles = [Line2D([0], [0], color=get_method_color(t), marker="o", linewidth=1.6) for t in tests]
    labels = [_pretty_test(t) for t in tests]
    handles += [
        Line2D([0], [0], color="#333333", marker="o", linewidth=1.6, linestyle="-"),
        Line2D([0], [0], color="#333333", marker="x", linewidth=1.0, linestyle="--", alpha=0.7),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6),
    ]
    labels += ["Corrected", "Uncorrected", f"Nominal {_alpha_label(alpha)}"]
    fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
    fig.suptitle("PPI-Corrected Power: Bias Opposing vs. Reinforcing the True Effect", y=1.03, fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_PPI_POWER_NLAB_GRID_NAME_RE = re.compile(r"^powernlab\.([a-z]+)\.n=(\d+)\.nlab=(\d+)\.es=([\d.]+)$")


def _parse_ppi_power_nlab_grid_name(name: str) -> tuple[str, int, int, float]:
    m = _PPI_POWER_NLAB_GRID_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized power n_lab-grid scenario name: {name!r}")
    return m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))


def print_ppi_power_nlab_grid_report(
    results: list[PPIResult], alpha: float, header: str = "bias reinforcing effect",
) -> None:
    """Corrected rejection rate (POWER) across the N x N_lab label/dataset-
    size grid (build_ppi_power_nlab_grid_reinforcing_sources/build_ppi_
    power_nlab_grid_opposing_sources), one table per test: rows are (N,
    N_lab) cells, columns are effect_size. Uncorrected rate is dropped here
    (unlike print_ppi_power_report) to keep the table narrow enough to read
    across up to 10 effect_size columns x 12 label-count rows -- see the
    CSV artifact and save_ppi_power_nlab_grid_plots for full
    corrected+uncorrected detail per cell. ``header`` distinguishes the
    reinforcing/opposing variants when this same function is reused for
    both (see run()'s power_nlab_grid_check block)."""
    if not results:
        print("\n  (no PPI power n_lab-grid results)")
        return
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    parsed = {r.name: _parse_ppi_power_nlab_grid_name(r.name) for r in results}
    cells = sorted({(n, nlab) for _, n, nlab, _ in parsed.values()})
    es_values = sorted({es for _, _, _, es in parsed.values()})

    print(f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- POWER vs. LABEL/DATASET-SIZE GRID ({header})\n"
          f"  Corrected rejection rate only; nominal alpha={alpha}\n{'='*88}")
    for t in tests:
        t_rows = [r for r in results if r.test == t]
        if not t_rows:
            continue
        print(f"\n  [{t}]")
        hdr = f"    {'N':>5}  {'N_lab':>6}" + "".join(f"  es={es:.3f}".rjust(10) for es in es_values)
        print(hdr)
        for (n, nlab) in cells:
            cell_all = [r for r in t_rows if parsed[r.name][1] == n and parsed[r.name][2] == nlab]
            if not cell_all:
                continue
            row = f"    {n:>5}  {nlab:>6}"
            for es in es_values:
                cr = [r for r in cell_all if parsed[r.name][3] == es]
                c_tot = sum(r.corrected_rejects for r in cr)
                n_tot = sum(r.n_reps for r in cr)
                rc = c_tot / n_tot if n_tot > 0 else float("nan")
                row += f"  {rc:>8.3f}"
            print(row)
    print()


def save_results_artifacts_ppi_power_nlab_grid(
    *, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str,
    header: str = "bias reinforcing effect",
) -> list[str]:
    """Write the PPI power-vs-n_lab grid results CSV under out_dir. Returns
    the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_power_nlab_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "n", "n_lab", "effect_size", "test", "n_reps",
            "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate",
        ])
        for r in results:
            et, n, nlab, es = _parse_ppi_power_nlab_grid_name(r.name)
            writer.writerow([
                r.name, r.tag, et, n, nlab, f"{es:.4f}", r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_power_nlab_grid_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_power_nlab_grid_report(results, alpha=alpha, header=header)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_power_nlab_grid_plots(
    *, results: list[PPIResult], alpha: float, out_dir: str, stem: str,
) -> list[str]:
    """One power-curve plot per (N, N_lab) cell of the label/dataset-size
    grid (build_ppi_power_nlab_grid_reinforcing_sources), plus one averaged
    summary plot pooling every cell -- the user-facing deliverable for "does
    more labels/data fix MWU's reinforcing-bias power anomaly, and if so how
    much." Each cell panel plots corrected (solid) and uncorrected (dashed)
    rejection rate vs. effect_size, one color per test present, mirroring
    save_ppi_power_direction_plot's single-eval_type/single-direction line
    convention. The averaged plot pools corrected_rejects/uncorrected_rejects/
    n_reps across ALL (N, N_lab) cells before dividing (reps-weighted mean,
    not a naive mean-of-rates), so cells with more reps aren't
    under/over-weighted."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not results:
        raise ValueError("No PPI power n_lab-grid results to plot.")
    parsed = {r.name: _parse_ppi_power_nlab_grid_name(r.name) for r in results}
    tests = _ppi_tests_present(results, nonstandard=False)
    cells = sorted({(n, nlab) for _, n, nlab, _ in parsed.values()})
    es_all = sorted({es for _, _, _, es in parsed.values()})
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []

    def _plot_rates(ax, cell_results: list[PPIResult], es_values: list[float]) -> None:
        ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
        for t in tests:
            t_rows = [r for r in cell_results if r.test == t]
            ys_c, ys_u = [], []
            for es in es_values:
                cr = [r for r in t_rows if parsed[r.name][3] == es]
                c_tot = sum(r.corrected_rejects for r in cr)
                u_tot = sum(r.uncorrected_rejects for r in cr)
                n_tot = sum(r.n_reps for r in cr)
                ys_c.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                ys_u.append(u_tot / n_tot if n_tot > 0 else float("nan"))
            if not any(np.isfinite(ys_c)):
                continue
            color = get_method_color(t)
            ax.plot(es_values, ys_c, marker="o", color=color, linewidth=1.6, markersize=4, label=_pretty_test(t), zorder=2)
            ax.plot(es_values, ys_u, marker="x", color=color, linewidth=1.2, linestyle="--", markersize=4, alpha=0.6, zorder=1)
        ax.set_ylim(-0.02, 1.02)

    def _legend_handles() -> tuple[list, list]:
        handles = [Line2D([0], [0], color=get_method_color(t), marker="o", linewidth=1.6, markersize=4) for t in tests]
        labels = [_pretty_test(t) for t in tests]
        handles += [
            Line2D([0], [0], color="#333333", marker="x", linewidth=1.2, linestyle="--", alpha=0.6),
            Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6),
        ]
        labels += ["Uncorrected (any test)", f"Nominal {_alpha_label(alpha)}"]
        return handles, labels

    for (n, nlab) in cells:
        cell_results = [r for r in results if parsed[r.name][1] == n and parsed[r.name][2] == nlab]
        es_values = sorted({parsed[r.name][3] for r in cell_results})
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        _plot_rates(ax, cell_results, es_values)
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Rejection rate")
        ax.set_title(f"N={n}, N_lab={nlab}")
        handles, labels = _legend_handles()
        fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
            fig.tight_layout()
        cell_path = str(out_base / f"{stem}_n{n}_nlab{nlab}.png")
        fig.savefig(cell_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(cell_path)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    _plot_rates(ax, results, es_all)
    ax.set_xlabel("Effect size")
    ax.set_ylabel("Rejection rate")
    ax.set_title(f"Averaged Across {len(cells)} (N, N_lab) Combinations")
    handles, labels = _legend_handles()
    fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
    fig.suptitle("PPI-Corrected Power vs. Effect Size, Averaged Over Label/Dataset-Size Grid", fontsize=11)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    avg_path = str(out_base / f"{stem}_averaged.png")
    fig.savefig(avg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(avg_path)
    return out_paths


def save_ppi_power_nlab_grid_direction_plot(
    *, opposing: list[PPIResult], reinforcing: list[PPIResult], alpha: float, out_path: str,
) -> str:
    """The key confirming figure for cases/pvalues.py's appendix writeup on
    MWU's reinforcing-bias power anomaly: corrected rejection rate vs.
    effect_size, averaged (reps-weighted) across the WHOLE N x N_lab grid,
    opposing (dashed) vs. reinforcing (solid) overlaid in one panel per
    test. Answers "is the anomaly specific to the reinforcing direction"
    directly and visually -- see save_ppi_power_direction_plot for the
    single-(N, N_lab)-point analogue this mirrors. Uncorrected rate is
    dropped here (unlike save_ppi_power_direction_plot): with both
    direction AND test already claiming a visual channel (linestyle,
    color), a third uncorrected line would need a fourth, and the
    reinforcing-vs-opposing CORRECTED comparison is the one this figure
    exists to make."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [(label, res) for label, res in [("opposing", opposing), ("reinforcing", reinforcing)] if res]
    if not rows:
        raise ValueError("No PPI power n_lab-grid direction results to plot.")
    all_results = [r for _, res in rows for r in res]
    tests = _ppi_tests_present(all_results, nonstandard=False)
    parsed_all = {r.name: _parse_ppi_power_nlab_grid_name(r.name) for r in all_results}
    es_values = sorted({es for _, _, _, es in parsed_all.values()})

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
    for direction_label, res in rows:
        parsed = {r.name: _parse_ppi_power_nlab_grid_name(r.name) for r in res}
        linestyle = "-" if direction_label == "reinforcing" else "--"
        for t in tests:
            t_rows = [r for r in res if r.test == t]
            ys = []
            for es in es_values:
                cell_rows = [r for r in t_rows if parsed[r.name][3] == es]
                c_tot = sum(r.corrected_rejects for r in cell_rows)
                n_tot = sum(r.n_reps for r in cell_rows)
                ys.append(c_tot / n_tot if n_tot > 0 else float("nan"))
            if not any(np.isfinite(ys)):
                continue
            color = get_method_color(t)
            ax.plot(
                es_values, ys, marker="o" if direction_label == "reinforcing" else "x",
                color=color, linewidth=1.8, linestyle=linestyle, markersize=5,
                label=_pretty_test(t) if direction_label == "reinforcing" else None, zorder=2,
            )

    ax.set_xlabel("Effect size")
    ax.set_ylabel("Rejection rate (corrected)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Averaged Across the N x N_lab Grid")
    handles = [Line2D([0], [0], color=get_method_color(t), marker="o", linewidth=1.8, markersize=5) for t in tests]
    labels = [_pretty_test(t) for t in tests]
    handles += [
        Line2D([0], [0], color="#333333", marker="o", linewidth=1.8, linestyle="-"),
        Line2D([0], [0], color="#333333", marker="x", linewidth=1.8, linestyle="--"),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6),
    ]
    labels += ["Reinforcing", "Opposing", f"Nominal {_alpha_label(alpha)}"]
    fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
    fig.suptitle("PPI-Corrected Power vs. Effect Size: Bias Direction, Averaged Over Label/Dataset-Size Grid", fontsize=10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_ppi_effect_report(results: list[PPIEffectResult], alpha: float, regime: str = "") -> None:
    """Bias & CI-coverage summary, mirroring sim_type_i_calibration.py's
    ``_print_effect_table``: per-test mean bias, worst |z|, coverage, worst
    coverage scenario, and mean CI width, plus a flagged-cells list (|bias
    z| > 3, or coverage meaningfully under the 1-alpha target).

    regime : str
        Optional label (e.g. "MCAR", "MNAR") appended to the header -- see
        print_ppi_report's own regime parameter."""
    if not results:
        print(
            "\n  (no PPI effect-check results -- active --tests must include at least one of "
            f"{', '.join(_PPI_EFFECT_TESTS)})"
        )
        return

    target_cov = 1.0 - alpha
    cov_flag_margin = 0.02
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]

    width = 96
    bar = "-" * width
    dbar = "=" * width
    print()
    print(dbar)
    print(f"  PVALUES (PPI-CORRECTED) -- EFFECT-SIZE CALIBRATION (bias & CI coverage){f' ({regime})' if regime else ''}")
    print("  (vs. Monte Carlo gold-reference null per scenario/test -- see estimate_judge_bias_gold_null_values)")
    print(dbar)
    print()
    header = (
        f"  {'Test':<12} {'n':>7} {'mean bias':>10} {'worst |z|':>10} "
        f"{'worst scen (bias)':<26} {'coverage':>9} {'cov min':>8} {'worst scen (cov)':<24} "
        f"{'mean width':>11}"
    )
    print(header)
    print(bar)

    flagged: list[str] = []
    for t in tests:
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        n_total = sum(r.n_samples for r in t_rows)
        weights = [r.n_samples for r in t_rows]
        mean_bias = float(np.average([r.mean_bias for r in t_rows], weights=weights))
        mean_width = float(np.average([r.mean_ci_width for r in t_rows], weights=weights))
        coverage = float(np.average([r.coverage for r in t_rows], weights=weights))
        worst_bias = max(t_rows, key=lambda r: abs(r.bias_z) if np.isfinite(r.bias_z) else 0.0)
        worst_cov = min(t_rows, key=lambda r: r.coverage if np.isfinite(r.coverage) else 1.0)

        print(
            f"  {t:<12} {n_total:>7} {mean_bias:>+10.4f} {abs(worst_bias.bias_z) if np.isfinite(worst_bias.bias_z) else 0.0:>10.2f} "
            f"{worst_bias.name:<26} {coverage:>9.3f} {worst_cov.coverage:>8.3f} {worst_cov.name:<24} "
            f"{mean_width:>11.4f}"
        )

        for r in t_rows:
            if np.isfinite(r.bias_z) and abs(r.bias_z) > 3.0:
                flagged.append(
                    f"    bias    {r.name:<28} {t:<10} mean={r.mean_bias:+.4f}  z={r.bias_z:+.2f}  (n={r.n_samples})"
                )
            lo_cov, hi_cov = _ppi_wilson_interval(int(round(r.coverage * r.n_samples)), r.n_samples)
            if hi_cov < target_cov - cov_flag_margin:
                flagged.append(
                    f"    cover   {r.name:<28} {t:<10} coverage={r.coverage:.3f}  "
                    f"Wilson=[{lo_cov:.3f},{hi_cov:.3f}]  (n={r.n_samples})"
                )

    print()
    if flagged:
        print(f"  Flagged cells (|bias z| > 3, or coverage Wilson upper bound < {target_cov - cov_flag_margin:.2f}):")
        for line in flagged:
            print(line)
    else:
        print("  No scenario x test cells flagged for bias or under-coverage.")
    print()


def latex_ppi_effect_overall_summary(results: list[PPIEffectResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-test bias and CI coverage (with
    its 95% MC band) and mean CI width of the PPI-corrected point estimate,
    averaged across scenarios -- complements latex_ppi_overall_summary's
    Type-I table with "is the estimate itself trustworthy," not just "does
    the p-value stay calibrated."""
    target_cov = 1.0 - alpha
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    rows = []
    for t in tests:
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        n_tot = sum(r.n_samples for r in t_rows)
        weights = [r.n_samples for r in t_rows]
        mean_bias = float(np.average([r.mean_bias for r in t_rows], weights=weights))
        mean_width = float(np.average([r.mean_ci_width for r in t_rows], weights=weights))
        cov_count = sum(int(round(r.coverage * r.n_samples)) for r in t_rows)
        coverage = cov_count / n_tot if n_tot > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(cov_count, n_tot)
        rows.append([
            escape_latex(t),
            f"{mean_bias:+.4f}",
            f"{coverage:.3f}" if np.isfinite(coverage) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{mean_width:.4f}" if np.isfinite(mean_width) else "-",
        ])

    return booktabs_table(
        caption=f"pvalues (PPI-corrected): bias and CI coverage of the corrected point estimate (nominal {target_cov:.0%}).",
        label="tab:pvalues_ppi_effect_overall",
        columns=["Test", "Mean bias", "Coverage", "95\\% MC band", "Mean CI width"],
        rows=rows,
    )


def save_results_artifacts_ppi_effect(*, results: list[PPIEffectResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False, regime: str = "") -> list[str]:
    """Write the PPI effect-size-sweep results CSV (and LaTeX summary if
    `latex=True`) under out_dir. Returns the written file paths."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_effect_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "n", "test", "n_samples", "null_value",
            "mean_bias", "bias_z", "coverage", "mean_ci_width", "uncorrected_bias_z",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.n, r.test, r.n_samples, r.null_value,
                r.mean_bias, r.bias_z, r.coverage, r.mean_ci_width, r.uncorrected_bias_z,
            ])
    summary_path = out_base / f"{run_stem}_ppi_effect_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_effect_report(results, alpha=alpha, regime=regime)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_ppi_effect_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_effect_plot(
    *, results: list[PPIEffectResult], alpha: float, out_path: str, ci_comparison: bool = False,
    nonstandard: bool = False, regime: str = "",
    width_norm: dict[str, float] | None = None,
) -> str:
    """Bias-z / CI-coverage / CI-width scatter, one jittered column per test
    -- mirrors sim_type_i_calibration.py's ``_plot_effect_results`` (3
    panels), reading directly off PPIEffectResult's already-aggregated
    per-scenario stats rather than raw bootstrap samples.

    ci_comparison : bool
        When False (default), plots only the standard/textbook tests
        (excludes bayes_bootstrap/bootstrap_t/tango/ppi_wilson/
        bootstrap_t_single/ppi_t_interval/ppi_logit_t -- see
        _PPI_NONSTANDARD_TESTS). When True, plots ONLY the curated
        4-method PPI-corrected CI comparison instead -- Tango, PPI Wilson,
        PPI logit-t, PPI t-interval -- see _PPI_CI_COMPARISON_TESTS for
        why exactly these four (and not the broader bootstrap/CI-based
        set _ppi_tests_present(nonstandard=True) would return).

    nonstandard : bool
        Ignored when ci_comparison=True (that branch has its own fixed
        4-method set). Otherwise, False (default) plots the standard/
        textbook tests as above; True plots the complementary broader
        bootstrap/CI-based set instead (_ppi_tests_present(nonstandard=
        True) -- everything ci_comparison's curated 4 are a subset of).

    width_norm : dict[str, float] | None
        Optional ``{scenario_name: divisor}`` map (typically each
        scenario's own eval-type scale SPAN, e.g. from
        EVAL_TYPE_SCALE_BOUNDS) used to rescale the CI Width panel's raw
        ``mean_ci_width`` before plotting. PPIEffectResult has no
        eval_type field of its own (see its docstring), so callers that
        want normalized widths must build this from the JudgeBiasSource
        list that produced ``results`` and pass it in -- see run()'s call
        site. None (the default) plots raw, un-normalized widths.

        Without this, t-interval/logit-t's shared CI Width panel plots
        grades (0-100 scale), likert (1-5), and continuous/binary (0-1)
        scenarios' raw widths on one shared axis -- e.g. grades widths
        would cluster far above continuous's purely because grades' scale
        is 100x continuous's, not because grades is worse-calibrated
        (coverage stays nominal across all of them). Dividing by each
        scenario's own scale span turns "raw score units" into "fraction
        of that eval type's natural range," which is directly comparable
        across eval types.
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("save_ppi_effect_plot: no PPI effect-check results to plot.")

    tests = _ppi_ci_comparison_tests_present(results) if ci_comparison else _ppi_tests_present(results, nonstandard=nonstandard)
    target_cov = 1.0 - alpha
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 5.5))
    rng = np.random.default_rng(0)

    for j, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        x = j + rng.uniform(-0.16, 0.16, size=len(t_rows))
        color = get_method_color(t)

        z = np.array([r.bias_z for r in t_rows])
        keep_z = np.isfinite(z)
        ax1.scatter(x[keep_z], z[keep_z], s=22, alpha=0.7, color=color, label=_pretty_test(t))

        cov = np.array([r.coverage for r in t_rows])
        keep_c = np.isfinite(cov)
        ax2.scatter(x[keep_c], cov[keep_c], s=22, alpha=0.7, color=color)

        if width_norm is not None:
            wid = np.array([r.mean_ci_width / width_norm.get(r.name, 1.0) for r in t_rows])
        else:
            wid = np.array([r.mean_ci_width for r in t_rows])
        keep_w = np.isfinite(wid)
        ax3.scatter(x[keep_w], wid[keep_w], s=22, alpha=0.7, color=color)

    ax1.axhline(0.0, color="black", ls="--", lw=1.0)
    ax1.axhline(3.0, color="red", ls=":", lw=0.9, label="|z| = 3 (flagged)")
    ax1.axhline(-3.0, color="red", ls=":", lw=0.9)
    ax1.set_xticks(np.arange(len(tests)))
    ax1.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Bias z-score")
    ax1.set_title("Estimate Bias (vs. Gold-Reference Null)")
    ax1.grid(axis="y", alpha=0.25, lw=0.8)

    ax2.axhline(target_cov, color="black", ls="--", lw=1.1, label=f"Target = {target_cov:.2f}")
    ax2.set_xticks(np.arange(len(tests)))
    ax2.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel("CI coverage of gold-reference null")
    ax2.set_title("CI Coverage")
    ax2.grid(axis="y", alpha=0.25, lw=0.8)
    ax2.legend(loc="lower left", fontsize=8)

    ax3.set_xticks(np.arange(len(tests)))
    ax3.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Mean CI width (fraction of eval-type scale)" if width_norm is not None else "Mean CI width")
    ax3.set_title("CI Width")
    ax3.grid(axis="y", alpha=0.25, lw=0.8)

    handles, labels = ax1.get_legend_handles_labels()
    if ci_comparison:
        title_suffix = " -- PPI-Corrected CI Methods (Bonett-Price / Wilson / Logit-t / t-interval)"
    elif nonstandard:
        title_suffix = " -- Nonstandard (Bootstrap/CI-Based) Tests"
    else:
        title_suffix = ""
    title_suffix += f" ({regime})" if regime else ""
    fig.suptitle(f"PPI-Corrected Effect-Size Calibration: Bias, Coverage, and Width{title_suffix}", fontsize=12)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, borderaxespad=0.5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this case's CLI flags: which of the four sweep modes to run,
    sample sizes/reps/effect sizes, data source, and output options -- see
    `run()` for how `--mode` resolves to the sweeps actually executed."""
    parser.add_argument("--mode", choices=MODES, default="all",
                         help="'pairwise' (non-PPI A/B), 'multiarm' (non-PPI k-arm), "
                              "'ppi' (PPI-corrected calibration), 'simultaneous_ci' (none vs. Bonferroni vs. "
                              "max-T simultaneous-CI coverage/width, on the same k-arm sources as multiarm), "
                              "'pairwise_multiarm' (just pairwise+multiarm), or 'all' (default: every mode "
                              "applicable to --data-source -- pairwise+multiarm+ppi+simultaneous_ci for "
                              "synthetic, pairwise+multiarm+simultaneous_ci for real data)")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs overall-summary table to each saved summary .log file.")

    # pairwise mode
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                         help="pairwise/multiarm modes: 'synthetic' (default), or a real-data source: " + ", ".join(REAL_PAIR_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="pairwise mode: synthetic scenario breadth for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=DEFAULT_EVAL_TYPES, metavar="TYPE",
                         help="pairwise/multiarm/simultaneous_ci modes: restrict to these eval types "
                              f"(default: {' '.join(DEFAULT_EVAL_TYPES)}; pass 'grades' explicitly to include it)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100], metavar="N",
                         help="pairwise/multiarm modes: sample sizes to sweep")
    parser.add_argument("--runs", type=int, default=1, metavar="R",
                         help="pairwise/multiarm modes: runs per input (R>1 activates binary majority-vote/nested paths; "
                              "real-data pairwise sources only support --runs 1)")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean",
                         help="pairwise/multiarm modes: statistic passed to evalstats.core.paired")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N",
                         help="pairwise/multiarm modes: bootstrap resample count")
    parser.add_argument("--icc-values", type=float, nargs="+", default=[0.05, 0.20, 0.40, 0.60, 0.80], metavar="ICC",
                         help="pairwise mode: ICC sweep for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.2, 0.4], metavar="D",
                         help="pairwise mode: alt-condition effect sizes for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID",
                         help="pairwise/multiarm modes, real data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                         help="pairwise/multiarm modes, real data: model names to filter to")
    parser.add_argument("--hf-token", default=None, help="pairwise/multiarm modes, real data")
    parser.add_argument("--cache-dir", default=None, help="pairwise/multiarm modes, real data")
    parser.add_argument("--min-pair-size", type=int, default=50,
                         help="pairwise/multiarm modes, real data: minimum shared items required "
                              "(multiarm: across ALL aligned models for a benchmark, not just a pair)")
    parser.add_argument("--inspect-csv", default=None,
                         help=f"pairwise/multiarm modes, real data: path to CSV from collect_inspect_benchmarks.py "
                              f"(used by --data-source inspect/real; defaults to {DEFAULT_INSPECT_CSV!r})")

    # multiarm mode (also used by simultaneous_ci mode -- same k-arm sources/grid)
    parser.add_argument("--k-arms", nargs="+", type=int, default=[4], metavar="K",
                        help="multiarm/simultaneous_ci modes: number of arms to sweep (one or more values, e.g. --k-arms 3 5 10); "
                             "max-T and post-hoc corrections are compared at each k. Real-data sources cap k at "
                             "however many aligned real models a benchmark has; larger k values are skipped with a warning.")
    parser.add_argument("--multiarm-method", default=BOOTSTRAP_T.name, metavar="METHOD",
                         choices=[BOOTSTRAP.name, BCA.name, BAYES_BOOTSTRAP.name, SMOOTH_BOOTSTRAP.name, PERMUTATION.name, BOOTSTRAP_T.name],
                         help="multiarm mode: only affects max_t's point estimate + bootstrap draws (none/holm/"
                              "bonferroni/fdr_bh correct the base paired p-value for the data type regardless -- "
                              "McNemar mid-p on binary, Wilcoxon otherwise) / "
                              "simultaneous_ci mode: only affects max_t's construction (none/bonferroni/sidak/boot "
                              "build on the canonical per-eval-type CI regardless) -- must be bootstrap-compatible "
                              "for max-T to apply")
    parser.add_argument("--multiarm-icc", type=float, default=0.20, metavar="ICC",
                         help="multiarm/simultaneous_ci modes: ICC for build_multiarm_sources' shared truth/noise model "
                              "(same meaning as --icc-values in pairwise mode)")
    parser.add_argument("--multiarm-cohens-d", type=float, default=0.3, metavar="D",
                         help="multiarm/simultaneous_ci modes: alt-condition effect size (Cohen's d) for build_multiarm_sources")
    parser.add_argument("--corrections", nargs="+", choices=[m.name for m in MULTIARM_CORRECTION_METHODS], default=None, metavar="CORRECTION",
                         help="multiarm mode: restrict to these correction strategies (default: all of "
                              f"{[m.name for m in MULTIARM_CORRECTION_METHODS]}) -- e.g. for a fast targeted re-run "
                              "of just the resampling-based corrections (max_t/romano_wolf/westfall_young) at "
                              "larger n without paying for the full correction set")
    parser.add_argument("--ci-methods", nargs="+", choices=[m.name for m in ALL_SIMULTANEOUS_CI_METHODS], default=None, metavar="METHOD",
                         help="simultaneous_ci mode: restrict to these CI methods (default: all of "
                              f"{[m.name for m in ALL_SIMULTANEOUS_CI_METHODS]}) -- e.g. --ci-methods boot sidak "
                              "to skip max_t's and none's/bonferroni's independent bootstrap/construction cost "
                              "entirely, not just skip reporting them (max_t and boot each pay for their own "
                              "separate bootstrap resample in this mode, unlike --mode multiarm's sharing)")

    # ppi mode
    parser.add_argument("--tests", nargs="+", choices=[m.name for m in PPI_TEST_METHODS], default=None, metavar="TEST",
                         help="ppi mode: restrict to these evalstats.tests names (default: all)")
    parser.add_argument("--ppi-n-boot", type=int, default=1000, metavar="N",
                         help="ppi mode: PPI bootstrap resample count")
    parser.add_argument("--effect-reps", type=int, default=200, metavar="N",
                         help="ppi mode: reps for the bias/CI-coverage effect-size check of the corrected "
                              "point estimate (separate, typically smaller, pass from --reps' Type-I sweep)")
    parser.add_argument("--effect-gold-mc", type=int, default=3000, metavar="N",
                         help="ppi mode: Monte Carlo reps used to estimate each scenario/test's gold-reference "
                              "null value (estimate_judge_bias_gold_null_values)")
    parser.add_argument("--no-typeI-check", action="store_true", default=False,
                         help="ppi mode: skip the base Type-I calibration sweep (build_judge_bias_sources, "
                              "by far the slowest single piece of --mode ppi). The other checks (effect/power/"
                              "comparison/factorial) don't consume its results, so this + --no-effect-check "
                              "--no-power-check --no-comparison-check --factorial-check runs JUST the factorial "
                              "sweep -- see official_args_ppi_factorial")
    parser.add_argument("--no-effect-check", action="store_true", default=False,
                         help="ppi mode: skip the bias/CI-coverage effect-size check, running Type-I calibration only")
    parser.add_argument("--no-power-check", action="store_true", default=False,
                         help="ppi mode: skip the power-under-bias check (build_ppi_power_sources), running "
                              "Type-I calibration (and, unless also disabled, the effect-size check) only")
    parser.add_argument("--no-comparison-check", action="store_true", default=False,
                         help="ppi mode: skip the 5-way estimator comparison (all_human/human_subset/llm_only/"
                              "llm_impute/ppi rejection rate vs. effect_size and label_frac, paired_t estimand)")
    parser.add_argument("--comparison-omnibus", action="store_true", default=False,
                         help="ppi mode: also pool the 4 omnibus/multi-group tests (anova_ind, anova_rep, friedman, "
                              "kruskal -- _COMPARISON_METHODS_OMNIBUS) through the SAME comparison_sources sweep "
                              "the 5-way estimator comparison already runs (--no-comparison-check's grid, NOT "
                              "--factorial-check's), producing a second 5-way comparison figure as a reader-facing "
                              "sanity check that the all_human > ppi > human_subset > llm_only/llm_impute story "
                              "isn't an artifact specific to the two-group/paired tests. Cheap relative to "
                              "--factorial-omnibus (comparison_sources is ~60 scenarios, not --factorial-check's "
                              "~6798), so this defaults on in official_args_ppi (unlike --factorial-omnibus, which "
                              "stays opt-in even there) -- see save_ppi_comparison_plot's label parameter.")
    parser.add_argument("--power-nlab-grid-check", action="store_true", default=False,
                         help="ppi mode: run the N x N_lab label/dataset-size power grid (likert), BOTH bias "
                              "directions -- build_ppi_power_nlab_grid_reinforcing_sources + build_ppi_power_"
                              "nlab_grid_opposing_sources, always run together (same convention as the "
                              "standard power check's opposing+reinforcing pair) -- investigating whether more "
                              "labels or a larger unlabeled pool change MWU's non-monotonic power anomaly, and "
                              "whether that anomaly is specific to bias reinforcing (vs. opposing) the effect. "
                              "Opt-in (default off): 4 N values x 3 N_lab values x 10 effect_size points x 2 "
                              "directions = 240 scenarios, on top of the standard power check -- restrict to "
                              "the test under investigation with e.g. --tests mwu for a fast targeted run. Uses "
                              "--effect-reps/--ppi-n-boot, same as the other power checks. Produces one plot "
                              "per (N, N_lab) cell plus one averaged summary plot per direction (see "
                              "save_ppi_power_nlab_grid_plots), plus one direction-comparison plot averaged "
                              "over the whole grid (save_ppi_power_nlab_grid_direction_plot).")
    parser.add_argument("--label-efficiency-reps", type=int, default=None, metavar="N",
                         help="ppi mode: reps for the label-efficiency check specifically. Defaults to "
                              "--effect-reps. Separate from it because the label-efficiency multipliers "
                              "feed a published rule of thumb and want more precision than the power/"
                              "comparison stages that also read --effect-reps; the official presets pin "
                              "this to 300.")
    parser.add_argument("--no-figure-titles", action="store_true", default=False,
                        help="ppi mode: draw label-efficiency figures without their in-figure "
                             "title and footnote strip, i.e. as the paper prints them "
                             "(equivalent to PPI_NO_FIGURE_TITLES=1, but settable per run). "
                             "Implied by the label-efficiency-only official preset.")
    parser.add_argument("--no-label-efficiency-check", action="store_true", default=False,
                         help="ppi mode: skip the label-efficiency check (run_ppi_label_efficiency_check) -- "
                              "for a fixed labeling budget, how many labels would a human-only classical test "
                              "need to match PPI's power, expressed against judge-human ALIGNMENT (not raw "
                              "llm_noise) so the curve is directly comparable across eval types; see "
                              "save_ppi_label_efficiency_plot's docstring.")
    parser.add_argument("--factorial-check", action="store_true", default=False,
                         help="ppi mode: run the full 7-factor factorial (bias_magnitude x N x N_lab x "
                              "label_mechanism x effect_size x bias_direction x llm_noise, continuous/paired_t) "
                              "-- opt-in (default off) since it's substantially more scenarios than the other "
                              "checks; see build_ppi_factorial_sources. Also produces the judge-human ALIGNMENT-"
                              "bucketed false-positive-rate view (weighted Cohen's kappa for likert, Pearson r for "
                              "continuous), derived from this same run's es=\"null\" cells rather than a separate "
                              "sweep -- see build_ppi_alignment_results_from_factorial/"
                              "save_ppi_alignment_sweep_plot's docstrings.")
    parser.add_argument("--factorial-reps", type=int, default=100, metavar="N",
                         help="ppi mode: reps for --factorial-check (default 100, a screening-tier rep count -- "
                              "bump toward --reps for a publication-precision confirmation pass)")
    parser.add_argument("--factorial-n-boot", type=int, default=500, metavar="N",
                         help="ppi mode: PPI bootstrap resample count for --factorial-check (default 500, "
                              "screening-tier -- bump toward --ppi-n-boot for a confirmation pass)")
    parser.add_argument("--factorial-likert-max", type=int, default=5, metavar="N",
                         help="ppi mode: top of the Likert scale's integer range for --factorial-check's likert "
                              "scenarios (default 5, the standard scale). A non-default value (e.g. 7) rescales "
                              "the SAME underlying distribution/bias/effect magnitudes onto a wider integer grid, "
                              "rather than generating a different one -- see scenarios.synthetic."
                              "build_ppi_factorial_sources' likert_max parameter. Ignored for continuous scenarios.")
    parser.add_argument("--factorial-fast-noise", action="store_true", default=False,
                         help="ppi mode: use PPI_FACTORIAL_NOISE_LEVELS_FAST (6 points, ratio 2, same 0.20 anchor) "
                              "instead of the full PPI_FACTORIAL_NOISE_LEVELS (11 points, ratio sqrt(2)) for "
                              "--factorial-check's es=\"null\" cells -- roughly halves the null-effect cell count "
                              "(and so the alignment-bucketed view's coverage/precision) for a quicker pass; the "
                              "noise=0.20 baseline GLM/heatmap outputs are unaffected either way. See "
                              "official_args_ppi_factorial_fast_noise for a ready-made preset.")
    parser.add_argument("--factorial-omnibus", action="store_true", default=False,
                         help="ppi mode: also run the 4 omnibus/multi-group tests (anova_ind, anova_rep, friedman, "
                              "kruskal -- _COMPARISON_METHODS_OMNIBUS) against --factorial-check's SAME sources, "
                              "on top of the default 4 two-group tests (ttest_welch/paired_t/mwu/wilcoxon). "
                              "Opt-in: uses generate_judge_bias_cell (the full generator, needed for the 3-group "
                              "structures anova_ind/kruskal and anova_rep/friedman read) for EVERY method now, not "
                              "just these 4, so enabling this meaningfully increases --factorial-check's runtime. "
                              "Reported and saved as its OWN pooled summary/log section (mean_of_4_omnibus), never "
                              "blended into the two-group tests' pooled rate -- see _COMPARISON_METHODS_OMNIBUS' "
                              "docstring for why (different hypothesis: 3-group omnibus vs. two-group location-shift).")
    parser.add_argument("--comparison-omnibus-tests", nargs="+", default=None, metavar="TEST",
                         help="ppi mode: REPLACE --comparison-omnibus's method set (default "
                              "anova_ind/anova_rep/friedman/kruskal) with these names. Same rationale as "
                              "--factorial-omnibus-tests: an override, not an addition, so a default run "
                              "still reproduces the committed five-way-comparison-omnibus figure. Use it "
                              "to re-run one omnibus test in isolation and splice.")
    parser.add_argument("--factorial-two-group-tests", nargs="+", default=None, metavar="TEST",
                         help="ppi mode: REPLACE --factorial-check's two-group method set (default "
                              "ttest/ttest_welch/paired_t/mwu/wilcoxon) with these names. Use when "
                              "re-running an OMNIBUS test in isolation: the two-group tests are then "
                              "pure overhead, but they cannot simply be dropped -- the GLM report, the "
                              "heatmap/alignment plots and the alignment sweep all consume them, so an "
                              "empty set breaks those stages. Passing the cheapest single test instead "
                              "(--factorial-two-group-tests ttest, ~0.007 s/src vs mwu's ~0.55) keeps "
                              "every downstream stage working AND preserves the cross-run machinery "
                              "anchor, while removing ~39%% of a single-omnibus run's cost. Must be "
                              "non-empty.")
    parser.add_argument("--factorial-omnibus-tests", nargs="+", default=None, metavar="TEST",
                         help="ppi mode: REPLACE --factorial-omnibus's method set (default "
                              "anova_ind/anova_rep/friedman/kruskal) with these names. Deliberately an "
                              "override rather than an addition to _COMPARISON_METHODS_OMNIBUS: appending "
                              "would change what a plain --factorial-omnibus run produces and so break "
                              "reproducibility of the committed typeI_factorial_*_compact.png figures, "
                              "whose provenance run used the default 4. Use it to re-run ONE omnibus test "
                              "in isolation (e.g. --factorial-omnibus-tests kruskal kruskal_influence) and "
                              "splice the resulting CSV against an existing full run. Note --tests does NOT "
                              "reach the factorial check -- this flag is the only way to change its methods.")
    parser.add_argument("--factorial-no-power-tune", action="store_true", default=False,
                         help="ppi mode: disable PPI++ power-tuning (power_tune=False, i.e. fixed lambda=1, the "
                              "original 2023 PPI estimator) for --factorial-check's two-group methods (ttest_welch/"
                              "paired_t/mwu/wilcoxon), instead of the evalstats.ppi.correct() default (power_tune=True). "
                              "Exists to isolate whether PPI++ power-tuning is the cause of "
                              "'corrected worse than uncorrected' null cells reappearing in the two-group family -- "
                              "run once with and once without this flag on the same seed/sources and diff the two "
                              "results' zero-bias null cells. Ignored by --factorial-omnibus' 4 omnibus methods, "
                              "which don't use power-tuning at all (see evalstats/ppi.py's power_tune docstring).")
    parser.add_argument("--factorial-alignment-mc", type=int, default=20000, metavar="N",
                         help="ppi mode: Monte Carlo sample size for --factorial-check's alignment-bucketed view's "
                              "per-(eval_type, llm_noise, bias_delta) alignment measurement (measure_judge_alignment "
                              "-- a separate, large, effectively noise-free calibration draw, not the small "
                              "labeled-subset the Type-I sweep itself uses; default 20000 keeps the realized "
                              "alignment percentage stable to within ~1 point). Ignored unless --factorial-check "
                              "produces es=\"null\" cells. Also used by --factorial-check-binary's alignment view.")
    parser.add_argument("--factorial-check-binary", action="store_true", default=False,
                         help="ppi mode: run the binary analogue of --factorial-check (bias_magnitude x N x N_lab x "
                              "label_mechanism x effect_size x bias_direction x llm_noise, ttest_welch/paired_t only "
                              "-- see build_ppi_factorial_sources_binary/_COMPARISON_METHODS_BINARY). Separately "
                              "opt-in from --factorial-check: different bias/noise magnitude convention "
                              "(PPI_BINARY_BIAS_MAGNITUDES/PPI_BINARY_NOISE_LEVELS), different pooled test set, so "
                              "kept as its own toggle rather than folded into the same flag. Reuses "
                              "--factorial-reps/--factorial-n-boot/--factorial-alignment-mc.")
    parser.add_argument("--nformula-check", action="store_true", default=False,
                         help="ppi mode: run the label-efficiency N-formula check (run_ppi_nformula_check) -- "
                              "extends the label-efficiency check (--no-label-efficiency-check) by ALSO sweeping "
                              "N (PPI_NFORMULA_N_VALUES) and effect size (PPI_NFORMULA_EFFECT_FRACS), not just "
                              "N_lab and judge alignment -- a one-off, opt-in (default off) check for deriving/"
                              "verifying a closed-form rule-of-thumb formula for the label-efficiency multiplier "
                              "that includes N explicitly and holds across effect sizes, distinct from --mode "
                              "ppi's main label-efficiency path. See official_args_ppi_nformula for a ready-made "
                              "official-precision preset, selectable on its own from the --official-tests menu.")
    parser.add_argument("--rho-drift-only", action="store_true", default=False,
                         help="ppi mode: run ONLY the rho-drift check, skipping the full "
                              "PPI calibration sweep that normally precedes it. That sweep is "
                              "115 scenarios at --reps/--ppi-n-boot and dominates the runtime, "
                              "so reaching the drift phase otherwise means waiting it out (or "
                              "shrinking it with --reps/--ppi-n-boot, which the --rho-drift-* "
                              "flags do NOT control). Implies --rho-drift-check.")
    parser.add_argument("--rho-drift-check", action="store_true", default=False,
                         help="ppi mode: run the rho effect-size drift check (run_ppi_rho_drift_check) -- "
                              "holds judge quality pinned at PPI_RHO_DRIFT_ALIGNMENT_TARGET and sweeps the "
                              "TRUE EFFECT (PPI_RHO_DRIFT_EFFECT_FRACS, much wider than the label-efficiency "
                              "check's 0.15-0.35), then inverts the measured multiplier back to the rho^2 it "
                              "implies. Tests _method_rho2's standing assumption that rho is a property of the "
                              "judge alone: exact for the mean-type estimands, false for every rank/dominance "
                              "one (see _METHOD_CORR_KIND's caveat). Opt-in, default off. See "
                              "official_args_ppi_rho_drift for a ready-made official-precision preset, "
                              "selectable on its own from the --official-tests menu.")
    parser.add_argument("--rho-drift-reps", type=int, default=None, metavar="N",
                         help="ppi mode: reps for --rho-drift-check. Default 200 with --rho-drift-check, "
                              "but 2000 with --rho-drift-only -- that flag exists to run this check ON ITS "
                              "OWN, which is the official-precision use case, and 200 cannot support the "
                              "control (see official_args_ppi_rho_drift). Pass this flag to override either "
                              "default. This check reads a VARIANCE "
                              "ratio rather than inverting a power curve, so it needs reps for precision on a "
                              "second moment -- roughly sqrt(2/reps) relative error, i.e. ~10%% at 200 and ~3%% "
                              "at 2000. Drifts below ~5%% need the higher tier to be distinguishable from noise.")
    parser.add_argument("--rho-drift-n-boot", type=int, default=500, metavar="N",
                         help="ppi mode: PPI bootstrap resample count for --rho-drift-check (default 500).")
    parser.add_argument("--rho-drift-shape", type=str, default=None, metavar="LABEL",
                         help="ppi mode: truth-marginal shape for --rho-drift-check "
                              "(a ShapeSpec label, e.g. 'cont-near-center'); default is "
                              "the eval type's representative shape. The default "
                              "'cont-right-skew' pins ~23%% of continuous truth values at "
                              "exactly 0, so moving the mean off that floor changes the "
                              "realized spread -- which shows up as a ~+5%% rise in "
                              "paired_t's rho^2 that is a property of the BOUND, not of "
                              "the estimator. 'cont-near-center' clips ~11%% and holds "
                              "both mean-type methods flat to ~1%%.")
    parser.add_argument("--rho-drift-nlab", type=int, default=100, metavar="N",
                         help="ppi mode: N_lab for --rho-drift-check (default 100, at N=PPI_LABEL_EFF_N). "
                              "label_frac is back-solved from it, the same absolute-N_lab convention "
                              "build_ppi_label_efficiency_sources uses.")
    parser.add_argument("--rho-drift-effects", type=float, nargs="+", default=None, metavar="D",
                         help="ppi mode: override PPI_RHO_DRIFT_EFFECT_FRACS for --rho-drift-check.")
    parser.add_argument("--nformula-reps", type=int, default=100, metavar="N",
                         help="ppi mode: reps for --nformula-check (default 100, a screening-tier rep count -- "
                              "bump toward --effect-reps for a publication-precision confirmation pass, see "
                              "official_args_ppi_nformula).")
    parser.add_argument("--nformula-n-boot", type=int, default=500, metavar="N",
                         help="ppi mode: PPI bootstrap resample count for --nformula-check (default 500, "
                              "screening-tier -- bump toward --ppi-n-boot for a confirmation pass).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential).")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset: pairwise + multiarm, synthetic data.
    PPI calibration is a separate, much slower preset (official_args_ppi)
    split out so it can be run/skipped independently in the
    --official-tests menu -- see official_variants().

    ``k_arms`` sweeps up to k=20 (190 pairwise comparisons -- pairs grow as
    k(k-1)/2, so this is roughly 4x the comparisons of k=10's 45) rather
    than stopping at 10. This is deliberate, not just "more thorough": a
    real multi-model comparison (an LLM leaderboard slice, an ablation with
    many variants) routinely has 10-20+ arms, and Bonferroni's per-comparison
    alpha budget (alpha/pairs) shrinks toward that same rate, which is
    exactly the regime where its extra conservativeness over max-T
    (--mode simultaneous_ci) -- and, on the p-value side, over max_t/holm/
    fdr_bh's better power at matched FWER (--mode multiarm) -- becomes most
    visible. Costs ~3.8x the k-sweep's compute vs. stopping at k=10, since
    per-cell cost tracks the pair count; both --mode multiarm and
    --mode simultaneous_ci reuse this same official_args() preset (see
    official_args_simultaneous_ci) and its k_arms sweep.

    Excludes "grades" from eval_types (also inherited by official_args_ppi/
    official_args_simultaneous_ci, which derive from this preset):
    "continuous" already covers the [0, 1]-scale case well (grades is just
    continuous rescaled to 0-100), while "likert" is kept as a genuinely
    distinct limiting case (integer-valued, few levels). Dropping grades
    cuts a third eval type out of the official sweep's runtime for no real
    loss of coverage."""
    return argparse.Namespace(
        mode="pairwise_multiarm", reps=300, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source="synthetic", scenario_suite="expanded", eval_types=["binary", "continuous", "likert"], sizes=[10, 20, 30, 50, 75, 100],
        runs=1, statistic="mean",
        bootstrap_n=2000, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3, 5, 10, 20], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=None, ppi_n_boot=2000, effect_reps=200, effect_gold_mc=3000, no_effect_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_args_pairwise(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for pairwise-only calibration (synthetic data).
    Split out from official_args() (which runs mode="pairwise_multiarm",
    i.e. both pairwise and multiarm together) so the pairwise sweep alone
    can be re-run on its own -- e.g. after a pairwise-specific performance
    or correctness fix -- without paying for the separate (and unrelated,
    much slower at high k) multiarm sweep every time."""
    args = official_args(base_seed)
    args.mode = "pairwise"
    return args


def official_args_multiarm(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for multiarm-only calibration (synthetic data).
    Split out from official_args() (which runs mode="pairwise_multiarm",
    i.e. both pairwise and multiarm together) so the multiarm sweep alone
    can be re-run on its own -- e.g. after a multiarm-specific performance
    or correctness fix -- without paying for the separate (and unrelated)
    pairwise sweep every time.

    Overrides official_args()'s sizes with official_args_simultaneous_ci's
    coarser 6-point n=15..500 sweep (rather than official_args()'s denser
    6-point sweep stopping at 100): multiarm's resampling-based FWER
    corrections (max_t/romano_wolf/westfall_young) are the direct p-value-
    side analogue of simultaneous_ci's CI constructions (same bootstrap-t/
    step-down machinery, same k-arm sources), so sweeping the same n range
    makes the two modes' small-N-to-large-N comparisons directly comparable
    instead of stopping multiarm's sweep short of the large-N regime
    simultaneous_ci's sweep was chosen to cover.

    Also overrides official_args()'s bootstrap_n=2000 with 5000:
    romano_wolf/westfall_young/boot's FWER ran consistently ~0.001-0.002
    above nominal alpha at bootstrap_n=500-2000 (confirmed via direct
    n=500-2000 sweeps, holding even at small k), traced to Monte Carlo noise
    in estimating the joint max-statistic's upper-tail quantile from too few
    draws -- not a correction-logic bug or a k-dependent effect (ruled out by
    `boot`, a structurally different bootstrap-based correction, showing the
    same excess). 5000 draws resolved it. This is no longer as costly as it
    used to be -- _bootstrap_t_matrix's resample construction was rewritten
    from an O(k_pairs*n_bootstrap*n) gather to a counts/matmul formulation
    (~12-27x faster for the "bootstrap" mode max_t/romano_wolf/boot share,
    ~2.3x for westfall_young's "permutation" mode). Left at 2000 for
    official_args()'s other consumers (pairwise, ppi) -- this finding is
    specific to resampling-based FWER corrections. simultaneous_ci sets its
    own 5000 (see official_args_simultaneous_ci): its `boot` is the same
    joint-bootstrap estimator, so the same argument carries."""
    args = official_args(base_seed)
    args.mode = "multiarm"
    args.sizes = [15, 30, 50, 100, 200, 500, 1000]
    args.bootstrap_n = 5000
    args.reps = 500
    return args


def official_args_ppi(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for PPI-corrected calibration (synthetic data --
    ppi has no real-data variant). Split out from official_args() since it's
    by far the slowest of the pvalues sub-modes (43 judge-bias scenarios x
    ~11 tests x reps), so it can be selected or skipped on its own in the
    --official-tests menu instead of always riding along with the faster
    pairwise/multiarm sweep. Runs the FULL rigorous PPI evaluation built up
    over this harness's development: Type-I calibration, the bias/coverage
    effect-size check, the power-under-bias check (plus its bias-direction
    and no-bias companions), the 5-way estimator comparison, the N x N_lab
    grid, and -- via factorial_check below -- the full 7-factor factorial
    (build_ppi_factorial_sources) plus the judge-human alignment-bucketed
    view it now also produces (build_ppi_alignment_results_from_factorial).
    All of these except factorial_check are
    already on by default in run() (--no-power-check/--no-comparison-check
    are opt-OUT), so this preset only needs to explicitly enable
    factorial_check (opt-IN by default, given its larger scenario count)
    and give it the SAME precision tier the other secondary checks already
    run at here -- effect_reps/ppi_n_boot (200/2000) -- rather than
    inventing a third reps/n_boot tier alongside --reps and
    --factorial-check's own screening-tier CLI default (100/500, meant for
    fast interactive iteration, not a result worth citing).

    Binary's power/comparison sweeps (build_ppi_power_sources_binary,
    build_ppi_comparison_label_frac_sources_binary) already ride along here
    for free -- official_args()'s eval_types already includes "binary", and
    those checks are opt-OUT (--no-power-check/--no-comparison-check), the
    same way continuous/likert/grades' versions are. factorial_check_binary
    is the one genuinely new opt-in toggle (separate from factorial_check:
    different bias/noise convention, different pooled test set -- see
    _COMPARISON_METHODS_BINARY), enabled here at the same precision tier.

    comparison_omnibus=True: also pools _COMPARISON_METHODS_OMNIBUS through
    the 5-way estimator comparison sweep above (comparison_sources -- NOT
    factorial_check's much larger grid), producing a second 5-way figure as
    a sanity check that the estimator-comparison story isn't specific to
    the two-group/paired tests. Unlike factorial_omnibus (deliberately left
    off here, opt-in only via the standalone official_args_ppi_factorial
    preset, since it meaningfully increases --factorial-check's runtime),
    this is cheap -- comparison_sources is ~60 scenarios vs. --factorial-
    check's ~6798 -- so it defaults on for every official_args_ppi* preset."""
    args = official_args(base_seed)
    args.mode = "ppi"
    # 300, not effect_reps' 200: the label-efficiency multipliers back a
    # published rule of thumb, so the paper's runs have always used the
    # higher rep count (see the Aug-2026 reps300 run the figures came
    # from). Pinned here so --official-tests can't silently produce a
    # noisier version than the one that was published.
    args.label_efficiency_reps = 300
    args.factorial_check = True
    args.factorial_reps = args.effect_reps
    args.factorial_n_boot = args.ppi_n_boot
    args.factorial_check_binary = True
    args.comparison_omnibus = True
    return args


def official_args_ppi_no_lmm(base_seed: int = 42) -> argparse.Namespace:
    """Same as official_args_ppi (same checks, same scenarios, same reps/
    n_boot), except --tests excludes the three LMM-based methods (lmm/
    lmm_factorial/lmm_runs). LMM is profiled at ~70% of --mode ppi's total
    runtime (its mixed-model fits dominate build_judge_bias_sources' Type-I
    sweep and the power check, both of which iterate active_tests over
    every scenario) -- see run_ppi_simulation/_run_ppi_cell's docstrings.
    This preset exists purely for a faster quality-check pass (e.g. after a
    change to one of the OTHER PPI tests, or before a merge) when LMM's own
    calibration isn't what's being verified, at a fraction of the
    wall-clock cost.

    The factorial/N x N_lab comparison sweep (_COMPARISON_METHODS) never
    ran LMM to begin with (it's ttest_welch/paired_t/mwu/wilcoxon
    only), so this only changes the main Type-I sweep and power check --
    it does NOT skip factorial_check itself; pair with --no-factorial-check
    (or official_args_ppi_factorial's already-LMM-free scope) if the
    factorial sweep's own cost also needs trimming."""
    args = official_args_ppi(base_seed)
    args.tests = [
        m.name for m in PPI_OFFICIAL_TEST_METHODS
        if m.name not in (LMM.name, LMM_FACTORIAL.name, LMM_RUNS.name)
    ]
    return args


def official_args_ppi_factorial(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the full 7-factor factorial sweep
    (build_ppi_factorial_sources, including its judge-human alignment-
    bucketed view -- build_ppi_alignment_results_from_factorial), split out
    from official_args_ppi the same way official_args_ppi itself is split
    from official_args -- lets --official-tests run/skip it independently,
    e.g. to iterate on the factorial analysis alone without re-paying for
    the base Type-I sweep (build_judge_bias_sources' ~85+ scenarios x ~11
    tests x reps -- by far the slowest piece of --mode ppi). Safe to isolate
    this way because the factorial sweep is fully self-contained: its own
    sources (build_ppi_factorial_sources), its own run_ppi_comparison_
    simulation call, no dependency on the Type-I/effect/power/comparison/
    label-efficiency checks' results -- this is a real subset of
    official_args_ppi's work, not an approximation of it. Disables every
    other --mode ppi check via --no-typeI-check/--no-effect-check/
    --no-power-check/--no-comparison-check/--no-label-efficiency-check (all
    opt-out; harmless to set even though official_args_ppi doesn't set them,
    since their defaults already run). label-efficiency in particular is
    NOT free to leave on here the way the others are "harmless" to
    explicitly disable: it defaults to running (reps=200, n_boot=1000,
    independent of factorial_reps/factorial_n_boot) and isn't scoped by
    any factorial_check flag, so omitting this line would silently run it
    as an uninvited addition to what this preset's name/docstring promise
    is "JUST" the factorial sweep -- caught when a --factorial-check-only
    dry run kept running well past when the (tiny, --factorial-reps 2)
    factorial checks should have finished.

    factorial_omnibus=True: also runs the 4 omnibus/multi-group tests
    (anova_ind/anova_rep/friedman/kruskal -- _COMPARISON_METHODS_OMNIBUS)
    against these same factorial sources, not just the original 4 two-group
    tests -- added once the main OFAT sweep (build_judge_bias_sources) and
    this factorial sweep's own two-group tests confirmed those 4 held up
    reasonably well under the combined-factor stress test, making it worth
    checking whether anova/friedman/kruskal (kruskal in particular already
    flagged as a milder, more diffuse Type-I outlier in the OFAT sweep) also
    hold up here, or blow up the way MWU's global rectifier did before the
    (since-reverted, and since removed entirely -- see MWU in methods.py)
    local-rectifier fix temporarily replaced it. NOT set on official_args_ppi/
    official_args_ppi_no_lmm (the "run
    everything" presets, already by far the slowest --mode ppi variants) --
    only this standalone factorial-only preset, so the extra cost (roughly
    2x the method count, using the full generator now for every method) is
    opt-in at the granularity where it's easiest to run/iterate on its own."""
    args = official_args_ppi(base_seed)
    args.no_typeI_check = True
    args.no_effect_check = True
    args.no_power_check = True
    args.no_comparison_check = True
    args.no_label_efficiency_check = True
    args.factorial_omnibus = True
    return args


def official_args_ppi_label_efficiency(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the label-efficiency sweep.

    Split out from official_args_ppi the same way official_args_ppi_factorial
    is, and for the same reason: the label-efficiency check is what the
    paper's rho^2 rule of thumb, its multiplier table (tab:le-mult) and four
    of its figures rest on, but reaching it through official_args_ppi means
    also paying for the Type-I sweep (build_judge_bias_sources' ~85 scenarios
    x ~11 tests x reps -- by far the slowest piece of --mode ppi), the
    effect/power/comparison checks, and the 7-factor factorial sweep. None of
    those feed it: the label-efficiency check builds its own sources and
    consumes no other check's results, so this is a real subset of
    official_args_ppi's work rather than an approximation.

    Runs at label_efficiency_reps=500, above official_args_ppi's 300. Being
    a preset rather than a documented flag combination is the point: the CLI's
    own default is lower still, so a hand-rolled `--mode ppi --no-*-check`
    invocation silently produces a lower-precision sweep that is NOT
    comparable with the paper's, and the filename records only the rep count,
    not that it was reduced. The runs behind the current table show exactly
    that split -- the paper's is reps300, two later diagnostic runs are
    reps200.

    One --mode ppi run at this preset writes all four figures the paper
    prints, under <stem>:
        _compact.png                  -> labeleff_compact (main text)
        _plot_lookup_row.png          -> labeleff_lookup_row
        _plot_threshold_pooled.png    -> labeleff_threshold_pooled
        _plot_noisefamily_compact.png -> labeleff_noisefamily_compact
    The noise-family pair needs >=2 noise families, so it is absent from a
    binary-only sweep (binary is deliberately gaussian-only, not being
    shape-sensitive); all three eval types are kept here, so it renders.

    Regenerate the multiplier table from the same run with
    simulations/make_appendix_tables.py --run <stem>."""
    args = official_args_ppi(base_seed)
    args.no_typeI_check = True
    args.no_effect_check = True
    args.no_power_check = True
    args.no_comparison_check = True
    # factorial is opt-in and official_args_ppi turns it on; this preset is
    # not "everything except Type-I", it is the label-efficiency sweep alone.
    args.factorial_check = False
    args.factorial_check_binary = False
    # This preset exists to produce the paper's figures, so it draws them the
    # way the paper prints them -- caption-only, no in-figure title.
    args.no_figure_titles = True
    # 500, not official_args_ppi's 300. At 300 the multiplier's bootstrap CIs
    # are wide enough that "by rho^2~0.6 it reaches 2.0-2.8x" flipped between
    # two runs on Likert's minimum (2.01 vs 1.92); 500 narrows the interval by
    # ~1.3x, enough to make that call while staying a sweep someone will
    # actually re-run. Not higher: this check runs several progress phases
    # (per eval type x noise family), each with its own bar, so its wall clock
    # is a multiple of what one bar's ETA suggests -- 1000 was chosen off a
    # reading of one phase as if it were the whole run.
    args.label_efficiency_reps = 500
    return args


def official_args_ppi_nformula(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the label-efficiency N-formula check
    (run_ppi_nformula_check) -- split out from official_args_ppi the same
    way official_args_ppi_factorial is, so --official-tests can run/skip it
    independently of the base Type-I sweep and the other --mode ppi checks.
    Disables every other opt-out check (--no-typeI-check/--no-effect-check/
    --no-power-check/--no-comparison-check/--no-label-efficiency-check) so
    this preset runs ONLY the N-formula sweep, not also the (much slower)
    base Type-I sweep or the original label-efficiency check it extends.

    nformula_reps/nformula_n_boot are set to the SAME precision tier the
    other secondary checks already run at (effect_reps/ppi_n_boot,
    200/2000) -- not --nformula-check's own screening-tier CLI default
    (100/500, meant for fast interactive iteration while designing the
    grid, not a result worth citing) -- the same convention official_args_
    ppi_factorial uses for factorial_reps/factorial_n_boot."""
    args = official_args_ppi(base_seed)
    args.no_typeI_check = True
    args.no_effect_check = True
    args.no_power_check = True
    args.no_comparison_check = True
    args.no_label_efficiency_check = True
    args.nformula_check = True
    args.nformula_reps = args.effect_reps
    args.nformula_n_boot = args.ppi_n_boot
    return args


def official_args_ppi_rho_drift(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the rho effect-size drift check
    (run_ppi_rho_drift_check) -- split out the same way
    official_args_ppi_nformula and official_args_ppi_factorial are, so
    --official-tests can run it on its own without the (much slower) base
    Type-I sweep or the other --mode ppi checks.

    rho_drift_reps is set to 2000, an order of magnitude above the CLI
    default and above the tier the other secondary checks use. That is not
    over-provisioning: this check reads a VARIANCE ratio directly rather than
    inverting a power curve, so its precision goes as sqrt(2/reps) -- ~10%%
    relative at 200 reps, which would swamp the sub-5%% drifts that separate
    "flat" (the mean-type methods, whose whole claim is exact invariance) from
    "mildly drifting". The rank methods' -13%% to -38%% would survive 200 reps;
    proving the mean-type ones FLAT is what needs the precision.

    Continuous only. The drift is a property of the estimand's influence
    function, not of the eval type, and continuous is the cheapest place to
    show it without discretisation muddying the rank statistics."""
    args = official_args_ppi(base_seed)
    args.no_typeI_check = True
    args.no_effect_check = True
    args.no_power_check = True
    args.no_comparison_check = True
    args.no_label_efficiency_check = True
    # official_args_ppi turns the 7-factor factorial ON; a preset described as
    # "JUST the rho drift check" must turn it back off, or selecting it costs a
    # ~6h factorial sweep nobody asked for (measured: reps200 factorial ran
    # 22:45->04:51). The label-efficiency preset already does this; this one
    # was missing the line.
    args.factorial_check = False
    args.rho_drift_check = True
    args.rho_drift_reps = 2000
    args.rho_drift_n_boot = args.ppi_n_boot
    args.rho_drift_nlab = 100
    args.eval_types = ["continuous"]
    return args


def official_args_ppi_factorial_likert7(base_seed: int = 42) -> argparse.Namespace:
    """Same as official_args_ppi_factorial, except likert scenarios are
    generated on a 1-7 scale instead of the standard 1-5 (factorial_likert_max
    = 7 -- see build_ppi_factorial_sources' likert_max parameter). Continuous
    scenarios are unaffected (likert_max is a no-op for them).

    Exists to test a specific hypothesis raised after the first factorial run
    (see simulations/out/official_20260718_213255): PPI-corrected
    Mann-Whitney's Type-I rate blew up specifically for likert scenarios
    under severe MNAR labeling (up to 0.445 at et=likert/bm=severe/n=400/
    nlab=80/lm=mnar_strong), while paired_t/wilcoxon/ttest_welch stayed
    well-calibrated in that exact same scenario, AND mwu itself stayed
    well-calibrated on continuous (effectively tie-free) data under the same
    severe MNAR mechanism -- pointing at Likert's coarse, heavily-tied 5-level
    discretization (not MNAR alone, and not rank tests generally) as the
    likely aggravating factor for mwu's independent-groups midrank
    construction specifically. Comparing this run's likert Type-I/power
    numbers against the 1-5 run's is the intended follow-up analysis."""
    args = official_args_ppi_factorial(base_seed)
    args.factorial_likert_max = 7
    return args


def official_args_ppi_factorial_fast_noise(base_seed: int = 42) -> argparse.Namespace:
    """Same as official_args_ppi_factorial, except llm_noise sweeps
    PPI_FACTORIAL_NOISE_LEVELS_FAST (6 points, ratio 2) instead of the full
    PPI_FACTORIAL_NOISE_LEVELS (11 points, ratio sqrt(2)) on the es="null"
    cells -- roughly halves that subset's cell count (and so the runtime of
    the slowest part of the factorial run, since non-null cells are
    unaffected either way). The noise=0.20 baseline GLM/heatmap outputs are
    numerically identical to a full-grid run at the same seed (that baseline
    slice doesn't depend on which OTHER noise levels were swept); only the
    judge-human alignment-bucketed view gets coarser, with fewer buckets
    populated. Meant as a faster check to run before committing to the full
    grid's longer runtime, not a replacement for it -- see
    PPI_FACTORIAL_NOISE_LEVELS_FAST's docstring."""
    args = official_args_ppi_factorial(base_seed)
    args.factorial_fast_noise = True
    return args


def official_args_ppi_factorial_binary(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the binary factorial sweep
    (build_ppi_factorial_sources_binary), split out the same way
    official_args_ppi_factorial is split from official_args_ppi -- lets
    --official-tests run/skip it independently. Self-contained (its own
    sources, its own run_ppi_comparison_simulation call with
    _COMPARISON_METHODS_BINARY), so isolating it costs nothing beyond not
    re-running the base Type-I sweep. Runs at the SAME precision tier as
    official_args_ppi_factorial (effect_reps/ppi_n_boot, 200/2000)."""
    args = official_args_ppi_factorial(base_seed)
    args.factorial_check = False
    args.factorial_check_binary = True
    return args


def official_args_simultaneous_ci(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for simultaneous-CI calibration only (synthetic
    data). Split out from official_args() for the same reason as
    official_args_ppi -- lets --official-tests select/skip it independently
    of the faster pairwise/multiarm sweep, even though it shares those
    modes' k-arm sources.

    Overrides two of official_args()'s defaults. (It also assigns
    scenario_suite="expanded", but that is the value official_args already
    sets -- the assignment only pins it against a change to the base preset,
    it does not override anything. This docstring previously described that
    line as selecting the smaller "standard" catalog, which was never what
    the code did.)
    - sizes spans n=15 to n=500 rather than official_args()'s n=10 to n=100
      (both are 6 points -- this one reaches further, it is not denser):
      save_simultaneous_ci_violin_vs_n_plot's per-n grouped violins (the
      canonical closed-form CI plus its sidak/boot widenings, alongside
      none/Bonferroni/max-T) are most informative for deciding a real
      default when they span the full small-N (where multiplicity eats the
      most power) to
      large-N (where all constructions should converge) range a real
      evaluation might have, not just the ~30 crossover this preset
      historically anchored on -- kept to 6 points, not official_args()'s
      density, since this mode's per-cell cost (bootstrap_t's nested double
      bootstrap, k(k-1)/2 marginal pairs plus the shared max-T resample,
      times the canonical/sidak/boot rows on top for binary sources) is
      already the most expensive of the pvalues sub-modes.
    - bootstrap_n=5000, overriding official_args()'s 2000, matching
      official_args_multiarm and real_official_args_simultaneous_ci. `boot`
      here IS the joint bootstrap whose FWER ran ~0.001-0.002 hot at
      bootstrap_n=500-2000 in multiarm (see official_args_multiarm), from
      Monte Carlo noise in the joint max-statistic's upper-tail quantile --
      the same estimator, so the same fix applies. This variant was the last
      resampling preset still at 2000, which left the synthetic
      simultaneous-CI figures inconsistent with both their real-data
      counterparts and the multi-arm figures they sit beside.
    """
    args = official_args(base_seed)
    args.mode = "simultaneous_ci"
    args.scenario_suite = "expanded"
    args.sizes = [15, 30, 50, 100, 200, 500]
    args.bootstrap_n = 5000
    return args


def real_official_args(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for real data sources (pairwise + multiarm only;
    ppi has no real-data variant, and simultaneous_ci is its own preset --
    see real_official_args_simultaneous_ci). Requires network/HF access."""
    return argparse.Namespace(
        mode="pairwise_multiarm", reps=300, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source="real", scenario_suite="expanded", eval_types=None, sizes=[10, 20, 30, 50, 75, 100],
        runs=1, statistic="mean",
        bootstrap_n=2000, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3, 5, 10], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=None, ppi_n_boot=2000, effect_reps=200, effect_gold_mc=3000, no_effect_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def real_official_args_pairwise(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for pairwise-only calibration, real data. Split
    out from real_official_args() (mode="pairwise_multiarm") the same way
    official_args_pairwise is split from official_args() -- lets the
    (network/HF-dependent) real-data pairwise sweep be re-run on its own
    without also paying for the real-data multiarm sweep."""
    args = real_official_args(base_seed)
    args.mode = "pairwise"
    return args


def real_official_args_multiarm(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for multiarm-only calibration, real data. Split
    out from real_official_args() (mode="pairwise_multiarm") the same way
    real_official_args_pairwise is -- lets the (network/HF-dependent)
    real-data multiarm sweep (FWER + best-arm power across
    none/holm/bonferroni/fdr_bh/hochberg/shaffer/friedman_nemenyi/max_t/
    romano_wolf/westfall_young -- see MULTIARM_CORRECTION_METHODS) be
    re-run on its own without also paying for the real-data pairwise sweep.

    Also bumps bootstrap_n from real_official_args()'s 2000 to 5000 -- see
    official_args_multiarm's docstring for why (Monte Carlo noise in the
    joint max-statistic's quantile at low bootstrap_n, not a k-dependent or
    correction-logic issue); applies identically regardless of data source."""
    args = real_official_args(base_seed)
    args.mode = "multiarm"
    args.bootstrap_n = 5000
    args.sizes=[10, 15, 20, 30, 50, 75, 100]
    args.k_arms = [3, 5]
    args.reps = 500
    return args


def real_official_args_simultaneous_ci(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for simultaneous-CI calibration only, real data
    (real multi-arm sources -- see build_real_multiarm_sources). Split out
    from real_official_args() the same way official_args_simultaneous_ci is."""
    args = real_official_args(base_seed)
    args.mode = "simultaneous_ci"
    args.bootstrap_n = 5000
    args.sizes=[10, 15, 20, 30, 50, 75, 100]
    args.k_arms = [3, 5]
    return args



def official_args_ppi_likert(base_seed: int = 42) -> argparse.Namespace:
    """Likert-only variant of official_args_ppi.

    Added 2026-08-24 alongside the judge-rounding fix in
    scenarios.synthetic.generate_judge_bias_cell: Likert judge scores were
    left on a continuous scale (only the ground truth was rounded), so the
    judge used for inference was not the integer-reporting judge a Likert
    rubric actually produces -- and not the judge
    measure_judge_alignment reported agreement for. Only likert changed;
    binary goes through _jb_llm_binary (already 0/1) and continuous is
    genuinely continuous, and a label-efficiency A/B confirmed this
    empirically (continuous rho^2 delta was exactly 0.0000 at every
    alignment target, binary unchanged within MC noise).

    So re-running the whole PPI suite would burn hours recomputing two
    eval types whose numbers cannot have moved. This preset restricts the
    sweep to likert, letting the existing binary/continuous results stand.
    eval_types filters SOURCES before run_ppi_simulation (not results
    afterwards), so the compute really is skipped.
    """
    args = official_args_ppi(base_seed)
    args.eval_types = ["likert"]
    args.factorial_check_binary = False   # binary unaffected by the fix
    return args


def official_args_ppi_factorial_likert(base_seed: int = 42) -> argparse.Namespace:
    """Likert-only variant of official_args_ppi_factorial -- the factorial
    plus judge-human alignment sweep on its own, for the same reason as
    official_args_ppi_likert. This is the one that feeds the alignment
    figure."""
    args = official_args_ppi_factorial(base_seed)
    args.eval_types = ["likert"]
    args.factorial_check_binary = False
    return args


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """All official-test variants for this case, as (label, args) pairs."""
    return [
        ("synthetic (pairwise + multiarm)", official_args(base_seed)),
        ("synthetic (pairwise)", official_args_pairwise(base_seed)),
        ("synthetic (multiarm)", official_args_multiarm(base_seed)),
        ("synthetic (ppi)", official_args_ppi(base_seed)),
        ("synthetic (ppi, no LMM)", official_args_ppi_no_lmm(base_seed)),
        ("synthetic (ppi factorial only, fast noise)", official_args_ppi_factorial_fast_noise(base_seed)),
        ("synthetic (ppi factorial only)", official_args_ppi_factorial(base_seed)),
        ("synthetic (ppi factorial only, likert 1-7)", official_args_ppi_factorial_likert7(base_seed)),
        ("synthetic (ppi factorial only, binary)", official_args_ppi_factorial_binary(base_seed)),
        ("synthetic (ppi, LIKERT ONLY -- judge-rounding re-run)", official_args_ppi_likert(base_seed)),
        ("synthetic (ppi factorial only, LIKERT ONLY)", official_args_ppi_factorial_likert(base_seed)),
        ("synthetic (ppi label-efficiency only)", official_args_ppi_label_efficiency(base_seed)),
        ("synthetic (ppi n-formula check only)", official_args_ppi_nformula(base_seed)),
        ("synthetic (ppi rho effect-size drift check only)", official_args_ppi_rho_drift(base_seed)),
        ("synthetic (simultaneous CI)", official_args_simultaneous_ci(base_seed)),
        ("real data (pairwise + multiarm)", real_official_args(base_seed)),
        ("real data (pairwise)", real_official_args_pairwise(base_seed)),
        ("real data (multiarm)", real_official_args_multiarm(base_seed)),
        ("real data (simultaneous CI)", real_official_args_simultaneous_ci(base_seed)),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast sanity-check preset for --quick-test: runs every mode applicable
    to `data_source` (mode="all" -- see run()'s "all" handling) with cut-down
    sweeps/reps/tests for a quick pass that confirms the pipeline (incl.
    --latex output) still works. Restricts eval_types/tests/k_arms rather
    than sweeping the full catalog -- this is for pipeline confidence, not a
    representative result.
    ``data_source="real"`` (or 'openeval'/'inspect') runs pairwise + multiarm
    + simultaneous_ci only -- ppi has no real-data variant
    (build_judge_bias_sources is synthetic-only, see README's "known
    exceptions"). It also switches eval_types to 'binary': real-data
    pairwise/multiarm sources are binary-only by construction
    (corpus_pair_to_ci_pair_source / multiarm_corpus_to_source hardcode
    eval_type="binary" for both openeval and inspect), so the synthetic
    variant's 'continuous' filter would leave zero sources. --quick-test
    calls this twice per case (synthetic, then real) so the real-data paths
    don't go unexercised between --official-tests runs. factorial_check=True
    (with trivial factorial_reps/factorial_n_boot) so build_ppi_factorial_
    sources/fit_ppi_factorial_model/save_ppi_factorial_heatmap_plot stay
    exercised here too -- it's opt-in in run() (unlike power_check/
    comparison_check, which are opt-OUT and so already covered by this
    preset's defaults), so a regression there would otherwise go completely
    uncaught between --official-tests runs. factorial_alignment_mc=200 (well
    below the 20000 default) keeps the alignment-bucketed view's per-(eval_
    type, noise, bias) calibration draws cheap here too -- this preset only
    needs the code path exercised, not a precise alignment percentage."""
    eval_types = ["binary", "continuous"] if data_source == "synthetic" else ["binary"]
    return argparse.Namespace(
        mode="all", reps=3, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source=data_source, scenario_suite="standard", eval_types=eval_types, sizes=[10, 30, 50],
        runs=1, statistic="mean",
        bootstrap_n=200, icc_values=[0.20], cohens_d_values=[0.3],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=[TTEST.name, MWU.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name, MJ_FLOOR.name], ppi_n_boot=200, latex=True,
        effect_reps=5, effect_gold_mc=200, no_effect_check=False,
        factorial_check=True, factorial_reps=2, factorial_n_boot=50, factorial_alignment_mc=200,
        factorial_check_binary=True,
        workers=1,
    )


def run(args: argparse.Namespace) -> CaseResult:
    """Case entry point. Resolves `args.mode` ("all"/"pairwise_multiarm"/a
    single mode) to the applicable set of {pairwise, multiarm, ppi,
    simultaneous_ci} sweeps for `args.data_source`, builds sources, runs
    each sweep, prints its console report, writes CSV/LaTeX artifacts, and
    returns a CaseResult summarizing what ran and where the outputs went."""
    # Publication figures: drop the in-figure title and footnote strip that
    # the LaTeX caption already carries. Mutates the module global rather
    # than the environment because PPI_NO_FIGURE_TITLES is read at import,
    # long before args exist; the plot helpers read this global at call time.
    if getattr(args, "no_figure_titles", False):
        globals()["_LABEL_EFF_FIGURE_TITLES"] = False
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_paths: list[str] = []
        key_metrics: dict = {}
        # "all" means "every mode applicable to this data source" -- ppi has
        # no real-data variant (build_judge_bias_sources is synthetic-only),
        # so real/openeval/inspect data sources skip it rather than silently
        # re-running the synthetic PPI sweep under a real-data preset.
        # "pairwise_multiarm" is "all" minus ppi regardless of data source --
        # lets --official-tests offer synthetic ppi as its own (slow) menu
        # entry, separate from the faster pairwise+multiarm sweep.
        if args.mode == "pairwise_multiarm":
            modes = ["pairwise", "multiarm"]
        elif args.mode != "all":
            modes = [args.mode]
        elif args.data_source == "synthetic":
            modes = ["pairwise", "multiarm", "ppi", "simultaneous_ci"]
        else:
            modes = ["pairwise", "multiarm", "simultaneous_ci"]

        if "pairwise" in modes:
            print(f"\npvalues simulation (pairwise, non-PPI) -- data_source={args.data_source}, statistic={args.statistic}")
            if args.data_source == "synthetic":
                sources = build_pair_sources(
                    suite=args.scenario_suite, icc_values=args.icc_values,
                    cohens_d_values=args.cohens_d_values, include_null=True,
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
                    inspect_csv=args.inspect_csv, include_null=True,
                )
            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CIPairSources left after filtering.")
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            pw_results = run_pairwise_simulation(
                sources, sample_sizes=args.sizes, runs=args.runs, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                alpha=args.alpha, statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1),
            )
            print_pairwise_report(pw_results, alpha=args.alpha)

            run_stem = f"pvalues_pairwise_{args.data_source}_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_pairwise(results=pw_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_paths = save_pairwise_typeI_power_plot(results=pw_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_power.png"))
                for plot_path in plot_paths:
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                reliability_path = save_pairwise_reliability_violin_plot(
                    results=pw_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")

            null_rows = [r for r in pw_results if r.condition == "null"]
            type1 = float(np.mean([r.rejects / r.n_reps for r in null_rows])) if null_rows else float("nan")
            key_metrics["pairwise_n_results"] = len(pw_results)
            key_metrics["pairwise_mean_type1"] = type1

        if "multiarm" in modes or "simultaneous_ci" in modes:
            # Shared by both modes -- they sweep the identical k-arm source
            # list/grid, just measuring something different per rep (reject/
            # best-arm vs. CI coverage/width), so build it once even when
            # --mode all runs both (avoids a second, possibly network-bound,
            # real-data fetch).
            k_values = args.k_arms if isinstance(args.k_arms, list) else [args.k_arms]
            print(f"\npvalues simulation (multi-arm sources) -- data_source={args.data_source}, k={k_values}")
            if args.data_source == "synthetic":
                ma_sources = build_multiarm_sources(
                    suite=args.scenario_suite, icc=args.multiarm_icc, cohens_d=args.multiarm_cohens_d,
                    eval_types=args.eval_types,
                )
            else:
                runs = args.runs
                if runs != 1:
                    print("  Warning: real-data sources only support --runs 1 in this pass; forcing runs=1.")
                    runs = 1
                args = argparse.Namespace(**{**vars(args), "runs": runs})
                ma_sources = build_real_multiarm_sources(
                    args.data_source, benchmarks=args.benchmarks, models=args.models,
                    hf_token=args.hf_token, cache_dir=args.cache_dir, min_arm_size=args.min_pair_size,
                    inspect_csv=args.inspect_csv,
                )
                if args.eval_types:
                    requested = set(args.eval_types)
                    ma_sources = [s for s in ma_sources if s.eval_type in requested]
            if not ma_sources:
                raise ValueError("No MultiArmSources left after filtering.")
            print(f"  {len(ma_sources)} sources, sizes={args.sizes}, k_values={k_values}, reps={args.reps}, alpha={args.alpha}")

        if "multiarm" in modes:
            print(f"\npvalues simulation (multi-arm, non-PPI) -- method={args.multiarm_method}")
            ma_results = run_multiarm_simulation(
                ma_sources, sample_sizes=args.sizes, runs=args.runs, k_values=k_values, n_reps=args.reps,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, multiarm_method=args.multiarm_method,
                statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1), corrections=getattr(args, "corrections", None),
            )
            print_multiarm_report(ma_results, alpha=args.alpha)

            run_stem = f"pvalues_multiarm_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_multiarm(results=ma_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_multiarm_fwer_power_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_power.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
                vs_k_path = save_multiarm_fwer_vs_k_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_vs_k.png"))
                if Path(vs_k_path).exists():
                    output_paths.append(vs_k_path)
                    print(f"Saved plot: {vs_k_path}")
                vs_n_path = save_multiarm_fwer_vs_n_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_vs_n.png"))
                if Path(vs_n_path).exists():
                    output_paths.append(vs_n_path)
                    print(f"Saved plot: {vs_n_path}")
                # Compact 1x4 version -- this is the one the paper prints.
                panels_path = save_multiarm_fwer_panels_plot(
                    results=ma_results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_fwer_panels.png"),
                )
                if Path(panels_path).exists():
                    output_paths.append(panels_path)
                    print(f"Saved plot: {panels_path}")
                reliability_path = save_multiarm_reliability_violin_plot(
                    results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")
                violin_n_path = save_multiarm_violin_vs_n_plot(
                    results=ma_results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_violin_vs_n.png"),
                )
                output_paths.append(violin_n_path)
                print(f"Saved plot: {violin_n_path}")

            null_rows = [r for r in ma_results if r.condition == "null"]
            fwer = sum(r.any_reject for r in null_rows) / sum(r.n_reps for r in null_rows) if null_rows else float("nan")
            key_metrics["multiarm_n_results"] = len(ma_results)
            key_metrics["multiarm_mean_fwer"] = float(fwer)

        if "simultaneous_ci" in modes:
            print(f"\npvalues simulation (simultaneous CI, non-PPI) -- method={args.multiarm_method}")
            sci_results = run_simultaneous_ci_simulation(
                ma_sources, sample_sizes=args.sizes, runs=args.runs, k_values=k_values, n_reps=args.reps,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, multiarm_method=args.multiarm_method,
                statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1), ci_methods=getattr(args, "ci_methods", None),
            )
            print_simultaneous_ci_report(sci_results, alpha=args.alpha)

            run_stem = f"pvalues_simultaneous_ci_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_simultaneous_ci(
                    results=sci_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                    latex=getattr(args, "latex", False),
                )
            if args.plots == "save":
                plot_path = save_simultaneous_ci_coverage_width_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_coverage_width.png"),
                )
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
                vs_k_path = save_simultaneous_ci_coverage_width_vs_k_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_vs_k.png"),
                )
                if Path(vs_k_path).exists():
                    output_paths.append(vs_k_path)
                    print(f"Saved plot: {vs_k_path}")
                vs_n_path = save_simultaneous_ci_coverage_width_vs_n_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_vs_n.png"),
                )
                if Path(vs_n_path).exists():
                    output_paths.append(vs_n_path)
                    print(f"Saved plot: {vs_n_path}")
                # Compact 1x4 version -- this is the one the paper prints.
                sc_panels_path = save_simultaneous_ci_panels_plot(
                    results=sci_results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_ci_panels.png"),
                )
                if Path(sc_panels_path).exists():
                    output_paths.append(sc_panels_path)
                    print(f"Saved plot: {sc_panels_path}")
                reliability_path = save_simultaneous_ci_reliability_violin_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")
                violin_vs_n_path = save_simultaneous_ci_violin_vs_n_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_violin_vs_n.png"),
                )
                output_paths.append(violin_vs_n_path)
                print(f"Saved plot: {violin_vs_n_path}")

            for cm_name in ("none", "bonferroni", "max_t", CORR_SIDAK.name, CORR_BOOT.name, CORR_BOOT_CAL.name):
                cm_null_rows = [r for r in sci_results if r.ci_method == cm_name and r.condition == "null"]
                if not cm_null_rows:
                    continue
                cov = sum(r.all_covered for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                width = sum(r.total_width for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                score = sum(r.total_score for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                key_metrics[f"simultaneous_ci_{cm_name}_coverage"] = float(cov)
                key_metrics[f"simultaneous_ci_{cm_name}_avg_width"] = float(width)
                key_metrics[f"simultaneous_ci_{cm_name}_avg_score"] = float(score)
            key_metrics["simultaneous_ci_n_results"] = len(sci_results)

        if "ppi" in modes:
            # The calibration sweep below is 115 scenarios and dominates the
            # runtime of this mode; the rho-drift check is a separate phase
            # appended after it. --rho-drift-only skips straight to the drift
            # phase, which is what you want when that is all you came for.
            if not getattr(args, "rho_drift_only", False):
                # Default (no --tests) runs the OFFICIAL subset -- excludes
                # kruskal_mnar_experimental (its local rectifier costs real MCAR
                # calibration; see methods.py) but it stays selectable
                # explicitly via --tests for comparison.
                active_tests = args.tests if args.tests else [m.name for m in PPI_OFFICIAL_TEST_METHODS]
                print(f"\npvalues simulation (PPI-corrected) -- tests={active_tests}")
                jb_sources = build_judge_bias_sources() + build_judge_bias_sources_binary()
                if args.eval_types:
                    requested = set(args.eval_types)
                    jb_sources = [s for s in jb_sources if s.eval_type in requested]
                if not jb_sources:
                    raise ValueError("No JudgeBiasSources left after filtering.")

                # MNAR (label_mnar=True -- the "label.*mnar-*"/"label.binary.
                # mnar-*" scenarios and their bias-magnitude companions) is kept
                # out of the headline results: this project assumes an MCAR
                # labeling regime, and MNAR is a known-adversarial condition for
                # PPI's rectifier (label selection depends on the outcome itself,
                # violating the missing-completely-at-random assumption the
                # simple rectifier relies on -- label.*.mnar-strong drives
                # Tango/Wilson's worst bias_z on binary data while
                # continuous/likert/grades stay well-calibrated under the same
                # mechanism). Reported separately, as an explicit limitation,
                # rather than pooled into the headline MCAR numbers.
                mnar_names = {s.name for s in jb_sources if s.label_mnar}

                # {scenario_name: eval-type scale span} -- turns save_ppi_effect_plot's
                # CI Width panel from raw score units (where grades' 0-100 scale
                # dwarfs continuous/binary's 0-1 and likert's 1-5, purely from
                # units, not calibration -- see that function's width_norm
                # docstring) into "fraction of eval-type scale," comparable
                # across eval types.
                width_norm = {
                    s.name: (EVAL_TYPE_SCALE_BOUNDS[s.eval_type][1] - EVAL_TYPE_SCALE_BOUNDS[s.eval_type][0])
                    for s in jb_sources
                }

                if not getattr(args, "no_typeI_check", False):
                    print(f"  {len(jb_sources)} scenarios, reps={args.reps}, n_boot={args.ppi_n_boot}, alpha={args.alpha}")

                    ppi_results = run_ppi_simulation(
                        jb_sources, active_tests=active_tests, n_reps=args.reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed, n_workers=getattr(args, "workers", 1),
                    )
                    ppi_results_mcar = [r for r in ppi_results if r.name not in mnar_names]
                    ppi_results_mnar = [r for r in ppi_results if r.name in mnar_names]
                    print_ppi_report(ppi_results_mcar, alpha=args.alpha, regime="MCAR")
                    if ppi_results_mnar:
                        print_ppi_report(ppi_results_mnar, alpha=args.alpha, regime="MNAR -- adversarial to PPI, reported as a known limitation, not part of the primary MCAR results")

                    run_stem = f"pvalues_ppi_reps{args.reps}_{stamp}"
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi(results=ppi_results_mcar, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False), regime="MCAR")
                        if ppi_results_mnar:
                            output_paths += save_results_artifacts_ppi(results=ppi_results_mnar, alpha=args.alpha, out_dir=args.out_dir, run_stem=f"{run_stem}_mnar", latex=getattr(args, "latex", False), regime="MNAR")
                    if args.plots == "save":
                        plot_path = save_ppi_typeI_plot(results=ppi_results_mcar, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"), regime="MCAR")
                        output_paths.append(plot_path)
                        print(f"Saved plot: {plot_path}")
                        if any(r.test in _PPI_NONSTANDARD_TESTS for r in ppi_results_mcar):
                            nonstd_plot_path = save_ppi_typeI_plot(
                                results=ppi_results_mcar, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard.png"),
                                nonstandard=True, regime="MCAR",
                            )
                            output_paths.append(nonstd_plot_path)
                            print(f"Saved plot: {nonstd_plot_path}")
                        if ppi_results_mnar:
                            mnar_plot_path = save_ppi_typeI_plot(
                                results=ppi_results_mnar, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_mnar.png"),
                                regime="MNAR",
                            )
                            output_paths.append(mnar_plot_path)
                            print(f"Saved plot: {mnar_plot_path}")
                            if any(r.test in _PPI_NONSTANDARD_TESTS for r in ppi_results_mnar):
                                nonstd_mnar_plot_path = save_ppi_typeI_plot(
                                    results=ppi_results_mnar, alpha=args.alpha,
                                    out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard_mnar.png"),
                                    nonstandard=True, regime="MNAR",
                                )
                                output_paths.append(nonstd_mnar_plot_path)
                                print(f"Saved plot: {nonstd_mnar_plot_path}")

                    # Headline key_metrics reflect the MCAR (primary) regime only.
                    c_tot = sum(r.corrected_rejects for r in ppi_results_mcar)
                    u_tot = sum(r.uncorrected_rejects for r in ppi_results_mcar)
                    n_tot = sum(r.n_reps for r in ppi_results_mcar)
                    key_metrics["ppi_n_results"] = len(ppi_results_mcar)
                    key_metrics["ppi_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
                    key_metrics["ppi_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

                if not getattr(args, "no_effect_check", False):
                    effect_reps = getattr(args, "effect_reps", 200)
                    effect_gold_mc = getattr(args, "effect_gold_mc", 3000)
                    print(f"\npvalues simulation (PPI-corrected, effect-size check) -- effect_reps={effect_reps}, gold_mc={effect_gold_mc}")
                    effect_results = run_ppi_effect_check(
                        jb_sources, active_tests=active_tests, n_reps=effect_reps, n_boot=args.ppi_n_boot,
                        gold_null_mc=effect_gold_mc, progress_mode=args.progress, seed=args.seed + 1,
                        n_workers=getattr(args, "workers", 1),
                    )
                    effect_results_mcar = [r for r in effect_results if r.name not in mnar_names]
                    effect_results_mnar = [r for r in effect_results if r.name in mnar_names]
                    print_ppi_effect_report(effect_results_mcar, alpha=args.alpha, regime="MCAR")
                    if effect_results_mnar:
                        print_ppi_effect_report(effect_results_mnar, alpha=args.alpha, regime="MNAR -- adversarial to PPI, reported as a known limitation, not part of the primary MCAR results")

                    effect_stem = f"pvalues_ppi_effect_reps{effect_reps}_{stamp}"
                    if effect_results_mcar:
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_effect(
                                results=effect_results_mcar, alpha=args.alpha, out_dir=args.out_dir, run_stem=effect_stem,
                                latex=getattr(args, "latex", False), regime="MCAR",
                            )
                            if effect_results_mnar:
                                output_paths += save_results_artifacts_ppi_effect(
                                    results=effect_results_mnar, alpha=args.alpha, out_dir=args.out_dir, run_stem=f"{effect_stem}_mnar",
                                    latex=getattr(args, "latex", False), regime="MNAR",
                                )
                        if args.plots == "save":
                            effect_plot_path = save_ppi_effect_plot(
                                results=effect_results_mcar, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width.png"),
                                regime="MCAR", width_norm=width_norm,
                            )
                            output_paths.append(effect_plot_path)
                            print(f"Saved plot: {effect_plot_path}")
                            if any(r.test in _PPI_CI_COMPARISON_TESTS for r in effect_results_mcar):
                                ci_comparison_plot_path = save_ppi_effect_plot(
                                    results=effect_results_mcar, alpha=args.alpha,
                                    out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width_ci_comparison.png"),
                                    ci_comparison=True, regime="MCAR", width_norm=width_norm,
                                )
                                output_paths.append(ci_comparison_plot_path)
                                print(f"Saved plot: {ci_comparison_plot_path}")
                            if effect_results_mnar:
                                mnar_effect_plot_path = save_ppi_effect_plot(
                                    results=effect_results_mnar, alpha=args.alpha,
                                    out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width_mnar.png"),
                                    regime="MNAR", width_norm=width_norm,
                                )
                                output_paths.append(mnar_effect_plot_path)
                                print(f"Saved plot: {mnar_effect_plot_path}")
                                if any(r.test in _PPI_CI_COMPARISON_TESTS for r in effect_results_mnar):
                                    ci_comparison_mnar_plot_path = save_ppi_effect_plot(
                                        results=effect_results_mnar, alpha=args.alpha,
                                        out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width_ci_comparison_mnar.png"),
                                        ci_comparison=True, regime="MNAR", width_norm=width_norm,
                                    )
                                    output_paths.append(ci_comparison_mnar_plot_path)
                                    print(f"Saved plot: {ci_comparison_mnar_plot_path}")

                        # Headline key_metrics reflect the MCAR (primary) regime only.
                        key_metrics["ppi_effect_n_results"] = len(effect_results_mcar)
                        finite_z = [r.bias_z for r in effect_results_mcar if np.isfinite(r.bias_z)]
                        key_metrics["ppi_effect_mean_abs_bias_z"] = float(np.mean(np.abs(finite_z))) if finite_z else float("nan")
                        finite_cov = [r.coverage for r in effect_results_mcar if np.isfinite(r.coverage)]
                        key_metrics["ppi_effect_mean_coverage"] = float(np.mean(finite_cov)) if finite_cov else float("nan")

                power_sources = build_ppi_power_sources()
                power_sources_binary = build_ppi_power_sources_binary()
                if args.eval_types:
                    requested = set(args.eval_types)
                    power_sources = [s for s in power_sources if s.eval_type in requested]
                    power_sources_binary = [s for s in power_sources_binary if s.eval_type in requested]

                if not getattr(args, "no_power_check", False) and (power_sources or power_sources_binary):
                    power_reps = getattr(args, "effect_reps", 200)
                    power_all_sources = power_sources + power_sources_binary
                    print(f"\npvalues simulation (PPI-corrected, power check) -- {len(power_all_sources)} scenarios, "
                          f"reps={power_reps}, n_boot={args.ppi_n_boot}")
                    power_results = run_ppi_simulation(
                        power_all_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 2, n_workers=getattr(args, "workers", 1),
                    )
                    print_ppi_power_report(power_results, alpha=args.alpha)

                    # No-bias baseline computed BEFORE the main power plot (not
                    # after, as originally ordered) so its corrected rate can be
                    # overlaid there as an "ideal" reference line -- does PPI
                    # correction cost power for nothing when there's no judge
                    # bias to correct for, and how close does the biased-
                    # condition line above track that ceiling? See
                    # build_ppi_power_nobias_sources' docstring.
                    nobias_sources = build_ppi_power_nobias_sources() + build_ppi_power_nobias_sources_binary()
                    if args.eval_types:
                        nobias_sources = [s for s in nobias_sources if s.eval_type in requested]
                    nobias_results: list[PPIResult] = []
                    if nobias_sources:
                        print(f"\npvalues simulation (PPI-corrected, power check -- no bias) -- "
                              f"{len(nobias_sources)} scenarios")
                        nobias_results = run_ppi_simulation(
                            nobias_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 5, n_workers=getattr(args, "workers", 1),
                        )
                        print_ppi_power_report(
                            nobias_results, alpha=args.alpha, header="POWER, NO JUDGE BIAS (bias_type=none)",
                        )
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_power(
                                results=nobias_results, alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=f"pvalues_ppi_power_nobias_reps{power_reps}_{stamp}",
                            )
                        key_metrics["ppi_power_nobias_n_results"] = len(nobias_results)

                    power_stem = f"pvalues_ppi_power_reps{power_reps}_{stamp}"
                    if power_results:
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_power(
                                results=power_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=power_stem,
                            )
                        if args.plots == "save":
                            power_plot_path = save_ppi_power_plot(
                                results=power_results, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{power_stem}_power_vs_effect_size.png"),
                            )
                            output_paths.append(power_plot_path)
                            print(f"Saved plot: {power_plot_path}")
                            if nobias_results:
                                nobias_plot_path = save_ppi_power_plot(
                                    results=nobias_results, alpha=args.alpha,
                                    out_path=str(Path(plots_dir) / f"{power_stem}_power_vs_effect_size_nobias.png"),
                                    title_suffix=" -- No Judge Bias",
                                )
                                output_paths.append(nobias_plot_path)
                                print(f"Saved plot: {nobias_plot_path}")

                        key_metrics["ppi_power_n_results"] = len(power_results)
                        top_es = max({_parse_ppi_power_name(r.name)[1] for r in power_results}, default=0.0)
                        top_rows = [r for r in power_results if _parse_ppi_power_name(r.name)[1] == top_es]
                        c_tot = sum(r.corrected_rejects for r in top_rows)
                        n_tot = sum(r.n_reps for r in top_rows)
                        key_metrics["ppi_power_mean_corrected_at_max_es"] = float(c_tot / n_tot) if n_tot else float("nan")

                    # Bias-direction check: does the "cancellation dip" (opposing
                    # bias vs. effect, already run above as power_results) look
                    # different from the reinforcing-bias case, where an
                    # uncorrected test would just quietly overstate the effect
                    # instead of showing a visible anomaly? See
                    # build_ppi_power_reinforcing_sources' docstring.
                    reinforcing_sources = build_ppi_power_reinforcing_sources() + build_ppi_power_reinforcing_sources_binary()
                    if args.eval_types:
                        reinforcing_sources = [s for s in reinforcing_sources if s.eval_type in requested]
                    reinforcing_results: list[PPIResult] = []
                    if reinforcing_sources:
                        print(f"\npvalues simulation (PPI-corrected, power check -- bias reinforcing effect) -- "
                              f"{len(reinforcing_sources)} scenarios")
                        reinforcing_results = run_ppi_simulation(
                            reinforcing_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 4, n_workers=getattr(args, "workers", 1),
                        )
                        print_ppi_power_report(
                            reinforcing_results, alpha=args.alpha,
                            header="POWER UNDER JUDGE BIAS (reinforcing the real effect)",
                        )
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_power(
                                results=reinforcing_results, alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=f"pvalues_ppi_power_reinforcing_reps{power_reps}_{stamp}",
                            )
                        if args.plots == "save" and power_results:
                            direction_plot_path = save_ppi_power_direction_plot(
                                opposing=power_results, reinforcing=reinforcing_results, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{power_stem}_power_direction.png"),
                            )
                            output_paths.append(direction_plot_path)
                            print(f"Saved plot: {direction_plot_path}")
                        key_metrics["ppi_power_reinforcing_n_results"] = len(reinforcing_results)

                if getattr(args, "power_nlab_grid_check", False):
                    # Both directions ALWAYS run together (never just reinforcing
                    # alone) -- matching how the base power check always runs
                    # build_ppi_power_sources (opposing) + build_ppi_power_
                    # reinforcing_sources together under one flag. The whole
                    # point of this grid is to test whether the anomaly is
                    # specific to the reinforcing direction; running only one
                    # direction can't answer that.
                    nlab_grid_variants = [
                        ("reinforcing", build_ppi_power_nlab_grid_reinforcing_sources(), args.seed + 15),
                        ("opposing", build_ppi_power_nlab_grid_opposing_sources(), args.seed + 16),
                    ]
                    if args.eval_types:
                        requested = set(args.eval_types)
                        nlab_grid_variants = [
                            (label, [s for s in srcs if s.eval_type in requested], seed)
                            for label, srcs, seed in nlab_grid_variants
                        ]
                    nlab_grid_reps = getattr(args, "effect_reps", 200)
                    nlab_grid_results_by_direction: dict[str, list[PPIResult]] = {}
                    for direction_label, nlab_grid_sources, direction_seed in nlab_grid_variants:
                        if not nlab_grid_sources:
                            continue
                        print(f"\npvalues simulation (PPI-corrected, power vs. label/dataset-size grid, "
                              f"bias {direction_label}) -- {len(nlab_grid_sources)} scenarios, "
                              f"reps={nlab_grid_reps}, n_boot={args.ppi_n_boot}")
                        nlab_grid_results = run_ppi_simulation(
                            nlab_grid_sources, active_tests=active_tests, n_reps=nlab_grid_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=direction_seed, n_workers=getattr(args, "workers", 1),
                        )
                        nlab_grid_results_by_direction[direction_label] = nlab_grid_results
                        print_ppi_power_nlab_grid_report(
                            nlab_grid_results, alpha=args.alpha, header=f"bias {direction_label} effect",
                        )
                        if not nlab_grid_results:
                            continue
                        direction_suffix = "_rf" if direction_label == "reinforcing" else "_op"
                        nlab_grid_stem = f"pvalues_ppi_power_nlab_grid{direction_suffix}_reps{nlab_grid_reps}_{stamp}"
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_power_nlab_grid(
                                results=nlab_grid_results, alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=nlab_grid_stem, header=f"bias {direction_label} effect",
                            )
                        if args.plots == "save":
                            nlab_grid_plot_paths = save_ppi_power_nlab_grid_plots(
                                results=nlab_grid_results, alpha=args.alpha, out_dir=plots_dir, stem=nlab_grid_stem,
                            )
                            output_paths += nlab_grid_plot_paths
                            for p in nlab_grid_plot_paths:
                                print(f"Saved plot: {p}")
                        key_metrics[f"ppi_power_nlab_grid_{direction_label}_n_results"] = len(nlab_grid_results)

                    reinforcing_grid_results = nlab_grid_results_by_direction.get("reinforcing", [])
                    opposing_grid_results = nlab_grid_results_by_direction.get("opposing", [])
                    if args.plots == "save" and reinforcing_grid_results and opposing_grid_results:
                        direction_plot_path = save_ppi_power_nlab_grid_direction_plot(
                            opposing=opposing_grid_results, reinforcing=reinforcing_grid_results, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"pvalues_ppi_power_nlab_grid_reps{nlab_grid_reps}_{stamp}_direction.png"),
                        )
                        output_paths.append(direction_plot_path)
                        print(f"Saved plot: {direction_plot_path}")

                comparison_results_pooled: list[PPIComparisonResult] = []
                comparison_results_omnibus_pooled: list[PPIComparisonResult] = []
                nlab_cal_pooled: list[PPIComparisonResult] = []
                nlab_pow_pooled: list[PPIComparisonResult] = []
                comparison_results_binary_pooled: list[PPIComparisonResult] = []
                nlab_cal_pooled_binary: list[PPIComparisonResult] = []
                nlab_pow_pooled_binary: list[PPIComparisonResult] = []
                if not getattr(args, "no_comparison_check", False):
                    comparison_sources = power_sources + build_ppi_comparison_label_frac_sources()
                    if args.eval_types:
                        requested = set(args.eval_types)
                        comparison_sources = [s for s in comparison_sources if s.eval_type in requested]
                    if comparison_sources:
                        comparison_reps = getattr(args, "effect_reps", 200)
                        print(f"\npvalues simulation (PPI-corrected, estimator comparison) -- "
                              f"{len(comparison_sources)} scenarios x {len(_COMPARISON_METHODS)} methods "
                              f"({_COMPARISON_METHODS_LABEL}), reps={comparison_reps}, n_boot={args.ppi_n_boot}")
                        comparison_results_raw = run_ppi_comparison_simulation(
                            comparison_sources, n_reps=comparison_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 3, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS,
                        )
                        comparison_results_pooled = pool_ppi_comparison_across_methods(comparison_results_raw)
                        print_ppi_comparison_report(comparison_results_pooled, alpha=args.alpha)

                        comparison_stem = f"pvalues_ppi_comparison_reps{comparison_reps}_{stamp}"
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_comparison(
                                results=comparison_results_raw, pooled_results=comparison_results_pooled,
                                alpha=args.alpha, out_dir=args.out_dir, run_stem=comparison_stem,
                            )
                        # Plot saved later (after the N x N_lab grid and binary
                        # comparison blocks below finish), once nlab_pow_pooled/
                        # comparison_results_binary_pooled are available too --
                        # see the "Flagship 5-way comparison plot" block after
                        # the binary comparison check.

                        key_metrics["ppi_comparison_n_results"] = len(comparison_results_pooled)
                        max_es_rows = [r for r in comparison_results_pooled if r.tag == "power" and r.effect_size == max((r.effect_size for r in comparison_results_pooled if r.tag == "power"), default=0.0)]
                        if max_es_rows:
                            key_metrics["ppi_comparison_power_all_human_at_max_es"] = float(
                                sum(r.rejects_all_human for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                            )
                            key_metrics["ppi_comparison_power_human_subset_at_max_es"] = float(
                                sum(r.rejects_human_subset for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                            )
                            key_metrics["ppi_comparison_power_ppi_at_max_es"] = float(
                                sum(r.rejects_ppi for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                            )

                        # Reader-facing sanity check (opt-in via --comparison-omnibus,
                        # on by default in official_args_ppi -- see its docstring):
                        # does the SAME estimator-comparison story (all_human > ppi >
                        # human_subset > llm_only/llm_impute) hold if the omnibus
                        # tests are pooled instead of _COMPARISON_METHODS? Reuses the
                        # SAME comparison_sources grid computed above (NOT the
                        # factorial sweep, which is a separate ~6798-scenario grid --
                        # see --factorial-omnibus for that one). comparison_sources
                        # is only ~60 scenarios, so this is cheap even at full
                        # reps/n_boot precision -- no screening-tier default needed
                        # the way --factorial-check has one.
                        if getattr(args, "comparison_omnibus", False):
                            comp_omni_methods = tuple(getattr(args, "comparison_omnibus_tests", None)
                                                      or _COMPARISON_METHODS_OMNIBUS)
                            comp_omni_label = "/".join(comp_omni_methods)
                            print(f"\npvalues simulation (PPI-corrected, estimator comparison, omnibus) -- "
                                  f"{len(comparison_sources)} scenarios x {len(comp_omni_methods)} methods "
                                  f"({comp_omni_label}), reps={comparison_reps}, n_boot={args.ppi_n_boot}")
                            comparison_results_omnibus_raw = run_ppi_comparison_simulation(
                                comparison_sources, n_reps=comparison_reps, n_boot=args.ppi_n_boot,
                                progress_mode=args.progress, seed=args.seed + 19, n_workers=getattr(args, "workers", 1),
                                methods=comp_omni_methods,
                            )
                            comparison_results_omnibus_pooled = pool_ppi_comparison_across_methods(comparison_results_omnibus_raw)
                            print_ppi_comparison_report(
                                comparison_results_omnibus_pooled, alpha=args.alpha, label=comp_omni_label,
                            )
                            comparison_omnibus_stem = f"pvalues_ppi_comparison_omnibus_reps{comparison_reps}_{stamp}"
                            if args.save_results == "save":
                                output_paths += save_results_artifacts_ppi_comparison(
                                    results=comparison_results_omnibus_raw, pooled_results=comparison_results_omnibus_pooled,
                                    alpha=args.alpha, out_dir=args.out_dir, run_stem=comparison_omnibus_stem,
                                    label=comp_omni_label,
                                )
                            key_metrics["ppi_comparison_omnibus_n_results"] = len(comparison_results_omnibus_pooled)

                    # N x N_lab grid: does calibration/power depend on the RATIO
                    # N_lab/N or the ABSOLUTE N_lab count? build_ppi_nlab_grid_sources
                    # covers continuous and likert (see its docstring); filter
                    # per-source by eval_type against --eval-types rather than an
                    # all-or-nothing check, so e.g. --eval-types likert alone
                    # still produces likert cells.
                    nlab_cal_sources = build_ppi_nlab_grid_sources(effect_frac=0.0)
                    nlab_pow_sources = build_ppi_nlab_grid_sources(effect_frac=PPI_COMPARISON_MODERATE_EFFECT_FRAC)
                    if args.eval_types:
                        requested = set(args.eval_types)
                        nlab_cal_sources = [s for s in nlab_cal_sources if s.eval_type in requested]
                        nlab_pow_sources = [s for s in nlab_pow_sources if s.eval_type in requested]
                    if nlab_cal_sources or nlab_pow_sources:
                        nlab_reps = getattr(args, "effect_reps", 200)
                        print(f"\npvalues simulation (PPI-corrected, N x N_lab grid) -- "
                              f"{len(nlab_cal_sources)} calibration + {len(nlab_pow_sources)} power scenarios "
                              f"x {len(_COMPARISON_METHODS)} methods, reps={nlab_reps}, n_boot={args.ppi_n_boot}")
                        nlab_cal_raw = run_ppi_comparison_simulation(
                            nlab_cal_sources, n_reps=nlab_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 6, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS,
                        ) if nlab_cal_sources else []
                        nlab_pow_raw = run_ppi_comparison_simulation(
                            nlab_pow_sources, n_reps=nlab_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 7, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS,
                        ) if nlab_pow_sources else []
                        nlab_cal_pooled = pool_ppi_comparison_across_methods(nlab_cal_raw) if nlab_cal_raw else []
                        nlab_pow_pooled = pool_ppi_comparison_across_methods(nlab_pow_raw) if nlab_pow_raw else []
                        print_ppi_nlab_grid_report(
                            nlab_cal_pooled, alpha=args.alpha, header="N x N_LAB GRID (calibration, effect_size=0)",
                        )
                        print_ppi_nlab_grid_report(
                            nlab_pow_pooled, alpha=args.alpha, header="N x N_LAB GRID (power, moderate effect_size)",
                        )

                        nlab_stem = f"pvalues_ppi_nlab_grid_reps{nlab_reps}_{stamp}"
                        if args.save_results == "save":
                            if nlab_cal_raw:
                                output_paths += save_results_artifacts_ppi_nlab_grid(
                                    results=nlab_cal_raw, pooled_results=nlab_cal_pooled,
                                    alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{nlab_stem}_calibration", header="N x N_LAB GRID (calibration, effect_size=0)",
                                )
                            if nlab_pow_raw:
                                output_paths += save_results_artifacts_ppi_nlab_grid(
                                    results=nlab_pow_raw, pooled_results=nlab_pow_pooled,
                                    alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{nlab_stem}_power", header="N x N_LAB GRID (power, moderate effect_size)",
                                )
                        if args.plots == "save":
                            nlab_plot_path = save_ppi_nlab_grid_plot(
                                calibration_results=nlab_cal_pooled or None, power_results=nlab_pow_pooled or None,
                                alpha=args.alpha, out_path=str(Path(plots_dir) / f"{nlab_stem}_heatmap.png"),
                            )
                            output_paths.append(nlab_plot_path)
                            print(f"Saved plot: {nlab_plot_path}")

                        key_metrics["ppi_nlab_grid_n_calibration_results"] = len(nlab_cal_pooled)
                        key_metrics["ppi_nlab_grid_n_power_results"] = len(nlab_pow_pooled)

                    # Null-effect 5-way bar chart and the flagship 5-way comparison
                    # plot are both saved later (after the binary comparison block
                    # below), so binary's leftmost panel can be included -- see
                    # those two save_ppi_*_plot calls after the binary block.

                if not getattr(args, "no_comparison_check", False):
                    # Binary's estimator-comparison sweep, kept entirely separate
                    # from comparison_sources/_COMPARISON_METHODS above: only 2 of
                    # that pool's 4 tests are valid on binary data (see
                    # _COMPARISON_METHODS_BINARY), so pooling would be apples-to-
                    # oranges. build_ppi_nlab_grid_sources_binary exists and is
                    # unit-tested but deliberately NOT wired in here yet -- its
                    # (N, N_lab) grid needs its own 2D heatmap-style report the
                    # way save_ppi_nlab_grid_plot gives the non-binary version,
                    # which print_ppi_comparison_report's single-x-axis table
                    # can't show correctly (a real follow-up, not an oversight).
                    comparison_sources_binary = power_sources_binary + build_ppi_comparison_label_frac_sources_binary()
                    if args.eval_types:
                        comparison_sources_binary = [s for s in comparison_sources_binary if s.eval_type in requested]
                    if comparison_sources_binary:
                        comparison_reps = getattr(args, "effect_reps", 200)
                        print(f"\npvalues simulation (PPI-corrected, binary estimator comparison) -- "
                              f"{len(comparison_sources_binary)} scenarios x {len(_COMPARISON_METHODS_BINARY)} methods "
                              f"({_COMPARISON_METHODS_BINARY_LABEL}), reps={comparison_reps}, n_boot={args.ppi_n_boot}")
                        comparison_binary_tags = [
                            ("power_binary", "effect_size", "es", "{:.2f}"),
                            ("complab_binary", "n_lab", "nlab", "{:d}"),
                        ]
                        comparison_results_binary_raw = run_ppi_comparison_simulation(
                            comparison_sources_binary, n_reps=comparison_reps, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 11, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS_BINARY,
                        )
                        comparison_results_binary_pooled = pool_ppi_comparison_across_methods(comparison_results_binary_raw)
                        print_ppi_comparison_report(
                            comparison_results_binary_pooled, alpha=args.alpha,
                            tags=comparison_binary_tags, label=_COMPARISON_METHODS_BINARY_LABEL,
                        )

                        comparison_binary_stem = f"pvalues_ppi_comparison_binary_reps{comparison_reps}_{stamp}"
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_comparison(
                                results=comparison_results_binary_raw, pooled_results=comparison_results_binary_pooled,
                                alpha=args.alpha, out_dir=args.out_dir, run_stem=comparison_binary_stem,
                                tags=comparison_binary_tags, label=_COMPARISON_METHODS_BINARY_LABEL,
                            )
                        key_metrics["ppi_comparison_binary_n_results"] = len(comparison_results_binary_pooled)

                    # Binary's N x N_lab grid -- the binary analogue of the
                    # continuous/likert nlab_grid block above (build_ppi_nlab_
                    # grid_sources_binary), previously computed nowhere: binary's
                    # comparison figures fell back to the single (N=100, N_lab=20)
                    # scenario while continuous/likert already got the full grid.
                    nlab_cal_sources_binary = build_ppi_nlab_grid_sources_binary(effect_frac=0.0)
                    nlab_pow_sources_binary = build_ppi_nlab_grid_sources_binary(effect_frac=PPI_COMPARISON_MODERATE_EFFECT_FRAC)
                    if args.eval_types:
                        nlab_cal_sources_binary = [s for s in nlab_cal_sources_binary if s.eval_type in requested]
                        nlab_pow_sources_binary = [s for s in nlab_pow_sources_binary if s.eval_type in requested]
                    if nlab_cal_sources_binary or nlab_pow_sources_binary:
                        nlab_reps_binary = getattr(args, "effect_reps", 200)
                        print(f"\npvalues simulation (PPI-corrected, N x N_lab grid, binary) -- "
                              f"{len(nlab_cal_sources_binary)} calibration + {len(nlab_pow_sources_binary)} power scenarios "
                              f"x {len(_COMPARISON_METHODS_BINARY)} methods, reps={nlab_reps_binary}, n_boot={args.ppi_n_boot}")
                        nlab_cal_raw_binary = run_ppi_comparison_simulation(
                            nlab_cal_sources_binary, n_reps=nlab_reps_binary, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 17, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS_BINARY,
                        ) if nlab_cal_sources_binary else []
                        nlab_pow_raw_binary = run_ppi_comparison_simulation(
                            nlab_pow_sources_binary, n_reps=nlab_reps_binary, n_boot=args.ppi_n_boot,
                            progress_mode=args.progress, seed=args.seed + 18, n_workers=getattr(args, "workers", 1),
                            methods=_COMPARISON_METHODS_BINARY,
                        ) if nlab_pow_sources_binary else []
                        nlab_cal_pooled_binary = pool_ppi_comparison_across_methods(nlab_cal_raw_binary) if nlab_cal_raw_binary else []
                        nlab_pow_pooled_binary = pool_ppi_comparison_across_methods(nlab_pow_raw_binary) if nlab_pow_raw_binary else []
                        print_ppi_nlab_grid_report(
                            nlab_cal_pooled_binary, alpha=args.alpha, header="N x N_LAB GRID (calibration, effect_size=0, binary)",
                        )
                        print_ppi_nlab_grid_report(
                            nlab_pow_pooled_binary, alpha=args.alpha, header="N x N_LAB GRID (power, moderate effect_size, binary)",
                        )
                        nlab_stem_binary = f"pvalues_ppi_nlab_grid_binary_reps{nlab_reps_binary}_{stamp}"
                        if args.save_results == "save":
                            if nlab_cal_raw_binary:
                                output_paths += save_results_artifacts_ppi_nlab_grid(
                                    results=nlab_cal_raw_binary, pooled_results=nlab_cal_pooled_binary,
                                    alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{nlab_stem_binary}_calibration", header="N x N_LAB GRID (calibration, effect_size=0, binary)",
                                )
                            if nlab_pow_raw_binary:
                                output_paths += save_results_artifacts_ppi_nlab_grid(
                                    results=nlab_pow_raw_binary, pooled_results=nlab_pow_pooled_binary,
                                    alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{nlab_stem_binary}_power", header="N x N_LAB GRID (power, moderate effect_size, binary)",
                                )
                        key_metrics["ppi_nlab_grid_binary_n_calibration_results"] = len(nlab_cal_pooled_binary)
                        key_metrics["ppi_nlab_grid_binary_n_power_results"] = len(nlab_pow_pooled_binary)

                    # Both comparison plots, saved here (not right after each
                    # sweep above) so binary's leftmost panel -- computed just
                    # above -- can be included. Binary was previously silently
                    # absent from both figures entirely (computed and reported
                    # in text/CSV, never plotted), which reads to a reviewer as
                    # binary having been skipped rather than shown elsewhere.
                    if args.plots == "save" and comparison_results_pooled:
                        comparison_reps_for_stem = getattr(args, "effect_reps", 200)
                        comparison_plot_path = save_ppi_comparison_plot(
                            results=comparison_results_pooled, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"pvalues_ppi_comparison_reps{comparison_reps_for_stem}_{stamp}_five_way_comparison.png"),
                            results_binary=comparison_results_binary_pooled or None,
                            nlab_pow_results=nlab_pow_pooled or None,
                            nlab_pow_results_binary=nlab_pow_pooled_binary or None,
                        )
                        output_paths.append(comparison_plot_path)
                        print(f"Saved plot: {comparison_plot_path}")

                        null_comparison_plot_path = save_ppi_null_comparison_plot(
                            results=comparison_results_pooled, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"pvalues_ppi_comparison_reps{comparison_reps_for_stem}_{stamp}_null_false_positive_rate.png"),
                            nlab_cal_results=nlab_cal_pooled or None,
                            results_binary=comparison_results_binary_pooled or None,
                            nlab_cal_results_binary=nlab_cal_pooled_binary or None,
                        )
                        output_paths.append(null_comparison_plot_path)
                        print(f"Saved plot: {null_comparison_plot_path}")

                        if comparison_results_omnibus_pooled:
                            # No results_binary/nlab_pow_results_binary equivalent --
                            # _COMPARISON_METHODS_OMNIBUS is never run against binary
                            # data anywhere in this harness (binary's own comparison
                            # sweep uses the unrelated 2-method
                            # _COMPARISON_METHODS_BINARY), so there's no omnibus-on-
                            # binary column to plot.
                            comparison_omnibus_plot_path = save_ppi_comparison_plot(
                                results=comparison_results_omnibus_pooled, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"pvalues_ppi_comparison_omnibus_reps{comparison_reps_for_stem}_{stamp}_five_way_comparison_omnibus.png"),
                                label=_COMPARISON_METHODS_OMNIBUS_LABEL,
                            )
                            output_paths.append(comparison_omnibus_plot_path)
                            print(f"Saved plot: {comparison_omnibus_plot_path}")

                # Label-efficiency check (run_ppi_label_efficiency_check):
                # self-contained (builds its own continuous/likert/binary
                # sources internally, no dependency on comparison_sources/
                # power_sources above), so it gets its own opt-out flag rather
                # than riding along with --no-comparison-check.
                if not getattr(args, "no_label_efficiency_check", False):
                    label_eff_reps = (getattr(args, "label_efficiency_reps", None)
                                      or getattr(args, "effect_reps", 200))
                    print(f"\npvalues simulation (PPI-corrected, label efficiency) -- "
                          f"reps={label_eff_reps}, n_boot={args.ppi_n_boot}")
                    label_eff_results, label_eff_raw, label_eff_calib_rows = run_ppi_label_efficiency_check(
                        n_reps=label_eff_reps, n_boot=args.ppi_n_boot,
                        seed=args.seed + 14, n_workers=getattr(args, "workers", 1), progress_mode=args.progress,
                    )
                    if args.eval_types:
                        requested = set(args.eval_types)
                        label_eff_results = [r for r in label_eff_results if r.eval_type in requested]
                        label_eff_raw = [r for r in label_eff_raw if r.eval_type in requested]
                        label_eff_calib_rows = [row for row in label_eff_calib_rows if row[0] in requested]
                    if label_eff_results:
                        print_ppi_label_efficiency_report(label_eff_results)
                        label_eff_stem = f"pvalues_ppi_label_efficiency_reps{label_eff_reps}_{stamp}"
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_label_efficiency(
                                results=label_eff_results, out_dir=args.out_dir, run_stem=label_eff_stem,
                            )
                            output_paths += save_results_artifacts_ppi_label_efficiency_raw(
                                raw=label_eff_raw, calib_rows=label_eff_calib_rows,
                                out_dir=args.out_dir, run_stem=label_eff_stem,
                            )
                        if args.plots == "save":
                            # The paper's compact 1x3 figure, rendered at the
                            # settings the paper uses (replot_labeleff_compact
                            # .PAPER_KWARGS) so a run reproduces it without
                            # anyone having to remember --design equiv
                            # --height 1.9. Reads the results CSV this run just
                            # wrote, so it is skipped when --save-results is off.
                            _le_csv = Path(args.out_dir) / f"{label_eff_stem}_ppi_label_efficiency_results.csv"
                            if _le_csv.exists():
                                try:
                                    # First plot written when every other check
                                    # is disabled (--official-tests' label-
                                    # efficiency-only preset), so plots_dir may
                                    # not exist yet -- the other plotters create
                                    # it themselves, render() does not.
                                    Path(plots_dir).mkdir(parents=True, exist_ok=True)
                                    from simulations.replot_labeleff_compact import (
                                        PAPER_KWARGS as _LE_PAPER, render as _render_le)
                                    _got = _render_le(str(_le_csv),
                                                      str(Path(plots_dir) / f"{label_eff_stem}_compact.png"),
                                                      **_LE_PAPER)
                                    if _got:
                                        output_paths.append(_got)
                                except Exception as _e:  # never fail a finished sweep on a plot
                                    print(f"  (compact label-efficiency figure skipped: {_e})")
                            else:
                                print("  (compact label-efficiency figure needs --save-results save)")
                            # One pooled figure + one per effect-size arm -- see
                            # save_ppi_label_efficiency_plots' docstring for why
                            # the per-es views are kept rather than only pooled.
                            for label_eff_plot_path in save_ppi_label_efficiency_plots(
                                label_eff_results, out_path=str(Path(plots_dir) / f"{label_eff_stem}_plot.png"),
                            ):
                                output_paths.append(label_eff_plot_path)
                                print(f"Saved plot: {label_eff_plot_path}")
                            # Per-method views alongside the pooled ones. The pooled
                            # multiplier inverts an average across methods, which
                            # conflates "what PPI buys for this test" with "how
                            # powerful this test is" -- see the per-method table's
                            # docstring. Reference curves are disk-cached, so after
                            # the first run this is a file read per method.
                            try:
                                pm_paths, pm_points = save_ppi_label_efficiency_plots_per_method(
                                    label_eff_raw, label_eff_calib_rows,
                                    out_path=str(Path(plots_dir) / f"{label_eff_stem}_plot.png"),
                                    seed=args.seed + 14,
                                )
                                for pm in pm_paths:
                                    output_paths.append(pm)
                                    print(f"Saved plot: {pm}")
                                if pm_points:
                                    output_paths.append(save_ppi_label_efficiency_per_method_table(
                                        pm_points, out_dir=args.out_dir, run_stem=label_eff_stem))
                                    # Judge-error-SHAPE robustness. Lives here
                                    # rather than in the pooled bundle because it
                                    # splits by test family, which the pooled
                                    # points have already averaged away -- and that
                                    # average cancels the effect (see the figure's
                                    # docstring). Non-fatal: a sweep filtered to one
                                    # noise family is a legitimate way to run.
                                    # "How good must the judge be?" -- ONE FIGURE PER
                                    # TEST FAMILY, each against its own
                                    # correlation. A single pooled figure labelled
                                    # "squared Pearson" but averaging rank tests
                                    # into its y-axis pointed Wilcoxon users at the
                                    # wrong statistic.
                                    for _kind, _lbl in (("pearson", "parametric"),
                                                        ("spearman", "rank"),
                                                        ("mixed", "pooled")):
                                        _sub = [q for k, v in pm_points.items()
                                                for q in v
                                                if _kind == "mixed"
                                                or _METHOD_CORR_KIND.get(k[2], (None, "pearson"))[1] == _kind]
                                        if not _sub:
                                            continue
                                        try:
                                            _tp = save_ppi_label_efficiency_threshold_plot(
                                                _sub,
                                                str(Path(plots_dir) / f"{label_eff_stem}_plot_threshold_{_lbl}.png"),
                                                corr_kind=_kind)
                                            output_paths.append(_tp)
                                            print(f"Saved plot: {_tp}")
                                        except Exception as exc:
                                            print(f"  (threshold figure [{_lbl}] skipped: "
                                                  f"{type(exc).__name__}: {exc})")
                                    try:
                                        _lg = save_ppi_label_efficiency_lookup_grid(
                                            pm_points,
                                            str(Path(plots_dir) / f"{label_eff_stem}_plot_lookup_grid.png"))
                                        output_paths.append(_lg)
                                        print(f"Saved plot: {_lg}")
                                        # compact 1x4 row -- the variant the paper prints
                                        # (media/simulations/labeleff_lookup_row.png). Emitted
                                        # here so a run produces it directly, rather than only
                                        # via replot_label_efficiency.py after the fact.
                                        _lr = save_ppi_label_efficiency_lookup_grid(
                                            pm_points,
                                            str(Path(plots_dir) / f"{label_eff_stem}_plot_lookup_row.png"),
                                            compact=True)
                                        output_paths.append(_lr)
                                        print(f"Saved plot: {_lr}")
                                    except Exception as exc:
                                        print(f"  (lookup grid skipped: {type(exc).__name__}: {exc})")
                                    try:
                                        _nf = save_ppi_label_efficiency_noise_family_plot(
                                            pm_points,
                                            str(Path(plots_dir) / f"{label_eff_stem}_plot_noisefamily.png"))
                                        output_paths.append(_nf)
                                        print(f"Saved plot: {_nf}")
                                        # compact variant -- the one the paper prints
                                        # (media/simulations/labeleff_noisefamily_compact.png).
                                        # Both need >=2 noise families in the sweep, so a
                                        # single-family run skips them with the reason below
                                        # rather than emitting a misleading one-family figure.
                                        _nfc = save_ppi_label_efficiency_noise_family_plot(
                                            pm_points,
                                            str(Path(plots_dir) / f"{label_eff_stem}_plot_noisefamily_compact.png"),
                                            compact=True)
                                        output_paths.append(_nfc)
                                        print(f"Saved plot: {_nfc}")
                                    except Exception as exc:
                                        print(f"  (noise-family figure skipped: "
                                              f"{type(exc).__name__}: {exc})")
                            except Exception as exc:
                                # Diagnostic output must never take down a sweep that
                                # has already produced its primary artifacts -- but
                                # say WHAT failed, with the type, so a NameError or
                                # signature slip is not mistaken for a data problem.
                                print(f"  (per-method label-efficiency output skipped: "
                                      f"{type(exc).__name__}: {exc})")
                        key_metrics["ppi_label_efficiency_n_results"] = len(label_eff_results)

                # N-formula check (run_ppi_nformula_check): opt-in, separate from
                # the label-efficiency check above -- see --nformula-check's help
                # text for why this is kept as its own toggle/output rather than
                # folded into --no-label-efficiency-check.
                if getattr(args, "nformula_check", False):
                    nformula_reps = getattr(args, "nformula_reps", 100)
                    nformula_n_boot = getattr(args, "nformula_n_boot", 500)
                    print(f"\npvalues simulation (PPI-corrected, label efficiency N-formula check) -- "
                          f"reps={nformula_reps}, n_boot={nformula_n_boot}")
                    nformula_results, nformula_raw, nformula_calib_rows = run_ppi_nformula_check(
                        n_reps=nformula_reps, n_boot=nformula_n_boot,
                        seed=args.seed + 15, n_workers=getattr(args, "workers", 1), progress_mode=args.progress,
                    )
                    if args.eval_types:
                        requested = set(args.eval_types)
                        nformula_results = [r for r in nformula_results if r.eval_type in requested]
                        nformula_raw = [r for r in nformula_raw if r.eval_type in requested]
                        nformula_calib_rows = [row for row in nformula_calib_rows if row[0] in requested]
                    if nformula_results:
                        print_ppi_nformula_report(nformula_results)
                        nformula_stem = f"pvalues_ppi_nformula_reps{nformula_reps}_{stamp}"
                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_nformula(
                                results=nformula_results, out_dir=args.out_dir, run_stem=nformula_stem,
                            )
                            output_paths += save_results_artifacts_ppi_label_efficiency_raw(
                                raw=nformula_raw, calib_rows=nformula_calib_rows,
                                out_dir=args.out_dir, run_stem=nformula_stem,
                            )
                        key_metrics["ppi_nformula_n_results"] = len(nformula_results)

                # rho effect-size drift check (run_ppi_rho_drift_check): opt-in,
                # and deliberately its own toggle rather than another arm of the
                # label-efficiency check -- it sweeps a DIFFERENT axis (the true
                # effect, held wide) with judge quality pinned, which is the exact
                # inverse of what the label-efficiency check holds fixed.
            if getattr(args, "rho_drift_check", False) or getattr(args, "rho_drift_only", False):
                # None = "not set on the command line": resolve it from WHICH
                # flag asked for the check. --rho-drift-only means "run just
                # this check", i.e. the official-precision case, and 200 reps
                # cannot support its control -- the variance ratio carries
                # relative SE ~sqrt(2/reps), ~10% at 200, and the pair
                # structures are worse still (D = truth_x - truth_y is
                # heavier-tailed than the group scores, so var_human_subset
                # converges more slowly). Measured at d=0, paired_t reads
                # -17.6% against its own rho2_score at 200 reps, -3.8% at 600
                # and +0.3% at 1500, so a 200-rep --rho-drift-only run reports
                # a mean-type method as badly broken when nothing is wrong.
                # That is not hypothetical: it cost a full root-cause hunt
                # (see print_ppi_rho_drift_report's STATUS item 3).
                rd_reps = getattr(args, "rho_drift_reps", None)
                if rd_reps is None:
                    rd_reps = 2000 if getattr(args, "rho_drift_only", False) else 200
                rd_n_boot = getattr(args, "rho_drift_n_boot", 500)
                rd_effects = tuple(getattr(args, "rho_drift_effects", None)
                                   or PPI_RHO_DRIFT_EFFECT_FRACS)
                rd_eval_types = tuple(args.eval_types) if args.eval_types else ("continuous",)
                print(f"\npvalues simulation (PPI-corrected, rho effect-size drift check) -- "
                      f"reps={rd_reps}, n_boot={rd_n_boot}, effects={list(rd_effects)}")
                rd_points, rd_calib = run_ppi_rho_drift_check(
                    n_reps=rd_reps, n_boot=rd_n_boot, seed=args.seed + 16,
                    effect_fracs=rd_effects,
                    n_lab_target=getattr(args, "rho_drift_nlab", 100),
                    eval_types=rd_eval_types,
                    n_workers=getattr(args, "workers", 1), progress_mode=args.progress,
                    shape_label=getattr(args, "rho_drift_shape", None),
                )
                if rd_points:
                    print_ppi_rho_drift_report(rd_points)
                    rd_stem = f"pvalues_ppi_rho_drift_reps{rd_reps}_{stamp}"
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_rho_drift(
                            points=rd_points, out_dir=args.out_dir, run_stem=rd_stem,
                        )
                    if args.plots == "save":
                        rd_plot = save_ppi_rho_drift_plot(
                            rd_points,
                            str(Path(plots_dir) / f"{rd_stem}_rho_vs_effect.png"),
                        )
                        output_paths.append(rd_plot)
                        print(f"Saved plot: {rd_plot}")
                    key_metrics["ppi_rho_drift_n_results"] = len(rd_points)

            if getattr(args, "factorial_check", False):
                factorial_likert_max = getattr(args, "factorial_likert_max", 5)
                factorial_omnibus = getattr(args, "factorial_omnibus", False)
                factorial_fast_noise = getattr(args, "factorial_fast_noise", False)
                factorial_noise_levels = PPI_FACTORIAL_NOISE_LEVELS_FAST if factorial_fast_noise else PPI_FACTORIAL_NOISE_LEVELS
                factorial_sources = build_ppi_factorial_sources(
                    likert_max=factorial_likert_max, noise_levels=factorial_noise_levels,
                )
                if args.eval_types:
                    requested = set(args.eval_types)
                    factorial_sources = [s for s in factorial_sources if s.eval_type in requested]
                if factorial_sources:
                    factorial_reps = getattr(args, "factorial_reps", 100)
                    factorial_n_boot = getattr(args, "factorial_n_boot", 500)
                    likert_note = f", likert_max={factorial_likert_max}" if factorial_likert_max != 5 else ""
                    noise_note = f", noise_levels=fast({len(factorial_noise_levels)}pt)" if factorial_fast_noise else ""
                    omnibus_methods = tuple(getattr(args, "factorial_omnibus_tests", None)
                                            or _COMPARISON_METHODS_OMNIBUS)
                    two_group_methods = tuple(getattr(args, "factorial_two_group_tests", None)
                                              or _COMPARISON_METHODS)
                    if not two_group_methods:
                        raise ValueError("--factorial-two-group-tests must be non-empty: the GLM "
                                         "report, plots and alignment sweep all consume the two-group "
                                         "results. Pass the cheapest single test (e.g. 'ttest').")
                    factorial_methods = two_group_methods + (omnibus_methods if factorial_omnibus else ())
                    omnibus_note = f" + {len(omnibus_methods)} omnibus tests {list(omnibus_methods)}" if factorial_omnibus else ""
                    print(f"\npvalues simulation (PPI-corrected, full factorial) -- "
                          f"{len(factorial_sources)} scenarios x {len(two_group_methods)} methods{omnibus_note}, "
                          f"reps={factorial_reps}, n_boot={factorial_n_boot}{likert_note}{noise_note}")
                    factorial_power_tune = not getattr(args, "factorial_no_power_tune", False)
                    factorial_results_raw = run_ppi_comparison_simulation(
                        factorial_sources, n_reps=factorial_reps, n_boot=factorial_n_boot,
                        progress_mode=args.progress, seed=args.seed + 8, n_workers=getattr(args, "workers", 1),
                        methods=factorial_methods, power_tune=factorial_power_tune,
                    )
                    factorial_results_raw_2group = [r for r in factorial_results_raw if r.method in two_group_methods]
                    factorial_results = pool_ppi_comparison_across_methods(factorial_results_raw_2group)
                    # GLM/heatmap/headline-report stay scoped to the llm_noise=0.20
                    # baseline (the only noise level non-null cells even have) --
                    # see _PPI_FACTORIAL_FORMULA's docstring for why llm_noise
                    # can't safely join that model as an eighth term. The FULL
                    # factorial_results (every noise level) is reserved for the
                    # alignment-bucketed view below.
                    factorial_results_baseline = [
                        r for r in factorial_results if _parse_ppi_factorial_name(r.name)["noise"] == 0.20
                    ]

                    omnibus_results = None
                    omnibus_results_baseline = None
                    factorial_results_raw_omnibus = None
                    if factorial_omnibus:
                        factorial_results_raw_omnibus = [
                            r for r in factorial_results_raw if r.method in omnibus_methods
                        ]
                        omnibus_results = pool_ppi_comparison_across_methods(factorial_results_raw_omnibus)
                        omnibus_results_baseline = [
                            r for r in omnibus_results if _parse_ppi_factorial_name(r.name)["noise"] == 0.20
                        ]

                    stem_lmax_suffix = f"_lmax{factorial_likert_max}" if factorial_likert_max != 5 else ""
                    stem_noise_suffix = "_fastnoise" if factorial_fast_noise else ""
                    factorial_stem = f"pvalues_ppi_factorial_reps{factorial_reps}{stem_lmax_suffix}{stem_noise_suffix}_{stamp}"

                    # Save the RAW simulation data to disk BEFORE attempting any
                    # GLM-dependent reporting below -- both print_ppi_factorial_
                    # report and save_results_artifacts_ppi_factorial's own .log
                    # step fit fit_ppi_factorial_model, which can raise (see
                    # _PPI_FACTORIAL_FORMULA_REFERENCE_LEVELS' docstring for the
                    # exact incident this ordering guards against: a run that
                    # spends 30-60+ minutes of compute must not lose that data
                    # just because the GLM/report stage afterward hits a
                    # problem). save_results_artifacts_ppi_factorial writes its
                    # CSV before attempting its own .log, so calling it here
                    # (rather than only the live console print below) is what
                    # actually persists the raw results even if the GLM step
                    # fails.
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_factorial(
                            results=factorial_results_raw, pooled_results=factorial_results_baseline,
                            alpha=args.alpha, out_dir=args.out_dir, run_stem=factorial_stem,
                            label="/".join(two_group_methods), null_results_full=factorial_results,
                            raw_results_full=factorial_results_raw_2group,
                        )
                        if omnibus_results_baseline is not None:
                            output_paths += save_results_artifacts_ppi_factorial(
                                results=factorial_results_raw, pooled_results=omnibus_results_baseline,
                                alpha=args.alpha, out_dir=args.out_dir, run_stem=factorial_stem,
                                write_csv=False, label="/".join(omnibus_methods),
                                null_results_full=omnibus_results, raw_results_full=factorial_results_raw_omnibus,
                            )

                    print_ppi_factorial_report(
                        factorial_results_baseline, alpha=args.alpha, label="/".join(two_group_methods),
                        null_results_full=factorial_results, raw_results_full=factorial_results_raw_2group,
                    )
                    if omnibus_results_baseline is not None:
                        print_ppi_factorial_report(
                            omnibus_results_baseline, alpha=args.alpha, label="/".join(omnibus_methods),
                            null_results_full=omnibus_results, raw_results_full=factorial_results_raw_omnibus,
                        )

                    if args.plots == "save":
                        factorial_plot_path = save_ppi_factorial_heatmap_plot(
                            results=factorial_results_baseline, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_stem}_slices.png"),
                        )
                        output_paths.append(factorial_plot_path)
                        print(f"Saved plot: {factorial_plot_path}")

                        factorial_plot_path_mnar = save_ppi_factorial_heatmap_plot(
                            results=factorial_results_baseline, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_stem}_slices_mnar.png"),
                            lm_fixed="mnar_strong",
                        )
                        output_paths.append(factorial_plot_path_mnar)
                        print(f"Saved plot: {factorial_plot_path_mnar}")

                        # MCAR (headline) and MNAR (stress-test) violin+strip
                        # views -- see save_ppi_factorial_typeI_violin_plot's
                        # docstring for why these are kept as two separate
                        # figures rather than one pooled-across-lm violin.
                        factorial_violin_mcar_path = save_ppi_factorial_typeI_violin_plot(
                            two_group_results=factorial_results_raw_2group,
                            omnibus_results=factorial_results_raw_omnibus or [],
                            alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_stem}_typeI_mcar.png"),
                            lm_filter="mcar",
                        )
                        output_paths.append(factorial_violin_mcar_path)
                        print(f"Saved plot: {factorial_violin_mcar_path}")

                        factorial_violin_mnar_path = save_ppi_factorial_typeI_violin_plot(
                            two_group_results=factorial_results_raw_2group,
                            omnibus_results=factorial_results_raw_omnibus or [],
                            alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_stem}_typeI_mnar.png"),
                            lm_filter="mnar",
                        )
                        output_paths.append(factorial_violin_mnar_path)
                        print(f"Saved plot: {factorial_violin_mnar_path}")

                    key_metrics["ppi_factorial_n_results"] = len(factorial_results_baseline)
                    key_metrics["ppi_factorial_likert_max"] = factorial_likert_max
                    # Unfiltered across every swept noise level -- NOT
                    # factorial_results_baseline -- since restricting the
                    # calibration headline number to the single baseline
                    # noise level (a restriction that only the GLM actually
                    # needs) was hiding the worst MNAR miscalibration, which
                    # concentrates at low noise. See print_ppi_factorial_
                    # report's null_results_full docstring.
                    null_results = [r for r in factorial_results if _parse_ppi_factorial_name(r.name)["es"] == "null"]
                    if null_results:
                        c_tot = sum(r.rejects_ppi for r in null_results)
                        n_tot = sum(r.n_reps for r in null_results)
                        key_metrics["ppi_factorial_mean_type1"] = float(c_tot / n_tot) if n_tot else float("nan")

                    if omnibus_results_baseline is not None:
                        key_metrics["ppi_factorial_omnibus_n_results"] = len(omnibus_results_baseline)
                        null_omnibus = [
                            r for r in omnibus_results if _parse_ppi_factorial_name(r.name)["es"] == "null"
                        ]
                        if null_omnibus:
                            c_tot_o = sum(r.rejects_ppi for r in null_omnibus)
                            n_tot_o = sum(r.n_reps for r in null_omnibus)
                            key_metrics["ppi_factorial_omnibus_mean_type1"] = (
                                float(c_tot_o / n_tot_o) if n_tot_o else float("nan")
                            )

                    # Judge-human alignment-bucketed view, derived from this SAME
                    # factorial run's es="null" cells (all llm_noise levels) --
                    # see build_ppi_alignment_results_from_factorial's docstring.
                    align_mc = getattr(args, "factorial_alignment_mc", 20000)
                    alignment_results = build_ppi_alignment_results_from_factorial(
                        factorial_sources, factorial_results, n_align_mc=align_mc, seed=args.seed + 9,
                    )
                    if alignment_results:
                        # Split into MCAR-only (the expected-use-case, "good
                        # experimental design" view) and MNAR-only (mild+
                        # strong pooled -- the worst-case/robustness view) --
                        # kept as separate reports/CSVs/plots rather than one
                        # pooled-across-label_mechanism view, since pooling
                        # was masking that MNAR's false-positive rate at a
                        # GIVEN alignment level can be far worse than MCAR's
                        # at that same level (see PPIAlignmentSweepResult.lm's
                        # docstring). human_human_rows is lm-independent
                        # (no labeling-mechanism concept applies to a pure
                        # two-synthetic-raters comparison) so it's only
                        # attached to the MCAR (primary) report, not repeated.
                        hh_rows = run_human_human_alignment_sweep(n_mc=align_mc, seed=args.seed + 10)

                        for lm_tag, lm_values, lm_hh_rows in (
                            ("mcar", ("mcar",), hh_rows),
                            ("mnar", ("mnar_mild", "mnar_strong"), None),
                        ):
                            lm_results = [r for r in alignment_results if r.lm in lm_values]
                            if not lm_results:
                                continue
                            print(f"\n  [[label_mechanism = {lm_tag}]]")
                            print_ppi_alignment_sweep_report(lm_results, alpha=args.alpha)
                            if lm_hh_rows is not None:
                                print_human_human_alignment_report(lm_hh_rows)

                            if args.save_results == "save":
                                output_paths += save_results_artifacts_ppi_alignment_sweep(
                                    results=lm_results, alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{factorial_stem}_alignment_{lm_tag}", human_human_rows=lm_hh_rows,
                                )
                            if args.plots == "save":
                                for view_et, view_metric, view_label, view_band_fn, view_band_source, view_symbol in _ALIGNMENT_VIEWS:
                                    if not any(r.eval_type == view_et and view_metric in r.alignment_metrics for r in lm_results):
                                        continue
                                    align_plot_path = save_ppi_alignment_sweep_plot(
                                        results=lm_results, eval_type=view_et, metric=view_metric,
                                        display_label=f"{view_label}, {lm_tag}", band_fn=view_band_fn,
                                        band_source=view_band_source, symbol=view_symbol, alpha=args.alpha,
                                        out_path=str(Path(plots_dir) / f"{factorial_stem}_alignment_{lm_tag}_{view_et}_{view_metric}.png"),
                                    )
                                    output_paths.append(align_plot_path)
                                    print(f"Saved plot: {align_plot_path}")

                                if lm_hh_rows is not None:
                                    hh_plot_path = save_human_human_alignment_plot(
                                        rows=lm_hh_rows,
                                        out_path=str(Path(plots_dir) / f"{factorial_stem}_alignment_human_human.png"),
                                    )
                                    output_paths.append(hh_plot_path)
                                    print(f"Saved plot: {hh_plot_path}")

                            key_metrics[f"ppi_alignment_sweep_{lm_tag}_n_results"] = len(lm_results)
                        key_metrics["ppi_alignment_sweep_n_results"] = len(alignment_results)

            if getattr(args, "factorial_check_binary", False):
                factorial_sources_binary = build_ppi_factorial_sources_binary()
                if args.eval_types:
                    requested = set(args.eval_types)
                    factorial_sources_binary = [s for s in factorial_sources_binary if s.eval_type in requested]
                if factorial_sources_binary:
                    factorial_reps = getattr(args, "factorial_reps", 100)
                    factorial_n_boot = getattr(args, "factorial_n_boot", 500)
                    print(f"\npvalues simulation (PPI-corrected, binary factorial) -- "
                          f"{len(factorial_sources_binary)} scenarios x {len(_COMPARISON_METHODS_BINARY)} methods "
                          f"({_COMPARISON_METHODS_BINARY_LABEL}), reps={factorial_reps}, n_boot={factorial_n_boot}")
                    factorial_binary_results_raw = run_ppi_comparison_simulation(
                        factorial_sources_binary, n_reps=factorial_reps, n_boot=factorial_n_boot,
                        progress_mode=args.progress, seed=args.seed + 12, n_workers=getattr(args, "workers", 1),
                        methods=_COMPARISON_METHODS_BINARY,
                    )
                    factorial_binary_results = pool_ppi_comparison_across_methods(factorial_binary_results_raw)
                    # GLM/heatmap/headline-report stay scoped to the
                    # llm_noise=PPI_BINARY_NOISE_BASELINE baseline, the only
                    # noise level non-null cells even have -- see
                    # _PPI_FACTORIAL_BINARY_FORMULA's docstring for why
                    # llm_noise can't safely join that model as a term. The
                    # FULL factorial_binary_results (every noise level) is
                    # reserved for the alignment-bucketed view below.
                    factorial_binary_results_baseline = [
                        r for r in factorial_binary_results
                        if _parse_ppi_factorial_binary_name(r.name)["noise"] == PPI_BINARY_NOISE_BASELINE
                    ]
                    print_ppi_factorial_binary_report(
                        factorial_binary_results_baseline, alpha=args.alpha, label=_COMPARISON_METHODS_BINARY_LABEL,
                        null_results_full=factorial_binary_results, raw_results_full=factorial_binary_results_raw,
                    )

                    factorial_binary_stem = f"pvalues_ppi_factorial_binary_reps{factorial_reps}_{stamp}"
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_factorial_binary(
                            results=factorial_binary_results_raw, pooled_results=factorial_binary_results_baseline,
                            alpha=args.alpha, out_dir=args.out_dir, run_stem=factorial_binary_stem,
                            label=_COMPARISON_METHODS_BINARY_LABEL, null_results_full=factorial_binary_results,
                            raw_results_full=factorial_binary_results_raw,
                        )
                    if args.plots == "save":
                        factorial_binary_plot_path = save_ppi_factorial_binary_heatmap_plot(
                            results=factorial_binary_results_baseline, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_binary_stem}_slices.png"),
                        )
                        output_paths.append(factorial_binary_plot_path)
                        print(f"Saved plot: {factorial_binary_plot_path}")

                        factorial_binary_plot_path_mnar = save_ppi_factorial_binary_heatmap_plot(
                            results=factorial_binary_results_baseline, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_binary_stem}_slices_mnar.png"),
                            lm_fixed="mnar_strong",
                        )
                        output_paths.append(factorial_binary_plot_path_mnar)
                        print(f"Saved plot: {factorial_binary_plot_path_mnar}")

                    key_metrics["ppi_factorial_binary_n_results"] = len(factorial_binary_results_baseline)
                    # Unfiltered across every swept noise level -- see the
                    # continuous/likert block's matching comment above.
                    null_results_binary = [
                        r for r in factorial_binary_results
                        if _parse_ppi_factorial_binary_name(r.name)["es"] == "null"
                    ]
                    if null_results_binary:
                        c_tot = sum(r.rejects_ppi for r in null_results_binary)
                        n_tot = sum(r.n_reps for r in null_results_binary)
                        key_metrics["ppi_factorial_binary_mean_type1"] = float(c_tot / n_tot) if n_tot else float("nan")

                    # Judge-human alignment-bucketed view (Cohen's kappa),
                    # derived from this SAME run's es="null" cells -- see
                    # build_ppi_alignment_results_from_factorial_binary.
                    # No human-human companion for binary yet -- measure_
                    # human_human_alignment has no binary branch.
                    align_mc = getattr(args, "factorial_alignment_mc", 20000)
                    alignment_results_binary = build_ppi_alignment_results_from_factorial_binary(
                        factorial_sources_binary, factorial_binary_results, n_align_mc=align_mc, seed=args.seed + 13,
                    )
                    if alignment_results_binary:
                        # Same MCAR/MNAR split as the continuous/likert block
                        # above -- see that block's comment for why.
                        for lm_tag, lm_values in (("mcar", ("mcar",)), ("mnar", ("mnar_mild", "mnar_strong"))):
                            lm_results = [r for r in alignment_results_binary if r.lm in lm_values]
                            if not lm_results:
                                continue
                            print(f"\n  [[label_mechanism = {lm_tag}]]")
                            print_ppi_alignment_sweep_report(lm_results, alpha=args.alpha)

                            if args.save_results == "save":
                                output_paths += save_results_artifacts_ppi_alignment_sweep(
                                    results=lm_results, alpha=args.alpha, out_dir=args.out_dir,
                                    run_stem=f"{factorial_binary_stem}_alignment_{lm_tag}",
                                )
                            if args.plots == "save":
                                for view_et, view_metric, view_label, view_band_fn, view_band_source, view_symbol in _ALIGNMENT_VIEWS:
                                    if not any(
                                        r.eval_type == view_et and view_metric in r.alignment_metrics
                                        for r in lm_results
                                    ):
                                        continue
                                    align_plot_path = save_ppi_alignment_sweep_plot(
                                        results=lm_results, eval_type=view_et, metric=view_metric,
                                        display_label=f"{view_label}, {lm_tag}", band_fn=view_band_fn,
                                        band_source=view_band_source, symbol=view_symbol, alpha=args.alpha,
                                        out_path=str(Path(plots_dir) / f"{factorial_binary_stem}_alignment_{lm_tag}_{view_et}_{view_metric}.png"),
                                    )
                                    output_paths.append(align_plot_path)
                                    print(f"Saved plot: {align_plot_path}")

                            key_metrics[f"ppi_alignment_sweep_binary_{lm_tag}_n_results"] = len(lm_results)
                        key_metrics["ppi_alignment_sweep_binary_n_results"] = len(alignment_results_binary)

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
