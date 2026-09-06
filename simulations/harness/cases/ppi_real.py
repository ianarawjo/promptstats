"""ppi_real -- PPI-correction calibration checks against REAL (human_label,
judge_score) data collected by simulations/collect_judge_bias_data.py, as a
real-data complement to pvalues.py's ``--mode ppi`` (which is synthetic-only
-- see that file's "Known exceptions" note).

A separate case module (rather than growing pvalues.py further, which is
already very large) that stays thin by importing pvalues.py's own PPI
result types, test-battery helpers, and report/plot functions directly --
they're fully generic over (name, tag, test) labeled results and have no
JudgeBiasSource-specific coupling, so they work unmodified here. See
scenarios/real_judge_bias.py's module docstring for the data-loading side.

Nine checks:

  single-sample bias/coverage
      Per judge model: does the PPI-corrected estimate of THIS corpus's
      true mean human_label recover it, with correct CI coverage, as
      label_frac (how much of the human-labeled subset is "revealed") and
      n vary? Uses evalstats.tests' _ppi_single_bootstrap_t (+
      _ppi_single_wilson for binary) -- functions pvalues.py's synthetic
      PPI sweep never exercises at all (it only ever tests group
      *comparisons*, never a plain one-sample mean estimate). Not a
      p-value/hypothesis-test check -- these functions return
      p_value=None.

  two-group Type-I null (random split, cross-judge)
      For every unique PAIR of judge models: bisect the corpus into two
      disjoint random subsamples -- a valid null by construction (both are
      random draws from the identical population, so their TRUE, human-
      label means are equal) -- but read group A through judge_a and group
      B through judge_b, not the same judge reading both. Runs the
      independent-samples tests (ttest/ttest_welch/mwu) PPI correction is
      supposed to keep calibrated,
      with real noise/skew/judge-bias characteristics instead of synthetic
      ones. Deliberately cross-judge, not same-judge-reads-both: a single
      judge reading two random halves of the same population applies its
      bias identically to both, so the bias cancels out of the A-vs-B
      difference regardless of its magnitude -- making "uncorrected" (a
      classical test on raw judge scores, no human labels) structurally
      unable to fail, which is uninformative about whether skipping PPI
      correction is actually risky. Two different judges reintroduces a
      genuine asymmetry an uncorrected test is vulnerable to -- see
      scenarios/real_judge_bias.py's generate_real_twogroup_null_cell for
      the full reasoning.

  paired Type-I null (cross-judge)
      For every unique PAIR of judge models scoring the SAME items: an
      EXACT null (stronger than the two-group check's "equal in
      distribution" -- here the true paired difference is exactly zero for
      every item, since both judges are noisy/biased reads of the
      IDENTICAL human_label). Runs the paired-samples tests (wilcoxon/
      paired_t/bayes_bootstrap/bootstrap_t/tango) that a single judge model
      alone can't exercise at all. Already cross-judge from the start (see
      the two-group check above, which was redesigned to match this one).
      With k judge models collected, all C(k, 2) pairs are checked (capped
      by --max-pairs) -- more judges means more independent (dataset,
      label_frac, n, pair) cells feeding the SAME calibration question,
      i.e. more power to catch a miscalibration than any single pair would
      give alone.

  within-item paired bias/coverage (wmt_da_paired, additive/optional)
      The paired-null check above pits two DIFFERENT judge models against
      each other as a proxy for real paired A/B data -- a real deployment
      more often has ONE judge scoring TWO conditions for the SAME item
      (e.g. response A vs response B for the same prompt), which is a
      different disagreement structure (inter-judge vs within-judge/
      across-condition) that the proxy can't exercise. This check instead
      uses simulations/collect_judge_bias_data.py's --types
      continuous_paired collection (RicardoRei/wmt-da-human-evaluation
      regrouped by source segment: the SAME source sentence translated by
      >=2 different MT systems, each independently human-DA-scored) --
      genuine within-item paired data for a single judge. Unlike the
      Type-I null checks above, system A and system B have REAL, generally
      DIFFERENT human labels (not a constructed identical label), so this
      is a bias/coverage check (does the PPI-corrected paired estimate
      recover the corpus's TRUE population paired difference, with correct
      CI coverage?) -- the paired-structure analogue of the single-sample
      check, sharing its PPIEffectResult/report/plot machinery. Entirely
      ADDITIVE to the three checks above (skipped with a warning, not a
      hard failure, if judge_bias_wmt_da_paired.csv hasn't been collected
      yet -- see --no-wmt-paired-check to disable explicitly).

  omnibus-independent Type-I null (3-group, cross-judge)
      The 3-group generalization of the two-group check: for every unique
      TRIPLE of judge models, bisect the corpus into three disjoint random
      subsamples read through three DIFFERENT judges -- same cross-judge
      reasoning as the two-group check (see generate_real_
      omnibus_independent_null_cell). Runs anova_ind and kruskal (NOT
      kruskal_mnar_experimental -- see _omnibus_independent_methods_for's
      docstring for why, same reasoning as dropping the local-rectifier
      MWU variants from the two-group check). With k judge models, all C(k, 3) triples
      are checked (capped by --max-triples).

  omnibus-repeated Type-I null (3-condition, cross-judge)
      The 3-condition generalization of the paired check: for every unique
      TRIPLE of judge models scoring the SAME items, an EXACT null (the
      true value is identical across all three conditions for every item).
      Runs anova_rep and friedman -- the repeated-measures tests a single
      judge model, or even a pair, can't exercise (see generate_real_
      omnibus_repeated_null_cell).

  positive-control power (twogroup-power, omnibus-independent-power,
  wmt-paired-power; --no-power-check to disable)
      The null checks above all ask "does correction avoid FALSE
      positives" -- this asks the complementary question, "does correction
      still find TRUE positives," using a genuine, EXACTLY-known non-zero
      real effect instead of an artificial null. twogroup-power/omnibus-
      independent-power reuse the SAME cross-judge machinery as their null-
      check counterparts, but split the drawn sample by RANK on
      human_label (top vs. bottom half/third) instead of randomly -- see
      generate_real_twogroup_power_cell/generate_real_omnibus_independent_
      power_cell's docstrings for why this stays as mechanically clean a
      construction as the null checks' random split (deliberately NOT the
      "natural categorical split" scoped out below -- a rank split on
      human_label is the target quantity itself, not an extraneous
      nuisance variable). wmt-paired-power reuses generate_real_wmt_
      paired_bias_cell unchanged (that data already has a genuine paired
      effect built in -- two different real MT systems' human DA scores
      for the same segment), just checking p < alpha instead of collecting
      bias/coverage stats. Additive to (and reuses the pairs/triples/sizes
      of) the twogroup/omnibus-independent/wmt-paired-bias checks above --
      no omnibus-repeated-power leg, since there's no real 3-condition
      same-item corpus available the way wmt_da_paired provides one for
      2 conditions (out of scope for the same reason the natural-
      categorical-split idea below is).

Explicitly OUT of scope: a "real" (non-null) group comparison via a
natural CATEGORICAL split (e.g. WMT by language pair, App Store by app_id)
-- still deferred, since it conflates a real content difference with judge
bias, a dirtier signal than either the synthetic power check or this
module's own rank-split power check above -- and LMM/LMM_FACTORIAL/
LMM_RUNS specifically (deferred for now, unlike the other omnibus tests
above).
"""

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as _mp
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.stats as scipy_stats


class _MWUResult:
    """Shim: the mwu call sites read only ``.p_value``. They now go through
    evalstats.tests._ppi_mannwhitney_corrected -- the same path mannwhitney()
    uses -- instead of calling _ppi_two_sample directly."""
    __slots__ = ("p_value",)

    def __init__(self, p):
        self.p_value = p


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.tests import (
        _ppi_single_bootstrap_t,
        _ppi_single_wilson,
        _ppi_single_logit_t,
        _ppi_single_t_interval,
        _ppi_two_sample,
    _ppi_mannwhitney_corrected,
        _ppi_paired_arrays,
        _ppi_paired_bayes_bootstrap,
        _ppi_paired_bootstrap_t,
        _ppi_paired_mj_floor,
        _ppi_paired_bonett_price,
        _ppi_paired_t_interval,
        _ppi_paired_logit_t,
        _p_x_gt_y_midrank,
        paired_walsh_midrank_theta,
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _kw_candidate_from_pairwise,
        _ppi_kruskal_wallis_influence,
        _kw_rowsum_from_pairwise,
        _ppi_kruskal_wallis_pairwise,
    )

from ..methods import (
    TTEST, TTEST_WELCH, MWU, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, MJ_FLOOR,
    PPI_BONETT_PRICE,
    PPI_T_INTERVAL, PPI_LOGIT_T, PPI_T_INTERVAL_SINGLE, PPI_LOGIT_T_SINGLE,
    ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL,
    KRUSKAL_ROWSUM,
    KRUSKAL_ROWSUM_LABELED,
    KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE, KRUSKAL_INFLUENCE_LOGO,
    KRUSKAL_INFLUENCE_FLOOR,
    PPI_TEST_METHODS, get_method_color,
)
from ..scenarios.real_judge_bias import (
    REAL_JUDGE_BIAS_DATASETS,
    DEFAULT_DATA_DIR,
    RealJudgeBiasCorpus,
    load_real_judge_bias_corpus,
    default_size_grid,
    all_judge_pairs,
    all_judge_triples,
    generate_real_single_cell,
    generate_real_twogroup_null_cell,
    generate_real_paired_null_cell,
    generate_real_omnibus_independent_null_cell,
    generate_real_omnibus_repeated_null_cell,
    generate_real_twogroup_power_cell,
    generate_real_omnibus_independent_power_cell,
    RealWithinItemPairedCorpus,
    load_real_wmt_paired_corpus,
    generate_real_wmt_paired_bias_cell,
)
from .pvalues import (
    PPIResult,
    PPIEffectResult,
    print_ppi_report,
    print_ppi_effect_report,
    save_ppi_typeI_plot,
    save_ppi_effect_plot,
    save_results_artifacts_ppi,
    save_results_artifacts_ppi_effect,
    _effect_cell_stats,
    _uncorrected_bias_z,
    _uncorrected_bayes_bootstrap_paired_p_value,
    _uncorrected_bootstrap_t_paired_p_value,
    _uncorrected_mj_floor_paired_p_value,
    _uncorrected_bonett_price_paired_p_value,
    _uncorrected_anova_independent_p_value,
    _uncorrected_anova_repeated_p_value,
    _uncorrected_friedman_p_value,
    _uncorrected_kruskal_p_value,
    _ProgressReporter,
    _ALPHA,
    _PPI_NONSTANDARD_TESTS,
    _ppi_wilson_interval,
    _alpha_label,
    _pretty_test,
    _metric_pct,
    _alignment_bucket,
)
from . import CaseResult

CASE_NAME = "ppi_real"

DEFAULT_LABEL_FRACS = [0.05, 0.10, 0.20, 0.40]
_SINGLE_METHOD_BOOTSTRAP_T = "bootstrap_t"
_SINGLE_METHOD_WILSON = "ppi_wilson"
"""Deliberately not "wilson" -- see methods.PPI_WILSON's docstring: that
name is already taken by ci_single.py's plain (non-PPI) Wilson CI."""

_RATER_NOISE_SD = 0.03
"""Fixed small std (on the shared [0, 1] rescaled score) for the
independent per-arm label noise generate_real_paired_null_cell/
generate_real_omnibus_repeated_null_cell inject via `rater_noise_sd` --
see those functions' docstrings and _independent_rater_copies for why an
exact-tie construction alone (rater_noise_sd=0) is an unrealistically
idealized worst case for a Type-I claim (real, independent human ratings
essentially never agree to floating-point precision). A fixed, modest
value rather than one calibrated per-dataset to a measured inter-rater
reliability figure -- those aren't available for every dataset here, and
a fixed small value is enough to break the exact tie without overstating
how noisy real annotation actually is."""

_DEGENERATE_LABEL_PROB = 0.10
"""Fraction of paired-null/repeated-null reps that use the exact-tie
construction (rater_noise_sd=0) rather than the noisy one -- the
degenerate case is still worth exercising directly (it's what caught a
real cross-fit bug in wilcoxon's power-tuning), but shouldn't dominate
the sweep now that a more realistic alternative exists. Reps of both
kinds are pooled into the SAME corrected/uncorrected counters (no
separate reporting) -- the reported rate is the average over both
regimes, weighted by this probability."""


# ---------------------------------------------------------------------------
# Per-cell test batteries
# ---------------------------------------------------------------------------


def _run_real_single_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """n_reps replicates of the single-sample bias/coverage check, capturing
    each method's (estimate, ci_low, ci_high, llm_estimate) per rep -- same
    tuple shape pvalues.py's _run_ppi_effect_cell collects, so
    _effect_cell_stats/_uncorrected_bias_z apply unchanged."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for _ in range(n_reps):
        judge, lab = generate_real_single_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _SINGLE_METHOD_BOOTSTRAP_T in methods:
                try:
                    r = _ppi_single_bootstrap_t(judge, lab, _ALPHA, n_boot, int(rng.integers(0, 2 ** 31)))
                    out[_SINGLE_METHOD_BOOTSTRAP_T].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if _SINGLE_METHOD_WILSON in methods:
                try:
                    r = _ppi_single_wilson(judge, lab, _ALPHA)
                    out[_SINGLE_METHOD_WILSON].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_LOGIT_T_SINGLE.name in methods:
                try:
                    # Closed-form logit-t, PPI_AUTO_METHOD_TABLE's "bounded_01"
                    # robustness method (both judge and human are already
                    # rescaled to [0, 1] -- see RealJudgeBiasCorpus) -- the
                    # non-binary counterpart to the Wilson branch above.
                    # PPI_LOGIT_T_SINGLE, NOT PPI_LOGIT_T -- this is a
                    # single-sample mean estimand, a different quantity from
                    # PPI_LOGIT_T's PAIRED mean-difference role in
                    # _run_real_paired_cell/_run_real_wmt_paired_bias_cell;
                    # using the same test name here would silently pool two
                    # different estimands' bias/coverage stats together in
                    # print_ppi_effect_report (name-only pooling) -- see
                    # PPI_T_INTERVAL_SINGLE/PPI_LOGIT_T_SINGLE's docstrings.
                    r = _ppi_single_logit_t(judge, lab, _ALPHA)
                    out[PPI_LOGIT_T_SINGLE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_T_INTERVAL_SINGLE.name in methods:
                try:
                    # Closed-form t-interval, PPI_AUTO_METHOD_TABLE's
                    # "unbounded" robustness method -- see PPI_LOGIT_T_SINGLE
                    # branch above for the naming-collision reasoning
                    # (same applies here vs. paired PPI_T_INTERVAL).
                    r = _ppi_single_t_interval(judge, lab, _ALPHA)
                    out[PPI_T_INTERVAL_SINGLE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

    return dict(out)


def _run_real_twogroup_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the two-group Type-I null check (random-split
    real data, cross-judge -- see generate_real_twogroup_null_cell's
    docstring for why group A and group B are read through two DIFFERENT
    judges, not the same one), mirroring pvalues.py's _run_ppi_cell
    independent-groups branches (ttest/ttest_welch/mwu only -- see
    _run_real_paired_cell for the paired-samples family)."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        a, b, lab_a, lab_b = generate_real_twogroup_null_cell(corpus, rng, n, label_frac, judge_a, judge_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if TTEST_WELCH.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if MWU.name in methods:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    uncorrected[MWU.name] += int(p_u < _ALPHA)
                    _e, _ci, _p, _rec, _lam = _ppi_mannwhitney_corrected(a, b, lab_a, lab_b, _ALPHA, n_boot, _rng_seed())
                    r = _MWUResult(_p)
                    corrected[MWU.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass




    return corrected, uncorrected


def _run_real_paired_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the paired Type-I null check (same items,
    scored by two DIFFERENT judges) -- mirrors pvalues.py's _run_ppi_cell
    paired-groups branches (wilcoxon/paired_t/ppi_t_interval/ppi_logit_t/
    tango). See generate_real_paired_null_cell for why this is an exact
    null rather than merely an equal-in-distribution one.

    Each rep independently draws exact-tie (rater_noise_sd=0, probability
    _DEGENERATE_LABEL_PROB) vs. independent-small-noise
    (rater_noise_sd=_RATER_NOISE_SD, the rest) labels -- see those
    constants' docstrings. Both kinds feed the SAME corrected/uncorrected
    counters below, so the reported rate is already the pooled average
    over both regimes; no separate reporting needed."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        noise_sd = 0.0 if rng.random() < _DEGENERATE_LABEL_PROB else _RATER_NOISE_SD
        llm_x, llm_y, lab_x, lab_y = generate_real_paired_null_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, rater_noise_sd=noise_sd,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if WILCOXON.name in methods:
                try:
                    p_u = float(scipy_stats.wilcoxon(llm_x, llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    # paired_walsh_midrank_theta (evalstats.ppi -- a Hodges-Lehmann
                    # Walsh-average midrank-sign statistic, matched estimator/
                    # rectifier), not a plain median: under heavy ties (e.g. real
                    # Likert-like human ratings, appstore's 1-5 stars), the
                    # population median of a paired difference can stay locked at
                    # exactly 0 even under a large, real, classical-Wilcoxon-
                    # detectable shift -- a wrong-estimand problem the Walsh-average
                    # construction avoids, though a known small-n_lab extreme-tie
                    # Type-I residual remains -- see that function's docstring in
                    # evalstats/ppi.py.
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, paired_walsh_midrank_theta, _ALPHA, n_boot, _rng_seed(), rectifier_func=paired_walsh_midrank_theta)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PAIRED_T.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PAIRED_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[PAIRED_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in methods:
                try:
                    p_u = _uncorrected_bayes_bootstrap_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BAYES_BOOTSTRAP.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bayes_bootstrap(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BAYES_BOOTSTRAP.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BOOTSTRAP_T.name in methods:
                try:
                    p_u = _uncorrected_bootstrap_t_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BOOTSTRAP_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bootstrap_t(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BOOTSTRAP_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if MJ_FLOOR.name in methods:
                try:
                    p_u = _uncorrected_mj_floor_paired_p_value(llm_x - llm_y)
                    uncorrected[MJ_FLOOR.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_mj_floor(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[MJ_FLOOR.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PPI_BONETT_PRICE.name in methods:
                try:
                    p_u = _uncorrected_bonett_price_paired_p_value(llm_x - llm_y)
                    uncorrected[PPI_BONETT_PRICE.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bonett_price(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[PPI_BONETT_PRICE.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PPI_T_INTERVAL.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PPI_T_INTERVAL.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_t_interval(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[PPI_T_INTERVAL.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PPI_LOGIT_T.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PPI_LOGIT_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_logit_t(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[PPI_LOGIT_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


def _run_real_omnibus_independent_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, judge_c: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the omnibus-independent Type-I null check (3-way
    random split, cross-judge -- see generate_real_omnibus_independent_
    null_cell's docstring), mirroring pvalues.py's _run_ppi_cell's
    ANOVA_IND/KRUSKAL branches. See _omnibus_independent_methods_for for
    why KRUSKAL_MNAR_EXPERIMENTAL is not wired up here."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        groups, groups_lab = generate_real_omnibus_independent_null_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, judge_c,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ANOVA_IND.name in methods:
                try:
                    p_u = _uncorrected_anova_independent_p_value(groups)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups, groups_lab, k=len(groups))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

            if KRUSKAL.name in methods:
                try:
                    p_u = _uncorrected_kruskal_p_value(groups)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    pass

            # Row-sum ("real Kruskal-Wallis") projection. Deliberately its
            # OWN bootstrap draw rather than reusing KRUSKAL's: sharing one
            # would shift KRUSKAL's rng stream and silently change every
            # existing kruskal number in this case. The uncorrected
            # comparator is the same classical KW on the judge scores --
            # this projection is the PPI correction OF that statistic, so it
            # is the directly matched pair.
            if any(_m.name in methods for _m in (KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                                                KRUSKAL_TWOPART, KRUSKAL_EIGENGAP,
                                                KRUSKAL_INFLUENCE,
                                                KRUSKAL_INFLUENCE_LOGO,
                                                KRUSKAL_INFLUENCE_FLOOR)):
                try:
                    p_u = _uncorrected_kruskal_p_value(groups)
                    pw_r = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    for _m, _w in ((KRUSKAL_ROWSUM, "full"), (KRUSKAL_ROWSUM_LABELED, "labeled")):
                        if _m.name in methods:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_rowsum_from_pairwise(pw_r, weights=_w)["wald_p"] < _ALPHA)
                    # The k>=5-conservatism candidates share pw_r too -- same
                    # bootstrap, different test form, so the comparison between
                    # them carries no Monte Carlo difference of its own.
                    for _m, _c in ((KRUSKAL_TWOPART, "twopart"),
                                   (KRUSKAL_EIGENGAP, "eigengap")):
                        if _m.name in methods:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_candidate_from_pairwise(pw_r, _c, _ALPHA)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE_LOGO.name in methods:
                        uncorrected[KRUSKAL_INFLUENCE_LOGO.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE_LOGO.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed(),
                                loo_group=True, floor_frac=0.5)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE_FLOOR.name in methods:
                        uncorrected[KRUSKAL_INFLUENCE_FLOOR.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE_FLOOR.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed(),
                                loo_group=False, floor_frac=0.5)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE.name in methods:
                        # Its own draw: it needs the raw groups, and the
                        # covariance replacement is the whole point, so it does
                        # not share pw_r.
                        uncorrected[KRUSKAL_INFLUENCE.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed()
                            )["wald_p"] < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


def _run_real_omnibus_repeated_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, judge_c: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the omnibus-repeated Type-I null check (same
    items, 3 different judges -- see generate_real_omnibus_repeated_
    null_cell's docstring), mirroring pvalues.py's _run_ppi_cell's
    ANOVA_REP/FRIEDMAN branches.

    Same exact-tie-vs-noisy per-rep draw as _run_real_paired_cell -- see
    that function's docstring and _DEGENERATE_LABEL_PROB/_RATER_NOISE_SD."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    for _ in range(n_reps):
        noise_sd = 0.0 if rng.random() < _DEGENERATE_LABEL_PROB else _RATER_NOISE_SD
        groups, groups_lab = generate_real_omnibus_repeated_null_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, judge_c, rater_noise_sd=noise_sd,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ANOVA_REP.name in methods:
                try:
                    p_u = _uncorrected_anova_repeated_p_value(groups)
                    uncorrected[ANOVA_REP.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_repeated_p_value(groups, groups_lab, k=len(groups))
                    corrected[ANOVA_REP.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

            if FRIEDMAN.name in methods:
                try:
                    p_u = _uncorrected_friedman_p_value(groups)
                    uncorrected[FRIEDMAN.name] += int(p_u < _ALPHA)
                    p = _ppi_friedman_p_value(groups, groups_lab, k=len(groups))
                    corrected[FRIEDMAN.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


# ---------------------------------------------------------------------------
# Positive-control (power) checks: same cross-judge machinery as the Type-I
# null checks above, but the drawn sample has a genuine, exactly-known
# non-zero true effect (see generate_real_twogroup_power_cell/generate_real_
# omnibus_independent_power_cell's docstrings for the rank-split
# construction, and generate_real_wmt_paired_bias_cell -- already used by
# the within-item bias/coverage check -- for the paired leg, which already
# has a genuine real paired effect built in). "corrected"/"uncorrected"
# dicts here count REJECTIONS same as the null-check runners, but here a
# rejection is the DESIRED outcome (power), not a false positive.
# ---------------------------------------------------------------------------


def _run_real_twogroup_power_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the two-group power check (rank-split real data,
    cross-judge -- see generate_real_twogroup_power_cell), mirroring
    _run_real_twogroup_cell's test battery exactly (ttest/ttest_welch/mwu)
    -- only the cell generator differs."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        a, b, lab_a, lab_b, _true_diff = generate_real_twogroup_power_cell(corpus, rng, n, label_frac, judge_a, judge_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if TTEST_WELCH.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if MWU.name in methods:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    uncorrected[MWU.name] += int(p_u < _ALPHA)
                    _e, _ci, _p, _rec, _lam = _ppi_mannwhitney_corrected(a, b, lab_a, lab_b, _ALPHA, n_boot, _rng_seed())
                    r = _MWUResult(_p)
                    corrected[MWU.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass




    return corrected, uncorrected


def _run_real_omnibus_independent_power_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, judge_c: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the omnibus-independent power check (3-way
    rank-split real data, cross-judge -- see generate_real_omnibus_
    independent_power_cell), mirroring _run_real_omnibus_independent_cell's
    ANOVA_IND/KRUSKAL branches exactly -- only the cell generator differs."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        groups, groups_lab, _true_range = generate_real_omnibus_independent_power_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, judge_c,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ANOVA_IND.name in methods:
                try:
                    p_u = _uncorrected_anova_independent_p_value(groups)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups, groups_lab, k=len(groups))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

            if KRUSKAL.name in methods:
                try:
                    p_u = _uncorrected_kruskal_p_value(groups)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    pass

            # Row-sum ("real Kruskal-Wallis") projection. Deliberately its
            # OWN bootstrap draw rather than reusing KRUSKAL's: sharing one
            # would shift KRUSKAL's rng stream and silently change every
            # existing kruskal number in this case. The uncorrected
            # comparator is the same classical KW on the judge scores --
            # this projection is the PPI correction OF that statistic, so it
            # is the directly matched pair.
            if any(_m.name in methods for _m in (KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
                                                KRUSKAL_TWOPART, KRUSKAL_EIGENGAP,
                                                KRUSKAL_INFLUENCE,
                                                KRUSKAL_INFLUENCE_LOGO,
                                                KRUSKAL_INFLUENCE_FLOOR)):
                try:
                    p_u = _uncorrected_kruskal_p_value(groups)
                    pw_r = _ppi_kruskal_wallis_influence(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    for _m, _w in ((KRUSKAL_ROWSUM, "full"), (KRUSKAL_ROWSUM_LABELED, "labeled")):
                        if _m.name in methods:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_rowsum_from_pairwise(pw_r, weights=_w)["wald_p"] < _ALPHA)
                    # The k>=5-conservatism candidates share pw_r too -- same
                    # bootstrap, different test form, so the comparison between
                    # them carries no Monte Carlo difference of its own.
                    for _m, _c in ((KRUSKAL_TWOPART, "twopart"),
                                   (KRUSKAL_EIGENGAP, "eigengap")):
                        if _m.name in methods:
                            uncorrected[_m.name] += int(p_u < _ALPHA)
                            corrected[_m.name] += int(
                                _kw_candidate_from_pairwise(pw_r, _c, _ALPHA)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE_LOGO.name in methods:
                        uncorrected[KRUSKAL_INFLUENCE_LOGO.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE_LOGO.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed(),
                                loo_group=True, floor_frac=0.5)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE_FLOOR.name in methods:
                        uncorrected[KRUSKAL_INFLUENCE_FLOOR.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE_FLOOR.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed(),
                                loo_group=False, floor_frac=0.5)["wald_p"] < _ALPHA)
                    if KRUSKAL_INFLUENCE.name in methods:
                        # Its own draw: it needs the raw groups, and the
                        # covariance replacement is the whole point, so it does
                        # not share pw_r.
                        uncorrected[KRUSKAL_INFLUENCE.name] += int(p_u < _ALPHA)
                        corrected[KRUSKAL_INFLUENCE.name] += int(
                            _ppi_kruskal_wallis_influence(
                                groups, groups_lab, _ALPHA, n_boot, _rng_seed()
                            )["wald_p"] < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


def _run_real_wmt_paired_power_cell(
    corpus: RealWithinItemPairedCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the within-item paired power check -- reuses
    generate_real_wmt_paired_bias_cell UNCHANGED (it already draws a
    genuine, non-artificial paired difference: two different real MT
    systems' human DA scores for the same source segment -- see that
    function's docstring), just checking p < alpha (rejection = correct
    detection) instead of collecting (estimate, ci_low, ci_high,
    llm_estimate) tuples for a bias/coverage check like _run_real_wmt_
    paired_bias_cell does. Test battery mirrors _run_real_paired_cell's
    paired-samples family (wilcoxon/paired_t/ppi_t_interval/ppi_logit_t;
    no tango -- see _WMT_PAIRED_POWER_METHODS)."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        llm_x, llm_y, lab_x, lab_y = generate_real_wmt_paired_bias_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if WILCOXON.name in methods:
                try:
                    p_u = float(scipy_stats.wilcoxon(llm_x, llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    # paired_walsh_midrank_theta, not np.median -- see
                    # _run_real_paired_cell's WILCOXON block / that function's
                    # docstring for why (median is the wrong estimand under
                    # heavy ties, independent of any bootstrap-degeneracy fix).
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, paired_walsh_midrank_theta, _ALPHA, n_boot, _rng_seed(), rectifier_func=paired_walsh_midrank_theta)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PAIRED_T.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PAIRED_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[PAIRED_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in methods:
                try:
                    p_u = _uncorrected_bayes_bootstrap_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BAYES_BOOTSTRAP.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bayes_bootstrap(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BAYES_BOOTSTRAP.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BOOTSTRAP_T.name in methods:
                try:
                    p_u = _uncorrected_bootstrap_t_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BOOTSTRAP_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bootstrap_t(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BOOTSTRAP_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PPI_T_INTERVAL.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PPI_T_INTERVAL.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_t_interval(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[PPI_T_INTERVAL.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PPI_LOGIT_T.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PPI_LOGIT_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_logit_t(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[PPI_LOGIT_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


_WMT_PAIRED_BIAS_METHODS = [WILCOXON.name, PAIRED_T.name, PPI_T_INTERVAL.name, PPI_LOGIT_T.name]
"""_paired_methods_for("continuous") minus BAYES_BOOTSTRAP/BOOTSTRAP_T --
not part of ppi_real's official CI-comparison set (see
_paired_methods_for). PPI_T_INTERVAL/PPI_LOGIT_T are safe to include here
because the single-sample bias/coverage check uses the distinct
PPI_T_INTERVAL_SINGLE/PPI_LOGIT_T_SINGLE Method identities (split from
these paired ones in methods.py, the same way PPI_BOOTSTRAP_T_SINGLE is
split from BOOTSTRAP_T), so there's no test-name collision in
print_ppi_effect_report's by-test-name pooling -- see those Methods'
docstrings in methods.py for the full reasoning. MJ_FLOOR excluded for the
same eval_type restriction _paired_methods_for applies (this corpus is
always eval_type="continuous_paired")."""

_WMT_PAIRED_POWER_METHODS = [WILCOXON.name, PAIRED_T.name, PPI_T_INTERVAL.name, PPI_LOGIT_T.name]
"""Unlike _WMT_PAIRED_BIAS_METHODS, PPI_LOGIT_T is safe to include here --
power_results (a PPIResult bucket keyed by rejection COUNTS, not
PPIEffectResult's bias/coverage stats) never mixes this check's
"ppi_logit_t" with the single-sample check's differently-estimand
"ppi_logit_t" the way print_ppi_effect_report's by-test-name pooling
would; every "ppi_logit_t" row across twogroup_power/paired power/
omnibus power/this check already means the same thing (a paired- or
independent-samples logit-t test), same as hypothesis_results already
does for the Type-I null checks. MJ_FLOOR still excluded, for the same
eval_type restriction _paired_methods_for applies (this corpus is
always eval_type="continuous_paired")."""


def _run_real_wmt_paired_bias_cell(
    corpus: RealWithinItemPairedCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """n_reps replicates of the within-item paired bias/coverage check
    (genuine single-judge, two-condition paired data -- see
    scenarios/real_judge_bias.py's generate_real_wmt_paired_bias_cell),
    capturing each method's (estimate, ci_low, ci_high, llm_estimate) per
    rep -- same tuple shape _run_real_single_cell collects, so
    _effect_cell_stats/_uncorrected_bias_z apply unchanged. Unlike that
    function, the correct null value differs PER TEST (corpus.
    true_paired_walsh_midrank_theta for wilcoxon's Walsh-average midrank-sign
    estimand, corpus.true_paired_mean for the other three's mean estimand) --
    see run()'s _consume for where that per-test split happens. wilcoxon's
    null target is corpus.true_paired_walsh_midrank_theta (evalstats.ppi.
    paired_walsh_midrank_theta), not a plain median: under heavy ties, the
    population median of a paired difference can stay locked at exactly 0
    even under a large, real, classical-Wilcoxon-detectable shift, so
    median isn't a valid target for the estimand wilcoxon's PPI correction
    actually uses -- see that function's docstring in evalstats/ppi.py."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for _ in range(n_reps):
        llm_x, llm_y, lab_x, lab_y = generate_real_wmt_paired_bias_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if WILCOXON.name in methods:
                try:
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, paired_walsh_midrank_theta, _ALPHA, n_boot,
                                            int(rng.integers(0, 2 ** 31)), rectifier_func=paired_walsh_midrank_theta)
                    out[WILCOXON.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PAIRED_T.name in methods:
                try:
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.mean, _ALPHA, n_boot,
                                            int(rng.integers(0, 2 ** 31)), rectifier_func=np.mean)
                    out[PAIRED_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in methods:
                try:
                    r = _ppi_paired_bayes_bootstrap(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot,
                                                     int(rng.integers(0, 2 ** 31)))
                    out[BAYES_BOOTSTRAP.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_T_INTERVAL.name in methods:
                try:
                    r = _ppi_paired_t_interval(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    out[PPI_T_INTERVAL.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PPI_LOGIT_T.name in methods:
                try:
                    r = _ppi_paired_logit_t(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    out[PPI_LOGIT_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            # BOOTSTRAP_T deliberately not included here -- see
            # _WMT_PAIRED_BIAS_METHODS' docstring (its test name collides
            # with the single-sample check's own bootstrap_t branch, which
            # targets a totally different estimand, and print_ppi_effect_
            # report pools by test name only). PPI_T_INTERVAL/PPI_LOGIT_T
            # above are safe since the single-sample check uses the distinct
            # PPI_T_INTERVAL_SINGLE/PPI_LOGIT_T_SINGLE Method identities
            # instead of reusing these paired ones -- see those Methods'
            # docstrings in methods.py. BOOTSTRAP_T is not wired up at all,
            # rather than left reachable-but-unused, since nothing calls
            # this function with a methods list that would ever hit it.

    return dict(out)


def _run_real_paired_bias_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Tuple-collecting sibling of _run_real_paired_cell, using the SAME
    generate_real_paired_null_cell (same items read through two DIFFERENT
    judges, so the true paired difference is EXACTLY 0 by construction -- no
    gold-reference estimation needed, unlike _run_real_wmt_paired_bias_
    cell's genuine two-condition data). Exists because the paired binary CI is a genuine
    point-estimate/CI construction (a Wilson-style score interval for the
    paired binary discordant-pair-rate difference -- see
    evalstats.tests._ppi_paired_bonett_price's docstring) restricted to binary
    corpora by _paired_methods_for, but _run_real_paired_cell only ever
    reports corrected/uncorrected REJECTION COUNTS (a Type-I check) -- it
    never had anywhere to report (estimate, ci_low, ci_high, llm_estimate)
    tuples, so a binary corpus like arena had no path into the bias/
    coverage view (print_ppi_effect_report / ci_methods_comparison_real.png)
    even though it has exactly the paired binary data the method needs. Currently
    only PPI_BONETT_PRICE uses this; null_value is always 0.0 for every test reachable
    here (see run()'s _consume)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for _ in range(n_reps):
        llm_x, llm_y, lab_x, lab_y = generate_real_paired_null_cell(corpus, rng, n, label_frac, judge_a, judge_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")


            if PPI_BONETT_PRICE.name in methods:
                try:
                    r = _ppi_paired_bonett_price(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    out[PPI_BONETT_PRICE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

    return dict(out)


_REAL_CORPORA: list[RealJudgeBiasCorpus] = []
"""Set by run() right before creating the worker Pool (fork context), so
worker processes inherit it via copy-on-write instead of the (potentially
large) corpus arrays being re-pickled through the task queue on every
work item -- same pattern as pvalues.py's _PAIRWISE_SOURCES /
_MULTIARM_SOURCES globals. Only meaningful in the child processes; the
parent addresses corpora directly."""

_REAL_WMT_PAIRED_CORPORA: list[RealWithinItemPairedCorpus] = []
"""Same fork-inheritance pattern as _REAL_CORPORA, kept as a SEPARATE list
(rather than folded into _REAL_CORPORA) since RealWithinItemPairedCorpus is
a different type loaded from a different file -- there's exactly one
wmt_da_paired corpus (index 0) when present, zero when the data hasn't
been collected yet (see run()'s try/except around load_real_wmt_paired_corpus)."""


def _run_ppi_real_cell_worker(args: tuple) -> dict:
    """Runs ONE (single|twogroup|paired|paired_bias|omnibus_independent|
    omnibus_repeated|wmt_paired_bias|twogroup_power|
    omnibus_independent_power|wmt_paired_power) cell and returns everything
    run() needs to fold it into effect_results/twogroup_results/
    paired_results/omnibus_results/power_results, with no dependency on the
    original work-item's position -- required since pool.imap_unordered
    does not preserve submission order."""
    check_type, corpus_idx, name, dataset, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed = args

    if check_type in ("wmt_paired_bias", "wmt_paired_power"):
        corpus = _REAL_WMT_PAIRED_CORPORA[corpus_idx]
        if check_type == "wmt_paired_bias":
            samples_by_test = _run_real_wmt_paired_bias_cell(corpus, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed)
            return {
                "check_type": check_type, "name": name, "dataset": dataset, "n": n,
                "true_paired_mean": corpus.true_paired_mean,
                "true_paired_walsh_midrank_theta": corpus.true_paired_walsh_midrank_theta,
                "samples_by_test": samples_by_test,
            }
        corrected, uncorrected = _run_real_wmt_paired_power_cell(corpus, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }

    corpus = _REAL_CORPORA[corpus_idx]

    if check_type == "single":
        samples_by_test = _run_real_single_cell(corpus, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "corpus_mean": corpus.corpus_mean, "samples_by_test": samples_by_test,
        }

    if check_type == "paired_bias":
        judge_a, judge_b = judge_or_group
        samples_by_test = _run_real_paired_bias_cell(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "samples_by_test": samples_by_test,
        }

    if check_type in ("omnibus_independent", "omnibus_repeated", "omnibus_independent_power"):
        judge_a, judge_b, judge_c = judge_or_group
        runner = {
            "omnibus_independent": _run_real_omnibus_independent_cell,
            "omnibus_repeated": _run_real_omnibus_repeated_cell,
            "omnibus_independent_power": _run_real_omnibus_independent_power_cell,
        }[check_type]
        corrected, uncorrected = runner(corpus, methods, n, label_frac, judge_a, judge_b, judge_c, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }

    judge_a, judge_b = judge_or_group
    if check_type in ("twogroup", "twogroup_power"):
        runner = _run_real_twogroup_cell if check_type == "twogroup" else _run_real_twogroup_power_cell
        corrected, uncorrected = runner(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }
    corrected, uncorrected = _run_real_paired_cell(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
    return {
        "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
        "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
    }


def _single_methods_for(eval_type: str) -> list[str]:
    """Official single-sample CI methods to check for *eval_type*.

    _SINGLE_METHOD_BOOTSTRAP_T is not part of the official set: it matches
    pvalues.py's synthetic ppi official test, which validates the curated
    4-method PPI-corrected CI comparison -- Tango, Wilson, logit-t,
    t-interval -- not the bootstrap-based alternatives; see
    _PPI_CI_COMPARISON_TESTS. PPI_LOGIT_T_SINGLE/PPI_T_INTERVAL_SINGLE are
    PPI_AUTO_METHOD_TABLE's "bounded_01"/"unbounded" robustness methods
    (every real dataset here is already rescaled to [0, 1] -- see
    RealJudgeBiasCorpus -- so both apply, not just logit-t's "correct"
    bounded_01 pick; this is a validation comparison, not production
    auto-routing, the same convention _paired_methods_for uses), the
    non-binary counterpart to Wilson's binary role. Not PPI_LOGIT_T/
    PPI_T_INTERVAL (unsuffixed) -- those names are reserved for the paired
    estimand elsewhere in this file (_paired_methods_for/
    _WMT_PAIRED_BIAS_METHODS/_WMT_PAIRED_POWER_METHODS); reusing them for
    this single-sample mean estimand would silently pool two different
    estimands' stats together in print_ppi_effect_report -- see
    PPI_T_INTERVAL_SINGLE/PPI_LOGIT_T_SINGLE's docstrings in methods.py.
    _ppi_single_bootstrap_t/_SINGLE_METHOD_BOOTSTRAP_T remain fully
    implemented and runnable via _run_real_single_cell for anyone who
    wants them back -- only the official roster changed."""
    if eval_type == "binary":
        return [_SINGLE_METHOD_WILSON]
    return [PPI_LOGIT_T_SINGLE.name, PPI_T_INTERVAL_SINGLE.name]


def _twogroup_methods_for(eval_type: str) -> list[str]:
    """Official two-group CI methods to check for *eval_type*.

    MWU (rank-based) isn't in pvalues.py's _PPI_BINARY_COMPATIBLE_TESTS --
    binary's massive ties break the rank-based judge-bias noise model
    there, same restriction applies here.

    The local-rectifier MWU variants (mwu_mnar_experimental,
    mwu_mnar_pooled, mwu_adaptive, mwu_ridge) were removed outright -- see
    MWU's comment in methods.py. They had never been in
    PPI_OFFICIAL_TEST_METHODS and were never right for this check anyway:
    they traded MCAR calibration for MNAR robustness, but this check's
    real-data labeling is MCAR by construction
    (generate_real_twogroup_null_cell's _reveal_labels call), so there was
    no MNAR risk here for a local rectifier to buy anything against. MWU
    (the global rectifier) is now the only midrank correction."""
    if _TWOGROUP_TESTS_OVERRIDE:
        # --twogroup-tests: restrict the two-group check to a named subset, so
        # re-validating ONE test after a fix costs a fraction of the run.
        # Mirrors --omnibus-tests. Binary still drops the rank test.
        want = list(_TWOGROUP_TESTS_OVERRIDE)
        if eval_type == "binary":
            want = [m for m in want if m != MWU.name]
        return want
    base = [TTEST.name, TTEST_WELCH.name]
    return base if eval_type == "binary" else base + [MWU.name]


def _has_standard_test(results: list) -> bool:
    return any(r.test not in _PPI_NONSTANDARD_TESTS for r in results)


def _has_nonstandard_test(results: list) -> bool:
    return any(r.test in _PPI_NONSTANDARD_TESTS for r in results)


def _paired_methods_for(eval_type: str) -> list[str]:
    """Official paired CI methods to check for *eval_type*.

    BAYES_BOOTSTRAP/BOOTSTRAP_T are not part of the official CI-comparison
    set -- see _single_methods_for's matching note. Binary uses
    PPI_BONETT_PRICE (PPI_AUTO_METHOD_TABLE's binary pairwise
    method); non-binary gets PPI_T_INTERVAL and PPI_LOGIT_T
    (PPI_AUTO_METHOD_TABLE's "unbounded"/"bounded_01" pairwise methods --
    both tested, not just the "correct" one per data_kind, since this is
    a validation comparison, not production auto-routing -- matching
    pvalues.py's own _PPI_CI_COMPARISON_TESTS convention). This function
    feeds count-based (PPIResult-style) results only, never
    PPIEffectResult tuples, so unlike _WMT_PAIRED_BIAS_METHODS there is
    no test-name-collision risk between PPI_LOGIT_T here and the
    single-sample check's own PPI_LOGIT_T branch."""
    if eval_type == "binary":
        return [PAIRED_T.name, PPI_BONETT_PRICE.name]
    return [WILCOXON.name, PAIRED_T.name, PPI_T_INTERVAL.name, PPI_LOGIT_T.name]


_TWOGROUP_TESTS_OVERRIDE: list[str] = []
"""Set by --twogroup-tests; empty means the default set."""

_OMNIBUS_TESTS_OVERRIDE: list[str] = []
"""Set by --omnibus-tests; empty means the default set."""


def _omnibus_independent_methods_for(eval_type: str) -> list[str]:
    """Official omnibus (independent-groups) methods to check for *eval_type*.

    anova_ind/kruskal/kruskal_mnar_experimental aren't in pvalues.py's
    _PPI_BINARY_COMPATIBLE_TESTS -- binary's massive ties break these
    rank/variance-based judge-bias models entirely, same restriction as
    the twogroup/paired families.

    KRUSKAL_MNAR_EXPERIMENTAL deliberately NOT included, for the identical
    reason the local-rectifier MWU variants were dropped from the twogroup
    check (see _twogroup_methods_for): it trades some MCAR calibration
    for MNAR robustness, but this check's real-data labeling is MCAR by
    construction, so there's no MNAR risk here for its local rectifier to
    buy anything against.

    KRUSKAL_ROWSUM/KRUSKAL_ROWSUM_LABELED ARE included: unlike the MNAR
    variant they are not a different rectifier competing on the same
    ground, they are the same correction projected onto classical KW's own
    contrasts -- so running them beside KRUSKAL on real data is exactly the
    "what does correcting the real KW cost" comparison, and it costs one
    extra bootstrap per rep."""
    if eval_type == "binary":
        return []
    if _OMNIBUS_TESTS_OVERRIDE:
        # --omnibus-tests: restrict the omnibus-independent checks to a named
        # subset. Every cell runs every listed method, so isolating one test
        # cuts the cost of a targeted validation roughly proportionally --
        # which is the point when re-validating a single method.
        return list(_OMNIBUS_TESTS_OVERRIDE)
    return [ANOVA_IND.name, KRUSKAL.name, KRUSKAL_ROWSUM.name, KRUSKAL_ROWSUM_LABELED.name,
            KRUSKAL_TWOPART.name, KRUSKAL_EIGENGAP.name]


def _omnibus_repeated_methods_for(eval_type: str) -> list[str]:
    # Same binary restriction as _omnibus_independent_methods_for --
    # anova_rep/friedman are equally rank/variance-based.
    if eval_type == "binary":
        return []
    return [ANOVA_REP.name, FRIEDMAN.name]


# ---------------------------------------------------------------------------
# Extra plots for hypothesis_results (twogroup + paired + omnibus): a
# dataset x label_frac heatmap per test, and a judge-alignment-bucketed bar
# chart per test -- real-data analogues of pvalues.py's save_ppi_nlab_grid_
# plot / save_ppi_alignment_sweep_plot, adapted to what real data actually
# varies (there's no synthetic bias_magnitude/label_mechanism/effect_size
# knob here, just which real dataset/label_frac/judges a cell used).
# ---------------------------------------------------------------------------

_PPI_REAL_LABFRAC_N_RE = re.compile(r"\.labfrac=([0-9.]+)\.n=(\d+)")
_PPI_REAL_JUDGES_RE = re.compile(r"\.(?:judge|pair|triple)=(.+)$")


def _parse_real_cell_labfrac_n(name: str) -> tuple[float, int]:
    """`name` is one of run()'s work_items names, e.g.
    "real.arena.labfrac=0.20.n=100.pair=judge_a~judge_b" -- pulls
    (label_frac, n) back out for the heatmap's axes."""
    m = _PPI_REAL_LABFRAC_N_RE.search(name)
    if not m:
        raise ValueError(f"Unrecognized ppi_real cell name: {name!r}")
    return float(m.group(1)), int(m.group(2))


def _parse_real_cell_judges(name: str) -> list[str]:
    """The 1 (single), 2 (twogroup/paired), or 3 (omnibus) judge_model
    names a cell drew on, in the order they were assigned to
    group/condition A/B/C -- everything after the final .judge=/.pair=/
    .triple= marker, split on "~" (see run()'s name-building f-strings)."""
    m = _PPI_REAL_JUDGES_RE.search(name)
    if not m:
        return []
    return m.group(1).split("~")


def save_ppi_real_labfrac_dataset_heatmap(
    *, results: list[PPIResult], alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Heatmap(s) of the PPI-corrected Type-I rate over (dataset, label_frac),
    one small-multiple panel per test -- pooled across every N and judge
    pair/triple that test's cells span, since (unlike pvalues.py's synthetic
    N x N_lab grid) real data has only 3-4 datasets and a handful of label
    fracs to show, so N/judge-pair are collapsed rather than given their own
    axis. Same TwoSlopeNorm diverging colormap centered on alpha as
    save_ppi_nlab_grid_plot, for the same reason: under- vs over-rejection
    should be visually distinct, not just "far from 0".

    `nonstandard` selects _PPI_NONSTANDARD_TESTS (bayes_bootstrap,
    bootstrap_t, mj_floor) instead of the hypothesis-test majority --
    same split save_ppi_typeI_plot already draws between p-value tests and
    CI-based methods, kept here too rather than mixing both families into
    one grid of panels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = [
        m.name for m in PPI_TEST_METHODS
        if any(r.test == m.name for r in results) and (m.name in _PPI_NONSTANDARD_TESTS) == nonstandard
    ]
    if not tests:
        raise ValueError(f"No {'nonstandard' if nonstandard else 'standard'} test results to plot.")
    datasets = sorted({r.tag.removeprefix("real_") for r in results})
    label_fracs = sorted({_parse_real_cell_labfrac_n(r.name)[0] for r in results})

    n_cols = min(4, len(tests))
    n_rows = -(-len(tests) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for idx, test in enumerate(tests):
        ax = axes[idx // n_cols][idx % n_cols]
        rejects = np.zeros((len(datasets), len(label_fracs)))
        totals = np.zeros((len(datasets), len(label_fracs)))
        for r in results:
            if r.test != test:
                continue
            ds = r.tag.removeprefix("real_")
            lf, _n = _parse_real_cell_labfrac_n(r.name)
            i, j = datasets.index(ds), label_fracs.index(lf)
            rejects[i, j] += r.corrected_rejects
            totals[i, j] += r.n_reps
        grid = np.where(totals > 0, np.divide(rejects, totals, out=np.full_like(rejects, np.nan), where=totals > 0), np.nan)

        vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
        for i in range(len(datasets)):
            for j in range(len(label_fracs)):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                        bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                    )
        ax.set_xticks(range(len(label_fracs)))
        ax.set_xticklabels([f"{lf:.2f}" for lf in label_fracs], fontsize=8)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_xlabel("label_frac", fontsize=8)
        ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(len(tests), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    family = "CI-Based Methods" if nonstandard else "Hypothesis Tests"
    fig.suptitle(
        f"PPI-Corrected Type-I Rate over Dataset x Label Fraction, {family} ({_alpha_label(alpha)})",
        y=1.02, fontsize=12,
    )
    fig.text(0.5, -0.01, "Pooled across N and judge pair/triple", ha="center", fontsize=8, color="#555555")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_real_alignment_plot(
    *, results: list[PPIResult], alignment_by_dataset_judge: dict[tuple[str, str], float],
    alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Bar chart of uncorrected vs. PPI-corrected Type-I rate bucketed by
    realized judge-human alignment, one panel per test -- the real-data
    counterpart to save_ppi_alignment_sweep_plot, but using DIRECTLY
    measured alignment (np.corrcoef(judge_scores, human_label) on each
    corpus's already-[0,1]-rescaled arrays -- see RealJudgeBiasCorpus) for
    each cell's actual judge(s), not a simulated sweep over synthetic noise
    levels. A cell's alignment is the mean of its 1-3 involved judges'
    individual alignment (single/twogroup/paired/omnibus cells draw on 1/2/
    2/3 judges respectively -- see _parse_real_cell_judges). No no_bias/
    bias_present regime split like the synthetic version: real judges are
    always whatever bias they actually have, there's no "off" switch to
    contrast against.

    `nonstandard` selects _PPI_NONSTANDARD_TESTS instead of the hypothesis-
    test majority, same split as save_ppi_real_labfrac_dataset_heatmap."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = [
        m.name for m in PPI_TEST_METHODS
        if any(r.test == m.name for r in results) and (m.name in _PPI_NONSTANDARD_TESTS) == nonstandard
    ]
    if not tests:
        raise ValueError(f"No {'nonstandard' if nonstandard else 'standard'} test results to plot.")

    def _cell_alignment(r: PPIResult) -> float | None:
        ds = r.tag.removeprefix("real_")
        judges = _parse_real_cell_judges(r.name)
        vals = [alignment_by_dataset_judge[(ds, jm)] for jm in judges if (ds, jm) in alignment_by_dataset_judge]
        return float(np.mean(vals)) if vals else None

    cell_align = {id(r): _cell_alignment(r) for r in results}
    buckets = sorted({_alignment_bucket(_metric_pct(a)) for a in cell_align.values() if a is not None})
    if not buckets:
        raise ValueError("No alignment data available for any ppi_real hypothesis cell.")

    n_cols = min(4, len(tests))
    n_rows = -(-len(tests) // n_cols)
    panel_w = max(2.2, 0.85 * len(buckets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, 4.0 * n_rows), squeeze=False)
    bar_width, gap = 0.35, 0.25
    for idx, test in enumerate(tests):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6, zorder=1, label="nominal α" if idx == 0 else None)
        xticks, xticklabels = [], []
        for bi, (lo, label) in enumerate(buckets):
            cells = [r for r in results if r.test == test and cell_align.get(id(r)) is not None
                     and _alignment_bucket(_metric_pct(cell_align[id(r)])) == (lo, label)]
            n_reps_tot = sum(c.n_reps for c in cells)
            for ai, (arm, color, ec) in enumerate([("uncorrected", "#999999", "none"), ("corrected", get_method_color(test), "#333333")]):
                x = bi * (2 * bar_width + gap) + ai * bar_width
                if n_reps_tot == 0:
                    continue
                rejects_tot = sum((c.uncorrected_rejects if arm == "uncorrected" else c.corrected_rejects) for c in cells)
                rate = rejects_tot / n_reps_tot
                lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
                ax.bar(x, rate, width=bar_width, color=color, edgecolor=ec, linewidth=1.0, zorder=2,
                       label=arm if (idx == 0 and bi == 0) else None)
                ax.errorbar(x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                             fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4)
            mid = bi * (2 * bar_width + gap) + bar_width / 2
            xticks.append(mid)
            xticklabels.append(f"{label}\n(n={len(cells)})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlim(-0.3, (2 * bar_width + gap) * len(buckets) - gap + 0.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
        if idx % n_cols == 0:
            ax.set_ylabel("Type-I rate", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
    for idx in range(len(tests), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    family = "CI-Based Methods" if nonstandard else "Hypothesis Tests"
    fig.suptitle(f"PPI-Corrected Type-I Rate by Judge-Human Alignment, {family} ({_alpha_label(alpha)})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Power (positive-control) report/plot -- see the _run_real_*_power_cell
# functions above. Deliberately NOT reusing pvalues.py's print_ppi_report/
# save_results_artifacts_ppi for this bucket even though it's the same
# PPIResult type as hypothesis_results: that report's 2-sigma/3-sigma
# inflation flags and Holm-Bonferroni miscalibration flag all assume a HIGH
# rejection rate is bad (Type-I error) -- exactly backwards for a power
# check, where a high rejection rate is the desired outcome. This section's
# report/plot instead just states corrected vs. uncorrected power per test,
# no miscalibration framing.
# ---------------------------------------------------------------------------


def print_ppi_real_power_report(results: list[PPIResult], alpha: float) -> None:
    """Pooled corrected/uncorrected power (rejection rate under a genuine,
    exactly-known real effect) per test -- the positive-control complement
    to print_ppi_report's null-check table."""
    if not results:
        print("\n  (no ppi_real power results)")
        return
    print(f"\n{'=' * 90}\n  PPI_REAL -- POWER (POSITIVE CONTROL: genuine, exactly-known real effect)\n"
          f"  {len(results)} cells; nominal {_alpha_label(alpha)}\n{'=' * 90}\n")
    print(f"    {'test':<22} {'n_cells':>8} {'uncorrected power':>18} {'corrected power':>16}")
    tests = [m.name for m in PPI_TEST_METHODS if any(r.test == m.name for r in results)]
    for t in tests:
        t_rows = [r for r in results if r.test == t]
        n_reps_tot = sum(r.n_reps for r in t_rows)
        u = sum(r.uncorrected_rejects for r in t_rows) / n_reps_tot if n_reps_tot else float("nan")
        c = sum(r.corrected_rejects for r in t_rows) / n_reps_tot if n_reps_tot else float("nan")
        print(f"    {_pretty_test(t):<22} {len(t_rows):>8d} {u:>18.3f} {c:>16.3f}")
    print()


def save_results_artifacts_ppi_real_power(
    *, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str,
) -> list[str]:
    """CSV + log artifacts for the power check -- same CSV shape as
    pvalues.py's save_results_artifacts_ppi, but the log uses
    print_ppi_real_power_report (see this section's header comment for why
    print_ppi_report itself isn't reused here)."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_power_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "n", "test", "n_reps", "corrected_rejects", "uncorrected_rejects",
            "corrected_power", "uncorrected_power",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.n, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_power_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_real_power_report(results, alpha=alpha)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_real_power_plot(
    *, results: list[PPIResult], alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Bar chart of uncorrected vs. PPI-corrected power (rejection rate
    under a genuine, exactly-known real effect -- see generate_real_
    twogroup_power_cell / generate_real_omnibus_independent_power_cell /
    generate_real_wmt_paired_bias_cell), one bar pair per test -- the
    positive-control complement to save_ppi_typeI_plot's null-check view:
    that plot answers "does correction avoid FALSE positives," this one
    answers "does correction still find TRUE positives." Same standard/
    nonstandard split, same Wilson-CI-on-pooled-rate convention as this
    module's other bar-chart plots (save_ppi_real_alignment_plot etc)."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No ppi_real power results to plot.")
    tests = [
        m.name for m in PPI_TEST_METHODS
        if any(r.test == m.name for r in results) and (m.name in _PPI_NONSTANDARD_TESTS) == nonstandard
    ]
    if not tests:
        raise ValueError(f"No {'nonstandard' if nonstandard else 'standard'} power results to plot.")

    fig, ax = plt.subplots(figsize=(1.6 * len(tests) + 2.0, 5.0))
    bar_width, gap = 0.35, 0.3
    for ti, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t]
        n_reps_tot = sum(r.n_reps for r in t_rows)
        for ai, (arm, color, ec) in enumerate([("uncorrected", "#999999", "none"), ("corrected", get_method_color(t), "#333333")]):
            x = ti * (2 * bar_width + gap) + ai * bar_width
            if n_reps_tot == 0:
                continue
            rejects_tot = sum((r.uncorrected_rejects if arm == "uncorrected" else r.corrected_rejects) for r in t_rows)
            rate = rejects_tot / n_reps_tot
            lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
            ax.bar(x, rate, width=bar_width, color=color, edgecolor=ec, linewidth=1.0, zorder=2,
                   label=arm if ti == 0 else None)
            ax.errorbar(x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                         fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4)
    ax.set_xticks([ti * (2 * bar_width + gap) + bar_width / 2 for ti in range(len(tests))])
    ax.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Power (correct-rejection rate)")
    family = "CI-Based Methods" if nonstandard else "Hypothesis Tests"
    ax.set_title(f"PPI-Corrected Power (Positive Control), {family} ({_alpha_label(alpha)})", fontsize=12)
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CI coverage/width vs. label_frac (effect_results: single-sample +
# wmt_paired_bias) -- save_ppi_effect_plot's per-test jittered scatter has
# no label_frac axis at all (every cell's label_frac collapses into one
# column per test); this answers "how many labels does the coverage/width
# tradeoff actually need" directly.
# ---------------------------------------------------------------------------


def save_ppi_real_coverage_width_by_labfrac_plot(
    *, results: list[PPIEffectResult], alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Top row: CI coverage vs. label_frac, one line per test, pooled
    (Wilson CI on the pooled coverage proportion) across every (dataset, n,
    judge) cell sharing a (test, label_frac) -- each cell's own `coverage`
    is already a proportion over its n_samples reps, so pooling multiple
    cells at the same label_frac treats them as one larger Bernoulli sample
    (exact only if every contributing cell were identically calibrated,
    same simplification this module's other pooled-rate plots make).
    Bottom row: mean CI width, weighted-averaged the same way, with error
    bars showing the weighted standard error across contributing cells
    (not a Wilson interval -- width isn't a proportion)."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No ppi_real effect results to plot.")
    tests = [
        m.name for m in PPI_TEST_METHODS
        if any(r.test == m.name for r in results) and (m.name in _PPI_NONSTANDARD_TESTS) == nonstandard
    ]
    if not tests:
        raise ValueError(f"No {'nonstandard' if nonstandard else 'standard'} effect results to plot.")
    label_fracs = sorted({_parse_real_cell_labfrac_n(r.name)[0] for r in results})

    fig, (ax_cov, ax_wid) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    target_cov = 1.0 - alpha
    for t in tests:
        color = get_method_color(t)
        cov_xs, cov_ys, cov_lo, cov_hi = [], [], [], []
        wid_xs, wid_ys, wid_err = [], [], []
        for lf in label_fracs:
            cells = [
                r for r in results if r.test == t and _parse_real_cell_labfrac_n(r.name)[0] == lf
                and r.n_samples > 0 and np.isfinite(r.coverage)
            ]
            if cells:
                n_tot = sum(c.n_samples for c in cells)
                successes = sum(int(round(c.coverage * c.n_samples)) for c in cells)
                cov_rate = successes / n_tot
                lo_ci, hi_ci = _ppi_wilson_interval(successes, n_tot)
                cov_xs.append(lf)
                cov_ys.append(cov_rate)
                cov_lo.append(max(0.0, cov_rate - lo_ci))
                cov_hi.append(max(0.0, hi_ci - cov_rate))

            width_cells = [r for r in results if r.test == t and _parse_real_cell_labfrac_n(r.name)[0] == lf
                            and r.n_samples > 0 and np.isfinite(r.mean_ci_width)]
            if width_cells:
                widths = np.array([c.mean_ci_width for c in width_cells])
                weights = np.array([c.n_samples for c in width_cells], dtype=float)
                w_mean = float(np.average(widths, weights=weights))
                w_se = (float(np.sqrt(np.average((widths - w_mean) ** 2, weights=weights)) / np.sqrt(len(widths)))
                        if len(widths) > 1 else 0.0)
                wid_xs.append(lf)
                wid_ys.append(w_mean)
                wid_err.append(w_se)

        if cov_xs:
            ax_cov.errorbar(cov_xs, cov_ys, yerr=[cov_lo, cov_hi], color=color, marker="o", ms=5,
                             lw=1.5, capsize=3, label=_pretty_test(t))
        if wid_xs:
            ax_wid.errorbar(wid_xs, wid_ys, yerr=wid_err, color=color, marker="o", ms=5, lw=1.5, capsize=3)

    ax_cov.axhline(target_cov, color="black", ls="--", lw=1.1, label=f"Target = {target_cov:.2f}")
    ax_cov.set_ylabel("CI coverage")
    ax_cov.set_ylim(0.0, 1.02)
    ax_cov.set_title("CI Coverage vs. Label Fraction")
    ax_cov.grid(alpha=0.25, lw=0.8)
    ax_cov.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)

    ax_wid.set_ylabel("Mean CI width")
    ax_wid.set_xlabel("label_frac")
    ax_wid.set_title("CI Width vs. Label Fraction")
    ax_wid.grid(alpha=0.25, lw=0.8)

    family = "CI-Based Methods" if nonstandard else "Standard Tests"
    fig.suptitle(f"PPI-Corrected CI Coverage/Width by Label Fraction, {family} ({_alpha_label(alpha)})", y=1.0, fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-judge leaderboard scatter -- the raw-resolution complement to
# save_ppi_real_alignment_plot's 10-point bucketed bars.
# ---------------------------------------------------------------------------


def save_ppi_real_judge_leaderboard_plot(
    *, results: list[PPIResult], alignment_by_dataset_judge: dict[tuple[str, str], float],
    alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Scatter of (judge-human alignment, pooled PPI-corrected Type-I rate)
    at PER-JUDGE resolution, one point per (dataset, judge_model) per test
    -- bucketing (save_ppi_real_alignment_plot) pools potentially very
    different judges into the same bar, which can hide a smooth alignment-
    calibration relationship or a sharp cliff alike. A cell can involve up
    to 3 judges (twogroup/paired cells: 2, omnibus cells: 3 -- see
    _parse_real_cell_judges), so a (dataset, judge_model)'s point pools
    EVERY cell that judge appears in for that test, regardless of which
    other judge(s) shared the cell. Point AREA encodes total pooled n_reps
    (more area = more replicates backing that point); color encodes
    dataset."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = [
        m.name for m in PPI_TEST_METHODS
        if any(r.test == m.name for r in results) and (m.name in _PPI_NONSTANDARD_TESTS) == nonstandard
    ]
    if not tests:
        raise ValueError(f"No {'nonstandard' if nonstandard else 'standard'} test results to plot.")
    datasets = sorted({r.tag.removeprefix("real_") for r in results})
    cmap = plt.get_cmap("tab10")
    ds_color = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    n_cols = min(4, len(tests))
    n_rows = -(-len(tests) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    for idx, test in enumerate(tests):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6, zorder=1)
        agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for r in results:
            if r.test != test:
                continue
            ds = r.tag.removeprefix("real_")
            for jm in _parse_real_cell_judges(r.name):
                if (ds, jm) not in alignment_by_dataset_judge:
                    continue
                agg[(ds, jm)][0] += r.corrected_rejects
                agg[(ds, jm)][1] += r.n_reps
        for (ds, jm), (rejects, n_tot) in agg.items():
            if n_tot == 0:
                continue
            x = alignment_by_dataset_judge[(ds, jm)]
            y = rejects / n_tot
            ax.scatter(x, y, s=max(12.0, 4.0 * np.sqrt(n_tot)), color=ds_color[ds], alpha=0.75,
                       edgecolors="#333333", linewidths=0.5, zorder=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
        if idx % n_cols == 0:
            ax.set_ylabel("Corrected Type-I rate", fontsize=9)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Judge-human alignment (Pearson r)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.8, zorder=0)
    for idx in range(len(tests), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ds_color[ds],
                   markeredgecolor="#333333", markersize=8, label=ds)
        for ds in datasets
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True, title="Dataset")
    family = "CI-Based Methods" if nonstandard else "Hypothesis Tests"
    fig.suptitle(f"Per-Judge Type-I Rate vs. Alignment, {family} ({_alpha_label(alpha)})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# N x label_frac heatmap for a small set of tests -- save_ppi_real_
# labfrac_dataset_heatmap pools N away entirely (default_size_grid derives
# N as a FRACTION of each dataset's own corpus_size, so absolute N values
# aren't comparable/poolable across datasets the way label_frac/dataset
# are). Restricted to a small `tests` subset by default (kruskal, friedman
# -- the two tests most sensitive to N/label_frac on real data)
# rather than every test, since a full (dataset x test) grid would be
# #datasets x #tests panels.
# ---------------------------------------------------------------------------


def save_ppi_real_n_labfrac_heatmap(
    *, results: list[PPIResult], alpha: float, out_path: str,
    tests: tuple[str, ...] = (KRUSKAL.name, FRIEDMAN.name),
) -> str:
    """One row per dataset, one column per test in `tests`, grid = (N,
    label_frac) -- pooled across judge triple/pair only (N and label_frac
    both stay as real axes here, unlike save_ppi_real_labfrac_dataset_
    heatmap, which pools N away). Same TwoSlopeNorm-centered-on-alpha
    diverging colormap convention as this module's other heatmap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = tuple(t for t in tests if any(r.test == t for r in results))
    if not tests:
        raise ValueError("None of the requested tests have results to plot.")
    datasets = sorted({r.tag.removeprefix("real_") for r in results if r.test in tests})
    if not datasets:
        raise ValueError("No datasets have results for the requested tests.")

    n_rows, n_cols = len(datasets), len(tests)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for ri, ds in enumerate(datasets):
        ds_results = [r for r in results if r.tag.removeprefix("real_") == ds]
        n_values = sorted({_parse_real_cell_labfrac_n(r.name)[1] for r in ds_results})
        label_fracs = sorted({_parse_real_cell_labfrac_n(r.name)[0] for r in ds_results})
        for ci, test in enumerate(tests):
            ax = axes[ri][ci]
            rejects = np.zeros((len(n_values), len(label_fracs)))
            totals = np.zeros((len(n_values), len(label_fracs)))
            for r in ds_results:
                if r.test != test:
                    continue
                lf, n = _parse_real_cell_labfrac_n(r.name)
                i, j = n_values.index(n), label_fracs.index(lf)
                rejects[i, j] += r.corrected_rejects
                totals[i, j] += r.n_reps
            grid = np.where(
                totals > 0, np.divide(rejects, totals, out=np.full_like(rejects, np.nan), where=totals > 0), np.nan,
            )
            vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
            im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
            for i in range(len(n_values)):
                for j in range(len(label_fracs)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                        )
            ax.set_xticks(range(len(label_fracs)))
            ax.set_xticklabels([f"{lf:.2f}" for lf in label_fracs], fontsize=8)
            ax.set_yticks(range(len(n_values)))
            ax.set_yticklabels([str(n) for n in n_values], fontsize=8)
            ax.set_xlabel("label_frac", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{ds}\nN", fontsize=8)
            ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PPI-Corrected Type-I Rate over N x Label Fraction ({_alpha_label(alpha)})", y=1.02, fontsize=12)
    fig.text(0.5, -0.01, "Pooled across judge pair/triple; one row per dataset", ha="center", fontsize=8, color="#555555")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help="Directory containing judge_bias_<dataset>.csv (from collect_judge_bias_data.py).")
    parser.add_argument("--datasets", choices=list(REAL_JUDGE_BIAS_DATASETS), nargs="+", default=None,
                         help="Which real datasets to check (default: all three; a dataset with no "
                              "judge_score data yet is skipped with a warning, not a hard failure).")
    parser.add_argument("--judge-models", nargs="+", default=None,
                         help="Which judge models to use (default: auto-discover every judge model "
                              "collected for each dataset, then apply --min-judge-coverage). Passing "
                              "this explicitly is trusted as-is, with NO coverage filtering -- only "
                              "items scored by EVERY selected judge are used either way.")
    parser.add_argument("--min-judge-coverage", type=float, default=0.9,
                         help="Auto-discovery only (ignored if --judge-models is given explicitly): "
                              "drop any judge whose distinct-item coverage of the dataset is below this "
                              "fraction (default 0.9) before aligning. Alignment takes the INTERSECTION "
                              "of items across every selected judge, so one incompletely-collected judge "
                              "(e.g. a --limit'd or still-running collect-judge-scores) otherwise drags "
                              "the usable corpus size down for every OTHER judge too, not just itself.")
    parser.add_argument("--max-pairs", type=int, default=None,
                         help="Cap on unique judge PAIRS checked by the twogroup/paired-null tests "
                              "(default: all C(k, 2) pairs for k judge models -- e.g. 28 for 8 judges. "
                              "Caps by random subset if k is large enough that C(k, 2) would be "
                              "impractically slow.")
    parser.add_argument("--max-triples", type=int, default=None,
                         help="Cap on unique judge TRIPLES checked by the omnibus-independent/omnibus-"
                              "repeated tests (default: all C(k, 3) triples for k judge models -- e.g. "
                              "56 for 8 judges. Caps by random subset if k is large enough that C(k, 3) "
                              "would be impractically slow.")
    parser.add_argument("--label-fracs", type=float, nargs="+", default=DEFAULT_LABEL_FRACS)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                         help="Sample sizes per rep (default: 3 sizes bounded by each dataset's actual corpus size).")
    parser.add_argument("--twogroup-tests", nargs="+", default=None,
                        help="Restrict the two-group Type-I check to these method "
                             "names (e.g. 'mwu'). Override, not addition, so a "
                             "default run is unchanged.")
    parser.add_argument("--omnibus-tests", nargs="+", default=None,
                         help="Restrict the omnibus-independent null/power checks to these "
                              "method names (e.g. 'kruskal kruskal_influence'). Cuts runtime "
                              "roughly in proportion, for re-validating one method without "
                              "paying for the others.")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--ppi-n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--progress", choices=["bar", "cell", "off"], default="bar")
    parser.add_argument("--save-results", choices=["save", "none"], default="save")
    parser.add_argument("--plots", choices=["save", "none"], default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--no-single-check", action="store_true", help="Skip the single-sample bias/coverage check.")
    parser.add_argument("--no-twogroup-check", action="store_true", help="Skip the two-group Type-I null check.")
    parser.add_argument("--no-paired-check", action="store_true", help="Skip the cross-judge paired Type-I null check.")
    parser.add_argument("--no-omnibus-independent-check", action="store_true",
                         help="Skip the 3-group cross-judge omnibus Type-I null check (anova_ind/kruskal).")
    parser.add_argument("--no-omnibus-repeated-check", action="store_true",
                         help="Skip the 3-condition cross-judge omnibus Type-I null check "
                              "(anova_rep/friedman).")
    parser.add_argument("--no-wmt-paired-check", action="store_true",
                         help="Skip the within-item (wmt_da_paired) bias/coverage check. Runs by default "
                              "IF judge_bias_wmt_da_paired.csv has been collected (see "
                              "collect_judge_bias_data.py --types continuous_paired); skipped with a "
                              "warning, not a hard failure, if it hasn't.")
    parser.add_argument("--no-power-check", action="store_true",
                         help="Skip the positive-control power checks (twogroup/omnibus-independent rank-"
                              "split cross-judge, plus within-item wmt_da_paired power if that data is "
                              "collected) -- does PPI correction still detect a genuine, exactly-known real "
                              "effect, not just avoid false positives on a null.")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential). Parallelizes "
                              "across single/twogroup/paired/omnibus cells (all corpora, label_fracs, "
                              "sizes, judge models/pairs/triples flattened into one work queue) using a "
                              "multiprocessing.Pool "
                              "with the fork start method, same as pvalues.py's run_*_simulation functions. "
                              "Cell seeds are fixed up front from --seed regardless of --workers, so "
                              "results are identical either way -- only completion order (and hence "
                              "progress-bar interleaving) differs.")


def official_args(base_seed: int = 46) -> argparse.Namespace:
    """Canonical official-test preset for real judge-bias data. Mostly mirrors
    add_arguments' own CLI defaults (reps=200, ppi_n_boot=2000, alpha=0.05,
    every --no-*-check flag off so all checks run) -- the one deliberate
    override is latex=True, so an official run always emits the LaTeX
    tables this case feeds into the paper, without requiring --latex on
    the command line."""
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9,
        max_pairs=None, max_triples=None,
        label_fracs=list(DEFAULT_LABEL_FRACS), sizes=None, reps=200, ppi_n_boot=2000,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False,
        no_omnibus_independent_check=False, no_omnibus_repeated_check=False, no_wmt_paired_check=False,
        no_power_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 46) -> list[tuple[str, argparse.Namespace]]:
    return [(
        "real judge-bias data (single-sample + two-group + paired null + omnibus null + "
        "within-item paired bias/coverage + positive-control power)",
        official_args(base_seed),
    )]


def quick_args(base_seed: int = 47, data_source: str = "synthetic") -> argparse.Namespace:
    """`data_source` accepted only for --quick-test's uniform
    (module.quick_args(), module.quick_args(data_source="real")) calling
    convention -- this case has no synthetic variant of its own (it's
    always real data), so both calls run the identical fast preset."""
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9,
        max_pairs=10, max_triples=10,
        label_fracs=[0.20], sizes=None, reps=5, ppi_n_boot=200,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False,
        no_omnibus_independent_check=False, no_omnibus_repeated_check=False, no_wmt_paired_check=False,
        no_power_check=False,
        latex=True, workers=1,
    )


def run(args: argparse.Namespace) -> CaseResult:
    """Case entry point. Loads every requested real judge-bias corpus (plus
    the within-item WMT paired corpus, unless disabled), flattens the full
    (corpus x label_frac x size x judge_model/pair x check_type) grid into
    one work list, and runs each cell's single-sample, two-group,
    cross-judge-paired, omnibus (independent/repeated), within-item-paired,
    and positive-control-power checks (any subset of which can be skipped
    via the --no-*-check flags). Prints and optionally saves a report,
    LaTeX tables, and diagnostic plots for each result family, and returns
    a CaseResult summarizing aggregate power."""
    t0 = time.time()
    global _TWOGROUP_TESTS_OVERRIDE, _OMNIBUS_TESTS_OVERRIDE
    _TWOGROUP_TESTS_OVERRIDE = list(getattr(args, "twogroup_tests", None) or [])
    if _TWOGROUP_TESTS_OVERRIDE:
        print(f"  two-group checks restricted to: {', '.join(_TWOGROUP_TESTS_OVERRIDE)}")
    _OMNIBUS_TESTS_OVERRIDE = list(getattr(args, "omnibus_tests", None) or [])
    if _OMNIBUS_TESTS_OVERRIDE:
        _known = {m.name for m in PPI_TEST_METHODS}
        _bad = [t for t in _OMNIBUS_TESTS_OVERRIDE if t not in _known]
        if _bad:
            raise ValueError(f"--omnibus-tests: unknown method name(s) {_bad}")
        print(f"  omnibus-independent checks restricted to: "
              f"{', '.join(_OMNIBUS_TESTS_OVERRIDE)}")
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_paths: list[str] = []
        key_metrics: dict = {}

        requested = args.datasets or list(REAL_JUDGE_BIAS_DATASETS)
        corpora: list[RealJudgeBiasCorpus] = []
        for ds in requested:
            try:
                corpus = load_real_judge_bias_corpus(
                    ds, data_dir=args.data_dir, judge_models=args.judge_models,
                    min_coverage=getattr(args, "min_judge_coverage", 0.9),
                )
                corpora.append(corpus)
                print(f"  Loaded {ds}: N={corpus.corpus_size}, judge_models={corpus.judge_models}, "
                      f"corpus_mean={corpus.corpus_mean:.4f}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping {ds}: {e}")

        # Directly-measured judge-human alignment per (dataset, judge_model)
        # -- corpus.judge_scores/human_label are already rescaled to [0,1]
        # for every eval_type (see RealJudgeBiasCorpus), so a plain Pearson
        # r is comparable across binary/continuous/likert here, unlike the
        # synthetic sweep's per-eval_type metric choice (_ALIGNMENT_VIEWS).
        # Feeds save_ppi_real_alignment_plot below.
        alignment_by_dataset_judge: dict[tuple[str, str], float] = {
            (corpus.dataset, jm): float(np.corrcoef(scores, corpus.human_label)[0, 1])
            for corpus in corpora for jm, scores in corpus.judge_scores.items()
        }

        wmt_paired_corpus: RealWithinItemPairedCorpus | None = None
        if not getattr(args, "no_wmt_paired_check", False):
            try:
                wmt_paired_corpus = load_real_wmt_paired_corpus(
                    data_dir=args.data_dir, judge_models=args.judge_models,
                    min_coverage=getattr(args, "min_judge_coverage", 0.9),
                )
                print(f"  Loaded wmt_da_paired: N={wmt_paired_corpus.corpus_size} pairs, "
                      f"judge_models={wmt_paired_corpus.judge_models}, "
                      f"true_paired_mean={wmt_paired_corpus.true_paired_mean:.4f}, "
                      f"true_paired_median={wmt_paired_corpus.true_paired_median:.4f}, "
                      f"true_paired_walsh_midrank_theta={wmt_paired_corpus.true_paired_walsh_midrank_theta:.4f}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping wmt_da_paired (within-item paired check): {e}")

        if not corpora and wmt_paired_corpus is None:
            raise ValueError(
                "No real judge-bias datasets with collected scores found under "
                f"{args.data_dir!r}. Run simulations/collect_judge_bias_data.py's "
                "collect-data then collect-judge-scores first."
            )

        effect_results: list[PPIEffectResult] = []
        twogroup_results: list[PPIResult] = []
        paired_results: list[PPIResult] = []
        omnibus_results: list[PPIResult] = []
        power_results: list[PPIResult] = []
        seed_counter = args.seed

        # Flatten every (corpus, label_frac, size, judge_model/pair, check
        # type) cell into one work list up front, in the SAME nested order
        # the old inline loop used to run them in, so seed_counter assigns
        # the identical seed to the identical cell regardless of --workers
        # -- results are therefore reproducible for a given --seed whether
        # this runs serially or in parallel; only completion order (and
        # hence progress-bar interleaving) can differ.
        work_items: list[tuple] = []

        for corpus_idx, corpus in enumerate(corpora):
            sizes = args.sizes or default_size_grid(corpus.corpus_size)
            single_methods = _single_methods_for(corpus.eval_type)
            twogroup_methods = _twogroup_methods_for(corpus.eval_type)
            paired_methods = _paired_methods_for(corpus.eval_type)
            omnibus_ind_methods = _omnibus_independent_methods_for(corpus.eval_type)
            omnibus_rep_methods = _omnibus_repeated_methods_for(corpus.eval_type)
            pairs = all_judge_pairs(corpus, max_pairs=args.max_pairs, rng=np.random.default_rng(args.seed))
            triples = all_judge_triples(corpus, max_triples=args.max_triples, rng=np.random.default_rng(args.seed))

            for label_frac in args.label_fracs:
                for n in sizes:
                    if not args.no_single_check:
                        for judge_model in corpus.judge_models:
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.judge={judge_model}"
                            seed_counter += 1
                            work_items.append((
                                "single", corpus_idx, name, corpus.dataset, single_methods,
                                n, label_frac, judge_model, args.reps, args.ppi_n_boot, seed_counter,
                            ))

                    # twogroup/paired both need judge PAIRS (cross-judge --
                    # see generate_real_twogroup_null_cell's docstring for
                    # why twogroup is cross-judge too, not one judge reading
                    # both random-split halves), so both loop over `pairs`.
                    if not args.no_twogroup_check or not args.no_paired_check:
                        for judge_a, judge_b in pairs:
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.pair={judge_a}~{judge_b}"

                            if not args.no_twogroup_check:
                                seed_counter += 1
                                work_items.append((
                                    "twogroup", corpus_idx, name, corpus.dataset, twogroup_methods,
                                    n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                            if not args.no_paired_check:
                                seed_counter += 1
                                work_items.append((
                                    "paired", corpus_idx, name, corpus.dataset, paired_methods,
                                    n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                            # Binary-only bias/coverage sibling of the check
                            # above: MJ_FLOOR is a genuine point-estimate/CI
                            # construction, but _run_real_paired_cell only
                            # ever reports corrected/uncorrected rejection
                            # COUNTS (a Type-I check), so Tango had no path
                            # into the bias/coverage view even on a binary
                            # corpus like arena that has exactly the paired
                            # binary data it needs. Reuses the SAME exact
                            # null (generate_real_paired_null_cell) as the
                            # Type-I check above -- see
                            # _run_real_paired_bias_cell.
                            if not args.no_paired_check and corpus.eval_type == "binary":
                                seed_counter += 1
                                work_items.append((
                                    "paired_bias", corpus_idx, name, corpus.dataset, [PPI_BONETT_PRICE.name],
                                    n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                    # omnibus-independent/omnibus-repeated both need judge
                    # TRIPLES (cross-judge, same reasoning as twogroup/paired
                    # above, just extended to 3 groups/conditions), so both
                    # loop over `triples`.
                    if not args.no_omnibus_independent_check or not args.no_omnibus_repeated_check:
                        for judge_a, judge_b, judge_c in triples:
                            name = (f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}"
                                    f".triple={judge_a}~{judge_b}~{judge_c}")

                            if not args.no_omnibus_independent_check:
                                seed_counter += 1
                                work_items.append((
                                    "omnibus_independent", corpus_idx, name, corpus.dataset, omnibus_ind_methods,
                                    n, label_frac, (judge_a, judge_b, judge_c), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                            if not args.no_omnibus_repeated_check:
                                seed_counter += 1
                                work_items.append((
                                    "omnibus_repeated", corpus_idx, name, corpus.dataset, omnibus_rep_methods,
                                    n, label_frac, (judge_a, judge_b, judge_c), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                    # Positive-control power checks: same pairs/triples as
                    # the null checks above, just a rank-split (not random-
                    # split) cell generator -- see generate_real_twogroup_
                    # power_cell / generate_real_omnibus_independent_power_
                    # cell's docstrings. twogroup_methods/omnibus_ind_methods
                    # reused as-is (identical test battery to the null
                    # checks, just exercised against a genuine effect).
                    if not getattr(args, "no_power_check", False):
                        for judge_a, judge_b in pairs:
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.pair={judge_a}~{judge_b}"
                            seed_counter += 1
                            work_items.append((
                                "twogroup_power", corpus_idx, name, corpus.dataset, twogroup_methods,
                                n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                            ))
                        for judge_a, judge_b, judge_c in triples:
                            name = (f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}"
                                    f".triple={judge_a}~{judge_b}~{judge_c}")
                            seed_counter += 1
                            work_items.append((
                                "omnibus_independent_power", corpus_idx, name, corpus.dataset, omnibus_ind_methods,
                                n, label_frac, (judge_a, judge_b, judge_c), args.reps, args.ppi_n_boot, seed_counter,
                            ))

        if wmt_paired_corpus is not None:
            _REAL_WMT_PAIRED_CORPORA.append(wmt_paired_corpus)
            wmt_paired_idx = 0
            sizes = args.sizes or default_size_grid(wmt_paired_corpus.corpus_size)
            for label_frac in args.label_fracs:
                for n in sizes:
                    for judge_model in wmt_paired_corpus.judge_models:
                        seed_counter += 1
                        name = f"real.{wmt_paired_corpus.dataset}.labfrac={label_frac:.2f}.n={n}.judge={judge_model}"
                        work_items.append((
                            "wmt_paired_bias", wmt_paired_idx, name, wmt_paired_corpus.dataset,
                            _WMT_PAIRED_BIAS_METHODS, n, label_frac, judge_model,
                            args.reps, args.ppi_n_boot, seed_counter,
                        ))
                        if not getattr(args, "no_power_check", False):
                            seed_counter += 1
                            work_items.append((
                                "wmt_paired_power", wmt_paired_idx, name, wmt_paired_corpus.dataset,
                                _WMT_PAIRED_POWER_METHODS, n, label_frac, judge_model,
                                args.reps, args.ppi_n_boot, seed_counter,
                            ))

        def _consume(result: dict) -> None:
            """Route one worker cell's result dict into the matching
            accumulator by `result["check_type"]`: "single" builds a
            PPIEffectResult per test against the corpus mean; "wmt_paired_bias"
            does the same but against a per-test null (the midrank-sign
            quantile for wilcoxon, the mean otherwise); "paired_bias" does the
            same against an exact null of 0.0 (same items, two judges); every
            other check_type (twogroup/paired/omnibus_*/*_power) instead
            appends one PPIResult per test into the twogroup/paired/omnibus/
            power bucket its check_type selects."""
            ct = result["check_type"]
            if ct == "single":
                for t, samples in result["samples_by_test"].items():
                    bias_mean, z, coverage, mean_width, n_ok = _effect_cell_stats(samples, result["corpus_mean"])
                    if n_ok == 0:
                        continue
                    unc_z = _uncorrected_bias_z(samples, result["corpus_mean"])
                    effect_results.append(PPIEffectResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n=result["n"], n_samples=n_ok,
                        null_value=result["corpus_mean"], mean_bias=bias_mean, bias_z=z,
                        coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
                    ))
            elif ct == "wmt_paired_bias":
                # Same PPIEffectResult shape as "single" above (reusing its
                # report/plot machinery unmodified), but the correct null
                # value differs PER TEST -- wilcoxon targets the population
                # MIDRANK-SIGN P(D>0)-0.5 quantity, the other three target
                # the MEAN (see _run_real_wmt_paired_bias_cell's docstring).
                for t, samples in result["samples_by_test"].items():
                    null_value = (result["true_paired_walsh_midrank_theta"] if t == WILCOXON.name
                                  else result["true_paired_mean"])
                    bias_mean, z, coverage, mean_width, n_ok = _effect_cell_stats(samples, null_value)
                    if n_ok == 0:
                        continue
                    unc_z = _uncorrected_bias_z(samples, null_value)
                    effect_results.append(PPIEffectResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n=result["n"], n_samples=n_ok,
                        null_value=null_value, mean_bias=bias_mean, bias_z=z,
                        coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
                    ))
            elif ct == "paired_bias":
                # Same PPIEffectResult shape as "single"/"wmt_paired_bias"
                # above, but null_value is always 0.0 for every test reached
                # here -- generate_real_paired_null_cell is an EXACT null by
                # construction (same items, two different judges), unlike
                # wmt_paired_bias's genuine two-condition paired data.
                for t, samples in result["samples_by_test"].items():
                    bias_mean, z, coverage, mean_width, n_ok = _effect_cell_stats(samples, 0.0)
                    if n_ok == 0:
                        continue
                    unc_z = _uncorrected_bias_z(samples, 0.0)
                    effect_results.append(PPIEffectResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n=result["n"], n_samples=n_ok,
                        null_value=0.0, mean_bias=bias_mean, bias_z=z,
                        coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
                    ))
            else:
                if ct == "twogroup":
                    bucket = twogroup_results
                elif ct == "paired":
                    bucket = paired_results
                elif ct in ("omnibus_independent", "omnibus_repeated"):
                    bucket = omnibus_results
                else:
                    # twogroup_power / omnibus_independent_power / wmt_paired_power
                    bucket = power_results
                for t in result["methods"]:
                    bucket.append(PPIResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n_reps=result["n_reps"],
                        corrected_rejects=result["corrected"][t], uncorrected_rejects=result["uncorrected"][t],
                        n=result["n"],
                    ))

        reporter = _ProgressReporter(max(len(work_items), 1), mode=args.progress, label="ppi_real")
        n_workers = max(1, getattr(args, "workers", 1))

        global _REAL_CORPORA
        _REAL_CORPORA = corpora

        if n_workers <= 1:
            for i, item in enumerate(work_items):
                result = _run_ppi_real_cell_worker(item)
                _consume(result)
                reporter.update(i + 1, detail=f"{result['dataset']} {result['check_type']}")
        else:
            ctx = _mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(_run_ppi_real_cell_worker, work_items)):
                    _consume(result)
                    reporter.update(i + 1, detail=f"{result['dataset']} {result['check_type']}")

        reporter.update(max(len(work_items), 1), detail="done")

        if effect_results:
            print_ppi_effect_report(effect_results, alpha=args.alpha)
            run_stem = f"ppi_real_effect_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi_effect(
                    results=effect_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                # bootstrap_t is classified "nonstandard" by pvalues.py's own
                # convention (_PPI_NONSTANDARD_TESTS); ppi_wilson (single-
                # sample, closed-form) isn't, and both are frequently present
                # together for binary datasets (arena) -- so, exactly like
                # pvalues.py's own run() does for its main PPI mode, this
                # calls save_ppi_effect_plot twice rather than once, so
                # bootstrap_t doesn't silently vanish from a single combined
                # plot (see save_ppi_effect_plot's nonstandard= filtering).
                # Each call is guarded by whether ITS bucket is non-empty --
                # a non-binary dataset's single-sample check has ONLY
                # bootstrap_t (no ppi_wilson), which is entirely
                # "nonstandard", so the plain (nonstandard=False) call would
                # otherwise get zero tests and crash building an empty legend.
                if _has_standard_test(effect_results):
                    plot_path = save_ppi_effect_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_bias_coverage_width.png"),
                    )
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                if _has_nonstandard_test(effect_results):
                    nonstd_plot_path = save_ppi_effect_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_bias_coverage_width_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")
                if _has_standard_test(effect_results):
                    cov_width_path = save_ppi_real_coverage_width_by_labfrac_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_coverage_width_by_labfrac.png"),
                    )
                    output_paths.append(cov_width_path)
                    print(f"Saved plot: {cov_width_path}")
                if _has_nonstandard_test(effect_results):
                    nonstd_cov_width_path = save_ppi_real_coverage_width_by_labfrac_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_coverage_width_by_labfrac_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_cov_width_path)
                    print(f"Saved plot: {nonstd_cov_width_path}")
            key_metrics["ppi_real_effect_n_results"] = len(effect_results)
            finite_z = [r.bias_z for r in effect_results if np.isfinite(r.bias_z)]
            key_metrics["ppi_real_effect_mean_abs_bias_z"] = float(np.mean(np.abs(finite_z))) if finite_z else float("nan")
            finite_cov = [r.coverage for r in effect_results if np.isfinite(r.coverage)]
            key_metrics["ppi_real_effect_mean_coverage"] = float(np.mean(finite_cov)) if finite_cov else float("nan")

        # twogroup + paired + omnibus combined into ONE Type-I report/plot,
        # matching pvalues.py's synthetic PPI sweep (run_ppi_simulation's
        # single combined `ppi_results` list, one print_ppi_report/
        # save_ppi_typeI_plot call) instead of separate ones per family --
        # safe to merge since all three families use entirely disjoint test
        # names (ttest/ttest_welch/mwu vs wilcoxon/paired_t/bayes_bootstrap/
        # bootstrap_t/tango vs anova_ind/anova_rep/friedman/kruskal, no
        # collision like the wmt_paired_bias/single-sample bootstrap_t one
        # elsewhere in this file), so this is purely additive -- no result
        # gets double-counted or aliased with another. Colors/legend come
        # from methods.py's Method.color per test, same registry pvalues.py's
        # plot uses, so a test keeps the same color whether it's plotted
        # from a synthetic or a real-data run.
        hypothesis_results = twogroup_results + paired_results + omnibus_results
        if hypothesis_results:
            print_ppi_report(hypothesis_results, alpha=args.alpha)
            run_stem = f"ppi_real_hypothesis_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(
                    results=hypothesis_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                # Same empty-bucket guard as the effect plot above -- a
                # binary dataset's paired methods are {paired_t,
                # bayes_bootstrap, tango}, i.e. only ONE standard test
                # (paired_t) but TWO nonstandard ones, so neither bucket is
                # reliably non-empty across eval_types.
                if _has_standard_test(hypothesis_results):
                    plot_path = save_ppi_typeI_plot(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"),
                    )
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                if _has_nonstandard_test(hypothesis_results):
                    nonstd_plot_path = save_ppi_typeI_plot(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")
                if _has_standard_test(hypothesis_results):
                    heatmap_path = save_ppi_real_labfrac_dataset_heatmap(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_dataset_labfrac_heatmap.png"),
                    )
                    output_paths.append(heatmap_path)
                    print(f"Saved plot: {heatmap_path}")
                if _has_nonstandard_test(hypothesis_results):
                    nonstd_heatmap_path = save_ppi_real_labfrac_dataset_heatmap(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_dataset_labfrac_heatmap_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_heatmap_path)
                    print(f"Saved plot: {nonstd_heatmap_path}")
                if _has_standard_test(hypothesis_results):
                    alignment_plot_path = save_ppi_real_alignment_plot(
                        results=hypothesis_results, alignment_by_dataset_judge=alignment_by_dataset_judge,
                        alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_alignment.png"),
                    )
                    output_paths.append(alignment_plot_path)
                    print(f"Saved plot: {alignment_plot_path}")
                if _has_nonstandard_test(hypothesis_results):
                    nonstd_alignment_plot_path = save_ppi_real_alignment_plot(
                        results=hypothesis_results, alignment_by_dataset_judge=alignment_by_dataset_judge,
                        alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_alignment_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_alignment_plot_path)
                    print(f"Saved plot: {nonstd_alignment_plot_path}")
                if _has_standard_test(hypothesis_results):
                    leaderboard_path = save_ppi_real_judge_leaderboard_plot(
                        results=hypothesis_results, alignment_by_dataset_judge=alignment_by_dataset_judge,
                        alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_judge_leaderboard.png"),
                    )
                    output_paths.append(leaderboard_path)
                    print(f"Saved plot: {leaderboard_path}")
                if _has_nonstandard_test(hypothesis_results):
                    nonstd_leaderboard_path = save_ppi_real_judge_leaderboard_plot(
                        results=hypothesis_results, alignment_by_dataset_judge=alignment_by_dataset_judge,
                        alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_judge_leaderboard_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_leaderboard_path)
                    print(f"Saved plot: {nonstd_leaderboard_path}")
                # Restricted to kruskal/friedman by default (save_ppi_real_
                # n_labfrac_heatmap's tests= default) -- not standard/
                # nonstandard-split like the plots above, both tests it
                # covers are standard hypothesis tests.
                try:
                    n_labfrac_path = save_ppi_real_n_labfrac_heatmap(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_n_labfrac_heatmap.png"),
                    )
                    output_paths.append(n_labfrac_path)
                    print(f"Saved plot: {n_labfrac_path}")
                except ValueError as e:
                    print(f"  Skipping N x label_frac heatmap: {e}")
            c_tot = sum(r.corrected_rejects for r in hypothesis_results)
            u_tot = sum(r.uncorrected_rejects for r in hypothesis_results)
            n_tot = sum(r.n_reps for r in hypothesis_results)
            key_metrics["ppi_real_hypothesis_n_results"] = len(hypothesis_results)
            key_metrics["ppi_real_hypothesis_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_real_hypothesis_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

        if power_results:
            print_ppi_real_power_report(power_results, alpha=args.alpha)
            run_stem = f"ppi_real_power_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi_real_power(
                    results=power_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                )
            if args.plots == "save":
                if _has_standard_test(power_results):
                    power_plot_path = save_ppi_real_power_plot(
                        results=power_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_power.png"),
                    )
                    output_paths.append(power_plot_path)
                    print(f"Saved plot: {power_plot_path}")
                if _has_nonstandard_test(power_results):
                    nonstd_power_plot_path = save_ppi_real_power_plot(
                        results=power_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_power_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_power_plot_path)
                    print(f"Saved plot: {nonstd_power_plot_path}")
            c_tot = sum(r.corrected_rejects for r in power_results)
            u_tot = sum(r.uncorrected_rejects for r in power_results)
            n_tot = sum(r.n_reps for r in power_results)
            key_metrics["ppi_real_power_n_results"] = len(power_results)
            key_metrics["ppi_real_power_mean_corrected"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_real_power_mean_uncorrected"] = float(u_tot / n_tot) if n_tot else float("nan")

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as e:
        return CaseResult(case_name=CASE_NAME, status="error", error=str(e), duration_s=time.time() - t0)
