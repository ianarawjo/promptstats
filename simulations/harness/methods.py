"""Canonical CI-method registry, shared across cases.

Ported from the ``_METHOD_COLORS`` dict and method-name constants duplicated
across ``sim_compare_boot.py`` / ``sim_compare_boot_nested.py`` /
``sim_dove.py`` / ``sim_tango_real.py``. Each method is one ``Method``
instance carrying its name and plot color; callers work with the instance
directly (``method.name``, ``method.color``) instead of threading a bare
string through one lookup table per case. Centralizing this means a method
keeps the same name and the same plot color in every case's tables and
figures, instead of each case file redefining its own copy that can
silently drift out of sync.

There is no separate ``ci_nested`` case: ``sim_compare_boot_nested.py`` was
folded into ``cases/ci_single.py``'s and ``cases/ci_paired.py``'s
``--nested-mode`` flag instead, so single- and paired-sample CI logic stays
in one place per estimand. The "flat" vs "*_nested" method pairs below exist
because nested mode reports both side by side (flat = cell-mean reduction,
nested = full N×R matrix) to show what ignoring run structure costs.

Add new Method instances/groupings here as new cases are ported -- don't
predeclare methods for cases that don't exist as code yet.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_METHOD_COLOR = "#333333"


@dataclass(frozen=True)
class Method:
    name: str
    color: str = DEFAULT_METHOD_COLOR

    def __str__(self) -> str:
        return self.name

    def __format__(self, format_spec: str) -> str:
        return format(self.name, format_spec)


# ---------------------------------------------------------------------------
# Bootstrap-family methods (apply to single-sample means and paired diffs,
# flat or nested)
# ---------------------------------------------------------------------------
BOOTSTRAP = Method("bootstrap", "#1f77b4")
BCA = Method("bca", "#2ca02c")
BAYES_BOOTSTRAP = Method("bayes_bootstrap", "#ff7f0e")
SMOOTH_BOOTSTRAP = Method("smooth_bootstrap", "#9467bd")
BOOTSTRAP_T = Method("bootstrap_t", "#d62728")
BOOTSTRAP_METHODS = [BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T]

# ---------------------------------------------------------------------------
# Single-sample extras
# ---------------------------------------------------------------------------
T_INTERVAL = Method("t_interval", "#8c564b")
WILSON = Method("wilson", "#e377c2")
JEFFREYS = Method("jeffreys", "#e9cd14")
WALD = Method("wald", "#7f7f7f")
CLOPPER_PEARSON = Method("clopper_pearson", "#bcbd22")
BAYES_SINGLE = Method("bayes_indep", "#17becf")
BETA = Method("beta", "#f0027f")
LOGIT_T = Method("logit_t", "#a6761d")
NIG = Method("nig", "#888888")
EL = Method("el", "#00441b")
LOGIT_T_2ND = Method("logit_t_2nd", "#bd5b17")
"""evalstats.core.resampling.logit_t_ci_1d(..., order=2) -- the optional
2nd-order (curvature) bias-corrected variant, registered here purely for
direct comparison against the order=1 default (LOGIT_T, used everywhere --
both ci_single.py and ci_paired.py). Gives a modest coverage improvement on
boundary-hugging/right-skewed single-sample data at negligible width cost,
but not part of the default battery: the gain wasn't judged worth changing
recorded results over. Does not help ci_paired's use of
logit_t (rescaled_ci recentres a paired diff near 0.5 regardless of raw
skew, where order=2's boundary-only correction never activates)."""

LOGIT_T_DITHER = Method("logit_t_dither", "#ceb483")  # pastel tint of LOGIT_T's #a6761d
SMOOTH_BOOTSTRAP_DITHER = Method("smooth_bootstrap_dither", "#c4abdb")  # pastel tint of SMOOTH_BOOTSTRAP's #9467bd
"""ci_paired.py-only, non-binary eval types (see that file's
add_dither_extras): the SAME logit_t/smooth_bootstrap paired-diff CI, but
with U(-half, +half) jitter added independently to each arm's raw values
before differencing (then clipped back to the scale), where half is
auto-detected per rep from the data's own quantization grid via
_detect_dither_halfwidth -- 0.0 (no jitter) if none is found. Fixes a real, severe
small-N pathology distinct from LOGIT_T_2ND's: on a PAIRED diff of two
highly-correlated (shared-item) LIKERT arms, rounding mostly cancels
between arms -- most items round to the identical integer in both arms
(diff=0), and only the rare item whose latent value sits near a rounding
boundary shows a nonzero diff. At small N it's entirely plausible NONE of
the sampled items are boundary-adjacent, so the sample's diffs come out
literally constant, collapsing the sample variance to ~0 regardless of the
(real, nonzero) population-level diff variance -- any variance-based CI
built from that is catastrophically overconfident. Dithering recovers
near-nominal coverage where plain logit_t badly under-covers in this
regime. NOT the same mechanism LOGIT_T_2ND targets (that's single-sample
boundary-hugging skew) and does NOT help ci_single, where plain logit_t is
already well-calibrated -- this is a paired-diff-specific pathology.
nig_ci_1d fixes the SAME failure via a wider prior instead, but costs far
more power at small N/moderate k, because nig's conservatism is
unconditional while dithering targets the actual missing variance
directly.

A hardcoded +-0.5 jitter (for transparency/direct comparison against
likert) does NOT work on CONTINUOUS data: +-0.5 is calibrated to undo
exactly one unit of INTEGER rounding, but on continuous's own [0, 1]-scale
data it's HALF the entire range, causing heavy boundary clipping and a
systematic bias in the mean that doesn't shrink with N while the CI does,
so coverage gets WORSE as N grows rather than converging. Replacing the
hardcoded width with _detect_dither_halfwidth's data-driven detection
fixes this generally: it returns 0.0 (no jitter, dither variant reduces
exactly to its base method) on genuinely continuous data with no
recurring gap, so it's now safe to run on any non-binary type, and it ALSO
catches the case a fixed eval_type check never could -- data labeled
"continuous" that's actually coarse in practice (e.g. a judge that only
emits a handful of distinct values), which would otherwise silently
re-trigger the same rounding-cancellation pathology likert has."""

BINARY_SINGLE_EXTRA_METHODS = [WILSON, JEFFREYS, WALD, CLOPPER_PEARSON, BAYES_SINGLE]
CONTINUOUS_EXTRA_METHODS = [BETA, LOGIT_T, NIG, EL]
CONTINUOUS_EXTRA_METHODS_WITH_LOGIT_T_2ND = [BETA, LOGIT_T, LOGIT_T_2ND, NIG, EL]
"""CONTINUOUS_EXTRA_METHODS plus logit_t_2nd -- opt-in via --methods, not
part of the default battery (LOGIT_T_2ND is a validation-only comparison
variant, not a distinct recommended method)."""

# ---------------------------------------------------------------------------
# Paired (pairwise-difference) extras -- for cases/ci_paired.py once ported
# ---------------------------------------------------------------------------
MJ_FLOOR = Method("mj_floor")  # no color in the legacy palette; uses the default
"""evalstats.tests._ppi_paired_mj_floor's default construction -- PPI++ closed-
form power-tuned lambda* since the validation documented at
MJ_FLOOR_FIXED_LAMBDA (below); see that Method's docstring for the legacy
fixed-lambda=1 construction and the comparison it's kept for."""
PPI_WILSON = Method("ppi_wilson", "#e377c2")
"""PPI-corrected single-sample Wilson score interval (evalstats.tests.
_ppi_single_wilson) -- a binary-proportion analogue of MJ_FLOOR's paired
Wilson-style effective-n trick, for a single-sample (not two/paired-group)
mean estimand. Deliberately not named "wilson" -- that name is already
BINARY_SINGLE_EXTRA_METHODS' plain (non-PPI-corrected) Wilson CI for
ci_single.py, a different statistical procedure that happens to share the
same textbook name; reusing it here would silently overwrite that entry
(same module-level name, last assignment wins). Exercised by both
cases/ppi_real.py's real-data single-sample bias/coverage check and
pvalues.py's synthetic PPI sweep -- see PPI_BOOTSTRAP_T_SINGLE below and
cases/pvalues.py's _run_ppi_effect_cell single-arm blocks."""
PPI_BOOTSTRAP_T_SINGLE = Method("bootstrap_t_single", "#9edae5")
"""PPI-corrected single-sample studentized-bootstrap CI (evalstats.tests.
_ppi_single_bootstrap_t) -- the bounded_01/continuous analogue of PPI_WILSON,
targeting the same single-sample (not paired) mean estimand PPI_AUTO_METHOD_
TABLE routes non-binary robustness CIs to. Deliberately distinct from
BOOTSTRAP_T (the paired/two-sample PPI method of the same underlying
construction) -- same reason PPI_WILSON isn't named "wilson": different
estimand, would silently collide if given the same Method name."""
PPI_BONETT_PRICE = Method("ppi_bonett_price", "#556b2f")
"""PPI-corrected Bonett-Price adjusted-Wald interval for the paired BINARY
difference (evalstats.tests._ppi_paired_bonett_price). Deliberately distinct
from BONETT_PRICE, the non-PPI ci_paired entry of the same underlying
construction: the two are reported in different sweeps, and sharing a Method
name would make a `bonett_price` row ambiguous between the corrected and
uncorrected estimand -- the same reason PPI_WILSON is not named "wilson".
Deliberately SHARES BONETT_PRICE's colour (#556b2f) so the method reads the
same across the ci_paired and PPI figures. Safe because the colour test
enforces distinctness only within co-plotted groups, and no figure draws a
PPI method beside its non-PPI namesake; against its actual figure-mates
(ppi_wilson/ppi_t_interval/ppi_logit_t) it sits at dE 46/82/50.
Replaced MJ_FLOOR in the official PPI set on 2026-08-26."""
PPI_T_INTERVAL = Method("ppi_t_interval", "#8c564b")
"""PPI-corrected closed-form (no-bootstrap) t-interval for an unbounded
numeric mean/mean-difference estimand (evalstats.tests._ppi_single_t_interval
/ _ppi_paired_t_interval, both thin wrappers around evalstats.ppi.
_analytic_mean_correct). The closed-form replacement for BOOTSTRAP_T's role
in PPI_AUTO_METHOD_TABLE's "unbounded" row (see evalstats.config). Uses the
analytic construction at every n_lab, not just below _MIN_LAB_RECOMMENDED,
mirroring the precedent evalstats.ppi._ANALYTIC_ALWAYS_PREFERRED set for
the Wilcoxon estimand. Run as a paired mean-difference test in this harness
(cell.llm_x/llm_y/lab_x/lab_y), the same structural role BOOTSTRAP_T/PAIRED_T
occupy -- see PPI_LOGIT_T below for its [0,1]-bounded sibling. Deliberately
not named "t_interval" -- that's already the plain (non-PPI-corrected)
classical T_INTERVAL method above, a different statistical procedure that
happens to share the textbook name (same reason PPI_WILSON isn't named
"wilson").

Paired estimand only -- see PPI_T_INTERVAL_SINGLE below for the
single-sample sibling, split out for the same reason PPI_BOOTSTRAP_T_SINGLE
is split from BOOTSTRAP_T: this Method's name is pooled across every
paired-family check that uses it (print_ppi_effect_report groups by test
name only, not by estimand), so a single-sample usage under the same name
would silently merge two different estimands' bias/coverage stats
together."""
PPI_T_INTERVAL_SINGLE = Method("ppi_t_interval_single", "#6baed6")  # lighter tint of PPI_T_INTERVAL
"""Single-sample sibling of PPI_T_INTERVAL (evalstats.tests.
_ppi_single_t_interval), split out for the same reason
PPI_BOOTSTRAP_T_SINGLE is split from BOOTSTRAP_T -- see PPI_T_INTERVAL's
docstring. Targets an unbounded numeric single-sample mean estimand, the
non-binary/non-[0,1]-bounded counterpart to PPI_WILSON's role."""
PPI_LOGIT_T = Method("ppi_logit_t", "#a6761d")
"""PPI-corrected closed-form (no-bootstrap) logit-t CI for a [lo, hi]-bounded
numeric mean/mean-difference estimand (evalstats.tests._ppi_single_logit_t /
_ppi_paired_logit_t, wrapping evalstats.ppi._analytic_logit_t_correct -- the
PPI analogue of evalstats.core.resampling.logit_t_ci_1d's delta-method construction,
reusing the same point estimate/p-value as PPI_T_INTERVAL and differing
only in the CI's shape). The closed-form replacement for BOOTSTRAP_T's
role in PPI_AUTO_METHOD_TABLE's "bounded_01" row. Run as a paired
mean-difference test here, rescaled onto [0, 1] via this harness's own
EVAL_TYPE_SCALE_BOUNDS[eval_type] (continuous/likert/grades; excluded from
binary scenarios the same way BOOTSTRAP_T is). Deliberately not named
"logit_t" -- see PPI_T_INTERVAL's docstring for why.

Paired estimand only -- see PPI_LOGIT_T_SINGLE below and PPI_T_INTERVAL's
matching docstring addendum for why (same name-pooling collision risk)."""
PPI_LOGIT_T_SINGLE = Method("ppi_logit_t_single", "#ccebc5")  # lighter tint of PPI_LOGIT_T
"""Single-sample sibling of PPI_LOGIT_T (evalstats.tests._ppi_single_logit_t),
split out for the same reason PPI_BOOTSTRAP_T_SINGLE is split from
BOOTSTRAP_T -- see PPI_T_INTERVAL_SINGLE's docstring (identical reasoning,
[0,1]-bounded instead of unbounded). PPI_AUTO_METHOD_TABLE's "bounded_01"
robustness method -- the non-binary counterpart to PPI_WILSON's binary
role; every real dataset ppi_real.py checks is already rescaled to [0, 1]
(see RealJudgeBiasCorpus), so this applies uniformly there."""
TANGO_SCC = Method("tango_scc", "#b15928")
#: The GENUINE Tango (1998) asymptotic score interval, in closed form via
#: Chang et al. (2024)'s quartic with the continuity correction set to zero.
#: Validated against the published limits in Fagerland, Lydersen & Laake
#: (2014) Table V. Added 2026-08-24 so the paper can compare the real Tango
#: against MJ_FLOOR, which was previously (and wrongly) labelled "tango".
TANGO_EXACT = Method("tango_exact", "#7b3294")
#: May & Johnson (1997) eq. 11 exactly as published, with NO discordance
#: floor. Included as the baseline that shows why MJ_FLOOR floors it: this
#: degenerates to zero width at n10=n01=0 and under-covers at low
#: discordance (0.787 vs nominal 0.95 at n=15, S=0.10).
MJ_UNFLOORED = Method("mj_unfloored", "#c2a5cf")
#: Bonett & Price (2012) Laplace-adjusted Wald -- the PRIME recommendation of
#: Fagerland, Lydersen & Laake (2014) Table IX for a CI on the difference
#: between paired proportions. Validated against their Table V.
BONETT_PRICE = Method("bonett_price", "#556b2f")  # olive -- #fdae61 sat only
#: deltaE 9 from bayes_indep_comp's #ffbb78, i.e. indistinguishable in a legend.
#: Newcombe (1998) method 10, the square-and-add / MOVER-Wilson interval --
#: also recommended by Fagerland et al. (2014) Table IX, and validated
#: against their Table V. This is the ONLY Newcombe interval in evalstats;
#: the previous discordant-pairs "newcombe_score" was removed 2026-08-24
#: because it is a different method and covers poorly.
NEWCOMBE_MOVER = Method("newcombe_mover", "#aec7e8")
BAYES_PAIR_INDEP = Method("bayes_indep_comp", "#ffbb78")
BAYES_PAIR_PAIRED = Method("bayes_paired_comp", "#98df8a")
WALD_PAIR_INDEP = Method("wald_indep", "#7f7f7f")  # same grey as ci_single's WALD -- both are the naive baseline
PAIRWISE_EXTRA_METHODS = [T_INTERVAL, LOGIT_T, NIG, EL]
DITHER_EXTRA_METHODS = [LOGIT_T_DITHER, SMOOTH_BOOTSTRAP_DITHER]
"""ci_paired.py-only, non-binary eval types -- see LOGIT_T_DITHER's
docstring. Structurally a SEPARATE list from PAIRWISE_EXTRA_METHODS (not
folded into it) since the actual jitter is data-gated (auto-detected per
rep, a no-op when the data shows no quantization grid), but runs BY
DEFAULT for all non-binary cells whenever --methods doesn't exclude them --
same default-inclusion behavior as PAIRWISE_EXTRA_METHODS itself
(ci_paired.py's `_want` returns True for everything when --methods is
unset), NOT the hidden opt-in-only precedent LOGIT_T_2ND uses. Pass
--methods without these two names to exclude them if only comparing the
pre-existing battery."""
BINARY_PAIRWISE_EXTRA_METHODS = [NEWCOMBE_MOVER, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP]

# ---------------------------------------------------------------------------
# Nested-mode methods -- for ci_single.py's/ci_paired.py's --nested-mode,
# ported from sim_compare_boot_nested.py. "Flat" methods here always reduce
# an (n, runs) matrix to per-input cell means first; "nested" methods apply
# directly to the full (n, runs) matrix. Both are reported side by side so
# the gap between them (the cost of ignoring run structure) is visible.
# ---------------------------------------------------------------------------
BOOTSTRAP_NESTED = Method("bootstrap_nested", "#aec7e8")
BAYES_NESTED = Method("bayes_bootstrap_nested", "#ffbb78")
SMOOTH_NESTED = Method("smooth_bootstrap_nested", "#c5b0d5")
BCA_NESTED = Method("bca_nested", "#98df8a")
BOOTSTRAP_T_NESTED = Method("bootstrap_t_nested", "#ff9896")
NESTED_METHODS = [BOOTSTRAP_NESTED, BAYES_NESTED, SMOOTH_NESTED, BCA_NESTED, BOOTSTRAP_T_NESTED]

WILSON_FLAT = Method("wilson_flat", "#e7298a")
WALD_FLAT = Method("wald_flat", "#66a61e")
CP_FLAT = Method("clopper_pearson_flat", "#e6ab02")
BAYES_INDEP_FLAT = Method("bayes_indep_flat", "#1b9e77")
BINARY_FLAT_METHODS = [WILSON_FLAT, WALD_FLAT, CP_FLAT, BAYES_INDEP_FLAT]

WILSON_OD = Method("wilson_od", "#666666")
WILSON_OD_BC = Method("wilson_od_bc", "#e31a1c")
WILSON_OD_T = Method("wilson_od_t", "#6a3d9a")
JEFFREYS_OD = Method("jeffreys_od", "#b2df8a")
CP_OD = Method("cp_od", "#fb9a99")
BB_BAYES = Method("bb_bayes", "#33a02c")
BB_BAYES_ROBUST = Method("bb_bayes_robust", "#ff7f00")
BINARY_NESTED_METHODS = [WILSON_OD, WILSON_OD_BC, WILSON_OD_T, JEFFREYS_OD, CP_OD, BB_BAYES, BB_BAYES_ROBUST]

BOOTSTRAP_DIFF_NESTED = Method("bootstrap_diff_nested", "#1b9e77")
BAYES_DIFF_NESTED = Method("bayes_diff_nested", "#d95f02")
SMOOTH_DIFF_NESTED = Method("smooth_diff_nested", "#7570b3")
PAIR_DIFF_NESTED_METHODS = [BOOTSTRAP_DIFF_NESTED, BAYES_DIFF_NESTED, SMOOTH_DIFF_NESTED]

MJ_FLOOR_FLAT = Method("mj_floor_flat", "#e7298a")
MJ_FLOOR_MEAN = Method("mj_floor_mean", "#8c564b")
NEWCOMBE_FLAT = Method("newcombe_flat", "#66a61e")
#: Bonett-Price on run 0 only -- the single-run reference the multi-run
#: variants have to beat, and the direct counterpart of MJ_FLOOR_FLAT.
#: Muted olive, deliberately in the same family as BONETT_PRICE's #556b2f
#: (deltaE 26, so still distinguishable) since it IS that method, on one run.
BONETT_PRICE_FLAT = Method("bonett_price_flat", "#a0a871")
BINARY_PAIR_FLAT_METHODS = [
    MJ_FLOOR_FLAT, NEWCOMBE_FLAT, BONETT_PRICE_FLAT,
    BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP,
]

#: RETIRED 2026-08-25. mj_floor_er's Kish effective-runs term cancels exactly
#: when its max() does not clamp and inflates variance up to 2.8x when it
#: does, making it inert in the high-ICC regime real eval data occupies and
#: conservative elsewhere; mj_floor_mmnt is algebraically the same interval as
#: the cluster variant whenever its floor does not clip. Neither is swept any
#: more. MJ_FLOOR_CLUSTER is retained as the one multi-run mj_floor comparator:
#: plain item-level variance, no R_eff, nothing to go wrong in the variance.
#: NOTE it still carries the family's centre shrinkage d_hat/(1 + z^2/n), which
#: uses the ITEM count only and is therefore untouched by R -- so it inherits
#: the same lopsided-scenario coverage tail. It is a comparator, not a fallback.
MJ_FLOOR_CLUSTER = Method("mj_floor_cluster", "#a6761d")

# Multi-run Bonett-Price (evalstats.core.resampling, added 2026-08-25 so the
# single-run winner has a multi-run entry to run against MJ_FLOOR_ER). All
# three are one estimator -- a Wald interval on the per-item mean difference
# over the sample augmented by two Laplace pseudo-items at delta = +1 and -1
# -- separated only by the floor they put on the item-level variance, so
# their widths always order CLUSTER <= MMNT <= ER. Each reduces EXACTLY to
# BONETT_PRICE at runs == 1. See the derivation block above
# _bp_item_moments in evalstats/core/resampling.py.
#: No floor: the single-run construction carried over unchanged, with the
#: item as the unit of analysis. The most principled of the three -- the
#: item-level variance already absorbs between-run correlation, and a
#: correctly-specified Kish design effect provably reduces to it.
BONETT_PRICE_CLUSTER = Method("bonett_price_cluster", "#3585f7")
#: RETIRED 2026-08-25: _er and _mmnt only add a floor to the item-level
#: variance, and neither floor ever fires -- the Laplace pseudo-items already
#: dominate it. On the real-data nested sweep all three agreed to four
#: decimals, so they were three rows of the same interval. CLUSTER is kept
#: because it is the one with no floor at all, and so the only one that
#: describes in a sentence: the single-run construction with the item as the
#: unit of analysis.
#: Yang, Sun & Hardin (2012) X^2_Score: Tango's score statistic with the
#: Eliasziw-Donner variance inflation, inverted through the same quartic as the
#: unclustered case. THE published competitor for clustered matched-pair CIs --
#: reproduces their worked example exactly and reduces to tango_scc(c=0) when
#: there is no clustering.
CLUSTERED_SCORE = Method("clustered_score", "#4a148c")
#: NOT SWEPT. Yang et al. (2010) modified Obuchowski is cluster-level and
#: estimates no ICC, but it carries no small-sample adjustment: at R=1 it is
#: bit-identical to the unregularised Wald on item differences, and it returns
#: a zero-width interval at zero discordance. Measured MinCov .613 with 231 of
#: 10140 real-data cells below .93 -- far worse than anything else credible.
#: The implementation and its validation against clust.bin.pair are retained
#: in evalstats.core.resampling as a citable negative result.
#: Pseudo-item MAGNITUDE Laplace-shrunk toward the R=1 reference of 1, with
#: BP's own weight of two pseudo-items:
#: m2 = (sum delta^2 + 2)/(sum u + 2). Reduces to BONETT_PRICE at R=1 by the
#: identity sum(delta^2) == sum(u) there. See
#: evalstats.core.resampling.bonett_price_paired_ci_multirun_shrunk.
BONETT_PRICE_SHRUNK = Method("bonett_price_shrunk", "#c2185b")

BINARY_PAIR_NESTED_METHODS = [
    MJ_FLOOR_CLUSTER, BONETT_PRICE_CLUSTER, BONETT_PRICE_SHRUNK, CLUSTERED_SCORE,
]
"""Every multi-run binary pairwise CI the harness can run, selectable by name
via --methods. NOT what runs by default -- see BINARY_PAIR_NESTED_OFFICIAL."""

BINARY_PAIR_NESTED_OFFICIAL = [
    m for m in BINARY_PAIR_NESTED_METHODS if m is not BONETT_PRICE_CLUSTER
]
"""The default (--methods unset) multi-run binary set. BONETT_PRICE_CLUSTER is
excluded: it is the same estimator as BONETT_PRICE_SHRUNK with the pseudo-item
magnitude pinned at 1 instead of shrunk, so reporting both invites readers to
treat a parameter setting as a competing method. It stays implemented and
selectable (--methods bonett_price_cluster) as the ablation showing what the
magnitude shrinkage buys."""

# ---------------------------------------------------------------------------
# cases/pvalues.py -- raw pairwise p-value/rejection procedures (non-PPI
# path), ported from sim_compare_pvalues.py's PAIRWISE_METHODS. These compute
# a p-value/rejection decision via evalstats.core.paired.pairwise_differences,
# not a CI -- distinct from the CI-coverage methods above even where a name
# overlaps conceptually (e.g. BOOTSTRAP/BCA/BAYES_BOOTSTRAP/SMOOTH_BOOTSTRAP
# are reused as-is; "newcombe"/"bayes_binary" are NOT the same underlying
# computation as ci_paired's "newcombe_mover"/"bayes_indep_comp", so they get
# distinct Method instances despite the conceptual overlap).
# ---------------------------------------------------------------------------
MCNEMAR = Method("mcnemar", "#393b79")
#: McNemar MID-P. Fagerland, Lydersen & Laake (2014) sec. 9.1 recommend the
#: asymptotic and mid-p McNemar tests and recommend AGAINST the exact
#: conditional test (MCNEMAR above) as markedly conservative. Added
#: 2026-08-25 so the sweep compares the recommended test, not only the
#: one evalstats currently reports alongside its binary paired CIs.
MCNEMAR_MIDP = Method("mcnemar_midp", "#00868b")
PERMUTATION = Method("permutation", "#8c6d31")
SIGN_TEST = Method("sign_test", "#843c39")
#: REMOVED from the p-value sweep 2026-08-25. "newcombe" is a CI method,
#: not a test: evalstats returns McNemar alongside the Newcombe interval,
#: so as a p-value row it reproduced mcnemar exactly (and now reproduces
#: mcnemar_midp exactly). Kept defined because summary labels still refer
#: to it, but no longer swept as if it were a distinct test.
NEWCOMBE_PVAL = Method("newcombe", "#7b4173")
BAYES_BINARY = Method("bayes_binary", "#5254a3")
WILCOXON = Method("wilcoxon", "#8ca252")
PAIRED_T = Method("paired_t", "#bd9e39")
PAIRWISE_PVALUE_METHODS = [
    MCNEMAR, MCNEMAR_MIDP, BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T,
    PERMUTATION, SIGN_TEST, BAYES_BINARY, WILCOXON, PAIRED_T,
]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- multi-arm multiplicity-correction strategies (non-PPI
# path), ported from sim_compare_pvalues.py's MULTIARM_CORRECTIONS.
# NONE/HOLM/BONFERRONI/FDR_BH/HOCHBERG/SHAFFER correct the Wilcoxon
# signed-rank p-value -- evalstats' canonical, eval-type-agnostic paired
# test (unlike Tango/Logit-t in --mode simultaneous_ci's
# CANONICAL_SIMULTANEOUS_CI_METHODS, one test covers binary/continuous/
# likert/grades alike, so there's no per-eval-type split here) -- rather
# than --multiarm-method's raw p-value; see cases/pvalues.py's
# _compute_multiarm_metrics. HOCHBERG is a closed-form step-up refinement of
# HOLM (never more conservative, valid under the non-negative dependence
# repeated-measures/shared-item designs produce -- see
# evalstats.core.stats_utils.correct_pvalues). SHAFFER is a closed-form
# step-down refinement of HOLM specific to *all-pairwise* comparisons among
# k arms, exploiting the transitivity of equality (if A=B and B=C then
# A=C) to use a smaller, non-constant divisor sequence instead of HOLM's
# plain (m, m-1, ..., 1) -- see
# evalstats.core.stats_utils._shaffer_adjusted_pvalues.
#
# MAX_T/ROMANO_WOLF/WESTFALL_YOUNG are the resampling-based family: MAX_T is
# *single-step* studentized-bootstrap max-T (one joint critical value for
# every pair); ROMANO_WOLF is the *step-down* refinement of the same
# bootstrap-t null (recomputing the max only over not-yet-rejected pairs at
# each step, so it dominates MAX_T in power for the same FWER guarantee);
# WESTFALL_YOUNG is the permutation-based (per-item sign-flip) analogue of
# ROMANO_WOLF's step-down algorithm, exact under exchangeability of the
# paired design rather than relying on the bootstrap's asymptotic
# justification. All three need a bootstrap/permutation-compatible
# resampling scheme (Wilcoxon has no joint max-T analogue), so they stay on
# --multiarm-method (bootstrap_t by default) / per-item paired differences
# regardless of eval type -- see cases/pvalues.py's
# _stepdown_max_t_pvalues. FRIEDMAN_NEMENYI is unaffected either way --
# already its own rank-based omnibus + post-hoc test.
#
# BOOT is the multiarm analogue of --mode simultaneous_ci's `boot` (see
# CANONICAL_SIMULTANEOUS_CI_METHODS below, whose CORR_BOOT instance this
# reuses as-is): it widens the canonical Wilcoxon p-value using a joint
# bootstrap critical value (the same max-over-all-pairs studentized-mean
# resample MAX_T/ROMANO_WOLF use -- see cases/pvalues.py's
# _bootstrap_t_matrix) instead of a fixed, correlation-blind factor the way
# HOLM/BONFERRONI/HOCHBERG/SHAFFER do -- rescaling raw_p by alpha/alpha_eff,
# where alpha_eff is that joint critical value translated back to an
# equivalent significance level (mirroring
# evalstats.core.paired._joint_bootstrap_scaled_simultaneous_cis's z<->alpha
# translation). Unlike MAX_T/ROMANO_WOLF/WESTFALL_YOUNG, BOOT is NOT tied to
# --multiarm-method -- like NONE/HOLM/etc. it always operates on the
# canonical Wilcoxon statistic, so it directly tests whether the small FWER
# excess observed for MAX_T/ROMANO_WOLF at n=500-2000 is specific to their
# studentized-mean bootstrap-t construction or a more general property of
# bootstrap-based FWER correction.
# ---------------------------------------------------------------------------
CORR_NONE = Method("none", "#9c9ede")
CORR_HOLM = Method("holm", "#cedb9c")
CORR_BONFERRONI = Method("bonferroni", "#e7ba52")
CORR_FDR_BH = Method("fdr_bh", "#ad494a")
CORR_HOCHBERG = Method("hochberg", "#e6550d")
CORR_SHAFFER = Method("shaffer", "#9e9ac8")
CORR_FRIEDMAN_NEMENYI = Method("friedman_nemenyi", "#a55194")
CORR_MAX_T = Method("max_t", "#5254a3")
CORR_ROMANO_WOLF = Method("romano_wolf", "#6baed6")
CORR_WESTFALL_YOUNG = Method("westfall_young", "#74c476")
CORR_BOOT = Method("boot", "#3182bd")
MULTIARM_CORRECTION_METHODS = [
    CORR_NONE, CORR_HOLM, CORR_BONFERRONI, CORR_FDR_BH, CORR_HOCHBERG, CORR_SHAFFER,
    CORR_FRIEDMAN_NEMENYI, CORR_MAX_T, CORR_ROMANO_WOLF, CORR_WESTFALL_YOUNG, CORR_BOOT,
]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- simultaneous-CI construction methods (non-PPI path),
# for --mode simultaneous_ci. Reuses CORR_NONE/CORR_MAX_T/CORR_BONFERRONI
# as-is (same name/color as their multiarm p-value-correction counterparts):
# CORR_NONE is the naive per-pair CI with no simultaneous adjustment at all
# (the "why do you need any correction?" baseline, same role as multiarm's
# own `none` row); CORR_MAX_T/CORR_BONFERRONI are the two constructions
# evalstats.core.paired's _simultaneous_cis_router picks between. Only these
# three of multiarm's six correction strategies have a well-established
# simultaneous-CI dual -- holm/fdr_bh are p-value-only adjustments with no
# associated CI, and friedman_nemenyi operates on rank differences rather
# than the raw mean-difference scale a CI here would need.
# ---------------------------------------------------------------------------
SIMULTANEOUS_CI_METHODS = [CORR_NONE, CORR_BONFERRONI, CORR_MAX_T]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- canonical-CI-based simultaneous-CI constructions
# (non-PPI path), for --mode simultaneous_ci. `none`/`bonferroni` (built on
# `matrix_raw.results`) and `sidak`/`boot` are all built on evalstats'
# *canonical* pairwise CI method for the scenario's eval type -- Tango for
# binary, Logit-t for continuous/likert (bounded [0, 1]/[lo, hi] numeric
# data; see evalstats.config.AUTO_ANALYZE_METHOD_TABLE's "binary"/
# "bounded_01" rows) -- rather than whatever --multiarm-method is in force.
# max_t is the one exception: it needs a bootstrap-compatible method to
# resample from (neither Tango nor Logit-t is), so it keeps using
# --multiarm-method (bootstrap_t by default) regardless of eval type.
# `grades` has no canonical default wired up here (deliberately out of
# scope -- see cases/pvalues.py's _CANONICAL_CI_FUNC_BY_EVAL_TYPE), so these
# rows are simply absent for grades scenarios.
# SIDAK ("sidak") and BOOT ("boot") widen the canonical CI to hold
# family-wise via evalstats.core.paired's generic (not method-specific)
# _sidak_simultaneous_cis / _joint_bootstrap_scaled_simultaneous_cis, called
# with whichever ci_func matches the scenario's eval type: Sidak's
# closed-form per-comparison alpha adjustment, and a joint-bootstrap
# critical value that accounts for correlation between comparisons and is
# substituted for the canonical CI's marginal normal quantile -- the two
# options a from-scratch multiplicity-correction analysis would reach for,
# per Gemini's Sidak/bootstrap-scaling suggestion this mode's docstring
# discusses. Named plain "sidak"/"boot" (not e.g. "tango_sidak") since which
# base CI they widen is now scenario-dependent, not fixed to one method.
# CORR_BOOT (not CORR_SIDAK) is reused as-is by MULTIARM_CORRECTION_METHODS
# above -- see its comment for the p-value-side analogue.
# ---------------------------------------------------------------------------
CORR_SIDAK = Method("sidak", "#31a354")
#: `boot`, but with the joint level calibrated against the per-pair CI
#: formula's OWN finite-sample behaviour instead of the nominal normal
#: quantile -- see evalstats.core.paired._calibrated_joint_critical_value.
#: Exists because `boot`'s alpha_eff step assumes ci_func(., a) covers
#: exactly 1-a, which Bonett-Price does not (delta up to +4.3pp at n=10).
CORR_BOOT_CAL = Method("boot_cal", "#756bb1")
CANONICAL_SIMULTANEOUS_CI_METHODS = [CORR_SIDAK, CORR_BOOT, CORR_BOOT_CAL]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- evalstats.tests wrapper names (PPI-corrected path),
# ported from sim_type_i_calibration.py's TEST_NAMES. Each is run twice per
# scenario (uncorrected -- the scipy-equivalent test on LLM-only scores -- and
# PPI-corrected -- the same test with sparse human labels) to check whether
# PPI correction fixes the Type-I inflation judge bias/miscalibration causes.
# WILCOXON is shared with the pairwise non-PPI registry above: same
# underlying paired-difference test, just on different data structures.
# PAIRED_T is likewise shared with the pairwise non-PPI registry above: the
# mean-based paired-difference sibling to WILCOXON's median-based one --
# also the entry point for binary (a proportion is just a mean), alongside
# TTEST/TTEST_WELCH -- see scenarios.synthetic's binary judge-bias comment.
# BAYES_BOOTSTRAP is shared with the bootstrap-family registry above: the
# same paired-mean-difference estimand as PAIRED_T, but PPI-corrected via a
# Dirichlet-weighted (Bayesian) bootstrap instead of evalstats.ppi.correct's
# classical one -- see evalstats.tests._ppi_paired_bayes_bootstrap -- kept
# as a validated alternative, not a recommended default (real-data testing
# found it underperforms; PAIRED_T remains the reasonable default for
# binary p-values). BOOTSTRAP_T is likewise shared: the same paired-mean
# estimand, PPI-corrected via a studentized-bootstrap pivot generalizing
# evalstats.core.resampling.bootstrap_t_ci_1d's per-replicate SE to PPI's
# two-term variance -- see evalstats.tests._ppi_paired_bootstrap_t. Numeric
# (continuous/likert/grades) only -- unlike PAIRED_T/BAYES_BOOTSTRAP, not
# extended to binary, since bootstrap_t's value is specifically for
# resampling-based CI estimation on numeric data at N>=50 (ci_paired.py).
# MJ_FLOOR (reusing ci_paired's existing "mj_floor" Method instance) is the
# mirror image: binary paired data ONLY, not numeric -- PPI-corrects
# evalstats.core.resampling.mj_floor_paired_ci's score interval by substituting
# an effective-n derived from PPI's two-term variance into its Wilson-style
# shrinkage formula (see evalstats.tests._ppi_paired_mj_floor); fully
# closed-form, no bootstrap resampling.
# ---------------------------------------------------------------------------
TTEST = Method("ttest", "#1f77b4")
TTEST_WELCH = Method("ttest_welch", "#d62728")
# MJ_FLOOR_FIXED_LAMBDA (evalstats.tests._ppi_paired_mj_floor(..., power_tune=False)):
# the legacy fixed-lambda=1 rectifier MJ_FLOOR itself used before PPI++'s
# closed-form variance-minimizing lambda* became the default -- the same
# derivation _analytic_mean_correct/_analytic_logit_t_correct already use
# for ppi_t_interval/ppi_logit_t (this estimand, mean(a_i - b_i), is
# identical to theirs; only Tango's Wilson-style effective-n CI shape
# differs). Kept selectable for direct comparison and for reproducing
# pre-flip results, not because it's still recommended: power_tune=True
# gives a meaningfully narrower mean CI width with no coverage cost at
# MCAR or MNAR labeling, and neither setting showed calibration problems
# in validation.
MJ_FLOOR_FIXED_LAMBDA = Method("mj_floor_fixed_lambda", "#41b6c4")  # teal -- distinct from MJ_FLOOR's default grey
# MWU family: five PPI corrections for the same classical test (Mann-Whitney
# U / independent two-group mid-rank estimand P_mid(A>B)-0.5), matching
# MWU is evalstats.tests.mannwhitney's only PPI correction (the global
# rectifier). Four local-rectifier variants -- mwu_mnar_experimental,
# mwu_mnar_pooled, mwu_adaptive, mwu_ridge -- were REMOVED on 2026-08-21:
# none was ever in PPI_OFFICIAL_TEST_METHODS or exercised by a single unit
# test, and all three local-rectifier constructions proved badly broken on
# binary data even under plain MCAR (coverage 0.00-0.06 at a real effect vs
# MWU's 0.989; see evalstats.tests._ppi_kruskal_wallis_pairwise_mnar_experimental's
# docstring for the mechanism). mannwhitney's "method" parameter went with
# them.
MWU = Method("mwu", "#2ca02c")
ANOVA_IND = Method("anova_ind", "#e6550d")
ANOVA_REP = Method("anova_rep", "#fd8d3c")
FRIEDMAN = Method("friedman", "#756bb1")  # purple -- distinct from the anova_*/lmm_* families
# KRUSKAL/KRUSKAL_MNAR_EXPERIMENTAL: two PPI corrections for the same
# omnibus test -- the two-group global-vs-local rectifier story generalized one level up
# (k independent groups instead of 2) -- see
# evalstats.tests.kruskalwallis's "method" docstring for the full
# mechanism/tradeoff. KRUSKAL="global" (the default, global rectifier),
# KRUSKAL_MNAR_EXPERIMENTAL="mnar_experimental" (local rectifier: fixes MNAR
# labeling at the cost of MCAR calibration, kept for direct comparison and
# for anyone deliberately studying MNAR robustness). Same color convention
# convention: the default occupies the original primary shade, the
# alternate gets a lighter tint.
KRUSKAL = Method("kruskal", "#e377c2")  # pink -- distinct from the anova_*/lmm_* families
KRUSKAL_MNAR_EXPERIMENTAL = Method("kruskal_mnar_experimental", "#f2b6d4")  # lighter tint
# KRUSKAL_ROWSUM/KRUSKAL_ROWSUM_LABELED: EXPERIMENTAL. Not another rectifier
# -- the SAME corrected pairwise vector and covariance as KRUSKAL, Wald-tested
# on its (k-1)-dimensional weighted row-sum projection, which is exactly the
# part classical Kruskal-Wallis's mean pooled ranks are an affine function of
# (see evalstats.tests._ppi_kruskal_wallis_rowsum). So KRUSKAL *replaces* the
# classical statistic and these two *correct* it. "_labeled" weights the
# projection by labeled counts instead of full group sizes; the two are
# bit-identical under a balanced design and only diverge when per-group label
# fractions differ. Same colour convention: lighter tints of KRUSKAL's pink.
KRUSKAL_ROWSUM = Method("kruskal_rowsum", "#c2559c")           # darker pink
KRUSKAL_ROWSUM_LABELED = Method("kruskal_rowsum_labeled", "#f7d6e8")  # palest tint
# KRUSKAL_TWOPART/KRUSKAL_EIGENGAP: EXPERIMENTAL candidates
# for the k>=5 conservatism of KRUSKAL itself (its df counts C(k,2) directions
# when Cov(delta_hat) is rank k-1 under H0 -- see
# evalstats.tests._kw_contrast_subspace and REPORT.md sections B-C). Same one
# bootstrap as KRUSKAL, different test form only. Not in the official set.
# The contrast-space Wald itself is KRUSKAL_ROWSUM -- there is deliberately
# no separate 'kruskal_contrast': an unweighted contrast basis and the
# n-weighted row space calibrate identically and only the weighted one is
# the classical KW contrast (see _kw_contrast_subspace).
KRUSKAL_TWOPART = Method("kruskal_twopart", "#d98cbb")     # mid tint
KRUSKAL_EIGENGAP = Method("kruskal_eigengap", "#8c5f7d")   # muted plum
# KRUSKAL_INFLUENCE: the phase-4 candidate -- the SAME estimator as KRUSKAL,
# with the Wald covariance replaced by a null-structured influence-function one
# (evalstats.tests._kw_influence_cov). Fixes the df defect by construction
# (rank k-1, not by truncation) and the variance-assembly conditioning defect
# (per-item differencing). Opt-in until validated on ppi_real.
KRUSKAL_INFLUENCE = Method("kruskal_influence", "#7b3f9c")  # violet
# KRUSKAL_INFLUENCE_LOGO: influence covariance + the two corner corrections
# (leave-one-group-out reference ECDFs, and a floor on each group's labeled
# composite variance). See REPORT.md section E.
KRUSKAL_INFLUENCE_LOGO = Method("kruskal_influence_logo", "#4a2d6b")  # deep violet
# KRUSKAL_INFLUENCE_FLOOR: the variance floor WITHOUT the LOGO ECDFs. The floor
# binds only when a group's labeled composite variance collapses, which real
# judge noise ratios (0.64-1.18 on privacy_judge) should never trigger -- so
# this is the candidate that fixes the sparse-Likert corner while being
# provably inert on real data. Inertness is checked by counting binding
# events (evalstats.tests._KW_FLOOR_AUDIT), not by comparing rejection rates.
KRUSKAL_INFLUENCE_FLOOR = Method("kruskal_influence_floor", "#9c5fbf")  # light violet
LMM = Method("lmm", "#74c476")
LMM_FACTORIAL = Method("lmm_factorial", "#a1d99b")
LMM_RUNS = Method("lmm_runs", "#c7e9c0")
PPI_TEST_METHODS = [
    TTEST, TTEST_WELCH, MWU, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, MJ_FLOOR, MJ_FLOOR_FIXED_LAMBDA, PPI_BONETT_PRICE, ANOVA_IND,
    ANOVA_REP, FRIEDMAN, KRUSKAL, KRUSKAL_MNAR_EXPERIMENTAL, KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
    KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE, KRUSKAL_INFLUENCE_LOGO,
    KRUSKAL_INFLUENCE_FLOOR,
    LMM, LMM_FACTORIAL, LMM_RUNS, PPI_WILSON,
    PPI_BOOTSTRAP_T_SINGLE, PPI_T_INTERVAL, PPI_LOGIT_T, PPI_T_INTERVAL_SINGLE, PPI_LOGIT_T_SINGLE,
]
"""Every PPI test method the harness knows how to run -- the full set
selectable via --tests. NOT what runs by default; see
PPI_OFFICIAL_TEST_METHODS for that."""
PPI_OFFICIAL_TEST_METHODS = [
    m for m in PPI_TEST_METHODS
    if m not in (
        KRUSKAL_MNAR_EXPERIMENTAL,
        # Experimental, opt-in via --tests kruskal_rowsum /
        # kruskal_rowsum_labeled: they answer "what does correcting the REAL
        # Kruskal-Wallis cost/buy", which is a study question, not part of
        # the shipped default set.
        KRUSKAL_ROWSUM, KRUSKAL_ROWSUM_LABELED,
        KRUSKAL_TWOPART, KRUSKAL_EIGENGAP, KRUSKAL_INFLUENCE, KRUSKAL_INFLUENCE_LOGO,
    KRUSKAL_INFLUENCE_FLOOR,
        LMM, LMM_FACTORIAL, LMM_RUNS, MJ_FLOOR_FIXED_LAMBDA,
        # The paired-binary PPI slot is PPI_BONETT_PRICE. MJ_FLOOR (and its
        # fixed-lambda sibling) remain implemented and selectable via
        # --tests, but are no longer part of the official sweep.
        MJ_FLOOR,
    )
]
"""The default (--tests unset) active-test set for --mode ppi -- every
PPI_TEST_METHODS entry except kruskal_mnar_experimental (it fixes real
MNAR-labeling miscalibration in its global-rectifier sibling, but costs real
MCAR calibration doing so -- see evalstats.tests.kruskalwallis's "method"
docstring). It remains selectable via --tests kruskal_mnar_experimental for
direct comparison or studying MNAR robustness deliberately. Its two-group
counterpart mwu_mnar_experimental, and the mwu_mnar_pooled/mwu_adaptive/
mwu_ridge variants, were removed entirely on 2026-08-21 -- see MWU's
comment above.

lmm/lmm_factorial/lmm_runs are excluded from the official set: not
currently part of the reported result set, so there's no point paying their
runtime cost (and reviewing their output) in every official pass. Still
fully selectable via --tests lmm/lmm_factorial/lmm_runs for anyone who
wants them.

PPI_WILSON/PPI_BOOTSTRAP_T_SINGLE (the single-sample robustness-CI methods
PPI_AUTO_METHOD_TABLE routes to) are part of the default set too --
pvalues.py's synthetic PPI sweep has a single-arm effect-check scenario
(see cases/pvalues.py's _run_ppi_effect_cell single-arm blocks) so these
aren't validated only by cases/ppi_real.py's real-data check."""

# ---------------------------------------------------------------------------
# Registry -- canonical ordering for tables/legends, and name -> Method lookup
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Between-subjects (unpaired) pairwise methods -- cases/ci_unpaired.py
# ---------------------------------------------------------------------------
# Two DIFFERENT estimands live here, and they are not interchangeable:
#
#   Delta-mean / Delta-p  : mean(A) - mean(B). What compare(design="unpaired")
#                           reports today for BINARY score types (via Welch's
#                           t on the 0/1 values -- the linear-probability-model
#                           patch documented in config.AUTO_UNPAIRED_METHOD_TABLE).
#   theta = P(A > B) + .5 P(A = B) : stochastic dominance. What
#                           compare(design="unpaired") reports today for
#                           CONTINUOUS/LIKERT/GRADE, via the Mann-Whitney /
#                           Kruskal-Wallis path.
#
# A method's coverage target therefore differs by family: mean-family methods
# are scored against the source's true_diff, theta-family methods against a
# Monte-Carlo-estimated true theta. cases/ci_unpaired.py records which
# estimand each row belongs to so the two are never averaged together.
WELCH_T = Method("welch_t", "#17becf")
STUDENT_T = Method("student_t", "#bcbd22")
WALD_UNPAIRED = Method("wald_unpaired", "#7f7f7f")  # grey: the naive baseline, as elsewhere in this file
AGRESTI_CAFFO = Method("agresti_caffo", "#98df8a")
NEWCOMBE_HYBRID = Method("newcombe_hybrid", "#c5b0d5")
MIETTINEN_NURMINEN = Method("miettinen_nurminen", "#ff9896")
BAYES_BETA_INDEP = Method("bayes_beta_indep", "#f7b6d2")

MOVER_T = Method("mover_t", "#969696")
MOVER_LOGIT_T = Method("mover_logit_t", "#31a354")
MOVER_NIG = Method("mover_nig", "#756bb1")

UNPAIRED_MEAN_EXTRA_METHODS = [WELCH_T, STUDENT_T, MOVER_T, MOVER_LOGIT_T, MOVER_NIG]
"""Applies to every eval type (binary included -- Welch's t on 0/1 is exactly
what the shipped unpaired binary path does today).

mover_t is the CONTROL, not a candidate: MOVER with plain t-interval arms.
Without it a win for mover_logit_t over welch_t is uninterpretable, because
those two differ in BOTH the combination rule and the arm interval. mover_t
holds the arm fixed at a t-interval and varies only the combination rule, so
the two comparisons together separate the effects. It also covers the paired
table's fourth row (unbounded -> t_interval).

A mover_el (empirical-likelihood arms) variant was tried and removed, noted
here so it is not re-added: it is invalid on binary (EL collapses to the
degenerate interval [1, 1] on a constant sample), and on continuous/likert it
was mid-pack on coverage while costing ~30x its MOVER siblings per call
(2.2 ms vs 0.07-0.10 ms). It never won a column, so it bought nothing for the
runtime.

mover_logit_t / mover_nig are the unpaired siblings of the PAIRED path's own
recommendations (config.AUTO_ANALYZE_METHOD_TABLE routes bounded_01 -> logit_t
and likert -> nig): the same shipped one-sample interval is built per arm and
the two are combined by MOVER. Included so the unpaired recommendation can be
consistent with the paired one rather than an unrelated method family."""

AGRESTI_MIN = Method("agresti_min", "#d6616b")

UNPAIRED_BINARY_METHODS = [
    WALD_UNPAIRED, AGRESTI_CAFFO, NEWCOMBE_HYBRID, MIETTINEN_NURMINEN, BAYES_BETA_INDEP,
    AGRESTI_MIN,
]
"""Binary-only Delta-p intervals from the two-independent-proportions
literature. None of these are shipped by evalstats today; this is the
candidate slate the ci_unpaired sweep exists to adjudicate."""

# A dominance-probability (theta) family -- theta_bootstrap, theta_bca,
# brunner_munzel, brunner_munzel_logit -- was built here and removed.
# theta = P(A>B) + .5 P(A=B) is what compare(design="unpaired") currently
# reports for continuous/likert via the Kruskal-Wallis post-hoc, so measuring
# it looked like due diligence. It is a DIFFERENT ESTIMAND from the mean
# difference every other recommendation in this project is stated in, which
# makes its coverage and width numbers incomparable with the rest of the
# table, and it cost ~65% of the sweep's runtime to produce them. If the
# shipped theta path ever needs calibrating, it needs its own case, not a
# second estimand bolted onto this one.

UNPAIRED_METHODS = UNPAIRED_MEAN_EXTRA_METHODS + UNPAIRED_BINARY_METHODS


REPORT_METHOD_ORDER: list[Method] = BOOTSTRAP_METHODS + [
    T_INTERVAL, WILSON, JEFFREYS, NEWCOMBE_MOVER, MJ_FLOOR, TANGO_SCC,
    WALD, CLOPPER_PEARSON, BAYES_SINGLE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP,
] + CONTINUOUS_EXTRA_METHODS + [LOGIT_T_2ND] + DITHER_EXTRA_METHODS + NESTED_METHODS + BINARY_FLAT_METHODS + BINARY_NESTED_METHODS + (
    PAIR_DIFF_NESTED_METHODS
    + [MJ_FLOOR_FLAT, NEWCOMBE_FLAT, BONETT_PRICE_FLAT] + BINARY_PAIR_NESTED_METHODS
    + [TANGO_EXACT, MJ_UNFLOORED, BONETT_PRICE]
) + [
    MCNEMAR, MCNEMAR_MIDP, PERMUTATION, SIGN_TEST, NEWCOMBE_PVAL, BAYES_BINARY, WILCOXON, PAIRED_T, PPI_T_INTERVAL, PPI_LOGIT_T,
    PPI_WILSON, PPI_BONETT_PRICE, PPI_BOOTSTRAP_T_SINGLE, PPI_T_INTERVAL_SINGLE, PPI_LOGIT_T_SINGLE,
] + MULTIARM_CORRECTION_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS + [
    TTEST, TTEST_WELCH, MWU, MJ_FLOOR_FIXED_LAMBDA,
    ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL, KRUSKAL_MNAR_EXPERIMENTAL,
    LMM, LMM_FACTORIAL, LMM_RUNS,
] + UNPAIRED_METHODS

METHODS_BY_NAME: dict[str, Method] = {m.name: m for m in REPORT_METHOD_ORDER}


def get_method(name: str) -> Method:
    """Look up a Method by name, falling back to a default-colored Method for unknown names."""
    return METHODS_BY_NAME.get(name, Method(name))


def get_method_color(name: str) -> str:
    return get_method(name).color


def order_present_methods(present_names: set[str]) -> list[Method]:
    """Filter REPORT_METHOD_ORDER down to methods actually present, preserving canonical order.

    Raises on a method that was computed but never registered in
    REPORT_METHOD_ORDER. Previously such a method was silently dropped, so it
    would burn simulation time and then produce zero rows in every table and
    plot with no diagnostic -- a failure that looks like "the sweep skipped my
    method" rather than "the registry is missing an entry".
    """
    known = {m.name for m in REPORT_METHOD_ORDER}
    unregistered = sorted(present_names - known)
    if unregistered:
        raise KeyError(
            f"methods present in results but absent from REPORT_METHOD_ORDER: "
            f"{unregistered}. Add them to REPORT_METHOD_ORDER in "
            f"simulations/harness/methods.py, or they will not appear in any "
            f"table or plot."
        )
    return [m for m in REPORT_METHOD_ORDER if m.name in present_names]
