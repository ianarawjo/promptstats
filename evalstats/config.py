"""Global configuration for evalstats."""

import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

# We use a global variable to store the default alpha for CI analyses,
# which can be set by the user via set_alpha_ci() and is used by default in
# all CI analyses across the library (but can be overridden on a per-analysis basis
# by passing an explicit alpha).
_alpha: float = 0.05

# Alpha levels used to build the gradient CI bands in terminal plots.
# Ordered widest→narrowest by CI: alpha 0.32, 0.10, 0.05, 0.01 give the
# 68%, 90%, 95%, and 99% intervals. The terminal legend prints them as
# [99%/95%/90%/68%]; keep this list and that legend in step.
GRADIENT_CI_ALPHAS: tuple[float, ...] = (0.32, 0.10, 0.05, 0.01)

# Hard minimum sample floor: below this many items per compared entity,
# evalstats refuses to report statistics at all (too noisy to be
# meaningful -- see the paper's "enforce a minimum sample floor" principle).
# Enforced at the top of both compare() (api.py) and the `evalstats analyze`
# CLI (cli.py), before any analysis runs.
MIN_SAMPLE_FLOOR: int = 15

# Default seed for every resampling step downstream of compare(): bootstrap
# CIs, the PPI bootstrap, permutation nulls. Fixed so that the same input
# gives the same output -- a user passing no rng= reasonably expects a
# deterministic answer, and before this the PPI omnibus p wandered in the
# third decimal between identical calls (sd 0.0025 on p=0.388 over 12 runs).
#
# The trade this makes: Monte-Carlo variability is now invisible unless asked
# for. A p-value sitting right at alpha is still seed-dependent; it just looks
# stable. Pass rng=None explicitly for a fresh nondeterministic draw per call,
# or sweep rng=0,1,2,... to see the spread.
DEFAULT_RNG_SEED: int = 37


def supports_ansi_color() -> bool:
    """Whether ANSI color escape codes should be emitted for console output.

    True for a real terminal. ``sys.stdout.isatty()`` is False for
    Jupyter's ipykernel output stream, but JupyterLab/Notebook still render
    ANSI codes (converted to HTML) in the output cell, so it's detected
    separately here rather than being treated like a plain redirected/piped
    stream. Respects the NO_COLOR / FORCE_COLOR conventions.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if sys.stdout.isatty():
        return True
    return type(sys.stdout).__module__.startswith("ipykernel")


# ---------------------------------------------------------------------------
# method="auto" resolution
# ---------------------------------------------------------------------------
# Every public entry point that accepts method="auto" (analyze(),
# resolve_resampling_method(), the max-T simultaneous-CI path, ...)
# resolves to a concrete method by consulting the tables below. Keeping
# them here means the full "what runs automatically, and when" surface is
# visible in one place instead of scattered across router.py / resampling.py
# / paired.py.
#
# Note: this does *not* cover the R >= 3 "seeded" threshold (whether a
# two-level nested bootstrap is used), which is a structural property of
# the input data shape (see BenchmarkResult.is_seeded in core/types.py)
# rather than a method-selection choice.

DataKind = Literal["binary", "bounded_01", "likert", "unbounded"]

# --- Bootstrap resampling variant (resolve_resampling_method) --------------
# Plain (non-binary) bootstrap CIs: sample_size >= this -> "bootstrap"
# (simpler and at least as accurate at that scale); below it -> "bootstrap_t".
BOOTSTRAP_AUTO_MIN_N: int = 200

# --- max-T simultaneous CI bootstrap variant --------------------------------
# Bonferroni is the default simultaneous-CI construction (see
# _simultaneous_cis_router in evalstats/core/paired.py); max-T is opt-in via
# prefer="max_t". When max-T IS requested, 'auto' always resolves to this
# variant regardless of sample size — max-T's studentization needs a stable
# per-replicate SE estimate, which the studentized bootstrap provides more
# robustly at all N.
MAX_T_AUTO_METHOD: str = "bootstrap_t"


@dataclass(frozen=True)
class AutoAnalyzeRule:
    """One row of the ``analyze(method="auto")`` routing matrix.

    A rule applies when the observed data matches ``data_kind`` and (for
    binary data) the per-template sample size ``N`` is below ``max_n``
    (``None`` = no upper bound, i.e. applies at any N).
    """
    data_kind: DataKind
    max_n: Optional[int]
    pairwise_method: str
    robustness_method_single_run: str
    robustness_method_seeded: str
    reason: str


# Ordered (N, data-kind) x (single-run, seeded) matrix used by analyze() to
# resolve method="auto". Read as: for this data_kind, when N < max_n, use
# this pairwise_method; the robustness (single-sample marginal CI) method
# additionally depends on whether the benchmark is seeded (R >= 3 runs).
#
# This table is the code-level source of truth for the paper's CI
# decision-tree figure (fig:ci-decision-tree) -- every row below should have
# a matching leaf there. "Wilson flat" / "Logit-t on run means" in that
# figure are not separate implementations: robustness_metrics() already
# collapses (N, M, R) score arrays to per-input cell means before dispatching
# on marginal_method, so plain "wilson" / "logit_t" applied to that
# already-collapsed array *is* the flat/run-means variant. Same story for
# "Logit-t on run mean differences" in all_pairwise()'s "logit_t" path. And
# the binary pairwise multirun variant is not a separate user-facing method
# name -- pairwise_method="bonett_price" internally detects R >= 3 seeded runs
# and switches to the clustered multirun variant
# (bonett_price_paired_ci_multirun_cluster). mj_floor and its own multirun
# variants remain available in core/resampling.py and callable by name, but
# are no longer what "auto" routes to for binary data.
#
# "bounded_01" (the data_kind label) no longer means the data is literally
# valued in [0, 1] -- it means router.py._analyze_single could establish a
# reliable [lo, hi] range for it via resolve_score_bounds() (core/
# resampling.py): either the caller's explicit score_range, or an exact
# [0, 1] match (still emits a UserWarning, since it's still an inference).
# There is deliberately no third option that approximates a range from the
# sample's own min/max -- that's not a safe substitute for a metric's true
# theoretical bounds (e.g. a 1-5 Likert scale sampled only between 2 and 4
# would silently produce a miscalibrated CI). Numeric data outside [0, 1]
# with no score_range given falls through to the "unbounded" row instead
# (also with a UserWarning, recommending an explicit score_range). Named
# "unbounded", not "continuous" -- the simulation harness's own eval_type
# taxonomy already uses "continuous" for a different concept (a bounded
# Beta-distributed shape), and reusing the same word here for an entirely
# different meaning (numeric data with NO reliable range at all) was a
# recurring source of confusion between the package and the harness.
AUTO_ANALYZE_METHOD_TABLE: tuple[AutoAnalyzeRule, ...] = (
    AutoAnalyzeRule(
        data_kind="binary", max_n=None,
        pairwise_method="bonett_price",
        robustness_method_single_run="wilson",
        robustness_method_seeded="wilson",
        reason=(
            "Binary data at every N: Bonett & Price (2012) adjusted-Wald "
            "pairwise, Wilson-flat marginal. This is a SINGLE row where there "
            "used to be two (Bayesian paired below N=50, mj_floor above) -- "
            "bonett_price is best-calibrated across the whole range, so the "
            "decision tree loses the split rather than just swapping a name. "
            "Its Laplace adjustment (two pseudo-items) keeps the interval "
            "well-behaved in the dominated/jointly-sparse pairs where the "
            "score-interval form under-covers, which is what motivated the "
            "old small-N branch in the first place. Seeded (R >= 3) data "
            "dispatches to the clustered multirun variant automatically -- "
            "see core/paired.py's bonett_price branch. Marginal CIs use plain "
            "Wilson regardless of seeding ('Wilson flat' in "
            "fig:ci-decision-tree)."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="bounded_01", max_n=None,
        pairwise_method="logit_t",
        robustness_method_single_run="logit_t",
        robustness_method_seeded="logit_t",
        reason=(
            "Numeric data with a reliable [lo, hi] range and no detected "
            "quantization grid (e.g. normalised accuracy, ROUGE, or any "
            "genuinely continuous metric declared via an explicit "
            "score_range): Logit-t pairwise and marginal CIs, per "
            "fig:ci-decision-tree. The range is either the caller's "
            "explicit score_range or an exact [0, 1] match -- see "
            "resolve_score_bounds() in core/resampling.py. Supersedes the "
            "earlier t_interval (pairwise) / nig, nig_nested (marginal) "
            "defaults for data in this range -- except discrete/ordinal "
            "data (Likert scales, integer percentage grades), which is now "
            "routed to the separate 'likert' row below instead."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="likert", max_n=None,
        pairwise_method="nig",
        robustness_method_single_run="logit_t",
        robustness_method_seeded="logit_t",
        reason=(
            "Discrete/ordinal bounded data (a Likert scale, an integer "
            "percentage grade, or anything else with a real quantization "
            "grid within its known [lo, hi] range) -- detected either from "
            "an explicit eval_type='likert', or auto-detected via "
            "detect_quantization_step() (core/resampling.py) when no "
            "eval_type is given, with a UserWarning explaining the switch. "
            "Uses NIG rather than logit-t for the pairwise case, for both "
            "single-run and seeded/multi-run data: a paired diff of two "
            "highly correlated Likert arms can lose real variance to "
            "rounding cancellation (most items round identically in both "
            "arms, only boundary-adjacent items differ), which at small N "
            "can leave the sample's diffs literally constant even though "
            "the true population diff variance is nonzero -- collapsing a "
            "variance-based CI like logit-t's. NIG's shrinkage prior "
            "protects against this without needing dithering/reconstruction. "
            "The k>=3 simultaneous/family-wise construction "
            "(core.paired._simultaneous_cis_router) widens NIG instead of "
            "logit-t for likert data for the same reason.\n\n"
            "Not extended to marginal/robustness CIs (the "
            "'nig'/'nig_nested' single-sample case in core/variance.py's "
            "robustness_metrics()) -- logit-t remains the default there, "
            "and for genuinely continuous 'bounded_01' data everywhere, "
            "where NIG's extra conservatism buys no corresponding "
            "robustness."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="unbounded", max_n=None,
        pairwise_method="t_interval",
        robustness_method_single_run="t_interval",
        robustness_method_seeded="t_interval",
        reason=(
            "Numeric data outside [0, 1] with no explicit score_range -- "
            "evalstats deliberately does not guess a [lo, hi] range from the "
            "sample's own min/max (unreliable; see resolve_score_bounds()), "
            "so this falls back to a plain t-interval, with a UserWarning "
            "recommending the caller pass score_range explicitly to get "
            "Logit-t instead (not covered by fig:ci-decision-tree, which "
            "assumes a known eval-metric range)."
        ),
    ),
)


def resolve_auto_analyze_methods(
    data_kind: DataKind, n: int, seeded: bool,
) -> tuple[str, str]:
    """Resolve ``analyze(method="auto")`` to concrete (pairwise, robustness) methods.

    Looks up :data:`AUTO_ANALYZE_METHOD_TABLE` for the first rule matching
    ``data_kind`` and ``n``, in table order.

    Parameters
    ----------
    data_kind : "binary", "bounded_01", or "unbounded"
        Detected data type (see ``is_binary_scores`` / ``is_bounded_01_scores``
        in ``core.resampling``).
    n : int
        Per-template sample size (number of inputs).
    seeded : bool
        Whether the benchmark carries R >= 3 runs (nested bootstrap path).

    Returns
    -------
    tuple[str, str]
        ``(pairwise_method, robustness_method)``.
    """
    for rule in AUTO_ANALYZE_METHOD_TABLE:
        if rule.data_kind != data_kind:
            continue
        if rule.max_n is not None and n >= rule.max_n:
            continue
        robustness = rule.robustness_method_seeded if seeded else rule.robustness_method_single_run
        pairwise = rule.pairwise_method
        return pairwise, robustness
    raise AssertionError(
        f"no AUTO_ANALYZE_METHOD_TABLE rule matched data_kind={data_kind!r}, n={n}"
    )


@dataclass(frozen=True)
class PPIAutoMethodRule:
    """One row of the PPI-alignment-correction ``method="auto"`` routing table.

    A separate table from :data:`AUTO_ANALYZE_METHOD_TABLE`: the non-aligned
    auto default for a given ``data_kind`` (e.g. ``"t_interval"`` for
    continuous data) does not necessarily have a validated PPI-corrected
    counterpart, so PPI alignment correction resolves ``"auto"`` to whichever
    method *does* have one, instead of reusing the non-aligned default and
    failing.
    """
    data_kind: DataKind
    pairwise_method: str
    robustness_method: str
    reason: str


# PPI alignment correction requires N >= 50 (enforced in
# evalstats.api._run_alignment_ppi), so there is no small-N branch here the
# way AUTO_ANALYZE_METHOD_TABLE has one for binary data.
PPI_AUTO_METHOD_TABLE: tuple[PPIAutoMethodRule, ...] = (
    PPIAutoMethodRule(
        data_kind="binary",
        pairwise_method="bonett_price",
        robustness_method="wilson",
        reason=(
            "Binary data: bonett_price (pairwise) and Wilson (marginal) both "
            "have closed-form PPI-corrected forms via an effective-n "
            "substitution (see evalstats.tests._ppi_paired_bonett_price / "
            "_ppi_single_wilson). Bonett-Price's Laplace adjustment keeps the "
            "interval well-behaved when the labeled subset carries little "
            "discordance information, where the score-interval form collapses "
            "toward zero width. "
            "Wilson matches the non-aligned default's own marginal choice "
            "(AUTO_ANALYZE_METHOD_TABLE's marginal is 'wilson' at every N). "
            "Pairwise is bonett_price even below the non-aligned default's "
            "N<50 cutoff for bayes_binary -- a forced deviation, not a choice: "
            "bayes_binary has no PPI-corrected form, so bonett_price is used "
            "at every N under PPI alignment rather than raising below N=50."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="bounded_01",
        pairwise_method="ppi_logit_t",
        robustness_method="ppi_logit_t",
        reason=(
            "Numeric [0, 1]-bounded data: closed-form (no-bootstrap) PPI-corrected "
            "logit-t (see evalstats.tests._ppi_paired_logit_t / "
            "_ppi_single_logit_t, wrapping evalstats.ppi._analytic_logit_t_correct), "
            "matching the non-aligned default's own logit_t choice for this "
            "data_kind (see AUTO_ANALYZE_METHOD_TABLE)."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="likert",
        pairwise_method="ppi_logit_t",
        robustness_method="ppi_logit_t",
        reason=(
            "Discrete/ordinal bounded data: there is no PPI-corrected NIG "
            "implementation (NIG's win over logit-t for likert is specific "
            "to the non-aligned/no-labels path -- see AUTO_ANALYZE_METHOD_"
            "TABLE's 'likert' row), so this falls back to the same "
            "ppi_logit_t used for 'bounded_01' rather than raising."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="unbounded",
        pairwise_method="ppi_t_interval",
        robustness_method="ppi_t_interval",
        reason=(
            "Numeric data with no reliable [lo, hi] range: closed-form "
            "(no-bootstrap) PPI-corrected t-interval (see evalstats.tests."
            "_ppi_paired_t_interval / _ppi_single_t_interval, wrapping "
            "evalstats.ppi._analytic_mean_correct), matching the non-aligned "
            "default's own t_interval fallback for this data_kind (see "
            "AUTO_ANALYZE_METHOD_TABLE) -- logit_t requires known bounds to do "
            "the logit transform at all, which this data_kind by definition lacks."
        ),
    ),
)


def resolve_ppi_auto_methods(data_kind: DataKind) -> tuple[str, str]:
    """Resolve PPI alignment correction's ``method="auto"`` to concrete
    ``(pairwise_method, robustness_method)``, for use by
    ``evalstats.api._run_alignment_ppi``.

    Raises
    ------
    ValueError
        If no PPI-corrected method is defined for ``data_kind``.
    """
    for rule in PPI_AUTO_METHOD_TABLE:
        if rule.data_kind == data_kind:
            return rule.pairwise_method, rule.robustness_method
    raise ValueError(
        f"No PPI-corrected auto method is defined for data_kind={data_kind!r}."
    )


# ---------------------------------------------------------------------------
# FWER correction auto-routing (fig:fwer-decision-tree)
# ---------------------------------------------------------------------------
# Governs the k>=3 "family of comparisons" branch of the paper's FWER
# decision tree: which simultaneous-CI construction (widens each pairwise
# CI) and which p-value-correction procedure (adjusts each pairwise p-value)
# evalstats picks automatically. The k=2 "single pairwise comparison" branch
# (Wilcoxon signed-ranks, unconditionally -- no FWER control needed) is a
# separate fix in compare()'s pairwise_test="auto" resolution, not covered
# by either table here.
#
# Both tables key off a *lopsided_binary* flag (see
# core.resampling.is_lopsided_binary: any compared group has fewer than 5
# observed instances of its rarer 0/1 outcome) which forces the small-N
# branch regardless of N or k -- resampling-based corrections (joint
# bootstrap, Romano-Wolf) can misbehave when one outcome is that sparse,
# while Sidak/Shaffer's closed-form adjustments stay reliable.
#
# The tree only distinguishes "binary" vs "numeric" data (not bounded_01 vs
# unbounded separately) for the simultaneous-CI table -- both non-binary
# DataKind values map to the "numeric" row below.


@dataclass(frozen=True)
class AutoSimultaneousCIRule:
    """One row of the simultaneous-CI ``prefer="auto"`` routing matrix."""
    data_kind: Literal["binary", "numeric"]
    max_n: Optional[int]
    method: str  # "sidak" or "boot" (joint bootstrap with effective alpha)
    reason: str


AUTO_SIMULTANEOUS_CI_METHOD_TABLE: tuple[AutoSimultaneousCIRule, ...] = (
    AutoSimultaneousCIRule(
        data_kind="binary", max_n=None,
        method="sidak",
        reason=(
            "Binary data, every N: Sidak. See the numeric rule below -- the "
            "reasoning is not data-kind specific, and binary is where the "
            "joint bootstrap failed hardest (worst-case family coverage 0.50 "
            "at n=15 and 0.74 at n=30 on the expanded scenario suite, against "
            "Sidak's 0.94)."
        ),
    ),
    AutoSimultaneousCIRule(
        data_kind="numeric", max_n=None,
        method="sidak",
        reason=(
            "Numeric data, every N: Sidak is the only construction whose "
            "worst-case family coverage held across the expanded scenario "
            "suite (min 0.913-0.943 for every eval type and N). The joint "
            "bootstrap ('boot') is better centred on average and 3-5%% "
            "narrower, but its worst case collapses (0.50 on sparse/lopsided "
            "binary at n=15) and it under-covers Likert at every N, since its "
            "alpha_eff step converts a bootstrap critical value through the "
            "normal cdf while the Likert pairwise formula (NIG) is a t "
            "interval.\n\n"
            "The width Sidak gives up is small and bounded: Tukey's "
            "studentized range is the optimal equal-width procedure for "
            "all-pairwise comparisons and beats Sidak by only 1.8-3.0%% "
            "(the shared-arm contrast correlation is close to the 0.5 that "
            "bound assumes), but Tukey needs normality/homoscedasticity "
            "(and sphericity for paired evals) that binary and Likert data "
            "violate. 'boot'/'boot_cal'/'max_t'/'bonferroni' all remain "
            "reachable via an explicit prefer= argument."
        ),
    ),
)


def resolve_auto_simultaneous_ci_method(
    data_kind: DataKind, n: int, *, lopsided_binary: bool = False,
) -> str:
    """Resolve simultaneous-CI ``prefer="auto"`` to a concrete method.

    Parameters
    ----------
    data_kind : "binary", "bounded_01", or "unbounded"
        Detected data type. The FWER tree only distinguishes "binary" vs
        "numeric" -- ``"bounded_01"``/``"unbounded"`` both map to the
        "numeric" row.
    n : int
        Per-entity sample size (number of items).
    lopsided_binary : bool
        When True, forces ``"sidak"`` regardless of N -- the tree's
        exception for a heavily skewed binary split, which applies
        "regardless of n or k".

    Returns
    -------
    str
        ``"sidak"`` or ``"boot"``.
    """
    tree_kind = "binary" if data_kind == "binary" else "numeric"
    if tree_kind == "binary" and lopsided_binary:
        return "sidak"
    for rule in AUTO_SIMULTANEOUS_CI_METHOD_TABLE:
        if rule.data_kind != tree_kind:
            continue
        if rule.max_n is not None and n >= rule.max_n:
            continue
        return rule.method
    raise AssertionError(
        f"no AUTO_SIMULTANEOUS_CI_METHOD_TABLE rule matched data_kind={tree_kind!r}, n={n}"
    )


@dataclass(frozen=True)
class AutoPValueCorrectionRule:
    """One row of the k>=3 p-value-correction ``correction="auto"`` routing matrix."""
    max_n: Optional[int]
    method: str  # "shaffer" or "romano_wolf"
    reason: str


AUTO_PVALUE_CORRECTION_METHOD_TABLE: tuple[AutoPValueCorrectionRule, ...] = (
    AutoPValueCorrectionRule(
        max_n=30, method="shaffer",
        reason="N < 30: Shaffer's modified step-down Holm procedure.",
    ),
    AutoPValueCorrectionRule(
        max_n=None, method="romano_wolf",
        reason=(
            "N >= 30: Romano-Wolf bootstrap step-down, which strictly "
            "dominates single-step corrections in power under positive "
            "correlation between comparisons -- the common case for "
            "repeated-measures/shared-item eval designs."
        ),
    ),
)


def resolve_auto_pvalue_correction_method(n: int, *, lopsided_binary: bool = False) -> str:
    """Resolve k>=3 p-value ``correction="auto"`` to a concrete method.

    ``lopsided_binary`` forces ``"shaffer"`` regardless of N -- the tree's
    binary-data exception for the p-value-correction branch.

    Returns
    -------
    str
        ``"shaffer"`` or ``"romano_wolf"``.
    """
    if lopsided_binary:
        return "shaffer"
    for rule in AUTO_PVALUE_CORRECTION_METHOD_TABLE:
        if rule.max_n is not None and n >= rule.max_n:
            continue
        return rule.method
    raise AssertionError(f"no AUTO_PVALUE_CORRECTION_METHOD_TABLE rule matched n={n}")


# ---------------------------------------------------------------------------
# Between-subjects (unpaired) design routing -- compare(design="unpaired")
# ---------------------------------------------------------------------------
# Separate from AUTO_ANALYZE_METHOD_TABLE above (which is paired-only): that
# table's data_kind taxonomy ("binary"/"bounded_01"/"likert"/"unbounded") is
# also different from the one used here ("binary"/"continuous"/"likert"/
# matching evalstats.loader._detect_score_type -- kept local rather
# than imported to avoid coupling this low-level module to the loader, same
# reasoning as DataKind above being declared locally rather than imported).
#
# Deliberately just two rows, decided after extensive discussion, not derived
# mechanically from AUTO_ANALYZE_METHOD_TABLE's finer-grained routing:
#
#   binary -> anova_oneway (omnibus) + ttest (pairwise, Welch's). The
#   textbook-correct test for comparing proportions is chi-square/Fisher's
#   exact, but neither has PPI correction machinery in this codebase, and
#   deriving one would be new, unvalidated statistical work. Treating a 0/1
#   outcome as a numeric mean and reusing the already-validated ttest/
#   anova_oneway PPI paths (the "linear probability model" approach) gives
#   Δp (proportion difference) with a CI -- the effect size a reader expects
#   for a binary outcome -- using entirely existing, validated machinery.
#   Known, accepted limitation: t-intervals on binary/bounded data can
#   produce out-of-[0,1]/[-1,1] CIs at extreme proportions or small N (why
#   the *paired* path uses mj_floor instead of a generic t-interval for binary
#   data -- there is no between-subjects Tango equivalent today). A
#   deliberate patch, not a clean solution.
#
#   continuous / likert -> kruskalwallis (omnibus + θ_ab pairwise
#   post-hoc) + mannwhitney (the k=2 special case -- Kruskal-Wallis reduces
#   to Mann-Whitney at k=2). Reports a stochastic-dominance probability
#   θ=P(a>b), not a mean difference -- less immediately interpretable for
#   continuous data than Δmean would be, but this is the only validated
#   multi-group (k>=3) pairwise mechanism in the codebase for any score
#   type; a Tukey-HSD-style joint mean-difference post-hoc for continuous
#   data does not exist and would itself be new, unvalidated work.
#   flagged as an assumption needing real-data validation, not a settled
#   choice (see PLAN §5).
UnpairedScoreType = Literal["binary", "continuous", "likert"]
UnpairedFamily = Literal["binary_proportion", "rank_based"]


@dataclass(frozen=True)
class AutoUnpairedRule:
    """One row of the ``compare(design="unpaired")`` routing table."""
    score_type: UnpairedScoreType
    family: UnpairedFamily
    omnibus_method: str    # "anova_oneway" or "kruskalwallis"
    pairwise_method: str   # "ttest" or "mannwhitney"
    reason: str


AUTO_UNPAIRED_METHOD_TABLE: tuple[AutoUnpairedRule, ...] = (
    AutoUnpairedRule(
        score_type="binary", family="binary_proportion",
        omnibus_method="anova_oneway", pairwise_method="ttest",
        reason=(
            "No PPI-corrected chi-square/Fisher's-exact exists in this "
            "codebase; treating the 0/1 outcome as a numeric mean and "
            "reusing the validated anova_oneway/ttest PPI paths reports "
            "the proportion difference a reader expects for a binary "
            "outcome, using entirely existing machinery. The non-PPI "
            "pairwise INTERVAL is Agresti-Caffo rather than Welch's t: "
            "exact enumeration over every (k_A, k_B) table puts Welch's "
            "worst-case coverage at 0.641, reached at p near 0 or 1 where "
            "binary eval data sits, against 0.930 for Agresti-Caffo at a "
            "narrower width (see core/unpaired._agresti_caffo_ci). This "
            "retires the 't-intervals on proportions misbehave at extreme "
            "values or small N' limitation previously noted here."
        ),
    ),
    AutoUnpairedRule(
        score_type="continuous", family="rank_based",
        omnibus_method="kruskalwallis", pairwise_method="mannwhitney",
        reason=(
            "Kruskal-Wallis omnibus with Mann-Whitney U post-hocs -- the "
            "pairing this project's PPI work validates. 'rank_based' names "
            "the tests, not the estimand: the reported effect and interval "
            "are the mean difference (Welch), matching the paired path and "
            "every other recommendation evalstats makes."
        ),
    ),
    AutoUnpairedRule(
        score_type="likert", family="rank_based",
        omnibus_method="kruskalwallis", pairwise_method="mannwhitney",
        reason=(
            "Ordinal data -- rank-based tests are the standard HCI "
            "convention, so Kruskal-Wallis/Mann-Whitney stay the tests. The "
            "reported effect is the mean difference (Welch interval); see "
            "the 'continuous' row for why the estimand is a mean."
        ),
    ),
)


def resolve_auto_unpaired_methods(score_type: str) -> tuple[UnpairedFamily, str, str]:
    """Resolve ``compare(design="unpaired")``'s routing to
    ``(family, omnibus_method, pairwise_method)`` -- see
    :data:`AUTO_UNPAIRED_METHOD_TABLE`.

    ``family`` is returned directly (not re-derived from ``pairwise_method``
    by the caller) so the table stays the actual source of truth for which
    engine runs -- editing a row here changes behavior, rather than editing
    ``omnibus_method``/``pairwise_method`` silently doing nothing because
    some other call site re-derives family from a hardcoded string check.

    The *k=2* special case (``mannwhitney``/``ttest`` used directly, no
    omnibus test needed since there's only one comparison) is handled by
    the caller (``evalstats.core.unpaired``), not this table.
    """
    for rule in AUTO_UNPAIRED_METHOD_TABLE:
        if rule.score_type == score_type:
            return rule.family, rule.omnibus_method, rule.pairwise_method
    raise AssertionError(
        f"no AUTO_UNPAIRED_METHOD_TABLE rule matched score_type={score_type!r}"
    )


def set_alpha_ci(alpha: float) -> None:
    """Set the default significance level used across all CI analyses.

    Parameters
    ----------
    alpha:
        Significance level (e.g. 0.05 for 95% CI, 0.01 for 99% CI).
        Must be in the open interval (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    global _alpha
    _alpha = alpha


def get_alpha_ci() -> float:
    """Return the current default significance level."""
    return _alpha
