"""Console summary formatters for analyze() results.

All terminal output produced by print_analysis_summary() lives here,
keeping the analysis router (router.py) free of display concerns.
"""

from __future__ import annotations

import re
from typing import Literal, Mapping, Optional, Union

import numpy as np

from .bundles import AnalysisBundle, MultiModelBundle
from .paired import PairedDiffResult, PairwiseMatrix
from .variance import SeedVarianceResult
from ..config import GRADIENT_CI_ALPHAS, get_alpha_ci, supports_ansi_color


# Sentinel used as a default so callers can distinguish "not passed" from
# "explicitly None (suppress p-values)".
_UNSET = object()

# ---------------------------------------------------------------------------
# ANSI color helpers (disabled when stdout is not a TTY or Jupyter kernel)
# ---------------------------------------------------------------------------

_ANSI = supports_ansi_color()

_RESET         = "\033[0m"  if _ANSI else ""
_BOLD          = "\033[1m"  if _ANSI else ""
_DIM           = "\033[2m"  if _ANSI else ""
_GREEN         = "\033[32m" if _ANSI else ""
_YELLOW        = "\033[33m" if _ANSI else ""
_CYAN          = "\033[36m" if _ANSI else ""
_BRIGHT_GREEN  = "\033[92m" if _ANSI else ""
_BRIGHT_YELLOW = "\033[93m" if _ANSI else ""
_BRIGHT_CYAN   = "\033[96m" if _ANSI else ""
_BRIGHT_RED    = "\033[91m" if _ANSI else ""
_BRIGHT_MAGENTA = "\033[95m" if _ANSI else ""  # "pink" -- reserved for the PPI/MCAR reminder, nothing else


def _p_best_color(p: float) -> str:
    """Return an opening ANSI code sequence for a P(Best) value.

    > 50%  → bold green (likely winner)
    < 5%   → dim (unlikely)
    else   → no color
    """
    if not _ANSI:
        return ""
    if p > 0.50:
        return _BOLD + _BRIGHT_GREEN
    if p < 0.05:
        return _DIM
    return ""


def _rank_method_label(bundle: "AnalysisBundle") -> str:
    """Return a short parenthetical note describing how ranks were computed.

    Mirrors the method-mapping in ``bootstrap_rank_distribution`` so the label
    reflects what was actually used, not the pairwise CI method.
    """
    method = (bundle.resolved_method or "bootstrap").lower()
    # Map pairwise methods that don't drive ranking to their bootstrap equivalent.
    if method in {"lmm", "permutation", "newcombe", "fisher", "sign", "bayes_binary"}:
        rank_method = "bootstrap"
    elif method == "bca":
        rank_method = "BCA bootstrap"
    elif method == "bayes_bootstrap":
        rank_method = "Bayes bootstrap"
    elif method == "smooth_bootstrap":
        rank_method = "smooth bootstrap"
    else:
        rank_method = "bootstrap"

    n = bundle.rank_dist.n_bootstrap
    first_result = next(iter(bundle.pairwise.results.values()), None)
    statistic = (first_result.statistic if first_result is not None else "mean").lower()
    if getattr(bundle, "ppi_applied", False):
        rank_method = f"PPI {rank_method}"
    return f"{rank_method}, n={n}, ranked by {statistic}"


def _uses_wilson_ci(bundle: "AnalysisBundle") -> bool:
    """Return True when single-sample CIs were computed with Wilson intervals."""
    method = (bundle.resolved_ci_method or "").lower()
    return method in {"wilson", "newcombe", "bayes_binary"}


def _pairwise_p_value_label(test_method: str) -> str:
    """Return a human-readable p-value method label for pairwise summaries."""
    method = test_method.lower()
    # All three binary paired CI methods report the same p-value, McNemar's
    # mid-p (see core.paired). Fagerland et al. (2014) sec. 9.1 recommend it
    # over the exact conditional test, which is markedly conservative.
    if "mj_floor" in method or "tango" in method or "newcombe" in method:
        return "McNemar mid-p"
    if "sign test" in method:
        return "paired sign test"
    if "wilcoxon" in method:
        return "Wilcoxon signed-rank"
    if "bootstrap" in method:
        return "bootstrap"
    return test_method


def _pairwise_display_pvalue(pair: PairedDiffResult) -> tuple[float, str]:
    """Choose the p-value shown in single-pair summaries.

    Default behavior is to display the Wilcoxon signed-rank p-value when
    available, while preserving exact-test paths (McNemar/sign test)
    where ``pair.p_value`` is the canonical inferential p-value.
    """
    method = pair.test_method.lower()
    is_exact_path = (
        "mj_floor" in method
        or "tango" in method
        or "newcombe" in method
        or "mcnemar" in method
        or "sign test" in method
    )
    if not is_exact_path and pair.wilcoxon_p is not None:
        return float(pair.wilcoxon_p), "Wilcoxon signed-rank"
    return float(pair.p_value), _pairwise_p_value_label(pair.test_method)


# ---------------------------------------------------------------------------
# Behavioral agreement helpers
# ---------------------------------------------------------------------------

def _agreement_bar(n11: int, n10: int, n01: int, n00: int, width: int = 20) -> str:
    """Build a proportional block-character bar for pass/fail agreement.

    Segments (left to right):
      █ (U+2588, FULL BLOCK)  — both pass (n11)
      ░ (U+2591, LIGHT SHADE) — both fail (n00)
      ▒ (U+2592, MEDIUM SHADE) — split/disagree (n10 + n01)

    Uses largest-remainder rounding to guarantee exactly ``width`` characters.
    """
    N = n11 + n10 + n01 + n00
    if N == 0:
        bar = "\u2592" * width
        return (_BRIGHT_RED + bar + _RESET) if _ANSI else bar
    n_split = n10 + n01
    counts = [n11, n00, n_split]
    exact = [c / N * width for c in counts]
    floored = [int(f) for f in exact]
    remainder_order = sorted(range(3), key=lambda i: exact[i] - floored[i], reverse=True)
    deficit = width - sum(floored)
    for i in range(deficit):
        floored[remainder_order[i]] += 1
    segments = []
    for ch, w, colored in zip(
        ["\u2588", "\u2591", "\u2592"],
        floored,
        [False, False, True],
    ):
        if w == 0:
            continue
        text = ch * w
        if colored and _ANSI:
            text = _BRIGHT_RED + text + _RESET
        segments.append(text)
    return "".join(segments)


def _mcc_strength(mcc: float) -> str:
    """Return a short verbal label for an agreement MCC value."""
    if mcc >= 0.7:
        return "very strong"
    if mcc >= 0.4:
        return "strong"
    if mcc >= 0.2:
        return "moderate"
    if mcc > -0.2:
        return "weak"
    if mcc > -0.5:
        return "inverse (moderate)"
    return "inverse (strong)"


def _mcc_interpretation(mcc: float) -> str:
    """Return a plain-language behavioral interpretation of an MCC agreement value."""
    if mcc >= 0.7:
        return "Performs nearly the same on the same items"
    if mcc >= 0.4:
        return "Performs similarly on the same items"
    if mcc >= 0.2:
        return "Disagree on some of the same items"
    if mcc > -0.2:
        return "Tends to disagree on same items"
    if mcc > -0.5:
        return "Disagrees most of the time on the same items"
    return "Virtually the opposite behavior"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_SEED_UNSET = object()


def _seed_note(rng_seed) -> str:
    """The ` | seed: N` suffix for a summary's metadata line.

    Empty when the caller didn't supply one at all (older call sites, direct
    print_analysis_summary() users), so nothing changes for them.
    """
    if rng_seed is _SEED_UNSET:
        return ""
    return f" | seed: {rng_seed}" if rng_seed is not None else " | seed: none (varies per call)"


def print_analysis_summary(
    analysis: Union[
        AnalysisBundle,
        MultiModelBundle,
        Mapping[str, AnalysisBundle],
        Mapping[str, MultiModelBundle],
    ],
    *,
    top_pairwise: int = None,
    line_width: int = 41,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
    style: Literal["line", "gradient"] = "gradient",
    p_value_method=_UNSET,
    min_meaningful_diff: Optional[float] = None,
    item_singular: str = "template",
    item_plural: str = "templates",
    show_rank_probabilities: bool = False,
    pareto: Optional[dict] = None,
    rng_seed: Optional[int] = _SEED_UNSET,
    metric: Optional[str] = None,
    factor_singular: str = "model",
    factor_plural: str = "models",
    ci_alpha: Optional[float] = None,
) -> None:
    """Print a concise console summary of analyze() results.

    Parameters
    ----------
    style : {"gradient", "line"}
        Interval plot style.  ``"gradient"`` (default) renders multi-band CI
        plots (90 / 95 / 99 / 99.9 % opacity gradient) when the bundle contains
        ``multi_ci`` data.  ``"line"`` always uses the classic dot-and-line plot.
    p_value_method : str or None, optional
        Override the p-value method for display.  When ``_UNSET`` (default),
        reads from the bundle's stored ``p_value_method``.
    show_rank_probabilities : bool
        Print the bootstrap "Rank Probabilities" block (P(Best)/E[Rank] per
        entity). Off by default -- see ``ComparisonResult.summary()``'s
        docstring for why.
    """
    if isinstance(analysis, MultiModelBundle):
        _print_multi_model_summary(
            analysis,
            rng_seed=rng_seed,
            top_pairwise=top_pairwise,
            line_width=line_width,
            pairwise_sort=pairwise_sort,
            style=style,
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
            factor_singular=factor_singular,
            factor_plural=factor_plural,
        )
        return

    if isinstance(analysis, AnalysisBundle):
        _print_bundle_summary(
            analysis,
            rng_seed=rng_seed,
            top_pairwise=top_pairwise,
            line_width=line_width,
            pairwise_sort=pairwise_sort,
            style=style,
            p_value_method=p_value_method,
            min_meaningful_diff=min_meaningful_diff,
            item_singular=item_singular,
            item_plural=item_plural,
            show_rank_probabilities=show_rank_probabilities,
            pareto=pareto,
            metric=metric,
            ci_alpha=ci_alpha,
        )
        return

    for evaluator_name, bundle in analysis.items():
        _print_loud_section(f"Evaluator: {evaluator_name}")
        if isinstance(bundle, MultiModelBundle):
            _print_multi_model_summary(
                bundle,
                rng_seed=rng_seed,
                top_pairwise=top_pairwise,
                line_width=line_width,
                pairwise_sort=pairwise_sort,
                style=style,
                min_meaningful_diff=min_meaningful_diff,
                show_rank_probabilities=show_rank_probabilities,
                factor_singular=factor_singular,
                factor_plural=factor_plural,
            )
        else:
            _print_bundle_summary(
                bundle,
                rng_seed=rng_seed,
                top_pairwise=top_pairwise,
                line_width=line_width,
                pairwise_sort=pairwise_sort,
                style=style,
                p_value_method=p_value_method,
                min_meaningful_diff=min_meaningful_diff,
                item_singular=item_singular,
                item_plural=item_plural,
                show_rank_probabilities=show_rank_probabilities,
            )
        print()


def print_brief_summary(
    analysis: Union[
        AnalysisBundle,
        MultiModelBundle,
        Mapping[str, AnalysisBundle],
        Mapping[str, MultiModelBundle],
    ],
    *,
    item_singular: str = "template",
    item_plural: str = "templates",
) -> None:
    """Print a compact leaderboard-only summary of analyze() results.

    Shows just the executive leaderboard — entity names, significance groups,
    mean scores, CIs, and verdicts — without the full statistical breakdown
    (no ASCII advantage plots, no pairwise tables, no robustness section).
    Use ``print_analysis_summary()`` for the complete output.
    """
    if isinstance(analysis, MultiModelBundle):
        _print_brief_multi_model(analysis)
        return

    if isinstance(analysis, AnalysisBundle):
        _print_brief_bundle(analysis, item_singular=item_singular, item_plural=item_plural)
        return

    # Per-evaluator dict.
    for evaluator_name, bundle in analysis.items():
        _print_loud_section(f"Evaluator: {evaluator_name}")
        if isinstance(bundle, MultiModelBundle):
            _print_brief_multi_model(bundle)
        else:
            _print_brief_bundle(bundle, item_singular=item_singular, item_plural=item_plural)
        print()


def _print_brief_bundle(
    bundle: AnalysisBundle,
    *,
    item_singular: str = "prompt",
    item_plural: str = "prompts",
) -> None:
    """Brief output for a single-model AnalysisBundle."""
    n = bundle.benchmark.n_templates
    m = bundle.benchmark.n_inputs
    n_runs = bundle.benchmark.n_runs
    method = bundle.resolved_method or "auto"
    alpha = get_alpha_ci()
    ci_pct = int(round((1 - alpha) * 100))
    runs_str = f" × {n_runs} runs" if n_runs > 1 else ""
    print(
        f"{n} {item_plural} | {m} inputs{runs_str} | "
        f"method={method} | {ci_pct}% CI"
    )
    print()
    _print_executive_summary(bundle, item_singular=item_singular)


def _print_brief_multi_model(bundle: MultiModelBundle) -> None:
    """Brief output for a MultiModelBundle."""
    n_models = bundle.benchmark.n_models
    n_templates = bundle.benchmark.n_templates
    m = bundle.benchmark.n_inputs
    n_runs = bundle.benchmark.n_runs
    method = bundle.model_level.resolved_method or "auto"
    alpha = get_alpha_ci()
    ci_pct = int(round((1 - alpha) * 100))
    runs_str = f" × {n_runs} runs" if n_runs > 1 else ""
    print(
        f"{n_models} models × {n_templates} prompts | {m} inputs{runs_str} | "
        f"method={method} | {ci_pct}% CI"
    )
    best_model, best_template = bundle.best_pair
    print(f"Best pair by mean: model='{best_model}'  prompt='{best_template}'")
    print()
    _print_executive_summary(bundle.model_level, item_singular="model")
    if n_templates > 1:
        print()
        _print_executive_summary(bundle.template_level, item_singular="prompt")


def print_pairwise_summary(
    pair: PairedDiffResult,
    *,
    alpha: Optional[float] = None,
    correction: str = "",
    line_width: int = 50,
    style: Literal["line", "gradient"] = "gradient",
) -> None:
    """Print a focused, human-readable summary for a single pairwise comparison.

    Displays the gap estimate, an ASCII interval plot of the confidence
    interval, and a plain-language verdict so you can immediately see whether
    the difference is statistically distinguishable from zero.

    Parameters
    ----------
    pair : PairedDiffResult
        A single pairwise comparison result, e.g. from
        ``report.pairwise.get("Model A", "Model B")``.
    alpha : float
        Significance threshold (default 0.01).
    correction : str
        Name of the multiple-comparisons correction applied, e.g. ``'fdr_bh'``.
        Shown in the header when provided.
    line_width : int
        Width of the ASCII interval plot (default 50 characters).

    Examples
    --------
    >>> pair = report.pairwise.get("Model A", "Model B")
    >>> from evalstats.core.summary import print_pairwise_summary
    >>> print_pairwise_summary(pair)

    Or use the convenience method directly on the pair or the matrix:

    >>> pair.summary()
    >>> report.pairwise.summary("Model A", "Model B")
    """
    if alpha is None:
        alpha = get_alpha_ci()
    a, b = pair.template_a, pair.template_b
    stat_label = pair.statistic.capitalize()
    ci_pct = int(round((1.0 - alpha) * 100))
    display_p_value, p_method_label = _pairwise_display_pvalue(pair)

    _print_loud_section(f"Pairwise: {a} vs. {b}")

    corr_str = f"  |  correction: {correction}" if correction else ""
    print(f"  method: {pair.test_method}{corr_str}  |  N={pair.n_inputs} inputs")
    print()

    # --- Gap and CI ---
    # Detect percentage-scale values for nicer formatting.
    all_vals = [pair.point_diff, pair.ci_low, pair.ci_high]
    looks_pct = all(abs(v) <= 1.5 for v in all_vals)
    if looks_pct:
        def _fmt(v: float) -> str:
            return f"{v:+.1%}"
    else:
        def _fmt(v: float) -> str:
            return f"{v:+.4f}"

    print(
        f"  {stat_label} gap ({a} − {b}):  "
        f"{_BOLD}{_fmt(pair.point_diff)}{_RESET}"
    )
    print(f"  {ci_pct}% CI:  [{_fmt(pair.ci_low)}, {_fmt(pair.ci_high)}]")
    print()

    # --- ASCII interval plot ---
    max_abs = max(
        1e-12,
        abs(pair.point_diff),
        abs(pair.ci_low),
        abs(pair.ci_high),
        abs(pair.point_diff - pair.std_diff),
        abs(pair.point_diff + pair.std_diff),
    )
    axis_low, axis_high = -max_abs, max_abs
    line = _choose_interval_line(
        mean=pair.point_diff,
        ci_low=pair.ci_low,
        ci_high=pair.ci_high,
        spread_low=pair.point_diff - pair.std_diff,
        spread_high=pair.point_diff + pair.std_diff,
        axis_low=axis_low,
        axis_high=axis_high,
        width=line_width,
        style=style,
        multi_ci=pair.multi_ci,
    )
    ci_legend = _legend_ci_label(style, ci_pct, pair.multi_ci is not None)
    mean_marker = _mean_marker_legend(style, pair.statistic)
    print(
        f"{_DIM}  axis: [{axis_low:+.3f}, {axis_high:+.3f}]  "
        f"(· ±1σ spread, {ci_legend}{mean_marker}, │ zero){_RESET}"
    )
    print(f"  {b} (<0) {line} (>0) {a}")
    print()

    # --- Effect size and p-value ---
    d = pair.rank_biserial
    p_str = _format_p_value(display_p_value)
    sig = display_p_value < alpha
    sig_color = _BRIGHT_GREEN if (sig and _ANSI) else (_YELLOW if _ANSI else "")
    sig_reset = _RESET if _ANSI else ""
    sig_label = "significant" if sig else "not significant"
    print(
        f"  Effect size (rank-biserial r):  {d:+.3f}   "
        f"p ({p_method_label}) = {sig_color}{p_str}{sig_reset}  ({sig_label})"
    )

    # --- Behavioral agreement (binary data only) ---
    if pair.agreement_mcc is not None and pair.binary_confusion is not None:
        mcc = pair.agreement_mcc
        strength = _mcc_strength(mcc)
        interpretation = _mcc_interpretation(mcc)
        print()
        print(f"  Behavioral agreement (MCC):  {mcc:+.3f}  — {strength} overlap - {interpretation}")

    print()


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------

def _print_loud_section(title: str) -> None:
    heading = f" {title.upper()} "
    border = "=" * len(heading)
    print(f"{_BOLD}{_BRIGHT_CYAN}{border}{_RESET}")
    print(f"{_BOLD}{_BRIGHT_CYAN}{heading}{_RESET}")
    print(f"{_BOLD}{_BRIGHT_CYAN}{border}{_RESET}")


def _print_subsection(title: str) -> None:
    """Print a secondary `--- Title ---` header in bold cyan."""
    print(f"{_BOLD}{_CYAN}{title}{_RESET}")


# ---------------------------------------------------------------------------
# Instability helpers
# ---------------------------------------------------------------------------

def _instability_label(instability: float) -> str:
    """Map an instability score (mean per-cell seed std) to a plain-language description.

    Thresholds are calibrated for scores normalised to roughly [0, 1].
    ``instability`` is the mean over inputs of the within-cell seed std,
    so a value of 0.10 means scores typically shift by ±0.10 across runs.
    """
    if np.isnan(instability):
        return "—"
    if instability >= 0.35:
        return "near-random across runs"
    if instability >= 0.20:
        return "highly noisy across runs"
    if instability >= 0.10:
        return "moderately noisy across runs"
    if instability >= 0.05:
        return "mostly stable across runs"
    if instability >= 0.01:
        return "very stable across runs"
    return "effectively deterministic across runs"


def _instability_color(instability: float) -> str:
    """Return an ANSI color for an instability score (empty string when colors off)."""
    if not _ANSI or np.isnan(instability):
        return ""
    if instability >= 0.20:
        return _BRIGHT_RED
    if instability >= 0.10:
        return _YELLOW
    return ""  # neutral — no color applied


def _stability_emoji_label(instability: float) -> str:
    """Return an emoji + label string for a stability column in the executive summary."""
    if np.isnan(instability):
        return "—"
    if instability >= 0.35:
        return "💀 Near-random"
    if instability >= 0.20:
        return "Noisy"
    if instability >= 0.10:
        return "Variable"
    if instability >= 0.05:
        return "Mostly Stable"
    return "Stable"


# ---------------------------------------------------------------------------
# Consistency (ICC) helpers
# ---------------------------------------------------------------------------

def _consistency_label(icc: float) -> str:
    """Map an ICC value to a plain-language description, per Koo & Li (2016) bands.

    <0.50 poor, 0.50-0.75 moderate, 0.75-0.90 good, >0.90 excellent. These
    bands are a commonly cited psychometrics convention, not separately
    validated for LLM evals.
    """
    if np.isnan(icc):
        return "—"
    if icc >= 0.90:
        return "excellent"
    if icc >= 0.75:
        return "good"
    if icc >= 0.50:
        return "moderate"
    return "poor"


def _consistency_color(icc: float) -> str:
    """Return an ANSI color for an ICC value (empty string when colors off).

    Inverted relative to ``_instability_color``: here low is concerning.
    """
    if not _ANSI or np.isnan(icc):
        return ""
    if icc < 0.50:
        return _BRIGHT_RED
    if icc < 0.75:
        return _YELLOW
    return ""  # neutral — no color applied


# ---------------------------------------------------------------------------
# Multi-model summary
# ---------------------------------------------------------------------------

def _display_order(bundle) -> "np.ndarray":
    """Indices giving a stable, readable display order: descending mean,
    label as tiebreak.

    These orderings used to read ``rank_dist.expected_ranks``/``p_best``,
    which forced the (opt-in) rank bootstrap purely to decide row order --
    see ``core.ranking.LazyRankDistribution``. Mean order is free, already
    what the leaderboard sorts by elsewhere, and deterministic.
    """
    means = np.asarray(bundle.robustness.mean, dtype=float)
    labels = list(bundle.labels)
    return np.array(
        sorted(range(len(labels)), key=lambda i: (-means[i], labels[i])),
        dtype=int,
    )


def _print_multi_model_summary(
    bundle: MultiModelBundle,
    *,
    rng_seed=_SEED_UNSET,
    top_pairwise: int = None,
    line_width: int,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
    style: Literal["line", "gradient"] = "gradient",
    min_meaningful_diff: Optional[float] = None,
    show_rank_probabilities: bool = False,
    factor_singular: str = "model",
    factor_plural: str = "models",
) -> None:
    """Print a multi-factor summary.

    ``factor_singular``/``factor_plural`` name the compared axis. The parser
    carries that axis in its model slot whatever the source column was called,
    so every label here comes from these rather than the word "model".
    """
    _print_loud_section("Analysis Summary")
    # Same "a × b × c" phrasing the per-section Shape lines use, rather than
    # the BenchmarkShape repr. Dimensions that are a single implicit level
    # (one prompt, one run) are left out instead of printed as 1, and the
    # counts line that used to sit under this one is gone: it restated these
    # numbers and carried only the seed note, which now rides here.
    bench, shape = bundle.benchmark, bundle.shape
    shape_parts = [f"{bench.n_models} {factor_plural}"]
    if bench.n_templates > 1:
        shape_parts.append(f"{bench.n_templates} prompts")
    shape_parts.append(f"{bench.n_inputs} inputs")
    if shape.n_input_vars > 1:
        shape_parts.append(f"{shape.n_input_vars} input vars")
    shape_parts.append(f"{shape.n_evaluators} evaluator{'s' if shape.n_evaluators != 1 else ''}")
    if bench.n_runs > 1:
        shape_parts.append(f"{bench.n_runs} runs")
    print(f"Shape: {' × '.join(shape_parts)}{_seed_note(rng_seed)}")
    model_str = ", ".join(bundle.benchmark.model_labels)
    print(f"{factor_plural.capitalize()}: {model_str}")
    best_model, best_template = bundle.best_pair
    if bundle.benchmark.n_templates > 1:
        print(f"{_BOLD}Best pair by mean:{_RESET} {factor_singular}='{_BRIGHT_GREEN}{best_model}{_RESET}'  template='{_BRIGHT_GREEN}{best_template}{_RESET}'")
    else:
        # One implicit template: naming it says nothing, and "pair" is a
        # misnomer when there is only one axis.
        print(f"{_BOLD}Best by mean:{_RESET} {factor_singular}='{_BRIGHT_GREEN}{best_model}{_RESET}'")
    print()

    # MultiModelBenchmark requires >= 2 models, so this section (comparing
    # across models) always has something to show. The per-template section
    # right below it doesn't have the same guarantee -- a single implicit
    # template is common (e.g. a plain model-only comparison) -- so it, and
    # the equally-degenerate per-model breakdown loop further down, are
    # skipped when there's nothing to compare there.
    # Named for the axis being compared rather than "model-level", and the
    # marginalization is only worth stating when there is more than one
    # template to marginalize over.
    factor_label = factor_singular
    across = (
        f" ({bundle.benchmark.n_templates} prompts pooled)"
        if bundle.benchmark.n_templates > 1 else ""
    )
    _print_loud_section(f"Comparison across '{factor_label}'{across}")
    _print_bundle_summary(
        bundle.model_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
        item_singular=factor_label,
        item_plural=f"{factor_label}s",
        pairwise_sort=pairwise_sort,
        style=style,
        min_meaningful_diff=min_meaningful_diff,
        show_rank_probabilities=show_rank_probabilities,
    )
    print()

    if bundle.benchmark.n_templates > 1:
        _print_loud_section(f"Cross-{factor_singular} per-template comparison ({factor_plural} collapsed):")
        _print_bundle_summary(
            bundle.template_level,
            top_pairwise=top_pairwise,
            line_width=line_width,
            item_singular="template",
            item_plural="templates",
            pairwise_sort=pairwise_sort,
            style=style,
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )
        print()

    # Instability across runs across models
    instability_rows = _collect_cross_model_seed_instability_rows(bundle)
    if instability_rows:
        _print_cross_model_seed_instability(bundle, rows=instability_rows)
        most_stable_model, instability, *_ = instability_rows[0]
        print(
            f"  {_BOLD}{_BRIGHT_GREEN}-> Most stable {factor_singular} across runs:{_RESET} "
            f"'{most_stable_model}' "
            f"(instability={instability:.4f}, {_instability_label(instability)})"
        )

    if bundle.benchmark.n_templates > 1:
        for model_label, model_bundle in bundle.per_model.items():
            print()
            _print_loud_section(f"Per-{factor_singular.capitalize()} Summary: {model_label}")
            _print_bundle_summary(
                model_bundle,
                top_pairwise=top_pairwise,
                line_width=line_width,
                pairwise_sort=pairwise_sort,
                style=style,
                guidance=False,
                show_rank_probabilities=show_rank_probabilities,
            )

    print()
    if bundle.benchmark.n_templates <= 1:
        # Every "model/template pair" is just a model, so the matrix, the
        # All-N listing and the pair leaderboard below restate the
        # model-level tables above with "/ default_prompt" appended. The
        # per-template section above is already gated the same way.
        return
    _print_loud_section(f"Cross-{factor_singular.capitalize()} Ranking (all {factor_singular}/template pairs)")
    _print_model_template_matrix(bundle)

    # The unconditional "Mean Performance" listing orders by mean, so it
    # needs nothing from the rank distribution. P(Best)/E[Rank] are read
    # only inside the show_rank_probabilities block below, which keeps the
    # rank bootstrap genuinely opt-in.
    rank_labels = bundle.cross_model.labels
    rank_pairs = [_split_model_template_label(label) for label in rank_labels]
    rank_bar_width = 14
    n_ranked_items = len(rank_labels)
    model_col_width = min(24, max(len(model) for model, _ in rank_pairs) + 2)
    template_col_width = min(24, max(len(template) for _, template in rank_pairs) + 2)
    top_indices = _display_order(bundle.cross_model)
    n_show = len(top_indices)

    if show_rank_probabilities:
        p_best = bundle.cross_model.rank_dist.p_best
        expected_ranks = bundle.cross_model.rank_dist.expected_ranks
        pbest_indices = np.argsort(-p_best)
        _print_subsection(f"--- Rank Probabilities: All {n_show} by P(Best) ({_rank_method_label(bundle.cross_model)}) ---")
        print(
            f"  {factor_singular.capitalize():<{model_col_width}s} "
            f"{'Template':<{template_col_width}s} "
            f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
            f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
        )
        for idx in pbest_indices[:n_show]:
            model_label, template_label = rank_pairs[idx]
            model_label = _truncate_label(model_label, model_col_width)
            template_label = _truncate_label(template_label, template_col_width)
            p_best_i = float(p_best[idx])
            expected_rank_i = float(expected_ranks[idx])
            p_color = _p_best_color(p_best_i)
            p_reset = _RESET if p_color else ""
            p_str = f"{p_best_i:>8.1%} {_ratio_bar(p_best_i, width=rank_bar_width)}"
            print(
                f"  {model_label:<{model_col_width}s} "
                f"{template_label:<{template_col_width}s} "
                f"{p_color}{p_str}{p_reset} "
                f"{expected_rank_i:>8.2f} "
                f"{_rank_hump_lane(expected_rank_i, n_ranked_items, width=rank_bar_width)}"
            )

    cross_rob = bundle.cross_model.robustness
    stat_label = "Mean"

    # Reference value (absolute scale) for the │ marker.
    ref_val = float(np.mean(cross_rob.mean))

    # Axis bounds cover means and marginal CIs.
    cross_means = cross_rob.mean
    cross_ci_lows = cross_rob.ci_low
    cross_ci_highs = cross_rob.ci_high
    cross_std = cross_rob.std
    cross_sigma_lows = cross_means - cross_std
    cross_sigma_highs = cross_means + cross_std
    all_vals = np.concatenate([
        cross_means,
        cross_ci_lows,
        cross_ci_highs,
        cross_sigma_lows,
        cross_sigma_highs,
    ])
    val_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(val_range * 0.05, 1e-4)
    ma_low = float(np.min(all_vals)) - pad
    ma_high = float(np.max(all_vals)) + pad

    ref_label_str = "grand mean"
    print()
    _print_subsection(
        f"--- {stat_label} Performance: All {n_show} "
        f"(marginal {int(round((1 - get_alpha_ci()) * 100))}% CIs) ---"
    )
    _ci_legend_mm = _legend_ci_label(style, int(round((1 - get_alpha_ci()) * 100)), cross_rob.multi_ci is not None)
    _mean_marker_mm = _mean_marker_legend(style, stat_label.lower())
    print(
        f"{_DIM}  axis: [{ma_low:.3f}, {ma_high:.3f}]  "
        f"(· ±1σ, {_ci_legend_mm}{_mean_marker_mm}, │ {ref_label_str}){_RESET}"
    )
    print(
        f"  {factor_singular.capitalize():<{model_col_width}s} "
        f"{'Template':<{template_col_width}s} "
        f"{'Interval Plot':<{line_width}s} "
        f"{stat_label:>8s} {'CI Low':>9s} {'CI High':>9s}"
    )

    # Build a label→index map for cross_rob (labels may differ from ma.labels order).
    cross_labels = list(cross_rob.labels)

    for idx in top_indices[:n_show]:
        pair_label = rank_labels[idx]
        model_label, template_label = _split_model_template_label(pair_label)
        model_label = _truncate_label(model_label, model_col_width)
        template_label = _truncate_label(template_label, template_col_width)
        try:
            rob_idx = cross_labels.index(pair_label)
        except ValueError:
            rob_idx = idx
        abs_mean = float(cross_means[rob_idx])
        abs_ci_low = float(cross_ci_lows[rob_idx])
        abs_ci_high = float(cross_ci_highs[rob_idx])
        abs_sigma_low = float(cross_sigma_lows[rob_idx])
        abs_sigma_high = float(cross_sigma_highs[rob_idx])
        line = _choose_interval_line(
            mean=abs_mean,
            ci_low=abs_ci_low,
            ci_high=abs_ci_high,
            spread_low=abs_sigma_low,
            spread_high=abs_sigma_high,
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
            reference=ref_val,
            style=style,
            multi_ci=_rob_multi_ci_at(cross_rob.multi_ci, rob_idx),
        )
        print(
            f"  {model_label:<{model_col_width}s} "
            f"{template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{abs_mean:>7.3f} "
            f"{abs_ci_low:>8.3f} "
            f"{abs_ci_high:>8.3f}"
        )

    print()
    _print_cross_model_executive_summary(bundle)
    print()


def _print_model_template_matrix(bundle: MultiModelBundle) -> None:
    """Print a model × template score matrix (mean ±std, heat encoding)."""
    model_labels = bundle.benchmark.model_labels
    template_labels = bundle.benchmark.template_labels
    cross = bundle.cross_model

    # Build (model, template) -> mean from the flat cross_model bundle.
    # Labels are formatted as "model / template" by get_flat_result().
    cell_mean: dict[tuple[str, str], float] = {}
    for label, m in zip(
        cross.labels,
        cross.robustness.mean,
    ):
        parts = label.split(" / ", 1)
        if len(parts) == 2:
            cell_mean[(parts[0], parts[1])] = float(m)

    all_means = list(cell_mean.values())
    mn, mx = min(all_means), max(all_means)
    heat_chars = "·░▒▓█"

    # Cells statistically tied for best -- the same CD-style significance-
    # group analysis used by the executive summary below (group "#1"),
    # rather than the single highest raw mean. A marginally higher point
    # estimate should not read as a decisive winner when the pairwise CIs
    # show it isn't distinguishable from its neighbors; that would defeat
    # the point of reporting calibrated intervals in the first place.
    cross_labels_all = list(cross.labels)
    cross_means_all = cross.robustness.mean
    sort_idx = list(np.argsort(-cross_means_all))
    labels_sorted = [cross_labels_all[i] for i in sort_idx]
    label_to_group = _assign_significance_groups(cross.pairwise, labels_sorted)
    best_cells: set[tuple[str, str]] = set()
    for label, group in label_to_group.items():
        if group == "#1":
            parts = label.split(" / ", 1)
            if len(parts) == 2:
                best_cells.add((parts[0], parts[1]))
    if not best_cells:
        # Fallback so a winner is still shown if group detection finds
        # nothing (e.g. degenerate pairwise matrix).
        best_cells = {max(cell_mean, key=cell_mean.get)}

    def _heat(v: float) -> str:
        if mx == mn:
            return heat_chars[-1]
        idx = min(int((v - mn) / (mx - mn) * len(heat_chars)), len(heat_chars) - 1)
        return heat_chars[idx]

    # Cell width: at least enough for "0.800 ▓*" (8 chars), but expand
    # when template labels are longer so header/data columns stay aligned.
    CELL_W = max(8, max(len(t) for t in template_labels))
    model_col_w = max(len(m) for m in model_labels)

    def _fmt_cell(mdl: str, t: str) -> str:
        if (mdl, t) not in cell_mean:
            return f"{'N/A':^{CELL_W}}"
        m = cell_mean[(mdl, t)]
        h = _heat(m)
        is_best = (mdl, t) in best_cells
        marker = "*" if is_best else " "
        plain = f"{m:.3f} {h}{marker}".rjust(CELL_W)
        if is_best:
            return f"{_BOLD}{_BRIGHT_GREEN}{plain}{_RESET}"
        return plain

    # Header
    header = f"  {'':>{model_col_w}}"
    for t in template_labels:
        header += f"  {t:^{CELL_W}}"
    print(header)

    # Data rows
    div = "  " + "─" * max(1, len(header) - 2)
    print(div)
    for mdl in model_labels:
        row = f"  {mdl:>{model_col_w}}"
        for t in template_labels:
            row += f"  {_fmt_cell(mdl, t)}"
        print(row)

    # Footer
    print(div)
    print(
        f"  * = statistically tied for best (95% CI, not significantly beaten)  |  "
        f"heat: · (low) → █ (high), range [{mn:.3f}, {mx:.3f}]"
    )
    print()


def _print_cross_model_executive_summary(bundle: MultiModelBundle) -> None:
    """Print executive leaderboard for cross-model (model/template) pairs."""
    cross = bundle.cross_model
    labels = list(cross.labels)
    n = len(labels)
    if n < 2:
        return

    means = cross.robustness.mean
    sort_idx = list(np.argsort(-means))
    labels_sorted = [labels[i] for i in sort_idx]
    label_to_group = _assign_significance_groups(cross.pairwise, labels_sorted)

    split_pairs = [_split_model_template_label(label) for label in labels]
    model_w = min(28, max(10, max(len(m) for m, _ in split_pairs) + 2))
    template_w = min(28, max(12, max(len(t) for _, t in split_pairs) + 2))
    grp_w = 4
    mean_w = 6
    ci_w = 15
    stab_w = 16
    noise_w = _NOISE_COL_W

    # Seed variance for stability + per-run noise columns (optional, mirrors
    # _print_executive_summary).  The noise strip uses the same global scale
    # as the "Per-input Variance Across Runs" table above it, so bar heights
    # are comparable across rows and across the two tables.
    sv = cross.seed_variance
    has_stability = sv is not None
    sv_labels = list(sv.labels) if has_stability else []
    global_cell_max = float(sv.per_cell_seed_std.max()) if has_stability else 0.0

    _print_subsection("--- Executive Summary (Cross-model pair leaderboard) ---")
    _cross_ci_header = "Wilson-flat CI" if _uses_wilson_ci(cross) else "CI"
    header = (
        f"  {'Model':<{model_w}s}"
        f"  {'Template':<{template_w}s}"
        f"  {'Grp':^{grp_w}s}"
        f"  {'Mean':>{mean_w}s}"
        f"  {_cross_ci_header:<{ci_w}s}"
    )
    if has_stability:
        header += f"  {_NOISE_STRIP_HEADER:<{noise_w}s}"
        header += f"  {'Stability':<{stab_w}s}"
    header += "  Verdict"
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    for label in labels_sorted:
        orig_idx = labels.index(label)
        mean_val = float(means[orig_idx])
        model_label, template_label = _split_model_template_label(label)

        ci_lo = float(cross.robustness.ci_low[orig_idx])
        ci_hi = float(cross.robustness.ci_high[orig_idx])
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

        group = label_to_group.get(label, "?")
        verdict = _exec_verdict(label, label_to_group, labels_sorted)

        plain_model = f"{_truncate_label(model_label, model_w):<{model_w}s}"
        plain_template = f"{_truncate_label(template_label, template_w):<{template_w}s}"
        plain_grp = f"{group:^{grp_w}s}"
        if group == "#1" and _ANSI:
            model_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_model}{_RESET}"
            template_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_template}{_RESET}"
            grp_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_grp}{_RESET}"
            verdict_str = f"{_BRIGHT_GREEN}{verdict}{_RESET}"
        else:
            model_str = plain_model
            template_str = plain_template
            grp_str = plain_grp
            verdict_str = verdict

        row = (
            f"  {model_str}"
            f"  {template_str}"
            f"  {grp_str}"
            f"  {mean_val:>{mean_w}.3f}"
            f"  {ci_str:<{ci_w}s}"
        )

        if has_stability:
            if label in sv_labels:
                sv_idx = sv_labels.index(label)
                instability_val = float(sv.instability[sv_idx])
                noise_plain = f"{_seed_noise_strip(sv.per_cell_seed_std[sv_idx], global_cell_max, max_width=_NOISE_STRIP_CHARS):<{noise_w}s}"
                stab_plain = f"{_stability_emoji_label(instability_val):<{stab_w}s}"
                row_color = _instability_color(instability_val)
            else:
                noise_plain = f"{'—':<{noise_w}s}"
                stab_plain = f"{'—':<{stab_w}s}"
                row_color = ""
            row += f"  {row_color}{noise_plain}{_RESET}" if row_color else f"  {noise_plain}"
            row += f"  {row_color}{stab_plain}{_RESET}" if row_color else f"  {stab_plain}"

        row += f"  {verdict_str}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Single-model bundle summary
# ---------------------------------------------------------------------------

def _pair_efficiency_cells(bundle, left, right) -> dict:
    """The four label-efficiency cells for one paired row, if available.

    Two separate lookups on purpose: the interval's efficiency comes from the
    correlation of the paired differences, the p-value's from the rank test's
    own. Missing entries yield None, which the renderer treats as "hide the
    column" rather than "print a blank".
    """
    def _get(store):
        d = getattr(bundle, store, None) or {}
        return d.get((str(left), str(right))) or d.get((str(right), str(left)))
    ci = _get("_pair_ci_eff")
    pv = _get("_pair_p_eff")
    # The p-side pair is filled in only when the p-value tests a DIFFERENT
    # estimand from the interval. See _p_side_efficiency_applies.
    return {
        "ci_rho2": ci[0] if ci else None,
        "ci_n_eff": ci[1] if ci else None,
        "_wsr_rho2": pv[0] if pv else None,
        "_wsr_n_eff": pv[1] if pv else None,
    }


def _p_side_efficiency_applies(eff_p_source: Optional[str], data_kind=None) -> bool:
    """Whether the p-value tests a different estimand from the interval.

    The paired path's p can come from four places. "boot"/"max_t" are
    bootstrap p-values on the SAME mean difference the interval covers, so
    their efficiency is the interval's -- printing a second, separately
    computed pair of columns beside it would imply a distinction that does not
    exist, and the two would differ only by Monte Carlo noise in the two
    judge_alignment calls. Only the rank-based sources ("wsr" Wilcoxon
    signed-rank, "nem" Nemenyi) test something else and need their own number.

    Getting this wrong is the exact failure _EFFICIENCY_TESTS warns about:
    reporting the efficiency of a test the user never ran.

    Suppressed entirely on BINARY data. A rank-based correlation on 0/1 scores
    is not something this project validates: _COMPARISON_METHODS_BINARY drops
    mwu/wilcoxon for exactly that reason ("rank-based and break down under that
    many ties"), and the paper's binary PPI claims cover the t-test path only.
    A reader can still force p_value_method="wsr" on binary data and get a
    Wilcoxon p; what they must not get is an efficiency figure implying that
    number rests on validated ground.
    """
    if str(data_kind) == "binary":
        return False
    return eff_p_source in {"wsr", "nem"}


def _paired_efficiency_row(bundle, left, right, eff_p_source) -> dict:
    """Efficiency cells for one paired row, with the p-side gated on source."""
    cells = _pair_efficiency_cells(bundle, left, right)
    applies = _p_side_efficiency_applies(
        eff_p_source, getattr(bundle, "resolved_data_kind", None))
    return {
        "ci_rho2": cells["ci_rho2"],
        "ci_n_eff": cells["ci_n_eff"],
        "rho2": cells["_wsr_rho2"] if applies else None,
        "n_eff": cells["_wsr_n_eff"] if applies else None,
    }


def _prepare_paired_pairwise_rows(
    bundle: "AnalysisBundle",
    *,
    p_value_method: Optional[str],
    sort: bool,
    pairwise_sort: Literal["grouped", "significance"],
    alpha: Optional[float] = None,
) -> tuple[Optional[list[dict]], dict]:
    """Normalize an AnalysisBundle's pairwise results into the common row
    shape :func:`_print_pairwise_section` renders, plus metadata describing
    which optional columns/sections apply.

    Extracted from the pre-unification body of ``_print_pairwise_section``
    verbatim (same swap/canonicalization/sort/p-value-method-resolution
    logic) so paired-path behavior is unchanged -- only repackaged so the
    row-rendering core can be shared with the unpaired path via
    :func:`_prepare_unpaired_pairwise_rows`. Returns ``(None, {})`` when
    there's exactly one entity (nothing to compare), matching the old
    early-return.
    """
    pair_item_col_width = 24

    first_result = next(iter(bundle.pairwise.results.values()), None)
    if first_result is None:
        return None, {}
    pair_stat_label = first_result.statistic.capitalize() if first_result else "Mean"

    is_newcombe_pairwise = "newcombe" in first_result.test_method.lower()
    is_sign_pairwise = "sign test" in first_result.test_method.lower()
    is_bootstrap_path = "bootstrap" in first_result.test_method.lower()
    using_max_t = bundle.pairwise.simultaneous_ci_method == "max_t"
    is_romano_wolf_active = (
        bundle.pairwise.correction_method == "romano_wolf"
        and len(bundle.pairwise.results) > 1
    )

    # Column-header PPI tag: only for p-value paths that are actually
    # PPI-corrected when alignment= is passed (McNemar/sign-test binary
    # paths and Nemenyi don't run through PPI, so they never get tagged).
    _ppi_tag = "PPI-" if getattr(bundle, "ppi_applied", False) else ""

    # Binary paired data must not fall through to Wilcoxon signed-rank. A rank
    # test on 0/1 scores is the thing _COMPARISON_METHODS_BINARY drops as
    # unsound under that many ties, and evalstats already computes the right
    # test for this cell -- it just was not being displayed. result.p_value
    # carries McNemar's mid-p without PPI (see core/paired.py's bonett_price
    # branch) and the PPI-corrected paired mean-difference test with it
    # (_ppi_paired_bonett_price), which is the binary PPI path the paper
    # validates. Both live on the "boot" source.
    #
    # Only when Romano-Wolf is NOT active: RW resamples its own p-values and
    # legitimately replaces whatever base test would otherwise run, binary
    # included.
    _is_binary_paired = str(getattr(bundle, "resolved_data_kind", None)) == "binary"

    if p_value_method == "auto":
        if is_romano_wolf_active:
            eff_p_source, p_col_header = "boot", f"p ({_ppi_tag}RW)"
        elif _is_binary_paired:
            eff_p_source = "boot"
            p_col_header = "p (PPI-paired-t)" if _ppi_tag else "p (mcnemar)"
        else:
            eff_p_source, p_col_header = "wsr", f"p ({_ppi_tag}wsr)"
    elif p_value_method == "boot":
        eff_p_source = "max_t" if (using_max_t and is_bootstrap_path) else "boot"
        p_col_header = f"p ({_ppi_tag}boot)"
    elif p_value_method == "wsr":
        eff_p_source, p_col_header = "wsr", f"p ({_ppi_tag}wsr)"
    elif p_value_method == "nem":
        eff_p_source, p_col_header = "nem", "p (nem)"
    else:  # None
        eff_p_source, p_col_header = None, None

    corr = bundle.pairwise.correction_method
    sim_ci_method = bundle.pairwise.simultaneous_ci_method
    # Same formatter the marginal section uses, so one run does not name the
    # same method two ways ("PPI Logit-t" above, "PPI ppi_logit_t" here).
    _pretty_ci_method = _pretty_marginal_ci_method(first_result.test_method) or first_result.test_method
    pair_results = list(bundle.pairwise.results.values())

    # Canonical left/right ordering based on expected-rank order keeps rows
    # readable by preventing arbitrary A/B flips between adjacent rows.
    _labels_for_order = list(bundle.labels)
    rank_order = {
        _labels_for_order[i]: idx
        for idx, i in enumerate(_display_order(bundle))
    }

    if pair_results:
        max_label_len = max(
            max(len(r.template_a), len(r.template_b)) for r in pair_results
        )
        pair_item_col_width = min(30, max(12, max_label_len + 2))

    rows = []
    for result in pair_results:
        a = result.template_a
        b = result.template_b
        pos_a = rank_order.get(a, len(rank_order))
        pos_b = rank_order.get(b, len(rank_order))
        swap = (pos_a > pos_b) or (pos_a == pos_b and a > b)

        if swap:
            left_item = b
            right_item = a
            point_diff = -float(result.point_diff)
            ci_low = -float(result.ci_high)
            ci_high = -float(result.ci_low)
            rank_biserial = -float(result.rank_biserial)
            left_pos = pos_b
            right_pos = pos_a
            # Flip multi_ci band bounds when direction is swapped.
            swapped_multi_ci = (
                {_a: (-hi, -lo) for _a, (lo, hi) in result.multi_ci.items()}
                if result.multi_ci is not None else None
            )
        else:
            left_item = a
            right_item = b
            point_diff = float(result.point_diff)
            ci_low = float(result.ci_low)
            ci_high = float(result.ci_high)
            rank_biserial = float(result.rank_biserial)
            left_pos = pos_a
            right_pos = pos_b
            swapped_multi_ci = result.multi_ci

        # PairedDiffResult.rank_biserial is computed from the raw judge
        # differences. When PPI is applied, prefer the corrected 2*theta
        # attached in api.py so the effect size does not sit uncorrected
        # beside a corrected mean, CI and p-value.
        _es_map = getattr(bundle, "_pair_es", None) or {}
        _es_ppi = _es_map.get((str(a), str(b)))
        if _es_ppi is not None:
            rank_biserial = -float(_es_ppi) if left_item == b else float(_es_ppi)

        if eff_p_source in {"max_t", "boot"}:
            display_p = result.p_value
        elif eff_p_source == "wsr":
            display_p = result.wilcoxon_p
        elif eff_p_source == "nem":
            display_p = (
                bundle.pairwise.friedman.get_nemenyi_p(str(left_item), str(right_item))
                if bundle.pairwise.friedman is not None else None
            )
        else:
            display_p = None

        # binary_confusion is symmetric in n11/n00; n10/n01 swap with direction
        # but for the bar we only need n_split = n10+n01, which is invariant.
        rows.append(
            {
                "left": left_item,
                "right": right_item,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "point_diff": point_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "std_diff": float(result.std_diff),
                "es_value": rank_biserial,
                "p_value": result.p_value,  # sort key -- always the bootstrap p, regardless of eff_p_source
                "display_p": display_p,
                "agreement_mcc": result.agreement_mcc,
                "binary_confusion": result.binary_confusion,
                "multi_ci": swapped_multi_ci,
                # Label efficiency, looked up in either key order because the
                # display pair may be swapped relative to how it was computed.
                **_paired_efficiency_row(bundle, left_item, right_item, eff_p_source),
            }
        )

    if pairwise_sort not in {"grouped", "significance"}:
        raise ValueError("pairwise_sort must be 'grouped' or 'significance'.")

    if sort:
        if pairwise_sort == "grouped":
            rows = sorted(
                rows,
                key=lambda row: (
                    row["left_pos"],
                    row["right_pos"],
                    row["p_value"],
                    -abs(row["point_diff"]),
                ),
            )
        else:
            rows = sorted(
                rows,
                key=lambda row: (
                    row["p_value"],
                    -abs(row["point_diff"]),
                    row["left_pos"],
                    row["right_pos"],
                ),
            )

    def _footer(_rows: list[dict], _max_pairs: int) -> None:
        print(f"{_DIM}  ES = Effect Size (r_rb) = rank biserial correlation (small≈0.1, medium≈0.3, large≈0.5){_RESET}")

        # Short p-value-method name (no correction detail -- that's stated
        # separately on the FWER-corrections line below) for the explicit
        # methods summary. The fuller descriptive line further down (with
        # correction detail folded in) still prints too.
        ppi_applied = getattr(bundle, "ppi_applied", False)
        # Nemenyi has no PPI-corrected variant (disallowed together with
        # alignment= at the call site), so it never gets the prefix.
        ppi_prefix = "PPI-" if ppi_applied else ""

        p_value_method_label = None
        if eff_p_source in {"max_t", "boot"}:
            if is_romano_wolf_active and eff_p_source == "boot":
                p_value_method_label = f"{ppi_prefix}Romano-Wolf step-down"
            elif is_newcombe_pairwise:
                p_value_method_label = "McNemar mid-p test"
            elif is_sign_pairwise:
                p_value_method_label = "Paired sign test"
            elif _is_binary_paired:
                p_value_method_label = (
                    f"{ppi_prefix}paired t-test (difference of proportions)"
                    if ppi_prefix else "McNemar mid-p test"
                )
            elif eff_p_source == "max_t":
                p_value_method_label = f"{ppi_prefix}Max-T bootstrap"
            else:
                p_value_method_label = f"{ppi_prefix}Bootstrap"
        elif eff_p_source == "wsr":
            p_value_method_label = f"{ppi_prefix}Wilcoxon signed-rank"
        elif eff_p_source == "nem":
            p_value_method_label = "Nemenyi post-hoc"

        # Explicit methods summary, directly above the p-value-method detail
        # line -- mirrors the matplotlib forest plot's subtitle, stated
        # plainly rather than nested into the section header. Two lines:
        # (1) CI method / p-value method / alpha, (2) simultaneous-CI method
        # and the FWER correction applied to p-values, broken out separately
        # since they can use different correction methods. Dimmed along with
        # the rest of this footnote block (ES=, p-value detail, stars:) --
        # methods detail, not part of the data itself.
        _alpha = get_alpha_ci() if alpha is None else alpha
        _line1 = [f"{int(round((1 - _alpha) * 100))}% CI method: {_pretty_ci_method}"]
        if p_value_method_label:
            _line1.append(f"p-value method: {p_value_method_label}")
        _line1.append(f"α={_alpha:g}")
        print(f"{_DIM}  {'  |  '.join(_line1)}{_RESET}")

        _line2 = [f"Simultaneous CI method: {_pretty_simultaneous_ci(sim_ci_method)}"]
        if p_value_method_label:
            _line2.append(f"FWER correction for p-values: {_pretty_correction(corr)}")
        print(f"{_DIM}  {'  |  '.join(_line2)}{_RESET}")

        if eff_p_source in {"max_t", "boot"}:
            if is_romano_wolf_active and eff_p_source == "boot":
                print(f"{_DIM}  {p_col_header} = {ppi_prefix}Romano-Wolf step-down (FWER-controlled){_RESET}")
            elif is_newcombe_pairwise:
                print(f"{_DIM}  {p_col_header} = McNemar mid-p test (two-sided, uncorrected){_RESET}")
            elif is_sign_pairwise:
                print(f"{_DIM}  {p_col_header} = paired sign test (two-sided exact, ties dropped, uncorrected){_RESET}")
            elif eff_p_source == "max_t":
                print(f"{_DIM}  {p_col_header} = {ppi_prefix}max-T bootstrap p-value (FWER-controlled, commensurate with simultaneous CIs){_RESET}")
            else:
                print(f"{_DIM}  {p_col_header} = {ppi_prefix}bootstrap p-value ({bundle.pairwise.correction_method}-corrected){_RESET}")
        elif eff_p_source == "wsr":
            print(f"{_DIM}  {p_col_header} = {ppi_prefix}Wilcoxon signed-rank ({bundle.pairwise.correction_method}-corrected){_RESET}")
        elif eff_p_source == "nem":
            print(f"{_DIM}  {p_col_header} = Nemenyi post-hoc (Friedman-based, FWER-controlled){_RESET}")
        print()
        _cd_labels = list(bundle.labels)
        labels_sorted = [_cd_labels[i] for i in _display_order(bundle)]
        _print_critical_difference_groups(
            bundle.pairwise,
            labels_sorted=labels_sorted,
            p_source="bootstrap",
        )

    meta = {
        "section_header": (
            f"--- Pairwise Comparisons "
            f"({int(round((1 - (get_alpha_ci() if alpha is None else alpha)) * 100))}% "
            f"{_pretty_ci_method} CIs) ---"
        ),
        "pair_stat_label": pair_stat_label,
        "pair_item_col_width": pair_item_col_width,
        "effect_label": "Left - Right",
        "es_label": "ES",
        "p_col_header": p_col_header,
        "footer_fn": _footer,
    }
    return rows, meta


_FAMILY_DISPLAY_UNPAIRED = {
    "binary_proportion": "proportion difference (Δp)",
    # "rank_based" names the TESTS (Kruskal-Wallis omnibus, Mann-Whitney U
    # post-hoc), not the estimand -- both families report a mean difference.
    "rank_based": "mean difference (Δ), Mann-Whitney tested",
}
_ESTIMAND_LABEL_UNPAIRED = {"mean_diff": "Δ"}


def _prepare_unpaired_pairwise_rows(
    result: "GroupComparisonResult",
    *,
    sort: bool,
    pairwise_sort: Literal["grouped", "significance"],
) -> tuple[list[dict], dict]:
    """Normalize a GroupComparisonResult's pairwise results into the same
    row shape :func:`_prepare_paired_pairwise_rows` produces.

    The between-subjects engine always uses exactly one FWER scheme
    (Bonferroni CI + Holm p) -- no Wilcoxon/Romano-Wolf/Newcombe/sign-test/
    max-T/Nemenyi method-family detection needed, and no ranking bootstrap
    to derive an alternate canonical left/right order from (natural
    factor-level order, i.e. ``result.labels``, is already canonical).
    ``point_diff``/``ci_low``/``ci_high`` are shifted by each pair's
    ``null_value`` (0.0 for every family now that all of them report a mean
    difference, so the shift is a no-op) so the shared axis/bar-rendering math in
    :func:`_print_pairwise_section` -- which assumes a signed quantity
    centered at zero, same convention the paired path's own "Left - Right"
    difference already has -- works identically for both estimand kinds.
    """
    show_p = result.show_p_values
    n_pairs = len(result.pairwise)
    null_value = result.pairwise[0].null_value if result.pairwise else 0.0
    estimand = result.pairwise[0].estimand if result.pairwise else "mean_diff"
    est_symbol = _ESTIMAND_LABEL_UNPAIRED.get(estimand, "Δ")
    # Every family's estimand is now a mean/proportion difference, whose null
    # is already 0 -- so the shift is a no-op and "Δ" describes the column
    # exactly. (``null_value`` is kept in the plumbing because the shared
    # bar-rendering math below consumes it, and because a future non-zero-null
    # estimand would need it again.)
    pair_stat_label = f"Δ{est_symbol}" if null_value != 0.0 else est_symbol

    # There is no secondary Δmean column: it existed only to put the old
    # dominance estimand back on the metric's own scale, and the primary
    # column now *is* that mean difference. The renderer's "ES" slot carries
    # the independent-samples rank-biserial, the paired path's counterpart.

    # Explicit pairwise test name -- previously the column header was just
    # "p" with no indication of which test produced it, PPI-corrected or
    # not. Mirrors the paired path's p_col_header/p-value-method labeling
    # (e.g. "p (PPI-wsr)" for Wilcoxon): tag with "PPI-" whenever
    # alignment= was passed, since both families' pairwise tests run
    # through the PPI rectifier in that case (_binary_pairwise_ppi /
    # _rank_based_pairwise_ppi above).
    ppi_applied = getattr(result, "ppi_applied", False)
    ppi_prefix = "PPI-" if ppi_applied else ""
    if result.family == "rank_based":
        pairwise_test_name, pairwise_test_abbrev = "Mann-Whitney U", "MWU"
    elif result.family == "binary_proportion":
        pairwise_test_name, pairwise_test_abbrev = "Welch's t-test", "Welch"
    else:
        pairwise_test_name = pairwise_test_abbrev = None
    p_col_header = (
        f"p ({ppi_prefix}{pairwise_test_abbrev})"
        if (show_p and pairwise_test_abbrev) else ("p" if show_p else None)
    )

    label_index = {lbl: i for i, lbl in enumerate(result.labels)}
    rows = []
    for p in result.pairwise:
        row = {
            "left": p.label_a,
            "right": p.label_b,
            "left_pos": label_index.get(p.label_a, 0),
            "right_pos": label_index.get(p.label_b, 0),
            "point_diff": p.point_estimate - null_value,
            "ci_low": p.ci_low - null_value,
            "ci_high": p.ci_high - null_value,
            "std_diff": 0.0,  # no ±1σ "spread" concept for a pairwise estimate itself
            "p_value": p.p_value,
            "display_p": p.p_value if show_p else None,
            "multi_ci": None,
            "rho2": p.rho2,
            "n_eff": p.n_eff,
            "es_value": p.rank_biserial,
        }
        rows.append(row)

    if pairwise_sort not in {"grouped", "significance"}:
        raise ValueError("pairwise_sort must be 'grouped' or 'significance'.")
    if sort:
        if pairwise_sort == "grouped":
            rows = sorted(
                rows,
                key=lambda row: (row["left_pos"], row["right_pos"], row["p_value"], -abs(row["point_diff"])),
            )
        else:
            rows = sorted(
                rows,
                key=lambda row: (row["p_value"], -abs(row["point_diff"]), row["left_pos"], row["right_pos"]),
            )

    def _footer(_rows: list[dict], _max_pairs: int) -> None:
        _line1 = ([f"{int(round((1 - result.alpha) * 100))}% CI method: {ci_method}"]
                  if ci_method else [])
        if _line1:
            if show_p and pairwise_test_name:
                _line1.append(f"p-value method: {ppi_prefix}{pairwise_test_name}")
            _line1.append(f"α={result.alpha:g}")
            print(f"{_DIM}  {'  |  '.join(_line1)}{_RESET}")
            if n_pairs > 1:
                _line2 = [f"Simultaneous CI method: {_pretty_correction(result.ci_correction)}"]
                if show_p:
                    _line2.append(
                        "FWER correction for p-values: "
                        f"{_pretty_correction(result.pvalue_correction)}"
                    )
                print(f"{_DIM}  {'  |  '.join(_line2)}{_RESET}")
        if show_p and pairwise_test_name:
            corr_note = (f"{result.pvalue_correction}-corrected" if n_pairs > 1
                         else "uncorrected, single comparison")
            print(f"{_DIM}  {p_col_header} = {ppi_prefix}{pairwise_test_name} ({corr_note}){_RESET}")
        _print_pairwise_efficiency_note(_rows, result)
        if n_pairs > 1 and show_p:
            print(
                f"  {_DIM}Verdict reflects the {result.ci_correction}-corrected CI; p is "
                f"independently {result.pvalue_correction}-corrected -- the two can rarely "
                f"disagree right at the boundary, since they're different (both valid) "
                f"FWER corrections.{_RESET}"
            )
        # Critical-difference rank bands, shared with the paired path's own
        # (see _prepare_paired_pairwise_rows) -- reuses
        # _critical_difference_groups/_print_critical_difference_groups
        # unmodified via _GroupDiffResultsAsPairwiseMatrix, an adapter whose
        # .simultaneous_ci_method sentinel routes it through the same
        # CI-exclusion significance check GroupDiffResult.significant
        # already uses (this engine has no p-value-threshold alternative
        # the way the paired path's Wilcoxon/Nemenyi paths do). Sorted by
        # mean descending -- there's no ranking bootstrap here, so this is
        # the same "best first" order the executive summary below uses.
        from evalstats.core.unpaired import _GroupDiffResultsAsPairwiseMatrix
        labels_sorted = [g.label for g in sorted(result.groups, key=lambda g: -g.mean)]
        print()
        _print_critical_difference_groups(
            _GroupDiffResultsAsPairwiseMatrix(result.pairwise),
            labels_sorted=labels_sorted,
            alpha=result.alpha,
            p_source="bootstrap",
        )

    label_width = min(24, max(8, max((len(g) for g in result.labels), default=8)))
    ci_method = getattr(result, "pairwise_ci_method", None)
    meta = {
        "section_header": (
            f"--- Pairwise Comparisons ({int(round((1 - result.alpha) * 100))}% {ci_method} CIs) ---"
            if ci_method
            else f"--- Pairwise Comparisons ({_FAMILY_DISPLAY_UNPAIRED[result.family]}) ---"
        ),
        "pair_stat_label": pair_stat_label,
        "pair_item_col_width": label_width,
        "effect_label": "Left - Right",
        # No secondary raw-mean-difference column: every family's primary
        # column already *is* that difference, so a copy would be redundant.
        # Independent-samples rank-biserial (2*theta), in the same column slot
        # and on the same scale as the paired path's, and PPI-corrected with
        # the comparison rather than computed from raw judge scores.
        "es_label": "ES" if any(r["es_value"] is not None for r in rows) else None,
        "p_col_header": p_col_header,
        "footer_fn": _footer,
    }
    return rows, meta


def _print_pairwise_section(
    bundle_or_result,
    *,
    top_pairwise: int = None,
    line_width: int,
    sort: bool = True,
    p_value_method: Optional[str] = None,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
    style: Literal["line", "gradient"] = "gradient",
    ci_alpha: Optional[float] = None,
) -> None:
    """Print the pairwise comparisons block for either a paired
    ``AnalysisBundle`` or an unpaired ``GroupComparisonResult``.

    Shared by the paired path's full bundle summary and the unpaired path's
    between-subjects summary (``print_group_comparison_summary``, also in
    this module) -- the axis/legend/header/row-rendering core is identical
    machinery either way (through ``_choose_interval_line``),
    so one function renders both instead of two independently-drifting
    implementations. What genuinely differs between the two designs (six
    CI/p-value method families + Friedman/Nemenyi + critical-difference
    rank bands for paired; one fixed Bonferroni-CI/Holm-p scheme for
    unpaired, no ranking bootstrap) is resolved up front by
    :func:`_prepare_paired_pairwise_rows`/:func:`_prepare_unpaired_pairwise_rows`
    into the common row + metadata shape this function actually renders.
    The Behavioral Agreement subsection (McNemar-style pass/fail bars) is
    NOT handled here -- see :func:`_print_behavioral_agreement_section`,
    paired-only, since ``agreement_mcc``/``binary_confusion`` need the same
    item scored by both entities, which has no between-subjects equivalent.

    Parameters
    ----------
    p_value_method : str or None
        Paired-only. Which p-value column to show. See the pre-unification
        docstring text preserved in :func:`_prepare_paired_pairwise_rows`
        for the full method-selection semantics. Ignored for unpaired data
        (that path's p-value display is controlled by
        ``GroupComparisonResult.show_p_values`` instead).
    pairwise_sort : {"grouped", "significance"}
        Sorting strategy for pairwise rows. ``"grouped"`` keeps a stable
        left-item grouping, while ``"significance"`` sorts by p-value then
        absolute effect size.
    """
    if isinstance(bundle_or_result, AnalysisBundle):
        rows, meta = _prepare_paired_pairwise_rows(
            bundle_or_result, p_value_method=p_value_method, sort=sort,
            pairwise_sort=pairwise_sort, alpha=ci_alpha,
        )
        if rows is None:
            return
    else:
        rows, meta = _prepare_unpaired_pairwise_rows(
            bundle_or_result, sort=sort, pairwise_sort=pairwise_sort,
        )

    _print_subsection(meta["section_header"])

    if top_pairwise is None:
        max_pairs = len(rows)
    else:
        max_pairs = max(0, min(top_pairwise, len(rows)))

    pair_item_col_width = meta["pair_item_col_width"]
    pair_stat_label = meta["pair_stat_label"]
    es_label = meta["es_label"]
    p_col_header = meta["p_col_header"]
    pair_p_col_width = max(10, len(p_col_header)) if p_col_header else 0

    if max_pairs > 0:
        pair_max_abs = max(
            1e-12,
            max(
                max(
                    abs(float(row["point_diff"])),
                    abs(float(row["ci_low"])),
                    abs(float(row["ci_high"])),
                    abs(float(row["point_diff"] - row["std_diff"])),
                    abs(float(row["point_diff"] + row["std_diff"])),
                )
                for row in rows[:max_pairs]
            ),
        )
        pair_low = -pair_max_abs
        pair_high = pair_max_abs
        # Clamp the shared axis ONCE for the whole block, not per row. A single
        # unbounded pair (paired._degenerate_pair_ci on zero-variance
        # differences with no declared score_range) makes pair_max_abs
        # infinite, and letting each row fall back to its own finite window
        # would silently put the rows on different scales -- two intervals of
        # identical width drawn at different lengths -- while the legend still
        # advertises one axis. The whole point of a shared axis is that bars
        # are comparable down the column, so the finite rows must keep sharing
        # it and the legend must report the axis actually drawn.
        if not (np.isfinite(pair_low) and np.isfinite(pair_high)):
            _cands: list[float] = []
            for row in rows[:max_pairs]:
                for key in ("point_diff", "ci_low", "ci_high"):
                    _cands.append(float(row[key]))
                _cands.append(float(row["point_diff"] - row["std_diff"]))
                _cands.append(float(row["point_diff"] + row["std_diff"]))
            _cands.append(0.0)  # the zero reference is always drawn
            pair_low, pair_high = _finite_axis(pair_low, pair_high, tuple(_cands))
        # gradient mode always produces a gradient (synthesized when multi_ci is absent)
        _any_multi_ci = (style == "gradient") or any(
            row.get("multi_ci") is not None for row in rows[:max_pairs]
        )
        _pair_ci_pct = int(round((1 - get_alpha_ci()) * 100))
        _pair_ci_legend = _legend_ci_label(style, _pair_ci_pct, _any_multi_ci)
        _pair_mean_marker = _mean_marker_legend(style, pair_stat_label.lower())
        print(
            f"{_DIM}  legend: (· ±1σ, {_pair_ci_legend}{_pair_mean_marker}, │ zero)    "
            f"axis: [{pair_low:+.3f}, {pair_high:+.3f}]    "
            f"effect: {meta['effect_label']}{_RESET}"
        )
        header = (
            f"  {'Left':<{pair_item_col_width}s} {'Right':<{pair_item_col_width}s} "
            f"{'Interval Plot':<{line_width}s} "
            f"{pair_stat_label:>7s} "
            f"{'CI Low':>8s} {'CI High':>8s}"
        )
        # Label efficiency, in TWO independent groups, each sitting beside the
        # quantity it describes. They are different estimands and generally
        # different numbers: a mean-difference interval's variance depends on
        # the correlation of the PAIRED DIFFERENCES, while the rank test's
        # depends on its own rank-based correlation. One shared column would
        # misdescribe whichever it was not computed for.
        #
        # Shown only when EVERY row has the pair: a half-populated column reads
        # as "this pair has no alignment" rather than "the extra call did not
        # run", so it is all or nothing.
        def _all_have(*keys):
            return bool(rows) and all(
                all(r.get(kk) is not None for kk in keys) for r in rows
            )
        _show_ci_eff = _all_have("ci_rho2", "ci_n_eff")
        _show_p_eff = bool(p_col_header) and _all_have("rho2", "n_eff")
        if _show_ci_eff:
            header += f" {'rho2(CI)':>8s} {'Neff(CI)':>8s}"
        if es_label:
            header += f" {es_label:>8s}"
        if p_col_header:
            header += f" {p_col_header:>{pair_p_col_width}s}"
        if _show_p_eff:
            header += f" {'rho2(p)':>7s} {'Neff(p)':>7s}"
        print(header)

        for row_data in rows[:max_pairs]:
            line = _choose_interval_line(
                mean=float(row_data["point_diff"]),
                ci_low=float(row_data["ci_low"]),
                ci_high=float(row_data["ci_high"]),
                spread_low=float(row_data["point_diff"] - row_data["std_diff"]),
                spread_high=float(row_data["point_diff"] + row_data["std_diff"]),
                axis_low=pair_low,
                axis_high=pair_high,
                width=line_width,
                style=style,
                multi_ci=row_data.get("multi_ci"),
            )
            left_label = _truncate_label(str(row_data["left"]), pair_item_col_width)
            right_label = _truncate_label(str(row_data["right"]), pair_item_col_width)
            row_str = (
                f"  {left_label:<{pair_item_col_width}s} "
                f"{right_label:<{pair_item_col_width}s} "
                f"{line:<{line_width}s} "
                f"{float(row_data['point_diff']):+7.3f} "
                f"{float(row_data['ci_low']):+8.3f} "
                f"{float(row_data['ci_high']):+8.3f}"
            )
            if _show_ci_eff:
                row_str += (f" {float(row_data['ci_rho2']):>8.2f}"
                            f" {float(row_data['ci_n_eff']):>8.0f}")
            if es_label:
                row_str += f" {float(row_data['es_value']):>8.3f}"
            if p_col_header:
                row_str += f" {_format_p_value(row_data.get('display_p')):>{pair_p_col_width}s}"
            if _show_p_eff:
                row_str += (f" {float(row_data['rho2']):>7.2f}"
                            f" {float(row_data['n_eff']):>7.0f}")
            print(row_str)

    if max_pairs == 0:
        print("  (no pairwise comparisons)")
    else:
        meta["footer_fn"](rows, max_pairs)


def _print_behavioral_agreement_section(
    bundle: "AnalysisBundle",
    *,
    top_pairwise: int = None,
    sort: bool = True,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    """Print the Pass/Fail Agreement (McNemar-style) subsection for binary
    paired data. Paired-only, by design: ``agreement_mcc``/``binary_confusion``
    require the *same item* scored by both entities to know whether they got
    it right or wrong together, which has no between-subjects equivalent
    (disjoint groups have no shared items at all).
    """
    rows, meta = _prepare_paired_pairwise_rows(
        bundle, p_value_method=None, sort=sort, pairwise_sort=pairwise_sort,
    )
    if rows is None:
        return
    if top_pairwise is None:
        max_pairs = len(rows)
    else:
        max_pairs = max(0, min(top_pairwise, len(rows)))

    agr_rows = [
        r for r in rows[:max_pairs]
        if r.get("agreement_mcc") is not None and r.get("binary_confusion") is not None
    ]
    agr_rows.sort(key=lambda row: float(row["agreement_mcc"]), reverse=True)
    if not agr_rows:
        return

    bar_width = 20
    mcc_col_width = 6
    strength_col_width = 14
    agr_item_col_width = meta["pair_item_col_width"]

    _print_subsection("\n--- Pass/Fail Agreement ---")
    print(f"  Are pairs getting the same items right and wrong?")
    print(f"  █ both pass  ░ both fail  {_BRIGHT_RED}▒{_RESET} disagree  "
          f"(MCC: 1=identical, 0=random, −1=opposite)")
    print()

    agr_header = (
        f"  {'Left':<{agr_item_col_width}s} {'Right':<{agr_item_col_width}s}"
        f" {'Plot':<{bar_width+2}s} {'MCC':>{mcc_col_width}s}"
        f"  {'Agreement':<{strength_col_width}s}  Interpretation"
    )
    print(agr_header)

    for row in agr_rows:
        n11, n10, n01, n00 = row["binary_confusion"]
        bar = _agreement_bar(n11, n10, n01, n00, width=bar_width)
        mcc = row["agreement_mcc"]
        left_label = _truncate_label(str(row["left"]), agr_item_col_width)
        right_label = _truncate_label(str(row["right"]), agr_item_col_width)
        print(
            f"  {left_label:<{agr_item_col_width}s} {right_label:<{agr_item_col_width}s}"
            f" [{bar}] {mcc:>+{mcc_col_width}.3f}"
            f"  {_mcc_strength(mcc):<{strength_col_width}s}  {_mcc_interpretation(mcc)}"
        )
    print()


def _print_mean_advantage(
    *,
    labels: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    multi_ci_per_entity: list,
    resolved_ci_method: str,
    item_singular: str = "template",
    line_width: int,
    template_col_width: int = 24,
    style: Literal["line", "gradient"] = "gradient",
    n_eff_per_entity: Optional[list] = None,
    rho2_per_entity: Optional[list] = None,
    ci_alpha: Optional[float] = None,
) -> None:
    """Print the absolute performance interval-plot table for a set of entities.

    Shows each entity's absolute mean with marginal bootstrap CIs (single-sample,
    independent per entity) and intrinsic spread bands.  A reference line marks
    the grand mean of the passed entities for visual comparison.

    Shared by the paired path (``_print_bundle_summary``, entities all scored
    on the same items -- a ``RobustnessResult``) and the unpaired path
    (``print_group_comparison_summary``, also in this module, disjoint
    items per group -- a ``list[GroupStat]``) -- both reduce to the same "N
    entities, each with a mean/CI/spread" shape by the time they call this,
    so one function renders both instead of two independently-drifting
    per-entity loops. ``multi_ci_per_entity`` takes an already-per-entity-
    sliced list (``{alpha: (lo, hi)}`` or ``None`` per entity) rather than a
    combined dict-of-arrays, so callers with either shape (a
    ``RobustnessResult.multi_ci`` sliced via ``_rob_multi_ci_at``, or a
    ``GroupStat.multi_ci`` list that's already this shape) both fit without
    conversion inside this function.
    """
    item_singular_title = item_singular.capitalize()
    stat_label = "Mean"
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    ref_val = float(np.mean(mean))

    # ±1σ spread around the absolute mean.
    sigma_lows = mean - std
    sigma_highs = mean + std

    # Axis bounds: cover means, CIs, and ±1σ spread.
    all_vals = np.concatenate([mean, ci_low, ci_high, sigma_lows, sigma_highs])
    val_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(val_range * 0.05, 1e-4)
    ma_low = float(np.min(all_vals)) - pad
    ma_high = float(np.max(all_vals)) + pad

    # The method goes in a footnote line under the table, the way the pairwise
    # section states its own. It used to be baked into this header off a
    # lookup, which had no entry for the ppi_* names and so announced
    # "marginal bootstrap CIs" for every PPI run -- naming a method that had
    # not been used.
    ci_pct = int(round((1 - (get_alpha_ci() if ci_alpha is None else ci_alpha)) * 100))
    _print_subsection(f"--- {stat_label} Performance (marginal {ci_pct}% CIs) ---")
    ref_label = "grand mean"
    _any_multi_ci = any(m is not None for m in multi_ci_per_entity)
    _ci_legend_ma = _legend_ci_label(style, ci_pct, _any_multi_ci)
    _mean_marker_ma = _mean_marker_legend(style, stat_label.lower())
    print(
        f"{_DIM}  axis: [{ma_low:.3f}, {ma_high:.3f}]"
        f"  (· ±1σ, {_ci_legend_ma}{_mean_marker_ma}, │ {ref_label}){_RESET}"
    )
    _show_neff = (
        n_eff_per_entity is not None
        and len(n_eff_per_entity) == len(labels)
        and all(v is not None for v in n_eff_per_entity)
    )
    _show_rho2 = (
        rho2_per_entity is not None
        and len(rho2_per_entity) == len(labels)
        and all(v is not None for v in rho2_per_entity)
    )
    # Widths and separators mirror the value row below exactly (7, 8, 8 with a
    # space between each); the header used to be 8/9/9 with no space between
    # the two CI columns, so it sat two characters right of its own numbers.
    print(
        f"  {item_singular_title:<{template_col_width}s} {'Interval Plot':<{line_width}s} {stat_label:>7s} "
        f"{'CI Low':>8s} {'CI High':>8s}"
        + (f" {'rho^2':>7s}" if _show_rho2 else "")
        + (f" {'N_eff':>7s}" if _show_neff else "")
    )
    for i, label in enumerate(labels):
        template_label = _truncate_label(label, template_col_width)
        line = _choose_interval_line(
            mean=float(mean[i]),
            ci_low=float(ci_low[i]),
            ci_high=float(ci_high[i]),
            spread_low=float(sigma_lows[i]),
            spread_high=float(sigma_highs[i]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
            reference=ref_val,
            style=style,
            multi_ci=multi_ci_per_entity[i],
        )
        print(
            f"  {template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{float(mean[i]):>7.3f} "
            f"{float(ci_low[i]):>8.3f} "
            f"{float(ci_high[i]):>8.3f}"
            + (f" {float(rho2_per_entity[i]):>7.2f}" if _show_rho2 else "")
            + (f" {float(n_eff_per_entity[i]):>7.0f}" if _show_neff else "")
        )

    # "marginal", not a bare "CI method:", so it reads distinctly from the
    # pairwise section's own line -- these are different methods on the same
    # run (here Logit-t against Paired NIG).
    _marginal_method = _pretty_marginal_ci_method(resolved_ci_method)
    if _marginal_method:
        print(f"{_DIM}  {ci_pct}% CI method: {_marginal_method}{_RESET}")


def _pretty_marginal_ci_method(code: Optional[str]) -> Optional[str]:
    """Display name for the single-sample CI method actually used.

    Unrecognized codes are returned as-is rather than mapped to a default, so
    a new method never gets announced under another one's name. A ``ppi_``
    prefix is split off and shown as such.
    """
    raw = (code or "").strip()
    if not raw:
        return None
    base, is_ppi = raw, False
    # The two spellings that reach here: resolved_ci_method's "ppi_logit_t"
    # and PairedDiffResult.test_method's already-prefixed "PPI ppi_logit_t".
    if base.upper().startswith("PPI "):
        base, is_ppi = base[4:].strip(), True
    if base.lower().startswith("ppi_"):
        base, is_ppi = base[4:], True
    names = {
        "wilson": "Wilson", "jeffreys": "Jeffreys", "clopper_pearson": "Clopper-Pearson",
        "newcombe": "Newcombe", "newcombe_mover": "Newcombe MOVER",
        "bayes_binary": "Bayesian (binary)", "bayes_indep": "Bayesian",
        "bayes_indep_comp": "Bayesian", "bayes_paired_comp": "Bayesian (paired)",
        "nig": "NIG", "logit_t": "Logit-t", "logit_t_dither": "Logit-t (dithered)",
        "t_interval": "t-interval", "beta": "Beta", "el": "Empirical likelihood",
        "bootstrap": "Bootstrap", "smooth_bootstrap": "Smooth bootstrap",
        "smooth_bootstrap_dither": "Smooth bootstrap (dithered)",
        "bootstrap_t": "Bootstrap-t", "bca": "BCa",
        "bonett_price": "Bonett-Price", "mj_floor": "May-Johnson (floored)",
        "mj_unfloored": "May-Johnson", "tango_scc": "Tango score",
        "tango_exact": "Tango exact", "wald_indep": "Wald", "wald": "Wald",
    }
    pretty = names.get(base.lower())
    if pretty is None:                      # unknown: never rename, just tidy
        pretty = base[0].upper() + base[1:] if base else base
    return f"PPI {pretty}" if is_ppi else pretty


def _print_ppi_banner() -> None:
    """Print the standard "PPI-CORRECTED" banner.

    Shared by the paired path's ``_print_bundle_summary`` and the unpaired
    path's ``print_group_comparison_summary`` (also in this module) --
    previously two copy-pasted, near-identical blocks; unified so a banner
    text/formatting change only needs to happen once.

    Does not reprint the alignment report itself: judge_alignment(...) is a
    required step before alignment=/PPI correction can be used at all, so
    by the time compare() prints this, the caller has already seen (or can
    call) judge_alignment(...).summary() -- reprinting it here on every
    compare() call duplicated it for no benefit.
    """
    banner = "═" * 58
    print(f"{_BOLD}{_BRIGHT_MAGENTA}{banner}{_RESET}")
    print(
        f"{_BOLD}{_BRIGHT_MAGENTA}PPI-CORRECTED. Every estimate below relies on the "
        f"judge_alignment(...) result passed via alignment=; see its .summary() for "
        f"the full alignment report.{_RESET}"
    )
    print(f"{_BOLD}{_BRIGHT_MAGENTA}{banner}{_RESET}")
    print()


def _print_omnibus_section(
    *,
    label: str,
    statistic: float,
    p_value: float,
    df: Optional[int] = None,
    corrected_p_value: Optional[float] = None,
    ppi_applied: bool = False,
    rho2_eff: Optional[float] = None,
    n_eff: Optional[float] = None,
    n_lab_per_entity: Optional[float] = None,
) -> None:
    """Print the boxed omnibus-test section, shared by the paired path's
    Friedman test and the unpaired path's Kruskal-Wallis / one-way ANOVA
    test -- its own subsection, ahead of the pairwise comparisons table.

    Originally two independently-drifting implementations (an inline line
    tucked inside the paired path's pairwise section vs. this boxed,
    multi-line form on the unpaired path); unified onto the unpaired path's
    presentation -- its own section, PPI-corrected p first, uncorrected
    stated explicitly with an anti-misuse warning -- since that is the
    clearer report of the two once a reader has to choose.

    ``df`` is Friedman-only (Kruskal-Wallis/ANOVA's ``TestResult`` doesn't
    carry one) and is omitted from the statistic line when ``None``.
    """
    ppi_prefix = "PPI-" if ppi_applied else ""
    _print_subsection(f"--- Omnibus Test: {ppi_prefix}{label} ---")
    df_note = f"({df})" if df is not None else ""
    display_p = corrected_p_value if (ppi_applied and corrected_p_value is not None) else p_value
    p_color = _BRIGHT_GREEN if display_p <= 0.05 else _YELLOW
    if ppi_applied:
        if corrected_p_value is not None:
            cp_str = _format_p_value(corrected_p_value)
            print(f"  PPI-corrected p = {p_color}{cp_str}{_RESET}")
        if rho2_eff is not None and n_eff is not None:
            # Named for the estimand, not the judge: this is the correlation of
            # the two sides' influence functions for THIS test, which for
            # Friedman/Kruskal-Wallis is a rank agreement and runs well below
            # the raw judge-human agreement in the alignment report above.
            # Calling it "judge alignment" invited readers to check it against
            # the rho^2>=0.2 usability rule, which is a mean-scale threshold.
            kind = ("rank agreement" if any(k in label.lower() for k in ("friedman", "kruskal"))
                    else "score agreement")
            print(f"  {kind}  rho^2 = {rho2_eff:.2f}")
            line = f"  N_eff (effective sample size) = {n_eff:.0f} labels/condition"
            if n_lab_per_entity:
                line += f", {n_eff / n_lab_per_entity:.1f}x the {n_lab_per_entity:.0f} you labeled"
            print(line)
        p_str = _format_p_value(p_value)
        # Dimmed: it is shown for contrast with the corrected p above it, and
        # should not read as one of the reported results.
        print(f"{_DIM}  uncorrected: statistic = {statistic:.4f}{df_note}, p = {p_str}{_RESET}")
        print(f"{_DIM}      ^ do not report this one. It treats the judge's scores as "
              f"if they were human labels.{_RESET}")
    else:
        p_str = _format_p_value(display_p)
        print(f"  statistic = {statistic:.4f}{df_note}   p = {p_color}{p_str}{_RESET}")
    if display_p > 0.05:
        print(
            f"  {_YELLOW}[!] {ppi_prefix}{label} p > 0.05: no significant omnibus "
            f"effect — treat pairwise results with caution.{_RESET}"
        )
    print()


def _shape_line(bundle, item_singular: str, item_plural: str) -> str:
    """Describe the benchmark in the caller's own factor terms.

    compare() runs a single-factor comparison through the template slot of
    a one-model benchmark, so ``bundle.shape`` reads ``models=1, prompts=k``
    even when the factor is ``model``. When the caller named its factor
    (anything but the default template wording), say ``k <factor>s`` instead
    of exposing that internal slot.
    """
    if item_singular in ("template", "prompt") or item_plural in ("templates", "prompts"):
        return repr(bundle.shape)
    bench, shape = bundle.benchmark, bundle.shape
    parts = [f"{bench.n_templates} {item_plural}", f"{bench.n_inputs} inputs"]
    if shape.n_input_vars > 1:
        parts.append(f"{shape.n_input_vars} input vars")
    parts.append(f"{shape.n_evaluators} evaluator{'s' if shape.n_evaluators != 1 else ''}")
    if bench.n_runs > 1:
        parts.append(f"{bench.n_runs} runs")
    return " × ".join(parts)


def _print_bundle_summary(
    bundle: AnalysisBundle,
    *,
    ci_alpha: Optional[float] = None,
    rng_seed=_SEED_UNSET,
    top_pairwise: int = None,
    line_width: int,
    item_singular: str = "template",
    item_plural: str = "templates",
    p_value_method=_UNSET,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
    style: Literal["line", "gradient"] = "gradient",
    guidance: bool = True,
    min_meaningful_diff: Optional[float] = None,
    show_rank_probabilities: bool = False,
    pareto: Optional[dict] = None,
    metric: Optional[str] = None,
) -> None:
    if p_value_method is _UNSET:
        p_value_method = bundle.p_value_method
    template_col_width = min(
        24, max(len(item_singular), max(len(l) for l in bundle.robustness.labels)) + 2
    )

    print(f"Shape: {_shape_line(bundle, item_singular, item_plural)}")
    n_runs = bundle.benchmark.n_runs
    item_singular_title = item_singular.capitalize()
    item_plural_title = item_plural.capitalize()
    print(
        f"{item_plural_title}: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {n_runs}" if n_runs > 1 else "")
        + _seed_note(rng_seed)
    )
    print()

    _eff_alpha = get_alpha_ci() if ci_alpha is None else ci_alpha
    _ppi_on = getattr(bundle, "ppi_applied", False)
    if _ppi_on:
        _print_ppi_banner()

    # Only the mean of this table is PPI-corrected; median, std, cv, iqr,
    # cvar_10 and the percentiles describe the raw judge scores, and cv is
    # std over the RAW mean so it contradicts the corrected mean beside it.
    # Correcting them needs a PPI quantile estimator this package does not
    # have, and printing them under the "every estimate below is corrected"
    # banner invites them to be read as corrected. The surviving mean is
    # already in Mean Performance below, with a CI and N_eff.
    if not _ppi_on:
        _print_subsection("--- Descriptive Statistics ---")
        _rob_df = bundle.robustness.summary_table()
        _rob_df.index.name = item_singular
        print(_rob_df.to_string())
        print()

    if show_rank_probabilities:
        _print_subsection(f"--- Rank Probabilities ({_rank_method_label(bundle)}) ---")
        max_rank_label_len = max((len(label) for label in bundle.rank_dist.labels), default=0)
        rank_label_col_width = min(40, max(len(item_singular_title) + 1, max_rank_label_len + 2))
        rank_bar_width = 14
        n_ranked_items = len(bundle.rank_dist.labels)
        print(
            f"  {item_singular_title:<{rank_label_col_width}s} "
            f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
            f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
        )
        for i, label in enumerate(bundle.rank_dist.labels):
            rank_label = _truncate_label(label, rank_label_col_width)
            p_best = float(bundle.rank_dist.p_best[i])
            expected_rank = float(bundle.rank_dist.expected_ranks[i])
            p_color = _p_best_color(p_best)
            p_reset = _RESET if p_color else ""
            p_str = f"{p_best:>8.1%} {_ratio_bar(p_best, width=rank_bar_width)}"
            print(
                f"  {rank_label:<{rank_label_col_width}s} "
                f"{p_color}{p_str}{p_reset} "
                f"{expected_rank:>8.2f} {_rank_hump_lane(expected_rank, n_ranked_items, width=rank_bar_width)}"
            )
        print("  E[Rank] lane: left is better (#1); peak is sharper near integer ranks, softer near half-ranks")
        print()

    _print_mean_advantage(
        labels=bundle.robustness.labels,
        mean=bundle.robustness.mean,
        std=bundle.robustness.std,
        ci_low=bundle.robustness.ci_low,
        ci_high=bundle.robustness.ci_high,
        multi_ci_per_entity=[
            _rob_multi_ci_at(bundle.robustness.multi_ci, i)
            for i in range(len(bundle.robustness.labels))
        ],
        resolved_ci_method=bundle.resolved_ci_method,
        item_singular=item_singular,
        line_width=line_width,
        template_col_width=template_col_width,
        style=style,
        n_eff_per_entity=getattr(bundle, "_marginal_n_eff", None),
        rho2_per_entity=getattr(bundle, "_marginal_rho2", None),
        ci_alpha=_eff_alpha,
    )
    print()

    if bundle.pairwise.friedman is not None:
        fr = bundle.pairwise.friedman
        _om_eff = getattr(bundle, "_omnibus_eff", None)
        _print_omnibus_section(
            label="Friedman",
            statistic=fr.statistic,
            df=fr.df,
            p_value=fr.p_value,
            corrected_p_value=fr.corrected_p_value,
            ppi_applied=getattr(bundle, "ppi_applied", False),
            rho2_eff=_om_eff[0] if _om_eff else None,
            n_eff=_om_eff[1] if _om_eff else None,
            n_lab_per_entity=getattr(bundle, "_n_lab_per_entity", None),
        )

    _print_pairwise_section(
        bundle,
        top_pairwise=top_pairwise,
        line_width=line_width,
        p_value_method=p_value_method,
        pairwise_sort=pairwise_sort,
        style=style,
        ci_alpha=_eff_alpha,
    )
    _print_behavioral_agreement_section(
        bundle,
        top_pairwise=top_pairwise,
        pairwise_sort=pairwise_sort,
    )

    # Seed variance section (only when seeded data is present).
    if bundle.seed_variance is not None:
        print()
        _print_seed_variance(
            bundle.seed_variance,
            template_col_width=template_col_width,
            item_singular=item_singular,
        )

    # LMM diagnostics (standard one-factor LMM).
    if bundle.lmm_info is not None:
        print()
        _print_lmm_summary(bundle)

    # Factorial LMM diagnostics (factor tests + marginal means).
    if bundle.factorial_lmm_info is not None:
        print()
        _print_factorial_lmm_summary(bundle, item_singular=item_singular, style=style)

    # Pareto front (primary metric vs. a secondary metric), when present --
    # printed right before the executive summary so the secondary-metric-
    # corrected, holistic verdict sits next to the primary-metric-only
    # leaderboard rather than trailing after everything else.
    if pareto is not None:
        print()
        _print_pareto_section(pareto, metric=metric, show_rank_probabilities=show_rank_probabilities)

    # Executive summary leaderboard (near the end — immediately visible in terminal).
    print()
    _print_executive_summary(bundle, item_singular=item_singular, pareto=pareto, metric=metric)
    if pareto is not None:
        _print_pareto_callout(pareto, metric=metric)

    # Not on the PPI path. The block's sample-size projection is a heuristic
    # that was never validated against a corrected comparison -- and it is
    # degenerate whenever its CI-width branch dominates, returning N*6.25
    # regardless of the interval, so it does not respond to the correction it
    # would appear to describe. The uncorrected paths keep it unchanged.
    if guidance and not _ppi_on:
        _print_next_steps_guidance(
            bundle,
            item_plural=item_plural,
            min_meaningful_diff=min_meaningful_diff,
        )


# ---------------------------------------------------------------------------
# Seed variance section
# ---------------------------------------------------------------------------

_BLOCK_CHARS = "▁▂▃▄▅▆▇█"

# Shared between the seed-variance detail table and both executive-summary
# leaderboards' "Per-run noise" column, so the header label and the column
# width it's padded to can't drift apart (the header text is longer than the
# strip itself, so the column must be sized off the header, not the strip).
_NOISE_STRIP_HEADER = "Per-run noise"
_NOISE_STRIP_CHARS = 8
_NOISE_COL_W = max(_NOISE_STRIP_CHARS, len(_NOISE_STRIP_HEADER))


def _seed_noise_strip(
    per_cell_values: np.ndarray,
    scale_max: float,
    max_width: int = 40,
) -> str:
    """One Unicode block char per input, scaled against ``scale_max``.

    If there are more inputs than ``max_width``, inputs are averaged into
    bins first so the strip always fits within the column.
    """
    m = len(per_cell_values)
    if m == 0:
        return ""
    if scale_max <= 0:
        return _BLOCK_CHARS[0] * min(m, max_width)
    if m > max_width:
        bins = np.array_split(per_cell_values, max_width)
        values = np.array([b.mean() for b in bins])
    else:
        values = per_cell_values
    chars = []
    for v in values:
        idx = int(round(float(v) / scale_max * (len(_BLOCK_CHARS) - 1)))
        chars.append(_BLOCK_CHARS[max(0, min(idx, len(_BLOCK_CHARS) - 1))])
    return "".join(chars)


def _print_seed_variance(
    sv: SeedVarianceResult,
    template_col_width: int = 24,
    strip_width: int = 24,
    item_singular: str = "template",
) -> None:
    """Print seed variance decomposition with per-input heat strip."""
    _print_subsection(f"--- Per-input Variance Across Runs (R={sv.n_runs} runs) ---")
    global_cell_max = float(sv.per_cell_seed_std.max())
    print(
        f"  key: ▁–█ = per-input noise   "
        f"(globally scaled; █ = {global_cell_max:.4f})"
    )
    # Wide enough for "instability" (11 chars) -- with num_w=10 that header
    # overflowed its own field by 1 char and dragged every header after it
    # out of alignment with the data rows below.
    num_w = len("instability")
    consistency_w = 18
    verdicts = [_instability_label(float(v)) for v in sv.instability]
    verdict_w = max(len("Verdict"), max(len(v) for v in verdicts))
    print(
        f"  {item_singular.capitalize():<{template_col_width}s}  "
        f"{'Per-input noise':<{strip_width}s}  "
        f"{'run_std':>{num_w}s}  "
        f"{'input_std':>{num_w}s}  "
        f"{'total_std':>{num_w}s}  "
        f"{'instability':>{num_w}s}  "
        f"{'Consistency (ICC)':<{consistency_w}s}  "
        f"{'Verdict':<{verdict_w}s}"
    )
    for i, label in enumerate(sv.labels):
        strip = _seed_noise_strip(
            sv.per_cell_seed_std[i], global_cell_max, max_width=strip_width
        )
        instability = float(sv.instability[i])
        verdict = verdicts[i]
        verdict_color = _instability_color(instability)
        icc = float(sv.icc[i])
        icc_str = "—" if np.isnan(icc) else f"{icc:.2f} ({_consistency_label(icc)})"
        icc_color = _consistency_color(icc)
        print(
            f"  {_truncate_label(label, template_col_width):<{template_col_width}s}  "
            f"{strip:<{strip_width}s}  "
            f"{np.sqrt(sv.seed_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.input_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.total_var[i]):>{num_w}.4f}  "
            f"{instability:>{num_w}.4f}  "
            f"{icc_color}{icc_str:<{consistency_w}s}{_RESET}  "
            f"{verdict_color}{verdict:<{verdict_w}s}{_RESET}"
        )
    print(
        f"{_DIM}  instability = how many points a score typically moves "
        f"between repeated runs{_RESET}"
    )
    print(
        f"{_DIM}  Consistency (ICC) = how much of the difference between "
        f"inputs is real signal, rather than run-to-run noise{_RESET}"
    )
    print()


def _collect_cross_model_seed_instability_rows(
    bundle: MultiModelBundle,
) -> list[tuple[str, float, float, float, str, float]]:
    """Collect sorted per-model instability rows for summary tables."""
    rows: list[tuple[str, float, float, float, str, float]] = []
    for model_label, model_bundle in bundle.per_model.items():
        sv = model_bundle.seed_variance
        if sv is None:
            continue

        overall_instability = float(np.mean(sv.per_cell_seed_std))
        template_instability_mean = float(np.mean(sv.instability))
        template_instability_std = float(np.std(sv.instability, ddof=0))

        noisiest_idx = int(np.argmax(sv.instability))
        noisiest_template = sv.labels[noisiest_idx]
        noisiest_value = float(sv.instability[noisiest_idx])

        rows.append((
            model_label,
            overall_instability,
            template_instability_mean,
            template_instability_std,
            noisiest_template,
            noisiest_value,
        ))

    rows.sort(key=lambda row: row[1])
    return rows


def _print_cross_model_seed_instability(
    bundle: MultiModelBundle,
    *,
    rows: Optional[list[tuple[str, float, float, float, str, float]]] = None,
) -> None:
    """Print cross-model instability comparison when seed variance is available."""
    if rows is None:
        rows = _collect_cross_model_seed_instability_rows(bundle)

    if len(rows) == 0:
        return

    print()
    _print_subsection("--- Cross-Model Instability (across templates & inputs) ---")
    print(
        "  lower is better (more stable): "
        "instability = mean within-cell run std"
    )
    model_w = max(16, min(34, max(len(row[0]) for row in rows)))
    print(
        f"  {'Model':<{model_w}s} "
        f"{'instability':>12s} "
        f"{'tpl_mean':>10s} "
        f"{'tpl_std':>9s} "
        f"{'Noisiest template':<24s} "
        "Verdict"
    )

    for (
        model_label,
        overall_instability,
        template_instability_mean,
        template_instability_std,
        noisiest_template,
        noisiest_value,
    ) in rows:
        noisiest_desc = f"{_truncate_label(noisiest_template, 16)} ({noisiest_value:.4f})"
        verdict = _instability_label(overall_instability)
        verdict_color = _instability_color(overall_instability)
        print(
            f"  {_truncate_label(model_label, model_w):<{model_w}s} "
            f"{overall_instability:>12.4f} "
            f"{template_instability_mean:>10.4f} "
            f"{template_instability_std:>9.4f} "
            f"{noisiest_desc:<24s} "
            f"{verdict_color}{verdict}{_RESET}"
        )
    print()


# ---------------------------------------------------------------------------
# LMM diagnostics
# ---------------------------------------------------------------------------

def _print_lmm_summary(bundle: AnalysisBundle) -> None:
    """Print LMM variance-component diagnostics for a standard (one-factor) LMM."""
    info = bundle.lmm_info
    if info is None:
        return
    _print_subsection("--- LMM Diagnostics ---")
    print(f"  Formula : {info.formula}")
    print(
        f"  ICC={info.icc:.3f}  σ_input={info.sigma_input:.4f}  "
        f"σ_resid={info.sigma_resid:.4f}  n_obs={info.n_obs}"
        + ("" if info.converged else f"  {_YELLOW}[convergence warning]{_RESET}")
    )


def _print_factorial_lmm_summary(
    bundle: AnalysisBundle,
    *,
    item_singular: str = "template",
    style: Literal["line", "gradient"] = "gradient",
) -> None:
    """Print factorial LMM diagnostics: variance components, factor tests, marginal means."""
    info = bundle.factorial_lmm_info
    if info is None:
        return

    # Build a display-name map from internal slot names ("model", "prompt") to
    # the user's original factor names when item_singular is a pipe-joined label.
    _std_slots = ["model", "prompt"]
    if "|" in item_singular:
        _parts = item_singular.split("|")
        _factor_display = {_std_slots[i]: _parts[i] for i in range(min(len(_parts), len(_std_slots)))}
    else:
        _factor_display = {}

    _print_subsection("--- Factorial LMM Diagnostics ---")
    print(f"  Formula : {info.formula}")
    print(
        f"  ICC={info.icc:.3f}  σ_input={info.sigma_input:.4f}  "
        f"σ_resid={info.sigma_resid:.4f}  n_obs={info.n_obs}"
        + ("" if info.converged else f"  {_YELLOW}[convergence warning]{_RESET}")
    )
    print()

    # Factor / interaction Wald tests
    _print_subsection("--- Factor Tests (Wald χ²) ---")
    ft = info.factor_tests
    if ft is not None and len(ft) > 0:
        ft_sorted = ft.sort_values(["p_value", "statistic"], ascending=[True, False])
        term_w = min(42, max(len("Term"), max(len(str(t)) for t in ft_sorted["term"]) + 2))
        bar_w = 12
        print(
            f"  {'Term':<{term_w}s}  {'χ²':>10s}  {'df':>4s}  {'p-value':>12s}  {'Evidence':<{bar_w}s}"
        )
        print(f"  {'-' * term_w}  {'-' * 10}  {'-' * 4}  {'-' * 12}  {'-' * bar_w}")
        for _, row in ft_sorted.iterrows():
            pval = float(row["p_value"])
            evidence = 1.0 - float(np.clip(pval, 0.0, 1.0)) if not np.isnan(pval) else np.nan
            p_str = _format_p_value(pval)
            print(
                f"  {_truncate_label(str(row['term']), term_w):<{term_w}s}  "
                f"{float(row['statistic']):>10.3f}  "
                f"{float(row['df']):>4.0f}  "
                f"{p_str:>12s}  "
                f"{_ratio_bar(evidence, width=bar_w)}"
            )
        n_sig = int(np.sum(ft_sorted["p_value"].to_numpy(dtype=float) < 0.05))
        if n_sig == 0:
            print(
                f"  {_YELLOW}[!] No factor/interaction terms pass p < 0.05; "
                "interpret level differences cautiously." + f"{_RESET}"
            )
        else:
            print(f"  Significant terms (p < 0.05): {n_sig}/{len(ft_sorted)}")
    else:
        print("  (no factor tests available)")

    _print_factorial_interaction_plot(bundle, factor_tests=ft)

    # Estimated marginal means per factor
    mm = info.marginal_means
    if mm:
        line_width = 41
        ci_pct = int(round((1 - get_alpha_ci()) * 100))
        for factor_name, mm_df in mm.items():
            display_name = _factor_display.get(factor_name, factor_name)
            print()
            _print_subsection(f"--- Marginal Means: {display_name} ---")
            if len(mm_df) == 0:
                print("  (no marginal means available)")
                continue

            mm_sorted = mm_df.sort_values(["mean", "level"], ascending=[False, True]).reset_index(drop=True)
            means = mm_sorted["mean"].to_numpy(dtype=float)
            ci_low = mm_sorted["ci_low"].to_numpy(dtype=float)
            ci_high = mm_sorted["ci_high"].to_numpy(dtype=float)

            factor_center = float(np.mean(means))
            centered_mean = means - factor_center
            centered_low = ci_low - factor_center
            centered_high = ci_high - factor_center

            axis_max = max(
                1e-12,
                float(
                    np.max(
                        np.abs(
                            np.concatenate([centered_mean, centered_low, centered_high])
                        )
                    )
                ),
            )
            axis_low = -axis_max
            axis_high = axis_max
            level_w = min(28, max(len("Level"), max(len(str(v)) for v in mm_sorted["level"]) + 2))

            _ci_legend_mm = _legend_ci_label(style, ci_pct, style == "gradient")
            _mean_marker_lmm = _mean_marker_legend(style, "mean")
            print(
                f"{_DIM}  axis: [{axis_low:+.3f}, {axis_high:+.3f}]  "
                f"(· ±SE, {_ci_legend_mm}{_mean_marker_lmm}, │ factor mean){_RESET}"
            )
            print(
                f"  {'Level':<{level_w}s} {'Interval Plot':<{line_width}s} "
                f"{'Mean':>8s} {'SE':>8s} {'CI Low':>9s} {'CI High':>9s} {'Δ vs avg':>10s}"
            )
            for i, row in mm_sorted.iterrows():
                _se = float(row["se"])
                _synth_mc = _synth_multi_ci_from_se(float(centered_mean[i]), _se)
                interval_line = _choose_interval_line(
                    mean=float(centered_mean[i]),
                    ci_low=float(centered_low[i]),
                    ci_high=float(centered_high[i]),
                    spread_low=float(centered_mean[i]) - _se,
                    spread_high=float(centered_mean[i]) + _se,
                    axis_low=axis_low,
                    axis_high=axis_high,
                    width=line_width,
                    style=style,
                    multi_ci=_synth_mc,
                )
                print(
                    f"  {_truncate_label(str(row['level']), level_w):<{level_w}s} "
                    f"{interval_line:<{line_width}s} "
                    f"{float(row['mean']):>8.4f}  "
                    f"{float(row['se']):>8.4f}  "
                    f"{float(row['ci_low']):>9.4f}  "
                    f"{float(row['ci_high']):>9.4f}  "
                    f"{float(centered_mean[i]):>+10.4f}"
                )
            best_idx = int(np.argmax(means))
            best_level = str(mm_sorted.iloc[best_idx]["level"])
            best_mean = float(means[best_idx])
            print(
                f"  {_BRIGHT_GREEN}-> Highest marginal mean:{_RESET} "
                f"'{_BOLD}{_BRIGHT_GREEN}{best_level}{_RESET}' (mean={best_mean:.4f}, Δ={centered_mean[best_idx]:+.4f} vs factor average)"
            )


def _factor_names_from_term(term: str) -> list[str]:
    """Extract factor names from a model-term string such as ``C(a):C(b)``."""
    names = re.findall(r"C\(([^)]+)\)", str(term))
    if names:
        return names
    return [p.strip() for p in str(term).split(":") if p.strip()]


def _print_factorial_interaction_plot(
    bundle: AnalysisBundle,
    *,
    factor_tests,
    alpha: Optional[float] = None,
) -> None:
    """Render an optional terminal interaction plot via plotext when interaction is significant."""
    if alpha is None:
        alpha = get_alpha_ci()
    info = bundle.factorial_lmm_info
    if info is None or factor_tests is None or len(factor_tests) == 0:
        return

    is_interaction = factor_tests["term"].astype(str).str.contains(":", regex=False)
    if not bool(np.any(is_interaction)):
        return

    sig_interactions = factor_tests.loc[
        is_interaction & (factor_tests["p_value"].to_numpy(dtype=float) < alpha)
    ]
    if len(sig_interactions) == 0:
        return

    tf = bundle.benchmark.template_factors
    if tf is None or len(tf) != bundle.benchmark.n_templates:
        return

    best_row = sig_interactions.sort_values(["p_value", "statistic"], ascending=[True, False]).iloc[0]
    term = str(best_row["term"])
    factors = _factor_names_from_term(term)
    if len(factors) < 2:
        return

    x_factor, line_factor = factors[0], factors[1]
    if x_factor not in tf.columns or line_factor not in tf.columns:
        return

    tf_plot = tf.copy()
    tf_plot["_score"] = bundle.robustness.mean.astype(float)

    group_cols = [x_factor, line_factor]
    grouped = (
        tf_plot.groupby(group_cols, observed=True, dropna=False)["_score"]
        .mean()
        .reset_index()
    )
    if len(grouped) == 0:
        return

    x_levels = [str(v) for v in tf_plot[x_factor].drop_duplicates().tolist()]
    line_levels = [str(v) for v in tf_plot[line_factor].drop_duplicates().tolist()]

    x_map = {str(v): i for i, v in enumerate(x_levels)}

    grouped["_x_label"] = grouped[x_factor].astype(str)
    grouped["_line_label"] = grouped[line_factor].astype(str)
    grouped["_x_ord"] = grouped["_x_label"].map(x_map)
    grouped = grouped.sort_values(["_line_label", "_x_ord"]).reset_index(drop=True)

    print()
    _print_subsection("--- Interaction Plot (significant interaction) ---")
    print(
        f"  term='{term}'  (p={_format_p_value(float(best_row['p_value']))}); "
        f"x='{x_factor}', lines='{line_factor}'"
    )

    if len(factors) > 2:
        held = ", ".join(factors[2:])
        print(
            f"  {_YELLOW}[!] Higher-order interaction detected; plot shows only first two factors "
            f"and averages over: {held}.{_RESET}"
        )

    try:
        import plotext as plt  # type: ignore[import-not-found]
    except Exception:
        print(
            f"  {_YELLOW}[!] plotext not installed; skipping terminal interaction plot. "
            "Install with `pip install plotext` to enable this view."
            f"{_RESET}"
        )
        return

    try:
        plt.clear_figure()
        plt.canvas_color("default")
        plt.axes_color("default")
        plt.ticks_color("white")
        plt.plotsize(92, 22)
        plt.title(f"Interaction: {x_factor} × {line_factor}")
        plt.xlabel(x_factor)
        plt.ylabel("mean score")

        x_tick_vals = list(range(len(x_levels)))
        plt.xticks(x_tick_vals, x_levels)

        for line_level in line_levels:
            part = grouped[grouped["_line_label"] == line_level]
            if len(part) == 0:
                continue
            x_vals = part["_x_ord"].to_numpy(dtype=float).tolist()
            y_vals = part["_score"].to_numpy(dtype=float).tolist()
            plt.plot(x_vals, y_vals, marker="dot", label=line_level)

        grid_fn = getattr(plt, "grid", None)
        if callable(grid_fn):
            try:
                grid_fn(True, True)
            except TypeError:
                try:
                    grid_fn(True)
                except Exception:
                    pass

        legend_fn = getattr(plt, "legend", None)
        if callable(legend_fn):
            try:
                legend_fn(True)
            except TypeError:
                try:
                    legend_fn()
                except Exception:
                    pass

        plt.show()
    except Exception as exc:
        print(
            f"  {_YELLOW}[!] plotext rendering failed ({type(exc).__name__}: {exc}); "
            "continuing without plot."
            f"{_RESET}"
        )


# ---------------------------------------------------------------------------
# ASCII rendering primitives
# ---------------------------------------------------------------------------

def _gradient_interval_line(
    *,
    mean: float,
    multi_ci: dict[float, tuple[float, float]],
    spread_low: float,
    spread_high: float,
    axis_low: float,
    axis_high: float,
    width: int,
    reference: float = 0.0,
) -> str:
    """Render a one-line gradient CI plot using Unicode block characters.

    Opacity mapping (outermost → innermost), for the default
    ``GRADIENT_CI_ALPHAS`` of (0.32, 0.10, 0.05, 0.01):
      beyond 99% CI  → ' ' (invisible)
      95% – 99% CI   → '░' (10 % opacity)
      90% – 95% CI   → '▒' (medium)
      68% – 90% CI   → '▓' (high)
      inside 68% CI  → '█' (fully opaque)

    Bands are paired to ``sorted(multi_ci)`` positionally, so a caller passing a
    different alpha ladder gets the same outermost-to-innermost shading at
    whatever levels it supplied.

    The ±1σ spread dots ('·') appear only where they peek beyond all CI bands.
    Falls back to ``_ascii_interval_line`` when fewer than 2 CI levels are present.
    """
    if len(multi_ci) < 2:
        lo = min(v[0] for v in multi_ci.values()) if multi_ci else mean
        hi = max(v[1] for v in multi_ci.values()) if multi_ci else mean
        return _ascii_interval_line(
            mean=mean, ci_low=lo, ci_high=hi,
            spread_low=spread_low, spread_high=spread_high,
            axis_low=axis_low, axis_high=axis_high, width=width, reference=reference,
        )

    width = max(9, int(width))
    axis_low = float(axis_low)
    axis_high = float(axis_high)
    if axis_high <= axis_low:
        axis_low -= 1.0
        axis_high += 1.0

    def to_idx(x: float) -> int:
        x_clamped = min(max(float(x), axis_low), axis_high)
        pos = (x_clamped - axis_low) / (axis_high - axis_low)
        return int(round(pos * (width - 1)))

    chars = [" "] * width

    # ±1σ spread dots drawn first; CI gradient bands will overwrite them.
    lo_spread = min(to_idx(spread_low), to_idx(spread_high))
    hi_spread = max(to_idx(spread_low), to_idx(spread_high))
    for i in range(lo_spread, hi_spread + 1):
        chars[i] = "·"

    # CI gradient: fill widest CI first so inner bands overwrite outer ones.
    # Sorted ascending by alpha ⟹ widest CI first (smallest alpha = widest).
    sorted_alphas = sorted(multi_ci.keys())              # e.g. [0.001, 0.01, 0.05, 0.10]
    band_chars = ("░", "▒", "▓", "█")                   # paired outermost→innermost
    # band_chars = ("·", "┈", "┄", "━")                   # thin→thick outermost→innermost
    for alpha, char in zip(sorted_alphas, band_chars):
        lo_ci, hi_ci = multi_ci[alpha]
        lo_idx = min(to_idx(lo_ci), to_idx(hi_ci))
        hi_idx = max(to_idx(lo_ci), to_idx(hi_ci))
        for i in range(lo_idx, hi_idx + 1):
            chars[i] = char

    ref_idx = to_idx(reference)
    # No marker is drawn at the mean, deliberately. A point marker invites the
    # reader to treat one value as the answer and the interval around it as
    # decoration, which is the reading gradient plots exist to avoid
    # (Correll & Gleicher, "Error bars considered harmful"). `mean` is still a
    # parameter because _ascii_interval_line, the <2-band fallback above, does
    # mark it.
    chars[ref_idx] = "│"

    # The reference line can obscure the tail of a CI band when the tail
    # exactly ends on it. This might cause the user to miss the fact that the
    # CI band crosses the reference line. To mitigate this, we add a hint character
    # to the side to visually suggest the presence of a "crossing"—this way, users can
    # distinguish between bands that cross, and bands that get close to the |
    # but do not cross it.
    outer_lo, outer_hi = multi_ci[sorted_alphas[0]]
    if float(outer_lo) < float(reference) < float(outer_hi):
        hint_char = band_chars[0]
        if ref_idx > 0 and chars[ref_idx - 1] in {" ", "·"}:
            chars[ref_idx - 1] = hint_char
        if ref_idx + 1 < width and chars[ref_idx + 1] in {" ", "·"}:
            chars[ref_idx + 1] = hint_char

    return "".join(chars)


def _synth_multi_ci_from_se(
    mean: float, se: float
) -> Optional[dict[float, tuple[float, float]]]:
    """Synthesize a multi_ci gradient from a mean and standard error.

    Uses normal z-scaling, appropriate for Wald-type CIs (LMM, t-interval)
    where CI = mean ± z * se exactly.
    """
    if se <= 1e-12:
        return None
    import scipy.stats
    return {
        a: (
            mean - float(scipy.stats.norm.ppf(1.0 - a / 2.0)) * se,
            mean + float(scipy.stats.norm.ppf(1.0 - a / 2.0)) * se,
        )
        for a in GRADIENT_CI_ALPHAS
    }


def _finite_axis(
    axis_low: float,
    axis_high: float,
    candidates: tuple[float, ...],
) -> tuple[float, float]:
    """Return a finite (axis_low, axis_high) for the ASCII interval plots.

    Substitutes the spread of whatever finite values the row does carry when a
    supplied bound is non-finite, and falls back to a unit window around 0 when
    nothing finite is available. Purely a drawing concern -- the printed
    numeric bounds are untouched.
    """
    lo, hi = float(axis_low), float(axis_high)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return lo, hi
    finite = [float(c) for c in candidates if c is not None and np.isfinite(float(c))]
    if np.isfinite(lo):
        finite.append(lo)
    if np.isfinite(hi):
        finite.append(hi)
    if not finite:
        return -1.0, 1.0
    f_lo, f_hi = min(finite), max(finite)
    if f_hi <= f_lo:
        pad = max(abs(f_lo), 1.0) * 0.5
        return f_lo - pad, f_hi + pad
    pad = (f_hi - f_lo) * 0.15
    return f_lo - pad, f_hi + pad


def _choose_interval_line(
    *,
    mean: float,
    ci_low: float,
    ci_high: float,
    spread_low: float,
    spread_high: float,
    axis_low: float,
    axis_high: float,
    width: int,
    reference: float = 0.0,
    style: str = "gradient",
    multi_ci: Optional[dict[float, tuple[float, float]]] = None,
) -> str:
    """Dispatch to gradient or line renderer based on style and data availability."""
    # A non-finite bound is a real result, not a bug: paired._degenerate_pair_ci
    # reports (-inf, +inf) for a zero-variance difference on unbounded data,
    # where no finite interval has guaranteed coverage. Both renderers map a
    # value onto an axis via (x - axis_low) / (axis_high - axis_low), which is
    # NaN when the axis itself is infinite, so clamp the drawn axis to the
    # finite information in the row. The interval still *prints* as -inf/inf in
    # the CI Low/CI High columns -- this only bounds the little ASCII plot, and
    # an interval that runs off both ends of it is the correct picture.
    axis_low, axis_high = _finite_axis(
        axis_low, axis_high,
        candidates=(mean, ci_low, ci_high, spread_low, spread_high, reference),
    )
    effective_multi_ci = multi_ci
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        # The primary interval is unbounded, but the gradient bands handed in
        # can still be zero-width: paired.py builds its `mci` dict per alpha
        # from the method's own CI function, and t_interval_ci_1d keeps its
        # (mean, mean) contract on a zero-variance sample. The renderer
        # normally trusts multi_ci over the primary CI, which here would draw
        # a single opaque block at the point estimate -- the exact picture of
        # false certainty this branch exists to retract -- immediately beside
        # a printed -inf/+inf. Drop the bands and draw the primary interval,
        # so the plot says what the numbers say. Whether that marginal
        # contract should itself move is a separate, statistical question.
        effective_multi_ci = None
    if style == "gradient" and effective_multi_ci is None:
        # Synthesize gradient from the primary CI via normal z-scaling.
        # Appropriate for Wald-type CIs (LMM); a reasonable approximation elsewhere.
        half = (ci_high - ci_low) / 2.0
        if half > 1e-12:
            import scipy.stats
            z_stored = float(scipy.stats.norm.ppf(1.0 - get_alpha_ci() / 2.0))
            effective_multi_ci = _synth_multi_ci_from_se(mean, half / z_stored)
    if style == "gradient" and effective_multi_ci is not None and len(effective_multi_ci) >= 2:
        return _gradient_interval_line(
            mean=mean, multi_ci=effective_multi_ci,
            spread_low=spread_low, spread_high=spread_high,
            axis_low=axis_low, axis_high=axis_high,
            width=width, reference=reference,
        )
    return _ascii_interval_line(
        mean=mean, ci_low=ci_low, ci_high=ci_high,
        spread_low=spread_low, spread_high=spread_high,
        axis_low=axis_low, axis_high=axis_high,
        width=width, reference=reference,
    )


def _rob_multi_ci_at(
    rob_multi_ci: Optional[dict[float, tuple[np.ndarray, np.ndarray]]],
    idx: int,
) -> Optional[dict[float, tuple[float, float]]]:
    """Extract a single-template slice from a RobustnessResult.multi_ci dict."""
    if rob_multi_ci is None:
        return None
    return {a: (float(lo[idx]), float(hi[idx])) for a, (lo, hi) in rob_multi_ci.items()}


_CORRECTION_DISPLAY_NAMES = {
    "romano_wolf": "Romano-Wolf",
    "fdr_bh": "FDR (BH)",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "hochberg": "Hochberg",
    "shaffer": "Shaffer",
    "max_t": "max-T",
    "none": "none",
}


def _pretty_correction(code: Optional[str]) -> str:
    """Human-readable name for a FWER correction method code."""
    if not code:
        return "none"
    return _CORRECTION_DISPLAY_NAMES.get(code, code.replace("_", " ").title())


_SIMULTANEOUS_CI_DISPLAY_NAMES = {
    "max_t": "max-T",
    "sidak": "Šidák",
    "boot": "Joint bootstrap",
    "bonferroni": "Bonferroni",
}


def _pretty_simultaneous_ci(code: Optional[str]) -> str:
    """Human-readable name for a simultaneous-CI method code."""
    if not code:
        return "none"
    return _SIMULTANEOUS_CI_DISPLAY_NAMES.get(code, code.replace("_", " ").title())


def _legend_ci_label(style: str, ci_pct: int, multi_ci_available: bool) -> str:
    """Return the CI portion of a legend string for the given style."""
    if style == "gradient" and multi_ci_available:
        bands = "/".join(
            f"{100 * (1.0 - alpha):g}%"
            for alpha in sorted(GRADIENT_CI_ALPHAS)
        )
        return f"░▒▓█ CI gradient [{bands}]"
    return f"─ {ci_pct}% CI"


def _mean_marker_legend(style: str, label: str) -> str:
    """Return the ', ● <label>' legend fragment, or '' when the renderer
    won't actually draw a '●' anywhere.

    _gradient_interval_line (style="gradient", the default everywhere in
    this file) computes a mean index but never assigns a '●' character --
    only _ascii_interval_line (style="line") does. A legend that always
    said "● mean" regardless of style was misleading readers into looking
    for a marker that gradient output never draws.
    """
    if style == "gradient":
        return ""
    return f", ● {label}"


def _ascii_interval_line(
    *,
    mean: float,
    ci_low: float,
    ci_high: float,
    spread_low: float,
    spread_high: float,
    axis_low: float,
    axis_high: float,
    width: int,
    reference: float = 0.0,
) -> str:
    """Render a one-line ASCII interval plot with a reference marker.

    Parameters
    ----------
    reference : float
        Position of the ``│`` reference marker on the axis (default ``0.0``).
        Pass the grand mean when using absolute-scale plots so the reference
        line marks the average rather than zero.
    """
    width = max(9, int(width))
    axis_low = float(axis_low)
    axis_high = float(axis_high)
    if axis_high <= axis_low:
        axis_low -= 1.0
        axis_high += 1.0

    def to_idx(x: float) -> int:
        x_clamped = min(max(float(x), axis_low), axis_high)
        pos = (x_clamped - axis_low) / (axis_high - axis_low)
        return int(round(pos * (width - 1)))

    lo_spread_idx = min(to_idx(spread_low), to_idx(spread_high))
    hi_spread_idx = max(to_idx(spread_low), to_idx(spread_high))
    lo_ci_idx = min(to_idx(ci_low), to_idx(ci_high))
    hi_ci_idx = max(to_idx(ci_low), to_idx(ci_high))
    mean_idx = to_idx(mean)

    chars = [" "] * width
    for idx in range(lo_spread_idx, hi_spread_idx + 1):
        chars[idx] = "·"
    for idx in range(lo_ci_idx, hi_ci_idx + 1):
        chars[idx] = "─"

    ref_idx = to_idx(reference)
    chars[ref_idx] = "│"
    chars[mean_idx] = "●"

    return "".join(chars)


def _truncate_label(text: str, width: int) -> str:
    """Fit text into a fixed-width column with ellipsis when needed."""
    width = max(1, int(width))
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 1] + "…"


def _split_model_template_label(label: str) -> tuple[str, str]:
    """Split labels of the form 'model / template' into separate columns."""
    parts = label.split(" / ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return label, ""


def _ratio_bar(value: float, width: int = 12) -> str:
    """Render a fixed-width progress bar for values in [0, 1]."""
    width = max(1, int(width))
    if np.isnan(value):
        return "░" * width
    clamped = float(np.clip(value, 0.0, 1.0))
    filled = int(round(clamped * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _rank_hump_lane(expected_rank: float, n_items: int, width: int = 14) -> str:
    """Render rank position as a horizontal lane with an adaptive hump.

    Left corresponds to rank #1 (best). The hump is sharper when
    ``expected_rank`` is near an integer and softer when it is near the
    midpoint between integers.
    """
    width = max(3, int(width))
    if n_items <= 1 or np.isnan(expected_rank):
        center = width // 2
        lane = ["─"] * width
        lane[center] = "█"
        return "".join(lane)

    clamped_rank = float(np.clip(expected_rank, 1.0, float(n_items)))
    pos = (clamped_rank - 1.0) / (float(n_items) - 1.0)
    center = int(round(pos * (width - 1)))

    frac_to_int = abs(clamped_rank - round(clamped_rank))
    sharpness = 1.0 - min(frac_to_int, 0.5) / 0.5

    if sharpness >= 0.67:
        profile = {0: "█", 1: "▆", 2: "▃"}
    elif sharpness >= 0.33:
        profile = {0: "▇", 1: "▅", 2: "▂"}
    else:
        profile = {0: "▆", 1: "▄", 2: "▁"}

    lane = ["─"] * width
    for offset, char in profile.items():
        left = center - offset
        right = center + offset
        if 0 <= left < width:
            lane[left] = char
        if 0 <= right < width:
            lane[right] = char

    return "".join(lane)


# ---------------------------------------------------------------------------
# p-value formatting
# ---------------------------------------------------------------------------

def _p_value_stars(p_value: Optional[float]) -> str:
    """Return significance stars for p-value thresholds (*, **, ***)."""
    if p_value is None:
        return ""
    if p_value < 0.0001:
        return "***"
    if p_value < 0.001:
        return "**"
    if p_value < 0.01:
        return "*"
    return ""


def _format_p_value(p_value: Optional[float]) -> str:
    """Format p-value with significance stars; return N/A for missing values.

    Bootstrap-based p-values can come back as exactly 0.0 when the observed
    effect never crosses zero in any resample (its true value is merely
    "smaller than this bootstrap can resolve", not literally zero) — shown
    as "<0.0001" rather than the misleading bare "0".
    """
    if p_value is None:
        return "N/A"
    stars = _p_value_stars(p_value)
    if p_value <= 0.0:
        return f"<0.0001{stars}"
    return f"{p_value:.4g}{stars}"


# ---------------------------------------------------------------------------
# Critical-difference group detection
# ---------------------------------------------------------------------------

def _pairwise_rank_band_p(
    pairwise: PairwiseMatrix,
    label_a: str,
    label_b: str,
    *,
    p_source: Literal["bootstrap", "wilcoxon"],
) -> Optional[float]:
    """Return the pairwise p-value used to decide rank-band indistinguishability."""
    try:
        result = pairwise.get(label_a, label_b)
    except KeyError:
        return None

    if p_source == "bootstrap":
        return float(result.p_value)
    if p_source == "wilcoxon":
        return None if result.wilcoxon_p is None else float(result.wilcoxon_p)

    p_values = [float(result.p_value)]
    if result.wilcoxon_p is not None:
        p_values.append(float(result.wilcoxon_p))
    return min(p_values) if p_values else None


def _critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> list[list[str]]:
    """Return contiguous, maximal non-significant rank bands.

    When simultaneous CIs have been computed, significance is determined by
    whether the pairwise CI excludes zero (consistent with the displayed CIs).
    Otherwise, correction-adjusted p-values compared to *alpha* are used.
    """
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return []

    n_labels = len(labels_sorted)
    use_ci = pairwise.simultaneous_ci_method is not None

    def _all_pairs_nonsignificant(group_labels: list[str]) -> bool:
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                if use_ci:
                    try:
                        result = pairwise.get(group_labels[i], group_labels[j])
                    except KeyError:
                        return False
                    if result.ci_low > 0 or result.ci_high < 0:
                        return False
                else:
                    p_value = _pairwise_rank_band_p(
                        pairwise, group_labels[i], group_labels[j], p_source=p_source,
                    )
                    if p_value is None or p_value < alpha:
                        return False
        return True

    candidate_groups: list[list[str]] = []
    for start_idx in range(n_labels - 1):
        best_group: Optional[list[str]] = None
        for end_idx in range(start_idx + 1, n_labels):
            group = labels_sorted[start_idx : end_idx + 1]
            if _all_pairs_nonsignificant(group):
                best_group = group
            else:
                break
        if best_group is not None:
            candidate_groups.append(best_group)

    def _is_contiguous_subsequence(smaller: list[str], larger: list[str]) -> bool:
        if len(smaller) >= len(larger):
            return False
        max_start = len(larger) - len(smaller)
        for start in range(max_start + 1):
            if larger[start : start + len(smaller)] == smaller:
                return True
        return False

    maximal_groups: list[list[str]] = []
    for group in candidate_groups:
        if any(
            _is_contiguous_subsequence(group, other)
            for other in candidate_groups
            if other is not group
        ):
            continue
        maximal_groups.append(group)

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in maximal_groups:
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(group)
    return deduped


def _single_clear_winner_label(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> Optional[str]:
    """Return the unique label that significantly beats every other label."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return None

    use_ci = pairwise.simultaneous_ci_method is not None
    winners: list[str] = []
    for candidate in labels_sorted:
        candidate_beats_all = True
        for other in labels_sorted:
            if other == candidate:
                continue

            try:
                result = pairwise.get(candidate, other)
            except KeyError:
                candidate_beats_all = False
                break

            if use_ci:
                sig = result.ci_low > 0 or result.ci_high < 0
                beats = float(result.point_diff) > 0
            else:
                p_value = _pairwise_rank_band_p(
                    pairwise, candidate, other, p_source=p_source,
                )
                sig = p_value is not None and p_value < alpha
                beats = float(result.point_diff) > 0

            if not sig or not beats:
                candidate_beats_all = False
                break

        if candidate_beats_all:
            winners.append(candidate)
            if len(winners) > 1:
                return None

    return winners[0] if len(winners) == 1 else None


def _print_critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> None:
    """Print a short CD-style summary of statistically indistinguishable groups."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return

    rank_pos = {label: idx + 1 for idx, label in enumerate(labels_sorted)}

    if pairwise.simultaneous_ci_method is not None:
        source_label = f"{(1-alpha)*100:.0f}% CI"
    else:
        source_label = {
            "bootstrap": "p (boot)",
            "wilcoxon": "p (wsr)",
        }[p_source]

    groups = _critical_difference_groups(
        pairwise,
        labels_sorted=labels_sorted,
        alpha=alpha,
        p_source=p_source,
    )
    if not groups:
        print(
            f"  Statistically indistinguishable rank bands "
            f"({source_label}): none"
        )
        return

    print(
        f"  Statistically indistinguishable rank bands "
        f"{_DIM}(similar to critical difference diagrams) computed from {source_label}:{_RESET}"
    )
    for group in groups:
        start_rank = rank_pos[group[0]]
        end_rank = rank_pos[group[-1]]
        rank_span = f"#{start_rank}" if start_rank == end_rank else f"#{start_rank}–#{end_rank}"
        print(f"    {rank_span}: [{' ─ '.join(group)}]")

    clear_winner = _single_clear_winner_label(
        pairwise,
        labels_sorted=labels_sorted,
        alpha=alpha,
        p_source=p_source,
    )
    if clear_winner is not None:
        print()
        print(
            f"  {_BRIGHT_GREEN}-> Evidence suggests a clear best option:{_RESET} "
            f"'{_BOLD}{_BRIGHT_GREEN}{clear_winner}{_RESET}'"
        )


# ---------------------------------------------------------------------------
# Executive summary helpers
# ---------------------------------------------------------------------------

def _assign_significance_groups(
    pairwise: PairwiseMatrix,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> dict[str, str]:
    """Assign numeric group IDs (#1, #2, #3…) to templates via CD-group analysis.

    Templates in the same maximal non-significant rank band share an ID, and
    IDs are non-decreasing down the rank-sorted list (group #1 always holds
    the rank-1 template). For #2 onward, when maximal bands overlap -- e.g.
    A~B and B~C are each individually non-significant but A~C is
    significant, the transitivity caveat inherent to critical-difference
    diagrams (Demsar 2006) -- the whole chain is merged into one group,
    since each entity can only carry a single ID in this table (unlike a CD
    diagram, which can draw overlapping bands as separate lines).

    #1 is deliberately NOT extended this way: it's the one tier
    ``_exec_verdict`` turns into an explicit "tied with X as best" claim, so
    membership there must mean "provably indistinguishable from the actual
    top performer" (the single maximal band containing rank 0), not merely
    "reachable from it via a chain of individually-nonsignificant
    neighbors". Chaining #1 the same way #2+ do would let a template many
    links down the chain -- one the rank-1 template IS significantly better
    than, directly -- inherit a "tied as best" verdict it doesn't deserve.
    Anything past #1's direct band still gets its own (possibly
    chain-merged) tier via the normal algorithm, so it correctly reads
    "Significant drop-off" instead.
    """
    if alpha is None:
        alpha = get_alpha_ci()
    groups = _critical_difference_groups(
        pairwise, labels_sorted=labels_sorted, alpha=alpha, p_source=p_source,
    )
    rank_of = {label: i for i, label in enumerate(labels_sorted)}

    # For each label, the rightmost rank index reached by any maximal CD band
    # it belongs to (its own rank if it belongs to none) -- used for #2+.
    reach = {label: rank_of[label] for label in labels_sorted}
    for group in groups:
        end_idx = max(rank_of[l] for l in group)
        for label in group:
            reach[label] = max(reach[label], end_idx)

    # #1's own (non-transitive) extent: just the single maximal band
    # containing labels_sorted[0], if any.
    top_reach = rank_of[labels_sorted[0]]
    for group in groups:
        if labels_sorted[0] in group:
            top_reach = max(top_reach, max(rank_of[l] for l in group))

    label_to_group: dict[str, str] = {}
    group_idx = 0
    current_end_idx = -1
    for idx, label in enumerate(labels_sorted):
        if idx > current_end_idx:
            group_idx += 1
            current_end_idx = top_reach if group_idx == 1 else reach[label]
        elif group_idx > 1:
            current_end_idx = max(current_end_idx, reach[label])
        # group_idx == 1 and idx <= current_end_idx: stay pinned at
        # top_reach regardless of this label's own (possibly further-
        # reaching) chain -- see the direct-vs-transitive note above.
        label_to_group[label] = f"#{group_idx}"

    return label_to_group


def _exec_verdict(
    label: str,
    label_to_group: dict[str, str],
    labels_sorted: list[str],
) -> str:
    """Human-readable verdict for a template in the executive summary."""
    my_group = label_to_group.get(label, "?")
    if my_group != "#1":
        return "Significant drop-off"
    group_1 = [l for l in labels_sorted if label_to_group.get(l) == "#1"]
    others = [l for l in group_1 if l != label]
    if not others:
        return "Likely best"
    if len(others) == 1:
        return f"Tied with {_truncate_label(others[0], 20)} as best"
    return f"Tied with {len(others)} others as best"


# Sort priority for Pareto status groups: frontier first (the calibrated
# "safe" verdict), then ambiguous (can't rule dominance in or out), then
# dominated last.
_PARETO_STATUS_ORDER = {"frontier": 0, "ambiguous": 1, "dominated": 2}


def _pareto_sorted_labels(pareto: dict) -> list[str]:
    """Entity order for the Pareto Front table: status group, then primary
    metric mean descending within each group."""
    result = pareto["result"]
    statuses = pareto["statuses"]
    point_primary = dict(zip(result.labels, result.point_primary))
    return sorted(
        result.labels,
        key=lambda lbl: (
            _PARETO_STATUS_ORDER.get(statuses[lbl].status, 3),
            -point_primary[lbl],
        ),
    )


# Glyph + color per Pareto status, shared by the table's merged Status
# column, the Executive Summary's Pareto column, and the scatterplot below --
# one visual vocabulary across every place a status shows up.
_PARETO_STATUS_GLYPH = {"frontier": "★", "ambiguous": "◌", "dominated": "×"}


def _pareto_status_glyph(status: str) -> str:
    return _PARETO_STATUS_GLYPH.get(status, "?")


def _pareto_status_color(status: str) -> str:
    if not _ANSI:
        return ""
    if status == "frontier":
        return _BRIGHT_GREEN
    if status == "dominated":
        return _DIM
    return _YELLOW  # ambiguous


def _join_names_capped(names: list[str], *, max_names: int = 2) -> str:
    """Join entity names for a status phrase, capping how many get spelled
    out -- an entity dominated by half a dozen others would otherwise blow
    up the Status column (and the whole table) with a name list as wide as
    the table itself. Mirrors the existing "Tied with N others as best"
    capping already used in the Executive Summary's Verdict column."""
    if len(names) <= max_names:
        return ", ".join(names)
    shown = ", ".join(names[:max_names])
    return f"{shown} and {len(names) - max_names} more"


def _pareto_status_phrase(status: "ParetoStatus", *, verbose: bool = True) -> str:
    """One-cell phrase combining a ParetoStatus's glyph with plain-language
    wording and its detail (dominated_by / ambiguous_vs), e.g.
    "× Worse than gpt-4o on both" -- used by both the Pareto Front table
    (verbose=True, includes the "(not confirmed)" qualifier) and the
    Executive Summary's narrower Pareto column (verbose=False, drops it)."""
    glyph = _pareto_status_glyph(status.status)
    if status.status == "dominated":
        text = f"Worse than {_join_names_capped(status.dominated_by)} on both"
    elif status.status == "ambiguous":
        suffix = " (not confirmed)" if verbose else ""
        text = f"Unclear vs {_join_names_capped(status.ambiguous_vs)}{suffix}"
    else:
        text = "Best trade-off"
    return f"{glyph} {text}"


_PARETO_MARKERS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _pareto_marker(i: int) -> str:
    """The scatterplot's per-entity marker character for position ``i`` in
    ``_pareto_sorted_labels`` order (1, 2, ..., 9, A, B, ..., then '#' past
    36 entities). Shared with :func:`_print_pareto_section`'s table so its
    new leading "#" column reproduces the exact same character the plot
    used for that row -- both iterate ``sorted_labels`` in the same order.
    """
    return _PARETO_MARKERS[i] if i < len(_PARETO_MARKERS) else "#"


def _print_pareto_scatter(
    pareto: dict,
    *,
    metric_label: str,
    width: int = 44,
    height: int = 9,
) -> None:
    """Print a compact ASCII scatterplot of primary vs. secondary metric,
    one point per entity, marked with its Pareto status glyph (see
    ``_pareto_status_glyph``) so the trade-off shape between metrics is
    visible at a glance, before the reader has to parse the numeric table
    below. Points are numbered (1, 2, 3, ...) rather than labeled by name
    to sidestep collisions when entities sit close together on the plot;
    a legend below maps each number back to its entity name.
    """
    result = pareto["result"]
    statuses = pareto["statuses"]
    secondary_col = pareto["secondary_metric"]
    direction = pareto["direction"]
    primary_rob = pareto.get("primary_robustness")
    secondary_rob = pareto.get("secondary_robustness")
    if primary_rob is None or secondary_rob is None or len(result.labels) < 2:
        return

    sorted_labels = _pareto_sorted_labels(pareto)
    prim_idx = {lbl: i for i, lbl in enumerate(primary_rob.labels)}
    sec_idx = {lbl: i for i, lbl in enumerate(secondary_rob.labels)}
    xs = [float(secondary_rob.mean[sec_idx[lbl]]) for lbl in sorted_labels]
    ys = [float(primary_rob.mean[prim_idx[lbl]]) for lbl in sorted_labels]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, abs(x_max) * 1e-6, 1e-9)
    y_pad = max((y_max - y_min) * 0.08, abs(y_max) * 1e-6, 1e-9)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    _marker = _pareto_marker

    grid = [[" "] * width for _ in range(height)]
    occupied: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        col = int(round((x - x_lo) / (x_hi - x_lo) * (width - 1)))
        row = int(round((y_hi - y) / (y_hi - y_lo) * (height - 1)))
        col = min(max(col, 0), width - 1)
        row = min(max(row, 0), height - 1)
        occupied.setdefault((row, col), []).append(i)

    for (row, col), idxs in occupied.items():
        # The first entity claiming a cell gets the glyph; cells shared by
        # more than one entity are flagged in a note below rather than
        # silently overwritten.
        lead = idxs[0]
        status = statuses[sorted_labels[lead]].status
        color = _pareto_status_color(status)
        reset = _RESET if color else ""
        grid[row][col] = f"{color}{_marker(lead)}{reset}"

    y_label_w = 9
    print(f"  {metric_label:>{y_label_w}}")
    for row in range(height):
        if row == 0:
            y_tick = f"{y_hi:>{y_label_w}.3g}"
        elif row == height - 1:
            y_tick = f"{y_lo:>{y_label_w}.3g}"
        else:
            y_tick = " " * y_label_w
        print(f"  {y_tick} │{''.join(grid[row])}│")
    border = "─" * width
    print(f"  {'':>{y_label_w}} └{border}┘")
    dir_arrow = "→ better" if direction == "max" else "← better"
    x_axis_label = f"{secondary_col} ({dir_arrow})"
    x_min_str = f"{x_min:.3g}"
    x_max_str = f"{x_max:.3g}"
    print(
        f"  {'':>{y_label_w}}  {x_min_str}"
        f"{x_axis_label:^{max(width - 12, 4)}}"
        f"{x_max_str}"
    )

    # The plot always stretches to fill its width/height regardless of how
    # small the real spread is -- flag it explicitly when an axis's true
    # range is tiny relative to its scale, so a reader doesn't mistake
    # bootstrap noise blown up to fill the plot for a real difference.
    def _near_degenerate(lo: float, hi: float) -> bool:
        return (hi - lo) < 0.02 * max(abs(hi), abs(lo), 1e-9)

    flat_axes = []
    if _near_degenerate(x_min, x_max):
        flat_axes.append(f"{secondary_col} ({x_min_str}–{x_max_str})")
    if _near_degenerate(y_min, y_max):
        flat_axes.append(f"{metric_label} ({y_min:.3g}–{y_max:.3g})")
    if flat_axes:
        print(
            f"  Note: {' and '.join(flat_axes)} barely varies across entities "
            "-- the spread above is mostly noise, not a real difference."
        )
    print()

    legend_cells = []
    for i, lbl in enumerate(sorted_labels):
        status = statuses[lbl].status
        color = _pareto_status_color(status)
        reset = _RESET if color else ""
        legend_cells.append(f"{color}{_marker(i)}{reset}={_truncate_label(lbl, 20)}")
    # Wrap the legend at a reasonable width instead of one long line.
    line = "  "
    for cell in legend_cells:
        plain_len = len(re.sub(r"\033\[[0-9;]*m", "", cell))
        if len(re.sub(r"\033\[[0-9;]*m", "", line)) + plain_len + 3 > 78 and line.strip():
            print(line.rstrip())
            line = "  "
        line += cell + "   "
    if line.strip():
        print(line.rstrip())

    collisions = [idxs for idxs in occupied.values() if len(idxs) > 1]
    for idxs in collisions:
        names = ", ".join(_marker(i) for i in idxs)
        print(f"  ({names} sit at nearly the same spot on this plot)")
    print()


def _print_pareto_section(
    pareto: dict,
    *,
    metric: Optional[str],
    show_rank_probabilities: bool,
) -> None:
    """Print the Pareto Front section (frontier/dominated/ambiguous per
    entity against a secondary metric) -- see ``ComparisonResult.pareto_status``.

    Printed immediately before the executive summary (see
    ``_print_bundle_summary``) so the secondary-metric-corrected, holistic
    verdict sits right next to the primary-metric-only leaderboard, rather
    than trailing after everything else where it's easy to miss.

    Shows each metric's own calibrated mean + CI side by side (not just the
    status label) -- ``pareto["primary_robustness"]``/``["secondary_robustness"]``
    are the same kind of :class:`~evalstats.core.variance.RobustnessResult`
    the rest of evalstats already shows for a single metric, so the numbers
    here carry the same guarantees. Sorted frontier -> ambiguous -> dominated
    (see :func:`_pareto_sorted_labels`), not raw entity order.
    """
    secondary_col = pareto["secondary_metric"]
    direction = pareto["direction"]
    statuses = pareto["statuses"]
    result = pareto["result"]
    primary_rob = pareto.get("primary_robustness")
    secondary_rob = pareto.get("secondary_robustness")

    dir_label = "lower is better" if direction == "min" else "higher is better"
    metric_label = metric or "primary metric"
    _print_subsection(
        f"--- {metric_label} vs. {secondary_col} Trade-off "
        f"(Pareto Front, {dir_label}) ---"
    )
    print(
        f"  A model has the 'best trade-off' when no other option beats it "
        f"on both {metric_label} and {secondary_col} at once."
    )
    print()
    _print_pareto_scatter(pareto, metric_label=metric_label)

    sorted_labels = _pareto_sorted_labels(pareto)
    # Capped the same way every other table's entity/model column is
    # (see e.g. _print_executive_summary's tpl_w) -- long entity names would
    # otherwise stretch this table arbitrarily wide.
    label_w = min(28, max(len("Entity"), max((len(lbl) for lbl in result.labels), default=6)))
    phrases = {lbl: _pareto_status_phrase(statuses[lbl], verbose=True) for lbl in result.labels}
    status_w = max([len("Status")] + [len(p) for p in phrases.values()])
    mean_w = 7
    ci_w = 17

    have_stats = primary_rob is not None and secondary_rob is not None
    # The scatterplot above numbers entities 1, 2, 3, ... (see
    # _print_pareto_scatter/_pareto_marker) only when it actually rendered
    # (have_stats and >= 2 entities); this leading "#" column reproduces
    # that same numbering here so a reader can jump from a point on the
    # plot straight to its row, instead of re-deriving the mapping from the
    # legend line by line.
    show_markers = have_stats and len(result.labels) >= 2
    marker_w = 1
    marker_prefix = f"{'#':<{marker_w}}  " if show_markers else ""
    if have_stats:
        prim_idx = {lbl: i for i, lbl in enumerate(primary_rob.labels)}
        sec_idx = {lbl: i for i, lbl in enumerate(secondary_rob.labels)}
        # Left-aligned (not centered) so each metric name reads as a label
        # spanning its own "Mean 95% CI" pair starting directly above the
        # Mean column, rather than visually drifting toward the wider CI
        # sub-column when centered over the combined width.
        metric_row = (
            f"  {'':<{len(marker_prefix)}}{'':<{label_w}}  {'':<{status_w}}  "
            f"{_truncate_label(metric_label, mean_w + ci_w + 1):<{mean_w + ci_w + 1}s}  "
            f"{_truncate_label(secondary_col, mean_w + ci_w + 1):<{mean_w + ci_w + 1}s}"
        )
        print(metric_row)
        header = (
            f"  {marker_prefix}{'Entity':<{label_w}}  {'Status':<{status_w}}  "
            f"{'Mean':>{mean_w}s} {'95% CI':<{ci_w}s}  "
            f"{'Mean':>{mean_w}s} {'95% CI':<{ci_w}s}"
        )
    else:
        header = f"  {marker_prefix}{'Entity':<{label_w}}  {'Status':<{status_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, label in enumerate(sorted_labels):
        row_marker = f"{_pareto_marker(i):<{marker_w}}  " if show_markers else ""
        status_disp = f"{phrases[label]:<{status_w}}"
        color = _pareto_status_color(statuses[label].status)
        status_str = f"{color}{status_disp}{_RESET}" if color else status_disp
        if have_stats:
            pi, si = prim_idx[label], sec_idx[label]
            p_mean = float(primary_rob.mean[pi])
            p_ci = (
                f"[{primary_rob.ci_low[pi]:.3g}, {primary_rob.ci_high[pi]:.3g}]"
                if primary_rob.ci_low is not None else "--"
            )
            s_mean = float(secondary_rob.mean[si])
            s_ci = (
                f"[{secondary_rob.ci_low[si]:.3g}, {secondary_rob.ci_high[si]:.3g}]"
                if secondary_rob.ci_low is not None else "--"
            )
            print(
                f"  {row_marker}{_truncate_label(label, label_w):<{label_w}}  {status_str}  "
                f"{p_mean:>{mean_w}.3g} {p_ci:<{ci_w}s}  "
                f"{s_mean:>{mean_w}.3g} {s_ci:<{ci_w}s}"
            )
        else:
            print(f"  {row_marker}{_truncate_label(label, label_w):<{label_w}}  {status_str}")
    print("  " + "-" * (len(header) - 2))
    if show_rank_probabilities:
        print()
        print("  How confident are we in each trade-off call?")
        print("  (P(Pareto-optimal): share of bootstrap resamples this entity wasn't beaten on both axes in)")
        bar_w = 20
        for label in sorted_labels:
            p = float(result.p_frontier[result.labels.index(label)])
            color = _p_best_color(p)
            reset = _RESET if color else ""
            print(f"    {_truncate_label(label, label_w):<{label_w}}  {p:>6.1%} {color}{_ratio_bar(p, width=bar_w)}{reset}")
    print()


def _print_executive_summary(
    bundle: AnalysisBundle,
    *,
    item_singular: str = "template",
    pareto: Optional[dict] = None,
    metric: Optional[str] = None,
) -> None:
    """Print a concise executive leaderboard after the stats-heavy blocks.

    Shows each template's significance group, mean score, bootstrap CI,
    optional stability (when seed data is present), optional Trade-off
    status (when ``compare(..., secondary_metric=...)`` was passed -- see
    ``_print_pareto_section``), and a plain-language verdict so the user can
    assess results at a glance without scrolling up. The Trade-off column
    surfaces the secondary-metric verdict right next to the primary-metric
    one, e.g. an entity that's "Likely best" on the primary metric but
    "Worse than X on both" once the secondary metric is considered -- and
    when Trade-off is shown, the last column is relabeled "On {metric}"
    (instead of the bare "Verdict") so it reads as scoped to the primary
    metric alone, rather than as the final word once a second axis exists.
    """
    labels = list(bundle.labels)
    n = len(labels)
    if n < 2:
        return

    # Sort by mean score descending (best first).
    means = bundle.robustness.mean
    sort_idx = list(np.argsort(-means))
    labels_sorted = [labels[i] for i in sort_idx]

    # Significance group letters via CD groups.
    label_to_group = _assign_significance_groups(bundle.pairwise, labels_sorted)
    verdict_by_label = {
        label: _exec_verdict(label, label_to_group, labels_sorted) for label in labels_sorted
    }

    # Seed variance for stability column (optional).
    sv = bundle.seed_variance
    has_stability = sv is not None
    has_pareto = pareto is not None
    pareto_statuses = pareto["statuses"] if has_pareto else None
    pareto_phrases: dict[str, str] = (
        {lbl: _pareto_status_phrase(st, verbose=False) for lbl, st in pareto_statuses.items()}
        if has_pareto else {}
    )

    item_title = item_singular.capitalize()
    _print_subsection(f"--- Executive Summary ({item_title} leaderboard) ---")

    # Column widths.
    tpl_w = min(28, max(16, max(len(l) for l in labels)))
    grp_w = 4
    mean_w = 6
    ci_w = 15  # e.g. "[0.950, 0.990]" = 14 chars + 1 padding
    stab_w = 16
    noise_w = _NOISE_COL_W
    # Noise strip uses the same global scale as the "Per-input Variance
    # Across Runs" table above it, so bar heights are comparable across rows
    # and across the two tables.
    global_cell_max = float(sv.per_cell_seed_std.max()) if has_stability else 0.0
    # "Trade-off vs {secondary_metric}" names the second axis explicitly (truncated
    # -- an arbitrary column name shouldn't be able to blow out this table's
    # width), pairing with "On {metric}" below so the two columns' headers
    # alone state both axes without needing the Pareto section above.
    tradeoff_secondary = pareto.get("secondary_metric") if has_pareto else None
    tradeoff_header = (
        f"Trade-off vs {_truncate_label(tradeoff_secondary, 16)}"
        if tradeoff_secondary else "Trade-off"
    )
    pareto_w = max([len(tradeoff_header)] + [len(p) for p in pareto_phrases.values()]) if has_pareto else 0

    # CI column header: Wilson CI when no bootstrap was used (binary data path).
    ci_col_header = "Wilson-flat CI" if _uses_wilson_ci(bundle) else "CI"

    # Header row (no ANSI codes so widths match exactly).
    header_parts = [
        f"  {item_title:<{tpl_w}s}",
        f"  {'Grp':^{grp_w}s}",
        f"  {'Mean':>{mean_w}s}",
        f"  {ci_col_header:<{ci_w}s}",
    ]
    if has_stability:
        header_parts.append(f"  {_NOISE_STRIP_HEADER:<{noise_w}s}")
        header_parts.append(f"  {'Stability':<{stab_w}s}")
    verdict_header = f"On {metric or 'primary metric'}" if has_pareto else "Verdict"
    # Only needs padding when it's no longer the last (unpadded) column,
    # i.e. once Trade-off follows it -- computed from the actual verdict
    # strings, which vary a lot ("Likely best" vs. "Tied with X as best").
    verdict_w = (
        max([len(verdict_header)] + [len(v) for v in verdict_by_label.values()])
        if has_pareto else 0
    )
    if has_pareto:
        # "On {metric}" first (echoes the Mean/CI columns just shown), then
        # "Trade-off vs {secondary_metric}" -- reads as "here's the primary-metric
        # call, and here's how that changes once the other axis counts too."
        header_parts.append(f"  {verdict_header:<{verdict_w}s}")
        header_parts.append(f"  {tradeoff_header}")
    else:
        header_parts.append(f"  {verdict_header}")
    header = "".join(header_parts)
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    for label in labels_sorted:
        orig_idx = labels.index(label)
        mean_val = float(means[orig_idx])

        ci_lo = float(bundle.robustness.ci_low[orig_idx])
        ci_hi = float(bundle.robustness.ci_high[orig_idx])
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

        group = label_to_group.get(label, "?")
        verdict = verdict_by_label[label]

        # Pre-format fixed-width parts, then optionally wrap with ANSI.
        plain_label = f"{_truncate_label(label, tpl_w):<{tpl_w}s}"
        plain_grp = f"{group:^{grp_w}s}"
        plain_verdict = f"{verdict:<{verdict_w}s}" if has_pareto else verdict
        if group == "#1" and _ANSI:
            label_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_label}{_RESET}"
            grp_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_grp}{_RESET}"
            verdict_str = f"{_BRIGHT_GREEN}{plain_verdict}{_RESET}"
        else:
            label_str = plain_label
            grp_str = plain_grp
            verdict_str = plain_verdict

        row = (
            f"  {label_str}"
            f"  {grp_str}"
            f"  {mean_val:>{mean_w}.3f}"
            f"  {ci_str:<{ci_w}s}"
        )

        if has_stability:
            sv_labels = list(sv.labels)
            if label in sv_labels:
                sv_idx = sv_labels.index(label)
                instability_val = float(sv.instability[sv_idx])
                noise_plain = f"{_seed_noise_strip(sv.per_cell_seed_std[sv_idx], global_cell_max, max_width=_NOISE_STRIP_CHARS):<{noise_w}s}"
                stab_plain = f"{_stability_emoji_label(instability_val):<{stab_w}s}"
                row_color = _instability_color(instability_val)
            else:
                noise_plain = f"{'—':<{noise_w}s}"
                stab_plain = f"{'—':<{stab_w}s}"
                row_color = ""
            row += f"  {row_color}{noise_plain}{_RESET}" if row_color else f"  {noise_plain}"
            row += f"  {row_color}{stab_plain}{_RESET}" if row_color else f"  {stab_plain}"

        # "On {metric}" (verdict) first, then "Trade-off vs {secondary_metric}" --
        # matches the header order above.
        row += f"  {verdict_str}"

        if has_pareto:
            pareto_plain = pareto_phrases.get(label, "—")
            pareto_color = (
                _pareto_status_color(pareto_statuses[label].status)
                if label in pareto_statuses else ""
            )
            pareto_str = f"{pareto_color}{pareto_plain}{_RESET}" if pareto_color else pareto_plain
            row += f"  {pareto_str}"

        print(row)

    print(sep)
    print()


def _print_pareto_callout(pareto: dict, *, metric: Optional[str]) -> None:
    """One-line bridge from the Executive Summary's primary-metric-only
    leader to the Pareto Front's holistic view, e.g. "'gpt-4o' leads on
    accuracy, but 'gpt-4o-mini' is a competitive trade-off on latency_s" --
    mirrors the existing "-> Evidence suggests a clear best option" callout
    used after Pairwise Comparisons, giving a skimming reader the "so what"
    without having to cross-reference the table above.
    """
    result = pareto["result"]
    statuses = pareto["statuses"]
    secondary_col = pareto["secondary_metric"]
    if len(result.labels) < 2:
        return

    leader_idx = int(np.argmax(result.point_primary))
    leader = result.labels[leader_idx]
    metric_label = metric or "the primary metric"

    other_frontier = [
        lbl for lbl in _pareto_sorted_labels(pareto)
        if lbl != leader and statuses[lbl].status == "frontier"
    ]
    if other_frontier:
        names = ", ".join(f"'{lbl}'" for lbl in other_frontier)
        is_are, article_or_plural = (
            ("is", "a competitive trade-off") if len(other_frontier) == 1
            else ("are", "competitive trade-offs")
        )
        print(
            f"  -> '{leader}' leads on {metric_label}, but {names} {is_are} "
            f"{article_or_plural} on {secondary_col} — see Pareto Front above."
        )
    else:
        print(
            f"  -> '{leader}' is also the best choice on {secondary_col} "
            "— no real trade-off here."
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Between-subjects (design="unpaired") summary
# ─────────────────────────────────────────────────────────────────────────────

_VERBOSE_SUMMARY = False


def _print_pairwise_efficiency_note(rows: list[dict], result: "GroupComparisonResult") -> None:
    """Explain the rho^2/N_eff columns in the reader's own numbers.

    Two lines by default, aimed at someone who does not read statistics. The
    qualifications a statistician would want -- that N_eff is an upper bound at
    the variance-minimizing lambda, and that rank-based rho^2 is tied to the
    effect size in this dataset -- are real but would double the length of a
    note most readers need only once, so they print under
    ``summary(verbose=True)``.
    """
    effs = [r.get("n_eff") for r in rows if r.get("n_eff") is not None]
    if not effs:
        return
    lo, hi = min(effs), max(effs)
    span = f"{lo:.0f}" if abs(hi - lo) < 0.5 else f"{lo:.0f} to {hi:.0f}"
    n_lab = result.n_lab_per_condition
    print(f"{_DIM}  rho^2 and N_eff describe the p-value only, not the interval "
          f"beside it.{_RESET}")
    tail = f", from the {n_lab:.0f} you labeled" if n_lab else ""
    print(f"{_DIM}  N_eff (effective sample size) = how many hand-labeled items per "
          f"condition{_RESET}")
    print(f"{_DIM}  would have given the test this much power. Here {span}{tail}.{_RESET}")
    if _VERBOSE_SUMMARY:
        print(f"{_DIM}  N_eff is the best case, at the variance-minimizing lambda. The "
              f"shipped test{_RESET}")
        print(f"{_DIM}  can realize less. For rank-based tests (Mann-Whitney, "
              f"Kruskal-Wallis) rho^2{_RESET}")
        print(f"{_DIM}  also falls as the true effect grows, so these numbers describe "
              f"this dataset{_RESET}")
        print(f"{_DIM}  and should not be reused to plan a future study.{_RESET}")


def print_group_comparison_summary(result: "GroupComparisonResult", *, style: str = "gradient", verbose: bool = False) -> None:
    """Print the console summary for a between-subjects
    ``compare(design="unpaired")`` result.

    Deliberately narrower than the paired path's summary (no forest-plot
    brackets) but otherwise mirrors it section for section: descriptive
    statistics, per-group means with gradient CIs, a pairwise comparison
    table (with critical-difference rank bands), the omnibus test at k>=3
    (when ``omnibus=True``, same opt-in default as the paired path's own
    Friedman test), a Pareto-front section when ``secondary_metric=`` was
    passed, and an executive summary leaderboard.

    Reuses the paired path's rendering functions directly rather than
    reimplementing them -- the PPI banner (``_print_ppi_banner``), the
    per-entity means table (``_print_mean_advantage``), the pairwise
    comparison table (``_print_pairwise_section``, whose
    ``_prepare_unpaired_pairwise_rows`` also drives the critical-difference
    bands), the Pareto-front section (``_print_pareto_section``, when
    present) and callout (``_print_pareto_callout``), and the executive
    summary (``_print_executive_summary``) are the *same* functions the
    paired path calls, not reimplementations, so a change to any of them
    renders identically for both paths. What's genuinely unpaired-specific
    (this engine's fixed Bonferroni-CI/Holm-p FWER scheme, vs. the paired
    path's six CI/p-value method families plus Friedman/Nemenyi; its own
    per-group joint bootstrap for the Pareto front, since there's no shared
    item pool across disjoint groups) is resolved into the same shapes
    those shared renderers already consume, via small duck-typed adapters
    in ``evalstats.core.unpaired`` (``_GroupStatsAsRobustness``,
    ``_GroupDiffResultsAsPairwiseMatrix``, ``_GroupComparisonResultAsBundle``).
    The Behavioral Agreement (McNemar-style) subsection is paired-only and
    never called here -- ``agreement_mcc``/``binary_confusion`` need the
    same item scored by both entities, which has no between-subjects
    equivalent.
    """
    global _VERBOSE_SUMMARY
    _VERBOSE_SUMMARY = bool(verbose)
    from evalstats.core.unpaired import _GroupComparisonResultAsBundle

    # Plain two-line header, mirroring the paired path's own "Shape: ...(...)"
    # + "{Entities}: N | {Items}: M | seed: X" format (see _print_bundle_summary)
    # instead of this path's original bold banner + separate "Item column"
    # line. Score type / family aren't restated here, same as the paired
    # header never restates its CI/test method -- both surface later, in the
    # pairwise section's own method labels (e.g. "p (PPI-MWU)").
    print(
        f"Shape: BetweenGroups(factor={result.factor_col!r}, "
        f"groups={len(result.groups)}, metric={result.metric_col!r})"
    )
    group_ns = sorted({g.n for g in result.groups})
    n_note = f"{group_ns[0]}/group" if len(group_ns) == 1 else f"{group_ns[0]}-{group_ns[-1]}/group"
    n_total = sum(g.n for g in result.groups)
    print(
        f"Groups: {len(result.groups)} | N: {n_total} ({n_note})"
        f"{_seed_note(getattr(result, 'rng_seed', None))}"
    )
    if result.item_col_synthetic:
        print("(no item column found -- each row treated as its own item)")
    print()

    if result.ppi_applied:
        _print_ppi_banner()

    # ── Descriptive statistics ──────────────────────────────────────────────
    # Same table the paired path prints (RobustnessResult.summary_table()),
    # built from each group's raw per-group scores -- mean/std come from
    # GroupStat (PPI-corrected when alignment= was passed, exactly like the
    # paired path's own mean/CI), the rest (median/cv/iqr/cvar_10/
    # percentiles) from the raw, uncorrected scores, which is why the whole
    # table is skipped once PPI is applied -- the paired path drops it for
    # the same reason. Only mean/ci_low/ci_high/multi_ci get PPI-overridden
    # (see api.py's PPI-correction block).
    if all(g.descriptive is not None for g in result.groups) and not result.ppi_applied:
        from evalstats.core.variance import RobustnessResult
        _desc = RobustnessResult(
            labels=[g.label for g in result.groups],
            mean=np.array([g.mean for g in result.groups]),
            median=np.array([g.descriptive["median"] for g in result.groups]),
            std=np.array([g.std for g in result.groups]),
            cv=np.array([g.descriptive["cv"] for g in result.groups]),
            iqr=np.array([g.descriptive["iqr"] for g in result.groups]),
            cvar_10=np.array([g.descriptive["cvar_10"] for g in result.groups]),
            percentiles={
                p: np.array([g.descriptive[f"p{p}"] for g in result.groups])
                for p in (10, 25, 50, 75, 90)
            },
            failure_rate=None, failure_threshold=None,
        )
        _print_subsection("--- Descriptive Statistics ---")
        _desc_df = _desc.summary_table()
        _desc_df.index.name = "group"
        print(_desc_df.to_string())
        print()

    # ── Per-group means ──────────────────────────────────────────────────────
    label_width = min(24, max(8, max(len(g.label) for g in result.groups)))
    line_width = 44
    _print_mean_advantage(
        labels=[g.label for g in result.groups],
        mean=np.array([g.mean for g in result.groups]),
        std=np.array([g.std for g in result.groups]),
        ci_low=np.array([g.ci_low for g in result.groups]),
        ci_high=np.array([g.ci_high for g in result.groups]),
        multi_ci_per_entity=[g.multi_ci for g in result.groups],
        resolved_ci_method=result.groups[0].method,
        item_singular="group",
        line_width=line_width,
        template_col_width=label_width,
        style=style,
        n_eff_per_entity=result.marginal_n_eff,
        rho2_per_entity=getattr(result, "marginal_rho2", None),
    )
    print()

    # ── Omnibus test ─────────────────────────────────────────────────────────
    if result.omnibus_test_name is not None:
        _print_omnibus_section(
            label=result.omnibus_test_name,
            statistic=result.omnibus_statistic,
            p_value=result.omnibus_p_value,
            corrected_p_value=result.omnibus_corrected_p_value,
            ppi_applied=result.ppi_applied,
            rho2_eff=result.omnibus_rho2,
            n_eff=result.omnibus_n_eff,
            n_lab_per_entity=result.n_lab_per_condition,
        )

    # ── Pairwise table (includes critical-difference rank bands) ───────────
    _print_pairwise_section(result, line_width=line_width, style=style)

    # ── Pareto front (secondary_metric=), printed right before the executive
    # summary -- same positioning as the paired path. ──────────────────────
    if result.pareto is not None:
        print()
        _print_pareto_section(result.pareto, metric=result.metric_col, show_rank_probabilities=False)

    # ── Executive summary leaderboard ───────────────────────────────────────
    print()
    _print_executive_summary(
        _GroupComparisonResultAsBundle(result),
        item_singular="group", pareto=result.pareto, metric=result.metric_col,
    )
    if result.pareto is not None:
        _print_pareto_callout(result.pareto, metric=result.metric_col)


def _print_next_steps_guidance(
    bundle: "AnalysisBundle",
    *,
    item_plural: str = "templates",
    alpha: Optional[float] = None,
    min_meaningful_diff: Optional[float] = None,
) -> None:
    """Print 'What to do next' guidance block below the executive summary."""
    if alpha is None:
        alpha = get_alpha_ci()

    N = bundle.benchmark.n_inputs
    n_runs = bundle.benchmark.n_runs
    pair_results = list(bundle.pairwise.results.values())
    if not pair_results or N < 2:
        return

    use_ci_for_sig = bundle.pairwise.simultaneous_ci_method is not None

    def _is_sig(r) -> bool:
        if use_ci_for_sig:
            return float(r.ci_low) > 0 or float(r.ci_high) < 0
        return float(r.p_value) < alpha

    any_sig = any(_is_sig(r) for r in pair_results)

    ci_halves = [(float(r.ci_high) - float(r.ci_low)) / 2.0 for r in pair_results]
    gaps = [abs(float(r.point_diff)) for r in pair_results]
    max_ci_half = max(ci_halves)
    max_gap = max(gaps)

    if not np.isfinite(max_ci_half):
        # An unbounded interval -- paired._degenerate_pair_ci reports
        # (-inf, +inf) for a pair whose per-input differences are all
        # identical when the metric has no declared bounds. Every branch
        # below scales a target sample size by max_ci_half, so they would
        # either print "~inf" as guidance or, where the projection is
        # rounded to an int, raise OverflowError outright. The useful advice
        # here isn't about sample size at all: bounds are what make the
        # interval finite, so ask for them and stop.
        print()
        print("  At least one comparison has an unbounded interval: its per-input")
        print("  differences are all identical, and this metric has no declared")
        print("  range, so its mean can't be bounded at any confidence level.")
        print("  More inputs won't resolve that on their own -- pass")
        print("  score_range=(min, max) to get a finite interval.")
        return

    # Entity-level grouping — mirrors the executive summary leaderboard
    labels = list(bundle.labels)
    means = bundle.robustness.mean
    sort_idx = list(np.argsort(-means))
    labels_sorted = [labels[i] for i in sort_idx]
    label_to_group = _assign_significance_groups(bundle.pairwise, labels_sorted)

    groups: dict[str, list[str]] = {}
    for lbl in labels_sorted:
        g = label_to_group.get(lbl, "?")
        groups.setdefault(g, []).append(lbl)
    group_ids = sorted(groups.keys(), key=lambda g: int(g[1:]) if g[1:].isdigit() else 999)
    top_group = groups.get("#1", [])
    n_entities = len(labels_sorted)

    # Run-fraction lever (seed variance)
    sv = bundle.seed_variance
    run_fraction = None
    if sv is not None and n_runs > 1:
        seed_var_mean = float(np.mean(sv.seed_var))
        total_var_mean = float(np.mean(sv.total_var))
        if total_var_mean > 1e-12:
            run_fraction = seed_var_mean / total_var_mean

    n_str = f"N={N:,}" + (f" × {n_runs} runs" if n_runs > 1 else "")

    def _entity_list(names: list[str], limit: int = 3) -> str:
        """Format a list of entity names for inline prose."""
        quoted = [f"'{_truncate_label(n, 20)}'" for n in names]
        if len(quoted) == 1:
            return quoted[0]
        if len(quoted) == 2:
            return f"{quoted[0]} and {quoted[1]}"
        if len(quoted) <= limit:
            return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
        return f"{len(names)} {item_plural}"

    _print_subsection("--- What to do next ---")

    if any_sig:
        # Case C: entities are at least partially ranked — some differences are clear
        lower_entities = [lbl for g in group_ids if g != "#1" for lbl in groups[g]]

        if len(top_group) == 1:
            print(f"  {_entity_list(top_group)} appears to be the clear leader ({n_str}).")
        else:
            print(f"  {_entity_list(top_group)} are statistically tied at the top ({n_str}).")

        if lower_entities:
            verb = "is" if len(lower_entities) == 1 else "are"
            print(f"  {_entity_list(lower_entities)} {verb} clearly ranked below.")

        # If the top group is tied, suggest a lever to separate them
        if len(top_group) > 1:
            top_set = set(top_group)
            top_ci_halves = [
                (float(r.ci_high) - float(r.ci_low)) / 2.0
                for r in pair_results
                if r.template_a in top_set and r.template_b in top_set
            ]
            top_gaps = [
                abs(float(r.point_diff))
                for r in pair_results
                if r.template_a in top_set and r.template_b in top_set
            ]
            if top_ci_halves:
                focus_ci_half = max(top_ci_halves)
                focus_gap = max(top_gaps) if top_gaps else focus_ci_half
                if min_meaningful_diff is not None:
                    target_half = min_meaningful_diff / 2.0
                else:
                    target_half = max(focus_gap * 0.6, focus_ci_half * 0.4)
                n_needed = int(np.ceil(N * (focus_ci_half / max(target_half, 1e-12)) ** 2))
                print()
                print(f"  If you need to pick between the top {len(top_group)}, more inputs could help.")
                if n_needed > N:
                    if min_meaningful_diff is not None:
                        print(f"    How many more? Rough estimate: ~{n_needed:,} inputs to detect a gap of {min_meaningful_diff:g}.")
                    else:
                        print(f"    How many more? Rough estimate: ~{n_needed:,} inputs (from {N:,} now).")

        print()
        print(f"  All results reflect the specific inputs tested here — different inputs")
        print(f"  may shift the rankings.")

    elif max_gap < max_ci_half * 0.5:
        # Case A: gaps small relative to uncertainty — likely null or underpowered
        print(f"  No clear differences detected — all {n_entities} {item_plural} are currently tied ({n_str}).")
        print()
        print(f"  The gaps are small relative to the noise. These {item_plural} may perform")
        print(f"  similarly, though differences may just be too small to see at this scale.")
        print()
        if min_meaningful_diff is not None:
            target_half = min_meaningful_diff / 2.0
            n_needed = int(np.ceil(N * (max_ci_half / max(target_half, 1e-12)) ** 2))
            if n_needed > N:
                print(f"  How you might get more certain if there's a difference: more inputs.")
                print(f"  To have a reasonable shot at detecting a gap of {min_meaningful_diff:g},")
                print(f"  try roughly {n_needed:,} inputs (from {N:,} now).")
            else:
                print(f"  At {N:,} inputs, you'd likely detect a gap of {min_meaningful_diff:g} if it existed.")
                print(f"  More data probably won't change the picture much.")
        else:
            n_needed_rough = N * 4
            print(f"  How you might get more certain if there's a difference: more inputs. At {N:,}, gaps smaller than ~{2*max_ci_half:.3f}")
            print(f"  are generally invisible to this test.")
            print(f"  Rough guide: ~{n_needed_rough:,} inputs could resolve gaps as small as ±{max_ci_half/2:.3f}.")
        if run_fraction is not None and run_fraction > 0.3:
            print()
            print(f"  Run-to-run variability accounts for ~{100*run_fraction:.0f}% of total variance.")
            print(f"  Adding more runs per input could also help narrow the CI.")

    else:
        # Case B: gaps look real but the test can't confirm them yet
        print(f"  No clear differences detected yet ({n_str}).")
        print()
        print(f"  The largest observed gap ({max_gap:.3f}) is close to the margin of uncertainty")
        print(f"  (±{max_ci_half:.3f}). This could be a real difference the data isn't quite")
        print(f"  large enough to confirm — or it could be noise.")
        print()
        print(f"  How you might get more certain if there's a difference: more inputs.")
        if min_meaningful_diff is not None:
            target_half = min_meaningful_diff / 2.0
            n_needed = int(np.ceil(N * (max_ci_half / max(target_half, 1e-12)) ** 2))
            if n_needed > N:
                print(f"  To reliably detect a gap of {min_meaningful_diff:g},")
                print(f"  try roughly {n_needed:,} inputs (from {N:,} now).")
            else:
                print(f"  Your current N may already be enough to detect a gap of {min_meaningful_diff:g}.")
                print(f"  The observed pattern could be real — more inputs might confirm it.")
        else:
            target_half = max(max_gap * 0.6, max_ci_half * 0.5)
            n_needed = int(np.ceil(N * (max_ci_half / max(target_half, 1e-12)) ** 2))
            n_needed = max(n_needed, int(N * 1.5))
            print(f"  Try roughly {n_needed:,} inputs (from {N:,} now) to see if the gaps hold.")
        if run_fraction is not None and run_fraction > 0.3:
            print()
            print(f"  Run-to-run variability accounts for ~{100*run_fraction:.0f}% of total variance.")
            print(f"  More runs per input could be a cheaper lever than adding more inputs.")

    print()
