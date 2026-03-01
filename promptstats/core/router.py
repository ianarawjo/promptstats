"""Central router for selecting the appropriate analysis pipeline.

Inspects the 'shape' of the input — number of models, prompt templates,
input variables, evaluators, and runs — and dispatches to the correct
analysis functions. Raises informative errors for shapes that are not yet
supported.

Supported shapes
----------------
* models=1, prompts>1, input_vars=1, runs>=1, evaluators>=1  →  AnalysisBundle
* models>1, prompts>1, input_vars=1, runs>=1, evaluators>=1  →  MultiModelBundle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Union

import numpy as np

from .types import BenchmarkResult, MultiModelBenchmark
from .paired import PairwiseMatrix, all_pairwise
from .ranking import RankDistribution, MeanAdvantageResult, bootstrap_ranks, bootstrap_mean_advantage
from .variance import RobustnessResult, SeedVarianceResult, robustness_metrics, seed_variance_decomposition


# ---------------------------------------------------------------------------
# Shape descriptor
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkShape:
    """Detected structural properties of a benchmark input.

    Attributes
    ----------
    n_models : int
        Number of distinct LLM models. 1 for BenchmarkResult; ≥2 for
        MultiModelBenchmark.
    n_prompts : int
        Number of prompt templates (templates per model).
    n_input_vars : int
        Number of independent input variables. 1 when each benchmark
        input is a single value; >1 when input_labels are tuples
        representing a cross-product of variables.
    n_evaluators : int
        Number of evaluators/scorers.
    n_runs : int
        Number of repeated runs (seeds) per cell. 1 means no seed dimension.
    """

    n_models: int
    n_prompts: int
    n_input_vars: int
    n_evaluators: int
    n_runs: int = 1

    def __repr__(self) -> str:
        runs_str = f", runs={self.n_runs}" if self.n_runs > 1 else ""
        return (
            f"BenchmarkShape(models={self.n_models}, prompts={self.n_prompts}, "
            f"input_vars={self.n_input_vars}, evaluators={self.n_evaluators}"
            f"{runs_str})"
        )


# ---------------------------------------------------------------------------
# Result bundles
# ---------------------------------------------------------------------------

@dataclass
class AnalysisBundle:
    """Consolidated results from a single-model benchmark analysis run.

    Attributes
    ----------
    benchmark : BenchmarkResult
        The underlying benchmark data.
    shape : BenchmarkShape
        Detected structural properties used for routing.
    pairwise : PairwiseMatrix
        All pairwise statistical comparisons between templates.
    mean_advantage : MeanAdvantageResult
        Mean advantage of each template over a reference, with
        epistemic CI and intrinsic spread bands.
    robustness : RobustnessResult
        Per-template robustness and variance metrics (on cell means).
    rank_dist : RankDistribution
        Bootstrap distribution over template rankings.
    seed_variance : SeedVarianceResult or None
        Seed-variance decomposition (instability scores).  Present only
        when the benchmark carries R >= 3 repeated runs.
    """

    benchmark: BenchmarkResult
    shape: BenchmarkShape
    pairwise: PairwiseMatrix
    mean_advantage: MeanAdvantageResult
    robustness: RobustnessResult
    rank_dist: RankDistribution
    seed_variance: Optional[SeedVarianceResult] = None


@dataclass
class MultiModelBundle:
    """Consolidated results from a multi-model benchmark analysis run.

    Contains three complementary views of the data:

    * **per_model** — one AnalysisBundle per model, answering "which
      prompt works best *within* each model?"
    * **model_level** — models compared on their mean score across all
      prompts, answering "which model is overall best?"
    * **cross_model** — all (model, template) pairs ranked together,
      answering "what is the single best model-prompt combination?"

    Attributes
    ----------
    benchmark : MultiModelBenchmark
        The underlying benchmark data.
    shape : BenchmarkShape
        Detected structural properties used for routing.
    per_model : dict[str, AnalysisBundle]
        One full analysis bundle per model, keyed by model label.
    model_level : AnalysisBundle
        Analysis where each 'template' is a model, scored by its mean
        performance across all prompts.
    cross_model : AnalysisBundle
        Analysis of all N_models * N_templates (model, template) pairs
        treated as a flat list of 'templates'.
    best_pair : tuple[str, str]
        The (model_label, template_label) pair with the highest
        probability of ranking first in the cross_model analysis.
    """

    benchmark: MultiModelBenchmark
    shape: BenchmarkShape
    per_model: Dict[str, AnalysisBundle]
    model_level: AnalysisBundle
    cross_model: AnalysisBundle
    best_pair: tuple[str, str]


# Type alias for the return type of analyze().
AnalysisResult = Union[AnalysisBundle, Dict[str, AnalysisBundle], MultiModelBundle]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    result: Union[BenchmarkResult, MultiModelBenchmark],
    *,
    evaluator_mode: Literal["aggregate", "per_evaluator"] = "aggregate",
    reference: str = "grand_mean",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> AnalysisResult:
    """Run all standard analyses for a benchmark result.

    When the benchmark includes a runs axis with R >= 3, all bootstrap
    analyses automatically use a two-level (nested) resample that propagates
    seed variance into confidence intervals and rank distributions.
    ``AnalysisBundle.seed_variance`` is populated with the per-template
    variance decomposition (instability scores).

    Parameters
    ----------
    result : BenchmarkResult or MultiModelBenchmark
        The benchmark data to analyze.
    evaluator_mode : str
        ``'aggregate'`` (default) analyzes the evaluator-averaged score
        matrix. ``'per_evaluator'`` runs analyses separately for each
        evaluator and returns a dict keyed by evaluator label.
        Not supported for MultiModelBenchmark.
    reference : str
        Reference for mean advantage: ``'grand_mean'`` (default) or a
        template label to compare all others against.
    method : str
        Statistical method for CIs and p-values: ``'auto'`` (default),
        ``'bootstrap'``, or ``'bca'``.
    ci : float
        Confidence level for bootstrap intervals (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    correction : str
        Multiple comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band in mean advantage
        (default ``(10, 90)``).
    failure_threshold : float, optional
        If provided, computes the fraction of inputs scoring below this
        value in robustness metrics.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    AnalysisResult
        AnalysisBundle, dict[str, AnalysisBundle], or MultiModelBundle
        depending on input type and evaluator_mode.

    Raises
    ------
    ValueError
        If the benchmark has fewer than 2 prompt templates.
    NotImplementedError
        If the benchmark shape is not yet supported.
    """
    if rng is None:
        rng = np.random.default_rng()

    kwargs = dict(
        reference=reference,
        method=method,
        ci=ci,
        n_bootstrap=n_bootstrap,
        correction=correction,
        spread_percentiles=spread_percentiles,
        failure_threshold=failure_threshold,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Multi-model path
    # ------------------------------------------------------------------
    if isinstance(result, MultiModelBenchmark):
        if evaluator_mode == "per_evaluator":
            raise NotImplementedError(
                "evaluator_mode='per_evaluator' is not yet supported for "
                "MultiModelBenchmark. Use evaluator_mode='aggregate' instead."
            )
        if evaluator_mode not in {"aggregate", "per_evaluator"}:
            raise ValueError(
                f"Unknown evaluator_mode '{evaluator_mode}'. "
                "Expected 'aggregate' or 'per_evaluator'."
            )
        shape = _detect_shape(result)
        _validate_supported(shape)
        return _analyze_multi_model(result=result, shape=shape, **kwargs)

    # ------------------------------------------------------------------
    # Single-model path (BenchmarkResult)
    # ------------------------------------------------------------------
    if evaluator_mode not in {"aggregate", "per_evaluator"}:
        raise ValueError(
            f"Unknown evaluator_mode '{evaluator_mode}'. "
            "Expected 'aggregate' or 'per_evaluator'."
        )

    shape = _detect_shape(result)
    _validate_supported(shape)

    if evaluator_mode == "aggregate":
        return _analyze_single(result=result, shape=shape, **kwargs)

    # per_evaluator mode — only applies to the 4-D (N, M, R, K) case.
    has_evaluator_axis = result.scores.ndim == 4
    evaluator_names = result.evaluator_names if has_evaluator_axis else ["score"]

    if not has_evaluator_axis:
        outputs: Dict[str, AnalysisBundle] = {
            "score": _analyze_single(result=result, shape=shape, **kwargs)
        }
        return outputs

    outputs = {}
    for evaluator_idx, evaluator_name in enumerate(evaluator_names):
        # Slice out one evaluator, keeping the run axis intact → (N, M, R).
        evaluator_result = BenchmarkResult(
            scores=result.scores[:, :, :, evaluator_idx],
            template_labels=result.template_labels,
            input_labels=result.input_labels,
            input_metadata=result.input_metadata,
            baseline_template=result.baseline_template,
        )
        outputs[evaluator_name] = _analyze_single(
            result=evaluator_result,
            shape=shape,
            **kwargs,
        )

    return outputs


# ---------------------------------------------------------------------------
# Internal analysis runners
# ---------------------------------------------------------------------------

def _analyze_single(
    result: BenchmarkResult,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: Literal["bootstrap", "bca", "auto"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
) -> AnalysisBundle:
    # Use get_run_scores() so that all analysis functions receive either
    # (N, M, R) with R >= 3 (seeded nested bootstrap) or (N, M, 1) which
    # they will collapse to (N, M) and treat as non-seeded.
    run_scores = result.get_run_scores()   # (N, M, R) or (N, M, 1)
    labels = result.template_labels

    pairwise = all_pairwise(
        run_scores, labels,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng,
    )
    mean_adv = bootstrap_mean_advantage(
        run_scores, labels,
        reference=reference,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles, rng=rng,
    )
    robustness = robustness_metrics(
        run_scores, labels,
        failure_threshold=failure_threshold,
    )
    rank_dist = bootstrap_ranks(
        run_scores, labels,
        n_bootstrap=n_bootstrap, rng=rng,
    )

    seed_var = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    return AnalysisBundle(
        benchmark=result,
        shape=shape,
        pairwise=pairwise,
        mean_advantage=mean_adv,
        robustness=robustness,
        rank_dist=rank_dist,
        seed_variance=seed_var,
    )


def _analyze_multi_model(
    result: MultiModelBenchmark,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: Literal["bootstrap", "bca", "auto"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
) -> MultiModelBundle:
    kwargs = dict(
        reference=reference,
        method=method,
        ci=ci,
        n_bootstrap=n_bootstrap,
        correction=correction,
        spread_percentiles=spread_percentiles,
        failure_threshold=failure_threshold,
        rng=rng,
    )

    per_model: Dict[str, AnalysisBundle] = {}
    single_model_shape = BenchmarkShape(
        n_models=1,
        n_prompts=shape.n_prompts,
        n_input_vars=shape.n_input_vars,
        n_evaluators=shape.n_evaluators,
        n_runs=shape.n_runs,
    )
    for model_label in result.model_labels:
        model_result = result.get_model_result(model_label)
        per_model[model_label] = _analyze_single(
            result=model_result,
            shape=single_model_shape,
            **kwargs,
        )

    model_mean_result = result.get_model_mean_result()
    model_level_shape = BenchmarkShape(
        n_models=shape.n_models,
        n_prompts=shape.n_models,
        n_input_vars=shape.n_input_vars,
        n_evaluators=shape.n_evaluators,
        n_runs=shape.n_runs,
    )
    model_level = _analyze_single(
        result=model_mean_result,
        shape=model_level_shape,
        **kwargs,
    )

    flat_result = result.get_flat_result()
    flat_shape = BenchmarkShape(
        n_models=shape.n_models,
        n_prompts=shape.n_models * shape.n_prompts,
        n_input_vars=shape.n_input_vars,
        n_evaluators=shape.n_evaluators,
        n_runs=shape.n_runs,
    )
    cross_model = _analyze_single(
        result=flat_result,
        shape=flat_shape,
        **kwargs,
    )

    best_flat_idx = int(np.argmax(cross_model.rank_dist.p_best))
    best_model_idx = best_flat_idx // result.n_templates
    best_template_idx = best_flat_idx % result.n_templates
    best_pair = (
        result.model_labels[best_model_idx],
        result.template_labels[best_template_idx],
    )

    return MultiModelBundle(
        benchmark=result,
        shape=shape,
        per_model=per_model,
        model_level=model_level,
        cross_model=cross_model,
        best_pair=best_pair,
    )


# ---------------------------------------------------------------------------
# Shape detection and validation
# ---------------------------------------------------------------------------

def _detect_shape(
    result: Union[BenchmarkResult, MultiModelBenchmark],
) -> BenchmarkShape:
    """Infer the structural shape of a benchmark input."""
    if isinstance(result, MultiModelBenchmark):
        n_models = result.n_models
        n_prompts = result.n_templates
        n_evaluators = result.n_evaluators
        n_runs = result.n_runs
        if result.input_labels and isinstance(result.input_labels[0], tuple):
            n_input_vars = len(result.input_labels[0])
        else:
            n_input_vars = 1
        return BenchmarkShape(
            n_models=n_models,
            n_prompts=n_prompts,
            n_input_vars=n_input_vars,
            n_evaluators=n_evaluators,
            n_runs=n_runs,
        )

    # BenchmarkResult
    n_models = 1
    n_prompts = result.n_templates
    n_evaluators = result.n_evaluators
    n_runs = result.n_runs
    if result.input_labels and isinstance(result.input_labels[0], tuple):
        n_input_vars = len(result.input_labels[0])
    else:
        n_input_vars = 1
    return BenchmarkShape(
        n_models=n_models,
        n_prompts=n_prompts,
        n_input_vars=n_input_vars,
        n_evaluators=n_evaluators,
        n_runs=n_runs,
    )


def _validate_supported(shape: BenchmarkShape) -> None:
    """Raise if the shape is outside the currently supported pipelines."""
    if shape.n_prompts < 2:
        raise ValueError(
            f"analyze() requires at least 2 prompt templates; got {shape.n_prompts}. "
            "Add more templates to enable comparative analysis."
        )

    if shape.n_input_vars > 1:
        raise NotImplementedError(
            f"Cross-product input analysis (n_input_vars={shape.n_input_vars}) is "
            "not yet supported. Flatten the input space to a single variable "
            "(e.g., by joining variable values into one label) before calling "
            "analyze()."
        )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_analysis_summary(
    analysis: Union[AnalysisBundle, MultiModelBundle, Mapping[str, AnalysisBundle]],
    *,
    top_pairwise: int = 5,
    line_width: int = 41,
) -> None:
    """Print a concise console summary of analyze() results."""
    if isinstance(analysis, MultiModelBundle):
        _print_multi_model_summary(
            analysis,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )
        return

    if isinstance(analysis, AnalysisBundle):
        _print_bundle_summary(
            analysis,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )
        return

    for evaluator_name, bundle in analysis.items():
        print(f"=== Evaluator: {evaluator_name} ===")
        _print_bundle_summary(
            bundle,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )
        print()


def _print_multi_model_summary(
    bundle: MultiModelBundle,
    *,
    top_pairwise: int,
    line_width: int,
) -> None:
    print("=== Multi-Model Analysis Summary ===")
    print(f"Shape: {bundle.shape}")
    print(
        f"Models: {bundle.benchmark.n_models} | "
        f"Templates: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {bundle.benchmark.n_runs}" if bundle.benchmark.n_runs > 1 else "")
    )
    model_str = ", ".join(bundle.benchmark.model_labels)
    print(f"Models: {model_str}")
    best_model, best_template = bundle.best_pair
    print(f"Best pair: model='{best_model}'  template='{best_template}'")
    print()

    print("--- Model-Level Comparison (mean across all prompts) ---")
    _print_bundle_summary(
        bundle.model_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
    )

    for model_label, model_bundle in bundle.per_model.items():
        print(f"\n--- Per-Model: '{model_label}' ---")
        _print_bundle_summary(
            model_bundle,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )

    print("\n=== Cross-Model Ranking (all model/template pairs) ===")
    print(
        f"  {len(bundle.cross_model.rank_dist.labels)} pairs ranked. "
        f"Top 5 by P(Best):"
    )
    p_best = bundle.cross_model.rank_dist.p_best
    top_indices = np.argsort(-p_best)[:5]
    for idx in top_indices:
        label = bundle.cross_model.rank_dist.labels[idx]
        print(f"    {label:<40s}  P(Best)={p_best[idx]:.1%}")
    print()


def _print_bundle_summary(
    bundle: AnalysisBundle,
    *,
    top_pairwise: int,
    line_width: int,
) -> None:
    template_col_width = 24
    pair_col_width = 32

    print("=== Analysis Summary ===")
    print(f"Shape: {bundle.shape}")
    n_runs = bundle.benchmark.n_runs
    print(
        f"Templates: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {n_runs}" if n_runs > 1 else "")
    )
    print()

    print("--- Robustness ---")
    print(bundle.robustness.summary_table().to_string())
    print()

    # Seed variance section (only when seeded data is present).
    if bundle.seed_variance is not None:
        sv = bundle.seed_variance
        print(f"--- Seed Variance (R={sv.n_runs} runs) ---")
        print(sv.summary_table().to_string())
        print()

    print("--- Rank Probabilities ---")
    print(f"  {'Template':<24s} {'P(Best)':>9s} {'E[Rank]':>9s}")
    for i, label in enumerate(bundle.rank_dist.labels):
        print(
            f"  {label:<24s} "
            f"{bundle.rank_dist.p_best[i]:>8.1%} "
            f"{bundle.rank_dist.expected_ranks[i]:>8.2f}"
        )
    print()

    print(f"--- Mean Advantage (reference={bundle.mean_advantage.reference}) ---")
    low_p, high_p = bundle.mean_advantage.spread_percentiles
    ma = bundle.mean_advantage
    ma_max_abs = max(
        1e-12,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            ma.mean_advantages,
                            ma.bootstrap_ci_low,
                            ma.bootstrap_ci_high,
                            ma.spread_low,
                            ma.spread_high,
                        ]
                    )
                )
            )
        ),
    )
    ma_low = -ma_max_abs
    ma_high = ma_max_abs
    print(f"  axis: [{ma_low:+.3f}, {ma_high:+.3f}]  (· spread, ─ CI, ● mean, │ zero)")
    print(
        f"  {'Template':<{template_col_width}s} {'Interval Plot':<{line_width}s} {'Mean':>8s} "
        f"{'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )
    for i, label in enumerate(bundle.mean_advantage.labels):
        template_label = _truncate_label(label, template_col_width)
        line = _ascii_interval_line(
            mean=float(bundle.mean_advantage.mean_advantages[i]),
            ci_low=float(bundle.mean_advantage.bootstrap_ci_low[i]),
            ci_high=float(bundle.mean_advantage.bootstrap_ci_high[i]),
            spread_low=float(bundle.mean_advantage.spread_low[i]),
            spread_high=float(bundle.mean_advantage.spread_high[i]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
        )
        print(
            f"  {template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{bundle.mean_advantage.mean_advantages[i]:>+7.3f} "
            f"{bundle.mean_advantage.bootstrap_ci_low[i]:>+8.3f} "
            f"{bundle.mean_advantage.bootstrap_ci_high[i]:>+8.3f} "
            f"{bundle.mean_advantage.spread_low[i]:>+9.3f} "
            f"{bundle.mean_advantage.spread_high[i]:>+9.3f}"
        )
    print(f"  spread percentiles = ({low_p:g}, {high_p:g})")
    print()

    print("--- Pairwise Comparisons (lowest p-value first) ---")
    pair_results = sorted(
        bundle.pairwise.results.values(),
        key=lambda r: (r.p_value, -abs(r.mean_diff)),
    )
    max_pairs = max(0, min(top_pairwise, len(pair_results)))
    if max_pairs > 0:
        pair_max_abs = max(
            1e-12,
            max(
                max(
                    abs(float(result.mean_diff)),
                    abs(float(result.ci_low)),
                    abs(float(result.ci_high)),
                    abs(float(result.mean_diff - result.std_diff)),
                    abs(float(result.mean_diff + result.std_diff)),
                )
                for result in pair_results[:max_pairs]
            ),
        )
        pair_low = -pair_max_abs
        pair_high = pair_max_abs
        print(
            f"  axis: [{pair_low:+.3f}, {pair_high:+.3f}]  "
            "(· ±1σ, ─ CI, ● mean, │ zero)"
        )
        print(
            f"  {'Pair':<{pair_col_width}s} {'Interval Plot':<{line_width}s} {'Mean':>8s} "
            f"{'CI Low':>9s} {'CI High':>9s} {'σ':>8s} {'p':>9s} {'sig':>5s}"
        )

    for result in pair_results[:max_pairs]:
        line = _ascii_interval_line(
            mean=float(result.mean_diff),
            ci_low=float(result.ci_low),
            ci_high=float(result.ci_high),
            spread_low=float(result.mean_diff - result.std_diff),
            spread_high=float(result.mean_diff + result.std_diff),
            axis_low=pair_low,
            axis_high=pair_high,
            width=line_width,
        )
        pair_label = _truncate_label(
            f"{result.template_a} vs {result.template_b}",
            pair_col_width,
        )
        print(
            f"  {pair_label:<{pair_col_width}s} "
            f"{line:<{line_width}s} "
            f"{result.mean_diff:+.4f} "
            f"{result.ci_low:+.4f} "
            f"{result.ci_high:+.4f} "
            f"{result.std_diff:>7.4f} "
            f"{result.p_value:>9.4g} "
            f"{str(result.significant):>5s}"
        )

    if max_pairs == 0:
        print("  (no pairwise comparisons)")


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
) -> str:
    """Render a one-line ASCII interval plot with zero marker."""
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

    zero_idx = to_idx(0.0)
    chars[zero_idx] = "│"
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
