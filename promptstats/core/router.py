"""Central router for selecting the appropriate analysis pipeline.

Inspects the 'shape' of the input — number of models, prompt templates,
input variables, evaluators, and runs — and dispatches to the correct
analysis functions. Raises informative errors for shapes that are not yet
supported.

Supported shapes
----------------
* models=1, prompts>1, input_vars=1, runs>=1, evaluators>=1  →  AnalysisBundle
* models>1, prompts>1, input_vars=1, runs>=1, evaluators>=1  →  MultiModelBundle
* models>1, prompts=1, input_vars=1, runs>=1, evaluators>=1  →  MultiModelBundle (warn)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Literal, Mapping, Optional, Union

import numpy as np

from promptstats.vis.critical_difference import plot_critical_difference

from .types import BenchmarkResult, MultiModelBenchmark
from .paired import PairwiseMatrix, all_pairwise
from .ranking import RankDistribution, PointAdvantageResult, bootstrap_ranks, bootstrap_point_advantage
from .variance import RobustnessResult, SeedVarianceResult, robustness_metrics, seed_variance_decomposition
from .tokens import TokenUsage, TokenAnalysisResult, analyze_tokens

if TYPE_CHECKING:
    from .mixed_effects import LMMInfo


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
    point_advantage : PointAdvantageResult
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
    point_advantage: PointAdvantageResult
    robustness: RobustnessResult
    rank_dist: RankDistribution
    seed_variance: Optional[SeedVarianceResult] = None
    token_analysis: Optional[TokenAnalysisResult] = None
    lmm_info: Optional[LMMInfo] = None


@dataclass
class MultiModelBundle:
    """Consolidated results from a multi-model benchmark analysis run.

    Contains three complementary views of the data:

    * **per_model** — one AnalysisBundle per model, answering "which
      prompt works best *within* each model?"
    * **model_level** — models compared on their mean score across all
      prompts, answering "which model is overall best?"
    * **template_level** — templates compared on their mean score across
      all models, answering "which prompt is best/worst overall?"
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
    template_level : AnalysisBundle
        Analysis where each 'template' is a prompt template, scored by
        its mean performance across all models.
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
    template_level: AnalysisBundle
    cross_model: AnalysisBundle
    best_pair: tuple[str, str]


# Type aliases for the return type of analyze().
PerEvaluatorSingleModel = Dict[str, AnalysisBundle]
PerEvaluatorMultiModel = Dict[str, MultiModelBundle]
AnalysisResult = Union[
    AnalysisBundle,
    PerEvaluatorSingleModel,
    MultiModelBundle,
    PerEvaluatorMultiModel,
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    result: Union[BenchmarkResult, MultiModelBenchmark],
    *,
    token_usage: Optional[TokenUsage] = None,
    evaluator_mode: Literal["aggregate", "per_evaluator"] = "aggregate",
    reference: str = "grand_mean",
    method: Literal["bootstrap", "bca", "auto", "lmm"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    template_model_collapse: Literal["mean", "as_runs"] = "as_runs",
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
    token_usage : TokenUsage, optional
        Token counts (input and/or output) aligned to the benchmark
        templates and inputs.  When provided, ``AnalysisBundle`` is
        extended with a ``token_analysis`` field containing per-template
        token CIs, pairwise token comparisons, and a Pareto frontier
        analysis combining token cost with performance.  Only supported
        for ``BenchmarkResult`` in ``evaluator_mode='aggregate'``.
    evaluator_mode : str
        ``'aggregate'`` (default) analyzes the evaluator-averaged score
        matrix. ``'per_evaluator'`` runs analyses separately for each
        evaluator and returns a dict keyed by evaluator label.
        Not supported for MultiModelBenchmark.
    reference : str
        Reference for advantage: ``'grand_mean'`` (default) or a
        template label to compare all others against.  The grand
        reference is always the per-input mean across templates
        regardless of ``statistic``; using the per-input median would
        make the middle-ranked template's advantages identically zero
        (degeneracy when N is odd).
    method : str
        Statistical method for CIs and p-values:

        * ``'auto'`` (default) — BCa bootstrap for 15 ≤ M ≤ 200,
          percentile bootstrap otherwise.
        * ``'bootstrap'`` — percentile bootstrap.
        * ``'bca'`` — bias-corrected and accelerated bootstrap.
        * ``'lmm'`` — Linear Mixed Model via pymer4/lme4.  Fits
          ``score ~ template + (1|input)`` on cell-mean scores.
          Produces Wald CIs and emmeans-based pairwise contrasts.
          Requires pymer4 and R (``pip install pymer4``).
          Prefer this when M < ~15 (bootstrap unstable) or when an
          ICC decomposition is desired.  ``AnalysisBundle.lmm_info``
          is populated with variance components and the ICC.
          Not compatible with ``statistic='median'``.
    ci : float
        Confidence level for intervals (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).  When
        ``method='lmm'`` this controls the number of parametric
        simulations used for the rank distribution.
    correction : str
        Multiple comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band in the advantage plot
        (default ``(10, 90)``).
    failure_threshold : float, optional
        If provided, computes the fraction of inputs scoring below this
        value in robustness metrics.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Central-tendency statistic for point estimates and bootstrap
        resampling: ``'mean'`` (default) or ``'median'``.  Mean works
        well for the majority of LLM benchmarks, including bounded and
        semi-discrete scoring rubrics (pass/fail, BERTScore, ROUGE),
        where the bootstrap already handles non-normality.  Use
        ``'median'`` when scores follow a genuinely continuous,
        heavy-tailed distribution where the median better represents
        typical performance than the mean; note that median will produce
        uninformative zero-width CIs whenever more than half of the
        per-input score differences between two templates are identical
        (common with clustered or ceiling-bounded scores).  All
        bootstrap CIs and p-values are computed using the same
        statistic.  Not compatible with ``method='lmm'``.
    template_model_collapse : str
        Multi-model only. Controls how the per-template (model-agnostic)
        view collapses the model axis:

        * ``'mean'`` averages over models.
        * ``'as_runs'`` (default) treats models as additional runs to preserve
            cross-model variation in uncertainty estimates.

    Returns
    -------
    AnalysisResult
        AnalysisBundle, dict[str, AnalysisBundle], or MultiModelBundle
        depending on input type and evaluator_mode.

    Raises
    ------
    ValueError
        If the benchmark has fewer than 2 prompt templates, or if
        ``statistic='median'`` is combined with ``method='lmm'``.
    NotImplementedError
        If the benchmark shape is not yet supported.
    ImportError
        If ``method='lmm'`` and pymer4 is not installed.
    """
    if rng is None:
        rng = np.random.default_rng()

    if statistic not in {"mean", "median"}:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Expected 'mean' or 'median'."
        )
    if template_model_collapse not in {"mean", "as_runs"}:
        raise ValueError(
            f"Unknown template_model_collapse '{template_model_collapse}'. "
            "Expected 'mean' or 'as_runs'."
        )

    if method != "lmm" and result.n_inputs < 15:
        warnings.warn(
            f"Only M={result.n_inputs} benchmark input(s) detected. "
            "Bootstrap confidence intervals are unreliable with fewer than ~15 inputs. "
            "Consider using method='lmm' for more stable inference with small samples "
            "(requires pymer4).",
            UserWarning,
            stacklevel=2,
        )

    kwargs = dict(
        reference=reference,
        method=method,
        ci=ci,
        n_bootstrap=n_bootstrap,
        correction=correction,
        spread_percentiles=spread_percentiles,
        failure_threshold=failure_threshold,
        rng=rng,
        statistic=statistic,
    )

    # ------------------------------------------------------------------
    # Multi-model path
    # ------------------------------------------------------------------
    if isinstance(result, MultiModelBenchmark):
        if token_usage is not None:
            warnings.warn(
                "token_usage is not yet supported for MultiModelBenchmark "
                "and will be ignored. Pass a BenchmarkResult instead.",
                UserWarning,
                stacklevel=2,
            )
        if evaluator_mode not in {"aggregate", "per_evaluator"}:
            raise ValueError(
                f"Unknown evaluator_mode '{evaluator_mode}'. "
                "Expected 'aggregate' or 'per_evaluator'."
            )
        if evaluator_mode == "per_evaluator":
            has_evaluator_axis = result.scores.ndim == 5
            if not has_evaluator_axis:
                shape = _detect_shape(result)
                _validate_supported(shape)
                return {
                    "score": _analyze_multi_model(
                        result=result,
                        shape=shape,
                        template_model_collapse=template_model_collapse,
                        **kwargs,
                    )
                }

            outputs: PerEvaluatorMultiModel = {}
            for evaluator_idx, evaluator_name in enumerate(result.evaluator_names):
                evaluator_result = MultiModelBenchmark(
                    scores=result.scores[:, :, :, :, evaluator_idx],
                    model_labels=result.model_labels,
                    template_labels=result.template_labels,
                    input_labels=result.input_labels,
                    input_metadata=result.input_metadata,
                )
                evaluator_shape = _detect_shape(evaluator_result)
                _validate_supported(evaluator_shape)
                outputs[evaluator_name] = _analyze_multi_model(
                    result=evaluator_result,
                    shape=evaluator_shape,
                    template_model_collapse=template_model_collapse,
                    **kwargs,
                )
            return outputs

        shape = _detect_shape(result)
        _validate_supported(shape)
        return _analyze_multi_model(
            result=result,
            shape=shape,
            template_model_collapse=template_model_collapse,
            **kwargs,
        )

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
        bundle = _analyze_single(result=result, shape=shape, **kwargs)
        if token_usage is not None:
            _validate_token_usage(token_usage, result)
            bundle.token_analysis = analyze_tokens(
                token_usage,
                bundle.pairwise,
                method=method,
                ci=ci,
                n_bootstrap=n_bootstrap,
                correction=correction,
                rng=rng,
            )
        return bundle

    # per_evaluator mode — only applies to the 4-D (N, M, R, K) case.
    if token_usage is not None:
        warnings.warn(
            "token_usage is ignored when evaluator_mode='per_evaluator'. "
            "Token analysis only runs in aggregate mode.",
            UserWarning,
            stacklevel=2,
        )
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
    method: Literal["bootstrap", "bca", "auto", "lmm"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
) -> AnalysisBundle:
    # ------------------------------------------------------------------
    # LMM path — fit score ~ template + (1|input) via pymer4/lme4
    # ------------------------------------------------------------------
    if method == "lmm":
        if statistic == "median":
            warnings.warn(
                "statistic='median' is not compatible with method='lmm' "
                "(the LMM is a mean-based model). Falling back to "
                "statistic='mean' for this analysis. Pass statistic='mean' "
                "explicitly to silence this warning, or switch to "
                "method='auto' to use median with the bootstrap.",
                UserWarning,
                stacklevel=2,
            )
            statistic = "mean"
        from .mixed_effects import lmm_analyze
        pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info = lmm_analyze(
            result,
            reference=reference,
            ci=ci,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            n_sim=n_bootstrap,
            rng=rng,
        )
        return AnalysisBundle(
            benchmark=result,
            shape=shape,
            pairwise=pairwise,
            point_advantage=mean_adv,
            robustness=robustness,
            rank_dist=rank_dist,
            seed_variance=seed_var,
            lmm_info=lmm_info,
        )

    # ------------------------------------------------------------------
    # Bootstrap path (default)
    # Use get_run_scores() so that all analysis functions receive either
    # (N, M, R) with R >= 3 (seeded nested bootstrap) or (N, M, 1) which
    # they will collapse to (N, M) and treat as non-seeded.
    # ------------------------------------------------------------------
    if result.has_missing:
        n_missing = int(np.sum(np.isnan(result.scores)))
        raise ValueError(
            f"scores contain {n_missing} NaN (missing) cell(s), which are not "
            "supported by the bootstrap analysis path. Either fill in missing "
            "cells or use method='lmm' to analyse benchmarks with incomplete "
            "designs."
        )

    run_scores = result.get_run_scores()   # (N, M, R) or (N, M, 1)
    labels = result.template_labels

    pairwise = all_pairwise(
        run_scores, labels,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng, statistic=statistic,
    )
    mean_adv = bootstrap_point_advantage(
        run_scores, labels,
        reference=reference,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles, rng=rng, statistic=statistic,
    )
    robustness = robustness_metrics(
        run_scores, labels,
        failure_threshold=failure_threshold,
    )
    rank_dist = bootstrap_ranks(
        run_scores, labels,
        n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
    )

    seed_var = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    return AnalysisBundle(
        benchmark=result,
        shape=shape,
        pairwise=pairwise,
        point_advantage=mean_adv,
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
    statistic: Literal["mean", "median"],
    template_model_collapse: Literal["mean", "as_runs"] = "as_runs",
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
        statistic=statistic,
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

    template_mean_result = result.get_template_mean_result(
        collapse_models=template_model_collapse,
    )
    template_level_shape = BenchmarkShape(
        n_models=1,
        n_prompts=shape.n_prompts,
        n_input_vars=shape.n_input_vars,
        n_evaluators=shape.n_evaluators,
        n_runs=template_mean_result.n_runs,
    )
    template_level = _analyze_single(
        result=template_mean_result,
        shape=template_level_shape,
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
        template_level=template_level,
        cross_model=cross_model,
        best_pair=best_pair,
    )


# ---------------------------------------------------------------------------
# Token usage validation
# ---------------------------------------------------------------------------

def _validate_token_usage(
    token_usage: TokenUsage,
    result: BenchmarkResult,
) -> None:
    """Raise ValueError if token_usage is incompatible with the benchmark."""
    if token_usage.template_labels != result.template_labels:
        raise ValueError(
            "token_usage.template_labels does not match benchmark template_labels.\n"
            f"  token_usage: {token_usage.template_labels}\n"
            f"  benchmark:   {result.template_labels}"
        )
    if token_usage.input_labels != result.input_labels:
        raise ValueError(
            "token_usage.input_labels does not match benchmark input_labels.\n"
            f"  token_usage: {token_usage.input_labels}\n"
            f"  benchmark:   {result.input_labels}"
        )
    N, M = result.n_templates, result.n_inputs
    out = token_usage.output_tokens
    if out.ndim not in (2, 3):
        raise ValueError(
            f"token_usage.output_tokens must be 2-D (N, M) or 3-D (N, M, R); "
            f"got shape {out.shape}."
        )
    if out.shape[:2] != (N, M):
        raise ValueError(
            f"token_usage.output_tokens shape {out.shape} is incompatible with "
            f"benchmark (N={N}, M={M}). Expected first two dims ({N}, {M})."
        )
    if token_usage.input_tokens is not None:
        inp = token_usage.input_tokens
        if inp.shape != (N, M):
            raise ValueError(
                f"token_usage.input_tokens shape {inp.shape} must be ({N}, {M})."
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
        if shape.n_models > 1 and shape.n_prompts == 1:
            warnings.warn(
                "Single-prompt multi-model analysis is supported, but per-model "
                "within-prompt comparisons are degenerate (only one template per "
                "model). Results are still computed for model-level and cross-model "
                "comparisons.",
                UserWarning,
                stacklevel=3,
            )
            return
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
    analysis: Union[
        AnalysisBundle,
        MultiModelBundle,
        Mapping[str, AnalysisBundle],
        Mapping[str, MultiModelBundle],
    ],
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
        _print_loud_section(f"Evaluator: {evaluator_name}")
        if isinstance(bundle, MultiModelBundle):
            _print_multi_model_summary(
                bundle,
                top_pairwise=top_pairwise,
                line_width=line_width,
            )
        else:
            _print_bundle_summary(
                bundle,
                top_pairwise=top_pairwise,
                line_width=line_width,
            )
        print()


def _print_loud_section(title: str) -> None:
    heading = f" {title.upper()} "
    border = "=" * len(heading)
    print(border)
    print(heading)
    print(border)


def _print_multi_model_summary(
    bundle: MultiModelBundle,
    *,
    top_pairwise: int,
    line_width: int,
) -> None:
    _print_loud_section("Multi-Model Analysis Summary")
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

    _print_loud_section("Model-level comparison (across all prompts):")
    _print_bundle_summary(
        bundle.model_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
        item_singular="model",
        item_plural="models",
    )

    print()
    _print_loud_section("Cross-model per-template comparison (models collapsed):")
    _print_bundle_summary(
        bundle.template_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
        item_singular="template",
        item_plural="templates",
    )
    best_idx = int(np.argmax(bundle.template_level.robustness.mean))
    best_template = bundle.template_level.benchmark.template_labels[best_idx]
    best_mean = float(bundle.template_level.robustness.mean[best_idx])
    print(
        "  -> Best-performing prompt across models (by mean score): "
        f"'{best_template}' (mean={best_mean:.3f})"
    )

    # Instability across runs across models 
    instability_rows = _collect_cross_model_seed_instability_rows(bundle)
    if instability_rows:
        _print_cross_model_seed_instability(bundle, rows=instability_rows)
        most_stable_model, instability, *_ = instability_rows[0]
        print(
            "  -> Most stable model across runs: "
            f"'{most_stable_model}' "
            f"(instability={instability:.4f}, {_instability_label(instability)})"
        )

    for model_label, model_bundle in bundle.per_model.items():
        print()
        _print_loud_section(f"Per-Model Summary: {model_label}")
        _print_bundle_summary(
            model_bundle,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )

    print()
    _print_loud_section("Cross-Model Ranking (all model/template pairs)")
    _print_model_template_matrix(bundle)
    p_best = bundle.cross_model.rank_dist.p_best
    expected_ranks = bundle.cross_model.rank_dist.expected_ranks
    rank_labels = bundle.cross_model.rank_dist.labels
    rank_pairs = [_split_model_template_label(label) for label in rank_labels]
    rank_bar_width = 14
    n_ranked_items = len(rank_labels)
    model_col_width = min(24, max(len(model) for model, _ in rank_pairs) + 2)
    template_col_width = min(24, max(len(template) for _, template in rank_pairs) + 2)
    top_indices = np.argsort(-p_best)
    n_show = len(top_indices)
    print(f"--- Rank Probabilities: All {n_show} by P(Best) ---")
    print(
        f"  {'Model':<{model_col_width}s} "
        f"{'Template':<{template_col_width}s} "
        f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
        f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
    )
    for idx in top_indices[:n_show]:
        model_label, template_label = rank_pairs[idx]
        model_label = _truncate_label(model_label, model_col_width)
        template_label = _truncate_label(template_label, template_col_width)
        p_best_i = float(p_best[idx])
        expected_rank_i = float(expected_ranks[idx])
        print(
            f"  {model_label:<{model_col_width}s} "
            f"{template_label:<{template_col_width}s} "
            f"{p_best_i:>8.1%} {_ratio_bar(p_best_i, width=rank_bar_width)} "
            f"{expected_rank_i:>8.2f} "
            f"{_rank_hump_lane(expected_rank_i, n_ranked_items, width=rank_bar_width)}"
        )

    ma = bundle.cross_model.point_advantage
    stat_label = ma.statistic.capitalize()
    low_p, high_p = ma.spread_percentiles
    ma_max_abs = max(
        1e-12,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            ma.point_advantages,
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
    print()
    print(
        f"--- {stat_label} Advantage: All {n_show} (reference={ma.reference}) ---"
    )
    print(
        f"  axis: [{ma_low:+.3f}, {ma_high:+.3f}]  "
        f"(· spread, ─ CI, ● {stat_label.lower()}, │ zero)  "
        f"spread percentiles = ({low_p:g}, {high_p:g})"
    )
    print(
        f"  {'Model':<{model_col_width}s} "
        f"{'Template':<{template_col_width}s} "
        f"{'Interval Plot':<{line_width}s} "
        f"{stat_label:>8s} {'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )

    if ma.reference == "grand_mean":
        ref_offset = float(np.mean(bundle.cross_model.robustness.mean))
    else:
        try:
            ref_idx = ma.labels.index(ma.reference)
            ref_offset = float(bundle.cross_model.robustness.mean[ref_idx])
        except ValueError:
            ref_offset = 0.0

    for idx in top_indices[:n_show]:
        model_label, template_label = _split_model_template_label(ma.labels[idx])
        model_label = _truncate_label(model_label, model_col_width)
        template_label = _truncate_label(template_label, template_col_width)
        line = _ascii_interval_line(
            mean=float(ma.point_advantages[idx]),
            ci_low=float(ma.bootstrap_ci_low[idx]),
            ci_high=float(ma.bootstrap_ci_high[idx]),
            spread_low=float(ma.spread_low[idx]),
            spread_high=float(ma.spread_high[idx]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
        )
        abs_point = float(ma.point_advantages[idx]) + ref_offset
        abs_ci_low = float(ma.bootstrap_ci_low[idx]) + ref_offset
        abs_ci_high = float(ma.bootstrap_ci_high[idx]) + ref_offset
        abs_spread_low = float(ma.spread_low[idx]) + ref_offset
        abs_spread_high = float(ma.spread_high[idx]) + ref_offset
        print(
            f"  {model_label:<{model_col_width}s} "
            f"{template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{abs_point:>7.3f} "
            f"{abs_ci_low:>8.3f} "
            f"{abs_ci_high:>8.3f} "
            f"{abs_spread_low:>9.3f} "
            f"{abs_spread_high:>9.3f}"
        )
    print()


def _print_model_template_matrix(bundle: MultiModelBundle) -> None:
    """Print a model × template score matrix (mean ±std, heat encoding)."""
    model_labels = bundle.benchmark.model_labels
    template_labels = bundle.benchmark.template_labels

    # Build (model, template) -> (mean, std) from the flat cross_model bundle.
    # Labels are formatted as "model / template" by get_flat_result().
    cell_mean: dict[tuple[str, str], float] = {}
    for label, m in zip(
        bundle.cross_model.rank_dist.labels,
        bundle.cross_model.robustness.mean,
    ):
        parts = label.split(" / ", 1)
        if len(parts) == 2:
            cell_mean[(parts[0], parts[1])] = float(m)

    all_means = list(cell_mean.values())
    mn, mx = min(all_means), max(all_means)
    best_mean = mx
    heat_chars = "·░▒▓█"

    def _heat(v: float) -> str:
        if mx == mn:
            return heat_chars[-1]
        idx = min(int((v - mn) / (mx - mn) * len(heat_chars)), len(heat_chars) - 1)
        return heat_chars[idx]

    # Row averages (per model), column averages (per template).
    # row_avg = {
    #     mdl: float(np.mean([cell_mean[(mdl, t)] for t in template_labels if (mdl, t) in cell_mean]))
    #     for mdl in model_labels
    # }
    # col_avg = {
    #     t: float(np.mean([cell_mean[(mdl, t)] for mdl in model_labels if (mdl, t) in cell_mean]))
    #     for t in template_labels
    # }

    # Cell width: at least enough for "0.800 ▓*" (8 chars), but expand
    # when template labels are longer so header/data columns stay aligned.
    CELL_W = max(8, max(len(t) for t in template_labels))
    model_col_w = max(len(m) for m in model_labels)

    def _fmt_cell(mdl: str, t: str) -> str:
        if (mdl, t) not in cell_mean:
            return f"{'N/A':^{CELL_W}}"
        m = cell_mean[(mdl, t)]
        h = _heat(m)
        marker = "*" if m == best_mean else " "
        return f"{m:.3f} {h}{marker}".rjust(CELL_W)

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
            # row += f"   │  {row_avg[mdl]:.3f}"
        print(row)

    # Footer (column averages)
    print(div)
    print(f"  * = global best pair  |  heat: · (low) → █ (high), range [{mn:.3f}, {mx:.3f}]")
    print()


def _print_bundle_summary(
    bundle: AnalysisBundle,
    *,
    top_pairwise: int,
    line_width: int,
    item_singular: str = "template",
    item_plural: str = "templates",
) -> None:
    template_col_width = 24
    pair_col_width = 32
    pair_stat_col_width = 8
    pair_ci_col_width = 9
    pair_sigma_col_width = 8
    pair_p_boot_col_width = 10
    pair_p_wsr_col_width = 9
    pair_p_nem_col_width = 9

    # print("=== Analysis Summary ===")
    print(f"Shape: {bundle.shape}")
    n_runs = bundle.benchmark.n_runs
    item_singular_title = item_singular.capitalize()
    item_plural_title = item_plural.capitalize()
    print(
        f"{item_plural_title}: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {n_runs}" if n_runs > 1 else "")
    )
    print()

    print("--- Robustness ---")
    print(bundle.robustness.summary_table().to_string())
    print()

    print("--- Rank Probabilities ---")
    rank_bar_width = 14
    n_ranked_items = len(bundle.rank_dist.labels)
    print(
        f"  {item_singular_title:<24s} "
        f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
        f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
    )
    for i, label in enumerate(bundle.rank_dist.labels):
        p_best = float(bundle.rank_dist.p_best[i])
        expected_rank = float(bundle.rank_dist.expected_ranks[i])
        print(
            f"  {label:<24s} "
            f"{p_best:>8.1%} {_ratio_bar(p_best, width=rank_bar_width)} "
            f"{expected_rank:>8.2f} {_rank_hump_lane(expected_rank, n_ranked_items, width=rank_bar_width)}"
        )
    print("  E[Rank] lane: left is better (#1); peak is sharper near integer ranks, softer near half-ranks")
    print()

    stat_label = bundle.point_advantage.statistic.capitalize()
    print(f"--- {stat_label} Advantage (reference={bundle.point_advantage.reference}) ---")
    low_p, high_p = bundle.point_advantage.spread_percentiles
    ma = bundle.point_advantage
    ma_max_abs = max(
        1e-12,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            ma.point_advantages,
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
    print(f"  axis: [{ma_low:+.3f}, {ma_high:+.3f}]  (· spread, ─ CI, ● {stat_label.lower()}, │ zero)  spread percentiles = ({low_p:g}, {high_p:g})")
    print(
        f"  {item_singular_title:<{template_col_width}s} {'Interval Plot':<{line_width}s} {stat_label:>8s} "
        f"{'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )
    for i, label in enumerate(bundle.point_advantage.labels):
        template_label = _truncate_label(label, template_col_width)
        line = _ascii_interval_line(
            mean=float(bundle.point_advantage.point_advantages[i]),
            ci_low=float(bundle.point_advantage.bootstrap_ci_low[i]),
            ci_high=float(bundle.point_advantage.bootstrap_ci_high[i]),
            spread_low=float(bundle.point_advantage.spread_low[i]),
            spread_high=float(bundle.point_advantage.spread_high[i]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
        )
        print(
            f"  {template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{bundle.point_advantage.point_advantages[i]:>+7.3f} "
            f"{bundle.point_advantage.bootstrap_ci_low[i]:>+8.3f} "
            f"{bundle.point_advantage.bootstrap_ci_high[i]:>+8.3f} "
            f"{bundle.point_advantage.spread_low[i]:>+9.3f} "
            f"{bundle.point_advantage.spread_high[i]:>+9.3f}"
        )
    print()

    # Determine statistic label from the first result (all share the same statistic).
    first_result = next(iter(bundle.pairwise.results.values()), None)
    pair_stat_label = first_result.statistic.capitalize() if first_result else "Mean"
    print(f"--- Pairwise Comparisons (lowest p-value first) ---")
    pair_results = sorted(
        bundle.pairwise.results.values(),
        key=lambda r: (r.p_value, -abs(r.point_diff)),
    )
    max_pairs = max(0, min(top_pairwise, len(pair_results)))
    # Friedman omnibus line (printed before the interval plot when pairs exist).
    if max_pairs > 0 and bundle.pairwise.friedman is not None:
        fr = bundle.pairwise.friedman
        # Plotting a critical difference diagram example:
        # fig = plot_critical_difference(fr, title="Friedman Test Critical Difference Diagram")
        # fig.savefig("friedman_cd.png")
        fr_p_str = _format_p_value(fr.p_value)
        print(f"  Friedman omnibus: χ²({fr.df}) = {fr.statistic:.3f}, p = {fr_p_str}")
        if fr.p_value > 0.05:
            print(f"  [!] Friedman p > 0.05: no significant omnibus effect — treat pairwise results with caution.")
    
    if max_pairs > 0:
        pair_max_abs = max(
            1e-12,
            max(
                max(
                    abs(float(result.point_diff)),
                    abs(float(result.ci_low)),
                    abs(float(result.ci_high)),
                    abs(float(result.point_diff - result.std_diff)),
                    abs(float(result.point_diff + result.std_diff)),
                )
                for result in pair_results[:max_pairs]
            ),
        )
        pair_low = -pair_max_abs
        pair_high = pair_max_abs
        print(
            f"  axis: [{pair_low:+.3f}, {pair_high:+.3f}]  "
            f"(· ±1σ, ─ CI, ● {pair_stat_label.lower()}, │ zero)"
        )
        print(
            f"  {'Pair':<{pair_col_width}s} {'Interval Plot':<{line_width}s} "
            f"{pair_stat_label:>{pair_stat_col_width}s} "
            f"{'CI Low':>{pair_ci_col_width}s} {'CI High':>{pair_ci_col_width}s} "
            f"{'r_rb':>{pair_sigma_col_width}s} "
            f"{'p (boot)':>{pair_p_boot_col_width}s} {'p (wsr)':>{pair_p_wsr_col_width}s} {'p (nem)':>{pair_p_nem_col_width}s}"
        )

    for result in pair_results[:max_pairs]:
        line = _ascii_interval_line(
            mean=float(result.point_diff),
            ci_low=float(result.ci_low),
            ci_high=float(result.ci_high),
            spread_low=float(result.point_diff - result.std_diff),
            spread_high=float(result.point_diff + result.std_diff),
            axis_low=pair_low,
            axis_high=pair_high,
            width=line_width,
        )
        pair_label = _truncate_label(
            f"{result.template_a} vs {result.template_b}",
            pair_col_width,
        )
        p_boot_str = _format_p_value(result.p_value)
        wsr_str = _format_p_value(result.wilcoxon_p)
        nem_p = bundle.pairwise.friedman.get_nemenyi_p(result.template_a, result.template_b) if bundle.pairwise.friedman is not None else None
        nem_str = _format_p_value(nem_p)
        d_val = result.rank_biserial
        d_str = f"{d_val:>{pair_sigma_col_width}.3f}"
        print(
            f"  {pair_label:<{pair_col_width}s} "
            f"{line:<{line_width}s} "
            f"{result.point_diff:+{pair_stat_col_width}.4f} "
            f"{result.ci_low:+{pair_ci_col_width}.4f} "
            f"{result.ci_high:+{pair_ci_col_width}.4f} "
            f"{d_str} "
            f"{p_boot_str:>{pair_p_boot_col_width}s} "
            f"{wsr_str:>{pair_p_wsr_col_width}s} "
            f"{nem_str:>{pair_p_nem_col_width}s}"
        )

    if max_pairs == 0:
        print("  (no pairwise comparisons)")
    elif max_pairs > 0:
        print(f"  r_rb = rank biserial correlation (effect size: small≈0.1, medium≈0.3, large≈0.5)")
        print(f"  p (boot) = bootstrap {bundle.pairwise.correction_method}-corrected; "
              f"p (wsr) = Wilcoxon signed-rank {bundle.pairwise.correction_method}-corrected; "
              f"p (nem) = Nemenyi post-hoc (Friedman-based, FWER-controlled)")
        print("  stars: * p<0.05, ** p<0.01, *** p<0.001")
        print()
        labels_sorted = [
            label
            for _, label in sorted(
                zip(bundle.rank_dist.expected_ranks, bundle.rank_dist.labels),
                key=lambda item: (float(item[0]), item[1]),
            )
        ]
        _print_critical_difference_groups(
            bundle.pairwise,
            labels_sorted=labels_sorted,
            p_source="bootstrap",
        )
    
    # Seed variance section (only when seeded data is present).
    if bundle.seed_variance is not None:
        print()
        _print_seed_variance(
            bundle.seed_variance,
            template_col_width=template_col_width,
            item_singular=item_singular,
        )

    # Token usage & Pareto analysis (only when token_usage was provided).
    if bundle.token_analysis is not None:
        print()
        _print_token_pareto_summary(bundle.token_analysis, bundle)


def _print_token_pareto_summary(
    token_analysis: TokenAnalysisResult,
    bundle: AnalysisBundle,
) -> None:
    """Print Option 2 (dual-bar table) and Option 3 (quality ladder).

    Option 2 shows a per-template table of token CIs alongside score means,
    with Pareto status markers.  Option 3 shows a quality ladder sorted by
    score for Pareto-optimal templates only, with marginal score gain and
    token cost between adjacent steps — backed by the existing pairwise CIs.
    """
    stats = token_analysis.stats
    frontier = set(token_analysis.pareto_frontier)
    dominated_by = token_analysis.dominated_by
    labels = stats.labels
    N = len(labels)

    score_means = bundle.robustness.mean          # (N,) — mean scores
    ci_pct = int(round(stats.ci * 100))
    M = bundle.benchmark.n_inputs
    has_input = stats.mean_input is not None

    col_type = "Total" if has_input else "Output"
    note = "(input + output)" if has_input else "(output tokens only; input not provided)"

    BAR_W = 8

    # Pre-format token strings to compute alignment widths.
    tok_strs = [f"{int(round(float(stats.mean_total[i]))):,}" for i in range(N)]
    ci_strs = [
        f"[{int(round(float(stats.ci_low_total[i]))):,}\u2013"
        f"{int(round(float(stats.ci_high_total[i]))):,}]"
        for i in range(N)
    ]
    max_tok_w = max(len(s) for s in tok_strs)
    max_ci_w = max(len(s) for s in ci_strs)
    tpl_w = max(16, min(28, max(len(l) for l in labels)))

    max_tokens = float(np.max(stats.mean_total))
    score_lo = float(np.min(score_means))
    score_hi = float(np.max(score_means))
    score_range = max(score_hi - score_lo, 1e-9)

    # -----------------------------------------------------------------------
    # Option 2: dual-bar table
    # -----------------------------------------------------------------------
    print(f"--- Token Usage vs. Performance [{col_type} tokens, {ci_pct}% CI] ---")
    print(f"  {note}  |  bootstrap over {M} inputs")
    print()

    hdr_tok = f"{col_type} tokens mean [{ci_pct}% CI]"
    hdr_sc = "Score mean"
    tok_col_w = max_tok_w + 1 + max_ci_w + 2 + BAR_W
    sc_col_w = 5 + 2 + BAR_W
    sep_len = 2 + tpl_w + 2 + tok_col_w + 2 + sc_col_w + 2 + 20
    print(
        f"  {'Template':<{tpl_w}s}  "
        f"{hdr_tok:<{tok_col_w}s}  "
        f"{hdr_sc:<{sc_col_w}s}  "
        f"Status"
    )
    print(f"  {'─' * sep_len}")

    for i, label in enumerate(labels):
        # Token bar — normalized to max mean total.
        mean_tok = float(stats.mean_total[i])
        filled_tok = int(round(mean_tok / max_tokens * BAR_W)) if max_tokens > 0 else 0
        filled_tok = max(0, min(filled_tok, BAR_W))
        tok_bar = "\u2588" * filled_tok + "\u2591" * (BAR_W - filled_tok)

        # Score bar — normalized to [min, max] range.
        mean_sc = float(score_means[i])
        filled_sc = int(round((mean_sc - score_lo) / score_range * BAR_W))
        filled_sc = max(0, min(filled_sc, BAR_W))
        sc_bar = "\u2588" * filled_sc + "\u2591" * (BAR_W - filled_sc)

        # Status marker.
        if label in dominated_by:
            doms = dominated_by[label]
            dom_str = _truncate_label(doms[0], 18)
            if len(doms) > 1:
                dom_str += f" +{len(doms) - 1}"
            status = f"\u00b7 ({dom_str})"
        else:
            status = "\u2605"  # ★

        tok_part = (
            f"{tok_strs[i]:>{max_tok_w}s} {ci_strs[i]:<{max_ci_w}s}  {tok_bar}"
        )
        sc_part = f"{mean_sc:.3f}  {sc_bar}"
        print(
            f"  {_truncate_label(label, tpl_w):<{tpl_w}s}  "
            f"{tok_part}  "
            f"{sc_part}  "
            f"{status}"
        )

    print(f"  {'─' * sep_len}")
    print(
        f"  \u2605 Pareto-optimal   "
        f"\u00b7 dominated by (statistically confirmed, {ci_pct}% CI)"
    )
    print()

    # -----------------------------------------------------------------------
    # Option 3: quality ladder (Pareto-optimal only, sorted by score)
    # -----------------------------------------------------------------------
    frontier_sorted = sorted(
        list(frontier),
        key=lambda lbl: float(score_means[labels.index(lbl)]),
    )
    n_excluded = N - len(frontier_sorted)

    print("--- Quality Ladder (Pareto-optimal only, sorted by score) ---")
    if n_excluded > 0:
        excluded = [l for l in labels if l not in frontier]
        ex_str = ", ".join(
            f"'{_truncate_label(l, 20)}'" for l in excluded
        )
        s = "s" if n_excluded > 1 else ""
        print(f"  {n_excluded} dominated template{s} excluded: {ex_str}")
    print()

    for rank, lbl in enumerate(frontier_sorted):
        i = labels.index(lbl)
        mean_sc = float(score_means[i])
        mean_tok = float(stats.mean_total[i])
        ci_lo = float(stats.ci_low_total[i])
        ci_hi = float(stats.ci_high_total[i])
        tok_ci_str = f"[{int(round(ci_lo)):,}\u2013{int(round(ci_hi)):,}]"
        print(
            f"  {mean_sc:.3f}  {_truncate_label(lbl, 24):<24s}  "
            f"{int(round(mean_tok)):,} tok {tok_ci_str}"
        )

        if rank < len(frontier_sorted) - 1:
            next_lbl = frontier_sorted[rank + 1]
            j = labels.index(next_lbl)

            # Marginal score gain (next − current).
            try:
                sc_r = bundle.pairwise.get(next_lbl, lbl)
                sc_diff_pp = sc_r.point_diff * 100
                sc_ci_lo_pp = sc_r.ci_low * 100
                sc_ci_hi_pp = sc_r.ci_high * 100
                sc_note = f", p={_format_p_value(sc_r.p_value)}"
            except KeyError:
                sc_diff_pp = (float(score_means[j]) - float(score_means[i])) * 100
                sc_ci_lo_pp = sc_ci_hi_pp = sc_diff_pp
                sc_note = ""

            # Marginal token cost (next − current).
            try:
                tok_r = token_analysis.pairwise.get(next_lbl, lbl)
                tok_diff = tok_r.point_diff
                tok_ci_lo = tok_r.ci_low
                tok_ci_hi = tok_r.ci_high
                tok_note = f", p={_format_p_value(tok_r.p_value)}"
            except KeyError:
                tok_diff = float(stats.mean_total[j]) - float(stats.mean_total[i])
                tok_ci_lo = tok_ci_hi = tok_diff
                tok_note = ""

            sc_part = (
                f"{sc_diff_pp:+.1f}pp "
                f"[{sc_ci_lo_pp:+.1f}\u2013{sc_ci_hi_pp:+.1f}{sc_note}]"
            )
            tok_part = (
                f"{int(round(tok_diff)):+,} tok "
                f"[{int(round(tok_ci_lo)):+,}\u2013{int(round(tok_ci_hi)):+,}{tok_note}]"
            )
            print(f"   \u2191 {sc_part}   {tok_part}")

    print()


_BLOCK_CHARS = "▁▂▃▄▅▆▇█"


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


def _print_seed_variance(
    sv: SeedVarianceResult,
    template_col_width: int = 24,
    strip_width: int = 24,
    item_singular: str = "template",
) -> None:
    """Print seed variance decomposition with per-input heat strip.

    Each bar encodes the per-cell instability contribution:
    ``per_cell_seed_var[j] / total_var``.  Because these values average to
    ``instability`` across inputs, the mean bar height directly corresponds
    to the reported instability score, while the bar *distribution* reveals
    whether noise is concentrated on a few inputs or spread uniformly.
    """
    print(f"--- Per-input Variance Across Runs (R={sv.n_runs} runs) ---")
    # Scale strip bars globally against the noisiest single cell so that
    # templates with low per-cell noise show short bars even if their
    # seed_fraction is high due to near-zero total_var.
    global_cell_max = float(sv.per_cell_seed_std.max())
    print(
        f"  key: ▁–█ = per-input noise   "
        f"(globally scaled; █ = {global_cell_max:.4f})"
    )
    num_w = 10
    print(
        f"  {item_singular.capitalize():<{template_col_width}s}  "
        f"{'Per-input noise':<{strip_width}s}  "
        f"{'seed_std':>{num_w}s}  "
        f"{'input_std':>{num_w}s}  "
        f"{'total_std':>{num_w}s}  "
        f"{'instability':>{num_w}s}  "
        f"Verdict"
    )
    for i, label in enumerate(sv.labels):
        strip = _seed_noise_strip(
            sv.per_cell_seed_std[i], global_cell_max, max_width=strip_width
        )
        instability = float(sv.instability[i])
        print(
            f"  {_truncate_label(label, template_col_width):<{template_col_width}s}  "
            f"{strip:<{strip_width}s}  "
            f"{np.sqrt(sv.seed_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.input_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.total_var[i]):>{num_w}.4f}  "
            f"{instability:>{num_w}.4f}  "
            f"{_instability_label(instability)}"
        )
    print()


def _collect_cross_model_seed_instability_rows(
    bundle: MultiModelBundle,
) -> list[tuple[str, float, float, float, str, float]]:
    """Collect sorted per-model instability rows for summary tables.

    Returns rows sorted by ascending overall instability.
    """
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

        rows.append(
            (
                model_label,
                overall_instability,
                template_instability_mean,
                template_instability_std,
                noisiest_template,
                noisiest_value,
            )
        )

    rows.sort(key=lambda row: row[1])
    return rows


def _print_cross_model_seed_instability(
    bundle: MultiModelBundle,
    *,
    rows: Optional[list[tuple[str, float, float, float, str, float]]] = None,
) -> None:
    """Print cross-model instability comparison when seed variance is available.

    Aggregates over templates and inputs so each model has one at-a-glance
    instability score (mean per-cell seed std), plus template-level spread.
    """
    if rows is None:
        rows = _collect_cross_model_seed_instability_rows(bundle)

    if len(rows) == 0:
        return

    print("\n--- Cross-Model Instability (across templates & inputs) ---")
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
        print(
            f"  {_truncate_label(model_label, model_w):<{model_w}s} "
            f"{overall_instability:>12.4f} "
            f"{template_instability_mean:>10.4f} "
            f"{template_instability_std:>9.4f} "
            f"{noisiest_desc:<24s} "
            f"{_instability_label(overall_instability)}"
        )
    print()


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
    midpoint between integers. Output looks like:

    E[Rank]               
    1.54 ▄▆▄▁──────────
    2.80 ───▂▅▇▅▂──────
    1.79 ▂▅▇▅▂─────────
    6.00 ───────────▃▆█
    4.31 ───────▂▅▇▅▂──
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
    # 1.0 when near integer (sharp), 0.0 when near half-step (soft).
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
    return min(p_values) if len(p_values) > 0 else None


def _critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: float = 0.05,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> list[list[str]]:
    """Return contiguous, maximal non-significant rank bands.

    Groups are built on rank-sorted labels (best first) and keep only
    intervals where every within-group pair has p >= alpha under
    ``p_source`` (bootstrap, Wilcoxon, or both).
    """
    if len(labels_sorted) < 2:
        return []

    n_labels = len(labels_sorted)

    def _all_pairs_nonsignificant(group_labels: list[str]) -> bool:
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                p_value = _pairwise_rank_band_p(
                    pairwise,
                    group_labels[i],
                    group_labels[j],
                    p_source=p_source,
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


def _print_critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: float = 0.05,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> None:
    """Print a short CD-style summary of statistically indistinguishable groups."""
    if len(labels_sorted) < 2:
        return

    rank_pos = {label: idx + 1 for idx, label in enumerate(labels_sorted)}

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
            f"({source_label}, α={alpha:g}): none"
        )
        return

    print(
        f"  Statistically indistinguishable rank bands "
        f"({source_label}, α={alpha:g}):"
    )
    for group in groups:
        start_rank = rank_pos[group[0]]
        end_rank = rank_pos[group[-1]]
        rank_span = f"#{start_rank}" if start_rank == end_rank else f"#{start_rank}–#{end_rank}"
        print(f"    {rank_span}: [{' ─ '.join(group)}]")


def _p_value_stars(p_value: Optional[float]) -> str:
    """Return significance stars for p-value thresholds.

    Thresholds:
    *   for p < 0.05
    **  for p < 0.01
    *** for p < 0.001
    """
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p_value(p_value: Optional[float]) -> str:
    """Format p-value with significance stars; return N/A for missing values."""
    if p_value is None:
        return "N/A"
    return f"{p_value:.4g}{_p_value_stars(p_value)}"
