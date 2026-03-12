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
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from .types import BenchmarkResult, MultiModelBenchmark
from .bundles import (
    BenchmarkShape,
    AnalysisBundle,
    MultiModelBundle,
    PerEvaluatorSingleModel,
    PerEvaluatorMultiModel,
    AnalysisResult,
)
from .paired import all_pairwise
from .ranking import bootstrap_ranks, bootstrap_point_advantage
from .variance import robustness_metrics, seed_variance_decomposition
from .tokens import TokenUsage, analyze_tokens

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    result: Union[BenchmarkResult, MultiModelBenchmark],
    *,
    token_usage: Optional[TokenUsage] = None,
    evaluator_mode: Literal["aggregate", "per_evaluator"] = "aggregate",
    reference: str = "grand_mean",
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "lmm"] = "auto",
    backend: Literal["statsmodels", "pymer4"] = "statsmodels",
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

                * ``'auto'`` (default) — smooth bootstrap.
        * ``'bootstrap'`` — percentile bootstrap.
        * ``'bca'`` — bias-corrected and accelerated bootstrap.
        * ``'bayes_bootstrap'`` — Bayesian bootstrap (Banks 1988).
          Uses Dirichlet(1,...,1) weights instead of multinomial resampling.
          Provides smoother CI coverage for small sample sizes (M < 15)
          compared to the standard percentile bootstrap.
        * ``'smooth_bootstrap'`` — Smoothed bootstrap via Gaussian KDE
          (Scott's rule bandwidth).  Resamples observations with replacement
          and adds Gaussian noise, smoothing the discrete empirical
          distribution.  May improve coverage for continuous data.
        * ``'lmm'`` — Linear Mixed Model.  Fits
          ``score ~ template + (1|input)`` on cell-mean scores.
          Produces Wald CIs via the fixed-effect covariance matrix.
          Prefer this when M < ~15 (bootstrap unstable) or when an
          ICC decomposition is desired.  ``AnalysisBundle.lmm_info``
          is populated with variance components and the ICC.
          Not compatible with ``statistic='median'``.
          The backend is controlled by the ``backend`` parameter.
    backend : str
        LMM fitting backend (only used when ``method='lmm'``):
        ``'statsmodels'`` (default, pure Python, no R required) or
        ``'pymer4'`` (wraps R/lme4, requires R with lme4 and emmeans).
        Ignored for bootstrap methods.
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
        If ``method='lmm'`` and the selected backend is not installed.
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

    if method not in {"lmm", "bayes_bootstrap", "smooth_bootstrap", "auto"} and result.n_inputs < 15:
        warnings.warn(
            f"Only M={result.n_inputs} benchmark input(s) detected. "
            "Bootstrap confidence intervals are unreliable with fewer than ~15 inputs. "
            "Consider using method='bayes_bootstrap', method='smooth_bootstrap', or method='lmm' "
            "for more stable inference with small samples.",
            UserWarning,
            stacklevel=2,
        )

    kwargs = dict(
        reference=reference,
        method=method,
        backend=backend,
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
# Factorial convenience entry point
# ---------------------------------------------------------------------------

def analyze_factorial(
    data: pd.DataFrame,
    factors: list[str],
    random_effect: str = "input_id",
    score_col: str = "score",
    *,
    run_col: Optional[str] = None,
    backend: Literal["statsmodels", "pymer4"] = "statsmodels",
    ci: float = 0.95,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    reference: str = "grand_mean",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    n_sim: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> AnalysisBundle:
    """Run factorial LMM analysis on a long-form DataFrame.

    Fits the mixed model::

        score ~ C(F1) * C(F2) * ... + (1 | random_effect)

    where ``F1, F2, ...`` are the factor columns specified in *factors*.
    The random effect (e.g. question ID, input document) absorbs
    between-input variation, producing cleaner estimates of each factor
    combination's performance and their main effects / interactions.

    This is a convenience wrapper around :func:`analyze` for two scenarios:

    * **Post-hoc tagged pipelines** — e.g. a RAG experiment where each
      output row records the ``chunker`` and ``retrieval_method`` used.
    * **Designed factorial experiments** — e.g. prompt templates that vary
      ``persona`` and ``few_shots``; see also ``BenchmarkResult`` with
      ``template_factors`` for the array-based path.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form DataFrame.  Must contain:

        * One column per factor name in *factors*.
        * *random_effect* column — unique identifier for each benchmark
          input (e.g. question ID).  At least 2 distinct values required.
        * *score_col* column — numeric evaluation score.

        Multiple rows with the same ``(random_effect, factor_combo)`` key
        are averaged (cell means) before fitting the LMM.

    factors : list[str]
        Column names of the fixed-effect factors.  Each must be a valid
        Python identifier and have at least 2 unique levels in *data*.

    random_effect : str
        Column that identifies benchmark inputs (default ``"input_id"``).

    score_col : str
        Column containing the numeric score (default ``"score"``).

    run_col : str, optional
        Column that identifies repeated runs / seeds (e.g. ``"seed"`` or
        ``"run"``).  When provided, each unique value in this column becomes
        one run slice in the underlying ``BenchmarkResult``, producing a
        3-D scores array ``(N_templates, M_inputs, R_runs)``.  This
        propagates seed variance into bootstrap confidence intervals and
        populates ``bundle.seed_variance``.  When *None* (default), multiple
        rows with the same ``(random_effect, factor_combo)`` key are averaged
        into a single cell mean, matching the previous behaviour.

    backend : str
        LMM fitting backend for the factorial analysis:
        ``'statsmodels'`` (default) or ``'pymer4'``.

    ci : float
        Confidence level for Wald intervals (default 0.95).

    correction : str
        Multiple-comparisons correction for pairwise tests:
        ``'holm'`` (default), ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.

    reference : str
        Reference for mean advantage: ``'grand_mean'`` (default) or the
        label of a specific factor-combination cell (e.g. ``'fixed_512|bm25'``
        when factors are ``['chunker', 'retrieval']``).

    spread_percentiles : tuple[float, float]
        Percentile bands for the point-advantage spread (default ``(10, 90)``).

    failure_threshold : float, optional
        Score threshold below which an observation counts as a failure
        for robustness metrics.

    n_sim : int
        Monte Carlo simulations for the rank distribution (default 10 000).

    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    AnalysisBundle
        All standard fields (``pairwise``, ``point_advantage``,
        ``robustness``, ``rank_dist``) are populated via the LMM path.
        ``bundle.factorial_lmm_info`` additionally contains:

        * ``factor_tests`` — Wald χ² tests per main effect and interaction.
        * ``marginal_means`` — estimated marginal means per factor.
        * ``icc``, ``sigma_input``, ``sigma_resid`` — variance components.

    Raises
    ------
    TypeError
        If *data* is not a :class:`pandas.DataFrame`.
    ValueError
        If required columns are missing, factor names are not valid Python
        identifiers, or any factor has fewer than 2 unique levels.

    Examples
    --------
    RAG pipeline with two factors (chunker × retrieval method):

    >>> import pandas as pd
    >>> import promptstats as ps
    >>> data = pd.DataFrame([
    ...     {"input_id": "q1", "chunker": "fixed_512", "retrieval": "bm25",  "score": 0.72},
    ...     {"input_id": "q1", "chunker": "fixed_512", "retrieval": "dense", "score": 0.85},
    ...     {"input_id": "q1", "chunker": "semantic",  "retrieval": "bm25",  "score": 0.78},
    ...     {"input_id": "q1", "chunker": "semantic",  "retrieval": "dense", "score": 0.91},
    ...     {"input_id": "q2", "chunker": "fixed_512", "retrieval": "bm25",  "score": 0.61},
    ...     {"input_id": "q2", "chunker": "fixed_512", "retrieval": "dense", "score": 0.74},
    ...     {"input_id": "q2", "chunker": "semantic",  "retrieval": "bm25",  "score": 0.65},
    ...     {"input_id": "q2", "chunker": "semantic",  "retrieval": "dense", "score": 0.82},
    ... ])
    >>> bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    >>> ps.print_analysis_summary(bundle)
    """
    import pandas as pd

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame; got {type(data).__name__}."
        )
    if not factors:
        raise ValueError("factors must be a non-empty list of column names.")

    required_cols = [*factors, random_effect, score_col]
    if run_col is not None:
        required_cols.append(run_col)
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}."
        )
    for factor in factors:
        if not str(factor).isidentifier():
            raise ValueError(
                f"Factor name '{factor}' is not a valid Python identifier. "
                "Rename it (e.g., replace spaces with underscores) so that "
                "it can be used in model formulas."
            )
        n_unique = data[factor].nunique(dropna=True)
        if n_unique < 2:
            raise ValueError(
                f"Factor '{factor}' has {n_unique} unique level(s). "
                "Each factor must have at least 2 distinct levels."
            )
    n_inputs = data[random_effect].nunique(dropna=True)
    if n_inputs < 2:
        raise ValueError(
            f"random_effect column '{random_effect}' has {n_inputs} unique value(s). "
            "At least 2 distinct inputs are required to fit the random intercept."
        )

    # ------------------------------------------------------------------
    # Build unique factor-combination → template label mapping
    # ------------------------------------------------------------------
    _SEP = "|"

    combos = (
        data[factors]
        .drop_duplicates()
        .sort_values(factors)
        .reset_index(drop=True)
    )
    template_labels: list[str] = [
        _SEP.join(str(row[f]) for f in factors)
        for _, row in combos.iterrows()
    ]
    if len(set(template_labels)) < len(template_labels):
        raise ValueError(
            "Some factor-level combinations produce identical template labels "
            f"when joined with the separator '{_SEP}'. Ensure factor values "
            f"do not contain '{_SEP}'."
        )

    template_factors_df = combos.copy()

    # ------------------------------------------------------------------
    # Normalise input IDs to strings, pivot to scores array
    # ------------------------------------------------------------------
    data_work = data.copy()
    data_work["_ps_input"] = data_work[random_effect].astype(str)
    data_work["_ps_template"] = data_work[factors].apply(
        lambda row: _SEP.join(str(row[f]) for f in factors), axis=1
    )

    input_labels: list[str] = sorted(
        data_work["_ps_input"].dropna().unique().tolist()
    )

    if run_col is not None:
        # Build a 3-D array: (N_templates, M_inputs, R_runs)
        run_labels: list[str] = sorted(
            data_work[run_col].dropna().astype(str).unique().tolist()
        )
        data_work["_ps_run"] = data_work[run_col].astype(str)
        slices = []
        for run in run_labels:
            run_df = data_work[data_work["_ps_run"] == run]
            pivot = run_df.pivot_table(
                index="_ps_input",
                columns="_ps_template",
                values=score_col,
                aggfunc="mean",
                observed=True,
            )
            pivot = pivot.reindex(index=input_labels, columns=template_labels)
            slices.append(pivot.to_numpy().T)  # (N_templates, M_inputs)
        scores_array = np.stack(slices, axis=2)  # (N_templates, M_inputs, R_runs)
    else:
        pivot = data_work.pivot_table(
            index="_ps_input",
            columns="_ps_template",
            values=score_col,
            aggfunc="mean",
            observed=True,
        )
        pivot = pivot.reindex(index=input_labels, columns=template_labels)
        scores_array = pivot.to_numpy().T  # (N_templates, M_inputs)

    # ------------------------------------------------------------------
    # Build BenchmarkResult and run the standard LMM analysis pipeline
    # ------------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    from .types import BenchmarkResult as _BR
    benchmark = _BR(
        scores=scores_array,
        template_labels=template_labels,
        input_labels=input_labels,
        template_factors=template_factors_df,
    )

    return analyze(  # type: ignore[return-value]
        benchmark,
        method="lmm",
        backend=backend,
        ci=ci,
        n_bootstrap=n_sim,
        correction=correction,
        reference=reference,
        spread_percentiles=spread_percentiles,
        failure_threshold=failure_threshold,
        rng=rng,
        statistic="mean",
    )


# ---------------------------------------------------------------------------
# Internal analysis runners
# ---------------------------------------------------------------------------

def _analyze_single(
    result: BenchmarkResult,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "lmm"],
    backend: Literal["statsmodels", "pymer4"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
) -> AnalysisBundle:
    # ------------------------------------------------------------------
    # LMM path — fit score ~ template + (1|input)
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
        from .mixed_effects import lmm_analyze, FactorialLMMInfo
        pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_result = lmm_analyze(
            result,
            backend=backend,
            reference=reference,
            ci=ci,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            n_sim=n_bootstrap,
            rng=rng,
        )
        if isinstance(lmm_result, FactorialLMMInfo):
            return AnalysisBundle(
                benchmark=result,
                shape=shape,
                pairwise=pairwise,
                point_advantage=mean_adv,
                robustness=robustness,
                rank_dist=rank_dist,
                seed_variance=seed_var,
                factorial_lmm_info=lmm_result,
            )
        return AnalysisBundle(
            benchmark=result,
            shape=shape,
            pairwise=pairwise,
            point_advantage=mean_adv,
            robustness=robustness,
            rank_dist=rank_dist,
            seed_variance=seed_var,
            lmm_info=lmm_result,
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

    # Auto-detect binary (0/1) evaluation data when method='auto'.
    # Switch to Newcombe score intervals for pairwise comparisons and
    # Wilson score intervals for single-sample advantage CIs.
    pairwise_method = method
    advantage_method = method
    if method == "auto":
        from .resampling import is_binary_scores
        if is_binary_scores(run_scores):
            pairwise_method = "newcombe"
            advantage_method = "wilson"

    pairwise = all_pairwise(
        run_scores, labels,
        method=pairwise_method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng, statistic=statistic,
    )
    mean_adv = bootstrap_point_advantage(
        run_scores, labels,
        reference=reference,
        method=advantage_method, ci=ci, n_bootstrap=n_bootstrap,
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
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto"],
    backend: Literal["statsmodels", "pymer4"],
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
        backend=backend,
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
        n_input_vars = (
            len(result.input_labels[0])
            if result.input_labels and isinstance(result.input_labels[0], tuple)
            else 1
        )
        return BenchmarkShape(
            n_models=result.n_models,
            n_prompts=result.n_templates,
            n_input_vars=n_input_vars,
            n_evaluators=result.n_evaluators,
            n_runs=result.n_runs,
        )

    # BenchmarkResult
    n_input_vars = (
        len(result.input_labels[0])
        if result.input_labels and isinstance(result.input_labels[0], tuple)
        else 1
    )
    return BenchmarkShape(
        n_models=1,
        n_prompts=result.n_templates,
        n_input_vars=n_input_vars,
        n_evaluators=result.n_evaluators,
        n_runs=result.n_runs,
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
