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

from .types import BenchmarkResult, MultiModelBenchmark, AnalyzeMethod, CompareMethod
from .bundles import (
    BenchmarkShape,
    AnalysisBundle,
    MultiModelBundle,
    PerEvaluatorMultiModel,
    AnalysisResult,
)
from .paired import all_pairwise
from .ranking import LazyRankDistribution, bootstrap_ranks
from .variance import robustness_metrics, seed_variance_decomposition
from ..config import get_alpha_ci, resolve_auto_analyze_methods

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _resolve_p_value_method(
    p_values: bool,
    pairwise_test: str,
) -> Optional[str]:
    """Resolve the effective p-value display method for the bundle.

    Returns one of: ``None`` (suppress), ``'auto'``, ``'boot'``, ``'wsr'``,
    ``'nem'``.

    Resolution rules:
    - If ``p_values=False`` and ``pairwise_test='auto'``: suppress (``None``).
    - ``pairwise_test='bootstrap'``  → ``'boot'`` (explicit: always the
      CI-construction method's own p-value, never redirected).
    - ``pairwise_test='wilcoxon'``   → ``'wsr'`` (explicit: always Wilcoxon,
      even when Romano-Wolf is the resolved FWER correction -- an explicit
      request is never silently swapped for a different statistic; it just
      keeps whatever valid correction Wilcoxon's own p-value got, which is
      Shaffer's in that case since Romano-Wolf has no Wilcoxon-compatible
      form).
    - ``pairwise_test='nemenyi'``    → ``'nem'``
    - ``pairwise_test='auto'`` with ``p_values=True``: ``'auto'`` -- final
      resolution deferred to print time (see ``core.summary``'s p-value-
      column logic), since it depends on which FWER correction actually
      fired for this bundle (Shaffer's vs. Romano-Wolf), which in turn
      depends on N/data-kind and isn't known until :func:`~evalstats.core.paired.all_pairwise`
      has run. The default is Wilcoxon signed-ranks for *any* k >= 2 (per
      fig:fwer-decision-tree's standard workflow: Friedman omnibus first
      when requested, then Wilcoxon pairwise, then FWER-corrected as
      post-hoc), except when Romano-Wolf step-down is what actually
      resolved -- it has no Wilcoxon-compatible joint construction (see
      :func:`~evalstats.core.paired.romano_wolf_stepdown_pvalues`'s
      docstring), so its own mean-based bootstrap-t p-value is shown
      instead in that one case.
    """
    explicit = pairwise_test != "auto"
    if not p_values and not explicit:
        return None
    if pairwise_test == "bootstrap":
        return "boot"
    if pairwise_test == "wilcoxon":
        return "wsr"
    if pairwise_test == "nemenyi":
        return "nem"
    # pairwise_test == 'auto' with p_values=True -- resolved at print time.
    return "auto"


def analyze(
    result: Union[BenchmarkResult, MultiModelBenchmark],
    *,
    evaluator_mode: Literal["aggregate", "per_evaluator"] = "aggregate",
    reference: str = "grand_mean",
    method: AnalyzeMethod = "auto",
    backend: Literal["statsmodels", "pymer4"] = "statsmodels",
    ci: Optional[float] = None,
    n_bootstrap: int = 10_000,
    correction: Literal["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"] = "auto",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    template_model_collapse: Literal["mean", "as_runs"] = "as_runs",
    simultaneous_ci: bool = True,
    omnibus: bool = False,
    p_values: bool = False,
    pairwise_test: Literal["auto", "bootstrap", "wilcoxon", "nemenyi"] = "auto",
    ci_style: Literal["gradient", "line"] = "gradient",
    score_range: Optional[tuple[float, float]] = None,
    eval_type: Optional[Literal["likert", "continuous"]] = None,
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
        Reference for advantage: ``'grand_mean'`` (default) or a
        template label to compare all others against.  The grand
        reference is always the per-input mean across templates
        regardless of ``statistic``; using the per-input median would
        make the middle-ranked template's advantages identically zero
        (degeneracy when N is odd).
    method : str
        Statistical method for CIs and p-values:

        * ``'auto'`` (default) — data-adaptive, following the CI decision
          tree (fig:ci-decision-tree): for binary data, uses Bayesian paired
          for N<50 (Tango under-covers at small N on real eval data), Tango
          otherwise (multi-run path uses the ER-Tango / effective-N variant);
          marginal CIs use plain Wilson regardless of run count. For
          numeric data with a known or
          auto-detected ``[0,1]`` range (e.g. normalised accuracy, ROUGE, or
          any scale declared via ``score_range`` — a Likert scale, a
          percentage grade), uses Logit-t for both pairwise comparisons and
          marginals, regardless of run count — data is rescaled onto
          ``[0,1]`` using ``score_range`` before the logit transform, then
          mapped back. Falls back to a plain t-interval for numeric data
          outside ``[0,1]`` when no ``score_range`` is given (a loud
          ``UserWarning`` explains this and recommends passing one). See
          ``config.AUTO_ANALYZE_METHOD_TABLE`` for the full routing matrix.
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
        * ``'permutation'`` — Paired randomization test (sign-flip) for
            pairwise p-values, with bootstrap confidence intervals for effect
            sizes.
        * ``'sign_test'`` — Paired exact sign test (two-sided; ties dropped)
            for pairwise p-values, with bootstrap confidence intervals for
            effect sizes.
        * ``'lmm'`` — Linear Mixed Model.  Fits
          ``score ~ template + (1|input)`` on cell-mean scores.
          Produces Wald CIs via the fixed-effect covariance matrix.
          Prefer this when M < ~15 (bootstrap unstable) or when an
          ICC decomposition is desired.  ``AnalysisBundle.lmm_info``
          is populated with variance components and the ICC.
          Not compatible with ``statistic='median'``.
          The backend is controlled by the ``backend`` parameter.
        * ``'wilson'`` — Binary-only frequentist mode. Uses Wilson score
            intervals for point-advantage CIs and Newcombe score intervals
            (+ McNemar mid-p p-values) for pairwise comparisons.
        * ``'newcombe'`` — Binary-only frequentist mode. Alias of
            ``'wilson'`` routing in ``analyze()``: pairwise comparisons use
            Newcombe score intervals (+ McNemar mid-p p-values), while
            point-advantage CIs use Wilson score intervals.
        * ``'mj_floor'`` — Binary-only frequentist mode. Pairwise
            comparisons use the floored May & Johnson (1997) score interval
            (+ McNemar mid-p p-values), while point-advantage CIs use Wilson
            score intervals. This is what ``'auto'`` selects for binary
            pairwise comparisons.
        * ``'tango'`` — Binary-only frequentist mode. Pairwise comparisons
            use the exact Tango (1998) score interval (+ McNemar mid-p
            p-values), while point-advantage CIs use Wilson score intervals.
            Single-run data only; it has no multi-run form.
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
        Multiple comparisons correction across pairwise p-values.
        ``'auto'`` (default) follows fig:fwer-decision-tree: Shaffer's
        step-down Holm procedure for N<30 (or a lopsided binary split
        regardless of N), else Romano-Wolf bootstrap step-down. Explicit
        alternatives: ``'shaffer'``, ``'romano_wolf'``, ``'holm'``,
        ``'bonferroni'``, ``'fdr_bh'`` (FDR, not FWER control -- use when
        that's the actual target), ``'hochberg'``, or ``'none'``. See
        :func:`~evalstats.core.paired.all_pairwise`'s ``correction=``
        docstring for the full routing rationale.
    simultaneous_ci : bool
        When ``True``, pairwise CIs are simultaneous (family-wise) rather
        than marginal. Bonferroni correction is used by default for all
        methods, since it is faster, simpler, and more robust at small N
        than the studentized bootstrap max-T method (which remains
        available internally but is no longer the default).
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

    p_values : bool
        When ``True``, p-values are shown in pairwise comparison tables.
        Defaults to ``False`` (p-values suppressed).  Setting
        ``pairwise_test`` to anything other than ``'auto'`` also enables
        p-value display implicitly.
    pairwise_test : str
        Which p-value to compute for pairwise comparisons.  Only relevant
        when ``p_values=True`` or ``pairwise_test`` is set explicitly.

        * ``'auto'`` (default) — when ``omnibus=True``, uses Wilcoxon
          signed-rank (the standard Friedman post-hoc); otherwise defers
          to the display layer, which picks bootstrap p-values for
          bootstrap CI paths and Wilcoxon for LMM/other paths.
        * ``'bootstrap'`` — bootstrap p-value (from the resampling
          distribution of the test statistic).
        * ``'wilcoxon'`` — Wilcoxon signed-rank test p-value.  Can be
          combined with bootstrap CIs (statistically inconsistent, but
          permitted when explicitly requested).
        * ``'nemenyi'`` — Nemenyi post-hoc p-value.
    ci_style : {"gradient", "line"}
        Controls whether analysis pipelines compute multi-band CI payloads
        (``multi_ci``) used by gradient terminal plots. ``"gradient"``
        (default) enables these bands; ``"line"`` disables them.
    score_range : tuple[float, float], optional
        The eval metric's true ``(min, max)`` range, e.g. ``(0, 1)`` for
        normalised accuracy or ``(1, 5)`` for a Likert scale. Only used for
        numeric (non-binary) data routed to a bounds-dependent method (the
        ``'auto'`` default, or explicit ``method='logit_t'``/``'nig'``);
        ignored otherwise. Declaring this explicitly is strongly
        recommended for any metric whose natural range isn't already
        exactly ``[0, 1]``, since evalstats has no reliable way to infer
        it on its own.
    eval_type : {"likert", "continuous"}, optional
        Only used with ``method='auto'`` and a known/declared
        ``score_range``. Distinguishes discrete/ordinal data (a Likert
        scale, an integer percentage grade) from genuinely continuous
        data within the same bounded range. When omitted (default),
        evalstats auto-detects discreteness from the data's own
        quantization grid and emits a ``UserWarning`` if it switches to
        the Likert treatment -- pass this explicitly to silence that
        warning either way. This changes every pairwise-comparison CI
        (NIG instead of logit-t) -- single-run, seeded/multi-run, and the
        k>=3 simultaneous-CI construction alike -- see
        ``config.AUTO_ANALYZE_METHOD_TABLE``'s "likert" row for the
        validation. Marginal/robustness CIs still use logit-t for likert
        data, pending their own dedicated validation.

        When omitted, evalstats always prints a ``UserWarning`` announcing
        what it assumed and which method it picked as a result:

        * If every score already lies in ``[0, 1]``, that range is used
          exactly (accuracy, ROUGE, similarity scores, etc. all satisfy
          this) and ``method='logit_t'`` is selected — but the warning still
          fires, since evalstats inferred rather than was told this.
        * Otherwise (e.g. a Likert scale, a percentage grade, or any other
          numeric metric outside ``[0, 1]``), evalstats does **not** guess
          the range from the observed sample's min/max — that's an
          unreliable substitute for the metric's true theoretical bounds
          (e.g. a 1-5 Likert scale sampled only between 2 and 4). Instead it
          falls back to ``method='t_interval'`` (bounds-agnostic) and warns,
          recommending an explicit ``score_range`` to get the
          better-calibrated logit-t method instead.

        Binary (0/1) data never triggers any of this — it's detected and
        routed to Wilson/Tango/Bayesian-paired before ``score_range``
        resolution is even considered.

        Passing an explicit ``score_range`` whose data falls outside it
        raises ``ValueError`` rather than silently clipping or ignoring the
        violation.

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
    # Normalize once at the funnel: callers may pass an int seed, None, or a
    # Generator, and the engines below this point are inconsistent about which
    # they accept (mixed_effects and parts of resampling assume a Generator).
    # compare() now defaults rng to an int seed, so this is the single place
    # that has to turn it into something every engine can use.
    rng = np.random.default_rng(rng)
    
    if ci is None:
        ci = 1.0 - get_alpha_ci()

    if statistic not in {"mean", "median"}:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Expected 'mean' or 'median'."
        )
    if template_model_collapse not in {"mean", "as_runs"}:
        raise ValueError(
            f"Unknown template_model_collapse '{template_model_collapse}'. "
            "Expected 'mean' or 'as_runs'."
        )
    if ci_style not in {"gradient", "line"}:
        raise ValueError(
            f"Unknown ci_style '{ci_style}'. Expected 'gradient' or 'line'."
        )

    include_multi_ci = ci_style == "gradient"

    if method not in {"lmm", "bayes_bootstrap", "smooth_bootstrap", "auto", "bayes_binary", "wilson", "mj_floor", "newcombe", "tango", "bonett_price", "permutation", "sign_test", "t_interval", "logit_t", "nig"} and result.n_inputs < 15:
        warnings.warn(
            f"Only M={result.n_inputs} benchmark input(s) detected. "
            "Bootstrap confidence intervals are unreliable with fewer than ~15 inputs. "
            "Consider using method='bayes_bootstrap', method='smooth_bootstrap', or method='lmm' "
            "for more stable inference with small samples.",
            UserWarning,
            stacklevel=2,
        )

    resolved_p_value_method = _resolve_p_value_method(p_values, pairwise_test)

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
        simultaneous_ci=simultaneous_ci,
        omnibus=omnibus,
        p_value_method=resolved_p_value_method,
        include_multi_ci=include_multi_ci,
        score_range=score_range,
        eval_type=eval_type,
    )

    # ------------------------------------------------------------------
    # Multi-model path
    # ------------------------------------------------------------------
    if isinstance(result, MultiModelBenchmark):
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
    ci: Optional[float] = None,
    correction: Literal["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"] = "fdr_bh",
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
        ``'fdr_bh'`` (default), ``'holm'``, ``'bonferroni'``, or ``'none'``.

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
        All standard fields (``pairwise``,
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
    >>> import evalstats as es
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
    >>> bundle = es.analyze_factorial(data, factors=["chunker", "retrieval"])
    >>> es.print_analysis_summary(bundle)
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
    
    if ci is None:
        ci = 1.0 - get_alpha_ci()

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

def resolve_auto_robustness_method(
    run_scores: np.ndarray,
    *,
    score_range: Optional[tuple[float, float]] = None,
    eval_type: Optional[Literal["likert", "continuous"]] = None,
    stacklevel: int = 2,
) -> tuple[str, str, Optional[tuple[float, float]], str]:
    """Auto-detect data kind (binary / likert / bounded_01 / unbounded) and
    resolve it to concrete (pairwise_method, robustness_method,
    resolved_score_range, data_kind).

    This is the exact "method='auto'" routing logic ``analyze()``/``compare()``
    use internally, factored out so the quick-primitive functions
    (``mean_ci``/``summarize`` in ``evalstats.quick``) can reuse it directly
    rather than re-deriving calibration choices in a second place that could
    silently drift out of sync with ``compare()``'s.

    Parameters
    ----------
    run_scores : np.ndarray
        Shape ``(N, M)`` or ``(N, M, R)``. Only the shape and values matter
        here (dtype/range/binary-ness detection and R for seeded routing) --
        not which entity is which.
    score_range : (float, float), optional
        Explicit ``[lo, hi]`` bounds, forwarded to :func:`resolve_score_bounds`.
    eval_type : {"likert", "continuous"}, optional
        Hint disambiguating discrete/ordinal (Likert-style) data from
        continuous bounded data when both look the same from the raw
        values alone. When omitted, discrete/ordinal data is auto-detected
        from its own quantization grid (see :func:`detect_quantization_step`).
        Ignored (with a warning) for binary data, which always uses the
        binary methods regardless of this hint.
    stacklevel : int
        Forwarded to any ``UserWarning`` raised here, so it points at the
        caller's caller appropriately regardless of how many wrapper frames
        sit between the actual user call and this function.

    Returns
    -------
    tuple[str, str, tuple[float, float] or None, str]
        ``(pairwise_method, robustness_method, resolved_score_range, data_kind)``.
    """
    from .resampling import binary_routing_applies, resolve_score_bounds, detect_quantization_step

    if run_scores.ndim == 3:
        R = run_scores.shape[2]
        N = run_scores.shape[1]
    else:
        R = 1
        N = run_scores.shape[1]

    if eval_type not in (None, "likert", "continuous"):
        raise ValueError(f"eval_type must be 'likert', 'continuous', or None, got {eval_type!r}")

    resolved_score_range: Optional[tuple[float, float]] = None
    # An explicitly passed score_range wider than [0, 1] overrides binary
    # auto-detection (and says so) -- see binary_routing_applies.
    if binary_routing_applies(run_scores, score_range, stacklevel=stacklevel + 1):
        data_kind = "binary"
        if eval_type is not None:
            warnings.warn(
                f"eval_type={eval_type!r} was given, but the data was "
                "auto-detected as binary (0/1) -- binary data always uses "
                "the binary methods regardless of eval_type, so this hint "
                "was ignored.",
                UserWarning,
                stacklevel=stacklevel,
            )
    else:
        # resolve_score_bounds returns a [lo, hi] range (with a
        # UserWarning if it had to auto-detect [0, 1] rather than being
        # told explicitly) when one can be reliably established, or None
        # when the data falls outside [0, 1] and no score_range was
        # given -- there's no safe way to infer a metric's true bounds
        # from an arbitrary numeric sample's own min/max. In the None
        # case, auto silently downgrades to the bounds-agnostic
        # "unbounded" (t_interval) row below, but says so loudly.
        resolved_score_range = resolve_score_bounds(run_scores, score_range, stacklevel=stacklevel + 1)
        if resolved_score_range is not None:
            if eval_type == "likert":
                data_kind = "likert"
            elif eval_type == "continuous":
                data_kind = "bounded_01"
            else:
                # No explicit hint: auto-detect discrete/ordinal (Likert-
                # style) data from its own quantization grid rather than
                # assuming continuous -- see detect_quantization_step's
                # docstring and config.AUTO_ANALYZE_METHOD_TABLE's
                # "likert" row for why this matters (NIG vs logit-t).
                step = detect_quantization_step(run_scores)
                if step is not None:
                    data_kind = "likert"
                    warnings.warn(
                        f"Bounded numeric evaluation data was auto-detected "
                        f"as discrete/ordinal (grid step={step:g} within "
                        f"range {resolved_score_range}). For pairwise "
                        "comparisons (single-run and multi-run alike), "
                        "evalstats uses NIG (validated as better-calibrated "
                        "than logit-t there for this kind of data); "
                        "marginal/robustness CIs on this data still use "
                        "logit-t, the same as continuous data, pending "
                        "their own validation -- see "
                        "config.AUTO_ANALYZE_METHOD_TABLE's 'likert' row. "
                        "Pass eval_type='likert' explicitly to silence this "
                        "warning, or eval_type='continuous' if this "
                        "discreteness is coincidental (e.g. a metric that "
                        "happens to only take a few values in your sample).",
                        UserWarning,
                        stacklevel=stacklevel,
                    )
                else:
                    data_kind = "bounded_01"
        else:
            data_kind = "unbounded"
            # Direct warn() call, one frame shallower than the
            # resolve_score_bounds() delegation above (no extra frame in
            # between) -- stacklevel here, not stacklevel + 1.
            warnings.warn(
                "Numeric evaluation data outside [0, 1] was auto-detected "
                "with no explicit score_range, so evalstats is using "
                "method='t_interval' (a bounds-agnostic default) rather "
                "than the better-calibrated logit-t/NIG methods. If you "
                "know this eval metric's true (min, max) range, pass it "
                "explicitly, e.g. score_range=(1, 5) for a Likert scale "
                "or score_range=(0, 100) for a percentage grade.",
                UserWarning,
                stacklevel=stacklevel,
            )
    # See config.AUTO_ANALYZE_METHOD_TABLE for the full auto-routing matrix
    # (which method is chosen for which data kind / N / seeded combination).
    pairwise_method, robustness_method = resolve_auto_analyze_methods(
        data_kind, N, seeded=R >= 3,
    )
    return pairwise_method, robustness_method, resolved_score_range, data_kind


def _analyze_single(
    result: BenchmarkResult,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: AnalyzeMethod,
    backend: Literal["statsmodels", "pymer4"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    simultaneous_ci: bool = True,
    omnibus: bool = False,
    p_value_method: Optional[str] = None,
    include_multi_ci: bool = True,
    score_range: Optional[tuple[float, float]] = None,
    eval_type: Optional[Literal["likert", "continuous"]] = None,
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
        pairwise, rank_dist, robustness, seed_var, lmm_result = lmm_analyze(
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
        # Recompute robustness with marginal per-entity CIs (overrides the one
        # inside lmm_analyze which does not compute them).
        scores_2d = result.get_2d_scores()
        robustness = robustness_metrics(
            scores_2d,
            robustness.labels,
            failure_threshold=failure_threshold,
            n_bootstrap=n_bootstrap,
            rng=rng,
            alpha=1.0 - ci,
            statistic="mean",
            marginal_method="smooth_bootstrap",
            multi_ci=include_multi_ci,
        )
        if isinstance(lmm_result, FactorialLMMInfo):
            return AnalysisBundle(
                benchmark=result,
                shape=shape,
                pairwise=pairwise,
                robustness=robustness,
                rank_dist=rank_dist,
                seed_variance=seed_var,
                factorial_lmm_info=lmm_result,
                resolved_method="lmm",
                resolved_ci_method="lmm",
                p_value_method=p_value_method,
            )
        return AnalysisBundle(
            benchmark=result,
            shape=shape,
            pairwise=pairwise,
            robustness=robustness,
            rank_dist=rank_dist,
            seed_variance=seed_var,
            lmm_info=lmm_result,
            resolved_method="lmm",
            resolved_ci_method="lmm",
            p_value_method=p_value_method,
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

    pairwise_method = method
    robustness_method = method
    resolved_score_range: Optional[tuple[float, float]] = None
    # Only the "auto" branch resolves a data kind; stays None otherwise so
    # the bundle records "no resolution happened" rather than a guess.
    data_kind: Optional[str] = None
    if method == "auto":
        pairwise_method, robustness_method, resolved_score_range, data_kind = resolve_auto_robustness_method(
            run_scores, score_range=score_range, eval_type=eval_type, stacklevel=2,
        )
    elif method == "bayes_binary":
        from .resampling import is_binary_scores
        if not is_binary_scores(run_scores):
            raise ValueError(
                "method='bayes_binary' requires binary (0/1) data, but the "
                "scores array contains non-binary values. Use is_binary_scores() "
                "to check before calling, or choose a different method."
            )
        # Single-sample marginal CIs use Wilson; pairwise uses the Bayesian model.
        pairwise_method = "bayes_binary"
        robustness_method = "wilson"
    elif method in {"wilson", "newcombe", "tango", "mj_floor", "bonett_price"}:
        from .resampling import is_binary_scores
        if not is_binary_scores(run_scores):
            raise ValueError(
                f"method='{method}' requires binary (0/1) data, but the "
                "scores array contains non-binary values. Use is_binary_scores() "
                "to check before calling, or choose a different method."
            )
        if method in ("tango", "mj_floor", "bonett_price"):
            pairwise_method = method
        else:
            # In analyze(), explicit frequentist binary methods route to:
            #   - pairwise Newcombe + McNemar mid-p p-values
            #   - single-sample marginal Wilson score CIs
            pairwise_method = "newcombe"
        robustness_method = "wilson"
    elif method == "sign_test":
        pairwise_method = "sign_test"
        robustness_method = "smooth_bootstrap"
    elif method == "logit_t":
        from .resampling import resolve_score_bounds
        resolved_score_range = resolve_score_bounds(run_scores, score_range, stacklevel=2)
        if resolved_score_range is None:
            raise ValueError(
                "method='logit_t' requires data with an inferable [lo, hi] "
                "range, but the scores fall outside [0, 1] and no "
                "score_range was given. Pass score_range=(lo, hi) explicitly "
                "(e.g. score_range=(1, 5) for a Likert scale), or use a "
                "different method (e.g. method='t_interval')."
            )
    elif method == "nig":
        from .resampling import resolve_score_bounds
        resolved_score_range = resolve_score_bounds(run_scores, score_range, stacklevel=2)
        if resolved_score_range is None:
            raise ValueError(
                "method='nig' requires data with an inferable [lo, hi] "
                "range, but the scores fall outside [0, 1] and no "
                "score_range was given. Pass score_range=(lo, hi) explicitly "
                "(e.g. score_range=(1, 5) for a Likert scale), or use a "
                "different method (e.g. method='t_interval')."
            )

    # eval_type resolved for the simultaneous-CI widening formula: reuse
    # the "auto" branch's already-made data_kind decision so it isn't
    # independently re-detected (and re-warned about) inside all_pairwise
    # -> _simultaneous_cis_router; for an explicit (non-"auto") method,
    # just pass through whatever eval_type the caller gave (possibly None,
    # in which case _simultaneous_cis_router does its own detection).
    if method == "auto":
        resolved_eval_type = (
            "likert" if data_kind == "likert"
            else "continuous" if data_kind == "bounded_01"
            else None
        )
    else:
        resolved_eval_type = eval_type

    pairwise = all_pairwise(
        run_scores, labels,
        method=pairwise_method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng, statistic=statistic,
        simultaneous_ci=simultaneous_ci, omnibus=omnibus,
        multi_ci=include_multi_ci, score_range=resolved_score_range,
        eval_type=resolved_eval_type,
    )
    robustness = robustness_metrics(
        run_scores, labels,
        failure_threshold=failure_threshold,
        n_bootstrap=n_bootstrap,
        rng=rng,
        alpha=1.0 - ci,
        statistic=statistic,
        marginal_method=robustness_method,
        multi_ci=include_multi_ci,
        score_range=resolved_score_range,
    )
    # Deferred: nothing here computes the rank bootstrap unless a caller
    # actually reads rank_probs/expected_ranks/p_best. See
    # LazyRankDistribution -- .labels/.n_bootstrap stay free.
    rank_dist = LazyRankDistribution(
        labels, n_bootstrap,
        lambda _rng: bootstrap_ranks(
            run_scores, labels,
            n_bootstrap=n_bootstrap, rng=_rng, statistic=statistic,
        ),
        rng=rng,
    )

    seed_var = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    return AnalysisBundle(
        benchmark=result,
        shape=shape,
        pairwise=pairwise,
        robustness=robustness,
        rank_dist=rank_dist,
        seed_variance=seed_var,
        resolved_method=pairwise_method,
        resolved_ci_method=robustness_method,
        resolved_score_range=resolved_score_range,
        resolved_data_kind=data_kind,
        p_value_method=p_value_method,
    )


def _analyze_multi_model(
    result: MultiModelBenchmark,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: CompareMethod,
    backend: Literal["statsmodels", "pymer4"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    template_model_collapse: Literal["mean", "as_runs"] = "as_runs",
    simultaneous_ci: bool = True,
    omnibus: bool = False,
    p_value_method: Optional[str] = None,
    include_multi_ci: bool = True,
    score_range: Optional[tuple[float, float]] = None,
    eval_type: Optional[Literal["likert", "continuous"]] = None,
) -> MultiModelBundle:
    from .resampling import is_binary_scores

    fallback_binary_methods = {"wilson", "newcombe", "tango", "mj_floor", "bonett_price"}

    def _effective_method(sub_result: BenchmarkResult) -> CompareMethod:
        """Fallback only for frequentist binary methods on auxiliary non-binary views."""
        if method in fallback_binary_methods and not is_binary_scores(sub_result.get_run_scores()):
            return "auto"
        return method

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
        simultaneous_ci=simultaneous_ci,
        omnibus=omnibus,
        p_value_method=p_value_method,
        include_multi_ci=include_multi_ci,
        score_range=score_range,
        eval_type=eval_type,
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
        **{**kwargs, "method": _effective_method(template_mean_result)},
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
