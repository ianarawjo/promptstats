"""High-level comparison API for prompt templates.

Provides ``compare_prompts()`` and ``compare_models()``, simple entry
points for the most common use cases: compare prompt variants within a
single model, or compare models while accounting for prompt sensitivity.

Internally, these helpers build ``BenchmarkResult`` or
``MultiModelBenchmark`` objects from dictionaries, run the full
``analyze()`` pipeline, and return lightweight report objects that surface
the most useful numbers without requiring knowledge of the underlying data
structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .core.types import BenchmarkResult, MultiModelBenchmark
from .core.router import analyze, AnalysisBundle, MultiModelBundle, print_analysis_summary
from .core.paired import PairwiseMatrix, PairedDiffResult
from .core.resampling import bootstrap_means_1d, bca_interval_1d, resolve_resampling_method


# ---------------------------------------------------------------------------
# Shared stats/report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntityStats:
    """Descriptive and inferential statistics for one compared entity."""

    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


@dataclass
class CompareReport:
    """Unified comparison report for prompts or models."""

    labels: list[str]
    entity_stats: dict[str, EntityStats]
    pairwise_p_values: dict[tuple[str, str], dict[str, Optional[float]]]
    winners: Optional[list[str]]
    p_best: float
    pairwise: PairwiseMatrix
    full_analysis: AnalysisBundle | MultiModelBundle
    alpha: float = 0.05
    statistic: Literal["mean", "median"] = "mean"
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm"
    entity_name_singular: str = "prompt"
    entity_name_plural: str = "prompts"

    @property
    def means(self) -> dict[str, float]:
        return {label: self.entity_stats[label].mean for label in self.labels}

    @property
    def prompt_stats(self) -> dict[str, EntityStats]:
        return self.entity_stats

    @property
    def model_stats(self) -> dict[str, EntityStats]:
        return self.entity_stats

    @property
    def significant(self) -> bool:
        return self.winners is not None

    @property
    def winner(self) -> Optional[str]:
        if self.winners is None or len(self.winners) != 1:
            return None
        return self.winners[0]

    def quick_summary(self) -> str:
        best_label = self._best_label()
        pair = self._best_pair()
        diff = pair.point_diff
        ci_lo, ci_hi = pair.ci_low, pair.ci_high
        p = pair.p_value
        n = len(self.labels)
        stat_name = self.statistic
        best_stat = getattr(self.entity_stats[best_label], stat_name)
        delta_name = f"Δ{stat_name}"
        correction_text = "uncorrected" if self.correction == "none" else f"{self.correction}-corrected"

        if n == 2:
            other = [label for label in self.labels if label != best_label][0]
            if self.winners is not None:
                return (
                    f"'{best_label}' is significantly better than '{other}' "
                    f"({delta_name}={diff:+.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                    f"p={p:.4g}, {correction_text})"
                )
            return (
                f"No significant difference between '{best_label}' and '{other}' "
                f"({delta_name}={diff:+.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                f"p={p:.4g}, {correction_text})"
            )

        if self.winners is not None:
            if len(self.winners) == 1:
                winner_text = f"winner: '{self.winners[0]}'"
            else:
                winner_text = "winners: " + ", ".join(f"'{winner}'" for winner in self.winners)
            return f"Top {self.entity_name_singular} set ({winner_text})"

        return (
            f"All {self.entity_name_plural} are tied under pairwise tests; '{best_label}' leads "
            f"numerically ({stat_name}={best_stat:.3f}) (min p={p:.4g}, {correction_text})"
        )

    def summary(self) -> None:
        print_analysis_summary(self.full_analysis)

    def print(self) -> None:
        self.summary()

    def _best_label(self) -> str:
        return max(self.labels, key=lambda label: getattr(self.entity_stats[label], self.statistic))

    def get_pairwise_p_values(self, a: str, b: str) -> dict[str, Optional[float]]:
        if (a, b) in self.pairwise_p_values:
            return self.pairwise_p_values[(a, b)]
        if (b, a) in self.pairwise_p_values:
            return self.pairwise_p_values[(b, a)]
        raise KeyError(f"No pairwise p-values found for ({a}, {b}).")

    def _best_pair(self) -> PairedDiffResult:
        best = self._best_label()
        others = [label for label in self.labels if label != best]
        return min(
            (self.pairwise.get(best, other) for other in others),
            key=lambda result: result.p_value,
        )


def _compute_winners(
    labels: list[str],
    pairwise: PairwiseMatrix,
    alpha: float,
) -> Optional[list[str]]:
    """Compute top-tier winners from directed significant-better relations.

    A directed edge i→j exists when i is significantly better than j
    (correction-adjusted p < alpha and positive point difference).
    Winners are labels with zero incoming edges. If there are no edges,
    all prompts are tied and ``None`` is returned.
    """
    incoming = {label: 0 for label in labels}
    edge_count = 0

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            result = pairwise.get(a, b)
            if result.p_value < alpha:
                if result.point_diff > 0:
                    incoming[b] += 1
                    edge_count += 1
                elif result.point_diff < 0:
                    incoming[a] += 1
                    edge_count += 1

    if edge_count == 0:
        return None

    winners = [label for label in labels if incoming[label] == 0]
    return winners if winners else None


def _normalize_compare_models_scores(
    scores: dict,
    template_labels: Optional[list[str]],
) -> tuple[list[str], list[np.ndarray], Optional[list[str]]]:
    labels = list(scores.keys())
    values = list(scores.values())

    if not values:
        return labels, [], template_labels

    nested_templates = all(isinstance(value, dict) for value in values)
    if any(isinstance(value, dict) for value in values) and not nested_templates:
        raise TypeError(
            "scores values must be consistently array-like or nested dicts of template scores; "
            "do not mix both forms."
        )

    if nested_templates:
        first_templates = list(values[0].keys())
        if template_labels is None:
            resolved_template_labels = first_templates
        else:
            if len(template_labels) != len(first_templates):
                raise ValueError(
                    "template_labels length must match the number of templates (N). "
                    f"Got {len(template_labels)} labels for N={len(first_templates)}."
                )
            if set(template_labels) != set(first_templates):
                raise ValueError(
                    "For nested dict input, template_labels must match the inner template keys. "
                    f"Expected {sorted(first_templates)}, got {sorted(template_labels)}."
                )
            resolved_template_labels = list(template_labels)

        expected_template_set = set(first_templates)
        arrays: list[np.ndarray] = []

        for model_label, model_templates in scores.items():
            model_template_set = set(model_templates.keys())
            if model_template_set != expected_template_set:
                raise ValueError(
                    "All nested model dicts must contain the same template keys. "
                    f"Expected {sorted(expected_template_set)}, got {sorted(model_template_set)} "
                    f"for model '{model_label}'."
                )

            per_template_arrays: list[np.ndarray] = []
            for template_label in resolved_template_labels:
                a = np.asarray(model_templates[template_label], dtype=np.float64)
                if a.ndim not in (1, 2):
                    raise ValueError(
                        f"Score array for model '{model_label}', template '{template_label}' "
                        f"has {a.ndim} dimensions. Expected 1-D (M inputs) or "
                        "2-D (M inputs, R runs)."
                    )
                per_template_arrays.append(a)

            arrays.append(np.stack(per_template_arrays, axis=0))

        return labels, arrays, resolved_template_labels

    arrays = []
    for label, arr in scores.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim not in (1, 2):
            raise ValueError(
                f"Score array for '{label}' has {a.ndim} dimensions. "
                "Expected 1-D (M inputs) or 2-D (M inputs, R runs). "
                "For multiple templates, use nested dict form: "
                "{model: {template: array}}."
            )

        if a.ndim == 1:
            a = a[np.newaxis, :]
        else:
            a = a[np.newaxis, :, :]

        arrays.append(a)

    return labels, arrays, template_labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_prompts(
    scores: dict,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> CompareReport:
    """Compare prompt templates with bootstrapped statistical tests.

    A convenience wrapper around :func:`analyze` for the common case where
    you have one score array per prompt variant and want a quick answer to
    "is any prompt significantly better than the others?"

    Parameters
    ----------
    scores : dict[str, array-like]
        Mapping from prompt label to score array.  Each value can be:

        * **1-D** ``(M,)`` — one score per benchmark input (single run).
        * **2-D** ``(M, R)`` — R repeated runs per input.  R ≥ 3 activates
          the nested two-level bootstrap so that run-to-run stochasticity
          is propagated into confidence intervals.

        All arrays must share the same M (and R when 2-D).

    alpha : float
        Significance threshold for declaring a winner (default 0.05).
        A prompt is named winner only if it beats at least one other prompt
        with a correction-adjusted p-value < alpha.
    n_bootstrap : int
        Bootstrap resamples (default 10,000).
    correction : str
        Multiple-comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    method : str
        Bootstrap variant: ``'auto'`` (default, picks BCa for 15 ≤ M ≤ 200),
        ``'bootstrap'`` (percentile), or ``'bca'``.
    statistic : str
        Central-tendency statistic: ``'mean'`` (default) or ``'median'``.
    ci : float
        Confidence level for intervals (default 0.95).
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    CompareReport

    Examples
    --------
    Binary pass/fail, two prompts:

    >>> import promptstats as ps
    >>> report = ps.compare_prompts({
    ...     "baseline": [1, 1, 0, 1, 0],
    ...     "v2":       [1, 1, 1, 1, 0],
    ... })
    >>> print(report.quick_summary())
    >>> print(report.prompt_stats["baseline"].ci_low,
    ...       report.prompt_stats["baseline"].ci_high)

    Three-way comparison with continuous scores:

    >>> report = ps.compare_prompts({
    ...     "zero-shot":        [0.80, 0.90, 0.70, 0.85],
    ...     "few-shot":         [0.75, 0.88, 0.65, 0.80],
    ...     "chain-of-thought": [0.82, 0.91, 0.73, 0.87],
    ... })
    >>> report.summary()

    Multi-run (nested bootstrap activated when R ≥ 3):

    >>> report = ps.compare_prompts({
    ...     "baseline": [[0.80, 0.82, 0.79], [0.90, 0.88, 0.91]],
    ...     "v2":       [[0.85, 0.87, 0.84], [0.92, 0.90, 0.93]],
    ... })
    """
    if not isinstance(scores, dict):
        raise TypeError(
            "scores must be a dict mapping prompt labels to score arrays. "
            "Example: {'baseline': [0.8, 0.9, 0.7], 'v2': [0.85, 0.92, 0.71]}"
        )
    if len(scores) < 2:
        raise ValueError(
            f"compare_prompts requires at least 2 prompts; got {len(scores)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    labels = list(scores.keys())
    arrays = []
    for label, arr in scores.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim not in (1, 2):
            raise ValueError(
                f"Score array for '{label}' has {a.ndim} dimensions. "
                "Expected 1-D (one score per input) or "
                "2-D of shape (M inputs, R runs)."
            )
        arrays.append(a)

    ndims = {a.ndim for a in arrays}
    if len(ndims) > 1:
        raise ValueError(
            "All score arrays must have the same number of dimensions. "
            "Got a mix of 1-D and 2-D arrays. "
            "Use 2-D arrays for all prompts when providing multiple runs."
        )

    ndim = next(iter(ndims))
    ms = [a.shape[0] for a in arrays]
    if len(set(ms)) > 1:
        raise ValueError(
            "All score arrays must have the same number of inputs (M). "
            f"Got: {dict(zip(labels, ms))}"
        )

    if ndim == 2:
        rs = [a.shape[1] for a in arrays]
        if len(set(rs)) > 1:
            raise ValueError(
                "All 2-D score arrays must have the same number of runs (R). "
                f"Got: {dict(zip(labels, rs))}"
            )

    M = ms[0]
    scores_arr = np.stack(arrays, axis=0)  # (N, M) or (N, M, R)
    input_labels = [f"input_{i}" for i in range(M)]

    benchmark = BenchmarkResult(
        scores=scores_arr,
        template_labels=labels,
        input_labels=input_labels,
    )

    full_analysis: AnalysisBundle = analyze(  # type: ignore[assignment]
        benchmark,
        method=method,
        n_bootstrap=n_bootstrap,
        correction=correction,
        statistic=statistic,
        ci=ci,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Per-template descriptive stats and bootstrapped CIs on the configured
    # statistic.
    # Cell means (averaged over runs) are used as the per-input observations
    # for a single-level bootstrap — appropriate for estimating uncertainty
    # in each template's absolute location independently.
    # ------------------------------------------------------------------
    scores_2d = benchmark.get_2d_scores()  # (N, M)
    alpha_ci = 1.0 - ci
    resolved_method = resolve_resampling_method(method, M)

    rob = full_analysis.robustness  # RobustnessResult indexed parallel to labels

    pairwise_p_values: dict[tuple[str, str], dict[str, Optional[float]]] = {
        (a, b): {
            "p_boot": float(result.p_value),
            "p_wilcoxon": (
                float(result.wilcoxon_p)
                if result.wilcoxon_p is not None
                else None
            ),
        }
        for (a, b), result in full_analysis.pairwise.results.items()
    }

    entity_stats: dict[str, EntityStats] = {}
    for i, label in enumerate(labels):
        row = scores_2d[i]  # (M,) cell means
        point_est = float(np.nanmean(row)) if statistic == "mean" else float(np.nanmedian(row))

        boot_stats = bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)

        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(row, point_est, boot_stats, alpha_ci, statistic=statistic)
        else:
            ci_low = float(np.percentile(boot_stats, 100 * alpha_ci / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha_ci / 2)))

        entity_stats[label] = EntityStats(
            mean=float(rob.mean[i]),
            median=float(rob.median[i]),
            std=float(rob.std[i]),
            ci_low=ci_low,
            ci_high=ci_high,
        )

    # ------------------------------------------------------------------
    # Winners: top tier under pairwise significance.
    # ------------------------------------------------------------------
    best_label = max(labels, key=lambda l: getattr(entity_stats[l], statistic))
    other_labels = [l for l in labels if l != best_label]
    best_pairs = [full_analysis.pairwise.get(best_label, other) for other in other_labels]
    p_best = float(min(r.p_value for r in best_pairs))
    winners = _compute_winners(labels, full_analysis.pairwise, alpha)

    return CompareReport(
        labels=labels,
        entity_stats=entity_stats,
        pairwise_p_values=pairwise_p_values,
        winners=winners,
        p_best=p_best,
        pairwise=full_analysis.pairwise,
        full_analysis=full_analysis,
        alpha=alpha,
        statistic=statistic,
        correction=correction,
        entity_name_singular="prompt",
        entity_name_plural="prompts",
    )


def compare_models(
    scores: dict,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
    ci: float = 0.95,
    template_model_collapse: Literal["mean", "as_runs"] = "as_runs",
    template_labels: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> CompareReport:
    """Compare models while accounting for prompt-template sensitivity.

    Parameters
    ----------
        scores : dict
            Mapping from model label to scores in one of these forms:

            * ``{"model": array}`` where each array is:
                - **1-D** ``(M,)`` for a single implicit template, or
                - **2-D** ``(M, R)`` for a single implicit template with R runs.
            * ``{"model": {"template": array}}`` where each inner array is:
                - **1-D** ``(M,)``, or
                - **2-D** ``(M, R)`` with R runs.

            For the nested-dict form, all models must provide the same template
            keys. If ``template_labels`` is omitted, inner-key order from the first
            model is used.
    alpha : float
        Significance threshold for declaring winner models.
    n_bootstrap : int
        Bootstrap resamples.
    correction : str
        Multiple-comparisons correction.
    method : str
        Bootstrap variant.
    statistic : str
        Central-tendency statistic: ``'mean'`` or ``'median'``.
    ci : float
        Confidence level for intervals.
    template_model_collapse : {"mean", "as_runs"}
        Passed through to :func:`analyze` for the template-level view.
    template_labels : list[str], optional
        Prompt-template labels. If omitted, defaults to
        ``template_0 ... template_{N-1}``.
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.
    """
    if not isinstance(scores, dict):
        raise TypeError(
            "scores must be a dict mapping model labels to score arrays. "
            "Example: {'gpt-4.1': [[...], [...]], 'llama-3.3': [[...], [...]]}"
        )
    if len(scores) < 2:
        raise ValueError(
            f"compare_models requires at least 2 models; got {len(scores)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    labels, arrays, normalized_template_labels = _normalize_compare_models_scores(
        scores,
        template_labels,
    )

    ndims = {a.ndim for a in arrays}
    if len(ndims) > 1:
        raise ValueError(
            "All score arrays must have the same number of dimensions. "
            "Got a mix of 2-D and 3-D arrays."
        )

    ndim = next(iter(ndims))
    ns = [a.shape[0] for a in arrays]
    ms = [a.shape[1] for a in arrays]
    if len(set(ns)) > 1:
        raise ValueError(
            "All score arrays must have the same number of templates (N). "
            f"Got: {dict(zip(labels, ns))}"
        )
    if len(set(ms)) > 1:
        raise ValueError(
            "All score arrays must have the same number of inputs (M). "
            f"Got: {dict(zip(labels, ms))}"
        )
    if ndim == 3:
        rs = [a.shape[2] for a in arrays]
        if len(set(rs)) > 1:
            raise ValueError(
                "All 3-D score arrays must have the same number of runs (R). "
                f"Got: {dict(zip(labels, rs))}"
            )

    n_templates = ns[0]
    n_inputs = ms[0]
    resolved_template_labels = (
        normalized_template_labels
        if normalized_template_labels is not None
        else [f"template_{i}" for i in range(n_templates)]
    )
    if len(resolved_template_labels) != n_templates:
        raise ValueError(
            "template_labels length must match the number of templates (N). "
            f"Got {len(resolved_template_labels)} labels for N={n_templates}."
        )

    scores_arr = np.stack(arrays, axis=0)
    input_labels = [f"input_{i}" for i in range(n_inputs)]
    benchmark = MultiModelBenchmark(
        scores=scores_arr,
        model_labels=labels,
        template_labels=resolved_template_labels,
        input_labels=input_labels,
    )

    full_analysis = analyze(
        benchmark,
        method=method,
        n_bootstrap=n_bootstrap,
        correction=correction,
        statistic=statistic,
        ci=ci,
        rng=rng,
        template_model_collapse=template_model_collapse,
    )
    if not isinstance(full_analysis, MultiModelBundle):
        raise RuntimeError("Expected multi-model analysis bundle from analyze().")

    model_analysis = full_analysis.model_level
    scores_2d = benchmark.get_model_mean_result().get_2d_scores()  # (P, M)
    alpha_ci = 1.0 - ci
    resolved_method = resolve_resampling_method(method, n_inputs)
    rob = model_analysis.robustness

    pairwise_p_values: dict[tuple[str, str], dict[str, Optional[float]]] = {
        (a, b): {
            "p_boot": float(result.p_value),
            "p_wilcoxon": (
                float(result.wilcoxon_p)
                if result.wilcoxon_p is not None
                else None
            ),
        }
        for (a, b), result in model_analysis.pairwise.results.items()
    }

    entity_stats: dict[str, EntityStats] = {}
    for i, label in enumerate(labels):
        row = scores_2d[i]
        point_est = float(np.nanmean(row)) if statistic == "mean" else float(np.nanmedian(row))

        boot_stats = bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(row, point_est, boot_stats, alpha_ci, statistic=statistic)
        else:
            ci_low = float(np.percentile(boot_stats, 100 * alpha_ci / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha_ci / 2)))

        entity_stats[label] = EntityStats(
            mean=float(rob.mean[i]),
            median=float(rob.median[i]),
            std=float(rob.std[i]),
            ci_low=ci_low,
            ci_high=ci_high,
        )

    best_label = max(labels, key=lambda l: getattr(entity_stats[l], statistic))
    other_labels = [l for l in labels if l != best_label]
    best_pairs = [model_analysis.pairwise.get(best_label, other) for other in other_labels]
    p_best = float(min(r.p_value for r in best_pairs))
    winners = _compute_winners(labels, model_analysis.pairwise, alpha)

    return CompareReport(
        labels=labels,
        entity_stats=entity_stats,
        pairwise_p_values=pairwise_p_values,
        winners=winners,
        p_best=p_best,
        pairwise=model_analysis.pairwise,
        full_analysis=full_analysis,
        alpha=alpha,
        statistic=statistic,
        correction=correction,
        entity_name_singular="model",
        entity_name_plural="models",
    )
