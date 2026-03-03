"""Token usage analysis: per-template CI estimates and Pareto frontier detection.

All statistical methods reuse the same bootstrap/BCa machinery as the score
analysis so that "dominated" labels are only applied when significance is
confirmed in both the score and token dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .paired import PairwiseMatrix, all_pairwise
from .resampling import bootstrap_ci_1d, resolve_resampling_method


@dataclass
class TokenUsage:
    """Token usage per API call, aligned to benchmark template and input labels.

    Parameters
    ----------
    output_tokens : np.ndarray
        Completion/output token counts.  Shape ``(N, M)`` for one count per
        template × input, or ``(N, M, R)`` when multiple runs are available.
        N must equal ``len(template_labels)``; M must equal
        ``len(input_labels)``.
    template_labels : list[str]
        Must match the ``template_labels`` of the ``BenchmarkResult`` passed
        to ``analyze()``.
    input_labels : list[str]
        Must match the ``input_labels`` of the ``BenchmarkResult`` passed to
        ``analyze()``.
    input_tokens : np.ndarray, optional
        Prompt/input token counts of shape ``(N, M)``; the same value for
        every run of the same template × input pair.  When provided, CIs are
        reported on input, output, and total token counts separately.  When
        absent, only output tokens are analysed and "total" figures equal the
        output figures.
    """

    output_tokens: np.ndarray
    template_labels: list[str]
    input_labels: list[str]
    input_tokens: Optional[np.ndarray] = None


@dataclass
class TokenStats:
    """Per-template token usage point estimates and bootstrap confidence intervals.

    All CIs are computed by bootstrapping over the M benchmark inputs,
    capturing epistemic uncertainty from the finite input sample.  When
    ``output_tokens`` has shape ``(N, M, R)``, per-input values are first
    averaged over R runs (cell means) before bootstrapping, which correctly
    propagates run-level variance into the input-level CI.

    Attributes
    ----------
    labels : list[str]
        Template labels (same order as the input ``TokenUsage``).
    mean_total : np.ndarray
        Shape ``(N,)``.  Mean total tokens (input + output) per API call.
        Equals ``mean_output`` when no ``input_tokens`` were provided.
    ci_low_total, ci_high_total : np.ndarray
        Shape ``(N,)``.  Bootstrap CI bounds on ``mean_total``.
    mean_output : np.ndarray
        Shape ``(N,)``.  Mean output/completion tokens per call.
    ci_low_output, ci_high_output : np.ndarray
        Shape ``(N,)``.  Bootstrap CI bounds on ``mean_output``.
    mean_input : np.ndarray or None
        Shape ``(N,)`` if ``input_tokens`` were provided, else ``None``.
    n_bootstrap : int
    ci : float
    """

    labels: list[str]
    mean_total: np.ndarray
    ci_low_total: np.ndarray
    ci_high_total: np.ndarray
    mean_output: np.ndarray
    ci_low_output: np.ndarray
    ci_high_output: np.ndarray
    mean_input: Optional[np.ndarray]
    n_bootstrap: int
    ci: float


@dataclass
class TokenAnalysisResult:
    """Complete token usage analysis for a benchmark.

    Attributes
    ----------
    usage : TokenUsage
        The raw token usage data passed in.
    stats : TokenStats
        Per-template mean estimates with bootstrap CIs.
    pairwise : PairwiseMatrix
        Pairwise comparisons of *total* token usage between all template
        pairs, using the same bootstrap method and multiple-comparisons
        correction as the score pairwise analysis.
    pareto_frontier : list[str]
        Template labels that are *not* dominated.  A template is labelled
        dominated only when another template is simultaneously and
        *statistically significantly* better on score **and** cheaper on
        tokens — both pairwise CIs exclude zero in the correct direction
        (after multiple-comparisons correction).
    dominated_by : dict[str, list[str]]
        Maps each dominated template to the list of templates that dominate
        it.  Templates on the Pareto frontier do not appear as keys.
    """

    usage: TokenUsage
    stats: TokenStats
    pairwise: PairwiseMatrix
    pareto_frontier: list[str]
    dominated_by: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Public analysis function
# ---------------------------------------------------------------------------

def analyze_tokens(
    token_usage: TokenUsage,
    score_pairwise: PairwiseMatrix,
    *,
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    rng: Optional[np.random.Generator] = None,
) -> TokenAnalysisResult:
    """Compute token usage statistics, pairwise comparisons, and Pareto frontier.

    Token pairwise comparisons reuse ``all_pairwise`` with the same bootstrap
    method and multiple-comparisons correction as the score analysis.  Pareto
    dominance labels are only applied when *both* the score improvement *and*
    the token saving are backed by statistically significant pairwise results
    (CI excludes zero after correction).

    Parameters
    ----------
    token_usage : TokenUsage
        Token counts aligned to benchmark templates and inputs.
    score_pairwise : PairwiseMatrix
        Pairwise score comparisons from the associated ``AnalysisBundle``.
        Used to determine whether score differences are significant when
        computing the Pareto frontier.
    method, ci, n_bootstrap, correction, rng :
        Same semantics as ``all_pairwise()``.

    Returns
    -------
    TokenAnalysisResult
    """
    if rng is None:
        rng = np.random.default_rng()

    labels = token_usage.template_labels
    out = np.asarray(token_usage.output_tokens, dtype=float)
    inp = (
        np.asarray(token_usage.input_tokens, dtype=float)
        if token_usage.input_tokens is not None
        else None
    )

    # Build total token array (same shape as output_tokens).
    if inp is not None:
        if out.ndim == 3:
            # Broadcast (N, M, 1) + (N, M, R) → (N, M, R)
            total = inp[:, :, np.newaxis] + out
        else:
            total = inp + out  # (N, M)
    else:
        total = out

    stats = _compute_token_stats(
        output_tokens=out,
        total_tokens=total,
        input_tokens=inp,
        labels=labels,
        method=method,
        ci=ci,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    # Pairwise comparisons on total tokens — all_pairwise handles both
    # (N, M) and (N, M, R) shapes, including nested bootstrap for R >= 3.
    token_pairwise = all_pairwise(
        total, labels,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng,
    )

    frontier, dominated_by = _compute_pareto_frontier(
        labels=labels,
        score_pairwise=score_pairwise,
        token_pairwise=token_pairwise,
    )

    return TokenAnalysisResult(
        usage=token_usage,
        stats=stats,
        pairwise=token_pairwise,
        pareto_frontier=frontier,
        dominated_by=dominated_by,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_token_stats(
    output_tokens: np.ndarray,
    total_tokens: np.ndarray,
    input_tokens: Optional[np.ndarray],
    labels: list[str],
    method: Literal["bootstrap", "bca", "auto"],
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> TokenStats:
    """Compute bootstrap CIs on per-template mean token counts."""
    N = len(labels)
    alpha = 1 - ci

    # Collapse run axis to cell means when present.
    out_cell = output_tokens.mean(axis=2) if output_tokens.ndim == 3 else output_tokens
    tot_cell = total_tokens.mean(axis=2) if total_tokens.ndim == 3 else total_tokens

    mean_output = out_cell.mean(axis=1)  # (N,)
    mean_total = tot_cell.mean(axis=1)   # (N,)
    mean_input = input_tokens.mean(axis=1) if input_tokens is not None else None

    M = out_cell.shape[1]
    resolved = resolve_resampling_method(method, M)

    ci_low_output = np.empty(N)
    ci_high_output = np.empty(N)
    ci_low_total = np.empty(N)
    ci_high_total = np.empty(N)

    for i in range(N):
        ci_low_output[i], ci_high_output[i] = bootstrap_ci_1d(
            out_cell[i], float(mean_output[i]), resolved, n_bootstrap, alpha, rng,
        )
        ci_low_total[i], ci_high_total[i] = bootstrap_ci_1d(
            tot_cell[i], float(mean_total[i]), resolved, n_bootstrap, alpha, rng,
        )

    return TokenStats(
        labels=labels,
        mean_total=mean_total,
        ci_low_total=ci_low_total,
        ci_high_total=ci_high_total,
        mean_output=mean_output,
        ci_low_output=ci_low_output,
        ci_high_output=ci_high_output,
        mean_input=mean_input,
        n_bootstrap=n_bootstrap,
        ci=ci,
    )


def _compute_pareto_frontier(
    labels: list[str],
    score_pairwise: PairwiseMatrix,
    token_pairwise: PairwiseMatrix,
) -> tuple[list[str], dict[str, list[str]]]:
    """Identify Pareto-dominated templates using corrected p-values.

    Template *b* is considered to dominate template *a* only when **both**:

     1. The score comparison (b vs a) has ``p_value < 0.05`` **and** b scores
         higher (``mean_diff > 0``).
     2. The token comparison (b vs a) has ``p_value < 0.05`` **and** b uses
         fewer tokens (``mean_diff < 0``).

    This is intentionally conservative: templates remain on the frontier
     whenever either comparison does not meet the p-value threshold.

    Returns
    -------
    frontier : list[str]
        Templates not dominated by any other.
    dominated_by : dict[str, list[str]]
        Dominated templates mapped to their dominator(s).  Frontier
        templates do not appear as keys.
    """
    pvalue_alpha = 0.05
    dominated_by_all: dict[str, list[str]] = {label: [] for label in labels}

    for a in labels:
        for b in labels:
            if a == b:
                continue
            try:
                # get(b, a) returns comparison with mean = value(b) − value(a)
                score_r = score_pairwise.get(b, a)
                token_r = token_pairwise.get(b, a)
            except KeyError:
                continue

            b_significantly_better = (
                score_r.p_value < pvalue_alpha and score_r.point_diff > 0
            )
            b_significantly_cheaper = (
                token_r.p_value < pvalue_alpha and token_r.point_diff < 0
            )

            if b_significantly_better and b_significantly_cheaper:
                dominated_by_all[a].append(b)

    frontier = [lbl for lbl in labels if not dominated_by_all[lbl]]
    dominated_by = {k: v for k, v in dominated_by_all.items() if v}

    return frontier, dominated_by
