"""Robustness and variance metrics for prompt templates.

Quantifies how consistent each template's performance is across inputs.
A template can be "best on average" but highly volatile — these metrics
make that tradeoff explicit.

When the score array includes a runs axis (R >= 3), ``robustness_metrics``
operates on per-input cell means (averaged over runs), which isolates
input-level variability.  Seed-level variability — how much of a template's
variance stems from LLM stochasticity rather than input sensitivity — is
separately captured by ``seed_variance_decomposition`` and
``SeedVarianceResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobustnessResult:
    """Per-template robustness metrics.

    Attributes
    ----------
    labels : list[str]
        Template labels.
    mean : np.ndarray
        Mean score per template.
    std : np.ndarray
        Standard deviation per template (between-input, on cell means).
    cv : np.ndarray
        Coefficient of variation (std / mean). NaN if mean is zero.
    iqr : np.ndarray
        Interquartile range per template.
    cvar_10 : np.ndarray
        Conditional Value at Risk: mean of the worst 10% of scores.
    percentiles : dict[int, np.ndarray]
        Score percentiles (10, 25, 50, 75, 90) per template.
    failure_rate : np.ndarray or None
        Fraction of inputs below threshold, if threshold was specified.
    failure_threshold : float or None
        The threshold used for failure_rate.
    """

    labels: list[str]
    mean: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    iqr: np.ndarray
    cvar_10: np.ndarray
    percentiles: dict[int, np.ndarray]
    failure_rate: Optional[np.ndarray]
    failure_threshold: Optional[float]

    def summary_table(self):
        """Return a pandas DataFrame summarizing all metrics."""
        import pandas as pd

        data = {
            "template": self.labels,
            "mean": self.mean,
            "std": self.std,
            "cv": self.cv,
            "iqr": self.iqr,
            "cvar_10": self.cvar_10,
            "p10": self.percentiles[10],
            "p25": self.percentiles[25],
            "p50": self.percentiles[50],
            "p75": self.percentiles[75],
            "p90": self.percentiles[90],
        }
        if self.failure_rate is not None:
            data[f"failure_rate (<{self.failure_threshold})"] = self.failure_rate

        return pd.DataFrame(data).set_index("template")


@dataclass
class SeedVarianceResult:
    """Per-template decomposition of score variance into seed and input components.

    Total score variance for a template is the sum of two orthogonal parts
    (ANOVA / law of total variance):

    .. code-block:: text

        Var_total[t] = Var_input[t]  +  Var_seed[t]
                       ─────────────    ────────────
                       between inputs   within cells
                       (cell means)     (across runs)

    ``instability`` is the fraction of total variance attributable to LLM
    stochasticity (seed draws), ranging from 0 (fully deterministic given
    the input) to 1 (output is pure noise independent of the input).

    Attributes
    ----------
    labels : list[str]
        Template labels.
    n_runs : int
        Number of repeated runs R used to compute these estimates.
    seed_var : np.ndarray
        Shape ``(N,)``.  Mean within-cell variance, averaged over inputs.
        Computed with ``ddof=1`` within each cell.
    input_var : np.ndarray
        Shape ``(N,)``.  Between-input variance of cell means (``ddof=1``).
    total_var : np.ndarray
        Shape ``(N,)``.  ``seed_var + input_var``.
    instability : np.ndarray
        Shape ``(N,)``.  ``per_cell_seed_std.mean(axis=1)``: the mean over
        inputs of the within-cell seed standard deviation.  Directly answers
        "on average across inputs, how much does the score vary across
        repeated runs?" in the same units as the score.  Unlike
        ``seed_fraction`` this does not inflate when total variance is near
        zero (e.g. a near-perfect template with one bad run).
    seed_fraction : np.ndarray
        Shape ``(N,)``.  ``seed_var / total_var``: the ANOVA partition of
        variance attributable to LLM stochasticity.  Ranges from 0 to 1, but
        becomes unreliable when ``total_var`` is near zero.  NaN when
        ``total_var == 0``.
    per_cell_seed_std : np.ndarray
        Shape ``(N, M)``.  Within-cell standard deviation for every
        (template, input) pair, useful for spotting which inputs are
        especially noisy.
    """

    labels: list[str]
    n_runs: int
    seed_var: np.ndarray
    input_var: np.ndarray
    total_var: np.ndarray
    instability: np.ndarray
    seed_fraction: np.ndarray
    per_cell_seed_std: np.ndarray

    def summary_table(self):
        """Return a pandas DataFrame with one row per template."""
        import pandas as pd

        data = {
            "template": self.labels,
            "seed_std": np.sqrt(self.seed_var),
            "input_std": np.sqrt(self.input_var),
            "total_std": np.sqrt(self.total_var),
            "instability": self.instability,
            "seed_fraction": self.seed_fraction,
        }
        return pd.DataFrame(data).set_index("template")


def robustness_metrics(
    scores: np.ndarray,
    labels: list[str],
    failure_threshold: Optional[float] = None,
) -> RobustnessResult:
    """Compute robustness metrics for each template.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(N, M)`` or ``(N, M, R)``.
        When 3-D, the metrics are computed on per-input cell means
        (averaged over the run axis), which isolates input-level variability.
        Use ``seed_variance_decomposition`` for the seed-level component.
    labels : list[str]
        Template labels.
    failure_threshold : float, optional
        If provided, computes the fraction of inputs scoring below this value.

    Returns
    -------
    RobustnessResult
    """
    if scores.ndim == 3:
        scores = scores.mean(axis=2)  # (N, M) cell means

    n_templates, m_inputs = scores.shape

    mean = scores.mean(axis=1)
    std = scores.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean != 0, std / np.abs(mean), np.nan)

    p10 = np.percentile(scores, 10, axis=1)
    p25 = np.percentile(scores, 25, axis=1)
    p50 = np.percentile(scores, 50, axis=1)
    p75 = np.percentile(scores, 75, axis=1)
    p90 = np.percentile(scores, 90, axis=1)
    iqr = p75 - p25

    # CVaR (Expected Shortfall): mean of the worst 10% of scores
    cvar_10 = np.empty(n_templates)
    k = max(1, int(np.floor(m_inputs * 0.10)))
    for i in range(n_templates):
        sorted_scores = np.sort(scores[i])
        cvar_10[i] = sorted_scores[:k].mean()

    failure_rate = None
    if failure_threshold is not None:
        failure_rate = (scores < failure_threshold).mean(axis=1)

    return RobustnessResult(
        labels=labels,
        mean=mean,
        std=std,
        cv=cv,
        iqr=iqr,
        cvar_10=cvar_10,
        percentiles={10: p10, 25: p25, 50: p50, 75: p75, 90: p90},
        failure_rate=failure_rate,
        failure_threshold=failure_threshold,
    )


def seed_variance_decomposition(
    scores: np.ndarray,
    labels: list[str],
) -> SeedVarianceResult:
    """Decompose per-template variance into seed and input components.

    Uses the law of total variance::

        Var_total = Var_input + Var_seed
                  = Var_i(E[X|i])  +  E_i[Var_r(X|i)]

    where the outer expectations are over benchmark inputs and the inner
    variance is over repeated runs (seeds) for a fixed input.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M, R)`` with ``R >= 3``.
    labels : list[str]
        Template labels, length N.

    Returns
    -------
    SeedVarianceResult
    """
    if scores.ndim != 3:
        raise ValueError(
            f"seed_variance_decomposition requires a 3-D array of shape "
            f"(N, M, R); got {scores.ndim}-D."
        )
    N, M, R = scores.shape
    if R < 3:
        raise ValueError(
            f"Seed-variance decomposition requires R >= 3 runs; got R={R}. "
            "Collect at least 3 repeated runs per (template, input) cell."
        )

    cell_means = scores.mean(axis=2)           # (N, M) — E[X | template, input]

    # Within-cell variance for each (template, input) pair.
    per_cell_seed_var = scores.var(axis=2, ddof=1)    # (N, M)
    per_cell_seed_std = np.sqrt(per_cell_seed_var)    # (N, M)

    # E_i[Var_r(X|i)]: average within-cell variance over inputs.
    seed_var = per_cell_seed_var.mean(axis=1)         # (N,)

    # Var_i(E[X|i]): between-input variance of cell means.
    input_var = cell_means.var(axis=1, ddof=1)        # (N,)

    total_var = seed_var + input_var                  # (N,)

    with np.errstate(divide="ignore", invalid="ignore"):
        seed_fraction = np.where(total_var > 0, seed_var / total_var, np.nan)

    # Mean over inputs of within-cell seed std: "on average across inputs,
    # how much do scores vary run-to-run?"  Does not inflate when total_var
    # is near zero, unlike the seed_fraction ratio.
    instability = per_cell_seed_std.mean(axis=1)          # (N,)

    return SeedVarianceResult(
        labels=labels,
        n_runs=R,
        seed_var=seed_var,
        input_var=input_var,
        total_var=total_var,
        instability=instability,
        seed_fraction=seed_fraction,
        per_cell_seed_std=per_cell_seed_std,
    )
