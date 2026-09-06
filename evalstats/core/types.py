"""Core data types for evalstats."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared method type aliases
# ---------------------------------------------------------------------------
CommonStatsMethods = Literal[
    "bootstrap",
    "bca",
    "bayes_bootstrap",
    "smooth_bootstrap",
    "bootstrap_t",
    "t_interval",
    "logit_t",
    "nig",
    "auto",
    "bayes_binary",
    "wilson",
    "newcombe",
    "tango",
    "mj_floor",
    "bonett_price",
    "permutation",
    "sign_test",
]
CompareMethod = CommonStatsMethods
AnalyzeMethod = Union[CommonStatsMethods, Literal["lmm"]]


# ---------------------------------------------------------------------------
# Private validation helpers shared by both dataclasses
# ---------------------------------------------------------------------------

def _warn_two_runs(shape: tuple, *, stacklevel: int = 4) -> None:
    """Emit a warning when only 2 repeated runs are detected."""
    warnings.warn(
        f"scores has shape {shape}: only 2 runs detected. "
        "Seed-variance analysis requires R >= 3 runs. "
        "Scores will be pre-averaged across runs before analysis.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _warn_evaluator_axis_confusion(
    evaluator_names: list,
    n_runs: int,
    shape: tuple,
    *,
    runs_axis: int,
    shape_hint: str,
    stacklevel: int = 4,
) -> None:
    """Warn when evaluator_names count accidentally matches the runs axis."""
    if evaluator_names != ["score"] and len(evaluator_names) == n_runs:
        warnings.warn(
            f"scores has shape {shape} and evaluator_names has "
            f"{len(evaluator_names)} entries matching axis {runs_axis}. "
            f"Axis {runs_axis} is now the *runs* axis, not the evaluator axis. "
            f"For K evaluators without repeated runs use shape {shape_hint}.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _check_evaluator_count(
    evaluator_names: list, n_evals: int, *, axis: int
) -> None:
    """Raise if evaluator_names does not match the evaluator axis length."""
    if len(evaluator_names) != n_evals:
        raise ValueError(
            f"evaluator_names length ({len(evaluator_names)}) "
            f"does not match scores axis {axis} ({n_evals})"
        )


def _check_label_length(
    labels: list, n: int, *, name: str, axis: int
) -> None:
    """Raise if a label list does not match the expected axis length."""
    if len(labels) != n:
        raise ValueError(
            f"{name} length ({len(labels)}) "
            f"does not match scores axis {axis} ({n})"
        )


def _check_labels_unique(labels: list, *, name: str) -> None:
    """Raise if a label list contains duplicates."""
    if len(labels) != len(set(labels)):
        raise ValueError(f"{name} must be unique")


def _check_no_inf(scores: np.ndarray) -> None:
    """Raise if the score array contains any infinite values."""
    if np.any(np.isinf(scores)):
        raise ValueError(
            "scores contain infinite values. "
            "Use np.nan to represent missing (not-evaluated) cells."
        )


def _check_metadata_length(
    metadata: Optional[pd.DataFrame], n_inputs: int
) -> None:
    """Raise if input_metadata length does not match the number of inputs."""
    if metadata is not None and len(metadata) != n_inputs:
        raise ValueError(
            f"input_metadata length ({len(metadata)}) "
            f"does not match number of inputs ({n_inputs})"
        )


@dataclass
class BenchmarkResult:
    """Container for benchmark scores across templates and inputs.

    The fundamental input to all evalstats analyses. Wraps a score matrix
    where every template has been evaluated on every input (complete design).

    Score array shape convention
    ----------------------------
    * ``(N, M)``       — no runs, no evaluators (original format)
    * ``(N, M, R)``    — R repeated runs per cell, no evaluators
    * ``(N, M, R, K)`` — R runs **and** K evaluators

    The runs axis (axis 2) is the seed-variance dimension.  Passing only
    K evaluators with no runs is no longer supported as a 3-D array; use
    shape ``(N, M, 1, K)`` in that case.

    Seed-variance analysis is activated when ``R >= 3``.  Exactly ``R = 2``
    emits a warning and the runs are pre-averaged before any analysis.

    Parameters
    ----------
    scores : np.ndarray
        Score array with one of the shapes described above.
    template_labels : list[str]
        Human-readable names for each template (axis 0).
    input_labels : list[str]
        Identifiers for each benchmark input (axis 1).
    evaluator_names : list[str], optional
        Names of each evaluator.  Required (and must match axis 3) when
        ``scores`` is 4-D.  Ignored for 2-D and 3-D arrays.
    input_metadata : pd.DataFrame, optional
        Metadata for each input (e.g., category, difficulty).
    baseline_template : str, optional
        Label of a designated baseline template for comparison.
    """

    scores: np.ndarray
    template_labels: list[str]
    input_labels: list[str]
    evaluator_names: list[str] = field(default_factory=lambda: ["score"])
    input_metadata: Optional[pd.DataFrame] = None
    baseline_template: Optional[str] = None
    template_factors: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self._validate()

    def _validate(self):
        s = self.scores
        if s.ndim == 2:
            n_templates, n_inputs = s.shape
        elif s.ndim == 3:
            n_templates, n_inputs, n_runs = s.shape
            if n_runs == 2:
                _warn_two_runs(s.shape)
            # Catch the common mistake of passing old-style (N, M, K) evaluators.
            _warn_evaluator_axis_confusion(
                self.evaluator_names, n_runs, s.shape,
                runs_axis=2, shape_hint="(N, M, 1, K)",
            )
        elif s.ndim == 4:
            n_templates, n_inputs, n_runs, n_evals = s.shape
            if n_runs == 2:
                _warn_two_runs(s.shape)
            _check_evaluator_count(self.evaluator_names, n_evals, axis=3)
        else:
            raise ValueError(
                f"scores must be 2-D (N, M), 3-D (N, M, R), or "
                f"4-D (N, M, R, K); got {s.ndim}-D"
            )

        _check_label_length(self.template_labels, n_templates, name="template_labels", axis=0)
        _check_label_length(self.input_labels, n_inputs, name="input_labels", axis=1)
        _check_labels_unique(self.template_labels, name="template_labels")
        _check_labels_unique(self.input_labels, name="input_labels")
        _check_no_inf(s)
        _check_metadata_length(self.input_metadata, n_inputs)

        if self.baseline_template is not None:
            if self.baseline_template not in self.template_labels:
                raise ValueError(
                    f"baseline_template '{self.baseline_template}' "
                    f"not found in template_labels"
                )

        if self.template_factors is not None:
            if len(self.template_factors) != n_templates:
                raise ValueError(
                    f"template_factors length ({len(self.template_factors)}) "
                    f"does not match number of templates ({n_templates})"
                )
            if len(self.template_factors.columns) == 0:
                raise ValueError(
                    "template_factors must have at least one column (factor)"
                )
            for col in self.template_factors.columns:
                if not str(col).isidentifier():
                    raise ValueError(
                        f"template_factors column name '{col}' is not a valid "
                        "Python identifier. Rename it (e.g., replace spaces with "
                        "underscores) so it can be used in model formulas."
                    )
            if self.template_factors.isnull().any(axis=None):
                warnings.warn(
                    "template_factors contains NaN values. Factor columns should "
                    "be fully specified for all templates.",
                    UserWarning,
                    stacklevel=3,
                )

        # Warn about zero-variance rows (using cell means for multi-run data).
        cell_means_2d = self.get_2d_scores()
        for i, label in enumerate(self.template_labels):
            row = cell_means_2d[i]
            if not np.all(np.isnan(row)) and np.nanstd(row) == 0:
                warnings.warn(
                    f"Template '{label}' has zero variance across inputs "
                    f"(all scores identical). This may indicate a problem.",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_templates(self) -> int:
        return self.scores.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.scores.shape[1]

    @property
    def n_runs(self) -> int:
        """Number of repeated runs (seeds) per cell.  1 when no run axis."""
        return self.scores.shape[2] if self.scores.ndim >= 3 else 1

    @property
    def n_evaluators(self) -> int:
        """Number of evaluators.  1 when no evaluator axis."""
        return self.scores.shape[3] if self.scores.ndim == 4 else 1

    @property
    def is_aggregated(self) -> bool:
        """True when scores are 2-D (no run or evaluator axes)."""
        return self.scores.ndim == 2

    @property
    def is_seeded(self) -> bool:
        """True when scores carry a run axis with R >= 3 independent runs."""
        return self.n_runs >= 3

    @property
    def has_missing(self) -> bool:
        """True when the score array contains NaN (missing) cells."""
        return bool(np.any(np.isnan(self.scores)))

    # ------------------------------------------------------------------
    # Score accessors
    # ------------------------------------------------------------------

    def get_2d_scores(self) -> np.ndarray:
        """Return ``(N, M)`` score matrix, averaging over runs and evaluators."""
        s = self.scores
        if s.ndim == 2:
            return s
        if s.ndim == 3:
            return s.mean(axis=2)      # average runs → (N, M)
        return s.mean(axis=(2, 3))     # average runs then evaluators → (N, M)

    def get_run_scores(self) -> np.ndarray:
        """Return ``(N, M, R)`` array, averaging evaluators if present.

        When no run axis exists (2-D input), returns shape ``(N, M, 1)``
        so callers can always index the run dimension uniformly.
        When ``R = 2``, returns the averaged 2-D data wrapped as
        ``(N, M, 1)`` (the warning was already issued in ``_validate``).
        """
        s = self.scores
        if s.ndim == 2:
            return s[:, :, np.newaxis]          # (N, M, 1)
        if s.ndim == 3:
            if s.shape[2] == 2:
                return s.mean(axis=2)[:, :, np.newaxis]  # pre-average → (N, M, 1)
            return s                             # (N, M, R)
        # 4-D (N, M, R, K): average evaluators
        run_scores = s.mean(axis=3)             # (N, M, R)
        if run_scores.shape[2] == 2:
            return run_scores.mean(axis=2)[:, :, np.newaxis]
        return run_scores

    def template_index(self, label: str) -> int:
        """Get the index of a template by label."""
        try:
            return self.template_labels.index(label)
        except ValueError:
            raise KeyError(f"Template '{label}' not found")


@dataclass
class MultiModelBenchmark:
    """Benchmark scores across multiple models, templates, and inputs.

    Extends BenchmarkResult to a structure that adds a model axis as the
    outermost dimension.  Every (model, template) pair must be evaluated on
    the same complete set of inputs.

    Score array shape convention
    ----------------------------
    * ``(P, N, M)``       — no runs, no evaluators
    * ``(P, N, M, R)``    — R repeated runs per cell, no evaluators
    * ``(P, N, M, R, K)`` — R runs **and** K evaluators

    Parameters
    ----------
    scores : np.ndarray
        Score tensor with one of the shapes above.  All values must be
        finite numeric.
    model_labels : list[str]
        Human-readable names for each model (axis 0).  At least 2 required.
    template_labels : list[str]
        Human-readable names for each prompt template (axis 1).
    input_labels : list[str]
        Identifiers for each benchmark input (axis 2).
    evaluator_names : list[str], optional
        Names of each evaluator.  Required (and must match axis 4) when
        ``scores`` is 5-D.
    input_metadata : pd.DataFrame, optional
        Metadata for each input.
    """

    scores: np.ndarray
    model_labels: list[str]
    template_labels: list[str]
    input_labels: list[str]
    evaluator_names: list[str] = field(default_factory=lambda: ["score"])
    input_metadata: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self._validate()

    def _validate(self):
        s = self.scores
        if s.ndim == 3:
            n_models, n_templates, n_inputs = s.shape
        elif s.ndim == 4:
            n_models, n_templates, n_inputs, n_runs = s.shape
            if n_runs == 2:
                _warn_two_runs(s.shape)
            _warn_evaluator_axis_confusion(
                self.evaluator_names, n_runs, s.shape,
                runs_axis=3, shape_hint="(P, N, M, 1, K)",
            )
        elif s.ndim == 5:
            n_models, n_templates, n_inputs, n_runs, n_evals = s.shape
            if n_runs == 2:
                _warn_two_runs(s.shape)
            _check_evaluator_count(self.evaluator_names, n_evals, axis=4)
        else:
            raise ValueError(
                f"scores must be 3-D (P, N, M), 4-D (P, N, M, R), or "
                f"5-D (P, N, M, R, K); got {s.ndim}-D"
            )

        if n_models < 2:
            raise ValueError(
                f"MultiModelBenchmark requires at least 2 models; got {n_models}. "
                "Use BenchmarkResult for single-model benchmarks."
            )

        _check_label_length(self.model_labels, n_models, name="model_labels", axis=0)
        _check_label_length(self.template_labels, n_templates, name="template_labels", axis=1)
        _check_label_length(self.input_labels, n_inputs, name="input_labels", axis=2)
        _check_labels_unique(self.model_labels, name="model_labels")
        _check_labels_unique(self.template_labels, name="template_labels")
        _check_labels_unique(self.input_labels, name="input_labels")
        _check_no_inf(s)
        _check_metadata_length(self.input_metadata, n_inputs)

        # Warn about zero-variance (model, template) cells using 2-D view.
        scores_3d = self._get_3d_cell_means()
        for m_idx, model_label in enumerate(self.model_labels):
            for t_idx, template_label in enumerate(self.template_labels):
                row = scores_3d[m_idx, t_idx]
                if not np.all(np.isnan(row)) and np.nanstd(row) == 0:
                    warnings.warn(
                        f"Template '{template_label}' for model '{model_label}' "
                        f"has zero variance across inputs (all scores identical). "
                        f"This may indicate a problem.",
                        UserWarning,
                        stacklevel=3,
                    )

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        return self.scores.shape[0]

    @property
    def n_templates(self) -> int:
        return self.scores.shape[1]

    @property
    def n_inputs(self) -> int:
        return self.scores.shape[2]

    @property
    def n_runs(self) -> int:
        """Number of repeated runs per cell.  1 when no run axis."""
        return self.scores.shape[3] if self.scores.ndim >= 4 else 1

    @property
    def n_evaluators(self) -> int:
        """Number of evaluators.  1 when no evaluator axis."""
        return self.scores.shape[4] if self.scores.ndim == 5 else 1

    @property
    def is_aggregated(self) -> bool:
        """True when scores are 3-D (no run or evaluator axes)."""
        return self.scores.ndim == 3

    @property
    def is_seeded(self) -> bool:
        """True when scores carry a run axis with R >= 3 independent runs."""
        return self.n_runs >= 3

    @property
    def has_missing(self) -> bool:
        """True when the score array contains NaN (missing) cells."""
        return bool(np.any(np.isnan(self.scores)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_3d_cell_means(self) -> np.ndarray:
        """Return ``(P, N, M)`` cell-mean scores (averaging runs/evaluators)."""
        s = self.scores
        if s.ndim == 3:
            return s
        if s.ndim == 4:
            return s.mean(axis=3)
        return s.mean(axis=(3, 4))

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def get_model_result(self, model: str) -> BenchmarkResult:
        """Slice out one model's scores as a BenchmarkResult.

        The returned array preserves the run and evaluator axes when present,
        yielding shape ``(N, M)``, ``(N, M, R)``, or ``(N, M, R, K)``.
        """
        try:
            idx = self.model_labels.index(model)
        except ValueError:
            raise KeyError(f"Model '{model}' not found in model_labels")
        return BenchmarkResult(
            scores=self.scores[idx],
            template_labels=self.template_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )

    def get_flat_result(self, sep: str = " / ") -> BenchmarkResult:
        """Flatten all (model, template) pairs into a single template axis.

        Labels are formatted as ``"<model><sep><template>"``.  The run and
        evaluator axes, if present, are preserved in the returned array.
        """
        n_flat = self.n_models * self.n_templates
        s = self.scores
        if s.ndim == 3:
            flat_scores = s.reshape(n_flat, self.n_inputs)
        elif s.ndim == 4:
            flat_scores = s.reshape(n_flat, self.n_inputs, self.n_runs)
        else:
            flat_scores = s.reshape(n_flat, self.n_inputs, self.n_runs, self.n_evaluators)

        flat_labels = [
            f"{m}{sep}{t}"
            for m in self.model_labels
            for t in self.template_labels
        ]
        return BenchmarkResult(
            scores=flat_scores,
            template_labels=flat_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )

    def get_model_mean_result(self) -> BenchmarkResult:
        """Aggregate each model's scores by averaging across templates.

        Each 'template' in the returned result represents one model, scored
        by its mean performance over all prompt templates.

        This aggregation preserves the runs axis (and evaluator axis, when
        present) so downstream paired comparisons can still use seeded
        nested-bootstrap procedures when ``R >= 3``.
        """
        collapsed_scores = self.scores.mean(axis=1)
        return BenchmarkResult(
            scores=collapsed_scores,
            template_labels=self.model_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )

    def get_template_mean_result(
        self,
        *,
        collapse_models: Literal["mean", "as_runs"] = "mean",
    ) -> BenchmarkResult:
        """Aggregate each template across models.

        Parameters
        ----------
        collapse_models : {"mean", "as_runs"}
            - ``"mean"`` (default): average model axis directly.
            - ``"as_runs"``: treat models as repeated runs so cross-model
              variation is retained in the run axis for downstream bootstrap
              and seed-variance style summaries.

        Returns
        -------
        BenchmarkResult
            One prompt template per row, aligned to the original inputs.
        """
        if collapse_models == "mean":
            collapsed_scores = self.scores.mean(axis=0)
        elif collapse_models == "as_runs":
            s = self.scores
            if s.ndim == 3:
                # (P, N, M) -> (N, M, P)
                collapsed_scores = np.transpose(s, (1, 2, 0))
            elif s.ndim == 4:
                # (P, N, M, R) -> (N, M, R, P) -> (N, M, R*P)
                transposed = np.transpose(s, (1, 2, 3, 0))
                collapsed_scores = transposed.reshape(
                    self.n_templates,
                    self.n_inputs,
                    self.n_runs * self.n_models,
                )
            else:
                # (P, N, M, R, K) -> (N, M, R, P, K) -> (N, M, R*P, K)
                transposed = np.transpose(s, (1, 2, 3, 0, 4))
                collapsed_scores = transposed.reshape(
                    self.n_templates,
                    self.n_inputs,
                    self.n_runs * self.n_models,
                    self.n_evaluators,
                )
        else:
            raise ValueError(
                f"Unknown collapse_models '{collapse_models}'. "
                "Expected 'mean' or 'as_runs'."
            )

        return BenchmarkResult(
            scores=collapsed_scores,
            template_labels=self.template_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )
