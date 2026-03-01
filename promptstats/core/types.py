"""Core data types for benchpress."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark scores across templates and inputs.

    The fundamental input to all benchpress analyses. Wraps a score matrix
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
                warnings.warn(
                    f"scores has shape {s.shape}: only 2 runs detected. "
                    "Seed-variance analysis requires R >= 3 runs. "
                    "Scores will be pre-averaged across runs before analysis.",
                    UserWarning,
                    stacklevel=3,
                )
            # Catch the common mistake of passing old-style (N, M, K) evaluators.
            if (
                self.evaluator_names != ["score"]
                and len(self.evaluator_names) == n_runs
            ):
                warnings.warn(
                    f"scores has shape {s.shape} and evaluator_names has "
                    f"{len(self.evaluator_names)} entries matching axis 2. "
                    "Axis 2 is now the *runs* axis, not the evaluator axis. "
                    "For K evaluators without repeated runs use shape (N, M, 1, K).",
                    UserWarning,
                    stacklevel=3,
                )
        elif s.ndim == 4:
            n_templates, n_inputs, n_runs, n_evals = s.shape
            if n_runs == 2:
                warnings.warn(
                    f"scores has shape {s.shape}: only 2 runs detected. "
                    "Seed-variance analysis requires R >= 3 runs. "
                    "Scores will be pre-averaged across runs before analysis.",
                    UserWarning,
                    stacklevel=3,
                )
            if len(self.evaluator_names) != n_evals:
                raise ValueError(
                    f"evaluator_names length ({len(self.evaluator_names)}) "
                    f"does not match scores axis 3 ({n_evals})"
                )
        else:
            raise ValueError(
                f"scores must be 2-D (N, M), 3-D (N, M, R), or "
                f"4-D (N, M, R, K); got {s.ndim}-D"
            )

        if len(self.template_labels) != n_templates:
            raise ValueError(
                f"template_labels length ({len(self.template_labels)}) "
                f"does not match scores axis 0 ({n_templates})"
            )
        if len(self.input_labels) != n_inputs:
            raise ValueError(
                f"input_labels length ({len(self.input_labels)}) "
                f"does not match scores axis 1 ({n_inputs})"
            )
        if len(self.template_labels) != len(set(self.template_labels)):
            raise ValueError("template_labels must be unique")
        if len(self.input_labels) != len(set(self.input_labels)):
            raise ValueError("input_labels must be unique")

        if not np.all(np.isfinite(s)):
            raise ValueError("scores contain NaN or infinite values")

        if self.input_metadata is not None:
            if len(self.input_metadata) != n_inputs:
                raise ValueError(
                    f"input_metadata length ({len(self.input_metadata)}) "
                    f"does not match number of inputs ({n_inputs})"
                )

        if self.baseline_template is not None:
            if self.baseline_template not in self.template_labels:
                raise ValueError(
                    f"baseline_template '{self.baseline_template}' "
                    f"not found in template_labels"
                )

        # Warn about zero-variance rows (using cell means for multi-run data).
        cell_means_2d = self.get_2d_scores()
        for i, label in enumerate(self.template_labels):
            if np.std(cell_means_2d[i]) == 0:
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
                warnings.warn(
                    f"scores has shape {s.shape}: only 2 runs detected. "
                    "Seed-variance analysis requires R >= 3 runs. "
                    "Scores will be pre-averaged across runs before analysis.",
                    UserWarning,
                    stacklevel=3,
                )
            if (
                self.evaluator_names != ["score"]
                and len(self.evaluator_names) == n_runs
            ):
                warnings.warn(
                    f"scores has shape {s.shape} and evaluator_names has "
                    f"{len(self.evaluator_names)} entries matching axis 3. "
                    "Axis 3 is now the *runs* axis, not the evaluator axis. "
                    "For K evaluators without repeated runs use shape (P, N, M, 1, K).",
                    UserWarning,
                    stacklevel=3,
                )
        elif s.ndim == 5:
            n_models, n_templates, n_inputs, n_runs, n_evals = s.shape
            if n_runs == 2:
                warnings.warn(
                    f"scores has shape {s.shape}: only 2 runs detected. "
                    "Seed-variance analysis requires R >= 3 runs. "
                    "Scores will be pre-averaged across runs before analysis.",
                    UserWarning,
                    stacklevel=3,
                )
            if len(self.evaluator_names) != n_evals:
                raise ValueError(
                    f"evaluator_names length ({len(self.evaluator_names)}) "
                    f"does not match scores axis 4 ({n_evals})"
                )
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

        if len(self.model_labels) != n_models:
            raise ValueError(
                f"model_labels length ({len(self.model_labels)}) "
                f"does not match scores axis 0 ({n_models})"
            )
        if len(self.template_labels) != n_templates:
            raise ValueError(
                f"template_labels length ({len(self.template_labels)}) "
                f"does not match scores axis 1 ({n_templates})"
            )
        if len(self.input_labels) != n_inputs:
            raise ValueError(
                f"input_labels length ({len(self.input_labels)}) "
                f"does not match scores axis 2 ({n_inputs})"
            )
        if len(self.model_labels) != len(set(self.model_labels)):
            raise ValueError("model_labels must be unique")
        if len(self.template_labels) != len(set(self.template_labels)):
            raise ValueError("template_labels must be unique")
        if len(self.input_labels) != len(set(self.input_labels)):
            raise ValueError("input_labels must be unique")

        if not np.all(np.isfinite(s)):
            raise ValueError("scores contain NaN or infinite values")

        if self.input_metadata is not None:
            if len(self.input_metadata) != n_inputs:
                raise ValueError(
                    f"input_metadata length ({len(self.input_metadata)}) "
                    f"does not match number of inputs ({n_inputs})"
                )

        # Warn about zero-variance (model, template) cells using 2-D view.
        scores_3d = self._get_3d_cell_means()
        for m_idx, model_label in enumerate(self.model_labels):
            for t_idx, template_label in enumerate(self.template_labels):
                if np.std(scores_3d[m_idx, t_idx]) == 0:
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
        by its mean cell-mean performance over all prompt templates.
        """
        return BenchmarkResult(
            scores=self._get_3d_cell_means().mean(axis=1),  # (P, M)
            template_labels=self.model_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )
