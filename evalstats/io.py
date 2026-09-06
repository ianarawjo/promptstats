"""Data ingestion helpers for evalstats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from evalstats.core.types import BenchmarkResult, MultiModelBenchmark
from evalstats.loader import _CANONICAL_ALIASES, _SCORE_ALIASES, _find_col


@dataclass
class DataLoadReport:
    """Structured load/coercion details emitted by ``from_dataframe``."""

    format_requested: Literal["auto", "wide", "long"]
    format_detected: Literal["wide", "long"]
    score_non_numeric_coerced: int = 0
    duplicate_groups_collapsed: int = 0
    run_nan_values_filled: int = 0
    notes: list[str] = field(default_factory=list)

    def to_lines(self) -> list[str]:
        lines = [
            f"format: requested={self.format_requested}, detected={self.format_detected}",
            f"score values coerced to NaN: {self.score_non_numeric_coerced}",
            f"duplicate groups collapsed by mean: {self.duplicate_groups_collapsed}",
            f"missing run slots imputed from cell mean: {self.run_nan_values_filled}",
        ]
        lines.extend(self.notes)
        return lines


ResultType = Union[BenchmarkResult, MultiModelBenchmark]
_IMPLICIT_TEMPLATE_COL = "__evalstats_template__"
_IMPLICIT_TEMPLATE_LABEL = "default_prompt"


def from_dataframe(
    df: pd.DataFrame,
    *,
    format: Literal["auto", "wide", "long"] = "auto",
    repair: bool = True,
    strict_complete_design: bool = True,
    return_report: bool = False,
) -> Union[ResultType, tuple[ResultType, DataLoadReport]]:
    """Create a benchmark object from a pandas DataFrame.

    This parser is forgiving by default:
    - score-like values are coerced to numeric where possible,
    - duplicate cells are averaged,
    - missing run slots are imputed from available runs in the same cell.

    Parameters
    ----------
    df : pd.DataFrame
        Input table in either wide or tidy format.
    format : {'auto', 'wide', 'long'}
        Explicit format or auto-detect.
    repair : bool
        When True, apply soft repairs (duplicate averaging and run-slot fill).
    strict_complete_design : bool
        When True, require every prompt/input combination; otherwise allow NaN.
    return_report : bool
        When True, return ``(result, DataLoadReport)``.
    """
    if format not in {"auto", "wide", "long"}:
        raise ValueError("format must be one of: 'auto', 'wide', 'long'")

    fmt = _detect_format(df) if format == "auto" else format
    report = DataLoadReport(format_requested=format, format_detected=fmt)

    if fmt == "wide":
        result = _from_wide(df, report=report, strict_complete_design=strict_complete_design)
    else:
        result = _from_long(
            df,
            report=report,
            repair=repair,
            strict_complete_design=strict_complete_design,
        )

    if return_report:
        return result, report
    return result


def _detect_format(df: pd.DataFrame) -> Literal["wide", "long"]:
    cols_lower = {c.lower().strip() for c in df.columns}
    has_template = bool(cols_lower & set(_CANONICAL_ALIASES["prompt"]))
    has_input = bool(cols_lower & set(_CANONICAL_ALIASES["item"]))
    has_score = bool(cols_lower & set(_SCORE_ALIASES))
    has_model = bool(cols_lower & set(_CANONICAL_ALIASES["model"]))
    return "long" if (has_score and has_input and (has_template or has_model)) else "wide"


def _from_wide(
    df: pd.DataFrame,
    *,
    report: DataLoadReport,
    strict_complete_design: bool,
) -> BenchmarkResult:
    if df.shape[1] < 3:
        raise ValueError(
            "wide format requires at least 3 columns: input + >=2 prompt/template columns"
        )

    input_col = df.columns[0]
    template_cols = list(df.columns[1:])
    raw_scores = df[template_cols]
    score_values = raw_scores.to_numpy()
    scores = raw_scores.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T

    coerced = int(np.sum(pd.isna(scores.T) & ~pd.isna(score_values)))
    report.score_non_numeric_coerced += coerced
    if coerced > 0:
        report.notes.append(
            f"coerced {coerced} non-numeric score value(s) to NaN in wide format"
        )

    template_labels = [str(c) for c in template_cols]
    input_labels = [str(v) for v in df[input_col].tolist()]

    _check_missing(
        scores,
        template_labels,
        input_labels,
        strict_complete_design=strict_complete_design,
    )
    return BenchmarkResult(
        scores=scores,
        template_labels=template_labels,
        input_labels=input_labels,
    )


def _from_long(
    df: pd.DataFrame,
    *,
    report: DataLoadReport,
    repair: bool,
    strict_complete_design: bool,
) -> ResultType:
    template_col = _find_col(df, _CANONICAL_ALIASES["prompt"])
    input_col = _find_col(df, _CANONICAL_ALIASES["item"])
    score_col = _find_col(df, _SCORE_ALIASES)
    model_col = _find_col(df, _CANONICAL_ALIASES["model"])
    run_col = _find_col(df, _CANONICAL_ALIASES["run"])
    # No canonical "evaluator" role in loader.py's alias table (it's an
    # io.py-specific concept), so this one stays local.
    evaluator_col = _find_col(df, ["evaluator", "eval", "judge", "criterion", "metric_name"])

    inject_implicit_template = (
        template_col is None
        and model_col is not None
        and input_col is not None
        and score_col is not None
    )

    if (template_col is None and not inject_implicit_template) or input_col is None or score_col is None:
        raise ValueError(
            "long format requires prompt/template (or model-only with implicit prompt), input, and score columns "
            "(aliases supported; see README quick start)"
        )

    working = df.copy()
    if inject_implicit_template:
        template_col = _IMPLICIT_TEMPLATE_COL
        working[template_col] = _IMPLICIT_TEMPLATE_LABEL
        report.notes.append(
            "missing prompt/template column; injected implicit template label "
            f"'{_IMPLICIT_TEMPLATE_LABEL}' for model/input/score long format"
        )

    raw = working[score_col]
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    report.score_non_numeric_coerced += int(np.sum(working[score_col].isna() & raw.notna()))

    working[template_col] = working[template_col].astype(str)
    working[input_col] = working[input_col].astype(str)
    template_labels = list(dict.fromkeys(str(v) for v in working[template_col]))
    input_labels = list(dict.fromkeys(str(v) for v in working[input_col]))

    evaluator_labels = None
    if evaluator_col is not None:
        working[evaluator_col] = working[evaluator_col].astype(str)
        evaluator_labels = list(dict.fromkeys(str(v) for v in working[evaluator_col]))

    if model_col is not None:
        working[model_col] = working[model_col].astype(str)
        model_labels = list(dict.fromkeys(str(v) for v in working[model_col]))
        scores = _pivot_multi_model(
            working,
            model_col=model_col,
            template_col=template_col,
            input_col=input_col,
            score_col=score_col,
            run_col=run_col,
            evaluator_col=evaluator_col,
            model_labels=model_labels,
            template_labels=template_labels,
            input_labels=input_labels,
            evaluator_labels=evaluator_labels,
            report=report,
            repair=repair,
            strict_complete_design=strict_complete_design,
        )
        return MultiModelBenchmark(
            scores=scores,
            model_labels=model_labels,
            template_labels=template_labels,
            input_labels=input_labels,
            evaluator_names=evaluator_labels or ["score"],
        )

    scores = _pivot_single_model(
        working,
        template_col=template_col,
        input_col=input_col,
        score_col=score_col,
        run_col=run_col,
        evaluator_col=evaluator_col,
        template_labels=template_labels,
        input_labels=input_labels,
        evaluator_labels=evaluator_labels,
        report=report,
        repair=repair,
        strict_complete_design=strict_complete_design,
    )
    return BenchmarkResult(
        scores=scores,
        template_labels=template_labels,
        input_labels=input_labels,
        evaluator_names=evaluator_labels or ["score"],
    )


def _count_duplicate_groups(df: pd.DataFrame, by: list[str]) -> int:
    counts = df.groupby(by, dropna=False).size()
    return int((counts > 1).sum())


def _pivot_single_model(
    df: pd.DataFrame,
    *,
    template_col: str,
    input_col: str,
    score_col: str,
    run_col: Optional[str],
    evaluator_col: Optional[str],
    template_labels: list[str],
    input_labels: list[str],
    evaluator_labels: Optional[list[str]],
    report: DataLoadReport,
    repair: bool,
    strict_complete_design: bool,
) -> np.ndarray:
    N, M = len(template_labels), len(input_labels)
    tpl_idx = {t: i for i, t in enumerate(template_labels)}
    inp_idx = {inp: j for j, inp in enumerate(input_labels)}

    if evaluator_labels is not None:
        K = len(evaluator_labels)
        eval_idx = {e: k for k, e in enumerate(evaluator_labels)}
        if run_col is not None:
            dup = _count_duplicate_groups(df, [template_col, input_col, run_col, evaluator_col])
            report.duplicate_groups_collapsed += dup
            run_labels = sorted(df[run_col].astype(str).unique().tolist())
            run_idx = {r: k for k, r in enumerate(run_labels)}
            scores = np.full((N, M, len(run_labels), K), np.nan)
            grp = df.assign(**{run_col: df[run_col].astype(str)}).groupby(
                [template_col, input_col, run_col, evaluator_col]
            )[score_col].mean()
            for (tpl, inp, run, ev), val in grp.items():
                scores[tpl_idx[tpl], inp_idx[inp], run_idx[run], eval_idx[ev]] = val
            if repair:
                for k in range(K):
                    report.run_nan_values_filled += _fill_missing_runs(scores[:, :, :, k])
            _check_missing(
                scores[:, :, 0, 0],
                template_labels,
                input_labels,
                strict_complete_design=strict_complete_design,
            )
            return scores

        dup = _count_duplicate_groups(df, [template_col, input_col, evaluator_col])
        report.duplicate_groups_collapsed += dup
        scores = np.full((N, M, 1, K), np.nan)
        grp = df.groupby([template_col, input_col, evaluator_col])[score_col].mean()
        for (tpl, inp, ev), val in grp.items():
            scores[tpl_idx[tpl], inp_idx[inp], 0, eval_idx[ev]] = val
        _check_missing(
            scores[:, :, 0, 0],
            template_labels,
            input_labels,
            strict_complete_design=strict_complete_design,
        )
        return scores

    if run_col is not None:
        dup = _count_duplicate_groups(df, [template_col, input_col, run_col])
        report.duplicate_groups_collapsed += dup
        run_labels = sorted(df[run_col].astype(str).unique().tolist())
        run_idx = {r: k for k, r in enumerate(run_labels)}
        scores = np.full((N, M, len(run_labels)), np.nan)
        grp = df.assign(**{run_col: df[run_col].astype(str)}).groupby(
            [template_col, input_col, run_col]
        )[score_col].mean()
        for (tpl, inp, run), val in grp.items():
            scores[tpl_idx[tpl], inp_idx[inp], run_idx[run]] = val
        if repair:
            report.run_nan_values_filled += _fill_missing_runs(scores)
        _check_missing(
            scores[:, :, 0],
            template_labels,
            input_labels,
            strict_complete_design=strict_complete_design,
        )
        return scores

    dup = _count_duplicate_groups(df, [template_col, input_col])
    report.duplicate_groups_collapsed += dup
    scores = np.full((N, M), np.nan)
    grp = df.groupby([template_col, input_col])[score_col].mean()
    for (tpl, inp), val in grp.items():
        scores[tpl_idx[tpl], inp_idx[inp]] = val
    _check_missing(
        scores,
        template_labels,
        input_labels,
        strict_complete_design=strict_complete_design,
    )
    return scores


def _pivot_multi_model(
    df: pd.DataFrame,
    *,
    model_col: str,
    template_col: str,
    input_col: str,
    score_col: str,
    run_col: Optional[str],
    evaluator_col: Optional[str],
    model_labels: list[str],
    template_labels: list[str],
    input_labels: list[str],
    evaluator_labels: Optional[list[str]],
    report: DataLoadReport,
    repair: bool,
    strict_complete_design: bool,
) -> np.ndarray:
    P, N, M = len(model_labels), len(template_labels), len(input_labels)
    model_idx = {m: p for p, m in enumerate(model_labels)}
    tpl_idx = {t: i for i, t in enumerate(template_labels)}
    inp_idx = {inp: j for j, inp in enumerate(input_labels)}

    if evaluator_labels is not None:
        K = len(evaluator_labels)
        eval_idx = {e: k for k, e in enumerate(evaluator_labels)}
        if run_col is not None:
            dup = _count_duplicate_groups(df, [model_col, template_col, input_col, run_col, evaluator_col])
            report.duplicate_groups_collapsed += dup
            run_labels = sorted(df[run_col].astype(str).unique().tolist())
            run_idx = {r: k for k, r in enumerate(run_labels)}
            scores = np.full((P, N, M, len(run_labels), K), np.nan)
            grp = df.assign(**{run_col: df[run_col].astype(str)}).groupby(
                [model_col, template_col, input_col, run_col, evaluator_col]
            )[score_col].mean()
            for (mdl, tpl, inp, run, ev), val in grp.items():
                scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp], run_idx[run], eval_idx[ev]] = val
            if repair:
                for p in range(P):
                    for k in range(K):
                        report.run_nan_values_filled += _fill_missing_runs(scores[p, :, :, :, k])
            for p, mdl in enumerate(model_labels):
                _check_missing(
                    scores[p, :, :, 0, 0],
                    template_labels,
                    input_labels,
                    context=f"model '{mdl}'",
                    strict_complete_design=strict_complete_design,
                )
            return scores

        dup = _count_duplicate_groups(df, [model_col, template_col, input_col, evaluator_col])
        report.duplicate_groups_collapsed += dup
        scores = np.full((P, N, M, 1, K), np.nan)
        grp = df.groupby([model_col, template_col, input_col, evaluator_col])[score_col].mean()
        for (mdl, tpl, inp, ev), val in grp.items():
            scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp], 0, eval_idx[ev]] = val
        for p, mdl in enumerate(model_labels):
            _check_missing(
                scores[p, :, :, 0, 0],
                template_labels,
                input_labels,
                context=f"model '{mdl}'",
                strict_complete_design=strict_complete_design,
            )
        return scores

    if run_col is not None:
        dup = _count_duplicate_groups(df, [model_col, template_col, input_col, run_col])
        report.duplicate_groups_collapsed += dup
        run_labels = sorted(df[run_col].astype(str).unique().tolist())
        run_idx = {r: k for k, r in enumerate(run_labels)}
        scores = np.full((P, N, M, len(run_labels)), np.nan)
        grp = df.assign(**{run_col: df[run_col].astype(str)}).groupby(
            [model_col, template_col, input_col, run_col]
        )[score_col].mean()
        for (mdl, tpl, inp, run), val in grp.items():
            scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp], run_idx[run]] = val
        if repair:
            report.run_nan_values_filled += _fill_missing_runs(scores)
        for p, mdl in enumerate(model_labels):
            _check_missing(
                scores[p, :, :, 0],
                template_labels,
                input_labels,
                context=f"model '{mdl}'",
                strict_complete_design=strict_complete_design,
            )
        return scores

    dup = _count_duplicate_groups(df, [model_col, template_col, input_col])
    report.duplicate_groups_collapsed += dup
    scores = np.full((P, N, M), np.nan)
    grp = df.groupby([model_col, template_col, input_col])[score_col].mean()
    for (mdl, tpl, inp), val in grp.items():
        scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp]] = val
    for p, mdl in enumerate(model_labels):
        _check_missing(
            scores[p],
            template_labels,
            input_labels,
            context=f"model '{mdl}'",
            strict_complete_design=strict_complete_design,
        )
    return scores


def _fill_missing_runs(scores: np.ndarray) -> int:
    """Fill NaN run slots in-place with the mean over available runs."""
    filled = 0
    it = np.nditer(scores[..., 0], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        cell = scores[idx]
        if np.any(np.isnan(cell)) and not np.all(np.isnan(cell)):
            missing_count = int(np.isnan(cell).sum())
            scores[idx][np.isnan(cell)] = float(np.nanmean(cell))
            filled += missing_count
        it.iternext()
    return filled


def _check_missing(
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
    *,
    context: str = "",
    strict_complete_design: bool,
) -> None:
    if not np.any(np.isnan(scores_2d)):
        return
    if not strict_complete_design:
        return

    missing = []
    for i, tpl in enumerate(template_labels):
        for j, inp in enumerate(input_labels):
            if np.isnan(scores_2d[i, j]):
                missing.append(f"  ({tpl!r}, {inp!r})")
    ctx = f" [{context}]" if context else ""
    raise ValueError(
        f"Incomplete design{ctx}: {len(missing)} missing (prompt, input) combination(s).\n"
        "All prompts must be evaluated on all inputs. You can either: "
        "(1) fill missing cells, (2) set strict_complete_design=False to keep NaNs, "
        "or (3) run analyze(..., method='lmm') for missing-aware modeling.\n"
        "Missing cells:\n"
        + "\n".join(missing[:10])
        + ("\n  ..." if len(missing) > 10 else "")
    )
