"""Command-line interface for promptstats.

Entry point declared in pyproject.toml::

    [project.scripts]
    promptstats = "promptstats.cli:main"

Usage::

    promptstats analyze data.csv
    promptstats analyze data.xlsx --sheet "Results"
    promptstats analyze data.csv --ci 0.90 --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        _cmd_analyze(args)
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

_ANALYZE_EPILOG = """\
FILE FORMATS
------------

Wide format  (rows = inputs, columns = templates):

    input,    Template A, Template B, Template C
    example_1,      0.85,       0.72,       0.91
    example_2,      0.63,       0.88,       0.77

  The first column contains input identifiers.  Each subsequent column is a
  prompt template.  All score values must be numeric.

Long / tidy format  (one observation per row):

    Required columns (case-insensitive):
      prompt    — prompt template name
      input     — input identifier
      score     — numeric score

    Optional columns:
      model     — model name  (enables multi-model analysis)
      run       — run index   (adds run dimension; ≥3 runs per cell enables
                               seed-variance / instability metrics)

    Example – single model, with runs:

        prompt,         input, run, score
        Template A, example_1,   0,  0.85
        Template A, example_1,   1,  0.83
        Template A, example_1,   2,  0.87
        Template B, example_1,   0,  0.72
        ...

    Example – multi-model:

        model,      prompt,     input, score
        GPT-4,  Template A, example_1,  0.85
        Claude, Template A, example_1,  0.90
        ...

    Column name aliases (all case-insensitive):
      prompt    → template, prompt_template
      input     → example, item, id, input_label
      score     → value, result, metric
      model     → model_label, model_name
      run       → seed, repeat, run_id, trial
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptstats",
        description="Statistical analysis for comparing prompt and model performance on benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    analyze = sub.add_parser(
        "analyze",
        help="Load a dataset and run statistical analysis.",
        description="Run statistical analysis on a benchmark dataset.",
        epilog=_ANALYZE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze.add_argument(
        "file",
        type=Path,
        help="Path to a CSV or XLSX benchmark file.",
    )
    analyze.add_argument(
        "--format",
        choices=["auto", "wide", "long"],
        default="auto",
        metavar="FORMAT",
        help=(
            "Data format: 'wide' (rows=inputs, cols=prompt templates), 'long' (tidy "
            "format with prompt/input/score columns), or 'auto' (default)."
        ),
    )
    analyze.add_argument(
        "--sheet",
        default="0",
        metavar="SHEET",
        help="Sheet name or 0-based index for XLSX files (default: 0).",
    )
    analyze.add_argument(
        "--ci",
        type=float,
        default=0.95,
        metavar="FLOAT",
        help="Confidence level for bootstrap intervals (default: 0.95).",
    )
    analyze.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        metavar="INT",
        help="Number of bootstrap resamples (default: 10000).",
    )
    analyze.add_argument(
        "--correction",
        choices=["holm", "bonferroni", "fdr_bh", "none"],
        default="holm",
        help="Multiple-comparisons p-value correction (default: holm).",
    )
    analyze.add_argument(
        "--reference",
        default="grand_mean",
        metavar="LABEL",
        help=(
            "Reference for mean advantage plot. 'grand_mean' (default) compares "
            "each prompt template against the average. Pass a prompt template label to compare "
            "everything against a specific baseline."
        ),
    )
    analyze.add_argument(
        "--failure-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Report fraction of inputs scoring below this value (robustness table).",
    )
    analyze.add_argument(
        "--top-pairwise",
        type=int,
        default=5,
        metavar="INT",
        help="Number of pairwise comparisons to show in summary (default: 5).",
    )
    return parser


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> None:
    path = args.file.expanduser().resolve()
    if not path.exists():
        _die(f"file not found: {path}")

    # --- Load file ---
    print(f"Loading {path.name} ...", flush=True)
    sheet = _parse_sheet(args.sheet)
    try:
        df = _load_file(path, sheet=sheet)
    except ImportError as exc:
        _die(
            f"{exc}\n"
            "Install openpyxl for XLSX support:  pip install openpyxl\n"
            "Or install with the xlsx extra:     pip install promptstats[xlsx]"
        )
    except Exception as exc:
        _die(f"could not read file: {exc}")

    print(f"  {len(df)} rows × {len(df.columns)} columns: {list(df.columns)}")

    # --- Detect / parse format ---
    fmt = args.format
    if fmt == "auto":
        fmt = _detect_format(df)
        print(f"  Detected format: {fmt}")

    try:
        if fmt == "wide":
            result = _load_wide(df)
        else:
            result = _load_long(df)
    except Exception as exc:
        _die(f"could not parse data: {exc}")

    # --- Show what was loaded ---
    from promptstats.core.types import BenchmarkResult, MultiModelBenchmark

    if isinstance(result, MultiModelBenchmark):
        runs_str = f" × {result.n_runs} runs" if result.n_runs > 1 else ""
        print(
            f"  MultiModelBenchmark: {result.n_models} models × "
            f"{result.n_templates} prompts × {result.n_inputs} inputs{runs_str}"
        )
        print(f"  Models:    {result.model_labels}")
        print(f"  Prompt templates: {result.template_labels}")
    else:
        runs_str = f" × {result.n_runs} runs" if result.n_runs > 1 else ""
        print(
            f"  BenchmarkResult: {result.n_templates} prompts × "
            f"{result.n_inputs} inputs{runs_str}"
        )
        print(f"  Prompt templates: {result.template_labels}")

    # --- Validate --reference ---
    if args.reference != "grand_mean":
        all_labels = (
            result.template_labels
            if not isinstance(result, MultiModelBenchmark)
            else result.template_labels
        )
        if args.reference not in all_labels:
            _die(
                f"--reference '{args.reference}' not found in template labels.\n"
                f"  Available: {all_labels}"
            )

    print()

    # --- Run analysis ---
    from promptstats.core.router import analyze, print_analysis_summary

    print("Running analysis ...", flush=True)
    try:
        analysis = analyze(
            result,
            ci=args.ci,
            n_bootstrap=args.n_bootstrap,
            correction=args.correction,
            reference=args.reference,
            failure_threshold=args.failure_threshold,
        )
    except (ValueError, NotImplementedError) as exc:
        _die(str(exc))

    print()
    print_analysis_summary(analysis, top_pairwise=args.top_pairwise)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _parse_sheet(s: str) -> Union[int, str]:
    """Convert a sheet argument to int if it looks like a number, else str."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def _load_file(path: Path, sheet: Union[int, str] = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls", ".ods"):
        return pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Accepted formats: .csv, .xlsx, .xls, .ods"
        )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(df: pd.DataFrame) -> str:
    """Return 'long' if the DataFrame looks like tidy data, else 'wide'."""
    cols_lower = {c.lower().strip() for c in df.columns}
    has_template = bool(cols_lower & {"template", "prompt", "prompt_template"})
    has_score = bool(cols_lower & {"score", "value", "result", "metric"})
    return "long" if (has_template and has_score) else "wide"


def _find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return the first column name matching any alias (case-insensitive), or None."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


# ---------------------------------------------------------------------------
# Wide-format parser
# ---------------------------------------------------------------------------

def _load_wide(df: pd.DataFrame):
    """Parse a wide-format DataFrame into a BenchmarkResult.

    The first column contains input labels.  Every subsequent column is a
    prompt template, with numeric score values.
    """
    from promptstats.core.types import BenchmarkResult

    input_col = df.columns[0]
    template_cols = list(df.columns[1:])

    if len(template_cols) < 2:
        raise ValueError(
            f"Wide format needs at least 2 template columns; got {len(template_cols)}. "
            "The first column is treated as input labels."
        )

    input_labels = [str(v) for v in df[input_col].tolist()]
    template_labels = [str(c) for c in template_cols]

    scores = df[template_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
    # scores shape: (N, M) = (n_templates, n_inputs)

    _check_missing(scores, template_labels, input_labels)

    return BenchmarkResult(
        scores=scores,
        template_labels=template_labels,
        input_labels=input_labels,
    )


# ---------------------------------------------------------------------------
# Long-format parser
# ---------------------------------------------------------------------------

def _load_long(df: pd.DataFrame):
    """Parse a long/tidy-format DataFrame into a BenchmarkResult or MultiModelBenchmark."""
    from promptstats.core.types import BenchmarkResult, MultiModelBenchmark

    template_col = _find_col(df, ["template", "prompt", "prompt_template"])
    input_col    = _find_col(df, ["input", "example", "item", "id", "input_label"])
    score_col    = _find_col(df, ["score", "value", "result", "metric"])
    model_col    = _find_col(df, ["model", "model_label", "model_name"])
    run_col      = _find_col(df, ["run", "seed", "repeat", "run_id", "trial"])

    if template_col is None:
        raise ValueError(
            "Long format requires a 'template' column "
            "(accepted names: template, prompt, prompt_template)."
        )
    if input_col is None:
        raise ValueError(
            "Long format requires an 'input' column "
            "(accepted names: input, example, item, id, input_label)."
        )
    if score_col is None:
        raise ValueError(
            "Long format requires a 'score' column "
            "(accepted names: score, value, result, metric)."
        )

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Preserve first-occurrence order for labels
    template_labels = list(dict.fromkeys(str(v) for v in df[template_col]))
    input_labels    = list(dict.fromkeys(str(v) for v in df[input_col]))
    df[template_col] = df[template_col].astype(str)
    df[input_col]    = df[input_col].astype(str)

    if model_col is not None:
        model_labels = list(dict.fromkeys(str(v) for v in df[model_col]))
        df[model_col] = df[model_col].astype(str)
        scores = _pivot_multi_model(
            df, model_col, template_col, input_col, score_col, run_col,
            model_labels, template_labels, input_labels,
        )
        if scores.ndim == 4:
            run_labels = _run_labels(df, run_col)
            if len(run_labels) < 3:
                import warnings
                warnings.warn(
                    f"Only {len(run_labels)} run(s) found. "
                    "Seed-variance analysis requires R ≥ 3 runs per cell.",
                    UserWarning,
                )
        return MultiModelBenchmark(
            scores=scores,
            model_labels=model_labels,
            template_labels=template_labels,
            input_labels=input_labels,
        )
    else:
        scores = _pivot_single_model(
            df, template_col, input_col, score_col, run_col,
            template_labels, input_labels,
        )
        if scores.ndim == 3:
            run_labels = _run_labels(df, run_col)
            if len(run_labels) < 3:
                import warnings
                warnings.warn(
                    f"Only {len(run_labels)} run(s) found. "
                    "Seed-variance analysis requires R ≥ 3 runs per cell.",
                    UserWarning,
                )
        return BenchmarkResult(
            scores=scores,
            template_labels=template_labels,
            input_labels=input_labels,
        )


def _run_labels(df: pd.DataFrame, run_col: Optional[str]) -> list:
    if run_col is None:
        return []
    return sorted(df[run_col].unique().tolist())


def _pivot_single_model(
    df: pd.DataFrame,
    template_col: str,
    input_col: str,
    score_col: str,
    run_col: Optional[str],
    template_labels: list[str],
    input_labels: list[str],
) -> np.ndarray:
    N, M = len(template_labels), len(input_labels)
    tpl_idx = {t: i for i, t in enumerate(template_labels)}
    inp_idx = {inp: j for j, inp in enumerate(input_labels)}

    if run_col is None:
        # (N, M)
        grp = df.groupby([template_col, input_col])[score_col].mean()
        scores = np.full((N, M), np.nan)
        for (tpl, inp), val in grp.items():
            if tpl in tpl_idx and inp in inp_idx:
                scores[tpl_idx[tpl], inp_idx[inp]] = val
        _check_missing(scores, template_labels, input_labels)
        return scores
    else:
        # (N, M, R)
        df = df.copy()
        df[run_col] = df[run_col].astype(str)
        run_labels = sorted(df[run_col].unique().tolist())
        R = len(run_labels)
        run_idx = {r: k for k, r in enumerate(run_labels)}
        grp = df.groupby([template_col, input_col, run_col])[score_col].mean()
        scores = np.full((N, M, R), np.nan)
        for (tpl, inp, run), val in grp.items():
            if tpl in tpl_idx and inp in inp_idx and run in run_idx:
                scores[tpl_idx[tpl], inp_idx[inp], run_idx[run]] = val
        # Fill missing runs with cell mean so BenchmarkResult validation passes
        _fill_missing_runs(scores)
        _check_missing(scores[:, :, 0], template_labels, input_labels)
        return scores


def _pivot_multi_model(
    df: pd.DataFrame,
    model_col: str,
    template_col: str,
    input_col: str,
    score_col: str,
    run_col: Optional[str],
    model_labels: list[str],
    template_labels: list[str],
    input_labels: list[str],
) -> np.ndarray:
    P, N, M = len(model_labels), len(template_labels), len(input_labels)
    model_idx = {m: p for p, m in enumerate(model_labels)}
    tpl_idx   = {t: i for i, t in enumerate(template_labels)}
    inp_idx   = {inp: j for j, inp in enumerate(input_labels)}

    if run_col is None:
        # (P, N, M)
        grp = df.groupby([model_col, template_col, input_col])[score_col].mean()
        scores = np.full((P, N, M), np.nan)
        for (mdl, tpl, inp), val in grp.items():
            if mdl in model_idx and tpl in tpl_idx and inp in inp_idx:
                scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp]] = val
        for p, mdl in enumerate(model_labels):
            _check_missing(scores[p], template_labels, input_labels, context=f"model '{mdl}'")
        return scores
    else:
        # (P, N, M, R)
        df = df.copy()
        df[run_col] = df[run_col].astype(str)
        run_labels = sorted(df[run_col].unique().tolist())
        R = len(run_labels)
        run_idx = {r: k for k, r in enumerate(run_labels)}
        grp = df.groupby([model_col, template_col, input_col, run_col])[score_col].mean()
        scores = np.full((P, N, M, R), np.nan)
        for (mdl, tpl, inp, run), val in grp.items():
            if mdl in model_idx and tpl in tpl_idx and inp in inp_idx and run in run_idx:
                scores[model_idx[mdl], tpl_idx[tpl], inp_idx[inp], run_idx[run]] = val
        _fill_missing_runs(scores)
        for p, mdl in enumerate(model_labels):
            _check_missing(scores[p, :, :, 0], template_labels, input_labels, context=f"model '{mdl}'")
        return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_missing_runs(scores: np.ndarray) -> None:
    """Fill NaN run slots with the mean of available runs in the same cell.

    Operates on the last axis (runs).  Modifies in place.  Cells where
    *all* runs are NaN are left as NaN (caught later by _check_missing).
    """
    # scores may be (N, M, R) or (P, N, M, R)
    # The run axis is always the last one.
    it = np.nditer(scores[..., 0], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        cell = scores[idx]  # shape (R,)
        if np.any(np.isnan(cell)) and not np.all(np.isnan(cell)):
            cell_mean = float(np.nanmean(cell))
            scores[idx][np.isnan(cell)] = cell_mean
        it.iternext()


def _check_missing(
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
    context: str = "",
) -> None:
    """Raise a clear error if any (template, input) combination is missing."""
    if not np.any(np.isnan(scores_2d)):
        return
    missing = []
    for i, tpl in enumerate(template_labels):
        for j, inp in enumerate(input_labels):
            if np.isnan(scores_2d[i, j]):
                missing.append(f"  ({tpl!r}, {inp!r})")
    ctx = f" [{context}]" if context else ""
    raise ValueError(
        f"Incomplete design{ctx}: {len(missing)} missing (template, input) "
        f"combination(s).\n"
        "All templates must be evaluated on all inputs.  Missing cells:\n"
        + "\n".join(missing[:10])
        + ("\n  ..." if len(missing) > 10 else "")
    )


def _die(msg: str) -> None:
    sys.stdout.flush()
    print(f"promptstats error: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
