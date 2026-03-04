"""Command-line interface for promptstats.

Entry point declared in pyproject.toml::

    [project.scripts]
    promptstats = "promptstats.cli:main"

Usage::

    promptstats analyze data.csv
    promptstats analyze data.xlsx --sheet "Results"
    promptstats analyze data.csv --ci 0.90 --n-bootstrap 5000
    promptstats analyze data.csv --evaluator-mode per_evaluator
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Union

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

Wide format  (rows = inputs, columns = prompt templates):

    input,    Template A, Template B, Template C
    example_1,      0.85,       0.72,       0.91
    example_2,      0.63,       0.88,       0.77

  The first column contains input identifiers.  Each subsequent column is a
  prompt template.  All score values must be numeric.
  Multiple evaluators are not supported in wide format.

Long / tidy format  (one observation per row):

    Required columns (case-insensitive):
      prompt     — prompt template name
      input      — input identifier
      score      — numeric score

    Optional columns:
      evaluator  — evaluator name  (enables multi-evaluator analysis; use
                   --evaluator-mode to control how evaluators are combined)
      model      — model name      (enables multi-model analysis)
      run        — run index       (adds run dimension; ≥3 runs per cell
                                   enables seed-variance / instability metrics)

    Example – single model, one implicit evaluator, a single run:

        prompt,     input, score
        Template A,  ex_1,  0.85
        Template A,  ex_1,  0.91
        Template B,  ex_1,  0.72
        Template B,  ex_1,  0.88
        ...

    Example – single model, multiple evaluators, with multiple runs:

        prompt,     input, run,  evaluator, score
        Template A,  ex_1,   0,   accuracy,  0.85
        Template A,  ex_1,   0,   fluency,   0.91
        Template A,  ex_1,   1,   accuracy,  0.83
        ...

    Example – multi-model:

        model,  prompt,     input, score
        GPT-4,  Template A,  ex_1,  0.85
        Claude, Template A,  ex_1,  0.90
        ...

    Column name aliases (all case-insensitive):
      prompt    → template, prompt_template
      input     → example, item, id, input_label
      score     → value, result, metric
      evaluator → eval, judge, criterion, metric_name
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
        "--evaluator-mode",
        choices=["aggregate", "per_evaluator"],
        default="aggregate",
        metavar="MODE",
        help=(
            "How to handle multiple evaluators: 'aggregate' (default) averages scores "
            "across evaluators before analysis; 'per_evaluator' runs a separate full "
            "analysis for each evaluator and prints each in turn.  "
            "Only applies when an 'evaluator' column is present in the data."
        ),
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
    analyze.add_argument(
        "--out",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Optional output artifact paths. Supported suffixes: .md/.txt (summary), "
            ".json (structured analysis), and .png (mean-advantage plot)."
        ),
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
    from promptstats.io import from_dataframe

    try:
        result, report = from_dataframe(
            df,
            format=args.format,
            return_report=True,
        )
    except Exception as exc:
        _die(f"could not parse data: {exc}")

    if args.format == "auto":
        print(f"  Detected format: {report.format_detected}")

    # --- Show what was loaded ---
    from promptstats.core.types import BenchmarkResult, MultiModelBenchmark

    if isinstance(result, MultiModelBenchmark):
        runs_str = f" × {result.n_runs} runs" if result.n_runs > 1 else ""
        evals_str = f" × {result.n_evaluators} evaluators" if result.n_evaluators > 1 else ""
        print(
            f"  MultiModelBenchmark: {result.n_models} models × "
            f"{result.n_templates} prompts × {result.n_inputs} inputs{runs_str}{evals_str}"
        )
        print(f"  Models:    {result.model_labels}")
        print(f"  Prompts:   {result.template_labels}")
        if result.n_evaluators > 1:
            print(f"  Evaluators: {result.evaluator_names}")
    else:
        runs_str = f" × {result.n_runs} runs" if result.n_runs > 1 else ""
        evals_str = f" × {result.n_evaluators} evaluators" if result.n_evaluators > 1 else ""
        print(
            f"  BenchmarkResult: {result.n_templates} prompts × "
            f"{result.n_inputs} inputs{runs_str}{evals_str}"
        )
        print(f"  Prompts:   {result.template_labels}")
        if result.n_evaluators > 1:
            print(f"  Evaluators: {result.evaluator_names}")

    # --- Validate --evaluator-mode ---
    evaluator_mode = args.evaluator_mode

    # --- Validate --reference ---
    if args.reference != "grand_mean":
        if args.reference not in result.template_labels:
            _die(
                f"--reference '{args.reference}' not found in prompt template labels.\n"
                f"  Available: {result.template_labels}"
            )

    print()

    # --- Run analysis ---
    from promptstats.core.router import analyze, print_analysis_summary

    print("Running analysis ...", flush=True)
    try:
        analysis = analyze(
            result,
            evaluator_mode=evaluator_mode,
            ci=args.ci,
            n_bootstrap=args.n_bootstrap,
            correction=args.correction,
            reference=args.reference,
            failure_threshold=args.failure_threshold,
        )
    except (ValueError, NotImplementedError) as exc:
        _die(str(exc))

    print()
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        print_analysis_summary(analysis, top_pairwise=args.top_pairwise)
    summary_text = summary_buffer.getvalue()
    print(summary_text, end="")

    out_paths = getattr(args, "out", None)
    if out_paths:
        _write_outputs(
            out_paths=out_paths,
            summary_text=summary_text,
            analysis=analysis,
            reference=args.reference,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
        )


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


def _die(msg: str) -> None:
    sys.stdout.flush()
    print(f"promptstats error: {msg}", file=sys.stderr)
    sys.exit(1)


def _to_builtin(value):
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_outputs(
    *,
    out_paths: list[str],
    summary_text: str,
    analysis,
    reference: str,
    n_bootstrap: int,
    ci: float,
) -> None:
    from promptstats.core.router import AnalysisBundle, MultiModelBundle
    from promptstats.vis.advantage import plot_point_advantage

    for raw in out_paths:
        out_path = Path(raw).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            if suffix == ".md":
                content = "# promptstats analysis\n\n```text\n" + summary_text.rstrip() + "\n```\n"
            else:
                content = summary_text
            out_path.write_text(content, encoding="utf-8")
            print(f"Wrote summary: {out_path}")
            continue

        if suffix == ".json":
            payload = {
                "type": "promptstats.analysis",
                "summary": summary_text,
                "analysis": _to_builtin(analysis),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote JSON: {out_path}")
            continue

        if suffix == ".png":
            if isinstance(analysis, AnalysisBundle):
                fig = plot_point_advantage(
                    analysis.benchmark,
                    reference=reference,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Wrote plot: {out_path}")
                continue
            if isinstance(analysis, MultiModelBundle):
                fig = plot_point_advantage(
                    analysis.model_level.benchmark,
                    reference="grand_mean",
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    title="Model-Level Mean Advantage",
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Wrote plot: {out_path}")
                continue
            if isinstance(analysis, dict):
                base = out_path.with_suffix("")
                for evaluator_name, evaluator_analysis in analysis.items():
                    target = base.with_name(f"{base.name}_{evaluator_name}").with_suffix(".png")
                    if isinstance(evaluator_analysis, MultiModelBundle):
                        fig = plot_point_advantage(
                            evaluator_analysis.model_level.benchmark,
                            reference="grand_mean",
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            title=f"Model-Level Mean Advantage ({evaluator_name})",
                        )
                    else:
                        fig = plot_point_advantage(
                            evaluator_analysis.benchmark,
                            reference=reference,
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            title=f"Mean Advantage ({evaluator_name})",
                        )
                    fig.savefig(target, dpi=150, bbox_inches="tight")
                    print(f"Wrote plot: {target}")
                continue

        _die(
            f"unsupported output file extension for '{out_path.name}'. "
            "Use one of: .txt, .md, .json, .png"
        )


if __name__ == "__main__":
    main()
