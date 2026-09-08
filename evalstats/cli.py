"""Command-line interface for evalstats.

Entry point declared in pyproject.toml::

    [project.scripts]
    evalstats = "evalstats.cli:main"

Usage::

    evalstats analyze data.csv
    evalstats analyze data.xlsx --sheet "Results"
    evalstats analyze data.csv --ci 0.90 --n-bootstrap 5000
    evalstats analyze data.csv --evaluator-mode per_evaluator

    evalstats label data.csv --metric llm_score
    evalstats label data.csv --metric llm_score --n-lab 20 --interactive
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from evalstats.config import set_alpha_ci, MIN_SAMPLE_FLOOR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "label":
        _cmd_label(args)
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

_LABEL_DESCRIPTION = """\
Picks which rows of your data need a human grade -- a genuinely random
sample, not "the ones I happened to eyeball" -- and, if you want, lets you
grade them right here in the terminal.

WHY THIS EXISTS: if you're using an LLM to score/judge your data and later
want to statistically correct for the judge's mistakes (judge_alignment()
and compare()'s PPI correction), the human-labeled subset has to be a
random sample of the full dataset. Hand-picking "the items I wasn't sure
about" -- the natural instinct -- breaks that assumption and silently
biases the correction. This command does the random part for you.

CONCRETE SCENARIOS -- what your spreadsheet can look like:

  1) Comparing several prompts/models on the SAME questions (the common
     case -- one row per (condition, item), item ids repeat across every
     condition):

         model,     item, llm_score
         baseline,     0,      0.8
         baseline,     1,      0.6
         cot,          0,      0.9
         cot,          1,      0.7
         ...

     -> picks --n-lab items ONCE and reuses them across every condition
        (15 items selected = 15 x n_conditions rows marked), since the same
        item ids repeat across models/prompts here. This is what the paper
        example (factor='model') looks like.

  2) A between-subjects study -- each participant/item appears under only
     ONE condition, so there's nothing to share across conditions:

         condition,  participant, helpfulness
         control,    p001,        3
         treatment,  p002,        5
         ...

     -> samples --n-lab participants independently WITHIN each condition
        instead (this needs --factor condition --item-col participant,
        since those column names aren't auto-detected -- see FILE FORMAT
        below).

  3) You don't have LLM judge scores yet -- you just want to sample and
     hand-label some ground truth first. Common for HCI researchers
     collecting labels before a judge model even exists. Omit --metric
     entirely, and declare what kind of grade you'll give with
     --score-type (there's no judge column to guess it from):

         model,     item, response_text
         gpt-4o,       0, "..."
         claude-3,     0, "..."
         ...

     -> samples items the same way, creates one generic human_label
        column instead of one per metric.

  4) Several judge metrics on the same content (e.g. accuracy AND
     fluency) -- pass --metric more than once; one round of grading
     covers every metric on the same sampled items:

         model, item, accuracy, fluency
         gpt-4o,   0,      0.8,     4.2
         ...

     -> --metric accuracy fluency shares one sampled item set, but each
        metric gets its own human_<metric> column.

Re-running on an already-marked file is safe -- it tops up any condition
still short of --n-lab without disturbing prior selections or labels
already filled in, so --interactive sessions can be stopped and resumed
freely.
"""

_LABEL_EPILOG = """\
FILE FORMAT
-----------

Deliberately looser than `analyze`'s: any CSV/XLSX with a numeric metric
column works (or no metric column at all -- see scenario 3 above). No
'run' column, no duplicate-row restriction, and this also accepts
between-subjects data with no shared item id across conditions -- it
doesn't need a full BenchmarkResult, just enough structure to sample from.

Column auto-detection (case-insensitive), same aliases load_from() uses,
except metric columns -- those are always given explicitly via --metric
(never auto-detected), and score types -- those are auto-detected from an
existing --metric column, or declared via --score-type when there isn't one:

  item column   : item, input, example, id, input_label
  factor column : model, model_label, model_name,
                  prompt, template, prompt_template

Both are optional and fall back gracefully when not found or not given:
  no item column   -> each row is treated as its own item (forces
                       independent per-condition sampling -- there's no
                       shared identity to reuse across conditions)
  no factor column -> every row is treated as one group

--factor/--item-col override auto-detection; use them when your columns
don't match the aliases above (as in scenario 2), or when the confirmation
prompt shows the wrong design.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalstats",
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
        default=None,
        metavar="FLOAT",
        help=(
            "Confidence level for intervals. If omitted, uses the project-wide "
            "default from evalstats.config.get_alpha_ci() (0.95)."
        ),
    )
    analyze.add_argument(
        "--ci-style",
        choices=["gradient", "line"],
        default="gradient",
        metavar="STYLE",
        help=(
            "Terminal CI plot style for printed summaries: 'gradient' (default) "
            "or 'line'. Choosing 'line' also skips computing multi-band "
            "gradient CI data (multi_ci)."
        ),
    )
    analyze.add_argument(
        "--method",
        choices=[
            "auto",
            "bootstrap",
            "bca",
            "bayes_bootstrap",
            "smooth_bootstrap",
            "permutation",
            "sign_test",
            "lmm",
            "bayes_binary",
            "wilson",
            "newcombe",
            "tango",
            "mj_floor",
            "bonett_price",
        ],
        default="auto",
        metavar="METHOD",
        help=(
            "Inference method (default: auto). Use 'lmm' for mixed-effects modeling; "
            "binary-only modes include 'bayes_binary', 'wilson', 'newcombe', "
            "'mj_floor' (floored May & Johnson) and 'tango' (the exact Tango score interval)."
        ),
    )
    analyze.add_argument(
        "--backend",
        choices=["statsmodels", "pymer4"],
        default="statsmodels",
        metavar="BACKEND",
        help=(
            "LMM backend when --method lmm (default: statsmodels). "
            "Ignored for non-LMM methods."
        ),
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
        choices=["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"],
        default="auto",
        help=(
            "Multiple-comparisons p-value correction (default: auto, matching "
            "analyze()'s own default -- resolves to 'shaffer' or 'romano_wolf' "
            "depending on N and data shape, never 'fdr_bh')."
        ),
    )
    analyze.add_argument(
        "--reference",
        default="grand_mean",
        metavar="LABEL",
        help=(
            "Template label to report advantages relative to, instead of the "
            "grand mean (default: grand_mean)."
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
        "--spread-percentiles",
        nargs=2,
        type=float,
        default=(10.0, 90.0),
        metavar=("LOW", "HIGH"),
        help=(
            "Percentiles for the per-input spread band shown alongside the CI "
            "(default: 10 90)."
        ),
    )
    analyze.add_argument(
        "--statistic",
        choices=["mean", "median"],
        default="mean",
        metavar="STAT",
        help="Central tendency for estimates and resampling (default: mean).",
    )
    analyze.add_argument(
        "--template-model-collapse",
        choices=["mean", "as_runs"],
        default="as_runs",
        metavar="MODE",
        help=(
            "Multi-model template collapse mode: 'mean' or 'as_runs' (default: as_runs)."
        ),
    )
    analyze.add_argument(
        "--simultaneous-ci",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use simultaneous (family-wise) pairwise CIs (default: enabled). "
            "Use --no-simultaneous-ci for marginal CIs."
        ),
    )
    analyze.add_argument(
        "--omnibus",
        action="store_true",
        help="Run an omnibus test in addition to pairwise comparisons.",
    )
    analyze.add_argument(
        "--p-values",
        action="store_true",
        default=False,
        help=(
            "Show p-values in pairwise comparison tables. The test used is "
            "determined by --pairwise-test (default: auto). When --omnibus is "
            "also set, 'auto' selects Wilcoxon signed-rank as the Friedman "
            "post-hoc; otherwise bootstrap p-values are shown for bootstrap "
            "methods and Wilcoxon for LMM/other methods."
        ),
    )
    analyze.add_argument(
        "--pairwise-test",
        choices=["auto", "bootstrap", "wilcoxon", "nemenyi"],
        default="auto",
        metavar="TEST",
        help=(
            "Pairwise p-value test to use when --p-values is enabled (or when "
            "this flag is set explicitly, which also enables p-values). "
            "Choices: 'auto' (default), 'bootstrap', 'wilcoxon', 'nemenyi'."
        ),
    )
    analyze.add_argument(
        "--show-rank-probabilities",
        action="store_true",
        default=False,
        help=(
            "Print the bootstrap 'Rank Probabilities' block (P(Best)/E[Rank] "
            "per entity). Off by default: a P(Best) figure reads as a "
            "confident, near-authoritative verdict even when entities are "
            "statistically indistinguishable once you look at the CIs next "
            "to it, so this is opt-in rather than opt-out."
        ),
    )
    analyze.add_argument(
        "--top-pairwise",
        type=int,
        default=5,
        metavar="INT",
        help="Number of pairwise comparisons to show in summary (default: 5).",
    )
    analyze.add_argument(
        "--brief",
        action="store_true",
        help=(
            "Print only the executive leaderboard (entity names, significance groups, "
            "means, CIs, verdicts). Omits the full statistical breakdown — interval "
            "plots, pairwise tables, and robustness section. Useful for a quick result "
            "at a glance. Use --out to save the full analysis alongside."
        ),
    )
    analyze.add_argument(
        "--out",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Optional output artifact paths. Supported suffixes: .md/.txt (summary), "
            ".json (structured analysis), and .png (robustness interval plot)."
        ),
    )
    analyze.add_argument(
        "--human-groundtruth",
        default=None,
        metavar="COL",
        help=(
            "Column of human labels for a random subset of rows (blank elsewhere), "
            "e.g. the human_<metric> column written by `evalstats label`. Marks "
            "--metric as an untrusted LLM-judge score: judge_alignment() runs first "
            "and prints its report, then every estimate is PPI-corrected. Long "
            "format only."
        ),
    )
    analyze.add_argument(
        "--metric",
        default=None,
        metavar="COL",
        help="The LLM-judge score column. Required with --human-groundtruth.",
    )
    analyze.add_argument(
        "--factor",
        default=None,
        metavar="COL",
        help=(
            "Condition column to compare with --human-groundtruth (default: 'model', "
            "else 'prompt')."
        ),
    )
    analyze.add_argument(
        "--label-selection",
        choices=["random", "unknown"],
        default="unknown",
        metavar="HOW",
        help=(
            "How the --human-groundtruth rows were chosen: 'random' if sampled "
            "uniformly (e.g. by `evalstats label`), else 'unknown' (default). PPI "
            "correction assumes a random sample."
        ),
    )
    analyze.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for resampling on the --human-groundtruth path.",
    )
    analyze.add_argument(
        "--score-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "The metric's true (min, max), e.g. `--score-range 1 5` for a 1-5 Likert "
            "scale or `--score-range 0 100` for a percentage. Selects the bounded, "
            "better-calibrated CI methods (logit-t / NIG). When omitted, evalstats "
            "infers the data kind from the values and says what it assumed."
        ),
    )

    label = sub.add_parser(
        "label",
        help="Randomly sample items for human labeling (for judge_alignment()/PPI).",
        description=_LABEL_DESCRIPTION,
        epilog=_LABEL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    label.add_argument(
        "file",
        type=Path,
        help=(
            "Path to a CSV or XLSX file. Looser than 'analyze': any numeric metric "
            "column works, no item/factor column is required -- see FILE FORMAT below."
        ),
    )
    label.add_argument(
        "--metric",
        nargs="*",
        default=[],
        metavar="COL",
        help=(
            "LLM-judge score column(s) to validate against human labels. Optional -- "
            "omit entirely to sample/label ground truth before you have any judge "
            "column at all (see scenario 3 above); pass --score-type in that case, "
            "since there's no judge score to detect it from. Multiple metrics share "
            "one sampled item set (one round of grading covers every metric on the "
            "same content) but each gets its own human_<metric> column."
        ),
    )
    label.add_argument(
        "--score-type",
        nargs="+",
        default=None,
        choices=["binary", "likert", "continuous"],
        metavar="TYPE",
        help=(
            "Declare the grading scale for --interactive, instead of auto-detecting "
            "it from an existing --metric column. Required when --metric is omitted "
            "(nothing to auto-detect from); optional otherwise, to override a wrong "
            "guess. Pass one value to apply to every metric, or one per --metric in "
            "the same order. binary=0/1, likert=small integer (e.g. 1-5), "
            "grade=0-100, continuous=any number."
        ),
    )
    label.add_argument(
        "--factor",
        default=None,
        metavar="COL",
        help=(
            "Condition/factor column (e.g. 'model' or 'prompt'). Auto-detected via "
            "the same column aliases load_from() uses when omitted."
        ),
    )
    label.add_argument(
        "--item-col",
        default=None,
        metavar="COL",
        help="Item/input identifier column. Auto-detected when omitted.",
    )
    label.add_argument(
        "--n-lab",
        type=int,
        default=15,
        metavar="INT",
        help="Target number of labeled items per condition (default: 15).",
    )
    label.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Random seed for sampling. If omitted, one is generated and printed -- "
            "record it for reproducibility."
        ),
    )
    label.add_argument(
        "--human-prefix",
        default="human_",
        metavar="STR",
        help="Prefix for the created human-label column(s) (default: 'human_').",
    )
    label.add_argument(
        "--sheet",
        default="0",
        metavar="SHEET",
        help="Sheet name or 0-based index for XLSX files (default: 0).",
    )
    label.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "Output file path. Defaults to '<name>_for_labeling<ext>' next to the "
            "input file, so the original is never silently overwritten."
        ),
    )
    label.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move sampled (marked) rows to the top of the output file, so a "
            "labeler doesn't have to scroll through the whole eval set to find "
            "them (default: on). Original relative row order is otherwise "
            "preserved within both groups. Pass --no-sort to keep the input "
            "file's row order untouched."
        ),
    )
    label.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Grade the sampled items right here in the terminal after marking them. "
            "Never shows the LLM judge's own score for the metric being graded, to "
            "avoid anchoring the human rater on it. Saves after every answer; 'q' "
            "quits and saves, 's' skips an item. Re-running with --interactive "
            "resumes on whatever's still ungraded."
        ),
    )
    label.add_argument(
        "-y", "--yes",
        action="store_true",
        help=(
            "Skip the 'does this match your experimental design?' confirmation "
            "prompt (auto-detected item/factor columns and paired/unpaired design). "
            "For scripted use; interactive terminal runs should leave this off."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> None:
    ci = getattr(args, "ci", None)
    if ci is not None:
        set_alpha_ci(1.0 - ci)

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
            "Or install with the xlsx extra:     pip install evalstats[xlsx]"
        )
    except Exception as exc:
        _die(f"could not read file: {exc}")

    print(f"  {len(df)} rows × {len(df.columns)} columns: {list(df.columns)}")

    if getattr(args, "human_groundtruth", None):
        _cmd_analyze_judge(args, df, ci)
        return

    # --- Detect / parse format ---
    from evalstats.io import from_dataframe
    from evalstats.loader import _SCORE_ALIASES

    # from_dataframe finds the score column by name and has no argument for
    # it, so --metric is applied by renaming. Without this the flag is read
    # only on the --human-groundtruth path above, and a table whose score
    # column is called something else fails _detect_format's has_score test,
    # gets parsed as wide, and reports every column as a missing prompt.
    metric_arg = getattr(args, "metric", None)
    if metric_arg:
        if metric_arg not in df.columns:
            _die(
                f"--metric '{metric_arg}' is not a column in this file.\n"
                f"  Available columns: {list(df.columns)}"
            )
        if metric_arg.strip().lower() not in _SCORE_ALIASES:
            clashes = [
                c for c in df.columns
                if c != metric_arg and c.strip().lower() in _SCORE_ALIASES
            ]
            if clashes:
                _die(
                    f"--metric '{metric_arg}' cannot be used while column "
                    f"'{clashes[0]}' is present: evalstats reads that one as the "
                    "score column too. Rename or drop one of them."
                )
            df = df.rename(columns={metric_arg: "score"})

    try:
        result, report = from_dataframe(
            df,
            format=args.format,
            return_report=True,
        )
    except Exception as exc:
        hint = ""
        if not ({c.strip().lower() for c in df.columns} & set(_SCORE_ALIASES)):
            hint = (
                "\n  No column is named like a score "
                f"({', '.join(_SCORE_ALIASES)}), so the table was read as wide "
                "format. If one of these columns holds the scores, name it with "
                f"--metric: {list(df.columns)}"
            )
        _die(f"could not parse data: {exc}{hint}")

    if args.format == "auto":
        print(f"  Detected format: {report.format_detected}")

    # --- Show what was loaded ---
    from evalstats.core.types import BenchmarkResult, MultiModelBenchmark

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

    # --- Enforce the documented minimum sample floor ---
    if result.n_inputs < MIN_SAMPLE_FLOOR:
        _die(
            f"only {result.n_inputs} input(s) per prompt/model -- evalstats "
            f"requires at least {MIN_SAMPLE_FLOOR} to report statistics (results "
            "below this floor are too noisy to be meaningful). Expand your eval set."
        )

    evaluator_mode = args.evaluator_mode

    # --- Validate --reference ---
    if args.reference != "grand_mean":
        if args.reference not in result.template_labels:
            _die(
                f"--reference '{args.reference}' not found in prompt template labels.\n"
                f"  Available: {result.template_labels}"
            )

    print()

    score_type, score_range = _resolve_score_kind(np.asarray(result.scores, dtype=float), _score_range_arg(args))

    # --- Run analysis ---
    from evalstats.core.router import analyze
    from evalstats.core.summary import print_analysis_summary

    analyze_kwargs = dict(
        evaluator_mode=evaluator_mode,
        reference=args.reference,
        method=getattr(args, "method", "auto"),
        backend=getattr(args, "backend", "statsmodels"),
        ci=ci,
        n_bootstrap=getattr(args, "n_bootstrap", 10_000),
        correction=args.correction,
        spread_percentiles=tuple(getattr(args, "spread_percentiles", (10, 90))),
        failure_threshold=getattr(args, "failure_threshold", None),
        statistic=getattr(args, "statistic", "mean"),
        template_model_collapse=getattr(args, "template_model_collapse", "as_runs"),
        simultaneous_ci=getattr(args, "simultaneous_ci", True),
        omnibus=getattr(args, "omnibus", False),
        p_values=getattr(args, "p_values", False),
        pairwise_test=getattr(args, "pairwise_test", "auto"),
        ci_style=getattr(args, "ci_style", "gradient"),
    )
    if score_range is not None:
        analyze_kwargs["score_range"] = score_range
    if score_type in ("likert", "continuous"):
        # The router still speaks eval_type; translate the way compare() does.
        analyze_kwargs["eval_type"] = score_type

    print("Running analysis ...", flush=True)
    try:
        with _quiet_score_kind_warnings():
            analysis = analyze(result, **analyze_kwargs)
    except (ValueError, NotImplementedError) as exc:
        _die(str(exc))

    print()
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        if getattr(args, "brief", False):
            from evalstats.core.summary import print_brief_summary
            print_brief_summary(analysis)
        else:
            print_analysis_summary(
                analysis,
                top_pairwise=args.top_pairwise,
                style=getattr(args, "ci_style", "gradient"),
                show_rank_probabilities=getattr(args, "show_rank_probabilities", False),
            )
    summary_text = summary_buffer.getvalue()
    print(summary_text, end="")

    out_paths = getattr(args, "out", None)
    if out_paths:
        if ci is None:
            from evalstats.config import get_alpha_ci

            ci_for_outputs = 1.0 - get_alpha_ci()
        else:
            ci_for_outputs = ci
        _write_outputs(
            out_paths=out_paths,
            summary_text=summary_text,
            analysis=analysis,
            reference=args.reference,
            n_bootstrap=args.n_bootstrap,
            ci=ci_for_outputs,
        )


def _score_range_arg(args: argparse.Namespace) -> Optional[tuple[float, float]]:
    sr = getattr(args, "score_range", None)
    if sr is None:
        return None
    lo, hi = float(sr[0]), float(sr[1])
    if not lo < hi:
        _die(f"--score-range LO HI needs LO < HI, got {lo:g} {hi:g}")
    return (lo, hi)


_SCORE_KIND_WARNING_RE = r"score_range|auto-detected|auto-detect"

# A grid spanning more than this many steps is a measurement that happens to
# be discrete, not an ordinal rating scale. 101 keeps whole-point percentage
# grades (0-100), the coarsest scale the Likert methods are calibrated for.
_MAX_ORDINAL_LEVELS = 101


@contextmanager
def _quiet_score_kind_warnings():
    """The engine warns about inferring the data kind; the CLI has already
    printed its own notice, so drop those specific warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*({_SCORE_KIND_WARNING_RE}).*", category=UserWarning)
        yield


def _resolve_score_kind(
    values: np.ndarray, score_range: Optional[tuple[float, float]],
) -> tuple[Optional[str], Optional[tuple[float, float]]]:
    """Decide the metric's score type and range for the engine, and say so.

    Returns ``(score_type, score_range)`` in compare()'s vocabulary:
    ``score_type`` is ``"likert"`` for values on a grid, ``"continuous"``
    otherwise, and ``None`` for binary data (which needs neither). The range
    is the one given, else ``(0, 1)`` for data already inside it, else the
    observed minimum and maximum. Prints a loud notice whenever the range
    was inferred, since that choice decides which CI family runs.
    """
    from evalstats.core.resampling import detect_quantization_step
    from evalstats.loader import _detect_score_type

    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None, score_range
    lo, hi = float(vals.min()), float(vals.max())
    n_distinct = int(np.unique(vals).size)
    observed = f"{n_distinct} distinct value{'s' if n_distinct != 1 else ''} in [{lo:g}, {hi:g}]"

    detected = _detect_score_type(pd.Series(vals))
    if detected == "binary" and score_range is None:
        print(f"Score type: binary 0/1 (detected)  |  observed: {observed}")
        print()
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = detect_quantization_step(vals.reshape(1, -1))
    # A grid only means "rating scale" when it is coarse enough to be one.
    # Token counts and latencies sit on a step-1 grid too; treating them as
    # ordinal would route them to the Likert methods, which are calibrated
    # for rubric-sized scales (up to a 0-100 grade), not for measurements
    # that merely happen to be whole numbers.
    n_levels = (hi - lo) / step + 1 if step else float("inf")
    ordinal = step is not None and n_levels <= _MAX_ORDINAL_LEVELS
    score_type = "likert" if ordinal else "continuous"
    if ordinal:
        kind_txt = f"discrete scores on a grid of {step:g} ({n_levels:.0f} levels)"
    elif step is not None:
        kind_txt = (f"numeric scores on a grid of {step:g}, too fine "
                    f"({n_levels:.0f} levels) to be a rating scale")
    else:
        kind_txt = "continuous scores"

    if score_range is not None:
        print(f"Score type: {score_type} ({kind_txt})  |  score range: [{score_range[0]:g}, "
              f"{score_range[1]:g}] (given)  |  observed: {observed}")
        print()
        return score_type, score_range

    if lo >= 0.0 and hi <= 1.0:
        # (0, 1) is the engine's own default for data inside it, so only the
        # type is declared -- but it IS declared, so the engine's uncapped
        # grid check can't reach a different verdict than the line above.
        print(f"Score type: {score_type} ({kind_txt}) in [0, 1] (detected)  |  observed: {observed}")
        print()
        return score_type, None

    bar = "!" * 72
    print(bar)
    print("NO --score-range GIVEN. evalstats inferred the data kind from the values:")
    print(f"  observed: {observed}")
    if lo == hi:
        print(f"  inferred: {kind_txt}, but every value is {lo:g}, so no range can be inferred;")
        print("            using the bounds-agnostic t-interval")
        print("Pass the metric's true range to use the bounded methods, e.g. --score-range 1 5.")
        print(bar)
        print()
        return score_type, None
    print(f"  inferred: {kind_txt}; using the observed minimum and maximum as the score")
    print(f"            range, --score-range {lo:g} {hi:g}, so the bounded, better-calibrated")
    print(f"            {'Likert' if score_type == 'likert' else 'continuous'} methods apply")
    print("If the metric's true range is wider (e.g. a 1-5 scale where nobody scored 5),")
    print(f"pass it: --score-range LO HI. If it is unbounded, pass nothing and ignore this.")
    print(bar)
    print()
    return score_type, (lo, hi)


def _cmd_analyze_judge(args: argparse.Namespace, df: pd.DataFrame, ci) -> None:
    """``analyze --human-groundtruth``: judge_alignment() then a PPI-corrected
    compare(), the CLI form of the API's ``alignment=`` path."""
    import evalstats as es
    from evalstats.alignment import judge_alignment

    metric = getattr(args, "metric", None)
    human_col = args.human_groundtruth
    if not metric:
        _die("--human-groundtruth needs --metric COL, the LLM-judge score column it validates.")
    for col, flag in ((metric, "--metric"), (human_col, "--human-groundtruth")):
        if col not in df.columns:
            _die(f"{flag} '{col}' not found in columns {list(df.columns)}")
    if metric == human_col:
        _die("--metric and --human-groundtruth must be different columns.")

    factor = getattr(args, "factor", None)
    if factor is None:
        factor = next((c for c in ("model", "prompt") if c in df.columns), None)
        if factor is None:
            _die("no 'model' or 'prompt' column; pass --factor COL to say what is being compared.")
    elif factor not in df.columns:
        _die(f"--factor '{factor}' not found in columns {list(df.columns)}")

    n_lab = int(df[human_col].notna().sum())
    if n_lab == 0:
        _die(f"--human-groundtruth '{human_col}' has no values; fill it in for a random subset of rows first.")

    try:
        evaldata = es.load_from(
            df, metric_cols=[metric, human_col],
            factors=None if factor in ("model", "prompt") else factor,
        )
    except Exception as exc:
        _die(f"could not load data for the judge path (long format only): {exc}")
    print(f"  {evaldata}")
    print(f"  Judge score: '{metric}'  |  human labels: '{human_col}' on {n_lab} rows  |  comparing: '{factor}'")
    item_col = getattr(evaldata, "_col", {}).get("item")
    n_items = int(df[item_col].nunique()) if item_col in df.columns else len(df)
    if n_items < MIN_SAMPLE_FLOOR:
        _die(
            f"only {n_items} item(s) -- evalstats requires at least "
            f"{MIN_SAMPLE_FLOOR} to report statistics. Expand your eval set."
        )
    print()

    score_type, score_range = _resolve_score_kind(
        pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float), _score_range_arg(args),
    )

    print("Checking judge alignment ...", flush=True)
    try:
        alignment = judge_alignment(
            evaldata, llm_metric=metric, human_groundtruth=human_col,
            selection=getattr(args, "label_selection", "unknown"),
        )
    except Exception as exc:
        _die(f"judge_alignment() failed: {exc}")
    alignment.summary()
    print()

    compare_kwargs = dict(
        factors=factor, metric=metric, alignment={metric: alignment},
        evaluator_mode=getattr(args, "evaluator_mode", "aggregate"),
        method=getattr(args, "method", "auto"),
        backend=getattr(args, "backend", "statsmodels"),
        n_bootstrap=getattr(args, "n_bootstrap", 10_000),
        correction=getattr(args, "correction", "auto"),
        spread_percentiles=tuple(getattr(args, "spread_percentiles", (10, 90))),
        failure_threshold=getattr(args, "failure_threshold", None),
        statistic=getattr(args, "statistic", "mean"),
        template_model_collapse=getattr(args, "template_model_collapse", "as_runs"),
        simultaneous_ci=getattr(args, "simultaneous_ci", True),
        omnibus=getattr(args, "omnibus", False),
        p_values=getattr(args, "p_values", False),
        pairwise_test=getattr(args, "pairwise_test", "auto"),
        ci_style=getattr(args, "ci_style", "gradient"),
    )
    reference = getattr(args, "reference", "grand_mean")
    if reference != "grand_mean":
        compare_kwargs["baseline"] = reference
    seed = getattr(args, "seed", None)
    if seed is not None:
        compare_kwargs["rng"] = seed
    if score_range is not None:
        compare_kwargs["score_range"] = score_range
    if score_type in ("likert", "continuous"):
        compare_kwargs["score_type"] = score_type

    print("Running PPI-corrected analysis ...", flush=True)
    try:
        with _quiet_score_kind_warnings():
            result = es.compare(evaldata, **compare_kwargs)
    except (ValueError, NotImplementedError) as exc:
        _die(str(exc))

    print()
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        if getattr(args, "brief", False):
            from evalstats.core.summary import print_brief_summary
            print_brief_summary(result.full_analysis)
        else:
            result.summary(
                top_pairwise=getattr(args, "top_pairwise", None),
                style=getattr(args, "ci_style", "gradient"),
                show_rank_probabilities=getattr(args, "show_rank_probabilities", False),
            )
    summary_text = summary_buffer.getvalue()
    print(summary_text, end="")

    out_paths = getattr(args, "out", None)
    if out_paths:
        if ci is None:
            from evalstats.config import get_alpha_ci
            ci_for_outputs = 1.0 - get_alpha_ci()
        else:
            ci_for_outputs = ci
        _write_outputs(
            out_paths=out_paths,
            summary_text=summary_text,
            analysis=result.full_analysis,
            reference=getattr(args, "reference", "grand_mean"),
            n_bootstrap=getattr(args, "n_bootstrap", 10_000),
            ci=ci_for_outputs,
        )


# ---------------------------------------------------------------------------
# label command
# ---------------------------------------------------------------------------

def _cmd_label(args: argparse.Namespace) -> None:
    from evalstats.labeling import (
        detect_design, describe_design, sample_for_labeling, run_interactive_labeling,
        MARKER_COL, GENERIC_LABEL_KEY, VALID_SCORE_TYPES,
    )

    metrics = list(getattr(args, "metric", None) or [])
    score_type_keys = metrics if metrics else [GENERIC_LABEL_KEY]

    # Argument-level validation first -- fails before touching the file at all.
    score_type_overrides: dict[str, str] = {}
    if args.score_type:
        if len(args.score_type) == 1:
            score_type_overrides = {k: args.score_type[0] for k in score_type_keys}
        elif len(args.score_type) == len(score_type_keys):
            score_type_overrides = dict(zip(score_type_keys, args.score_type))
        else:
            _die(
                f"--score-type expects 1 value (applied to all) or "
                f"{len(score_type_keys)} (one per --metric, in order); "
                f"got {len(args.score_type)}."
            )
    if (
        getattr(args, "interactive", False)
        and not metrics
        and GENERIC_LABEL_KEY not in score_type_overrides
    ):
        _die(
            "--interactive with no --metric needs --score-type -- there's no judge "
            f"column to auto-detect it from. Pass one of {VALID_SCORE_TYPES}."
        )

    path = args.file.expanduser().resolve()
    if not path.exists():
        _die(f"file not found: {path}")

    print(f"Loading {path.name} ...", flush=True)
    sheet = _parse_sheet(getattr(args, "sheet", "0"))
    try:
        df = _load_file(path, sheet=sheet)
    except ImportError as exc:
        _die(
            f"{exc}\n"
            "Install openpyxl for XLSX support:  pip install openpyxl\n"
            "Or install with the xlsx extra:     pip install evalstats[xlsx]"
        )
    except Exception as exc:
        _die(f"could not read file: {exc}")

    print(f"  {len(df)} rows × {len(df.columns)} columns: {list(df.columns)}")
    print()

    try:
        design = detect_design(
            df,
            metrics=metrics,
            factor=args.factor,
            item_col=args.item_col,
        )
    except ValueError as exc:
        _die(str(exc))

    print(describe_design(design))
    print()

    if not getattr(args, "yes", False):
        reply = input("Does this match your experimental design? [Y/n]: ").strip().lower()
        if reply not in ("", "y", "yes"):
            _die(
                "Aborted -- override auto-detection with --factor/--item-col "
                "if it got something wrong, then re-run."
            )

    try:
        marked_df, info = sample_for_labeling(
            design["df"],
            metrics=metrics,
            factor=design["factor_col"],
            item_col=design["item_col"],
            n_lab=args.n_lab,
            seed=args.seed,
            human_col_prefix=args.human_prefix,
            sort_labeled_first=getattr(args, "sort", True),
        )
    except ValueError as exc:
        _die(str(exc))

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else path.with_name(f"{path.stem}_for_labeling{path.suffix}")
    )

    def _save(d: pd.DataFrame) -> None:
        _write_table(d, out_path)

    _save(marked_df)

    print()
    print(f"Random seed used: {info['seed']}  (record this for reproducibility)")
    print(f"Marker column: '{MARKER_COL}'   Human label column(s): {list(info['human_cols'].values())}")
    print("Coverage (labeled/marked so far vs. target):")
    for lvl, n in info["coverage"].items():
        print(f"  {lvl}: {n}/{args.n_lab}")
    print(f"Wrote: {out_path}")

    if getattr(args, "interactive", False):
        try:
            run_interactive_labeling(
                marked_df, info, save_fn=_save, score_type_overrides=score_type_overrides
            )
        except ValueError as exc:
            _die(str(exc))
        print(f"Wrote: {out_path}")
    else:
        human_cols = list(info["human_cols"].values())
        print()
        print(
            "Hand this file to your labeler (fill in the human_* column(s) for "
            "marked rows), or re-run with --interactive to grade it here."
        )
        if metrics:
            print(
                f"Once labeled, run `evalstats analyze <file> --metric {metrics[0]} "
                f"--human-groundtruth {human_cols[0]} --label-selection random`, or call "
                f"judge_alignment(evaldata, llm_metric={metrics[0]!r}, "
                f"human_groundtruth={human_cols[0]!r}, selection='random')."
            )
        else:
            print(
                "No --metric given -- this samples ground-truth labels ahead of time. "
                "Once you also have LLM judge scores for this data, merge them in and call"
            )
            print(
                f"judge_alignment(evaldata, llm_metric=..., "
                f"human_groundtruth={human_cols[0]!r}, selection='random')."
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


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output file type '{suffix}'. Use .csv, .xlsx, or .xls."
        )


def _die(msg: str) -> None:
    sys.stdout.flush()
    print(f"evalstats error: {msg}", file=sys.stderr)
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
    from evalstats.core.router import AnalysisBundle, MultiModelBundle
    from evalstats.vis.point_estimates import plot_point_estimates

    for raw in out_paths:
        out_path = Path(raw).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            if suffix == ".md":
                content = "# evalstats analysis\n\n```text\n" + summary_text.rstrip() + "\n```\n"
            else:
                content = summary_text
            out_path.write_text(content, encoding="utf-8")
            print(f"Wrote summary: {out_path}")
            continue

        if suffix == ".json":
            payload = {
                "type": "evalstats.analysis",
                "summary": summary_text,
                "analysis": _to_builtin(analysis),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote JSON: {out_path}")
            continue

        if suffix == ".png":
            if isinstance(analysis, AnalysisBundle):
                fig = plot_point_estimates(
                    analysis.benchmark,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Wrote plot: {out_path}")
                continue
            if isinstance(analysis, MultiModelBundle):
                fig = plot_point_estimates(
                    analysis.model_level.benchmark,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    title="Model-Level Robustness Intervals",
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Wrote plot: {out_path}")
                continue
            if isinstance(analysis, dict):
                base = out_path.with_suffix("")
                for evaluator_name, evaluator_analysis in analysis.items():
                    target = base.with_name(f"{base.name}_{evaluator_name}").with_suffix(".png")
                    if isinstance(evaluator_analysis, MultiModelBundle):
                        fig = plot_point_estimates(
                            evaluator_analysis.model_level.benchmark,
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            title=f"Model-Level Robustness Intervals ({evaluator_name})",
                        )
                    else:
                        fig = plot_point_estimates(
                            evaluator_analysis.benchmark,
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            title=f"Robustness Intervals ({evaluator_name})",
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
