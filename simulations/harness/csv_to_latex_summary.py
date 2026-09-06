"""Recompute a ci_single/ci_paired-style overall-summary LaTeX table directly
from a saved ``*_results.csv`` file, without re-running the simulation.

Mirrors ``cases/ci_single.py``'s and ``cases/ci_paired.py``'s (identical)
``latex_overall_summary()`` aggregation exactly: same per-n-then-across-n
averaging (``_headline_cov_width_score``), same Monte-Carlo proportion band,
same combined-variance time-stat formula -- and the same fix applied there
for methods tested against more than one eval-type group (binary vs.
numeric/continuous+likert+grades): such a method gets **two rows**,
``"<method> (binary)"`` and ``"<method> (numeric)"``, each computed from only
that group's data, rather than one row that averages Cov/Width/Score across
two different scales/regimes and reports "all" in the Eval types column. No
row's Eval types column is ever "all" here. If the CSV has an ``is_null``
column (``ci_paired``'s null/placebo scenario rows), those rows are excluded
first, matching ``latex_overall_summary``'s own ``non_null`` filtering.

The output is then piped through ``revise_latex_tables.revise_table()``, so
it matches exactly what ``python -m simulations.harness.revise_latex_tables``
would have produced from a live run's ``*_summary.log`` -- same dropped MC
band column (summarized in the caption instead), same trimmed Time (ms)
column, same header renames, same bin/num abbreviation, same ``table*``
float.

Requires a CSV with ``total_score``/``mean_score`` and
``total_time``/``total_time_sq`` columns (added to ``save_results_artifacts``
2026-07-15) -- older CSVs predating that change won't have Score and will
error out with a clear message rather than silently omitting it or guessing.

Default LaTeX label is ``tab:overall_summary`` (override with --label).

Usage:
  python -m simulations.harness.csv_to_latex_summary path/to/..._results.csv
  python -m simulations.harness.csv_to_latex_summary path/to/..._results.csv \\
      --alpha 0.05 --caption "My table" --label "tab:my_table"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .latex_tables import booktabs_table, escape_latex, eval_type_group, eval_type_label
from .revise_latex_tables import revise_table

REQUIRED_COLUMNS = {
    "eval_type", "n", "method", "n_reps", "covered", "mean_width", "coverage",
    "mean_score", "total_time", "total_time_sq",
}


def _mc_proportion_stats(successes: float, total: float, z: float = 1.96) -> tuple[float, float, float, float]:
    """Same formula as cases/ci_single.py's _mc_proportion_stats."""
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _headline_cov_width_score(
    per_n_vals: dict[tuple[str, int], list[tuple[float, float, float]]],
    m: str,
    sizes_present: list[int],
) -> tuple[float, float, float]:
    """Same formula as cases/ci_single.py's _headline_cov_width_score: average
    per n first (one number per n, unweighted across whatever cells
    contributed at that n), then average those per-n numbers across n."""
    per_n_means = []
    for n in sizes_present:
        vals = per_n_vals.get((m, n))
        if vals:
            per_n_means.append((
                float(np.mean([v[0] for v in vals])),
                float(np.mean([v[1] for v in vals])),
                float(np.mean([v[2] for v in vals])),
            ))
    if not per_n_means:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean([c for c, _, _ in per_n_means])),
        float(np.mean([w for _, w, _ in per_n_means])),
        float(np.mean([s for _, _, s in per_n_means])),
    )


def _time_stats(sub: pd.DataFrame) -> tuple[float, float]:
    """Same combined-variance formula as cases/ci_single.py's _time_stats,
    applied to a (method, group)-filtered slice of raw CSV rows instead of
    SimResult objects."""
    total_reps = float(sub["n_reps"].sum())
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = float(sub["total_time"].sum())
    sum_t2 = float(sub["total_time_sq"].sum())
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    return avg * 1000.0, float(np.sqrt(var / total_reps)) * 1000.0


def build_latex_summary(df: pd.DataFrame, *, alpha: float, caption: str, label: str) -> str:
    """Recompute the overall-summary table from a raw results CSV and return
    the fully revised (post-revise_table) LaTeX source."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)}. "
            "'mean_score'/'total_time'/'total_time_sq' were only added to "
            "ci_single's and ci_paired's results CSVs on 2026-07-15 -- re-run "
            "the simulation to regenerate the CSV with these columns."
        )

    target = 1.0 - alpha
    df = df.copy()
    if "is_null" in df.columns:
        # ci_paired's CSV includes null/placebo scenario rows (for Type-I-style
        # checks) that latex_overall_summary excludes from the overall summary --
        # mirror that here. Handles both a real bool dtype and the "True"/"False"
        # string literals csv.writer(...).writerow(bool) produces.
        is_null = df["is_null"].astype(str).str.strip().str.lower() == "true"
        df = df[~is_null]
    df["group"] = df["eval_type"].map(eval_type_group)
    eval_types_present = set(df["eval_type"].unique())
    sizes_present = sorted(df["n"].unique())
    # First-seen order in the CSV (a live run's canonical REPORT_METHOD_ORDER
    # isn't available offline from a CSV alone; re-sort the input CSV first
    # if exact canonical column ordering matters for the paper).
    method_order = list(dict.fromkeys(df["method"]))

    agg: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    method_group_types: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in df.itertuples(index=False):
        g = row.group
        key = (g, row.method, row.n)
        agg[key].append((row.coverage, row.mean_width, row.mean_score))
        c_prev, t_prev = agg_counts[key]
        agg_counts[key] = (c_prev + row.covered, t_prev + row.n_reps)
        method_group_types[(row.method, g)].add(row.eval_type)

    method_groups: dict[str, set[str]] = defaultdict(set)
    for (g, m, _n) in agg:
        method_groups[m].add(g)

    rows: list[list[str]] = []
    for m in method_order:
        groups = sorted(method_groups.get(m, ()))
        multi_group = len(groups) > 1
        for g in groups:
            per_n_vals: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
            all_counts: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
            per_n_counts: dict[tuple[str, int], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
            for n in sizes_present:
                vals = agg.get((g, m, n))
                if vals:
                    per_n_vals[(m, n)] = list(vals)
                c, t = agg_counts.get((g, m, n), (0.0, 0.0))
                c_prev, t_prev = all_counts[m]
                all_counts[m] = (c_prev + c, t_prev + t)
                per_n_counts[(m, n)] = (c, t)

            mc, mw, ms = _headline_cov_width_score(per_n_vals, m, sizes_present)
            c_tot, t_tot = all_counts[m]
            _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)

            sub = df[(df["method"] == m) & (df["group"] == g)]
            avg_ms, se_ms = _time_stats(sub)
            time_str = f"${avg_ms:.3f} \\pm {se_ms:.3f}$" if np.isfinite(avg_ms) else "-"

            et_label = eval_type_label(method_group_types[(m, g)], eval_types_present)
            label_str = f"{escape_latex(m)} ({g})" if multi_group else escape_latex(m)
            out_row = [
                label_str,
                f"{mc:.3f}" if np.isfinite(mc) else "-",
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{mw:.4f}" if np.isfinite(mw) else "-",
                f"{ms:.4f}" if np.isfinite(ms) else "-",
                time_str,
                et_label,
            ]
            for n in sizes_present:
                c_n, t_n = per_n_counts.get((m, n), (0.0, 0.0))
                cov_n = c_n / t_n if t_n > 0 else float("nan")
                out_row.append(f"{cov_n:.3f}" if np.isfinite(cov_n) else "-")
            rows.append(out_row)

    raw_table = booktabs_table(
        caption=caption,
        label=label,
        columns=["Method", "Coverage", "95\\% MC band", "Mean width", "Score", "Time (ms)", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )
    return revise_table(raw_table)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="Path to a *_results.csv file (ci_single schema).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for the run (default 0.05).")
    parser.add_argument("--caption", default=None, help="LaTeX caption (default: a generic one inferred from the CSV).")
    parser.add_argument("--label", default="tab:overall_summary", help="LaTeX \\label{} (default: tab:overall_summary).")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv_path)

    n_reps_mode = int(df["n_reps"].mode().iloc[0]) if not df.empty and "n_reps" in df.columns else 0
    target_pct = f"{(1 - args.alpha):.0%}"
    caption = args.caption or (
        f"Overall CI coverage summary (nominal {target_pct}, reps/cell={n_reps_mode}), "
        f"recomputed from \\texttt{{{escape_latex(args.csv_path)}}}. Score is the interval score "
        "(width + $\\frac{2}{\\alpha}\\times$miss-distance; lower is better). Methods tested on both "
        "binary and numeric data are reported as two rows, one per eval-type group, so no row "
        "averages across incomparable scales."
    )

    try:
        print(build_latex_summary(df, alpha=args.alpha, caption=caption, label=args.label))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
