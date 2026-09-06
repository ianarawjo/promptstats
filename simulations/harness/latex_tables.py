"""Shared LaTeX booktabs-table formatting for --latex CLI output.

Each case's `save_results_artifacts*` appends one of these tables to the same
`*_summary.log` file it already writes (not a separate artifact), so the
official-test manifest doesn't need new paths. Cases pass in numbers they've
already aggregated for the plain-text report -- this module only formats.
"""

from __future__ import annotations

import math

NUMERIC_EVAL_TYPES = {"continuous", "likert", "grades"}


def escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_")


def eval_type_group(et: str) -> str:
    """Map a raw eval_type to its LaTeX-table reporting group.

    'binary' or 'numeric' (continuous/likert/grades collapsed together,
    matching NUMERIC_EVAL_TYPES) -- the coarser two-way split
    latex_overall_summary uses to decide whether a method needs one row or
    two. A method present in only one group gets a single row; a method
    present in both gets two rows ("<method> (binary)"/"<method>
    (numeric)"), each computed from only that group's data -- averaging
    Cov/Width/Score across binary and numeric data mixes two different
    scales/regimes into one number that isn't comparable to any group-pure
    method's row, which is what an "all" Eval-types value used to paper
    over.
    """
    return "numeric" if et in NUMERIC_EVAL_TYPES else "binary"


#: Order eval-type blocks appear in, top to bottom, in every per-type table.
GROUP_ORDER = ["bin", "cont", "lik", "grades"]


def report_eval_type_group(et: str) -> str:
    """Short per-eval-type group label for the per-type LaTeX tables --
    FINER than `eval_type_group`, which only splits binary vs. a single
    "numeric" bucket covering continuous+likert+grades together.

    Likert gets its own block rather than being averaged into "numeric"
    alongside continuous: the two have materially different small-N
    paired-diff behavior, and pooling them hides exactly
    that distinction -- concretely, it also mixes likert's 1--5-scale widths
    with grades' 0--100-scale widths. The coarser `eval_type_group` is kept
    for `csv_to_latex_summary.py` and `csv_to_simultaneous_ci_summary.py`,
    which still report the 2-way split.
    """
    return {"binary": "bin", "continuous": "cont", "likert": "lik", "grades": "grades"}.get(et, et)


def sort_groups(groups) -> list[str]:
    """Order eval-type groups by GROUP_ORDER, unknown groups last, so every
    per-type table in the paper stacks its blocks the same way."""
    return sorted(
        groups,
        key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER),
    )


def eval_type_label(covered: set[str], all_present: set[str]) -> str:
    """Summarize which eval types a row's data actually covers.

    'all' if it covers everything present in this run; 'numeric' if it covers
    exactly the numeric (non-binary) eval types present; 'binary' if it's
    binary-only; otherwise an explicit comma-separated list.
    """
    if not covered:
        return "-"
    if covered >= all_present:
        return "all"
    numeric_present = NUMERIC_EVAL_TYPES & all_present
    if covered == numeric_present and covered:
        return "numeric"
    if covered == {"binary"}:
        return "binary"
    return ", ".join(sorted(covered))


def coverage_cell(cov: float, target: float) -> str:
    """Format a coverage value, shading it with \\cellcolor when it falls
    outside the acceptable band around `target` -- so miscalibration is
    visible at a glance instead of requiring the reader to parse every
    number. Requires \\usepackage[table]{xcolor} in the including document.

    Undercoverage (below `target - 0.001`) shades red; over-conservative
    coverage (above `target + 0.02`) shades blue. Shading intensity scales
    linearly with distance outside that band, from faint at the edge to
    near-saturated at `target - 0.15` (red) or at 1.0 -- the hard ceiling a
    coverage proportion can't exceed (blue) -- rather than a hard two-tier
    cutoff, since a fixed threshold would make two adjacent values (e.g.
    0.948 vs 0.950) look categorically different when they're barely
    distinguishable.

    The threshold comparison uses the same 3-decimal rounding as the
    displayed text, not the raw float -- otherwise a value like 0.9486
    prints as "0.949" (matching the stated 0.949 threshold) but would still
    shade red, which reads as a bug: a cell that visibly equals the
    boundary shouldn't render on the wrong side of it. Coverage this close
    to nominal is within Monte Carlo noise anyway, not a real miscalibration
    signal.
    """
    if cov is None or not math.isfinite(cov):
        return "-"
    cov = round(cov, 3)
    text = f"{cov:.3f}"
    lower_bad = target - 0.001
    upper_bad = target + 0.02
    if cov < lower_bad:
        red_anchor = target - 0.15
        frac = min(1.0, (lower_bad - cov) / (lower_bad - red_anchor))
        pct = round(15 + 50 * frac)
        return f"\\cellcolor{{red!{pct}}}{text}"
    if cov > upper_bad:
        frac = min(1.0, (cov - upper_bad) / (1.0 - upper_bad))
        pct = round(15 + 50 * frac)
        return f"\\cellcolor{{blue!{pct}}}{text}"
    return text


def error_rate_cell(rate: float, alpha: float) -> str:
    """Format a Type-I error / FWER value, shading it the same way
    `coverage_cell` shades coverage -- so a reader moving between the CI
    tables and the p-value/FWER tables reads one colour language: red means
    anti-conservative (the test rejects too often / the interval misses too
    often), blue means over-conservative.

    This is the exact dual of `coverage_cell` under ``rate = 1 - coverage``.
    A two-sided test at nominal `alpha` corresponds to a `1 - alpha`
    interval, so mapping each coverage threshold through ``1 - x`` gives:
    inflated error (above ``alpha + 0.001``) shades red, saturating at
    ``alpha + 0.15``; conservative error (below ``alpha - 0.02``) shades
    blue, saturating at 0.0 -- the hard floor a rejection proportion can't
    go below, mirroring the 1.0 ceiling on coverage. Keeping the ramp
    identical means a cell shaded ``red!40`` carries the same magnitude of
    miscalibration in either family of tables.

    As in `coverage_cell`, the comparison uses the same 3-decimal rounding
    as the displayed text, so a value that visibly equals a threshold never
    renders on the wrong side of it.
    """
    if rate is None or not math.isfinite(rate):
        return "-"
    rate = round(rate, 3)
    text = f"{rate:.3f}"
    # Round the thresholds to the displayed precision too, not just the
    # value: `alpha - 0.02` is 0.030000000000000002 in binary floating
    # point, which would shade an exactly-0.030 rate blue while its
    # coverage dual (0.970) stays unshaded.
    upper_bad = round(alpha + 0.001, 3)
    lower_bad = round(alpha - 0.02, 3)
    if rate > upper_bad:
        red_anchor = alpha + 0.15
        frac = min(1.0, (rate - upper_bad) / (red_anchor - upper_bad))
        pct = round(15 + 50 * frac)
        return f"\\cellcolor{{red!{pct}}}{text}"
    if rate < lower_bad:
        frac = min(1.0, (lower_bad - rate) / lower_bad) if lower_bad > 0 else 1.0
        pct = round(15 + 50 * frac)
        return f"\\cellcolor{{blue!{pct}}}{text}"
    return text


def mark_best_and_runnerup(
    cells: list[str], values: list[float], *, higher_is_better: bool = False
) -> list[str]:
    """Wrap the best (lowest) value's cell in \\textbf{}, and the runner-up's
    in \\underline{} -- e.g. for one table block's Score column, where lower
    is better. `cells` and `values` must be parallel; a non-finite value
    (NaN/inf, i.e. no data for that row) is excluded from ranking but its
    cell is still returned unmodified. Only the Score column gets this
    treatment, not Coverage or Width -- Score already combines coverage-miss
    and width into one number, so marking it alone gives a single unambiguous
    "best" per block instead of risking Score and Coverage disagreeing on
    which row wins.

    ``higher_is_better`` flips the ranking, for columns where larger wins --
    power in the p-value and FWER tables, which plays the same role Score
    plays in the CI tables: the one "more is better, no nominal target"
    column worth marking. Type-I/FWER is *not* marked this way; it has a
    nominal target, so it gets `error_rate_cell` shading instead, and a
    method with the lowest Type-I error is usually just the most
    conservative rather than the best.
    """
    ranked = sorted(
        (i for i, v in enumerate(values) if v is not None and math.isfinite(v)),
        key=lambda i: values[i],
        reverse=higher_is_better,
    )
    out = list(cells)
    if not ranked:
        return out
    best = ranked[0]
    out[best] = f"\\textbf{{{cells[best]}}}"
    if len(ranked) > 1:
        runner_up = ranked[1]
        out[runner_up] = f"\\underline{{{cells[runner_up]}}}"
    return out


def booktabs_table(
    *, caption: str, label: str, columns: list[str], rows: list[list[str]],
    col_align: str | None = None, rule_before: set[int] | None = None,
) -> str:
    """Assemble a booktabs-style LaTeX table from pre-formatted string cells.

    ``rule_before``: 0-indexed row positions before which to insert an extra
    ``\\midrule`` -- e.g. to divide a table into blocks (binary vs. numeric
    eval-type rows). Most callers leave this ``None``. ``revise_table``
    (revise_latex_tables.py) knows to preserve these when post-processing."""
    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\"
    rule_before = rule_before or set()
    body_lines = []
    for i, row in enumerate(rows):
        if i in rule_before:
            body_lines.append(r"\midrule")
        body_lines.append(" & ".join(row) + r" \\")
    body = "\n".join(body_lines)
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        f"\\begin{{tabular}}{{{col_align}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table*}\n"
    )
