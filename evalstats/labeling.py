"""Guided random sampling and interactive grading for human labeling.

Option 1 of evalstats' MCAR usability tooling (see judge_alignment()'s
selection= parameter and its representativeness checks for the disclosure/
detection side): most developers passing LLM-judge scores into compare() or
judge_alignment() won't know that PPI correction requires the human-labeled
subset to be a random sample of the full item pool. Rather than relying on
the developer to do that sampling correctly by hand, this module performs
it for them and marks the result, so the labeled subset genuinely is what
selection="random" claims it is.

Deliberately accepts a *superset* of what compare()/analyze() require:
PPI correction works on between-subjects (unpaired, disjoint item pools per
condition) data just as well as the within-subjects (paired, shared item
ids) case compare() is usually demonstrated with, and doesn't need an item
column at all -- every function here degrades gracefully to "each row is
its own item" when one isn't found, rather than requiring the fuller
structure load_from() validates (canonical roles, no duplicate keys, etc.).

Three-phase flow (mirrors the CLI's ``evalstats label``):
  1. :func:`detect_design` -- pure detection, no sampling. Resolves the
     item/factor columns, figures out whether the design is paired or
     unpaired, and enforces PPI's own minimum dataset size (50 rows) up
     front so nobody spends effort hand-labeling a dataset PPI can't
     actually use. :func:`describe_design` renders it for a confirmation
     prompt -- the CLI asks the user to confirm this matches their actual
     experimental design before sampling anything.
  2. :func:`sample_for_labeling` -- marks a boolean column
     (``_sampled_for_labeling``) identifying which rows need a human grade,
     targeting N_lab labeled items per condition. Idempotent: re-running on
     an already-marked table preserves prior selections and any labels
     already filled in, rather than re-sampling.
  3. :func:`run_interactive_labeling` -- opt-in. Walks every marked-but-
     ungraded row in the terminal, asking for a grade per requested metric,
     saving after every answer so a Ctrl-C loses at most one item.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from evalstats.loader import _CANONICAL_ALIASES, _find_col, _detect_score_type
from evalstats.core.design import detect_paired as _detect_paired

MARKER_COL = "_sampled_for_labeling"
SYNTHETIC_ITEM_COL = "_row_item"

# Synthetic human_cols key used when no --metric is given at all -- e.g. an
# HCI researcher sampling/collecting ground-truth labels before any LLM
# judge exists yet. Produces one generic "<prefix>label" column instead of
# "<prefix><metric>".
GENERIC_LABEL_KEY = "label"

VALID_SCORE_TYPES = ("binary", "likert", "continuous")

# Mirrors the n_all < 50 floor _run_alignment_ppi() enforces in api.py ("PPI
# is only beneficial at scale"). Kept as a local constant rather than a
# cross-module import since that check is inline there, not exported -- if
# it ever changes, update both.
MIN_TOTAL_FOR_PPI = 50


class QuitLabeling(Exception):
    """Raised internally when the user quits an interactive session early."""


# ─────────────────────────────────────────────────────────────────────────────
# Design detection
# ─────────────────────────────────────────────────────────────────────────────

# _detect_paired is now evalstats.core.design.detect_paired (imported above as
# _detect_paired for backwards compatibility with this module's internal call
# site) -- moved so evalstats.api.compare() can share the exact same
# implementation for its design="auto" detection without importing this
# labeling-CLI-focused module.


def _resolve_columns(
    df: pd.DataFrame,
    *,
    metrics: Optional[list[str]],
    factor: Optional[str],
    item_col: Optional[str],
) -> tuple[pd.DataFrame, str, Optional[str], bool]:
    metrics = metrics or []
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(
            f"metric column(s) {missing} not found in data. Available: {list(df.columns)}"
        )

    # Always work on a copy from here on -- callers (including sample_for_labeling
    # re-running detect_design internally) must never see their input DataFrame
    # mutated in place by column assignments below (MARKER_COL, human_* cols, or
    # the synthetic item column).
    df = df.copy()

    resolved_item = item_col or _find_col(df, _CANONICAL_ALIASES["item"])
    item_synthetic = resolved_item is None
    if item_synthetic:
        # No item/input column found -- not an error. PPI/judge_alignment
        # don't require one either; without it there's simply no shared
        # identity to pair rows across conditions on, so every row is
        # treated as its own item (forces the unpaired sampling path).
        df[SYNTHETIC_ITEM_COL] = np.arange(len(df))
        resolved_item = SYNTHETIC_ITEM_COL
    elif resolved_item not in df.columns:
        raise ValueError(f"item_col {resolved_item!r} not found in data.")

    # "model" and "prompt" are both legitimate compare()/analyze() factor
    # roles (see loader.py's canonical-role list) -- a prompt-only A/B test
    # on a single model has no "model" column at all, so falling back to
    # "model" alone would silently miss its factor and treat every prompt's
    # rows as one undifferentiated group.
    resolved_factor = factor
    if resolved_factor is None:
        resolved_factor = _find_col(df, _CANONICAL_ALIASES["model"]) or _find_col(df, _CANONICAL_ALIASES["prompt"])
    if resolved_factor is not None and resolved_factor not in df.columns:
        raise ValueError(f"factor column {resolved_factor!r} not found in data.")

    return df, resolved_item, resolved_factor, item_synthetic


def detect_design(
    df: pd.DataFrame,
    *,
    metrics: Optional[list[str]] = None,
    factor: Optional[str] = None,
    item_col: Optional[str] = None,
    min_total: int = MIN_TOTAL_FOR_PPI,
) -> dict:
    """Resolve columns and detect the experimental design, without sampling
    anything. Raises ``ValueError`` if the dataset is below PPI's own
    minimum size (default 50 rows) -- deliberately fails before any
    labeling effort is spent, not after.

    ``metrics`` is optional: pass ``None``/``[]`` to sample/label ground
    truth ahead of having any LLM judge column at all (e.g. an HCI
    researcher collecting labels before building a judge). Sampling and
    design detection don't need it -- only the eventual PPI validation does.

    Returns a dict (including the possibly-copied DataFrame, under "df" --
    only different from the input when a synthetic item column had to be
    added) meant to be rendered with :func:`describe_design` and passed
    straight to :func:`sample_for_labeling`.
    """
    metrics = metrics or []
    if len(df) < min_total:
        raise ValueError(
            f"sample_for_labeling requires at least {min_total} rows in the full "
            f"dataset (matches judge_alignment()/compare()'s own PPI floor -- PPI "
            f"is only beneficial at scale); got {len(df)}."
        )

    df2, item_col_r, factor_r, item_synthetic = _resolve_columns(
        df, metrics=metrics, factor=factor, item_col=item_col
    )

    if item_synthetic:
        paired = False
        n_items_universe = len(df2)
    else:
        paired = _detect_paired(df2, factor_r, item_col_r)
        n_items_universe = int(df2[item_col_r].nunique())

    if factor_r is not None:
        levels = list(df2[factor_r].dropna().unique())
        per_level_n = {lvl: int((df2[factor_r] == lvl).sum()) for lvl in levels}
    else:
        levels = [None]
        per_level_n = {"(all rows)": len(df2)}

    return {
        "df": df2,
        "n_total": len(df2),
        "item_col": item_col_r,
        "item_col_synthetic": item_synthetic,
        "factor_col": factor_r,
        "paired": paired,
        "levels": levels,
        "per_level_n": per_level_n,
        "n_items_universe": n_items_universe,
        "metrics": metrics,
    }


def describe_design(design: dict) -> str:
    """Human-readable rendering of a :func:`detect_design` result, meant to
    be shown to the user for confirmation before sampling proceeds.
    """
    lines = [f"Detected {design['n_total']} rows."]

    if design["item_col_synthetic"]:
        lines.append(
            "No item/input column found -- treating each row as its own item "
            "(no cross-condition pairing is possible without one)."
        )
    else:
        lines.append(
            f"Item column:   '{design['item_col']}'  ({design['n_items_universe']} distinct items)"
        )

    if design["factor_col"] is not None:
        lines.append(f"Factor column: '{design['factor_col']}'")
        for lvl, n in design["per_level_n"].items():
            lines.append(f"    {lvl}: {n} rows")
    else:
        lines.append("No factor/condition column found -- treating all rows as one group.")

    if design["paired"]:
        kind = (
            "PAIRED / within-subjects (item ids repeat across conditions -- will "
            "sample once and reuse the same items across every condition)"
        )
    else:
        kind = (
            "UNPAIRED / between-subjects (item pools differ by condition, or there's "
            "no shared item id -- will sample independently within each condition)"
        )
    lines.append(f"Design: {kind}")
    if design["metrics"]:
        lines.append(f"Metric(s) to validate: {design['metrics']}")
    else:
        lines.append(
            "No --metric given -- sampling/labeling ground truth only, no LLM "
            "judge column to validate against yet."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_for_labeling(
    df: pd.DataFrame,
    *,
    metrics: Optional[list[str]] = None,
    factor: Optional[str] = None,
    item_col: Optional[str] = None,
    n_lab: int = 15,
    seed: Optional[int] = None,
    human_col_prefix: str = "human_",
    sort_labeled_first: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Mark rows for human labeling, targeting ``n_lab`` labeled items per
    condition (auto-detected as paired or unpaired -- see
    :func:`detect_design`) and sharing one sampled item set across every
    metric in ``metrics`` (one round of grading can cover several axes on
    the same content).

    ``metrics`` is optional. When omitted (e.g. sampling/labeling ground
    truth before any LLM judge exists), one generic column named
    ``f"{human_col_prefix}{GENERIC_LABEL_KEY}"`` (``"human_label"`` by
    default) is created instead of one per metric.

    Idempotent: pass in a DataFrame that already has ``_sampled_for_labeling``
    set (e.g. re-loading a partially-labeled file) and this only tops up any
    condition still short of ``n_lab``, without disturbing existing
    selections or already-filled human labels.

    sort_labeled_first : bool, default True
        Move every marked (``_sampled_for_labeling=True``) row to the top of
        the returned DataFrame, in a stable sort (rows keep their original
        relative order within each of the two groups). A human labeler
        opening the output file otherwise has to scroll through the entire
        eval set to find the handful of rows that actually need a score --
        with a paired design those rows aren't even contiguous, since each
        marked item's rows are spread across every condition. Pass ``False``
        to keep the input's original row order untouched instead (e.g. if
        you rely on that order elsewhere, or hand-edit files and want a
        stable diff against the un-sampled version).

    Returns
    -------
    (df, info) : the marked DataFrame and a dict with the seed actually
        used, the detected design, and a per-condition coverage report --
        meant for the CLI to print, and for passing straight to
        :func:`run_interactive_labeling`.
    """
    metrics = metrics or []
    design = detect_design(df, metrics=metrics, factor=factor, item_col=item_col)
    df = design["df"]
    item_col_r = design["item_col"]
    factor_r = design["factor_col"]
    paired = design["paired"]
    levels = design["levels"]

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32 - 1))
    rng = np.random.default_rng(seed)

    if MARKER_COL not in df.columns:
        df[MARKER_COL] = False
    else:
        df[MARKER_COL] = df[MARKER_COL].fillna(False).astype(bool)

    if metrics:
        human_cols = {m: f"{human_col_prefix}{m}" for m in metrics}
    else:
        human_cols = {GENERIC_LABEL_KEY: f"{human_col_prefix}{GENERIC_LABEL_KEY}"}
    for hcol in human_cols.values():
        if hcol not in df.columns:
            df[hcol] = np.nan

    coverage: dict = {}

    if paired:
        universe = list(pd.unique(df[item_col_r].dropna()))
        already = set(df.loc[df[MARKER_COL], item_col_r].dropna().unique())
        pool = [it for it in universe if it not in already]
        rng.shuffle(pool)
        need = max(0, n_lab - len(already))
        chosen = already | set(pool[:need])
        if chosen:
            df.loc[df[item_col_r].isin(chosen), MARKER_COL] = True
        for lvl in levels:
            sub = df if lvl is None else df[df[factor_r] == lvl]
            coverage[lvl if lvl is not None else "(all rows)"] = int(sub[MARKER_COL].sum())
    else:
        for lvl in levels:
            lvl_mask = df[factor_r] == lvl if factor_r is not None else pd.Series(True, index=df.index)
            sub_items = list(pd.unique(df.loc[lvl_mask, item_col_r].dropna()))
            already = set(df.loc[lvl_mask & df[MARKER_COL], item_col_r].dropna().unique())
            pool = [it for it in sub_items if it not in already]
            rng.shuffle(pool)
            need = max(0, n_lab - len(already))
            chosen = already | set(pool[:need])
            if chosen:
                df.loc[lvl_mask & df[item_col_r].isin(chosen), MARKER_COL] = True
            coverage[lvl if lvl is not None else "(all rows)"] = int((lvl_mask & df[MARKER_COL]).sum())

    if sort_labeled_first:
        # Stable sort: marked rows first, but original relative order is
        # preserved within both the marked and unmarked groups.
        df = df.sort_values(
            by=MARKER_COL, ascending=False, kind="stable", ignore_index=True,
        )

    info = {
        "seed": seed,
        "paired": paired,
        "item_col": item_col_r,
        "factor_col": factor_r,
        "human_cols": human_cols,
        "coverage": coverage,
        "n_lab_target": n_lab,
    }
    return df, info


# ─────────────────────────────────────────────────────────────────────────────
# Interactive grading
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_for_grade(score_type: str, metric_name: str, in_: Callable[[str], str]) -> Optional[float]:
    """Prompt for one grade, validated against the metric's detected score
    type. Returns None on skip; raises QuitLabeling on quit.
    """
    if score_type == "binary":
        hint = "1=pass / 0=fail"
    elif score_type == "likert":
        hint = "small integer, e.g. 1-5"
    else:
        hint = "numeric score"

    while True:
        raw = in_(f"  {metric_name} [{hint}, s=skip, q=quit]: ").strip().lower()
        if raw == "q":
            raise QuitLabeling()
        if raw in ("s", ""):
            return None
        try:
            val = float(raw)
        except ValueError:
            print("    not a number -- try again")
            continue
        if score_type == "binary" and val not in (0.0, 1.0):
            print("    binary metric -- enter 0 or 1")
            continue
        if score_type == "likert" and not (1 <= val <= 10):
            print("    likert metric -- enter a small positive integer")
            continue
        return val


def resolve_score_types(
    df: pd.DataFrame,
    keys: list[str],
    *,
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Resolve a score type per grading target (one per key in
    ``info["human_cols"]``): an explicit override when given, else
    auto-detected from ``df[key]`` -- which only exists when ``key`` is a
    real LLM-judge metric column, not the synthetic
    :data:`GENERIC_LABEL_KEY` used when no ``--metric`` was given.

    Raises ``ValueError`` (naming the offending key) when neither an
    override nor a real column is available, rather than silently guessing
    "continuous" from an all-NaN column -- the whole point of a labeling-
    only session is that there's no judge score to infer a type from, so
    the type has to be declared, not detected.
    """
    overrides = overrides or {}
    resolved: dict[str, str] = {}
    for key in keys:
        if key in overrides:
            resolved[key] = overrides[key]
        elif key in df.columns:
            resolved[key] = _detect_score_type(df[key].dropna())
        else:
            raise ValueError(
                f"Can't auto-detect a score type for {key!r} -- there's no LLM "
                "judge column to infer it from (labeling-only mode, no --metric "
                f"given). Pass score_type= explicitly for it (one of {VALID_SCORE_TYPES})."
            )
    return resolved


def run_interactive_labeling(
    df: pd.DataFrame,
    info: dict,
    *,
    save_fn: Callable[[pd.DataFrame], None],
    display_cols: Optional[list[str]] = None,
    input_fn: Callable[[str], str] = input,
    score_type_overrides: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Walk every marked-but-ungraded row, prompting for a grade per metric
    and saving after each answer (a Ctrl-C or 'q' loses at most one item).

    Deliberately never shows the LLM judge's own score for the metric being
    graded -- an independent human check is the whole point, and seeing the
    judge's score first would anchor the human toward agreeing with it.

    ``score_type_overrides`` lets the caller declare a grading target's
    score type explicitly (keyed the same as ``info["human_cols"]``) --
    required when there's no judge column to auto-detect from (no
    ``--metric`` given), and otherwise usable to correct a wrong guess.
    """
    metrics = list(info["human_cols"].keys())
    score_types = resolve_score_types(df, metrics, overrides=score_type_overrides)
    human_cols = list(info["human_cols"].values())

    exclude = {info["item_col"], info.get("factor_col"), MARKER_COL, *human_cols, *metrics}
    if display_cols is None:
        display_cols = [c for c in df.columns if c not in exclude]

    todo_mask = df[MARKER_COL] & df[human_cols].isna().any(axis=1)
    todo_idx = df.index[todo_mask].tolist()
    total = len(todo_idx)

    print(f"\n{total} row(s) need grading ({len(metrics)} metric(s) each).")
    print("Enter a grade for each, 's' to skip an item, 'q' to save and quit.\n")

    done = 0
    try:
        for idx in todo_idx:
            row = df.loc[idx]
            pending = [m for m in metrics if pd.isna(row[info["human_cols"][m]])]
            if not pending:
                continue
            print("─" * 58)
            for c in display_cols:
                print(f"  {c}: {row[c]}")
            for m in pending:
                val = _prompt_for_grade(score_types[m], m, input_fn)
                if val is not None:
                    df.at[idx, info["human_cols"][m]] = val
            save_fn(df)
            done += 1
    except QuitLabeling:
        print(f"\nStopped early -- graded {done}/{total} this session. Progress saved.")
        save_fn(df)
        return df

    print(f"\nDone -- graded {done}/{total} this session.")
    return df
