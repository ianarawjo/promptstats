"""EvalResults data container and load_from() entry point.

Provides the high-level data-loading API described in the evalstats spec:
  - ``load_from(data, *, metric_cols, col_map)`` → ``EvalResults``
  - ``EvalResults`` — long-format container with structure inspection methods
  - ``EvalLoadError`` — raised when data cannot be parsed or validated
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

ScoreType = Literal["binary", "continuous", "likert"]
_VALID_SCORE_TYPES = frozenset({"binary", "continuous", "likert"})

# ─────────────────────────────────────────────────────────────────────────────
# Canonical column aliases
# ─────────────────────────────────────────────────────────────────────────────

_CANONICAL_ALIASES: dict[str, list[str]] = {
    "model":  ["model", "model_label", "model_name"],
    "prompt": ["prompt", "template", "prompt_template"],
    "item":   ["item", "input", "example", "id", "input_label"],
    "run":    ["run", "seed", "repeat", "run_id", "trial"],
}

_SCORE_ALIASES = ["score", "value", "result", "metric"]


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class EvalLoadError(ValueError):
    """Raised when eval data cannot be parsed or validated."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_score_type(series: pd.Series) -> ScoreType:
    """Infer score type from column values."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return "continuous"

    vals = clean.to_numpy(dtype=float)
    unique = set(np.unique(vals))

    # Binary: only 0 and 1
    if unique <= {0.0, 1.0}:
        return "binary"

    # Likert means discrete here: whole numbers on a scale that starts at or
    # above zero. The width is not capped, since a 0-25 rubric is as ordinal as
    # a 1-5 one. Negative values are excluded because a rating scale running
    # below zero is not a rating scale; that data reads as continuous instead.
    # A metric that is only incidentally whole-numbered is declared with
    # score_type="continuous".
    all_int = bool(np.all(vals == np.floor(vals)))
    if all_int and float(vals.min()) >= 0.0:
        return "likert"

    # Continuous: floats bounded in [0, 1]
    if float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0:
        return "continuous"

    return "continuous"


def _find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return the first column name in df that matches any alias (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _infer_canonical_cols(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Map canonical roles to actual column names via alias matching."""
    return {role: _find_col(df, aliases) for role, aliases in _CANONICAL_ALIASES.items()}


def _infer_metric_cols(df: pd.DataFrame, canonical_used: set[str]) -> list[str]:
    """Find metric columns: prefer canonical names, then numeric non-canonical columns."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in _SCORE_ALIASES:
        if alias in lower_map:
            col = lower_map[alias]
            if col not in canonical_used:
                return [col]
    # Fall back: any numeric column not already assigned a canonical role
    return [
        col for col in df.columns
        if col not in canonical_used and pd.api.types.is_numeric_dtype(df[col])
    ]


def _check_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    run_col: Optional[str],
) -> None:
    """Raise EvalLoadError when duplicate key rows exist without a run column."""
    if run_col is not None:
        return
    counts = df.groupby(key_cols, dropna=False).size()
    dups = counts[counts > 1]
    if len(dups) == 0:
        return
    examples = [str(k) for k in list(dups.index)[:3]]
    raise EvalLoadError(
        f"Found {len(dups)} duplicate ({', '.join(key_cols)}) combination(s). "
        "If these are intentional repeated runs, add a 'run' column before loading.\n"
        f"Example duplicates: {'; '.join(examples)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EvalResults
# ─────────────────────────────────────────────────────────────────────────────

class EvalResults:
    """Long-format evaluation results with detected structure.

    Created by :func:`load_from`. Do not instantiate directly.

    Attributes
    ----------
    score_type : str or dict[str, str]
        Detected score type(s). A string when there is one metric column;
        a dict mapping column name → type when there are multiple.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        score_types: dict[str, ScoreType],
        metric_cols: list[str],
        col: dict[str, Optional[str]],
        factor_cols: list[str],
        declared_factors: Optional[list[str]] = None,
        declared_score_types: Optional[dict[str, str]] = None,
    ):
        self._df = df
        self._score_types = score_types       # metric_col → score_type
        self._metric_cols = metric_cols       # ordered list of metric column names
        self._col = col                       # role → actual col name (or None)
        self._factor_cols = factor_cols       # non-canonical, non-metric columns
        self._declared_factors = list(declared_factors or [])  # named via factors=
        # Types the caller stated, as opposed to ones detected from the sample.
        self._declared_score_types = dict(declared_score_types or {})

    # ── public properties ────────────────────────────────────────────────────

    @property
    def score_type(self) -> Union[ScoreType, dict[str, ScoreType]]:
        """Detected score type. String if one metric column, dict if multiple."""
        if len(self._score_types) == 1:
            return next(iter(self._score_types.values()))
        return dict(self._score_types)

    # ── inspection methods ───────────────────────────────────────────────────

    def summary(self) -> None:
        """Print detected structure and descriptive statistics.

        Always run this before calling ``compare`` or other analyses to verify
        that column assignments match your intent.
        """
        col = self._col
        df = self._df

        mc = col.get("model")
        pc = col.get("prompt")
        ic = col.get("item")
        rc = col.get("run")

        n_models  = df[mc].nunique() if mc and mc in df else 1
        n_prompts = df[pc].nunique() if pc and pc in df else 1
        n_items   = df[ic].nunique() if ic and ic in df else 1
        n_runs    = df[rc].nunique() if rc and rc in df else 1

        key_parts = [c for c in [mc, pc, ic] if c and c in df]
        if key_parts:
            actual_combos   = df.groupby(key_parts).ngroups
            expected_combos = n_models * n_prompts * n_items
            is_full = (actual_combos == expected_combos)
        else:
            is_full = True

        sep = "─" * 43
        print("EvalResults")
        print(sep)

        # Score type(s)
        for mc_col, stype in self._score_types.items():
            print(f"Score column : {mc_col} ({stype})")

        # Structure line
        dim_parts: list[str] = []
        if mc and mc in df:
            dim_parts.append(f"{n_models} models")
        if pc and pc in df:
            dim_parts.append(f"{n_prompts} prompts")
        if ic and ic in df:
            dim_parts.append(f"{n_items} items")
        total_obs = len(df)
        print(f"Structure    : {' × '.join(dim_parts)} = {total_obs} rows")

        factorial_str = "✓" if is_full else "✗ (incomplete)"
        print(f"               [full factorial: {factorial_str}]")

        runs_str = str(n_runs) if rc and rc in df else "none"
        print(f"               [repeated runs: {runs_str}]")

        # Detected factor columns
        if self._factor_cols:
            print()
            print("Factors detected (non-canonical columns):")
            for fc in self._factor_cols:
                if fc in df.columns:
                    levels = df[fc].dropna().unique()
                    sample = [str(lv) for lv in levels[:4]]
                    if len(levels) > 4:
                        sample.append("...")
                    print(f"  {fc:<12s}: {len(levels)} levels [{', '.join(sample)}]")

        # Column assignments
        print()
        print("Column assignments:")
        for role, canonical_name in [("model", "model"), ("prompt", "prompt"),
                                      ("item", "item"), ("run", "run")]:
            actual = col.get(role)
            if actual:
                match = "✓" if actual == canonical_name else f"(alias for '{canonical_name}')"
                print(f"  {role:<8s}→ '{actual}'  {match}")
        for mc_col in self._metric_cols:
            print(f"  {'score':<8s}→ '{mc_col}'  ✓")

        # Warnings
        warns = self._missing_warnings()
        if warns:
            print()
            print("Warnings:")
            for w in warns:
                print(f"  • {w}")

        print(sep)

        # Descriptive statistics (no inference)
        print("Descriptive statistics (no inference):")
        primary_metric = self._metric_cols[0]
        group_col = (mc if mc and mc in df else
                     pc if pc and pc in df else None)
        if group_col:
            for level in df[group_col].dropna().unique():
                sub = df[df[group_col] == level][primary_metric].dropna()
                if len(sub) > 0:
                    print(f"  {str(level):<22s}  mean={sub.mean():.2f}  sd={sub.std():.2f}")
        else:
            sub = df[primary_metric].dropna()
            if len(sub) > 0:
                print(f"  overall  mean={sub.mean():.2f}  sd={sub.std():.2f}")

    def _missing_warnings(self) -> list[str]:
        warns: list[str] = []
        df = self._df
        mc = self._col.get("model")
        ic = self._col.get("item")
        primary = self._metric_cols[0] if self._metric_cols else None
        if not primary:
            return warns
        if mc and mc in df and ic and ic in df:
            for model in df[mc].dropna().unique():
                sub = df[df[mc] == model]
                n_missing = sub[primary].isna().sum()
                if n_missing > 0:
                    warns.append(
                        f"{n_missing} missing scores for {mc}='{model}'. "
                        "Use evaldata.missing() for details."
                    )
        return warns

    def head(self, n: int = 5) -> None:
        """Print the first *n* rows."""
        print(self._df.head(n).to_string())

    def show(self) -> None:
        """Print the full dataset."""
        print(self._df.to_string())

    def missing(self) -> pd.DataFrame:
        """Return a DataFrame summarising missing score observations."""
        df = self._df
        key_parts = [c for c in [
            self._col.get("model"),
            self._col.get("prompt"),
            self._col.get("item"),
        ] if c and c in df]

        if not key_parts or not self._metric_cols:
            return pd.DataFrame()

        primary = self._metric_cols[0]
        summary = (
            df.groupby(key_parts)[primary]
            .apply(lambda s: int(s.isna().sum()))
            .reset_index()
        )
        summary.columns = key_parts + ["n_missing"]
        return summary[summary["n_missing"] > 0].reset_index(drop=True)

    def plot(self, method: str = "auto"):
        """Plot raw score distributions (descriptive only — no inference)."""
        import matplotlib.pyplot as plt

        primary = self._metric_cols[0]
        mc = self._col.get("model")
        pc = self._col.get("prompt")
        df = self._df

        group_col = mc if (mc and mc in df) else (pc if (pc and pc in df) else None)

        print("[Note: This shows descriptive statistics only — no inference has been run.]")

        if group_col:
            levels = df[group_col].dropna().unique()
            means = [df[df[group_col] == lv][primary].dropna().mean() for lv in levels]
            stds  = [df[df[group_col] == lv][primary].dropna().std()  for lv in levels]
            labels = [str(lv) for lv in levels]

            fig, ax = plt.subplots()
            ax.bar(labels, means, yerr=stds, capsize=4)
            ax.set_ylabel(primary)
            ax.set_title(f"Score distribution by {group_col}")
        else:
            fig, ax = plt.subplots()
            df[primary].dropna().hist(ax=ax, bins="auto")
            ax.set_xlabel(primary)
            ax.set_title("Score distribution")

        plt.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        mc  = self._col.get("model")
        pc  = self._col.get("prompt")
        ic  = self._col.get("item")
        rc  = self._col.get("run")
        df  = self._df

        n_models  = df[mc].nunique() if mc and mc in df else 1
        n_prompts = df[pc].nunique() if pc and pc in df else 1
        n_items   = df[ic].nunique() if ic and ic in df else 1
        n_runs    = df[rc].nunique() if rc and rc in df else 1

        parts = []
        if mc and mc in df:
            parts.append(f"{n_models} models")
        if pc and pc in df:
            parts.append(f"{n_prompts} prompts")
        if ic and ic in df:
            parts.append(f"{n_items} items")
        if rc and rc in df:
            parts.append(f"{n_runs} runs")
        parts.append(f"{len(df)} rows")

        stype = self.score_type
        if isinstance(stype, dict):
            stype_str = ", ".join(f"{k}={v}" for k, v in stype.items())
        else:
            stype_str = stype

        return f"EvalResults({', '.join(parts)}, score_type={stype_str!r})"

    # ── alternative constructors ─────────────────────────────────────────────

    @classmethod
    def from_scores(
        cls,
        scores: dict,
        *,
        factors: Union[str, list] = "model",
    ) -> "EvalResults":
        """Construct an :class:`EvalResults` from a dict-of-arrays.

        This is a convenience constructor for the compact dict API used by
        the legacy ``compare_models`` / ``compare_prompts`` functions.

        Parameters
        ----------
        scores : dict
            One of two forms:

            * **Flat** ``{entity_name: array}`` — each array is 1-D ``(M,)``
              (one score per item) or 2-D ``(M, R)`` (R repeated runs).
              The entity column name is taken from *factors* when it is a
              plain string (e.g. ``"model"`` or ``"prompt"``).

            * **Nested** ``{model_name: {template_name: array}}`` — outer
              keys become a ``"model"`` column; inner keys become a
              ``"prompt"`` column.

        factors : str or list
            Controls the entity column name for flat dicts (e.g. ``"model"``,
            ``"prompt"``). Ignored for nested dicts. Defaults to ``"model"``.

        Returns
        -------
        EvalResults

        Examples
        --------
        >>> import evalstats as es
        >>> import numpy as np
        >>> evaldata = es.EvalResults.from_scores({
        ...     "gpt-4o":      np.array([0.82, 0.91, 0.78, 0.86]),
        ...     "claude-3-5":  np.array([0.75, 0.84, 0.71, 0.79]),
        ... }, factors="model")

        >>> # Multi-run
        >>> evaldata = es.EvalResults.from_scores({
        ...     "prompt_a": [[0.80, 0.82, 0.79], [0.90, 0.88, 0.91]],
        ...     "prompt_b": [[0.75, 0.77, 0.74], [0.85, 0.83, 0.86]],
        ... }, factors="prompt")
        """
        df = _scores_dict_to_df(scores, factors=factors)
        return load_from(df)


# ─────────────────────────────────────────────────────────────────────────────
# Scores-dict helpers (flat and nested dict-of-arrays input)
# ─────────────────────────────────────────────────────────────────────────────

def _is_nested_scores_dict(d: dict) -> bool:
    """Return True if d is a nested dict {outer_key: {inner_key: array}}."""
    if not d:
        return False
    first_val = next(iter(d.values()))
    return isinstance(first_val, dict)


def _scores_dict_to_df(
    scores: dict,
    factors: Union[str, list],
) -> pd.DataFrame:
    """Convert a scores dict to a long-format DataFrame.

    Handles two forms:

    * **Flat** ``{entity: array}`` — each array is 1-D ``(M,)`` or 2-D
      ``(M, R)`` (M items, R runs).  The entity column name is derived from
      *factors* when it is a plain string (e.g. ``"model"`` or ``"prompt"``);
      otherwise it defaults to ``"entity"``.

    * **Nested** ``{model: {template: array}}`` — the outer key maps to a
      ``"model"`` column and the inner key to a ``"prompt"`` column.
      Arrays may be 1-D or 2-D.

    The resulting DataFrame always has an ``"item"`` column (0-indexed).
    A ``"run"`` column is added when arrays are 2-D.
    """
    if not isinstance(scores, dict) or not scores:
        raise EvalLoadError("scores must be a non-empty dict.")

    rows: list[dict] = []

    if _is_nested_scores_dict(scores):
        # Nested: {model: {template: array}}
        for model_name, templates in scores.items():
            if not isinstance(templates, dict):
                raise EvalLoadError(
                    f"Nested scores dict: expected inner value to be a dict for "
                    f"key '{model_name}', got {type(templates).__name__}."
                )
            for tmpl_name, arr in templates.items():
                a = np.asarray(arr, dtype=float)
                if a.ndim == 1:
                    for i, s in enumerate(a):
                        rows.append({"model": model_name, "prompt": str(tmpl_name),
                                     "item": str(i), "score": float(s)})
                elif a.ndim == 2:
                    M, R = a.shape
                    for i in range(M):
                        for r in range(R):
                            rows.append({"model": model_name, "prompt": str(tmpl_name),
                                         "item": str(i), "run": str(r),
                                         "score": float(a[i, r])})
                else:
                    raise EvalLoadError(
                        f"Score array for model='{model_name}' template='{tmpl_name}' "
                        f"has {a.ndim} dimensions; expected 1-D or 2-D."
                    )
    else:
        # Flat: {entity: array}
        entity_col = (
            factors if isinstance(factors, str) and factors not in ("auto",)
            else "entity"
        )
        for entity_name, arr in scores.items():
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                for i, s in enumerate(a):
                    rows.append({entity_col: str(entity_name),
                                 "item": str(i), "score": float(s)})
            elif a.ndim == 2:
                M, R = a.shape
                for i in range(M):
                    for r in range(R):
                        rows.append({entity_col: str(entity_name),
                                     "item": str(i), "run": str(r),
                                     "score": float(a[i, r])})
            else:
                raise EvalLoadError(
                    f"Score array for '{entity_name}' has {a.ndim} dimensions; "
                    "expected 1-D (M items) or 2-D (M items × R runs)."
                )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# load_from
# ─────────────────────────────────────────────────────────────────────────────

def load_from(
    data: Union[pd.DataFrame, list[dict], str, "os.PathLike[str]"],
    *,
    metric_cols: Optional[Union[str, list[str], dict[str, str]]] = None,
    col_map: Optional[dict[str, str]] = None,
    factors: Optional[Union[str, list[str]]] = None,
) -> EvalResults:
    """Parse evaluation results into an :class:`EvalResults` object.

    Parameters
    ----------
    data : pd.DataFrame, list[dict], str, or os.PathLike
        Eval results in long format — one row per (model, prompt, item, score)
        observation. A path to a ``.csv``, ``.tsv``, ``.json``, ``.jsonl``, or
        ``.parquet`` file is read with pandas first; the extension selects the
        reader, and an unrecognised extension is read as CSV.
    metric_cols : str, list[str], or dict[str, str], optional
        Metric column(s) to use.

        * ``str`` — a single metric column name.
        * ``list[str]`` — multiple metric columns.
        * ``dict[str, str]`` — keys are metric column names, values are
          explicit score types (``"binary"``, ``"continuous"``, ``"likert"``).

        When omitted, :func:`load_from` infers the metric column(s).
    col_map : dict[str, str], optional
        Rename non-canonical column names to canonical ones before loading.
        E.g. ``{"llm": "model", "template": "prompt", "q_id": "item"}``.
    factors : str or list[str], optional
        Column(s) that vary the condition being compared, for data whose factor
        is not one of the canonical roles. Naming them here makes them part of
        the key identifying a row, so a within-subjects design that scores the
        same item under several conditions loads without renaming anything.
        Pass the same name to :func:`~evalstats.compare`'s ``factors=``.

    Returns
    -------
    EvalResults

    Raises
    ------
    EvalLoadError
        If the data is malformed, a metric column is missing, or duplicate
        ``(model, prompt, item)`` rows are found without a ``run`` column.

    Examples
    --------
    >>> import evalstats as es
    >>> import pandas as pd
    >>> df = pd.read_csv("results.csv")
    >>> evaldata = es.load_from(df, col_map={"llm": "model", "template": "prompt"})
    >>> evaldata.summary()

    >>> # Straight from a file, without reading it yourself
    >>> evaldata = es.load_from("results.csv")

    >>> # From a list of dicts (JSONL-style)
    >>> records = [{"model": "gpt-4", "prompt": "p1", "item": "q1", "score": 1}, ...]
    >>> evaldata = es.load_from(records)
    """
    # ── coerce input ─────────────────────────────────────────────────────────
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if not path.exists():
            raise EvalLoadError(f"No such file: {path}")
        suffix = path.suffix.lower()
        try:
            if suffix in (".tsv", ".tab"):
                df = pd.read_csv(path, sep="\t")
            elif suffix == ".jsonl":
                df = pd.read_json(path, lines=True)
            elif suffix == ".json":
                df = pd.read_json(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                # .csv and anything unrecognised: CSV is the common case, and a
                # pandas parse error names the real problem better than a guess.
                df = pd.read_csv(path)
        except EvalLoadError:
            raise
        except Exception as exc:
            raise EvalLoadError(f"Could not read {path}: {exc}") from exc
    elif isinstance(data, list):
        if not data:
            raise EvalLoadError("data is an empty list.")
        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            raise EvalLoadError(f"Could not construct DataFrame from list: {exc}") from exc
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise EvalLoadError(
            f"data must be a pandas DataFrame, list[dict], or a path to a "
            f"csv/tsv/json/jsonl/parquet file; got {type(data).__name__}."
        )

    if df.empty:
        raise EvalLoadError("data is empty (no rows).")

    dupe_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dupe_cols:
        raise EvalLoadError(
            f"Duplicate column name(s) in data: {dupe_cols}. "
            "This usually comes from a bad CSV merge/export. "
            "Rename or drop the duplicate(s) before calling load_from()."
        )

    # ── apply column remapping ────────────────────────────────────────────────
    if col_map:
        unknown = [k for k in col_map if k not in df.columns]
        if unknown:
            warnings.warn(
                f"col_map references columns not found in data: {unknown}. "
                "They will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── infer canonical columns ───────────────────────────────────────────────
    col = _infer_canonical_cols(df)

    if col["item"] is None:
        raise EvalLoadError(
            "Could not find an item/input column. "
            "Expected a column named 'item', 'input', or similar. "
            "Use col_map to remap your column name."
        )

    canonical_used: set[str] = {v for v in col.values() if v}

    # ── resolve metric_cols ───────────────────────────────────────────────────
    explicit_score_types: dict[str, str] = {}

    if metric_cols is None:
        resolved = _infer_metric_cols(df, canonical_used)
        if not resolved:
            raise EvalLoadError(
                "Could not find any metric (score) columns. "
                "Add a 'score' column or specify metric_cols explicitly."
            )
    elif isinstance(metric_cols, str):
        resolved = [metric_cols]
    elif isinstance(metric_cols, list):
        resolved = list(metric_cols)
    elif isinstance(metric_cols, dict):
        resolved = list(metric_cols.keys())
        explicit_score_types = {k: v for k, v in metric_cols.items()}
    else:
        raise EvalLoadError(
            f"metric_cols must be str, list, or dict; got {type(metric_cols).__name__}."
        )

    for mc in resolved:
        if mc not in df.columns:
            raise EvalLoadError(
                f"Metric column '{mc}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

    # ── coerce metric columns to numeric ──────────────────────────────────────
    for mc in resolved:
        original = df[mc].copy()
        df[mc] = pd.to_numeric(df[mc], errors="coerce")
        n_coerced = int((original.notna() & df[mc].isna()).sum())
        if n_coerced > 0:
            warnings.warn(
                f"Coerced {n_coerced} non-numeric value(s) to NaN in column '{mc}'.",
                UserWarning,
                stacklevel=2,
            )

    # ── validate score type constraints ──────────────────────────────────────
    score_types: dict[str, ScoreType] = {}
    for mc in resolved:
        if mc in explicit_score_types:
            declared = explicit_score_types[mc]
            if declared not in _VALID_SCORE_TYPES:
                raise EvalLoadError(
                    f"Column '{mc}' declared as score type {declared!r}, which is "
                    f"not one of {sorted(_VALID_SCORE_TYPES)}."
                )
            detected = _detect_score_type(df[mc])
            # Warn if explicit type disagrees with detected type for strict cases.
            if declared == "binary" and detected != "binary":
                raise EvalLoadError(
                    f"Column '{mc}' declared as 'binary' but contains values "
                    f"outside {{0, 1}}. Detected type: '{detected}'."
                )
            score_types[mc] = declared
        else:
            score_types[mc] = _detect_score_type(df[mc])

    # ── check for duplicates ─────────────────────────────────────────────────
    # Declared factors join the key. Without them, a within-subjects design that
    # scores the same item under several conditions looks like duplicate items.
    # Only declared ones: every other leftover column is also in `factor_cols`
    # below, and putting a per-row column such as a word count in the key would
    # make almost every row unique and stop this check catching anything.
    declared_factors = [factors] if isinstance(factors, str) else list(factors or [])
    missing = [c for c in declared_factors if c not in df.columns]
    if missing:
        raise EvalLoadError(
            f"factors={missing!r} not found in the data. "
            f"Available columns: {list(df.columns)}"
        )
    key_parts = [c for c in [col.get("model"), col.get("prompt"), col.get("item")]
                 if c and c in df]
    key_parts += [c for c in declared_factors if c not in key_parts]
    if key_parts:
        _check_duplicates(df, key_parts, run_col=col.get("run"))

    # ── identify factor columns ───────────────────────────────────────────────
    metric_set = set(resolved)
    factor_cols = [
        c for c in df.columns
        if c not in canonical_used and c not in metric_set
    ]
    # Declared factors first, so anything picking "the" factor column finds the
    # one the caller named rather than an incidental leftover.
    factor_cols.sort(key=lambda c: (c not in declared_factors, list(df.columns).index(c)))

    return EvalResults(
        df=df,
        score_types=score_types,
        metric_cols=resolved,
        col=col,
        factor_cols=factor_cols,
        declared_factors=declared_factors,
        declared_score_types=dict(explicit_score_types),
    )
