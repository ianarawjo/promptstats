"""Raw accuracy bar chart ("scoreboard") for prompt or model comparisons.

A simple first-look visualization before statistical testing.  Shows mean
accuracy per entity as a bar, with an optional dashed baseline reference
line to make relative gains immediately visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ._palette import GRID, TEXT, TEXT_SECONDARY

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Color palette (consistent with the rest of the vis module)
# ---------------------------------------------------------------------------

_PALETTE = {
    "bar":           "#94b8e0",  # soft blue   — bars
    "bar_highlight": "#4a90d9",  # medium blue — highlighted bar
    "baseline_line": "#777777",  # mid gray    — baseline reference
    "grid":          GRID,
    "text":          TEXT,
    "text_secondary":TEXT_SECONDARY,
    "errorbar":      TEXT,
}


def plot_accuracy_bar(
    scores: Union[dict, "CompareReport"],  # noqa: F821
    baseline: Optional[str] = None,
    cis: Optional[Mapping[str, Sequence[float]]] = None,
    sort_by: str = "input_order",
    as_percent: bool = True,
    score_range: Optional[tuple[float, float]] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Bar chart of mean accuracy per prompt or model (raw scoreboard).

    A quick, un-corrected view intended as a starting point before
    statistical analysis.  To visualise confidence intervals and
    significance tiers, use :func:`plot_ci_forest` instead.

    Parameters
    ----------
    scores : dict or CompareReport
        One of:

        * ``dict[str, float]`` — pre-computed mean per entity.
        * ``dict[str, array-like]`` — raw score arrays; means are computed
          internally.
        * :class:`~evalstats.compare.CompareReport` — uses the
          ``entity_stats`` means from the report.
    baseline : str, optional
        Label of a baseline entity.  A dashed reference line is drawn at
        its mean accuracy so gains and losses over the baseline are
        immediately visible.
    cis : mapping[str, sequence[float]], optional
        Optional confidence intervals per entity label, used to draw
        vertical error bars on each bar.

        Expected format: ``{label: (ci_low, ci_high)}`` in raw score units
        (0-1), regardless of ``as_percent``.
    sort_by : str
        Bar ordering:

        * ``"input_order"`` (default) — preserves the dict / label order.
        * ``"mean"`` — descending by mean; best entity leftmost.
        * ``"label"`` — alphabetical.
    as_percent : bool
        When ``True`` (default), display values as percentages (0–100).
        Set to ``False`` to keep raw scores in their native units.
    score_range : tuple[float, float], optional
        The metric's true (min, max) range, e.g. ``(1, 5)`` for a 5-point
        Likert scale or ``(0, 100)`` for a percentage grade. When omitted,
        the y-axis assumes accuracy/probability data in ``[0, 1]``
        (``[0, 100]`` when ``as_percent=True``) -- the right default for
        binary pass/fail scores, but wrong for any other score type. Data
        outside the assumed range is never clipped (the axis auto-expands
        to fit it instead), but pass this explicitly for non-binary scores
        so the axis reflects the metric's real bounds rather than an
        expanded guess.
    figsize : tuple[float, float], optional
        Figure size.  Defaults to ``(max(5, 0.9 * N + 1.5), 3.8)``.
    title : str, optional
        Plot title.  A descriptive default is generated when omitted.
    ax : Axes, optional
        Existing axes to draw into.  A new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---- normalise input --------------------------------------------------
    entity_name = "entity"

    if isinstance(scores, dict):
        labels = list(scores.keys())
        means_raw = {}
        for label, val in scores.items():
            arr = np.asarray(val, dtype=np.float64)
            means_raw[label] = float(np.nanmean(arr)) if arr.ndim > 0 else float(arr)
    elif hasattr(scores, "labels") and hasattr(scores, "entity_stats"):
        # ComparisonResult or any duck-type equivalent (previously CompareReport)
        entity_name = getattr(scores, "entity_name_singular", "entity")
        labels = scores.labels
        means_raw = {l: scores.entity_stats[l].mean for l in labels}
    else:
        raise TypeError(
            "scores must be a dict or a ComparisonResult. "
            f"Got {type(scores).__name__!r}."
        )

    n = len(labels)
    if n == 0:
        raise ValueError("scores is empty.")

    # ---- sort order -------------------------------------------------------
    means_arr = np.array([means_raw[l] for l in labels])
    if sort_by == "input_order":
        order = list(range(n))
    elif sort_by == "mean":
        order = list(np.argsort(-means_arr))
    elif sort_by == "label":
        order = sorted(range(n), key=lambda i: labels[i])
    else:
        raise ValueError(
            f"Unknown sort_by: {sort_by!r}. "
            "Expected 'input_order', 'mean', or 'label'."
        )

    ordered_labels = [labels[i] for i in order]
    scale = 100.0 if as_percent else 1.0
    ordered_means = [means_raw[l] * scale for l in ordered_labels]

    yerr = None
    if cis is not None:
        missing = [label for label in ordered_labels if label not in cis]
        if missing:
            raise ValueError(
                "cis is missing labels present in scores: "
                f"{missing}."
            )

        yerr_low = []
        yerr_high = []
        for label in ordered_labels:
            ci_bounds = cis[label]
            if len(ci_bounds) != 2:
                raise ValueError(
                    f"cis[{label!r}] must be a pair (ci_low, ci_high)."
                )

            ci_low = float(ci_bounds[0])
            ci_high = float(ci_bounds[1])
            mean = means_raw[label]

            if ci_low > ci_high:
                raise ValueError(
                    f"cis[{label!r}] has ci_low > ci_high: "
                    f"({ci_low}, {ci_high})."
                )

            if not (ci_low <= mean <= ci_high):
                raise ValueError(
                    f"Mean for {label!r} ({mean:.6g}) is outside the provided "
                    f"CI ({ci_low:.6g}, {ci_high:.6g})."
                )

            yerr_low.append((mean - ci_low) * scale)
            yerr_high.append((ci_high - mean) * scale)

        yerr = np.vstack([yerr_low, yerr_high])

    # ---- figure setup -----------------------------------------------------
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (max(5.0, 0.9 * n + 1.5), 3.8)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("white")

    # ---- bars -------------------------------------------------------------
    x = np.arange(n)
    ax.bar(
        x,
        ordered_means,
        color=_PALETTE["bar"],
        width=0.6,
        yerr=yerr,
        ecolor=_PALETTE["errorbar"],
        capsize=3,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "zorder": 5},
        zorder=3,
    )

    # ---- baseline reference line ------------------------------------------
    if baseline is not None:
        if baseline not in means_raw:
            raise ValueError(
                f"baseline label {baseline!r} not found in scores. "
                f"Available labels: {list(means_raw.keys())}"
            )
        baseline_val = means_raw[baseline] * scale
        ax.axhline(
            baseline_val,
            color=_PALETTE["baseline_line"],
            lw=1.2, ls="--",
            label=f"Baseline ({baseline})",
            zorder=4,
        )
        ax.legend(fontsize=8)

    # ---- axes styling -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        ordered_labels, rotation=35, ha="right",
        fontsize=9, color=_PALETTE["text"],
    )
    ax.set_ylabel(
        "Accuracy (%)" if as_percent else "Score",
        fontsize=10, color=_PALETTE["text"],
    )

    # ---- y-axis limits -----------------------------------------------------
    # The natural floor/ceiling for the metric: an explicit score_range when
    # given, else the accuracy/probability assumption ([0, 1] or [0, 100])
    # that's right for binary pass/fail data but wrong for anything else
    # (e.g. a 1-5 Likert score plotted with as_percent=False used to get
    # silently clipped to a [0, 1] axis -- every bar rendered as a flat
    # line at the top instead of its real height). Data is never clipped:
    # when it exceeds the assumed/declared range the axis expands to fit
    # it; when it fits, the axis keeps the previous tight, exact bounds.
    if score_range is not None:
        floor, ceiling = score_range[0] * scale, score_range[1] * scale
    else:
        floor, ceiling = (0.0, 100.0) if as_percent else (0.0, 1.0)

    all_vals = list(ordered_means)
    if yerr is not None:
        all_vals += [m + hi for m, hi in zip(ordered_means, yerr[1])]
        all_vals += [m - lo for m, lo in zip(ordered_means, yerr[0])]
    if baseline is not None:
        all_vals.append(baseline_val)
    data_lo, data_hi = min(all_vals), max(all_vals)

    if data_lo >= floor and data_hi <= ceiling:
        axis_lo, axis_hi = floor, ceiling
    else:
        span = max(data_hi - data_lo, 1e-9)
        pad = 0.06 * span
        axis_lo, axis_hi = min(floor, data_lo - pad), max(ceiling, data_hi + pad)

    ax.set_ylim(axis_lo, axis_hi)

    if as_percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax.grid(axis="y", color=_PALETTE["grid"], alpha=0.9, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=_PALETTE["text_secondary"], labelsize=9)

    # ---- title ------------------------------------------------------------
    if title is None:
        title = f"Mean accuracy per {entity_name}  (raw scoreboard, no correction)"

    ax.set_title(
        title,
        fontsize=10, color=_PALETTE["text"],
        pad=10, loc="center",
    )

    if own_fig:
        fig.tight_layout()

    return fig
