"""Run-to-run disagreement plot for binary/pass-fail reliability.

Complements the ICC/instability numbers (:class:`~evalstats.core.variance.SeedVarianceResult`)
with a visual answer to "which specific items is this model unstable on?".

Ink appears only where an item's runs actually disagreed -- taller bar, more
of a split vote -- so a model that is *consistently wrong* looks exactly as
quiet as one that's *consistently right*. This deliberately keeps raw
accuracy out of the encoding: the plot is about reliability, not
correctness, matching the terminal's own noise-strip convention of using
bar height (not color) to show per-item run-to-run spread.

Typical use::

    from evalstats.vis.reliability import plot_run_disagreement
    result = es.compare(evaldata, factors="model", score_range=(0, 1))
    fig = plot_run_disagreement(result.full_analysis)
    fig.savefig("reliability.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from evalstats.core.resampling import is_binary_scores
from evalstats.vis.forest import _PALETTE

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from evalstats.core.bundles import AnalysisBundle


def plot_run_disagreement(
    bundle: "AnalysisBundle",
    *,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Bar chart of per-item run-to-run disagreement, one row per model/template.

    Requires *bundle* to carry a seed-variance decomposition, i.e. it must
    come from data with R >= 3 repeated runs (see
    :attr:`~evalstats.core.bundles.AnalysisBundle.seed_variance`), and the
    underlying scores must be binary (0/1, e.g. pass/fail or correct/
    incorrect) -- "disagreement" is defined as a split vote across runs,
    which isn't a well-defined per-item quantity for continuous scores.

    Parameters
    ----------
    bundle : AnalysisBundle
        A bundle from :func:`~evalstats.compare` or :func:`~evalstats.analyze`
        (e.g. ``result.full_analysis``). Its ``seed_variance`` supplies the
        per-model instability/ICC annotations; its ``benchmark`` supplies
        the raw per-run scores used to compute per-item disagreement.
    title : str, optional
        Plot title. A descriptive default is generated when omitted.
    figsize : tuple[float, float], optional
        Figure size. Defaults to a compact height that scales with the
        number of models/templates shown.
    ax : matplotlib.axes.Axes, optional
        Draw into this existing axes instead of creating a new figure --
        for composing this plot into a multi-panel figure alongside others.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sv = bundle.seed_variance
    if sv is None:
        raise ValueError(
            "bundle.seed_variance is None -- plot_run_disagreement requires "
            "data with R >= 3 repeated runs per item."
        )

    run_scores = bundle.benchmark.get_run_scores()  # (N, M, R)
    if not is_binary_scores(run_scores):
        raise ValueError(
            "plot_run_disagreement requires binary (0/1) scores -- "
            "'disagreement' across runs is only well-defined for pass/fail "
            "data. For continuous scores, use the instability/ICC numbers "
            "in bundle.seed_variance directly instead."
        )

    labels = list(sv.labels)
    n_models = len(labels)
    n_items = run_scores.shape[1]
    n_runs = sv.n_runs
    max_minority = n_runs // 2  # e.g. 2 for R=5 (a 3-2 split is the most divided it can get)

    order = list(np.argsort(sv.instability))
    labels_sorted = [labels[i] for i in order]
    instability_sorted = sv.instability[order]
    icc_sorted = sv.icc[order]

    n_correct = np.nansum(run_scores, axis=2)  # (N, M) -- runs that scored 1, per item
    minority = np.minimum(n_correct, n_runs - n_correct)  # (N, M), 0..max_minority
    minority_sorted = minority[order]

    ROW_H = 0.28
    GAP = 0.15
    TOP_PAD = 0.08  # headroom so a full-height bar in row 0 doesn't crowd the subtitle
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (9.5, 0.5 * n_models * (ROW_H + GAP) + 1.1)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()
    ax.set_facecolor("white")

    x = np.arange(n_items) + 0.5
    y_ticks = []
    y = TOP_PAD
    for row, m, inst, icc in zip(minority_sorted, labels_sorted, instability_sorted, icc_sorted):
        baseline = y + ROW_H  # bars grow upward from the row's bottom edge
        heights = (row / max_minority) * ROW_H if max_minority > 0 else np.zeros_like(row, dtype=float)
        ax.bar(x, heights, width=0.85, bottom=baseline - heights, color=_PALETTE["text"], zorder=2)
        # Full border around the row (not just a baseline), so the reader
        # can see exactly how much whitespace = "no disagreement" rather
        # than only implying it. A Rectangle patch (not a hand-drawn
        # polyline) keeps the stroke width visually uniform on all 4 sides.
        ax.add_patch(Rectangle(
            (0, y), n_items, ROW_H,
            fill=False, edgecolor=_PALETTE["ref_line"], linewidth=0.8, zorder=1,
        ))
        y_ticks.append(y + ROW_H / 2)
        y_mid = y + ROW_H / 2
        # Stacked (Instability above, Consistency below) instead of side by
        # side, to save horizontal space -- offset in points, not data
        # units, so the gap between the two lines stays legible regardless
        # of how compressed ROW_H is.
        ax.annotate(
            f"Instability {inst:.3f}", xy=(1.03, y_mid), xycoords=("axes fraction", "data"),
            xytext=(0, 5), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=_PALETTE["text"],
        )
        ax.annotate(
            f"Consistency (ICC) {icc*100:.0f}%", xy=(1.03, y_mid), xycoords=("axes fraction", "data"),
            xytext=(0, -5), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=_PALETTE["text"],
        )
        y += ROW_H + GAP

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels_sorted, fontsize=10, color=_PALETTE["text"])
    ax.set_xlabel("Items (evaluation order)", fontsize=10, color=_PALETTE["text"], labelpad=8)
    ax.set_xlim(0, n_items)
    ax.set_ylim(y - GAP, -0.05)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", colors=_PALETTE["text_secondary"], labelsize=9)
    ax.tick_params(axis="y", length=0)

    if title is None:
        title = "Run-to-Run Reliability"
    ax.set_title(f"{title}  |  R={n_runs} runs", fontsize=12, color=_PALETTE["text"], pad=24)
    ax.text(
        0.5, 1.02, "taller bar = the model's answer flipped more across runs; no bar = it agreed with itself every time",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color=_PALETTE["text_secondary"],
    )

    if own_fig:
        fig.subplots_adjust(right=0.62)
    return fig
