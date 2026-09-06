"""Absolute robustness interval plot.

Plots per-template absolute point estimates from ``RobustnessResult.mean`` with
their marginal confidence intervals from ``ci_low``/``ci_high``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from evalstats.core.types import BenchmarkResult
from evalstats.core.variance import RobustnessResult, robustness_metrics


# -- Color palette --
_PALETTE = {
    "spread_band": "#DCE8F5",       # soft blue — intrinsic spread
    "spread_edge": "#B8CCE3",       # muted blue — spread edge
    "ci_band": "#3A78A4",           # medium-dark blue — CI band
    "ci_edge": "#2B5F85",           # darker blue — CI edge
    "point_pos": "#1E5A85",         # deep blue — positive mean
    "point_neg": "#A34A63",         # muted rose — negative mean
    "point_zero": "#5C6470",        # cool gray — CI overlaps zero
    "zero_line": "#D4D8DE",         # soft gray — zero reference
    "grid": "#EEF1F4",              # very light gray — x grid
    "row_alt": "#FAFBFC",           # alternating row background
    "text": "#2D333B",              # dark slate — labels
    "text_secondary": "#6B7280",    # muted gray — secondary text
}


def plot_point_estimates(
    result: Union[BenchmarkResult, RobustnessResult],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    sort_by: str = "mean",
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Figure:
    """Plot absolute performance means with marginal confidence intervals.

    Parameters
    ----------
    result : BenchmarkResult or RobustnessResult
        Either raw benchmark data (computed internally) or a pre-computed
        robustness result.
    n_bootstrap : int
        Bootstrap iterations used when computing robustness from raw data.
    ci : float
        Confidence level used when computing robustness from raw data.
    sort_by : str
        Sort order: 'mean' (descending), 'label' (alphabetical),
        or 'ci_width' (ascending).
    figsize : tuple[float, float], optional
        Figure size. Defaults to (10, 0.5 * N_templates + 1.5).
    title : str, optional
        Plot title. Defaults to a descriptive title.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Compute robustness if given raw BenchmarkResult.
    computed_here = isinstance(result, BenchmarkResult)
    if computed_here:
        scores = result.get_2d_scores()
        rob = robustness_metrics(
            scores, result.template_labels,
            n_bootstrap=n_bootstrap, rng=rng, alpha=1.0 - ci,
            statistic="mean", marginal_method="smooth_bootstrap",
        )
    else:
        rob = result

    n = len(rob.labels)

    # Sort
    if sort_by == "mean":
        order = np.argsort(-rob.mean)
    elif sort_by == "label":
        order = np.argsort(rob.labels)
    elif sort_by == "ci_width":
        if rob.ci_low is None or rob.ci_high is None:
            order = np.argsort(-rob.mean)
        else:
            order = np.argsort(rob.ci_high - rob.ci_low)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    labels = [rob.labels[i] for i in order]
    means = rob.mean[order]
    if rob.ci_low is None or rob.ci_high is None:
        ci_lo = means.copy()
        ci_hi = means.copy()
    else:
        ci_lo = rob.ci_low[order]
        ci_hi = rob.ci_high[order]

    # Figure setup
    if figsize is None:
        figsize = (10, max(3, 0.5 * n + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)

    # Band thicknesses (line-based for a cleaner look)
    ci_lw = 5

    # Draw global mean reference line.
    global_mean = float(np.mean(rob.mean))
    ax.axvline(x=global_mean, color=_PALETTE["zero_line"], linewidth=1.2, zorder=1)

    # Subtle alternating rows to improve readability
    for i, y in enumerate(y_positions):
        if i % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color=_PALETTE["row_alt"], zorder=0)

    for i, y in enumerate(y_positions):
        # Determine point color based on significance
        if ci_lo[i] > 0:
            point_color = _PALETTE["point_pos"]
        elif ci_hi[i] < 0:
            point_color = _PALETTE["point_neg"]
        else:
            point_color = _PALETTE["point_zero"]

        # Marginal CI on the mean
        ax.plot(
            [ci_lo[i], ci_hi[i]],
            [y, y],
            color=_PALETTE["ci_band"],
            linewidth=ci_lw,
            solid_capstyle="round",
            zorder=3,
        )

        # Center point: mean value
        ax.plot(
            means[i], y,
            "o",
            color=point_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=6.5,
            zorder=4,
        )

        # Significance indicator
        if ci_lo[i] > 0 or ci_hi[i] < 0:
            ax.plot(
                means[i], y,
                "o",
                color="white",
                markersize=2.8,
                zorder=5,
            )

    # Axis configuration
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10, color=_PALETTE["text"])
    ax.invert_yaxis()  # best template at top

    ax.set_xlabel("Absolute score", fontsize=10, color=_PALETTE["text"], labelpad=8)

    # x-limits with breathing room
    all_x = np.concatenate([ci_lo, ci_hi, means])
    x_min, x_max = np.min(all_x), np.max(all_x)
    if np.isclose(x_min, x_max):
        pad = 0.1 if np.isclose(x_min, 0.0) else max(0.05 * abs(x_min), 0.1)
    else:
        pad = 0.08 * (x_max - x_min)
    ax.set_xlim(x_min - pad, x_max + pad)

    # Grid
    ax.xaxis.grid(True, color=_PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)

    # Remove spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_PALETTE["zero_line"])

    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=_PALETTE["text_secondary"], labelsize=9)

    # Title
    if title is None:
        title = "Absolute Performance (Robustness Means)"

    ax.set_title(
        title,
        fontsize=12,
        color=_PALETTE["text"],
        pad=12,
        loc="left",
        fontweight="semibold",
    )

    # Legend
    legend_handles = [
        Line2D(
            [0], [0],
            color=_PALETTE["ci_band"],
            linewidth=ci_lw,
            solid_capstyle="round",
            label=(
                f"{int(ci * 100)}% marginal CI on mean (n={n_bootstrap:,})"
                if computed_here
                else "Marginal CI on mean"
            ),
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=_PALETTE["point_pos"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=6,
            label="Mean score",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor=_PALETTE["grid"],
        framealpha=0.95,
        handlelength=2.5,
    )

    # Annotation
    ax.annotate(
        "bars = uncertainty on mean; line = global mean",
        xy=(0.5, -0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=7.5,
        color=_PALETTE["text_secondary"],
        style="italic",
    )

    fig.tight_layout()
    return fig
