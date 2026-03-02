"""Mean advantage plot with dual uncertainty bands.

The signature promptstats visualization. For each template, shows:

- A point for the mean advantage over a reference
- A thin inner band for the bootstrap CI on the mean (epistemic uncertainty:
  "how sure are we about the mean?")
- A wider outer band for the score spread (intrinsic variance: "how
  consistent is this template?")

This separates two fundamentally different concerns:
- Epistemic uncertainty shrinks with more benchmark inputs.
- Intrinsic variance is a property of the template and won't shrink.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from promptstats.core.types import BenchmarkResult
from promptstats.core.ranking import bootstrap_mean_advantage, MeanAdvantageResult


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


def plot_mean_advantage(
    result: BenchmarkResult | MeanAdvantageResult,
    reference: str = "grand_mean",
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    spread_percentiles: tuple[float, float] = (10, 90),
    sort_by: str = "mean",
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Figure:
    """Plot mean advantage with dual uncertainty bands.

    Parameters
    ----------
    result : BenchmarkResult or MeanAdvantageResult
        Either raw benchmark data (will compute advantage internally) or
        a pre-computed MeanAdvantageResult.
    reference : str
        Reference for advantage computation. Either 'grand_mean' or a
        template label. Ignored if result is already a MeanAdvantageResult.
    n_bootstrap : int
        Bootstrap iterations. Ignored if result is a MeanAdvantageResult.
    ci : float
        Confidence level. Ignored if result is a MeanAdvantageResult.
    spread_percentiles : tuple[float, float]
        Percentiles for the spread band. Ignored if result is a
        MeanAdvantageResult.
    sort_by : str
        Sort order: 'mean' (descending by mean advantage), 'label'
        (alphabetical), or 'spread' (ascending by spread width).
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
    # Compute advantage if given raw BenchmarkResult
    if isinstance(result, BenchmarkResult):
        scores = result.get_2d_scores()
        adv = bootstrap_mean_advantage(
            scores=scores,
            labels=result.template_labels,
            reference=reference,
            n_bootstrap=n_bootstrap,
            ci=ci,
            spread_percentiles=spread_percentiles,
            rng=rng,
        )
    else:
        adv = result

    n = len(adv.labels)

    # Sort
    if sort_by == "mean":
        order = np.argsort(-adv.mean_advantages)
    elif sort_by == "label":
        order = np.argsort(adv.labels)
    elif sort_by == "spread":
        spread_widths = adv.spread_high - adv.spread_low
        order = np.argsort(spread_widths)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    labels = [adv.labels[i] for i in order]
    means = adv.mean_advantages[order]
    ci_lo = adv.bootstrap_ci_low[order]
    ci_hi = adv.bootstrap_ci_high[order]
    sp_lo = adv.spread_low[order]
    sp_hi = adv.spread_high[order]

    # Figure setup
    if figsize is None:
        figsize = (10, max(3, 0.5 * n + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)

    # Band thicknesses (line-based for a cleaner look)
    spread_lw = 10
    ci_lw = 5

    # Draw zero reference line
    ax.axvline(x=0, color=_PALETTE["zero_line"], linewidth=1.2, zorder=1)

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

        # Outer band: intrinsic spread (10th-90th percentile)
        ax.plot(
            [sp_lo[i], sp_hi[i]],
            [y, y],
            color=_PALETTE["spread_band"],
            linewidth=spread_lw,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [sp_lo[i], sp_hi[i]],
            [y, y],
            color=_PALETTE["spread_edge"],
            linewidth=1.0,
            solid_capstyle="round",
            zorder=2.2,
            alpha=0.9,
        )

        # Inner band: bootstrap CI on the mean
        ax.plot(
            [ci_lo[i], ci_hi[i]],
            [y, y],
            color=_PALETTE["ci_band"],
            linewidth=ci_lw,
            solid_capstyle="round",
            zorder=3,
        )

        # Center point: mean advantage
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

    ax.set_xlabel(
        f"Advantage over {adv.reference}",
        fontsize=10,
        color=_PALETTE["text"],
        labelpad=8,
    )

    # x-limits with breathing room
    all_x = np.concatenate([sp_lo, sp_hi, ci_lo, ci_hi, means])
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
        ref_desc = (
            "grand mean" if adv.reference == "grand_mean"
            else f"'{adv.reference}'"
        )
        title = f"Mean advantage over {ref_desc}"

    ax.set_title(
        title,
        fontsize=12,
        color=_PALETTE["text"],
        pad=12,
        loc="left",
        fontweight="semibold",
    )

    # Legend
    sp_lo_pct, sp_hi_pct = adv.spread_percentiles
    legend_handles = [
        Line2D(
            [0], [0],
            color=_PALETTE["spread_band"],
            linewidth=spread_lw,
            solid_capstyle="round",
            label=f"Score spread ({sp_lo_pct}th–{sp_hi_pct}th pctl)",
        ),
        Line2D(
            [0], [0],
            color=_PALETTE["ci_band"],
            linewidth=ci_lw,
            solid_capstyle="round",
            label=f"{int(ci * 100)}% CI on mean (bootstrap, n={adv.n_bootstrap:,})",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=_PALETTE["point_pos"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=6,
            label="Mean advantage",
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

    # Annotation explaining the two bands
    ax.annotate(
        "light band = template spread · dark band = uncertainty on mean",
        xy=(0.5, -0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=7.5,
        color=_PALETTE["text_secondary"],
        style="italic",
    )

    fig.tight_layout()
    return fig
