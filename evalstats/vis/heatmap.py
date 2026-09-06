"""Model × Prompt mean-score heatmap.

Visualises the mean score for each (model, prompt) cell as a colour-coded
grid, with numeric annotations in every cell.  Rows are models, columns are
prompt templates.

Typical use::

    from evalstats.vis.heatmap import plot_model_prompt_heatmap
    fig = plot_model_prompt_heatmap(benchmark, metric_name="Originality")
    fig.savefig("heatmap.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from evalstats.core.types import MultiModelBenchmark
from ._palette import TEXT, TEXT_SECONDARY


def plot_model_prompt_heatmap(
    data: Union[MultiModelBenchmark, np.ndarray],
    *,
    model_labels: Optional[list[str]] = None,
    prompt_labels: Optional[list[str]] = None,
    metric_name: str = "Score",
    title: Optional[str] = None,
    sort_models: str = "mean_desc",
    sort_prompts: str = "input_order",
    cmap: str = "viridis",
    annot_fmt: str = ".2f",
    annot_fontsize: float = 9.0,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Heatmap of mean scores across models (rows) and prompts (columns).

    Parameters
    ----------
    data : MultiModelBenchmark or np.ndarray
        Either a :class:`~evalstats.core.types.MultiModelBenchmark` (the
        model and prompt labels are taken from it) or a pre-computed 2-D
        ``(n_models, n_prompts)`` mean-score array.  When passing an array,
        supply *model_labels* and *prompt_labels* explicitly.
    model_labels : list[str], optional
        Row labels.  Required when *data* is an ndarray; ignored otherwise.
    prompt_labels : list[str], optional
        Column labels.  Required when *data* is an ndarray; ignored otherwise.
    metric_name : str
        Human-readable name for the score shown in the colourbar label and
        default title (e.g. ``"Originality"``, ``"Accuracy"``).
    title : str, optional
        Figure title.  A descriptive default is generated when omitted.
    sort_models : str
        Row ordering:

        * ``"mean_desc"`` (default) — descending by row mean; best model top.
        * ``"mean_asc"`` — ascending by row mean.
        * ``"label"`` — alphabetical.
        * ``"input_order"`` — keep original order.
    sort_prompts : str
        Column ordering:

        * ``"input_order"`` (default) — keep original order.
        * ``"mean_desc"`` — descending by column mean.
        * ``"mean_asc"`` — ascending by column mean.
        * ``"label"`` — alphabetical.
    cmap : str
        Matplotlib colourmap name.  Defaults to ``"viridis"``.
    annot_fmt : str
        Format string for cell annotations (passed to Python's ``format``).
        Defaults to ``".2f"``.
    annot_fontsize : float
        Font size for the numeric cell annotations.
    figsize : tuple[float, float], optional
        Figure size in inches.  Defaults to
        ``(max(6, 0.9 * n_prompts + 2), max(4, 0.55 * n_models + 1.5))``.
    ax : matplotlib.axes.Axes, optional
        Draw into this existing axes instead of creating a new figure --
        for composing this plot into a multi-panel figure alongside others.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ------------------------------------------------------------------ #
    # Normalise input                                                      #
    # ------------------------------------------------------------------ #
    if isinstance(data, MultiModelBenchmark):
        matrix = data._get_3d_cell_means().mean(axis=2)   # (P, N, M) -> (P, N)
        row_labels = list(data.model_labels)
        col_labels = list(data.template_labels)
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(
                f"When passing an ndarray, it must be 2-D (n_models, n_prompts); "
                f"got shape {data.shape}."
            )
        matrix = np.asarray(data, dtype=np.float64)
        if model_labels is None or prompt_labels is None:
            raise ValueError(
                "model_labels and prompt_labels are required when data is an ndarray."
            )
        row_labels = list(model_labels)
        col_labels = list(prompt_labels)
        if len(row_labels) != matrix.shape[0]:
            raise ValueError(
                f"model_labels length ({len(row_labels)}) does not match "
                f"matrix rows ({matrix.shape[0]})."
            )
        if len(col_labels) != matrix.shape[1]:
            raise ValueError(
                f"prompt_labels length ({len(col_labels)}) does not match "
                f"matrix columns ({matrix.shape[1]})."
            )
    else:
        raise TypeError(
            "data must be a MultiModelBenchmark or a 2-D np.ndarray. "
            f"Got {type(data).__name__!r}."
        )

    n_models, n_prompts = matrix.shape

    # ------------------------------------------------------------------ #
    # Sorting                                                              #
    # ------------------------------------------------------------------ #
    def _sort_order(labels, means, mode):
        if mode == "mean_desc":
            return list(np.argsort(-means))
        if mode == "mean_asc":
            return list(np.argsort(means))
        if mode == "label":
            return sorted(range(len(labels)), key=lambda i: labels[i])
        if mode == "input_order":
            return list(range(len(labels)))
        raise ValueError(
            f"Unknown sort mode {mode!r}. "
            "Expected 'mean_desc', 'mean_asc', 'label', or 'input_order'."
        )

    row_order = _sort_order(row_labels, matrix.mean(axis=1), sort_models)
    col_order = _sort_order(col_labels, matrix.mean(axis=0), sort_prompts)

    matrix = matrix[np.ix_(row_order, col_order)]
    row_labels = [row_labels[i] for i in row_order]
    col_labels = [col_labels[i] for i in col_order]

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    if figsize is None:
        w = max(6.0, 0.9 * n_prompts + 2.5)
        h = max(4.0, 0.55 * n_models + 1.5)
        figsize = (w, h)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()
    ax.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Heatmap                                                              #
    # ------------------------------------------------------------------ #
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # ------------------------------------------------------------------ #
    # Cell annotations                                                     #
    # ------------------------------------------------------------------ #
    # Choose text colour (light or dark) based on relative cell brightness
    # so the numbers stay readable over both dark and bright cells.
    norm_matrix = (matrix - vmin) / max(vmax - vmin, 1e-12)

    for r in range(n_models):
        for c in range(n_prompts):
            val = matrix[r, c]
            if np.isnan(val):
                continue
            brightness = norm_matrix[r, c]
            text_color = "white" if brightness < 0.55 else "#1a1a1a"
            ax.text(
                c, r,
                format(val, annot_fmt),
                ha="center", va="center",
                fontsize=annot_fontsize,
                color=text_color,
                fontweight="normal",
            )

    # ------------------------------------------------------------------ #
    # Axes ticks and labels                                                #
    # ------------------------------------------------------------------ #
    ax.set_xticks(np.arange(n_prompts))
    ax.set_xticklabels(col_labels, fontsize=9, color=TEXT)

    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(row_labels, fontsize=9, color=TEXT, ha="right")

    ax.set_xlabel("Prompt", fontsize=10, color=TEXT, labelpad=8)
    ax.set_ylabel("Model", fontsize=10, color=TEXT, labelpad=8)

    ax.tick_params(axis="both", length=0, pad=5)

    # Minor grid lines to delineate cells
    ax.set_xticks(np.arange(-0.5, n_prompts, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    # ------------------------------------------------------------------ #
    # Colourbar                                                            #
    # ------------------------------------------------------------------ #
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(
        f"Avg {metric_name} Score",
        fontsize=9,
        color=TEXT,
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cbar.outline.set_visible(False)

    # ------------------------------------------------------------------ #
    # Title                                                                #
    # ------------------------------------------------------------------ #
    if title is None:
        title = f"Mean {metric_name}: Model \u00d7 Prompt"

    ax.set_title(
        title,
        fontsize=12,
        color=TEXT,
        pad=12,
        fontweight="semibold",
    )

    if own_fig:
        fig.tight_layout()
    return fig
