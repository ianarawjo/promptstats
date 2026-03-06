"""Critical difference diagram for Nemenyi post-hoc comparisons.

Delegates rendering to ``scikit_posthocs.critical_difference_diagram``,
which implements the Demšar (2006) layout.  This module is responsible for
translating a :class:`~promptstats.core.paired.FriedmanResult` into the
inputs that function expects (a rank dict and a symmetric p-value DataFrame)
and for wrapping the result in a properly sized, titled Figure.

References
----------
Demšar, J. (2006). Statistical comparisons of classifiers over multiple
data sets. Journal of Machine Learning Research, 7, 1–30.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import studentized_range

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from promptstats.core.paired import FriedmanResult


def _critical_difference(k: int, n: int, alpha: float) -> float:
    """CD threshold for Nemenyi at the given alpha level (Demšar 2006).

        CD = (q_{alpha,k,∞} / √2) · √(k(k+1) / 6N)
    """
    q = float(studentized_range.ppf(1.0 - alpha, k, np.inf)) / np.sqrt(2.0)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    return q * se


def _sig_matrix(friedman: FriedmanResult) -> pd.DataFrame:
    """Build a symmetric p-value DataFrame from the upper-triangle nemenyi_p.

    Diagonal entries are set to 1.0 (a treatment is not significantly
    different from itself).  ``scikit_posthocs.critical_difference_diagram``
    treats values *below* ``alpha`` as significant.
    """
    labels = list(friedman.avg_ranks.keys())
    mat = pd.DataFrame(1.0, index=labels, columns=labels)
    for (a, b), p in friedman.nemenyi_p.items():
        mat.loc[a, b] = p
        mat.loc[b, a] = p
    return mat


def plot_critical_difference(
    friedman: FriedmanResult,
    alpha: float = 0.05,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Plot a critical difference diagram for Nemenyi post-hoc comparisons.

    Uses :func:`scikit_posthocs.critical_difference_diagram` for rendering
    (Demšar 2006 layout: rank axis, left/right labels, crossbars for
    non-significant groups, CD bracket).

    Parameters
    ----------
    friedman : FriedmanResult
        Output of :func:`promptstats.friedman_nemenyi`.
    alpha : float
        Significance threshold for crossbar grouping (default 0.05).
    figsize : tuple[float, float], optional
        Figure size.  Defaults to ``(max(7, k + 3), 3.5)``.
    title : str, optional
        Plot title.  A descriptive default is generated when omitted.
    ax : Axes, optional
        Existing axes to draw into.  A new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        from scikit_posthocs import critical_difference_diagram
    except ImportError as exc:
        raise ImportError(
            "scikit-posthocs is required for plot_critical_difference. "
            "Install it with: pip install scikit-posthocs"
        ) from exc

    k = friedman.n_templates
    n = friedman.n_inputs

    ranks = friedman.avg_ranks                   # dict[str, float]
    sig_mat = _sig_matrix(friedman)              # symmetric DataFrame
    cd = _critical_difference(k, n, alpha)

    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (max(7.0, k * 0.9 + 3.0), 4.0)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    critical_difference_diagram(ranks, sig_mat, ax=ax, alpha=alpha)

    if title is None:
        title = (
            f"Critical Difference Diagram  ·  "
            f"Friedman χ²({friedman.df}) = {friedman.statistic:.2f},  "
            f"p = {friedman.p_value:.3g}"
        )
    ax.set_title(
        title,
        fontsize=10, pad=10, loc="left", fontweight="semibold",
    )

    if own_fig:
        fig.tight_layout()

    return fig
