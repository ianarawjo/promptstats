"""High-level comparison API: compare(), ComparisonResult, and thin wrappers.

Provides the new spec API on top of the existing statistical engine:

    evaldata = load_from(df)
    result   = compare(evaldata, factors="model", metric="score")
    result.summary()
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm

from evalstats.loader import EvalResults, EvalLoadError, load_from, _scores_dict_to_df
from evalstats.io import from_dataframe
from evalstats.config import get_alpha_ci, GRADIENT_CI_ALPHAS, MIN_SAMPLE_FLOOR
from evalstats.core.router import analyze, analyze_factorial
from evalstats.core.bundles import AnalysisBundle, MultiModelBundle, AnalysisResult
from evalstats.core.design import detect_paired
from evalstats.core.unpaired import compare_unpaired, GroupComparisonResult
from evalstats.core.stats_utils import correct_pvalues
from evalstats.core.summary import (
    print_analysis_summary,
    print_brief_summary,
    _assign_significance_groups,
    _UNSET as _SUMMARY_UNSET,
)


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonResult
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonResult:
    """Statistical comparison results returned by :func:`compare`.

    Wraps the underlying :class:`~evalstats.core.bundles.AnalysisBundle` or
    :class:`~evalstats.core.bundles.MultiModelBundle` with the new spec API.

    Call :meth:`summary` to print the full terminal output (with gradient CI
    plots), :meth:`to_frame` to get DataFrames for downstream work, or
    :meth:`to_dict` for a JSON-friendly representation.
    """

    def __init__(
        self,
        analysis: AnalysisResult,
        *,
        factors: Union[str, list[str]],
        metric: str,
        baseline: Optional[str],
        alpha: float,
        filtered_df: pd.DataFrame,
        _mmb_view: Literal["model_level", "template_level", "cross_model"] = "model_level",
        min_meaningful_diff: Optional[float] = None,
        show_rank_probabilities: bool = False,
    ):
        self._analysis = analysis
        self._factors = factors
        self._metric = metric
        self._baseline = baseline
        self._alpha = alpha
        self._df = filtered_df
        self._mmb_view = _mmb_view  # which MultiModelBundle view is primary
        self._min_meaningful_diff = min_meaningful_diff
        self._variance_components: Optional[dict] = None  # set by MC alignment loop
        self._pareto: Optional[dict] = None  # set by _run_pareto_if_needed when secondary_metric= is passed
        # Bootstrap P(Best)/E[Rank] output is opt-in, not opt-out: it reads as
        # a confident, almost authoritative verdict (e.g. "63.6% probability
        # of being best") even when the underlying CIs overlap heavily and the
        # entities are statistically indistinguishable -- in practice this
        # single number gets weighed far more than the wider, more honest CI
        # picture sitting right next to it. Ranking is still computed
        # (bootstrap_ranks() runs unconditionally in router.analyze()) so
        # .rank_dist stays available for programmatic use either way; this
        # flag only controls whether summary()/to_dict()/to_frame() surface it.
        self._show_rank_probabilities = show_rank_probabilities

    # ── print methods ────────────────────────────────────────────────────────

    _UNSET = object()  # sentinel distinguishing "not passed" from None

    def summary(
        self,
        *,
        top_pairwise: Optional[int] = None,
        style: Literal["gradient", "line"] = "gradient",
        p_value_method=_UNSET,
        pairwise_sort: Literal["grouped", "significance"] = "grouped",
        show_rank_probabilities: Optional[bool] = None,
    ) -> None:
        """Print the full terminal summary with gradient CI plots.

        This delegates directly to the existing ``print_analysis_summary``
        which produces the gradient multi-band CI plots that are the
        signature output of evalstats.

        Parameters
        ----------
        top_pairwise : int, optional
            Number of pairwise comparisons to show. None shows all.
        style : {"gradient", "line"}
            CI plot style. ``"gradient"`` (default) renders multi-band opacity
            plots; ``"line"`` uses the classic dot-and-line style.
        p_value_method : str or None, optional
            Override the p-value display method.  When not passed (default),
            uses the method stored in the bundle (from ``p_values=`` /
            ``pairwise_test=`` kwargs).  Pass ``None`` to explicitly suppress,
            or a string like ``"wsr"`` / ``"boot"`` to force a column.
        pairwise_sort : {"grouped", "significance"}
            Pairwise row ordering. ``"grouped"`` (default) keeps related pairs
            together; ``"significance"`` sorts by p-value then effect size.
        show_rank_probabilities : bool, optional
            Print the bootstrap "Rank Probabilities" block (P(Best)/E[Rank]
            per entity). Off by default (see ``compare(...,
            show_rank_probabilities=)``) -- overrides that default for this
            call only when passed explicitly.
        """
        # Map our sentinel to the summary module's _UNSET so it reads from bundle.
        pvm = _SUMMARY_UNSET if p_value_method is ComparisonResult._UNSET else p_value_method
        show_rank = self._show_rank_probabilities if show_rank_probabilities is None else show_rank_probabilities
        item_singular, item_plural = _factor_item_labels(self._factors)
        print_analysis_summary(
            self._analysis,
            top_pairwise=top_pairwise,
            style=style,
            p_value_method=pvm,
            pairwise_sort=pairwise_sort,
            show_rank_probabilities=show_rank,
            min_meaningful_diff=self._min_meaningful_diff,
            item_singular=item_singular,
            item_plural=item_plural,
            pareto=self._pareto,
            metric=self._metric,
        )

    def brief(self) -> None:
        """Print a compact one-line-per-entity summary."""
        item_singular, item_plural = _factor_item_labels(self._factors)
        print_brief_summary(self._analysis, item_singular=item_singular, item_plural=item_plural)

    def print_ci_table(self, sort_by: str = "mean", as_percent: bool = True) -> None:
        """Print a compact table of entity means and confidence intervals.

        Parameters
        ----------
        sort_by : str
            Row ordering: ``"mean"`` (default, descending), ``"label"``
            (alphabetical), or ``"input_order"`` (preserves label order).
        as_percent : bool
            When ``True`` (default), display values as percentages (0–100).
        """
        bundle = self._primary_bundle()
        if bundle is None:
            print("No analysis available.")
            return

        ci_pct = int((1 - self._alpha) * 100)
        entity_name = self.entity_name_singular.capitalize()
        labels = list(bundle.benchmark.template_labels)
        rob = bundle.robustness
        unbeaten_set = set(self.unbeaten or [])

        rows = []
        for i, lbl in enumerate(labels):
            mean = float(rob.mean[i])
            ci_lo = float(rob.ci_low[i]) if rob.ci_low is not None else None
            ci_hi = float(rob.ci_high[i]) if rob.ci_high is not None else None
            rows.append((str(lbl), mean, ci_lo, ci_hi))

        if sort_by == "mean":
            rows.sort(key=lambda r: r[1], reverse=True)
        elif sort_by == "label":
            rows.sort(key=lambda r: r[0])

        scale = 100.0 if as_percent else 1.0
        pct = "%" if as_percent else ""
        col_w = max((len(r[0]) for r in rows), default=8)
        col_w = max(col_w, len(entity_name))

        header = f"  {entity_name:<{col_w}}  {'Mean':>7}  {ci_pct}% CI                  Status"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lbl, mean, lo, hi in rows:
            if lo is not None and hi is not None:
                ci_str = f"[{lo*scale:.1f}{pct}, {hi*scale:.1f}{pct}]"
            else:
                ci_str = "—"
            status = "✓" if lbl in unbeaten_set else "—"
            print(f"  {lbl:<{col_w}}  {mean*scale:>6.1f}{pct}  {ci_str:<22}  {status}")

    def print_pair(self, entity_a: str, entity_b: str) -> None:
        """Print the pairwise comparison summary for a specific pair.

        Parameters
        ----------
        entity_a, entity_b : str
            Labels of the two entities to compare.  The pair must be present
            in the pairwise matrix (order does not matter).
        """
        pw = self.pairwise
        if pw is None:
            print("No pairwise analysis available.")
            return
        pair = pw.get(entity_a, entity_b)
        if pair is None:
            print(f"No pairwise comparison found for ({entity_a!r}, {entity_b!r}).")
            return
        pair.summary()

    def plot(self, method: str = "forest", **kwargs):
        """Visualize comparison results.

        Parameters
        ----------
        method : str
            Plot type:

            * ``"forest"`` (default) — horizontal CI forest plot via
              :func:`~evalstats.vis.forest.plot_ci_forest`, gradient-banded
              (68/90/95/99% nested confidence bands) by default -- the same
              richer CI picture the terminal's ``.summary()`` gradient plot
              already shows, in matplotlib. Pass ``style="single"`` to fall
              back to one plain CI band per entity, or ``color_rule="factor"``
              / a colour name to color by entity identity instead of
              significance tier.
            * ``"bar"`` — accuracy bar chart via
              :func:`~evalstats.vis.scoreboard.plot_accuracy_bar`. A quick,
              uncorrected view (no CIs) -- useful before statistical
              analysis, not as a substitute for it.
            * ``"cd"`` — critical difference diagram via
              :func:`~evalstats.vis.critical_difference.plot_critical_difference`.
            * ``"pareto"`` — uncertainty-aware trade-off scatter via
              :func:`~evalstats.vis.pareto.plot_pareto_tradeoff`. Only
              available when ``compare(..., secondary_metric=...)`` was passed;
              raises otherwise. One bootstrap point cloud per entity plus a
              percentile band over per-replicate Pareto frontiers, colored
              by calibrated status (frontier / dominated / ambiguous) --
              see :attr:`pareto_status`.

        **kwargs
            Forwarded to the underlying plot function.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if method == "bar":
            from evalstats.vis.scoreboard import plot_accuracy_bar
            return plot_accuracy_bar(self, **kwargs)
        elif method == "forest":
            from evalstats.vis.forest import plot_ci_forest
            return plot_ci_forest(self, **kwargs)
        elif method == "cd":
            from evalstats.vis.critical_difference import plot_critical_difference
            return plot_critical_difference(self, **kwargs)
        elif method == "pareto":
            if self._pareto is None:
                raise ValueError(
                    "method='pareto' requires compare(..., secondary_metric=...) "
                    "to have been passed -- no Pareto analysis was run for "
                    "this result."
                )
            from evalstats.vis.pareto import plot_pareto_tradeoff
            return plot_pareto_tradeoff(self._pareto, metric=self._metric, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot method: {method!r}. "
                "Expected 'bar', 'forest', 'cd', or 'pareto'."
            )

    def report(self, format: str = "markdown") -> str:
        """[Deferred] Export formatted report.

        Not yet implemented. Use :meth:`summary` for terminal output or
        :meth:`to_dict` / :meth:`to_frame` for programmatic access.
        """
        raise NotImplementedError(
            "report() export is not yet implemented. "
            "Use summary() for terminal output or to_dict()/to_frame() for data."
        )

    # ── duck-type compatibility properties ───────────────────────────────────
    # These allow ComparisonResult to be passed directly to vis functions
    # (plot_ci_forest, plot_accuracy_bar, plot_critical_difference) that
    # previously accepted CompareReport objects.

    @property
    def labels(self) -> list:
        """Entity labels (for vis function compatibility)."""
        bundle = self._primary_bundle()
        return list(bundle.benchmark.template_labels) if bundle else []

    @property
    def entity_stats(self) -> dict:
        """Per-entity stats dict (for vis function compatibility)."""
        from types import SimpleNamespace
        bundle = self._primary_bundle()
        if bundle is None:
            return {}
        rob = bundle.robustness
        return {
            str(lbl): SimpleNamespace(
                mean=float(rob.mean[i]),
                ci_low=float(rob.ci_low[i]) if rob.ci_low is not None else 0.0,
                ci_high=float(rob.ci_high[i]) if rob.ci_high is not None else 1.0,
                median=float(rob.median[i]),
                std=float(rob.std[i]),
                multi_ci=(
                    {a: (float(lo[i]), float(hi[i])) for a, (lo, hi) in rob.multi_ci.items()}
                    if rob.multi_ci is not None else None
                ),
            )
            for i, lbl in enumerate(bundle.benchmark.template_labels)
        }

    @property
    def unbeaten(self) -> Optional[list]:
        """Entities not significantly beaten at the stored alpha level.

        Returns ``None`` when no pairwise differences are significant (the
        concept of "unbeaten" does not apply when there is no winner).
        """
        bundle = self._primary_bundle()
        if bundle is None:
            return None
        labels = list(bundle.benchmark.template_labels)
        beaten: set = set()
        any_sig = False
        for (a, b), pair in bundle.pairwise.results.items():
            if pair.p_value is not None and pair.p_value < self._alpha:
                any_sig = True
                if pair.point_diff > 0:
                    beaten.add(str(b))
                else:
                    beaten.add(str(a))
        if not any_sig:
            return None
        result_unbeaten = [str(lbl) for lbl in labels if str(lbl) not in beaten]
        return result_unbeaten if result_unbeaten else None

    @property
    def entity_name_singular(self) -> str:
        """Entity type name (for vis function compatibility)."""
        f = self._factors
        if isinstance(f, str) and f in ("model", "prompt", "entity"):
            return f
        return "entity"

    @property
    def pairwise(self):
        """Pairwise matrix from the primary bundle."""
        bundle = self._primary_bundle()
        return bundle.pairwise if bundle else None

    @property
    def rank_dist(self):
        """Rank distribution from the primary bundle."""
        bundle = self._primary_bundle()
        return bundle.rank_dist if bundle else None

    @property
    def alpha(self) -> float:
        """Significance level used for this comparison (e.g. 0.05 → 95% CIs)."""
        return self._alpha

    @property
    def p_value_method(self) -> Optional[str]:
        """Stored p-value method from the underlying bundle (or None if not requested)."""
        bundle = self._primary_bundle()
        return bundle.p_value_method if bundle is not None else None

    @property
    def simultaneous_ci(self) -> bool:
        """Whether simultaneous (family-wise) CIs were used for pairwise comparisons."""
        bundle = self._primary_bundle()
        if bundle is None:
            return False
        return bool(bundle.pairwise.simultaneous_ci)

    @property
    def full_analysis(self):
        """Underlying AnalysisBundle (for vis function compatibility)."""
        return self._primary_bundle()

    @property
    def cross_model(self):
        """Flat ranking over every (model, template) pair, or ``None``.

        Populated only for two-factor comparisons (e.g.
        ``factors=["model", "prompt"]``, or ``factors="model"`` when a
        prompt column is also present). ``None`` for single-factor
        comparisons. Call ``.summary()`` on the returned
        :class:`~evalstats.core.bundles.AnalysisBundle` for the full
        cross-model comparison — every (model, template) cell with its own
        CI, ranked and grouped into statistically-indistinguishable bands,
        the same "unbeaten" logic :attr:`unbeaten` applies within a single
        factor, extended across both.
        """
        if isinstance(self._analysis, MultiModelBundle):
            return self._analysis.cross_model
        return None

    @property
    def best_pairs(self) -> Optional[list]:
        """(model, template) pairs statistically tied for best, or ``None``.

        This is the top significance group (``"#1"``) of :attr:`cross_model`
        — every cell whose CI is not distinguishable from the top
        performer's — not a single "best pair by mean". A higher point
        estimate alone does not make one cell the winner over another it
        isn't significantly different from; use this instead of manually
        picking the row with the highest mean out of :attr:`cross_model`.
        Mirrors :attr:`unbeaten` for the two-factor case: ``None`` when this
        isn't a two-factor comparison, or when there are fewer than two
        pairs to compare.
        """
        if not isinstance(self._analysis, MultiModelBundle):
            return None
        cross = self._analysis.cross_model
        labels = list(cross.labels)
        if len(labels) < 2:
            return None
        means = cross.robustness.mean
        sort_idx = list(np.argsort(-means))
        labels_sorted = [labels[i] for i in sort_idx]
        label_to_group = _assign_significance_groups(cross.pairwise, labels_sorted, alpha=self._alpha)
        top_pairs: list = []
        for label in labels_sorted:
            if label_to_group.get(label) == "#1":
                parts = label.split(" / ", 1)
                if len(parts) == 2:
                    top_pairs.append((parts[0], parts[1]))
        return top_pairs or None

    @property
    def model_labels(self) -> Optional[list]:
        """Model-axis labels for a two-factor (model, prompt) comparison, or ``None``.

        Populated under the same condition as :attr:`cross_model` — this is
        that bundle's model axis, in its original (pre-sort) order.
        """
        if not isinstance(self._analysis, MultiModelBundle):
            return None
        return list(self._analysis.benchmark.model_labels)

    @property
    def prompt_labels(self) -> Optional[list]:
        """Prompt/template-axis labels for a two-factor comparison, or ``None``.

        Populated under the same condition as :attr:`cross_model` — this is
        that bundle's template axis, in its original (pre-sort) order.
        """
        if not isinstance(self._analysis, MultiModelBundle):
            return None
        return list(self._analysis.benchmark.template_labels)

    def as_view(self, factor: Literal["model", "prompt"]) -> "ComparisonResult":
        """Return this two-factor comparison collapsed onto a single axis.

        E.g. ``result.as_view("model")`` averages over prompts to compare
        models; ``result.as_view("prompt")`` averages over models to compare
        prompts. Only valid for a two-factor comparison (see
        :attr:`cross_model`) — raises otherwise.
        """
        if not isinstance(self._analysis, MultiModelBundle):
            raise ValueError(
                "as_view() requires a two-factor comparison (built with "
                "compare(..., factors=['model', 'prompt']), or "
                "factors='model'/'prompt' when both columns are present)."
            )
        view_map = {"model": "model_level", "prompt": "template_level"}
        if factor not in view_map:
            raise ValueError(f"factor={factor!r} must be 'model' or 'prompt'.")
        return ComparisonResult(
            self._analysis,
            factors=self._factors,
            metric=self._metric,
            baseline=self._baseline,
            alpha=self._alpha,
            filtered_df=self._df,
            _mmb_view=view_map[factor],
            min_meaningful_diff=self._min_meaningful_diff,
            show_rank_probabilities=self._show_rank_probabilities,
        )

    @property
    def pareto_status(self) -> Optional[dict]:
        """Per-entity three-state Pareto classification, or ``None``.

        Populated only when ``compare(..., secondary_metric=...)`` was passed.
        Keys are entity labels, values are
        :class:`~evalstats.core.pareto.ParetoStatus` (``.status`` is one of
        ``"frontier"``, ``"dominated"``, ``"ambiguous"`` -- see that class's
        docstring for what distinguishes them). This is the calibrated
        default view: an entity is only ever reported ``"dominated"`` when
        the joint bootstrap actually supports it (FWER-aware across its
        possible dominators), not merely because its point estimate lost on
        both axes -- that weaker case is ``"ambiguous"`` instead of a false
        "dominates" call.
        """
        return self._pareto["statuses"] if self._pareto is not None else None

    @property
    def pareto_frontier_probability(self) -> Optional[dict]:
        """Per-entity ``P(entity is Pareto-optimal)``, or ``None``.

        Populated only when ``compare(..., secondary_metric=...)`` was passed. Keys
        are entity labels, values are the fraction of joint bootstrap
        replicates in which that entity was non-dominated -- a continuous
        probability, not the calibrated three-state label
        :attr:`pareto_status` gives. Exists for the same reason
        ``show_rank_probabilities``/P(Best) is opt-in rather than the
        default view elsewhere in evalstats: a raw probability like "82%
        Pareto-optimal" reads as more decisive than it is when the
        underlying joint CIs actually overlap heavily. Prefer
        :attr:`pareto_status` for reporting; use this for downstream
        numeric work.
        """
        if self._pareto is None:
            return None
        result = self._pareto["result"]
        return dict(zip(result.labels, result.p_frontier.tolist()))

    # ── data access ──────────────────────────────────────────────────────────

    def to_dict(self, *, show_rank_probabilities: Optional[bool] = None) -> dict:
        """Return a JSON-friendly dict with CIs, p-values, and pairwise diffs.

        Returns a dict with structure::

            {
                "factors": ...,
                "metric": ...,
                "alpha": ...,
                "entities": {
                    name: {
                        "mean": float,
                        "ci_low": float,
                        "ci_high": float,
                        "p_best": float,  # P(rank 1) from bootstrap -- only
                                          # present when show_rank_probabilities
                                          # resolves to True; see compare()
                    }
                },
                "pairwise": [
                    {"a": str, "b": str, "diff": float, "ci_low": float,
                     "ci_high": float, "p_value": float | None}
                ],
            }

        Parameters
        ----------
        show_rank_probabilities : bool, optional
            Include each entity's ``p_best``. Off by default (see
            ``compare(..., show_rank_probabilities=)``) -- overrides that
            default for this call only when passed explicitly.
        """
        bundle = self._primary_bundle()
        if bundle is None:
            return {
                "factors": self._factors,
                "metric": self._metric,
                "alpha": self._alpha,
                "note": "Multi-bundle result; use to_frame() for structured access.",
            }
        show_rank = self._show_rank_probabilities if show_rank_probabilities is None else show_rank_probabilities

        rob = bundle.robustness
        rank = bundle.rank_dist
        pairwise = bundle.pairwise

        entities: dict[str, dict] = {}
        labels = bundle.benchmark.template_labels
        for i, name in enumerate(labels):
            entry: dict[str, Any] = {
                "mean": float(rob.mean[i]),
                "ci_low": float(rob.ci_low[i]) if rob.ci_low is not None else None,
                "ci_high": float(rob.ci_high[i]) if rob.ci_high is not None else None,
            }
            if rank is not None and show_rank:
                entry["p_best"] = float(rank.p_best[i])
            entities[str(name)] = entry

        pw_list: list[dict] = []
        for (a, b), pair_result in pairwise.results.items():
            pw_entry: dict[str, Any] = {
                "a": str(a),
                "b": str(b),
                "diff": float(pair_result.point_diff),
                "ci_low": float(pair_result.ci_low),
                "ci_high": float(pair_result.ci_high),
            }
            if pair_result.p_value is not None:
                pw_entry["p_value"] = float(pair_result.p_value)
            pw_list.append(pw_entry)

        result: dict[str, Any] = {
            "factors": self._factors,
            "metric": self._metric,
            "alpha": self._alpha,
            "entities": entities,
            "pairwise": pw_list,
        }
        if self._variance_components is not None:
            result["variance_components"] = self._variance_components
        if self._pareto is not None:
            pareto_entities: dict[str, dict] = {}
            for label, st in self._pareto["statuses"].items():
                entry: dict[str, Any] = {"status": st.status}
                if st.dominated_by:
                    entry["dominated_by"] = list(st.dominated_by)
                if st.ambiguous_vs:
                    entry["ambiguous_vs"] = list(st.ambiguous_vs)
                if show_rank:
                    entry["p_pareto_optimal"] = float(self.pareto_frontier_probability[label])
                pareto_entities[str(label)] = entry
            result["pareto"] = {
                "secondary_metric": self._pareto["secondary_metric"],
                "direction": self._pareto["direction"],
                "entities": pareto_entities,
            }
        return result

    def to_frame(self, *, show_rank_probabilities: Optional[bool] = None) -> dict[str, pd.DataFrame]:
        """Return analysis results as a dict of DataFrames.

        Keys:

        * ``"entities"`` — one row per entity with mean, CI bounds, and (when
          ``show_rank_probabilities`` resolves to True) P(best).
        * ``"pairwise"`` — one row per pairwise comparison.
        * ``"raw"`` — the filtered input data that was analyzed.

        Parameters
        ----------
        show_rank_probabilities : bool, optional
            Include each entity's ``p_best``. Off by default (see
            ``compare(..., show_rank_probabilities=)``) -- overrides that
            default for this call only when passed explicitly.
        """
        bundle = self._primary_bundle()
        frames: dict[str, pd.DataFrame] = {"raw": self._df.copy()}

        if bundle is None:
            return frames
        show_rank = self._show_rank_probabilities if show_rank_probabilities is None else show_rank_probabilities

        rob = bundle.robustness
        rank = bundle.rank_dist
        pairwise = bundle.pairwise
        labels = bundle.benchmark.template_labels

        entity_rows: list[dict] = []
        for i, name in enumerate(labels):
            row: dict[str, Any] = {
                "entity": str(name),
                "mean": float(rob.mean[i]),
                "ci_low": float(rob.ci_low[i]) if rob.ci_low is not None else None,
                "ci_high": float(rob.ci_high[i]) if rob.ci_high is not None else None,
            }
            if rank is not None and show_rank:
                row["p_best"] = float(rank.p_best[i])
            entity_rows.append(row)
        frames["entities"] = pd.DataFrame(entity_rows)

        pw_rows: list[dict] = []
        for (a, b), pair_result in pairwise.results.items():
            row_pw: dict[str, Any] = {
                "a": str(a),
                "b": str(b),
                "diff": float(pair_result.point_diff),
                "ci_low": float(pair_result.ci_low),
                "ci_high": float(pair_result.ci_high),
            }
            if pair_result.p_value is not None:
                row_pw["p_value"] = float(pair_result.p_value)
            pw_rows.append(row_pw)
        frames["pairwise"] = pd.DataFrame(pw_rows)

        if self._pareto is not None:
            pareto_rows: list[dict] = []
            for label, st in self._pareto["statuses"].items():
                row_p: dict[str, Any] = {
                    "entity": str(label),
                    "status": st.status,
                    "dominated_by": ", ".join(st.dominated_by) or None,
                    "ambiguous_vs": ", ".join(st.ambiguous_vs) or None,
                }
                if show_rank:
                    row_p["p_pareto_optimal"] = float(self.pareto_frontier_probability[label])
                pareto_rows.append(row_p)
            frames["pareto"] = pd.DataFrame(pareto_rows)

        return frames

    def disagreements(
        self,
        by: Optional[str] = None,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return items where models/prompts disagree most.

        Returns rows from the raw filtered data where the score variance
        across the compared entities is highest — useful for finding examples
        where models diverge strongly.

        Parameters
        ----------
        by : str, optional
            Column to aggregate over (default: the item column).
        threshold : float, optional
            Only include items with score std ≥ threshold.
        top_n : int, optional
            Return only the top-N most disagreed-upon items.
        """
        df = self._df.copy()

        # Determine the entity column (what we're comparing)
        factors = [self._factors] if isinstance(self._factors, str) else self._factors
        item_col = by

        # Identify item col from the EvalResults column mapping — not stored directly,
        # so we check common item column names in df.columns.
        if item_col is None:
            for candidate in ["item", "input", "example", "id", "input_label"]:
                if candidate in df.columns:
                    item_col = candidate
                    break

        if item_col is None:
            raise ValueError(
                "Could not detect item column for disagreement analysis. "
                "Pass by='your_item_column'."
            )

        score_col = self._metric
        if score_col not in df.columns:
            return pd.DataFrame()

        key_cols = [item_col] + [
            f for f in factors if f in df.columns and f != item_col
        ]

        try:
            agg = (
                df.groupby(item_col)[score_col]
                .std()
                .reset_index()
                .rename(columns={score_col: "score_std"})
            )
        except (TypeError, KeyError):
            return pd.DataFrame()

        agg = agg.sort_values("score_std", ascending=False)

        if threshold is not None:
            agg = agg[agg["score_std"] >= threshold]

        if top_n is not None:
            agg = agg.head(top_n)

        # Join back the raw per-entity rows for the selected items, so callers
        # get the actual scores that produced each disagreement, not just the
        # std -- ordered by descending disagreement, item as the tiebreaker.
        out_cols = key_cols + [score_col]
        out = df[out_cols].merge(agg[[item_col, "score_std"]], on=item_col)
        out = out.sort_values(["score_std", item_col], ascending=[False, True])
        return out.reset_index(drop=True)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _primary_bundle(self) -> Optional[AnalysisBundle]:
        """Return a single AnalysisBundle from the underlying analysis, if available."""
        if isinstance(self._analysis, AnalysisBundle):
            return self._analysis
        if isinstance(self._analysis, MultiModelBundle):
            return getattr(self._analysis, self._mmb_view)
        # Per-evaluator dicts: return first
        if isinstance(self._analysis, dict):
            first = next(iter(self._analysis.values()), None)
            if isinstance(first, AnalysisBundle):
                return first
            if isinstance(first, MultiModelBundle):
                return getattr(first, self._mmb_view)
        return None

    def __repr__(self) -> str:
        bundle = self._primary_bundle()
        n_entities = "?" if bundle is None else len(bundle.benchmark.template_labels)
        return (
            f"ComparisonResult("
            f"factors={self._factors!r}, "
            f"metric={self._metric!r}, "
            f"n_entities={n_entities})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: bridge EvalResults → existing analysis engine
# ─────────────────────────────────────────────────────────────────────────────

_STANDARD_FACTOR_NAMES = {"model", "prompt", "template"}
_FACTOR_STD_SLOTS = ["model", "prompt"]  # canonical names for the first two custom factors


def _factor_item_labels(factors) -> tuple[str, str]:
    """Derive (item_singular, item_plural) from a factors argument."""
    if factors is None:
        return "template", "templates"
    if isinstance(factors, str):
        singular = factors
    else:
        singular = "|".join(factors)
    if "|" in singular:
        plural = singular + " combinations"
    elif singular.endswith("s"):
        plural = singular
    else:
        plural = singular + "s"
    return singular, plural


def _custom_factor_col_map(
    factors_list: list[str], df: pd.DataFrame
) -> dict[str, str]:
    """Return a col_map that remaps non-standard factor columns to standard slots.

    Non-standard factor columns (not "model", "prompt", or "template") that
    exist in *df* are mapped to "model" then "prompt" in order.  This allows
    load_from() to include them in the uniqueness key so the duplicate check
    passes.
    """
    col_map: dict[str, str] = {}
    slot_idx = 0
    for f in factors_list:
        if f not in _STANDARD_FACTOR_NAMES and f in df.columns and slot_idx < len(_FACTOR_STD_SLOTS):
            col_map[f] = _FACTOR_STD_SLOTS[slot_idx]
            slot_idx += 1
    return col_map


def _apply_kwarg_filters(
    df: pd.DataFrame,
    kwargs: dict[str, Any],
    known_params: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split kwargs into column filters vs. engine kwargs, then filter."""
    col_filters: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in known_params:
            engine_kwargs[k] = v
        elif k in df.columns:
            col_filters[k] = v
        else:
            warnings.warn(
                f"compare(): unknown keyword argument '{k}'. "
                "If this is a column filter, the column was not found in the data. "
                "If it's an analysis parameter, check the spelling.",
                UserWarning,
                stacklevel=3,
            )

    for col, val in col_filters.items():
        if isinstance(val, (list, tuple)):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]

    if df.empty:
        raise EvalLoadError(
            "After applying column filters from keyword arguments, no data remains."
        )

    return df, engine_kwargs


_ANALYZE_PARAMS = {
    "evaluator_mode", "reference", "method", "backend", "n_bootstrap",
    "correction", "spread_percentiles", "failure_threshold", "rng", "statistic",
    "template_model_collapse", "simultaneous_ci", "omnibus", "p_values",
    "pairwise_test", "ci_style", "score_range", "eval_type",
}


def _bridge_to_io(
    df: pd.DataFrame,
    *,
    factor_col: str,
    item_col: str,
    metric_col: str,
    run_col: Optional[str] = None,
    block_col: Optional[str] = None,
) -> pd.DataFrame:
    """Rename columns to the canonical names expected by from_dataframe()."""
    rename: dict[str, str] = {}
    if factor_col != "template":
        rename[factor_col] = "template"
    if item_col != "input":
        rename[item_col] = "input"
    if metric_col != "score":
        rename[metric_col] = "score"
    if run_col and run_col != "run":
        rename[run_col] = "run"

    # When block is a second key (e.g. model when comparing prompts), pass as "model"
    if block_col and block_col not in rename and block_col != "model":
        rename[block_col] = "model"

    df_io = df.copy()
    df_io = df_io.rename(columns=rename)

    keep_cols = {"template", "input", "score"}
    if run_col:
        keep_cols.add(rename.get(run_col, run_col))
    if block_col:
        keep_cols.add(rename.get(block_col, block_col))

    keep_cols = keep_cols & set(df_io.columns)
    return df_io[list(keep_cols)]


# ─────────────────────────────────────────────────────────────────────────────
# compare()
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    evaldata: EvalResults,
    *,
    factors: Union[str, list[str]],
    metric: Optional[str] = None,
    baseline: Optional[str] = None,
    block: Union[str, list[str], Literal["auto"]] = "auto",
    slices=None,         # deferred
    secondary_metric: Optional[dict[str, Literal["min", "max"]]] = None,
    alignment=None,
    n_mc: int = 200,
    min_meaningful_diff: Optional[float] = None,
    alpha: Optional[float] = None,
    p_values: Optional[bool] = None,
    omnibus: Optional[bool] = None,
    pairwise_test: Literal["auto", "bootstrap", "wilcoxon", "nemenyi"] = "auto",
    show_rank_probabilities: bool = False,
    design: Literal["auto", "paired", "unpaired"] = "auto",
    **kwargs: Any,
) -> Union[ComparisonResult, GroupComparisonResult]:
    """Compare entities along one or more factor axes.

    Parameters
    ----------
    evaldata : EvalResults
        Evaluation data from :func:`load_from`.
    factors : str or list[str]
        What to compare. Common values:

        * ``"model"`` — compare models
        * ``"prompt"`` — compare prompt templates
        * ``["model", "prompt"]`` — factorial design (uses LMM backend)
        * Any other column name — compares levels of that column

    metric : str, optional
        Metric column to analyze. Defaults to the first metric column
        detected by ``load_from``.
    baseline : str, optional
        Name of the baseline entity to compare all others against.
        When ``None``, uses grand-mean reference.
    block : str or "auto"
        Blocking column — typically ``"item"`` or ``"input"``. ``"auto"``
        (default) uses the item column detected by ``load_from``. Only a
        single blocking column is supported; passing a list uses its first
        element and warns.
    secondary_metric : dict[str, {"min", "max"}], optional
        Run an uncertainty-aware Pareto-front analysis against a second
        metric, e.g. ``secondary_metric={"latency_ms": "min"}`` to find the
        accuracy/latency frontier (``"min"`` for a cost-like metric where
        lower is better, ``"max"`` for a benefit-like one). Currently
        supports exactly one secondary metric and a single-factor result
        (not yet supported for multi-model/factorial comparisons or seeded
        R>=3 benchmarks). On the paired path (default), also requires a
        complete design (every entity scored on every item for the
        secondary metric too) and resamples both metrics *jointly* via a
        shared per-item bootstrap draw (not two independent marginal
        bootstraps) so correlation between them (e.g. harder items being
        both slower and less accurate) is preserved rather than dropped —
        a marginally-better point estimate on both axes isn't reported as
        a confident "dominates" call when the data can't actually support
        it. On the unpaired path (``design="unpaired"``), the same idea
        applies at row granularity instead — see ``design=``'s docstring
        for exactly how. See :attr:`ComparisonResult.pareto_status` /
        :attr:`ComparisonResult.pareto_frontier_probability` (also exposed
        identically on :class:`~evalstats.core.unpaired.GroupComparisonResult`
        for the unpaired path).
    alignment : dict[str, AlignmentResult], optional
        Apply PPI (prediction-powered inference) correction for LLM-judge
        measurement error, using a sparse subset of human labels. Pass
        ``{metric: judge_alignment(...)}`` where ``metric`` matches the
        metric column being compared. Requires a single-factor, single-model
        (non-factorial) comparison. See :func:`~evalstats.alignment.judge_alignment`.
    n_mc : int
        Bootstrap resample count used by the PPI-correction path when
        ``alignment=`` is set (floored at 1000 internally). Has no effect
        otherwise, and no effect on the unpaired path (use ``n_bootstrap=``
        there instead).
    min_meaningful_diff : float, optional
        A difference of practical interest, in the metric's own units.
        When set, the printed summary adds a rough "how many more inputs
        would you need" estimate based on the observed variance.
    alpha : float, optional
        Significance level / CI width: ``alpha=0.05`` → 95 % CIs.
        When ``None`` (default), uses the global value set by
        :func:`~evalstats.config.set_alpha_ci` (default 0.05).
    p_values : bool
        When ``True``, print a p-value column in the pairwise comparisons
        table (default: bootstrap p-values). Combine with ``omnibus=True``
        to switch this to Wilcoxon signed-rank (the standard Friedman
        post-hoc), or set ``pairwise_test=`` explicitly to pick one
        directly. When ``alignment=`` is also passed, bootstrap and
        Wilcoxon p-values are both PPI-corrected.
    omnibus : bool
        When ``True``, run and print the Friedman omnibus test ("are ANY
        of the compared entities different?") above the pairwise table.
        Also PPI-corrected when ``alignment=`` is passed.
    pairwise_test : {"auto", "bootstrap", "wilcoxon", "nemenyi"}
        Which p-value to show in the pairwise table. ``"auto"`` (default)
        always picks Wilcoxon signed-ranks, for any number of entities --
        the standard workflow fig:fwer-decision-tree assumes throughout:
        Friedman omnibus first when requested (``omnibus=True``), then
        Wilcoxon for every pairwise comparison, then FWER-corrected as
        post-hoc (see ``correction=`` on the underlying analysis engine).
        Pass ``pairwise_test="bootstrap"`` explicitly for the CI-construction
        method's own p-value instead. ``"nemenyi"`` requires ``omnibus=True``
        and is not supported together with
        ``alignment=`` (no validated PPI-corrected Nemenyi exists yet).
    show_rank_probabilities : bool
        When ``True``, include the bootstrap "Rank Probabilities" block
        (P(Best)/E[Rank] per entity) in ``.summary()`` output and the
        ``p_best`` field in ``.to_dict()``/``.to_frame()``. Off by default:
        a P(Best) figure reads as a confident, near-authoritative verdict
        even when entities are statistically indistinguishable once you
        look at the CIs sitting next to it, so this output is opt-in rather
        than opt-out. Ranking is still computed either way; this only
        controls whether it's surfaced. Can be overridden per-call via the
        same-named argument on ``.summary()``/``.to_dict()``/``.to_frame()``.
    design : {"auto", "paired", "unpaired"}
        Experimental design for single-factor comparisons (``factors`` names
        one column, and no factorial/multi-model second axis applies).
        ``"auto"`` (default) checks whether items are shared across the
        compared groups: when they are (the normal within-subjects case —
        every entity scored on the same items), analysis proceeds exactly
        as before. When items are disjoint per group (a between-subjects
        design — e.g. independent user cohorts, one per condition), a
        ``ValueError`` is raised rather than silently forcing a paired
        analysis onto unpaired data, since ``compare()``'s default engine
        assumes paired items. Pass ``design="unpaired"`` to explicitly run
        the between-subjects path instead: a per-group descriptive summary
        plus all-pairs comparisons (Kruskal-Wallis omnibus / Mann-Whitney U
        post-hoc for continuous, likert, and grade metrics; one-way ANOVA /
        Welch's t-test for binary metrics), Bonferroni-corrected CIs and
        Holm-corrected p-values, PPI-corrected when ``alignment=`` is
        passed. Between-subjects data commonly has no natural item/reviewer
        id at all (e.g. just group + rating) — ``load_from()`` still
        requires *some* item column to build ``evaldata`` in the first
        place, so add a throwaway one first if needed, e.g.
        ``df["item"] = range(len(df))``, before calling ``load_from()``.
        Returns a :class:`~evalstats.core.unpaired.GroupComparisonResult`
        instead of :class:`ComparisonResult` — see its ``.summary()``,
        ``.to_dict()``, ``.to_frame()``, ``.groups_to_frame()``. Pass
        ``design="paired"`` to force the existing paired analysis even on
        data that looks between-subjects (matches pre-``design=`` behavior).
        Not supported for factorial (2+ factor) comparisons; for
        ``method="lmm"``/``"factorial_lmm"``, which already tolerate
        unbalanced/disjoint designs natively via random effects; for any
        other explicit ``method=``/``backend=`` override (the between-
        subjects engine's CI construction isn't a pluggable-method
        surface); or together with multi-run (seeded) data.
        ``secondary_metric=`` (Pareto-front analysis) IS supported here —
        unlike the paired path's shared-item-index joint bootstrap (every
        entity resampled at the same item positions), the between-subjects
        version resamples each group's own rows independently (there's no
        shared item pool across disjoint groups to preserve correlation
        through), still preserving each row's own primary/secondary
        pairing. Populates
        :attr:`~evalstats.core.unpaired.GroupComparisonResult.pareto_status`/
        ``pareto_frontier_probability`` exactly like the paired path's own
        attributes. ``score_range=`` is honored (passed through to the
        per-group marginal CI's auto-method resolution, same as the paired
        path). ``n_mc=`` has no effect — the equivalent knob is
        ``n_bootstrap=``. ``p_values=`` and ``omnibus=`` are honored, but
        with unpaired-specific *defaults of True* (not ``compare()``'s own
        ``False``) — leave them unset to get this path's normal, always-
        shown report; pass ``p_values=False`` to hide the pairwise table's
        p-value column (the underlying values stay in ``.to_dict()``/
        ``.to_frame()``), or ``omnibus=False`` to skip running the omnibus
        test entirely at 3+ groups. ``baseline=``, ``pairwise_test=``, and
        ``show_rank_probabilities=`` still have no effect on this path —
        it always reports all-pairs comparisons (no baseline-relative
        view) and has no rank-probability view.
    **kwargs
        Two uses:

        1. **Column filters** — keyword matching a column name in the data
           filters rows before analysis.
           E.g. ``compare(evaldata, factors="model", split="test")``
           keeps only rows where ``split == "test"``.
           Pass a list to select multiple values:
           ``model=["gpt-4o", "claude-3-5-sonnet"]``.

        2. **Analysis engine overrides** — any other keyword argument
           accepted by :func:`~evalstats.core.router.analyze` (e.g.
           ``method="bca"``, ``n_bootstrap=5000``, ``score_range=(1, 5)``
           for a Likert-scale metric — see ``analyze()``'s ``score_range``
           parameter for when this matters).

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> import evalstats as es
    >>> evaldata = es.load_from(df, col_map={"llm": "model", "q_id": "item"})
    >>> result = es.compare(evaldata, factors="model")
    >>> result.summary()

    >>> result = es.compare(evaldata, factors="prompt", method="bca")
    >>> result.to_frame()["entities"]
    """
    if slices is not None:
        warnings.warn("slices= is not yet implemented and will be ignored.", UserWarning, stacklevel=2)
    # ── resolve alpha (explicit > global default) ─────────────────────────────
    if alpha is None:
        alpha = get_alpha_ci()

    # Preserve the original factor names as provided by the caller; internal
    # dispatch may remap them to standard column names ("model", "prompt") so
    # that load_from() can detect uniqueness constraints correctly.
    _user_factors = factors

    # ── coerce non-EvalResults input types ────────────────────────────────────
    if isinstance(evaldata, list):
        # list[dict] in long format — detect custom factor columns and remap
        # them to standard names so load_from() includes them in the uniqueness
        # key (otherwise the duplicate check rejects multi-factor long tables).
        _f_list = [factors] if isinstance(factors, str) else list(factors)
        _tmp_df = pd.DataFrame(evaldata) if evaldata else pd.DataFrame()
        _cmap = _custom_factor_col_map(_f_list, _tmp_df)
        if _cmap:
            evaldata = load_from(evaldata, col_map=_cmap)
            _f_list = [_cmap.get(f, f) for f in _f_list]
            factors = _f_list[0] if len(_f_list) == 1 else _f_list
        else:
            evaldata = load_from(evaldata)
    elif isinstance(evaldata, dict):
        # dict-of-arrays (flat or nested) — convert via scores-dict helper,
        # then ensure factor column names resolve to standard slot names.
        # _scores_dict_to_df normalizes nested-dict keys to "model"/"prompt"
        # regardless of factors; flat-dict custom names stay as-is and need
        # renaming so load_from() includes them in the uniqueness key.
        _f_list = [factors] if isinstance(factors, str) else list(factors)
        df_from_dict = _scores_dict_to_df(evaldata, factors=factors)
        _f_actual: list[str] = []
        for _i, _f in enumerate(_f_list):
            _std = _FACTOR_STD_SLOTS[_i] if _i < len(_FACTOR_STD_SLOTS) else _f
            if _f in df_from_dict.columns and _f not in _STANDARD_FACTOR_NAMES:
                # Flat-dict custom column — rename to standard slot in place
                df_from_dict = df_from_dict.rename(columns={_f: _std})
                _f_actual.append(_std)
            elif _std in df_from_dict.columns:
                # Nested-dict already produced the standard slot name
                _f_actual.append(_std)
            else:
                _f_actual.append(_f)
        factors = _f_actual[0] if len(_f_actual) == 1 else _f_actual
        evaldata = load_from(df_from_dict)
    elif not isinstance(evaldata, EvalResults):
        raise TypeError(
            f"compare() expects EvalResults, list[dict], or dict-of-arrays; "
            f"got {type(evaldata).__name__}. "
            "Use load_from() to construct an EvalResults object from a DataFrame."
        )

    # ── get raw DataFrame and column roles ────────────────────────────────────
    df = evaldata._df.copy()
    col = evaldata._col
    metric_cols = evaldata._metric_cols

    # Resolve metric column
    if metric is None:
        metric_col = metric_cols[0]
    else:
        if metric not in df.columns:
            raise EvalLoadError(
                f"metric column '{metric}' not found in data. "
                f"Available metric columns: {metric_cols}"
            )
        metric_col = metric

    # Resolve item (blocking) column
    if block == "auto":
        item_col = col.get("item")
    elif isinstance(block, str):
        item_col = block
    else:
        if len(block) > 1:
            warnings.warn(
                f"block={block!r} has more than one column; only a single "
                "blocking column is supported, so only the first "
                f"({block[0]!r}) is used.",
                UserWarning,
                stacklevel=2,
            )
        item_col = block[0] if block else col.get("item")

    if not item_col or item_col not in df.columns:
        raise EvalLoadError(
            "Could not determine the item/blocking column. "
            "Specify block='your_item_column' or ensure your data has an 'item' column."
        )

    run_col = col.get("run")

    # ── fold the named p-value engine params back into kwargs so the existing
    # column-filter/engine-kwarg split below (and the analyze() calls it
    # feeds) doesn't need to change ──────────────────────────────────────────
    kwargs = dict(kwargs)
    kwargs["p_values"] = p_values
    kwargs["omnibus"] = omnibus
    kwargs["pairwise_test"] = pairwise_test

    # ── split kwargs into column filters vs. engine kwargs ────────────────────
    df, engine_kwargs = _apply_kwarg_filters(df, kwargs, _ANALYZE_PARAMS)

    # ── set CI level from alpha ───────────────────────────────────────────────
    ci_level = 1.0 - alpha

    # ── dispatch by factor type ───────────────────────────────────────────────
    factors_list = [factors] if isinstance(factors, str) else list(factors)

    # Detect canonical mappings
    model_col  = col.get("model")
    prompt_col = col.get("prompt")

    is_model_comparison  = (len(factors_list) == 1 and
                             factors_list[0] in {"model"} and model_col and model_col in df)
    is_prompt_comparison = (len(factors_list) == 1 and
                             factors_list[0] in {"prompt", "template"} and prompt_col and prompt_col in df)
    is_canonical_col = (len(factors_list) == 1 and factors_list[0] in df.columns and
                        not is_model_comparison and not is_prompt_comparison)
    is_factorial = len(factors_list) >= 2

    # Reject NaN/missing values in factor column(s) early with a clear,
    # correctly-attributed error -- otherwise a NaN factor value silently
    # becomes its own group and only surfaces later as a confusing "scores
    # contain N NaN cells" error that blames the metric column instead.
    for _f in factors_list:
        _resolved_factor_col = (
            model_col if (_f == "model" and model_col and model_col in df.columns) else
            prompt_col if (_f in {"prompt", "template"} and prompt_col and prompt_col in df.columns) else
            _f if _f in df.columns else None
        )
        if _resolved_factor_col is not None:
            _n_na_factor = int(df[_resolved_factor_col].isna().sum())
            if _n_na_factor > 0:
                raise ValueError(
                    f"factor column {_resolved_factor_col!r} contains {_n_na_factor} "
                    "missing (NaN) value(s). Every row must have a value for the "
                    "factor being compared -- drop or fill these rows before "
                    "calling compare()."
                )

    # Also handle the case where factor is neither "model" nor "prompt" but names
    # a canonical-alias column directly (e.g. user mapped "llm" → "model", then
    # passes factors="model" which now IS model_col).
    if not is_model_comparison and not is_prompt_comparison and not is_factorial:
        factor_col_name = factors_list[0]
        if factor_col_name in df.columns:
            is_canonical_col = True

    # ── enforce the documented minimum sample floor ───────────────────────────
    # Below MIN_SAMPLE_FLOOR items per compared entity, results are too noisy
    # to be meaningful -- refuse rather than silently print stats built on too
    # little data. Uses the per-entity item count when a single clear factor
    # column is resolved (the common case); falls back to the overall unique
    # item count for factorial comparisons, where "N" isn't a single number.
    _floor_factor_col = (
        model_col if is_model_comparison else
        prompt_col if is_prompt_comparison else
        factors_list[0] if (is_canonical_col and factors_list[0] in df.columns) else
        None
    )
    if _floor_factor_col is not None:
        _min_n = int(df.groupby(_floor_factor_col)[item_col].nunique().min())
    else:
        _min_n = int(df[item_col].nunique())
    if _min_n < MIN_SAMPLE_FLOOR:
        raise ValueError(
            f"Only {_min_n} item(s) per compared entity -- evalstats requires at "
            f"least {MIN_SAMPLE_FLOOR} to report statistics (results below this "
            "floor are too noisy to be meaningful). Expand your eval set before "
            "calling compare()."
        )

    # ── design detection / routing (paired vs. unpaired) ─────────────────────
    # Scoped to "pure" single-factor cases only -- i.e. whichever of paths A/B/C
    # would run below, and only when that path's own implicit multi-model second
    # axis (block_col) is absent, since the multi-model/factorial machinery is
    # out of scope here. Factorial calls and method="lmm"/"factorial_lmm" are
    # exempt entirely: LMM already tolerates incomplete/disjoint designs via
    # random effects, and no currently-passing non-LMM call can be affected by
    # this new check, because the bootstrap path already hard-crashes on
    # genuinely unpaired data (has_missing) -- so paired-path behavior for every
    # existing call is unchanged.
    _design_backend = engine_kwargs.get("method") or engine_kwargs.get("backend")
    _design_exempt = is_factorial or _design_backend in {"lmm", "factorial_lmm"}

    if _design_exempt:
        if design == "unpaired":
            raise ValueError(
                'design="unpaired" is not supported for factorial (2+ factor) '
                'comparisons or for method="lmm"/"factorial_lmm", which already '
                "handle unbalanced/disjoint designs natively via random effects."
            )
    else:
        if is_model_comparison:
            _design_factor_col = model_col
            _design_is_pure_single_factor = not (prompt_col and prompt_col in df.columns)
        elif is_prompt_comparison:
            _design_factor_col = prompt_col
            _design_is_pure_single_factor = not (model_col and model_col in df.columns)
        elif is_canonical_col or (not is_factorial and factors_list[0] in df.columns):
            _design_factor_col = factors_list[0]
            _design_is_pure_single_factor = True
        else:
            _design_factor_col = None
            _design_is_pure_single_factor = False

        if _design_is_pure_single_factor and _design_factor_col:
            if design == "unpaired" and run_col and run_col in df.columns and df[run_col].nunique() > 1:
                raise ValueError(
                    f'design="unpaired" does not yet support multi-run (seeded) data '
                    f"-- column {run_col!r} has more than one run per item. Treating "
                    "each run as its own row would silently inflate the effective "
                    "sample size and break the independence assumption the between-"
                    "subjects tests rely on (same scoping precedent as PPI alignment's "
                    "own seeded-benchmark refusal). Aggregate runs to a single score "
                    f"per item first, e.g. df.groupby([{_design_factor_col!r}, "
                    f"{item_col!r}])[{metric_col!r}].mean().reset_index()."
                )
            if design == "unpaired" and _design_backend not in (None, "auto"):
                raise ValueError(
                    f'method={_design_backend!r} is not supported together with '
                    'design="unpaired" -- the between-subjects engine\'s CI '
                    "construction (Bonferroni/Holm pairwise, Kruskal-Wallis/ANOVA "
                    "omnibus) isn't a pluggable-method surface the way the paired "
                    'path is. Drop method= for this comparison. score_range= is '
                    "still honored."
                )
            if design == "unpaired":
                # Unlike the paired path, this narrower report defaults both
                # to True (an unpaired-specific default, not compare()'s own
                # False) -- unset (None, meaning the caller didn't pass
                # either) preserves the always-shown behavior this path was
                # built and battle-tested with; an explicit True/False is
                # honored as a real suppress/show toggle.
                _up_p_values = True if engine_kwargs.get("p_values") is None else bool(engine_kwargs.get("p_values"))
                _up_omnibus = True if engine_kwargs.get("omnibus") is None else bool(engine_kwargs.get("omnibus"))
                return compare_unpaired(
                    df, factor_col=_design_factor_col, metric_col=metric_col,
                    item_col=item_col, alignment=alignment, alpha=alpha,
                    n_boot=engine_kwargs.get("n_bootstrap", 2000),
                    rng=engine_kwargs.get("rng"),
                    score_range=engine_kwargs.get("score_range"),
                    p_values=_up_p_values, omnibus=_up_omnibus,
                    secondary_metric=secondary_metric,
                )
            if design == "auto" and not detect_paired(df, _design_factor_col, item_col):
                raise ValueError(
                    f"Data for factor {_design_factor_col!r} looks between-subjects "
                    "(items are not shared across the compared groups), but "
                    "compare()'s default analysis assumes within-subjects (paired) "
                    'data. Pass design="unpaired" to run the between-subjects '
                    'comparison instead, or design="paired" to force the existing '
                    "paired analysis anyway."
                )
        elif design == "unpaired":
            raise ValueError(
                'design="unpaired" is not supported for this comparison (it '
                "implies a multi-model/multi-template second axis, which is out "
                "of scope for the between-subjects path)."
            )

    # ── path A: model comparison ──────────────────────────────────────────────
    if is_model_comparison:
        factor_col_name = model_col
        block_col = prompt_col  # prompts become the template axis if present

        if block_col and block_col in df.columns:
            # Multi-model path: map model→"model" axis and prompt→"template" axis.
            # This keeps labels natural: MultiModelBundle.model_level compares models,
            # template_level compares prompts, per_model shows per-model prompt analysis.
            df_multi = df[[factor_col_name, block_col, item_col, metric_col]
                          + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_multi = {
                factor_col_name: "model",     # actual models → "model" axis
                block_col: "template",        # prompts → "template" axis
                item_col: "input",
                metric_col: "score",
            }
            if run_col and run_col in df.columns:
                rename_multi[run_col] = "run"
            df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
            bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        else:
            # No prompt col — single-model BenchmarkResult with model as template axis
            df_io_keep = df[[factor_col_name, item_col, metric_col]
                            + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
            if run_col and run_col in df.columns:
                rename_io[run_col] = "run"
            df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
            bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        # model→"model" axis means model_level compares models (what the user requested).
        # When bench is a BenchmarkResult (no prompts), the analysis is an AnalysisBundle
        # and _mmb_view is irrelevant.
        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="model_level",
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        _run_pareto_if_needed(
            cr, secondary_metric=secondary_metric, df=df, factor_col=factor_col_name,
            item_col=item_col, alpha=alpha, n_boot=max(n_mc, 1000),
            rng=engine_kwargs.get("rng"),
        )
        return cr

    # ── path B: prompt/template comparison ───────────────────────────────────
    if is_prompt_comparison:
        factor_col_name = prompt_col
        block_col = model_col  # if models present, they become block axis

        if block_col and block_col in df.columns:
            # Multi-model path: prompt as template, model as model.
            df_multi = df[[block_col, factor_col_name, item_col, metric_col]
                          + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_multi = {
                factor_col_name: "template",
                block_col: "model",
                item_col: "input",
                metric_col: "score",
            }
            if run_col and run_col in df.columns:
                rename_multi[run_col] = "run"
            df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
            bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        else:
            df_io_keep = df[[factor_col_name, item_col, metric_col]
                            + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
            if run_col and run_col in df.columns:
                rename_io[run_col] = "run"
            df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
            bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="template_level",
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        _run_pareto_if_needed(
            cr, secondary_metric=secondary_metric, df=df, factor_col=factor_col_name,
            item_col=item_col, alpha=alpha, n_boot=max(n_mc, 1000),
            rng=engine_kwargs.get("rng"),
        )
        return cr

    # ── path C: arbitrary single factor column ────────────────────────────────
    if is_canonical_col or (not is_factorial and factors_list[0] in df.columns):
        factor_col_name = factors_list[0]

        df_io_keep = df[[factor_col_name, item_col, metric_col]
                        + ([run_col] if run_col and run_col in df.columns else [])].copy()
        rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
        if run_col and run_col in df.columns:
            rename_io[run_col] = "run"
        df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
        bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        _run_pareto_if_needed(
            cr, secondary_metric=secondary_metric, df=df, factor_col=factor_col_name,
            item_col=item_col, alpha=alpha, n_boot=max(n_mc, 1000),
            rng=engine_kwargs.get("rng"),
        )
        return cr

    # ── path D-pre: canonical 2-factor ["model","prompt"] → multi-model path ──
    # When the user passes exactly the standard factor names (not custom names
    # that were remapped), prefer the richer MultiModelBundle path over factorial
    # LMM — it's simpler, faster, and shows the cross-model ranking output the
    # user usually wants.  An explicit LMM backend opt-in bypasses this.
    _user_factors_list = [_user_factors] if isinstance(_user_factors, str) else list(_user_factors)
    _is_canonical_2factor = (
        is_factorial
        and len(_user_factors_list) == 2
        and all(f in _STANDARD_FACTOR_NAMES for f in _user_factors_list)
        and model_col and model_col in df.columns
        and prompt_col and prompt_col in df.columns
        and engine_kwargs.get("backend") not in {"lmm", "factorial_lmm"}
    )
    if _is_canonical_2factor:
        df_multi = df[
            [model_col, prompt_col, item_col, metric_col]
            + ([run_col] if run_col and run_col in df.columns else [])
        ].copy()
        rename_multi = {
            model_col: "model",
            prompt_col: "template",
            item_col: "input",
            metric_col: "score",
        }
        if run_col and run_col in df.columns:
            rename_multi[run_col] = "run"
        df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
        bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)
        return ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="model_level",
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )

    # ── path D: factorial (multiple factors → LMM) ────────────────────────────
    if is_factorial:
        # Validate all factor columns exist
        missing_factors = [f for f in factors_list if f not in df.columns]
        if missing_factors:
            raise EvalLoadError(
                f"Factor column(s) {missing_factors} not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Rename metric and item cols to what analyze_factorial expects
        df_fact = df.copy()
        rename_fact: dict[str, str] = {}
        if metric_col != "score":
            rename_fact[metric_col] = "score"
            df_fact = df_fact.rename(columns=rename_fact)
            score_col_name = "score"
        else:
            score_col_name = "score"

        factorial_kwargs = {
            k: v for k, v in engine_kwargs.items()
            if k in {"backend", "ci", "correction", "reference",
                     "spread_percentiles", "failure_threshold", "n_sim", "rng"}
        }
        if "ci" not in factorial_kwargs:
            factorial_kwargs["ci"] = ci_level

        run_col_fact = run_col if run_col and run_col in df_fact.columns else None

        analysis = analyze_factorial(
            df_fact,
            factors=factors_list,
            random_effect=item_col,
            score_col=score_col_name,
            run_col=run_col_fact,
            **factorial_kwargs,
        )

        return ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            min_meaningful_diff=min_meaningful_diff,
            show_rank_probabilities=show_rank_probabilities,
        )

    raise EvalLoadError(
        f"Could not dispatch compare() for factors={factors!r}. "
        f"Factor column(s) not found in data. Available columns: {list(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thin wrappers
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(evaldata: EvalResults, **kwargs) -> ComparisonResult:
    """Compare models — equivalent to ``compare(evaldata, factors="model", ...)``.

    All keyword arguments are forwarded to :func:`compare`.
    """
    return compare(evaldata, factors="model", **kwargs)


def compare_prompts(evaldata: EvalResults, **kwargs) -> ComparisonResult:
    """Compare prompt templates — equivalent to ``compare(evaldata, factors="prompt", ...)``.

    All keyword arguments are forwarded to :func:`compare`.
    """
    return compare(evaldata, factors="prompt", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# PPI alignment correction
# ─────────────────────────────────────────────────────────────────────────────

_PPI_PAIRWISE_SUPPORTED = ("bonett_price", "mj_floor", "t_interval", "bootstrap", "wilcoxon", "mannwhitney", "bootstrap_t", "bayes_bootstrap", "ppi_t_interval", "ppi_logit_t")
_PPI_ROBUSTNESS_SUPPORTED = ("wilson", "bootstrap", "bootstrap_t", "ppi_t_interval", "ppi_logit_t")


def _ppi_pairwise_dispatch(method: str, a, b, a_lab, b_lab, alpha: float, n_boot: int, rng,
                           score_range: Optional[tuple[float, float]] = None):
    """Dispatch to the PPI-corrected pairwise implementation of *method*.

    Only methods with a validated PPI-corrected counterpart (see
    ``evalstats.tests``'s ``_ppi_paired_*``/``_ppi_two_sample`` functions,
    calibrated via ``simulations/harness --mode ppi``) are supported here.

    "ppi_t_interval"/"ppi_logit_t" are DISTINCT method strings from the
    existing bare "t_interval" (below) -- that one already maps to
    ``_ppi_paired_arrays(..., np.mean, rectifier_func=np.mean)``, the
    generic PPI-mean-diff bootstrap routine, not the closed-form analytic
    construction these two use. Reusing "t_interval"/"logit_t" here would
    silently collide with that existing mapping.
    """
    from evalstats.tests import (
        _ppi_paired_mj_floor, _ppi_paired_bonett_price, _ppi_paired_bootstrap_t,
        _ppi_paired_bayes_bootstrap,
        _ppi_paired_arrays, _ppi_two_sample, _p_x_gt_y_midrank,
        _ppi_paired_t_interval, _ppi_paired_logit_t,
    )
    if method == "bonett_price":
        return _ppi_paired_bonett_price(a, b, a_lab, b_lab, alpha)
    if method == "mj_floor":
        return _ppi_paired_mj_floor(a, b, a_lab, b_lab, alpha)
    if method == "bootstrap_t":
        return _ppi_paired_bootstrap_t(a, b, a_lab, b_lab, alpha, n_boot, rng)
    if method == "bayes_bootstrap":
        return _ppi_paired_bayes_bootstrap(a, b, a_lab, b_lab, alpha, n_boot, rng)
    if method == "ppi_t_interval":
        return _ppi_paired_t_interval(a, b, a_lab, b_lab, alpha)
    if method == "ppi_logit_t":
        # lo/hi default (0.0, 1.0): this dispatch path has no score_range
        # concept (see _run_alignment_ppi's is_bounded_01_scores check --
        # "bounded_01" always means raw scores are literally in [0, 1] here).
        # lo/hi MUST come from the resolved score_range, not the (0, 1)
        # default: logit-t is scale-dependent, and this method is reachable
        # for any bounded numeric scale (likert 1-5, grades 0-100), not just
        # [0, 1]. Passing the wrong bounds returns a CI on the [0, 1] scale
        # while the estimand lives on the real one -- 0% coverage, not a
        # subtle miscalibration.
        _lo, _hi = score_range if score_range is not None else (0.0, 1.0)
        return _ppi_paired_logit_t(a, b, a_lab, b_lab, alpha, lo=_lo, hi=_hi)
    if method in ("t_interval", "bootstrap"):
        return _ppi_paired_arrays(a, b, a_lab, b_lab, np.mean, alpha, n_boot, rng, rectifier_func=np.mean)
    if method == "wilcoxon":
        return _ppi_paired_arrays(a, b, a_lab, b_lab, np.median, alpha, n_boot, rng, rectifier_func=np.mean)
    if method == "mannwhitney":
        # Matches evalstats.tests.mannwhitney's method="global" default
        # (reinstated 2026-08-02, a few hours after "ridge" -- see
        # mannwhitney's `method` docstring's "REVERTED TO 'global'" note
        # for the full six-default history). NOT a finding against
        # "ridge" -- it remains the best-validated option by every number
        # gathered -- reverted because the harness's --official-tests
        # suite has always tested plain "global" under the name "mwu"
        # (hardcoded, doesn't track mannwhitney()'s actual default), so
        # production stays aligned with what's actually been exercised by
        # that sanctioned pipeline until "ridge" gets its own official
        # pass.
        # KNOWN, NOT FIXED (2026-08-24): conservative on likert. Mid-rank
        # placement over a 5-level scale collapses the influence function
        # (~22% tied pairs), giving corrected Type-I ~0.011 vs ~0.04 for the
        # parametric tests. Unlike wilcoxon, this path never reaches
        # correct()'s smoothed-bootstrap jitter (ppi._tie_jitter_scale) --
        # the likeliest fix, but unvalidated here. Errs safe (under-rejects).
        return _ppi_two_sample(a, b, a_lab, b_lab, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, alpha, n_boot, rng)
    raise ValueError(
        f"PPI alignment correction has no validated implementation for pairwise "
        f"method {method!r}. Supported pairwise methods: "
        f"{', '.join(repr(m) for m in _PPI_PAIRWISE_SUPPORTED)}. "
        "Pass method=\"auto\" to let PPI correction pick a supported method "
        "automatically, or choose one of the methods above explicitly."
    )


def _ppi_pairwise_unpaired_fallback(a, b, a_lab, b_lab, alpha: float, n_boot: int, rng):
    """Independent-groups PPI fallback for a paired-mean-diff estimand.

    Used when two entities don't have enough commonly-labeled items (same
    item labeled for both) for a proper paired PPI correction, but each
    individually has enough of its own labels. Mathematically valid because
    ``mean(a) - mean(b)`` decomposes into two independent rectifiers — this
    is exactly ``evalstats.tests._ppi_two_sample``'s validated "TTEST" PPI
    form (see ``simulations/harness/cases/pvalues.py``), just applied to
    what would otherwise be a paired comparison.
    """
    from evalstats.tests import _ppi_two_sample
    return _ppi_two_sample(a, b, a_lab, b_lab, lambda ya, yb: float(ya.mean() - yb.mean()), alpha, n_boot, rng)


_JOINT_BOOT_SE_REL_FLOOR = 0.20
"""Relative floor on a bootstrap replicate's SE, as a fraction of the
observed SE, inside :func:`_ppi_bootstrap_t_joint_stats`. See the comment at
its use site for the failure mode this prevents and how 0.20 was calibrated.
Set to 0.0 to reproduce the pre-fix behaviour exactly."""


def _ppi_bootstrap_t_joint_stats(
    scores_2d: np.ndarray,
    lab_matrix: np.ndarray,
    pair_keys: list,
    entity_idx: dict,
    n_boot: int,
    rng: np.random.Generator,
    *,
    power_tune: bool = True,
):
    """Joint studentized-bootstrap statistics shared by every PPI compound
    construction that needs the correlation structure between pairs: max-T
    simultaneous CIs, "boot" CI-widening (via ``_ppi_alpha_eff_from_M_b``),
    and Romano-Wolf step-down p-values (via
    ``_ppi_romano_wolf_pvalues_from_joint_stats``) all derive from this ONE
    joint resample, computed once and reused across whichever apply, rather
    than re-bootstrapping per construction.

    Mirrors ``_ppi_paired_bootstrap_t``'s per-pair two-term variance
    decomposition (full-sample mean + labeled-subset rectifier), but draws
    ONE shared item-resample per bootstrap replicate across every pair
    instead of independent per-pair resamples — this is what makes the
    joint distribution of the standardized statistics across pairs valid
    (the same principle as the non-PPI ``bootstrap_t`` branch of
    ``evalstats.core.paired._max_stat_simultaneous_cis``, generalized to
    PPI's two-term SE).

    Requires every pair to share the same labeled-item positions — true
    whenever items (not (entity, item) cells) are labeled, matching
    evalstats' paired-by-input design. Returns ``None`` if that doesn't
    hold, or if there are fewer than 15 shared labeled items, so the caller
    can fall back to Bonferroni/Shaffer.

    ``power_tune`` : when *True* (the default), each pair's point
    estimate/variance use the same closed-form variance-minimizing
    lambda* :func:`evalstats.ppi._analytic_mean_point_se` derives for
    ``ppi_t_interval``/``ppi_logit_t``/Tango (since evalstats.tests.
    _ppi_paired_mj_floor's own power-tuning flip), instead of the fixed
    lambda=1 rectifier -- generalized here to a per-pair lambda*, computed
    once per pair (closed-form, not re-estimated per bootstrap replicate,
    avoiding the "double dipping" undercoverage a same-draw lambda/CI
    estimate would introduce), then held fixed across every bootstrap
    replicate for that pair, the same way a fixed weight would be held
    fixed while bootstrapping a weighted mean. Passing *False* reproduces
    the original fixed-lambda=1 construction exactly (bit-for-bit): at
    lambda=1 the general two-term variance collapses back to
    ``Var(unlabeled diffs)/n_unlab + Var(rectifier residuals)/n_lab``
    precisely, the identity the un-tuned path still computes directly
    rather than via the general formula, so *False* pays none of *True*'s
    extra per-pair covariance bookkeeping.

    Validated across 15 (k, N, label_frac) conditions spanning k=3-5 arms,
    N=50-400 items, label fractions 10-40% (including the realistic
    N >> N_lab regime), plus a dedicated high-rep recheck (600 reps) of
    the conditions with the largest apparent FWER movement at screening-
    tier rep counts, which confirmed that movement was Monte Carlo noise,
    not real inflation -- worst observed FWER across the full grid was
    within ~1 SE of nominal alpha, with zero power regressions and
    frequently substantial power gains (e.g. Romano-Wolf power at N=400,
    40% labeled: 78% -> 100%). This restores Romano-Wolf's power edge over
    Shaffer that Tango's own power-tuning flip had temporarily broken (see
    tests/test_compound_ppi_fwer.py's TestRomanoWolfCalibration). See
    ``simulations/investigate_joint_bootstrap_power_tune*.py`` and
    ``simulations/investigate_joint_bootstrap_fwer_highrep.py`` for the
    full validation.

    Returns
    -------
    tuple or None
        ``(point_ests, obs_se, valid_pairs, T, t_obs)`` — ``point_ests`` and
        ``obs_se`` have shape ``(k,)`` (one per pair, in *pair_keys* order),
        ``valid_pairs`` is a boolean mask over pairs with non-degenerate SE,
        ``T`` has shape ``(n_boot, k)`` (the per-replicate, per-pair
        studentized statistic -- callers needing only the joint max (for
        max-T/"boot") should reduce it via
        ``np.max(np.abs(T[:, valid_pairs]), axis=1)``; callers needing the
        full matrix (Romano-Wolf's step-down) use it directly), and
        ``t_obs`` has shape ``(k,)``.
    """
    from evalstats.ppi import (
        _adaptive_shrink_lambda, _analytic_mean_lambda_replicates, _lambda_var_inflation,
    )

    k = len(pair_keys)

    lab_masks = []
    for (ea, eb) in pair_keys:
        ia, ib = entity_idx[ea], entity_idx[eb]
        lab_masks.append(~np.isnan(lab_matrix[ia]) & ~np.isnan(lab_matrix[ib]))
    first_mask = lab_masks[0]
    if not all(np.array_equal(m, first_mask) for m in lab_masks[1:]):
        return None
    lab_positions = np.where(first_mask)[0]
    # unlab_positions must be DISJOINT from lab_positions -- obs_se/boot_se
    # below assume the full-sample and rectifier terms are independent,
    # which only holds for disjoint samples. See _ppi_two_sample and
    # evalstats.ppi.correct's docstring.
    unlab_positions = np.where(~first_mask)[0]
    n_lab = len(lab_positions)
    n_unlab = len(unlab_positions)
    if n_lab < 15:
        return None

    diffs_unlab = np.empty((k, n_unlab))
    lab_true_items = np.empty((k, n_lab))
    lab_llm_items = np.empty((k, n_lab))
    point_ests = np.empty(k)
    obs_se = np.empty(k)
    lam = np.ones(k)
    lambda_extra_var = np.zeros(k)  # per-pair r_term**2 * Var(lambda_hat), held fixed across replicates like lam
    for p_idx, (ea, eb) in enumerate(pair_keys):
        ia, ib = entity_idx[ea], entity_idx[eb]
        d_all = scores_2d[ia] - scores_2d[ib]
        d_unlab = d_all[unlab_positions]
        d_lab_llm = d_all[lab_positions]
        d_lab_true = (lab_matrix[ia] - lab_matrix[ib])[lab_positions]
        diffs_unlab[p_idx] = d_unlab
        lab_true_items[p_idx] = d_lab_true
        lab_llm_items[p_idx] = d_lab_llm

        f_unlab = float(np.mean(d_unlab))
        f_lab = float(np.mean(d_lab_true))
        f_hat_lab = float(np.mean(d_lab_llm))

        if not power_tune:
            # Exact fixed-lambda=1 identity: Var(d_lab_true - d_lab_llm, ddof=1)
            # equals var_lab*n_lab + var_hat_lab*n_lab - 2*cov_lab_hatlab*n_lab
            # (the ddof=1 sample-variance identity), so this reproduces the
            # general formula's lambda=1 case without computing the
            # covariance term at all.
            var_unlab = float(np.var(d_unlab, ddof=1))
            var_rect = float(np.var(d_lab_true - d_lab_llm, ddof=1)) if n_lab > 1 else 0.0
            point_ests[p_idx] = f_unlab + (f_lab - f_hat_lab)
            obs_se[p_idx] = np.sqrt(var_unlab / n_unlab + var_rect / n_lab)
            continue

        var_unlab_n = float(np.var(d_unlab, ddof=1)) / n_unlab if n_unlab > 1 else 0.0
        var_lab_n = float(np.var(d_lab_true, ddof=1)) / n_lab if n_lab > 1 else 0.0
        var_hat_lab_n = float(np.var(d_lab_llm, ddof=1)) / n_lab if n_lab > 1 else 0.0
        cov_lab_n = (
            float(np.cov(d_lab_true, d_lab_llm, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0
        )

        lam_p_raw = 1.0
        denom = var_unlab_n + var_hat_lab_n
        if denom > 1e-12:
            lam_p_raw = min(max(cov_lab_n / denom, 0.0), 1.0)
        # Adaptive shrinkage (see evalstats.ppi._adaptive_shrink_lambda's
        # docstring for the shared rationale) -- was previously fixed
        # toward a target of 1 regardless of what the data supported,
        # unlike every other power_tune site in this codebase. Falls back
        # to target=1 when d_lab_true is near-degenerate, same guard
        # evalstats.ppi._analytic_mean_point_se uses.
        raw_var_lab_true = float(np.var(d_lab_true, ddof=1)) if n_lab > 1 else 0.0
        raw_var_lab_llm = float(np.var(d_lab_llm, ddof=1)) if n_lab > 1 else 0.0
        if n_lab <= 1 or raw_var_lab_true < raw_var_lab_llm * 1e-6:
            lam_p_replicates = None
        else:
            lam_p_replicates = _analytic_mean_lambda_replicates(d_lab_true, d_lab_llm, var_unlab_n, n_lab)
        lam_p = _adaptive_shrink_lambda(lam_p_raw, lam_p_replicates, n_lab)
        lam[p_idx] = lam_p

        point_ests[p_idx] = f_lab + lam_p * (f_unlab - f_hat_lab)
        var_estimate = max(
            var_lab_n + lam_p * lam_p * (var_unlab_n + var_hat_lab_n) - 2.0 * lam_p * cov_lab_n, 0.0,
        )
        # Precompute the extra variance ONCE per pair, from the OBSERVED
        # (fixed) r_term -- not re-derived per bootstrap replicate below --
        # matching how lam itself is held fixed across every replicate for
        # this pair. An earlier version used each replicate's own resampled
        # r_term_b instead, making the injected variance data-dependent
        # within the bootstrap itself; a paired high-rep recheck confirmed
        # that was the cause of a real FWER regression in one tested
        # condition (fixed by holding r_term fixed here) -- see
        # simulations/out/results_why_ppi_shrink_1_over_0.md Addendum 20/21.
        lambda_extra_var[p_idx] = _lambda_var_inflation(f_unlab - f_hat_lab, lam_p_replicates)
        var_estimate += lambda_extra_var[p_idx]
        obs_se[p_idx] = np.sqrt(var_estimate)

    boot_theta = np.empty((n_boot, k))
    boot_se = np.empty((n_boot, k))
    chunk_size = max(1, min(n_boot, 512))
    start = 0
    while start < n_boot:
        stop = min(start + chunk_size, n_boot)
        m = stop - start
        idx_all = rng.integers(0, n_unlab, size=(m, n_unlab))  # shared across pairs
        idx_lab = rng.integers(0, n_lab, size=(m, n_lab))      # shared across pairs
        unlab_samples = diffs_unlab[:, idx_all]  # (k, m, n_unlab)

        if not power_tune:
            rect_samples = lab_true_items[:, idx_lab] - lab_llm_items[:, idx_lab]  # (k, m, n_lab)
            boot_theta[start:stop] = unlab_samples.mean(axis=2).T + rect_samples.mean(axis=2).T
            boot_se[start:stop] = np.sqrt(
                unlab_samples.var(axis=2, ddof=1).T / n_unlab
                + rect_samples.var(axis=2, ddof=1).T / n_lab
            )
            start = stop
            continue

        lab_true_samples = lab_true_items[:, idx_lab]  # (k, m, n_lab)
        lab_llm_samples = lab_llm_items[:, idx_lab]    # (k, m, n_lab)
        f_unlab_b = unlab_samples.mean(axis=2)   # (k, m)
        f_lab_b = lab_true_samples.mean(axis=2)  # (k, m)
        f_hat_lab_b = lab_llm_samples.mean(axis=2)  # (k, m)
        var_unlab_b = unlab_samples.var(axis=2, ddof=1) / n_unlab      # (k, m)
        var_lab_b = lab_true_samples.var(axis=2, ddof=1) / n_lab       # (k, m)
        var_hat_lab_b = lab_llm_samples.var(axis=2, ddof=1) / n_lab    # (k, m)
        # Per-(pair, replicate) covariance of the two labeled-subset means --
        # np.cov doesn't vectorize over the extra replicate axis, so computed
        # directly from the mean-deviation product (the ddof=1 formula).
        cov_lab_b = (
            (lab_true_samples - f_lab_b[:, :, None]) * (lab_llm_samples - f_hat_lab_b[:, :, None])
        ).sum(axis=2) / (n_lab - 1) / n_lab if n_lab > 1 else np.zeros((k, m))

        lam_col = lam[:, np.newaxis]  # (k, 1), broadcasts against (k, m)
        theta_b = f_lab_b + lam_col * (f_unlab_b - f_hat_lab_b)
        var_b = np.maximum(
            var_lab_b + lam_col * lam_col * (var_unlab_b + var_hat_lab_b) - 2.0 * lam_col * cov_lab_b, 0.0,
        )
        # Same fixed per-pair extra variance as obs_se above, broadcast
        # across every replicate (not re-derived per replicate) -- see the
        # comment there.
        var_b = var_b + lambda_extra_var[:, np.newaxis]
        boot_theta[start:stop] = theta_b.T
        boot_se[start:stop] = np.sqrt(var_b).T
        start = stop

    # Floor each replicate's SE relative to the OBSERVED one before
    # studentizing. T is a studentized statistic, so the meaningful scale
    # for "this replicate's SE is degenerate" is obs_se, not an absolute
    # constant -- and the previous guard was absolute (1e-12), which cannot
    # catch a boot_se that is small-but-nonzero.
    #
    # Why it matters: when a pair is near-degenerate (a paired difference
    # with almost no item-level spread -- e.g. two nearly identical arms, or
    # a generator that shifts every item by the same constant), a resample
    # can draw an almost-constant vector, collapsing boot_se far below
    # obs_se and sending |T| to 60-2000. Because BOTH consumers of this
    # joint resample reduce it with a MAX over pairs
    # (_ppi_romano_wolf_pvalues_from_joint_stats' step-down suffix-max, and
    # _M_b_from_T for max-T/"boot" CI widening), one such pair poisons every
    # other pair in the family: measured on a k=3 cell, a degenerate pair
    # with |T|max=66 drove the UNRELATED extreme pair's Romano-Wolf p from
    # ~0 to 0.363 while its own CI still excluded 0 by a wide margin -- the
    # p-value and the CI contradicting each other inside one bundle.
    #
    # Calibration of the 0.20 coefficient: under regularity boot_se/obs_se
    # concentrates at 1 with sd ~ 1/sqrt(2*n_lab) (~0.11 at n_lab=40), so
    # 0.20 sits many sd below anything a well-behaved resample produces.
    # Measured binding rates (fraction of replicates floored): 0.0000% on
    # every non-degenerate condition tested (k=3/5, N=100-400,
    # n_lab=20-160, judge rho=0.80-0.99, including the small-n_lab +
    # excellent-judge corner where boot_se is most variable), versus
    # 12-19% on the degenerate cells this exists for. It is inert where the
    # bootstrap is healthy and only engages where it has broken down.
    # Validated for FWER, not just power. Note the ordinary nulls are
    # UNINFORMATIVE for that: under them the paired truth difference is
    # exactly constant (uniq==1), so boot_se never collapses and the floor
    # is inert -- Type-I is then identical for trivial reasons. The real
    # test is a null where the floor DOES bind, i.e. the near-identical-arms
    # case above: arms sharing a base, each perturbing a small random subset
    # of items by +/- delta with mean-zero signs, so the null holds exactly
    # while d_true has several distinct values and tiny variance. With
    # binding up to 12% of replicates under that null, FWER is unchanged
    # (largest move +0.0025 at 0.28% binding, 0.23 MC SE, on 400 reps).
    # Power on the degenerate alternative recovers 0.6425 -> 0.9975 (k=3
    # N=100), 0.3850 -> 1.0000 (k=5) with FWER byte-identical.
    # See simulations/investigate_joint_bootstrap_se_floor_*.py.
    boot_se = np.maximum(boot_se, _JOINT_BOOT_SE_REL_FLOOR * obs_se[np.newaxis, :])
    se_boot_safe = np.where(boot_se > 1e-12, boot_se, 1.0)
    T = (boot_theta - point_ests[np.newaxis, :]) / se_boot_safe  # (n_boot, k)

    valid_pairs = obs_se > 1e-12
    if not np.any(valid_pairs):
        return None

    obs_se_safe = np.where(valid_pairs, obs_se, 1.0)
    t_obs = np.abs(point_ests) / obs_se_safe

    return point_ests, obs_se, valid_pairs, T, t_obs


def _ppi_romano_wolf_pvalues_from_joint_stats(
    point_ests: np.ndarray,
    obs_se: np.ndarray,
    valid_pairs: np.ndarray,
    T: np.ndarray,
    t_obs: np.ndarray,
    pair_keys: list,
) -> dict:
    """Romano & Wolf (2005) bootstrap step-down p-values, built on the SAME
    joint resample ``_ppi_bootstrap_t_joint_stats`` produces for max-T/"boot"
    (no re-bootstrapping) -- the PPI-corrected analogue of
    ``evalstats.core.paired.romano_wolf_stepdown_pvalues``, using the exact
    same step-down algorithm (verified structurally identical below), just
    driven by PPI's joint (unlabeled + rectifier) studentized statistic
    instead of a raw per-item bootstrap-t.

    Recovers power over Shaffer's (evalstats' other PPI p-value-correction
    option) by refining the joint critical value at each step to the max
    over pairs NOT YET rejected, rather than a single joint statistic (boot)
    or Shaffer's static combinatorial divisor sequence -- the same
    "correlation-aware, adaptively-shrinking" advantage Romano-Wolf has over
    Bonferroni/Shaffer on the non-PPI side. Because PPI's own marginal
    p-value is noisier than a raw estimator's (the labeled-subset rectifier
    adds a second variance term), the RANKING this step-down relies on to
    remove clearly-non-null pairs early is itself noisier here, so the
    power recovery is real but smaller than the non-PPI case's -- see
    ``tests/test_compound_ppi_fwer.py``'s calibration tests for the
    magnitude actually observed.

    Parameters
    ----------
    point_ests, obs_se, valid_pairs, T, t_obs
        Exactly the tuple ``_ppi_bootstrap_t_joint_stats`` returns.
    pair_keys : list
        Pairs in the same order as the *point_ests*/etc. arrays.

    Returns
    -------
    dict[tuple[str, str], float]
        Maps each pair to its Romano-Wolf FWER-adjusted p-value (monotonized
        via a running max along the testing order, same as Holm's/the
        non-PPI Romano-Wolf's own adjusted p-values). Pairs with degenerate
        (near-zero) SE are excluded from the family being stepped down over
        and assigned p-value 1.0, since they carry no information to test.
    """
    k = len(pair_keys)
    n_boot = T.shape[0]

    if not np.any(valid_pairs):
        return {pair: 1.0 for pair in pair_keys}

    valid_idx = np.where(valid_pairs)[0]
    t_obs_v = t_obs[valid_idx]
    T_v = T[:, valid_idx]  # (n_boot, k_valid)

    order = np.argsort(-t_obs_v)  # descending observed |t|: tested first
    T_abs_sorted = np.abs(T_v)[:, order]  # (n_boot, k_valid)
    # suffix_max[:, step] = max over pairs tested at or after `step` (per
    # bootstrap draw) -- the step-down "remaining hypotheses" set.
    suffix_max = np.maximum.accumulate(T_abs_sorted[:, ::-1], axis=1)[:, ::-1]  # (n_boot, k_valid)
    t_obs_sorted = t_obs_v[order]
    extreme_counts = (suffix_max >= t_obs_sorted[np.newaxis, :]).sum(axis=0)  # (k_valid,)
    raw_step_p_sorted = (extreme_counts + 1) / (n_boot + 1)
    adjusted_sorted = np.minimum(np.maximum.accumulate(raw_step_p_sorted), 1.0)

    adjusted_valid = np.empty(len(valid_idx))
    adjusted_valid[order] = adjusted_sorted

    adjusted = np.ones(k)
    adjusted[valid_idx] = adjusted_valid
    return {pair: float(adjusted[i]) for i, pair in enumerate(pair_keys)}


def _M_b_from_T(T: np.ndarray, valid_pairs: np.ndarray) -> np.ndarray:
    """Reduce the full joint studentized-T matrix (from
    ``_ppi_bootstrap_t_joint_stats``) to the joint max-|T| statistic per
    bootstrap replicate, for callers (max-T CIs, "boot" CI-widening) that
    only need the single-step joint maximum rather than Romano-Wolf's
    full per-replicate matrix."""
    return np.max(np.abs(T[:, valid_pairs]), axis=1)


def _max_t_from_joint_stats(
    point_ests: np.ndarray,
    obs_se: np.ndarray,
    valid_pairs: np.ndarray,
    M_b: np.ndarray,
    t_obs: np.ndarray,
    pair_keys: list,
    ci: float,
):
    """Turn joint bootstrap-t statistics (from
    ``_ppi_bootstrap_t_joint_stats``) into simultaneous CIs + max-T
    p-values at a given confidence level, without re-bootstrapping —
    ``M_b`` is reused across every confidence level needed (headline +
    gradient bands) since only its quantile changes.
    """
    c = float(np.quantile(M_b, ci))
    B_total = len(M_b)
    sim_cis: dict = {}
    max_t_pvalues: dict = {}
    for p_idx, pair in enumerate(pair_keys):
        if valid_pairs[p_idx]:
            half = c * float(obs_se[p_idx])
            sim_cis[pair] = (float(point_ests[p_idx] - half), float(point_ests[p_idx] + half))
            extreme = int(np.sum(M_b >= t_obs[p_idx]))
            max_t_pvalues[pair] = float((extreme + 1) / (B_total + 1))
        else:
            sim_cis[pair] = (float(point_ests[p_idx]), float(point_ests[p_idx]))
            max_t_pvalues[pair] = 1.0
    return sim_cis, max_t_pvalues


def _ppi_alpha_eff_from_M_b(M_b: np.ndarray, ci: float) -> float:
    """Convert a joint bootstrap max-|T| distribution into an effective
    per-pair alpha, for widening an EXISTING alpha-parameterized CI formula
    instead of building a new Wald-type CI directly from ``M_b`` (which is
    what :func:`_max_t_from_joint_stats` does).

    Mirrors ``evalstats.core.paired._joint_bootstrap_scaled_simultaneous_cis``'s
    own critical-value -> effective-alpha conversion (``alpha_eff =
    2*(1-Phi(c))``) exactly, so any already-validated closed-form PPI
    dispatch (Tango/ppi_logit_t/ppi_t_interval/...) can be evaluated at
    ``alpha_eff`` in place of the marginal alpha, accounting for the
    correlation between pairs the way Sidak/Bonferroni's independence-based
    adjustments cannot -- without discarding that method's own CI shape the
    way reusing ``M_b``'s point estimate/SE directly would.
    """
    c = float(np.quantile(M_b, ci))
    alpha_eff = float(2.0 * (1.0 - _scipy_norm.cdf(c)))
    return min(max(alpha_eff, 1e-9), 1.0 - 1e-9)


def _ppi_robustness_dispatch(method: str, a, a_lab, alpha: float, n_boot: int, rng,
                             score_range: Optional[tuple[float, float]] = None):
    """Dispatch to the PPI-corrected single-sample implementation of *method*."""
    from evalstats.tests import (
        _ppi_single_wilson, _ppi_single_bootstrap_t, _ppi_single_t_interval, _ppi_single_logit_t,
    )
    if method == "wilson":
        return _ppi_single_wilson(a, a_lab, alpha)
    if method == "bootstrap_t":
        return _ppi_single_bootstrap_t(a, a_lab, alpha, n_boot, rng)
    if method == "ppi_t_interval":
        return _ppi_single_t_interval(a, a_lab, alpha)
    if method == "ppi_logit_t":
        # lo/hi from the resolved score_range -- see _ppi_pairwise_dispatch's
        # matching note for why the (0, 1) default is wrong here.
        _lo, _hi = score_range if score_range is not None else (0.0, 1.0)
        return _ppi_single_logit_t(a, a_lab, alpha, lo=_lo, hi=_hi)
    if method == "bootstrap":
        from evalstats.ppi import correct as _ppi_correct
        mask = ~np.isnan(a_lab)
        if mask.sum() == 0:
            raise ValueError("No positions have human labels in a_lab.")
        # Y_hat_unlab must be DISJOINT from the labeled positions -- correct()'s
        # bootstrap independently resamples the two terms, which is only valid
        # for genuinely separate samples. `a` is the full per-entity array (a[mask]
        # are the same items as a_lab[mask]), so exclude them here.
        return _ppi_correct(
            np.mean, Y_lab=a_lab[mask], Y_hat_lab=a[mask], Y_hat_unlab=a[~mask],
            alpha=alpha, n_boot=n_boot, rng=rng, compute_pvalue=False,
        )
    raise ValueError(
        f"PPI alignment correction has no validated implementation for robustness "
        f"(single-entity) method {method!r}. Supported robustness methods: "
        f"{', '.join(repr(m) for m in _PPI_ROBUSTNESS_SUPPORTED)}. "
        "Pass method=\"auto\" to let PPI correction pick a supported method "
        "automatically, or choose one of the methods above explicitly."
    )


def _run_alignment_ppi(
    cr: "ComparisonResult",
    *,
    df: pd.DataFrame,
    metric_col: str,
    factor_col: str,
    item_col: str,
    alignment_result,
    alpha: float,
    n_boot: int,
    correction: str,
    method: str,
    rng,
    prefer: str = "auto",
) -> None:
    """Override CIs/p-values in ``cr._analysis`` in-place using Prediction-Powered
    Inference (PPI), by dispatching to the specific PPI-corrected implementation
    of whichever pairwise/robustness method applies (see
    ``_ppi_pairwise_dispatch``/``_ppi_robustness_dispatch`` above).

    ``prefer`` mirrors ``_simultaneous_cis_router``'s knob of the same name and
    resolves through the SAME N-threshold table
    (``resolve_auto_simultaneous_ci_method``) the non-PPI path uses --
    matching evalstats' decision tree exactly, which has no max-T node at
    all: ``"sidak"`` for small N (or a lopsided binary split regardless of
    N), else ``"boot"`` (joint bootstrap with an effective alpha). Sidak is a
    pure closed-form alpha adjustment applied to whichever PPI dispatch
    method is in play, so it needs no resampling and works for every pair
    regardless of branch. "boot" reuses ``_ppi_bootstrap_t_joint_stats``'s
    joint resampling (valid for any paired PPI method, since the point
    estimate/variance it computes -- f_unlab + rectifier, and the additive
    two-term SE -- is the shared structure every one of them wraps a
    method-specific CI shape around) to translate a joint critical value into
    an effective alpha, then evaluates ``pairwise_method``'s own closed-form
    CI at that adjusted level (see ``_ppi_alpha_eff_from_M_b``) -- ALWAYS,
    regardless of which method resolved, never switching to a different
    construction for one particular method. Both "sidak" and "boot" fall back
    to Bonferroni when the required shared-labeled-item structure isn't
    available (matching ``_simultaneous_cis_router``'s own
    boot-degenerates-to-Bonferroni precedent). "max_t" is a separate,
    non-tree construction reachable ONLY via an explicit ``prefer="max_t"``
    request (never selected by ``prefer="auto"``, matching the non-PPI side's
    identical convention) and requires ``pairwise_method=="bootstrap_t"``,
    whose own natural Wald-type construction happens to match the joint
    statistic's shape exactly, so it is used directly (its own point
    estimate, CI, AND joint p-value) rather than wrapped. Pass
    ``prefer="max_t"``/``"bonferroni"``/``"sidak"``/``"boot"`` to force one
    directly.

    IMPORTANT CAVEAT: unlike the non-PPI Sidak/boot (validated via
    ``simulations/harness/cases/pvalues.py --mode simultaneous_ci``, built
    entirely on raw/uncorrected scores), applying that SAME construction to
    a PPI-corrected ``ci_func`` has not itself been swept by the harness --
    no existing simulation combines PPI alignment correction with multi-arm
    FWER control (``--mode ppi`` is 2-group/single-arm only; ``--mode
    multiarm``/``simultaneous_ci`` never touch PPI-corrected estimates; see
    this file's own compound-scenario tests in
    ``tests/test_compound_ppi_fwer.py`` for what IS checked so far -- null
    calibration and power in one binary scenario, not a harness-scale sweep).
    The math generalizes cleanly (Sidak only needs a valid per-alpha CI;
    "boot" only needs a joint distribution of a standardized statistic, both
    of which a PPI-corrected estimate still provides), but treat this as
    provisional until validated at harness scale.

    ``correction`` similarly mirrors ``resolve_auto_pvalue_correction_method``'s
    non-PPI tree: ``"auto"`` (the default) resolves to ``"romano_wolf"`` at
    N>=30 (or a lopsided binary split forces ``"shaffer"`` regardless of N),
    else ``"shaffer"``. PPI-generalized Romano-Wolf
    (``_ppi_romano_wolf_pvalues_from_joint_stats``) reuses the SAME joint
    resample "boot" needs above (computed once, shared between both when
    both apply) and runs Romano-Wolf's exact step-down algorithm on PPI's
    joint (unlabeled + rectifier) studentized statistic instead of a raw
    per-item bootstrap -- so it shares "boot"'s requirements (every pair
    "dispatch" branch, >=15 shared labeled items) and falls back to
    Shaffer's when they aren't met, with a warning. UNLIKE the Sidak/boot
    CI caveat above, this WAS validated before becoming the default: a
    9-condition grid (k=3-5 arms, N=50-200, label fractions 20-40%, binary
    data, paired Shaffer-vs-Romano-Wolf comparisons on identical data) found
    worst-case FWER 0.067 against nominal 0.05 (within normal Monte Carlo
    noise at the rep counts used, no systematic inflation across the grid)
    and power at or above Shaffer's in every condition tested (0/9
    regressions, gains ranging from negligible to ~50% relative at small
    N/sparse labels). Still smaller-scale than a harness sweep -- see
    ``tests/test_compound_ppi_fwer.py``'s ``TestRomanoWolfCalibration`` for
    the pytest-scale version of this check, and treat further validation
    (more label fractions, non-binary data, larger N) as still open.

    For ``method="auto"`` the PPI-specific auto table
    (``evalstats.config.resolve_ppi_auto_methods``) picks a method validated
    for PPI use, which need not match the non-aligned auto default for the
    same data (e.g. binary data defaults to ``bayes_binary``/``mj_floor``
    depending on N without alignment, but always ``mj_floor`` once PPI
    correction is in play, since ``bayes_binary`` has no PPI-corrected form).
    When the user passes an explicit ``method=``, that exact method's
    PPI-corrected counterpart is used, and a clear ``ValueError`` is raised if
    none exists — PPI correction never silently substitutes or falls back to
    an unvalidated method.

    Every comparison in evalstats is paired by input (see
    ``evalstats.core.paired``'s module docstring), so this pairs items by
    their shared position in ``bundle.benchmark``'s item-aligned score matrix
    (``get_2d_scores()``), rather than treating LLM scores as independent
    per-entity groups.
    """
    bundle = cr._primary_bundle()
    if bundle is None:
        warnings.warn(
            "PPI alignment correction is not yet supported for multi-bundle "
            "(multi-model) results. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return

    if bundle.benchmark.is_seeded:
        raise ValueError(
            "PPI alignment correction does not yet support seeded benchmarks "
            "(R >= 3 repeated runs). Aggregate runs to a single score per "
            "(template, input) cell before passing alignment=."
        )

    if bundle.p_value_method == "nem":
        raise ValueError(
            "PPI alignment correction has no validated implementation for "
            "pairwise_test=\"nemenyi\" (Nemenyi post-hoc p-values derive from "
            "Friedman ranks and there is no PPI-corrected Nemenyi in evalstats "
            "yet). Use p_values=True (default bootstrap) or "
            "pairwise_test=\"wilcoxon\" instead, both of which have validated "
            "PPI corrections."
        )

    rng = np.random.default_rng(rng)

    labels = list(bundle.benchmark.template_labels)
    input_labels = list(bundle.benchmark.input_labels)
    n_entities = len(labels)
    n_items = len(input_labels)
    entity_idx = {e: i for i, e in enumerate(labels)}
    item_idx = {it: j for j, it in enumerate(input_labels)}

    # Item-aligned LLM score matrix: scores_2d[i, j] = entity i's score on
    # item j (NaN for incomplete-design cells; averaged over runs/evaluators).
    scores_2d = bundle.benchmark.get_2d_scores()

    # ── Extract labeled/unlabeled arrays (dataset-level counts only) ──────────
    from evalstats.ppi import resolve_arrays
    Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab = resolve_arrays(
        df, metric_col=metric_col, group_col=factor_col, alignment_result=alignment_result
    )

    # NOTE: resolve_arrays' Y_hat_unlab EXCLUDES the labeled rows (disjoint,
    # as correct() requires) -- len(df) is the right "total dataset size"
    # for the checks below, not len(Y_hat_unlab) (which undercounts by
    # n_lab).
    n_all = len(df)
    n_lab = len(Y_lab)

    # ── Minimum sample-size requirements ─────────────────────────────────────
    if n_lab < 15:
        raise ValueError(
            f"PPI alignment requires at least 15 human-labeled items; "
            f"got n_lab={n_lab}. Expand the alignment set and re-run "
            "judge_alignment()."
        )
    if n_all < 50:
        raise ValueError(
            f"PPI alignment requires at least 50 items in the full dataset; "
            f"got N={n_all}. PPI is only beneficial at scale."
        )
    if n_lab < 30:
        warnings.warn(
            f"PPI alignment: only {n_lab} human-labeled items (recommend ≥ 30). "
            "Confidence intervals may under-cover at this sample size.",
            UserWarning,
            stacklevel=4,
        )
    if n_all < 100:
        warnings.warn(
            f"PPI alignment: only {n_all} total items (recommend ≥ 100). "
            "Confidence intervals may under-cover at this sample size.",
            UserWarning,
            stacklevel=4,
        )

    # ── Item-aligned human-label matrix (n_entities x n_items) ───────────────
    human_col = alignment_result.human_col
    lab_rows = df.loc[df[human_col].notna(), [factor_col, item_col, human_col]]
    lab_matrix = np.full((n_entities, n_items), np.nan)
    for e_val, it_val, h_val in zip(
        lab_rows[factor_col].astype(str).to_numpy(),
        lab_rows[item_col].astype(str).to_numpy(),
        lab_rows[human_col].to_numpy(dtype=float),
    ):
        ei = entity_idx.get(e_val)
        ij = item_idx.get(it_val)
        if ei is not None and ij is not None:
            lab_matrix[ei, ij] = h_val

    missing_entities = [e for i, e in enumerate(labels) if np.all(np.isnan(lab_matrix[i]))]
    if missing_entities:
        warnings.warn(
            f"PPI alignment: the following entities have no human-labeled items and "
            f"will keep their uncorrected LLM-only estimate: {missing_entities}. "
            "Consider expanding the alignment set to cover all entities.",
            UserWarning,
            stacklevel=4,
        )

    # ── Label efficiency for the corrected estimates ──────────────────────
    #
    # Attached to `cr` for the summary to render. Within-subjects, so every
    # correlation is a "within" one and n_eff comes back against a single
    # condition's item count (_pair_total_n returns one condition's length for
    # design="within", since all conditions share the same items) -- no
    # division by condition count here, unlike the between-subjects path.
    #
    # Three different estimands, three different correlations, deliberately:
    #   marginal mean  -> that entity's own Pearson r^2 (mean influence
    #                     function is the identity)
    #   pairwise CI    -> the correlation of the PAIRED DIFFERENCES, which is
    #                     what a mean-difference interval's variance depends on
    #   pairwise p     -> Wilcoxon's own rank-based correlation
    # Reporting one of these next to all three would misdescribe two of them.
    try:
        from evalstats.alignment import _efficiency_metric, _marginal_efficiency
        _conds = {
            str(lbl): (scores_2d[i], lab_matrix[i]) for i, lbl in enumerate(labels)
        }
        cr._marginal_n_eff = [
            _marginal_efficiency(scores_2d[i], lab_matrix[i])[1]
            for i in range(len(labels))
        ]
        if any(v is None for v in cr._marginal_n_eff):
            cr._marginal_n_eff = None
        _, _ci_pairs = _efficiency_metric(_conds, test="ttest", design="within",
                                          want_pairs=True)
        _, _p_pairs = _efficiency_metric(_conds, test="wilcoxon", design="within",
                                         want_pairs=True)
        cr._pair_ci_eff = _ci_pairs
        cr._pair_p_eff = _p_pairs
        cr._omnibus_eff = None
        if len(labels) >= 3:
            _om, _ = _efficiency_metric(_conds, test="friedman", design="within",
                                        want_pairs=False)
            cr._omnibus_eff = _om
        _counts = [int(np.count_nonzero(~np.isnan(lab_matrix[i]))) for i in range(len(labels))]
        cr._n_lab_per_entity = float(np.mean(_counts)) if _counts else None
        # _print_bundle_summary is handed the bundle, not the ComparisonResult,
        # so mirror the values onto it; cr keeps them for programmatic access.
        for _a in ("_marginal_n_eff", "_pair_ci_eff", "_pair_p_eff",
                   "_omnibus_eff", "_n_lab_per_entity"):
            setattr(bundle, _a, getattr(cr, _a))
    except Exception:
        # Reporting extra: never allowed to break a correction that worked.
        cr._marginal_n_eff = cr._pair_ci_eff = cr._pair_p_eff = None
        cr._omnibus_eff = cr._n_lab_per_entity = None

    # ── Resolve the PPI-specific pairwise/robustness method ──────────────────
    from evalstats.core.resampling import is_binary_scores, is_bounded_01_scores
    from evalstats.config import resolve_ppi_auto_methods

    # Reuse the ONE data-kind decision method="auto"'s router already made
    # (recorded on the bundle) rather than re-deriving it here. The previous
    # local re-derivation was a binary/bounded_01/unbounded test with no
    # "likert" branch that consulted neither score_range nor eval_type, so
    # Likert data on e.g. a 1-5 scale fell through to "unbounded" and
    # silently took ppi_t_interval -- making PPI_AUTO_METHOD_TABLE's
    # "likert" row (ppi_logit_t) unreachable in every case it exists for.
    # The local test remains as the fallback for non-"auto" callers, where
    # the router records no resolution.
    data_kind = getattr(bundle, "resolved_data_kind", None)
    if data_kind is None:
        if is_binary_scores(scores_2d):
            data_kind = "binary"
        elif is_bounded_01_scores(scores_2d):
            data_kind = "bounded_01"
        else:
            data_kind = "unbounded"

    # Bounds for the scale-dependent dispatches (ppi_logit_t). Prefer the
    # router's resolved range; fall back to (0, 1) only when it recorded none.
    ppi_score_range = getattr(bundle, "resolved_score_range", None)
    if method == "auto":
        pairwise_method, robustness_method = resolve_ppi_auto_methods(data_kind)
    else:
        pairwise_method = bundle.resolved_method or method
        robustness_method = bundle.resolved_ci_method or method

    # ── Point estimates (per entity) ──────────────────────────────────────────
    final_means  = np.array(bundle.robustness.mean, dtype=float, copy=True)
    final_ci_low = np.array(bundle.robustness.ci_low, dtype=float, copy=True)
    final_ci_high = np.array(bundle.robustness.ci_high, dtype=float, copy=True)
    multi_ci_lo = {a: np.array(bundle.robustness.multi_ci[a][0], dtype=float, copy=True) for a in GRADIENT_CI_ALPHAS}
    multi_ci_hi = {a: np.array(bundle.robustness.multi_ci[a][1], dtype=float, copy=True) for a in GRADIENT_CI_ALPHAS}
    entity_rectifier = {e: 0.0 for e in labels}

    for i, e in enumerate(labels):
        if e in missing_entities:
            continue  # keep the uncorrected LLM-only estimate already copied above
        valid = ~np.isnan(scores_2d[i])
        arr = scores_2d[i, valid]
        lab_arr = lab_matrix[i, valid]

        res = _ppi_robustness_dispatch(robustness_method, arr, lab_arr, alpha, n_boot, rng, ppi_score_range)
        final_means[i] = res.estimate
        final_ci_low[i] = res.ci_low
        final_ci_high[i] = res.ci_high
        entity_rectifier[e] = res.rectifier
        for a in GRADIENT_CI_ALPHAS:
            g = _ppi_robustness_dispatch(robustness_method, arr, lab_arr, a, n_boot, rng, ppi_score_range)
            multi_ci_lo[a][i] = g.ci_low
            multi_ci_hi[a][i] = g.ci_high

    final_multi_ci = {a: (multi_ci_lo[a], multi_ci_hi[a]) for a in GRADIENT_CI_ALPHAS}

    # ── Pairwise diffs ────────────────────────────────────────────────────────
    pair_keys = list(bundle.pairwise.results.keys())
    n_pairs = len(pair_keys)

    # Simultaneous (family-wise) CIs. See the Sidak/boot/max-T resolution
    # block below (after pair classification) for how *pair_alpha* actually
    # gets set for each construction.
    use_simultaneous = bool(bundle.pairwise.simultaneous_ci) and n_pairs > 1

    # ── Classify pairs up front (cheap: no bootstrapping) ─────────────────────
    # Determines paired-dispatch vs. unpaired-fallback vs. skip-uncorrected for
    # every pair, so we know before running any bootstrap whether a joint
    # max-T correction is even possible (it requires every pair to use the
    # full paired dispatch — see below).
    pair_arrays: dict = {}
    pair_branch: dict = {}
    skipped_pairs = []
    fallback_pairs = []
    for (ea, eb) in pair_keys:
        ia, ib = entity_idx[ea], entity_idx[eb]
        valid = ~np.isnan(scores_2d[ia]) & ~np.isnan(scores_2d[ib])
        a_arr, b_arr = scores_2d[ia, valid], scores_2d[ib, valid]
        a_lab_arr, b_lab_arr = lab_matrix[ia, valid], lab_matrix[ib, valid]
        pair_arrays[(ea, eb)] = (a_arr, b_arr, a_lab_arr, b_lab_arr)

        n_overlap = int(np.sum(~np.isnan(a_lab_arr) & ~np.isnan(b_lab_arr)))
        n_a_only  = int(np.sum(~np.isnan(a_lab_arr)))
        n_b_only  = int(np.sum(~np.isnan(b_lab_arr)))
        if n_overlap >= 15:
            pair_branch[(ea, eb)] = "dispatch"
        elif n_a_only >= 15 and n_b_only >= 15:
            pair_branch[(ea, eb)] = "fallback"
            fallback_pairs.append((ea, eb))
        else:
            pair_branch[(ea, eb)] = "skip"
            skipped_pairs.append((ea, eb))

    # ── Simultaneous CIs: mirrors _simultaneous_cis_router's non-PPI tree
    # (Sidak for small N, joint bootstrap with an effective alpha ["boot"]
    # for larger N), not just a Bonferroni-only special case ─────────────────
    # prefer="auto" resolves via the SAME N-threshold table the non-PPI path
    # uses (resolve_auto_simultaneous_ci_method) -- "sidak" or "boot" -- so a
    # compound PPI+FWER comparison gets the same powerful construction the
    # non-PPI path would use at this N, not a silently-weaker fallback.
    #
    # Sidak (below) is a pure closed-form alpha adjustment: it widens
    # whichever alpha-parameterized PPI dispatch (_ppi_pairwise_dispatch) was
    # already going to be shown, exactly like the Bonferroni fallback does,
    # just with 1-(1-alpha)**(1/k) instead of alpha/k -- it needs no
    # resampling and applies to every pair regardless of branch (dispatch,
    # fallback, or skip).
    #
    # "boot" needs a joint distribution across pairs to derive an effective
    # alpha, ALWAYS applied uniformly regardless of which pairwise_method
    # resolved (never switching to a different, richer construction for one
    # particular method -- see the "max_t is never an auto-resolved outcome"
    # note below). _ppi_bootstrap_t_joint_stats builds exactly that: each
    # pair's PPI point estimate decomposes as f_unlab + rectifier and its SE
    # as sqrt(Var(unlab)/n_unlab + Var(rect)/n_lab) -- true for every PPI
    # paired method here (Tango's n_eff-shrinkage score interval, logit-t's
    # delta-method transform, ... all wrap a method-specific CI SHAPE around
    # this SAME additive point-estimate/variance structure), so its joint
    # critical value is a valid basis for widening any of them, not only
    # bootstrap_t's own construction. That critical value is translated into
    # an effective per-pair alpha (mirroring
    # _joint_bootstrap_scaled_simultaneous_cis's own c -> alpha_eff
    # conversion) and used to widen whichever CI shape ``pairwise_method``'s
    # own dispatch produces at that adjusted alpha -- the point estimate
    # stays whatever that method's dispatch produces, exactly like
    # Bonferroni/Sidak do, just with a joint (not per-pair-independent)
    # alpha adjustment. Requires every pair to share the same labeled-item
    # positions (enforced by _ppi_bootstrap_t_joint_stats); when that fails
    # (or any pair fell back to the unpaired/skip path), this falls back to
    # Bonferroni -- mirroring _simultaneous_cis_router's own
    # boot-degenerates-to-Bonferroni fallback (not Sidak, for consistency
    # with that precedent) rather than attempting a partial correction.
    #
    # "max_t" is a SEPARATE, non-tree construction (evalstats' own decision
    # tree has no max-T node -- Sidak/boot are the only two, exactly
    # mirroring the non-PPI path) and is reachable ONLY via an explicit
    # prefer="max_t" request, requiring pairwise_method=="bootstrap_t" (its
    # own natural Wald-type construction happens to match the joint
    # statistic's shape exactly, so it can be used directly instead of
    # wrapped -- yielding its own point estimate, CI, AND joint p-value).
    # It is never selected by prefer="auto", matching
    # _simultaneous_cis_router's identical convention on the non-PPI side.
    from evalstats.core.resampling import is_lopsided_binary
    _lopsided = data_kind == "binary" and is_lopsided_binary(scores_2d)
    # Per-entity item count (scores_2d.shape[1]), matching exactly what
    # _simultaneous_cis_router passes as `n` (scores.shape[1]) for the
    # non-PPI tree -- NOT n_all (the total across every entity), which
    # would apply the wrong threshold whenever there are more than one
    # compared entity. Shared by both the simultaneous-CI and p-value-
    # correction auto-resolutions below.
    _n_items_per_entity = scores_2d.shape[1]

    resolved_prefer = prefer
    if prefer == "auto":
        from evalstats.config import resolve_auto_simultaneous_ci_method
        resolved_prefer = resolve_auto_simultaneous_ci_method(
            data_kind, _n_items_per_entity, lopsided_binary=_lopsided,
        )  # "sidak" or "boot"

    # ── P-value correction: mirrors resolve_auto_pvalue_correction_method's
    # non-PPI tree (Shaffer's for small N, Romano-Wolf step-down for N>=30) --
    # see _ppi_romano_wolf_pvalues_from_joint_stats for how Romano-Wolf's
    # step-down generalizes to PPI's joint (unlabeled + rectifier) statistic,
    # reusing the SAME joint resample the CI-side "boot"/"max_t" would (no
    # separate bootstrap pass). Falls back to Shaffer's when the required
    # shared-labeled-item structure isn't available, mirroring "boot"'s own
    # degrade-to-Bonferroni precedent.
    resolved_correction = correction
    if correction == "auto":
        from evalstats.config import resolve_auto_pvalue_correction_method
        resolved_correction = resolve_auto_pvalue_correction_method(
            _n_items_per_entity, lopsided_binary=_lopsided,
        )  # "shaffer" or "romano_wolf"

    # Compute the shared joint resample ONCE if either the CI side ("boot"/
    # explicit max_t) or the p-value side (Romano-Wolf) needs it -- both
    # draw on the identical underlying bootstrap, so there's no reason to
    # resample twice even though they're conceptually independent knobs.
    _need_joint_for_ci = use_simultaneous and (
        resolved_prefer == "boot"
        or (resolved_prefer == "max_t" and pairwise_method == "bootstrap_t")
    )
    _need_joint_for_correction = n_pairs > 1 and resolved_correction == "romano_wolf"
    _attempt_joint = (
        (_need_joint_for_ci or _need_joint_for_correction)
        and not fallback_pairs
        and not skipped_pairs
    )
    joint = _ppi_bootstrap_t_joint_stats(
        scores_2d, lab_matrix, pair_keys, entity_idx, n_boot, rng,
    ) if _attempt_joint else None
    if _attempt_joint and joint is None:
        warnings.warn(
            "PPI alignment: could not build a joint bootstrap distribution "
            "(pairs don't share a common set of labeled items, or fewer "
            "than 15 shared labeled items). Falling back to Bonferroni for "
            "simultaneous CIs and/or Shaffer's for p-value correction, as "
            "applicable. Labeling the same items across every entity "
            "enables the more powerful joint-bootstrap constructions.",
            UserWarning,
            stacklevel=4,
        )

    used_max_t = joint is not None and _need_joint_for_ci and resolved_prefer == "max_t"
    used_boot = joint is not None and _need_joint_for_ci and resolved_prefer == "boot"
    boot_M_b = _M_b_from_T(joint[3], joint[2]) if used_boot else None

    used_romano_wolf = joint is not None and _need_joint_for_correction
    romano_wolf_pvalues = (
        _ppi_romano_wolf_pvalues_from_joint_stats(*joint, pair_keys)
        if used_romano_wolf else None
    )
    # Shaffer's is the fallback correction whenever Romano-Wolf was resolved
    # (by auto or explicitly) but couldn't actually run.
    effective_correction = "shaffer" if resolved_correction == "romano_wolf" and not used_romano_wolf else resolved_correction

    def _pair_alpha_for(level_alpha: float) -> float:
        """Resolve the per-pair alpha (or shared effective alpha) a
        dispatch call should use at a given confidence level, for both the
        headline CI and every gradient-band CI -- keeping them consistent
        with whichever simultaneous-CI construction is actually in effect."""
        if not use_simultaneous:
            return level_alpha
        if used_boot:
            return _ppi_alpha_eff_from_M_b(boot_M_b, 1.0 - level_alpha)
        if resolved_prefer == "sidak":
            return 1.0 - (1.0 - level_alpha) ** (1.0 / n_pairs)
        return level_alpha / n_pairs  # Bonferroni (explicit, or any other fallback)

    pair_alpha = _pair_alpha_for(alpha)

    final_diffs = np.empty(n_pairs, dtype=float)
    pair_ci_lo  = np.empty(n_pairs, dtype=float)
    pair_ci_hi  = np.empty(n_pairs, dtype=float)
    pair_pvals  = np.empty(n_pairs, dtype=float)
    pair_multi_ci: dict = {a: {} for a in GRADIENT_CI_ALPHAS}
    pair_test_method: dict = {}
    pair_wilcoxon_p: dict = {}

    from evalstats.tests import _ppi_paired_arrays as _ppi_wilcoxon_arrays
    from evalstats.ppi import paired_walsh_midrank_theta as _wilcoxon_statistic

    for k, (ea, eb) in enumerate(pair_keys):
        pr = bundle.pairwise.results[(ea, eb)]
        branch = pair_branch[(ea, eb)]
        a_arr, b_arr, a_lab_arr, b_lab_arr = pair_arrays[(ea, eb)]

        if branch == "dispatch":
            pair_test_method[(ea, eb)] = f"PPI {pairwise_method}"
            # Companion PPI-corrected Wilcoxon signed-rank p-value (shown when
            # pairwise_test="wilcoxon"), computed the same way es.tests.wilcoxon()
            # does with x_lab/y_lab — independent of whichever method drove the
            # headline point_diff/p_value above, so always computed regardless
            # of the max-T shortcut below.
            pair_wilcoxon_p[(ea, eb)] = _ppi_wilcoxon_arrays(
                a_arr, b_arr, a_lab_arr, b_lab_arr, _wilcoxon_statistic, pair_alpha, n_boot, rng,
                rectifier_func=_wilcoxon_statistic,
            ).p_value

            if used_max_t:
                # point estimate/CI/p-value are set below from the joint
                # bootstrap (computed once, shared across every pair) —
                # skip the redundant per-alpha dispatch calls that would
                # just be overwritten.
                continue

            dispatch = lambda a_, n_boot_, rng_: _ppi_pairwise_dispatch(
                pairwise_method, a_arr, b_arr, a_lab_arr, b_lab_arr, a_, n_boot_, rng_,
                ppi_score_range,
            )
        elif branch == "fallback":
            # Not enough items are labeled for *both* entities to run the
            # paired PPI method, but each entity individually has enough of
            # its own labels — fall back to independent-groups PPI (see
            # _ppi_pairwise_unpaired_fallback), which only needs that.
            dispatch = lambda a_, n_boot_, rng_: _ppi_pairwise_unpaired_fallback(
                a_arr, b_arr, a_lab_arr, b_lab_arr, a_, n_boot_, rng_
            )
            pair_test_method[(ea, eb)] = "PPI mean-diff (unpaired fallback, insufficient item overlap)"
            # No validated PPI-corrected Wilcoxon signed-rank for the unpaired
            # fallback case (it's an inherently paired test) — show as
            # unavailable rather than a stale/uncorrected number.
            pair_wilcoxon_p[(ea, eb)] = None
        else:  # "skip"
            final_diffs[k] = pr.point_diff
            pair_ci_lo[k]  = pr.ci_low
            pair_ci_hi[k]  = pr.ci_high
            pair_pvals[k]  = pr.p_value
            pair_test_method[(ea, eb)] = pr.test_method
            pair_wilcoxon_p[(ea, eb)] = pr.wilcoxon_p
            for a in GRADIENT_CI_ALPHAS:
                pair_multi_ci[a][(ea, eb)] = pr.multi_ci.get(a, (pr.ci_low, pr.ci_high)) if pr.multi_ci else (pr.ci_low, pr.ci_high)
            continue

        res = dispatch(pair_alpha, n_boot, rng)
        final_diffs[k] = res.estimate
        pair_ci_lo[k]  = res.ci_low
        pair_ci_hi[k]  = res.ci_high
        pair_pvals[k]  = res.p_value if res.p_value is not None else 1.0

        for a in GRADIENT_CI_ALPHAS:
            g_alpha = _pair_alpha_for(a)
            g = dispatch(g_alpha, n_boot, rng)
            pair_multi_ci[a][(ea, eb)] = (g.ci_low, g.ci_high)

    if fallback_pairs:
        warnings.warn(
            f"PPI alignment: the following pairs don't have enough commonly-labeled "
            f"items for a paired PPI correction and used an independent-groups "
            f"fallback instead: {fallback_pairs}. Consider labeling the same items "
            "for every entity to enable the (more efficient) paired correction.",
            UserWarning,
            stacklevel=4,
        )
    if skipped_pairs:
        warnings.warn(
            f"PPI alignment: the following pairs have fewer than 15 labeled items "
            f"for either entity and keep their uncorrected estimate: {skipped_pairs}. "
            "Consider expanding the alignment set to cover more entities.",
            UserWarning,
            stacklevel=4,
        )

    # Multiple-comparison correction on marginal p-values (unaffected by the
    # simultaneous-CI adjustment above, which only widens CIs). Skipped for
    # pair_pvals when max-T or Romano-Wolf already applies below/here --
    # both are already family-wise controlled via their own joint null and
    # would just be overwritten, exactly mirroring all_pairwise()'s non-PPI
    # convention where a max-T/Romano-Wolf p-value supersedes rather than
    # stacks with correct_pvalues().
    if effective_correction != "none" and n_pairs > 1:
        if used_romano_wolf:
            for k, key in enumerate(pair_keys):
                pair_pvals[k] = romano_wolf_pvalues[key]
        elif not used_max_t:
            pair_pvals = correct_pvalues(pair_pvals, effective_correction, n_groups=len(labels))

        # Companion Wilcoxon p-values always get their own correction as
        # their own family, regardless of max-T/Romano-Wolf — matching
        # all_pairwise()'s convention, which never applies either to
        # wilcoxon_p (see its docstring). Romano-Wolf's joint construction
        # is specific to the paired-mean/PPI estimand (point_ests/obs_se
        # above) and has no Wilcoxon-signed-rank-compatible form, so
        # Shaffer's (with Holm as the same subset-safe fallback the non-PPI
        # path uses) substitutes whenever the primary correction is
        # Romano-Wolf -- mirroring the non-PPI docstring's own statement
        # that Wilcoxon "everywhere...except when Romano-Wolf is what
        # resolved" note.
        wsr_keys = [key for key in pair_keys if pair_wilcoxon_p[key] is not None]
        if len(wsr_keys) > 1:
            wsr_vals = np.array([pair_wilcoxon_p[key] for key in wsr_keys], dtype=float)
            _wsr_base = "shaffer" if effective_correction == "romano_wolf" else effective_correction
            # Shaffer's needs the complete n_groups*(n_groups-1)/2 all-pairs
            # set; wsr_keys can be a strict subset (e.g. no validated PPI
            # Wilcoxon for an unpaired-fallback pair -- see above). Holm has
            # no such requirement and is still FWER-valid for any subset.
            _wsr_correction = (
                "holm" if _wsr_base == "shaffer" and len(wsr_keys) != len(labels) * (len(labels) - 1) // 2
                else _wsr_base
            )
            wsr_adj = correct_pvalues(wsr_vals, _wsr_correction, n_groups=len(labels))
            for key, adj_p in zip(wsr_keys, wsr_adj):
                pair_wilcoxon_p[key] = float(adj_p)

    if used_max_t:
        point_ests_j, obs_se_j, valid_pairs_j, T_j, t_obs_j = joint
        M_b = _M_b_from_T(T_j, valid_pairs_j)
        headline_ci, headline_p = _max_t_from_joint_stats(
            point_ests_j, obs_se_j, valid_pairs_j, M_b, t_obs_j, pair_keys, 1.0 - alpha,
        )
        for k, key in enumerate(pair_keys):
            final_diffs[k] = point_ests_j[k]
            pair_ci_lo[k], pair_ci_hi[k] = headline_ci[key]
            pair_pvals[k] = headline_p[key]
        for a in GRADIENT_CI_ALPHAS:
            g_ci, _ = _max_t_from_joint_stats(
                point_ests_j, obs_se_j, valid_pairs_j, M_b, t_obs_j, pair_keys, 1.0 - a,
            )
            pair_multi_ci[a] = g_ci

    # ── Recompute rank distribution (P(Best)/E[Rank]) under PPI ───────────────
    # bundle.rank_dist was built from the raw, uncorrected LLM scores and does
    # not reflect the correction above — without this, P(Best)/E[Rank] would
    # silently stay frozen at pre-correction values even as means/CIs shift.
    from evalstats.core.ranking import LazyRankDistribution, ppi_bootstrap_ranks
    bundle.rank_dist = LazyRankDistribution(
        labels, n_boot,
        lambda _rng: ppi_bootstrap_ranks(scores_2d, lab_matrix, labels, n_boot, _rng),
        rng=rng,
    )
    bundle.ppi_applied = True
    bundle.alignment_result = alignment_result

    # ── Override _analysis in-place ───────────────────────────────────────────
    bundle.robustness.mean     = final_means
    bundle.robustness.ci_low   = final_ci_low
    bundle.robustness.ci_high  = final_ci_high
    bundle.robustness.multi_ci = final_multi_ci

    for k, key in enumerate(pair_keys):
        pr = bundle.pairwise.results[key]
        pr.point_diff = float(final_diffs[k])
        pr.p_value    = float(pair_pvals[k])
        pr.ci_low     = float(pair_ci_lo[k])
        pr.ci_high    = float(pair_ci_hi[k])
        pr.multi_ci   = {a: pair_multi_ci[a][key] for a in GRADIENT_CI_ALPHAS}
        pr.test_method = pair_test_method[key]
        wp = pair_wilcoxon_p[key]
        pr.wilcoxon_p = float(wp) if wp is not None else None

    # ── Recompute the Friedman omnibus test (if requested via omnibus=True) ───
    # bundle.pairwise.friedman is built from the raw, uncorrected LLM scores;
    # without this it would silently stay frozen (same χ²/p) even though the
    # means/CIs/pairwise p-values above just changed under the correction.
    if bundle.pairwise.friedman is not None and n_entities >= 3:
        from evalstats.tests import _ppi_friedman_p_value
        groups = [scores_2d[i] for i in range(n_entities)]
        groups_lab = [lab_matrix[i] for i in range(n_entities)]
        corrected_friedman_p = _ppi_friedman_p_value(groups, groups_lab, n_entities)
        if corrected_friedman_p is not None:
            bundle.pairwise.friedman.p_value = float(corrected_friedman_p)

    # Update bundle method metadata so summary() headers reflect the PPI method.
    bundle.resolved_method = pairwise_method
    bundle.resolved_ci_method = robustness_method
    if used_max_t:
        _sim_ci_label = "max_t"
    elif used_boot:
        _sim_ci_label = "boot"
    elif use_simultaneous:
        _sim_ci_label = "sidak" if resolved_prefer == "sidak" else "bonferroni"
    else:
        _sim_ci_label = None
    bundle.pairwise.simultaneous_ci_method = _sim_ci_label
    # correction_method must reflect the correction ACTUALLY applied to the
    # final, PPI-corrected p-values -- previously left untouched here, so it
    # silently kept whatever the initial non-PPI analyze() call had set
    # (e.g. "romano_wolf" from the raw-score pass, even when PPI's own
    # correction fell back to "shaffer"), which PairwiseMatrix.summary()
    # displays via each pair's own .summary(correction=...).
    bundle.pairwise.correction_method = (
        effective_correction if (effective_correction != "none" and n_pairs > 1) else None
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    cr._variance_components = {
        "method": "ppi",
        "n_all": n_all,
        "n_lab": n_lab,
        "n_boot": n_boot,
        "pairwise_method": pairwise_method,
        "robustness_method": robustness_method,
        "data_kind": data_kind,
        "entities": {
            lbl: {
                "n_labeled": int(np.sum(~np.isnan(lab_matrix[i]))),
                "llm_mean": float(np.nanmean(scores_2d[i])) if not np.all(np.isnan(scores_2d[i])) else None,
                "rectifier": entity_rectifier[lbl],
                "ppi_mean": float(final_means[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "note": (
            "Prediction-Powered Inference (PPI), dispatched per resolved "
            f"method (pairwise={pairwise_method!r}, robustness={robustness_method!r}). "
            "Items are paired by shared position in the benchmark's item-aligned "
            "score matrix, matching evalstats' paired-by-input design."
        ),
    }


def _run_judge_alignment_if_needed(
    cr: "ComparisonResult",
    *,
    alignment,
    metric_col: str,
    n_mc: int,
    alpha: float,
    ci_level: float,
    engine_kwargs: dict,
    df: pd.DataFrame,
    factor_col: str,
    item_col: str,
    run_col: Optional[str],
) -> None:
    """Validate alignment= and dispatch to _run_alignment_ppi when appropriate."""
    if alignment is None:
        return
    if not isinstance(alignment, dict):
        warnings.warn(
            "alignment= must be a dict mapping metric column names to AlignmentResult objects. "
            "Example: alignment={'score': my_alignment_result}. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return
    ar = alignment.get(metric_col)
    if ar is None:
        warnings.warn(
            f"alignment= dict has no entry for metric column '{metric_col}'. "
            f"Keys present: {list(alignment.keys())}. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return
    if not isinstance(cr._analysis, AnalysisBundle):
        warnings.warn(
            "PPI alignment correction is not yet supported for multi-model or factorial "
            "analyses. alignment= will be ignored for this comparison.",
            UserWarning,
            stacklevel=4,
        )
        return
    _run_alignment_ppi(
        cr,
        df=df,
        metric_col=metric_col,
        factor_col=factor_col,
        item_col=item_col,
        alignment_result=ar,
        alpha=alpha,
        n_boot=max(n_mc, 1000),
        # Default "auto": resolves to Romano-Wolf step-down at N>=30 (see
        # _ppi_romano_wolf_pvalues_from_joint_stats), falling back to
        # Shaffer's below that threshold or when the required shared-
        # labeled-item structure isn't available -- mirroring the non-PPI
        # tree's own auto default exactly, now that a PPI-generalized
        # Romano-Wolf exists. Validated across a 9-condition grid (k=3-5
        # arms, N=50-200, label fractions 20-40%, binary data): worst-case
        # FWER 0.067 vs nominal 0.05 (within normal MC noise, no systematic
        # inflation), and power at or above Shaffer's in every condition
        # tested (0/9 regressions) -- see tests/test_compound_ppi_fwer.py's
        # TestRomanoWolfCalibration for the pytest-scale version of this
        # check.
        correction=engine_kwargs.get("correction", "auto"),
        method=engine_kwargs.get("method", "auto"),
        rng=engine_kwargs.get("rng"),
    )


def _run_pareto_if_needed(
    cr: "ComparisonResult",
    *,
    secondary_metric,
    df: pd.DataFrame,
    factor_col: str,
    item_col: str,
    alpha: float,
    n_boot: int,
    rng,
) -> None:
    """Run uncertainty-aware Pareto-front analysis and store it on *cr*, if
    ``secondary_metric=`` was passed.

    Mirrors ``_run_judge_alignment_if_needed``'s validate-and-dispatch shape:
    warns and no-ops on a malformed ``secondary_metric=``, and is only supported for
    a single-factor result (a plain ``AnalysisBundle`` -- multi-model and
    factorial results are not yet supported, same restriction as
    ``alignment=``).
    """
    if secondary_metric is None:
        return
    if not isinstance(secondary_metric, dict):
        warnings.warn(
            "secondary_metric= must be a dict mapping a metric column name to "
            "'min' or 'max', e.g. secondary_metric={'latency_ms': 'min'}. "
            "secondary_metric= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return
    if len(secondary_metric) != 1:
        raise NotImplementedError(
            f"secondary_metric= currently supports exactly one secondary metric "
            f"(bivariate Pareto fronts only); got {len(secondary_metric)}: "
            f"{list(secondary_metric.keys())}. N-way Pareto fronts are not yet "
            "implemented."
        )
    (secondary_col, direction), = secondary_metric.items()
    if direction not in ("min", "max"):
        raise ValueError(
            f"secondary_metric={{'{secondary_col}': {direction!r}}} -- direction "
            "must be 'min' or 'max'."
        )
    if secondary_col not in df.columns:
        raise EvalLoadError(
            f"secondary metric column '{secondary_col}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    if not isinstance(cr._analysis, AnalysisBundle):
        # cr._primary_bundle() always unwraps a MultiModelBundle down to some
        # single AnalysisBundle view (e.g. .model_level) -- checking that
        # would never actually catch the multi-model case. Must check
        # cr._analysis itself, same as _run_judge_alignment_if_needed does.
        warnings.warn(
            "Pareto-front analysis (secondary_metric=) is not yet supported for "
            "multi-model or factorial analyses. secondary_metric= will be ignored "
            "for this comparison.",
            UserWarning,
            stacklevel=4,
        )
        return
    bundle = cr._primary_bundle()
    if bundle.benchmark.is_seeded:
        raise ValueError(
            "Pareto-front analysis (secondary_metric=) does not yet support seeded "
            "benchmarks (R >= 3 repeated runs). Aggregate runs to a single "
            "score per (template, input) cell before passing secondary_metric=."
        )

    from evalstats.core.pareto import pareto_bootstrap, classify_pareto_status, orient_higher_is_better

    labels = list(bundle.benchmark.template_labels)
    input_labels = list(bundle.benchmark.input_labels)
    n_entities = len(labels)
    n_items = len(input_labels)
    entity_idx = {e: i for i, e in enumerate(labels)}
    item_idx = {it: j for j, it in enumerate(input_labels)}

    # Primary metric's item-aligned score matrix (already computed by the
    # main analysis) -- averaged over runs/evaluators the same way PPI's
    # scores_2d is, so both metrics share the exact same (entity, item) grid.
    scores_primary = bundle.benchmark.get_2d_scores()

    # Secondary metric's item-aligned score matrix, built directly from the
    # filtered dataframe the same way _run_alignment_ppi builds lab_matrix --
    # more robust than re-deriving via from_dataframe() a second time, since
    # it guarantees identical (entity, item) index alignment with scores_primary.
    scores_secondary = np.full((n_entities, n_items), np.nan)
    sec_rows = df[[factor_col, item_col, secondary_col]].dropna(subset=[secondary_col])
    for e_val, it_val, s_val in zip(
        sec_rows[factor_col].astype(str).to_numpy(),
        sec_rows[item_col].astype(str).to_numpy(),
        sec_rows[secondary_col].to_numpy(dtype=float),
    ):
        ei = entity_idx.get(e_val)
        ij = item_idx.get(it_val)
        if ei is not None and ij is not None:
            scores_secondary[ei, ij] = s_val

    if np.any(np.isnan(scores_secondary)):
        n_missing = int(np.sum(np.isnan(scores_secondary)))
        raise ValueError(
            f"secondary_metric='{secondary_col}' has {n_missing} missing (entity, item) "
            f"cell(s) out of {n_entities * n_items} -- Pareto-front analysis "
            "currently requires a complete design (every entity scored on "
            "every item for the secondary metric too)."
        )

    scores_secondary_oriented = orient_higher_is_better(scores_secondary, direction)
    rng_gen = np.random.default_rng(rng)

    result = pareto_bootstrap(
        scores_primary, scores_secondary_oriented, labels,
        n_bootstrap=n_boot, rng=rng_gen,
        # Retained for plot_pareto_tradeoff()'s bootstrap point cloud, so it
        # draws from the exact same replicates the calibrated status/P(Pareto-
        # optimal) numbers come from, rather than a second independent
        # bootstrap. Cheap: O(N x n_bootstrap) floats, not O(N^2).
        return_replicates=True,
    )
    statuses = classify_pareto_status(result, alpha=alpha)

    # Secondary metric's own calibrated marginal CI, in its real (un-oriented)
    # units -- same auto data-kind/N routing analyze() uses for the primary
    # metric (see router.py's _analyze_single), not a cheap percentile CI off
    # the joint dominance bootstrap: that bootstrap is built for the
    # dominance question, and reusing it here would quietly ship an
    # uncalibrated shortcut for the one number this library is otherwise
    # careful never to show uncalibrated.
    from evalstats.core.resampling import is_binary_scores, resolve_score_bounds
    from evalstats.core.variance import robustness_metrics
    from evalstats.config import resolve_auto_analyze_methods

    if is_binary_scores(scores_secondary):
        sec_data_kind = "binary"
        sec_score_range = None
    else:
        sec_score_range = resolve_score_bounds(scores_secondary, None, stacklevel=5)
        sec_data_kind = "bounded_01" if sec_score_range is not None else "unbounded"
    _, sec_robustness_method = resolve_auto_analyze_methods(sec_data_kind, n_items, seeded=False)
    secondary_robustness = robustness_metrics(
        scores_secondary, labels,
        n_bootstrap=n_boot, rng=rng_gen, alpha=alpha,
        marginal_method=sec_robustness_method,
        score_range=sec_score_range,
    )

    cr._pareto = {
        "secondary_metric": secondary_col,
        "direction": direction,
        "result": result,
        "statuses": statuses,
        "primary_robustness": bundle.robustness,
        "secondary_robustness": secondary_robustness,
    }
