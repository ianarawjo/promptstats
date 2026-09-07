"""Linear Mixed Model (LMM) analysis — statsmodels (default) or pymer4/lme4 backend.

Fits the one-way mixed model::

    score ~ template + (1 | input)

``template`` is a fixed effect (the quantity we care about); ``input``
identity is a random intercept that absorbs between-input variance.  This
is the correct model for a paired benchmark design where every template is
evaluated on every input (complete block design).

When ``R >= 3`` runs are available, scores are first collapsed to per-input
cell means before fitting (averaging over runs).  The between-run (seed)
variance decomposition is still reported separately via the existing
``SeedVarianceResult``.

Outputs are mapped to the same result types as the bootstrap path
(``PairwiseMatrix``, ``MeanAdvantageResult``, ``RankDistribution``), so
consumers of ``AnalysisBundle`` do not need to know which method was used.
The one addition is ``LMMInfo``, stored on ``AnalysisBundle.lmm_info``,
which exposes the ICC and variance components from the fitted model.

Requirements
------------
* ``pymer4 >= 0.9`` (``pip install pymer4``)
* ``pyarrow`` (needed by pymer4's polars→pandas bridge: ``pip install pyarrow``)
* R with the ``lme4``, ``lmerTest``, ``emmeans``, ``broom.mixed``, and
  ``parameters`` packages installed

When to prefer LMM over bootstrap
-----------------------------------
* M inputs < ~15  — bootstrap CIs are unstable; LMM borrows strength
  from the model structure and gives better-calibrated CIs.
* You want a clean ICC decomposition of between-input vs. residual variance.
* Score distributions are sufficiently well-behaved (roughly Gaussian
  conditional on the random effect).

Limitations (Phase 1)
----------------------
* Template labels must not contain the substring `` - `` (space-dash-space),
  as this is used to parse emmeans contrast strings.
* Multi-model analysis (``MultiModelBenchmark``) is supported: LMM is run
  independently per model, exactly like the bootstrap path.
* The ``method='lmm'`` option is not compatible with ``method='bca'`` or
  ``method='auto'``; it must be specified explicitly.

Implementation note (pymer4 0.9)
---------------------------------
pymer4 0.9+ uses **Polars** DataFrames internally and dropped the old
pandas-based API.  The key differences from pymer4 ≤ 0.8 are:

* Data passed to ``lmer()`` must be a Polars DataFrame (we construct one
  from the numpy score matrix).
* ``model.set_factors({"template": labels})`` must be called *before*
  ``model.fit()`` so that pymer4 tracks ``template`` as a categorical
  predictor and routes ``model.emmeans()`` to marginal means rather than
  marginal trends.
* Fixed effects live in ``model.result_fit`` (Polars) instead of the
  old ``model.coefs`` (pandas).  Column names are ``term``, ``estimate``,
  ``std_error``, ``conf_low``, ``conf_high``, ``t_stat``, ``df``,
  ``p_value``.
* Random-effect variance components live in ``model.ranef_var`` (Polars)
  with columns ``group``, ``term``, ``estimate``; **values are SDs not
  variances** (broom.mixed returns ``sd__*`` terms by default).
* There is no ``model.vcov``; we call R's ``stats::vcov()`` directly.
* Convergence is reported via ``model.convergence_status`` (string).
* Pairwise contrasts come from ``model.emmeans("template",
  contrasts="pairwise", p_adjust="none")``, returning a Polars DataFrame
  with columns ``contrast``, ``estimate``, ``SE``, ``df``, ``t_ratio``,
  ``p_value``.  Contrast labels are plain "A - B", not "templateA - templateB".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np
import scipy.stats

from .paired import PairedDiffResult, PairwiseMatrix
from .ranking import RankDistribution
from .variance import RobustnessResult, SeedVarianceResult, robustness_metrics, seed_variance_decomposition
from .stats_utils import correct_pvalues

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# LMM diagnostics data class
# ---------------------------------------------------------------------------

@dataclass
class LMMInfo:
    """Variance components and fit diagnostics from the fitted LMM.

    Attributes
    ----------
    icc : float
        Intraclass correlation coefficient: σ²_input / (σ²_input + σ²_resid).
        Fraction of total score variance explained by between-input differences.
        High ICC (> 0.5) means inputs are very heterogeneous relative to
        within-cell noise; the paired design is especially valuable here.
    sigma_input : float
        Estimated standard deviation of the input random effect (between-input SD).
    sigma_resid : float
        Estimated residual standard deviation (within-cell SD).
    n_obs : int
        Number of observations used to fit the model (N_templates × M_inputs,
        minus any missing cells).
    formula : str
        The model formula used.
    converged : bool
        Whether lme4 reported a successful convergence.
    """

    icc: float
    sigma_input: float
    sigma_resid: float
    n_obs: int
    formula: str
    converged: bool = True


@dataclass
class FactorialLMMInfo:
    """Variance components and fit diagnostics from a factorial LMM.

    Extends the one-factor design to support multiple fixed-effect factors
    (and their interactions), as specified by ``BenchmarkResult.template_factors``.

    The model formula is ``score ~ F1 * F2 * ... + (1 | input)``, where the
    ``*`` operator expands to all main effects and interactions.

    Attributes
    ----------
    icc : float
        Intraclass correlation: σ²_input / (σ²_input + σ²_resid).
    sigma_input : float
        Estimated SD of the input random effect.
    sigma_resid : float
        Estimated residual SD.
    n_obs : int
        Number of observations used to fit the model.
    formula : str
        Full model formula (e.g. ``'score ~ C(persona) * C(few_shots) + (1|input)'``).
    converged : bool
        Whether the optimizer reported successful convergence.
    factor_names : list[str]
        Factor column names from ``BenchmarkResult.template_factors``.
    factor_tests : pd.DataFrame
        Wald test results per model term (main effects + interactions).
        Columns: ``term``, ``statistic``, ``df``, ``p_value``.
    marginal_means : dict[str, pd.DataFrame]
        Estimated marginal means per factor (averaged equally over all
        levels of other factors). Keys are factor names; each value is a
        DataFrame with columns ``level``, ``mean``, ``se``, ``ci_low``,
        ``ci_high``.
    """

    icc: float
    sigma_input: float
    sigma_resid: float
    n_obs: int
    formula: str
    factor_names: list[str]
    factor_tests: "pd.DataFrame"
    marginal_means: dict[str, "pd.DataFrame"]
    converged: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_pymer4() -> Any:
    """Import and return ``pymer4.models.lmer``, or raise a helpful ImportError.

    pymer4 0.9+ uses Polars internally.  Its Polars→pandas bridge calls
    ``polars.DataFrame.to_pandas()`` which requires ``pyarrow``.  We check
    for both dependencies here and give actionable error messages.
    """
    try:
        from pymer4.models import lmer  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "pymer4 is required for method='lmm'. Install it with:\n"
            "    pip install pymer4 pyarrow\n\n"
            "pymer4 also requires R with the lme4, lmerTest, emmeans, broom.mixed,\n"
            "and parameters packages:\n"
            "    install.packages(c('lme4', 'lmerTest', 'emmeans', 'broom.mixed',\n"
            "                       'parameters', 'performance'))\n\n"
            "See https://eshinjolly.com/pymer4/ for full setup instructions."
        ) from None

    try:
        import pyarrow  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyarrow is required by pymer4 0.9+ for its Polars↔pandas bridge. "
            "Install it with:\n    pip install pyarrow"
        ) from None

    return lmer


def _col_pl(df: Any, candidates: list[str]) -> str:
    """Return the first column name from *candidates* that exists in *df*.

    Handles minor API differences across pymer4 / R package versions.
    Raises ``KeyError`` with a helpful message if none are found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find any of {candidates} in DataFrame columns {list(df.columns)}. "
        "This may indicate a pymer4 or R package version incompatibility. "
        "Please open an issue with the output of `model.result_fit` / `model.ranef_var`."
    )


def _scores_to_long_df(
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
) -> Any:
    """Convert an ``(N, M)`` cell-mean score matrix to a long-form Polars DataFrame.

    Returns a Polars DataFrame with columns ``'template'``, ``'input'``,
    ``'score'``.  Missing (NaN) cells are dropped so lme4 receives only
    observed observations.  The ``'template'`` column uses ``pl.Enum`` with
    categories in the order given by *template_labels* so that lme4's
    treatment coding uses ``template_labels[0]`` as the reference level.
    """
    import polars as pl

    N, M = scores_2d.shape
    templates = np.repeat(template_labels, M).tolist()
    inputs = np.tile(input_labels, N).tolist()
    scores_flat = scores_2d.ravel().tolist()

    df = pl.DataFrame({"template": templates, "input": inputs, "score": scores_flat})
    # Drop rows with missing scores so lme4 receives only observed observations.
    df = df.filter(pl.col("score").is_not_nan())
    # Explicit Enum category order → first label is the reference in treatment coding.
    df = df.with_columns(pl.col("template").cast(pl.Enum(template_labels)))
    return df


def _fit_lmm(df: Any, lmer: Any, template_labels: list[str]) -> Any:
    """Fit ``score ~ template + (1|input)`` with Satterthwaite DFs.

    Uses REML estimation (better for variance components).  We call
    ``model.set_factors`` *before* ``model.fit`` so that pymer4 correctly
    identifies ``template`` as a categorical predictor when routing
    ``model.emmeans()`` (otherwise it dispatches to ``emtrends``).
    """
    model = lmer("score ~ template + (1|input)", data=df)
    # Register template as a factor with explicit level ordering so that:
    #   (a) emmeans dispatches to marginal means, not marginal trends, and
    #   (b) the contrast coding (treatment, reference = template_labels[0]) is
    #       set explicitly rather than inferred from the Polars Enum sort order.
    model.set_factors({"template": template_labels})
    model.fit()
    return model


def _get_vcov(model: Any) -> np.ndarray:
    """Extract the fixed-effects variance–covariance matrix as a numpy array.

    pymer4 0.9 does not expose ``model.vcov``; we call R's ``stats::vcov()``
    directly and convert the resulting ``dpoMatrix`` to a plain numpy 2D array
    via ``base::as.matrix()``.
    """
    from rpy2.robjects.packages import importr
    base_r = importr("base")
    stats_r = importr("stats")
    vcov_r = stats_r.vcov(model.r_model)
    mat_r = base_r.as_matrix(vcov_r)
    return np.array(mat_r)   # (N, N)


def _extract_template_means(model: Any, labels: list[str]) -> np.ndarray:
    """Compute fitted marginal means for each template from treatment-coded LMM.

    With R's default treatment coding the first category is the reference:

    * μ₀  = intercept
    * μᵢ  = intercept + β_i   for i > 0

    Returns shape ``(N,)``.
    """
    rf = model.result_fit
    est_col = _col_pl(rf, ["estimate", "Estimate", "coefficient", "Coefficient"])
    betas = rf[est_col].to_numpy()   # (N,): [intercept, β₁, …, β_{N-1}]

    N = len(labels)
    means = np.empty(N)
    means[0] = betas[0]
    means[1:] = betas[0] + betas[1:]
    return means


# ---------------------------------------------------------------------------
# Shared low-level helpers (backend-agnostic)
# ---------------------------------------------------------------------------

def _apply_pvalue_correction(
    results: dict,
    pairs: list,
    correction: str,
    *,
    n_groups: Optional[int] = None,
) -> str:
    """Apply multiple-comparisons correction to a ``results`` dict in-place.

    Parameters
    ----------
    results : dict[tuple[str, str], PairedDiffResult]
        Mapping from ``(label_a, label_b)`` to a ``PairedDiffResult``.
        Updated in-place: corrected p-values and an updated ``test_method``
        string are written back for each pair.
    pairs : list[tuple[str, str]]
        Ordered list of pairs in the same order as ``results`` was populated,
        used to extract and replace p-values in a consistent order.
    correction : str
        Correction method accepted by :func:`correct_pvalues` (e.g.
        ``'holm'``, ``'bonferroni'``, ``'fdr_bh'``, ``'shaffer'``,
        ``'none'``), or ``'auto'``. ``'auto'`` always resolves to
        ``'shaffer'`` here -- unlike :func:`~evalstats.core.paired.all_pairwise`'s
        N-threshold-routed "auto" (which can pick Romano-Wolf step-down at
        N>=30), the LMM path's Wald p-values have no raw bootstrap
        distribution to build a step-down from, so this always picks the
        one auto-table method that doesn't need one -- still a real
        improvement on the previous unconditional ``'fdr_bh'`` (FDR, not
        FWER) default. When ``'none'`` or ``len(pairs) <= 1`` the function
        is a no-op (returns the resolved method without correcting anything).
    n_groups : int, optional
        Number of arms/groups being compared -- forwarded to
        :func:`correct_pvalues` (required for ``'shaffer'``).

    Returns
    -------
    str
        The resolved correction method actually applied (e.g. ``'auto'``
        resolves to ``'shaffer'``) -- callers should store this, not the
        original *correction* argument, in ``PairwiseMatrix.correction_method``.
    """
    resolved = "shaffer" if correction == "auto" else correction
    if resolved == "none" or len(pairs) <= 1:
        return resolved
    p_values = np.array([results[p].p_value for p in pairs])
    adjusted = correct_pvalues(p_values, resolved, n_groups=n_groups)
    for pair, adj_p in zip(pairs, adjusted):
        r = results[pair]
        results[pair] = PairedDiffResult(
            template_a=r.template_a,
            template_b=r.template_b,
            point_diff=r.point_diff,
            std_diff=r.std_diff,
            ci_low=r.ci_low,
            ci_high=r.ci_high,
            p_value=float(adj_p),
            test_method=f"{r.test_method} ({resolved}-corrected)",
            n_inputs=r.n_inputs,
            per_input_diffs=r.per_input_diffs,
            n_runs=r.n_runs,
            statistic=r.statistic,
        )
    return resolved


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def _lmm_to_pairwise(
    model: Any,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    correction: str,
) -> PairwiseMatrix:
    """Build a ``PairwiseMatrix`` from LMM emmeans pairwise contrasts.

    Uses ``model.emmeans("template", contrasts="pairwise", p_adjust="none")``
    which calls R's ``emmeans::contrast()`` under the hood, giving Wald CIs
    and Satterthwaite degrees of freedom for each pairwise contrast.

    Multiple-comparisons correction is applied afterwards using the same
    ``correct_pvalues()`` function used by the bootstrap path.

    pymer4 0.9 note: emmeans returns a Polars DataFrame with columns
    ``contrast``, ``estimate``, ``SE``, ``df``, ``t_ratio``, ``p_value``.
    Contrast labels are plain "A - B" (not "templateA - templateB").
    """
    alpha = 1 - ci

    contrasts_df = model.emmeans("template", contrasts="pairwise", p_adjust="none")

    # Defensive column name lookup — names may vary across pymer4 / R versions.
    contrast_col = _col_pl(contrasts_df, ["contrast", "Contrast"])
    est_col      = _col_pl(contrasts_df, ["estimate", "Estimate"])
    se_col       = _col_pl(contrasts_df, ["SE", "se", "std_error", "std.error"])
    df_col       = _col_pl(contrasts_df, ["df", "DF", "Df"])
    pval_col     = _col_pl(contrasts_df, ["p_value", "p.value", "P-val", "P.Value"])

    results: dict[tuple[str, str], PairedDiffResult] = {}
    pairs: list[tuple[str, str]] = []

    for row in contrasts_df.iter_rows(named=True):
        contrast_str = str(row[contrast_col])

        # pymer4 0.9 emmeans: labels are plain "A - B" (no "template" prefix).
        # Guard against older versions that might prefix with the variable name.
        for label in labels:
            contrast_str = contrast_str.replace(f"template{label}", label)
            contrast_str = contrast_str.replace(f"template {label}", label)

        if " - " not in contrast_str:
            warnings.warn(
                f"Unexpected contrast string from pymer4/emmeans: '{row[contrast_col]}'. "
                "Skipping. Check that template labels do not contain ' - '.",
                UserWarning,
                stacklevel=4,
            )
            continue

        label_a, label_b = (s.strip() for s in contrast_str.split(" - ", 1))

        if label_a not in labels or label_b not in labels:
            warnings.warn(
                f"Contrast '{contrast_str}' could not be matched to known labels "
                f"{labels}. Skipping.",
                UserWarning,
                stacklevel=4,
            )
            continue

        estimate = float(row[est_col])
        se       = float(row[se_col])
        df_val   = float(row[df_col])
        p_val    = float(row[pval_col])

        t_crit  = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))
        ci_low  = estimate - t_crit * se
        ci_high = estimate + t_crit * se

        idx_a = labels.index(label_a)
        idx_b = labels.index(label_b)
        per_input_diffs = cell_means_2d[idx_a] - cell_means_2d[idx_b]

        # Use NaN-safe std and count only inputs where both templates observed.
        n_complete = int(np.sum(~np.isnan(per_input_diffs)))
        std_diff = (
            float(np.nanstd(per_input_diffs, ddof=1)) if n_complete > 1 else 0.0
        )

        res = PairedDiffResult(
            template_a=label_a,
            template_b=label_b,
            point_diff=estimate,
            std_diff=std_diff,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_val,
            test_method=f"lmm wald (pymer4, df={df_val:.0f})",
            n_inputs=n_complete,
            per_input_diffs=per_input_diffs,
            n_runs=1,  # cell means are already run-averaged
            statistic="mean",  # LMM is a mean-based model
        )
        results[(label_a, label_b)] = res
        pairs.append((label_a, label_b))

    if not results:
        raise RuntimeError(
            "pymer4 emmeans returned no usable contrasts. "
            "Check that template labels are simple strings without ' - '."
        )

    _resolved_correction = _apply_pvalue_correction(results, pairs, correction, n_groups=len(labels))
    return PairwiseMatrix(labels=labels, results=results, correction_method=_resolved_correction)


# ---------------------------------------------------------------------------
# Rank distribution
# ---------------------------------------------------------------------------

def _extract_variance_components(model: Any) -> tuple[float, float]:
    """Return (sigma_input, sigma_resid) from the fitted LMM.

    In pymer4 0.9, ``model.ranef_var`` is a Polars DataFrame produced by
    ``broom.mixed::tidy(effects="ran_pars")``.  The ``estimate`` column
    contains **standard deviations** (not variances) — broom.mixed uses the
    ``"sdcor"`` scale by default.  Column layout::

        group     | term             | estimate
        --------- | ---------------- | --------
        input     | sd__(Intercept)  | σ_input
        Residual  | sd__Observation  | σ_resid
    """
    rv = model.ranef_var
    group_col = _col_pl(rv, ["group", "Group", "grp", "Groups"])
    term_col  = _col_pl(rv, ["term", "Term", "name", "Name"])
    est_col   = _col_pl(rv, ["estimate", "Estimate", "Var", "var", "variance"])

    sigma_input = 0.0
    sigma_resid = 0.0

    for row in rv.iter_rows(named=True):
        group = str(row[group_col]).strip()
        term  = str(row[term_col]).strip().lower()
        val   = float(row[est_col])

        if group.lower() in ("residual", "resid", "residuals") or "observation" in term:
            # Already a standard deviation in pymer4 0.9
            sigma_resid = max(val, 0.0)
        else:
            # Random-effect group (we expect "input"); also an SD in 0.9
            sigma_input = max(val, 0.0)

    return sigma_input, sigma_resid


def _simulate_rank_dist(
    template_means: np.ndarray,
    sigma_input: float,
    sigma_resid: float,
    M: int,
    labels: list[str],
    n_sim: int,
    rng: np.random.Generator,
) -> RankDistribution:
    """Parametric rank distribution via simulation — shared by both backends.

    At each iteration:

    1. Draw M new input random effects ~ N(0, σ²_input).
    2. Draw residuals ~ N(0, σ²_resid) for each (template, input) cell.
    3. Rank templates by their mean over the M simulated inputs.

    This propagates both the estimation uncertainty (via the fixed-effect
    means) and the structural variance (σ²_input, σ²_resid) into the rank
    distribution, making it more informative than a bootstrap on cell means
    when M is small.
    """
    N = len(labels)

    rank_counts = np.zeros((N, N), dtype=np.int64)

    for _ in range(n_sim):
        input_effects = (
            rng.normal(0.0, sigma_input, size=M)
            if sigma_input > 0 else np.zeros(M)
        )
        resid = (
            rng.normal(0.0, sigma_resid, size=(N, M))
            if sigma_resid > 0 else np.zeros((N, M))
        )
        sim_scores = template_means[:, None] + input_effects[None, :] + resid
        order = np.argsort(-sim_scores.mean(axis=1))
        for rank, tidx in enumerate(order):
            rank_counts[tidx, rank] += 1

    rank_probs     = rank_counts / n_sim
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best         = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_sim,
    )


def _lmm_to_rank_dist(
    model: Any,
    labels: list[str],
    cell_means_2d: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> RankDistribution:
    """Parametric rank distribution from a fitted pymer4 model."""
    M = cell_means_2d.shape[1]
    template_means           = _extract_template_means(model, labels)
    sigma_input, sigma_resid = _extract_variance_components(model)
    return _simulate_rank_dist(template_means, sigma_input, sigma_resid, M, labels, n_sim, rng)


# ---------------------------------------------------------------------------
# LMM diagnostics
# ---------------------------------------------------------------------------

def _compute_icc(sigma_input: float, sigma_resid: float) -> float:
    """Intraclass correlation: σ²_input / (σ²_input + σ²_resid).

    Returns 0.0 when total variance is zero (degenerate model).
    """
    var_input = sigma_input ** 2
    var_resid = sigma_resid ** 2
    total_var = var_input + var_resid
    return var_input / total_var if total_var > 0 else 0.0


def _build_lmm_info(model: Any, n_obs: int) -> LMMInfo:
    """Extract ``LMMInfo`` from a fitted pymer4 ``lmer`` model."""
    sigma_input, sigma_resid = _extract_variance_components(model)
    icc = _compute_icc(sigma_input, sigma_resid)

    # pymer4 0.9 stores the convergence result in model.convergence_status
    # as a string containing the R output; "TRUE" signals successful convergence.
    converged = True
    status = getattr(model, "convergence_status", None)
    if status is not None:
        status_str = str(status).strip()
        if "TRUE" not in status_str.upper():
            converged = False

    return LMMInfo(
        icc=icc,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
        n_obs=n_obs,
        formula="score ~ template + (1|input)",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# statsmodels backend
# ---------------------------------------------------------------------------

def _require_statsmodels() -> Any:
    """Import and return ``statsmodels.formula.api``, or raise a helpful ImportError."""
    try:
        import statsmodels.formula.api as smf  # type: ignore[import]
        return smf
    except ImportError:
        raise ImportError(
            "statsmodels is required for method='lmm' with backend='statsmodels'.\n"
            "Install it with:\n    pip install statsmodels"
        ) from None


def _scores_to_long_df_pandas(
    scores: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
) -> "pd.DataFrame":
    """Convert a score array to a long-form pandas DataFrame.

    Accepts either:

    * ``(N, M)`` — one row per ``(template, input)`` cell mean.
    * ``(N, M, R)`` — one row per ``(template, input, run)`` observation;
      a ``'run'`` column is added.  This lets the LMM use individual
      run observations as i.i.d. residuals rather than pre-averaged cell
      means, which propagates seed variance into fixed-effect CIs.

    Returns a DataFrame with columns ``'template'``, ``'input'``, ``'score'``
    (and ``'run'`` when 3-D input is given).  The ``'template'`` column is a
    ``pd.Categorical`` with ``template_labels[0]`` as the first (reference)
    category.  NaN rows are dropped.
    """
    import pandas as pd

    if scores.ndim == 2:
        N, M = scores.shape
        templates   = np.repeat(template_labels, M).tolist()
        inputs      = np.tile(input_labels, N).tolist()
        scores_flat = scores.ravel().tolist()
        df = pd.DataFrame({"template": templates, "input": inputs, "score": scores_flat})
    elif scores.ndim == 3:
        N, M, R = scores.shape
        # Layout after ravel(): template0/input0/run0, t0/i0/run1, ..., t0/i1/run0, ...
        templates   = np.repeat(template_labels, M * R).tolist()
        inputs      = np.tile(np.repeat(input_labels, R), N).tolist()
        runs        = np.tile(np.arange(R), N * M).tolist()
        scores_flat = scores.ravel().tolist()
        df = pd.DataFrame({"template": templates, "input": inputs, "run": runs, "score": scores_flat})
    else:
        raise ValueError(
            f"scores must be 2-D (N, M) or 3-D (N, M, R); got {scores.ndim}-D array."
        )

    df = df.dropna(subset=["score"])
    df["template"] = pd.Categorical(df["template"], categories=template_labels)
    return df


def _sm_param_names(template_labels: list[str]) -> list[str]:
    """Return the statsmodels fixed-effect parameter names for treatment coding.

    statsmodels ``C(template)`` produces names like::

        ["Intercept", "C(template)[T.T1]", "C(template)[T.T2]", ...]

    where ``template_labels[0]`` is the (omitted) reference level.
    """
    return ["Intercept"] + [f"C(template)[T.{lbl}]" for lbl in template_labels[1:]]


def _fit_lmm_sm(df_pandas: "pd.DataFrame", template_labels: list[str]) -> Any:
    """Fit ``score ~ C(template) + (1|input)`` via statsmodels MixedLM (REML).

    Returns a fitted ``MixedLMResults`` object.
    """
    import statsmodels.formula.api as smf  # type: ignore[import]
    model = smf.mixedlm("score ~ C(template)", data=df_pandas, groups=df_pandas["input"])
    return model.fit(reml=True, disp=False)


def _extract_template_means_sm(sm_result: Any, labels: list[str]) -> np.ndarray:
    """Extract fitted marginal means from a statsmodels MixedLMResults object.

    With treatment coding (reference = ``labels[0]``):

    * μ₀  = intercept
    * μᵢ  = intercept + β_i   for i > 0
    """
    N = len(labels)
    param_names = _sm_param_names(labels)

    intercept = float(sm_result.fe_params["Intercept"])
    means = np.empty(N)
    means[0] = intercept
    for i in range(1, N):
        means[i] = intercept + float(sm_result.fe_params[param_names[i]])
    return means


def _get_vcov_sm(sm_result: Any, template_labels: list[str]) -> np.ndarray:
    """Extract the fixed-effects covariance matrix as a numpy array ``(N, N)``.

    Slices ``result.cov_params()`` to the treatment-coding parameters
    ``[Intercept, β₁, …, β_{N-1}]`` in order.
    """
    param_names = _sm_param_names(template_labels)
    vcov_df = sm_result.cov_params().loc[param_names, param_names]
    return vcov_df.to_numpy()


def _extract_variance_components_sm(sm_result: Any) -> tuple[float, float]:
    """Return ``(sigma_input, sigma_resid)`` from a statsmodels MixedLMResults.

    ``result.cov_re`` is the random-effect covariance (1×1 for random intercepts);
    its single entry is σ²_input.  ``result.scale`` is σ²_resid.
    Both are returned as standard deviations.
    """
    var_input = float(sm_result.cov_re.iloc[0, 0])
    var_resid = float(sm_result.scale)
    sigma_input = float(np.sqrt(max(var_input, 0.0)))
    sigma_resid = float(np.sqrt(max(var_resid, 0.0)))
    return sigma_input, sigma_resid


def _lmm_to_pairwise_sm(
    sm_result: Any,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    correction: str,
) -> PairwiseMatrix:
    """Build a ``PairwiseMatrix`` from statsmodels fixed effects via delta method.

    For each pair (A, B) the contrast vector ``c`` satisfies ``c @ betas = μ_A - μ_B``
    under treatment coding.  The variance is ``c @ vcov @ c``.  A conservative
    t-distribution with ``result.df_resid`` degrees of freedom is used.
    """
    alpha    = 1 - ci
    N        = len(labels)
    betas    = np.empty(N)
    betas[0] = float(sm_result.fe_params["Intercept"])
    for i in range(1, N):
        betas[i] = float(sm_result.fe_params[_sm_param_names(labels)[i]])

    vcov  = _get_vcov_sm(sm_result, labels)
    df_val = float(sm_result.df_resid)

    # Build N contrast rows: L[i] @ betas = μ_i
    # Reuse the already-implemented advantage contrast matrix with ref=None (grand mean)
    # but we only need the μ_i-row structure; it's simpler to build directly.
    # Under treatment coding: μ₀ = β₀;  μᵢ = β₀ + βᵢ (i>0)
    # So the contrast vector for μᵢ is L_i where:
    #   L_0 = [1, 0, 0, ...]
    #   L_i = [1, 0, ..., 1, ..., 0]  (1 at position i)
    L = np.zeros((N, N))
    L[:, 0] = 1.0          # intercept column
    for i in range(1, N):
        L[i, i] = 1.0     # β_i column

    results: dict[tuple[str, str], PairedDiffResult] = {}
    pairs: list[tuple[str, str]] = []

    for idx_a in range(N):
        for idx_b in range(idx_a + 1, N):
            label_a = labels[idx_a]
            label_b = labels[idx_b]

            c        = L[idx_a] - L[idx_b]          # contrast: μ_a - μ_b
            estimate = float(c @ betas)
            var_est  = float(c @ vcov @ c)
            se       = float(np.sqrt(max(var_est, 0.0)))

            t_crit  = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))
            ci_low  = estimate - t_crit * se
            ci_high = estimate + t_crit * se

            t_stat  = estimate / se if se > 0 else 0.0
            p_val   = float(2 * scipy.stats.t.sf(abs(t_stat), df=df_val))

            per_input_diffs = cell_means_2d[idx_a] - cell_means_2d[idx_b]
            n_complete = int(np.sum(~np.isnan(per_input_diffs)))
            std_diff   = (
                float(np.nanstd(per_input_diffs, ddof=1)) if n_complete > 1 else 0.0
            )

            res = PairedDiffResult(
                template_a=label_a,
                template_b=label_b,
                point_diff=estimate,
                std_diff=std_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_val,
                test_method=f"lmm wald (statsmodels, df={df_val:.0f})",
                n_inputs=n_complete,
                per_input_diffs=per_input_diffs,
                n_runs=1,
                statistic="mean",
            )
            results[(label_a, label_b)] = res
            pairs.append((label_a, label_b))

    if not results:
        raise RuntimeError("statsmodels LMM returned no usable contrasts.")

    _resolved_correction = _apply_pvalue_correction(results, pairs, correction, n_groups=len(labels))
    return PairwiseMatrix(labels=labels, results=results, correction_method=_resolved_correction)


def _build_lmm_info_sm(sm_result: Any, n_obs: int) -> LMMInfo:
    """Extract ``LMMInfo`` from a fitted statsmodels ``MixedLMResults``."""
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    icc = _compute_icc(sigma_input, sigma_resid)
    converged = bool(getattr(sm_result, "converged", True))

    return LMMInfo(
        icc=icc,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
        n_obs=n_obs,
        formula="score ~ template + (1|input)",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# PPI-corrected fixed effects (statsmodels backend)
# ---------------------------------------------------------------------------

@dataclass
class PPILMMResult:
    """Closed-form PPI-corrected fixed effects for an LMM with one ``(1|input)``
    random intercept, any number of crossed fixed factors, and optional
    nested run replication.

    Attributes
    ----------
    param_names : list[str]
        Fixed-effect parameter names exactly as fitted by statsmodels for
        whatever formula was used (``["Intercept", "C(template)[T.T1]", ...]``
        for a single factor; also includes interaction terms like
        ``"C(model)[T.b]:C(prompt)[T.v2]"`` for a factorial design).
    beta_unlab, beta_ppi : np.ndarray
        Uncorrected (LLM-only) and PPI-corrected fixed-effect estimates, shape ``(P,)``.
    cov_unlab : np.ndarray
        Model-based (naive Wald) covariance of ``beta_unlab`` at the fixed
        REML variance components, shape ``(P, P)``.
    cov_ppi : np.ndarray
        Cluster-robust sandwich covariance of ``beta_ppi``, shape ``(P, P)``
        — ``cov_unlab`` plus an inflation term from the labeled-input
        rectifier covariance, minus a cross-covariance correction (labeled
        inputs contribute to both ``beta_unlab`` and the rectifier, so the
        two are not independent).
    n_inputs : int
        Total number of inputs (random-effect groups) in the full dataset.
    n_lab : int
        Number of inputs with at least one labeled template cell.
    sigma_input, sigma_resid : float
        Variance-component SDs, estimated once from the full LLM-scored fit
        and held fixed as nuisance parameters. For nested run replication
        (R > 1), ``sigma_resid`` is already the run-averaged residual SD —
        see ``_fit_lmm_general``.
    """

    param_names: list[str]
    beta_unlab: np.ndarray
    beta_ppi: np.ndarray
    cov_unlab: np.ndarray
    cov_ppi: np.ndarray
    n_inputs: int
    n_lab: int
    sigma_input: float
    sigma_resid: float


def _fit_lmm_general(
    groups: list[np.ndarray],
    template_labels: list[str],
    factors: Optional["pd.DataFrame"] = None,
) -> tuple[Any, "pd.DataFrame", np.ndarray, int]:
    """Fit ``score ~ <fixed factors> + (1|input)`` for any number of crossed
    fixed factors and optional nested run replication, shared by
    ``_ppi_lmm_fixed_effects`` and ``es.tests.lmm()``'s uncorrected branch
    so both use identical fitting/design-matrix logic.

    ``groups`` entries are each ``(n_inputs,)`` or, for R nested runs per
    cell, ``(n_inputs, R)``; all entries must share the same shape. ``factors``
    is a DataFrame with one row per ``template_labels`` entry and one column
    per fixed factor, or ``None`` for the implicit single ``"template"`` factor.

    Returns ``(sm_result, df_full, X_row, R)``: the fitted ``MixedLMResults``,
    the long-form DataFrame it was fit on, the ``(k, P)`` per-template/group
    mean design vector (see ``_get_template_design_vectors`` — constant
    across inputs/runs since neither enters the fixed-effects formula), and
    the detected run count R (1 if groups are 1-D).

    When R > 1, the fit is on the run-*collapsed* (mean-over-R) data, not
    the raw per-run observations. A cell's R repeated runs generally share
    a fixed residual component (e.g. genuine item-level heterogeneity in
    how a template performs on that item) on top of independent per-run
    LLM sampling noise — fitting the raw per-run rows would estimate one
    blended ``sigma_resid`` for both and offer no way to recover just the
    per-run-noise share that repetition should shrink. Averaging first
    sidesteps this: for a balanced design (every cell has the same R),
    fixed-effect point estimates are unaffected, and the fitted
    ``sigma_resid`` on the averaged data is *exactly*
    ``Var(cell residual) + Var(run noise)/R`` — precisely the quantity every
    downstream GLS computation needs, with no further per-R adjustment.
    """
    ndims = {g.ndim for g in groups}
    if ndims not in ({1}, {2}):
        raise ValueError("groups must be all 1-D (n_inputs,) or all 2-D (n_inputs, R).")
    n_inputs = groups[0].shape[0]

    if ndims == {1}:
        R = 1
    else:
        run_counts = {g.shape[1] for g in groups}
        if len(run_counts) != 1:
            raise ValueError(
                f"All groups must have the same number of runs R; got {run_counts}."
            )
        R = run_counts.pop()

    input_labels = [f"_input{j}" for j in range(n_inputs)]
    scores_arr = (
        np.column_stack(groups).T if R == 1
        else np.stack(groups, axis=0).mean(axis=-1)
    )  # (k, n_inputs) -- run-averaged when R > 1, see docstring above

    if factors is None:
        # "template" is already the group-identifier column _scores_to_long_df_pandas
        # produces, so the implicit single-factor case needs no factor-table join --
        # it's already a valid factorial formula input of one factor.
        factor_names = ["template"]
        df_full = _scores_to_long_df_pandas(scores_arr, template_labels, input_labels)
    else:
        factor_names = list(factors.columns)
        df_full = _scores_to_long_df_factorial_pandas(scores_arr, template_labels, input_labels, factors)
    sm_result, _ = _fit_factorial_lmm_sm(df_full, factor_names)

    # Per-template design row, mean exog vector for that template/group --
    # generalizes the old hand-rolled treatment-coding X_row to any formula.
    X_row = _get_template_design_vectors(sm_result, df_full, template_labels)  # (k, P)
    return sm_result, df_full, X_row, R


def _ppi_lmm_fixed_effects(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    template_labels: list[str],
    factors: Optional["pd.DataFrame"] = None,
    *,
    precomputed_fit: Optional[tuple] = None,
) -> PPILMMResult:
    """Closed-form PPI correction for an LMM's fixed effects: any number of
    crossed fixed factors, one ``(1|input)`` random intercept, and optional
    nested run replication.

    LMM fixed effects come from (restricted) maximum likelihood — a
    nonlinear M-estimator — so independently refitting the model on the
    full LLM data, the labeled-human data, and the labeled-LLM data and
    adding the three coefficient vectors (the "plug-in" recipe used
    elsewhere in this module) is *not* a valid bias correction, and
    re-estimating variance components on a small labeled subset is often
    unstable or non-convergent. Instead this solves the *combined*
    rectified estimating equation directly — the general M-estimation form
    of PPI (Angelopoulos et al. 2023):

        (1/N) Σ_unlab ψ(β; Ŷ, X) + (1/n) Σ_lab [ψ(β; Y, X) − ψ(β; Ŷ, X)] = 0

    where ψ is the GLS score ``Xᵗ V⁻¹(Y − Xβ)``. σ²_input/σ²_resid (hence
    V) are estimated once from the full LLM-scored fit and held fixed as
    nuisance parameters — only β is PPI-corrected. Because the model is
    linear in β, the β-dependent terms in the rectifier cancel exactly,
    collapsing the combined equation to one closed-form GLS solve:

        β̂_PPI = β̂_unlab + (n_inputs/n_lab) · M⁻¹ r

    ``M = Xᵗ V⁻¹X`` is recovered directly from the full fit's Wald
    covariance (``M⁻¹ = cov_params``) rather than rebuilt by hand — that
    covariance already accounts for any ragged/missing template cells the
    same way the rest of the LMM path does, and (via ``_fit_factorial_lmm_sm``)
    for any number of crossed fixed factors and any number of nested runs
    per cell. ``r = Σ_{i∈labeled} Xᵢᵗ Vᵢ⁻¹(Yᵢ − Ŷᵢ)`` is the GLS rectifier,
    summed over labeled *inputs* (the natural cluster/exchangeability unit
    here, mirroring the per-subject overlap requirement other
    repeated-measures PPI tests in this module use) — a labeled input needs
    only one labeled template cell, not all of them, and the same is true
    regardless of how many fixed factors define "template."

    Multiple fixed factors are handled by delegating to the existing
    factorial-LMM scaffolding (``_fit_factorial_lmm_sm`` /
    ``_scores_to_long_df_factorial_pandas`` / ``_get_template_design_vectors``)
    instead of hand-building a single-factor treatment-coding matrix — the
    per-template design row ``Xᵢ`` and the fitted ``β̂_unlab``/``cov_unlab``
    come out exactly the same way regardless of how many factors (or
    interactions) the formula has, so the rest of the correction (rectifier,
    sandwich, cross-covariance) needs no changes at all for N factors.

    Nested run replication (``groups[t]`` shape ``(n_inputs, R)`` instead of
    ``(n_inputs,)``) is handled by ``_fit_lmm_general`` fitting on the
    run-*collapsed* (mean-over-R) data directly rather than the raw per-run
    rows — see its docstring for why: a cell's R runs generally share a
    fixed residual component (real item-level heterogeneity) on top of
    independent per-run LLM noise, so naively fitting the raw rows and
    dividing the resulting ``sigma_resid**2`` by R afterward would wrongly
    shrink *all* of it, including the part that doesn't shrink with more
    runs. Averaging first makes the fitted ``sigma_resid`` already equal
    the correct combined quantity, so every GLS computation below uses
    ``sigma_resid**2`` directly with no further per-R adjustment. Human
    labels (``groups_lab``) stay one value per item regardless of R: a
    single human judgment is compared against the LLM's run-average for
    that item.

    Inference uses a cluster-robust sandwich over labeled inputs rather
    than a bootstrap: refitting variance components on a small labeled
    subset at every bootstrap draw would be unstable, whereas the sandwich
    only needs the empirical covariance of the per-input rectifier
    contributions.

    The sandwich also includes a cross-covariance correction between
    ``β̂_unlab`` and the rectifier-driven shift, which a naive "treat the two
    pieces as independent" sandwich (the convention this module's bootstrap
    tests use, by resampling the unlabeled and labeled draws separately)
    would miss. Labeled inputs are literally a *subset* of the full
    dataset, so for those inputs the same LLM measurement noise perturbs
    ``β̂_unlab`` (positively) and the rectifier (negatively) — they partially
    cancel, which *reduces* true sampling variance relative to treating the
    two pieces as independent. Omitting this term doesn't bias the point
    estimate, but it overstates the variance (Wald CIs too wide, tests too
    conservative) by an amount that scales with the labeled fraction
    n_lab/n_inputs, which is not always small. It's estimated empirically:
    for each labeled input, alongside its rectifier contribution ``e_i``,
    compute its *full-row* (all k templates, not just the labeled ones)
    unlabeled GLS residual ``u_i = Xᵢᵗ Vᵢ⁻¹(Ŷᵢ − Xᵢβ̂_unlab)``, then take the
    empirical cross-covariance of ``{u_i}`` and ``{e_i}`` across labeled
    inputs — the same cluster-sample-covariance machinery already used for
    the rectifier's own variance, just crossed with the matching unlabeled
    residual instead of squared against itself.

    Parameters
    ----------
    groups : list of np.ndarray
        One array per template/group, each shape ``(n_inputs,)`` (no run
        replication) or ``(n_inputs, R)`` (R nested runs per cell). All
        groups must share the same shape/R.
    groups_lab : list of np.ndarray
        One array per template/group, each shape ``(n_inputs,)`` — sparse
        human labels (NaN where unlabeled), one per item regardless of R.
    template_labels : list[str]
        Identifier per group/template; row order must match ``groups``.
    factors : pd.DataFrame, optional
        One row per ``template_labels`` entry (same order), one column per
        fixed factor. Builds ``score ~ C(F1) * C(F2) * ... + (1|input)``.
        Defaults to a single implicit ``"template"`` factor (the original
        single-factor model) when not given.
    precomputed_fit : tuple, optional
        The exact ``(sm_result, df_full, X_row, R)`` tuple ``_fit_lmm_general``
        would otherwise compute internally. Callers that already need the
        LLM-only fit for something else (e.g. an uncorrected Wald F-test
        computed from the same ``groups``) can pass it here to skip
        refitting the identical MixedLM model a second time -- refitting is
        by far the most expensive step in this function.
    """
    k = len(groups)
    n_inputs = groups[0].shape[0]

    for lab in groups_lab:
        if lab.ndim != 1 or lab.shape[0] != n_inputs:
            raise ValueError(
                "groups_lab entries must be 1-D with one label per item "
                "(shape (n_inputs,)), regardless of run replication in groups."
            )

    if precomputed_fit is not None:
        sm_result, df_full, X_row, R = precomputed_fit
    else:
        sm_result, df_full, X_row, R = _fit_lmm_general(groups, template_labels, factors)
    groups_collapsed = groups if R == 1 else [g.mean(axis=1) for g in groups]

    param_names = sm_result.fe_params.index.tolist()
    beta_unlab = sm_result.fe_params.to_numpy()
    cov_unlab = _get_fe_vcov_sm(sm_result)
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    var_input = sigma_input ** 2
    # Already the correct run-collapsed residual variance: _fit_lmm_general
    # fits on mean-over-R data directly, so no further /R adjustment here.
    var_resid_eff = sigma_resid ** 2
    V_full = var_resid_eff * np.eye(k) + var_input * np.ones((k, k))

    e_list = []
    u_list = []
    for j in range(n_inputs):
        labeled_t = [t for t in range(k) if not np.isnan(groups_lab[t][j])]
        if not labeled_t:
            continue
        X_i = X_row[labeled_t]
        y_i = np.array([groups_lab[t][j] for t in labeled_t])
        yhat_i = np.array([groups_collapsed[t][j] for t in labeled_t])
        m = len(labeled_t)
        V_i = var_resid_eff * np.eye(m) + var_input * np.ones((m, m))
        e_list.append(X_i.T @ np.linalg.solve(V_i, y_i - yhat_i))

        # Full-row (all k templates) unlabeled GLS residual for this same
        # input, at the fitted beta_unlab -- needed for the cross-covariance
        # correction below, not for the rectifier itself.
        yhat_full_j = np.array([groups_collapsed[t][j] for t in range(k)])
        resid_full_j = yhat_full_j - X_row @ beta_unlab
        u_list.append(X_row.T @ np.linalg.solve(V_full, resid_full_j))

    n_lab = len(e_list)
    if n_lab < 2:
        raise ValueError(
            "PPI-corrected LMM requires at least 2 inputs with a labeled "
            f"template cell (found {n_lab})."
        )

    E = np.column_stack(e_list)  # (P, n_lab)
    r = E.sum(axis=1)

    beta_ppi = beta_unlab + (n_inputs / n_lab) * (cov_unlab @ r)

    # Cluster-robust "meat": empirical covariance of the per-input rectifier
    # contributions across labeled inputs.
    e_centered = E - E.mean(axis=1, keepdims=True)
    S_hat = (e_centered @ e_centered.T) / (n_lab - 1)

    # Cross-covariance correction: Cov(beta_unlab, shift) is not zero because
    # labeled inputs contribute to both terms -- see docstring. Estimated as
    # the empirical cross-covariance of the full-row unlabeled residuals
    # {u_i} and the rectifier contributions {e_i} across labeled inputs.
    U = np.column_stack(u_list)  # (P, n_lab)
    u_centered = U - U.mean(axis=1, keepdims=True)
    C_hat = (u_centered @ e_centered.T) / (n_lab - 1)

    cov_ppi = (
        cov_unlab
        + (n_inputs ** 2 / n_lab) * (cov_unlab @ S_hat @ cov_unlab)
        + n_inputs * (cov_unlab @ (C_hat + C_hat.T) @ cov_unlab)
    )

    return PPILMMResult(
        param_names=param_names,
        beta_unlab=beta_unlab,
        beta_ppi=beta_ppi,
        cov_unlab=cov_unlab,
        cov_ppi=cov_ppi,
        n_inputs=n_inputs,
        n_lab=n_lab,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
    )


# ---------------------------------------------------------------------------
# Factorial LMM — statsmodels backend
# ---------------------------------------------------------------------------

def _build_factorial_formula(factor_names: list[str]) -> str:
    """Build ``score ~ C(F1) * C(F2) * ... + (1|input)`` formula string.

    The ``*`` expansion yields all main effects and interactions.
    Returns a plain string suitable for ``smf.mixedlm``; the random-effect
    part ``(1|input)`` is specified separately via the ``groups`` argument.
    """
    fixed = " * ".join(f"C({f})" for f in factor_names)
    return f"score ~ {fixed}"


def _scores_to_long_df_factorial_pandas(
    scores: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
    template_factors: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build a long-form pandas DataFrame with factor columns appended.

    Calls :func:`_scores_to_long_df_pandas` (which accepts both 2-D and 3-D
    *scores*) then left-joins the factor columns from *template_factors*,
    aligning rows positionally to *template_labels*.
    """
    import pandas as pd

    df = _scores_to_long_df_pandas(scores, template_labels, input_labels)

    # Positional alignment: row i of template_factors → template_labels[i]
    factor_map = template_factors.copy()
    factor_map.index = template_labels
    df = df.join(factor_map, on="template")
    return df


def _get_fe_vcov_sm(sm_result: Any) -> np.ndarray:
    """Return the fixed-effect-only covariance matrix from a statsmodels MixedLMResults.

    ``sm_result.cov_params()`` includes both fixed-effect and random-effect
    variance parameters.  This helper slices it to just the rows/columns
    corresponding to ``sm_result.fe_params.index``.
    """
    fe_names = sm_result.fe_params.index.tolist()
    return sm_result.cov_params().loc[fe_names, fe_names].to_numpy()


def _fit_factorial_lmm_sm(df_pandas: "pd.DataFrame", factor_names: list[str]) -> tuple[Any, str]:
    """Fit ``score ~ C(F1) * C(F2) * ... + (1|input)`` via statsmodels MixedLM.

    Returns ``(sm_result, formula_str)``.
    """
    import statsmodels.formula.api as smf
    formula = _build_factorial_formula(factor_names)
    model = smf.mixedlm(formula, data=df_pandas, groups=df_pandas["input"])
    return model.fit(reml=True, disp=False), formula


def _get_template_design_vectors(
    sm_result: Any,
    df_pandas: "pd.DataFrame",
    labels: list[str],
) -> np.ndarray:
    """Return ``(N, P)`` matrix: row i = mean design vector for template i.

    ``sm_result.model.exog`` rows are aligned with ``df_pandas`` rows (statsmodels
    preserves input row order).  Averaging per-template gives the effective
    contrast vector ``c_i`` such that ``c_i @ fe_params == predicted cell mean``.
    """
    exog = sm_result.model.exog          # (n_obs, P)
    templates = df_pandas["template"].values  # (n_obs,)

    N = len(labels)
    P = exog.shape[1]
    design_vecs = np.zeros((N, P))
    for i, label in enumerate(labels):
        mask = templates == label
        if not mask.any():
            raise RuntimeError(
                f"Template '{label}' missing from long-form DataFrame. "
                "This is a bug — please open an issue."
            )
        design_vecs[i] = exog[mask].mean(axis=0)
    return design_vecs


def _extract_factor_tests_sm(sm_result: Any, factor_names: list[str]) -> "pd.DataFrame":
    """Return a DataFrame of Wald tests per model term (main effects + interactions).

    Uses ``sm_result.wald_test_terms()`` which tests H₀: all coefficients for
    each term are zero simultaneously.  The ``Intercept`` row is dropped since
    it is not a factor.  Falls back to an empty DataFrame with a warning if
    the method is unavailable or raises an error.

    Columns: ``term``, ``statistic``, ``df``, ``p_value``.
    statsmodels returns column names ``statistic``, ``pvalue``, ``df_constraint``;
    values may be 0-D or 2-D numpy arrays and are extracted as Python floats.
    """
    import pandas as pd

    try:
        wt = sm_result.wald_test_terms(skip_single=False)
        table = wt.table.copy()

        # Normalise column names (statsmodels uses 'statistic', 'pvalue', 'df_constraint').
        col_map: dict[str, str] = {}
        for col in table.columns:
            lc = col.lower().replace(" ", "").replace("_", "").replace("-", "")
            if lc == "statistic" or "fstat" in lc or "chi2" in lc:
                col_map[col] = "statistic"
            elif lc in ("pvalue", "prob", "p>f", "p>chi2"):
                col_map[col] = "p_value"
            elif "dfconstraint" in lc or lc in ("df", "dfnum", "numdf"):
                col_map[col] = "df"
        table = table.rename(columns=col_map)

        rows = []
        for term, row in table.iterrows():
            if str(term) == "Intercept":
                continue
            stat = row.get("statistic", float("nan"))
            pval = row.get("p_value", float("nan"))
            df_c = row.get("df", float("nan"))
            # Unwrap numpy arrays of any shape to a scalar float.
            def _scalar(v: Any) -> float:
                try:
                    import numpy as _np
                    return float(_np.asarray(v).flat[0])
                except Exception:
                    return float("nan")
            rows.append({"term": str(term), "statistic": _scalar(stat),
                         "df": _scalar(df_c), "p_value": _scalar(pval)})

        return pd.DataFrame(rows, columns=["term", "statistic", "df", "p_value"])

    except Exception as exc:
        warnings.warn(
            f"Could not extract Wald tests for factorial terms: {exc}. "
            "factor_tests will be empty.",
            UserWarning,
            stacklevel=4,
        )
        return pd.DataFrame(columns=["term", "statistic", "df", "p_value"])


def _extract_marginal_means_sm(
    sm_result: Any,
    df_pandas: "pd.DataFrame",
    factor_names: list[str],
    ci: float,
) -> dict[str, "pd.DataFrame"]:
    """Estimated marginal means per factor from a fitted factorial LMM.

    For each focal factor F, the EMM for level ℓ is the prediction at
    (F=ℓ) averaged *equally* over all observed combinations of the remaining
    factors.  This matches the equal-weights marginal mean used by R's
    ``emmeans`` for balanced designs and is a sensible approximation for
    unbalanced designs.

    Returns ``dict[factor_name → pd.DataFrame]`` where each DataFrame has
    columns ``level``, ``mean``, ``se``, ``ci_low``, ``ci_high``.
    """
    import pandas as pd
    from itertools import product as iterproduct

    alpha = 1 - ci
    t_crit = float(scipy.stats.t.ppf(1 - alpha / 2, df=sm_result.df_resid))

    exog  = sm_result.model.exog           # (n_obs, P)
    betas = sm_result.fe_params.to_numpy() # (P,)
    vcov  = _get_fe_vcov_sm(sm_result)     # (P, P)
    tmpl_col = df_pandas["template"].values

    emm_per_factor: dict[str, "pd.DataFrame"] = {}

    for focal in factor_names:
        other = [f for f in factor_names if f != focal]
        focal_levels = sorted(df_pandas[focal].dropna().unique().tolist(),
                              key=lambda x: (str(type(x)), x))
        other_level_sets = [
            sorted(df_pandas[f].dropna().unique().tolist(),
                   key=lambda x: (str(type(x)), x))
            for f in other
        ]
        other_combos = list(iterproduct(*other_level_sets)) if other else [()]

        rows = []
        for level in focal_levels:
            cell_vecs = []
            for combo in other_combos:
                # Build mask: focal factor == level AND each other factor == its value
                mask = (df_pandas[focal] == level).values
                for f, v in zip(other, combo):
                    mask = mask & (df_pandas[f] == v).values
                if mask.any():
                    cell_vecs.append(exog[mask].mean(axis=0))  # (P,)

            if not cell_vecs:
                continue

            # Equal-weight average over present cells
            c   = np.mean(cell_vecs, axis=0)   # (P,)
            emm = float(c @ betas)
            se  = float(np.sqrt(max(float(c @ vcov @ c), 0.0)))

            rows.append({
                "level":   level,
                "mean":    emm,
                "se":      se,
                "ci_low":  emm - t_crit * se,
                "ci_high": emm + t_crit * se,
            })

        emm_per_factor[focal] = pd.DataFrame(rows)

    return emm_per_factor


def _build_factorial_lmm_info_sm(
    sm_result: Any,
    factor_names: list[str],
    n_obs: int,
    formula: str,
    ci: float,
    df_pandas: "pd.DataFrame",
) -> "FactorialLMMInfo":
    """Build a ``FactorialLMMInfo`` from a fitted statsmodels MixedLMResults."""
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    icc = _compute_icc(sigma_input, sigma_resid)
    converged = bool(getattr(sm_result, "converged", True))

    factor_tests  = _extract_factor_tests_sm(sm_result, factor_names)
    marginal_means = _extract_marginal_means_sm(sm_result, df_pandas, factor_names, ci)

    # Append the random-effect part to the formula string for display.
    display_formula = formula.replace("score ~", "score ~") + " + (1|input)"

    return FactorialLMMInfo(
        icc=icc,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
        n_obs=n_obs,
        formula=display_formula,
        converged=converged,
        factor_names=list(factor_names),
        factor_tests=factor_tests,
        marginal_means=marginal_means,
    )


def _lmm_to_pairwise_factorial_sm(
    sm_result: Any,
    labels: list[str],
    df_pandas: "pd.DataFrame",
    cell_means_2d: np.ndarray,
    ci: float,
    correction: str,
) -> PairwiseMatrix:
    """Pairwise Wald contrasts between all N treatment cells in a factorial LMM.

    Each template corresponds to a unique factor-level combination.  The
    contrast vector ``c = design_vec_a − design_vec_b`` encodes the
    difference in predicted cell means; its variance is ``c @ vcov @ c``.
    """
    alpha = 1 - ci
    N     = len(labels)

    design_vecs = _get_template_design_vectors(sm_result, df_pandas, labels)
    betas       = sm_result.fe_params.to_numpy()
    vcov        = _get_fe_vcov_sm(sm_result)
    df_val      = float(sm_result.df_resid)
    t_crit      = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))

    results: dict[tuple[str, str], PairedDiffResult] = {}
    pairs: list[tuple[str, str]] = []

    for idx_a in range(N):
        for idx_b in range(idx_a + 1, N):
            label_a, label_b = labels[idx_a], labels[idx_b]

            c        = design_vecs[idx_a] - design_vecs[idx_b]
            estimate = float(c @ betas)
            var_est  = float(c @ vcov @ c)
            se       = float(np.sqrt(max(var_est, 0.0)))
            t_stat   = estimate / se if se > 0 else 0.0
            p_val    = float(2 * scipy.stats.t.sf(abs(t_stat), df=df_val))

            ci_low  = estimate - t_crit * se
            ci_high = estimate + t_crit * se

            per_input_diffs = cell_means_2d[idx_a] - cell_means_2d[idx_b]
            n_complete = int(np.sum(~np.isnan(per_input_diffs)))
            std_diff   = float(np.nanstd(per_input_diffs, ddof=1)) if n_complete > 1 else 0.0

            res = PairedDiffResult(
                template_a=label_a,
                template_b=label_b,
                point_diff=estimate,
                std_diff=std_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_val,
                test_method=f"factorial lmm wald (statsmodels, df={df_val:.0f})",
                n_inputs=n_complete,
                per_input_diffs=per_input_diffs,
                n_runs=1,
                statistic="mean",
            )
            results[(label_a, label_b)] = res
            pairs.append((label_a, label_b))

    if not results:
        raise RuntimeError("Factorial LMM produced no usable pairwise contrasts.")

    _resolved_correction = _apply_pvalue_correction(results, pairs, correction, n_groups=len(labels))
    return PairwiseMatrix(labels=labels, results=results, correction_method=_resolved_correction)


def _lmm_analyze_factorial_sm(
    result: Any,
    *,
    reference: str,
    ci: float,
    correction: str,
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    n_sim: int,
    rng: np.random.Generator,
) -> tuple[
    PairwiseMatrix,
    RankDistribution,
    RobustnessResult,
    Optional[SeedVarianceResult],
    FactorialLMMInfo,
]:
    """Full factorial LMM pipeline (statsmodels backend only).

    Fits ``score ~ C(F1) * C(F2) * ... + (1|input)`` on cell-mean scores
    and returns the same tuple as :func:`lmm_analyze`:
    ``(pairwise, rank_dist, robustness, seed_var, factorial_lmm_info)``.
    """
    _require_statsmodels()

    M      = result.n_inputs
    labels = result.template_labels
    inputs = result.input_labels

    template_factors = result.template_factors
    factor_names     = list(template_factors.columns)

    cell_means_2d = result.get_2d_scores()  # (N, M)
    run_scores    = result.get_run_scores() # (N, M, R)

    robustness = robustness_metrics(run_scores, labels, failure_threshold=failure_threshold)
    seed_var: Optional[SeedVarianceResult] = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    # When runs are available, fit on individual observations so that seed
    # variance enters the residual and inflates fixed-effect CIs appropriately.
    lmm_scores = run_scores if result.is_seeded else cell_means_2d
    df       = _scores_to_long_df_factorial_pandas(lmm_scores, labels, inputs, template_factors)
    n_obs    = len(df)
    sm_result, formula = _fit_factorial_lmm_sm(df, factor_names)

    lmm_info = _build_factorial_lmm_info_sm(sm_result, factor_names, n_obs, formula, ci, df)

    if not lmm_info.converged:
        warnings.warn(
            "The factorial LMM optimizer did not converge. "
            "Results may be unreliable. Consider using method='auto' "
            "or simplifying the model.",
            UserWarning,
            stacklevel=4,
        )

    pairwise = _lmm_to_pairwise_factorial_sm(sm_result, labels, df, cell_means_2d, ci, correction)
    design_vecs              = _get_template_design_vectors(sm_result, df, labels)
    template_means           = design_vecs @ sm_result.fe_params.to_numpy()
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    rank_dist = _simulate_rank_dist(
        template_means, sigma_input, sigma_resid, M, labels, n_sim, rng
    )

    return pairwise, rank_dist, robustness, seed_var, lmm_info


# ---------------------------------------------------------------------------
# Factorial LMM — pymer4 / lme4 backend
# ---------------------------------------------------------------------------

def _get_r_model_matrix(model: Any) -> np.ndarray:
    """Extract the fixed-effect model matrix from a fitted lme4/pymer4 model.

    Calls ``stats::model.matrix(r_model)`` via rpy2 and converts the result
    to a numpy array of shape ``(n_obs, P)`` where ``n_obs`` is the number
    of observations used in the fit and ``P`` the number of fixed-effect
    parameters.  Row order matches the data passed to ``lmer``.
    """
    from rpy2.robjects.packages import importr
    base_r  = importr("base")
    stats_r = importr("stats")
    mm_r = stats_r.model_matrix(model.r_model)
    return np.asarray(base_r.as_matrix(mm_r))   # (n_obs, P)


def _get_fe_params_pymer4(model: Any) -> np.ndarray:
    """Return the fixed-effect parameter vector from a pymer4 model as numpy.

    Reads ``model.result_fit`` (a Polars DataFrame) and returns the
    ``estimate`` column as a 1-D array of shape ``(P,)`` in the same
    parameter order as the model matrix columns.
    """
    rf      = model.result_fit
    est_col = _col_pl(rf, ["estimate", "Estimate", "coefficient", "Coefficient"])
    return rf[est_col].to_numpy()


def _get_template_design_vectors_pymer4(
    model: Any,
    df_pandas: "pd.DataFrame",
    labels: list[str],
) -> np.ndarray:
    """Return ``(N, P)`` per-template design vectors from a pymer4 factorial model.

    Uses ``stats::model.matrix`` (via rpy2) to extract the ``(n_obs, P)``
    design matrix, then averages rows within each template cell — mirroring
    :func:`_get_template_design_vectors` for the statsmodels path.
    """
    exog      = _get_r_model_matrix(model)        # (n_obs, P)
    templates = df_pandas["template"].values       # (n_obs,) — str/categorical

    if exog.shape[0] != len(df_pandas):
        # Row-count mismatch: the model dropped rows that we didn't expect.
        # Silently fall back to rows that are present in both.
        df_pandas = df_pandas.dropna(subset=["score"]).reset_index(drop=True)

    N = len(labels)
    P = exog.shape[1]
    design_vecs = np.zeros((N, P))
    for i, label in enumerate(labels):
        mask = templates == label
        if not mask.any():
            raise RuntimeError(
                f"Template '{label}' missing from long-form DataFrame "
                "after pymer4 model fitting. This is a bug — please open an issue."
            )
        design_vecs[i] = exog[mask].mean(axis=0)
    return design_vecs


def _fit_factorial_lmm_pymer4(
    df_pandas: "pd.DataFrame",
    factor_names: list[str],
    lmer: Any,
) -> tuple[Any, str]:
    """Fit ``score ~ F1 * F2 * ... + (1|input)`` with pymer4/lme4.

    *factor_names* are the raw column names (not ``C(F)``-wrapped as in the
    statsmodels path — R's treatment coding is configured by ``set_factors``).
    Returns ``(model, formula_str)``.
    """
    import polars as pl

    factor_formula = " * ".join(factor_names)
    formula        = f"score ~ {factor_formula} + (1|input)"

    # pymer4 0.9 requires a Polars DataFrame.
    df_pl = pl.from_pandas(df_pandas)

    model = lmer(formula, data=df_pl)

    # Register factor columns so R uses treatment coding with a stable
    # reference level (first level alphabetically).
    factor_levels = {
        f: sorted(df_pandas[f].dropna().unique().tolist(), key=str)
        for f in factor_names
    }
    model.set_factors(factor_levels)
    model.fit()
    return model, formula


def _extract_factor_tests_pymer4(
    model: Any,
    factor_names: list[str],
) -> "pd.DataFrame":
    """Extract per-term Wald tests from a pymer4 factorial LMM via ``car::Anova``.

    Requires the R package *car*.  Raises ``RuntimeError`` if *car* is not
    installed so callers can surface the requirement clearly.

    Columns: ``term``, ``statistic``, ``df``, ``p_value``.
    """
    import pandas as pd
    from rpy2.robjects.packages import importr, PackageNotInstalledError

    try:
        car = importr("car")
    except PackageNotInstalledError as exc:
        raise RuntimeError(
            "The R package 'car' is required for factor Wald tests when using "
            "backend='pymer4'. Install it in R with: install.packages('car')"
        ) from exc

    anova_r = car.Anova(model.r_model, type=3)

    terms     = list(anova_r.rownames)
    col_names = list(anova_r.colnames)

    # car::Anova type-3 columns: "Chisq", "Df", "Pr(>Chisq)"
    chisq_col = next(
        (c for c in col_names if "chisq" in c.lower() or c.lower() in ("f", "f value")),
        col_names[0],
    )
    df_col = next(
        (c for c in col_names if c.lower() in ("df", "numdf", "num df")),
        col_names[1] if len(col_names) > 1 else col_names[0],
    )
    p_col = next(
        (c for c in col_names if "pr" in c.lower() or "p.val" in c.lower()),
        col_names[-1],
    )

    chisq_vals = list(anova_r.rx2(chisq_col))
    df_vals    = list(anova_r.rx2(df_col))
    p_vals     = list(anova_r.rx2(p_col))

    rows = []
    for term, stat, df_v, p_v in zip(terms, chisq_vals, df_vals, p_vals):
        if term in ("(Intercept)", "Intercept"):
            continue
        rows.append({
            "term":      str(term),
            "statistic": float(stat),
            "df":        float(df_v),
            "p_value":   float(p_v),
        })
    return pd.DataFrame(rows, columns=["term", "statistic", "df", "p_value"])


def _extract_marginal_means_pymer4(
    model: Any,
    df_pandas: "pd.DataFrame",
    factor_names: list[str],
    ci: float,
) -> dict[str, "pd.DataFrame"]:
    """Estimated marginal means per factor from a fitted pymer4 factorial LMM.

    Uses the same design-vector approach as :func:`_extract_marginal_means_sm`:
    the model matrix is extracted from R via rpy2, then EMMs are computed
    from the fixed-effect parameters and covariance matrix.
    """
    import pandas as pd
    from itertools import product as iterproduct

    alpha = 1 - ci

    # Conservative t critical value: use the minimum Satterthwaite DF from
    # the per-parameter result_fit as a safe lower bound.
    rf       = model.result_fit
    df_col_n = next(
        (c for c in rf.columns if c.lower() in ("df", "df_error", "ddf", "denomdf")),
        None,
    )
    t_df   = float(rf[df_col_n].min()) if df_col_n else float(len(df_pandas) - len(factor_names) - 1)
    t_crit = float(scipy.stats.t.ppf(1 - alpha / 2, df=t_df))

    exog  = _get_r_model_matrix(model)       # (n_obs, P)
    betas = _get_fe_params_pymer4(model)     # (P,)
    vcov  = _get_vcov(model)                 # (P, P)

    if exog.shape[0] != len(df_pandas):
        df_pandas = df_pandas.dropna(subset=["score"]).reset_index(drop=True)

    emm_per_factor: dict[str, pd.DataFrame] = {}
    for focal in factor_names:
        other            = [f for f in factor_names if f != focal]
        focal_levels     = sorted(df_pandas[focal].dropna().unique().tolist(), key=str)
        other_level_sets = [
            sorted(df_pandas[f].dropna().unique().tolist(), key=str)
            for f in other
        ]
        other_combos = list(iterproduct(*other_level_sets)) if other else [()]

        rows = []
        for level in focal_levels:
            cell_vecs = []
            for combo in other_combos:
                mask = (df_pandas[focal] == level).values
                for f, v in zip(other, combo):
                    mask = mask & (df_pandas[f] == v).values
                if mask.any():
                    cell_vecs.append(exog[mask].mean(axis=0))

            if not cell_vecs:
                continue

            c   = np.mean(cell_vecs, axis=0)
            emm = float(c @ betas)
            se  = float(np.sqrt(max(float(c @ vcov @ c), 0.0)))
            rows.append({
                "level":   level,
                "mean":    emm,
                "se":      se,
                "ci_low":  emm - t_crit * se,
                "ci_high": emm + t_crit * se,
            })

        emm_per_factor[focal] = pd.DataFrame(rows)

    return emm_per_factor


def _build_factorial_lmm_info_pymer4(
    model: Any,
    factor_names: list[str],
    n_obs: int,
    formula: str,
    ci: float,
    df_pandas: "pd.DataFrame",
) -> FactorialLMMInfo:
    """Build a :class:`FactorialLMMInfo` from a fitted pymer4 factorial model."""
    sigma_input, sigma_resid = _extract_variance_components(model)
    icc       = _compute_icc(sigma_input, sigma_resid)
    converged = True
    status    = getattr(model, "convergence_status", None)
    if status is not None and "TRUE" not in str(status).upper():
        converged = False

    factor_tests   = _extract_factor_tests_pymer4(model, factor_names)
    marginal_means = _extract_marginal_means_pymer4(model, df_pandas, factor_names, ci)

    return FactorialLMMInfo(
        icc=icc,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
        n_obs=n_obs,
        formula=formula,      # already contains (1|input)
        converged=converged,
        factor_names=list(factor_names),
        factor_tests=factor_tests,
        marginal_means=marginal_means,
    )


def _lmm_to_pairwise_factorial_pymer4(
    model: Any,
    labels: list[str],
    df_pandas: "pd.DataFrame",
    cell_means_2d: np.ndarray,
    ci: float,
    correction: str,
) -> PairwiseMatrix:
    """Pairwise Wald contrasts for a pymer4 factorial LMM via the delta method.

    Mirrors :func:`_lmm_to_pairwise_factorial_sm` using the R model matrix
    (via rpy2) and the fixed-effect covariance matrix from pymer4.
    """
    alpha = 1 - ci
    N     = len(labels)

    design_vecs = _get_template_design_vectors_pymer4(model, df_pandas, labels)
    betas       = _get_fe_params_pymer4(model)
    vcov        = _get_vcov(model)

    # Conservative DF: minimum Satterthwaite DF across all fixed-effect parameters.
    rf       = model.result_fit
    df_col_n = next(
        (c for c in rf.columns if c.lower() in ("df", "df_error", "ddf", "denomdf")),
        None,
    )
    df_val = float(rf[df_col_n].min()) if df_col_n else float(len(df_pandas) - N - 1)
    t_crit = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))

    results: dict[tuple[str, str], PairedDiffResult] = {}
    pairs:   list[tuple[str, str]] = []

    for idx_a in range(N):
        for idx_b in range(idx_a + 1, N):
            label_a, label_b = labels[idx_a], labels[idx_b]

            c        = design_vecs[idx_a] - design_vecs[idx_b]
            estimate = float(c @ betas)
            var_est  = float(c @ vcov @ c)
            se       = float(np.sqrt(max(var_est, 0.0)))
            t_stat   = estimate / se if se > 0 else 0.0
            p_val    = float(2 * scipy.stats.t.sf(abs(t_stat), df=df_val))
            ci_low   = estimate - t_crit * se
            ci_high  = estimate + t_crit * se

            per_input_diffs = cell_means_2d[idx_a] - cell_means_2d[idx_b]
            n_complete      = int(np.sum(~np.isnan(per_input_diffs)))
            std_diff        = float(np.nanstd(per_input_diffs, ddof=1)) if n_complete > 1 else 0.0

            res = PairedDiffResult(
                template_a=label_a,
                template_b=label_b,
                point_diff=estimate,
                std_diff=std_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_val,
                test_method=f"factorial lmm wald (pymer4, df={df_val:.0f})",
                n_inputs=n_complete,
                per_input_diffs=per_input_diffs,
                n_runs=1,
                statistic="mean",
            )
            results[(label_a, label_b)] = res
            pairs.append((label_a, label_b))

    if not results:
        raise RuntimeError("Factorial LMM (pymer4) produced no usable pairwise contrasts.")

    _resolved_correction = _apply_pvalue_correction(results, pairs, correction, n_groups=len(labels))
    return PairwiseMatrix(labels=labels, results=results, correction_method=_resolved_correction)


def _lmm_analyze_factorial_pymer4(
    result: Any,
    *,
    reference: str,
    ci: float,
    correction: str,
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    n_sim: int,
    rng: np.random.Generator,
) -> tuple[
    PairwiseMatrix,
    RankDistribution,
    RobustnessResult,
    Optional[SeedVarianceResult],
    FactorialLMMInfo,
]:
    """Full factorial LMM pipeline — pymer4/lme4 backend.

    Fits ``score ~ F1 * F2 * ... + (1|input)`` using R's lme4 via pymer4.
    Pairwise contrasts and advantages are computed from the R model matrix
    (extracted via rpy2) and the fixed-effect covariance matrix, mirroring
    the statsmodels factorial path.  Factor Wald tests require the R *car*
    package (``car::Anova``); if *car* is unavailable, ``factor_tests`` is
    returned as an empty DataFrame with a warning.
    """
    lmer = _require_pymer4()

    M      = result.n_inputs
    labels = result.template_labels
    inputs = result.input_labels

    template_factors = result.template_factors
    factor_names     = list(template_factors.columns)

    cell_means_2d = result.get_2d_scores()
    run_scores    = result.get_run_scores()

    robustness = robustness_metrics(run_scores, labels, failure_threshold=failure_threshold)
    seed_var: Optional[SeedVarianceResult] = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    lmm_scores = run_scores if result.is_seeded else cell_means_2d
    df_pandas  = _scores_to_long_df_factorial_pandas(lmm_scores, labels, inputs, template_factors)
    n_obs      = len(df_pandas)

    model, formula = _fit_factorial_lmm_pymer4(df_pandas, factor_names, lmer)

    if not model.fitted:
        raise RuntimeError(
            "Factorial LMM (pymer4) failed to fit. Check that scores have "
            "sufficient variance and that factor labels are well-formed."
        )

    lmm_info = _build_factorial_lmm_info_pymer4(
        model, factor_names, n_obs, formula, ci, df_pandas
    )

    if not lmm_info.converged:
        warnings.warn(
            "The factorial LMM optimizer (pymer4) did not converge. "
            "Results may be unreliable. Consider using method='auto' "
            "or simplifying the model.",
            UserWarning,
            stacklevel=4,
        )

    pairwise = _lmm_to_pairwise_factorial_pymer4(
        model, labels, df_pandas, cell_means_2d, ci, correction
    )
    design_vecs              = _get_template_design_vectors_pymer4(model, df_pandas, labels)
    template_means           = design_vecs @ _get_fe_params_pymer4(model)
    sigma_input, sigma_resid = _extract_variance_components(model)
    rank_dist = _simulate_rank_dist(
        template_means, sigma_input, sigma_resid, M, labels, n_sim, rng
    )

    return pairwise, rank_dist, robustness, seed_var, lmm_info


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lmm_analyze(
    result: Any,
    *,
    backend: Literal["statsmodels", "pymer4"] = "statsmodels",
    reference: str = "grand_mean",
    ci: float = 0.95,
    correction: str = "fdr_bh",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    n_sim: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> "tuple[PairwiseMatrix, RankDistribution, RobustnessResult, Optional[SeedVarianceResult], LMMInfo | FactorialLMMInfo]":
    """Run the full LMM analysis pipeline on a ``BenchmarkResult``.

    Fits ``score ~ template + (1|input)`` on the cell-mean scores and maps
    the model output to the same result types as the bootstrap path.

    Parameters
    ----------
    result : BenchmarkResult
        The benchmark data to analyse.
    backend : str
        Which fitting library to use: ``'statsmodels'`` (default, pure Python,
        no R required) or ``'pymer4'`` (wraps R/lme4, requires pymer4 + R with
        lme4 and emmeans installed).  Both produce Wald CIs from the
        fixed-effect covariance matrix.  pymer4 uses per-contrast Satterthwaite
        DFs via emmeans; statsmodels uses a single conservative residual DF.
    reference : str
        Retained for API compatibility. Ignored in robustness-only mode.
    ci : float
        Confidence level for Wald intervals (default 0.95).
    correction : str
        Multiple-comparisons correction: ``'fdr_bh'`` (default),
        ``'holm'``, ``'bonferroni'``, or ``'none'``.
    spread_percentiles : tuple[float, float]
        Retained for API compatibility. Ignored in robustness-only mode.
    failure_threshold : float, optional
        Threshold for failure-rate computation in robustness metrics.
    n_sim : int
        Number of parametric simulations for the rank distribution
        (default 10,000).  Analogous to ``n_bootstrap`` in the bootstrap path.
    rng : np.random.Generator, optional
        Random number generator for the rank simulation.

    Returns
    -------
    tuple
        ``(pairwise, rank_dist, robustness, seed_var, lmm_info)``
        where types match those returned by the bootstrap analysis path.

    Raises
    ------
    ImportError
        If the selected backend's dependencies are not installed.
    RuntimeError
        If the model fails to converge or returns unusable contrasts.
    ValueError
        If ``backend`` is not ``'statsmodels'`` or ``'pymer4'``.
    """
    # Normalize as every other engine entry point does (unpaired.py,
    # resampling.py): callers may pass an int seed, None, or a Generator.
    # This path previously only ever saw None or a Generator because compare()
    # left rng unset by default; it now defaults to an int seed.
    rng = np.random.default_rng(rng)
    if backend not in ("statsmodels", "pymer4"):
        raise ValueError(f"backend must be 'statsmodels' or 'pymer4', got {backend!r}")

    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Factorial path — activated when BenchmarkResult.template_factors is set.
    # ------------------------------------------------------------------
    if getattr(result, "template_factors", None) is not None:
        _factorial_kwargs = dict(
            reference=reference,
            ci=ci,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            n_sim=n_sim,
            rng=rng,
        )
        if backend == "pymer4":
            return _lmm_analyze_factorial_pymer4(result, **_factorial_kwargs)
        return _lmm_analyze_factorial_sm(result, **_factorial_kwargs)

    N = result.n_templates
    M = result.n_inputs

    if M < 5:
        warnings.warn(
            f"LMM analysis with only M={M} inputs may be unreliable. "
            "Consider using the default bootstrap method (method='auto') "
            "or collecting more benchmark inputs.",
            UserWarning,
            stacklevel=3,
        )

    if result.has_missing:
        n_missing = int(np.sum(np.isnan(result.get_2d_scores())))
        n_total = N * M
        warnings.warn(
            f"scores contain {n_missing} missing (NaN) cell(s) out of "
            f"{n_total} total ({100 * n_missing / n_total:.1f}%). "
            "LMM analysis will use available observations under the MAR "
            "(Missing At Random) assumption. Results may be biased if "
            "missingness is related to true score values (MNAR).",
            UserWarning,
            stacklevel=3,
        )

    # Use cell means for model fitting; keep run scores for seed_var.
    cell_means_2d = result.get_2d_scores()   # (N, M)
    run_scores    = result.get_run_scores()  # (N, M, R)
    labels        = result.template_labels
    inputs        = result.input_labels

    robustness = robustness_metrics(run_scores, labels, failure_threshold=failure_threshold)

    seed_var: Optional[SeedVarianceResult] = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    # ------------------------------------------------------------------
    # statsmodels path
    # ------------------------------------------------------------------
    if backend == "statsmodels":
        _require_statsmodels()
        # When runs are available, fit on individual observations so that
        # seed variance enters the residual and inflates fixed-effect CIs
        # appropriately.  Fall back to cell means when R < 3.
        lmm_scores = run_scores if result.is_seeded else cell_means_2d
        df    = _scores_to_long_df_pandas(lmm_scores, labels, inputs)
        n_obs = len(df)
        sm_result = _fit_lmm_sm(df, labels)

        lmm_info = _build_lmm_info_sm(sm_result, n_obs)
        if not lmm_info.converged:
            warnings.warn(
                "The statsmodels LMM optimizer did not converge. "
                "Results may be unreliable. Consider using the bootstrap method "
                "(method='auto') or simplifying the model.",
                UserWarning,
                stacklevel=3,
            )

        pairwise = _lmm_to_pairwise_sm(sm_result, labels, cell_means_2d, ci, correction)
        template_means           = _extract_template_means_sm(sm_result, labels)
        sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
        rank_dist = _simulate_rank_dist(
            template_means, sigma_input, sigma_resid, M, labels, n_sim, rng
        )

        return pairwise, rank_dist, robustness, seed_var, lmm_info

    # ------------------------------------------------------------------
    # pymer4 path (original implementation)
    # ------------------------------------------------------------------
    lmer = _require_pymer4()
    df    = _scores_to_long_df(cell_means_2d, labels, inputs)
    n_obs = len(df)
    model = _fit_lmm(df, lmer, labels)

    if not model.fitted:
        raise RuntimeError(
            "LMM failed to fit. Check that scores have sufficient variance "
            "across inputs and that template labels are well-formed."
        )

    pairwise  = _lmm_to_pairwise(model, labels, cell_means_2d, ci, correction)
    rank_dist = _lmm_to_rank_dist(model, labels, cell_means_2d, n_sim, rng)
    lmm_info  = _build_lmm_info(model, n_obs)

    if not lmm_info.converged:
        warnings.warn(
            "The LMM optimizer reported a convergence warning or singular fit. "
            "Results may be unreliable. Consider using the bootstrap method "
            "(method='auto') or simplifying the model.",
            UserWarning,
            stacklevel=3,
        )

    return pairwise, rank_dist, robustness, seed_var, lmm_info
