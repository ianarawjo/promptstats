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
from .ranking import PointAdvantageResult, RankDistribution
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


def _t_crit_from_result_fit(model: Any, alpha: float) -> float:
    """Return the conservative t critical value from the fixed-effects DFs."""
    rf = model.result_fit
    df_col = next((c for c in ["df", "DF", "df_error", "Df", "Ddf"] if c in rf.columns), None)
    if df_col is not None:
        min_df = float(rf[df_col].min())
        return float(scipy.stats.t.ppf(1 - alpha / 2, df=min_df))
    # Fall back to standard normal (conservative for large N)
    return float(scipy.stats.norm.ppf(1 - alpha / 2))


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

    # Apply multiple-comparisons correction
    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)
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
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


# ---------------------------------------------------------------------------
# Mean advantage
# ---------------------------------------------------------------------------

def _build_advantage_contrast_matrix(N: int, ref_idx: Optional[int]) -> np.ndarray:
    """Build the (N × N) contrast matrix L for template advantages.

    Each row L[i] is a vector of coefficients such that::

        advantage[i] = L[i] @ beta

    where ``beta = [intercept, β₁, …, β_{N-1}]`` uses treatment coding
    with ``template_labels[0]`` as the reference level.

    Parameters
    ----------
    N : int
        Number of templates.
    ref_idx : int or None
        Index of the reference template, or ``None`` for grand-mean reference.
    """
    L = np.zeros((N, N))

    if ref_idx is None:
        # Grand-mean reference: advantage_i = μ_i - (1/N) * Σ_k μ_k
        #
        # μ₀ = β₀,   μⱼ = β₀ + βⱼ  (j > 0)
        # grand_mean = β₀ + (1/N) * Σ_{j>0} βⱼ
        #
        # advantage_0 = -(1/N) * Σ_{j>0} βⱼ  →  L[0, 1:] = -1/N
        # advantage_i = βᵢ - (1/N) * Σ_{j>0} βⱼ
        #             →  L[i, i] = 1 - 1/N,  L[i, j≠i, j>0] = -1/N
        L[0, 1:] = -1.0 / N
        for i in range(1, N):
            L[i, 1:] = -1.0 / N
            L[i, i]  =  1.0 - 1.0 / N
    else:
        # Specific-template reference: advantage_i = μ_i - μ_{ref}
        #
        # Cases (treatment coding, ref level = template 0):
        #   i == ref_idx               → advantage = 0   (L[i] = 0)
        #   i == 0, ref_idx > 0        → -β_{ref}
        #   i > 0,  ref_idx == 0       → β_i
        #   i > 0,  ref_idx > 0, i≠ref → β_i - β_{ref}
        for i in range(N):
            if i == ref_idx:
                continue  # all zeros
            if i == 0:
                L[i, ref_idx] = -1.0
            elif ref_idx == 0:
                L[i, i] = 1.0
            else:
                L[i, i]       =  1.0
                L[i, ref_idx] = -1.0

    return L


def _lmm_to_mean_advantage(
    model: Any,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    spread_percentiles: tuple[float, float],
    reference: str,
) -> PointAdvantageResult:
    """Compute mean advantages from LMM fixed effects using the delta method.

    Point estimates are the fitted LMM marginal means (equal to raw cell-mean
    averages in the balanced design).  Confidence intervals use Wald intervals
    derived from the fixed-effects variance–covariance matrix via the delta
    method, so they correctly account for the correlated estimation of
    treatment effects.

    The intrinsic spread bands (``spread_low`` / ``spread_high``) are still
    computed from raw cell-mean advantages, exactly as in the bootstrap path.

    Notes
    -----
    ``n_bootstrap=0`` on the returned ``MeanAdvantageResult`` signals that
    these are parametric Wald intervals, not bootstrap intervals.
    """
    N = len(labels)
    alpha = 1 - ci

    rf = model.result_fit
    est_col = _col_pl(rf, ["estimate", "Estimate", "coefficient", "Coefficient"])
    betas = rf[est_col].to_numpy()   # (N,): [intercept, β₁, …, β_{N-1}]

    # Fitted template means
    template_means = np.empty(N)
    template_means[0] = betas[0]
    template_means[1:] = betas[0] + betas[1:]

    # Reference
    if reference == "grand_mean":
        ref_value = template_means.mean()
        ref_idx   = None
        ref_label = "grand_mean"
    else:
        ref_idx   = labels.index(reference)
        ref_value = template_means[ref_idx]
        ref_label = reference

    mean_advantages = template_means - ref_value   # (N,)

    # --- Wald CIs via delta method -----------------------------------------
    vcov = _get_vcov(model)     # (N, N)
    L    = _build_advantage_contrast_matrix(N, ref_idx)   # (N, N)

    # Variance of each advantage: var(L[i] @ beta) = L[i] @ vcov @ L[i].T
    cov_adv  = L @ vcov @ L.T          # (N, N)
    var_adv  = np.diag(cov_adv)        # (N,)
    se_adv   = np.sqrt(np.maximum(var_adv, 0.0))

    t_crit   = _t_crit_from_result_fit(model, alpha)
    ci_low   = mean_advantages - t_crit * se_adv
    ci_high  = mean_advantages + t_crit * se_adv

    # --- Raw cell-mean advantages for spread bands -------------------------
    if reference == "grand_mean":
        cell_ref = np.nanmean(cell_means_2d, axis=0)   # (M,) — NaN-safe grand mean
    else:
        cell_ref = cell_means_2d[ref_idx]               # (M,)

    raw_advantages = cell_means_2d - cell_ref[np.newaxis, :]  # (N, M)
    spread_low  = np.nanpercentile(raw_advantages, spread_percentiles[0], axis=1)
    spread_high = np.nanpercentile(raw_advantages, spread_percentiles[1], axis=1)

    return PointAdvantageResult(
        labels=labels,
        point_advantages=mean_advantages,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=raw_advantages,
        n_bootstrap=0,           # 0 = parametric Wald, not bootstrap
        spread_percentiles=spread_percentiles,
        statistic="mean",        # LMM is a mean-based model
    )


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

def _build_lmm_info(model: Any, n_obs: int) -> LMMInfo:
    """Extract ``LMMInfo`` from a fitted pymer4 ``lmer`` model."""
    sigma_input, sigma_resid = _extract_variance_components(model)

    var_input = sigma_input ** 2
    var_resid = sigma_resid ** 2
    total_var = var_input + var_resid
    icc = var_input / total_var if total_var > 0 else 0.0

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
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
) -> "pd.DataFrame":
    """Convert an ``(N, M)`` cell-mean score matrix to a long-form pandas DataFrame.

    Returns a DataFrame with columns ``'template'``, ``'input'``, ``'score'``.
    The ``'template'`` column is a ``pd.Categorical`` with ``template_labels[0]``
    as the first (reference) category.  NaN rows are dropped.
    """
    import pandas as pd

    N, M = scores_2d.shape
    templates   = np.repeat(template_labels, M).tolist()
    inputs      = np.tile(input_labels, N).tolist()
    scores_flat = scores_2d.ravel().tolist()

    df = pd.DataFrame({"template": templates, "input": inputs, "score": scores_flat})
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

    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)
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
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


def _lmm_to_mean_advantage_sm(
    sm_result: Any,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    spread_percentiles: tuple[float, float],
    reference: str,
) -> PointAdvantageResult:
    """Compute mean advantages from statsmodels fixed effects via delta method."""
    N     = len(labels)
    alpha = 1 - ci

    template_means = _extract_template_means_sm(sm_result, labels)
    vcov           = _get_vcov_sm(sm_result, labels)
    df_val         = float(sm_result.df_resid)

    if reference == "grand_mean":
        ref_value = template_means.mean()
        ref_idx   = None
        ref_label = "grand_mean"
    else:
        ref_idx   = labels.index(reference)
        ref_value = template_means[ref_idx]
        ref_label = reference

    mean_advantages = template_means - ref_value

    L       = _build_advantage_contrast_matrix(N, ref_idx)
    cov_adv = L @ vcov @ L.T
    var_adv = np.diag(cov_adv)
    se_adv  = np.sqrt(np.maximum(var_adv, 0.0))

    t_crit  = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))
    ci_low  = mean_advantages - t_crit * se_adv
    ci_high = mean_advantages + t_crit * se_adv

    if reference == "grand_mean":
        cell_ref = np.nanmean(cell_means_2d, axis=0)
    else:
        cell_ref = cell_means_2d[ref_idx]

    raw_advantages = cell_means_2d - cell_ref[np.newaxis, :]
    spread_low  = np.nanpercentile(raw_advantages, spread_percentiles[0], axis=1)
    spread_high = np.nanpercentile(raw_advantages, spread_percentiles[1], axis=1)

    return PointAdvantageResult(
        labels=labels,
        point_advantages=mean_advantages,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=raw_advantages,
        n_bootstrap=0,
        spread_percentiles=spread_percentiles,
        statistic="mean",
    )


def _build_lmm_info_sm(sm_result: Any, n_obs: int) -> LMMInfo:
    """Extract ``LMMInfo`` from a fitted statsmodels ``MixedLMResults``."""
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)

    var_input = sigma_input ** 2
    var_resid = sigma_resid ** 2
    total_var = var_input + var_resid
    icc       = var_input / total_var if total_var > 0 else 0.0

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
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
    template_factors: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build a long-form pandas DataFrame with factor columns appended.

    Calls :func:`_scores_to_long_df_pandas` then left-joins the factor
    columns from *template_factors*, aligning rows positionally to
    *template_labels*.
    """
    import pandas as pd

    df = _scores_to_long_df_pandas(scores_2d, template_labels, input_labels)

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

    var_input = sigma_input ** 2
    var_resid = sigma_resid ** 2
    total_var = var_input + var_resid
    icc       = var_input / total_var if total_var > 0 else 0.0
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

    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)
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
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


def _lmm_to_mean_advantage_factorial_sm(
    sm_result: Any,
    labels: list[str],
    df_pandas: "pd.DataFrame",
    cell_means_2d: np.ndarray,
    ci: float,
    spread_percentiles: tuple[float, float],
    reference: str,
) -> PointAdvantageResult:
    """Mean advantages for a factorial LMM via the delta method.

    Cell means come from the fitted fixed effects (``design_vec_i @ betas``).
    CIs use the delta method applied to the contrast matrix
    ``C[i] = design_vec_i − reference_vec``.
    """
    N     = len(labels)
    alpha = 1 - ci

    design_vecs = _get_template_design_vectors(sm_result, df_pandas, labels)
    betas       = sm_result.fe_params.to_numpy()
    vcov        = _get_fe_vcov_sm(sm_result)
    df_val      = float(sm_result.df_resid)

    template_means = design_vecs @ betas  # (N,)

    if reference == "grand_mean":
        ref_value = template_means.mean()
        ref_label = "grand_mean"
        ref_vec   = design_vecs.mean(axis=0)  # (P,)
    else:
        ref_idx   = labels.index(reference)
        ref_value = template_means[ref_idx]
        ref_label = reference
        ref_vec   = design_vecs[ref_idx]

    mean_advantages = template_means - ref_value  # (N,)

    # Delta method: var(advantage_i) = (design_i - ref_vec) @ vcov @ (design_i - ref_vec)
    C       = design_vecs - ref_vec[np.newaxis, :]   # (N, P)
    cov_adv = C @ vcov @ C.T                          # (N, N)
    var_adv = np.diag(cov_adv)
    se_adv  = np.sqrt(np.maximum(var_adv, 0.0))

    t_crit  = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))
    ci_low  = mean_advantages - t_crit * se_adv
    ci_high = mean_advantages + t_crit * se_adv

    if reference == "grand_mean":
        cell_ref = np.nanmean(cell_means_2d, axis=0)
    else:
        cell_ref = cell_means_2d[labels.index(reference)]

    raw_advantages = cell_means_2d - cell_ref[np.newaxis, :]
    spread_low  = np.nanpercentile(raw_advantages, spread_percentiles[0], axis=1)
    spread_high = np.nanpercentile(raw_advantages, spread_percentiles[1], axis=1)

    return PointAdvantageResult(
        labels=labels,
        point_advantages=mean_advantages,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=raw_advantages,
        n_bootstrap=0,
        spread_percentiles=spread_percentiles,
        statistic="mean",
    )


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
    PointAdvantageResult,
    RankDistribution,
    RobustnessResult,
    Optional[SeedVarianceResult],
    FactorialLMMInfo,
]:
    """Full factorial LMM pipeline (statsmodels backend only).

    Fits ``score ~ C(F1) * C(F2) * ... + (1|input)`` on cell-mean scores
    and returns the same tuple as :func:`lmm_analyze`:
    ``(pairwise, mean_adv, rank_dist, robustness, seed_var, factorial_lmm_info)``.
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

    df       = _scores_to_long_df_factorial_pandas(cell_means_2d, labels, inputs, template_factors)
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
    mean_adv = _lmm_to_mean_advantage_factorial_sm(
        sm_result, labels, df, cell_means_2d, ci, spread_percentiles, reference
    )

    design_vecs              = _get_template_design_vectors(sm_result, df, labels)
    template_means           = design_vecs @ sm_result.fe_params.to_numpy()
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    rank_dist = _simulate_rank_dist(
        template_means, sigma_input, sigma_resid, M, labels, n_sim, rng
    )

    return pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lmm_analyze(
    result: Any,
    *,
    backend: Literal["statsmodels", "pymer4"] = "statsmodels",
    reference: str = "grand_mean",
    ci: float = 0.95,
    correction: str = "holm",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    n_sim: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> "tuple[PairwiseMatrix, PointAdvantageResult, RankDistribution, RobustnessResult, Optional[SeedVarianceResult], LMMInfo | FactorialLMMInfo]":
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
        Reference for mean advantage: ``'grand_mean'`` or a template label.
    ci : float
        Confidence level for Wald intervals (default 0.95).
    correction : str
        Multiple-comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band (default ``(10, 90)``).
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
        ``(pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info)``
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
    if backend not in ("statsmodels", "pymer4"):
        raise ValueError(f"backend must be 'statsmodels' or 'pymer4', got {backend!r}")

    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Factorial path — activated when BenchmarkResult.template_factors is set.
    # Currently only the statsmodels backend is supported for factorial models.
    # ------------------------------------------------------------------
    if getattr(result, "template_factors", None) is not None:
        if backend == "pymer4":
            warnings.warn(
                "Factorial LMM analysis currently supports only backend='statsmodels'. "
                "Switching to statsmodels for this analysis.",
                UserWarning,
                stacklevel=2,
            )
        return _lmm_analyze_factorial_sm(
            result,
            reference=reference,
            ci=ci,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            n_sim=n_sim,
            rng=rng,
        )

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
        df    = _scores_to_long_df_pandas(cell_means_2d, labels, inputs)
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
        mean_adv = _lmm_to_mean_advantage_sm(
            sm_result, labels, cell_means_2d, ci, spread_percentiles, reference
        )
        template_means           = _extract_template_means_sm(sm_result, labels)
        sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
        rank_dist = _simulate_rank_dist(
            template_means, sigma_input, sigma_resid, M, labels, n_sim, rng
        )

        return pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info

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
    mean_adv  = _lmm_to_mean_advantage(
        model, labels, cell_means_2d, ci, spread_percentiles, reference
    )
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

    return pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info
