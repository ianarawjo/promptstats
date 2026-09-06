"""Where does false-positive risk peak, as a function of judge bias, for every IRR metric?

The paper claims that on Likert data the uncorrected false-positive rate is
worst when a judge looks "almost perfect" by quadratic-weighted kappa or
ICC(2,1) -- around 0.80-0.90. That claim currently rests on the alignment-
bucketed view derived from cases/pvalues.py's --factorial-check run, which
crosses bias_magnitude x llm_noise but then, via _alignment_regime, collapses
the bias axis to a single curve: "bias_present" is bias_label == "severe"
ALONE (bias_delta = 0.30 population SDs). So the plotted x-axis moves purely
through llm_noise at ONE fixed bias.

That invites two obvious reviewer questions the factorial view cannot answer:

  1. Does the peak sit at 0.80-0.90 for ANY judge bias, or only for the one
     magnitude that sweep happens to fix?
  2. Is 0.80-0.90 specific to weighted kappa / ICC(2,1), or does every IRR
     metric peak in the same place?

This case answers both by sweeping bias_delta x llm_noise as a full 2-D grid
and reporting, for EVERY metric in _alignment_metric_dict's panel (quadratic
and linear weighted kappa, Cohen's kappa, Pearson, Spearman, Kendall tau-b,
ICC(2,1), Lin's CCC, Krippendorff's alpha, Gwet's AC1, PABAK, percent
agreement), the metric value at which the uncorrected false-positive rate
peaks.

It is deliberately a SEPARATE case rather than another flag on pvalues.py:

  * pvalues.py is already ~15k lines, and this needs none of its GLM/
    heatmap/label-efficiency machinery.
  * The paper's existing figures come from --factorial-check. Adding a bias
    axis there would change the scenario count and therefore the numbers
    already reported. This case shares pvalues.py's simulation driver
    (run_ppi_comparison_simulation) and synthetic.py's judge model
    (generate_judge_bias_cell / measure_judge_alignment) so results are
    directly comparable, while leaving every existing sweep untouched.

WHAT THE GRID IS. The cross product of four axes:

  * PANELS: (eval_type, likert_max) -- binary, continuous, likert-5, likert-7.
    One figure each. Likert appears twice because the number of scale points
    is what sets the rounding threshold below, so it is not a cosmetic choice.
  * VARIANTS: one-factor-at-a-time perturbations of the fixed baseline (sample
    size, heavy-tailed judge error, scale compression, an item-level confound,
    truth ICC). Every one of these moved the peak by more than the estimator's
    own seed-to-seed noise when probed, which is why they are swept rather
    than assumed away.
  * judge bias: BIAS_FRACS, standardized as a fraction of the eval type's own
    population SD, spanning well below to well above the discretization
    threshold described next.
  * judge noise: NOISE_FRACS, on the same standardized footing.

Bias and noise are swept as standardized FRACTIONS and converted to the raw
absolute offsets JudgeBiasSource takes, per panel, via _jb_bias_magnitude --
the same conversion build_ppi_factorial_sources performs. Handing a fraction
straight to bias_delta would mean wildly different real severities per eval
type (a raw 0.30 is 0.26 SD on Likert but 2.5 SD on continuous). Binary
bypasses this entirely: its judge model is a confusion matrix, so its grids
are flip probabilities used as-is, and variants that only touch slope or
noise_family are dropped there because that model implements neither.

every cell at effect_size=0 (a true null), so rejects_llm_only / n_reps IS
the uncorrected Type-I rate directly. rejects_ppi is carried alongside as the
control: PPI correction should hold nominal everywhere on this grid, and a
cell where it does not is a bug in this case, not a finding.

Each subplot draws every variant as a faint thin line and their mean as a bold
line carrying the peak marker, so the reader sees the spread rather than one
summary curve standing in for a sweep that disagrees with itself.

RUNTIME. The default grid is a multi-hour job and prints an estimate before
starting. Measured cost is ``3.14 s/cell + 8.5 ms per bootstrap draw`` at
reps=1000 on 15 workers, so at the default n_boot the bootstrap is under 10%
and the FIXED term dominates. The levers, in order: `reps` (linear), the grid
size (--panels / --variants), and only then --bootstrap-n. `--variants
baseline` alone reproduces the original single-curve sweep.

Alignment is cheap (~28 ms/cell), deduplicated across variants that share a
judge model (see _ALIGN_IRRELEVANT), and measured in parallel -- it used to be
a serial loop and was the half of the runtime --workers did not scale.

The per-cell CSV carries raw reject COUNTS as well as rates, plus every swept
parameter, so a later re-plot or re-analysis never has to re-run the sweep.

THE DISCRETIZATION THRESHOLD, and why it matters for reading the output.
measure_judge_alignment rounds Likert judge scores to the integer grid before
computing any metric (see its closing note), and the human truth is already
integer. A differential bias smaller than half a scale point therefore rounds
away completely at zero noise: the judge becomes EXACTLY the truth, every
metric reads 1.0, and the false-positive rate returns to nominal. The hump
exists because noise must be large enough to carry biased values across
rounding boundaries, yet small enough not to swamp the signal -- so the peak's
location is pinned by the 0.5-point grid geometry rather than by bias, which
is why it moves so little across sub-threshold bias magnitudes.

Half a scale point is an ABSOLUTE quantity, so as a standardized fraction it
depends on the scale width: 0.437 on a 5-point Likert scale (SD 1.1447) and
0.298 on a 7-point one (SD 1.679). BIAS_FRACS straddles both deliberately,
and that is where
the qualitative behaviour changes: above it the metric can no longer reach
1.0 at any noise level, the "looks perfect but isn't" regime disappears, and
the false-positive rate is high everywhere instead of humped. Continuous data
is never rounded and so has no such threshold -- its curves rise
monotonically, which is the contrast the paper already reports.

Run:
    # the full default grid (4 panels x 11 variants) -- multi-hour
    python -m simulations.harness.cli --workers 15 irr_peak
    # one panel, baseline only -- minutes
    python -m simulations.harness.cli --workers 15 irr_peak \\
        --panels likert5 --variants baseline --reps 300
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np

from . import CaseResult
from ..scenarios import JudgeBiasSource
from ..scenarios.synthetic import (
    EVAL_TYPE_POPULATION_SD,
    _jb_bias_magnitude,
    measure_judge_alignment,
)
from .pvalues import (
    _COMPARISON_METHODS,
    pool_ppi_comparison_across_methods,
    run_ppi_comparison_simulation,
)

CASE_NAME = "irr_peak"

_ALPHA = 0.05

BIAS_FRACS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.44, 0.60, 0.90, 1.30)
"""Judge bias as a FRACTION of the eval type's own population SD -- a
Cohen's-d-style standardized severity, the convention every bias tier and
noise level in this harness shares (see _jb_bias_magnitude).

These are NOT the values handed to JudgeBiasSource. That field is a raw
additive offset on the eval type's native scale, so build_sources converts
each frac with _jb_bias_magnitude(eval_type, frac, scale_bounds=...) exactly
as build_ppi_factorial_sources does. Passing a frac straight through as
bias_delta is a real hazard: on continuous (population SD 0.1206) a raw 0.30
is 2.5 SD and a raw 1.30 is 10.8 SD, which saturates the false-positive rate
at 1.0 across most of the grid and leaves no peak to find.

0.0 is the no-bias control, but only for variants whose judge has no OTHER
source of between-group difference. The confound variants carry
confound_shift_a=1.0, a nuisance covariate that differs between conditions
independently of bias_delta, so they legitimately reject well above nominal at
bias=0 -- that is the confound's whole point, not a mis-specified grid. Every
other variant should sit at nominal there; print_report checks them
separately for exactly this reason.

0.30 is exactly PPI_FACTORIAL_BIAS_MAGNITUDES' "severe", the single value the
paper's existing factorial view plots, so this sweep contains that curve as
one column and can be checked against it.

The levels straddle the Likert rounding threshold on BOTH scales. That
threshold is half a scale point in ABSOLUTE units on any integer scale, which
is a different frac per scale width: 0.5/1.1447 = 0.437 on a 5-point scale,
0.5/1.679 = 0.298 on a 7-point one. Hence both 0.22-0.30 (bracketing the
7-point threshold) and 0.44 (at the 5-point one) are present."""

NOISE_FRACS: tuple[float, ...] = (
    0.025, 0.05, 0.084, 0.137, 0.20, 0.283, 0.36,
    0.46, 0.586, 0.75, 0.95, 1.55, 2.51, 4.08,
)
"""llm_noise as a fraction of the eval type's population SD, converted to an
absolute value per panel the same way BIAS_FRACS is.

NOT the geometric grid PPI_FACTORIAL_NOISE_LEVELS uses. This case pays for
three extra axes (bias, variant, panel), so a uniform 20-point geometric grid
would cost ~40% more runtime to add resolution where nothing happens. Instead
the points are dense through 0.2-1.0, which is where every peak observed so
far actually sits, and sparse in the tails, which only need enough points to
establish the shape:

  - the low end must reach 0.025 because above the rounding threshold the peak
    moves TO the minimum-noise cell (there is no interior peak there);
  - the high end must reach ~4 because the exact-match metrics (percent
    agreement, AC1, PABAK) are non-monotonic in noise and their curve does not
    close until the noise swamps the bias.

Measured peak locations were unchanged between this grid and the 20-point one
at matched settings."""

BINARY_BIAS_LEVELS: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.45, 0.60)
BINARY_NOISE_LEVELS: tuple[float, ...] = (
    0.025, 0.0354, 0.05, 0.0707, 0.10, 0.1414, 0.20, 0.2828, 0.35, 0.40,
)
"""Binary's own grids, in its own units -- used DIRECTLY, with no
_jb_bias_magnitude conversion.

Binary's judge model (_jb_llm_binary) is a confusion-matrix model, not an
additive one: llm_noise is a symmetric flip probability and bias_delta pulls
the two error rates apart. Neither is a multiple of a population SD, and
_jb_bias_magnitude's docstring says outright that binary "is not meant to be
passed here."

The noise grid is PPI_BINARY_NOISE_LEVELS' range and stops at 0.40 for the
reason that constant gives: at 0.50 the judge is a coin flip regardless of
truth, and past it the judge anti-correlates with truth, which is not a
"noisier judge" any more. Reusing the continuous grid here would push seven of
its fourteen points past that line, up to a nonsensical flip probability of
4.08.

The bias grid is anchored on PPI_BINARY_BIAS_MAGNITUDES (moderate 0.10,
severe 0.30) and extends to 0.60 to reach the same "well past severe" regime
the other panels get. At the high-bias/low-noise corner one of the two flip
probabilities clips against [0, 1] -- the harness's own binary factorial
exercises that corner too, so it is a known regime rather than a new one."""

_BINARY_INAPPLICABLE = frozenset({"slope", "noise_family", "contam_frac", "contam_scale"})
"""Judge-model fields _jb_llm_binary does not implement. Its docstring: "slope
and noise_family have no analogue here (not yet meaningful for a plain
flip-probability judge) and are simply not modeled."

Variants that only touch these are therefore silent no-ops on the binary
panel -- they would burn compute to redraw the baseline curve and stack
duplicate lines in every binary subplot, implying a spread that is not real.
build_sources drops them for that panel instead."""

PANELS: tuple[tuple[str, int], ...] = (
    ("binary", 5), ("continuous", 5), ("likert", 5), ("likert", 7),
)
"""(eval_type, likert_max) pairs, one big figure each. likert_max is ignored
for binary/continuous (carried only so one key type covers every panel).

Likert appears TWICE, at 5 and 7 scale points, because the discretization
threshold this whole case is built around is half of ONE scale point -- so the
number of points is not a cosmetic choice, it is the parameter that sets where
the deceptive regime ends. A 5-point and a 7-point judge with identical bias
and noise do not peak at the same agreement level, and 7-point instruments are
common in HCI, so reporting only the 5-point number would be misleading."""


def panel_id(eval_type: str, likert_max: int) -> str:
    """Filename/CSV-safe panel key: 'likert5' vs 'likert7' must not collide."""
    return f"likert{likert_max}" if eval_type == "likert" else eval_type


VARIANTS: tuple[tuple[str, dict], ...] = (
    ("baseline", {}),
    ("n=60", {"n": 60}),
    ("n=200", {"n": 200}),
    ("n=400", {"n": 400}),
    ("contaminated", {"noise_family": "contaminated"}),
    ("slope=0.7", {"slope_a": 0.7}),
    ("slope=0.5", {"slope_a": 0.5}),
    ("confound=0.05", {"confound_weight_frac": 0.05, "confound_truth_corr": 0.3,
                       "confound_shift_a": 1.0}),
    ("confound=0.08", {"confound_weight_frac": 0.08, "confound_truth_corr": 0.3,
                       "confound_shift_a": 1.0}),
    ("icc=0.10", {"icc": 0.10}),
    ("icc=0.50", {"icc": 0.50}),
)
"""One-factor-at-a-time perturbations of _BASE, each sweeping the full
bias x noise grid. NOT a full cross: crossing these axes would be ~200x the
cells for a picture no one can read, and OFAT already answers the question the
variants exist to answer ("does the peak sit where the baseline says, or is
that an artifact of what the baseline happens to fix?").

Why each is here -- every one of these moved the peak by more than the
estimator's own noise (peak-location sd is ~0.02 across seeds) when probed:

  n           the baseline fixes n=100. Sample size barely moves the peak's
              LOCATION but scales its HEIGHT hard (peak false-positive rate
              ~0.11 at n=60 to ~0.42 at n=400 at the same bias), so a single n
              understates how bad the phenomenon gets.
  contaminated  real judges are mostly right and occasionally catastrophically
              wrong, not Gaussian. Since the mechanism is rounding-boundary
              crossings, the SHAPE of the error at fixed variance matters:
              this moves the peak down (~0.90 -> ~0.82).
  slope       scale compression -- judges clustering on middle scores -- is a
              hallmark LLM-judge failure and is NOT a location shift, so it
              hits the correlation metrics differently. It also makes a judge
              WORSE at the same apparent agreement.
  confound    a per-item nuisance covariate (e.g. response length) that
              differs between conditions. JudgeBiasSource's own docstring
              calls this "structurally different" from bias_delta/slope_*,
              which "can only stretch or shift a group as a whole." With NO
              additive bias at all it still reaches a high false-positive rate
              at high agreement -- and unlike additive bias, it does NOT
              return to nominal as agreement approaches its ceiling.
  icc         truth signal-to-noise. Included as a near-null control: it moved
              the peak least of any axis probed."""

_BASE = dict(
    icc=0.20, n=100, n2=None, n3=None,
    label_frac=0.25, llm_noise=0.20, llm_noise2=None, llm_noise3=None,
    bias_type="differential", bias_delta=0.30, bias_const=0.40,
    bias_extra_a=0.0, bias_extra_b=0.0, bias_extra_c=0.0, bias_extra_d=0.0,
    slope_a=1.0, slope_b=1.0, slope_c=1.0, slope_d=1.0,
    label_mnar=False, mnar_strength=1.0, mnar_mode="high",
    repeated_corr=0.0,
)
"""The baseline every VARIANT perturbs. Mirrors build_ppi_factorial_sources'
own baseline at its MCAR / es="null" corner (label_mnar=False, effect_size=0),
so a cell here is the same kind of cell the paper's factorial view buckets."""

_ALIGN_IRRELEVANT = frozenset({
    "n", "n2", "n3", "label_frac", "label_mnar", "mnar_strength", "mnar_mode",
    "repeated_corr",
})
"""Fields measure_judge_alignment's result cannot depend on, and which
therefore must not enter the alignment cache key.

measure_judge_alignment opens with ``replace(sc, n=n_mc)``, so the scenario's
own n is discarded before anything is drawn -- the whole point is a
population-level judge-vs-truth measurement, not a sample-sized one. Labeling
fields are likewise unused: no labels are drawn at all, only truth and judge
scores. This is what lets the four sample-size VARIANTS share ONE alignment
computation instead of four identical ones, and it is verified empirically --
n=60 and n=100 return identical metrics at matched noise.

Anything NOT listed here goes into the key, including fields that may turn out
not to matter: over-keying only costs a cache miss, under-keying returns wrong
numbers."""

_METRIC_LABELS = {
    "weighted_kappa": "quadratic weighted $\\kappa$",
    "linear_weighted_kappa": "linear weighted $\\kappa$",
    "kappa": "Cohen's $\\kappa$",
    "pearson_r": "Pearson $r$",
    "spearman_r": "Spearman $\\rho$",
    "kendall_tau_b": "Kendall $\\tau_b$",
    "icc_21": "ICC(2,1)",
    "lin_ccc": "Lin's CCC",
    "krippendorff_alpha": "Krippendorff's $\\alpha$",
    "gwet_ac1": "Gwet's AC1",
    "pabak": "PABAK",
    "percent_agreement": "percent agreement",
    "rho2": "$\\rho^2$",
}
"""Display names only. Which metrics actually appear for a given eval type is
decided by _alignment_metric_dict, not here -- this maps whatever it returns."""

_METRIC_ORDER = tuple(_METRIC_LABELS)


def panel_grid(eval_type: str, likert_max: int) -> tuple[tuple[float, ...], tuple[float, ...], object]:
    """(bias_grid, noise_grid, scale_bounds) in the units that panel's judge
    model actually takes.

    Binary gets its own grids in flip-probability units and no conversion;
    everything else gets the standardized fracs plus the scale_bounds that
    make a frac mean the same severity on a wider Likert scale."""
    if eval_type == "binary":
        return BINARY_BIAS_LEVELS, BINARY_NOISE_LEVELS, None
    bounds = (1.0, float(likert_max)) if eval_type == "likert" else None
    return BIAS_FRACS, NOISE_FRACS, bounds


def applicable_variants(eval_type: str,
                        variants: tuple[tuple[str, dict], ...]) -> tuple[tuple[str, dict], ...]:
    """Drop variants whose every override is a no-op for this judge model, so
    they neither burn compute nor stack duplicate curves (see
    _BINARY_INAPPLICABLE)."""
    if eval_type != "binary":
        return variants
    out = []
    for label, over in variants:
        if over and all(any(k.startswith(p) for p in _BINARY_INAPPLICABLE) for k in over):
            continue
        out.append((label, over))
    return tuple(out)


def _resolve_overrides(over: dict, eval_type: str, bounds) -> dict:
    """Translate a variant's standardized override keys into the absolute
    values JudgeBiasSource takes.

    Only `confound_weight_frac` needs this today. confound_weight is in the
    eval type's own score units, exactly like bias_delta -- the harness's own
    confound scenarios set it as `_jb_bias_magnitude(et, 0.06)` and tier it at
    fracs of 0.03/0.05/0.08, with a comment noting those values were chosen to
    stay out of a saturated rejection range. Passing a frac through raw walks
    straight into that trap: 0.2 raw on continuous is ~1.7 SD against an
    intended 0.08 SD, which pins the false-positive rate at 1.0 everywhere and
    contributes nothing but a flat ceiling line to the variant band.

    Binary is the exception again -- there the confound rides inside
    _jb_llm_binary's flip-probability skew rather than an additive score term
    (see confound_weight's docstring), so the frac is used as a probability
    directly, on PPI_BINARY_BIAS_MAGNITUDES' scale."""
    if "confound_weight_frac" not in over:
        return dict(over)
    out = {k: v for k, v in over.items() if k != "confound_weight_frac"}
    frac = over["confound_weight_frac"]
    out["confound_weight"] = (
        frac * _BINARY_CONFOUND_SCALE if eval_type == "binary"
        else _jb_bias_magnitude(eval_type, frac, scale_bounds=bounds)
    )
    return out


_BINARY_CONFOUND_SCALE = 2.5
"""Maps a confound frac onto binary's flip-probability skew scale: the
0.05/0.08 fracs used elsewhere become 0.125/0.20, inside
PPI_BINARY_BIAS_MAGNITUDES' own ~0.10-0.30 range (its "moderate" is 0.10,
"severe" 0.30). A single factor rather than a second grid keeps the variant
labels meaning the same severity tier on every panel."""


def build_sources(
    panels: tuple[tuple[str, int], ...] = PANELS,
    variants: tuple[tuple[str, dict], ...] = VARIANTS,
) -> list[tuple[JudgeBiasSource, str, str, dict]]:
    """One null-effect cell per (panel, variant, bias, noise).

    Bias and noise arrive as standardized FRACTIONS and are converted to the
    absolute values JudgeBiasSource actually takes, per panel -- see
    BIAS_FRACS. Binary bypasses the conversion entirely, since its model is a
    confusion matrix rather than an additive offset.

    Returns (source, panel_id, variant_label, overrides) rather than bare
    sources: the overrides have to reach the CSV so a later re-analysis can
    tell the curves apart without re-deriving them from the scenario name."""
    out: list[tuple[JudgeBiasSource, str, str, dict]] = []
    for eval_type, lmax in panels:
        pid = panel_id(eval_type, lmax)
        biases, noises, bounds = panel_grid(eval_type, lmax)
        for vlabel, over in applicable_variants(eval_type, variants):
            over_abs = _resolve_overrides(over, eval_type, bounds)
            for bias_frac in biases:
                for noise_frac in noises:
                    if eval_type == "binary":
                        bias_abs, noise_abs = bias_frac, noise_frac
                    else:
                        bias_abs = _jb_bias_magnitude(eval_type, bias_frac, scale_bounds=bounds)
                        noise_abs = _jb_bias_magnitude(eval_type, noise_frac, scale_bounds=bounds)
                    sc = JudgeBiasSource(
                        name=f"irrpeak|{pid}|{vlabel}|b{bias_frac:g}|z{noise_frac:g}",
                        tag="irr_peak", eval_type=eval_type, effect_size=0.0,
                        likert_max=lmax,
                        # bias_type mirrors build_ppi_factorial_sources_binary:
                        # a zero bias is "none", not a differential of size 0.
                        **{**_BASE, "bias_delta": bias_abs, "llm_noise": noise_abs,
                           "bias_type": "none" if bias_frac == 0.0 else "differential",
                           **over_abs},
                    )
                    out.append((sc, pid, vlabel, {**over, "_bias_frac": bias_frac,
                                                  "_noise_frac": noise_frac}))
    return out


def _align_key(sc: JudgeBiasSource, pid: str) -> tuple:
    """Cache key for measure_judge_alignment: every judge-model field that can
    change the answer, and none that cannot (see _ALIGN_IRRELEVANT)."""
    fields = ("bias_type", "bias_delta", "bias_const", "llm_noise", "slope_a",
              "noise_family", "contam_frac", "contam_scale", "confound_weight",
              "confound_truth_corr", "confound_shift_a", "confound_shift_b",
              "icc", "likert_max", "eval_type")
    return (pid,) + tuple(getattr(sc, f, None) for f in fields)


def _align_worker(args):
    sc, mc, seed = args
    return measure_judge_alignment(sc, n_mc=mc, seed=seed)


def measure_alignments(
    cells: list[tuple[JudgeBiasSource, str, str, dict]], *, align_mc: int,
    seed: int, n_workers: int = 1,
) -> dict:
    """IRR panel for every DISTINCT judge model in `cells`, keyed by
    _align_key.

    Two optimisations, both of which matter at this grid's size. First,
    deduplication: variants differing only in n (or any other
    _ALIGN_IRRELEVANT field) share one measurement instead of repeating an
    identical one -- with four sample-size variants that is most of a 4x
    saving on this phase. Second, the survivors are measured in parallel; this
    used to be a serial loop and was the half of the runtime that --workers
    did not scale."""
    todo: dict[tuple, JudgeBiasSource] = {}
    for sc, pid, _v, _o in cells:
        todo.setdefault(_align_key(sc, pid), sc)
    keys = list(todo)
    payload = [(todo[k], align_mc, seed + 9 + i) for i, k in enumerate(keys)]

    if n_workers > 1 and len(payload) > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_align_worker, payload, chunksize=8))
    else:
        results = [_align_worker(p) for p in payload]
    return dict(zip(keys, results))


def run_sweep(
    cells: list[tuple[JudgeBiasSource, str, str, dict]], *, n_reps: int,
    n_boot: int, align_mc: int, seed: int, progress_mode: str = "bar",
    n_workers: int = 1,
) -> list[dict]:
    """Run every cell and pair its false-positive rates with its measured IRR
    panel. Returns plain dicts (one per cell) rather than a dataclass: the
    metric panel is eval-type-dependent, so a fixed set of fields would either
    be mostly empty or need per-type subclasses.

    Every row carries the raw reject COUNTS alongside the rates, and every
    swept parameter alongside the variant label, so the CSV is sufficient on
    its own -- a later re-plot or re-analysis (different summary, binomial CIs,
    a metric this run did not chart) never has to re-run the sweep."""
    sources = [c[0] for c in cells]
    raw = run_ppi_comparison_simulation(
        sources, n_reps=n_reps, n_boot=n_boot, progress_mode=progress_mode,
        seed=seed, n_workers=n_workers, methods=_COMPARISON_METHODS,
    )
    pooled = pool_ppi_comparison_across_methods(
        [r for r in raw if r.method in _COMPARISON_METHODS]
    )
    by_name = {r.name: r for r in pooled}
    align = measure_alignments(cells, align_mc=align_mc, seed=seed, n_workers=n_workers)

    rows: list[dict] = []
    for sc, pid, vlabel, over in cells:
        res = by_name.get(sc.name)
        if res is None or not res.n_reps:
            continue
        metrics = align[_align_key(sc, pid)]
        rows.append({
            "panel": pid,
            "eval_type": sc.eval_type,
            "likert_max": sc.likert_max,
            "variant": vlabel,
            "variant_overrides": ";".join(f"{k}={v}" for k, v in sorted(over.items())
                                          if not k.startswith("_")) or "-",
            # `bias`/`noise` are the STANDARDIZED sweep coordinates (population-SD
            # fractions; flip probabilities on binary) -- comparable across
            # panels, and what the plots and peak summary are keyed on. The
            # absolute values actually handed to the judge model are kept
            # beside them, since they are what the model saw and they differ by
            # panel for the same frac.
            "bias": float(over["_bias_frac"]),
            "noise": float(over["_noise_frac"]),
            "bias_delta_abs": float(sc.bias_delta),
            "llm_noise_abs": float(sc.llm_noise),
            # Every swept judge-model parameter, so a re-analysis can group or
            # filter on the actual value instead of parsing `variant`.
            "n": sc.n,
            "icc": float(sc.icc),
            "slope_a": float(sc.slope_a),
            "noise_family": sc.noise_family,
            "contam_frac": float(getattr(sc, "contam_frac", float("nan"))),
            "contam_scale": float(getattr(sc, "contam_scale", float("nan"))),
            "confound_weight": float(getattr(sc, "confound_weight", 0.0)),
            "confound_truth_corr": float(getattr(sc, "confound_truth_corr", 0.0)),
            "confound_shift_a": float(getattr(sc, "confound_shift_a", 0.0)),
            "bias_type": sc.bias_type,
            "label_frac": float(sc.label_frac),
            "alpha": _ALPHA,
            # Raw counts as well as rates: binomial CIs and any re-pooling need
            # the numerator and denominator, which a rounded rate cannot give.
            "n_reps": res.n_reps,
            "rejects_llm_only": res.rejects_llm_only,
            "rejects_ppi": res.rejects_ppi,
            "rejects_all_human": res.rejects_all_human,
            "rejects_human_subset": res.rejects_human_subset,
            "fpr_uncorrected": res.rejects_llm_only / res.n_reps,
            "fpr_ppi": res.rejects_ppi / res.n_reps,
            "n_failed": res.n_failed,
            **{k: float(v) for k, v in metrics.items()},
        })
    return rows


def find_peaks(rows: list[dict]) -> list[dict]:
    """For each (eval_type, bias, metric): the metric value where the
    uncorrected false-positive rate is highest across the noise sweep.

    Also reports the metric's ceiling over the sweep (`metric_max`) and the
    rate at that ceiling (`fpr_at_metric_max`). Those two are what separate
    the two regimes the module docstring describes: below the discretization
    threshold the ceiling is ~1.0 and the rate there falls back to nominal (a
    genuine interior peak -- the judge can look perfect while being biased);
    above it the ceiling is well under 1.0 and the rate at the ceiling is
    still high (no deceptive regime -- the judge looks bad and is bad).

    Emits one row per (panel, variant, bias, metric), PLUS a row per
    (panel, bias, metric) with variant=MEAN_LABEL for the across-variant mean
    curve -- so a reader can ask both "where does this configuration peak?"
    and "where does the sweep peak on average?" from the same file."""
    out: list[dict] = []
    for pid in sorted({r["panel"] for r in rows}):
        p_rows = [r for r in rows if r["panel"] == pid]
        present = [m for m in _METRIC_ORDER
                   if any(m in r and np.isfinite(r[m]) for r in p_rows)]
        groups: list[tuple[str, list[dict]]] = [
            (v, [r for r in p_rows if r["variant"] == v])
            for v in sorted({r["variant"] for r in p_rows})
        ]
        groups.append((MEAN_LABEL, mean_curve(p_rows, present)))
        for vlabel, v_rows in groups:
            for bias in sorted({r["bias"] for r in v_rows}):
                cells = [r for r in v_rows if r["bias"] == bias]
                if not cells:
                    continue
                peak = max(cells, key=lambda r: r["fpr_uncorrected"])
                for m in present:
                    vals = [(r[m], r) for r in cells if m in r and np.isfinite(r[m])]
                    if not vals:
                        continue
                    top_val, top_row = max(vals, key=lambda v: v[0])
                    out.append({
                        "panel": pid,
                        "eval_type": cells[0]["eval_type"],
                        "likert_max": cells[0]["likert_max"],
                        "variant": vlabel, "bias": bias, "metric": m,
                        "peak_metric_value": peak.get(m, float("nan")),
                        "peak_fpr": peak["fpr_uncorrected"],
                        "peak_noise": peak["noise"],
                        "metric_max": top_val,
                        "fpr_at_metric_max": top_row["fpr_uncorrected"],
                        "fpr_ppi_at_peak": peak["fpr_ppi"],
                        "n_variants_pooled": (len({r["variant"] for r in p_rows})
                                              if vlabel == MEAN_LABEL else 1),
                    })
    return out


MEAN_LABEL = "<mean>"
"""Reserved variant label for the across-variant average curve."""


def mean_curve(p_rows: list[dict], metrics: list[str]) -> list[dict]:
    """Average across variants at each (bias, noise), NOT at each metric value.

    This is the only well-defined way to average these curves. Two variants at
    the SAME noise land at DIFFERENT metric values -- that spread is the whole
    point of plotting them together -- so there is no common x-grid to average
    over. Averaging at matched (bias, noise) and then reporting the mean metric
    value as the x-coordinate keeps both coordinates on the same footing, and
    is exactly what the underlying design varies."""
    out: list[dict] = []
    keys = sorted({(r["bias"], r["noise"]) for r in p_rows})
    for bias, noise in keys:
        cells = [r for r in p_rows if r["bias"] == bias and r["noise"] == noise]
        if not cells:
            continue
        row = {
            "panel": cells[0]["panel"], "eval_type": cells[0]["eval_type"],
            "likert_max": cells[0]["likert_max"], "variant": MEAN_LABEL,
            "bias": bias, "noise": noise,
            "fpr_uncorrected": float(np.mean([c["fpr_uncorrected"] for c in cells])),
            "fpr_ppi": float(np.mean([c["fpr_ppi"] for c in cells])),
            "n_reps": int(np.sum([c["n_reps"] for c in cells])),
        }
        for m in metrics:
            vals = [c[m] for c in cells if m in c and np.isfinite(c[m])]
            row[m] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def print_controls(rows: list[dict]) -> None:
    """Two checks that should hold on every run, printed before the results so
    a bad grid is obvious rather than buried.

    The bias=0 check EXCLUDES the confound variants. Their nuisance covariate
    differs between conditions regardless of bias_delta, so they reject above
    nominal there by construction; folding them in turns a sharp control into
    a meaningless average (0.13 pooled, versus 0.05 for every other variant)."""
    conf = [r for r in rows if r["confound_weight"] > 0]
    plain0 = [r for r in rows if r["bias"] == 0 and r["confound_weight"] == 0]
    print(f"\n{'=' * 84}\n  CONTROLS\n{'=' * 84}")
    if plain0:
        rate = sum(r["rejects_llm_only"] for r in plain0) / sum(r["n_reps"] for r in plain0)
        worst = max(r["fpr_uncorrected"] for r in plain0)
        ok = "ok" if abs(rate - _ALPHA) < 0.015 else "OFF -- grid may be mis-specified"
        print(f"  bias=0, no confound: uncorrected FPR {rate:.4f} (nominal {_ALPHA}), "
              f"worst cell {worst:.3f}   [{ok}]")
    if conf:
        c0 = [r for r in conf if r["bias"] == 0]
        if c0:
            rate = sum(r["rejects_llm_only"] for r in c0) / sum(r["n_reps"] for r in c0)
            print(f"  bias=0, WITH confound: {rate:.4f} -- expected above nominal "
                  f"(a between-condition nuisance covariate is a bias in its own right)")
    rate = sum(r["rejects_ppi"] for r in rows) / sum(r["n_reps"] for r in rows)
    worst = max(r["fpr_ppi"] for r in rows)
    ok = "ok" if abs(rate - _ALPHA) < 0.015 else "OFF -- PPI should hold nominal everywhere"
    print(f"  PPI-corrected, ALL cells: {rate:.4f}, worst cell {worst:.3f}   [{ok}]")
    failed = sum(1 for r in rows if r["n_failed"])
    print(f"  cells with failed replicates: {failed}")


def print_report(rows: list[dict], peaks: list[dict]) -> None:
    """Per panel: the across-variant MEAN peak location, plus the spread of
    per-variant peaks, so the console shows whether the variants agree."""
    print_controls(rows)
    for pid in sorted({r["panel"] for r in rows}):
        print(f"\n{'=' * 84}\n  {pid.upper()} -- uncorrected false-positive peak by judge bias\n{'=' * 84}")
        p_peaks = [p for p in peaks if p["panel"] == pid]
        mean_pk = [p for p in p_peaks if p["variant"] == MEAN_LABEL]
        if not mean_pk:
            continue
        et = mean_pk[0]["eval_type"]
        lmax = mean_pk[0]["likert_max"]
        metrics = sorted({p["metric"] for p in mean_pk}, key=_METRIC_ORDER.index)
        biases = sorted({p["bias"] for p in mean_pk})
        if et == "likert":
            sd = EVAL_TYPE_POPULATION_SD["likert"]
            print(f"  ({lmax}-point scale; half a point = {0.5 / sd * (5 - 1) / (lmax - 1):.3f} SD "
                  f"= the rounding threshold)")
        print(f"\n  peak METRIC VALUE of the across-variant mean curve, by bias_delta (SD units)")
        print("  " + f"{'metric':<24}" + "".join(f"{b:>8.2f}" for b in biases))
        for m in metrics:
            cells = []
            for b in biases:
                q = next((x for x in mean_pk if x["metric"] == m and x["bias"] == b), None)
                cells.append(f"{q['peak_metric_value']:>8.3f}" if q else f"{'--':>8}")
            print(f"  {m:<24}" + "".join(cells))
        print(f"\n  peak FPR of the mean curve (alpha={_ALPHA}); PPI-corrected below")
        print(f"  {'':<24}" + "".join(
            f"{next((q['peak_fpr'] for q in mean_pk if q['bias'] == b), float('nan')):>8.3f}"
            for b in biases))
        print(f"  {'PPI there':<24}" + "".join(
            f"{next((q['fpr_ppi_at_peak'] for q in mean_pk if q['bias'] == b), float('nan')):>8.3f}"
            for b in biases))
        # Spread across variants -- the reason the variants exist. A wide
        # range here means the peak's location is an artifact of whatever the
        # baseline happens to fix, not a property of the metric.
        head = _HEADLINE_METRIC.get(et, "pearson_r")
        if head in metrics:
            print(f"\n  spread of per-variant peaks for {head} (min-max across "
                  f"{len({p['variant'] for p in p_peaks}) - 1} variants)")
            cells = []
            for b in biases:
                vs = [p["peak_metric_value"] for p in p_peaks
                      if p["metric"] == head and p["bias"] == b and p["variant"] != MEAN_LABEL]
                cells.append(f"{min(vs):.2f}-{max(vs):.2f}" if vs else "--")
            print(f"  {'':<24}" + "".join(f"{c:>10}" for c in cells))


_HEADLINE_METRIC = {"likert": "weighted_kappa", "continuous": "pearson_r", "binary": "kappa"}
"""The metric each eval type's claim is usually stated in -- used only to pick
which row gets the extra across-variant spread line in the console report."""


def save_results_artifacts(*, rows: list[dict], peaks: list[dict], out_dir: str, run_stem: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    cells_path = str(Path(out_dir) / f"{run_stem}_irr_peak_cells.csv")
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(cells_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    paths.append(cells_path)

    peaks_path = str(Path(out_dir) / f"{run_stem}_irr_peak_summary.csv")
    if peaks:
        with open(peaks_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(peaks[0]))
            w.writeheader()
            w.writerows(peaks)
        paths.append(peaks_path)
    return paths


def save_irr_peak_grid_plot(rows: list[dict], panel: str, out_path: str) -> str | None:
    """One panel per (metric, bias): uncorrected false-positive rate against
    the metric, with the curve traced by the noise sweep and the peak marked.

    Rows are metrics and columns are bias levels, so reading DOWN a column
    answers "do all the IRR metrics agree about where the danger is at this
    bias?" and reading ACROSS a row answers "does this metric's danger point
    move with bias?" -- the two questions the single-bias factorial view
    cannot address.

    The bias=0 column is NOT plotted. It is a flat-at-nominal control -- an
    unbiased judge cannot inflate the false-positive rate at any noise level,
    which is worth having as a check but carries no information about where
    the danger point sits. It stays in the sweep and in the per-cell CSV (and
    print_report still shows it), so the control is still auditable; it just
    costs a tenth of the figure's width to say nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    et_rows = [r for r in rows if r["panel"] == panel]
    if not et_rows:
        return None
    eval_type = et_rows[0]["eval_type"]
    metrics = [m for m in _METRIC_ORDER
               if any(m in r and np.isfinite(r[m]) for r in et_rows) and m != "rho2"]
    biases = sorted({r["bias"] for r in et_rows if r["bias"] > 0})
    if not metrics or not biases:
        return None
    variants = sorted({r["variant"] for r in et_rows})
    mrows = mean_curve(et_rows, metrics)

    nr, nc = len(metrics), len(biases)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.35 * nr),
                             sharex=False, sharey=True, squeeze=False)
    for ri, m in enumerate(metrics):
        # percent_agreement is the one metric on a 0-100 scale; put it on the
        # same 0-1 axis as everything else so a row is readable against its
        # neighbours. The CSV keeps the raw value.
        scale = 0.01 if m == "percent_agreement" else 1.0
        # Chance-corrected coefficients (AC1, PABAK, the kappas) go NEGATIVE
        # once a judge is worse than chance, which the high-bias columns
        # reach -- so the row's limits come from its own data rather than a
        # hardcoded [0, 1]. Shared across the row so columns stay comparable.
        row_vals = [r[m] * scale for r in et_rows
                    if r["bias"] in biases and m in r and np.isfinite(r[m])]
        lo, hi = (min(row_vals), max(row_vals)) if row_vals else (0.0, 1.0)
        pad = max(0.04, 0.05 * (hi - lo))
        for ci, b in enumerate(biases):
            ax = axes[ri][ci]
            # Ordered by NOISE, not by the metric. The exact-match metrics
            # (percent agreement, Gwet's AC1, PABAK) are non-monotonic in
            # noise once the bias exceeds half a scale point: a deterministic
            # ~1.5-point shift bottoms exact agreement out near 10%, then
            # noise rounds some items back onto the truth and lifts it to
            # ~27%, then more noise randomises it away again. So one metric
            # value corresponds to TWO different noise levels with very
            # different false-positive rates. Sorting by metric interleaves
            # those two branches and joins them with near-vertical jumps,
            # which reads as a rendering glitch; drawing the real trajectory
            # shows it honestly, as a curve that doubles back.
            # Every VARIANT lightly sketched, so the spread across the sweep is
            # visible as a band rather than hidden behind one summary line.
            for v in variants:
                vc = sorted([r for r in et_rows if r["bias"] == b and r["variant"] == v
                             and m in r and np.isfinite(r[m])], key=lambda r: r["noise"])
                if len(vc) < 2:
                    continue
                ax.plot([r[m] * scale for r in vc], [r["fpr_uncorrected"] for r in vc],
                        "-", color="#c1442e", lw=0.45, alpha=0.32, zorder=2,
                        solid_capstyle="round")
                ax.plot([r[m] * scale for r in vc], [r["fpr_ppi"] for r in vc],
                        "-", color="#3a6ea5", lw=0.35, alpha=0.22, zorder=1)
            # The across-variant mean, drawn solid on top with the peak dot.
            cells = sorted([r for r in mrows if r["bias"] == b and m in r
                            and np.isfinite(r[m])], key=lambda r: r["noise"])
            if cells:
                xs = [r[m] * scale for r in cells]
                ax.plot(xs, [r["fpr_uncorrected"] for r in cells], "-",
                        color="#c1442e", lw=1.35, zorder=4)
                ax.plot(xs, [r["fpr_ppi"] for r in cells], "-", color="#3a6ea5",
                        lw=1.0, alpha=0.85, zorder=3)
                # Hollow marker at the lowest-noise end, so a curve that
                # doubles back can be read in the right direction.
                ax.plot([cells[0][m] * scale], [cells[0]["fpr_uncorrected"]],
                        "o", ms=2.6, mfc="none", mec="#7a2a1c", mew=0.7, zorder=5)
                pk = max(cells, key=lambda r: r["fpr_uncorrected"])
                ax.plot([pk[m] * scale], [pk["fpr_uncorrected"]], "o", ms=3.4,
                        color="#c1442e", mec="white", mew=0.6, zorder=5)
                ax.annotate(f"{pk[m] * scale:.2f}", (pk[m] * scale, pk["fpr_uncorrected"]),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=5.0, color="#c1442e")
            ax.axhline(_ALPHA, color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=0)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(lo - pad, hi + pad)
            ax.tick_params(labelsize=5.2, length=2, pad=1.4)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if ri == 0:
                ax.set_title(f"bias {b:g} SD", fontsize=6.4, pad=4)
            if ci == 0:
                ax.set_ylabel(_METRIC_LABELS.get(m, m), fontsize=5.9)
            if ri != nr - 1:
                ax.set_xticklabels([])

    fig.suptitle(
        f"{panel}: uncorrected false-positive rate (red) vs. each IRR metric, by judge bias "
        f"-- PPI-corrected in blue, nominal $\\alpha$ dashed.  Faint lines are the "
        f"{len(variants)} sweep variants; bold is their mean, dot marks its peak.",
        fontsize=8.2, y=0.997,
    )
    fig.supxlabel("IRR metric value (curve traced by the llm_noise sweep)", fontsize=7.0, y=0.004)
    fig.tight_layout(rect=(0, 0.012, 1, 0.985), h_pad=0.45, w_pad=0.35)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_peak_shift_plot(peaks: list[dict], panel: str, out_path: str) -> str | None:
    """The headline view: where each metric's danger point sits as bias grows.

    x is bias_delta, y is the metric value at which the false-positive rate
    peaks, one line per metric. A flat line means that metric's danger point
    does not move with bias; a falling line means a more biased judge is
    dangerous at a LOWER apparent agreement. The dotted marker at the right of
    each line flags the point past which the metric can no longer reach 1.0 at
    any noise level, i.e. where the deceptive regime ends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only the across-variant mean curve: overlaying 11 variants x 11 metrics
    # here would be unreadable. The per-variant spread lives in the grid plot
    # and in the summary CSV, which carries a row per variant.
    et = [p for p in peaks
          if p["panel"] == panel and p["bias"] > 0 and p["variant"] == MEAN_LABEL]
    if not et:
        return None
    eval_type = et[0]["eval_type"]
    lmax = et[0]["likert_max"]
    metrics = sorted({p["metric"] for p in et}, key=_METRIC_ORDER.index)
    metrics = [m for m in metrics if m != "rho2"]
    cmap = plt.get_cmap("turbo")
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for i, m in enumerate(metrics):
        pts = sorted([p for p in et if p["metric"] == m], key=lambda p: p["bias"])
        if not pts:
            continue
        # percent_agreement is on 0-100; everything else on 0-1 (see the grid
        # plot's note). Both the line and the ceiling test below use the
        # rescaled value so one y-axis serves the whole panel.
        sc = 0.01 if m == "percent_agreement" else 1.0
        col = cmap(i / max(1, len(metrics) - 1))
        # Several of these coincide almost exactly (ICC(2,1) and Lin's CCC are
        # the same estimator up to a bias-correction term; the kappas track
        # each other closely), so a solid line would hide one under another
        # entirely. Cycling dash patterns keeps every metric readable where
        # they overlap.
        ls = ("-", (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5)))[i % 4]
        ax.plot([p["bias"] for p in pts], [p["peak_metric_value"] * sc for p in pts],
                marker="o", ms=3.0, lw=1.3, ls=ls, color=col,
                label=_METRIC_LABELS.get(m, m))
        deceptive = [p for p in pts if p["metric_max"] * sc >= 0.995]
        if deceptive:
            last = max(deceptive, key=lambda p: p["bias"])
            ax.plot([last["bias"]], [last["peak_metric_value"] * sc], "s", ms=6.0,
                    mfc="none", mec=col, mew=1.1, zorder=4)
    if eval_type == "likert":
        # Half a scale point in SD units. The 5-point distribution is
        # rescaled onto a wider integer grid for likert_max>5 (see
        # sample_group_truth), so the population SD grows with the number of
        # points and half a point becomes a SMALLER fraction of it.
        thr = 0.5 / EVAL_TYPE_POPULATION_SD["likert"] * (5 - 1) / (lmax - 1)
        ax.axvline(thr, color="0.35", lw=0.9, ls=(0, (4, 2)))
        # Anchored to the axes' top-left in axes coordinates, not to a data
        # y-value: the lines' vertical extent depends on the run, and a
        # data-anchored label lands on top of the legend whenever the
        # chance-corrected metrics stay high.
        ax.annotate("half a scale point\n(bias rounds away to its left)",
                    xy=(thr, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points",
                    fontsize=6.4, color="0.3", ha="left", va="top")
    ax.set_xlabel("judge bias $b$ (population SDs)", fontsize=8.5)
    ax.set_ylabel("IRR value at which\nfalse positives peak", fontsize=8.5)
    ax.set_title(f"{panel}: does the danger point move with judge bias?", fontsize=9.5)
    ax.tick_params(labelsize=7.2)
    # Chance-corrected metrics (Krippendorff's alpha especially) go NEGATIVE
    # once the judge is worse than chance, which the high-bias end reaches on
    # continuous data -- clamping at 0 would silently cut those lines off
    # mid-descent, hiding the very behaviour this panel exists to show.
    ys = [p["peak_metric_value"] * (0.01 if p["metric"] == "percent_agreement" else 1.0)
          for p in et if p["metric"] in metrics]
    ymin = min([0.0, *ys]) if ys else 0.0
    ax.set_ylim(ymin - 0.04, 1.02)
    if ymin < 0:
        ax.axhline(0.0, color="0.6", lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.3, ncol=2, frameon=False, loc="lower left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--panels", nargs="+", default=[panel_id(e, l) for e, l in PANELS],
                        choices=[panel_id(e, l) for e, l in PANELS],
                        help="Which panels to sweep (default: all four). Each becomes its own "
                             "figure. likert5/likert7 are separate because the number of scale "
                             "points sets the rounding threshold this case is built around.")
    parser.add_argument("--variants", nargs="+", default=None, metavar="NAME",
                        help="Subset of VARIANTS by label (default: all). 'baseline' alone "
                             "reproduces the single-curve sweep and is ~11x faster.")
    parser.add_argument("--reps", type=int, default=1000, metavar="N",
                        help="Monte Carlo replicates per cell (default: 1000). The peak "
                             "LOCATION is stable far below this (sd ~0.02 at 300); it is the "
                             "peak HEIGHT that needs the reps.")
    parser.add_argument("--bootstrap-n", type=int, default=30, metavar="N",
                        help="Bootstrap draws inside each PPI cell (default: 30). ONLY the PPI "
                             "control curve uses these; the uncorrected rate, which is the "
                             "actual finding, does not. 30 is deliberately low: the control "
                             "reads the same at 20, 100 and 400 (0.045-0.059 / 0.042-0.052 / "
                             "0.041-0.053, all within MC noise), so paying more buys nothing. "
                             "Raise it only if you want the PPI curve itself to be a result "
                             "rather than a sanity check.")
    parser.add_argument("--alignment-mc", type=int, default=20000, metavar="N",
                        help="Sample size for the large-sample IRR measurement (default: "
                             "20000, matching --factorial-alignment-mc). Cheap (~28 ms/cell) "
                             "and deduplicated across variants that share a judge model.")
    parser.add_argument("--seed", type=int, default=42, metavar="N")
    parser.add_argument("--progress", choices=["bar", "plain", "none"], default="bar")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--save-results", choices=["save", "none"], default="save")
    parser.add_argument("--plots", choices=["save", "none"], default="save")


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    return [("IRR false-positive peak vs. judge bias (4 panels x all variants)",
             official_args(base_seed))]


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset: every panel and variant, at
    add_arguments' own CLI defaults (reps=1000, bootstrap_n=30,
    alignment_mc=20000) -- the one deliberate override is
    progress="plain" instead of "bar", since an official run's output is
    typically captured to a log file, where a redrawing progress bar is
    noise rather than signal."""
    return argparse.Namespace(
        panels=[panel_id(e, l) for e, l in PANELS], variants=None,
        reps=1000, bootstrap_n=30, alignment_mc=20000, seed=base_seed,
        progress="plain", out_dir="simulations/out", plots_dir=None,
        save_results="save", plots="save",
    )


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    return argparse.Namespace(
        panels=["likert5"], variants=["baseline", "n=60"], reps=3, bootstrap_n=20,
        alignment_mc=200, seed=base_seed, progress="plain",
        out_dir="simulations/out", plots_dir=None, save_results="save", plots="save",
    )


_SEC_PER_CELL_FIXED = 2.60
_SEC_PER_BOOT_DRAW = 0.0085
"""Measured cost model: wall-clock seconds per cell at reps=1000 on 15
workers is ``_SEC_PER_CELL_FIXED + _SEC_PER_BOOT_DRAW * n_boot``. At the
default n_boot the FIXED term -- five classical tests x reps, plus PPI's
non-bootstrap work -- dominates, so `reps` and the grid size are the real
runtime levers, not n_boot. Treat the printed estimate as +-30%."""


def estimate_runtime_s(n_cells: int, reps: int, n_boot: int, n_workers: int) -> float:
    """Rough wall-clock estimate, linear in reps and in n_boot, rescaled from
    the 15 workers the model was measured on. Deliberately approximate -- it
    exists to distinguish "20 minutes" from "12 hours", not to be accurate to
    the minute."""
    per = _SEC_PER_CELL_FIXED + _SEC_PER_BOOT_DRAW * max(0, n_boot)
    return n_cells * per * (reps / 1000.0) * (15.0 / max(1, n_workers))


def run(args: argparse.Namespace) -> CaseResult:
    """Case entry point: build the (panel x variant x bias x noise) cell grid
    from *args*, print a runtime estimate, run the sweep, locate each
    panel/metric's false-positive peak, print/save the report, save the
    grid and peak-shift plots per panel, and return a CaseResult
    summarizing the peak metric range per panel."""
    t0 = time.time()
    try:
        warnings.filterwarnings("ignore")
        want = set(getattr(args, "panels", None) or [panel_id(e, l) for e, l in PANELS])
        panels = tuple(p for p in PANELS if panel_id(*p) in want)
        vsel = getattr(args, "variants", None)
        variants = tuple(v for v in VARIANTS if vsel is None or v[0] in set(vsel))
        if not panels or not variants:
            return CaseResult(case_name=CASE_NAME, status="error",
                              error="no panels or no variants selected",
                              duration_s=time.time() - t0)

        cells = build_sources(panels, variants)
        n_boot = getattr(args, "bootstrap_n", 30)
        workers = getattr(args, "workers", 1)
        est = estimate_runtime_s(len(cells), args.reps, n_boot, workers)
        n_align = len({_align_key(sc, pid) for sc, pid, _v, _o in cells})
        print(f"\nirr_peak simulation -- {len(cells)} cells "
              f"({len(panels)} panels x <=%d variants x %d bias x %d noise), "
              "reps=%d, n_boot=%d" % (len(variants), len(BIAS_FRACS), len(NOISE_FRACS), args.reps, n_boot))
        print(f"  alignment measurements: {n_align} distinct judge models "
              f"({len(cells) - n_align} deduplicated away)")
        boot_share = _SEC_PER_BOOT_DRAW * n_boot / (_SEC_PER_CELL_FIXED + _SEC_PER_BOOT_DRAW * n_boot)
        print(f"  estimated wall clock: {est / 3600:.1f} h at {workers} workers "
              f"({boot_share:.0%} of it the PPI bootstrap; scale with --reps or "
              f"--panels/--variants)")

        rows = run_sweep(
            cells, n_reps=args.reps, n_boot=n_boot,
            align_mc=getattr(args, "alignment_mc", 20000), seed=args.seed,
            progress_mode=args.progress, n_workers=workers,
        )
        if not rows:
            return CaseResult(case_name=CASE_NAME, status="error",
                              error="no cells produced results", duration_s=time.time() - t0)

        peaks = find_peaks(rows)
        print_report(rows, peaks)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"irr_peak_reps{args.reps}_{stamp}"
        output_paths: list[str] = []
        if args.save_results == "save":
            output_paths += save_results_artifacts(
                rows=rows, peaks=peaks, out_dir=args.out_dir, run_stem=run_stem)

        if getattr(args, "plots", "save") == "save":
            plots_dir = getattr(args, "plots_dir", None) or str(Path(args.out_dir) / "plots")
            for eval_type, lmax in panels:
                pid = panel_id(eval_type, lmax)
                grid = save_irr_peak_grid_plot(
                    rows, pid, str(Path(plots_dir) / f"{run_stem}_{pid}_grid.png"))
                if grid:
                    output_paths.append(grid)
                    print(f"Saved plot: {grid}")
                shift = save_peak_shift_plot(
                    peaks, pid, str(Path(plots_dir) / f"{run_stem}_{pid}_peak_shift.png"))
                if shift:
                    output_paths.append(shift)
                    print(f"Saved plot: {shift}  (key figure)")

        key: dict = {"n_cells": len(rows), "n_variants": len(variants)}
        for eval_type, lmax in panels:
            pid = panel_id(eval_type, lmax)
            head = _HEADLINE_METRIC.get(eval_type, "pearson_r")
            sub = [p for p in peaks if p["panel"] == pid and p["metric"] == head
                   and p["bias"] > 0 and p["variant"] == MEAN_LABEL]
            if sub:
                key[f"{pid}_{head}_peak_range"] = (
                    f"{min(p['peak_metric_value'] for p in sub):.3f}-"
                    f"{max(p['peak_metric_value'] for p in sub):.3f}")
        return CaseResult(case_name=CASE_NAME, status="ok", output_paths=output_paths,
                          key_metrics=key, duration_s=time.time() - t0)
    except Exception as exc:
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc),
                          duration_s=time.time() - t0)
