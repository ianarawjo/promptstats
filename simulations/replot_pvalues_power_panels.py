"""Redraw the two supplementary power panels with PAIRED uncertainty bands.

The shipped figures (supfig:pvalues:pairwise:real, supfig:fwer:pvalue:real) draw
each method's 95% CI on its across-scenario mean. On a Type-I / FWER axis that
is the right band -- it answers "is this rate above nominal?". On a POWER axis
it is the wrong one, and badly so:

  * Per-scenario power spans essentially the whole range (0.00-1.00 on both
    suites), because real benchmark pairs differ hugely in true effect size.
    The marginal band therefore measures the SCENARIO MIX, not the method --
    which is why every method's band comes out the same width to three
    decimals (+-0.21 on multiarm, +-0.057 on pairwise).
  * Every method sees the SAME scenarios, so that variation is common-mode and
    cancels in a comparison. The marginal band throws the pairing away and is
    10-40x wider than the differences it should resolve, so the panel renders
    as overlapping mud that reads "these are indistinguishable" -- when paired,
    McNemar exact loses 4.8 power points to Wilcoxon at +-0.6 (~8 sigma).

This script keeps the marginal band on the null-condition panel and replaces it
on the alt panel with per-rep MONTE CARLO error, 1.96*sqrt(p(1-p)/N_reps)
(--alt-band mc, the shipped default). That is a property of the method's own
curve and nothing else's, so the panel stays a plain absolute-power plot with
no reference method anywhere -- a drop-in replacement at the same size, not a
re-conception. It answers the narrower question ("how precisely is this suite's
average pinned down") that _scenario_values' docstring rejects in general; the
rejection does not bind here, because on a power axis the scenario spread it
prefers is common-mode and swamps the curves.

--alt-band paired instead sizes the band by the paired difference against a
reference method (1.96*sd(method_s - ref_s)/sqrt(n_scenarios), scenarios matched
by (eval_type, label)), and --alt-band none draws bare lines. --diagnostic adds
a third panel plotting the paired differences directly, which is the clearest
view of method ordering but changes what the panel is.

Usage:
    python simulations/replot_pvalues_power_panels.py                # in place
    python simulations/replot_pvalues_power_panels.py --out-dir /tmp/x
    python simulations/replot_pvalues_power_panels.py --diagnostic
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import statistics as st
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import csv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "simulations" / "out"

#: (csv, figure stem, null-condition label, alt aggregation, reference method)
FIGURES = {
    "pairwise_real": dict(
        csv=OUT / "official_20260831_001109"
            / "pvalues_pairwise_real_reps300_20260831_001149_pairwise_results.csv",
        method_col="method", numer="rejects", eval_filter="binary",
        ref="wilcoxon",
        stem="pvalues_pairwise_real_reps300_20260831_001149_typeI_power_binary",
        figsize=(11.0, 4.2),
        null_title="binary: Type-I error", null_y="Rejection rate (null)",
        alt_y="Rejection rate (alt)", xlabel="n",
        title="pvalues (pairwise, non-PPI): Type-I + Power [binary]",
    ),
    "multiarm_real": dict(
        csv=OUT / "official_20260831_130350"
            / "pvalues_multiarm_reps500_20260831_130354_multiarm_results.csv",
        method_col="correction", numer="best_selected", null_numer="any_reject",
        eval_filter=None, ref="holm",
        stem="pvalues_multiarm_reps500_20260831_130354_fwer_vs_n",
        figsize=(10.0, 4.5),
        null_title="FWER vs. sample size", null_y="FWER (null)",
        alt_y="Best-arm selection power (alt)", xlabel="n (sample size)",
        title="Family-Wise Error Rate and Best-Arm Selection Power vs. Sample Size",
    ),
}

ALPHA = 0.05


def load(spec):
    rows = list(csv.DictReader(open(spec["csv"])))
    if spec["eval_filter"]:
        rows = [r for r in rows if r.get("eval_type") == spec["eval_filter"]]
    return rows


def per_scenario(rows, numer):
    """{(eval_type, label): rate} -- the scenario is the unit of replication."""
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        a = acc[(r["eval_type"], r["label"])]
        a[0] += float(r[numer])
        a[1] += float(r["n_reps"])
    return {k: n / d for k, (n, d) in acc.items() if d > 0}


def marginal_ci(vals):
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    return 1.96 * st.stdev(vals) / math.sqrt(len(vals))


def mc_ci(rows, numer):
    """Per-rep Monte Carlo error on the pooled rate: 1.96*sqrt(p(1-p)/N_reps).

    Answers only "how precisely is THIS suite's average pinned down", not how
    the method behaves across scenarios -- but on a power axis that narrower
    question is the one the panel can actually answer, since scenario spread
    is common-mode and swamps everything else.
    """
    c = sum(float(r[numer]) for r in rows)
    t = sum(float(r["n_reps"]) for r in rows)
    if t <= 0:
        return 0.0
    p = c / t
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / t)


def paired_ci(vals_by_scen, ref_by_scen):
    """CI half-width on the mean paired difference, scenario as the unit."""
    keys = [k for k in vals_by_scen if k in ref_by_scen]
    if len(keys) < 2:
        return 0.0, 0.0
    d = [vals_by_scen[k] - ref_by_scen[k] for k in keys]
    if len(set(d)) == 1:
        return sum(d) / len(d), 0.0
    return sum(d) / len(d), 1.96 * st.stdev(d) / math.sqrt(len(d))


def build(name, spec, out_dir, diagnostic=False, alt_band="mc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import simulations.harness.cases.pvalues as PV

    rows = load(spec)
    mcol, ref = spec["method_col"], spec["ref"]
    null_numer = spec.get("null_numer", spec["numer"])
    sizes = sorted({int(r["n"]) for r in rows})
    methods = [m for m in PV.order_present_methods({r[mcol] for r in rows})] \
        if spec["method_col"] == "method" else \
        [m for m in PV.MULTIARM_PLOT_METHODS if m.name in {r[mcol] for r in rows}]

    ncols = 3 if diagnostic else 2
    w, h = spec["figsize"]
    fig, axes = plt.subplots(1, ncols, figsize=((14.0, 4.0) if diagnostic else (w, h)))
    ax_null, ax_abs = axes[0], axes[1]
    ax_diff = axes[2] if diagnostic else None
    ax_null.axhline(ALPHA, color="black", lw=1.0, ls="--")
    ax_null.axhspan(*PV.bradley_bounds(ALPHA), color="#DDDDDD", alpha=0.4, zorder=0)
    if ax_diff is not None:
        ax_diff.axhline(0.0, color="black", lw=1.0, ls="--")

    # Reference curve, per n, per scenario.
    ref_scen = {}
    for n in sizes:
        sub = [r for r in rows if int(r["n"]) == n and r["condition"] != "null"
               and r[mcol] == ref]
        ref_scen[n] = per_scenario(sub, spec["numer"]) if sub else {}

    for m in methods:
        m_rows = [r for r in rows if r[mcol] == m.name]
        if not m_rows:
            continue
        # --- null panel: marginal band, unchanged from the shipped figure ---
        xs, ys, half = [], [], []
        for n in sizes:
            sub = [r for r in m_rows if int(r["n"]) == n and r["condition"] == "null"]
            if not sub:
                continue
            scen = per_scenario(sub, null_numer)
            xs.append(n); ys.append(sum(scen.values()) / len(scen))
            half.append(marginal_ci(scen.values()))
        if xs:
            ax_null.plot(xs, ys, "-o", color=m.color, ms=4, lw=1.2, label=m.name, alpha=.85)
            ax_null.fill_between(xs, [a - b for a, b in zip(ys, half)],
                                 [a + b for a, b in zip(ys, half)],
                                 color=m.color, alpha=.22, lw=0)
        # --- alt panels: PAIRED band against the reference ---
        xa, ya, dm, dh = [], [], [], []
        for n in sizes:
            sub = [r for r in m_rows if int(r["n"]) == n and r["condition"] != "null"]
            if not sub or not ref_scen.get(n):
                continue
            scen = per_scenario(sub, spec["numer"])
            d, h = paired_ci(scen, ref_scen[n])
            xa.append(n); ya.append(sum(scen.values()) / len(scen))
            dm.append(d)
            dh.append(mc_ci(sub, spec["numer"]) if alt_band == "mc" else h)
        if not xa:
            continue
        is_ref = m.name == ref
        ax_abs.plot(xa, ya, "-o", color=m.color, ms=4, lw=1.2, label=m.name, alpha=.85)
        if alt_band != "none" and not (alt_band == "paired" and is_ref):
            ax_abs.fill_between(xa, [a - b for a, b in zip(ya, dh)],
                                [a + b for a, b in zip(ya, dh)],
                                color=m.color, alpha=.28, lw=0)
        if ax_diff is not None:
            ax_diff.plot(xa, dm, "-o", color=m.color, ms=4, lw=1.2,
                         label=m.name, alpha=.85, ls="--" if is_ref else "-")
            if not is_ref:
                ax_diff.fill_between(xa, [a - b for a, b in zip(dm, dh)],
                                     [a + b for a, b in zip(dm, dh)],
                                     color=m.color, alpha=.28, lw=0)

    ax_null.set_title(f"{spec['null_title']}\nband: 95% CI on the across-scenario mean")
    ax_null.set_ylabel(spec["null_y"])
    _band_note = {"paired": f"band: paired 95% CI vs {ref}",
                  "mc": "band: 95% Monte Carlo error on the pooled rate",
                  "none": "absolute level is scenario-dominated; see the table for spread"}[alt_band]
    ax_abs.set_title(f"power (mean over alt conditions)\n{_band_note}")
    ax_abs.set_ylabel(spec["alt_y"])
    if ax_diff is not None:
        ax_diff.set_title(f"power difference vs {ref}\nsame paired 95% CI, level removed")
        ax_diff.set_ylabel(f"$\\Delta$ power vs {ref}")
    for ax in axes:
        ax.set_xlabel(spec["xlabel"])
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = ax_null.get_legend_handles_labels()
    axes[-1].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0, fontsize=7)
    fig.suptitle(f"{spec['title']}  |  alpha={ALPHA}", fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / ((f"{name}_paired_bands_diagnostic.png") if diagnostic
                           else f"{spec['stem']}.png")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default=str(ROOT / "simulations" / "out" / "paper_overleaf_src"
                               / "media" / "simulations"))
    ap.add_argument("--only", choices=sorted(FIGURES), help="just one figure")
    ap.add_argument("--alt-band", choices=("mc", "paired", "none"), default="mc",
                    help="uncertainty band on the power panel (default: mc)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="add a third panel plotting the paired differences directly")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for name, spec in FIGURES.items():
        if a.only and name != a.only:
            continue
        if not Path(spec["csv"]).exists():
            print(f"  SKIP {name}: {spec['csv']} not found")
            continue
        print("wrote", build(name, spec, a.out_dir, a.diagnostic, a.alt_band))


if __name__ == "__main__":
    main()
