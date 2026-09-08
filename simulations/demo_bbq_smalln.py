#!/usr/bin/env python3
"""Numbers behind the small-n BBQ demonstration (Sec. 8.1).

Ground truth is the full 1,000-item BBQ benchmark from inspect_benchmarks.csv.
We repeatedly draw the 30-item eval set a student would actually build and ask,
for each analysis they might run, how often it reaches the right conclusion.

Three models, chosen so one pair is genuinely tied and two are genuinely apart.
A single eval run (run_idx=0) is used throughout, because that is what a student
executes; truth is that run's accuracy over all 1,000 items, so the only error
being measured is item sampling.

    python simulations/demo_bbq_smalln.py
    python simulations/demo_bbq_smalln.py --n 120     # the scale-up advice
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from evalstats.core.resampling import bonett_price_paired_ci, wilson_ci_1d

CSV = "simulations/out/inspect_benchmarks.csv"
TRIO = ["openrouter/openai/gpt-4o-mini",
        "openrouter/ibm-granite/granite-4.1-8b",
        "openrouter/google/gemma-3n-e4b-it"]
SHORT = ["gpt-4o-mini", "granite-4.1-8b", "gemma-3n-e4b-it"]


def load(run_idx=0):
    d = pd.read_csv(CSV)
    d = d[(d.benchmark == "bbq") & (d.run_idx == run_idx) & (d.model.isin(TRIO))]
    return d.pivot(index="item_id", columns="model", values="score")[TRIO].to_numpy()


def boot_quantiles(x, B, rng):
    """Bootstrap the mean once, then return (percentile, BCa) CI functions of alpha.

    Calling scipy.stats.bootstrap separately per alpha would redo the resampling
    each time; the resampling distribution does not depend on alpha, so it is
    drawn once here. Verified against scipy.stats.bootstrap (see --check).
    """
    n = len(x)
    bs = np.sort(x[rng.integers(0, n, (B, n))].mean(1))
    # BCa bias-correction and acceleration, from the same resamples.
    z0 = stats.norm.ppf(np.clip((bs < x.mean()).mean(), 1 / B, 1 - 1 / B))
    jk = (x.sum() - x) / (n - 1)
    jm = jk.mean()
    den = 6.0 * (((jm - jk) ** 2).sum() ** 1.5)
    acc = ((jm - jk) ** 3).sum() / den if den > 0 else 0.0

    def pct(alpha):
        return (np.percentile(bs, 100 * alpha / 2),
                np.percentile(bs, 100 * (1 - alpha / 2)))

    def bca(alpha):
        out = []
        for z in (stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)):
            zz = z0 + (z0 + z) / (1 - acc * (z0 + z))
            out.append(np.percentile(bs, 100 * stats.norm.cdf(zz)))
        return tuple(out)

    return pct, bca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X = load()
    truth = X.mean(0)
    N, n = len(X), a.n
    PAIRS = list(itertools.combinations(range(3), 2))  # (0,1) tied, (0,2)&(1,2) real

    print(f"corpus: {N} BBQ items x 3 models, run 0   n = {n}")
    for j, s in enumerate(SHORT):
        print(f"   {s:<18} {truth[j]:.4f}")
    for i, j in PAIRS:
        print(f"   {SHORT[i]} - {SHORT[j]}: {truth[i]-truth[j]:+.4f}")

    rng = np.random.default_rng(a.seed)
    meth = ["overlap", "percentile", "bca", "bonett_price"]
    rej = {m: np.zeros(3) for m in meth}
    cov = {m: np.zeros(3) for m in meth if m != "overlap"}
    # matched-Type-I: sweep alpha for each method, read power at matched FPR
    grid = np.array([.001, .005, .01, .02, .03, .05, .07, .09,
                     .12, .16, .20, .25, .30, .40])
    mrej = {m: np.zeros((len(grid), 3)) for m in ("percentile", "bca", "bonett_price")}

    for _ in range(a.reps):
        s = X[rng.choice(N, n, replace=False)]
        wil = [wilson_ci_1d(s[:, j], 0.05) for j in range(3)]
        for k, (i, j) in enumerate(PAIRS):
            tr = truth[i] - truth[j]
            rej["overlap"][k] += wil[i][0] > wil[j][1] or wil[j][0] > wil[i][1]
            df = s[:, i] - s[:, j]
            pct, bca = boot_quantiles(df, a.n_boot, rng)
            cis = {"percentile": pct(0.05), "bca": bca(0.05),
                   "bonett_price": bonett_price_paired_ci(s[:, i], s[:, j], 0.05)}
            for m, (lo, hi) in cis.items():
                rej[m][k] += lo > 0 or hi < 0
                cov[m][k] += lo <= tr <= hi
            for gi, al in enumerate(grid):
                for m, f in (("percentile", pct), ("bca", bca),
                             ("bonett_price",
                              lambda al: bonett_price_paired_ci(s[:, i], s[:, j], al))):
                    lo, hi = f(al)
                    mrej[m][gi, k] += lo > 0 or hi < 0
    R = a.reps
    print(f"\n{'method':<24}{'FPR (tied pair)':>17}{'power (2 real gaps)':>22}{'coverage':>12}")
    for m, lab in [("overlap","read overlap of CIs"),("percentile","percentile bootstrap"),
                   ("bca","BCa bootstrap"),("bonett_price","evalstats (Bonett-Price)")]:
        c = f"{cov[m].mean()/R:.3f}" if m in cov else "--"
        print(f"  {lab:<22}{rej[m][0]/R:>17.3f}{rej[m][1:].mean()/R:>22.3f}{c:>12}")

    print(f"\nfull alpha sweep (FPR on the tied pair | power on the two real gaps):")
    print(f"  {'alpha':>7}" + "".join(f"{m:>26}" for m in ("percentile","BCa","Bonett-Price")))
    print(f"  {'':>7}" + "".join(f"{'FPR':>13}{'power':>13}" for _ in range(3)))
    for gi, al in enumerate(grid):
        row = f"  {al:>7.3f}"
        for m in ("percentile","bca","bonett_price"):
            row += f"{mrej[m][gi,0]/R:>13.3f}{mrej[m][gi,1:].mean()/R:>13.3f}"
        print(row)

    print(f"\nmatched false-positive rate on the tied pair:")
    print(f"  {'method':<22}{'alpha':>8}{'FPR':>8}{'power':>8}")
    for m in ("percentile", "bca", "bonett_price"):
        fpr = mrej[m][:, 0] / R
        gi = int(np.argmin(np.abs(fpr - 0.05)))
        print(f"  {m:<22}{grid[gi]:>8.3f}{fpr[gi]:>8.3f}{mrej[m][gi,1:].mean()/R:>8.3f}")
    print(f"\nreps {a.reps}, B={a.n_boot}, seed {a.seed}")


if __name__ == "__main__":
    main()
