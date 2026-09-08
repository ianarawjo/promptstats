#!/usr/bin/env python3
"""Numbers behind the small-n summarization demonstration (Sec. 8.1).

Ground truth is the full 500-item XSum ROUGE-L corpus collected by
``collect_summarization_rouge.py``. We repeatedly draw the 15-item eval set a
developer would actually build and ask, for each analysis they might run, how
often it reaches the right conclusion.

Three models, chosen so one pair is genuinely tied and two are genuinely apart:
that way false positives and power are measured on the same three systems.

    python simulations/demo_summarization_smalln.py            # headline numbers
    python simulations/demo_summarization_smalln.py --n 30
    python simulations/demo_summarization_smalln.py --draw     # one example eval set
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from evalstats.core.resampling import logit_t_ci_1d
from evalstats.core.stats_utils import rescaled_ci

CSV = "simulations/out/summarization_rouge.csv"
TRIO = ["meta-llama/llama-3.1-8b-instruct",
        "google/gemma-3-12b-it",
        "mistralai/ministral-3b-2512"]
SHORT = ["llama-3.1-8b", "gemma-3-12b", "ministral-3b"]


def load():
    d = pd.read_csv(CSV)
    w = d.pivot(index="item_id", columns="model", values="rouge_l")
    return w[TRIO].to_numpy()


def es_ci(x, alpha=0.05):
    """The interval compare() actually reports for paired [0,1] scores: logit-t
    on the differences, rescaled from their [-1, 1] span (evalstats/core/paired.py)."""
    return rescaled_ci(logit_t_ci_1d, np.asarray(x, float), alpha, -1.0, 1.0)


def marg_ci(x, alpha=0.05):
    """Marginal logit-t, the per-model interval compare() plots."""
    return rescaled_ci(logit_t_ci_1d, np.asarray(x, float), alpha, 0.0, 1.0)


def t_ci(x, alpha=0.05):
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return x.mean() - t * se, x.mean() + t * se


def scipy_boot(x, B, rng, method):
    """What a developer following an LLM assistant's advice actually runs:
    scipy.stats.bootstrap on the paired differences."""
    r = stats.bootstrap((x,), np.mean, method=method, confidence_level=0.95,
                        n_resamples=B, random_state=rng, vectorized=False)
    return r.confidence_interval.low, r.confidence_interval.high


def pct_ci(x, B, rng):
    return scipy_boot(x, B, rng, "percentile")


def bca_ci(x, B, rng):
    return scipy_boot(x, B, rng, "BCa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--boot-reps", type=int, default=8000,
                    help="reps for the bootstrap arms, which cost ~B x more")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", action="store_true",
                    help="print one example 15-item eval set instead of the rates")
    a = ap.parse_args()

    X = load()
    truth = X.mean(0)
    N, n = len(X), a.n
    PAIRS = list(itertools.combinations(range(3), 2))  # (0,1) tied, (1,2)&(0,2) real
    a_sid = 1 - (1 - 0.05) ** (1 / 3)

    print(f"corpus: {N} items x 3 models   n = {n}")
    for j, s in enumerate(SHORT):
        print(f"   {s:<14} {truth[j]:.4f}")
    for i, j in PAIRS:
        df = X[:, i] - X[:, j]
        print(f"   {SHORT[i]} - {SHORT[j]}: {df.mean():+.4f}  "
              f"d_z={df.mean()/df.std():+.3f}  p={stats.ttest_rel(X[:,i],X[:,j])[1]:.2g}")

    if a.draw:
        rng = np.random.default_rng(a.seed)
        s = X[rng.choice(N, n, replace=False)]
        print(f"\none {n}-item eval set (seed {a.seed}):")
        for j, nm in enumerate(SHORT):
            lo, hi = t_ci(s[:, j])
            print(f"   {nm:<14} mean {s[:,j].mean():.4f}   marginal 95% CI [{lo:.4f}, {hi:.4f}]")
        for i, j in PAIRS:
            df = s[:, i] - s[:, j]
            lo, hi = t_ci(df)
            print(f"   {SHORT[i]} - {SHORT[j]}: {df.mean():+.4f}  paired 95% CI "
                  f"[{lo:+.4f}, {hi:+.4f}]  {'EXCLUDES 0' if lo>0 or hi<0 else 'includes 0'}")
        return

    # ---- cheap arms: no bootstrap, so run them at full reps ----
    rng = np.random.default_rng(a.seed)
    ovl = np.zeros(3); pair_t = np.zeros(3); pair_sid = np.zeros(3); stud = np.zeros(3)
    simul_marg = simul_unc = simul_sid = 0
    for _ in range(a.reps):
        s = X[rng.choice(N, n, replace=False)]
        cis = [marg_ci(s[:, j]) for j in range(3)]
        simul_marg += all(lo <= truth[j] <= hi for j, (lo, hi) in enumerate(cis))
        ok_u = ok_s = True
        for k, (i, j) in enumerate(PAIRS):
            ovl[k] += cis[i][0] > cis[j][1] or cis[j][0] > cis[i][1]
            df = s[:, i] - s[:, j]; tr = truth[i] - truth[j]
            lo, hi = es_ci(df);        pair_t[k] += lo > 0 or hi < 0;  ok_u &= lo <= tr <= hi
            lo, hi = es_ci(df, a_sid); pair_sid[k] += lo > 0 or hi < 0; ok_s &= lo <= tr <= hi
            lo, hi = t_ci(df);         stud[k] += lo > 0 or hi < 0
        simul_unc += ok_u; simul_sid += ok_s
    R = a.reps

    # ---- bootstrap arms, on the tied pair (FPR) and the real pairs (power) ----
    rng = np.random.default_rng(a.seed + 1)
    bt = {"percentile": np.zeros(3), "BCa": np.zeros(3)}
    bw = {"percentile": 0.0, "BCa": 0.0}
    tw = 0.0
    for _ in range(a.boot_reps):
        s = X[rng.choice(N, n, replace=False)]
        for k, (i, j) in enumerate(PAIRS):
            df = s[:, i] - s[:, j]
            for nm, ci in (("percentile", pct_ci(df, a.n_boot, rng)),
                           ("BCa", bca_ci(df, a.n_boot, rng))):
                lo, hi = ci
                bt[nm][k] += lo > 0 or hi < 0
                bw[nm] += hi - lo
            lo, hi = es_ci(df); tw += hi - lo
    BR = a.boot_reps * 3

    print(f"\n{'method':<22}{'FPR (tied pair)':>17}{'power (2 real pairs)':>22}{'mean width':>13}")
    print(f"  {'overlap of marginal':<20}{ovl[0]/R:>17.3f}{ovl[1:].mean()/R:>22.3f}{'--':>13}")
    print(f"  {'95% error bars':<20}{'':>17}{'':>22}{'':>13}")
    print(f"  {'percentile bootstrap':<20}{bt['percentile'][0]/a.boot_reps:>17.3f}"
          f"{bt['percentile'][1:].mean()/a.boot_reps:>22.3f}{bw['percentile']/BR:>13.4f}")
    print(f"  {'BCa bootstrap':<20}{bt['BCa'][0]/a.boot_reps:>17.3f}"
          f"{bt['BCa'][1:].mean()/a.boot_reps:>22.3f}{bw['BCa']/BR:>13.4f}")
    print(f"  {'evalstats (logit-t)':<20}{pair_t[0]/R:>17.3f}{pair_t[1:].mean()/R:>22.3f}{tw/BR:>13.4f}")
    print(f"  {'  + Sidak':<20}{pair_sid[0]/R:>17.3f}{pair_sid[1:].mean()/R:>22.3f}{'--':>13}")
    print(f"  {'(paired t, ref only)':<20}{stud[0]/R:>17.3f}{stud[1:].mean()/R:>22.3f}{'--':>13}")

    print(f"\nsimultaneous coverage of all three intervals (nominal .95):")
    print(f"  marginal, uncorrected  {simul_marg/R:.3f}")
    print(f"  paired,   uncorrected  {simul_unc/R:.3f}")
    print(f"  paired,   Sidak        {simul_sid/R:.3f}")
    print(f"\nreps {a.reps} (bootstrap arms {a.boot_reps} x B={a.n_boot}), seed {a.seed}")


if __name__ == "__main__":
    main()
