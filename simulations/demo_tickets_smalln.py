#!/usr/bin/env python3
"""Numbers behind the small-n prompt-comparison demonstration (Sec. 8.1).

Data is the support-ticket classifier eval shipped with the toolkit website
(website/notebooks/support_ticket_eval_multirun.csv): 8 prompts x 120 tickets
x 5 runs of real model output. The website's own 20- and 40-item eval files are
exact subsets of these 120 tickets.

All 8 prompts are used. Selecting a subset would set the headline number rather
than measure it: the rate at which top-by-mean ships a worse prompt runs from
0.24 to 0.52 across different 4-prompt subsets, and it depends on how many
candidates are tried, which is the practice under examination.

Truth is all 120 tickets averaged over the 5 runs. A draw is n of those tickets
scored on a single run, which is what a practitioner actually executes.

    python simulations/demo_tickets_smalln.py
    python simulations/demo_tickets_smalln.py --n 40
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from evalstats.core.resampling import bonett_price_paired_ci

CSV = "website/notebooks/support_ticket_eval_multirun.csv"


def boot_quantiles(x, B, rng):
    """Bootstrap the mean once; return (percentile, BCa) CI functions of alpha."""
    n = len(x)
    bs = np.sort(x[rng.integers(0, n, (B, n))].mean(1))
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
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--boot-reps", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_csv(CSV)
    W = d.pivot_table(index="input_id", columns="prompt_id", values="correct",
                      aggfunc="mean")
    P = list(W.columns)
    truth = W.mean().to_numpy()
    best = int(np.argmax(truth))
    K, N, n = len(P), len(W), a.n
    runs = {r: d[d.run_idx == r].pivot(index="input_id", columns="prompt_id",
                                       values="correct")[P].to_numpy()
            for r in sorted(d.run_idx.unique())}

    # Acceptable ships: prompts the full 120 tickets cannot separate from the best.
    ok = {best} | {j for j in range(K) if j != best
                   and stats.ttest_rel(W.iloc[:, best], W.iloc[:, j])[1] > .05}
    PAIRS = list(itertools.combinations(range(K), 2))
    NULL = [k for k, (i, j) in enumerate(PAIRS)
            if stats.ttest_rel(W.iloc[:, i], W.iloc[:, j])[1] > .05]
    REAL = [k for k in range(len(PAIRS)) if k not in NULL]

    print(f"truth, {N} tickets x {len(runs)} runs:")
    for j in np.argsort(-truth):
        tag = "  <- best" if j == best else ("  (tied with best)" if j in ok else "")
        print(f"   {P[j]:<22}{truth[j]:.4f}{tag}")
    print(f"\n{len(PAIRS)} pairs: {len(NULL)} indistinguishable on all {N}, {len(REAL)} real")
    print(f"n = {n} tickets, {a.reps} draws (bootstrap arms {a.boot_reps} x B={a.n_boot})")

    # ---- beats 1, 2, 4: selection, error bars, tie band -------------------
    rng = np.random.default_rng(a.seed)
    hit = okc = sep1 = sep1_bad = 0
    reg, band, in_band = [], [], 0
    bp_rej = np.zeros(len(PAIRS)); bp_cov = np.zeros(len(PAIRS))
    for _ in range(a.reps):
        X = runs[rng.choice(list(runs))]
        s = X[rng.choice(N, n, replace=False)]
        m = s.mean(0)
        sel = int(np.argmax(m))
        hit += sel == best; okc += sel in ok; reg.append(truth[best] - truth[sel])
        se = np.sqrt(np.clip(m * (1 - m), 0, None) / n)
        o = np.argsort(-m)
        if m[o[0]] - se[o[0]] > m[o[1]] + se[o[1]]:
            sep1 += 1; sep1_bad += o[0] not in ok
        tb = [sel]
        for k, (i, j) in enumerate(PAIRS):
            lo, hi = bonett_price_paired_ci(s[:, i], s[:, j], .05)
            bp_rej[k] += lo > 0 or hi < 0
            bp_cov[k] += lo <= truth[i] - truth[j] <= hi
            if sel in (i, j):
                other = j if sel == i else i
                if not (lo > 0 or hi < 0):
                    tb.append(other)
        band.append(len(tb)); in_band += best in tb
    R = a.reps; reg = np.array(reg)

    print(f"\n--- 1. pick the top prompt by mean (the eval-dashboard default) ---")
    print(f"   picks the true best                {hit/R:.3f}")
    print(f"   picks an acceptable prompt         {okc/R:.3f}   (ships a worse one {1-okc/R:.3f})")
    print(f"   mean accuracy given up             {reg.mean():.4f}")
    print(f"   gives up more than 10 points       {np.mean(reg>.10):.3f}   (worst {reg.max():.3f})")
    print(f"\n--- 2. add +/-1 SE error bars and require the top bar to clear the runner-up ---")
    print(f"   declares a winner                  {sep1/R:.4f}   ({sep1} of {R} draws)")
    print(f"\n--- 4. evalstats ---")
    print(f"   tie band size                      {np.mean(band):.2f} of {K}")
    print(f"   band contains the true best        {in_band/R:.3f}")
    print(f"   Bonett-Price FPR / power / cov     {bp_rej[NULL].mean()/R:.3f} / "
          f"{bp_rej[REAL].mean()/R:.3f} / {bp_cov.mean()/R:.3f}")

    # ---- beat 3: the bootstraps -------------------------------------------
    rng = np.random.default_rng(a.seed + 1)
    br = {m: np.zeros(len(PAIRS)) for m in ("percentile", "bca")}
    bc = {m: np.zeros(len(PAIRS)) for m in ("percentile", "bca")}
    for _ in range(a.boot_reps):
        X = runs[rng.choice(list(runs))]
        s = X[rng.choice(N, n, replace=False)]
        for k, (i, j) in enumerate(PAIRS):
            df = s[:, i] - s[:, j]
            pct, bca = boot_quantiles(df, a.n_boot, rng)
            for m, ci in (("percentile", pct(.05)), ("bca", bca(.05))):
                lo, hi = ci
                br[m][k] += lo > 0 or hi < 0
                bc[m][k] += lo <= truth[i] - truth[j] <= hi
    B = a.boot_reps
    print(f"\n--- 3. bootstrap intervals on the same draws ---")
    print(f"   {'method':<14}{'FPR':>8}{'power':>8}{'coverage':>11}")
    for m in ("percentile", "bca"):
        print(f"   {m:<14}{br[m][NULL].mean()/B:>8.3f}{br[m][REAL].mean()/B:>8.3f}"
              f"{bc[m].mean()/B:>11.3f}")


if __name__ == "__main__":
    main()
