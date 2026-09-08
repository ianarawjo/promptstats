"""Does PPI correction survive DIFFERENTIAL scale bias?

The harness sweeps scale/slope miscalibration (scale.compress / scale.expand
in build_judge_bias_sources) but only symmetrically: every scenario sets the
same slope for all groups, and slope is not in the factorial cross. So the
covered case is "the judge compresses everyone", never "the judge compresses
one condition more than the other" -- the scale analogue of differential
additive bias, which the harness does model precisely because it does NOT
cancel in a comparison.

This checks whether that gap matters, using the judge model's own form:

    judge = anchor + slope_group * (truth - anchor) + noise

Reported per configuration:
  - Type I error at a true effect of zero (the calibration question), and
  - CI coverage of the true effect at a real effect.

Diagnostic settings (25 seeds/cell); not for final numbers.

    python simulations/investigate_differential_scale_bias.py
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evalstats as es

warnings.filterwarnings("ignore")

N, N_LAB = 60, 15
BASE, SD_TRUE = 12.5, 4.0
ANCHOR = 12.5            # midpoint of the 0-25 scale
JUDGE_NOISE, HUMAN_NOISE = 2.5, 1.5


def run(seed, true_diff, slope_b, slope_s):
    rng = np.random.default_rng(seed)
    tb = np.clip(rng.normal(BASE, SD_TRUE, N), 0, 25)
    ts = np.clip(tb + true_diff + rng.normal(0, 1.0, N), 0, 25)
    jb = np.clip(np.round(ANCHOR + slope_b * (tb - ANCHOR)
                          + rng.normal(0, JUDGE_NOISE, N)), 0, 25)
    js = np.clip(np.round(ANCHOR + slope_s * (ts - ANCHOR)
                          + rng.normal(0, JUDGE_NOISE, N)), 0, 25)
    lab = rng.choice(N, size=N_LAB, replace=False)
    hb, hs = np.full(N, np.nan), np.full(N, np.nan)
    hb[lab] = np.clip(np.round(tb[lab] + rng.normal(0, HUMAN_NOISE, N_LAB)), 0, 25)
    hs[lab] = np.clip(np.round(ts[lab] + rng.normal(0, HUMAN_NOISE, N_LAB)), 0, 25)
    rows = []
    for i in range(N):
        rows.append({"item": f"i{i}", "condition": "baseline", "score": jb[i], "human_score": hb[i]})
        rows.append({"item": f"i{i}", "condition": "skill", "score": js[i], "human_score": hs[i]})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    ed = es.load_from(df, metric_cols={"score": "likert"}, col_map={"condition": "model"})
    naive = es.compare(ed, factors="model", metric="score", score_range=(0, 25), design="paired")
    ar = es.judge_alignment(ed, llm_metric="score", human_groundtruth="human_score",
                            selection="random")
    corr = es.compare(ed, factors="model", metric="score", score_range=(0, 25),
                      design="paired", alignment={"score": ar})
    return naive, corr, ar


def ci(result):
    r = result.to_dict()["pairwise"][0]
    lo, hi = r["ci_low"], r["ci_high"]
    if (r["a"], r["b"]) == ("baseline", "skill"):
        lo, hi = -hi, -lo
    return lo, hi


def cell(true_diff, slope_b, slope_s, n_seeds=25):
    ncov = ccov = nsig = csig = flagged = n = 0
    for s in range(1, n_seeds + 1):
        try:
            na, co, ar = run(s, true_diff, slope_b, slope_s)
            nlo, nhi = ci(na); clo, chi = ci(co)
        except Exception:
            continue
        n += 1
        ncov += (nlo <= true_diff <= nhi); ccov += (clo <= true_diff <= chi)
        nsig += not (nlo <= 0 <= nhi); csig += not (clo <= 0 <= chi)
        flagged += not ar.bias_check["passed"]
    return n, 100*ncov/n, 100*ccov/n, 100*nsig/n, 100*csig/n, 100*flagged/n


CONFIGS = [
    ("none            (1.00 / 1.00)", 1.00, 1.00),
    ("uniform compress(0.80 / 0.80)", 0.80, 0.80),
    ("DIFFERENTIAL    (1.00 / 0.80)", 1.00, 0.80),
    ("DIFFERENTIAL    (1.00 / 0.60)", 1.00, 0.60),
    ("DIFFERENTIAL    (0.80 / 1.20)", 0.80, 1.20),
]

if __name__ == "__main__":
    for td, label in ((0.0, "TRUE EFFECT = 0  (Type I error; nominal 5%)"),
                      (1.5, "TRUE EFFECT = 1.5  (coverage; nominal 95%)")):
        print(f"\n=== {label} ===")
        print(f"{'slope baseline/skill':<32}{'naive cov':>10}{'corr cov':>10}"
              f"{'naive sig':>11}{'corr sig':>10}{'bias flagged':>14}")
        for name, sb, ss in CONFIGS:
            n, nc, cc, ns_, cs_, fl = cell(td, sb, ss)
            print(f"{name:<32}{nc:>9.0f}%{cc:>9.0f}%{ns_:>10.0f}%{cs_:>9.0f}%{fl:>13.0f}%")
