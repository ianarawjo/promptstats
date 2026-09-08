"""Battle-test harness for the new between-subjects (design="unpaired") path
(2026-08-15) -- evalstats/core/unpaired.py + compare(design=...) routing in
api.py.

Two parts:

1. A wide crash/sanity grid: every combination of score_type x k x group-size
   balance x PPI-alignment x seed, asserting the engine never crashes and
   every returned GroupComparisonResult is internally consistent (correct
   group/pair counts, CI ordering, p in [0,1], to_dict/to_frame/
   groups_to_frame all work).
2. A lightweight Type-I / power calibration check: under a true null (all
   groups drawn identically), the omnibus test and the Bonferroni/Holm-
   corrected pairwise family should each reject at ~alpha, not far above
   it -- this is the one thing the crash grid can't catch, since wrong-but-
   non-crashing FWER math still "works" mechanically. Under a real effect,
   the omnibus test should have reasonable power.

Deliberately modest N_REPS/N_BOOT for a battle-test script, not a final
calibration number -- see feedback_speed_up_diagnostic_scripts memory.

Not part of the harness / --official-tests: standalone script. Run directly:

    .venv/bin/python simulations/investigate_unpaired_battle_test.py
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd

from evalstats.core.unpaired import compare_unpaired, GroupComparisonResult
from evalstats.alignment import judge_alignment
from evalstats.loader import load_from

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260815
N_BOOT = 400  # modest -- battle test, not a final number


def _rng(seed):
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def make_group_df(score_type, means, n_per_group, seed):
    """means: dict[label, float] on a [0,1]-ish latent scale for all types."""
    rng = _rng(seed)
    rows = []
    for g, mean in means.items():
        n = n_per_group[g] if isinstance(n_per_group, dict) else n_per_group
        for i in range(n):
            if score_type == "binary":
                p = float(np.clip(mean, 0.02, 0.98))
                score = float(rng.binomial(1, p))
            elif score_type == "continuous":
                score = float(np.clip(rng.normal(mean, 0.15), 0, 1))
            elif score_type == "likert":
                cats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                center = 1 + mean * 4
                probs = np.exp(-0.5 * ((cats - center) / 1.1) ** 2)
                probs /= probs.sum()
                score = float(rng.choice(cats, p=probs))
            elif score_type == "wide_ordinal":
                # 0-100 whole numbers: still the discrete path, just a wide scale.
                score = float(np.clip(np.round(rng.normal(mean * 100, 15)), 0, 100))
            else:
                raise ValueError(score_type)
            rows.append({"group": g, "item": f"{g}_{i}", "score": score})
    return pd.DataFrame(rows)


def add_sparse_human_col(df, n_labeled_per_group, seed, noise=0.05):
    rng = _rng(seed)
    human = np.full(len(df), np.nan)
    for g in df["group"].unique():
        idx = df.index[df["group"] == g].to_numpy()
        chosen = rng.choice(idx, size=min(n_labeled_per_group, len(idx)), replace=False)
        for j in chosen:
            base = df.loc[j, "score"]
            human[j] = base + rng.normal(0, noise) if not np.isnan(base) else np.nan
    df = df.copy()
    df["human_score"] = human
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: crash / sanity grid
# ─────────────────────────────────────────────────────────────────────────────

def run_crash_grid():
    print("=" * 78)
    print("PART 1: crash/sanity grid")
    print("=" * 78)
    score_types = ["binary", "continuous", "likert", "wide_ordinal"]
    k_values = [2, 3, 4, 6]
    balance_modes = ["balanced", "unbalanced"]
    ppi_modes = [False, True]
    seeds = [0, 1, 2]

    n_total = 0
    n_failed = 0
    failures = []

    for score_type, k, balance, ppi, seed in itertools.product(
        score_types, k_values, balance_modes, ppi_modes, seeds
    ):
        n_total += 1
        labels = [f"G{i}" for i in range(k)]
        rng = _rng(seed)
        # spread means across [0.3, 0.7] so no two groups are identical, but
        # not so separated that small groups degenerate (e.g. all-0/all-1 binary).
        means = {lbl: 0.3 + 0.4 * (i / max(k - 1, 1)) for i, lbl in enumerate(labels)}
        if balance == "balanced":
            n_per_group = 30
        else:
            n_per_group = {lbl: int(rng.integers(8, 60)) for lbl in labels}

        try:
            df = make_group_df(score_type, means, n_per_group, seed=seed * 1000 + k)
            alignment = None
            if ppi:
                df = add_sparse_human_col(df, n_labeled_per_group=8, seed=seed * 1000 + k)
                evaldata = load_from(df, col_map={"model": "group", "item": "item"})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar = judge_alignment(evaldata, llm_metric="score", human_groundtruth="human_score")
                alignment = {"score": ar}

            r = compare_unpaired(
                df, factor_col="group", metric_col="score",
                alignment=alignment, n_boot=N_BOOT, rng=seed,
            )

            # ── internal-consistency assertions ──────────────────────────────
            assert isinstance(r, GroupComparisonResult)
            assert len(r.groups) == k, f"expected {k} groups, got {len(r.groups)}"
            n_pairs_expected = k * (k - 1) // 2
            assert len(r.pairwise) == n_pairs_expected, (
                f"expected {n_pairs_expected} pairs, got {len(r.pairwise)}"
            )
            assert (k >= 3) == (r.omnibus_test_name is not None), (
                f"omnibus presence mismatch at k={k}: {r.omnibus_test_name!r}"
            )
            for p in r.pairwise:
                assert p.ci_low <= p.ci_high, f"CI inverted: {p.ci_low} > {p.ci_high}"
                assert 0.0 <= p.p_value <= 1.0, f"p out of range: {p.p_value}"
                assert 0.0 <= p.raw_p_value <= 1.0, f"raw p out of range: {p.raw_p_value}"
                assert p.n_a > 0 and p.n_b > 0
            for g in r.groups:
                assert g.ci_low <= g.ci_high, f"group CI inverted: {g.label}"
                assert g.n > 0
            if ppi:
                assert r.ppi_applied is True

            # ── reporting surface doesn't crash ──────────────────────────────
            d = r.to_dict()
            assert d["design"] == "unpaired"
            frame = r.to_frame()
            assert len(frame) == n_pairs_expected
            gframe = r.groups_to_frame()
            assert len(gframe) == k
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r.summary()
            assert len(buf.getvalue()) > 0

        except Exception as e:  # noqa: BLE001 -- battle test, want to catch everything
            n_failed += 1
            failures.append((score_type, k, balance, ppi, seed, repr(e)))

    print(f"Total combinations: {n_total}, failures: {n_failed}")
    if failures:
        print("\nFAILURES:")
        for f in failures[:20]:
            print(f"  score_type={f[0]:<10s} k={f[1]} balance={f[2]:<11s} ppi={f[3]!s:<5s} seed={f[4]}  ->  {f[5]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    else:
        print("All combinations passed internal-consistency checks.")
    return n_failed


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Type-I / power calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(score_type, k, n_per_group, n_reps, alpha=0.05, effect=0.0, label=""):
    """effect=0.0 -> null (all groups identical); effect>0 -> last group shifted up."""
    rng = _rng(SEED)
    omnibus_rejections = 0
    any_pairwise_rejections = 0  # family-wise: at least one pair flagged significant
    n_omnibus = 0
    t0 = time.time()

    for rep in range(n_reps):
        seed = int(rng.integers(0, 2**31 - 1))
        labels = [f"G{i}" for i in range(k)]
        means = {lbl: 0.5 for lbl in labels}
        if effect > 0:
            means[labels[-1]] = 0.5 + effect
        df = make_group_df(score_type, means, n_per_group, seed=seed)
        r = compare_unpaired(df, factor_col="group", metric_col="score", n_boot=N_BOOT, rng=seed)
        if r.omnibus_test_name is not None:
            n_omnibus += 1
            if r.omnibus_p_value < alpha:
                omnibus_rejections += 1
        if any(p.significant for p in r.pairwise):
            any_pairwise_rejections += 1

    elapsed = time.time() - t0
    omnibus_rate = omnibus_rejections / n_omnibus if n_omnibus else float("nan")
    pairwise_rate = any_pairwise_rejections / n_reps
    kind = "Type-I (null)" if effect == 0.0 else f"Power (effect={effect})"
    print(f"  [{label}] {kind}: omnibus reject rate = {omnibus_rate:.3f}  "
          f"(any-pair FWER reject rate = {pairwise_rate:.3f})  n_reps={n_reps}  ({elapsed:.1f}s)")
    return omnibus_rate, pairwise_rate


def run_calibration_suite():
    print()
    print("=" * 78)
    print("PART 2: Type-I / power calibration (alpha=0.05)")
    print("=" * 78)
    alpha = 0.05
    n_reps = 300

    print("\n-- continuous, k=3, n=30/group --")
    ty1_cont, fw1_cont = run_calibration("continuous", 3, 30, n_reps, alpha=alpha, effect=0.0, label="null")
    pw1_cont, _ = run_calibration("continuous", 3, 30, 150, alpha=alpha, effect=0.25, label="effect")

    print("\n-- binary, k=3, n=30/group --")
    ty1_bin, fw1_bin = run_calibration("binary", 3, 30, n_reps, alpha=alpha, effect=0.0, label="null")
    pw1_bin, _ = run_calibration("binary", 3, 30, 150, alpha=alpha, effect=0.3, label="effect")

    print("\n-- likert, k=4, n=25/group --")
    ty1_lik, fw1_lik = run_calibration("likert", 4, 25, n_reps, alpha=alpha, effect=0.0, label="null")

    print()
    print("Interpretation:")
    print(f"  Nominal alpha = {alpha}. Type-I rows should be near or below alpha")
    print(f"  (some conservatism from Bonferroni/Holm is expected and fine; wildly")
    print(f"  ABOVE alpha, e.g. > ~0.10, would indicate a real FWER-control bug).")
    print(f"  Effect rows should show clearly elevated rejection rates vs. the null")
    print(f"  rows above them (confirms the tests have power, not just conservatism).")

    return {
        "continuous_typeI_omnibus": ty1_cont, "continuous_typeI_fwer": fw1_cont, "continuous_power": pw1_cont,
        "binary_typeI_omnibus": ty1_bin, "binary_typeI_fwer": fw1_bin, "binary_power": pw1_bin,
        "likert_typeI_omnibus": ty1_lik, "likert_typeI_fwer": fw1_lik,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Pareto-front (secondary_metric=) crash grid
# ─────────────────────────────────────────────────────────────────────────────

def run_pareto_grid():
    print()
    print("=" * 78)
    print("PART 3: Pareto-front (secondary_metric=) crash/sanity grid")
    print("=" * 78)
    k_values = [2, 3, 4]
    balance_modes = ["balanced", "unbalanced"]
    directions = ["min", "max"]
    ppi_modes = [False, True]
    seeds = [0, 1]

    n_total = 0
    n_failed = 0
    failures = []

    for k, balance, direction, ppi, seed in itertools.product(
        k_values, balance_modes, directions, ppi_modes, seeds
    ):
        n_total += 1
        labels = [f"G{i}" for i in range(k)]
        rng = _rng(seed * 7919 + k)
        score_means = {lbl: 0.3 + 0.4 * (i / max(k - 1, 1)) for i, lbl in enumerate(labels)}
        secondary_means = {lbl: 100 + 40 * (i / max(k - 1, 1)) for i, lbl in enumerate(labels)}
        n_per_group = 30 if balance == "balanced" else {lbl: int(rng.integers(10, 55)) for lbl in labels}

        try:
            df = make_group_df("continuous", score_means, n_per_group, seed=seed * 1000 + k)
            sec_rng = _rng(seed * 1000 + k + 500)
            df["secondary"] = [
                float(sec_rng.normal(secondary_means[g], 15)) for g in df["group"]
            ]
            alignment = None
            if ppi:
                df = add_sparse_human_col(df, n_labeled_per_group=8, seed=seed * 1000 + k)
                evaldata = load_from(df, col_map={"model": "group", "item": "item"})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar = judge_alignment(evaldata, llm_metric="score", human_groundtruth="human_score")
                alignment = {"score": ar}

            r = compare_unpaired(
                df, factor_col="group", metric_col="score",
                secondary_metric={"secondary": direction},
                alignment=alignment, n_boot=N_BOOT, rng=seed,
            )

            assert r.pareto is not None
            assert set(r.pareto_status.keys()) == set(r.labels)
            for lbl in r.labels:
                p = r.pareto_frontier_probability[lbl]
                assert 0.0 <= p <= 1.0, f"p_frontier out of range for {lbl}: {p}"
                assert r.pareto_status[lbl].status in {"frontier", "dominated", "ambiguous"}
            d = r.to_dict()
            assert "pareto" in d

            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r.summary()
            assert "Trade-off" in buf.getvalue()

        except Exception as e:  # noqa: BLE001
            n_failed += 1
            failures.append((k, balance, direction, ppi, seed, repr(e)))

    print(f"Total combinations: {n_total}, failures: {n_failed}")
    if failures:
        print("\nFAILURES:")
        for f in failures[:20]:
            print(f"  k={f[0]} balance={f[1]:<11s} direction={f[2]:<4s} ppi={f[3]!s:<5s} seed={f[4]}  ->  {f[5]}")
    else:
        print("All combinations passed internal-consistency checks.")
    return n_failed


if __name__ == "__main__":
    n_failed = run_crash_grid()
    results = run_calibration_suite()
    n_pareto_failed = run_pareto_grid()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Crash grid failures: {n_failed}")
    print(f"Pareto grid failures: {n_pareto_failed}")
    flags = []
    for k, v in results.items():
        if "typeI" in k and not np.isnan(v) and v > 0.10:
            flags.append(f"  FLAG: {k} = {v:.3f} (> 0.10, well above nominal alpha=0.05)")
    if flags:
        print("Calibration flags:")
        for f in flags:
            print(f)
    else:
        print("No calibration flags (all Type-I rates within a reasonable band of alpha=0.05).")
