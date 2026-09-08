#!/usr/bin/env python3
"""Stress `evalstats analyze`'s score-kind/range inference over the metric
shapes an eval spreadsheet actually arrives in.

When --score-range is omitted, the CLI infers the data kind from the values
and declares both the kind and (where one can be established) the observed
range to the engine, so the bounded methods apply instead of the
bounds-agnostic t-interval fallback. Two things can go wrong with that, and
neither is caught by unit tests over one dataset:

  1. a metric routed to the wrong CI family -- most likely a measurement
     that merely happens to be whole-numbered (token counts, latencies)
     being read as an ordinal rating scale, and
  2. the printed notice disagreeing with what the engine then ran, since
     the engine repeats the discreteness check on its own.

Two passes:
  detect  the inference alone, one row per metric shape (fast, no analysis)
  e2e     the real `analyze` on synthetic 3-condition data, both the plain
          and the --human-groundtruth judge path, checking that nothing
          raises, CIs are finite and ordered, and no CI escapes the
          declared range

Exits non-zero if the e2e pass flags anything, so it can gate a change.

    python -m simulations.stress_cli_score_detection
    python -m simulations.stress_cli_score_detection --mode detect
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalstats import cli  # noqa: E402

N_ITEMS = 40
GROUPS = ["A", "B", "C"]
CI_RE = re.compile(r"([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)")
RANGE_RE = re.compile(r"range, --score-range (-?[\d.eE+]+) (-?[\d.eE+]+),")
GIVEN_RE = re.compile(r"score range: \[(-?[\d.eE+]+), (-?[\d.eE+]+)\]")


def shapes():
    """(name, values_fn) -- values_fn(rng, n, group_index) -> scores.

    The group index shifts each condition slightly so the comparison has
    something to find; for bounded metrics that shift can push values past
    the nominal ceiling, which is itself a case worth covering.
    """
    ramp = lambda lo, hi: (lambda rng, n, g: rng.uniform(lo, hi, n) + g * (hi - lo) * 0.05)
    return [
        ("binary", lambda rng, n, g: rng.binomial(1, 0.4 + 0.1 * g, n).astype(float)),
        ("binary lopsided", lambda rng, n, g: rng.binomial(1, 0.03 + 0.01 * g, n).astype(float)),
        ("unit interval", ramp(0.0, 1.0)),
        ("unit near ceiling", ramp(0.9, 1.0)),
        ("unit with 0s and 1s", lambda rng, n, g: np.clip(rng.uniform(-0.1, 1.1, n) + g * 0.02, 0, 1)),
        ("likert 1-5", lambda rng, n, g: rng.integers(1, 6, n).astype(float)),
        ("likert 1-5 shifted", lambda rng, n, g: np.clip(rng.integers(1, 6, n) + g, 1, 5).astype(float)),
        ("likert 1-7", lambda rng, n, g: rng.integers(1, 8, n).astype(float)),
        ("likert 0-10", lambda rng, n, g: rng.integers(0, 11, n).astype(float)),
        ("half-point 1-5", lambda rng, n, g: np.round(rng.uniform(1, 5, n) * 2) / 2),
        ("bipolar -2..2", lambda rng, n, g: rng.integers(-2, 3, n).astype(float)),
        ("percent int 0-100", lambda rng, n, g: rng.integers(0, 101, n).astype(float)),
        ("percent float", ramp(0.0, 100.0)),
        ("grade 0-4", lambda rng, n, g: np.round(rng.uniform(0, 4, n), 2)),
        ("latency ms", ramp(50.0, 5000.0)),
        ("token counts", lambda rng, n, g: rng.integers(10, 4000, n).astype(float)),
        ("log-odds", lambda rng, n, g: rng.normal(g * 0.3, 2, n)),
        ("huge scale", ramp(0.0, 1e6)),
        ("tiny costs", lambda rng, n, g: np.round(rng.uniform(1e-4, 5e-2, n), 5)),
        ("near-constant", lambda rng, n, g: np.where(rng.random(n) < 0.05, 4.0, 3.0)),
        ("constant", lambda rng, n, g: np.full(n, 3.0)),
        ("two-valued 1/5", lambda rng, n, g: rng.choice([1.0, 5.0], n)),
        ("with NaNs", lambda rng, n, g: np.where(rng.random(n) < 0.15, np.nan, rng.integers(1, 6, n))),
        ("likert w/ 99 typo", lambda rng, n, g: np.where(np.arange(n) == 3, 99.0, rng.integers(1, 6, n).astype(float))),
        ("likert w/ 0 sentinel", lambda rng, n, g: np.where(rng.random(n) < 0.1, 0.0, rng.integers(1, 6, n).astype(float))),
        ("groups differ in range", lambda rng, n, g: rng.uniform(0, 10 ** (g + 1), n)),
        ("one distinct per group", lambda rng, n, g: np.full(n, float(g + 1))),
        ("negatives + positives grid", lambda rng, n, g: rng.integers(-5, 6, n).astype(float) * 0.5),
    ]


def _kind_from(out: str) -> str:
    for marker, kind in (
        ("binary 0/1 (detected)", "binary"),
        ("in [0, 1] (detected)", "unit"),
        ("inferred: discrete scores", "likert"),
        ("too fine", "cont(grid)"),
        ("inferred: continuous", "cont"),
        ("no range can be inferred", "const"),
    ):
        if marker in out:
            return kind
    return "?"


def _declared_from(out: str):
    m = GIVEN_RE.search(out) or RANGE_RE.search(out)
    return (float(m.group(1)), float(m.group(2))) if m else None


def run_detect(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in shapes():
        vals = np.asarray(fn(rng, 60, 0), dtype=float)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_type, score_range = cli._resolve_score_kind(vals, None)
        out = buf.getvalue()
        rows.append((
            name, str(score_type),
            f"({score_range[0]:g},{score_range[1]:g})" if score_range else "-",
            "LOUD" if "NO --score-range GIVEN" in out else "quiet",
            _kind_from(out),
        ))
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'metric':{w}}{'score_type':>12}{'range':>22}{'notice':>8}{'kind':>13}")
    for r in rows:
        print(f"{r[0]:{w}}{r[1]:>12}{r[2]:>22}{r[3]:>8}{r[4]:>13}")
    print()


def _frame(fn, judge: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labeled = {f"i{i}" for i in range(N_ITEMS // 2)}
    rows = []
    for g_i, g in enumerate(GROUPS):
        for i, x in enumerate(fn(rng, N_ITEMS, g_i)):
            row = {"model": g, "item": f"i{i}", "score": float(x)}
            if judge:
                row["human_score"] = float(x) if f"i{i}" in labeled else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _args(path: Path, judge: bool) -> argparse.Namespace:
    kw = dict(
        file=path, format="long", sheet="0", evaluator_mode="aggregate", ci=None,
        ci_style="gradient", method="auto", backend="statsmodels", n_bootstrap=200,
        correction="auto", spread_percentiles=(10.0, 90.0), reference="grand_mean",
        failure_threshold=None, statistic="mean", template_model_collapse="as_runs",
        simultaneous_ci=True, omnibus=False, p_values=True, pairwise_test="auto",
        top_pairwise=5, brief=False, out=None, show_rank_probabilities=False,
        score_range=None,
    )
    if judge:
        kw.update(human_groundtruth="human_score", metric="score", factor=None,
                  label_selection="random", seed=0)
    return argparse.Namespace(**kw)


def _check_cis(out: str, declared) -> list[str]:
    problems = []
    parts = out.split("--- Mean Performance", 1)
    if len(parts) < 2:
        return ["no Mean Performance block"]
    for line in parts[1].split("---", 1)[0].splitlines():
        m = CI_RE.search(line)
        if not m:
            continue
        mean, lo, hi = (float(x) for x in m.groups())
        if not all(np.isfinite(v) for v in (mean, lo, hi)):
            problems.append("non-finite CI")
        if lo > hi:
            problems.append("inverted CI")
        if declared is not None and (lo < declared[0] - 1e-6 or hi > declared[1] + 1e-6):
            problems.append(f"CI [{lo:g},{hi:g}] outside declared {declared}")
    return problems


def run_e2e(seed: int, tmp_dir: Path) -> int:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in shapes():
        for judge in (False, True):
            path = tmp_dir / f"{name.replace(' ', '_').replace('/', '')}_{int(judge)}.csv"
            _frame(fn, judge, seed).to_csv(path, index=False)
            buf, err = io.StringIO(), None
            try:
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cli._cmd_analyze(_args(path, judge))
            except SystemExit:
                # _die() -- a refusal with a message, not a crash.
                err = "refused: " + (buf.getvalue().strip().splitlines() or ["?"])[-1][:48]
            except Exception as exc:  # noqa: BLE001 -- reporting, not handling
                err = f"{type(exc).__name__}: {exc}"[:70]
            out = buf.getvalue()
            declared = _declared_from(out)
            method = re.search(r"--- Pairwise Comparisons \(([^)]+)\)", out)
            rows.append((
                name, "judge" if judge else "plain", _kind_from(out),
                f"({declared[0]:g},{declared[1]:g})" if declared else "-",
                (method.group(1).replace(" CIs", "") if method else "")[:26],
                err or "; ".join(_check_cis(out, declared))[:58] or "ok",
            ))
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'metric':{w}}{'path':>7}{'kind':>12}{'range':>22}  {'CI method':<28}status")
    for r in rows:
        print(f"{r[0]:{w}}{r[1]:>7}{r[2]:>12}{r[3]:>22}  {r[4]:<28}{r[5]}")
    flagged = [r for r in rows if r[5] != "ok"]
    print(f"\n{len(rows) - len(flagged)}/{len(rows)} clean, {len(flagged)} flagged")
    # Missing cells are refused by the bootstrap path by design (it points the
    # user at method='lmm'), so they are reported but don't fail the run.
    real = [r for r in flagged if not r[5].startswith("refused")]
    for r in real:
        print(f"  FAIL {r[0]} ({r[1]}): {r[5]}")
    return 1 if real else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["detect", "e2e", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default="simulations/out/stress_cli_score_detection")
    args = ap.parse_args()

    status = 0
    if args.mode in ("detect", "both"):
        print("=== inference only ===")
        run_detect(args.seed)
    if args.mode in ("e2e", "both"):
        print("=== end-to-end `analyze` ===")
        status = run_e2e(args.seed, Path(args.tmp_dir))
    sys.exit(status)


if __name__ == "__main__":
    main()
