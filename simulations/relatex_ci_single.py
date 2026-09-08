"""Regenerate ci_single's LaTeX overall-summary table from a finished results CSV.

The sibling of `relatex_ci_paired.py`, for the mean-point-estimate case. The
2026-08-19 ci_single sweep was rendered before commit a3a00d0 added the MinCov
and Pen columns to the renderer, so its published table lacks them while
ci_paired's (re-run afterwards) has them. The sweep does not need repeating:
both quantities are recoverable from the saved CSV.

  MinCov  the renderer takes it as min over per-scenario coverage, which the
          CSV carries per cell.
  Pen     the CSV omits total_pen_under / total_pen_over, but the interval
          score is Width + Penalty by construction, so Penalty = Score - Width.
          The renderer only ever uses the two penalty fields as a sum, so the
          whole penalty is carried in total_pen_under and total_pen_over is
          left at zero. Verified against ci_paired, where the CSV stores both
          the penalty fields and score/width: the identity holds to 1e-8, the
          CSV's stored precision.

    python simulations/relatex_ci_single.py RESULTS.csv > table.tex
    python simulations/relatex_ci_single.py RESULTS.csv --eval-types binary
"""
import argparse

import pandas as pd

from simulations.harness.cases.ci_single import SimResult, latex_overall_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--eval-types", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["method"] = df["method"].astype(str)
    if args.methods:
        df = df[df.method.isin(args.methods)]
    if args.eval_types:
        df = df[df.eval_type.isin(args.eval_types)]
    if df.empty:
        raise SystemExit("no rows after filtering")

    has_pen = "mean_pen_under" in df.columns and "mean_pen_over" in df.columns

    results = []
    for r in df.itertuples():
        if has_pen:
            pen_u = float(r.mean_pen_under) * int(r.n_reps)
            pen_o = float(r.mean_pen_over) * int(r.n_reps)
        else:
            pen_u = (float(r.total_score) - float(r.total_width))
            pen_o = 0.0
        results.append(SimResult(
            source=str(r.source), label=str(r.label), eval_type=str(r.eval_type),
            n=int(r.n), method=str(r.method), n_reps=int(r.n_reps),
            covered=int(r.covered), total_width=float(r.total_width),
            total_score=float(r.total_score),
            total_pen_under=pen_u, total_pen_over=pen_o,
            total_time=float(r.total_time), total_time_sq=float(r.total_time_sq),
        ))

    print(latex_overall_summary(results, args.alpha, int(df.n_reps.iloc[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
