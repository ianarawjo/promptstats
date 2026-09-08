#!/usr/bin/env python3
"""Terminal output for the paper's Fig. 1: three SummEval summarizers compared
on an LLM-judge coherence score, PPI-corrected with a small random subset of
expert labels. Fully reproducible from --seed.

Data: simulations/out/summeval_items.csv and summeval_judge_scores.csv, from
collect_summeval_judge_scores.py (expert labels are the 3-rater mean from
SummEval, Fabbri et al. 2021; judge scores are one OpenRouter judge).

What one run does, all from one seed:
  1. draws --n-items articles uniformly at random from the 100,
  2. draws --n-lab of those articles to carry their expert labels (the same
     articles for every system, so the labeled subset is paired too),
  3. writes the resulting spreadsheet (one row per article x system) to
     --out-csv, row order shuffled with the same seed,
  4. runs judge_alignment() then compare(alignment=...) on it and prints the
     evalstats output.

Usage:
    python -m simulations.make_fig1_summeval_terminal
    python -m simulations.make_fig1_summeval_terminal --seed 3 --n-lab 30
    python -m simulations.make_fig1_summeval_terminal --raw   # uncorrected run first
    python -m simulations.make_fig1_summeval_terminal --save-output simulations/out/fig1_terminal.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalstats as es  # noqa: E402
from evalstats.alignment import judge_alignment  # noqa: E402

DEFAULT_ITEMS = "simulations/out/summeval_items.csv"
DEFAULT_SCORES = "simulations/out/summeval_judge_scores.csv"
DEFAULT_OUT_CSV = "simulations/out/fig1_summeval_eval.csv"
DEFAULT_SYSTEMS = ["BART", "T5", "GPT-2"]
DEFAULT_JUDGE = "anthropic/claude-haiku-4.5"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def build_frame(*, items_path: str, scores_path: str, systems: list[str], dimension: str,
                judge: str, n_items: int, n_lab: int, seed: int) -> pd.DataFrame:
    items = {r["item_id"]: r for r in csv.DictReader(open(items_path, encoding="utf-8"))}
    scores = [r for r in csv.DictReader(open(scores_path, encoding="utf-8"))
              if r["dimension"] == dimension and r["judge_model"] == judge and r["run_idx"] == "0"]
    if not scores:
        raise SystemExit(f"No {dimension} scores from {judge!r} in {scores_path}.")
    rows = []
    for s in scores:
        it = items[s["item_id"]]
        if it["system"] not in systems:
            continue
        rows.append({
            "item": it["doc_id"], "model": it["system"],
            dimension: float(s["judge_score"]),
            "_expert": float(it[f"expert_{dimension}"]),
        })
    df = pd.DataFrame(rows)
    missing = set(systems) - set(df["model"])
    if missing:
        raise SystemExit(f"No scores for systems {sorted(missing)}; known: {sorted(set(df['model']))}")
    counts = df.groupby("item")["model"].nunique()
    docs = sorted(counts.index[counts == len(systems)])
    if len(docs) < n_items:
        raise SystemExit(f"Only {len(docs)} articles have all {len(systems)} systems scored; asked for {n_items}.")

    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(docs, size=n_items, replace=False))
    labeled = set(rng.choice(pick, size=n_lab, replace=False))
    df = df[df["item"].isin(pick)].copy()
    df[f"human_{dimension}"] = np.where(df["item"].isin(labeled), df["_expert"], np.nan)
    df = df.drop(columns="_expert")
    # Row order carries no information; shuffle so the spreadsheet is not
    # sorted by system.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=60, help="Articles drawn (default 60).")
    ap.add_argument("--n-lab", type=int, default=15, help="Articles whose expert labels are revealed (default 15).")
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--dimension", default="coherence")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--items", default=DEFAULT_ITEMS)
    ap.add_argument("--scores", default=DEFAULT_SCORES)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Where the analyzed spreadsheet is written.")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--raw", action="store_true",
                    help="Also print the uncorrected run (judge scores taken at face value) first.")
    ap.add_argument("--no-alignment-report", action="store_true", help="Skip printing judge_alignment().summary().")
    ap.add_argument("--omnibus", action="store_true")
    ap.add_argument("--save-output", default=None, help="Also write everything printed to this file.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    if args.save_output:
        Path(args.save_output).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.__stdout__, open(args.save_output, "w", encoding="utf-8"))

    df = build_frame(items_path=args.items, scores_path=args.scores, systems=args.systems,
                     dimension=args.dimension, judge=args.judge, n_items=args.n_items,
                     n_lab=args.n_lab, seed=args.seed)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    human_col = f"human_{args.dimension}"
    print(f"# SummEval {args.dimension}, judge={args.judge}, systems={args.systems}, "
          f"{args.n_items} articles, {args.n_lab} expert-labeled, seed={args.seed}")
    print(f"# spreadsheet: {out_csv} ({len(df)} rows; {int(df[human_col].notna().sum())} with human labels)")
    print()

    common = dict(factors="model", metric=args.dimension, score_range=(1, 5),
                  rng=args.seed + 1, n_bootstrap=args.n_bootstrap, p_values=True)
    if args.omnibus:
        common["omnibus"] = True

    if args.raw:
        print("=" * 72)
        print("RAW: judge scores taken at face value (no human labels passed)")
        print("=" * 72)
        raw_ev = es.load_from(df.drop(columns=human_col))
        es.compare(raw_ev, **common).summary()
        print()
        print("=" * 72)
        print("CORRECTED: same spreadsheet, human labels passed via alignment=")
        print("=" * 72)

    ev = es.load_from(df)
    ar = judge_alignment(ev, llm_metric=args.dimension, human_groundtruth=human_col, selection="random")
    if not args.no_alignment_report:
        ar.summary()
        print()
    result = es.compare(ev, alignment={args.dimension: ar}, **common)
    result.summary()


if __name__ == "__main__":
    main()
