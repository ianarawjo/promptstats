#!/usr/bin/env python3
"""Terminal output for the paper's Fig. 1: language models compared on an LLM-judge
quality score, PPI-corrected with a small random subset of expert human ratings.
Fully reproducible from --seed.

Data: the ``human_eval`` split of prometheus-eval/BiGGen-Bench-Results
(Kim et al. 2024, CC-BY-SA-4.0), downloaded and cached by huggingface_hub on
first run. It holds 695 instances answered by all four of Llama-2-13b,
Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1 and gpt-3.5-turbo-0125,
each response rated 1-5 by a trained annotator against an instance-specific
rubric, alongside 1-5 scores from three LLM judges on the same rubric.

The human rating is the ground truth; only the --n-lab subset of it is passed
to evalstats, and the rest of the analysis sees judge scores alone.

What one run does, all from one seed:
  1. draws --n-items instances uniformly at random from the complete ones,
  2. draws --n-lab of those to carry their human ratings (the same instances
     for every model, so the labeled subset is paired too),
  3. writes the resulting spreadsheet (one row per instance x model) to
     --out-csv, row order shuffled with the same seed,
  4. runs judge_alignment() then compare(alignment=...) and prints the output.

Defaults (80 instances, 25 labeled, three models) were chosen by sweeping
budgets over eight seeds: they reproduce the same leaderboard shape in 7 of 8
draws with no spurious split between the top two.

Usage:
    python -m simulations.make_fig1_biggen_terminal
    python -m simulations.make_fig1_biggen_terminal --raw
    python -m simulations.make_fig1_biggen_terminal --seed 3 --n-items 100 --n-lab 30
    python -m simulations.make_fig1_biggen_terminal --save-output simulations/out/fig1_terminal.txt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalstats as es  # noqa: E402
from evalstats.alignment import judge_alignment  # noqa: E402

HF_REPO = "prometheus-eval/BiGGen-Bench-Results"
HF_FILE = "data/human_eval-00000-of-00001.parquet"

# Shortened for the leaderboard column; the full HF model_name is kept in the
# written spreadsheet so the mapping is always recoverable.
MODEL_NAMES = {
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Llama-2-13b-hf": "Llama-2-13b",
}
DEFAULT_MODELS = ["Mixtral-8x7B", "GPT-3.5-turbo", "Llama-2-13b"]
# gpt4_04_turbo is the best-aligned of the three shipped judges (r=.63,
# rho^2=.40 over all 2,764 rated responses); gpt4 is .36 and claude .31.
JUDGE_COLUMNS = {
    "gpt4-turbo": "gpt4_04_turbo_score",
    "gpt4": "gpt4_score",
    "claude": "claude_score",
}
DEFAULT_OUT_CSV = "simulations/out/fig1_biggen_eval.csv"
METRIC = "quality"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def load_human_eval() -> pd.DataFrame:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def build_frame(*, models: list[str], judge_col: str, n_items: int, n_lab: int,
                seed: int, capability: str | None) -> pd.DataFrame:
    raw = load_human_eval()
    raw = raw[raw["human_score"] > 0].copy()  # a handful of rows carry -1
    raw["model"] = raw["model_name"].map(MODEL_NAMES)
    if capability:
        raw = raw[raw["capability"] == capability]
        if raw.empty:
            raise SystemExit(f"No rows for capability {capability!r}.")
    unknown = set(models) - set(MODEL_NAMES.values())
    if unknown:
        raise SystemExit(f"Unknown model(s) {sorted(unknown)}; known: {sorted(MODEL_NAMES.values())}")

    # Only instances every selected model answered, so the design is complete.
    counts = raw[raw["model"].isin(models)].groupby("id")["model"].nunique()
    docs = sorted(counts.index[counts == len(models)])
    if len(docs) < n_items:
        raise SystemExit(f"Only {len(docs)} instances answered by all {len(models)} models; asked for {n_items}.")

    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(docs, size=n_items, replace=False))
    labeled = set(rng.choice(pick, size=n_lab, replace=False))

    df = raw[raw["model"].isin(models) & raw["id"].isin(pick)][
        ["id", "model", "model_name", "capability", "task", judge_col, "human_score"]
    ].copy()
    df = df.rename(columns={"id": "item", judge_col: METRIC})
    df[f"human_{METRIC}"] = np.where(df["item"].isin(labeled), df["human_score"], np.nan)
    df = df.drop(columns="human_score")
    # Row order carries no information; shuffle so the sheet is not sorted by model.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=80, help="Instances drawn (default 80).")
    ap.add_argument("--n-lab", type=int, default=25, help="Instances whose human ratings are revealed (default 25).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--judge", choices=sorted(JUDGE_COLUMNS), default="gpt4-turbo")
    ap.add_argument("--capability", default=None,
                    help="Restrict to one BiGGen capability (e.g. instruction_following). Default: all.")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--raw", action="store_true",
                    help="Also print the uncorrected run (judge scores at face value) first.")
    ap.add_argument("--no-alignment-report", action="store_true")
    ap.add_argument("--omnibus", action="store_true")
    ap.add_argument("--save-output", default=None)
    ap.add_argument("--data-only", action="store_true",
                    help="Write the spreadsheet and stop, for the CLI to analyze (see fig1.sh).")
    ap.add_argument("--human-only-out", default=None,
                    help="Also write the human-labeled instances alone, their expert rating as the "
                         "metric and no judge column: what the same labeling budget buys unaided.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    if args.save_output:
        Path(args.save_output).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.__stdout__, open(args.save_output, "w", encoding="utf-8"))

    judge_col = JUDGE_COLUMNS[args.judge]
    df = build_frame(models=args.models, judge_col=judge_col, n_items=args.n_items,
                     n_lab=args.n_lab, seed=args.seed, capability=args.capability)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    human_col = f"human_{METRIC}"
    n_lab_rows = int(df[human_col].notna().sum())
    print(f"# BiGGen-Bench human_eval, judge={args.judge} ({judge_col}), models={args.models}, "
          f"{args.n_items} instances, {args.n_lab} human-rated, seed={args.seed}"
          + (f", capability={args.capability}" if args.capability else ""))
    print(f"# spreadsheet: {out_csv} ({len(df)} rows; {n_lab_rows} with human ratings)")

    if args.human_only_out:
        human = (df[df[human_col].notna()][["item", "model", human_col]]
                 .rename(columns={human_col: "human_rating"}))
        Path(args.human_only_out).parent.mkdir(parents=True, exist_ok=True)
        human.to_csv(args.human_only_out, index=False)
        print(f"# human-only:  {args.human_only_out} ({len(human)} rows, "
              f"{human['item'].nunique()} instances)")

    print()
    if args.data_only:
        return

    analysis_cols = ["item", "model", METRIC, human_col]
    common = dict(factors="model", metric=METRIC, score_range=(1, 5),
                  rng=args.seed + 1, n_bootstrap=args.n_bootstrap, p_values=True)
    if args.omnibus:
        common["omnibus"] = True

    if args.raw:
        print("=" * 72)
        print("RAW: judge scores taken at face value (no human ratings passed)")
        print("=" * 72)
        es.compare(es.load_from(df[["item", "model", METRIC]]), **common).summary()
        print()
        print("=" * 72)
        print("CORRECTED: same spreadsheet, human ratings passed via alignment=")
        print("=" * 72)

    ev = es.load_from(df[analysis_cols])
    ar = judge_alignment(ev, llm_metric=METRIC, human_groundtruth=human_col, selection="random")
    if not args.no_alignment_report:
        ar.summary()
        print()
    es.compare(ev, alignment={METRIC: ar}, **common).summary()


if __name__ == "__main__":
    main()
