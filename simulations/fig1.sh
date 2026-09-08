#!/usr/bin/env bash
#
# Prints the terminal output behind the paper's Figure 1, ready to screenshot.
#
#   ./simulations/fig1.sh
#
# Three BiGGen-Bench models compared on an LLM-judge quality score, corrected
# with expert human ratings for a random quarter of the instances:
#
#   80 instances x 3 models (Mixtral-8x7B, GPT-3.5-turbo, Llama-2-13b) = 240 rows
#   25 of those instances carry their expert 1-5 rating (75 of the 240 rows)
#   judge  = GPT-4-0409-turbo, scoring each response against the instance rubric
#   human  = one trained annotator per response, from BiGGen-Bench's human_eval
#            split (prometheus-eval/BiGGen-Bench-Results, CC-BY-SA-4.0)
#
# The spreadsheet is built once into simulations/out/ (untracked) on the first
# run, downloading BiGGen-Bench through huggingface_hub; after that this only
# runs evalstats, so it is fast and the numbers never move. Pass --rebuild to
# redraw it, or edit SEED/N_ITEMS/N_LAB below for a different draw.
#
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=0
N_ITEMS=80
N_LAB=25
CSV=simulations/out/fig1_biggen_eval.csv
PY=.venv/bin/python
CLI=.venv/bin/evalstats

[ -x "$PY" ] || { echo "No $PY -- run this from the repo with its venv installed." >&2; exit 1; }

if [ "${1:-}" = "--rebuild" ]; then rm -f "$CSV"; fi

if [ ! -f "$CSV" ]; then
    echo "Building $CSV (first run downloads BiGGen-Bench) ..." >&2
    "$PY" -m simulations.make_fig1_biggen_terminal \
        --seed "$SEED" --n-items "$N_ITEMS" --n-lab "$N_LAB" --data-only >&2
    echo >&2
fi

# --score-range 1 5 states the rubric's scale rather than letting evalstats
# infer it, which also drops the inference notice from the screenshot.
# --seed 1 is compare()'s resampling seed (the script's own SEED+1), so the
# bootstrap CIs come out identical to make_fig1_biggen_terminal.py's.
# PYTHONWARNINGS keeps the "fewer than ~30 labeled items" UserWarning off the
# screen; 25 labels is the point of the demonstration, and the alignment
# report already shows the resulting CI width.
# --correction shaffer keeps the reported p-values Wilcoxon signed-rank ones,
# corrected in place. Left on auto they would resolve to Romano-Wolf, which
# REPLACES the Wilcoxon p with its own joint statistic -- fine statistically,
# and the appendix recommends it at n>=30, but it puts a bootstrap p in a
# figure whose point is the rank test.
exec env PYTHONWARNINGS=ignore "$CLI" analyze "$CSV" \
    --metric quality \
    --human-groundtruth human_quality \
    --factor model \
    --score-range 1 5 \
    --label-selection random \
    --correction shaffer \
    --seed $((SEED + 1)) \
    --p-values \
    --omnibus
