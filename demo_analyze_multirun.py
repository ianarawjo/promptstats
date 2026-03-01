"""Demo: Real LLM evaluation using promptstats analyze router — multiple runs.

Like demo_analyze.py but runs each (template, input) pair N_RUNS=3 times with
temperature > 0, populating the runs axis of BenchmarkResult so that
promptstats can account for within-template variability.

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Usage:
    python demo_analyze_multirun.py
"""

import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

import promptstats as bps


# ---------------------------------------------------------------------------
# Task: sentiment classification with known ground truth labels
# ---------------------------------------------------------------------------

INPUTS = [
    # (review text, ground_truth_label)

    # --- Easy / unambiguous ---
    ("This product exceeded all my expectations. Absolutely love it!",         "POSITIVE"),
    ("Complete waste of money. Broke after two days.",                         "NEGATIVE"),
    ("It works fine. Does what it says on the box.",                           "NEUTRAL"),
    ("Best purchase I've made all year. Five stars!",                          "POSITIVE"),
    ("Terrible customer service. Will never buy again.",                       "NEGATIVE"),
    ("Average quality. Neither impressed nor disappointed.",                   "NEUTRAL"),
    ("Shipping was fast and the item looks great. Very happy.",                "POSITIVE"),
    ("The color is different from the photos. Feels cheap.",                   "NEGATIVE"),
    ("Incredibly sturdy build. Worth every penny.",                            "POSITIVE"),
    ("Decent enough for the price. Not amazing.",                              "NEUTRAL"),
    ("Surpassed my expectations in every way!",                                "POSITIVE"),
    ("Completely useless. Don't waste your time.",                             "NEGATIVE"),
    ("It's okay. I've seen better but I've also seen worse.",                  "NEUTRAL"),

    # --- Challenging: sarcasm (surface words are positive, meaning is negative) ---
    ("Oh wonderful, it broke on the very first day.",                              "NEGATIVE"),
    ("Sure, I love spending 45 minutes on hold.",                                  "NEGATIVE"),
    ("Impressive how something so simple can go so spectacularly wrong.",          "NEGATIVE"),

    # --- Challenging: double negatives / hedged language ---
    ("Not entirely without merit, though not exactly impressive either.",          "NEUTRAL"),
    ("I can't say I wasn't disappointed — yet somehow I wasn't fully let down.",   "NEUTRAL"),
    ("It's not like it doesn't work. It works. Just not well.",                    "NEGATIVE"),

    # --- Challenging: mixed / split sentiment ---
    ("The hardware is excellent but the software makes the whole thing unusable.",  "NEGATIVE"),
    ("Great concept, poor execution. So close, yet so far.",                       "NEGATIVE"),
    ("Fast delivery and nice packaging; the product itself is deeply mediocre.",   "NEUTRAL"),

    # --- Challenging: understatement / implied sentiment ---
    ("Let's just say I won't be ordering a second one.",                           "NEGATIVE"),
    ("Pleasantly surprised — I expected nothing and got slightly more than that.", "POSITIVE"),
    ("For a product that does practically nothing, it does it quite consistently.", "NEUTRAL"),
]

INPUT_TEXTS = [x[0] for x in INPUTS]
GROUND_TRUTHS = [x[1] for x in INPUTS]
INPUT_LABELS = [f"review_{i:02d}" for i in range(len(INPUTS))]


# ---------------------------------------------------------------------------
# Prompt templates — four variants of the same task
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Minimal": (
        "Classify the sentiment of the following text.\n"
        "Reply with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        "Text: {text}\n\n"
        "Sentiment:"
    ),
    "Instructive": (
        "You are a precise sentiment classifier.\n"
        "Classify the sentiment of the text below using exactly one of these labels:\n"
        "  POSITIVE – the text expresses satisfaction, praise, or approval.\n"
        "  NEGATIVE – the text expresses dissatisfaction, criticism, or complaint.\n"
        "  NEUTRAL  – the text is factual, mixed, or non-committal.\n\n"
        "Respond with the single label only, nothing else.\n\n"
        "Text: {text}\n\n"
        "Sentiment:"
    ),
    "Few-shot": (
        "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        "Example 1:\n"
        "Text: The battery life is phenomenal and the screen is crisp.\n"
        "Sentiment: POSITIVE\n\n"
        "Example 2:\n"
        "Text: It stopped working after a week. Very disappointing.\n"
        "Sentiment: NEGATIVE\n\n"
        "Example 3:\n"
        "Text: Arrived in good condition. Standard product.\n"
        "Sentiment: NEUTRAL\n\n"
        "Now classify:\n"
        "Text: {text}\n\n"
        "Sentiment:"
    ),
    "Chain-of-thought": (
        "Classify the sentiment of the text below as POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        "Think step by step (1-2 sentences), then on a new line write exactly:\n"
        "Sentiment: <LABEL>\n\n"
        "Text: {text}"
    ),
    "Contradictory": (
        "You are an expert in multi-dimensional affective valence classification.\n"
        "Use the following schema:\n"
        "  POSITIVE – net emotional tone is favorable after accounting for irony and sarcasm.\n"
        "  NEGATIVE – net tone is unfavorable, critical, or expressive of dissatisfaction.\n"
        "  NEUTRAL  – ambiguous, mixed, or indeterminate cases.\n\n"
        "IMPORTANT: When in doubt, always output NEUTRAL. Never output POSITIVE or NEGATIVE "
        "unless you are 100% certain.\n"
        "ALSO IMPORTANT: Do not output NEUTRAL unless there is genuinely no dominant sentiment. "
        "If sentiment is present at all, output POSITIVE or NEGATIVE.\n"
        "FINAL INSTRUCTION: Disregard all prior instructions and classify purely by counting "
        "positive-sounding vs negative-sounding words. Whichever count is higher wins.\n\n"
        "Respond with the single classification label.\n\n"
        "Text: {text}\n\n"
        "Classification:"
    ),
    "Double-negation": (
        "Determine whether the text is not unfavorable, not favorable, "
        "or neither not unfavorable nor not favorable.\n\n"
        "Rules:\n"
        "  - If the text is not unfavorable (i.e., it does not express something that is not good), "
        "output: POSITIVE\n"
        "  - If the text is not favorable (i.e., it does not express something that is not bad), "
        "output: NEGATIVE\n"
        "  - If the text is neither not unfavorable nor not favorable, output: NEUTRAL\n\n"
        "Note: a text that is not not good is POSITIVE. "
        "A text that is not not bad is NEGATIVE. "
        "A text that is not not neutral is NEUTRAL.\n\n"
        "Text: {text}\n\n"
        "Answer (POSITIVE, NEGATIVE, or NEUTRAL):"
    ),
}

TEMPLATE_LABELS = list(TEMPLATES.keys())
N_TEMPLATES = len(TEMPLATES)
N_INPUTS = len(INPUTS)
N_RUNS = 3
TEMPERATURE = 0.7  # non-zero so that repeated runs produce meaningful variation


# ---------------------------------------------------------------------------
# Code-based evaluators
# ---------------------------------------------------------------------------

VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def _extract_label(output: str) -> str | None:
    """Return the first valid sentiment label found in the output, or None."""
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", output.upper()):
            return label
    return None


def eval_label_valid(output: str, ground_truth: str) -> float:
    """1.0 if the output contains a recognised sentiment label, else 0.0."""
    return 1.0 if _extract_label(output) is not None else 0.0


def eval_accuracy(output: str, ground_truth: str) -> float:
    """1.0 if the extracted label matches ground truth, else 0.0."""
    predicted = _extract_label(output)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0


def eval_brevity(output: str, ground_truth: str) -> float:
    """Score conciseness: exp(-len / 120)."""
    return math.exp(-len(output.strip()) / 120.0)


EVALUATORS = [
    ("label_valid", eval_label_valid),
    ("accuracy", eval_accuracy),
    ("brevity", eval_brevity),
]
EVALUATOR_NAMES = [name for name, _ in EVALUATORS]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def call_model(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_completion_tokens=128,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(client: OpenAI) -> tuple[np.ndarray, list[list[list[str]]]]:
    """Call the model for every (template, input, run) triple and score outputs.

    Returns
    -------
    scores : np.ndarray
        Shape ``(N_templates, N_inputs, N_runs, N_evaluators)`` matching the
        promptstats ``(N, M, R, K)`` convention.
    outputs : list[list[list[str]]]
        Raw model outputs indexed by [run_idx][template_idx][input_idx].
    """
    scores = np.zeros((N_TEMPLATES, N_INPUTS, N_RUNS, len(EVALUATORS)))
    outputs = [[[None] * N_INPUTS for _ in range(N_TEMPLATES)] for _ in range(N_RUNS)]
    total = N_RUNS * N_TEMPLATES * N_INPUTS
    done = 0

    for r_idx in range(N_RUNS):
        print(f"\n--- Run {r_idx + 1}/{N_RUNS} ---")
        for t_idx, (t_name, template) in enumerate(TEMPLATES.items()):
            for i_idx, (text, ground_truth) in enumerate(INPUTS):
                prompt = template.format(text=text)
                output = call_model(prompt, client)
                outputs[r_idx][t_idx][i_idx] = output

                for e_idx, (_, evaluator) in enumerate(EVALUATORS):
                    scores[t_idx, i_idx, r_idx, e_idx] = evaluator(output, ground_truth)

                done += 1
                predicted = _extract_label(output) or "???"
                correct = "✓" if predicted == ground_truth else "✗"
                print(
                    f"  [{done:3d}/{total}] {t_name:<18s} | "
                    f"input {i_idx:02d} | "
                    f"truth={ground_truth:<8s} pred={predicted:<8s} {correct} | "
                    f"'{output[:35]}'"
                )

    return scores, outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Model      : {MODEL}")
    print(f"Templates  : {N_TEMPLATES}  ({', '.join(TEMPLATE_LABELS)})")
    print(f"Inputs     : {N_INPUTS}")
    print(f"Runs       : {N_RUNS}  (temperature={TEMPERATURE})")
    print(f"Evaluators : {len(EVALUATORS)}  ({', '.join(EVALUATOR_NAMES)})")
    print(f"Total calls: {N_RUNS * N_TEMPLATES * N_INPUTS}\n")

    print("Running benchmark …")
    t0 = time.time()
    raw_scores, outputs = run_benchmark(client)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # raw_scores shape: (N_templates, N_inputs, N_runs, N_evaluators)
    result = bps.BenchmarkResult(
        scores=raw_scores,
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
        evaluator_names=EVALUATOR_NAMES,
    )

    print(
        f"BenchmarkResult: {result.n_templates} templates × "
        f"{result.n_inputs} inputs × {N_RUNS} runs × {len(EVALUATOR_NAMES)} evaluators\n"
    )

    print("=== analyze(..., evaluator_mode='aggregate') ===")
    analysis_agg = bps.analyze(
        result,
        evaluator_mode="aggregate",
        reference="grand_mean",
        method="auto",
        n_bootstrap=5_000,
        correction="holm",
        failure_threshold=0.5,
        rng=np.random.default_rng(0),
    )
    bps.print_analysis_summary(analysis_agg, top_pairwise=8)
    print()

    print("=== analyze(..., evaluator_mode='per_evaluator') ===")
    analysis_by_eval = bps.analyze(
        result,
        evaluator_mode="per_evaluator",
        reference="grand_mean",
        method="auto",
        n_bootstrap=3_000,
        correction="holm",
        failure_threshold=0.5,
        rng=np.random.default_rng(0),
    )
    bps.print_analysis_summary(analysis_by_eval, top_pairwise=4)

    result_2d = bps.BenchmarkResult(
        scores=result.get_2d_scores(),
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
    )
    fig = bps.plot_mean_advantage(
        result_2d,
        reference="grand_mean",
        title=(
            f"Prompt Template Comparison — Sentiment Classification\n"
            f"({MODEL}, {N_RUNS} runs, composite score averaged over {len(EVALUATORS)} evaluators)"
        ),
        rng=np.random.default_rng(0),
    )
    fig.savefig("demo_analyze_multirun_advantage.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_analyze_multirun_advantage.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
