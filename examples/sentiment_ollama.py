"""Demo: Real LLM evaluation using promptstats analyze router (Ollama).

Calls a locally running Ollama model (gpt-oss:20b) to classify the sentiment
of customer reviews using prompt template variants, scores outputs with
code-based evaluators, and then runs `promptstats.analyze` in both aggregate
and per-evaluator modes.

Requirements:
    - Install Ollama and start it: `ollama serve`
    - Pull model once: `ollama pull gpt-oss:20b`

Usage:
    python examples/sentiment_ollama.py
"""

import json
import math
import re
import time
from urllib import error, request

import numpy as np

import promptstats as pstats


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
    ("It arrived on time. Haven't tested it fully yet.",                       "NEUTRAL"),
    ("Incredibly sturdy build. Worth every penny.",                            "POSITIVE"),
    ("Instructions were confusing and a piece was missing.",                   "NEGATIVE"),
    ("Decent enough for the price. Not amazing.",                              "NEUTRAL"),
    ("Surpassed my expectations in every way!",                                "POSITIVE"),
    ("Completely useless. Don't waste your time.",                             "NEGATIVE"),
    ("It's okay. I've seen better but I've also seen worse.",                  "NEUTRAL"),

    # --- Challenging: sarcasm (surface words are positive, meaning is negative) ---
    ("Oh wonderful, it broke on the very first day. Just what I always wanted.",   "NEGATIVE"),
    ("Sure, I love spending 45 minutes on hold. Best. Support. Ever.",             "NEGATIVE"),
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
# Prompt templates — variants of the same task
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
# Ollama call
# ---------------------------------------------------------------------------

MODEL = "gemma3:4b" # gpt-oss:20b
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


def call_model(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
        },
    }
    req = request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Failed to call Ollama at http://127.0.0.1:11434. "
            "Is `ollama serve` running and is model `gpt-oss:20b` pulled?"
        ) from exc

    data = json.loads(body)
    return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> tuple[np.ndarray, list[list[str]]]:
    """Call the model for every (template, input) pair and score each output.

    Returns
    -------
    scores : np.ndarray
        Shape ``(N_templates, N_inputs, 1, N_evaluators)`` — the unit runs
        axis (axis 2) matches the promptstats ``(N, M, R, K)`` convention.
    outputs : list[list[str]]
        Raw model outputs indexed by [template_idx][input_idx].
    """
    scores = np.zeros((N_TEMPLATES, N_INPUTS, 1, len(EVALUATORS)))
    outputs = [[None] * N_INPUTS for _ in range(N_TEMPLATES)]
    total = N_TEMPLATES * N_INPUTS
    done = 0

    for t_idx, (t_name, template) in enumerate(TEMPLATES.items()):
        for i_idx, (text, ground_truth) in enumerate(INPUTS):
            prompt = template.format(text=text)
            output = call_model(prompt)
            outputs[t_idx][i_idx] = output

            for e_idx, (_, evaluator) in enumerate(EVALUATORS):
                scores[t_idx, i_idx, 0, e_idx] = evaluator(output, ground_truth)

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
    print(f"Model      : {MODEL}")
    print(f"Templates  : {N_TEMPLATES}  ({', '.join(TEMPLATE_LABELS)})")
    print(f"Inputs     : {N_INPUTS}")
    print(f"Evaluators : {len(EVALUATORS)}  ({', '.join(EVALUATOR_NAMES)})")
    print(f"Total calls: {N_TEMPLATES * N_INPUTS}\n")

    print("Running benchmark …")
    t0 = time.time()
    raw_scores, outputs = run_benchmark()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # raw_scores shape: (N_templates, N_inputs, 1, N_evaluators) — the unit
    # runs axis is already in place from run_benchmark.
    result_3d = pstats.BenchmarkResult(
        scores=raw_scores,
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
        evaluator_names=EVALUATOR_NAMES,
    )

    print(
        f"BenchmarkResult: {result_3d.n_templates} templates × "
        f"{result_3d.n_inputs} inputs × 1 run × {len(EVALUATOR_NAMES)} evaluators\n"
    )

    print("=== analyze(..., evaluator_mode='aggregate') ===")
    analysis_agg = pstats.analyze(
        result_3d,
        evaluator_mode="aggregate",
        reference="grand_mean",
        method="auto",
        n_bootstrap=5_000,
        correction="holm",
        failure_threshold=0.5,
        rng=np.random.default_rng(0),
    )
    pstats.print_analysis_summary(analysis_agg, top_pairwise=8)
    print()

    print("=== analyze(..., evaluator_mode='per_evaluator') ===")
    analysis_by_eval = pstats.analyze(
        result_3d,
        evaluator_mode="per_evaluator",
        reference="grand_mean",
        method="auto",
        n_bootstrap=3_000,
        correction="holm",
        failure_threshold=0.5,
        rng=np.random.default_rng(0),
    )
    pstats.print_analysis_summary(analysis_by_eval, top_pairwise=4)

    result_2d = pstats.BenchmarkResult(
        scores=result_3d.get_2d_scores(),
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
    )
    fig = pstats.plot_point_advantage(
        result_2d,
        reference="grand_mean",
        title=(
            "Prompt Template Comparison — Sentiment Classification\n"
            f"({MODEL}, composite score averaged over {len(EVALUATORS)} evaluators)"
        ),
        rng=np.random.default_rng(0),
    )
    fig.savefig("demo_analyze_advantage_ollama.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_analyze_advantage_ollama.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
