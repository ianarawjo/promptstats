"""Demo: compare two small models using promptstats multi-model analysis.

Copies the workflow of demo_analyze.py, but runs two models and routes through
the MultiModelBenchmark path in `promptstats.core.router.analyze`.

Like demo_analyze_multirun.py, this version runs each (model, template, input)
triple multiple times (N_RUNS=3) with temperature > 0 so the runs axis is
populated and within-template variability is captured.

Models (defaults):
    - OpenAI: gpt-4.1-nano
    - OpenRouter: google/gemini-2.0-flash-lite-001

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    export OPENROUTER_API_KEY="sk-or-..."

Optional environment overrides:
    OPENAI_MODEL
    OPENROUTER_MODEL
    OPENROUTER_HTTP_REFERER
    OPENROUTER_APP_NAME

Usage:
    python demo_compare_models.py
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
    ("This product exceeded all my expectations. Absolutely love it!", "POSITIVE"),
    ("Complete waste of money. Broke after two days.", "NEGATIVE"),
    ("It works fine. Does what it says on the box.", "NEUTRAL"),
    ("Best purchase I've made all year. Five stars!", "POSITIVE"),
    ("Terrible customer service. Will never buy again.", "NEGATIVE"),
    ("Average quality. Neither impressed nor disappointed.", "NEUTRAL"),
    ("Shipping was fast and the item looks great. Very happy.", "POSITIVE"),
    ("The color is different from the photos. Feels cheap.", "NEGATIVE"),
    ("It arrived on time. Haven't tested it fully yet.", "NEUTRAL"),
    ("Incredibly sturdy build. Worth every penny.", "POSITIVE"),
    ("Instructions were confusing and a piece was missing.", "NEGATIVE"),
    ("Decent enough for the price. Not amazing.", "NEUTRAL"),
    ("Surpassed my expectations in every way!", "POSITIVE"),
    ("Completely useless. Don't waste your time.", "NEGATIVE"),
    ("It's okay. I've seen better but I've also seen worse.", "NEUTRAL"),
    ("Oh wonderful, it broke on the very first day. Just what I always wanted.", "NEGATIVE"),
    ("Sure, I love spending 45 minutes on hold. Best. Support. Ever.", "NEGATIVE"),
    ("Impressive how something so simple can go so spectacularly wrong.", "NEGATIVE"),
    ("Not entirely without merit, though not exactly impressive either.", "NEUTRAL"),
    ("I can't say I wasn't disappointed — yet somehow I wasn't fully let down.", "NEUTRAL"),
    ("It's not like it doesn't work. It works. Just not well.", "NEGATIVE"),
    ("The hardware is excellent but the software makes the whole thing unusable.", "NEGATIVE"),
    ("Great concept, poor execution. So close, yet so far.", "NEGATIVE"),
    ("Fast delivery and nice packaging; the product itself is deeply mediocre.", "NEUTRAL"),
    ("Let's just say I won't be ordering a second one.", "NEGATIVE"),
    ("Pleasantly surprised — I expected nothing and got slightly more than that.", "POSITIVE"),
    ("For a product that does practically nothing, it does it quite consistently.", "NEUTRAL"),
]

INPUT_LABELS = [f"review_{i:02d}" for i in range(len(INPUTS))]


# ---------------------------------------------------------------------------
# Prompt templates
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
}

TEMPLATE_LABELS = list(TEMPLATES.keys())
N_TEMPLATES = len(TEMPLATE_LABELS)
N_INPUTS = len(INPUTS)
N_RUNS = 3
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def _extract_label(output: str) -> str | None:
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", output.upper()):
            return label
    return None


def eval_label_valid(output: str, ground_truth: str) -> float:
    return 1.0 if _extract_label(output) is not None else 0.0


def eval_accuracy(output: str, ground_truth: str) -> float:
    predicted = _extract_label(output)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0


def eval_brevity(output: str, ground_truth: str) -> float:
    return math.exp(-len(output.strip()) / 120.0)


EVALUATORS = [
    ("label_valid", eval_label_valid),
    ("accuracy", eval_accuracy),
    ("brevity", eval_brevity),
]
EVALUATOR_NAMES = [name for name, _ in EVALUATORS]


# ---------------------------------------------------------------------------
# Models/providers
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "mistralai/ministral-8b-2512")

MODEL_SPECS = [
    {
        "label": f"OpenAI: {OPENAI_MODEL}",
        "provider": "openai",
        "model": OPENAI_MODEL,
    },
    {
        "label": f"OpenRouter: {OPENROUTER_MODEL}",
        "provider": "openrouter",
        "model": OPENROUTER_MODEL,
    },
]


def _openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _make_clients() -> dict[str, OpenAI]:
    openai_key = os.environ.get("OPENAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)

    clients: dict[str, OpenAI] = {}
    clients["openai"] = OpenAI(api_key=openai_key)

    headers = _openrouter_headers()
    if headers:
        clients["openrouter"] = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
        )
    else:
        clients["openrouter"] = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )

    return clients


def call_model(prompt: str, *, model: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_completion_tokens=128,
    )
    return response.choices[0].message.content.strip()


def run_multi_model_benchmark(clients: dict[str, OpenAI]) -> tuple[np.ndarray, dict[str, list[list[list[str]]]]]:
    """Run all (model, template, input, run) calls and score each output.

    Returns
    -------
    scores : np.ndarray
        Shape ``(N_models, N_templates, N_inputs, N_runs, N_evaluators)``
        matching the promptstats ``(P, N, M, R, K)``
        convention.
    outputs_by_model : dict[str, list[list[list[str]]]]
        Raw outputs keyed by model label, indexed [run_idx][template_idx][input_idx].
    """
    n_models = len(MODEL_SPECS)
    scores = np.zeros((n_models, N_TEMPLATES, N_INPUTS, N_RUNS, len(EVALUATORS)))
    outputs_by_model: dict[str, list[list[list[str]]]] = {
        spec["label"]: [[[""] * N_INPUTS for _ in range(N_TEMPLATES)] for _ in range(N_RUNS)]
        for spec in MODEL_SPECS
    }

    total = n_models * N_TEMPLATES * N_INPUTS * N_RUNS
    done = 0

    for m_idx, spec in enumerate(MODEL_SPECS):
        model_label = spec["label"]
        provider = spec["provider"]
        model_name = spec["model"]
        client = clients[provider]

        for r_idx in range(N_RUNS):
            for t_idx, (t_name, template) in enumerate(TEMPLATES.items()):
                for i_idx, (text, ground_truth) in enumerate(INPUTS):
                    prompt = template.format(text=text)
                    output = call_model(prompt, model=model_name, client=client)
                    outputs_by_model[model_label][r_idx][t_idx][i_idx] = output

                    for e_idx, (_, evaluator) in enumerate(EVALUATORS):
                        scores[m_idx, t_idx, i_idx, r_idx, e_idx] = evaluator(output, ground_truth)

                    done += 1
                    predicted = _extract_label(output) or "???"
                    correct = "✓" if predicted == ground_truth else "✗"
                    print(
                        f"  [{done:3d}/{total}] {model_label[:28]:<28s} | "
                        f"run {r_idx + 1}/{N_RUNS} | {t_name:<16s} | input {i_idx:02d} | "
                        f"truth={ground_truth:<8s} pred={predicted:<8s} {correct} | "
                        f"'{output[:35]}'"
                    )

    return scores, outputs_by_model


def demo_compare_models() -> None:
    clients = _make_clients()

    model_labels = [spec["label"] for spec in MODEL_SPECS]
    print(f"Models     : {', '.join(model_labels)}")
    print(f"Templates  : {N_TEMPLATES}  ({', '.join(TEMPLATE_LABELS)})")
    print(f"Inputs     : {N_INPUTS}")
    print(f"Runs       : {N_RUNS}  (temperature={TEMPERATURE})")
    print(f"Evaluators : {len(EVALUATORS)}  ({', '.join(EVALUATOR_NAMES)})")
    print(f"Total calls: {len(MODEL_SPECS) * N_TEMPLATES * N_INPUTS * N_RUNS}\n")

    print("Running multi-model benchmark …")
    t0 = time.time()
    raw_scores, _outputs = run_multi_model_benchmark(clients)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # raw_scores shape: (N_models, N_templates, N_inputs, N_runs, N_evaluators)
    multi_result = bps.MultiModelBenchmark(
        scores=raw_scores,
        model_labels=model_labels,
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
        evaluator_names=EVALUATOR_NAMES,
    )

    print(
        f"MultiModelBenchmark: {multi_result.n_models} models × "
        f"{multi_result.n_templates} templates × {multi_result.n_inputs} inputs × "
        f"{N_RUNS} runs × {len(EVALUATOR_NAMES)} evaluators\n"
    )

    print("=== analyze(..., evaluator_mode='aggregate') ===")
    analysis = bps.analyze(
        multi_result,
        evaluator_mode="aggregate",
        reference="grand_mean",
        method="auto",
        n_bootstrap=5_000,
        correction="holm",
        failure_threshold=0.5,
        rng=np.random.default_rng(0),
    )
    bps.print_analysis_summary(analysis, top_pairwise=8)

    print("\nDone!")


def main() -> None:
    demo_compare_models()


if __name__ == "__main__":
    main()
