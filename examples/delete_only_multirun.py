"""Demo: Real LLM multirun eval for paragraph compression by deletion only.

Task
----
Given a paragraph, delete less-important words while preserving grammaticality.
The rewritten text should only remove words from the original text (no additions or
substitutions), and should target roughly 20% compression (new/original word ratio
near 0.80).

Evaluators
----------
1) deletion_only (code): checks whether output words are an ordered subsequence of
   original words (i.e., only deletions were performed).
2) compression_target (code): rewards ratios close to 0.80.
3) grammaticality (LLM): asks a judge model if output is grammatical.

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/delete_only_multirun.py
"""

import math
import os
import sys
import re
import time

import numpy as np

from openai import OpenAI

import promptstats as pstats


# ---------------------------------------------------------------------------
# Data: input paragraphs
# ---------------------------------------------------------------------------

INPUT_TEXTS = [
    (
        "During the weekly planning meeting, the product team reviewed customer "
        "feedback from the last release and identified several recurring pain points "
        "that were slowing adoption among new users. They noticed that onboarding "
        "steps were unclear, especially for people trying the platform for the first "
        "time. The team grouped complaints into themes and prioritized fixes that "
        "could be shipped quickly. By the end of the session, they had a concrete "
        "plan for the next sprint and clear owners for each task."
    ),
    (
        "Although the museum was crowded and the line moved slowly, the exhibit "
        "was thoughtfully organized, and the clear descriptions helped visitors "
        "understand the historical context. Curators arranged artifacts in a way "
        "that made the timeline easy to follow from room to room. Short audio "
        "segments added personal stories without overwhelming people with details. "
        "Even with the noise, most visitors stayed longer than expected and left "
        "with a strong sense of the period."
    ),
    (
        "After months of testing in different environments, the engineers confirmed "
        "that the updated sensor remained stable under temperature swings and "
        "continued to report reliable measurements. They compared the new readings "
        "against calibrated reference equipment and found only minor deviations. "
        "When unexpected spikes appeared, the team traced them to wiring issues "
        "rather than the sensor design itself. The final report concluded that the "
        "update was ready for broader deployment."
    ),
    (
        "The neighborhood association proposed planting additional trees along the "
        "main street because summer heat has increased noticeably and shaded "
        "sidewalks would make walking safer for children and older residents. "
        "Residents shared examples of afternoons when pavement temperatures became "
        "uncomfortable and bus stops offered little relief. Volunteers mapped "
        "locations where roots would not interfere with utilities or driveways. "
        "The proposal also included a maintenance schedule to keep the new trees "
        "healthy through dry months."
    ),
    (
        "When the conference ended, several attendees stayed behind to compare notes, "
        "exchange contact information, and discuss how they might collaborate on "
        "related research projects over the coming year. They discovered overlapping "
        "interests in data quality, reproducibility, and low-cost field methods. "
        "One group proposed a shared repository so teams could reuse templates and "
        "avoid duplicating setup work. Before leaving, they scheduled a follow-up "
        "call and agreed on a short list of initial experiments."
    ),
    (
        "Before publishing the report, the analysts double-checked each chart, "
        "verified the underlying assumptions, and rewrote the summary so that "
        "non-technical readers could understand the key conclusions. They removed "
        "jargon where possible and replaced acronyms with plain-language definitions. "
        "A final review caught a mislabeled axis and an outdated footnote from an "
        "earlier draft. After those fixes, the document told a clearer and more "
        "trustworthy story."
    ),
    (
        "Because the trail became muddy after overnight rain, hikers were advised "
        "to wear waterproof boots, keep to marked paths, and allow extra time for "
        "the steep sections near the summit. Rangers also recommended trekking poles "
        "for better balance on slippery inclines. Several switchbacks had standing "
        "water, so visitors were asked not to cut across vegetation. Conditions "
        "improved by afternoon, but caution remained necessary on shaded sections."
    ),
    (
        "The school introduced a new reading program that combines daily independent "
        "practice with small-group instruction, and early results suggest students "
        "are gaining confidence as well as fluency. Teachers used brief assessments "
        "to place students in groups tailored to their current skill levels. "
        "Families received simple take-home activities so practice could continue "
        "outside the classroom. After six weeks, many students were reading aloud "
        "more smoothly and participating more actively in discussions."
    ),
    (
        "Even though the initial prototype looked promising in demonstrations, "
        "the team discovered during field trials that battery life dropped sharply "
        "in colder conditions and needed further optimization. Performance remained "
        "acceptable indoors, but outdoor tests exposed a consistent decline after "
        "sunset. Engineers adjusted power management settings and insulated key "
        "components to reduce thermal loss. The revised build showed improvement, "
        "though additional validation was still required."
    ),
    (
        "To reduce processing delays, the operations manager reorganized the intake "
        "queue, clarified handoff procedures between teams, and scheduled brief "
        "check-ins to resolve blockers before they accumulated. The new workflow "
        "defined priority levels so urgent cases could move forward without waiting "
        "behind routine requests. Team leads tracked turnaround time daily and flagged "
        "steps where work repeatedly stalled. Within two weeks, average completion "
        "time had dropped and fewer items were aging in the queue."
    ),
]

INPUT_LABELS = [f"paragraph_{i:02d}" for i in range(len(INPUT_TEXTS))]
N_INPUTS = len(INPUT_TEXTS)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Minimal": (
        "Delete less important words from the paragraph while keeping it grammatical.\n"
        "Do not add, replace, or reorder words. Only delete words.\n"
        "Target about 20% fewer words.\n\n"
        "Paragraph:\n{text}\n\n"
        "Rewritten paragraph:"
    ),
    "Strict Rules": (
        "Rewrite the paragraph by deletion only.\n"
        "Hard constraints:\n"
        "1) Keep original word order.\n"
        "2) Do not introduce new words.\n"
        "3) Do not substitute words.\n"
        "4) Keep grammar natural.\n"
        "5) Aim for about 80% of original word count.\n\n"
        "Return only the rewritten paragraph.\n\n"
        "Paragraph:\n{text}"
    ),
    "Few-shot": (
        "Task: compress by deleting less-important words only.\n"
        "Do not add or substitute words. Keep grammar intact.\n\n"
        "Example:\n"
        "Original: The committee met on Tuesday to review the budget proposal and discuss possible adjustments for next quarter.\n"
        "Rewritten: The committee met Tuesday to review the budget proposal and discuss adjustments for next quarter.\n\n"
        "Now rewrite this paragraph:\n{text}\n\n"
        "Rewritten paragraph:"
    ),
    "Edit Pass": (
        "You are performing a deletion-only edit pass.\n"
        "Remove unnecessary words, preserve meaning and grammaticality, and keep all remaining words exactly as written in the source.\n"
        "Aim for roughly 20% fewer words.\n\n"
        "Source paragraph:\n{text}\n\n"
        "Output only the edited paragraph."
    ),
}

TEMPLATE_LABELS = list(TEMPLATES.keys())
N_TEMPLATES = len(TEMPLATES)
N_RUNS = 3
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _is_subsequence(candidate: list[str], reference: list[str]) -> bool:
    if not candidate:
        return True
    idx = 0
    for token in reference:
        if idx < len(candidate) and candidate[idx] == token:
            idx += 1
            if idx == len(candidate):
                return True
    return idx == len(candidate)


def eval_deletion_only(output: str, source_text: str) -> float:
    """1.0 if output can be formed by deleting words from source, else 0.0."""
    source_tokens = _tokenize_words(source_text)
    output_tokens = _tokenize_words(output)
    return 1.0 if _is_subsequence(output_tokens, source_tokens) else 0.0


def eval_compression_target(output: str, source_text: str) -> float:
    """Reward output/original word ratio close to 0.80 (20% fewer words)."""
    source_count = len(_tokenize_words(source_text))
    output_count = len(_tokenize_words(output))
    if source_count == 0 or output_count == 0:
        return 0.0

    ratio = output_count / source_count
    sigma = 0.15
    score = math.exp(-((ratio - 0.80) / sigma) ** 2)

    # Extra penalty when output is longer than source.
    if ratio > 1.0:
        score *= 0.25
    return float(score)


GRAMMAR_MODEL = "gpt-4.1-nano"


def _extract_grammar_label(text: str) -> str:
    up = text.upper()
    if "GRAMMATICAL" in up and "UNGRAMMATICAL" not in up:
        return "GRAMMATICAL"
    if "UNGRAMMATICAL" in up:
        return "UNGRAMMATICAL"
    return "UNKNOWN"


def eval_grammaticality_llm(output: str, source_text: str, client: OpenAI) -> float:
    """LLM judge: 1.0 for grammatical output, 0.0 otherwise."""
    prompt = (
        "You are a strict grammar judge for English text.\n"
        "Assess only whether the rewritten paragraph is grammatically well-formed.\n"
        "Ignore style preference and compression level.\n\n"
        "Original paragraph (context only):\n"
        f"{source_text}\n\n"
        "Rewritten paragraph to judge:\n"
        f"{output}\n\n"
        "Respond with exactly one token: GRAMMATICAL or UNGRAMMATICAL."
    )
    response = client.chat.completions.create(
        model=GRAMMAR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=8,
    )
    label = _extract_grammar_label(response.choices[0].message.content.strip())
    return 1.0 if label == "GRAMMATICAL" else 0.0


EVALUATOR_NAMES = ["deletion_only", "compression_target", "grammaticality"]


# ---------------------------------------------------------------------------
# Generation model call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def call_model(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_completion_tokens=220,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(client: OpenAI) -> tuple[np.ndarray, list[list[list[str]]]]:
    """Run each (template, input) pair N times and score with 3 evaluators.

    Returns
    -------
    scores : np.ndarray
        Shape (N_templates, N_inputs, N_runs, 3).
    outputs : list[list[list[str]]]
        Raw outputs indexed by [run_idx][template_idx][input_idx].
    """
    scores = np.zeros((N_TEMPLATES, N_INPUTS, N_RUNS, 3))
    outputs = [[[None] * N_INPUTS for _ in range(N_TEMPLATES)] for _ in range(N_RUNS)]

    total = N_RUNS * N_TEMPLATES * N_INPUTS
    done = 0

    for r_idx in range(N_RUNS):
        print(f"\n--- Run {r_idx + 1}/{N_RUNS} ---")
        for t_idx, (template_name, template) in enumerate(TEMPLATES.items()):
            for i_idx, source_text in enumerate(INPUT_TEXTS):
                prompt = template.format(text=source_text)
                output = call_model(prompt, client)
                outputs[r_idx][t_idx][i_idx] = output

                deletion_score = eval_deletion_only(output, source_text) * 0.5
                compression_score = eval_compression_target(output, source_text)
                grammar_score = eval_grammaticality_llm(output, source_text, client)

                scores[t_idx, i_idx, r_idx, 0] = deletion_score
                scores[t_idx, i_idx, r_idx, 1] = compression_score
                scores[t_idx, i_idx, r_idx, 2] = grammar_score

                done += 1
                source_wc = len(_tokenize_words(source_text))
                output_wc = len(_tokenize_words(output))
                ratio = output_wc / source_wc if source_wc else 0.0
                print(
                    f"  [{done:3d}/{total}] {template_name:<12s} | input {i_idx:02d} | "
                    f"del={deletion_score:.0f} gram={grammar_score:.0f} ratio={ratio:.2f} | "
                    f"'{output[:45]}'"
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
    print(f"Grammar LLM: {GRAMMAR_MODEL}")
    print(f"Templates  : {N_TEMPLATES}  ({', '.join(TEMPLATE_LABELS)})")
    print(f"Inputs     : {N_INPUTS}")
    print(f"Runs       : {N_RUNS}  (temperature={TEMPERATURE})")
    print(f"Evaluators : {len(EVALUATOR_NAMES)}  ({', '.join(EVALUATOR_NAMES)})")
    print(f"Total generation calls: {N_RUNS * N_TEMPLATES * N_INPUTS}")
    print(f"Total grammar-judge calls: {N_RUNS * N_TEMPLATES * N_INPUTS}\n")

    print("Running benchmark …")
    t0 = time.time()
    raw_scores, _ = run_benchmark(client)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    result = pstats.BenchmarkResult(
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
    analysis_agg = pstats.analyze(
        result,
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
        result,
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
        scores=result.get_2d_scores(),
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
    )
    fig = pstats.plot_point_advantage(
        result_2d,
        reference="grand_mean",
        title=(
            "Prompt Template Comparison — Deletion-Only Paragraph Compression\n"
            f"({MODEL}, {N_RUNS} runs, composite score over {len(EVALUATOR_NAMES)} evaluators)"
        ),
        rng=np.random.default_rng(0),
    )
    fig.savefig("demo_analyze_multirun_delete_words_advantage.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_analyze_multirun_delete_words_advantage.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
