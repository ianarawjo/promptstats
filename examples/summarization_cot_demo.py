"""Demo: Should I add chain-of-thought to my production summarization prompt?

A common engineering instinct is to assume that chain-of-thought (CoT) prompting
always improves results. But for a *production* system the right question isn't
"what's the highest average score?" — it's "which prompt is the *safest*, most
consistent choice across the full range of documents I'll encounter?"

This example benchmarks four prompt variants for a document summarization task:

  Baseline     — simple, direct instruction
  Few-shot     — two example document→summary pairs
  CoT          — "identify key points, then summarize"
  CoT+Few-shot — examples plus chain-of-thought structure

Summaries are scored by an LLM judge on two dimensions (each 0–1):
  faithfulness — does the summary accurately reflect the source?
  coherence    — is the summary clear and readable?

With 30 documents and 3 runs each, evalstats reveals not just which prompt
has the highest mean score, but which has the lowest variance — the property
that matters most when deploying to production.

Key finding (that a naive mean would miss): CoT tends to slightly improve mean
scores but introduces more variance. It can be brilliant on complex documents
yet unreliable on short or simple ones, where extra reasoning steps sometimes
overcomplicate the output. The safest production choice may be the boring one.

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/summarization_cot_demo.py
"""

import json
import os
import sys
import time

import numpy as np
from openai import OpenAI

import evalstats as estats


# ---------------------------------------------------------------------------
# Configuration — tweak these to trade off cost vs. statistical power
# ---------------------------------------------------------------------------

SUMMARIZER_MODEL = "gpt-4.1-nano"
JUDGE_MODEL = "gpt-4.1-nano"
N_RUNS = 3        # repeated runs per (template, document) pair
TEMPERATURE = 0.7  # non-zero so repeated runs produce meaningful variation


# ---------------------------------------------------------------------------
# Task: document summarization — 30 varied documents
#
# Documents span three difficulty tiers:
#   Easy (01–10):   clear structure, single main point, short
#   Medium (11–20): multiple facts, mild nuance, or mixed signals
#   Hard (21–30):   technical language, statistical caveats, or conflicting info
#
# The hard tier is where CoT prompts tend to diverge: sometimes they help by
# forcing structured planning; other times the extra reasoning step leads the
# model to fixate on wrong details or produce overcrowded summaries.
# ---------------------------------------------------------------------------

DOCUMENTS = [
    # ── Easy ─────────────────────────────────────────────────────────────────
    # 01
    "Meridian Coffee announced the opening of its 50th location in downtown "
    "Portland on Tuesday. The new café features a rooftop terrace, a dedicated "
    "workspace area with 30 seats, and a limited-edition seasonal menu. The "
    "company said it plans to add 15 more locations across the Pacific Northwest "
    "by end of year.",

    # 02
    "The Springfield school district has approved a $4.2 million renovation of "
    "Roosevelt Elementary, including new classrooms, upgraded HVAC systems, and "
    "an expanded gymnasium. Construction will begin in June and is expected to "
    "be completed before the start of the fall semester.",

    # 03
    "Lakewood City Council unanimously voted Tuesday to convert the vacant lot "
    "at 5th and Maple into a community garden. Local nonprofit GreenRoots will "
    "manage the space and provide free garden plots to residents on a first-come, "
    "first-served basis starting this spring.",

    # 04
    "The Redwood Valley Marathon recorded its highest-ever participation this "
    "year, with 12,400 registered runners completing the course. Event organizers "
    "credited a new half-marathon category and expanded accessibility options "
    "for the jump in turnout.",

    # 05
    "Vertex Analytics released version 4.0 of its data visualization platform "
    "today, adding real-time collaboration features, a new charting library, and "
    "support for importing data directly from Salesforce. The update is available "
    "to all existing customers at no additional charge.",

    # 06
    "Harbor Foods reported fourth-quarter revenue of $1.14 billion, up 9% "
    "year-over-year, beating analyst estimates of $1.08 billion. The company "
    "attributed the outperformance to strong demand in its frozen meal segment "
    "and raised full-year guidance by $80 million.",

    # 07
    "The National Weather Service has issued a winter storm warning for the "
    "greater Denver area, forecasting 8–14 inches of snow between Thursday "
    "evening and Saturday morning. Residents are advised to limit travel and "
    "prepare emergency kits.",

    # 08
    "Sunrise Health Systems will open a new pediatric urgent care clinic in "
    "the Elm Park neighborhood on April 1. The facility will offer extended "
    "evening and weekend hours and accept all major insurance plans.",

    # 09
    "Aldgate University announced that Dr. Priya Nair has been appointed as "
    "the new dean of its School of Engineering, effective July 1. Dr. Nair "
    "joins from MIT, where she led a lab focused on sustainable materials science.",

    # 10
    "The Coastal Restoration Fund raised $3.1 million at its annual gala last "
    "weekend, exceeding its $2.5 million goal. Funds will be used to restore "
    "six miles of wetland habitat along the Gulf Coast over the next two years.",

    # ── Medium ────────────────────────────────────────────────────────────────
    # 11
    "Pinnacle Retail Group reported a 12% increase in total sales for Q3, "
    "driven primarily by its online channel, which grew 34%. However, gross "
    "margins contracted by 2.1 percentage points due to elevated shipping costs "
    "and promotional discounting. Management guided for margin recovery in Q4 "
    "as freight rates normalize.",

    # 12
    "Crestfield Asset Management announced Monday that it has acquired a "
    "controlling stake in MedLogix, a healthcare data analytics startup, for an "
    "undisclosed sum. The deal gives Crestfield a foothold in health-tech while "
    "providing MedLogix access to Crestfield's distribution network and advisory "
    "resources. Regulatory approval is expected within 60 days.",

    # 13
    "Effective February 1, employees' dental and vision coverage will transfer "
    "from ClearPath Insurance to Summit Benefits. Premiums, deductibles, and "
    "in-network providers remain unchanged for the current plan year. Employees "
    "with pre-authorization requests in progress should contact HR by January 20 "
    "to ensure continuity of care.",

    # 14
    "GreenEdge Technologies and Nairobi Power Authority signed a 10-year "
    "partnership to develop three utility-scale solar farms totaling 400 MW "
    "across western Kenya. Financed through a blended public-private model, the "
    "project is projected to bring electricity access to approximately 250,000 "
    "households by 2028.",

    # 15
    "Following a board-approved review of capital allocation, the marketing "
    "budget will be reduced by 15% in H2, with the freed capital redirected to "
    "R&D and supply-chain resilience. Team leads should submit revised campaign "
    "plans by the 15th of next month.",

    # 16
    "Beginning March 1, all inbound customer support requests will be triaged "
    "through the new Zenith ticketing platform before assignment to agents. "
    "Response-time SLAs remain unchanged: four hours for P1, 24 hours for P2, "
    "72 hours for P3. Mandatory training sessions will be held February 12–13.",

    # 17
    "Atlas Software will relocate its headquarters from San Jose to Austin in "
    "Q3, citing lower operating costs and access to a growing engineering talent "
    "pool. The San Jose office will remain open as a West Coast hub with reduced "
    "headcount; affected employees will be offered relocation packages or "
    "remote-work arrangements.",

    # 18
    "We are pleased to welcome Marcus Okonkwo as Vice President of Sales, "
    "effective immediately. Marcus brings 18 years of enterprise software sales "
    "experience, most recently as Regional Director at Salesforce overseeing a "
    "$400 million book of business. He will report directly to the CEO and be "
    "based in New York.",

    # 19
    "After 12 years of partnership, Orion Logistics has informed us that it "
    "will not renew its freight contract, citing a strategic shift toward "
    "in-house logistics. Current Orion-managed shipments will transition to our "
    "secondary carrier, Western Freight, effective April 30. No service "
    "disruptions are anticipated during the transition.",

    # 20
    "In the fiscal year ending December 31, the company posted total revenue of "
    "$892 million (up 7% YoY), adjusted EBITDA of $134 million (up 14% YoY), "
    "and free cash flow of $97 million. Net debt decreased to $210 million. "
    "The board declared a $0.22 per share dividend, payable February 28.",

    # ── Hard ──────────────────────────────────────────────────────────────────
    # 21
    "As of version 3.9, the /v1/completions endpoint will be deprecated and "
    "removed in version 4.0, scheduled for release in Q2. All integrations must "
    "migrate to /v1/chat/completions, which supports the same models via the "
    "messages[] array format. Authentication tokens issued before 2023-06-01 "
    "will also be invalidated at the same time; developers should rotate "
    "credentials now to avoid service interruption.",

    # 22
    "On January 14, we identified unauthorized access to a subset of our "
    "logging infrastructure. The actor gained entry via a misconfigured OAuth2 "
    "token with excessive scopes and maintained access for approximately 6 hours. "
    "No production databases were accessed and no customer PII was exposed. "
    "Affected tokens have been revoked and scope restrictions are now enforced "
    "at the API gateway level.",

    # 23
    "A randomized controlled trial (n=312) evaluated the efficacy of once-daily "
    "liraglutide 1.8 mg versus placebo in adults with non-alcoholic fatty liver "
    "disease. At week 48, 39% of the liraglutide group achieved the primary "
    "endpoint (≥1-point reduction in NAFLD Activity Score) versus 24% of placebo "
    "(p=0.003). Secondary endpoints, including liver stiffness by elastography, "
    "did not reach statistical significance.",

    # 24
    "The parties have reached a settlement in the class-action suit alleging "
    "deceptive subscription billing practices. The company will pay $18.5 million "
    "into a settlement fund; approximately 35% will cover attorneys' fees and "
    "administration. Class members who submit valid claims by the deadline will "
    "receive a pro-rata share, estimated at $12–$48 per claimant depending on "
    "claim volume. The company denies all liability.",

    # 25
    "Halcyon Medical is voluntarily recalling lot numbers M220401–M220490 of "
    "its OptiFlux II infusion pump due to a firmware defect that can cause the "
    "device to deliver up to 8% more medication than programmed when operating "
    "in piggyback mode. No adverse events have been reported. Healthcare "
    "providers should immediately suspend use of affected units and contact "
    "Halcyon Field Service for a complimentary firmware update.",

    # 26
    "The FDA has granted accelerated approval to Verovax-T for relapsed or "
    "refractory follicular lymphoma in patients who have received at least two "
    "prior lines of therapy. Approval is contingent on the sponsor completing a "
    "confirmatory Phase III randomized trial within 48 months; failure to do so "
    "may result in withdrawal of approval. The recommended dose is 400 mg IV "
    "every three weeks.",

    # 27
    "After evaluating event sourcing with CQRS, a traditional relational model, "
    "and a document store, the platform team has decided to adopt PostgreSQL with "
    "JSONB columns for the Order service. This choice prioritizes operational "
    "simplicity and team familiarity over the eventual-consistency advantages of "
    "event sourcing, accepting a higher long-term migration cost if write "
    "throughput requirements exceed current 5× projections.",

    # 28
    "Phase II data from the CLARITY-2 trial (n=186) showed that 61% of patients "
    "receiving the combination regimen (venetoclax + azacitidine) achieved "
    "complete remission at cycle 6, versus 42% in the monotherapy arm (p=0.01). "
    "Median progression-free survival was 14.3 months versus 9.1 months "
    "(HR=0.64, 95% CI: 0.46–0.89). Grade 3/4 neutropenia occurred in 58% of "
    "combination patients, requiring dose modification in 22% of cases.",

    # 29
    "Our addressable market in North America is estimated at $14.2 billion, "
    "growing at ~11% annually. We currently hold approximately 4% market share, "
    "concentrated in the mid-market segment. Key competitive threats include "
    "legacy incumbents with high switching costs and two well-funded startups "
    "targeting enterprise with AI-native offerings. Our near-term differentiator "
    "is integrations breadth; our long-term moat — data network effects — "
    "requires scale we have not yet achieved.",

    # 30
    "Integration of the Northgate acquisition is proceeding on schedule across "
    "most workstreams; however, the CRM consolidation has slipped by six weeks "
    "due to data-mapping complexity between our Salesforce instance and "
    "Northgate's legacy Dynamics CRM. ERP cutover remains on track for Q3. "
    "Synergy realization stands at $22M annualized versus a Year 1 target of "
    "$35M, with the shortfall attributable to delayed headcount rationalization "
    "in the acquired entity's finance function.",
]

assert len(DOCUMENTS) == 30, f"Expected 30 documents, got {len(DOCUMENTS)}"

INPUT_LABELS = [f"doc_{i + 1:02d}" for i in range(len(DOCUMENTS))]
N_INPUTS = len(DOCUMENTS)


# ---------------------------------------------------------------------------
# Few-shot examples
# These documents are *not* in the test set above.
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = (
    "Example 1:\n"
    "Document: City planners have approved a $9 million redesign of Harborview "
    "Park, which will include new walking paths, a children's splash pad, and "
    "improved lighting throughout the grounds. Work is expected to begin in "
    "September and be completed within 14 months.\n"
    "Summary: Harborview Park will undergo a $9 million redesign featuring new "
    "walking paths, a splash pad, and improved lighting. Construction begins in "
    "September with a 14-month timeline.\n\n"
    "Example 2:\n"
    "Document: A 12-week NIH study of 240 adults found that daily 20-minute "
    "walks reduced symptoms of mild depression by 31% compared with standard "
    "care alone. Lead author Dr. Sarah Chen cautioned that the findings should "
    "complement existing treatments rather than replace them.\n"
    "Summary: An NIH study found 20-minute daily walks reduced mild depression "
    "symptoms by 31% over 12 weeks. Researchers emphasize the approach "
    "supplements, rather than replaces, standard treatment.\n\n"
)

_FEW_SHOT_COT_EXAMPLES = (
    "Example 1:\n"
    "Document: City planners have approved a $9 million redesign of Harborview "
    "Park, which will include new walking paths, a children's splash pad, and "
    "improved lighting throughout the grounds. Work is expected to begin in "
    "September and be completed within 14 months.\n"
    "Key points: $9M redesign, walking paths + splash pad + lighting, "
    "September start, 14-month timeline\n"
    "Summary: Harborview Park will undergo a $9 million redesign featuring new "
    "walking paths, a splash pad, and improved lighting. Construction begins in "
    "September with a 14-month timeline.\n\n"
    "Example 2:\n"
    "Document: A 12-week NIH study of 240 adults found that daily 20-minute "
    "walks reduced symptoms of mild depression by 31% compared with standard "
    "care alone. Lead author Dr. Sarah Chen cautioned that the findings should "
    "complement existing treatments rather than replace them.\n"
    "Key points: NIH study, 240 adults, 12 weeks, 20-min walks, 31% symptom "
    "reduction, complements (not replaces) treatment\n"
    "Summary: An NIH study found 20-minute daily walks reduced mild depression "
    "symptoms by 31% over 12 weeks. Researchers emphasize the approach "
    "supplements, rather than replaces, standard treatment.\n\n"
)


# ---------------------------------------------------------------------------
# Prompt templates — four variants of the same task
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Baseline": (
        "Summarize the following document in 2–3 concise sentences.\n\n"
        "Document:\n{document}\n\n"
        "Summary:"
    ),
    "Few-shot": (
        "Summarize the following document in 2–3 concise sentences.\n\n"
        + _FEW_SHOT_EXAMPLES
        + "Now summarize:\n"
        "Document:\n{document}\n\n"
        "Summary:"
    ),
    "CoT": (
        "Summarize the following document in 2–3 concise sentences.\n\n"
        "Before writing your summary, identify the 2–3 most important facts "
        "you will cover. Format your response exactly as:\n\n"
        "Key points: <your brief notes>\n\n"
        "Summary: <your 2–3 sentence summary>\n\n"
        "Document:\n{document}"
    ),
    "CoT+Few-shot": (
        "Summarize the following document in 2–3 concise sentences.\n\n"
        "Before writing your summary, identify the 2–3 most important facts "
        "you will cover. Format your response exactly as:\n\n"
        "Key points: <your brief notes>\n\n"
        "Summary: <your 2–3 sentence summary>\n\n"
        + _FEW_SHOT_COT_EXAMPLES
        + "Now summarize:\n"
        "Document:\n{document}"
    ),
}

TEMPLATE_LABELS = list(TEMPLATES.keys())
N_TEMPLATES = len(TEMPLATES)

EVALUATOR_NAMES = ["faithfulness", "coherence"]


# ---------------------------------------------------------------------------
# LLM call — summarizer
# ---------------------------------------------------------------------------

def call_model(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=SUMMARIZER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()


def extract_summary(output: str) -> str:
    """Extract the final summary from CoT-style outputs.

    CoT templates instruct the model to produce "Key points: ..." followed by
    "Summary: ...". This helper returns just the text after the last "Summary:"
    marker so the judge evaluates only the summary, not the reasoning trace.
    For Baseline and Few-shot outputs the full text is returned unchanged.
    """
    if "Summary:" in output:
        return output.split("Summary:")[-1].strip()
    return output.strip()


# ---------------------------------------------------------------------------
# LLM judge — faithfulness and coherence scored 0–10, normalised to 0–1
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating the quality of a document summary.

Document:
{document}

Summary:
{summary}

Rate the summary on two dimensions (each scored 0–10):
- faithfulness: Does the summary accurately represent the document? \
Penalise if it adds false information or misrepresents facts. \
(0 = fabricated/completely wrong, 10 = perfectly accurate)
- coherence: Is the summary clear, readable, and well-formed as standalone \
text? (0 = incomprehensible, 10 = excellent prose)

Respond with JSON only, no other text:
{{"faithfulness": <integer 0-10>, "coherence": <integer 0-10>}}"""


def call_judge(document: str, summary: str, client: OpenAI) -> dict[str, float]:
    """Return {{'faithfulness': 0–1, 'coherence': 0–1}} from the judge model."""
    prompt = _JUDGE_PROMPT.format(document=document, summary=summary)
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=64,
        )
        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)
        return {
            "faithfulness": float(scores["faithfulness"]) / 10.0,
            "coherence": float(scores["coherence"]) / 10.0,
        }
    except Exception:
        # Return midpoint on parse failure rather than crashing the benchmark.
        return {"faithfulness": 0.5, "coherence": 0.5}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(client: OpenAI) -> np.ndarray:
    """Run all (template, document, run) triples and judge each summary.

    Returns
    -------
    scores : np.ndarray
        Shape ``(N_templates, N_inputs, N_runs, N_evaluators)``
        matching the evalstats ``(N, M, R, K)`` convention.
    """
    n_evals = len(EVALUATOR_NAMES)
    scores = np.zeros((N_TEMPLATES, N_INPUTS, N_RUNS, n_evals))
    total = N_TEMPLATES * N_INPUTS * N_RUNS
    done = 0

    for r_idx in range(N_RUNS):
        print(f"\n--- Run {r_idx + 1}/{N_RUNS} ---")
        for t_idx, (t_name, template) in enumerate(TEMPLATES.items()):
            for i_idx, document in enumerate(DOCUMENTS):
                prompt = template.format(document=document)
                raw_output = call_model(prompt, client)
                summary = extract_summary(raw_output)

                judge = call_judge(document, summary, client)
                scores[t_idx, i_idx, r_idx, 0] = judge["faithfulness"]
                scores[t_idx, i_idx, r_idx, 1] = judge["coherence"]

                done += 1
                composite = (judge["faithfulness"] + judge["coherence"]) / 2
                print(
                    f"  [{done:3d}/{total}] {t_name:<14s} | "
                    f"run {r_idx + 1}/{N_RUNS} | doc {i_idx + 1:02d} | "
                    f"faithful={judge['faithfulness']:.2f}  "
                    f"coherent={judge['coherence']:.2f}  "
                    f"composite={composite:.2f} | "
                    f"'{summary[:50]}'"
                )

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    summarizer_calls = N_TEMPLATES * N_INPUTS * N_RUNS
    judge_calls = summarizer_calls  # one judge call per summary
    total_calls = summarizer_calls + judge_calls

    print(f"Summarizer : {SUMMARIZER_MODEL}")
    print(f"Judge      : {JUDGE_MODEL}")
    print(f"Templates  : {N_TEMPLATES}  ({', '.join(TEMPLATE_LABELS)})")
    print(f"Documents  : {N_INPUTS}  (easy: 1–10, medium: 11–20, hard: 21–30)")
    print(f"Runs       : {N_RUNS}  (temperature={TEMPERATURE})")
    print(f"Evaluators : {len(EVALUATOR_NAMES)}  ({', '.join(EVALUATOR_NAMES)})")
    print(f"API calls  : {total_calls}  "
          f"({summarizer_calls} summarizer + {judge_calls} judge)\n")

    print("Running benchmark …")
    t0 = time.time()
    raw_scores = run_benchmark(client)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # Build BenchmarkResult — shape (N_templates, N_inputs, N_runs, N_evaluators)
    result = estats.BenchmarkResult(
        scores=raw_scores,
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
        evaluator_names=EVALUATOR_NAMES,
    )
    print(
        f"BenchmarkResult: {result.n_templates} templates × "
        f"{result.n_inputs} inputs × {N_RUNS} runs × "
        f"{len(EVALUATOR_NAMES)} evaluators\n"
    )

    # ------------------------------------------------------------------
    # Per-evaluator breakdown — faithfulness and coherence separately
    # ------------------------------------------------------------------
    print("=== analyze(evaluator_mode='per_evaluator') — faithfulness & coherence ===")
    analysis_per = estats.analyze(
        result,
        evaluator_mode="per_evaluator",

        method="auto",
        n_bootstrap=3_000,
        correction="holm",
        rng=np.random.default_rng(0),
    )
    estats.print_analysis_summary(analysis_per, top_pairwise=4)

    # ------------------------------------------------------------------
    # Full analysis — evaluators averaged into a single composite score
    # ------------------------------------------------------------------
    print("=== analyze(evaluator_mode='aggregate') — composite score ===")
    analysis_agg = estats.analyze(
        result,
        evaluator_mode="aggregate",

        method="auto",
        n_bootstrap=5_000,
        correction="holm",
        rng=np.random.default_rng(0),
    )
    estats.print_analysis_summary(analysis_agg, top_pairwise=6)
    print()

    # ------------------------------------------------------------------
    # Save advantage plot (composite score, faithfulness + coherence averaged)
    # ------------------------------------------------------------------
    result_2d = estats.BenchmarkResult(
        scores=result.get_2d_scores(),
        template_labels=TEMPLATE_LABELS,
        input_labels=INPUT_LABELS,
    )
    fig = estats.plot_point_estimates(
        result_2d,
        title=(
            "Prompt Template Comparison — Document Summarization\n"
            f"({SUMMARIZER_MODEL}, {N_RUNS} runs, "
            f"composite of {' + '.join(EVALUATOR_NAMES)})"
        ),
        rng=np.random.default_rng(0),
    )
    fname = "demo_summarization_cot_advantage.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    print("\nDone!")


if __name__ == "__main__":
    main()
