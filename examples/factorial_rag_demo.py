"""Factorial LMM demo: RAG pipeline with chunker × retrieval_method.

Simulates a benchmark where 30 questions are each evaluated with every
combination of two pipeline factors:

  * chunker   : fixed_512 | sliding_256 | semantic
  * retrieval : bm25 | dense | hybrid

This gives 3 × 3 = 9 treatment cells × 30 questions = 270 observations.
Each row in the DataFrame represents one pipeline output and records which
chunker and retrieval method produced it — the typical shape of a post-hoc
tagged RAG experiment.

Ground-truth effects used to generate the data:

  chunker effect   : fixed_512 = 0.00, sliding_256 = +0.03, semantic = +0.08
  retrieval effect : bm25 = 0.00, dense = +0.06, hybrid = +0.10
    interaction (hidden): semantic × hybrid = +0.05, fixed_512 × dense = -0.02

Usage::

    python examples/factorial_rag_demo.py
"""

import numpy as np
import pandas as pd

import promptstats as ps


# ---------------------------------------------------------------------------
# Simulate data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

CHUNKERS   = ["fixed_512", "sliding_256", "semantic"]
RETRIEVALS = ["bm25", "dense", "hybrid"]

CHUNKER_EFFECT   = {"fixed_512": 0.00, "sliding_256": 0.03, "semantic": 0.08}
RETRIEVAL_EFFECT = {"bm25": 0.00, "dense": 0.06, "hybrid": 0.10}
INTERACTION_EFFECT = {
    ("semantic", "hybrid"): 0.05,
    ("fixed_512", "dense"): -0.2,
}

N_QUESTIONS = 30
BASE_SCORE  = 0.65
SIGMA_INPUT = 0.10   # between-question variance (drives ICC)
SIGMA_RESID = 0.06   # residual within-cell noise

question_intercepts = rng.normal(0, SIGMA_INPUT, N_QUESTIONS)

rows = []
for q_idx in range(N_QUESTIONS):
    q_id = f"q{q_idx + 1:03d}"
    for chunker in CHUNKERS:
        for retrieval in RETRIEVALS:
            score = (
                BASE_SCORE
                + question_intercepts[q_idx]
                + CHUNKER_EFFECT[chunker]
                + RETRIEVAL_EFFECT[retrieval]
                + INTERACTION_EFFECT.get((chunker, retrieval), 0.0)
                + rng.normal(0, SIGMA_RESID)
            )
            rows.append({
                "input_id":  q_id,
                "chunker":   chunker,
                "retrieval": retrieval,
                "score":     float(np.clip(score, 0.0, 1.0)),
            })

data = pd.DataFrame(rows)

print(
    f"Dataset: {len(data)} rows, "
    f"{data['input_id'].nunique()} questions, "
    f"{data['chunker'].nunique()} chunkers × "
    f"{data['retrieval'].nunique()} retrieval methods\n"
)

# ---------------------------------------------------------------------------
# Run the factorial analysis
# ---------------------------------------------------------------------------

bundle = ps.analyze_factorial(
    data,
    factors=["chunker", "retrieval"],
    random_effect="input_id",
    score_col="score",
    rng=np.random.default_rng(0),
)

# ---------------------------------------------------------------------------
# Print results with the standard PromptStats summary renderer
# ---------------------------------------------------------------------------

ps.print_analysis_summary(bundle, top_pairwise=10)