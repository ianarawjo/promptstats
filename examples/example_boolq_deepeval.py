# example_boolq_deepeval.py
#
# Runs a small subset of BoolQ via deepeval for three models (OpenAI + OpenRouter),
# then passes results to promptstats for comparative analysis.
# Requires `deepeval`, `openai`, and API keys in the environment.
#
# Install deps:
#   pip install deepeval openai promptstats

import os
import pandas as pd
from deepeval.benchmarks import BoolQ
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

import promptstats  # your library


# ---------------------------------------------------------------------------
# 1. Wrap your LLM in DeepEvalBaseLLM
#    deepeval requires this thin adapter so it can call your model uniformly.
#    Swap out the body of generate() for any provider (Anthropic, local, etc.)
# ---------------------------------------------------------------------------

class OpenAIModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gpt-4.1-nano"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,       # BoolQ only needs "True" or "False"
            temperature=0,       # deterministic — important for benchmarking
        )
        return response.choices[0].message.content.strip()

    async def a_generate(self, prompt: str) -> str:
        # deepeval needs an async variant; simplest approach is to wrap generate()
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "mistralai/ministral-8b-2512"):
        self.model_name = model_name
        api_key = os.environ["OPENROUTER_API_KEY"]

        headers: dict[str, str] = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        if headers:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=headers,
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# 2. Configure and run BoolQ on a small subset for both models
#
#    n_problems: how many questions to evaluate (full dataset = 3270)
#    n_shots:    few-shot examples prepended to each prompt (max 5)
# ---------------------------------------------------------------------------

N_PROBLEMS = 50  # adjust as desired for a quick test or fuller evaluation
N_SHOTS_OPTIONS = [5, 3, 0]  # compare 5-shot, 3-shot, and 0-shot performance

openai_model_name = "gpt-4.1-nano"
openrouter_model_A = "mistralai/ministral-8b-2512"
openrouter_model_B = "google/gemma-3-4b-it"
openrouter_model_C = "qwen/qwen3-vl-8b-instruct"

models = [
    OpenAIModel(openai_model_name),
    OpenRouterModel(openrouter_model_A),
    OpenRouterModel(openrouter_model_B),
    OpenRouterModel(openrouter_model_C),
]


def run_boolq(model: DeepEvalBaseLLM, *, n_problems: int, n_shots: int) -> tuple[float, pd.DataFrame]:
    benchmark = BoolQ(n_problems=n_problems, n_shots=n_shots)
    benchmark.evaluate(model=model)
    return benchmark.overall_score, benchmark.predictions.copy()


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str:
    normalized = {
        str(col).strip().lower().replace("_", " "): col
        for col in df.columns
    }
    for alias in aliases:
        key = alias.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    raise KeyError(
        f"Could not find any of columns {aliases} in predictions DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# 3. Inspect raw results per model
#
#    overall_score  → float in [0, 1], proportion of correct answers
#    predictions    → DataFrame with columns: input, expected_output, prediction, correct
# ---------------------------------------------------------------------------

results_by_model: list[pd.DataFrame] = []

for model in models:
    for n_shots in N_SHOTS_OPTIONS:
        overall_score, predictions = run_boolq(
            model,
            n_problems=N_PROBLEMS,
            n_shots=n_shots,
        )
        print(
            f"[{model.get_model_name()} | {n_shots}-shot] "
            f"Overall accuracy: {overall_score:.3f}"
        )
        print(predictions.head())

        input_col = _resolve_col(predictions, ["input", "question"])
        correct_col = _resolve_col(predictions, ["correct", "is_correct", "accuracy"])

        model_rows = pd.DataFrame(
            {
                "model": model.get_model_name(),
                "template": f"{n_shots}-shot",
                # Keep alignment robust across model runs by keying on the question text.
                "input": predictions[input_col].astype(str),
                "score": predictions[correct_col].astype(float),
                "evaluator": "accuracy",
            }
        )
        results_by_model.append(model_rows)


# ---------------------------------------------------------------------------
# 4. Pass results into promptstats
#
#    promptstats.analyze() expects a BenchmarkResult / MultiModelBenchmark.
#    Build a long-form DataFrame with canonical columns, then parse it via
#    promptstats.from_dataframe(...).
# ---------------------------------------------------------------------------

results_long = pd.concat(results_by_model, ignore_index=True)

benchmark_result, load_report = promptstats.from_dataframe(
    results_long,
    format="long",
    return_report=True,
)

for line in load_report.to_lines():
    print(line)

analysis = promptstats.analyze(benchmark_result)
promptstats.print_analysis_summary(analysis)

"""
Output should look something like:

==============================
 MULTI-MODEL ANALYSIS SUMMARY 
==============================
Shape: BenchmarkShape(models=3, prompts=2, input_vars=1, evaluators=1)
Models: 3 | Templates: 2 | Inputs: 50
Models: gpt-4.1-nano, mistralai/ministral-8b-2512, google/gemma-3-4b-it
Best pair: model='mistralai/ministral-8b-2512'  template='0-shot'

Model-level comparison (mean across all prompts):
=== Analysis Summary ===
Shape: BenchmarkShape(models=3, prompts=3, input_vars=1, evaluators=1)
Models: 3 | Inputs: 50

--- Robustness ---
                             mean  median       std        cv  iqr  cvar_10  p10  p25  p50  p75  p90
template                                                                                            
gpt-4.1-nano                 0.69     1.0  0.389793  0.564918  0.5      0.0  0.0  0.5  1.0  1.0  1.0
mistralai/ministral-8b-2512  0.75     1.0  0.381324  0.508432  0.5      0.0  0.0  0.5  1.0  1.0  1.0
google/gemma-3-4b-it         0.75     1.0  0.367701  0.490268  0.5      0.0  0.0  0.5  1.0  1.0  1.0

--- Rank Probabilities ---
  Model                      P(Best)   E[Rank]
  gpt-4.1-nano                 4.6%     2.70
  mistralai/ministral-8b-2512    50.4%     1.59
  google/gemma-3-4b-it        44.9%     1.71

--- Mean Advantage (reference=grand_mean) ---
  axis: [-0.333, +0.333]  (· spread, ─ CI, ● mean, │ zero)  spread percentiles = (10, 90)
  Model                    Interval Plot                                 Mean    CI Low   CI High  Spread Lo  Spread Hi
  gpt-4.1-nano             ··············────●─│──·········           -0.040   -0.093   +0.027    -0.333    +0.183
  mistralai/ministral-8b-…           ·······───│●───······            +0.020   -0.050   +0.067    -0.167    +0.167
  google/gemma-3-4b-it     ·················───│●────···············  +0.020   -0.053   +0.090    -0.333    +0.333

--- Pairwise Comparisons (lowest p-value first) ---
  axis: [-0.484, +0.484]  (· ±1σ, ─ CI, ● mean, │ zero)
  Pair                             Interval Plot                                 Mean    CI Low   CI High        σ   p (boot)   p (wsr)
  gpt-4.1-nano vs mistralai/minis…    ···········────●─│─···········          -0.0600   -0.1500   +0.0300   0.3446     0.6245    0.7034
  gpt-4.1-nano vs google/gemma-3-… ·············─────●─│──·············       -0.0600   -0.1800   +0.0500   0.4243     0.6323    0.7125
  mistralai/ministral-8b-2512 vs …    ············─────●────·············     +0.0000   -0.1300   +0.1000   0.4165          1    0.8935
  p (boot) = bootstrap holm-corrected; p (wsr) = Wilcoxon signed-rank holm-corrected
  stars: * p<0.05, ** p<0.01, *** p<0.001

=================================
 PER-MODEL SUMMARY: GPT-4.1-NANO 
=================================
=== Analysis Summary ===
Shape: BenchmarkShape(models=1, prompts=2, input_vars=1, evaluators=1)
Templates: 2 | Inputs: 50

--- Robustness ---
          mean  median       std        cv  iqr  cvar_10  p10  p25  p50  p75  p90
template                                                                         
3-shot    0.62     1.0  0.490314  0.790830  1.0      0.0  0.0  0.0  1.0  1.0  1.0
0-shot    0.76     1.0  0.431419  0.567657  0.0      0.0  0.0  1.0  1.0  1.0  1.0

--- Rank Probabilities ---
  Template                   P(Best)   E[Rank]
  3-shot                       3.2%     1.97
  0-shot                      96.8%     1.03

--- Mean Advantage (reference=grand_mean) ---
  axis: [-0.500, +0.500]  (· spread, ─ CI, ● mean, │ zero)  spread percentiles = (10, 90)
  Template                 Interval Plot                                 Mean    CI Low   CI High  Spread Lo  Spread Hi
  3-shot                   ··············───●──│                      -0.070   -0.150   -0.010    -0.500    +0.000
  0-shot                                       │──●──···············  +0.070   +0.000   +0.130    +0.000    +0.500

--- Pairwise Comparisons (lowest p-value first) ---
  axis: [-0.635, +0.635]  (· ±1σ, ─ CI, ● mean, │ zero)
  Pair                             Interval Plot                                 Mean    CI Low   CI High        σ   p (boot)   p (wsr)
  3-shot vs 0-shot                 ···········─────●───│···········           -0.1400   -0.3000   -0.0400   0.4953    0.0486*    0.0522
  p (boot) = bootstrap holm-corrected; p (wsr) = Wilcoxon signed-rank holm-corrected
  stars: * p<0.05, ** p<0.01, *** p<0.001

================================================
 PER-MODEL SUMMARY: MISTRALAI/MINISTRAL-8B-2512 
================================================
=== Analysis Summary ===
Shape: BenchmarkShape(models=1, prompts=2, input_vars=1, evaluators=1)
Templates: 2 | Inputs: 50

--- Robustness ---
          mean  median       std        cv  iqr  cvar_10  p10  p25  p50  p75  p90
template                                                                         
3-shot    0.72     1.0  0.453557  0.629941  1.0      0.0  0.0  0.0  1.0  1.0  1.0
0-shot    0.78     1.0  0.418452  0.536477  0.0      0.0  0.0  1.0  1.0  1.0  1.0

--- Rank Probabilities ---
  Template                   P(Best)   E[Rank]
  3-shot                      20.5%     1.79
  0-shot                      79.5%     1.21

--- Mean Advantage (reference=grand_mean) ---
  axis: [-0.500, +0.500]  (· spread, ─ CI, ● mean, │ zero)  spread percentiles = (10, 90)
  Template                 Interval Plot                                 Mean    CI Low   CI High  Spread Lo  Spread Hi
  3-shot                   ················───●│─                     -0.030   -0.100   +0.020    -0.500    +0.000
  0-shot                                     ──│●──·················  +0.030   -0.040   +0.080    +0.000    +0.500

--- Pairwise Comparisons (lowest p-value first) ---
  axis: [-0.484, +0.484]  (· ±1σ, ─ CI, ● mean, │ zero)
  Pair                             Interval Plot                                 Mean    CI Low   CI High        σ   p (boot)   p (wsr)
  3-shot vs 0-shot                 ············──────●─│─··············       -0.0600   -0.2000   +0.0200   0.4243     0.3125    0.3173
  p (boot) = bootstrap holm-corrected; p (wsr) = Wilcoxon signed-rank holm-corrected
  stars: * p<0.05, ** p<0.01, *** p<0.001

=========================================
 PER-MODEL SUMMARY: GOOGLE/GEMMA-3-4B-IT 
=========================================
=== Analysis Summary ===
Shape: BenchmarkShape(models=1, prompts=2, input_vars=1, evaluators=1)
Templates: 2 | Inputs: 50

--- Robustness ---
          mean  median       std        cv  iqr  cvar_10  p10  p25  p50  p75  p90
template                                                                         
3-shot    0.72     1.0  0.453557  0.629941  1.0      0.0  0.0  0.0  1.0  1.0  1.0
0-shot    0.78     1.0  0.418452  0.536477  0.0      0.0  0.0  1.0  1.0  1.0  1.0

--- Rank Probabilities ---
  Template                   P(Best)   E[Rank]
  3-shot                      22.4%     1.78
  0-shot                      77.6%     1.22

--- Mean Advantage (reference=grand_mean) ---
  axis: [-0.500, +0.500]  (· spread, ─ CI, ● mean, │ zero)  spread percentiles = (10, 90)
  Template                 Interval Plot                                 Mean    CI Low   CI High  Spread Lo  Spread Hi
  3-shot                   ················───●│─                     -0.030   -0.110   +0.020    -0.500    +0.000
  0-shot                                     ──│●──·················  +0.030   -0.040   +0.080    +0.000    +0.500

--- Pairwise Comparisons (lowest p-value first) ---
  axis: [-0.530, +0.530]  (· ±1σ, ─ CI, ● mean, │ zero)
  Pair                             Interval Plot                                 Mean    CI Low   CI High        σ   p (boot)   p (wsr)
  3-shot vs 0-shot                 ············──────●─│──·············       -0.0600   -0.2200   +0.0400   0.4699     0.3685    0.3657
  p (boot) = bootstrap holm-corrected; p (wsr) = Wilcoxon signed-rank holm-corrected
  stars: * p<0.05, ** p<0.01, *** p<0.001

================================================
 CROSS-MODEL RANKING (ALL MODEL/TEMPLATE PAIRS) 
================================================
  6 pairs ranked. Top 5 by P(Best):
    mistralai/ministral-8b-2512 / 0-shot      P(Best)=34.8%
    google/gemma-3-4b-it / 0-shot             P(Best)=27.4%
    gpt-4.1-nano / 0-shot                     P(Best)=24.0%
    google/gemma-3-4b-it / 3-shot             P(Best)=9.3%
    mistralai/ministral-8b-2512 / 3-shot      P(Best)=4.4%


"""