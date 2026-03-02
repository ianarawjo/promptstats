# promptstats

Utilities for statistically sane analyses for comparing prompt and LLM performance. Compute statistics and visualize the results.

`promptstats` helps you answer questions like:
 - Is Prompt A actually better than Prompt B, or just slightly luckier on this dataset?
 - Does Model A beat Model B, or only under a specific prompt phrasing?
 - How sensitive is model performance to prompt wording?
 - Are my performance differences large enough to be meaningful, or just noise?
 - How stable are scores across runs, evaluators, or inputs?

The idea is simple: you give `promptstats` your benchmark data, and it runs statistically appropriate analyses that quantify uncertainty and provide confidence bounds on your claims. Datasets can include eval scores, prompts, inputs, evaluator names, and (optionally) models. `promptstats` provides:
- Plots and tests comparing prompt performance, with bootstrapped CIs and variance
- Plots and tests comparing model performance across prompt variations
- Constraints that guide you into performing best practices, like always considering prompt sensitivity when benchmarking model performance

## Sample output

Running `pstats.analyze()` and then `pstats.print_analysis_summary(analysis)` prints a full statistical report to the terminal, including confidence interval line plots, pairwise comparisons between prompt templates, and per-input stability across runs (how stable the model is across multiple runs for the same input). Below is example excerpt from an analysis of a 4-template sentiment-classification benchmark (GPT-4.1-nano, 27 inputs, 3 runs, 3 evaluators):

![Example terminal output](docs/example-output.png)

From this output, we can see that Minimal and Instructive are the most promising candidates, but it is statistically unclear which is better. We also see that Chain-of-thought gives the least consistent outputs across multiple runs for the same inputs, compared to the other methods. 

You can also plot within notebook environments (although this feature is being actively built out over time). The `plot_mean_advantage` function produces a chart showing each template's mean score advantage over the grand mean, with dual uncertainty bands — the narrow dark band is the bootstrapped CI on the mean, and the wide light band is the 10th–90th percentile spread (template consistency):

![Mean advantage plot](docs/mean_advantage.png)

## Installation and Quick start CLI

```bash
pip install promptstats[all]
```

From the command line, `promptstats` can read a CSV or Excel file directly and print a statistical summary:

```bash
promptstats analyze results.csv
```

The input file should have columns `template`, `input`, and `score` (run and evaluator columns are optional). Run `promptstats analyze --help` for the full list of options and supported column aliases.

## Python API

`promptstats` main use case is as a Python API, which provides a similar entry point, the `analyze()` function. Simply pass your benchmark data in the correct format, and pass it to `analyze` to get a battery of results:

```python
import numpy as np
import promptstats as pstats

# scores shape: (n_templates, n_inputs, n_runs, n_evaluators)
# For a single evaluator and single run, shape is (N, M, 1, 1)
scores = np.array(your_scores).reshape(n_templates, n_inputs, 1, 1)

result = pstats.BenchmarkResult(
    scores=scores,
    template_labels=["Minimal", "Instructive", "Few-shot", "Chain-of-thought"],
    input_labels=[f"input_{i}" for i in range(n_inputs)],
)

analysis = pstats.analyze(result, reference="grand_mean", n_bootstrap=5_000)
pstats.print_analysis_summary(analysis)
```

To visualize mean advantage relative to the grand mean, with bootstrapped confidence intervals:

```python
fig = pstats.plot_mean_advantage(result, reference="grand_mean")
fig.savefig("advantage.png", dpi=150, bbox_inches="tight")
```

## Motivation

Most eval tools in the LLM evaluation space don't help users perform _any_ statistical tests, let alone showcase variances in performance between prompts or models. They instead present bar charts of average performance. Developers then glance at the bar chart and decide that "prompt/model A is better than B." But was it really?

Relying purely on bar charts and averages can very, very easily lead to erroneous conclusions—B might actually be more robust than A, or B performs well on an important subset of data, or there's not enough data to conclude one way or the other.

Why do people do evals this way? Well, they don't have the time, tools, or knowledge on how to do it better—frequently, they don't even know there's a better way.

`promptstats` aims to rectify this with simple, powerful defaults—just throw us your data and we'll run the stats and plot the results for you. Upstream applications, like LLM observability platforms, could take `promptstats` results and plot them in their own front-ends. Prompt optimization tools could also use `promptstats` to decide, e.g., when to cull a candidate prompt and how to present results to users.

## Examples

### Is one prompt "better" than others? Quantify uncertainty

When you have scores for multiple prompt templates across a set of inputs, `promptstats` computes bootstrapped confidence intervals and pairwise significance tests so you can see not just which prompt scored highest on average, but how certain you can be about that ranking. It plots these to the terminal so you can check at a glance:

![Comparing across prompts output](docs/compare-prompts-output.png)

### Comparing across models while accounting for prompt sensitivity

A common failure mode in LLM benchmarking, both in academic papers and practitioner evaluations, is testing each model with a single prompt template and reporting the resulting scores as if they reflect stable model capabilities. In reality, model rankings can flip under semantically equivalent paraphrases of the same instruction. A benchmark result that says "Model A beats Model B" may be an artifact of prompt phrasing, not a meaningful capability difference. 

Here, we can see the difference between OpenAI's `gpt-4.1-nano` and MistalAI's `ministral-8b-2512` on a small sentiment classification benchmark, quantified by bootstrapped confidence intervals:

![Comparing across models output](docs/compare-models-output.png)

In this run, multiple prompt template variations were considered, making this result more robust than trying a single prompt and calling it at day. 

### How stable is the performance across runs?

LLMs are stochastic at temperature>0. Will the performance stay similar, even upon multiple runs for the same inputs? `promptstats` offers a helpful "noise plot" which visualizes (in)stability across runs:

![Per-input noise across runs](docs/per-input-noise.png)

## Future

We aim to continue to contribute to `promptstats`. Ideas for future features:
- Mixed-effects models (LMMs and potentially GLMMs) for more complex data. Currently, `promptstats` only supports the case of one input per prompt template, rather than a grid search (cross product) of different prompt variations.
- A default "report" mode that outputs a PDF summarizing findings and diving into the details
- Help developers quantify the "semantic variance" of the provided prompt templates, and perhaps even factor this into the calculation in an intelligent way. This is important because the current implementation doesn't know about the diversity/representativity of the input dataset and prompts. 

## License

MIT — see [LICENSE](LICENSE).
