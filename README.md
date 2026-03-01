# promptstats

Utilities for statistically sane analyses for comparing prompt and LLM performance. 
Compute statistics and visualize results. 

You provide eval scores, prompts, inputs, evaluator names, and (optionally) models. 
These should be the results from a benchmark, such that all prompts were tested across all inputs
with all evaluators present (no missing data). We compute the rest. 

`promptstats` provides:
 - Plots and tests comparing prompt performance, with bootstrapped CIs and variance
 - Plots and tests comparing model performance across prompt variations
 - Constraints that guide you into performing best practices, like always considering prompt sensitivity when benchmarking model performance
 - A default "report" mode that outputs a PDF summarizing findings and diving into the details

The idea is that you provide `promptstats` with your data, and we run more statistically appropriate tests quantifying uncertainty and providing confidence bounds on claims. 

## Motivation

Most eval tools in the LLM evaluation space don't help users perform _any_ statistical tests, let alone showcase variances in performance between prompts or models. They instead present bar charts of average performance. Developers then glance at the bar chart and decide that "prompt/model A is better than B." But was it really? 

Relying purely on bar charts and averages can easily lead to erroneous conclusions—B might actually be more robust than A, or B performs well on an important subset of data, or there's not enough to data to conclude one way or the other. 

Why do people do evals this way? Well, they don't have the time, tools, or knowledge on how to do it better—frequently, they don't even know there's a better way. 

`prompt-stats` aims to rectify this with simple, powerful defaults—just throw us your data and we'll run the stats and plot the results for you. Upstream applications, like LLM observability platforms, could take `promptstats` results and plot them in their own front-ends. Prompt optimization tools could also use `promptstats` to decide, e.g., when to cull a candidate prompt and how to present results to users. 

### Quantify (un)certainty that one prompt is "better" than others

### Compare LLM performance with confidence bounds and statistical tests

## Comparing models while accounting for prompt sensitivity

A common failure mode in LLM benchmarking, both in academic papers and practitioner evaluations, is testing each model with a single prompt template and reporting the resulting scores as if they reflect stable model capabilities. In reality, model rankings can flip under semantically equivalent paraphrases of the same instruction. A benchmark result that says "Model A beats Model B" may be an artifact of prompt phrasing, not a meaningful capability difference.

This packages helps you answer:

- **Is the model ranking stable across prompt phrasings, or does it flip?**
- **Which model is most robust to prompt variation?** (practitioners care about this for production reliability)
- **Does Model A actually beat Model B, or only under a specific phrasing?** (academics need this to make defensible claims)
- **How many prompt variants do I need to trust a comparison?** (both audiences)

## Future

We aim to iteratively contribute to `promptstats`. Ideas for future features:
 - Help developers quantify the "variance" of provided prompt templates, and perhaps even factor this into the calculation in an intelligent way.