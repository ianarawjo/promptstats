import numpy as np
import pytest

import promptstats as ps


def test_analyze_single_model_recovers_means_and_best_prompt():
    # Use a fixed RNG seed so this test is deterministic and reproducible.
    rng = np.random.default_rng(123)
    n_inputs = 250

    # Single-model benchmark with three prompts.
    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Ground-truth parameters used to generate synthetic scores.
    # Prompt A should be best on average.
    target_means = np.array([8.2, 7.5, 6.8])
    target_scales = np.array([0.55, 0.75, 0.9])

    # Create one score per prompt/input from known Normal distributions.
    scores = np.vstack(
        [
            rng.normal(loc=target_means[i], scale=target_scales[i], size=n_inputs)
            for i in range(len(prompt_labels))
        ]
    )
    # Keep values in a plausible scoring range.
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    # Run the top-level analysis entrypoint.
    analysis = ps.analyze(
        result,
        n_bootstrap=1500,
        rng=np.random.default_rng(7),
    )

    assert isinstance(analysis, ps.AnalysisBundle)

    # Estimated means should be close to the generating means.
    # We use a tolerance because the data is sampled and finite.
    np.testing.assert_allclose(
        analysis.robustness.mean,
        target_means,
        atol=0.20,
    )
    np.testing.assert_allclose(
        analysis.robustness.std,
        target_scales,
        atol=0.12,
    )

    # The prompt with highest P(best) should match the true best prompt.
    best_prompt_idx = int(np.argmax(analysis.rank_dist.p_best))
    assert analysis.rank_dist.labels[best_prompt_idx] == "Prompt A"
    assert best_prompt_idx == int(np.argmax(target_means))

    # Mean-advantage values should align with known offsets vs grand mean.
    grand_mean_target = float(target_means.mean())
    expected_advantages = target_means - grand_mean_target
    np.testing.assert_allclose(
        analysis.mean_advantage.mean_advantages,
        expected_advantages,
        atol=0.20,
    )


def test_analyze_multimodel_recovers_best_model_and_best_pair():
    # Deterministic synthetic benchmark across two models and three prompts.
    rng = np.random.default_rng(2026)
    n_inputs = 220

    model_labels = ["Model 1", "Model 2"]
    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Ground-truth means by (model, prompt):
    # Model 2 dominates overall; Model 2 + Prompt B is the best pair.
    target_means = np.array(
        [
            [7.2, 7.6, 6.9],
            [8.1, 8.7, 8.0],
        ]
    )
    target_scales = np.array(
        [
            [0.70, 0.65, 0.75],
            [0.55, 0.50, 0.60],
        ]
    )

    # Build score tensor with shape (models, prompts, inputs).
    scores = np.empty((len(model_labels), len(prompt_labels), n_inputs), dtype=float)
    for model_idx in range(len(model_labels)):
        for prompt_idx in range(len(prompt_labels)):
            scores[model_idx, prompt_idx] = rng.normal(
                loc=target_means[model_idx, prompt_idx],
                scale=target_scales[model_idx, prompt_idx],
                size=n_inputs,
            )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=model_labels,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=1800,
        rng=np.random.default_rng(11),
    )

    # Multi-model inputs should route to a MultiModelBundle.
    assert isinstance(analysis, ps.MultiModelBundle)

    # Model-level winner should be Model 2.
    model_best_idx = int(np.argmax(analysis.model_level.rank_dist.p_best))
    assert analysis.model_level.rank_dist.labels[model_best_idx] == "Model 2"

    # Cross-model winner should be the intended best (model, prompt) pair.
    expected_best_pair = ("Model 2", "Prompt B")
    assert analysis.best_pair == expected_best_pair

    # Recovered model means should match the generation-time ground truth.
    expected_model_means = target_means.mean(axis=1)
    # For each model, model-level scores are per-input means across prompts.
    # With independent prompt noise, std(mean of 3 prompts) = sqrt(sum(s_i^2) / 3^2).
    expected_model_stds = np.sqrt((target_scales**2).sum(axis=1) / (len(prompt_labels) ** 2))
    np.testing.assert_allclose(
        analysis.model_level.robustness.mean,
        expected_model_means,
        atol=0.22,
    )
    np.testing.assert_allclose(
        analysis.model_level.robustness.std,
        expected_model_stds,
        atol=0.12,
    )


def test_analyze_multimodel_single_prompt_warns_and_runs():
    rng = np.random.default_rng(2027)
    n_models = 3
    n_inputs = 180

    model_labels = ["Model 1", "Model 2", "Model 3"]
    prompt_labels = ["Prompt A"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    target_means = np.array([7.4, 8.2, 7.8])
    target_scales = np.array([0.7, 0.6, 0.65])

    scores = np.empty((n_models, 1, n_inputs), dtype=float)
    for model_idx in range(n_models):
        scores[model_idx, 0] = rng.normal(
            loc=target_means[model_idx],
            scale=target_scales[model_idx],
            size=n_inputs,
        )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=model_labels,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    with pytest.warns(UserWarning, match="Single-prompt multi-model analysis is supported"):
        analysis = ps.analyze(
            result,
            n_bootstrap=1200,
            rng=np.random.default_rng(13),
        )

    assert isinstance(analysis, ps.MultiModelBundle)

    # Best model should still be recovered from model-level ranking.
    model_best_idx = int(np.argmax(analysis.model_level.rank_dist.p_best))
    assert analysis.model_level.rank_dist.labels[model_best_idx] == "Model 2"

    # With one prompt, the best pair should be (best model, that prompt).
    assert analysis.best_pair == ("Model 2", "Prompt A")


def test_analyze_seeded_multirun_populates_seed_variance_and_ordering():
    # Seeded benchmark: (templates, inputs, runs) with R >= 3
    # so analyze() should populate seed_variance.
    rng = np.random.default_rng(314)
    n_inputs = 140
    n_runs = 5

    prompt_labels = ["Stable Winner", "Noisy Second", "Medium Third"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Base signal by input (shared across prompts) + prompt-specific mean offsets.
    base_by_input = rng.normal(loc=0.0, scale=0.45, size=n_inputs)
    target_means = np.array([8.20, 7.95, 7.65])

    # Intentionally different run-level noise to test seed_variance ordering.
    run_noise_scales = np.array([0.08, 0.62, 0.24])

    # Build score tensor with shared per-input signal + per-prompt run noise.
    scores = np.empty((len(prompt_labels), n_inputs, n_runs), dtype=float)
    for prompt_idx in range(len(prompt_labels)):
        for run_idx in range(n_runs):
            run_noise = rng.normal(loc=0.0, scale=run_noise_scales[prompt_idx], size=n_inputs)
            scores[prompt_idx, :, run_idx] = (
                target_means[prompt_idx] + base_by_input + run_noise
            )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=1600,
        rng=np.random.default_rng(9),
    )

    # R >= 3 enables seed-variance decomposition in the output bundle.
    assert isinstance(analysis, ps.AnalysisBundle)
    assert analysis.seed_variance is not None

    seed_var = analysis.seed_variance
    assert seed_var.n_runs == n_runs

    # Winner should still be the highest-mean prompt.
    best_idx = int(np.argmax(analysis.rank_dist.p_best))
    assert analysis.rank_dist.labels[best_idx] == "Stable Winner"

    # Instability ranking should mirror the synthetic run noise scales.
    most_stable_idx = int(np.argmin(seed_var.instability))
    most_noisy_idx = int(np.argmax(seed_var.instability))
    assert seed_var.labels[most_stable_idx] == "Stable Winner"
    assert seed_var.labels[most_noisy_idx] == "Noisy Second"

    # Recovered means should be close to their construction targets.
    np.testing.assert_allclose(
        analysis.robustness.mean,
        target_means,
        atol=0.20,
    )
    # Robustness metrics use per-input cell means (averaged over runs),
    # so expected std combines input signal variance and reduced run noise.
    expected_stds = np.sqrt(0.45**2 + (run_noise_scales**2) / n_runs)
    np.testing.assert_allclose(
        analysis.robustness.std,
        expected_stds,
        atol=0.12,
    )


def test_analyze_per_evaluator_returns_expected_winners():
    # Evaluator-aware benchmark: (templates, inputs, runs, evaluators).
    # Each evaluator is given a different "true" best prompt.
    rng = np.random.default_rng(808)
    n_inputs = 160
    n_runs = 4

    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    evaluator_names = ["quality", "brevity"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # evaluator 0 ('quality'): Prompt A wins
    # evaluator 1 ('brevity'): Prompt B wins
    target_means_by_eval = np.array(
        [
            [8.6, 7.8, 7.1],
            [7.4, 8.5, 7.2],
        ]
    )
    run_noise_scales_by_eval = np.array(
        [
            [0.25, 0.28, 0.30],
            [0.24, 0.22, 0.31],
        ]
    )

    # Build a 4-D score tensor and keep the evaluator axis explicit.
    scores = np.empty(
        (len(prompt_labels), n_inputs, n_runs, len(evaluator_names)),
        dtype=float,
    )

    # For each evaluator, create a distinct synthetic objective landscape.
    for evaluator_idx in range(len(evaluator_names)):
        base_by_input = rng.normal(loc=0.0, scale=0.35, size=n_inputs)
        for prompt_idx in range(len(prompt_labels)):
            for run_idx in range(n_runs):
                run_noise = rng.normal(
                    loc=0.0,
                    scale=run_noise_scales_by_eval[evaluator_idx, prompt_idx],
                    size=n_inputs,
                )
                scores[prompt_idx, :, run_idx, evaluator_idx] = (
                    target_means_by_eval[evaluator_idx, prompt_idx]
                    + base_by_input
                    + run_noise
                )

    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
        evaluator_names=evaluator_names,
    )

    analysis = ps.analyze(
        result,
        evaluator_mode="per_evaluator",
        n_bootstrap=1400,
        rng=np.random.default_rng(99),
    )

    # per_evaluator mode should return one AnalysisBundle per evaluator.
    assert isinstance(analysis, dict)
    assert set(analysis.keys()) == set(evaluator_names)

    for evaluator_name in evaluator_names:
        assert isinstance(analysis[evaluator_name], ps.AnalysisBundle)
        # Runs axis is preserved during evaluator slicing.
        assert analysis[evaluator_name].seed_variance is not None

    # Winners should match each evaluator's synthetic ground truth.
    quality_best_idx = int(np.argmax(analysis["quality"].rank_dist.p_best))
    brevity_best_idx = int(np.argmax(analysis["brevity"].rank_dist.p_best))

    assert analysis["quality"].rank_dist.labels[quality_best_idx] == "Prompt A"
    assert analysis["brevity"].rank_dist.labels[brevity_best_idx] == "Prompt B"

    # Evaluator-specific means should recover generation-time means.
    np.testing.assert_allclose(
        analysis["quality"].robustness.mean,
        target_means_by_eval[0],
        atol=0.20,
    )
    np.testing.assert_allclose(
        analysis["brevity"].robustness.mean,
        target_means_by_eval[1],
        atol=0.20,
    )

    # For each evaluator/prompt, cell-mean std is
    # sqrt(base_input_std^2 + run_noise_std^2 / n_runs).
    expected_stds_by_eval = np.sqrt(0.35**2 + (run_noise_scales_by_eval**2) / n_runs)
    np.testing.assert_allclose(
        analysis["quality"].robustness.std,
        expected_stds_by_eval[0],
        atol=0.12,
    )
    np.testing.assert_allclose(
        analysis["brevity"].robustness.std,
        expected_stds_by_eval[1],
        atol=0.12,
    )


@pytest.mark.parametrize("n_inputs", [4, 10, 20, 60, 260])
def test_analyze_bca_and_bootstrap_are_consistent(n_inputs):
    # Same synthetic benchmark analyzed with two CI methods.
    # Goal: ensure both methods run and BCa is not pathologically different.
    rng = np.random.default_rng(4242)

    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    target_means = np.array([8.1, 7.45, 6.9])
    target_scales = np.array([0.65, 0.8, 0.85])

    scores = np.vstack(
        [
            rng.normal(loc=target_means[i], scale=target_scales[i], size=n_inputs)
            for i in range(len(prompt_labels))
        ]
    )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis_bootstrap = ps.analyze(
        result,
        method="bootstrap",
        n_bootstrap=1700,
        rng=np.random.default_rng(101),
    )
    analysis_bca = ps.analyze(
        result,
        method="bca",
        n_bootstrap=1700,
        rng=np.random.default_rng(101),
    )

    assert isinstance(analysis_bootstrap, ps.AnalysisBundle)
    assert isinstance(analysis_bca, ps.AnalysisBundle)

    # Point estimates should agree very closely regardless of CI method.
    np.testing.assert_allclose(
        analysis_bca.robustness.mean,
        analysis_bootstrap.robustness.mean,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        analysis_bca.mean_advantage.mean_advantages,
        analysis_bootstrap.mean_advantage.mean_advantages,
        atol=1e-12,
    )

    # Best prompt should be the same across methods.
    best_bootstrap = int(np.argmax(analysis_bootstrap.rank_dist.p_best))
    best_bca = int(np.argmax(analysis_bca.rank_dist.p_best))
    assert best_bca == best_bootstrap

    # With enough samples, both methods should also recover the known true winner.
    if n_inputs >= 60:
        assert analysis_bootstrap.rank_dist.labels[best_bootstrap] == "Prompt A"
        assert analysis_bca.rank_dist.labels[best_bca] == "Prompt A"

    # BCa CI widths should be in a similar ballpark to percentile bootstrap.
    width_bootstrap = (
        analysis_bootstrap.mean_advantage.bootstrap_ci_high
        - analysis_bootstrap.mean_advantage.bootstrap_ci_low
    )
    width_bca = (
        analysis_bca.mean_advantage.bootstrap_ci_high
        - analysis_bca.mean_advantage.bootstrap_ci_low
    )

    # Avoid division-by-zero for degenerate cases (should be unlikely here).
    safe_width_bootstrap = np.maximum(width_bootstrap, 1e-9)
    width_ratio = width_bca / safe_width_bootstrap
    # For tiny n_inputs, BCa can be less stable; allow a wider sanity band.
    if n_inputs <= 20:
        assert np.all(width_ratio > 0.15)
        assert np.all(width_ratio < 6.0)
    else:
        assert np.all(width_ratio > 0.35)
        assert np.all(width_ratio < 2.5)

    # CI centers should also stay reasonably close.
    center_bootstrap = (
        analysis_bootstrap.mean_advantage.bootstrap_ci_high
        + analysis_bootstrap.mean_advantage.bootstrap_ci_low
    ) / 2
    center_bca = (
        analysis_bca.mean_advantage.bootstrap_ci_high
        + analysis_bca.mean_advantage.bootstrap_ci_low
    ) / 2
    center_atol = 0.18 if n_inputs <= 20 else 0.08
    np.testing.assert_allclose(center_bca, center_bootstrap, atol=center_atol)


def test_analyze_pathological_many_prompts_recovers_best_prompt():
    # Stress case: many prompt templates in a single-model benchmark.
    rng = np.random.default_rng(909)
    n_templates = 81
    n_inputs = 160

    prompt_labels = [f"Prompt {i:02d}" for i in range(n_templates)]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Prompt 0 is the true winner with a clear margin; others decay in quality.
    target_means = 8.1 - 0.05 * np.arange(n_templates)
    target_means[0] += 0.30
    target_scales = 0.48 + 0.006 * np.arange(n_templates)

    scores = np.empty((n_templates, n_inputs), dtype=float)
    for template_idx in range(n_templates):
        scores[template_idx] = rng.normal(
            loc=target_means[template_idx],
            scale=target_scales[template_idx],
            size=n_inputs,
        )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=300,
        rng=np.random.default_rng(17),
    )

    assert isinstance(analysis, ps.AnalysisBundle)

    best_idx = int(np.argmax(analysis.rank_dist.p_best))
    assert analysis.rank_dist.labels[best_idx] == "Prompt 00"
    assert best_idx == int(np.argmax(target_means))

    # Even with many templates, the top-ranked expected rank should be near 1.
    top_expected_rank = float(analysis.rank_dist.expected_ranks[best_idx])
    assert top_expected_rank < 2.5


def test_analyze_pathological_many_models_recovers_best_model_and_pair():
    # Stress case: many models and many prompts.
    rng = np.random.default_rng(910)
    n_models = 12
    n_prompts = 20
    n_inputs = 40

    model_labels = [f"Model {i:02d}" for i in range(n_models)]
    prompt_labels = [f"Prompt {i:02d}" for i in range(n_prompts)]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Model 00 has the strongest base quality; Prompt 00 is globally strongest.
    model_offsets = 0.9 - 0.10 * np.arange(n_models)
    prompt_offsets = 0.35 - 0.02 * np.arange(n_prompts)
    prompt_offsets[0] += 0.30

    target_means = 7.2 + model_offsets[:, np.newaxis] + prompt_offsets[np.newaxis, :]
    target_scales = 0.55 + 0.02 * np.arange(n_models)[:, np.newaxis]

    scores = np.empty((n_models, n_prompts, n_inputs), dtype=float)
    for model_idx in range(n_models):
        for prompt_idx in range(n_prompts):
            scores[model_idx, prompt_idx] = rng.normal(
                loc=target_means[model_idx, prompt_idx],
                scale=target_scales[model_idx, 0],
                size=n_inputs,
            )
    scores = np.clip(scores, 0.0, 10.0)

    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=model_labels,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=8,
        rng=np.random.default_rng(21),
    )

    assert isinstance(analysis, ps.MultiModelBundle)

    model_best_idx = int(np.argmax(analysis.model_level.rank_dist.p_best))
    assert analysis.model_level.rank_dist.labels[model_best_idx] == "Model 00"

    expected_best_pair = ("Model 00", "Prompt 00")
    assert analysis.best_pair == expected_best_pair


def test_analyze_pathological_many_runs_detects_seed_noise_and_winner():
    # Stress case: many runs per cell with distinct run-noise profiles.
    rng = np.random.default_rng(911)
    n_inputs = 120
    n_runs = 28

    prompt_labels = ["Winner Stable", "RunnerUp Noisy", "Middle"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    target_means = np.array([8.25, 8.05, 7.85])
    run_noise_scales = np.array([0.07, 0.72, 0.22])

    base_by_input = rng.normal(loc=0.0, scale=0.42, size=n_inputs)
    scores = np.empty((len(prompt_labels), n_inputs, n_runs), dtype=float)

    for template_idx in range(len(prompt_labels)):
        for run_idx in range(n_runs):
            run_noise = rng.normal(
                loc=0.0,
                scale=run_noise_scales[template_idx],
                size=n_inputs,
            )
            scores[template_idx, :, run_idx] = (
                target_means[template_idx] + base_by_input + run_noise
            )

    scores = np.clip(scores, 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=260,
        rng=np.random.default_rng(31),
    )

    assert isinstance(analysis, ps.AnalysisBundle)
    assert analysis.seed_variance is not None

    best_idx = int(np.argmax(analysis.rank_dist.p_best))
    assert analysis.rank_dist.labels[best_idx] == "Winner Stable"

    seed_var = analysis.seed_variance
    assert seed_var.n_runs == n_runs

    most_stable_idx = int(np.argmin(seed_var.instability))
    most_noisy_idx = int(np.argmax(seed_var.instability))
    assert seed_var.labels[most_stable_idx] == "Winner Stable"
    assert seed_var.labels[most_noisy_idx] == "RunnerUp Noisy"
