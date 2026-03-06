import numpy as np
import pytest

import promptstats as ps
from promptstats.core import router as router_core
from promptstats.core.paired import PairedDiffResult, PairwiseMatrix


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
        analysis.point_advantage.point_advantages,
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


def test_print_multimodel_summary_includes_collapsed_template_level_section(capsys):
    model_labels = ["Model 1", "Model 2"]
    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    input_labels = ["i1", "i2", "i3"]

    # Prompt C is worst regardless of model, so it should be called out.
    scores = np.array(
        [
            [
                [0.90, 0.88, 0.91],
                [0.80, 0.79, 0.78],
                [0.52, 0.50, 0.49],
            ],
            [
                [0.89, 0.87, 0.90],
                [0.77, 0.76, 0.75],
                [0.45, 0.47, 0.46],
            ],
        ],
        dtype=float,
    )

    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=model_labels,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=300,
        rng=np.random.default_rng(123),
        template_model_collapse="mean"
    )

    ps.print_analysis_summary(analysis, top_pairwise=3)
    out = capsys.readouterr().out

    assert "CROSS-MODEL PER-TEMPLATE COMPARISON " in out
    assert "Best-performing prompt across models (by mean score)" in out
    assert "'Prompt A'" in out


def test_print_summary_includes_critical_difference_groups(capsys):
    # Three identical templates => Nemenyi should mark all as indistinguishable.
    scores = np.array(
        [
            [0.7, 0.8, 0.9, 0.8, 0.7, 0.9],
            [0.7, 0.8, 0.9, 0.8, 0.7, 0.9],
            [0.7, 0.8, 0.9, 0.8, 0.7, 0.9],
            [0.1, 0.2, 0.3, 0.2, 0.1, 0.3],
        ],
        dtype=float,
    )
    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=["Prompt A", "Prompt B", "Prompt C", "Prompt D"],
        input_labels=[f"i{i}" for i in range(scores.shape[1])],
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=300,
        rng=np.random.default_rng(42),
    )

    ps.print_analysis_summary(analysis, top_pairwise=3)
    out = capsys.readouterr().out

    assert "Statistically indistinguishable" in out
    assert "[Prompt A ─ Prompt B ─ Prompt C]" in out
    assert "#" in out
    assert "Prompt D" in out


def test_critical_difference_groups_has_two_separate_rank_bands():
    labels = ["Prompt A", "Prompt B", "Prompt C", "Prompt D", "Prompt E"]
    p_boot = {
        ("Prompt A", "Prompt B"): 0.42,
        ("Prompt A", "Prompt C"): 0.01,
        ("Prompt B", "Prompt C"): 0.02,
        ("Prompt C", "Prompt D"): 0.01,
        ("Prompt D", "Prompt E"): 0.31,
    }
    p_wsr = {
        ("Prompt A", "Prompt B"): 0.35,
        ("Prompt A", "Prompt C"): 0.02,
        ("Prompt B", "Prompt C"): 0.01,
        ("Prompt C", "Prompt D"): 0.03,
        ("Prompt D", "Prompt E"): 0.40,
    }

    results: dict[tuple[str, str], PairedDiffResult] = {}
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            boot_p = p_boot.get((label_a, label_b), 0.001)
            wsr_p = p_wsr.get((label_a, label_b), 0.001)
            results[(label_a, label_b)] = PairedDiffResult(
                template_a=label_a,
                template_b=label_b,
                point_diff=0.0,
                std_diff=0.0,
                ci_low=0.0,
                ci_high=0.0,
                p_value=float(boot_p),
                test_method="bootstrap",
                n_inputs=30,
                per_input_diffs=np.zeros(30, dtype=float),
                n_runs=1,
                statistic="mean",
                wilcoxon_p=float(wsr_p),
            )

    pairwise = PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method="holm",
        friedman=None,
    )

    groups = router_core._critical_difference_groups(
        pairwise,
        labels_sorted=labels,
        alpha=0.05,
        p_source="both",
    )

    assert groups == [
        ["Prompt A", "Prompt B"],
        ["Prompt D", "Prompt E"],
    ]


def test_critical_difference_groups_can_have_overlapping_rank_bands():
    labels = ["Prompt A", "Prompt B", "Prompt C", "Prompt D", "Prompt E"]
    p_boot = {
        ("Prompt A", "Prompt B"): 0.70,
        ("Prompt A", "Prompt C"): 0.21,
        ("Prompt B", "Prompt C"): 0.56,
        ("Prompt A", "Prompt D"): 0.01,
        ("Prompt B", "Prompt D"): 0.01,
        ("Prompt C", "Prompt D"): 0.43,
        ("Prompt C", "Prompt E"): 0.25,
        ("Prompt D", "Prompt E"): 0.52,
    }
    p_wsr = {
        ("Prompt A", "Prompt B"): 0.61,
        ("Prompt A", "Prompt C"): 0.18,
        ("Prompt B", "Prompt C"): 0.67,
        ("Prompt A", "Prompt D"): 0.02,
        ("Prompt B", "Prompt D"): 0.01,
        ("Prompt C", "Prompt D"): 0.40,
        ("Prompt C", "Prompt E"): 0.20,
        ("Prompt D", "Prompt E"): 0.44,
    }

    results: dict[tuple[str, str], PairedDiffResult] = {}
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            boot_p = p_boot.get((label_a, label_b), 0.001)
            wsr_p = p_wsr.get((label_a, label_b), 0.001)
            results[(label_a, label_b)] = PairedDiffResult(
                template_a=label_a,
                template_b=label_b,
                point_diff=0.0,
                std_diff=0.0,
                ci_low=0.0,
                ci_high=0.0,
                p_value=float(boot_p),
                test_method="bootstrap",
                n_inputs=30,
                per_input_diffs=np.zeros(30, dtype=float),
                n_runs=1,
                statistic="mean",
                wilcoxon_p=float(wsr_p),
            )

    pairwise = PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method="holm",
        friedman=None,
    )

    groups = router_core._critical_difference_groups(
        pairwise,
        labels_sorted=labels,
        alpha=0.05,
        p_source="both",
    )

    assert groups == [
        ["Prompt A", "Prompt B", "Prompt C"],
        ["Prompt C", "Prompt D", "Prompt E"],
    ]


def test_single_clear_winner_label_detects_unique_statistical_winner():
    labels = ["Prompt A", "Prompt B", "Prompt C"]
    results: dict[tuple[str, str], PairedDiffResult] = {
        ("Prompt A", "Prompt B"): PairedDiffResult(
            template_a="Prompt A",
            template_b="Prompt B",
            point_diff=0.30,
            std_diff=0.05,
            ci_low=0.20,
            ci_high=0.40,
            p_value=0.001,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.002,
        ),
        ("Prompt A", "Prompt C"): PairedDiffResult(
            template_a="Prompt A",
            template_b="Prompt C",
            point_diff=0.28,
            std_diff=0.06,
            ci_low=0.16,
            ci_high=0.40,
            p_value=0.003,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.004,
        ),
        ("Prompt B", "Prompt C"): PairedDiffResult(
            template_a="Prompt B",
            template_b="Prompt C",
            point_diff=0.01,
            std_diff=0.08,
            ci_low=-0.10,
            ci_high=0.12,
            p_value=0.42,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.50,
        ),
    }

    pairwise = PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method="holm",
        friedman=None,
    )

    winner = router_core._single_clear_winner_label(
        pairwise,
        labels_sorted=labels,
        alpha=0.05,
        p_source="bootstrap",
    )
    assert winner == "Prompt A"


def test_print_critical_difference_groups_includes_clear_winner_line(capsys):
    labels = ["Prompt A", "Prompt B", "Prompt C"]
    results: dict[tuple[str, str], PairedDiffResult] = {
        ("Prompt A", "Prompt B"): PairedDiffResult(
            template_a="Prompt A",
            template_b="Prompt B",
            point_diff=0.30,
            std_diff=0.05,
            ci_low=0.20,
            ci_high=0.40,
            p_value=0.001,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.002,
        ),
        ("Prompt A", "Prompt C"): PairedDiffResult(
            template_a="Prompt A",
            template_b="Prompt C",
            point_diff=0.28,
            std_diff=0.06,
            ci_low=0.16,
            ci_high=0.40,
            p_value=0.003,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.004,
        ),
        ("Prompt B", "Prompt C"): PairedDiffResult(
            template_a="Prompt B",
            template_b="Prompt C",
            point_diff=0.01,
            std_diff=0.08,
            ci_low=-0.10,
            ci_high=0.12,
            p_value=0.42,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.zeros(50, dtype=float),
            n_runs=1,
            statistic="mean",
            wilcoxon_p=0.50,
        ),
    }

    pairwise = PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method="holm",
        friedman=None,
    )

    router_core._print_critical_difference_groups(
        pairwise,
        labels_sorted=labels,
        alpha=0.05,
        p_source="bootstrap",
    )
    out = capsys.readouterr().out

    assert "Statistically distinguishable, clear winner" in out
    assert "'Prompt A'" in out
    assert "Statistically indistinguishable rank bands" in out


def test_analyze_multimodel_template_collapse_as_runs_preserves_model_variance():
    model_labels = ["m1", "m2", "m3"]
    prompt_labels = ["Prompt A", "Prompt B"]
    input_labels = ["i1", "i2"]

    # Shape: (models, templates, inputs) with model-to-model spread.
    scores = np.array(
        [
            [[0.9, 0.8], [0.5, 0.4]],
            [[0.7, 0.6], [0.55, 0.45]],
            [[0.8, 0.7], [0.6, 0.5]],
        ],
        dtype=float,
    )
    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=model_labels,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=300,
        rng=np.random.default_rng(99),
        template_model_collapse="as_runs",
    )

    # Model axis is retained as run replicates in template-level benchmark.
    assert analysis.template_level.benchmark.n_runs == 3
    assert analysis.template_level.benchmark.scores.shape == (2, 2, 3)

    # Mean still matches direct average across models.
    expected = scores.mean(axis=0).mean(axis=1)
    np.testing.assert_allclose(analysis.template_level.robustness.mean, expected, atol=1e-12)

    # Effective runs >= 3 should expose seed variance decomposition.
    assert analysis.template_level.seed_variance is not None


def test_analyze_rejects_unknown_template_model_collapse():
    scores = np.array(
        [
            [[0.9, 0.8], [0.7, 0.6]],
            [[0.8, 0.7], [0.6, 0.5]],
        ],
        dtype=float,
    )
    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=["m1", "m2"],
        template_labels=["Prompt A", "Prompt B"],
        input_labels=["i1", "i2"],
    )

    with pytest.raises(ValueError, match="Unknown template_model_collapse"):
        ps.analyze(result, template_model_collapse="median")


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


def test_analyze_multimodel_per_evaluator_returns_mapping():
    rng = np.random.default_rng(99)
    n_models = 2
    n_prompts = 2
    n_inputs = 120
    n_runs = 3
    n_evaluators = 2

    scores = rng.normal(
        loc=0.7,
        scale=0.2,
        size=(n_models, n_prompts, n_inputs, n_runs, n_evaluators),
    )
    scores = np.clip(scores, 0.0, 1.0)

    result = ps.MultiModelBenchmark(
        scores=scores,
        model_labels=["Model A", "Model B"],
        template_labels=["Prompt A", "Prompt B"],
        input_labels=[f"item_{i:03d}" for i in range(n_inputs)],
        evaluator_names=["accuracy", "format"],
    )

    analysis = ps.analyze(
        result,
        evaluator_mode="per_evaluator",
        n_bootstrap=400,
        rng=np.random.default_rng(5),
    )

    assert isinstance(analysis, dict)
    assert set(analysis.keys()) == {"accuracy", "format"}
    for bundle in analysis.values():
        assert isinstance(bundle, ps.MultiModelBundle)


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
        analysis_bca.point_advantage.point_advantages,
        analysis_bootstrap.point_advantage.point_advantages,
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
        analysis_bootstrap.point_advantage.bootstrap_ci_high
        - analysis_bootstrap.point_advantage.bootstrap_ci_low
    )
    width_bca = (
        analysis_bca.point_advantage.bootstrap_ci_high
        - analysis_bca.point_advantage.bootstrap_ci_low
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
        analysis_bootstrap.point_advantage.bootstrap_ci_high
        + analysis_bootstrap.point_advantage.bootstrap_ci_low
    ) / 2
    center_bca = (
        analysis_bca.point_advantage.bootstrap_ci_high
        + analysis_bca.point_advantage.bootstrap_ci_low
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


# ---------------------------------------------------------------------------
# statistic='median' correctness and divergence from mean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bootstrap", "bca"])
def test_analyze_statistic_median_propagates_and_is_correct(method):
    """statistic='median' is labelled on all result objects and produces point
    estimates that match manual np.median computation exactly.

    Both the bootstrap and BCa CI paths are exercised via parametrize.
    """
    rng = np.random.default_rng(555)
    n_inputs = 80

    prompt_labels = ["Prompt A", "Prompt B", "Prompt C"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Clipped-normal scores with a clear ordering: A > B > C.
    target_medians = [7.5, 6.8, 6.1]
    scores = np.vstack(
        [
            np.clip(rng.normal(loc=m, scale=0.9, size=n_inputs), 0.0, 10.0)
            for m in target_medians
        ]
    )

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        statistic="median",
        method=method,
        n_bootstrap=1500,
        rng=np.random.default_rng(42),
    )

    assert isinstance(analysis, ps.AnalysisBundle)

    # statistic label must propagate to all result objects.
    assert analysis.point_advantage.statistic == "median"
    for pair_result in analysis.pairwise.results.values():
        assert pair_result.statistic == "median"

    # Point advantages must match manual median computation exactly.
    # Grand reference is always the per-input *mean* across templates (not
    # median) to avoid degeneracy when the template count is odd.
    grand_mean_per_input = scores.mean(axis=0)
    advantages = scores - grand_mean_per_input[np.newaxis, :]
    expected_point_advantages = np.median(advantages, axis=1)
    np.testing.assert_allclose(
        analysis.point_advantage.point_advantages,
        expected_point_advantages,
        atol=1e-10,
    )

    # Pairwise: point_diff for A vs B must equal median of per-input diffs.
    ab_result = analysis.pairwise.get("Prompt A", "Prompt B")
    expected_ab_diff = float(np.median(scores[0] - scores[1]))
    np.testing.assert_allclose(ab_result.point_diff, expected_ab_diff, atol=1e-10)

    # Confidence intervals must be finite and properly ordered.
    assert np.all(np.isfinite(analysis.point_advantage.bootstrap_ci_low))
    assert np.all(np.isfinite(analysis.point_advantage.bootstrap_ci_high))
    assert np.all(
        analysis.point_advantage.bootstrap_ci_low
        <= analysis.point_advantage.bootstrap_ci_high
    )

    # The prompt with the highest median advantage should be Prompt A.
    best_idx = int(np.argmax(analysis.point_advantage.point_advantages))
    assert prompt_labels[best_idx] == "Prompt A"


def test_analyze_median_and_mean_diverge_on_heavy_tailed_data():
    """On LLM-style data with catastrophic-failure outliers, median and mean
    give materially different—and opposite—conclusions about which prompt wins.

    Design
    ------
    Prompt A  90% of inputs score ~6.5 (good output), 10% score ~0.0 (crash).
              mean ≈ 5.85,  median ≈ 6.5

    Prompt B  All inputs score ~6.0 (stable, never crashes).
              mean ≈ 6.0,   median ≈ 6.0

    Per-input differences  A − B:
      ~90 % of diffs ≈  +0.5  (A is better on most inputs)
      ~10 % of diffs ≈  −6.0  (catastrophic failures)

    Mean statistic   The −6 outliers drag the average down → point_diff < 0 → B wins.
    Median statistic The +0.5 majority dominates → point_diff ≈ +0.5 → A wins.

    This pattern — occasional LLM failures that are invisible to the median but
    dominate the mean — is the primary motivation for preferring median in
    prompt benchmarking.
    """
    rng = np.random.default_rng(2024)
    n_inputs = 300

    prompt_labels = ["Prompt A", "Prompt B"]
    input_labels = [f"item_{i:03d}" for i in range(n_inputs)]

    # Prompt A: occasional catastrophic failures pull the mean far below median.
    is_good = rng.random(n_inputs) < 0.90
    scores_a = np.where(
        is_good,
        rng.normal(loc=6.5, scale=0.25, size=n_inputs),
        rng.normal(loc=0.0, scale=0.25, size=n_inputs),
    )
    # Prompt B: stable performance, never crashes.
    scores_b = rng.normal(loc=6.0, scale=0.25, size=n_inputs)

    scores = np.clip(np.vstack([scores_a, scores_b]), 0.0, 10.0)

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=prompt_labels,
        input_labels=input_labels,
    )

    analysis_mean = ps.analyze(
        result,
        statistic="mean",
        method="bootstrap",
        n_bootstrap=2000,
        rng=np.random.default_rng(7),
    )
    analysis_median = ps.analyze(
        result,
        statistic="median",
        method="bootstrap",
        n_bootstrap=2000,
        rng=np.random.default_rng(7),
    )

    # Statistic labels round-trip correctly.
    assert analysis_mean.point_advantage.statistic == "mean"
    assert analysis_median.point_advantage.statistic == "median"

    ab_mean = analysis_mean.pairwise.get("Prompt A", "Prompt B")
    ab_median = analysis_median.pairwise.get("Prompt A", "Prompt B")

    # Mean says B is better: A's failures drag its mean below B's.
    assert ab_mean.point_diff < 0.0, (
        f"Mean: expected A−B point_diff < 0 (B wins by mean), "
        f"got {ab_mean.point_diff:.3f}"
    )
    # Median says A is better: 90 % of per-input diffs are ≈ +0.5.
    assert ab_median.point_diff > 0.3, (
        f"Median: expected A−B point_diff > 0.3 (A wins by median), "
        f"got {ab_median.point_diff:.3f}"
    )

    # The two statistics must disagree substantially on A's point advantage.
    adv_a_mean = float(analysis_mean.point_advantage.point_advantages[0])
    adv_a_median = float(analysis_median.point_advantage.point_advantages[0])
    # Advantage values are computed vs the grand-mean reference, which halves
    # the raw per-input diff when N=2, so the threshold is correspondingly smaller.
    assert abs(adv_a_median - adv_a_mean) > 0.20, (
        f"Expected mean/median advantage of A to differ by > 0.20; "
        f"got mean_adv={adv_a_mean:.3f}, median_adv={adv_a_median:.3f}"
    )

    # Median 95% CI for A-vs-B should be entirely positive: with 300 inputs,
    # the bootstrap median will draw ≥ 150 positive diffs in every resample
    # (only 10 % of diffs are negative), so the lower bound stays > 0.
    assert ab_median.ci_low > 0.0, (
        f"Median 95% CI lower bound should be positive (A > B is clear), "
        f"got ci_low={ab_median.ci_low:.3f}"
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank outputs
# ---------------------------------------------------------------------------

def test_pairwise_wilcoxon_matches_scipy_and_is_symmetric_on_flip():
    from scipy.stats import wilcoxon

    scores = np.array(
        [
            [1.2, 2.0, 3.1, 2.8, 4.2, 5.0],
            [1.0, 1.6, 2.9, 3.1, 3.8, 4.7],
        ],
        dtype=float,
    )
    diffs = scores[0] - scores[1]

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=["A", "B"],
        input_labels=[f"item_{i}" for i in range(scores.shape[1])],
    )
    analysis = ps.analyze(
        result,
        method="bootstrap",
        correction="none",
        n_bootstrap=300,
        rng=np.random.default_rng(101),
    )

    pair_ab = analysis.pairwise.get("A", "B")
    pair_ba = analysis.pairwise.get("B", "A")
    expected_p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)

    np.testing.assert_allclose(pair_ab.wilcoxon_p, expected_p, atol=1e-12)
    np.testing.assert_allclose(pair_ba.wilcoxon_p, expected_p, atol=1e-12)


def test_pairwise_wilcoxon_is_none_when_all_differences_are_zero():
    scores = np.array(
        [
            [2.1, 3.4, 1.8, 4.0, 2.7, 3.3],
            [2.1, 3.4, 1.8, 4.0, 2.7, 3.3],
        ],
        dtype=float,
    )

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=["A", "B"],
        input_labels=[f"item_{i}" for i in range(scores.shape[1])],
    )
    analysis = ps.analyze(
        result,
        method="bootstrap",
        correction="none",
        n_bootstrap=300,
        rng=np.random.default_rng(102),
    )

    pair_ab = analysis.pairwise.get("A", "B")
    assert pair_ab.wilcoxon_p is None


def test_pairwise_wilcoxon_respects_multiple_testing_correction():
    scores = np.array(
        [
            [9.0, 8.7, 8.9, 8.8, 9.1, 8.6, 8.9, 9.0, 8.8, 8.7],
            [8.2, 8.1, 8.4, 8.2, 8.3, 8.1, 8.5, 8.3, 8.0, 8.2],
            [7.8, 7.9, 7.7, 8.0, 7.6, 7.8, 7.9, 7.7, 7.8, 7.6],
        ],
        dtype=float,
    )
    labels = ["A", "B", "C"]

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=labels,
        input_labels=[f"item_{i}" for i in range(scores.shape[1])],
    )

    analysis_raw = ps.analyze(
        result,
        method="bootstrap",
        correction="none",
        n_bootstrap=300,
        rng=np.random.default_rng(103),
    )
    analysis_bonf = ps.analyze(
        result,
        method="bootstrap",
        correction="bonferroni",
        n_bootstrap=300,
        rng=np.random.default_rng(103),
    )

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    n_pairs = len(pairs)
    for a, b in pairs:
        raw_p = analysis_raw.pairwise.get(a, b).wilcoxon_p
        corr_p = analysis_bonf.pairwise.get(a, b).wilcoxon_p
        expected_corr = min(float(raw_p) * n_pairs, 1.0)
        np.testing.assert_allclose(corr_p, expected_corr, atol=1e-12)


def test_rank_distribution_splits_mass_evenly_when_all_templates_tie():
    n_templates = 4
    n_inputs = 30
    labels = [f"T{i+1}" for i in range(n_templates)]
    input_labels = [f"item_{i}" for i in range(n_inputs)]

    # All templates are exactly tied on every input.
    scores = np.ones((n_templates, n_inputs), dtype=float)
    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=600,
        rng=np.random.default_rng(2026),
    )

    expected_p_best = np.full(n_templates, 1.0 / n_templates)
    expected_rank = np.full(n_templates, (n_templates + 1) / 2)
    np.testing.assert_allclose(analysis.rank_dist.p_best, expected_p_best, atol=1e-12)
    np.testing.assert_allclose(analysis.rank_dist.expected_ranks, expected_rank, atol=1e-12)


def test_rank_distribution_splits_top_tie_between_templates():
    labels = ["A", "B", "C"]
    n_inputs = 40
    input_labels = [f"item_{i}" for i in range(n_inputs)]

    # A and B are tied for best on every input; C is always lower.
    scores = np.array(
        [
            np.full(n_inputs, 1.0),
            np.full(n_inputs, 1.0),
            np.full(n_inputs, 0.0),
        ],
        dtype=float,
    )
    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=labels,
        input_labels=input_labels,
    )

    analysis = ps.analyze(
        result,
        n_bootstrap=600,
        rng=np.random.default_rng(2027),
    )

    np.testing.assert_allclose(
        analysis.rank_dist.p_best,
        np.array([0.5, 0.5, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        analysis.rank_dist.expected_ranks,
        np.array([1.5, 1.5, 3.0]),
        atol=1e-12,
    )
