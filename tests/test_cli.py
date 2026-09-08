import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalstats import cli
from evalstats.core.types import BenchmarkResult, MultiModelBenchmark


_MOCK_N_INPUTS = 15  # at/above MIN_SAMPLE_FLOOR, so these mocks don't trip the sample-floor guard


def _make_single_model_result() -> BenchmarkResult:
    rng = np.random.default_rng(0)
    return BenchmarkResult(
        scores=rng.uniform(0.5, 0.9, size=(2, _MOCK_N_INPUTS)),
        template_labels=["Prompt A", "Prompt B"],
        input_labels=[f"i{i+1}" for i in range(_MOCK_N_INPUTS)],
    )


def _make_multi_model_result() -> MultiModelBenchmark:
    rng = np.random.default_rng(0)
    return MultiModelBenchmark(
        scores=rng.uniform(0.5, 0.9, size=(2, 2, _MOCK_N_INPUTS)),
        model_labels=["m1", "m2"],
        template_labels=["Prompt A", "Prompt B"],
        input_labels=[f"i{i+1}" for i in range(_MOCK_N_INPUTS)],
    )


def _write_example_data(path: Path, df: pd.DataFrame) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix == ".xlsx":
        pytest.importorskip("openpyxl")
        df.to_excel(path, index=False)
        return
    raise ValueError(f"unsupported test file suffix: {path.suffix}")


def _make_binary_long_df(n_inputs: int = 20) -> pd.DataFrame:
    rows = []
    for evaluator_idx, evaluator in enumerate(["acc", "fluency"]):
        for prompt_idx, prompt in enumerate(["Prompt A", "Prompt B"]):
            for input_idx in range(n_inputs):
                score = float(((input_idx + 2 * prompt_idx + evaluator_idx) % 3) == 0)
                rows.append(
                    {
                        "prompt": prompt,
                        "input": f"i{input_idx}",
                        "evaluator": evaluator,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


_SMOKE_METHODS = [
    "auto",
    "bootstrap",
    "bca",
    "bayes_bootstrap",
    "smooth_bootstrap",
    "permutation",
    "sign_test",
    "lmm",
    "bayes_binary",
    "wilson",
    "newcombe",
    "tango",
]


_SMOKE_ANALYZE_CASES = [
    (method, evaluator_mode, ci, statistic)
    for method, evaluator_mode, ci, statistic in product(
        _SMOKE_METHODS,
        ["aggregate", "per_evaluator"],
        [None, 0.95],
        ["mean", "median"],
    )
    if not (method == "lmm" and statistic == "median")
    if not (
        method in {"bayes_binary", "wilson", "newcombe", "tango"}
        and evaluator_mode == "aggregate"
    )
]


# This test is a smoke test to check that the analyze command runs without error for 
# a variety of option combinations.
@pytest.mark.parametrize(
    "method,evaluator_mode,ci,statistic",
    _SMOKE_ANALYZE_CASES,
)
def test_cmd_analyze_smoke_runs_for_param_grid(
    tmp_path,
    capsys,
    method,
    evaluator_mode,
    ci,
    statistic,
):
    csv_path = tmp_path / "smoke_long_binary.csv"
    _make_binary_long_df().to_csv(csv_path, index=False)

    args = argparse.Namespace(
        file=csv_path,
        format="long",
        sheet="0",
        evaluator_mode=evaluator_mode,
        ci=ci,
        ci_style="gradient",
        method=method,
        backend="statsmodels",
        n_bootstrap=30,
        correction="fdr_bh",
        spread_percentiles=(10.0, 90.0),
        reference="Prompt A",
        failure_threshold=0.5,
        statistic=statistic,
        template_model_collapse="as_runs",
        simultaneous_ci=False,
        omnibus=True,
        top_pairwise=3,
        out=None,
    )

    cli._cmd_analyze(args)
    out = capsys.readouterr().out

    assert "Running analysis ..." in out
    assert "Prompts:" in out
    assert len(out.strip()) > 0


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_load_file_reads_csv_and_xlsx_from_disk(tmp_path, suffix):
    df = pd.DataFrame(
        {
            "input": ["i1", "i2"],
            "Prompt A": [0.9, 0.8],
            "Prompt B": [0.7, 0.6],
        }
    )
    file_path = tmp_path / f"example{suffix}"
    _write_example_data(file_path, df)

    loaded = cli._load_file(file_path, sheet=0)

    pd.testing.assert_frame_equal(loaded, df)


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_cmd_analyze_runs_from_disk_for_csv_and_xlsx(tmp_path, monkeypatch, suffix):
    rng = np.random.default_rng(0)
    n = _MOCK_N_INPUTS
    df = pd.DataFrame(
        {
            "input": [f"i{i+1}" for i in range(n)],
            "Prompt A": rng.uniform(0.5, 0.9, size=n),
            "Prompt B": rng.uniform(0.5, 0.9, size=n),
        }
    )
    file_path = tmp_path / f"benchmark{suffix}"
    _write_example_data(file_path, df)

    analysis_call = {}
    summary_call = {}

    def fake_analyze(benchmark, **kwargs):
        analysis_call.update({"benchmark": benchmark, **kwargs})
        return {"ok": True}

    def fake_print_summary(analysis, top_pairwise, style, show_rank_probabilities=False, **kwargs):
        summary_call.update(
            {
                "analysis": analysis,
                "top_pairwise": top_pairwise,
                "style": style,
                "show_rank_probabilities": show_rank_probabilities,
            }
        )

    monkeypatch.setattr("evalstats.core.router.analyze", fake_analyze)
    monkeypatch.setattr("evalstats.core.summary.print_analysis_summary", fake_print_summary)

    args = argparse.Namespace(
        file=file_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="line",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=0.2,
        top_pairwise=7,
    )

    cli._cmd_analyze(args)

    assert isinstance(analysis_call["benchmark"], BenchmarkResult)
    assert analysis_call["benchmark"].template_labels == ["Prompt A", "Prompt B"]
    assert analysis_call["benchmark"].input_labels == [f"i{i+1}" for i in range(n)]
    assert analysis_call["evaluator_mode"] == "aggregate"
    assert analysis_call["reference"] == "grand_mean"
    assert analysis_call["method"] == "auto"
    assert analysis_call["backend"] == "statsmodels"
    assert analysis_call["ci"] == 0.95
    assert analysis_call["n_bootstrap"] == 100
    assert analysis_call["correction"] == "holm"
    assert analysis_call["spread_percentiles"] == (10, 90)
    assert analysis_call["failure_threshold"] == 0.2
    assert analysis_call["statistic"] == "mean"
    assert analysis_call["template_model_collapse"] == "as_runs"
    assert analysis_call["simultaneous_ci"] is True
    assert analysis_call["omnibus"] is False
    assert analysis_call["ci_style"] == "line"
    assert summary_call == {
        "analysis": {"ok": True},
        "top_pairwise": 7,
        "style": "line",
        "show_rank_probabilities": False,
    }


def test_cmd_analyze_sets_global_alpha_from_ci(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    df = pd.DataFrame({"input": ["i1", "i2"], "Prompt A": [1.0, 1.1], "Prompt B": [0.9, 1.0]})

    captured = {"alpha": None}

    args = argparse.Namespace(
        file=csv_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    monkeypatch.setattr(cli, "_load_file", lambda path, sheet: df)
    monkeypatch.setattr(
        "evalstats.io.from_dataframe",
        lambda input_df, **kwargs: (
            _make_single_model_result(),
            type("Report", (), {"format_detected": "wide"})(),
        ),
    )
    monkeypatch.setattr("evalstats.cli.set_alpha_ci", lambda alpha: captured.update(alpha=alpha))
    monkeypatch.setattr("evalstats.core.router.analyze", lambda *a, **k: {"ok": True})
    monkeypatch.setattr("evalstats.core.summary.print_analysis_summary", lambda *a, **k: None)

    cli._cmd_analyze(args)

    assert captured["alpha"] == pytest.approx(0.05)


def test_build_parser_accepts_all_option_permutations():
    parser = cli._build_parser()

    formats = ["auto", "wide", "long"]
    sheets = ["0", "Results"]
    evaluator_modes = ["aggregate", "per_evaluator"]
    cis = ["0.90", "0.99"]
    ci_styles = ["gradient", "line"]
    n_bootstraps = ["100", "2500"]
    corrections = ["auto", "holm", "bonferroni", "fdr_bh", "hochberg", "shaffer", "romano_wolf", "none"]
    references = ["grand_mean", "Prompt A"]
    failure_thresholds = [None, "0.35"]
    top_pairwise_vals = ["1", "10"]

    combos_checked = 0
    for (
        fmt,
        sheet,
        evaluator_mode,
        ci,
        ci_style,
        n_bootstrap,
        correction,
        reference,
        failure_threshold,
        top_pairwise,
    ) in product(
        formats,
        sheets,
        evaluator_modes,
        cis,
        ci_styles,
        n_bootstraps,
        corrections,
        references,
        failure_thresholds,
        top_pairwise_vals,
    ):
        argv = [
            "analyze",
            "data.csv",
            "--format",
            fmt,
            "--sheet",
            sheet,
            "--evaluator-mode",
            evaluator_mode,
            "--ci",
            ci,
            "--ci-style",
            ci_style,
            "--n-bootstrap",
            n_bootstrap,
            "--correction",
            correction,
            "--reference",
            reference,
            "--top-pairwise",
            top_pairwise,
        ]
        if failure_threshold is not None:
            argv.extend(["--failure-threshold", failure_threshold])

        args = parser.parse_args(argv)

        assert args.command == "analyze"
        assert args.file == Path("data.csv")
        assert args.format == fmt
        assert args.sheet == sheet
        assert args.evaluator_mode == evaluator_mode
        assert args.ci == float(ci)
        assert args.ci_style == ci_style
        assert args.n_bootstrap == int(n_bootstrap)
        assert args.correction == correction
        assert args.reference == reference
        assert args.failure_threshold == (
            None if failure_threshold is None else float(failure_threshold)
        )
        assert args.top_pairwise == int(top_pairwise)
        combos_checked += 1

    assert combos_checked == 6144


def test_build_parser_correction_defaults_to_auto():
    # Regression test: --correction used to default to "fdr_bh", which
    # resolve_auto_pvalue_correction_method() (the actual auto-resolution
    # logic behind analyze()'s own correction="auto" default) never
    # produces -- it only ever resolves to "shaffer" or "romano_wolf". The
    # CLI's default now matches analyze()'s.
    parser = cli._build_parser()
    args = parser.parse_args(["analyze", "data.csv"])
    assert args.correction == "auto"


@pytest.mark.parametrize(
    "fmt,detected_fmt,suffix",
    [
        ("wide", "long", ".csv"),
        ("wide", "long", ".xlsx"),
        ("long", "wide", ".csv"),
        ("long", "wide", ".xlsx"),
        ("auto", "wide", ".csv"),
        ("auto", "wide", ".xlsx"),
        ("auto", "long", ".csv"),
        ("auto", "long", ".xlsx"),
    ],
)
def test_cmd_analyze_routes_format_and_forwards_options(
    tmp_path,
    monkeypatch,
    capsys,
    fmt,
    detected_fmt,
    suffix,
):
    source_df = pd.DataFrame({"prompt": ["Prompt A"], "input": ["i1"], "score": [0.9]})
    file_path = tmp_path / f"data{suffix}"
    _write_example_data(file_path, source_df)

    from_dataframe_calls = []
    analysis_call = {}
    summary_call = {}

    result = _make_single_model_result()

    def fake_from_dataframe(df, *, format, return_report):
        from_dataframe_calls.append(
            {
                "df": df,
                "format": format,
                "return_report": return_report,
            }
        )
        report = type("Report", (), {"format_detected": detected_fmt})()
        return result, report

    def fake_analyze(benchmark, **kwargs):
        analysis_call.update({"benchmark": benchmark, **kwargs})
        return {"ok": True}

    def fake_print_summary(analysis, top_pairwise, style, show_rank_probabilities=False, **kwargs):
        summary_call.update(
            {
                "analysis": analysis,
                "top_pairwise": top_pairwise,
                "style": style,
                "show_rank_probabilities": show_rank_probabilities,
            }
        )

    monkeypatch.setattr("evalstats.io.from_dataframe", fake_from_dataframe)
    monkeypatch.setattr("evalstats.core.router.analyze", fake_analyze)
    monkeypatch.setattr("evalstats.core.summary.print_analysis_summary", fake_print_summary)

    args = argparse.Namespace(
        file=file_path,
        format=fmt,
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.9,
        ci_style="line",
        method="permutation",
        backend="statsmodels",
        n_bootstrap=1234,
        correction="fdr_bh",
        spread_percentiles=(5.0, 95.0),
        reference="Prompt A",
        failure_threshold=0.2,
        statistic="median",
        template_model_collapse="mean",
        simultaneous_ci=False,
        omnibus=True,
        top_pairwise=11,
    )

    cli._cmd_analyze(args)
    out = capsys.readouterr().out

    assert len(from_dataframe_calls) == 1
    assert from_dataframe_calls[0]["format"] == fmt
    assert from_dataframe_calls[0]["return_report"] is True

    assert analysis_call == {
        "benchmark": result,
        # Inferred from the values and declared, so the engine can't reach a
        # different verdict than the notice printed.
        "eval_type": "continuous",
        "evaluator_mode": "aggregate",
        "reference": "Prompt A",
        "method": "permutation",
        "backend": "statsmodels",
        "ci": 0.9,
        "ci_style": "line",
        "n_bootstrap": 1234,
        "correction": "fdr_bh",
        "spread_percentiles": (5.0, 95.0),
        "failure_threshold": 0.2,
        "statistic": "median",
        "template_model_collapse": "mean",
        "simultaneous_ci": False,
        "omnibus": True,
        "p_values": False,
        "pairwise_test": "auto",
    }
    assert summary_call == {
        "analysis": {"ok": True},
        "top_pairwise": 11,
        "style": "line",
        "show_rank_probabilities": False,
    }
    assert "Running analysis ..." in out


def test_cmd_analyze_rejects_reference_not_in_templates(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        file=csv_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="missing prompt",
        failure_threshold=None,
        top_pairwise=5,
    )

    df = pd.DataFrame({"input": ["i1", "i2"], "Prompt A": [1.0, 1.1], "Prompt B": [0.9, 1.0]})

    with pytest.raises(SystemExit, match="1"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cli, "_load_file", lambda path, sheet: df)
            mp.setattr(
                "evalstats.io.from_dataframe",
                lambda input_df, **kwargs: (
                    _make_single_model_result(),
                    type("Report", (), {"format_detected": "wide"})(),
                ),
            )
            cli._cmd_analyze(args)


def test_cmd_analyze_allows_per_evaluator_for_multimodel(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        file=csv_path,
        format="long",
        sheet="0",
        evaluator_mode="per_evaluator",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    df = pd.DataFrame({"prompt": ["Prompt A"], "input": ["i1"], "score": [1.0]})

    analysis_call = {}

    def fake_analyze(benchmark, **kwargs):
        analysis_call["benchmark"] = benchmark
        analysis_call.update(kwargs)
        return {"accuracy": {"ok": True}}

    monkeypatch.setattr(cli, "_load_file", lambda path, sheet: df)
    monkeypatch.setattr(
        "evalstats.io.from_dataframe",
        lambda input_df, **kwargs: (
            _make_multi_model_result(),
            type("Report", (), {"format_detected": "long"})(),
        ),
    )
    monkeypatch.setattr("evalstats.core.router.analyze", fake_analyze)
    monkeypatch.setattr("evalstats.core.summary.print_analysis_summary", lambda *a, **k: None)

    cli._cmd_analyze(args)

    assert isinstance(analysis_call["benchmark"], MultiModelBenchmark)
    assert analysis_call["evaluator_mode"] == "per_evaluator"


def test_cmd_analyze_writes_requested_outputs(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    df = pd.DataFrame({"input": ["i1", "i2"], "Prompt A": [1.0, 1.1], "Prompt B": [0.9, 1.0]})

    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"

    args = argparse.Namespace(
        file=csv_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
        out=[str(out_md), str(out_json)],
    )

    monkeypatch.setattr(cli, "_load_file", lambda path, sheet: df)
    monkeypatch.setattr(
        "evalstats.io.from_dataframe",
        lambda input_df, **kwargs: (
            _make_single_model_result(),
            type("Report", (), {"format_detected": "wide"})(),
        ),
    )
    monkeypatch.setattr("evalstats.core.router.analyze", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        "evalstats.core.summary.print_analysis_summary",
        lambda *a, **k: print("mock summary"),
    )

    cli._cmd_analyze(args)

    assert out_md.exists()
    assert out_json.exists()
    assert "mock summary" in out_md.read_text(encoding="utf-8")
    payload = out_json.read_text(encoding="utf-8")
    assert "evalstats.analysis" in payload


@pytest.mark.parametrize(
    "load_exc,expected_message_fragment",
    [
        (
            ImportError("No module named 'openpyxl'"),
            "Install openpyxl for XLSX support",
        ),
        (
            ValueError("bad csv"),
            "could not read file: bad csv",
        ),
    ],
)
def test_cmd_analyze_maps_file_read_errors_in_stderr(
    tmp_path,
    monkeypatch,
    capsys,
    load_exc,
    expected_message_fragment,
):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        file=csv_path,
        format="auto",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    def fake_load_file(path, sheet):
        raise load_exc

    monkeypatch.setattr(cli, "_load_file", fake_load_file)

    with pytest.raises(SystemExit, match="1"):
        cli._cmd_analyze(args)

    stderr = capsys.readouterr().err
    assert expected_message_fragment in stderr


def test_cmd_analyze_maps_analysis_value_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    df = pd.DataFrame({"input": ["i1", "i2"], "Prompt A": [1.0, 1.1], "Prompt B": [0.9, 1.0]})

    args = argparse.Namespace(
        file=csv_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style="gradient",
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    monkeypatch.setattr(cli, "_load_file", lambda path, sheet: df)
    monkeypatch.setattr(
        "evalstats.io.from_dataframe",
        lambda input_df, **kwargs: (
            _make_single_model_result(),
            type("Report", (), {"format_detected": "wide"})(),
        ),
    )
    monkeypatch.setattr(
        "evalstats.core.router.analyze",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("analysis failed")),
    )

    with pytest.raises(SystemExit, match="1"):
        cli._cmd_analyze(args)


def test_main_dispatches_to_cmd_analyze(monkeypatch):
    called = {"args": None}

    def fake_cmd_analyze(args):
        called["args"] = args

    monkeypatch.setattr(cli, "_cmd_analyze", fake_cmd_analyze)
    monkeypatch.setattr("sys.argv", ["evalstats", "analyze", "data.csv"])

    cli.main()

    assert called["args"] is not None
    assert called["args"].command == "analyze"
    assert called["args"].file == Path("data.csv")


def test_parse_sheet_converts_numeric_strings_and_preserves_names():
    assert cli._parse_sheet("0") == 0
    assert cli._parse_sheet("12") == 12
    assert cli._parse_sheet("Results") == "Results"
    assert cli._parse_sheet("01_summary") == "01_summary"


@pytest.mark.parametrize(
    "ci_style,expected_present,expected_absent,expect_gradient_glyphs",
    [
        ("gradient", "CI gradient [", "─ 95% CI", True),
        ("line", "─ 95% CI", "CI gradient [", False),
    ],
)
def test_cmd_analyze_prints_ci_plot_style_in_summary_output(
    tmp_path,
    capsys,
    ci_style,
    expected_present,
    expected_absent,
    expect_gradient_glyphs,
):
    csv_path = tmp_path / "style_long_binary.csv"
    _make_binary_long_df(n_inputs=18).to_csv(csv_path, index=False)

    args = argparse.Namespace(
        file=csv_path,
        format="long",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        ci_style=ci_style,
        method="bootstrap",
        backend="statsmodels",
        n_bootstrap=120,
        correction="fdr_bh",
        spread_percentiles=(10.0, 90.0),
        reference="Prompt A",
        failure_threshold=0.5,
        statistic="mean",
        template_model_collapse="as_runs",
        simultaneous_ci=False,
        omnibus=False,
        p_values=False,
        pairwise_test="auto",
        top_pairwise=3,
        brief=False,
        out=None,
    )

    cli._cmd_analyze(args)
    out = capsys.readouterr().out

    assert "legend:" in out
    assert expected_present in out
    assert expected_absent not in out

    # Verify gradient shade glyphs appear (or do not appear) on the interval
    # rows in the mean-performance section specifically.
    lines = out.splitlines()
    mean_start = next(i for i, line in enumerate(lines) if line.startswith("--- Mean Performance"))
    mean_end = next(
        i
        for i, line in enumerate(lines[mean_start + 1 :], start=mean_start + 1)
        if line.startswith("--- ")
    )
    mean_section = lines[mean_start:mean_end]
    mean_rows = [line for line in mean_section if line.strip().startswith("Prompt ")]

    assert mean_rows, "Expected prompt interval rows in mean-performance section"
    has_gradient_chars = any(any(ch in row for ch in "░▒▓█") for row in mean_rows)
    assert has_gradient_chars is expect_gradient_glyphs

# ---------------------------------------------------------------------------
# analyze --human-groundtruth (judge path)
# ---------------------------------------------------------------------------


def _make_judge_long_df(n_items: int = 30, n_lab: int = 15, seed: int = 0) -> pd.DataFrame:
    """Three models on shared items; a judge score everywhere, a human label
    on the same random n_lab items for every model. n_lab=15 is the paired
    PPI floor; below it every pair is left uncorrected."""
    rng = np.random.default_rng(seed)
    truth = {f"i{i}": rng.uniform(1, 5) for i in range(n_items)}
    labeled = set(rng.choice(sorted(truth), size=n_lab, replace=False))
    rows = []
    for m_idx, model in enumerate(["A", "B", "C"]):
        for item, t in truth.items():
            human = min(5.0, max(1.0, t + 0.4 * m_idx))
            judge = float(np.clip(round(human + rng.normal(0, 0.8)), 1, 5))
            rows.append({
                "model": model, "item": item, "score": judge,
                "human_score": human if item in labeled else np.nan,
            })
    return pd.DataFrame(rows)


def _judge_args(csv_path, **overrides):
    base = dict(
        file=csv_path, format="auto", sheet="0", evaluator_mode="aggregate", ci=None,
        ci_style="gradient", method="auto", backend="statsmodels", n_bootstrap=50,
        correction="auto", spread_percentiles=(10.0, 90.0), reference="grand_mean",
        failure_threshold=None, statistic="mean", template_model_collapse="as_runs",
        simultaneous_ci=True, omnibus=False, p_values=True, pairwise_test="auto",
        top_pairwise=5, brief=False, out=None, show_rank_probabilities=False,
        human_groundtruth="human_score", metric="score", factor=None,
        label_selection="random", seed=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_parser_accepts_human_groundtruth_flags():
    parser = cli._build_parser()
    args = parser.parse_args([
        "analyze", "results.csv", "--metric", "score", "--human-groundtruth", "human_score",
        "--label-selection", "random", "--factor", "model", "--seed", "7", "--p-values",
    ])
    assert args.human_groundtruth == "human_score"
    assert args.metric == "score"
    assert args.label_selection == "random"
    assert args.factor == "model"
    assert args.seed == 7
    # Defaults: the judge path is off unless asked for.
    plain = parser.parse_args(["analyze", "results.csv"])
    assert plain.human_groundtruth is None and plain.metric is None
    assert plain.label_selection == "unknown"


def test_cmd_analyze_human_groundtruth_runs_alignment_then_ppi_compare(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_judge_args(csv_path))
    out = capsys.readouterr().out

    assert "Judge alignment report" in out
    assert "Label selection: ✓ random" in out
    assert "PPI-CORRECTED" in out
    assert out.index("Judge alignment report") < out.index("PPI-CORRECTED")
    assert "Executive Summary" in out
    assert "PPI-" in out.split("Pairwise Comparisons", 1)[1]


def test_cmd_analyze_human_groundtruth_is_reproducible_from_seed(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_judge_args(csv_path, seed=3))
    first = capsys.readouterr().out
    cli._cmd_analyze(_judge_args(csv_path, seed=3))
    second = capsys.readouterr().out
    assert first == second


def test_cmd_analyze_human_groundtruth_infers_factor_and_writes_out(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)
    out_md = tmp_path / "summary.md"

    cli._cmd_analyze(_judge_args(csv_path, factor=None, out=[str(out_md)]))
    out = capsys.readouterr().out

    assert "comparing: 'model'" in out
    assert out_md.exists() and "PPI-CORRECTED" in out_md.read_text()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"metric": None}, "needs --metric"),
        ({"metric": "nope"}, "--metric 'nope' not found"),
        ({"human_groundtruth": "nope"}, "--human-groundtruth 'nope' not found"),
        ({"factor": "nope"}, "--factor 'nope' not found"),
    ],
)
def test_cmd_analyze_human_groundtruth_rejects_bad_columns(tmp_path, capsys, overrides, message):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)

    with pytest.raises(SystemExit):
        cli._cmd_analyze(_judge_args(csv_path, **overrides))
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --score-range and the inferred-data-kind notice
# ---------------------------------------------------------------------------


def test_build_parser_accepts_score_range():
    parser = cli._build_parser()
    args = parser.parse_args(["analyze", "results.csv", "--score-range", "1", "5"])
    assert args.score_range == [1.0, 5.0]
    assert parser.parse_args(["analyze", "results.csv"]).score_range is None


def test_cmd_analyze_prints_loud_notice_when_score_range_omitted(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_judge_args(csv_path))
    out = capsys.readouterr().out

    assert "NO --score-range GIVEN" in out
    assert "observed: 5 distinct values in [1, 5]" in out
    assert "--score-range 1 5" in out
    assert "discrete" in out
    assert out.index("NO --score-range GIVEN") < out.index("Judge alignment report")
    # The inferred (1, 5) range reaches the engine: bounded methods, not the t-interval.
    # Normalized: the summary prints display names ("Logit-t"), not raw codes.
    pairwise = out.split("Pairwise Comparisons", 1)[1].lower().replace("-", "_")
    assert "t_interval" not in pairwise and ("logit_t" in pairwise or "nig" in pairwise)


def test_cmd_analyze_score_range_is_forwarded_and_silences_notice(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_judge_args(csv_path, score_range=[1, 5]))
    out = capsys.readouterr().out

    assert "NO --score-range GIVEN" not in out
    assert "score range: [1, 5] (given)" in out
    # Bounded methods are only reachable once the range is declared.
    assert "logit_t" in out.split("Pairwise Comparisons", 1)[1].lower().replace("-", "_")


def test_cmd_analyze_regular_path_notice_and_forwarding(tmp_path, capsys, monkeypatch):
    csv_path = tmp_path / "smoke_long_binary.csv"
    _make_binary_long_df().to_csv(csv_path, index=False)
    seen = {}
    from evalstats.core import router as _router
    real_analyze = _router.analyze

    def spy(result, **kwargs):
        seen.update(kwargs)
        return real_analyze(result, **kwargs)

    monkeypatch.setattr(_router, "analyze", spy)
    base = dict(
        file=csv_path, format="long", sheet="0", evaluator_mode="aggregate", ci=None,
        ci_style="gradient", method="auto", backend="statsmodels", n_bootstrap=30,
        correction="auto", spread_percentiles=(10.0, 90.0), reference="grand_mean",
        failure_threshold=None, statistic="mean", template_model_collapse="as_runs",
        simultaneous_ci=False, omnibus=False, top_pairwise=3, out=None,
    )
    cli._cmd_analyze(argparse.Namespace(**base))
    out = capsys.readouterr().out
    assert "NO --score-range GIVEN" not in out and "binary 0/1 (detected)" in out
    assert "score_range" not in seen and "eval_type" not in seen

    cli._cmd_analyze(argparse.Namespace(**base, score_range=[0, 1]))
    out = capsys.readouterr().out
    assert "score range: [0, 1] (given)" in out
    assert seen["score_range"] == (0.0, 1.0)


def test_cmd_analyze_regular_path_infers_observed_range_for_likert(tmp_path, capsys, monkeypatch):
    rng = np.random.default_rng(0)
    rows = [{"prompt": p, "input": f"i{i}", "score": float(rng.integers(1, 6))}
            for p in ("A", "B") for i in range(30)]
    csv_path = tmp_path / "likert_long.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    seen = {}
    from evalstats.core import router as _router
    real_analyze = _router.analyze

    def spy(result, **kwargs):
        seen.update(kwargs)
        return real_analyze(result, **kwargs)

    monkeypatch.setattr(_router, "analyze", spy)
    cli._cmd_analyze(argparse.Namespace(
        file=csv_path, format="long", sheet="0", evaluator_mode="aggregate", ci=None,
        ci_style="gradient", method="auto", backend="statsmodels", n_bootstrap=30,
        correction="auto", spread_percentiles=(10.0, 90.0), reference="grand_mean",
        failure_threshold=None, statistic="mean", template_model_collapse="as_runs",
        simultaneous_ci=False, omnibus=False, top_pairwise=3, out=None,
    ))
    out = capsys.readouterr().out
    assert "NO --score-range GIVEN" in out
    assert "observed minimum and maximum" in out
    assert seen["score_range"] == (1.0, 5.0)
    assert seen["eval_type"] == "likert"
    assert "NIG" in out.split("Pairwise Comparisons", 1)[1]


def test_cmd_analyze_constant_metric_gets_no_range(tmp_path, capsys, monkeypatch):
    rows = [{"prompt": p, "input": f"i{i}", "score": 3.0} for p in ("A", "B") for i in range(30)]
    csv_path = tmp_path / "const_long.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    seen = {}
    from evalstats.core import router as _router
    real_analyze = _router.analyze
    monkeypatch.setattr(_router, "analyze", lambda result, **kw: (seen.update(kw), real_analyze(result, **kw))[1])
    cli._cmd_analyze(argparse.Namespace(
        file=csv_path, format="long", sheet="0", evaluator_mode="aggregate", ci=None,
        ci_style="gradient", method="auto", backend="statsmodels", n_bootstrap=30,
        correction="auto", spread_percentiles=(10.0, 90.0), reference="grand_mean",
        failure_threshold=None, statistic="mean", template_model_collapse="as_runs",
        simultaneous_ci=False, omnibus=False, top_pairwise=3, out=None,
    ))
    out = capsys.readouterr().out
    assert "no range can be inferred" in out
    assert "score_range" not in seen


def test_cmd_analyze_rejects_inverted_score_range(tmp_path, capsys):
    csv_path = tmp_path / "judge_long.csv"
    _make_judge_long_df().to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_judge_args(csv_path, score_range=[5, 1]))
    assert "LO < HI" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Score-kind inference across realistic metric shapes
# ---------------------------------------------------------------------------


_RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "name, values, want_type, want_range, want_loud",
    [
        # Nothing to declare: binary, and data already inside the engine's
        # own [0, 1] default. These stay quiet.
        ("binary", _RNG.integers(0, 2, 60).astype(float), None, None, False),
        ("all ones", np.ones(60), None, None, False),
        ("unit floats", _RNG.uniform(0, 1, 60), "continuous", None, False),
        ("quarters in [0,1]", _RNG.choice([0.0, 0.25, 0.5, 0.75, 1.0], 60), "likert", None, False),
        ("fine grid in [0,1]", np.round(_RNG.uniform(0, 1, 60), 5), "continuous", None, False),
        # Range inferred from the values: loud.
        ("likert 1-5", _RNG.integers(1, 6, 60).astype(float), "likert", (1.0, 5.0), True),
        ("likert 1-7", _RNG.integers(1, 8, 60).astype(float), "likert", (1.0, 7.0), True),
        ("half points 1-5", np.round(_RNG.uniform(1, 5, 60) * 2) / 2, "likert", (1.0, 5.0), True),
        ("bipolar -2..2", _RNG.integers(-2, 3, 60).astype(float), "likert", (-2.0, 2.0), True),
        ("percent int", np.arange(0, 101, dtype=float), "likert", (0.0, 100.0), True),
        # Discrete, but far too many levels to be a rating scale.
        ("token counts", _RNG.integers(10, 4000, 60).astype(float), "continuous", None, True),
        ("latency ms", _RNG.uniform(50, 5000, 60), "continuous", None, True),
        ("log-odds", _RNG.normal(0, 2, 60), "continuous", None, True),
        # No spread, so no range can be inferred -- loud, but nothing declared.
        ("constant", np.full(60, 3.0), "continuous", None, True),
        ("empty", np.full(60, np.nan), None, None, False),
    ],
)
def test_resolve_score_kind_over_realistic_metrics(capsys, name, values, want_type, want_range, want_loud):
    vals = np.asarray(values, dtype=float)
    score_type, score_range = cli._resolve_score_kind(vals, None)
    out = capsys.readouterr().out
    assert score_type == want_type, f"{name}: score_type {score_type!r}"
    if want_range is None:
        assert score_range is None or score_range == (float(np.nanmin(vals)), float(np.nanmax(vals)))
    else:
        assert score_range == want_range, f"{name}: range {score_range!r}"
    assert ("NO --score-range GIVEN" in out) is want_loud, f"{name}: notice mismatch"
    # An inferred range is always the observed span, never a guessed rubric.
    if score_range is not None:
        assert score_range == (float(np.nanmin(vals)), float(np.nanmax(vals)))


def test_resolve_score_kind_honours_an_explicit_range(capsys):
    vals = np.array([1.0, 2.0, 3.0, 4.0] * 15)
    score_type, score_range = cli._resolve_score_kind(vals, (1.0, 5.0))
    out = capsys.readouterr().out
    assert score_range == (1.0, 5.0) and score_type == "likert"
    assert "NO --score-range GIVEN" not in out and "(given)" in out


# ---------------------------------------------------------------------------
# --metric on the plain analyze path (no --human-groundtruth)
# ---------------------------------------------------------------------------

def _two_metric_df() -> pd.DataFrame:
    """Two numeric columns, neither named like a score, with opposite orderings
    so the reported means say which one was actually analyzed."""
    return pd.DataFrame([
        {"item": f"i{i}", "model": m, "expert_rating": 1.0 + j, "other_num": 9.0 - j}
        for j, m in enumerate(["A", "B"]) for i in range(_MOCK_N_INPUTS)
    ])


def _plain_args(csv_path, **overrides):
    args = _judge_args(csv_path, human_groundtruth=None, metric=None)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.mark.parametrize("metric,leader,leader_mean", [
    ("expert_rating", "B", 2.0),   # expert_rating: A=1, B=2
    ("other_num", "A", 9.0),       # other_num:     A=9, B=8
])
def test_metric_selects_the_column_on_the_plain_path(tmp_path, capsys, metric, leader, leader_mean):
    """--metric used to be read only on the --human-groundtruth path, so a
    score column named anything else failed _detect_format's has_score test,
    was parsed as wide, and died reporting every column as a missing prompt."""
    csv_path = tmp_path / "two_metrics.csv"
    _two_metric_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_plain_args(csv_path, metric=metric))
    out = capsys.readouterr().out

    assert "Detected format: long" in out
    summary = out.split("Executive Summary", 1)[1]
    top = [ln for ln in summary.splitlines() if ln.strip().startswith(leader)][0]
    assert f"{leader_mean:.3f}" in top, top


def test_metric_rejects_a_column_that_is_not_there(tmp_path, capsys):
    csv_path = tmp_path / "two_metrics.csv"
    _two_metric_df().to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_plain_args(csv_path, metric="nope"))
    err = capsys.readouterr().err
    assert "not a column" in err and "expert_rating" in err


def test_metric_rejects_a_clash_with_an_existing_score_column(tmp_path, capsys):
    csv_path = tmp_path / "clash.csv"
    _two_metric_df().assign(score=99.0).to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_plain_args(csv_path, metric="expert_rating"))
    err = capsys.readouterr().err
    assert "score" in err and "Rename or drop" in err


def test_unnamed_score_column_error_points_at_the_metric_flag(tmp_path, capsys):
    """Without --metric the parse still fails, but the message has to name the
    real cause instead of only reporting an incomplete design."""
    csv_path = tmp_path / "two_metrics.csv"
    _two_metric_df().to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_plain_args(csv_path))
    err = capsys.readouterr().err
    assert "--metric" in err and "expert_rating" in err


# ---------------------------------------------------------------------------
# --factor on the plain analyze path
# ---------------------------------------------------------------------------

def _condition_df() -> pd.DataFrame:
    """A factor column named nothing the parser recognizes as an axis."""
    return pd.DataFrame([
        {"item": f"i{i}", "condition": c, "score": 1.0 + j}
        for j, c in enumerate(["A", "B", "C"]) for i in range(_MOCK_N_INPUTS)
    ])


def test_factor_lets_the_plain_path_compare_a_non_model_column(tmp_path, capsys):
    """The parser only knows a model axis under _CANONICAL_ALIASES, so before
    --factor was honored here a 'condition' column simply failed to load."""
    csv_path = tmp_path / "cond.csv"
    _condition_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_plain_args(csv_path, factor="condition", score_range=[1, 3]))
    out = capsys.readouterr().out

    assert "3 conditions" in out
    assert "COMPARISON ACROSS 'CONDITION'" in out
    assert "Condition leaderboard" in out


def test_factor_never_leaks_the_internal_model_slot(tmp_path, capsys):
    """The column is carried in the model slot to reach the parser; the user
    named the axis something else and must never be shown the slot's name."""
    csv_path = tmp_path / "cond.csv"
    _condition_df().to_csv(csv_path, index=False)

    cli._cmd_analyze(_plain_args(csv_path, factor="condition", score_range=[1, 3]))
    out = capsys.readouterr().out

    leaks = [ln for ln in out.splitlines() if "model" in ln.lower()]
    assert not leaks, leaks


def test_factor_rejects_a_column_that_is_not_there(tmp_path, capsys):
    csv_path = tmp_path / "cond.csv"
    _condition_df().to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_plain_args(csv_path, factor="nope"))
    assert "not a column" in capsys.readouterr().err


def test_factor_rejects_a_clash_with_a_real_model_column(tmp_path, capsys):
    csv_path = tmp_path / "clash.csv"
    _condition_df().assign(model="m").to_csv(csv_path, index=False)
    with pytest.raises(SystemExit):
        cli._cmd_analyze(_plain_args(csv_path, factor="condition"))
    assert "Rename or drop" in capsys.readouterr().err


def test_default_axis_is_still_reported_as_models(tmp_path, capsys):
    csv_path = tmp_path / "models.csv"
    _condition_df().rename(columns={"condition": "model"}).to_csv(csv_path, index=False)
    cli._cmd_analyze(_plain_args(csv_path, score_range=[1, 3]))
    out = capsys.readouterr().out
    assert "3 models" in out and "COMPARISON ACROSS 'MODEL'" in out
