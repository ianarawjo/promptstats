import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promptstats import cli
from promptstats.core.types import BenchmarkResult, MultiModelBenchmark


def _make_single_model_result() -> BenchmarkResult:
    return BenchmarkResult(
        scores=np.array([[0.9, 0.8], [0.7, 0.6]], dtype=float),
        template_labels=["Prompt A", "Prompt B"],
        input_labels=["i1", "i2"],
    )


def _make_multi_model_result() -> MultiModelBenchmark:
    return MultiModelBenchmark(
        scores=np.array(
            [
                [[0.9, 0.8], [0.7, 0.6]],
                [[0.8, 0.7], [0.6, 0.5]],
            ],
            dtype=float,
        ),
        model_labels=["m1", "m2"],
        template_labels=["Prompt A", "Prompt B"],
        input_labels=["i1", "i2"],
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
    df = pd.DataFrame(
        {
            "input": ["i1", "i2"],
            "Prompt A": [0.9, 0.8],
            "Prompt B": [0.7, 0.6],
        }
    )
    file_path = tmp_path / f"benchmark{suffix}"
    _write_example_data(file_path, df)

    analysis_call = {}
    summary_call = {}

    def fake_analyze(
        benchmark,
        evaluator_mode,
        ci,
        n_bootstrap,
        correction,
        reference,
        failure_threshold,
    ):
        analysis_call.update(
            {
                "benchmark": benchmark,
                "evaluator_mode": evaluator_mode,
                "ci": ci,
                "n_bootstrap": n_bootstrap,
                "correction": correction,
                "reference": reference,
                "failure_threshold": failure_threshold,
            }
        )
        return {"ok": True}

    def fake_print_summary(analysis, top_pairwise):
        summary_call.update({"analysis": analysis, "top_pairwise": top_pairwise})

    monkeypatch.setattr("promptstats.core.router.analyze", fake_analyze)
    monkeypatch.setattr("promptstats.core.router.print_analysis_summary", fake_print_summary)

    args = argparse.Namespace(
        file=file_path,
        format="wide",
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.95,
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=0.2,
        top_pairwise=7,
    )

    cli._cmd_analyze(args)

    assert isinstance(analysis_call["benchmark"], BenchmarkResult)
    assert analysis_call["benchmark"].template_labels == ["Prompt A", "Prompt B"]
    assert analysis_call["benchmark"].input_labels == ["i1", "i2"]
    assert analysis_call["evaluator_mode"] == "aggregate"
    assert analysis_call["ci"] == 0.95
    assert analysis_call["n_bootstrap"] == 100
    assert analysis_call["correction"] == "holm"
    assert analysis_call["reference"] == "grand_mean"
    assert analysis_call["failure_threshold"] == 0.2
    assert summary_call == {"analysis": {"ok": True}, "top_pairwise": 7}


def test_build_parser_accepts_all_option_permutations():
    parser = cli._build_parser()

    formats = ["auto", "wide", "long"]
    sheets = ["0", "Results"]
    evaluator_modes = ["aggregate", "per_evaluator"]
    cis = ["0.90", "0.99"]
    n_bootstraps = ["100", "2500"]
    corrections = ["holm", "bonferroni", "fdr_bh", "none"]
    references = ["grand_mean", "Prompt A"]
    failure_thresholds = [None, "0.35"]
    top_pairwise_vals = ["1", "10"]

    combos_checked = 0
    for (
        fmt,
        sheet,
        evaluator_mode,
        ci,
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
        assert args.n_bootstrap == int(n_bootstrap)
        assert args.correction == correction
        assert args.reference == reference
        assert args.failure_threshold == (
            None if failure_threshold is None else float(failure_threshold)
        )
        assert args.top_pairwise == int(top_pairwise)
        combos_checked += 1

    assert combos_checked == 1536


@pytest.mark.parametrize(
    "fmt,detected_fmt,expected_loader,suffix",
    [
        ("wide", "long", "wide", ".csv"),
        ("wide", "long", "wide", ".xlsx"),
        ("long", "wide", "long", ".csv"),
        ("long", "wide", "long", ".xlsx"),
        ("auto", "wide", "wide", ".csv"),
        ("auto", "wide", "wide", ".xlsx"),
        ("auto", "long", "long", ".csv"),
        ("auto", "long", "long", ".xlsx"),
    ],
)
def test_cmd_analyze_routes_format_and_forwards_options(
    tmp_path,
    monkeypatch,
    capsys,
    fmt,
    detected_fmt,
    expected_loader,
    suffix,
):
    source_df = pd.DataFrame({"prompt": ["Prompt A"], "input": ["i1"], "score": [0.9]})
    file_path = tmp_path / f"data{suffix}"
    _write_example_data(file_path, source_df)

    detect_calls = []
    selected_loader = {"name": None}
    analysis_call = {}
    summary_call = {}

    result = _make_single_model_result()

    def fake_detect_format(df):
        detect_calls.append(df)
        return detected_fmt

    def fake_load_wide(df):
        selected_loader["name"] = "wide"
        return result

    def fake_load_long(df):
        selected_loader["name"] = "long"
        return result

    def fake_analyze(
        benchmark,
        evaluator_mode,
        ci,
        n_bootstrap,
        correction,
        reference,
        failure_threshold,
    ):
        analysis_call.update(
            {
                "benchmark": benchmark,
                "evaluator_mode": evaluator_mode,
                "ci": ci,
                "n_bootstrap": n_bootstrap,
                "correction": correction,
                "reference": reference,
                "failure_threshold": failure_threshold,
            }
        )
        return {"ok": True}

    def fake_print_summary(analysis, top_pairwise):
        summary_call.update({"analysis": analysis, "top_pairwise": top_pairwise})

    monkeypatch.setattr(cli, "_detect_format", fake_detect_format)
    monkeypatch.setattr(cli, "_load_wide", fake_load_wide)
    monkeypatch.setattr(cli, "_load_long", fake_load_long)
    monkeypatch.setattr("promptstats.core.router.analyze", fake_analyze)
    monkeypatch.setattr("promptstats.core.router.print_analysis_summary", fake_print_summary)

    args = argparse.Namespace(
        file=file_path,
        format=fmt,
        sheet="0",
        evaluator_mode="aggregate",
        ci=0.9,
        n_bootstrap=1234,
        correction="fdr_bh",
        reference="Prompt A",
        failure_threshold=0.2,
        top_pairwise=11,
    )

    cli._cmd_analyze(args)
    out = capsys.readouterr().out

    if fmt == "auto":
        assert len(detect_calls) == 1
    else:
        assert len(detect_calls) == 0
    assert selected_loader["name"] == expected_loader

    assert analysis_call == {
        "benchmark": result,
        "evaluator_mode": "aggregate",
        "ci": 0.9,
        "n_bootstrap": 1234,
        "correction": "fdr_bh",
        "reference": "Prompt A",
        "failure_threshold": 0.2,
    }
    assert summary_call == {"analysis": {"ok": True}, "top_pairwise": 11}
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
            mp.setattr(cli, "_load_wide", lambda input_df: _make_single_model_result())
            cli._cmd_analyze(args)


def test_cmd_analyze_rejects_per_evaluator_for_multimodel(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        file=csv_path,
        format="long",
        sheet="0",
        evaluator_mode="per_evaluator",
        ci=0.95,
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    df = pd.DataFrame({"prompt": ["Prompt A"], "input": ["i1"], "score": [1.0]})

    with pytest.raises(SystemExit, match="1"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cli, "_load_file", lambda path, sheet: df)
            mp.setattr(cli, "_load_long", lambda input_df: _make_multi_model_result())
            cli._cmd_analyze(args)


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
        n_bootstrap=100,
        correction="holm",
        reference="grand_mean",
        failure_threshold=None,
        top_pairwise=5,
    )

    monkeypatch.setattr(cli, "_load_file", lambda path, sheet: df)
    monkeypatch.setattr(cli, "_load_wide", lambda input_df: _make_single_model_result())
    monkeypatch.setattr(
        "promptstats.core.router.analyze",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("analysis failed")),
    )

    with pytest.raises(SystemExit, match="1"):
        cli._cmd_analyze(args)


def test_main_dispatches_to_cmd_analyze(monkeypatch):
    called = {"args": None}

    def fake_cmd_analyze(args):
        called["args"] = args

    monkeypatch.setattr(cli, "_cmd_analyze", fake_cmd_analyze)
    monkeypatch.setattr("sys.argv", ["promptstats", "analyze", "data.csv"])

    cli.main()

    assert called["args"] is not None
    assert called["args"].command == "analyze"
    assert called["args"].file == Path("data.csv")


def test_parse_sheet_converts_numeric_strings_and_preserves_names():
    assert cli._parse_sheet("0") == 0
    assert cli._parse_sheet("12") == 12
    assert cli._parse_sheet("Results") == "Results"
    assert cli._parse_sheet("01_summary") == "01_summary"