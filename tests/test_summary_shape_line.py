"""The Shape line at the top of compare().summary() must describe the data in
the caller's factor terms, not the template slot compare() routes through."""
from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import evalstats as es


def _frame(factor: str, n_items: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for level in ["A", "B", "C"]:
        for i in range(n_items):
            rows.append({factor: level, "item": f"i{i}", "score": float(rng.integers(1, 6))})
    return pd.DataFrame(rows)


def _summary_lines(df: pd.DataFrame, factor: str) -> list[str]:
    kw = {} if factor in ("model", "prompt") else {"factors": factor}
    buf = io.StringIO()
    with warnings.catch_warnings(), redirect_stdout(buf):
        warnings.simplefilter("ignore")
        es.compare(es.load_from(df, **kw), factors=factor, metric="score",
                   score_range=(1, 5), n_bootstrap=50, rng=0).summary()
    return buf.getvalue().splitlines()


def test_shape_line_counts_models_not_prompts():
    lines = _summary_lines(_frame("model"), "model")
    shape = next(l for l in lines if l.startswith("Shape:"))
    assert shape == "Shape: 3 models × 20 inputs × 1 evaluator"
    assert "prompts=3" not in shape
    assert any(l.startswith("Models: 3 | Inputs: 20") for l in lines)


def test_shape_line_uses_a_custom_factor_name():
    lines = _summary_lines(_frame("app"), "app")
    shape = next(l for l in lines if l.startswith("Shape:"))
    assert shape == "Shape: 3 apps × 20 inputs × 1 evaluator"


def test_shape_line_keeps_benchmark_shape_for_prompt_comparisons():
    lines = _summary_lines(_frame("prompt"), "prompt")
    shape = next(l for l in lines if l.startswith("Shape:"))
    assert shape.startswith("Shape: BenchmarkShape(models=1, prompts=3")
