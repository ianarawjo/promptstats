"""Real-data adapter: OpenEval / Inspect AI corpora exposed as CISources.

A real-data ``Corpus`` (one (model, benchmark) pair's full set of finite,
deduplicated instance scores) is exposed through the same ``CISource``
interface used by ``scenarios/synthetic.py``: ``generate(rng, n)`` draws an
i.i.d.-without-replacement subsample of size n from the corpus, and
``true_mean`` is the corpus mean (the "ground truth" estimand). This lets
``cases/ci_single.py`` run one simulation loop over either synthetic or real
sources.

Five single-arm sources are available: "openeval" (downloaded from the
HuggingFace Hub on first use, then cached), "inspect" (a CSV of locally-run
benchmark results produced by ``simulations/collect_inspect_benchmarks.py``),
"appstore" (real human 1-5 star ratings, likert), "privacy_judge" (real
human 1-5 survey-mean scores, continuous), and "real", which combines
openeval + inspect + privacy_judge for maximum real-data diversity, skipping
any that aren't available locally with a note rather than failing. For
paired data, ``PAIR_SOURCES`` adds "wmt_da_paired" (real human paired
translation-quality judgments) alongside "openeval"/"inspect"/"real".

Sample sizes >= a corpus's size cannot be drawn without replacement and are
silently skipped by the caller (see ``cases/ci_single.py``).
"""

from __future__ import annotations

import csv as _csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from . import CISource, CIPairSource, MultiArmSource, EVAL_TYPE_SCALE_BOUNDS
from .synthetic import ShapeSpec

SOURCES = ["openeval", "inspect", "appstore", "privacy_judge", "real"]
PAIR_SOURCES = ["openeval", "inspect", "wmt_da_paired", "real"]

# ─────────────────────────────────────────────────────────────────────────────
# OpenEval -- constants and benchmark specs
# ─────────────────────────────────────────────────────────────────────────────

OPENEVAL_REPO = "human-centered-eval/OpenEval"

# response_id format: {source}_{YYYYMMDDTHHMMSSZ}_{index}_{model_name}_{run}
# item_id  is just  : {source}_{YYYYMMDDTHHMMSSZ}_{index}
_OE_RESP_ID_RE = re.compile(r"^(.+?)_(\d{8}T\d{6}Z)_(\d+)")

OPENEVAL_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("falcon-7b-instruct", "mmlu-pro"),
    ("gpt-4o", "culturalbench"),
    ("o4-mini", "opentom"),
    ("llama-65b", "bbq"),
    ("vicuna-13b-v1.3", "cnndm"),
    ("DeepSeek-V3-0324", "do-not-answer"),
    ("DeepSeek-R1", "emobench"),
    ("qwen-3-80b-instruct", "gpqa"),
    ("gpt-4.1-mini", "hi-tom"),
    ("gemma-3-27b-it", "ifeval"),
    ("falcon-40b-instruct", "imdb"),
    ("qwen-2.5-72b-instruct", "omni-math"),
    ("kimi-k2", "salad-bench"),
    ("llama-2-70b", "xsum"),
    ("grok-4", "truthfulqa"),
]


@dataclass
class OpenEvalBenchmarkSpec:
    benchmark_id: str
    eval_type: str
    description: str
    metric_name: str | None = None
    score_scale: float = 1.0
    score_bounds: tuple[float, float] | None = None


OPENEVAL_BENCHMARK_SPECS: dict[str, OpenEvalBenchmarkSpec] = {
    "mmlu-pro": OpenEvalBenchmarkSpec(
        benchmark_id="mmlu-pro", eval_type="binary",
        description="MMLU-Pro knowledge/reasoning (0/1 correct)",
    ),
    "gpqa": OpenEvalBenchmarkSpec(
        benchmark_id="gpqa", eval_type="binary",
        description="GPQA graduate-level science reasoning (0/1 correct)",
    ),
    "boolq": OpenEvalBenchmarkSpec(
        benchmark_id="boolq", eval_type="binary",
        description="BoolQ yes/no reading comprehension (0/1 correct)",
    ),
    "imdb": OpenEvalBenchmarkSpec(
        benchmark_id="imdb", eval_type="binary",
        description="IMDB sentiment classification (0/1 correct)",
    ),
    "truthfulqa": OpenEvalBenchmarkSpec(
        benchmark_id="truthfulqa", eval_type="continuous",
        description="TruthfulQA BLEU-max fluency/truthfulness score in [0,1]",
        metric_name="bleu_max", score_scale=0.01,
    ),
    "culturalbench": OpenEvalBenchmarkSpec(
        benchmark_id="culturalbench", eval_type="binary",
        description="CulturalBench cultural knowledge QA (0/1 correct)",
    ),
    "opentom": OpenEvalBenchmarkSpec(
        benchmark_id="opentom", eval_type="binary",
        description="OpenToM Theory-of-Mind reasoning QA (0/1 correct)",
    ),
    "bbq": OpenEvalBenchmarkSpec(
        benchmark_id="bbq", eval_type="binary",
        description="BBQ social-bias benchmark for QA (0/1 correct)",
    ),
    "bold": OpenEvalBenchmarkSpec(
        benchmark_id="bold", eval_type="binary",
        description="BOLD bias-sensitive generation benchmark",
    ),
    "do-not-answer": OpenEvalBenchmarkSpec(
        benchmark_id="do-not-answer", eval_type="continuous",
        description="Do-Not-Answer safety refusal benchmark (scores 0-6, rescaled to [0,1])",
        score_bounds=(0.0, 6.0),
    ),
    "hi-tom": OpenEvalBenchmarkSpec(
        benchmark_id="hi-tom", eval_type="binary",
        description="Hi-ToM higher-order theory-of-mind reasoning benchmark",
    ),
    "ifeval": OpenEvalBenchmarkSpec(
        benchmark_id="ifeval", eval_type="continuous",
        description="Instruction-following evaluation benchmark",
    ),
    "omni-math": OpenEvalBenchmarkSpec(
        benchmark_id="omni-math", eval_type="binary",
        description="Omni-Math mathematical reasoning benchmark",
    ),
    "salad-bench": OpenEvalBenchmarkSpec(
        benchmark_id="salad-bench", eval_type="binary",
        description="SALAD-Bench safety/alignment benchmark",
    ),
    "cnndm": OpenEvalBenchmarkSpec(
        benchmark_id="cnndm", eval_type="continuous",
        description="CNN/DailyMail summarization ROUGE-L in [0,1]",
        metric_name="rouge_l",
    ),
    "xsum": OpenEvalBenchmarkSpec(
        benchmark_id="xsum", eval_type="continuous",
        description="XSUM abstractive summarization ROUGE-L in [0,1]",
        metric_name="rouge_l",
    ),
    "emobench": OpenEvalBenchmarkSpec(
        benchmark_id="emobench", eval_type="binary",
        description="EmoBench emotional intelligence benchmark",
    ),
}

OPENEVAL_DEFAULT_BENCHMARKS: list[str] = list(dict.fromkeys(b for _, b in OPENEVAL_DEFAULT_PAIRS))


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corpus:
    """Full benchmark corpus for one (model, benchmark) pair -- the 'population'."""
    model: str
    benchmark_id: str
    eval_type: str
    source: str  # "openeval" | "inspect"
    scores: np.ndarray  # shape (N,); all deduplicated, finite instance scores
    corpus_mean: float  # ground truth estimand = mean of all scores
    corpus_size: int  # N


def corpus_to_ci_source(corpus: Corpus) -> CISource:
    """Wrap a real-data Corpus as a CISource (WOR subsampling, bounded by corpus size)."""
    scores = corpus.scores
    n_total = corpus.corpus_size

    def _generate(rng: np.random.Generator, n: int, _scores: np.ndarray = scores, _n_total: int = n_total) -> np.ndarray:
        idxs = rng.choice(_n_total, size=n, replace=False)
        return _scores[idxs]

    return CISource(
        label=f"{corpus.model}/{corpus.benchmark_id}",
        eval_type=corpus.eval_type,
        true_mean=corpus.corpus_mean,
        generate=_generate,
        source=corpus.source,
        max_n=corpus.corpus_size,
        model=corpus.model,
        benchmark_id=corpus.benchmark_id,
    )


def corpus_to_shape_spec(corpus: Corpus) -> ShapeSpec:
    """Wrap a real-data Corpus as a "custom" ShapeSpec, so it can be used
    anywhere a synthetic shape can: ``sample_group_truth``'s k-group/icc/
    base_corr machinery, ``group_total_std``, ``build_pair_sources``,
    ``build_multiarm_sources``, and the judge-bias scenario family all
    already work with any "custom" shape via its bare ``custom_sampler``,
    with no further changes needed.

    Draws are bootstrap resamples (with replacement) of the corpus's real
    scores, rather than the without-replacement subsampling
    ``corpus_to_ci_source``/``corpus_pair_to_ci_pair_source`` use elsewhere.
    Bootstrap draws compose cleanly with this machinery regardless of how
    many items/groups/runs are requested -- without-replacement sampling
    would otherwise need n bounded by the corpus size in every one of those
    call sites, several of which draw far more than one "item" at once
    (e.g. a k-group multi-arm sweep, or a large Monte Carlo estimate of the
    shape's own mean/variance).
    """
    scores = corpus.scores

    def _sampler(rng: np.random.Generator, n: int, _scores: np.ndarray = scores) -> np.ndarray:
        return rng.choice(_scores, size=n, replace=True)

    return ShapeSpec(
        label=f"real:{corpus.source}:{corpus.model}/{corpus.benchmark_id}",
        eval_type=corpus.eval_type, kind="custom", custom_sampler=_sampler,
    )


def _score_dist_summary(scores: np.ndarray, eval_type: str) -> str:
    """Score distribution diagnostic for corpus load-time output."""
    n = len(scores)
    if n == 0:
        return "[empty]"
    pct_zero = 100.0 * np.mean(np.isclose(scores, 0.0))
    pct_one = 100.0 * np.mean(np.isclose(scores, 1.0))
    looks_binary = (pct_zero + pct_one) >= 99.0
    parts = [
        f"min={scores.min():.4f}", f"max={scores.max():.4f}", f"std={scores.std():.4f}",
        f"zeros={pct_zero:.1f}%", f"ones={pct_one:.1f}%",
    ]
    flag = ""
    if eval_type == "binary" and not looks_binary:
        flag = "  *** NOT cleanly binary -- binary CI methods may be unreliable ***"
    elif eval_type != "binary" and looks_binary:
        flag = f"  *** looks binary but eval_type={eval_type!r} -- consider changing to binary ***"
    line1 = "[" + "  ".join(parts) + "]" + flag
    if not flag:
        return line1
    sorted_scores = np.sort(scores)
    idxs = np.linspace(0, n - 1, min(10, n), dtype=int)
    sample_str = "  ".join(f"{sorted_scores[i]:.4f}" for i in idxs)
    return line1 + f"\n        sample values: [{sample_str}]"


# ─────────────────────────────────────────────────────────────────────────────
# OpenEval -- loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _oe_parse(val: Any) -> Any:
    """Coerce a value that may be a JSON string or an already-parsed object."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _oe_get_model_name(model_val: Any) -> str | None:
    """Extract the model name from OpenEval's (possibly JSON-string) model field."""
    obj = _oe_parse(model_val)
    if isinstance(obj, dict):
        return obj.get("name")
    return None


def _oe_parse_response_id(response_id: str) -> tuple[str, str] | tuple[None, None]:
    """Extract (source_benchmark, item_id) from an OpenEval response_id."""
    m = _OE_RESP_ID_RE.match(response_id)
    if m is None:
        return None, None
    source = m.group(1)
    item_id = f"{source}_{m.group(2)}_{m.group(3)}"
    return source, item_id


def _oe_extract_score(scores_val: Any, metric_name: str | None) -> float | None:
    """Extract a numeric score from OpenEval's scores field (3 observed layouts)."""
    data = _oe_parse(scores_val)

    if isinstance(data, list):
        if not data:
            return None

        def _list_entry_metric_name(e: Any) -> str | None:
            if not isinstance(e, dict):
                return None
            metric = _oe_parse(e.get("metric"))
            if isinstance(metric, dict):
                return metric.get("name")
            return None

        if metric_name is None:
            entry = data[0]
        else:
            entry = next((e for e in data if _list_entry_metric_name(e) == metric_name), None)
            if entry is None:
                return None
        if not isinstance(entry, dict):
            return None
        val = entry.get("value")

    elif isinstance(data, dict):
        metrics_raw = _oe_parse(data.get("metric"))
        value_raw = data.get("value")
        if isinstance(metrics_raw, list) and isinstance(value_raw, list):
            if not value_raw:
                return None
            if metric_name is None:
                val = value_raw[0]
            else:
                val = None
                for m_obj, v in zip(metrics_raw, value_raw):
                    m_obj = _oe_parse(m_obj)
                    if isinstance(m_obj, dict) and m_obj.get("name") == metric_name:
                        val = v
                        break
        else:
            val = value_raw
    else:
        return None

    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _oe_first_metric_name(scores_val: Any) -> str | None:
    """Return the first metric name present in OpenEval's scores field, if any."""
    data = _oe_parse(scores_val)
    if isinstance(data, list) and data:
        e0 = data[0]
        if isinstance(e0, dict):
            metric = _oe_parse(e0.get("metric"))
            if isinstance(metric, dict):
                return metric.get("name")
    elif isinstance(data, dict):
        metric = _oe_parse(data.get("metric"))
        if isinstance(metric, dict):
            return metric.get("name")
        if isinstance(metric, list) and metric:
            m0 = _oe_parse(metric[0])
            if isinstance(m0, dict):
                return m0.get("name")
    return None


def _load_openeval_response_table(
    openeval_repo: str, hf_token: str | None, cache_dir: str | None,
) -> Any:
    """Load OpenEval's full "response" table.

    The repo's own dataset card declares a split literally named "all" for
    this config (a convenience "every benchmark" split) -- but recent
    ``datasets`` versions reserve "all" as a special keyword and refuse to
    build a DatasetDict containing a split by that name, raising a ValueError
    before we ever get to select "train". Loading via an explicit
    ``data_files`` glob bypasses the repo's YAML split definitions entirely
    (the library treats it as a one-off parquet load instead), which dodges
    the crash and gives the same combined row set the "all" split would have.
    """
    from datasets import load_dataset
    return load_dataset(
        openeval_repo, data_files="response/*.parquet", split="train",
        token=hf_token, cache_dir=cache_dir,
    )


def list_openeval_models(
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    benchmark_filter: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Return {model_name: {benchmark_source: response_count}}."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError("pip install datasets")

    filter_set = set(benchmark_filter) if benchmark_filter else None
    msg = (
        f"Scanning OpenEval responses for benchmarks {sorted(filter_set)} ..."
        if filter_set else
        "Scanning OpenEval response table for model names and benchmark coverage ..."
    )
    print(msg)
    ds = _load_openeval_response_table(openeval_repo, hf_token, cache_dir)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in ds:
        name = _oe_get_model_name(row.get("model"))
        source, _ = _oe_parse_response_id(row.get("response_id", ""))
        if name and source:
            if filter_set is None or source in filter_set:
                counts[name][source] += 1
    return {model: dict(bench_counts) for model, bench_counts in counts.items() if bench_counts}


def list_openeval_benchmarks(
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, int]:
    """Return {benchmark_source: response_count} for all benchmarks in OpenEval."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError("pip install datasets")

    print("Scanning OpenEval response table for benchmark IDs ...")
    ds = _load_openeval_response_table(openeval_repo, hf_token, cache_dir)
    counts: dict[str, int] = defaultdict(int)
    for row in ds:
        source, _ = _oe_parse_response_id(row.get("response_id", ""))
        if source:
            counts[source] += 1
    return dict(sorted(counts.items()))


def build_openeval_corpora(
    pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_corpus_size: int = 50,
) -> list[Corpus]:
    """Load OpenEval corpora for the given (model, benchmark_id) pairs.

    No item table needed -- response_id encodes both the benchmark source and
    item_id. Pairs must be confirmed to have data; run --list-models
    --benchmarks <id> to discover valid combinations.
    """
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError("pip install datasets")

    unknown = [b for _, b in pairs if b not in OPENEVAL_BENCHMARK_SPECS]
    if unknown:
        print(
            f"Warning: unknown OpenEval benchmark IDs {sorted(set(unknown))}.\n"
            f"  Run --list-benchmarks to see exact IDs present in the dataset.\n"
            f"  Known IDs: {list(OPENEVAL_BENCHMARK_SPECS)}"
        )
        pairs = [(m, b) for m, b in pairs if b in OPENEVAL_BENCHMARK_SPECS]
    if not pairs:
        return []

    pairs_set: set[tuple[str, str]] = set(pairs)
    benchmark_ids_set = {b for _, b in pairs}

    print("Loading OpenEval response table (~6.4 GB; cached after first download) ...")
    response_ds = _load_openeval_response_table(openeval_repo, hf_token, cache_dir)

    print(f"  Filtering to {len(pairs)} (model, benchmark) pairs ...")

    def _keep_row(batch: dict) -> list[bool]:
        keep = []
        for rid, model_val in zip(batch["response_id"], batch["model"]):
            source, _ = _oe_parse_response_id(rid)
            if source not in benchmark_ids_set:
                keep.append(False)
                continue
            mname = _oe_get_model_name(model_val)
            keep.append((mname, source) in pairs_set)
        return keep

    response_ds = response_ds.filter(_keep_row, batched=True, batch_size=5_000)
    n_filtered = len(response_ds)
    print(f"  {n_filtered:,} responses match requested pairs.")

    if n_filtered == 0:
        print(
            "  No responses found. Tips:\n"
            "    - Run --list-benchmarks to verify benchmark source IDs.\n"
            "    - Run --list-models --benchmarks <ids> to verify model names."
        )
        return []

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in response_ds:
        src, _ = _oe_parse_response_id(row.get("response_id", ""))
        mname = _oe_get_model_name(row.get("model"))
        if mname and src:
            pair_counts[(mname, src)] += 1

    print("  Response counts per pair:")
    any_zero = False
    for model_name, bench in pairs:
        n = pair_counts.get((model_name, bench), 0)
        flag = "  <- no data!" if n == 0 else ""
        print(f"    {model_name}/{bench}: {n:,}{flag}")
        if n == 0:
            any_zero = True
    if any_zero:
        print("  Tip: run --list-models --benchmarks <ids> to find models with data.")
    print()

    score_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen_items: dict[tuple[str, str], set[str]] = defaultdict(set)
    metric_seen: dict[str, str] = {}
    n_dedup = 0
    n_no_score = 0
    _score_fail_samples: list[Any] = []

    for row in response_ds:
        response_id = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(response_id)
        if source is None:
            continue

        model_name = _oe_get_model_name(row.get("model"))
        if model_name is None or (model_name, source) not in pairs_set:
            continue

        key = (model_name, source)
        if item_id in seen_items[key]:
            n_dedup += 1
            continue
        seen_items[key].add(item_id)

        spec = OPENEVAL_BENCHMARK_SPECS[source]
        scores_raw = row.get("scores")
        score = _oe_extract_score(scores_raw, spec.metric_name)
        if score is None or not np.isfinite(score):
            n_no_score += 1
            if len(_score_fail_samples) < 5:
                _score_fail_samples.append(scores_raw)
            continue

        if source not in metric_seen:
            mname = _oe_first_metric_name(scores_raw)
            if mname:
                metric_seen[source] = mname

        score_accum[key].append(float(score) * spec.score_scale)

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate (item x model) rows removed.")
    if n_no_score > 0:
        print(f"  {n_no_score:,} rows skipped (score missing or non-finite).")
        if _score_fail_samples:
            print("  Sample 'scores' field values that failed extraction:")
            for i, raw in enumerate(_score_fail_samples):
                r = repr(raw)
                if len(r) > 300:
                    r = r[:300] + " ..."
                print(f"    [{i}] type={type(raw).__name__}  value={r}")

    corpora: list[Corpus] = []
    seen_benches: set[str] = set()
    for model_name, bench in pairs:
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        if bench not in seen_benches:
            mname = metric_seen.get(bench, spec.metric_name or "first")
            print(f"\n  Benchmark: {bench}  [metric used: {mname}]")
            seen_benches.add(bench)
        scores_list = score_accum.get((model_name, bench), [])
        arr = np.array(scores_list, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_corpus_size:
            print(f"  Skip  {model_name}/{bench}: N={len(arr)} < {min_corpus_size}")
            continue
        if spec.score_bounds is not None:
            lo, hi = spec.score_bounds
            arr = (arr - lo) / (hi - lo)
        # Clip to the eval_type's canonical bounds -- upstream scores or
        # score_scale/score_bounds rescaling can overshoot by a float
        # epsilon (e.g. 1.0000000000000004), which CI methods with strict
        # domain checks (logit_t_ci_1d) reject outright. Caught via a real
        # case: grok-4/truthfulqa had exactly one such value, silently
        # turning every WOR sample that included it into a raised exception
        # -- and cases/ci_single.py's blanket ``except Exception: ci_low =
        # ci_high = obs_mean`` fallback converted that into a zero-width,
        # essentially-never-covering interval with no warning, mimicking a
        # real (but nonexistent) coverage failure that got worse as N grew
        # toward the corpus size (since that item's WOR inclusion
        # probability scales with n/corpus_size).
        clo, chi = EVAL_TYPE_SCALE_BOUNDS[spec.eval_type]
        arr = np.clip(arr, clo, chi)
        print(
            f"  OK    {model_name}/{bench}: N={len(arr)}, mean={np.mean(arr):.4f}\n"
            f"        {_score_dist_summary(arr, spec.eval_type)}"
        )
        corpora.append(Corpus(
            model=model_name, benchmark_id=bench,
            eval_type=spec.eval_type, source="openeval",
            scores=arr, corpus_mean=float(np.mean(arr)),
            corpus_size=len(arr),
        ))

    print(f"\n  {len(corpora)} corpora loaded successfully.\n")
    return corpora


def build_judge_bias_items_corpus(
    dataset_key: str, *, eval_type: str, native_bounds: tuple[float, float],
    data_dir: str = "simulations/out",
) -> Corpus | None:
    """Build a single-arm Corpus from a collect_judge_bias_data.py
    ``collect-data`` items CSV's ``human_label`` column alone -- no judge
    scores needed (unlike real_judge_bias.py's loaders, which need the
    merged CSV -- i.e. also running collect-judge-scores -- for the
    judge-vs-human bias check those feed). The human-only analogue of
    build_wmt_da_paired_corpus_pair, for datasets whose item_id needs no
    special parsing (one row = one independent item, unlike
    wmt_da_paired's segment-pair encoding) -- e.g. "appstore" (likert,
    1-5 star ratings) or "privacy_judge" (continuous, 1-5 scale, already
    the mean of ~50-68 human survey raters per item).

    Returns None (with a print explaining how to fix it) if the items CSV
    hasn't been collected yet, mirroring build_wmt_da_paired_corpus_pair's
    missing-CSV handling.

    Rescales from `native_bounds` to `EVAL_TYPE_SCALE_BOUNDS[eval_type]`
    (NOT hardcoded to [0, 1]) -- ci_single.py/ci_paired.py expect a
    CISource/CorpusPair's values to already sit on eval_type's OWN
    canonical scale (identity for "binary"/"continuous", whose canonical
    bounds already are (0, 1); the RAW native 1-5/0-100 scale for "likert"/
    "grades", which ci_single.py/ci_paired.py rescale to [0, 1] themselves
    per-method via EVAL_TYPE_SCALE_BOUNDS -- see ci_single.py's
    "rescale likert (1-5) / grades (0-100) onto [0, 1] first" comment).
    Getting this wrong for "likert" specifically (pre-rescaling to [0, 1]
    here, then letting ci_single.py rescale AGAIN via its (1, 5) bounds)
    was caught live: appstore's real review-score distribution is heavily
    boundary-saturated (~80% exactly 1 or 5 stars), and the resulting
    double-rescaled values collapsed logit_t_ci_1d's x̄ near/at 0 every
    single rep, showing up as a flat 0% coverage / zero-width column --
    not a real evalstats bug, a bug in this loader.
    """
    path = Path(data_dir) / f"judge_bias_{dataset_key}_items.csv"
    if not path.exists():
        print(
            f"  Note: {path} not found -- run\n"
            f"    python -m simulations.collect_judge_bias_data collect-data --types <binary|continuous|likert>\n"
            f"  first (no LLM API key needed; human labels only)."
        )
        return None

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        print(f"  Note: {path} has no rows.")
        return None

    lo, hi = native_bounds
    canon_lo, canon_hi = EVAL_TYPE_SCALE_BOUNDS[eval_type]
    scores = np.array([
        canon_lo + (float(r["human_label"]) - lo) / (hi - lo) * (canon_hi - canon_lo) for r in rows
    ])
    print(f"  {dataset_key}: N={len(scores)} items, mean={np.mean(scores):.4f}\n"
          f"        {_score_dist_summary(scores, eval_type)}")
    return Corpus(
        model=f"{dataset_key}_human", benchmark_id=dataset_key, eval_type=eval_type,
        source=dataset_key, scores=scores, corpus_mean=float(np.mean(scores)), corpus_size=len(scores),
    )


def build_real_data_sources(
    source: str,
    *,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_corpus_size: int = 50,
    inspect_csv: str | None = None,
) -> list[CISource]:
    """Resolve (model, benchmark) pairs for `source` and return them as CISources.

    `source` is one of "openeval", "inspect", "appstore", "privacy_judge",
    or "real". When `benchmarks` is given, only the default pairs whose
    benchmark_id is in that list are used; `models` similarly filters by
    model name (neither applies to "appstore"/"privacy_judge" -- see
    build_judge_bias_items_corpus). With no filters, each source's curated
    default pairs (OPENEVAL_DEFAULT_PAIRS) or full CSV contents (inspect)
    are used.

    "appstore" (likert, real human 1-5 star ratings) and "privacy_judge"
    (continuous, real human 1-5 survey-mean scores) are the harness's first
    real Likert/continuous single-arm sources with no LLM judge involved at
    all -- see build_judge_bias_items_corpus.

    "real" combines openeval + inspect + privacy_judge for maximum
    real-data diversity, skipping any that aren't available locally with a
    note rather than failing -- but deliberately excludes "appstore" for
    now: it's currently the only real Likert source (a single
    dataset/population, no paired Likert source at all), too thin to treat
    as a general real-data Likert validation the way the 5 continuous
    OpenEval benchmarks + privacy_judge support continuous. Still directly
    testable via --data-source appstore explicitly; just not folded into
    the default "real" sweep until there's more Likert diversity to back
    it up. Report continuous-only real-data results for now as the
    numeric-data sanity check -- Likert needs more data first.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown real-data source: {source!r}. Choices: {SOURCES}")

    def _filter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = pairs
        if benchmarks:
            out = [(m, b) for m, b in out if b in benchmarks]
        if models:
            out = [(m, b) for m, b in out if m in models]
        return out

    corpora: list[Corpus] = []
    if source in ("openeval", "real"):
        corpora += build_openeval_corpora(
            _filter_pairs(OPENEVAL_DEFAULT_PAIRS),
            hf_token=hf_token, cache_dir=cache_dir, min_corpus_size=min_corpus_size,
        )
    if source == "inspect":
        corpora += build_inspect_corpora(
            inspect_csv or DEFAULT_INSPECT_CSV, models=models, benchmarks=benchmarks,
            min_corpus_size=min_corpus_size,
        )
    if source == "real":
        csv_path = inspect_csv or DEFAULT_INSPECT_CSV
        if Path(csv_path).exists():
            corpora += build_inspect_corpora(
                csv_path, models=models, benchmarks=benchmarks, min_corpus_size=min_corpus_size,
            )
        else:
            print(f"  Note: --real requested but inspect CSV not found at {csv_path!r} -- skipping inspect, using openeval only.")
    if source == "appstore" and (not benchmarks or "appstore" in benchmarks):
        # Deliberately NOT folded into "real" -- see docstring: too thin a
        # Likert data point on its own to fold into the default real-data
        # sweep yet. Still runnable directly via --data-source appstore.
        c = build_judge_bias_items_corpus("appstore", eval_type="likert", native_bounds=(1.0, 5.0))
        if c is not None and c.corpus_size >= min_corpus_size:
            corpora.append(c)
    if source in ("privacy_judge", "real") and (not benchmarks or "privacy_judge" in benchmarks):
        c = build_judge_bias_items_corpus("privacy_judge", eval_type="continuous", native_bounds=(1.0, 5.0))
        if c is not None and c.corpus_size >= min_corpus_size:
            corpora.append(c)

    return [corpus_to_ci_source(c) for c in corpora]


# ─────────────────────────────────────────────────────────────────────────────
# Paired (shared-item) real data
#
# Supports any OPENEVAL_BENCHMARK_SPECS eval_type (binary or continuous).
# build_real_pair_sources below is the flat (R=1) variant; multi-run real
# pairs are built by build_real_pair_sources_nested (used by ci_paired.py's
# --nested-mode with --data-source inspect).
# ─────────────────────────────────────────────────────────────────────────────

# Default (model, benchmark) pairs confirmed to have data in OpenEval.
OPENEVAL_PAIR_DEFAULT_MODEL_BENCH: list[tuple[str, str]] = [
    ("falcon-40b", "bbq"),
    ("llama-7b", "bbq"),
    ("qwen-2.5-32b-instruct", "bbq"),
    ("gemma-2-27b-it", "mmlu-pro"),
    ("gemma-3-4b-it", "mmlu-pro"),
    ("qwen-3-30b-instruct", "mmlu-pro"),
    ("llama-2-70b-hf", "mmlu-pro"),
    ("DeepSeek-R1", "culturalbench"),
    ("gpt-4o", "culturalbench"),
    ("gpt-4.1-mini", "culturalbench"),
    ("o4-mini", "culturalbench"),
    ("grok-4", "culturalbench"),
    ("gpt-4o-mini", "hi-tom"),
    ("grok-4", "hi-tom"),
    ("phi-4", "hi-tom"),
    ("DeepSeek-R1", "hi-tom"),
    ("DeepSeek-R1", "opentom"),
    ("gpt-4.1", "opentom"),
    ("o1", "opentom"),
    ("DeepSeek-R1", "salad-bench"),
    ("grok-4", "salad-bench"),
    ("kimi-k2", "salad-bench"),
    ("phi-4", "salad-bench"),
    ("gemma-3-27b-it", "omni-math"),
    ("qwen-2.5-14b-instruct", "omni-math"),
    ("qwen-2.5-72b-instruct", "omni-math"),
    ("llama-2-13b-hf", "omni-math"),
    # Continuous benchmarks. Each pair confirmed live to share >=500 aligned
    # items with no non-binary skips/rounding (they're already continuous,
    # so nothing gets rounded).
    ("bloom", "cnndm"),
    ("opt-66b", "cnndm"),
    ("DeepSeek-V3-0324", "do-not-answer"),
    ("Phi-4", "do-not-answer"),
    ("gemma-2b-it", "ifeval"),
    ("qwen-3-30b-instruct", "ifeval"),
    ("Llama-2-13b-hf", "truthfulqa"),
    ("gemma-1.1-7b-it", "truthfulqa"),
    ("curie", "xsum"),
    ("davinci", "xsum"),
]


@dataclass
class CorpusPair:
    """Aligned paired scores for two models on the same benchmark (R=1 only)."""
    model_a: str
    model_b: str
    benchmark_id: str
    source: str  # "openeval" | "inspect"
    scores_a: np.ndarray  # shape (N,)
    scores_b: np.ndarray  # shape (N,)
    true_diff: float  # mean(scores_a - scores_b) -- population ground truth
    corpus_size: int  # N = number of shared items
    eval_type: str = "binary"  # "binary" | "continuous" -- see OPENEVAL_BENCHMARK_SPECS


def corpus_pair_to_ci_pair_source(cp: CorpusPair) -> CIPairSource:
    """Wrap a real-data CorpusPair as a CIPairSource (WOR subsampling, R=1 only)."""
    scores_a, scores_b, n_total = cp.scores_a, cp.scores_b, cp.corpus_size

    def _generate_pair(
        rng: np.random.Generator, n: int, runs: int,
        _a: np.ndarray = scores_a, _b: np.ndarray = scores_b, _n_total: int = n_total,
    ) -> tuple[np.ndarray, np.ndarray]:
        if runs != 1:
            raise ValueError("Real-data pair sources only support runs=1 in this pass.")
        idxs = rng.choice(_n_total, size=n, replace=False)
        return _a[idxs].reshape(n, 1), _b[idxs].reshape(n, 1)

    return CIPairSource(
        label=f"{cp.model_a} vs {cp.model_b}/{cp.benchmark_id}",
        eval_type=cp.eval_type,
        true_diff=cp.true_diff,
        generate_pair=_generate_pair,
        source=cp.source,
        max_n=cp.corpus_size,
        model_a=cp.model_a,
        model_b=cp.model_b,
        benchmark_id=cp.benchmark_id,
    )


def corpus_pair_to_null_ci_pair_source(cp: CorpusPair) -> CIPairSource:
    """Build a real-data *null* (H0: no true difference) CIPairSource from a
    CorpusPair, for Type-I error calibration checks (ci_paired.py's "TYPE I
    ERROR RATE" section, pvalues.py's pairwise Type-I column).

    Two distinct real models can't be forced to have a "true" zero
    difference the way a synthetic scenario can (there's no d=0 analogue of
    a real A-vs-B corpus). Instead this uses the standard permutation-null
    construction: each rep independently label-swaps (a_i, b_i) -> (b_i,
    a_i) per item with probability 0.5. Every emitted value is still a real,
    unmodified observed score -- preserving the corpus's actual noise,
    skew, and item-level correlation -- but symmetrizing the A/B assignment
    makes E[mean(A) - mean(B)] = 0 exactly. This is the same random
    sign-flip resampling a permutation test's null distribution is built
    from, applied once per rep instead of many.
    """
    scores_a, scores_b, n_total = cp.scores_a, cp.scores_b, cp.corpus_size

    def _generate_pair(
        rng: np.random.Generator, n: int, runs: int,
        _a: np.ndarray = scores_a, _b: np.ndarray = scores_b, _n_total: int = n_total,
    ) -> tuple[np.ndarray, np.ndarray]:
        if runs != 1:
            raise ValueError("Real-data pair sources only support runs=1 in this pass.")
        idxs = rng.choice(_n_total, size=n, replace=False)
        a, b = _a[idxs], _b[idxs]
        swap = rng.random(n) < 0.5
        a_null = np.where(swap, b, a)
        b_null = np.where(swap, a, b)
        return a_null.reshape(n, 1), b_null.reshape(n, 1)

    return CIPairSource(
        label=f"{cp.model_a} vs {cp.model_b}/{cp.benchmark_id}|null",
        eval_type=cp.eval_type,
        true_diff=0.0,
        generate_pair=_generate_pair,
        source=cp.source,
        max_n=cp.corpus_size,
        model_a=cp.model_a,
        model_b=cp.model_b,
        benchmark_id=cp.benchmark_id,
        is_null=True,
    )


def build_openeval_corpus_pairs(
    model_bench_pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs from OpenEval by aligning on item_id (any eval_type -- see OPENEVAL_BENCHMARK_SPECS)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError("pip install datasets")

    unknown_benches = [b for _, b in model_bench_pairs if b not in OPENEVAL_BENCHMARK_SPECS]
    if unknown_benches:
        print(
            f"Warning: unsupported OpenEval benchmark IDs: {sorted(set(unknown_benches))}.\n"
            f"  Supported: {list(OPENEVAL_BENCHMARK_SPECS)}"
        )
        model_bench_pairs = [(m, b) for m, b in model_bench_pairs if b in OPENEVAL_BENCHMARK_SPECS]
    if not model_bench_pairs:
        return []

    pairs_set = set(model_bench_pairs)
    bench_set = {b for _, b in model_bench_pairs}

    print("Loading OpenEval response table (~1.4 GB; cached after first download) ...")
    response_ds = _load_openeval_response_table(openeval_repo, hf_token, cache_dir)

    def _keep_row(batch: dict) -> list[bool]:
        keep = []
        for rid, model_val in zip(batch["response_id"], batch["model"]):
            source, _ = _oe_parse_response_id(rid)
            if source not in bench_set:
                keep.append(False)
                continue
            mname = _oe_get_model_name(model_val)
            keep.append((mname, source) in pairs_set)
        return keep

    response_ds = response_ds.filter(_keep_row, batched=True, batch_size=5_000)
    print(f"  {len(response_ds):,} responses after filtering.")

    item_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    n_dedup = 0
    for row in response_ds:
        rid = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(rid)
        if source is None:
            continue
        mname = _oe_get_model_name(row.get("model"))
        if mname is None or (mname, source) not in pairs_set:
            continue
        key = (mname, source)
        if item_id in item_maps[key]:
            n_dedup += 1
            continue
        spec = OPENEVAL_BENCHMARK_SPECS[source]
        score = _oe_extract_score(row.get("scores"), spec.metric_name)
        if score is not None and np.isfinite(score):
            item_maps[key][item_id] = float(score) * spec.score_scale

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate rows removed (kept first per item x model).")

    for (model, bench), scores_map in list(item_maps.items()):
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        keys = list(scores_map.keys())
        vals = np.array([scores_map[k] for k in keys], dtype=float)
        if spec.eval_type == "binary":
            non_binary_mask = ~np.isin(vals, [0.0, 1.0])
            if np.any(non_binary_mask):
                rounded_vals = np.clip(np.rint(vals), 0.0, 1.0)
                unique_bad = np.unique(vals[non_binary_mask])[:5]
                print(f"  Warning: {model}/{bench} has {int(np.sum(non_binary_mask)):,} non-binary scores (e.g. {unique_bad}). Rounded to {{0,1}}.")
                item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rounded_vals)}
        elif spec.score_bounds is not None:
            # Rescale to [0,1] using the benchmark's known theoretical range
            # (matching build_openeval_corpora's single-arm handling) so
            # continuous benchmarks like do-not-answer/ifeval land in the
            # same range logit_t/nig/el assume -- not just score_scale
            # (truthfulqa/cnndm/xsum are already in [0,1] via score_scale
            # alone and have no score_bounds set).
            lo, hi = spec.score_bounds
            rescaled_vals = (vals - lo) / (hi - lo)
            item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rescaled_vals)}

    corpus_pairs: list[CorpusPair] = []
    for bench in sorted(bench_set):
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        requested = list(dict.fromkeys(m for m, b in model_bench_pairs if b == bench))
        bench_models = [m for m in requested if (m, bench) in item_maps and item_maps[(m, bench)]]
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: fewer than 2 models with data.")
            continue
        print(f"\n  Benchmark: {bench}")
        for model in bench_models:
            print(f"    {model}: {len(item_maps[(model, bench)]):,} items")
        for model_a, model_b in combinations(bench_models, 2):
            map_a = item_maps[(model_a, bench)]
            map_b = item_maps[(model_b, bench)]
            shared_ids = sorted(map_a.keys() & map_b.keys())
            if len(shared_ids) < min_pair_size:
                print(f"  Skip  ({model_a}, {model_b}) on {bench}: {len(shared_ids)} shared items < {min_pair_size}")
                continue
            scores_a = np.array([map_a[k] for k in shared_ids])
            scores_b = np.array([map_b[k] for k in shared_ids])
            if spec.eval_type == "binary":
                # The per-(model,bench) rounding pass above already forces
                # binary benchmarks to exactly {0,1}; this only catches the
                # (should-be-impossible) case where alignment somehow
                # reintroduced a non-{0,1} value.
                bad_a = scores_a[~np.isin(scores_a, [0.0, 1.0])]
                bad_b = scores_b[~np.isin(scores_b, [0.0, 1.0])]
                if len(bad_a) > 0 or len(bad_b) > 0:
                    print(f"  Skip  ({model_a} vs {model_b}) on {bench}: non-binary scores after alignment. Skipping pair.")
                    continue
            else:
                # Clip float-epsilon overshoot from score_scale/score_bounds
                # rescaling -- see build_openeval_corpora's identical clip
                # for why (a real case, grok-4/truthfulqa, had exactly one
                # such value).
                clo, chi = EVAL_TYPE_SCALE_BOUNDS[spec.eval_type]
                scores_a = np.clip(scores_a, clo, chi)
                scores_b = np.clip(scores_b, clo, chi)
            true_diff = float(np.mean(scores_a - scores_b))
            print(
                f"  Pair  ({model_a} vs {model_b}): N={len(shared_ids)}, mean_A={np.mean(scores_a):.4f}, "
                f"mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="openeval",
                scores_a=scores_a, scores_b=scores_b, true_diff=true_diff, corpus_size=len(shared_ids),
                eval_type=spec.eval_type,
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from OpenEval.\n")
    return corpus_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Inspect AI -- manually-collected local benchmark run data
#
# This is run data the project owner ran and collected locally (via
# simulations/collect_inspect_benchmarks.py), not something downloaded from a
# hub on first use. Stored as a flat CSV: benchmark, model, item_id, run_idx,
# score. Several runs per item are typically present (run_idx > 0); both
# loaders below keep only run_idx == 0, since the single-sample/paired
# simulations in this harness need one independent score per item, not
# several repeated judge runs on the same item.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INSPECT_CSV = "simulations/out/inspect_benchmarks.csv"


def _load_inspect_item_maps(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Parse the Inspect CSV into {(model, benchmark): {item_id: score}},
    keeping only run_idx == 0 rows. Shared by build_inspect_corpora (single-
    sample) and build_inspect_corpus_pairs (paired)."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Inspect data file not found: {csv_path}\n  Run collect_inspect_benchmarks.py first to generate it."
        )

    item_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    print(f"Loading Inspect AI data from: {csv_path}")
    n_rows = 0
    with p.open(newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            bench = row.get("benchmark", "").strip()
            model = row.get("model", "").strip()
            item_id = row.get("item_id", "").strip()
            try:
                run_idx = int(row.get("run_idx", 0))
                score = float(row.get("score", float("nan")))
            except (ValueError, TypeError):
                continue
            if run_idx != 0 or not bench or not model or not item_id or not np.isfinite(score):
                continue
            if benchmarks is not None and bench not in benchmarks:
                continue
            if models is not None and model not in models:
                continue
            item_maps[(model, bench)][item_id] = score
            n_rows += 1

    print(f"  {n_rows:,} rows loaded (run_idx=0 only).")
    return item_maps


def build_inspect_corpora(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_corpus_size: int = 50,
) -> list[Corpus]:
    """Build single-sample Corpora (one score per item, run_idx == 0 only)
    from a CSV produced by collect_inspect_benchmarks.py -- one Corpus per
    (model, benchmark) pair found in the CSV."""
    item_maps = _load_inspect_item_maps(csv_path, models=models, benchmarks=benchmarks)
    if not item_maps:
        print("  No data found -- check --benchmarks / --models filters match the CSV.")
        return []

    corpora: list[Corpus] = []
    for (model, bench), scores_map in sorted(item_maps.items()):
        arr = np.array(list(scores_map.values()), dtype=float)
        if len(arr) < min_corpus_size:
            print(f"  Skip  {model}/{bench}: N={len(arr)} < {min_corpus_size}")
            continue
        print(f"  OK    {model}/{bench}: N={len(arr)}, mean={np.mean(arr):.4f}")
        corpora.append(Corpus(
            model=model, benchmark_id=bench, eval_type="binary", source="inspect",
            scores=arr, corpus_mean=float(np.mean(arr)), corpus_size=len(arr),
        ))

    print(f"\n  {len(corpora)} corpora loaded from Inspect AI data.\n")
    return corpora


def build_inspect_corpus_pairs(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs (R=1 only) from a CSV produced by collect_inspect_benchmarks.py."""
    item_maps = _load_inspect_item_maps(csv_path, models=models, benchmarks=benchmarks)
    if not item_maps:
        print("  No data found -- check --benchmarks / --models filters match the CSV.")
        return []

    all_benches = sorted({b for _, b in item_maps.keys()})
    corpus_pairs: list[CorpusPair] = []

    for bench in all_benches:
        bench_models = sorted(m for m, b in item_maps.keys() if b == bench)
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: only {len(bench_models)} model(s) -- need >= 2 to form a pair")
            continue
        print(f"\n  Benchmark: {bench}")
        for model in bench_models:
            print(f"    {model}: {len(item_maps[(model, bench)]):,} items")

        for model_a, model_b in combinations(bench_models, 2):
            map_a = item_maps[(model_a, bench)]
            map_b = item_maps[(model_b, bench)]
            shared_ids = sorted(map_a.keys() & map_b.keys())
            if len(shared_ids) < min_pair_size:
                print(f"  Skip  ({model_a} vs {model_b}) on {bench}: {len(shared_ids)} shared items < {min_pair_size}")
                continue
            scores_a = np.array([map_a[k] for k in shared_ids])
            scores_b = np.array([map_b[k] for k in shared_ids])
            true_diff = float(np.mean(scores_a - scores_b))
            short_a = model_a.split("/")[-1] if "/" in model_a else model_a
            short_b = model_b.split("/")[-1] if "/" in model_b else model_b
            print(
                f"  Pair  ({short_a} vs {short_b}): N={len(shared_ids)}, mean_A={np.mean(scores_a):.4f}, "
                f"mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="inspect",
                scores_a=scores_a, scores_b=scores_b, true_diff=true_diff, corpus_size=len(shared_ids),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from Inspect AI data.\n")
    return corpus_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Multi-arm (shared-item, k>=2) real data -- cases/pvalues.py's --mode multiarm
#
# Same restriction as the pairwise real-data sources above: known-binary
# benchmarks only, R=1 only. For each benchmark, ALL of its available real
# models (not just one pair) are aligned on shared item_id and ordered by
# descending real corpus mean -- arm 0 is therefore always the empirically
# best-performing model, matching MultiArmSource's "arm 0 carries the
# alternative-hypothesis shift" convention. A run requesting more arms
# (--k-arms) than a benchmark has real models is skipped by the caller via
# MultiArmSource.max_k, the same way an oversized --sizes n is skipped via
# max_n.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MultiArmCorpus:
    """Aligned real data for >=2 models sharing one benchmark (R=1 only)."""
    benchmark_id: str
    source: str  # "openeval" | "inspect"
    models: list[str]  # ordered by descending real corpus mean; models[0] is "arm 0"
    scores: np.ndarray  # shape (len(models), N) -- aligned on shared item_id
    corpus_size: int  # N = number of shared items
    eval_type: str = "binary"  # "binary" | "continuous" -- see OPENEVAL_BENCHMARK_SPECS


def multiarm_corpus_to_source(mc: MultiArmCorpus) -> MultiArmSource:
    """Wrap a real-data MultiArmCorpus as a MultiArmSource (WOR item
    subsampling, R=1 only, k bounded by the number of aligned real models)."""
    scores, n_total, max_k = mc.scores, mc.corpus_size, len(mc.models)

    def _generate_scores(
        rng: np.random.Generator, n: int, runs: int, k: int, delta: float,
        _scores: np.ndarray = scores, _n_total: int = n_total, _max_k: int = max_k,
    ) -> np.ndarray:
        if runs != 1:
            raise ValueError("Real-data multiarm sources only support runs=1 in this pass.")
        if k > _max_k:
            raise ValueError(
                f"Requested k={k} exceeds {_max_k} real arms available for {mc.benchmark_id} "
                f"-- filter --k-arms so k <= max_k (see MultiArmSource.max_k)."
            )
        idxs = rng.choice(_n_total, size=n, replace=False)
        sub = _scores[:k][:, idxs]  # (k, n) -- the k empirically-best real models
        if delta == 0.0:
            # Permutation null (H0: no true difference between arms): each
            # sampled item's k real scores are independently shuffled across
            # arm slots, so every arm's marginal distribution converges to
            # the same across-model mixture (equal expected value for every
            # arm) while every emitted value is still a real, unmodified
            # observed score -- same permutation-null idea as
            # corpus_pair_to_null_ci_pair_source, generalized to k arms.
            perm = np.argsort(rng.random((k, n)), axis=0)
            sub = np.take_along_axis(sub, perm, axis=0)
        return sub[:, :, None]  # (k, n, 1) -- runs=1

    true_arm_means = scores.mean(axis=1)  # (max_k,) -- exact, same arm order as _scores[:k] above

    def _true_means(k: int, delta: float, _means: np.ndarray = true_arm_means) -> np.ndarray:
        if delta == 0.0:
            # Permutation null: the construction above makes every arm's
            # expected value equal, so the "true" value to check CI coverage
            # against is 0 for every pairwise diff, not the real corpus means.
            return np.zeros(k)
        return _means[:k]

    return MultiArmSource(
        label=f"{mc.source}:{mc.benchmark_id}", eval_type=mc.eval_type,
        generate_scores=_generate_scores, alt_delta=1.0, source=mc.source,
        max_n=n_total, max_k=max_k, benchmark_id=mc.benchmark_id, true_means=_true_means,
    )


def build_openeval_multiarm_corpora(
    model_bench_pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_arm_size: int = 50,
) -> list[MultiArmCorpus]:
    """Build MultiArmCorpora from OpenEval: for each benchmark, align ALL its
    requested models on shared item_id (any eval_type -- see
    OPENEVAL_BENCHMARK_SPECS, same as build_openeval_corpus_pairs)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError("pip install datasets")

    unknown_benches = [b for _, b in model_bench_pairs if b not in OPENEVAL_BENCHMARK_SPECS]
    if unknown_benches:
        print(
            f"Warning: unsupported OpenEval benchmark IDs: {sorted(set(unknown_benches))}.\n"
            f"  Supported: {list(OPENEVAL_BENCHMARK_SPECS)}"
        )
        model_bench_pairs = [(m, b) for m, b in model_bench_pairs if b in OPENEVAL_BENCHMARK_SPECS]
    if not model_bench_pairs:
        return []

    pairs_set = set(model_bench_pairs)
    bench_set = {b for _, b in model_bench_pairs}

    print("Loading OpenEval response table (~1.4 GB; cached after first download) ...")
    response_ds = _load_openeval_response_table(openeval_repo, hf_token, cache_dir)

    def _keep_row(batch: dict) -> list[bool]:
        keep = []
        for rid, model_val in zip(batch["response_id"], batch["model"]):
            source, _ = _oe_parse_response_id(rid)
            if source not in bench_set:
                keep.append(False)
                continue
            mname = _oe_get_model_name(model_val)
            keep.append((mname, source) in pairs_set)
        return keep

    response_ds = response_ds.filter(_keep_row, batched=True, batch_size=5_000)
    print(f"  {len(response_ds):,} responses after filtering.")

    item_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    n_dedup = 0
    for row in response_ds:
        rid = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(rid)
        if source is None:
            continue
        mname = _oe_get_model_name(row.get("model"))
        if mname is None or (mname, source) not in pairs_set:
            continue
        key = (mname, source)
        if item_id in item_maps[key]:
            n_dedup += 1
            continue
        spec = OPENEVAL_BENCHMARK_SPECS[source]
        score = _oe_extract_score(row.get("scores"), spec.metric_name)
        if score is not None and np.isfinite(score):
            item_maps[key][item_id] = float(score) * spec.score_scale

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate rows removed (kept first per item x model).")

    for (model, bench), scores_map in list(item_maps.items()):
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        keys = list(scores_map.keys())
        vals = np.array([scores_map[k] for k in keys], dtype=float)
        if spec.eval_type == "binary":
            non_binary_mask = ~np.isin(vals, [0.0, 1.0])
            if np.any(non_binary_mask):
                rounded_vals = np.clip(np.rint(vals), 0.0, 1.0)
                unique_bad = np.unique(vals[non_binary_mask])[:5]
                print(f"  Warning: {model}/{bench} has {int(np.sum(non_binary_mask)):,} non-binary scores (e.g. {unique_bad}). Rounded to {{0,1}}.")
                item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rounded_vals)}
        elif spec.score_bounds is not None:
            lo, hi = spec.score_bounds
            rescaled_vals = (vals - lo) / (hi - lo)
            item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rescaled_vals)}

    corpora: list[MultiArmCorpus] = []
    for bench in sorted(bench_set):
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        requested = list(dict.fromkeys(m for m, b in model_bench_pairs if b == bench))
        bench_models = [m for m in requested if (m, bench) in item_maps and item_maps[(m, bench)]]
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: fewer than 2 models with data.")
            continue
        shared_ids = sorted(set.intersection(*(set(item_maps[(m, bench)].keys()) for m in bench_models)))
        if len(shared_ids) < min_arm_size:
            print(f"  Skip  {bench}: {len(shared_ids)} shared items across {len(bench_models)} models < {min_arm_size}")
            continue
        means = {m: float(np.mean([item_maps[(m, bench)][i] for i in shared_ids])) for m in bench_models}
        ordered_models = sorted(bench_models, key=lambda m: means[m], reverse=True)
        scores = np.array([[item_maps[(m, bench)][i] for i in shared_ids] for m in ordered_models])
        # Clip float-epsilon overshoot -- see build_openeval_corpora's clip.
        clo, chi = EVAL_TYPE_SCALE_BOUNDS[spec.eval_type]
        scores = np.clip(scores, clo, chi)
        print(
            f"  OK    {bench}: {len(ordered_models)} models, N={len(shared_ids)} shared items, "
            f"means={[round(means[m], 4) for m in ordered_models]}"
        )
        corpora.append(MultiArmCorpus(
            benchmark_id=bench, source="openeval", models=ordered_models,
            scores=scores, corpus_size=len(shared_ids), eval_type=spec.eval_type,
        ))

    print(f"\n  {len(corpora)} multi-arm corpora built from OpenEval.\n")
    return corpora


def build_inspect_multiarm_corpora(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_arm_size: int = 50,
) -> list[MultiArmCorpus]:
    """Build MultiArmCorpora from a CSV produced by collect_inspect_benchmarks.py:
    for each benchmark, align ALL its available models on shared item_id."""
    item_maps = _load_inspect_item_maps(csv_path, models=models, benchmarks=benchmarks)
    if not item_maps:
        print("  No data found -- check --benchmarks / --models filters match the CSV.")
        return []

    all_benches = sorted({b for _, b in item_maps.keys()})
    corpora: list[MultiArmCorpus] = []

    for bench in all_benches:
        bench_models = sorted(m for m, b in item_maps.keys() if b == bench)
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: only {len(bench_models)} model(s) -- need >= 2")
            continue
        shared_ids = sorted(set.intersection(*(set(item_maps[(m, bench)].keys()) for m in bench_models)))
        if len(shared_ids) < min_arm_size:
            print(f"  Skip  {bench}: {len(shared_ids)} shared items across {len(bench_models)} models < {min_arm_size}")
            continue
        means = {m: float(np.mean([item_maps[(m, bench)][i] for i in shared_ids])) for m in bench_models}
        ordered_models = sorted(bench_models, key=lambda m: means[m], reverse=True)
        scores = np.array([[item_maps[(m, bench)][i] for i in shared_ids] for m in ordered_models])
        print(
            f"  OK    {bench}: {len(ordered_models)} models, N={len(shared_ids)} shared items, "
            f"means={[round(means[m], 4) for m in ordered_models]}"
        )
        corpora.append(MultiArmCorpus(
            benchmark_id=bench, source="inspect", models=ordered_models,
            scores=scores, corpus_size=len(shared_ids),
        ))

    print(f"\n  {len(corpora)} multi-arm corpora built from Inspect AI data.\n")
    return corpora


def build_real_multiarm_sources(
    source: str,
    *,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_arm_size: int = 50,
    inspect_csv: str | None = None,
) -> list[MultiArmSource]:
    """Resolve real multi-arm (k>=2 aligned models per benchmark) groups for
    `source` and return them as MultiArmSources. `source` is one of
    "openeval", "inspect", or "real" (combines both, skipping "inspect" with
    a warning rather than failing if its CSV isn't present locally). R=1
    only -- see module docstring."""
    if source not in PAIR_SOURCES:
        raise ValueError(f"Unknown real-data multiarm source: {source!r}. Choices: {PAIR_SOURCES}")

    def _filter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = pairs
        if benchmarks:
            out = [(m, b) for m, b in out if b in benchmarks]
        if models:
            out = [(m, b) for m, b in out if m in models]
        return out

    corpora: list[MultiArmCorpus] = []
    if source in ("openeval", "real"):
        corpora += build_openeval_multiarm_corpora(
            _filter_pairs(OPENEVAL_PAIR_DEFAULT_MODEL_BENCH),
            hf_token=hf_token, cache_dir=cache_dir, min_arm_size=min_arm_size,
        )
    if source == "inspect":
        corpora += build_inspect_multiarm_corpora(
            inspect_csv or DEFAULT_INSPECT_CSV, models=models, benchmarks=benchmarks, min_arm_size=min_arm_size,
        )
    if source == "real":
        csv_path = inspect_csv or DEFAULT_INSPECT_CSV
        if Path(csv_path).exists():
            corpora += build_inspect_multiarm_corpora(
                csv_path, models=models, benchmarks=benchmarks, min_arm_size=min_arm_size,
            )
        else:
            print(f"  Note: --real requested but inspect CSV not found at {csv_path!r} -- skipping inspect, using openeval only.")

    return [multiarm_corpus_to_source(mc) for mc in corpora]


# ─────────────────────────────────────────────────────────────────────────────
# Inspect AI -- multi-run (nested) real-data sources
#
# Unlike the R=1 loaders above, these keep ALL run indices per item, enabling
# nested-bootstrap / multi-run CI methods (ci_single.py/ci_paired.py
# --nested-mode) to be exercised on real LLM eval data.  Items with fewer
# than ``min_runs`` distinct run indices are excluded.
# ─────────────────────────────────────────────────────────────────────────────

def _load_inspect_item_maps_multirun(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Parse the Inspect CSV into {(model, benchmark): {item_id: run_scores}},
    loading ALL run indices.  run_scores is a 1-D ndarray of scores sorted by
    ascending run_idx."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Inspect data file not found: {csv_path}\n  Run collect_inspect_benchmarks.py first to generate it."
        )

    raw: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    print(f"Loading Inspect AI multi-run data from: {csv_path}")
    n_rows = 0
    with p.open(newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            bench = row.get("benchmark", "").strip()
            model = row.get("model", "").strip()
            item_id = row.get("item_id", "").strip()
            try:
                run_idx = int(row.get("run_idx", 0))
                score = float(row.get("score", float("nan")))
            except (ValueError, TypeError):
                continue
            if not bench or not model or not item_id or not np.isfinite(score):
                continue
            if benchmarks is not None and bench not in benchmarks:
                continue
            if models is not None and model not in models:
                continue
            raw[(model, bench)][item_id][run_idx] = score
            n_rows += 1

    print(f"  {n_rows:,} rows loaded (all run_idx values).")
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, items in raw.items():
        result[key] = {
            item_id: np.array([v for _, v in sorted(runs.items())], dtype=float)
            for item_id, runs in items.items()
        }
    return result


def build_inspect_corpora_multirun(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_corpus_size: int = 50,
    min_runs: int = 2,
) -> list[CISource]:
    """Build multi-run CISources from an Inspect CSV.

    Each source has both ``generate`` (WOR item sample, run_idx=0) and
    ``generate_runs(rng, n, runs)`` (WOR item sample then per-item run
    sampling).  Items with fewer than ``min_runs`` distinct run indices are
    excluded.  When ``runs`` exceeds an item's available run count, the
    remaining columns are bootstrap-resampled from that item's runs.
    """
    raw = _load_inspect_item_maps_multirun(csv_path, models=models, benchmarks=benchmarks)
    if not raw:
        print("  No multi-run data found -- check --benchmarks / --models filters match the CSV.")
        return []

    sources: list[CISource] = []
    for (model, bench), items in sorted(raw.items()):
        eligible = {iid: arr for iid, arr in items.items() if len(arr) >= min_runs}
        if len(eligible) < min_corpus_size:
            n_total = len(items)
            print(f"  Skip  {model}/{bench}: only {len(eligible)}/{n_total} items with >={min_runs} runs (need {min_corpus_size})")
            continue

        item_ids = sorted(eligible.keys())
        run_scores: dict[str, np.ndarray] = {k: eligible[k] for k in item_ids}
        n_items = len(item_ids)
        all_flat = np.concatenate(list(run_scores.values()))
        true_mean = float(np.mean(all_flat))
        avg_runs = float(np.mean([len(v) for v in run_scores.values()]))
        print(f"  OK    {model}/{bench}: {n_items} items, avg_runs={avg_runs:.1f}, mean={true_mean:.4f}")

        def _make_generate(item_ids=item_ids, run_scores=run_scores, n_items=n_items):
            def _generate(rng: np.random.Generator, n: int) -> np.ndarray:
                idxs = rng.choice(n_items, size=n, replace=False)
                return np.array([run_scores[item_ids[i]][0] for i in idxs], dtype=float)
            return _generate

        def _make_generate_runs(item_ids=item_ids, run_scores=run_scores, n_items=n_items):
            def _generate_runs(rng: np.random.Generator, n: int, runs: int) -> np.ndarray:
                idxs = rng.choice(n_items, size=n, replace=False)
                out = np.empty((n, runs), dtype=float)
                for row_i, item_idx in enumerate(idxs):
                    arr = run_scores[item_ids[item_idx]]
                    R = len(arr)
                    if runs <= R:
                        sel = rng.choice(R, size=runs, replace=False)
                    else:
                        sel = rng.choice(R, size=runs, replace=True)
                    out[row_i] = arr[sel]
                return out
            return _generate_runs

        sources.append(CISource(
            label=f"{model}/{bench}",
            eval_type="binary",
            true_mean=true_mean,
            generate=_make_generate(),
            generate_runs=_make_generate_runs(),
            source="inspect",
            max_n=n_items,
            model=model,
            benchmark_id=bench,
        ))

    print(f"\n  {len(sources)} multi-run corpora loaded from Inspect AI data.\n")
    return sources


def build_inspect_corpus_pairs_multirun(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_pair_size: int = 50,
    min_runs: int = 2,
) -> list[CIPairSource]:
    """Build multi-run CIPairSources (one per model-pair on each benchmark).

    ``generate_pair(rng, n, runs)`` returns ``(a, b)``, each shape ``(n,
    runs)``: WOR item sampling, then per-item WOR run sampling (same sampled
    run indices used for both models to preserve within-item correlation).
    """
    raw = _load_inspect_item_maps_multirun(csv_path, models=models, benchmarks=benchmarks)
    if not raw:
        print("  No multi-run data found -- check --benchmarks / --models filters match the CSV.")
        return []

    all_benches = sorted({b for _, b in raw.keys()})
    pair_sources: list[CIPairSource] = []

    for bench in all_benches:
        bench_models = sorted(m for m, b in raw.keys() if b == bench)
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: only {len(bench_models)} model(s) -- need >= 2")
            continue
        print(f"\n  Benchmark: {bench}")

        for model_a, model_b in combinations(bench_models, 2):
            items_a = raw[(model_a, bench)]
            items_b = raw[(model_b, bench)]
            shared_ids = sorted(
                iid for iid in items_a.keys() & items_b.keys()
                if len(items_a[iid]) >= min_runs and len(items_b[iid]) >= min_runs
            )
            if len(shared_ids) < min_pair_size:
                print(f"  Skip  ({model_a} vs {model_b}): {len(shared_ids)} shared items with >={min_runs} runs (need {min_pair_size})")
                continue

            runs_a = {iid: items_a[iid] for iid in shared_ids}
            runs_b = {iid: items_b[iid] for iid in shared_ids}
            n_items = len(shared_ids)

            all_a = np.concatenate(list(runs_a.values()))
            all_b = np.concatenate(list(runs_b.values()))
            true_diff = float(np.mean(all_a) - np.mean(all_b))
            short_a = model_a.split("/")[-1] if "/" in model_a else model_a
            short_b = model_b.split("/")[-1] if "/" in model_b else model_b
            print(
                f"  Pair  ({short_a} vs {short_b}): N={n_items}, "
                f"mean_A={float(np.mean(all_a)):.4f}, mean_B={float(np.mean(all_b)):.4f}, true_diff={true_diff:+.4f}"
            )

            def _make_generate_pair(shared_ids=shared_ids, runs_a=runs_a, runs_b=runs_b, n_items=n_items):
                def _generate_pair(rng: np.random.Generator, n: int, runs: int):
                    idxs = rng.choice(n_items, size=n, replace=False)
                    out_a = np.empty((n, runs), dtype=float)
                    out_b = np.empty((n, runs), dtype=float)
                    for row_i, item_idx in enumerate(idxs):
                        iid = shared_ids[item_idx]
                        arr_a = runs_a[iid]
                        arr_b = runs_b[iid]
                        Ra, Rb = len(arr_a), len(arr_b)
                        min_R = min(Ra, Rb)
                        if runs <= min_R:
                            sel = rng.choice(min_R, size=runs, replace=False)
                        else:
                            sel = rng.choice(min_R, size=runs, replace=True)
                        out_a[row_i] = arr_a[sel]
                        out_b[row_i] = arr_b[sel]
                    return out_a, out_b
                return _generate_pair

            pair_sources.append(CIPairSource(
                label=f"{model_a} vs {model_b}/{bench}",
                eval_type="binary",
                true_diff=true_diff,
                generate_pair=_make_generate_pair(),
                source="inspect",
                max_n=n_items,
                is_null=False,
                model_a=model_a,
                model_b=model_b,
                benchmark_id=bench,
            ))

    print(f"\n  {len(pair_sources)} multi-run corpus pairs built from Inspect AI data.\n")
    return pair_sources


def build_real_data_sources_nested(
    csv_path: str,
    *,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    min_corpus_size: int = 50,
    min_runs: int = 2,
) -> list[CISource]:
    """Multi-run CISources from the inspect CSV (nested-mode analogue of
    build_real_data_sources).  Only 'inspect' data supports multi-run."""
    return build_inspect_corpora_multirun(
        csv_path, models=models, benchmarks=benchmarks,
        min_corpus_size=min_corpus_size, min_runs=min_runs,
    )


def build_real_pair_sources_nested(
    csv_path: str,
    *,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    min_pair_size: int = 50,
    min_runs: int = 2,
) -> list[CIPairSource]:
    """Multi-run CIPairSources from the inspect CSV (nested-mode analogue of
    build_real_pair_sources).  Only 'inspect' data supports multi-run."""
    return build_inspect_corpus_pairs_multirun(
        csv_path, models=models, benchmarks=benchmarks,
        min_pair_size=min_pair_size, min_runs=min_runs,
    )


def build_wmt_da_paired_corpus_pair(*, data_dir: str = "simulations/out") -> CorpusPair | None:
    """Build a real, continuous CorpusPair from wmt_da_paired's human-only
    Direct Assessment scores (``collect_judge_bias_data.py collect-data
    --types continuous_paired``) -- the harness's first real continuous
    PAIRED data source, and the first real-data pair source that needs no
    OpenEval/Inspect model-vs-model comparison at all.

    Deliberately does NOT reuse real_judge_bias.py's
    load_real_wmt_paired_corpus: that function requires the MERGED csv
    (human labels + judge scores), i.e. also running collect-judge-scores,
    since it feeds cases/ppi_real.py's judge-vs-human bias check. This
    reads the plain ``_items.csv`` collect-data alone produces -- no LLM
    judge calls, no API key -- since ci_paired.py's real-data coverage
    sweep only ever needs the real human ground truth, never a judge's
    estimate of it.

    Each of the corpus's segments has two different MT systems' real human
    DA scores -- system "A"/"B" per segment is an arbitrary but fixed
    labeling (the two most-different-scoring systems for that segment, not
    "the better one"; see fetch_wmt_da_paired_items's docstring in
    collect_judge_bias_data.py) -- but that's exactly what CorpusPair
    already models: a paired-by-position (scores_a[i], scores_b[i]), with
    true_diff = mean(scores_a - scores_b) as the real population target,
    regardless of whether "a"/"b" name the same two systems throughout.

    Returns None (with a print explaining how to fix it) if the items CSV
    hasn't been collected yet, mirroring build_inspect_corpora's
    missing-CSV handling.
    """
    from .real_judge_bias import _ITEM_ID_PAIRED_RE, WMT_PAIRED_BOUNDS

    path = Path(data_dir) / "judge_bias_wmt_da_paired_items.csv"
    if not path.exists():
        print(
            f"  Note: {path} not found -- run\n"
            f"    python -m simulations.collect_judge_bias_data collect-data --types continuous_paired\n"
            f"  first (no LLM API key needed; human labels only)."
        )
        return None

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    by_seg: dict[int, dict[int, float]] = defaultdict(dict)
    for r in rows:
        m = _ITEM_ID_PAIRED_RE.match(r["item_id"])
        if m is None:
            continue
        by_seg[int(m.group(1))][int(m.group(2))] = float(r["human_label"])

    aligned_segs = sorted(seg for seg, sides in by_seg.items() if 0 in sides and 1 in sides)
    if not aligned_segs:
        print(f"  Note: {path} has no segments with both sides present -- nothing to build.")
        return None

    lo, hi = WMT_PAIRED_BOUNDS
    scores_a = np.array([(by_seg[s][0] - lo) / (hi - lo) for s in aligned_segs])
    scores_b = np.array([(by_seg[s][1] - lo) / (hi - lo) for s in aligned_segs])
    true_diff = float(np.mean(scores_a - scores_b))
    print(
        f"  wmt_da_paired: N={len(aligned_segs)} aligned segments, "
        f"mean_A={np.mean(scores_a):.4f}, mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
    )
    return CorpusPair(
        model_a="sysA", model_b="sysB", benchmark_id="wmt_da_paired", source="wmt_da_paired",
        scores_a=scores_a, scores_b=scores_b, true_diff=true_diff,
        corpus_size=len(aligned_segs), eval_type="continuous",
    )


def build_real_pair_sources(
    source: str,
    *,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
    inspect_csv: str | None = None,
    include_null: bool = False,
) -> list[CIPairSource]:
    """Resolve (model, benchmark) pairs for `source` and return them as CIPairSources.

    `source` is one of "openeval", "inspect", "wmt_da_paired", or "real"
    ("real" combines "openeval", "inspect", and "wmt_da_paired" for maximum
    real-data diversity, skipping any of the latter two with a note rather
    than failing if their data isn't collected locally). "wmt_da_paired" is
    real, continuous, human-only paired data (see
    build_wmt_da_paired_corpus_pair) -- unlike "openeval"/"inspect", it has
    no `benchmarks`/`models` filtering (there's exactly one benchmark and
    no named models to filter by); its permutation-null variant (for
    include_null=True) reuses corpus_pair_to_null_ci_pair_source the same
    as every other CorpusPair, since that construction doesn't depend on
    model identity. R=1 only -- see module docstring.

    include_null : bool
        If True, also emit a permutation-null CIPairSource (see
        corpus_pair_to_null_ci_pair_source) for every corpus pair, doubling
        the source count. Needed for any Type-I error calibration check on
        real data -- without it there are no is_null=True rows to measure
        Type-I against (real A-vs-B pairs have a genuine, generally
        nonzero, true difference by construction).
    """
    if source not in PAIR_SOURCES:
        raise ValueError(f"Unknown real-data pair source: {source!r}. Choices: {PAIR_SOURCES}")

    def _filter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = pairs
        if benchmarks:
            out = [(m, b) for m, b in out if b in benchmarks]
        if models:
            out = [(m, b) for m, b in out if m in models]
        return out

    corpus_pairs: list[CorpusPair] = []
    if source in ("openeval", "real"):
        corpus_pairs += build_openeval_corpus_pairs(
            _filter_pairs(OPENEVAL_PAIR_DEFAULT_MODEL_BENCH),
            hf_token=hf_token, cache_dir=cache_dir, min_pair_size=min_pair_size,
        )
    if source == "inspect":
        corpus_pairs += build_inspect_corpus_pairs(
            inspect_csv or DEFAULT_INSPECT_CSV, models=models, benchmarks=benchmarks, min_pair_size=min_pair_size,
        )
    if source == "real":
        csv_path = inspect_csv or DEFAULT_INSPECT_CSV
        if Path(csv_path).exists():
            corpus_pairs += build_inspect_corpus_pairs(
                csv_path, models=models, benchmarks=benchmarks, min_pair_size=min_pair_size,
            )
        else:
            print(f"  Note: --real requested but inspect CSV not found at {csv_path!r} -- skipping inspect, using openeval only.")
    if source in ("wmt_da_paired", "real") and (not benchmarks or "wmt_da_paired" in benchmarks):
        wmt_pair = build_wmt_da_paired_corpus_pair()
        if wmt_pair is not None and wmt_pair.corpus_size >= min_pair_size:
            corpus_pairs.append(wmt_pair)

    sources = [corpus_pair_to_ci_pair_source(cp) for cp in corpus_pairs]
    if include_null:
        sources += [corpus_pair_to_null_ci_pair_source(cp) for cp in corpus_pairs]
    return sources
