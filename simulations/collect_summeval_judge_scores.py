#!/usr/bin/env python3
"""SummEval judge-score collector: real multi-system summarization data with
expert human ratings, scored by LLM judges via OpenRouter (or Ollama).

SummEval (Fabbri et al. 2021) has 100 CNN/DailyMail articles, each summarized
by 16 named systems (LEAD-3, BART, Pegasus, T5, ...), each summary rated 1-5
by 3 expert annotators on coherence, consistency, fluency and relevance. Every
system covers every article, so the data is genuinely within-item paired,
with named AI systems as the conditions.

Sources (both public, fetched on first run and cached in simulations/out/):
  * expert annotations + decoded summaries: the SummEval release file
    model_annotations.aligned.jsonl (Yale-LILY/SummEval on GitHub)
  * article text: the Hugging Face mirror mteb/summeval, joined on doc id

Writes to its own files, never to the judge_bias_*.csv files the simulation
harness reads:
  simulations/out/summeval_items.csv         one row per (article, system)
  simulations/out/summeval_judge_scores.csv  one row per (item, judge, dimension, run)
  simulations/out/summeval_judged.csv        merged view with human_label

Setup:
    pip install openai datasets   # if not already installed
    export OPENROUTER_API_KEY=...

Usage:
    # 1. Build the items file (downloads ~6 MB; no API key needed):
    python -m simulations.collect_summeval_judge_scores items

    # 2. Score every (article, system) with one judge on coherence:
    python -m simulations.collect_summeval_judge_scores judge \
        --models anthropic/claude-haiku-4.5

    # Several judges, all four dimensions, only three systems:
    python -m simulations.collect_summeval_judge_scores judge \
        --models anthropic/claude-haiku-4.5 google/gemma-4-26b-a4b-it \
        --dimensions all --systems BART Pegasus T5

    # Smoke test (3 calls):
    python -m simulations.collect_summeval_judge_scores judge \
        --models anthropic/claude-haiku-4.5 --limit 3

The `judge` stage builds the items file itself if it is missing. Re-running
is safe and additive: already-scored (item, judge, dimension, run) combos are
skipped. After each judge finishes, Pearson r and rho^2 against the expert
mean are printed per dimension and per system, so judge alignment can be
checked before any subset is drawn for the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulations.collect_judge_bias_data import (  # noqa: E402
    _make_client, _call_judge, _check_model_available, _progress,
)

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl"
)
ARTICLES_HF_DATASET = "mteb/summeval"

OUT_DIR = Path("simulations/out")
DEFAULT_ANNOTATIONS_CACHE = OUT_DIR / "summeval_model_annotations.aligned.jsonl"
DEFAULT_ITEMS_PATH = OUT_DIR / "summeval_items.csv"
DEFAULT_SCORES_PATH = OUT_DIR / "summeval_judge_scores.csv"
DEFAULT_MERGED_PATH = OUT_DIR / "summeval_judged.csv"

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

# Model ids -> system names, from the SummEval README's model table. Only
# these 16 systems carry expert annotations. model_id is kept in every output
# row so a naming correction never requires re-collection.
SYSTEM_NAMES = {
    "M0": "LEAD-3",
    "M1": "NEUSUM",
    "M2": "BanditSum",
    "M5": "RNES",
    "M8": "Pointer-Generator",
    "M9": "Fast-abs-rl",
    "M10": "Bottom-Up",
    "M11": "Improve-abs",
    "M12": "Unified-ext-abs",
    "M13": "ROUGESal",
    "M14": "Multi-task",
    "M15": "Closed-book-decoder",
    "M17": "T5",
    "M20": "GPT-2",
    "M22": "BART",
    "M23": "Pegasus",
}

ITEMS_FIELDNAMES = [
    "item_id", "doc_id", "model_id", "system", "n_expert_raters",
    "expert_coherence", "expert_consistency", "expert_fluency", "expert_relevance",
    "judge_input",
]
SCORES_FIELDNAMES = [
    "item_id", "judge_model", "dimension", "run_idx", "judge_score", "raw_response", "collected_at",
]
MERGED_FIELDNAMES = [
    "item_id", "doc_id", "model_id", "system", "dimension", "human_label",
    "judge_model", "run_idx", "judge_score",
]

# Dimension definitions follow the SummEval annotation guidelines
# (Fabbri et al. 2021, Sec. 4.1); the judge sees the same definition the
# expert annotators were given.
DIMENSION_DEFINITIONS = {
    "coherence": (
        "Coherence: the collective quality of all sentences. The summary should be "
        "well-structured and well-organized. It should not just be a heap of related "
        "information, but should build from sentence to sentence into a coherent body "
        "of information about the topic."
    ),
    "consistency": (
        "Consistency: the factual alignment between the summary and the source article. "
        "A consistent summary contains only statements that are entailed by the source "
        "article. Penalize summaries that contain hallucinated facts."
    ),
    "fluency": (
        "Fluency: the quality of individual sentences. Sentences should have no formatting "
        "problems, capitalization errors, or obviously ungrammatical sentences (e.g., "
        "fragments, missing components) that make the text difficult to read."
    ),
    "relevance": (
        "Relevance: selection of important content from the source. The summary should "
        "include only important information from the source article. Penalize summaries "
        "that contain redundancies and excess information."
    ),
}


# ---------------------------------------------------------------------------
# CSV helpers (same conventions as the other collectors)
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# items: SummEval annotations + article text -> summeval_items.csv
# ---------------------------------------------------------------------------


def _download_annotations(cache: Path) -> Path:
    if cache.exists() and cache.stat().st_size > 0:
        print(f"Using cached annotations {cache}")
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ANNOTATIONS_URL} ...")
    with urllib.request.urlopen(ANNOTATIONS_URL, timeout=120) as resp, cache.open("wb") as f:
        f.write(resp.read())
    print(f"  -> {cache} ({cache.stat().st_size / 1e6:.1f} MB)")
    return cache


def _load_articles() -> dict[str, str]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    print(f"Loading article text from {ARTICLES_HF_DATASET} ...")
    ds = load_dataset(ARTICLES_HF_DATASET, split="test")
    return {row["id"]: row["text"] for row in ds}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def build_items(*, annotations_cache: Path, items_path: Path) -> list[dict]:
    ann_path = _download_annotations(annotations_cache)
    articles = _load_articles()

    items: list[dict] = []
    n_missing_article = n_unknown_model = 0
    with ann_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            model_id = r["model_id"]
            if model_id not in SYSTEM_NAMES:
                n_unknown_model += 1
                continue
            doc_id = r["id"]
            article = articles.get(doc_id)
            if article is None:
                n_missing_article += 1
                continue
            experts = r["expert_annotations"]
            row = {
                "item_id": f"summeval_{doc_id}_{model_id}",
                "doc_id": doc_id,
                "model_id": model_id,
                "system": SYSTEM_NAMES[model_id],
                "n_expert_raters": len(experts),
            }
            for dim in DIMENSIONS:
                row[f"expert_{dim}"] = round(_mean([float(e[dim]) for e in experts]), 4)
            row["judge_input"] = json.dumps({"article": article, "summary": r["decoded"]})
            items.append(row)

    items.sort(key=lambda x: (x["doc_id"], int(x["model_id"][1:])))
    _write_csv(items_path, items, ITEMS_FIELDNAMES)
    n_docs = len({it["doc_id"] for it in items})
    n_sys = len({it["system"] for it in items})
    print(f"Items -> {items_path}: {len(items)} (article, system) rows, "
          f"{n_docs} articles x {n_sys} systems.")
    if n_missing_article:
        print(f"  WARNING: {n_missing_article} annotation rows had no article text in "
              f"{ARTICLES_HF_DATASET} and were dropped.")
    if n_unknown_model:
        print(f"  ({n_unknown_model} rows for model ids outside the 16 expert-annotated systems skipped.)")
    return items


# ---------------------------------------------------------------------------
# judge: prompt, parse, collect
# ---------------------------------------------------------------------------


def _build_summeval_prompt(ji: dict, dimension: str) -> list[dict]:
    definition = DIMENSION_DEFINITIONS[dimension]
    return [
        {"role": "system", "content": (
            "You are an expert evaluator of news summaries. You will be given a news article "
            "and a machine-generated summary of it. Rate the summary on ONE dimension, "
            "using a scale from 1 (worst) to 5 (best).\n\n"
            f"{definition}\n\n"
            "Respond with ONLY a single integer from 1 to 5."
        )},
        {"role": "user", "content": (
            f"Article:\n{ji['article']}\n\n"
            f"Summary:\n{ji['summary']}\n\n"
            f"{dimension.capitalize()} score (1-5):"
        )},
    ]


def _parse_summeval_response(text: str) -> float | None:
    m = re.search(r"[1-5]", text)
    return None if m is None else float(m.group(0))


def _judge_one(client, model, backend, max_retries, sleep_s, item, dimension, run_idx):
    ji = json.loads(item["judge_input"])
    messages = _build_summeval_prompt(ji, dimension)
    raw = _call_judge(client, model, messages, backend=backend, max_retries=max_retries, sleep_s=sleep_s)
    score = None if raw is None else _parse_summeval_response(raw)
    return item, dimension, run_idx, raw, score


def _parse_model_backends(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--model-backends entries must be MODEL=BACKEND, got {p!r}")
        model, backend = p.split("=", 1)
        out[model] = backend
    return out


def _resolve_systems(requested: list[str] | None) -> set[str] | None:
    """Accept system names or model ids; None means all 16."""
    if not requested:
        return None
    by_name = {v.lower(): v for v in SYSTEM_NAMES.values()}
    out = set()
    for s in requested:
        if s in SYSTEM_NAMES:
            out.add(SYSTEM_NAMES[s])
        elif s.lower() in by_name:
            out.add(by_name[s.lower()])
        else:
            raise SystemExit(f"Unknown system {s!r}. Known: {sorted(SYSTEM_NAMES.values())} or ids {sorted(SYSTEM_NAMES)}")
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5


def _print_alignment(items_by_id: dict[str, dict], scores: list[dict], model: str) -> None:
    """Pearson r and rho^2 of judge_score against the expert mean, per
    dimension overall and per system, for one judge."""
    per_dim: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    per_dim_sys: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for s in scores:
        if s["judge_model"] != model:
            continue
        it = items_by_id.get(s["item_id"])
        if it is None:
            continue
        dim = s["dimension"]
        h, j = float(it[f"expert_{dim}"]), float(s["judge_score"])
        per_dim[dim][0].append(h); per_dim[dim][1].append(j)
        per_dim_sys[(dim, it["system"])][0].append(h); per_dim_sys[(dim, it["system"])][1].append(j)
    if not per_dim:
        return
    print(f"  Alignment of {model} against the expert mean:")
    for dim in DIMENSIONS:
        if dim not in per_dim:
            continue
        h, j = per_dim[dim]
        r = _pearson(h, j)
        r_txt = "n/a" if r is None else f"r={r:+.2f}  rho^2={r * r:.2f}"
        print(f"    {dim:<12} n={len(h):<5} {r_txt}")
        for (d, system), (hs, js) in sorted(per_dim_sys.items()):
            if d != dim:
                continue
            rs = _pearson(hs, js)
            rs_txt = "n/a" if rs is None else f"r={rs:+.2f}"
            print(f"      {system:<22} n={len(hs):<4} expert mean={_mean(hs):.2f}  judge mean={_mean(js):.2f}  {rs_txt}")


def run_judge(args: argparse.Namespace) -> None:
    items_path, scores_path, merged_path = Path(args.items), Path(args.scores_out), Path(args.merged_out)
    model_backends = _parse_model_backends(args.model_backends)
    dimensions = DIMENSIONS if args.dimensions == ["all"] else args.dimensions
    for d in dimensions:
        if d not in DIMENSIONS:
            raise SystemExit(f"Unknown dimension {d!r}; choose from {DIMENSIONS} or 'all'.")
    systems = _resolve_systems(args.systems)

    items = _read_csv(items_path)
    if not items:
        print(f"No items at {items_path}; building them first.")
        items = build_items(annotations_cache=Path(args.annotations_cache), items_path=items_path)
        print()
    if systems is not None:
        items = [it for it in items if it["system"] in systems]
    print(f"Loaded {len(items)} (article, system) items from {items_path}"
          + (f" restricted to systems {sorted(systems)}" if systems else ""))
    print(f"Dimensions: {dimensions}")
    print(f"Scores -> {scores_path}")
    print()

    existing_scores = _read_csv(scores_path)
    done_keys = {(r["item_id"], r["judge_model"], r["dimension"], r["run_idx"]) for r in existing_scores}
    write_lock = threading.Lock()

    clients: dict[str, object] = {}
    for model_i, model in enumerate(args.models, start=1):
        backend = model_backends.get(model, args.backend)
        print(f"{'=' * 72}\n[{model_i}/{len(args.models)}] model={model!r} backend={backend!r}\n{'=' * 72}")

        if backend not in clients:
            try:
                clients[backend] = _make_client(backend)
            except SystemExit as e:
                print(f"  SKIPPING backend {backend!r}: {e}")
                continue
        client = clients[backend]

        err = _check_model_available(client, model, backend)
        if err is not None:
            print(f"  SKIPPING {model!r} -- not reachable: {err}")
            continue

        work: list[tuple[dict, str, int]] = []
        n_skipped = 0
        for item in items:
            for dim in dimensions:
                for run_idx in range(args.runs):
                    if (item["item_id"], model, dim, str(run_idx)) in done_keys:
                        n_skipped += 1
                        continue
                    work.append((item, dim, run_idx))
        if args.limit is not None:
            work = work[:args.limit]

        print(f"{len(items)} items x {len(dimensions)} dimensions x runs={args.runs}: "
              f"{n_skipped} combos already done, {len(work)} to collect, concurrency={args.concurrency}.")

        n_new = n_failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_judge_one, client, model, backend, args.max_retries, args.sleep, item, dim, run_idx)
                for item, dim, run_idx in work
            ]
            pbar = _progress(as_completed(futures), total=len(futures), desc=model, unit="call")
            for fut in pbar:
                try:
                    item, dim, run_idx, raw, score = fut.result()
                except Exception:  # noqa: BLE001 -- one bad item must not abort the run
                    n_failed += 1
                    continue
                if score is None:
                    n_failed += 1
                    msg = f"    could not parse response for {item['item_id']} [{dim}] run {run_idx}: {raw!r}"
                    pbar.write(msg) if hasattr(pbar, "write") else print(msg)
                    continue
                row = {
                    "item_id": item["item_id"], "judge_model": model, "dimension": dim, "run_idx": run_idx,
                    "judge_score": score, "raw_response": raw,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                with write_lock:
                    _append_csv_row(scores_path, row, SCORES_FIELDNAMES)
                n_new += 1
            if hasattr(pbar, "close"):
                pbar.close()

        print(f"  -> {n_new} new scores written ({n_skipped} already done, {n_failed} failed/unparseable)")
        _print_alignment({it["item_id"]: it for it in _read_csv(items_path)}, _read_csv(scores_path), model)
        print()

    _write_merged_view(items_path, scores_path, merged_path)


def _write_merged_view(items_path: Path, scores_path: Path, merged_path: Path) -> None:
    items_by_id = {r["item_id"]: r for r in _read_csv(items_path)}
    merged = []
    for s in _read_csv(scores_path):
        it = items_by_id.get(s["item_id"])
        if it is None:
            continue
        merged.append({
            "item_id": s["item_id"], "doc_id": it["doc_id"], "model_id": it["model_id"],
            "system": it["system"], "dimension": s["dimension"],
            "human_label": it[f"expert_{s['dimension']}"],
            "judge_model": s["judge_model"], "run_idx": s["run_idx"], "judge_score": s["judge_score"],
        })
    _write_csv(merged_path, merged, MERGED_FIELDNAMES)
    print(f"Merged view -> {merged_path}: {len(merged)} (item, judge, dimension, run) rows.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_items = sub.add_parser("items", help="Download SummEval and build the items CSV.")
    p_items.add_argument("--annotations-cache", default=str(DEFAULT_ANNOTATIONS_CACHE))
    p_items.add_argument("--items", default=str(DEFAULT_ITEMS_PATH))

    p_judge = sub.add_parser("judge", help="Score items with LLM judges.")
    p_judge.add_argument("--annotations-cache", default=str(DEFAULT_ANNOTATIONS_CACHE))
    p_judge.add_argument("--items", default=str(DEFAULT_ITEMS_PATH))
    p_judge.add_argument("--scores-out", default=str(DEFAULT_SCORES_PATH))
    p_judge.add_argument("--merged-out", default=str(DEFAULT_MERGED_PATH))
    p_judge.add_argument("--models", nargs="+", required=True)
    p_judge.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter",
                         help="Default backend for any model not listed in --model-backends.")
    p_judge.add_argument("--model-backends", nargs="+", default=None,
                         help="Per-model backend overrides as MODEL=BACKEND.")
    p_judge.add_argument("--dimensions", nargs="+", default=["coherence"],
                         help=f"Any of {DIMENSIONS}, or 'all'. Default: coherence.")
    p_judge.add_argument("--systems", nargs="+", default=None,
                         help="Restrict to these systems (names or M-ids). Default: all 16.")
    p_judge.add_argument("--runs", type=int, default=1)
    p_judge.add_argument("--limit", type=int, default=None,
                         help="Cap total (item, dimension, run) calls per model, for a smoke test.")
    p_judge.add_argument("--concurrency", type=int, default=4)
    p_judge.add_argument("--max-retries", type=int, default=3)
    p_judge.add_argument("--sleep", type=float, default=0.0, help="Delay after each successful call.")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "items":
        build_items(annotations_cache=Path(args.annotations_cache), items_path=Path(args.items))
    elif args.cmd == "judge":
        run_judge(args)


if __name__ == "__main__":
    main()
