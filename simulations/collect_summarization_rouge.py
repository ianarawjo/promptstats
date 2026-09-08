#!/usr/bin/env python3
"""
collect_summarization_rouge.py — Score several models on the SAME summarization
items via OpenRouter and save per-item ROUGE-L to a CSV.

Why this exists
---------------
The existing real-data corpora cannot support a paired, CONTINUOUS model
comparison:

  * ``inspect_benchmarks.csv`` has shared items and six models, but every score
    is binary -- ``collect_inspect_benchmarks.py``'s ``_to_binary`` rounds any
    partial credit to 0/1 before writing, and none of the 132 ``inspect_evals``
    tasks is a summarization task in the first place.
  * OpenEval carries the continuous metrics (ROUGE-L on cnndm/xsum, BLEU-max on
    truthfulqa) but pairs ONE benchmark with ONE model, so there are no shared
    items to difference across.
  * The judge-bias corpora are continuous and shared-item, but the things being
    compared are judges, not the system under test.

ROUGE-L is not a new metric for this paper: it is already the continuous
real-data metric of Appendix~A's corpus table ("Continuous metrics: ROUGE-L
(cnndm, xsum), BLEU-max (truthfulqa)"). Token-F1 was considered and rejected --
the Limitations section lists F1 among metrics left to future work, so a
demonstration built on it would contradict the paper's own stated scope.

Output feeds a paired continuous comparison: N models x the same items, one
ROUGE-L in [0,1] per (model, item), with the full item set standing in as
ground truth for a subsample of 15-30.

Install:
  pip install openai datasets rouge-score

Usage:
  export OPENROUTER_API_KEY=...
  python simulations/collect_summarization_rouge.py \
      --models openai/gpt-4o-mini \
               google/gemma-3-12b-it \
               google/gemma-3-4b-it \
      --dataset xsum --limit 500 \
      --output simulations/out/summarization_rouge.csv

  # resume after an interruption (skips finished model/item pairs)
  python simulations/collect_summarization_rouge.py --resume ...

Output CSV columns:
  dataset, model, item_id, rouge_l, rouge_1, gen_chars, ref_chars, n_words_gen
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# One-sentence references (xsum) keep outputs short and cheap, and make ROUGE-L
# more discriminative than the multi-sentence cnn_dailymail highlights; both are
# named in the paper's corpus table, so either is defensible.
DATASETS = {
    "xsum": dict(hf="EdinburghNLP/xsum", split="test", doc="document",
                 ref="summary", sentences=1),
    "cnn_dailymail": dict(hf="abisee/cnn_dailymail", config="3.0.0", split="test",
                          doc="article", ref="highlights", sentences=3),
}

PROMPT = (
    "Summarize the following article in {n} sentence{s}. "
    "Respond with the summary only -- no preamble, no bullet points, no title.\n\n"
    "ARTICLE:\n{doc}"
)


# ---------------------------------------------------------------------------
# Client (OpenAI-compatible; same thin wrapper the judge-bias collector uses)
# ---------------------------------------------------------------------------

def _make_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY must be set.")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(base_url=base, api_key=key)


def _check_ids(models):
    """Validate model ids against OpenRouter's public catalog before spending
    calls. Note a listed id can still 404 with "No endpoints found" when no
    provider currently serves it, so this narrows the failure but does not
    replace the live preflight below."""
    import json, urllib.request
    try:
        raw = json.load(urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=30))["data"]
    except Exception as e:  # noqa: BLE001 -- catalog is a convenience, not a gate
        print(f"  (could not reach the model catalog: {e})")
        return
    ids = {m["id"] for m in raw}
    bad = [m for m in models if m not in ids]
    if bad:
        raise SystemExit(
            "not in the OpenRouter catalog: " + ", ".join(bad) +
            "\nBrowse valid ids at https://openrouter.ai/models")


def _summarize(client, model, doc, n_sent, *, max_retries=4, sleep_s=0.0):
    """One completion, retried with backoff. Returns the summary text or None."""
    msg = [{"role": "user", "content": PROMPT.format(
        n=n_sent, s="" if n_sent == 1 else "s", doc=doc)}]
    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=msg, temperature=0.0, max_tokens=512,
                # Same reason as the judge-bias collector: let reasoning models
                # think, but strip the trace so message.content is the answer
                # rather than an empty string starved by the token budget.
                extra_body={"reasoning": {"exclude": True}},
            )
            if sleep_s:
                time.sleep(sleep_s)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 -- retry any transport/API error
            last = e
            time.sleep(min(2 ** attempt, 10))
    print(f"    WARNING: {model} failed after {max_retries} attempts: {last}")
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _make_scorer():
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise SystemExit("pip install rouge-score")
    # use_stemmer=True is the standard configuration these scores are reported
    # under; without it ROUGE-L is depressed by pure morphology and the
    # between-model spread narrows for no substantive reason.
    return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def _score(scorer, ref: str, gen: str) -> tuple[float, float]:
    s = scorer.score(ref, gen)
    return s["rougeL"].fmeasure, s["rouge1"].fmeasure


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _done_pairs(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["dataset"], r["model"], r["item_id"]) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="OpenRouter model ids, e.g. openai/gpt-4o-mini")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="xsum")
    ap.add_argument("--limit", type=int, default=500,
                    help="items, shared across every model (default 500)")
    ap.add_argument("--output", default="simulations/out/summarization_rouge.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause after each successful call")
    ap.add_argument("--resume", action="store_true",
                    help="skip (dataset, model, item) rows already in --output")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the one-call-per-model reachability check")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed for item selection; fixed so every model "
                         "and every re-run sees the SAME items")
    a = ap.parse_args()

    spec = DATASETS[a.dataset]
    from datasets import load_dataset
    kw = {"name": spec["config"]} if "config" in spec else {}
    ds = load_dataset(spec["hf"], split=spec["split"], **kw)
    # Fixed shuffle then take: the paired comparison is only valid if every
    # model is scored on an identical item set, so selection must not depend on
    # model, run order, or resume state.
    ds = ds.shuffle(seed=a.seed).select(range(min(a.limit, len(ds))))
    id_col = "id" if "id" in ds.column_names else None
    items = [(str(r[id_col]) if id_col else f"{a.dataset}_{i}",
              r[spec["doc"]], r[spec["ref"]]) for i, r in enumerate(ds)]
    print(f"{a.dataset}: {len(items)} items x {len(a.models)} models "
          f"= {len(items)*len(a.models):,} calls")

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size and not a.resume:
        raise SystemExit(f"{out} exists. Pass --resume to continue it, or give a "
                         f"different --output; appending without --resume would "
                         f"duplicate items and silently break the pairing.")
    done = _done_pairs(out) if a.resume else set()
    if done:
        print(f"resuming: {len(done):,} rows already collected")
    client, scorer = _make_client(), _make_scorer()

    # Preflight: a mistyped model id otherwise fails every one of its calls
    # through the full retry ladder, which is slow and easy to miss in the log.
    if not a.no_preflight:
        print("preflight:")
        _check_ids(a.models)
        bad = []
        for model in a.models:
            r = _summarize(client, model, items[0][1], spec["sentences"], max_retries=1)
            print(f"  {model}: {'ok -- ' + r[:60].replace(chr(10), ' ') if r else 'FAILED'}")
            if not r:
                bad.append(model)
        if bad:
            raise SystemExit(f"unreachable model id(s): {bad}. Check them against "
                             f"https://openrouter.ai/models, or pass --no-preflight.")

    need_header = not out.exists() or out.stat().st_size == 0
    f = out.open("a", newline="")
    w = csv.writer(f)
    if need_header:
        w.writerow(["dataset", "model", "item_id", "rouge_l", "rouge_1",
                    "gen_chars", "ref_chars", "n_words_gen"])

    total = 0
    for model in a.models:
        todo = [it for it in items if (a.dataset, model, it[0]) not in done]
        if not todo:
            print(f"  {model}: already complete"); continue
        print(f"  {model}: {len(todo)} calls")
        got = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_summarize, client, model, doc, spec["sentences"],
                              sleep_s=a.sleep): (iid, ref)
                    for iid, doc, ref in todo}
            for fut in as_completed(futs):
                iid, ref = futs[fut]
                gen = fut.result()
                if gen is None or not gen.strip():
                    continue
                rl, r1 = _score(scorer, ref, gen)
                w.writerow([a.dataset, model, iid, f"{rl:.6f}", f"{r1:.6f}",
                            len(gen), len(ref), len(gen.split())])
                got += 1
                if got % 50 == 0:
                    f.flush(); print(f"    {got}/{len(todo)}")
        total += got
        f.flush()
        print(f"  {model}: {got} scored")
    f.close()
    print(f"\nDone. {total:,} rows appended to {out}")
    print("NOTE: a paired comparison needs every model scored on every item -- "
          "check for stragglers with --resume before analysing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
