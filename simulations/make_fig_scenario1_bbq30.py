#!/usr/bin/env python3
"""Figure for the small-n BBQ demonstration (Sec. 8.1).

One 30-item eval set drawn from the 1,000-item BBQ benchmark, run through
compare() exactly as the demonstration describes. Seed 0, the first draw.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evalstats as es

TRIO = ["openrouter/openai/gpt-4o-mini",
        "openrouter/ibm-granite/granite-4.1-8b",
        "openrouter/google/gemma-3n-e4b-it"]
SHORT = dict(zip(TRIO, ["gpt-4o-mini", "granite-4.1-8b", "gemma-3n-e4b-it"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="simulations/out/paper_overleaf_src/media/"
                                     "simulations/scenario1_bbq30.png")
    a = ap.parse_args()

    d = pd.read_csv("simulations/out/inspect_benchmarks.csv")
    b = d[(d.benchmark == "bbq") & (d.run_idx == 0) & (d.model.isin(TRIO))]
    w = b.pivot(index="item_id", columns="model", values="score")[TRIO]
    rng = np.random.default_rng(a.seed)
    items = w.index.to_numpy()[rng.choice(len(w), a.n, replace=False)]
    sub = (w.loc[items].reset_index()
             .melt(id_vars="item_id", var_name="model", value_name="score")
             .rename(columns={"item_id": "item"}))
    sub["model"] = sub.model.map(SHORT)

    r = es.compare(es.load_from(sub), factors="model", score_range=(0, 1),
                   rng=np.random.default_rng(a.seed))
    r.summary()
    r.plot(title=f"Accuracy on {a.n} BBQ questions")
    plt.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
