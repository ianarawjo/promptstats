#!/usr/bin/env python3
"""Figure for the small-n summarization demonstration (Sec. 8.1).

One 15-item eval set drawn from the 500-item XSum ROUGE-L corpus, run through
compare() exactly as the demonstration describes. Seed 0, not selected: the
first draw, so the figure is not a favourable pick.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evalstats as es

TRIO = ["meta-llama/llama-3.1-8b-instruct",
        "google/gemma-3-12b-it",
        "mistralai/ministral-3b-2512"]
SHORT = dict(zip(TRIO, ["llama-3.1-8b", "gemma-3-12b", "ministral-3b"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="simulations/out/paper_overleaf_src/media/"
                                     "simulations/scenario1_rouge_smalln.png")
    a = ap.parse_args()

    d = pd.read_csv("simulations/out/summarization_rouge.csv")
    w = d.pivot(index="item_id", columns="model", values="rouge_l")[TRIO]
    rng = np.random.default_rng(a.seed)
    items = w.index.to_numpy()[rng.choice(len(w), a.n, replace=False)]
    sub = (w.loc[items].reset_index()
             .melt(id_vars="item_id", var_name="model", value_name="score")
             .rename(columns={"item_id": "item"}))
    sub["model"] = sub.model.map(SHORT)

    r = es.compare(es.load_from(sub), factors="model", score_range=(0, 1),
                   rng=np.random.default_rng(a.seed))
    r.summary()
    r.plot(title=f"Brief quality (ROUGE-L), {a.n} postings")
    # The forest plot hardcodes "Accuracy (%)" for [0,1] scores; this metric is
    # ROUGE-L, so relabel rather than mislabel it.
    plt.gca().set_xlabel("ROUGE-L", fontsize=13)
    plt.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
