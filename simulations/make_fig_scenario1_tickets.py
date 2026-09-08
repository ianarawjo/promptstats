#!/usr/bin/env python3
"""Figure for the small-n prompt-comparison demonstration (Sec. 8.1).

One 20-ticket eval set drawn from the 120-ticket support-ticket corpus, all 8
prompts, run through compare() as the demonstration describes. Seed 0.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evalstats as es

CSV = "website/notebooks/support_ticket_eval_multirun.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--out", default="simulations/out/paper_overleaf_src/media/"
                                     "simulations/scenario1_tickets20.png")
    a = ap.parse_args()

    d = pd.read_csv(CSV)
    d = d[d.run_idx == a.run]
    w = d.pivot(index="input_id", columns="prompt_id", values="correct")
    rng = np.random.default_rng(a.seed)
    items = w.index.to_numpy()[rng.choice(len(w), a.n, replace=False)]
    sub = (w.loc[items].reset_index()
             .melt(id_vars="input_id", var_name="prompt", value_name="score")
             .rename(columns={"input_id": "item"}))
    # Drop the P#_ ordering prefix; it costs label width and says nothing.
    sub["prompt"] = sub["prompt"].str.replace(r"^P\d_", "", regex=True)

    r = es.compare(es.load_from(sub), factors="prompt", score_range=(0, 1),
                   rng=np.random.default_rng(a.seed))
    r.summary()
    # 8 rows render small at \demowidth in the paper, so scale the type up and
    # widen the aspect ratio rather than giving the figure a full column.
    r.plot(title=f"Ticket classifier accuracy, {a.n} tickets",
           figsize=(11.0, 5.4), font_scale=1.7)
    plt.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
