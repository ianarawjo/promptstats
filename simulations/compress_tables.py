"""Compress default-rendered LaTeX result tables for in-paper use.

The Monte Carlo harness's ``--latex`` output is deliberately exhaustive:
every method swept, every sample size, synthetic and real runs as
separate tables. That is the right form for a supplement, and this
script does NOT change how the harness renders it. It is a downstream
transform: it reads those already-rendered tables and emits a compact
version for the appendix, where a reviewer needs the recommendations to
be checkable without paging through 40-row tables.

What it does, all of it purely rearrangement:

* copies every cell VERBATIM, including \\cellcolor shading and the
  \\textbf/\\underline best-marks, so no value is ever recomputed here
* folds each table's real-data companion in as a right-hand column
  group, separated by a padded vertical rule
* drops rows for methods no claim rests on (configurable; a row
  carrying a best-mark is never dropped)
* packs paired tables into one float, each keeping its own caption,
  number and label

Only the layout changes. The originals are reproduced untouched by
``--supplement`` for the supplementary document.

Usage::

    # preview the compressed tables on their own
    python simulations/compress_tables.py --merge --standalone --out preview.tex

    # rewrite an appendix in place (writes to OUT, never to --input)
    python simulations/compress_tables.py --merge --apply appendix.new.tex

    # emit the untouched originals as a supplement section
    python simulations/compress_tables.py --supplement s4.tex

NOT idempotent. ``--apply`` consumes default-rendered tables, so re-run
it against fresh harness output or a pristine backup, never against a
file it has already rewritten.

Tuning knobs live in the constants below: which sample sizes to show
(NS_*, REAL_NS_*), which methods to drop (DROP_FAMILY, EXTRA_DROP),
which real metrics to carry (REAL_METRICS), and the layout constants
(TABCOLSEP, RULE_PAD, METHOD_COL_WIDTH, MERGE_GROUPS).
"""

from __future__ import annotations

import argparse
import pathlib
import re

PAPER = pathlib.Path(__file__).resolve().parent / "out" / "paper_overleaf_src"

# n-columns to retain. The single-run sweep is n=10..100 in 11 steps; the
# multi-run sweep is a coarser 10/20/30/50/75/100, so it has no n=15 or
# n=80 and gets the nearest available set instead.
# The stacked layout leaves ~200pt of the 506pt table* width unused, so
# the single-run tables carry the COMPLETE swept n axis rather than a
# subset. n=10 is omitted throughout: evalstats does not support it.
NS_SINGLE = ["15", "30", "50", "80", "100"]
# Trimmed from the full 10-value axis 2026-08-25 to make room for the Type-I
# and Power columns, and to match REAL_NS_SINGLE so the two halves of the
# table line up. Coverage moves smoothly in n, so the dropped values
# (20/40/60/70/90) carry little the neighbouring columns do not; the full
# axis remains in the Supplementary tables.
NS_MULTI = ["20", "30", "50", "75", "100"]

# Real-data coverage is reported at a few n as well as overall: a single
# pooled Real Cov cannot support the per-n claims the captions make
# (logit-t/NIG "drifting further above nominal as n grows"; Tango holding
# nominal "at n>=15"). Stacked tables have the width for it.
# real per-n at the same sample sizes the synthetic subset used;
# REAL_NS_MULTI is the complete multi-run real grid (n=10 aside)
REAL_NS_SINGLE = ["15", "30", "50", "80", "100"]
REAL_NS_MULTI = ["20", "30", "50", "75", "100"]

# The real block mirrors the Overall block rather than reporting coverage
# alone: a reviewer who sees only real coverage will immediately ask what
# happened to width and interval score, and the interval score is the
# metric the recommendations are actually chosen on.
REAL_METRICS = ["Cov", "Width", "Score"]

# Column sets differ by table family. CI tables report coverage/width/
# interval score; the pairwise p-value tables report Type-I error and
# power. "95\% MC band" is dropped throughout: its half-widths are
# ~0.001 and it costs a wide column to say so.
BASE_CI = ["Method", "Cov", "MinCov", "Width", "Pen", "Score", "Type-I", "Power", "Time (ms)"]
# MinCov and Penalty added 2026-08-24. The headline Cov/Score average the
# coverage tail away -- Score is ~90% Width, so the narrowest method wins it
# even while covering worst. MinCov is the worst single (scenario, n) cell and
# Penalty is the score's miss term; both surface that tail. Drop either here
# if the compressed tables need the horizontal space back.
# ci_single has no Type-I / Power: those are pairwise-test quantities, and the
# mean-point-estimate case never computes them. Without this the ci_single
# tables cannot be rebuilt at all, which is why they still lack MinCov and Pen.
BASE_CI_SINGLE = ["Method", "Cov", "MinCov", "Width", "Pen", "Score", "Time (ms)"]
BASE_PV = ["Method", "Type-I error", "Mean power"]
REAL_PV = ["Type-I error", "Mean power"]
NS_PAIRWISE = ["20", "30", "50", "75", "100"]
REAL_NS_PAIRWISE = ["20", "50", "100"]

# Rows kept for the one table that gets row-trimmed (option D). Every
# method named in a recommendation bullet, every popular default whose
# failure the text relies on, and every row carrying a best/runner-up
# mark (dropping a marked row would leave the table with no bold cell).
# Resampling methods are ~half of every block and no bootstrap-family row
# carries a best/runner-up mark, so culling them never leaves a table
# without its bold cell. We keep two representatives everywhere, for
# consistency across tables: the plain percentile bootstrap (what people
# actually reach for) and smooth_bootstrap (the conservative alternative
# the recommendations name). Nested/dithered families likewise keep one.
# A row carrying \textbf or \underline is never dropped.
# bca, bayes_bootstrap and bootstrap_t are deliberately NOT culled: the paper
# makes claims about bootstrap calibration, so the rows those claims rest on
# have to be visible in the appendix rather than only in the supplement. The
# MinCov column is where they fail hardest (~.16 on binary against Wilson's
# .908), which is invisible if the rows are dropped.
DROP_FAMILY = {
    "bca_nested", "bayes_bootstrap_nested", "bootstrap_t_nested",
    "smooth_bootstrap_nested",
    "bayes_diff_nested", "smooth_diff_nested",
}

# Extra per-table drops the family rule doesn't cover: the multi-run
# marginal table sweeps six overdispersion variants no claim references.
EXTRA_DROP = {
    "ci_multirun": {"wilson_od", "wilson_od_bc", "wilson_od_t", "jeffreys_od",
                    "cp_od", "clopper_pearson_flat", "bayes_indep_flat"},
}

MULTIRUN_KEEP_UNUSED = {
    ("t_interval", "bin"), ("nig", "bin"), ("wilson_flat", "bin"),
    ("wald_flat", "bin"), ("bb_bayes", "bin"), ("bb_bayes_robust", "bin"),
    ("logit_t", "cont"), ("nig", "cont"), ("t_interval", "cont"),
    ("bootstrap", "cont"), ("bootstrap_nested", "cont"), ("bca_nested", "cont"),
    ("logit_t", "lik"), ("nig", "lik"), ("t_interval", "lik"),
    ("bootstrap", "lik"), ("bootstrap_nested", "lik"), ("bca_nested", "lik"),
}

# ---- width controls -------------------------------------------------
# Two tables side by side inside a table* need each to fit in about
# 248pt (textwidth 506pt, less a gap). The compressed tables start at
# 350-370pt, and ~132pt of that is inter-column padding at the 6pt
# default, so tabcolsep is the biggest single lever.
TABCOLSEP = "2pt"

# Separator before the real-data block. "!{}" inserts the rule while
# KEEPING the surrounding \tabcolsep (a bare "|" leaves it flush against
# the adjacent digits at a 2pt sep, which reads as cramped), so the gap is
# tabcolsep + RULE_PAD on each side.
RULE_PAD = "4pt"
RULE_SPEC = r"!{\hspace{" + RULE_PAD + r"}\vrule\hspace{" + RULE_PAD + "}}"
PROBE_DROP: set[str] = set()

# Wrapping the method column spends vertical space (measured as free in
# this document) to buy horizontal space, which side-by-side needs.
METHOD_COL_WIDTH = None   # stacked layout has width to spare; no wrapping needed
SHORT_REAL_HEADER = True

# Time is not a claim-bearing quantity anywhere in the paper, so it is
# rounded and its header shortened. Coverage/width/score keep full
# printed precision -- only the redundant leading zero is dropped, which
# changes how a value is typeset, not the value.
SHORTEN_TIME = True
STRIP_LEADING_ZERO = True

BLOCK_NAME = {"bin": "Binary", "cont": "Continuous", "lik": "Likert"}

# Captions merge the original synthetic caption with the substantive claims
# from the real-data caption the "Real Cov" column absorbs. Claims that rely
# on the per-n real trajectory (which one column cannot show) are kept but
# redirected to the supplementary table.
# The boilerplate that used to repeat in all four captions (what the
# per-n columns mean, what Real Cov means, where the full tables live)
# now appears ONCE in the appendix prose, per PROSE_NOTE below. Captions
# carry only what the table shows and what it demonstrates.
PROSE_NOTE = (
    r"Each table below reports overall performance and coverage at every swept sample "
    r"size on synthetic data ($n{=}10$ excepted, which \evalstats{} does not support), "
    r"and, to the right of the rule, the same metrics on real evals data; `--' marks an "
    r"eval type the real corpora do not cover. Bold and underlined mark the best and "
    r"runner-up interval score within each block, computed over every method swept and "
    r"reported separately for synthetic and real data. Rows are a representative subset: "
    r"of the resampling methods we report the percentile and smooth bootstrap, and we "
    r"omit six overdispersion variants of the multi-run mean that no recommendation rests "
    r"on. Complete tables appear in Supplementary~S4.")   # refs cannot cross documents

CAPTIONS = {
 "pvalues_pairwise": r"""Pairwise p-value methods (nominal $\alpha{=}0.05$), on the synthetic
suite and, to the right of the rule, on real evals data. Per-$n$ columns give Type-I error at
that sample size; anything above $\alpha$ is inflated. On binary data the exact conditional
tests coincide---McNemar, the sign test and the sign-flip permutation test are the same test,
and all three run at 0.010 against a nominal 0.05, losing power accordingly. McNemar's mid-$p$
sits between them and \texttt{wilcoxon} (0.019 Type-I, 0.332 power against 0.027 and 0.354),
recovering much of that lost power while staying the more conservative of the two.
\texttt{bayes\_bootstrap} attains the best power in every block at the cost of running mildly
anti-conservative on continuous and Likert data. \texttt{paired\_t} and \texttt{wilcoxon} are
the dependable general-purpose choices, holding close to nominal from $n{=}20$ upward on both
synthetic and real data. Note the real suite covers binary and continuous only.""",
 "ci_single": r"""CI methods for mean point estimates, single-run (nominal 95\%, 2000 MC reps
per cell). Wilson and Jeffreys score intervals are best for binary data, at a fraction of
\texttt{bayes\_indep}'s cost. For numeric data the logit-transformed t-interval is the
best-calibrated across $n$, without the variability of NIG; smooth bootstrap is the
conservative resampling alternative, provided $n\approx80$ or greater on continuous data.
On real continuous data both logit-t and NIG run slightly conservative.""",

 "ci_multirun": r"""CI methods for mean point estimates, multi-run (nominal 95\%, 500 MC reps
per cell, runs=5). Taking the mean and using \texttt{logit\_t} or \texttt{nig} retains
reasonable performance for numeric data. Some methods post lower interval scores while
under-covering across several $n$, an inflated Type I error rate we optimize against.
On real binary data Wilson flat is strongest, at conservative coverage, while NIG under-covers.""",

 "ci_paired_single": r"""CI methods for pairwise comparisons, single-run (nominal 95\%, 300 MC
reps per cell). T-I and Pow are the rate at which the interval excludes zero on the null and
alternative scenarios respectively---the decision users act on, directly and through the
simultaneous-CI path. For binary data \texttt{mj\_floor} attains the lowest interval score,
but entirely on width: it carries the largest penalty term of the closed-form methods and much
the worst coverage tail (MinCov .800 synthetic, .803 real; 237 of 1980 synthetic cells fall
below .93, against 14 for \texttt{bonett\_price}). \texttt{bonett\_price} is the only method
that never falls below .90 in any cell on either source while also holding Type-I error under
nominal throughout, at the lowest cost of any method here; it gives up 6--13\% of power
relative to \texttt{mj\_floor} to do so. \texttt{logit\_t} is the best-calibrated choice for
continuous data, with \texttt{nig} marginally ahead on interval score on real data. For Likert
the same trade recurs: \texttt{nig} reaches the lower interval score while \texttt{logit\_t}
holds the better worst-case coverage.""",

 "ci_paired_nested": r"""CI methods for pairwise comparisons, multi-run (nominal 95\%, 600 MC
reps per cell synthetic / 1000 real, runs=5). \texttt{bonett\_price\_shrunk} carries the
single-run construction over with the item as the unit of analysis, shrinking the pseudo-item
magnitude toward its single-run value so the adjustment does not outweigh the data as runs
accumulate; it attains the lowest interval score of any method holding worst-case coverage above
.92, and the highest worst-case coverage of any method on real data. \texttt{mj\_floor\_cluster}
is narrower and so attains the lower interval score outright, but keeps the family's centre
shrinkage $\hat\delta/(1+z^2/n)$, whose denominator involves the item count only and is therefore
untouched by the number of runs, leaving a coverage tail no number of runs can remove.
\texttt{clustered\_score} is the published clustered competitor \citep{yang2012clustered}; it is
competitive on coverage but wider, and two orders of magnitude slower. Wald and the t-interval,
the most common practitioner approximations, both perform poorly: t-interval under-covers at
small $n$, and Wald is far too wide.""",
}


TABLES = [
    dict(key="ci_single", synth="tab:ci_single:sim", real="tab:ci_single:real",
         ns=NS_SINGLE, real_ns=REAL_NS_SINGLE, keep=None,
         base_cols=BASE_CI_SINGLE),
    dict(key="ci_multirun", synth="tab:ci_single:multirun",
         real="tab:ci_single:multirun:real", ns=NS_MULTI,
         real_ns=REAL_NS_MULTI, keep=None,
         base_cols=BASE_CI_SINGLE),
    dict(key="ci_paired_single", synth="tab:ci_paired:single:synth",
         real="tab:ci_paired:single:real", ns=NS_SINGLE,
         real_ns=REAL_NS_SINGLE, keep=None),
    dict(key="ci_paired_nested", synth="tab:ci_paired:nested:synth",
         real="tab:ci_paired:nested:real", ns=NS_MULTI,
         real_ns=REAL_NS_MULTI, keep=None),
    dict(key="pvalues_pairwise", synth="tab:pvalues:pairwise:synth",
         real="tab:pvalues:pairwise:real", ns=NS_PAIRWISE,
         real_ns=REAL_NS_PAIRWISE, keep=None,
         base_cols=BASE_PV, real_metrics=REAL_PV, n_group="Type-I error"),
]


def norm(name: str) -> str:
    """Strip LaTeX escaping and the '(bin)'/'(cont)'/'(lik)' disambiguator
    the harness appends when one method appears in several blocks, so a
    row can be matched to its counterpart in the real-data table."""
    n = name.replace("\\_", "_").replace("\\", "").strip()
    return re.sub(r"\s*\((bin|cont|lik)\)\s*$", "", n)



_NUM = re.compile(r"(?<![0-9.])0\.(\d+)")


def breakable(name: str) -> str:
    r"""Allow a p{} column to wrap a\_b\_c names. LaTeX has no break point
    at an escaped underscore, so without this the cell overruns into the
    next column instead of wrapping."""
    return name.replace(r"\_", r"\_\hspace{0pt}") if METHOD_COL_WIDTH else name


def strip_zero(cell: str) -> str:
    r"""0.825 -> .825, leaving \cellcolor / \textbf / \underline intact."""
    return _NUM.sub(r".\1", cell) if STRIP_LEADING_ZERO else cell


def fmt_time(cell: str) -> str:
    """Shorten the Time column to ~2 significant figures. Fixed decimals
    are wrong here: the values span 0.035ms to 50ms, and rounding to one
    decimal collapses 0.048 and 0.098 to the same "0.0", erasing the
    speed differences the prose actually claims."""
    if not SHORTEN_TIME:
        return cell
    try:
        v = float(cell)
    except ValueError:
        return cell
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.3f}"


def parse_table(tex: str, label: str) -> dict:
    m = re.search(r"\\label\{" + re.escape(label) + r"\}", tex)
    if not m:
        raise KeyError(label)
    begin = tex.rindex("\\begin{table", 0, m.start())
    env = "table*" if tex.startswith("\\begin{table*}", begin) else "table"
    end = tex.index("\\end{" + env + "}", m.start())
    blk = tex[begin:end]
    tab = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", blk, re.S)
    header, rows = None, []
    for line in tab.group(2).splitlines():
        s = line.strip()
        if not s or s.startswith("\\toprule") or s.startswith("\\bottomrule"):
            continue
        if s.startswith("\\midrule"):
            continue
        cells = [c.strip() for c in re.sub(r"\\\\\s*$", "", s).split("&")]
        if header is None:
            header = cells
        else:
            rows.append(cells)
    cap = re.search(r"\\caption\{(.*)\}\s*\\label", blk, re.S)
    return dict(env=env, header=header, rows=rows,
                caption=cap.group(1).strip() if cap else "", label=label)


def col_index(header: list[str], name: str) -> int:
    for i, h in enumerate(header):
        if h.replace("$\\downarrow$", "").strip() == name:
            return i
    for i, h in enumerate(header):
        if h.startswith(name):
            return i
    raise KeyError(f"{name} not in {header}")


def build(tex: str, spec: dict) -> str:
    synth = parse_table(tex, spec["synth"])
    real = parse_table(tex, spec["real"]) if spec["real"] else None

    h = synth["header"]
    i_type = col_index(h, "Type")
    base_names = [c for c in spec.get("base_cols", BASE_CI) if c not in PROBE_DROP]
    keep_idx = [col_index(h, n) for n in base_names]
    n_idx = [col_index(h, f"n={n}") for n in spec["ns"]]

    real_cov, real_ns = {}, spec.get("real_ns") or []
    if real:
        rh = real["header"]
        r_type = col_index(rh, "Type")
        r_met = [col_index(rh, m) for m in spec.get("real_metrics", REAL_METRICS)]
        r_n = [col_index(rh, f"n={n}") for n in real_ns]
        for r in real["rows"]:
            real_cov[(norm(r[0]), r[r_type])] = [r[i] for i in r_met + r_n]

    n_real_cols = (len(spec.get("real_metrics", REAL_METRICS)) + len(real_ns)) if real else 0
    ncol = len(keep_idx) + len(n_idx) + n_real_cols
    first = (r">{\raggedright\arraybackslash}p{" + METHOD_COL_WIDTH + "}"
             if METHOD_COL_WIDTH else "l")
    n_synth_cols = len(keep_idx) - 1 + len(n_idx)
    spec_str = first + "r" * n_synth_cols
    if real:
        spec_str += RULE_SPEC + "r" * n_real_cols

    out = [f"\\begin{{tabular}}{{{spec_str}}}", "\\toprule"]
    _HDR = {"Method": "Method", "Cov": "Cov", "MinCov": "MinCov",
            "Width": "Width", "Pen": "Pen $\\downarrow$",
            "Type-I": "T-I", "Power": "Pow $\\uparrow$",
            "Score": "Score $\\downarrow$", "Time (ms)": "T (ms)",
            "Type-I error": "Type-I", "Mean power": "Power"}
    base_hdr = [_HDR[c] for c in base_names]
    n_over = len(base_hdr) - 1
    group = ["", f"\\multicolumn{{{n_over}}}{{c}}{{Overall}}",
             f"\\multicolumn{{{len(n_idx)}}}{{c}}{{{spec.get('n_group', 'Coverage')} by $n$}}"]
    if real:
        group.append(f"\\multicolumn{{{n_real_cols}}}{{c}}{{Real evals data}}")
    out.append(" & ".join(group) + " \\\\")
    c0 = 2
    rules = [f"\\cmidrule(lr){{{c0}-{c0 + n_over - 1}}}"]
    c0 += n_over
    rules.append(f"\\cmidrule(lr){{{c0}-{c0 + len(n_idx) - 1}}}")
    c0 += len(n_idx)
    if real:
        rules.append(f"\\cmidrule(lr){{{c0}-{c0 + n_real_cols - 1}}}")
    out.append("".join(rules))
    hdr = list(base_hdr) + [f"${n}$" for n in spec["ns"]]
    if real:
        hdr += [_HDR.get(m, m) for m in spec.get("real_metrics", REAL_METRICS)]
        hdr += [f"${n}$" for n in real_ns]
    out.append(" & ".join(hdr) + " \\\\")

    dropped, unmatched = 0, []
    last_block = None
    for r in synth["rows"]:
        et = r[i_type]
        base = norm(r[0])
        key = (base, et)
        marked = "\\textbf" in " ".join(r) or "\\underline" in " ".join(r)
        if not marked and (base in DROP_FAMILY
                           or base in EXTRA_DROP.get(spec["key"], set())):
            dropped += 1
            continue
        if et != last_block:
            out.append("\\midrule")
            # split across the rule so the vertical separator stays
            # unbroken: one \multicolumn spanning everything would drop it
            label = f"\\textit{{{BLOCK_NAME.get(et, et)}}}"
            if real:
                out.append(f"\\multicolumn{{{n_synth_cols + 1}}}{{l}}{{{label}}}"
                           f" & \\multicolumn{{{n_real_cols}}}{{c}}{{}} \\\\")
            else:
                out.append(f"\\multicolumn{{{ncol}}}{{l}}{{{label}}} \\\\")
            last_block = et
        base_cells = [r[i] for i in keep_idx]
        if "Time (ms)" in base_names and "Time (ms)" not in PROBE_DROP:
            base_cells[base_names.index("Time (ms)")] = fmt_time(
                base_cells[base_names.index("Time (ms)")])
        cells = base_cells + [r[i] for i in n_idx]
        cells = [cells[0]] + [strip_zero(c) for c in cells[1:]]
        # method name loses its now-redundant "(bin)" disambiguator, since
        # the block header above it already says which eval type it is
        cells[0] = breakable(cells[0].replace(" (bin)", "")
                             .replace(" (cont)", "").replace(" (lik)", ""))
        if real:
            v = real_cov.get(key)
            if v is None:
                unmatched.append(key)
                cells += ["--"] * n_real_cols
            else:
                cells += [strip_zero(x) for x in v]
        out.append(" & ".join(cells) + " \\\\")

    out += ["\\bottomrule", "\\end{tabular}"]
    # every unmatched synth row should be an eval type the real corpora
    # simply don't cover; and no real row may be silently lost
    real_types = {k[1] for k in real_cov} if real else set()
    bad = [k for k in unmatched if k[1] in real_types]
    used = {(norm(r[0]), r[i_type]) for r in synth["rows"]}
    orphan = [k for k in real_cov if k not in used]
    print(f"  {spec['key']:18s} rows {len(synth['rows'])}->{len(synth['rows'])-dropped}"
          f"  cols {len(h)}->{ncol}"
          f"  real types {sorted(real_types)}"
          f"  unmatched-in-covered-type {len(bad)}{bad[:4] if bad else ''}"
          f"  real-rows-lost {len(orphan)}{orphan[:4] if orphan else ''}")
    return dict(env=synth["env"], tabular="\n".join(out),
                caption=CAPTIONS[spec["key"]], label=spec["synth"])



# Page count in this document is bound by float PLACEMENT, not float size:
# starred floats can only sit at a page top in two-column acmart, and the
# appendix has 43 of them. Removing rows saves nothing; removing floats
# saves ~0.4-0.7 pages each. So paired tables share one float, keeping
# their own \caption (and therefore their own number and \label) inside it.
MERGE_GROUPS = [("ci_single", "ci_multirun"),
                ("ci_paired_single", "ci_paired_nested")]


def as_float(parts: list[dict], fontsize: str, side_by_side: bool = False) -> str:
    env = parts[0]["env"]
    out = [f"\\begin{{{env}}}[t]", "\\centering", f"\\{fontsize}",
           f"\\setlength{{\\tabcolsep}}{{{TABCOLSEP}}}"]
    if side_by_side and len(parts) == 2:
        for i, p in enumerate(parts):
            out.append(r"\begin{minipage}[t]{0.49\textwidth}\centering")
            out += [p["tabular"], "\\caption{" + p["caption"] + "}",
                    f"\\label{{{p['label']}}}"]
            out.append(r"\end{minipage}" + (r"\hfill" if i == 0 else ""))
        out.append(f"\\end{{{env}}}")
        return "\n".join(out)
    for i, p in enumerate(parts):
        if i:
            out.append("\\vspace{1.2em}")
        out += [p["tabular"],
                "\\caption{" + p["caption"] + "}",
                f"\\label{{{p['label']}}}"]
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def assemble(built: dict, merge: bool, fontsize: str, sbs: bool = False) -> list[str]:
    if not merge:
        return [as_float([b], fontsize) for b in built.values()]
    done, out = set(), []
    for group in MERGE_GROUPS:
        if all(k in built for k in group):
            out.append(as_float([built[k] for k in group], fontsize, sbs))
            done.update(group)
    out += [as_float([b], fontsize) for k, b in built.items() if k not in done]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PAPER / "appendix.tex"),
                    help="source .tex holding the default-rendered tables. NOT "
                         "idempotent: once --apply has rewritten a file, re-run "
                         "against fresh harness output or a pristine backup, not "
                         "against the transformed file.")
    ap.add_argument("--out", default=str(PAPER / "trimmed_tables.tex"))
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="pack each table pair into a single float")
    ap.add_argument("--fontsize", default="footnotesize")
    ap.add_argument("--stack-out", dest="stack_out", default=None)
    ap.add_argument("--stack", default="",
                    help="groups of labels to pack into shared floats, e.g. "
                         "'a,b;c,d,e'. Unlike the CI path this does NOT fold "
                         "real data into columns -- these tables are already "
                         "86-109%% of \\textwidth, so there is no horizontal "
                         "room. Each table keeps its own caption and label, "
                         "and tabcolsep drops to TABCOLSEP for consistency.")
    ap.add_argument("--only", default="",
                    help="comma-separated table keys to process; the rest are "
                         "left alone. Needed once part of a file has already "
                         "been compressed, since this script is not idempotent.")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--supplement", metavar="OUT",
                    help="emit the untouched original tables as a supplement "
                         "section, relabelled suptab:* (refs cannot cross documents)")
    ap.add_argument("--probe-drop", default="",
                    help="comma-separated base columns to drop (probe only)")
    ap.add_argument("--keep-all-rows", action="store_true",
                    help="disable the bootstrap-family cull (for A/B measurement)")
    ap.add_argument("--apply", metavar="APPENDIX_OUT",
                    help="write a copy of appendix.tex with the 8 original CI "
                         "tables replaced by the 4 compressed ones")
    args = ap.parse_args()

    globals()["PROBE_DROP"] = {c.strip() for c in args.probe_drop.split(",") if c.strip()}
    if args.keep_all_rows:
        DROP_FAMILY.clear(); EXTRA_DROP.clear()
    tex = pathlib.Path(args.input).read_text()
    if args.only:
        keep = {k.strip() for k in args.only.split(",") if k.strip()}
        globals()["TABLES"] = [s for s in TABLES if s["key"] in keep]
        globals()["MERGE_GROUPS"] = [g for g in MERGE_GROUPS
                                     if all(k in keep for k in g)]
    if args.stack:
        out = tex

        def span(label):
            m = re.search(r"\\label\{" + re.escape(label) + r"\}", tex)
            if not m:
                raise KeyError(label)
            b = tex.rindex("\\begin{table", 0, m.start())
            env = "table*" if tex.startswith("\\begin{table*}", b) else "table"
            return b, tex.index("\\end{" + env + "}", m.start()) + len("\\end{" + env + "}"), env

        def body(label):
            """tabular + caption + label, wrapper stripped."""
            b, e, env = span(label)
            blk = tex[b:e]
            tab = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", blk, re.S).group(0)
            cap = re.search(r"(\\caption\{.*?\}\s*\\label\{[^}]*\})", blk, re.S)
            return tab + "\n" + (cap.group(1) if cap else "")

        edits = []
        for group in [g for g in args.stack.split(";") if g.strip()]:
            labs = [x.strip() for x in group.split(",") if x.strip()]
            spans = [span(l) for l in labs]
            env = "table*" if any(s[2] == "table*" for s in spans) else "table"
            parts = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize",
                     f"\\setlength{{\\tabcolsep}}{{{TABCOLSEP}}}"]
            for i, l in enumerate(labs):
                if i:
                    parts.append("\\vspace{1.2em}")
                parts.append(body(l))
            parts.append(f"\\end{{{env}}}")
            order = sorted((s[0], s[1]) for s in spans)
            edits.append((order[0][0], order[0][1], "\n".join(parts)))
            edits += [(s, e, "") for s, e in order[1:]]
            print(f"  stacked {len(labs)} tables into one float: {', '.join(labs)}")
        for s, e, repl in sorted(edits, key=lambda x: -x[0]):
            out = out[:s] + repl + out[e:]
        pathlib.Path(args.stack_out or args.apply).write_text(out)
        print(f"wrote {args.stack_out or args.apply}")
        return

    print("building trimmed tables:")
    built = {s["key"]: build(tex, s) for s in TABLES}
    bodies = assemble(built, args.merge, args.fontsize, args.side_by_side)

    if args.supplement:
        # The originals move verbatim -- same rows, same columns, same
        # values -- only the \label is rewritten, since the appendix now
        # uses the original labels for the compressed versions.
        parts = ["\\section{Complete confidence-interval method tables}",
                 "\\label{supsec:ci-tables}", "",
                 "The appendix reports a representative subset of the methods swept, with "
                 "the full $n$ axis and real-data performance. This section reproduces the "
                 "complete tables exactly as generated, with every method and both the "
                 "synthetic and real-data runs reported separately.", ""]
        for spec in TABLES:
            for lab in (spec["synth"], spec["real"]):
                if not lab:
                    continue
                m = re.search(r"\\label\{" + re.escape(lab) + r"\}", tex)
                b = tex.rindex("\\begin{table", 0, m.start())
                env = "table*" if tex.startswith("\\begin{table*}", b) else "table"
                e = tex.index("\\end{" + env + "}", m.start()) + len("\\end{" + env + "}")
                blk = tex[b:e].replace("\\label{" + lab + "}",
                                       "\\label{sup" + lab + "}")
                parts += [blk, ""]
        pathlib.Path(args.supplement).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.supplement).write_text("\n".join(parts))
        print(f"wrote {args.supplement}  (8 original tables, relabelled)")
        return

    if args.apply:
        # Resolve every original table's span FIRST, then apply edits from
        # the end of the file backwards. Searching for a label after an
        # insertion is unsafe: the emitted float contains the labels of the
        # tables it replaces, so a later lookup matches the new float and
        # deletes it instead of the original.
        def span(label):
            m = re.search(r"\\label\{" + re.escape(label) + r"\}", tex)
            if not m:
                raise KeyError(label)
            b = tex.rindex("\\begin{table", 0, m.start())
            env = "table*" if tex.startswith("\\begin{table*}", b) else "table"
            return b, tex.index("\\end{" + env + "}", m.start()) + len("\\end{" + env + "}")

        groups, seen = [], set()
        if args.merge:
            for g in MERGE_GROUPS:
                if all(k in built for k in g):
                    groups.append(list(g)); seen.update(g)
        groups += [[s["key"]] for s in TABLES if s["key"] not in seen]
        by_key = {s["key"]: s for s in TABLES}

        edits = []
        for keys, body in zip(groups, bodies):
            spans = []
            for k in keys:
                for lab in (by_key[k]["synth"], by_key[k]["real"]):
                    if lab:
                        spans.append(span(lab))
            spans.sort()
            edits.append((spans[0][0], spans[0][1], body))     # first -> new float
            edits += [(s, e, "") for s, e in spans[1:]]         # rest -> deleted
        out = tex
        for s, e, repl in sorted(edits, key=lambda x: -x[0]):
            out = out[:s] + repl + out[e:]
        pathlib.Path(args.apply).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.apply).write_text(out)
        print(f"wrote {args.apply}  ({len(edits)} table spans rewritten)")
        return

    doc = []
    if args.standalone:
        # preview in the paper's own class so column widths, font size and
        # table* span match what these will look like in the appendix
        doc += [r"\documentclass[sigconf,nonacm]{acmart}",
                r"\settopmatter{printacmref=false}",
                r"\usepackage{booktabs}", r"\usepackage{array}", r"\usepackage[table]{xcolor}",
                r"\newcommand{\evalstats}{\texttt{evalstats}}",
                r"\begin{document}",
                r"\title{Trimmed CI tables (preview)}",
                r"\author{Preview}\affiliation{\institution{n/a}\country{n/a}}",
                r"\maketitle",
                r"\section*{Preview}",
                r"Preview of the compressed appendix CI tables."]
    doc += bodies
    if args.standalone:
        doc.append(r"\end{document}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text("\n\n".join(doc) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
