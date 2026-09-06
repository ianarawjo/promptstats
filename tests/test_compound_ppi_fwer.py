"""Tests for the COMPOUND scenario: PPI alignment correction layered on top
of FWER-controlled (simultaneous-CI / multiple-comparison-corrected)
pairwise results, via the compare(alignment=..., simultaneous_ci=..., correction=...)
path.

This is the "real practice" case none of the existing calibration sweeps
(simulations/sim_alignment_methods.py, simulations/sim_type_i_calibration.py)
exercise end-to-end: those validate PPI correction and FWER correction each
in isolation (2-arm PPI-only, or k-arm FWER-only on raw LLM scores), never
both stacked together on a k>=3 comparison with a noisy judge and sparse
human labels. See evalstats/api.py's ``_run_alignment_ppi`` docstring for
what this compound path's simultaneous-CI resolution (sidak/boot, mirroring
the non-PPI decision tree) does and doesn't have harness-scale validation
for yet.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.alignment import judge_alignment


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _make_multiarm_binary(
    n_entities: int = 3,
    n_items: int = 150,
    n_labeled: int = 60,
    p_base: float = 0.55,
    effect_size: float = 0.0,
    agreement_rate: float = 0.85,
    seed: int = 0,
):
    """k-arm binary data with human labels on the SAME item positions for
    every entity, so every pair has full labeled-item overlap (the PPI
    "dispatch" branch -- see _run_alignment_ppi's pair_branch classification).

    ``effect_size=0.0`` gives a true null (every arm has the same success
    probability); a positive value spreads arm probabilities evenly around
    ``p_base`` for a power check.
    """
    rng = _rng(seed)
    if effect_size == 0.0:
        probs = np.full(n_entities, p_base)
    else:
        probs = np.linspace(p_base - effect_size / 2, p_base + effect_size / 2, n_entities)
    rows = []
    for i in range(n_entities):
        lbl = f"M{i}"
        vals = rng.binomial(1, probs[i], n_items).astype(float)
        for j in range(n_items):
            rows.append((lbl, j, vals[j]))
    df = pd.DataFrame(rows, columns=["model", "item", "llm_score"])

    human = np.full(len(df), np.nan)
    labeled_items = rng.choice(n_items, size=n_labeled, replace=False)
    mask = df["item"].isin(labeled_items)
    idxs = df.index[mask]
    agree = rng.random(len(idxs)) < agreement_rate
    human[idxs] = np.where(agree, df.loc[idxs, "llm_score"], 1.0 - df.loc[idxs, "llm_score"])
    df["human_score"] = human

    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata


def _make_multiarm_continuous(
    n_entities: int = 3,
    n_items: int = 200,
    n_labeled: int = 80,
    seed: int = 0,
):
    """k-arm [0,1]-bounded continuous data, fully shared labeled positions
    across every entity (all pairs "dispatch")."""
    rng = _rng(seed)
    means = np.linspace(0.55, 0.75, n_entities)
    rows = []
    for i in range(n_entities):
        lbl = f"M{i}"
        vals = np.clip(rng.normal(means[i], 0.12, n_items), 0, 1)
        for j in range(n_items):
            rows.append((lbl, j, vals[j]))
    df = pd.DataFrame(rows, columns=["model", "item", "llm_score"])

    human = np.full(len(df), np.nan)
    labeled_items = rng.choice(n_items, size=n_labeled, replace=False)
    mask = df["item"].isin(labeled_items)
    idxs = df.index[mask]
    human[idxs] = np.clip(df.loc[idxs, "llm_score"] + rng.normal(0, 0.05, len(idxs)), 0, 1)
    df["human_score"] = human

    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata


def _make_mixed_branch_binary(n_items: int = 150, seed: int = 0):
    """4-arm binary data where M0/M1/M2 share full labeled-item overlap
    ("dispatch") but M3's labels come from a disjoint, sub-threshold set of
    items -- every pair touching M3 has zero overlap AND fewer than 15 of
    M3's own labels, so it lands in the "skip" branch (kept uncorrected).

    Exercises the fact that a single compound compare() call can return a
    HETEROGENEOUS mix of PPI-corrected and un-corrected pairs.
    """
    rng = _rng(seed)
    entities = ["M0", "M1", "M2", "M3"]
    rows = []
    for lbl in entities:
        vals = rng.binomial(1, 0.55, n_items).astype(float)
        for j in range(n_items):
            rows.append((lbl, j, vals[j]))
    df = pd.DataFrame(rows, columns=["model", "item", "llm_score"])

    human = np.full(len(df), np.nan)
    shared_items = rng.choice(n_items, size=60, replace=False)
    remaining = np.setdiff1d(np.arange(n_items), shared_items)
    m3_items = rng.choice(remaining, size=8, replace=False)

    for lbl, items in (("M0", shared_items), ("M1", shared_items), ("M2", shared_items), ("M3", m3_items)):
        mask = (df["model"] == lbl) & (df["item"].isin(items))
        idxs = df.index[mask]
        agree = rng.random(len(idxs)) < 0.85
        human[idxs] = np.where(agree, df.loc[idxs, "llm_score"], 1.0 - df.loc[idxs, "llm_score"])
    df["human_score"] = human

    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return evaldata, ["M0", "M1", "M2", "M3"]


# ---------------------------------------------------------------------------
# Structural / plumbing tests (fast, deterministic where possible)
# ---------------------------------------------------------------------------

class TestCompoundStructuralPlumbing:
    def test_simultaneous_ci_flag_and_valid_output_survive_ppi(self):
        """simultaneous_ci=True + alignment= together still produce a fully
        valid pairwise matrix: flag preserved, every CI ordered/finite, every
        p-value in [0, 1]."""
        evaldata = _make_multiarm_binary(n_entities=3, seed=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer",
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.simultaneous_ci is True
        assert bundle.ppi_applied is True
        for pr in bundle.pairwise.results.values():
            assert np.isfinite(pr.ci_low) and np.isfinite(pr.ci_high)
            assert pr.ci_low <= pr.ci_high
            assert 0.0 <= pr.p_value <= 1.0

    def test_bonferroni_pair_alpha_widens_ppi_cis_vs_non_simultaneous(self):
        """Forcing a specific PPI method (deterministic, closed-form 'mj_floor'),
        simultaneous_ci=True must divide alpha by n_pairs and thus produce
        strictly wider (or equal, in a degenerate case) CIs than
        simultaneous_ci=False on the identical data."""
        evaldata = _make_multiarm_binary(n_entities=3, seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

            result_sim = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=True, correction="none",
            )
            result_nosim = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=False, correction="none",
            )

        pw_sim = result_sim._primary_bundle().pairwise
        pw_nosim = result_nosim._primary_bundle().pairwise
        assert pw_sim.simultaneous_ci is True
        assert pw_nosim.simultaneous_ci is False

        widths_sim = {k: (v.ci_high - v.ci_low) for k, v in pw_sim.results.items()}
        widths_nosim = {k: (v.ci_high - v.ci_low) for k, v in pw_nosim.results.items()}
        for key in widths_sim:
            assert widths_sim[key] >= widths_nosim[key] - 1e-9, (
                f"Pair {key}: simultaneous width {widths_sim[key]} should be >= "
                f"non-simultaneous width {widths_nosim[key]}"
            )
        # At least one pair should be strictly wider (guards against the
        # widening being silently a no-op).
        assert any(widths_sim[k] > widths_nosim[k] + 1e-9 for k in widths_sim)

    def test_correction_only_changes_pvalues_not_cis(self):
        """correction= (p-value FWER control) and simultaneous_ci= (CI
        widening) are independent knobs -- correction must NOT move the CI
        bounds, only the p-values, matching the non-PPI convention documented
        in _run_alignment_ppi (Bonferroni CI construction vs. p-value
        correction being separate steps)."""
        evaldata = _make_multiarm_binary(n_entities=3, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

            # Both calls share a seeded rng: at this N (150/entity, binary),
            # simultaneous_ci=True now resolves to "boot" (joint bootstrap
            # with an effective alpha, per the paper's decision tree), which
            # is stochastic -- an unseeded rng would make the two calls'
            # CIs differ for a reason unrelated to correction=.
            result_none = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=True, correction="none", rng=_rng(112),
            )
            result_shaffer = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=True, correction="shaffer", rng=_rng(112),
            )

        pw_none = result_none._primary_bundle().pairwise
        pw_shaffer = result_shaffer._primary_bundle().pairwise
        for key in pw_none.results:
            r_none = pw_none.results[key]
            r_shaffer = pw_shaffer.results[key]
            assert abs(r_none.ci_low - r_shaffer.ci_low) < 1e-12
            assert abs(r_none.ci_high - r_shaffer.ci_high) < 1e-12
            # Shaffer's correction can only make p-values >= the uncorrected ones.
            assert r_shaffer.p_value >= r_none.p_value - 1e-12

    def test_default_auto_reaches_sidak(self):
        """The compound path's default (method="auto", prefer="auto") mirrors
        the paper's decision tree, which is now Sidak at every N and eval
        type -- widening whichever closed-form PPI method resolved
        (ppi_logit_t here), not a silent Bonferroni downgrade. "boot" is no
        longer an auto-resolved outcome anywhere; it stays reachable via the
        private _run_alignment_ppi(prefer="boot"), exercised by
        test_romano_wolf_reuses_joint_resample_shared_with_boot. See
        resolve_auto_simultaneous_ci_method."""
        evaldata = _make_multiarm_continuous(n_entities=3, n_items=200, n_labeled=80, seed=13)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer", rng=_rng(13),
                # method="auto" (default), prefer="auto" (default)
            )
        bundle = result._primary_bundle()
        assert bundle.resolved_method in ("ppi_logit_t", "ppi_t_interval")
        assert bundle.pairwise.simultaneous_ci_method == "sidak"

    def test_default_auto_reaches_sidak_below_n_threshold(self):
        """Below the tree's N threshold, "auto" resolves to "sidak" (a pure
        closed-form alpha adjustment), not Bonferroni -- matching the
        non-PPI path's small-N branch instead of a narrower PPI-only
        fallback."""
        evaldata = _make_multiarm_continuous(n_entities=3, n_items=25, n_labeled=15, seed=131)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer",
                # method="auto" (default), prefer="auto" (default)
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.simultaneous_ci_method == "sidak"

    def test_method_bootstrap_t_uses_auto_tree_not_max_t(self):
        """Forcing method="bootstrap_t" under the default prefer="auto" gets
        the SAME auto-tree treatment (now Sidak) as every other method, not a
        special full max-T construction --
        max_t is never an auto-resolved outcome (matches evalstats' decision
        tree, which has no max-T node at all, and the non-PPI
        _simultaneous_cis_router's identical convention: max_t is reachable
        only via an explicit, literal prefer="max_t" request)."""
        evaldata = _make_multiarm_binary(n_entities=3, n_items=150, n_labeled=60, seed=14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=50, method="bootstrap_t",
                simultaneous_ci=True, correction="shaffer",
                rng=_rng(14),
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.simultaneous_ci_method == "sidak"
        for pr in bundle.pairwise.results.values():
            assert np.isfinite(pr.ci_low) and np.isfinite(pr.ci_high)
            assert pr.ci_low <= pr.ci_high
            assert 0.0 <= pr.p_value <= 1.0

    def test_prefer_kwarg_is_not_forwarded_through_compare(self):
        """compare(alignment=..., prefer=...) does not raise, but silently
        drops prefer= as an unrecognized keyword -- it has no effect on the
        PPI simultaneous-CI construction. This documents a real gap: there is
        currently no way for a compare() caller to force
        prefer="bonferroni"/"max_t"/"sidak"/"boot" for the PPI path directly
        (unlike the non-PPI all_pairwise()/analyze() path, where prefer= IS a
        recognized engine kwarg) -- only the internal, private
        _run_alignment_ppi function accepts it (see
        test_explicit_max_t_reachable_via_private_function below)."""
        evaldata = _make_multiarm_binary(n_entities=3, n_items=150, n_labeled=60, seed=141)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=50, method="bootstrap_t",
                simultaneous_ci=True, correction="shaffer", prefer="max_t",
                rng=_rng(141),
            )
        messages = [str(w.message) for w in caught]
        assert any("unknown keyword argument 'prefer'" in m for m in messages), messages
        bundle = result._primary_bundle()
        # Still resolves via the auto tree (now Sidak) -- prefer="max_t" was
        # NOT honored, confirming it has no effect through compare().
        assert bundle.pairwise.simultaneous_ci_method == "sidak"

    def test_explicit_max_t_reachable_via_private_function(self):
        """max_t is a real, working construction -- just not reachable from
        the public compare() API (see test_prefer_kwarg_is_not_forwarded_
        through_compare). Calling the private _run_alignment_ppi directly
        with prefer="max_t" (mirroring how the non-PPI all_pairwise() DOES
        expose prefer= publicly) confirms it still resolves correctly."""
        from evalstats.api import _run_alignment_ppi
        from evalstats.loader import load_from as _load_from

        evaldata = _make_multiarm_binary(n_entities=3, n_items=150, n_labeled=60, seed=142)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                method="bootstrap_t", simultaneous_ci=True, correction="shaffer",
            )
            df = evaldata._df.copy()
            _run_alignment_ppi(
                result, df=df, metric_col="llm_score", factor_col="model", item_col="item",
                alignment_result=ar, alpha=0.05, n_boot=1000, correction="shaffer",
                method="bootstrap_t", rng=_rng(142), prefer="max_t",
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.simultaneous_ci_method == "max_t"
        for pr in bundle.pairwise.results.values():
            assert np.isfinite(pr.ci_low) and np.isfinite(pr.ci_high)
            assert pr.ci_low <= pr.ci_high
            assert 0.0 <= pr.p_value <= 1.0

    def test_max_t_silently_falls_back_to_bonferroni_when_overlap_insufficient(self):
        """When some pairs lack the shared-labeled-item structure max-T needs
        (fallback/skip branches present for pairs touching M3), the compound
        path falls back to Bonferroni for the WHOLE comparison (max-T is all-
        or-nothing across pairs) -- but, unlike the "labeled positions don't
        match across pairs" case internal to _ppi_bootstrap_t_joint_stats,
        this specific pre-emptive fallback (via the `not fallback_pairs and
        not skipped_pairs` guard) does NOT emit its own explanatory warning.
        The only warnings raised are the generic fallback/skip notices already
        covered by test_skipped_pair_stays_uncorrected_while_others_are_ppi_corrected.
        This is worth knowing operationally: a user won't be told *why* they
        didn't get max-T in this case, only that some pairs used a weaker
        correction."""
        evaldata, _labels = _make_mixed_branch_binary(seed=15)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=50, method="bootstrap_t",
                simultaneous_ci=True, correction="shaffer",
                rng=_rng(15),
            )
        bundle = result._primary_bundle()
        # Since the auto tree resolves to Sidak, which needs none of the
        # shared-labeled-item structure max-T does, insufficient overlap no
        # longer costs the whole comparison a Bonferroni downgrade -- it just
        # uses Sidak. (Under the old boot default this asserted "bonferroni".)
        assert bundle.pairwise.simultaneous_ci_method == "sidak"
        messages = [str(w.message) for w in caught]
        assert not any("Falling back to Bonferroni" in m for m in messages), messages

    def test_skipped_pair_stays_uncorrected_while_others_are_ppi_corrected(self):
        """In the mixed-branch dataset, pairs touching M3 (insufficient
        overlap) must keep their un-annotated, non-PPI test_method, while
        every pair among M0/M1/M2 (full overlap) must be PPI-corrected --
        i.e. a single compound compare() result can legitimately be a
        heterogeneous mix of corrected/uncorrected pairs."""
        evaldata, labels = _make_mixed_branch_binary(seed=16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer",
            )
        bundle = result._primary_bundle()
        for (a, b), pr in bundle.pairwise.results.items():
            touches_m3 = "M3" in (a, b)
            if touches_m3:
                assert "PPI" not in pr.test_method, (a, b, pr.test_method)
            else:
                assert "PPI" in pr.test_method, (a, b, pr.test_method)

    def test_two_arm_compound_does_not_widen_for_fwer(self):
        """With only one pair (k=2 arms), use_simultaneous is False regardless
        of the simultaneous_ci= flag (no family to control) -- CIs should
        match the non-simultaneous call exactly."""
        evaldata = _make_multiarm_binary(n_entities=2, seed=17)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result_sim = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=True, correction="none",
            )
            result_nosim = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30, method="mj_floor",
                simultaneous_ci=False, correction="none",
            )
        pw_sim = result_sim._primary_bundle().pairwise
        pw_nosim = result_nosim._primary_bundle().pairwise
        for key in pw_sim.results:
            assert abs(pw_sim.results[key].ci_low - pw_nosim.results[key].ci_low) < 1e-12
            assert abs(pw_sim.results[key].ci_high - pw_nosim.results[key].ci_high) < 1e-12

    def test_correction_auto_resolves_instead_of_crashing(self):
        """compare(alignment=..., correction="auto") must not crash.

        Regression test: correction="auto" is the non-PPI path's own
        default and a perfectly reasonable thing to pass explicitly, but
        _run_alignment_ppi used to forward "auto" straight into
        correct_pvalues() (which has no "auto" case of its own -- only
        all_pairwise()'s non-PPI path pre-resolves "auto" before reaching
        it), raising "ValueError: Unknown correction method: auto". At
        N=150 (>= the auto tree's N>=30 threshold), "auto" now resolves to
        "romano_wolf" (see TestRomanoWolfPvalues), not "shaffer" -- this
        test just confirms the crash is gone and every p-value is valid,
        regardless of which correction resolved."""
        evaldata = _make_multiarm_binary(n_entities=3, n_items=150, n_labeled=60, seed=143)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="auto",
                rng=_rng(143),
            )
        bundle = result._primary_bundle()
        for pr in bundle.pairwise.results.values():
            assert 0.0 <= pr.p_value <= 1.0


# ---------------------------------------------------------------------------
# PPI-generalized Romano-Wolf step-down p-values (structural tests)
# ---------------------------------------------------------------------------
# Romano-Wolf's step-down refinement reuses the SAME joint bootstrap
# resample "boot" (simultaneous CIs) already needs -- see
# _ppi_romano_wolf_pvalues_from_joint_stats in evalstats/api.py -- so it
# shares that construction's requirements: every pair must be "dispatch"
# branch (full shared-labeled-item overlap), with >= 15 shared labeled
# items. Falls back to Shaffer's when that structure isn't available,
# mirroring "boot"'s own degrade-to-Bonferroni precedent.

class TestRomanoWolfPvalues:
    def test_explicit_romano_wolf_produces_valid_pvalues(self):
        evaldata = _make_multiarm_binary(n_entities=4, n_items=150, n_labeled=60, seed=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="romano_wolf",
                rng=_rng(200),
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.correction_method == "romano_wolf"
        for pr in bundle.pairwise.results.values():
            assert 0.0 <= pr.p_value <= 1.0

    def test_auto_resolves_to_romano_wolf_at_n_ge_30(self):
        """Mirrors the non-PPI decision tree: correction="auto" resolves to
        Romano-Wolf at N >= 30 (numeric) when the shared-structure
        requirement is met, not just Shaffer's regardless of N."""
        evaldata = _make_multiarm_binary(n_entities=4, n_items=100, n_labeled=40, seed=201)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="auto",
                rng=_rng(201),
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.correction_method == "romano_wolf"

    def test_auto_resolves_to_shaffer_below_n_30(self):
        evaldata = _make_multiarm_binary(n_entities=4, n_items=25, n_labeled=15, seed=202)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="auto",
                rng=_rng(202),
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.correction_method == "shaffer"

    def test_romano_wolf_falls_back_to_shaffer_when_overlap_insufficient(self):
        """When some pairs lack the shared-labeled-item structure Romano-Wolf
        needs (fallback/skip branches present for pairs touching M3), the
        whole comparison's p-value correction falls back to Shaffer's --
        Romano-Wolf is all-or-nothing across pairs, same as "boot"."""
        evaldata, _labels = _make_mixed_branch_binary(seed=203)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="romano_wolf",
                rng=_rng(203),
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.correction_method == "shaffer"
        for pr in bundle.pairwise.results.values():
            assert 0.0 <= pr.p_value <= 1.0

    def test_wilcoxon_companion_pvalues_use_shaffer_not_romano_wolf(self):
        """Romano-Wolf's joint construction is specific to the paired-mean/
        PPI estimand and has no Wilcoxon-signed-rank-compatible form (same
        as the non-PPI path) -- the companion wilcoxon_p correction must
        substitute Shaffer's (or Holm for a subset) rather than crash trying
        to pass "romano_wolf" into correct_pvalues(), which doesn't know
        that method name."""
        evaldata = _make_multiarm_binary(n_entities=4, n_items=150, n_labeled=60, seed=204)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="romano_wolf",
                rng=_rng(204),
            )
        bundle = result._primary_bundle()
        for pr in bundle.pairwise.results.values():
            if pr.wilcoxon_p is not None:
                assert 0.0 <= pr.wilcoxon_p <= 1.0

    def test_correction_method_field_reflects_actual_correction(self):
        """bundle.pairwise.correction_method must reflect the correction
        ACTUALLY applied to the final PPI-corrected p-values -- previously
        left untouched by _run_alignment_ppi, so it silently kept whatever
        the initial non-PPI analyze() pass had set."""
        evaldata = _make_multiarm_binary(n_entities=4, n_items=150, n_labeled=60, seed=205)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result_none = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="none",
                rng=_rng(205),
            )
            result_shaffer = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="shaffer",
                rng=_rng(205),
            )
            result_rw = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=False, correction="romano_wolf",
                rng=_rng(205),
            )
        assert result_none._primary_bundle().pairwise.correction_method is None
        assert result_shaffer._primary_bundle().pairwise.correction_method == "shaffer"
        assert result_rw._primary_bundle().pairwise.correction_method == "romano_wolf"

    def test_romano_wolf_reuses_joint_resample_shared_with_boot(self):
        """When BOTH the joint bootstrap ("boot") and correction="romano_wolf"
        apply together, both must still produce fully valid output -- this
        exercises that sharing the one joint bootstrap resample between the
        two constructions doesn't corrupt either.

        "boot" is no longer an auto-resolved outcome (the tree is Sidak
        everywhere), and compare() does not forward prefer= on the PPI path
        (see test_prefer_kwarg_is_not_forwarded_through_compare), so this
        drives the private _run_alignment_ppi directly -- the same route
        test_explicit_max_t_reachable_via_private_function uses. Without
        that, the resample-sharing path would no longer be covered at all."""
        from evalstats.api import _run_alignment_ppi

        evaldata = _make_multiarm_binary(n_entities=4, n_items=150, n_labeled=60, seed=206)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                simultaneous_ci=True, correction="romano_wolf",
                rng=_rng(206),
            )
            _run_alignment_ppi(
                result, df=evaldata._df.copy(), metric_col="llm_score",
                factor_col="model", item_col="item", alignment_result=ar,
                alpha=0.05, n_boot=1000, correction="romano_wolf",
                method="bootstrap", rng=_rng(206), prefer="boot",
            )
        bundle = result._primary_bundle()
        assert bundle.pairwise.simultaneous_ci_method == "boot"
        assert bundle.pairwise.correction_method == "romano_wolf"
        for pr in bundle.pairwise.results.values():
            assert np.isfinite(pr.ci_low) and np.isfinite(pr.ci_high)
            assert pr.ci_low <= pr.ci_high
            assert 0.0 <= pr.p_value <= 1.0

