"""Centralized simulation harness CLI.

Usage:
  python -m simulations.harness.cli --list-cases
  python -m simulations.harness.cli ci_single --reps 50 --sizes 10 20
  python -m simulations.harness.cli ci_single --data-source real
  python -m simulations.harness.cli --official-tests
  python -m simulations.harness.cli --quick-test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .cases import ci_single, ci_paired, ci_unpaired, pvalues, ppi_real, compare_e2e, irr_peak
from .manifest import write_manifest

CASES = {
    ci_single.CASE_NAME: ci_single,
    ci_paired.CASE_NAME: ci_paired,
    ci_unpaired.CASE_NAME: ci_unpaired,
    pvalues.CASE_NAME: pvalues,
    ppi_real.CASE_NAME: ppi_real,
    compare_e2e.CASE_NAME: compare_e2e,
    irr_peak.CASE_NAME: irr_peak,
}


def _case_summary(module) -> str:
    """First non-blank line of a case module's docstring, used as its --list-cases/subparser help text."""
    doc = module.__doc__ or ""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser: --list-cases/--official-tests/--quick-test flags plus one subparser per case."""
    parser = argparse.ArgumentParser(
        prog="python -m simulations.harness.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-cases", action="store_true", help="List available cases and exit.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                        help="Parallel worker processes for --official-tests/--quick-test presets (default: cpu_count-1).")
    parser.add_argument(
        "--official-tests", action="store_true", default=False,
        help=(
            "Show an interactive menu of official-test variants (synthetic, real data, "
            "nested) and run the selected subset, writing all artifacts into one "
            "timestamped simulations/out/official_<timestamp>/ directory."
        ),
    )
    parser.add_argument(
        "--quick-test", action="store_true", default=False,
        help=(
            "Run a fast sanity-check preset for every case (sizes=10,30,50; "
            "reps=3; --plots save; --progress bar; --latex), TWICE per case "
            "(once --data-source synthetic, once real), writing every "
            "artifact into one timestamped simulations/out/quick_<timestamp>/ "
            "directory with a manifest.json index. For confirming the pipeline "
            "still works (synthetic AND real-data paths) before committing to "
            "a full --official-tests run."
        ),
    )
    subparsers = parser.add_subparsers(dest="case", metavar="case")
    for name, module in CASES.items():
        sp = subparsers.add_parser(name, help=_case_summary(module))
        module.add_arguments(sp)
    return parser


# ---------------------------------------------------------------------------
# Official-test menu
# ---------------------------------------------------------------------------

def _collect_official_variants() -> list[tuple[str, str, argparse.Namespace]]:
    """Return all official-test variants as (case_name, label, args) triples."""
    variants: list[tuple[str, str, argparse.Namespace]] = []
    for name, module in CASES.items():
        for label, args in module.official_variants():
            variants.append((name, label, args))
    return variants


def _show_official_menu(variants: list[tuple[str, str, argparse.Namespace]]) -> list[tuple[str, str, argparse.Namespace]]:
    """Print the official-test menu and return the user-selected variants."""
    sep = "=" * 72
    print(f"\n{sep}\n  OFFICIAL TESTS — select variants to run\n{sep}\n")
    for i, (case_name, label, _) in enumerate(variants, 1):
        print(f"  {i:2d}.  {case_name:<15}  {label}")
    print(f"\n   A   Run all {len(variants)} variants sequentially")
    print("   Q   Quit\n")

    while True:
        try:
            raw = input("Select tests to run (space-separated numbers, A, or Q): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return []

        if not raw:
            continue
        if raw.lower() == "q":
            return []
        if raw.lower() == "a":
            return list(variants)

        try:
            indices = [int(x) - 1 for x in raw.split()]
            if all(0 <= idx < len(variants) for idx in indices):
                return [variants[idx] for idx in indices]
            print(f"  Please enter numbers between 1 and {len(variants)}, 'A', or 'Q'.")
        except ValueError:
            print(f"  Invalid input — enter numbers like \"1 3\", 'A' for all, or 'Q' to quit.")


def _run_official_selected(
    selected: list[tuple[str, str, argparse.Namespace]],
    n_workers: int,
    manifest_out_dir: str,
) -> bool:
    """Run the selected (case_name, label, args) variants and write a manifest. Returns True on full success."""
    results = []
    args_list = []
    sep = "=" * 72
    for case_name, label, case_args in selected:
        module = CASES[case_name]
        case_args.out_dir = manifest_out_dir
        case_args.plots_dir = str(Path(manifest_out_dir) / "plots")
        case_args.workers = n_workers
        print(f"\n{sep}\nOFFICIAL TEST: {case_name} — {label}  [workers={n_workers}]\n{sep}")
        result = module.run(case_args)
        results.append(result)
        args_list.append(case_args)
        if result.status != "ok":
            print(f"\n  *** {case_name} ({label}) FAILED: {result.error}")

    write_manifest(manifest_out_dir, results, args_list)

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"\n{len(failed)} variant(s) failed.")
    else:
        print(f"\nAll {len(results)} variant(s) completed successfully.")
        print(f"Artifacts: {manifest_out_dir}/")
    return not failed


# ---------------------------------------------------------------------------
# Quick-test preset runner (unchanged)
# ---------------------------------------------------------------------------

def _run_preset(case_names: list[str], args_factory, dir_prefix: str, label: str, n_workers: int = 1) -> bool:
    """Run `args_factory(module)` for each named case -- it may return a single
    Namespace or a list of Namespaces to run back to back (e.g. --quick-test's
    synthetic + real data-source calls) -- writing every artifact into one
    timestamped simulations/out/<dir_prefix>_<timestamp>/ directory with a
    manifest.json index. Returns True if every case (and every variant) succeeded."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest_out_dir = f"simulations/out/{dir_prefix}_{stamp}"
    results = []
    args_list: list[argparse.Namespace] = []

    for name in case_names:
        module = CASES[name]
        variants = args_factory(module)
        if isinstance(variants, argparse.Namespace):
            variants = [variants]
        for case_args in variants:
            case_args.out_dir = manifest_out_dir
            case_args.plots_dir = str(Path(manifest_out_dir) / "plots")
            case_args.workers = n_workers
            data_source = getattr(case_args, "data_source", None)
            suffix = f" (data_source={data_source})" if data_source else ""
            print(f"\n{'=' * 72}\n{label}: {name}{suffix}  [workers={n_workers}]\n{'=' * 72}")
            result = module.run(case_args)
            results.append(result)
            args_list.append(case_args)
            if result.status != "ok":
                print(f"\n  Case {name} FAILED: {result.error}")

    write_manifest(manifest_out_dir, results, args_list)

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"\n{len(failed)} case(s) failed: {[r.case_name for r in failed]}")
    return not failed


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: dispatches to --list-cases, --official-tests, --quick-test, or a single named case."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_cases:
        print("Available cases:")
        for name, module in CASES.items():
            print(f"  {name:<20} {_case_summary(module)}")
        return

    if args.official_tests:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        manifest_out_dir = f"simulations/out/official_{stamp}"
        all_variants = _collect_official_variants()
        selected = _show_official_menu(all_variants)
        if not selected:
            print("No variants selected — exiting.")
            return
        ok = _run_official_selected(selected, n_workers=args.workers, manifest_out_dir=manifest_out_dir)
        if not ok:
            sys.exit(1)
        return

    if args.quick_test:
        ok = _run_preset(
            list(CASES),
            lambda module: [module.quick_args(), module.quick_args(data_source="real")],
            "quick", "QUICK TEST", n_workers=args.workers,
        )
        if not ok:
            sys.exit(1)
        return

    if not args.case:
        parser.error("Specify a case (e.g. ci_single) or use --official-tests. Use --list-cases to see options.")

    module = CASES[args.case]
    result = module.run(args)
    if result.status != "ok":
        print(f"\nCase {result.case_name} failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
