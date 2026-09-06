"""Simulation cases for the evalstats harness.

Each case module exposes:
  CASE_NAME: str
  add_arguments(subparser) -> None        register case-specific CLI flags
  run(args) -> CaseResult                 orchestrate + write output artifacts
  official_variants(base_seed) -> list[(label, Namespace)]
                                           the paper's official preset(s)
  quick_args(base_seed, data_source) -> Namespace
                                           a fast smoke-test preset
  official_args(base_seed) -> Namespace   internal building block some
                                           modules use to construct the
                                           presets above; not called directly
                                           by cli.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseResult:
    case_name: str
    status: str  # "ok" | "error"
    output_paths: list[str] = field(default_factory=list)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    error: str | None = None
