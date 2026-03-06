"""promptstats: Statistical analysis and visualization for prompt benchmarking."""

from promptstats.core.types import BenchmarkResult, MultiModelBenchmark
from promptstats.core.paired import pairwise_differences, all_pairwise, vs_baseline, friedman_nemenyi, FriedmanResult
from promptstats.core.ranking import bootstrap_ranks, bootstrap_point_advantage
from promptstats.core.variance import (
    robustness_metrics,
    seed_variance_decomposition,
    SeedVarianceResult,
)
from promptstats.core.tokens import TokenUsage, TokenAnalysisResult
from promptstats.core.router import (
    analyze,
    AnalysisBundle,
    AnalysisResult,
    BenchmarkShape,
    MultiModelBundle,
    print_analysis_summary,
)
from promptstats.vis.advantage import plot_point_advantage
from promptstats.vis.critical_difference import plot_critical_difference
from promptstats.io import from_dataframe, DataLoadReport
from promptstats.compare import (
    compare_prompts,
    compare_models,
    CompareReport,
    EntityStats,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "MultiModelBenchmark",
    "pairwise_differences",
    "all_pairwise",
    "vs_baseline",
    "friedman_nemenyi",
    "FriedmanResult",
    "bootstrap_ranks",
    "bootstrap_point_advantage",
    "robustness_metrics",
    "seed_variance_decomposition",
    "SeedVarianceResult",
    "TokenUsage",
    "TokenAnalysisResult",
    "analyze",
    "AnalysisBundle",
    "AnalysisResult",
    "BenchmarkShape",
    "MultiModelBundle",
    "print_analysis_summary",
    "plot_point_advantage",
    "plot_critical_difference",
    "from_dataframe",
    "DataLoadReport",
    "compare_prompts",
    "compare_models",
    "CompareReport",
    "EntityStats",
]

# LMMInfo is exported lazily so that pymer4 is not a hard dependency.
# Access via: from promptstats.core.mixed_effects import LMMInfo
# or inspect bundle.lmm_info at runtime.
try:
    from promptstats.core.mixed_effects import LMMInfo
    __all__ = __all__ + ["LMMInfo"]
except ImportError:
    pass
