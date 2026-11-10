"""Public experiment API, backed by focused E1-E12 modules."""

from concurrent.futures import ProcessPoolExecutor

from .experiment_support import (
    _aggregate_history,
    _ensure_evolution_feasible,
    _eval_seeds,
    _history_rows,
    _metric_row,
    _require_training_seeds,
    _run_config,
    _run_metric_summary,
    _stats,
    _summary_value,
    train_evolution,
)
from .experiments_learning import (
    _evaluate_behavior_set,
    e1_learning_curve,
    e2_baselines,
)
from .experiments_sweeps import (
    e3_externalities,
    e4_requirements,
    e5_agents,
    e6_generalization,
    e7_diversity,
    e8_districts,
)
from .experiments_robustness import (
    e9_path_dependence,
    e10_regulation,
    e11_shocks,
)
from .experiments_actions import e12_factorial_actions

__all__ = [
    "train_evolution",
    "e1_learning_curve",
    "e2_baselines",
    "e3_externalities",
    "e4_requirements",
    "e5_agents",
    "e6_generalization",
    "e7_diversity",
    "e8_districts",
    "e9_path_dependence",
    "e10_regulation",
    "e11_shocks",
    "e12_factorial_actions",
    "_aggregate_history",
    "_ensure_evolution_feasible",
    "_eval_seeds",
    "_evaluate_behavior_set",
    "_history_rows",
    "_metric_row",
    "_require_training_seeds",
    "_run_config",
    "_run_metric_summary",
    "_stats",
    "_summary_value",
]

_GLOBAL_EXECUTOR: ProcessPoolExecutor | None = None
