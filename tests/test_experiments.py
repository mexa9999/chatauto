from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from citysim.config import EvolutionConfig, quick_city_config
from citysim import experiments


class TestExperiments(unittest.TestCase):
    def test_invalid_seed_counts_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_seed_count"):
            experiments._eval_seeds(100, 0)
        with self.assertRaisesRegex(ValueError, "training seed"):
            experiments._require_training_seeds([], "E1")

    def test_worker_count_uses_all_available_cpus_up_to_task_count(self) -> None:
        from unittest.mock import patch

        from citysim.config import quick_city_config, quick_evolution_config
        from citysim.evolution import EvolutionEngine

        with (
            patch("citysim.evolution.os.process_cpu_count", return_value=16),
            patch("citysim.evolution.ProcessPoolExecutor") as pool,
        ):
            EvolutionEngine(
                quick_evolution_config(),
                quick_city_config(),
            )

        self.assertEqual(pool.call_args.kwargs["max_workers"], 4)

    def test_worker_count_can_use_more_than_eight_cpus(self) -> None:
        from dataclasses import replace
        from citysim.config import quick_city_config, quick_evolution_config
        from citysim.evolution import EvolutionEngine

        config = replace(
            quick_evolution_config(),
            episodes_per_generation=20,
            max_workers=16,
        )
        with (
            patch("citysim.evolution.os.process_cpu_count", return_value=16),
            patch("citysim.evolution.ProcessPoolExecutor") as pool,
        ):
            EvolutionEngine(config, quick_city_config())

        self.assertEqual(pool.call_args.kwargs["max_workers"], 16)

    def test_cli_exposes_worker_limit(self) -> None:
        from citysim.cli import build_configs, build_parser

        args = build_parser().parse_args(
            ["train", "--quick", "--workers", "12", "--quiet"]
        )
        _, evolution_config, _ = build_configs(args)

        self.assertEqual(evolution_config.max_workers, 12)

    def test_e1_keeps_aggregated_learning_curve(self) -> None:
        history = [
            {
                "gen": 0,
                "best": 1.0,
                "mean": 0.5,
                "diversity": 0.2,
                "unique_behaviors": 3,
            }
        ]
        result = SimpleNamespace(
            history=history,
            best_genome=[],
            best_fitness=1.0,
            best_generation=0,
            best_behavior=[],
        )
        with tempfile.TemporaryDirectory() as output:
            with (
                patch.object(experiments, "EvolutionEngine") as engine,
                patch.object(experiments, "save_genome"),
                patch.object(experiments, "write_csv"),
                patch.object(experiments, "write_json"),
                patch.object(experiments, "plot_learning_curve") as plot_curve,
                patch.object(experiments, "plot_satisfaction_curve"),
            ):
                engine.return_value.run.return_value = result
                experiments.e1_learning_curve(
                    EvolutionConfig(generations=1, use_parallel=False),
                    quick_city_config(n_agents=2, actions_per_agent=1),
                    [1],
                    output,
                )

        plot_curve.assert_called_once()

    def test_e4_aggregates_each_profile_pair_across_seeds(self) -> None:
        metric_keys = (
            "fitness",
            "profit",
            "city_quality",
            "mean_satisfaction",
            "green_share",
            "type_entropy",
            "accessibility",
            "violations",
            "damage_total",
            "built_cells",
        )
        evaluation_index = 0

        def evaluate(*_args, **_kwargs):
            nonlocal evaluation_index
            evaluation_index += 1
            return {
                key: {"mean": float(evaluation_index), "ci95": 0.0}
                for key in metric_keys
            }

        csv_rows: dict[str, list[dict]] = {}
        result = SimpleNamespace(best_genome=[])
        with tempfile.TemporaryDirectory() as output:
            with (
                patch.object(experiments, "EvolutionEngine") as engine,
                patch.object(experiments, "evaluate_homogeneous", side_effect=evaluate),
                patch.object(
                    experiments,
                    "write_csv",
                    side_effect=lambda path, rows: csv_rows.setdefault(
                        Path(path).name, rows
                    ),
                ),
                patch.object(experiments, "write_json"),
                patch.object(experiments, "plot_profit_quality"),
            ):
                engine.return_value.run.return_value = result
                experiments.e4_requirements(
                    EvolutionConfig(generations=1, use_parallel=False),
                    quick_city_config(n_agents=2, actions_per_agent=1),
                    [1, 2],
                    output,
                )

        rows = csv_rows["e4_summary.csv"]
        self.assertEqual(len(rows), 9)
        soft_to_soft = next(
            row
            for row in rows
            if row["trained_profile"] == "soft" and row["eval_profile"] == "soft"
        )
        self.assertEqual(soft_to_soft["fitness_mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
