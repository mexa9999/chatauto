from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from citysim.config import EvolutionConfig, quick_city_config
from citysim import (
    experiments,
    experiments_learning,
    experiments_robustness,
    experiments_sweeps,
)


class TestExperiments(unittest.TestCase):
    def test_invalid_seed_counts_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_seed_count"):
            experiments._eval_seeds(100, 0)
        with self.assertRaisesRegex(ValueError, "training seed"):
            experiments._require_training_seeds([], "E1")

    def test_e5_feasibility_uses_focal_agents(self) -> None:
        from citysim.experiment_support import _ensure_evolution_feasible
        from citysim.evolution import (
            focal_agents_per_episode,
            minimum_episodes_for_population,
        )

        config = EvolutionConfig(
            population_size=32,
            episodes_per_generation=16,
            league_fraction=0.8,
            use_parallel=False,
        )
        self.assertEqual(focal_agents_per_episode(2, 0.8), 1)
        adjusted = _ensure_evolution_feasible(config, 2)
        self.assertEqual(adjusted.population_size, 32)
        self.assertEqual(adjusted.episodes_per_generation, 32)
        self.assertEqual(
            adjusted.episodes_per_generation,
            minimum_episodes_for_population(adjusted.population_size, 1),
        )
        self.assertIs(_ensure_evolution_feasible(config, 10), config)

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

    def test_cli_exposes_resume_and_checkpoint_interval(self) -> None:
        from citysim.cli import build_parser
        from citysim.training_history import DEFAULT_HISTORY_DIR

        args = build_parser().parse_args(
            [
                "experiment",
                "all",
                "--resume",
                "--cache-dir",
                "runs/cache",
                "--checkpoint-every",
                "3",
            ]
        )

        self.assertTrue(args.resume)
        self.assertEqual(args.cache_dir, "runs/cache")
        self.assertEqual(args.checkpoint_every, 3)

        parser = build_parser()
        train_args = parser.parse_args(["train"])
        experiment_args = parser.parse_args(["experiment", "e1"])
        dashboard_args = parser.parse_args(["dashboard", "--no-browser"])
        history_args = parser.parse_args(["history", "branches"])
        self.assertEqual(train_args.history_dir, str(DEFAULT_HISTORY_DIR))
        self.assertEqual(experiment_args.cache_dir, str(DEFAULT_HISTORY_DIR))
        self.assertEqual(dashboard_args.history_dir, str(DEFAULT_HISTORY_DIR))
        self.assertEqual(history_args.history_dir, str(DEFAULT_HISTORY_DIR))

    def test_cli_exposes_dashboard_and_progress_hooks(self) -> None:
        from citysim.cli import build_parser

        parser = build_parser()
        train_args = parser.parse_args(
            [
                "train",
                "--progress-file",
                "progress.json",
                "--stop-file",
                "stop.request",
            ]
        )
        dashboard_args = parser.parse_args(["dashboard", "--no-browser"])

        self.assertEqual(train_args.progress_file, "progress.json")
        self.assertEqual(train_args.stop_file, "stop.request")
        self.assertEqual(dashboard_args.host, "127.0.0.1")
        self.assertTrue(dashboard_args.no_browser)

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
                patch.object(experiments_learning, "EvolutionEngine") as engine,
                patch.object(experiments_learning, "save_genome"),
                patch.object(experiments_learning, "write_csv"),
                patch.object(experiments_learning, "write_json"),
                patch.object(experiments_learning, "plot_learning_curve") as plot_curve,
                patch.object(experiments_learning, "plot_satisfaction_curve"),
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
                patch.object(experiments_sweeps, "EvolutionEngine") as engine,
                patch.object(
                    experiments_sweeps,
                    "evaluate_homogeneous",
                    side_effect=evaluate,
                ),
                patch.object(
                    experiments_sweeps,
                    "write_csv",
                    side_effect=lambda path, rows: csv_rows.setdefault(
                        Path(path).name, rows
                    ),
                ),
                patch.object(experiments_sweeps, "write_json"),
                patch.object(experiments_sweeps, "plot_profit_quality"),
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

    def test_public_facade_reexports_all_experiments(self) -> None:
        public_modules = (
            (experiments_learning, ("e1_learning_curve", "e2_baselines")),
            (
                experiments_sweeps,
                (
                    "e3_externalities",
                    "e4_requirements",
                    "e5_agents",
                    "e6_generalization",
                    "e7_diversity",
                    "e8_districts",
                ),
            ),
            (
                experiments_robustness,
                ("e9_path_dependence", "e10_regulation", "e11_shocks"),
            ),
        )
        for module, names in public_modules:
            for experiment in names:
                with self.subTest(experiment=experiment):
                    self.assertIs(
                        getattr(experiments, experiment),
                        getattr(module, experiment),
                    )


if __name__ == "__main__":
    unittest.main()
