from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from citysim.config import EvolutionConfig, quick_city_config
from citysim.experiments_actions import e12_factorial_actions


class TestFactorialActionExperiment(unittest.TestCase):
    def test_e12_runs_all_four_factor_conditions_and_greedy_baseline(self) -> None:
        city = quick_city_config(
            n_agents=2,
            actions_per_agent=2,
            initial_budget=5000,
        )
        evolution = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_generation=2,
            elites=1,
            islands=2,
            use_parallel=False,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = e12_factorial_actions(
                evolution,
                city,
                [17],
                Path(directory),
                eval_seed_count=1,
            )

            conditions = summary["factorial_conditions"]
            self.assertEqual(len(conditions), 5)
            self.assertIn("extended_actions_synergy", conditions)
            self.assertIn("greedy_external_baseline", conditions)
            self.assertTrue((Path(directory) / "e12_factorial.csv").is_file())
            self.assertTrue((Path(directory) / "e12_summary.json").is_file())
            self.assertTrue((Path(directory) / "e12_city_quality.png").is_file())


if __name__ == "__main__":
    unittest.main()
