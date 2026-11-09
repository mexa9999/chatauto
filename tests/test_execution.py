from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from citysim.config import EvolutionConfig, quick_city_config
from citysim.evolution import (
    Individual,
    evaluate_population,
    layout_from_config,
)
from citysim.execution import EpisodeExecutor, EpisodeTask
from citysim.policy import random_genome


class TestEpisodeExecution(unittest.TestCase):
    def test_serial_and_process_episode_results_match(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        evolution = EvolutionConfig(
            population_size=2,
            generations=1,
            episodes_per_generation=1,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        layout = layout_from_config(evolution)
        genome = random_genome(np.random.default_rng(17), layout)
        tasks = [
            EpisodeTask([genome, genome], seed)
            for seed in (31, 32)
        ]

        with EpisodeExecutor(
            city, layout, parallel=False, max_workers=1
        ) as serial_executor:
            serial = serial_executor.map(tasks)
        with EpisodeExecutor(
            city, layout, parallel=True, max_workers=2
        ) as process_executor:
            parallel = process_executor.map(tasks)

        self.assertEqual(len(serial), len(parallel))
        for serial_result, parallel_result in zip(serial, parallel):
            np.testing.assert_allclose(serial_result[0], parallel_result[0])
            np.testing.assert_allclose(serial_result[1], parallel_result[1])
            np.testing.assert_allclose(serial_result[2], parallel_result[2])

    def test_population_evaluation_matches_serial_and_parallel(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        layout_config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        layout = layout_from_config(layout_config)
        rng = np.random.default_rng(29)
        population = [
            Individual(genome=random_genome(rng, layout))
            for _ in range(layout_config.population_size)
        ]

        serial = evaluate_population(
            population,
            city,
            layout_config,
            seeds=[3, 5],
            episodes_per_seed=2,
        )
        parallel = evaluate_population(
            population,
            city,
            replace(layout_config, use_parallel=True, max_workers=2),
            seeds=[3, 5],
            episodes_per_seed=2,
        )

        np.testing.assert_allclose(
            serial["per_individual_fitness"],
            parallel["per_individual_fitness"],
        )
        np.testing.assert_allclose(
            serial["per_individual_behavior"],
            parallel["per_individual_behavior"],
        )
        self.assertEqual(serial["per_seed"], parallel["per_seed"])
        self.assertAlmostEqual(serial["fitness"]["mean"], parallel["fitness"]["mean"])


if __name__ == "__main__":
    unittest.main()
