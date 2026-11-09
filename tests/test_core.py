"""Core smoke tests (unittest, no extra dependencies)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from citysim.config import (
    CityConfig,
    EvolutionConfig,
    REQUIREMENT_PROFILES,
    quick_city_config,
)
from citysim.episode import run_episode, score_episode
from citysim.evolution import EvolutionEngine
from citysim.policy import (
    DEFAULT_LAYOUT,
    GenomeLayout,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    random_genome,
)
from citysim.world import Action, CityEnvironment


class TestGenome(unittest.TestCase):
    def test_parameter_count(self) -> None:
        self.assertEqual(DEFAULT_LAYOUT.parameter_count, 6217)
        rng = np.random.default_rng(0)
        genome = random_genome(rng, DEFAULT_LAYOUT)
        self.assertEqual(genome.shape, (6217,))
        policy = NeuralPolicy(genome, DEFAULT_LAYOUT)
        self.assertEqual(policy.W1.shape, (22, 64))
        self.assertEqual(policy.W_type.shape, (64, 5))

    def test_custom_hidden_size(self) -> None:
        layout = GenomeLayout(hidden_size=16)
        self.assertEqual(layout.parameter_count, 22 * 16 + 16 + 16 * 16 + 16 + 16 + 1 + 16 * 5 + 5 + 16 * 3 + 3)


class TestEnvironment(unittest.TestCase):
    def test_observation_and_action(self) -> None:
        config = quick_city_config(n_agents=3, actions_per_agent=2)
        env = CityEnvironment(config, seed=7)
        obs = env.observe(0)
        self.assertEqual(obs.cell_features.shape, (config.n_cells, 16))
        self.assertEqual(obs.global_features.shape, (6,))
        self.assertEqual(obs.action_mask.shape, (config.n_cells, 5, 3))
        self.assertEqual(obs.cost_table.shape, (config.n_cells, 5, 3))

        cells = np.flatnonzero(obs.valid_cells)
        self.assertGreater(cells.size, 0)
        cell = int(cells[0])
        types, qualities = np.where(obs.action_mask[cell])
        self.assertGreater(types.size, 0)
        action = Action(
            cell=cell,
            type_id=int(types[0]),
            quality=int(qualities[0]),
            cost=float(obs.cost_table[cell, types[0], qualities[0]]),
        )
        before = env.agents[0].budget
        accepted = env.apply_action(0, action)
        self.assertTrue(accepted)
        self.assertEqual(env.type_grid.ravel()[cell], int(types[0]) + 1)
        self.assertLess(env.agents[0].budget, before)
        env.end_round()
        metrics = env.city_metrics()
        self.assertEqual(metrics.built_cells, 1)
        self.assertGreaterEqual(metrics.city_quality, 0.0)
        self.assertLessEqual(metrics.city_quality, 1.0)

    def test_reject_occupied_cell(self) -> None:
        config = quick_city_config(n_agents=2, actions_per_agent=1)
        env = CityEnvironment(config, seed=3)
        obs = env.observe(0)
        cell = int(np.flatnonzero(obs.valid_cells)[0])
        types, qualities = np.where(obs.action_mask[cell])
        action = Action(cell, int(types[0]), int(qualities[0]))
        self.assertTrue(env.apply_action(0, action))
        obs2 = env.observe(1)
        self.assertFalse(obs2.valid_cells[cell])
        self.assertFalse(env.apply_action(1, action))

    def test_industry_buffer_mask(self) -> None:
        config = CityConfig(
            height=6,
            width=6,
            n_agents=2,
            actions_per_agent=2,
            initial_budget=10_000,
            requirements=REQUIREMENT_PROFILES["hard"],
        )
        env = CityEnvironment(config, seed=11)
        buildable = np.argwhere(env.buildable)
        self.assertGreater(buildable.shape[0], 0)
        row, col = (int(x) for x in buildable[0])
        self.assertTrue(env.apply_action(0, Action(row * config.width + col, 0, 1)))
        obs = env.observe(1)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < config.height and 0 <= c < config.width:
                    cell = r * config.width + c
                    if cell != row * config.width + col:
                        self.assertFalse(obs.action_mask[cell, 3].any())

    def test_policies_return_valid_or_pass(self) -> None:
        config = quick_city_config(n_agents=2, actions_per_agent=2)
        env = CityEnvironment(config, seed=5)
        rng = np.random.default_rng(0)
        for policy in [
            RandomPolicy(),
            GreedyPolicy(),
            ManualPolicy(),
            NeuralPolicy(random_genome(rng), DEFAULT_LAYOUT),
        ]:
            obs = env.observe(0)
            action = policy.act(env, 0, rng)
            if action.cell is not None:
                cell = int(action.cell)
                self.assertTrue(obs.valid_cells[cell])
                self.assertTrue(obs.action_mask[cell, action.type_id, action.quality])


class TestEpisode(unittest.TestCase):
    def test_episode_and_score(self) -> None:
        config = quick_city_config(n_agents=4, actions_per_agent=2)
        policies = [RandomPolicy() for _ in range(config.n_agents)]
        result = run_episode(config, policies, seed=42, record_history=True)
        self.assertEqual(len(result.agents), config.n_agents)
        self.assertEqual(len(result.history), config.actions_per_agent)
        scores = score_episode(result)
        self.assertEqual(scores.shape, (config.n_agents,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreaterEqual(result.city.city_quality, 0.0)
        self.assertLessEqual(result.city.city_quality, 1.0)

    def test_tiny_evolution(self) -> None:
        city = quick_city_config(n_agents=3, actions_per_agent=2)
        evo = EvolutionConfig(
            population_size=8,
            generations=2,
            episodes_per_generation=3,
            elites=2,
            islands=2,
            fresh_blood_count=1,
            verbose=False,
        )
        engine = EvolutionEngine(evo, city)
        result = engine.run()
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.best_genome.shape, (6217,))
        self.assertTrue(np.isfinite(result.best_fitness))
        self.assertEqual(len(result.population), 8)
        self.assertTrue(all(ind.episodes >= 1 for ind in result.population))


class TestRequirements(unittest.TestCase):
    def test_profiles_are_ordered(self) -> None:
        self.assertGreater(
            REQUIREMENT_PROFILES["hard"].green_min,
            REQUIREMENT_PROFILES["medium"].green_min,
        )
        self.assertGreater(
            REQUIREMENT_PROFILES["hard"].industry_buffer,
            REQUIREMENT_PROFILES["soft"].industry_buffer,
        )


if __name__ == "__main__":
    unittest.main()
