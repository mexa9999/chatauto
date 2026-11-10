"""Core smoke and unit tests for the second version of CitySim."""

from __future__ import annotations

import argparse
import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from citysim.citizens import CitizenPool
from citysim.config import (
    BID_MAX,
    EvolutionConfig,
    MAX_FLOORS_BY_ACTION,
    N_CELL_FEATURES,
    N_GLOBAL_FEATURES,
    REGULATION_SCENARIOS,
    REQUIREMENT_PROFILES,
    quick_city_config,
)
from citysim.episode import run_episode, score_breakdown, score_episode
from citysim.evolution import EvolutionEngine
from citysim.grid import district_ids
from citysim.isometric import building_height, cell_center, hit_test
from citysim.metrics import (
    cluster_counts,
    grid_distance,
    kendall_tau,
    local_entropy,
    morans_i,
    spatial_metrics,
    type_distribution,
)
from citysim.policy import (
    DEFAULT_LAYOUT,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    random_genome,
)
from citysim.world import Action, CityEnvironment


def first_buildable(env: CityEnvironment, skip: int = 0) -> int:
    cells = np.flatnonzero(env.buildable.ravel())
    return int(cells[skip])


class TestGenome(unittest.TestCase):
    def test_parameter_count_and_shapes(self) -> None:
        layout = DEFAULT_LAYOUT
        self.assertEqual(N_CELL_FEATURES, 24)
        self.assertEqual(N_GLOBAL_FEATURES, 10)
        self.assertEqual(layout.cell_mlp_input, 45)
        self.assertEqual(layout.parameter_count, 24136)
        self.assertLessEqual(layout.parameter_count, 5 * 22860)
        rng = np.random.default_rng(0)
        genome = random_genome(rng, layout)
        self.assertEqual(genome.shape, (24136,))
        policy = NeuralPolicy(genome, layout)
        self.assertEqual(policy.W1.shape, (45, 48))
        self.assertEqual(policy.W_type.shape, (48, 5))
        self.assertEqual(policy.W_floors.shape, (48, 20))
        self.assertEqual(policy.W_roof.shape, (48, 4))

    def test_old_genome_layouts_are_rejected(self) -> None:
        for size in (22860, 6540, 6491):
            with self.subTest(size=size), self.assertRaises(ValueError):
                DEFAULT_LAYOUT.normalize(np.zeros(size))

    def test_per_cell_gru_state_updates_and_resets(self) -> None:
        policy = NeuralPolicy(random_genome(np.random.default_rng(8)))
        cell_hidden = np.ones((12, DEFAULT_LAYOUT.hidden_size))
        first = policy._update_recurrent_state(cell_hidden, 3, 4)
        self.assertEqual(policy._recurrent_state.shape, cell_hidden.shape)
        policy._update_recurrent_state(cell_hidden * 2.0, 3, 4)
        self.assertFalse(np.allclose(first, policy._recurrent_state))
        policy.reset_state()
        self.assertIsNone(policy._recurrent_state)
        policy._update_recurrent_state(cell_hidden, 2, 6)
        self.assertEqual(policy._recurrent_shape, (2, 6))

    def test_episode_reuse_resets_policy_state(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2)
        genome = random_genome(np.random.default_rng(9))
        policies = [NeuralPolicy(genome.copy()) for _ in range(city.n_agents)]
        first = run_episode(city, policies, seed=10)
        second = run_episode(city, policies, seed=10)
        np.testing.assert_array_equal(score_episode(first), score_episode(second))
        np.testing.assert_array_equal(first.type_grid, second.type_grid)

    def test_forward_scores_are_finite(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        env = CityEnvironment(city, seed=3)
        obs = env.observe(0)
        policy = NeuralPolicy(random_genome(np.random.default_rng(1)))
        forward = policy._forward(obs)
        self.assertEqual(forward["scores"].shape, (city.n_cells,))
        self.assertTrue(np.isfinite(forward["scores"]).all())
        self.assertEqual(forward["shape"].tolist(), [city.height, city.width])

    def test_neural_bids_are_capped(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2)
        env = CityEnvironment(city, seed=4)
        policy = NeuralPolicy(random_genome(np.random.default_rng(2)))
        actions = policy.propose_bids(env, 0, np.random.default_rng(0), count=3)
        self.assertGreater(len(actions), 0)
        for action in actions:
            self.assertLessEqual(action.bid, BID_MAX * city.initial_budget + 1e-9)
            self.assertLessEqual(action.bid, city.initial_budget / 10.0 + 1e-9)

    def test_neural_heads_choose_factorized_dimensions(self) -> None:
        city = quick_city_config(
            n_agents=1, actions_per_agent=2, initial_budget=50_000
        )
        env = CityEnvironment(city, seed=5)
        policy = NeuralPolicy(np.zeros(DEFAULT_LAYOUT.parameter_count))
        policy.b_floors[11] = 100.0
        policy.b_roof[3] = 100.0
        action = policy.act(env, 0, np.random.default_rng(6))
        self.assertFalse(action.is_pass)
        self.assertEqual(action.floors, 12)
        self.assertEqual(action.roof_quality, 3)

        old_space = CityEnvironment(
            replace(city, enable_extended_actions=False), seed=5
        )
        neutral = policy.act(old_space, 0, np.random.default_rng(6))
        self.assertEqual(neutral.floors, 1)
        self.assertEqual(neutral.roof_quality, 0)


class TestObservationAndEconomy(unittest.TestCase):
    def test_observation_contract(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        env = CityEnvironment(city, seed=11)
        obs = env.observe(0)
        self.assertEqual(obs.cell_features.shape, (city.n_cells, N_CELL_FEATURES))
        self.assertEqual(obs.global_features.shape, (N_GLOBAL_FEATURES,))
        self.assertEqual(obs.action_mask.shape, (city.n_cells, 5, 3))
        self.assertEqual(obs.demolish_mask.shape, (city.n_cells,))
        self.assertEqual(obs.demolish_value.shape, (city.n_cells,))
        self.assertTrue(np.isfinite(obs.demolish_value).all())
        self.assertEqual(obs.cost_table.shape, (city.n_cells, 5, 3))
        self.assertTrue(np.isfinite(obs.cell_features).all())
        self.assertTrue(np.isfinite(obs.global_features).all())
        self.assertEqual(getattr(obs, "_shape_hint"), (city.height, city.width))
        labels, n_districts = district_ids(city.height, city.width, 3)
        self.assertEqual(getattr(obs, "_n_districts"), n_districts)

    def test_factorized_action_dimensions_change_cost_and_city_capacity(self) -> None:
        city = quick_city_config(
            n_agents=1, actions_per_agent=2, initial_budget=50_000
        )
        env = CityEnvironment(city, seed=111)
        cell = first_buildable(env)
        base_cost = env.cost_of(cell, type_id=1, quality=1)
        self.assertAlmostEqual(
            base_cost,
            float(env.observe(0).cost_table[cell, 1, 1]),
        )
        self.assertGreater(
            env.cost_of(cell, 1, 1, floors=8)
            - env.cost_of(cell, 1, 1, floors=1),
            env.cost_of(cell, 1, 1, floors=2)
            - env.cost_of(cell, 1, 1, floors=1),
        )
        self.assertTrue(
            env.apply_action(
                0,
                Action(
                    cell=cell,
                    type_id=1,
                    quality=1,
                    floors=MAX_FLOORS_BY_ACTION[1],
                    roof_quality=3,
                ),
            )
        )
        row, col = divmod(cell, env.w)
        self.assertEqual(int(env.floors_grid[row, col]), 20)
        self.assertEqual(int(env.roof_quality_grid[row, col]), 3)
        self.assertEqual(env.jobs, 20 * 8.0)
        self.assertGreater(float(env.cost_grid[row, col]), base_cost)
        self.assertEqual(env.actions[-1].floors, 20)
        self.assertEqual(env.actions[-1].roof_quality, 3)

        stable_city = replace(
            city,
            enable_externalities=False,
            enable_district_synergy=False,
        )
        lowrise = CityEnvironment(stable_city, seed=111)
        highrise = CityEnvironment(stable_city, seed=111)
        test_cell = first_buildable(lowrise)
        self.assertTrue(
            lowrise.apply_action(
                0, Action(cell=test_cell, type_id=1, quality=1, floors=1)
            )
        )
        self.assertTrue(
            highrise.apply_action(
                0, Action(cell=test_cell, type_id=1, quality=1, floors=9)
            )
        )
        lowrise_income = float(lowrise._income_grid().ravel()[test_cell])
        highrise_income = float(highrise._income_grid().ravel()[test_cell])
        self.assertAlmostEqual(highrise_income / lowrise_income, 3.0, places=6)

        invalid_env = CityEnvironment(city, seed=112)
        self.assertFalse(
            invalid_env.apply_action(
                0,
                Action(
                    cell=first_buildable(invalid_env),
                    type_id=1,
                    floors=MAX_FLOORS_BY_ACTION[1] + 1,
                ),
            )
        )

    def test_greedy_actions_keep_neutral_extended_dimensions(self) -> None:
        city = quick_city_config(
            n_agents=1, actions_per_agent=2, initial_budget=5000
        )
        env = CityEnvironment(city, seed=113)
        action = GreedyPolicy().act(env, 0, np.random.default_rng(114))
        if not action.is_pass and not action.demolish:
            self.assertEqual(action.floors, 1)
            self.assertEqual(action.roof_quality, 0)

    def test_district_synergy_accumulates_and_decays(self) -> None:
        city = replace(
            quick_city_config(
                n_agents=1, actions_per_agent=3, initial_budget=50_000
            ),
            enable_externalities=False,
        )
        env = CityEnvironment(city, seed=115)
        district_cells = np.flatnonzero(
            (env.district_labels.ravel() == env.district_labels.ravel()[first_buildable(env)])
            & env.buildable.ravel()
        )
        first, second = (int(cell) for cell in district_cells[:2])
        self.assertTrue(
            env.apply_action(0, Action(cell=first, type_id=0, quality=1))
        )
        self.assertTrue(
            env.apply_action(0, Action(cell=second, type_id=1, quality=1))
        )
        district = int(env.district_labels.ravel()[first])
        self.assertGreater(float(env.district_synergy_target[district]), 0.0)
        income_without_lag = env._income_grid().copy()
        greedy_projection = env.projected_income_table(0).copy()
        env.end_round()
        accumulated = float(env.district_synergy_state[district])
        self.assertGreater(accumulated, 0.0)
        income_with_lag = env._income_grid()
        self.assertGreater(
            float(income_with_lag.ravel()[first]),
            float(income_without_lag.ravel()[first]),
        )
        np.testing.assert_array_equal(
            env.projected_income_table(0),
            greedy_projection,
        )

        env._clear_cell(*divmod(second, env.w))
        env._refresh_layers(advance_population=False)
        env.end_round()
        self.assertLess(float(env.district_synergy_state[district]), accumulated)

        disabled = CityEnvironment(
            replace(city, enable_district_synergy=False), seed=115
        )
        self.assertTrue(np.all(disabled.district_synergy == 0.0))

    def test_isometric_cell_hit_testing(self) -> None:
        height, width = 4, 5
        types = np.zeros((height, width), dtype=np.int8)
        qualities = np.full((height, width), -1, dtype=np.int8)
        origin = (height * 16.0, 48.0)
        for row in range(height):
            for col in range(width):
                elevation = building_height(
                    int(types[row, col]), int(qualities[row, col]), 16.0
                )
                point = cell_center(
                    row, col, 32.0, 16.0, *origin, elevation
                )
                self.assertEqual(
                    hit_test(
                        point, types, qualities, 32.0, 16.0, *origin
                    ),
                    row * width + col,
                )
        types[2, 3] = 4
        qualities[2, 3] = 2
        elevated_center = cell_center(
            2,
            3,
            32.0,
            16.0,
            *origin,
            building_height(4, 2, 16.0),
        )
        self.assertEqual(
            hit_test(elevated_center, types, qualities, 32, 16, *origin),
            2 * width + 3,
        )
        self.assertGreater(building_height(4, 2, 16), building_height(4, 0, 16))
        self.assertGreater(
            building_height(4, 2, 16, floors=12),
            building_height(4, 2, 16, floors=1),
        )

    def test_land_value_pollution_and_noise(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2, initial_budget=2000)
        env = CityEnvironment(city, seed=12)
        self.assertTrue(((env.land_value >= 0.0) & (env.land_value <= 1.0)).all())
        self.assertTrue(((env.pollution >= 0.0) & (env.pollution <= 1.0)).all())
        self.assertTrue(((env.noise >= 0.0) & (env.noise <= 1.0)).all())
        cell = first_buildable(env)
        clean = float(env.pollution.ravel()[cell])
        row, col = divmod(cell, env.w)
        before = env.pollution.copy()
        ok = env.apply_action(0, Action(cell=cell, type_id=3, quality=2))
        self.assertTrue(ok)
        self.assertGreater(float(env.pollution[row, col]), clean)
        self.assertGreater(float(env.pollution.sum()), float(before.sum()))
        self.assertGreater(float(env.local_same[4].sum()), 0.0)

    def test_global_stocks_and_utilization(self) -> None:
        city = quick_city_config(
            n_agents=2, actions_per_agent=4, initial_budget=5000
        )
        city = replace(city, requirements=REQUIREMENT_PROFILES["soft"])
        env = CityEnvironment(city, seed=13)
        cells = np.flatnonzero(env.buildable.ravel())[:8]
        # Two housing, two commercial, two industry, one green.
        plan = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 3), (5, 3), (6, 4)]
        for cell, type_id in plan:
            self.assertTrue(
                env.apply_action(0, Action(cell=int(cells[cell]), type_id=type_id, quality=1))
            )
        env.end_round()
        self.assertGreater(env.population, 0.0)
        self.assertGreater(env.jobs, 0.0)
        self.assertGreater(env.goods_supply, 0.0)
        self.assertGreaterEqual(env.goods_demand, 0.0)
        self.assertGreaterEqual(env.goods_balance, -1.0)
        self.assertLessEqual(env.goods_balance, 1.0)
        for value in env.utilization.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(env.mean_utilization, 0.0)
        self.assertLessEqual(env.mean_utilization, 1.0)

    def test_demolition_refund(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2, initial_budget=2000)
        env = CityEnvironment(city, seed=14)
        cell = first_buildable(env)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=0, quality=1)))
        row, col = divmod(cell, env.w)
        cost = float(env.cost_grid[row, col])
        budget_before = env.agents[0].budget
        self.assertTrue(env.apply_action(0, Action(cell=cell, demolish=True)))
        self.assertEqual(int(env.type_grid[row, col]), 0)
        self.assertAlmostEqual(
            env.agents[0].budget, budget_before + 0.5 * cost, places=6
        )
        self.assertAlmostEqual(env.agents[0].refund_received, 0.5 * cost, places=6)
        self.assertAlmostEqual(env.cost_grid[row, col], 0.0, places=6)

    def test_neural_policy_can_demolish_in_sequential_and_auction_actions(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2, initial_budget=5000)
        env = CityEnvironment(city, seed=141)
        cell = first_buildable(env)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=0, quality=1)))
        genome = random_genome(np.random.default_rng(142))
        policy = NeuralPolicy(genome)
        policy.b_demolish[0] = 100.0
        action = policy.act(env, 0, np.random.default_rng(143))
        self.assertTrue(action.demolish)
        self.assertEqual(action.cell, cell)
        bids = policy.propose_bids(env, 0, np.random.default_rng(144), count=1)
        self.assertEqual(len(bids), 1)
        self.assertTrue(bids[0].demolish)
        self.assertEqual(bids[0].bid, 0.0)

    def test_baseline_policies_offer_owned_demolitions(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2, initial_budget=5000)
        env = CityEnvironment(city, seed=145)
        cell = first_buildable(env)
        other_cell = first_buildable(env, skip=1)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=0, quality=1)))
        self.assertTrue(env.apply_action(1, Action(cell=other_cell, type_id=0, quality=1)))
        self.assertFalse(env.observe(0).demolish_mask[other_cell])
        row, col = divmod(cell, env.w)
        env.cost_grid[row, col] = 1e6
        env._income_cache = None
        env._current_income = None
        for policy in (GreedyPolicy(), ManualPolicy()):
            action = policy.act(env, 0, np.random.default_rng(146))
            self.assertTrue(action.demolish)
            self.assertEqual(action.cell, cell)
        for policy in (RandomPolicy(), GreedyPolicy(), ManualPolicy()):
            actions = policy.propose_bids(
                env, 0, np.random.default_rng(147), count=env.n_cells
            )
            demolitions = [action for action in actions if action.demolish]
            self.assertTrue(demolitions)
            self.assertTrue(all(action.cell == cell and action.bid == 0.0 for action in demolitions))

    def test_bankruptcy_clears_buildings(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=3, initial_budget=5000)
        env = CityEnvironment(city, seed=15)
        cell = first_buildable(env)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=0, quality=1)))
        row, col = divmod(cell, env.w)
        env.agents[0].budget = -1000.0
        for _ in range(3):
            env.end_round()
        self.assertTrue(env.agents[0].bankrupt)
        self.assertEqual(int(env.type_grid[row, col]), 0)
        self.assertEqual(int(env.owner_grid[row, col]), -1)
        self.assertEqual(env.agents[0].budget, 0.0)


class TestActionModes(unittest.TestCase):
    def _run(self, policy_factory, use_auction: bool):
        city = quick_city_config(
            n_agents=4,
            actions_per_agent=3,
            initial_budget=1000,
            use_auction=use_auction,
        )
        policies = [policy_factory() for _ in range(city.n_agents)]
        return run_episode(city, policies, seed=21, record_history=True)

    def test_auction_and_sequential_modes(self) -> None:
        auction = self._run(GreedyPolicy, True)
        sequential = self._run(GreedyPolicy, False)
        for result in (auction, sequential):
            self.assertGreater(result.city.built_cells, 0)
            self.assertEqual(len(result.history), result.config.actions_per_agent)
            self.assertTrue(np.isfinite(score_episode(result)).all())
        self.assertGreater(
            sum(a.bid_paid for a in auction.agents), 0.0
        )
        self.assertEqual(
            sum(a.bid_paid for a in sequential.agents), 0.0
        )
        winning_bids = [a.bid for a in auction.actions if a.accepted and not a.demolish]
        self.assertTrue(any(bid > 0 for bid in winning_bids))

    def test_buildings_per_action(self) -> None:
        city = quick_city_config(
            n_agents=3, actions_per_agent=2, initial_budget=5000,
            buildings_per_action=2,
        )
        result = run_episode(
            city, [GreedyPolicy() for _ in range(3)], seed=22
        )
        per_round: dict[tuple[int, int], int] = {}
        for record in result.actions:
            if record.accepted and not record.demolish:
                key = (record.round, record.agent_id)
                per_round[key] = per_round.get(key, 0) + 1
        self.assertEqual(max(per_round.values(), default=0), 2)
        builds_per_agent = [a.builds for a in result.agents]
        self.assertGreater(max(builds_per_agent), 1)
        self.assertLessEqual(max(builds_per_agent), 4)


class TestCitizensAndMetrics(unittest.TestCase):
    def test_satisfaction_metrics(self) -> None:
        city = quick_city_config(n_agents=3, actions_per_agent=3)
        result = run_episode(
            city, [ManualPolicy() for _ in range(3)], seed=31,
            record_history=True,
        )
        metrics = result.city
        for value in (
            metrics.mean_satisfaction,
            metrics.gini_satisfaction,
            metrics.rich_satisfaction,
            metrics.poor_satisfaction,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(len(result.history), city.actions_per_agent)
        history_row = result.history[-1]
        self.assertIn("mean_satisfaction", history_row)
        self.assertIn("bankrupt_agents", history_row)
        unshared_config = replace(
            result.config,
            fitness=replace(result.config.fitness, shared_reward=0.0),
        )
        unshared_result = replace(result, config=unshared_config)
        expected_bonus = result.config.fitness.shared_reward * 0.5 * (
            result.city.city_quality + result.city.mean_satisfaction
        )
        np.testing.assert_allclose(
            score_episode(result) - score_episode(unshared_result),
            expected_bonus,
        )

    def test_citizen_pool_reproducible(self) -> None:
        a = CitizenPool(seed=5)
        b = CitizenPool(seed=5)
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        env = CityEnvironment(city, seed=32)
        self.assertEqual(a.evaluate_city(env), b.evaluate_city(env))

    def test_spatial_metrics(self) -> None:
        block = np.zeros((9, 9), dtype=np.int8)
        block[0:3, 0:3] = 1
        block[0:3, 3:6] = 2
        block[0:3, 6:9] = 3
        block[3:6, 0:3] = 4
        block[3:6, 3:6] = 5
        block[3:6, 6:9] = 1
        block[6:9, 0:3] = 2
        block[6:9, 3:6] = 4
        block[6:9, 6:9] = 5
        random_grid = np.random.default_rng(0).integers(0, 6, size=(12, 12))
        self.assertGreater(morans_i(block), 0.05)
        self.assertGreater(morans_i(block), morans_i(random_grid))
        self.assertLess(local_entropy(block), 0.5)
        counts = cluster_counts(block)
        self.assertEqual(sum(counts.values()), 9)
        spatial = spatial_metrics(block)
        for key in ("morans_i", "local_entropy", "cluster_count", "green_share"):
            self.assertIn(key, spatial)
        self.assertTrue(np.isfinite(type_distribution(block)).all())
        self.assertAlmostEqual(float(type_distribution(block).sum()), 1.0)
        self.assertGreaterEqual(kendall_tau([1, 2, 3], [1, 2, 3]), 0.99)
        self.assertEqual(grid_distance(block, block), 0.0)

    def test_fitness_breakdown_and_bankruptcy_penalty(self) -> None:
        city = quick_city_config(n_agents=4, actions_per_agent=3)
        result = run_episode(
            city, [RandomPolicy() for _ in range(4)], seed=33
        )
        breakdown = score_breakdown(result)
        self.assertEqual(breakdown["score"].shape, (4,))
        self.assertTrue(np.isfinite(breakdown["score"]).all())
        self.assertIn("city_quality", breakdown)
        self.assertIn("satisfaction", breakdown)

        def agent(bankrupt: bool):
            return SimpleNamespace(
                profit=50.0,
                builds=2,
                skips=0,
                violations=0.0,
                damage_caused=0.0,
                bankrupt=bankrupt,
            )

        fake = SimpleNamespace(
            agents=[agent(False), agent(True)],
            city=SimpleNamespace(city_quality=0.5, mean_satisfaction=0.5),
            config=city,
        )
        scores = score_breakdown(fake)["score"]
        self.assertAlmostEqual(
            float(scores[0] - scores[1]), city.fitness.bankruptcy, places=6
        )


class TestShocksAndRegulation(unittest.TestCase):
    def test_shocks(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=2, initial_budget=5000)
        env = CityEnvironment(city, seed=41)
        cells = np.flatnonzero(env.buildable.ravel())
        for cell in cells[:4]:
            self.assertTrue(
                env.apply_action(0, Action(cell=int(cell), type_id=3, quality=1))
            )
        event = env.apply_shock("industry_removal", np.random.default_rng(1), 0.5)
        self.assertEqual(event["kind"], "industry_removal")
        self.assertEqual(event["cleared"], 2)
        self.assertEqual(int(np.sum(env.type_grid == 4)), 2)

        env2 = CityEnvironment(city, seed=42)
        env2.apply_shock("population_boom", np.random.default_rng(1), 200)
        self.assertAlmostEqual(env2.population_extra, 200.0)

        env3 = CityEnvironment(city, seed=43)
        for cell in np.flatnonzero(env3.buildable.ravel())[:4]:
            env3.apply_action(0, Action(cell=int(cell), type_id=0, quality=1))
        event3 = env3.apply_shock("earthquake", np.random.default_rng(2), 0.5)
        self.assertEqual(event3["cleared"], 2)
        self.assertEqual(int(np.sum(env3.type_grid > 0)), 2)

    def test_regulation_scenarios(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=3, initial_budget=5000)
        env = CityEnvironment(
            replace(city, regulation=REGULATION_SCENARIOS["green_subsidy"]),
            seed=44,
        )
        cell = first_buildable(env)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=4, quality=1)))
        env.end_round()
        self.assertGreater(env.agents[0].subsidy_received, 0.0)

        env_tax = CityEnvironment(
            replace(city, regulation=REGULATION_SCENARIOS["no_tax"]), seed=45
        )
        cell = first_buildable(env_tax)
        env_tax.apply_action(0, Action(cell=cell, type_id=0, quality=1))
        env_tax.end_round()
        self.assertEqual(env_tax.agents[0].tax_paid, 0.0)

        env_default = CityEnvironment(city, seed=46)
        cell = first_buildable(env_default)
        env_default.apply_action(0, Action(cell=cell, type_id=0, quality=1))
        env_default.end_round()
        self.assertGreater(env_default.agents[0].tax_paid, 0.0)


class TestEvolution(unittest.TestCase):
    def _config(self) -> tuple[EvolutionConfig, CityConfig]:
        city = quick_city_config(n_agents=3, actions_per_agent=2)
        evo = EvolutionConfig(
            population_size=8,
            generations=2,
            episodes_per_generation=3,
            elites=2,
            islands=2,
            fresh_blood_count=1,
            hidden_size=48,
            use_parallel=False,
            verbose=False,
            log_every=1,
            use_map_elites=True,
            map_elites_bins=3,
        )
        return evo, city

    def test_evolution_and_map_elites(self) -> None:
        evo, city = self._config()
        result = EvolutionEngine(evo, city).run()
        self.assertEqual(result.best_genome.shape, (24136,))
        self.assertTrue(np.isfinite(result.best_fitness))
        self.assertEqual(len(result.history), 2)
        row = result.history[-1]
        for key in (
            "best", "mean", "diversity", "eval_satisfaction",
            "eval_city_quality", "eval_utilization", "eval_bankrupt_agents",
        ):
            self.assertIn(key, row)
        self.assertTrue(result.archive)
        for key, individual in result.archive.items():
            self.assertEqual(len(key), 2)
            for bucket in key:
                self.assertGreaterEqual(bucket, 0)
                self.assertLess(bucket, 3)
            self.assertEqual(individual.genome.shape, (24136,))
            self.assertTrue(np.isfinite(individual.fitness))


class TestCli(unittest.TestCase):
    def test_parser_and_configs(self) -> None:
        from citysim.cli import build_configs, build_parser, parse_grid, parse_seeds

        self.assertEqual(parse_grid("12x9"), (12, 9))
        self.assertEqual(parse_grid("12х9"), (12, 9))
        self.assertEqual(parse_grid("12×9"), (12, 9))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_grid("12-by-9")
        self.assertEqual(parse_seeds("0-2"), [0, 1, 2])
        parser = build_parser()
        args = parser.parse_args(
            [
                "experiment", "e12", "--quick", "--buildings-per-action", "2",
                "--requirements", "hard", "--regulation", "no_tax",
                "--no-auction", "--hidden", "32",
            ]
        )
        city, evo, seeds = build_configs(args)
        self.assertEqual(args.projection, "isometric")
        self.assertEqual(city.buildings_per_action, 2)
        self.assertFalse(city.use_auction)
        self.assertEqual(city.requirements.name, "hard")
        self.assertEqual(city.regulation.name, "no_tax")
        self.assertEqual(evo.hidden_size, 32)
        self.assertEqual(seeds, [0, 1])

    def test_load_old_genome_layout_is_rejected(self) -> None:
        from citysim.cli import _load_genome

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-genome.npy"
            np.save(path, np.zeros(22860))
            with self.assertRaises(SystemExit):
                _load_genome(str(path), DEFAULT_LAYOUT, seed=0)

    def test_isometric_and_top_down_png_renderers(self) -> None:
        from citysim.plotting import plot_city, plot_compare_grids

        city = quick_city_config(n_agents=2, actions_per_agent=2)
        env_a = CityEnvironment(city, seed=201)
        env_b = CityEnvironment(city, seed=202)
        with tempfile.TemporaryDirectory() as directory:
            iso_path = plot_city(env_a, Path(directory) / "iso.png")
            top_down_path = plot_city(
                env_a,
                Path(directory) / "top-down.png",
                projection="top-down",
            )
            comparison_path = plot_compare_grids(
                env_a,
                env_b,
                Path(directory) / "compare.png",
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in (
                iso_path,
                top_down_path,
                comparison_path,
            )))


if __name__ == "__main__":
    unittest.main()
