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
    ACTION_TYPE_NAMES,
    BID_MAX,
    CITIZEN_GOODS_DEMAND,
    COMMERCIAL_GOODS_DEMAND,
    EvolutionConfig,
    GLOBAL_FEATURE_NAMES,
    GOODS_IMPORT_MARKUP,
    INDUSTRY_INCOME_MULTIPLIER,
    INCOME_BASE,
    MIGRATION_RATE,
    MAX_FLOORS_BY_ACTION,
    MAX_DEMAND,
    N_CELL_FEATURES,
    N_GLOBAL_FEATURES,
    OPTIMAL_DISTANCE,
    QUALITY_INCOME_MULT,
    ROAD_ACTION_ID,
    ROAD_COST_BASE,
    ROAD_COST_LAND_MULTIPLIER,
    ROAD_DEMOLISH_REFUND,
    ROAD_MAINTENANCE_RATE,
    REGULATION_SCENARIOS,
    REQUIREMENT_PROFILES,
    quick_city_config,
)
from citysim.episode import run_episode, score_breakdown, score_episode
from citysim.evolution import EvolutionEngine
from citysim.grid import district_ids, grid_shortest_paths, road_network_fields
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
    DISTRICT_ALLOWED,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    bid_cap,
    cap_bid,
    random_genome,
    strategic_bid,
)
from citysim.world import Action, CityEnvironment


def first_buildable(env: CityEnvironment, skip: int = 0) -> int:
    cells = np.flatnonzero(env.buildable.ravel())
    return int(cells[skip])


class TestGenome(unittest.TestCase):
    def test_parameter_count_and_shapes(self) -> None:
        layout = DEFAULT_LAYOUT
        self.assertEqual(N_CELL_FEATURES, 29)
        self.assertEqual(N_GLOBAL_FEATURES, 16)
        self.assertEqual(layout.cell_mlp_input, 50)
        rng = np.random.default_rng(0)
        genome = random_genome(rng, layout)
        self.assertEqual(genome.shape, (layout.parameter_count,))
        policy = NeuralPolicy(genome, layout)
        self.assertEqual(policy.W1.shape, (50, 48))
        self.assertEqual(policy.W_type.shape, (48, len(ACTION_TYPE_NAMES)))
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

    def test_actions_per_round_runs_sequential_action_cycles(self) -> None:
        city = replace(
            quick_city_config(n_agents=1, actions_per_agent=1),
            use_auction=False,
            actions_per_round=3,
        )
        observed = []
        run_episode(
            city,
            [RandomPolicy()],
            seed=14,
            on_action=lambda agent, action, accepted, env: observed.append(
                (env.round, agent)
            ),
        )
        self.assertEqual(observed, [(0, 0), (0, 0), (0, 0)])

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


class TestRoadInfrastructure(unittest.TestCase):
    def test_grid_shortest_paths_use_weighted_four_neighbour_routes(self) -> None:
        costs = np.ones((3, 3), dtype=np.float64)
        costs[1, 1] = np.inf
        sources = np.zeros((3, 3), dtype=bool)
        sources[0, 0] = True

        distances = grid_shortest_paths(costs, sources)

        self.assertEqual(distances[1, 1], np.inf)
        self.assertEqual(distances[2, 2], 4.0)

    def test_travel_maps_update_after_settlement_for_next_round(self) -> None:
        env = CityEnvironment(
            quick_city_config(
                n_agents=1, actions_per_agent=2, initial_budget=5000
            ),
            seed=204,
        )
        before = env.travel_time_jobs.copy()
        rows, cols = np.nonzero(env.buildable)
        center_y, center_x = (env.h - 1) / 2.0, (env.w - 1) / 2.0
        farthest = int(
            np.argmax((rows - center_y) ** 2 + (cols - center_x) ** 2)
        )
        row, col = int(rows[farthest]), int(cols[farthest])
        env.type_grid[row, col] = 2
        env.quality_grid[row, col] = 1
        env.floors_grid[row, col] = 1
        env.owner_grid[row, col] = 0
        env.cost_grid[row, col] = 100.0
        env._refresh_layers(advance_population=False)
        np.testing.assert_array_equal(env.travel_time_jobs, before)

        env.end_round()
        observation = env.observe(0)

        self.assertEqual(env.travel_time_jobs[row, col], 0.0)
        self.assertEqual(observation.cell_features.shape[1], 29)
        self.assertEqual(observation.cell_features[row * env.w + col, 28], 0.0)

    def test_congestion_is_quadratic_and_relieved_by_active_roads(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=1), seed=205
        )
        env.density.fill(1.0)
        env.static["water_blocked"][:] = False
        env.static["road_mask"][:] = False
        row, col = env.h // 2, env.w // 2
        env.static["road_mask"][row, col] = True

        without_roads = env._travel_cell_costs()[row, col]
        for road_row, road_col in (
            (row - 1, col - 1),
            (row - 1, col),
            (row, col - 1),
        ):
            env.player_road_grid[road_row, road_col] = True
        with_roads = env._travel_cell_costs()[row, col]

        self.assertAlmostEqual(without_roads, 3.0)
        self.assertAlmostEqual(with_roads, 1.5)

    def test_demand_uses_soft_load_penalty_without_density_bonus(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=1), seed=206
        )
        expected = np.clip(
            env.static["base_demand"]
            + 0.10 * env.static["road"]
            - 0.05 * env.load
            - 0.15 * env.share_industry
            + 0.15 * env.attractiveness
            - 0.10 * env.local_saturation,
            0.10,
            MAX_DEMAND,
        )
        np.testing.assert_allclose(env.demand, expected)

    def test_road_fields_calculate_access_density_and_wait(self) -> None:
        static_access = np.full((3, 3), 0.8)
        static_mask = np.zeros((3, 3), dtype=bool)
        static_mask[1, 0] = True
        active = np.zeros((3, 3), dtype=bool)
        active[1, 1:3] = True
        pending = np.zeros((3, 3), dtype=bool)
        pending[0, 0] = True

        access, density, wait = road_network_fields(
            static_access, static_mask, active, pending
        )

        self.assertAlmostEqual(density[1, 1], 1.0 / 3.0)
        self.assertAlmostEqual(access[1, 1], 0.8 + 0.5 / 3.0)
        self.assertEqual(wait[0, 0], 1.0)
        self.assertEqual(wait[1, 1], 1.0)
        self.assertEqual(wait[2, 2], 0.0)
        clipped, _, _ = road_network_fields(
            np.ones((3, 3)), static_mask, active, pending
        )
        self.assertTrue(np.all(clipped == 1.0))

    def test_roads_are_pending_owned_maintained_demolishable_and_soft(self) -> None:
        city = replace(
            quick_city_config(
                n_agents=1, actions_per_agent=2, initial_budget=5000
            ),
            use_auction=False,
        )
        env = CityEnvironment(city, seed=201)
        cell = first_buildable(env)
        row, col = divmod(cell, env.w)
        expected_cost = (
            ROAD_COST_BASE
            + ROAD_COST_LAND_MULTIPLIER * env.land_value[row, col]
        )
        obs = env.observe(0)
        self.assertEqual(obs.action_mask.shape[1], len(ACTION_TYPE_NAMES))
        self.assertAlmostEqual(obs.cost_table[cell, ROAD_ACTION_ID, 0], expected_cost)
        self.assertTrue(obs.action_mask[cell, ROAD_ACTION_ID, 0])
        self.assertFalse(obs.action_mask[cell, ROAD_ACTION_ID, 1:].any())

        self.assertTrue(
            env.apply_action(
                0, Action(cell=cell, type_id=ROAD_ACTION_ID, bid=3.0)
            )
        )
        self.assertAlmostEqual(env.agents[0].spend, expected_cost)
        self.assertAlmostEqual(env.agents[0].bid_paid, 3.0)
        self.assertEqual(env.type_grid[row, col], 0)
        self.assertEqual(env.pending_road_grid[row, col], 1)
        self.assertFalse(env.player_road_grid[row, col])
        pending_obs = env.observe(0)
        self.assertEqual(pending_obs.cell_features[cell, 26], 1.0)
        self.assertEqual(pending_obs.cell_features[cell, 25], 0.0)
        self.assertFalse(pending_obs.action_mask[cell].any())
        self.assertGreater(pending_obs.demolish_value[cell], 0.0)

        env.end_round()
        self.assertTrue(env.player_road_grid[row, col])
        self.assertEqual(env.pending_road_grid[row, col], 0)
        self.assertFalse(env.observe(0).action_mask[cell].any())
        self.assertAlmostEqual(
            env.agents[0].maintenance, expected_cost * ROAD_MAINTENANCE_RATE
        )
        self.assertGreater(env.player_road_density[row, col], 0.0)
        self.assertAlmostEqual(
            env.road_access[row, col],
            min(1.0, env.static["road"][row, col] + 0.5 * env.player_road_density[row, col]),
        )

        budget_before_demolition = env.agents[0].budget
        self.assertTrue(env.apply_action(0, Action(cell=cell, demolish=True)))
        self.assertAlmostEqual(
            env.agents[0].budget - budget_before_demolition,
            expected_cost * ROAD_DEMOLISH_REFUND,
        )
        self.assertFalse(env.player_road_grid[row, col])
        self.assertEqual(env.road_owner_grid[row, col], -1)

        self.assertTrue(
            env.apply_action(0, Action(cell=cell, type_id=ROAD_ACTION_ID))
        )
        self.assertTrue(env.apply_action(0, Action(cell=cell, demolish=True)))
        self.assertEqual(env.pending_road_grid[row, col], 0)

        env.static["road"][row, col] = 1.0
        env.static["road_mask"][row, col] = True
        env._refresh_layers(advance_population=False)
        self.assertTrue(
            env.apply_action(0, Action(cell=cell, type_id=0, quality=1))
        )

    def test_neural_policy_can_select_road_with_extended_actions(self) -> None:
        city = replace(
            quick_city_config(
                n_agents=1, actions_per_agent=1, initial_budget=5000
            ),
            use_auction=False,
        )
        env = CityEnvironment(city, seed=202)
        policy = NeuralPolicy(np.zeros(DEFAULT_LAYOUT.parameter_count))
        policy.b_type[ROAD_ACTION_ID] = 10.0

        action = policy.act(env, 0, np.random.default_rng(203))

        self.assertEqual(action.type_id, ROAD_ACTION_ID)
        self.assertEqual(action.quality, 0)
        self.assertEqual(action.floors, 1)


class TestObservationAndEconomy(unittest.TestCase):
    def test_observation_contract(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        env = CityEnvironment(city, seed=11)
        obs = env.observe(0)
        self.assertEqual(obs.cell_features.shape, (city.n_cells, N_CELL_FEATURES))
        self.assertEqual(obs.global_features.shape, (N_GLOBAL_FEATURES,))
        self.assertEqual(obs.action_mask.shape, (city.n_cells, 6, 3))
        self.assertEqual(obs.demolish_mask.shape, (city.n_cells,))
        self.assertEqual(obs.demolish_value.shape, (city.n_cells,))
        self.assertTrue(np.isfinite(obs.demolish_value).all())
        self.assertEqual(obs.cost_table.shape, (city.n_cells, 6, 3))
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
        env.end_round()
        accumulated = float(env.district_synergy_state[district])
        self.assertGreater(accumulated, 0.0)
        income_with_lag = env._income_grid()
        self.assertGreater(
            float(income_with_lag.ravel()[first]),
            float(income_without_lag.ravel()[first]),
        )
        greedy_projection = env.projected_income_table(0).copy()
        env.district_synergy = np.ones_like(env.district_synergy)
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

    def test_industry_and_commercial_same_type_interactions_are_positive(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=14,
        )
        row, col = env.h // 2, env.w // 2
        for action_type, expected_same_effect in (
            (0, -0.03),
            (1, 0.03),
            (2, -0.03),
            (3, 0.03),
            (4, -0.03),
        ):
            env.type_grid.fill(0)
            env.type_grid[row, col] = action_type + 1
            env.type_grid[row, col + 1] = action_type + 1
            env._refresh_layers()
            green_bonus = 0.04 if action_type == 4 else 0.0
            with self.subTest(action_type=action_type):
                self.assertAlmostEqual(
                    float(env._compatibility_fields()[action_type, row, col + 1]),
                    1.0 + expected_same_effect + green_bonus,
                )

    def test_nearby_parks_reduce_industry_pollution_and_noise(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=15,
        )
        env.type_grid.fill(0)
        row, col = env.h // 2, env.w // 2 - 2
        env.type_grid[row, col] = 4
        env.type_grid[row, col + 1] = 1
        env.type_grid[row, col + 2] = 1
        env._refresh_layers()
        housing_cell = (row, col + 1)
        pollution_without_park = float(env.pollution[housing_cell])
        noise_without_park = float(env.noise[housing_cell])
        density_without_park = float(env.density[housing_cell])

        env.type_grid[row, col + 2] = 5
        env._refresh_layers()

        self.assertGreater(float(env.share_green[housing_cell]), 0.0)
        self.assertAlmostEqual(float(env.density[housing_cell]), density_without_park)
        self.assertLess(float(env.pollution[housing_cell]), pollution_without_park)
        self.assertLess(float(env.noise[housing_cell]), noise_without_park)

    def test_global_stocks_and_utilization(self) -> None:
        city = quick_city_config(
            n_agents=2, actions_per_agent=4, initial_budget=5000
        )
        city = replace(city, requirements=REQUIREMENT_PROFILES["soft"])
        env = CityEnvironment(city, seed=13)
        cells = np.flatnonzero(env.buildable.ravel())[:8]
        # Two housing, two commercial, two industry, one green, one mixed.
        plan = [
            (0, 0), (1, 0), (2, 1), (3, 1),
            (4, 3), (5, 3), (6, 4), (7, 2),
        ]
        for cell, type_id in plan:
            self.assertTrue(
                env.apply_action(0, Action(cell=int(cells[cell]), type_id=type_id, quality=1))
            )
        env.end_round()
        self.assertGreater(env.population, 0.0)
        self.assertGreater(env.jobs, 0.0)
        self.assertGreater(env.goods_supply, 0.0)
        self.assertAlmostEqual(
            env.goods_demand,
            CITIZEN_GOODS_DEMAND * env.population
            + COMMERCIAL_GOODS_DEMAND * 2.5,
        )
        self.assertGreaterEqual(env.goods_demand, 0.0)
        self.assertGreaterEqual(env.goods_balance, -1.0)
        self.assertLessEqual(env.goods_balance, 1.0)
        self.assertAlmostEqual(
            env.local_goods_coverage,
            min(1.0, env.goods_supply / env.goods_demand),
        )
        for value in env.utilization.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(env.mean_utilization, 0.0)
        self.assertLessEqual(env.mean_utilization, 1.0)

    def test_population_migration_vacancies_and_global_features(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=130,
        )
        env.type_grid.fill(0)
        env.type_grid[1:3, 1:3] = 1
        env.type_grid[4, 4] = 2
        env.floors_grid.fill(0)
        env.floors_grid[1:3, 1:3] = 1
        env.floors_grid[4, 4] = 1
        env._refresh_layers(advance_population=False)

        env.population = 10.0
        env._refresh_layers(advance_population=True)
        expected_population = 10.0 + MIGRATION_RATE * (env.jobs - 10.0)
        self.assertAlmostEqual(env.population, expected_population)
        self.assertAlmostEqual(env.net_migration, expected_population - 10.0)
        self.assertAlmostEqual(env.employed, env.jobs)
        self.assertAlmostEqual(env.unemployed, env.population - env.jobs)
        self.assertAlmostEqual(env.job_vacancies, 0.0)
        self.assertAlmostEqual(
            env.housing_vacancies, env.housing_capacity - env.population
        )

        observation = env.observe(0)
        global_features = dict(zip(GLOBAL_FEATURE_NAMES, observation.global_features))
        self.assertEqual(N_GLOBAL_FEATURES, 16)
        self.assertAlmostEqual(
            global_features["housing_vacancy_rate"],
            env.housing_vacancies / env.housing_capacity,
        )
        self.assertAlmostEqual(
            global_features["unemployment_rate"],
            env.unemployed / env.population,
        )
        self.assertAlmostEqual(
            global_features["net_migration_rate"],
            env.net_migration / env.base_population,
        )
        self.assertAlmostEqual(
            global_features["local_goods_coverage"], env.local_goods_coverage
        )
        population_after_migration = env.population
        migration_after_round = env.net_migration
        env._refresh_layers(advance_population=False)
        self.assertAlmostEqual(env.population, population_after_migration)
        self.assertAlmostEqual(env.net_migration, migration_after_round)

        previous_population = env.population
        env.type_grid[4, 5] = 2
        env.floors_grid[4, 5] = 1
        env._refresh_layers(advance_population=True)
        target_population = min(env.housing_capacity, env.jobs)
        expected_population = previous_population + MIGRATION_RATE * (
            target_population - previous_population
        )
        self.assertAlmostEqual(env.population, expected_population)
        self.assertGreater(env.net_migration, 0.0)
        self.assertLessEqual(env.population, env.housing_capacity)
        self.assertGreater(env.job_vacancies, 0.0)
        self.assertEqual(env.unemployed, 0.0)
        observation = env.observe(0)
        global_features = dict(zip(GLOBAL_FEATURE_NAMES, observation.global_features))
        self.assertAlmostEqual(
            global_features["job_vacancy_rate"], env.job_vacancy_rate
        )
        env.agents[0].debt = 2000.0
        debt_observation = env.observe(0)
        debt_features = dict(zip(GLOBAL_FEATURE_NAMES, debt_observation.global_features))
        self.assertAlmostEqual(debt_features["debt_ratio"], 0.4)
        metrics = env.city_metrics()
        self.assertAlmostEqual(metrics.housing_vacancy_rate, env.housing_vacancy_rate)
        self.assertAlmostEqual(metrics.job_vacancy_rate, env.job_vacancy_rate)
        self.assertAlmostEqual(metrics.unemployment_rate, env.unemployment_rate)
        self.assertAlmostEqual(metrics.net_migration_rate, env.net_migration_rate)
        self.assertAlmostEqual(metrics.total_debt, 2000.0)
        self.assertAlmostEqual(metrics.mean_debt_ratio, 0.4)
        self.assertAlmostEqual(
            metrics.local_goods_coverage, env.local_goods_coverage
        )
        self.assertEqual(metrics.economic_phase, env._economic_phase()[0])
        self.assertAlmostEqual(
            metrics.tax_rate_multiplier, env._economic_phase()[1]
        )

    def test_import_markup_reduces_commercial_and_mixed_income(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=131,
        )
        cells = np.flatnonzero(env.buildable.ravel())[:2]
        for cell, world_type in zip(cells, (2, 3)):
            row, col = divmod(int(cell), env.w)
            env.type_grid[row, col] = world_type
            env.quality_grid[row, col] = 1
            env.floors_grid[row, col] = 1
            env.owner_grid[row, col] = 0
        env._refresh_layers()
        self.assertEqual(env.local_goods_coverage, 0.0)
        imported_income = env._income_grid().copy()
        imported_projections = env.projected_income_table(0)

        env.local_goods_coverage = 1.0
        env._income_cache = None
        local_income = env._income_grid().copy()
        local_projections = env.projected_income_table(0)
        import_factor = 1.0 - GOODS_IMPORT_MARKUP
        for index, cell in enumerate(cells):
            row, col = divmod(int(cell), env.w)
            self.assertAlmostEqual(
                imported_income[row, col],
                import_factor * local_income[row, col],
            )
            action_type = index + 1
            self.assertAlmostEqual(
                imported_projections[cell, action_type, 1],
                import_factor * local_projections[cell, action_type, 1],
            )
        self.assertEqual(imported_projections[cells[0], 0, 1],
                         local_projections[cells[0], 0, 1])

    def test_load_observation_does_not_apply_a_second_income_penalty(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=132,
        )
        cell = first_buildable(env)
        row, col = divmod(cell, env.w)
        env.type_grid[row, col] = 2
        env.quality_grid[row, col] = 1
        env.floors_grid[row, col] = 1
        env.owner_grid[row, col] = 0
        env._refresh_layers()
        income = env._income_grid().copy()
        projections = env.projected_income_table(0).copy()

        env.load = np.ones_like(env.load)
        env._income_cache = None

        np.testing.assert_array_equal(env._income_grid(), income)
        np.testing.assert_array_equal(env.projected_income_table(0), projections)

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

    def test_operating_deficits_accrue_debt_without_liquidation(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=3, initial_budget=5000)
        env = CityEnvironment(city, seed=15)
        cell = first_buildable(env)
        self.assertTrue(env.apply_action(0, Action(cell=cell, type_id=0, quality=1)))
        row, col = divmod(cell, env.w)
        env.agents[0].budget = -1000.0
        for _ in range(3):
            env.end_round()
        self.assertFalse(env.agents[0].bankrupt)
        self.assertGreater(env.agents[0].debt, 0.0)
        self.assertEqual(int(env.type_grid[row, col]), 1)
        self.assertEqual(int(env.owner_grid[row, col]), 0)
        self.assertEqual(env.agents[0].budget, 0.0)

    def test_debt_interest_and_automatic_repayment(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=150,
        )
        agent = env.agents[0]
        agent.budget = 0.0
        agent.debt = 1000.0
        env.end_round()
        self.assertAlmostEqual(agent.interest_paid, 20.0)
        self.assertAlmostEqual(agent.debt, 1020.0)
        self.assertEqual(agent.budget, 0.0)
        self.assertAlmostEqual(agent.profit, -6020.0)
        self.assertAlmostEqual(env.agent_metrics()[0].debt_ratio, 1020.0 / 5000.0)

        agent.budget = 200.0
        agent.debt = 100.0
        env.end_round()
        self.assertAlmostEqual(agent.interest_paid, 22.0)
        self.assertAlmostEqual(agent.debt, 0.0)
        self.assertAlmostEqual(agent.budget, 98.0)


class TestActionModes(unittest.TestCase):
    def test_bid_caps_are_shared_and_custom_bids_are_clamped(self) -> None:
        self.assertEqual(bid_cap(100.0), BID_MAX * 100.0)
        self.assertEqual(cap_bid(100.0, 100.0), BID_MAX * 100.0)
        self.assertEqual(cap_bid(-1.0, 100.0), 0.0)
        self.assertEqual(strategic_bid(10_000.0, 1.0, 100.0), BID_MAX * 100.0)

        city = quick_city_config(
            n_agents=1, actions_per_agent=1, initial_budget=5000.0
        )

        class OverbiddingPolicy:
            name = "overbidding"

            def act(
                self,
                env: CityEnvironment,
                agent_id: int,
                rng: np.random.Generator,
            ) -> Action:
                del env, agent_id, rng
                return Action.pass_action()

            def propose_bids(
                self,
                env: CityEnvironment,
                agent_id: int,
                rng: np.random.Generator,
                count: int,
                exclude_cells: set[int] | None = None,
            ) -> list[Action]:
                del agent_id, rng, count, exclude_cells
                return [
                    Action(
                        cell=first_buildable(env),
                        type_id=0,
                        quality=1,
                        bid=env.agents[0].budget,
                    )
                ]

        result = run_episode(city, [OverbiddingPolicy()], seed=35)
        accepted = [action for action in result.actions if action.accepted]
        self.assertTrue(accepted)
        self.assertLessEqual(accepted[0].bid, BID_MAX * city.initial_budget)

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
        city = replace(
            quick_city_config(
                n_agents=3, actions_per_agent=2, initial_budget=5000,
                buildings_per_action=2,
            ),
            actions_per_round=1,
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
        self.assertIn("total_debt", history_row)
        self.assertIn("mean_debt_ratio", history_row)
        self.assertIn("housing_vacancy_rate", history_row)
        self.assertIn("job_vacancy_rate", history_row)
        self.assertIn("unemployment_rate", history_row)
        self.assertIn("net_migration_rate", history_row)
        self.assertIn("economic_phase", history_row)
        self.assertIn("tax_rate_multiplier", history_row)
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

    def test_fitness_breakdown_and_smooth_debt_penalty(self) -> None:
        city = quick_city_config(n_agents=4, actions_per_agent=3)
        result = run_episode(
            city, [RandomPolicy() for _ in range(4)], seed=33
        )
        breakdown = score_breakdown(result)
        self.assertEqual(breakdown["score"].shape, (4,))
        self.assertTrue(np.isfinite(breakdown["score"]).all())
        self.assertIn("city_quality", breakdown)
        self.assertIn("satisfaction", breakdown)

        def agent(debt_ratio: float, profit: float = 50.0):
            return SimpleNamespace(
                profit=profit,
                debt_ratio=debt_ratio,
                builds=2,
                skips=0,
                violations=0.0,
                damage_caused=0.0,
                bankrupt=False,
            )

        fake = SimpleNamespace(
            agents=[agent(0.0), agent(1.0)],
            city=SimpleNamespace(city_quality=0.5, mean_satisfaction=0.5),
            config=city,
        )
        scores = score_breakdown(fake)["score"]
        self.assertAlmostEqual(
            float(scores[0] - scores[1]), city.fitness.debt / 2.0, places=6
        )
        self.assertAlmostEqual(score_breakdown(fake)["debt_penalty"][1], 0.5)
        ranked = SimpleNamespace(
            city=fake.city,
            config=fake.config,
            agents=[
                agent(0.0, profit=10.0),
                agent(0.0, profit=10.0),
                agent(0.0, profit=20.0),
            ],
        )
        np.testing.assert_allclose(
            score_breakdown(ranked)["profit_norm"], [0.25, 0.25, 1.0]
        )
        single = SimpleNamespace(
            agents=[agent(0.0, profit=5.0)],
            city=fake.city,
            config=fake.config,
        )
        np.testing.assert_allclose(score_breakdown(single)["profit_norm"], [0.5])
        self.assertAlmostEqual(city.fitness.city_quality, 0.20)

    def test_industry_income_discount_and_district_action_prior(self) -> None:
        self.assertTrue(
            all(
                allowed == tuple(range(len(ACTION_TYPE_NAMES)))
                for allowed in DISTRICT_ALLOWED
            )
        )
        city = quick_city_config(
            n_agents=1,
            actions_per_agent=2,
            initial_budget=5000,
            enable_externalities=False,
        )
        env = CityEnvironment(city, seed=34)
        cell = first_buildable(env)
        row, col = divmod(cell, env.w)
        projected = env.projected_income_table(0)[cell, 3, 1]
        demand_factor = 0.5 + 0.7 * env.demand[row, col] / MAX_DEMAND
        access_fit = env._access_fit_map(4)[row, col]
        expected = (
            INCOME_BASE[4]
            * demand_factor
            * (0.7 + 0.6 * env.static["road"][row, col])
            * (0.6 + 0.8 * access_fit)
            * (0.9 + 0.2 * env.static["soil"][row, col])
            * QUALITY_INCOME_MULT[1]
            * INDUSTRY_INCOME_MULTIPLIER
        )
        self.assertAlmostEqual(float(projected), float(expected))
        self.assertTrue(
            env.apply_action(0, Action(cell=cell, type_id=3, quality=1))
        )
        actual = env._income_grid()[row, col]
        self.assertAlmostEqual(float(actual), float(expected))


class TestShocksAndRegulation(unittest.TestCase):
    def test_industry_buffer_is_not_a_build_lock(self) -> None:
        city = quick_city_config(
            n_agents=1,
            actions_per_agent=2,
            initial_budget=5000,
            requirements=REQUIREMENT_PROFILES["hard"],
        )
        env = CityEnvironment(city, seed=40)
        cells = np.flatnonzero(env.buildable.ravel())
        pair = next(
            (
                (int(first), int(second))
                for first in cells
                for second in cells
                if first != second
                and abs(first // env.w - second // env.w) <= 2
                and abs(first % env.w - second % env.w) <= 2
            ),
            None,
        )
        self.assertIsNotNone(pair)
        industry_cell, housing_cell = pair
        self.assertTrue(
            env.apply_action(
                0, Action(cell=industry_cell, type_id=3, quality=1)
            )
        )
        self.assertTrue(env.observe(0).action_mask[housing_cell, 0, 0])
        self.assertTrue(
            env.apply_action(
                0, Action(cell=housing_cell, type_id=0, quality=1)
            )
        )
        housing_row, housing_col = divmod(housing_cell, env.w)
        self.assertAlmostEqual(
            float(env._compatibility_fields()[0, housing_row, housing_col]),
            0.95,
        )

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
        green_cell = next(
            candidate
            for candidate in np.flatnonzero(env_tax.buildable.ravel())
            if candidate != cell
        )
        env_tax.apply_action(0, Action(cell=int(green_cell), type_id=4, quality=1))
        env_tax.end_round()
        self.assertEqual(env_tax.agents[0].tax_paid, 0.0)
        self.assertEqual(env_tax.agents[0].subsidy_received, 0.0)

        env_default = CityEnvironment(city, seed=46)
        cell = first_buildable(env_default)
        env_default.apply_action(0, Action(cell=cell, type_id=0, quality=1))
        commercial_cell = next(
            candidate
            for candidate in np.flatnonzero(env_default.buildable.ravel())
            if candidate != cell
        )
        env_default.apply_action(
            0, Action(cell=int(commercial_cell), type_id=1, quality=1)
        )
        env_default.end_round()
        self.assertGreater(env_default.agents[0].tax_paid, 0.0)

    def test_progressive_tax_rates_and_economic_phases(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=47,
        )
        cells = np.flatnonzero(env.buildable.ravel())[:5]
        types = (1, 2, 3, 4, 5)
        qualities = (0, 1, 2, 1, 0)
        for cell, world_type, quality in zip(cells, types, qualities):
            row, col = divmod(int(cell), env.w)
            env.type_grid[row, col] = world_type
            env.quality_grid[row, col] = quality
            env.floors_grid[row, col] = 1
        env._refresh_layers(advance_population=False)
        incomes = np.zeros((env.h, env.w), dtype=np.float64)
        incomes.ravel()[cells] = 100.0
        base_rates = (0.02, 0.04, 0.03, 0.05, 0.0)
        quality_multipliers = (0.8, 1.0, 1.2)

        env.population = 10.0
        env.unemployment_rate = 0.15
        recession_taxes, _ = env._fiscal_grids(incomes)
        self.assertEqual(env._economic_phase(), ("recession", 0.75))
        for cell, world_type, quality in zip(cells, types, qualities):
            row, col = divmod(int(cell), env.w)
            self.assertAlmostEqual(
                recession_taxes[row, col],
                100.0 * base_rates[world_type - 1]
                * quality_multipliers[quality] * 0.75,
            )

        env.unemployment_rate = 0.05
        self.assertEqual(env._economic_phase(), ("boom", 1.25))
        env.population = 0.0
        self.assertEqual(env._economic_phase(), ("normal", 1.0))

    def test_affordable_housing_and_occupied_jobs_subsidies(self) -> None:
        env = CityEnvironment(
            quick_city_config(n_agents=1, actions_per_agent=2, initial_budget=5000),
            seed=48,
        )
        cells = np.flatnonzero(env.buildable.ravel())[:3]
        for cell, world_type, quality, floors in (
            (cells[0], 1, 0, 2),
            (cells[1], 2, 1, 1),
            (cells[2], 5, 1, 1),
        ):
            row, col = divmod(int(cell), env.w)
            env.type_grid[row, col] = world_type
            env.quality_grid[row, col] = quality
            env.floors_grid[row, col] = floors
        env._refresh_layers(advance_population=False)
        env.jobs = 8.0
        env.employed = 4.0
        _, subsidies = env._fiscal_grids(
            np.zeros((env.h, env.w), dtype=np.float64)
        )
        self.assertEqual(subsidies.ravel()[cells[0]], 8.0)
        self.assertEqual(subsidies.ravel()[cells[1]], 2.0)
        self.assertEqual(subsidies.ravel()[cells[2]], 4.0)


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
        self.assertEqual(result.best_genome.shape, (DEFAULT_LAYOUT.parameter_count,))
        self.assertTrue(np.isfinite(result.best_fitness))
        self.assertEqual(len(result.history), 2)
        row = result.history[-1]
        for key in (
            "best", "mean", "diversity", "eval_satisfaction",
            "eval_city_quality", "eval_utilization", "eval_bankrupt_agents",
            "eval_housing_vacancy_rate", "eval_job_vacancy_rate",
            "eval_unemployment_rate", "eval_net_migration_rate",
            "eval_total_debt", "eval_mean_debt_ratio",
            "eval_local_goods_coverage",
            "eval_tax_rate_multiplier",
        ):
            self.assertIn(key, row)
        self.assertTrue(result.archive)
        for key, individual in result.archive.items():
            self.assertEqual(len(key), 2)
            for bucket in key:
                self.assertGreaterEqual(bucket, 0)
                self.assertLess(bucket, 3)
            self.assertEqual(
                individual.genome.shape, (DEFAULT_LAYOUT.parameter_count,)
            )
            self.assertTrue(np.isfinite(individual.fitness))


class TestCli(unittest.TestCase):
    def test_parser_and_configs(self) -> None:
        from citysim.cli import build_configs, build_parser, parse_grid, parse_seeds

        self.assertEqual(parse_grid("12x9"), (12, 9))
        self.assertEqual(parse_grid("12х9"), (12, 9))
        self.assertEqual(parse_grid("12×9"), (12, 9))
        self.assertEqual(parse_grid("64x64"), (64, 64))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_grid("12-by-9")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_grid("65x12")
        self.assertEqual(parse_seeds("0-2"), [0, 1, 2])
        parser = build_parser()
        args = parser.parse_args(
            [
                "experiment", "e12", "--quick", "--buildings-per-action", "2",
                "--requirements", "hard", "--regulation", "no_tax",
                "--no-auction", "--hidden", "32",
                "--grid", "10x11", "--actions-per-round", "3",
            ]
        )
        city, evo, seeds = build_configs(args)
        self.assertEqual(args.projection, "isometric")
        self.assertEqual(city.buildings_per_action, 2)
        self.assertEqual((city.height, city.width), (10, 11))
        self.assertEqual(city.actions_per_round, 3)
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
        road_cells = np.flatnonzero(env_a.buildable.ravel())[:2]
        env_a.player_road_grid.ravel()[road_cells[0]] = True
        env_a.pending_road_grid.ravel()[road_cells[1]] = 1
        env_a._refresh_layers(advance_population=False)
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
