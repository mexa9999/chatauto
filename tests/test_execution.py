from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from citysim.config import EvolutionConfig, quick_city_config, quick_evolution_config
from citysim.evolution import (
    EvolutionEngine,
    Individual,
    LARGE_MUTATION_PROBABILITY,
    LARGE_MUTATION_RATE,
    LARGE_MUTATION_SIGMA_MULTIPLIER,
    SIGMA_BOOST_FACTOR,
    SIGMA_MAX_INITIAL_MULTIPLE,
    SIGMA_STAGNATION_GENERATIONS,
    _TRAINING_PROTOCOL_VERSION,
    _configuration_digest,
    evaluate_population,
    layout_from_config,
)
from citysim.execution import EpisodeExecutor, EpisodeTask
from citysim.experiment_support import _run_evolution, train_evolution, training_cache
from citysim.policy import mutate_genome, random_genome
from citysim.training_history import TrainingHistoryStore


class TestEpisodeExecution(unittest.TestCase):
    def test_evolution_protocol_defaults_and_quick_preset(self) -> None:
        config = EvolutionConfig()
        self.assertEqual(config.generations, 125)
        self.assertEqual(config.episodes_per_generation, 16)
        self.assertEqual(config.tournament_size, 5)
        self.assertEqual(config.sharing_radius, 0.40)
        quick = quick_evolution_config()
        self.assertEqual(quick.generations, 20)
        self.assertEqual(quick.episodes_per_generation, 4)

    def test_sigma_controller_uses_uncertainty_and_caps_boost(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            elites=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        engine = EvolutionEngine(config, city)
        engine.sigma_reference_fitness = 1.0
        engine.sigma_reference_se = 0.1

        engine._update_sigma_stagnation(1.2, 0.1)
        self.assertEqual(engine.sigma_stagnation_count, 1)
        engine._update_sigma_stagnation(1.0, None)
        self.assertEqual(engine.sigma_stagnation_count, 1)

        with patch("citysim.evolution.SIGMA_STAGNATION_GENERATIONS", 2):
            engine._update_sigma_stagnation(1.0, 0.1)
            self.assertEqual(engine._adapt_sigma(), "boost")
        self.assertEqual(engine.sigma_stagnation_count, 0)
        self.assertAlmostEqual(
            engine.sigma,
            config.mutation_sigma * SIGMA_BOOST_FACTOR,
        )

        engine._update_sigma_stagnation(1.4, 0.1)
        self.assertEqual(engine.sigma_stagnation_count, 0)
        self.assertEqual(engine.sigma_reference_fitness, 1.4)

        engine.sigma = config.mutation_sigma * SIGMA_MAX_INITIAL_MULTIPLE
        engine.sigma_stagnation_count = SIGMA_STAGNATION_GENERATIONS
        self.assertEqual(engine._adapt_sigma(), "boost_capped")
        self.assertEqual(
            engine.sigma, config.mutation_sigma * SIGMA_MAX_INITIAL_MULTIPLE
        )

    def test_large_mutation_is_sparse_and_uses_engine_rng(self) -> None:
        self.assertEqual(LARGE_MUTATION_PROBABILITY, 0.02)
        self.assertEqual(LARGE_MUTATION_RATE, 0.01)
        self.assertEqual(LARGE_MUTATION_SIGMA_MULTIPLIER, 4.0)
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            elites=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=15,
        )
        engine = EvolutionEngine(config, city)
        genome = np.zeros(2000)
        with (
            patch("citysim.evolution.LARGE_MUTATION_PROBABILITY", 1.0),
            patch("citysim.evolution.mutate_genome", wraps=mutate_genome) as mutate,
        ):
            child, was_large = engine._mutate_offspring(genome)
        self.assertTrue(was_large)
        changed = np.flatnonzero(child)
        self.assertGreater(changed.size, 0)
        self.assertLess(changed.size, genome.size)
        mutate.assert_called_once_with(
            genome,
            engine.rng,
            engine.sigma * LARGE_MUTATION_SIGMA_MULTIPLIER,
            mutation_rate=LARGE_MUTATION_RATE,
        )

    def test_generation_checkpoint_resumes_identically(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=4,
            episodes_per_generation=4,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_map_elites=True,
            map_elites_bins=2,
            use_parallel=False,
            seed=12,
        )
        with (
            patch("citysim.evolution.SIGMA_STAGNATION_GENERATIONS", 1),
            patch("citysim.evolution.SIGMA_SIGNIFICANCE_Z", 1e9),
            patch("citysim.evolution.LARGE_MUTATION_PROBABILITY", 1.0),
        ):
            uninterrupted = EvolutionEngine(config, city).run()

            with tempfile.TemporaryDirectory() as output:
                checkpoint = Path(output) / "resume.npz"
                interrupted_engine = EvolutionEngine(config, city)
                evaluate_generation = interrupted_engine._evaluate_generation

                def stop_after_first_generation(generation: int) -> None:
                    if generation == 1:
                        raise RuntimeError("simulated interruption")
                    evaluate_generation(generation)

                with patch.object(
                    interrupted_engine,
                    "_evaluate_generation",
                    side_effect=stop_after_first_generation,
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        interrupted_engine.run(checkpoint_path=checkpoint)

                resumed = EvolutionEngine(config, city).run(
                    checkpoint_path=checkpoint,
                    resume=True,
                )

        self.assertEqual(uninterrupted.history, resumed.history)
        self.assertEqual(uninterrupted.history[1]["sigma_adjustment"], "boost")
        self.assertEqual(uninterrupted.history[1]["large_mutation_events"], 3)
        self.assertEqual(uninterrupted.best_fitness, resumed.best_fitness)
        self.assertEqual(uninterrupted.best_generation, resumed.best_generation)
        np.testing.assert_array_equal(uninterrupted.best_genome, resumed.best_genome)
        self.assertEqual(uninterrupted.archive.keys(), resumed.archive.keys())
        for expected, actual in zip(uninterrupted.population, resumed.population):
            np.testing.assert_array_equal(expected.genome, actual.genome)
            self.assertEqual(expected.fitness, actual.fitness)
            np.testing.assert_array_equal(expected.behavior, actual.behavior)

    def test_legacy_checkpoint_schema_is_rejected(self) -> None:
        from citysim.checkpoint_io import read_checkpoint, write_checkpoint

        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_generation=2,
            elites=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output) / "legacy.npz"
            EvolutionEngine(config, city).run(checkpoint_path=checkpoint)
            metadata, arrays = read_checkpoint(checkpoint)
            metadata["schema"] = 2
            write_checkpoint(checkpoint, metadata, arrays)
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint schema"):
                EvolutionEngine(config, city).run(
                    checkpoint_path=checkpoint,
                    resume=True,
                )

    def test_checkpoint_from_previous_training_protocol_is_rejected(self) -> None:
        from citysim.checkpoint_io import read_checkpoint, write_checkpoint

        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_generation=2,
            elites=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output) / "previous-protocol.npz"
            EvolutionEngine(config, city).run(checkpoint_path=checkpoint)
            metadata, arrays = read_checkpoint(checkpoint)
            self.assertEqual(
                metadata["training_protocol_version"],
                _TRAINING_PROTOCOL_VERSION,
            )
            metadata.pop("training_protocol_version")
            write_checkpoint(checkpoint, metadata, arrays)
            with self.assertRaisesRegex(ValueError, "checkpoint training protocol"):
                EvolutionEngine(config, city).run(
                    checkpoint_path=checkpoint,
                    resume=True,
                    allow_config_changes=True,
                )

    def test_checkpoint_interval_still_saves_final_generation(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=5,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
        )
        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output) / "interval.npz"
            engine = EvolutionEngine(config, city)
            write_checkpoint = engine._write_checkpoint
            saved_generations = []

            def record_checkpoint(path, next_generation, complete):
                saved_generations.append((next_generation, complete))
                write_checkpoint(path, next_generation, complete)

            with patch.object(engine, "_write_checkpoint", side_effect=record_checkpoint):
                engine.run(checkpoint_path=checkpoint, checkpoint_every=2)

        self.assertEqual(
            saved_generations,
            [(2, False), (4, False), (5, True)],
        )

    def test_requested_stop_saves_and_resumes_on_generation_boundary(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=3,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=29,
        )
        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output) / "stop.npz"
            progress = []
            stopped = EvolutionEngine(config, city).run(
                checkpoint_path=checkpoint,
                checkpoint_every=10,
                progress_callback=lambda generation, row, total: progress.append(
                    (generation, total)
                ),
                stop_requested=lambda: True,
            )
            from citysim.checkpoint_io import read_checkpoint_metadata

            metadata = read_checkpoint_metadata(checkpoint)
            resumed = EvolutionEngine(config, city).run(
                checkpoint_path=checkpoint,
                resume=True,
            )

        self.assertEqual(progress, [(0, 3)])
        self.assertEqual(len(stopped.history), 1)
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["next_generation"], 1)
        self.assertEqual(len(resumed.history), 3)

    def test_completed_training_is_reused_only_for_same_configuration(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=17,
        )
        factory_calls = 0

        def counting_factory(evolution_config, city_config):
            nonlocal factory_calls
            factory_calls += 1
            return EvolutionEngine(evolution_config, city_config)

        with tempfile.TemporaryDirectory() as output:
            with training_cache(output):
                first = _run_evolution(config, city, engine_factory=counting_factory)
                second = _run_evolution(
                    config,
                    city,
                    engine_factory=lambda *_: self.fail("cache should be reused"),
                )
                changed = _run_evolution(
                    replace(config, seed=config.seed + 1),
                    city,
                    engine_factory=counting_factory,
                )

        self.assertEqual(factory_calls, 2)
        np.testing.assert_array_equal(first.best_genome, second.best_genome)
        self.assertFalse(np.array_equal(first.best_genome, changed.best_genome))
        self.assertEqual(first.history, second.history)

    def test_training_cache_requires_explicit_resume_for_partial_checkpoint(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=3,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=23,
        )

        def stop_after_first_generation(evolution_config, city_config):
            engine = EvolutionEngine(evolution_config, city_config)
            evaluate_generation = engine._evaluate_generation

            def stop_before_second_generation(generation: int) -> None:
                if generation == 1:
                    raise RuntimeError("simulated interruption")
                evaluate_generation(generation)

            engine._evaluate_generation = stop_before_second_generation
            return engine

        with tempfile.TemporaryDirectory() as output:
            with training_cache(output, checkpoint_every=1):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    _run_evolution(
                        config,
                        city,
                        engine_factory=stop_after_first_generation,
                    )

            with training_cache(output):
                with self.assertRaisesRegex(FileExistsError, "rerun with --resume"):
                    _run_evolution(config, city)

            with training_cache(output, resume=True, checkpoint_every=1):
                resumed = _run_evolution(config, city)

            self.assertEqual(len(resumed.history), config.generations)
            store = TrainingHistoryStore(output)
            report = store.verify()
            self.assertTrue(report["ok"])
            self.assertEqual(report["experiment_checkpoint_count"], 1)
            digest = _configuration_digest(config, city)
            restored = Path(output) / "restored-checkpoint.npz"
            entry = store.restore_experiment_checkpoint(digest, restored)
            self.assertIsNotNone(entry)
            self.assertTrue(entry["complete"])
            checkpoint = store.objects / f"{entry['checkpoint_hash']}.npz"
            self.assertTrue(checkpoint.is_file())
            self.assertNotIn(checkpoint.name, store.gc_preview()["objects"])

    def test_training_history_branches_and_forks_from_generation(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=3,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=31,
        )
        with tempfile.TemporaryDirectory() as output:
            store = TrainingHistoryStore(output)
            train_evolution(
                config,
                city,
                history_store=store,
                checkpoint_every=1,
            )
            with self.assertRaisesRegex(RuntimeError, "advanced during training"):
                store.record_checkpoint(
                    "main",
                    store.checkpoint_path("main"),
                    3,
                    True,
                    config,
                    city,
                    expected_parent=None,
                )
            store.create_branch("city-test", "main@2")
            branch_city = replace(city, requirements=replace(city.requirements, green_min=0.1))
            branch_config = replace(
                config,
                generations=4,
                mutation_sigma=0.08,
                seed=config.seed + 1,
            )
            forked = train_evolution(
                branch_config,
                branch_city,
                history_store=store,
                branch="city-test",
                resume=True,
                checkpoint_every=1,
            )

            main_history = store.history("main")
            branch_history = store.history("city-test")
            branch_config_saved, evolution_config_saved = store.config_for_branch(
                "city-test"
            )
            store.reset_branch("city-test", "main@1")
            reset_history = store.history("city-test")

        self.assertEqual([row["generation"] for row in main_history], [1, 2, 3])
        self.assertEqual(
            [row["generation"] for row in branch_history],
            [1, 2, 3, 4],
        )
        self.assertEqual(branch_history[2]["branch_name"], "city-test")
        self.assertEqual(branch_city, branch_config_saved)
        self.assertEqual(branch_config, evolution_config_saved)
        self.assertEqual(forked.evolution_config.generations, 4)
        self.assertEqual([row["gen"] for row in forked.history], [2, 3])
        self.assertGreaterEqual(forked.best_generation, 2)
        self.assertEqual(reset_history[-1]["generation"], 1)

    def test_new_history_run_replaces_orphaned_work_checkpoint(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=32,
        )
        with tempfile.TemporaryDirectory() as output:
            store = TrainingHistoryStore(output)
            orphaned_checkpoint = store.work_path("main")
            orphaned_checkpoint.write_bytes(b"interrupted before history snapshot")

            result = train_evolution(
                config,
                city,
                history_store=store,
                checkpoint_every=1,
            )

            self.assertEqual(len(result.history), 1)
            self.assertTrue(orphaned_checkpoint.is_file())
            self.assertNotEqual(
                orphaned_checkpoint.read_bytes(),
                b"interrupted before history snapshot",
            )
            self.assertEqual(store.history("main")[-1]["generation"], 1)

    def test_training_history_rejects_architecture_change_on_branch(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=47,
        )
        with tempfile.TemporaryDirectory() as output:
            store = TrainingHistoryStore(output)
            train_evolution(config, city, history_store=store, checkpoint_every=1)
            store.create_branch("wrong-network", "main@1")

            with self.assertRaisesRegex(ValueError, "different neural architecture"):
                train_evolution(
                    replace(config, generations=3, hidden_size=9),
                    city,
                    history_store=store,
                    branch="wrong-network",
                    resume=True,
                    checkpoint_every=1,
                )

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
