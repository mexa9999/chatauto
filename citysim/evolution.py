"""Population-level neuroevolution: islands, tournament, sharing, migration."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import BEHAVIOR_DIM, CityConfig, EvolutionConfig
from .episode import EpisodeResult, run_episode, score_episode
from .execution import (
    CitySummary,
    EpisodeExecutor,
    EpisodeTask,
    _run_episode_worker,
    _worker_init,
)
from .metrics import spatial_metrics
from .progress import ProgressCallback, StopRequested
from .policy import (
    GenomeLayout,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    Policy,
    RandomPolicy,
    mutate_genome,
    random_genome,
)


_CHECKPOINT_SCHEMA = 1
# Bump when training semantics change so old cache entries are not reused.
_TRAINING_PROTOCOL_VERSION = 1


def _configuration_digest(config: EvolutionConfig, city_config: CityConfig) -> str:
    payload = json.dumps(
        {
            "protocol": _TRAINING_PROTOCOL_VERSION,
            "evolution": asdict(config),
            "city": asdict(city_config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _max_workers(config: EvolutionConfig, task_count: int) -> int:
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)()
    available = max(1, cpu_count or 1)
    requested = config.max_workers or available
    return max(1, min(int(requested), int(task_count)))


def layout_from_config(config: EvolutionConfig) -> GenomeLayout:
    return GenomeLayout(
        hidden_size=config.hidden_size,
        attention_dim=config.attention_dim,
        district_hidden=config.district_hidden,
        global_hidden=config.global_hidden,
    )


@dataclass
class Individual:
    genome: np.ndarray
    island: int = 0
    fitness: float = -1e18
    shaped_fitness: float = -1e18
    behavior: np.ndarray = field(
        default_factory=lambda: np.zeros(BEHAVIOR_DIM, dtype=np.float64)
    )
    episodes: int = 0


@dataclass
class EvolutionResult:
    evolution_config: EvolutionConfig
    city_config: CityConfig
    history: list[dict]
    population: list[Individual]
    best_genome: np.ndarray
    best_fitness: float
    best_behavior: np.ndarray
    best_generation: int
    hall_of_fame: list[tuple[int, float, np.ndarray]]
    archive: dict[tuple[int, int], Individual] = field(default_factory=dict)


def balanced_episode_indices(
    rng: np.random.Generator, population_size: int, n_agents: int, episodes: int
) -> list[np.ndarray]:
    """Assign genomes to episodes with balanced coverage.

    Each episode contains distinct genomes.  Every genome is guaranteed to be
    evaluated at least once when ``episodes * n_agents >= population_size``.
    """
    if n_agents > population_size:
        raise ValueError("n_agents cannot exceed population_size")
    counts = np.zeros(population_size, dtype=np.int64)
    out: list[np.ndarray] = []
    for _ in range(episodes):
        keys = counts + rng.random(population_size) * 0.49
        chosen = np.argsort(keys, kind="stable")[:n_agents]
        out.append(chosen.astype(np.int64))
        counts[chosen] += 1
    return out


def mean_pairwise_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(points.shape[0], k=1)
    return float(np.mean(dist[iu]))


def unique_behaviors(points: np.ndarray, decimals: int = 2) -> int:
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals)
    return int(np.unique(rounded, axis=0).shape[0])


class EvolutionEngine:
    def __init__(
        self,
        config: EvolutionConfig,
        city_config: CityConfig,
        executor: ProcessPoolExecutor | None = None,
    ):
        if config.episodes_per_generation * city_config.n_agents < config.population_size:
            raise ValueError(
                "episodes_per_generation * n_agents must be >= population_size "
                "so that every genome is evaluated"
            )
        self.config = config
        self.city_config = city_config
        self.layout = layout_from_config(config)
        self.rng = np.random.default_rng(config.seed)
        self.sigma = float(config.mutation_sigma)
        self.generation = 0

        self.population = [
            Individual(
                genome=random_genome(self.rng, self.layout),
                island=i % config.islands if config.use_islands else 0,
            )
            for i in range(config.population_size)
        ]
        self.history: list[dict] = []
        self.archive: dict[tuple[int, int], Individual] = {}
        self._last_city_metrics: dict[str, float] = {}
        self.best_ever = -1e18
        self.best_ever_genome: np.ndarray | None = None
        self.best_ever_behavior: np.ndarray = np.zeros(BEHAVIOR_DIM)
        self.best_ever_generation = -1
        self.hall_of_fame: list[tuple[int, float, np.ndarray]] = []

        self.episode_executor = EpisodeExecutor(
            city_config,
            self.layout,
            parallel=config.use_parallel,
            max_workers=_max_workers(config, config.episodes_per_generation),
            executor=executor,
            process_pool_factory=ProcessPoolExecutor,
        )
        self.executor = self.episode_executor.executor

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
        checkpoint_callback: Callable[[Path, int, bool], None] | None = None,
        allow_config_changes: bool = False,
        progress_callback: ProgressCallback | None = None,
        stop_requested: StopRequested | None = None,
    ) -> EvolutionResult:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        try:
            if checkpoint_every < 1:
                raise ValueError("checkpoint_every must be at least one")
            if resume and checkpoint is None:
                raise ValueError("resume requires a checkpoint path")
            if stop_requested is not None and checkpoint is None:
                raise ValueError("safe stop requires a checkpoint path")
            start_generation = 0
            if checkpoint is not None and checkpoint.exists():
                if not resume:
                    raise FileExistsError(
                        f"checkpoint already exists at {checkpoint}; pass resume=True"
                    )
                metadata = self._restore_checkpoint(
                    checkpoint,
                    allow_config_changes=allow_config_changes,
                )
                if metadata["complete"] and (
                    not allow_config_changes
                    or self.config.generations
                    <= int(metadata["next_generation"])
                ):
                    return self._result()
                start_generation = int(metadata["next_generation"])
                if start_generation >= self.config.generations:
                    raise ValueError(
                        "checkpoint generation is not complete but is beyond the "
                        "configured generation count"
                    )

            for gen in range(start_generation, self.config.generations):
                self.generation = gen
                self._evaluate_generation(gen)
                row = self._log_generation(gen)
                self.history.append(row)
                if progress_callback is not None:
                    progress_callback(gen, row, self.config.generations)
                if self.config.verbose and (
                    gen % max(1, self.config.log_every) == 0
                    or gen == self.config.generations - 1
                ):
                    print(
                        f"[gen {gen:4d}] best={row['best']:.4f} "
                        f"mean={row['mean']:.4f} div={row['diversity']:.4f} "
                        f"sigma={row['sigma']:.4f} best_ever={row['best_ever']:.4f}"
                    )
                stopping = bool(stop_requested and stop_requested())
                if gen == self.config.generations - 1:
                    if checkpoint is not None:
                        self._write_checkpoint(
                            checkpoint,
                            next_generation=gen + 1,
                            complete=True,
                        )
                        if checkpoint_callback is not None:
                            checkpoint_callback(checkpoint, gen + 1, True)
                    break
                self._select()
                if (
                    self.config.use_islands
                    and self.config.migration_interval > 0
                    and (gen + 1) % self.config.migration_interval == 0
                ):
                    self._migrate()
                if (
                    self.config.fresh_blood_interval > 0
                    and (gen + 1) % self.config.fresh_blood_interval == 0
                ):
                    self._fresh_blood()
                self.sigma = max(
                    self.config.min_sigma, self.sigma * self.config.sigma_decay
                )
                if checkpoint is not None and (
                    (gen + 1) % checkpoint_every == 0 or stopping
                ):
                    self._write_checkpoint(
                        checkpoint,
                        next_generation=gen + 1,
                        complete=False,
                    )
                    if checkpoint_callback is not None:
                        checkpoint_callback(checkpoint, gen + 1, False)
                if stopping:
                    break

            if self.best_ever_genome is None:
                best = max(self.population, key=lambda ind: ind.fitness)
                self.best_ever = best.fitness
                self.best_ever_genome = best.genome.copy()
                self.best_ever_behavior = best.behavior.copy()
                self.best_ever_generation = self.generation
            return self._result()
        finally:
            self.episode_executor.shutdown()
            if self.episode_executor.owns_executor:
                self.executor = None

    def _result(self) -> EvolutionResult:
        if self.best_ever_genome is None:
            raise RuntimeError("evolution has no completed generation")
        return EvolutionResult(
            evolution_config=self.config,
            city_config=self.city_config,
            history=self.history,
            population=self.population,
            best_genome=self.best_ever_genome.copy(),
            best_fitness=float(self.best_ever),
            best_behavior=self.best_ever_behavior.copy(),
            best_generation=int(self.best_ever_generation),
            hall_of_fame=list(self.hall_of_fame),
            archive=dict(self.archive),
        )

    def _write_checkpoint(
        self,
        path: Path,
        next_generation: int,
        complete: bool,
    ) -> None:
        from .checkpoint_io import write_checkpoint

        archive_items = sorted(self.archive.items())
        archive_genomes = (
            np.stack([item.genome for _, item in archive_items])
            if archive_items
            else np.empty((0, self.layout.parameter_count), dtype=np.float64)
        )
        archive_behaviors = (
            np.stack([item.behavior for _, item in archive_items])
            if archive_items
            else np.empty((0, BEHAVIOR_DIM), dtype=np.float64)
        )
        metadata = {
            "schema": _CHECKPOINT_SCHEMA,
            "evolution_config": asdict(self.config),
            "city_config": asdict(self.city_config),
            "configuration_digest": _configuration_digest(
                self.config, self.city_config
            ),
            "next_generation": next_generation,
            "complete": complete,
            "sigma": self.sigma,
            "rng_state": self.rng.bit_generator.state,
            "best_ever": self.best_ever,
            "best_ever_generation": self.best_ever_generation,
            "history": self.history,
            "last_city_metrics": self._last_city_metrics,
            "hall_of_fame_generations": [
                generation for generation, _, _ in self.hall_of_fame
            ],
            "hall_of_fame_fitness": [
                fitness for _, fitness, _ in self.hall_of_fame
            ],
            "archive_keys": [list(key) for key, _ in archive_items],
        }
        arrays = {
            "population_genomes": np.stack(
                [individual.genome for individual in self.population]
            ),
            "population_behaviors": np.stack(
                [individual.behavior for individual in self.population]
            ),
            "population_islands": np.asarray(
                [individual.island for individual in self.population], dtype=np.int64
            ),
            "population_fitness": np.asarray(
                [individual.fitness for individual in self.population],
                dtype=np.float64,
            ),
            "population_shaped_fitness": np.asarray(
                [individual.shaped_fitness for individual in self.population],
                dtype=np.float64,
            ),
            "population_episodes": np.asarray(
                [individual.episodes for individual in self.population],
                dtype=np.int64,
            ),
            "best_ever_genome": self.best_ever_genome,
            "best_ever_behavior": self.best_ever_behavior,
            "hall_of_fame_genomes": (
                np.stack([genome for _, _, genome in self.hall_of_fame])
                if self.hall_of_fame
                else np.empty((0, self.layout.parameter_count), dtype=np.float64)
            ),
            "archive_genomes": archive_genomes,
            "archive_behaviors": archive_behaviors,
            "archive_islands": np.asarray(
                [item.island for _, item in archive_items], dtype=np.int64
            ),
            "archive_fitness": np.asarray(
                [item.fitness for _, item in archive_items], dtype=np.float64
            ),
            "archive_shaped_fitness": np.asarray(
                [item.shaped_fitness for _, item in archive_items], dtype=np.float64
            ),
        }
        write_checkpoint(path, metadata, arrays)

    def _restore_checkpoint(
        self,
        path: Path,
        allow_config_changes: bool = False,
    ) -> dict:
        from .checkpoint_io import read_checkpoint

        metadata, arrays = read_checkpoint(path)
        if metadata.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema in {path}")
        city_changed = (
            allow_config_changes
            and metadata.get("city_config") != asdict(self.city_config)
        )
        expected_digest = _configuration_digest(self.config, self.city_config)
        config_matches = metadata.get("configuration_digest") == expected_digest
        if not config_matches and not allow_config_changes:
            raise ValueError(
                "checkpoint configuration does not match the requested run"
            )
        if not config_matches:
            self._validate_branch_configuration(metadata, path)
        if arrays["population_genomes"].shape != (
            self.config.population_size,
            self.layout.parameter_count,
        ):
            raise ValueError("checkpoint population shape does not match configuration")

        self.population = [
            Individual(
                genome=arrays["population_genomes"][index],
                island=int(arrays["population_islands"][index]),
                fitness=float(arrays["population_fitness"][index]),
                shaped_fitness=float(arrays["population_shaped_fitness"][index]),
                behavior=arrays["population_behaviors"][index],
                episodes=int(arrays["population_episodes"][index]),
            )
            for index in range(self.config.population_size)
        ]
        self.generation = int(metadata["next_generation"])
        self.sigma = float(metadata["sigma"])
        previous_config = metadata.get("evolution_config", {})
        if (
            allow_config_changes
            and int(previous_config.get("seed", self.config.seed)) != self.config.seed
        ):
            self.rng = np.random.default_rng(self.config.seed)
        else:
            self.rng.bit_generator.state = metadata["rng_state"]
        self.history = metadata["history"]
        self._last_city_metrics = metadata["last_city_metrics"]
        self.best_ever = float(metadata["best_ever"])
        self.best_ever_generation = int(metadata["best_ever_generation"])
        self.best_ever_genome = arrays["best_ever_genome"]
        self.best_ever_behavior = arrays["best_ever_behavior"]

        self.hall_of_fame = [
            (
                int(generation),
                float(fitness),
                genome,
            )
            for generation, fitness, genome in zip(
                metadata["hall_of_fame_generations"],
                metadata["hall_of_fame_fitness"],
                arrays["hall_of_fame_genomes"],
            )
        ]
        archive_keys = [tuple(int(value) for value in key) for key in metadata["archive_keys"]]
        self.archive = {
            key: Individual(
                genome=arrays["archive_genomes"][index],
                island=int(arrays["archive_islands"][index]),
                fitness=float(arrays["archive_fitness"][index]),
                shaped_fitness=float(arrays["archive_shaped_fitness"][index]),
                behavior=arrays["archive_behaviors"][index],
            )
            for index, key in enumerate(archive_keys)
        }
        if city_changed:
            self.history = []
            self._last_city_metrics = {}
            self.best_ever = -1e18
            self.best_ever_generation = -1
            self.best_ever_genome = np.zeros(self.layout.parameter_count)
            self.best_ever_behavior = np.zeros(BEHAVIOR_DIM)
            self.hall_of_fame = []
            self.archive = {}
            for individual in self.population:
                individual.fitness = -1e18
                individual.shaped_fitness = -1e18
                individual.behavior = np.zeros(BEHAVIOR_DIM)
                individual.episodes = 0
        return metadata

    def _validate_branch_configuration(self, metadata: dict, path: Path) -> None:
        previous = metadata.get("evolution_config")
        if not isinstance(previous, dict):
            raise ValueError(
                f"checkpoint at {path} lacks the configuration needed to validate a branch"
            )
        architecture = (
            "hidden_size",
            "attention_dim",
            "district_hidden",
            "global_hidden",
        )
        if any(previous.get(key) != getattr(self.config, key) for key in architecture):
            raise ValueError(
                "cannot branch with a different neural architecture; "
                "start a new training run instead"
            )
        if previous.get("population_size") != self.config.population_size:
            raise ValueError(
                "cannot branch with a different population size; start a new training run"
            )
        if any(
            previous.get(key) != getattr(self.config, key)
            for key in ("islands", "use_islands", "use_map_elites", "map_elites_bins")
        ):
            raise ValueError(
                "branching cannot change island or MAP-Elites population structure"
            )

        mutable = {
            "generations",
            "mutation_sigma",
            "sigma_decay",
            "min_sigma",
            "tournament_size",
            "elites",
            "migration_interval",
            "fresh_blood_interval",
            "fresh_blood_count",
            "use_fitness_sharing",
            "sharing_lambda",
            "sharing_radius",
            "seed",
            "train_seed_base",
            "validation_seed_base",
            "test_seed_base",
            "verbose",
            "log_every",
            "max_workers",
            "use_parallel",
        }
        current = asdict(self.config)
        changed = [
            key
            for key, value in current.items()
            if key not in mutable and previous.get(key) != value
        ]
        if changed:
            raise ValueError(
                "branch changes unsupported evolution settings: "
                + ", ".join(sorted(changed))
            )

    @classmethod
    def load_completed(
        cls,
        checkpoint_path: str | Path,
        config: EvolutionConfig,
        city_config: CityConfig,
    ) -> EvolutionResult:
        engine = cls.__new__(cls)
        engine.config = config
        engine.city_config = city_config
        engine.layout = layout_from_config(config)
        engine.rng = np.random.default_rng(config.seed)
        metadata = engine._restore_checkpoint(Path(checkpoint_path))
        if not metadata["complete"]:
            raise ValueError(f"checkpoint at {checkpoint_path} is not complete")
        return engine._result()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_generation(self, gen: int) -> None:
        cfg = self.config
        episodes = balanced_episode_indices(
            self.rng,
            cfg.population_size,
            self.city_config.n_agents,
            cfg.episodes_per_generation,
        )
        fitness_sum = np.zeros(cfg.population_size, dtype=np.float64)
        behavior_sum = np.zeros((cfg.population_size, BEHAVIOR_DIM), dtype=np.float64)
        appearances = np.zeros(cfg.population_size, dtype=np.int64)
        city_keys = (
            "city_quality",
            "mean_satisfaction",
            "rich_satisfaction",
            "poor_satisfaction",
            "gini_satisfaction",
            "mean_utilization",
            "population",
            "green_share",
            "externality_damage_total",
            "bankrupt_agents",
        )
        city_sum = {key: 0.0 for key in city_keys}
        city_episodes = 0

        tasks = []
        for episode_idx, devs in enumerate(episodes):
            seed = cfg.train_seed_base + gen * cfg.episodes_per_generation + episode_idx
            genomes = [self.population[int(idx)].genome for idx in devs]
            tasks.append(EpisodeTask(genomes, seed))
        results = self.episode_executor.map(tasks)

        for result, devs in zip(results, episodes):
            city_episodes += 1
            for key, value in zip(city_keys, result.city):
                city_sum[key] += value
            for slot, idx in enumerate(devs):
                idx = int(idx)
                fitness_sum[idx] += float(result.scores[slot])
                behavior_sum[idx] += np.asarray(
                    result.behaviors[slot], dtype=np.float64
                )
                appearances[idx] += 1

        if city_episodes:
            self._last_city_metrics = {
                key: city_sum[key] / city_episodes for key in city_keys
            }
        else:
            self._last_city_metrics = {key: 0.0 for key in city_keys}

        for i, ind in enumerate(self.population):
            ind.episodes = int(appearances[i])
            if appearances[i] > 0:
                ind.fitness = float(fitness_sum[i] / appearances[i])
                ind.behavior = behavior_sum[i] / appearances[i]
            else:
                ind.fitness = -1e18
                ind.behavior = np.zeros(BEHAVIOR_DIM)

        self._apply_fitness_sharing()
        self._update_archive()

        best = max(self.population, key=lambda ind: ind.fitness)
        if best.fitness > self.best_ever:
            self.best_ever = float(best.fitness)
            self.best_ever_genome = best.genome.copy()
            self.best_ever_behavior = best.behavior.copy()
            self.best_ever_generation = gen
        self.hall_of_fame.append((gen, float(best.fitness), best.genome.copy()))
        self.hall_of_fame.sort(key=lambda item: item[1], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:10]

    def _apply_fitness_sharing(self) -> None:
        cfg = self.config
        if not cfg.use_fitness_sharing:
            for ind in self.population:
                ind.shaped_fitness = ind.fitness
            return
        if cfg.use_islands:
            groups = [
                [i for i, ind in enumerate(self.population) if ind.island == island]
                for island in range(cfg.islands)
            ]
        else:
            groups = [list(range(cfg.population_size))]
        for group in groups:
            points = np.array([self.population[i].behavior for i in group], dtype=np.float64)
            diff = points[:, None, :] - points[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            similarity = np.exp(-d2 / (2.0 * cfg.sharing_radius**2))
            np.fill_diagonal(similarity, 0.0)
            penalties = cfg.sharing_lambda * similarity.sum(axis=1)
            for local, idx in enumerate(group):
                self.population[idx].shaped_fitness = (
                    self.population[idx].fitness - float(penalties[local])
                )

    def _update_archive(self) -> None:
        if not self.config.use_map_elites:
            return
        bins = self.config.map_elites_bins
        for ind in self.population:
            d_bucket = int(np.clip(ind.behavior[0] * bins, 0, bins - 1))
            g_bucket = int(np.clip(ind.behavior[3] * bins, 0, bins - 1))
            key = (d_bucket, g_bucket)
            current = self.archive.get(key)
            if current is None or ind.fitness > current.fitness:
                self.archive[key] = Individual(
                    genome=ind.genome.copy(),
                    island=ind.island,
                    fitness=ind.fitness,
                    shaped_fitness=ind.shaped_fitness,
                    behavior=ind.behavior.copy(),
                )

    # ------------------------------------------------------------------
    # Selection and diversity operators
    # ------------------------------------------------------------------

    def _tournament(self, pool: list[Individual]) -> Individual:
        if not pool:
            raise ValueError("empty tournament pool")
        size = min(self.config.tournament_size, len(pool))
        indices = self.rng.integers(0, len(pool), size=size)
        best = max((pool[int(i)] for i in indices), key=lambda ind: ind.shaped_fitness)
        return best

    def _select(self) -> None:
        cfg = self.config
        next_population: list[Individual] = []
        if cfg.use_islands:
            base_elites = cfg.elites // cfg.islands
            extra = cfg.elites % cfg.islands
            for island in range(cfg.islands):
                members = [ind for ind in self.population if ind.island == island]
                members.sort(key=lambda ind: ind.fitness, reverse=True)
                elite_count = base_elites + (1 if island < extra else 0)
                elite_count = min(elite_count, len(members))
                elites = [
                    Individual(
                        genome=ind.genome.copy(),
                        island=island,
                        fitness=ind.fitness,
                        shaped_fitness=ind.shaped_fitness,
                        behavior=ind.behavior.copy(),
                    )
                    for ind in members[:elite_count]
                ]
                children_needed = len(members) - elite_count
                children = []
                for _ in range(children_needed):
                    parent = self._tournament(members)
                    child = Individual(
                        genome=mutate_genome(parent.genome, self.rng, self.sigma),
                        island=island,
                    )
                    children.append(child)
                next_population.extend(elites + children)
        else:
            members = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            elite_count = min(cfg.elites, len(members))
            elites = [
                Individual(
                    genome=ind.genome.copy(),
                    island=0,
                    fitness=ind.fitness,
                    shaped_fitness=ind.shaped_fitness,
                    behavior=ind.behavior.copy(),
                )
                for ind in members[:elite_count]
            ]
            children = []
            for _ in range(cfg.population_size - elite_count):
                parent = self._tournament(members)
                children.append(
                    Individual(
                        genome=mutate_genome(parent.genome, self.rng, self.sigma),
                        island=0,
                    )
                )
            next_population = elites + children
        self.population = next_population

    def _migrate(self) -> None:
        cfg = self.config
        best_by_island: list[Individual] = []
        for island in range(cfg.islands):
            members = [ind for ind in self.population if ind.island == island]
            if not members:
                continue
            best_by_island.append(max(members, key=lambda ind: ind.fitness))
        if len(best_by_island) < 2:
            return
        for src in range(cfg.islands):
            dst = (src + 1) % cfg.islands
            destination = [ind for ind in self.population if ind.island == dst]
            if not destination:
                continue
            worst = min(destination, key=lambda ind: ind.fitness)
            worst.genome = best_by_island[src % len(best_by_island)].genome.copy()
            worst.fitness = -1e18
            worst.shaped_fitness = -1e18
            worst.behavior = np.zeros(BEHAVIOR_DIM)
            worst.episodes = 0

    def _fresh_blood(self) -> None:
        cfg = self.config
        if cfg.fresh_blood_count <= 0:
            return
        for island in range(cfg.islands if cfg.use_islands else 1):
            members = [ind for ind in self.population if ind.island == island]
            if len(members) <= 1:
                continue
            replace_n = min(cfg.fresh_blood_count, len(members) - 1)
            members.sort(key=lambda ind: ind.fitness)
            for ind in members[:replace_n]:
                ind.genome = random_genome(self.rng, self.layout)
                ind.fitness = -1e18
                ind.shaped_fitness = -1e18
                ind.behavior = np.zeros(BEHAVIOR_DIM)
                ind.episodes = 0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_generation(self, gen: int) -> dict:
        fitness = np.array([ind.fitness for ind in self.population], dtype=np.float64)
        shaped = np.array([ind.shaped_fitness for ind in self.population], dtype=np.float64)
        behaviors = np.array([ind.behavior for ind in self.population], dtype=np.float64)
        row = {
            "gen": gen,
            "best": float(fitness.max()),
            "mean": float(fitness.mean()),
            "std": float(fitness.std()),
            "best_shaped": float(shaped.max()),
            "mean_shaped": float(shaped.mean()),
            "diversity": mean_pairwise_distance(behaviors),
            "unique_behaviors": unique_behaviors(behaviors),
            "sigma": float(self.sigma),
            "best_ever": float(self.best_ever),
            "best_green_share": float(behaviors[:, 3].max()),
            "archive_coverage": float(len(self.archive)),
        }
        city = self._last_city_metrics
        row.update(
            {
                "eval_city_quality": float(city.get("city_quality", 0.0)),
                "eval_satisfaction": float(city.get("mean_satisfaction", 0.0)),
                "eval_rich_satisfaction": float(city.get("rich_satisfaction", 0.0)),
                "eval_poor_satisfaction": float(city.get("poor_satisfaction", 0.0)),
                "eval_gini_satisfaction": float(city.get("gini_satisfaction", 0.0)),
                "eval_utilization": float(city.get("mean_utilization", 0.0)),
                "eval_population": float(city.get("population", 0.0)),
                "eval_green_share": float(city.get("green_share", 0.0)),
                "eval_damage": float(city.get("externality_damage_total", 0.0)),
                "eval_bankrupt_agents": float(city.get("bankrupt_agents", 0.0)),
            }
        )
        if (
            self.config.track_spatial
            and (gen % max(1, self.config.spatial_every) == 0
                 or gen == self.config.generations - 1)
        ):
            row.update(self._spatial_snapshot())
        return row

    def _spatial_snapshot(self) -> dict[str, float]:
        if self.best_ever_genome is None:
            return {}
        best = max(self.population, key=lambda ind: ind.fitness)
        policies = [
            NeuralPolicy(best.genome, self.layout)
            for _ in range(self.city_config.n_agents)
        ]
        result = run_episode(
            self.city_config,
            policies,
            seed=self.config.validation_seed_base + 777,
            rng_seed=self.config.seed + 8888,
        )
        metrics = spatial_metrics(result.type_grid)
        metrics["snapshot_satisfaction"] = result.city.mean_satisfaction
        metrics["snapshot_quality"] = result.city.city_quality
        return metrics


# ---------------------------------------------------------------------------
# Evaluation helpers used by experiments
# ---------------------------------------------------------------------------


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci, "n": int(x.size)}


def evaluate_population(
    population: list[Individual],
    city_config: CityConfig,
    evolution_config: EvolutionConfig,
    seeds: list[int],
    episodes_per_seed: int | None = None,
) -> dict:
    """Evaluate the whole population on held-out seed blocks.

    Every genome appears at least once per seed when
    ``episodes_per_seed * n_agents >= population_size``.
    """
    k = len(population)
    n = city_config.n_agents
    episodes = episodes_per_seed or max(1, int(np.ceil(k / n)))
    layout = layout_from_config(evolution_config)
    fitness_sum = np.zeros(k, dtype=np.float64)
    behavior_sum = np.zeros((k, BEHAVIOR_DIM), dtype=np.float64)
    appearance = np.zeros(k, dtype=np.int64)
    per_seed: list[dict] = []

    tasks: list[EpisodeTask] = []
    seed_groups: list[tuple[int, list[np.ndarray], int]] = []
    for seed in seeds:
        rng = np.random.default_rng((int(seed) * 6_364_136_223 + 0x1234) % (2**63))
        assignments = balanced_episode_indices(rng, k, n, episodes)
        task_start = len(tasks)
        for ep, devs in enumerate(assignments):
            episode_seed = int(seed) * 1000 + ep
            genomes = [population[int(idx)].genome for idx in devs]
            tasks.append(EpisodeTask(genomes, episode_seed))
        seed_groups.append((int(seed), assignments, task_start))

    if tasks:
        executor = EpisodeExecutor(
            city_config,
            layout,
            parallel=evolution_config.use_parallel,
            max_workers=_max_workers(evolution_config, len(tasks)),
            process_pool_factory=ProcessPoolExecutor,
        )
        try:
            results = executor.map(tasks)
        finally:
            executor.shutdown()
    else:
        results = []

    for seed, assignments, task_start in seed_groups:
        seed_results = results[task_start : task_start + len(assignments)]
        seed_scores: list[float] = []
        seed_cities: list[CitySummary] = []
        for result, devs in zip(seed_results, assignments):
            seed_scores.extend(float(value) for value in result.scores)
            seed_cities.append(result.city)
            for slot, idx in enumerate(devs):
                idx = int(idx)
                fitness_sum[idx] += float(result.scores[slot])
                behavior_sum[idx] += np.asarray(
                    result.behaviors[slot], dtype=np.float64
                )
                appearance[idx] += 1
        per_seed.append(
            {
                "seed": seed,
                "fitness_mean": float(np.mean(seed_scores)),
                "fitness_std": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
                "city_quality": float(
                    np.mean([city.city_quality for city in seed_cities])
                ),
                "mean_satisfaction": float(
                    np.mean([city.mean_satisfaction for city in seed_cities])
                ),
                "green_share": float(np.mean([city.green_share for city in seed_cities])),
                "damage_total": float(
                    np.mean([city.externality_damage_total for city in seed_cities])
                ),
            }
        )

    per_individual = np.full(k, np.nan, dtype=np.float64)
    per_behavior = np.full((k, BEHAVIOR_DIM), np.nan, dtype=np.float64)
    for i in range(k):
        if appearance[i] > 0:
            per_individual[i] = fitness_sum[i] / appearance[i]
            per_behavior[i] = behavior_sum[i] / appearance[i]
    seed_means = [row["fitness_mean"] for row in per_seed]
    return {
        "per_individual_fitness": per_individual,
        "per_individual_behavior": per_behavior,
        "per_seed": per_seed,
        "fitness": _stats(seed_means),
        "diversity": mean_pairwise_distance(per_behavior),
        "unique_behaviors": unique_behaviors(per_behavior),
    }


def evaluate_league(
    focal_factory: Callable[[], Policy],
    city_config: CityConfig,
    seeds: list[int],
    opponent_factories: list[Callable[[], Policy]] | None = None,
) -> dict:
    """Evaluate one focal policy among a rotating league of opponents.

    This is the fair comparison protocol for E2: a strategy is scored in the
    same mixed multi-agent setting in which it was evolved, not in a
    monoculture of its own copies.  The focal agent's slot rotates over seeds.
    """
    if opponent_factories is None:
        opponent_factories = [
            lambda: ManualPolicy(),
            lambda: GreedyPolicy(),
            lambda: RandomPolicy(),
        ]
    per_seed: list[dict] = []
    behaviors: list[np.ndarray] = []
    n = city_config.n_agents
    for index, seed in enumerate(seeds):
        policies = [
            opponent_factories[(index + j) % len(opponent_factories)]()
            for j in range(n)
        ]
        slot = index % n
        policies[slot] = focal_factory()
        result = run_episode(city_config, policies, int(seed))
        scores = score_episode(result)
        focal = result.agents[slot]
        behaviors.append(focal.behavior)
        per_seed.append(
            {
                "seed": int(seed),
                "fitness": float(scores[slot]),
                "profit": float(focal.profit),
                "city_quality": float(result.city.city_quality),
                "mean_satisfaction": float(result.city.mean_satisfaction),
                "gini_satisfaction": float(result.city.gini_satisfaction),
                "green_share": float(result.city.green_share),
                "green_component": float(result.city.green_component),
                "type_entropy": float(result.city.type_entropy),
                "accessibility": float(result.city.accessibility),
                "compatibility": float(result.city.compatibility),
                "violations": float(result.city.violations_total),
                "damage_total": float(result.city.externality_damage_total),
                "damage_per_agent": float(result.city.externality_damage_per_agent),
                "share_agents_causing_damage": float(result.city.share_agents_causing_damage),
                "gini_profit": float(result.city.gini_profit),
                "built_cells": int(result.city.built_cells),
                "tax_revenue": float(result.city.tax_revenue),
                "subsidy_total": float(result.city.subsidy_total),
                "bankrupt_agents": int(result.city.bankrupt_agents),
                "behavior": focal.behavior.tolist(),
            }
        )
    keys = [
        "fitness",
        "profit",
        "city_quality",
        "mean_satisfaction",
        "gini_satisfaction",
        "green_share",
        "green_component",
        "type_entropy",
        "accessibility",
        "compatibility",
        "violations",
        "damage_total",
        "damage_per_agent",
        "share_agents_causing_damage",
        "gini_profit",
        "built_cells",
        "tax_revenue",
        "subsidy_total",
        "bankrupt_agents",
    ]
    summary = {key: _stats([row[key] for row in per_seed]) for key in keys}
    summary["behavior"] = np.mean(behaviors, axis=0).tolist() if behaviors else [1.0, 0.0, 0.0, 0.0]
    summary["per_seed"] = per_seed
    return summary


def evaluate_homogeneous(
    policy_factory: Callable[[], Policy],
    city_config: CityConfig,
    seeds: list[int],
) -> dict:
    """Evaluate one policy in a city populated by copies of itself."""
    per_seed: list[dict] = []
    behavior_accumulator = []
    for seed in seeds:
        policies = [policy_factory() for _ in range(city_config.n_agents)]
        result: EpisodeResult = run_episode(city_config, policies, int(seed))
        scores = score_episode(result)
        profits = np.array([a.profit for a in result.agents], dtype=np.float64)
        behavior = np.mean([a.behavior for a in result.agents], axis=0)
        behavior_accumulator.append(behavior)
        per_seed.append(
            {
                "seed": int(seed),
                "fitness": float(np.mean(scores)),
                "fitness_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "profit": float(np.mean(profits)),
                "city_quality": float(result.city.city_quality),
                "mean_satisfaction": float(result.city.mean_satisfaction),
                "gini_satisfaction": float(result.city.gini_satisfaction),
                "rich_satisfaction": float(result.city.rich_satisfaction),
                "poor_satisfaction": float(result.city.poor_satisfaction),
                "green_share": float(result.city.green_share),
                "green_component": float(result.city.green_component),
                "type_entropy": float(result.city.type_entropy),
                "accessibility": float(result.city.accessibility),
                "compatibility": float(result.city.compatibility),
                "violations": float(result.city.violations_total),
                "damage_total": float(result.city.externality_damage_total),
                "damage_per_agent": float(result.city.externality_damage_per_agent),
                "share_agents_causing_damage": float(result.city.share_agents_causing_damage),
                "gini_profit": float(result.city.gini_profit),
                "built_cells": int(result.city.built_cells),
                "population": float(result.city.population),
                "mean_utilization": float(result.city.mean_utilization),
                "tax_revenue": float(result.city.tax_revenue),
                "subsidy_total": float(result.city.subsidy_total),
                "bankrupt_agents": int(result.city.bankrupt_agents),
                "affordable_share": float(result.city.affordable_share),
                "behavior": behavior.tolist(),
            }
        )
    keys = [
        "fitness",
        "profit",
        "city_quality",
        "mean_satisfaction",
        "gini_satisfaction",
        "rich_satisfaction",
        "poor_satisfaction",
        "green_share",
        "green_component",
        "type_entropy",
        "accessibility",
        "compatibility",
        "violations",
        "damage_total",
        "damage_per_agent",
        "share_agents_causing_damage",
        "gini_profit",
        "built_cells",
        "population",
        "mean_utilization",
        "tax_revenue",
        "subsidy_total",
        "bankrupt_agents",
        "affordable_share",
    ]
    summary = {key: _stats([row[key] for row in per_seed]) for key in keys}
    summary["behavior"] = np.mean(behavior_accumulator, axis=0).tolist()
    summary["per_seed"] = per_seed
    return summary
