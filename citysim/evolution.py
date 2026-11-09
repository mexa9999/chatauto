"""Population-level neuroevolution: islands, tournament, sharing, migration."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import CityConfig, EvolutionConfig
from .episode import EpisodeResult, run_episode, score_episode
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

def _run_episode_worker(genomes, city_config, seed, layout):
    from .policy import NeuralPolicy
    policies = [NeuralPolicy(g, layout) for g in genomes]
    from .episode import run_episode, score_episode
    result = run_episode(city_config, policies, seed)
    scores = score_episode(result, city_config.fitness)
    behaviors = [a.behavior for a in result.agents]
    return scores, behaviors


@dataclass
class Individual:
    genome: np.ndarray
    island: int = 0
    fitness: float = -1e18
    shaped_fitness: float = -1e18
    behavior: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
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
    def __init__(self, config: EvolutionConfig, city_config: CityConfig, executor=None):
        if config.episodes_per_generation * city_config.n_agents < config.population_size:
            raise ValueError(
                "episodes_per_generation * n_agents must be >= population_size "
                "so that every genome is evaluated"
            )
        self.config = config
        self.city_config = city_config
        self.layout = GenomeLayout(hidden_size=config.hidden_size)
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
        self.best_ever = -1e18
        self.best_ever_genome: np.ndarray | None = None
        self.best_ever_behavior: np.ndarray = np.zeros(3)
        self.best_ever_generation = -1
        self.hall_of_fame: list[tuple[int, float, np.ndarray]] = []
        self._own_executor = executor is None
        self.executor = executor if executor is not None else ProcessPoolExecutor(max_workers=os.cpu_count())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> EvolutionResult:
        for gen in range(self.config.generations):
            self.generation = gen
            self._evaluate_generation(gen)
            row = self._log_generation(gen)
            self.history.append(row)
            if self.config.verbose and (
                gen % max(1, self.config.log_every) == 0
                or gen == self.config.generations - 1
            ):
                print(
                    f"[gen {gen:4d}] best={row['best']:.4f} "
                    f"mean={row['mean']:.4f} div={row['diversity']:.4f} "
                    f"sigma={row['sigma']:.4f} best_ever={row['best_ever']:.4f}"
                )
            if gen == self.config.generations - 1:
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

        if self.best_ever_genome is None:
            best = max(self.population, key=lambda ind: ind.fitness)
            self.best_ever = best.fitness
            self.best_ever_genome = best.genome.copy()
            self.best_ever_behavior = best.behavior.copy()
            self.best_ever_generation = self.generation
        if self._own_executor:
            self.executor.shutdown(wait=True)
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
        )

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
        behavior_sum = np.zeros((cfg.population_size, 3), dtype=np.float64)
        appearances = np.zeros(cfg.population_size, dtype=np.int64)

        # Подготовка задач для параллельного запуска
        # Подготовка задач
        tasks = []
        for episode_idx, devs in enumerate(episodes):
            seed = cfg.train_seed_base + gen * cfg.episodes_per_generation + episode_idx
            genomes = [self.population[int(idx)].genome for idx in devs]
            tasks.append((genomes, seed))
            
        # Параллельный запуск в постоянном пуле
        futures = [
            self.executor.submit(_run_episode_worker, genomes, self.city_config, seed, self.layout)
            for genomes, seed in tasks
        ]
        results = [f.result() for f in futures]
        
        for (scores, behaviors), devs in zip(results, episodes):
            for slot, idx in enumerate(devs):
                idx = int(idx)
                fitness_sum[idx] += float(scores[slot])
                behavior_sum[idx] += np.array(behaviors[slot])
                appearances[idx] += 1
        
        for i, ind in enumerate(self.population):
            ind.episodes = int(appearances[i])
            if appearances[i] > 0:
                ind.fitness = float(fitness_sum[i] / appearances[i])
                ind.behavior = behavior_sum[i] / appearances[i]
            else:
                ind.fitness = -1e18
                ind.behavior = np.zeros(3)

        self._apply_fitness_sharing()

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
        groups: list[list[int]]
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
                members = [
                    ind for ind in self.population if ind.island == island
                ]
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
            worst.behavior = np.zeros(3)
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
                ind.behavior = np.zeros(3)
                ind.episodes = 0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_generation(self, gen: int) -> dict:
        fitness = np.array([ind.fitness for ind in self.population], dtype=np.float64)
        shaped = np.array([ind.shaped_fitness for ind in self.population], dtype=np.float64)
        behaviors = np.array([ind.behavior for ind in self.population], dtype=np.float64)
        return {
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
        }


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
    layout = GenomeLayout(hidden_size=evolution_config.hidden_size)
    fitness_sum = np.zeros(k, dtype=np.float64)
    behavior_sum = np.zeros((k, 3), dtype=np.float64)
    appearance = np.zeros(k, dtype=np.int64)
    per_seed: list[dict] = []

    for seed in seeds:
        rng = np.random.default_rng((int(seed) * 6_364_136_223 + 0x1234) % (2**63))
        assignments = balanced_episode_indices(rng, k, n, episodes)
        seed_scores: list[float] = []
        for ep, devs in enumerate(assignments):
            episode_seed = int(seed) * 1000 + ep
            policies = [
                NeuralPolicy(population[int(idx)].genome, layout) for idx in devs
            ]
            result = run_episode(city_config, policies, episode_seed)
            scores = score_episode(result)
            seed_scores.extend(float(x) for x in scores)
            for slot, idx in enumerate(devs):
                idx = int(idx)
                fitness_sum[idx] += float(scores[slot])
                behavior_sum[idx] += result.agents[slot].behavior
                appearance[idx] += 1
        per_seed.append(
            {
                "seed": int(seed),
                "fitness_mean": float(np.mean(seed_scores)),
                "fitness_std": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
            }
        )

    per_individual = np.full(k, np.nan, dtype=np.float64)
    per_behavior = np.full((k, 3), np.nan, dtype=np.float64)
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
                "behavior": focal.behavior.tolist(),
            }
        )
    keys = [
        "fitness",
        "profit",
        "city_quality",
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
    ]
    summary = {key: _stats([row[key] for row in per_seed]) for key in keys}
    summary["behavior"] = np.mean(behaviors, axis=0).tolist() if behaviors else [1.0, 0.0, 0.0]
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
                "behavior": behavior.tolist(),
            }
        )
    keys = [
        "fitness",
        "profit",
        "city_quality",
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
    ]
    summary = {
        key: _stats([row[key] for row in per_seed])
        for key in keys
    }
    summary["behavior"] = np.mean(behavior_accumulator, axis=0).tolist()
    summary["per_seed"] = per_seed
    return summary
