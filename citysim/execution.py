"""Execution helpers for independent simulation episodes."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import numpy as np

from .config import CityConfig
from .episode import run_episode, score_episode
from .policy import GenomeLayout, NeuralPolicy

_WORKER_LAYOUT: GenomeLayout | None = None
_WORKER_CITY_CONFIG: CityConfig | None = None


@dataclass(frozen=True)
class EpisodeTask:
    genomes: Sequence[np.ndarray]
    seed: int


class CitySummary(NamedTuple):
    city_quality: float
    mean_satisfaction: float
    rich_satisfaction: float
    poor_satisfaction: float
    gini_satisfaction: float
    mean_utilization: float
    population: float
    total_debt: float
    mean_debt_ratio: float
    housing_vacancy_rate: float
    job_vacancy_rate: float
    unemployment_rate: float
    net_migration_rate: float
    local_goods_coverage: float
    tax_rate_multiplier: float
    green_share: float
    externality_damage_total: float
    bankrupt_agents: float


class EpisodeSummary(NamedTuple):
    scores: np.ndarray
    behaviors: list[np.ndarray]
    city: CitySummary


def _worker_init(layout: GenomeLayout, city_config: CityConfig) -> None:
    global _WORKER_LAYOUT, _WORKER_CITY_CONFIG
    _WORKER_LAYOUT = layout
    _WORKER_CITY_CONFIG = city_config


def _run_episode_worker(
    genomes: Sequence[np.ndarray],
    seed: int,
    city_config: CityConfig | None = None,
    layout: GenomeLayout | None = None,
) -> EpisodeSummary:
    if city_config is None:
        city_config = _WORKER_CITY_CONFIG
    if layout is None:
        layout = _WORKER_LAYOUT
    if city_config is None or layout is None:
        raise RuntimeError("episode worker was not initialized with its configuration")

    policies = [NeuralPolicy(genome, layout) for genome in genomes]
    result = run_episode(city_config, policies, seed)
    scores = score_episode(result, city_config.fitness)
    behaviors = [agent.behavior for agent in result.agents]
    city = result.city
    city_summary = CitySummary(
        city.city_quality,
        city.mean_satisfaction,
        city.rich_satisfaction,
        city.poor_satisfaction,
        city.gini_satisfaction,
        city.mean_utilization,
        city.population,
        city.total_debt,
        city.mean_debt_ratio,
        city.housing_vacancy_rate,
        city.job_vacancy_rate,
        city.unemployment_rate,
        city.net_migration_rate,
        city.local_goods_coverage,
        city.tax_rate_multiplier,
        city.green_share,
        city.externality_damage_total,
        float(city.bankrupt_agents),
    )
    return EpisodeSummary(scores, behaviors, city_summary)


class EpisodeExecutor:
    """Run independent neural-policy episodes serially or in worker processes."""

    def __init__(
        self,
        city_config: CityConfig,
        layout: GenomeLayout,
        *,
        parallel: bool,
        max_workers: int,
        executor: Executor | None = None,
        process_pool_factory: Callable[..., Executor] = ProcessPoolExecutor,
        worker_context: bool = False,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.city_config = city_config
        self.layout = layout
        self.executor = executor
        self._owns_executor = False
        self._uses_worker_context = worker_context

        if self.executor is None and parallel:
            self.executor = process_pool_factory(
                max_workers=max_workers,
                initializer=_worker_init,
                initargs=(layout, city_config),
            )
            self._owns_executor = True
            self._uses_worker_context = True

    @property
    def owns_executor(self) -> bool:
        return self._owns_executor

    def map(
        self, tasks: Sequence[EpisodeTask]
    ) -> list[EpisodeSummary]:
        if not tasks:
            return []
        if self.executor is None:
            return [
                _run_episode_worker(
                    task.genomes, task.seed, self.city_config, self.layout
                )
                for task in tasks
            ]

        if self._uses_worker_context:
            futures = [
                self.executor.submit(_run_episode_worker, task.genomes, task.seed)
                for task in tasks
            ]
        else:
            futures = [
                self.executor.submit(
                    _run_episode_worker,
                    task.genomes,
                    task.seed,
                    self.city_config,
                    self.layout,
                )
                for task in tasks
            ]
        return [future.result() for future in futures]

    def shutdown(self) -> None:
        if self._owns_executor and self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def __enter__(self) -> EpisodeExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()
