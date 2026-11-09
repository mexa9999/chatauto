"""Benchmark serial and process execution for independent neural episodes."""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from citysim.config import EvolutionConfig, quick_city_config
from citysim.evolution import layout_from_config
from citysim.execution import EpisodeExecutor, EpisodeTask
from citysim.policy import random_genome


def _csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _grid_sizes(value: str) -> list[tuple[int, int]]:
    try:
        grids = []
        for part in value.split(","):
            height, width = part.lower().split("x", maxsplit=1)
            grids.append((int(height), int(width)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected grids such as 12x12,20x20") from exc
    if not grids or any(height < 1 or width < 1 for height, width in grids):
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return grids


def _tasks(
    genome: np.ndarray,
    n_agents: int,
    world_count: int,
    seed: int,
) -> list[EpisodeTask]:
    genomes = [genome] * n_agents
    return [
        EpisodeTask(genomes, seed + index)
        for index in range(world_count)
    ]


def _same_results(expected, actual) -> bool:
    if len(expected) != len(actual):
        return False
    return all(
        np.allclose(left[0], right[0])
        and np.allclose(left[1], right[1])
        and np.allclose(left[2], right[2])
        for left, right in zip(expected, actual)
    )


def benchmark(
    grids: list[tuple[int, int]],
    agent_counts: list[int],
    world_counts: list[int],
    worker_counts: list[int],
    repeats: int,
    actions: int,
    seed: int,
) -> None:
    layout_config = EvolutionConfig(
        population_size=2,
        generations=1,
        episodes_per_generation=1,
        elites=1,
        islands=1,
        use_islands=False,
        hidden_size=24,
        use_parallel=False,
    )
    layout = layout_from_config(layout_config)
    genome = random_genome(np.random.default_rng(seed), layout)
    print(
        "grid,agents,worlds,workers,serial_s,parallel_s,speedup,"
        "same_results"
    )

    for height, width in grids:
        for n_agents in agent_counts:
            city = quick_city_config(
                height=height,
                width=width,
                n_agents=n_agents,
                actions_per_agent=actions,
            )
            with EpisodeExecutor(
                city, layout, parallel=False, max_workers=1
            ) as serial_executor:
                serial_executor.map(_tasks(genome, n_agents, 1, seed - 1))
                for world_count in world_counts:
                    for effective_workers in sorted(
                        {min(workers, world_count) for workers in worker_counts}
                    ):
                        with EpisodeExecutor(
                            city,
                            layout,
                            parallel=True,
                            max_workers=effective_workers,
                        ) as process_executor:
                            warmup_tasks = _tasks(
                                genome,
                                n_agents,
                                world_count,
                                seed - world_count,
                            )
                            process_executor.map(warmup_tasks)
                            serial_times: list[float] = []
                            process_times: list[float] = []
                            same_results = True
                            for repeat in range(repeats):
                                tasks = _tasks(
                                    genome,
                                    n_agents,
                                    world_count,
                                    seed + repeat * max(world_counts),
                                )
                                start = perf_counter()
                                expected = serial_executor.map(tasks)
                                serial_times.append(perf_counter() - start)

                                start = perf_counter()
                                actual = process_executor.map(tasks)
                                process_times.append(perf_counter() - start)
                                same_results = same_results and _same_results(
                                    expected, actual
                                )

                            if not same_results:
                                raise RuntimeError(
                                    "serial and process episode results differ "
                                    f"for {height}x{width}, {n_agents} agents, "
                                    f"{world_count} worlds, {effective_workers} workers"
                                )

                            serial_seconds = median(serial_times)
                            process_seconds = median(process_times)
                            speedup = (
                                serial_seconds / process_seconds
                                if process_seconds > 0
                                else float("inf")
                            )
                            print(
                                f"{height}x{width},{n_agents},{world_count},"
                                f"{effective_workers},{serial_seconds:.6f},"
                                f"{process_seconds:.6f},{speedup:.3f},"
                                f"{same_results}"
                            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=_grid_sizes, default=_grid_sizes("12x12"))
    parser.add_argument("--agents", type=_csv_ints, default=_csv_ints("10,15"))
    parser.add_argument("--worlds", type=_csv_ints, default=_csv_ints("1,2,4"))
    parser.add_argument("--workers", type=_csv_ints, default=_csv_ints("1,2,4"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1 or args.actions < 1:
        parser.error("--repeats and --actions must be positive")
    benchmark(
        args.grids,
        args.agents,
        args.worlds,
        args.workers,
        args.repeats,
        args.actions,
        args.seed,
    )


if __name__ == "__main__":
    main()
