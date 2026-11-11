"""Shared helpers for experiment orchestration and reporting."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import os
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np

from .config import CityConfig, EvolutionConfig, seed_bases_for_run
from .evolution import (
    EvolutionEngine,
    EvolutionResult,
    _configuration_digest,
)
from .io import ensure_dir, save_genome, write_csv, write_json
from .plotting import plot_learning_curve
from .progress import ProgressCallback, StopRequested, TrainingStopped
from .training_history import TrainingHistoryStore


_TRAINING_CACHE: ContextVar[
    tuple[
        TrainingHistoryStore,
        Path,
        bool,
        int,
        ProgressCallback | None,
        StopRequested | None,
    ]
    | None
] = ContextVar(
    "citysim_training_cache", default=None
)


@contextmanager
def training_cache(
    cache_dir: str | Path,
    resume: bool = False,
    checkpoint_every: int = 10,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
) -> Iterator[None]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least one")
    history_store = TrainingHistoryStore(cache_dir)
    token = _TRAINING_CACHE.set(
        (
            history_store,
            Path(cache_dir),
            resume,
            checkpoint_every,
            progress_callback,
            stop_requested,
        )
    )
    try:
        yield
    finally:
        _TRAINING_CACHE.reset(token)


def _run_evolution(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    engine_factory: Callable[
        [EvolutionConfig, CityConfig], EvolutionEngine
    ] = EvolutionEngine,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
) -> EvolutionResult:
    cache = _TRAINING_CACHE.get()
    if cache is None:
        return engine_factory(evolution_config, city_config).run(
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )

    (
        history_store,
        cache_dir,
        resume,
        checkpoint_every,
        progress_callback,
        stop_requested,
    ) = cache
    configuration_digest = _configuration_digest(evolution_config, city_config)
    checkpoint = history_store.work / (
        f"experiment-{configuration_digest}-{os.getpid()}.npz"
    )
    if stop_requested is not None and stop_requested():
        raise TrainingStopped("training stopped before starting the next cached run")

    cache_entry = history_store.restore_experiment_checkpoint(
        configuration_digest,
        checkpoint,
    )
    if cache_entry is None:
        legacy_checkpoint = cache_dir / f"{configuration_digest}.npz"
        if legacy_checkpoint.is_file():
            shutil.copyfile(legacy_checkpoint, checkpoint)
            history_store.record_experiment_checkpoint(
                configuration_digest, checkpoint
            )
            cache_entry = history_store.restore_experiment_checkpoint(
                configuration_digest,
                checkpoint,
            )
    if cache_entry is not None:
        if cache_entry["complete"]:
            print(f"  [cache] Reusing completed training {configuration_digest[:12]}.")
            return EvolutionEngine.load_completed(
                checkpoint, evolution_config, city_config
            )
        if not resume:
            raise FileExistsError(
                f"incomplete checkpoint for {configuration_digest[:12]}; "
                "rerun with --resume"
            )
        print(f"  [resume] Continuing training {configuration_digest[:12]}.")

    def save_checkpoint(path: Path, generation: int, complete: bool) -> None:
        history_store.record_experiment_checkpoint(configuration_digest, path)

    result = engine_factory(evolution_config, city_config).run(
        checkpoint_path=checkpoint,
        resume=resume or cache_entry is not None,
        checkpoint_every=checkpoint_every,
        checkpoint_callback=save_checkpoint,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )
    if stop_requested is not None and stop_requested():
        from .checkpoint_io import read_checkpoint_metadata

        if not read_checkpoint_metadata(checkpoint)["complete"]:
            raise TrainingStopped("training stopped after saving a generation checkpoint")
    return result


def _stats(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci, "n": int(x.size)}


def _run_config(base: EvolutionConfig, run_seed: int) -> EvolutionConfig:
    train, validation, test = seed_bases_for_run(run_seed)
    return replace(
        base,
        seed=int(run_seed),
        train_seed_base=train,
        validation_seed_base=validation,
        test_seed_base=test,
        verbose=True,
    )


def _ensure_evolution_feasible(
    config: EvolutionConfig, n_agents: int
) -> EvolutionConfig:
    """Bump population/episodes so every agent can host a distinct genome."""
    population = max(config.population_size, int(n_agents))
    if config.use_islands and config.islands > 1:
        population = int(
            np.ceil(population / config.islands) * config.islands
        )
    episodes = int(config.episodes_per_generation)
    while episodes * int(n_agents) < population:
        episodes += 1
    if population == config.population_size and episodes == config.episodes_per_generation:
        return config
    return replace(
        config,
        population_size=population,
        episodes_per_generation=episodes,
    )


def _history_rows(result: EvolutionResult) -> list[dict]:
    rows = []
    for row in result.history:
        flat = {}
        for key, value in row.items():
            if isinstance(value, list):
                flat[key] = "|".join(str(x) for x in value)
            else:
                flat[key] = value
        rows.append(flat)
    return rows


def train_evolution(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    out_dir: str | Path | None = None,
    tag: str = "run",
    cache_dir: str | Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 10,
    history_store: TrainingHistoryStore | None = None,
    branch: str = "main",
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    run_id: str | None = None,
) -> EvolutionResult:
    out = ensure_dir(out_dir) if out_dir is not None else None
    if history_store is not None:
        result = _run_with_history(
            evolution_config,
            city_config,
            history_store,
            branch,
            resume,
            checkpoint_every,
            progress_callback,
            stop_requested,
            run_id,
        )
    elif cache_dir is not None:
        with training_cache(
            cache_dir,
            resume=resume,
            checkpoint_every=checkpoint_every,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        ):
            result = _run_evolution(evolution_config, city_config)
    else:
        result = _run_evolution(
            evolution_config,
            city_config,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
    if out is not None:
        write_csv(out / f"{tag}_history.csv", _history_rows(result))
        save_genome(out / f"{tag}_best_genome.npy", result.best_genome)
        archive_rows = [
            {
                "niche": f"{key[0]}_{key[1]}",
                "fitness": ind.fitness,
                "behavior": ind.behavior,
            }
            for key, ind in result.archive.items()
        ]
        if archive_rows:
            write_csv(out / f"{tag}_archive.csv", archive_rows)
            np.savez(
                out / f"{tag}_archive_genomes.npz",
                **{f"niche_{key[0]}_{key[1]}": ind.genome for key, ind in result.archive.items()},
            )
        write_json(
            out / f"{tag}_summary.json",
            {
                "best_fitness": result.best_fitness,
                "best_generation": result.best_generation,
                "best_behavior": result.best_behavior,
                "evolution_config": evolution_config.as_dict(),
                "city_config": city_config.as_dict(),
                "archive": archive_rows,
            },
        )
        plot_learning_curve([result.history], out / f"{tag}_learning_curve.png")
    return result


def _run_with_history(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    history_store: TrainingHistoryStore,
    branch: str,
    resume: bool,
    checkpoint_every: int,
    progress_callback: ProgressCallback | None,
    stop_requested: StopRequested | None,
    run_id: str | None,
) -> EvolutionResult:
    branches = {row["name"]: row for row in history_store.branch_names()}
    if branch not in branches:
        raise ValueError(f"unknown history branch: {branch}")
    head = branches[branch]["head_commit"]
    checkpoint = history_store.work_path(branch)
    if resume:
        if head is None:
            raise ValueError(f"branch has no checkpoint to resume: {branch}")
        shutil.copyfile(history_store.checkpoint_path(branch), checkpoint)
    elif head is not None:
        raise FileExistsError(
            f"branch {branch!r} already has history; pass --resume or create a branch"
        )
    elif checkpoint.exists():
        # A work checkpoint without a branch commit is orphaned and cannot be
        # resumed through the history API; start the empty branch cleanly.
        checkpoint.unlink()

    def record_snapshot(path: Path, generation: int, complete: bool) -> None:
        nonlocal head
        head = history_store.record_checkpoint(
            branch,
            path,
            generation,
            complete,
            evolution_config,
            city_config,
            expected_parent=head,
            run_id=run_id,
        )

    return EvolutionEngine(evolution_config, city_config).run(
        checkpoint_path=checkpoint,
        resume=resume,
        checkpoint_every=checkpoint_every,
        allow_config_changes=resume,
        checkpoint_callback=record_snapshot,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )


def _aggregate_history(histories: list[list[dict]], keys: Sequence[str]) -> list[dict]:
    max_len = max((len(h) for h in histories), default=0)
    rows = []
    for gen in range(max_len):
        row: dict = {"gen": gen}
        for key in keys:
            values = [
                float(h[gen][key])
                for h in histories
                if gen < len(h) and key in h[gen]
            ]
            if values:
                stats = _stats(values)
                row[f"{key}_mean"] = stats["mean"]
                row[f"{key}_std"] = stats["std"]
                row[f"{key}_ci95"] = stats["ci95"]
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
                row[f"{key}_ci95"] = float("nan")
        rows.append(row)
    return rows


def _summary_value(summary: dict, key: str) -> float:
    return float(summary[key]["mean"])


def _run_metric_summary(summaries: Sequence[dict], key: str) -> dict[str, float]:
    return _stats([_summary_value(summary, key) for summary in summaries])


def _eval_seeds(base: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("eval_seed_count must be at least one")
    return [int(base) + i for i in range(count)]


def _require_training_seeds(seeds: Sequence[int], experiment: str) -> None:
    if len(seeds) == 0:
        raise ValueError(f"{experiment} needs at least one training seed")


def _metric_row(metrics: dict, keys: Sequence[str]) -> dict:
    return {
        **{f"{key}_mean": metrics[key]["mean"] for key in keys},
        **{f"{key}_ci95": metrics[key]["ci95"] for key in keys},
    }
