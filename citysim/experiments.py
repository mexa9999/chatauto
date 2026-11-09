"""Experiment runners E1-E11 and shared aggregation helpers."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import (
    CityConfig,
    EvolutionConfig,
    REGULATION_SCENARIOS,
    REQUIREMENT_PROFILES,
    seed_bases_for_run,
)
from .episode import run_episode, score_episode
from .evolution import (
    EvolutionEngine,
    EvolutionResult,
    evaluate_homogeneous,
    evaluate_league,
    evaluate_population,
    layout_from_config,
    mean_pairwise_distance,
    unique_behaviors,
)
from .io import ensure_dir, save_genome, write_csv, write_json
from .metrics import (
    grid_distance,
    kendall_tau,
    resilience_metrics,
    spatial_metrics,
    type_distribution,
)
from .plotting import (
    plot_bars,
    plot_distribution_matrix,
    plot_learning_curve,
    plot_metric_vs_x,
    plot_profit_quality,
    plot_profit_satisfaction,
    plot_satisfaction_curve,
)
from .policy import (
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    manual_variants,
    random_genome,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _max_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


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
) -> EvolutionResult:
    out = ensure_dir(out_dir) if out_dir is not None else None
    result = EvolutionEngine(evolution_config, city_config).run()
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
    return [int(base) + i for i in range(count)]


def _metric_row(metrics: dict, keys: Sequence[str]) -> dict:
    return {
        **{f"{key}_mean": metrics[key]["mean"] for key in keys},
        **{f"{key}_ci95": metrics[key]["ci95"] for key in keys},
    }


def _shared_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=_max_workers())


# ---------------------------------------------------------------------------
# E1. Learning curves
# ---------------------------------------------------------------------------


def e1_learning_curve(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    del eval_seed_count
    out = ensure_dir(out_dir)
    histories: list[list[dict]] = []
    runs: list[dict] = []
    with _shared_executor() as shared_executor:
        for run_seed in seeds:
            print(f"  [E1] Training neuroevolution, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            histories.append(result.history)
            save_genome(out / f"best_genome_seed{run_seed}.npy", result.best_genome)
            best_curve = np.array([row["best"] for row in result.history], dtype=np.float64)
            final_best = float(best_curve[-1])
            if final_best > 0.0:
                threshold = 0.9 * final_best
                reached_mask = best_curve >= threshold
                reached = int(np.argmax(reached_mask)) if reached_mask.any() else -1
            else:
                reached = -1
            runs.append(
                {
                    "seed": int(run_seed),
                    "best_fitness": result.best_fitness,
                    "best_generation": result.best_generation,
                    "final_best": final_best,
                    "final_mean": float(result.history[-1]["mean"]),
                    "final_diversity": float(result.history[-1]["diversity"]),
                    "final_unique_behaviors": int(result.history[-1]["unique_behaviors"]),
                    "generation_to_90pct": reached,
                    "best_behavior": result.best_behavior,
                }
            )
            print(f"  [E1] seed {run_seed} training done.")
    aggregate = _aggregate_history(
        histories,
        [
            "best", "mean", "std", "diversity", "unique_behaviors", "sigma",
            "best_ever", "best_green_share", "archive_coverage",
            "morans_i", "local_entropy", "cluster_count", "snapshot_satisfaction",
            "eval_city_quality", "eval_satisfaction", "eval_rich_satisfaction",
            "eval_poor_satisfaction", "eval_utilization", "eval_population",
            "eval_damage", "eval_bankrupt_agents",
        ],
    )
    write_csv(out / "e1_history.csv", aggregate)
    plot_learning_curve(histories, out / "e1_learning_curve.png")
    plot_satisfaction_curve(histories, out / "e1_satisfaction.png")
    summary = {
        "experiment": "E1",
        "seeds": [int(s) for s in seeds],
        "runs": runs,
        "best_fitness": _stats([r["best_fitness"] for r in runs]),
        "final_diversity": _stats([r["final_diversity"] for r in runs]),
        "generation_to_90pct": _stats([r["generation_to_90pct"] for r in runs]),
    }
    write_json(out / "e1_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E2. Neural evolution vs baselines
# ---------------------------------------------------------------------------


def _evaluate_behavior_set(
    factories: Sequence[Callable[[], object]],
    city_config: CityConfig,
    seeds: Sequence[int],
) -> np.ndarray:
    behaviors = []
    for factory in factories:
        summary = evaluate_league(factory, city_config, list(seeds))
        behaviors.append(np.asarray(summary["behavior"], dtype=np.float64))
    return np.array(behaviors)


def e2_baselines(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
    behavior_variants: int = 32,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_000_000, eval_seed_count)
    layout = layout_from_config(evolution_config)
    neural_summaries: list[dict] = []
    neural_run_records: list[dict] = []
    evolved_behaviors: list[np.ndarray] = []

    with _shared_executor() as shared_executor:
        for run_seed in seeds:
            print(f"  [E2] Training neuroevolution, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            save_genome(out / f"best_genome_seed{run_seed}.npy", result.best_genome)
            summary = evaluate_league(
                lambda: NeuralPolicy(result.best_genome, layout),
                city_config,
                eval_seeds,
            )
            homogeneous = evaluate_homogeneous(
                lambda: NeuralPolicy(result.best_genome, layout),
                city_config,
                eval_seeds,
            )
            neural_summaries.append(summary)
            evolved_behaviors.append(
                np.array([ind.behavior for ind in result.population], dtype=np.float64)
            )
            neural_run_records.append(
                {
                    "seed": int(run_seed),
                    "best_fitness": result.best_fitness,
                    "league_fitness": _summary_value(summary, "fitness"),
                    "homogeneous_fitness": _summary_value(homogeneous, "fitness"),
                    "eval_profit": _summary_value(summary, "profit"),
                    "eval_city_quality": _summary_value(summary, "city_quality"),
                    "eval_satisfaction": _summary_value(summary, "mean_satisfaction"),
                    "eval_damage": _summary_value(summary, "damage_total"),
                    "eval_violations": _summary_value(summary, "violations"),
                }
            )
            print(f"  [E2] seed {run_seed} training done.")

    print("  [E2] Evaluating baselines (manual, greedy, random)...")
    baseline_factories: dict[str, Callable[[], object]] = {
        "manual": lambda: ManualPolicy(),
        "greedy": lambda: GreedyPolicy(),
        "random": lambda: RandomPolicy(),
    }
    baseline_summaries = {
        name: evaluate_league(factory, city_config, eval_seeds)
        for name, factory in baseline_factories.items()
    }
    baseline_homogeneous = {
        name: evaluate_homogeneous(factory, city_config, eval_seeds)
        for name, factory in baseline_factories.items()
    }

    behavior_seeds = eval_seeds[: min(3, len(eval_seeds))]
    rng = np.random.default_rng(int(evolution_config.seed) + 7001)
    manual_set = _evaluate_behavior_set(
        [lambda p=p: ManualPolicy(p) for p in manual_variants(rng, behavior_variants)],
        city_config,
        behavior_seeds,
    )
    random_set = _evaluate_behavior_set(
        [
            lambda g=random_genome(rng, layout): NeuralPolicy(g, layout)
            for _ in range(behavior_variants)
        ],
        city_config,
        behavior_seeds,
    )
    evolved_diversity = [mean_pairwise_distance(b) for b in evolved_behaviors]
    evolved_unique = [unique_behaviors(b) for b in evolved_behaviors]

    rows = []
    neural_fitness = _run_metric_summary(neural_summaries, "fitness")
    for condition, summary in [("neuroevolution", None), *baseline_summaries.items()]:
        if condition == "neuroevolution":
            rows.append(
                {
                    "condition": condition,
                    "fitness_mean": neural_fitness["mean"],
                    "fitness_ci95": neural_fitness["ci95"],
                    "homogeneous_fitness_mean": _stats(
                        [r["homogeneous_fitness"] for r in neural_run_records]
                    )["mean"],
                    "profit_mean": _run_metric_summary(neural_summaries, "profit")["mean"],
                    "city_quality_mean": _run_metric_summary(neural_summaries, "city_quality")["mean"],
                    "satisfaction_mean": _run_metric_summary(neural_summaries, "mean_satisfaction")["mean"],
                    "damage_total_mean": _run_metric_summary(neural_summaries, "damage_total")["mean"],
                    "violations_mean": _run_metric_summary(neural_summaries, "violations")["mean"],
                    "behavior_diversity": float(np.mean(evolved_diversity)),
                    "unique_behaviors": float(np.mean(evolved_unique)),
                }
            )
        else:
            rows.append(
                {
                    "condition": condition,
                    "fitness_mean": _summary_value(summary, "fitness"),
                    "fitness_ci95": summary["fitness"]["ci95"],
                    "homogeneous_fitness_mean": _summary_value(baseline_homogeneous[condition], "fitness"),
                    "profit_mean": _summary_value(summary, "profit"),
                    "city_quality_mean": _summary_value(summary, "city_quality"),
                    "satisfaction_mean": _summary_value(summary, "mean_satisfaction"),
                    "damage_total_mean": _summary_value(summary, "damage_total"),
                    "violations_mean": _summary_value(summary, "violations"),
                    "behavior_diversity": 0.0 if condition == "greedy" else float(
                        mean_pairwise_distance(manual_set if condition == "manual" else random_set)
                    ),
                    "unique_behaviors": 0.0 if condition == "greedy" else float(
                        unique_behaviors(manual_set if condition == "manual" else random_set)
                    ),
                }
            )

    write_csv(out / "e2_summary.csv", rows)
    plot_bars(
        [row["condition"] for row in rows],
        [row["fitness_mean"] for row in rows],
        [row["fitness_ci95"] for row in rows],
        out / "e2_fitness.png",
        "mean fitness",
        "E2: neuroevolution vs baselines",
    )
    best_baseline = max(
        row["fitness_mean"] for row in rows if row["condition"] != "neuroevolution"
    )
    hypotheses = {
        "H1_fitness": bool(neural_fitness["mean"] > best_baseline),
        "H1_behaviour_diversity": bool(
            np.mean(evolved_diversity)
            > max(mean_pairwise_distance(manual_set), mean_pairwise_distance(random_set))
        ),
        "neural_fitness": neural_fitness,
        "best_baseline_fitness": best_baseline,
        "evolved_diversity": float(np.mean(evolved_diversity)),
        "manual_diversity": float(mean_pairwise_distance(manual_set)),
        "random_diversity": float(mean_pairwise_distance(random_set)),
    }
    summary = {
        "experiment": "E2",
        "eval_seeds": eval_seeds,
        "rows": rows,
        "runs": neural_run_records,
        "hypotheses": hypotheses,
    }
    write_json(out / "e2_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E3. Externality ablation
# ---------------------------------------------------------------------------


def e3_externalities(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_100_000, eval_seed_count)
    full_city = replace(city_config, enable_externalities=True)
    conditions = [
        ("with_externalities", True),
        ("without_externalities", False),
    ]
    rows: list[dict] = []
    run_records: list[dict] = []
    with _shared_executor() as shared_executor:
        for condition, enabled in conditions:
            train_city = replace(city_config, enable_externalities=enabled)
            matching_summaries: list[dict] = []
            full_summaries: list[dict] = []
            for run_seed in seeds:
                print(f"  [E3] Training neuroevolution, seed={run_seed}...")
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, train_city, executor=shared_executor).run()
                layout = layout_from_config(evolution_config)
                matching = evaluate_homogeneous(
                    lambda: NeuralPolicy(result.best_genome, layout), train_city, eval_seeds
                )
                full = evaluate_homogeneous(
                    lambda: NeuralPolicy(result.best_genome, layout), full_city, eval_seeds
                )
                matching_summaries.append(matching)
                full_summaries.append(full)
                run_records.append(
                    {
                        "seed": int(run_seed),
                        "condition": condition,
                        "train_best": result.best_fitness,
                        "matching_damage": _summary_value(matching, "damage_total"),
                        "full_damage": _summary_value(full, "damage_total"),
                    }
                )
                print(f"  [E3] seed {run_seed} done.")
            for eval_name, summaries in [
                ("matching", matching_summaries),
                ("full", full_summaries),
            ]:
                metrics = {
                    key: _run_metric_summary(summaries, key)
                    for key in (
                        "fitness", "profit", "city_quality", "mean_satisfaction",
                        "damage_total", "damage_per_agent", "share_agents_causing_damage",
                        "violations", "built_cells",
                    )
                }
                rows.append(
                    {
                        "condition": condition,
                        "eval_env": eval_name,
                        **_metric_row(metrics, list(metrics)),
                    }
                )
    write_csv(out / "e3_summary.csv", rows)
    labels = [f"{row['condition']}\n{row['eval_env']}" for row in rows]
    plot_bars(
        labels,
        [row["damage_total_mean"] for row in rows],
        [row["damage_total_ci95"] for row in rows],
        out / "e3_damage.png",
        "externality damage (total)",
        "E3: externality ablation",
    )
    with_full = next(
        row for row in rows if row["condition"] == "with_externalities" and row["eval_env"] == "full"
    )
    without_full = next(
        row for row in rows if row["condition"] == "without_externalities" and row["eval_env"] == "full"
    )
    hypotheses = {
        "H3_cross_eval_damage": bool(
            without_full["damage_total_mean"] > with_full["damage_total_mean"]
        ),
        "with_externalities_full_damage": with_full["damage_total_mean"],
        "without_externalities_full_damage": without_full["damage_total_mean"],
        "with_externalities_share_agents_causing_damage": with_full[
            "share_agents_causing_damage_mean"
        ],
    }
    summary = {
        "experiment": "E3",
        "eval_seeds": eval_seeds,
        "rows": rows,
        "runs": run_records,
        "hypotheses": hypotheses,
    }
    write_json(out / "e3_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E4. Planning requirements
# ---------------------------------------------------------------------------


def e4_requirements(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_200_000, eval_seed_count)
    profiles = ["soft", "medium", "hard"]
    rows: list[dict] = []
    with _shared_executor() as shared_executor:
        for train_profile in profiles:
            train_city = replace(
                city_config, requirements=REQUIREMENT_PROFILES[train_profile]
            )
            per_eval_profile: dict[str, list[dict]] = {p: [] for p in profiles}
            for run_seed in seeds:
                print(f"  [E4] Training profile={train_profile}, seed={run_seed}...")
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, train_city, executor=shared_executor).run()
                layout = layout_from_config(evolution_config)
                for eval_profile in profiles:
                    eval_city = replace(
                        city_config, requirements=REQUIREMENT_PROFILES[eval_profile]
                    )
                    summary = evaluate_homogeneous(
                        lambda: NeuralPolicy(result.best_genome, layout),
                        eval_city,
                        eval_seeds,
                    )
                    per_eval_profile[eval_profile].append(summary)
                print(f"  [E4] seed {run_seed} done.")
            for eval_profile in profiles:
                summaries = per_eval_profile[eval_profile]
                metrics = {
                    key: _run_metric_summary(summaries, key)
                    for key in (
                        "fitness", "profit", "city_quality", "mean_satisfaction",
                        "green_share", "type_entropy", "accessibility",
                        "violations", "damage_total", "built_cells",
                    )
                }
                rows.append(
                    {
                        "trained_profile": train_profile,
                        "eval_profile": eval_profile,
                        **_metric_row(metrics, list(metrics)),
                    }
                )
    write_csv(out / "e4_summary.csv", rows)
    diagonal = [row for row in rows if row["trained_profile"] == row["eval_profile"]]
    plot_profit_quality(
        [
            {
                "condition": row["trained_profile"],
                "profit_mean": row["profit_mean"],
                "city_quality_mean": row["city_quality_mean"],
            }
            for row in diagonal
        ],
        out / "e4_profit_quality.png",
        "E4: profit vs city quality by requirement profile",
    )
    by_profile = {row["trained_profile"]: row for row in diagonal}
    profit = {name: by_profile[name]["profit_mean"] for name in profiles}
    quality = {name: by_profile[name]["city_quality_mean"] for name in profiles}
    hypotheses = {
        "H2_profit_decreases": bool(
            profit["soft"] >= profit["medium"] >= profit["hard"]
        ),
        "H2_quality_increases": bool(
            quality["soft"] <= quality["medium"] <= quality["hard"]
        ),
        "H2_both_fall_at_hard": bool(
            profit["hard"] < profit["medium"] and quality["hard"] < quality["medium"]
        ),
        "profit": profit,
        "city_quality": quality,
    }
    summary = {
        "experiment": "E4",
        "eval_seeds": eval_seeds,
        "rows": rows,
        "hypotheses": hypotheses,
    }
    write_json(out / "e4_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E5. Number of agents
# ---------------------------------------------------------------------------


def e5_agents(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    n_values: Sequence[int] = (5, 10, 15),
    grid: tuple[int, int] = (12, 12),
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_300_000, eval_seed_count)
    rows: list[dict] = []
    with _shared_executor() as shared_executor:
        for n_agents in n_values:
            sweep_city = replace(
                city_config,
                n_agents=int(n_agents),
                height=int(grid[0]),
                width=int(grid[1]),
            )
            summaries: list[dict] = []
            layout = layout_from_config(evolution_config)
            for run_seed in seeds:
                cfg = _ensure_evolution_feasible(
                    _run_config(evolution_config, run_seed), int(n_agents)
                )
                result = EvolutionEngine(cfg, sweep_city, executor=shared_executor).run()
                summaries.append(
                    evaluate_homogeneous(
                        lambda: NeuralPolicy(result.best_genome, layout),
                        sweep_city,
                        eval_seeds,
                    )
                )
            metrics = {
                key: _run_metric_summary(summaries, key)
                for key in (
                    "fitness", "profit", "city_quality", "mean_satisfaction",
                    "damage_total", "damage_per_agent", "share_agents_causing_damage",
                    "violations", "built_cells",
                )
            }
            built = max(1.0, metrics["built_cells"]["mean"])
            rows.append(
                {
                    "n_agents": int(n_agents),
                    "grid": f"{grid[0]}x{grid[1]}",
                    **_metric_row(metrics, list(metrics)),
                    "damage_per_built": metrics["damage_total"]["mean"] / built,
                }
            )
    write_csv(out / "e5_summary.csv", rows)
    plot_metric_vs_x(
        [row["n_agents"] for row in rows],
        {
            "damage total": (
                [row["damage_total_mean"] for row in rows],
                [row["damage_total_ci95"] for row in rows],
            ),
            "damage per agent": (
                [row["damage_per_agent_mean"] for row in rows],
                [row["damage_per_agent_ci95"] for row in rows],
            ),
        },
        out / "e5_damage_vs_agents.png",
        "number of agents",
        "externality damage",
        "E5: damage vs number of agents",
    )
    n_arr = np.array([row["n_agents"] for row in rows], dtype=np.float64)
    damage_arr = np.array([row["damage_total_mean"] for row in rows], dtype=np.float64)
    per_built_arr = np.array([row["damage_per_built"] for row in rows], dtype=np.float64)
    corr_total = float(np.corrcoef(n_arr, damage_arr)[0, 1]) if len(rows) > 1 else 0.0
    corr_per_built = float(np.corrcoef(n_arr, per_built_arr)[0, 1]) if len(rows) > 1 else 0.0
    hypotheses = {
        "H3_damage_grows_with_agents": bool(corr_total > 0),
        "H3_damage_per_built_grows": bool(corr_per_built > 0),
        "correlation_total": corr_total,
        "correlation_per_built": corr_per_built,
        "note": "E5 uses a larger fixed grid to avoid saturation by total builds.",
    }
    summary = {
        "experiment": "E5",
        "eval_seeds": eval_seeds,
        "rows": rows,
        "hypotheses": hypotheses,
    }
    write_json(out / "e5_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E6. Generalisation to held-out seeds
# ---------------------------------------------------------------------------


def e6_generalization(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
    episodes_per_seed: int | None = None,
) -> dict:
    out = ensure_dir(out_dir)
    rows: list[dict] = []
    run_records: list[dict] = []
    with _shared_executor() as shared_executor:
        for run_seed in seeds:
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            validation_seeds = _eval_seeds(cfg.validation_seed_base, eval_seed_count)
            test_seeds = _eval_seeds(cfg.test_seed_base, eval_seed_count)
            validation = evaluate_population(
                result.population, city_config, cfg, validation_seeds,
                episodes_per_seed=episodes_per_seed,
            )
            test = evaluate_population(
                result.population, city_config, cfg, test_seeds,
                episodes_per_seed=episodes_per_seed,
            )
            layout = layout_from_config(evolution_config)
            best_test = evaluate_homogeneous(
                lambda: NeuralPolicy(result.best_genome, layout),
                city_config,
                test_seeds,
            )
            run_records.append(
                {
                    "seed": int(run_seed),
                    "train_mean": float(result.history[-1]["mean"]),
                    "train_best": float(result.history[-1]["best"]),
                    "validation_mean": validation["fitness"]["mean"],
                    "test_mean": test["fitness"]["mean"],
                    "test_diversity": test["diversity"],
                    "best_homogeneous_test": _summary_value(best_test, "fitness"),
                }
            )
    for condition, key in [
        ("train", "train_mean"),
        ("validation", "validation_mean"),
        ("test", "test_mean"),
    ]:
        stats = _stats([r[key] for r in run_records])
        rows.append(
            {
                "condition": condition,
                "fitness_mean": stats["mean"],
                "fitness_ci95": stats["ci95"],
            }
        )
    write_csv(out / "e6_summary.csv", rows)
    plot_bars(
        [row["condition"] for row in rows],
        [row["fitness_mean"] for row in rows],
        [row["fitness_ci95"] for row in rows],
        out / "e6_generalization.png",
        "mean population fitness",
        "E6: train vs held-out seeds",
    )
    summary = {
        "experiment": "E6",
        "rows": rows,
        "runs": run_records,
        "generalization_gap_test_minus_train": rows[2]["fitness_mean"]
        - rows[0]["fitness_mean"],
    }
    write_json(out / "e6_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E7. Fitness sharing ablation
# ---------------------------------------------------------------------------


def e7_diversity(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
) -> dict:
    out = ensure_dir(out_dir)
    rows: list[dict] = []
    curves: dict[str, list[list[dict]]] = {}
    with _shared_executor() as shared_executor:
        for condition, use_sharing in [("sharing_on", True), ("sharing_off", False)]:
            histories: list[list[dict]] = []
            best_values = []
            diversity_values = []
            for run_seed in seeds:
                cfg = replace(
                    _run_config(evolution_config, run_seed),
                    use_fitness_sharing=use_sharing,
                )
                result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
                histories.append(result.history)
                best_values.append(result.best_fitness)
                diversity_values.append(float(result.history[-1]["diversity"]))
            curves[condition] = histories
            rows.append(
                {
                    "condition": condition,
                    "best_fitness_mean": _stats(best_values)["mean"],
                    "best_fitness_ci95": _stats(best_values)["ci95"],
                    "final_diversity_mean": _stats(diversity_values)["mean"],
                    "final_diversity_ci95": _stats(diversity_values)["ci95"],
                }
            )
            plot_learning_curve(
                histories,
                out / f"e7_{condition}_learning_curve.png",
                title=f"E7: {condition}",
            )
    write_csv(out / "e7_summary.csv", rows)
    gens = list(range(max(len(h) for hs in curves.values() for h in hs)))
    series = {}
    for condition, histories in curves.items():
        values = np.full((len(histories), len(gens)), np.nan)
        for i, hist in enumerate(histories):
            for row in hist:
                values[i, int(row["gen"])] = row["diversity"]
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(means)
        series[condition] = (means, 1.96 * stds / np.sqrt(max(1, len(histories))))
    plot_metric_vs_x(
        gens,
        series,
        out / "e7_diversity.png",
        "generation",
        "mean pairwise behaviour distance",
        "E7: fitness sharing ablation",
    )
    on = next(row for row in rows if row["condition"] == "sharing_on")
    off = next(row for row in rows if row["condition"] == "sharing_off")
    hypotheses = {
        "H_diversity_on_greater": bool(
            on["final_diversity_mean"] > off["final_diversity_mean"]
        ),
        "sharing_on": on,
        "sharing_off": off,
    }
    summary = {
        "experiment": "E7",
        "rows": rows,
        "hypotheses": hypotheses,
    }
    write_json(out / "e7_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E8. Emergent districts
# ---------------------------------------------------------------------------


def e8_districts(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_500_000, eval_seed_count)
    cfg_base = replace(evolution_config, track_spatial=True, spatial_every=20)
    layout = layout_from_config(cfg_base)
    rows: list[dict] = []
    histories: list[list[dict]] = []
    with _shared_executor() as shared_executor:
        for run_seed in seeds:
            cfg = _run_config(cfg_base, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            histories.append(result.history)
            for eval_seed in eval_seeds:
                policies = [
                    NeuralPolicy(result.best_genome, layout)
                    for _ in range(city_config.n_agents)
                ]
                episode = run_episode(city_config, policies, int(eval_seed))
                spatial = spatial_metrics(episode.type_grid)
                rows.append(
                    {
                        "seed": int(run_seed),
                        "eval_seed": int(eval_seed),
                        "city_quality": episode.city.city_quality,
                        "mean_satisfaction": episode.city.mean_satisfaction,
                        "green_share": episode.city.green_share,
                        **spatial,
                    }
                )
            print(f"  [E8] seed {run_seed} done.")
    write_csv(out / "e8_summary.csv", rows)
    metrics = {
        key: _stats([row[key] for row in rows])
        for key in ("morans_i", "local_entropy", "cluster_count", "green_share", "city_quality", "mean_satisfaction")
    }
    aggregate = _aggregate_history(
        histories, ["morans_i", "local_entropy", "cluster_count", "snapshot_satisfaction"]
    )
    write_csv(out / "e8_history.csv", aggregate)
    gens = [row["gen"] for row in aggregate]
    plot_metric_vs_x(
        gens,
        {
            "Moran's I": (
                [row["morans_i_mean"] for row in aggregate],
                [row["morans_i_ci95"] for row in aggregate],
            ),
            "local entropy": (
                [row["local_entropy_mean"] for row in aggregate],
                [row["local_entropy_ci95"] for row in aggregate],
            ),
        },
        out / "e8_spatial.png",
        "generation",
        "spatial structure",
        "E8: emergent districts",
    )
    summary = {
        "experiment": "E8",
        "eval_seeds": eval_seeds,
        "metrics": metrics,
        "rows": rows,
        "hypothesis": {
            "H4_clustering": bool(metrics["morans_i"]["mean"] > 0.10),
            "morans_i": metrics["morans_i"]["mean"],
            "local_entropy": metrics["local_entropy"]["mean"],
        },
    }
    write_json(out / "e8_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E9. Path dependence
# ---------------------------------------------------------------------------


def e9_path_dependence(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    paths: int = 10,
) -> dict:
    out = ensure_dir(out_dir)
    if not seeds:
        raise ValueError("E9 needs at least one training seed")
    train_seed = int(seeds[0])
    cfg = _run_config(evolution_config, train_seed)
    result = EvolutionEngine(cfg, city_config).run()
    layout = layout_from_config(evolution_config)
    city_seed = 9_400_000
    distributions = []
    grids = []
    rows = []
    for index in range(paths):
        rng_seed = int(seeds[index % len(seeds)]) * 1000 + index + 1
        policies = [
            NeuralPolicy(result.best_genome, layout)
            for _ in range(city_config.n_agents)
        ]
        episode = run_episode(
            city_config, policies, city_seed, rng_seed=rng_seed
        )
        distribution = type_distribution(episode.type_grid)
        distributions.append(distribution)
        grids.append(episode.type_grid)
        rows.append(
            {
                "path": index,
                "rng_seed": rng_seed,
                "city_quality": episode.city.city_quality,
                "mean_satisfaction": episode.city.mean_satisfaction,
                "type_distribution": "|".join(f"{x:.4f}" for x in distribution),
            }
        )
    taus = []
    distances = []
    for i in range(paths):
        for j in range(i + 1, paths):
            taus.append(kendall_tau(distributions[i], distributions[j]))
            distances.append(grid_distance(grids[i], grids[j]))
    unique_grids = len({g.tobytes() for g in grids})
    np.savez(
        out / "e9_grids.npz",
        **{f"path_{i}": grid for i, grid in enumerate(grids)},
    )
    matrix = np.array(distributions)
    plot_distribution_matrix(
        matrix,
        [f"path {i}" for i in range(paths)],
        out / "e9_distributions.png",
        "E9: final type distribution by path",
    )
    write_csv(out / "e9_summary.csv", rows)
    summary = {
        "experiment": "E9",
        "train_seed": train_seed,
        "city_seed": city_seed,
        "paths": paths,
        "kendall_tau": _stats(taus),
        "grid_distance": _stats(distances),
        "unique_final_grids": unique_grids,
        "hypothesis": {
            "H5_path_dependence": bool(
                _stats(taus)["mean"] < 0.90 or _stats(distances)["mean"] > 0.20
            ),
        },
        "rows": rows,
    }
    write_json(out / "e9_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E10. Regulation
# ---------------------------------------------------------------------------


def e10_regulation(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_600_000, eval_seed_count)
    scenarios = ["no_tax", "property_tax_5", "industry_tax_10", "green_subsidy", "affordable_quota"]
    rows: list[dict] = []
    with _shared_executor() as shared_executor:
        for scenario in scenarios:
            regulation = REGULATION_SCENARIOS[scenario]
            train_city = replace(city_config, regulation=regulation)
            summaries = []
            for run_seed in seeds:
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, train_city, executor=shared_executor).run()
                layout = layout_from_config(evolution_config)
                summaries.append(
                    evaluate_homogeneous(
                        lambda: NeuralPolicy(result.best_genome, layout),
                        train_city,
                        eval_seeds,
                    )
                )
            metrics = {
                key: _run_metric_summary(summaries, key)
                for key in (
                    "fitness", "profit", "city_quality", "mean_satisfaction",
                    "green_share", "affordable_share", "tax_revenue",
                    "subsidy_total", "violations", "built_cells",
                )
            }
            rows.append(
                {
                    "condition": scenario,
                    **_metric_row(metrics, list(metrics)),
                }
            )
            print(f"  [E10] scenario {scenario} done.")
    write_csv(out / "e10_summary.csv", rows)
    plot_profit_satisfaction(
        rows, out / "e10_profit_satisfaction.png", "E10: regulation scenarios"
    )
    plot_bars(
        [row["condition"] for row in rows],
        [row["green_share_mean"] for row in rows],
        [row["green_share_ci95"] for row in rows],
        out / "e10_green_share.png",
        "green share",
        "E10: green share by regulation",
    )
    summary = {
        "experiment": "E10",
        "rows": rows,
        "hypothesis": {
            "green_subsidy_raises_green": bool(
                next(r for r in rows if r["condition"] == "green_subsidy")["green_share_mean"]
                > next(r for r in rows if r["condition"] == "no_tax")["green_share_mean"]
            ),
            "taxes_reduce_profit": bool(
                next(r for r in rows if r["condition"] == "property_tax_5")["profit_mean"]
                < next(r for r in rows if r["condition"] == "no_tax")["profit_mean"]
            ),
        },
    }
    write_json(out / "e10_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# E11. Shocks and resilience
# ---------------------------------------------------------------------------


def e11_shocks(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_700_000, eval_seed_count)
    shock_round = max(1, city_config.actions_per_agent // 2)
    shock_specs = [
        ("no_shock", []),
        ("industry_removal", [{"round": shock_round, "kind": "industry_removal", "magnitude": 0.5}]),
        ("population_boom", [{"round": shock_round, "kind": "population_boom", "magnitude": 200}]),
        ("earthquake", [{"round": shock_round, "kind": "earthquake", "magnitude": 0.20}]),
    ]
    layout = layout_from_config(evolution_config)
    neural_genomes = []
    with _shared_executor() as shared_executor:
        for run_seed in seeds:
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            neural_genomes.append(result.best_genome)
    conditions: dict[str, Callable[[], object]] = {
        "neural": lambda: NeuralPolicy(neural_genomes[0], layout),
        "greedy": lambda: GreedyPolicy(),
        "manual": lambda: ManualPolicy(),
    }
    # For neural, average over all trained genomes by running each genome on a
    # subset of evaluation seeds; baselines use the same seeds.
    rows: list[dict] = []
    for condition, factory in conditions.items():
        for shock_name, shocks in shock_specs:
            metrics = []
            for i, eval_seed in enumerate(eval_seeds):
                if condition == "neural":
                    genome = neural_genomes[i % len(neural_genomes)]
                    policies = [
                        NeuralPolicy(genome, layout)
                        for _ in range(city_config.n_agents)
                    ]
                else:
                    policies = [factory() for _ in range(city_config.n_agents)]
                episode = run_episode(
                    city_config,
                    policies,
                    int(eval_seed),
                    record_history=True,
                    shocks=shocks,
                )
                resilience = resilience_metrics(episode.history, shock_round)
                metrics.append(
                    {
                        "recovery": resilience["recovery"],
                        "drop": resilience["drop"],
                        "final_quality": episode.city.city_quality,
                        "final_satisfaction": episode.city.mean_satisfaction,
                        "final_profit": float(np.mean([a.profit for a in episode.agents])),
                    }
                )
            rows.append(
                {
                    "condition": condition,
                    "shock": shock_name,
                    "recovery_mean": _stats([m["recovery"] for m in metrics])["mean"],
                    "recovery_ci95": _stats([m["recovery"] for m in metrics])["ci95"],
                    "drop_mean": _stats([m["drop"] for m in metrics])["mean"],
                    "final_quality_mean": _stats([m["final_quality"] for m in metrics])["mean"],
                    "final_satisfaction_mean": _stats([m["final_satisfaction"] for m in metrics])["mean"],
                    "final_profit_mean": _stats([m["final_profit"] for m in metrics])["mean"],
                }
            )
    write_csv(out / "e11_summary.csv", rows)
    labels = [f"{row['condition']}\n{row['shock']}" for row in rows]
    plot_bars(
        labels,
        [row["recovery_mean"] for row in rows],
        [row["recovery_ci95"] for row in rows],
        out / "e11_recovery.png",
        "recovery ratio",
        "E11: resilience to shocks",
    )
    summary = {
        "experiment": "E11",
        "shock_round": shock_round,
        "rows": rows,
        "hypothesis": {
            "H6_neural_more_resilient": bool(
                next(r for r in rows if r["condition"] == "neural" and r["shock"] == "earthquake")["recovery_mean"]
                > next(r for r in rows if r["condition"] == "greedy" and r["shock"] == "earthquake")["recovery_mean"]
            ),
        },
    }
    write_json(out / "e11_summary.json", summary)
    return summary
