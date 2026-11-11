"""Path-dependence, regulation, and resilience experiments (E9-E11)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import CityConfig, EvolutionConfig, REGULATION_SCENARIOS
from .episode import run_episode
from .evolution import EvolutionEngine, evaluate_homogeneous, layout_from_config
from .experiment_support import (
    _eval_seeds,
    _metric_row,
    _require_training_seeds,
    _run_evolution,
    _run_config,
    _run_metric_summary,
    _stats,
)
from .io import ensure_dir, write_csv, write_json
from .metrics import (
    grid_distance,
    kendall_tau,
    resilience_metrics,
    type_distribution,
)
from .plotting import plot_bars, plot_distribution_matrix, plot_profit_satisfaction
from .policy import GreedyPolicy, ManualPolicy, NeuralPolicy


def e9_path_dependence(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    paths: int = 10,
) -> dict:
    if len(seeds) == 0:
        raise ValueError("E9 needs at least one training seed")
    if paths < 2:
        raise ValueError("E9 needs at least two paths")
    out = ensure_dir(out_dir)
    train_seed = int(seeds[0])
    cfg = _run_config(evolution_config, train_seed)
    result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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


def e10_regulation(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E10")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_600_000, eval_seed_count)
    scenarios = ["no_tax", "property_tax_5", "industry_tax_10", "green_subsidy", "affordable_quota"]
    rows: list[dict] = []
    for scenario in scenarios:
            regulation = REGULATION_SCENARIOS[scenario]
            train_city = replace(city_config, regulation=regulation)
            summaries = []
            for run_seed in seeds:
                cfg = _run_config(evolution_config, run_seed)
                result = _run_evolution(cfg, train_city, engine_factory=EvolutionEngine)
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


def e11_shocks(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E11")
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
    for run_seed in seeds:
        cfg = _run_config(evolution_config, run_seed)
        result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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
                    include_automatic_shocks=False,
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
