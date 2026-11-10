"""Environment, constraint, scale, and diversity experiments (E3-E8)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CityConfig, EvolutionConfig, REQUIREMENT_PROFILES
from .episode import run_episode
from .evolution import (
    EvolutionEngine,
    evaluate_homogeneous,
    evaluate_population,
    layout_from_config,
)
from .experiment_support import (
    _aggregate_history,
    _ensure_evolution_feasible,
    _eval_seeds,
    _metric_row,
    _require_training_seeds,
    _run_evolution,
    _run_config,
    _run_metric_summary,
    _stats,
    _summary_value,
)
from .io import ensure_dir, write_csv, write_json
from .metrics import spatial_metrics
from .plotting import (
    plot_bars,
    plot_learning_curve,
    plot_metric_vs_x,
    plot_profit_quality,
)
from .policy import NeuralPolicy


def e3_externalities(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E3")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_100_000, eval_seed_count)
    full_city = replace(city_config, enable_externalities=True)
    conditions = [
        ("with_externalities", True),
        ("without_externalities", False),
    ]
    rows: list[dict] = []
    run_records: list[dict] = []
    for condition, enabled in conditions:
            train_city = replace(city_config, enable_externalities=enabled)
            matching_summaries: list[dict] = []
            full_summaries: list[dict] = []
            for run_seed in seeds:
                print(f"  [E3] Training neuroevolution, seed={run_seed}...")
                cfg = _run_config(evolution_config, run_seed)
                result = _run_evolution(cfg, train_city, engine_factory=EvolutionEngine)
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


def e4_requirements(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E4")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_200_000, eval_seed_count)
    profiles = ["soft", "medium", "hard"]
    rows: list[dict] = []
    for train_profile in profiles:
        train_city = replace(
            city_config, requirements=REQUIREMENT_PROFILES[train_profile]
        )
        per_eval_profile: dict[str, list[dict]] = {p: [] for p in profiles}
        for run_seed in seeds:
            print(f"  [E4] Training profile={train_profile}, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = _run_evolution(cfg, train_city, engine_factory=EvolutionEngine)
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


def e5_agents(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    n_values: Sequence[int] = (5, 10, 15),
    grid: tuple[int, int] = (12, 12),
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E5")
    if len(n_values) == 0:
        raise ValueError("E5 needs at least one agent count")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_300_000, eval_seed_count)
    rows: list[dict] = []
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
            result = _run_evolution(cfg, sweep_city, engine_factory=EvolutionEngine)
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


def e6_generalization(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
    episodes_per_seed: int | None = None,
) -> dict:
    _require_training_seeds(seeds, "E6")
    out = ensure_dir(out_dir)
    rows: list[dict] = []
    run_records: list[dict] = []
    for run_seed in seeds:
        cfg = _run_config(evolution_config, run_seed)
        result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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


def e7_diversity(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
) -> dict:
    _require_training_seeds(seeds, "E7")
    out = ensure_dir(out_dir)
    rows: list[dict] = []
    curves: dict[str, list[list[dict]]] = {}
    for condition, use_sharing in [("sharing_on", True), ("sharing_off", False)]:
        histories: list[list[dict]] = []
        best_values = []
        diversity_values = []
        for run_seed in seeds:
            cfg = replace(
                _run_config(evolution_config, run_seed),
                use_fitness_sharing=use_sharing,
            )
            result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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


def e8_districts(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    _require_training_seeds(seeds, "E8")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_500_000, eval_seed_count)
    cfg_base = replace(evolution_config, track_spatial=True, spatial_every=20)
    layout = layout_from_config(cfg_base)
    rows: list[dict] = []
    histories: list[list[dict]] = []
    for run_seed in seeds:
        cfg = _run_config(cfg_base, run_seed)
        result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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
