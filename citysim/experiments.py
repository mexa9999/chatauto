"""Experiment runners E1-E7 and shared aggregation helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

from .config import (
    CityConfig,
    EvolutionConfig,
    REQUIREMENT_PROFILES,
    quick_city_config,
    quick_evolution_config,
    seed_bases_for_run,
)
from .evolution import (
    EvolutionEngine,
    EvolutionResult,
    evaluate_homogeneous,
    evaluate_league,
    evaluate_population,
    mean_pairwise_distance,
    unique_behaviors,
)
from .io import ensure_dir, save_genome, write_csv, write_json
from .plotting import (
    plot_bars,
    plot_learning_curve,
    plot_metric_vs_x,
    plot_profit_quality,
)
from .policy import (
    DEFAULT_LAYOUT,
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


def _stats(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
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
        write_json(
            out / f"{tag}_summary.json",
            {
                "best_fitness": result.best_fitness,
                "best_generation": result.best_generation,
                "best_behavior": result.best_behavior,
                "evolution_config": evolution_config.as_dict(),
                "city_config": city_config.as_dict(),
            },
        )
        plot_learning_curve([result.history], out / f"{tag}_learning_curve.png")
    return result


def _aggregate_history(histories: list[list[dict]], keys: Sequence[str]) -> list[dict]:
    max_len = max(len(h) for h in histories)
    rows = []
    for gen in range(max_len):
        row: dict = {"gen": gen}
        for key in keys:
            values = [float(h[gen][key]) for h in histories if gen < len(h)]
            stats = _stats(values)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_ci95"] = stats["ci95"]
        rows.append(row)
    return rows


def _summary_value(summary: dict, key: str) -> float:
    return float(summary[key]["mean"])


def _run_metric_summary(summaries: Sequence[dict], key: str) -> dict[str, float]:
    return _stats([_summary_value(summary, key) for summary in summaries])


def _eval_seeds(base: int, count: int) -> list[int]:
    return [int(base) + i for i in range(count)]


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
    del eval_seed_count  # E1 only needs training curves
    out = ensure_dir(out_dir)
    histories: list[list[dict]] = []
    runs: list[dict] = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for run_seed in seeds:
            print(f"  [E1] Training neuroevolution, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            histories.append(result.history)
            save_genome(out / f"best_genome_seed{run_seed}.npy", result.best_genome)
            best_curve = np.array([row["best"] for row in result.history], dtype=np.float64)
            final_best = float(best_curve[-1])
            threshold = 0.9 * final_best if final_best > -1e17 else np.inf
            reached = int(np.argmax(best_curve >= threshold)) if np.isfinite(threshold) else -1
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
            print(f"  [E1] seed {run_seed} training done. Evaluating...")
    aggregate = _aggregate_history(
        histories,
        ["best", "mean", "std", "diversity", "unique_behaviors", "sigma", "best_ever"],
    )
    write_csv(out / "e1_history.csv", aggregate)
    plot_learning_curve(histories, out / "e1_learning_curve.png")
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
    test_base = 9_000_000
    eval_seeds = _eval_seeds(test_base, eval_seed_count)
    neural_summaries: list[dict] = []
    neural_run_records: list[dict] = []
    evolved_behaviors: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for run_seed in seeds:
            print(f"  [E2] Training neuroevolution, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            save_genome(out / f"best_genome_seed{run_seed}.npy", result.best_genome)
            layout = DEFAULT_LAYOUT if evolution_config.hidden_size == 64 else result_layout_from_config(evolution_config)
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
            behavior_matrix = np.array([ind.behavior for ind in result.population], dtype=np.float64)
            evolved_behaviors.append(behavior_matrix)
            neural_run_records.append(
                {
                    "seed": int(run_seed),
                    "best_fitness": result.best_fitness,
                    "league_fitness": _summary_value(summary, "fitness"),
                    "homogeneous_fitness": _summary_value(homogeneous, "fitness"),
                    "eval_profit": _summary_value(summary, "profit"),
                    "eval_city_quality": _summary_value(summary, "city_quality"),
                    "eval_damage": _summary_value(summary, "damage_total"),
                    "eval_violations": _summary_value(summary, "violations"),
                }
            )
            print(f"  [E2] seed {run_seed} training done. Evaluating...")

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
    random_layout = DEFAULT_LAYOUT if evolution_config.hidden_size == 64 else result_layout_from_config(evolution_config)
    random_set = _evaluate_behavior_set(
        [
            lambda g=random_genome(rng, random_layout): NeuralPolicy(g, random_layout)
            for _ in range(behavior_variants)
        ],
        city_config,
        behavior_seeds,
    )
    evolved_diversity = [mean_pairwise_distance(b) for b in evolved_behaviors]
    evolved_unique = [unique_behaviors(b) for b in evolved_behaviors]

    rows = []
    neural_fitness = _run_metric_summary(neural_summaries, "fitness")
    neural_profit = _run_metric_summary(neural_summaries, "profit")
    neural_quality = _run_metric_summary(neural_summaries, "city_quality")
    neural_damage = _run_metric_summary(neural_summaries, "damage_total")
    neural_violations = _run_metric_summary(neural_summaries, "violations")
    neural_homogeneous = _stats([r["homogeneous_fitness"] for r in neural_run_records])
    rows.append(
        {
            "condition": "neuroevolution",
            "fitness_mean": neural_fitness["mean"],
            "fitness_ci95": neural_fitness["ci95"],
            "homogeneous_fitness_mean": neural_homogeneous["mean"],
            "profit_mean": neural_profit["mean"],
            "city_quality_mean": neural_quality["mean"],
            "damage_total_mean": neural_damage["mean"],
            "violations_mean": neural_violations["mean"],
            "behavior_diversity": float(np.mean(evolved_diversity)),
            "unique_behaviors": float(np.mean(evolved_unique)),
        }
    )
    for name, summary in baseline_summaries.items():
        rows.append(
            {
                "condition": name,
                "fitness_mean": _summary_value(summary, "fitness"),
                "fitness_ci95": summary["fitness"]["ci95"],
                "homogeneous_fitness_mean": _summary_value(baseline_homogeneous[name], "fitness"),
                "profit_mean": _summary_value(summary, "profit"),
                "city_quality_mean": _summary_value(summary, "city_quality"),
                "damage_total_mean": _summary_value(summary, "damage_total"),
                "violations_mean": _summary_value(summary, "violations"),
                "behavior_diversity": 0.0 if name == "greedy" else float(
                    mean_pairwise_distance(manual_set if name == "manual" else random_set)
                ),
                "unique_behaviors": 0.0 if name == "greedy" else float(
                    unique_behaviors(manual_set if name == "manual" else random_set)
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


def result_layout_from_config(evolution_config: EvolutionConfig):
    from .policy import GenomeLayout

    return GenomeLayout(hidden_size=evolution_config.hidden_size)


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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for condition, enabled in conditions:
            train_city = replace(city_config, enable_externalities=enabled)
            matching_summaries: list[dict] = []
            full_summaries: list[dict] = []
            for run_seed in seeds:
                print(f"  [E3] Training neuroevolution, seed={run_seed}...")
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, train_city, executor=shared_executor).run()
                layout = result_layout_from_config(evolution_config)
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
                print(f"  [E3] seed {run_seed} training done. Evaluating...")
            for eval_name, summaries in [
                ("matching", matching_summaries),
                ("full", full_summaries),
            ]:
                metrics = {
                    key: _run_metric_summary(summaries, key)
                    for key in (
                        "fitness",
                        "profit",
                        "city_quality",
                        "damage_total",
                        "damage_per_agent",
                        "share_agents_causing_damage",
                        "violations",
                        "built_cells",
                    )
                }
                rows.append(
                    {
                        "condition": condition,
                        "eval_env": eval_name,
                        **{f"{key}_mean": value["mean"] for key, value in metrics.items()},
                        **{f"{key}_ci95": value["ci95"] for key, value in metrics.items()},
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for train_profile in profiles:
            train_city = replace(
                city_config, requirements=REQUIREMENT_PROFILES[train_profile]
            )
            per_eval_profile: dict[str, list[dict]] = {p: [] for p in profiles}
            for run_seed in seeds:
                print(f"  [E4] Training profile={train_profile}, seed={run_seed}...")
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, train_city, executor=shared_executor).run()
                layout = result_layout_from_config(evolution_config)
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
                print(f"  [E4] seed {run_seed} done. Evaluated.")
            for eval_profile in profiles:
                summaries = per_eval_profile[eval_profile]
                metrics = {
                    key: _run_metric_summary(summaries, key)
                    for key in (
                        "fitness",
                        "profit",
                        "city_quality",
                        "green_share",
                        "type_entropy",
                        "accessibility",
                        "violations",
                        "damage_total",
                        "built_cells",
                    )
                }
                rows.append(
                    {
                        "trained_profile": train_profile,
                        "eval_profile": eval_profile,
                        **{f"{key}_mean": value["mean"] for key, value in metrics.items()},
                        **{f"{key}_ci95": value["ci95"] for key, value in metrics.items()},
                    }
                )
    write_csv(out / "e4_summary.csv", rows)
    diagonal = [
        row for row in rows if row["trained_profile"] == row["eval_profile"]
    ]
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
    profit_soft = by_profile["soft"]["profit_mean"]
    profit_medium = by_profile["medium"]["profit_mean"]
    profit_hard = by_profile["hard"]["profit_mean"]
    quality_soft = by_profile["soft"]["city_quality_mean"]
    quality_medium = by_profile["medium"]["city_quality_mean"]
    quality_hard = by_profile["hard"]["city_quality_mean"]
    hypotheses = {
        "H2_profit_decreases": bool(profit_soft >= profit_medium >= profit_hard),
        "H2_quality_increases": bool(quality_soft <= quality_medium <= quality_hard),
        "H2_both_fall_at_hard": bool(
            profit_hard < profit_medium and quality_hard < quality_medium
        ),
        "profit": {"soft": profit_soft, "medium": profit_medium, "hard": profit_hard},
        "city_quality": {
            "soft": quality_soft,
            "medium": quality_medium,
            "hard": quality_hard,
        },
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for n_agents in n_values:
            sweep_city = replace(
                city_config,
                n_agents=int(n_agents),
                height=int(grid[0]),
                width=int(grid[1]),
            )
            summaries: list[dict] = []
            for run_seed in seeds:
                cfg = _run_config(evolution_config, run_seed)
                result = EvolutionEngine(cfg, sweep_city, executor=shared_executor).run()
                layout = result_layout_from_config(evolution_config)
                summary = evaluate_homogeneous(
                    lambda: NeuralPolicy(result.best_genome, layout),
                    sweep_city,
                    eval_seeds,
                )
                summaries.append(summary)
            metrics = {
                key: _run_metric_summary(summaries, key)
                for key in (
                    "fitness",
                    "profit",
                    "city_quality",
                    "damage_total",
                    "damage_per_agent",
                    "share_agents_causing_damage",
                    "violations",
                    "built_cells",
                )
            }
            built = max(1.0, metrics["built_cells"]["mean"])
            rows.append(
                {
                    "n_agents": int(n_agents),
                    "grid": f"{grid[0]}x{grid[1]}",
                    **{f"{key}_mean": value["mean"] for key, value in metrics.items()},
                    **{f"{key}_ci95": value["ci95"] for key, value in metrics.items()},
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
        for run_seed in seeds:
            cfg = _run_config(evolution_config, run_seed)
            result = EvolutionEngine(cfg, city_config, executor=shared_executor).run()
            validation_seeds = _eval_seeds(cfg.validation_seed_base, eval_seed_count)
            test_seeds = _eval_seeds(cfg.test_seed_base, eval_seed_count)
            validation = evaluate_population(
                result.population,
                city_config,
                cfg,
                validation_seeds,
                episodes_per_seed=episodes_per_seed,
            )
            test = evaluate_population(
                result.population,
                city_config,
                cfg,
                test_seeds,
                episodes_per_seed=episodes_per_seed,
            )
            layout = result_layout_from_config(evolution_config)
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
    rows.append(
        {
            "condition": "train",
            "fitness_mean": _stats([r["train_mean"] for r in run_records])["mean"],
            "fitness_ci95": _stats([r["train_mean"] for r in run_records])["ci95"],
        }
    )
    rows.append(
        {
            "condition": "validation",
            "fitness_mean": _stats([r["validation_mean"] for r in run_records])["mean"],
            "fitness_ci95": _stats([r["validation_mean"] for r in run_records])["ci95"],
        }
    )
    rows.append(
        {
            "condition": "test",
            "fitness_mean": _stats([r["test_mean"] for r in run_records])["mean"],
            "fitness_ci95": _stats([r["test_mean"] for r in run_records])["ci95"],
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as shared_executor:
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