"""Learning-curve and policy-baseline experiments (E1-E2)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import CityConfig, EvolutionConfig
from .evolution import (
    EvolutionEngine,
    evaluate_homogeneous,
    evaluate_league,
    layout_from_config,
    mean_pairwise_distance,
    unique_behaviors,
)
from .experiment_support import (
    _aggregate_history,
    _eval_seeds,
    _require_training_seeds,
    _run_evolution,
    _run_config,
    _run_metric_summary,
    _stats,
    _summary_value,
)
from .io import ensure_dir, save_genome, write_csv, write_json
from .plotting import plot_bars, plot_learning_curve, plot_satisfaction_curve
from .policy import (
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    manual_variants,
    random_genome,
)


def e1_learning_curve(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    import time

    _require_training_seeds(seeds, "E1")
    t0 = time.perf_counter()
    del eval_seed_count
    out = ensure_dir(out_dir)
    histories: list[list[dict]] = []
    runs: list[dict] = []
    for run_seed in seeds:
        print(f"  [E1] Training neuroevolution, seed={run_seed}...")
        cfg = _run_config(evolution_config, run_seed)
        result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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
    print(f"  [E1] done in {time.perf_counter()-t0:.2f}s")
    return summary


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
    _require_training_seeds(seeds, "E2")
    if behavior_variants < 1:
        raise ValueError("behavior_variants must be at least one")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_000_000, eval_seed_count)
    layout = layout_from_config(evolution_config)
    neural_summaries: list[dict] = []
    neural_run_records: list[dict] = []
    evolved_behaviors: list[np.ndarray] = []

    for run_seed in seeds:
            print(f"  [E2] Training neuroevolution, seed={run_seed}...")
            cfg = _run_config(evolution_config, run_seed)
            result = _run_evolution(cfg, city_config, engine_factory=EvolutionEngine)
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
