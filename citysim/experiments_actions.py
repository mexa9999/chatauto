"""Factorial ablation for the expanded action space and district synergy (E12)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import CityConfig, EvolutionConfig
from .evolution import evaluate_league, layout_from_config
from .experiment_support import (
    _eval_seeds,
    _require_training_seeds,
    _run_config,
    _run_evolution,
    _stats,
    _summary_value,
)
from .io import ensure_dir, save_genome, write_csv, write_json
from .plotting import plot_bars
from .policy import GreedyPolicy, NeuralPolicy


_CONDITIONS = (
    ("legacy_actions_no_synergy", False, False),
    ("extended_actions_no_synergy", True, False),
    ("legacy_actions_synergy", False, True),
    ("extended_actions_synergy", True, True),
)


def e12_factorial_actions(
    evolution_config: EvolutionConfig,
    city_config: CityConfig,
    seeds: Sequence[int],
    out_dir: str | Path,
    eval_seed_count: int = 5,
) -> dict:
    """Compare old/new action factors and old/new district synergy on shared seeds."""
    _require_training_seeds(seeds, "E12")
    out = ensure_dir(out_dir)
    eval_seeds = _eval_seeds(9_120_000, eval_seed_count)
    layout = layout_from_config(evolution_config)
    rows: list[dict] = []

    for condition, extended_actions, district_synergy in _CONDITIONS:
        condition_city = replace(
            city_config,
            enable_extended_actions=extended_actions,
            enable_district_synergy=district_synergy,
        )
        for run_seed in seeds:
            print(f"  [E12] Training {condition}, seed={run_seed}...")
            result = _run_evolution(
                _run_config(evolution_config, int(run_seed)),
                condition_city,
            )
            save_genome(
                out / f"e12_{condition}_seed{run_seed}.npy",
                result.best_genome,
            )
            summary = evaluate_league(
                lambda genome=result.best_genome: NeuralPolicy(genome, layout),
                condition_city,
                eval_seeds,
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": int(run_seed),
                    "extended_actions": extended_actions,
                    "district_synergy": district_synergy,
                    "best_fitness": float(result.best_fitness),
                    "fitness": _summary_value(summary, "fitness"),
                    "profit": _summary_value(summary, "profit"),
                    "city_quality": _summary_value(summary, "city_quality"),
                    "mean_satisfaction": _summary_value(
                        summary, "mean_satisfaction"
                    ),
                }
            )

    greedy_city = replace(
        city_config,
        enable_extended_actions=True,
        enable_district_synergy=True,
    )
    greedy = evaluate_league(
        GreedyPolicy,
        greedy_city,
        eval_seeds,
    )
    rows.append(
        {
            "condition": "greedy_external_baseline",
            "seed": "",
            "extended_actions": False,
            "district_synergy": True,
            "best_fitness": "",
            "fitness": _summary_value(greedy, "fitness"),
            "profit": _summary_value(greedy, "profit"),
            "city_quality": _summary_value(greedy, "city_quality"),
            "mean_satisfaction": _summary_value(
                greedy, "mean_satisfaction"
            ),
        }
    )
    write_csv(out / "e12_factorial.csv", rows)

    grouped = {}
    for condition, _, _ in _CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        grouped[condition] = {
            key: _stats([float(row[key]) for row in condition_rows])
            for key in (
                "best_fitness",
                "fitness",
                "profit",
                "city_quality",
                "mean_satisfaction",
            )
        }
    greedy_row = rows[-1]
    grouped["greedy_external_baseline"] = {
        key: _stats([float(greedy_row[key])])
        for key in ("fitness", "profit", "city_quality", "mean_satisfaction")
    }

    labels = [condition for condition, _, _ in _CONDITIONS]
    labels.append("greedy")
    for metric, filename, ylabel in (
        ("fitness", "e12_fitness.png", "Mean fitness"),
        ("city_quality", "e12_city_quality.png", "City quality"),
    ):
        values = [grouped[label][metric]["mean"] for label in labels[:-1]]
        errors = [grouped[label][metric]["ci95"] for label in labels[:-1]]
        values.append(grouped["greedy_external_baseline"][metric]["mean"])
        errors.append(grouped["greedy_external_baseline"][metric]["ci95"])
        plot_bars(
            labels,
            values,
            errors,
            out / filename,
            ylabel,
            f"E12: {ylabel} by action/economy condition",
        )

    summary = {
        "experiment": "E12",
        "training_seeds": [int(seed) for seed in seeds],
        "evaluation_seeds": eval_seeds,
        "factorial_conditions": grouped,
        "greedy_external_baseline": grouped["greedy_external_baseline"],
        "notes": (
            "All four neural conditions share training and evaluation seeds. "
            "Greedy is evaluated unchanged in the expanded economy."
        ),
    }
    write_json(out / "e12_summary.json", summary)
    return summary
