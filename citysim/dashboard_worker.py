"""Worker routines for dashboard jobs that do not map to existing CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .checkpoint_io import read_checkpoint
from .evolution import evaluate_homogeneous, evaluate_league, layout_from_config
from .policy import NeuralPolicy
from .training_history import TrainingHistoryStore


def _flatten(value: dict, prefix: str = "") -> dict[str, object]:
    flattened = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, name))
        else:
            flattened[name] = item
    return flattened


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_compare(
    job_path: str | Path,
    result_path: str | Path,
    progress_path: str | Path,
    stop_path: str | Path,
) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    result_target = Path(result_path)
    progress_target = Path(progress_path)
    stop_target = Path(stop_path)
    store = TrainingHistoryStore(job["history_dir"])
    refs = (job["left_ref"], job["right_ref"])
    reference_ref = job["reference_ref"]
    reference_city, _ = store.config_for_branch(reference_ref)
    seeds = [
        int(job["seed_base"]) + index
        for index in range(int(job["seed_count"]))
    ]

    reports = {}
    settings = {}
    completed_steps = 0

    def stop_after_method() -> bool:
        if not stop_target.exists():
            return False
        _write_json(
            progress_target,
            {
                "state": "stopped",
                "phase": "comparison stopped after an evaluation",
                "generation": completed_steps,
                "generations": 2 * len(refs),
            },
        )
        return True

    for label, ref in zip(("left", "right"), refs):
        if stop_after_method():
            return
        commit_id = store.resolve_ref(ref)
        training_city, evolution_config = store.config_for_branch(ref)
        checkpoint = store.checkpoint_path(ref)
        _, arrays = read_checkpoint(checkpoint)
        genome = arrays["best_ever_genome"]
        layout = layout_from_config(evolution_config)
        factory = lambda genome=genome, layout=layout: NeuralPolicy(genome, layout)
        methods = {}
        for method, evaluator in (
            ("homogeneous", evaluate_homogeneous),
            ("league", evaluate_league),
        ):
            if stop_after_method():
                return
            _write_json(
                progress_target,
                {
                    "state": "running",
                    "phase": f"{label} snapshot {ref}: {method}",
                    "generation": 0,
                    "generations": 2 * len(refs),
                },
            )
            methods[method] = evaluator(factory, reference_city, seeds)
            completed_steps += 1
        reports[label] = {
            "ref": ref,
            "commit_id": commit_id,
            "evolution_config": evolution_config.as_dict(),
            "training_city_config": training_city.as_dict(),
            "methods": methods,
        }
        settings[label] = {
            "evolution": _flatten(evolution_config.as_dict()),
            "city": _flatten(training_city.as_dict()),
        }

    setting_differences = {}
    for category in ("evolution", "city"):
        left_values = settings["left"][category]
        right_values = settings["right"][category]
        setting_differences[category] = {
            key: {"left": left_values.get(key), "right": right_values.get(key)}
            for key in sorted(set(left_values) | set(right_values))
            if left_values.get(key) != right_values.get(key)
        }
    metric_differences = {}
    for method in ("homogeneous", "league"):
        left_metrics = reports["left"]["methods"][method]
        right_metrics = reports["right"]["methods"][method]
        metric_differences[method] = {
            key: {
                "left_mean": left_metrics[key]["mean"],
                "right_mean": right_metrics[key]["mean"],
                "delta_right_minus_left": (
                    right_metrics[key]["mean"] - left_metrics[key]["mean"]
                ),
                "left_ci95": left_metrics[key]["ci95"],
                "right_ci95": right_metrics[key]["ci95"],
            }
            for key in (
                "fitness",
                "profit",
                "city_quality",
                "mean_satisfaction",
                "green_share",
                "violations",
                "damage_total",
                "bankrupt_agents",
            )
        }

    _write_json(
        result_target,
        {
            "refs": list(refs),
            "reference_ref": reference_ref,
            "city_config": reference_city.as_dict(),
            "seeds": seeds,
            "config_differences": setting_differences,
            "metric_differences": metric_differences,
            "reports": reports,
        },
    )
    _write_json(
        progress_target,
        {
            "state": "completed",
            "phase": "comparison complete",
            "generation": 2 * len(refs),
            "generations": 2 * len(refs),
            "eta_seconds": 0,
        },
    )


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 4:
        raise SystemExit(
            "usage: _dashboard-compare-worker JOB.json RESULT.json PROGRESS.json STOP"
        )
    run_compare(*arguments)


if __name__ == "__main__":
    main()
