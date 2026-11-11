"""Persistent, portable analytics records for dashboard runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np


_SUMMARY_VERSION = 1


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_value(payload),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = np.asarray(
        [row.get(key) for row in rows if isinstance(row.get(key), (int, float))],
        dtype=np.float64,
    )
    if not values.size:
        return None
    x = np.arange(values.size, dtype=np.float64)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    slope = (
        float(np.dot(centered_x, values - values.mean()) / denominator)
        if denominator
        else 0.0
    )
    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "trend_per_round": slope,
    }


class AnalyticsRecorder:
    """Write action and round records while retaining only compact summaries."""

    def __init__(self, directory: str | Path, run_kind: str, seed: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        self.started_clock = time.perf_counter()
        self.run_kind = run_kind
        self.seed = int(seed)
        self.action_count = 0
        self.action_counts: dict[str, dict[str, int]] = {
            "by_agent": {},
            "by_round": {},
            "by_round_agent": {},
            "by_type": {},
            "by_outcome": {},
        }
        self._rounds: list[dict[str, Any]] = []
        self._actions = (self.directory / "actions.ndjson").open(
            "w", encoding="utf-8", newline="\n"
        )
        self._round_stream = (self.directory / "rounds.ndjson").open(
            "w", encoding="utf-8", newline="\n"
        )

    def record_action(self, agent_id: int, action: Any, accepted: bool, env: Any) -> None:
        outcome = "accepted" if accepted else "rejected"
        round_number = int(env.round)
        action_record = (
            env.actions[-1]
            if env.actions
            and env.actions[-1].agent_id == int(agent_id)
            and env.actions[-1].round == round_number
            else None
        )
        row = {
            "event": "action",
            "seq": self.action_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started_clock,
            "round": round_number,
            "agent_id": int(agent_id),
            "outcome": outcome,
            "damage": float(action_record.damage) if action_record is not None else 0.0,
            "action": {
                "kind": (
                    "pass"
                    if action.cell is None
                    else "demolish"
                    if action.demolish
                    else "build"
                ),
                "cell": action.cell,
                "type_id": int(action.type_id),
                "quality": int(action.quality),
                "floors": int(action.floors),
                "roof_quality": int(action.roof_quality),
                "cost": float(action.cost),
                "bid": float(action.bid),
                "demolish": bool(action.demolish),
                "priority": int(action.priority),
            },
            "agent_state": {
                "budget": float(env.agents[agent_id].budget),
                "debt": float(env.agents[agent_id].debt),
                "debt_ratio": float(env.agents[agent_id].debt_ratio),
                "profit": float(env.agents[agent_id].profit),
                "bankrupt": bool(env.agents[agent_id].bankrupt),
            },
        }
        self._write_line(self._actions, row)
        self.action_count += 1
        self._increment("by_agent", str(agent_id))
        self._increment("by_round", str(round_number))
        self._increment("by_round_agent", f"{round_number}:{agent_id}")
        type_key = (
            "pass"
            if action.cell is None
            else "demolish"
            if action.demolish
            else str(action.type_id)
        )
        self._increment("by_type", type_key)
        self._increment("by_outcome", outcome)

    def record_round(self, round_number: int, env: Any) -> None:
        city = _json_value(env.city_metrics())
        row = {
            "event": "round",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started_clock,
            "round": int(round_number),
            "city": city,
        }
        self._write_line(self._round_stream, row)
        self._actions.flush()
        self._round_stream.flush()
        self._rounds.append(row)

    def finish(
        self,
        result: Any,
        scores: Iterable[float],
        score_breakdown: dict[str, Any],
    ) -> dict[str, Any]:
        self.close()
        metric_keys = sorted(
            {
                key
                for row in self._rounds
                for key, value in row["city"].items()
                if isinstance(value, (int, float))
            }
        )
        metrics = {
            key: _metric_summary(
                [row["city"] for row in self._rounds],
                key,
            )
            for key in metric_keys
        }
        summary = {
            "schema_version": _SUMMARY_VERSION,
            "kind": self.run_kind,
            "seed": self.seed,
            "started_at": self.started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - self.started_clock,
            "round_count": len(self._rounds),
            "action_count": self.action_count,
            "rounds_file": "rounds.ndjson",
            "actions_file": "actions.ndjson",
            "action_counts": self.action_counts,
            "metrics": metrics,
            "city": _json_value(result.city),
            "agents": _json_value(result.agents),
            "scores": _json_value(list(scores)),
            "score_breakdown": _json_value(score_breakdown),
            "shocks": _json_value(result.shocks),
            "config": _json_value(result.config.as_dict()),
        }
        _write_json(self.directory / "analytics_summary.json", summary)
        return summary

    def close(self) -> None:
        for stream in (self._actions, self._round_stream):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()

    def _increment(self, group: str, key: str) -> None:
        values = self.action_counts[group]
        values[key] = values.get(key, 0) + 1

    @staticmethod
    def _write_line(stream: Any, row: dict[str, Any]) -> None:
        stream.write(
            json.dumps(
                _json_value(row),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        stream.write("\n")

    def __enter__(self) -> AnalyticsRecorder:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def write_training_analytics(
    directory: str | Path,
    history: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in history
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    result = {
        "schema_version": _SUMMARY_VERSION,
        "metadata": metadata,
        "generation_count": len(history),
        "series": _json_value(history),
        "metrics": {
            key: _metric_summary(history, key)
            for key in numeric_keys
        },
    }
    _write_json(Path(directory) / "training_metrics.json", result)
    return result


def load_analytics(
    directory: str | Path,
    *,
    round_start: int | None = None,
    round_end: int | None = None,
    agent_id: int | None = None,
    action_limit: int = 2_000,
) -> dict[str, Any]:
    root = Path(directory)
    training_progress_path = root / "training_progress.ndjson"
    event_root = root
    if not training_progress_path.is_file() and not (event_root / "actions.ndjson").is_file() and (
        event_root / "final_evaluation" / "actions.ndjson"
    ).is_file():
        event_root = event_root / "final_evaluation"
    summary_path = root / "analytics_summary.json"
    if not summary_path.is_file() and event_root != root:
        summary_path = event_root / "analytics_summary.json"
    summary = (
        _read_json(summary_path)
        if summary_path.is_file()
        else {"schema_version": _SUMMARY_VERSION, "round_count": 0, "action_count": 0}
    )
    rounds = _load_rounds(event_root, round_start, round_end)
    actions, action_metrics = _load_actions(
        event_root, round_start, round_end, agent_id, action_limit
    )
    summary["metrics"] = {
        key: _metric_summary([row.get("city", {}) for row in rounds], key)
        for key in sorted(
            {
                key
                for row in rounds
                for key, value in row.get("city", {}).items()
                if isinstance(value, (int, float))
            }
        )
    }
    summary["selected_round_count"] = len(rounds)
    if rounds:
        summary["selected_city"] = rounds[-1].get("city", {})
    summary["selected_action_count"] = action_metrics["count"]
    training_path = root / "training_metrics.json"
    training = _read_json(training_path) if training_path.is_file() else None
    if training is None:
        training = _load_legacy_training(root)
    if training_progress_path.is_file():
        progress_rows = _load_training_progress(training_progress_path)
        if progress_rows:
            rows_by_generation = {
                int(row.get("gen", row.get("generation", -1))): row
                for row in (training or {}).get("series", [])
                if isinstance(row, dict)
            }
            for row in progress_rows:
                rows_by_generation[int(row["gen"])] = row
            training = dict(training or {})
            training["series"] = [
                rows_by_generation[generation]
                for generation in sorted(rows_by_generation)
            ]
            training["generation_count"] = len(training["series"])
    return {
        "summary": summary,
        "rounds": rounds,
        "actions": actions,
        "action_metrics": action_metrics,
        "training": training,
        "actions_truncated": action_metrics["count"] > len(actions),
    }


def _load_training_progress(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError(
                f"Invalid training progress record on line {index + 1}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("gen"), int):
            raise ValueError(f"Invalid training progress record on line {index + 1}")
        rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Analytics file exceeds 32 MiB: {path.name}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid analytics object: {path.name}")
    return value


def _load_rounds(
    root: Path, round_start: int | None, round_end: int | None
) -> list[dict[str, Any]]:
    path = root / "rounds.ndjson"
    if path.is_file():
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                round_number = int(row["round"])
                if (
                    (round_start is None or round_number >= round_start)
                    and (round_end is None or round_number <= round_end)
                ):
                    rows.append(row)
        return rows
    history = root / "history.csv"
    if not history.is_file():
        return []
    rows = []
    with history.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            round_number = int(row["round"])
            if (
                (round_start is None or round_number >= round_start)
                and (round_end is None or round_number <= round_end)
            ):
                values = {
                    key: _number(value)
                    for key, value in row.items()
                    if key != "round" and value not in (None, "")
                }
                rows.append({"round": round_number, "city": values})
    return rows


def _load_actions(
    root: Path,
    round_start: int | None,
    round_end: int | None,
    agent_id: int | None,
    action_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "actions.ndjson"
    if not path.is_file():
        return [], {
            "count": 0,
            "accepted": 0,
            "rejected": 0,
            "by_agent": {},
            "by_round_agent": {},
            "accepted_by_round_agent": {},
            "by_type": {},
            "building_count": 0,
            "floors_bins": [],
            "roof_quality_counts": {},
            "cost_bins": [],
            "bid_bins": [],
            "damage_bins": [],
        }
    count = accepted = rejected = 0
    by_agent: dict[str, int] = {}
    by_round_agent: dict[str, int] = {}
    accepted_by_round_agent: dict[str, int] = {}
    by_type: dict[str, int] = {}
    floor_values: list[float] = []
    roof_quality_counts: dict[str, int] = {}
    values_sample: list[tuple[float, float, float]] = []
    sample: list[dict[str, Any]] = []
    rng = random.Random(0)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            number = int(row["round"])
            if (
                (round_start is not None and number < round_start)
                or (round_end is not None and number > round_end)
                or (agent_id is not None and int(row["agent_id"]) != agent_id)
            ):
                continue
            count += 1
            agent_key = str(row["agent_id"])
            by_agent[agent_key] = by_agent.get(agent_key, 0) + 1
            round_agent_key = f"{number}:{agent_key}"
            by_round_agent[round_agent_key] = by_round_agent.get(round_agent_key, 0) + 1
            action = row["action"]
            kind = action.get("kind")
            type_key = (
                kind
                if kind in ("pass", "demolish")
                else str(action.get("type_id", 0))
            )
            by_type[type_key] = by_type.get(type_key, 0) + 1
            if row["outcome"] == "accepted":
                accepted += 1
                accepted_by_round_agent[round_agent_key] = (
                    accepted_by_round_agent.get(round_agent_key, 0) + 1
                )
                if action.get("kind") == "build" and not action.get("demolish", False):
                    floor_values.append(float(action.get("floors", 1)))
                    roof_key = str(action.get("roof_quality", 0))
                    roof_quality_counts[roof_key] = (
                        roof_quality_counts.get(roof_key, 0) + 1
                    )
            else:
                rejected += 1
            values = (
                float(action.get("cost", 0.0)),
                float(action.get("bid", 0.0)),
                float(row.get("damage", 0.0)),
            )
            if len(values_sample) < 20_000:
                values_sample.append(values)
            else:
                index = rng.randrange(count)
                if index < len(values_sample):
                    values_sample[index] = values
            if len(sample) < action_limit:
                sample.append(row)
            elif action_limit and rng.randrange(count) < action_limit:
                sample[rng.randrange(action_limit)] = row
    sample.sort(key=lambda row: int(row.get("seq", 0)))
    return sample, {
        "count": count,
        "accepted": accepted,
        "rejected": rejected,
        "by_agent": by_agent,
        "by_round_agent": by_round_agent,
        "accepted_by_round_agent": accepted_by_round_agent,
        "by_type": by_type,
        "building_count": len(floor_values),
        "floors_bins": _histogram(floor_values),
        "roof_quality_counts": roof_quality_counts,
        "sample_count": len(values_sample),
        "cost_bins": _histogram([values[0] for values in values_sample]),
        "bid_bins": _histogram([values[1] for values in values_sample]),
        "damage_bins": _histogram([values[2] for values in values_sample]),
    }


def _histogram(values: list[float], bins: int = 12) -> list[dict[str, float | int]]:
    if not values:
        return []
    counts, edges = np.histogram(values, bins=min(bins, max(1, len(set(values)))))
    return [
        {
            "from": float(edges[index]),
            "to": float(edges[index + 1]),
            "count": int(counts[index]),
        }
        for index in range(len(counts))
    ]


def _number(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _load_legacy_training(root: Path) -> dict[str, Any] | None:
    histories = sorted(root.glob("*_history.csv"))
    if not histories:
        return None
    series = []
    for path in histories:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                point = {
                    key: _number(value)
                    for key, value in row.items()
                    if value not in (None, "")
                }
                point["series"] = path.stem
                series.append(point)
    return {"series": series, "generation_count": len(series), "legacy": True}
