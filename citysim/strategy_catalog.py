"""Read models for the dashboard's branch graph and hierarchical strategy catalog."""

from __future__ import annotations

from collections import defaultdict

from .training_history import TrainingHistoryStore


def branch_graph(store: TrainingHistoryStore) -> dict:
    branches = store.branch_names()
    commits: dict[str, dict] = {}
    for branch in branches:
        if branch["head_commit"] is None:
            continue
        for commit in store.history(branch["name"]):
            entry = commits.setdefault(
                commit["commit_id"],
                {
                    "commit_id": commit["commit_id"],
                    "parent_id": commit["parent_id"],
                    "branch_name": commit["branch_name"],
                    "branch_names": [],
                    "generation": commit["generation"],
                    "run_id": commit.get("run_id"),
                    "best_fitness": commit["best_fitness"],
                    "complete": bool(commit["complete"]),
                },
            )
            entry["branch_names"].append(branch["name"])
    return {
        "branches": branches,
        "commits": sorted(
            commits.values(),
            key=lambda commit: (commit["generation"], commit["commit_id"]),
        ),
    }


def branch_training_runs(store: TrainingHistoryStore, branch: str) -> list[dict]:
    branch_info = next(
        (item for item in store.branch_names() if item["name"] == branch), None
    )
    if branch_info is None:
        raise ValueError(f"unknown branch: {branch}")
    if branch_info["head_commit"] is None:
        return []

    groups: dict[str, list[dict]] = defaultdict(list)
    for commit in store.history(branch):
        run_id = commit.get("run_id")
        key = f"run:{run_id}" if run_id is not None else "legacy"
        groups[key].append(commit)
    return [
        {
            "run_id": None if run_key == "legacy" else run_key.removeprefix("run:"),
            "key": run_key,
            "label": (
                "Исторические снимки без ID запуска"
                if run_key == "legacy"
                else f"Запуск {run_key.removeprefix('run:')[:8]}"
            ),
            "checkpoint_count": len(commits),
            "first_generation": commits[0]["generation"],
            "last_generation": commits[-1]["generation"],
        }
        for run_key, commits in sorted(
            groups.items(),
            key=lambda pair: pair[1][-1]["generation"],
            reverse=True,
        )
    ]


def run_checkpoints(
    store: TrainingHistoryStore, branch: str, run_key: str
) -> list[dict]:
    runs = branch_training_runs(store, branch)
    if not any(run["key"] == run_key for run in runs):
        raise ValueError(f"unknown training run: {run_key}")
    branch_info = next(item for item in store.branch_names() if item["name"] == branch)
    if branch_info["head_commit"] is None:
        return []
    return [
        {
            "commit_id": commit["commit_id"],
            "parent_id": commit["parent_id"],
            "generation": commit["generation"],
            "best_fitness": commit["best_fitness"],
            "complete": bool(commit["complete"]),
        }
        for commit in store.history(branch)
        if (
            f"run:{commit['run_id']}"
            if commit.get("run_id") is not None
            else "legacy"
        ) == run_key
    ]
