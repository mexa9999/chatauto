"""Episode runner and fitness scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CityConfig, FitnessWeights
from .policy import Policy
from .world import ActionRecord, AgentMetrics, CityMetrics, CityEnvironment


@dataclass
class EpisodeResult:
    seed: int
    city: CityMetrics
    agents: list[AgentMetrics]
    history: list[dict]
    actions: list[ActionRecord]
    config: CityConfig


def run_episode(
    config: CityConfig,
    policies: list[Policy],
    seed: int,
    record_history: bool = False,
) -> EpisodeResult:
    if len(policies) != config.n_agents:
        raise ValueError(
            f"expected {config.n_agents} policies, got {len(policies)}"
        )
    env = CityEnvironment(config, seed, record_history=record_history)
    rng = np.random.default_rng(
        (int(seed) * 2_654_435_761 + 0x9E3779B9) % (2**63)
    )
    for _ in range(config.actions_per_agent):
        order = rng.permutation(config.n_agents)
        for agent_id in order:
            action = policies[int(agent_id)].act(env, int(agent_id), rng)
            env.apply_action(int(agent_id), action)
        env.end_round()
    return EpisodeResult(
        seed=int(seed),
        city=env.city_metrics(),
        agents=env.agent_metrics(),
        history=env.history,
        actions=env.actions,
        config=config,
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def score_breakdown(
    result: EpisodeResult,
    weights: FitnessWeights | None = None,
) -> dict[str, np.ndarray]:
    weights = weights or result.config.fitness
    profits = np.array([a.profit for a in result.agents], dtype=np.float64)
    builds = np.array([a.builds for a in result.agents], dtype=np.float64)
    skips = np.array([a.skips for a in result.agents], dtype=np.float64)
    violations = np.array([a.violations for a in result.agents], dtype=np.float64)
    damage = np.array([a.damage_caused for a in result.agents], dtype=np.float64)
    actions = max(1, result.config.actions_per_agent)

    profit_norm = np.clip(profits / result.config.initial_budget, -1.0, 2.0)
    
    build_norm = np.clip(builds / actions, 0.0, 1.0)
    violation_norm = violations / max(1e-9, float(violations.max())) if violations.max() > 0 else np.zeros_like(violations)
    damage_norm = damage / (damage + 50.0)
    skip_norm = np.clip(skips / actions, 0.0, 1.0)

    scores = (
        weights.profit * profit_norm
        + weights.build_count * build_norm
        - weights.violations * violation_norm
        - weights.externality_damage * damage_norm
        - weights.skip_penalty * skip_norm
    )
    return {
        "score": scores,
        "profit_norm": profit_norm,
        "build_norm": build_norm,
        "violation_norm": violation_norm,
        "damage_norm": damage_norm,
        "skip_norm": skip_norm,
        "city_quality": np.full_like(scores, result.city.city_quality),
    }


def score_episode(
    result: EpisodeResult,
    weights: FitnessWeights | None = None,
) -> np.ndarray:
    return score_breakdown(result, weights)["score"]
