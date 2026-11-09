"""Episode runner, sealed-bid plot auction and fitness scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import CityConfig, FitnessWeights
from .policy import Policy
from .world import Action, ActionRecord, AgentMetrics, CityMetrics, CityEnvironment


@dataclass
class EpisodeResult:
    seed: int
    city: CityMetrics
    agents: list[AgentMetrics]
    history: list[dict]
    actions: list[ActionRecord]
    config: CityConfig
    shocks: list[dict] = field(default_factory=list)
    type_grid: np.ndarray | None = None


ActionCallback = Callable[[int, Action, bool, CityEnvironment], None]
RoundCallback = Callable[[int, CityEnvironment], None]


def _apply_winner_actions(
    env: CityEnvironment,
    winners: dict[int, list[tuple[float, int, int, Action]]],
    slots: list[int],
    on_action: ActionCallback | None,
) -> None:
    """Apply up to ``slots[agent]`` winning actions for each agent."""
    for agent_id, proposals in winners.items():
        if slots[agent_id] <= 0:
            continue
        # Highest bid first, then original priority.
        proposals.sort(key=lambda item: (-item[0], item[1]))
        for _, _, _, action in proposals:
            if slots[agent_id] <= 0:
                break
            if env.agents[agent_id].bankrupt:
                break
            success = env.apply_action(agent_id, action)
            slots[agent_id] -= 1
            if on_action is not None:
                on_action(agent_id, action, success, env)


def _resolve_bids(
    proposals: dict[int, list[Action]],
    rng: np.random.Generator,
) -> dict[int, list[tuple[float, int, int, Action]]]:
    """Resolve sealed bids for each cell.

    Returns ``agent_id -> [(bid, priority, tie_break, action), ...]``.  Only the
    highest bid per cell survives.  Ties are broken randomly.
    """
    best_by_cell: dict[int, tuple[float, int, int, int, Action]] = {}
    for agent_id, actions in proposals.items():
        for action in actions:
            if action.cell is None:
                continue
            cell = int(action.cell)
            key = (float(action.bid), int(action.priority), int(rng.integers(0, 2**31)))
            current = best_by_cell.get(cell)
            if current is None or key[:3] > current[:3]:
                best_by_cell[cell] = (key[0], key[1], key[2], agent_id, action)
    winners: dict[int, list[tuple[float, int, int, Action]]] = {}
    for bid, priority, tie, agent_id, action in best_by_cell.values():
        winners.setdefault(agent_id, []).append((bid, priority, tie, action))
    return winners


def _run_auction_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None,
) -> None:
    n = config.n_agents
    slots = [config.buildings_per_action for _ in range(n)]
    active = [i for i, agent in enumerate(env.agents) if not agent.bankrupt]

    # Phase 1: everybody submits a small portfolio of sealed bids.
    proposals = {
        i: policies[i].propose_bids(
            env, i, rng, count=config.bids_per_agent
        )
        for i in active
    }
    winners = _resolve_bids(proposals, rng)
    _apply_winner_actions(env, winners, slots, on_action)

    # Phase 2: agents that still have a free slot may react to occupied cells.
    active2 = [i for i in active if slots[i] > 0 and not env.agents[i].bankrupt]
    if active2:
        proposals2 = {
            i: policies[i].propose_bids(
                env, i, rng, count=max(1, slots[i] + 1)
            )
            for i in active2
        }
        winners2 = _resolve_bids(proposals2, rng)
        _apply_winner_actions(env, winners2, slots, on_action)

    # Unfilled slots are economic skips.
    for i in active:
        while slots[i] > 0:
            env._record_pass(i)
            slots[i] -= 1
            if on_action is not None:
                on_action(i, Action.pass_action(), False, env)


def _run_sequential_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None,
) -> None:
    order = rng.permutation(config.n_agents)
    for agent_id in order:
        agent_id = int(agent_id)
        if env.agents[agent_id].bankrupt:
            continue
        for _ in range(config.buildings_per_action):
            action = policies[agent_id].act(env, agent_id, rng)
            success = env.apply_action(agent_id, action)
            if on_action is not None:
                on_action(agent_id, action, success, env)
            if not success:
                break


def run_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None = None,
) -> None:
    """Run one full bidding/building round without ending it."""
    if config.use_auction:
        _run_auction_round(env, policies, rng, config, on_action)
    else:
        _run_sequential_round(env, policies, rng, config, on_action)


def run_episode(
    config: CityConfig,
    policies: list[Policy],
    seed: int,
    record_history: bool = False,
    rng_seed: int | None = None,
    shocks: list[dict] | None = None,
    on_action: ActionCallback | None = None,
    on_round_start: RoundCallback | None = None,
    on_round_end: RoundCallback | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> EpisodeResult:
    if len(policies) != config.n_agents:
        raise ValueError(
            f"expected {config.n_agents} policies, got {len(policies)}"
        )
    env = CityEnvironment(config, seed, record_history=record_history)
    if rng_seed is None:
        rng_seed = (int(seed) * 2_654_435_761 + 0x9E3779B9) % (2**63)
    rng = np.random.default_rng(int(rng_seed))
    shocks = list(shocks or [])
    for round_idx in range(config.actions_per_agent):
        if should_continue is not None and not should_continue():
            break
        if on_round_start is not None:
            on_round_start(round_idx, env)
        for shock in shocks:
            if int(shock.get("round", -1)) == round_idx:
                env.apply_shock(
                    str(shock.get("kind", "earthquake")),
                    rng,
                    magnitude=float(shock.get("magnitude", 0.5)),
                )
        run_round(env, policies, rng, config, on_action)
        env.end_round()
        if on_round_end is not None:
            on_round_end(round_idx, env)
    return EpisodeResult(
        seed=int(seed),
        city=env.city_metrics(),
        agents=env.agent_metrics(),
        history=env.history,
        actions=env.actions,
        config=config,
        shocks=env.shock_events,
        type_grid=env.type_grid.copy(),
    )


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
    bankrupt = np.array([a.bankrupt for a in result.agents], dtype=np.float64)
    actions = max(
        1,
        result.config.actions_per_agent * result.config.buildings_per_action,
    )

    # Smooth, scale-aware normalisation.  tanh does not saturate at the very
    # short horizons used by the quick experiments and keeps the sign.
    profit_norm = np.tanh(profits / 1000.0)
    build_norm = np.clip(builds / actions, 0.0, 1.0)
    violation_norm = violations / (violations + 10.0)
    damage_norm = damage / (damage + 50.0)
    skip_norm = np.clip(skips / actions, 0.0, 1.0)

    scores = (
        weights.profit * profit_norm
        + weights.city_quality * result.city.city_quality
        + weights.build_count * build_norm
        - weights.violations * violation_norm
        - weights.externality_damage * damage_norm
        - weights.skip_penalty * skip_norm
        - weights.bankruptcy * bankrupt
    )
    return {
        "score": scores,
        "profit_norm": profit_norm,
        "build_norm": build_norm,
        "violation_norm": violation_norm,
        "damage_norm": damage_norm,
        "skip_norm": skip_norm,
        "bankruptcy": bankrupt,
        "city_quality": np.full_like(scores, result.city.city_quality),
        "satisfaction": np.full_like(scores, result.city.mean_satisfaction),
    }


def score_episode(
    result: EpisodeResult,
    weights: FitnessWeights | None = None,
) -> np.ndarray:
    return score_breakdown(result, weights)["score"]
