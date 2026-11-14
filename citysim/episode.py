"""Episode runner, sealed-bid plot auction and fitness scoring."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .config import ROAD_ACTION_ID, CityConfig, FitnessWeights
from .policy import Policy, auction_reserve, cap_bid_for_action
from .shocks import automatic_shocks, shock_rng
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
    player_road_grid: np.ndarray | None = None
    pending_road_grid: np.ndarray | None = None
    road_owner_grid: np.ndarray | None = None
    road_cost_grid: np.ndarray | None = None


ActionCallback = Callable[[int, Action, bool, CityEnvironment], bool | None]
RoundCallback = Callable[[int, CityEnvironment], None]


def _apply_winner_actions(
    env: CityEnvironment,
    winners: dict[int, list[tuple[float, int, int, Action]]],
    slots: list[int],
    on_action: ActionCallback | None,
) -> bool:
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
            if success:
                slots[agent_id] -= 1
            if on_action is not None:
                if on_action(agent_id, action, success, env) is False:
                    return False
    return True


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
            bid = float(action.bid)
            if not np.isfinite(bid):
                continue
            if bid + 1e-12 < auction_reserve(action):
                continue
            cell = int(action.cell)
            key = (bid, int(action.priority), int(rng.integers(0, 2**31)))
            current = best_by_cell.get(cell)
            if current is None or key[:3] > current[:3]:
                best_by_cell[cell] = (key[0], key[1], key[2], agent_id, action)
    winners: dict[int, list[tuple[float, int, int, Action]]] = {}
    for bid, priority, tie, agent_id, action in best_by_cell.values():
        winners.setdefault(agent_id, []).append((bid, priority, tie, action))
    return winners


def _proposal_cost(env: CityEnvironment, action: Action) -> float:
    """Return the construction cost of a proposal.

    Custom policies may omit ``Action.cost``.  The auction still has to price
    the plot, so the cost is recomputed exactly as ``apply_action`` would.
    """
    cost = float(action.cost or 0.0)
    if action.demolish or action.cell is None or cost > 0.0:
        return cost
    cell = int(action.cell)
    try:
        if int(action.type_id) == ROAD_ACTION_ID:
            return float(env.road_cost_of(cell))
        return float(
            env.cost_of(
                cell,
                int(action.type_id),
                int(action.quality),
                int(action.floors),
                int(action.roof_quality),
            )
        )
    except (TypeError, ValueError):
        # Invalid proposals are rejected later by ``apply_action``; the
        # auction must not crash while pricing them.
        return 0.0


def _cap_proposed_bids(
    env: CityEnvironment,
    agent_id: int,
    actions: list[Action],
) -> list[Action]:
    budget = env.agents[agent_id].budget
    capped: list[Action] = []
    for action in actions:
        try:
            desired = float(action.bid)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(desired):
            continue
        cost = _proposal_cost(env, action)
        capped.append(
            replace(
                action,
                cost=cost,
                bid=cap_bid_for_action(desired, budget, cost),
            )
        )
    return capped


def _run_auction_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None,
) -> bool:
    n = config.n_agents
    slots = [config.buildings_per_action for _ in range(n)]
    active = [i for i, agent in enumerate(env.agents) if not agent.bankrupt]

    # Phase 1: everybody submits a small portfolio of sealed bids.
    proposals = {
        i: _cap_proposed_bids(
            env,
            i,
            policies[i].propose_bids(env, i, rng, count=config.bids_per_agent),
        )
        for i in active
    }
    winners = _resolve_bids(proposals, rng)
    if not _apply_winner_actions(env, winners, slots, on_action):
        return False

    # Phase 2: agents that still have a free slot may react to occupied cells.
    active2 = [i for i in active if slots[i] > 0 and not env.agents[i].bankrupt]
    if active2:
        proposals2 = {
            i: _cap_proposed_bids(
                env,
                i,
                policies[i].propose_bids(
                    env, i, rng, count=max(1, slots[i] + 1)
                ),
            )
            for i in active2
        }
        winners2 = _resolve_bids(proposals2, rng)
        if not _apply_winner_actions(env, winners2, slots, on_action):
            return False

    # Unfilled slots are economic skips.
    for i in active:
        while slots[i] > 0:
            env._record_pass(i)
            slots[i] -= 1
            if on_action is not None:
                if on_action(i, Action.pass_action(), False, env) is False:
                    return False
    return True


def _run_sequential_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None,
) -> bool:
    order = rng.permutation(config.n_agents)
    for agent_id in order:
        agent_id = int(agent_id)
        if env.agents[agent_id].bankrupt:
            continue
        for _ in range(config.buildings_per_action):
            action = policies[agent_id].act(env, agent_id, rng)
            success = env.apply_action(agent_id, action)
            if on_action is not None:
                if on_action(agent_id, action, success, env) is False:
                    return False
            if not success:
                break
    return True


def run_round(
    env: CityEnvironment,
    policies: list[Policy],
    rng: np.random.Generator,
    config: CityConfig,
    on_action: ActionCallback | None = None,
) -> bool:
    """Run one full round, with each agent taking the configured action turns."""
    for _ in range(config.actions_per_round):
        env._current_income = env._income_grid()
        if config.use_auction:
            completed = _run_auction_round(env, policies, rng, config, on_action)
        else:
            completed = _run_sequential_round(env, policies, rng, config, on_action)
        if not completed:
            return False
    return True


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
    include_automatic_shocks: bool = True,
) -> EpisodeResult:
    if len(policies) != config.n_agents:
        raise ValueError(
            f"expected {config.n_agents} policies, got {len(policies)}"
        )
    for policy in policies:
        reset_state = getattr(policy, "reset_state", None)
        if callable(reset_state):
            reset_state()
    env = CityEnvironment(config, seed, record_history=record_history)
    if rng_seed is None:
        rng_seed = (int(seed) * 2_654_435_761 + 0x9E3779B9) % (2**63)
    rng = np.random.default_rng(int(rng_seed))
    scheduled_shocks = (
        [shock.as_dict() for shock in automatic_shocks(config.actions_per_agent)]
        if include_automatic_shocks
        else []
    )
    scheduled_shocks.extend(dict(shock) for shock in shocks or [])
    shocks_by_round: dict[int, list[tuple[int, dict]]] = {}
    for event_index, shock in enumerate(scheduled_shocks):
        round_index = int(shock.get("round", -1))
        shocks_by_round.setdefault(round_index, []).append((event_index, shock))
    for round_idx in range(config.actions_per_agent):
        if should_continue is not None and not should_continue():
            break
        for event_index, shock in shocks_by_round.get(round_idx, []):
            kind = str(shock.get("kind", "earthquake"))
            env.apply_shock(
                kind,
                shock_rng(seed, round_idx, event_index, kind),
                magnitude=float(shock.get("magnitude", 0.5)),
                duration=(
                    int(shock["duration"]) if shock.get("duration") is not None else None
                ),
            )
        if on_round_start is not None:
            on_round_start(round_idx, env)
        if not run_round(env, policies, rng, config, on_action):
            break
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
        player_road_grid=env.player_road_grid.copy(),
        pending_road_grid=env.pending_road_grid.copy(),
        road_owner_grid=env.road_owner_grid.copy(),
        road_cost_grid=env.road_cost_grid.copy(),
    )


def _profit_reference(config: CityConfig, weights: FitnessWeights) -> float:
    """Scale the absolute-profit midpoint with the episode's economic size.

    The default reference (250k) is calibrated for a 25x25 city with 10 agents,
    100 rounds of two actions and a 500 budget.  Smaller or shorter episodes
    scale the midpoint down so that absolute profit still carries signal.
    """
    scale = (
        (config.height * config.width) / 625.0
        * (
            config.actions_per_agent
            * config.actions_per_round
            * config.buildings_per_action
        )
        / 200.0
        * (config.n_agents / 10.0)
        * (config.initial_budget / 500.0)
    )
    return max(1e-9, float(weights.profit_reference) * scale)


def score_breakdown(
    result: EpisodeResult,
    weights: FitnessWeights | None = None,
) -> dict[str, np.ndarray]:
    weights = weights or result.config.fitness
    agents = result.agents
    profits = np.array([a.profit for a in agents], dtype=np.float64)
    builds = np.array([a.builds for a in agents], dtype=np.float64)
    skips = np.array([a.skips for a in agents], dtype=np.float64)
    violations = np.array([a.violations for a in agents], dtype=np.float64)
    damage = np.array([a.damage_caused for a in agents], dtype=np.float64)
    bankrupt = np.array([float(a.bankrupt) for a in agents], dtype=np.float64)
    debt_ratios = np.array([a.debt_ratio for a in agents], dtype=np.float64)
    green_shares = np.array(
        [float(getattr(a, "green_share", 0.0)) for a in agents],
        dtype=np.float64,
    )
    incompatible = np.array(
        [float(getattr(a, "incompatible_neighbors", 0.0)) for a in agents],
        dtype=np.float64,
    )
    city_welfare = float(
        np.clip(
            0.5 * (result.city.city_quality + result.city.mean_satisfaction),
            0.0,
            1.0,
        )
    )
    slots_per_round = (
        result.config.actions_per_round * result.config.buildings_per_action
    )
    actions = max(1, result.config.actions_per_agent * slots_per_round)

    if profits.size > 1:
        _, inverse, counts = np.unique(
            profits, return_inverse=True, return_counts=True
        )
        rank_starts = np.cumsum(np.r_[0, counts[:-1]])
        average_ranks = rank_starts + (counts - 1) / 2.0
        profit_norm = average_ranks[inverse] / (profits.size - 1)
    elif profits.size == 1:
        profit_norm = np.full(1, 0.5, dtype=np.float64)
    else:
        profit_norm = np.empty(0, dtype=np.float64)

    reference = _profit_reference(result.config, weights)
    profit_absolute = profits / (np.abs(profits) + reference)

    green_min = float(result.config.requirements.green_min)
    if green_min > 0.0:
        green_norm = np.clip(green_shares / green_min, 0.0, 1.0)
    else:
        green_norm = np.ones_like(green_shares)

    build_norm = np.minimum(np.maximum(builds / actions, 0.0), 1.0)
    violation_norm = violations / (violations + weights.violation_saturation)
    damage_norm = damage / (damage + weights.damage_saturation)
    skip_norm = np.minimum(np.maximum(skips / actions, 0.0), 1.0)
    debt_penalty = debt_ratios / (1.0 + debt_ratios)
    compatibility_norm = 1.0 / (
        1.0 + incompatible / weights.compatibility_saturation
    )

    scores = (
        weights.profit * profit_norm
        + weights.profit_absolute * profit_absolute
        + weights.city_quality * result.city.city_quality
        + weights.build_count * build_norm
        + weights.green_share * green_norm
        + weights.compatibility * compatibility_norm
        - weights.violations * violation_norm
        - weights.externality_damage * damage_norm
        - weights.skip_penalty * skip_norm
        - weights.debt * debt_penalty
        - weights.bankruptcy * bankrupt
        + weights.shared_reward * city_welfare
    )
    return {
        "score": scores,
        "profit_norm": profit_norm,
        "profit_absolute": profit_absolute,
        "build_norm": build_norm,
        "violation_norm": violation_norm,
        "damage_norm": damage_norm,
        "skip_norm": skip_norm,
        "green_norm": green_norm,
        "compatibility_norm": compatibility_norm,
        "debt_ratio": debt_ratios,
        "debt_penalty": debt_penalty,
        "bankruptcy": bankrupt,
        "bankruptcy_penalty": bankrupt,
        "city_welfare": np.full_like(scores, city_welfare),
        "shared_reward": np.full_like(scores, weights.shared_reward * city_welfare),
        "city_quality": np.full_like(scores, result.city.city_quality),
        "satisfaction": np.full_like(scores, result.city.mean_satisfaction),
        "profit_reference": np.full_like(scores, reference),
    }


def score_episode(
    result: EpisodeResult,
    weights: FitnessWeights | None = None,
) -> np.ndarray:
    return score_breakdown(result, weights)["score"]
