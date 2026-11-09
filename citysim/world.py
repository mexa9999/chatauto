"""Multi-agent city environment.

The environment owns all map layers, validates actions, updates the dynamic
layers and attributes externality damage to the agent whose action caused it.
Policies only see :class:`Observation` objects produced by :meth:`observe`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    ACTION_TYPE_NAMES,
    BUILD_COST,
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    GREEN_TARGET,
    INCOME_BASE,
    MAINTENANCE_RATE,
    MAX_DEMAND,
    MAX_PRICE,
    N_CELL_FEATURES,
    N_GLOBAL_FEATURES,
    OPTIMAL_DISTANCE,
    QUALITY_COST_MULT,
    QUALITY_INCOME_MULT,
    QUALITY_NAMES,
    TYPE_NAMES,
    CityConfig,
)
from .grid import box_sum, distance_to_mask, incompatible_edges, make_static_layers, type_entropy


@dataclass(frozen=True)
class Action:
    """A build action.  ``type_id`` is a policy index 0..4 (not a world id)."""

    cell: int | None
    type_id: int = 0
    quality: int = 0
    cost: float = 0.0

    @property
    def is_pass(self) -> bool:
        return self.cell is None

    @classmethod
    def pass_action(cls) -> "Action":
        return cls(cell=None)


@dataclass(frozen=True)
class ActionRecord:
    round: int
    agent_id: int
    cell: int | None
    type_id: int
    quality: int
    cost: float
    damage: float
    accepted: bool


@dataclass
class AgentState:
    agent_id: int
    initial_budget: float
    budget: float = field(init=False)
    revenue: float = 0.0
    spend: float = 0.0
    maintenance: float = 0.0
    damage_caused: float = 0.0
    damage_suffered: float = 0.0
    build_count: int = 0
    skip_count: int = 0

    def __post_init__(self) -> None:
        self.budget = float(self.initial_budget)

    @property
    def profit(self) -> float:
        return float(self.budget - self.initial_budget)


@dataclass
class Observation:
    agent_id: int
    cell_features: np.ndarray      # (n_cells, F)
    global_features: np.ndarray    # (G,)
    action_mask: np.ndarray        # (n_cells, 5, 3) bool
    valid_cells: np.ndarray        # (n_cells,) bool
    cost_table: np.ndarray         # (n_cells, 5, 3)
    remaining_rounds: int
    budget: float


@dataclass
class AgentMetrics:
    agent_id: int
    profit: float
    revenue: float
    spend: float
    maintenance: float
    budget: float
    builds: int
    skips: int
    damage_caused: float
    damage_suffered: float
    violations: float
    incompatible_neighbors: float
    green_deficit_share: float
    density_excess: float
    mean_distance: float
    high_quality_fraction: float
    type_entropy: float
    behavior: np.ndarray


@dataclass
class CityMetrics:
    green_share: float
    green_component: float
    type_entropy: float
    accessibility: float
    compatibility: float
    mean_load: float
    city_quality: float
    violations_total: float
    externality_damage_total: float
    externality_damage_per_agent: float
    share_agents_causing_damage: float
    gini_profit: float
    built_cells: int
    type_counts: dict[int, int]


class CityEnvironment:
    def __init__(self, config: CityConfig, seed: int, record_history: bool = False):
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.h = config.height
        self.w = config.width
        self.n_cells = self.h * self.w
        self.record_history = bool(record_history)

        yy, xx = np.mgrid[0:self.h, 0:self.w]
        self.yy = yy.astype(np.float64)
        self.xx = xx.astype(np.float64)
        self.static = make_static_layers(self.rng, self.h, self.w)
        self.buildable = self.static["buildable"].astype(bool)

        self.type_grid = np.zeros((self.h, self.w), dtype=np.int8)
        self.quality_grid = np.full((self.h, self.w), -1, dtype=np.int8)
        self.owner_grid = np.full((self.h, self.w), -1, dtype=np.int16)
        self.cost_grid = np.zeros((self.h, self.w), dtype=np.float64)

        self.agents = [
            AgentState(agent_id=i, initial_budget=config.initial_budget)
            for i in range(config.n_agents)
        ]
        self.round = 0
        self.actions: list[ActionRecord] = []
        self.history: list[dict] = []

        ones = np.ones((self.h, self.w), dtype=np.float64)
        self._window_n1 = box_sum(ones, 1)
        self._window_n2 = box_sum(ones, 2)
        self._neighbor_n1 = np.maximum(1.0, self._window_n1 - 1.0)
        self._neighbor_n2 = np.maximum(1.0, self._window_n2 - 1.0)

        self._industry_count = -1
        self.dist_to_industry = np.ones((self.h, self.w), dtype=np.float64)
        self._refresh_layers()
        
        self._static_features = np.empty((self.h, self.w, N_CELL_FEATURES), dtype=np.float64)
        self._static_features[..., 0] = 0.0  # price (динамический, заполним позже)
        self._static_features[..., 1] = self.static["soil"]
        self._static_features[..., 2] = self.static["water"]
        self._static_features[..., 3] = self.static["road"]
        self._static_features[..., 4] = self.static["dist_center"]
        self._static_features[..., 12] = self.buildable.astype(np.float64)
        self._cached_income = self._income_grid()

    # ------------------------------------------------------------------
    # Observation and cost tables
    # ------------------------------------------------------------------

    def cost_table(self) -> np.ndarray:
        """Build costs for every (cell, action type, quality)."""
        land_mult = (0.7 + 0.8 * self.price).ravel()
        soil_mult = (0.9 + 0.2 * self.static["soil"]).ravel()
        cell_mult = (land_mult * soil_mult)[:, None, None]
        return (
            BUILD_COST[1:][None, :, None]
            * QUALITY_COST_MULT[None, None, :]
            * cell_mult
        )

    def cost_of(self, cell: int, type_id: int, quality: int) -> float:
        c = int(np.clip(cell, 0, self.n_cells - 1))
        price = float(self.price.ravel()[c])
        soil = float(self.static["soil"].ravel()[c])
        return float(
            BUILD_COST[type_id + 1]
            * QUALITY_COST_MULT[quality]
            * (0.7 + 0.8 * price)
            * (0.9 + 0.2 * soil)
        )

    def observe(self, agent_id: int) -> Observation:
        built = self.type_grid > 0
        free = ~built
        agent = self.agents[agent_id]

        own = (self.owner_grid == agent_id).astype(np.float64)
        own_count = box_sum(own, 2) - own
        competition = np.clip(
            (self._built_count_r2 - own_count) / self._neighbor_n2, 0.0, 1.0
        )

        f = self._static_features.copy()
        f[..., 0] = np.clip(self.price / MAX_PRICE, 0.0, 1.0)
        f[..., 5] = self.density
        f[..., 6] = self.share_housing
        f[..., 7] = self.share_commercial
        f[..., 8] = self.share_industry
        f[..., 9] = self.mean_quality
        f[..., 10] = self.load
        f[..., 11] = np.clip(self.demand / MAX_DEMAND, 0.0, 1.0)
        f[..., 12] = self.buildable.astype(np.float64)
        f[..., 13] = free.astype(np.float64)
        f[..., 14] = competition
        f[..., 15] = self.dist_to_industry

        if not self.config.enable_externalities:
            # Ablation: the policy does not see any neighbour-dependent signal.
            f[..., 5:12] = 0.0
            f[..., 14] = 0.0
            f[..., 15] = 0.0

        costs = self.cost_table()
        affordable = costs <= agent.budget + 1e-9
        free_buildable = (free & self.buildable).ravel()

        buffer = self.config.requirements.industry_buffer
        if buffer > 0:
            industry_mask = (self.type_grid == 4).astype(np.float64)
            sensitive_mask = (
                (self.type_grid == 1)
                | (self.type_grid == 2)
                | (self.type_grid == 3)
            ).astype(np.float64)
            near_industry = (box_sum(industry_mask, buffer) > 0.0).ravel()
            near_sensitive = (box_sum(sensitive_mask, buffer) > 0.0).ravel()
            planning = np.ones((self.n_cells, 5), dtype=bool)
            planning[:, 0:3] &= ~near_industry[:, None]
            planning[:, 3] &= ~near_sensitive
        else:
            planning = np.ones((self.n_cells, 5), dtype=bool)

        action_mask = (
            free_buildable[:, None, None]
            & affordable
            & planning[:, :, None]
        )
        valid = action_mask.any(axis=(1, 2))

        built_fraction = float(built.mean())
        mean_price = float(np.clip(self.price.mean() / MAX_PRICE, 0.0, 1.0))
        globals_ = np.array(
            [
                np.clip(agent.budget / self.config.initial_budget, 0.0, 3.0),
                np.clip(self.round / self.config.actions_per_agent, 0.0, 1.0),
                np.clip(agent.profit / self.config.initial_budget, -1.0, 2.0),
                built_fraction,
                mean_price,
                np.clip(self.config.n_agents / 15.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        return Observation(
            agent_id=agent_id,
            cell_features=f.reshape(self.n_cells, N_CELL_FEATURES),
            global_features=globals_,
            action_mask=action_mask,
            valid_cells=valid,
            cost_table=costs,
            remaining_rounds=max(0, self.config.actions_per_agent - self.round),
            budget=float(agent.budget),
        )

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(self, agent_id: int, action: Action) -> bool:
        agent = self.agents[agent_id]

        if action.cell is None:
            self._record_pass(agent_id)
            return False

        cell = int(action.cell)
        if not 0 <= cell < self.n_cells:
            self._record_pass(agent_id)
            return False
        row, col = divmod(cell, self.w)
        if self.type_grid[row, col] != 0 or not self.buildable[row, col]:
            self._record_pass(agent_id)
            return False
        if not 0 <= action.type_id < 5 or not 0 <= action.quality < 3:
            self._record_pass(agent_id)
            return False
        if self._violates_industry_buffer(row, col, action.type_id):
            self._record_pass(agent_id)
            return False

        cost = self.cost_of(cell, action.type_id, action.quality)
        if cost > agent.budget + 1e-9:
            self._record_pass(agent_id)
            return False

        pre_income = self._cached_income
        pre_built = self.type_grid > 0

        self.type_grid[row, col] = action.type_id + 1
        self.quality_grid[row, col] = action.quality
        self.owner_grid[row, col] = agent_id
        self.cost_grid[row, col] = cost
        agent.budget = max(0.0, agent.budget - cost)
        agent.spend += cost
        agent.build_count += 1

        self._refresh_layers()
        post_income = self._income_grid()
        self._cached_income = post_income
        loss = np.maximum(0.0, pre_income - post_income)
        loss[~pre_built] = 0.0
        total_damage = float(loss.sum())
        agent.damage_caused += total_damage
        if total_damage > 0:
            owners = self.owner_grid[loss > 0]
            values = loss[loss > 0]
            for owner in np.unique(owners):
                if owner >= 0:
                    self.agents[int(owner)].damage_suffered += float(
                        values[owners == owner].sum()
                    )

        self.actions.append(
            ActionRecord(
                round=self.round,
                agent_id=agent_id,
                cell=cell,
                type_id=action.type_id,
                quality=action.quality,
                cost=cost,
                damage=total_damage,
                accepted=True,
            )
        )
        return True

    def _violates_industry_buffer(self, row: int, col: int, action_type: int) -> bool:
        """Hard planning rule: incompatible uses must keep a buffer distance."""
        buffer = self.config.requirements.industry_buffer
        if buffer <= 0:
            return False
        world_type = action_type + 1
        if world_type == 4:
            sensitive = (self.type_grid == 1) | (self.type_grid == 2) | (self.type_grid == 3)
        elif world_type in (1, 2, 3):
            sensitive = self.type_grid == 4
        else:
            return False
        y0 = max(0, row - buffer)
        y1 = min(self.h, row + buffer + 1)
        x0 = max(0, col - buffer)
        x1 = min(self.w, col + buffer + 1)
        return bool(sensitive[y0:y1, x0:x1].any())

    def _record_pass(self, agent_id: int) -> None:
        self.agents[agent_id].skip_count += 1
        self.actions.append(
            ActionRecord(
                round=self.round,
                agent_id=agent_id,
                cell=None,
                type_id=0,
                quality=0,
                cost=0.0,
                damage=0.0,
                accepted=False,
            )
        )

    # ------------------------------------------------------------------
    # Round accounting
    # ------------------------------------------------------------------

    def end_round(self) -> None:
        incomes = self._income_grid()
        maintenance = MAINTENANCE_RATE * self.cost_grid
        for agent in self.agents:
            mask = self.owner_grid == agent.agent_id
            if not mask.any():
                continue
            revenue = float(incomes[mask].sum())
            maint = float(maintenance[mask].sum())
            agent.revenue += revenue
            agent.maintenance += maint
            agent.budget = max(0.0, agent.budget + revenue - maint)

        if self.record_history:
            city = self.city_metrics()
            self.history.append(
                {
                    "round": self.round,
                    "built_cells": city.built_cells,
                    "green_share": city.green_share,
                    "type_entropy": city.type_entropy,
                    "mean_load": city.mean_load,
                    "city_quality": city.city_quality,
                    "damage_total": city.externality_damage_total,
                }
            )
        self.round += 1
        self._cached_income = self._income_grid()

    # ------------------------------------------------------------------
    # Dynamic layers
    # ------------------------------------------------------------------

    def _refresh_layers(self) -> None:
        built = (self.type_grid > 0).astype(np.float64)
        built_bool = built > 0.0

        self._built_count_r1 = box_sum(built, 1) - built
        self._built_count_r2 = box_sum(built, 2) - built
        self.density = np.minimum(np.maximum(self._built_count_r1 / self._neighbor_n1, 0.0), 1.0)

        counts: dict[int, np.ndarray] = {}
        for world_type in range(1, 6):
            mask = (self.type_grid == world_type).astype(np.float64)
            counts[world_type] = box_sum(mask, 2) - mask
        self._type_counts_r2 = counts

        self.share_housing = np.minimum(np.maximum(counts[1] / self._neighbor_n2, 0.0), 1.0)
        self.share_commercial = np.minimum(np.maximum(counts[2] / self._neighbor_n2, 0.0), 1.0)
        self.share_mixed =  np.minimum(np.maximum(counts[3] / self._neighbor_n2, 0.0), 1.0)
        self.share_industry =  np.minimum(np.maximum(counts[4] / self._neighbor_n2, 0.0), 1.0)
        self.share_green =  np.minimum(np.maximum(counts[5] / self._neighbor_n2, 0.0), 1.0)
        max_share = np.maximum.reduce(
            [
                self.share_housing,
                self.share_commercial,
                self.share_mixed,
                self.share_industry,
                self.share_green,
            ]
        )
        self.load =  np.minimum(np.maximum(0.5 * self.density + 0.5 * max_share, 0.0), 1.0)

        quality = np.where(built_bool, self.quality_grid.astype(np.float64), 0.0)
        quality_sum = box_sum(quality, 2) - quality
        self.mean_quality =  np.minimum(np.maximum(
            quality_sum / np.maximum(1.0, self._built_count_r2) / 2.0, 0.0), 1.0
        )

        if self.config.enable_externalities:
            self.demand = np.clip(
                self.static["base_demand"]
                + 0.25 * (1.0 - self.density)
                + 0.10 * self.static["road"]
                - 0.20 * self.load
                - 0.15 * self.share_industry,
                0.10,
                MAX_DEMAND,
            )
            self.price = np.clip(
                0.55 * self.static["land_price"]
                + 0.35 * (self.demand / MAX_DEMAND)
                + 0.10 * (1.0 - self.density),
                0.0,
                MAX_PRICE,
            )
        else:
            self.demand = self.static["base_demand"].copy()
            self.price = self.static["land_price"].copy()

        industry_count = int(np.sum(self.type_grid == 4))
        if industry_count != self._industry_count:
            self._industry_count = industry_count
            self.dist_to_industry = distance_to_mask(self.type_grid == 4)

    # ------------------------------------------------------------------
    # Income and externalities
    # ------------------------------------------------------------------

    def _compatibility_fields(self) -> np.ndarray:
        """Compatibility multiplier for each action type, shape (5, H, W)."""
        counts = self._type_counts_r2
        housing = counts[1]
        commercial = counts[2]
        mixed = counts[3]
        industry = counts[4]
        green = counts[5]
        sensitive = housing + commercial + mixed
        fields = np.empty((5, self.h, self.w), dtype=np.float64)
        for action_type in range(5):
            world_type = action_type + 1
            same = counts[world_type]
            if world_type == 4:
                incompatible = sensitive
                complementary = green
            elif world_type == 1:
                incompatible = industry
                complementary = green + commercial + mixed
            elif world_type == 2:
                incompatible = industry
                complementary = green + housing + mixed
            elif world_type == 3:
                incompatible = industry
                complementary = green + housing + commercial
            else:
                incompatible = np.zeros_like(housing)
                complementary = housing + commercial + mixed
            fields[action_type] = np.clip(
                1.0
                + 0.05 * complementary
                - 0.10 * incompatible
                - 0.03 * same
                + 0.04 * green,
                0.35,
                1.55,
            )
        return fields

    def _income_grid(self) -> np.ndarray:
        built = self.type_grid > 0
        if not built.any():
            return np.zeros((self.h, self.w), dtype=np.float64)

        world_type = self.type_grid.astype(np.int64)
        quality = np.clip(self.quality_grid.astype(np.int64), 0, 2)
        type_for_index = np.clip(world_type - 1, 0, 4)

        type_income = INCOME_BASE[world_type]
        quality_income = QUALITY_INCOME_MULT[quality]
        demand_factor = 0.5 + 0.7 * (self.demand / MAX_DEMAND)
        road_factor = 0.7 + 0.6 * self.static["road"]
        access_fit = 1.0 - np.abs(
            self.static["dist_center"] - OPTIMAL_DISTANCE[world_type]
        )
        access_factor = 0.6 + 0.8 * np.clip(access_fit, 0.0, 1.0)
        soil_factor = 0.9 + 0.2 * self.static["soil"]

        if self.config.enable_externalities:
            compatibility = self._compatibility_fields()[type_for_index, self.yy.astype(int), self.xx.astype(int)]
            load_factor = 1.0 - 0.10 * self.load
        else:
            compatibility = np.ones((self.h, self.w), dtype=np.float64)
            load_factor = np.ones((self.h, self.w), dtype=np.float64)

        income = (
            type_income
            * quality_income
            * demand_factor
            * road_factor
            * access_factor
            * compatibility
            * load_factor
            * soil_factor
        )
        income[~built] = 0.0
        return np.maximum(income, 0.0)

    def projected_income_table(self, agent_id: int) -> np.ndarray:
        """Expected one-round income for building in every empty cell.

        Shape: (n_cells, 5 action types, 3 qualities).  Used by the greedy
        baseline; learned policies never call this method.
        """
        fields = self._compatibility_fields() if self.config.enable_externalities else None
        if self.config.enable_externalities:
            load_factor = 1.0 - 0.10 * self.load
            compatibility = fields
        else:
            load_factor = np.ones((self.h, self.w), dtype=np.float64)
            compatibility = np.ones((5, self.h, self.w), dtype=np.float64)

        demand_factor = 0.5 + 0.7 * (self.demand / MAX_DEMAND)
        road_factor = 0.7 + 0.6 * self.static["road"]
        soil_factor = 0.9 + 0.2 * self.static["soil"]

        out = np.zeros((self.n_cells, 5, 3), dtype=np.float64)
        for action_type in range(5):
            world_type = action_type + 1
            access_fit = 1.0 - np.abs(
                self.static["dist_center"] - OPTIMAL_DISTANCE[world_type]
            )
            access_factor = 0.6 + 0.8 * np.clip(access_fit, 0.0, 1.0)
            base = INCOME_BASE[world_type] * (
                demand_factor * road_factor * access_factor * soil_factor
            )
            for quality in range(3):
                out[:, action_type, quality] = (
                    base * QUALITY_INCOME_MULT[quality]
                    * compatibility[action_type]
                    * load_factor
                ).ravel()
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _agent_violations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-agent (violations, incompatible neighbours, density excess)."""
        type_grid = self.type_grid
        counts = self._type_counts_r2
        sensitive = counts[1] + counts[2] + counts[3]
        per_cell_incompatible = np.where(
            type_grid == 4,
            sensitive,
            np.where((type_grid == 1) | (type_grid == 2) | (type_grid == 3), counts[4], 0.0),
        )
        built = type_grid > 0
        built_count = int(built.sum())
        green_count = int(np.sum(type_grid == 5))
        green_share = green_count / max(1, built_count)
        green_deficit = max(0.0, self.config.requirements.green_min - green_share)
        cap = self.config.requirements.density_cap
        density_excess_cell = np.maximum(0.0, self.density - cap)

        n = len(self.agents)
        violations = np.zeros(n, dtype=np.float64)
        incompatible = np.zeros(n, dtype=np.float64)
        density_excess = np.zeros(n, dtype=np.float64)
        for agent in self.agents:
            mask = self.owner_grid == agent.agent_id
            own_builds = int(mask.sum())
            if own_builds == 0:
                continue
            incompatible[agent.agent_id] = float(per_cell_incompatible[mask].sum())
            density_excess[agent.agent_id] = float(density_excess_cell[mask].sum())
            deficit_share = green_deficit * (own_builds / max(1, built_count))
            violations[agent.agent_id] = (
                incompatible[agent.agent_id]
                + 4.0 * deficit_share
                + 4.0 * density_excess[agent.agent_id]
            )
        return violations, incompatible, density_excess

    def city_metrics(self) -> CityMetrics:
        type_grid = self.type_grid
        built = type_grid > 0
        built_count = int(built.sum())
        counts = {t: int(np.sum(type_grid == t)) for t in range(1, 6)}
        green_share = counts[5] / max(1, built_count)
        entropy = type_entropy(type_grid)
        if built_count:
            accessibility = 1.0 - float(self.static["dist_center"][built].mean())
            mean_load = float(self.load[built].mean())
        else:
            accessibility = 0.0
            mean_load = 0.0
        inc_edges, total_edges = incompatible_edges(type_grid)
        compatibility = 1.0 - inc_edges / max(1, total_edges)
        green_component = float(
            np.clip(1.0 - abs(green_share - GREEN_TARGET) / GREEN_TARGET, 0.0, 1.0)
        )
        city_quality = float(
            np.clip(
                0.30 * green_component
                + 0.25 * entropy
                + 0.20 * accessibility
                + 0.25 * compatibility
                - 0.20 * mean_load,
                0.0,
                1.0,
            )
        )
        violations, _, _ = self._agent_violations()
        damage_total = float(sum(a.damage_caused for a in self.agents))
        profits = np.array([a.profit for a in self.agents], dtype=np.float64)
        shifted = profits - profits.min() if profits.size else profits
        return CityMetrics(
            green_share=float(green_share),
            green_component=green_component,
            type_entropy=float(entropy),
            accessibility=float(accessibility),
            compatibility=float(compatibility),
            mean_load=float(mean_load),
            city_quality=city_quality,
            violations_total=float(violations.sum()),
            externality_damage_total=damage_total,
            externality_damage_per_agent=damage_total / max(1, len(self.agents)),
            share_agents_causing_damage=float(
                np.mean([a.damage_caused > 1e-9 for a in self.agents])
            ),
            gini_profit=float(gini_from_values(shifted)),
            built_cells=built_count,
            type_counts=counts,
        )

    def agent_metrics(self) -> list[AgentMetrics]:
        violations, incompatible, density_excess = self._agent_violations()
        built = self.type_grid > 0
        built_count = int(built.sum())
        green_share = np.sum(self.type_grid == 5) / max(1, built_count)
        green_deficit = max(0.0, self.config.requirements.green_min - green_share)
        metrics: list[AgentMetrics] = []
        for agent in self.agents:
            mask = self.owner_grid == agent.agent_id
            own_builds = int(mask.sum())
            if own_builds:
                mean_distance = float(self.static["dist_center"][mask].mean())
                high_quality = float(np.mean(self.quality_grid[mask] == 2))
                own_types = self.type_grid[mask]
                counts = np.array(
                    [np.sum(own_types == t) for t in range(1, 6)], dtype=np.float64
                )
                total = counts.sum()
                p = counts[counts > 0] / total
                own_entropy = max(0.0, float(-np.sum(p * np.log(p)) / np.log(5.0)))
            else:
                mean_distance = 1.0
                high_quality = 0.0
                own_entropy = 0.0
            deficit_share = green_deficit * (own_builds / max(1, built_count))
            metrics.append(
                AgentMetrics(
                    agent_id=agent.agent_id,
                    profit=agent.profit,
                    revenue=agent.revenue,
                    spend=agent.spend,
                    maintenance=agent.maintenance,
                    budget=agent.budget,
                    builds=agent.build_count,
                    skips=agent.skip_count,
                    damage_caused=agent.damage_caused,
                    damage_suffered=agent.damage_suffered,
                    violations=float(violations[agent.agent_id]),
                    incompatible_neighbors=float(incompatible[agent.agent_id]),
                    green_deficit_share=float(deficit_share),
                    density_excess=float(density_excess[agent.agent_id]),
                    mean_distance=mean_distance,
                    high_quality_fraction=high_quality,
                    type_entropy=own_entropy,
                    behavior=np.array([mean_distance, high_quality, own_entropy], dtype=np.float64),
                )
            )
        return metrics


def gini_from_values(values: np.ndarray) -> float:
    """Gini for non-negative values; zero for constant arrays."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0.0, None)
    if np.allclose(x, 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1.0) / n)
