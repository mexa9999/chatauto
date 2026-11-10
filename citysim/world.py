"""Multi-agent city environment.

The environment owns all map layers, validates actions, updates the dynamic
layers and attributes externality damage to the agent whose action caused it.
Policies only see :class:`Observation` objects produced by :meth:`observe`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .citizens import CitizenPool
from .config import (
    ACTION_TYPE_NAMES,
    BANKRUPTCY_STREAK,
    BANKRUPTCY_THRESHOLD,
    BUILD_COST,
    CITIZEN_GOODS_DEMAND,
    COMMERCIAL_CUSTOMERS,
    COMMERCIAL_GOODS_DEMAND,
    COMMERCIAL_JOBS,
    DISTRICT_SIZE,
    GREEN_TARGET,
    HOUSING_CAPACITY,
    INDUSTRY_GOODS_SUPPLY,
    INDUSTRY_JOBS,
    INCOME_BASE,
    LAND_VALUE_MAX,
    LAND_VALUE_MIN,
    MAINTENANCE_RATE,
    MAX_DEMAND,
    MAX_PRICE,
    MIXED_CAPACITY,
    MIXED_JOBS,
    N_CELL_FEATURES,
    N_CITIZENS,
    N_GLOBAL_FEATURES,
    NOISE_RADIUS,
    OPTIMAL_DISTANCE,
    POLLUTION_RADIUS,
    POPULATION_INERTIA,
    QUALITY_COST_MULT,
    QUALITY_INCOME_MULT,
    SATURATION_ALPHA,
    SATURATION_RADIUS,
    TYPE_NAMES,
    CityConfig,
)
from .grid import (
    box_mean,
    box_sum,
    distance_to_mask,
    district_ids,
    incompatible_edges,
    make_static_layers,
    type_entropy,
)


@dataclass(frozen=True)
class Action:
    """A build or demolition action.

    ``type_id`` is a policy index 0..4 (not a world id).  ``bid`` is the
    auction payment for the plot, and ``demolish`` selects demolition mode.
    """

    cell: int | None
    type_id: int = 0
    quality: int = 0
    cost: float = 0.0
    bid: float = 0.0
    demolish: bool = False
    priority: int = 0

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
    bid: float = 0.0
    demolish: bool = False


@dataclass
class AgentState:
    agent_id: int
    initial_budget: float
    budget: float = field(init=False)
    revenue: float = 0.0
    spend: float = 0.0
    maintenance: float = 0.0
    tax_paid: float = 0.0
    subsidy_received: float = 0.0
    bid_paid: float = 0.0
    refund_received: float = 0.0
    damage_caused: float = 0.0
    damage_suffered: float = 0.0
    build_count: int = 0
    skip_count: int = 0
    bankrupt_streak: int = 0
    bankrupt: bool = False

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
    demolish_mask: np.ndarray | None = None  # (n_cells,) bool
    demolish_value: np.ndarray | None = None  # (n_cells,) estimated value vs. retention


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
    tax_paid: float = 0.0
    subsidy_received: float = 0.0
    bid_paid: float = 0.0
    refund_received: float = 0.0
    bankrupt: bool = False
    affordable_share: float = 0.0
    green_share: float = 0.0


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
    population: float = 0.0
    jobs: float = 0.0
    goods_supply: float = 0.0
    goods_demand: float = 0.0
    goods_balance: float = 0.0
    mean_utilization: float = 0.0
    land_value_mean: float = 0.0
    pollution_mean: float = 0.0
    noise_mean: float = 0.0
    mean_satisfaction: float = 0.0
    gini_satisfaction: float = 0.0
    rich_satisfaction: float = 0.0
    poor_satisfaction: float = 0.0
    tax_revenue: float = 0.0
    subsidy_total: float = 0.0
    bankrupt_agents: int = 0
    affordable_share: float = 0.0


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
        self.district_labels, self.n_districts = district_ids(
            self.h, self.w, DISTRICT_SIZE
        )
        self.district_labels_flat = self.district_labels.ravel()

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
        self.shock_events: list[dict] = []

        self.base_population = float(N_CITIZENS)
        self.population_extra = 0.0
        self.population = 0.0
        self.jobs = 0.0
        self.goods_supply = 0.0
        self.goods_demand = 0.0
        self.goods_balance = 0.0
        self.utilization: dict[int, float] = {t: 0.0 for t in range(1, 6)}
        self.util_grid = np.zeros((self.h, self.w), dtype=np.float64)
        self.mean_utilization = 0.0
        self.district_util = np.zeros((self.h, self.w), dtype=np.float64)
        self.district_type = np.zeros((self.h, self.w), dtype=np.float64)
        self.land_value = np.zeros((self.h, self.w), dtype=np.float64)
        self.pollution = np.zeros((self.h, self.w), dtype=np.float64)
        self.noise = np.zeros((self.h, self.w), dtype=np.float64)
        self.attractiveness = np.zeros((self.h, self.w), dtype=np.float64)
        self.local_saturation = np.zeros((self.h, self.w), dtype=np.float64)
        self.local_same: dict[int, np.ndarray] = {
            t: np.zeros((self.h, self.w), dtype=np.float64) for t in range(1, 6)
        }

        ones = np.ones((self.h, self.w), dtype=np.float64)
        self._window_n1 = box_sum(ones, 1)
        self._window_n2 = box_sum(ones, 2)
        self._neighbor_n1 = np.maximum(1.0, self._window_n1 - 1.0)
        self._neighbor_n2 = np.maximum(1.0, self._window_n2 - 1.0)

        self._industry_count = -1
        self.dist_to_industry = np.ones((self.h, self.w), dtype=np.float64)
        self._cluster_labels: dict[int, np.ndarray] = {}
        self.citizens = CitizenPool(seed=self.seed + 7919)
        self._cached_compat = None
        self._income_cache = None
        self._current_income = None
        self._refresh_layers(advance_population=False)

    # ------------------------------------------------------------------
    # Observation and cost tables
    # ------------------------------------------------------------------

    def cost_table(self) -> np.ndarray:
        """Build costs for every (cell, action type, quality)."""
        land_mult = (0.4 + 1.6 * self.land_value).ravel()
        soil_mult = (0.9 + 0.2 * self.static["soil"]).ravel()
        cell_mult = (land_mult * soil_mult)[:, None, None]
        return (
            BUILD_COST[1:][None, :, None]
            * QUALITY_COST_MULT[None, None, :]
            * cell_mult
        )

    def cost_of(self, cell: int, type_id: int, quality: int) -> float:
        c = int(np.clip(cell, 0, self.n_cells - 1))
        land = float(self.land_value.ravel()[c])
        soil = float(self.static["soil"].ravel()[c])
        return float(
            BUILD_COST[type_id + 1]
            * QUALITY_COST_MULT[quality]
            * (0.4 + 1.6 * land)
            * (0.9 + 0.2 * soil)
        )

    def observe(self, agent_id: int) -> Observation:
        built = self.type_grid > 0
        free = ~built
        agent = self.agents[agent_id]

        own = (self.owner_grid == agent_id).astype(np.float64)
        own_count = box_sum(own, 2) - own
        competition = (self._built_count_r2 - own_count) / self._neighbor_n2

        f = np.empty((self.h, self.w, N_CELL_FEATURES), dtype=np.float64)
        f[..., 0] = self.price / MAX_PRICE
        f[..., 1] = self.static["soil"]
        f[..., 2] = self.static["water"]
        f[..., 3] = self.static["road"]
        f[..., 4] = self.static["dist_center"]
        f[..., 5] = self.density
        f[..., 6] = self.share_housing
        f[..., 7] = self.share_commercial
        f[..., 8] = self.share_industry
        f[..., 9] = self.mean_quality
        f[..., 10] = self.load
        f[..., 11] = self.demand / MAX_DEMAND
        f[..., 12] = self.buildable.astype(np.float64)
        f[..., 13] = free.astype(np.float64)
        f[..., 14] = competition
        f[..., 15] = self.dist_to_industry
        f[..., 16] = self.land_value
        f[..., 17] = self.pollution
        f[..., 18] = self.noise
        f[..., 19] = self.attractiveness
        f[..., 20] = self.local_saturation
        f[..., 21] = self.district_type / 5.0
        f[..., 22] = self.district_util

        if not self.config.enable_externalities:
            # Ablation: the policy does not see any neighbour-dependent signal.
            f[..., 5:12] = 0.0
            f[..., 14] = 0.0
            f[..., 15] = 0.0
            f[..., 17] = 0.0
            f[..., 18] = 0.0
            f[..., 19] = 0.0
            f[..., 20] = 0.0
            f[..., 22] = 0.0

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
        if agent.bankrupt:
            action_mask[:] = False
        valid = action_mask.any(axis=(1, 2))
        demolish_mask = (self.owner_grid == agent_id) & (self.type_grid > 0)
        if agent.bankrupt:
            demolish_mask[:] = False
        costs_flat = self.cost_grid.ravel()
        income_flat = self._income_grid().ravel()
        type_flat = self.type_grid.ravel()
        owned_cells = np.flatnonzero(demolish_mask.ravel())
        if owned_cells.size:
            owned_types = type_flat[owned_cells]
            unutilized = np.array(
                [self.utilization[int(world_type)] <= 0.0 for world_type in owned_types]
            )
            if unutilized.any():
                projected_income = self.projected_income_table(agent_id)
                unutilized_cells = owned_cells[unutilized]
                income_flat[unutilized_cells] = projected_income[
                    unutilized_cells,
                    owned_types[unutilized] - 1,
                    self.quality_grid.ravel()[unutilized_cells],
                ]
        regulation = self.config.regulation
        net_income = (
            income_flat
            - MAINTENANCE_RATE * costs_flat
            - regulation.property_tax_rate * costs_flat
            - regulation.industry_tax_rate * income_flat * (type_flat == 4)
            + regulation.green_subsidy * (type_flat == 5)
        )
        demolish_value = np.where(
            net_income < 0.0,
            0.5 * costs_flat
            - net_income * max(0, self.config.actions_per_agent - self.round),
            0.0,
        )
        demolish_value[~demolish_mask.ravel()] = 0.0

        built_fraction = float(built.mean())
        mean_price = float(np.clip(self.price.mean() / MAX_PRICE, 0.0, 1.0))
        util_values = [self.utilization[t] for t in range(1, 6) if self.utilization[t] > 0]
        globals_ = np.array(
            [
                np.clip(agent.budget / self.config.initial_budget, 0.0, 3.0),
                np.clip(self.round / self.config.actions_per_agent, 0.0, 1.0),
                np.clip(agent.profit / self.config.initial_budget, -1.0, 2.0),
                built_fraction,
                mean_price,
                np.clip(self.config.n_agents / 15.0, 0.0, 1.0),
                np.clip(self.population / self.base_population, 0.0, 1.5),
                np.clip(self.jobs / max(1.0, self.population), 0.0, 2.0) / 2.0,
                np.clip(self.goods_balance, -1.0, 1.0),
                float(np.mean(util_values)) if util_values else 0.0,
            ],
            dtype=np.float64,
        )
        obs = Observation(
            agent_id=agent_id,
            cell_features=f.reshape(self.n_cells, N_CELL_FEATURES),
            global_features=globals_,
            action_mask=action_mask,
            valid_cells=valid,
            cost_table=costs,
            remaining_rounds=max(0, self.config.actions_per_agent - self.round),
            budget=float(agent.budget),
            demolish_mask=demolish_mask.ravel(),
            demolish_value=demolish_value,
        )
        # Private shape metadata used by the neural policy's grid convolutions.
        obs._shape_hint = (self.h, self.w)  # type: ignore[attr-defined]
        obs._district_labels = self.district_labels  # type: ignore[attr-defined]
        obs._n_districts = self.n_districts  # type: ignore[attr-defined]
        return obs

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(self, agent_id: int, action: Action) -> bool:
        agent = self.agents[agent_id]
        if agent.bankrupt:
            self._record_pass(agent_id)
            return False

        if action.cell is None:
            self._record_pass(agent_id)
            return False

        cell = int(action.cell)
        if not 0 <= cell < self.n_cells:
            self._record_pass(agent_id)
            return False
        row, col = divmod(cell, self.w)

        if action.demolish:
            if self.owner_grid[row, col] != agent_id or self.type_grid[row, col] == 0:
                self._record_pass(agent_id)
                return False
            refund = 0.5 * float(self.cost_grid[row, col])
            self.type_grid[row, col] = 0
            self.quality_grid[row, col] = -1
            self.owner_grid[row, col] = -1
            self.cost_grid[row, col] = 0.0
            agent.budget += refund
            agent.refund_received += refund
            self._refresh_layers(advance_population=False)
            self.actions.append(
                ActionRecord(
                    round=self.round,
                    agent_id=agent_id,
                    cell=cell,
                    type_id=-1,
                    quality=-1,
                    cost=0.0,
                    damage=0.0,
                    accepted=True,
                    bid=0.0,
                    demolish=True,
                )
            )
            return True

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
        bid = max(0.0, float(action.bid))
        if cost + bid > agent.budget + 1e-9:
            self._record_pass(agent_id)
            return False

        pre_income = self._income_grid()
        pre_built = self.type_grid > 0

        self.type_grid[row, col] = action.type_id + 1
        self.quality_grid[row, col] = action.quality
        self.owner_grid[row, col] = agent_id
        self.cost_grid[row, col] = cost
        agent.budget -= cost + bid
        agent.spend += cost
        agent.bid_paid += bid
        agent.build_count += 1

        self._refresh_layers(advance_population=False)
        post_income = self._income_grid()
        self._current_income = post_income  # <--- СОХРАНЯЕМ ДЛЯ СЛЕДУЮЩЕГО ДЕЙСТВИЯ
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
                bid=bid,
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
        self._refresh_layers(advance_population=True)
        incomes = self._income_grid()
        maintenance = MAINTENANCE_RATE * self.cost_grid
        regulation = self.config.regulation
        for agent in self.agents:
            mask = self.owner_grid == agent.agent_id
            if not mask.any():
                continue
            revenue = float(incomes[mask].sum())
            maint = float(maintenance[mask].sum())
            property_tax = float(regulation.property_tax_rate * self.cost_grid[mask].sum())
            industry_tax = float(
                regulation.industry_tax_rate
                * incomes[mask & (self.type_grid == 4)].sum()
            )
            green_count = int(np.sum(mask & (self.type_grid == 5)))
            subsidy = float(regulation.green_subsidy * green_count)
            agent.revenue += revenue
            agent.maintenance += maint
            agent.tax_paid += property_tax + industry_tax
            agent.subsidy_received += subsidy
            agent.budget += revenue - maint - property_tax - industry_tax + subsidy

        self._process_bankruptcies()

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
                    "population": city.population,
                    "jobs": city.jobs,
                    "goods_balance": city.goods_balance,
                    "mean_utilization": city.mean_utilization,
                    "mean_satisfaction": city.mean_satisfaction,
                    "gini_satisfaction": city.gini_satisfaction,
                    "rich_satisfaction": city.rich_satisfaction,
                    "poor_satisfaction": city.poor_satisfaction,
                    "gini_profit": city.gini_profit,
                    "violations": city.violations_total,
                    "tax_revenue": city.tax_revenue,
                    "subsidy_total": city.subsidy_total,
                    "bankrupt_agents": city.bankrupt_agents,
                }
            )
        self.round += 1

    def _process_bankruptcies(self) -> None:
        threshold = BANKRUPTCY_THRESHOLD
        any_bankrupt = False
        for agent in self.agents:
            if agent.bankrupt:
                continue
            if agent.budget < threshold:
                agent.bankrupt_streak += 1
            else:
                agent.bankrupt_streak = 0
            if agent.bankrupt_streak >= BANKRUPTCY_STREAK:
                agent.bankrupt = True
                mask = self.owner_grid == agent.agent_id
                self.type_grid[mask] = 0
                self.quality_grid[mask] = -1
                self.owner_grid[mask] = -1
                self.cost_grid[mask] = 0.0
                agent.budget = 0.0
                any_bankrupt = True
        if any_bankrupt:
            self._refresh_layers(advance_population=False)

    # ------------------------------------------------------------------
    # Dynamic layers
    # ------------------------------------------------------------------

    def _refresh_layers(self, advance_population: bool = False) -> None:
        self._cached_compat = None  # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        self._income_cache = None   # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        built = (self.type_grid > 0).astype(np.float64)
        built_bool = built > 0.0

        self._built_count_r1 = box_sum(built, 1) - built
        self._built_count_r2 = box_sum(built, 2) - built
        self.density = self._built_count_r1 / self._neighbor_n1

        counts: dict[int, np.ndarray] = {}
        for world_type in range(1, 6):
            mask = (self.type_grid == world_type).astype(np.float64)
            counts[world_type] = box_sum(mask, 2) - mask
        self._type_counts_r2 = counts

        self.share_housing = counts[1] / self._neighbor_n2
        self.share_commercial = counts[2] / self._neighbor_n2
        self.share_mixed = counts[3] / self._neighbor_n2
        self.share_industry = counts[4] / self._neighbor_n2
        self.share_green = counts[5] / self._neighbor_n2
        max_share = np.maximum.reduce(
            [
                self.share_housing,
                self.share_commercial,
                self.share_mixed,
                self.share_industry,
                self.share_green,
            ]
        )
        self.load = 0.5 * self.density + 0.5 * max_share

        quality = np.where(built_bool, self.quality_grid.astype(np.float64), 0.0)
        quality_sum = box_sum(quality, 2) - quality
        self.mean_quality = quality_sum / np.maximum(1.0, self._built_count_r2) / 2.0

        # Local saturation around every cell.
        for world_type in range(1, 6):
            mask = (self.type_grid == world_type).astype(np.float64)
            self.local_same[world_type] = (
                box_sum(mask, SATURATION_RADIUS) - mask
            ).astype(np.float64)
        self.local_saturation = box_sum(built, SATURATION_RADIUS) / np.maximum(
            1.0, box_sum(np.ones((self.h, self.w), dtype=np.float64), SATURATION_RADIUS)
        )

        industry_mask = (self.type_grid == 4).astype(np.float64)
        industry_count = int(industry_mask.sum())
        if industry_count != self._industry_count:
            self._industry_count = industry_count
            self.dist_to_industry = distance_to_mask(industry_mask > 0)
        road_mask = self.static["road_mask"]
        industry_dist = (
            distance_to_mask(industry_mask > 0)
            if industry_count
            else np.ones((self.h, self.w), dtype=np.float64)
        )
        self.pollution = (
            0.80 * np.exp(-6.0 * industry_dist)
            + 0.15 * box_mean(road_mask, POLLUTION_RADIUS)
        )
        commercial_mask = (self.type_grid == 2).astype(np.float64)
        self.noise = (
            0.45 * box_mean(industry_mask + commercial_mask, NOISE_RADIUS)
            + 0.45 * box_mean(road_mask, NOISE_RADIUS)
            + 0.10 * self.density
        )
        self.attractiveness = (
            0.32 * box_mean((self.type_grid == 5).astype(np.float64), 1)
            + 0.20 * self.static["water"]
            + 0.20 * (1.0 - self.static["dist_center"])
            + 0.15 * self.static["road"]
            + 0.13 * (1.0 - self.density)
            - 0.30 * self.pollution
            - 0.15 * self.noise
        )

        raw_land = (
            0.38 * self.static["land_price"]
            + 0.18 * (1.0 - self.static["dist_center"])
            + 0.14 * self.static["road"]
            + 0.15 * self.attractiveness
            + 0.10 * self.density
            - 0.20 * self.pollution
            - 0.12 * self.noise
        )
        raw_land = np.minimum(np.maximum(raw_land, LAND_VALUE_MIN), LAND_VALUE_MAX)
        self.land_value = (raw_land - LAND_VALUE_MIN) / max(
            1e-9, LAND_VALUE_MAX - LAND_VALUE_MIN
        )

        if self.config.enable_externalities:
            self.demand =np.minimum(np.maximum(
                self.static["base_demand"]
                + 0.25 * (1.0 - self.density)
                + 0.10 * self.static["road"]
                - 0.20 * self.load
                - 0.15 * self.share_industry
                + 0.15 * self.attractiveness
                - 0.10 * self.local_saturation,
                0.10),
                MAX_DEMAND,
            )
            self.price = np.minimum(np.maximum(
                0.55 * self.static["land_price"]
                + 0.35 * (self.demand / MAX_DEMAND)
                + 0.10 * (1.0 - self.density)
                + 0.10 * self.land_value
                - 0.10 * self.pollution,
                0.0),
                MAX_PRICE,
            )
        else:
            self.demand = self.static["base_demand"].copy()
            self.price = self.static["land_price"].copy()

        self._update_stocks(advance_population=advance_population)
        self._refresh_districts()

    def _update_stocks(self, advance_population: bool) -> None:
        counts = {
            t: int(np.sum(self.type_grid == t)) for t in range(1, 6)
        }
        built_count = sum(counts.values())
        housing_capacity = counts[1] * HOUSING_CAPACITY + counts[3] * MIXED_CAPACITY
        target_population = min(
            self.base_population + self.population_extra, housing_capacity
        )
        target_population = max(0.0, target_population)
        if advance_population:
            self.population = (
                POPULATION_INERTIA * self.population
                + (1.0 - POPULATION_INERTIA) * target_population
            )
        else:
            self.population = min(self.population, target_population)

        self.jobs = (
            counts[2] * COMMERCIAL_JOBS
            + counts[3] * MIXED_JOBS
            + counts[4] * INDUSTRY_JOBS
        )
        self.goods_supply = counts[4] * INDUSTRY_GOODS_SUPPLY
        self.goods_demand = (
            CITIZEN_GOODS_DEMAND * self.population
            + counts[2] * COMMERCIAL_GOODS_DEMAND
        )
        self.goods_balance = float(
            (self.goods_supply - self.goods_demand)
            / max(1.0, self.goods_supply + self.goods_demand)
        )

        housing_capacity = max(1e-9, counts[1] * HOUSING_CAPACITY)
        commercial_capacity = max(1e-9, counts[2] * COMMERCIAL_CUSTOMERS)
        mixed_capacity = max(1e-9, counts[3] * MIXED_CAPACITY)
        self.utilization = {
            1: min(1.0, self.population / housing_capacity) if counts[1] else 0.0,
            2: min(1.0, self.population / commercial_capacity) if counts[2] else 0.0,
            3: min(1.0, self.population / mixed_capacity) if counts[3] else 0.0,
            4: min(1.0, self.goods_demand / max(1e-9, self.goods_supply))
            if counts[4]
            else 0.0,
            5: 1.0 if counts[5] else 0.0,
        }
        self.util_grid = np.zeros((self.h, self.w), dtype=np.float64)
        for world_type in range(1, 6):
            self.util_grid[self.type_grid == world_type] = self.utilization[world_type]
        self.mean_utilization = (
            float(self.util_grid[self.type_grid > 0].mean()) if built_count else 0.0
        )

    def _refresh_districts(self) -> None:
        labels = self.district_labels_flat
        built = (self.type_grid > 0).ravel()
        built_counts = np.bincount(
            labels, weights=built.astype(np.float64), minlength=self.n_districts
        )
        # Dominant building type per district.
        type_scores = np.zeros((self.n_districts, 6), dtype=np.float64)
        for world_type in range(1, 6):
            type_scores[:, world_type] = np.bincount(
                labels,
                weights=(self.type_grid.ravel() == world_type).astype(np.float64),
                minlength=self.n_districts,
            )
        dominant = np.argmax(type_scores, axis=1)
        self.district_type = dominant[labels].reshape(self.h, self.w).astype(np.float64)
        util_sums = np.bincount(
            labels, weights=self.util_grid.ravel(), minlength=self.n_districts
        )
        district_util = util_sums / np.maximum(1.0, built_counts)
        self.district_util = district_util[labels].reshape(self.h, self.w)

    # ------------------------------------------------------------------
    # Income and externalities
    # ------------------------------------------------------------------

    def _compatibility_fields(self) -> np.ndarray:
        """Compatibility multiplier for each action type, shape (5, H, W)."""
        if self._cached_compat is not None:
            return self._cached_compat
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
            fields[action_type] = np.minimum(np.maximum(
                1.0 + 0.05 * complementary - 0.10 * incompatible - 0.03 * same + 0.04 * green,
                0.35), 1.55)
        self._cached_compat = fields
        return fields

    def _sector_quality_penalty(self, action_type: int) -> np.ndarray:
        if action_type == 0:      # housing
            return 1.0 - 0.30 * self.pollution - 0.15 * self.noise
        if action_type == 1:      # commercial
            return 1.0 - 0.15 * self.noise - 0.10 * self.pollution
        if action_type == 2:      # mixed
            return 1.0 - 0.20 * self.pollution - 0.12 * self.noise
        if action_type == 3:      # industry
            return 1.0 - 0.05 * self.noise
        return np.ones((self.h, self.w), dtype=np.float64)

    def _income_grid(self) -> np.ndarray:
        if self._income_cache is not None:
            return self._income_cache
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
        access_factor = 0.6 + 0.8 * access_fit
        soil_factor = 0.9 + 0.2 * self.static["soil"]

        if self.config.enable_externalities:
            compatibility = self._compatibility_fields()[
                type_for_index, self.yy.astype(int), self.xx.astype(int)
            ]
            load_factor = 1.0 - 0.10 * self.load
            penalties = np.ones((self.h, self.w), dtype=np.float64)
            for action_type in range(5):
                mask = world_type == action_type + 1
                if mask.any():
                    penalties[mask] = self._sector_quality_penalty(action_type)[mask]
            util = np.zeros((self.h, self.w), dtype=np.float64)
            saturation = np.ones((self.h, self.w), dtype=np.float64)
            for action_type in range(5):
                wt = action_type + 1
                mask = world_type == wt
                util[mask] = self.utilization[wt]
                saturation[mask] = 1.0 / (
                    1.0 + SATURATION_ALPHA * self.local_same[wt][mask]
                )
        else:
            compatibility = np.ones((self.h, self.w), dtype=np.float64)
            load_factor = np.ones((self.h, self.w), dtype=np.float64)
            penalties = np.ones((self.h, self.w), dtype=np.float64)
            util = np.ones((self.h, self.w), dtype=np.float64)
            saturation = np.ones((self.h, self.w), dtype=np.float64)

        income = (
            type_income
            * quality_income
            * demand_factor
            * road_factor
            * access_factor
            * compatibility
            * load_factor
            * penalties
            * util
            * saturation
            * soil_factor
        )
        income[~built] = 0.0
        self._income_cache = np.maximum(income, 0.0)
        return self._income_cache

    def projected_income_table(self, agent_id: int) -> np.ndarray:
        """Expected one-round income for building in every empty cell.

        Shape: (n_cells, 5 action types, 3 qualities).  Used by the greedy
        baseline; learned policies never call this method.
        """
        del agent_id
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
            access_factor = 0.6 + 0.8 * access_fit
            base = INCOME_BASE[world_type] * (
                demand_factor * road_factor * access_factor * soil_factor
            )
            if self.config.enable_externalities:
                sector_penalty = self._sector_quality_penalty(action_type)
                util = self.utilization[world_type]
                if util <= 0.0:
                    util = 1.0  # a new sector starts with latent demand
                saturation = 1.0 / (
                    1.0 + SATURATION_ALPHA * self.local_same[world_type]
                )
            else:
                sector_penalty = 1.0
                util = 1.0
                saturation = 1.0
            for quality in range(3):
                out[:, action_type, quality] = (
                    base
                    * QUALITY_INCOME_MULT[quality]
                    * compatibility[action_type]
                    * load_factor
                    * sector_penalty
                    * util
                    * saturation
                ).ravel()
        return out

    # ------------------------------------------------------------------
    # Shocks
    # ------------------------------------------------------------------

    def apply_shock(self, kind: str, rng: np.random.Generator, magnitude: float = 0.5) -> dict:
        """Apply an exogenous shock and return a short description."""
        cleared = 0
        if kind == "industry_removal":
            cells = np.flatnonzero(self.type_grid == 4)
            if cells.size:
                n = max(1, int(round(magnitude * cells.size)))
                chosen = rng.choice(cells, size=n, replace=False)
                for cell in chosen:
                    row, col = divmod(int(cell), self.w)
                    self._clear_cell(row, col)
                    cleared += 1
        elif kind == "population_boom":
            self.population_extra += float(magnitude)
            self._refresh_layers(advance_population=False)
            self.shock_events.append(
                {"round": self.round, "kind": kind, "magnitude": float(magnitude), "cleared": 0}
            )
            return self.shock_events[-1]
        elif kind == "earthquake":
            cells = np.flatnonzero(self.type_grid > 0)
            if cells.size:
                n = max(1, int(round(magnitude * cells.size)))
                chosen = rng.choice(cells, size=n, replace=False)
                for cell in chosen:
                    row, col = divmod(int(cell), self.w)
                    self._clear_cell(row, col)
                    cleared += 1
        else:
            raise ValueError(f"unknown shock kind: {kind}")
        self._refresh_layers(advance_population=False)
        event = {
            "round": self.round,
            "kind": kind,
            "magnitude": float(magnitude),
            "cleared": int(cleared),
        }
        self.shock_events.append(event)
        return event

    def _clear_cell(self, row: int, col: int) -> None:
        self.type_grid[row, col] = 0
        self.quality_grid[row, col] = -1
        self.owner_grid[row, col] = -1
        self.cost_grid[row, col] = 0.0

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
        quota = self.config.regulation.affordable_housing_quota
        affordable_count = int(
            np.sum((type_grid == 1) & (self.quality_grid == 0))
        )
        affordable_share = affordable_count / max(1, built_count)
        affordable_deficit = max(0.0, quota - affordable_share)

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
            affordable_share_agent = affordable_deficit * (
                own_builds / max(1, built_count)
            )
            violations[agent.agent_id] = (
                incompatible[agent.agent_id]
                + 4.0 * deficit_share
                + 4.0 * density_excess[agent.agent_id]
                + 4.0 * affordable_share_agent
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
            land_value_mean = float(self.land_value[built].mean())
            pollution_mean = float(self.pollution[built].mean())
            noise_mean = float(self.noise[built].mean())
            affordable_share = float(
                np.sum((type_grid == 1) & (self.quality_grid == 0)) / built_count
            )
        else:
            accessibility = 0.0
            mean_load = 0.0
            land_value_mean = float(self.land_value.mean())
            pollution_mean = float(self.pollution.mean())
            noise_mean = float(self.noise.mean())
            affordable_share = 0.0
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
        satisfaction = self.citizens.evaluate_city(self)
        util_values = [v for v in self.utilization.values() if v > 0]
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
            population=float(self.population),
            jobs=float(self.jobs),
            goods_supply=float(self.goods_supply),
            goods_demand=float(self.goods_demand),
            goods_balance=float(self.goods_balance),
            mean_utilization=float(np.mean(util_values)) if util_values else 0.0,
            land_value_mean=land_value_mean,
            pollution_mean=pollution_mean,
            noise_mean=noise_mean,
            mean_satisfaction=satisfaction["mean_satisfaction"],
            gini_satisfaction=satisfaction["gini_satisfaction"],
            rich_satisfaction=satisfaction["rich_satisfaction"],
            poor_satisfaction=satisfaction["poor_satisfaction"],
            tax_revenue=float(sum(a.tax_paid for a in self.agents)),
            subsidy_total=float(sum(a.subsidy_received for a in self.agents)),
            bankrupt_agents=int(sum(a.bankrupt for a in self.agents)),
            affordable_share=affordable_share,
        )

    def agent_metrics(self) -> list[AgentMetrics]:
        violations, incompatible, density_excess = self._agent_violations()
        built = self.type_grid > 0
        built_count = int(built.sum())
        green_share = np.sum(self.type_grid == 5) / max(1, built_count)
        green_deficit = max(0.0, self.config.requirements.green_min - green_share)
        quota = self.config.regulation.affordable_housing_quota
        affordable_count = int(np.sum((self.type_grid == 1) & (self.quality_grid == 0)))
        affordable_share = affordable_count / max(1, built_count)
        affordable_deficit = max(0.0, quota - affordable_share)
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
                own_entropy = float(-np.sum(p * np.log(p)) / np.log(5.0))
                own_green = float(np.sum(own_types == 5) / max(1, total))
                own_affordable = float(
                    np.sum((own_types == 1) & (self.quality_grid[mask] == 0))
                    / max(1, total)
                )
            else:
                mean_distance = 1.0
                high_quality = 0.0
                own_entropy = 0.0
                own_green = 0.0
                own_affordable = 0.0
            deficit_share = green_deficit * (own_builds / max(1, built_count))
            affordable_deficit_share = affordable_deficit * (
                own_builds / max(1, built_count)
            )
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
                    green_deficit_share=float(deficit_share + affordable_deficit_share),
                    density_excess=float(density_excess[agent.agent_id]),
                    mean_distance=mean_distance,
                    high_quality_fraction=high_quality,
                    type_entropy=own_entropy,
                    behavior=np.array(
                        [mean_distance, high_quality, own_entropy, own_green],
                        dtype=np.float64,
                    ),
                    tax_paid=agent.tax_paid,
                    subsidy_received=agent.subsidy_received,
                    bid_paid=agent.bid_paid,
                    refund_received=agent.refund_received,
                    bankrupt=agent.bankrupt,
                    affordable_share=own_affordable,
                    green_share=own_green,
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
