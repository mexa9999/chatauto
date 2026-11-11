"""Central configuration and frozen model constants.

All numbers that define the model live here.  Experiment scripts must not
change the fitness weights: they are part of the protocol (see docs/MODEL.md).
The module intentionally keeps the old, compact CitySim API while exposing the
richer economy used by the second version of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# World vocabulary
# ---------------------------------------------------------------------------

TYPE_NAMES = ("empty", "housing", "commercial", "mixed", "industry", "green")
TYPE_IDS = {name: idx for idx, name in enumerate(TYPE_NAMES)}
ACTION_TYPE_NAMES = TYPE_NAMES[1:] + ("road",)
ROAD_ACTION_ID = len(ACTION_TYPE_NAMES) - 1
QUALITY_NAMES = ("low", "medium", "high")
MAX_FLOORS_BY_ACTION = (12, 20, 12, 8, 3)
N_ROOF_QUALITIES = 4
ROAD_COST_BASE = 8.0
ROAD_COST_LAND_MULTIPLIER = 12.0
ROAD_ACTIVATION_DELAY = 1
ROAD_MAINTENANCE_RATE = 0.02
ROAD_DEMOLISH_REFUND = 0.50
ROAD_CONNECTED_WEIGHT = 1.5
ROAD_TRAVEL_COST = 1.0
WALK_TRAVEL_COST = 3.0
BUILDING_TRAVEL_COST = 5.0
CONGESTION_ALPHA = 2.0
ROAD_RELIEF_MAX = 0.5
ROAD_SUPPLY_FULL = 3.0

# Arrays are indexed by world type id (0 = empty).
BUILD_COST = np.array([0.0, 80.0, 110.0, 100.0, 85.0, 25.0], dtype=np.float64)
INCOME_BASE = np.array([0.0, 30.0, 48.0, 36.0, 33.0, 7.5], dtype=np.float64)
OPTIMAL_DISTANCE = np.array(
    [0.0, 0.40, 0.10, 0.30, 0.50, 0.50], dtype=np.float64
)
QUALITY_COST_MULT = np.array([0.70, 1.00, 1.40], dtype=np.float64)
QUALITY_INCOME_MULT = np.array([0.65, 1.00, 1.35], dtype=np.float64)
MAINTENANCE_RATE = 0.02
MAX_PRICE = 1.5
MAX_DEMAND = 1.2
GREEN_TARGET = 0.20

# ---------------------------------------------------------------------------
# Local saturation (economy level 1)
# ---------------------------------------------------------------------------

SATURATION_ALPHA = 0.12
SATURATION_RADIUS = 2

# ---------------------------------------------------------------------------
# Global stocks (economy level 2)
# ---------------------------------------------------------------------------

HOUSING_CAPACITY = 4.0
MIXED_CAPACITY = 3.0
COMMERCIAL_JOBS = 8.0
INDUSTRY_JOBS = 10.0
MIXED_JOBS = 4.0
COMMERCIAL_CUSTOMERS = 30.0
COMMERCIAL_GOODS_DEMAND = 25.0
INDUSTRY_GOODS_SUPPLY = 5.0
INDUSTRY_INCOME_MULTIPLIER = 0.80
GOODS_IMPORT_MARKUP = 0.25
MIGRATION_RATE = 0.10
CITIZEN_GOODS_DEMAND = 0.5
DEBT_INTEREST_RATE = 0.02

# ---------------------------------------------------------------------------
# Land value, pollution and noise (economy level 3)
# ---------------------------------------------------------------------------

LAND_VALUE_MIN = 0.20
LAND_VALUE_MAX = 1.00
POLLUTION_RADIUS = 2
NOISE_RADIUS = 1
# ---------------------------------------------------------------------------
# Regulation: taxes and subsidies are city parameters, not agents
# ---------------------------------------------------------------------------

TYPE_TAX_RATES = (0.02, 0.04, 0.03, 0.05, 0.0)
QUALITY_TAX_MULTIPLIERS = (0.8, 1.0, 1.2)
TAX_RECESSION_UNEMPLOYMENT = 0.15
TAX_BOOM_UNEMPLOYMENT = 0.05
TAX_PHASE_MULTIPLIERS = (0.75, 1.0, 1.25)
GREEN_SUBSIDY = 4.0
AFFORDABLE_HOUSING_SUBSIDY_PER_FLOOR = 4.0
EMPLOYMENT_SUBSIDY_PER_JOB = 0.5
AFFORDABLE_HOUSING_QUOTA = 0.20

# ---------------------------------------------------------------------------
# Citizens: an independent evaluation oracle, never part of selection
# ---------------------------------------------------------------------------

N_CITIZENS = 100
CITIZEN_RICH_FRACTION = 0.30

# ---------------------------------------------------------------------------
# Auctions and hierarchy
# ---------------------------------------------------------------------------

BIDS_PER_AGENT = 3
BID_MIN = 0.0
BID_MAX = 0.05          # maximal fraction of the current budget
DISTRICT_SIZE = 3

# ---------------------------------------------------------------------------
# Policy features (the order is part of the genome contract)
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "land_price",              # 0 dynamic price / MAX_PRICE
    "soil",                    # 1 static soil category / 2
    "water",                   # 2 static water proximity
    "road",                    # 3 static road access
    "dist_center",             # 4 distance to city centre, normalised
    "density",                 # 5 built share within radius 1
    "share_housing",           # 6 neighbourhood shares within radius 2
    "share_commercial",        # 7
    "share_industry",          # 8
    "mean_quality",            # 9 neighbours' mean quality / 2
    "load",                    # 10 congestion proxy
    "demand",                  # 11 dynamic demand / MAX_DEMAND
    "buildable",               # 12 static buildability mask
    "free",                    # 13 cell is empty
    "competition",             # 14 built neighbours owned by other agents
    "dist_industry",           # 15 distance to nearest industry, normalised
    "land_value",              # 16 intrinsic land value, normalised
    "pollution",               # 17 local industrial pollution
    "noise",                   # 18 local noise
    "attractiveness",          # 19 amenity index
    "local_saturation",        # 20 built share within SATURATION_RADIUS
    "district_type",           # 21 dominant type in the 3x3 district / 5
    "district_util",           # 22 mean utilisation in the district
    "district_synergy",        # 23 lagged neighbourhood complementarity
    "road_access",             # 24 static and player-road accessibility
    "player_road_density",     # 25 weighted active player-road density, radius 1
    "infra_wait_rounds",       # 26 rounds until a nearby pending road activates
    "travel_time_center",      # 27 normalised road-network time from city centre
    "travel_time_jobs",        # 28 normalised road-network time to jobs
    "flooded",                 # 29 temporarily flooded cell
)

GLOBAL_FEATURE_NAMES = (
    "budget",                  # budget / initial budget
    "turn_fraction",           # completed rounds / actions per agent
    "profit",                  # cumulative net cash / initial budget
    "built_fraction",          # built cells / all cells
    "mean_price",              # mean price / MAX_PRICE
    "n_agents",                # number of agents / 15
    "population_ratio",        # population / N_CITIZENS
    "jobs_ratio",              # jobs / max(1, population), clipped
    "goods_balance",           # (supply - demand) / max(1, supply + demand)
    "util_mean",               # mean utilisation of built sectors
    "housing_vacancy_rate",    # vacant housing / housing capacity
    "job_vacancy_rate",        # vacant jobs / job capacity
    "unemployment_rate",       # unemployed residents / population
    "net_migration_rate",      # net migration / base population
    "debt_ratio",              # debt / initial budget, clipped to 3
    "local_goods_coverage",    # local supply / total goods demand
    "recession_multiplier",    # current city-wide income multiplier
    "technology_housing",      # current technology bonus sector
    "technology_commercial",
    "technology_mixed",
    "technology_industry",
)

N_CELL_FEATURES = len(FEATURE_NAMES)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)
assert N_CELL_FEATURES == 30
assert N_GLOBAL_FEATURES == 21

# Behaviour descriptor length used by diversity and MAP-Elites.
BEHAVIOR_DIM = 4


# ---------------------------------------------------------------------------
# Frozen protocol objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitnessWeights:
    """Fitness weights fixed before experiments (week 2 of the plan).

    ``city_quality`` is kept separate from the shared city-welfare bonus.
    Debt is penalized smoothly using each agent's debt-to-initial-budget ratio.
    """

    profit: float = 0.50
    city_quality: float = 0.20
    build_count: float = 0.10
    violations: float = 0.10
    externality_damage: float = 0.10
    skip_penalty: float = 0.05
    debt: float = 0.15
    shared_reward: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "w_profit": self.profit,
            "w_city_quality": self.city_quality,
            "w_build_count": self.build_count,
            "w_violations": self.violations,
            "w_externality_damage": self.externality_damage,
            "w_skip_penalty": self.skip_penalty,
            "w_debt": self.debt,
            "w_shared_reward": self.shared_reward,
        }


@dataclass(frozen=True)
class Regulation:
    """City-level taxes and subsidies.  Agents never control these."""

    name: str = "default"
    type_tax_rates: tuple[float, ...] = TYPE_TAX_RATES
    quality_tax_multipliers: tuple[float, ...] = QUALITY_TAX_MULTIPLIERS
    recession_unemployment_threshold: float = TAX_RECESSION_UNEMPLOYMENT
    boom_unemployment_threshold: float = TAX_BOOM_UNEMPLOYMENT
    tax_phase_multipliers: tuple[float, ...] = TAX_PHASE_MULTIPLIERS
    green_subsidy: float = GREEN_SUBSIDY
    affordable_housing_subsidy_per_floor: float = AFFORDABLE_HOUSING_SUBSIDY_PER_FLOOR
    employment_subsidy_per_job: float = EMPLOYMENT_SUBSIDY_PER_JOB
    affordable_housing_quota: float = AFFORDABLE_HOUSING_QUOTA

    def __post_init__(self) -> None:
        if len(self.type_tax_rates) != 5 or any(
            not 0.0 <= rate <= 1.0 for rate in self.type_tax_rates
        ):
            raise ValueError("type_tax_rates must contain five rates in [0, 1]")
        if len(self.quality_tax_multipliers) != 3 or any(
            multiplier < 0.0 for multiplier in self.quality_tax_multipliers
        ):
            raise ValueError(
                "quality_tax_multipliers must contain three non-negative values"
            )
        if not (
            0.0
            <= self.boom_unemployment_threshold
            < self.recession_unemployment_threshold
            <= 1.0
        ):
            raise ValueError(
                "unemployment thresholds must satisfy 0 <= boom < recession <= 1"
            )
        if len(self.tax_phase_multipliers) != 3 or any(
            multiplier < 0.0 for multiplier in self.tax_phase_multipliers
        ):
            raise ValueError(
                "tax_phase_multipliers must contain three non-negative values"
            )
        if self.green_subsidy < 0.0:
            raise ValueError("green_subsidy must be >= 0")
        if self.affordable_housing_subsidy_per_floor < 0.0:
            raise ValueError("affordable_housing_subsidy_per_floor must be >= 0")
        if self.employment_subsidy_per_job < 0.0:
            raise ValueError("employment_subsidy_per_job must be >= 0")
        if not 0.0 <= self.affordable_housing_quota <= 1.0:
            raise ValueError("affordable_housing_quota must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type_tax_rates": self.type_tax_rates,
            "quality_tax_multipliers": self.quality_tax_multipliers,
            "recession_unemployment_threshold": self.recession_unemployment_threshold,
            "boom_unemployment_threshold": self.boom_unemployment_threshold,
            "tax_phase_multipliers": self.tax_phase_multipliers,
            "green_subsidy": self.green_subsidy,
            "affordable_housing_subsidy_per_floor": self.affordable_housing_subsidy_per_floor,
            "employment_subsidy_per_job": self.employment_subsidy_per_job,
            "affordable_housing_quota": self.affordable_housing_quota,
        }


REGULATION_SCENARIOS: dict[str, Regulation] = {
    "default": Regulation(),
    "no_tax": Regulation(
        name="no_tax",
        type_tax_rates=(0.0, 0.0, 0.0, 0.0, 0.0),
        green_subsidy=0.0,
        affordable_housing_subsidy_per_floor=0.0,
        employment_subsidy_per_job=0.0,
        affordable_housing_quota=0.0,
    ),
    "property_tax_5": Regulation(
        name="property_tax_5",
        type_tax_rates=(0.05, 0.05, 0.05, 0.05, 0.05),
    ),
    "industry_tax_10": Regulation(
        name="industry_tax_10",
        type_tax_rates=(0.0, 0.0, 0.0, 0.10, 0.0),
    ),
    "green_subsidy": Regulation(name="green_subsidy", green_subsidy=20.0),
    "affordable_quota": Regulation(
        name="affordable_quota", affordable_housing_quota=0.20
    ),
}


@dataclass(frozen=True)
class Requirements:
    """City planning requirements.  Violations are measured against them."""

    name: str = "medium"
    green_min: float = 0.15
    density_cap: float = 0.85
    industry_buffer: int = 1  # Retained for loading legacy configs; no longer enforced.

    def __post_init__(self) -> None:
        if not 0.0 <= self.green_min <= 1.0:
            raise ValueError("green_min must be in [0, 1]")
        if not 0.0 < self.density_cap <= 1.0:
            raise ValueError("density_cap must be in (0, 1]")
        if self.industry_buffer < 0:
            raise ValueError("industry_buffer must be >= 0")

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "green_min": self.green_min,
            "density_cap": self.density_cap,
            "industry_buffer": self.industry_buffer,
        }


REQUIREMENT_PROFILES: dict[str, Requirements] = {
    "soft": Requirements(name="soft", green_min=0.05, density_cap=1.0, industry_buffer=0),
    "medium": Requirements(name="medium", green_min=0.15, density_cap=0.85, industry_buffer=1),
    "hard": Requirements(name="hard", green_min=0.25, density_cap=0.65, industry_buffer=2),
}


@dataclass(frozen=True)
class CityConfig:
    """One episode of the city simulation."""

    height: int = 16
    width: int = 16
    n_agents: int = 10
    actions_per_agent: int = 20
    initial_budget: float = 600.0
    buildings_per_action: int = 1
    enable_externalities: bool = True
    enable_extended_actions: bool = True
    enable_district_synergy: bool = True
    use_auction: bool = True
    bids_per_agent: int = BIDS_PER_AGENT
    requirements: Requirements = field(default_factory=lambda: REQUIREMENT_PROFILES["medium"])
    regulation: Regulation = field(default_factory=Regulation)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    actions_per_round: int = 2

    def __post_init__(self) -> None:
        if not 3 <= self.height <= 64 or not 3 <= self.width <= 64:
            raise ValueError("grid sides must be between 3 and 64")
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if self.actions_per_agent < 1:
            raise ValueError("actions_per_agent must be >= 1")
        if not 1 <= self.actions_per_round <= 4:
            raise ValueError("actions_per_round must be between 1 and 4")
        if self.buildings_per_action < 1:
            raise ValueError("buildings_per_action must be >= 1")
        if self.bids_per_agent < 1:
            raise ValueError("bids_per_agent must be >= 1")
        if self.initial_budget <= 0:
            raise ValueError("initial_budget must be positive")

    @property
    def n_cells(self) -> int:
        return self.height * self.width

    def as_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "n_agents": self.n_agents,
            "actions_per_agent": self.actions_per_agent,
            "actions_per_round": self.actions_per_round,
            "initial_budget": self.initial_budget,
            "buildings_per_action": self.buildings_per_action,
            "enable_externalities": self.enable_externalities,
            "enable_extended_actions": self.enable_extended_actions,
            "enable_district_synergy": self.enable_district_synergy,
            "use_auction": self.use_auction,
            "bids_per_agent": self.bids_per_agent,
            "requirements": self.requirements.as_dict(),
            "regulation": self.regulation.as_dict(),
            "fitness": self.fitness.as_dict(),
        }


@dataclass(frozen=True)
class EvolutionConfig:
    """Neuroevolution protocol.  Population size is fixed per experiment."""

    population_size: int = 32
    generations: int = 125
    episodes_per_generation: int = 16
    elites: int = 4
    tournament_size: int = 5
    mutation_sigma: float = 0.05
    sigma_decay: float = 0.99
    min_sigma: float = 0.005
    islands: int = 4
    migration_interval: int = 10
    fresh_blood_interval: int = 20
    fresh_blood_count: int = 2
    use_islands: bool = True
    use_fitness_sharing: bool = True
    sharing_lambda: float = 0.03
    sharing_radius: float = 0.40
    hidden_size: int = 48
    attention_dim: int = 8
    district_hidden: int = 12
    global_hidden: int = 8
    district_size: int = DISTRICT_SIZE
    use_map_elites: bool = False
    map_elites_bins: int = 3
    track_spatial: bool = False
    spatial_every: int = 10
    use_parallel: bool = True
    max_workers: int | None = None
    seed: int = 0
    train_seed_base: int = 1_000_000
    validation_seed_base: int = 2_000_000
    test_seed_base: int = 3_000_000
    verbose: bool = True
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.episodes_per_generation < 1:
            raise ValueError("episodes_per_generation must be >= 1")
        if self.islands < 1:
            raise ValueError("islands must be >= 1")
        if self.use_islands and self.population_size % self.islands != 0:
            raise ValueError("population_size must be divisible by islands")
        if self.elites < 0 or self.elites >= self.population_size:
            raise ValueError("elites must be in [0, population_size)")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if self.mutation_sigma <= 0:
            raise ValueError("mutation_sigma must be positive")
        if not 0.0 < self.sigma_decay <= 1.0:
            raise ValueError("sigma_decay must be in (0, 1]")
        if self.fresh_blood_count < 0:
            raise ValueError("fresh_blood_count must be >= 0")
        if self.sharing_radius <= 0:
            raise ValueError("sharing_radius must be positive")
        if self.hidden_size < 4:
            raise ValueError("hidden_size must be >= 4")
        if self.attention_dim < 1 or self.district_hidden < 1 or self.global_hidden < 1:
            raise ValueError("architecture dimensions must be positive")
        if self.district_size < 1:
            raise ValueError("district_size must be >= 1")
        if self.map_elites_bins < 1:
            raise ValueError("map_elites_bins must be >= 1")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be >= 1 when set")

    def as_dict(self) -> dict:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "episodes_per_generation": self.episodes_per_generation,
            "elites": self.elites,
            "tournament_size": self.tournament_size,
            "mutation_sigma": self.mutation_sigma,
            "sigma_decay": self.sigma_decay,
            "min_sigma": self.min_sigma,
            "islands": self.islands,
            "migration_interval": self.migration_interval,
            "fresh_blood_interval": self.fresh_blood_interval,
            "fresh_blood_count": self.fresh_blood_count,
            "use_islands": self.use_islands,
            "use_fitness_sharing": self.use_fitness_sharing,
            "sharing_lambda": self.sharing_lambda,
            "sharing_radius": self.sharing_radius,
            "hidden_size": self.hidden_size,
            "attention_dim": self.attention_dim,
            "district_hidden": self.district_hidden,
            "global_hidden": self.global_hidden,
            "district_size": self.district_size,
            "use_map_elites": self.use_map_elites,
            "max_workers": self.max_workers,
            "seed": self.seed,
            "train_seed_base": self.train_seed_base,
            "validation_seed_base": self.validation_seed_base,
            "test_seed_base": self.test_seed_base,
        }


# ---------------------------------------------------------------------------
# Helpers used by experiment runners
# ---------------------------------------------------------------------------

SEED_STRIDE = 100_000


def seed_bases_for_run(run_seed: int) -> tuple[int, int, int]:
    """Disjoint train/validation/test seed blocks for one independent run."""
    if run_seed < 0:
        raise ValueError("run_seed must be >= 0")
    offset = run_seed * SEED_STRIDE
    return (
        1_000_000 + offset,
        2_000_000 + offset,
        3_000_000 + offset,
    )


def quick_city_config(**overrides) -> CityConfig:
    """Small configuration for smoke tests and interactive demos."""
    values = dict(
        height=8,
        width=8,
        n_agents=6,
        actions_per_agent=6,
        initial_budget=500.0,
        buildings_per_action=1,
    )
    values.update(overrides)
    return CityConfig(**values)


def quick_evolution_config(**overrides) -> EvolutionConfig:
    """Small evolution run that finishes in seconds."""
    values = dict(
        population_size=16,
        generations=20,
        episodes_per_generation=4,
        elites=4,
        islands=4,
        hidden_size=48,
        verbose=True,
        log_every=5,
    )
    values.update(overrides)
    return EvolutionConfig(**values)
