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
ACTION_TYPE_NAMES = TYPE_NAMES[1:]  # 5 buildable types, policy indices 0..4
QUALITY_NAMES = ("low", "medium", "high")

# Arrays are indexed by world type id (0 = empty).
BUILD_COST = np.array([0.0, 80.0, 110.0, 100.0, 85.0, 25.0], dtype=np.float64)
INCOME_BASE = np.array([0.0, 30.0, 48.0, 36.0, 33.0, 7.5], dtype=np.float64)
OPTIMAL_DISTANCE = np.array([0.0, 0.40, 0.10, 0.30, 0.75, 0.50], dtype=np.float64)
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
COMMERCIAL_GOODS_DEMAND = 15.0
INDUSTRY_GOODS_SUPPLY = 5.0
POPULATION_INERTIA = 0.35
CITIZEN_GOODS_DEMAND = 0.5

# ---------------------------------------------------------------------------
# Land value, pollution and noise (economy level 3)
# ---------------------------------------------------------------------------

LAND_VALUE_MIN = 0.20
LAND_VALUE_MAX = 1.00
POLLUTION_RADIUS = 2
NOISE_RADIUS = 1
BANKRUPTCY_THRESHOLD = 0.0
BANKRUPTCY_STREAK = 3
BANKRUPTCY_FITNESS_PENALTY = 0.35

# ---------------------------------------------------------------------------
# Regulation: taxes and subsidies are city parameters, not agents
# ---------------------------------------------------------------------------

PROPERTY_TAX_RATE = 0.02
INDUSTRY_TAX_RATE = 0.03
GREEN_SUBSIDY = 4.0
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
)

N_CELL_FEATURES = len(FEATURE_NAMES)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)
assert N_CELL_FEATURES == 23
assert N_GLOBAL_FEATURES == 10

# Behaviour descriptor length used by diversity and MAP-Elites.
BEHAVIOR_DIM = 4


# ---------------------------------------------------------------------------
# Frozen protocol objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitnessWeights:
    """Fitness weights fixed before experiments (week 2 of the plan).

    ``city_quality`` is kept for reporting but, by protocol, has zero weight in
    the selection score.  Banruptcy is a separate, smooth penalty.
    """

    profit: float = 0.50
    city_quality: float = 0.00
    build_count: float = 0.10
    violations: float = 0.10
    externality_damage: float = 0.10
    skip_penalty: float = 0.05
    bankruptcy: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "w_profit": self.profit,
            "w_city_quality": self.city_quality,
            "w_build_count": self.build_count,
            "w_violations": self.violations,
            "w_externality_damage": self.externality_damage,
            "w_skip_penalty": self.skip_penalty,
            "w_bankruptcy": self.bankruptcy,
        }


@dataclass(frozen=True)
class Regulation:
    """City-level taxes and subsidies.  Agents never control these."""

    name: str = "default"
    property_tax_rate: float = PROPERTY_TAX_RATE
    industry_tax_rate: float = INDUSTRY_TAX_RATE
    green_subsidy: float = GREEN_SUBSIDY
    affordable_housing_quota: float = AFFORDABLE_HOUSING_QUOTA

    def __post_init__(self) -> None:
        if not 0.0 <= self.property_tax_rate <= 1.0:
            raise ValueError("property_tax_rate must be in [0, 1]")
        if not 0.0 <= self.industry_tax_rate <= 1.0:
            raise ValueError("industry_tax_rate must be in [0, 1]")
        if self.green_subsidy < 0.0:
            raise ValueError("green_subsidy must be >= 0")
        if not 0.0 <= self.affordable_housing_quota <= 1.0:
            raise ValueError("affordable_housing_quota must be in [0, 1]")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "property_tax_rate": self.property_tax_rate,
            "industry_tax_rate": self.industry_tax_rate,
            "green_subsidy": self.green_subsidy,
            "affordable_housing_quota": self.affordable_housing_quota,
        }


REGULATION_SCENARIOS: dict[str, Regulation] = {
    "default": Regulation(),
    "no_tax": Regulation(
        name="no_tax",
        property_tax_rate=0.0,
        industry_tax_rate=0.0,
        green_subsidy=0.0,
        affordable_housing_quota=0.0,
    ),
    "property_tax_5": Regulation(name="property_tax_5", property_tax_rate=0.05),
    "industry_tax_10": Regulation(name="industry_tax_10", industry_tax_rate=0.10),
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
    industry_buffer: int = 1

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

    height: int = 8
    width: int = 8
    n_agents: int = 10
    actions_per_agent: int = 6
    initial_budget: float = 600.0
    buildings_per_action: int = 1
    enable_externalities: bool = True
    use_auction: bool = True
    bids_per_agent: int = BIDS_PER_AGENT
    requirements: Requirements = field(default_factory=lambda: REQUIREMENT_PROFILES["medium"])
    regulation: Regulation = field(default_factory=Regulation)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)

    def __post_init__(self) -> None:
        if self.height < 3 or self.width < 3:
            raise ValueError("grid must be at least 3x3")
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if self.actions_per_agent < 1:
            raise ValueError("actions_per_agent must be >= 1")
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
            "initial_budget": self.initial_budget,
            "buildings_per_action": self.buildings_per_action,
            "enable_externalities": self.enable_externalities,
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
    generations: int = 200
    episodes_per_generation: int = 10
    elites: int = 4
    tournament_size: int = 3
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
    sharing_radius: float = 0.25
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
