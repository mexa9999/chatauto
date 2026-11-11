"""Citizens as an independent evaluation oracle.

Citizens never influence developer income and never enter the evolutionary
fitness.  They only produce a "human quality of the city" report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    CITIZEN_RICH_FRACTION,
    N_CITIZENS,
    QUALITY_NAMES,
)
from .grid import gini


@dataclass
class Citizen:
    is_rich: bool
    price_weight: float
    green_weight: float
    transport_weight: float
    pollution_weight: float
    accessibility_weight: float

    def evaluate(self, state: dict[str, float]) -> float:
        positive = (
            self.price_weight * (1.0 - state["price_index"])
            + self.green_weight * state["green_share"]
            + self.transport_weight * state["transport_index"]
            + self.accessibility_weight * state["accessibility"]
        )
        negative = self.pollution_weight * state["pollution"]
        housing_penalty = 0.30 * max(0.0, 1.0 - state["housing_ratio"])
        score = positive - negative - housing_penalty
        # Smooth 0..1 satisfaction.
        return float(1.0 / (1.0 + np.exp(-6.0 * (score - 0.18))))


class CitizenPool:
    """A reproducible population of 100 citizens with heterogeneous tastes."""

    def __init__(
        self,
        seed: int,
        n: int = N_CITIZENS,
        rich_fraction: float = CITIZEN_RICH_FRACTION,
    ):
        if n < 1:
            raise ValueError("n must be positive")
        if not 0.0 <= rich_fraction <= 1.0:
            raise ValueError("rich_fraction must be in [0, 1]")
        rng = np.random.default_rng(int(seed))
        self.citizens: list[Citizen] = []
        for i in range(n):
            is_rich = bool(i < round(n * rich_fraction))
            if is_rich:
                price = rng.uniform(0.04, 0.12)
                green = rng.uniform(0.20, 0.32)
                transport = rng.uniform(0.18, 0.28)
                pollution = rng.uniform(0.22, 0.34)
                access = rng.uniform(0.20, 0.30)
            else:
                price = rng.uniform(0.22, 0.34)
                green = rng.uniform(0.12, 0.22)
                transport = rng.uniform(0.20, 0.30)
                pollution = rng.uniform(0.20, 0.32)
                access = rng.uniform(0.12, 0.22)
            total = price + green + transport + pollution + access
            self.citizens.append(
                Citizen(
                    is_rich=is_rich,
                    price_weight=price / total,
                    green_weight=green / total,
                    transport_weight=transport / total,
                    pollution_weight=pollution / total,
                    accessibility_weight=access / total,
                )
            )

    def evaluate_city(self, env) -> dict[str, float]:
        env._ensure_travel_time_maps()
        type_grid = env.type_grid
        built = type_grid > 0
        built_count = int(built.sum())
        if built_count:
            green_share = float(np.sum(type_grid == 5) / built_count)
            pollution = float(env.pollution[built].mean())
            noise = float(env.noise[built].mean())
            transport = float(
                env.static["road"][built].mean() * (1.0 - env.load[built].mean())
            )
            accessibility = float(
                1.0 - env.travel_time_center_norm[built].mean()
            )
            price_index = float(np.clip(env.land_value[built].mean(), 0.0, 1.0))
        else:
            green_share = 0.0
            pollution = 0.0
            noise = 0.0
            transport = 0.0
            accessibility = 0.0
            price_index = float(np.clip(env.land_value.mean(), 0.0, 1.0))

        target_population = max(1.0, env.base_population + env.population_extra)
        housing_ratio = float(np.clip(env.population / target_population, 0.0, 1.0))
        state = {
            "price_index": price_index,
            "green_share": green_share,
            "transport_index": transport,
            "pollution": float(np.clip(pollution + 0.35 * noise, 0.0, 1.0)),
            "accessibility": accessibility,
            "housing_ratio": housing_ratio,
        }
        values = np.array([c.evaluate(state) for c in self.citizens], dtype=np.float64)
        rich = np.array([c.is_rich for c in self.citizens], dtype=bool)
        return {
            "mean_satisfaction": float(values.mean()),
            "gini_satisfaction": float(gini(values)),
            "rich_satisfaction": float(values[rich].mean()) if rich.any() else float(values.mean()),
            "poor_satisfaction": float(values[~rich].mean()) if (~rich).any() else float(values.mean()),
        }


def describe_citizen_pool(pool: CitizenPool) -> dict[str, float]:
    rich = np.array([c.is_rich for c in pool.citizens], dtype=bool)
    return {
        "n_citizens": float(len(pool.citizens)),
        "rich_fraction": float(rich.mean()),
        "mean_price_weight": float(np.mean([c.price_weight for c in pool.citizens])),
        "quality_names": float(len(QUALITY_NAMES)),  # keeps the import intentional
    }
