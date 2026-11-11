"""Deterministic schedules and random streams for city events."""

from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np


RECESSION_INTERVAL = 10
RECESSION_DURATION = 2
RECESSION_INCOME_MULTIPLIER = 0.80
FLOOD_DURATION = 2
FLOOD_CELL_FRACTION = 0.10
TECHNOLOGY_SHIFT_INCOME_MULTIPLIER = 1.20


@dataclass(frozen=True)
class ScheduledShock:
    round_index: int
    kind: str
    magnitude: float
    duration: int | None = None

    def as_dict(self) -> dict[str, int | float | str]:
        shock = {
            "round": self.round_index,
            "kind": self.kind,
            "magnitude": self.magnitude,
        }
        if self.duration is not None:
            shock["duration"] = self.duration
        return shock


def automatic_shocks(round_count: int) -> list[ScheduledShock]:
    """Build the standard deterministic event schedule for an episode."""
    if round_count < 0:
        raise ValueError("round_count must be non-negative")
    if round_count == 0:
        return []

    shocks = [
        ScheduledShock(
            round_index,
            "recession",
            RECESSION_INCOME_MULTIPLIER,
            RECESSION_DURATION,
        )
        for round_index in range(RECESSION_INTERVAL - 1, round_count, RECESSION_INTERVAL)
    ]
    midpoint = round_count // 2
    shocks.extend(
        [
            ScheduledShock(
                midpoint,
                "flood",
                FLOOD_CELL_FRACTION,
                FLOOD_DURATION,
            ),
            ScheduledShock(
                midpoint,
                "technology_shift",
                TECHNOLOGY_SHIFT_INCOME_MULTIPLIER,
            ),
        ]
    )
    return sorted(shocks, key=lambda shock: (shock.round_index, shock.kind))


def shock_rng(
    episode_seed: int, round_index: int, event_index: int, kind: str
) -> np.random.Generator:
    """Derive a stable event RNG without consuming the policy RNG stream."""
    seed = int(episode_seed) % (1 << 64)
    seed_words = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]
    seed_words.extend(
        [
            int(round_index),
            int(event_index),
            zlib.crc32(kind.encode("utf-8")),
        ]
    )
    return np.random.default_rng(np.random.SeedSequence(seed_words))
