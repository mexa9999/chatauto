"""Policies: NumPy neuroevolution genome, plus random/manual/greedy baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import N_CELL_FEATURES, N_GLOBAL_FEATURES
from .world import Action

if TYPE_CHECKING:
    from .world import CityEnvironment, Observation


class Policy(Protocol):
    name: str

    def act(
        self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator
    ) -> Action: ...


# ---------------------------------------------------------------------------
# Genome layout and mutation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenomeLayout:
    n_cell_features: int = N_CELL_FEATURES
    n_global_features: int = N_GLOBAL_FEATURES
    hidden_size: int = 64
    n_types: int = 5
    n_qualities: int = 3

    @property
    def input_size(self) -> int:
        return self.n_cell_features + self.n_global_features

    @property
    def parameter_count(self) -> int:
        inp = self.input_size
        h = self.hidden_size
        return (
            inp * h + h          # W1, b1
            + h * h + h          # W2, b2
            + h + 1              # score head
            + h * self.n_types + self.n_types
            + h * self.n_qualities + self.n_qualities
        )

    def unpack(self, vector: np.ndarray):
        if vector.shape != (self.parameter_count,):
            raise ValueError(
                f"genome has shape {vector.shape}, expected ({self.parameter_count},)"
            )
        inp = self.input_size
        h = self.hidden_size
        pos = 0

        def take(shape):
            nonlocal pos
            size = int(np.prod(shape))
            out = vector[pos : pos + size].reshape(shape)
            pos += size
            return out

        W1 = take((inp, h))
        b1 = take((h,))
        W2 = take((h, h))
        b2 = take((h,))
        w3 = take((h,))
        b3 = take((1,))
        W_type = take((h, self.n_types))
        b_type = take((self.n_types,))
        W_quality = take((h, self.n_qualities))
        b_quality = take((self.n_qualities,))
        return W1, b1, W2, b2, w3, b3, W_type, b_type, W_quality, b_quality


DEFAULT_LAYOUT = GenomeLayout()


def random_genome(rng: np.random.Generator, layout: GenomeLayout = DEFAULT_LAYOUT) -> np.ndarray:
    inp = layout.input_size
    h = layout.hidden_size
    vector = np.empty(layout.parameter_count, dtype=np.float64)
    pos = 0

    def put(arr):
        nonlocal pos
        flat = np.asarray(arr, dtype=np.float64).ravel()
        vector[pos : pos + flat.size] = flat
        pos += flat.size

    put(rng.normal(0.0, np.sqrt(2.0 / inp), size=(inp, h)))
    put(np.zeros(h))
    put(rng.normal(0.0, np.sqrt(2.0 / h), size=(h, h)))
    put(np.zeros(h))
    put(rng.normal(0.0, 0.02, size=h))
    put(np.zeros(1))
    put(rng.normal(0.0, 0.02, size=(h, layout.n_types)))
    put(np.zeros(layout.n_types))
    put(rng.normal(0.0, 0.02, size=(h, layout.n_qualities)))
    put(np.zeros(layout.n_qualities))
    return vector


def mutate_genome(
    genome: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    mutation_rate: float = 1.0,
) -> np.ndarray:
    child = genome.copy()
    if mutation_rate >= 1.0:
        child += rng.normal(0.0, sigma, size=child.size)
    else:
        mask = rng.random(child.size) < mutation_rate
        child[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    return child


def genome_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Neural policy
# ---------------------------------------------------------------------------


class NeuralPolicy:
    name = "neural"

    def __init__(
        self,
        genome: np.ndarray,
        layout: GenomeLayout = DEFAULT_LAYOUT,
    ):
        self.genome = np.asarray(genome, dtype=np.float64)
        self.layout = layout
        (
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.w3,
            self.b3,
            self.W_type,
            self.b_type,
            self.W_quality,
            self.b_quality,
        ) = layout.unpack(self.genome)

    def _scores(self, obs: "Observation") -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(
            [
                obs.cell_features,
                np.broadcast_to(obs.global_features, (obs.cell_features.shape[0], obs.global_features.size)),
            ],
            axis=1,
        )
        h1 = np.maximum(x @ self.W1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0.0)
        scores = h2 @ self.w3 + float(self.b3[0])
        return scores, h2

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        obs = env.observe(agent_id)
        if not obs.valid_cells.any():
            return Action.pass_action()

        scores, hidden = self._scores(obs)
        scores = scores.copy()
        scores[~obs.valid_cells] = -np.inf
        best_cell = int(np.argmax(scores))

        embedding = hidden[best_cell]
        type_logits = embedding @ self.W_type + self.b_type
        quality_logits = embedding @ self.W_quality + self.b_quality
        joint = type_logits[:, None] + quality_logits[None, :]
        mask = obs.action_mask[best_cell]
        if not mask.any():
            return Action.pass_action()
        joint = joint.copy()
        joint[~mask] = -np.inf
        flat = int(np.argmax(joint))
        type_id, quality = divmod(flat, self.layout.n_qualities)
        cost = float(obs.cost_table[best_cell, type_id, quality])
        return Action(cell=best_cell, type_id=type_id, quality=quality, cost=cost)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomPolicy:
    name = "random"

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        obs = env.observe(agent_id)
        valid_cells = np.flatnonzero(obs.valid_cells)
        if valid_cells.size == 0:
            return Action.pass_action()
        cell = int(rng.choice(valid_cells))
        combos = np.argwhere(obs.action_mask[cell])
        if combos.size == 0:
            return Action.pass_action()
        type_id, quality = (int(x) for x in combos[int(rng.integers(combos.shape[0]))])
        cost = float(obs.cost_table[cell, type_id, quality])
        return Action(cell=cell, type_id=type_id, quality=quality, cost=cost)


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------


class GreedyPolicy:
    name = "greedy"

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        obs = env.observe(agent_id)
        if not obs.valid_cells.any() or obs.remaining_rounds <= 0:
            return Action.pass_action()
        income = env.projected_income_table(agent_id)
        value = income * float(obs.remaining_rounds) - obs.cost_table
        value[~obs.action_mask] = -np.inf
        flat = int(np.argmax(value))
        if not np.isfinite(value.ravel()[flat]) or value.ravel()[flat] <= 0.0:
            return Action.pass_action()
        cell, type_id, quality = np.unravel_index(flat, value.shape)
        cell, type_id, quality = int(cell), int(type_id), int(quality)
        cost = float(obs.cost_table[cell, type_id, quality])
        return Action(cell=cell, type_id=type_id, quality=quality, cost=cost)


# ---------------------------------------------------------------------------
# Hand-designed parametric baseline (10 parameters)
# ---------------------------------------------------------------------------


MANUAL_PARAMETER_NAMES = (
    "centre",       # 1 prefers high accessibility
    "cheap",        # 2 prefers low land price
    "road",         # 3 road access
    "water",        # 4 water proximity
    "soil",         # 5 soil quality
    "green",        # 6 green preference
    "housing",      # 7 housing preference
    "commercial",   # 8 commercial preference
    "industry",     # 9 industry preference
    "quality",      # 10 quality level in [0, 1] -> low/medium/high
)


def default_manual_genome() -> np.ndarray:
    # Hand-tuned plausible developer; deliberately not optimised.
    return np.array(
        [
            0.80,   # centre
            0.25,   # cheap
            0.60,   # road
            0.15,   # water
            0.30,   # soil
            0.05,   # green
            0.55,   # housing
            0.45,   # commercial
            -0.35,  # industry
            0.60,   # quality
        ],
        dtype=np.float64,
    )


def manual_variants(
    rng: np.random.Generator, count: int, scale: float = 0.20
) -> list[np.ndarray]:
    base = default_manual_genome()
    return [
        np.clip(base + rng.normal(0.0, scale, size=base.size), -1.0, 1.0)
        for _ in range(count)
    ]


class ManualPolicy:
    """Interpretable 10-parameter developer heuristic."""

    name = "manual"

    def __init__(self, params: np.ndarray | None = None):
        self.params = (
            default_manual_genome() if params is None else np.asarray(params, dtype=np.float64)
        )
        if self.params.shape != (10,):
            raise ValueError("manual genome must have exactly 10 parameters")

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        obs = env.observe(agent_id)
        if not obs.valid_cells.any():
            return Action.pass_action()

        p = self.params
        f = obs.cell_features.reshape(env.h, env.w, -1)
        utility = (
            p[0] * (1.0 - f[..., 4])
            + p[1] * (1.0 - f[..., 0])
            + p[2] * f[..., 3]
            + p[3] * f[..., 2]
            + p[4] * f[..., 1]
            - 0.35 * f[..., 5]
            + 0.20 * f[..., 9]
        ).ravel()

        housing_share = f[..., 6].ravel()
        commercial_share = f[..., 7].ravel()
        industry_share = f[..., 8].ravel()
        density = f[..., 5].ravel()
        mean_quality = f[..., 9].ravel()
        compat = np.empty((5, env.h * env.w), dtype=np.float64)
        compat[0] = 0.50 * housing_share + 0.45 * commercial_share - 0.70 * industry_share + 0.20 * mean_quality
        compat[1] = 0.45 * housing_share + 0.40 * commercial_share - 0.60 * industry_share + 0.15 * mean_quality
        compat[2] = 0.45 * (housing_share + commercial_share) - 0.50 * industry_share + 0.15 * mean_quality
        compat[3] = 0.30 * industry_share - 0.45 * (housing_share + commercial_share)
        compat[4] = 0.25 * density - 0.20 * industry_share
        for t in range(5):
            compat[t] = compat[t].ravel()

        type_pref = np.array(
            [
                p[6],
                p[7],
                0.5 * (p[6] + p[7]),
                p[8],
                p[5],
            ],
            dtype=np.float64,
        )
        base_scores = utility[None, :] + type_pref[:, None] + compat

        preferred_quality = int(
            np.clip(np.round(np.clip(p[9], 0.0, 1.0) * 2.0), 0, 2)
        )
        qualities = [preferred_quality]
        qualities += [q for q in (2, 1, 0) if q != preferred_quality]
        max_cells = env.n_cells
        income = env.projected_income_table(agent_id)
        economic_value = income * float(obs.remaining_rounds) - obs.cost_table
        for quality in qualities:
            mask = obs.action_mask[:, :, quality].T  # (5, cells)
            mask &= economic_value[:, :, quality].T > 0.0
            if not mask.any():
                continue
            scores = base_scores.copy()
            scores[~mask] = -np.inf
            flat = int(np.argmax(scores))
            type_id, cell = divmod(flat, max_cells)
            cost = float(obs.cost_table[cell, type_id, quality])
            return Action(cell=cell, type_id=type_id, quality=quality, cost=cost)
        return Action.pass_action()


def homogeneous_policies(
    factory,
    n_agents: int,
) -> list[Policy]:
    return [factory() for _ in range(n_agents)]
