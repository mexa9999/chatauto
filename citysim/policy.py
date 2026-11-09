"""Policies: NumPy neuroevolution genome, plus random/manual/greedy baselines.

The learned policy is a small district-aware network with local attention over
the eight neighbouring cells.  Everything is evaluated with NumPy; no automatic
differentiation and no deep-learning framework is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import (
    BID_MAX,
    BID_MIN,
    N_CELL_FEATURES,
    N_GLOBAL_FEATURES,
)
from .world import Action

if TYPE_CHECKING:
    from .world import CityEnvironment, Observation


class Policy(Protocol):
    name: str

    def act(
        self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator
    ) -> Action: ...

    def propose_bids(
        self,
        env: "CityEnvironment",
        agent_id: int,
        rng: np.random.Generator,
        count: int,
        exclude_cells: set[int] | None = None,
    ) -> list[Action]: ...


# A soft district-level prior: which action types are natural in a district
# whose learned type is housing / commercial / industry / green / mixed.
DISTRICT_ALLOWED: tuple[tuple[int, ...], ...] = (
    (0, 2, 4),
    (1, 2, 4),
    (3, 4),
    (4, 0),
    (0, 1, 2, 3, 4),
)

_DISTRICT_ACTION_BIAS = np.zeros(
    (len(DISTRICT_ALLOWED), len(DISTRICT_ALLOWED), 3), dtype=np.float64
)
for _district_id, _allowed_types in enumerate(DISTRICT_ALLOWED):
    for _type_id in _allowed_types:
        _DISTRICT_ACTION_BIAS[_district_id, _type_id, :] = 0.40
_DISTRICT_ACTION_BIAS.setflags(write=False)


# ---------------------------------------------------------------------------
# Genome layout and mutation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenomeLayout:
    n_cell_features: int = N_CELL_FEATURES
    n_global_features: int = N_GLOBAL_FEATURES
    hidden_size: int = 48
    attention_dim: int = 8
    district_hidden: int = 12
    global_hidden: int = 8
    n_types: int = 5
    n_qualities: int = 3
    n_district_types: int = 5

    @property
    def input_size(self) -> int:
        return self.n_cell_features + self.n_global_features

    @property
    def cell_mlp_input(self) -> int:
        return (
            self.n_cell_features
            + self.attention_dim
            + self.global_hidden
            + self.n_district_types
        )

    @property
    def parameter_count(self) -> int:
        f = self.n_cell_features
        g = self.n_global_features
        a = self.attention_dim
        dh = self.district_hidden
        gh = self.global_hidden
        h = self.hidden_size
        return (
            3 * f * a + a               # attention query/key/value + bias
            + (f + g) * dh + dh          # district encoder
            + dh * self.n_district_types + self.n_district_types
            + (2 * f + g) * gh + gh      # global summary
            + self.cell_mlp_input * h + h
            + h * h + h
            + h + 1                      # cell score head
            + h * self.n_types + self.n_types
            + h * self.n_qualities + self.n_qualities
            + h + 1                      # bid head
        )

    def unpack(self, vector: np.ndarray):
        if vector.shape != (self.parameter_count,):
            raise ValueError(
                f"genome has shape {vector.shape}, expected ({self.parameter_count},)"
            )
        f = self.n_cell_features
        g = self.n_global_features
        a = self.attention_dim
        dh = self.district_hidden
        gh = self.global_hidden
        h = self.hidden_size
        pos = 0

        def take(shape):
            nonlocal pos
            size = int(np.prod(shape))
            out = vector[pos : pos + size].reshape(shape)
            pos += size
            return out

        Wq = take((f, a))
        Wk = take((f, a))
        Wv = take((f, a))
        b_att = take((a,))
        Wd = take((f + g, dh))
        bd = take((dh,))
        Wd_type = take((dh, self.n_district_types))
        bd_type = take((self.n_district_types,))
        Wg = take((2 * f + g, gh))
        bg = take((gh,))
        W1 = take((self.cell_mlp_input, h))
        b1 = take((h,))
        W2 = take((h, h))
        b2 = take((h,))
        w3 = take((h,))
        b3 = take((1,))
        W_type = take((h, self.n_types))
        b_type = take((self.n_types,))
        W_quality = take((h, self.n_qualities))
        b_quality = take((self.n_qualities,))
        w_bid = take((h,))
        b_bid = take((1,))
        return (
            Wq, Wk, Wv, b_att,
            Wd, bd, Wd_type, bd_type,
            Wg, bg,
            W1, b1, W2, b2,
            w3, b3,
            W_type, b_type,
            W_quality, b_quality,
            w_bid, b_bid,
        )


DEFAULT_LAYOUT = GenomeLayout()


def random_genome(rng: np.random.Generator, layout: GenomeLayout = DEFAULT_LAYOUT) -> np.ndarray:
    f = layout.n_cell_features
    g = layout.n_global_features
    a = layout.attention_dim
    dh = layout.district_hidden
    gh = layout.global_hidden
    h = layout.hidden_size
    vector = np.empty(layout.parameter_count, dtype=np.float64)
    pos = 0

    def put(arr):
        nonlocal pos
        flat = np.asarray(arr, dtype=np.float64).ravel()
        vector[pos : pos + flat.size] = flat
        pos += flat.size

    def normal(shape, scale):
        put(rng.normal(0.0, scale, size=shape))

    normal((f, a), np.sqrt(2.0 / f))
    normal((f, a), np.sqrt(2.0 / f))
    normal((f, a), np.sqrt(2.0 / f))
    put(np.zeros(a))
    normal((f + g, dh), np.sqrt(2.0 / (f + g)))
    put(np.zeros(dh))
    normal((dh, layout.n_district_types), 0.02)
    put(np.zeros(layout.n_district_types))
    normal((2 * f + g, gh), np.sqrt(2.0 / (2 * f + g)))
    put(np.zeros(gh))
    normal((layout.cell_mlp_input, h), np.sqrt(2.0 / layout.cell_mlp_input))
    put(np.zeros(h))
    normal((h, h), np.sqrt(2.0 / h))
    put(np.zeros(h))
    normal((h,), 0.02)
    put(np.zeros(1))
    normal((h, layout.n_types), 0.02)
    put(np.zeros(layout.n_types))
    normal((h, layout.n_qualities), 0.02)
    put(np.zeros(layout.n_qualities))
    normal((h,), 0.02)
    put(np.zeros(1))
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


def strategic_bid(expected_surplus: float, cost: float, budget: float) -> float:
    """Small value-proportional sealed bid, capped by the current budget.

    Baselines use the same cap as the learned policy, so the auction is not
    automatically won by the neural agent just because it is the only bidder.
    """
    if budget <= 0.0 or expected_surplus <= 0.0 or cost <= 0.0:
        return 0.0
    fraction = min(BID_MAX, 0.08 * expected_surplus / cost)
    cap = min(BID_MAX * budget, budget / 20.0)
    return float(min(max(fraction * budget, BID_MIN), cap))


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
            self.Wq, self.Wk, self.Wv, self.b_att,
            self.Wd, self.bd, self.Wd_type, self.bd_type,
            self.Wg, self.bg,
            self.W1, self.b1, self.W2, self.b2,
            self.w3, self.b3,
            self.W_type, self.b_type,
            self.W_quality, self.b_quality,
            self.w_bid, self.b_bid,
        ) = layout.unpack(self.genome)

    # -- low-level forward -------------------------------------------------

    @staticmethod
    def _neighbour_stack(f: np.ndarray) -> np.ndarray:
        """Return (H, W, 8, F) neighbour features with zero padding."""
        h, w, features = f.shape
        padded = np.pad(f, ((1, 1), (1, 1), (0, 0)))
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbours.append(
                    padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, :]
                )
        return np.stack(neighbours, axis=2)

    @staticmethod
    def _neighbour_validity(h: int, w: int) -> np.ndarray:
        """Boolean (H, W, 8) mask: the neighbour lies inside the grid."""
        if not hasattr(NeuralPolicy, "_validity_cache"):
            NeuralPolicy._validity_cache = {}
        if (h, w) in NeuralPolicy._validity_cache:
            return NeuralPolicy._validity_cache[(h, w)]
            
        ones = np.ones((h, w), dtype=np.float64)
        padded = np.pad(ones, 1)
        valid = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                valid.append(padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
        res = np.stack(valid, axis=2) > 0.5
        NeuralPolicy._validity_cache[(h, w)] = res
        return res

    def _forward(self, obs: "Observation") -> dict[str, np.ndarray]:
        layout = self.layout
        f = obs.cell_features.reshape(-1, layout.n_cell_features)
        n_cells = f.shape[0]
        grid_side = int(round(np.sqrt(n_cells)))
        h, w = getattr(obs, "_shape_hint", (grid_side, grid_side))
        f_grid = f.reshape(h, w, layout.n_cell_features)
        global_vec = obs.global_features

        # --- Ускоренный Attention (без np.stack) ---
        padded = np.pad(f_grid, ((1, 1), (1, 1), (0, 0)))
        q = f_grid @ self.Wq + self.b_att
        padded_keys = padded @ self.Wk
        padded_vals = padded @ self.Wv

        att_logits = np.zeros((h, w, 8), dtype=np.float64)
        idx = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                keys_slice = padded_keys[1+dy:1+dy+h, 1+dx:1+dx+w, :]
                att_logits[:, :, idx] = np.einsum('hwa,hwa->hw', keys_slice, q)
                idx += 1
        
        att_logits /= np.sqrt(max(1, layout.attention_dim))
        valid = self._neighbour_validity(h, w)
        att_logits[~valid] = -1e9
        att_logits -= att_logits.max(axis=2, keepdims=True)
        att = np.exp(att_logits)
        att /= np.maximum(1e-9, att.sum(axis=2, keepdims=True))

        context = np.zeros((h, w, layout.attention_dim))
        idx = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                vals_slice = padded_vals[1+dy:1+dy+h, 1+dx:1+dx+w, :]
                context += att[:, :, idx, None] * vals_slice
                idx += 1

        # --- District encoder ---
        labels = getattr(obs, "_district_labels")
        n_districts = int(getattr(obs, "_n_districts"))
        labels_flat = labels.ravel()
        counts = np.bincount(labels_flat, minlength=n_districts).astype(np.float64)

        district_sum_f = np.array([
            np.bincount(labels_flat, weights=f_grid[..., k].ravel(), minlength=n_districts)
            for k in range(layout.n_cell_features)
        ]).T
        district_sum_g = global_vec[None, :] * counts[:, None]
        district_sum = np.concatenate([district_sum_f, district_sum_g], axis=1)

        district_input = district_sum / np.maximum(1.0, counts)[:, None]
        district_hidden = np.maximum(district_input @ self.Wd + self.bd, 0.0)
        district_logits = district_hidden @ self.Wd_type + self.bd_type
        district_choice = np.argmax(district_logits, axis=1)
        
        district_type_cells = district_choice[labels_flat].reshape(h, w)
        district_logits_cells = district_logits[labels_flat].reshape(
            h, w, layout.n_district_types
        )

        # --- Global summary ---
        summary = np.concatenate([
            f_grid.mean(axis=(0, 1)), 
            f_grid.max(axis=(0, 1)), 
            global_vec
        ])
        global_embedding = np.maximum(summary @ self.Wg + self.bg, 0.0)

        # --- Блочное умножение ---
        f_dim = layout.n_cell_features
        a_dim = layout.attention_dim
        g_dim = layout.global_hidden
        
        W1_f = self.W1[:f_dim, :]
        W1_c = self.W1[f_dim:f_dim+a_dim, :]
        W1_g = self.W1[f_dim+a_dim:f_dim+a_dim+g_dim, :]
        W1_d = self.W1[f_dim+a_dim+g_dim:, :]

        h1 = (
            f_grid @ W1_f
            + context @ W1_c
            + (global_embedding @ W1_g)[None, None, :]
            + W1_d[district_choice][labels_flat].reshape(h, w, -1)
            + self.b1
        )
        h1 = np.maximum(h1, 0.0)

        h2 = np.maximum(h1.reshape(n_cells, -1) @ self.W2 + self.b2, 0.0)
        scores = h2 @ self.w3 + float(self.b3[0])
        
        type_logits = (h2 @ self.W_type + self.b_type).reshape(h, w, layout.n_types)
        quality_logits = (h2 @ self.W_quality + self.b_quality).reshape(h, w, layout.n_qualities)
        bid_fraction = 1.0 / (1.0 + np.exp(-(h2 @ self.w_bid + float(self.b_bid[0]))))

        # --- ПРЕДВЫЧИСЛЕНИЕ JOINT LOGITS ---
        joint_all = type_logits[:, :, :, None] + quality_logits[:, :, None, :]
        
        joint_all += _DISTRICT_ACTION_BIAS[district_type_cells]

        return {
            "scores": scores,
            "joint_all": joint_all,  # <--- НОВОЕ ПОЛЕ
            "bid_fraction": bid_fraction.reshape(h, w),
            "district_type": district_type_cells,
            "district_logits": district_logits_cells,
            "h2": h2,
            "shape": np.array([h, w], dtype=np.int64),
        }

    # -- action selection --------------------------------------------------

    def _joint_logits(
        self,
        forward: dict[str, np.ndarray],
        cell: int,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        h, w = (int(x) for x in forward["shape"])
        row, col = divmod(int(cell), w)
        joint = forward["joint_all"][row, col].copy()
        mask = action_mask[cell]
        joint[~mask] = -np.inf
        return joint

    def _action_at(
        self,
        obs: "Observation",
        forward: dict[str, np.ndarray],
        cell: int,
        priority: int = 0,
    ) -> Action | None:
        joint = self._joint_logits(forward, cell, obs.action_mask)
        if not np.isfinite(joint).any():
            return None
        flat = int(np.argmax(joint))
        type_id, quality = divmod(flat, self.layout.n_qualities)
        cost = float(obs.cost_table[cell, type_id, quality])
        h, w = (int(x) for x in forward["shape"])
        row, col = divmod(int(cell), w)
        budget = max(0.0, float(obs.budget))
        bid_cap = max(0.0, min(BID_MAX * budget, budget / 10.0))
        bid = float(min(max(forward["bid_fraction"][row, col] * budget, BID_MIN), bid_cap))
        if cost + bid > budget + 1e-9:
            return None
        return Action(
            cell=int(cell),
            type_id=int(type_id),
            quality=int(quality),
            cost=cost,
            bid=bid,
            priority=int(priority),
        )

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        del rng
        obs = env.observe(agent_id)
        if not obs.valid_cells.any():
            return Action.pass_action()
        forward = self._forward(obs)
        scores = forward["scores"].copy()
        scores[~obs.valid_cells] = -np.inf
        cell = int(np.argmax(scores))
        action = self._action_at(obs, forward, cell)
        return action if action is not None else Action.pass_action()

    def propose_bids(
        self,
        env: "CityEnvironment",
        agent_id: int,
        rng: np.random.Generator,
        count: int,
        exclude_cells: set[int] | None = None,
    ) -> list[Action]:
        del rng
        obs = env.observe(agent_id)
        if not obs.valid_cells.any():
            return []
        forward = self._forward(obs)
        scores = forward["scores"].copy()
        scores[~obs.valid_cells] = -np.inf
        if exclude_cells:
            for cell in exclude_cells:
                if 0 <= int(cell) < scores.size:
                    scores[int(cell)] = -np.inf
        # Rank cells by cell score plus the best conditioned type/quality.
        #h, w = (int(x) for x in forward["shape"])
        candidates: list[tuple[float, int]] = []
        valid_mask = np.isfinite(scores)
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return []
        
        # Находим top-N быстро
        k = min(count * 2, valid_indices.size)
        top_idx = np.argpartition(scores[valid_indices], -k)[-k:]
        top_cells = valid_indices[top_idx]
        top_cells = top_cells[np.argsort(scores[top_cells])[::-1]]

        for cell in top_cells:
            cell = int(cell)
            if not np.isfinite(scores[cell]):
                break
            joint = self._joint_logits(forward, cell, obs.action_mask)
            if not np.isfinite(joint).any():
                continue
            candidates.append((float(scores[cell] + np.max(joint)), cell))
            if len(candidates) >= count:
                break
        actions: list[Action] = []
        for priority, (_, cell) in enumerate(candidates):
            action = self._action_at(obs, forward, cell, priority=priority)
            if action is not None:
                actions.append(action)
        return actions


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

    def propose_bids(
        self,
        env: "CityEnvironment",
        agent_id: int,
        rng: np.random.Generator,
        count: int,
        exclude_cells: set[int] | None = None,
    ) -> list[Action]:
        obs = env.observe(agent_id)
        valid_cells = [
            int(c) for c in np.flatnonzero(obs.valid_cells)
            if not exclude_cells or int(c) not in exclude_cells
        ]
        if not valid_cells:
            return []
        chosen = rng.choice(
            valid_cells, size=min(count, len(valid_cells)), replace=False
        )
        actions: list[Action] = []
        for priority, cell in enumerate(int(c) for c in np.atleast_1d(chosen)):
            combos = np.argwhere(obs.action_mask[cell])
            if combos.size == 0:
                continue
            type_id, quality = (int(x) for x in combos[int(rng.integers(combos.shape[0]))])
            actions.append(
                Action(
                    cell=cell,
                    type_id=type_id,
                    quality=quality,
                    cost=float(obs.cost_table[cell, type_id, quality]),
                    priority=priority,
                )
            )
        return actions


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------


class GreedyPolicy:
    name = "greedy"

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        del rng
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

    def propose_bids(
        self,
        env: "CityEnvironment",
        agent_id: int,
        rng: np.random.Generator,
        count: int,
        exclude_cells: set[int] | None = None,
    ) -> list[Action]:
        del rng
        obs = env.observe(agent_id)
        if not obs.valid_cells.any():
            return []
        income = env.projected_income_table(agent_id)
        value = income * float(max(1, obs.remaining_rounds)) - obs.cost_table
        value[~obs.action_mask] = -np.inf
        order = np.argsort(value, axis=None)[::-1]
        actions: list[Action] = []
        used: set[int] = set(exclude_cells or set())
        for flat in order:
            if not np.isfinite(value.ravel()[flat]) or value.ravel()[flat] <= 0.0:
                break
            cell, type_id, quality = np.unravel_index(int(flat), value.shape)
            cell, type_id, quality = int(cell), int(type_id), int(quality)
            if cell in used:
                continue
            used.add(cell)
            cost = float(obs.cost_table[cell, type_id, quality])
            actions.append(
                Action(
                    cell=cell,
                    type_id=type_id,
                    quality=quality,
                    cost=cost,
                    bid=strategic_bid(
                        float(value[cell, type_id, quality]), cost, obs.budget
                    ),
                    priority=len(actions),
                )
            )
            if len(actions) >= count:
                break
        return actions


# ---------------------------------------------------------------------------
# Hand-designed parametric baseline (10 parameters)
# ---------------------------------------------------------------------------


MANUAL_PARAMETER_NAMES = (
    "centre",
    "cheap",
    "road",
    "water",
    "soil",
    "green",
    "housing",
    "commercial",
    "industry",
    "quality",
)


def default_manual_genome() -> np.ndarray:
    # Hand-tuned plausible developer; deliberately not optimised.
    return np.array(
        [
            0.80,
            0.25,
            0.60,
            0.15,
            0.30,
            0.05,
            0.55,
            0.45,
            -0.35,
            0.60,
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

    def _score_table(self, env: "CityEnvironment", agent_id: int):
        obs = env.observe(agent_id)
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
            + 0.15 * f[..., 16]
            - 0.20 * f[..., 17]
            - 0.10 * f[..., 18]
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
        type_pref = np.array(
            [p[6], p[7], 0.5 * (p[6] + p[7]), p[8], p[5]],
            dtype=np.float64,
        )
        base_scores = utility[None, :] + type_pref[:, None] + compat
        income = env.projected_income_table(agent_id)
        economic_value = income * float(max(1, obs.remaining_rounds)) - obs.cost_table
        return obs, base_scores, economic_value

    def _quality_order(self) -> list[int]:
        preferred_quality = int(
            np.clip(np.round(np.clip(self.params[9], 0.0, 1.0) * 2.0), 0, 2)
        )
        qualities = [preferred_quality]
        qualities += [q for q in (2, 1, 0) if q != preferred_quality]
        return qualities

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        del rng
        obs, base_scores, economic_value = self._score_table(env, agent_id)
        if not obs.valid_cells.any():
            return Action.pass_action()
        max_cells = env.n_cells
        for quality in self._quality_order():
            mask = obs.action_mask[:, :, quality].T  # (5, cells)
            mask = mask & (economic_value[:, :, quality].T > 0.0)
            if not mask.any():
                continue
            scores = base_scores.copy()
            scores[~mask] = -np.inf
            flat = int(np.argmax(scores))
            type_id, cell = divmod(flat, max_cells)
            cost = float(obs.cost_table[cell, type_id, quality])
            return Action(cell=cell, type_id=type_id, quality=quality, cost=cost)
        return Action.pass_action()

    def propose_bids(
        self,
        env: "CityEnvironment",
        agent_id: int,
        rng: np.random.Generator,
        count: int,
        exclude_cells: set[int] | None = None,
    ) -> list[Action]:
        del rng
        obs, base_scores, economic_value = self._score_table(env, agent_id)
        if not obs.valid_cells.any():
            return []
        max_cells = env.n_cells
        actions: list[Action] = []
        used: set[int] = set(exclude_cells or set())
        for quality in self._quality_order():
            if len(actions) >= count:
                break
            mask = obs.action_mask[:, :, quality].T
            mask = mask & (economic_value[:, :, quality].T > 0.0)
            for cell in used:
                if 0 <= cell < max_cells:
                    mask[:, cell] = False
            while mask.any() and len(actions) < count:
                scores = base_scores.copy()
                scores[~mask] = -np.inf
                flat = int(np.argmax(scores))
                type_id, cell = divmod(flat, max_cells)
                cell = int(cell)
                type_id = int(type_id)
                used.add(cell)
                cost = float(obs.cost_table[cell, type_id, quality])
                actions.append(
                    Action(
                        cell=cell,
                        type_id=type_id,
                        quality=quality,
                        cost=cost,
                        bid=strategic_bid(
                            float(economic_value[cell, type_id, quality]),
                            cost,
                            obs.budget,
                        ),
                        priority=len(actions),
                    )
                )
                mask[:, cell] = False
        return actions


def homogeneous_policies(
    factory,
    n_agents: int,
) -> list[Policy]:
    return [factory() for _ in range(n_agents)]
