"""Policies: NumPy neuroevolution genome, plus random/manual/greedy baselines.

The learned policy is a small district-aware network with local attention over
the eight neighbouring cells.  Everything is evaluated with NumPy; no automatic
differentiation and no deep-learning framework is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import (
    BID_MAX,
    BID_MIN,
    MAX_FLOORS_BY_ACTION,
    N_ROOF_QUALITIES,
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
    n_floors: int = max(MAX_FLOORS_BY_ACTION)
    n_roof_qualities: int = N_ROOF_QUALITIES
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
            + h * self.n_floors + self.n_floors
            + h * self.n_roof_qualities + self.n_roof_qualities
            + h + 1                      # bid head
            + h + 1                      # demolition head
            + 7 * h * h + 4 * h          # per-cell GRU and recurrent projection
        )

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape == (self.parameter_count,):
            return vector
        raise ValueError(
            f"genome has shape {vector.shape}, expected ({self.parameter_count},)"
        )

    def unpack(self, vector: np.ndarray):
        vector = self.normalize(vector)
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
        W_floors = take((h, self.n_floors))
        b_floors = take((self.n_floors,))
        W_roof = take((h, self.n_roof_qualities))
        b_roof = take((self.n_roof_qualities,))
        w_bid = take((h,))
        b_bid = take((1,))
        w_demolish = take((h,))
        b_demolish = take((1,))
        W_gru_input = take((h, 3 * h))
        W_gru_state = take((h, 3 * h))
        b_gru = take((3 * h,))
        W_recurrent_out = take((h, h))
        b_recurrent_out = take((h,))
        return (
            Wq, Wk, Wv, b_att,
            Wd, bd, Wd_type, bd_type,
            Wg, bg,
            W1, b1, W2, b2,
            w3, b3,
            W_type, b_type,
            W_quality, b_quality,
            W_floors, b_floors,
            W_roof, b_roof,
            w_bid, b_bid, w_demolish, b_demolish,
            W_gru_input, W_gru_state, b_gru, W_recurrent_out, b_recurrent_out,
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
    normal((h, layout.n_floors), 0.02)
    put(np.zeros(layout.n_floors))
    normal((h, layout.n_roof_qualities), 0.02)
    put(np.zeros(layout.n_roof_qualities))
    normal((h,), 0.02)
    put(np.zeros(1))
    normal((h,), 0.02)
    put(np.array([-0.5]))
    normal((h, 3 * h), np.sqrt(2.0 / h))
    normal((h, 3 * h), np.sqrt(1.0 / h))
    put(np.zeros(3 * h))
    normal((h, h), 0.02)
    put(np.zeros(h))
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
            self.W_floors, self.b_floors,
            self.W_roof, self.b_roof,
            self.w_bid, self.b_bid, self.w_demolish, self.b_demolish,
            self.W_gru_input, self.W_gru_state, self.b_gru,
            self.W_recurrent_out, self.b_recurrent_out,
        ) = layout.unpack(self.genome)
        self.genome = layout.normalize(self.genome)
        self._recurrent_state: np.ndarray | None = None
        self._recurrent_shape: tuple[int, int] | None = None
        self._recurrent_enabled = bool(
            np.any(self.W_recurrent_out) or np.any(self.b_recurrent_out)
        )

    def reset_state(self) -> None:
        self._recurrent_state = None
        self._recurrent_shape = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))

    def _update_recurrent_state(
        self, cell_hidden: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        if not self._recurrent_enabled:
            return np.zeros_like(cell_hidden)
        if self._recurrent_state is None or self._recurrent_shape != (height, width):
            previous = np.zeros_like(cell_hidden)
        else:
            previous = self._recurrent_state

        input_gates = cell_hidden @ self.W_gru_input + self.b_gru
        state_gates = previous @ self.W_gru_state
        input_reset, input_update, input_candidate = np.split(input_gates, 3, axis=1)
        state_reset, state_update, state_candidate = np.split(state_gates, 3, axis=1)
        reset = self._sigmoid(input_reset + state_reset)
        update = self._sigmoid(input_update + state_update)
        candidate = np.tanh(input_candidate + reset * state_candidate)
        state = update * previous + (1.0 - update) * candidate
        self._recurrent_state = state
        self._recurrent_shape = (height, width)
        return np.maximum(state @ self.W_recurrent_out + self.b_recurrent_out, 0.0)

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
        recurrent = self._update_recurrent_state(h2, h, w)
        h2 = np.maximum(h2 + recurrent, 0.0)
        scores = h2 @ self.w3 + float(self.b3[0])
        
        type_logits = (h2 @ self.W_type + self.b_type).reshape(h, w, layout.n_types)
        quality_logits = (h2 @ self.W_quality + self.b_quality).reshape(h, w, layout.n_qualities)
        floor_logits = (h2 @ self.W_floors + self.b_floors).reshape(
            h, w, layout.n_floors
        )
        roof_logits = (h2 @ self.W_roof + self.b_roof).reshape(
            h, w, layout.n_roof_qualities
        )
        bid_fraction = 1.0 / (1.0 + np.exp(-(h2 @ self.w_bid + float(self.b_bid[0]))))
        demolish_logits = h2 @ self.w_demolish + float(self.b_demolish[0])

        # --- ПРЕДВЫЧИСЛЕНИЕ JOINT LOGITS ---
        joint_all = type_logits[:, :, :, None] + quality_logits[:, :, None, :]
        
        joint_all += _DISTRICT_ACTION_BIAS[district_type_cells]

        return {
            "scores": scores,
            "joint_all": joint_all,  # <--- НОВОЕ ПОЛЕ
            "floor_logits": floor_logits,
            "roof_logits": roof_logits,
            "bid_fraction": bid_fraction.reshape(h, w),
            "demolish_logits": demolish_logits,
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
        env: "CityEnvironment",
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
        h, w = (int(x) for x in forward["shape"])
        row, col = divmod(int(cell), w)
        budget = max(0.0, float(obs.budget))
        bid_cap = max(0.0, min(BID_MAX * budget, budget / 10.0))
        bid = float(min(max(forward["bid_fraction"][row, col] * budget, BID_MIN), bid_cap))
        floors = 1
        roof_quality = 0
        if env.config.enable_extended_actions:
            floor_limit = MAX_FLOORS_BY_ACTION[type_id]
            floors = int(
                np.argmax(forward["floor_logits"][row, col, :floor_limit]) + 1
            )
            roof_quality = int(np.argmax(forward["roof_logits"][row, col]))
        cost = env.cost_of(cell, type_id, quality, floors, roof_quality)
        while cost + bid > budget + 1e-9 and (roof_quality > 0 or floors > 1):
            if roof_quality > 0:
                roof_quality -= 1
            else:
                floors -= 1
            cost = env.cost_of(cell, type_id, quality, floors, roof_quality)
        if cost + bid > budget + 1e-9:
            return None
        return Action(
            cell=int(cell),
            type_id=int(type_id),
            quality=int(quality),
            cost=cost,
            bid=bid,
            priority=int(priority),
            floors=floors,
            roof_quality=roof_quality,
        )

    @staticmethod
    def _demolition_action(cell: int, priority: int = 0) -> Action:
        return Action(cell=int(cell), demolish=True, priority=int(priority))

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        del rng
        obs = env.observe(agent_id)
        forward = self._forward(obs)
        scores = forward["scores"].copy()
        scores[~obs.valid_cells] = -np.inf
        build_cell = int(np.argmax(scores))
        build_score = float(scores[build_cell])
        build_action = (
            self._action_at(env, obs, forward, build_cell)
            if np.isfinite(build_score)
            else None
        )
        demolition_mask = obs.demolish_mask
        if demolition_mask is not None and demolition_mask.any():
            demolition_scores = forward["scores"] + forward["demolish_logits"]
            demolition_scores[~demolition_mask] = -np.inf
            demolition_cell = int(np.argmax(demolition_scores))
            if demolition_scores[demolition_cell] > build_score:
                return self._demolition_action(demolition_cell)
        return build_action if build_action is not None else Action.pass_action()

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
        if count <= 0:
            return []
        forward = self._forward(obs)
        scores = forward["scores"].copy()
        scores[~obs.valid_cells] = -np.inf
        if exclude_cells:
            for cell in exclude_cells:
                if 0 <= int(cell) < scores.size:
                    scores[int(cell)] = -np.inf
        candidates: list[tuple[float, Action]] = []
        valid_cells = np.flatnonzero(np.isfinite(scores))
        for cell in valid_cells:
            cell = int(cell)
            joint = self._joint_logits(forward, cell, obs.action_mask)
            if np.isfinite(joint).any():
                action = self._action_at(env, obs, forward, cell)
                if action is not None:
                    candidates.append(
                        (float(scores[cell] + np.max(joint)), action)
                    )

        demolition_mask = obs.demolish_mask
        if demolition_mask is not None:
            demolition_scores = forward["scores"] + forward["demolish_logits"]
            demolition_scores[~demolition_mask] = -np.inf
            if exclude_cells:
                for cell in exclude_cells:
                    if 0 <= int(cell) < demolition_scores.size:
                        demolition_scores[int(cell)] = -np.inf
            for cell in np.flatnonzero(np.isfinite(demolition_scores)):
                action = self._demolition_action(int(cell))
                candidates.append((float(demolition_scores[cell]), action))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [
            replace(action, priority=priority)
            for priority, (_, action) in enumerate(candidates[:count])
        ]


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomPolicy:
    name = "random"

    def act(self, env: "CityEnvironment", agent_id: int, rng: np.random.Generator) -> Action:
        obs = env.observe(agent_id)
        demolition_mask = obs.demolish_mask
        demolition_cells = (
            np.flatnonzero(demolition_mask)
            if demolition_mask is not None
            else np.empty(0, dtype=np.int64)
        )
        choices = np.concatenate((np.flatnonzero(obs.valid_cells), demolition_cells))
        if choices.size == 0:
            return Action.pass_action()
        cell = int(rng.choice(choices))
        if demolition_mask is not None and demolition_mask[cell]:
            return Action(cell=cell, demolish=True)
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
        demolition_mask = obs.demolish_mask
        demolition_cells = (
            np.flatnonzero(demolition_mask)
            if demolition_mask is not None
            else np.empty(0, dtype=np.int64)
        )
        valid_cells = [
            int(c) for c in np.concatenate((np.flatnonzero(obs.valid_cells), demolition_cells))
            if not exclude_cells or int(c) not in exclude_cells
        ]
        if not valid_cells:
            return []
        chosen = rng.choice(
            valid_cells, size=min(count, len(valid_cells)), replace=False
        )
        actions: list[Action] = []
        for priority, cell in enumerate(int(c) for c in np.atleast_1d(chosen)):
            if demolition_mask is not None and demolition_mask[cell]:
                actions.append(Action(cell=cell, demolish=True, priority=priority))
                continue
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
        if obs.remaining_rounds <= 0:
            return Action.pass_action()
        best_build_value = 0.0
        best_build_action = Action.pass_action()
        if obs.valid_cells.any():
            income = env.projected_income_table(agent_id)
            value = income * float(obs.remaining_rounds) - obs.cost_table
            value[~obs.action_mask] = -np.inf
            flat = int(np.argmax(value))
            if np.isfinite(value.ravel()[flat]) and value.ravel()[flat] > 0.0:
                best_build_value = float(value.ravel()[flat])
                cell, type_id, quality = np.unravel_index(flat, value.shape)
                best_build_action = Action(
                    cell=int(cell),
                    type_id=int(type_id),
                    quality=int(quality),
                    cost=float(obs.cost_table[cell, type_id, quality]),
                )
        demolition = self._best_demolition(obs)
        if demolition is not None and demolition[0] > best_build_value:
            return demolition[1]
        return best_build_action

    @staticmethod
    def _best_demolition(obs: "Observation") -> tuple[float, Action] | None:
        if obs.demolish_mask is None or obs.demolish_value is None:
            return None
        values = np.asarray(obs.demolish_value).copy()
        values[~obs.demolish_mask] = -np.inf
        cell = int(np.argmax(values))
        if not np.isfinite(values[cell]) or values[cell] <= 0.0:
            return None
        return float(values[cell]), Action(cell=cell, demolish=True)

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
        if count <= 0:
            return []
        candidates: list[tuple[float, Action]] = []
        used: set[int] = set(exclude_cells or set())
        if obs.valid_cells.any():
            income = env.projected_income_table(agent_id)
            value = income * float(max(1, obs.remaining_rounds)) - obs.cost_table
            value[~obs.action_mask] = -np.inf
            order = np.argsort(value, axis=None)[::-1]
            for flat in order:
                if not np.isfinite(value.ravel()[flat]) or value.ravel()[flat] <= 0.0:
                    break
                cell, type_id, quality = np.unravel_index(int(flat), value.shape)
                cell, type_id, quality = int(cell), int(type_id), int(quality)
                if cell in used:
                    continue
                used.add(cell)
                cost = float(obs.cost_table[cell, type_id, quality])
                candidates.append(
                    (
                        float(value[cell, type_id, quality]),
                        Action(
                            cell=cell,
                            type_id=type_id,
                            quality=quality,
                            cost=cost,
                            bid=strategic_bid(
                                float(value[cell, type_id, quality]), cost, obs.budget
                            ),
                        ),
                    )
                )
                if len(candidates) >= count:
                    break
        demolition_values = self._demolition_values(obs)
        demolition_order = np.argsort(demolition_values)[::-1]
        demolition_count = 0
        for cell in demolition_order:
            if not np.isfinite(demolition_values[cell]):
                break
            if int(cell) in used:
                continue
            used.add(int(cell))
            candidates.append(
                (float(demolition_values[cell]), Action(cell=int(cell), demolish=True))
            )
            demolition_count += 1
            if demolition_count >= count:
                break
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [
            replace(action, priority=priority)
            for priority, (_, action) in enumerate(candidates[:count])
        ]

    @staticmethod
    def _demolition_values(obs: "Observation") -> np.ndarray:
        if obs.demolish_mask is None or obs.demolish_value is None:
            return np.full(obs.valid_cells.shape, -np.inf)
        values = np.asarray(obs.demolish_value).copy()
        values[~obs.demolish_mask] = -np.inf
        values[values <= 0.0] = -np.inf
        return values


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
        max_cells = env.n_cells
        build_action = Action.pass_action()
        build_value = 0.0
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
            build_action = Action(
                cell=cell, type_id=type_id, quality=quality, cost=cost
            )
            build_value = float(economic_value[cell, type_id, quality])
            break
        demolition = GreedyPolicy._best_demolition(obs)
        if demolition is not None and demolition[0] > build_value:
            return demolition[1]
        return build_action

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
        if count <= 0:
            return []
        max_cells = env.n_cells
        candidates: list[tuple[float, Action]] = []
        used: set[int] = set(exclude_cells or set())
        for quality in self._quality_order():
            if len(candidates) >= count:
                break
            mask = obs.action_mask[:, :, quality].T
            mask = mask & (economic_value[:, :, quality].T > 0.0)
            for cell in used:
                if 0 <= cell < max_cells:
                    mask[:, cell] = False
            while mask.any() and len(candidates) < count:
                scores = base_scores.copy()
                scores[~mask] = -np.inf
                flat = int(np.argmax(scores))
                type_id, cell = divmod(flat, max_cells)
                cell = int(cell)
                type_id = int(type_id)
                used.add(cell)
                cost = float(obs.cost_table[cell, type_id, quality])
                surplus = float(economic_value[cell, type_id, quality])
                candidates.append(
                    (
                        surplus,
                        Action(
                            cell=cell,
                            type_id=type_id,
                            quality=quality,
                            cost=cost,
                            bid=strategic_bid(surplus, cost, obs.budget),
                        ),
                    )
                )
                mask[:, cell] = False
        demolition_values = GreedyPolicy._demolition_values(obs)
        demolition_order = np.argsort(demolition_values)[::-1]
        demolition_count = 0
        for cell in demolition_order:
            if not np.isfinite(demolition_values[cell]):
                break
            if int(cell) in used:
                continue
            candidates.append(
                (
                    float(demolition_values[cell]),
                    Action(cell=int(cell), demolish=True),
                )
            )
            demolition_count += 1
            if demolition_count >= count:
                break
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [
            replace(action, priority=priority)
            for priority, (_, action) in enumerate(candidates[:count])
        ]


def homogeneous_policies(
    factory,
    n_agents: int,
) -> list[Policy]:
    return [factory() for _ in range(n_agents)]
