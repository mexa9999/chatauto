"""Vectorised grid primitives: box filters, layer generation, metrics."""

from __future__ import annotations

import heapq

import numpy as np

try:  # numba is optional at runtime, but strongly recommended
    from numba import njit
except Exception:  # pragma: no cover - exercised only without numba
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator

_window_sums_cache = {}

@njit(cache=True)
def _box_sum_numba(a, radius):
    h, w = a.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    for i in range(h):
        row_sum = 0.0
        for j in range(w):
            row_sum += a[i, j]
            integral[i + 1, j + 1] = integral[i, j + 1] + row_sum

    result = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            result[y, x] = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
    return result


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) square, clipped at the boundary."""
    a = np.ascontiguousarray(values, dtype=np.float64)
    return _box_sum_numba(a, int(radius))

def box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    a = np.ascontiguousarray(values, dtype=np.float64)
    key = (a.shape, radius)
    if key not in _window_sums_cache:
        ones = np.ones_like(a)
        _window_sums_cache[key] = np.maximum(1.0, box_sum(ones, radius))
    return box_sum(a, radius) / _window_sums_cache[key]


def grid_shortest_paths(
    cell_costs: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    """Return weighted 4-neighbour distances from one or more source cells."""
    costs = np.asarray(cell_costs, dtype=np.float64)
    source_mask = np.asarray(sources, dtype=bool)
    if costs.ndim != 2 or source_mask.shape != costs.shape:
        raise ValueError("cell costs and sources must have the same 2D shape")
    if np.isnan(costs).any() or np.any(costs < 0.0):
        raise ValueError("cell costs must be non-negative and not NaN")

    distances = np.full(costs.shape, np.inf, dtype=np.float64)
    valid_sources = np.argwhere(source_mask & np.isfinite(costs))
    queue: list[tuple[float, int, int]] = []
    for row, col in valid_sources:
        distances[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))

    height, width = costs.shape
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance != distances[row, col]:
            continue
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            next_cost = costs[next_row, next_col]
            if not np.isfinite(next_cost):
                continue
            edge_cost = 0.5 * (costs[row, col] + next_cost)
            candidate = distance + edge_cost
            if candidate < distances[next_row, next_col]:
                distances[next_row, next_col] = candidate
                heapq.heappush(queue, (candidate, next_row, next_col))
    return distances


def road_network_fields(
    static_road_access: np.ndarray,
    static_road_mask: np.ndarray,
    player_roads: np.ndarray,
    pending_roads: np.ndarray,
    connected_weight: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return road access, weighted player-road density, and nearby wait rounds."""
    static_roads = np.asarray(static_road_mask, dtype=bool)
    active_roads = np.asarray(player_roads, dtype=bool)
    pending = np.asarray(pending_roads, dtype=bool)
    if not (
        static_road_access.shape == static_roads.shape
        == active_roads.shape
        == pending.shape
    ):
        raise ValueError("all road layers must have the same shape")

    adjacent_roads = np.zeros_like(active_roads)
    adjacent_roads[1:, :] |= static_roads[:-1, :] | active_roads[:-1, :]
    adjacent_roads[:-1, :] |= static_roads[1:, :] | active_roads[1:, :]
    adjacent_roads[:, 1:] |= static_roads[:, :-1] | active_roads[:, :-1]
    adjacent_roads[:, :-1] |= static_roads[:, 1:] | active_roads[:, 1:]
    connected_roads = active_roads & (
        adjacent_roads | static_roads
    )
    road_weights = np.where(
        connected_roads,
        connected_weight,
        1.0,
    ) * active_roads
    density = np.clip(box_mean(road_weights, 1), 0.0, 1.0)
    access = np.clip(
        static_road_access + 0.5 * np.minimum(1.0, density),
        0.0,
        1.0,
    )
    wait_rounds = np.minimum(1.0, box_sum(pending, 1))
    return access, density, wait_rounds


def smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    """Boundary-aware box blur."""
    return box_mean(values, radius)


def normalise01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every cell to the nearest True cell.

    Returns ones when the mask is empty.  The result is normalised by the
    grid diagonal so it is in [0, 1].
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    if not mask.any():
        return np.ones((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.argwhere(mask)
    dy = yy[..., None] - coords[:, 0]
    dx = xx[..., None] - coords[:, 1]
    dist = np.sqrt(dy * dy + dx * dx).min(axis=2)
    max_dist = float(np.hypot(h - 1, w - 1))
    if max_dist < 1e-12:
        return np.zeros((h, w), dtype=np.float64)
    return np.clip(dist / max_dist, 0.0, 1.0)


def make_static_layers(rng: np.random.Generator, height: int, width: int) -> dict[str, np.ndarray]:
    """Generate one reproducible city landscape from a seed."""
    h, w = height, width
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = float(np.hypot(h - 1, w - 1)) or 1.0
    dist_center = np.clip(np.hypot(yy - cy, xx - cx) / max_dist, 0.0, 1.0)

    raw_noise = rng.normal(size=(h, w))
    smooth_noise = normalise01(smooth(raw_noise, radius=1))
    fine_noise = normalise01(rng.normal(size=(h, w)))

    # Water: one or two smooth blobs.  Water-heavy cells are not buildable.
    water = np.zeros((h, w), dtype=np.float64)
    n_blobs = 2 if rng.random() < 0.45 else 1
    for _ in range(n_blobs):
        blob_y = rng.uniform(0.0, h - 1.0)
        blob_x = rng.uniform(0.0, w - 1.0)
        scale_y = rng.uniform(0.9, 2.0)
        scale_x = rng.uniform(0.9, 2.0)
        water += np.exp(
            -((yy - blob_y) ** 2) / (2 * scale_y**2)
            - ((xx - blob_x) ** 2) / (2 * scale_x**2)
        )
    water = np.clip(water, 0.0, 1.0)
    water_threshold = max(0.55, float(np.quantile(water, 0.82)))

    relief = np.clip(0.5 * smooth_noise + 0.5 * fine_noise, 0.0, 1.0)
    soil_raw = 0.5 + 0.5 * (smooth_noise - 0.5) + 0.25 * (fine_noise - 0.5)
    soil_category = np.digitize(soil_raw, [0.33, 0.66]).astype(np.float64)
    soil = soil_category / 2.0

    road_mask = ((yy % 3 == 0) | (xx % 3 == 0)).astype(np.float64)
    road = np.clip(0.35 + 0.65 * smooth(road_mask, radius=1), 0.0, 1.0)

    land_price = np.clip(
        0.60 * (1.0 - dist_center) + 0.25 * road + 0.15 * smooth_noise,
        0.0,
        1.0,
    )
    base_demand = np.clip(
        0.45 + 0.35 * (1.0 - dist_center) + 0.20 * road + 0.10 * fine_noise,
        0.10,
        1.0,
    )

    buildable = water < water_threshold
    buildable &= relief < 0.93
    # Keep at least 65% of the grid usable even for an unlucky landscape.
    if buildable.mean() < 0.65:
        buildable = water < max(0.75, float(np.quantile(water, 0.80)))

    return {
        "dist_center": dist_center,
        "soil": soil,
        "water": np.clip(water, 0.0, 1.0),
        "water_blocked": (water >= water_threshold),
        "road": road,
        "relief": relief,
        "land_price": land_price,
        "base_demand": base_demand,
        "buildable": buildable.astype(np.float64),
        "road_mask": road_mask,
    }


def district_ids(height: int, width: int, size: int = 3) -> tuple[np.ndarray, int]:
    """Return a district label grid and the number of districts."""
    yy, xx = np.mgrid[0:height, 0:width]
    labels = (yy // size) * ((width + size - 1) // size) + (xx // size)
    labels = labels.astype(np.int64)
    # Compact labels in case the last district row/column is partial.
    unique, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(height, width).astype(np.int64), int(unique.size)


def incompatible_edges(type_grid: np.ndarray) -> tuple[int, int]:
    """Count unique 8-neighbour pairs (industry <-> housing/commercial/mixed).

    Returns (incompatible_pairs, built_pairs).
    """
    industry = type_grid == 4
    sensitive = (type_grid == 1) | (type_grid == 2) | (type_grid == 3)
    built = type_grid > 0
    incompatible = 0
    total = 0
    offsets = ((1, 0), (0, 1), (1, 1), (1, -1))
    h, w = type_grid.shape
    for dy, dx in offsets:
        if dy >= 0:
            ys_a = slice(0, h - dy)
            ys_b = slice(dy, h)
        else:
            ys_a = slice(-dy, h)
            ys_b = slice(0, h + dy)
        if dx >= 0:
            xs_a = slice(0, w - dx)
            xs_b = slice(dx, w)
        else:
            xs_a = slice(-dx, w)
            xs_b = slice(0, w + dx)
        a_ind = industry[ys_a, xs_a]
        b_ind = industry[ys_b, xs_b]
        a_sen = sensitive[ys_a, xs_a]
        b_sen = sensitive[ys_b, xs_b]
        a_built = built[ys_a, xs_a]
        b_built = built[ys_b, xs_b]
        both_built = a_built & b_built
        incompatible += int(np.sum(both_built & ((a_ind & b_sen) | (b_ind & a_sen))))
        total += int(np.sum(both_built))
    return incompatible, total


def type_entropy(type_grid: np.ndarray) -> float:
    """Normalised Shannon entropy of the five building types."""
    counts = np.array([np.sum(type_grid == t) for t in range(1, 6)], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    entropy = -float(np.sum(p * np.log(p)))
    return float(entropy / np.log(5.0))


def gini(values: np.ndarray) -> float:
    """Gini coefficient for a non-negative distribution."""
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
