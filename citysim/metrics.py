"""Spatial and path-dependence metrics used by E8, E9 and E11."""

from __future__ import annotations

from collections import deque

import numpy as np

from .grid import type_entropy


def _neighbour_pairs(mask: np.ndarray):
    """Yield (source_slice, target_slice) for the 8-neighbourhood."""
    h, w = mask.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ):
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
        yield (ys_a, xs_a), (ys_b, xs_b)


def morans_i(type_grid: np.ndarray, types: tuple[int, ...] = (1, 2, 3, 4, 5)) -> float:
    """Average Moran's I of the five building-type indicator variables.

    Only built cells participate.  The weights are queen contiguity.  Values
    close to +1 indicate strong spatial clustering, values near 0 randomness.
    """
    grid = np.asarray(type_grid)
    built = grid > 0
    n = int(built.sum())
    if n < 3:
        return 0.0
    values = []
    for type_id in types:
        x = np.where(built, (grid == type_id).astype(np.float64), 0.0)
        xbar = float(x[built].mean())
        dev = np.where(built, x - xbar, 0.0)
        den = float(np.sum(dev * dev))
        if den < 1e-12:
            continue
        num = 0.0
        weight_sum = 0
        for (ys_a, xs_a), (ys_b, xs_b) in _neighbour_pairs(built):
            a_mask = built[ys_a, xs_a]
            b_mask = built[ys_b, xs_b]
            both = a_mask & b_mask
            num += float(np.sum(dev[ys_a, xs_a][both] * dev[ys_b, xs_b][both]))
            weight_sum += int(both.sum())
        if weight_sum == 0:
            continue
        values.append((n / weight_sum) * num / den)
    if not values:
        return 0.0
    return float(np.clip(np.mean(values), -1.0, 1.0))


def local_entropy(type_grid: np.ndarray, radius: int = 1) -> float:
    """Mean Shannon entropy inside the (2r+1) window around built cells."""
    grid = np.asarray(type_grid)
    h, w = grid.shape
    entropies = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                continue
            window = grid[
                max(0, y - radius) : min(h, y + radius + 1),
                max(0, x - radius) : min(w, x + radius + 1),
            ]
            types = window[window > 0]
            if types.size <= 1:
                entropies.append(0.0)
                continue
            counts = np.array(
                [np.sum(types == t) for t in range(1, 6)], dtype=np.float64
            )
            counts = counts[counts > 0]
            p = counts / counts.sum()
            entropies.append(float(-np.sum(p * np.log(p)) / np.log(5.0)))
    return float(np.mean(entropies)) if entropies else 0.0


def _components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            count += 1
            queue = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
    return count


def cluster_counts(type_grid: np.ndarray) -> dict[int, int]:
    grid = np.asarray(type_grid)
    return {t: _components(grid == t) for t in range(1, 6)}


def type_distribution(type_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(type_grid)
    counts = np.array([np.sum(grid == t) for t in range(1, 6)], dtype=np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros(5, dtype=np.float64)


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau between two short vectors (no scipy dependency)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(x.size):
        for j in range(i + 1, x.size):
            sign = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return float((concordant - discordant) / total)


def grid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Share of cells whose building type differs."""
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape:
        raise ValueError("grids must have the same shape")
    return float(np.mean(x != y))


def spatial_metrics(type_grid: np.ndarray) -> dict[str, float]:
    grid = np.asarray(type_grid)
    clusters = cluster_counts(grid)
    built = int(np.sum(grid > 0))
    return {
        "morans_i": morans_i(grid),
        "local_entropy": local_entropy(grid),
        "cluster_count": float(sum(clusters.values())),
        "largest_cluster": float(max(clusters.values()) if clusters else 0),
        "green_share": float(np.sum(grid == 5) / max(1, built)),
        "type_entropy": type_entropy(grid),
        "built_cells": float(built),
    }


def resilience_metrics(history: list[dict], shock_round: int) -> dict[str, float]:
    """Compare city quality before and after an exogenous shock."""
    if not history:
        return {"pre_quality": 0.0, "post_quality": 0.0, "recovery": 0.0, "drop": 0.0}
    pre = [row["city_quality"] for row in history if int(row["round"]) < shock_round]
    post = [row["city_quality"] for row in history if int(row["round"]) >= shock_round]
    pre_mean = float(np.mean(pre)) if pre else float(history[0]["city_quality"])
    post_mean = float(np.mean(post)) if post else pre_mean
    final = float(history[-1]["city_quality"])
    drop = max(0.0, pre_mean - min(post) if post else 0.0)
    recovery = final / max(1e-9, pre_mean)
    return {
        "pre_quality": pre_mean,
        "post_quality": post_mean,
        "recovery": float(recovery),
        "drop": float(drop),
    }
