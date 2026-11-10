"""Shared geometry helpers for isometric city projections."""

from __future__ import annotations

import numpy as np


def grid_bounds(
    height: int,
    width: int,
    tile_width: float,
    tile_height: float,
    max_building_height: float = 0.0,
) -> tuple[int, int]:
    return (
        int(np.ceil((height + width) * tile_width / 2.0)),
        int(np.ceil((height + width) * tile_height / 2.0 + max_building_height)),
    )


def cell_center(
    row: int,
    col: int,
    tile_width: float,
    tile_height: float,
    origin_x: float,
    origin_y: float,
    elevation: float = 0.0,
) -> tuple[float, float]:
    return (
        origin_x + (col - row) * tile_width / 2.0,
        origin_y + (col + row) * tile_height / 2.0 - elevation,
    )


def diamond(
    center_x: float, center_y: float, tile_width: float, tile_height: float
) -> list[tuple[float, float]]:
    return [
        (center_x, center_y - tile_height / 2.0),
        (center_x + tile_width / 2.0, center_y),
        (center_x, center_y + tile_height / 2.0),
        (center_x - tile_width / 2.0, center_y),
    ]


def building_height(
    type_id: int, quality: int, tile_height: float, floors: int = 1
) -> float:
    if type_id <= 0:
        return 0.0
    if type_id == 5:
        return float(tile_height * 0.25 * max(0, int(floors) - 1))
    type_scale = (0.90, 1.20, 1.05, 1.45)[min(type_id - 1, 3)]
    quality_scale = 0.75 + 0.15 * (int(np.clip(quality, 0, 2)) + 1)
    return float(tile_height * type_scale * quality_scale * np.sqrt(max(1, floors)))


def cell_faces(
    row: int,
    col: int,
    height: int,
    tile_width: float,
    tile_height: float,
    origin_x: float,
    origin_y: float,
    elevation: float = 0.0,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    center_x, ground_y = cell_center(
        row, col, tile_width, tile_height, origin_x, origin_y
    )
    ground = diamond(center_x, ground_y, tile_width, tile_height)
    if elevation <= 0.0:
        return ground, [], [], ground
    top = diamond(
        center_x, ground_y - elevation, tile_width, tile_height
    )
    left_side = [
        ground[3],
        ground[2],
        top[2],
        top[3],
    ]
    right_side = [
        ground[2],
        ground[1],
        top[1],
        top[2],
    ]
    return ground, left_side, right_side, top


def _contains(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def hit_test(
    point: tuple[float, float],
    type_grid: np.ndarray,
    quality_grid: np.ndarray,
    tile_width: float,
    tile_height: float,
    origin_x: float,
    origin_y: float,
    floors_grid: np.ndarray | None = None,
) -> int | None:
    height, width = type_grid.shape
    if floors_grid is None:
        floors_grid = np.ones(type_grid.shape, dtype=np.int8)
    for row, col in sorted(
        ((r, c) for r in range(height) for c in range(width)),
        key=lambda cell: (cell[0] + cell[1], cell[1]),
        reverse=True,
    ):
        type_id = int(type_grid[row, col])
        elevation = building_height(
            type_id,
            int(quality_grid[row, col]),
            tile_height,
            int(floors_grid[row, col]),
        )
        ground, left, right, top = cell_faces(
            row,
            col,
            height,
            tile_width,
            tile_height,
            origin_x,
            origin_y,
            elevation,
        )
        if any(_contains(point, polygon) for polygon in (left, right, top, ground) if polygon):
            return row * width + col
    return None


def darken(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)
