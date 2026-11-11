"""Real-time pygame visualizer: live charts, cell inspection, side-by-side mode."""

from __future__ import annotations

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is optional
    pygame = None

import numpy as np

from .config import ACTION_TYPE_NAMES, MAX_FLOORS_BY_ACTION, TYPE_NAMES
from .isometric import (
    building_height,
    cell_center,
    cell_faces,
    darken,
    grid_bounds,
    hit_test,
)


# Cell type colors (RGB)
TYPE_COLORS = [
    (242, 242, 242),   # empty
    (76, 120, 168),    # housing
    (245, 133, 24),    # commercial
    (157, 117, 93),    # mixed
    (182, 66, 60),     # industry
    (84, 162, 75),     # green
]

# Distinct colors for up to 20 agents
AGENT_COLORS = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (0, 130, 200),   (245, 130, 48),
    (145, 30, 180),  (70, 240, 240),  (240, 50, 230),  (210, 245, 60),  (250, 190, 190),
    (0, 128, 128),   (170, 110, 40),  (128, 0, 0),     (170, 255, 195), (128, 128, 0),
    (255, 215, 180), (0, 0, 128),     (128, 128, 128), (255, 105, 180), (139, 69, 19),
]

PANEL_BG = (24, 24, 24)
GRID_BG = (28, 28, 28)
TEXT = (235, 239, 245)
TEXT_DIM = (195, 202, 212)
ACCENT = (90, 190, 255)
GREEN_ROOF = (84, 162, 75)
ACTIVE_ROAD = (94, 105, 117)
PENDING_ROAD = (232, 184, 70)


def _roof_color(color: tuple[int, int, int], quality: int) -> tuple[int, int, int]:
    blend = float(np.clip(quality, 0, 3)) / 3.0 * 0.6
    return tuple(
        int(base * (1.0 - blend) + green * blend)
        for base, green in zip(color, GREEN_ROOF)
    )


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _behavior_diversity(env) -> float:
    agents = env.agent_metrics()
    if len(agents) < 2:
        return 0.0
    points = np.array([a.behavior for a in agents], dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(points.shape[0], k=1)
    return float(np.mean(dist[iu]))


class _BaseVisualizer:
    """Shared pygame plumbing for single and comparison windows."""

    def __init__(self, speed_ms: int = 80):
        if pygame is None:
            raise RuntimeError(
                "pygame is required for --live mode.\n"
                "Install with: pip install pygame"
            )
        pygame.init()
        self.speed_ms = int(speed_ms)
        self.paused = False
        self.running = True
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("segoeui", 16)
        self.font = pygame.font.SysFont("segoeui", 17)
        self.font_header = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_tiny = pygame.font.SysFont("segoeui", 14)

    def tick(self) -> None:
        if self.paused:
            for event in pygame.event.get():
                self._handle_event(event)
            self.clock.tick(30)
            return
        elapsed = 0
        delay = max(10, self.speed_ms)
        while elapsed < delay and self.running:
            dt = self.clock.tick(60)
            elapsed += dt
            for event in pygame.event.get():
                self._handle_event(event)

    def _handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
                self.draw()
            elif event.key == pygame.K_UP:
                self.speed_ms = min(500, self.speed_ms + 20)
            elif event.key == pygame.K_DOWN:
                self.speed_ms = max(10, self.speed_ms - 20)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos)

    def _on_click(self, pos) -> None:  # pragma: no cover - GUI only
        pass

    def draw(self) -> None:  # pragma: no cover - GUI only
        raise NotImplementedError

    def close(self) -> None:
        pygame.quit()

    # -- testable helpers -------------------------------------------------

    @staticmethod
    def _chart_values(series: list[float], lo: float, hi: float) -> list[float]:
        if hi - lo < 1e-9:
            hi = lo + 1.0
        return [float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) for v in series]

    def _draw_chart(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        series: list[list[float]],
        colors: list[tuple[int, int, int]],
        labels: list[str] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        every: int = 1,
    ) -> None:
        pygame.draw.rect(self.screen, (34, 34, 34), (x, y, w, h))
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, w, h), 1)
        self.screen.blit(self.font_tiny.render(title, True, TEXT_DIM), (x + 6, y + 3))
        plot_x = x + 36
        plot_y = y + 23
        plot_w = max(10, w - 44)
        plot_h = max(6, h - 35)
        if not series or not series[0]:
            self.screen.blit(
                self.font_tiny.render("collecting data...", True, TEXT_DIM),
                (plot_x, plot_y + plot_h // 2 - 6),
            )
            return
        all_values = [v for line in series for v in line]
        lo = min(all_values) if vmin is None else vmin
        hi = max(all_values) if vmax is None else vmax
        if hi - lo < 1e-9:
            hi = lo + 1.0
        n = max(len(line) for line in series)
        for line, color in zip(series, colors):
            if len(line) < 1:
                continue
            points = []
            for i, value in enumerate(line):
                px = plot_x + (i / max(1, n - 1)) * plot_w
                py = plot_y + plot_h - float(np.clip((value - lo) / (hi - lo), 0.0, 1.0)) * plot_h
                points.append((int(px), int(py)))
            if len(points) == 1:
                pygame.draw.circle(self.screen, color, points[0], 2)
            else:
                pygame.draw.lines(self.screen, color, False, points, 2)
        self.screen.blit(self.font_tiny.render(f"{hi:.2f}", True, TEXT_DIM), (x + 3, plot_y - 2))
        self.screen.blit(
            self.font_tiny.render(f"{lo:.2f}", True, TEXT_DIM),
            (x + 3, plot_y + plot_h - 8),
        )
        if labels:
            label = "  ".join(labels[:4])
            self.screen.blit(
                self.font_tiny.render(label, True, TEXT_DIM), (plot_x, y + h - 15)
            )


class CityVisualizer(_BaseVisualizer):
    """Pygame window that renders one city, its agents and four live charts."""

    def __init__(
        self,
        env,
        cell_size: int = 36,
        speed_ms: int = 80,
        projection: str = "isometric",
    ):
        super().__init__(speed_ms=speed_ms)
        if projection not in ("isometric", "top-down"):
            raise ValueError("projection must be 'isometric' or 'top-down'")
        self.env = env
        self.h = env.h
        self.w = env.w
        self.n_agents = len(env.agents)
        self.projection = projection

        self.panel_width = 410
        max_grid_w = 1060
        scale_denominator = (
            (self.h + self.w) / 2.0
            if projection == "isometric"
            else max(self.h, self.w)
        )
        self.cell_size = min(
            cell_size,
            max(14, int((max_grid_w - self.panel_width) / scale_denominator)),
        )
        self.tile_width = float(self.cell_size)
        self.tile_height = max(8.0, self.cell_size / 2.0)
        self.header_h = 42
        self.footer_h = 26
        self.max_building_height = max(
            building_height(
                type_id,
                quality,
                self.tile_height,
                MAX_FLOORS_BY_ACTION[type_id - 1],
            )
            for type_id in range(1, 6)
            for quality in range(3)
        )
        if projection == "isometric":
            self.grid_w, self.grid_h = grid_bounds(
                self.h,
                self.w,
                self.tile_width,
                self.tile_height,
                self.max_building_height,
            )
        else:
            self.grid_w = self.w * self.cell_size
            self.grid_h = self.h * self.cell_size
        self.width = self.grid_w + self.panel_width
        self.height = max(self.grid_h + self.header_h + self.footer_h + 8, 720)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CitySim — live simulation")

        self.latest_action: dict[int, tuple[int | None, int, bool]] = {}
        self.round_idx = 0
        self.selected_cell: int | None = None
        self.action_log: list[str] = []
        self.max_log = 9

        self.agent_builds = [0] * self.n_agents
        self.agent_skips = [0] * self.n_agents
        self.agent_profits = [0.0] * self.n_agents
        self.agent_damage = [0.0] * self.n_agents
        self.profit_series: list[list[float]] = [[] for _ in range(self.n_agents)]
        self.quality_series: list[float] = []
        self.satisfaction_series: list[float] = []
        self.util_series: dict[str, list[float]] = {
            "housing": [],
            "commercial": [],
            "industry": [],
            "mean": [],
        }
        self.diversity_series: list[float] = []

    # -- drawing ----------------------------------------------------------

    def _draw_grid(self) -> None:
        if self.projection == "isometric":
            self._draw_isometric_grid()
            return
        for y in range(self.h):
            for x in range(self.w):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size + self.header_h,
                    self.cell_size,
                    self.cell_size,
                )
                type_id = int(self.env.type_grid[y, x])
                pygame.draw.rect(self.screen, TYPE_COLORS[type_id], rect)
                pygame.draw.rect(self.screen, (55, 55, 55), rect, 1)
                if self.env.player_road_grid[y, x]:
                    road_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, ACTIVE_ROAD, road_rect)
                    pygame.draw.line(
                        self.screen, (205, 211, 218),
                        road_rect.midleft, road_rect.midright, 1,
                    )
                elif self.env.pending_road_grid[y, x] > 0:
                    road_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, PENDING_ROAD, road_rect, 2)

                if type_id > 0:
                    owner = int(self.env.owner_grid[y, x])
                    if 0 <= owner < self.n_agents:
                        r = max(3, self.cell_size // 6)
                        pygame.draw.circle(
                            self.screen,
                            AGENT_COLORS[owner % len(AGENT_COLORS)],
                            (rect.right - r - 2, rect.top + r + 2),
                            r,
                        )

                cell_id = y * self.w + x
                for agent_id, (cell, round_at, ok) in self.latest_action.items():
                    if cell is not None and cell == cell_id and round_at == self.round_idx:
                        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
                        pygame.draw.rect(self.screen, color, rect, 3)
                        break

                if self.selected_cell == cell_id:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)

    def _isometric_origin(self) -> tuple[float, float]:
        return (
            self.h * self.tile_width / 2.0,
            self.header_h + self.max_building_height + self.tile_height / 2.0,
        )

    def _draw_isometric_grid(self) -> None:
        origin_x, origin_y = self._isometric_origin()
        for row, col in sorted(
            ((r, c) for r in range(self.h) for c in range(self.w)),
            key=lambda cell: (cell[0] + cell[1], cell[1]),
        ):
            type_id = int(self.env.type_grid[row, col])
            elevation = building_height(
                type_id,
                int(self.env.quality_grid[row, col]),
                self.tile_height,
                int(self.env.floors_grid[row, col]),
            )
            ground, left, right, top = cell_faces(
                row,
                col,
                self.h,
                self.tile_width,
                self.tile_height,
                origin_x,
                origin_y,
                elevation,
            )
            pygame.draw.polygon(
                self.screen,
                TYPE_COLORS[type_id],
                [(int(x), int(y)) for x, y in ground],
            )
            if self.env.player_road_grid[row, col]:
                pygame.draw.polygon(self.screen, ACTIVE_ROAD, ground)
                pygame.draw.polygon(self.screen, (205, 211, 218), ground, 1)
            elif self.env.pending_road_grid[row, col] > 0:
                pygame.draw.polygon(self.screen, PENDING_ROAD, ground, 2)
            if left:
                pygame.draw.polygon(
                    self.screen,
                    darken(TYPE_COLORS[type_id], 0.58),
                    [(int(x), int(y)) for x, y in left],
                )
                pygame.draw.polygon(
                    self.screen,
                    (55, 55, 55),
                    [(int(x), int(y)) for x, y in left],
                    1,
                )
            if right:
                pygame.draw.polygon(
                    self.screen,
                    darken(TYPE_COLORS[type_id], 0.76),
                    [(int(x), int(y)) for x, y in right],
                )
                pygame.draw.polygon(
                    self.screen,
                    (55, 55, 55),
                    [(int(x), int(y)) for x, y in right],
                    1,
                )
            pygame.draw.polygon(
                self.screen,
                _roof_color(
                    TYPE_COLORS[type_id],
                    int(self.env.roof_quality_grid[row, col]),
                ),
                [(int(x), int(y)) for x, y in top],
            )
            pygame.draw.polygon(
                self.screen, (55, 55, 55), [(int(x), int(y)) for x, y in top], 1
            )
            if type_id == 0:
                if self.env.player_road_grid[row, col]:
                    pygame.draw.polygon(self.screen, ACTIVE_ROAD, ground)
                    pygame.draw.polygon(self.screen, (205, 211, 218), ground, 1)
                elif self.env.pending_road_grid[row, col] > 0:
                    pygame.draw.polygon(self.screen, PENDING_ROAD, ground, 2)

            if type_id > 0:
                owner = int(self.env.owner_grid[row, col])
                if 0 <= owner < self.n_agents:
                    center = cell_center(
                        row,
                        col,
                        self.tile_width,
                        self.tile_height,
                        origin_x,
                        origin_y,
                        elevation,
                    )
                    pygame.draw.circle(
                        self.screen,
                        AGENT_COLORS[owner % len(AGENT_COLORS)],
                        (int(center[0]), int(center[1])),
                        max(2, self.cell_size // 8),
                    )

            cell_id = row * self.w + col
            for agent_id, (cell, round_at, _) in self.latest_action.items():
                if cell is not None and cell == cell_id and round_at == self.round_idx:
                    pygame.draw.polygon(
                        self.screen,
                        AGENT_COLORS[agent_id % len(AGENT_COLORS)],
                        [(int(x), int(y)) for x, y in top],
                        3,
                    )
                    break
            if self.selected_cell == cell_id:
                pygame.draw.polygon(self.screen, (255, 255, 255), top, 2)

    def _draw_header(self) -> None:
        built = int((self.env.type_grid > 0).sum())
        roads = int(self.env.player_road_grid.sum())
        pending_roads = int((self.env.pending_road_grid > 0).sum())
        total = self.h * self.w
        status = "PAUSED" if self.paused else "RUNNING"
        title = self.font_header.render(
            f"CitySim  |  Round {self.round_idx + 1}/{self.env.config.actions_per_agent}"
            f"  |  Built: {built}/{total}"
            f"  |  Roads: {roads}+{pending_roads}"
            f"  |  {self.projection}  |  {status}"
            f"  |  Space pause, ↑/↓ speed, Esc quit, click a cell",
            True,
            (240, 240, 240),
        )
        self.screen.blit(title, (10, 12))

    def _draw_legend_footer(self) -> None:
        x = 8
        y = self.header_h + self.grid_h + 5
        for tid, name in enumerate(TYPE_NAMES):
            pygame.draw.rect(self.screen, TYPE_COLORS[tid], (x, y, 12, 12))
            pygame.draw.rect(self.screen, (80, 80, 80), (x, y, 12, 12), 1)
            label = self.font_tiny.render(name, True, TEXT_DIM)
            self.screen.blit(label, (x + 15, y - 1))
            x += 18 + label.get_width()
        for color, name, outline in (
            (ACTIVE_ROAD, "road", False),
            (PENDING_ROAD, "pending", True),
        ):
            pygame.draw.rect(self.screen, color, (x, y, 12, 12), 1 if outline else 0)
            label = self.font_tiny.render(name, True, TEXT_DIM)
            self.screen.blit(label, (x + 15, y - 1))
            x += 18 + label.get_width()

    def _draw_agent_table(self, px: int, y: int) -> int:
        header = self.font_header.render("Agents", True, (220, 220, 220))
        self.screen.blit(header, (px, y))
        y += 25
        if self.n_agents <= 12:
            indices = list(range(self.n_agents))
        else:
            indices = list(
                np.argsort(self.agent_profits)[::-1][:11].astype(int).tolist()
            )
        for i in indices:
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            pygame.draw.circle(self.screen, color, (px + 6, y + 7), 5)
            bankrupt = self.env.agents[i].bankrupt
            text = (
                f"A{i:02d} B{self.agent_builds[i]:>2} S{self.agent_skips[i]:>2} "
                f"P{self.agent_profits[i]:>+6.0f} D{self.agent_damage[i]:>5.0f}"
                f"{'  BANKRUPT' if bankrupt else ''}"
            )
            label = self.font_small.render(text, True, (200, 200, 200))
            self.screen.blit(label, (px + 18, y))
            y += 21
        if self.n_agents > 12:
            self.screen.blit(
                self.font_tiny.render("(top 11 by profit)", True, TEXT_DIM),
                (px + 18, y),
            )
            y += 19
        return y

    def _draw_selected(self, px: int, y: int) -> int:
        if self.selected_cell is None:
            return y
        cell = int(self.selected_cell)
        row, col = divmod(cell, self.w)
        type_id = int(self.env.type_grid[row, col])
        owner = int(self.env.owner_grid[row, col])
        is_road = bool(
            self.env.player_road_grid[row, col]
            or self.env.pending_road_grid[row, col] > 0
        )
        if is_road:
            owner = int(self.env.road_owner_grid[row, col])
        quality = int(self.env.quality_grid[row, col])
        cost = float(
            self.env.road_cost_grid[row, col]
            if is_road
            else self.env.cost_grid[row, col]
        )
        district = int(self.env.district_labels[row, col])
        lines = [
            f"Cell {cell} ({row},{col})  district {district}",
            f"type {'road' if is_road else TYPE_NAMES[type_id]}"
            f"  owner {owner if owner >= 0 else '-'}"
            f"  quality {quality if quality >= 0 and not is_road else '-'}",
            f"floors {int(self.env.floors_grid[row, col]) if not is_road else '-'}"
            f"  roof {int(self.env.roof_quality_grid[row, col])}/3"
            if not is_road
            else "floors -  roof -",
            f"synergy {self.env.district_synergy[row, col]:.2f}",
            f"cost {cost:.1f}  land {self.env.land_value[row, col]:.2f}",
            f"road access {self.env.road_access[row, col]:.2f}"
            f"  density {self.env.player_road_density[row, col]:.2f}",
            f"road status {'pending' if self.env.pending_road_grid[row, col] > 0 else 'active' if self.env.player_road_grid[row, col] else '-'}"
            f"  wait {self.env.infra_wait_rounds[row, col]:.0f}",
            f"pollution {self.env.pollution[row, col]:.2f}"
            f"  noise {self.env.noise[row, col]:.2f}",
            f"attractiveness {self.env.attractiveness[row, col]:.2f}"
            f"  util {self.env.util_grid[row, col]:.2f}",
        ]
        pygame.draw.rect(
            self.screen, (40, 40, 40), (px - 2, y - 2, 395, 19 * len(lines) + 4)
        )
        for line in lines:
            self.screen.blit(self.font_tiny.render(line, True, ACCENT), (px + 2, y))
            y += 19
        return y + 4

    def _draw_panel(self) -> None:
        px = self.grid_w + 12
        y = self.header_h + 2
        y = self._draw_agent_table(px, y)
        y += 6
        header = self.font_header.render("Recent actions", True, (220, 220, 220))
        self.screen.blit(header, (px, y))
        y += 24
        for line in self.action_log[-self.max_log:]:
            self.screen.blit(self.font_small.render(line, True, TEXT_DIM), (px, y))
            y += 19
        y += 4
        y = self._draw_selected(px, y)

        # Four live charts anchored to the bottom of the panel.
        chart_x = px
        chart_w = self.panel_width - 22
        chart_h = 66
        bottom = self.height - 8
        charts = [
            (
                "Profit by agent",
                [self.profit_series[i] for i in range(min(8, self.n_agents))],
                [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(min(8, self.n_agents))],
                None,
            ),
            (
                "City quality / mean satisfaction",
                [self.quality_series, self.satisfaction_series],
                [(76, 120, 168), (84, 162, 75)],
                ["quality", "satisfaction"],
            ),
            (
                "Utilization: housing / commercial / industry / mean",
                [
                    self.util_series["housing"],
                    self.util_series["commercial"],
                    self.util_series["industry"],
                    self.util_series["mean"],
                ],
                [(76, 120, 168), (245, 133, 24), (182, 66, 60), (200, 200, 200)],
                ["housing", "commerce", "industry", "mean"],
            ),
            (
                "Behavioural diversity",
                [self.diversity_series],
                [(230, 25, 75)],
                ["mean pairwise distance"],
            ),
        ]
        y_chart = bottom - chart_h * len(charts) - 4 * (len(charts) - 1)
        if y_chart < y + 4:
            # Very small window: show only the two most important charts.
            charts = charts[1:3]
            y_chart = bottom - chart_h * len(charts) - 4 * (len(charts) - 1)
        for index, (title, series, colors, labels) in enumerate(charts):
            vmin = 0.0 if index > 0 else None
            vmax = 1.0 if index > 0 else None
            self._draw_chart(
                chart_x,
                y_chart + index * (chart_h + 4),
                chart_w,
                chart_h,
                title,
                [line for line in series if line],
                colors,
                labels=labels,
                vmin=vmin,
                vmax=vmax,
            )

    def draw(self) -> None:
        self.screen.fill(GRID_BG)
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (self.grid_w, 0, self.panel_width, self.height),
        )
        self._draw_header()
        self._draw_grid()
        self._draw_legend_footer()
        self._draw_panel()
        pygame.display.flip()

    # -- events -----------------------------------------------------------

    def _on_click(self, pos) -> None:
        x, y = pos
        if x >= self.grid_w or y < self.header_h:
            return
        if self.projection == "isometric":
            origin_x, origin_y = self._isometric_origin()
            cell = hit_test(
                (x, y),
                self.env.type_grid,
                self.env.quality_grid,
                self.tile_width,
                self.tile_height,
                origin_x,
                origin_y,
                self.env.floors_grid,
            )
            if cell is not None:
                self.selected_cell = cell
                self.draw()
            return
        col = x // self.cell_size
        row = (y - self.header_h) // self.cell_size
        if 0 <= col < self.w and 0 <= row < self.h:
            self.selected_cell = int(row * self.w + col)
            self.draw()

    # -- simulation hooks -------------------------------------------------

    def start_round(self, round_idx: int) -> None:
        self.round_idx = int(round_idx)

    def handle_action(self, agent_id: int, action, success: bool) -> None:
        if not self.running:
            return
        if success and action.cell is not None:
            self.agent_builds[agent_id] += 1
            if action.demolish:
                action_name = "DEMO"
            else:
                action_name = ACTION_TYPE_NAMES[action.type_id][:4].upper()
            bid = f" bid{action.bid:.0f}" if action.bid else ""
            road_action = action.type_id == len(ACTION_TYPE_NAMES) - 1
            dimensions = (
                ""
                if action.demolish or road_action
                else f" f{action.floors} r{action.roof_quality}"
            )
            quality = "" if action.demolish or road_action else f" q{action.quality}"
            self.action_log.append(
                f"R{self.round_idx + 1} A{agent_id:02d} {action_name:>4}"
                f"{quality}{dimensions} @{action.cell}{bid}"
            )
        else:
            self.agent_skips[agent_id] += 1
            self.action_log.append(f"R{self.round_idx + 1} A{agent_id:02d} SKIP")
        if len(self.action_log) > 80:
            self.action_log = self.action_log[-60:]
        self.latest_action[agent_id] = (action.cell, self.round_idx, success)
        self._update_stats()
        self.draw()
        self.tick()

    def _update_stats(self) -> None:
        for agent in self.env.agents:
            self.agent_profits[agent.agent_id] = agent.profit
            self.agent_damage[agent.agent_id] = agent.damage_caused

    def end_round_pause(self) -> None:
        if not self.running:
            return
        self._update_stats()
        self._record_round()
        self.draw()
        for _ in range(5):
            self.tick()
            if not self.running:
                break

    def _record_round(self) -> None:
        metrics = self.env.city_metrics()
        for i, agent in enumerate(self.env.agents):
            self.profit_series[i].append(float(agent.profit))
        self.quality_series.append(float(metrics.city_quality))
        self.satisfaction_series.append(float(metrics.mean_satisfaction))
        self.util_series["housing"].append(
            float(self.env.utilization.get(1, 0.0))
        )
        self.util_series["commercial"].append(
            float(self.env.utilization.get(2, 0.0))
        )
        self.util_series["industry"].append(
            float(self.env.utilization.get(4, 0.0))
        )
        self.util_series["mean"].append(float(metrics.mean_utilization))
        self.diversity_series.append(_behavior_diversity(self.env))


class ComparisonVisualizer(_BaseVisualizer):
    """Two cities in one window, with shared rounds and side-by-side metrics."""

    def __init__(
        self,
        env_a,
        env_b,
        label_a: str = "A",
        label_b: str = "B",
        cell_size: int = 28,
        speed_ms: int = 60,
        projection: str = "isometric",
    ):
        super().__init__(speed_ms=speed_ms)
        if projection not in ("isometric", "top-down"):
            raise ValueError("projection must be 'isometric' or 'top-down'")
        self.envs = {"a": env_a, "b": env_b}
        self.labels = {"a": label_a, "b": label_b}
        self.h = env_a.h
        self.w = env_a.w
        self.n_agents = len(env_a.agents)
        self.projection = projection

        self.panel_width = 320
        max_total_grid = 1320
        scale_denominator = (
            self.h + self.w
            if projection == "isometric"
            else 2 * max(self.h, self.w)
        )
        base = max(
            12,
            int((max_total_grid - self.panel_width - 16) / scale_denominator),
        )
        self.cell_size = min(cell_size, base)
        self.tile_width = float(self.cell_size)
        self.tile_height = max(8.0, self.cell_size / 2.0)
        self.header_h = 40
        self.footer_h = 24
        self.max_building_height = max(
            building_height(
                type_id,
                quality,
                self.tile_height,
                MAX_FLOORS_BY_ACTION[type_id - 1],
            )
            for type_id in range(1, 6)
            for quality in range(3)
        )
        if projection == "isometric":
            self.grid_w, self.grid_h = grid_bounds(
                self.h,
                self.w,
                self.tile_width,
                self.tile_height,
                self.max_building_height,
            )
        else:
            self.grid_w = self.w * self.cell_size
            self.grid_h = self.h * self.cell_size
        self.width = 2 * self.grid_w + 8 + self.panel_width
        self.height = max(self.grid_h + self.header_h + self.footer_h + 6, 640)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CitySim — strategy comparison")

        self.round_idx = 0
        self.selected: dict[str, int | None] = {"a": None, "b": None}
        self.latest_action: dict[str, dict[int, tuple[int | None, int, bool]]] = {
            "a": {},
            "b": {},
        }
        self.action_log: list[str] = []
        self.quality: dict[str, list[float]] = {"a": [], "b": []}
        self.satisfaction: dict[str, list[float]] = {"a": [], "b": []}
        self.profit_mean: dict[str, list[float]] = {"a": [], "b": []}
        self.built: dict[str, list[float]] = {"a": [], "b": []}

    # -- drawing ----------------------------------------------------------

    def _grid_origin(self, side: str) -> int:
        return 0 if side == "a" else self.grid_w + 8

    def _draw_grid(self, side: str) -> None:
        if self.projection == "isometric":
            self._draw_isometric_grid(side)
            return
        env = self.envs[side]
        x0 = self._grid_origin(side)
        for y in range(self.h):
            for x in range(self.w):
                rect = pygame.Rect(
                    x0 + x * self.cell_size,
                    y * self.cell_size + self.header_h,
                    self.cell_size,
                    self.cell_size,
                )
                type_id = int(env.type_grid[y, x])
                pygame.draw.rect(self.screen, TYPE_COLORS[type_id], rect)
                pygame.draw.rect(self.screen, (55, 55, 55), rect, 1)
                if env.player_road_grid[y, x]:
                    road_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, ACTIVE_ROAD, road_rect)
                    pygame.draw.line(
                        self.screen, (205, 211, 218),
                        road_rect.midleft, road_rect.midright, 1,
                    )
                elif env.pending_road_grid[y, x] > 0:
                    road_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, PENDING_ROAD, road_rect, 2)
                if type_id > 0:
                    owner = int(env.owner_grid[y, x])
                    if 0 <= owner < self.n_agents:
                        r = max(2, self.cell_size // 6)
                        pygame.draw.circle(
                            self.screen,
                            AGENT_COLORS[owner % len(AGENT_COLORS)],
                            (rect.right - r - 2, rect.top + r + 2),
                            r,
                        )
                cell_id = y * self.w + x
                for agent_id, (cell, round_at, ok) in self.latest_action[side].items():
                    if cell is not None and cell == cell_id and round_at == self.round_idx:
                        pygame.draw.rect(
                            self.screen,
                            AGENT_COLORS[agent_id % len(AGENT_COLORS)],
                            rect,
                            2,
                        )
                        break
                if self.selected[side] == cell_id:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
        label = self.font_header.render(
            f"{side.upper()}: {self.labels[side]}", True, (240, 240, 240)
        )
        self.screen.blit(label, (x0 + 4, self.header_h - 18))

    def _draw_isometric_grid(self, side: str) -> None:
        env = self.envs[side]
        x0 = self._grid_origin(side)
        origin_x = self.h * self.tile_width / 2.0
        origin_y = self.header_h + self.max_building_height + self.tile_height / 2.0
        for row, col in sorted(
            ((r, c) for r in range(self.h) for c in range(self.w)),
            key=lambda cell: (cell[0] + cell[1], cell[1]),
        ):
            type_id = int(env.type_grid[row, col])
            elevation = building_height(
                type_id,
                int(env.quality_grid[row, col]),
                self.tile_height,
                int(env.floors_grid[row, col]),
            )
            ground, left, right, top = cell_faces(
                row,
                col,
                self.h,
                self.tile_width,
                self.tile_height,
                origin_x,
                origin_y,
                elevation,
            )
            shifted_ground = [(int(x0 + x), int(y)) for x, y in ground]
            if env.player_road_grid[row, col]:
                pygame.draw.polygon(self.screen, ACTIVE_ROAD, shifted_ground)
                pygame.draw.polygon(
                    self.screen, (205, 211, 218), shifted_ground, 1
                )
            elif env.pending_road_grid[row, col] > 0:
                pygame.draw.polygon(
                    self.screen, PENDING_ROAD, shifted_ground, 2
                )
            faces = (
                (ground, TYPE_COLORS[type_id]),
                (left, darken(TYPE_COLORS[type_id], 0.58)),
                (right, darken(TYPE_COLORS[type_id], 0.76)),
                (
                    top,
                    _roof_color(
                        TYPE_COLORS[type_id],
                        int(env.roof_quality_grid[row, col]),
                    ),
                ),
            )
            for face_index, (polygon, color) in enumerate(faces):
                if polygon:
                    shifted = [(int(x0 + x), int(y)) for x, y in polygon]
                    pygame.draw.polygon(self.screen, color, shifted)
                    if face_index > 0:
                        pygame.draw.polygon(self.screen, (55, 55, 55), shifted, 1)
            if type_id == 0:
                if env.player_road_grid[row, col]:
                    pygame.draw.polygon(self.screen, ACTIVE_ROAD, shifted_ground)
                    pygame.draw.polygon(
                        self.screen, (205, 211, 218), shifted_ground, 1
                    )
                elif env.pending_road_grid[row, col] > 0:
                    pygame.draw.polygon(
                        self.screen, PENDING_ROAD, shifted_ground, 2
                    )
            if type_id > 0:
                owner = int(env.owner_grid[row, col])
                if 0 <= owner < self.n_agents:
                    center = cell_center(
                        row,
                        col,
                        self.tile_width,
                        self.tile_height,
                        origin_x + x0,
                        origin_y,
                        elevation,
                    )
                    pygame.draw.circle(
                        self.screen,
                        AGENT_COLORS[owner % len(AGENT_COLORS)],
                        (int(center[0]), int(center[1])),
                        max(2, self.cell_size // 8),
                    )

            cell_id = row * self.w + col
            for agent_id, (cell, round_at, _) in self.latest_action[side].items():
                if cell is not None and cell == cell_id and round_at == self.round_idx:
                    pygame.draw.polygon(
                        self.screen,
                        AGENT_COLORS[agent_id % len(AGENT_COLORS)],
                        [(int(x0 + x), int(y)) for x, y in top],
                        2,
                    )
                    break
            if self.selected[side] == cell_id:
                pygame.draw.polygon(
                    self.screen,
                    (255, 255, 255),
                    [(int(x0 + x), int(y)) for x, y in top],
                    2,
                )
        label = self.font_header.render(
            f"{side.upper()}: {self.labels[side]}", True, (240, 240, 240)
        )
        self.screen.blit(label, (x0 + 4, self.header_h - 18))

    def _draw_side_panel(self) -> None:
        px = 2 * self.grid_w + 16
        y = 12
        title = self.font_header.render(
            f"Round {self.round_idx + 1}/{self.envs['a'].config.actions_per_agent}",
            True,
            (240, 240, 240),
        )
        self.screen.blit(title, (px, y))
        y += 26
        for side in ("a", "b"):
            env = self.envs[side]
            metrics = env.city_metrics()
            profit = _mean([a.profit for a in env.agents])
            lines = [
                f"{self.labels[side][:26]}",
                f"built {metrics.built_cells:>2}  debt {metrics.mean_debt_ratio:>4.2f}x"
                f"  profit {profit:>+7.1f}",
                f"quality {metrics.city_quality:.3f}"
                f"  sat {metrics.mean_satisfaction:.3f}",
                f"util {metrics.mean_utilization:.3f}"
                f"  damage {metrics.externality_damage_total:>6.1f}",
            ]
            for index, line in enumerate(lines):
                color = ACCENT if index == 0 else TEXT
                if index == 0:
                    pygame.draw.circle(self.screen, (76, 120, 168) if side == "a" else (245, 133, 24), (px + 5, y + 7), 5)
                    self.screen.blit(self.font_small.render(line, True, color), (px + 16, y))
                else:
                    self.screen.blit(self.font_small.render(line, True, color), (px + 4, y))
                y += 20
            y += 6

        # selected cell info for the most recently clicked side
        selected_side = "a" if self.selected["a"] is not None else ("b" if self.selected["b"] is not None else None)
        if selected_side is not None:
            cell = self.selected[selected_side]
            if cell is not None:
                row, col = divmod(int(cell), self.w)
                env = self.envs[selected_side]
                type_id = int(env.type_grid[row, col])
                owner = int(env.owner_grid[row, col])
                is_road = bool(
                    env.player_road_grid[row, col]
                    or env.pending_road_grid[row, col] > 0
                )
                if is_road:
                    owner = int(env.road_owner_grid[row, col])
                lines = [
                    f"{selected_side.upper()} cell {cell} ({row},{col})",
                    f"{'road' if is_road else TYPE_NAMES[type_id]} owner {owner if owner >= 0 else '-'}",
                    f"status {'pending' if env.pending_road_grid[row, col] > 0 else 'active' if env.player_road_grid[row, col] else '-'}",
                    f"land {env.land_value[row, col]:.2f}"
                    f"  pollution {env.pollution[row, col]:.2f}",
                    f"road access {env.road_access[row, col]:.2f}"
                    f"  density {env.player_road_density[row, col]:.2f}",
                ]
                for line in lines:
                    self.screen.blit(self.font_tiny.render(line, True, ACCENT), (px + 4, y))
                    y += 18
                y += 4

        chart_x = px
        chart_w = self.panel_width - 18
        chart_h = 100
        bottom = self.height - 10
        start_y = bottom - chart_h * 3 - 8
        if start_y < y + 2:
            start_y = y + 2
        self._draw_chart(
            chart_x, start_y, chart_w, chart_h,
            "City quality",
            [self.quality["a"], self.quality["b"]],
            [(76, 120, 168), (245, 133, 24)],
            ["A", "B"], 0.0, 1.0,
        )
        self._draw_chart(
            chart_x, start_y + chart_h + 4, chart_w, chart_h,
            "Mean satisfaction",
            [self.satisfaction["a"], self.satisfaction["b"]],
            [(76, 120, 168), (245, 133, 24)],
            ["A", "B"], 0.0, 1.0,
        )
        self._draw_chart(
            chart_x, start_y + 2 * (chart_h + 4), chart_w, chart_h,
            "Mean profit",
            [self.profit_mean["a"], self.profit_mean["b"]],
            [(76, 120, 168), (245, 133, 24)],
            ["A", "B"],
        )

    def draw(self) -> None:
        self.screen.fill(GRID_BG)
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (2 * self.grid_w + 8, 0, self.panel_width, self.height),
        )
        self._draw_grid("a")
        self._draw_grid("b")
        pygame.draw.line(
            self.screen,
            (70, 70, 70),
            (self.grid_w + 3, self.header_h),
            (self.grid_w + 3, self.header_h + self.grid_h),
        )
        self._draw_side_panel()
        pygame.display.flip()

    def _on_click(self, pos) -> None:
        x, y = pos
        if y < self.header_h:
            return
        if self.projection == "isometric":
            if x < self.grid_w:
                side = "a"
            elif self.grid_w + 8 <= x < 2 * self.grid_w + 8:
                side = "b"
            else:
                return
            x0 = self._grid_origin(side)
            cell = hit_test(
                (x - x0, y),
                self.envs[side].type_grid,
                self.envs[side].quality_grid,
                self.tile_width,
                self.tile_height,
                self.h * self.tile_width / 2.0,
                self.header_h + self.max_building_height + self.tile_height / 2.0,
                self.envs[side].floors_grid,
            )
            if cell is not None:
                self.selected[side] = cell
                self.draw()
            return
        row = (y - self.header_h) // self.cell_size
        if not 0 <= row < self.h:
            return
        if x < self.grid_w:
            side, col = "a", x // self.cell_size
        elif self.grid_w + 8 <= x < 2 * self.grid_w + 8:
            side, col = "b", (x - self.grid_w - 8) // self.cell_size
        else:
            return
        if 0 <= col < self.w:
            self.selected[side] = int(row * self.w + col)
            self.draw()

    # -- simulation hooks -------------------------------------------------

    def start_round(self, round_idx: int) -> None:
        self.round_idx = int(round_idx)

    def handle_action(self, side: str, agent_id: int, action, success: bool) -> None:
        if not self.running:
            return
        self.latest_action[side][agent_id] = (action.cell, self.round_idx, success)
        if len(self.action_log) > 60:
            self.action_log = self.action_log[-50:]
        self.draw()
        self.tick()

    def end_round(self, round_idx: int) -> None:
        self.round_idx = int(round_idx)
        for side in ("a", "b"):
            env = self.envs[side]
            metrics = env.city_metrics()
            self.quality[side].append(float(metrics.city_quality))
            self.satisfaction[side].append(float(metrics.mean_satisfaction))
            self.profit_mean[side].append(
                _mean([a.profit for a in env.agents])
            )
            self.built[side].append(float(metrics.built_cells))
        self.draw()
        for _ in range(4):
            self.tick()
            if not self.running:
                break
