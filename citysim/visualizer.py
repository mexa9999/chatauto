"""Real-time pygame visualizer for the city simulation."""

from __future__ import annotations

try:
    import pygame
except ImportError:
    pygame = None

import numpy as np

from .config import TYPE_NAMES


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


class CityVisualizer:
    """Pygame window that renders the city in real time."""

    def __init__(self, env, cell_size: int = 36, speed_ms: int = 80):
        if pygame is None:
            raise RuntimeError(
                "pygame is required for --live mode.\n"
                "Install with: pip install pygame"
            )

        pygame.init()
        self.env = env
        self.h = env.h
        self.w = env.w
        self.n_agents = len(env.agents)

        # Adaptive cell size
        panel_width = 340
        max_grid_w = 1100
        self.cell_size = min(cell_size, max(12, (max_grid_w - panel_width) // max(self.h, self.w)))
        self.panel_width = panel_width
        self.grid_w = self.w * self.cell_size
        self.grid_h = self.h * self.cell_size
        self.header_h = 44
        self.width = self.grid_w + self.panel_width
        self.height = max(self.grid_h + self.header_h + 20, 600)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CitySim — live simulation")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("consolas", 12)
        self.font = pygame.font.SysFont("consolas", 13)
        self.font_header = pygame.font.SysFont("consolas", 15, bold=True)

        self.speed_ms = speed_ms
        self.paused = False
        self.running = True

        self.latest_action: dict[int, tuple[int | None, int, bool]] = {}
        self.round_idx = 0
        self.agent_order = []

        self.action_log: list[str] = []
        self.max_log = 12

        self.agent_builds = [0] * self.n_agents
        self.agent_skips = [0] * self.n_agents
        self.agent_profits = [0.0] * self.n_agents
        self.agent_damage = [0.0] * self.n_agents

    # ------------------------------------------------------------------

    def _draw_grid(self):
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

                # Agent ownership marker (small circle in top-right corner)
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

                # Flash the most recent action
                cell_id = y * self.w + x
                for agent_id, (cell, round_at, ok) in self.latest_action.items():
                    if cell is not None and cell == cell_id and round_at == self.round_idx:
                        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
                        pygame.draw.rect(self.screen, color, rect, 3)
                        break

    def _draw_header(self):
        built = int((self.env.type_grid > 0).sum())
        total = self.h * self.w
        status = "PAUSED" if self.paused else "RUNNING"
        title = self.font_header.render(
            f"CitySim  |  Round {self.round_idx + 1}/{self.env.config.actions_per_agent}"
            f"  |  Built: {built}/{total}"
            f"  |  {status}  |  ↑/↓ speed, Space pause, Esc quit",
            True,
            (240, 240, 240),
        )
        self.screen.blit(title, (10, 12))

    def _draw_panel(self):
        px = self.grid_w + 12
        y = self.header_h + 4

        header = self.font_header.render("Agents", True, (220, 220, 220))
        self.screen.blit(header, (px, y))
        y += 22

        for i in range(self.n_agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            pygame.draw.circle(self.screen, color, (px + 6, y + 7), 5)
            text = (
                f"A{i:02d}  B{self.agent_builds[i]:>2}  "
                f"S{self.agent_skips[i]:>2}  "
                f"P{self.agent_profits[i]:>+6.0f}  "
                f"D{self.agent_damage[i]:>5.0f}"
            )
            label = self.font_small.render(text, True, (200, 200, 200))
            self.screen.blit(label, (px + 18, y))
            y += 18

        y += 12
        header = self.font_header.render("Recent actions", True, (220, 220, 220))
        self.screen.blit(header, (px, y))
        y += 22

        for line in self.action_log[-self.max_log:]:
            label = self.font_small.render(line, True, (200, 200, 200))
            self.screen.blit(label, (px, y))
            y += 16

        # Legend at the bottom of the panel
        legend_y = self.height - 15 * len(TYPE_NAMES) - 20
        legend = self.font_header.render("Legend", True, (220, 220, 220))
        self.screen.blit(legend, (px, legend_y - 22))
        for tid, name in enumerate(TYPE_NAMES):
            pygame.draw.rect(self.screen, TYPE_COLORS[tid], (px, legend_y + tid * 15, 12, 12))
            pygame.draw.rect(self.screen, (60, 60, 60), (px, legend_y + tid * 15, 12, 12), 1)
            label = self.font_small.render(f"  {name}", True, (200, 200, 200))
            self.screen.blit(label, (px + 14, legend_y + tid * 15 - 1))

    def draw(self):
        self.screen.fill((28, 28, 28))
        self._draw_header()
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()

    # ------------------------------------------------------------------

    def start_round(self, round_idx: int):
        self.round_idx = round_idx

    def handle_action(self, agent_id: int, action, success: bool):
        """Register an action, update stats, redraw."""
        if success and action.cell is not None:
            self.agent_builds[agent_id] += 1
            type_name = TYPE_NAMES[action.type_id + 1]
            self.action_log.append(
                f"R{self.round_idx + 1} A{agent_id:02d}  {type_name[:4]:>4} q{action.quality} @{action.cell}"
            )
        else:
            self.agent_skips[agent_id] += 1
            self.action_log.append(f"R{self.round_idx + 1} A{agent_id:02d}  SKIP")
        if len(self.action_log) > 60:
            self.action_log = self.action_log[-40:]

        self.latest_action[agent_id] = (action.cell, self.round_idx, success)
        self._update_stats()
        self.draw()
        self.tick()

    def _update_stats(self):
        for agent in self.env.agents:
            self.agent_profits[agent.agent_id] = agent.profit
            self.agent_damage[agent.agent_id] = agent.damage_caused

    def end_round_pause(self):
        """Small pause at round boundary so users see the round transition."""
        self._update_stats()
        self.draw()
        for _ in range(6):
            self.tick()
            if not self.running:
                break

    # ------------------------------------------------------------------

    def tick(self):
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

    def _handle_event(self, event):
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

    def close(self):
        pygame.quit()