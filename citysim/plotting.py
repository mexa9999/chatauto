"""Small matplotlib helpers.  All plots are written to disk (Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for file output
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (kept for API compat)
import numpy as np

from .config import TYPE_NAMES
from .world import CityEnvironment

TYPE_COLORS = ["#f2f2f2", "#4c78a8", "#f58518", "#9d755d", "#b6423c", "#54a24b"]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_city(env: CityEnvironment, path: str | Path, title: str = "City") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = matplotlib.colors.ListedColormap(TYPE_COLORS)
    ax.imshow(env.type_grid, cmap=cmap, vmin=-0.5, vmax=5.5)
    ax.set_xticks(range(env.w))
    ax.set_yticks(range(env.h))
    ax.set_xticks(np.arange(-0.5, env.w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    ax.set_title(title)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(-0.5, 5.5)),
        ax=ax,
        ticks=range(6),
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(TYPE_NAMES)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_compare_grids(
    env_a: CityEnvironment,
    env_b: CityEnvironment,
    path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap = matplotlib.colors.ListedColormap(TYPE_COLORS)
    for ax, env, label in ((axes[0], env_a, label_a), (axes[1], env_b, label_b)):
        ax.imshow(env.type_grid, cmap=cmap, vmin=-0.5, vmax=5.5)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _history_arrays(histories: Sequence[Sequence[dict]], key: str):
    max_len = max(len(h) for h in histories)
    values = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    gens = np.arange(max_len)
    for i, hist in enumerate(histories):
        for row in hist:
            if key in row:
                values[i, int(row["gen"])] = float(row[key])
    return gens, values


def _mean_ci(values: np.ndarray):
    mean = np.nanmean(values, axis=0)
    if values.shape[0] > 1:
        std = np.nanstd(values, axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(values.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, ci


def plot_learning_curve(
    histories: Sequence[Sequence[dict]],
    path: str | Path,
    title: str = "Neuroevolution learning curve",
) -> Path:
    """Plot best/mean fitness and behavioural diversity with 95% CI bands."""
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label, color in [
        ("best", "best fitness", "#1f77b4"),
        ("mean", "mean fitness", "#2ca02c"),
    ]:
        gens, values = _history_arrays(histories, key)
        mean, ci = _mean_ci(values)
        axes[0].plot(gens, mean, label=label, color=color)
        axes[0].fill_between(gens, mean - ci, mean + ci, color=color, alpha=0.16)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Fitness")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    gens, values = _history_arrays(histories, "diversity")
    mean, ci = _mean_ci(values)
    axes[1].plot(gens, mean, color="#d62728")
    axes[1].fill_between(gens, mean - ci, mean + ci, color="#d62728", alpha=0.16)
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("mean pairwise behaviour distance")
    axes[1].set_title("Behavioural diversity")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_satisfaction_curve(
    histories: Sequence[Sequence[dict]],
    path: str | Path,
    title: str = "Citizen satisfaction",
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    candidates = [
        (("mean_satisfaction", "eval_satisfaction"), "mean", "#2ca02c"),
        (("rich_satisfaction", "eval_rich_satisfaction"), "rich", "#1f77b4"),
        (("poor_satisfaction", "eval_poor_satisfaction"), "poor", "#d62728"),
    ]
    plotted = False
    for keys, label, color in candidates:
        gens, values = _history_arrays(histories, keys[0])
        if np.all(np.isnan(values)):
            gens, values = _history_arrays(histories, keys[1])
        if np.all(np.isnan(values)):
            continue
        mean, ci = _mean_ci(values)
        ax.plot(gens, mean, label=label, color=color)
        ax.fill_between(gens, mean - ci, mean + ci, color=color, alpha=0.14)
        plotted = True
    ax.set_xlabel("generation / round")
    ax.set_ylabel("satisfaction")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_history_panels(history: Sequence[dict], path: str | Path, title: str = "City history") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rounds = [int(row["round"]) + 1 for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("city_quality", "city quality", "#4c78a8"),
        ("mean_satisfaction", "mean satisfaction", "#54a24b"),
        ("mean_utilization", "mean utilisation", "#f58518"),
        ("damage_total", "cumulative externality damage", "#b6423c"),
    ]
    for ax, (key, label, color) in zip(axes.ravel(), panels):
        values = [float(row.get(key, 0.0)) for row in history]
        ax.plot(rounds, values, marker="o", color=color)
        ax.set_xlabel("round")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_vs_x(
    x: Sequence[float],
    series: dict[str, tuple[Sequence[float], Sequence[float] | None]],
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    """Line/scatter plot with optional per-point 95% CI error bars."""
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (values, errors) in series.items():
        values = np.asarray(values, dtype=np.float64)
        ax.plot(x, values, marker="o", label=label)
        if errors is not None:
            ax.errorbar(x, values, yerr=np.asarray(errors, dtype=np.float64), fmt="none", capsize=3, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float] | None,
    path: str | Path,
    ylabel: str,
    title: str,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4, color="#4c78a8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profit_quality(
    rows: Iterable[dict], path: str | Path, title: str = "Profit vs city quality"
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for row in rows:
        ax.scatter(
            row["profit_mean"],
            row["city_quality_mean"],
            s=70,
            label=row["condition"],
        )
        ax.annotate(
            row["condition"],
            (row["profit_mean"], row["city_quality_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("mean profit per agent")
    ax.set_ylabel("city quality")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profit_satisfaction(
    rows: Sequence[dict], path: str | Path, title: str = "Profit vs satisfaction"
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        ax.scatter(row["profit_mean"], row["mean_satisfaction_mean"], s=80)
        ax.annotate(
            row["condition"],
            (row["profit_mean"], row["mean_satisfaction_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("mean profit per agent")
    ax.set_ylabel("mean citizen satisfaction")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distribution_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: str | Path,
    title: str = "Path dependence",
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.asarray(matrix), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="type share")
    ax.set_title(title)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
