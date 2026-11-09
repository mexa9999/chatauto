"""Small matplotlib helpers.  All plots are written to disk (Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _history_arrays(histories: Sequence[Sequence[dict]], key: str):
    max_len = max(len(h) for h in histories)
    values = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    gens = np.arange(max_len)
    for i, hist in enumerate(histories):
        for row in hist:
            values[i, int(row["gen"])] = float(row[key])
    return gens, values


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
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(mean)
        ci = 1.96 * std / np.sqrt(max(1, len(histories)))
        axes[0].plot(gens, mean, label=label, color=color)
        axes[0].fill_between(gens, mean - ci, mean + ci, color=color, alpha=0.16)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Fitness")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    gens, values = _history_arrays(histories, "diversity")
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(mean)
    ci = 1.96 * std / np.sqrt(max(1, len(histories)))
    axes[1].plot(gens, mean, color="#d62728")
    axes[1].fill_between(gens, mean - ci, mean + ci, color="#d62728", alpha=0.16)
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("mean pairwise behaviour distance")
    axes[1].set_title("Behavioural diversity")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
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
    fig.tight_layout()
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
    fig.tight_layout()
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
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
