"""Command line interface: python -m citysim ..."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import (
    CityConfig,
    EvolutionConfig,
    quick_city_config,
    quick_evolution_config,
)
from .episode import run_episode, score_episode
from .experiments import (
    e1_learning_curve,
    e2_baselines,
    e3_externalities,
    e4_requirements,
    e5_agents,
    e6_generalization,
    e7_diversity,
    train_evolution,
)
from .io import ensure_dir, write_csv, write_json
from .plotting import plot_city
from .policy import (
    GenomeLayout,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
)
from .world import CityEnvironment


def parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.replace(" ", ",").split(",") if part]


def parse_grid(text: str) -> tuple[int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must look like 8x8")
    return int(parts[0]), int(parts[1])


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None, help="e.g. 0-9 or 0,1,2")
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--grid", type=str, default=None, help="HxW, e.g. 8x8")
    parser.add_argument("--actions", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--live", action="store_true", help="pygame real-time window")
    parser.add_argument("--speed", type=int, default=80, help="ms between actions")
    parser.add_argument("--cell-size", type=int, default=36, help="pixel size of each grid cell")


def build_configs(args) -> tuple[CityConfig, EvolutionConfig, list[int]]:
    if args.quick:
        city = quick_city_config()
        evo = quick_evolution_config()
        default_seeds = [0, 1]
        eval_seeds = 3
    else:
        city = CityConfig()
        evo = EvolutionConfig()
        default_seeds = list(range(10))
        eval_seeds = 5

    if args.n_agents is not None:
        city = replace(city, n_agents=args.n_agents)
    if args.grid is not None:
        h, w = parse_grid(args.grid)
        city = replace(city, height=h, width=w)
    if args.actions is not None:
        city = replace(city, actions_per_agent=args.actions)
    if args.generations is not None:
        evo = replace(evo, generations=args.generations)
    if args.episodes is not None:
        evo = replace(evo, episodes_per_generation=args.episodes)
    if args.population is not None:
        evo = replace(evo, population_size=args.population)
    if args.seed is not None and args.seeds is None:
        evo = replace(evo, seed=args.seed)
    if eval_seeds is not None and args.eval_seeds is None:
        args.eval_seeds = eval_seeds
    seeds = parse_seeds(args.seeds) if args.seeds else default_seeds
    return city, evo, seeds


def _layout(hidden: int) -> GenomeLayout:
    return GenomeLayout(hidden_size=hidden)


def command_demo(args) -> None:
    city, _, _ = build_configs(args)
    out = ensure_dir(args.out or "runs/demo")
    policy_name = args.policy
    layout = _layout(args.hidden)
    if policy_name == "neural":
        if not args.genome:
            raise SystemExit("--genome is required for --policy neural")
        genome = np.load(args.genome)
        factories = [lambda: NeuralPolicy(genome, layout) for _ in range(city.n_agents)]
    elif policy_name == "manual":
        factories = [lambda: ManualPolicy() for _ in range(city.n_agents)]
    elif policy_name == "greedy":
        factories = [lambda: GreedyPolicy() for _ in range(city.n_agents)]
    else:
        factories = [lambda: RandomPolicy() for _ in range(city.n_agents)]

    env = CityEnvironment(city, args.seed, record_history=True)
    rng = np.random.default_rng(args.seed + 991)
    policies = [factory() for factory in factories]

    viz = None
    if getattr(args, "live", False):
        from .visualizer import CityVisualizer
        viz = CityVisualizer(
            env,
            cell_size=getattr(args, "cell_size", 36),
            speed_ms=getattr(args, "speed", 80),
        )

    snapshots = []
    for round_idx in range(city.actions_per_agent):
        order = rng.permutation(city.n_agents)
        if viz:
            viz.start_round(round_idx)
        for agent_id in order:
            if viz and not viz.running:
                break
            action = policies[int(agent_id)].act(env, int(agent_id), rng)
            success = env.apply_action(int(agent_id), action)
            if viz:
                viz.handle_action(int(agent_id), action, success)
        if viz and not viz.running:
            break
        env.end_round()
        if viz:
            viz.end_round_pause()
        snapshot = plot_city(
            env,
            out / f"round_{round_idx:02d}.png",
            title=f"{policy_name}, round {round_idx + 1}",
        )
        snapshots.append(str(snapshot))

    if viz:
        viz.close()

    city_metrics = env.city_metrics()
    agents = env.agent_metrics()
    synthetic_result = type(
        "DemoResult",
        (),
        {
            "seed": args.seed,
            "city": city_metrics,
            "agents": agents,
            "history": env.history,
            "actions": env.actions,
            "config": city,
        },
    )()
    scores = score_episode(synthetic_result)
    write_csv(out / "history.csv", env.history)
    write_json(
        out / "metrics.json",
        {
            "policy": policy_name,
            "seed": args.seed,
            "city": city_metrics,
            "agents": agents,
            "scores": scores,
            "snapshots": snapshots,
            "config": city,
        },
    )
    final = plot_city(env, out / "final.png", title=f"{policy_name}, final city")
    print(f"demo saved to {out}")
    print(
        f"built={city_metrics.built_cells} quality={city_metrics.city_quality:.3f} "
        f"profit_mean={np.mean([a.profit for a in agents]):.1f} "
        f"damage_total={city_metrics.externality_damage_total:.2f}"
    )
    print(f"final grid: {final}")


def command_train(args) -> None:
    city, evo, _ = build_configs(args)
    evo = replace(evo, verbose=not args.quiet)
    out = ensure_dir(args.out or "runs/train")
    result = train_evolution(evo, city, out_dir=out, tag="train")
    print(f"best fitness: {result.best_fitness:.4f} at generation {result.best_generation}")
    print(f"artifacts: {out}")


def command_experiment(args) -> None:
    city, evo, seeds = build_configs(args)
    out = Path(args.out or f"runs/{args.name}")
    out = ensure_dir(out)
    if not args.quiet:
        print(
            f"experiment {args.name}: seeds={seeds} generations={evo.generations} "
            f"episodes={evo.episodes_per_generation} population={evo.population_size}"
        )
    if args.name == "e1":
        summary = e1_learning_curve(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
    elif args.name == "e2":
        summary = e2_baselines(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
    elif args.name == "e3":
        summary = e3_externalities(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
    elif args.name == "e4":
        summary = e4_requirements(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
    elif args.name == "e5":
        n_values = [int(x) for x in args.n_values.split(",")]
        grid = parse_grid(args.sweep_grid)
        summary = e5_agents(
            evo,
            city,
            seeds,
            out,
            n_values=n_values,
            grid=grid,
            eval_seed_count=args.eval_seeds,
        )
    elif args.name == "e6":
        summary = e6_generalization(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
    elif args.name == "e7":
        summary = e7_diversity(evo, city, seeds, out)
    elif args.name == "all":
        summaries = {}
        summaries["e1"] = e1_learning_curve(evo, city, seeds, out / "e1", args.eval_seeds)
        summaries["e2"] = e2_baselines(evo, city, seeds, out / "e2", args.eval_seeds)
        summaries["e3"] = e3_externalities(evo, city, seeds, out / "e3", args.eval_seeds)
        summaries["e4"] = e4_requirements(evo, city, seeds, out / "e4", args.eval_seeds)
        summaries["e5"] = e5_agents(
            evo,
            city,
            seeds,
            out / "e5",
            n_values=[int(x) for x in args.n_values.split(",")],
            grid=parse_grid(args.sweep_grid),
            eval_seed_count=args.eval_seeds,
        )
        summaries["e6"] = e6_generalization(evo, city, seeds, out / "e6", args.eval_seeds)
        summaries["e7"] = e7_diversity(evo, city, seeds, out / "e7")
        write_json(out / "all_summary.json", summaries)
        summary = {"experiments": list(summaries)}
    else:
        raise SystemExit(f"unknown experiment {args.name}")
    write_json(out / "last_summary.json", summary)
    print(f"summary: {out / 'last_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citysim",
        description="Neuroevolution of developer strategies in a multi-agent city.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run and render one episode")
    add_common(demo)
    demo.add_argument("--policy", choices=["random", "greedy", "manual", "neural"], default="manual")
    demo.add_argument("--genome", type=str, default=None)
    demo.add_argument("--hidden", type=int, default=64)
    demo.set_defaults(func=command_demo)

    train = sub.add_parser("train", help="run one neuroevolution training")
    add_common(train)
    train.set_defaults(func=command_train)

    exp = sub.add_parser("experiment", help="run E1-E7")
    add_common(exp)
    exp.add_argument(
        "name",
        choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "all"],
    )
    exp.add_argument("--n-values", type=str, default="5,10,15")
    exp.add_argument("--sweep-grid", type=str, default="12x12")
    exp.set_defaults(func=command_experiment)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
