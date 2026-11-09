"""Command line interface: python -m citysim ..."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import (
    REGULATION_SCENARIOS,
    REQUIREMENT_PROFILES,
    CityConfig,
    EvolutionConfig,
    quick_city_config,
    quick_evolution_config,
)
from .episode import run_episode, run_round, score_episode
from .evolution import layout_from_config
from .experiments import (
    e1_learning_curve,
    e2_baselines,
    e3_externalities,
    e4_requirements,
    e5_agents,
    e6_generalization,
    e7_diversity,
    e8_districts,
    e9_path_dependence,
    e10_regulation,
    e11_shocks,
    train_evolution,
)
from .io import ensure_dir, write_csv, write_json
from .plotting import plot_city, plot_compare_grids
from .policy import (
    GenomeLayout,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    random_genome,
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
    parser.add_argument(
        "--buildings-per-action",
        type=int,
        default=None,
        help="how many buildings an agent may place per round",
    )
    parser.add_argument("--bids", type=int, default=None, help="sealed bids per agent")
    parser.add_argument("--no-auction", action="store_true", help="sequential instead of auction")
    parser.add_argument("--hidden", type=int, default=None, help="neural hidden layer size")
    parser.add_argument(
        "--requirements",
        choices=tuple(REQUIREMENT_PROFILES),
        default=None,
        help="planning requirement profile",
    )
    parser.add_argument(
        "--regulation",
        choices=tuple(REGULATION_SCENARIOS),
        default=None,
        help="tax/subsidy scenario",
    )
    parser.add_argument("--no-externalities", action="store_true")
    parser.add_argument("--map-elites", action="store_true", help="keep a MAP-Elites archive")
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
    if args.buildings_per_action is not None:
        city = replace(city, buildings_per_action=args.buildings_per_action)
    if args.bids is not None:
        city = replace(city, bids_per_agent=args.bids)
    if getattr(args, "no_auction", False):
        city = replace(city, use_auction=False)
    if getattr(args, "requirements", None) is not None:
        city = replace(city, requirements=REQUIREMENT_PROFILES[args.requirements])
    if getattr(args, "regulation", None) is not None:
        city = replace(city, regulation=REGULATION_SCENARIOS[args.regulation])
    if getattr(args, "no_externalities", False):
        city = replace(city, enable_externalities=False)
    if args.generations is not None:
        evo = replace(evo, generations=args.generations)
    if args.episodes is not None:
        evo = replace(evo, episodes_per_generation=args.episodes)
    if args.population is not None:
        evo = replace(evo, population_size=args.population)
    if getattr(args, "hidden", None) is not None:
        evo = replace(evo, hidden_size=args.hidden)
    if getattr(args, "map_elites", False):
        evo = replace(evo, use_map_elites=True)
    if args.seed is not None and args.seeds is None:
        evo = replace(evo, seed=args.seed)
    evo = replace(evo, verbose=not args.quiet)
    if eval_seeds is not None and args.eval_seeds is None:
        args.eval_seeds = eval_seeds
    seeds = parse_seeds(args.seeds) if args.seeds else default_seeds
    return city, evo, seeds


def _layout(evo: EvolutionConfig, args=None) -> GenomeLayout:
    if args is not None and getattr(args, "hidden", None) is not None:
        return GenomeLayout(hidden_size=int(args.hidden))
    return layout_from_config(evo)


def _load_genome(path: str | None, layout: GenomeLayout, seed: int) -> np.ndarray:
    if path:
        genome = np.load(path)
        if genome.shape != (layout.parameter_count,):
            raise SystemExit(
                f"genome {path} has shape {genome.shape}, "
                f"expected ({layout.parameter_count},)"
            )
        return genome
    print(
        "no --genome supplied: using a random genome "
        f"({layout.parameter_count} parameters)"
    )
    return random_genome(np.random.default_rng(seed + 12_345), layout)


def _policy_factory(
    name: str,
    layout: GenomeLayout | None,
    genome_path: str | None,
    seed: int,
):
    if name == "neural":
        if layout is None:
            raise SystemExit("neural policy needs a genome layout")
        genome = _load_genome(genome_path, layout, seed)

        def factory():
            return NeuralPolicy(genome, layout)

        return factory
    if name == "manual":
        return lambda: ManualPolicy()
    if name == "greedy":
        return lambda: GreedyPolicy()
    return lambda: RandomPolicy()


def command_demo(args) -> None:
    city, evo, _ = build_configs(args)
    layout = _layout(evo, args)
    out = ensure_dir(args.out or "runs/demo")
    policy_name = args.policy
    factory = _policy_factory(policy_name, layout, args.genome, args.seed)
    policies = [factory() for _ in range(city.n_agents)]

    viz = None
    snapshots: list[str] = []
    should_continue = lambda: viz is None or viz.running  # noqa: E731

    def on_round_start(round_idx: int, env: CityEnvironment) -> None:
        nonlocal viz
        if getattr(args, "live", False) and viz is None:
            from .visualizer import CityVisualizer

            viz = CityVisualizer(
                env,
                cell_size=getattr(args, "cell_size", 36),
                speed_ms=getattr(args, "speed", 80),
            )
        if viz is not None:
            viz.start_round(round_idx)

    def on_action(agent_id: int, action, success: bool, env: CityEnvironment) -> None:
        if viz is not None:
            viz.handle_action(agent_id, action, success)

    def on_round_end(round_idx: int, env: CityEnvironment) -> None:
        if viz is not None:
            viz.end_round_pause()
        snapshots.append(
            str(
                plot_city(
                    env,
                    out / f"round_{round_idx:02d}.png",
                    title=f"{policy_name}, round {round_idx + 1}",
                )
            )
        )

    shocks = []
    if getattr(args, "shock", None):
        shock_round = max(1, city.actions_per_agent // 2)
        magnitude = float(getattr(args, "shock_magnitude", 0.5))
        if args.shock == "population_boom" and magnitude <= 1.0:
            magnitude = 200.0
        shocks = [
            {"round": shock_round, "kind": args.shock, "magnitude": magnitude}
        ]

    result = run_episode(
        city,
        policies,
        args.seed,
        record_history=True,
        shocks=shocks,
        on_action=on_action,
        on_round_start=on_round_start,
        on_round_end=on_round_end,
        should_continue=should_continue,
    )

    if viz is not None:
        viz.close()

    scores = score_episode(result)
    write_csv(out / "history.csv", result.history)
    write_json(
        out / "metrics.json",
        {
            "policy": policy_name,
            "seed": args.seed,
            "city": result.city,
            "agents": result.agents,
            "scores": scores,
            "snapshots": snapshots,
            "shocks": result.shocks,
            "config": city,
        },
    )
    final = plot_city(
        type("EnvProxy", (), {"type_grid": result.type_grid, "w": city.width, "h": city.height})(),
        out / "final.png",
        title=f"{policy_name}, final city",
    )
    print(f"demo saved to {out}")
    print(
        f"built={result.city.built_cells} quality={result.city.city_quality:.3f} "
        f"satisfaction={result.city.mean_satisfaction:.3f} "
        f"utilization={result.city.mean_utilization:.3f} "
        f"profit_mean={np.mean([a.profit for a in result.agents]):.1f} "
        f"damage_total={result.city.externality_damage_total:.2f}"
    )
    if snapshots:
        print(f"last snapshot: {snapshots[-1]}")


def command_compare(args) -> None:
    city, evo, _ = build_configs(args)
    layout = _layout(evo, args)
    out = ensure_dir(args.out or "runs/compare")
    factory_a = _policy_factory(args.policy_a, layout, args.genome_a, args.seed + 1)
    factory_b = _policy_factory(args.policy_b, layout, args.genome_b, args.seed + 2)
    policies_a = [factory_a() for _ in range(city.n_agents)]
    policies_b = [factory_b() for _ in range(city.n_agents)]

    env_a = CityEnvironment(city, args.seed, record_history=True)
    env_b = CityEnvironment(city, args.seed, record_history=True)
    rng_a = np.random.default_rng(args.seed + 991)
    rng_b = np.random.default_rng(args.seed + 991)

    viz = None
    if getattr(args, "live", False):
        from .visualizer import ComparisonVisualizer

        viz = ComparisonVisualizer(
            env_a,
            env_b,
            args.policy_a,
            args.policy_b,
            cell_size=getattr(args, "cell_size", 30),
            speed_ms=getattr(args, "speed", 60),
        )

    def callback(side: str):
        def inner(agent_id: int, action, success: bool, env: CityEnvironment) -> None:
            if viz is not None:
                viz.handle_action(side, agent_id, action, success)

        return inner

    for round_idx in range(city.actions_per_agent):
        if viz is not None:
            if not viz.running:
                break
            viz.start_round(round_idx)
        run_round(env_a, policies_a, rng_a, city, callback("a"))
        env_a.end_round()
        run_round(env_b, policies_b, rng_b, city, callback("b"))
        env_b.end_round()
        if viz is not None:
            viz.end_round(round_idx)

    if viz is not None:
        viz.close()

    city_a = env_a.city_metrics()
    city_b = env_b.city_metrics()
    agents_a = env_a.agent_metrics()
    agents_b = env_b.agent_metrics()
    scores_a = score_episode(
        type("R", (), {"agents": agents_a, "city": city_a, "config": city})()
    )
    scores_b = score_episode(
        type("R", (), {"agents": agents_b, "city": city_b, "config": city})()
    )
    rows = [
        {
            "side": "A",
            "policy": args.policy_a,
            "built": city_a.built_cells,
            "city_quality": city_a.city_quality,
            "mean_satisfaction": city_a.mean_satisfaction,
            "profit_mean": float(np.mean([a.profit for a in agents_a])),
            "damage_total": city_a.externality_damage_total,
            "bankrupt_agents": city_a.bankrupt_agents,
            "fitness_mean": float(np.mean(scores_a)),
        },
        {
            "side": "B",
            "policy": args.policy_b,
            "built": city_b.built_cells,
            "city_quality": city_b.city_quality,
            "mean_satisfaction": city_b.mean_satisfaction,
            "profit_mean": float(np.mean([a.profit for a in agents_b])),
            "damage_total": city_b.externality_damage_total,
            "bankrupt_agents": city_b.bankrupt_agents,
            "fitness_mean": float(np.mean(scores_b)),
        },
    ]
    write_csv(out / "compare_summary.csv", rows)
    write_csv(out / "history_a.csv", env_a.history)
    write_csv(out / "history_b.csv", env_b.history)
    write_json(
        out / "compare_summary.json",
        {
            "seed": args.seed,
            "policy_a": args.policy_a,
            "policy_b": args.policy_b,
            "config": city,
            "city_a": city_a,
            "city_b": city_b,
            "rows": rows,
        },
    )
    plot_compare_grids(
        env_a,
        env_b,
        out / "compare_grids.png",
        label_a=f"A: {args.policy_a}",
        label_b=f"B: {args.policy_b}",
    )
    for row in rows:
        print(
            f"{row['policy']:>10}: built={row['built']} "
            f"quality={row['city_quality']:.3f} "
            f"satisfaction={row['mean_satisfaction']:.3f} "
            f"profit={row['profit_mean']:.1f} "
            f"fitness={row['fitness_mean']:.3f}"
        )
    print(f"comparison saved to {out}")


def command_train(args) -> None:
    city, evo, _ = build_configs(args)
    out = ensure_dir(args.out or "runs/train")
    result = train_evolution(evo, city, out_dir=out, tag=args.tag)
    print(f"best fitness: {result.best_fitness:.4f} at generation {result.best_generation}")
    if result.archive:
        print(f"MAP-Elites archive: {len(result.archive)} niches")
    print(f"artifacts: {out}")


def command_experiment(args) -> None:
    city, evo, seeds = build_configs(args)
    out = ensure_dir(Path(args.out or f"runs/{args.name}"))
    if not args.quiet:
        print(
            f"experiment {args.name}: seeds={seeds} generations={evo.generations} "
            f"episodes={evo.episodes_per_generation} population={evo.population_size}"
        )
    name = args.name
    if name == "e1":
        summary = e1_learning_curve(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e2":
        summary = e2_baselines(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e3":
        summary = e3_externalities(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e4":
        summary = e4_requirements(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e5":
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
    elif name == "e6":
        summary = e6_generalization(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e7":
        summary = e7_diversity(evo, city, seeds, out)
    elif name == "e8":
        summary = e8_districts(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e9":
        summary = e9_path_dependence(evo, city, seeds, out, paths=args.paths)
    elif name == "e10":
        summary = e10_regulation(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "e11":
        summary = e11_shocks(evo, city, seeds, out, eval_seed_count=args.eval_seeds)
    elif name == "all":
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
        summaries["e8"] = e8_districts(evo, city, seeds, out / "e8", args.eval_seeds)
        summaries["e9"] = e9_path_dependence(evo, city, seeds, out / "e9", args.paths)
        summaries["e10"] = e10_regulation(evo, city, seeds, out / "e10", args.eval_seeds)
        summaries["e11"] = e11_shocks(evo, city, seeds, out / "e11", args.eval_seeds)
        write_json(out / "all_summary.json", summaries)
        summary = {"experiments": list(summaries)}
    else:
        raise SystemExit(f"unknown experiment {name}")
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
    demo.add_argument(
        "--policy", choices=["random", "greedy", "manual", "neural"], default="manual"
    )
    demo.add_argument("--genome", type=str, default=None)
    demo.add_argument(
        "--shock",
        choices=["industry_removal", "population_boom", "earthquake"],
        default=None,
    )
    demo.add_argument("--shock-magnitude", type=float, default=0.5)
    demo.set_defaults(func=command_demo)

    compare = sub.add_parser("compare", help="run two policies on the same seed")
    add_common(compare)
    compare.add_argument(
        "--policy-a", choices=["random", "greedy", "manual", "neural"], default="greedy"
    )
    compare.add_argument(
        "--policy-b", choices=["random", "greedy", "manual", "neural"], default="manual"
    )
    compare.add_argument("--genome-a", type=str, default=None)
    compare.add_argument("--genome-b", type=str, default=None)
    compare.set_defaults(func=command_compare)

    train = sub.add_parser("train", help="run one neuroevolution training")
    add_common(train)
    train.add_argument("--tag", type=str, default="train")
    train.set_defaults(func=command_train)

    exp = sub.add_parser("experiment", help="run experiments E1-E11")
    add_common(exp)
    exp.add_argument(
        "name",
        choices=[f"e{i}" for i in range(1, 12)] + ["all"],
    )
    exp.add_argument("--n-values", type=str, default="5,10,15")
    exp.add_argument("--sweep-grid", type=str, default="12x12")
    exp.add_argument("--paths", type=int, default=10, help="E9 alternative histories")
    exp.set_defaults(func=command_experiment)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
