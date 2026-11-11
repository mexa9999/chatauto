"""Command line interface: python -m citysim ..."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import os
import uuid
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from .config import (
    REGULATION_SCENARIOS,
    REQUIREMENT_PROFILES,
    CityConfig,
    EvolutionConfig,
    quick_city_config,
    quick_evolution_config,
)
from .analytics import AnalyticsRecorder, write_training_analytics
from .episode import run_episode, run_round, score_breakdown, score_episode
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
    e12_factorial_actions,
    train_evolution,
)
from .experiment_support import training_cache
from .io import ensure_dir, write_csv, write_json
from .plotting import plot_city, plot_compare_grids
from .progress import TrainingProgress, TrainingStopped
from .policy import (
    GenomeLayout,
    GreedyPolicy,
    ManualPolicy,
    NeuralPolicy,
    RandomPolicy,
    random_genome,
)
from .world import CityEnvironment
from .training_history import DEFAULT_HISTORY_DIR, TrainingHistoryStore


def parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.replace(" ", ",").split(",") if part]


def parse_grid(text: str) -> tuple[int, int]:
    normalized = text.translate(str.maketrans({"х": "x", "Х": "x", "×": "x"}))
    parts = normalized.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must look like 8x8")
    try:
        height, width = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "grid dimensions must be positive integers, e.g. 8x8"
        ) from exc
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError(
            "grid dimensions must be positive integers, e.g. 8x8"
        )
    if height < 3 or width < 3 or height > 64 or width > 64:
        raise argparse.ArgumentTypeError(
            "grid dimensions must be between 3 and 64, e.g. 12x12"
        )
    return height, width


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="maximum CPU worker processes (default: all available CPUs)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="e.g. 0-9 or 0,1,2")
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--grid", type=str, default=None, help="HxW, e.g. 8x8")
    parser.add_argument("--actions", type=int, default=None)
    parser.add_argument(
        "--actions-per-round",
        type=int,
        choices=range(1, 5),
        default=None,
        help="sequential actions per agent in each round (1-4)",
    )
    parser.add_argument(
        "--buildings-per-action",
        type=int,
        default=None,
        help="how many buildings an agent may place per round",
    )
    parser.add_argument("--bids", type=int, default=None, help="sealed bids per agent")
    parser.add_argument("--no-auction", action="store_true", help="sequential instead of auction")
    parser.add_argument("--hidden", type=int, default=None, help="neural hidden layer size")
    parser.add_argument("--attention-dim", type=int, default=None)
    parser.add_argument("--district-hidden", type=int, default=None)
    parser.add_argument("--global-hidden", type=int, default=None)
    parser.add_argument("--mutation-sigma", type=float, default=None)
    parser.add_argument("--sigma-decay", type=float, default=None)
    parser.add_argument("--tournament-size", type=int, default=None)
    parser.add_argument("--elites", type=int, default=None)
    parser.add_argument("--train-seed-base", type=int, default=None)
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
    parser.add_argument(
        "--projection",
        choices=("isometric", "top-down"),
        default="isometric",
        help="map projection for PNG and live visualizations",
    )


def build_configs(
    args,
    base_config: tuple[CityConfig, EvolutionConfig] | None = None,
) -> tuple[CityConfig, EvolutionConfig, list[int]]:
    if base_config is not None and not args.quick:
        city, evo = base_config
        default_seeds = [0, 1]
        eval_seeds = 3
    elif args.quick:
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
    if getattr(args, "actions_per_round", None) is not None:
        city = replace(city, actions_per_round=args.actions_per_round)
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
    if args.workers is not None:
        evo = replace(evo, max_workers=args.workers)
    if getattr(args, "hidden", None) is not None:
        evo = replace(evo, hidden_size=args.hidden)
    for argument, field_name in (
        ("attention_dim", "attention_dim"),
        ("district_hidden", "district_hidden"),
        ("global_hidden", "global_hidden"),
        ("mutation_sigma", "mutation_sigma"),
        ("sigma_decay", "sigma_decay"),
        ("tournament_size", "tournament_size"),
        ("elites", "elites"),
        ("train_seed_base", "train_seed_base"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            evo = replace(evo, **{field_name: value})
    if getattr(args, "map_elites", False):
        evo = replace(evo, use_map_elites=True)
    if getattr(args, "seed", None) is not None and args.seeds is None:
        evo = replace(evo, seed=args.seed)
    evo = replace(evo, verbose=not args.quiet)
    if eval_seeds is not None and args.eval_seeds is None:
        args.eval_seeds = eval_seeds
    seeds = parse_seeds(args.seeds) if args.seeds else default_seeds
    return city, evo, seeds


def _layout(evo: EvolutionConfig) -> GenomeLayout:
    return layout_from_config(evo)


def _load_genome(path: str | None, layout: GenomeLayout, seed: int) -> np.ndarray:
    if path:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "best_ever_genome" not in loaded.files:
                    raise ValueError("checkpoint has no best_ever_genome array")
                genome = loaded["best_ever_genome"].copy()
            finally:
                loaded.close()
        else:
            genome = loaded
        try:
            return layout.normalize(genome)
        except ValueError as exc:
            raise SystemExit(
                f"genome {path} has shape {genome.shape}, "
                f"expected ({layout.parameter_count},); old genome layouts "
                "are incompatible with the expanded action space"
            ) from exc
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
    layout = _layout(evo)
    out = ensure_dir(args.out or "runs/demo")
    analytics = AnalyticsRecorder(out, "demo", args.seed)
    policy_name = args.policy
    factory = _policy_factory(policy_name, layout, args.genome, args.seed)
    policies = [factory() for _ in range(city.n_agents)]
    recorder = None
    state_file = getattr(args, "live_state_file", None)
    if state_file:
        from .live_demo import LiveDemoRecorder

        control_file = getattr(args, "control_file", None)
        if not control_file:
            raise SystemExit("--live-state-file requires --control-file")
        recorder = LiveDemoRecorder(
            state_file,
            control_file,
            {
                "kind": "metadata",
                "policy": policy_name,
                "seed": args.seed,
                "projection": getattr(args, "projection", "isometric"),
                "config": city,
            },
        )

    viz = None
    snapshots: list[str] = []
    should_continue = lambda: (  # noqa: E731
        recorder.wait_until_active()
        if recorder is not None
        else viz is None or viz.running
    )

    def on_round_start(round_idx: int, env: CityEnvironment) -> None:
        nonlocal viz
        if getattr(args, "live", False) and viz is None:
            from .visualizer import CityVisualizer

            viz = CityVisualizer(
                env,
                cell_size=getattr(args, "cell_size", 36),
                speed_ms=getattr(args, "speed", 80),
                projection=getattr(args, "projection", "isometric"),
            )
        if viz is not None:
            viz.start_round(round_idx)
        if recorder is not None:
            recorder.record_state(env, "round_start", round_idx)

    def on_action(agent_id: int, action, success: bool, env: CityEnvironment):
        analytics.record_action(agent_id, action, success, env)
        if viz is not None:
            viz.handle_action(agent_id, action, success)
        if recorder is not None:
            return recorder.after_action(agent_id, action, success, env)
        return None

    def on_round_end(round_idx: int, env: CityEnvironment) -> None:
        analytics.record_round(round_idx, env)
        if viz is not None:
            viz.end_round_pause()
        if recorder is not None:
            recorder.record_state(env, "round_end", round_idx)
        else:
            snapshots.append(
                str(
                    plot_city(
                        env,
                        out / f"round_{round_idx:02d}.png",
                        title=f"{policy_name}, round {round_idx + 1}",
                        projection=getattr(args, "projection", "isometric"),
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

    try:
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
    finally:
        analytics.close()
        if viz is not None:
            viz.close()
        if recorder is not None:
            recorder.close()

    scores = score_episode(result)
    analytics_summary = analytics.finish(
        result, scores, score_breakdown(result)
    )
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
            "state_stream": "frames.jsonl" if state_file else None,
            "shocks": result.shocks,
            "config": city,
            "analytics": "analytics_summary.json",
        },
    )
    if recorder is None:
        plot_city(
            type(
                "EnvProxy",
                (),
                {
                    "type_grid": result.type_grid,
                    "player_road_grid": result.player_road_grid,
                    "pending_road_grid": result.pending_road_grid,
                    "w": city.width,
                    "h": city.height,
                },
            )(),
            out / "final.png",
            title=f"{policy_name}, final city",
            projection=getattr(args, "projection", "isometric"),
        )
    print(f"demo saved to {out}")
    print(
        f"built={result.city.built_cells} quality={result.city.city_quality:.3f} "
        f"satisfaction={result.city.mean_satisfaction:.3f} "
        f"utilization={result.city.mean_utilization:.3f} "
        f"profit_mean={np.mean([a.profit for a in result.agents]):.1f} "
        f"debt_total={result.city.total_debt:.1f} "
        f"damage_total={result.city.externality_damage_total:.2f}"
    )
    if snapshots:
        print(f"last snapshot: {snapshots[-1]}")


def command_compare(args) -> None:
    city, evo, _ = build_configs(args)
    layout = _layout(evo)
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
            projection=getattr(args, "projection", "isometric"),
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
        projection=getattr(args, "projection", "isometric"),
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
    out = ensure_dir(args.out or "runs/train")
    history_dir = args.history_dir or DEFAULT_HISTORY_DIR
    history = TrainingHistoryStore(history_dir)
    branch = args.branch or history.active_branch()
    base_config = history.config_for_branch(branch) if args.resume else None
    city, evo, _ = build_configs(args, base_config=base_config)
    layout = _layout(evo)
    training_metadata = {
        "branch": branch,
        "seed": evo.seed,
        "evolution_config": evo.as_dict(),
        "city_config": city.as_dict(),
    }
    initial_history = []
    if args.resume:
        initial_history = read_checkpoint_metadata(
            history.checkpoint_path(branch)
        ).get("history", [])
    write_training_analytics(out, initial_history, training_metadata)
    (out / "training_progress.ndjson").write_text("", encoding="utf-8")
    progress = (
        TrainingProgress(
            args.progress_file,
            args.stop_file,
            analytics_directory=out,
        )
        if args.progress_file and args.stop_file
        else None
    )
    result = train_evolution(
        evo,
        city,
        out_dir=out,
        tag=args.tag,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        history_store=history,
        branch=branch,
        run_id=getattr(args, "run_id", None) or uuid.uuid4().hex,
        progress_callback=progress.generation_completed if progress else None,
        stop_requested=progress.stop_requested if progress else None,
    )
    write_training_analytics(out, result.history, training_metadata)
    evaluation_seed = evo.validation_seed_base + 777
    evaluation_policies = [
        NeuralPolicy(result.best_genome, layout)
        for _ in range(city.n_agents)
    ]
    evaluation = AnalyticsRecorder(
        out / "final_evaluation", "training-final-evaluation", evaluation_seed
    )
    try:
        evaluation_result = run_episode(
            city,
            evaluation_policies,
            evaluation_seed,
            record_history=True,
            on_action=evaluation.record_action,
            on_round_end=evaluation.record_round,
        )
    finally:
        evaluation.close()
    evaluation_scores = score_episode(evaluation_result)
    evaluation.finish(
        evaluation_result,
        evaluation_scores,
        score_breakdown(evaluation_result),
    )
    training_progress_path = out / "training_progress.ndjson"
    if training_progress_path.is_file():
        training_progress_path.unlink()
    if progress is not None:
        from .checkpoint_io import read_checkpoint_metadata

        checkpoint_metadata = read_checkpoint_metadata(history.checkpoint_path(branch))
        progress.finish(
            "completed" if checkpoint_metadata["complete"] else "stopped"
        )
    print(f"best fitness: {result.best_fitness:.4f} at generation {result.best_generation}")
    if result.archive:
        print(f"MAP-Elites archive: {len(result.archive)} niches")
    print(f"artifacts: {out}")


def command_experiment(args) -> None:
    city, evo, seeds = build_configs(args)
    out = ensure_dir(Path(args.out or f"runs/{args.name}"))
    cache_dir = args.cache_dir or DEFAULT_HISTORY_DIR
    progress = (
        TrainingProgress(args.progress_file, args.stop_file)
        if args.progress_file and args.stop_file
        else None
    )
    try:
        with training_cache(
            cache_dir,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            progress_callback=progress.generation_completed if progress else None,
            stop_requested=progress.stop_requested if progress else None,
        ):
            _run_experiment(args, city, evo, seeds, out)
            series = []
            for history_path in sorted(out.glob("*_history.csv")):
                with history_path.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        numeric = {}
                        for key, value in row.items():
                            if value is None:
                                continue
                            try:
                                numeric[key] = float(value)
                            except ValueError:
                                numeric[key] = value
                        numeric["series"] = history_path.stem
                        series.append(numeric)
            if series:
                write_training_analytics(
                    out,
                    series,
                    {
                        "experiment": args.name,
                        "seeds": seeds,
                        "evolution_config": evo.as_dict(),
                        "city_config": city.as_dict(),
                    },
                )
            if progress is not None:
                progress.finish("completed")
    except TrainingStopped:
        print("training stopped after saving a generation checkpoint")
        if progress is not None:
            progress.finish("stopped")


def _run_experiment(args, city, evo, seeds, out: Path) -> None:
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
    elif name == "e12":
        summary = e12_factorial_actions(
            evo, city, seeds, out, eval_seed_count=args.eval_seeds
        )
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
        summaries["e12"] = e12_factorial_actions(
            evo, city, seeds, out / "e12", args.eval_seeds
        )
        write_json(out / "all_summary.json", summaries)
        summary = {"experiments": list(summaries)}
    else:
        raise SystemExit(f"unknown experiment {name}")
    write_json(out / "last_summary.json", summary)
    print(f"summary: {out / 'last_summary.json'}")


def command_history(args) -> None:
    history = TrainingHistoryStore(args.history_dir)
    if args.history_action == "branches":
        for branch in history.branch_names():
            generation = branch["generation"]
            head = branch["commit_id"]
            suffix = (
                "no snapshots"
                if head is None
                else f"generation {generation}, {head[:12]}"
            )
            active = "*" if branch["name"] == history.active_branch() else " "
            print(f"{active} {branch['name']}: {suffix}")
        return
    if args.history_action == "log":
        branch = args.branch or history.active_branch()
        try:
            commits = history.history(branch)
            head = history.resolve_ref(branch)
        except ValueError as exc:
            if str(exc) == f"branch has no snapshots yet: {branch}":
                print(f"history for {branch}: no snapshots")
                return
            raise
        print(f"history for {branch}:")
        for commit in commits:
            pointer = "->" if commit["commit_id"] == head else "  "
            best = commit["best_fitness"]
            fitness = "n/a" if best is None else f"{best:.4f}"
            print(
                f"{pointer} gen {commit['generation']:>5} "
                f"{commit['commit_id'][:12]} "
                f"best={fitness} branch={commit['branch_name']}"
            )
        return
    if args.history_action == "branch-create":
        parent = history.create_branch(args.name, args.from_ref)
        print(f"created branch {args.name} from {parent[:12] if parent else 'empty'}")
        return
    if args.history_action == "checkout":
        history.checkout(args.name)
        print(f"active branch: {args.name}")
        return
    if args.history_action == "reset":
        head = history.reset_branch(args.name, args.to_ref)
        print(f"moved {args.name} to {head[:12]}")


def command_dashboard(args) -> None:
    from .dashboard import run_dashboard

    run_dashboard(
        history_dir=args.history_dir,
        output_dir=args.out,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def command_dashboard_compare_worker(args) -> None:
    from .dashboard_worker import run_compare

    run_compare(
        args.job_file,
        args.result_file,
        args.progress_file,
        args.stop_file,
    )


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
    demo.add_argument("--live-state-file", type=str, default=None)
    demo.add_argument("--control-file", type=str, default=None)
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
    train.add_argument(
        "--history-dir",
        "--cache-dir",
        dest="history_dir",
        default=str(DEFAULT_HISTORY_DIR),
    )
    train.add_argument("--branch", type=str, default=None)
    train.add_argument("--run-id", type=str, default=None)
    train.add_argument(
        "--resume",
        action="store_true",
        help="resume incomplete generation checkpoints from the training cache",
    )
    train.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="atomically save progress every N generations (default: 10)",
    )
    train.add_argument("--progress-file", type=str, default=None)
    train.add_argument("--stop-file", type=str, default=None)
    train.set_defaults(func=command_train)

    exp = sub.add_parser("experiment", help="run experiments E1-E12")
    add_common(exp)
    exp.add_argument(
        "--cache-dir",
        "--history-dir",
        dest="cache_dir",
        type=str,
        default=str(DEFAULT_HISTORY_DIR),
        help="shared SQLite history and experiment checkpoint store",
    )
    exp.add_argument(
        "--resume",
        action="store_true",
        help="resume incomplete training checkpoints and reuse completed trainings",
    )
    exp.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="atomically save progress every N generations (default: 10)",
    )
    exp.add_argument("--progress-file", type=str, default=None)
    exp.add_argument("--stop-file", type=str, default=None)
    exp.add_argument(
        "name",
        choices=[f"e{i}" for i in range(1, 13)] + ["all"],
    )
    exp.add_argument("--n-values", type=str, default="5,10,15")
    exp.add_argument("--sweep-grid", type=str, default="12x12")
    exp.add_argument("--paths", type=int, default=10, help="E9 alternative histories")
    exp.set_defaults(func=command_experiment)

    dashboard = sub.add_parser(
        "dashboard", help="open the local training and history dashboard"
    )
    dashboard.add_argument(
        "--history-dir", type=str, default=str(DEFAULT_HISTORY_DIR)
    )
    dashboard.add_argument("--out", type=str, default="runs/train")
    dashboard.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--no-browser", action="store_true")
    dashboard.set_defaults(func=command_dashboard)

    compare_worker = sub.add_parser(
        "_dashboard-compare-worker",
        help=argparse.SUPPRESS,
    )
    compare_worker.add_argument("job_file")
    compare_worker.add_argument("result_file")
    compare_worker.add_argument("progress_file")
    compare_worker.add_argument("stop_file")
    compare_worker.set_defaults(func=command_dashboard_compare_worker)

    history = sub.add_parser("history", help="inspect and manage training branches")
    history_commands = history.add_subparsers(
        dest="history_action",
        required=True,
    )
    history_log = history_commands.add_parser("log", help="show snapshot ancestry")
    history_log.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    history_log.add_argument("--branch", type=str, default=None)
    history_log.set_defaults(func=command_history)

    history_branches = history_commands.add_parser(
        "branches",
        help="list branches and their heads",
    )
    history_branches.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    history_branches.set_defaults(func=command_history)

    history_create = history_commands.add_parser(
        "branch-create",
        help="create a branch from a branch head or branch@generation",
    )
    history_create.add_argument("name")
    history_create.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    history_create.add_argument("--from", dest="from_ref", default=None)
    history_create.set_defaults(func=command_history)

    history_checkout = history_commands.add_parser(
        "checkout",
        help="set the active branch for subsequent training",
    )
    history_checkout.add_argument("name")
    history_checkout.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    history_checkout.set_defaults(func=command_history)

    history_reset = history_commands.add_parser(
        "reset",
        help="move a branch pointer without deleting snapshots",
    )
    history_reset.add_argument("name")
    history_reset.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    history_reset.add_argument("--to", dest="to_ref", required=True)
    history_reset.set_defaults(func=command_history)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    finally:
        # Close any executor that survived the experiment runner.
        # wait=False avoids the multi-second join on Windows spawn.
        from . import experiments
        if getattr(experiments, "_GLOBAL_EXECUTOR", None) is not None:
            experiments._GLOBAL_EXECUTOR.shutdown(wait=False, cancel_futures=True)
            experiments._GLOBAL_EXECUTOR = None


if __name__ == "__main__":
    main()
