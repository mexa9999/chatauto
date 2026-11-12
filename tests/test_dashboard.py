from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from starlette.requests import Request
from citysim.config import EvolutionConfig, quick_city_config
from citysim.dashboard import (
    BranchRequest,
    BranchRenameRequest,
    CheckoutRequest,
    FastForwardRequest,
    GCRequest,
    GenomeCloneRequest,
    GenomeCompareRequest,
    GenomeMetadataRequest,
    JobManager,
    StorageSelectionRequest,
    StartJobRequest,
    _pid_is_running,
    create_app,
)
from citysim.dashboard_worker import run_compare
from citysim.experiment_store import ExperimentStore
from citysim.experiment_support import train_evolution
from citysim.live_demo import update_control_file
from citysim.strategy_catalog import branch_graph, branch_training_runs, run_checkpoints
from citysim.training_history import TrainingHistoryStore


async def _collect_response(response) -> bytes:
    return b"".join([chunk async for chunk in response.body_iterator])


class TestDashboard(unittest.TestCase):
    def test_pid_probe_detects_exited_process_while_handle_is_open(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=10)

        self.assertFalse(_pid_is_running(process.pid))

    def _create_history(
        self, path: str, *, run_id: str | None = None, **architecture: int
    ) -> TrainingHistoryStore:
        store = TrainingHistoryStore(path)
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            league_fraction=1.0,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=81,
            **architecture,
        )
        train_evolution(
            config,
            city,
            history_store=store,
            checkpoint_every=1,
            run_id=run_id,
        )
        return store

    def test_empty_branch_and_run_scoped_strategy_tree(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            store = self._create_history(output, run_id="a" * 32)
            store.create_branch("empty", empty=True)

            runs = branch_training_runs(store, "main")
            checkpoints = run_checkpoints(store, "main", runs[0]["key"])
            groups = store.checkpoint_genome_groups(checkpoints[-1]["commit_id"])
            genomes = store.list_checkpoint_genomes(
                checkpoints[-1]["commit_id"],
                groups[0]["member_kind"],
                limit=2,
            )
            graph = branch_graph(store)
            app = create_app(store.root, Path(output) / "dashboard-output")
            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }

            self.assertEqual(store.branch_names()[0]["name"], "empty")
            self.assertIsNone(next(item for item in store.branch_names() if item["name"] == "empty")["head_commit"])
            self.assertEqual(runs[0]["run_id"], "a" * 32)
            self.assertEqual(len(checkpoints), 2)
            self.assertGreater(groups[0]["count"], 0)
            self.assertLessEqual(len(genomes), 2)
            self.assertEqual(len(graph["branches"]), 2)
            self.assertEqual(len(graph["commits"]), 2)
            self.assertEqual(
                routes[("/api/branches/{name}/strategy-runs", "GET")]("main")["runs"][0]["run_id"],
                "a" * 32,
            )
            self.assertEqual(
                len(routes[("/api/history/graph", "GET")]()["commits"]),
                2,
            )
            with store._connect() as connection:
                connection.execute(
                    "UPDATE commits SET run_id = NULL WHERE commit_id = ?",
                    (checkpoints[0]["commit_id"],),
                )
                connection.execute(
                    "UPDATE commits SET run_id = 'legacy' WHERE commit_id = ?",
                    (checkpoints[1]["commit_id"],),
                )
            legacy_runs = branch_training_runs(store, "main")
            self.assertEqual({run["key"] for run in legacy_runs}, {"legacy", "run:legacy"})

    def test_start_job_rejects_out_of_range_grid_and_action_count(self) -> None:
        with self.assertRaises(ValueError):
            StartJobRequest(grid="65x12")
        with self.assertRaises(ValueError):
            StartJobRequest(actions_per_round=5)

    def test_start_job_defaults_match_evolution_config(self) -> None:
        request = StartJobRequest()
        self.assertEqual(request.generations, 125)
        self.assertEqual(request.episodes, 16)

    def test_training_job_requires_focal_agents_to_cover_population(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manager = JobManager(root / "history", root / "out")
            job_dir = root / "job"
            with self.assertRaises(HTTPException) as raised:
                manager._command(
                    StartJobRequest(
                        kind="train",
                        population=32,
                        n_agents=2,
                        episodes=16,
                    ).model_dump(),
                    job_dir,
                    job_dir / "progress.json",
                    job_dir / "stop.request",
                )
            self.assertEqual(raised.exception.status_code, 422)
            self.assertIn("At least 32 episodes", raised.exception.detail)

            command = manager._command(
                StartJobRequest(
                    kind="train",
                    population=32,
                    n_agents=2,
                    episodes=32,
                ).model_dump(),
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )
            self.assertEqual(command[command.index("--episodes") + 1], "32")

    def test_verify_gc_preview_and_apply_preserve_reachable_history(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            store = self._create_history(output)
            discarded_commit = store.history("main")[-1]["commit_id"]
            store.create_branch("keep", "main@1")
            store.reset_branch("main", "main@1")

            preview = store.gc_preview()
            self.assertIn(discarded_commit, preview["commit_ids"])
            with self.assertRaisesRegex(ValueError, "changed after the GC preview"):
                store.gc_apply("stale")

            result = store.gc_apply(preview["confirmation_token"])
            self.assertIn(discarded_commit, result["deleted_commits"])
            self.assertTrue(store.verify()["ok"])
            self.assertEqual([row["generation"] for row in store.history("keep")], [1])

    def test_verify_detects_checkpoint_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            store = self._create_history(output)
            checkpoint = store.checkpoint_path("main")
            with checkpoint.open("ab") as stream:
                stream.write(b"corruption")

            report = store.verify()

            self.assertFalse(report["ok"])
            self.assertTrue(
                any("hash mismatch" in issue for issue in report["issues"])
            )

    def test_compare_worker_evaluates_both_methods_on_shared_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            store = self._create_history(str(Path(output) / "history"))
            job_path = Path(output) / "job.json"
            result_path = Path(output) / "result.json"
            progress_path = Path(output) / "progress.json"
            job_path.write_text(
                json.dumps(
                    {
                        "history_dir": str(store.root),
                        "left_ref": "main@1",
                        "right_ref": "main@2",
                        "reference_ref": "main@1",
                        "seed_base": 1234,
                        "seed_count": 1,
                    }
                ),
                encoding="utf-8",
            )

            run_compare(
                job_path,
                result_path,
                progress_path,
                Path(output) / "stop.request",
            )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["seeds"], [1234])
            self.assertEqual(result["reports"]["left"]["ref"], "main@1")
            self.assertEqual(
                set(result["reports"]["left"]["methods"]),
                {"homogeneous", "league"},
            )
            self.assertIn("fitness", result["metric_differences"]["league"])
            self.assertIn("debt_ratio", result["metric_differences"]["league"])
            self.assertIn(
                "local_goods_coverage", result["metric_differences"]["league"]
            )
            self.assertIn(
                "tax_rate_multiplier", result["metric_differences"]["league"]
            )
            self.assertIn("evolution", result["config_differences"])
            for report in result["reports"].values():
                for method in report["methods"].values():
                    self.assertEqual(method["per_seed"][0]["seed"], 1234)

    def test_api_exposes_history_verify_and_safe_gc_preview(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            history = str(Path(output) / "history")
            app = create_app(history, str(Path(output) / "out"), port=18765)
            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }
            self.assertTrue(
                Path(app.routes[0].endpoint().path).is_file()
            )
            self.assertEqual(routes[("/api/branches", "GET")]()["active"], "main")
            self.assertTrue(routes[("/api/verify", "GET")]()["ok"])
            self.assertEqual(routes[("/api/demo/checkpoints", "GET")]()["checkpoints"], [])
            preview = routes[("/api/gc/preview", "GET")]()
            self.assertEqual(preview["commit_ids"], [])
            routes[("/api/gc", "POST")](
                GCRequest(confirmation_token=preview["confirmation_token"])
            )
            self.assertEqual(
                routes[("/api/branches", "POST")](BranchRequest(name="trial"))["name"],
                "trial",
            )
            self.assertEqual(
                routes[("/api/checkout", "POST")](
                    CheckoutRequest(branch="trial")
                )["active"],
                "trial",
            )
            self.assertEqual(
                routes[("/api/data", "GET")]()["datasets"],
                [],
            )
            self.assertEqual(
                routes[("/api/data/history/default", "POST")]()[
                    "active_history_dir"
                ],
                str(Path(history).resolve()),
            )

    def test_storage_api_manages_branches_and_genomes_with_trash_and_npz_import(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            history = self._create_history(str(Path(output) / "history"))
            app = create_app(history.root, str(Path(output) / "out"), port=18766)
            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }

            routes[("/api/branches", "POST")](BranchRequest(name="feature", from_ref="main@1"))
            routes[("/api/branches/{name}/fast-forward", "POST")](
                "feature", FastForwardRequest(to_ref="main@2")
            )
            routes[("/api/branches/{name}", "PUT")](
                "feature", BranchRenameRequest(new_name="renamed")
            )
            routes[("/api/branches/{name}", "DELETE")]("renamed")
            trash = routes[("/api/data/trash", "GET")]()["items"]
            self.assertTrue(any(item["entity_type"] == "branch" for item in trash))
            routes[("/api/data/restore", "POST")](
                StorageSelectionRequest(entity_type="branch", ids=["renamed"])
            )

            genome_page = routes[("/api/genomes", "GET")](
                branch=None, query="", include_deleted=False, limit=1, offset=0
            )
            self.assertEqual(len(genome_page["genomes"]), 1)
            self.assertTrue(genome_page["has_more"])
            next_genome_page = routes[("/api/genomes", "GET")](
                branch=None, query="", include_deleted=False, limit=1, offset=1
            )
            self.assertEqual(len(next_genome_page["genomes"]), 1)
            self.assertNotEqual(
                genome_page["genomes"][0]["genome_id"],
                next_genome_page["genomes"][0]["genome_id"],
            )
            genomes = genome_page["genomes"]
            self.assertTrue(genomes)
            genome_id = genomes[0]["genome_id"]
            details = routes[("/api/genomes/{genome_id}", "GET")](genome_id)
            self.assertGreater(details["vector_summary"]["parameter_count"], 0)
            routes[("/api/genomes/{genome_id}", "PUT")](
                genome_id, GenomeMetadataRequest(updates={"name": "Reviewed genome"})
            )
            clone = routes[("/api/genomes/{genome_id}/clone", "POST")](
                genome_id, GenomeCloneRequest(name="Clone")
            )
            clone_id = clone["genome_id"]
            comparison = routes[("/api/genomes/compare", "POST")](
                GenomeCompareRequest(left_id=genome_id, right_id=clone_id)
            )
            self.assertEqual(comparison["max_absolute_difference"], 0)

            export_response = routes[("/api/genomes/{genome_id}/export", "GET")](genome_id)
            chunks = asyncio.run(_collect_response(export_response))

            async def receive():
                return {"type": "http.request", "body": chunks, "more_body": False}

            request = Request(
                {"type": "http", "method": "POST", "path": "/api/genomes/import", "headers": []},
                receive,
            )
            imported = asyncio.run(
                routes[("/api/genomes/import", "POST")](request, name="roundtrip")
            )
            self.assertEqual(imported["metadata"]["name"], "roundtrip")

            routes[("/api/data/trash", "POST")](
                StorageSelectionRequest(entity_type="genome", ids=[clone_id])
            )
            routes[("/api/data/restore", "POST")](
                StorageSelectionRequest(entity_type="genome", ids=[clone_id])
            )
            routes[("/api/data/trash", "POST")](
                StorageSelectionRequest(entity_type="genome", ids=[clone_id])
            )
            routes[("/api/data/purge", "POST")](
                StorageSelectionRequest(entity_type="genome", ids=[clone_id])
            )
            with self.assertRaises(HTTPException):
                routes[("/api/genomes/{genome_id}", "GET")](clone_id)

    def test_purge_all_data_clears_every_trash_entity_type(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            history = self._create_history(str(Path(output) / "history"))
            history.create_branch("discarded")
            history.delete_branch("discarded")
            genomes = history.list_genomes()
            history.delete_genomes([genomes[0]["genome_id"]])

            app = create_app(history.root, output, port=18767)
            store = ExperimentStore(
                Path(output) / ".dashboard" / "store", history.root
            )
            run_id = "b" * 32
            run_dir = Path(output) / ".dashboard" / "jobs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"fitness": 0.5}), encoding="utf-8"
            )
            store.register_run(run_id, "experiment", {}, run_dir)
            store.update_run(run_id, "completed")
            archive = Path(output) / "run.zip"
            store.create_export(run_id, "light", history.root, archive)
            preview = store.stage_import(archive, "trash-me")
            dataset = store.commit_import(preview["token"])
            store.trash_entities("run", [run_id])
            store.trash_entities("dataset", [dataset["dataset_id"]])

            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }
            result = routes[("/api/data/purge-all", "POST")]()

            self.assertEqual(result, {"status": "purged", "count": 4})
            self.assertEqual(routes[("/api/data/trash", "GET")]()["items"], [])
            self.assertEqual(store.list_runs(), [])
            self.assertEqual(store.list_datasets(), [])

    def test_demo_job_command_uses_isolated_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manager = JobManager(root / "history", root / "out")
            job_dir = root / "job"
            command = manager._command(
                StartJobRequest(
                    kind="demo",
                    quick=True,
                    policy="greedy",
                    grid="8x8",
                    seed=17,
                    shock="earthquake",
                ).model_dump(),
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )

            self.assertEqual(command[2:4], ["citysim", "demo"])
            self.assertIn(str(job_dir / "artifacts"), command)
            self.assertIn("--quick", command)
            self.assertIn("greedy", command)
            self.assertIn("earthquake", command)
            self.assertIn("--live-state-file", command)
            self.assertIn(str(job_dir / "artifacts" / "frames.jsonl"), command)
            self.assertIn("--control-file", command)
            self.assertNotIn("--stop-file", command)

    def test_round_count_is_passed_to_demo_training_and_experiment_commands(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manager = JobManager(root / "history", root / "out")

            for kind in ("demo", "train", "experiment"):
                job_dir = root / kind
                request = StartJobRequest(
                    kind=kind,
                    quick=True,
                    rounds=13,
                    policy="greedy",
                )
                command = manager._command(
                    request.model_dump(),
                    job_dir,
                    job_dir / "progress.json",
                    job_dir / "stop.request",
                )
                actions_index = command.index("--actions")
                self.assertEqual(command[actions_index + 1], "13", kind)
                self.assertIn("--grid", command)
                self.assertIn("--actions-per-round", command)
                if kind == "train":
                    self.assertEqual(command[command.index("--run-id") + 1], job_dir.name)

    def test_history_api_returns_empty_commits_for_empty_branch(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            app = create_app(
                str(Path(output) / "history"),
                str(Path(output) / "out"),
                port=18769,
            )
            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }

            self.assertEqual(
                routes[("/api/history", "GET")]("main"),
                {"branch": "main", "commits": []},
            )
            with self.assertRaises(HTTPException) as raised:
                routes[("/api/history", "GET")]("missing")
            self.assertEqual(raised.exception.status_code, 404)

    def test_job_history_reports_exited_worker_as_failed_when_progress_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manager = JobManager(root / "history", root / "out")
            job_id = "a" * 32
            job_dir = manager.jobs_dir / job_id
            job_dir.mkdir()
            (job_dir / "request.json").write_text(
                json.dumps({"kind": "train", "created_at": 1}),
                encoding="utf-8",
            )
            (job_dir / "progress.json").write_text(
                json.dumps({"state": "running", "generation": 5}),
                encoding="utf-8",
            )
            process = subprocess.Popen([sys.executable, "-c", "pass"])
            process.wait(timeout=10)
            (job_dir / "pid.txt").write_text(str(process.pid), encoding="ascii")

            jobs = manager.list_jobs()

            self.assertEqual(jobs[0]["state"], "failed")

    def test_neural_demo_resolves_selected_history_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            history = root / "history"
            self._create_history(str(history))
            manager = JobManager(history, root / "out")
            job_dir = root / "job"

            command = manager._command(
                StartJobRequest(
                    kind="demo",
                    policy="neural",
                    genome_ref="main@1",
                ).model_dump(),
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )

            genome_index = command.index("--genome")
            self.assertEqual(
                command[genome_index + 1],
                str(TrainingHistoryStore(history).checkpoint_path("main@1")),
            )
            self.assertEqual(command[genome_index + 2], "--hidden")

    def test_dashboard_runs_demo_from_selected_neural_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            history = root / "history"
            self._create_history(str(history))
            manager = JobManager(history, root / "out")
            job_dir = root / "job"
            job_dir.mkdir()
            update_control_file(job_dir / "control.json", speed_ms=0)
            command = manager._command(
                StartJobRequest(
                    kind="demo",
                    quick=True,
                    policy="neural",
                    genome_ref="main@1",
                    speed_ms=0,
                ).model_dump(),
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )

            subprocess.run(command, check=True, capture_output=True, text=True)

            metrics = json.loads(
                (job_dir / "artifacts" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["policy"], "neural")
            self.assertEqual(metrics["state_stream"], "frames.jsonl")
            self.assertTrue((job_dir / "artifacts" / "frames.jsonl").is_file())

    def test_dashboard_runs_neural_demo_from_custom_checkpoint_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            history = root / "history"
            self._create_history(
                str(history),
                attention_dim=4,
                district_hidden=5,
                global_hidden=3,
            )
            manager = JobManager(history, root / "out")
            job_dir = root / "job"

            command = manager._command(
                StartJobRequest(
                    kind="demo",
                    quick=True,
                    policy="neural",
                    genome_ref="main@1",
                    speed_ms=0,
                ).model_dump(),
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )
            for option, expected in (
                ("--attention-dim", "4"),
                ("--district-hidden", "5"),
                ("--global-hidden", "3"),
            ):
                index = command.index(option)
                self.assertEqual(command[index + 1], expected)

            subprocess.run(command, check=True, capture_output=True, text=True)
            metrics = json.loads(
                (job_dir / "artifacts" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["policy"], "neural")

    def test_dashboard_demo_streams_state_and_renders_png_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            app = create_app(str(root / "history"), str(root / "out"), port=18766)
            manager = app.state.job_manager
            job_id = manager.start(
                StartJobRequest(kind="demo", quick=True, policy="greedy", speed_ms=0)
            )
            deadline = time.monotonic() + 30
            while manager.status()["state"] in ("running", "stopping"):
                if time.monotonic() >= deadline:
                    self.fail("dashboard demo did not finish in time")
                time.sleep(0.05)

            job_dir = manager._job_dir(job_id)
            stream_path = job_dir / "artifacts" / "frames.jsonl"
            self.assertTrue(stream_path.is_file())
            self.assertFalse(list((job_dir / "artifacts").glob("*.png")))
            frames = manager.live_frames(job_id, 0)["frames"]
            renderable = [frame for frame in frames if "type_grid" in frame]
            self.assertTrue(renderable)
            self.assertTrue(all("owner_grid" in frame for frame in renderable))

            endpoints = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }
            response = endpoints[
                ("/api/jobs/{job_id}/frames/{sequence}/image", "GET")
            ](job_id, renderable[-1]["seq"], "isometric")
            png_path = Path(response.path)
            try:
                self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            finally:
                png_path.unlink(missing_ok=True)

    def test_frame_png_render_uses_saved_quality_grid(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            app = create_app(str(root / "history"), str(root / "out"), port=18767)
            manager = app.state.job_manager
            endpoints = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }
            frame = {
                "type_grid": [[0, 1], [2, 3]],
                "quality_grid": [[0, 2], [1, 0]],
                "width": 2,
                "height": 2,
                "round": 1,
            }

            def render_png(_env, destination, **_kwargs):
                Path(destination).write_bytes(b"\x89PNG\r\n\x1a\n")

            with (
                patch.object(manager, "live_frame", return_value=frame),
                patch("citysim.dashboard.plot_city", side_effect=render_png) as render,
            ):
                response = endpoints[
                    ("/api/jobs/{job_id}/frames/{sequence}/image", "GET")
                ]("a" * 32, 1, "isometric")

            try:
                np.testing.assert_array_equal(
                    render.call_args.args[0].quality_grid,
                    frame["quality_grid"],
                )
            finally:
                Path(response.path).unlink(missing_ok=True)

    def test_demo_control_endpoint_returns_saved_pause_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            app = create_app(str(root / "history"), str(root / "out"), port=18768)
            manager = app.state.job_manager
            job_id = manager.start(
                StartJobRequest(
                    kind="demo",
                    quick=False,
                    policy="greedy",
                    speed_ms=200,
                )
            )
            endpoints = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }
            try:
                manager.control_demo(job_id, paused=True, speed_ms=350)
                self.assertEqual(
                    endpoints[("/api/jobs/{job_id}/control", "GET")](job_id),
                    {"paused": True, "speed_ms": 350},
                )
            finally:
                manager.shutdown()

    def test_demo_can_pause_and_stop_while_the_process_is_live(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manager = JobManager(root / "history", root / "out")
            job_id = manager.start(
                StartJobRequest(
                    kind="demo",
                    quick=False,
                    policy="greedy",
                    speed_ms=200,
                )
            )
            try:
                deadline = time.monotonic() + 20
                batch = {"frames": []}
                while not any(frame.get("kind") == "action" for frame in batch["frames"]):
                    if time.monotonic() >= deadline:
                        self.fail("demo emitted no action states")
                    batch = manager.live_frames(job_id, 0)
                    time.sleep(0.05)

                manager.control_demo(job_id, paused=True, speed_ms=0)
                settled = manager.live_frames(job_id, batch["next_offset"])
                time.sleep(0.35)
                paused_batch = manager.live_frames(job_id, settled["next_offset"])
                self.assertEqual(paused_batch["frames"], [])

                manager.stop(job_id)
                while manager.status()["state"] in ("running", "stopping"):
                    if time.monotonic() >= deadline:
                        self.fail("paused demo did not stop")
                    time.sleep(0.05)
                self.assertEqual(manager.status()["state"], "stopped")
            finally:
                manager.shutdown()

    def test_dashboard_worker_stops_at_a_saved_generation(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            history = root / "history"
            manager = JobManager(history, root / "out")
            request = StartJobRequest(
                kind="train",
                quick=True,
                generations=100,
                episodes=8,
                population=8,
                n_agents=2,
                workers=1,
                checkpoint_every=10,
            )
            job_id = manager.start(request)
            restarted_manager = JobManager(history, root / "out")
            self.assertEqual(restarted_manager.status()["job_id"], job_id)
            restarted_manager.stop(job_id)
            deadline = time.monotonic() + 60
            while restarted_manager.status()["state"] in ("running", "stopping"):
                if time.monotonic() >= deadline:
                    self.fail("dashboard training worker did not stop in time")
                time.sleep(0.1)

            status = restarted_manager.status()
            self.assertEqual(
                status["state"],
                "stopped",
                manager.read_log(job_id, 0, 131_072)["text"],
            )
            commits = TrainingHistoryStore(history).history("main")
            self.assertEqual(commits[-1]["generation"], 1)
            self.assertFalse(commits[-1]["complete"])
            self.assertEqual(status["progress"]["generation"], 1)
            manager.status()
            restarted_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
