from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from citysim.config import EvolutionConfig, quick_city_config
from citysim.dashboard import (
    BranchRequest,
    CheckoutRequest,
    GCRequest,
    JobManager,
    StartJobRequest,
    _pid_is_running,
    create_app,
)
from citysim.dashboard_worker import run_compare
from citysim.experiment_support import train_evolution
from citysim.live_demo import update_control_file
from citysim.training_history import TrainingHistoryStore


class TestDashboard(unittest.TestCase):
    def test_pid_probe_detects_exited_process_while_handle_is_open(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=10)

        self.assertFalse(_pid_is_running(process.pid))

    def _create_history(self, path: str, **architecture: int) -> TrainingHistoryStore:
        store = TrainingHistoryStore(path)
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            episodes_per_generation=2,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=False,
            seed=81,
            **architecture,
        )
        train_evolution(config, city, history_store=store, checkpoint_every=1)
        return store

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
                episodes=4,
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
