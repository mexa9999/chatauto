from __future__ import annotations

import logging
import os
from pathlib import Path
import pstats
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from citysim.cli import build_parser, main
from citysim.config import EvolutionConfig, quick_city_config
from citysim.evolution import layout_from_config
from citysim.execution import EpisodeExecutor, EpisodeTask
from citysim.logging_setup import (
    LOGGER_NAME,
    configure_logging,
    resolve_log_level,
)
from citysim.policy import random_genome
from citysim.profiling import (
    PROFILE_CHILD_ENV,
    ProfileSession,
    default_profile_path,
    profile_format_for,
    run_with_pyspy,
    strip_profile_args,
)


class TestLoggingSetup(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging("WARNING", force=True)

    def test_log_level_resolution_prefers_explicit_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_log_level(None), logging.INFO)
            self.assertEqual(
                resolve_log_level(None, quiet=True), logging.WARNING
            )
        with patch.dict(
            os.environ, {"CITYSIM_LOG_LEVEL": "debug"}, clear=True
        ):
            self.assertEqual(resolve_log_level(None), logging.DEBUG)
            self.assertEqual(resolve_log_level("warning"), logging.WARNING)
        self.assertEqual(resolve_log_level("ERROR"), logging.ERROR)
        with self.assertRaisesRegex(ValueError, "unknown log level"):
            resolve_log_level("verbose")

    def test_configure_logging_writes_file_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "run.log"
            configure_logging("DEBUG", path, force=True)
            logging.getLogger("citysim.test").debug("hello logging")
            for handler in logging.getLogger(LOGGER_NAME).handlers:
                handler.flush()

            text = path.read_text(encoding="utf-8")
            self.assertIn("DEBUG", text)
            self.assertIn("citysim.test: hello logging", text)
            configure_logging("WARNING", force=True)

    def test_configure_logging_replaces_managed_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.log"
            second = Path(directory) / "second.log"
            configure_logging("INFO", first, force=True)
            configure_logging("INFO", second, force=True)

            logging.getLogger("citysim.test").info("second target")
            for handler in logging.getLogger(LOGGER_NAME).handlers:
                handler.flush()

            self.assertIn("second target", second.read_text(encoding="utf-8"))
            self.assertEqual(first.read_text(encoding="utf-8"), "")
            configure_logging("WARNING", force=True)


class TestProfiling(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging("WARNING", force=True)

    def test_profile_formats_follow_output_suffix(self) -> None:
        self.assertEqual(profile_format_for("run.svg"), "flamegraph")
        self.assertEqual(profile_format_for("run.raw"), "raw")
        self.assertEqual(profile_format_for("run.txt"), "raw")
        self.assertEqual(profile_format_for("run.json"), "speedscope")
        self.assertEqual(profile_format_for("run.unknown"), "flamegraph")

    def test_default_cprofile_paths_follow_command_outputs(self) -> None:
        self.assertEqual(
            default_profile_path("train", None),
            Path("runs/train/profile.prof"),
        )
        self.assertEqual(
            default_profile_path("experiment", None, name="e5"),
            Path("runs/e5/profile.prof"),
        )
        self.assertEqual(
            default_profile_path(
                "experiment", None, out_dir="custom", name="e5"
            ),
            Path("custom/profile.prof"),
        )
        self.assertEqual(
            default_profile_path("demo", "raw"), Path("raw.prof")
        )
        self.assertEqual(
            default_profile_path("compare", "custom.prof"),
            Path("custom.prof"),
        )
        self.assertEqual(
            default_profile_path("compare", "custom.stats"),
            Path("custom.prof"),
        )

    def test_default_pyspy_paths_follow_command_outputs(self) -> None:
        self.assertEqual(
            default_profile_path("train", None, engine="py-spy"),
            Path("runs/train/profile.svg"),
        )
        self.assertEqual(
            default_profile_path(
                "experiment", None, name="e5", engine="py-spy"
            ),
            Path("runs/e5/profile.svg"),
        )
        self.assertEqual(
            default_profile_path("demo", "raw", engine="py-spy"),
            Path("raw.svg"),
        )
        self.assertEqual(
            default_profile_path(
                "compare", "custom.raw", engine="py-spy"
            ),
            Path("custom.raw"),
        )
        self.assertEqual(
            default_profile_path(
                "compare", "custom.stats", engine="py-spy"
            ),
            Path("custom.svg"),
        )

    def test_strip_profile_args_removes_control_flags(self) -> None:
        self.assertEqual(
            strip_profile_args(
                [
                    "train",
                    "--profile",
                    "--profile-engine",
                    "py-spy",
                    "--profile-output",
                    "runs/profile.svg",
                    "--quick",
                ]
            ),
            ["train", "--quick"],
        )
        self.assertEqual(
            strip_profile_args(
                [
                    "train",
                    "--profile-output=runs/profile.svg",
                    "--profile-engine=cprofile",
                    "--profile",
                ]
            ),
            ["train"],
        )

    def test_profile_session_without_workers_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.prof"
            with ProfileSession(path):
                sum(i * i for i in range(10000))

            self.assertTrue(path.is_file())
            report = path.with_suffix(".txt")
            self.assertTrue(report.is_file())
            self.assertIn(
                "worker processes merged: 0",
                report.read_text(encoding="utf-8"),
            )
            stats = pstats.Stats(str(path))
            self.assertGreater(stats.total_tt, 0.0)
            self.assertFalse(path.with_name("profile.workers").exists())

    def test_profile_session_merges_worker_profiles(self) -> None:
        city = quick_city_config(n_agents=2, actions_per_agent=1)
        evolution = EvolutionConfig(
            population_size=2,
            generations=1,
            episodes_per_generation=1,
            elites=1,
            islands=1,
            use_islands=False,
            hidden_size=8,
            use_parallel=True,
            max_workers=2,
        )
        layout = layout_from_config(evolution)
        genome = random_genome(np.random.default_rng(17), layout)
        tasks = [
            EpisodeTask([genome, genome], seed) for seed in (31, 32)
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.prof"
            with ProfileSession(path):
                with EpisodeExecutor(
                    city, layout, parallel=True, max_workers=2
                ) as executor:
                    results = executor.map(tasks)

            self.assertEqual(len(results), 2)
            worker_dir = path.with_name("profile.workers")
            worker_profiles = sorted(worker_dir.glob("worker-*.prof"))
            self.assertGreaterEqual(len(worker_profiles), 1)

            report = path.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn(
                f"worker processes merged: {len(worker_profiles)}",
                report,
            )
            self.assertIn("_run_episode_worker_impl", report)
            stats = pstats.Stats(str(path))
            function_names = {key[2] for key in stats.stats}
            self.assertIn("_run_episode_worker_impl", function_names)

    def test_run_with_pyspy_builds_subprocess_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pyspy = root / "py-spy.exe"
            pyspy.write_text("", encoding="utf-8")
            output = root / "profiles" / "run.raw"
            completed = subprocess.CompletedProcess([], 0)
            with patch(
                "citysim.profiling.subprocess.run", return_value=completed
            ) as run:
                code = run_with_pyspy(
                    [
                        "train",
                        "--quick",
                        "--profile",
                        "--profile-engine",
                        "py-spy",
                        "--profile-output",
                        "old.svg",
                    ],
                    output,
                    executable=pyspy,
                )

            self.assertEqual(code, 0)
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(pyspy))
            self.assertEqual(command[1], "record")
            self.assertIn("--subprocesses", command)
            self.assertEqual(
                command[command.index("--format") + 1], "raw"
            )
            self.assertEqual(
                command[command.index("--output") + 1], str(output)
            )
            separator = command.index("--")
            self.assertEqual(
                command[separator + 1 :],
                [sys.executable, "-m", "citysim", "train", "--quick"],
            )
            self.assertEqual(
                run.call_args.kwargs["env"][PROFILE_CHILD_ENV], "1"
            )
            self.assertTrue(output.parent.is_dir())

    def test_run_with_pyspy_rejects_missing_binary(self) -> None:
        with patch("citysim.profiling.find_pyspy", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "py-spy is required"):
                run_with_pyspy(["train"], Path("profile.svg"))

    def test_cli_delegates_pyspy_profile_to_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.svg"
            with patch(
                "citysim.cli.run_with_pyspy", return_value=0
            ) as run:
                main(
                    [
                        "train",
                        "--quick",
                        "--profile",
                        "--profile-engine",
                        "py-spy",
                        "--profile-output",
                        str(output),
                    ]
                )

            raw_argv, profile_path = run.call_args.args
            self.assertIn("--profile-engine", raw_argv)
            self.assertEqual(profile_path, output)

    def test_cli_uses_cprofile_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.prof"
            with patch("citysim.cli.command_train") as command:
                main(
                    [
                        "train",
                        "--quick",
                        "--profile",
                        "--profile-output",
                        str(output),
                    ]
                )

            command.assert_called_once()
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".txt").is_file())
            self.assertFalse(
                output.with_name("profile.workers").exists()
            )

    def test_cli_accepts_observability_and_profile_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "train",
                "--quick",
                "--profile",
                "--profile-engine",
                "py-spy",
                "--profile-output",
                "runs/custom.svg",
                "--log-level",
                "debug",
                "--log-file",
                "runs/test.log",
            ]
        )
        self.assertTrue(args.profile)
        self.assertEqual(args.profile_engine, "py-spy")
        self.assertEqual(args.profile_output, "runs/custom.svg")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.log_file, "runs/test.log")

        default = build_parser().parse_args(["train", "--profile"])
        self.assertEqual(default.profile_engine, "cprofile")

        dashboard = build_parser().parse_args(
            ["dashboard", "--log-level", "warning"]
        )
        self.assertEqual(dashboard.log_level, "WARNING")

        history = build_parser().parse_args(
            ["history", "log", "--log-level", "error"]
        )
        self.assertEqual(history.log_level, "ERROR")


if __name__ == "__main__":
    unittest.main()
