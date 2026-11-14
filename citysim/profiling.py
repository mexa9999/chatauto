"""Opt-in profiling for CLI commands, including worker processes.

The default engine uses ``cProfile`` in the main process and every episode
worker, then merges the results with ``pstats``. An optional ``py-spy`` engine
produces a flamegraph covering the whole process tree.
"""

from __future__ import annotations

from contextlib import contextmanager
import cProfile
import logging
from multiprocessing import util as multiprocessing_util
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import sys
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

PROFILE_CHILD_ENV = "CITYSIM_PROFILE_CHILD"
PY_SPY_OVERRIDE_ENV = "CITYSIM_PY_SPY"
WORKER_PROFILE_DIR_ENV = "CITYSIM_PROFILE_WORKER_DIR"
_PROFILE_FORMATS = {
    ".svg": "flamegraph",
    ".raw": "raw",
    ".txt": "raw",
    ".json": "speedscope",
}

_worker_profiler: cProfile.Profile | None = None


def profile_format_for(path: str | Path) -> str:
    """Return the py-spy output format matching the file suffix."""
    return _PROFILE_FORMATS.get(Path(path).suffix.lower(), "flamegraph")


def default_profile_path(
    command: str,
    output: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    name: str | None = None,
    engine: str = "cprofile",
) -> Path:
    """Return the profile path, mirroring each command's output dir."""
    if output is not None:
        path = Path(output)
    else:
        if out_dir is not None:
            base = Path(out_dir)
        elif command == "train":
            base = Path("runs/train")
        elif command == "experiment":
            base = Path("runs") / (name or "experiment")
        elif command == "demo":
            base = Path("runs/demo")
        elif command == "compare":
            base = Path("runs/compare")
        else:
            base = Path("runs")
        path = base / ("profile.prof" if engine == "cprofile" else "profile.svg")
    if engine == "cprofile":
        if path.suffix.lower() != ".prof":
            path = path.with_suffix(".prof")
    elif path.suffix.lower() not in _PROFILE_FORMATS:
        path = path.with_suffix(".svg")
    return path


def strip_profile_args(argv: Sequence[str]) -> list[str]:
    """Remove profiler-only flags before re-running the command under py-spy."""
    result: list[str] = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
            continue
        if argument == "--profile":
            continue
        if argument in ("--profile-output", "--profile-engine"):
            skip_value = True
            continue
        if argument.startswith("--profile-output="):
            continue
        if argument.startswith("--profile-engine="):
            continue
        result.append(argument)
    return result


def start_worker_profiling() -> None:
    """Start collecting cProfile data in the current process pool worker."""
    global _worker_profiler
    if _worker_profiler is not None:
        return
    if not os.environ.get(WORKER_PROFILE_DIR_ENV):
        return
    _worker_profiler = cProfile.Profile()
    multiprocessing_util.Finalize(
        None,
        _dump_worker_profile,
        exitpriority=0,
    )


def _dump_worker_profile() -> None:
    global _worker_profiler
    profiler = _worker_profiler
    if profiler is None:
        return
    _worker_profiler = None
    profiler.disable()
    directory = os.environ.get(WORKER_PROFILE_DIR_ENV)
    if not directory:
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(path / f"worker-{os.getpid()}.prof")
    except OSError:
        logger.exception("failed to write worker profile")


@contextmanager
def profile_worker_task() -> Iterator[None]:
    """Profile only active worker task execution, not pool idle time."""
    profiler = _worker_profiler
    if profiler is None:
        yield
        return
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()


class ProfileSession:
    """Collect cProfile data from the main process and worker processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        sort: str = "tottime",
        limit: int = 40,
    ) -> None:
        self.path = Path(path)
        self.report_path = (
            self.path.with_suffix(".txt")
            if self.path.suffix == ".prof"
            else self.path.with_name(self.path.name + ".txt")
        )
        self.worker_dir = self.path.with_name(f"{self.path.stem}.workers")
        self.sort = sort
        self.limit = limit
        self._profiler: cProfile.Profile | None = None
        self._previous_worker_dir: str | None = None

    def __enter__(self) -> ProfileSession:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.worker_dir.glob("worker-*.prof"):
            try:
                stale.unlink()
            except OSError:
                logger.warning("could not remove stale profile: %s", stale)
        self._previous_worker_dir = os.environ.get(WORKER_PROFILE_DIR_ENV)
        os.environ[WORKER_PROFILE_DIR_ENV] = str(self.worker_dir)
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        profiler = self._profiler
        if profiler is None:
            return False
        profiler.disable()
        try:
            self._save(profiler)
        except Exception:
            logger.exception("failed to aggregate profile data")
        finally:
            self._restore_worker_environment()
            self._profiler = None
        return False

    def _save(self, profiler: cProfile.Profile) -> None:
        main_path = self.path.with_name(
            f".{self.path.stem}.{os.getpid()}.main.prof"
        )
        profiler.dump_stats(main_path)
        worker_paths = sorted(self.worker_dir.glob("worker-*.prof"))
        stats: pstats.Stats | None = None
        for candidate in [main_path, *worker_paths]:
            try:
                if stats is None:
                    stats = pstats.Stats(str(candidate))
                else:
                    stats.add(str(candidate))
            except (OSError, EOFError, ValueError):
                logger.warning("skipping unreadable profile: %s", candidate)
        if stats is None:
            raise RuntimeError("no profile data was collected")

        stats.strip_dirs().sort_stats(self.sort)
        stats.dump_stats(str(self.path))
        with self.report_path.open("w", encoding="utf-8") as stream:
            stream.write(f"worker processes merged: {len(worker_paths)}\n")
            stats.stream = stream
            stats.print_stats(self.limit)
            stream.write(
                "\nTop functions by cumulative time "
                "(worker task roots included)\n"
            )
            stats.sort_stats("cumulative")
            stats.print_stats(self.limit)
        main_path.unlink(missing_ok=True)
        if not worker_paths:
            try:
                self.worker_dir.rmdir()
            except OSError:
                pass
        logger.info(
            "profile saved: %s (workers=%d, report: %s)",
            self.path,
            len(worker_paths),
            self.report_path,
        )

    def _restore_worker_environment(self) -> None:
        if self._previous_worker_dir is None:
            os.environ.pop(WORKER_PROFILE_DIR_ENV, None)
        else:
            os.environ[WORKER_PROFILE_DIR_ENV] = self._previous_worker_dir


def find_pyspy() -> Path | None:
    """Locate py-spy next to the active Python or on PATH."""
    override = os.environ.get(PY_SPY_OVERRIDE_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    executable = Path(sys.executable)
    for candidate in (
        executable.with_name("py-spy.exe"),
        executable.with_name("py-spy"),
    ):
        if candidate.is_file():
            return candidate
    found = shutil.which("py-spy")
    if found:
        return Path(found)
    return None


def run_with_pyspy(
    argv: Sequence[str],
    output: str | Path,
    *,
    executable: str | Path | None = None,
) -> int:
    """Run the current CLI command under ``py-spy record --subprocesses``.

    The child process sees the original command without profiling flags;
    ``CITYSIM_PROFILE_CHILD`` prevents recursion.
    """
    pyspy = Path(executable) if executable is not None else find_pyspy()
    if pyspy is None or not pyspy.is_file():
        raise RuntimeError(
            "py-spy is required for --profile-engine py-spy; install project "
            "dependencies or set CITYSIM_PY_SPY to the executable"
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(pyspy),
        "record",
        "--subprocesses",
        "--format",
        profile_format_for(path),
        "--output",
        str(path),
        "--",
        sys.executable,
        "-m",
        "citysim",
        *strip_profile_args(argv),
    ]
    environment = os.environ.copy()
    environment[PROFILE_CHILD_ENV] = "1"
    logger.debug("running: %s", " ".join(command))
    return subprocess.run(command, env=environment).returncode
