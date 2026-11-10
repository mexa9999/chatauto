"""Loopback-only web dashboard for training history and long-running jobs."""

from __future__ import annotations

import asyncio
import csv
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Literal
import uuid
import webbrowser
import zipfile

import numpy as np

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.staticfiles import StaticFiles

from .experiment_store import (
    MAX_ARCHIVE_BYTES,
    ArchiveError,
    ExperimentStore,
    ImportConflict,
)
from .analytics import load_analytics
from .config import EvolutionConfig
from .evolution import layout_from_config
from .live_demo import (
    read_control_file,
    read_live_frame,
    read_live_frames,
    update_control_file,
)
from .plotting import plot_city
from .checkpoint_io import read_checkpoint_metadata
from .training_history import TrainingHistoryStore

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_EXPERIMENTS = tuple([f"e{i}" for i in range(1, 13)] + ["all"])


class StartJobRequest(BaseModel):
    kind: Literal["demo", "train", "experiment"] = "train"
    branch: str = "main"
    resume: bool = False
    experiment: str = "e1"
    policy: Literal["random", "greedy", "manual", "neural"] = "manual"
    genome_ref: str | None = None
    grid: str = "12x12"
    shock: Literal["industry_removal", "population_boom", "earthquake"] | None = None
    projection: Literal["isometric", "top-down"] = "isometric"
    quick: bool = False
    rounds: int | None = Field(default=None, ge=1, le=10_000)
    generations: int = Field(default=100, ge=1, le=100_000)
    episodes: int = Field(default=10, ge=1, le=100_000)
    population: int = Field(default=32, ge=8, le=10_000)
    workers: int | None = Field(default=None, ge=1, le=1024)
    n_agents: int = Field(default=10, ge=1, le=1000)
    seed: int = Field(default=0, ge=0)
    speed_ms: int = Field(default=80, ge=0, le=2000)
    checkpoint_every: int = Field(default=10, ge=1, le=100_000)
    seeds: str = Field(default="0-2", min_length=1, max_length=512)
    eval_seeds: int = Field(default=5, ge=1, le=10_000)


class CompareRequest(BaseModel):
    left_ref: str
    right_ref: str
    reference_ref: str
    seed_base: int = Field(default=9_000_000, ge=0)
    seed_count: int = Field(default=5, ge=1, le=10_000)


class BranchRequest(BaseModel):
    name: str
    from_ref: str | None = None


class BranchRenameRequest(BaseModel):
    new_name: str


class BranchArchiveRequest(BaseModel):
    archived: bool = True


class FastForwardRequest(BaseModel):
    to_ref: str


class CheckoutRequest(BaseModel):
    branch: str


class ResetRequest(BaseModel):
    branch: str
    to_ref: str


class GCRequest(BaseModel):
    confirmation_token: str


class ImportCommitRequest(BaseModel):
    token: str
    action: Literal["skip", "rename", "replace"] | None = None
    name: str | None = None


class StorageSelectionRequest(BaseModel):
    entity_type: Literal["run", "dataset", "branch", "genome"]
    ids: list[str] = Field(min_length=1, max_length=1000)


class GenomeMetadataRequest(BaseModel):
    updates: dict


class GenomeCloneRequest(BaseModel):
    name: str | None = None


class GenomeCompareRequest(BaseModel):
    left_id: str
    right_id: str


class DemoControlRequest(BaseModel):
    paused: bool | None = None
    speed_ms: int | None = Field(default=None, ge=0, le=2000)


def _genome_npz(store: TrainingHistoryStore, genome_id: str) -> bytes:
    item, vector = store.genome(genome_id)
    metadata = dict(item["metadata"])
    if item["source_kind"] == "checkpoint":
        checkpoint_metadata = read_checkpoint_metadata(
            store.objects / f"{item['checkpoint_hash']}.npz"
        )
        architecture = checkpoint_metadata.get("evolution_config")
    else:
        architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"genome has no architecture metadata: {genome_id}")
    evolution_config = EvolutionConfig(**architecture)
    expected = layout_from_config(evolution_config).parameter_count
    if vector.shape != (expected,):
        raise ValueError(
            f"genome has {vector.size} parameters; architecture requires {expected}"
        )
    metadata.update(
        {
            "format": "citysim-genome",
            "version": 1,
            "parameter_count": expected,
            "architecture": architecture,
            "source": {
                "genome_id": genome_id,
                "source_kind": item["source_kind"],
                "branch": item.get("branch_name") or metadata.get("branch"),
                "generation": item.get("generation"),
                "member_kind": item["member_kind"],
                "member_index": item.get("member_index"),
                "parent_genome_id": item.get("parent_genome_id"),
            },
        }
    )
    buffer = io.BytesIO()
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    np.savez_compressed(
        buffer,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        genome=vector,
    )
    return buffer.getvalue()


def _pid_is_running(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        open_process.restype = wintypes.HANDLE
        get_exit_code_process = kernel32.GetExitCodeProcess
        get_exit_code_process.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        get_exit_code_process.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        handle = open_process(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error in (6, 87, 1168):
                return False
            if error == 5:
                return True
            raise ctypes.WinError(error)
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code_process(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == still_active
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if getattr(exc, "winerror", None) in (87, 1168):
            return False
        raise
    return True


class JobManager:
    def __init__(
        self,
        history_dir: Path,
        output_dir: Path,
        experiment_store: ExperimentStore | None = None,
    ):
        self.history_dir = history_dir
        self.output_dir = output_dir
        self.jobs_dir = output_dir / ".dashboard" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_store = experiment_store
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.job_id: str | None = None
        self.external_pid: int | None = None
        self.last_request = self._find_last_request()
        self._recover_active_job()

    def _find_last_request(self) -> dict | None:
        requests = sorted(
            self.jobs_dir.glob("*/request.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in requests:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("kind") in ("train", "experiment"):
                return data
        return None

    def _active(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return self.external_pid is not None and _pid_is_running(self.external_pid)

    def _recover_active_job(self) -> None:
        for request_path in sorted(
            self.jobs_dir.glob("*/request.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            job_dir = request_path.parent
            pid_path = job_dir / "pid.txt"
            progress_path = job_dir / "progress.json"
            if not pid_path.exists():
                continue
            progress = (
                json.loads(progress_path.read_text(encoding="utf-8"))
                if progress_path.exists()
                else {}
            )
            if progress.get("state") in ("completed", "stopped"):
                continue
            pid = int(pid_path.read_text(encoding="utf-8"))
            if _pid_is_running(pid):
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.job_id = job_dir.name
                self.external_pid = pid
                if request.get("kind") in ("train", "experiment"):
                    self.last_request = request
                return

    def start(self, request: StartJobRequest | CompareRequest) -> str:
        with self.lock:
            if self._active():
                raise HTTPException(409, "A dashboard job is already running")
            job_id = uuid.uuid4().hex
            job_dir = self.jobs_dir / job_id
            job_data = request.model_dump()
            job_data["kind"] = getattr(request, "kind", "compare")
            job_data["history_dir"] = str(self.history_dir)
            command = self._command(
                job_data,
                job_dir,
                job_dir / "progress.json",
                job_dir / "stop.request",
            )
            job_dir.mkdir(parents=True, exist_ok=False)
            request_path = job_dir / "request.json"
            job_data["created_at"] = time.time()
            request_path.write_text(
                json.dumps(job_data, sort_keys=True), encoding="utf-8"
            )
            if job_data["kind"] == "demo":
                update_control_file(
                    job_dir / "control.json",
                    paused=False,
                    speed_ms=job_data["speed_ms"],
                )
            if self.experiment_store is not None:
                self.experiment_store.register_run(
                    job_id, job_data["kind"], job_data, job_dir / "artifacts"
                )
            progress_path = job_dir / "progress.json"
            stop_path = job_dir / "stop.request"
            log_path = job_dir / "training.log"
            log_stream = log_path.open("ab")
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parent.parent,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError:
                log_stream.close()
                if self.experiment_store is not None:
                    self.experiment_store.update_run(job_id, "failed")
                raise
            log_stream.close()
            self.job_id = job_id
            self.external_pid = self.process.pid
            (job_dir / "pid.txt").write_text(
                str(self.process.pid), encoding="ascii"
            )
            if job_data["kind"] in ("train", "experiment"):
                self.last_request = job_data
            return job_id

    def _command(
        self,
        job: dict,
        job_dir: Path,
        progress_path: Path,
        stop_path: Path,
    ) -> list[str]:
        if job["kind"] in ("train", "experiment"):
            if job["population"] < job["n_agents"]:
                raise HTTPException(
                    422, "Population must be at least as large as the city agent count"
                )
            if job["population"] % 4:
                raise HTTPException(
                    422, "Population must be divisible by the default four islands"
                )
            if job["episodes"] * job["n_agents"] < job["population"]:
                raise HTTPException(
                    422, "Episodes times city agents must cover the whole population"
                )
        common = [
            sys.executable,
            "-m",
            "citysim",
        ]
        if job["kind"] == "compare":
            return common + [
                "_dashboard-compare-worker",
                str(job_dir / "request.json"),
                str(job_dir / "artifacts" / "compare.json"),
                str(progress_path),
                str(stop_path),
            ]
        artifact_dir = job_dir / "artifacts"
        if job["kind"] == "demo":
            command = common + [
                "demo",
                "--out",
                str(artifact_dir),
                "--policy",
                job["policy"],
                "--grid",
                job["grid"],
                "--seed",
                str(job["seed"]),
                "--projection",
                job["projection"],
                "--live-state-file",
                str(artifact_dir / "frames.jsonl"),
                "--control-file",
                str(job_dir / "control.json"),
            ]
            if job["policy"] == "neural":
                if not job.get("genome_ref"):
                    raise HTTPException(
                        422, "Select a trained checkpoint for the neural policy"
                    )
                try:
                    history = TrainingHistoryStore(self.history_dir)
                    genome_path = history.checkpoint_path(job["genome_ref"])
                    _, evolution = history.config_for_branch(job["genome_ref"])
                except (ValueError, FileNotFoundError) as exc:
                    raise HTTPException(400, f"Invalid trained checkpoint: {exc}") from exc
                command.extend(
                    (
                        "--genome",
                        str(genome_path),
                        "--hidden",
                        str(evolution.hidden_size),
                        "--attention-dim",
                        str(evolution.attention_dim),
                        "--district-hidden",
                        str(evolution.district_hidden),
                        "--global-hidden",
                        str(evolution.global_hidden),
                    )
                )
            if job["quick"]:
                command.append("--quick")
            if job["shock"] is not None:
                command.extend(("--shock", job["shock"]))
            if job["rounds"] is not None:
                command.extend(("--actions", str(job["rounds"])))
            return command
        if job["kind"] == "train":
            branch = job["branch"]
            store = TrainingHistoryStore(self.history_dir)
            existing = next(
                (item for item in store.branch_names() if item["name"] == branch),
                None,
            )
            if existing is None:
                raise HTTPException(404, f"Unknown branch: {branch}")
            if job["resume"] and existing["head_commit"] is None:
                raise HTTPException(400, "Cannot resume an empty branch")
            if not job["resume"] and existing["head_commit"] is not None:
                raise HTTPException(
                    409, "Branch already has snapshots; enable resume or create a branch"
                )
            command = common + [
                "train",
                "--out",
                str(artifact_dir),
                "--history-dir",
                str(self.history_dir),
                "--branch",
                branch,
                "--tag",
                f"dashboard_{job_dir.name}",
            ]
        elif job["kind"] == "experiment":
            if job["experiment"] not in _EXPERIMENTS:
                raise HTTPException(400, "Unknown experiment")
            command = common + [
                "experiment",
                job["experiment"],
                "--out",
                str(artifact_dir),
                "--history-dir",
                str(self.history_dir),
            ]
        else:
            raise HTTPException(400, "Unknown dashboard job type")

        if job["quick"]:
            command.append("--quick")
        for name, value in (
            ("--generations", job["generations"]),
            ("--episodes", job["episodes"]),
            ("--population", job["population"]),
            ("--n-agents", job["n_agents"]),
            ("--seed", job["seed"]),
            ("--checkpoint-every", job["checkpoint_every"]),
            ("--eval-seeds", job["eval_seeds"]),
        ):
            command.extend((name, str(value)))
        if job["rounds"] is not None:
            command.extend(("--actions", str(job["rounds"])))
        if job["workers"] is not None:
            command.extend(("--workers", str(job["workers"])))
        if job["kind"] == "experiment":
            command.extend(("--seeds", job["seeds"]))
        if job["resume"]:
            command.append("--resume")
        command.extend(("--progress-file", str(progress_path), "--stop-file", str(stop_path)))
        return command

    def stop(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        with self.lock:
            if self.job_id != job_id or not self._active():
                raise HTTPException(409, "Job is not running")
            (job_dir / "stop.request").touch()

    def control_demo(
        self, job_id: str, paused: bool | None, speed_ms: int | None
    ) -> dict:
        job_dir = self._job_dir(job_id)
        with self.lock:
            if self.job_id != job_id or not self._active():
                raise HTTPException(409, "Job is not running")
            request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
            if request["kind"] != "demo":
                raise HTTPException(409, "Only demo jobs have live controls")
            return update_control_file(
                job_dir / "control.json", paused=paused, speed_ms=speed_ms
            )

    def demo_control(self, job_id: str) -> dict:
        job_dir = self._job_dir(job_id)
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        if request["kind"] != "demo":
            raise HTTPException(409, "Only demo jobs have live controls")
        try:
            return read_control_file(job_dir / "control.json")
        except (OSError, ValueError) as exc:
            raise HTTPException(500, f"Could not read demo controls: {exc}") from exc

    def live_frames(self, job_id: str, offset: int) -> dict:
        job_dir = self._job_dir(job_id)
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        if request.get("kind") != "demo":
            raise HTTPException(404, "Job has no live simulation state")
        return read_live_frames(job_dir / "artifacts" / "frames.jsonl", offset)

    def live_frame(self, job_id: str, sequence: int) -> dict:
        job_dir = self._job_dir(job_id)
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        if request.get("kind") != "demo":
            raise HTTPException(404, "Job has no live simulation state")
        try:
            return read_live_frame(
                job_dir / "artifacts" / "frames.jsonl", sequence
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    def resume_last(self) -> str:
        if self.last_request is None:
            raise HTTPException(404, "No prior training or experiment job to resume")
        request = StartJobRequest.model_validate({**self.last_request, "resume": True})
        return self.start(request)

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "Unknown job")
        path = self.jobs_dir / job_id
        if not path.is_dir():
            raise HTTPException(404, "Unknown job")
        return path

    def status(self) -> dict:
        if self.job_id is None:
            return {"state": "idle", "job_id": None}
        job_dir = self._job_dir(self.job_id)
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        progress_path = job_dir / "progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else None
        )
        if self._active():
            state = "stopping" if (job_dir / "stop.request").exists() else "running"
            exit_code = None
        else:
            exit_code = self.process.poll() if self.process is not None else None
            progress_state = progress.get("state") if progress else None
            if progress_state in ("completed", "stopped"):
                state = progress_state
            elif (job_dir / "stop.request").exists() and exit_code == 0:
                state = "stopped"
            else:
                state = "completed" if exit_code == 0 else "failed"
        if self.experiment_store is not None:
            self.experiment_store.update_run(self.job_id, state)
        result_path = job_dir / "artifacts" / "compare.json"
        if not result_path.exists():
            result_path = job_dir / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        return {
            "job_id": self.job_id,
            "kind": request["kind"],
            "state": state,
            "exit_code": exit_code,
            "progress": progress,
            "result": result,
            "log_url": f"/api/jobs/{self.job_id}/logs",
        }

    def list_jobs(self) -> list[dict]:
        jobs = []
        for request_path in sorted(
            self.jobs_dir.glob("*/request.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            job_dir = request_path.parent
            job_id = job_dir.name
            request = json.loads(request_path.read_text(encoding="utf-8"))
            progress_path = job_dir / "progress.json"
            progress = (
                json.loads(progress_path.read_text(encoding="utf-8"))
                if progress_path.exists()
                else {}
            )
            if job_id == self.job_id:
                state = self.status()["state"]
            else:
                state = progress.get("state", "unknown")
                if state in ("running", "stopping"):
                    pid_path = job_dir / "pid.txt"
                    if pid_path.is_file() and not _pid_is_running(
                        int(pid_path.read_text(encoding="ascii"))
                    ):
                        state = "failed"
            jobs.append(
                {
                    "job_id": job_id,
                    "kind": request.get("kind", "unknown"),
                    "state": state,
                    "created_at": request.get("created_at"),
                    "progress": progress,
                }
            )
        return jobs

    def read_log(self, job_id: str, offset: int, limit: int) -> dict:
        path = self._job_dir(job_id) / "training.log"
        if not path.exists():
            return {"offset": 0, "next_offset": 0, "text": ""}
        with path.open("rb") as stream:
            size = path.stat().st_size
            start = min(offset, size)
            stream.seek(start)
            content = stream.read(min(limit, 131_072))
        return {
            "offset": start,
            "next_offset": start + len(content),
            "text": content.decode("utf-8", errors="replace"),
        }

    def shutdown(self) -> None:
        if self._active() and self.job_id is not None:
            (self._job_dir(self.job_id) / "stop.request").touch()
            if self.process is not None:
                self.process.wait()
            elif self.external_pid is not None:
                while _pid_is_running(self.external_pid):
                    time.sleep(0.25)


def create_app(history_dir: str | Path, output_dir: str | Path, port: int = 8765) -> FastAPI:
    requested_history_path = Path(history_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    experiment_store = ExperimentStore(
        output_path / ".dashboard" / "store", requested_history_path
    )
    history_path = experiment_store.active_history_dir()
    store = TrainingHistoryStore(history_path)
    manager = JobManager(history_path, output_path, experiment_store)
    expected_origin = f"http://127.0.0.1:{port}"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="CitySim Training Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    web_root = Path(__file__).parent / "web"
    web_build = web_root / "build"
    @app.middleware("http")
    async def restrict_to_local_browser(request: Request, call_next):
        client_host = request.client.host if request.client is not None else ""
        host = request.headers.get("host", "").split(":", 1)[0].lower()
        if client_host not in ("127.0.0.1", "::1", "localhost") or host not in (
            "127.0.0.1",
            "localhost",
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "Dashboard accepts loopback connections only"},
            )
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if origin.rstrip("/") != expected_origin:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin actions are not allowed"},
                )
        return await call_next(request)

    @app.get("/")
    def dashboard_page():
        built_page = web_build / "index.html"
        return FileResponse(
            built_page
            if built_page.is_file()
            else web_root / "dashboard.html"
        )

    if (web_build / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=web_build / "assets"),
            name="dashboard-assets",
        )

    def current_history_store() -> TrainingHistoryStore:
        return TrainingHistoryStore(manager.history_dir)

    @app.get("/api/branches")
    def branches():
        current = current_history_store()
        return {"active": current.active_branch(), "branches": current.branch_names()}

    @app.get("/api/demo/checkpoints")
    def demo_checkpoints():
        current = current_history_store()
        checkpoints = []
        for branch in current.branch_names():
            if branch["head_commit"] is None:
                continue
            for commit in current.history(branch["name"]):
                checkpoints.append(
                    {
                        "ref": f"{branch['name']}@{commit['generation']}",
                        "branch": branch["name"],
                        "generation": commit["generation"],
                        "best_fitness": commit["best_fitness"],
                        "complete": bool(commit["complete"]),
                    }
                )
        return {"checkpoints": checkpoints}

    @app.get("/api/history")
    def history(branch: str = Query(default="main")):
        store = current_history_store()
        branch_info = next(
            (item for item in store.branch_names() if item["name"] == branch),
            None,
        )
        if branch_info is None:
            raise HTTPException(404, f"Unknown branch: {branch}")
        if branch_info["head_commit"] is None:
            return {"branch": branch, "commits": []}
        try:
            return {"branch": branch, "commits": store.history(branch)}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/verify")
    def verify():
        return current_history_store().verify()

    @app.get("/api/gc/preview")
    def gc_preview():
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before collecting history")
        return current_history_store().gc_preview()

    @app.post("/api/gc")
    def gc_apply(request: GCRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before collecting history")
        try:
            return current_history_store().gc_apply(request.confirmation_token)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/branches")
    def create_branch(request: BranchRequest):
        try:
            parent = current_history_store().create_branch(
                request.name, request.from_ref
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"name": request.name, "parent": parent}

    @app.put("/api/branches/{name}")
    def rename_branch(name: str, request: BranchRenameRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before renaming branches")
        try:
            current_history_store().rename_branch(name, request.new_name)
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"name": request.new_name}

    @app.post("/api/branches/{name}/archive")
    def archive_branch(name: str, request: BranchArchiveRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before archiving branches")
        try:
            current_history_store().archive_branch(name, request.archived)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"name": name, "archived": request.archived}

    @app.post("/api/branches/{name}/fast-forward")
    def fast_forward_branch(name: str, request: FastForwardRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before moving branch heads")
        try:
            head = current_history_store().fast_forward_branch(name, request.to_ref)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"name": name, "head_commit": head}

    @app.delete("/api/branches/{name}")
    def delete_branch(name: str):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before deleting branches")
        try:
            current_history_store().delete_branch(name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "trashed", "name": name}

    @app.post("/api/branches/{name}/restore")
    def restore_branch(name: str):
        try:
            current_history_store().restore_branch(name)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "restored", "name": name}

    @app.get("/api/genomes")
    def genomes(
        branch: str | None = None,
        query: str = Query(default="", max_length=200),
        include_deleted: bool = False,
    ):
        try:
            return {"genomes": current_history_store().list_genomes(
                branch=branch, query=query, include_deleted=include_deleted
            )}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/genomes/{genome_id}")
    def genome_details(genome_id: str):
        try:
            item, vector = current_history_store().genome(genome_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        item["vector_summary"] = {
            "parameter_count": int(vector.size),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "l2_norm": float(np.linalg.norm(vector)),
        }
        return item

    @app.put("/api/genomes/{genome_id}")
    def update_genome(genome_id: str, request: GenomeMetadataRequest):
        try:
            return {
                "genome_id": genome_id,
                "metadata": current_history_store().update_genome_metadata(
                    genome_id, request.updates
                ),
            }
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/genomes/{genome_id}/clone")
    def clone_genome(genome_id: str, request: GenomeCloneRequest):
        try:
            return current_history_store().clone_genome(genome_id, request.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/genomes/compare")
    def compare_genomes(request: GenomeCompareRequest):
        try:
            return current_history_store().compare_genomes(
                request.left_id, request.right_id
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/genomes")
    def delete_genomes(request: StorageSelectionRequest):
        if request.entity_type != "genome":
            raise HTTPException(422, "entity_type must be genome")
        return {
            "status": "trashed",
            "count": current_history_store().delete_genomes(request.ids),
        }

    @app.post("/api/genomes/{genome_id}/restore")
    def restore_genome(genome_id: str):
        try:
            current_history_store().restore_genome(genome_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "restored", "genome_id": genome_id}

    @app.get("/api/genomes/{genome_id}/export")
    def export_genome(genome_id: str):
        try:
            payload = _genome_npz(current_history_store(), genome_id)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return StreamingResponse(
            io.BytesIO(payload),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="citysim-genome-{genome_id[:8]}.npz"'
            },
        )

    @app.post("/api/genomes/import")
    async def import_genome(request: Request, name: str = Query(default="Imported genome", max_length=100)):
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > 64 * 1024 * 1024:
                raise HTTPException(413, "Genome archive exceeds 64 MiB")
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = archive.infolist()
                if (
                    len(members) != 2
                    or {member.filename for member in members}
                    != {"metadata.npy", "genome.npy"}
                    or sum(member.file_size for member in members)
                    > 128 * 1024 * 1024
                ):
                    raise ValueError("genome NPZ must contain only metadata and one vector under 128 MiB")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "genome"}:
                    raise ValueError("genome NPZ must contain metadata and genome arrays")
                raw_metadata = archive["metadata"]
                metadata = json.loads(np.asarray(raw_metadata, dtype=np.uint8).tobytes().decode("utf-8"))
                raw_vector = np.asarray(archive["genome"])
            if not isinstance(metadata, dict):
                raise ValueError("genome metadata must be a JSON object")
            if metadata.get("format") != "citysim-genome" or type(metadata.get("version")) is not int or metadata["version"] != 1:
                raise ValueError("unsupported genome file; expected CitySim genome format v1")
            if raw_vector.ndim != 1 or raw_vector.dtype.kind not in "fiu":
                raise ValueError("genome vector must be a one-dimensional real numeric array")
            vector = np.asarray(raw_vector, dtype=np.float64)
            architecture = metadata.get("architecture")
            if not isinstance(architecture, dict):
                raise ValueError("genome file has no architecture metadata")
            evolution = EvolutionConfig(**architecture)
            expected = layout_from_config(evolution).parameter_count
            parameter_count = metadata.get("parameter_count")
            if (
                type(parameter_count) is not int
                or parameter_count != expected
                or vector.shape != (expected,)
            ):
                raise ValueError(
                    f"genome has {vector.size} parameters; architecture requires {expected}"
                )
            if not np.isfinite(vector).all():
                raise ValueError("genome vector contains NaN or infinite values")
            metadata["name"] = name.strip() or "Imported genome"
            return current_history_store().import_genome(metadata, vector)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise HTTPException(422, f"Genome import validation failed: {exc}") from exc

    @app.post("/api/checkout")
    def checkout(request: CheckoutRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before changing branches")
        try:
            current_history_store().checkout(request.branch)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"active": request.branch}

    @app.post("/api/reset")
    def reset(request: ResetRequest):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before moving branch pointers")
        try:
            head = current_history_store().reset_branch(
                request.branch, request.to_ref
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"branch": request.branch, "head": head}

    @app.post("/api/jobs")
    def start_job(request: StartJobRequest):
        if request.kind not in ("demo", "train", "experiment"):
            raise HTTPException(400, "Unknown dashboard job type")
        if request.kind == "experiment" and request.experiment not in _EXPERIMENTS:
            raise HTTPException(400, "Unknown experiment")
        return {"job_id": manager.start(request)}

    @app.post("/api/compare")
    def compare(request: CompareRequest):
        for ref in (request.left_ref, request.right_ref, request.reference_ref):
            try:
                current_history_store().resolve_ref(ref)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        compare_job = request.model_dump()
        compare_job["kind"] = "compare"
        return {"job_id": manager.start(CompareRequest(**compare_job))}

    @app.get("/api/jobs")
    def job_status():
        return manager.status()

    @app.get("/api/jobs/history")
    def job_history():
        return {"jobs": manager.list_jobs()}

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(
        job_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=65_536, ge=1, le=131_072),
    ):
        return manager.read_log(job_id, offset, limit)

    @app.get("/api/jobs/{job_id}/frames")
    def job_frames(
        job_id: str,
        offset: int = Query(default=0, ge=0),
    ):
        try:
            return manager.live_frames(job_id, offset)
        except ValueError as exc:
            raise HTTPException(500, f"Could not read live state: {exc}") from exc

    @app.get("/api/jobs/{job_id}/frames/{sequence}/image")
    def job_frame_image(
        job_id: str,
        sequence: int,
        projection: Literal["isometric", "top-down"] = "isometric",
    ):
        frame = manager.live_frame(job_id, sequence)
        if "type_grid" not in frame:
            raise HTTPException(400, "Selected frame has no city state")
        destination = experiment_store.pending / f"frame-{uuid.uuid4().hex}.png"
        try:
            plot_city(
                SimpleNamespace(
                    type_grid=np.asarray(frame["type_grid"], dtype=np.int8),
                    quality_grid=np.asarray(
                        frame["quality_grid"], dtype=np.int8
                    ),
                    floors_grid=np.asarray(
                        frame.get(
                            "floors_grid",
                            np.where(
                                np.asarray(frame["type_grid"], dtype=np.int8) > 0,
                                1,
                                0,
                            ),
                        ),
                        dtype=np.int8,
                    ),
                    roof_quality_grid=np.asarray(
                        frame.get(
                            "roof_quality_grid",
                            np.zeros(
                                (frame["height"], frame["width"]), dtype=np.int8
                            ),
                        ),
                        dtype=np.int8,
                    ),
                    w=frame["width"],
                    h=frame["height"],
                ),
                destination,
                title=f"CitySim round {frame['round']}",
                projection=projection,
            )
        except (OSError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(500, f"Could not render state frame: {exc}") from exc
        return FileResponse(
            destination,
            media_type="image/png",
            filename=f"citysim-{job_id}-frame-{sequence}.png",
            background=BackgroundTask(destination.unlink, missing_ok=True),
        )

    @app.get("/api/jobs/{job_id}/frames/{sequence}")
    def job_frame(job_id: str, sequence: int):
        return manager.live_frame(job_id, sequence)

    @app.get("/api/jobs/{job_id}/control")
    def demo_control(job_id: str):
        return manager.demo_control(job_id)

    @app.post("/api/jobs/{job_id}/control")
    def control_demo(job_id: str, request: DemoControlRequest):
        return manager.control_demo(job_id, request.paused, request.speed_ms)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        manager.stop(job_id)
        return {"state": "stopping"}

    @app.post("/api/jobs/resume")
    def resume_job():
        return {"job_id": manager.resume_last()}

    @app.get("/api/data")
    def data_catalog():
        return {
            "runs": experiment_store.list_runs(),
            "datasets": experiment_store.list_datasets(),
            "active_history_dir": str(manager.history_dir),
        }

    @app.get("/api/data/trash")
    def data_trash():
        history = current_history_store()
        return {
            "items": [
                *experiment_store.trash(),
                *[
                    {"entity_type": "branch", **item}
                    for item in history.trash_branches()
                ],
                *[
                    {
                        "entity_type": "genome",
                        "id": item["genome_id"],
                        "genome_id": item["genome_id"],
                        "name": item["metadata"].get("name", item["genome_id"]),
                        "deleted_at": item["deleted_at"],
                    }
                    for item in history.trash_genomes()
                ],
            ]
        }

    @app.post("/api/data/trash")
    def trash_data(request: StorageSelectionRequest):
        if request.entity_type in {"branch", "genome"} and manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before changing branches or genomes")
        try:
            if request.entity_type in {"run", "dataset"}:
                count = experiment_store.trash_entities(request.entity_type, request.ids)
            elif request.entity_type == "genome":
                count = current_history_store().delete_genomes(request.ids)
            else:
                count = current_history_store().delete_branches(request.ids)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "trashed", "count": count}

    @app.post("/api/data/restore")
    def restore_data(request: StorageSelectionRequest):
        try:
            if request.entity_type in {"run", "dataset"}:
                count = experiment_store.restore_entities(request.entity_type, request.ids)
            elif request.entity_type == "genome":
                count = current_history_store().restore_genomes(request.ids)
            else:
                count = current_history_store().restore_branches(request.ids)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "restored", "count": count}

    @app.post("/api/data/purge")
    def purge_data(request: StorageSelectionRequest):
        try:
            if request.entity_type in {"run", "dataset"}:
                count = experiment_store.purge_entities(request.entity_type, request.ids)
            elif request.entity_type == "genome":
                count = current_history_store().purge_genomes(request.ids)
            else:
                count = current_history_store().purge_branches(request.ids)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "purged", "count": count}

    @app.get("/api/analytics")
    def analytics_catalog():
        return {"sources": experiment_store.analytics_sources()}

    @app.get("/api/analytics/{source_type}/{source_id}")
    def analytics_detail(
        source_type: Literal["run", "dataset"],
        source_id: str,
        round_start: int | None = Query(default=None, ge=0),
        round_end: int | None = Query(default=None, ge=0),
        agent_id: int | None = Query(default=None, ge=0),
    ):
        if not _JOB_ID.fullmatch(source_id):
            raise HTTPException(404, "Unknown analytics source")
        source = next(
            (
                item
                for item in experiment_store.analytics_sources()
                if item["source_type"] == source_type
                and item["source_id"] == source_id
            ),
            None,
        )
        if source is None:
            raise HTTPException(404, "Unknown analytics source")
        if round_start is not None and round_end is not None and round_start > round_end:
            raise HTTPException(422, "round_start must not exceed round_end")
        directory = experiment_store.analytics_directory(source_type, source_id)
        if directory is None:
            raise HTTPException(404, "Unknown analytics source")
        try:
            detail = load_analytics(
                directory,
                round_start=round_start,
                round_end=round_end,
                agent_id=agent_id,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(500, f"Could not read analytics: {exc}") from exc
        return {"source": source, **detail}

    @app.get("/api/analytics/{source_type}/{source_id}/export")
    def export_analytics(
        source_type: Literal["run", "dataset"],
        source_id: str,
        format: Literal["json", "csv"] = "json",
        round_start: int | None = Query(default=None, ge=0),
        round_end: int | None = Query(default=None, ge=0),
        agent_id: int | None = Query(default=None, ge=0),
    ):
        if not _JOB_ID.fullmatch(source_id):
            raise HTTPException(404, "Unknown analytics source")
        source = next(
            (
                item
                for item in experiment_store.analytics_sources()
                if item["source_type"] == source_type
                and item["source_id"] == source_id
            ),
            None,
        )
        directory = experiment_store.analytics_directory(source_type, source_id)
        if source is None or directory is None:
            raise HTTPException(404, "Unknown analytics source")
        if round_start is not None and round_end is not None and round_start > round_end:
            raise HTTPException(422, "round_start must not exceed round_end")
        filename = f"citysim-{source_id}-analytics.{format}"
        if format == "csv":
            def csv_rows():
                buffer = io.StringIO(newline="")
                writer = csv.writer(buffer)
                writer.writerow(
                    (
                        "timestamp", "elapsed_seconds", "round", "agent_id",
                        "outcome", "action_kind", "cell", "type_id", "quality",
                        "floors", "roof_quality", "cost", "bid", "damage",
                        "demolish", "budget", "profit",
                    )
                )
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)
                action_path = directory / "actions.ndjson"
                if not action_path.is_file():
                    return
                with action_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        round_number = int(event["round"])
                        current_agent = int(event["agent_id"])
                        if (
                            (round_start is not None and round_number < round_start)
                            or (round_end is not None and round_number > round_end)
                            or (agent_id is not None and current_agent != agent_id)
                        ):
                            continue
                        action = event["action"]
                        state = event["agent_state"]
                        writer.writerow(
                            (
                                event["timestamp"], event["elapsed_seconds"],
                                round_number, current_agent, event["outcome"],
                                action["kind"], action["cell"], action["type_id"],
                                action["quality"], action.get("floors", 1),
                                action.get("roof_quality", 0), action["cost"],
                                action["bid"], event["damage"], action["demolish"],
                                state["budget"], state["profit"],
                            )
                        )
                        yield buffer.getvalue()
                        buffer.seek(0)
                        buffer.truncate(0)

            return StreamingResponse(
                csv_rows(),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        def json_rows():
            summary_path = directory / "analytics_summary.json"
            summary = (
                summary_path.read_text(encoding="utf-8").strip()
                if summary_path.is_file()
                else "{}"
            )
            yield '{"source":'
            yield json.dumps(source, ensure_ascii=False)
            yield ',"summary":'
            yield summary
            yield ',"rounds":['
            first = True
            rounds_path = directory / "rounds.ndjson"
            if rounds_path.is_file():
                with rounds_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        number = int(event["round"])
                        if (
                            (round_start is not None and number < round_start)
                            or (round_end is not None and number > round_end)
                        ):
                            continue
                        if not first:
                            yield ","
                        first = False
                        yield json.dumps(event, ensure_ascii=False)
            yield '],"actions":['
            first = True
            actions_path = directory / "actions.ndjson"
            if actions_path.is_file():
                with actions_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        number = int(event["round"])
                        if (
                            (round_start is not None and number < round_start)
                            or (round_end is not None and number > round_end)
                            or (
                                agent_id is not None
                                and int(event["agent_id"]) != agent_id
                            )
                        ):
                            continue
                        if not first:
                            yield ","
                        first = False
                        yield json.dumps(event, ensure_ascii=False)
            training_path = directory / "training_metrics.json"
            if training_path.is_file():
                yield '],"training":'
                yield training_path.read_text(encoding="utf-8").strip()
                yield "}"
            else:
                yield "]}"

        return StreamingResponse(
            json_rows(),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/data/datasets/{dataset_id}")
    def dataset_details(dataset_id: str):
        try:
            return experiment_store.dataset_details(dataset_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/data/runs/{run_id}/export")
    def export_run(run_id: str, mode: Literal["light", "full"] = "light"):
        destination = experiment_store.pending / f"export-{uuid.uuid4().hex}.zip"
        try:
            experiment_store.create_export(
                run_id, mode, manager.history_dir, destination
            )
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(500, f"Could not create archive: {exc}") from exc
        return FileResponse(
            destination,
            media_type="application/zip",
            filename=f"citysim-{run_id}-{mode}.zip",
            background=BackgroundTask(destination.unlink, missing_ok=True),
        )

    @app.post("/api/data/import/preview")
    async def preview_import(name: str, request: Request):
        destination = experiment_store.pending / f"upload-{uuid.uuid4().hex}.zip"
        size = 0
        try:
            with destination.open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise HTTPException(
                            413, "Archive exceeds the 20 GiB upload limit"
                        )
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "Upload an archive file")
            return experiment_store.stage_import(destination, name)
        except (ArchiveError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            destination.unlink(missing_ok=True)

    @app.post("/api/data/import/commit")
    def commit_import(request: ImportCommitRequest):
        try:
            return experiment_store.commit_import(
                request.token, request.action, request.name
            )
        except ImportConflict as exc:
            raise HTTPException(
                409, {"message": str(exc), "conflicts": exc.conflicts}
            ) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f"Could not install dataset: {exc}") from exc

    @app.post("/api/data/datasets/{dataset_id}/activate")
    def activate_dataset(dataset_id: str):
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before switching history")
        try:
            manager.history_dir = experiment_store.activate_dataset(dataset_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        app.state.history_store = current_history_store()
        return {"active_history_dir": str(manager.history_dir)}

    @app.post("/api/data/history/default")
    def activate_default_history():
        if manager.status()["state"] in ("running", "stopping"):
            raise HTTPException(409, "Stop the active job before switching history")
        manager.history_dir = experiment_store.deactivate_dataset()
        app.state.history_store = current_history_store()
        return {"active_history_dir": str(manager.history_dir)}

    app.state.history_store = store
    app.state.job_manager = manager
    app.state.experiment_store = experiment_store
    return app


def run_dashboard(
    history_dir: str | Path,
    output_dir: str | Path,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    if host != "127.0.0.1":
        raise ValueError("the dashboard must bind to 127.0.0.1")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        create_app(history_dir, output_dir, port),
        host=host,
        port=port,
        log_level="info",
    )
