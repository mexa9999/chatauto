"""Progress reporting shared by CLI training and the local dashboard."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Callable


class TrainingStopped(Exception):
    """Raised after a batch experiment reaches a requested checkpoint boundary."""


class TrainingProgress:
    def __init__(self, path: str | Path, stop_path: str | Path):
        self.path = Path(path)
        self.stop_path = Path(stop_path)
        self.started = time.monotonic()
        self.last_generation: int | None = None
        self.last_update = self.started
        self.run_started = self.started
        self.run_index = 0
        self.durations: deque[float] = deque(maxlen=10)

    def generation_completed(
        self, generation: int, row: dict, total_generations: int
    ) -> None:
        now = time.monotonic()
        if self.last_generation is None or generation <= self.last_generation:
            self.run_index += 1
            self.run_started = now
            self.durations.clear()
        else:
            self.durations.append(max(0.0, now - self.last_update))
        self.last_generation = generation
        self.last_update = now
        remaining = max(0, total_generations - generation - 1)
        average = (
            sum(self.durations) / len(self.durations)
            if self.durations
            else 0.0
        )
        self._write(
            {
                "state": "running",
                "run_index": self.run_index,
                "generation": generation + 1,
                "generations": total_generations,
                "completed_generations": generation + 1,
                "best_fitness": float(row.get("best_ever", row.get("best", 0.0))),
                "mean_fitness": float(row.get("mean", 0.0)),
                "diversity": float(row.get("diversity", 0.0)),
                "elapsed_seconds": now - self.started,
                "run_elapsed_seconds": now - self.run_started,
                "eta_seconds": average * remaining if self.durations else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def finish(self, state: str) -> None:
        payload = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {}
        )
        payload["state"] = state
        payload["elapsed_seconds"] = time.monotonic() - self.started
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if state == "completed":
            payload["eta_seconds"] = 0
        self._write(payload)

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.025 * (2**attempt))
        finally:
            if temporary.exists():
                temporary.unlink()


ProgressCallback = Callable[[int, dict, int], None]
StopRequested = Callable[[], bool]
