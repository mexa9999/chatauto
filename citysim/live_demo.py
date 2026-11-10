"""Compact state recording and control for dashboard demo runs."""

from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import uuid
import time
from typing import Any
import zlib

import numpy as np


_MAX_FRAME_BYTES = 8 * 1024 * 1024


def _encode_grid(grid: np.ndarray, dtype: str) -> str:
    packed = np.ascontiguousarray(grid, dtype=dtype).tobytes()
    return base64.b64encode(zlib.compress(packed)).decode("ascii")


def _decode_grid(encoded: str, dtype: str, height: int, width: int) -> list[list[int]]:
    try:
        packed = zlib.decompress(base64.b64decode(encoded, validate=True))
        values = np.frombuffer(packed, dtype=dtype)
        if values.size != height * width:
            raise ValueError("state grid has an invalid size")
    except (ValueError, zlib.error) as exc:
        raise ValueError(f"invalid compressed state grid: {exc}") from exc
    return values.reshape((height, width)).astype(np.int64).tolist()


class LiveDemoRecorder:
    def __init__(
        self,
        state_path: str | Path,
        control_path: str | Path,
        metadata: dict[str, Any],
    ):
        self.state_path = Path(state_path)
        self.control_path = Path(control_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.state_path.open("w", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._write({"kind": "metadata", **metadata})

    def _write(self, frame: dict[str, Any]) -> None:
        frame["seq"] = self._sequence
        self._sequence += 1
        self._stream.write(
            json.dumps(frame, separators=(",", ":"), allow_nan=False, default=_json_value)
            + "\n"
        )
        self._stream.flush()

    def record_state(
        self,
        env,
        kind: str,
        round_index: int,
        agent_id: int | None = None,
        action=None,
        accepted: bool | None = None,
    ) -> None:
        self._write(
            {
                "kind": kind,
                "round": round_index + 1,
                "agent_id": agent_id,
                "action": asdict(action) if action is not None else None,
                "accepted": accepted,
                "height": env.h,
                "width": env.w,
                "type_grid": _encode_grid(env.type_grid, "i1"),
                "quality_grid": _encode_grid(env.quality_grid, "i1"),
                "floors_grid": _encode_grid(
                    getattr(env, "floors_grid", np.where(env.type_grid > 0, 1, 0)),
                    "i1",
                ),
                "roof_quality_grid": _encode_grid(
                    getattr(
                        env,
                        "roof_quality_grid",
                        np.zeros_like(env.type_grid),
                    ),
                    "i1",
                ),
                "owner_grid": _encode_grid(env.owner_grid, "<i2"),
            }
        )

    def after_action(self, agent_id: int, action, accepted: bool, env) -> bool:
        self.record_state(
            env,
            "action",
            env.round,
            agent_id=agent_id,
            action=action,
            accepted=accepted,
        )
        return self.wait_until_active(delay=True)

    def wait_until_active(self, delay: bool = False) -> bool:
        remaining = self._read_control().get("speed_ms", 80) / 1000 if delay else 0
        last_checked = time.monotonic()
        while True:
            control = self._read_control()
            if (self.control_path.parent / "stop.request").exists():
                return False
            now = time.monotonic()
            if control.get("paused", False):
                time.sleep(0.1)
                last_checked = time.monotonic()
                continue
            if remaining > 0:
                remaining = max(0, remaining - (now - last_checked))
                last_checked = now
                time.sleep(min(0.05, remaining))
                continue
            return True

    def _read_control(self) -> dict[str, Any]:
        return read_control_file(self.control_path)

    def close(self) -> None:
        self._stream.close()


def read_control_file(path: str | Path) -> dict[str, Any]:
    control_path = Path(path)
    if not control_path.exists():
        return {"paused": False, "speed_ms": 80}
    value = json.loads(control_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("demo control file must contain a JSON object")
    return {"paused": False, "speed_ms": 80, **value}


def read_live_frames(
    path: str | Path,
    offset: int,
    max_bytes: int = _MAX_FRAME_BYTES,
) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.is_file():
        return {"frames": [], "next_offset": offset}
    size = state_path.stat().st_size
    start = min(offset, size)
    with state_path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(max_bytes)
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        if len(data) == max_bytes:
            raise ValueError("live state frame exceeds the read limit")
        return {"frames": [], "next_offset": start}
    complete_data = data[: last_newline + 1]
    frames = []
    for line in complete_data.splitlines():
        if not line:
            continue
        try:
            frame = json.loads(line)
            if "type_grid" in frame:
                height, width = frame["height"], frame["width"]
                for name, dtype in (
                    ("type_grid", "i1"),
                    ("quality_grid", "i1"),
                    ("floors_grid", "i1"),
                    ("roof_quality_grid", "i1"),
                    ("owner_grid", "<i2"),
                ):
                    if name in frame:
                        frame[name] = _decode_grid(
                            frame[name], dtype, height, width
                        )
            frames.append(frame)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid live state frame: {exc}") from exc
    return {
        "frames": frames,
        "next_offset": start + len(complete_data),
    }


def read_live_frame(path: str | Path, sequence: int) -> dict[str, Any]:
    state_path = Path(path)
    with state_path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid live state frame: {exc}") from exc
            if frame.get("seq") == sequence:
                if "type_grid" in frame:
                    height, width = frame["height"], frame["width"]
                    for name, dtype in (
                        ("type_grid", "i1"),
                        ("quality_grid", "i1"),
                        ("floors_grid", "i1"),
                        ("roof_quality_grid", "i1"),
                        ("owner_grid", "<i2"),
                    ):
                        if name in frame:
                            frame[name] = _decode_grid(
                                frame[name], dtype, height, width
                            )
                return frame
    raise ValueError(f"unknown live state frame: {sequence}")


def update_control_file(
    path: str | Path,
    paused: bool | None = None,
    speed_ms: int | None = None,
) -> dict[str, Any]:
    control_path = Path(path)
    current = read_control_file(control_path)
    if paused is not None:
        current["paused"] = paused
    if speed_ms is not None:
        current["speed_ms"] = speed_ms
    temporary = control_path.with_name(
        f".{control_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(current, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(temporary, control_path)
    finally:
        temporary.unlink(missing_ok=True)
    return current


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
