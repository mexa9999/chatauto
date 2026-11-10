"""Atomic, pickle-free storage for evolution checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def write_checkpoint(
    path: str | Path,
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_value,
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = _read_metadata(archive)
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "metadata"
        }
    return metadata, arrays


def read_checkpoint_metadata(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as archive:
        return _read_metadata(archive)


def _read_metadata(archive: np.lib.npyio.NpzFile) -> dict:
    return json.loads(archive["metadata"].tobytes().decode("utf-8"))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"unsupported JSON value in checkpoint: {type(value).__name__}")
