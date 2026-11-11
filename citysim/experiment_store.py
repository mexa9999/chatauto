"""Catalog and portable archives for dashboard runs and imported datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
from typing import Iterator
import uuid
import zipfile

_ARCHIVE_VERSION = 1
MAX_ARCHIVE_BYTES = 20 * 1024**3
_MAX_UNPACKED_BYTES = 50 * 1024**3
_MAX_ARCHIVE_FILES = 100_000
_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
_NAME_RE = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]{1,100}$")


class ArchiveError(ValueError):
    """An uploaded dataset is not a valid CitySim archive."""


class ImportConflict(ValueError):
    def __init__(self, conflicts: dict[str, dict]):
        self.conflicts = conflicts
        super().__init__("A dataset with this name or archive already exists")


class ExperimentStore:
    def __init__(self, root: str | Path, default_history_dir: str | Path):
        self.root = Path(root).resolve()
        self.database = self.root / "catalog.sqlite3"
        self.datasets = self.root / "datasets"
        self.pending = self.root / "pending"
        self.default_history_dir = Path(default_history_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._remove_expired_previews()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    request_json TEXT NOT NULL,
                    directory TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    mode TEXT NOT NULL CHECK (mode IN ('light', 'full')),
                    archive_hash TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    directory TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS import_previews (
                    token TEXT PRIMARY KEY,
                    archive_path TEXT NOT NULL,
                    archive_hash TEXT NOT NULL,
                    requested_name TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            for table in ("runs", "datasets"):
                columns = {
                    row["name"]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                }
                if "deleted_at" not in columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN deleted_at TEXT")
            connection.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES ('source_id', ?)",
                (uuid.uuid4().hex,),
            )

    def active_history_dir(self) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = 'active_history_dir'"
            ).fetchone()
        return Path(row["value"]).resolve() if row else self.default_history_dir

    def register_run(
        self, run_id: str, kind: str, request: dict, directory: str | Path
    ) -> None:
        safe_request = {
            key: value
            for key, value in request.items()
            if key not in {"history_dir", "created_at"}
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, kind, state, created_at, request_json, directory
                ) VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    kind,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(safe_request, sort_keys=True),
                    str(Path(directory).resolve()),
                ),
            )

    def update_run(self, run_id: str, state: str) -> None:
        completed_at = (
            datetime.now(timezone.utc).isoformat()
            if state in {"completed", "stopped", "failed"}
            else None
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET state = ?, completed_at = COALESCE(?, completed_at) "
                "WHERE run_id = ?",
                (state, completed_at, run_id),
            )

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, kind, state, created_at, completed_at, request_json
                FROM runs WHERE deleted_at IS NULL ORDER BY created_at DESC
                """
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "run_id": row["run_id"],
                    "kind": row["kind"],
                    "state": row["state"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                    "request": json.loads(row["request_json"]),
                }
            )
        return result

    def analytics_sources(self) -> list[dict]:
        with self._connect() as connection:
            runs = connection.execute(
                """
                SELECT run_id, kind, state, created_at, directory
                FROM runs WHERE deleted_at IS NULL ORDER BY created_at DESC
                """
            ).fetchall()
            datasets = connection.execute(
                """
                SELECT dataset_id, name, created_at, manifest_json, directory
                FROM datasets WHERE deleted_at IS NULL ORDER BY created_at DESC
                """
            ).fetchall()
        sources = [
            {
                "source_id": row["run_id"],
                "source_type": "run",
                "label": f"{row['kind']} · {row['run_id'][:8]}",
                "kind": row["kind"],
                "state": row["state"],
                "created_at": row["created_at"],
                "available": _has_analytics(Path(row["directory"])),
            }
            for row in runs
        ]
        for row in datasets:
            run = json.loads(row["manifest_json"]).get("run", {})
            run_directory = Path(row["directory"]) / "run"
            sources.append(
                {
                    "source_id": row["dataset_id"],
                    "source_type": "dataset",
                    "label": row["name"],
                    "kind": run.get("kind", "imported"),
                    "state": run.get("state", "imported"),
                    "created_at": row["created_at"],
                    "available": _has_analytics(run_directory),
                }
            )
        return sources

    def analytics_directory(self, source_type: str, source_id: str) -> Path | None:
        if source_type == "run":
            query = "SELECT directory FROM runs WHERE run_id = ?"
        elif source_type == "dataset":
            query = "SELECT directory FROM datasets WHERE dataset_id = ?"
        else:
            raise ValueError(f"Unknown analytics source type: {source_type}")
        with self._connect() as connection:
            row = connection.execute(query, (source_id,)).fetchone()
        if row is None:
            return None
        directory = Path(row["directory"])
        return directory if source_type == "run" else directory / "run"

    def list_datasets(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT dataset_id, name, mode, source_id, created_at, manifest_json
                FROM datasets WHERE deleted_at IS NULL ORDER BY created_at DESC
                """
            ).fetchall()
            active = connection.execute(
                "SELECT value FROM settings WHERE key = 'active_history_dir'"
            ).fetchone()
        active_path = Path(active["value"]).resolve() if active else None
        datasets = []
        for row in rows:
            manifest = json.loads(row["manifest_json"])
            datasets.append(
                {
                    "dataset_id": row["dataset_id"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "source_id": row["source_id"],
                    "created_at": row["created_at"],
                    "run": manifest.get("run", {}),
                    "active": (
                        row["mode"] == "full"
                        and active_path
                        == (self.datasets / row["dataset_id"] / "history").resolve()
                    ),
                }
            )
        return datasets

    def trash(self) -> list[dict]:
        with self._connect() as connection:
            runs = connection.execute(
                """
                SELECT run_id AS id, kind AS name, 'run' AS entity_type, deleted_at
                FROM runs WHERE deleted_at IS NOT NULL
                """
            ).fetchall()
            datasets = connection.execute(
                """
                SELECT dataset_id AS id, name, 'dataset' AS entity_type, deleted_at
                FROM datasets WHERE deleted_at IS NOT NULL
                """
            ).fetchall()
        return sorted(
            [dict(row) for row in (*runs, *datasets)],
            key=lambda row: row["deleted_at"],
            reverse=True,
        )

    def trash_entities(self, entity_type: str, ids: list[str]) -> int:
        if entity_type not in {"run", "dataset"}:
            raise ValueError("entity_type must be run or dataset")
        if not ids:
            raise ValueError("select at least one item")
        placeholders = ",".join("?" for _ in ids)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            if entity_type == "run":
                rows = connection.execute(
                    f"SELECT run_id, state FROM runs WHERE run_id IN ({placeholders})",
                    ids,
                ).fetchall()
                if len(rows) != len(set(ids)):
                    raise ValueError("one or more runs do not exist")
                if any(row["state"] in {"running", "stopping"} for row in rows):
                    raise ValueError("running tasks cannot be moved to the trash")
                cursor = connection.execute(
                    f"UPDATE runs SET deleted_at = ? WHERE run_id IN ({placeholders}) "
                    "AND deleted_at IS NULL",
                    (now, *ids),
                )
            else:
                rows = connection.execute(
                    f"SELECT dataset_id, directory FROM datasets "
                    f"WHERE dataset_id IN ({placeholders})",
                    ids,
                ).fetchall()
                if len(rows) != len(set(ids)):
                    raise ValueError("one or more datasets do not exist")
                active_path = self.active_history_dir()
                if any(
                    active_path == (Path(row["directory"]) / "history").resolve()
                    for row in rows
                ):
                    raise ValueError("switch away from the active dataset before deleting it")
                cursor = connection.execute(
                    f"UPDATE datasets SET deleted_at = ? "
                    f"WHERE dataset_id IN ({placeholders}) AND deleted_at IS NULL",
                    (now, *ids),
                )
            return cursor.rowcount

    def restore_entities(self, entity_type: str, ids: list[str]) -> int:
        if entity_type not in {"run", "dataset"}:
            raise ValueError("entity_type must be run or dataset")
        if not ids:
            raise ValueError("select at least one item")
        table, key = (
            ("runs", "run_id") if entity_type == "run" else ("datasets", "dataset_id")
        )
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE {table} SET deleted_at = NULL "
                f"WHERE {key} IN ({placeholders}) AND deleted_at IS NOT NULL",
                ids,
            )
            if cursor.rowcount != len(set(ids)):
                raise ValueError("one or more items are not in the trash")
            return cursor.rowcount

    def purge_entities(self, entity_type: str, ids: list[str]) -> int:
        if entity_type not in {"run", "dataset"}:
            raise ValueError("entity_type must be run or dataset")
        if not ids:
            raise ValueError("select at least one item")
        table, key, root = (
            ("runs", "run_id", self.root.parent / "jobs")
            if entity_type == "run"
            else ("datasets", "dataset_id", self.datasets)
        )
        placeholders = ",".join("?" for _ in ids)
        state_column = ", state" if entity_type == "run" else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {key}, directory{state_column} FROM {table} "
                f"WHERE {key} IN ({placeholders}) AND deleted_at IS NOT NULL",
                ids,
            ).fetchall()
            if len(rows) != len(set(ids)):
                raise ValueError("one or more items are not in the trash")
            if entity_type == "run" and any(row["state"] in {"running", "stopping"} for row in rows):
                raise ValueError("running tasks cannot be permanently deleted")
            if entity_type == "dataset":
                active_path = self.active_history_dir()
                if any(
                    active_path == (Path(row["directory"]) / "history").resolve()
                    for row in rows
                ):
                    raise ValueError("switch away from the active dataset before deleting it")
            paths = [Path(row["directory"]).resolve() for row in rows]
            root = root.resolve()
            if any(not path.is_relative_to(root) for path in paths):
                raise ValueError("refusing to delete data outside the managed storage root")
            for path in paths:
                if path.exists():
                    shutil.rmtree(path)
            connection.execute(
                f"DELETE FROM {table} WHERE {key} IN ({placeholders})", ids
            )
        return len(rows)

    def dataset_details(self, dataset_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name, mode, manifest_json, directory "
                "FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Unknown dataset")
        directory = Path(row["directory"])
        summaries = []
        run_directory = directory / "run"
        if run_directory.is_dir():
            for path in sorted(run_directory.rglob("*summary.json")):
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError(
                        f"Summary file exceeds 2 MiB: {path.name}"
                    )
                summaries.append(
                    {
                        "file": path.relative_to(run_directory).as_posix(),
                        "data": json.loads(path.read_text(encoding="utf-8")),
                    }
                )
        return {
            "dataset_id": dataset_id,
            "name": row["name"],
            "mode": row["mode"],
            "run": json.loads(row["manifest_json"]).get("run", {}),
            "summaries": summaries,
        }

    def create_export(
        self, run_id: str, mode: str, history_dir: str | Path, destination: Path
    ) -> Path:
        if mode not in {"light", "full"}:
            raise ValueError("archive mode must be light or full")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            source_row = connection.execute(
                "SELECT value FROM settings WHERE key = 'source_id'"
            ).fetchone()
        if row is None:
            raise ValueError("Unknown run")
        if row["state"] not in {"completed", "stopped"}:
            raise ValueError("Only completed or stopped runs can be exported")

        run_directory = Path(row["directory"])
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Run output is missing: {run_directory}")
        history_root = Path(history_dir).resolve()
        files: list[tuple[str, Path]] = []
        for path in sorted(run_directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Run output contains a symbolic link: {path}")
            if path.is_file():
                files.append((path.relative_to(run_directory).as_posix(), path))

        with tempfile.TemporaryDirectory(
            prefix="citysim-export-", dir=self.pending
        ) as temporary:
            work = Path(temporary)
            archive_files: list[tuple[str, Path]] = []
            for relative, path in files:
                archive_files.append((f"run/{relative}", path))
            if mode == "full":
                database = history_root / "history.sqlite3"
                if not database.is_file():
                    raise FileNotFoundError(f"Training history is missing: {database}")
                snapshot = work / "history.sqlite3"
                with closing(sqlite3.connect(database)) as source, closing(
                    sqlite3.connect(snapshot)
                ) as backup:
                    source.backup(backup)
                archive_files.append(("history/history.sqlite3", snapshot))
                objects = history_root / "objects"
                if objects.exists():
                    for path in sorted(objects.glob("*.npz")):
                        if path.is_symlink() or not path.is_file():
                            raise ValueError(f"Invalid checkpoint object: {path}")
                        if not _HASH_RE.fullmatch(path.stem):
                            continue
                        archive_files.append(
                            (f"history/objects/{path.name}", path)
                        )

            manifest = {
                "format": "citysim-experiment",
                "version": _ARCHIVE_VERSION,
                "mode": mode,
                "source_id": source_row["value"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run": {
                    "run_id": run_id,
                    "kind": row["kind"],
                    "state": row["state"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                    "parameters": json.loads(row["request_json"]),
                },
                "files": [
                    {
                        "path": archive_path,
                        "size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for archive_path, path in archive_files
                ],
            }
            with zipfile.ZipFile(
                destination, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                )
                for archive_path, path in archive_files:
                    archive.write(path, archive_path)
        return destination

    def stage_import(self, archive_path: Path, name: str) -> dict:
        _validate_name(name)
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ArchiveError("Archive exceeds the 20 GiB upload limit")
        archive_hash = _sha256_file(archive_path)
        manifest = _read_manifest(archive_path)
        token = uuid.uuid4().hex
        stored_path = self.pending / f"{token}.zip"
        os.replace(archive_path, stored_path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO import_previews (
                    token, archive_path, archive_hash, requested_name,
                    manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, strftime('%s', 'now'))
                """,
                (
                    token,
                    str(stored_path),
                    archive_hash,
                    name.strip(),
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                ),
            )
            conflicts = self._conflicts(connection, name.strip(), archive_hash)
        return {
            "token": token,
            "manifest": manifest,
            "name": name.strip(),
            "conflicts": conflicts,
        }

    def commit_import(
        self, token: str, action: str | None = None, name: str | None = None
    ) -> dict:
        with self._connect() as connection:
            preview = connection.execute(
                "SELECT * FROM import_previews WHERE token = ?", (token,)
            ).fetchone()
            if preview is None:
                raise ValueError("Import preview expired; upload the archive again")
            conflicts = self._conflicts(
                connection, preview["requested_name"], preview["archive_hash"]
            )
        if conflicts and action is None:
            raise ImportConflict(conflicts)
        if action not in {None, "skip", "rename", "replace"}:
            raise ValueError("Unknown import conflict action")
        if conflicts and action == "skip":
            self._discard_preview(token, Path(preview["archive_path"]))
            return {"status": "skipped", "conflicts": conflicts}

        target_name = preview["requested_name"]
        if action == "rename":
            if name is None:
                raise ValueError("A new name is required when renaming a dataset")
            _validate_name(name)
            target_name = name.strip()
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT dataset_id, name, mode FROM datasets WHERE name = ?",
                    (target_name,),
                ).fetchone()
                conflicts = {"name": dict(row)} if row else {}
            if conflicts:
                raise ImportConflict(conflicts)
        elif action == "replace":
            if not conflicts:
                raise ValueError("There is no conflicting dataset to replace")
            conflict_ids = {
                item["dataset_id"] for item in conflicts.values()
            }
            if len(conflict_ids) > 1:
                raise ImportConflict(conflicts)
            target_name = (
                conflicts["name"]["name"]
                if "name" in conflicts
                else conflicts["archive"]["name"]
            )

        dataset_id = uuid.uuid4().hex
        staged = self.datasets / f".{dataset_id}.staging"
        final = self.datasets / dataset_id
        backup: Path | None = None
        archive_path = Path(preview["archive_path"])
        manifest = json.loads(preview["manifest_json"])
        committed = False
        try:
            _extract_verified(archive_path, staged, manifest)
            if manifest["mode"] == "full":
                _verify_history(staged / "history")
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT dataset_id, directory FROM datasets WHERE name = ?",
                    (target_name,),
                ).fetchone()
                if existing is None and action == "replace":
                    existing = connection.execute(
                        "SELECT dataset_id, directory FROM datasets "
                        "WHERE archive_hash = ?",
                        (preview["archive_hash"],),
                    ).fetchone()
                    if existing is not None:
                        target_name = connection.execute(
                            "SELECT name FROM datasets WHERE dataset_id = ?",
                            (existing["dataset_id"],),
                        ).fetchone()["name"]
            if existing is not None:
                if action != "replace":
                    raise ImportConflict({"name": dict(existing)})
                active_path = self.active_history_dir()
                old_path = (self.datasets / existing["dataset_id"] / "history").resolve()
                if active_path == old_path:
                    raise ValueError("Switch away from the active dataset before replacing it")
                backup = self.datasets / f".{existing['dataset_id']}.replaced"
                if backup.exists():
                    raise FileExistsError(f"Replacement backup already exists: {backup}")
                os.replace(self.datasets / existing["dataset_id"], backup)
            os.replace(staged, final)
            with self._connect() as connection:
                if existing is not None:
                    connection.execute(
                        "DELETE FROM datasets WHERE dataset_id = ?",
                        (existing["dataset_id"],),
                    )
                connection.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id, name, mode, archive_hash, source_id,
                        created_at, manifest_json, directory
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        target_name,
                        manifest["mode"],
                        preview["archive_hash"],
                        manifest["source_id"],
                        manifest["created_at"],
                        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                        str(final),
                    ),
                )
                connection.execute(
                    "DELETE FROM import_previews WHERE token = ?", (token,)
                )
            committed = True
        finally:
            if not committed:
                if final.exists():
                    shutil.rmtree(final)
                if backup is not None and backup.exists():
                    original_id = backup.name.removeprefix(".").removesuffix(".replaced")
                    os.replace(backup, self.datasets / original_id)
                if staged.exists():
                    shutil.rmtree(staged)
        cleanup_warnings = []
        try:
            archive_path.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_warnings.append(f"Import archive could not be removed: {exc}")
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                cleanup_warnings.append(f"Replacement backup could not be removed: {exc}")
        result = {
            "status": "imported",
            "dataset_id": dataset_id,
            "name": target_name,
            "mode": manifest["mode"],
        }
        if cleanup_warnings:
            result["warning"] = " ".join(cleanup_warnings)
        return result

    def activate_dataset(self, dataset_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT mode, directory FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Unknown dataset")
            if row["mode"] != "full":
                raise ValueError("Only full archives contain training history")
            history = (Path(row["directory"]) / "history").resolve()
            connection.execute(
                """
                INSERT INTO settings (key, value) VALUES ('active_history_dir', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(history),),
            )
        return history

    def deactivate_dataset(self) -> Path:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM settings WHERE key = 'active_history_dir'"
            )
        return self.default_history_dir

    def _conflicts(
        self, connection: sqlite3.Connection, name: str, archive_hash: str
    ) -> dict[str, dict]:
        by_name = connection.execute(
            "SELECT dataset_id, name, mode FROM datasets WHERE name = ?", (name,)
        ).fetchone()
        by_hash = connection.execute(
            "SELECT dataset_id, name, mode FROM datasets WHERE archive_hash = ?",
            (archive_hash,),
        ).fetchone()
        result = {}
        if by_name:
            result["name"] = dict(by_name)
        if by_hash:
            result["archive"] = dict(by_hash)
        return result

    def _discard_preview(self, token: str, archive_path: Path) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM import_previews WHERE token = ?", (token,)
            )
        archive_path.unlink(missing_ok=True)

    def _remove_expired_previews(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT token, archive_path FROM import_previews "
                "WHERE CAST(strftime('%s', 'now') AS REAL) - created_at > 86400"
            ).fetchall()
            connection.execute(
                "DELETE FROM import_previews "
                "WHERE CAST(strftime('%s', 'now') AS REAL) - created_at > 86400"
            )
        for row in rows:
            Path(row["archive_path"]).unlink(missing_ok=True)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name.strip()):
        raise ValueError("Dataset name must be 1-100 characters without path characters")


def _has_analytics(directory: Path) -> bool:
    return any(
        (directory / name).is_file()
        for name in (
            "analytics_summary.json",
            "training_metrics.json",
            "training_progress.ndjson",
            "history.csv",
            "metrics.json",
        )
    ) or any(directory.glob("*_history.csv"))


def _read_manifest(archive_path: Path) -> dict:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_FILES + 1:
                raise ArchiveError("Archive contains too many files")
            info_names = [info.filename for info in infos]
            if len(info_names) != len(set(info_names)):
                raise ArchiveError("Archive contains duplicate file paths")
            for info in infos:
                _validate_archive_path(info)
            if "manifest.json" not in info_names:
                raise ArchiveError("Archive has no manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("format") != "citysim-experiment"
                or manifest.get("version") != _ARCHIVE_VERSION
                or manifest.get("mode") not in {"light", "full"}
                or not isinstance(manifest.get("source_id"), str)
                or not manifest["source_id"]
                or not isinstance(manifest.get("run"), dict)
            ):
                raise ArchiveError("Unsupported CitySim archive format or version")
            if not isinstance(manifest.get("files"), list):
                raise ArchiveError("Archive manifest has no file list")
            expected = {"manifest.json"}
            total_size = 0
            for entry in manifest["files"]:
                if not isinstance(entry, dict):
                    raise ArchiveError("Archive manifest contains an invalid entry")
                member = entry.get("path")
                if not isinstance(member, str):
                    raise ArchiveError("Archive manifest contains an invalid path")
                _validate_relative_path(member)
                if not member.startswith(("run/", "history/")):
                    raise ArchiveError(f"Unexpected archive path: {member}")
                if member in expected:
                    raise ArchiveError(f"Duplicate manifest path: {member}")
                expected.add(member)
                if (
                    not isinstance(entry.get("size"), int)
                    or entry["size"] < 0
                    or not _HASH_RE.fullmatch(str(entry.get("sha256", "")))
                ):
                    raise ArchiveError(f"Invalid archive metadata for {member}")
                total_size += entry["size"]
                if total_size > _MAX_UNPACKED_BYTES:
                    raise ArchiveError("Archive expands beyond the supported size")
                info = archive.getinfo(member)
                if info.file_size != entry["size"]:
                    raise ArchiveError(f"File size mismatch for {member}")
            if set(info_names) != expected:
                raise ArchiveError("Archive members do not match the manifest")
            if archive.testzip() is not None:
                raise ArchiveError("Archive has a corrupt ZIP member")
            actual_full = "history/history.sqlite3" in expected
            if (manifest["mode"] == "full") != actual_full:
                raise ArchiveError("Archive mode does not match its contents")
            return manifest
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        raise ArchiveError(f"Cannot read CitySim archive: {exc}") from exc


def _validate_archive_path(info: zipfile.ZipInfo) -> None:
    name = info.filename
    _validate_relative_path(name.rstrip("/"))
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ArchiveError(f"Symbolic links are not allowed: {name}")


def _validate_relative_path(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ArchiveError(f"Unsafe archive path: {name!r}")


def _extract_verified(archive_path: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    expected = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        for name, entry in expected.items():
            _validate_relative_path(name)
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ArchiveError(f"File is larger than declared: {name}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ArchiveError(f"File hash mismatch: {name}")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _verify_history(history_dir: Path) -> None:
    database = history_dir / "history.sqlite3"
    if not database.is_file():
        raise ArchiveError("Full archive has no training history database")
    try:
        with closing(
            sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        ) as db:
            check = db.execute("PRAGMA integrity_check").fetchone()
            if check is None or check[0] != "ok":
                raise ArchiveError("Imported history database failed integrity_check")
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"commits", "branches", "experiment_cache"} <= tables:
                raise ArchiveError("Imported history database has an unknown schema")
            required_columns = {
                "commits": {
                    "commit_id", "parent_id", "checkpoint_hash", "generation",
                    "complete", "best_fitness", "evolution_json", "city_json",
                    "branch_name", "created_at",
                },
                "branches": {"name", "head_commit", "created_at"},
                "settings": {"key", "value"},
                "experiment_cache": {
                    "configuration_digest", "checkpoint_hash", "generation",
                    "complete", "evolution_json", "city_json", "updated_at",
                },
            }
            for table, required in required_columns.items():
                columns = {
                    row[1] for row in db.execute(f"PRAGMA table_info({table})")
                }
                if not required <= columns:
                    raise ArchiveError(
                        f"Imported history table {table!r} has an unknown schema"
                    )
            if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ArchiveError("Imported history database has foreign key errors")
    except sqlite3.DatabaseError as exc:
        raise ArchiveError(f"Imported history database is invalid: {exc}") from exc
    objects = history_dir / "objects"
    if objects.exists():
        for path in objects.glob("*.npz"):
            if not _HASH_RE.fullmatch(path.stem) or _sha256_file(path) != path.stem:
                raise ArchiveError(f"Imported checkpoint hash mismatch: {path.name}")
    from .training_history import TrainingHistoryStore

    report = TrainingHistoryStore(history_dir).verify()
    if not report["ok"]:
        issues = "; ".join(report["issues"][:5])
        raise ArchiveError(f"Imported history failed verification: {issues}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
