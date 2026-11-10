"""SQLite-backed immutable checkpoints and Git-like training branches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .checkpoint_io import read_checkpoint_metadata
from .config import CityConfig, EvolutionConfig, FitnessWeights, Regulation, Requirements

_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
DEFAULT_HISTORY_DIR = Path("runs/.citysim-history")


class TrainingHistoryStore:
    """Persist training snapshots, branch refs, and their ancestry."""

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.database = self.root / "history.sqlite3"
        self.objects = self.root / "objects"
        self.work = self.root / "work"
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS commits (
                    commit_id TEXT PRIMARY KEY,
                    parent_id TEXT REFERENCES commits(commit_id),
                    checkpoint_hash TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
                    best_fitness REAL,
                    evolution_json TEXT NOT NULL,
                    city_json TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS branches (
                    name TEXT PRIMARY KEY,
                    head_commit TEXT REFERENCES commits(commit_id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_cache (
                    configuration_digest TEXT PRIMARY KEY,
                    checkpoint_hash TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
                    evolution_json TEXT NOT NULL,
                    city_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS commits_parent_idx
                    ON commits(parent_id);
                CREATE INDEX IF NOT EXISTS commits_branch_generation_idx
                    ON commits(branch_name, generation);
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "INSERT OR IGNORE INTO branches (name, head_commit, created_at) "
                "VALUES ('main', NULL, ?)",
                (now,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES ('active_branch', 'main')"
            )

    @staticmethod
    def _validate_branch_name(name: str) -> None:
        if not _BRANCH_RE.fullmatch(name):
            raise ValueError(
                "branch names must start with a letter or digit and contain only "
                "letters, digits, '.', '_' or '-' (up to 64 characters)"
            )

    def branch_names(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.name, b.head_commit, c.generation, c.commit_id
                FROM branches AS b
                LEFT JOIN commits AS c ON c.commit_id = b.head_commit
                ORDER BY b.name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def active_branch(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = 'active_branch'"
            ).fetchone()
        if row is None:
            raise RuntimeError("training history has no active branch setting")
        return str(row["value"])

    def checkout(self, name: str) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM branches WHERE name = ?", (name,)
            ).fetchone()
            if exists is None:
                raise ValueError(f"unknown branch: {name}")
            connection.execute(
                "UPDATE settings SET value = ? WHERE key = 'active_branch'",
                (name,),
            )

    def config_for_branch(self, branch: str) -> tuple[CityConfig, EvolutionConfig]:
        commit_id = self.resolve_ref(branch)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT evolution_json, city_json FROM commits WHERE commit_id = ?",
                (commit_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"missing history commit: {commit_id}")
        evolution = EvolutionConfig(**json.loads(row["evolution_json"]))
        city_values = json.loads(row["city_json"])
        city_values["requirements"] = Requirements(**city_values["requirements"])
        city_values["regulation"] = Regulation(**city_values["regulation"])
        city_values["fitness"] = FitnessWeights(**city_values["fitness"])
        return CityConfig(**city_values), evolution

    def create_branch(self, name: str, from_ref: str | None = None) -> str | None:
        self._validate_branch_name(name)
        if from_ref is None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT head_commit FROM branches WHERE name = ?",
                    (self.active_branch(),),
                ).fetchone()
            head = row["head_commit"] if row is not None else None
        else:
            head = self.resolve_ref(from_ref)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO branches (name, head_commit, created_at) VALUES (?, ?, ?)",
                    (name, head, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"branch already exists: {name}") from exc
        return head

    def reset_branch(self, name: str, to_ref: str) -> str:
        head = self.resolve_ref(to_ref)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE branches SET head_commit = ? WHERE name = ?",
                (head, name),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"unknown branch: {name}")
        return head

    def resolve_ref(self, ref: str) -> str:
        if "@" in ref:
            branch, generation_text = ref.rsplit("@", 1)
            try:
                generation = int(generation_text)
            except ValueError as exc:
                raise ValueError(f"invalid generation in reference: {ref}") from exc
            with self._connect() as connection:
                row = connection.execute(
                    """
                    WITH RECURSIVE ancestry(commit_id, parent_id) AS (
                        SELECT c.commit_id, c.parent_id
                        FROM commits AS c
                        JOIN branches AS b ON b.head_commit = c.commit_id
                        WHERE b.name = ?
                        UNION ALL
                        SELECT parent.commit_id, parent.parent_id
                        FROM commits AS parent
                        JOIN ancestry ON ancestry.parent_id = parent.commit_id
                    )
                    SELECT commit_id
                    FROM commits
                    WHERE generation = ?
                    AND commit_id IN (SELECT commit_id FROM ancestry)
                    LIMIT 1
                    """,
                    (branch, generation),
                ).fetchone()
            if row is None:
                raise ValueError(f"no snapshot at {ref}")
            return str(row["commit_id"])

        with self._connect() as connection:
            row = connection.execute(
                "SELECT head_commit FROM branches WHERE name = ?", (ref,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown branch: {ref}")
        if row["head_commit"] is None:
            raise ValueError(f"branch has no snapshots yet: {ref}")
        return str(row["head_commit"])

    def history(self, ref: str) -> list[dict]:
        commit_id = self.resolve_ref(ref)
        rows: list[dict] = []
        with self._connect() as connection:
            while commit_id:
                row = connection.execute(
                    "SELECT * FROM commits WHERE commit_id = ?", (commit_id,)
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"missing history commit: {commit_id}")
                item = dict(row)
                item["evolution"] = json.loads(item.pop("evolution_json"))
                item["city"] = json.loads(item.pop("city_json"))
                rows.append(item)
                commit_id = row["parent_id"]
        rows.reverse()
        return rows

    def checkpoint_path(self, branch: str) -> Path:
        commit_id = self.resolve_ref(branch)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT checkpoint_hash FROM commits WHERE commit_id = ?",
                (commit_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"missing history commit: {commit_id}")
        path = self.objects / f"{row['checkpoint_hash']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint object is missing: {path}")
        return path

    def restore_experiment_checkpoint(
        self, configuration_digest: str, destination: str | Path
    ) -> dict | None:
        if not _DIGEST_RE.fullmatch(configuration_digest):
            raise ValueError("invalid experiment configuration digest")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_cache WHERE configuration_digest = ?",
                (configuration_digest,),
            ).fetchone()
        if row is None:
            return None

        source = self.objects / f"{row['checkpoint_hash']}.npz"
        if not source.is_file():
            raise FileNotFoundError(
                f"experiment checkpoint object is missing: {source}"
            )
        if _sha256_file(source) != row["checkpoint_hash"]:
            raise ValueError(f"experiment checkpoint hash mismatch: {source}")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return dict(row)

    def record_experiment_checkpoint(
        self, configuration_digest: str, checkpoint_path: str | Path
    ) -> None:
        if not _DIGEST_RE.fullmatch(configuration_digest):
            raise ValueError("invalid experiment configuration digest")
        source = Path(checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"experiment checkpoint is missing: {source}")
        metadata = read_checkpoint_metadata(source)
        if metadata.get("configuration_digest") != configuration_digest:
            raise ValueError("experiment checkpoint configuration does not match key")

        checkpoint_hash = _sha256_file(source)
        object_path = self.objects / f"{checkpoint_hash}.npz"
        if object_path.exists():
            if _sha256_file(object_path) != checkpoint_hash:
                raise ValueError(
                    f"experiment checkpoint object hash mismatch: {object_path}"
                )
        else:
            temporary = object_path.with_name(
                f".{checkpoint_hash}.{uuid.uuid4().hex}.tmp"
            )
            try:
                shutil.copyfile(source, temporary)
                with temporary.open("rb+") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                if _sha256_file(temporary) != checkpoint_hash:
                    raise ValueError(
                        "experiment checkpoint changed while being stored"
                    )
                os.replace(temporary, object_path)
            finally:
                if temporary.exists():
                    temporary.unlink()

        generation = int(metadata["next_generation"])
        complete = bool(metadata["complete"])
        evolution_json = json.dumps(metadata["evolution_config"], sort_keys=True)
        city_json = json.dumps(metadata["city_config"], sort_keys=True)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_cache (
                    configuration_digest, checkpoint_hash, generation, complete,
                    evolution_json, city_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(configuration_digest) DO UPDATE SET
                    checkpoint_hash = excluded.checkpoint_hash,
                    generation = excluded.generation,
                    complete = excluded.complete,
                    evolution_json = excluded.evolution_json,
                    city_json = excluded.city_json,
                    updated_at = excluded.updated_at
                WHERE excluded.generation > experiment_cache.generation
                   OR (
                       excluded.generation = experiment_cache.generation
                       AND excluded.complete >= experiment_cache.complete
                   )
                """,
                (
                    configuration_digest,
                    checkpoint_hash,
                    generation,
                    int(complete),
                    evolution_json,
                    city_json,
                    now,
                ),
            )

    def verify(self) -> dict:
        issues: list[str] = []
        with self._connect() as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            commits = connection.execute(
                """
                SELECT commit_id, checkpoint_hash, generation, complete,
                       evolution_json, city_json
                FROM commits
                """
            ).fetchall()
            cached_checkpoints = connection.execute(
                "SELECT * FROM experiment_cache"
            ).fetchall()
        for row in integrity:
            if row[0] != "ok":
                issues.append(f"SQLite integrity check: {row[0]}")
        for row in foreign_keys:
            issues.append(f"foreign key violation in {row[0]} row {row[1]}")

        for row in commits:
            digest = str(row["checkpoint_hash"])
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                issues.append(f"{row['commit_id']}: invalid checkpoint hash {digest!r}")
                continue
            path = self.objects / f"{digest}.npz"
            if not path.is_file():
                issues.append(f"{row['commit_id']}: missing checkpoint {digest}")
                continue
            if _sha256_file(path) != digest:
                issues.append(f"{row['commit_id']}: checkpoint hash mismatch {digest}")
                continue
            try:
                metadata = read_checkpoint_metadata(path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                issues.append(f"{row['commit_id']}: invalid checkpoint: {exc}")
                continue
            if not isinstance(metadata, dict):
                issues.append(f"{row['commit_id']}: checkpoint metadata is not an object")
                continue
            try:
                checkpoint_generation = int(metadata.get("next_generation", -1))
            except (TypeError, ValueError):
                issues.append(f"{row['commit_id']}: invalid checkpoint generation")
                continue
            if checkpoint_generation != int(row["generation"]):
                issues.append(f"{row['commit_id']}: checkpoint generation mismatch")
            if bool(metadata.get("complete")) != bool(row["complete"]):
                issues.append(f"{row['commit_id']}: checkpoint completion mismatch")
            if metadata.get("evolution_config") != json.loads(row["evolution_json"]):
                issues.append(f"{row['commit_id']}: evolution configuration mismatch")
            if metadata.get("city_config") != json.loads(row["city_json"]):
                issues.append(f"{row['commit_id']}: city configuration mismatch")

        for row in cached_checkpoints:
            key = str(row["configuration_digest"])
            digest = str(row["checkpoint_hash"])
            if not _DIGEST_RE.fullmatch(key) or not _DIGEST_RE.fullmatch(digest):
                issues.append(f"experiment cache {key}: invalid digest")
                continue
            path = self.objects / f"{digest}.npz"
            if not path.is_file():
                issues.append(f"experiment cache {key}: missing checkpoint {digest}")
                continue
            if _sha256_file(path) != digest:
                issues.append(f"experiment cache {key}: checkpoint hash mismatch")
                continue
            try:
                metadata = read_checkpoint_metadata(path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                issues.append(f"experiment cache {key}: invalid checkpoint: {exc}")
                continue
            if not isinstance(metadata, dict):
                issues.append(f"experiment cache {key}: checkpoint metadata is not an object")
                continue
            if metadata.get("configuration_digest") != key:
                issues.append(f"experiment cache {key}: configuration digest mismatch")
            try:
                checkpoint_generation = int(metadata.get("next_generation", -1))
            except (TypeError, ValueError):
                issues.append(f"experiment cache {key}: invalid checkpoint generation")
                continue
            if checkpoint_generation != int(row["generation"]):
                issues.append(f"experiment cache {key}: generation mismatch")
            if bool(metadata.get("complete")) != bool(row["complete"]):
                issues.append(f"experiment cache {key}: completion mismatch")
            if metadata.get("evolution_config") != json.loads(row["evolution_json"]):
                issues.append(f"experiment cache {key}: evolution configuration mismatch")
            if metadata.get("city_config") != json.loads(row["city_json"]):
                issues.append(f"experiment cache {key}: city configuration mismatch")

        plan = self._gc_plan()
        return {
            "ok": not issues,
            "issues": issues,
            "commit_count": len(commits),
            "experiment_checkpoint_count": len(cached_checkpoints),
            "unreachable_commit_count": len(plan["commit_ids"]),
            "orphan_object_count": len(plan["objects"]),
        }

    def gc_preview(self) -> dict:
        return self._gc_plan()

    def gc_apply(self, confirmation_token: str) -> dict:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            plan = self._gc_plan(connection)
            if plan["confirmation_token"] != confirmation_token:
                raise ValueError(
                    "history changed after the GC preview; generate a new preview"
                )
            pending = set(plan["commit_ids"])
            rows = connection.execute(
                "SELECT commit_id, parent_id FROM commits"
            ).fetchall()
            parents = {
                row["commit_id"]: row["parent_id"]
                for row in rows
                if row["commit_id"] in pending
            }
            while pending:
                referenced = {
                    parent for parent in parents.values() if parent in pending
                }
                leaves = pending - referenced
                if not leaves:
                    raise RuntimeError("unreachable commit graph contains a cycle")
                for commit_id in leaves:
                    connection.execute(
                        "DELETE FROM commits WHERE commit_id = ?", (commit_id,)
                    )
                pending -= leaves

        removed_objects = []
        for name in plan["objects"]:
            path = self.objects / name
            if path.exists():
                path.unlink()
                removed_objects.append(name)
        return {
            "deleted_commits": plan["commit_ids"],
            "deleted_objects": removed_objects,
            "deleted_bytes": plan["bytes"],
        }

    def _gc_plan(self, connection: sqlite3.Connection | None = None) -> dict:
        if connection is None:
            with self._connect() as owned_connection:
                return self._gc_plan(owned_connection)

        rows = connection.execute(
            "SELECT commit_id, parent_id, checkpoint_hash FROM commits"
        ).fetchall()
        commits = {row["commit_id"]: dict(row) for row in rows}
        heads = connection.execute(
            "SELECT head_commit FROM branches WHERE head_commit IS NOT NULL"
        ).fetchall()
        reachable: set[str] = set()
        pending = [row["head_commit"] for row in heads]
        while pending:
            commit_id = pending.pop()
            if commit_id in reachable or commit_id not in commits:
                continue
            reachable.add(commit_id)
            parent_id = commits[commit_id]["parent_id"]
            if parent_id is not None:
                pending.append(parent_id)

        unreachable = sorted(set(commits) - reachable)
        referenced_objects = {
            commits[commit_id]["checkpoint_hash"] for commit_id in reachable
        }
        cached_objects = connection.execute(
            "SELECT checkpoint_hash FROM experiment_cache"
        ).fetchall()
        referenced_objects.update(row["checkpoint_hash"] for row in cached_objects)
        orphan_objects = sorted(
            path.name
            for path in self.objects.glob("*.npz")
            if path.is_file()
            and re.fullmatch(r"[a-f0-9]{64}", path.stem)
            and path.stem not in referenced_objects
        )
        total_bytes = sum((self.objects / name).stat().st_size for name in orphan_objects)
        payload = json.dumps(
            {"commits": unreachable, "objects": orphan_objects},
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "commit_ids": unreachable,
            "objects": orphan_objects,
            "bytes": total_bytes,
            "confirmation_token": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    def work_path(self, branch: str) -> Path:
        self._validate_branch_name(branch)
        return self.work / f"{branch}.npz"

    def record_checkpoint(
        self,
        branch: str,
        checkpoint_path: str | Path,
        generation: int,
        complete: bool,
        evolution_config: EvolutionConfig,
        city_config: CityConfig,
        expected_parent: str | None,
    ) -> str:
        source = Path(checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint file is missing: {source}")
        metadata = read_checkpoint_metadata(source)
        if int(metadata["next_generation"]) != generation:
            raise ValueError("checkpoint generation does not match history record")

        digest = _sha256_file(source)
        object_path = self.objects / f"{digest}.npz"
        if object_path.exists():
            if _sha256_file(object_path) != digest:
                raise ValueError(f"checkpoint object hash mismatch: {object_path}")
        else:
            temporary = object_path.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(source, temporary)
                with temporary.open("rb+") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                if _sha256_file(temporary) != digest:
                    raise ValueError("checkpoint changed while being snapshotted")
                os.replace(temporary, object_path)
            finally:
                if temporary.exists():
                    temporary.unlink()

        commit_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            branch_row = connection.execute(
                "SELECT head_commit FROM branches WHERE name = ?", (branch,)
            ).fetchone()
            if branch_row is None:
                raise ValueError(f"unknown branch: {branch}")
            if branch_row["head_commit"] != expected_parent:
                raise RuntimeError(
                    f"branch {branch!r} advanced during training; "
                    "refusing to overwrite its history"
                )
            connection.execute(
                """
                INSERT INTO commits (
                    commit_id, parent_id, checkpoint_hash, generation, complete,
                    best_fitness, evolution_json, city_json, branch_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commit_id,
                    branch_row["head_commit"],
                    digest,
                    generation,
                    int(complete),
                    (
                        float(metadata["history"][-1]["best_ever"])
                        if metadata.get("history")
                        else None
                    ),
                    json.dumps(asdict(evolution_config), sort_keys=True),
                    json.dumps(asdict(city_config), sort_keys=True),
                    branch,
                    now,
                ),
            )
            connection.execute(
                "UPDATE branches SET head_commit = ? WHERE name = ?",
                (commit_id, branch),
            )
        return commit_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
