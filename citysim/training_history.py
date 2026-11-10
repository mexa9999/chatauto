"""SQLite-backed immutable checkpoints and Git-like training branches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import io
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

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
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    deleted_at TEXT
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
                CREATE TABLE IF NOT EXISTS genomes (
                    genome_id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL CHECK (source_kind IN ('checkpoint', 'import')),
                    commit_id TEXT REFERENCES commits(commit_id) ON DELETE CASCADE,
                    checkpoint_hash TEXT,
                    member_kind TEXT,
                    member_index INTEGER,
                    object_hash TEXT,
                    parent_genome_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT,
                    CHECK (
                        (source_kind = 'checkpoint' AND commit_id IS NOT NULL
                            AND checkpoint_hash IS NOT NULL AND member_kind IS NOT NULL
                            AND member_index IS NOT NULL AND object_hash IS NULL)
                        OR
                        (source_kind = 'import' AND commit_id IS NULL
                            AND checkpoint_hash IS NULL AND member_kind IS NULL
                            AND member_index IS NULL AND object_hash IS NOT NULL)
                    )
                );
                CREATE INDEX IF NOT EXISTS genomes_commit_idx ON genomes(commit_id);
                CREATE INDEX IF NOT EXISTS genomes_deleted_idx ON genomes(deleted_at);
                CREATE TABLE IF NOT EXISTS indexed_genome_commits (
                    commit_id TEXT PRIMARY KEY REFERENCES commits(commit_id) ON DELETE CASCADE,
                    indexed_at TEXT NOT NULL
                );
                """
            )
            branch_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(branches)")
            }
            if "archived_at" not in branch_columns:
                connection.execute("ALTER TABLE branches ADD COLUMN archived_at TEXT")
            if "deleted_at" not in branch_columns:
                connection.execute("ALTER TABLE branches ADD COLUMN deleted_at TEXT")
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
                SELECT b.name, b.head_commit, c.generation, c.commit_id,
                       b.archived_at, b.deleted_at
                FROM branches AS b
                LEFT JOIN commits AS c ON c.commit_id = b.head_commit
                WHERE b.deleted_at IS NULL
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
                "SELECT archived_at FROM branches WHERE name = ? AND deleted_at IS NULL",
                (name,),
            ).fetchone()
            if exists is None:
                raise ValueError(f"unknown branch: {name}")
            if exists["archived_at"] is not None:
                raise ValueError(f"branch is archived: {name}")
            connection.execute(
                "UPDATE settings SET value = ? WHERE key = 'active_branch'",
                (name,),
            )

    def rename_branch(self, name: str, new_name: str) -> None:
        self._validate_branch_name(new_name)
        old_work = self.work_path(name)
        new_work = self.work_path(new_name)
        if old_work.exists() and new_work.exists():
            raise FileExistsError(f"branch work file already exists: {new_work}")
        moved_work = old_work.is_file()
        if moved_work:
            os.replace(old_work, new_work)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT 1 FROM branches WHERE name = ? AND deleted_at IS NULL",
                    (name,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"unknown branch: {name}")
                if connection.execute(
                    "SELECT 1 FROM branches WHERE name = ?", (new_name,)
                ).fetchone():
                    raise ValueError(f"branch already exists: {new_name}")
                connection.execute(
                    "UPDATE branches SET name = ? WHERE name = ?", (new_name, name)
                )
                connection.execute(
                    "UPDATE settings SET value = ? "
                    "WHERE key = 'active_branch' AND value = ?",
                    (new_name, name),
                )
        except Exception:
            if moved_work and new_work.exists():
                os.replace(new_work, old_work)
            raise

    def archive_branch(self, name: str, archived: bool = True) -> None:
        if name == self.active_branch() and archived:
            raise ValueError("checkout another branch before archiving the active branch")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE branches SET archived_at = ? "
                "WHERE name = ? AND deleted_at IS NULL",
                (
                    datetime.now(timezone.utc).isoformat() if archived else None,
                    name,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"unknown branch: {name}")

    def delete_branch(self, name: str) -> None:
        self.delete_branches([name])

    def delete_branches(self, names: list[str]) -> int:
        if not names:
            raise ValueError("select at least one branch")
        unique_names = list(dict.fromkeys(names))
        if self.active_branch() in unique_names:
            raise ValueError("checkout another branch before deleting the active branch")
        placeholders = ",".join("?" for _ in unique_names)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT name FROM branches WHERE name IN ({placeholders}) "
                "AND deleted_at IS NULL",
                unique_names,
            ).fetchall()
            if len(rows) != len(unique_names):
                raise ValueError("one or more branches are unknown or already in the trash")
            live_count = connection.execute(
                "SELECT COUNT(*) FROM branches WHERE deleted_at IS NULL"
            ).fetchone()[0]
            if live_count <= len(unique_names):
                raise ValueError("cannot delete the last active branch")
            cursor = connection.execute(
                f"UPDATE branches SET deleted_at = ? WHERE name IN ({placeholders}) "
                "AND deleted_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), *unique_names),
            )
            return cursor.rowcount

    def restore_branch(self, name: str) -> None:
        self.restore_branches([name])

    def restore_branches(self, names: list[str]) -> int:
        if not names:
            raise ValueError("select at least one branch")
        unique_names = list(dict.fromkeys(names))
        placeholders = ",".join("?" for _ in unique_names)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT name FROM branches WHERE name IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                unique_names,
            ).fetchall()
            if len(rows) != len(unique_names):
                raise ValueError("one or more branches are not in the trash")
            cursor = connection.execute(
                f"UPDATE branches SET deleted_at = NULL WHERE name IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                unique_names,
            )
            return cursor.rowcount

    def trash_branches(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, head_commit, deleted_at FROM branches "
                "WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_branch(self, name: str) -> None:
        if name == self.active_branch():
            raise ValueError("the active branch cannot be permanently deleted")
        with self._connect() as connection:
            live_count = connection.execute(
                "SELECT COUNT(*) FROM branches WHERE deleted_at IS NULL"
            ).fetchone()[0]
            cursor = connection.execute(
                "DELETE FROM branches WHERE name = ? AND deleted_at IS NOT NULL",
                (name,),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"branch is not in the trash: {name}")
            if live_count == 0:
                raise RuntimeError("training history must keep at least one branch")

    def trash_genomes(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT genome_id, metadata_json, deleted_at
                FROM genomes WHERE deleted_at IS NOT NULL
                ORDER BY deleted_at DESC
                """
            ).fetchall()
        return [
            {
                "genome_id": row["genome_id"],
                "metadata": json.loads(row["metadata_json"]),
                "deleted_at": row["deleted_at"],
            }
            for row in rows
        ]

    def purge_genomes(self, genome_ids: list[str]) -> int:
        if not genome_ids:
            raise ValueError("select at least one genome")
        placeholders = ",".join("?" for _ in genome_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT genome_id FROM genomes WHERE genome_id IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                genome_ids,
            ).fetchall()
            if len(rows) != len(set(genome_ids)):
                raise ValueError("one or more genomes are not in the trash")
            connection.execute(
                f"DELETE FROM genomes WHERE genome_id IN ({placeholders})", genome_ids
            )
        return len(rows)

    def purge_branches(self, names: list[str]) -> int:
        if not names:
            raise ValueError("select at least one branch")
        if self.active_branch() in names:
            raise ValueError("the active branch cannot be permanently deleted")
        placeholders = ",".join("?" for _ in names)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT name FROM branches WHERE name IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                names,
            ).fetchall()
            if len(rows) != len(set(names)):
                raise ValueError("one or more branches are not in the trash")
            total_live = connection.execute(
                "SELECT COUNT(*) FROM branches WHERE deleted_at IS NULL"
            ).fetchone()[0]
            connection.execute(
                f"DELETE FROM branches WHERE name IN ({placeholders})", names
            )
            if total_live < 1:
                raise ValueError("training history must keep at least one branch")
        return len(rows)

    def fast_forward_branch(self, name: str, to_ref: str) -> str:
        target = self.resolve_ref(to_ref)
        with self._connect() as connection:
            branch = connection.execute(
                "SELECT head_commit FROM branches "
                "WHERE name = ? AND deleted_at IS NULL AND archived_at IS NULL",
                (name,),
            ).fetchone()
            if branch is None:
                raise ValueError(f"unknown or archived branch: {name}")
            current = branch["head_commit"]
            if current is not None:
                row = connection.execute(
                    """
                    WITH RECURSIVE ancestry(commit_id, parent_id) AS (
                        SELECT commit_id, parent_id FROM commits WHERE commit_id = ?
                        UNION ALL
                        SELECT c.commit_id, c.parent_id
                        FROM commits c JOIN ancestry a ON c.commit_id = a.parent_id
                    )
                    SELECT 1 FROM ancestry WHERE commit_id = ? LIMIT 1
                    """,
                    (target, current),
                ).fetchone()
                if row is None:
                    raise ValueError("fast-forward target is not a descendant of the branch head")
            connection.execute(
                "UPDATE branches SET head_commit = ? WHERE name = ?", (target, name)
            )
        return target

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

    def list_genomes(
        self,
        *,
        branch: str | None = None,
        query: str = "",
        include_deleted: bool = False,
    ) -> list[dict]:
        self._index_missing_genomes()
        with self._connect() as connection:
            branch_names = (
                [branch]
                if branch is not None
                else [
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM branches WHERE deleted_at IS NULL"
                    )
                ]
            )
            visible_commits: set[str] = set()
            for name in branch_names:
                row = connection.execute(
                    "SELECT head_commit FROM branches "
                    "WHERE name = ? AND deleted_at IS NULL",
                    (name,),
                ).fetchone()
                if row is None:
                    if branch is not None:
                        raise ValueError(f"unknown branch: {name}")
                    continue
                commit_id = row["head_commit"]
                while commit_id and commit_id not in visible_commits:
                    visible_commits.add(commit_id)
                    parent = connection.execute(
                        "SELECT parent_id FROM commits WHERE commit_id = ?",
                        (commit_id,),
                    ).fetchone()
                    commit_id = parent["parent_id"] if parent else None
            deleted_filter = "" if include_deleted else "AND g.deleted_at IS NULL"
            rows = connection.execute(
                f"""
                SELECT g.*, c.generation, c.branch_name
                FROM genomes g
                LEFT JOIN commits c ON c.commit_id = g.commit_id
                WHERE 1 = 1 {deleted_filter}
                ORDER BY g.created_at DESC, g.genome_id
                """
            ).fetchall()
        needle = query.casefold().strip()
        results = []
        for row in rows:
            item = dict(row)
            metadata = json.loads(item.pop("metadata_json"))
            item["metadata"] = metadata
            if item["source_kind"] == "checkpoint" and item["commit_id"] not in visible_commits:
                continue
            if branch is not None and item["source_kind"] == "import":
                if metadata.get("branch") != branch:
                    continue
            if needle and needle not in " ".join(
                str(value)
                for value in (
                    item["genome_id"],
                    item["member_kind"],
                    item["branch_name"],
                    item["generation"],
                    metadata,
                )
            ).casefold():
                continue
            results.append(item)
        return results

    def genome(self, genome_id: str) -> tuple[dict, np.ndarray]:
        self._index_missing_genomes()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT g.*, c.generation, c.branch_name
                FROM genomes g LEFT JOIN commits c ON c.commit_id = g.commit_id
                WHERE g.genome_id = ?
                """,
                (genome_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown genome: {genome_id}")
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        if item["source_kind"] == "checkpoint":
            path = self.objects / f"{item['checkpoint_hash']}.npz"
            key = {
                "best": "best_ever_genome",
                "population": "population_genomes",
                "hall_of_fame": "hall_of_fame_genomes",
                "map_elites": "archive_genomes",
            }[item["member_kind"]]
            with np.load(path, allow_pickle=False) as archive:
                vector = archive[key]
                if vector.ndim == 2:
                    vector = vector[int(item["member_index"])]
                vector = np.asarray(vector, dtype=np.float64).copy()
        else:
            path = self.genome_objects / f"{item['object_hash']}.npy"
            with path.open("rb") as stream:
                vector = np.load(stream, allow_pickle=False)
            vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"stored genome has invalid numeric data: {genome_id}")
        return item, vector

    @property
    def genome_objects(self) -> Path:
        path = self.root / "genome_objects"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def import_genome(self, metadata: dict, vector: np.ndarray) -> dict:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
            raise ValueError("genome must be a non-empty finite one-dimensional vector")
        if metadata.get("format") != "citysim-genome" or metadata.get("version") != 1:
            raise ValueError("unsupported genome file; expected CitySim genome format v1")
        if int(metadata.get("parameter_count", -1)) != vector.size:
            raise ValueError("genome parameter_count does not match the vector size")
        buffer = io.BytesIO()
        np.save(buffer, vector, allow_pickle=False)
        payload = buffer.getvalue()
        object_hash = hashlib.sha256(payload).hexdigest()
        object_path = self.genome_objects / f"{object_hash}.npy"
        if not object_path.exists():
            temporary = object_path.with_name(f".{object_hash}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, object_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        genome_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        clean_metadata = dict(metadata)
        clean_metadata.setdefault("name", "Imported genome")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO genomes (
                    genome_id, source_kind, object_hash, parent_genome_id,
                    metadata_json, created_at
                ) VALUES (?, 'import', ?, NULL, ?, ?)
                """,
                (
                    genome_id,
                    object_hash,
                    json.dumps(clean_metadata, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        return {"genome_id": genome_id, "metadata": clean_metadata}

    def update_genome_metadata(self, genome_id: str, updates: dict) -> dict:
        allowed = {"name", "notes", "tags"}
        if set(updates) - allowed:
            raise ValueError(f"only these metadata fields can be edited: {', '.join(sorted(allowed))}")
        if "name" in updates:
            name = str(updates["name"]).strip()
            if not name or len(name) > 100:
                raise ValueError("genome name must contain 1-100 characters")
            updates["name"] = name
        if "notes" in updates and len(str(updates["notes"])) > 2000:
            raise ValueError("genome notes must not exceed 2000 characters")
        if "tags" in updates:
            tags = updates["tags"]
            if not isinstance(tags, list) or any(
                not isinstance(tag, str) or len(tag) > 40 for tag in tags
            ):
                raise ValueError("genome tags must be a list of strings up to 40 characters")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM genomes WHERE genome_id = ? AND deleted_at IS NULL",
                (genome_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown active genome: {genome_id}")
            metadata = json.loads(row["metadata_json"])
            metadata.update(updates)
            connection.execute(
                "UPDATE genomes SET metadata_json = ? WHERE genome_id = ?",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), genome_id),
            )
        return metadata

    def clone_genome(self, genome_id: str, name: str | None = None) -> dict:
        source, vector = self.genome(genome_id)
        metadata = dict(source["metadata"])
        if source["source_kind"] == "checkpoint":
            checkpoint_metadata = read_checkpoint_metadata(
                self.objects / f"{source['checkpoint_hash']}.npz"
            )
            metadata["architecture"] = checkpoint_metadata["evolution_config"]
        metadata.update(
            {
                "format": "citysim-genome",
                "version": 1,
                "parameter_count": int(vector.size),
            }
        )
        metadata["name"] = (name or f"{metadata.get('name', 'Genome')} copy").strip()
        if not metadata["name"] or len(metadata["name"]) > 100:
            raise ValueError("genome name must contain 1-100 characters")
        metadata["cloned_from"] = genome_id
        result = self.import_genome(metadata, vector)
        with self._connect() as connection:
            connection.execute(
                "UPDATE genomes SET parent_genome_id = ? WHERE genome_id = ?",
                (genome_id, result["genome_id"]),
            )
        return result

    def delete_genomes(self, genome_ids: list[str]) -> int:
        if not genome_ids:
            raise ValueError("select at least one genome")
        unique_ids = list(dict.fromkeys(genome_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT genome_id FROM genomes WHERE genome_id IN ({placeholders}) "
                "AND deleted_at IS NULL",
                unique_ids,
            ).fetchall()
            if len(rows) != len(unique_ids):
                raise ValueError("one or more genomes are unknown or already in the trash")
            cursor = connection.execute(
                f"UPDATE genomes SET deleted_at = ? WHERE genome_id IN ({placeholders}) "
                "AND deleted_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), *unique_ids),
            )
            return cursor.rowcount

    def restore_genome(self, genome_id: str) -> None:
        self.restore_genomes([genome_id])

    def restore_genomes(self, genome_ids: list[str]) -> int:
        if not genome_ids:
            raise ValueError("select at least one genome")
        unique_ids = list(dict.fromkeys(genome_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT genome_id FROM genomes WHERE genome_id IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                unique_ids,
            ).fetchall()
            if len(rows) != len(unique_ids):
                raise ValueError("one or more genomes are not in the trash")
            cursor = connection.execute(
                f"UPDATE genomes SET deleted_at = NULL WHERE genome_id IN ({placeholders}) "
                "AND deleted_at IS NOT NULL",
                unique_ids,
            )
            return cursor.rowcount

    def compare_genomes(self, left_id: str, right_id: str) -> dict:
        left_meta, left = self.genome(left_id)
        right_meta, right = self.genome(right_id)
        if left.shape != right.shape:
            raise ValueError("genomes use different parameter counts and cannot be compared")
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        cosine = (
            float(np.dot(left, right) / (left_norm * right_norm))
            if left_norm and right_norm
            else None
        )
        difference = left - right
        return {
            "left": {"genome_id": left_id, "metadata": left_meta["metadata"]},
            "right": {"genome_id": right_id, "metadata": right_meta["metadata"]},
            "parameter_count": int(left.size),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "l2_distance": float(np.linalg.norm(difference)),
            "cosine_similarity": cosine,
        }

    def _index_missing_genomes(self) -> None:
        with self._connect() as connection:
            commits = connection.execute(
                """
                SELECT c.commit_id, c.checkpoint_hash, c.generation, c.branch_name
                FROM commits c LEFT JOIN indexed_genome_commits i ON i.commit_id = c.commit_id
                WHERE i.commit_id IS NULL ORDER BY c.created_at
                """
            ).fetchall()
        for commit in commits:
            checkpoint_path = self.objects / f"{commit['checkpoint_hash']}.npz"
            records = []
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                for kind, array_name in (
                    ("best", "best_ever_genome"),
                    ("population", "population_genomes"),
                    ("hall_of_fame", "hall_of_fame_genomes"),
                    ("map_elites", "archive_genomes"),
                ):
                    if array_name not in archive.files:
                        continue
                    genomes = archive[array_name]
                    if genomes.ndim == 1:
                        genomes = genomes.reshape(1, -1)
                    fitness_name = {
                        "population": "population_fitness",
                        "hall_of_fame": "hall_of_fame_fitness",
                        "map_elites": "archive_fitness",
                    }.get(kind)
                    fitness = archive[fitness_name] if fitness_name in archive.files else None
                    for index, vector in enumerate(genomes):
                        if vector.ndim != 1 or not np.isfinite(vector).all():
                            raise ValueError(
                                f"invalid {kind} genome in checkpoint {commit['commit_id']}"
                            )
                        genome_id = hashlib.sha256(
                            f"{commit['commit_id']}:{kind}:{index}".encode("ascii")
                        ).hexdigest()
                        metadata = {
                            "name": f"{kind.replace('_', ' ').title()} · gen {commit['generation']} · {index + 1}",
                            "kind": kind,
                            "generation": int(commit["generation"]),
                            "branch": str(commit["branch_name"]),
                            "parameter_count": int(vector.size),
                        }
                        if fitness is not None and index < len(fitness):
                            metadata["fitness"] = float(fitness[index])
                        records.append(
                            (
                                genome_id,
                                commit["commit_id"],
                                commit["checkpoint_hash"],
                                kind,
                                index,
                                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                            )
                        )
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO genomes (
                        genome_id, source_kind, commit_id, checkpoint_hash,
                        member_kind, member_index, metadata_json, created_at
                    ) VALUES (?, 'checkpoint', ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (*record, datetime.now(timezone.utc).isoformat())
                        for record in records
                    ],
                )
                connection.execute(
                    "INSERT OR IGNORE INTO indexed_genome_commits (commit_id, indexed_at) VALUES (?, ?)",
                    (commit["commit_id"], datetime.now(timezone.utc).isoformat()),
                )

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
            path = self.root / name
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
            f"objects/{path.name}"
            for path in self.objects.glob("*.npz")
            if path.is_file()
            and re.fullmatch(r"[a-f0-9]{64}", path.stem)
            and path.stem not in referenced_objects
        )
        referenced_genomes = {
            row["object_hash"]
            for row in connection.execute(
                "SELECT object_hash FROM genomes WHERE object_hash IS NOT NULL"
            )
        }
        orphan_objects.extend(
            f"genome_objects/{path.name}"
            for path in self.genome_objects.glob("*.npy")
            if path.is_file()
            and re.fullmatch(r"[a-f0-9]{64}", path.stem)
            and path.stem not in referenced_genomes
        )
        orphan_objects.sort()
        total_bytes = sum((self.root / name).stat().st_size for name in orphan_objects)
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
