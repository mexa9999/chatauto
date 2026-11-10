from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from citysim.experiment_store import ArchiveError, ExperimentStore
from citysim.training_history import TrainingHistoryStore


class TestExperimentStore(unittest.TestCase):
    def _store_with_run(self, root: Path) -> tuple[ExperimentStore, Path]:
        history = root / "history"
        TrainingHistoryStore(history)
        store = ExperimentStore(root / "catalog", history)
        run_id = "a" * 32
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(
            json.dumps({"fitness": 0.75}), encoding="utf-8"
        )
        store.register_run(
            run_id, "experiment", {"experiment": "e1", "history_dir": str(history)}, run_dir
        )
        store.update_run(run_id, "completed")
        return store, history

    def test_light_archive_import_and_duplicate_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, history = self._store_with_run(root)
            archive = root / "light.zip"
            store.create_export("a" * 32, "light", history, archive)
            archive_bytes = archive.read_bytes()
            duplicate = root / "light-copy.zip"
            duplicate.write_bytes(archive_bytes)

            preview = store.stage_import(archive, "machine-a")
            self.assertEqual(preview["manifest"]["mode"], "light")
            imported = store.commit_import(preview["token"])
            self.assertEqual(imported["status"], "imported")
            self.assertEqual(store.list_datasets()[0]["name"], "machine-a")

            duplicate_preview = store.stage_import(duplicate, "machine-a-copy")
            self.assertIn("archive", duplicate_preview["conflicts"])
            copied = store.commit_import(
                duplicate_preview["token"], action="rename", name="machine-a-copy"
            )
            self.assertEqual(copied["name"], "machine-a-copy")
            self.assertEqual(len(store.list_datasets()), 2)
            details = store.dataset_details(imported["dataset_id"])
            self.assertEqual(details["summaries"][0]["data"]["fitness"], 0.75)

            skipped_archive = root / "light-skip.zip"
            skipped_archive.write_bytes(archive_bytes)
            skipped_preview = store.stage_import(skipped_archive, "machine-a")
            skipped = store.commit_import(skipped_preview["token"], action="skip")
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(len(store.list_datasets()), 2)

    def test_full_archive_can_be_activated_for_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, history = self._store_with_run(root)
            archive = root / "full.zip"
            store.create_export("a" * 32, "full", history, archive)

            preview = store.stage_import(archive, "machine-b")
            imported = store.commit_import(preview["token"])

            activated = store.activate_dataset(imported["dataset_id"])
            self.assertTrue((activated / "history.sqlite3").is_file())
            self.assertEqual(store.active_history_dir(), activated)

    def test_import_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, _ = self._store_with_run(root)
            archive = root / "unsafe.zip"
            manifest = {
                "format": "citysim-experiment",
                "version": 1,
                "mode": "light",
                "source_id": "test",
                "run": {"run_id": "source", "kind": "demo"},
                "files": [
                    {
                        "path": "../outside.txt",
                        "size": 1,
                        "sha256": "0" * 64,
                    }
                ],
            }
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("manifest.json", json.dumps(manifest))
                output.writestr("../outside.txt", "x")

            with self.assertRaises(ArchiveError):
                store.stage_import(archive, "unsafe")


if __name__ == "__main__":
    unittest.main()
