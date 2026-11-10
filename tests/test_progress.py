from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from citysim.progress import TrainingProgress


class TestTrainingProgress(unittest.TestCase):
    def test_write_retries_temporary_replace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            progress = TrainingProgress(path, Path(directory) / "stop.request")
            replace = os.replace
            attempts = 0

            def replace_with_temporary_lock(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("file is temporarily locked")
                replace(source, destination)

            with patch(
                "citysim.progress.os.replace",
                side_effect=replace_with_temporary_lock,
            ) as mocked_replace:
                progress._write({"state": "running", "generation": 2})

            self.assertEqual(mocked_replace.call_count, 3)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"state": "running", "generation": 2},
            )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_write_surfaces_persistent_replace_failure_and_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            progress = TrainingProgress(path, Path(directory) / "stop.request")

            with patch(
                "citysim.progress.os.replace",
                side_effect=PermissionError("file access denied"),
            ) as mocked_replace:
                with self.assertRaisesRegex(PermissionError, "access denied"):
                    progress._write({"state": "running"})

            self.assertEqual(mocked_replace.call_count, 5)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
