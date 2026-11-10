from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from citysim.analytics import AnalyticsRecorder, load_analytics, write_training_analytics
from citysim.config import quick_city_config
from citysim.dashboard import create_app
from citysim.episode import run_episode, score_breakdown, score_episode
from citysim.experiment_store import ExperimentStore
from citysim.policy import RandomPolicy
from citysim.training_history import TrainingHistoryStore


class TestAnalytics(unittest.TestCase):
    def test_episode_records_actions_rounds_and_filtered_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            city = quick_city_config(n_agents=2, actions_per_agent=3)
            recorder = AnalyticsRecorder(root, "demo", seed=17)
            result = run_episode(
                city,
                [RandomPolicy(), RandomPolicy()],
                17,
                record_history=True,
                on_action=recorder.record_action,
                on_round_end=recorder.record_round,
            )
            scores = score_episode(result)
            recorder.finish(result, scores, score_breakdown(result))

            action_rows = [
                json.loads(line)
                for line in (root / "actions.ndjson").read_text(encoding="utf-8").splitlines()
            ]
            round_rows = [
                json.loads(line)
                for line in (root / "rounds.ndjson").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(action_rows), len(result.actions))
            self.assertEqual(len(round_rows), city.actions_per_agent)
            self.assertIn("mean_satisfaction", round_rows[-1]["city"])
            self.assertIn("damage", action_rows[0])
            self.assertIn("floors", action_rows[0]["action"])
            self.assertIn("roof_quality", action_rows[0]["action"])

            detail = load_analytics(root, round_start=1, agent_id=0)
            self.assertTrue(all(row["round"] >= 1 for row in detail["actions"]))
            self.assertTrue(all(row["agent_id"] == 0 for row in detail["actions"]))
            self.assertEqual(
                detail["action_metrics"]["count"],
                detail["action_metrics"]["accepted"] + detail["action_metrics"]["rejected"],
            )
            self.assertEqual(detail["summary"]["selected_round_count"], 2)
            self.assertIn("median", detail["summary"]["metrics"]["city_quality"])
            self.assertEqual(
                detail["action_metrics"]["building_count"],
                sum(detail["action_metrics"]["roof_quality_counts"].values()),
            )

    def test_training_series_are_persisted_with_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = write_training_analytics(
                directory,
                [
                    {"gen": 0, "best": 0.1, "mean": 0.0},
                    {"gen": 1, "best": 0.5, "mean": 0.2},
                ],
                {"branch": "main"},
            )
            loaded = json.loads(
                (Path(directory) / "training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["generation_count"], 2)
            self.assertEqual(loaded["metrics"]["best"]["max"], 0.5)
            self.assertEqual(loaded["metrics"]["best"]["trend_per_round"], 0.4)

    def test_api_serves_local_and_imported_analytics_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history"
            TrainingHistoryStore(history)
            output = root / "output"
            artifacts = output / ".dashboard" / "jobs" / ("a" * 32) / "artifacts"
            artifacts.mkdir(parents=True)
            store = ExperimentStore(output / ".dashboard" / "store", history)
            store.register_run("a" * 32, "demo", {}, artifacts)
            store.update_run("a" * 32, "completed")
            (artifacts / "actions.ndjson").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "elapsed_seconds": 1,
                        "round": 0,
                        "agent_id": 0,
                        "outcome": "accepted",
                        "damage": 2,
                        "action": {
                            "kind": "build",
                            "cell": 1,
                            "type_id": 2,
                            "quality": 1,
                            "cost": 3,
                            "bid": 4,
                            "demolish": False,
                        },
                        "agent_state": {"budget": 10, "profit": 1},
                    }
                ) + "\n",
                encoding="utf-8",
            )
            (artifacts / "rounds.ndjson").write_text(
                json.dumps({"round": 0, "city": {"city_quality": 0.5}}) + "\n",
                encoding="utf-8",
            )
            (artifacts / "analytics_summary.json").write_text(
                json.dumps({"city": {"city_quality": 0.5}, "agents": [], "scores": []}),
                encoding="utf-8",
            )
            app = create_app(str(history), str(output), port=18765)
            routes = {
                (route.path, method): route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
                for method in getattr(route, "methods", ())
            }

            catalog = routes[("/api/analytics", "GET")]()
            self.assertEqual(catalog["sources"][0]["source_id"], "a" * 32)
            detail = routes[("/api/analytics/{source_type}/{source_id}", "GET")](
                "run", "a" * 32, None, None, None
            )
            self.assertEqual(detail["action_metrics"]["count"], 1)
            self.assertEqual(detail["rounds"][0]["city"]["city_quality"], 0.5)
            response = routes[
                ("/api/analytics/{source_type}/{source_id}/export", "GET")
            ]("run", "a" * 32, "csv", None, None, None)

            async def read_stream() -> str:
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(
                        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    )
                return "".join(chunks)

            csv_data = asyncio.run(read_stream())
            self.assertIn("floors", csv_data.splitlines()[0])
            self.assertIn("roof_quality", csv_data.splitlines()[0])
            self.assertIn("accepted", csv_data)

    def test_action_sample_is_bounded_and_histogram_covers_all_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = AnalyticsRecorder(root, "demo", seed=5)
            city = quick_city_config(n_agents=2, actions_per_agent=2)
            result = run_episode(
                city,
                [RandomPolicy(), RandomPolicy()],
                5,
                on_action=recorder.record_action,
                on_round_end=recorder.record_round,
            )
            recorder.finish(result, score_episode(result), score_breakdown(result))

            detail = load_analytics(root, action_limit=1)
            self.assertLessEqual(len(detail["actions"]), 1)
            histogram_count = sum(
                item["count"] for item in detail["action_metrics"]["cost_bins"]
            )
            self.assertEqual(histogram_count, detail["action_metrics"]["sample_count"])
            self.assertTrue(np.isfinite(detail["summary"]["metrics"]["city_quality"]["mean"]))


if __name__ == "__main__":
    unittest.main()
