from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from citysim.config import quick_city_config
from citysim.cli import _load_genome
from citysim.episode import run_episode
from citysim.live_demo import (
    LiveDemoRecorder,
    read_control_file,
    read_live_frame,
    read_live_frames,
    update_control_file,
)
from citysim.policy import GenomeLayout
from citysim.world import Action


class TestLiveDemo(unittest.TestCase):
    def test_state_frames_round_trip_and_incremental_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "frames.jsonl"
            control_path = root / "control.json"
            recorder = LiveDemoRecorder(
                state_path, control_path, {"policy": "greedy", "seed": 3}
            )
            environment = SimpleNamespace(
                h=2,
                w=3,
                round=0,
                type_grid=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8),
                quality_grid=np.array([[-1, 0, 1], [2, 0, 1]], dtype=np.int8),
                owner_grid=np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int16),
            )
            recorder.record_state(
                environment,
                "action",
                0,
                agent_id=1,
                action=Action(cell=2, type_id=1, quality=1),
                accepted=True,
            )
            recorder.close()

            first = read_live_frames(state_path, 0)
            self.assertEqual([frame["seq"] for frame in first["frames"]], [0, 1])
            self.assertEqual(first["frames"][0]["kind"], "metadata")
            state = first["frames"][1]
            self.assertEqual(state["type_grid"], [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(state["quality_grid"], [[-1, 0, 1], [2, 0, 1]])
            self.assertEqual(state["floors_grid"], [[0, 1, 1], [1, 1, 1]])
            self.assertEqual(state["roof_quality_grid"], [[0, 0, 0], [0, 0, 0]])
            self.assertEqual(state["owner_grid"], [[-1, 0, 1], [2, 3, 4]])
            self.assertEqual(state["action"]["cell"], 2)
            self.assertEqual(state["action"]["floors"], 1)
            self.assertEqual(state["action"]["roof_quality"], 0)
            self.assertEqual(read_live_frame(state_path, 1)["round"], 1)

            next_batch = read_live_frames(state_path, first["next_offset"])
            self.assertEqual(next_batch["frames"], [])
            self.assertEqual(next_batch["next_offset"], first["next_offset"])

    def test_control_updates_preserve_other_values_and_stop_action_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "frames.jsonl"
            control_path = root / "control.json"
            recorder = LiveDemoRecorder(state_path, control_path, {})
            update_control_file(control_path, paused=True, speed_ms=250)
            self.assertEqual(
                read_control_file(control_path),
                {"paused": True, "speed_ms": 250},
            )
            self.assertEqual(
                update_control_file(control_path, speed_ms=0),
                {"paused": True, "speed_ms": 0},
            )
            (root / "stop.request").touch()
            self.assertFalse(recorder.wait_until_active())
            recorder.close()

    def test_episode_stops_after_action_callback_returns_false(self) -> None:
        config = quick_city_config(n_agents=1, actions_per_agent=2)
        config = replace(config, use_auction=False, buildings_per_action=3)

        class PassingPolicy:
            def act(self, env, agent_id, rng):
                return Action.pass_action()

        result = run_episode(
            config,
            [PassingPolicy()],
            seed=1,
            on_action=lambda *_: False,
        )

        self.assertEqual(len(result.actions), 1)

    def test_trained_checkpoint_archive_provides_its_best_genome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            layout = GenomeLayout(hidden_size=8)
            expected = np.arange(layout.parameter_count, dtype=np.float64)
            np.savez(path, best_ever_genome=expected)

            actual = _load_genome(str(path), layout, seed=4)

            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
