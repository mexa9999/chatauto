"""Benchmark batched CitySim policy inference on AMD DirectML vs NumPy CPU."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
    import torch_directml
except ImportError as exc:
    raise SystemExit(
        "GPU benchmark dependencies are missing. Use a separate Python 3.12 "
        "environment and install torch-directml and numpy; see docs/GPU_BENCHMARK.md."
    ) from exc

from citysim.config import quick_city_config
from citysim.policy import DISTRICT_ALLOWED, NeuralPolicy, random_genome
from citysim.world import Action, CityEnvironment, Observation


_NEIGHBOURS = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dy != 0 or dx != 0
)


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def _make_observations(count: int, seed: int) -> list[Observation]:
    config = quick_city_config()
    observations = []
    for index in range(count):
        world = CityEnvironment(config, seed + index)
        agent_id = index % config.n_agents
        buildable = np.flatnonzero(world.buildable.ravel())
        if buildable.size:
            world.apply_action(
                agent_id,
                Action(cell=int(buildable[0]), type_id=0, quality=1),
            )
        observations.append(world.observe(agent_id))
    return observations


class _DirectMLBatchPolicy:
    def __init__(self, policy: NeuralPolicy, device: torch.device) -> None:
        self.policy = policy
        self.device = device
        self.layout = policy.layout

        weights = (
            "Wq", "Wk", "Wv", "b_att", "Wd", "bd", "Wd_type", "bd_type",
            "Wg", "bg", "W1", "b1", "W2", "b2", "w3", "b3",
            "W_type", "b_type", "W_quality", "b_quality", "w_bid", "b_bid",
        )
        self.weights = {
            name: torch.tensor(
                np.asarray(getattr(policy, name)).copy(), dtype=torch.float32
            ).to(device)
            for name in weights
        }
        district_bias = np.zeros(
            (
                self.layout.n_district_types,
                self.layout.n_types,
                self.layout.n_qualities,
            ),
            dtype=np.float32,
        )
        for district_id, allowed_types in enumerate(DISTRICT_ALLOWED):
            for type_id in allowed_types:
                district_bias[district_id, type_id, :] = 0.40
        self.district_bias = torch.tensor(district_bias).to(device)

    def _prepare(
        self, observations: Sequence[Observation]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        torch.Tensor,
    ]:
        first = observations[0]
        height, width = first._shape_hint
        district_labels = first._district_labels.reshape(-1)
        if any(
            obs._shape_hint != first._shape_hint
            or not np.array_equal(obs._district_labels.reshape(-1), district_labels)
            for obs in observations[1:]
        ):
            raise ValueError("all observations in a batch must use the same grid")

        features = torch.tensor(
            np.stack(
                [
                    obs.cell_features.reshape(
                        height, width, self.layout.n_cell_features
                    )
                    for obs in observations
                ]
            ),
            dtype=torch.float32,
        ).to(self.device)
        global_features = torch.tensor(
            np.stack([obs.global_features for obs in observations]),
            dtype=torch.float32,
        ).to(self.device)
        labels_cpu = torch.tensor(district_labels.copy(), dtype=torch.int64)
        labels = labels_cpu.to(self.device)
        n_districts = int(first._n_districts)
        district_mask = functional.one_hot(
            labels_cpu, num_classes=n_districts
        ).to(dtype=torch.float32).to(self.device)
        valid = torch.tensor(
            NeuralPolicy._neighbour_validity(height, width).copy(),
            dtype=torch.bool,
        ).to(self.device)
        return features, global_features, labels, district_mask, height, width, valid

    def __call__(self, observations: Sequence[Observation]) -> tuple[np.ndarray, ...]:
        x, global_features, labels, district_mask, height, width, valid = self._prepare(
            observations
        )
        batch_size = x.shape[0]
        weights = self.weights
        padded = functional.pad(
            x.permute(0, 3, 1, 2), (1, 1, 1, 1)
        ).permute(0, 2, 3, 1)
        query = x @ weights["Wq"] + weights["b_att"]
        keys = padded @ weights["Wk"]
        values = padded @ weights["Wv"]

        attention_logits = []
        neighbour_values = []
        for dy, dx in _NEIGHBOURS:
            y_slice = slice(1 + dy, 1 + dy + height)
            x_slice = slice(1 + dx, 1 + dx + width)
            key_slice = keys[:, y_slice, x_slice, :]
            neighbour_values.append(values[:, y_slice, x_slice, :])
            attention_logits.append((key_slice * query).sum(dim=-1))

        attention = torch.stack(attention_logits, dim=-1)
        attention = attention / np.sqrt(max(1, self.layout.attention_dim))
        attention = attention.masked_fill(~valid[None, :, :, :], -1e9).softmax(dim=-1)
        context = torch.zeros(
            (batch_size, height, width, self.layout.attention_dim),
            dtype=x.dtype,
            device=self.device,
        )
        for index, neighbour in enumerate(neighbour_values):
            context = context + attention[:, :, :, index, None] * neighbour

        flat_features = x.reshape(
            batch_size, height * width, self.layout.n_cell_features
        )
        district_mask_t = district_mask.T[None, :, :].expand(batch_size, -1, -1)
        district_sums = torch.bmm(district_mask_t, flat_features)
        district_counts = district_mask.sum(dim=0).clamp(min=1).reshape(1, -1, 1)
        district_global = (
            global_features[:, None, :].expand(-1, district_counts.shape[1], -1)
            * district_counts
        )
        district_input = torch.cat((district_sums, district_global), dim=-1)
        district_input = district_input / district_counts
        district_hidden = torch.relu(
            district_input @ weights["Wd"] + weights["bd"]
        )
        district_choice = (
            district_hidden @ weights["Wd_type"] + weights["bd_type"]
        ).argmax(dim=-1)
        cell_districts = labels[None, :].expand(batch_size, -1)
        cell_district_choice = torch.gather(district_choice, 1, cell_districts)
        cell_district_choice = cell_district_choice.reshape(
            batch_size, height, width
        )

        global_summary = torch.cat(
            (x.mean(dim=(1, 2)), x.amax(dim=(1, 2)), global_features), dim=-1
        )
        global_embedding = torch.relu(
            global_summary @ weights["Wg"] + weights["bg"]
        )

        feature_dim = self.layout.n_cell_features
        attention_dim = self.layout.attention_dim
        global_dim = self.layout.global_hidden
        w1 = weights["W1"]
        w1_features = w1[:feature_dim, :]
        w1_context = w1[feature_dim : feature_dim + attention_dim, :]
        w1_global = w1[
            feature_dim + attention_dim : feature_dim + attention_dim + global_dim,
            :,
        ]
        w1_district = w1[feature_dim + attention_dim + global_dim :, :]
        flat_district_choices = cell_district_choice.reshape(batch_size, -1)
        district_indices = flat_district_choices[:, :, None].expand(
            -1, -1, w1_district.shape[1]
        )
        district_embedding = torch.gather(
            w1_district[None, :, :].expand(batch_size, -1, -1),
            1,
            district_indices,
        ).reshape(batch_size, height, width, -1)

        hidden1 = x @ w1_features + context @ w1_context
        hidden1 = hidden1 + (global_embedding @ w1_global)[:, None, None, :]
        hidden1 = torch.relu(hidden1 + district_embedding + weights["b1"])
        hidden2 = torch.relu(
            hidden1.reshape(batch_size, height * width, -1) @ weights["W2"]
            + weights["b2"]
        )
        scores = hidden2 @ weights["w3"] + weights["b3"]
        type_logits = (hidden2 @ weights["W_type"] + weights["b_type"]).reshape(
            batch_size, height, width, self.layout.n_types
        )
        quality_logits = (
            hidden2 @ weights["W_quality"] + weights["b_quality"]
        ).reshape(batch_size, height, width, self.layout.n_qualities)
        bid_fraction = torch.sigmoid(
            hidden2 @ weights["w_bid"] + weights["b_bid"]
        ).reshape(batch_size, height, width)

        joint_logits = (
            type_logits[:, :, :, :, None]
            + quality_logits[:, :, :, None, :]
            + self.district_bias[cell_district_choice]
        )

        n_cells = height * width
        n_joint = n_cells * self.layout.n_types * self.layout.n_qualities
        packed = torch.cat(
            (
                scores.reshape(batch_size, n_cells),
                joint_logits.reshape(batch_size, n_joint),
                bid_fraction.reshape(batch_size, n_cells),
            ),
            dim=1,
        ).cpu().numpy()
        scores_end = n_cells
        joint_end = scores_end + n_joint
        scores_out = packed[:, :scores_end]
        joint_out = packed[:, scores_end:joint_end].reshape(
            batch_size,
            height,
            width,
            self.layout.n_types,
            self.layout.n_qualities,
        )
        bids_out = packed[:, joint_end:].reshape(batch_size, height, width)
        return scores_out, joint_out, bids_out


def _cpu_forward(
    policy: NeuralPolicy, observations: Sequence[Observation]
) -> tuple[np.ndarray, ...]:
    outputs = [policy._forward(obs) for obs in observations]
    return (
        np.stack([output["scores"] for output in outputs]),
        np.stack([output["joint_all"] for output in outputs]),
        np.stack([output["bid_fraction"] for output in outputs]),
    )


def _median_ms(function: Callable[[], object], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "torch-directml currently requires a separate Python 3.12 environment; "
            f"this interpreter is {sys.version.split()[0]}."
        )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=_parse_batch_sizes("1,4,16,64,256"),
    )
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.worlds < 1 or max(args.batch_sizes) > args.worlds:
        parser.error("--worlds must be at least the largest batch size")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    if torch_directml.device_count() < 1:
        raise SystemExit("DirectML did not find a supported GPU.")

    device = torch_directml.device()
    gpu_name = torch_directml.device_name(0).rstrip("\x00")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"DirectML GPU: {gpu_name}")
    print(f"Preparing {args.worlds} independent world observations...")
    observations = _make_observations(args.worlds, args.seed)
    policy = NeuralPolicy(random_genome(np.random.default_rng(args.seed)))
    gpu_forward = _DirectMLBatchPolicy(policy, device)

    for batch_size in args.batch_sizes:
        batch = observations[:batch_size]
        cpu_outputs = _cpu_forward(policy, batch)
        for _ in range(args.warmup):
            gpu_forward(batch)

        cpu_ms = _median_ms(lambda: _cpu_forward(policy, batch), args.repeats)
        gpu_ms = _median_ms(lambda: gpu_forward(batch), args.repeats)
        gpu_outputs = gpu_forward(batch)
        max_error = max(
            float(np.max(np.abs(cpu - gpu)))
            for cpu, gpu in zip(cpu_outputs, gpu_outputs)
        )
        print(
            f"batch={batch_size:>4}: CPU={cpu_ms:>8.2f} ms  "
            f"GPU+transfer={gpu_ms:>8.2f} ms  speedup={cpu_ms / gpu_ms:>5.2f}x  "
            f"max_abs_error={max_error:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
