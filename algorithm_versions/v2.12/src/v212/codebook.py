"""Physically interpretable 96-command sensing library and split contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch


TRAIN_VERTICAL_EDGES = ((0, 4), (4, 8), (8, 12), (1, 5))
VALIDATION_EDGES = ((2, 6), (3, 7), (5, 9), (6, 10))
TEST_EDGES = ((7, 11), (9, 13), (10, 14), (11, 15))


def horizontal_edges() -> tuple[tuple[int, int], ...]:
    return tuple((4 * row + col, 4 * row + col + 1) for row in range(4) for col in range(3))


@dataclass(frozen=True)
class MeasurementCodebook:
    weights: torch.Tensor
    kinds: tuple[str, ...]
    edges: tuple[tuple[int, int] | None, ...]
    splits: dict[str, tuple[int, ...]]
    random_seed: int
    phase_convention: str = "second_active_element_plus_90_degrees"

    def __post_init__(self) -> None:
        if not torch.is_complex(self.weights) or self.weights.shape != (96, 16):
            raise ValueError("weights must be complex with shape [96,16]")
        if len(self.kinds) != 96 or len(self.edges) != 96:
            raise ValueError("kinds and edges must contain 96 entries")
        expected = {"train": 64, "validation": 16, "test": 16}
        seen: set[int] = set()
        for name, count in expected.items():
            indices = self.splits.get(name)
            if indices is None or len(indices) != count:
                raise ValueError(f"{name} split must contain {count} commands")
            if seen.intersection(indices):
                raise ValueError("command splits overlap")
            seen.update(indices)
        if seen != set(range(96)):
            raise ValueError("command splits must cover all commands exactly once")
        norms = torch.linalg.vector_norm(self.weights, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6):
            raise ValueError("all commands must have unit L2 norm")
        self._validate_train_graph()

    def _validate_train_graph(self) -> None:
        adjacency: dict[int, set[int]] = {index: set() for index in range(16)}
        for index in self.splits["train"]:
            edge = self.edges[index]
            if edge is not None:
                left, right = edge
                adjacency[left].add(right)
                adjacency[right].add(left)
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node] - reached:
                reached.add(neighbour)
                frontier.append(neighbour)
        if len(reached) != 16:
            raise ValueError("training pair graph must connect all 16 elements")

    def indices(self, split: str) -> torch.Tensor:
        if split not in self.splits:
            raise ValueError(f"unknown split: {split}")
        return torch.tensor(self.splits[split], dtype=torch.long)

    def with_supplemental_commands(self, supplemental: torch.Tensor) -> "MeasurementCodebook":
        supplemental = torch.as_tensor(supplemental)
        if not torch.is_complex(supplemental) or supplemental.shape != (32, 16):
            raise ValueError("supplemental must be complex with shape [32,16]")
        if not torch.allclose(
            supplemental.abs(),
            torch.full_like(supplemental.abs(), 0.25),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("supplemental commands must be element-wise constant modulus 1/4")
        random_indices = [index for index, kind in enumerate(self.kinds) if kind == "random"]
        weights = self.weights.clone()
        weights[random_indices] = supplemental.to(weights)
        kinds = list(self.kinds)
        for index in random_indices:
            kinds[index] = "optimized"
        return MeasurementCodebook(
            weights=weights,
            kinds=tuple(kinds),
            edges=self.edges,
            splits=self.splits,
            random_seed=self.random_seed,
            phase_convention=self.phase_convention,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "v212-measurement-codebook-1",
            "normalization": "unit_l2",
            "phase_convention": self.phase_convention,
            "random_seed": self.random_seed,
            "splits": {name: list(indices) for name, indices in self.splits.items()},
            "commands": [
                {
                    "kind": kind,
                    "edge": list(edge) if edge is not None else None,
                    "real": self.weights[index].real.tolist(),
                    "imag": self.weights[index].imag.tolist(),
                }
                for index, (kind, edge) in enumerate(zip(self.kinds, self.edges))
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MeasurementCodebook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        commands = payload["commands"]
        weights = torch.complex(
            torch.tensor([item["real"] for item in commands], dtype=torch.float32),
            torch.tensor([item["imag"] for item in commands], dtype=torch.float32),
        )
        return cls(
            weights=weights,
            kinds=tuple(str(item["kind"]) for item in commands),
            edges=tuple(
                None if item["edge"] is None else tuple(int(x) for x in item["edge"])
                for item in commands
            ),
            splits={name: tuple(int(x) for x in indices) for name, indices in payload["splits"].items()},
            random_seed=int(payload["random_seed"]),
            phase_convention=str(payload["phase_convention"]),
        )


def _pair_commands(edge: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = edge
    in_phase = torch.zeros(16, dtype=torch.complex64)
    quadrature = torch.zeros(16, dtype=torch.complex64)
    scale = 1.0 / math.sqrt(2.0)
    in_phase[left], in_phase[right] = scale, scale
    quadrature[left], quadrature[right] = scale, 1j * scale
    return in_phase, quadrature


def build_baseline_codebook(random_seed: int = 21296) -> MeasurementCodebook:
    generator = torch.Generator().manual_seed(random_seed)
    train_edges = (*horizontal_edges(), *TRAIN_VERTICAL_EDGES)
    split_edges = {
        "train": train_edges,
        "validation": VALIDATION_EDGES,
        "test": TEST_EDGES,
    }
    random_counts = {"train": 16, "validation": 8, "test": 8}
    commands: list[torch.Tensor] = []
    kinds: list[str] = []
    edges: list[tuple[int, int] | None] = []
    splits: dict[str, tuple[int, ...]] = {}

    for split in ("train", "validation", "test"):
        start = len(commands)
        if split == "train":
            for element in range(16):
                command = torch.zeros(16, dtype=torch.complex64)
                command[element] = 1.0
                commands.append(command)
                kinds.append("single")
                edges.append(None)
        for edge in split_edges[split]:
            pair_zero, pair_ninety = _pair_commands(edge)
            commands.extend((pair_zero, pair_ninety))
            kinds.extend(("pair_0", "pair_90"))
            edges.extend((edge, edge))
        phases = 2.0 * math.pi * torch.rand(
            random_counts[split], 16, generator=generator
        )
        for command in torch.polar(torch.full_like(phases, 0.25), phases):
            commands.append(command.to(torch.complex64))
            kinds.append("random")
            edges.append(None)
        splits[split] = tuple(range(start, len(commands)))

    return MeasurementCodebook(
        weights=torch.stack(commands),
        kinds=tuple(kinds),
        edges=tuple(edges),
        splits=splits,
        random_seed=random_seed,
    )


def recover_pair_entry(
    single_left_power: torch.Tensor,
    single_right_power: torch.Tensor,
    in_phase_power: torch.Tensor,
    quadrature_power: torch.Tensor,
) -> torch.Tensor:
    """Recover Q_ij=g_i* g_j for the +j second-element convention."""
    mean_single = 0.5 * (single_left_power + single_right_power)
    real = in_phase_power - mean_single
    imag = mean_single - quadrature_power
    return torch.complex(real, imag)

