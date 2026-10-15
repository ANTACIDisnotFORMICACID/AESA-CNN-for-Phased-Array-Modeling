"""Direction-conditioned, physically masked aperture graph CNN for V2.15."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from v210.models.blocks_2d import ResidualBlock2D
from v212.encoding import DirectionFourierEncoder
from v212.operator import CoherentRank1Operator


GRAPH_MODES = ("graph_geometry", "graph_s_true", "graph_s_shuffled")


def _symmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    value = np.asarray(adjacency, dtype=np.float64)
    degree = value.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    return inverse[:, None] * value * inverse[None, :]


def build_coupling_graphs(
    s_matrix: np.ndarray,
    mode: str,
    *,
    top_k: int = 4,
    shuffle_seed: int = 21500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized row, column and optional S-supported graphs."""
    if mode not in GRAPH_MODES:
        raise ValueError(f"unknown V2.15 graph mode: {mode}")
    if not 1 <= top_k <= 15:
        raise ValueError("top_k must be between 1 and 15")
    s_matrix = np.asarray(s_matrix)
    if s_matrix.shape != (16, 16) or not np.iscomplexobj(s_matrix) or not np.isfinite(s_matrix).all():
        raise ValueError("s_matrix must be finite complex [16,16]")
    row = np.zeros((16, 16), dtype=np.float64)
    col = np.zeros((16, 16), dtype=np.float64)
    for index in range(16):
        r, c = divmod(index, 4)
        for other in range(16):
            rr, cc = divmod(other, 4)
            if c == cc and abs(r - rr) == 1:
                row[index, other] = 1.0
            if r == rr and abs(c - cc) == 1:
                col[index, other] = 1.0
    row = _symmetric_normalize(row)
    col = _symmetric_normalize(col)
    if mode == "graph_geometry":
        return row, col, np.zeros((16, 16), dtype=np.float64)
    magnitude = np.abs(s_matrix).astype(np.float64)
    np.fill_diagonal(magnitude, 0.0)
    mask = np.zeros_like(magnitude, dtype=bool)
    strongest = np.argsort(magnitude, axis=1)[:, -top_k:]
    mask[np.arange(16)[:, None], strongest] = True
    retained = np.where(mask | mask.T, magnitude, 0.0)
    retained = np.maximum(retained, retained.T)
    s_graph = _symmetric_normalize(retained)
    if mode == "graph_s_shuffled":
        permutation = np.random.default_rng(shuffle_seed).permutation(16)
        s_graph = s_graph[np.ix_(permutation, permutation)]
    return row, col, s_graph


def _geometry_maps() -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


class DirectionConditionedGraphResidual(nn.Module):
    """Zero-start Fourier CNN with a direction-gated physical graph block."""

    def __init__(
        self,
        s_matrix: np.ndarray,
        *,
        coupling_mode: str,
        width: int = 32,
        maximum_magnitude: float = 2.0,
        top_k: int = 4,
        shuffle_seed: int = 21500,
    ) -> None:
        super().__init__()
        if width <= 0 or maximum_magnitude <= 0:
            raise ValueError("width and maximum_magnitude must be positive")
        self.encoder = DirectionFourierEncoder(4)
        self.stem = nn.Conv2d(31, width, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.local_block = ResidualBlock2D(width)
        self.direction_gate = nn.Linear(self.encoder.output_channels, 3)
        self.graph_projection = nn.Conv2d(width, width, kernel_size=1)
        self.head = nn.Conv2d(width, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        row, col, coupling = build_coupling_graphs(
            s_matrix, coupling_mode, top_k=top_k, shuffle_seed=shuffle_seed
        )
        self.register_buffer("row_graph", torch.as_tensor(row, dtype=torch.float32))
        self.register_buffer("col_graph", torch.as_tensor(col, dtype=torch.float32))
        self.register_buffer("coupling_graph", torch.as_tensor(coupling, dtype=torch.float32))
        self.register_buffer("geometry_maps", _geometry_maps())
        self.coupling_mode = coupling_mode
        self.width = int(width)
        self.maximum_magnitude = float(maximum_magnitude)
        self.top_k = int(top_k)
        self.shuffle_seed = int(shuffle_seed)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, query_count = unit_direction.shape[:2]
        encoded = self.encoder(unit_direction)
        direction_maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(unit_direction)[None, None].expand(
            batch, query_count, -1, -1, -1
        )
        features = torch.cat((direction_maps, geometry), dim=2)
        hidden = self.activation(self.stem(features.reshape(batch * query_count, 31, 4, 4)))
        hidden = self.local_block(hidden)
        flat = hidden.reshape(batch * query_count, self.width, 16)
        gate = torch.tanh(self.direction_gate(encoded.reshape(batch * query_count, -1)))
        graphs = (self.row_graph, self.col_graph, self.coupling_graph)
        message = torch.zeros_like(flat)
        for graph_index, graph in enumerate(graphs):
            propagated = torch.einsum("nm,bcm->bcn", graph.to(flat), flat)
            message = message + gate[:, graph_index, None, None] * propagated
        graph_delta = self.graph_projection(message.reshape(-1, self.width, 4, 4))
        raw = self.head(self.activation(hidden + graph_delta))
        raw = raw.reshape(batch, query_count, 2, 4, 4)
        return self.maximum_magnitude * torch.tanh(raw / self.maximum_magnitude)


class V215GraphOperator(CoherentRank1Operator):
    """V2.12 coherent operator with the complete V2.15 graph residual CNN."""

    def __init__(
        self,
        *args,
        s_matrix: np.ndarray,
        coupling_mode: str,
        graph_width: int = 32,
        graph_top_k: int = 4,
        graph_shuffle_seed: int = 21500,
        **kwargs,
    ) -> None:
        super().__init__(*args, residual_enabled=True, **kwargs)
        self.residual = DirectionConditionedGraphResidual(
            s_matrix,
            coupling_mode=coupling_mode,
            width=graph_width,
            maximum_magnitude=2.0,
            top_k=graph_top_k,
            shuffle_seed=graph_shuffle_seed,
        )
        self.coupling_mode = coupling_mode
        self.graph_width = int(graph_width)
        self.graph_top_k = int(graph_top_k)
        self.graph_shuffle_seed = int(graph_shuffle_seed)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "family": "v215-direction-conditioned-physical-graph",
            "coupling_mode": self.coupling_mode,
            "graph_width": self.graph_width,
            "graph_top_k": self.graph_top_k,
            "graph_shuffle_seed": self.graph_shuffle_seed,
            "analytic_phase_preserved": True,
            "free_16x16_matrix": False,
            "zero_start_output_head": True,
        }
