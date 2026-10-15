"""Optional S16P adapter with explicit receive/source 4x4 topology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import ARRAY_COLS, ARRAY_ROWS


@dataclass(frozen=True)
class TouchstoneNetwork:
    frequency_hz: np.ndarray
    s_parameters_iq: np.ndarray

    def __post_init__(self) -> None:
        expected = (
            self.frequency_hz.size,
            2,
            ARRAY_ROWS,
            ARRAY_COLS,
            ARRAY_ROWS,
            ARRAY_COLS,
        )
        if self.s_parameters_iq.shape != expected:
            raise ValueError(
                "s_parameters_iq must have shape "
                "[frequency,IQ,receive_row,receive_col,source_row,source_col]"
            )


def s_matrix_to_grid(s_matrix: np.ndarray) -> np.ndarray:
    """Convert [F,receive_port,source_port] using explicit row-major mapping."""
    matrix = np.asarray(s_matrix)
    if matrix.ndim != 3 or matrix.shape[1:] != (16, 16):
        raise ValueError("S matrix must have shape [F,16,16]")
    grid = np.empty(
        (matrix.shape[0], ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, ARRAY_COLS),
        dtype=np.complex128,
    )
    for receive_port in range(16):
        receive_row, receive_col = divmod(receive_port, ARRAY_COLS)
        for source_port in range(16):
            source_row, source_col = divmod(source_port, ARRAY_COLS)
            grid[:, receive_row, receive_col, source_row, source_col] = matrix[
                :, receive_port, source_port
            ]
    return grid


def load_touchstone(path: str | Path) -> TouchstoneNetwork:
    try:
        import skrf  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Touchstone import requires the optional 'scikit-rf' dependency; "
            "install v2.8 with the 'hfss' extra"
        ) from error
    network = skrf.Network(str(Path(path).resolve()))
    if network.nports != 16:
        raise ValueError(f"expected S16P with 16 ports, found {network.nports}")
    grid = s_matrix_to_grid(network.s)
    return TouchstoneNetwork(
        frequency_hz=np.asarray(network.f, dtype=np.float64),
        s_parameters_iq=np.stack((grid.real, grid.imag), axis=1),
    )

