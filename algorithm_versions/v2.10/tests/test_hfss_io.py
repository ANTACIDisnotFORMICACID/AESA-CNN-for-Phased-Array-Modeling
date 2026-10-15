from pathlib import Path
import tempfile
import unittest

import numpy as np

from v210.contracts import ArrayGeometry
from v210.io.hfss_ffd import load_hfss_ffd_basis, parse_ffd
from v210.io.touchstone import s_matrix_to_grid


def _write_ffd(path: Path, scale: float) -> None:
    path.write_text(
        "0 90 2\n0 180 2\nFrequency 47e9\n"
        + "\n".join(f"{scale} 0 {2 * scale} 0" for _ in range(4)),
        encoding="utf-8",
    )


class HfssIoTests(unittest.TestCase):
    def test_s_matrix_is_mapped_to_two_explicit_4_by_4_axes(self) -> None:
        matrix = np.zeros((1, 16, 16), dtype=np.complex128)
        for receive in range(16):
            for source in range(16):
                matrix[0, receive, source] = receive * 100 + source
        grid = s_matrix_to_grid(matrix)
        self.assertEqual(grid.shape, (1, 4, 4, 4, 4))
        self.assertEqual(grid[0, 2, 1, 3, 2], matrix[0, 9, 14])

    def test_parse_ffd_grid_and_complex_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "one.ffd"
            _write_ffd(source, 1.0)
            theta, phi, e_theta, e_phi = parse_ffd(source)
            np.testing.assert_array_equal(theta, [0.0, 0.0, 90.0, 90.0])
            np.testing.assert_array_equal(phi, [0.0, 180.0, 0.0, 180.0])
            np.testing.assert_array_equal(e_theta, np.ones(4))
            np.testing.assert_array_equal(e_phi, np.full(4, 2.0))

    def test_load_requires_and_preserves_explicit_row_major_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = ArrayGeometry()
            mapping = {}
            for index, port in enumerate(geometry.port_names, start=1):
                source = root / f"port_{index}.ffd"
                _write_ffd(source, float(index))
                mapping[port] = source
            export = load_hfss_ffd_basis(mapping, geometry)
            self.assertEqual(export.operator_iq.shape, (4, 3, 2, 4, 4))
            # theta=90, phi=0: Etheta points -Z and Ephi points +Y.
            np.testing.assert_allclose(export.operator_iq[2, 1, 0], 2.0 * np.arange(1, 17).reshape(4, 4))
            np.testing.assert_allclose(export.operator_iq[2, 2, 0], -np.arange(1, 17).reshape(4, 4))


if __name__ == "__main__":
    unittest.main()

