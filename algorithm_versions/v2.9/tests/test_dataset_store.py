from pathlib import Path
import tempfile
import unittest

import numpy as np

from v29.contracts import ArrayGeometry
from v29.io import BasisDataset, DatasetManifest, load_basis_dataset, save_basis_dataset
from v29.training import angular_block_split


class DatasetStoreTests(unittest.TestCase):
    def test_dataset_round_trip_including_topological_s_parameters(self) -> None:
        geometry = ArrayGeometry()
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dataset = BasisDataset(
            direction=directions,
            operator_iq=np.zeros((2, 3, 2, 4, 4)),
            theta_deg=np.array([90.0, 90.0]),
            phi_deg=np.array([0.0, 90.0]),
            manifest=DatasetManifest.for_geometry(geometry, "synthetic_fixture"),
            s_frequency_hz=np.array([47.0e9]),
            s_parameters_iq=np.zeros((1, 2, 4, 4, 4, 4)),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_basis_dataset(dataset, Path(directory))
            restored = load_basis_dataset(directory)
        np.testing.assert_array_equal(restored.direction, dataset.direction)
        np.testing.assert_array_equal(restored.operator_iq, dataset.operator_iq)
        self.assertEqual(restored.s_parameters_iq.shape, (1, 2, 4, 4, 4, 4))

    def test_angular_blocks_are_not_split_between_sets(self) -> None:
        theta, phi = np.meshgrid(np.arange(0.0, 50.0, 5.0), np.arange(-40.0, 50.0, 5.0))
        theta = theta.reshape(-1)
        phi = phi.reshape(-1)
        split = angular_block_split(theta, phi, theta_block_deg=10.0, phi_block_deg=10.0)
        memberships = {}
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for index in indices:
                block = (int(theta[index] // 10.0), int((phi[index] + 180.0) // 10.0))
                memberships.setdefault(block, set()).add(name)
        self.assertTrue(all(len(value) == 1 for value in memberships.values()))


if __name__ == "__main__":
    unittest.main()
