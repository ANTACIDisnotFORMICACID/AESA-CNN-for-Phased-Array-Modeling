import numpy as np
import torch
import unittest

from v29.contracts import ArrayGeometry
from v29.coordinates import (
    direction_to_hfss_angles,
    hfss_angles_to_direction,
    query_from_cartesian_targets,
)


class ContractCoordinateTests(unittest.TestCase):
    def test_geometry_is_explicit_4_by_4_and_far_field_boundary(self) -> None:
        geometry = ArrayGeometry()
        self.assertEqual(geometry.positions_m.shape, (4, 4, 3))
        self.assertAlmostEqual(geometry.fraunhofer_distance_m, 0.7654263270, places=9)
        near = torch.tensor([[[0.1, 0.0, 0.0]]])
        with self.assertRaisesRegex(ValueError, "Fraunhofer"):
            query_from_cartesian_targets(near, geometry)

    def test_hfss_angle_round_trip(self) -> None:
        theta = np.array([90.0, 60.0, 120.0])
        phi = np.array([0.0, 35.0, -70.0])
        direction = hfss_angles_to_direction(theta, phi)
        theta_back, phi_back = direction_to_hfss_angles(direction)
        np.testing.assert_allclose(theta_back, theta, atol=1e-10)
        np.testing.assert_allclose(phi_back, phi, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
