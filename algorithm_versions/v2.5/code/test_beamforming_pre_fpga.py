"""FPGA 前算法 V2.2+ 的数学、拓扑和接口回归测试。"""

import copy
import math
import unittest

import torch

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    analytic_far_field_weights,
    array_positions,
    element_response_from_amp_phase,
    field_from_weights_iq,
    profile_topology_model,
    steering_at_directions_iq,
    steering_iq,
)
from physics_initialized_frontend import (
    PhysicsInitializedApertureEstimator,
    align_global_phase,
    analytic_near_field_weights,
    complex_matrix_to_real_block,
    complex_to_iq,
    design_probe_grid,
    far_field_matrix,
    iq_to_complex,
    near_field_green_matrix,
    near_to_far_transform,
    regularized_pseudoinverse,
)
from sparse_beamforming_v24 import (
    ComplexDiskInt8Quantizer,
    PreFpgaAlgorithmContract,
    SparseFieldObjective,
)
from experiment_common import SOURCE, make_samples
from run_v24_sparse_end_to_end_study import (
    calibrated_nominal_sparse_coefficients,
    d_optimal_fixed_angles,
)


class TestTopologyPhysics(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260821)
        torch.set_num_threads(1)

    def test_certified_geometries_and_numbering(self) -> None:
        planar = ArrayGeometryConfig.theoretical_4x4()
        linear = ArrayGeometryConfig.hardware_1x4()
        self.assertEqual(planar.num_elements, 16)
        self.assertEqual(linear.num_elements, 4)
        p4 = array_positions(planar, dtype=torch.float64)
        p1 = array_positions(linear, dtype=torch.float64)
        self.assertEqual(tuple(p4.shape), (16, 3))
        self.assertEqual(tuple(p1.shape), (4, 3))
        torch.testing.assert_close(
            p1,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 15.0],
                    [0.0, 0.0, 30.0],
                    [0.0, 0.0, 45.0],
                ],
                dtype=torch.float64,
            ),
            rtol=0.0,
            atol=0.0,
        )

    def test_analytic_gain_equals_element_count_for_both_topologies(self) -> None:
        targets = torch.tensor(
            [
                [math.radians(67.2), math.radians(-22.5)],
                [math.radians(93.0), math.radians(11.0)],
                [math.radians(116.0), math.radians(27.0)],
            ],
            dtype=torch.float64,
        )
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            weights = analytic_far_field_weights(config, targets)
            steering = steering_at_directions_iq(config, targets)
            field = field_from_weights_iq(weights, steering)
            gain = torch.sqrt(field[:, 0].square() + field[:, 1].square())
            torch.testing.assert_close(
                gain,
                torch.full_like(gain, float(config.num_elements)),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    def test_sample_dependent_position_and_response_match_complex_sum(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        target = torch.tensor([[math.radians(81.0), math.radians(19.0)]], dtype=torch.float64)
        weights = torch.randn(1, 2, 1, 4, dtype=torch.float64)
        position = torch.zeros(1, 3, 1, 4, dtype=torch.float64)
        position[:, 2, 0] = torch.tensor([0.01, -0.02, 0.03, -0.04], dtype=torch.float64)
        amplitude = torch.tensor([[[0.90, 1.05, 0.97, 1.08]]], dtype=torch.float64)
        phase = torch.tensor([[[0.10, -0.13, 0.07, -0.05]]], dtype=torch.float64)
        response = element_response_from_amp_phase(amplitude, phase)
        steering = steering_at_directions_iq(
            config, target, position_errors=position
        )
        actual = field_from_weights_iq(
            weights, steering, element_response_iq=response
        )[0]

        complex_weights = torch.complex(weights[:, 0], weights[:, 1]).flatten()
        complex_response = torch.complex(response[:, 0], response[:, 1]).flatten()
        positions = array_positions(config, dtype=torch.float64) + position.flatten(2).transpose(1, 2)[0]
        theta, phi = target[0]
        direction = torch.stack(
            (
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            )
        )
        propagation = torch.exp(
            1j * (2.0 * math.pi / config.wavelength) * (positions @ direction)
        )
        expected = torch.sum(complex_weights * complex_response * propagation)
        torch.testing.assert_close(actual[0], expected.real, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(actual[1], expected.imag, rtol=1.0e-12, atol=1.0e-12)


class TestTopologyController(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260821)
        torch.set_num_threads(1)
        self.target = torch.tensor([[math.radians(76.0), math.radians(-12.0)]])

    def _exercise(self, config: ArrayGeometryConfig, global_context: str) -> None:
        initial = analytic_far_field_weights(config, self.target)
        model = TopologyAdaptiveBeamformingCNN(
            config,
            width=12,
            num_blocks=2,
            global_context=global_context,
            output_bits=8,
        )
        result = model(initial, self.target, quantize_output=False)
        torch.testing.assert_close(result["weights_iq"], initial, rtol=0.0, atol=0.0)
        self.assertEqual(
            tuple(result["weights_iq"].shape),
            (1, 2, config.x_num, config.y_num),
        )
        expected_kernel = (3, 3) if config.x_num > 1 else (1, 3)
        self.assertEqual(model.kernel_size, expected_kernel)
        self.assertFalse(
            any(isinstance(module, nn_pool_types()) for module in model.modules())
        )

        with torch.no_grad():
            model.head.bias.fill_(12.0)
        for quantize in (False, True):
            weights = model(
                initial, self.target, quantize_output=quantize
            )["weights_iq"]
            amplitude = torch.sqrt(weights[:, 0].square() + weights[:, 1].square())
            self.assertLessEqual(float(amplitude.max().detach()), 1.0 + 1.0e-6)

    def test_4x4_uses_2d_kernel_without_destructive_gap(self) -> None:
        self._exercise(ArrayGeometryConfig.theoretical_4x4(), "none")
        self._exercise(ArrayGeometryConfig.theoretical_4x4(), "broadcast")

    def test_1x4_uses_anisotropic_kernel_without_wasted_rows(self) -> None:
        self._exercise(ArrayGeometryConfig.hardware_1x4(), "none")
        self._exercise(ArrayGeometryConfig.hardware_1x4(), "broadcast")

    def test_profiles_are_more_than_one_order_below_legacy(self) -> None:
        legacy_parameters = 2_352_770
        legacy_macs = 31_932_928
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            model = TopologyAdaptiveBeamformingCNN(
                config, width=16, num_blocks=3, global_context="none"
            )
            profile = profile_topology_model(model)
            self.assertLess(profile["total_parameters"], legacy_parameters // 10)
            self.assertLess(profile["macs_per_sample"], legacy_macs // 10)

    def test_export_contract_is_explicit_and_versioned(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        model = TopologyAdaptiveBeamformingCNN(config)
        contract = model.export_contract()
        self.assertEqual(contract["algorithm_version"], "2.2.0")
        self.assertEqual(contract["topology"], "1x4")
        self.assertEqual(contract["kernel_size"], [1, 3])
        self.assertEqual(contract["output_shape"], ["B", 2, 1, 4])


class TestPhysicsInitialisedFrontend(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23001)
        torch.set_num_threads(1)

    def test_complex_real_block_is_exact(self) -> None:
        matrix = torch.randn(5, 7, dtype=torch.complex64)
        value = torch.randn(3, 7, dtype=torch.complex64)
        block = complex_matrix_to_real_block(matrix)
        vector = torch.cat((value.real, value.imag), dim=1)
        actual = vector @ block.T
        expected = value @ matrix.T
        torch.testing.assert_close(actual[:, :5], expected.real, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(actual[:, 5:], expected.imag, rtol=1.0e-5, atol=1.0e-6)

    def test_physics_fc_and_near_to_far_transform(self) -> None:
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            probes = design_probe_grid(config)
            near = near_field_green_matrix(config, probes)
            self.assertEqual(int(torch.linalg.matrix_rank(near)), config.num_elements)
            inverse = regularized_pseudoinverse(near, ridge=1.0e-6)
            estimator = PhysicsInitializedApertureEstimator(
                inverse, adapter_rank=0, train_base=False
            )
            aperture = torch.randn(4, config.num_elements, dtype=torch.complex64)
            measurements = aperture @ near.T
            estimate = iq_to_complex(estimator(complex_to_iq(measurements)))
            torch.testing.assert_close(estimate, aperture, rtol=2.0e-4, atol=2.0e-5)

            theta = torch.linspace(math.radians(60.0), math.radians(120.0), 7)
            phi = torch.linspace(math.radians(-30.0), math.radians(30.0), 7)
            far = far_field_matrix(config, theta, phi)
            transform = near_to_far_transform(near, far, ridge=1.0e-6)
            direct = aperture @ far.T
            transformed = measurements @ transform.T
            torch.testing.assert_close(transformed, direct, rtol=3.0e-4, atol=3.0e-5)

    def test_near_weights_converge_to_far_weights(self) -> None:
        config = ArrayGeometryConfig.theoretical_4x4()
        theta = torch.tensor([math.radians(78.0)], dtype=torch.float64)
        phi = torch.tensor([math.radians(13.0)], dtype=torch.float64)
        direction = torch.stack(
            (
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ),
            dim=1,
        )
        positions = array_positions(config, dtype=torch.float64)
        centre = positions.mean(dim=0)
        aperture = torch.cdist(positions, positions).amax()
        fraunhofer = 2.0 * aperture.square() / config.wavelength
        point = centre + 10.0 * fraunhofer * direction
        near = iq_to_complex(analytic_near_field_weights(config, point).flatten(2))
        far = iq_to_complex(analytic_far_field_weights(
            config, torch.stack((theta, phi), dim=1)
        ).flatten(2))
        aligned = align_global_phase(far, near)
        phase_error = torch.angle(aligned * torch.conj(far)).abs().amax()
        self.assertLessEqual(float(torch.rad2deg(phase_error)), 1.25)
        steering = torch.polar(
            torch.ones(config.num_elements, dtype=torch.float64),
            (2.0 * math.pi / config.wavelength) * (positions @ direction[0]),
        )
        gain = torch.abs((near[0] * steering).sum())
        loss_db = 20.0 * torch.log10(gain / config.num_elements)
        self.assertGreaterEqual(float(loss_db), -0.002)


class TestSparseAndIntegerReference(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(24001)
        torch.set_num_threads(1)

    def test_int8_codes_strictly_obey_complex_disk(self) -> None:
        quantizer = ComplexDiskInt8Quantizer()
        values = 3.0 * torch.randn(128, 2, 4, 4)
        result = quantizer(values)
        codes = result["codes_iq"].to(torch.int32)
        radius_sq = codes[:, 0].square() + codes[:, 1].square()
        self.assertTrue(bool(torch.all(radius_sq <= 127 * 127)))
        self.assertEqual(int(result["max_integer_radius_sq"]), int(radius_sq.max()))
        torch.testing.assert_close(
            result["dequantized_iq"], codes.float() / 127.0, rtol=0.0, atol=0.0
        )
        for invalid in (float("nan"), float("inf"), -float("inf")):
            bad = torch.zeros(1, 2, 1, 4)
            bad[0, 0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                quantizer(bad)

    def test_int8_codes_are_nearest_feasible_lattice_points(self) -> None:
        quantizer = ComplexDiskInt8Quantizer()
        values = 1.4 * torch.randn(8, 2, 1, 1)
        codes = quantizer(values)["codes_iq"].to(torch.float64).reshape(8, 2)
        axis = torch.arange(-127, 128, dtype=torch.float64)
        real, imag = torch.meshgrid(axis, axis, indexing="ij")
        lattice = torch.stack((real.flatten(), imag.flatten()), dim=1)
        lattice = lattice[lattice.square().sum(dim=1) <= 127 * 127]
        target = (values[:, :, 0, 0] * 127.0).to(torch.float64)
        chosen_distance = (codes - target).square().sum(dim=1)
        optimum_distance = (lattice[None] - target[:, None]).square().sum(dim=2).amin(dim=1)
        torch.testing.assert_close(chosen_distance, optimum_distance, rtol=0.0, atol=1.0e-9)

    def test_calibrated_surrogate_gradient_coefficients_ignore_true_tensors(self) -> None:
        config = ArrayGeometryConfig.theoretical_4x4()
        samples = make_samples(4, SOURCE, config, seed=24500, targets_per_profile=2)
        fixed = d_optimal_fixed_angles(config, 7)
        expected = calibrated_nominal_sparse_coefficients(config, samples, fixed)
        altered = copy.deepcopy(samples)
        altered.true_position = 100.0 * torch.randn_like(altered.true_position)
        altered.true_response = 100.0 * torch.randn_like(altered.true_response)
        altered.coupling = 100.0 * torch.randn_like(altered.coupling)
        actual = calibrated_nominal_sparse_coefficients(config, altered, fixed)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_sparse_losses_handle_phase_reference_explicitly(self) -> None:
        predicted = torch.randn(6, 12, dtype=torch.complex64, requires_grad=True)
        phase = torch.linspace(-2.0, 2.0, 6)
        rotated = predicted.detach() * torch.polar(torch.ones(6), phase)[:, None]
        coherent = SparseFieldObjective(mode="coherent_iq")(predicted, rotated)
        invariant = SparseFieldObjective(mode="phase_invariant_iq")(predicted, rotated)
        power = SparseFieldObjective(mode="power_only")(predicted, rotated)
        power_real = SparseFieldObjective(mode="power_only")(
            predicted, rotated.abs().square()
        )
        self.assertGreater(float(coherent.detach()), 1.0e-3)
        self.assertLess(float(invariant.detach()), 1.0e-10)
        self.assertLess(float(power.detach()), 1.0e-10)
        self.assertLess(float(power_real.detach()), 1.0e-10)
        (coherent + invariant + power + power_real).backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_prefpga_contract_covers_both_topologies(self) -> None:
        c4 = PreFpgaAlgorithmContract.create(
            ArrayGeometryConfig.theoretical_4x4(), 32, "coherent_iq"
        ).to_dict()
        c1 = PreFpgaAlgorithmContract.create(
            ArrayGeometryConfig.hardware_1x4(), 8, "phase_invariant_iq"
        ).to_dict()
        self.assertEqual(c4["controller_output_shape"], ["B", 2, 4, 4])
        self.assertEqual(c1["controller_output_shape"], ["B", 2, 1, 4])
        self.assertEqual(c4["integer_output_constraint"], "qI^2+qQ^2<=127^2")
        self.assertEqual(c4["total_measurement_count"], 32)
        self.assertEqual(c4["fixed_measurement_count"], 31)
        self.assertEqual(c4["dynamic_target_measurement_count"], 1)


def nn_pool_types():
    return (
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
    )


if __name__ == "__main__":
    unittest.main()
