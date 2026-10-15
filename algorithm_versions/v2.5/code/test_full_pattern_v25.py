"""V2.5 完整复方向图兼容层的独立数学与接口回归测试。"""

import math
import unittest
from pathlib import Path

import torch

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    analytic_far_field_weights,
    element_response_from_amp_phase,
    profile_topology_model,
)


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
CONTROLLER_OUTPUT_DIR = VERSION_DIR / "output" / "controller_training"


def _load_published_v24_controller(
    topology: str,
) -> tuple[ArrayGeometryConfig, TopologyAdaptiveBeamformingCNN, dict]:
    checkpoint = torch.load(
        CONTROLLER_OUTPUT_DIR / f"v24_{topology}_low_update_controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = ArrayGeometryConfig(**checkpoint["array_config"])
    controller = TopologyAdaptiveBeamformingCNN(
        config, **checkpoint["model_config"]
    )
    controller.load_state_dict(checkpoint["state_dict"], strict=True)
    controller.eval()
    return config, controller, checkpoint
from full_pattern_v25 import (
    FullPatternPhysics,
    PatternGridConfig,
    V25BeamformingSystem,
    apply_coupling_iq,
    legacy_forward_1x4_config,
    profile_full_pattern,
    weights_iq_from_amplitude_phase,
)


def _direct_complex_reference(
    config: ArrayGeometryConfig,
    grid: PatternGridConfig,
    weights_iq: torch.Tensor,
    *,
    position_errors: torch.Tensor | None = None,
    element_response_iq: torch.Tensor | None = None,
    coupling: torch.Tensor | None = None,
) -> torch.Tensor:
    """不调用被测 steering/field 辅助函数的复数真值实现。"""

    batch = weights_iq.shape[0]
    theta = torch.deg2rad(
        torch.linspace(
            grid.theta_min_deg,
            grid.theta_max_deg,
            grid.theta_points,
            dtype=weights_iq.dtype,
            device=weights_iq.device,
        )
    )
    phi = torch.deg2rad(
        torch.linspace(
            grid.phi_min_deg,
            grid.phi_max_deg,
            grid.phi_points,
            dtype=weights_iq.dtype,
            device=weights_iq.device,
        )
    )
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="xy")
    directions = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ),
        dim=-1,
    )
    positions = torch.tensor(
        [
            (0.0, config.x_spacing * xi, config.y_spacing * yi)
            for xi in range(config.x_num)
            for yi in range(config.y_num)
        ],
        device=weights_iq.device,
        dtype=weights_iq.dtype,
    ).unsqueeze(0).expand(batch, -1, -1)
    if position_errors is not None:
        positions = positions + position_errors.flatten(2).transpose(1, 2)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "ptc,bnc->bptn", directions, positions
    )
    propagation = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    weights = torch.complex(weights_iq[:, 0], weights_iq[:, 1]).flatten(1)
    if coupling is not None:
        weights = torch.einsum("bnm,bm->bn", coupling, weights)
    if element_response_iq is not None:
        response = torch.complex(
            element_response_iq[:, 0], element_response_iq[:, 1]
        ).flatten(1)
        weights = response * weights
    field = torch.einsum("bptn,bn->bpt", propagation, weights)
    return torch.stack((field.real, field.imag), dim=1)


class TestFullPatternPhysics(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260823)
        torch.set_num_threads(1)
        self.grid = PatternGridConfig.legacy_64()

    def test_legacy_grid_contract_and_zero_floor(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        renderer = FullPatternPhysics(config, dtype=torch.float64)
        result = renderer(torch.zeros(1, 2, 1, 4, dtype=torch.float64))
        self.assertEqual(tuple(result["field_iq"].shape), (1, 2, 64, 64))
        self.assertEqual(tuple(result["magnitude"].shape), (1, 64, 64))
        self.assertAlmostEqual(
            math.degrees(float(result["theta_axis_rad"][0])), 60.0, places=11
        )
        self.assertAlmostEqual(
            math.degrees(float(result["theta_axis_rad"][-1])), 120.0, places=11
        )
        self.assertAlmostEqual(
            math.degrees(float(result["phi_axis_rad"][0])), -30.0, places=11
        )
        self.assertAlmostEqual(
            math.degrees(float(result["phi_axis_rad"][-1])), 30.0, places=11
        )
        torch.testing.assert_close(
            result["magnitude_db"],
            torch.full((1, 64, 64), -80.0, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            result["magnitude_db_normalized"],
            torch.full((1, 64, 64), -80.0, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(renderer.trainable_parameter_count, 0)
        self.assertEqual(renderer.state_dict(), {})

    def test_fp64_ideal_matches_independent_reference_for_both_topologies(self) -> None:
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            weights = torch.randn(
                3, 2, config.x_num, config.y_num, dtype=torch.float64
            )
            renderer = FullPatternPhysics(config, self.grid, dtype=torch.float64)
            actual = renderer(weights)["field_iq"]
            expected = _direct_complex_reference(config, self.grid, weights)
            torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_fp64_nonideal_path_matches_independent_reference(self) -> None:
        config = ArrayGeometryConfig.theoretical_4x4()
        batch, elements = 2, config.num_elements
        weights = torch.randn(2, 2, 4, 4, dtype=torch.float64)
        position = 0.015 * torch.randn(2, 3, 4, 4, dtype=torch.float64)
        amplitude = 0.9 + 0.2 * torch.rand(2, 4, 4, dtype=torch.float64)
        phase = 0.12 * torch.randn(2, 4, 4, dtype=torch.float64)
        response = element_response_from_amp_phase(amplitude, phase)
        identity = torch.eye(elements, dtype=torch.complex128).expand(batch, -1, -1)
        coupling = identity + 0.012 * torch.complex(
            torch.randn(batch, elements, elements, dtype=torch.float64),
            torch.randn(batch, elements, elements, dtype=torch.float64),
        )
        renderer = FullPatternPhysics(config, self.grid, dtype=torch.float64)
        actual = renderer(
            weights,
            position_errors=position,
            element_response_iq=response,
            coupling_iq=coupling,
        )["field_iq"]
        expected = _direct_complex_reference(
            config,
            self.grid,
            weights,
            position_errors=position,
            element_response_iq=response,
            coupling=coupling,
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_explicit_iq_coupling_equals_complex_coupling(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        weights = torch.randn(2, 2, 1, 4, dtype=torch.float64)
        coupling = torch.complex(
            torch.randn(2, 4, 4, dtype=torch.float64),
            torch.randn(2, 4, 4, dtype=torch.float64),
        )
        explicit = torch.stack((coupling.real, coupling.imag), dim=1)
        torch.testing.assert_close(
            apply_coupling_iq(weights, coupling),
            apply_coupling_iq(weights, explicit),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_gradient_reaches_weights_and_position_errors(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        small_grid = PatternGridConfig(theta_points=9, phi_points=7)
        renderer = FullPatternPhysics(config, small_grid, dtype=torch.float64)
        weights = torch.randn(2, 2, 1, 4, dtype=torch.float64, requires_grad=True)
        position = (
            0.01 * torch.randn(2, 3, 1, 4, dtype=torch.float64)
        ).requires_grad_()
        field = renderer(weights, position_errors=position)["field_iq"]
        loss = field.square().mean()
        loss.backward()
        for gradient in (weights.grad, position.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_legacy_payload_is_json_compatible_and_axis_explicit(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        renderer = FullPatternPhysics(config, dtype=torch.float64)
        weights = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        weights[:, 0] = 1.0
        payload = renderer.to_legacy_payload(
            renderer(weights),
            weights_iq=weights,
            run_id="unit-test",
            targets={"theta_deg": torch.tensor(90.0)},
            saved_at="2026-08-23 00:00:00",
        )
        self.assertEqual(payload["data_shape"], [64, 64])
        self.assertEqual(len(payload["mag_db_rows"]), 64)
        self.assertEqual(len(payload["mag_db_rows"][0]), 64)
        self.assertEqual(len(payload["phase_rows"]), 64)
        self.assertEqual(payload["header"]["row_axis"], "phi")
        self.assertEqual(payload["header"]["column_axis"], "theta")
        self.assertEqual(payload["run_id"], "unit-test")
        self.assertEqual(payload["targets"]["theta_deg"], 90.0)
        self.assertEqual(payload["dtype"], "float64")
        self.assertEqual(payload["applied_amp_errors"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(payload["applied_phase_errors_rad"], [0.0, 0.0, 0.0, 0.0])

    def test_old_amplitude_phase_adapter_and_historical_geometry(self) -> None:
        config = legacy_forward_1x4_config()
        self.assertEqual(config.topology_name, "1x4")
        self.assertEqual(config.wavelength, 6.83)
        amplitude = torch.tensor([[0.7, 0.8, 0.9, 1.0]], dtype=torch.float64)
        phase = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float64)
        weights = weights_iq_from_amplitude_phase(amplitude, phase, config)
        renderer = FullPatternPhysics(config, dtype=torch.float64)
        adapted = renderer.from_amplitude_phase(amplitude, phase)["field_iq"]
        direct = renderer(weights)["field_iq"]
        torch.testing.assert_close(adapted, direct, rtol=0.0, atol=0.0)

    def test_payload_rejects_grid_mismatch(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        renderer = FullPatternPhysics(config)
        bad_result = {
            "magnitude_db": torch.zeros(1, 5, 7),
            "phase_rad": torch.zeros(1, 5, 7),
        }
        with self.assertRaises(ValueError):
            renderer.to_legacy_payload(
                bad_result, weights_iq=torch.zeros(1, 2, 1, 4)
            )

    def test_nan_and_bad_shapes_are_rejected(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        renderer = FullPatternPhysics(config)
        with self.assertRaises(ValueError):
            renderer(torch.zeros(1, 2, 4))
        weights = torch.zeros(1, 2, 1, 4)
        weights[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            renderer(weights)

    def test_mixed_precision_is_rejected_instead_of_silent_upcast(self) -> None:
        renderer = FullPatternPhysics(
            ArrayGeometryConfig.hardware_1x4(), dtype=torch.float32
        )
        with self.assertRaises(ValueError):
            renderer(torch.zeros(1, 2, 1, 4, dtype=torch.float64))
        renderer.double()
        with self.assertRaises(ValueError):
            renderer(torch.zeros(1, 2, 1, 4, dtype=torch.float64))
        renderer64 = FullPatternPhysics(
            ArrayGeometryConfig.hardware_1x4(), dtype=torch.float64
        )
        weights64 = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        with self.assertRaises(ValueError):
            renderer64(
                weights64,
                position_errors=torch.zeros(1, 3, 1, 4, dtype=torch.float32),
            )
        with self.assertRaises(ValueError):
            renderer64(
                weights64,
                coupling_iq=torch.eye(4, dtype=torch.complex64).unsqueeze(0),
            )

    def test_normalized_db_handles_tiny_pattern_and_deep_null(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        grid = PatternGridConfig(theta_points=3, phi_points=3)
        renderer = FullPatternPhysics(config, grid, dtype=torch.float64)
        weights = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        weights[0, 0, 0, 0] = 2.0e-12
        weights[0, 0, 0, 1] = -2.0e-12
        normalized = renderer(weights)["magnitude_db_normalized"]
        self.assertAlmostEqual(float(normalized.amax()), 0.0, places=10)
        torch.testing.assert_close(
            normalized[0, :, 1],
            torch.full((3,), -80.0, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-10,
        )


class TestV25SystemAndComplexity(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260823)
        torch.set_num_threads(1)

    def test_wrapper_does_not_change_v24_controller_output(self) -> None:
        for topology in ("4x4", "1x4"):
            config, controller, checkpoint = _load_published_v24_controller(topology)
            self.assertEqual(checkpoint["algorithm_version"], "2.4.0")
            self.assertEqual(checkpoint["selected_update"], "zero_shot")
            renderer = FullPatternPhysics(config)
            system = V25BeamformingSystem(controller, renderer)
            target = torch.tensor(
                [[math.radians(82.0), math.radians(-17.0)]], dtype=torch.float32
            )
            initial = analytic_far_field_weights(config, target)
            calibration = torch.zeros_like(initial)
            calibration[:, 0] = 1.0
            calibration[:, 1] = 0.02
            position = torch.zeros(1, 3, config.x_num, config.y_num)
            position[:, 2] = 0.001 * config.wavelength
            state = torch.tensor([[0.01, -0.02, 0.03, -0.04]])
            inputs = {
                "calibration_iq": calibration,
                "position_errors": position,
                "state": state,
                "quantize_output": False,
            }
            direct = controller(initial, target, **inputs)
            combined = system(
                initial, target, **inputs, render_pattern=True
            )
            for key in ("weights_iq", "prequant_weights_iq", "delta_iq"):
                torch.testing.assert_close(combined[key], direct[key], rtol=0.0, atol=0.0)
            self.assertEqual(
                tuple(combined["full_pattern"]["field_iq"].shape),
                (1, 2, 64, 64),
            )
            without_pattern = system(
                initial, target, **inputs, render_pattern=False
            )
            self.assertNotIn("full_pattern", without_pattern)
            self.assertFalse(
                any(key.startswith("renderer.") for key in system.state_dict())
            )

    def test_static_profiles_have_exact_expected_counts(self) -> None:
        expected = {
            "4x4": (262_144, 1_024, 64, 524_288, 67_104, 329_248, 330_336),
            "1x4": (65_536, 64, 16, 131_072, 15_264, 80_800, 80_880),
        }
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            controller = TopologyAdaptiveBeamformingCNN(
                config, width=16, num_blocks=3, global_context="none"
            )
            controller_macs = profile_topology_model(controller)["macs_per_sample"]
            profile = profile_full_pattern(config, controller_macs=controller_macs)
            (
                renderer_macs,
                coupling_macs,
                response_macs,
                cache_bytes,
                expected_controller,
                end_to_end,
                nonideal_end_to_end,
            ) = expected[config.topology_name]
            self.assertEqual(controller_macs, expected_controller)
            self.assertEqual(profile["renderer_real_macs"], renderer_macs)
            self.assertEqual(profile["optional_coupling_real_macs"], coupling_macs)
            self.assertEqual(
                profile["optional_element_response_real_macs"], response_macs
            )
            self.assertEqual(profile["cached_steering_bytes"], cache_bytes)
            self.assertEqual(profile["end_to_end_real_macs"], end_to_end)
            self.assertEqual(
                profile["end_to_end_nonideal_contraction_real_macs"],
                nonideal_end_to_end,
            )
            self.assertEqual(profile["trainable_parameters"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
