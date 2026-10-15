from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.per_array_models import (  # noqa: E402
    GeometryMatchedOperator,
    GeometryMatchedResidual,
    parameter_count,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21720)
    direction = torch.randn(2, 7, 3)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    arrays = {}
    all_passed = True
    for array_id in ("a1x4", "a2x4", "a4x4", "a1x8"):
        control = GeometryMatchedResidual(array_id, "control")
        candidate = GeometryMatchedResidual(array_id, "candidate")
        control_output = control(direction)
        candidate_output = candidate(direction)
        control_parameters = parameter_count(control)
        candidate_parameters = parameter_count(candidate)
        difference = candidate_parameters / control_parameters - 1.0
        control_operator = GeometryMatchedOperator(array_id, "control")
        candidate_operator = GeometryMatchedOperator(array_id, "candidate")
        element_count = control.config.rows * control.config.cols
        command_iq = torch.randn(2, 3, element_count, 2)
        commands = torch.complex(command_iq[..., 0], command_iq[..., 1])
        commands = commands / torch.linalg.vector_norm(commands, dim=-1, keepdim=True)
        control_complex = control_operator.operator(direction)
        candidate_complex = candidate_operator.operator(direction)
        control_amplitude = control_operator(direction, commands)
        candidate_amplitude = candidate_operator(direction, commands)
        full_control_parameters = parameter_count(control_operator)
        full_candidate_parameters = parameter_count(candidate_operator)
        full_difference = full_candidate_parameters / full_control_parameters - 1.0
        gradient_checks = {}
        target = torch.complex(
            torch.randn_like(control_complex.real), torch.randn_like(control_complex.real)
        )
        for name, model in (("control", control_operator), ("candidate", candidate_operator)):
            loss = torch.mean(torch.abs(model.operator(direction) - target) ** 2)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            gradient_checks[f"{name}_gradient_finite"] = bool(
                gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            )
            gradient_checks[f"{name}_gradient_nonzero"] = bool(
                any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
            )
        expected = (2, 7, 2, control.config.rows, control.config.cols)
        checks = {
            "control_shape": list(control_output.shape) == list(expected),
            "candidate_shape": list(candidate_output.shape) == list(expected),
            "control_finite": bool(torch.isfinite(control_output).all()),
            "candidate_finite": bool(torch.isfinite(candidate_output).all()),
            "control_zero_start": bool(torch.count_nonzero(control_output) == 0),
            "candidate_zero_start": bool(torch.count_nonzero(candidate_output) == 0),
            "parameter_difference_within_0p5_percent": abs(difference) <= 0.005,
            "full_operator_shape": list(control_complex.shape) == [2, 7, element_count],
            "full_amplitude_shape": list(control_amplitude.shape) == [2, 3, 7],
            "full_operator_finite": bool(
                torch.isfinite(control_complex).all() and torch.isfinite(candidate_complex).all()
            ),
            "full_amplitude_finite": bool(
                torch.isfinite(control_amplitude).all() and torch.isfinite(candidate_amplitude).all()
            ),
            "full_candidate_control_same_zero_start": bool(
                torch.equal(control_complex, candidate_complex)
                and torch.equal(control_amplitude, candidate_amplitude)
            ),
            "full_parameter_difference_within_0p5_percent": abs(full_difference) <= 0.005,
            "full_separate_parameter_storage": not any(
                left.data_ptr() == right.data_ptr()
                for left, right in zip(control_operator.parameters(), candidate_operator.parameters())
            ),
            "paired_common_initialization": bool(
                torch.equal(control_operator.angular_hidden.weight, candidate_operator.angular_hidden.weight)
                and torch.equal(control_operator.angular_output.bias, candidate_operator.angular_output.bias)
                and torch.equal(control_operator.residual.stem.weight, candidate_operator.residual.stem.weight)
            ),
            **gradient_checks,
            "separate_parameter_storage": not any(
                left.data_ptr() == right.data_ptr()
                for left, right in zip(control.parameters(), candidate.parameters())
            ),
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        arrays[array_id] = {
            "passed": passed,
            "control": control.metadata(),
            "candidate": candidate.metadata(),
            "control_parameters": control_parameters,
            "candidate_parameters": candidate_parameters,
            "parameter_difference_fraction": difference,
            "full_control_parameters": full_control_parameters,
            "full_candidate_parameters": full_candidate_parameters,
            "full_parameter_difference_fraction": full_difference,
            "checks": checks,
        }
    payload = {
        "schema": "v217-per-array-model-static-audit-1",
        "passed": all_passed,
        "scope": "architecture_shape_zero_start_parameter_match_and_gradient_smoke_only",
        "training_performed": False,
        "validation_read": False,
        "blind_test_read": False,
        "arrays": arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
