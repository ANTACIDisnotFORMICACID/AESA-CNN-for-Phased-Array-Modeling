from __future__ import annotations

import torch

from v212.representation import (
    branch_energy_decomposition,
    per_direction_nmse_db,
    tangent_frame,
    tangent_jacobian_norm,
    tangent_perturbations,
    weighted_projection,
)


def test_weighted_projection_handles_nonorthogonal_basis() -> None:
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    basis = torch.stack((torch.ones_like(x), 1.0 + x), dim=1)
    values = (2.0 - 0.4 * x)[:, None].to(torch.complex128)
    projection = weighted_projection(basis, values)
    torch.testing.assert_close(projection, values, atol=1e-10, rtol=1e-10)


def test_branch_energy_decomposition_detects_exact_low_mode() -> None:
    x = torch.linspace(-1.0, 1.0, 21, dtype=torch.float64)
    basis = torch.stack((torch.ones_like(x), x), dim=1)
    ssh = torch.zeros((len(x), 1), dtype=torch.complex128)
    fourier = (1.0 + 2.0j) * x[:, None]
    report = branch_energy_decomposition(basis, ssh, fourier)
    assert abs(report["fourier_low_fraction"] - 1.0) < 1e-10
    assert report["fourier_high_energy"] < 1e-20


def test_tangent_frame_and_perturbations_remain_on_sphere() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.9327379]])
    q = q / q.norm(dim=1, keepdim=True)
    first, second = tangent_frame(q)
    torch.testing.assert_close((q * first).sum(1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close((q * second).sum(1), torch.zeros(2), atol=1e-6, rtol=0)
    for perturbed in tangent_perturbations(q, step_rad=1e-3):
        torch.testing.assert_close(perturbed.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0)


def test_tangent_jacobian_norm_for_known_two_axis_derivatives() -> None:
    center = torch.ones((3, 2))
    step = 0.01
    plus_first = center + step * 2.0
    minus_first = center - step * 2.0
    plus_second = center + step * 3.0
    minus_second = center - step * 3.0
    raw, normalized = tangent_jacobian_norm(
        plus_first, minus_first, plus_second, minus_second, center, step_rad=step
    )
    expected = torch.full((3,), (2 * (2.0**2 + 3.0**2)) ** 0.5)
    torch.testing.assert_close(raw, expected)
    torch.testing.assert_close(normalized, expected / (2.0**0.5))


def test_per_direction_nmse_db() -> None:
    target = torch.ones((2, 4))
    prediction = torch.stack((torch.zeros(4), torch.full((4,), 0.9)))
    value = per_direction_nmse_db(prediction, target)
    torch.testing.assert_close(value, torch.tensor([0.0, -20.0]), atol=1e-5, rtol=0)
