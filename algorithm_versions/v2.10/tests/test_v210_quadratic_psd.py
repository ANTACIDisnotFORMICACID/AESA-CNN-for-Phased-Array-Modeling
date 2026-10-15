from __future__ import annotations

import unittest

import torch

from v210.models.quadratic_psd import (
    V210LowRankPSD,
    V210VectorBaseline,
    gains_from_factor,
    quadratic_form,
)


class QuadraticPSDTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.query = torch.randn(7, 3)
        self.query = self.query / self.query.norm(dim=-1, keepdim=True)
        self.command = torch.randn(5, 16, dtype=torch.complex64)
        self.command = self.command / self.command.norm(dim=-1, keepdim=True)

    def test_psd_and_quadratic_contraction(self) -> None:
        factor = V210LowRankPSD()(self.query)
        matrix = quadratic_form(factor)
        eigenvalues = torch.linalg.eigvalsh(matrix).real
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-6)
        direct = gains_from_factor(factor, self.command)
        quadratic = torch.einsum(
            "cn,qnm,cm->cq", self.command.conj(), matrix, self.command
        ).real
        self.assertTrue(torch.allclose(direct, quadratic, atol=1e-6))

    def test_baseline_is_transverse(self) -> None:
        factor = V210VectorBaseline()(self.query)
        longitudinal = torch.sum(self.query[:, :, None] * factor, dim=1)
        self.assertLess(float(longitudinal.abs().max()), 1e-5)

    def test_command_common_phase_invariance(self) -> None:
        factor = V210LowRankPSD()(self.query)
        phase = torch.exp(torch.tensor(0.7j))
        self.assertTrue(
            torch.allclose(
                gains_from_factor(factor, self.command),
                gains_from_factor(factor, self.command * phase),
                atol=1e-7,
            )
        )


if __name__ == "__main__":
    unittest.main()

