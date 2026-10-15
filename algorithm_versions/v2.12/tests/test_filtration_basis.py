from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from v212.basis import FiltrationSSHTransform, real_spherical_harmonics


def directions(count: int = 512) -> torch.Tensor:
    generator = torch.Generator().manual_seed(212)
    value = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    value[:, 0] = value[:, 0].abs().clamp_min(1e-3)
    return value / value.norm(dim=-1, keepdim=True)


def test_weighted_filtration_is_orthonormal_and_preserves_prefix_span() -> None:
    q = directions()
    transform = FiltrationSSHTransform.fit(q, 4, truncation_ratio=1e-10)
    phi = transform.evaluate(q)
    gram = phi.T @ phi / len(q)
    torch.testing.assert_close(gram, torch.eye(phi.shape[1], dtype=gram.dtype), atol=2e-8, rtol=2e-8)
    assert transform.transformed_condition < 1.00001
    assert transform.retained_condition <= 1e10

    raw = real_spherical_harmonics(q, 4)
    raw_end = 0
    phi_end = 0
    for degree, rank in enumerate(transform.ranks):
        raw_end += 2 * degree + 1
        phi_end += rank
        solution = torch.linalg.lstsq(phi[:, :phi_end], raw[:, :raw_end]).solution
        relative = (phi[:, :phi_end] @ solution - raw[:, :raw_end]).norm() / raw[:, :raw_end].norm()
        assert float(relative) < 1e-8


def test_equal_loss_measure_is_not_double_weighted_and_artifact_roundtrips() -> None:
    q = directions(256)
    implicit = FiltrationSSHTransform.fit(q, 3)
    explicit = FiltrationSSHTransform.fit(q, 3, loss_weights=torch.ones(len(q)))
    gram_implicit = implicit.evaluate(q) @ implicit.evaluate(q).T
    gram_explicit = explicit.evaluate(q) @ explicit.evaluate(q).T
    torch.testing.assert_close(gram_implicit, gram_explicit, atol=1e-8, rtol=1e-8)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "basis.npz"
        implicit.save(path)
        loaded = FiltrationSSHTransform.load(path)
    torch.testing.assert_close(loaded.transform, implicit.transform)
    assert loaded.ranks == implicit.ranks
    assert len(loaded.discarded_mode_vectors) == 4
