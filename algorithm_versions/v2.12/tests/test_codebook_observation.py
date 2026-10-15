from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from v212.codebook import build_baseline_codebook, recover_pair_entry
from v212.observation import ObservationKind, ReceiverObservation


def test_baseline_codebook_counts_norms_splits_and_roundtrip() -> None:
    codebook = build_baseline_codebook()
    assert codebook.kinds.count("single") == 16
    assert codebook.kinds.count("pair_0") == 24
    assert codebook.kinds.count("pair_90") == 24
    assert codebook.kinds.count("random") == 32
    torch.testing.assert_close(codebook.weights.norm(dim=-1), torch.ones(96))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "codebook.json"
        codebook.save(path)
        loaded = codebook.load(path)
    torch.testing.assert_close(loaded.weights, codebook.weights)
    assert loaded.splits == codebook.splits


def test_normalized_polarization_identity_recovers_complex_q_entry() -> None:
    g = torch.tensor([1.2 - 0.4j, -0.3 + 0.9j])
    p_left, p_right = g.abs().square()
    p_zero = ((g[0] + g[1]) / 2**0.5).abs().square()
    p_ninety = ((g[0] + 1j * g[1]) / 2**0.5).abs().square()
    recovered = recover_pair_entry(p_left, p_right, p_zero, p_ninety)
    torch.testing.assert_close(recovered, g[0].conj() * g[1])


def test_observation_guard_rejects_uncalibrated_free_space_claim() -> None:
    effective = ReceiverObservation(ObservationKind.ENVIRONMENT_EFFECTIVE, 47e9)
    try:
        effective.validate_output_label("free_space_operator")
    except ValueError as error:
        assert "free_space_operator" in str(error)
    else:
        raise AssertionError("uncalibrated environment response was mislabeled as free space")
    hfss = ReceiverObservation(ObservationKind.HFSS_EPHI, 47e9)
    hfss.validate_output_label("free_space_operator")
    field = torch.randn(5, 2, 16, dtype=torch.complex64)
    torch.testing.assert_close(hfss.project(field), field[:, 1])
    effective_field = torch.randn(5, 16, dtype=torch.complex64)
    torch.testing.assert_close(effective.project(effective_field), effective_field)
