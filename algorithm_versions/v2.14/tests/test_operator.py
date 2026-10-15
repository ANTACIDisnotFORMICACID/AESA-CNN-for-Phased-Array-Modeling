import torch
import importlib.util
import json
from pathlib import Path

from v210.contracts import QueryBatch
from v212.basis import FiltrationSSHTransform
from v214.operator import CrossBranchFusionOperator


def _models():
    direction = torch.tensor([
        [1.0, 0.0, 0.0], [0.9, 0.1, 0.424264], [0.8, -0.3, 0.519615],
        [0.7, 0.5, 0.509902], [0.6, -0.6, 0.52915], [0.5, 0.7, 0.509902],
    ])
    direction = direction / direction.norm(dim=1, keepdim=True)
    transform = FiltrationSSHTransform.fit(direction.double(), degree=2)
    torch.manual_seed(21401)
    off = CrossBranchFusionOperator(
        ssh_transform=transform, ssh_seed=21401, cross_branch_access=False
    )
    torch.manual_seed(21401)
    on = CrossBranchFusionOperator(
        ssh_transform=transform, ssh_seed=21401, cross_branch_access=True
    )
    return direction, off, on


def test_parameter_matched_and_zero_start_equivalent():
    direction, off, on = _models()
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in on.parameters())
    query = QueryBatch(direction[None])
    off_envelope, off_terms = off.envelope(query)
    on_envelope, on_terms = on.envelope(query)
    torch.testing.assert_close(off_envelope, on_envelope)
    assert torch.count_nonzero(off_terms["fusion_delta_iq"]) == 0
    assert torch.count_nonzero(on_terms["fusion_delta_iq"]) == 0


def test_factorized_grid_closes_at_identical_direction():
    direction, _, model = _models()
    generator = torch.Generator().manual_seed(21402)
    command = torch.complex(
        torch.randn(3, 16, generator=generator),
        torch.randn(3, 16, generator=generator),
    )
    direct = model.forward_command_grid(command, direction)
    factorized = model.forward_factorized_grid(command, direction, direction)
    torch.testing.assert_close(direct, factorized, atol=1e-6, rtol=1e-6)


def test_coverage_selector_is_deterministic_unique_and_leaves_holdout():
    script = Path(__file__).resolve().parents[1] / "tools" / "run_coverage_causal_validation.py"
    spec = importlib.util.spec_from_file_location("coverage_validation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    theta = torch.deg2rad(torch.linspace(30.0, 39.0, 10))
    phi = torch.deg2rad(torch.linspace(0.0, 9.0, 10))
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    direction = torch.stack((torch.sin(tt) * torch.cos(pp), torch.sin(tt) * torch.sin(pp), torch.cos(tt)), dim=-1).reshape(-1, 3)
    first = module._maximin_per_cell(direction, 10)
    second = module._maximin_per_cell(direction, 10)
    assert torch.equal(first, second)
    assert len(first) == len(torch.unique(first)) == 10
    assert len(direction) - len(first) == 90


def test_completed_coverage_summary_closes_preregistered_contract():
    summary_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts" / "coverage_causal_20260913" / "COVERAGE_CAUSAL_SUMMARY.json"
    )
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["status"] == "coverage_hypothesis_pass"
    assert report["external_test_used"] is False
    assert report["nested_train_only"] is True
    assert report["budget"]["control_training_directions"] == report["budget"]["coverage_training_directions"]
    assert len(report["paired_results"]) == 3
    assert all(report["acceptance"].values())
