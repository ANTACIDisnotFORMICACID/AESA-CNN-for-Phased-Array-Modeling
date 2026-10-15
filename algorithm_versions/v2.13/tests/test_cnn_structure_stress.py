from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_cnn_structure_stress_audit.py"
SPEC = importlib.util.spec_from_file_location("cnn_structure_stress", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rotations_preserve_unit_norm_and_angle():
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.0, 0.6]], dtype=torch.float32)
    for shifted in MODULE._rotations(direction, 0.5):
        assert torch.allclose(shifted.norm(dim=1), torch.ones(2), atol=1e-6)
        angle = torch.rad2deg(torch.acos((shifted * direction).sum(dim=1).clamp(-1.0, 1.0)))
        assert torch.allclose(angle, torch.full((2,), 0.5), atol=1e-3)


def test_parameter_groups_cover_all_parameters():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b0_raw = torch.nn.Parameter(torch.zeros(1))
            self.ssh_alpha = torch.nn.Parameter(torch.zeros(2))
            self.residual = torch.nn.Linear(2, 3)

    groups = MODULE._parameter_groups(Tiny())
    assert groups["total"] == 12
    assert sum(groups[name] for name in groups if name not in {"total"}) == groups["total"]


def test_worst_cells_prioritizes_larger_error():
    direction = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9848078, 0.1736482, 0.0],
        [0.8660254, 0.0, 0.5],
    ]).numpy()
    rows = MODULE._worst_cells(direction, [-20.0, -10.0, 3.0], count=1)
    assert rows[0]["median_per_direction_nmse_db"] == 3.0


def test_rotations_report_boundary_crossing_explicitly():
    direction = torch.tensor([[0.001, 0.9999995, 0.0]], dtype=torch.float32)
    shifted = MODULE._rotations(direction, 1.0)
    assert any(bool(torch.any(item[:, 0] <= 0)) for item in shifted)
