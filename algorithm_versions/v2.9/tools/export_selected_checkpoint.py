"""Export frozen weights for the existing V2.9 inference/optimization interface."""
from pathlib import Path
import sys
import torch
from v29.research.evaluate import verify_freeze
from v29.research.protocol import digest, read_json, write_json

root = Path(sys.argv[1])
output = Path(sys.argv[2])
freeze = verify_freeze(root)
source = root/"runs/final"/f"{freeze['winner']}_seed{freeze['selected_seed']}"/"best.pt"
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
if output.exists():
    raise FileExistsError(output)
payload = {
    "schema_version": "v2.9-amplitude-operator-1",
    "model_kind": "v29_physics_layered_amplitude_operator",
    "model_config": checkpoint["model_config"],
    "model_state": checkpoint["model_state"],
    "parameter_count": 34973,
    "seed": freeze["selected_seed"],
    "selected_epoch": checkpoint["epoch"],
    "requested_total_epochs": 2000,
    "source_checkpoint_sha256": digest(source),
    "research_protocol_sha256": digest(root/"protocol.json"),
    "freeze_sha256": digest(root/"frozen_selection.json"),
    "selection_source": "validation_only",
    "evaluation_entrypoint": "v29.cli.research_campaign evaluate",
    "input_contract": {
        "command_iq": ["batch", "iq", "row", "col"],
        "unit_direction": ["batch", "query", "xyz"],
        "receiver_observation": "single_polarization_scalar_amplitude",
        "complex_convention": "transmit_h_transpose_w",
        "measured_phase_used": False, "s_parameters_used": False, "blind_labels_used": False,
    },
    "dataset_manifest": {
        "source_type": "hfss_amplitude", "receiver_channel": "ephi",
        "array_id": "v2_8_array_4x4_row_pitch_15mm_47GHz",
        "session_id": "hfss_prospective_20260905",
        "frequency_hz": 47e9, "reference_distance_m": 1.0,
        "amplitude_unit": "V/m_at_1m_from_HFSS_rE",
    },
    "deployment_eligible": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, output)
loaded = torch.load(output, map_location="cpu", weights_only=True)
for key in checkpoint["model_state"]:
    torch.testing.assert_close(checkpoint["model_state"][key], loaded["model_state"][key], rtol=0, atol=0)
write_json(output.with_suffix(".json"), {
    "checkpoint_sha256": digest(output), "source_checkpoint_sha256": digest(source),
    "selected_seed": freeze["selected_seed"], "selected_epoch": checkpoint["epoch"],
    "weights_exactly_preserved": True, "deployment_eligible": False,
})
print("EXPORTED", output, digest(output))
