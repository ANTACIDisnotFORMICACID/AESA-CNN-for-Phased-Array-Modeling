from controller_v26 import VitisBeamformingCNNV26
from deployment_v26 import audit_core, deployment_contract


def test_static_core_audit_has_no_non_dpu_math():
    report = audit_core(VitisBeamformingCNNV26())
    assert report["status"] == "pass"
    assert report["unsupported_nodes"] == []
    assert report["input"]["shape"] == [1, 15, 1, 4]
    assert report["output"]["shape"] == [1, 2, 1, 4]


def test_deployment_contract_preserves_all_reason_bits():
    bits = deployment_contract()["reason_bits"]
    assert bits == {
        "OUT_OF_DOMAIN": 1,
        "AMBIGUOUS_LOBE": 2,
        "POINTING_UNACHIEVABLE": 4,
        "CALIBRATION_UNCERTAIN": 8,
        "INSUFFICIENT_ELEMENTS": 16,
    }

