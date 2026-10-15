from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stage(array_id: str, name: str) -> dict[str, object]:
    path = ROOT / "raw_exports" / array_id / "stages" / f"{name}.json"
    report = load(path)
    return {
        "status": "PASS" if report and report.get("passed") else ("FAIL" if report else "PENDING"),
        "receipt": str(path),
        "error": report.get("error") if report and not report.get("passed") else None,
    }


def main() -> None:
    manifest_path = ROOT / "dataset_manifest.json"
    manifest = load(manifest_path) or {"schema": "cross_array_47ghz_v1", "arrays": {}}
    arrays = manifest.setdefault("arrays", {})
    specs = {"a1x4": [1, 4], "a2x4": [2, 4], "a4x4": [4, 4], "a1x8": [1, 8]}
    for array_id, shape in specs.items():
        entry = arrays.setdefault(array_id, {"array_id": array_id})
        entry["logical_shape"] = shape
        entry["campaign_stages"] = {
            name: stage(array_id, name)
            for name in ("build", "solve", "export_s", "export_eep", "direct_audit")
        }
        all_passed = all(value["status"] == "PASS" for value in entry["campaign_stages"].values())
        if array_id == "a4x4" and entry.get("status") == "canonical_complete":
            entry["campaign_status"] = "PASS_REUSED_FROZEN_DATA"
        elif all_passed and entry.get("status") == "canonical_complete":
            entry["campaign_status"] = "PASS"
            entry.pop("blocker_receipt", None)
        elif (
            entry["campaign_stages"]["build"]["status"] == "PASS"
            and entry["campaign_stages"]["solve"]["status"] == "FAIL"
        ):
            entry["campaign_status"] = "BLOCKED_AT_SOLVE"
            entry["blocker_receipt"] = entry["campaign_stages"]["solve"]["receipt"]
        elif entry["campaign_stages"]["build"]["status"] == "PASS":
            entry["campaign_status"] = "IN_PROGRESS_BUILD_PASS"
        elif array_id == "a2x4" and arrays.get("a1x4", {}).get("campaign_status") == "PASS":
            entry["campaign_status"] = "PENDING_READY_AFTER_A1X4_GATE"
        elif array_id == "a1x8" and arrays.get("a2x4", {}).get("campaign_status") == "PASS":
            entry["campaign_status"] = "PENDING_READY_AFTER_A2X4_GATE"
        else:
            entry["campaign_status"] = "PENDING_PREVIOUS_ARRAY_GATE"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["execution_order"] = ["a1x4", "a2x4", "a1x8", "a4x4"]
    if arrays.get("a1x4", {}).get("campaign_status") != "PASS":
        manifest["next_action"] = "complete_a1x4_full_data_gate"
    elif arrays.get("a2x4", {}).get("campaign_status") == "BLOCKED_AT_SOLVE":
        manifest["next_action"] = "solve_a2x4_in_ordinary_interactive_GUI_then_reuse_verified_extraction_route"
    elif arrays.get("a2x4", {}).get("campaign_status") != "PASS":
        manifest["next_action"] = "build_then_solve_export_and_audit_a2x4"
    elif arrays.get("a1x8", {}).get("campaign_status") == "IN_PROGRESS_BUILD_PASS":
        manifest["next_action"] = "solve_a1x8_in_ordinary_interactive_GUI_then_reuse_verified_extraction_route"
    elif arrays.get("a1x8", {}).get("campaign_status") != "PASS":
        manifest["next_action"] = "build_then_solve_export_and_audit_a1x8"
    else:
        manifest["next_action"] = "all_hfss_data_gates_complete_prepare_v217_stage_a_without_blind_test"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value["campaign_status"] for key, value in arrays.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
