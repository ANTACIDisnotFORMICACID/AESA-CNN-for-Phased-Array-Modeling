# -*- coding: utf-8 -*-
"""Read the 16 direct-audit fields inside an ordinary AEDT GUI.

This follow-up avoids CreateReport/Data Table rendering.  It reads the solved
far-field arrays through GetSolutionDataPerVariation and writes compact binary
float64 payloads plus JSON metadata.  It never solves or saves the project.
"""

from __future__ import print_function

import json
import os
import struct
import time
import traceback


DATASET_ROOT = os.environ.get(
    "AESA_CNN_DATASET_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)
TARGET_PROJECT = "a1x8_47ghz_directapi_20261014T024216Z"
OUTPUT = os.path.join(DATASET_ROOT, "raw_exports", "a1x8", "direct_api_gate_20261014T024216Z")
AUDIT_MANIFEST = os.path.join(DATASET_ROOT, "direct_audit", "a1x8", "manifest.json")
RECEIPT = os.path.join(OUTPUT, "DIRECT_API_RECEIPT.log")


def mkdir_new(path):
    if os.path.exists(path):
        raise RuntimeError("Refusing to overwrite existing output: " + path)
    os.makedirs(path)


def record(message):
    if not os.path.isdir(OUTPUT):
        os.makedirs(OUTPUT)
    stream = open(RECEIPT, "a")
    try:
        stream.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
        stream.flush()
    finally:
        stream.close()


def write_json(path, value):
    stream = open(path, "w")
    try:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    finally:
        stream.close()


def write_float64(path, values):
    values = [float(value) for value in values]
    stream = open(path, "wb")
    try:
        stream.write(struct.pack("<{0}d".format(len(values)), *values))
    finally:
        stream.close()


def get_desktop():
    try:
        return oDesktop
    except NameError:
        import ScriptEnv
        ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
        return oDesktop


def edit_sources(solution_module, sources):
    solution_module.EditSources(
        ["IncludePortPostProcessing:=", True, "SpecifySystemPower:=", False]
    )
    for port in sorted(sources):
        values = sources[port]
        solution_module.EditSources(
            [
                "Name:=", port,
                "Magnitude:=", "{0:.15g}W".format(values["power_w"]),
                "Phase:=", "{0:.15g}deg".format(values["phase_deg"]),
            ]
        )


def unique(values):
    answer = []
    seen = set()
    for value in values:
        number = float(value)
        if number not in seen:
            seen.add(number)
            answer.append(number)
    return answer


mkdir_new(OUTPUT)
record("START ordinary-GUI direct API extraction; no report, no solve, no save")
try:
    desktop = get_desktop()
    project = desktop.GetActiveProject()
    if project is None:
        raise RuntimeError("No active AEDT project")
    if str(project.GetName()).lower() != TARGET_PROJECT.lower():
        raise RuntimeError("Active project mismatch: " + str(project.GetName()))
    design = project.SetActiveDesign("HFSSDesign1")
    report_module = design.GetModule("ReportSetup")
    solution_module = design.GetModule("Solutions")
    manifest = json.load(open(AUDIT_MANIFEST, "r"))
    record("PROJECT_VERIFIED " + str(project.GetName()))

    completed = []
    for item in manifest["commands"]:
        command_id = item["command_id"]
        record("DIRECT_API_BEGIN " + command_id)
        edit_sources(solution_module, item["sources"])
        variations = list(
            report_module.GetSolutionDataPerVariation(
                "Far Fields",
                "Setup2 : LastAdaptive",
                ["Context:=", "SphereInteger"],
                ["Freq:=", ["47GHz"], "Theta:=", ["All"], "Phi:=", ["All"]],
                ["re(rEPhi)", "im(rEPhi)"],
            )
        )
        if len(variations) != 1:
            raise RuntimeError("Expected one direct-field variation for " + command_id)
        variation = variations[0]
        real = list(variation.GetRealDataValues("re(rEPhi)", False))
        imag = list(variation.GetRealDataValues("im(rEPhi)", False))
        if len(real) != len(imag) or len(real) != 181 * 361:
            raise RuntimeError(
                "Unexpected direct-field size for {0}: {1}, {2}".format(command_id, len(real), len(imag))
            )
        folder = os.path.join(OUTPUT, command_id)
        os.makedirs(folder)
        write_float64(os.path.join(folder, "real.float64le"), real)
        write_float64(os.path.join(folder, "imag.float64le"), imag)
        sweep_names = [str(value) for value in variation.GetSweepNames()]
        metadata = {
            "command_id": command_id,
            "sample_count": len(real),
            "sweep_names": sweep_names,
            "theta_values": unique(variation.GetSweepValues("Theta", False)),
            "theta_unit": str(variation.GetSweepUnits("Theta")),
            "phi_values": unique(variation.GetSweepValues("Phi", False)),
            "phi_unit": str(variation.GetSweepUnits("Phi")),
            "frequency_values": unique(variation.GetSweepValues("Freq", False)),
            "frequency_unit": str(variation.GetSweepUnits("Freq")),
        }
        write_json(os.path.join(folder, "metadata.json"), metadata)
        completed.append(command_id)
        record("DIRECT_API_EXPORTED {0} samples={1}".format(command_id, len(real)))

    write_json(os.path.join(OUTPUT, "exported_commands.json"), completed)
    record("PASS all direct fields exported; close project WITHOUT SAVING")
except Exception as exc:
    record("FAIL " + repr(exc))
    record(traceback.format_exc())
    raise

