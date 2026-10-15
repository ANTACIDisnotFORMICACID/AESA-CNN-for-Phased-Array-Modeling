# -*- coding: utf-8 -*-
"""Export a solved a1x8 recovery copy from inside an ordinary AEDT GUI.

Run this file with AEDT's Tools > Run Script command after opening the exact
TARGET_PROJECT below.  The script never calls Analyze or Save.  Temporary
reports and source weights exist only in the recovery copy's live session.
"""

from __future__ import print_function

import json
import os
import time
import traceback


DATASET_ROOT = os.environ.get(
    "AESA_CNN_DATASET_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)
TARGET_PROJECT = "a1x8_47ghz_recovery_20261013T191747Z"
OUTPUT = os.path.join(DATASET_ROOT, "raw_exports", "a1x8", "inprocess_gate_20261014T032500Z")
AUDIT_MANIFEST = os.path.join(DATASET_ROOT, "direct_audit", "a1x8", "manifest.json")
RECEIPT = os.path.join(OUTPUT, "GUI_EXPORT_RECEIPT.log")
PORT_COUNT = 8


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
    finally:
        stream.close()


def write_json(path, value):
    stream = open(path, "w")
    try:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    finally:
        stream.close()


def get_desktop():
    try:
        return oDesktop
    except NameError:
        import ScriptEnv
        ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
        return oDesktop


def export_s_parameters(report_module):
    expressions = []
    for source in range(1, PORT_COUNT + 1):
        for receive in range(1, PORT_COUNT + 1):
            expressions.append("S({0},{1})".format(receive, source))
    variations = report_module.GetSolutionDataPerVariation(
        "Modal Solution Data",
        "Setup2 : LastAdaptive",
        [],
        ["Freq:=", ["All"]],
        expressions,
    )
    variations = list(variations)
    if len(variations) != 1:
        raise RuntimeError("Expected one nominal S-parameter variation, got {0}".format(len(variations)))
    variation = variations[0]
    frequencies = list(variation.GetSweepValues("Freq", False))
    unit = str(variation.GetSweepUnits("Freq"))
    if not frequencies:
        raise RuntimeError("S-parameter frequency axis is empty")
    values = {}
    for expression in expressions:
        real = list(variation.GetRealDataValues(expression, False))
        imag = list(variation.GetImagDataValues(expression, False))
        if len(real) != len(frequencies) or len(imag) != len(frequencies):
            raise RuntimeError("Incomplete trace " + expression)
        values[expression] = {"real": real, "imag": imag}
    write_json(
        os.path.join(OUTPUT, "s_parameters.json"),
        {
            "setup": "Setup2 : LastAdaptive",
            "frequency_unit": unit,
            "frequency_values": frequencies,
            "port_count": PORT_COUNT,
            "expressions": expressions,
            "values": values,
        },
    )
    record("S_PARAMETERS_EXPORTED traces={0} samples={1} unit={2}".format(len(expressions), len(frequencies), unit))


def export_eep(radfield_module, design):
    nominal = design.GetNominalVariation()
    for label, sphere in (("integer", "SphereInteger"), ("offset", "SphereOffset")):
        folder = os.path.join(OUTPUT, "eep", label)
        os.makedirs(folder)
        radfield_module.ExportElementPatternToFile(
            [
                "ExportFileName:=", os.path.join(folder, "eep.ffd"),
                "SetupName:=", sphere,
                "IntrinsicVariationKey:=", "Freq='47GHz'",
                "DesignVariationKey:=", nominal,
                "SolutionName:=", "Setup2 : LastAdaptive",
            ]
        )
        mapping = os.path.join(folder, "eep.txt")
        if not os.path.isfile(mapping) or os.path.getsize(mapping) <= 0:
            raise RuntimeError("EEP mapping missing for " + sphere)
        record("EEP_EXPORTED " + label + " mapping=" + mapping)


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


def export_direct_reports(report_module, solution_module):
    manifest = json.load(open(AUDIT_MANIFEST, "r"))
    raw = os.path.join(OUTPUT, "direct_audit", "raw_reports")
    os.makedirs(raw)
    completed = []
    for item in manifest["commands"]:
        command_id = item["command_id"]
        edit_sources(solution_module, item["sources"])
        plot = "V217_INPROCESS_" + command_id
        report_module.CreateReport(
            plot,
            "Far Fields",
            "Data Table",
            "Setup2 : LastAdaptive",
            ["Context:=", "SphereInteger"],
            ["Freq:=", ["47GHz"], "Theta:=", ["All"], "Phi:=", ["All"]],
            ["X Component:=", "Theta", "Y Component:=", ["re(rEPhi)", "im(rEPhi)"]],
        )
        destination = os.path.join(raw, command_id + ".csv")
        try:
            report_module.ExportToFile(plot, destination)
        finally:
            report_module.DeleteReports([plot])
        if not os.path.isfile(destination) or os.path.getsize(destination) <= 0:
            raise RuntimeError("Direct report missing for " + command_id)
        completed.append(command_id)
        record("DIRECT_REPORT_EXPORTED " + command_id)
    write_json(os.path.join(OUTPUT, "direct_audit", "exported_commands.json"), completed)


mkdir_new(OUTPUT)
record("START ordinary-GUI in-process extraction; no solve and no save")
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
    radfield_module = design.GetModule("RadField")
    record("PROJECT_VERIFIED " + str(project.GetName()))
    export_s_parameters(report_module)
    export_eep(radfield_module, design)
    export_direct_reports(report_module, solution_module)
    record("PASS raw content exported; close project WITHOUT SAVING")
except Exception as exc:
    record("FAIL " + repr(exc))
    record(traceback.format_exc())
    raise

