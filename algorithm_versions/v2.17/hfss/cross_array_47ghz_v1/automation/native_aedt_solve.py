# -*- coding: utf-8 -*-
"""Run Setup2 inside AEDT, without an external PyAEDT/gRPC controller."""

import os
import time
import traceback


PROJECT = r"W:\projects\native_solve_work\a1x4_47ghz_native_gui.aedt"
RECEIPT = r"W:\raw_exports\a1x4\native_aedt_solve.log"


def record(message):
    folder = os.path.dirname(RECEIPT)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    stream = open(RECEIPT, "a")
    try:
        stream.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
    finally:
        stream.close()


record("START internal AEDT solve")
try:
    # RunScriptAndExit provides oDesktop in-process.  Calling
    # ScriptEnv.Initialize here would start a second desktop and request a
    # second GUI/HFSS license.
    oProject = oDesktop.OpenProject(PROJECT)
    oDesktop.SetActiveProject(oProject.GetName())
    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    record("PROJECT_OPEN " + oProject.GetName())
    oDesign.Analyze("Setup2")
    record("ANALYZE_RETURNED")
    oProject.Save()
    record("PROJECT_SAVED")
except Exception as exc:
    record("ERROR " + repr(exc))
    record(traceback.format_exc())
    raise

