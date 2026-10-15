[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('a2x4','a1x8')][string]$ArrayId
)

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$aedt = 'D:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe'
$python = 'D:\Program Files\ANSYS Inc\v252\AnsysEM\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$drive = 'W:'
$project = "$drive\projects\manual_gui_work\${ArrayId}_47ghz_manual_gui.aedt"
$projectName = "${ArrayId}_47ghz_manual_gui"
$script = "$drive\automation\attach_pid_solve.py"
$report = "$drive\raw_exports\$ArrayId\pid_gui_solve.json"

$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    $desktopProcess = Start-Process -FilePath $aedt -ArgumentList @($project) -PassThru
    Start-Sleep -Seconds 15
    & $python -X utf8 $script --pid $desktopProcess.Id --project-name $projectName --report $report
    if ($LASTEXITCODE -ne 0) { throw "PID-attached GUI solve failed with exit code $LASTEXITCODE" }
    $null = $desktopProcess.CloseMainWindow()
    $desktopProcess.WaitForExit(30000) | Out-Null
}
catch {
    if ($desktopProcess -and -not $desktopProcess.HasExited) {
        $null = $desktopProcess.CloseMainWindow()
        if (-not $desktopProcess.WaitForExit(15000)) {
            Stop-Process -Id $desktopProcess.Id -ErrorAction SilentlyContinue
        }
    }
    throw
}
finally {
    & subst.exe $drive /D 2>$null
}
