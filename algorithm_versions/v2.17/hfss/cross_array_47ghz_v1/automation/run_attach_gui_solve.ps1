[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$aedt = 'D:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe'
$drive = 'W:'
$project = 'W:\projects\native_solve_work\a1x4_47ghz_com_gui.aedt'
$script = 'W:\automation\attach_gui_solve.vbs'

$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    $desktopProcess = Start-Process -FilePath $aedt -ArgumentList @($project) -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 12
    & cscript.exe '//nologo' $script
    if ($LASTEXITCODE -ne 0) { throw "Attached GUI solve failed with exit code $LASTEXITCODE" }
    $desktopProcess.WaitForExit(30000) | Out-Null
}
finally {
    & subst.exe $drive /D 2>$null
}
