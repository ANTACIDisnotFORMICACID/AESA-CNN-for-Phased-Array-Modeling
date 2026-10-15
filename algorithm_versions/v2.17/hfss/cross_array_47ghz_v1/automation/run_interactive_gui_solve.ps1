[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('a2x4','a1x8')][string]$ArrayId
)

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$aedt = 'D:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe'
$drive = 'W:'
$project = "$drive\projects\manual_gui_work\${ArrayId}_47ghz_manual_gui.aedt"
$projectName = "${ArrayId}_47ghz_manual_gui"
$script = "$drive\automation\attach_interactive_gui_solve.vbs"
$receipt = "$drive\raw_exports\$ArrayId\interactive_gui_solve.log"

$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    $desktopProcess = Start-Process -FilePath $aedt -ArgumentList @($project) -PassThru
    Start-Sleep -Seconds 15
    & cscript.exe '//nologo' $script $projectName $receipt
    if ($LASTEXITCODE -ne 0) { throw "Interactive GUI solve failed with exit code $LASTEXITCODE" }
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
