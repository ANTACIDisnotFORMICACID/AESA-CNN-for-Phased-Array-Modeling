[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('a2x4','a1x8')][string]$ArrayId,
    [Parameter(Mandatory=$true)][int]$Rows,
    [Parameter(Mandatory=$true)][int]$Cols,
    [string]$Project,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $PythonPath) {
    $PythonPath = if ($env:AESA_HFSS_PYTHON) { $env:AESA_HFSS_PYTHON } else { (Get-Command python).Source }
}
$drive = 'W:'

if (
    ($ArrayId -eq 'a2x4' -and ($Rows -ne 2 -or $Cols -ne 4)) -or
    ($ArrayId -eq 'a1x8' -and ($Rows -ne 1 -or $Cols -ne 8))
) {
    throw "Rows/Cols does not match $ArrayId"
}
if (-not $Project) {
    $Project = Join-Path $datasetRoot "projects\manual_gui_work\${ArrayId}_47ghz_manual_gui.aedt"
}
$sourceProject = (Resolve-Path -LiteralPath $Project).Path
$sourceDirectory = Split-Path -Parent $sourceProject
$sourceStem = [System.IO.Path]::GetFileNameWithoutExtension($sourceProject)
$sourceResults = Join-Path $sourceDirectory ($sourceStem + '.aedtresults')
if (-not (Test-Path -LiteralPath $sourceResults -PathType Container)) {
    throw "Matching AEDT results directory is missing: $sourceResults"
}
if (Get-Process ansysedt,HFSSCOMENGINE -ErrorAction SilentlyContinue) {
    throw 'AEDT or HFSSCOMENGINE is still running; close the GUI after saving before recovery'
}

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$recoveryDirectory = Join-Path $datasetRoot "projects\recovery_work\${ArrayId}_$stamp"
$targetStem = "${ArrayId}_47ghz_recovery_$stamp"
$targetProject = Join-Path $recoveryDirectory ($targetStem + '.aedt')
$targetResults = Join-Path $recoveryDirectory ($targetStem + '.aedtresults')
$output = Join-Path $datasetRoot "raw_exports\$ArrayId\recovery_gate_$stamp"
$report = Join-Path $datasetRoot "raw_exports\$ArrayId\recovery_gate_$stamp.json"
if ((Test-Path -LiteralPath $recoveryDirectory) -or (Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $report)) {
    throw 'Timestamped recovery target already exists; refusing to overwrite'
}
$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

New-Item -ItemType Directory -Path $recoveryDirectory | Out-Null
Copy-Item -LiteralPath $sourceProject -Destination $targetProject -Force
Copy-Item -LiteralPath $sourceResults -Destination $targetResults -Recurse -Force

$relativeProject = [System.IO.Path]::GetRelativePath($datasetRoot, $targetProject)
$relativeOutput = [System.IO.Path]::GetRelativePath($datasetRoot, $output)
$relativeReport = [System.IO.Path]::GetRelativePath($datasetRoot, $report)
try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    & $PythonPath -X utf8 "$drive\automation\recover_existing_solution.py" `
        --project "$drive\$relativeProject" `
        --array-id $ArrayId `
        --rows $Rows `
        --cols $Cols `
        --audit-manifest "$drive\direct_audit\$ArrayId\manifest.json" `
        --output "$drive\$relativeOutput" `
        --report "$drive\$relativeReport"
    if ($LASTEXITCODE -ne 0) {
        throw "Verified recovery extraction failed with exit code $LASTEXITCODE"
    }
}
finally {
    & subst.exe $drive /D 2>$null
}

[pscustomobject]@{
    ArrayId = $ArrayId
    RecoveryProject = $targetProject
    RecoveryResults = $targetResults
    Output = $output
    Report = $report
}
