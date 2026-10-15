param(
    [Parameter(Mandatory=$true)][ValidateSet('build','solve','export_s','export_eep')][string]$Stage,
    [Parameter(Mandatory=$true)][ValidateSet('a1x4','a2x4','a1x8')][string]$ArrayId,
    [int]$Cores = 4,
    [switch]$Graphical,
    [string]$SourceProject,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $SourceProject) {
    $SourceProject = Join-Path $datasetRoot '..\..\..\v2.8\HFSS\projects\source_1x4_20250326.aedt'
}
if (-not $PythonPath) {
    $PythonPath = if ($env:AESA_HFSS_PYTHON) { $env:AESA_HFSS_PYTHON } else { (Get-Command python).Source }
}
if (-not (Test-Path -LiteralPath $SourceProject -PathType Leaf)) {
    throw "Source AEDT project is not included; pass -SourceProject explicitly: $SourceProject"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "HFSS Python environment is missing: $PythonPath"
}
$source = (Resolve-Path -LiteralPath $SourceProject).Path
$drive = 'W:'
$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    $arguments = @(
        '-X', 'utf8',
        "$drive\automation\hfss_pipeline.py",
        '--stage', $Stage,
        '--array-id', $ArrayId,
        '--root', "$drive\",
        '--source', $source,
        '--cores', $Cores
    )
    if ($Graphical) { $arguments += '--graphical' }
    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "HFSS stage failed with exit code $LASTEXITCODE" }
}
finally {
    & subst.exe $drive /D 2>$null
}
