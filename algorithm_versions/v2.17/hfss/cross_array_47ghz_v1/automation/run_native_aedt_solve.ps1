[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$datasetRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$aedt = 'D:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe'
$drive = 'W:'

if (-not (Test-Path -LiteralPath $aedt -PathType Leaf)) {
    throw "AEDT executable is missing: $aedt"
}

$existing = & subst.exe $drive 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    throw "$drive is already assigned; refusing to replace it"
}

try {
    & subst.exe $drive $datasetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary ASCII drive mapping' }
    $aedtArguments = @(
        '-RunScriptAndExit', 'W:\automation\native_aedt_solve.py',
        '-Iconic',
        '-Logfile', 'W:\raw_exports\a1x4\native_aedt_console.log'
    )
    $process = Start-Process -FilePath $aedt -ArgumentList $aedtArguments -PassThru -Wait -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "AEDT internal solve exited with code $($process.ExitCode)" }
}
finally {
    & subst.exe $drive /D 2>$null
}
