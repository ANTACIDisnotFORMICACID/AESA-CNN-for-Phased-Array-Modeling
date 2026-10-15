[CmdletBinding()]
param(
    [switch]$Graphical
)

$ErrorActionPreference = 'Stop'
$Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($Identity -notmatch '(?i)\\ASUS$') {
    throw "请在 FORMIC-ACID\ASUS 的普通 PowerShell/Windows Terminal 中运行；当前身份为 $Identity。"
}

$Python = 'D:\Desktop\wireless\BigChuang\floquet\.venv_hfss\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "找不到既有 PyAEDT Python：$Python"
}

$Root = $PSScriptRoot
$ExistingMapping = (& subst.exe V: 2>$null | Out-String).Trim()
$CreatedMapping = $false
if ($ExistingMapping) {
    if ($ExistingMapping -notmatch [regex]::Escape($Root)) {
        throw "V: 已被映射到其他目录，请先解除该映射： $ExistingMapping"
    }
} else {
    & subst.exe V: $Root
    if ($LASTEXITCODE -ne 0) {
        throw "无法建立 V: 到 V2.8/HFSS 的 ASCII 路径映射。"
    }
    $CreatedMapping = $true
}

try {
    $Arguments = @(
        '-X', 'utf8',
        'V:\automation\build_4x4.py',
        '--source', 'V:\projects\source_1x4_20250326.aedt',
        '--target', 'V:\projects\v2_8_array_4x4.aedt',
        '--report', 'V:\artifacts\build_4x4_report.json'
    )
    if ($Graphical) {
        $Arguments += '--graphical'
    }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "4x4 构建失败；请查看 artifacts\build_4x4_report.json。"
    }

    Write-Host '4x4 AEDT 构建完成。请先检查 artifacts\build_4x4_report.json，再运行求解导出脚本。'
}
finally {
    if ($CreatedMapping) {
        & subst.exe V: /D
    }
}
