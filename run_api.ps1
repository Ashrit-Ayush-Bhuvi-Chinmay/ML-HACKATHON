<#
.SYNOPSIS
  Start the PathSnap API (laptop 1). Leave this window open: it is the console the evaluator watches.
.EXAMPLE
  .\run_api.ps1                                   # rehearsal: the supplied student_test trips
  .\run_api.ps1 -EvalDir "D:\evaluator_test"      # demo day: the evaluator's folder
#>
param(
    [string]$EvalDir = "",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { throw "No .venv yet: run .\setup.ps1 -Role api first." }

if ($EvalDir) {
    $resolved = Resolve-Path $EvalDir -ErrorAction SilentlyContinue
    if (-not $resolved) { throw "EvalDir not found: $EvalDir" }
    $env:PATHSNAP_EVAL_DIR = $resolved.Path
} else {
    Remove-Item Env:PATHSNAP_EVAL_DIR -ErrorAction SilentlyContinue  # auto-detect student_test
}

$rule = Get-NetFirewallRule -DisplayName "PathSnap API" -ErrorAction SilentlyContinue
if (-not $rule) {
    Write-Warning ("No firewall rule for port $($Port): laptop 2 may not reach this API. In an ADMIN PowerShell run:`n" +
        "  New-NetFirewallRule -DisplayName 'PathSnap API' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow")
}

Push-Location (Join-Path $Root "backend")
try {
    & $vpy -m uvicorn main:app --host 0.0.0.0 --port $Port
} finally { Pop-Location }
