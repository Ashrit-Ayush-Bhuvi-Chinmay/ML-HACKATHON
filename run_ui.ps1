<#
.SYNOPSIS
  Start the PathSnap UI (laptop 2), then open the printed http://localhost:5173 address.
  In the UI's "API address" box enter laptop 1's address, as printed in the API window
  (e.g. http://192.168.1.23:8000).
#>
param([int]$Port = 5173)
$ErrorActionPreference = "Stop"
$ui = Join-Path $PSScriptRoot "ui"
if (-not (Test-Path (Join-Path $ui "node_modules"))) { throw "No node_modules yet: run .\setup.ps1 -Role ui first." }
Push-Location $ui
try { npm run dev -- --port $Port } finally { Pop-Location }
