<#
.SYNOPSIS
  One-time setup of PathSnap on a Windows laptop.
.EXAMPLE
  .\setup.ps1 -Role api     # laptop 1: Python env, ML package, backend
  .\setup.ps1 -Role ui      # laptop 2: Node packages for the React UI
  .\setup.ps1 -Role all     # both on one machine (development)
#>
param(
    [ValidateSet("api", "ui", "all")] [string]$Role = "all",
    [switch]$SkipTorch  # skip PyTorch: the two neural models in the leaderboard then cannot run
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

if ($Role -in "api", "all") {
    Step "Python 3.12"
    $py = $null
    try { $v = & py -3.12 -c "import sys; print(sys.version.split()[0])" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py -3.12" } } catch {}
    if (-not $py) { throw "Python 3.12 not found. Install it from python.org (tick 'Add to PATH'), then re-run." }
    Write-Host "found Python $v"

    $venv = Join-Path $Root ".venv"
    $vpy = Join-Path $venv "Scripts\python.exe"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    Step "virtual environment in .venv"
    if (-not (Test-Path $vpy)) {
        if ($uv) { uv venv --python 3.12 $venv } else { & py -3.12 -m venv $venv }
    }
    function PipInstall([string[]]$pipArgs) {
        if ($uv) { uv pip install --python $vpy @pipArgs } else { & $vpy -m pip install @pipArgs }
        if ($LASTEXITCODE -ne 0) { throw "package install failed: $pipArgs" }
    }
    if (-not $uv) { & $vpy -m pip install --upgrade pip | Out-Null }

    Step "ML package + dependencies"
    PipInstall @("-r", (Join-Path $Root "requirements.txt"))
    PipInstall @("-e", $Root)
    Step "backend dependencies"
    PipInstall @("-r", (Join-Path $Root "backend\requirements.txt"))
    if (-not $SkipTorch) {
        Step "PyTorch (CPU build, for the neural models in the leaderboard)"
        PipInstall @("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    }

    Step "check"
    & $vpy -c "from pathsnap.pipeline import load_bundle; b = load_bundle(); print('model bundle OK:', b['model_name'])"
    if ($LASTEXITCODE -ne 0) { throw "the model bundle could not be loaded (artifacts/pathsnap_model.joblib)" }
}

if ($Role -in "ui", "all") {
    Step "Node.js"
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { throw "Node.js not found. Install the LTS from nodejs.org (or: winget install OpenJS.NodeJS.LTS), then re-run." }
    $nv = (node -v).TrimStart("v")
    if ([version]$nv -lt [version]"22.12.0") {
        Write-Warning "Node $nv is older than 22.12 (Vite asks for >= 22.12). It may still work; to update: winget upgrade --id OpenJS.NodeJS.22"
    }
    Push-Location (Join-Path $Root "ui")
    try {
        Step "npm install (ui)"
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
        Step "build check"
        npm run build *> $null
        if ($LASTEXITCODE -ne 0) {
            # npm sometimes skips the optional Windows native binaries of rolldown/oxlint
            Write-Host "build failed once; installing the Windows native bindings and retrying"
            $rv = node -p "require('./node_modules/rolldown/package.json').version"
            $ov = node -p "require('./node_modules/oxlint/package.json').version"
            npm install --no-save "@rolldown/binding-win32-x64-msvc@$rv" "@oxlint/binding-win32-x64-msvc@$ov"
            npm run build
            if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
        }
        Write-Host "UI build OK"
    } finally { Pop-Location }
}

Write-Host "`nSetup done. Start with .\run_api.ps1 (laptop 1) and .\run_ui.ps1 (laptop 2)." -ForegroundColor Green
