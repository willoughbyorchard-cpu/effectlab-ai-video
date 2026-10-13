$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python is required. Install Python 3.11 or 3.12 from python.org and enable Add Python to PATH." }
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { python -m venv .venv }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Could not install CPU PyTorch." }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install EffectLab requirements." }
Write-Host ""
Write-Host "Setup complete. Next, run .\run-local.ps1"
