$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run .\setup-laptop.ps1 first." }
& $python -c "import torch, cv2, fastapi, uvicorn"
if ($LASTEXITCODE -ne 0) { throw "EffectLab dependencies are missing. Run .\setup-laptop.ps1." }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "weights") | Out-Null
$env:EFFECTLAB_CHECKPOINT = Join-Path $PSScriptRoot "weights\effectlab_scratch.pt"
$env:EFFECTLAB_TRAINING_ROOT = Join-Path $PSScriptRoot "training-data"
$env:EFFECTLAB_WORK_ROOT = Join-Path $env:LOCALAPPDATA "EffectLab\work"
Write-Host ""
Write-Host "EffectLab local UI: http://127.0.0.1:7860"
Write-Host "Training page:       http://127.0.0.1:7860/train.html"
Write-Host "No API key is needed. The engine listens only on this computer; keep this window open."
Write-Host "Your media remains on this laptop; no external AI renderer is called."
Write-Host "Press Ctrl+C to stop."
& $python -m uvicorn app:app --host 127.0.0.1 --port 7860
