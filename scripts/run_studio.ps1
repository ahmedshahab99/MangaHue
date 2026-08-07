param(
  [string]$Addr = "127.0.0.1:7878",
  [string]$JobsDir = ".mangahue\jobs",
  [string]$StudioDir = "web\app\out"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".mangahue\python-runtime"
$Python = Join-Path $Runtime "Scripts\python.exe"

Push-Location $Root
try {
  if (-not (Test-Path $Python)) {
    uv venv --python 3.11 $Runtime
  }
  $runtimeReady = $false
  try {
    & $Python -c "import numpy, onnxruntime, PIL" 2>$null
    $runtimeReady = $LASTEXITCODE -eq 0
  } catch {}
  if (-not $runtimeReady) {
    uv pip install --python $Python numpy pillow onnxruntime
  }
  $env:MANGAHUE_PYTHON = $Python
  $env:PYTHONPATH = Join-Path $Root "python"

  cargo +stable-x86_64-pc-windows-gnu run -p mangahue-cli -- serve --addr $Addr --jobs-dir $JobsDir --studio-dir $StudioDir
}
finally {
  Pop-Location
}
