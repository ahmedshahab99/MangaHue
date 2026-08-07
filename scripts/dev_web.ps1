param(
  [string]$ApiAddr = "127.0.0.1:7878",
  [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "web\app"
$Runtime = Join-Path $Root ".mangahue\python-runtime"
$Python = Join-Path $Runtime "Scripts\python.exe"
$webProcess = $null

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

  if (-not (Test-Path (Join-Path $Web "node_modules"))) {
    npm --prefix $Web install
  }

  $webProcess = Start-Process npm.cmd `
    -ArgumentList @("--prefix", $Web, "run", "dev", "--", "--port", $WebPort) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

  Write-Host "MangaHue web: http://127.0.0.1:$WebPort"
  Write-Host "MangaHue API: http://$ApiAddr"
  cargo +stable-x86_64-pc-windows-gnu run -p mangahue-api -- `
    --addr $ApiAddr `
    --jobs-dir ".mangahue\jobs" `
    --studio-dir "web\app\out" `
    --cors-origin "http://127.0.0.1:$WebPort" `
    --cors-origin "http://localhost:$WebPort"
}
finally {
  if ($null -ne $webProcess -and -not $webProcess.HasExited) {
    Stop-Process -Id $webProcess.Id -Force
  }
  Pop-Location
}
