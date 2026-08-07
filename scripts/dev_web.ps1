param(
  [string]$ApiAddr = "127.0.0.1:7878",
  [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "web\app"
$Runtime = Join-Path $Root ".mangahue\python-runtime"
$Python = Join-Path $Runtime "Scripts\python.exe"
$StudioRuntime = Join-Path $Root ".mangahue\studio-runtime"
$StudioPython = Join-Path $StudioRuntime "Scripts\python.exe"
$StudioModel = Join-Path $Root "models\studio-lcm-int8"
$webProcess = $null

Push-Location $Root
try {
  if (-not (Test-Path $Python)) {
    uv venv --python 3.11 $Runtime
  }
  $runtimeReady = $false
  try {
    & $Python -c "import cv2, numpy, onnxruntime, PIL; assert hasattr(cv2, 'CascadeClassifier')" 2>$null
    $runtimeReady = $LASTEXITCODE -eq 0
  } catch {}
  if (-not $runtimeReady) {
    uv pip install --python $Python numpy pillow onnxruntime 'opencv-python-headless>=4.9,<5'
  }
  $env:MANGAHUE_PYTHON = $Python
  $env:PYTHONPATH = Join-Path $Root "python"
  $FaceModel = Join-Path $Root "models\anime-face\lbpcascade_animeface.xml"
  if (Test-Path $FaceModel) {
    $env:MANGAHUE_FACE_MODEL = Split-Path -Parent $FaceModel
  }
  $DefaultModel = Join-Path $Root "models\ddcolor-full-fp16.onnx"
  if ((Test-Path $StudioPython) -and (Test-Path (Join-Path $StudioModel "unet\openvino_model.xml"))) {
    $env:MANGAHUE_STUDIO_PYTHON = $StudioPython
    $env:MANGAHUE_WORKER_TIMEOUT_SECONDS = "900"
    $DefaultModel = $StudioModel
  }

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
    --default-model $DefaultModel `
    --cors-origin "http://127.0.0.1:$WebPort" `
    --cors-origin "http://localhost:$WebPort"
}
finally {
  if ($null -ne $webProcess -and -not $webProcess.HasExited) {
    Stop-Process -Id $webProcess.Id -Force
  }
  Pop-Location
}
