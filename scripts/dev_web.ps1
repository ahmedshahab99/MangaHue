param(
  [string]$ApiAddr = "127.0.0.1:7878",
  [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "web\app"
$webProcess = $null

Push-Location $Root
try {
  $FaceModel = Join-Path $Root "models\anime-face\lbpcascade_animeface.xml"
  if (Test-Path $FaceModel) {
    $env:MANGAHUE_FACE_MODEL = Split-Path -Parent $FaceModel
  }
  $DefaultModel = Join-Path $Root "models\ddcolor-full-fp32.onnx"

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
