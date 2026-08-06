param(
  [string]$ApiAddr = "127.0.0.1:7878",
  [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "web\app"

if (-not (Test-Path (Join-Path $Web "node_modules"))) {
  npm --prefix $Web install
}

$webProcess = Start-Process npm.cmd `
  -ArgumentList @("--prefix", $Web, "run", "dev", "--", "--port", $WebPort) `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -PassThru

try {
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
  if (-not $webProcess.HasExited) {
    Stop-Process -Id $webProcess.Id -Force
  }
}
