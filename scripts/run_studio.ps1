param(
  [string]$Addr = "127.0.0.1:7878",
  [string]$JobsDir = ".mangahue\jobs",
  [string]$StudioDir = "web\app\out"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
  cargo +stable-x86_64-pc-windows-gnu run -p mangahue-cli -- serve --addr $Addr --jobs-dir $JobsDir --studio-dir $StudioDir
}
finally {
  Pop-Location
}
