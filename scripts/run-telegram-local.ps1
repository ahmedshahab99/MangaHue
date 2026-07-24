[CmdletBinding()]
param(
    [string]$Token,
    [string]$Model = "models/manga-onnx/manga-colorize-fp16.onnx",
    [string]$Preset = "configs/engine.onnx-small.yaml",
    [string]$WorkDir = ".mangahue/telegram",
    [ValidateRange(1, 8)]
    [int]$Threads = 2,
    [ValidateRange(5, 1800)]
    [int]$TimeoutSeconds = 180,
    [ValidateRange(1, 2016)]
    [int]$MaxPages = 64,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$secureToken = $null
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust cargo is required and was not found in PATH."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required and was not found in PATH."
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "ONNX model not found: $Model. Run scripts/download_onnx_small.ps1 first."
}
if (-not (Test-Path -LiteralPath $Preset -PathType Leaf)) {
    throw "Engine preset not found: $Preset"
}

if ($ValidateOnly) {
    [pscustomobject]@{
        Status = "ready"
        Model = (Resolve-Path -LiteralPath $Model).Path
        ModelMB = [math]::Round((Get-Item -LiteralPath $Model).Length / 1MB, 1)
        Preset = (Resolve-Path -LiteralPath $Preset).Path
        Threads = $Threads
        TimeoutSeconds = $TimeoutSeconds
        MaxPages = $MaxPages
    }
    return
}

if ([string]::IsNullOrWhiteSpace($Token)) {
    $secureToken = Read-Host "Enter the NEW Telegram bot token" -AsSecureString
    $Token = [System.Net.NetworkCredential]::new("", $secureToken).Password
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    throw "Telegram bot token cannot be empty."
}

$previousToken = $env:TELEGRAM_BOT_TOKEN
$previousPythonPath = $env:PYTHONPATH
$previousThreads = $env:MANGAHUE_MODEL_THREADS
$previousTimeout = $env:MANGAHUE_MODEL_TIMEOUT_SECONDS

try {
    $env:TELEGRAM_BOT_TOKEN = $Token
    $env:PYTHONPATH = Join-Path $repo "python"
    $env:MANGAHUE_MODEL_THREADS = $Threads.ToString()
    $env:MANGAHUE_MODEL_TIMEOUT_SECONDS = $TimeoutSeconds.ToString()

    Write-Host "Starting MangaHue Telegram with the compact ONNX engine..."
    & uv run --python 3.11 `
        --with onnxruntime `
        --with numpy `
        --with pillow `
        cargo run -p mangahue-telegram --release -- `
        --work-dir $WorkDir `
        --max-pages $MaxPages `
        --max-concurrent-jobs 1 `
        --model $Model `
        --preset $Preset `
        --hint-propagate
    if ($LASTEXITCODE -ne 0) {
        throw "MangaHue Telegram exited with code $LASTEXITCODE."
    }
}
finally {
    $env:TELEGRAM_BOT_TOKEN = $previousToken
    $env:PYTHONPATH = $previousPythonPath
    $env:MANGAHUE_MODEL_THREADS = $previousThreads
    $env:MANGAHUE_MODEL_TIMEOUT_SECONDS = $previousTimeout
    $Token = $null
    if ($null -ne $secureToken) {
        $secureToken.Dispose()
    }
}
