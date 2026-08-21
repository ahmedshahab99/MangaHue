[CmdletBinding()]
param(
    [string]$Token,
    [string]$Model = "models/ddcolor-full-fp32.onnx",
    [string]$Preset = "configs/engine.default.yaml",
    [string]$Style = "configs/styles/soft_cel_anime_v2.yaml",
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
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Production DDColor Full model not found: $Model"
}
if (-not (Test-Path -LiteralPath $Preset -PathType Leaf)) {
    throw "Engine preset not found: $Preset"
}
if (-not (Test-Path -LiteralPath $Style -PathType Leaf)) {
    throw "Production style not found: $Style"
}

if ($ValidateOnly) {
    [pscustomobject]@{
        Status = "ready"
        Model = (Resolve-Path -LiteralPath $Model).Path
        ModelMB = [math]::Round((Get-Item -LiteralPath $Model).Length / 1MB, 1)
        Preset = (Resolve-Path -LiteralPath $Preset).Path
        Style = (Resolve-Path -LiteralPath $Style).Path
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
$previousThreads = $env:MANGAHUE_MODEL_THREADS
$previousTimeout = $env:MANGAHUE_MODEL_TIMEOUT_SECONDS

try {
    $env:TELEGRAM_BOT_TOKEN = $Token
    $env:MANGAHUE_MODEL_THREADS = $Threads.ToString()
    $env:MANGAHUE_MODEL_TIMEOUT_SECONDS = $TimeoutSeconds.ToString()

    Write-Host "Starting MangaHue Telegram with the production DDColor Full engine..."
    & cargo run -p mangahue-telegram --release -- `
        --work-dir $WorkDir `
        --max-pages $MaxPages `
        --max-concurrent-jobs 1 `
        --model $Model `
        --preset $Preset `
        --style $Style
    if ($LASTEXITCODE -ne 0) {
        throw "MangaHue Telegram exited with code $LASTEXITCODE."
    }
}
finally {
    $env:TELEGRAM_BOT_TOKEN = $previousToken
    $env:MANGAHUE_MODEL_THREADS = $previousThreads
    $env:MANGAHUE_MODEL_TIMEOUT_SECONDS = $previousTimeout
    $Token = $null
    if ($null -ne $secureToken) {
        $secureToken.Dispose()
    }
}
