[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$Target = "mangahue",
    [string]$Baseline,
    [string]$Output = "benchmarks/quality-report.json",
    [ValidateRange(1, 100000)]
    [int]$MinPages = 50,
    [ValidateRange(0.0, 200.0)]
    [double]$MaxDeltaE = 28.0,
    [ValidateRange(-1.0, 1.0)]
    [double]$MinSsim = 0.55,
    [ValidateRange(0.0, 1.0)]
    [double]$MinLinePreservation = 0.92
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required and was not found in PATH."
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Benchmark manifest not found: $Manifest"
}

$env:PYTHONPATH = Join-Path $repo "python"
$arguments = @(
    "run", "--python", "3.11",
    "--with", "numpy",
    "--with", "pillow",
    "python", "-m", "mangahue_train.benchmark_quality",
    "--manifest", $Manifest,
    "--target", $Target,
    "--output", $Output,
    "--min-pages", $MinPages,
    "--max-delta-e", $MaxDeltaE,
    "--min-ssim", $MinSsim,
    "--min-line-preservation", $MinLinePreservation
)
if (-not [string]::IsNullOrWhiteSpace($Baseline)) {
    $arguments += @("--baseline", $Baseline)
}

& uv @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MangaHue quality gate failed with code $LASTEXITCODE."
}
