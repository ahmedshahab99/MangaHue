#requires -Version 5.1
<#
.SYNOPSIS
    End-to-end helper: prepare -> train -> export -> validate -> colorize.
.DESCRIPTION
    Wraps the Python training tools and the Rust CLI. Script lives at the repo
    root and assumes `pwsh` is available. Adjust the parameters to match your
    dataset layout and CUDA availability.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string] $Linearts,
    [Parameter(Mandatory=$true)] [string] $Colors,
    [Parameter(Mandatory=$true)] [string] $ModelOut,
    [string] $Input,
    [string] $Output,
    [int]    $Epochs = 20,
    [int]    $BatchSize = 2,
    [double] $Lr = 2e-4,
    [int]    $Size = 768,
    [string] $DataDir = "data",
    [string] $ModelsDir = "models",
    [switch] $SkipPrepare,
    [switch] $SkipTrain,
    [switch] $SkipExport,
    [switch] $SkipValidate,
    [switch] $SkipColorize,
    [switch] $SyntheticHints
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $RepoRoot

$trainJson = Join-Path $DataDir "train.json"
$valJson   = Join-Path $DataDir "val.json"
$ckpt      = Join-Path $ModelsDir "checkpoint.pt"
$bestCkpt  = Join-Path $ModelsDir "checkpoint.best.pt"
$meta      = [System.IO.Path]::ChangeExtension($ModelOut, $null) + ".mangahue.json"

if (-not (Test-Path -LiteralPath $DataDir))    { New-Item -ItemType Directory -Path $DataDir | Out-Null }
if (-not (Test-Path -LiteralPath $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

function Step($label, $block) {
    Write-Host ""
    Write-Host "==> $label" -ForegroundColor Cyan
    & $block
}

Step "prepare_dataset" {
    if ($SkipPrepare) { Write-Host "skipped"; return }
    $args = @(
        "--linearts", $Linearts,
        "--colors",   $Colors,
        "--output-train", $trainJson,
        "--output-val",   $valJson,
        "--val-ratio", "0.1"
    )
    if ($SyntheticHints) { $args += "--synthetic-hints" }
    python -m mangahue_train.prepare_dataset @args
}

Step "train" {
    if ($SkipTrain) { Write-Host "skipped"; return }
    $args = @(
        "run",
        "-m", "mangahue_train.train",
        "--manifest", $trainJson,
        "--val-manifest", $valJson,
        "--output", $ckpt,
        "--best-output", $bestCkpt,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--lr", $Lr,
        "--size", $Size
    )
    if ($SyntheticHints) {
        $args += @("--synthetic-hints", "--synthetic-hints-prob", "0.5")
    }
    python @args
}

Step "export_onnx" {
    if ($SkipExport) { Write-Host "skipped"; return }
    python -m mangahue_train.export_onnx `
        --checkpoint $bestCkpt `
        --output      $ModelOut `
        --metadata    $meta `
        --validate
}

Step "rust model validate-onnx" {
    if ($SkipValidate) { Write-Host "skipped"; return }
    cargo run -p mangahue-cli --quiet -- model validate-onnx --model $ModelOut --metadata $meta
}

Step "rust colorize" {
    if ($SkipColorize) { Write-Host "skipped"; return }
    if (-not $Input -or -not $Output) {
        Write-Host "no -Input / -Output provided; skipping colorize"
        return
    }
    cargo run -p mangahue-cli --quiet -- colorize `
        --input $Input `
        --output $Output `
        --model $ModelOut `
        --metadata $meta
}

Write-Host ""
Write-Host "Done. Model at $ModelOut" -ForegroundColor Green
