param(
    [string]$InputPath = "examples\input\smoke_page.png",
    [string]$Output = "examples\output\pretrained.png",
    [string]$ModelRoot = $env:MANGAHUE_PRETRAINED_MODEL,
    [string]$Python = $env:MANGAHUE_PYTHON,
    [string]$Preset = "configs\engine.pretrained-small.yaml"
)

if (-not $ModelRoot) {
    $ModelRoot = Join-Path $env:USERPROFILE "OneDrive\Desktop\MangaColorizer-Small"
}
if (-not $Python) {
    $Python = Join-Path $env:USERPROFILE ".venvs\mangahue-small\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "MangaHue Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelRoot "networks\generator.zip") -PathType Leaf)) {
    throw "Compact pretrained model not found or incomplete: $ModelRoot"
}

$outputParent = Split-Path -Parent $Output
if ($outputParent) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
$env:MANGAHUE_PYTHON = $Python
if (-not $env:MANGAHUE_MODEL_THREADS) {
    $env:MANGAHUE_MODEL_THREADS = "2"
}

cargo run -q -p mangahue-cli -- colorize $InputPath `
    --output $Output `
    --model $ModelRoot `
    --preset $Preset

if ($LASTEXITCODE -ne 0) {
    throw "MangaHue pretrained colorization failed with exit code $LASTEXITCODE"
}
