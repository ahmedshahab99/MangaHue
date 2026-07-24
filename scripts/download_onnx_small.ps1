param(
    [string]$Output = "models\manga-onnx\manga-colorize-fp16.onnx",
    [string]$Url = "https://huggingface.co/Faridzar/manga-colorization-v2-onnx/resolve/main/manga-colorize-fp16.onnx"
)

$expectedSha256 = "39660d0047ea6f1a0ddee6aa89054997f95ea566f4d56ff762f66dbcf1a1a7ef"
$parent = Split-Path -Parent $Output
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

if (Test-Path -LiteralPath $Output -PathType Leaf) {
    $current = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($current -eq $expectedSha256) {
        Write-Output "MangaHue ONNX model already verified: $Output"
        exit 0
    }
}

$partial = "$Output.partial"
Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
curl.exe -L --fail --retry 3 --output $partial $Url
if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    throw "ONNX model download failed with exit code $LASTEXITCODE"
}

$actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expectedSha256) {
    Remove-Item -LiteralPath $partial -Force
    throw "ONNX model checksum mismatch: expected $expectedSha256, got $actual"
}

Move-Item -LiteralPath $partial -Destination $Output -Force
Write-Output "Downloaded and verified MangaHue ONNX model: $Output"
