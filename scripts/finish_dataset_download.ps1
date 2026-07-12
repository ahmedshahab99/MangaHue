param(
    [Parameter(Mandatory = $true)][int]$DownloadProcessId,
    [Parameter(Mandatory = $true)][string]$DatasetRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Wait-Process -Id $DownloadProcessId
$downloadLog = Join-Path (Split-Path $DatasetRoot -Parent) "anime-sketch-download.log"
$downloadErrorLog = Join-Path (Split-Path $DatasetRoot -Parent) "anime-sketch-download.err.log"
if ((Test-Path $downloadErrorLog) -and (Select-String -Path $downloadErrorLog -Pattern "Traceback" -Quiet)) {
    throw "Dataset download failed. See $downloadErrorLog"
}

& $Python (Join-Path $RepositoryRoot "scripts\prepare_paired_dataset.py") $DatasetRoot
if ($LASTEXITCODE -ne 0) { throw "Pair preparation failed" }

$env:PYTHONPATH = Join-Path $RepositoryRoot "python"
$prepared = Join-Path $DatasetRoot "mangahue-prepared"
foreach ($split in @("train", "validation", "test")) {
    & $Python -m mangahue_train.audit_dataset `
        --manifest (Join-Path $prepared "$split.json") `
        --output (Join-Path $prepared "$split.audited.json") `
        --strict-rights
    if ($LASTEXITCODE -ne 0) { throw "Dataset audit failed for $split" }
}

"Download, preparation, and audits completed at $(Get-Date -Format o)" | Set-Content (Join-Path $prepared "READY")
