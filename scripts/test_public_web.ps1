param(
  [int]$Port = 7881
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunId = [guid]::NewGuid().ToString("N")
$TestRoot = Join-Path $Root ".mangahue\public-e2e\$RunId"
$JobsDir = Join-Path $TestRoot "jobs"
$Database = Join-Path $TestRoot "production.db"
$Sample = Join-Path $Root "examples\input\smoke_page.png"
$BaseUrl = "http://127.0.0.1:$Port"
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root "target" }
$BuiltApi = Join-Path $TargetDir "debug\mangahue-api.exe"
$TestApi = Join-Path $TestRoot "mangahue-api.exe"
$apiProcess = $null

function Get-BootstrapToken {
  param([string]$Email, [string]$Name)
  $lines = & cargo +stable-x86_64-pc-windows-gnu run -q -p mangahue-cli -- `
    production bootstrap --database $Database --email $Email --name $Name --tier free
  $token = $lines | Where-Object { $_ -like "api_token=*" } | Select-Object -Last 1
  if (-not $token) { throw "Could not create test access token" }
  return $token.Substring("api_token=".Length)
}

function Wait-ForHealth {
  $deadline = (Get-Date).AddMinutes(2)
  do {
    Start-Sleep -Milliseconds 500
    try {
      $health = Invoke-RestMethod "$BaseUrl/health" -TimeoutSec 3
      if ($health.status -eq "ok" -and $health.model.ready) { return }
    } catch {}
  } while ((Get-Date) -lt $deadline)
  throw "Public API did not become healthy"
}

New-Item -ItemType Directory -Force $TestRoot | Out-Null
if (-not (Test-Path $Sample)) { throw "Missing integration sample: $Sample" }
if (-not (Test-Path $BuiltApi)) {
  & cargo +stable-x86_64-pc-windows-gnu build -q -p mangahue-api
  if ($LASTEXITCODE -ne 0) { throw "Could not build the API integration-test binary" }
}
Copy-Item -LiteralPath $BuiltApi -Destination $TestApi

Push-Location $Root
try {
  $tokenOne = Get-BootstrapToken "artist-one-$RunId@example.invalid" "Artist One"
  $tokenTwo = Get-BootstrapToken "artist-two-$RunId@example.invalid" "Artist Two"
  $arguments = @(
    "--addr", "127.0.0.1:$Port",
    "--jobs-dir", $JobsDir,
    "--production-db", $Database,
    "--studio-dir", "web/app/out",
    "--public-api"
  )
  $apiProcess = Start-Process $TestApi -ArgumentList $arguments -WorkingDirectory $Root `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $TestRoot "api.out.log") `
    -RedirectStandardError (Join-Path $TestRoot "api.err.log") -PassThru
  Wait-ForHealth

  $unauthorized = 0
  try { Invoke-WebRequest "$BaseUrl/api/styles" -UseBasicParsing | Out-Null } catch {
    $unauthorized = [int]$_.Exception.Response.StatusCode
  }
  if ($unauthorized -ne 401) { throw "Anonymous API request was not rejected" }

  $headersOne = @{ Authorization = "Bearer $tokenOne" }
  $headersTwo = @{ Authorization = "Bearer $tokenTwo" }
  $styles = Invoke-RestMethod "$BaseUrl/api/styles" -Headers $headersOne
  if (-not $styles -or -not $styles[0].path) { throw "No trusted styles were returned" }

  $uploadJson = & curl.exe -sS --fail -H "Authorization: Bearer $tokenOne" `
    -F "file=@$Sample;type=image/png" "$BaseUrl/api/upload"
  $upload = $uploadJson | ConvertFrom-Json
  if ($upload.kind -ne "page") { throw "Page upload was not classified correctly" }

  $template = @{
    input_path = $upload.path
    output_path = "ignored.png"
    model_path = "untrusted-model.pkl"
    reference_image_path = $null
    config_path = "untrusted-config.yaml"
    preset_path = $null
    style_path = $styles[0].path
    project_path = $null
    region_locks_path = $null
    character_ids = @()
    user_hints_path = $null
    debug_dir = $null
    use_mock_backend = $true
    hint_propagation_enabled = $true
  }
  $job = Invoke-RestMethod "$BaseUrl/api/colorize" -Method Post -Headers $headersOne `
    -ContentType "application/json" -Body (@{ request = $template; ads_token = $null } | ConvertTo-Json -Depth 8)

  $deadline = (Get-Date).AddMinutes(2)
  do {
    Start-Sleep -Milliseconds 500
    $job = Invoke-RestMethod "$BaseUrl/api/jobs/$($job.id)" -Headers $headersOne
  } while ($job.status -in @("queued", "running") -and (Get-Date) -lt $deadline)
  if ($job.status -ne "complete") { throw "Public job ended as $($job.status): $($job.error)" }
  if ($job.result.report.model_path -notlike "*ddcolor-full-fp32.onnx") {
    throw "Server did not force the trusted default model"
  }

  $crossUserStatus = 0
  try { Invoke-WebRequest "$BaseUrl/api/jobs/$($job.id)" -Headers $headersTwo -UseBasicParsing | Out-Null } catch {
    $crossUserStatus = [int]$_.Exception.Response.StatusCode
  }
  if ($crossUserStatus -ne 404) { throw "Cross-user job access was not denied" }

  $resultQuery = [uri]::EscapeDataString([string]$job.result.output_path)
  $download = Join-Path $TestRoot "result.png"
  Invoke-WebRequest "$BaseUrl/api/file?path=$resultQuery&job_id=$($job.id)" `
    -Headers $headersOne -UseBasicParsing -OutFile $download
  if ((Get-Item $download).Length -le 0) { throw "Authorized result download was empty" }

  [pscustomobject]@{
    health = "ok"
    anonymous_rejected = $true
    real_model = "ddcolor-full-fp32.onnx"
    job_status = $job.status
    cross_user_denied = $true
    result_bytes = (Get-Item $download).Length
  } | ConvertTo-Json
}
finally {
  $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($listener) { Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue }
  if ($apiProcess -and -not $apiProcess.HasExited) {
    Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
  }
  Pop-Location
  $safeRoot = [IO.Path]::GetFullPath((Join-Path $Root ".mangahue\public-e2e"))
  $resolvedTestRoot = [IO.Path]::GetFullPath($TestRoot)
  if ($resolvedTestRoot.StartsWith($safeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}
