param(
  [string]$ModelId = "OpenVINO/LCM_Dreamshaper_v7-int8-ov"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".mangahue\studio-runtime"
$Python = Join-Path $Runtime "Scripts\python.exe"
$ModelDir = Join-Path $Root "models\studio-lcm-int8"

Push-Location $Root
try {
  if (-not (Test-Path $Python)) {
    uv venv --python 3.11 $Runtime
  }
  uv pip install --python $Python `
    "openvino-genai>=2025.2,<2027" `
    "huggingface-hub>=0.25" `
    "pillow>=10" `
    "numpy<3"

  New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
  & $Python -c "from huggingface_hub import snapshot_download; snapshot_download('$ModelId', local_dir=r'$ModelDir')"

  $Unet = Join-Path $ModelDir "unet\openvino_model.xml"
  $Encoder = Join-Path $ModelDir "vae_encoder\openvino_model.xml"
  $Decoder = Join-Path $ModelDir "vae_decoder\openvino_model.xml"
  if (-not ((Test-Path $Unet) -and (Test-Path $Encoder) -and (Test-Path $Decoder))) {
    throw "OpenVINO Studio model download is incomplete"
  }
  & $Python -c "import openvino, openvino_genai; print('MangaHue Studio runtime ready')"
  Write-Host "Model: $ModelDir"
}
finally {
  Pop-Location
}
