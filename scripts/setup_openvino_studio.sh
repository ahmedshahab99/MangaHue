#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${MANGAHUE_STUDIO_RUNTIME:-$ROOT/.studio-runtime}"
MODEL_DIR="${MANGAHUE_STUDIO_MODEL:-$ROOT/models/studio-lcm-int8}"
MODEL_ID="${MANGAHUE_STUDIO_MODEL_ID:-OpenVINO/LCM_Dreamshaper_v7-int8-ov}"

python3 -m venv "$RUNTIME"
"$RUNTIME/bin/python" -m pip install --quiet --upgrade pip
"$RUNTIME/bin/python" -m pip install --quiet \
  "openvino-genai>=2025.2,<2027" \
  "huggingface-hub>=0.25" \
  "pillow>=10" \
  "numpy<3"

mkdir -p "$MODEL_DIR"
"$RUNTIME/bin/python" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY

test -f "$MODEL_DIR/unet/openvino_model.xml"
test -f "$MODEL_DIR/vae_encoder/openvino_model.xml"
printf 'MangaHue Studio runtime: %s\n' "$RUNTIME"
printf 'MangaHue Studio model: %s\n' "$MODEL_DIR"
