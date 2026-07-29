#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/models/ddcolor-tiny/ddcolor-tiny-fp16.onnx}"
URL="https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx"
SHA256="2653da00dc15e54a45e5200b61dbf82ee9ceaf56b02bb9b9657569ac775e82e6"

mkdir -p "$(dirname "$DEST")"
curl --fail --location --retry 5 --continue-at - --output "$DEST" "$URL"
printf '%s  %s\n' "$SHA256" "$DEST" | sha256sum --check -
printf 'DDColor Tiny ready: %s\n' "$DEST"
