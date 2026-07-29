#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/models/reference-cgan/074000_gray.pt}"
URL="https://huggingface.co/Keiser41/Example_Based_Manga_Colorization/resolve/main/experiments/Color2Manga_gray/074000_gray.pt"
SHA256="e2f4785d00d4463ecb5f02d79f79f9da747a57179b5b016408e65da0e4f62572"

mkdir -p "$(dirname "$DEST")"
curl --fail --location --retry 5 --continue-at - --output "$DEST" "$URL"
printf '%s  %s\n' "$SHA256" "$DEST" | sha256sum --check -
printf 'Reference cGAN ready: %s\n' "$DEST"
