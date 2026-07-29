#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/models/anime-face/lbpcascade_animeface.xml}"
URL="https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

mkdir -p "$(dirname "$DEST")"
curl --fail --location --retry 5 --output "$DEST" "$URL"
test -s "$DEST"
printf 'Anime face palette model ready: %s\n' "$DEST"
