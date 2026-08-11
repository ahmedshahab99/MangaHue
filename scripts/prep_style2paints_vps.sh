#!/bin/bash
# One-shot Style2Paints V3 prep on the VPS: weights -> ONNX -> int8, native Rust after.
# Run as root on 153.75.251.215. Torch is installed temporarily and NOT needed at runtime.
set -euo pipefail

VENV=/opt/MangaHue/.venv
MODEL_DIR=/opt/MangaHue/models/style2paints-v3
WEIGHTS_DIR=$MODEL_DIR/weights
REPO=/opt/style2paints
ONNX=$MODEL_DIR/style2paints-v3.onnx
INT8=$MODEL_DIR/style2paints-v3-int8.onnx
PY=$VENV/bin/python

echo "==> freeing RAM (stopping the bot)"
systemctl stop mangahue-pro || true
sleep 2

mkdir -p "$WEIGHTS_DIR"

echo "==> cloning style2paints repo"
if [ ! -d "$REPO" ]; then
  git clone --depth 1 https://github.com/lllyasviel/style2paints.git "$REPO"
fi

echo "==> installing temp tooling in the venv (torch-cpu, onnx, gdown)"
$PY -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
$PY -m pip install -q onnx gdown

echo "==> downloading V3 weights (encoder/decoder/pfc)"
if [ ! -s "$WEIGHTS_DIR/encoder.pth" ] || [ ! -s "$WEIGHTS_DIR/decoder.pth" ]; then
  # 1) scrape drive.google.com links from the repo README(s) and try them
  for f in "$REPO"/style2paintsV3/README.md "$REPO"/README.md "$REPO"/style2paintsV3/*.md; do
    [ -f "$f" ] || continue
    grep -oE 'https://drive\.google\.com/(open\?id=|uc\?id=|file/d/)[^ )"'"'"']*' "$f" \
      | sed -E 's#file/d/([^/]+)/.*#https://drive.google.com/uc?id=\1#; s#open\?id=#uc?id=#' \
      | sort -u >> /tmp/s2p_drive_links.txt || true
  done
  # 2) known mirrors
  cat >> /tmp/s2p_drive_links.txt <<'EOF'
https://huggingface.co/datasets/camenduru/style2paints-v3/resolve/main/encoder.pth
https://huggingface.co/datasets/camenduru/style2paints-v3/resolve/main/decoder.pth
https://huggingface.co/datasets/camenduru/style2paints-v3/resolve/main/pfc.pth
EOF
  sort -u /tmp/s2p_drive_links.txt -o /tmp/s2p_drive_links.txt
  n=0
  while read -r url; do
    [ -n "$url" ] || continue
    n=$((n+1))
    echo "  trying [$n]: $url"
    if gdown --fuzzy -q -O "$WEIGHTS_DIR/try_$n.pth" "$url" 2>/dev/null \
       && [ -s "$WEIGHTS_DIR/try_$n.pth" ]; then
      sz=$(stat -c%s "$WEIGHTS_DIR/try_$n.pth")
      if [ "$sz" -gt 5000000 ]; then
        echo "    candidate $n: ${sz} bytes — kept"
      else
        rm -f "$WEIGHTS_DIR/try_$n.pth"
      fi
    fi
  done < /tmp/s2p_drive_links.txt
  # assign candidates to encoder/decoder/pfc by state-dict key shape
  "$PY" - <<'PYEOF'
import torch, glob, shutil
from pathlib import Path
cands = sorted(Path('/opt/MangaHue/models/style2paints-v3/weights').glob('try_*.pth'))
by = {'encoder.pth': None, 'decoder.pth': None, 'pfc.pth': None}
for c in cands:
    try:
        sd = torch.load(c, map_location='cpu', weights_only=True)
    except Exception:
        continue
    keys = list(sd.keys())
    if not keys:
        continue
    # encoder: conv-in features are shallow; decoder: starts with convtranspose/up;
    # pfc: small net. Match by first-key name heuristics + size.
    first = keys[0].lower()
    total = sum(v.numel() for v in sd.values())
    if 'encoder' in str(c) or 'decoder' in str(c) or 'pfc' in str(c):
        pass
    if 'up' in first or 'convtranspose' in first or 'deconv' in first:
        by['decoder.pth'] = c
    elif total < 20_000_000:
        by['pfc.pth'] = c
    else:
        by['encoder.pth'] = c
for name, src in by.items():
    if src and not Path(f'/opt/MangaHue/models/style2paints-v3/weights/{name}').exists():
        shutil.move(src, f'/opt/MangaHue/models/style2paints-v3/weights/{name}')
        print(f'{name} <- {src.name}')
for c in cands:
    if c.exists():
        c.unlink()
missing = [k for k, v in by.items() if v is None]
if missing:
    print('WARNING: could not assign', missing, '- manual download needed (see below)')
PYEOF
fi
ls -la "$WEIGHTS_DIR" || true
for w in encoder.pth decoder.pth pfc.pth; do
  if [ ! -s "$WEIGHTS_DIR/$w" ]; then
    echo "!! MISSING $w — upload it to $WEIGHTS_DIR/$w and re-run this script"
    exit 1
  fi
done

echo "==> exporting ONNX (CPU, one-time)"
$PY /opt/MangaHue-release/scripts/export_style2paints_v3.py \
  --repo "$REPO" \
  --weights "$WEIGHTS_DIR" \
  --output "$ONNX"

echo "==> int8 quantization (dynamic, via the engine's validated tool)"
if $PY -c "import mangahue_train" 2>/dev/null; then
  $PY -m mangahue_train.export_quantized \
    --model "$ONNX" \
    --output "$INT8" \
    --dynamic \
    --channels 3 \
    --spatial 512 || echo "!! export_quantized failed — shipping fp32/fp16 instead"
else
  echo "!! mangahue_train not in venv — skipping int8 (ship fp32)"
fi

echo "==> restarting the bot"
systemctl start mangahue-pro
sleep 5
systemctl is-active mangahue-pro

echo ""
echo "==> DONE. Artifacts:"
ls -la "$MODEL_DIR"
echo ""
echo "Next: point the bot at the new model:"
echo "  --model $MODEL_DIR/style2paints-v3.onnx  (or -int8.onnx)"
echo "  (I handle the config swap + benchmark from here.)"
