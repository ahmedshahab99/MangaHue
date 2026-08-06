#!/usr/bin/env python
"""Persistent DDColor worker for MangaHue — JSON-line protocol over stdin/stdout."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

_MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "ddcolor_repo"
sys.path.insert(0, str(_MODEL_ROOT))

from ddcolor import DDColor, build_ddcolor_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent DDColor worker")
    p.add_argument("--model-path", type=Path, required=True, help="Path to DDColor .pt weights")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--input-size", type=int, default=512, help="Model input resolution")
    p.add_argument("--model-size", choices=("tiny", "large"), default="tiny", help="Model variant")
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


def _run_request(model, request: dict, input_size: int, device) -> dict:
    lineart_path = str(request["lineart"])
    output_path = Path(str(request["output"]))

    with Image.open(lineart_path) as src:
        original_size = src.size
        img = np.asarray(src.convert("RGB"), dtype=np.float32) / 255.0

    # DDColor outputs ab channels in LAB space; merge with the input's L.
    # Model expects a raw [0,1] RGB tensor (do_normalize=False).
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        result = model(tensor)

    if isinstance(result, torch.Tensor):
        ab = result.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    else:
        ab = np.asarray(result).squeeze(0).transpose(1, 2, 0)

    # Build LAB and convert to RGB (mirrors ColorizationPipeline.process).
    if ab.shape[2] != 2:
        raise ValueError(f"unexpected DDColor output channels: {ab.shape}")
    img_lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
    l_channel = img_lab[:, :, :1]
    # DDColor ab is in [-110, 110] roughly; scale from tanh output.
    merged = np.concatenate([l_channel, ab * 110.0], axis=2)
    rgb = cv2.cvtColor(merged.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=False)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"ok": True, "id": request.get("id"), "width": original_size[0], "height": original_size[1]}


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"status": "error", "message": "CUDA requested but unavailable"}), flush=True)
        sys.exit(1)

    model = build_ddcolor_model(
        DDColor,
        model_path=str(args.model_path),
        input_size=args.input_size,
        model_size=args.model_size,
        device=device,
    )

    print(json.dumps({"status": "ready", "backend": "ddcolor-v1", "size": args.input_size}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _run_request(model, request, args.input_size, device)
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
