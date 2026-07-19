from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent MangaHue pretrained colorizer worker")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def _validate_model_root(root: Path) -> None:
    required = (
        root / "colorizator.py",
        root / "networks" / "generator.zip",
        root / "denoising" / "models" / "net_rgb.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pretrained model files: " + ", ".join(missing))


def _load_colorizer(root: Path, device: str):
    sys.path.insert(0, str(root))
    module = importlib.import_module("colorizator")
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        return module.MangaColorizator(
            device,
            str(root / "networks" / "generator.zip"),
            str(root / "networks" / "extractor.pth"),
        )
    finally:
        os.chdir(previous_cwd)


def _open_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _apply_hints(colorizer, hint_map_path: str, hint_mask_path: str) -> None:
    target_h, target_w = colorizer.current_image.shape[-2:]
    pad_h, pad_w = colorizer.current_pad
    content_h = target_h - pad_h
    content_w = target_w - pad_w

    with Image.open(hint_map_path) as image:
        hint = np.asarray(
            image.convert("RGB").resize((content_w, content_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    with Image.open(hint_mask_path) as image:
        mask = np.asarray(
            image.convert("L").resize((content_w, content_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    if pad_h or pad_w:
        hint = np.pad(hint, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    colorizer.update_hint(hint, mask)


def _run_request(colorizer, request: dict[str, object], size: int) -> dict[str, object]:
    input_path = str(request["lineart"])
    output_path = Path(str(request["output"]))
    with Image.open(input_path) as source:
        original_size = source.size
        lineart = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0

    colorizer.set_image(lineart, size=size, apply_denoise=False)
    hint_map = request.get("hint_map")
    hint_mask = request.get("hint_mask")
    if hint_map and hint_mask:
        _apply_hints(colorizer, str(hint_map), str(hint_mask))

    with torch.inference_mode():
        result = colorizer.colorize()
    image = Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=False)

    colorizer.current_image = None
    colorizer.current_hint = None
    gc.collect()
    return {"ok": True, "id": request.get("id"), "width": original_size[0], "height": original_size[1]}


def main() -> None:
    args = _parse_args()
    if args.size < 256 or args.size % 32:
        raise ValueError("--size must be at least 256 and divisible by 32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    _validate_model_root(args.model_root)
    colorizer = _load_colorizer(args.model_root.resolve(), args.device)
    print(json.dumps({"status": "ready", "backend": "manga-colorization-v2", "size": args.size}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _run_request(colorizer, request, args.size)
        except Exception as error:  # The Rust caller needs a structured failure, not a dead worker.
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
