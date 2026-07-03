from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MangaHue ONNX tile")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lineart", type=Path, required=True)
    parser.add_argument("--hint-map", type=Path, required=True)
    parser.add_argument("--hint-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    channels = int(metadata["input_channels"])
    lineart = np.asarray(Image.open(args.lineart).convert("L"), dtype=np.float32) / 255.0
    h, w = lineart.shape

    if channels == 1:
        x = lineart[None, None, :, :]
    elif channels == 5:
        hint_map = np.asarray(Image.open(args.hint_map).convert("RGB"), dtype=np.float32) / 255.0
        hint_mask = np.asarray(Image.open(args.hint_mask).convert("L"), dtype=np.float32) / 255.0
        if hint_map.shape[:2] != (h, w) or hint_mask.shape != (h, w):
            raise SystemExit("hint map/mask dimensions do not match lineart")
        x = np.concatenate(
            [
                lineart[None, :, :],
                hint_map.transpose(2, 0, 1),
                hint_mask[None, :, :],
            ],
            axis=0,
        )[None, :, :, :]
    else:
        raise SystemExit(f"unsupported input_channels: {channels}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    y = session.run([output_name], {input_name: x.astype(np.float32)})[0]
    if y.shape[0] != 1 or y.shape[1] != 3:
        raise SystemExit(f"unexpected ONNX output shape: {y.shape}")
    image = (np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(args.output)


if __name__ == "__main__":
    main()
