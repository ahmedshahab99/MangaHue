from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent MangaHue ONNX colorizer worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def _load_input(request: dict[str, object], channels: int) -> np.ndarray:
    with Image.open(str(request["lineart"])) as image:
        lineart = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = lineart.shape

    if channels == 1:
        return lineart[None, None, :, :]
    if channels != 5:
        raise ValueError(f"unsupported input channel count: {channels}")

    with Image.open(str(request["hint_map"])) as image:
        hint_map = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(str(request["hint_mask"])) as image:
        hint_mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if hint_map.shape[:2] != (height, width) or hint_mask.shape != (height, width):
        raise ValueError("hint map/mask dimensions do not match lineart")

    normalized_hints = ((hint_map - 0.5) / 0.5) * hint_mask[:, :, None]
    return np.concatenate(
        [
            lineart[None, :, :],
            normalized_hints.transpose(2, 0, 1),
            hint_mask[None, :, :],
        ],
        axis=0,
    )[None, :, :, :]


def _run_request(
    session: ort.InferenceSession,
    request: dict[str, object],
    channels: int,
    input_name: str,
    output_name: str,
) -> dict[str, object]:
    tensor = _load_input(request, channels).astype(np.float32, copy=False)
    result = session.run([output_name], {input_name: tensor})[0]
    if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
        raise ValueError(f"unexpected ONNX output shape: {result.shape}")

    rgb = np.clip(result[0].transpose(1, 2, 0), 0.0, 1.0)
    image = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    output = Path(str(request["output"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
    return {
        "ok": True,
        "id": request.get("id"),
        "width": image.width,
        "height": image.height,
    }


def main() -> None:
    args = _parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    channels = int(metadata["input_channels"])

    threads = max(1, min(args.threads, 8))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(
        json.dumps(
            {
                "status": "ready",
                "backend": "onnx",
                "input": input_name,
                "output": output_name,
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _run_request(
                session,
                request,
                channels,
                input_name,
                output_name,
            )
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
