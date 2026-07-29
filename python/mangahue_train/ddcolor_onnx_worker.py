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

MODEL_SIZE = 512


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent DDColor ONNX worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def _session(model: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    return ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - 0.055,
    )


def _rgb_to_lab_l(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    y = (
        linear[..., 0] * 0.2126729
        + linear[..., 1] * 0.7151522
        + linear[..., 2] * 0.0721750
    )
    delta = 6.0 / 29.0
    f = np.where(
        y > delta**3,
        np.cbrt(y),
        y / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return 116.0 * f - 16.0


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    l_channel = lab[..., 0]
    a_channel = lab[..., 1]
    b_channel = lab[..., 2]
    fy = (l_channel + 16.0) / 116.0
    fx = fy + a_channel / 500.0
    fz = fy - b_channel / 200.0
    delta = 6.0 / 29.0

    def inverse(value: np.ndarray) -> np.ndarray:
        return np.where(
            value > delta,
            value**3,
            3.0 * delta**2 * (value - 4.0 / 29.0),
        )

    x = inverse(fx) * 0.95047
    y = inverse(fy)
    z = inverse(fz) * 1.08883
    linear = np.stack(
        (
            x * 3.2404542 + y * -1.5371385 + z * -0.4985314,
            x * -0.9692660 + y * 1.8760108 + z * 0.0415560,
            x * 0.0556434 + y * -0.2040259 + z * 1.0572252,
        ),
        axis=-1,
    )
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0)


def _resize_float_channel(channel: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(channel.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def colorize(
    session: ort.InferenceSession,
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    with Image.open(input_path) as source:
        original = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        original_size = source.size
        resized = np.asarray(
            source.convert("RGB").resize(
                (MODEL_SIZE, MODEL_SIZE),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0

    l_resized = _rgb_to_lab_l(resized)
    neutral_lab = np.stack(
        (l_resized, np.zeros_like(l_resized), np.zeros_like(l_resized)),
        axis=-1,
    )
    neutral_rgb = _lab_to_rgb(neutral_lab)
    tensor = np.ascontiguousarray(neutral_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    output_ab = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    if output_ab.shape != (2, MODEL_SIZE, MODEL_SIZE):
        raise ValueError(f"unexpected DDColor output shape: {output_ab.shape}")

    width, height = original_size
    a_channel = _resize_float_channel(output_ab[0], (width, height))
    b_channel = _resize_float_channel(output_ab[1], (width, height))
    original_l = _rgb_to_lab_l(original)
    output_lab = np.stack((original_l, a_channel, b_channel), axis=-1)
    output_rgb = (_lab_to_rgb(output_lab) * 255.0).round().astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        output_path,
        format="PNG",
        optimize=False,
    )
    return {
        "ok": True,
        "width": width,
        "height": height,
        "mean_ab": float(np.mean(np.abs(output_ab))),
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    session = _session(args.model.resolve(), args.threads)
    print(
        json.dumps(
            {
                "status": "ready",
                "backend": "ddcolor-tiny-onnx",
                "size": MODEL_SIZE,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = colorize(
                session,
                Path(request["lineart"]),
                Path(request["output"]),
            )
            response["id"] = request.get("id")
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
