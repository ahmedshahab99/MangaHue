from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CanvasTransform:
    original_size: tuple[int, int]
    content_box: tuple[int, int, int, int]


def _aligned(value: int, alignment: int = 64) -> int:
    return max(alignment, ((value + alignment - 1) // alignment) * alignment)


def prepare_canvas(source: Image.Image, max_dimension: int) -> tuple[Image.Image, CanvasTransform]:
    """Fit a page into a 64-aligned canvas without stretching or cropping."""
    if max_dimension < 64:
        raise ValueError("max_dimension must be at least 64")
    image = source.convert("RGB")
    scale = min(max_dimension / image.width, max_dimension / image.height, 1.0)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas_size = (
        min(max_dimension, _aligned(resized.width)),
        min(max_dimension, _aligned(resized.height)),
    )
    left = (canvas_size[0] - resized.width) // 2
    top = (canvas_size[1] - resized.height) // 2
    box = (left, top, left + resized.width, top + resized.height)
    canvas = Image.new("RGB", canvas_size, "white")
    canvas.paste(resized, (left, top))
    return canvas, CanvasTransform(image.size, box)


def restore_canvas(canvas: Image.Image, transform: CanvasTransform) -> Image.Image:
    content = canvas.convert("RGB").crop(transform.content_box)
    return content.resize(transform.original_size, Image.Resampling.LANCZOS)


def fuse_generated_chroma(source: Image.Image, generated: Image.Image) -> Image.Image:
    """Use generated chroma while preserving source luminance and white paper."""
    source_rgb = source.convert("RGB")
    generated_rgb = generated.convert("RGB").resize(source_rgb.size, Image.Resampling.LANCZOS)
    source_y = np.asarray(source_rgb.convert("YCbCr"), dtype=np.float32)[..., 0]
    generated_ycc = np.asarray(generated_rgb.convert("YCbCr"), dtype=np.float32)
    cb = generated_ycc[..., 1] - 128.0
    cr = generated_ycc[..., 2] - 128.0
    offsets = np.stack(
        (1.402 * cr, -0.344136 * cb - 0.714136 * cr, 1.772 * cb), axis=-1
    )
    # Scale chroma into the RGB gamut instead of clipping channels. Clipping
    # is what changes luminance in dark or bright areas.
    scale = np.ones(source_y.shape, dtype=np.float32)
    for channel in range(3):
        delta = offsets[..., channel]
        positive = delta > 0
        negative = delta < 0
        scale[positive] = np.minimum(
            scale[positive], (255.0 - source_y[positive]) / delta[positive]
        )
        scale[negative] = np.minimum(
            scale[negative], source_y[negative] / -delta[negative]
        )
    scale = np.clip(scale, 0.0, 1.0)
    scale[source_y >= 245] = 0.0
    rgb = source_y[..., None] + offsets * scale[..., None]
    return Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO studio colorization worker")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-dimension", type=int, default=576)
    args = parser.parse_args()
    try:
        import openvino_genai  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "openvino-genai is required for studio inference; install mangahue-train[studio]"
        ) from error
    raise SystemExit(
        "The model-specific OpenVINO generation adapter is not configured. "
        "Use the production DDColor ONNX worker until a licensed studio model is selected."
    )


if __name__ == "__main__":
    main()
