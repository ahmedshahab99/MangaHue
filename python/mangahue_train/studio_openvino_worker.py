from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DEFAULT_PROMPT = (
    "masterpiece professional colored manga page, polished anime illustration, "
    "natural warm skin tones, detailed hair with soft highlights, realistic fabric "
    "colors, cinematic soft lighting, rich controlled color harmony, clean cel "
    "shading, preserve exact composition, preserve line art, speech bubbles and text"
)


@dataclass(frozen=True)
class CanvasTransform:
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]
    content_box: tuple[int, int, int, int]


def _multiple_of_64(value: int) -> int:
    return max(256, int(math.ceil(value / 64.0)) * 64)


def prepare_canvas(
    image: Image.Image,
    long_edge: int,
) -> tuple[Image.Image, CanvasTransform]:
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("input image has invalid dimensions")

    scale = long_edge / max(width, height)
    content_width = max(1, int(round(width * scale)))
    content_height = max(1, int(round(height * scale)))
    canvas_width = _multiple_of_64(content_width)
    canvas_height = _multiple_of_64(content_height)
    resized = source.resize((content_width, content_height), Image.Resampling.LANCZOS)
    left = (canvas_width - content_width) // 2
    top = (canvas_height - content_height) // 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(resized, (left, top))
    transform = CanvasTransform(
        source_size=source.size,
        canvas_size=canvas.size,
        content_box=(left, top, left + content_width, top + content_height),
    )
    return canvas, transform


def restore_canvas(image: Image.Image, transform: CanvasTransform) -> Image.Image:
    if image.size != transform.canvas_size:
        image = image.resize(transform.canvas_size, Image.Resampling.LANCZOS)
    return image.crop(transform.content_box).resize(
        transform.source_size,
        Image.Resampling.LANCZOS,
    )


def fuse_generated_chroma(
    lineart: Image.Image,
    generated: Image.Image,
    chroma_strength: float = 0.92,
) -> Image.Image:
    source = lineart.convert("RGB")
    candidate = generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    source_ycc = np.asarray(source.convert("YCbCr"), dtype=np.float32)
    candidate_ycc = np.asarray(candidate.convert("YCbCr"), dtype=np.float32)

    luminance = source_ycc[..., 0]
    candidate_chroma = candidate_ycc[..., 1:3] - 128.0
    chroma_magnitude = np.linalg.norm(candidate_chroma, axis=-1)

    # Paper and speech balloons remain neutral. A blurred darkness map retains color
    # near light skin and fabric while suppressing color in truly empty white areas.
    darkness = np.asarray(
        Image.fromarray((255.0 - luminance).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(2.0, min(source.size) / 220.0))
        ),
        dtype=np.float32,
    )
    structure_weight = np.clip((darkness - 2.0) / 22.0, 0.0, 1.0)
    paper_weight = np.clip((248.0 - luminance) / 18.0, 0.0, 1.0)
    confidence = np.maximum(structure_weight, paper_weight * 0.72)
    confidence *= np.clip(chroma_magnitude / 10.0, 0.0, 1.0)

    # Preserve subtle colors and roll off extreme hallucinated saturation.
    magnitude_scale = np.minimum(1.0, 58.0 / np.maximum(chroma_magnitude, 1.0))
    chroma = candidate_chroma * magnitude_scale[..., None]
    chroma *= (confidence * np.clip(chroma_strength, 0.0, 1.25))[..., None]

    cb = chroma[..., 0]
    cr = chroma[..., 1]
    delta = np.stack(
        (
            1.402 * cr,
            -0.344136 * cb - 0.714136 * cr,
            1.772 * cb,
        ),
        axis=-1,
    )
    y = luminance[..., None]
    positive_limit = np.where(delta > 0.0, (255.0 - y) / np.maximum(delta, 1e-6), 1.0)
    negative_limit = np.where(delta < 0.0, y / np.maximum(-delta, 1e-6), 1.0)
    gamut_scale = np.minimum(
        1.0,
        np.minimum(positive_limit.min(axis=-1), negative_limit.min(axis=-1)),
    )
    rgb = y + delta * gamut_scale[..., None]
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).round().astype(np.uint8), mode="RGB")


def generate(
    model_root: Path,
    input_path: Path,
    output_path: Path,
    *,
    long_edge: int,
    steps: int,
    strength: float,
    seed: int,
    prompt: str,
    cache_dir: Path | None,
    raw_output: Path | None,
) -> dict[str, object]:
    import openvino as ov
    import openvino_genai as ov_genai

    started = time.monotonic()
    with Image.open(input_path) as source_file:
        source = ImageOps.exif_transpose(source_file).convert("RGB")
    canvas, transform = prepare_canvas(source, long_edge)

    properties: dict[str, str] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        properties["CACHE_DIR"] = str(cache_dir)
    pipeline = ov_genai.Image2ImagePipeline(str(model_root), "CPU", **properties)
    loaded_at = time.monotonic()
    tensor = ov.Tensor(np.asarray(canvas, dtype=np.uint8)[None, ...])
    result = pipeline.generate(
        prompt,
        tensor,
        width=canvas.width,
        height=canvas.height,
        num_inference_steps=steps,
        guidance_scale=1.0,
        strength=strength,
        rng_seed=seed,
    )
    generated_at = time.monotonic()
    raw = restore_canvas(Image.fromarray(result.data[0]), transform)
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw.save(raw_output, format="PNG", optimize=False)

    fused = fuse_generated_chroma(source, raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.save(output_path, format="PNG", optimize=False)
    return {
        "ok": True,
        "backend": "openvino-lcm-studio",
        "width": source.width,
        "height": source.height,
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "load_seconds": round(loaded_at - started, 3),
        "generation_seconds": round(generated_at - loaded_at, 3),
        "total_seconds": round(time.monotonic() - started, 3),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MangaHue OpenVINO Studio colorizer")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--long-edge", type=int, default=576)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--strength", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=7391)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.long_edge < 256:
        raise ValueError("--long-edge must be at least 256")
    if args.steps < 2 or args.steps > 12:
        raise ValueError("--steps must be between 2 and 12")
    if not 0.05 <= args.strength <= 1.0:
        raise ValueError("--strength must be between 0.05 and 1.0")
    if not args.model_root.joinpath("unet", "openvino_model.xml").is_file():
        raise FileNotFoundError(f"invalid OpenVINO diffusion model: {args.model_root}")

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.threads)))
    result = generate(
        args.model_root.resolve(),
        args.input.resolve(),
        args.output.resolve(),
        long_edge=args.long_edge,
        steps=args.steps,
        strength=args.strength,
        seed=args.seed,
        prompt=args.prompt,
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        raw_output=args.raw_output.resolve() if args.raw_output else None,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
