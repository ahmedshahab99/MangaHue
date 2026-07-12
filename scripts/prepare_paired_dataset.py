from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def colorfulness(image: Image.Image) -> float:
    sample = image.convert("RGB").resize((64, 64))
    array = np.asarray(sample, dtype=np.float32)
    return float(np.abs(array[..., 0] - array[..., 1]).mean() + np.abs(array[..., 1] - array[..., 2]).mean())


def split_pair(path: Path) -> tuple[Image.Image, Image.Image] | None:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width >= height * 1.8:
        midpoint = width // 2
        first, second = image.crop((0, 0, midpoint, height)), image.crop((midpoint, 0, width, height))
    elif height >= width * 1.8:
        midpoint = height // 2
        first, second = image.crop((0, 0, width, midpoint)), image.crop((0, midpoint, width, height))
    else:
        return None
    if first.size != second.size or min(first.size) < 64:
        return None
    return (first, second) if colorfulness(first) <= colorfulness(second) else (second, first)


def stable_split(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare side-by-side sketch/color pairs")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = args.root / "mangahue-prepared"
    lineart_dir, color_dir = output / "lineart", output / "color"
    lineart_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        path for path in args.root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and output not in path.parents
    ]
    manifests: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    rejected = 0
    for index, path in enumerate(sorted(candidates)):
        try:
            pair = split_pair(path)
            if pair is None:
                rejected += 1
                continue
            lineart, color = pair
            sample_id = hashlib.sha256(str(path.relative_to(args.root)).encode()).hexdigest()[:24]
            lineart_path, color_path = lineart_dir / f"{sample_id}.png", color_dir / f"{sample_id}.png"
            lineart.convert("L").save(lineart_path, optimize=True)
            color.save(color_path, optimize=True)
            split = stable_split(sample_id)
            manifests[split].append({
                "sample_id": sample_id,
                "group_id": sample_id,
                "lineart": str(lineart_path.relative_to(output)).replace("\\", "/"),
                "color": str(color_path.relative_to(output)).replace("\\", "/"),
                "provenance": "ktaebum/anime-sketch-colorization-pair",
                "license": "CC0-1.0",
            })
        except (OSError, ValueError):
            rejected += 1
        if index and index % 500 == 0:
            print(f"processed {index}/{len(candidates)}", flush=True)
    for split, samples in manifests.items():
        payload = {"schema_version": 2, "dataset": "anime-sketch-colorization-pair", "split": split, "samples": samples}
        (output / f"{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {"candidates": len(candidates), "rejected": rejected, **{key: len(value) for key, value in manifests.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
