from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent anime face palette worker")
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def _rgb_from_luma_chroma(
    luma: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
) -> np.ndarray:
    green = (-0.2126 * red - 0.0722 * blue) / 0.7152
    channels = np.stack((red, green, blue), axis=-1)
    positive = np.where(channels > 0.0, (255.0 - luma[..., None]) / np.maximum(channels, 1e-6), 1.0)
    negative = np.where(channels < 0.0, luma[..., None] / np.maximum(-channels, 1e-6), 1.0)
    scale = np.minimum(1.0, np.minimum(positive, negative).min(axis=-1))
    return np.clip(luma[..., None] + channels * scale[..., None], 0.0, 255.0)


def _detect_faces(cascade: cv2.CascadeClassifier, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    scale = min(1.0, 1024.0 / max(height, width))
    detect = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    minimum = max(20, round(min(detect.shape) * 0.045))
    found = cascade.detectMultiScale(
        detect,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(minimum, minimum),
    )
    inverse = 1.0 / scale
    faces = [
        (
            max(0, round(float(x) * inverse)),
            max(0, round(float(y) * inverse)),
            min(width, round(float(w) * inverse)),
            min(height, round(float(h) * inverse)),
        )
        for x, y, w, h in found
    ]
    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)


def _face_palette_candidate(
    gray: np.ndarray,
    faces: list[tuple[int, int, int, int]],
) -> np.ndarray:
    height, width = gray.shape
    luma = gray.astype(np.float32)
    red = np.zeros_like(luma)
    blue = np.zeros_like(luma)
    confidence = np.zeros_like(luma)
    yy, xx = np.mgrid[0:height, 0:width]

    for x, y, w, h in faces:
        center_x = x + w * 0.5
        center_y = y + h * 0.57
        radius_x = max(8.0, w * 0.43)
        radius_y = max(8.0, h * 0.43)
        dx = (xx - center_x) / radius_x
        dy = (yy - center_y) / radius_y
        radial = np.clip(1.0 - dx * dx - dy * dy, 0.0, 1.0)
        drawable = np.clip((luma - 92.0) / 84.0, 0.0, 1.0)
        face_weight = np.sqrt(radial) * drawable

        cheek_y = y + h * 0.62
        cheek_radius_x = max(4.0, w * 0.12)
        cheek_radius_y = max(3.0, h * 0.07)
        left_cheek = np.exp(
            -2.5
            * (
                ((xx - (x + w * 0.31)) / cheek_radius_x) ** 2
                + ((yy - cheek_y) / cheek_radius_y) ** 2
            )
        )
        right_cheek = np.exp(
            -2.5
            * (
                ((xx - (x + w * 0.69)) / cheek_radius_x) ** 2
                + ((yy - cheek_y) / cheek_radius_y) ** 2
            )
        )
        blush = np.maximum(left_cheek, right_cheek) * face_weight
        replace = face_weight > confidence
        red_candidate = 38.0 + blush * 13.0
        blue_candidate = -26.0 - blush * 5.0
        red = np.where(replace, red_candidate, red)
        blue = np.where(replace, blue_candidate, blue)
        confidence = np.maximum(confidence, face_weight)

    red *= confidence
    blue *= confidence
    palette_luma = luma * (1.0 - confidence) + np.minimum(luma, 212.0) * confidence
    rgb = _rgb_from_luma_chroma(palette_luma, red, blue)
    ink = np.clip((88.0 - luma) / 68.0, 0.0, 1.0)[..., None]
    neutral = np.repeat(luma[..., None], 3, axis=-1)
    return np.clip(rgb * (1.0 - ink) + neutral * ink, 0.0, 255.0).round().astype(np.uint8)


def colorize(
    cascade: cv2.CascadeClassifier,
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    with Image.open(input_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.uint8)
    faces = _detect_faces(cascade, gray)
    output = _face_palette_candidate(gray, faces)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path, format="PNG", optimize=False)
    return {
        "ok": True,
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "faces": [list(face) for face in faces],
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))
    cv2.setNumThreads(max(1, args.threads))
    cascade = cv2.CascadeClassifier(str(args.cascade.resolve()))
    if cascade.empty():
        raise RuntimeError(f"failed to load anime face cascade: {args.cascade}")
    print(json.dumps({"status": "ready", "backend": "anime-face-palette"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = colorize(
                cascade,
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
