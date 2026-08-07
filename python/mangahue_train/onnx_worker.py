from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

SKIN      = np.array([242, 208, 182], dtype=np.float32) / 255.0
SKIN_SHA  = np.array([200, 148, 122], dtype=np.float32) / 255.0
HAIR_D    = np.array([38, 28, 34], dtype=np.float32) / 255.0
HAIR_M    = np.array([65, 48, 44], dtype=np.float32) / 255.0
CLOTH_L   = np.array([178, 168, 160], dtype=np.float32) / 255.0
CLOTH_D   = np.array([52, 48, 56], dtype=np.float32) / 255.0
BG_WARM   = np.array([228, 222, 215], dtype=np.float32) / 255.0
BG_COOL   = np.array([195, 192, 200], dtype=np.float32) / 255.0
WHITE     = np.array([255, 253, 250], dtype=np.float32) / 255.0
HINT_STRENGTH = 0.75


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def _fast_label(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Fast connected components using cv2 (10-100x faster than scipy)."""
    uint8 = (mask.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uint8, connectivity=8)
    result = np.zeros_like(mask, dtype=bool)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = True
    return result


def _detect_regions(gray: np.ndarray) -> dict[str, np.ndarray]:
    h, w = gray.shape
    ink = gray < 0.08
    bright = gray > 0.88
    uint8_bright = (bright.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uint8_bright, connectivity=8)
    bubble = np.zeros_like(gray, dtype=bool)
    max_bubble_area = gray.size * 0.12
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 100 < area < max_bubble_area:
            bubble[labels == i] = True
    skin = (gray > 0.38) & (gray < 0.92) & ~bubble & ~ink
    skin_f = _fast_label(skin, min_area=60)
    hair = (gray > 0.08) & (gray < 0.35) & ~ink
    hair_f = _fast_label(hair, min_area=50)
    clothing = (gray > 0.12) & (gray < 0.75) & ~skin_f & ~hair_f & ~bubble & ~ink
    bg = ~ink & ~skin_f & ~hair_f & ~clothing & ~bubble
    return {"ink": ink, "bubble": bubble, "skin": skin_f, "hair": hair_f, "clothing": clothing, "bg": bg}


def _make_hints(gray: np.ndarray, regions: dict[str, np.ndarray], strength: float = HINT_STRENGTH) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    cmap = np.zeros((h, w, 3), dtype=np.float32)
    cmask = np.zeros((h, w), dtype=np.float32)
    skin = regions["skin"]
    cmap[skin & (gray > 0.6)] = SKIN
    cmap[skin & (gray <= 0.6)] = SKIN_SHA
    cmask[skin] = strength
    hair = regions["hair"]
    cmap[hair & (gray < 0.2)] = HAIR_D
    cmap[hair & (gray >= 0.2)] = HAIR_M
    cmask[hair] = strength * 0.85
    cloth = regions["clothing"]
    cmap[cloth & (gray > 0.45)] = CLOTH_L
    cmap[cloth & (gray <= 0.45)] = CLOTH_D
    cmask[cloth] = strength * 0.7
    bg = regions["bg"]
    cmap[bg & (gray > 0.5)] = BG_WARM
    cmap[bg & (gray <= 0.5)] = BG_COOL
    cmask[bg] = strength * 0.55
    cmap[regions["bubble"]] = WHITE
    cmask[regions["bubble"]] = strength * 0.95
    cmask = cv2.GaussianBlur(cmask, (5, 5), 2.0)
    return cmap, np.clip(cmask, 0, 1)


def _load_input(request: dict, channels: int, spatial_multiple: int) -> tuple[np.ndarray, int, int, np.ndarray]:
    gray = cv2.imread(str(request["lineart"]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    height, width = gray.shape
    if channels == 1:
        return gray[None, None, :, :], height, width, gray
    regions = _detect_regions(gray)
    cmap, cmask = _make_hints(gray, regions)
    user_map = request.get("hint_map")
    user_mask = request.get("hint_mask")
    if user_map and user_mask:
        umap_path = Path(user_map)
        umask_path = Path(user_mask)
        if umap_path.is_file() and umask_path.is_file():
            umap = cv2.imread(str(umap_path), cv2.IMREAD_COLOR)
            umask = cv2.imread(str(umask_path), cv2.IMREAD_GRAYSCALE)
            if umap is not None and umask is not None:
                umap = cv2.cvtColor(umap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                umask = umask.astype(np.float32) / 255.0
                if umap.shape[:2] == (height, width) and umask.shape == (height, width):
                    # Rust has the full-page bubble, character-memory, and
                    # region-lock context. Its map is authoritative; mixing
                    # the worker's grayscale-only guesses back into unhinted
                    # pixels mislabeled paper and clothing as skin.
                    cmap = umap
                    cmask = umask
                else:
                    print(json.dumps({
                        "warn": "hint_map/hint_mask dimensions do not match lineart; using internal hints",
                    }), flush=True, file=sys.stderr)
        else:
            print(json.dumps({
                "warn": "hint_map/hint_mask paths missing; using internal hints",
            }), flush=True, file=sys.stderr)
    # Contract input_range: "0_1" — hint RGB (0..1) and hint mask (0..1) are
    # kept as separate channels, exactly like dataset.py training inputs.
    tensor = np.concatenate(
        [gray[None, :, :], cmap.transpose(2, 0, 1), cmask[None, :, :]],
        axis=0,
    )[None, :, :, :]
    ph = ((height + spatial_multiple - 1) // spatial_multiple) * spatial_multiple
    pw = ((width + spatial_multiple - 1) // spatial_multiple) * spatial_multiple
    if (ph, pw) != (height, width):
        padded = np.zeros((1, channels, ph, pw), dtype=np.float32)
        padded[:, :, :height, :width] = tensor
        padded[:, 0, height:, :] = 1.0
        padded[:, 0, :, width:] = 1.0
        tensor = padded
    return tensor, height, width, gray


def _post_color(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    purple = np.clip((r + b) / 2.0 - g, 0, 1)
    midtone = (luma > 0.15) & (luma < 0.85)
    kill = purple * midtone * 0.3
    rgb[:, :, 2] -= kill
    rgb[:, :, 0] -= kill * 0.5
    rgb[gray > 0.92] = 1.0
    rgb[gray < 0.08] = 0.0
    return np.clip(rgb, 0, 1)


def _run_request(
    session: ort.InferenceSession,
    request: dict,
    channels: int,
    input_name: str,
    output_name: str,
    spatial_multiple: int,
) -> dict:
    t0 = time.monotonic()
    tensor, height, width, gray = _load_input(request, channels, spatial_multiple)
    t_load = time.monotonic()
    result = session.run([output_name], {input_name: tensor})[0]
    t_infer = time.monotonic()
    if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
        raise ValueError(f"unexpected ONNX output shape: {result.shape}")
    rgb = np.clip(result[0, :, :height, :width].transpose(1, 2, 0), 0.0, 1.0)
    rgb = _post_color(rgb, gray)
    image = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    output = Path(str(request["output"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
    t_save = time.monotonic()
    return {
        "ok": True,
        "id": request.get("id"),
        "width": image.width,
        "height": image.height,
        "ms_load": int((t_load - t0) * 1000),
        "ms_infer": int((t_infer - t_load) * 1000),
        "ms_save": int((t_save - t_infer) * 1000),
    }


def main() -> None:
    args = _parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    channels = int(metadata["input_channels"])
    spatial_multiple = int(metadata.get("contract", {}).get("spatial_multiple", 1))
    threads = max(2, min(args.threads, 8))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["MALLOC_ARENA_MAX"] = "2"
    options = ort.SessionOptions()
    options.inter_op_num_threads = threads
    options.intra_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    print(json.dumps({"status": "loading", "model": str(args.model), "threads": threads}), flush=True, file=sys.stderr)
    t0 = time.monotonic()
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    _ = session.run([output_name], {input_name: np.zeros((1, channels, 64, 64), dtype=np.float32)})
    t_load = time.monotonic()
    print(json.dumps({"status": "ready", "backend": "onnx-hints-cv2", "threads": threads, "load_ms": int((t_load - t0) * 1000)}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _run_request(session, request, channels, input_name, output_name, spatial_multiple)
        except Exception as error:
            response = {"ok": False, "error": str(error), "traceback": traceback.format_exc(limit=8)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
