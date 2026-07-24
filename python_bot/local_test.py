"""Local test — colorize one manga page and save locally for review."""
import sys, os, asyncio
sys.path.insert(0, '/opt/mangahue_bot')

import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
from pathlib import Path

from hints import AdvancedHintGenerator
from postprocess import apply_manga_pipeline

MODEL_PATH = "/opt/MangaHue/models/manga-onnx/manga-colorize-fp16.onnx"
TEST_URL = sys.argv[1] if len(sys.argv) > 1 else "https://zrocdn.xyz/galleries/1033488/1.jpg"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mangahue_test_output.png"
TILE = 384

async def download(url):
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={"User-Agent": "Mozilla/5.0"}) as r:
            return await r.read()

async def main():
    print(f"Downloading: {TEST_URL}")
    data = await download(TEST_URL)
    in_path = "/tmp/mangahue_test_input.jpg"
    with open(in_path, "wb") as f:
        f.write(data)
    print(f"Saved to {in_path}, size: {len(data)} bytes")

    # Colorize
    img = Image.open(in_path).convert("L")
    orig_w, orig_h = img.size
    print(f"Image: {orig_w}x{orig_h}")
    orig_arr = np.array(img, dtype=np.float32) / 255.0

    print("Generating hints...")
    gen = AdvancedHintGenerator()
    hm, hk = gen.generate(np.array(img))
    hm_pil = Image.fromarray(hm).filter(ImageFilter.GaussianBlur(radius=12))
    hk_pil = Image.fromarray(hk).filter(ImageFilter.GaussianBlur(radius=12))
    hm = np.array(hm_pil)
    hk = np.array(hk_pil)
    print(f"Hints: map={hm.shape}, mask={hk.shape}")

    print("Loading ONNX model...")
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    print(f"Model loaded: input={iname}, output={oname}")

    # Tiling
    result = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
    weight = np.zeros((orig_h, orig_w), dtype=np.float32)
    overlap = 48
    step = TILE - overlap
    total_tiles = 0

    for y in range(0, orig_h, step):
        for x in range(0, orig_w, step):
            x2 = min(x + TILE, orig_w)
            y2 = min(y + TILE, orig_h)
            x1 = max(0, x2 - TILE)
            y1 = max(0, y2 - TILE)

            tile_arr = orig_arr[y1:y2, x1:x2]
            tile_hm = hm[y1:y2, x1:x2]
            tile_hk = hk[y1:y2, x1:x2]

            h, w = tile_arr.shape
            hf = tile_hm.astype(np.float32) / 255.0
            hkf = tile_hk.astype(np.float32) / 255.0
            ph, pw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
            t = np.zeros((1, 5, ph, pw), dtype=np.float32)
            t[0, 0, :h, :w] = tile_arr
            t[0, 1, :h, :w] = hf[:, :, 0]
            t[0, 2, :h, :w] = hf[:, :, 1]
            t[0, 3, :h, :w] = hf[:, :, 2]
            t[0, 4, :h, :w] = hkf

            out = sess.run([oname], {iname: t})[0]
            tile_rgb = out[0].transpose(1, 2, 0)[:h, :w]

            th, tw = tile_rgb.shape[:2]
            wy = np.hanning(th)[:, None].astype(np.float32)
            wx = np.hanning(tw)[None, :].astype(np.float32)
            blend = wy * wx

            result[y1:y2, x1:x2] += tile_rgb * blend[:, :, None]
            weight[y1:y2, x1:x2] += blend
            total_tiles += 1

    mask = weight > 1e-6
    result[mask] /= weight[mask, None]
    result = (result * 255).clip(0, 255).astype(np.uint8)
    print(f"Colorized: {total_tiles} tiles")

    # Save intermediate
    colored_path = "/tmp/mangahue_test_colored.png"
    Image.fromarray(result).save(colored_path)
    print(f"Saved colored: {colored_path}")

    # Post-process
    print("Post-processing...")
    apply_manga_pipeline(colored_path, in_path, OUTPUT)
    print(f"Final: {OUTPUT}")

    # Cleanup
    os.remove(in_path)
    os.remove(colored_path)

if __name__ == "__main__":
    asyncio.run(main())
