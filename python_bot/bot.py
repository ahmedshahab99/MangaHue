"""
MangaHue — Local Manga Colorization Bot
Uses local ONNX model with smart semantic hints for beautiful, natural manga colors.
"""
import asyncio, json, os, re, shutil, sys, time, traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
import aiohttp
from bs4 import BeautifulSoup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters,
)
from telegram.constants import ParseMode

# -- config -----------------------------------------------------------------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
LIMIT = int(os.environ.get("DAILY_PAGE_LIMIT", "100"))
BSIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "5"))
WDIR = Path(os.environ.get("MANGAHUE_WORK_DIR", ".mangahue/telegram"))
MPATH = Path(os.environ.get("MANGAHUE_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "manga-onnx" / "manga-colorize-fp16.onnx")))
MAX_CONC = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
DELAY = float(os.environ.get("MIN_SEND_INTERVAL", "3.1"))
ADMIN_ID = os.environ.get("ADMIN_CHAT_ID", "")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
JUNK_KW = {"avatar", "logo", "icon", "spinner", "loading", "thumbnail", "banner",
           "cover", "preview", "profile", "rating", "badge", "button", "pixel",
           "tracker", "analytics", "favicon", "default", "placeholder", "blur"}
MAX_IMG = 50 * 1024 * 1024

# -- BotState ---------------------------------------------------------------
class BotState:
    def __init__(self):
        self.usage: dict[int, int] = {}
        self.settings: dict[int, dict] = {}
        self.cancels: dict[int, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        WDIR.mkdir(parents=True, exist_ok=True)
        self._fp = WDIR / "usage" / "daily.json"
        self._fp.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            d = json.loads(self._fp.read_text(encoding="utf-8"))
            if d.get("date") == today:
                self.usage = {int(k): v for k, v in d.get("pages", {}).items()}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    async def _save(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with self._lock:
            self._fp.parent.mkdir(parents=True, exist_ok=True)
            self._fp.write_text(json.dumps({"date": today, "pages": self.usage}), encoding="utf-8")

    def remaining(self, cid: int) -> int:
        return max(0, LIMIT - self.usage.get(cid, 0))

    def used(self, cid: int) -> int:
        return self.usage.get(cid, 0)

    async def record(self, cid: int, n: int = 1):
        self.usage[cid] = self.usage.get(cid, 0) + n
        await self._save()

    def get_settings(self, cid: int) -> dict:
        if cid not in self.settings:
            self.settings[cid] = {"batch": BSIZE}
        return self.settings[cid]

    def try_start(self, cid: int) -> Optional[asyncio.Event]:
        if cid in self.cancels and not self.cancels[cid].is_set():
            return None
        if sum(1 for e in self.cancels.values() if not e.is_set()) >= MAX_CONC:
            return None
        e = asyncio.Event()
        self.cancels[cid] = e
        return e

    def finish(self, cid: int):
        self.cancels.pop(cid, None)

    def is_active(self, cid: int) -> bool:
        return cid in self.cancels and not self.cancels[cid].is_set()

    async def cancel(self, cid: int) -> bool:
        e = self.cancels.get(cid)
        if e is None or e.is_set():
            return False
        e.set()
        return True

# -- optional cloudflare bypass --------------------------------------------
try:
    import cloudscraper
    _scraper = cloudscraper.create_scraper()
    HAS_CLOUDSCRAPER = True
except Exception:
    _scraper = None
    HAS_CLOUDSCRAPER = False

# -- MangaDownloader --------------------------------------------------------
class MangaDownloader:
    def __init__(self):
        self._s: Optional[aiohttp.ClientSession] = None
        self._last = 0.0

    async def _session(self):
        if not self._s:
            self._s = aiohttp.ClientSession(headers={"User-Agent": UA}, timeout=aiohttp.ClientTimeout(total=30))
        return self._s

    async def _wait(self):
        now = time.monotonic()
        if (gap := 0.5 - (now - self._last)) > 0:
            await asyncio.sleep(gap)
        self._last = time.monotonic()

    @staticmethod
    def _site(uri: str) -> str:
        h = (urlparse(uri).hostname or "").lower()
        if "nhentai.net" in h and "/g/" in uri:
            return "nhentai"
        if "mangadex.org" in h and "/chapter/" in uri:
            return "mangadex"
        if any(urlparse(uri).path.lower().endswith(e) for e in IMG_EXTS):
            return "direct"
        return "generic"

    async def extract_urls(self, uri: str, max_p: int = 100) -> list[str]:
        site = self._site(uri)
        sess = await self._session()
        await self._wait()
        if site == "direct":
            return [uri]
        if site == "nhentai":
            return await self._nhentai(uri, sess, max_p)
        if site == "mangadex":
            return await self._mangadex(uri, sess, max_p)
        return await self._generic(uri, sess, max_p)

    async def _nhentai(self, uri: str, sess, max_p: int) -> list[str]:
        m = re.search(r"/g/(\d+)", uri)
        if not m:
            return []
        gid = m.group(1)
        pages: dict[int, str] = {}
        
        # Use nhentai.to mirror (no Cloudflare)
        mirror = uri.replace("nhentai.net", "nhentai.to")
        try:
            async with sess.get(mirror, headers={"User-Agent": UA}) as r:
                html = await r.text() if r.status == 200 else ""
        except Exception:
            html = ""

        if html:
            soup = BeautifulSoup(html, "html.parser")
            # Extract all image src from img tags (not just noscript)
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if not src or src.startswith("data:") or "base64" in src or src.endswith(".svg"):
                    continue
                # Convert thumbnail URL to full image: 1t.jpg → 1.jpg
                src = re.sub(r"/(\d+)t\.(jpg|png|webp)", r"/\1.\2", src)
                if pm := re.search(r"/(\d+)\.(jpg|png|webp)", src):
                    pages.setdefault(int(pm.group(1)), src)

            # Extract from noscript as fallback
            if not pages:
                for ns in soup.find_all("noscript"):
                    inner = BeautifulSoup(ns.decode_contents(), "html.parser")
                    for img in inner.find_all("img"):
                        s = img.get("src") or ""
                        s = re.sub(r"/t(\d*)\.", r"/i\1.", s)
                        s = re.sub(r"/(\d+)t\.(jpg|png|webp)", r"/\1.\2", s)
                        if pm := re.search(r"/(\d+)\.(jpg|png|webp)", s):
                            pages.setdefault(int(pm.group(1)), s)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for ns in soup.find_all("noscript"):
                inner = BeautifulSoup(ns.decode_contents(), "html.parser")
                for img in inner.find_all("img"):
                    if s := img.get("src"):
                        s = re.sub(r"://t(\d*)\.", r"://i\1.", s)
                        s = re.sub(r"/(\d+)t\.(jpg|png|webp|gif)", r"/\1.\2", s)
                        if pm := re.search(r"/(\d+)(?:t)?\.(?:jpg|png|webp|gif)", s):
                            pages.setdefault(int(pm.group(1)), s)
        if not pages:
            try:
                api_url = f"https://nhentai.net/api/gallery/{gid}"
                if HAS_CLOUDSCRAPER:
                    r = _scraper.get(api_url, timeout=20)
                    d = r.json()
                else:
                    async with sess.get(api_url, headers={"User-Agent": UA, "Accept": "application/json"}) as r:
                        if r.status != 200:
                            return []
                        d = await r.json()
                mid, n = d.get("media_id"), int(d.get("num_pages", 0))
                em = {1: "png", 2: "jpg", 3: "png", 4: "webp", 5: "gif"}
                ext = "jpg"
                for i, pg in enumerate(d.get("images", {}).get("pages", []), 1):
                    t = pg.get("t", 2)
                    ext = em.get(t, "jpg") if isinstance(t, int) else (t if isinstance(t, str) else "jpg")
                    pages.setdefault(i, f"https://i3.nhentai.net/galleries/{mid}/{i}.{ext}")
                if not pages and mid and n:
                    for p in range(1, min(n + 1, max_p + 1)):
                        pages.setdefault(p, f"https://i3.nhentai.net/galleries/{mid}/{p}.{ext}")
            except Exception:
                return []
        return [pages[p] for p in sorted(pages)]

    async def _mangadex(self, uri: str, sess, max_p: int) -> list[str]:
        m = re.search(r"/chapter/([a-f0-9-]+)", uri)
        if not m:
            return []
        cid = m.group(1)
        async with sess.get(f"https://api.mangadex.org/at-home/server/{cid}") as r:
            if r.status != 200:
                raise RuntimeError(f"MangaDex API {r.status}")
            d = await r.json()
        base, ch = d["baseUrl"], d["chapter"]
        hsh, q = ch["hash"], ("data" if "data" in ch else "data-saver")
        return [f"{base}/{q}/{hsh}/{f}" for f in ch.get(q, [])[:max_p]]

    async def _generic(self, uri: str, sess, max_p: int) -> list[str]:
        async with sess.get(uri) as r:
            html = await r.text()
        soup = BeautifulSoup(html, "html.parser")
        cands: set[str] = set()
        for img in soup.find_all("img"):
            for a in ("src", "data-src", "data-original"):
                s = img.get(a)
                if s and isinstance(s, str) and s.lower().endswith(tuple(IMG_EXTS)):
                    if not any(k in s.lower() for k in JUNK_KW):
                        cands.add(s)
        return sorted(cands)[:max_p]

    async def download_batch(self, urls: list[str], out_dir: str, start: int = 1) -> list[str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sess = await self._session()
        saved = []
        for i, img_uri in enumerate(urls, start=start):
            await self._wait()
            ext = Path(urlparse(img_uri).path).suffix
            if not ext or ext.lower() not in IMG_EXTS:
                ext = ".jpg"
            fp = out / f"{i:04d}{ext}"
            try:
                async with sess.get(img_uri, timeout=aiohttp.ClientTimeout(total=90),
                                    headers={"Referer": img_uri}) as r:
                    if r.status != 200:
                        continue
                    data = await r.read()
                    if len(data) > MAX_IMG:
                        continue
                fp.write_bytes(data)
                saved.append(str(fp))
            except Exception:
                continue
        return saved

    async def close(self):
        if self._s:
            await self._s.close()
            self._s = None

    async def __aenter__(self):
        await self._session()
        return self

    async def __aexit__(self, *a):
        await self.close()

# -- SmartHintGenerator -----------------------------------------------------
class SmartHintGenerator:
    SKIN = [(252,215,185), (245,200,170), (238,185,155), (255,222,195)]
    HAIR = [(38,28,22), (55,40,32), (45,32,26), (160,125,90), (80,58,42), (175,135,100)]
    CLOTHING = [(155,48,42), (42,72,148), (52,42,75), (38,75,55), (185,168,148),
                (72,50,44), (90,118,160), (148,128,118), (62,85,55), (165,145,130)]
    BG = [(248,246,240), (240,236,228), (235,230,220)]
    EYE = [(28,38,105), (68,142,178), (42,88,160)]
    
    @staticmethod
    def classify_pixel(gray_val: float, local_mean: float, local_var: float, y_norm: float, near_edge: bool) -> str:
        """Classify what a pixel region likely represents."""
        brightness = gray_val / 255.0
        var_norm = min(local_var / 800.0, 1.0)
        
        # Very bright, uniform → background
        if brightness > 0.85 and var_norm < 0.15:
            return "bg"
        # Bright but textured → screentone/sky — skip hints
        if brightness > 0.80 and var_norm > 0.2:
            return "skip"
        # Dark, textured, upper portion → hair
        if brightness < 0.3 and var_norm > 0.2 and y_norm < 0.4:
            return "hair"
        # Very dark → hair or solid ink
        if brightness < 0.2:
            return "hair" if y_norm < 0.5 else "clothing"
        # Medium brightness, smooth, center area → skin
        if 0.35 < brightness < 0.7 and var_norm < 0.25 and not near_edge:
            if y_norm < 0.6:
                return "skin"
        # Medium-dark, detailed → clothing
        if brightness < 0.5 and var_norm > 0.15:
            return "clothing"
        # Default: clothing for non-background areas
        if brightness < 0.75:
            return "clothing"
        return "skip"
    
    @staticmethod
    def generate(gray_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        h, w = gray_img.shape
        hm = np.zeros((h, w, 3), dtype=np.uint8)
        hk = np.zeros((h, w), dtype=np.uint8)
        
        gf = gray_img.astype(np.float32)
        # Local mean and variance for region classification
        local_mean = cv2.blur(gf, (31, 31))
        local_sq = cv2.blur(gf * gf, (31, 31))
        local_var = np.maximum(local_sq - local_mean * local_mean, 0)
        
        # --- Feature 6: Auto-brightness detection ---
        global_brightness = gray_img.mean()
        brightness_adj = 1.0
        if global_brightness < 100:
            brightness_adj = 1.10  # Dark page → brighter hints
        elif global_brightness > 180:
            brightness_adj = 0.90  # Light page → richer, more saturated hints
        
        palettes = {
            "skin": [tuple(np.clip(np.array(c, dtype=np.float32) * brightness_adj, 0, 255).astype(int)) for c in SmartHintGenerator.SKIN],
            "hair": [tuple(np.clip(np.array(c, dtype=np.float32) * brightness_adj, 0, 255).astype(int)) for c in SmartHintGenerator.HAIR],
            "clothing": [tuple(np.clip(np.array(c, dtype=np.float32) * brightness_adj, 0, 255).astype(int)) for c in SmartHintGenerator.CLOTHING],
            "bg": [tuple(np.clip(np.array(c, dtype=np.float32) * brightness_adj, 0, 255).astype(int)) for c in SmartHintGenerator.BG],
        }
        counters = {"skin": 0, "hair": 0, "clothing": 0, "bg": 0}
        
        # --- Feature 7: Background/margin detection via 5% edge sampling ---
        left_sample_cols = max(1, int(w * 0.05))
        right_sample_cols = max(1, int(w * 0.05))
        top_sample_rows = max(1, int(h * 0.05))
        bottom_sample_rows = max(1, int(h * 0.05))
        left_margin = bool(np.mean(gray_img[:, :left_sample_cols] > 230) > 0.9)
        right_margin = bool(np.mean(gray_img[:, -right_sample_cols:] > 230) > 0.9)
        top_margin = bool(np.mean(gray_img[:top_sample_rows, :] > 230) > 0.9)
        bottom_margin = bool(np.mean(gray_img[-bottom_sample_rows:, :] > 230) > 0.9)
        
        # Skip pure white margins (no content)
        content = gray_img < np.maximum(local_mean * 0.88, 225)
        
        # Also mask out detected margin regions
        if left_margin:
            content[:, :int(w * 0.05)] = False
        if right_margin:
            content[:, int(w * 0.95):] = False
        if top_margin:
            content[:int(h * 0.05), :] = False
        if bottom_margin:
            content[int(h * 0.95):, :] = False
        
        # Adaptive grid based on content density
        density = content.mean()
        grid = max(8, min(24, int(8 + density * 28)))
        step_y, step_x = max(1, h // grid), max(1, w // grid)
        
        # --- Feature 2: Color harmony tracking grid ---
        grid_w = (w + step_x - 1) // step_x
        grid_h = (h + step_y - 1) // step_y
        assigned_colors = np.full((grid_h + 1, grid_w + 1, 3), -1, dtype=np.float32)
        
        for gy in range(0, h, step_y):
            for gx in range(0, w, step_x):
                cy = gy + step_y // 2
                cx = gx + step_x // 2
                if cy >= h or cx >= w:
                    continue
                if not content[cy, cx]:
                    continue
                
                # --- Feature 1: Adaptive hint density by position ---
                # Denser hints in center 60% of page, sparser at edges
                x_rel = cx / w
                y_rel = cy / h
                in_center_x = 0.2 < x_rel < 0.8
                in_center_y = 0.2 < y_rel < 0.8
                if not in_center_x or not in_center_y:
                    # Skip every other cell in edge regions
                    if ((gx // step_x) + (gy // step_y)) % 2 != 0:
                        continue
                
                # Classify this region
                y_norm = cy / h
                near_edge = (cx < w * 0.06 or cx > w * 0.94 or cy < h * 0.04 or cy > h * 0.96)
                label = SmartHintGenerator.classify_pixel(
                    float(gray_img[cy, cx]),
                    float(local_mean[cy, cx]),
                    float(local_var[cy, cx]),
                    y_norm, near_edge
                )
                if label == "skip":
                    continue
                
                # --- Feature 2: Color harmony — maximize RGB distance from neighbors ---
                pal = palettes[label]
                grid_x = gx // step_x
                grid_y = gy // step_y
                
                best_color = pal[0]
                best_dist = -1.0
                for col in pal:
                    dist = 0.0
                    if grid_x > 0 and assigned_colors[grid_y, grid_x - 1, 0] >= 0:
                        dist += np.sum((np.array(col, dtype=np.float32) - assigned_colors[grid_y, grid_x - 1]) ** 2)
                    if grid_y > 0 and assigned_colors[grid_y - 1, grid_x, 0] >= 0:
                        dist += np.sum((np.array(col, dtype=np.float32) - assigned_colors[grid_y - 1, grid_x]) ** 2)
                    if dist > best_dist:
                        best_dist = dist
                        best_color = col
                
                col = best_color
                assigned_colors[grid_y, grid_x] = np.array(col, dtype=np.float32)
                counters[label] += 1
                r = max(2, min(step_y, step_x) // 10)
                
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                if y1 <= y0 or x1 <= x0:
                    continue
                
                py, px = np.ogrid[y0:y1, x0:x1]
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                m = d2 < r * r
                if not np.any(m):
                    continue
                
                alpha = (1.0 - d2[m] / (r * r)) * 0.32
                am = np.clip(alpha * 255, 0, 255).astype(np.uint8)
                
                hk_v = hk[y0:y1, x0:x1]
                np.maximum(hk_v[m], am, out=hk_v[m])
                for ch in range(3):
                    sl = hm[y0:y1, x0:x1, ch].astype(np.float32)
                    sl[m] = sl[m] * (1.0 - alpha) + col[ch] * alpha
                    hm[y0:y1, x0:x1, ch] = np.clip(sl, 0, 255).astype(np.uint8)
        
        return hm, hk

# -- Colorizer (ONNX + smart hints) -----------------------------------------
class Colorizer:
    _CC: dict[str, ort.InferenceSession] = {}

    def __init__(self, mp: str = ""):
        self._mp = Path(mp) if mp else MPATH

    def _sess(self):
        k = str(self._mp)
        if k not in Colorizer._CC:
            if not self._mp.is_file():
                raise RuntimeError(f"Model not found: {self._mp}")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            Colorizer._CC[k] = ort.InferenceSession(k, opts, providers=["CPUExecutionProvider"])
        return Colorizer._CC[k]

    async def colorize(self, in_path: str, out_path: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._colorize_sync, in_path, out_path)

    def _colorize_sync(self, in_path: str, out_path: str) -> bool:
        from PIL import Image as PILImage, ImageFilter
        
        img = PILImage.open(in_path).convert("L")
        orig_w, orig_h = img.size
        orig_arr = np.array(img, dtype=np.float32) / 255.0

        gen = SmartHintGenerator()
        hm, hk = gen.generate(np.array(img))
        hm_pil = PILImage.fromarray(hm)
        hk_pil = PILImage.fromarray(hk)
        hm_pil = hm_pil.filter(ImageFilter.GaussianBlur(radius=8))
        hk_pil = hk_pil.filter(ImageFilter.GaussianBlur(radius=8))
        hm = np.array(hm_pil)
        hk = np.array(hk_pil)

        # 384px = model's native input size (2x faster than 512)
        TILE = 384
        OVERLAP = 48
        STEP = TILE - OVERLAP
        result = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        weight = np.zeros((orig_h, orig_w), dtype=np.float32)

        for y in range(0, orig_h, STEP):
                for x in range(0, orig_w, STEP):
                    x2 = min(x + TILE, orig_w)
                    y2 = min(y + TILE, orig_h)
                    x1 = max(0, x2 - TILE)
                    y1 = max(0, y2 - TILE)

                    tile_arr = orig_arr[y1:y2, x1:x2]
                    tile_hm = hm[y1:y2, x1:x2]
                    tile_hk = hk[y1:y2, x1:x2]
                    tile_rgb = self._infer_tile_from_hints(tile_arr, tile_hm, tile_hk)

                    th, tw = tile_rgb.shape[:2]
                    wy = np.hanning(th)[:, None].astype(np.float32)
                    wx = np.hanning(tw)[None, :].astype(np.float32)
                    blend = wy * wx

                    result[y1:y2, x1:x2] += tile_rgb * blend[:, :, None]
                    weight[y1:y2, x1:x2] += blend

        mask = weight > 1e-6
        result[mask] /= weight[mask, None]
        result = (result * 255).clip(0, 255).astype(np.uint8)

        res_img = PILImage.fromarray(result)
        res_img.save(out_path)
        return True

    def _infer_tile_from_hints(self, arr: np.ndarray, hm: np.ndarray, hk: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        hf = hm.astype(np.float32) / 255.0
        hkf = hk.astype(np.float32) / 255.0
        ph, pw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        t = np.zeros((1, 5, ph, pw), dtype=np.float32)
        t[0, 0, :h, :w] = arr
        t[0, 1, :h, :w] = hf[:, :, 0]
        t[0, 2, :h, :w] = hf[:, :, 1]
        t[0, 3, :h, :w] = hf[:, :, 2]
        t[0, 4, :h, :w] = hkf
        sess = self._sess()
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        out = sess.run([oname], {iname: t})[0]
        return out[0].transpose(1, 2, 0)[:h, :w]

# -- Feature 4: Screentone detection via FFT --------------------------------
def detect_screentones(gray: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Detect screentone regions using 2D FFT on overlapping 32x32 blocks.
    
    Screentones (regular dot patterns) produce strong frequency peaks.
    Returns a binary mask where 1 = screentone detected.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    step = block_size // 2  # 50% overlap
    
    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            block = gray[y:y + block_size, x:x + block_size].astype(np.float32)
            fft = np.fft.fft2(block)
            fft_mag = np.fft.fftshift(np.abs(fft))
            
            # Zero out DC component (center 5x5 region)
            dc_c = block_size // 2
            fft_mag[dc_c - 2:dc_c + 3, dc_c - 2:dc_c + 3] = 0
            
            max_val = np.max(fft_mag)
            nonzero = fft_mag[fft_mag > 0]
            mean_val = np.mean(nonzero) if len(nonzero) > 0 else 0.0
            
            # Strong isolated frequency peak → periodic pattern (screentone)
            if max_val > 0 and mean_val > 0 and max_val / (mean_val + 1e-6) > 8.0:
                mask[y:y + block_size, x:x + block_size] = 1
    
    return mask

# -- Post-processing pipeline -----------------------------------------------
def apply_manga_pipeline(colored_path: str, original_path: str, output_path: str):
    import cv2
    orig = Image.open(original_path).convert("L")
    im = np.array(Image.open(colored_path).convert("RGB"), dtype=np.float64)
    gray = np.array(orig, dtype=np.float64)

    # 1. Highlight-based white balance (top 5% brightest → white)
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    mask = lum >= np.percentile(lum, 95)
    if mask.sum() > 0:
        hm = im[mask].mean(axis=0)
        gain = np.clip(255.0 / np.maximum(hm, 1e-6), 0.5, 2.0)
        im = np.clip(im * gain[None, None, :], 0, 255)

    # 2. Manga toning: subtle warm shift in midtones
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    m = np.clip(1.0 - np.abs(lum / 127.5 - 1.0), 0, 1)
    im[:, :, 0] = np.clip(im[:, :, 0] + m * 5.0, 0, 255)
    im[:, :, 2] = np.clip(im[:, :, 2] - m * 3.5, 0, 255)

    # 3. Local-variance mask (high var = screentone, low var = flat color)
    gf = gray.astype(np.float32)
    g2 = cv2.blur(gf, (5, 5))
    var = np.maximum(cv2.blur(gf * gf, (5, 5)) - g2 * g2, 0)
    vmask = np.clip(var / max(var.max(), 1.0), 0, 1)

    # 4. Feature 4: Screentone-aware desaturation — FFT-detected screentones preserved
    st_mask = detect_screentones(np.array(orig)).astype(np.float64)
    # Merge FFT screentone mask with local-variance mask
    combined_vmask = np.maximum(vmask, st_mask * 0.6)
    desat = 0.78 + 0.22 * combined_vmask
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    im = np.clip(lum[:, :, None] + desat[:, :, None] * (im - lum[:, :, None]), 0, 255)

    # 5. Feature 3: Ink-aware line extraction — nonlinear strength mapping
    ink_dark = np.clip(1.0 - gray / 60.0, 0, 1)
    ink_strength = 0.05 + np.power(ink_dark, 1.5) * 0.10
    ink = cv2.dilate((ink_strength * 255).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float64) / 255.0

    # 6. Edge-aware smooth
    try:
        smooth = cv2.edgePreservingFilter(im.astype(np.uint8), flags=1, sigma_s=30, sigma_r=25)
        smooth_f = smooth.astype(np.float64)
    except Exception:
        smooth_f = np.array(Image.fromarray(im.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), dtype=np.float64)
    ew = np.clip(1.0 - ink[:, :, None] * 0.6, 0.4, 1.0)
    im = smooth_f * (1.0 - ew) + im * ew

    # 7. Feature 3: Ink-aware line restoration (proportional, nonlinear)
    im = np.clip(im * (1.0 - ink[:, :, None]), 0, 255)

    # 8. Color boost in flat areas (screentone areas get less boost)
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    boost = 1.0 + 0.15 * (1.0 - combined_vmask)
    chroma = im - lum[:, :, None]
    im = np.clip(lum[:, :, None] + boost[:, :, None] * chroma, 0, 255)

    # 9. Gentle unsharp mask
    blurred = np.array(Image.fromarray(im.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float64)
    im = np.clip(im + 0.5 * (im - blurred), 0, 255)

    Image.fromarray(im.astype(np.uint8)).save(output_path)

# -- Helpers ----------------------------------------------------------------
def _bar(done: int, total: int, w: int = 14) -> str:
    if total <= 0:
        return "░" * w
    f = min(round(done / total * w), w)
    return "▓" * f + "░" * (w - f)

def _eta(times: list[float], done: int, total: int) -> Optional[str]:
    if not times or done >= total or done <= 0:
        return None
    s = sum(times) / len(times) * (total - done)
    if s < 60:
        return f"~{s:.0f}s remaining"
    elif s < 3600:
        return f"~{s / 60:.0f}m remaining"
    else:
        return f"~{s / 3600:.1f}h remaining"

def _extract_url(text: str) -> Optional[str]:
    for t in re.split(r"\s+", text):
        t = t.strip("\"'()[]<>,;")
        if t.startswith(("http://", "https://")):
            return t.rstrip(".,)!]?>")
    return None

# -- Command handlers -------------------------------------------------------
async def _start(u, _):
    await u.message.reply_text("*MangaHue* — send a manga chapter URL.\n/help for commands.", parse_mode=ParseMode.MARKDOWN)

async def _help(u, _):
    await u.message.reply_text(f"*Commands:*\n/start /help /status /cancel\n/batch N /stats /clean\n\nLimit: *{LIMIT}* pages/day.", parse_mode=ParseMode.MARKDOWN)

async def _status(u, c):
    s: BotState = c.bot_data["state"]
    await u.message.reply_text("Running — /cancel to stop." if s.is_active(u.effective_chat.id) else "Idle.")

async def _cancel(u, c):
    s: BotState = c.bot_data["state"]
    ok = await s.cancel(u.effective_chat.id)
    await u.message.reply_text("Cancelling..." if ok else "Nothing to cancel.")

async def _batch(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    arg = " ".join(c.args) if c.args else ""
    stg = s.get_settings(cid)
    if arg.isdigit() and 1 <= int(arg) <= 10:
        stg["batch"] = int(arg)
        await u.message.reply_text(f"Batch: *{arg}*.", parse_mode=ParseMode.MARKDOWN)
    else:
        await u.message.reply_text(f"Batch: *{stg['batch']}*. `/batch 3` (1–10).", parse_mode=ParseMode.MARKDOWN)

async def _stats(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    await u.message.reply_text(f"`{_bar(s.used(cid), LIMIT, 16)}` *{s.used(cid)}* / {LIMIT}", parse_mode=ParseMode.MARKDOWN)

async def _clean(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    if s.is_active(cid):
        await u.message.reply_text("/cancel first.")
        return
    d = WDIR / str(cid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        s.settings.pop(cid, None)
    await u.message.reply_text("Cleaned up.")

# -- Callback handler -------------------------------------------------------
async def _callback(u, c):
    q = u.callback_query
    await q.answer()
    s: BotState = c.bot_data["state"]
    d = q.data or ""
    if d == "cancel_job":
        ok = await s.cancel(u.effective_chat.id)
        await q.edit_message_text("Cancelled." if ok else "No active job.")
    elif d == "colorize_another":
        await q.message.reply_text("Send another URL.")

# -- On-message handler -----------------------------------------------------
async def _on_message(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    url = _extract_url(u.message.text.strip())
    if not url:
        await u.message.reply_text("Send a manga chapter URL. /help")
        return
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        await u.message.reply_text("Invalid URL.")
        return
    if h in ("localhost", "127.0.0.1") or h.startswith(("192.168.", "10.")):
        await u.message.reply_text("Private URLs not allowed.")
        return
    if s.remaining(cid) <= 0:
        await u.message.reply_text(f"Daily limit ({LIMIT}) reached. /stats")
        return
    evt = s.try_start(cid)
    if evt is None:
        await u.message.reply_text("Job running or pool full.")
        return
    asyncio.create_task(_process(u, c, url, cid, evt))

# -- Core pipeline ----------------------------------------------------------
async def _process(u, c, url: str, cid: int, cancel: asyncio.Event):
    s: BotState = c.bot_data["state"]
    stg = s.get_settings(cid)
    bsize = stg["batch"]
    jid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job = WDIR / str(cid) / jid
    dl_dir = job / "downloads"; out_dir = job / "colored"
    dl_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel_job")]])
    msg = await u.message.reply_text("Analyzing...")
    cr = Colorizer()

    try:
        async with MangaDownloader() as dl:
            urls = await dl.extract_urls(url)
            if not urls:
                await msg.edit_text("No page images found.")
                return

            total = min(len(urls), s.remaining(cid) or len(urls))
            if total <= 0:
                await msg.edit_text("Daily limit reached.")
                return

            batches = max(1, (total + bsize - 1) // bsize)
            await msg.edit_text(f"Found *{total}* pages · *{batches}* batch(es) of {bsize}",
                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb)

            sent, failed, times = 0, 0, []

            for bi in range(batches):
                if cancel.is_set():
                    await msg.edit_text(f"Canceled. Sent {sent}.")
                    break

                b0, b1 = bi * bsize, min(bi * bsize + bsize, total)
                saved = await dl.download_batch(urls[b0:b1], str(dl_dir), start=b0 + 1)
                idx_map: dict[int, Path] = {}
                for p in saved:
                    try:
                        idx_map[int(Path(p).stem) - 1] = Path(p)
                    except ValueError:
                        continue

                for pi in range(b0, b1):
                    if cancel.is_set():
                        break
                    if (lp := idx_map.get(pi)) is None:
                        failed += 1
                        continue

                    t0 = time.monotonic()
                    op = out_dir / f"page_{pi:03d}.png"
                    ok = False
                    try:
                        # Show progress BEFORE starting (tiling takes time)
                        try:
                            await msg.edit_text(
                                f"`{_bar(sent + failed, total)}` {(sent + failed) / total * 100:.0f}%\n"
                                f"Coloring p.{pi + 1}/{total}...",
                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                        except Exception:
                            pass
                        ok = await cr.colorize(str(lp), str(op))
                        if ok:
                            apply_manga_pipeline(str(op), str(lp), str(op))
                    except Exception as exc:
                        sys.stderr.write(f"[colorize] p.{pi}: {exc}\n")

                    if cancel.is_set():
                        await msg.edit_text(f"Canceled. Sent {sent}.")
                        break

                    if ok:
                        try:
                            await c.bot.send_chat_action(cid, action="upload_document")
                            with open(op, "rb") as f:
                                await c.bot.send_document(cid, document=f, filename=op.name,
                                                          caption=f"MangaHue · p.{pi + 1}",
                                                          read_timeout=120, write_timeout=120)
                            sent += 1
                            await s.record(cid)
                        except Exception as ex:
                            sys.stderr.write(f"[send] {ex}\n")
                            failed += 1
                    else:
                        failed += 1

                    times.append(time.monotonic() - t0)
                    lp.unlink(missing_ok=True); op.unlink(missing_ok=True)

                    done = sent + failed
                    eta_t = _eta(times, done, total)
                    txt = f"`{_bar(done, total)}` {(done / total * 100) if total else 0:.0f}%\n{done}/{total} · b{bi + 1}/{batches}"
                    if failed:
                        txt += f" · fail {failed}"
                    if eta_t:
                        txt += f" · {eta_t}"
                    try:
                        await msg.edit_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                    except Exception:
                        pass
                    await asyncio.sleep(DELAY)

        final = f"Done. Sent *{sent}* page(s)."
        if failed:
            final += f" *{failed}* failed. Resend to retry."
        kb_final = InlineKeyboardMarkup([[InlineKeyboardButton("Colorize Another", callback_data="colorize_another")]])
        try:
            await msg.edit_text(final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)
        except Exception:
            await c.bot.send_message(cid, final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)

    except Exception as exc:
        traceback.print_exc()
        try:
            await msg.edit_text(f"Error: {exc}")
        except Exception:
            await c.bot.send_message(cid, f"MangaHue error: {exc}")
        if ADMIN_ID:
            try:
                await c.bot.send_message(int(ADMIN_ID), f"[Error] chat={cid}: {exc}")
            except Exception:
                pass
    finally:
        shutil.rmtree(job, ignore_errors=True)
        s.finish(cid)

# -- main -------------------------------------------------------------------
def main():
    if not TOKEN:
        print("Set TELEGRAM_BOT_TOKEN.", file=sys.stderr)
        sys.exit(1)
    state = BotState()
    app = Application.builder().token(TOKEN).build()
    app.bot_data["state"] = state
    for cmd, fn in [("start", _start), ("help", _help), ("status", _status),
                     ("cancel", _cancel), ("batch", _batch), ("stats", _stats),
                     ("clean", _clean)]:
        app.add_handler(CommandHandler(cmd, fn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))
    app.add_handler(CallbackQueryHandler(_callback))
    print("MangaHue bot starting (long-polling)...")
    app.run_polling(allowed_updates=["message", "callback_query"])

if __name__ == "__main__":
    main()
