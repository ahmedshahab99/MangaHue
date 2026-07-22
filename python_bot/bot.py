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
    COLORS = [
        (248,210,182), (240,195,168), (235,185,158), (250,218,192), (228,178,150), (242,205,175),
        (38,28,22), (52,40,32), (45,35,28), (155,120,88), (78,58,42), (170,132,98), (58,45,38), (120,90,65),
        (155,48,42), (42,72,148), (52,42,75), (38,75,55), (185,168,148), (72,50,44),
        (90,118,160), (148,128,118), (62,85,55), (140,110,95), (55,60,70), (165,145,130),
        (28,38,105), (68,142,178), (42,88,160),
        (248,246,240), (240,236,228), (235,230,220), (250,248,244),
        (200,60,50), (45,115,175), (165,135,55), (145,105,125),
    ]

    @staticmethod
    def generate(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        h, w = gray.shape
        hm = np.zeros((h, w, 3), dtype=np.uint8)
        hk = np.zeros((h, w), dtype=np.uint8)

        local_mean = cv2.blur(gray.astype(np.float32), (21, 21))
        content_mask = gray < np.maximum(local_mean * 0.9, 220)
        content_pct = content_mask.mean()
        grid = max(10, min(20, int(10 + content_pct * 16)))
        sy, sx = max(1, h // grid), max(1, w // grid)

        cols = SmartHintGenerator.COLORS
        for gy in range(grid):
            for gx in range(grid):
                cy = int(gy * sy + sy * 0.5)
                cx = int(gx * sx + sx * 0.5)
                if cy >= h or cx >= w or not content_mask[cy, cx]:
                    continue
                idx = (gy * 7 + gx * 13) % len(cols)
                col = cols[idx]
                r = max(2, min(sy, sx) // 8)
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                if y1 <= y0 or x1 <= x0:
                    continue
                py, px = np.ogrid[y0:y1, x0:x1]
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                m = d2 < r * r
                if not np.any(m):
                    continue
                alpha = (1.0 - d2[m] / (r * r)) * 0.28
                am = np.clip(alpha * 255, 0, 255).astype(np.uint8)
                hk_view = hk[y0:y1, x0:x1]
                np.maximum(hk_view[m], am, out=hk_view[m])
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

# -- Post-processing pipeline -----------------------------------------------
def apply_manga_pipeline(colored_path: str, original_path: str, output_path: str):
    import cv2
    orig = Image.open(original_path).convert("L")
    im = np.array(Image.open(colored_path).convert("RGB"), dtype=np.float64)
    gray = np.array(orig, dtype=np.float64)

    # 1. Gray-world white balance (more neutral than highlight-based)
    means = im.mean(axis=(0, 1))
    g = means.mean()
    if g > 1e-6:
        gain = np.clip(g / np.maximum(means, 1e-6), 0.7, 1.4)
        im = np.clip(im * gain[None, None, :], 0, 255)

    # 2. Ink mask for protection
    ink = cv2.dilate((gray < 55).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float64)

    # 3. Edge-aware smooth (preserve ink, smooth colors)
    try:
        smooth = cv2.edgePreservingFilter(im.astype(np.uint8), flags=1, sigma_s=25, sigma_r=20)
        smooth_f = smooth.astype(np.float64)
    except Exception:
        smooth_f = np.array(Image.fromarray(im.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), dtype=np.float64)
    blend = np.clip(1.0 - ink[:, :, None] * 0.7, 0.35, 1.0)
    im = smooth_f * (1.0 - blend) + im * blend

    # 4. Gentle desaturation (manga should be slightly muted)
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    im = np.clip(lum[:, :, None] + 0.90 * (im - lum[:, :, None]), 0, 255)

    # 5. Screentone-aware desaturation (high variance areas stay natural)
    gf = gray.astype(np.float32)
    g2 = cv2.blur(gf, (7, 7))
    var = np.maximum(cv2.blur(gf * gf, (7, 7)) - g2 * g2, 0)
    vmask = np.clip(var / max(var.max(), 1.0), 0, 1)
    desat = 0.85 + 0.15 * vmask
    im = np.clip(lum[:, :, None] + desat[:, :, None] * (im - lum[:, :, None]), 0, 255)

    # 6. Adaptive line restoration (darker ink = stronger darkening)
    line_strength = np.clip(ink * 1.2, 0, 1)
    im = np.clip(im * (1.0 - 0.12 * line_strength[:, :, None]), 0, 255)

    # 7. Gentle sharpen
    blurred = np.array(Image.fromarray(im.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float64)
    im = np.clip(im + 0.4 * (im - blurred), 0, 255)

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
    return f"~{s:.0f}s" if s < 60 else f"~{s / 60:.1f}m"

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
