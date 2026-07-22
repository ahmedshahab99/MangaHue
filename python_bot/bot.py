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
        
        # Try cloudscraper first (bypasses Cloudflare)
        if HAS_CLOUDSCRAPER:
            try:
                r = _scraper.get(uri, timeout=30)
                if r.status_code == 200 and len(r.text) > 1000 and "cloudflare" not in r.text.lower()[:500]:
                    html = r.text
                else:
                    html = ""
            except Exception:
                html = ""
        else:
            try:
                async with sess.get(uri, headers={"User-Agent": UA}) as r:
                    html = await r.text() if r.status == 200 else ""
            except Exception:
                html = ""
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
    PAL = {
        "skin":     [(255, 218, 185), (240, 200, 170), (220, 175, 150)],
        "hair":     [(40, 30, 50), (180, 130, 80), (50, 40, 60), (200, 100, 50)],
        "eyes":     [(20, 40, 120), (80, 160, 200), (50, 100, 180)],
        "clothing": [(180, 50, 50), (40, 70, 140), (60, 100, 60), (200, 180, 60)],
        "bg":       [(230, 230, 235), (200, 210, 220), (180, 200, 210)],
    }
    MS = 800
    MH = 12

    @staticmethod
    def _lines(g: np.ndarray) -> np.ndarray:
        f = g.astype(np.float64)
        h, w, s = f.shape[0], f.shape[1], 31
        p = np.pad(f, ((1, 0), (1, 0)), mode="constant")
        i = p.cumsum(axis=0).cumsum(axis=1)
        r = s // 2
        y, x = np.arange(h)[:, None], np.arange(w)[None, :]
        y0 = np.maximum(0, y - r); x0 = np.maximum(0, x - r)
        y1 = np.minimum(h, y + r + 1); x1 = np.minimum(w, x + r + 1)
        area = (y1 - y0) * (x1 - x0)
        sums = i[y1, x1] - i[y0, x1] - i[y1, x0] + i[y0, x0]
        ratio = f / np.maximum(sums / area, 1.0)
        m = 1.0 - np.clip(ratio / 0.82, 0.0, 1.0)
        return (m * m * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _regions(lines: np.ndarray, h: int, w: int) -> list[dict]:
        vis = np.zeros((h, w), dtype=bool)
        reg = []
        d8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for y in range(h):
            for x in range(w):
                if vis[y, x] or lines[y, x] > 70:
                    continue
                q = deque([(x, y)])
                vis[y, x] = True
                px = []
                mnx, mny, mxx, mxy = x, y, x, y
                while q:
                    cx, cy = q.popleft()
                    px.append((cx, cy))
                    mnx = min(mnx, cx); mny = min(mny, cy)
                    mxx = max(mxx, cx); mxy = max(mxy, cy)
                    for dx, dy in d8:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and not vis[ny, nx] and lines[ny, nx] <= 70:
                            vis[ny, nx] = True
                            q.append((nx, ny))
                if len(px) >= SmartHintGenerator.MS:
                    reg.append({"cx": sum(p[0] for p in px) / len(px),
                                "cy": sum(p[1] for p in px) / len(px),
                                "w": mxx - mnx + 1, "h": mxy - mny + 1})
        reg.sort(key=lambda r: r["w"] * r["h"], reverse=True)
        return reg[:SmartHintGenerator.MH]

    @staticmethod
    def _seed(reg, hm, hk, h, w):
        for i, r in enumerate(reg):
            rh, rw = r["cy"] / h, r["cx"] / w
            label = ("hair" if rh < 0.35 else "eyes" if rh < 0.55 and 0.3 < rw < 0.7 else
                     "clothing" if rh < 0.75 else "clothing" if 0.2 < rw < 0.8 else "bg")
            pal = SmartHintGenerator.PAL.get(label, SmartHintGenerator.PAL["bg"])
            col = pal[hash(str(i)) % len(pal)]
            cx, cy = int(r["cx"]), int(r["cy"])
            rad = max(int(max(r["w"], r["h"]) * 0.55), 12)
            y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
            x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
            rr = rad * rad
            for py in range(y0, y1):
                for px in range(x0, x1):
                    d2 = (px - cx) ** 2 + (py - cy) ** 2
                    if d2 >= rr:
                        continue
                    alpha = (1.0 - (d2 / rr) ** 0.5) * 0.55
                    am = int(alpha * 255)
                    hk[py, px] = max(hk[py, px], am)
                    for ch in range(3):
                        hm[py, px, ch] = min(255, max(0, int(hm[py, px, ch] * (1 - alpha) + col[ch] * alpha)))

    def generate(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = gray.shape
        hm = np.zeros((h, w, 3), dtype=np.uint8)
        hk = np.zeros((h, w), dtype=np.uint8)
        lines = self._lines(gray)
        regions = self._regions(lines, h, w)
        if regions:
            self._seed(regions, hm, hk, h, w)
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
            Colorizer._CC[k] = ort.InferenceSession(k, providers=["CPUExecutionProvider"])
        return Colorizer._CC[k]

    async def colorize(self, in_path: str, out_path: str) -> bool:
        img = Image.open(in_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        gen = SmartHintGenerator()
        hm, hk = gen.generate(np.array(img))
        hf = hm.astype(np.float32) / 255.0
        hkf = hk.astype(np.float32) / 255.0

        h, w = arr.shape
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
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: sess.run([oname], {iname: t})[0])
        result = (out[0].transpose(1, 2, 0)[:h, :w] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(result).save(out_path)
        return True

# -- Post-processing pipeline -----------------------------------------------
def apply_manga_pipeline(colored_path: str, original_path: str, output_path: str):
    orig = Image.open(original_path).convert("L")
    im = np.array(Image.open(colored_path).convert("RGB"), dtype=np.float64)
    gray = np.array(orig, dtype=np.float64)
    ink = (255.0 - gray) / 255.0

    # 1. Gray-world white balance
    means = im.mean(axis=(0, 1))
    g = means.mean()
    gain = np.clip(np.where(means > 1e-6, g / means, 1.0), 0.5, 2.0)
    im = np.clip(im * gain[None, None, :], 0, 255)

    # 2. Gentle desaturation
    lum = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    im = np.clip(lum[:, :, None] + 0.88 * (im - lum[:, :, None]), 0, 255)

    # 3. Edge-preserving smooth (median + ink-weighted blend)
    smooth = np.array(Image.fromarray(im.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), dtype=np.float64)
    ew = np.clip(1.0 - ink[:, :, None] * 0.6, 0.4, 1.0)
    im = smooth * (1.0 - ew) + im * ew

    # 4. Adaptive line restoration
    im = np.clip(im * (1.0 - 0.1 * ink[:, :, None]), 0, 255)

    # 5. Gentle unsharp mask
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
