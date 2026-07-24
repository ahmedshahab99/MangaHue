"""
MangaHue — Local Manga Colorization Bot
Uses local ONNX model with smart semantic hints for beautiful, natural manga colors.
"""
import asyncio, concurrent.futures, json, os, random, re, shutil, signal, sys, threading, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters,
)
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut

from config import *

# -- config -----------------------------------------------------------------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
LIMIT = int(os.environ.get("DAILY_PAGE_LIMIT", str(DAILY_PAGE_LIMIT)))
BSIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
WDIR = Path(os.environ.get("MANGAHUE_WORK_DIR", ".mangahue/telegram"))
MPATH = Path(os.environ.get("MANGAHUE_MODEL_PATH", MODEL_PATH))
MAX_CONC = int(os.environ.get("MAX_CONCURRENT_JOBS", str(MAX_CONCURRENT_JOBS)))
DELAY = float(os.environ.get("MIN_SEND_INTERVAL", "3.1"))
ADMIN_ID = int(os.environ.get("ADMIN_CHAT_ID", "0") or "0")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
JUNK_KW = {"avatar", "logo", "icon", "spinner", "loading", "thumbnail", "banner",
           "cover", "preview", "profile", "rating", "badge", "button", "pixel",
           "tracker", "analytics", "favicon", "default", "placeholder", "blur"}
MAX_IMG = MAX_IMAGE_BYTES
MEDIA_GROUP_DELAY = 2.0

# -- BotState ---------------------------------------------------------------
class BotState:
    def __init__(self):
        self.usage: dict[int, int] = {}
        self.settings: dict[int, dict] = {}
        self.cancels: dict[int, asyncio.Event] = {}
        self.blocked: set[int] = set()
        self.custom_limits: dict[int, int] = {}
        self.last_active: dict[int, str] = {}
        self.known_users: set[int] = set()
        self.media_groups: dict[str, list[tuple[str, int]]] = {}
        self._lock = asyncio.Lock()
        self._date = ""
        WDIR.mkdir(parents=True, exist_ok=True)
        self._fp = WDIR / "usage" / "daily.json"
        self._fp.parent.mkdir(parents=True, exist_ok=True)
        self._users_fp = WDIR / "state" / "users.json"
        self._users_fp.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._load_users()

    def _load(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            d = json.loads(self._fp.read_text(encoding="utf-8"))
            self._date = d.get("date", "")
            if self._date == today:
                self.usage = {int(k): v for k, v in d.get("pages", {}).items()}
            else:
                self.usage = {}
                self._date = today
        except (FileNotFoundError, json.JSONDecodeError):
            self._date = today
            self.usage = {}

    def _load_users(self):
        try:
            d = json.loads(self._users_fp.read_text(encoding="utf-8"))
            self.blocked = {int(x) for x in d.get("blocked", [])}
            self.custom_limits = {int(k): v for k, v in d.get("custom_limits", {}).items()}
            self.last_active = {int(k): v for k, v in d.get("last_active", {}).items()}
            self.known_users = {int(x) for x in d.get("known_users", [])}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _check_date_reset(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._date != today:
            self._date = today
            self.usage.clear()

    async def _save(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with self._lock:
            self._fp.parent.mkdir(parents=True, exist_ok=True)
            self._fp.write_text(json.dumps({"date": today, "pages": self.usage}), encoding="utf-8")

    async def _save_users(self):
        async with self._lock:
            self._users_fp.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "blocked": sorted(self.blocked),
                "custom_limits": {str(k): v for k, v in self.custom_limits.items()},
                "last_active": {str(k): v for k, v in self.last_active.items()},
                "known_users": sorted(self.known_users),
            }
            self._users_fp.write_text(json.dumps(data), encoding="utf-8")

    def remaining(self, cid: int) -> int:
        self._check_date_reset()
        limit = self.custom_limits.get(cid, LIMIT)
        return max(0, limit - self.usage.get(cid, 0))

    def used(self, cid: int) -> int:
        self._check_date_reset()
        return self.usage.get(cid, 0)

    def limit_for(self, cid: int) -> int:
        return self.custom_limits.get(cid, LIMIT)

    async def record(self, cid: int, n: int = 1):
        self._check_date_reset()
        self.usage[cid] = self.usage.get(cid, 0) + n
        now = datetime.now(timezone.utc).isoformat()
        self.last_active[cid] = now
        self.known_users.add(cid)
        await self._save()
        await self._save_users()

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

    def active_today_count(self) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        count = 0
        for cid, ts in self.last_active.items():
            if ts.startswith(today):
                count += 1
        for cid in self.usage:
            if cid not in self.last_active:
                count += 1
        return count

from logger import get_logger
from hints import AdvancedHintGenerator
from postprocess import apply_manga_pipeline
from downloader import MangaDownloader

# -- Colorizer (ONNX + smart hints) -----------------------------------------
class Colorizer:
    _CC: dict[str, tuple[ort.InferenceSession, threading.Lock]] = {}

    def __init__(self, mp: str = ""):
        self._mp = Path(mp) if mp else MPATH
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

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
            providers = ["CPUExecutionProvider"]
            for p in ("CUDAExecutionProvider", "DirectMLExecutionProvider"):
                if p in ort.get_available_providers():
                    providers.insert(0, p)
            Colorizer._CC[k] = (ort.InferenceSession(k, opts, providers=providers), threading.Lock())
        return Colorizer._CC[k]

    async def colorize(self, in_path: str, out_path: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._colorize_sync, in_path, out_path)

    def _colorize_sync(self, in_path: str, out_path: str) -> bool:
        from PIL import Image as PILImage, ImageFilter
        
        img = PILImage.open(in_path).convert("L")
        orig_w, orig_h = img.size
        orig_arr = np.array(img, dtype=np.float32) / 255.0

        gen = AdvancedHintGenerator()
        hm, hk = gen.generate(np.array(img))
        hm_pil = PILImage.fromarray(hm)
        hk_pil = PILImage.fromarray(hk)
        hm_pil = hm_pil.filter(ImageFilter.GaussianBlur(radius=12))
        hk_pil = hk_pil.filter(ImageFilter.GaussianBlur(radius=12))
        hm = np.array(hm_pil)
        hk = np.array(hk_pil)

        # 384px = model's native input size (2x faster than 512)
        TILE = TILE_SIZE
        OVERLAP = TILE_OVERLAP
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
        sess, lock = self._sess()
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name

        def _run_onnx():
            with lock:
                return sess.run([oname], {iname: t})[0]

        future = self._executor.submit(_run_onnx)
        try:
            out = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            sys.stderr.write("[onnx] tile inference timed out after 120s — skipping page\n")
            future.cancel()
            raise RuntimeError("ONNX inference timed out")
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

# -- Safe send with flood-wait handling ------------------------------------
async def safe_send(func, *args, _max_retries=3, **kwargs):
    last_exc = None
    for attempt in range(_max_retries):
        try:
            return await func(*args, **kwargs)
        except RetryAfter as e:
            wait = e.retry_after + random.uniform(0, 2)
            sys.stderr.write(
                f"[retry] {getattr(func, '__name__', func)} RetryAfter "
                f"{e.retry_after:.0f}s + jitter, attempt {attempt + 1}/{_max_retries}\n"
            )
            await asyncio.sleep(wait * (2 ** attempt))
            last_exc = e
        except TimedOut as e:
            wait = 2 ** (attempt + 1) + random.uniform(0, 1)
            sys.stderr.write(
                f"[retry] {getattr(func, '__name__', func)} TimedOut, "
                f"attempt {attempt + 1}/{_max_retries}, backoff {wait:.1f}s\n"
            )
            await asyncio.sleep(wait)
            last_exc = e
    raise last_exc

# -- Retry keyboard helpers --------------------------------------------------
def _build_retry_keyboard(failed_fp: Path) -> tuple[str, Optional[InlineKeyboardMarkup]]:
    try:
        info = json.loads(failed_fp.read_text(encoding="utf-8"))
    except Exception:
        return "Failed pages (data expired).", None
    urls = info.get("urls", {})
    retried = info.get("retried", {})
    job_id = failed_fp.stem
    buttons: list[list[InlineKeyboardButton]] = []
    has_pending = False
    for key in sorted(urls.keys(), key=int):
        idx = int(key)
        status = retried.get(key)
        if status is True:
            buttons.append([InlineKeyboardButton(f"p.{idx + 1} \u2705", callback_data="retry_done")])
        elif status is False:
            buttons.append([InlineKeyboardButton(f"Retry page {idx + 1} \u274c", callback_data=f"retry_page_{job_id}_{idx}")])
            has_pending = True
        else:
            buttons.append([InlineKeyboardButton(f"Retry page {idx + 1}", callback_data=f"retry_page_{job_id}_{idx}")])
            has_pending = True
    if has_pending:
        buttons.append([InlineKeyboardButton("Retry All Failed", callback_data=f"retry_all_{job_id}")])
    if not buttons:
        return "All pages processed.", None
    return "Failed pages \u2014 click to retry:", InlineKeyboardMarkup(buttons)

def _mark_page_retried(failed_fp: Path, page_idx: int, success: bool):
    try:
        info = json.loads(failed_fp.read_text(encoding="utf-8"))
        info.setdefault("retried", {})[str(page_idx)] = success
        info.setdefault("retry_count", {})
        info["retry_count"][str(page_idx)] = info["retry_count"].get(str(page_idx), 0) + 1
        failed_fp.write_text(json.dumps(info), encoding="utf-8")
    except Exception:
        pass

def _is_admin(u: Update) -> bool:
    if not ADMIN_ID:
        return False
    return u.effective_chat.id == ADMIN_ID

# -- Command handlers -------------------------------------------------------
async def _start(u, _):
    await safe_send(u.message.reply_text, "*MangaHue* — send a manga chapter URL.\n/help for commands.", parse_mode=ParseMode.MARKDOWN)

async def _help(u, _):
    await safe_send(u.message.reply_text, f"*Commands:*\n/start /help /status /cancel\n/batch N /stats /clean\n\nLimit: *{LIMIT}* pages/day.", parse_mode=ParseMode.MARKDOWN)

async def _status(u, c):
    s: BotState = c.bot_data["state"]
    await safe_send(u.message.reply_text, "Running — /cancel to stop." if s.is_active(u.effective_chat.id) else "Idle.")

async def _cancel(u, c):
    s: BotState = c.bot_data["state"]
    ok = await s.cancel(u.effective_chat.id)
    await safe_send(u.message.reply_text, "Cancelling..." if ok else "Nothing to cancel.")

async def _batch(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    arg = " ".join(c.args) if c.args else ""
    stg = s.get_settings(cid)
    if arg.isdigit() and 1 <= int(arg) <= 10:
        stg["batch"] = int(arg)
        await safe_send(u.message.reply_text, f"Batch: *{arg}*.", parse_mode=ParseMode.MARKDOWN)
    else:
        await safe_send(u.message.reply_text, f"Batch: *{stg['batch']}*. `/batch 3` (1–10).", parse_mode=ParseMode.MARKDOWN)

async def _stats(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    limit = s.limit_for(cid)
    await safe_send(u.message.reply_text, f"`{_bar(s.used(cid), limit, 16)}` *{s.used(cid)}* / {limit}", parse_mode=ParseMode.MARKDOWN)

async def _clean(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    if s.is_active(cid):
        await safe_send(u.message.reply_text, "/cancel first.")
        return
    d = WDIR / str(cid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        s.settings.pop(cid, None)
    await safe_send(u.message.reply_text, "Cleaned up.")

# -- Callback handler -------------------------------------------------------
async def _callback(u, c):
    q = u.callback_query
    await q.answer()
    s: BotState = c.bot_data["state"]
    d = q.data or ""
    cid = u.effective_chat.id
    if d == "cancel_job":
        ok = await s.cancel(cid)
        await safe_send(q.edit_message_text, "Cancelled." if ok else "No active job.")
    elif d == "colorize_another":
        await safe_send(q.message.reply_text, "Send another URL or image.")
    elif d.startswith("retry_page_"):
        asyncio.create_task(_handle_retry_page(u, c, q, d))
    elif d.startswith("retry_all_"):
        asyncio.create_task(_handle_retry_all(u, c, q, d))
    elif d == "retry_done":
        pass

# -- Failed page retry handlers ---------------------------------------------
async def _handle_retry_page(u, c, q, data: str):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    parts = data.split("_")
    if len(parts) < 4:
        await q.answer("Invalid retry data.", show_alert=True)
        return
    page_idx = int(parts[-1])
    job_id = "_".join(parts[2:-1])
    failed_fp = WDIR / str(cid) / "failed" / f"{job_id}.json"
    if not failed_fp.exists():
        await q.answer("Job data expired.", show_alert=True)
        return
    try:
        info = json.loads(failed_fp.read_text(encoding="utf-8"))
    except Exception:
        await q.answer("Failed to load job data.", show_alert=True)
        return
    url = info.get("urls", {}).get(str(page_idx))
    if not url:
        await q.answer("Page URL not found.", show_alert=True)
        return

    await q.answer(f"Retrying page {page_idx + 1}...")
    await safe_send(q.edit_message_text, f"Retrying page {page_idx + 1}...", reply_markup=None)

    retry_jid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job_dir = WDIR / str(cid) / retry_jid
    dl_dir = job_dir / "downloads"; out_dir = job_dir / "colored"
    dl_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    cr = Colorizer()

    try:
        async with MangaDownloader() as dl:
            saved = await dl.download_batch([url], str(dl_dir), start_index=1)
            if not saved:
                await safe_send(q.edit_message_text, f"Download failed for page {page_idx + 1}.")
                _mark_page_retried(failed_fp, page_idx, False)
                text, kb = _build_retry_keyboard(failed_fp)
                await safe_send(q.edit_message_text, text, reply_markup=kb)
                return

            lp = Path(saved[0])
            op = out_dir / f"page_{page_idx + 1:03d}.png"
            ok = await cr.colorize(str(lp), str(op))
            if ok:
                apply_manga_pipeline(str(op), str(lp), str(op))

            if ok:
                await safe_send(c.bot.send_chat_action, cid, action="upload_document")
                with open(op, "rb") as f:
                    await safe_send(c.bot.send_document, cid, document=f, filename=op.name,
                                    caption=f"MangaHue \u00b7 p.{page_idx + 1}",
                                    read_timeout=120, write_timeout=120)
                await s.record(cid)
                _mark_page_retried(failed_fp, page_idx, True)
                await safe_send(q.edit_message_text, f"Page {page_idx + 1} colorized successfully.")
            else:
                _mark_page_retried(failed_fp, page_idx, False)
                await safe_send(q.edit_message_text, f"Colorization failed for page {page_idx + 1}.")

            lp.unlink(missing_ok=True); op.unlink(missing_ok=True)
    except Exception as exc:
        traceback.print_exc()
        _mark_page_retried(failed_fp, page_idx, False)
        try:
            await safe_send(q.edit_message_text, f"Retry error: {exc}")
        except Exception:
            pass
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    text, kb = _build_retry_keyboard(failed_fp)
    if kb:
        await safe_send(c.bot.send_message, cid, text, reply_markup=kb)


async def _handle_retry_all(u, c, q, data: str):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    parts = data.split("_")
    if len(parts) < 3:
        await q.answer("Invalid retry data.", show_alert=True)
        return
    job_id = "_".join(parts[2:])
    failed_fp = WDIR / str(cid) / "failed" / f"{job_id}.json"
    if not failed_fp.exists():
        await q.answer("Job data expired.", show_alert=True)
        return
    try:
        info = json.loads(failed_fp.read_text(encoding="utf-8"))
    except Exception:
        await q.answer("Failed to load job data.", show_alert=True)
        return

    urls: dict[str, str] = info.get("urls", {})
    retried_map = info.get("retried", {})
    pending = [(int(k), v) for k, v in urls.items() if not retried_map.get(k, False)]
    if not pending:
        await q.answer("All pages already retried.", show_alert=True)
        return

    total = len(pending)
    remaining = s.remaining(cid)
    if remaining < total:
        await q.answer(f"Only {remaining} pages left in limit, need {total}.", show_alert=True)
        return

    evt = s.try_start(cid)
    if evt is None:
        await q.answer("Job running or pool full.", show_alert=True)
        return

    await q.answer(f"Retrying {total} failed page(s)...")
    await safe_send(q.edit_message_text, f"Retrying {total} page(s)...", reply_markup=None)

    retry_jid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job_dir = WDIR / str(cid) / retry_jid
    dl_dir = job_dir / "downloads"; out_dir = job_dir / "colored"
    dl_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    cr = Colorizer()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel_job")]])
    status_msg = await safe_send(c.bot.send_message, cid, f"Retrying {total} page(s)...", reply_markup=kb)

    sent, failed = 0, 0
    try:
        async with MangaDownloader() as dl:
            for page_idx, url in sorted(pending):
                if evt.is_set():
                    break
                saved = await dl.download_batch([url], str(dl_dir), start_index=1)
                if not saved:
                    failed += 1
                    _mark_page_retried(failed_fp, page_idx, False)
                    continue
                lp = Path(saved[0])
                op = out_dir / f"page_{page_idx + 1:03d}.png"
                ok = await cr.colorize(str(lp), str(op))
                if ok:
                    apply_manga_pipeline(str(op), str(lp), str(op))
                if ok:
                    await safe_send(c.bot.send_chat_action, cid, action="upload_document")
                    with open(op, "rb") as f:
                        await safe_send(c.bot.send_document, cid, document=f, filename=op.name,
                                        caption=f"MangaHue \u00b7 p.{page_idx + 1}",
                                        read_timeout=120, write_timeout=120)
                    sent += 1
                    await s.record(cid)
                    _mark_page_retried(failed_fp, page_idx, True)
                else:
                    failed += 1
                    _mark_page_retried(failed_fp, page_idx, False)
                lp.unlink(missing_ok=True); op.unlink(missing_ok=True)
                await asyncio.sleep(DELAY)
                try:
                    await safe_send(status_msg.edit_text,
                                    f"Retry: {sent + failed}/{total} \u00b7 sent {sent} \u00b7 failed {failed}",
                                    reply_markup=kb)
                except Exception:
                    pass

        final = f"Retry done. Sent *{sent}*."
        if failed:
            final += f" *{failed}* still failed."
        try:
            await safe_send(status_msg.edit_text, final, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass
    except Exception as exc:
        traceback.print_exc()
        try:
            await safe_send(status_msg.edit_text, f"Retry error: {exc}")
        except Exception:
            pass
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        s.finish(cid)

    text, retry_kb = _build_retry_keyboard(failed_fp)
    if retry_kb:
        await safe_send(c.bot.send_message, cid, text, reply_markup=retry_kb)


# -- On-message handler -----------------------------------------------------
async def _on_message(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id

    if cid in s.blocked:
        return

    url = _extract_url(u.message.text.strip())
    if not url:
        await safe_send(u.message.reply_text, "Send a manga chapter URL. /help")
        return
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        await safe_send(u.message.reply_text, "Invalid URL.")
        return
    if h in ("localhost", "127.0.0.1") or h.startswith(("192.168.", "10.")):
        await safe_send(u.message.reply_text, "Private URLs not allowed.")
        return
    if s.remaining(cid) <= 0:
        await safe_send(u.message.reply_text, f"Daily limit ({s.limit_for(cid)}) reached. /stats")
        return
    evt = s.try_start(cid)
    if evt is None:
        await safe_send(u.message.reply_text, "Job running or pool full.")
        return
    asyncio.create_task(_process(u, c, url, cid, evt))


# -- Image upload handler ----------------------------------------------------
async def _on_image(u, c):
    s: BotState = c.bot_data["state"]
    cid = u.effective_chat.id
    msg = u.message

    if cid in s.blocked:
        return

    if msg.photo:
        file_id = msg.photo[-1].file_id
        is_album = bool(msg.media_group_id)
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
        file_id = msg.document.file_id
        is_album = False
    else:
        return

    if is_album:
        gid = msg.media_group_id
        if gid not in s.media_groups:
            s.media_groups[gid] = []
        s.media_groups[gid].append((file_id, msg.message_id))
        if len(s.media_groups[gid]) == 1:
            asyncio.create_task(_process_album(u, c, gid, cid))
        return

    await _process_uploaded_images(u, c, cid, [file_id])


async def _process_album(u, c, gid: str, cid: int):
    s: BotState = c.bot_data["state"]
    await asyncio.sleep(MEDIA_GROUP_DELAY)
    items = s.media_groups.pop(gid, [])
    if not items:
        return
    file_ids = [fi for fi, _ in items]
    await _process_uploaded_images(u, c, cid, file_ids)


async def _process_uploaded_images(u, c, cid: int, file_ids: list[str]):
    s: BotState = c.bot_data["state"]
    total = len(file_ids)

    remaining = s.remaining(cid)
    if remaining <= 0:
        await safe_send(c.bot.send_message, cid, f"Daily limit ({s.limit_for(cid)}) reached.")
        return

    actual = min(total, remaining)
    if total > remaining:
        await safe_send(c.bot.send_message, cid, f"Limit allows {remaining}/{total}. Processing {actual}.")
        file_ids = file_ids[:actual]

    evt = s.try_start(cid)
    if evt is None:
        await safe_send(c.bot.send_message, cid, "Job running or pool full.")
        return

    jid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job_dir = WDIR / str(cid) / jid
    img_dir = job_dir / "downloads"; out_dir = job_dir / "colored"
    img_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel_job")]])
    msg = await safe_send(c.bot.send_message, cid, f"Processing {len(file_ids)} image(s)...", reply_markup=kb)
    cr = Colorizer()
    sent, failed, times = 0, 0, []
    n = len(file_ids)

    try:
        for i, fid in enumerate(file_ids):
            if evt.is_set():
                await safe_send(msg.edit_text, f"Canceled. Sent {sent}.")
                break

            t0 = time.monotonic()
            img_path = img_dir / f"img_{i:04d}.png"
            op = out_dir / f"page_{i + 1:03d}.png"

            try:
                tg_file = await c.bot.get_file(fid)
                await tg_file.download_to_drive(img_path)
            except Exception as exc:
                sys.stderr.write(f"[dl] image {i}: {exc}\n")
                failed += 1
                continue

            ok = False
            try:
                await safe_send(msg.edit_text, f"Coloring {i + 1}/{n}...", parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                ok = await cr.colorize(str(img_path), str(op))
                if ok:
                    apply_manga_pipeline(str(op), str(img_path), str(op))
            except Exception as exc:
                sys.stderr.write(f"[colorize] img {i}: {exc}\n")

            if ok:
                try:
                    await safe_send(c.bot.send_chat_action, cid, action="upload_document")
                    with open(op, "rb") as f:
                        await safe_send(c.bot.send_document, cid, document=f, filename=op.name,
                                        caption=f"MangaHue \u00b7 img {i + 1}",
                                        read_timeout=120, write_timeout=120)
                    sent += 1
                    await s.record(cid)
                except Exception as ex:
                    sys.stderr.write(f"[send] {ex}\n")
                    failed += 1
            else:
                failed += 1

            times.append(time.monotonic() - t0)
            img_path.unlink(missing_ok=True); op.unlink(missing_ok=True)

            done = sent + failed
            eta_t = _eta(times, done, n)
            txt = f"`{_bar(done, n)}` {done / n * 100:.0f}% \u00b7 {done}/{n}"
            if failed:
                txt += f" \u00b7 fail {failed}"
            if eta_t:
                txt += f" \u00b7 {eta_t}"
            try:
                await safe_send(msg.edit_text, txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
            except Exception:
                pass
            await asyncio.sleep(DELAY)

        final = f"Done. Sent *{sent}* image(s)."
        if failed:
            final += f" *{failed}* failed."
        kb_final = InlineKeyboardMarkup([[InlineKeyboardButton("Colorize Another", callback_data="colorize_another")]])
        try:
            await safe_send(msg.edit_text, final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)
        except Exception:
            await safe_send(c.bot.send_message, cid, final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)

    except Exception as exc:
        traceback.print_exc()
        try:
            await safe_send(msg.edit_text, f"Error: {exc}")
        except Exception:
            await safe_send(c.bot.send_message, cid, f"MangaHue error: {exc}")
        if ADMIN_ID:
            try:
                await safe_send(c.bot.send_message, ADMIN_ID, f"[Error] chat={cid}: {exc}")
            except Exception:
                pass
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        s.finish(cid)

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
    msg = await safe_send(u.message.reply_text, "Analyzing...")
    cr = Colorizer()

    failed_pages: dict[int, str] = {}

    try:
        async with MangaDownloader() as dl:
            urls = await dl.extract_urls(url)
            if not urls:
                await safe_send(msg.edit_text, "No page images found.")
                return

            total = min(len(urls), s.remaining(cid) or len(urls))
            if total <= 0:
                await safe_send(msg.edit_text, "Daily limit reached.")
                return

            batches = max(1, (total + bsize - 1) // bsize)
            await safe_send(msg.edit_text, f"Found *{total}* pages \u00b7 *{batches}* batch(es) of {bsize}",
                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb)

            sent, failed, times = 0, 0, []

            for bi in range(batches):
                if cancel.is_set():
                    for pi in range(bi * bsize, total):
                        failed_pages[pi] = urls[pi]
                    await safe_send(msg.edit_text, f"Canceled. Sent {sent}.")
                    break

                b0, b1 = bi * bsize, min(bi * bsize + bsize, total)
                saved = []
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        saved = await dl.download_batch(urls[b0:b1], str(dl_dir), start_index=b0 + 1)
                        break
                    except Exception:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                idx_map: dict[int, Path] = {}
                for p in saved:
                    try:
                        idx_map[int(Path(p).stem) - 1] = Path(p)
                    except ValueError:
                        continue

                for pi in range(b0, b1):
                    if cancel.is_set():
                        failed_pages[pi] = urls[pi]
                        continue
                    if (lp := idx_map.get(pi)) is None:
                        failed += 1
                        failed_pages[pi] = urls[pi]
                        continue

                    t0 = time.monotonic()
                    op = out_dir / f"page_{pi:03d}.png"
                    ok = False
                    try:
                        # Show progress BEFORE starting (tiling takes time)
                        try:
                            await safe_send(msg.edit_text, 
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
                        if not ok:
                            failed_pages[pi] = urls[pi]
                        await safe_send(msg.edit_text, f"Canceled. Sent {sent}.")
                        break

                    if ok:
                        try:
                            await safe_send(c.bot.send_chat_action, cid, action="upload_document")
                            with open(op, "rb") as f:
                                await safe_send(c.bot.send_document, cid, document=f, filename=op.name,
                                                          caption=f"MangaHue \u00b7 p.{pi + 1}",
                                                          read_timeout=120, write_timeout=120)
                            sent += 1
                            await s.record(cid)
                        except Exception as ex:
                            sys.stderr.write(f"[send] {ex}\n")
                            failed += 1
                            failed_pages[pi] = urls[pi]
                    else:
                        failed += 1
                        failed_pages[pi] = urls[pi]

                    times.append(time.monotonic() - t0)
                    lp.unlink(missing_ok=True); op.unlink(missing_ok=True)

                    done = sent + failed
                    eta_t = _eta(times, done, total)
                    txt = f"`{_bar(done, total)}` {(done / total * 100) if total else 0:.0f}%\n{done}/{total} \u00b7 b{bi + 1}/{batches}"
                    if failed:
                        txt += f" \u00b7 fail {failed}"
                    if eta_t:
                        txt += f" \u00b7 {eta_t}"
                    try:
                        await safe_send(msg.edit_text, txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                    except Exception:
                        pass
                    await asyncio.sleep(DELAY)

        final = f"Done. Sent *{sent}* page(s)."
        kb_final = InlineKeyboardMarkup([[InlineKeyboardButton("Colorize Another", callback_data="colorize_another")]])

        if failed_pages:
            final += f" *{len(failed_pages)}* failed."
            try:
                failed_dir = WDIR / str(cid) / "failed"
                failed_dir.mkdir(parents=True, exist_ok=True)
                info = {
                    "urls": {str(k): v for k, v in failed_pages.items()},
                    "original_url": url,
                    "total_sent": sent,
                    "retried": {},
                    "retry_count": {},
                }
                failed_fp = failed_dir / f"{jid}.json"
                failed_fp.write_text(json.dumps(info), encoding="utf-8")

                buttons: list[list[InlineKeyboardButton]] = []
                for pi in sorted(failed_pages.keys()):
                    buttons.append([InlineKeyboardButton(
                        f"Retry page {pi + 1}",
                        callback_data=f"retry_page_{jid}_{pi}")])
                buttons.append([InlineKeyboardButton(
                    "Retry All Failed",
                    callback_data=f"retry_all_{jid}")])
                retry_kb = InlineKeyboardMarkup(buttons)
                await safe_send(c.bot.send_message, cid, "Failed pages \u2014 click to retry:", reply_markup=retry_kb)
            except Exception as ex:
                sys.stderr.write(f"[retry save] {ex}\n")
                final += " Resend URL to retry."

        try:
            await safe_send(msg.edit_text, final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)
        except Exception:
            await safe_send(c.bot.send_message, cid, final, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_final)

    except Exception as exc:
        traceback.print_exc()
        try:
            await safe_send(msg.edit_text, f"Error: {exc}")
        except Exception:
            await safe_send(c.bot.send_message, cid, f"MangaHue error: {exc}")
        if ADMIN_ID:
            try:
                await safe_send(c.bot.send_message, ADMIN_ID, f"[Error] chat={cid}: {exc}")
            except Exception:
                pass
    finally:
        shutil.rmtree(job, ignore_errors=True)
        s.finish(cid)

# -- Admin commands ---------------------------------------------------------
async def _admin_users(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    total_users = len(s.known_users)
    active_today = s.active_today_count()
    await safe_send(u.message.reply_text,
        f"*Users:* {total_users} total \u00b7 {active_today} active today",
        parse_mode=ParseMode.MARKDOWN)

async def _admin_user(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    args = c.args
    if not args or not args[0].lstrip("-").isdigit():
        await safe_send(u.message.reply_text, "Usage: `/user <chat_id>`", parse_mode=ParseMode.MARKDOWN)
        return
    target = int(args[0])
    usage = s.used(target)
    limit = s.limit_for(target)
    last = s.last_active.get(target, "unknown")
    blocked = target in s.blocked
    stg = s.get_settings(target)
    await safe_send(u.message.reply_text,
        f"*User:* `{target}`\n"
        f"*Pages used:* {usage}/{limit}\n"
        f"*Last active:* {last[:19] if last != 'unknown' else last}\n"
        f"*Blocked:* {'yes' if blocked else 'no'}\n"
        f"*Settings:* {json.dumps(stg)}",
        parse_mode=ParseMode.MARKDOWN)

async def _admin_broadcast(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    msg_text = " ".join(c.args) if c.args else ""
    if not msg_text:
        await safe_send(u.message.reply_text, "Usage: `/broadcast <message>`", parse_mode=ParseMode.MARKDOWN)
        return
    users = list(s.known_users)
    if not users:
        await safe_send(u.message.reply_text, "No users to broadcast to.")
        return
    await safe_send(u.message.reply_text, f"Broadcasting to {len(users)} users...")
    sent_count, fail_count = 0, 0
    for uid in users:
        try:
            await safe_send(c.bot.send_message, uid, f"[Admin broadcast]\n{msg_text}")
            sent_count += 1
            await asyncio.sleep(0.05)
        except Exception:
            fail_count += 1
    await safe_send(u.message.reply_text, f"Broadcast done: {sent_count} sent, {fail_count} failed.")

async def _admin_block(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    args = c.args
    if not args or not args[0].lstrip("-").isdigit():
        await safe_send(u.message.reply_text, "Usage: `/block <chat_id>`", parse_mode=ParseMode.MARKDOWN)
        return
    target = int(args[0])
    if target == ADMIN_ID:
        await safe_send(u.message.reply_text, "Cannot block yourself.")
        return
    s.blocked.add(target)
    await s._save_users()
    await safe_send(u.message.reply_text, f"Blocked `{target}`.", parse_mode=ParseMode.MARKDOWN)

async def _admin_unblock(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    args = c.args
    if not args or not args[0].lstrip("-").isdigit():
        await safe_send(u.message.reply_text, "Usage: `/unblock <chat_id>`", parse_mode=ParseMode.MARKDOWN)
        return
    target = int(args[0])
    s.blocked.discard(target)
    await s._save_users()
    await safe_send(u.message.reply_text, f"Unblocked `{target}`.", parse_mode=ParseMode.MARKDOWN)

async def _admin_limits(u: Update, c):
    if not _is_admin(u):
        return
    s: BotState = c.bot_data["state"]
    args = c.args
    if len(args) < 2 or not args[0].lstrip("-").isdigit() or not args[1].isdigit():
        await safe_send(u.message.reply_text, "Usage: `/limits <chat_id> <new_limit>`", parse_mode=ParseMode.MARKDOWN)
        return
    target, new_limit = int(args[0]), int(args[1])
    if new_limit < 0:
        await safe_send(u.message.reply_text, "Limit must be >= 0.")
        return
    s.custom_limits[target] = new_limit
    await s._save_users()
    await safe_send(u.message.reply_text, f"Limit for `{target}` set to *{new_limit}*.", parse_mode=ParseMode.MARKDOWN)

# -- main -------------------------------------------------------------------
def main():
    if not TOKEN:
        print("Set TELEGRAM_BOT_TOKEN.", file=sys.stderr)
        sys.exit(1)

    log = get_logger("mangahue.bot")
    state = BotState()
    app = Application.builder().token(TOKEN).build()
    app.bot_data["state"] = state
    for cmd, fn in [("start", _start), ("help", _help), ("status", _status),
                     ("cancel", _cancel), ("batch", _batch), ("stats", _stats),
                     ("clean", _clean)]:
        app.add_handler(CommandHandler(cmd, fn))
    for cmd, fn in [("users", _admin_users), ("user", _admin_user),
                     ("broadcast", _admin_broadcast), ("block", _admin_block),
                     ("unblock", _admin_unblock), ("limits", _admin_limits)]:
        app.add_handler(CommandHandler(cmd, fn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, _on_image))
    app.add_handler(CallbackQueryHandler(_callback))

    async def _start_health(post_app):
        health_port = os.environ.get("HEALTH_PORT", "")
        if not health_port:
            return
        from aiohttp import web
        async def health(request):
            checks = {}
            status = "ok"
            checks["model"] = len(Colorizer._CC) > 0
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.telegram.org/bot{TOKEN}/getMe",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        checks["telegram"] = resp.status == 200
            except Exception:
                checks["telegram"] = False
            try:
                usage = shutil.disk_usage(WDIR)
                free_pct = round(usage.free / usage.total * 100, 1)
                checks["disk"] = {"free_pct": free_pct}
                if free_pct < 5:
                    status = "degraded"
            except Exception:
                checks["disk"] = {"free_pct": -1}
                status = "degraded"
            if not checks.get("model") or not checks.get("telegram"):
                status = "degraded"
            return web.json_response({"status": status, "checks": checks})
        web_app = web.Application()
        web_app.router.add_get("/health", health)
        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", int(health_port))
        await site.start()
        post_app.bot_data["_health"] = (runner, site)
        log.info("Health server listening on :%s", health_port)

    app.post_init = _start_health

    log.info("MangaHue bot starting (long-polling)...")
    try:
        app.run_polling(allowed_updates=["message", "callback_query"])
    finally:
        log.info("Cleaning up temp files...")
        jobs_dir = WDIR / "jobs"
        if jobs_dir.exists():
            shutil.rmtree(jobs_dir, ignore_errors=True)
        log.info("Shutdown complete.")

if __name__ == "__main__":
    main()
