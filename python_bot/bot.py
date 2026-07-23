"""
MangaHue — Local Manga Colorization Bot
Uses local ONNX model with smart semantic hints for beautiful, natural manga colors.
"""
import asyncio, json, os, re, shutil, signal, sys, threading, time, traceback
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

from logger import get_logger
from hints import AdvancedHintGenerator
from postprocess import apply_manga_pipeline
from downloader import MangaDownloader

# -- Colorizer (ONNX + smart hints) -----------------------------------------
class Colorizer:
    _CC: dict[str, tuple[ort.InferenceSession, threading.Lock]] = {}

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
            Colorizer._CC[k] = (ort.InferenceSession(k, opts, providers=["CPUExecutionProvider"]), threading.Lock())
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
        hm_pil = hm_pil.filter(ImageFilter.GaussianBlur(radius=6))
        hk_pil = hk_pil.filter(ImageFilter.GaussianBlur(radius=6))
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
        sess, lock = self._sess()
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        with lock:
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
                saved = []
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        saved = await dl.download_batch(urls[b0:b1], str(dl_dir), start=b0 + 1)
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

# -- health server ----------------------------------------------------------
def _run_health_server(port: int):
    from aiohttp import web

    async def health(request):
        return web.json_response({"status": "ok"})

    async def start():
        app = web.Application()
        app.router.add_get("/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        while True:
            await asyncio.sleep(3600)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start())
    except Exception:
        pass

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))
    app.add_handler(CallbackQueryHandler(_callback))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _make_shutdown(sig_name):
        def handler():
            log.info("Received %s, shutting down...", sig_name)
            asyncio.create_task(app.stop())
        return handler

    for sig_name, sig_val in [("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)]:
        try:
            loop.add_signal_handler(sig_val, _make_shutdown(sig_name))
        except (NotImplementedError, ValueError):
            pass

    health_port = os.environ.get("HEALTH_PORT", "")
    if health_port:
        t = threading.Thread(target=_run_health_server, args=(int(health_port),),
                             daemon=True, name="health-server")
        t.start()
        log.info("Health server listening on :%s", health_port)

    log.info("MangaHue bot starting (long-polling)...")
    try:
        app.run_polling(allowed_updates=["message", "callback_query"])
    finally:
        log.info("Cleaning up temp files...")
        shutil.rmtree(WDIR, ignore_errors=True)
        log.info("Shutdown complete.")

if __name__ == "__main__":
    main()
