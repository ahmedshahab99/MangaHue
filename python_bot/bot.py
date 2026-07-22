"""
MangaHue Telegram Bot — Async long-polling bot using python-telegram-bot v20+
Send a manga chapter URL, receive colorized pages one-by-one.
"""
import asyncio
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes,
)

from downloader import MangaDownloader
from colorize import Colorizer
from postprocess import apply_manga_pipeline

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DAILY_PAGE_LIMIT = int(os.environ.get("DAILY_PAGE_LIMIT", "100"))
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "")
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "5"))
WORK_DIR = Path(os.environ.get("MANGAHUE_WORK_DIR", ".mangahue/telegram"))
USAGE_FILE = WORK_DIR / "usage" / "daily.json"
STYLES_DIR = Path(os.environ["MANGAHUE_STYLES_DIR"]) if os.environ.get("MANGAHUE_STYLES_DIR") else None
MOCK = os.environ.get("MANGAHUE_MOCK", "0") == "1"
COLORIZE_CMD = os.environ.get("MANGAHUE_COLORIZE_CMD", "cargo run -p mangahue-cli -- colorize")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
SEND_DELAY = float(os.environ.get("MIN_SEND_INTERVAL", "3.1"))

BUILTIN_STYLES = [
    ("natural_manga", "Natural Manga (default)"),
    ("realistic_comic", "Realistic Comic"),
    ("dark_seinen", "Dark Seinen"),
    ("watercolor_soft", "Watercolor Soft"),
    ("retro_90s_anime", "Retro 90s Anime"),
]

DIRECT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


class BotState:
    def __init__(self) -> None:
        self.usage: dict[int, int] = {}
        self.settings: dict[int, dict] = {}
        self.cancels: dict[int, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today:
                self.usage = {int(k): v for k, v in data.get("pages", {}).items()}
        except (FileNotFoundError, json.JSONDecodeError):
            self.usage = {}

    async def _save(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with self._lock:
            USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            USAGE_FILE.write_text(json.dumps({"date": today, "pages": self.usage}), encoding="utf-8")

    def remaining(self, cid: int) -> int:
        return max(0, DAILY_PAGE_LIMIT - self.usage.get(cid, 0))

    def used(self, cid: int) -> int:
        return self.usage.get(cid, 0)

    async def record(self, cid: int, n: int) -> None:
        self.usage[cid] = self.usage.get(cid, 0) + n
        await self._save()

    def get_settings(self, cid: int) -> dict:
        if cid not in self.settings:
            self.settings[cid] = {"style": None, "batch": DEFAULT_BATCH_SIZE}
        return self.settings[cid]

    def try_start(self, cid: int) -> Optional[asyncio.Event]:
        if cid in self.cancels and not self.cancels[cid].is_set():
            return None
        if sum(1 for e in self.cancels.values() if not e.is_set()) >= MAX_CONCURRENT:
            return None
        e = asyncio.Event()
        self.cancels[cid] = e
        return e

    def finish(self, cid: int) -> None:
        self.cancels.pop(cid, None)

    def is_active(self, cid: int) -> bool:
        e = self.cancels.get(cid)
        return e is not None and not e.is_set()

    async def cancel(self, cid: int) -> bool:
        e = self.cancels.get(cid)
        if e is None or e.is_set():
            return False
        e.set()
        return True

    def styles(self) -> list[tuple[str, str]]:
        out = list(BUILTIN_STYLES)
        if STYLES_DIR and STYLES_DIR.is_dir():
            for p in sorted(STYLES_DIR.glob("*.json")):
                name = None
                try:
                    name = json.loads(p.read_text(encoding="utf-8")).get("name")
                except Exception:
                    pass
                out.append((p.stem, name or p.stem))
        return out


def _bar(done: int, total: int, w: int = 14) -> str:
    if total <= 0:
        return "░" * w
    f = min(round((done / total) * w), w)
    return "▓" * f + "░" * (w - f)


def _eta(times: list[float], done: int, total: int) -> Optional[str]:
    if not times or done >= total or done <= 0:
        return None
    avg = sum(times) / len(times)
    s = avg * (total - done)
    return f"~{s:.0f}s" if s < 60 else f"~{s / 60:.1f}m"


def _extract_url(text: str) -> Optional[str]:
    for t in re.split(r"\s+", text):
        t = t.strip("\"'()[]<>,;")
        if t.startswith(("http://", "https://")):
            return t.rstrip(".,)!]?>")
    return None


def _validate(url: str) -> tuple[Optional[str], Optional[str]]:
    try:
        p = urlparse(url)
    except Exception:
        return "Invalid URL.", None
    if p.scheme not in ("http", "https"):
        return "Only HTTP/HTTPS URLs are supported.", None
    h = p.hostname or ""
    if h in ("localhost", "127.0.0.1", "0.0.0.0") or h.startswith(("192.168.", "10.")):
        return "Private network URLs not allowed.", None
    if any(p.path.lower().endswith(ext) for ext in DIRECT_EXTS):
        return None, "Processing single image..."
    return None, None


async def _colorize(inp: Path, out: Path, style: Optional[str] = None) -> bool:
    colorizer = Colorizer(backend="auto")
    try:
        ok = await colorizer.colorize(str(inp), str(out))
        if ok:
            apply_manga_pipeline(str(out), str(inp), str(out))
        return ok
    except Exception as exc:
        sys.stderr.write(f"[colorize] error: {exc}\n")
        return False


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*MangaHue* — send a manga chapter URL or direct image link "
        "and I will colorize it with AI.\n\nUse /help for commands.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*MangaHue Commands:*\n\n"
        "/start — Start\n/help — This help\n/status — Active job\n"
        "/cancel — Stop job\n/style — Styles\n/batch — Batch size (1–10)\n"
        "/stats — Daily usage\n/clean — Delete job data\n\n"
        "*How:* Send a manga chapter URL. Pages are colorized in batches "
        f"of {DEFAULT_BATCH_SIZE} and sent immediately.\n"
        f"Limit: {DAILY_PAGE_LIMIT} pages/day.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    await update.message.reply_text(
        "A job is running. /cancel to stop."
        if s.is_active(update.effective_chat.id)
        else "No active job. Send a URL to start."
    )


async def _cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    ok = await s.cancel(update.effective_chat.id)
    await update.message.reply_text(
        "Cancel requested. Will stop at next safe point."
        if ok else "No active job to cancel."
    )


async def _style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    arg = " ".join(context.args) if context.args else ""
    styles = s.styles()
    cur = s.get_settings(cid).get("style") or "natural_manga"

    if arg:
        found = next((st for st in styles if st[0] == arg), None)
        if found:
            s.get_settings(cid)["style"] = found[0]
            await update.message.reply_text(f"Style: *{found[1]}*.", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"Style '{arg}' not found. Use /style to list.")
        return

    rows = [[InlineKeyboardButton(n, callback_data=f"style:{i}") for i, n in styles[k:k+2]]
            for k in range(0, len(styles), 2)]
    await update.message.reply_text(f"Current: *{cur}*", parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text("Choose:", reply_markup=InlineKeyboardMarkup(rows))


async def _batch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    arg = " ".join(context.args) if context.args else ""
    stg = s.get_settings(cid)

    if not arg:
        await update.message.reply_text(
            f"Batch size: *{stg['batch']}*. `/batch N` (1–10).",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if not arg.isdigit() or not (1 <= int(arg) <= 10):
        await update.message.reply_text("Batch size must be 1–10.")
        return

    stg["batch"] = int(arg)
    await update.message.reply_text(f"Batch size: *{arg}*.", parse_mode=ParseMode.MARKDOWN)


async def _stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    u = s.used(cid)
    r = s.remaining(cid)
    await update.message.reply_text(
        f"`{_bar(u, DAILY_PAGE_LIMIT, 16)}` *{u}* / {DAILY_PAGE_LIMIT}\n"
        f"*{r}* remaining today",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _clean(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    if s.is_active(cid):
        await update.message.reply_text("Job running. /cancel first.")
        return
    d = WORK_DIR / str(cid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        s.settings.pop(cid, None)
        await update.message.reply_text("Job data cleaned up.")
    else:
        await update.message.reply_text("No data to clean.")


async def _callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    d = q.data

    if d == "cancel_job":
        ok = await s.cancel(cid)
        await q.edit_message_text("Cancelled." if ok else "No active job.")
    elif d == "colorize_another":
        await q.message.reply_text("Send another URL.")
    elif d == "retry_failed":
        await q.message.reply_text("Send the URL again to retry.")
    elif d and d.startswith("style:"):
        sid = d.split(":", 1)[1]
        found = next((x for x in s.styles() if x[0] == sid), None)
        if found:
            s.get_settings(cid)["style"] = found[0]
            await q.edit_message_text(f"Style: *{found[1]}*.", parse_mode=ParseMode.MARKDOWN)


async def _on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: BotState = context.bot_data["state"]
    cid = update.effective_chat.id
    url = _extract_url(update.message.text.strip())

    if not url:
        await update.message.reply_text("Send a manga chapter URL. Use /help for commands.")
        return

    err, warn = _validate(url)
    if warn:
        await update.message.reply_text(warn)
    if err:
        await update.message.reply_text(err)
        return
    if not err and not warn:
        await update.message.reply_text("Processing chapter...")

    if s.remaining(cid) <= 0:
        await update.message.reply_text(
            f"Daily limit ({DAILY_PAGE_LIMIT}) reached. Use /stats."
        )
        return

    evt = s.try_start(cid)
    if evt is None:
        await update.message.reply_text("Job already active or pool full.")
        return

    asyncio.create_task(_process(update, context, url, cid, evt))


async def _process(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    url: str, cid: int, cancel: asyncio.Event,
) -> None:
    s: BotState = context.bot_data["state"]
    stg = s.get_settings(cid)
    bsize = stg["batch"]
    style = stg.get("style")
    jid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job = WORK_DIR / str(cid) / jid
    dldir = job / "downloads"
    outdir = job / "colored"
    dldir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    msg = await update.message.reply_text("Downloading pages...")
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel_job")]])

    try:
        async with MangaDownloader() as dl:
            urls = await dl.extract_urls(url)
            if not urls:
                await msg.edit_text("No page images found.")
                return

            rem = s.remaining(cid)
            total = min(len(urls), rem) if rem else len(urls)
            if total <= 0:
                await msg.edit_text(f"Daily limit reached.")
                return

            batches = (total + bsize - 1) // bsize
            await msg.edit_text(
                f"Found *{total}* page(s). {batches} batch(es) of *{bsize}*.",
                parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
            )

            sent = 0
            failed = 0
            times: list[float] = []

            for bi in range(batches):
                if cancel.is_set():
                    await msg.edit_text(f"Cancelled after {sent} page(s).")
                    break

                bstart = bi * bsize
                bend = min(bstart + bsize, total)
                saved = await dl.download_batch(urls[bstart:bend], str(dldir), start_index=bstart + 1)

                smap: dict[int, Path] = {}
                for p in saved:
                    smap[int(Path(p).stem) - 1] = Path(p)

                for pi in range(bstart, bend):
                    if cancel.is_set():
                        break

                    lp = smap.get(pi)
                    if not lp:
                        failed += 1
                        continue

                    t0 = time.monotonic()
                    op = outdir / f"page_{pi:03d}.png"
                    ok = await _colorize(lp, op, style)

                    if cancel.is_set():
                        await msg.edit_text(f"Cancelled after {sent} page(s).")
                        break

                    if ok:
                        try:
                            await context.bot.send_chat_action(chat_id=cid, action="upload_document")
                            with open(op, "rb") as f:
                                await context.bot.send_document(
                                    chat_id=cid, document=f, filename=op.name,
                                    caption=f"MangaHue p.{pi + 1}",
                                    read_timeout=120, write_timeout=120,
                                )
                            sent += 1
                            await s.record(cid, 1)
                        except Exception:
                            failed += 1
                    else:
                        failed += 1

                    times.append(time.monotonic() - t0)
                    lp.unlink(missing_ok=True)
                    op.unlink(missing_ok=True)

                    done = sent + failed
                    eta = _eta(times, done, total)
                    txt = (
                        f"`{_bar(done, total)}` {(done / total * 100) if total else 0:.0f}%\n"
                        f"Pages: {done}/{total}  Batch: {bi + 1}/{batches}"
                    )
                    if failed:
                        txt += f"  Failed: {failed}"
                    if eta:
                        txt += f"  ETA: {eta}"
                    try:
                        await msg.edit_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                    except Exception:
                        pass
                    await asyncio.sleep(SEND_DELAY)

        final = f"Done. Sent *{sent}* colored page(s)."
        if failed:
            final += f" *{failed}* failed. Send the URL again to retry."
        dkb = InlineKeyboardMarkup([[InlineKeyboardButton("Colorize Another", callback_data="colorize_another")]])
        try:
            await msg.edit_text(final, parse_mode=ParseMode.MARKDOWN, reply_markup=dkb)
        except Exception:
            await context.bot.send_message(chat_id=cid, text=final,
                                           parse_mode=ParseMode.MARKDOWN, reply_markup=dkb)

    except Exception as exc:
        try:
            await msg.edit_text(f"Error: {exc}")
        except Exception:
            await context.bot.send_message(chat_id=cid, text=f"MangaHue error: {exc}")
        if ADMIN_CHAT_ID:
            try:
                await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID),
                                               text=f"[Alert] {cid} failed: {exc}")
            except Exception:
                pass
    finally:
        shutil.rmtree(job, ignore_errors=True)
        s.finish(cid)


def main() -> None:
    if not TOKEN:
        print("Set TELEGRAM_BOT_TOKEN.", file=sys.stderr)
        sys.exit(1)

    st = BotState()
    app = Application.builder().token(TOKEN).build()
    app.bot_data["state"] = st

    for cmd, fn in [("start", _start), ("help", _help), ("status", _status),
                     ("cancel", _cancel), ("style", _style), ("batch", _batch),
                     ("stats", _stats), ("clean", _clean)]:
        app.add_handler(CommandHandler(cmd, fn))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))
    app.add_handler(CallbackQueryHandler(_callback))

    print("MangaHue Telegram bot starting (long-polling)...")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
