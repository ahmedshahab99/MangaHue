"""Manga Telegram Bot - send any manga URL and receive pages 10 at a time."""
import io
import logging
import asyncio
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

from manga_extractor import MangaExtractor, max_resolve

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("manga_bot")

BOT_TOKEN = "8872444303:AAFn8lwLhzvGrPyHpJKzwLeZowakmi_YWIw"
IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://nhentai.net/",
}


async def send_batches(chat_id, urls: list[str], ctx) -> int:
    """Download images in batches of 10 and send via sendMediaGroup."""
    sent = 0
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for i in range(0, len(urls), 10):
            batch = urls[i : i + 10]
            media = []
            for url in batch:
                hi = max_resolve(url)
                try:
                    r = await client.get(hi, headers=IMG_HEADERS)
                    if r.status_code == 200:
                        media.append(InputMediaPhoto(media=io.BytesIO(r.content)))
                except Exception as e:
                    logger.debug(f"DL fail {hi[:60]}: {e}")
            if media:
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
                sent += len(media)
            await asyncio.sleep(1)
    return sent


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a manga URL or name.\n"
        "Supported: MangaDex, MangaNato, MangaKakalot, nhentai, AsuraScans, etc.\n"
        "I download and send pages 10 at a time in max resolution."
    )


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    await update.message.reply_text("Cancelled.")


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.startswith("/"):
        return
    if text.startswith(("http://", "https://")):
        await handle_url(update, ctx, text)
        return
    await handle_search(update, ctx, text)


async def handle_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE, query: str):
    msg = await update.message.reply_text(f"Searching: {query}")
    try:
        async with MangaExtractor() as mx:
            results = await mx.search(query)
        if not results:
            await msg.edit_text("Nothing found on MangaDex. Try a URL instead.")
            return
        kb = [[InlineKeyboardButton(m["title"], callback_data=f"m_{m['id']}")] for m in results]
        await msg.edit_text("Pick one:", reply_markup=InlineKeyboardMarkup(kb))
    except Exception as e:
        await msg.edit_text(f"Search error: {e}")


async def handle_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE, url: str):
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower().replace("www.", "")
    msg = await update.message.reply_text(f"Scraping {domain}...")
    cid = update.effective_chat.id

    if "mangadex" in domain:
        import re
        cid_match = re.search(r"/chapter/([a-f0-9-]{36})", url)
        mid_match = re.search(r"/title/([a-f0-9-]{36})", url)
        try:
            async with MangaExtractor() as mx:
                if cid_match:
                    await msg.edit_text("Fetching chapter pages...")
                    pages = await mx.from_url(url)
                    if not pages:
                        await msg.edit_text("No pages found.")
                        return
                    await msg.edit_text(f"Sending {len(pages)} pages (10 per batch)...")
                    sent = await send_batches(cid, pages, ctx)
                    await ctx.bot.send_message(chat_id=cid, text=f"Done - {sent} pages sent.")
                elif mid_match:
                    await msg.edit_text("Loading chapters...")
                    chapters = await mx.get_chapters(url)
                    if not chapters:
                        await msg.edit_text("No chapters found.")
                        return
                    ctx.user_data["chapters"] = chapters
                    kb = []
                    for ch in chapters[:100]:
                        label = f"Ch. {ch['chapter']}"
                        if ch.get("title"):
                            label += f" - {ch['title'][:35]}"
                        kb.append([InlineKeyboardButton(label, callback_data=f"ch_{ch['id']}")])
                    if len(chapters) > 1:
                        kb.append([InlineKeyboardButton("Download ALL", callback_data="dl_all")])
                    await msg.edit_text(f"{len(chapters)} chapters:", reply_markup=InlineKeyboardMarkup(kb))
                else:
                    await msg.edit_text("Not a valid MangaDex URL.")
        except Exception as e:
            await msg.edit_text(f"Error: {e}")
        return

    try:
        async with MangaExtractor() as mx:
            pages = await mx.from_url(url)
        if not pages:
            await msg.edit_text(f"No manga images found on {domain}.")
            return
        await msg.edit_text(f"Sending {len(pages)} pages (10 per batch)...")
        sent = await send_batches(cid, pages, ctx)
        await ctx.bot.send_message(chat_id=cid, text=f"Done - {sent} pages sent.")
    except Exception as e:
        logger.error(f"Scrape error: {e}", exc_info=True)
        await msg.edit_text(f"Error: {e}")


async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    cid = q.message.chat_id

    try:
        if data.startswith("m_"):
            mid = data[2:]
            await q.edit_message_text("Loading chapters...")
            async with MangaExtractor() as mx:
                chapters = await mx.get_chapters(f"https://mangadex.org/title/{mid}")
            if not chapters:
                await q.edit_message_text("No chapters.")
                return
            ctx.user_data["chapters"] = chapters
            kb = []
            for ch in chapters[:100]:
                label = f"Ch. {ch['chapter']}"
                if ch.get("title"):
                    label += f" - {ch['title'][:35]}"
                kb.append([InlineKeyboardButton(label, callback_data=f"ch_{ch['id']}")])
            if len(chapters) > 1:
                kb.append([InlineKeyboardButton("Download ALL", callback_data="dl_all")])
            await q.edit_message_text(f"{len(chapters)} chapters:", reply_markup=InlineKeyboardMarkup(kb))

        elif data.startswith("ch_"):
            cid_ch = data[3:]
            await q.edit_message_text("Downloading pages...")
            async with MangaExtractor() as mx:
                pages = await mx.get_pages(cid_ch)
            if not pages:
                await q.edit_message_text("No pages.")
                return
            await q.edit_message_text(f"Sending {len(pages)} pages...")
            sent = await send_batches(cid, pages, ctx)
            await ctx.bot.send_message(chat_id=cid, text=f"Done - {sent} pages sent.")

        elif data == "dl_all":
            chapters = ctx.user_data.get("chapters", [])
            if not chapters:
                await q.edit_message_text("No chapters loaded. Search again.")
                return
            await q.edit_message_text(f"Downloading {len(chapters)} chapters...")
            total = 0
            async with MangaExtractor() as mx:
                for ch in chapters:
                    try:
                        pages = await mx.get_pages(ch["id"])
                        await ctx.bot.send_message(chat_id=cid, text=f"Ch. {ch['chapter']} ({len(pages)} pages)...")
                        sent = await send_batches(cid, pages, ctx)
                        total += sent
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        await ctx.bot.send_message(chat_id=cid, text=f"Ch. {ch['chapter']} failed: {e}")
            await ctx.bot.send_message(chat_id=cid, text=f"All done - {total} pages total.")

    except Exception as e:
        logger.error(f"Button error: {e}", exc_info=True)
        await q.edit_message_text(f"Error: {e}")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(button_handler))
    logger.info("Manga Bot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
