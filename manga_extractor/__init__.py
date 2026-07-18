"""Manga Extractor - download manga from any site and forward to Telegram.

Usage:
    from manga_extractor import MangaExtractor, TelegramSender

    async with MangaExtractor() as mx:
        pages = await mx.from_url("https://...")
        results = await mx.search("One Piece")
"""

from .engine import MangaExtractor
from .sender import TelegramSender
from .utils import max_resolve

__all__ = ["MangaExtractor", "TelegramSender", "max_resolve"]
__version__ = "2.0.0"
