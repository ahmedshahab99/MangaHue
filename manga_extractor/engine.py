"""MangaExtractor - main download engine with retry, concurrency, progress."""
import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional
import httpx
from .sites import get_handler, get_mangadex, HEADERS
from .utils import max_resolve, chunked

logger = logging.getLogger(__name__)


class MangaExtractor:
    """Download manga pages from any supported site.

    Usage:
        async with MangaExtractor() as mx:
            pages = await mx.from_url("https://mangadex.org/chapter/...")
            # pages is a list of max-resolution image URLs
    """

    def __init__(
        self,
        concurrency: int = 5,
        max_retries: int = 3,
        timeout: float = 30,
        progress: Optional[Callable] = None,
    ):
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._timeout = timeout
        self.progress = progress
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use 'async with MangaExtractor()' context manager")
        return self._client

    # ─── Public API ─────────────────────────────────────────────────

    async def from_url(self, url: str) -> list[str]:
        """Extract all page image URLs from any manga URL.

        Returns max-resolution URLs. For MangaDex chapter URLs or other
        chapter pages, returns the pages directly. For MangaDex title
        URLs, returns pages from the first chapter.
        """
        handler = get_handler(url)
        if not handler:
            return []
        raw = await handler.extract_pages(url, self.client)
        return [max_resolve(u) for u in raw]

    async def search(self, title: str) -> list[dict]:
        md = get_mangadex()
        return await md.search(title, self.client)

    async def get_chapters(self, url_or_id: str) -> list[dict]:
        handler = get_handler(url_or_id)
        if handler:
            return await handler.extract_chapters(url_or_id, self.client)
        md = get_mangadex()
        return await md.extract_chapters(url_or_id, self.client)

    async def get_pages(self, chapter_id: str) -> list[str]:
        md = get_mangadex()
        raw = await md.extract_pages(f"https://mangadex.org/chapter/{chapter_id}", self.client)
        return [max_resolve(u) for u in raw]

    async def download_chapter(
        self,
        chapter_id: str,
        output_dir: str | Path,
        progress: Optional[Callable] = None,
    ) -> list[Path]:
        """Download all pages of a MangaDex chapter to disk."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pages = await self.get_pages(chapter_id)
        paths = await self.download_images(pages, output, progress)
        return paths

    async def download_images(
        self,
        urls: list[str],
        output_dir: str | Path,
        progress: Optional[Callable] = None,
    ) -> list[Path]:
        """Download multiple images concurrently to a directory."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self._concurrency)
        paths = [None] * len(urls)

        async def _dl(i, u):
            async with sem:
                for attempt in range(self._max_retries):
                    try:
                        r = await self.client.get(u)
                        if r.status_code == 200:
                            ext = u.rsplit(".", 1)[-1].split("?")[0].split("#")[0]
                            if ext.lower() not in ("jpg", "jpeg", "png", "webp", "avif", "gif"):
                                ext = "jpg"
                            fname = output / f"{i+1:04d}.{ext}"
                            fname.write_bytes(r.content)
                            paths[i] = fname
                            if progress:
                                progress(i + 1, len(urls))
                            return
                    except Exception:
                        if attempt < self._max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                        else:
                            logger.error(f"Failed to download {u}")

        await asyncio.gather(*[_dl(i, u) for i, u in enumerate(urls)])
        return [p for p in paths if p is not None]

    async def download_to_bytes(self, url: str) -> bytes | None:
        """Download a single image to bytes."""
        for attempt in range(self._max_retries):
            try:
                r = await self.client.get(url)
                if r.status_code == 200:
                    return r.content
            except Exception:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return None
