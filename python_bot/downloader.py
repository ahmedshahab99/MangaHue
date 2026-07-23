from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

HTML_TIMEOUT = aiohttp.ClientTimeout(total=30)
IMAGE_TIMEOUT = aiohttp.ClientTimeout(total=60)
REQUEST_DELAY = 0.5  # seconds between requests
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

AVATAR_KEYWORDS = {
    "avatar", "logo", "icon", "spinner", "loading", "thumbnail", "banner",
    "cover", "preview", "profile", "rating", "badge", "button", "pixel",
    "tracker", "analytics", "favicon", "default", "placeholder", "blur",
}


class Site(Enum):
    NHENTAI = auto()
    MANGADEX = auto()
    DIRECT = auto()
    GENERIC = auto()
    EHENTAI = auto()
    MANGAPLUS = auto()
    WEBTOONS = auto()
    BATOTO = auto()


# ---------------------------------------------------------------------------
# helper predicates
# ---------------------------------------------------------------------------

def detect_site(url: str) -> Site:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if "nhentai.net" in host and "/g/" in url:
        return Site.NHENTAI
    if "mangadex.org" in host and "/chapter/" in url:
        return Site.MANGADEX
    if is_direct_image(url):
        return Site.DIRECT
    if ("e-hentai.org" in host or "exhentai.org" in host) and "/g/" in url:
        return Site.EHENTAI
    if "mangaplus.shueisha.co.jp" in host and "/viewer/" in url:
        return Site.MANGAPLUS
    if "webtoons.com" in host and "/viewer" in url:
        return Site.WEBTOONS
    if any(d in host for d in ("bato.to", "batotoo.com", "mto.to", "wto.to", "dto.to")):
        return Site.BATOTO
    return Site.GENERIC


def is_direct_image(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_high_res(url: str) -> bool:
    """Return False if `url` looks like a low-res / thumbnail variant."""
    lower = url.lower()

    if re.search(r"(thumb|thumbnail|_t|_thumb|small|icon|avatar|logo)", lower):
        return False
    if re.search(r"[-_]\d{2,4}x\d{2,4}", lower):
        return False
    if "placeholder" in lower or "loading" in lower:
        return False
    return True


def _normalise_nhentai_url(url: str) -> str:
    """Convert nhentai thumbnail → full-res CDN URL."""
    lower = url.lower()
    if "/galleries/" not in lower and "/g/" not in lower:
        return url

    # subdomain: i{n} for galleries, t{n} for thumbnails
    url = re.sub(r"://t(\d*)\.", r"://i\1.", url)
    url = re.sub(r"://t(\d*)\.nhentai\.net", r"://i\1.nhentai.net", url)

    # remove thumbnail markers inside path
    url = re.sub(r"/(\d+)t\.(jpg|png|webp|gif)", r"/\1.\2", url)
    return url


def _is_valid_page_url(url: str) -> bool:
    """Filter out avatars / spinners / UI chrome from candidate image URLs."""
    lower = url.lower()
    for keyword in AVATAR_KEYWORDS:
        if keyword in lower:
            return False
    if lower.endswith((".svg", ".ico")):
        return False
    return True


# ---------------------------------------------------------------------------
# site-specific extractors
# ---------------------------------------------------------------------------

async def _extract_nhentai(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    match = re.search(r"/g/(\d+)", url)
    if not match:
        raise ValueError(f"Could not parse nhentai gallery ID from: {url}")
    gallery_id = match.group(1)

    pages_map: dict[int, str] = {}

    # Strategy 1: scrape CDN images from HTML (fastest)
    try:
        async with session.get(url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
            html = await resp.text()
    except Exception as exc:
        logger.warning("nhentai HTML fetch failed for %s: %s", url, exc)
        html = ""

    if html:
        soup = BeautifulSoup(html, "html.parser")

        # <noscript><img src="..."></noscript>
        for noscript in soup.find_all("noscript"):
            inner = BeautifulSoup(noscript.decode_contents(), "html.parser")
            for img in inner.find_all("img"):
                src = img.get("src")
                if src and isinstance(src, str):
                    src = _normalise_nhentai_url(src)
                    match_page = re.search(r"/(\d+)(?:t)?\.(?:jpg|png|webp|gif)", src)
                    if match_page:
                        page_num = int(match_page.group(1))
                        pages_map.setdefault(page_num, src)

        # inline <img> fallback
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src or not isinstance(src, str):
                continue
            if "galleries" not in src and "/g/" not in src:
                continue
            src = _normalise_nhentai_url(src)
            match_page = re.search(r"/(\d+)(?:t)?\.(?:jpg|png|webp|gif)", src)
            if match_page:
                page_num = int(match_page.group(1))
                pages_map.setdefault(page_num, src)

        # <script> JSON approach
        for script in soup.find_all("script"):
            text = script.string or ""
            if "media_id" in text and "num_pages" in text:
                try:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    data = json.loads(text[start:end])
                    media_id = data.get("media_id")
                    num_pages = data.get("num_pages", 0)
                    ext = data.get("images", {}).get("cover", {}).get("t", "jpg")
                    if isinstance(ext, str) and ext not in ("j", "p", "g", "w"):
                        pass  # keep as-is
                    else:
                        ext = {1: "png", 2: "jpg", 3: "png", 4: "webp", 5: "gif"}.get(
                            data.get("images", {}).get("cover", {}).get("t", "jpg") if isinstance(data.get("images", {}).get("cover", {}).get("t"), int) else None,
                            "jpg",
                        )
                    if media_id and num_pages:
                        for p in range(1, min(num_pages + 1, max_pages + 1)):
                            pages_map.setdefault(
                                p,
                                f"https://i3.nhentai.net/galleries/{media_id}/{p}.{ext}",
                            )
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    # Strategy 2: nhentai API fallback
    if not pages_map:
        try:
            api_url = f"https://nhentai.net/api/gallery/{gallery_id}"
            async with session.get(api_url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
                data = await resp.json()
            media_id = data.get("media_id")
            num_pages = int(data.get("num_pages", 0))
            images = data.get("images", {})
            pages_data = images.get("pages", [])
            ext_map = {1: "png", 2: "jpg", 3: "png", 4: "webp", 5: "gif"}

            for idx, page_info in enumerate(pages_data, start=1):
                if idx > max_pages:
                    break
                t_val = page_info.get("t", "j")
                ext = ext_map.get(t_val, "jpg") if isinstance(t_val, int) else (t_val if isinstance(t_val, str) else "jpg")
                pages_map.setdefault(idx, f"https://i3.nhentai.net/galleries/{media_id}/{idx}.{ext}")

            if not pages_map and media_id and num_pages:
                ext = "jpg"
                cover_ext = images.get("cover", {}).get("t", "j")
                ext = ext_map.get(cover_ext, "jpg") if isinstance(cover_ext, int) else (cover_ext if isinstance(cover_ext, str) else "jpg")
                for p in range(1, min(num_pages + 1, max_pages + 1)):
                    pages_map.setdefault(p, f"https://i3.nhentai.net/galleries/{media_id}/{p}.{ext}")
        except Exception as exc:
            logger.warning("nhentai API fallback failed for %s: %s", url, exc)

    return [pages_map[p] for p in sorted(pages_map)]


async def _extract_mangadex(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    match = re.search(r"/chapter/([a-f0-9-]+)", url)
    if not match:
        raise ValueError(f"Could not parse MangaDex chapter ID from: {url}")
    chapter_id = match.group(1)

    api_url = f"https://api.mangadex.org/at-home/server/{chapter_id}"
    async with session.get(api_url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"MangaDex API returned {resp.status} for chapter {chapter_id}")
        data = await resp.json()

    base_url = data["baseUrl"]
    chapter_data = data["chapter"]
    hash_val = chapter_data["hash"]

    quality = "data" if "data" in chapter_data else "data-saver"
    page_filenames = chapter_data.get(quality, chapter_data.get("data", chapter_data.get("data-saver", [])))
    page_filenames = page_filenames[:max_pages]

    result: list[str] = []
    for filename in page_filenames:
        result.append(f"{base_url}/{quality}/{hash_val}/{filename}")
    return result


async def _extract_generic(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    try:
        async with session.get(url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
            html = await resp.text()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    soup = BeautifulSoup(html, "html.parser")
    candidates: set[str] = set()

    # 1) og:image meta tag
    for meta in soup.find_all("meta", attrs={"property": "og:image"}):
        content = meta.get("content")
        if content and isinstance(content, str):
            candidates.add(content)

    # 2) img tags with relevant src patterns
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy-src"):
            src = img.get(attr)
            if not src or not isinstance(src, str):
                continue
            lowered = src.lower()
            if any(kw in lowered for kw in ("chapter", "page", "galleries", "scan", "manga")):
                candidates.add(src)

    # 3) broader img scan (filter aggressively)
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src or not isinstance(src, str):
            continue
        if not src.lower().endswith(tuple(IMAGE_EXTENSIONS)):
            continue
        if _is_valid_page_url(src) and is_high_res(src):
            candidates.add(src)

    # deduplicate, filter, sort
    result = sorted(c for c in candidates if _is_valid_page_url(c) and is_high_res(c))
    return result[:max_pages]


async def _extract_ehentai(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    headers = {"User-Agent": USER_AGENT}
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme
    base_url = url.rstrip("/")

    pages_map: dict[int, str] = {}
    seen_pages: set[int] = set()
    pending_gallery: list[str] = [base_url]

    while pending_gallery and len(pages_map) < max_pages:
        gallery_url = pending_gallery.pop(0)

        p_match = re.search(r"[?&]p=(\d+)", gallery_url)
        p_val = int(p_match.group(1)) if p_match else 0
        if p_val in seen_pages:
            continue
        seen_pages.add(p_val)

        try:
            async with session.get(gallery_url, timeout=HTML_TIMEOUT, headers=headers) as resp:
                html = await resp.text()
        except Exception as exc:
            logger.warning("e-hentai gallery page failed: %s: %s", gallery_url, exc)
            continue

        soup = BeautifulSoup(html, "html.parser")

        # collet pagination links from the first page visited
        if gallery_url == base_url:
            for link in soup.select("table.ptt a"):
                href = link.get("href")
                if not href or not isinstance(href, str):
                    continue
                full_href = urljoin(gallery_url, href)
                pp_match = re.search(r"[?&]p=(\d+)", full_href)
                pp_val = int(pp_match.group(1)) if pp_match else 0
                if pp_val not in seen_pages and full_href not in pending_gallery:
                    pending_gallery.append(full_href)

        gdtm_links = soup.select(".gdtm a")
        if gdtm_links:
            for link in gdtm_links:
                if len(pages_map) >= max_pages:
                    break
                href = link.get("href")
                if not href or not isinstance(href, str):
                    continue
                page_url = urljoin(gallery_url, href)

                page_num_match = re.search(r"/(\d+)-(\d+)$", page_url)
                page_num = int(page_num_match.group(2)) if page_num_match else len(pages_map) + 1

                try:
                    async with session.get(
                        page_url, timeout=HTML_TIMEOUT,
                        headers={**headers, "Referer": gallery_url}
                    ) as page_resp:
                        page_html = await page_resp.text()
                except Exception as exc:
                    logger.warning("e-hentai individual page failed: %s: %s", page_url, exc)
                    continue

                page_soup = BeautifulSoup(page_html, "html.parser")
                img_tag = page_soup.select_one("#img")
                if img_tag:
                    src = img_tag.get("src")
                    if src and isinstance(src, str):
                        pages_map[page_num] = src
        else:
            for img in soup.find_all("img"):
                if len(pages_map) >= max_pages:
                    break
                src = img.get("src")
                if not src or not isinstance(src, str):
                    continue
                if not src.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    continue
                if not _is_valid_page_url(src) or not is_high_res(src):
                    continue
                num_match = re.search(r"/(\d+)[^/]*\.\w+$", src)
                if num_match:
                    pages_map.setdefault(int(num_match.group(1)), src)
                else:
                    pages_map.setdefault(len(pages_map) + 1, src)

    return [pages_map[p] for p in sorted(pages_map)][:max_pages]


async def _extract_mangaplus(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    match = re.search(r"/viewer/(\d+)", url)
    if not match:
        raise ValueError(f"Could not parse MangaPlus chapter ID from: {url}")
    chapter_id = match.group(1)

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": url,
        "Origin": "https://mangaplus.shueisha.co.jp",
    }

    result: list[str] = []

    for quality in ("super_high", "high"):
        api_url = (
            f"https://mangaplus.shueisha.co.jp/api/manga_viewer"
            f"?chapter_id={chapter_id}&split=yes&img_quality={quality}"
        )
        try:
            async with session.get(api_url, timeout=HTML_TIMEOUT, headers=headers) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json()
        except Exception as exc:
            logger.warning("MangaPlus API %s failed: %s", quality, exc)
            continue

        try:
            pages = data["success"]["mangaViewer"]["pages"]
            for page in pages[:max_pages]:
                manga_page = page.get("mangaPage", page)
                image_url = manga_page.get("imageUrl")
                if image_url and isinstance(image_url, str):
                    result.append(image_url)
            if result:
                break
        except (KeyError, TypeError) as exc:
            logger.warning("MangaPlus JSON parse error for %s: %s", url, exc)
            continue

    return result[:max_pages]


async def _extract_webtoons(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    try:
        async with session.get(url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
            html = await resp.text()
    except Exception as exc:
        logger.warning("Webtoons fetch failed for %s: %s", url, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")
    result: list[str] = []

    for script in soup.find_all("script"):
        text = script.string or ""
        if "imageList" not in text:
            continue

        match = re.search(r'imageList\s*=\s*(\[.+?\])\s*;', text, re.DOTALL)
        if not match:
            continue

        try:
            image_list = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

        if isinstance(image_list, list):
            for item in image_list[:max_pages]:
                if isinstance(item, str):
                    url_item = item
                elif isinstance(item, dict):
                    url_item = (
                        item.get("url")
                        or item.get("imageUrl")
                        or item.get("src")
                        or ""
                    )
                else:
                    continue
                if url_item and isinstance(url_item, str):
                    url_item = re.sub(r"\?type=q\d+$", "", url_item)
                    url_item = re.sub(r"[?&]w=\d+", "", url_item)
                    url_item = re.sub(r"[?&]h=\d+", "", url_item)
                    result.append(url_item)

        break

    return result[:max_pages]


async def _extract_batoto(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": url,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with session.get(url, timeout=HTML_TIMEOUT, headers=headers) as resp:
            html = await resp.text()
    except Exception as exc:
        logger.warning("Batoto fetch failed for %s: %s", url, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")
    result: list[str] = []

    for img in soup.select("img.page-img"):
        src = img.get("data-src") or img.get("src") or img.get("data-original")
        if src and isinstance(src, str):
            result.append(src)

    if not result:
        for container in soup.select(".page-break"):
            for img in container.find_all("img"):
                src = img.get("data-src") or img.get("src") or img.get("data-original")
                if src and isinstance(src, str) and _is_valid_page_url(src):
                    result.append(src)

    if not result:
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src") or img.get("data-original")
            if not src or not isinstance(src, str):
                continue
            if not _is_valid_page_url(src) or not is_high_res(src):
                continue
            lowered = src.lower()
            if any(kw in lowered for kw in ("chapter", "page", "scan", "manga", "bato")):
                result.append(src)

    return result[:max_pages]


async def _extract_generic_enhanced(url: str, session: aiohttp.ClientSession, max_pages: int) -> list[str]:
    try:
        async with session.get(url, timeout=HTML_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
            html = await resp.text()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    soup = BeautifulSoup(html, "html.parser")
    candidates: set[str] = set()

    SELECTOR_PRIORITY = [
        "img.chapter-img",
        "img.page-img",
        "img.reader-img",
        "#chapter-image",
        "img.manga-page",
        "img[src*='chapter']",
        "img[src*='pages']",
    ]

    for selector in SELECTOR_PRIORITY:
        for img in soup.select(selector):
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-original")
                or img.get("data-lazy-src")
            )
            if not src or not isinstance(src, str):
                continue
            if not _is_valid_page_url(src):
                continue

            w = img.get("width")
            h = img.get("height")
            if w and h:
                try:
                    wi, hi = int(w), int(h)
                    if wi < 200 or hi < 200:
                        continue
                except (ValueError, TypeError):
                    pass

            candidates.add(src)

        if candidates:
            break

    if not candidates:
        for meta in soup.find_all("meta", attrs={"property": "og:image"}):
            content = meta.get("content")
            if content and isinstance(content, str):
                candidates.add(content)

        for img in soup.find_all("img"):
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-original")
                or img.get("data-lazy-src")
            )
            if not src or not isinstance(src, str):
                continue
            if not src.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                continue

            w = img.get("width")
            h = img.get("height")
            if w and h:
                try:
                    wi, hi = int(w), int(h)
                    if wi < 200 or hi < 200:
                        continue
                except (ValueError, TypeError):
                    pass

            if _is_valid_page_url(src) and is_high_res(src):
                candidates.add(src)

    result = sorted(c for c in candidates if _is_valid_page_url(c) and is_high_res(c))
    return result[:max_pages]


# ---------------------------------------------------------------------------
# main downloader class
# ---------------------------------------------------------------------------

class MangaDownloader:
    """Async manga page-image downloader with site auto-detection."""

    def __init__(
        self,
        *,
        user_agent: str = USER_AGENT,
        request_delay: float = REQUEST_DELAY,
        html_timeout: int = 30,
        image_timeout: int = 60,
        max_image_size: int = MAX_IMAGE_SIZE,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._user_agent = user_agent
        self._request_delay = request_delay
        self._html_timeout = aiohttp.ClientTimeout(total=html_timeout)
        self._image_timeout = aiohttp.ClientTimeout(total=image_timeout)
        self._max_image_size = max_image_size
        self._own_session = session is None
        self._session = session
        self._last_request_time = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self._user_agent},
                timeout=self._html_timeout,
            )
        return self._session

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._request_delay:
            await asyncio.sleep(self._request_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def extract_urls(self, url: str, max_pages: int = 50) -> list[str]:
        """Return deduplicated, sorted list of full-resolution image URLs.

        Parameters
        ----------
        url : str
            The manga gallery / chapter / page URL.
        max_pages : int
            Maximum number of image URLs to return (default 50).
        """
        site = detect_site(url)
        session = await self._get_session()
        await self._rate_limit()

        if site == Site.NHENTAI:
            result = await _extract_nhentai(url, session, max_pages)
        elif site == Site.MANGADEX:
            result = await _extract_mangadex(url, session, max_pages)
        elif site == Site.DIRECT:
            result = [url]
        elif site == Site.EHENTAI:
            result = await _extract_ehentai(url, session, max_pages)
        elif site == Site.MANGAPLUS:
            result = await _extract_mangaplus(url, session, max_pages)
        elif site == Site.WEBTOONS:
            result = await _extract_webtoons(url, session, max_pages)
        elif site == Site.BATOTO:
            result = await _extract_batoto(url, session, max_pages)
        elif site == Site.GENERIC:
            result = await _extract_generic_enhanced(url, session, max_pages)
        else:
            raise ValueError(f"Unknown site type for URL: {url}")

        logger.info("extract_urls(%s) → %d image URLs", url, len(result))
        return result

    async def download_batch(
        self,
        urls: list[str],
        output_dir: str,
        start_index: int = 1,
    ) -> list[str]:
        """Download a batch of image URLs to *output_dir*, returning
        the list of local file paths (in order).

        Files are named ``{index:04d}.{ext}`` with *index* starting at
        *start_index*.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        session = await self._get_session()
        saved: list[str] = []

        for idx, image_url in enumerate(urls, start=start_index):
            await self._rate_limit()

            ext = Path(urlparse(image_url).path).suffix
            if not ext or ext.lower() not in IMAGE_EXTENSIONS:
                ext = ".jpg"

            filepath = out / f"{idx:04d}{ext}"
            try:
                await self._download_single(session, image_url, filepath)
                saved.append(str(filepath))
                logger.debug("downloaded %s → %s", image_url, filepath)
            except Exception as exc:
                logger.warning("failed to download %s: %s", image_url, exc)

        return saved

    async def _download_single(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        filepath: Path,
    ) -> None:
        async with session.get(
            image_url,
            timeout=self._image_timeout,
            headers={
                "User-Agent": self._user_agent,
                "Referer": image_url,
            },
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {image_url}")

            content_type = resp.headers.get("Content-Type", "")
            content_length = resp.headers.get("Content-Length")

            if content_length and int(content_length) > self._max_image_size:
                raise RuntimeError(
                    f"Image too large ({int(content_length)} bytes, "
                    f"max {self._max_image_size}) at {image_url}"
                )

            data = await resp.read()

            if len(data) > self._max_image_size:
                raise RuntimeError(
                    f"Image exceeds max size ({len(data)} bytes) at {image_url}"
                )

            if content_type and "image" not in content_type.lower():
                logger.warning("non-image content-type %s for %s", content_type, image_url)

        filepath.write_bytes(data)

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MangaDownloader:
        await self._get_session()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._own_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
