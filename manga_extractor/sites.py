"""Site-specific manga scrapers with a common interface."""
import re
import logging
import asyncio
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from .utils import is_image_url, is_likely_page, max_resolve, extract_domain

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}

IMG_ATTRS = [
    "data-src", "data-original", "data-lazy-src",
    "data-full", "data-hi-res", "data-hires",
    "data-large", "data-zoom", "src",
]

# ─── CSS selectors for popular manga reader themes ─────────────────

_CHAPTER_SELECTORS = [
    ".container-chapter-reader img",
    ".reading-content img",
    ".page-break img",
    ".manga-page img",
    "div[class*='chapter'] img",
    "div[class*='reader'] img",
    "div[class*='page'] img",
    "noscript img",
    "img[class*='page']",
    "img[class*='manga']",
    "img[class*='chapter']",
    "#readerarea img",
    ".entry-content img",
    ".comic-page img",
    ".ts-main-image img",
    ".iv-card img",
    "div.text-center img",
    "div.img-container img",
    ".img-responsive[class*='chapter']",
    ".vung-doc img",
    "div#vungdoc img",
]

_SITE_SPECIFIC = {
    "manganato.com": [".container-chapter-reader img"],
    "readmanganato.com": [".container-chapter-reader img"],
    "mangakakalot.com": [".container-chapter-reader img", "div.vung-doc img"],
    "chapmanganato.com": [".container-chapter-reader img"],
    "mangabat.com": ["div#vungdoc img", ".container img"],
    "mangahere.com": [".read-content img", ".chapter-container img"],
    "asura.gg": ["#readerarea img", ".ts-main-image img"],
    "asurascans.com": ["#readerarea img"],
    "flamescans.org": ["#readerarea img"],
    "reaperscans.com": ["#readerarea img", ".reading-content img"],
    "void-scans.com": ["#readerarea img"],
    "luminousscans.com": ["#readerarea img"],
    "mangareader.to": ["#reader img", ".reading-content img"],
    "mangaplus.shueisha.co.jp": ["div[class*='page'] img"],
    "comick.io": [".comic-page img"],
    "bato.to": [".page-content img"],
    "mangafire.to": [".reader-area img", "#reader img"],
    "nhentai.to": ["#thumbnail-container img", "div.gallery img", ".thumb-container img"],
    "nhentai.net": ["#thumbnail-container img", ".thumb-container img"],
    "nhentai.xxx": ["#thumbnail-container img", ".thumb-container img"],
}


# ─── Base ───────────────────────────────────────────────────────────

class BaseSite(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, url: str) -> bool: ...

    @abstractmethod
    async def extract_pages(self, url: str, client: httpx.AsyncClient) -> list[str]: ...

    async def extract_chapters(self, url: str, client: httpx.AsyncClient) -> list[dict]:
        return []


# ─── MangaDex ───────────────────────────────────────────────────────

_MD_API = "https://api.mangadex.org"
_MD_ID_RE = re.compile(r"/title/([a-f0-9-]{36})|/chapter/([a-f0-9-]{36})")


class MangaDexSite(BaseSite):
    name = "mangadex"

    def can_handle(self, url: str) -> bool:
        return "mangadex." in urlparse(url).netloc.lower()

    async def extract_pages(self, url: str, client: httpx.AsyncClient) -> list[str]:
        cid = self._extract_chapter_id(url)
        if cid:
            r = await client.get(f"{_MD_API}/at-home/server/{cid}")
            r.raise_for_status()
            d = r.json()
            base = d["baseUrl"]
            h = d["chapter"]["hash"]
            return [f"{base}/data/{h}/{p}" for p in d["chapter"]["data"]]
        mid = self._extract_manga_id(url)
        if mid:
            chapters = await self._fetch_chapters(mid, client)
            r = await client.get(f"{_MD_API}/at-home/server/{chapters[0]['id']}")
            r.raise_for_status()
            d = r.json()
            base = d["baseUrl"]
            h = d["chapter"]["hash"]
            return [f"{base}/data/{h}/{p}" for p in d["chapter"]["data"]]
        return []

    async def extract_chapters(self, url: str, client: httpx.AsyncClient) -> list[dict]:
        mid = self._extract_manga_id(url)
        if mid:
            return await self._fetch_chapters(mid, client)
        return []

    async def _fetch_chapters(self, mid: str, client: httpx.AsyncClient) -> list[dict]:
        chapters = []
        offset = 0
        while True:
            r = await client.get(f"{_MD_API}/manga/{mid}/feed", params={
                "translatedLanguage[]": "en", "order[chapter]": "asc",
                "limit": 100, "offset": offset,
                "contentRating[]": ["safe", "suggestive", "erotica"],
            })
            r.raise_for_status()
            items = r.json().get("data", [])
            if not items:
                break
            for it in items:
                a = it["attributes"]
                chapters.append({
                    "id": it["id"], "chapter": a.get("chapter", "0"),
                    "title": a.get("title", "") or "",
                })
            offset += 100
        return chapters

    async def search(self, title: str, client: httpx.AsyncClient) -> list[dict]:
        r = await client.get(f"{_MD_API}/manga", params={"title": title, "limit": 10})
        r.raise_for_status()
        results = []
        for m in r.json().get("data", []):
            t = m["attributes"]["title"]
            display = t.get("en") or next(iter(t.values()), "?")
            results.append({"id": m["id"], "title": display, "source": "mangadex"})
        return results

    def _extract_manga_id(self, url: str) -> str | None:
        m = re.search(r"/title/([a-f0-9-]{36})", url)
        return m.group(1) if m else None

    def _extract_chapter_id(self, url: str) -> str | None:
        m = re.search(r"/chapter/([a-f0-9-]{36})", url)
        return m.group(1) if m else None


# ─── Generic HTML Scraper ──────────────────────────────────────────

class GenericSite(BaseSite):
    name = "generic"

    def can_handle(self, url: str) -> bool:
        return True

    async def _fetch_html(self, url: str, client: httpx.AsyncClient) -> str | None:
        for attempt in range(2):
            try:
                r = await client.get(url, follow_redirects=True)
                if r.status_code in (403, 503, 429):
                    logger.warning(f"httpx blocked ({r.status_code}) on {url}, trying curl_cffi")
                    break
                return r.text
            except Exception as e:
                logger.warning(f"httpx failed: {e}")
                break
        # Fallback to curl_cffi (browser TLS fingerprint)
        try:
            from curl_cffi import requests as curl_req
            r = curl_req.get(url, impersonate="chrome110", timeout=30)
            if r.status_code == 200:
                return r.text
            logger.warning(f"curl_cffi returned {r.status_code} for {url}")
        except Exception as e:
            logger.error(f"curl_cffi also failed: {e}")
        return None

    def _extract_images(self, html: str, domain: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        selectors = _SITE_SPECIFIC.get(domain, _CHAPTER_SELECTORS)
        seen = set()
        urls = []
        for sel in selectors:
            for img in soup.select(sel):
                for attr in IMG_ATTRS:
                    v = img.get(attr)
                    if v and v.startswith("http") and v not in seen:
                        seen.add(v)
                        urls.append(v)
            if len(urls) >= 2:
                break
        if not urls:
            for img in soup.find_all("img"):
                for attr in IMG_ATTRS:
                    v = img.get(attr)
                    if v and v.startswith("http") and is_image_url(v) and v not in seen:
                        seen.add(v)
                        urls.append(v)
        return [u for u in urls if is_likely_page(u)]

    async def extract_pages(self, url: str, client: httpx.AsyncClient) -> list[str]:
        domain = extract_domain(url)
        html = await self._fetch_html(url, client)
        if not html:
            return []
        return self._extract_images(html, domain)

    async def extract_chapters(self, url: str, client: httpx.AsyncClient) -> list[dict]:
        return []


# ─── Site Registry ─────────────────────────────────────────────────

_SITES: list[BaseSite] = [
    MangaDexSite(),
    GenericSite(),
]


def get_handler(url: str) -> BaseSite | None:
    for site in _SITES:
        if site.can_handle(url):
            return site
    return None


def get_mangadex() -> MangaDexSite:
    return _SITES[0]  # type: ignore
