"""URL resolution, thumbnail-to-full-resolution conversion, helpers."""
import re
import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp", ".tiff"}
_UI_WORDS = {
    "logo", "icon", "avatar", "banner", "ad", "ads", "px", "spacer",
    "pixel", "favicon", "header", "footer", "nav", "button", "btn",
    "rating", "star", "flag", "emoji", "badge", "social", "share",
    "search", "menu", "sidebar", "widget", "tracking", "analytics",
}

def is_image_url(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    return any(url_lower.endswith(ext) or ext + "?" in url_lower for ext in _IMG_EXT)

def is_likely_page(url: str) -> bool:
    """Check if URL looks like a manga page (not UI element)."""
    if not is_image_url(url):
        return False
    stem = urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    for word in _UI_WORDS:
        if word in stem:
            return False
    if stem in ("", "0", "1", "2", "3"):
        return False
    return True

def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")

# ─── Max Resolution ─────────────────────────────────────────────────

_THUMB_STRIP_RE = re.compile(r"[-_](thumb|small|tn|thumbnail|prev|preview|medium|mini|low|md|sm)(?:[-_.]|$)", re.I)
_THUMB_TRAILING_T = re.compile(r"t$", re.I)
_DIM_SUFFIX_RE = re.compile(r"[-_]?\d+x\d+$")
_THUMB_PREFIX_RE = re.compile(r"^(thumb|small|tn|thumbnail|preview)[-_]", re.I)
_W_H_PARAM_RE = re.compile(r"([?&])(?:w|width|h|height|size|quality|q)=\d+", re.I)

_GALLERY_SITES = {
    "nhentai": {
        "hosts": {"nhentai.to", "nhentai.net", "nhentai.xxx"},
        "cdn_patterns": ["zrocdn.xyz", "i3.nhentai.net", "t3.nhentai.net"],
    },
}

def _strip_thumb_indicators(name: str) -> str:
    cleaned = name
    cleaned = _THUMB_PREFIX_RE.sub("", cleaned)
    cleaned = _THUMB_STRIP_RE.sub("", cleaned)
    cleaned = _DIM_SUFFIX_RE.sub("", cleaned)
    # Strip double separators, trailing separators
    cleaned = re.sub(r"[-_]{2,}", "-", cleaned)
    cleaned = cleaned.strip("-_")
    if len(cleaned) >= 2 and cleaned.endswith("t") and cleaned[:-1].lstrip("-_").isdigit():
        cleaned = cleaned[:-1]
    if cleaned.endswith("t") and len(cleaned) <= 3:
        cleaned = cleaned[:-1]
    return cleaned or name

def max_resolve(url: str) -> str:
    """Convert thumbnail/preview URL to maximum-resolution URL."""
    parsed = urlparse(url)
    path = parsed.path
    dir_part = path.rsplit("/", 1)[0] if "/" in path else ""
    filename = path.rsplit("/", 1)[-1] if "/" in path else path

    if "." not in filename:
        return url

    base, rest = filename.rsplit(".", 1)
    ext = rest.split("?")[0].split("#")[0]
    cleaned = _strip_thumb_indicators(base)

    if cleaned == base:
        result = url
    else:
        new_filename = f"{cleaned}.{ext}"
        new_path = f"{dir_part}/{new_filename}" if dir_part else new_filename
        result = urlunparse(parsed._replace(path=new_path))

    # Normalize known CDN URLs that block automated downloads
    result = _normalize_cdn(result)
    return result


def _normalize_cdn(url: str) -> str:
    """Rewrite CDN URLs to use accessible endpoints."""
    import re
    # nhentai: t[1-9].nhentai.net → i.nhentai.net (unblocked CDN)
    url = re.sub(r"https?://t\d+\.nhentai\.net/", "https://i.nhentai.net/", url)
    url = re.sub(r"https?://(\d+)\.nhentai\.net/", r"https://i.nhentai.net/", url)
    return url

def chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
