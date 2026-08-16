use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{REFERER, USER_AGENT};
use scraper::{Html, Selector};
use std::collections::HashSet;
use std::io::Read;
use std::time::Duration;
use url::Url;

/// Browser-like UA used for all site fetches. Also used by the
/// Cloudflare-safe fetch fallback (curl) in the ingest lib.
pub const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

#[derive(Debug)]
pub enum MangaSite {
    Nhentai,
    Mangadex,
    Manganato,
    Ehentai,
    Imhentai,
    AllPornComic,
    IlikeComix,
    Generic,
}

impl MangaSite {
    pub fn detect(url: &Url) -> Self {
        let host = url.host_str().unwrap_or("").to_ascii_lowercase();
        if host.contains("nhentai.net") || host.contains("nhentai.to") {
            Self::Nhentai
        } else if host.contains("mangadex.org") {
            Self::Mangadex
        } else if host.contains("manganato.com")
            || host.contains("mangakakalot.com")
            || host.contains("chapmanganato.")
            || host.contains("readmanganato.")
        {
            Self::Manganato
        } else if host.contains("e-hentai.org") || host.contains("exhentai.org") {
            Self::Ehentai
        } else if host.contains("imhentai.xxx") || host.contains("imhentai.com") {
            Self::Imhentai
        } else if host.contains("allporncomic.com") || host.contains("allporncomic.net") {
            Self::AllPornComic
        } else if host.contains("ilikecomix.com") {
            Self::IlikeComix
        } else {
            Self::Generic
        }
    }
}

pub fn extract_manga_urls(
    client: &Client,
    url: &Url,
    html: &str,
    base: &Url,
    max_pages: usize,
) -> Result<Vec<String>> {
    match MangaSite::detect(url) {
        MangaSite::Nhentai => extract_nhentai(client, url, html, base, max_pages),
        MangaSite::Mangadex => extract_mangadex(client, url, max_pages),
        MangaSite::Manganato => extract_manganato(html, base, max_pages),
        MangaSite::Ehentai => extract_ehentai(client, url, html, base, max_pages),
        MangaSite::Imhentai => extract_imhentai(html, base, max_pages),
        MangaSite::AllPornComic => extract_madara(client, url, html, base, max_pages),
        MangaSite::IlikeComix => extract_madara(client, url, html, base, max_pages),
        MangaSite::Generic => extract_generic(html, base, max_pages),
    }
}

// ── nhentai.net ──────────────────────────────────────────────────────────
// Gallery: https://nhentai.net/g/{id}/
// Images:  https://i.nhentai.net/galleries/{media_id}/{page}.jpg
// Thumbs:  https://t.nhentai.net/galleries/{media_id}/{page}t.jpg
fn extract_nhentai(
    _client: &Client,
    _url: &Url,
    html: &str,
    base: &Url,
    max_pages: usize,
) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let mut urls: Vec<String> = Vec::new();
    let mut seen = HashSet::new();

    // All candidates are normalized (thumbnail -> full size, double
    // extensions collapsed) and deduplicated as they arrive.
    let mut push = |candidate: &str| {
        let absolute = match Url::parse(candidate) {
            Ok(_) => candidate.to_string(),
            Err(_) => base
                .join(candidate)
                .map(|u| u.to_string())
                .unwrap_or_else(|_| candidate.to_string()),
        };
        let normalized = normalize_nhentai_url(&absolute);
        if seen.insert(normalized.clone()) {
            urls.push(normalized);
        }
    };

    // Method 1: <noscript> images (already absolute on the site)
    let noscript_sel = Selector::parse("noscript img").unwrap();
    for img in document.select(&noscript_sel) {
        if let Some(src) = img.value().attr("src") {
            push(src);
        }
    }

    // Method 2: lazy-loaded images (data-src / src).
    // NOTE: not gated on method 1 being empty: a partial noscript set must
    // not suppress the other methods (that caused silent page loss).
    let img_sel = Selector::parse(".thumb-container img").unwrap();
    for img in document.select(&img_sel) {
        for attr in &["data-src", "src"] {
            if let Some(src) = img.value().attr(attr) {
                push(src);
            }
        }
    }

    // Method 3: gallery JSON embedded in the page body (complete page list).
    if let Some(gid) = extract_nhentai_gallery_id(base) {
        if let Some(parsed) = parse_nhentai_embedded_json(html, &gid) {
            for img in parsed {
                push(&img);
            }
        }
    }

    // Method 4: gallery API as a completeness fallback. The official API is
    // now gated ("Use new API"), but the legacy endpoint still answers for
    // some galleries; merging it can only add missing pages.
    if let Some(gid) = extract_nhentai_gallery_id(base) {
        let api_url = format!("https://nhentai.net/api/gallery/{}", gid);
        if let Ok(resp) = _client
            .get(&api_url)
            .header(USER_AGENT, UA)
            .timeout(Duration::from_secs(15))
            .send()
        {
            let mut reader = resp.take(16 * 1024 * 1024);
            let mut body = Vec::new();
            if reader.read_to_end(&mut body).is_ok() {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) {
                    if let (Some(mid), Some(images)) = (
                        json["media_id"].as_str().map(|s| s.to_string()),
                        json["images"]["pages"].as_array(),
                    ) {
                        for page in images.iter().take(max_pages) {
                            let ext = match page["t"].as_str().unwrap_or("j") {
                                "p" => "png",
                                "w" => "webp",
                                _ => "jpg",
                            };
                            let num = page["num"].as_u64().unwrap_or(1);
                            let img =
                                format!("https://i.nhentai.net/galleries/{}/{}.{}", mid, num, ext);
                            push(&img);
                        }
                    }
                }
            }
        }
    }

    // Defensive: drop any remaining thumbnail markers, then order pages
    // numerically (methods return them in different orders / sets).
    urls.retain(|u| !is_nhentai_thumbnail(u));
    urls.sort_by_key(|u| page_number_of(u).unwrap_or(u64::MAX));
    urls.truncate(max_pages);
    Ok(urls)
}

/// True if the URL still names a thumbnail (`{n}t.{ext}`) after normalization.
fn is_nhentai_thumbnail(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };
    let Some(file) = parsed.path().rsplit('/').next() else {
        return false;
    };
    let Some(stem) = file.split('.').next() else {
        return false;
    };
    stem.strip_suffix('t')
        .is_some_and(|num| !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()))
}

/// Extract the 1-based page number from a normalized nhentai image URL.
fn page_number_of(url: &str) -> Option<u64> {
    let path = Url::parse(url).ok()?.path().to_string();
    let file = path.rsplit('/').next()?;
    let stem = file.split('.').next()?;
    if stem.chars().all(|c| c.is_ascii_digit()) {
        stem.parse().ok()
    } else {
        None
    }
}

/// Normalize a nhentai image URL to the full-size page variant.
///
/// The site's thumbnail markup is messy: hosts are `t[1-9]?.nhentai.net`
/// instead of `i...`, and the current HTML emits filenames like
/// `3t.webp.webp` (thumbnail marker + double extension). This collapses
/// both: `3t.webp.webp` -> `3.webp`, `1t.jpg` -> `1.jpg`, and plain
/// `2.webp` is left untouched.
fn normalize_nhentai_url(src: &str) -> String {
    let trimmed = src.trim();
    let mut parsed = match Url::parse(trimmed) {
        Ok(p) => p,
        Err(_) => {
            // Protocol-relative (`//t1.nhentai.net/...`) or bare path.
            let absolute = if trimmed.starts_with("//") {
                format!("https:{trimmed}")
            } else {
                trimmed.to_string()
            };
            let Ok(p) = Url::parse(&absolute) else {
                return trimmed.to_string();
            };
            p
        }
    };

    // t[1-9]? .nhentai.net -> i[1-9]? .nhentai.net (thumb -> full host).
    if let Some(host) = parsed.host_str().map(str::to_string) {
        let lower = host.to_ascii_lowercase();
        for domain in [".nhentai.net", ".nhentai.to"] {
            if let Some(suffix) = lower
                .strip_prefix('t')
                .and_then(|value| value.strip_suffix(domain))
            {
                let _ = parsed.set_host(Some(&format!("i{suffix}{domain}")));
                break;
            }
        }
    }

    // Collapse thumbnail marker + repeated extensions in the filename.
    let cleaned = cleanup_nhentai_filename(parsed.path());
    parsed.set_path(&cleaned);
    parsed.to_string()
}

/// `3t.webp.webp` -> `3.webp`; `1t.jpg` -> `1.jpg`; `4.webp` stays `4.webp`.
fn cleanup_nhentai_filename(path: &str) -> String {
    let (dir, file) = match path.rsplit_once('/') {
        Some((d, f)) => (d, f),
        None => ("", path),
    };

    // Strip repeated known extensions, keeping the last one.
    let mut ext: Option<&str> = None;
    let mut stem = file;
    while let Some((s, e)) = stem.rsplit_once('.') {
        if matches!(
            e.to_ascii_lowercase().as_str(),
            "webp" | "jpg" | "jpeg" | "png" | "gif"
        ) {
            ext = Some(e);
            stem = s;
        } else {
            break;
        }
    }

    // Thumbnail marker: {n}t -> {n}.
    if let Some(num) = stem.strip_suffix('t') {
        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
            stem = num;
        }
    }

    match ext {
        Some(e) => {
            let name = format!("{stem}.{e}");
            if dir.is_empty() {
                name
            } else {
                format!("{dir}/{name}")
            }
        }
        None => path.to_string(),
    }
}

fn extract_nhentai_gallery_id(url: &Url) -> Option<String> {
    let path = url.path();
    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();
    for seg in segs {
        if seg.chars().all(|c| c.is_ascii_digit()) && seg.len() >= 3 {
            return Some(seg.to_string());
        }
    }
    None
}

/// Parse the gallery JSON that nhentai embeds as an escaped string inside
/// `"body":"{...}"`. The main gallery's media_id and page paths are used to
/// build full image URLs; related-gallery entries are ignored.
fn parse_nhentai_embedded_json(html: &str, gid: &str) -> Option<Vec<String>> {
    // Find the media_id that belongs to THIS gallery. The embedded JSON has
    // the pattern `"id":<gid>,"media_id":"<mid>"`. We scan for the gid first,
    // then capture the following media_id.
    let gid_needle = format!("\"id\":{},", gid);
    let gid_pos = html.find(&gid_needle)?;
    let after_gid = &html[gid_pos..];
    let mid_needle = "\"media_id\":\"";
    let mid_pos = after_gid.find(mid_needle)? + mid_needle.len();
    let media_id: String = after_gid[mid_pos..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if media_id.is_empty() {
        return None;
    }

    // Collect page paths scoped to THIS media_id: galleries/{media_id}/{n}.{ext}
    let mut urls = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let path_prefix = format!("galleries/{}/", media_id);
    let mut search_from = 0usize;
    while let Some(rel) = html[search_from..].find(&path_prefix) {
        let start = search_from + rel + path_prefix.len();
        let rest = &html[start..];
        let mut end = 0usize;
        for (i, c) in rest.char_indices() {
            if !c.is_ascii_alphanumeric() && c != '.' {
                end = i;
                break;
            }
            end = i + c.len_utf8();
        }
        let file = &rest[..end];
        // Skip thumbnail paths ({n}t.{ext}) — they 404 on the full-size
        // host and are mixed into the embedded JSON alongside the real
        // pages. Full paths like `3.webp` are kept.
        let file_stem = file.split('.').next().unwrap_or(file);
        if let Some(num) = file_stem.strip_suffix('t') {
            if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
                search_from = start + end;
                continue;
            }
        }
        if let Some(ext) = file.rsplit('.').next() {
            if matches!(ext, "jpg" | "jpeg" | "png" | "webp" | "gif") {
                let img = format!("https://i.nhentai.net/galleries/{}/{}", media_id, file);
                if seen.insert(img.clone()) {
                    urls.push(img);
                }
            }
        }
        search_from = start + end;
    }
    if urls.len() < 2 {
        return None;
    }
    let _ = gid;
    Some(urls)
}

// ── mangadex.org ─────────────────────────────────────────────────────────
fn extract_mangadex(client: &Client, url: &Url, max_pages: usize) -> Result<Vec<String>> {
    let chapter_id = url
        .path_segments()
        .and_then(|mut segments| segments.rfind(|part| !part.is_empty()))
        .unwrap_or("")
        .to_string();

    if chapter_id.is_empty() || !chapter_id.chars().all(|c| c.is_alphanumeric() || c == '-') {
        anyhow::bail!("invalid mangadex chapter URL: {}", url);
    }

    let at_home = format!("https://api.mangadex.org/at-home/server/{}", chapter_id);
    let resp = client
        .get(&at_home)
        .header(USER_AGENT, UA)
        .timeout(Duration::from_secs(15))
        .send()
        .context("mangadex API request failed")?
        .error_for_status()
        .context("mangadex API returned error")?;

    let body: serde_json::Value = resp.json().context("mangadex API JSON parse failed")?;

    let base_url = body["baseUrl"]
        .as_str()
        .context("mangadex missing baseUrl")?;
    let hash = body["chapter"]["hash"]
        .as_str()
        .context("mangadex missing chapter hash")?;

    let pages = body["chapter"]["data"]
        .as_array()
        .or_else(|| body["chapter"]["dataSaver"].as_array())
        .context("mangadex missing page data")?;

    let quality = if body["chapter"]["data"].as_array().is_some() {
        "data"
    } else {
        "data-saver"
    };

    let urls: Vec<String> = pages
        .iter()
        .take(max_pages)
        .filter_map(|p| p.as_str())
        .map(|page| format!("{}/{}/{}/{}", base_url, quality, hash, page))
        .collect();

    if urls.is_empty() {
        // Fallback: data-saver
        if let Some(saver) = body["chapter"]["dataSaver"].as_array() {
            return Ok(saver
                .iter()
                .take(max_pages)
                .filter_map(|p| p.as_str())
                .map(|page| format!("{}/data-saver/{}/{}", base_url, hash, page))
                .collect());
        }
        anyhow::bail!("no pages found in mangadex chapter");
    }

    Ok(urls)
}

// ── content tags (for content-aware style selection) ──────────────────────
/// Extract genre/tag keywords from the source so the bot can pick a color
/// style that fits the manga's nature (dark/horror -> dark styles,
/// romance -> pastels, fantasy -> vibrant, ...). Best-effort: any failure
/// yields an empty list; the caller falls back to rotation.
pub fn extract_manga_tags(client: &Client, url: &Url, html: &str) -> Vec<String> {
    match MangaSite::detect(url) {
        MangaSite::Nhentai => parse_nhentai_tags(html, url),
        MangaSite::Mangadex => extract_mangadex_tags(client, url),
        _ => Vec::new(),
    }
}

/// nhentai: the gallery JSON is embedded ESCAPED inside
/// `"body":"{...}"` (i.e. `\"id\":<gid>,\"tags\":[...]`), so the
/// needles below carry the literal backslash-quote.
fn parse_nhentai_tags(html: &str, url: &Url) -> Vec<String> {
    let Some(gid) = extract_nhentai_gallery_id(url) else {
        return Vec::new();
    };
    // The embedded gallery JSON is escaped once inside `"body":"{...}"`,
    // so keys read `\"id\":<gid>,` (literal backslash-quote).
    let bs = r"\";
    let gid_needle = format!("{bs}\"id{bs}\":{},", gid);
    let Some(gid_pos) = html.find(&gid_needle) else {
        return Vec::new();
    };
    let tags_needle = format!("{bs}\"tags{bs}\":[");
    let Some(tags_pos) = html[gid_pos..].find(&tags_needle) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut search_from = gid_pos + tags_pos + tags_needle.len();
    let bound = gid_pos + tags_pos + 96 * 1024;
    let name_needle = format!("{bs}\"name{bs}\":{bs}\"");
    while search_from < bound && search_from < html.len() {
        let rest = &html[search_from..];
        let Some(rel) = rest.find(&name_needle) else {
            break;
        };
        let start = search_from + rel + name_needle.len();
        let name: String = html[start..]
            .chars()
            .take_while(|c| *c != '"' && *c != '\\')
            .collect();
        if !name.is_empty() {
            out.push(name.to_ascii_lowercase());
        }
        search_from = start + name.len();
        if out.len() >= 64 {
            break;
        }
    }
    out
}

/// MangaDex: chapter API -> manga id -> manga tags API. Two best-effort
/// GETs, same client/UA pattern as the page extractor.
fn extract_mangadex_tags(client: &Client, url: &Url) -> Vec<String> {
    let chapter_id = url
        .path_segments()
        .and_then(|mut segments| segments.rfind(|part| !part.is_empty()))
        .unwrap_or("")
        .to_string();
    if chapter_id.is_empty() || !chapter_id.chars().all(|c| c.is_alphanumeric() || c == '-') {
        return Vec::new();
    }

    let Ok(resp) = client
        .get(format!(
            "https://api.mangadex.org/chapter/{chapter_id}?includes[]=manga"
        ))
        .header(USER_AGENT, UA)
        .timeout(Duration::from_secs(15))
        .send()
    else {
        return Vec::new();
    };
    let Ok(body) = resp.json::<serde_json::Value>() else {
        return Vec::new();
    };
    let manga_id = body["data"]["relationships"].as_array().and_then(|rels| {
        rels.iter().find_map(|rel| {
            if rel["type"] == "manga" {
                rel["id"].as_str().map(str::to_string)
            } else {
                None
            }
        })
    });
    let Some(manga_id) = manga_id else {
        return Vec::new();
    };

    let Ok(resp) = client
        .get(format!(
            "https://api.mangadex.org/manga/{manga_id}?includes[]=genre"
        ))
        .header(USER_AGENT, UA)
        .timeout(Duration::from_secs(15))
        .send()
    else {
        return Vec::new();
    };
    let Ok(body) = resp.json::<serde_json::Value>() else {
        return Vec::new();
    };
    body["data"]["attributes"]["tags"]
        .as_array()
        .map(|tags| {
            tags.iter()
                .filter_map(|tag| tag["attributes"]["name"]["en"].as_str())
                .map(|name| name.to_ascii_lowercase())
                .collect::<Vec<String>>()
        })
        .unwrap_or_default()
}

// ── manganato / mangakakalot ─────────────────────────────────────────────
fn extract_manganato(html: &str, base: &Url, max_pages: usize) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    // Manganato uses .container-chapter-reader img
    for selector_text in &[
        ".container-chapter-reader img",
        ".chapter-content img",
        "#vungdoc img",
        ".reading-content img",
        "img[onerror]",
    ] {
        if let Ok(sel) = Selector::parse(selector_text) {
            for img in document.select(&sel) {
                for attr in &["src", "data-src", "data-lazy-src", "data-original"] {
                    if let Some(src) = img.value().attr(attr) {
                        let src = src.trim();
                        if src.is_empty()
                            || src.contains("avatar")
                            || src.contains("logo")
                            || src.contains("loading")
                        {
                            continue;
                        }
                        if let Ok(abs) = base.join(src) {
                            let s = abs.to_string();
                            if (s.ends_with(".jpg") || s.ends_with(".png") || s.ends_with(".webp"))
                                && seen.insert(s.clone())
                            {
                                urls.push(s);
                            }
                        }
                    }
                }
            }
        }
    }

    sort_by_page_hint_if_complete(&mut urls);
    Ok(urls.into_iter().take(max_pages).collect())
}

// ── e-hentai / exhentai ──────────────────────────────────────────────────
fn extract_ehentai(
    client: &Client,
    url: &Url,
    html: &str,
    base: &Url,
    max_pages: usize,
) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    // Method 1: Standard gallery page
    let gdt_sel = Selector::parse(".gdtm a, .gdtl a").unwrap();
    let mut page_links: Vec<String> = Vec::new();

    for a in document.select(&gdt_sel) {
        if let Some(href) = a.value().attr("href") {
            if let Ok(abs) = base.join(href) {
                page_links.push(abs.to_string());
            }
        }
    }

    if page_links.is_empty() {
        // Method 2: Try .gdtm img data-src directly for thumbnail expansion
        let thumb_sel = Selector::parse(".gdtm img, .gdtl img").unwrap();
        for img in document.select(&thumb_sel) {
            if let Some(src) = img
                .value()
                .attr("data-src")
                .or_else(|| img.value().attr("src"))
            {
                if seen.insert(src.to_string()) {
                    if let Ok(abs) = base.join(src) {
                        urls.push(abs.to_string());
                    }
                }
            }
        }
    }

    // For each page link, fetch the individual page to get full image
    for page_url in page_links.iter().take(max_pages) {
        if let Ok(page_resp) = client
            .get(page_url)
            .header(USER_AGENT, UA)
            .header(REFERER, url.as_str())
            .timeout(Duration::from_secs(10))
            .send()
        {
            // Cap the body: an unbounded .text() lets a server stream
            // gigabytes into RAM for every sub-page fetched.
            let mut reader = page_resp.take(16 * 1024 * 1024);
            let mut buffer = Vec::new();
            let page_html = match reader.read_to_end(&mut buffer) {
                Ok(_) => String::from_utf8_lossy(&buffer).into_owned(),
                Err(_) => continue,
            };
            if !page_html.is_empty() {
                let page_doc = Html::parse_document(&page_html);
                let img_sel = Selector::parse("#img").unwrap();
                if let Some(img) = page_doc.select(&img_sel).next() {
                    if let Some(src) = img.value().attr("src") {
                        if seen.insert(src.to_string()) {
                            urls.push(src.to_string());
                        }
                    }
                }
            }
        }
    }

    sort_by_page_hint_if_complete(&mut urls);
    Ok(urls.into_iter().take(max_pages).collect())
}

// ── imhentai.xxx ─────────────────────────────────────────────────────────
fn extract_imhentai(html: &str, base: &Url, max_pages: usize) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    // imhentai uses <img> with high-res images inside gallery
    for selector_text in &[
        ".gallery img",
        "#gallery_id img",
        "img.gallery-img",
        ".thumb-image img",
        "picture source",
    ] {
        if let Ok(sel) = Selector::parse(selector_text) {
            for el in document.select(&sel) {
                for attr in &["src", "data-src", "srcset", "data-full"] {
                    if let Some(src) = el.value().attr(attr) {
                        for candidate in expand_srcset_simple(src) {
                            if candidate.is_empty() || candidate.contains("data:") {
                                continue;
                            }
                            if let Ok(abs) = base.join(&candidate) {
                                let s = abs.to_string();
                                if seen.insert(s.clone()) {
                                    urls.push(s);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    sort_by_page_hint_if_complete(&mut urls);
    Ok(urls.into_iter().take(max_pages).collect())
}

/// Selector/script/DOM unioning can discover page 3 before a script-only page
/// 2. When every candidate exposes a trailing page number, restore natural
/// chapter order. If any URL has no reliable hint, preserve discovery order.
fn sort_by_page_hint_if_complete(urls: &mut [String]) {
    if urls.len() >= 2 && urls.iter().all(|url| trailing_page_number(url).is_some()) {
        urls.sort_by_key(|url| trailing_page_number(url).unwrap_or(u64::MAX));
    }
}

fn trailing_page_number(value: &str) -> Option<u64> {
    let parsed = Url::parse(value).ok()?;
    let file = parsed.path().rsplit('/').next()?;
    let stem = file.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(file);
    let reversed = stem
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if reversed.is_empty() {
        return None;
    }
    reversed.chars().rev().collect::<String>().parse().ok()
}

// ── Madara / wp-manga theme sites (allporncomic.com, ilikecomix.com) ─────
// Both run the same Madara reader: chapter pages carry the page images as
// `.reading-content img.wp-manga-chapter-img` with lazy `data-src` attrs
// (allporncomic even puts a leading space in data-src — trimmed here).
//
// allporncomic:  series https://allporncomic.com/porncomic/<slug>/
//                chapter https://allporncomic.com/porncomic/<slug>/<chapter>/
// ilikecomix:    series https://ilikecomix.com/manga/c/<slug>/
//                chapter https://ilikecomix.com/manga/c/<slug>/<chapter>/
//
// A series URL resolves to its most recent chapter (the first chapter link
// in the page, matching the "Read Last" button the site itself uses).
fn extract_madara(
    client: &Client,
    url: &Url,
    html: &str,
    base: &Url,
    max_pages: usize,
) -> Result<Vec<String>> {
    let chapter_html = if is_madara_series_page(url) {
        match first_madara_chapter(html, base) {
            Some(chapter_url) => {
                // Follow the chapter link: series pages have no page images.
                // Cloudflare-fronted variants (allporncomic) challenge
                // rustls: use the challenge-safe fetch (curl fallback);
                // non-challenged sites (ilikecomix) take the reqwest path.
                match crate::fetch_html_cloudflare_safe(
                    client,
                    chapter_url.as_str(),
                    16 * 1024 * 1024,
                ) {
                    Some(fetched) => fetched,
                    None => {
                        anyhow::bail!("failed to load chapter page {chapter_url} for series {url}");
                    }
                }
            }
            None => {
                anyhow::bail!("no chapters found on series page {url} (is it a chapter URL?)");
            }
        }
    } else {
        html.to_string()
    };

    let document = Html::parse_document(&chapter_html);
    let urls = parse_madara_chapter(&document, base, max_pages)?;
    if urls.is_empty() {
        anyhow::bail!("no page images found in the Madara chapter");
    }
    Ok(urls)
}

/// Extract the ordered page images from a parsed Madara chapter page.
fn parse_madara_chapter(document: &Html, base: &Url, max_pages: usize) -> Result<Vec<String>> {
    let mut urls: Vec<String> = Vec::new();
    let mut seen = HashSet::new();

    // Precise reader selector: only the chapter's own page images (covers,
    // avatars and ad slots live outside .reading-content).
    collect_madara_images(
        document,
        ".reading-content img.wp-manga-chapter-img",
        base,
        &mut urls,
        &mut seen,
    );
    // Defensive fallback: some Madara variants omit the class and only keep
    // plain images inside the reader container.
    if urls.len() < 2 {
        collect_madara_images(document, ".reading-content img", base, &mut urls, &mut seen);
    }
    Ok(urls.into_iter().take(max_pages).collect())
}

fn collect_madara_images(
    document: &Html,
    selector_text: &str,
    base: &Url,
    urls: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let Ok(sel) = Selector::parse(selector_text) else {
        return;
    };
    for img in document.select(&sel) {
        for attr in &["data-src", "src"] {
            let Some(raw) = img.value().attr(attr) else {
                continue;
            };
            let candidate = normalize_embedded_url(raw);
            if candidate.is_empty() || candidate.starts_with("data:") {
                continue;
            }
            if let Ok(abs) = base.join(&candidate) {
                let mut abs = abs;
                abs.set_fragment(None);
                let normalized = abs.to_string();
                if seen.insert(normalized.clone()) {
                    urls.push(normalized);
                }
            }
        }
    }
}

/// Series pages: /porncomic/<slug>/ (allporncomic, 2 segments) or
/// /manga/c/<slug>/ (ilikecomix, 3 segments). Chapter pages have one more
/// path segment than the series page.
fn is_madara_series_page(url: &Url) -> bool {
    let segments: Vec<&str> = url
        .path_segments()
        .map(|segments| segments.filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    (segments.len() == 2 && segments[0] == "porncomic")
        || (segments.len() == 3 && segments[0] == "manga" && segments[1] == "c")
}

/// First chapter link on a series page (the site's own "Read Last" button).
fn first_madara_chapter(html: &str, base: &Url) -> Option<Url> {
    let document = Html::parse_document(html);
    // Prefer the explicit "Read Last" button, then the chapter list, then a
    // site-specific deep link (allporncomic only has the /porncomic/ prefix;
    // ilikecomix uses /manga/c/).
    for selector_text in [
        "#btn-read-last[href]",
        "li.wp-manga-chapter a[href]",
        "a[href*='/porncomic/']",
        "a[href*='/manga/c/']",
    ] {
        let Ok(sel) = Selector::parse(selector_text) else {
            continue;
        };
        for a in document.select(&sel) {
            let Some(href) = a.value().attr("href") else {
                continue;
            };
            if let Ok(abs) = base.join(href) {
                if is_madara_chapter_url(&abs) {
                    return Some(abs);
                }
            }
        }
    }
    None
}

/// A chapter URL has one segment beyond the series prefix.
fn is_madara_chapter_url(url: &Url) -> bool {
    let segments: Vec<&str> = url
        .path_segments()
        .map(|segments| segments.filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    (segments.len() == 3 && segments[0] == "porncomic")
        || (segments.len() == 4 && segments[0] == "manga" && segments[1] == "c")
}

// ── Generic ──────────────────────────────────────────────────────────────
fn extract_generic(html: &str, base: &Url, max_pages: usize) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let image_attrs = [
        "data-srcset",
        "srcset",
        "data-original",
        "data-full",
        "data-src",
        "data-lazy-src",
        "data-url",
        "data-image",
        "data-cfsrc",
        "src",
    ];

    let mut reader = Vec::new();
    let mut reader_seen = HashSet::new();
    collect_dom_images(
        &document,
        "#readerarea img, #reader-area img, #manga-reader img, #chapter-reader img, \
         .reader-area img, .reader-content img, .reading-content img, \
         .chapter-content img, .chapter-reader img, .container-chapter-reader img, \
         .page-break img, .manga-page img, [data-page] img, img[data-page]",
        &image_attrs,
        base,
        &mut reader,
        &mut reader_seen,
    );
    let mut scripts = Vec::new();
    let mut script_seen = HashSet::new();
    collect_script_images(&document, base, &mut scripts, &mut script_seen);
    let mut dom = Vec::new();
    let mut dom_seen = HashSet::new();
    collect_dom_images(
        &document,
        "img, picture source",
        &image_attrs,
        base,
        &mut dom,
        &mut dom_seen,
    );

    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    for candidate in reader.into_iter().chain(scripts).chain(dom) {
        if seen.insert(candidate.clone()) {
            urls.push(candidate);
        }
    }
    if urls.is_empty() {
        collect_dom_images(
            &document,
            "meta[property='og:image'], meta[name='twitter:image'], \
             meta[property='twitter:image']",
            &["content"],
            base,
            &mut urls,
            &mut seen,
        );
    }

    sort_by_page_hint_if_complete(&mut urls);
    Ok(urls.into_iter().take(max_pages).collect())
}

fn expand_srcset_simple(value: &str) -> Vec<String> {
    if value.contains(',') {
        let mut candidates = value
            .split(',')
            .filter_map(|entry| {
                let trimmed = entry.trim();
                let mut parts = trimmed.split_whitespace();
                let url = parts.next()?.to_string();
                let descriptor = parts
                    .next()
                    .and_then(srcset_descriptor_value)
                    .unwrap_or(1.0);
                Some((descriptor, url))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
        candidates
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect()
    } else {
        vec![value.trim().to_string()]
    }
}

fn srcset_descriptor_value(value: &str) -> Option<f32> {
    value
        .strip_suffix('w')
        .or_else(|| value.strip_suffix('x'))
        .and_then(|number| number.parse::<f32>().ok())
}

fn collect_dom_images(
    document: &Html,
    selector_text: &str,
    attrs: &[&str],
    base: &Url,
    output: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let Ok(selector) = Selector::parse(selector_text) else {
        return;
    };
    for element in document.select(&selector) {
        if element_is_tiny(
            element.value().attr("width"),
            element.value().attr("height"),
        ) {
            continue;
        }
        let before = output.len();
        for attr in attrs {
            let Some(value) = element.value().attr(attr) else {
                continue;
            };
            let candidates = expand_srcset_simple(value);
            for candidate in candidates {
                if push_normalized_candidate(base, &candidate, false, output, seen) {
                    break;
                }
            }
            if output.len() > before {
                break;
            }
        }
    }
}

fn element_is_tiny(width: Option<&str>, height: Option<&str>) -> bool {
    let parse = |value: Option<&str>| {
        value.and_then(|raw| raw.trim().trim_end_matches("px").parse::<u32>().ok())
    };
    matches!((parse(width), parse(height)), (Some(w), Some(h)) if w <= 160 && h <= 160)
}

fn collect_script_images(
    document: &Html,
    base: &Url,
    output: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let Ok(selector) = Selector::parse("script") else {
        return;
    };
    for script in document.select(&selector) {
        let text = script.inner_html();
        let script_type = script.value().attr("type").unwrap_or("");
        let trimmed = text.trim();
        let looks_like_json =
            script_type.contains("json") || matches!(trimmed.chars().next(), Some('{') | Some('['));
        if looks_like_json {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
                collect_json_image_strings(&value, base, output, seen);
            }
        }
        for token in quoted_script_tokens(&text) {
            push_normalized_candidate(base, &token, true, output, seen);
        }
    }
}

fn collect_json_image_strings(
    value: &serde_json::Value,
    base: &Url,
    output: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    match value {
        serde_json::Value::String(candidate) => {
            push_normalized_candidate(base, candidate, true, output, seen);
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_json_image_strings(value, base, output, seen);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                collect_json_image_strings(value, base, output, seen);
            }
        }
        _ => {}
    }
}

fn quoted_script_tokens(script: &str) -> Vec<String> {
    let bytes = script.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        let quote = bytes[index];
        if quote != b'"' && quote != b'\'' {
            index += 1;
            continue;
        }
        index += 1;
        let start = index;
        let mut escaped = false;
        while index < bytes.len() {
            let byte = bytes[index];
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == quote {
                if index > start {
                    tokens.push(script[start..index].to_string());
                }
                index += 1;
                break;
            }
            index += 1;
        }
    }
    tokens
}

fn push_normalized_candidate(
    base: &Url,
    raw: &str,
    require_image_marker: bool,
    output: &mut Vec<String>,
    seen: &mut HashSet<String>,
) -> bool {
    let candidate = normalize_embedded_url(raw);
    if !is_likely_content_image(&candidate)
        || (require_image_marker && !has_image_marker(&candidate))
    {
        return false;
    }
    let Ok(mut absolute) = base.join(&candidate) else {
        return false;
    };
    absolute.set_fragment(None);
    let normalized = absolute.to_string();
    if seen.insert(normalized.clone()) {
        output.push(normalized);
        true
    } else {
        false
    }
}

fn normalize_embedded_url(raw: &str) -> String {
    raw.trim()
        .trim_matches(['"', '\'', '(', ')'])
        .replace("\\u002F", "/")
        .replace("\\u002f", "/")
        .replace("\\/", "/")
        .replace("&amp;", "&")
}

fn has_image_marker(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    [
        ".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tif", ".tiff",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || lower.contains("format=image")
        || lower.contains("type=image")
}

fn is_likely_content_image(url: &str) -> bool {
    if url.is_empty() || url.starts_with("data:") || url.starts_with("blob:") {
        return false;
    }
    let lower = url.to_ascii_lowercase();
    if lower.contains("avatar")
        || lower.contains("logo")
        || lower.contains("icon")
        || lower.contains("banner")
        || lower.contains("pixel")
        || lower.contains("loading")
        || lower.contains("spinner")
        || lower.contains("1x1")
        || lower.contains("blank")
        || lower.contains("favicon")
        || lower.contains("emoji")
        || lower.contains("tracking")
        || lower.contains("advert")
        || lower.contains("/ads/")
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manganato_merges_pages_from_all_reader_selectors() {
        let base = Url::parse("https://chapmanganato.example/chapter/1").unwrap();
        let html = r#"
            <div class="container-chapter-reader"><img src="/page-1.jpg"></div>
            <div class="reading-content"><img data-src="/page-2.jpg"></div>
        "#;
        let urls = extract_manganato(html, &base, 10).unwrap();
        assert_eq!(urls.len(), 2);
        assert!(urls[0].ends_with("/page-1.jpg"));
        assert!(urls[1].ends_with("/page-2.jpg"));
    }

    #[test]
    fn imhentai_merges_pages_from_all_gallery_selectors() {
        let base = Url::parse("https://imhentai.example/gallery/1").unwrap();
        let html = r#"
            <div class="gallery"><img src="/page-1.jpg"></div>
            <img class="gallery-img" data-full="/page-2.jpg">
        "#;
        let urls = extract_imhentai(html, &base, 10).unwrap();
        assert_eq!(urls.len(), 2);
        assert!(urls[0].ends_with("/page-1.jpg"));
        assert!(urls[1].ends_with("/page-2.jpg"));
    }

    #[test]
    fn generic_extractor_merges_reader_script_and_dom_pages() {
        let base = Url::parse("https://reader.example/chapter/1").unwrap();
        let html = r#"
            <div id="readerarea"><img src="/page-1.jpg"></div>
            <script>window.pages = ["/page-2.jpg"];</script>
            <img src="/page-3.jpg">
        "#;
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(urls.len(), 3);
        assert!(urls.iter().any(|url| url.ends_with("/page-1.jpg")));
        assert!(urls.iter().any(|url| url.ends_with("/page-2.jpg")));
        assert!(urls.iter().any(|url| url.ends_with("/page-3.jpg")));
    }

    #[test]
    fn generic_merge_restores_natural_order_for_partial_sources() {
        let base = Url::parse("https://reader.example/chapter/1").unwrap();
        let html = r#"
            <div id="readerarea">
                <img src="/page-001.jpg">
                <img src="/page-003.jpg">
            </div>
            <script>window.pages = ["/page-002.jpg"];</script>
        "#;
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(
            urls.iter()
                .map(|url| url.rsplit('/').next().unwrap())
                .collect::<Vec<_>>(),
            vec!["page-001.jpg", "page-002.jpg", "page-003.jpg"]
        );
    }

    #[test]
    fn detects_nhentai_url() {
        let url = Url::parse("https://nhentai.net/g/177013/").unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::Nhentai));
    }

    #[test]
    fn detects_allporncomic_url() {
        let url = Url::parse(
            "https://allporncomic.com/porncomic/2-x-1-lunch-dragon-ball-yamamoto/1-2-x-1-lunch/",
        )
        .unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::AllPornComic));
    }

    #[test]
    fn detects_ilikecomix_url() {
        let url = Url::parse(
            "https://ilikecomix.com/manga/c/bubble-butt-princess-jabcomix/8-bubble-butt-princess-issue-8/",
        )
        .unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::IlikeComix));
    }

    #[test]
    fn madara_series_page_detection() {
        let apc_series = Url::parse("https://allporncomic.com/porncomic/some-comic/").unwrap();
        let apc_chapter =
            Url::parse("https://allporncomic.com/porncomic/some-comic/chapter-1/").unwrap();
        let ilc_series = Url::parse("https://ilikecomix.com/manga/c/some-comic/").unwrap();
        let ilc_chapter =
            Url::parse("https://ilikecomix.com/manga/c/some-comic/chapter-1/").unwrap();
        assert!(is_madara_series_page(&apc_series));
        assert!(!is_madara_series_page(&apc_chapter));
        assert!(is_madara_series_page(&ilc_series));
        assert!(!is_madara_series_page(&ilc_chapter));
        // Homepage / category pages are not series pages either.
        let home = Url::parse("https://allporncomic.com/").unwrap();
        let cat = Url::parse("https://allporncomic.com/porncomic-cat/hentaib/").unwrap();
        assert!(!is_madara_series_page(&home));
        assert!(!is_madara_series_page(&cat));
    }

    #[test]
    fn allporncomic_chapter_extracts_reader_images_only() {
        // Real markup from allporncomic.com: wp-manga-chapter-img elements
        // with a LEADING SPACE in data-src (site quirk), plus a cover image
        // and a lazy placeholder that must NOT leak into the result.
        let html = r#"
<div class="reading-content">
  <img id="image-0" data-src=" https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//abc/01_PAGE001.jpg" class="wp-manga-chapter-img img-responsive lazyload-ordered effect-fade">
  <img id="image-1" data-src=" https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//abc/02_PAGE002.jpg" class="wp-manga-chapter-img img-responsive lazyload-ordered effect-fade">
  <img id="image-2" src="https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//abc/03_PAGE003.jpg" class="wp-manga-chapter-img img-responsive">
</div>
<img class="img-responsive" src="data:image/svg+xml;base64,PHN2Zz4=" data-src="https://cdn.allporncomic.com/wp-content/uploads/2018/12/books-stack-of-three-1.png">
"#;
        let base = Url::parse("https://allporncomic.com/porncomic/some-comic/chapter-1/").unwrap();
        let urls = parse_madara_chapter(&Html::parse_document(html), &base, 100).unwrap();
        assert_eq!(urls.len(), 3);
        assert_eq!(
            urls[0],
            "https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//abc/01_PAGE001.jpg"
        );
        assert_eq!(
            urls[2],
            "https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//abc/03_PAGE003.jpg"
        );
    }

    #[test]
    fn madara_series_page_picks_read_last_button() {
        let html = r#"
<ul class="wp-manga-chapters">
  <li class="wp-manga-chapter"><a href="https://allporncomic.com/porncomic/some-comic/chapter-3/">Chapter 3</a></li>
  <li class="wp-manga-chapter"><a href="https://allporncomic.com/porncomic/some-comic/chapter-2/">Chapter 2</a></li>
</ul>
<a href="https://allporncomic.com/porncomic/some-comic/chapter-3/" id="btn-read-last" class="c-btn c-btn_style-1">Read Last</a>
"#;
        let base = Url::parse("https://allporncomic.com/porncomic/some-comic/").unwrap();
        let chapter = first_madara_chapter(html, &base).unwrap();
        assert_eq!(
            chapter.as_str(),
            "https://allporncomic.com/porncomic/some-comic/chapter-3/"
        );
    }

    #[test]
    fn madara_ilikecomix_series_picks_first_chapter_link() {
        let html = r#"
<ul class="wp-manga-chapters">
  <li class="wp-manga-chapter"><a href="https://ilikecomix.com/manga/c/some-comic/1-first-issue/">Issue 1</a></li>
  <li class="wp-manga-chapter"><a href="https://ilikecomix.com/manga/c/some-comic/2-second-issue/">Issue 2</a></li>
</ul>
"#;
        let base = Url::parse("https://ilikecomix.com/manga/c/some-comic/").unwrap();
        let chapter = first_madara_chapter(html, &base).unwrap();
        assert_eq!(
            chapter.as_str(),
            "https://ilikecomix.com/manga/c/some-comic/1-first-issue/"
        );
    }

    #[test]
    fn allporncomic_chapter_without_class_falls_back_to_reader() {
        let html = r#"
<div class="reading-content">
  <img data-src="https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//x/1.jpg">
  <img data-src="https://cdn.allporncomic.com/wp-content/uploads/WP-manga/data//x/2.jpg">
</div>
"#;
        let base = Url::parse("https://allporncomic.com/porncomic/some-comic/chapter-1/").unwrap();
        let urls = parse_madara_chapter(&Html::parse_document(html), &base, 100).unwrap();
        assert_eq!(urls.len(), 2);
    }

    /// Live check against the real site (run with `-- --ignored`; not part
    /// of the default offline suite).
    #[test]
    #[ignore = "network"]
    fn live_allporncomic_chapter_extraction() {
        let client = reqwest::blocking::Client::builder()
            .user_agent(UA)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let url = Url::parse(
            "https://allporncomic.com/porncomic/2-x-1-lunch-dragon-ball-yamamoto/1-2-x-1-lunch/",
        )
        .unwrap();
        // The site is Cloudflare-fronted and challenges rustls: exercise the
        // same challenge-safe path the bot uses (curl fallback).
        let html = crate::fetch_html_cloudflare_safe(&client, url.as_str(), 16 * 1024 * 1024)
            .expect("chapter HTML should fetch (reqwest or curl fallback)");
        assert!(html.contains("reading-content"));
        let urls = extract_madara(&client, &url, &html, &url, 500).unwrap();
        assert!(urls.len() >= 30, "expected ~35 pages, got {}", urls.len());
        assert!(
            urls[0].contains("WP-manga/data"),
            "unexpected first URL: {}",
            urls[0]
        );
        assert!(urls[0].starts_with("https://cdn.allporncomic.com/"));
    }

    /// A series URL must resolve to its most recent chapter's pages.
    #[test]
    #[ignore = "network"]
    fn live_allporncomic_series_follows_latest_chapter() {
        let client = reqwest::blocking::Client::builder()
            .user_agent(UA)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let url =
            Url::parse("https://allporncomic.com/porncomic/2-x-1-lunch-dragon-ball-yamamoto/")
                .unwrap();
        let html = crate::fetch_html_cloudflare_safe(&client, url.as_str(), 16 * 1024 * 1024)
            .expect("series HTML should fetch (reqwest or curl fallback)");
        let urls = extract_madara(&client, &url, &html, &url, 500).unwrap();
        assert!(
            urls.len() >= 30,
            "series URL should follow to the latest chapter, got {} pages",
            urls.len()
        );
        assert!(urls[0].contains("WP-manga/data"));
    }

    /// ilikecomix.com — same Madara reader, no Cloudflare challenge.
    #[test]
    #[ignore = "network"]
    fn live_ilikecomix_chapter_extraction() {
        let client = reqwest::blocking::Client::builder()
            .user_agent(UA)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let url = Url::parse(
            "https://ilikecomix.com/manga/c/bubble-butt-princess-jabcomix/8-bubble-butt-princess-issue-8/",
        )
        .unwrap();
        let html = crate::fetch_html_cloudflare_safe(&client, url.as_str(), 16 * 1024 * 1024)
            .expect("chapter HTML should fetch");
        assert!(html.contains("reading-content"));
        let urls = extract_madara(&client, &url, &html, &url, 500).unwrap();
        assert!(urls.len() >= 10, "expected many pages, got {}", urls.len());
        assert!(
            urls[0].contains("WP-manga/data"),
            "unexpected first URL: {}",
            urls[0]
        );
    }

    #[test]
    fn detects_mangadex_url() {
        let url = Url::parse("https://mangadex.org/chapter/abc-123-def").unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::Mangadex));
    }

    #[test]
    fn detects_manganato_url() {
        let url = Url::parse("https://chapmanganato.to/manga-xy123/chapter-1").unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::Manganato));
    }

    #[test]
    fn nhentai_thumb_to_full_converts_correctly() {
        let thumb = "https://t.nhentai.net/galleries/1234567/1t.jpg";
        let full = normalize_nhentai_url(thumb);
        assert_eq!(full, "https://i.nhentai.net/galleries/1234567/1.jpg");
    }

    #[test]
    fn extract_nhentai_gallery_id_works() {
        let url = Url::parse("https://nhentai.net/g/177013/1/").unwrap();
        assert_eq!(extract_nhentai_gallery_id(&url), Some("177013".to_string()));
    }

    #[test]
    fn parses_nhentai_embedded_tags() {
        // Real nhentai markup: the gallery JSON is escaped inside
        // "body":"{...}" (literal backslash-quote pairs). Build the
        // fixture the same way the site embeds it.
        let inner = r#"{"id":177013,"media_id":"12345","tags":[{"id":1,"name":"horror"},{"id":2,"name":"dark skin"}]}"#;
        let bs = r"\";
        let escaped = inner.replace('"', &format!("{bs}\""));
        let html = format!(r#"<script id="galleryData">{{"body":"{escaped}"}}</script>"#);
        let url = Url::parse("https://nhentai.net/g/177013/").unwrap();
        let tags = parse_nhentai_tags(&html, &url);
        assert!(tags.iter().any(|t| t == "horror"), "got {tags:?}");
        assert!(tags.iter().any(|t| t == "dark skin"), "got {tags:?}");
    }

    #[test]
    fn generic_extracts_from_og_image() {
        let html = r#"<html><head><meta property="og:image" content="/page1.png"></head><body></body></html>"#;
        let base = Url::parse("https://example.com/manga/").unwrap();
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(urls, vec!["https://example.com/page1.png"]);
    }

    #[test]
    fn generic_filters_avatars_and_logos() {
        let html = r#"<img src="/avatar.jpg"><img src="/logo.png"><img src="/page1.png">"#;
        let base = Url::parse("https://example.com/").unwrap();
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(urls, vec!["https://example.com/page1.png"]);
    }

    #[test]
    fn generic_prefers_reader_pages_over_open_graph_cover() {
        let html = r#"
            <meta property="og:image" content="/cover.jpg">
            <div class="reading-content">
              <img data-src="/chapter/page-001.webp">
              <img data-src="/chapter/page-002.webp">
            </div>
        "#;
        let base = Url::parse("https://example.com/title/chapter-1").unwrap();
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(
            urls,
            vec![
                "https://example.com/chapter/page-001.webp",
                "https://example.com/chapter/page-002.webp"
            ]
        );
    }

    #[test]
    fn generic_extracts_pages_from_embedded_javascript() {
        let html = r#"
            <script>
              window.__PAGES__ = ["https:\/\/cdn.example.test\/p01.jpg",
                                  "https:\/\/cdn.example.test\/p02.jpg"];
            </script>
        "#;
        let base = Url::parse("https://example.com/chapter/1").unwrap();
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(
            urls,
            vec![
                "https://cdn.example.test/p01.jpg",
                "https://cdn.example.test/p02.jpg"
            ]
        );
    }

    #[test]
    fn generic_chooses_highest_resolution_srcset() {
        let html = r#"
            <div id="readerarea">
              <img srcset="/p1-small.jpg 480w, /p1-large.jpg 1600w">
              <img srcset="/p2-small.jpg 1x, /p2-large.jpg 2x">
            </div>
        "#;
        let base = Url::parse("https://example.com/chapter/").unwrap();
        let urls = extract_generic(html, &base, 10).unwrap();
        assert_eq!(
            urls,
            vec![
                "https://example.com/p1-large.jpg",
                "https://example.com/p2-large.jpg"
            ]
        );
    }

    #[test]
    fn nhentai_thumbnail_conversion_handles_numbered_hosts() {
        assert_eq!(
            normalize_nhentai_url("https://t2.nhentai.net/galleries/12/23t.webp"),
            "https://i2.nhentai.net/galleries/12/23.webp"
        );
    }
}

#[cfg(test)]
mod nhentai_normalize_tests {
    use super::*;

    #[test]
    fn thumbnail_double_ext_collapses() {
        // The exact malformed URLs seen in production logs.
        assert_eq!(
            normalize_nhentai_url("https://i2.nhentai.net/galleries/3257072/3t.webp.webp"),
            "https://i2.nhentai.net/galleries/3257072/3.webp"
        );
    }

    #[test]
    fn thumbnail_single_ext_collapses() {
        assert_eq!(
            normalize_nhentai_url("https://t.nhentai.net/galleries/3257072/1t.jpg"),
            "https://i.nhentai.net/galleries/3257072/1.jpg"
        );
        assert_eq!(
            normalize_nhentai_url("https://i3.nhentai.net/galleries/3257072/23t.webp"),
            "https://i3.nhentai.net/galleries/3257072/23.webp"
        );
    }

    #[test]
    fn full_pages_unchanged() {
        assert_eq!(
            normalize_nhentai_url("https://i1.nhentai.net/galleries/3257072/2.webp"),
            "https://i1.nhentai.net/galleries/3257072/2.webp"
        );
        assert_eq!(
            normalize_nhentai_url("https://i.nhentai.net/galleries/3257072/14.png"),
            "https://i.nhentai.net/galleries/3257072/14.png"
        );
    }

    #[test]
    fn thumbnail_detector() {
        assert!(is_nhentai_thumbnail(
            "https://i2.nhentai.net/galleries/3257072/3t.webp"
        ));
        assert!(!is_nhentai_thumbnail(
            "https://i2.nhentai.net/galleries/3257072/3.webp"
        ));
    }

    #[test]
    fn page_number_extraction() {
        assert_eq!(
            page_number_of("https://i1.nhentai.net/galleries/3257072/2.webp"),
            Some(2)
        );
        assert_eq!(
            page_number_of("https://i1.nhentai.net/galleries/3257072/14.png"),
            Some(14)
        );
        assert_eq!(
            page_number_of("https://i1.nhentai.net/galleries/3257072/x.webp"),
            None
        );
    }
}
