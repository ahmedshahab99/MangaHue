use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{REFERER, USER_AGENT};
use scraper::{Html, Selector};
use std::collections::HashSet;
use std::io::Read;
use std::time::Duration;
use url::Url;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

#[derive(Debug)]
pub enum MangaSite {
    Nhentai,
    Mangadex,
    Manganato,
    Ehentai,
    Imhentai,
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
    let _ = parsed.set_path(&cleaned);
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
    loop {
        let Some((s, e)) = stem.rsplit_once('.') else {
            break;
        };
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
        if !urls.is_empty() {
            break;
        }
    }

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
        if !urls.is_empty() {
            break;
        }
    }

    Ok(urls.into_iter().take(max_pages).collect())
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
    if reader.len() >= 2 {
        return Ok(reader.into_iter().take(max_pages).collect());
    }

    let mut scripts = Vec::new();
    let mut script_seen = HashSet::new();
    collect_script_images(&document, base, &mut scripts, &mut script_seen);
    if scripts.len() >= 2 {
        return Ok(scripts.into_iter().take(max_pages).collect());
    }

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
    fn detects_nhentai_url() {
        let url = Url::parse("https://nhentai.net/g/177013/").unwrap();
        assert!(matches!(MangaSite::detect(&url), MangaSite::Nhentai));
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
