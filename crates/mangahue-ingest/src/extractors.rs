use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{REFERER, USER_AGENT};
use scraper::{Html, Selector};
use std::collections::HashSet;
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
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    // Method 1: Extract from <noscript> images (thumbnails → full image)
    let noscript_sel = Selector::parse("noscript img").unwrap();
    for img in document.select(&noscript_sel) {
        if let Some(src) = img.value().attr("src") {
            let full = nhentai_thumb_to_full(src);
            if seen.insert(full.clone()) {
                if let Ok(abs) = base.join(&full) {
                    urls.push(abs.to_string());
                }
            }
        }
    }

    // Method 2: Extract from lazy-loaded images (data-src)
    if urls.is_empty() {
        let img_sel = Selector::parse(".thumb-container img").unwrap();
        for img in document.select(&img_sel) {
            for attr in &["data-src", "src"] {
                if let Some(src) = img.value().attr(attr) {
                    let full = nhentai_thumb_to_full(src);
                    if seen.insert(full.clone()) {
                        if let Ok(abs) = base.join(&full) {
                            urls.push(abs.to_string());
                        }
                    }
                }
            }
        }
    }

    // Method 3: Build URLs from gallery data in JSON/script
    if urls.is_empty() {
        if let Ok(script_sel) = Selector::parse("script") {
            for script in document.select(&script_sel) {
                let text = script.inner_html();
                if text.contains("media_id") && text.contains("num_pages") {
                    if let (Some(mid_start), Some(np_start)) = (
                        text.find("\"media_id\":").map(|i| i + 12),
                        text.find("\"num_pages\":").map(|i| i + 13),
                    ) {
                        let media_id: String = text[mid_start..]
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect();
                        let num_pages: usize = text[np_start..]
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse()
                            .unwrap_or(0);

                        if !media_id.is_empty() && num_pages > 0 {
                            let ext = if html.contains(".webp") || html.contains("image/webp") {
                                "webp"
                            } else {
                                "jpg"
                            };
                            for page in 1..=num_pages.min(max_pages) {
                                let img_url = format!(
                                    "https://i.nhentai.net/galleries/{}/{}.{}",
                                    media_id, page, ext
                                );
                                if seen.insert(img_url.clone()) {
                                    urls.push(img_url);
                                }
                            }
                        }
                    }
                    if !urls.is_empty() {
                        break;
                    }
                }
            }
        }
    }

    // Method 4: Use the API as fallback
    if urls.is_empty() {
        if let Some(gid) = extract_nhentai_gallery_id(base) {
            let api_url = format!("https://nhentai.net/api/gallery/{}", gid);
            if let Ok(resp) = _client
                .get(&api_url)
                .header(USER_AGENT, UA)
                .timeout(Duration::from_secs(15))
                .send()
            {
                if let Ok(body) = resp.text() {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let (Some(mid), Some(images)) = (
                            json["media_id"].as_str().map(|s| s.to_string()),
                            json["images"]["pages"].as_array(),
                        ) {
                            for page in images.iter().take(max_pages) {
                                let ext = page["t"].as_str().unwrap_or("j");
                                let num = page["num"].as_u64().unwrap_or(1);
                                let img = format!(
                                    "https://i.nhentai.net/galleries/{}/{}.{}",
                                    mid, num, ext
                                );
                                if seen.insert(img.clone()) {
                                    urls.push(img);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(urls)
}

fn nhentai_thumb_to_full(src: &str) -> String {
    // Convert any //t, //t1, //t2, ... //t999 subdomain to //i, //i1, //i2, ...
    let mut s = src.to_string();
    if let Some(t_pos) = s.find("//t") {
        let after_slash = &s[t_pos + 2..];
        let mut end = 1; // after 't'
        while end < after_slash.len() && after_slash.as_bytes()[end].is_ascii_digit() {
            end += 1;
        }
        if end < after_slash.len() && &after_slash[end..end + 12] == ".nhentai.net" {
            let new_host = format!("//i{}.nhentai.net", &after_slash[1..end]);
            s = format!("{}{}{}", &s[..t_pos], new_host, &s[t_pos + 2 + end + 12..]);
        } else if end < after_slash.len() && &after_slash[end..end + 11] == ".nhentai.to" {
            let new_host = format!("//i{}.nhentai.to", &after_slash[1..end]);
            s = format!("{}{}{}", &s[..t_pos], new_host, &s[t_pos + 2 + end + 11..]);
        }
    }

    // Remove 't' suffix before extension: 1t.jpg → 1.jpg, 23t.png → 23.png
    if let Some(ext_dot) = s.rfind('.') {
        let base = &s[..ext_dot];
        let ext = &s[ext_dot..];
        if let Some(without_t) = base.strip_suffix('t') {
            if !without_t.is_empty() && without_t.ends_with(|c: char| c.is_ascii_digit()) {
                return format!("{}{}", without_t, ext);
            }
        }
    }
    s
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
            if let Ok(page_html) = page_resp.text() {
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
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    let selectors: &[(&str, &[&str])] = &[
        ("meta[property='og:image']", &["content"]),
        ("meta[name='twitter:image']", &["content"]),
        (
            "img.chapter-img, img.page-img, img.reader-img, img.manga-page",
            &["src", "data-src", "data-original", "data-lazy-src"],
        ),
        (
            "img[src*='chapter'], img[src*='page'], img[src*='galleries']",
            &["src", "data-src", "data-original"],
        ),
        (
            "img",
            &["src", "data-src", "data-original", "data-lazy-src"],
        ),
        ("source", &["srcset"]),
    ];

    for (selector_text, attrs) in selectors {
        if let Ok(sel) = Selector::parse(selector_text) {
            for element in document.select(&sel) {
                for attr in *attrs {
                    if let Some(value) = element.value().attr(attr) {
                        for candidate in expand_srcset_simple(value) {
                            if is_likely_content_image(&candidate) {
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
        if !urls.is_empty() {
            break;
        }
    }

    Ok(urls.into_iter().take(max_pages).collect())
}

fn expand_srcset_simple(value: &str) -> Vec<String> {
    if value.contains(',') {
        value
            .split(',')
            .filter_map(|entry| {
                let trimmed = entry.trim();
                trimmed.split_whitespace().next().map(|s| s.to_string())
            })
            .collect()
    } else {
        vec![value.trim().to_string()]
    }
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
        let full = nhentai_thumb_to_full(thumb);
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
}
