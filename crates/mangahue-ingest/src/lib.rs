use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;
use url::Url;

const DEFAULT_USER_AGENT: &str = "MangaHue/0.1 (+local-authorized-colorization)";
const DEFAULT_MAX_PAGES: usize = 32;
const DEFAULT_MAX_BYTES_PER_PAGE: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub source_url: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestReport {
    pub source_url: String,
    pub output_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub pages: Vec<IngestedPage>,
    pub skipped: Vec<SkippedAsset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedPage {
    pub index: usize,
    pub source_url: String,
    pub local_path: PathBuf,
    pub content_type: Option<String>,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedAsset {
    pub source_url: String,
    pub reason: String,
}

pub fn ingest_url(request: IngestRequest) -> Result<IngestReport> {
    let max_pages = request.max_pages.max(1);
    fs::create_dir_all(&request.output_dir)
        .with_context(|| format!("failed to create {}", request.output_dir.display()))?;

    let client = Client::builder()
        .timeout(Duration::from_secs(45))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .context("failed to build HTTP client")?;

    let user_agent = request
        .user_agent
        .clone()
        .unwrap_or_else(|| DEFAULT_USER_AGENT.to_string());
    let source_url = Url::parse(&request.source_url)
        .with_context(|| format!("invalid URL: {}", request.source_url))?;
    let mut pages = Vec::new();
    let mut skipped = Vec::new();

    let response = client
        .get(source_url.clone())
        .header(USER_AGENT, &user_agent)
        .send()
        .with_context(|| format!("failed to fetch {source_url}"))?
        .error_for_status()
        .with_context(|| format!("server rejected {source_url}"))?;
    let content_type = response_content_type(&response);

    if is_image_response(&source_url, content_type.as_deref()) {
        match download_image_response(
            response,
            &request.output_dir,
            1,
            &source_url,
            request.max_bytes_per_page,
        ) {
            Ok(page) => pages.push(page),
            Err(error) => skipped.push(SkippedAsset {
                source_url: source_url.to_string(),
                reason: error.to_string(),
            }),
        }
    } else {
        let html = response
            .text()
            .with_context(|| format!("failed to read HTML from {source_url}"))?;
        let image_urls = extract_image_urls(source_url.as_str(), &html)?;
        for image_url in image_urls.into_iter().take(max_pages) {
            let parsed = Url::parse(&image_url)?;
            match download_image_url(
                &client,
                &user_agent,
                parsed,
                &request.output_dir,
                pages.len() + 1,
                request.max_bytes_per_page,
            ) {
                Ok(page) => pages.push(page),
                Err(error) => skipped.push(SkippedAsset {
                    source_url: image_url,
                    reason: error.to_string(),
                }),
            }
            if pages.len() >= max_pages {
                break;
            }
        }
    }

    let manifest_path = request.output_dir.join("mangahue_ingest_manifest.json");
    let report = IngestReport {
        source_url: request.source_url,
        output_dir: request.output_dir.clone(),
        manifest_path,
        pages,
        skipped,
    };
    fs::write(
        &report.manifest_path,
        serde_json::to_string_pretty(&report)?,
    )
    .with_context(|| format!("failed to write {}", report.manifest_path.display()))?;
    Ok(report)
}

pub fn extract_image_urls(base_url: &str, html: &str) -> Result<Vec<String>> {
    let base = Url::parse(base_url).with_context(|| format!("invalid base URL: {base_url}"))?;
    let document = Html::parse_document(html);
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    for (selector_text, attr) in [
        ("meta[property='og:image']", "content"),
        ("meta[name='twitter:image']", "content"),
        ("img", "src"),
        ("img", "data-src"),
        ("img", "data-original"),
        ("source", "srcset"),
        ("img", "srcset"),
    ] {
        let selector = Selector::parse(selector_text)
            .map_err(|error| anyhow::anyhow!("invalid selector {selector_text}: {error}"))?;
        for element in document.select(&selector) {
            let Some(value) = element.value().attr(attr) else {
                continue;
            };
            for candidate in expand_srcset(value) {
                if let Some(url) = normalize_image_url(&base, candidate) {
                    if seen.insert(url.clone()) {
                        urls.push(url);
                    }
                }
            }
        }
    }
    Ok(urls)
}

fn download_image_url(
    client: &Client,
    user_agent: &str,
    url: Url,
    output_dir: &Path,
    index: usize,
    max_bytes: u64,
) -> Result<IngestedPage> {
    let response = client
        .get(url.clone())
        .header(USER_AGENT, user_agent)
        .send()
        .with_context(|| format!("failed to fetch {url}"))?
        .error_for_status()
        .with_context(|| format!("server rejected {url}"))?;
    download_image_response(response, output_dir, index, &url, max_bytes)
}

fn download_image_response(
    response: reqwest::blocking::Response,
    output_dir: &Path,
    index: usize,
    url: &Url,
    max_bytes: u64,
) -> Result<IngestedPage> {
    let content_type = response_content_type(&response);
    if !is_image_response(url, content_type.as_deref()) {
        anyhow::bail!("not an image response");
    }
    if let Some(length) = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        if length > max_bytes {
            anyhow::bail!("image is too large: {length} bytes");
        }
    }
    let extension = image_extension(url, content_type.as_deref());
    let local_path = output_dir.join(format!("page_{index:03}.{extension}"));
    let mut reader = response.take(max_bytes + 1);
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .with_context(|| format!("failed to read image bytes from {url}"))?;
    if data.len() as u64 > max_bytes {
        anyhow::bail!("image exceeded {max_bytes} bytes");
    }
    fs::write(&local_path, &data)
        .with_context(|| format!("failed to write {}", local_path.display()))?;
    Ok(IngestedPage {
        index,
        source_url: url.to_string(),
        local_path,
        content_type,
        bytes: data.len() as u64,
    })
}

fn response_content_type(response: &reqwest::blocking::Response) -> Option<String> {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
}

fn is_image_response(url: &Url, content_type: Option<&str>) -> bool {
    content_type
        .map(|value| value.to_ascii_lowercase().starts_with("image/"))
        .unwrap_or(false)
        || image_extension_from_url(url).is_some()
}

fn normalize_image_url(base: &Url, raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() || raw.starts_with("data:") || raw.starts_with("blob:") {
        return None;
    }
    let raw = raw.split_whitespace().next().unwrap_or(raw);
    let joined = base.join(raw).ok()?;
    if image_extension_from_url(&joined).is_some() || joined.path().contains("/image") {
        Some(joined.to_string())
    } else {
        None
    }
}

fn expand_srcset(value: &str) -> Vec<&str> {
    if value.contains(',') {
        value
            .split(',')
            .filter_map(|entry| entry.split_whitespace().next())
            .collect()
    } else {
        vec![value]
    }
}

fn image_extension(url: &Url, content_type: Option<&str>) -> &'static str {
    image_extension_from_url(url).unwrap_or(match content_type {
        Some("image/jpeg") | Some("image/jpg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/tiff") => "tif",
        Some("image/bmp") => "bmp",
        _ => "png",
    })
}

fn image_extension_from_url(url: &Url) -> Option<&'static str> {
    let extension = Path::new(url.path())
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())?;
    match extension.as_str() {
        "png" => Some("png"),
        "jpg" | "jpeg" => Some("jpg"),
        "webp" => Some("webp"),
        "tif" | "tiff" => Some("tif"),
        "bmp" => Some("bmp"),
        _ => None,
    }
}

fn default_max_pages() -> usize {
    DEFAULT_MAX_PAGES
}

fn default_max_bytes_per_page() -> u64 {
    DEFAULT_MAX_BYTES_PER_PAGE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_ordered_unique_image_urls_from_html() {
        let html = r#"
            <html>
              <head><meta property="og:image" content="/cover.jpg"></head>
              <body>
                <img src="/page-001.png">
                <img data-src="https://cdn.example.test/page-002.webp">
                <img srcset="/small.jpg 400w, /large.jpg 1200w">
                <img src="data:image/png;base64,aaa">
                <img src="/page-001.png">
              </body>
            </html>
        "#;
        let urls = extract_image_urls("https://example.test/chapter/1", html).unwrap();
        assert_eq!(
            urls,
            vec![
                "https://example.test/cover.jpg",
                "https://example.test/page-001.png",
                "https://cdn.example.test/page-002.webp",
                "https://example.test/small.jpg",
                "https://example.test/large.jpg",
            ]
        );
    }

    #[test]
    fn ignores_non_image_links() {
        let html = r#"<img src="/read/next"><img src="/page.png?token=1">"#;
        let urls = extract_image_urls("https://example.test/chapter/1", html).unwrap();
        assert_eq!(urls, vec!["https://example.test/page.png?token=1"]);
    }
}
