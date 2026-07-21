mod extractors;

use anyhow::{Context, Result};
use extractors::extract_manga_urls;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;
use url::Url;

fn serialize_duration_ms<S>(d: &Duration, s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_u64(d.as_millis() as u64)
}

fn deserialize_duration_ms<'de, D>(d: D) -> std::result::Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let ms = u64::deserialize(d)?;
    Ok(Duration::from_millis(ms))
}

const DEFAULT_USER_AGENT: &str = concat!(
    "MangaHue/",
    env!("CARGO_PKG_VERSION"),
    " (+local-authorized-colorization)"
);
const DEFAULT_MIN_REQUEST_INTERVAL_MS: u64 = 500;
const DEFAULT_MAX_PAGES: usize = 32;
const DEFAULT_MAX_BYTES_PER_PAGE: u64 = 32 * 1024 * 1024;
const DEFAULT_SELECTORS: &[(&str, &str)] = &[
    ("meta[property='og:image']", "content"),
    ("meta[name='twitter:image']", "content"),
    ("img", "src"),
    ("img", "data-src"),
    ("img", "data-original"),
    ("source", "srcset"),
    ("img", "srcset"),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub source_url: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    #[serde(
        default = "default_min_request_interval",
        serialize_with = "serialize_duration_ms",
        deserialize_with = "deserialize_duration_ms"
    )]
    pub min_request_interval: Duration,
    #[serde(default = "default_selectors")]
    pub selectors: Vec<(String, String)>,
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
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("too many redirects");
            }
            match validate_public_url(attempt.url()) {
                Ok(()) => attempt.follow(),
                Err(error) => attempt.error(error),
            }
        }))
        .build()
        .context("failed to build HTTP client")?;

    let user_agent = request
        .user_agent
        .clone()
        .unwrap_or_else(|| DEFAULT_USER_AGENT.to_string());
    let source_url = Url::parse(&request.source_url)
        .with_context(|| format!("invalid URL: {}", request.source_url))?;
    validate_public_url(&source_url)?;
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
        let image_urls = extract_manga_urls(
            &client,
            &source_url,
            &html,
            &source_url,
            max_pages,
        )?;
        for image_url in image_urls.into_iter().take(max_pages) {
            let parsed = Url::parse(&image_url)?;
            if let Err(error) = validate_public_url(&parsed) {
                skipped.push(SkippedAsset {
                    source_url: image_url,
                    reason: error.to_string(),
                });
                continue;
            }
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
            if !request.min_request_interval.is_zero() {
                std::thread::sleep(request.min_request_interval);
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

pub fn validate_public_url(url: &Url) -> Result<()> {
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "only HTTP(S) URLs are allowed"
    );
    let host = url.host_str().context("URL has no host")?;
    anyhow::ensure!(
        !matches!(
            host.to_ascii_lowercase().as_str(),
            "localhost" | "localhost.localdomain"
        ),
        "local network URLs are not allowed"
    );
    let port = url
        .port_or_known_default()
        .context("URL has no valid port")?;
    let addresses = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("could not resolve URL host {host}"))?
        .collect::<Vec<_>>();
    anyhow::ensure!(!addresses.is_empty(), "URL host resolved to no addresses");
    for address in addresses {
        anyhow::ensure!(
            is_public_ip(address.ip()),
            "private or reserved network URLs are not allowed"
        );
    }
    Ok(())
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.octets()[0] == 0
                || ip.octets()[0] >= 224)
        }
        IpAddr::V6(ip) => {
            !(ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || ip.is_multicast())
        }
    }
}

pub fn extract_image_urls(base_url: &str, html: &str) -> Result<Vec<String>> {
    extract_image_urls_with_selectors(base_url, html, &default_selectors())
}

pub fn extract_image_urls_with_selectors(
    base_url: &str,
    html: &str,
    selectors: &[(String, String)],
) -> Result<Vec<String>> {
    let base = Url::parse(base_url).with_context(|| format!("invalid base URL: {base_url}"))?;
    let document = Html::parse_document(html);
    let mut urls = Vec::new();
    let mut seen = HashSet::new();

    for (selector_text, attr) in selectors {
        let selector = Selector::parse(selector_text)
            .map_err(|error| anyhow::anyhow!("invalid selector {selector_text}: {error}"))?;
        for element in document.select(&selector) {
            let Some(value) = element.value().attr(attr.as_str()) else {
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

fn default_min_request_interval() -> Duration {
    Duration::from_millis(DEFAULT_MIN_REQUEST_INTERVAL_MS)
}

pub fn default_selectors() -> Vec<(String, String)> {
    DEFAULT_SELECTORS
        .iter()
        .map(|&(sel, attr)| (sel.to_string(), attr.to_string()))
        .collect()
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

    #[test]
    fn rejects_private_and_unsafe_ingest_destinations() {
        for value in [
            "http://127.0.0.1/page.png",
            "http://10.0.0.1/page.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/page.png",
            "file:///etc/passwd",
        ] {
            let url = Url::parse(value).unwrap();
            assert!(
                validate_public_url(&url).is_err(),
                "accepted unsafe URL {value}"
            );
        }
    }
}
