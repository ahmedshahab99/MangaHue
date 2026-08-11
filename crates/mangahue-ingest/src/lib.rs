mod extractors;

use anyhow::{Context, Result};
pub use extractors::{extract_manga_tags, extract_manga_urls};
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, CONTENT_LENGTH, CONTENT_TYPE, REFERER, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
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

    // DNS-rebinding guard: the custom resolver validates EVERY resolution
    // (initial request, extracted image URLs, redirect hops) and the
    // connection uses exactly the validated addresses. Validation and
    // connection can no longer disagree, so "public IP for the check,
    // private IP for the fetch" attacks are impossible.
    let client = Client::builder()
        .timeout(Duration::from_secs(45))
        .dns_resolver(std::sync::Arc::new(SafePublicResolver))
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
        // Cap the HTML body: an unbounded .text() lets a malicious server
        // stream gigabytes into RAM within the 45s timeout.
        let html = {
            let mut reader = response.take(16 * 1024 * 1024);
            let mut buffer = Vec::new();
            reader
                .read_to_end(&mut buffer)
                .with_context(|| format!("failed to read HTML from {source_url}"))?;
            String::from_utf8_lossy(&buffer).into_owned()
        };
        let image_urls = extract_manga_urls(&client, &source_url, &html, &source_url, max_pages)?;
        for image_url in image_urls.into_iter().take(max_pages) {
            let parsed = Url::parse(&image_url)?;
            if let Err(error) = validate_public_url(&parsed) {
                skipped.push(SkippedAsset {
                    source_url: image_url,
                    reason: error.to_string(),
                });
                continue;
            }
            match download_image_url_with_referer(
                &client,
                &user_agent,
                parsed,
                Some(&source_url),
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

/// Fetch a page's HTML with a Cloudflare-challenge fallback.
///
/// Some sites (Cloudflare-fronted ones like allporncomic.com) fingerprint
/// the TLS stack: rustls (reqwest's default here) gets a JS challenge
/// ("Just a moment…", HTTP 403/503) while OpenSSL-based clients like curl
/// pass. When reqwest hits a challenge, the page is re-fetched with the
/// system `curl` (present on every Linux host and Windows 10+). The URL must
/// already be SSRF-validated by the caller; the fallback re-validates it.
pub fn fetch_html_cloudflare_safe(client: &Client, url: &str, max_bytes: u64) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    if validate_public_url(&parsed).is_err() {
        return None;
    }
    // Try reqwest first; ANY failure (challenge, reset, timeout, 4xx/5xx)
    // falls through to the curl fallback — the rustls TLS fingerprint is
    // exactly what Cloudflare rejects, so the direct attempt failing is
    // the expected case for challenged sites, not an error.
    let direct = (|| -> Option<String> {
        let response = client
            .get(url)
            .header(USER_AGENT, crate::extractors::UA)
            .header(
                ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .ok()?;
        let status = response.status();
        let mut bytes = Vec::new();
        response.take(max_bytes + 1).read_to_end(&mut bytes).ok()?;
        if bytes.len() as u64 > max_bytes {
            return None;
        }
        let body = String::from_utf8_lossy(&bytes).into_owned();
        if status.is_success() && !looks_like_cloudflare_challenge(&body) {
            Some(body)
        } else {
            None
        }
    })();
    direct.or_else(|| curl_fetch_html(url, max_bytes))
}

/// True when a body is a Cloudflare JS challenge rather than real content.
/// NOTE: "challenge-platform" must NOT be checked — Cloudflare embeds its
/// beacon script on every legit page, so that marker appears on real
/// content too (this exact mistake silently rejected valid pages).
fn looks_like_cloudflare_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("just a moment")
        || lower.contains("attention required")
        || lower.contains("cf-error-details")
        || lower.contains("cf-chl-opt")
}

/// Shell out to the system curl. No shell involved (arg-array exec), the
/// URL is validated by the caller + re-validated in the wrapper, and
/// redirects are capped so a compromised site cannot point curl at internal
/// hosts more than a few hops.
fn curl_fetch_html(url: &str, max_bytes: u64) -> Option<String> {
    use std::process::Command;
    let output = Command::new("curl")
        .args([
            "-sL",
            "--max-time",
            "30",
            "--max-redirs",
            "3",
            "-A",
            crate::extractors::UA,
            "-H",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "-H",
            "Accept-Language: en-US,en;q=0.9",
        ])
        .arg(url)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes = output.stdout;
    if bytes.len() as u64 > max_bytes {
        return None;
    }
    let body = String::from_utf8_lossy(&bytes).into_owned();
    if looks_like_cloudflare_challenge(&body) {
        return None;
    }
    Some(body)
}
/// reserved, or otherwise unsafe addresses. Because reqwest connects to the
/// addresses this resolver returns, the check and the connection share a
/// single resolution — closing the DNS-rebinding window.
///
/// Attach to any client that fetches user-supplied URLs:
/// `Client::builder().dns_resolver(Arc::new(SafePublicResolver))`.
#[derive(Debug, Clone)]
pub struct SafePublicResolver;

impl reqwest::dns::Resolve for SafePublicResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(resolve_public(host))
    }
}

async fn resolve_public(
    host: String,
) -> std::result::Result<reqwest::dns::Addrs, Box<dyn std::error::Error + Send + Sync>> {
    let addresses = (host.as_str(), 0u16)
        .to_socket_addrs()
        .map_err(|error| format!("could not resolve URL host {host}: {error}"))?;
    let mut validated: Vec<std::net::SocketAddr> = Vec::new();
    for address in addresses {
        let ip = address.ip();
        if !is_public_ip(ip) {
            return Err(
                format!("URL host {host} resolved to non-public address {ip}; blocked").into(),
            );
        }
        validated.push(address);
    }
    if validated.is_empty() {
        return Err(format!("URL host {host} resolved to no addresses").into());
    }
    Ok(Box::new(validated.into_iter()))
}

fn is_public_ip(ip: IpAddr) -> bool {
    // Canonicalize first: maps IPv4-in-IPv6 (::ffff:a.b.c.d, ::a.b.c.d) to
    // IPv4 so those cannot bypass the v4 checks.
    let ip = ip.to_canonical();
    match ip {
        IpAddr::V4(ip) => {
            let octets = ip.octets();
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || octets[0] == 0
                || octets[0] >= 224
                // 100.64.0.0/10 (CGNAT): carrier-grade NAT, not public.
                || (octets[0] == 100 && octets[1] >= 64 && octets[1] <= 127)
                // 192.0.0.0/24 (IETF protocol assignments) and
                // 192.88.99.0/24 (6to4 relay anycast).
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
                // 198.18.0.0/15 (benchmarking).
                || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19)))
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            !(ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || ip.is_multicast()
                // 2001::/32: Teredo tunneling (can embed any IPv4 target) and
                // special-purpose ranges.
                || segments[0] == 0x2001
                // NAT64 well-known prefix 64:ff9b::/96 (embeds IPv4),
                // including the RFC 8215 local-use 64:ff9b:1::/48.
                || segments[0] == 0x0064 && segments[1] == 0xff9b
                // 6to4 2002::/16 (embeds IPv4).
                || segments[0] == 0x2002)
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

pub fn download_image_url(
    client: &Client,
    user_agent: &str,
    url: Url,
    output_dir: &Path,
    index: usize,
    max_bytes: u64,
) -> Result<IngestedPage> {
    download_image_url_with_referer(client, user_agent, url, None, output_dir, index, max_bytes)
}

pub fn download_image_url_with_referer(
    client: &Client,
    user_agent: &str,
    url: Url,
    referer: Option<&Url>,
    output_dir: &Path,
    index: usize,
    max_bytes: u64,
) -> Result<IngestedPage> {
    // nhentai throttles individual CDN hosts (i1 is a common offender) while
    // the same image lives on all four servers; retry with backoff across
    // the sibling hosts before giving up.
    let mut candidates: Vec<Url> = vec![url.clone()];
    candidates.extend(alternate_nhentai_hosts(&url));
    let mut last_error: Option<anyhow::Error> = None;
    for candidate in candidates {
        for attempt in 0..3usize {
            let outcome = (|| -> Result<IngestedPage> {
                let mut request = client
                    .get(candidate.clone())
                    .header(USER_AGENT, user_agent)
                    .header(
                        ACCEPT,
                        // No image/avif: the image crate build has no avif
                        // decoder, and nhentai serves avif when asked for it.
                        "image/webp,image/png,image/jpeg,image/*;q=0.9,*/*;q=0.5",
                    );
                if let Some(referer) = referer {
                    request = request.header(REFERER, referer.as_str());
                }
                let response = request
                    .send()
                    .with_context(|| format!("failed to fetch {candidate}"))?
                    .error_for_status()
                    .with_context(|| format!("server rejected {candidate}"))?;
                download_image_response(response, output_dir, index, &candidate, max_bytes)
            })();
            match outcome {
                Ok(page) => return Ok(page),
                Err(error) => {
                    last_error = Some(error);
                    if attempt < 2 {
                        std::thread::sleep(Duration::from_secs(1 + attempt as u64 * 2));
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("download failed after retries: {url}")))
}

/// nhentai serves every image on all CDN hosts (`i1..i4.nhentai.net` plus
/// the canonical `i.nhentai.net`). When one host throttles us, the same
/// path on a sibling host still works; return those siblings (empty for
/// any non-nhentai URL).
fn alternate_nhentai_hosts(url: &Url) -> Vec<Url> {
    let Some(host) = url.host_str().map(str::to_string) else {
        return Vec::new();
    };
    let lower = host.to_ascii_lowercase();
    let Some(domain) = [".nhentai.net", ".nhentai.to"]
        .iter()
        .find(|d| lower.ends_with(**d))
    else {
        return Vec::new();
    };
    let prefix = &lower[..lower.len() - domain.len()]; // e.g. "i1" or "i"
    let Some(num_str) = prefix.strip_prefix('i') else {
        return Vec::new();
    };
    if num_str.is_empty() || !num_str.chars().all(|c| c.is_ascii_digit()) {
        return Vec::new();
    }
    let num: u32 = num_str.parse().unwrap_or(0);
    let mut siblings = Vec::new();
    for n in 1..=4u32 {
        if n != num {
            let mut sibling = url.clone();
            if sibling.set_host(Some(&format!("i{n}{domain}"))).is_ok() {
                siblings.push(sibling);
            }
        }
    }
    let mut canonical = url.clone();
    if canonical.set_host(Some(&format!("i{domain}"))).is_ok() {
        siblings.push(canonical);
    }
    siblings
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
    let partial_path = output_dir.join(format!(".page_{index:03}.{extension}.partial"));
    let mut reader = response.take(max_bytes + 1);
    let streamed = (|| -> Result<u64> {
        let mut output = File::create(&partial_path)
            .with_context(|| format!("failed to create {}", partial_path.display()))?;
        let bytes = std::io::copy(&mut reader, &mut output)
            .with_context(|| format!("failed to stream image bytes from {url}"))?;
        output
            .flush()
            .with_context(|| format!("failed to flush {}", partial_path.display()))?;
        Ok(bytes)
    })();
    let bytes = match streamed {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&partial_path);
            return Err(error);
        }
    };
    if bytes > max_bytes {
        let _ = fs::remove_file(&partial_path);
        anyhow::bail!("image exceeded {max_bytes} bytes");
    }
    // Decode-verify BEFORE committing: throttled CDN hosts sometimes return
    // 200 with an HTML error body or an undecodable variant (e.g. avif)
    // under an image/* content type. The caller's retry loop re-tries the
    // sibling CDN hosts when this fails, which usually gets the real image.
    // Magic-first (never trust the extension): CDNs routinely serve JPEG
    // content under .png URLs, which image::open would reject.
    image::ImageReader::open(&partial_path)
        .with_context(|| format!("failed to open downloaded image from {url}"))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format from {url}"))?
        .decode()
        .with_context(|| format!("undecodable image content from {url} ({bytes} bytes)"))?;
    if local_path.exists() {
        fs::remove_file(&local_path)
            .with_context(|| format!("failed to replace {}", local_path.display()))?;
    }
    fs::rename(&partial_path, &local_path).with_context(|| {
        format!(
            "failed to commit {} to {}",
            partial_path.display(),
            local_path.display()
        )
    })?;
    Ok(IngestedPage {
        index,
        source_url: url.to_string(),
        local_path,
        content_type,
        bytes,
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
            "http://127.0.0.2/page.png",
            "http://10.0.0.1/page.png",
            "http://172.16.0.1/page.png",
            "http://172.31.255.255/page.png",
            "http://192.168.1.1/page.png",
            "http://100.64.0.1/page.png",
            "http://100.127.255.254/page.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/page.png",
            "http://[::ffff:7f00:1]/page.png",
            "http://[::ffff:169.254.169.254]/page.png",
            "http://[2001:db8::1]/page.png",
            "http://[2001::1]/page.png",
            "http://[64:ff9b:0:0:0:0:a00:1]/page.png",
            "http://[64:ff9b:1::a00:1]/page.png",
            "http://[2002:0:0:0:0:0:7f00:1]/page.png",
            "http://0.0.0.0/page.png",
            "http://localhost/page.png",
            "http://localhost.localdomain/page.png",
            "file:///etc/passwd",
        ] {
            let url = Url::parse(value).unwrap();
            assert!(
                validate_public_url(&url).is_err(),
                "accepted unsafe URL {value}"
            );
        }
    }

    #[test]
    fn accepts_public_ingest_destinations() {
        for value in [
            "https://example.com/page.png",
            "http://example.org/chapter/1",
            "https://mangadex.org/chapter/abc-123",
        ] {
            let url = Url::parse(value).unwrap();
            assert!(
                validate_public_url(&url).is_ok(),
                "rejected public URL {value}"
            );
        }
    }
}
