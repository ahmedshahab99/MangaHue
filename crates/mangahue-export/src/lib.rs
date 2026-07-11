use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageEncoder, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WatermarkMode {
    None,
    Forensic,
    Preview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProfile {
    pub id: String,
    pub max_dimension: u32,
    pub format: ExportFormat,
    pub quality: u8,
    pub watermark: WatermarkMode,
    pub preserve_resolution: bool,
}

impl ExportProfile {
    pub fn free_preview() -> Self {
        Self {
            id: "free_preview".to_string(),
            max_dimension: 1600,
            format: ExportFormat::Jpeg,
            quality: 82,
            watermark: WatermarkMode::Preview,
            preserve_resolution: false,
        }
    }

    pub fn creator() -> Self {
        Self {
            id: "creator".to_string(),
            max_dimension: 4096,
            format: ExportFormat::Png,
            quality: 95,
            watermark: WatermarkMode::Forensic,
            preserve_resolution: false,
        }
    }

    pub fn studio_master() -> Self {
        Self {
            id: "studio_master".to_string(),
            max_dimension: u32::MAX,
            format: ExportFormat::Png,
            quality: 100,
            watermark: WatermarkMode::None,
            preserve_resolution: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub input: PathBuf,
    pub output: PathBuf,
    pub profile_id: String,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub sha256: String,
    pub watermark_id: Option<String>,
}

pub fn render_export(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    profile: &ExportProfile,
    account_id: Option<&str>,
) -> Result<ExportResult> {
    let input = input.as_ref();
    let output = output.as_ref();
    let image = image::open(input)
        .with_context(|| format!("failed to open export source {}", input.display()))?;
    let mut rgb = resize_for_profile(image, profile).to_rgb8();
    let watermark_id = match profile.watermark {
        WatermarkMode::None => None,
        mode => {
            let id = watermark_identity(account_id.unwrap_or("anonymous"), input);
            apply_watermark(&mut rgb, &id, mode);
            Some(id)
        }
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = encode(&rgb, profile)?;
    fs::write(output, &bytes)?;
    Ok(ExportResult {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        profile_id: profile.id.clone(),
        width: rgb.width(),
        height: rgb.height(),
        bytes: bytes.len() as u64,
        sha256: hex_sha256(&bytes),
        watermark_id,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportBundleManifest {
    pub id: String,
    pub profile: ExportProfile,
    pub pages: Vec<ExportResult>,
}

pub fn export_bundle(
    inputs: &[PathBuf],
    output: impl AsRef<Path>,
    profile: &ExportProfile,
    account_id: Option<&str>,
) -> Result<ExportBundleManifest> {
    let output = output.as_ref();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary_dir = tempfile_path(output);
    fs::create_dir_all(&temporary_dir)?;
    let mut pages = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let extension = match profile.format {
            ExportFormat::Png => "png",
            ExportFormat::Jpeg => "jpg",
            ExportFormat::Webp => "webp",
        };
        let page_path = temporary_dir.join(format!("page_{:03}.{extension}", index + 1));
        pages.push(render_export(input, &page_path, profile, account_id)?);
    }
    let manifest = ExportBundleManifest {
        id: Uuid::new_v4().to_string(),
        profile: profile.clone(),
        pages,
    };
    let file = fs::File::create(output)?;
    let mut writer = ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);
    for page in &manifest.pages {
        let name = page
            .output
            .file_name()
            .and_then(|value| value.to_str())
            .context("invalid page name")?;
        writer.start_file(name, options)?;
        writer.write_all(&fs::read(&page.output)?)?;
    }
    writer.start_file("mangahue_export_manifest.json", options)?;
    writer.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    writer.finish()?;
    fs::remove_dir_all(&temporary_dir).ok();
    Ok(manifest)
}

fn resize_for_profile(image: DynamicImage, profile: &ExportProfile) -> DynamicImage {
    if profile.preserve_resolution {
        return image;
    }
    let (width, height) = image.dimensions();
    if width.max(height) <= profile.max_dimension {
        image
    } else {
        image.resize(
            profile.max_dimension,
            profile.max_dimension,
            image::imageops::FilterType::Lanczos3,
        )
    }
}

fn encode(image: &RgbImage, profile: &ExportProfile) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    match profile.format {
        ExportFormat::Png => {
            image::codecs::png::PngEncoder::new(&mut output).write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )?;
        }
        ExportFormat::Jpeg => {
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, profile.quality)
                .encode_image(image)?;
        }
        ExportFormat::Webp => {
            image::codecs::webp::WebPEncoder::new_lossless(&mut output).encode(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )?;
        }
    }
    Ok(output)
}

fn apply_watermark(image: &mut RgbImage, identity: &str, mode: WatermarkMode) {
    let digest = Sha256::digest(identity.as_bytes());
    let cell = (image.width().min(image.height()) / 96).clamp(2, 12);
    let spacing = cell * 7;
    for gy in (spacing / 2..image.height()).step_by(spacing as usize) {
        for gx in (spacing / 2..image.width()).step_by(spacing as usize) {
            let bit = digest[((gx / spacing + gy / spacing) as usize) % digest.len()] & 1;
            if bit == 0 && mode == WatermarkMode::Forensic {
                continue;
            }
            for y in gy.saturating_sub(cell)..(gy + cell).min(image.height()) {
                for x in gx.saturating_sub(cell)..(gx + cell).min(image.width()) {
                    let diagonal = (x as i64 - gx as i64).abs() == (y as i64 - gy as i64).abs();
                    if !diagonal && mode == WatermarkMode::Forensic {
                        continue;
                    }
                    let pixel = *image.get_pixel(x, y);
                    let target = if mode == WatermarkMode::Preview {
                        Rgb([255, 255, 255])
                    } else {
                        Rgb([12, 12, 12])
                    };
                    let alpha = if mode == WatermarkMode::Preview {
                        0.16
                    } else {
                        0.035
                    };
                    image.put_pixel(x, y, blend(pixel, target, alpha));
                }
            }
        }
    }
    if mode == WatermarkMode::Preview {
        let band_height = (image.height() / 18).max(8);
        let start = image.height() / 2 - band_height / 2;
        for y in start..(start + band_height).min(image.height()) {
            for x in 0..image.width() {
                let pixel = *image.get_pixel(x, y);
                image.put_pixel(x, y, blend(pixel, Rgb([255, 255, 255]), 0.13));
            }
        }
    }
}

fn blend(a: Rgb<u8>, b: Rgb<u8>, alpha: f32) -> Rgb<u8> {
    Rgb([
        (a[0] as f32 * (1.0 - alpha) + b[0] as f32 * alpha).round() as u8,
        (a[1] as f32 * (1.0 - alpha) + b[1] as f32 * alpha).round() as u8,
        (a[2] as f32 * (1.0 - alpha) + b[2] as f32 * alpha).round() as u8,
    ])
}

fn watermark_identity(account_id: &str, path: &Path) -> String {
    hex_sha256(format!("{account_id}:{}", path.display()).as_bytes())[..24].to_string()
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn tempfile_path(output: &Path) -> PathBuf {
    output.with_extension(format!("work-{}", Uuid::new_v4()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_export_resizes_and_watermarks() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let output = dir.path().join("output.jpg");
        RgbImage::from_pixel(200, 300, Rgb([40, 80, 120]))
            .save(&input)
            .unwrap();
        let mut profile = ExportProfile::free_preview();
        profile.max_dimension = 100;
        let result = render_export(&input, &output, &profile, Some("user-1")).unwrap();
        assert_eq!(result.height, 100);
        assert!(result.watermark_id.is_some());
        assert!(output.is_file());
    }
}
