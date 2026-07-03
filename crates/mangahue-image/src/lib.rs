use anyhow::{anyhow, Context, Result};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageFormat as ExternalFormat, Rgb, RgbImage,
};
use mangahue_core::{BoundingBox, ImageFormat, ImageSize, MangaPageInfo};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct MangaImage {
    pub info: MangaPageInfo,
    pub image: DynamicImage,
}

pub fn load_image(path: impl AsRef<Path>) -> Result<MangaImage> {
    let path = path.as_ref();
    let image = image::open(path).with_context(|| format!("failed to load {}", path.display()))?;
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(anyhow!(
            "image has zero width or height: {}",
            path.display()
        ));
    }
    let size = ImageSize::new(width, height);
    let format = format_from_path(path);
    let id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("page")
        .to_string();
    let info = MangaPageInfo {
        id,
        path: path.to_path_buf(),
        size,
        aspect_ratio: size.aspect_ratio(),
        format,
        is_webtoon_like: height > width.saturating_mul(3),
    };
    Ok(MangaImage { info, image })
}

pub fn save_rgb(path: impl AsRef<Path>, image: &RgbImage) -> Result<()> {
    let path = path.as_ref();
    ensure_parent(path)?;
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

pub fn save_gray(path: impl AsRef<Path>, image: &GrayImage) -> Result<()> {
    let path = path.as_ref();
    ensure_parent(path)?;
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

pub fn save_dynamic(path: impl AsRef<Path>, image: &DynamicImage) -> Result<()> {
    let path = path.as_ref();
    ensure_parent(path)?;
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

pub fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

pub fn blank_rgb(size: ImageSize, color: mangahue_core::ColorRgb) -> RgbImage {
    RgbImage::from_pixel(size.width, size.height, Rgb([color.r, color.g, color.b]))
}

pub fn blank_gray(size: ImageSize, value: u8) -> GrayImage {
    GrayImage::from_pixel(size.width, size.height, image::Luma([value]))
}

pub fn crop_rgb(image: &RgbImage, bbox: BoundingBox) -> RgbImage {
    let safe = bbox.clamp_to(ImageSize::new(image.width(), image.height()));
    image::imageops::crop_imm(image, safe.x, safe.y, safe.width, safe.height).to_image()
}

pub fn crop_gray(image: &GrayImage, bbox: BoundingBox) -> GrayImage {
    let safe = bbox.clamp_to(ImageSize::new(image.width(), image.height()));
    image::imageops::crop_imm(image, safe.x, safe.y, safe.width, safe.height).to_image()
}

pub fn paste_rgb(target: &mut RgbImage, source: &RgbImage, x: u32, y: u32) {
    image::imageops::replace(target, source, x.into(), y.into());
}

pub fn resize_to_max(image: &DynamicImage, max_dimension: u32) -> DynamicImage {
    if max_dimension == 0 {
        return image.clone();
    }
    let (width, height) = image.dimensions();
    let longest = width.max(height);
    if longest <= max_dimension {
        return image.clone();
    }
    let scale = max_dimension as f32 / longest as f32;
    let new_width = (width as f32 * scale).round().max(1.0) as u32;
    let new_height = (height as f32 * scale).round().max(1.0) as u32;
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

pub fn format_from_path(path: &Path) -> ImageFormat {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => ImageFormat::Png,
        Some("jpg" | "jpeg") => ImageFormat::Jpeg,
        Some("webp") => ImageFormat::Webp,
        Some("tif" | "tiff") => ImageFormat::Tiff,
        Some("bmp") => ImageFormat::Bmp,
        _ => ImageFormat::Unknown,
    }
}

pub fn external_format_from_path(path: &Path) -> Option<ExternalFormat> {
    match format_from_path(path) {
        ImageFormat::Png => Some(ExternalFormat::Png),
        ImageFormat::Jpeg => Some(ExternalFormat::Jpeg),
        ImageFormat::Webp => Some(ExternalFormat::WebP),
        ImageFormat::Tiff => Some(ExternalFormat::Tiff),
        ImageFormat::Bmp => Some(ExternalFormat::Bmp),
        ImageFormat::Unknown => None,
    }
}

pub fn sibling_with_suffix(path: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    parent.join(format!("{stem}{suffix}.{extension}"))
}
