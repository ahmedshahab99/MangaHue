use anyhow::{anyhow, Context, Result};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageFormat as ExternalFormat, Rgb, RgbImage,
};
use mangahue_core::{BoundingBox, ImageFormat, ImageSize, MangaPageInfo};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

#[derive(Debug, Clone)]
pub struct MangaImage {
    pub info: MangaPageInfo,
    pub image: DynamicImage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterImportOptions {
    pub max_pages: usize,
    pub max_bytes_per_page: u64,
    pub overwrite: bool,
}

impl Default for ChapterImportOptions {
    fn default() -> Self {
        Self {
            max_pages: 500,
            max_bytes_per_page: 50 * 1024 * 1024,
            overwrite: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterManifest {
    pub source: PathBuf,
    pub root: PathBuf,
    pub pages: Vec<ChapterPage>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterPage {
    pub index: usize,
    pub id: String,
    pub path: PathBuf,
    pub size: ImageSize,
    pub format: ImageFormat,
    pub bytes: u64,
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

pub fn import_chapter_to_dir(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: ChapterImportOptions,
) -> Result<ChapterManifest> {
    let input = input.as_ref();
    let output_dir = output_dir.as_ref();
    if options.overwrite && output_dir.exists() {
        fs::remove_dir_all(output_dir)
            .with_context(|| format!("failed to clear {}", output_dir.display()))?;
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    if input.is_dir() {
        import_chapter_folder(input, output_dir, &options)
    } else if is_archive_path(input) {
        import_chapter_archive(input, output_dir, &options)
    } else if is_supported_image_path(input) {
        import_single_page(input, output_dir, &options)
    } else {
        anyhow::bail!(
            "unsupported chapter input {}; expected folder, image, zip, or cbz",
            input.display()
        );
    }
}

pub fn load_chapter_from_folder(folder: impl AsRef<Path>) -> Result<ChapterManifest> {
    let folder = folder.as_ref();
    let mut warnings = Vec::new();
    let mut pages = Vec::new();
    let paths = collect_image_paths(folder)?;
    for (index, path) in paths.into_iter().enumerate() {
        match chapter_page_from_path(index + 1, &path) {
            Ok(page) => pages.push(page),
            Err(error) => warnings.push(format!("{}: {error}", path.display())),
        }
    }
    if pages.is_empty() {
        anyhow::bail!("no valid manga page images found in {}", folder.display());
    }
    Ok(ChapterManifest {
        source: folder.to_path_buf(),
        root: folder.to_path_buf(),
        pages,
        warnings,
    })
}

pub fn export_chapter_archive(
    input_dir: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
) -> Result<ChapterManifest> {
    let input_dir = input_dir.as_ref();
    let output_file = output_file.as_ref();
    let manifest = load_chapter_from_folder(input_dir)?;
    ensure_parent(output_file)?;
    let file = fs::File::create(output_file)
        .with_context(|| format!("failed to create {}", output_file.display()))?;
    let mut writer = ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);
    for page in &manifest.pages {
        let extension = page
            .path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("png");
        let archive_name = format!("page_{:03}.{extension}", page.index);
        writer.start_file(archive_name, options)?;
        let data = fs::read(&page.path)
            .with_context(|| format!("failed to read {}", page.path.display()))?;
        writer.write_all(&data)?;
    }
    writer.finish()?;
    Ok(manifest)
}

pub fn write_chapter_manifest(
    manifest: &ChapterManifest,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    ensure_parent(output_path)?;
    fs::write(output_path, serde_json::to_string_pretty(manifest)?)
        .with_context(|| format!("failed to write {}", output_path.display()))
}

pub fn collect_image_paths(root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let root = root.as_ref();
    let mut paths = Vec::new();
    collect_image_paths_recursive(root, &mut paths)?;
    natural_sort_paths(&mut paths);
    Ok(paths)
}

pub fn natural_sort_paths(paths: &mut [PathBuf]) {
    paths.sort_by(|a, b| natural_cmp(&path_sort_key(a), &path_sort_key(b)));
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

fn import_single_page(
    input: &Path,
    output_dir: &Path,
    options: &ChapterImportOptions,
) -> Result<ChapterManifest> {
    validate_file_size(input, options.max_bytes_per_page)?;
    let extension = normalized_image_extension(input).unwrap_or("png");
    let output = output_dir.join(format!("page_{:03}.{extension}", 1));
    fs::copy(input, &output)
        .with_context(|| format!("failed to copy {} to {}", input.display(), output.display()))?;
    let page = chapter_page_from_path(1, &output)?;
    Ok(ChapterManifest {
        source: input.to_path_buf(),
        root: output_dir.to_path_buf(),
        pages: vec![page],
        warnings: Vec::new(),
    })
}

fn import_chapter_folder(
    input: &Path,
    output_dir: &Path,
    options: &ChapterImportOptions,
) -> Result<ChapterManifest> {
    let paths = collect_image_paths(input)?;
    if paths.is_empty() {
        anyhow::bail!("no image pages found in {}", input.display());
    }
    if paths.len() > options.max_pages {
        anyhow::bail!(
            "chapter has {} pages, above limit {}",
            paths.len(),
            options.max_pages
        );
    }

    let mut pages = Vec::new();
    let mut warnings = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        let page_index = index + 1;
        if let Err(error) = validate_file_size(&path, options.max_bytes_per_page) {
            warnings.push(format!("{} skipped: {error}", path.display()));
            continue;
        }
        let extension = normalized_image_extension(&path).unwrap_or("png");
        let output = output_dir.join(format!("page_{page_index:03}.{extension}"));
        fs::copy(&path, &output).with_context(|| {
            format!("failed to copy {} to {}", path.display(), output.display())
        })?;
        match chapter_page_from_path(page_index, &output) {
            Ok(page) => pages.push(page),
            Err(error) => warnings.push(format!("{} skipped: {error}", path.display())),
        }
    }
    if pages.is_empty() {
        anyhow::bail!("no valid pages imported from {}", input.display());
    }
    Ok(ChapterManifest {
        source: input.to_path_buf(),
        root: output_dir.to_path_buf(),
        pages,
        warnings,
    })
}

fn import_chapter_archive(
    input: &Path,
    output_dir: &Path,
    options: &ChapterImportOptions,
) -> Result<ChapterManifest> {
    let file =
        fs::File::open(input).with_context(|| format!("failed to open {}", input.display()))?;
    let mut archive =
        ZipArchive::new(file).with_context(|| format!("failed to read {}", input.display()))?;
    let mut entries = Vec::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        if entry.is_dir() {
            continue;
        }
        let Some(name) = safe_archive_name(entry.name()) else {
            continue;
        };
        if is_supported_image_path(Path::new(&name)) {
            entries.push((index, name));
        }
    }
    entries.sort_by(|(_, a), (_, b)| natural_cmp(a, b));
    if entries.is_empty() {
        anyhow::bail!("no image pages found in {}", input.display());
    }
    if entries.len() > options.max_pages {
        anyhow::bail!(
            "archive has {} image pages, above limit {}",
            entries.len(),
            options.max_pages
        );
    }

    let mut pages = Vec::new();
    let mut warnings = Vec::new();
    for (page_index, (entry_index, name)) in entries.into_iter().enumerate() {
        let page_number = page_index + 1;
        let entry = archive.by_index(entry_index)?;
        if entry.size() > options.max_bytes_per_page {
            warnings.push(format!("{} skipped: image is too large", name));
            continue;
        }
        let mut data = Vec::new();
        let mut reader = entry.take(options.max_bytes_per_page + 1);
        reader
            .read_to_end(&mut data)
            .with_context(|| format!("failed to read {name} from {}", input.display()))?;
        if data.len() as u64 > options.max_bytes_per_page {
            warnings.push(format!("{} skipped: image exceeded byte limit", name));
            continue;
        }
        image::load_from_memory(&data)
            .with_context(|| format!("invalid image in archive: {name}"))?;
        let extension = normalized_image_extension(Path::new(&name)).unwrap_or("png");
        let output = output_dir.join(format!("page_{page_number:03}.{extension}"));
        fs::write(&output, data)
            .with_context(|| format!("failed to write {}", output.display()))?;
        match chapter_page_from_path(page_number, &output) {
            Ok(page) => pages.push(page),
            Err(error) => warnings.push(format!("{} skipped: {error}", name)),
        }
    }
    if pages.is_empty() {
        anyhow::bail!("no valid pages imported from {}", input.display());
    }
    Ok(ChapterManifest {
        source: input.to_path_buf(),
        root: output_dir.to_path_buf(),
        pages,
        warnings,
    })
}

fn chapter_page_from_path(index: usize, path: &Path) -> Result<ChapterPage> {
    let image = load_image(path)?;
    let bytes = fs::metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .len();
    Ok(ChapterPage {
        index,
        id: format!("page_{index:03}"),
        path: path.to_path_buf(),
        size: image.info.size,
        format: image.info.format,
        bytes,
    })
}

fn collect_image_paths_recursive(root: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_image_paths_recursive(&path, paths)?;
        } else if is_supported_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn validate_file_size(path: &Path, max_bytes: u64) -> Result<()> {
    let len = fs::metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .len();
    if len > max_bytes {
        anyhow::bail!(
            "{} is too large: {} bytes exceeds {}",
            path.display(),
            len,
            max_bytes
        );
    }
    Ok(())
}

fn is_archive_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("zip" | "cbz")
    )
}

pub fn is_supported_image_path(path: &Path) -> bool {
    normalized_image_extension(path).is_some()
}

fn normalized_image_extension(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Some("png"),
        Some("jpg" | "jpeg") => Some("jpg"),
        Some("webp") => Some("webp"),
        Some("tif" | "tiff") => Some("tif"),
        Some("bmp") => Some("bmp"),
        _ => None,
    }
}

fn safe_archive_name(name: &str) -> Option<String> {
    let normalized = name.replace('\\', "/");
    if normalized.starts_with('/') || normalized.contains("../") || normalized.contains("..\\") {
        return None;
    }
    let path = Path::new(&normalized);
    if path.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        return None;
    }
    Some(normalized)
}

fn path_sort_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut left = a.chars().peekable();
    let mut right = b.chars().peekable();

    loop {
        match (left.peek(), right.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(l), Some(r)) if l.is_ascii_digit() && r.is_ascii_digit() => {
                let left_number = take_number(&mut left);
                let right_number = take_number(&mut right);
                let ordering = compare_number_strings(&left_number, &right_number);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            (Some(_), Some(_)) => {
                let l = left.next().unwrap().to_ascii_lowercase();
                let r = right.next().unwrap().to_ascii_lowercase();
                let ordering = l.cmp(&r);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
        }
    }
}

fn take_number<I>(chars: &mut std::iter::Peekable<I>) -> String
where
    I: Iterator<Item = char>,
{
    let mut out = String::new();
    while let Some(ch) = chars.peek() {
        if ch.is_ascii_digit() {
            out.push(*ch);
            chars.next();
        } else {
            break;
        }
    }
    out
}

fn compare_number_strings(left: &str, right: &str) -> Ordering {
    let left_trimmed = left.trim_start_matches('0');
    let right_trimmed = right.trim_start_matches('0');
    let left_cmp = if left_trimmed.is_empty() {
        "0"
    } else {
        left_trimmed
    };
    let right_cmp = if right_trimmed.is_empty() {
        "0"
    } else {
        right_trimmed
    };
    left_cmp
        .len()
        .cmp(&right_cmp.len())
        .then_with(|| left_cmp.cmp(right_cmp))
        .then_with(|| left.len().cmp(&right.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn natural_sort_orders_pages_like_humans() {
        let mut paths = vec![
            PathBuf::from("page10.png"),
            PathBuf::from("page2.png"),
            PathBuf::from("page1.png"),
            PathBuf::from("page001.png"),
        ];
        natural_sort_paths(&mut paths);
        assert_eq!(
            paths,
            vec![
                PathBuf::from("page1.png"),
                PathBuf::from("page001.png"),
                PathBuf::from("page2.png"),
                PathBuf::from("page10.png")
            ]
        );
    }

    #[test]
    fn imports_folder_with_normalized_page_names() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let output = dir.path().join("imported");
        fs::create_dir_all(&source).unwrap();
        write_test_png(&source.join("page10.png"), [10, 0, 0]);
        write_test_png(&source.join("page2.png"), [2, 0, 0]);
        write_test_png(&source.join("page1.png"), [1, 0, 0]);

        let manifest =
            import_chapter_to_dir(&source, &output, ChapterImportOptions::default()).unwrap();
        assert_eq!(manifest.pages.len(), 3);
        assert_eq!(manifest.pages[0].path.file_name().unwrap(), "page_001.png");
        assert_eq!(manifest.pages[1].path.file_name().unwrap(), "page_002.png");
        assert_eq!(manifest.pages[2].path.file_name().unwrap(), "page_003.png");
        assert_eq!(manifest.pages[0].size, ImageSize::new(8, 8));
    }

    #[test]
    fn exports_and_imports_cbz_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("colored");
        let archive = dir.path().join("chapter.cbz");
        let imported = dir.path().join("imported");
        fs::create_dir_all(&source).unwrap();
        write_test_png(&source.join("page1.png"), [1, 2, 3]);
        write_test_png(&source.join("page2.png"), [4, 5, 6]);

        let exported = export_chapter_archive(&source, &archive).unwrap();
        assert_eq!(exported.pages.len(), 2);
        assert!(archive.exists());

        let imported_manifest =
            import_chapter_to_dir(&archive, &imported, ChapterImportOptions::default()).unwrap();
        assert_eq!(imported_manifest.pages.len(), 2);
        assert_eq!(
            imported_manifest.pages[0].path.file_name().unwrap(),
            "page_001.png"
        );
    }

    fn write_test_png(path: &Path, color: [u8; 3]) {
        let image = RgbImage::from_pixel(8, 8, Rgb(color));
        image.save(path).unwrap();
    }
}
