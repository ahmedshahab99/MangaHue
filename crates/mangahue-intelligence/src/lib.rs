use anyhow::{Context, Result};
use image::{GenericImageView, GrayImage, Luma, RgbImage};
use mangahue_brain::{ChapterBrainReport, ChapterSafetyReport};
use mangahue_core::{BoundingBox, ColorRgb, ImageSize};
use mangahue_hints::{
    RegionCoordinateSpace, RegionLock, RegionLockFile, RegionLockMode, RegionSemantic, RegionShape,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticClass {
    Skin,
    Hair,
    Clothing,
    Background,
    Speech,
    Effect,
    Unknown,
}

impl SemanticClass {
    fn region_semantic(self) -> RegionSemantic {
        match self {
            Self::Skin => RegionSemantic::Skin,
            Self::Hair => RegionSemantic::Hair,
            Self::Clothing => RegionSemantic::Clothing,
            Self::Background => RegionSemantic::Background,
            Self::Speech => RegionSemantic::Speech,
            Self::Effect => RegionSemantic::Effect,
            Self::Unknown => RegionSemantic::Custom,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegion {
    pub id: String,
    pub class: SemanticClass,
    pub bounds: BoundingBox,
    pub confidence: f32,
    pub coverage: f32,
    pub mean_color: ColorRgb,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPageReport {
    pub page_id: String,
    pub input_path: PathBuf,
    pub size: ImageSize,
    pub regions: Vec<SemanticRegion>,
    pub class_coverage: HashMap<SemanticClass, f32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysisConfig {
    pub min_component_pixels: u64,
    pub max_regions_per_class: usize,
    pub color_lock_strength: f32,
}

impl Default for SemanticAnalysisConfig {
    fn default() -> Self {
        Self {
            min_component_pixels: 96,
            max_regions_per_class: 24,
            color_lock_strength: 0.72,
        }
    }
}

pub fn analyze_semantics(
    path: impl AsRef<Path>,
    config: &SemanticAnalysisConfig,
) -> Result<SemanticPageReport> {
    let path = path.as_ref();
    let image = image::open(path)
        .with_context(|| format!("failed to open semantic source {}", path.display()))?;
    let (width, height) = image.dimensions();
    anyhow::ensure!(width > 0 && height > 0, "semantic source is empty");
    let rgb = image.to_rgb8();
    let size = ImageSize::new(width, height);
    let mut masks: HashMap<SemanticClass, GrayImage> = [
        SemanticClass::Skin,
        SemanticClass::Hair,
        SemanticClass::Clothing,
        SemanticClass::Background,
        SemanticClass::Speech,
        SemanticClass::Effect,
    ]
    .into_iter()
    .map(|class| (class, GrayImage::from_pixel(width, height, Luma([0]))))
    .collect();

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
        let max = r.max(g).max(b) as f32;
        let min = r.min(g).min(b) as f32;
        let saturation = if max <= 1.0 { 0.0 } else { (max - min) / max };
        let cb = 128.0 - 0.168736 * r as f32 - 0.331264 * g as f32 + 0.5 * b as f32;
        let cr = 128.0 + 0.5 * r as f32 - 0.418688 * g as f32 - 0.081312 * b as f32;
        let border = x < width / 30
            || y < height / 30
            || x + width / 30 >= width
            || y + height / 30 >= height;
        let class = if luma > 0.91 && saturation < 0.09 {
            if border {
                SemanticClass::Background
            } else {
                SemanticClass::Speech
            }
        } else if (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr) && luma > 0.18 {
            SemanticClass::Skin
        } else if luma < 0.24 && y < height.saturating_mul(3) / 5 {
            SemanticClass::Hair
        } else if saturation > 0.22 && y > height / 4 {
            SemanticClass::Clothing
        } else if saturation > 0.48 && luma > 0.55 {
            SemanticClass::Effect
        } else if border {
            SemanticClass::Background
        } else {
            continue;
        };
        masks.get_mut(&class).unwrap().put_pixel(x, y, Luma([255]));
    }

    let mut regions = Vec::new();
    let mut class_coverage = HashMap::new();
    let total_pixels = width as u64 * height as u64;
    for (class, mask) in &masks {
        let active = mask.pixels().filter(|pixel| pixel[0] > 0).count() as u64;
        class_coverage.insert(*class, active as f32 / total_pixels.max(1) as f32);
        let mut components = mask_components(mask, config.min_component_pixels);
        components.sort_by_key(|component| std::cmp::Reverse(component.1));
        components.truncate(config.max_regions_per_class);
        for (bounds, pixels) in components {
            regions.push(SemanticRegion {
                id: format!("{:?}_{}", class, Uuid::new_v4()).to_ascii_lowercase(),
                class: *class,
                bounds,
                confidence: semantic_confidence(*class, bounds, size, pixels),
                coverage: pixels as f32 / total_pixels.max(1) as f32,
                mean_color: mean_color(&rgb, bounds),
            });
        }
    }
    regions.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut warnings = Vec::new();
    if class_coverage
        .get(&SemanticClass::Skin)
        .copied()
        .unwrap_or(0.0)
        < 0.002
    {
        warnings.push(
            "no reliable skin regions; grayscale pages require the trained semantic model"
                .to_string(),
        );
    }
    Ok(SemanticPageReport {
        page_id: path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page")
            .to_string(),
        input_path: path.to_path_buf(),
        size,
        regions,
        class_coverage,
        warnings,
    })
}

pub fn semantic_region_locks(
    report: &SemanticPageReport,
    palette: &HashMap<SemanticClass, ColorRgb>,
    config: &SemanticAnalysisConfig,
) -> RegionLockFile {
    let locks = report
        .regions
        .iter()
        .filter_map(|region| {
            let color = palette.get(&region.class).copied();
            let mode = if region.class == SemanticClass::Speech {
                // The pixel classifier marks ALL near-white interior pixels
                // as Speech; on a manga page that is usually the whole paper
                // background, not speech bubbles. A Preserve lock larger
                // than half the page would gray out the page when enforced,
                // so only bubble-sized regions are preserved.
                if region.coverage > 0.5 {
                    return None;
                }
                RegionLockMode::Preserve
            } else if color.is_some() {
                RegionLockMode::Colorize
            } else {
                return None;
            };
            Some(RegionLock {
                id: region.id.clone(),
                label: Some(format!("auto_{:?}", region.class).to_ascii_lowercase()),
                semantic: region.class.region_semantic(),
                mode,
                region: RegionShape::Rectangle {
                    x: region.bounds.x as f32 / report.size.width as f32,
                    y: region.bounds.y as f32 / report.size.height as f32,
                    width: region.bounds.width as f32 / report.size.width as f32,
                    height: region.bounds.height as f32 / report.size.height as f32,
                },
                color,
                strength: (config.color_lock_strength * region.confidence).clamp(0.15, 1.0),
                feather_radius: 4.0,
                enabled: true,
            })
        })
        .collect();
    RegionLockFile {
        version: 1,
        page_id: Some(report.page_id.clone()),
        coordinate_space: RegionCoordinateSpace::Normalized,
        source_width: Some(report.size.width),
        source_height: Some(report.size.height),
        locks,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterObservation {
    pub page_id: String,
    pub region_id: String,
    pub bounds: BoundingBox,
    pub embedding: Vec<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterTrack {
    pub id: String,
    pub observations: Vec<CharacterObservation>,
    pub centroid: Vec<f32>,
    pub confidence: f32,
}

pub fn character_observations(report: &SemanticPageReport) -> Vec<CharacterObservation> {
    let skin = report
        .regions
        .iter()
        .filter(|region| region.class == SemanticClass::Skin);
    skin.map(|region| CharacterObservation {
        page_id: report.page_id.clone(),
        region_id: region.id.clone(),
        bounds: region.bounds,
        embedding: region_embedding(region, report.size),
        confidence: region.confidence,
    })
    .collect()
}

pub fn track_characters(
    observations: &[CharacterObservation],
    similarity_threshold: f32,
) -> Vec<CharacterTrack> {
    let mut tracks: Vec<CharacterTrack> = Vec::new();
    for observation in observations.iter().cloned() {
        let best = tracks
            .iter()
            .enumerate()
            .map(|(index, track)| (index, cosine(&track.centroid, &observation.embedding)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((index, similarity)) = best.filter(|(_, value)| *value >= similarity_threshold)
        {
            tracks[index].observations.push(observation);
            tracks[index].centroid = mean_embedding(&tracks[index].observations);
            tracks[index].confidence =
                (tracks[index].confidence * 0.7 + similarity * 0.3).clamp(0.0, 1.0);
        } else {
            tracks.push(CharacterTrack {
                id: format!("character_{:03}", tracks.len() + 1),
                centroid: observation.embedding.clone(),
                confidence: observation.confidence,
                observations: vec![observation],
            });
        }
    }
    tracks.sort_by_key(|track| std::cmp::Reverse(track.observations.len()));
    tracks
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferMatch {
    pub source_character_id: String,
    pub target_region_id: String,
    pub similarity: f32,
    pub accepted: bool,
}

pub fn match_cross_chapter(
    source_tracks: &[CharacterTrack],
    target_observations: &[CharacterObservation],
    threshold: f32,
) -> Vec<TransferMatch> {
    target_observations
        .iter()
        .filter_map(|target| {
            source_tracks
                .iter()
                .map(|track| (track, cosine(&track.centroid, &target.embedding)))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|(track, similarity)| TransferMatch {
                    source_character_id: track.id.clone(),
                    target_region_id: target.region_id.clone(),
                    similarity,
                    accepted: similarity >= threshold,
                })
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratorCandidate {
    pub page_id: String,
    pub lineart: PathBuf,
    pub color: PathBuf,
    pub correction_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedSample {
    pub candidate: CuratorCandidate,
    pub score: f32,
    pub split: DatasetSplit,
    pub safety_score: f32,
    pub consistency_score: f32,
    pub correction_value: f32,
    pub reasons: Vec<String>,
}

pub fn curate_dataset(
    candidates: Vec<CuratorCandidate>,
    brain: &ChapterBrainReport,
    safety: &ChapterSafetyReport,
) -> Vec<CuratedSample> {
    let brain_pages: HashMap<usize, _> =
        brain.pages.iter().map(|page| (page.index, page)).collect();
    let safety_pages: HashMap<usize, _> =
        safety.pages.iter().map(|page| (page.index, page)).collect();
    let mut samples = candidates
        .into_iter()
        .map(|candidate| {
            let index = page_number(&candidate.page_id);
            let brain_page = index.and_then(|value| brain_pages.get(&value).copied());
            let safety_page = index.and_then(|value| safety_pages.get(&value).copied());
            let safety_score = safety_page
                .map(|page| 1.0 - page.severity)
                .unwrap_or(0.25)
                .clamp(0.0, 1.0);
            let consistency_score = brain_page
                .map(|page| page.consistency_score)
                .unwrap_or(0.25)
                .clamp(0.0, 1.0);
            let correction_value = (candidate.correction_count as f32 / 8.0).clamp(0.0, 1.0);
            let hint_score = safety_page
                .map(|page| page.hint_adherence_score)
                .unwrap_or(0.4)
                .clamp(0.0, 1.0);
            let score = safety_score * 0.38
                + consistency_score * 0.27
                + hint_score * 0.2
                + correction_value * 0.15;
            let mut reasons = Vec::new();
            if safety_score < 0.55 {
                reasons.push("safety risk".to_string());
            }
            if consistency_score < 0.55 {
                reasons.push("chapter drift".to_string());
            }
            if correction_value > 0.5 {
                reasons.push("high correction information value".to_string());
            }
            let split = if score < 0.48 {
                DatasetSplit::Reject
            } else {
                deterministic_split(&candidate.page_id)
            };
            CuratedSample {
                candidate,
                score,
                split,
                safety_score,
                consistency_score,
                correction_value,
                reasons,
            }
        })
        .collect::<Vec<_>>();
    samples.sort_by(|left, right| right.score.total_cmp(&left.score));
    samples
}

fn mask_components(mask: &GrayImage, min_pixels: u64) -> Vec<(BoundingBox, u64)> {
    let (width, height) = mask.dimensions();
    let mut visited = vec![false; width as usize * height as usize];
    let mut output = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if visited[index] || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let mut queue = VecDeque::from([(x, y)]);
            visited[index] = true;
            let (mut min_x, mut min_y, mut max_x, mut max_y, mut pixels) = (x, y, x, y, 0u64);
            while let Some((cx, cy)) = queue.pop_front() {
                pixels += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let next = (ny as u32 * width + nx as u32) as usize;
                    if !visited[next] && mask.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        visited[next] = true;
                        queue.push_back((nx as u32, ny as u32));
                    }
                }
            }
            if pixels >= min_pixels {
                output.push((
                    BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                ));
            }
        }
    }
    output
}

fn semantic_confidence(
    class: SemanticClass,
    bounds: BoundingBox,
    size: ImageSize,
    pixels: u64,
) -> f32 {
    let density = pixels as f32 / (bounds.area().max(1) as f32);
    let scale = bounds.area() as f32 / (size.area().max(1) as f32);
    let base = match class {
        SemanticClass::Skin => 0.68,
        SemanticClass::Speech => 0.72,
        SemanticClass::Background => 0.7,
        _ => 0.55,
    };
    (base * 0.6 + density * 0.3 + scale.min(0.2) * 0.5).clamp(0.15, 0.96)
}

fn mean_color(image: &RgbImage, bounds: BoundingBox) -> ColorRgb {
    let bounds = bounds.clamp_to(ImageSize::new(image.width(), image.height()));
    let (mut r, mut g, mut b, mut count) = (0u64, 0u64, 0u64, 0u64);
    let stride = ((bounds.area() / 2048).max(1)) as usize;
    for (index, y) in (bounds.y..bounds.bottom()).enumerate() {
        if !index.is_multiple_of(stride) {
            continue;
        }
        for x in (bounds.x..bounds.right()).step_by(stride) {
            let pixel = image.get_pixel(x, y);
            r += pixel[0] as u64;
            g += pixel[1] as u64;
            b += pixel[2] as u64;
            count += 1;
        }
    }
    ColorRgb::new(
        (r / count.max(1)) as u8,
        (g / count.max(1)) as u8,
        (b / count.max(1)) as u8,
    )
}

fn region_embedding(region: &SemanticRegion, size: ImageSize) -> Vec<f32> {
    vec![
        region.mean_color.r as f32 / 255.0,
        region.mean_color.g as f32 / 255.0,
        region.mean_color.b as f32 / 255.0,
        region.bounds.x as f32 / size.width.max(1) as f32,
        region.bounds.y as f32 / size.height.max(1) as f32,
        region.bounds.width as f32 / size.width.max(1) as f32,
        region.bounds.height as f32 / size.height.max(1) as f32,
        region.confidence,
    ]
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let l = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let r = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    if l <= f32::EPSILON || r <= f32::EPSILON {
        0.0
    } else {
        (dot / (l * r)).clamp(0.0, 1.0)
    }
}

fn mean_embedding(observations: &[CharacterObservation]) -> Vec<f32> {
    let length = observations
        .first()
        .map(|item| item.embedding.len())
        .unwrap_or(0);
    let mut mean = vec![0.0; length];
    for observation in observations {
        for (index, value) in observation.embedding.iter().enumerate() {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= observations.len().max(1) as f32;
    }
    mean
}

fn page_number(page_id: &str) -> Option<usize> {
    page_id
        .split(|ch: char| !ch.is_ascii_digit())
        .rfind(|part| !part.is_empty())?
        .parse()
        .ok()
}

fn deterministic_split(page_id: &str) -> DatasetSplit {
    let digest = Sha256::digest(page_id.as_bytes());
    match digest[0] % 20 {
        0 => DatasetSplit::Test,
        1 | 2 => DatasetSplit::Validation,
        _ => DatasetSplit::Train,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn semantics_generate_regions_and_locks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("page_001.png");
        let mut image = RgbImage::from_pixel(128, 128, Rgb([245, 245, 245]));
        for y in 30..90 {
            for x in 35..90 {
                image.put_pixel(x, y, Rgb([210, 150, 120]));
            }
        }
        image.save(&path).unwrap();
        let report = analyze_semantics(
            &path,
            &SemanticAnalysisConfig {
                min_component_pixels: 20,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report
            .regions
            .iter()
            .any(|region| region.class == SemanticClass::Skin));
        let palette = HashMap::from([(SemanticClass::Skin, ColorRgb::new(220, 160, 130))]);
        assert!(
            !semantic_region_locks(&report, &palette, &Default::default())
                .locks
                .is_empty()
        );
    }

    #[test]
    fn character_tracking_clusters_similar_observations() {
        let observation = |page: &str, color: f32| CharacterObservation {
            page_id: page.to_string(),
            region_id: page.to_string(),
            bounds: BoundingBox::new(0, 0, 10, 10),
            embedding: vec![color, 0.5, 0.4],
            confidence: 0.8,
        };
        let tracks = track_characters(
            &[
                observation("a", 0.8),
                observation("b", 0.79),
                observation("c", 0.1),
            ],
            0.94,
        );
        assert_eq!(tracks[0].observations.len(), 2);
    }
}
