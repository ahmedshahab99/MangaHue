use anyhow::{Context, Result};
use image::{GenericImageView, GrayImage, Luma, RgbImage};
use mangahue_brain::{ChapterBrainReport, ChapterSafetyReport};
use mangahue_core::{BoundingBox, ColorRgb, ImageSize, MangaPanel};
use mangahue_hints::{
    RegionCoordinateSpace, RegionLock, RegionLockFile, RegionLockMode, RegionRun, RegionSemantic,
    RegionShape, SemanticMasks,
};
use mangahue_memory::{ColorBible, ColorContext, ColorSelector, NormalizedRegion};
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

#[derive(Debug)]
struct MaskComponent {
    bounds: BoundingBox,
    pixels: u64,
    runs: Vec<RegionRun>,
}

fn mask_components_with_runs(mask: &GrayImage, min_pixels: u64) -> Vec<MaskComponent> {
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
            let mut points = Vec::new();
            visited[index] = true;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (x, y, x, y);
            while let Some((cx, cy)) = queue.pop_front() {
                points.push((cx, cy));
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
            if points.len() as u64 >= min_pixels {
                points.sort_unstable_by_key(|(px, py)| (*py, *px));
                let mut runs: Vec<RegionRun> = Vec::new();
                for (px, py) in points.iter().copied() {
                    if let Some(last) = runs.last_mut() {
                        if last.y == py && last.x_end.saturating_add(1) == px {
                            last.x_end = px;
                            continue;
                        }
                    }
                    runs.push(RegionRun {
                        y: py,
                        x_start: px,
                        x_end: px,
                    });
                }
                output.push(MaskComponent {
                    bounds: BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels: points.len() as u64,
                    runs,
                });
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

pub const STRUCTURE_GRAPH_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureNodeKind {
    Panel,
    SpeechBubble,
    Text,
    Character,
    Skin,
    Hair,
    Eyes,
    Clothing,
    Background,
    Effect,
    Protected,
    Object,
}

impl StructureNodeKind {
    pub fn semantic(self) -> Option<&'static str> {
        match self {
            Self::Skin => Some("skin"),
            Self::Hair => Some("hair"),
            Self::Eyes => Some("eyes"),
            Self::Clothing => Some("clothing"),
            Self::Background => Some("background"),
            Self::Effect => Some("effect"),
            Self::Object => Some("object"),
            Self::Panel | Self::SpeechBubble | Self::Text | Self::Character | Self::Protected => {
                None
            }
        }
    }

    pub fn correctable(self) -> bool {
        self.semantic().is_some()
    }

    pub fn protected(self) -> bool {
        matches!(self, Self::SpeechBubble | Self::Text | Self::Protected)
    }

    fn region_semantic(self) -> RegionSemantic {
        match self {
            Self::Skin => RegionSemantic::Skin,
            Self::Hair => RegionSemantic::Hair,
            Self::Clothing => RegionSemantic::Clothing,
            Self::Background => RegionSemantic::Background,
            Self::SpeechBubble | Self::Text | Self::Protected => RegionSemantic::Speech,
            Self::Effect => RegionSemantic::Effect,
            Self::Object => RegionSemantic::Prop,
            Self::Eyes | Self::Panel | Self::Character => RegionSemantic::Custom,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureNode {
    pub id: String,
    pub kind: StructureNodeKind,
    pub bounds: BoundingBox,
    pub normalized_bounds: NormalizedRegion,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_id: Option<String>,
    pub protected: bool,
    pub source: String,
    /// Exact page-resolution semantic/protected mask. Geometry-only nodes
    /// (panels and face estimates) leave this empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_runs: Vec<RegionRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureGraph {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chapter_id: Option<String>,
    pub page_id: String,
    pub input_path: PathBuf,
    pub size: ImageSize,
    pub lighting: String,
    pub nodes: Vec<StructureNode>,
    pub edges: Vec<StructureEdge>,
}

impl StructureGraph {
    pub fn node_at(&self, x: f32, y: f32) -> Option<&StructureNode> {
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return None;
        }
        self.nodes
            .iter()
            .filter(|node| {
                node.kind.correctable() && !node.protected && node.normalized_bounds.contains(x, y)
            })
            .min_by(|left, right| {
                left.bounds
                    .area()
                    .cmp(&right.bounds.area())
                    .then_with(|| right.confidence.total_cmp(&left.confidence))
            })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let graph: Self = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("failed to read structure graph {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse structure graph {}", path.display()))?;
        anyhow::ensure!(
            graph.schema_version == STRUCTURE_GRAPH_SCHEMA_VERSION,
            "unsupported structure graph schema {}",
            graph.schema_version
        );
        Ok(graph)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        std::fs::rename(&temporary, path)
            .with_context(|| format!("failed to commit structure graph {}", path.display()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_structure_graph(
    series_id: Option<String>,
    chapter_id: Option<String>,
    page_id: impl Into<String>,
    input_path: PathBuf,
    size: ImageSize,
    panels: &[MangaPanel],
    face_boxes: &[[u32; 4]],
    semantic_masks: &SemanticMasks,
    bubble_mask: &GrayImage,
    text_mask: &GrayImage,
    protected_mask: &GrayImage,
    lighting: impl Into<String>,
) -> Result<StructureGraph> {
    anyhow::ensure!(size.width > 0 && size.height > 0, "page is empty");
    anyhow::ensure!(
        bubble_mask.dimensions() == (size.width, size.height)
            && text_mask.dimensions() == (size.width, size.height)
            && protected_mask.dimensions() == (size.width, size.height),
        "structure masks do not match page dimensions"
    );
    let expected = size.width as usize * size.height as usize;
    anyhow::ensure!(
        [
            semantic_masks.skin.len(),
            semantic_masks.hair.len(),
            semantic_masks.clothing.len(),
            semantic_masks.background.len(),
            semantic_masks.bubble.len(),
        ]
        .into_iter()
        .all(|len| len == expected),
        "semantic masks do not match page dimensions"
    );

    let page_id = page_id.into();
    let mut nodes = panels
        .iter()
        .map(|panel| StructureNode {
            id: panel.id.clone(),
            kind: StructureNodeKind::Panel,
            bounds: panel.bbox.clamp_to(size),
            normalized_bounds: normalize_box(panel.bbox, size),
            confidence: panel.confidence,
            parent_id: None,
            panel_id: Some(panel.id.clone()),
            character_id: None,
            location_id: None,
            object_id: None,
            protected: false,
            source: "layout".to_string(),
            mask_runs: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut edges = Vec::new();

    for (index, face) in face_boxes.iter().enumerate() {
        let character_id = format!("character_{:03}", index + 1);
        let face_bounds = BoundingBox::new(face[0], face[1], face[2], face[3]).clamp_to(size);
        if face_bounds.area() == 0 {
            continue;
        }
        let character_node_id = format!("{}_{}", page_id, character_id);
        let panel_id = containing_panel(panels, face_bounds);
        nodes.push(StructureNode {
            id: character_node_id.clone(),
            kind: StructureNodeKind::Character,
            bounds: face_bounds,
            normalized_bounds: normalize_box(face_bounds, size),
            confidence: 0.82,
            parent_id: panel_id.clone(),
            panel_id: panel_id.clone(),
            character_id: Some(character_id.clone()),
            location_id: None,
            object_id: None,
            protected: false,
            source: "anime_face".to_string(),
            mask_runs: Vec::new(),
        });
        if let Some(panel_id) = panel_id {
            edges.push(StructureEdge {
                from: panel_id,
                to: character_node_id.clone(),
                relation: "contains".to_string(),
                confidence: 0.82,
            });
        }
        for (suffix, kind, bounds, confidence) in face_material_bounds(face_bounds, size) {
            let id = format!("{character_node_id}_{suffix}");
            nodes.push(StructureNode {
                id: id.clone(),
                kind,
                bounds,
                normalized_bounds: normalize_box(bounds, size),
                confidence,
                parent_id: Some(character_node_id.clone()),
                panel_id: containing_panel(panels, bounds),
                character_id: Some(character_id.clone()),
                location_id: None,
                object_id: None,
                protected: false,
                source: "anime_face_geometry".to_string(),
                mask_runs: Vec::new(),
            });
            edges.push(StructureEdge {
                from: character_node_id.clone(),
                to: id,
                relation: "has_material".to_string(),
                confidence,
            });
        }
    }

    add_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        &semantic_masks.hair,
        StructureNodeKind::Hair,
        "grayscale_semantics",
        48,
        24,
        face_boxes,
    );
    add_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        &semantic_masks.clothing,
        StructureNodeKind::Clothing,
        "grayscale_semantics",
        96,
        48,
        face_boxes,
    );
    add_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        &semantic_masks.background,
        StructureNodeKind::Background,
        "grayscale_semantics",
        128,
        32,
        face_boxes,
    );
    add_gray_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        bubble_mask,
        StructureNodeKind::SpeechBubble,
        "bubble_detector",
        128,
        64,
    );
    add_gray_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        text_mask,
        StructureNodeKind::Text,
        "text_detector",
        8,
        128,
    );
    add_gray_mask_nodes(
        &mut nodes,
        &mut edges,
        panels,
        size,
        protected_mask,
        StructureNodeKind::Protected,
        "protected_mask",
        32,
        128,
    );

    nodes.sort_by(|left, right| {
        node_order(left.kind)
            .cmp(&node_order(right.kind))
            .then_with(|| left.bounds.y.cmp(&right.bounds.y))
            .then_with(|| left.bounds.x.cmp(&right.bounds.x))
    });
    Ok(StructureGraph {
        schema_version: STRUCTURE_GRAPH_SCHEMA_VERSION,
        series_id,
        chapter_id,
        page_id,
        input_path,
        size,
        lighting: lighting.into(),
        nodes,
        edges,
    })
}

pub fn color_bible_region_locks(graph: &StructureGraph, bible: &ColorBible) -> RegionLockFile {
    let mut locks = Vec::new();
    for node in &graph.nodes {
        let shape = node_region_shape(node, graph.size);
        if node.kind.protected() {
            locks.push(RegionLock {
                id: format!("bible_protect_{}", node.id),
                label: Some(format!("protected_{:?}", node.kind).to_ascii_lowercase()),
                semantic: node.kind.region_semantic(),
                mode: RegionLockMode::Preserve,
                region: shape,
                color: None,
                strength: 1.0,
                feather_radius: 0.0,
                enabled: true,
            });
            locks.push(RegionLock {
                id: format!("bible_exclude_{}", node.id),
                label: Some(format!("exclude_{:?}", node.kind).to_ascii_lowercase()),
                semantic: node.kind.region_semantic(),
                mode: RegionLockMode::Exclude,
                region: node_region_shape(node, graph.size),
                color: None,
                strength: 1.0,
                feather_radius: 0.0,
                enabled: true,
            });
            continue;
        }
        let Some(semantic) = node.kind.semantic() else {
            continue;
        };
        let context = ColorContext {
            page_id: Some(&graph.page_id),
            chapter_id: graph.chapter_id.as_deref(),
            character_id: node.character_id.as_deref(),
            location_id: node.location_id.as_deref(),
            object_id: node.object_id.as_deref(),
            region_id: Some(&node.id),
            semantic: Some(semantic),
        };
        let Some(entry) = bible.resolve_color(&context) else {
            continue;
        };
        locks.push(RegionLock {
            id: format!("bible_color_{}", node.id),
            label: Some(format!("color_bible_{semantic}")),
            semantic: node.kind.region_semantic(),
            mode: RegionLockMode::Colorize,
            region: shape,
            color: Some(entry.color),
            strength: (node.confidence * entry.confidence).clamp(0.35, 1.0),
            feather_radius: 2.5,
            enabled: true,
        });
    }
    RegionLockFile {
        version: 1,
        page_id: Some(graph.page_id.clone()),
        coordinate_space: RegionCoordinateSpace::Normalized,
        source_width: Some(graph.size.width),
        source_height: Some(graph.size.height),
        locks,
    }
}

fn node_region_shape(node: &StructureNode, size: ImageSize) -> RegionShape {
    if node.mask_runs.is_empty() {
        RegionShape::Rectangle {
            x: node.normalized_bounds.x,
            y: node.normalized_bounds.y,
            width: node.normalized_bounds.width,
            height: node.normalized_bounds.height,
        }
    } else {
        RegionShape::MaskRle {
            source_width: size.width,
            source_height: size.height,
            runs: node.mask_runs.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConsistency {
    pub node_id: String,
    pub semantic: String,
    pub expected: ColorRgb,
    pub observed: ColorRgb,
    pub score: f32,
    pub confidence: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorBibleConsistencyReport {
    pub bible_revision: u64,
    pub evaluated_regions: usize,
    pub mean_score: f32,
    pub minimum_score: f32,
    pub threshold: f32,
    pub passed: bool,
    pub regions: Vec<RegionConsistency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionImpactPage {
    pub page_id: String,
    pub graph_path: PathBuf,
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionImpact {
    pub pages: Vec<CorrectionImpactPage>,
    pub affected_pages: usize,
    pub affected_regions: usize,
}

pub fn plan_correction_impact(
    graphs_root: impl AsRef<Path>,
    selector: &ColorSelector,
) -> Result<CorrectionImpact> {
    selector.validate()?;
    let mut paths = Vec::new();
    collect_graph_paths(graphs_root.as_ref(), &mut paths)?;
    paths.sort();
    let mut pages = Vec::new();
    for path in paths {
        let Ok(graph) = StructureGraph::load(&path) else {
            continue;
        };
        let region_ids = graph
            .nodes
            .iter()
            .filter(|node| node.kind.correctable())
            .filter(|node| {
                selector.matches_context(&ColorContext {
                    page_id: Some(&graph.page_id),
                    chapter_id: graph.chapter_id.as_deref(),
                    character_id: node.character_id.as_deref(),
                    location_id: node.location_id.as_deref(),
                    object_id: node.object_id.as_deref(),
                    region_id: Some(&node.id),
                    semantic: node.kind.semantic(),
                })
            })
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        if !region_ids.is_empty() {
            pages.push(CorrectionImpactPage {
                page_id: graph.page_id,
                graph_path: path,
                region_ids,
            });
        }
    }
    let affected_regions = pages.iter().map(|page| page.region_ids.len()).sum();
    Ok(CorrectionImpact {
        affected_pages: pages.len(),
        affected_regions,
        pages,
    })
}

fn collect_graph_paths(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        output.push(root.to_path_buf());
        return Ok(());
    }
    if !root.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("failed to read graph directory {}", root.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_graph_paths(&path, output)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("json") {
            output.push(path);
        }
    }
    Ok(())
}

pub fn verify_color_bible_consistency(
    output: &RgbImage,
    graph: &StructureGraph,
    bible: &ColorBible,
    line_mask: &GrayImage,
    protected_mask: &GrayImage,
    threshold: f32,
) -> ColorBibleConsistencyReport {
    let threshold = threshold.clamp(0.0, 1.0);
    let mut regions = Vec::new();
    for node in graph.nodes.iter().filter(|node| node.kind.correctable()) {
        let semantic = node.kind.semantic().unwrap_or("object");
        let context = ColorContext {
            page_id: Some(&graph.page_id),
            chapter_id: graph.chapter_id.as_deref(),
            character_id: node.character_id.as_deref(),
            location_id: node.location_id.as_deref(),
            object_id: node.object_id.as_deref(),
            region_id: Some(&node.id),
            semantic: Some(semantic),
        };
        let Some(entry) = bible.resolve_color(&context) else {
            continue;
        };
        let Some(observed) =
            mean_unprotected_color(output, node, graph.size, line_mask, protected_mask)
        else {
            continue;
        };
        let score = chromatic_identity_score(entry.color, observed);
        let confidence = (node.confidence * entry.confidence).clamp(0.0, 1.0);
        regions.push(RegionConsistency {
            node_id: node.id.clone(),
            semantic: semantic.to_string(),
            expected: entry.color,
            observed,
            score,
            confidence,
            passed: score >= threshold || confidence < 0.45,
        });
    }
    let evaluated_regions = regions.len();
    let mean_score = if regions.is_empty() {
        1.0
    } else {
        let weight = regions.iter().map(|region| region.confidence).sum::<f32>();
        if weight <= f32::EPSILON {
            regions.iter().map(|region| region.score).sum::<f32>() / regions.len() as f32
        } else {
            regions
                .iter()
                .map(|region| region.score * region.confidence)
                .sum::<f32>()
                / weight
        }
    };
    let minimum_score = regions
        .iter()
        .filter(|region| region.confidence >= 0.45)
        .map(|region| region.score)
        .reduce(f32::min)
        .unwrap_or(1.0);
    let passed = regions.iter().all(|region| region.passed);
    ColorBibleConsistencyReport {
        bible_revision: bible.revision,
        evaluated_regions,
        mean_score,
        minimum_score,
        threshold,
        passed,
        regions,
    }
}

#[allow(clippy::too_many_arguments)]
fn add_mask_nodes(
    nodes: &mut Vec<StructureNode>,
    edges: &mut Vec<StructureEdge>,
    panels: &[MangaPanel],
    size: ImageSize,
    mask: &[bool],
    kind: StructureNodeKind,
    source: &str,
    min_pixels: u64,
    max_regions: usize,
    face_boxes: &[[u32; 4]],
) {
    let mut image = GrayImage::from_pixel(size.width, size.height, Luma([0]));
    for (index, active) in mask.iter().copied().enumerate() {
        if active {
            image.put_pixel(
                index as u32 % size.width,
                index as u32 / size.width,
                Luma([255]),
            );
        }
    }
    add_gray_mask_nodes_inner(
        nodes,
        edges,
        panels,
        size,
        &image,
        kind,
        source,
        min_pixels,
        max_regions,
        face_boxes,
    );
}

#[allow(clippy::too_many_arguments)]
fn add_gray_mask_nodes(
    nodes: &mut Vec<StructureNode>,
    edges: &mut Vec<StructureEdge>,
    panels: &[MangaPanel],
    size: ImageSize,
    mask: &GrayImage,
    kind: StructureNodeKind,
    source: &str,
    min_pixels: u64,
    max_regions: usize,
) {
    add_gray_mask_nodes_inner(
        nodes,
        edges,
        panels,
        size,
        mask,
        kind,
        source,
        min_pixels,
        max_regions,
        &[],
    );
}

#[allow(clippy::too_many_arguments)]
fn add_gray_mask_nodes_inner(
    nodes: &mut Vec<StructureNode>,
    edges: &mut Vec<StructureEdge>,
    panels: &[MangaPanel],
    size: ImageSize,
    mask: &GrayImage,
    kind: StructureNodeKind,
    source: &str,
    min_pixels: u64,
    max_regions: usize,
    face_boxes: &[[u32; 4]],
) {
    let mut components = mask_components_with_runs(mask, min_pixels);
    components.sort_by_key(|component| std::cmp::Reverse(component.pixels));
    components.truncate(max_regions);
    let offset = nodes.iter().filter(|node| node.kind == kind).count();
    for (index, component) in components.into_iter().enumerate() {
        let bounds = component.bounds;
        let pixels = component.pixels;
        let id = format!("{:?}_{:04}", kind, offset + index + 1).to_ascii_lowercase();
        let panel_id = containing_panel(panels, bounds);
        let character_id = if matches!(kind, StructureNodeKind::Hair | StructureNodeKind::Clothing)
        {
            nearest_character(face_boxes, bounds, size)
        } else {
            None
        };
        let location_id = (kind == StructureNodeKind::Background).then(|| "scene".to_string());
        let confidence = component_confidence(bounds, pixels, size, kind);
        nodes.push(StructureNode {
            id: id.clone(),
            kind,
            bounds,
            normalized_bounds: normalize_box(bounds, size),
            confidence,
            parent_id: panel_id.clone(),
            panel_id: panel_id.clone(),
            character_id,
            location_id,
            object_id: None,
            protected: kind.protected(),
            source: source.to_string(),
            mask_runs: component.runs,
        });
        if let Some(panel_id) = panel_id {
            edges.push(StructureEdge {
                from: panel_id,
                to: id,
                relation: "contains".to_string(),
                confidence,
            });
        }
    }
}

fn normalize_box(bounds: BoundingBox, size: ImageSize) -> NormalizedRegion {
    let bounds = bounds.clamp_to(size);
    NormalizedRegion {
        x: bounds.x as f32 / size.width.max(1) as f32,
        y: bounds.y as f32 / size.height.max(1) as f32,
        width: bounds.width as f32 / size.width.max(1) as f32,
        height: bounds.height as f32 / size.height.max(1) as f32,
    }
}

fn containing_panel(panels: &[MangaPanel], bounds: BoundingBox) -> Option<String> {
    let center_x = bounds.x.saturating_add(bounds.width / 2);
    let center_y = bounds.y.saturating_add(bounds.height / 2);
    panels
        .iter()
        .filter(|panel| panel.bbox.contains(center_x, center_y))
        .min_by_key(|panel| panel.bbox.area())
        .map(|panel| panel.id.clone())
}

fn face_material_bounds(
    face: BoundingBox,
    size: ImageSize,
) -> Vec<(&'static str, StructureNodeKind, BoundingBox, f32)> {
    let skin = BoundingBox::new(
        face.x.saturating_add(face.width / 10),
        face.y.saturating_add(face.height / 4),
        face.width.saturating_mul(8) / 10,
        face.height.saturating_mul(7) / 10,
    )
    .clamp_to(size);
    let hair = BoundingBox::new(
        face.x.saturating_sub(face.width / 12),
        face.y.saturating_sub(face.height / 5),
        face.width.saturating_mul(7) / 6,
        face.height.saturating_mul(11) / 20,
    )
    .clamp_to(size);
    let eyes = BoundingBox::new(
        face.x.saturating_add(face.width / 5),
        face.y.saturating_add(face.height.saturating_mul(2) / 5),
        face.width.saturating_mul(3) / 5,
        (face.height / 6).max(1),
    )
    .clamp_to(size);
    vec![
        ("skin", StructureNodeKind::Skin, skin, 0.78),
        ("hair", StructureNodeKind::Hair, hair, 0.72),
        ("eyes", StructureNodeKind::Eyes, eyes, 0.68),
    ]
}

fn nearest_character(faces: &[[u32; 4]], bounds: BoundingBox, size: ImageSize) -> Option<String> {
    let center = (
        bounds.x as f32 + bounds.width as f32 * 0.5,
        bounds.y as f32 + bounds.height as f32 * 0.5,
    );
    let diagonal = ((size.width as f32).powi(2) + (size.height as f32).powi(2)).sqrt();
    faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            let face_center = (
                face[0] as f32 + face[2] as f32 * 0.5,
                face[1] as f32 + face[3] as f32 * 0.75,
            );
            let distance =
                ((center.0 - face_center.0).powi(2) + (center.1 - face_center.1).powi(2)).sqrt();
            (index, distance)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .filter(|(_, distance)| *distance <= diagonal * 0.32)
        .map(|(index, _)| format!("character_{:03}", index + 1))
}

fn component_confidence(
    bounds: BoundingBox,
    pixels: u64,
    size: ImageSize,
    kind: StructureNodeKind,
) -> f32 {
    let density = pixels as f32 / bounds.area().max(1) as f32;
    let size_score = (bounds.area() as f32 / size.area().max(1) as f32 * 8.0).min(0.2);
    let base = match kind {
        StructureNodeKind::SpeechBubble
        | StructureNodeKind::Text
        | StructureNodeKind::Protected => 0.78,
        StructureNodeKind::Background => 0.66,
        StructureNodeKind::Hair | StructureNodeKind::Clothing => 0.58,
        _ => 0.55,
    };
    (base + density * 0.18 + size_score).clamp(0.2, 0.97)
}

fn node_order(kind: StructureNodeKind) -> u8 {
    match kind {
        StructureNodeKind::Panel => 0,
        StructureNodeKind::Character => 1,
        StructureNodeKind::Skin
        | StructureNodeKind::Hair
        | StructureNodeKind::Eyes
        | StructureNodeKind::Clothing => 2,
        StructureNodeKind::Object | StructureNodeKind::Effect | StructureNodeKind::Background => 3,
        StructureNodeKind::SpeechBubble
        | StructureNodeKind::Text
        | StructureNodeKind::Protected => 4,
    }
}

fn mean_unprotected_color(
    output: &RgbImage,
    node: &StructureNode,
    graph_size: ImageSize,
    line_mask: &GrayImage,
    protected_mask: &GrayImage,
) -> Option<ColorRgb> {
    if output.width() == 0 || output.height() == 0 {
        return None;
    }
    let region = node.normalized_bounds;
    let min_x = (region.x * output.width() as f32).floor().max(0.0) as u32;
    let min_y = (region.y * output.height() as f32).floor().max(0.0) as u32;
    let max_x = ((region.x + region.width) * output.width() as f32)
        .ceil()
        .min(output.width() as f32) as u32;
    let max_y = ((region.y + region.height) * output.height() as f32)
        .ceil()
        .min(output.height() as f32) as u32;
    let area = max_x.saturating_sub(min_x) as u64 * max_y.saturating_sub(min_y) as u64;
    let stride = ((area as f32 / 4096.0).sqrt().floor() as u32).max(1);
    let (mut r, mut g, mut b, mut weight) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in (min_y..max_y).step_by(stride as usize) {
        for x in (min_x..max_x).step_by(stride as usize) {
            if !node_mask_contains(node, graph_size, output, x, y) {
                continue;
            }
            let mask_x = (x as u64 * line_mask.width().max(1) as u64 / output.width().max(1) as u64)
                .min(line_mask.width().saturating_sub(1) as u64) as u32;
            let mask_y = (y as u64 * line_mask.height().max(1) as u64
                / output.height().max(1) as u64)
                .min(line_mask.height().saturating_sub(1) as u64) as u32;
            if line_mask.get_pixel(mask_x, mask_y)[0] > 150
                || protected_mask.get_pixel(mask_x, mask_y)[0] > 96
            {
                continue;
            }
            let pixel = output.get_pixel(x, y);
            let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
            if !(12..=247).contains(&luma) {
                continue;
            }
            let local_weight = (1.0 - ((luma as f32 - 130.0).abs() / 180.0)).max(0.25) as f64;
            r += pixel[0] as f64 * local_weight;
            g += pixel[1] as f64 * local_weight;
            b += pixel[2] as f64 * local_weight;
            weight += local_weight;
        }
    }
    (weight >= 4.0).then(|| {
        ColorRgb::new(
            (r / weight).round().clamp(0.0, 255.0) as u8,
            (g / weight).round().clamp(0.0, 255.0) as u8,
            (b / weight).round().clamp(0.0, 255.0) as u8,
        )
    })
}

fn node_mask_contains(
    node: &StructureNode,
    graph_size: ImageSize,
    output: &RgbImage,
    x: u32,
    y: u32,
) -> bool {
    if node.mask_runs.is_empty() {
        return true;
    }
    let source_x = (x as u64 * graph_size.width.max(1) as u64 / output.width().max(1) as u64)
        .min(graph_size.width.saturating_sub(1) as u64) as u32;
    let source_y = (y as u64 * graph_size.height.max(1) as u64 / output.height().max(1) as u64)
        .min(graph_size.height.saturating_sub(1) as u64) as u32;
    let start = node.mask_runs.partition_point(|run| run.y < source_y);
    node.mask_runs[start..]
        .iter()
        .take_while(|run| run.y == source_y)
        .any(|run| source_x >= run.x_start && source_x <= run.x_end)
}

fn chromatic_identity_score(expected: ColorRgb, observed: ColorRgb) -> f32 {
    let chromaticity = |color: ColorRgb| {
        let sum = (color.r as f32 + color.g as f32 + color.b as f32).max(1.0);
        [
            color.r as f32 / sum,
            color.g as f32 / sum,
            color.b as f32 / sum,
        ]
    };
    let expected_c = chromaticity(expected);
    let observed_c = chromaticity(observed);
    let distance = expected_c
        .iter()
        .zip(observed_c)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        .sqrt();
    (1.0 - distance / 0.52).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use mangahue_memory::{ColorBibleCorrection, ColorSelector, CorrectionScope};

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

    #[test]
    fn structure_graph_drives_constraints_and_verification() {
        let size = ImageSize::new(100, 100);
        let panel = MangaPanel {
            id: "panel_001".to_string(),
            bbox: BoundingBox::new(0, 0, 100, 100),
            reading_order: 0,
            confidence: 0.95,
        };
        let mut masks = SemanticMasks {
            skin: vec![false; 10_000],
            hair: vec![false; 10_000],
            clothing: vec![false; 10_000],
            background: vec![false; 10_000],
            bubble: vec![false; 10_000],
        };
        for y in 24..45 {
            for x in 28..66 {
                masks.hair[(y * 100 + x) as usize] = true;
            }
        }
        let mut bubble = GrayImage::from_pixel(100, 100, Luma([0]));
        let mut text = bubble.clone();
        let mut protected = bubble.clone();
        for y in 70..90 {
            for x in 60..90 {
                bubble.put_pixel(x, y, Luma([255]));
                protected.put_pixel(x, y, Luma([255]));
            }
        }
        for y in 76..82 {
            for x in 68..82 {
                text.put_pixel(x, y, Luma([255]));
            }
        }
        let graph = build_structure_graph(
            Some("series".to_string()),
            Some("chapter_1".to_string()),
            "page_1",
            PathBuf::from("page.png"),
            size,
            &[panel],
            &[[30, 30, 30, 30]],
            &masks,
            &bubble,
            &text,
            &protected,
            "daylight",
        )
        .unwrap();
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.kind == StructureNodeKind::Hair));
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.kind == StructureNodeKind::SpeechBubble && node.protected));
        assert!(graph.node_at(0.4, 0.35).is_some());

        let canonical = ColorRgb::new(180, 60, 40);
        let mut bible = ColorBible::new("series", "Test Series").unwrap();
        bible
            .apply_correction(ColorBibleCorrection {
                selector: ColorSelector {
                    scope: CorrectionScope::Series,
                    page_id: None,
                    chapter_id: None,
                    character_id: None,
                    location_id: None,
                    object_id: None,
                    region_id: None,
                    semantic: Some("hair".to_string()),
                    region: None,
                },
                color: canonical,
                confidence: 1.0,
                source: "test".to_string(),
                reference_ids: Vec::new(),
                note: None,
            })
            .unwrap();
        let locks = color_bible_region_locks(&graph, &bible);
        assert!(locks
            .locks
            .iter()
            .any(|lock| lock.mode == RegionLockMode::Colorize));
        assert!(locks
            .locks
            .iter()
            .any(|lock| lock.mode == RegionLockMode::Preserve));
        assert!(locks
            .locks
            .iter()
            .any(|lock| lock.mode == RegionLockMode::Exclude));

        let no_mask = GrayImage::from_pixel(100, 100, Luma([0]));
        let matching = RgbImage::from_pixel(100, 100, Rgb([180, 60, 40]));
        assert!(
            verify_color_bible_consistency(&matching, &graph, &bible, &no_mask, &no_mask, 0.7)
                .passed
        );
        let wrong = RgbImage::from_pixel(100, 100, Rgb([30, 80, 210]));
        assert!(
            !verify_color_bible_consistency(&wrong, &graph, &bible, &no_mask, &no_mask, 0.7).passed
        );

        let graph_dir = tempfile::tempdir().unwrap();
        graph.save(graph_dir.path().join("page.json")).unwrap();
        let impact = plan_correction_impact(
            graph_dir.path(),
            &ColorSelector {
                scope: CorrectionScope::Series,
                page_id: None,
                chapter_id: None,
                character_id: None,
                location_id: None,
                object_id: None,
                region_id: None,
                semantic: Some("hair".to_string()),
                region: None,
            },
        )
        .unwrap();
        assert_eq!(impact.affected_pages, 1);
        assert!(impact.affected_regions >= 1);
    }
}
