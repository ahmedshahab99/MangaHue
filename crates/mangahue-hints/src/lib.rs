use anyhow::{Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_config::HintsSection;
use mangahue_core::{
    CharacterProfile, ColorHint, ColorRgb, HintSource, ImageSize, StyleProfile, UserHintFile,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

const MAX_REGION_LOCKS: usize = 4096;
const MAX_POLYGON_POINTS: usize = 16_384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintResult {
    #[serde(skip)]
    pub hint_map: RgbImage,
    #[serde(skip)]
    pub hint_mask: GrayImage,
    pub applied_hints: usize,
    pub style_hints_applied: usize,
    pub memory_hints_applied: usize,
    pub user_hints_applied: usize,
    #[serde(default)]
    pub region_style_hints_applied: usize,
    pub notes: Vec<String>,
}

pub fn load_user_hints(path: impl AsRef<Path>) -> Result<UserHintFile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read hints {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse hints {}", path.display()))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionCoordinateSpace {
    #[default]
    Pixels,
    Normalized,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionSemantic {
    Hair,
    Eyes,
    Skin,
    Clothing,
    Background,
    Speech,
    Prop,
    Effect,
    #[default]
    Custom,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionLockMode {
    #[default]
    Colorize,
    Preserve,
    Exclude,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegionPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum RegionShape {
    Rectangle {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    Ellipse {
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: f32,
    },
    Polygon {
        points: Vec<RegionPoint>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionLock {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub semantic: RegionSemantic,
    #[serde(default)]
    pub mode: RegionLockMode,
    pub region: RegionShape,
    #[serde(default)]
    pub color: Option<ColorRgb>,
    #[serde(default = "default_region_strength")]
    pub strength: f32,
    #[serde(default = "default_feather_radius")]
    pub feather_radius: f32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionLockFile {
    #[serde(default = "default_region_lock_version")]
    pub version: u32,
    #[serde(default)]
    pub page_id: Option<String>,
    #[serde(default)]
    pub coordinate_space: RegionCoordinateSpace,
    #[serde(default)]
    pub source_width: Option<u32>,
    #[serde(default)]
    pub source_height: Option<u32>,
    #[serde(default)]
    pub locks: Vec<RegionLock>,
}

#[derive(Debug, Clone)]
pub struct RegionLockRaster {
    pub color_map: RgbImage,
    pub color_mask: GrayImage,
    pub preserve_mask: GrayImage,
    pub exclude_mask: GrayImage,
    pub applied_locks: usize,
    pub active_pixels: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionLockReport {
    pub applied_locks: usize,
    pub colorized_pixels: u64,
    pub preserved_pixels: u64,
    pub excluded_hint_pixels: u64,
}

fn default_region_lock_version() -> u32 {
    1
}

fn default_region_strength() -> f32 {
    1.0
}

fn default_feather_radius() -> f32 {
    2.0
}

fn default_true() -> bool {
    true
}

pub fn load_region_locks(path: impl AsRef<Path>) -> Result<RegionLockFile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read region locks {}", path.display()))?;
    let locks: RegionLockFile = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse region locks {}", path.display()))?;
    validate_region_locks(&locks)?;
    Ok(locks)
}

pub fn validate_region_locks(file: &RegionLockFile) -> Result<()> {
    anyhow::ensure!(
        file.version == 1,
        "unsupported region lock version {}",
        file.version
    );
    anyhow::ensure!(
        file.locks.len() <= MAX_REGION_LOCKS,
        "region lock count {} exceeds limit {}",
        file.locks.len(),
        MAX_REGION_LOCKS
    );
    for lock in &file.locks {
        anyhow::ensure!(!lock.id.trim().is_empty(), "region lock id cannot be empty");
        anyhow::ensure!(
            lock.strength.is_finite(),
            "region lock {} has invalid strength",
            lock.id
        );
        anyhow::ensure!(
            lock.feather_radius.is_finite() && lock.feather_radius >= 0.0,
            "region lock {} has invalid feather radius",
            lock.id
        );
        if lock.mode == RegionLockMode::Colorize {
            anyhow::ensure!(
                lock.color.is_some(),
                "colorize region lock {} needs a color",
                lock.id
            );
        }
        match &lock.region {
            RegionShape::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                anyhow::ensure!(
                    [x, y, width, height].iter().all(|value| value.is_finite()),
                    "region lock {} rectangle contains invalid coordinates",
                    lock.id
                );
                anyhow::ensure!(
                    *width != 0.0 && *height != 0.0,
                    "region lock {} rectangle is empty",
                    lock.id
                );
            }
            RegionShape::Ellipse {
                center_x,
                center_y,
                radius_x,
                radius_y,
            } => {
                anyhow::ensure!(
                    [center_x, center_y, radius_x, radius_y]
                        .iter()
                        .all(|value| value.is_finite()),
                    "region lock {} ellipse contains invalid coordinates",
                    lock.id
                );
                anyhow::ensure!(
                    *radius_x > 0.0 && *radius_y > 0.0,
                    "region lock {} ellipse is empty",
                    lock.id
                );
            }
            RegionShape::Polygon { points } => {
                anyhow::ensure!(
                    points.len() >= 3,
                    "region lock {} polygon needs at least 3 points",
                    lock.id
                );
                anyhow::ensure!(
                    points.len() <= MAX_POLYGON_POINTS,
                    "region lock {} polygon exceeds point limit {}",
                    lock.id,
                    MAX_POLYGON_POINTS
                );
                anyhow::ensure!(
                    points
                        .iter()
                        .all(|point| point.x.is_finite() && point.y.is_finite()),
                    "region lock {} polygon contains invalid coordinates",
                    lock.id
                );
            }
        }
    }
    Ok(())
}

pub fn generate_hints(
    size: ImageSize,
    user_hints: Option<&UserHintFile>,
    style: &StyleProfile,
    characters: &[CharacterProfile],
    config: &HintsSection,
) -> HintResult {
    let mut hint_map = RgbImage::from_pixel(size.width, size.height, Rgb([0, 0, 0]));
    let mut hint_mask = GrayImage::from_pixel(size.width, size.height, Luma([0]));
    let mut applied = 0usize;
    let mut style_hints_applied = 0usize;
    let mut memory_hints_applied = 0usize;
    let mut user_hints_applied = 0usize;
    let mut notes = Vec::new();

    if config.style_hints_enabled {
        style_hints_applied = if config.adaptive_placement_enabled {
            apply_adaptive_style_hints(&mut hint_map, &mut hint_mask, style, config.hint_strength)
        } else {
            apply_style_atmosphere(&mut hint_map, &mut hint_mask, style, config.hint_strength)
        };
        applied += style_hints_applied;
    }

    if config.character_memory_enabled && !characters.is_empty() {
        notes.push(format!("loaded {} character profile(s)", characters.len()));
        if let Some(file) = user_hints {
            for hint in &file.hints {
                if let Some(memory_hint) = memory_hint_from_user_hint(hint, characters) {
                    rasterize_hint(
                        &mut hint_map,
                        &mut hint_mask,
                        &memory_hint,
                        config.hint_strength,
                    );
                    memory_hints_applied += 1;
                    applied += 1;
                }
            }
        }
    }

    if config.user_hints_enabled {
        if let Some(file) = user_hints {
            for hint in &file.hints {
                rasterize_hint(&mut hint_map, &mut hint_mask, hint, config.hint_strength);
                user_hints_applied += 1;
                applied += 1;
            }
        }
    }

    if config.hint_blur_radius > 0.0 {
        soften_mask_edges(&mut hint_mask, config.hint_blur_radius.round() as u32);
    }

    HintResult {
        hint_map,
        hint_mask,
        applied_hints: applied,
        style_hints_applied,
        memory_hints_applied,
        user_hints_applied,
        region_style_hints_applied: 0,
        notes,
    }
}

pub fn apply_region_style_hints(
    result: &mut HintResult,
    lineart: &GrayImage,
    bubble_mask: &GrayImage,
    style: &StyleProfile,
    config: &HintsSection,
) -> usize {
    if !config.region_style_hints_enabled
        || config.region_style_hint_budget == 0
        || style.palette.is_empty()
        || lineart.dimensions() != result.hint_mask.dimensions()
        || bubble_mask.dimensions() != result.hint_mask.dimensions()
    {
        return 0;
    }

    let (width, height) = lineart.dimensions();
    let len = width as usize * height as usize;
    let mut visited = vec![false; len];
    let mut components = Vec::<StyleRegion>::new();
    let page_area = len.max(1) as u64;
    let minimum_area = (page_area / 5000).max(48);
    let maximum_area = page_area * 28 / 100;

    for start_y in 0..height {
        for start_x in 0..width {
            let start = (start_y * width + start_x) as usize;
            if visited[start]
                || lineart.get_pixel(start_x, start_y)[0] < 168
                || bubble_mask.get_pixel(start_x, start_y)[0] > 72
            {
                continue;
            }

            let mut queue = VecDeque::from([(start_x, start_y)]);
            visited[start] = true;
            let mut region = StyleRegion::new(start_x, start_y);
            while let Some((x, y)) = queue.pop_front() {
                let gray = lineart.get_pixel(x, y)[0];
                let bubble = bubble_mask.get_pixel(x, y)[0];
                if gray < 168 || bubble > 72 {
                    continue;
                }
                region.accumulate(x, y, gray, width, height);
                for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let nx = nx as u32;
                    let ny = ny as u32;
                    let index = (ny * width + nx) as usize;
                    if !visited[index]
                        && lineart.get_pixel(nx, ny)[0] >= 168
                        && bubble_mask.get_pixel(nx, ny)[0] <= 72
                    {
                        visited[index] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            if region.area >= minimum_area
                && region.area <= maximum_area
                && !region.touches_page_border
            {
                components.push(region);
            }
        }
    }

    components.sort_by(|left, right| {
        region_priority(right, page_area).total_cmp(&region_priority(left, page_area))
    });
    let palette = style
        .palette
        .iter()
        .copied()
        .filter(|color| palette_chroma(*color) >= 14)
        .collect::<Vec<_>>();
    if palette.is_empty() {
        return 0;
    }

    let mut applied = 0usize;
    let mut previous_color: Option<ColorRgb> = None;
    for (rank, region) in components
        .into_iter()
        .take(config.region_style_hint_budget)
        .enumerate()
    {
        let luma = region.luma_sum as f32 / region.area.max(1) as f32;
        let color = select_region_palette_color(&palette, luma, rank, previous_color);
        let radius = ((region.area as f32).sqrt() * 0.075)
            .round()
            .clamp(3.0, 14.0) as u32;
        let (x, y) = find_region_seed(
            lineart,
            bubble_mask,
            region.center_x(),
            region.center_y(),
            &region,
        );
        let hint = ColorHint {
            x,
            y,
            radius,
            color,
            label: Some("region_style".to_string()),
            strength: (0.16 * config.hint_strength).clamp(0.0, 0.28),
            source: HintSource::PaletteRule,
        };
        rasterize_hint(&mut result.hint_map, &mut result.hint_mask, &hint, 1.0);
        previous_color = Some(color);
        applied += 1;
    }

    result.applied_hints += applied;
    result.region_style_hints_applied += applied;
    if applied > 0 {
        result.notes.push(format!(
            "region palette director applied {applied} bounded material hint(s)"
        ));
    }
    applied
}

#[derive(Debug, Clone, Copy)]
struct StyleRegion {
    area: u64,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    x_sum: u64,
    y_sum: u64,
    luma_sum: u64,
    touches_page_border: bool,
}

impl StyleRegion {
    fn new(x: u32, y: u32) -> Self {
        Self {
            area: 0,
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            x_sum: 0,
            y_sum: 0,
            luma_sum: 0,
            touches_page_border: false,
        }
    }

    fn accumulate(&mut self, x: u32, y: u32, gray: u8, width: u32, height: u32) {
        self.area += 1;
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.x_sum += x as u64;
        self.y_sum += y as u64;
        self.luma_sum += gray as u64;
        self.touches_page_border |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;
    }

    fn center_x(self) -> u32 {
        (self.x_sum / self.area.max(1)) as u32
    }

    fn center_y(self) -> u32 {
        (self.y_sum / self.area.max(1)) as u32
    }
}

fn region_priority(region: &StyleRegion, page_area: u64) -> f32 {
    let area = region.area as f32 / page_area.max(1) as f32;
    let width = (region.max_x - region.min_x + 1) as f32;
    let height = (region.max_y - region.min_y + 1) as f32;
    let fill = region.area as f32 / (width * height).max(1.0);
    area.sqrt() * 0.72 + fill.clamp(0.0, 1.0) * 0.28
}

fn palette_chroma(color: ColorRgb) -> u8 {
    color
        .r
        .max(color.g)
        .max(color.b)
        .saturating_sub(color.r.min(color.g).min(color.b))
}

fn color_luma(color: ColorRgb) -> f32 {
    0.2126 * color.r as f32 + 0.7152 * color.g as f32 + 0.0722 * color.b as f32
}

fn select_region_palette_color(
    palette: &[ColorRgb],
    region_luma: f32,
    rank: usize,
    previous: Option<ColorRgb>,
) -> ColorRgb {
    palette
        .iter()
        .copied()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            let score = |index: usize, color: ColorRgb| {
                let luma_cost = (color_luma(color) - region_luma).abs();
                let repeat_cost = previous
                    .map(|last| color_distance(last, color) < 34.0)
                    .unwrap_or(false) as u8 as f32
                    * 22.0;
                let diversity = ((index + rank * 3) % palette.len()) as f32 * 0.35;
                luma_cost + repeat_cost + diversity
            };
            score(*left_index, *left).total_cmp(&score(*right_index, *right))
        })
        .map(|(_, color)| color)
        .unwrap_or(palette[0])
}

fn color_distance(left: ColorRgb, right: ColorRgb) -> f32 {
    let dr = left.r as f32 - right.r as f32;
    let dg = left.g as f32 - right.g as f32;
    let db = left.b as f32 - right.b as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn find_region_seed(
    lineart: &GrayImage,
    bubble_mask: &GrayImage,
    center_x: u32,
    center_y: u32,
    region: &StyleRegion,
) -> (u32, u32) {
    let mut best = (center_x, center_y);
    let mut best_score = 0_u8;
    let step_x = ((region.max_x - region.min_x + 1) / 16).max(1);
    let step_y = ((region.max_y - region.min_y + 1) / 16).max(1);
    let mut y = region.min_y;
    while y <= region.max_y {
        let mut x = region.min_x;
        while x <= region.max_x {
            if bubble_mask.get_pixel(x, y)[0] <= 72 {
                let score = lineart.get_pixel(x, y)[0];
                let distance_bias =
                    (x.abs_diff(center_x) + y.abs_diff(center_y)).min(255) as u8 / 8;
                let score = score.saturating_sub(distance_bias);
                if score > best_score {
                    best = (x, y);
                    best_score = score;
                }
            }
            x = x.saturating_add(step_x);
            if x == 0 {
                break;
            }
        }
        y = y.saturating_add(step_y);
        if y == 0 {
            break;
        }
    }
    best
}

pub fn rasterize_hint(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    hint: &ColorHint,
    global_strength: f32,
) {
    let radius = hint.radius.max(1);
    let radius_sq = (radius as i64) * (radius as i64);
    let min_x = hint.x.saturating_sub(radius);
    let min_y = hint.y.saturating_sub(radius);
    let max_x = (hint.x + radius + 1).min(hint_map.width());
    let max_y = (hint.y + radius + 1).min(hint_map.height());
    let strength = (hint.strength * global_strength).clamp(0.0, 1.0);
    let color = Rgb([hint.color.r, hint.color.g, hint.color.b]);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = x as i64 - hint.x as i64;
            let dy = y as i64 - hint.y as i64;
            let distance_sq = dx * dx + dy * dy;
            if distance_sq > radius_sq {
                continue;
            }
            let falloff = 1.0 - (distance_sq as f32 / radius_sq as f32).sqrt();
            let alpha = (falloff * strength).clamp(0.0, 1.0);
            let old = hint_map.get_pixel(x, y);
            let blended = [
                blend(old[0], color[0], alpha),
                blend(old[1], color[1], alpha),
                blend(old[2], color[2], alpha),
            ];
            hint_map.put_pixel(x, y, Rgb(blended));
            let mask_value = ((alpha * 255.0).round() as u8).max(hint_mask.get_pixel(x, y)[0]);
            hint_mask.put_pixel(x, y, Luma([mask_value]));
        }
    }
}

pub fn hint_from_character_label(
    x: u32,
    y: u32,
    radius: u32,
    label: &str,
    character: &CharacterProfile,
) -> Option<ColorHint> {
    let color = match label {
        "hair" => character.hair_color,
        "skin" => character.skin_color,
        "eyes" | "eye" => character.eye_color,
        "outfit" => character.outfit_colors.first().copied(),
        "accent" => character.accent_colors.first().copied(),
        _ => None,
    }?;
    Some(ColorHint {
        x,
        y,
        radius,
        color,
        label: Some(label.to_string()),
        strength: character.confidence.clamp(0.0, 1.0),
        source: HintSource::CharacterMemory,
    })
}

pub fn memory_hint_from_user_hint(
    user_hint: &ColorHint,
    characters: &[CharacterProfile],
) -> Option<ColorHint> {
    let label = user_hint.label.as_deref()?;
    let (character_selector, channel) = parse_character_label(label);
    let character = match character_selector {
        Some(selector) => characters
            .iter()
            .find(|character| character_matches(character, selector))?,
        None if characters.len() == 1 => &characters[0],
        None => return None,
    };
    let mut hint = hint_from_character_label(
        user_hint.x,
        user_hint.y,
        (user_hint.radius as f32 * 1.45).round().max(1.0) as u32,
        channel,
        character,
    )?;
    hint.strength = (hint.strength * 0.62).clamp(0.0, 0.82);
    hint.label = Some(format!("{}.{}", character.id, channel));
    Some(hint)
}

fn parse_character_label(label: &str) -> (Option<&str>, &str) {
    let label = label.trim();
    if let Some((character, channel)) = label.split_once(':') {
        return (Some(character.trim()), channel.trim());
    }
    if let Some((character, channel)) = label.split_once('.') {
        return (Some(character.trim()), channel.trim());
    }
    (None, label)
}

fn character_matches(character: &CharacterProfile, selector: &str) -> bool {
    character.id.eq_ignore_ascii_case(selector)
        || character.name.eq_ignore_ascii_case(selector)
        || character
            .aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(selector))
}

fn apply_style_atmosphere(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    style: &StyleProfile,
    global_strength: f32,
) -> usize {
    let Some(primary) = style.palette.first().copied() else {
        return 0;
    };
    let secondary = style
        .palette
        .get(1)
        .copied()
        .unwrap_or(ColorRgb::new(80, 100, 140));
    let width = hint_map.width();
    let height = hint_map.height();
    let radius = (width.min(height) / 12).max(12);
    let hints = [
        (width / 5, height / 5, primary),
        (width.saturating_mul(4) / 5, height / 5, secondary),
        (width / 5, height.saturating_mul(4) / 5, secondary),
        (
            width.saturating_mul(4) / 5,
            height.saturating_mul(4) / 5,
            primary,
        ),
    ];
    for (x, y, color) in hints {
        let hint = ColorHint {
            x,
            y,
            radius,
            color,
            label: Some("style_atmosphere".to_string()),
            strength: 0.18,
            source: HintSource::StyleProfile,
        };
        rasterize_hint(hint_map, hint_mask, &hint, global_strength);
    }
    4
}

fn apply_adaptive_style_hints(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    style: &StyleProfile,
    global_strength: f32,
) -> usize {
    let width = hint_map.width();
    let height = hint_map.height();
    let palette = &style.palette;
    if palette.is_empty() {
        return 0;
    }

    let grid_cols = 3;
    let grid_rows = 3;
    let cell_w = width / grid_cols;
    let cell_h = height / grid_rows;
    let radius = (cell_w.min(cell_h) / 3).max(8);
    let mut applied = 0;

    for row in 0..grid_rows {
        for col in 0..grid_cols {
            let cx = col * cell_w + cell_w / 2;
            let cy = row * cell_h + cell_h / 2;
            let palette_idx = (row * grid_cols + col) % palette.len();
            let color = palette[palette_idx];

            let strength = if (row + col) % 2 == 0 {
                0.22
            } else {
                0.14
            };

            let hint = ColorHint {
                x: cx,
                y: cy,
                radius,
                color,
                label: Some("adaptive_grid".to_string()),
                strength,
                source: HintSource::StyleProfile,
            };
            rasterize_hint(hint_map, hint_mask, &hint, global_strength);
            applied += 1;
        }
    }

    let accent = palette.get(2).copied().unwrap_or(palette[0]);
    let accent_hints = [
        (width / 2, height / 4, radius * 2, 0.12),
        (width / 4, height / 2, radius, 0.10),
        (width * 3 / 4, height * 3 / 4, radius, 0.10),
    ];
    for (x, y, r, str) in accent_hints {
        let hint = ColorHint {
            x,
            y,
            radius: r,
            color: accent,
            label: Some("adaptive_accent".to_string()),
            strength: str,
            source: HintSource::StyleProfile,
        };
        rasterize_hint(hint_map, hint_mask, &hint, global_strength);
        applied += 1;
    }

    applied
}

fn soften_mask_edges(mask: &mut GrayImage, radius: u32) {
    if radius == 0 {
        return;
    }
    let source = mask.clone();
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let current = source.get_pixel(x, y)[0];
            if current == 255 {
                continue;
            }
            let min_x = x.saturating_sub(radius);
            let min_y = y.saturating_sub(radius);
            let max_x = (x + radius + 1).min(mask.width());
            let max_y = (y + radius + 1).min(mask.height());
            let mut best = current;
            for py in min_y..max_y {
                for px in min_x..max_x {
                    let value = source.get_pixel(px, py)[0];
                    if value > best {
                        let dx = px as f32 - x as f32;
                        let dy = py as f32 - y as f32;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let falloff = (1.0 - distance / radius as f32).clamp(0.0, 1.0);
                        best = best.max((value as f32 * falloff).round() as u8);
                    }
                }
            }
            mask.put_pixel(x, y, Luma([best]));
        }
    }
}

fn blend(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

pub fn propagate_hints(
    hint_map: &RgbImage,
    hint_mask: &GrayImage,
    lineart: &GrayImage,
    propagation_radius: u32,
    similarity_threshold: f32,
) -> (RgbImage, GrayImage) {
    let (w, h) = (hint_map.width(), hint_map.height());
    let mut propagated_map = hint_map.clone();
    let mut propagated_mask = hint_mask.clone();
    if propagation_radius == 0 || w == 0 || h == 0 {
        return (propagated_map, propagated_mask);
    }
    let pixel_count = (w as usize).saturating_mul(h as usize);
    let mut distance = vec![u32::MAX; pixel_count];
    let mut queue = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            if hint_mask.get_pixel(x, y)[0] > 0 {
                distance[(y * w + x) as usize] = 0;
                queue.push_back((x, y));
            }
        }
    }

    const NEIGHBORS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    while let Some((x, y)) = queue.pop_front() {
        let current_index = (y * w + x) as usize;
        let current_distance = distance[current_index];
        if current_distance >= propagation_radius {
            continue;
        }
        let current_luma = lineart.get_pixel(x, y)[0] as f32 / 255.0;
        let current_strength = propagated_mask.get_pixel(x, y)[0] as f32 / 255.0;
        let current_color = *propagated_map.get_pixel(x, y);

        for (dx, dy) in NEIGHBORS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let nx = nx as u32;
            let ny = ny as u32;
            let neighbor_index = (ny * w + nx) as usize;
            let next_distance = current_distance + 1;
            if distance[neighbor_index] < next_distance {
                continue;
            }
            let neighbor_luma = lineart.get_pixel(nx, ny)[0] as f32 / 255.0;
            let similarity = 1.0 - (current_luma - neighbor_luma).abs();
            let line_barrier = if neighbor_luma < 0.18 { 0.12 } else { 1.0 };
            let distance_decay = 1.0 - next_distance as f32 / (propagation_radius + 1) as f32;
            let weight = current_strength * similarity * line_barrier * distance_decay.max(0.0);
            if weight <= similarity_threshold {
                continue;
            }
            let candidate = (weight * 255.0).round().clamp(0.0, 255.0) as u8;
            if distance[neighbor_index] == next_distance
                && propagated_mask.get_pixel(nx, ny)[0] >= candidate
            {
                continue;
            }
            distance[neighbor_index] = next_distance;
            propagated_map.put_pixel(nx, ny, current_color);
            propagated_mask.put_pixel(nx, ny, Luma([candidate]));
            queue.push_back((nx, ny));
        }
    }

    (propagated_map, propagated_mask)
}

pub fn rasterize_region_locks(file: &RegionLockFile, size: ImageSize) -> Result<RegionLockRaster> {
    validate_region_locks(file)?;
    anyhow::ensure!(
        size.width > 0 && size.height > 0,
        "region lock target cannot be empty"
    );

    let mut raster = RegionLockRaster {
        color_map: RgbImage::from_pixel(size.width, size.height, Rgb([0, 0, 0])),
        color_mask: GrayImage::from_pixel(size.width, size.height, Luma([0])),
        preserve_mask: GrayImage::from_pixel(size.width, size.height, Luma([0])),
        exclude_mask: GrayImage::from_pixel(size.width, size.height, Luma([0])),
        applied_locks: 0,
        active_pixels: 0,
    };

    for lock in file.locks.iter().filter(|lock| lock.enabled) {
        let mut mask = rasterize_region_shape(file, lock, size);
        if lock.feather_radius > 0.0 {
            mask = image::imageops::blur(&mask, lock.feather_radius.min(128.0));
        }
        let strength = lock.strength.clamp(0.0, 1.0);
        match lock.mode {
            RegionLockMode::Colorize => {
                let color = lock.color.expect("validated color lock");
                composite_color_lock(
                    &mut raster.color_map,
                    &mut raster.color_mask,
                    &mask,
                    color,
                    strength,
                );
            }
            RegionLockMode::Preserve => {
                composite_mask(&mut raster.preserve_mask, &mask, strength);
            }
            RegionLockMode::Exclude => {
                composite_mask(&mut raster.exclude_mask, &mask, strength);
            }
        }
        raster.applied_locks += 1;
    }
    raster.active_pixels = (0..size.height)
        .flat_map(|y| (0..size.width).map(move |x| (x, y)))
        .filter(|(x, y)| {
            raster.color_mask.get_pixel(*x, *y)[0] > 0
                || raster.preserve_mask.get_pixel(*x, *y)[0] > 0
                || raster.exclude_mask.get_pixel(*x, *y)[0] > 0
        })
        .count() as u64;
    Ok(raster)
}

pub fn apply_region_locks_to_hints(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    raster: &RegionLockRaster,
) -> RegionLockReport {
    let width = hint_map.width().min(raster.color_map.width());
    let height = hint_map.height().min(raster.color_map.height());
    let mut report = RegionLockReport {
        applied_locks: raster.applied_locks,
        colorized_pixels: 0,
        preserved_pixels: 0,
        excluded_hint_pixels: 0,
    };

    for y in 0..height {
        for x in 0..width {
            let color_alpha = raster.color_mask.get_pixel(x, y)[0];
            if color_alpha > 0 {
                let alpha = color_alpha as f32 / 255.0;
                let current = *hint_map.get_pixel(x, y);
                let locked = *raster.color_map.get_pixel(x, y);
                hint_map.put_pixel(x, y, blend_rgb(current, locked, alpha));
                hint_mask.put_pixel(x, y, Luma([color_alpha.max(hint_mask.get_pixel(x, y)[0])]));
                report.colorized_pixels += 1;
            }
            if raster.preserve_mask.get_pixel(x, y)[0] > 0 {
                report.preserved_pixels += 1;
            }
            let exclude_alpha = raster.exclude_mask.get_pixel(x, y)[0];
            if exclude_alpha > 0 {
                let retained = ((hint_mask.get_pixel(x, y)[0] as f32)
                    * (1.0 - exclude_alpha as f32 / 255.0))
                    .round() as u8;
                hint_mask.put_pixel(x, y, Luma([retained]));
                if retained == 0 {
                    hint_map.put_pixel(x, y, Rgb([0, 0, 0]));
                }
                report.excluded_hint_pixels += 1;
            }
        }
    }
    report
}

pub fn enforce_region_locks(
    output: &mut RgbImage,
    original_grayscale: &GrayImage,
    line_mask: &GrayImage,
    raster: &RegionLockRaster,
) -> RegionLockReport {
    let width = output
        .width()
        .min(original_grayscale.width())
        .min(raster.color_map.width());
    let height = output
        .height()
        .min(original_grayscale.height())
        .min(raster.color_map.height());
    let mut report = RegionLockReport {
        applied_locks: raster.applied_locks,
        colorized_pixels: 0,
        preserved_pixels: 0,
        excluded_hint_pixels: 0,
    };

    for y in 0..height {
        for x in 0..width {
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let color_alpha = raster.color_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if color_alpha > 0.0 && ink < 0.82 {
                let target = *raster.color_map.get_pixel(x, y);
                let gray = original_grayscale.get_pixel(x, y)[0];
                let locked = color_with_luminance(target, gray);
                let alpha = color_alpha * (1.0 - ink).powf(1.4);
                let current = *output.get_pixel(x, y);
                output.put_pixel(x, y, blend_rgb(current, locked, alpha));
                report.colorized_pixels += 1;
            }

            let preserve_alpha = raster.preserve_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if preserve_alpha > 0.0 {
                let gray = original_grayscale.get_pixel(x, y)[0];
                let current = *output.get_pixel(x, y);
                output.put_pixel(
                    x,
                    y,
                    blend_rgb(current, Rgb([gray, gray, gray]), preserve_alpha),
                );
                report.preserved_pixels += 1;
            }
        }
    }
    report
}

fn rasterize_region_shape(file: &RegionLockFile, lock: &RegionLock, size: ImageSize) -> GrayImage {
    let mut mask = GrayImage::from_pixel(size.width, size.height, Luma([0]));
    let shape = ScaledRegionShape::new(file, &lock.region, size);
    let (min_x, min_y, max_x, max_y) = shape.bounds(size);
    for y in min_y..max_y {
        for x in min_x..max_x {
            let point = RegionPoint {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
            };
            if shape.contains(point) {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
    mask
}

fn scale_region_point(file: &RegionLockFile, point: RegionPoint, size: ImageSize) -> RegionPoint {
    match file.coordinate_space {
        RegionCoordinateSpace::Normalized => RegionPoint {
            x: point.x.clamp(0.0, 1.0) * size.width as f32,
            y: point.y.clamp(0.0, 1.0) * size.height as f32,
        },
        RegionCoordinateSpace::Pixels => RegionPoint {
            x: point.x * size.width as f32 / file.source_width.unwrap_or(size.width).max(1) as f32,
            y: point.y * size.height as f32
                / file.source_height.unwrap_or(size.height).max(1) as f32,
        },
    }
}

enum ScaledRegionShape {
    Rectangle {
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
    },
    Ellipse {
        center: RegionPoint,
        radius_x: f32,
        radius_y: f32,
    },
    Polygon(Vec<RegionPoint>),
}

impl ScaledRegionShape {
    fn new(file: &RegionLockFile, shape: &RegionShape, size: ImageSize) -> Self {
        match shape {
            RegionShape::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                let a = scale_region_point(file, RegionPoint { x: *x, y: *y }, size);
                let b = scale_region_point(
                    file,
                    RegionPoint {
                        x: *x + *width,
                        y: *y + *height,
                    },
                    size,
                );
                Self::Rectangle {
                    min_x: a.x.min(b.x),
                    min_y: a.y.min(b.y),
                    max_x: a.x.max(b.x),
                    max_y: a.y.max(b.y),
                }
            }
            RegionShape::Ellipse {
                center_x,
                center_y,
                radius_x,
                radius_y,
            } => {
                let center = scale_region_point(
                    file,
                    RegionPoint {
                        x: *center_x,
                        y: *center_y,
                    },
                    size,
                );
                let edge = scale_region_point(
                    file,
                    RegionPoint {
                        x: *center_x + *radius_x,
                        y: *center_y + *radius_y,
                    },
                    size,
                );
                Self::Ellipse {
                    center,
                    radius_x: (edge.x - center.x).abs().max(0.5),
                    radius_y: (edge.y - center.y).abs().max(0.5),
                }
            }
            RegionShape::Polygon { points } => Self::Polygon(
                points
                    .iter()
                    .copied()
                    .map(|point| scale_region_point(file, point, size))
                    .collect(),
            ),
        }
    }

    fn contains(&self, point: RegionPoint) -> bool {
        match self {
            Self::Rectangle {
                min_x,
                min_y,
                max_x,
                max_y,
            } => point.x >= *min_x && point.y >= *min_y && point.x <= *max_x && point.y <= *max_y,
            Self::Ellipse {
                center,
                radius_x,
                radius_y,
            } => {
                let dx = (point.x - center.x) / radius_x;
                let dy = (point.y - center.y) / radius_y;
                dx * dx + dy * dy <= 1.0
            }
            Self::Polygon(points) => point_in_polygon(point, points),
        }
    }

    fn bounds(&self, size: ImageSize) -> (u32, u32, u32, u32) {
        let (min_x, min_y, max_x, max_y) = match self {
            Self::Rectangle {
                min_x,
                min_y,
                max_x,
                max_y,
            } => (*min_x, *min_y, *max_x, *max_y),
            Self::Ellipse {
                center,
                radius_x,
                radius_y,
            } => (
                center.x - radius_x,
                center.y - radius_y,
                center.x + radius_x,
                center.y + radius_y,
            ),
            Self::Polygon(points) => {
                points
                    .iter()
                    .fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |bounds, point| {
                        (
                            bounds.0.min(point.x),
                            bounds.1.min(point.y),
                            bounds.2.max(point.x),
                            bounds.3.max(point.y),
                        )
                    })
            }
        };
        (
            min_x.floor().max(0.0) as u32,
            min_y.floor().max(0.0) as u32,
            max_x.ceil().clamp(0.0, size.width as f32) as u32,
            max_y.ceil().clamp(0.0, size.height as f32) as u32,
        )
    }
}

fn point_in_polygon(point: RegionPoint, polygon: &[RegionPoint]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let a = polygon[current];
        let b = polygon[previous];
        let crosses = (a.y > point.y) != (b.y > point.y)
            && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
        if crosses {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn composite_color_lock(
    map: &mut RgbImage,
    target_mask: &mut GrayImage,
    source_mask: &GrayImage,
    color: ColorRgb,
    strength: f32,
) {
    let color = Rgb([color.r, color.g, color.b]);
    for (x, y, source) in source_mask.enumerate_pixels() {
        let alpha = source[0] as f32 / 255.0 * strength;
        if alpha <= 0.0 {
            continue;
        }
        let current_alpha = target_mask.get_pixel(x, y)[0] as f32 / 255.0;
        let combined = 1.0 - (1.0 - current_alpha) * (1.0 - alpha);
        let mix = if combined > 0.0 {
            alpha / combined
        } else {
            0.0
        };
        map.put_pixel(x, y, blend_rgb(*map.get_pixel(x, y), color, mix));
        target_mask.put_pixel(x, y, Luma([(combined * 255.0).round() as u8]));
    }
}

fn composite_mask(target: &mut GrayImage, source: &GrayImage, strength: f32) {
    for (x, y, pixel) in source.enumerate_pixels() {
        let value = (pixel[0] as f32 * strength).round() as u8;
        if value > target.get_pixel(x, y)[0] {
            target.put_pixel(x, y, Luma([value]));
        }
    }
}

fn blend_rgb(a: Rgb<u8>, b: Rgb<u8>, alpha: f32) -> Rgb<u8> {
    Rgb([
        blend(a[0], b[0], alpha),
        blend(a[1], b[1], alpha),
        blend(a[2], b[2], alpha),
    ])
}

fn color_with_luminance(color: Rgb<u8>, luminance: u8) -> Rgb<u8> {
    let target_luma =
        0.2126 * color[0] as f32 + 0.7152 * color[1] as f32 + 0.0722 * color[2] as f32;
    let desired = luminance as f32;
    let chroma_scale = 0.72;
    Rgb([
        (desired + (color[0] as f32 - target_luma) * chroma_scale).clamp(0.0, 255.0) as u8,
        (desired + (color[1] as f32 - target_luma) * chroma_scale).clamp(0.0, 255.0) as u8,
        (desired + (color[2] as f32 - target_luma) * chroma_scale).clamp(0.0, 255.0) as u8,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangahue_config::EngineConfig;

    #[test]
    fn rasterizes_user_hint() {
        let config = EngineConfig::default();
        let hint = ColorHint {
            x: 10,
            y: 10,
            radius: 4,
            color: ColorRgb::new(255, 0, 0),
            label: Some("hair".to_string()),
            strength: 1.0,
            source: HintSource::User,
        };
        let hints = UserHintFile { hints: vec![hint] };
        let result = generate_hints(
            ImageSize::new(32, 32),
            Some(&hints),
            &StyleProfile::default(),
            &[],
            &config.hints,
        );
        assert!(result.applied_hints >= 1);
        assert!(result.hint_mask.get_pixel(10, 10)[0] > 0);
    }

    #[test]
    fn character_memory_can_shadow_labeled_user_hint() {
        let character = CharacterProfile {
            id: "hero".to_string(),
            name: "Hero".to_string(),
            aliases: Vec::new(),
            hair_color: Some(ColorRgb::new(1, 2, 3)),
            skin_color: None,
            eye_color: None,
            outfit_colors: Vec::new(),
            accent_colors: Vec::new(),
            confidence: 0.9,
            reference_images: Vec::new(),
            notes: None,
        };
        let hint = ColorHint {
            x: 10,
            y: 10,
            radius: 4,
            color: ColorRgb::new(255, 0, 0),
            label: Some("hero.hair".to_string()),
            strength: 1.0,
            source: HintSource::User,
        };
        let memory = memory_hint_from_user_hint(&hint, &[character]).unwrap();
        assert_eq!(memory.color, ColorRgb::new(1, 2, 3));
        assert!(memory.radius > hint.radius);
    }

    #[test]
    fn region_locks_rasterize_normalized_shapes_and_exclusions() {
        let file = RegionLockFile {
            version: 1,
            page_id: Some("page_001".to_string()),
            coordinate_space: RegionCoordinateSpace::Normalized,
            source_width: None,
            source_height: None,
            locks: vec![
                RegionLock {
                    id: "hero_hair".to_string(),
                    label: Some("Hero hair".to_string()),
                    semantic: RegionSemantic::Hair,
                    mode: RegionLockMode::Colorize,
                    region: RegionShape::Ellipse {
                        center_x: 0.5,
                        center_y: 0.5,
                        radius_x: 0.25,
                        radius_y: 0.25,
                    },
                    color: Some(ColorRgb::new(210, 40, 80)),
                    strength: 1.0,
                    feather_radius: 0.0,
                    enabled: true,
                },
                RegionLock {
                    id: "bubble".to_string(),
                    label: None,
                    semantic: RegionSemantic::Speech,
                    mode: RegionLockMode::Exclude,
                    region: RegionShape::Rectangle {
                        x: 0.0,
                        y: 0.0,
                        width: 0.25,
                        height: 0.25,
                    },
                    color: None,
                    strength: 1.0,
                    feather_radius: 0.0,
                    enabled: true,
                },
            ],
        };
        let raster = rasterize_region_locks(&file, ImageSize::new(64, 64)).unwrap();
        assert_eq!(raster.applied_locks, 2);
        assert!(raster.color_mask.get_pixel(32, 32)[0] > 0);
        assert!(raster.exclude_mask.get_pixel(4, 4)[0] > 0);
        assert_eq!(raster.color_mask.get_pixel(2, 60)[0], 0);
    }

    #[test]
    fn region_lock_enforcement_preserves_ink_and_luminance() {
        let file = RegionLockFile {
            version: 1,
            page_id: None,
            coordinate_space: RegionCoordinateSpace::Pixels,
            source_width: Some(16),
            source_height: Some(16),
            locks: vec![RegionLock {
                id: "coat".to_string(),
                label: None,
                semantic: RegionSemantic::Clothing,
                mode: RegionLockMode::Colorize,
                region: RegionShape::Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: 16.0,
                    height: 16.0,
                },
                color: Some(ColorRgb::new(20, 70, 220)),
                strength: 1.0,
                feather_radius: 0.0,
                enabled: true,
            }],
        };
        let raster = rasterize_region_locks(&file, ImageSize::new(16, 16)).unwrap();
        let mut output = RgbImage::from_pixel(16, 16, Rgb([150, 150, 150]));
        let gray = GrayImage::from_pixel(16, 16, Luma([150]));
        let mut ink = GrayImage::from_pixel(16, 16, Luma([0]));
        ink.put_pixel(8, 8, Luma([255]));
        enforce_region_locks(&mut output, &gray, &ink, &raster);
        assert!(output.get_pixel(2, 2)[2] > output.get_pixel(2, 2)[0]);
        assert_eq!(*output.get_pixel(8, 8), Rgb([150, 150, 150]));
    }

    #[test]
    fn hint_propagation_is_bounded_and_respects_ink_barriers() {
        let mut map = RgbImage::from_pixel(32, 16, Rgb([0, 0, 0]));
        let mut mask = GrayImage::from_pixel(32, 16, Luma([0]));
        let mut lineart = GrayImage::from_pixel(32, 16, Luma([255]));
        map.put_pixel(4, 8, Rgb([220, 30, 40]));
        mask.put_pixel(4, 8, Luma([255]));
        for y in 0..16 {
            lineart.put_pixel(16, y, Luma([0]));
        }
        let (_, propagated) = propagate_hints(&map, &mask, &lineart, 20, 0.2);
        assert!(propagated.get_pixel(10, 8)[0] > 0);
        assert_eq!(propagated.get_pixel(20, 8)[0], 0);

        let (_, unchanged) = propagate_hints(&map, &mask, &lineart, 0, 0.2);
        assert_eq!(unchanged, mask);
    }

    #[test]
    fn region_style_director_seeds_enclosed_materials_and_excludes_bubbles() {
        let mut config = EngineConfig::default();
        config.hints.style_hints_enabled = false;
        config.hints.region_style_hints_enabled = true;
        config.hints.region_style_hint_budget = 8;
        let size = ImageSize::new(96, 64);
        let mut result = generate_hints(size, None, &StyleProfile::default(), &[], &config.hints);
        let mut lineart = GrayImage::from_pixel(96, 64, Luma([255]));
        let mut bubbles = GrayImage::from_pixel(96, 64, Luma([0]));
        for x in 8..40 {
            lineart.put_pixel(x, 8, Luma([0]));
            lineart.put_pixel(x, 40, Luma([0]));
        }
        for y in 8..=40 {
            lineart.put_pixel(8, y, Luma([0]));
            lineart.put_pixel(40, y, Luma([0]));
        }
        for x in 54..86 {
            lineart.put_pixel(x, 8, Luma([0]));
            lineart.put_pixel(x, 40, Luma([0]));
        }
        for y in 8..=40 {
            lineart.put_pixel(54, y, Luma([0]));
            lineart.put_pixel(86, y, Luma([0]));
        }
        for y in 9..40 {
            for x in 55..86 {
                bubbles.put_pixel(x, y, Luma([255]));
            }
        }

        let applied = apply_region_style_hints(
            &mut result,
            &lineart,
            &bubbles,
            &StyleProfile::default(),
            &config.hints,
        );

        assert_eq!(applied, 1);
        assert!((9..40).any(|y| (9..40).any(|x| result.hint_mask.get_pixel(x, y)[0] > 0)));
        assert!((9..40).all(|y| (55..86).all(|x| result.hint_mask.get_pixel(x, y)[0] == 0)));
    }
}
