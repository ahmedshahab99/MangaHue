use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma};
use mangahue_config::PreprocessSection;
use mangahue_core::BoundingBox;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessReport {
    pub background_level: u8,
    pub adaptive_threshold: u8,
    pub line_density: f32,
    pub ink_component_count: usize,
    pub large_ink_components: usize,
    pub speckle_pixels_removed: u64,
    pub bubble_regions: usize,
    pub bubble_coverage: f32,
    pub text_ink_pixels: u64,
    pub protected_coverage: f32,
    pub screen_tone_score: f32,
    pub page_complexity_score: f32,
    pub top_ink_components: Vec<ComponentStats>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PreprocessResult {
    pub clean_lineart: GrayImage,
    pub original_grayscale: GrayImage,
    pub line_mask: GrayImage,
    pub tone_mask: GrayImage,
    pub bubble_mask: GrayImage,
    pub text_mask: GrayImage,
    pub protected_mask: GrayImage,
    pub report: PreprocessReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStats {
    pub bbox: BoundingBox,
    pub area: u64,
    pub density: f32,
    pub kind: InkComponentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InkComponentKind {
    Speckle,
    Stroke,
    BorderCandidate,
    DenseMass,
    ScreenToneCluster,
}

pub fn preprocess_page(
    image: &DynamicImage,
    config: &PreprocessSection,
) -> Result<PreprocessResult> {
    let original_grayscale = image.to_luma8();
    let background_level = estimate_background_level(&original_grayscale);
    let mut clean_lineart = if config.background_cleanup {
        whiten_background(&original_grayscale, background_level)
    } else {
        original_grayscale.clone()
    };

    let mut speckle_pixels_removed = 0;
    if config.denoise_enabled {
        for _ in 0..config.denoise_strength.max(1) {
            let (next, removed) = remove_speckles_with_stats(&clean_lineart);
            speckle_pixels_removed += removed;
            clean_lineart = next;
        }
    }

    if config.line_sharpening > 0.0 {
        clean_lineart = sharpen_ink(&clean_lineart, config.line_sharpening);
    }

    let threshold = adaptive_threshold(&clean_lineart);
    let line_mask = create_line_mask(&clean_lineart, threshold);
    let tone_mask = create_tone_mask(&clean_lineart, &line_mask);
    let line_density =
        count_active(&line_mask) as f32 / (line_mask.width() * line_mask.height()) as f32;
    let (bubble_mask, bubble_regions) = detect_speech_bubbles_basic(&clean_lineart, &line_mask);
    let text_mask = detect_text_like_dark_strokes(&line_mask, &bubble_mask);
    let protected_mask = create_protected_mask(&bubble_mask, &text_mask);
    let components = connected_components(&line_mask);
    let top_ink_components = top_components(&components, 12);
    let large_ink_components = components
        .iter()
        .filter(|component| {
            matches!(
                component.kind,
                InkComponentKind::BorderCandidate | InkComponentKind::DenseMass
            )
        })
        .count();
    let bubble_coverage = coverage(&bubble_mask);
    let text_ink_pixels = count_active(&text_mask);
    let protected_coverage = coverage(&protected_mask);
    let screen_tone_score = coverage(&tone_mask);
    let page_complexity_score =
        page_complexity_score(line_density, screen_tone_score, components.len());
    let mut warnings = Vec::new();
    if line_density < 0.005 {
        warnings.push(
            "very low ink density; input may be washed out or not manga line art".to_string(),
        );
    }
    if line_density > 0.42 {
        warnings.push(
            "very high ink density; page may need stronger cleanup before inference".to_string(),
        );
    }
    if screen_tone_score > 0.18 {
        warnings.push(
            "heavy screen-tone texture detected; model may need tone-aware training data"
                .to_string(),
        );
    }

    Ok(PreprocessResult {
        clean_lineart,
        original_grayscale,
        line_mask,
        tone_mask,
        bubble_mask,
        text_mask,
        protected_mask,
        report: PreprocessReport {
            background_level,
            adaptive_threshold: threshold,
            line_density,
            ink_component_count: components.len(),
            large_ink_components,
            speckle_pixels_removed,
            bubble_regions,
            bubble_coverage,
            text_ink_pixels,
            protected_coverage,
            screen_tone_score,
            page_complexity_score,
            top_ink_components,
            warnings,
        },
    })
}

pub fn estimate_background_level(gray: &GrayImage) -> u8 {
    let mut histogram = [0u32; 256];
    for pixel in gray.pixels() {
        histogram[pixel[0] as usize] += 1;
    }
    let target = (gray.width() as u64 * gray.height() as u64 * 92 / 100) as u32;
    let mut cumulative = 0u32;
    for (value, count) in histogram.iter().enumerate() {
        cumulative += *count;
        if cumulative >= target {
            return value as u8;
        }
    }
    255
}

pub fn whiten_background(gray: &GrayImage, background_level: u8) -> GrayImage {
    let cutoff = background_level.saturating_sub(18);
    let mut output = gray.clone();
    for pixel in output.pixels_mut() {
        let value = pixel[0];
        let lifted = if value >= cutoff {
            255
        } else {
            let scale = 255.0 / cutoff.max(1) as f32;
            (value as f32 * scale).round().clamp(0.0, 255.0) as u8
        };
        *pixel = Luma([lifted]);
    }
    output
}

pub fn remove_speckles(gray: &GrayImage) -> GrayImage {
    remove_speckles_with_stats(gray).0
}

pub fn remove_speckles_with_stats(gray: &GrayImage) -> (GrayImage, u64) {
    let mut output = gray.clone();
    let width = gray.width();
    let height = gray.height();
    if width < 3 || height < 3 {
        return (output, 0);
    }
    let mut removed = 0u64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let value = gray.get_pixel(x, y)[0];
            if value > 96 {
                continue;
            }
            let mut dark_neighbors = 0u8;
            for ny in y - 1..=y + 1 {
                for nx in x - 1..=x + 1 {
                    if nx == x && ny == y {
                        continue;
                    }
                    if gray.get_pixel(nx, ny)[0] < 120 {
                        dark_neighbors += 1;
                    }
                }
            }
            if dark_neighbors <= 1 {
                output.put_pixel(x, y, Luma([255]));
                removed += 1;
            }
        }
    }
    (output, removed)
}

pub fn sharpen_ink(gray: &GrayImage, strength: f32) -> GrayImage {
    let mut output = gray.clone();
    let strength = strength.clamp(0.0, 1.0);
    for pixel in output.pixels_mut() {
        let value = pixel[0] as f32 / 255.0;
        let curved = if value < 0.55 {
            value * (1.0 - 0.45 * strength)
        } else {
            1.0 - (1.0 - value) * (1.0 + 0.2 * strength)
        };
        *pixel = Luma([(curved * 255.0).round().clamp(0.0, 255.0) as u8]);
    }
    output
}

pub fn adaptive_threshold(gray: &GrayImage) -> u8 {
    let background = estimate_background_level(gray);
    background.saturating_sub(95).clamp(72, 150)
}

pub fn create_line_mask(gray: &GrayImage, threshold: u8) -> GrayImage {
    let mut mask = GrayImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let active = if pixel[0] <= threshold { 255 } else { 0 };
        mask.put_pixel(x, y, Luma([active]));
    }
    mask
}

pub fn create_tone_mask(gray: &GrayImage, line_mask: &GrayImage) -> GrayImage {
    let mut mask = GrayImage::new(gray.width(), gray.height());
    if gray.width() < 5 || gray.height() < 5 {
        return mask;
    }
    for y in 2..gray.height() - 2 {
        for x in 2..gray.width() - 2 {
            if line_mask.get_pixel(x, y)[0] > 0 {
                continue;
            }
            let value = gray.get_pixel(x, y)[0];
            if !(72..=226).contains(&value) {
                continue;
            }
            let mut local_min = 255u8;
            let mut local_max = 0u8;
            let mut mid_count = 0u8;
            for py in y - 2..=y + 2 {
                for px in x - 2..=x + 2 {
                    let sample = gray.get_pixel(px, py)[0];
                    local_min = local_min.min(sample);
                    local_max = local_max.max(sample);
                    if (72..=226).contains(&sample) {
                        mid_count += 1;
                    }
                }
            }
            let texture = local_max.saturating_sub(local_min);
            if mid_count >= 10 && texture >= 18 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
    mask
}

pub fn detect_speech_bubbles_basic(gray: &GrayImage, line_mask: &GrayImage) -> (GrayImage, usize) {
    let width = gray.width();
    let height = gray.height();
    let mut visited = vec![false; (width as usize) * (height as usize)];
    let mut bubble_mask = GrayImage::new(width, height);
    let min_area = ((width as u64 * height as u64) / 800).max(80);
    let max_area = (width as u64 * height as u64) / 10;
    let mut components = 0usize;

    for y in 0..height {
        for x in 0..width {
            let index = idx(x, y, width);
            if visited[index] || gray.get_pixel(x, y)[0] < 246 {
                continue;
            }
            let component = flood_white(gray, &mut visited, x, y);
            let touches_border = component
                .iter()
                .any(|&(px, py)| px == 0 || py == 0 || px + 1 == width || py + 1 == height);
            let area = component.len() as u64;
            if touches_border || area < min_area || area > max_area {
                continue;
            }
            let bbox = component_bbox(&component);
            if !bubble_geometry_is_plausible(bbox, width, height, area) {
                continue;
            }
            let border_ink = border_ink_ratio(line_mask, bbox);
            if border_ink < 0.015 {
                continue;
            }
            for (px, py) in component {
                bubble_mask.put_pixel(px, py, Luma([255]));
            }
            components += 1;
        }
    }

    (bubble_mask, components)
}

fn bubble_geometry_is_plausible(
    bbox: BoundingBox,
    page_width: u32,
    page_height: u32,
    area: u64,
) -> bool {
    if bbox.width > page_width.saturating_mul(7) / 10 {
        return false;
    }
    if bbox.height > page_height.saturating_mul(3) / 10 {
        return false;
    }
    let long_axis = bbox.width.max(bbox.height) as f32;
    let short_axis = bbox.width.min(bbox.height).max(1) as f32;
    if long_axis / short_axis > 4.5 {
        return false;
    }
    let fill_ratio = area as f32 / bbox.area().max(1) as f32;
    (0.45..=0.98).contains(&fill_ratio)
}

pub fn detect_text_like_dark_strokes(line_mask: &GrayImage, bubble_mask: &GrayImage) -> GrayImage {
    let mut text = GrayImage::new(line_mask.width(), line_mask.height());
    for (x, y, pixel) in line_mask.enumerate_pixels() {
        if pixel[0] > 0 && near_mask(bubble_mask, x, y, 4) {
            text.put_pixel(x, y, Luma([255]));
        }
    }
    text
}

pub fn create_protected_mask(bubble_mask: &GrayImage, text_mask: &GrayImage) -> GrayImage {
    let mut protected = bubble_mask.clone();
    for (x, y, pixel) in text_mask.enumerate_pixels() {
        if pixel[0] == 0 {
            continue;
        }
        let min_x = x.saturating_sub(1);
        let min_y = y.saturating_sub(1);
        let max_x = (x + 2).min(protected.width());
        let max_y = (y + 2).min(protected.height());
        for py in min_y..max_y {
            for px in min_x..max_x {
                protected.put_pixel(px, py, Luma([255]));
            }
        }
    }
    protected
}

pub fn connected_components(mask: &GrayImage) -> Vec<ComponentStats> {
    let width = mask.width();
    let height = mask.height();
    let mut visited = vec![false; (width as usize) * (height as usize)];
    let mut components = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = idx(x, y, width);
            if visited[index] || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let points = flood_mask(mask, &mut visited, x, y);
            let bbox = component_bbox(&points);
            let area = points.len() as u64;
            let density = area as f32 / bbox.area().max(1) as f32;
            let kind = classify_component(area, bbox, density, width as u64 * height as u64);
            components.push(ComponentStats {
                bbox,
                area,
                density,
                kind,
            });
        }
    }
    components
}

fn flood_mask(mask: &GrayImage, visited: &mut [bool], x: u32, y: u32) -> Vec<(u32, u32)> {
    let width = mask.width();
    let height = mask.height();
    let mut points = Vec::new();
    let mut queue = VecDeque::from([(x, y)]);
    visited[idx(x, y, width)] = true;

    while let Some((cx, cy)) = queue.pop_front() {
        points.push((cx, cy));
        let neighbors = [
            (cx.wrapping_sub(1), cy),
            (cx + 1, cy),
            (cx, cy.wrapping_sub(1)),
            (cx, cy + 1),
        ];
        for (nx, ny) in neighbors {
            if nx >= width || ny >= height {
                continue;
            }
            let nidx = idx(nx, ny, width);
            if visited[nidx] || mask.get_pixel(nx, ny)[0] == 0 {
                continue;
            }
            visited[nidx] = true;
            queue.push_back((nx, ny));
        }
    }
    points
}

fn classify_component(
    area: u64,
    bbox: BoundingBox,
    density: f32,
    page_area: u64,
) -> InkComponentKind {
    if area <= 3 {
        return InkComponentKind::Speckle;
    }
    let long_axis = bbox.width.max(bbox.height) as f32;
    let short_axis = bbox.width.min(bbox.height).max(1) as f32;
    let elongated = long_axis / short_axis;
    if area > page_area / 80 && density < 0.16 && elongated > 4.0 {
        return InkComponentKind::BorderCandidate;
    }
    if area > page_area / 40 && density < 0.12 {
        return InkComponentKind::BorderCandidate;
    }
    if area > page_area / 120 && density > 0.42 {
        return InkComponentKind::DenseMass;
    }
    if area > 24 && area < page_area / 120 && density < 0.18 {
        return InkComponentKind::ScreenToneCluster;
    }
    InkComponentKind::Stroke
}

fn top_components(components: &[ComponentStats], count: usize) -> Vec<ComponentStats> {
    let mut components = components.to_vec();
    components.sort_by_key(|component| std::cmp::Reverse(component.area));
    components.truncate(count);
    components
}

fn flood_white(gray: &GrayImage, visited: &mut [bool], x: u32, y: u32) -> Vec<(u32, u32)> {
    let width = gray.width();
    let height = gray.height();
    let mut points = Vec::new();
    let mut queue = VecDeque::from([(x, y)]);
    visited[idx(x, y, width)] = true;

    while let Some((cx, cy)) = queue.pop_front() {
        points.push((cx, cy));
        let neighbors = [
            (cx.wrapping_sub(1), cy),
            (cx + 1, cy),
            (cx, cy.wrapping_sub(1)),
            (cx, cy + 1),
        ];
        for (nx, ny) in neighbors {
            if nx >= width || ny >= height {
                continue;
            }
            let nidx = idx(nx, ny, width);
            if visited[nidx] || gray.get_pixel(nx, ny)[0] < 246 {
                continue;
            }
            visited[nidx] = true;
            queue.push_back((nx, ny));
        }
    }
    points
}

fn component_bbox(points: &[(u32, u32)]) -> BoundingBox {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0;
    let mut max_y = 0;
    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
}

fn border_ink_ratio(mask: &GrayImage, bbox: BoundingBox) -> f32 {
    let grow = 3;
    let min_x = bbox.x.saturating_sub(grow);
    let min_y = bbox.y.saturating_sub(grow);
    let max_x = (bbox.right() + grow).min(mask.width());
    let max_y = (bbox.bottom() + grow).min(mask.height());
    let mut samples = 0u32;
    let mut ink = 0u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let on_border = x < bbox.x || x >= bbox.right() || y < bbox.y || y >= bbox.bottom();
            if on_border {
                samples += 1;
                if mask.get_pixel(x, y)[0] > 0 {
                    ink += 1;
                }
            }
        }
    }
    if samples == 0 {
        0.0
    } else {
        ink as f32 / samples as f32
    }
}

fn near_mask(mask: &GrayImage, x: u32, y: u32, radius: u32) -> bool {
    let min_x = x.saturating_sub(radius);
    let min_y = y.saturating_sub(radius);
    let max_x = (x + radius + 1).min(mask.width());
    let max_y = (y + radius + 1).min(mask.height());
    for py in min_y..max_y {
        for px in min_x..max_x {
            if mask.get_pixel(px, py)[0] > 0 {
                return true;
            }
        }
    }
    false
}

fn count_active(mask: &GrayImage) -> u64 {
    mask.pixels().filter(|pixel| pixel[0] > 0).count() as u64
}

fn coverage(mask: &GrayImage) -> f32 {
    count_active(mask) as f32 / (mask.width() as u64 * mask.height() as u64).max(1) as f32
}

fn page_complexity_score(line_density: f32, screen_tone_score: f32, components: usize) -> f32 {
    let component_term = (components as f32 / 1800.0).clamp(0.0, 1.0);
    (line_density * 1.8 + screen_tone_score * 2.4 + component_term * 0.8).clamp(0.0, 1.0)
}

fn idx(x: u32, y: u32, width: u32) -> usize {
    (y as usize) * (width as usize) + x as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};
    use mangahue_config::EngineConfig;

    #[test]
    fn creates_line_mask_for_dark_ink() {
        let mut gray = GrayImage::from_pixel(8, 8, Luma([255]));
        gray.put_pixel(3, 3, Luma([0]));
        let mask = create_line_mask(&gray, 100);
        assert_eq!(mask.get_pixel(3, 3)[0], 255);
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn preprocess_keeps_dimensions() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(32, 24, Luma([250])));
        let config = EngineConfig::default();
        let result = preprocess_page(&img, &config.preprocess).unwrap();
        assert_eq!(result.clean_lineart.dimensions(), (32, 24));
        assert_eq!(result.line_mask.dimensions(), (32, 24));
    }
}
