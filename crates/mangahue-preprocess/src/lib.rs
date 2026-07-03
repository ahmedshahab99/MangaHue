use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma};
use mangahue_config::PreprocessSection;
use mangahue_core::BoundingBox;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Named constants (replacing magic numbers)
// ---------------------------------------------------------------------------

/// Margin subtracted from the background level to determine the cutoff for
/// background whitening.
const BACKGROUND_MARGIN: u8 = 18;

/// Amount subtracted from the estimated background level to derive a single
/// global threshold in legacy threshold calculations.
#[allow(dead_code)]
const THRESHOLD_DELTA: u8 = 95;

/// Lower clamp for the legacy global adaptive threshold.
#[allow(dead_code)]
const THRESHOLD_CLAMP_MIN: u8 = 72;

/// Upper clamp for the legacy global adaptive threshold.
#[allow(dead_code)]
const THRESHOLD_CLAMP_MAX: u8 = 150;

/// Pixels darker than this value are candidates for speckle removal.
const SPECKLE_MIN: u8 = 96;

/// Maximum neighbour value still considered "dark" during speckle removal.
const SPECKLE_MAX: u8 = 120;

/// Line-density warning: below this value the page may be washed out.
const LINE_DENSITY_LOW_WARN: f32 = 0.005;

/// Line-density warning: above this value the page is heavily inked.
const LINE_DENSITY_HIGH_WARN: f32 = 0.42;

/// Screen-tone coverage warning: above this value the page has heavy texture.
const SCREEN_TONE_WARN: f32 = 0.18;

/// Dynamic-range constant for the Sauvola binarization formula.
const SAUVOLA_R: f32 = 128.0;

/// Number of largest ink components included in the preprocess report.
const TOP_COMPONENTS_COUNT: usize = 12;

/// Variance ratio threshold below which the histogram is considered nearly
/// unimodal and Otsu's method falls back to percentile estimation.
const OTSU_UNIMODAL_RATIO: f32 = 0.3;

// ---------------------------------------------------------------------------
// Advanced configuration struct
// ---------------------------------------------------------------------------

/// Fine-tuning knobs for the advanced preprocessing pipeline.
///
/// Every value has a sensible default so you can start with
/// `AdvancedPreprocessConfig::default()`.
#[derive(Debug, Clone)]
pub struct AdvancedPreprocessConfig {
    /// Local window radius for Sauvola adaptive thresholding (pixels).
    pub window_radius: u32,
    /// Sensitivity parameter `k` in the Sauvola formula.  Higher values produce
    /// a more aggressive binarization.
    pub sauvola_k: f32,
    /// Minimum pixel area for a region to be considered a speech bubble.
    pub min_bubble_area: u64,
    /// Lower bound of the circularity range for bubble-like shapes.
    pub bubble_circularity_min: f32,
    /// Upper bound of the circularity range for bubble-like shapes.
    pub bubble_circularity_max: f32,
    /// Smallest stroke width (px) that qualifies a pixel as text in the
    /// stroke-width transform.
    pub min_text_stroke_width: u32,
    /// Largest stroke width (px) that qualifies a pixel as text.
    pub max_text_stroke_width: u32,
    /// Margin subtracted from the background level during background whitening.
    pub background_margin: u8,
    /// Pixels darker than this are candidates for speckle removal.
    pub speckle_min: u8,
    /// Neighbour value threshold for the "dark neighbour" test in speckle
    /// removal.
    pub speckle_max: u8,
}

impl Default for AdvancedPreprocessConfig {
    fn default() -> Self {
        Self {
            window_radius: 15,
            sauvola_k: 0.5,
            min_bubble_area: 1000,
            bubble_circularity_min: 0.7,
            bubble_circularity_max: 1.3,
            min_text_stroke_width: 1,
            max_text_stroke_width: 4,
            background_margin: BACKGROUND_MARGIN,
            speckle_min: SPECKLE_MIN,
            speckle_max: SPECKLE_MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// Report & result types
// ---------------------------------------------------------------------------

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
    /// Number of bubble candidates that passed the contour circularity check.
    pub contour_count: usize,
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

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn preprocess_page(
    image: &DynamicImage,
    config: &PreprocessSection,
    advanced: &AdvancedPreprocessConfig,
) -> Result<PreprocessResult> {
    let original_grayscale = image.to_luma8();
    let background_level = estimate_background_level(&original_grayscale);
    let mut clean_lineart = if config.background_cleanup {
        whiten_background(
            &original_grayscale,
            background_level,
            advanced.background_margin,
        )
    } else {
        original_grayscale.clone()
    };

    let mut speckle_pixels_removed = 0;
    if config.denoise_enabled {
        for _ in 0..config.denoise_strength.max(1) {
            let (next, removed) = remove_speckles_with_stats(
                &clean_lineart,
                advanced.speckle_min,
                advanced.speckle_max,
            );
            speckle_pixels_removed += removed;
            clean_lineart = next;
        }
    }

    if config.line_sharpening > 0.0 {
        clean_lineart = sharpen_ink(&clean_lineart, config.line_sharpening);
    }

    let threshold_map =
        adaptive_threshold(&clean_lineart, advanced.window_radius, advanced.sauvola_k);
    let line_mask = create_line_mask(&clean_lineart, &threshold_map);
    let tone_mask = create_tone_mask(&clean_lineart, &line_mask);
    let line_density =
        count_active(&line_mask) as f32 / (line_mask.width() * line_mask.height()) as f32;
    let (bubble_mask, bubble_regions, contour_count) =
        detect_speech_bubbles_basic(&clean_lineart, &line_mask, advanced);
    let text_mask = detect_text_like_dark_strokes(
        &line_mask,
        &bubble_mask,
        advanced.min_text_stroke_width,
        advanced.max_text_stroke_width,
    );
    let protected_mask = create_protected_mask(&bubble_mask, &text_mask);
    let components = connected_components(&line_mask);
    let top_ink_components = top_components(&components, TOP_COMPONENTS_COUNT);
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

    let representative_threshold = median_of_threshold_map(&threshold_map);

    let mut warnings = Vec::new();
    if line_density < LINE_DENSITY_LOW_WARN {
        warnings.push(
            "very low ink density; input may be washed out or not manga line art".to_string(),
        );
    }
    if line_density > LINE_DENSITY_HIGH_WARN {
        warnings.push(
            "very high ink density; page may need stronger cleanup before inference".to_string(),
        );
    }
    if screen_tone_score > SCREEN_TONE_WARN {
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
            adaptive_threshold: representative_threshold,
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
            contour_count,
            warnings,
        },
    })
}

// ---------------------------------------------------------------------------
// Background estimation — Otsu's method with percentile fall-back
// ---------------------------------------------------------------------------

/// Estimates the page background level using Otsu's method.
///
/// If the histogram is nearly unimodal (between-class / total variance ratio
/// below `OTSU_UNIMODAL_RATIO`), the function falls back to the 92nd
/// percentile.
pub fn estimate_background_level(gray: &GrayImage) -> u8 {
    let total = (gray.width() as u64) * (gray.height() as u64);
    let mut histogram = [0u64; 256];
    for pixel in gray.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    // Cumulative sum for variance calculation.
    let mut sum = 0u64;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i as u64 * count;
    }

    let mut w_b = 0u64;
    let mut sum_b = 0u64;
    let mut max_between = 0.0f64;
    let mut otsu_threshold = 0u8;

    for (t, &count) in histogram.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as u64 * count;
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let between = (w_b * w_f) as f64 * (m_b - m_f).powi(2);
        if between > max_between {
            max_between = between;
            otsu_threshold = t as u8;
        }
    }

    // Total variance for the unimodality check.
    let mean = sum as f64 / total as f64;
    let mut total_var = 0.0f64;
    for (i, &count) in histogram.iter().enumerate() {
        total_var += count as f64 * (i as f64 - mean).powi(2);
    }
    total_var /= total as f64;

    let variance_ratio = if total_var > 0.0 {
        max_between / ((total as f64) * total_var)
    } else {
        0.0
    };

    if variance_ratio < OTSU_UNIMODAL_RATIO as f64 {
        return percentile_background(&histogram, total, 92);
    }

    // Cluster mean of the brighter class.
    let mut bright_sum = 0u64;
    let mut bright_count = 0u64;
    for (i, &count) in histogram
        .iter()
        .enumerate()
        .skip(otsu_threshold as usize + 1)
    {
        bright_sum += i as u64 * count;
        bright_count += count;
    }
    bright_sum
        .checked_div(bright_count)
        .map(|value| value as u8)
        .unwrap_or(otsu_threshold)
}

fn percentile_background(histogram: &[u64; 256], total: u64, pct: u64) -> u8 {
    let target = (total * pct / 100) as u32;
    let mut cumulative = 0u32;
    for (value, &count) in histogram.iter().enumerate() {
        cumulative += count as u32;
        if cumulative >= target {
            return value as u8;
        }
    }
    255
}

// ---------------------------------------------------------------------------
// Background whitening
// ---------------------------------------------------------------------------

pub fn whiten_background(gray: &GrayImage, background_level: u8, margin: u8) -> GrayImage {
    let cutoff = background_level.saturating_sub(margin);
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

// ---------------------------------------------------------------------------
// Speckle removal
// ---------------------------------------------------------------------------

pub fn remove_speckles(gray: &GrayImage) -> GrayImage {
    remove_speckles_with_stats(gray, SPECKLE_MIN, SPECKLE_MAX).0
}

pub fn remove_speckles_with_stats(
    gray: &GrayImage,
    speckle_min: u8,
    speckle_max: u8,
) -> (GrayImage, u64) {
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
            if value > speckle_min {
                continue;
            }
            let mut dark_neighbors = 0u8;
            for ny in y - 1..=y + 1 {
                for nx in x - 1..=x + 1 {
                    if nx == x && ny == y {
                        continue;
                    }
                    if gray.get_pixel(nx, ny)[0] < speckle_max {
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

// ---------------------------------------------------------------------------
// Unsharp mask sharpening (replaces per-pixel exponential curve)
// ---------------------------------------------------------------------------

/// Sharpens ink lines using an unsharp mask with a 3×3 Gaussian blur kernel.
///
/// The operation is `original + strength · (original − blurred)`, clamped to
/// `[0, 255]`.
pub fn sharpen_ink(gray: &GrayImage, strength: f32) -> GrayImage {
    let (w, h) = gray.dimensions();
    let blurred = blur_gaussian_3x3(gray);
    let mut output = GrayImage::new(w, h);
    let strength = strength.clamp(0.0, 1.0);

    for y in 0..h {
        for x in 0..w {
            let orig = gray.get_pixel(x, y)[0] as f32;
            let blur = blurred.get_pixel(x, y)[0] as f32;
            let sharp = orig + strength * (orig - blur);
            output.put_pixel(x, y, Luma([sharp.clamp(0.0, 255.0) as u8]));
        }
    }
    output
}

/// Approximates a 3×3 Gaussian blur with the kernel
/// `[[1,2,1],[2,4,2],[1,2,1]] / 16`.
fn blur_gaussian_3x3(gray: &GrayImage) -> GrayImage {
    let (w, h) = gray.dimensions();
    let mut result = GrayImage::new(w, h);

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let sum = gray.get_pixel(x - 1, y - 1)[0] as u32
                + gray.get_pixel(x, y - 1)[0] as u32 * 2
                + gray.get_pixel(x + 1, y - 1)[0] as u32
                + gray.get_pixel(x - 1, y)[0] as u32 * 2
                + gray.get_pixel(x, y)[0] as u32 * 4
                + gray.get_pixel(x + 1, y)[0] as u32 * 2
                + gray.get_pixel(x - 1, y + 1)[0] as u32
                + gray.get_pixel(x, y + 1)[0] as u32 * 2
                + gray.get_pixel(x + 1, y + 1)[0] as u32;
            result.put_pixel(x, y, Luma([(sum / 16) as u8]));
        }
    }
    // Copy edges verbatim.
    for x in 0..w {
        result.put_pixel(x, 0, *gray.get_pixel(x, 0));
        result.put_pixel(x, h - 1, *gray.get_pixel(x, h - 1));
    }
    for y in 0..h {
        result.put_pixel(0, y, *gray.get_pixel(0, y));
        result.put_pixel(w - 1, y, *gray.get_pixel(w - 1, y));
    }
    result
}

// ---------------------------------------------------------------------------
// Sauvola local adaptive thresholding
// ---------------------------------------------------------------------------

/// Produces a per-pixel threshold map using Sauvola binarization.
///
/// For each pixel in a local window of `window_radius`:
///
/// ```text
/// T = mean · (1 + k · (stddev / R − 1))
/// ```
///
/// where `R = 128`.
pub fn adaptive_threshold(gray: &GrayImage, window_radius: u32, k: f32) -> GrayImage {
    let (w, h) = gray.dimensions();
    let integral = compute_integral_image(gray);
    let sq_integral = compute_squared_integral_image(gray);
    let mut threshold = GrayImage::new(w, h);
    let r = SAUVOLA_R;

    for y in 0..h {
        for x in 0..w {
            let (mean, stddev) =
                integral_window_stats(&integral, &sq_integral, x, y, window_radius, w, h);
            let t = mean * (1.0 + k * (stddev / r - 1.0));
            threshold.put_pixel(x, y, Luma([t.clamp(0.0, 255.0) as u8]));
        }
    }
    threshold
}

fn median_of_threshold_map(map: &GrayImage) -> u8 {
    let mut values: Vec<u8> = map.pixels().map(|p| p[0]).collect();
    if values.is_empty() {
        return 128;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn compute_integral_image(gray: &GrayImage) -> Vec<u64> {
    let (w, h) = gray.dimensions();
    let stride = (w + 1) as usize;
    let mut integral = vec![0u64; stride * (h as usize + 1)];
    for y in 0..h {
        for x in 0..w {
            let val = gray.get_pixel(x, y)[0] as u64;
            let i = (y as usize + 1) * stride + (x as usize + 1);
            let left = integral[(y as usize + 1) * stride + x as usize];
            let top = integral[y as usize * stride + (x as usize + 1)];
            let top_left = integral[y as usize * stride + x as usize];
            integral[i] = val + left + top - top_left;
        }
    }
    integral
}

fn compute_squared_integral_image(gray: &GrayImage) -> Vec<u64> {
    let (w, h) = gray.dimensions();
    let stride = (w + 1) as usize;
    let mut integral = vec![0u64; stride * (h as usize + 1)];
    for y in 0..h {
        for x in 0..w {
            let v = gray.get_pixel(x, y)[0] as u64;
            let val = v * v;
            let i = (y as usize + 1) * stride + (x as usize + 1);
            let left = integral[(y as usize + 1) * stride + x as usize];
            let top = integral[y as usize * stride + (x as usize + 1)];
            let top_left = integral[y as usize * stride + x as usize];
            integral[i] = val + left + top - top_left;
        }
    }
    integral
}

fn integral_window_stats(
    integral: &[u64],
    sq_integral: &[u64],
    x: u32,
    y: u32,
    radius: u32,
    w: u32,
    h: u32,
) -> (f32, f32) {
    let x1 = x.saturating_sub(radius);
    let y1 = y.saturating_sub(radius);
    let x2 = (x + radius + 1).min(w);
    let y2 = (y + radius + 1).min(h);
    let n = ((x2 - x1) * (y2 - y1)) as f32;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let stride = (w + 1) as usize;
    let a = y1 as usize * stride + x1 as usize;
    let b = y1 as usize * stride + x2 as usize;
    let c = y2 as usize * stride + x1 as usize;
    let d = y2 as usize * stride + x2 as usize;

    let sum = (integral[d] + integral[a] - integral[b] - integral[c]) as f32;
    let sq_sum = (sq_integral[d] + sq_integral[a] - sq_integral[b] - sq_integral[c]) as f32;

    let mean = sum / n;
    let variance = (sq_sum / n) - (mean * mean);
    let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    (mean, stddev)
}

// ---------------------------------------------------------------------------
// Line / tone masks
// ---------------------------------------------------------------------------

pub fn create_line_mask(gray: &GrayImage, threshold: &GrayImage) -> GrayImage {
    let mut mask = GrayImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let active = if pixel[0] <= threshold.get_pixel(x, y)[0] {
            255
        } else {
            0
        };
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

// ---------------------------------------------------------------------------
// Speech-bubble detection (flood-fill + contour circularity second pass)
// ---------------------------------------------------------------------------

pub fn detect_speech_bubbles_basic(
    gray: &GrayImage,
    line_mask: &GrayImage,
    advanced: &AdvancedPreprocessConfig,
) -> (GrayImage, usize, usize) {
    let width = gray.width();
    let height = gray.height();
    let mut visited = vec![false; (width as usize) * (height as usize)];
    let min_area = ((width as u64 * height as u64) / 800).max(80);
    let max_area = (width as u64 * height as u64) / 10;
    let mut candidates: Vec<Vec<(u32, u32)>> = Vec::new();

    // First pass: flood-fill white regions and apply original heuristics.
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
            candidates.push(component);
        }
    }

    // Second pass: filter by contour circularity.
    let contour_components: Vec<Vec<(u32, u32)>> = candidates
        .iter()
        .filter(|c| {
            let area = c.len() as f64;
            if area < advanced.min_bubble_area as f64 {
                return false;
            }
            let perimeter = component_perimeter(c, width, height);
            let c_val = circularity(perimeter, area);
            c_val >= advanced.bubble_circularity_min as f64
                && c_val <= advanced.bubble_circularity_max as f64
        })
        .cloned()
        .collect();

    let contour_count = contour_components.len();

    // Fall back to the original flood-fill result if contour check finds
    // nothing.
    let selected: &[Vec<(u32, u32)>] = if contour_components.is_empty() {
        &candidates
    } else {
        &contour_components
    };

    let mut bubble_mask = GrayImage::new(width, height);
    let mut regions = 0usize;
    for component in selected {
        for &(px, py) in component {
            bubble_mask.put_pixel(px, py, Luma([255]));
        }
        regions += 1;
    }

    (bubble_mask, regions, contour_count)
}

/// Counts the 4-connected boundary length of a component.
fn component_perimeter(points: &[(u32, u32)], width: u32, height: u32) -> f64 {
    let set: HashSet<(u32, u32)> = points.iter().copied().collect();
    let mut perimeter = 0f64;
    for &(x, y) in points {
        if x == 0 || !set.contains(&(x - 1, y)) {
            perimeter += 1.0;
        }
        if x + 1 >= width || !set.contains(&(x + 1, y)) {
            perimeter += 1.0;
        }
        if y == 0 || !set.contains(&(x, y - 1)) {
            perimeter += 1.0;
        }
        if y + 1 >= height || !set.contains(&(x, y + 1)) {
            perimeter += 1.0;
        }
    }
    perimeter
}

/// Circularity metric: `perimeter² / (4π · area)`.
///
/// A perfect circle evaluates to 1.0; irregular shapes yield larger values.
fn circularity(perimeter: f64, area: f64) -> f64 {
    if area <= 0.0 {
        return f64::MAX;
    }
    (perimeter * perimeter) / (4.0 * std::f64::consts::PI * area)
}

// ---------------------------------------------------------------------------
// Text-stroke detection via stroke-width transform
// ---------------------------------------------------------------------------

/// Identifies text-like dark strokes using a stroke-width transform.
///
/// For each dark pixel in the line mask, the distance to the nearest
/// background pixel is computed (Manhattan distance transform).  Pixels whose
/// stroke width falls within `[min_sw, max_sw]` **and** have a high density
/// of similarly-stroked neighbours are labelled as text.
pub fn detect_text_like_dark_strokes(
    line_mask: &GrayImage,
    _bubble_mask: &GrayImage,
    min_stroke_width: u32,
    max_stroke_width: u32,
) -> GrayImage {
    let (w, h) = line_mask.dimensions();
    let dist = distance_transform(line_mask);
    let mut text = GrayImage::new(w, h);
    let radius = 3u32;

    for y in 0..h {
        for x in 0..w {
            let sw = dist[idx(x, y, w)];
            if sw < min_stroke_width || sw > max_stroke_width {
                continue;
            }
            let mut similar = 0u32;
            let mut total = 0u32;
            let min_y = y.saturating_sub(radius);
            let max_y = (y + radius + 1).min(h);
            let min_x = x.saturating_sub(radius);
            let max_x = (x + radius + 1).min(w);
            for ny in min_y..max_y {
                for nx in min_x..max_x {
                    total += 1;
                    let nsw = dist[idx(nx, ny, w)];
                    if nsw.max(sw) - nsw.min(sw) <= 1 {
                        similar += 1;
                    }
                }
            }
            if total > 0 && similar >= total / 2 {
                text.put_pixel(x, y, Luma([255]));
            }
        }
    }
    text
}

/// Two-pass Manhattan distance transform.
///
/// Background pixels (line_mask == 0) are seeded with distance 0, foreground
/// pixels are propagated in two raster passes.
fn distance_transform(mask: &GrayImage) -> Vec<u32> {
    let (w, h) = mask.dimensions();
    let size = (w * h) as usize;
    let max_val = u32::MAX / 2;
    let mut dist = vec![max_val; size];

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] == 0 {
                dist[idx(x, y, w)] = 0;
            }
        }
    }

    // Forward pass (top-left → bottom-right).
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            if dist[i] == 0 {
                continue;
            }
            let mut d = dist[i];
            if x > 0 {
                d = d.min(dist[idx(x - 1, y, w)].saturating_add(1));
            }
            if y > 0 {
                d = d.min(dist[idx(x, y - 1, w)].saturating_add(1));
            }
            dist[i] = d;
        }
    }

    // Backward pass (bottom-right → top-left).
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = idx(x, y, w);
            let mut d = dist[i];
            if x + 1 < w {
                d = d.min(dist[idx(x + 1, y, w)].saturating_add(1));
            }
            if y + 1 < h {
                d = d.min(dist[idx(x, y + 1, w)].saturating_add(1));
            }
            dist[i] = d;
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Protected mask
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Connected components analysis
// ---------------------------------------------------------------------------

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

#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};
    use mangahue_config::EngineConfig;

    #[test]
    fn creates_line_mask_for_dark_ink() {
        let mut gray = GrayImage::from_pixel(8, 8, Luma([255]));
        gray.put_pixel(3, 3, Luma([0]));
        let threshold = GrayImage::from_pixel(8, 8, Luma([100]));
        let mask = create_line_mask(&gray, &threshold);
        assert_eq!(mask.get_pixel(3, 3)[0], 255);
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn preprocess_keeps_dimensions() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(32, 24, Luma([250])));
        let config = EngineConfig::default();
        let advanced = AdvancedPreprocessConfig::default();
        let result = preprocess_page(&img, &config.preprocess, &advanced).unwrap();
        assert_eq!(result.clean_lineart.dimensions(), (32, 24));
        assert_eq!(result.line_mask.dimensions(), (32, 24));
    }
}
