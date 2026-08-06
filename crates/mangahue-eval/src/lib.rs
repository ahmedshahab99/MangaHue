use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub line_preservation_score: f32,
    pub bubble_cleanliness_score: f32,
    pub protected_region_violation_score: f32,
    pub hint_adherence_score: f32,
    pub output_sharpness_estimate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceMatchReport {
    /// 0-100 overall closeness to the reference photo.
    pub match_score: f32,
    /// LAB L-channel mean/std distance (grading match).
    pub lab_lightness_distance: f32,
    /// LAB a/b channel distance (hue/saturation match).
    pub lab_chroma_distance: f32,
    /// Fraction of palette (k-means) distance, 0 = identical.
    pub palette_distance: f32,
    /// Per-material tint deltas (skin/hair/bg) in 0-100.
    pub region_tint_delta: f32,
}

/// Compare a colorized output against a reference photo.
///
/// The reference photo is the target look (e.g. a professionally colored
/// page). Scores are computed in LAB space so "match" means similar
/// lightness, hue and saturation distribution, not identical pixels.
pub fn reference_match_score(
    output: &RgbImage,
    reference: &RgbImage,
    line_mask: &GrayImage,
) -> ReferenceMatchReport {
    let o_stats = lab_stats(output);
    let r_stats = lab_stats(reference);

    // Lightness distance: mean + std of L channel.
    let lightness_dist = (o_stats.l_mean - r_stats.l_mean).abs() / 100.0
        + (o_stats.l_std - r_stats.l_std).abs() / 50.0;

    // Chroma distance: a/b mean + std.
    let chroma_dist = ((o_stats.a_mean - r_stats.a_mean).abs()
        + (o_stats.b_mean - r_stats.b_mean).abs())
        / 128.0
        + ((o_stats.a_std - r_stats.a_std).abs() + (o_stats.b_std - r_stats.b_std).abs()) / 64.0;

    let palette_dist = palette_distance(output, reference, 16);
    let region_delta = region_tint_delta(output, reference, line_mask);

    // All terms are 0-1 here; region_delta is 0-100 and scales the global
    // match proportionally instead of being subtracted in a different unit.
    let match_score = ((1.0 - lightness_dist * 0.4 - chroma_dist * 0.4 - palette_dist * 0.2)
        .clamp(0.0, 1.0)
        * (1.0 - region_delta / 100.0))
        .clamp(0.0, 1.0)
        * 100.0;

    ReferenceMatchReport {
        match_score,
        lab_lightness_distance: lightness_dist,
        lab_chroma_distance: chroma_dist,
        palette_distance: palette_dist,
        region_tint_delta: region_delta,
    }
}

#[derive(Debug, Clone, Copy)]
struct LabStats {
    l_mean: f32,
    l_std: f32,
    a_mean: f32,
    b_mean: f32,
    a_std: f32,
    b_std: f32,
}

fn lab_stats(image: &RgbImage) -> LabStats {
    // Bound memory on full-resolution inputs: sample at most ~1M pixels.
    let total = (image.width() as usize) * (image.height() as usize);
    let step = (total / 1_000_000).max(1);
    let mut l_vals = Vec::with_capacity((total / step).min(1_000_000));
    let mut a_vals = Vec::with_capacity(l_vals.capacity());
    let mut b_vals = Vec::with_capacity(l_vals.capacity());
    for pixel in image.pixels().step_by(step) {
        let (l, a, b) = rgb_to_lab(pixel[0], pixel[1], pixel[2]);
        l_vals.push(l);
        a_vals.push(a);
        b_vals.push(b);
    }
    let l_mean = mean(&l_vals);
    let a_mean = mean(&a_vals);
    let b_mean = mean(&b_vals);
    LabStats {
        l_mean,
        l_std: std_dev(&l_vals, l_mean),
        a_mean,
        b_mean,
        a_std: std_dev(&a_vals, a_mean),
        b_std: std_dev(&b_vals, b_mean),
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn std_dev(values: &[f32], m: f32) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

/// sRGB -> CIELAB (D65).
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = srgb_to_linear(r as f32 / 255.0);
    let g = srgb_to_linear(g as f32 / 255.0);
    let b = srgb_to_linear(b as f32 / 255.0);
    // D65 white point
    let x = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
    let fx = lab_f(x / 0.95047);
    let fy = lab_f(y / 1.0);
    let fz = lab_f(z / 1.08883);
    (
        (116.0 * fy - 16.0),
        (500.0 * (fx - fy)),
        (200.0 * (fy - fz)),
    )
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_f(t: f32) -> f32 {
    if t > 0.008856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

/// Simple k-means palette distance (mean per-cluster RGB distance, 0-1).
fn palette_distance(output: &RgbImage, reference: &RgbImage, k: usize) -> f32 {
    let o_palette = kmeans_palette(output, k);
    let r_palette = kmeans_palette(reference, k);
    if o_palette.is_empty() || r_palette.is_empty() {
        return 1.0;
    }
    let mut total = 0.0f32;
    let mut pairs = 0;
    for oc in &o_palette {
        // Nearest reference color.
        let mut best = f32::MAX;
        for rc in &r_palette {
            let d = (oc.0 - rc.0).powi(2) + (oc.1 - rc.1).powi(2) + (oc.2 - rc.2).powi(2);
            if d < best {
                best = d;
            }
        }
        total += (best.sqrt() / 441.67295).min(1.0);
        pairs += 1;
    }
    if pairs == 0 {
        1.0
    } else {
        total / pairs as f32
    }
}

/// Tiny k-means (Forgy init, 6 iterations) on downsampled pixels.
fn kmeans_palette(image: &RgbImage, k: usize) -> Vec<(f32, f32, f32)> {
    let step = ((image.width() * image.height()) / 4000).max(1) as usize;
    let pixels: Vec<(f32, f32, f32)> = image
        .pixels()
        .step_by(step)
        .map(|p| (p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();
    if pixels.len() < k {
        return pixels;
    }
    // Deterministic init: evenly spaced pixels.
    let mut centers: Vec<(f32, f32, f32)> = (0..k)
        .map(|i| pixels[(i * pixels.len() / k).min(pixels.len() - 1)])
        .collect();
    for _ in 0..6 {
        let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
        let mut counts = vec![0usize; k];
        for p in &pixels {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for (i, c) in centers.iter().enumerate() {
                let d = (p.0 - c.0).powi(2) + (p.1 - c.1).powi(2) + (p.2 - c.2).powi(2);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            sums[best].0 += p.0;
            sums[best].1 += p.1;
            sums[best].2 += p.2;
            counts[best] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centers[i] = (
                    sums[i].0 / counts[i] as f32,
                    sums[i].1 / counts[i] as f32,
                    sums[i].2 / counts[i] as f32,
                );
            }
        }
    }
    centers
}

/// Compare mean colors of line-delineated regions (skin/hair/bg proxy).
fn region_tint_delta(output: &RgbImage, reference: &RgbImage, line_mask: &GrayImage) -> f32 {
    if output.dimensions() != reference.dimensions()
        || output.dimensions() != line_mask.dimensions()
    {
        return 50.0;
    }
    let mut o_sum = (0.0f32, 0.0f32, 0.0f32);
    let mut r_sum = (0.0f32, 0.0f32, 0.0f32);
    let mut count = 0u32;
    for (x, y, mask) in line_mask.enumerate_pixels() {
        if mask[0] > 100 {
            continue; // Skip ink lines.
        }
        let op = output.get_pixel(x, y);
        let rp = reference.get_pixel(x, y);
        o_sum.0 += op[0] as f32;
        o_sum.1 += op[1] as f32;
        o_sum.2 += op[2] as f32;
        r_sum.0 += rp[0] as f32;
        r_sum.1 += rp[1] as f32;
        r_sum.2 += rp[2] as f32;
        count += 1;
    }
    if count == 0 {
        return 50.0;
    }
    let delta = ((o_sum.0 - r_sum.0).abs() + (o_sum.1 - r_sum.1).abs() + (o_sum.2 - r_sum.2).abs())
        / (3.0 * 255.0 * count as f32);
    (delta * 100.0).min(100.0)
}

pub fn evaluate_output(
    output: &RgbImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
) -> QualityReport {
    evaluate_output_with_guidance(output, line_mask, bubble_mask, None, None, None)
}

pub fn evaluate_output_with_guidance(
    output: &RgbImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: Option<&GrayImage>,
    hint_map: Option<&RgbImage>,
    hint_mask: Option<&GrayImage>,
) -> QualityReport {
    QualityReport {
        line_preservation_score: line_preservation_score(output, line_mask),
        bubble_cleanliness_score: bubble_cleanliness_score(output, bubble_mask),
        protected_region_violation_score: protected_region_violation_score(
            output,
            protected_mask.unwrap_or(bubble_mask),
        ),
        hint_adherence_score: match (hint_map, hint_mask) {
            (Some(hint_map), Some(hint_mask)) => hint_adherence_score(output, hint_map, hint_mask),
            _ => 1.0,
        },
        output_sharpness_estimate: output_sharpness_estimate(output),
    }
}

pub fn line_preservation_score(output: &RgbImage, line_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut dark = 0u32;
    for (x, y, mask) in line_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
        if luma < 48 {
            dark += 1;
        }
    }
    // A page with no detectable line work cannot violate line preservation.
    if samples == 0 {
        1.0
    } else {
        ratio(dark, samples)
    }
}

pub fn bubble_cleanliness_score(output: &RgbImage, bubble_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut clean = 0u32;
    for (x, y, mask) in bubble_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let max_delta = pixel[0].abs_diff(pixel[1]).max(pixel[1].abs_diff(pixel[2]));
        if pixel[0] > 235 && pixel[1] > 235 && pixel[2] > 235 && max_delta < 8 {
            clean += 1;
        }
    }
    if samples == 0 {
        1.0
    } else {
        ratio(clean, samples)
    }
}

pub fn protected_region_violation_score(output: &RgbImage, protected_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut clean = 0u32;
    for (x, y, mask) in protected_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let chroma = pixel[0].abs_diff(pixel[1]).max(pixel[1].abs_diff(pixel[2]));
        let bright = pixel[0] > 230 && pixel[1] > 230 && pixel[2] > 230;
        let dark = pixel[0] < 36 && pixel[1] < 36 && pixel[2] < 36;
        if bright || dark || chroma < 8 {
            clean += 1;
        }
    }
    if samples == 0 {
        1.0
    } else {
        ratio(clean, samples)
    }
}

pub fn hint_adherence_score(output: &RgbImage, hint_map: &RgbImage, hint_mask: &GrayImage) -> f32 {
    if output.dimensions() != hint_map.dimensions() || output.dimensions() != hint_mask.dimensions()
    {
        return 0.0;
    }
    let mut weighted_error = 0.0f32;
    let mut total_weight = 0.0f32;
    for (x, y, mask) in hint_mask.enumerate_pixels() {
        let weight = mask[0] as f32 / 255.0;
        if weight <= 0.02 {
            continue;
        }
        let output = output.get_pixel(x, y);
        let hint = hint_map.get_pixel(x, y);
        let dr = output[0] as f32 - hint[0] as f32;
        let dg = output[1] as f32 - hint[1] as f32;
        let db = output[2] as f32 - hint[2] as f32;
        let error = (dr * dr + dg * dg + db * db).sqrt() / 441.67295;
        weighted_error += error * weight;
        total_weight += weight;
    }
    if total_weight == 0.0 {
        1.0
    } else {
        (1.0 - weighted_error / total_weight).clamp(0.0, 1.0)
    }
}

pub fn output_sharpness_estimate(output: &RgbImage) -> f32 {
    if output.width() < 3 || output.height() < 3 {
        return 0.0;
    }
    let mut total = 0f32;
    let mut samples = 0u32;
    for y in 1..output.height() - 1 {
        for x in 1..output.width() - 1 {
            let center = luma(output.get_pixel(x, y).0);
            let right = luma(output.get_pixel(x + 1, y).0);
            let down = luma(output.get_pixel(x, y + 1).0);
            total += (center - right).abs() + (center - down).abs();
            samples += 2;
        }
    }
    (total / samples.max(1) as f32 / 255.0).clamp(0.0, 1.0)
}

fn luma(pixel: [u8; 3]) -> f32 {
    0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32
}

fn ratio(count: u32, total: u32) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}
