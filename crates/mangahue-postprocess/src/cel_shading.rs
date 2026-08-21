//! Soft cel shading.
//!
//! The DDColor family produces continuous-tone painterly color, the look a
//! camera would capture. Professional anime and manga color uses
//! **cel shading**: discrete flat regions with a small number of brightness
//! bands (classic 2-tone shadow/mid, soft cel 3-tone shadow/mid/highlight).
//!
//! This pass converts the model's continuous output into cel bands while
//! preserving the model's local hue. Per pixel:
//!
//! 1. Compute the local "region base color" by averaging a window of
//!    pixels. This captures the hue the model painted for the region.
//! 2. Classify the pixel's brightness into a band (shadow / mid /
//!    highlight) using the pixel's own luma — NOT a ratio to the base.
//!    The ratio approach breaks when the window is diluted by surrounding
//!    paper: a colored pixel next to paper gets a low ratio even though
//!    it's bright.
//! 3. Map the band to a brightness factor with smoothstep edges so the
//!    transition looks painterly rather than hard.
//! 4. Scale the base color to the band brightness, preserving hue.
//! 5. Blend with the original pixel by `cel_strength` so the model keeps
//!    final say on detail (edge contours, fine shading) while the cel
//!    bands carry the overall impression.
//!
//! Ink, speech bubbles, and explicitly protected regions are never
//! touched, so line art and text stay clean.

use image::{GrayImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CelShadingReport {
    pub pixels_shaded: u64,
}

/// Compute the soft cel shading.
#[allow(clippy::too_many_arguments)]
pub fn apply_cel_shading(
    image: &mut RgbImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: Option<&GrayImage>,
    shadow_threshold: f32,
    highlight_threshold: f32,
    shadow_factor: f32,
    highlight_factor: f32,
    smoothing: f32,
    cel_strength: f32,
    window_radius: u8,
) -> CelShadingReport {
    let mut report = CelShadingReport::default();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return report;
    }
    if image.dimensions() != line_mask.dimensions()
        || image.dimensions() != bubble_mask.dimensions()
    {
        return report;
    }

    let radius = window_radius.max(1) as i32;
    let smoothing = smoothing.clamp(0.01, 0.5);
    let shadow_factor = shadow_factor.clamp(0.1, 0.95);
    let highlight_factor = highlight_factor.clamp(1.0, 2.0);
    let strength = cel_strength.clamp(0.0, 1.0);

    // shadow_threshold and highlight_threshold are in 0..1 (normalized luma).
    let shadow_threshold = shadow_threshold.clamp(0.10, 0.60);
    let highlight_threshold = highlight_threshold.max(shadow_threshold + 0.15);

    // The original image is read while we write into a fresh buffer so a
    // pixel never sees its own recent updates as the "base" of its
    // neighbor's average.
    let mut next = image.clone();

    for y in 0..height {
        for x in 0..width {
            // Skip ink, bubbles, and protected regions.
            if line_mask.get_pixel(x, y)[0] > 184 {
                continue;
            }
            if bubble_mask.get_pixel(x, y)[0] > 96 {
                continue;
            }
            if let Some(pm) = protected_mask {
                if pm.get_pixel(x, y)[0] > 127 {
                    continue;
                }
            }

            // Pixel's own brightness (normalized 0..1).
            let orig = image.get_pixel(x, y);
            let pixel_luma =
                (0.299 * orig[0] as f32 + 0.587 * orig[1] as f32 + 0.114 * orig[2] as f32) / 255.0;

            // Skip near-paper and near-ink: the cel pass is meaningless there.
            if !(0.06..=0.94).contains(&pixel_luma) {
                continue;
            }

            // Local base color from a square window. Used only for hue,
            // NOT for band classification (the ratio approach breaks when
            // the window is diluted by paper).
            let (br, bg, bb, count) = local_mean(&*image, x, y, width, height, radius);
            if count < 4 {
                continue;
            }
            let base_luma = (0.299 * br + 0.587 * bg + 0.114 * bb) / 255.0;
            // Skip if the local base is near-paper: the hue is meaningless
            // in a mostly-paper window.
            if !(0.04..=0.92).contains(&base_luma) {
                continue;
            }

            // Map pixel luma to a band (0 = shadow, 1 = mid, 2 = highlight).
            // Smoothstep edges make transitions painterly.
            let band = luma_to_band(pixel_luma, shadow_threshold, highlight_threshold, smoothing);

            // Brightness factor for this band.
            let factor = cel_factor(band, shadow_factor, highlight_factor);

            // Target brightness for the cel output.
            let target_luma = pixel_luma * factor;

            // Scale the base color to the target brightness while
            // preserving the hue. Each channel scales by the same ratio.
            let scale = target_luma / base_luma.max(0.001);
            let cr = (br * scale).clamp(0.0, 255.0);
            let cg = (bg * scale).clamp(0.0, 255.0);
            let cb = (bb * scale).clamp(0.0, 255.0);

            // Blend with the original so the model keeps detail.
            let r = lerp(orig[0] as f32, cr, strength);
            let g = lerp(orig[1] as f32, cg, strength);
            let b = lerp(orig[2] as f32, cb, strength);
            next.put_pixel(
                x,
                y,
                Rgb([
                    r.round().clamp(0.0, 255.0) as u8,
                    g.round().clamp(0.0, 255.0) as u8,
                    b.round().clamp(0.0, 255.0) as u8,
                ]),
            );
            report.pixels_shaded += 1;
        }
    }

    *image = next;
    report
}

fn local_mean(
    image: &RgbImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    radius: i32,
) -> (f32, f32, f32, u32) {
    let x0 = x.saturating_sub(radius as u32);
    let y0 = y.saturating_sub(radius as u32);
    let x1 = (x + radius as u32 + 1).min(width);
    let y1 = (y + radius as u32 + 1).min(height);
    let mut r = 0.0f64;
    let mut g = 0.0f64;
    let mut b = 0.0f64;
    let mut n = 0u32;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let p = image.get_pixel(xx, yy);
            r += p[0] as f64;
            g += p[1] as f64;
            b += p[2] as f64;
            n += 1;
        }
    }
    let nf = n.max(1) as f64;
    ((r / nf) as f32, (g / nf) as f32, (b / nf) as f32, n)
}

/// Map normalized pixel luma (0..1) to a band number (0 = shadow, 1 = mid,
/// 2 = highlight) with smoothstep edges.
fn luma_to_band(luma: f32, shadow_threshold: f32, highlight_threshold: f32, smoothing: f32) -> f32 {
    // Shadow to mid transition.
    let shadow_low = (shadow_threshold - smoothing).max(0.01);
    let shadow_high = shadow_threshold + smoothing;
    if luma <= shadow_low {
        0.0
    } else if luma < shadow_high {
        let t = (luma - shadow_low) / (shadow_high - shadow_low).max(0.001);
        smoothstep(0.0, 1.0, t)
    } else if luma <= highlight_threshold - smoothing {
        1.0
    } else if luma < highlight_threshold + smoothing {
        let t = (luma - (highlight_threshold - smoothing)) / (2.0 * smoothing).max(0.001);
        1.0 + smoothstep(0.0, 1.0, t)
    } else {
        2.0
    }
}

/// Brightness factor for a band level. 0 = shadow_factor, 1 = 1.0, 2 =
/// highlight_factor. Smoothstep interpolation between bands.
fn cel_factor(band: f32, shadow_factor: f32, highlight_factor: f32) -> f32 {
    let bands = [
        (0.0_f32, shadow_factor),
        (1.0_f32, 1.0_f32),
        (2.0_f32, highlight_factor),
    ];
    let clamped = band.clamp(0.0, 2.0);
    let lower_idx = clamped as usize;
    let lower = bands[lower_idx];
    let upper_idx = (lower_idx + 1).min(2);
    let upper = bands[upper_idx];
    let frac = (clamped - lower.0) / (upper.0 - lower.0).max(0.001);
    let frac = smoothstep(0.0, 1.0, frac);
    lower.1 + (upper.1 - lower.1) * frac
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn cel_pass_preserves_paper_and_ink() {
        let mut img = RgbImage::from_pixel(20, 20, Rgb([240, 240, 240]));
        for x in 0..20 {
            img.put_pixel(x, 10, Rgb([0, 0, 0]));
        }
        let lines = GrayImage::from_pixel(20, 20, Luma([0]));
        let bubbles = GrayImage::from_pixel(20, 20, Luma([0]));
        let report = apply_cel_shading(
            &mut img, &lines, &bubbles, None, 0.40, 0.80, 0.58, 1.18, 0.18, 0.7, 4,
        );
        // All pixels are paper or ink; the cel pass should skip both.
        assert_eq!(report.pixels_shaded, 0);
    }

    #[test]
    fn cel_pass_quantizes_brightness_on_colored_region() {
        // Fill the entire image with a smooth gradient so the local mean
        // is NOT diluted by paper.
        let mut img = RgbImage::new(20, 20);
        for y in 0..20u32 {
            for x in 0..20u32 {
                let r = 80 + ((x + y) * 3) as u8;
                img.put_pixel(x, y, Rgb([r, 50, 70]));
            }
        }
        let lines = GrayImage::from_pixel(20, 20, Luma([0]));
        let bubbles = GrayImage::from_pixel(20, 20, Luma([0]));
        let original_red: Vec<u8> = (0..20u32).map(|x| img.get_pixel(x, 10)[0]).collect();
        let report = apply_cel_shading(
            &mut img, &lines, &bubbles, None, 0.40, 0.80, 0.58, 1.18, 0.18, 0.7, 4,
        );
        assert!(
            report.pixels_shaded > 0,
            "cel pass should shade the colored band"
        );
        let new_red: Vec<u8> = (0..20u32).map(|x| img.get_pixel(x, 10)[0]).collect();
        let changed = original_red
            .iter()
            .zip(new_red.iter())
            .filter(|(a, b)| **a != **b)
            .count();
        assert!(changed > 0, "cel pass should modify at least some pixels");
    }

    #[test]
    fn cel_shadow_factor_is_below_one() {
        let factor = cel_factor(0.0, 0.58, 1.18);
        assert!(factor < 1.0);
    }

    #[test]
    fn cel_highlight_factor_is_above_one() {
        let factor = cel_factor(2.0, 0.58, 1.18);
        assert!(factor > 1.0);
    }

    #[test]
    fn cel_shadow_pixel_classified_correctly() {
        // luma = 0.2, shadow_threshold = 0.4, smoothing = 0.18
        // shadow range: < 0.22 = shadow, 0.22-0.58 = transition
        let band = luma_to_band(0.15, 0.40, 0.80, 0.18);
        assert!(band < 0.1, "luma 0.15 should be deep shadow, got {band}");
    }

    #[test]
    fn cel_mid_pixel_classified_correctly() {
        let band = luma_to_band(0.50, 0.40, 0.80, 0.18);
        assert!(
            (0.85..=1.15).contains(&band),
            "luma 0.50 should be mid, got {band}"
        );
    }

    #[test]
    fn cel_highlight_pixel_classified_correctly() {
        let band = luma_to_band(0.90, 0.40, 0.80, 0.18);
        assert!(band > 1.8, "luma 0.90 should be highlight, got {band}");
    }
}
