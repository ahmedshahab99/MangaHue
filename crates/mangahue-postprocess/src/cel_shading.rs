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
//!    pixels. This captures the hue the model painted for the region
//!    without snapping to a palette.
//! 2. Compare the pixel's luma to the base's luma. The ratio tells us
//!    whether this pixel is darker (shadow), similar (mid), or brighter
//!    (highlight) than the region.
//! 3. Map the ratio to one of N discrete brightness levels (the bands),
//!    with smoothstep edges so the transition looks painterly rather than
//!    hard.
//! 4. The output color is the local base hue scaled to the band brightness.
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
    celery_strength: f32,
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
    let strength = celery_strength.clamp(0.0, 1.0);

    // Avoid divide-by-zero or unstable thresholds.
    let shadow_threshold = shadow_threshold.clamp(0.30, 0.95);
    let highlight_threshold = highlight_threshold.max(shadow_threshold + 0.05);

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

            // Local base color from a square window.
            let (br, bg, bb, count) = local_mean(&*image, x, y, width, height, radius);
            if count < 4 {
                continue;
            }
            let base_luma = 0.299 * br + 0.587 * bg + 0.114 * bb;
            // Skip the PIXEL if it's near-paper or near-ink: the cel pass is
            // meaningless there. We rely on the pixel's own luma, not the
            // local mean — the base_luma is the local mean which is often
            // diluted by surrounding paper and would wrongly drop colored
            // pixels whose window is mostly paper. When the base_luma is
            // high, the band ratio will be near 1.0 and the cel pass will
            // be a no-op on those pixels — naturally quiet on paper-adjacent
            // colored regions.
            let orig = image.get_pixel(x, y);
            let orig_luma = 0.299 * orig[0] as f32
                + 0.587 * orig[1] as f32
                + 0.114 * orig[2] as f32;
            // base_luma and orig_luma are in 0..255 (channel sums). Normalize
            // for the skip test so the threshold is a clean 0..1 quantity.
            if orig_luma / 255.0 < 0.06 || orig_luma / 255.0 > 0.94 {
                continue;
            }

            let ratio = (orig_luma / 255.0) / (base_luma / 255.0).max(0.001);

            // Map ratio to a band factor with smoothstep edges so the
            // transition between bands is painterly, not hard.
            let band = map_to_cel_band(ratio, shadow_threshold, highlight_threshold, smoothing);

            // Construct the cel pixel: the local base hue, scaled to the
            // band brightness. Channels are scaled together so the hue is
            // preserved while the value moves to the chosen band.
            let (cr, cg, cb) = scaled_channels(
                br,
                bg,
                bb,
                base_luma,
                band,
                shadow_factor,
                highlight_factor,
            );

            // Blend with the original so the model keeps detail.
            let r = lerp(orig[0] as f32, cr, strength);
            let g = lerp(orig[1] as f32, cg, strength);
            let b = lerp(orig[2] as f32, cb, strength);
            next.put_pixel(x, y, Rgb([r.round() as u8, g.round() as u8, b.round() as u8]));
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

fn map_to_cel_band(ratio: f32, shadow_threshold: f32, highlight_threshold: f32, smoothing: f32) -> f32 {
    let low = (shadow_threshold - smoothing).max(0.30);
    let high = shadow_threshold + smoothing;
    let low_high = (highlight_threshold - smoothing).max(high + 0.01);
    let high_high = highlight_threshold + smoothing;
    if ratio <= low {
        0.0
    } else if ratio < high {
        let t = (ratio - low) / (high - low).max(0.001);
        smoothstep(0.0, 1.0, t)
    } else if ratio <= low_high {
        1.0
    } else if ratio < high_high {
        let t = (ratio - low_high) / (high_high - low_high).max(0.001);
        1.0 + smoothstep(0.0, 1.0, t)
    } else {
        2.0
    }
}

/// Scale the local base color to the band brightness. The band is a number
/// in [0, 2] where 0 = shadow, 1 = mid, 2 = highlight. We preserve the
/// chrome (each channel scaled by the same factor) so the hue stays the
/// model's hue while the brightness moves to the chosen band.
fn scaled_channels(
    r: f32,
    g: f32,
    b: f32,
    base_luma: f32,
    band: f32,
    shadow_factor: f32,
    highlight_factor: f32,
) -> (f32, f32, f32) {
    let factor = cel_factor(band, shadow_factor, highlight_factor);
    let target_luma = (base_luma * factor).clamp(0.0, 255.0);
    if base_luma <= 0.0001 {
        return (r, g, b);
    }
    let scale = target_luma / base_luma;
    let nr = (r * scale).clamp(0.0, 255.0);
    let ng = (g * scale).clamp(0.0, 255.0);
    let nb = (b * scale).clamp(0.0, 255.0);
    (nr, ng, nb)
}

// Compute the brightness factor for a band level. 0 = shadow_factor,
// 1 = 1.0, 2 = highlight_factor. Smoothstep interpolation between bands.
fn cel_factor(band: f32, shadow_factor: f32, highlight_factor: f32) -> f32 {
    let bands = [(0.0, shadow_factor), (1.0, 1.0_f32), (2.0, highlight_factor)];
    let clamped = band.clamp(0.0, 2.0);
    let lower = bands[clamped as usize];
    let upper_idx = (clamped as usize + 1).min(2);
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
            &mut img,
            &lines,
            &bubbles,
            None,
            0.7,
            1.15,
            0.58,
            1.18,
            0.18,
            0.7,
            4,
        );
        // All pixels are paper or ink; the cel pass should skip both.
        assert_eq!(report.pixels_shaded, 0);
    }

    #[test]
    fn cel_pass_quantizes_brightness_on_colored_region() {
        // Build a 20x20 image with a smooth gradient band (luma 100-150).
        // The cel pass should quantize the brightness into bands.
        let mut img = RgbImage::from_pixel(20, 20, Rgb([0, 0, 0]));
        for x in 0..20u32 {
            let r = 100 + (x * 3) as u8;
            img.put_pixel(x, 10, Rgb([r, 60, 80]));
        }
        let lines = GrayImage::from_pixel(20, 20, Luma([0]));
        let bubbles = GrayImage::from_pixel(20, 20, Luma([0]));
        let original_red: Vec<u8> = (0..20u32).map(|x| img.get_pixel(x, 10)[0]).collect();
        let report = apply_cel_shading(
            &mut img,
            &lines,
            &bubbles,
            None,
            0.7,
            1.15,
            0.58,
            1.18,
            0.18,
            0.7,
            4,
        );
        assert!(report.pixels_shaded > 0, "cel pass should shade the colored band");
        // After cel quantization at least some red values should change.
        let new_red: Vec<u8> = (0..20u32).map(|x| img.get_pixel(x, 10)[0]).collect();
        let changed = original_red
            .iter()
            .zip(new_red.iter())
            .filter(|(a, b)| **a != **b)
            .count();
        assert!(changed > 0, "cel pass should modify at least some pixels");
    }

    #[test]
    fn map_to_cel_band_in_shadow_returns_below_one() {
        let factor = cel_factor(0.0, 0.58, 1.18);
        assert!(factor < 1.0);
    }

    #[test]
    fn map_to_cel_band_in_highlight_returns_above_one() {
        let factor = cel_factor(2.0, 0.58, 1.18);
        assert!(factor > 1.0);
    }
}
