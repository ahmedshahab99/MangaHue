//! Per-chapter palette feedback: extract the dominant hues of a colored
//! page and return them as accent colors for the rest of the chapter.
//!
//! Each manga then carries its own color universe (creative variety) while
//! the hues themselves come from the engine's own model-driven output
//! (natural, no invented colors). Paper, ink and near-neutral pixels are
//! excluded; only clearly colored hues with meaningful coverage qualify.

use image::{imageops, RgbImage};
use mangahue_core::ColorRgb;

const BINS: usize = 16;
const MAX_SAMPLE_DIM: u32 = 512;

/// Extract up to 10 dominant accent hues from a colored page.
///
/// Sampling: downscale to <= 512px, skip ink (luma < 60), paper (luma > 235
/// with tiny chroma) and speech-bubble white. Hue is the opponent-space
/// angle (atan2 of B-L vs R-L); bins accumulate weighted mean color.
/// A bin qualifies when its mean chroma >= 14 and it covers >= 0.4% of the
/// sampled pixels; the top bins by coverage become the accents.
pub fn extract_accent_palette(image: &RgbImage) -> Vec<ColorRgb> {
    let small = if image.width() > MAX_SAMPLE_DIM || image.height() > MAX_SAMPLE_DIM {
        imageops::resize(
            image,
            MAX_SAMPLE_DIM,
            MAX_SAMPLE_DIM,
            imageops::FilterType::Lanczos3,
        )
    } else {
        image.clone()
    };
    let (w, h) = small.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut sums = vec![[0.0f64; 4]; BINS]; // r, g, b, weight
    let mut counts = [0u64; BINS];
    let mut total = 0u64;

    for pixel in small.pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma < 60.0 {
            continue; // ink
        }
        let cr = r - luma;
        let cb = b - luma;
        let chroma = cr.abs() + cb.abs();
        if luma > 235.0 && chroma < 12.0 {
            continue; // paper
        }
        if chroma < 14.0 {
            continue; // near-neutral; not a confident hue
        }
        let angle = cb.atan2(cr);
        // Exclude the warm SKIN sector (-50..-25 deg in opponent space):
        // skin hues already have dedicated palette slots, so accents should
        // diversify backgrounds/clothing instead of duplicating skin. The
        // band is narrow enough to keep orange/yellow clothing (-55..-70).
        let angle_deg = angle * 180.0 / std::f32::consts::PI;
        if cr > 0.0 && cb < 0.0 && (-50.0..=-25.0).contains(&angle_deg) {
            continue;
        }
        // [-pi, pi] -> [0, BINS)
        let bin = (((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)) * BINS as f32)
            as usize;
        let bin = bin.min(BINS - 1);
        sums[bin][0] += r as f64;
        sums[bin][1] += g as f64;
        sums[bin][2] += b as f64;
        sums[bin][3] += chroma as f64;
        counts[bin] += 1;
        total += 1;
    }
    if total == 0 {
        return Vec::new();
    }

    let min_coverage = (total as f64 * 0.004).max(50.0);
    let mut candidates: Vec<(u64, ColorRgb)> = Vec::new();
    for (bin, count) in counts.iter().enumerate() {
        if *count < min_coverage as u64 {
            continue;
        }
        let mean_chroma = sums[bin][3] / *count as f64;
        if mean_chroma < 14.0 {
            continue;
        }
        let r = (sums[bin][0] / *count as f64).round().clamp(0.0, 255.0) as u8;
        let g = (sums[bin][1] / *count as f64).round().clamp(0.0, 255.0) as u8;
        let b = (sums[bin][2] / *count as f64).round().clamp(0.0, 255.0) as u8;
        candidates.push((*count, ColorRgb::new(r, g, b)));
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    candidates.truncate(10);
    candidates.into_iter().map(|(_, color)| color).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn extracts_dominant_hues_only() {
        // Blue top half, red bottom half, white margin.
        let mut image = RgbImage::new(64, 64);
        for y in 0..32 {
            for x in 0..64 {
                image.put_pixel(x, y, Rgb([40, 80, 200]));
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                image.put_pixel(x, y, Rgb([200, 60, 50]));
            }
        }
        let palette = extract_accent_palette(&image);
        assert!(!palette.is_empty(), "expected accent hues");
        assert!(palette.len() <= 10);
        // The two dominant hues must be present (blue and red families).
        let has_blue = palette.iter().any(|c| c.b as i16 - c.r as i16 > 40);
        let has_red = palette.iter().any(|c| c.r as i16 - c.b as i16 > 60);
        assert!(has_blue, "missing blue accent: {palette:?}");
        assert!(has_red, "missing red accent: {palette:?}");
    }

    #[test]
    fn skin_hues_are_excluded_from_accents() {
        // Skin tone page with a blue accent: only the blue should qualify.
        let mut image = RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                image.put_pixel(x, y, Rgb([239, 199, 174])); // skin
            }
        }
        for y in 0..32 {
            for x in 0..32 {
                image.put_pixel(x, y, Rgb([40, 80, 200])); // blue shirt
            }
        }
        let palette = extract_accent_palette(&image);
        assert!(!palette.is_empty(), "expected a blue accent");
        let has_skin = palette.iter().any(|c| {
            let luma = 0.2126 * c.r as f32 + 0.7152 * c.g as f32 + 0.0722 * c.b as f32;
            let cr = c.r as f32 - luma;
            let cb = c.b as f32 - luma;
            let angle = cb.atan2(cr) * 180.0 / std::f32::consts::PI;
            cr > 0.0 && cb < 0.0 && (-50.0..=-25.0).contains(&angle)
        });
        assert!(!has_skin, "skin hues must not become accents: {palette:?}");
    }

    #[test]
    fn blank_page_yields_nothing() {
        let image = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
        assert!(extract_accent_palette(&image).is_empty());
    }
}
