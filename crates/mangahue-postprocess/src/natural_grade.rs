use image::{GrayImage, Rgb, RgbImage};

/// Natural color grading pass.
///
/// Shifts the flat model output toward natural painted colors:
/// - warm peach skin grading on detected skin regions
/// - desaturation of oversaturated (plastic/neon) colors
/// - deeper, warmer shadows in dark midtones
/// - gentle warmth in highlights
/// Ink and paper pixels are preserved untouched.
pub(crate) fn natural_grade(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
        || bubble_mask.dimensions() != (width, height)
    {
        return;
    }

    for y in 0..height {
        for x in 0..width {
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let bubble = bubble_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if ink > 0.72 || bubble > 0.38 {
                continue;
            }

            let gray = original_gray.get_pixel(x, y)[0] as f32 / 255.0;
            // Soft gates: only TRUE paper (>0.997) and TRUE ink (<0.05) are
            // skipped. Bright skin (0.92-0.997) gets a reduced boost + warm
            // shift so faces/chests/backgrounds never stay flat white.
            let bright = smoothstep(0.92, 0.997, gray);
            let dark = smoothstep(0.13, 0.05, gray);
            if bright > 0.97 || dark > 0.98 {
                continue;
            }

            let pixel = *image.get_pixel(x, y);
            let (mut h, mut s, mut v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);

            // Deep-color saturation curve: midtones boosted hardest,
            // near-grays gentle, vivid soft-tapered (never neon).
            s = if s < 0.10 {
                s * 1.6
            } else if s < 0.45 {
                0.16 + (s - 0.10) * 1.9
            } else {
                0.825 + (s - 0.45) * 0.9
            }
            .min(0.9);
            // Bright pixels get a reduced but real boost (paper stays paper).
            s *= (1.0 - bright * 0.30) * (1.0 - dark * 0.20);

            // Value shaping: deep shadows, lifted highlights — the only
            // luminance depth pass in this module.
            v -= ((0.55 - gray) * 0.06).clamp(0.0, 0.028);
            v += ((gray - 0.78) * 0.05).clamp(0.0, 0.012);
            v = v.clamp(0.0, 1.0);

            // Skin warm-up extends to bright skin (0.50-0.97) so light faces
            // get the peach shift; deep skin keeps warm shadows.
            let is_skin = (0.40..=0.97).contains(&gray) && skin_hue_range(h);
            if is_skin {
                h = h * 0.85 + PEACH_HUE * 0.15;
                s = s.clamp(0.32, 0.62);
                v = (v + 0.02).clamp(0.0, 1.0);
            } else if s > 0.68 {
                // Keep rich colors from going neon (raised from 0.6).
                s = 0.68 + (s - 0.68) * 0.55;
            }

            let (r, g, b) = hsv_to_rgb(h, s, v);
            image.put_pixel(
                x,
                y,
                Rgb([
                    (r.clamp(0.0, 255.0)) as u8,
                    (g.clamp(0.0, 255.0)) as u8,
                    (b.clamp(0.0, 255.0)) as u8,
                ]),
            );
        }
    }
}

const PEACH_HUE: f32 = 0.055;

/// Muted target grade: reshapes each pixel's saturation toward the measured
/// per-luma saturation profile of the reference style (low-key, muted warm:
/// deep shadows ~0.21, midtones ~0.28, brights ~0.22, paper ~0.035). Hue
/// and value are preserved; only saturation is pulled toward the target so
/// the whole page reads as one coherent muted grade instead of neon colors.
pub(crate) fn muted_target_grade(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    strength: f32,
    hue_shift_deg: f32,
    dark_pull: f32,
    dark_sat: f32,
    mid_sat: f32,
    bright_sat: f32,
    paper_sat: f32,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
        || bubble_mask.dimensions() != (width, height)
    {
        return;
    }
    let strength = strength.clamp(0.0, 1.0);
    for y in 0..height {
        for x in 0..width {
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let bubble = bubble_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if ink > 0.72 || bubble > 0.38 {
                continue;
            }
            let gray = original_gray.get_pixel(x, y)[0] as f32;
            // Per-luma saturation targets measured from the reference style.
            let (target, band_strength) = if gray < 60.0 {
                continue;
            } else if gray < 120.0 {
                (dark_sat, 0.92)
            } else if gray < 180.0 {
                (mid_sat, 0.8)
            } else if gray < 225.0 {
                (bright_sat, 0.8)
            } else {
                (paper_sat, 0.8)
            };
            let pixel = *image.get_pixel(x, y);
            let (mut h, mut s, mut v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
            if s <= 0.001 {
                continue;
            }
            // Soft knee: genuinely strong colors (the reference has 22%
            // vivid punch) keep a reduced share; washed midtones are
            // pulled down to the muted band target.
            if s > 0.55 {
                s *= 0.5;
            } else {
                s = s + (target - s) * strength * band_strength;
            }
            // The reference leans deep RED-ORANGE (68% of hue mass in the
            // 0-30 deg band): shift yellows gently toward red-orange.
            let hue_deg = h * 360.0;
            if (20.0..75.0).contains(&hue_deg) {
                h = ((hue_deg - hue_shift_deg) / 360.0).rem_euclid(1.0);
            }
            // The reference's dark-mid range is deeper than ours: pull
            // dark-mid values down for the low-key look.
            if (60.0..120.0).contains(&gray) {
                v *= dark_pull.clamp(0.7, 1.0);
            }
            let (r, g, b) = hsv_to_rgb(h, s.clamp(0.0, 1.0), v);
            image.put_pixel(
                x,
                y,
                Rgb([
                    (r.clamp(0.0, 255.0)) as u8,
                    (g.clamp(0.0, 255.0)) as u8,
                    (b.clamp(0.0, 255.0)) as u8,
                ]),
            );
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn skin_hue_range(h: f32) -> bool {
    // Warm skin hues: 0.0-0.12 (orange/peach) and 0.90-1.0 (pink/red).
    (0.90..=1.0).contains(&h) || (0.0..=0.12).contains(&h)
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        // Rust % keeps the dividend sign; normalize to [0, 360).
        let sector = ((g - b) / delta) % 6.0;
        60.0 * if sector < 0.0 { sector + 6.0 } else { sector }
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h / 360.0, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = ((h % 1.0) + 1.0) % 1.0;
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let (r, g, b) = match i as u32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    (r * 255.0, g * 255.0, b * 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn hsv_round_trip_preserves_pixel() {
        for (r, g, b) in [
            (120u8, 80u8, 60u8),
            (200, 180, 170),
            (30, 30, 30),
            (255, 255, 255),
        ] {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let (nr, ng, nb) = hsv_to_rgb(h, s, v);
            assert!((nr - r as f32).abs() < 2.0);
            assert!((ng - g as f32).abs() < 2.0);
            assert!((nb - b as f32).abs() < 2.0);
        }
    }

    #[test]
    fn grading_preserves_ink_and_paper() {
        let mut image = RgbImage::from_pixel(32, 32, Rgb([120, 90, 100]));
        let gray = GrayImage::from_pixel(32, 32, Luma([10]));
        let lines = GrayImage::from_pixel(32, 32, Luma([255]));
        let bubbles = GrayImage::from_pixel(32, 32, Luma([0]));
        let before = *image.get_pixel(0, 0);
        natural_grade(&mut image, &gray, &lines, &bubbles);
        assert_eq!(*image.get_pixel(0, 0), before);
    }

    #[test]
    fn muted_grade_pulls_saturation_toward_target_and_preserves_ink() {
        // A saturated midtone orange that should be desaturated toward the
        // muted mid target (0.27) but keep its hue and value.
        let mut image = RgbImage::from_pixel(16, 16, Rgb([220, 120, 40]));
        let gray = GrayImage::from_pixel(16, 16, Luma([150]));
        let lines = GrayImage::from_pixel(16, 16, Luma([0]));
        let bubbles = GrayImage::from_pixel(16, 16, Luma([0]));
        muted_target_grade(
            &mut image, &gray, &lines, &bubbles, 1.0, 0.0, 1.0, 0.19, 0.27, 0.22, 0.034,
        );
        let p = image.get_pixel(8, 8);
        let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
        // Saturation pulled down from ~0.82 toward 0.27.
        assert!(s < 0.55, "expected desaturation, got s={s:.2}");
        assert!(s > 0.20, "expected not fully gray, got s={s:.2}");
        // Hue stays orange (0.02-0.1 range).
        assert!(h < 0.12 || h > 0.9);
        // Value preserved.
        assert!(v > 0.6);
    }

    #[test]
    fn muted_grade_preserves_ink_pixels() {
        let mut image = RgbImage::from_pixel(8, 8, Rgb([255, 80, 40]));
        let gray = GrayImage::from_pixel(8, 8, Luma([150]));
        let lines = GrayImage::from_pixel(8, 8, Luma([255]));
        let bubbles = GrayImage::from_pixel(8, 8, Luma([0]));
        let before = *image.get_pixel(4, 4);
        muted_target_grade(
            &mut image, &gray, &lines, &bubbles, 1.0, 20.0, 0.85, 0.19, 0.27, 0.22, 0.034,
        );
        assert_eq!(*image.get_pixel(4, 4), before);
    }
}
