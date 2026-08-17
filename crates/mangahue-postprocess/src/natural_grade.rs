use image::{GrayImage, Rgb, RgbImage};

/// Natural color grading pass.
///
/// Shifts the flat model output toward natural painted colors:
/// - warm peach skin grading on detected skin regions
/// - desaturation of oversaturated (plastic/neon) colors
/// - deeper, warmer shadows in dark midtones
/// - gentle warmth in highlights
///
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
            // get the peach shift; deep skin keeps warm shadows. The gate
            // requires red dominance plus a blue floor so orange/gold hair
            // and strongly blue-deficient materials are NOT peached.
            let (pr, pg, pb) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            // YCbCr skin detection works better across diverse skin tones
            // than RGB thresholds alone.
            let is_skin = (0.40..=0.97).contains(&gray) && {
                let y = 0.299 * pr + 0.587 * pg + 0.114 * pb;
                let cb = 128.0 - 0.168736 * pr - 0.331264 * pg + 0.5 * pb;
                let cr = 128.0 + 0.5 * pr - 0.418688 * pg - 0.081312 * pb;
                (77.0..=127.0).contains(&cb)    // Cb range for skin
                        && (133.0..=173.0).contains(&cr)   // Cr range for skin
                        && y > 80.0 // not too dark
            };
            if is_skin {
                h = h * 0.85 + PEACH_HUE * 0.15;
                // Wider band (0.28-0.62) allows more natural skin variety
                // across different lighting and skin tones.
                s = s.clamp(0.28, 0.62);
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

pub(crate) const PEACH_HUE: f32 = 0.055;

/// Smart color governor: a per-page, measured adaptation pass that runs
/// LAST of the color passes. Instead of a fixed grade, it measures the
/// page's own average HSV saturation and Hasler-Suestrunk colorfulness,
/// then applies a bounded chroma gain that moves those metrics toward the
/// preset targets. The model's own chroma signal (mean opponent magnitude
/// right after inference) scales the target so the color budget follows
/// what the model judged this page to be — a muted page stays muted, a
/// colorful page keeps its life. Hue and luma are preserved; ink and paper
/// are untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn smart_color_governor(
    image: &mut RgbImage,
    line_mask: &GrayImage,
    target_saturation: f32,
    target_colorfulness: f32,
    min_gain: f32,
    max_gain: f32,
    target_warmth: f32,
    chroma_signal: Option<f32>,
    hint_mask: Option<&GrayImage>,
) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || line_mask.dimensions() != (width, height) {
        return;
    }
    // Hint-corrected pixels are ground truth (palette-accurate material
    // hues): the governor's chroma gain and warmth compression are tuned
    // for the model's golden cast and would crush them (measured: face
    // saturation 0.136 -> 0.076 through grading).
    let is_hint_pixel = |x: u32, y: u32| -> bool {
        hint_mask.is_some_and(|mask| {
            mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 127
        })
    };
    // Measure (sampled, <= ~120k px): saturation, colorfulness and warmth
    // (mean R-B over midtones, 0-255 units).
    let total = width as usize * height as usize;
    let step = (total / 120_000).max(1);
    let mut sat_sum = 0.0f64;
    let mut rg_sum = 0.0f64;
    let mut rg_sq = 0.0f64;
    let mut yb_sum = 0.0f64;
    let mut yb_sq = 0.0f64;
    let mut warm_sum = 0.0f64;
    let mut warm_n = 0u64;
    let mut n = 0u64;
    for (x, y, pixel) in image.enumerate_pixels() {
        let index = y as usize * width as usize + x as usize;
        if !index.is_multiple_of(step) {
            continue;
        }
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        sat_sum += if max > 0.0 { (max - min) / max } else { 0.0 };
        let rg = r - g;
        let yb = 0.5 * (r + g) - b;
        rg_sum += rg;
        rg_sq += rg * rg;
        yb_sum += yb;
        yb_sq += yb * yb;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        if (61.0..199.0).contains(&luma) {
            warm_sum += r - b;
            warm_n += 1;
        }
        n += 1;
    }
    if n == 0 {
        return;
    }
    let nf = n as f64;
    let measured_sat = (sat_sum / nf) as f32;
    let rg_mean = rg_sum / nf;
    let rg_var = (rg_sq / nf - rg_mean * rg_mean).max(0.0);
    let yb_mean = yb_sum / nf;
    let yb_var = (yb_sq / nf - yb_mean * yb_mean).max(0.0);
    let measured_cf = ((rg_var.sqrt() + yb_var.sqrt())
        + 0.3 * (rg_mean * rg_mean + yb_mean * yb_mean).sqrt()) as f32;
    let measured_warmth = if warm_n > 0 {
        (warm_sum / warm_n as f64) as f32
    } else {
        0.0
    };
    if measured_sat <= 0.001 {
        return;
    }
    let sat_err = target_saturation / measured_sat;
    let cf_err = target_colorfulness / measured_cf.max(0.01);
    let mut gain = (0.5 * sat_err + 0.5 * cf_err).clamp(min_gain, max_gain);
    // Model budget: a page the model judged muted stays muted; a colorful
    // page keeps its life (bounded so the governor never fights the model).
    if let Some(signal) = chroma_signal {
        gain *= (signal / 50.0).clamp(0.85, 1.15);
    }
    if (gain - 1.0).abs() < 0.01 {
        return;
    }
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if is_hint_pixel(x, y) {
            continue;
        }
        if line_mask.get_pixel(x, y)[0] > 184 {
            continue;
        }
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let chroma = (r - luma).abs() + (b - luma).abs();
        if luma > 235.0 && chroma < 12.0 {
            continue; // paper
        }
        let new_r = (luma + (r - luma) * gain).clamp(0.0, 255.0);
        let new_b = (luma + (b - luma) * gain).clamp(0.0, 255.0);
        let new_g = ((luma - 0.2126 * new_r - 0.0722 * new_b) / 0.7152).clamp(0.0, 255.0);
        *pixel = Rgb([
            new_r.round() as u8,
            new_g.round() as u8,
            new_b.round() as u8,
        ]);
    }

    // Warmth compression: the model over-produces warm/yellow casts on
    // manga pages, and the muted grade's target_warmth (mean R-B over
    // midtones) is what presets tune to kill it. The measured warmth is
    // pushed toward the target on midtones only (the same 61..199 luma band
    // the measurement uses), preserving luma and paper/ink. Bounded to +-10
    // Lab-ish units per pass so a wrong target can never invert a page.
    if target_warmth > 0.0 {
        let warmth_err = (target_warmth - measured_warmth).clamp(-30.0, 30.0);
        if warmth_err.abs() >= 1.0 {
            // Stronger warmth shift (0.50 vs 0.35) with lower dead zone
            // (1.0 vs 2.0) pushes shadows toward warm red/orange like
            // professional manga grading.
            let shift = warmth_err * 0.50;
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                if is_hint_pixel(x, y) {
                    continue;
                }
                if line_mask.get_pixel(x, y)[0] > 184 {
                    continue;
                }
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let chroma = (r - luma).abs() + (b - luma).abs();
                if luma > 235.0 && chroma < 12.0 {
                    continue; // paper
                }
                if !(61.0..199.0).contains(&luma) {
                    continue; // midtones only, same band as the measurement
                }
                let new_r = (r + shift).clamp(0.0, 255.0);
                let new_b = (b - shift).clamp(0.0, 255.0);
                let new_g = ((luma - 0.2126 * new_r - 0.0722 * new_b) / 0.7152).clamp(0.0, 255.0);
                *pixel = Rgb([
                    new_r.round() as u8,
                    new_g.round() as u8,
                    new_b.round() as u8,
                ]);
            }
        }
    }
}

/// Muted target grade: reshapes each pixel's saturation toward the measured
/// per-luma saturation profile of the reference style (low-key, muted warm:
/// deep shadows ~0.21, midtones ~0.28, brights ~0.22, paper ~0.035). Hue
/// and value are preserved; only saturation is pulled toward the target so
/// the whole page reads as one coherent muted grade instead of neon colors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn muted_target_grade(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    hint_mask: Option<&GrayImage>,
    strength: f32,
    hue_shift_deg: f32,
    yellow_shift_deg: f32,
    dark_pull: f32,
    dark_sat: f32,
    mid_sat: f32,
    bright_sat: f32,
    paper_sat: f32,
    vivid_preserve: f32,
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
            // Hint-corrected pixels carry palette-accurate ground truth:
            // the grade was tuned to reshape the MODEL's colors and would
            // otherwise crush/re-hue exactly the blind spots the hints
            // just fixed (measured: face saturation 0.136 -> 0.076).
            if let Some(mask) = hint_mask {
                if mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 127 {
                    continue;
                }
            }
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
            // pulled down to the muted band target. muted_vivid_preserve
            // scales how much of the vivid share survives (0.0 = halve,
            // 1.0 = keep) so presets can aim for "moderate" saturation.
            if s > 0.55 {
                s *= 0.5 + 0.5 * vivid_preserve.clamp(0.0, 1.0);
            } else {
                s = s + (target - s) * strength * band_strength;
            }
            // The reference leans deep RED-ORANGE (68% of hue mass in the
            // 0-30 deg band): shift yellows gently toward red-orange.
            let hue_deg = h * 360.0;
            if (20.0..85.0).contains(&hue_deg) {
                h = ((hue_deg - hue_shift_deg) / 360.0).rem_euclid(1.0);
            }
            // Yellow-band-only shift: the model over-produces yellow (40-85
            // deg) on manga pages; the soft-cel reference is orange/skin
            // with almost no yellow. Skin (0-20 deg) is untouched, so faces
            // never turn red-orange.
            if yellow_shift_deg > 0.0 && (40.0..85.0).contains(&hue_deg) {
                h = ((hue_deg - yellow_shift_deg) / 360.0).rem_euclid(1.0);
            }
            // The reference's dark-mid range is deeper than ours: pull
            // dark-mid values down for the low-key look. The clamp is a
            // sanity bound only — presets like muted_dark_pull 0.5 must be
            // honored (the old 0.7 floor silently ignored every preset).
            if (60.0..120.0).contains(&gray) {
                v *= dark_pull.clamp(0.2, 1.0);
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

pub(crate) fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    // NOTE: never floor the denominator with .max(1e-6) — that destroys
    // the sign of (edge1 - edge0) and turns every inverted ramp (edge1 <
    // edge0, e.g. the dark gate) into a constant 1.0, skipping all pixels.
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return f32::from(value >= edge1);
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn skin_hue_range(h: f32) -> bool {
    // Warm skin hues: 0.0-0.12 (orange/peach) and 0.90-1.0 (pink/red).
    (0.90..=1.0).contains(&h) || (0.0..=0.12).contains(&h)
}

pub(crate) fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
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

pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
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
            &mut image, &gray, &lines, &bubbles, None, 1.0, 0.0, 0.0, 1.0, 0.19, 0.27, 0.22, 0.034,
            0.0,
        );
        let p = image.get_pixel(8, 8);
        let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
        // Saturation pulled down from ~0.82 toward 0.27.
        assert!(s < 0.55, "expected desaturation, got s={s:.2}");
        assert!(s > 0.20, "expected not fully gray, got s={s:.2}");
        // Hue stays orange (0.02-0.1 range).
        assert!(!(0.12..=0.9).contains(&h));
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
            &mut image, &gray, &lines, &bubbles, None, 1.0, 20.0, 15.0, 0.85, 0.19, 0.27, 0.22,
            0.034, 0.0,
        );
        assert_eq!(*image.get_pixel(4, 4), before);
    }

    #[test]
    fn muted_grade_leaves_hint_corrected_pixels_untouched() {
        // A saturated skin pixel carrying a strong hint must survive the
        // grade intact (hints are ground truth; grading is for the model's
        // own colors). Without the hint mask the grade would pull it toward
        // the muted mid target.
        let mut image = RgbImage::from_pixel(8, 8, Rgb([235, 165, 130]));
        let gray = GrayImage::from_pixel(8, 8, Luma([150]));
        let lines = GrayImage::from_pixel(8, 8, Luma([0]));
        let bubbles = GrayImage::from_pixel(8, 8, Luma([0]));
        let hints = GrayImage::from_pixel(8, 8, Luma([200]));
        let before = *image.get_pixel(4, 4);
        muted_target_grade(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            Some(&hints),
            1.0,
            16.0,
            18.0,
            0.55,
            0.24,
            0.35,
            0.33,
            0.02,
            0.9,
        );
        assert_eq!(
            *image.get_pixel(4, 4),
            before,
            "hint-corrected pixel regraded"
        );
        // Sanity: the same grade WITHOUT the hint mask does change the pixel.
        let mut plain = RgbImage::from_pixel(8, 8, Rgb([235, 165, 130]));
        muted_target_grade(
            &mut plain, &gray, &lines, &bubbles, None, 1.0, 16.0, 18.0, 0.55, 0.24, 0.35, 0.33,
            0.02, 0.9,
        );
        assert_ne!(
            *plain.get_pixel(4, 4),
            before,
            "grade without hint mask is inert"
        );
    }
}
