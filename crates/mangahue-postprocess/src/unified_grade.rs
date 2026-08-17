//! Unified grade stage: the single measured, bounded adaptation pass.
//!
//! The legacy pipeline tail ran four overlapping chroma-shaping passes
//! (natural grade → semantic chroma restore → muted grade → smart governor),
//! each re-shaping the same saturation signal with its own curve, so one
//! pass's "fix" became the next pass's "noise" — and the governor's chroma
//! gain re-amplified dark-zone noise (the page-dependent red dots).
//!
//! This stage replaces the whole tail with ONE stage in three moves:
//!
//! 1. measure the authored page once (sampled mean saturation,
//!    Hasler-Suestrunk colorfulness and midtone warmth),
//! 2. compute one bounded chroma gain from the governor targets (scaled by
//!    the model's own chroma signal, so the page's color budget follows the
//!    model's judgment),
//! 3. sub-pass A — apply, per pixel, in one coherent decision: the
//!    per-luma saturation pull toward the reference profile (dark/mid/
//!    bright/paper targets), the vivid knee, the dark-zone attenuation
//!    (near-ink pixels keep only a whisper of chroma — the anti-red-dot
//!    floor), the merged hue decisions (skin warm-up + yellow-band shift)
//!    and the merged value shaping (shadow deepen / highlight lift / dark
//!    pull),
//! 4. measure the SHAPED page's warmth (the hue shifts warm the page, so
//!    measuring before them would leave the golden cast; the gain is not
//!    applied yet — it desaturates and would deflate the reading),
//! 5. sub-pass B — apply the dark-zone-attenuated gain and the bounded
//!    midtone warmth compression in one final loop.
//!
//! Hint-corrected and protected pixels are ground truth and are skipped,
//! exactly as in the legacy passes.

use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use serde::{Deserialize, Serialize};

use super::natural_grade::{hsv_to_rgb, rgb_to_hsv, smoothstep, PEACH_HUE};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct UnifiedGradeReport {
    pub measured_saturation: f32,
    pub measured_colorfulness: f32,
    pub measured_warmth: f32,
    pub gain: f32,
    pub warmth_shift: f32,
    pub pixels_graded: u64,
}

/// One measured, bounded adaptation pass. See the module docs for the
/// design; every sub-behavior stays gated by its own legacy flag so
/// `unified_grade: true` never silently changes a config's intent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn unified_grade(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: Option<&GrayImage>,
    hint_mask: Option<&GrayImage>,
    config: &PostprocessSection,
    chroma_signal: Option<f32>,
) -> UnifiedGradeReport {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
        || bubble_mask.dimensions() != (width, height)
    {
        return UnifiedGradeReport::default();
    }

    // ---- Compute the page's decisions on the SHAPED page -----------------
    let governor = config.smart_governor;
    let reshape = config.muted_target_grade || config.natural_grading;
    if !governor && !reshape {
        // No grading sub-behavior is enabled: the stage is a no-op (the
        // legacy pipeline also ran nothing when all three flags were off).
        return UnifiedGradeReport::default();
    }
    let mut report = UnifiedGradeReport::default();

    let strength = config.muted_strength.clamp(0.0, 1.0);
    let hue_shift_deg = config.muted_hue_shift_deg;
    let yellow_shift_deg = config.muted_yellow_shift_deg;
    let dark_pull = config.muted_dark_pull.clamp(0.2, 1.0);
    let vivid_preserve = config.muted_vivid_preserve.clamp(0.0, 1.0);
    let is_hint_pixel = |x: u32, y: u32| -> bool {
        hint_mask.is_some_and(|mask| {
            mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 127
        })
    };
    let total = width as usize * height as usize;

    // ---- Sub-pass A: saturation + hue + value shaping (no gain) --------
    // The governor's gain is applied in sub-pass B so the warmth measurement
    // below sees the SHAPED page (legacy measured after the muted grade,
    // before its own gain — the gain desaturates and would otherwise
    // deflate the warmth reading and under-apply the cooling).
    let mut graded = 0u64;
    for y in 0..height {
        for x in 0..width {
            let pixel = *image.get_pixel(x, y);
            let (mut h, mut s, mut v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
            // DDColor mauve suppression — FIRST, before every skip: the
            // hint pass only corrects weak model chroma (sat < 0.30), so
            // the model's CONFIDENT mauve regions (sat 0.5-0.75, measured
            // 27% of a page) sit under the hint mask uncorrected, and the
            // grade's hint skip would otherwise bypass them entirely. The
            // reference profile has almost no magenta mass, so the
            // purple-magenta band is desaturated harder — a saturated
            // legit purple keeps 55% of its chroma.
            let hue_deg = h * 360.0;
            if (280.0..360.0).contains(&hue_deg) {
                // Authoritative cap, not a factor: the gain (measured on
                // the shaped page) would otherwise re-amplify the loudest
                // mauve back over the visibility threshold. Raised from 0.28
                // to 0.45 so purple hair ribbons, clothing, and backgrounds
                // retain natural variety instead of being flattened.
                s = s.min(0.45);
            }
            if s <= 0.001 {
                continue; // nothing to shape
            }
            let gray8 = original_gray.get_pixel(x, y)[0] as f32;
            // Hint-corrected pixels are palette-accurate ground truth in
            // midtones and brights: they skip the full grade. BUT the hint
            // pass only corrects weak model chroma (sat < 0.30) — the
            // model's CONFIDENT mauve under the hint mask is uncorrected
            // noise, so the suppression above must still reach the image:
            // write the suppressed color and move on. In the DARK zone the
            // hint mask also covers the model's most confident noise, so
            // dark pixels get the authoritative floor below regardless of
            // the hint mask.
            if is_hint_pixel(x, y) && gray8 >= 60.0 {
                let (r, g, b) = hsv_to_rgb(h, s, v);
                image.put_pixel(
                    x,
                    y,
                    Rgb([
                        r.round().clamp(0.0, 255.0) as u8,
                        g.round().clamp(0.0, 255.0) as u8,
                        b.round().clamp(0.0, 255.0) as u8,
                    ]),
                );
                continue;
            }
            // Protected regions (speech bubbles, text, user-locked areas)
            // survive the grade untouched: they are protected exactly
            // because nothing may re-paint them.
            let protected = protected_mask.is_some_and(|mask| {
                mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 127
            });
            if protected {
                continue;
            }
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let bubble = bubble_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if ink > 0.72 || bubble > 0.38 {
                continue;
            }
            let gray = gray8 / 255.0;
            // Soft gates (same as the natural grade): only true paper
            // (>0.997) and true ink (<0.05) are fully skipped.
            let bright = smoothstep(0.92, 0.997, gray);
            let dark = smoothstep(0.13, 0.05, gray);
            if bright > 0.97 || dark > 0.98 {
                continue;
            }

            // 1. Saturation shaping, honoring the config's intent: the
            // muted profile (per-luma zone targets + vivid knee) when the
            // muted grade is on; the natural curve (midtone boost, gentle
            // oversaturation taper) when only the natural grade is on.
            // The dark zone (<60) is the anti-dot floor and is
            // AUTHORITATIVE in both profiles: near-ink pixels keep only a
            // whisper of chroma (the target scales to zero with darkness),
            // which replaces the removed tone_chroma_cap and kills the
            // red-dot amplification at its source.
            if reshape {
                let dark_floor = if gray8 < 60.0 {
                    let t = (gray8 / 60.0).powf(1.4);
                    Some(config.muted_dark_sat * (0.25 + 0.75 * t))
                } else {
                    None
                };
                if let Some(floor) = dark_floor {
                    // A hard cap, not a pull: at muted_strength 0.4 a pull
                    // leaves 0.5+ saturation noise visible (measured). The
                    // hue survives at a muted tint.
                    s = s.min(floor);
                } else if config.muted_target_grade {
                    let (target, band_strength) = if gray8 < 120.0 {
                        (config.muted_dark_sat, 0.92)
                    } else if gray8 < 180.0 {
                        (config.muted_mid_sat, 0.8)
                    } else if gray8 < 225.0 {
                        (config.muted_bright_sat, 0.8)
                    } else {
                        (config.muted_paper_sat, 0.8)
                    };
                    // Vivid knee: genuinely strong colors keep a reduced
                    // share; everything else is pulled toward its target.
                    if s > 0.55 {
                        s *= 0.5 + 0.5 * vivid_preserve;
                    } else {
                        s = s + (target - s) * strength * band_strength;
                    }
                } else {
                    // The natural grade's saturation curve, unchanged.
                    s = if s < 0.10 {
                        s * 1.6
                    } else if s < 0.45 {
                        0.16 + (s - 0.10) * 1.9
                    } else {
                        0.825 + (s - 0.45) * 0.9
                    }
                    .min(0.9);
                    s *= (1.0 - bright * 0.30) * (1.0 - dark * 0.20);
                }
            }
            s = s.clamp(0.0, 1.0);
            // Mauve cap AFTER the shaping: the pull toward the midtone
            // target (0.35) would otherwise raise the capped band back over
            // the visibility threshold. Raised from 0.28 to 0.45 so purple
            // tones retain natural saturation variety.
            let hue_deg = h * 360.0;
            if (280.0..360.0).contains(&hue_deg) {
                s = s.min(0.45);
            }

            // 3. Hue + value: the merged shaping decisions, each gated by
            // its own legacy flag so a natural-only or muted-only config
            // keeps exactly its intended look.
            let (pr, pg, pb) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            // Skin warm-up (natural grade's peach shift): YCbCr-based
            // detection works better across diverse skin tones than RGB
            // thresholds alone.
            let is_skin = config.natural_grading && (0.40..=0.97).contains(&gray) && {
                let y = 0.299 * pr + 0.587 * pg + 0.114 * pb;
                let cb = 128.0 - 0.168736 * pr - 0.331264 * pg + 0.5 * pb;
                let cr = 128.0 + 0.5 * pr - 0.418688 * pg - 0.081312 * pb;
                (77.0..=127.0).contains(&cb)    // Cb range for skin
                        && (133.0..=173.0).contains(&cr)   // Cr range for skin
                        && y > 80.0 // not too dark
            };
            if config.natural_grading {
                if is_skin {
                    h = h * 0.85 + PEACH_HUE * 0.15;
                    // Only apply the saturation floor when the pixel
                    // already carries meaningful model chroma (s > 0.12).
                    // Near-gray pixels — averaged screentone dots, dark
                    // shading, or model noise — must not be pushed to a
                    // visible 28% floor; the gain in sub-pass B would
                    // further amplify them into colored dot artifacts.
                    if s > 0.12 {
                        s = s.clamp(0.28, 0.62);
                    }
                    v = (v + 0.02).clamp(0.0, 1.0);
                } else if s > 0.68 {
                    // Keep rich colors from going neon.
                    s = 0.68 + (s - 0.68) * 0.55;
                }
                // Value shaping: shadow deepen + highlight lift.
                v -= ((0.55 - gray) * 0.06).clamp(0.0, 0.028);
                v += ((gray - 0.78) * 0.05).clamp(0.0, 0.012);
            }
            if config.muted_target_grade {
                // Yellow-band shift: the model over-produces yellow; the
                // soft-cel reference is orange/skin. Skin (0-20 deg) is
                // untouched so faces never turn red-orange.
                let hue_deg = h * 360.0;
                if hue_shift_deg != 0.0 && (20.0..85.0).contains(&hue_deg) {
                    h = ((hue_deg - hue_shift_deg) / 360.0).rem_euclid(1.0);
                }
                if yellow_shift_deg > 0.0 && (40.0..85.0).contains(&hue_deg) {
                    h = ((hue_deg - yellow_shift_deg) / 360.0).rem_euclid(1.0);
                }
                // The muted grade's dark-mid value pull.
                if (60.0..120.0).contains(&gray8) {
                    v *= dark_pull;
                }
            }
            v = v.clamp(0.0, 1.0);

            let (mut r, g, mut b) = hsv_to_rgb(h, s, v);
            // Warm shadow tint: push shadows below 0.45 luma toward
            // red/orange color temperature for a professional look.
            if gray < 0.45 && config.natural_grading {
                let shadow_warmth = ((0.45 - gray) * 0.08).clamp(0.0, 0.025);
                r = (r + shadow_warmth * 255.0).clamp(0.0, 255.0);
                b = (b - shadow_warmth * 200.0).clamp(0.0, 255.0);
            }
            image.put_pixel(
                x,
                y,
                Rgb([
                    r.round().clamp(0.0, 255.0) as u8,
                    g.round().clamp(0.0, 255.0) as u8,
                    b.round().clamp(0.0, 255.0) as u8,
                ]),
            );
            graded += 1;
        }
    }
    report.pixels_graded = graded;

    // ---- Measure the SHAPED page (sampled, <= ~120k px) -----------------
    // Same statistics as the legacy governor (which measured after the
    // muted grade): HSV saturation, Hasler-Suestrunk colorfulness and mean
    // R-B over midtones. Measuring AFTER the shaping means the mauve
    // suppression and zone pulls are reflected in the gain — a page whose
    // saturation was mostly invented mauve is not punished, and skin keeps
    // its life.
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
        return report;
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
    report.measured_saturation = measured_sat;
    report.measured_colorfulness = measured_cf;
    report.measured_warmth = measured_warmth;

    // Bounded chroma gain toward the governor targets, scaled by the
    // model's own chroma signal (identical math to the legacy governor).
    let mut gain = 1.0f32;
    if governor {
        if measured_sat > 0.001 {
            let sat_err = config.governor_target_saturation / measured_sat;
            let cf_err = config.governor_target_colorfulness / measured_cf.max(0.01);
            gain = (0.5 * sat_err + 0.5 * cf_err)
                .clamp(config.governor_min_gain, config.governor_max_gain);
            if let Some(signal) = chroma_signal {
                gain *= (signal / 50.0).clamp(0.85, 1.15);
            }
        }
        if (gain - 1.0).abs() < 0.01 {
            gain = 1.0;
        }
    }
    report.gain = gain;

    // ---- Sub-pass B: gain + warmth compression --------------------------
    // Measure the SHAPED page's warmth (the hue shifts above warmed it —
    // measuring before them would undercount and leave the golden cast),
    // then apply the governor's gain and the bounded midtone cooling in one
    // final loop. The gain is dark-zone attenuated so it can never
    // re-amplify near-black chroma noise (legacy applied it blindly, which
    // is how dark dots came back after grading).
    let mut warmth_shift = 0.0f32;
    if governor && config.governor_target_warmth > 0.0 {
        let mut warm_sum = 0.0f64;
        let mut warm_n = 0u64;
        let step = (total / 120_000).max(1);
        for (x, y, pixel) in image.enumerate_pixels() {
            let index = y as usize * width as usize + x as usize;
            if !index.is_multiple_of(step) {
                continue;
            }
            let r = pixel[0] as f64;
            let g = pixel[1] as f64;
            let b = pixel[2] as f64;
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            if (61.0..199.0).contains(&luma) {
                warm_sum += r - b;
                warm_n += 1;
            }
        }
        let shaped_warmth = if warm_n > 0 {
            (warm_sum / warm_n as f64) as f32
        } else {
            0.0
        };
        report.measured_warmth = shaped_warmth;
        let err = (config.governor_target_warmth - shaped_warmth).clamp(-30.0, 30.0);
        // Stronger warmth shift (0.50 vs 0.35) with lower dead zone
        // (1.0 vs 2.0) pushes shadows toward warm red/orange like
        // professional manga grading.
        warmth_shift = if err.abs() >= 1.0 { err * 0.50 } else { 0.0 };
        report.warmth_shift = warmth_shift;
    }
    if governor && ((gain - 1.0).abs() >= 0.01 || warmth_shift.abs() >= 2.0) {
        for y in 0..height {
            for x in 0..width {
                if is_hint_pixel(x, y) {
                    continue;
                }
                let protected = protected_mask.is_some_and(|mask| {
                    mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 127
                });
                if protected {
                    continue;
                }
                if line_mask.get_pixel(x, y)[0] > 184 {
                    continue;
                }
                let mut r = image.get_pixel(x, y)[0] as f32;
                let mut g = image.get_pixel(x, y)[1] as f32;
                let mut b = image.get_pixel(x, y)[2] as f32;
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let chroma = (r - luma).abs() + (b - luma).abs();
                if luma > 235.0 && chroma < 12.0 {
                    continue; // paper
                }
                // Warmth compression FIRST, on the pre-gain luma: the band
                // membership is stable here, so every midtone pixel that
                // drove the measurement actually gets compressed (an HSV
                // gain before this check shifts warm luma upward and would
                // push a third of the band out of range). Magenta-band
                // pixels are skipped: the cooling shift (r-10.5, b+10.5)
                // INCREASES a mauve pixel's saturation and pushes it toward
                // pure purple — the cap in sub-pass A already muted it, and
                // the warmth would undo that.
                if warmth_shift.abs() >= 2.0 && (61.0..199.0).contains(&luma) {
                    let mx = r.max(g).max(b);
                    let mn = r.min(g).min(b);
                    let hue = if mx > mn {
                        if mx == r {
                            60.0 * (((g - b) / (mx - mn)) % 6.0)
                        } else if mx == g {
                            60.0 * ((b - r) / (mx - mn) + 2.0)
                        } else {
                            60.0 * ((r - g) / (mx - mn) + 4.0)
                        }
                    } else {
                        0.0
                    };
                    let hue_deg = ((hue % 360.0) + 360.0) % 360.0;
                    if !(280.0..360.0).contains(&hue_deg) {
                        let nr = (r + warmth_shift).clamp(0.0, 255.0);
                        let nb = (b - warmth_shift).clamp(0.0, 255.0);
                        g = ((luma - 0.2126 * nr - 0.0722 * nb) / 0.7152).clamp(0.0, 255.0);
                        r = nr;
                        b = nb;
                    }
                }
                // Gain second, in RGB around the luma (legacy math, luma
                // preserved so paper/ink never shift), attenuated in the
                // dark zone so near-black chroma noise is never amplified.
                // Magenta-band pixels are skipped: the luma-preserving gain
                // WIDENS the spread of an r>b>g pixel (max above luma, min
                // below), re-saturating the capped mauve (measured 0.28 ->
                // 0.51).
                if (gain - 1.0).abs() >= 0.01 {
                    let mx = r.max(g).max(b);
                    let mn = r.min(g).min(b);
                    let hue = if mx > mn {
                        if mx == r {
                            60.0 * (((g - b) / (mx - mn)) % 6.0)
                        } else if mx == g {
                            60.0 * ((b - r) / (mx - mn) + 2.0)
                        } else {
                            60.0 * ((r - g) / (mx - mn) + 4.0)
                        }
                    } else {
                        0.0
                    };
                    let hue_deg = ((hue % 360.0) + 360.0) % 360.0;
                    if !(280.0..360.0).contains(&hue_deg) {
                        let gray8 = original_gray.get_pixel(x, y)[0] as f32;
                        let zone_atten = if gray8 < 60.0 { gray8 / 60.0 } else { 1.0 };
                        let g_eff = 1.0 + (gain - 1.0) * zone_atten;
                        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        let nr = (luma + (r - luma) * g_eff).clamp(0.0, 255.0);
                        let nb = (luma + (b - luma) * g_eff).clamp(0.0, 255.0);
                        let ng = ((luma - 0.2126 * nr - 0.0722 * nb) / 0.7152).clamp(0.0, 255.0);
                        r = nr;
                        g = ng;
                        b = nb;
                    }
                }
                image.put_pixel(
                    x,
                    y,
                    Rgb([r.round() as u8, g.round() as u8, b.round() as u8]),
                );
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn config() -> PostprocessSection {
        PostprocessSection {
            muted_target_grade: true,
            muted_strength: 0.95,
            muted_hue_shift_deg: 16.0,
            muted_yellow_shift_deg: 18.0,
            muted_dark_pull: 0.55,
            muted_dark_sat: 0.19,
            muted_mid_sat: 0.27,
            muted_bright_sat: 0.22,
            muted_paper_sat: 0.034,
            muted_vivid_preserve: 0.9,
            natural_grading: false,
            smart_governor: true,
            governor_target_saturation: 0.20,
            governor_target_colorfulness: 36.0,
            governor_min_gain: 0.8,
            governor_max_gain: 1.25,
            governor_target_warmth: 45.0,
            ..PostprocessSection::default()
        }
    }

    #[test]
    fn unified_grade_preserves_ink_bubble_and_hint_pixels() {
        let image = RgbImage::from_pixel(16, 16, Rgb([220, 120, 40]));
        let gray = GrayImage::from_pixel(16, 16, Luma([150]));
        let lines = GrayImage::from_pixel(16, 16, Luma([0]));
        let bubbles = GrayImage::from_pixel(16, 16, Luma([0]));
        let hints = GrayImage::from_pixel(16, 16, Luma([0]));
        let before = *image.get_pixel(8, 8);

        // Ink pixel: line_mask 255 → untouched.
        let mut ink_img = image.clone();
        let mut ink_lines = lines.clone();
        for p in ink_lines.pixels_mut() {
            p[0] = 255;
        }
        unified_grade(
            &mut ink_img,
            &gray,
            &ink_lines,
            &bubbles,
            None,
            Some(&hints),
            &config(),
            None,
        );
        assert_eq!(*ink_img.get_pixel(8, 8), before);

        // Hint pixel: hint_mask > 127 → untouched.
        let mut hint_img = image.clone();
        let mut hint_mask = hints.clone();
        for p in hint_mask.pixels_mut() {
            p[0] = 200;
        }
        unified_grade(
            &mut hint_img,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hint_mask),
            &config(),
            None,
        );
        assert_eq!(*hint_img.get_pixel(8, 8), before);

        // Sanity: no mask, the grade does change the saturated orange.
        let mut plain = image.clone();
        let report = unified_grade(
            &mut plain,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hints),
            &config(),
            None,
        );
        assert_ne!(*plain.get_pixel(8, 8), before);
        assert!(report.pixels_graded > 0);
        assert!(report.measured_saturation > 0.5);
        assert!(report.gain >= 0.8 && report.gain <= 1.25);
    }

    #[test]
    fn dark_zone_chroma_is_capped_not_amplified() {
        // A dark pixel with the model's invented chroma must be pulled
        // toward neutral — the anti-red-dot floor — even when the governor
        // wants to amplify (max gain).
        let mut image = RgbImage::from_pixel(8, 8, Rgb([30, 8, 8]));
        let gray = GrayImage::from_pixel(8, 8, Luma([20]));
        let lines = GrayImage::from_pixel(8, 8, Luma([0]));
        let bubbles = GrayImage::from_pixel(8, 8, Luma([0]));
        let hints = GrayImage::from_pixel(8, 8, Luma([0]));
        let before = *image.get_pixel(4, 4);
        let mut cfg = config();
        cfg.governor_max_gain = 1.3;
        unified_grade(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hints),
            &cfg,
            Some(60.0),
        );
        let after = *image.get_pixel(4, 4);
        // Chroma (max-min) must not grow past ~1.3x the capped target, and
        // the pixel must stay dark (never a saturated red dot).
        let chroma = after.0.iter().max().unwrap() - after.0.iter().min().unwrap();
        assert!(chroma <= 28, "dark chroma not capped: {chroma} ({after:?})");
        assert!(
            after.0.iter().all(|c| *c < 45),
            "dark pixel brightened: {after:?}"
        );
        assert_ne!(after, before, "grade was inert");
    }

    #[test]
    fn dark_zone_hint_pixels_are_capped_not_exempt() {
        // The hint mask used to exempt dark pixels from the grade, but it
        // also covers the model's most confident purple/red noise (the
        // dots). A dark hint pixel must get the authoritative floor.
        let mut image = RgbImage::from_pixel(8, 8, Rgb([60, 25, 40])); // purple-ish, sat ~0.6
        let gray = GrayImage::from_pixel(8, 8, Luma([35]));
        let lines = GrayImage::from_pixel(8, 8, Luma([0]));
        let bubbles = GrayImage::from_pixel(8, 8, Luma([0]));
        let hints = GrayImage::from_pixel(8, 8, Luma([200]));
        unified_grade(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hints),
            &config(),
            None,
        );
        let p = image.get_pixel(4, 4);
        let (_, s, _) = rgb_to_hsv(p[0], p[1], p[2]);
        assert!(s < 0.28, "dark hint pixel must be floored, got s={s:.2}");
    }

    #[test]
    fn midtone_saturation_is_pulled_toward_target() {
        // A vivid midtone orange (the legacy muted test scenario) must be
        // desaturated toward the mid target but keep its hue and value.
        let mut image = RgbImage::from_pixel(16, 16, Rgb([220, 120, 40]));
        let gray = GrayImage::from_pixel(16, 16, Luma([150]));
        let lines = GrayImage::from_pixel(16, 16, Luma([0]));
        let bubbles = GrayImage::from_pixel(16, 16, Luma([0]));
        let hints = GrayImage::from_pixel(16, 16, Luma([0]));
        let mut cfg = config();
        cfg.smart_governor = false; // isolate the reshape behavior
        cfg.muted_vivid_preserve = 0.0; // no vivid knee: pure zone pull
        unified_grade(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hints),
            &cfg,
            None,
        );
        let p = image.get_pixel(8, 8);
        let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
        assert!(s < 0.55, "expected desaturation, got s={s:.2}");
        assert!(s > 0.20, "expected not fully gray, got s={s:.2}");
        assert!(!(0.12..=0.9).contains(&h), "hue drifted: h={h:.2}");
        assert!(v > 0.6, "value not preserved: v={v:.2}");
    }
}

#[cfg(test)]
mod suppression_debug {
    use super::*;
    use image::Luma;

    #[test]
    fn magenta_pixel_is_suppressed() {
        let mut image = RgbImage::from_pixel(8, 8, Rgb([158, 40, 85])); // model mauve, sat 0.75
        let gray = GrayImage::from_pixel(8, 8, Luma([87]));
        let lines = GrayImage::from_pixel(8, 8, Luma([0]));
        let bubbles = GrayImage::from_pixel(8, 8, Luma([0]));
        let hints = GrayImage::from_pixel(8, 8, Luma([0]));
        let cfg = PostprocessSection {
            muted_target_grade: true,
            muted_strength: 0.4,
            smart_governor: false,
            ..PostprocessSection::default()
        };
        unified_grade(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            None,
            Some(&hints),
            &cfg,
            None,
        );
        let p = image.get_pixel(4, 4);
        let (_, s, _) = rgb_to_hsv(p[0], p[1], p[2]);
        eprintln!("magenta pixel after grade: {p:?} s={s:.3}");
        assert!(s < 0.45, "magenta must be suppressed, got s={s:.3}");
    }
}
