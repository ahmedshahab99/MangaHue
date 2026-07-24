use anyhow::Result;
use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use mangahue_core::StyleProfile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct PostprocessInput<'a> {
    pub colorized: RgbImage,
    pub original_grayscale: &'a GrayImage,
    pub line_mask: &'a GrayImage,
    pub bubble_mask: &'a GrayImage,
    pub tone_mask: Option<&'a GrayImage>,
    pub protected_mask: Option<&'a GrayImage>,
    pub style: &'a StyleProfile,
    pub config: &'a PostprocessSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessReport {
    pub bubble_pixels_restored: u64,
    pub protected_pixels_restored: u64,
    pub bleed_pixels_corrected: u64,
    pub tone_pixels_preserved: u64,
    pub sharpen_radius: f32,
    pub sharpen_amount: f32,
}

pub fn finalize(input: PostprocessInput<'_>) -> Result<(RgbImage, PostprocessReport)> {
    let mut output = input.colorized;
    let mut report = PostprocessReport {
        bubble_pixels_restored: 0,
        protected_pixels_restored: 0,
        bleed_pixels_corrected: 0,
        tone_pixels_preserved: 0,
        sharpen_radius: 1.0,
        sharpen_amount: input.config.final_sharpening,
    };

    if input.config.bleeding_correction {
        reduce_color_near_ink(
            &mut output,
            input.line_mask,
            &mut report.bleed_pixels_corrected,
        );
    }

    if input.config.palette_correction {
        let lut = MangaColorLUT::new(input.style);
        apply_lut_grading(&mut output, &lut, input.line_mask, input.original_grayscale);
    }

    if let Some(tone_mask) = input.tone_mask {
        preserve_screentones(
            &mut output,
            input.original_grayscale,
            tone_mask,
            input.config.line_strength,
            &mut report.tone_pixels_preserved,
        );
    }

    edge_aware_smooth(&mut output, input.line_mask);

    if input.config.bubble_cleanup {
        clean_speech_bubbles(
            &mut output,
            input.original_grayscale,
            input.bubble_mask,
            &mut report.bubble_pixels_restored,
        );
        if let Some(protected_mask) = input.protected_mask {
            clean_speech_bubbles(
                &mut output,
                input.original_grayscale,
                protected_mask,
                &mut report.protected_pixels_restored,
            );
        }
    }

    if input.config.restore_lines {
        restore_lines_adaptive(
            &mut output,
            input.line_mask,
            input.config.line_strength * input.style.line_strength,
        );
    }

    if input.config.final_sharpening > 0.0 {
        sharpen_lightly(&mut output, input.config.final_sharpening);
    }
    Ok((output, report))
}

pub fn restore_lines(image: &mut RgbImage, line_mask: &GrayImage, strength: f32) {
    let strength = strength.clamp(0.0, 2.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink <= 0.0 {
            continue;
        }
        let keep = (1.0 - ink * strength).clamp(0.0, 1.0);
        pixel[0] = (pixel[0] as f32 * keep).round() as u8;
        pixel[1] = (pixel[1] as f32 * keep).round() as u8;
        pixel[2] = (pixel[2] as f32 * keep).round() as u8;
    }
}

pub fn clean_speech_bubbles(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    bubble_mask: &GrayImage,
    bubble_pixels_restored: &mut u64,
) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if bubble_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let original_luma = original_gray.get_pixel(x, y)[0] as f32;
        let t = (original_luma / 255.0).powf(1.4).clamp(0.0, 1.0);
        for c in 0..3 {
            pixel[c] = blend(pixel[c], 255, t);
        }
        *bubble_pixels_restored += 1;
    }
}

pub fn reduce_color_near_ink(
    image: &mut RgbImage,
    line_mask: &GrayImage,
    bleed_pixels_corrected: &mut u64,
) {
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 {
        return;
    }
    let w_i32 = w as i32;
    let h_i32 = h as i32;

    let source: Vec<Rgb<u8>> = image.pixels().copied().collect();

    let radius: i32 = 3;
    let spatial_sigma: f64 = 1.5;
    let range_sigma: f64 = 50.0;

    for y in 0..h_i32 {
        for x in 0..w_i32 {
            let idx = (y * w_i32 + x) as usize;
            if line_mask.get_pixel(x as u32, y as u32)[0] > 0 {
                continue;
            }

            let mut near_ink = false;
            'outer: for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w_i32 || ny >= h_i32 {
                        continue;
                    }
                    if line_mask.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        near_ink = true;
                        break 'outer;
                    }
                }
            }
            if !near_ink {
                continue;
            }

            let center = source[idx];
            let mut sum = [0.0f64; 3];
            let mut total_weight: f64 = 0.0;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w_i32 || ny >= h_i32 {
                        continue;
                    }
                    let nidx = (ny * w_i32 + nx) as usize;
                    if line_mask.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        continue;
                    }
                    let neighbor = source[nidx];
                    let spatial_dist = ((dx * dx + dy * dy) as f64).sqrt();
                    let spatial_weight = (-0.5 * (spatial_dist / spatial_sigma).powi(2)).exp();

                    let mut color_dist_sq: f64 = 0.0;
                    for c in 0..3 {
                        let diff = center[c] as f64 - neighbor[c] as f64;
                        color_dist_sq += diff * diff;
                    }
                    let color_dist = color_dist_sq.sqrt();
                    let range_weight = (-0.5 * (color_dist / range_sigma).powi(2)).exp();

                    let weight = spatial_weight * range_weight;
                    for c in 0..3 {
                        sum[c] += neighbor[c] as f64 * weight;
                    }
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                image.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([
                        (sum[0] / total_weight).round().clamp(0.0, 255.0) as u8,
                        (sum[1] / total_weight).round().clamp(0.0, 255.0) as u8,
                        (sum[2] / total_weight).round().clamp(0.0, 255.0) as u8,
                    ]),
                );
                *bleed_pixels_corrected += 1;
            }
        }
    }
}

pub fn apply_palette_curve(image: &mut RgbImage, style: &StyleProfile) {
    for pixel in image.pixels_mut() {
        let gray = (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32)
            .round()
            .clamp(0.0, 255.0);
        let mut channels = [pixel[0], pixel[1], pixel[2]];
        for channel in &mut channels {
            let saturated = gray + (*channel as f32 - gray) * style.saturation;
            let contrasted = (saturated - 128.0) * style.contrast + 128.0;
            let brightened = contrasted + style.brightness * 255.0;
            *channel = brightened.round().clamp(0.0, 255.0) as u8;
        }
        *pixel = Rgb(channels);
    }
}

// ── 3D LUT Manga Color Grading ──────────────────────────────────────────
const LUT_SIZE: usize = 17;

struct MangaColorLUT {
    table: Vec<[[Rgb<u8>; LUT_SIZE]; LUT_SIZE]>,
}

impl MangaColorLUT {
    fn new(style: &StyleProfile) -> Self {
        let contrast = style.contrast.clamp(0.9, 1.15);
        let sat = style.saturation.clamp(0.7, 1.0);
        let bright = style.brightness * 15.0;
        let mut table = vec![[[Rgb([0; 3]); LUT_SIZE]; LUT_SIZE]; LUT_SIZE];

        for (ri, red_slice) in table.iter_mut().enumerate() {
            for (gi, green_slice) in red_slice.iter_mut().enumerate() {
                for (bi, entry) in green_slice.iter_mut().enumerate() {
                    let r = ri as f32 * 16.0;
                    let g = gi as f32 * 16.0;
                    let b = bi as f32 * 16.0;

                    // Subtle desaturate toward grayscale (natural manga look)
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    let sr = gray + (r - gray) * sat;
                    let sg = gray + (g - gray) * sat;
                    let sb = gray + (b - gray) * sat;

                    // Gentle contrast around 128
                    let cr = ((sr - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;
                    let cg = ((sg - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;
                    let cb = ((sb - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;

                    *entry = Rgb([cr, cg, cb]);
                }
            }
        }

        MangaColorLUT { table }
    }
}

fn lut_lookup(lut: &MangaColorLUT, r: u8, g: u8, b: u8) -> Rgb<u8> {
    let rf = r as f32 / 16.0;
    let gf = g as f32 / 16.0;
    let bf = b as f32 / 16.0;
    let r0 = (rf.floor() as usize).min(LUT_SIZE - 2);
    let g0 = (gf.floor() as usize).min(LUT_SIZE - 2);
    let b0 = (bf.floor() as usize).min(LUT_SIZE - 2);
    let rd = rf - r0 as f32;
    let gd = gf - g0 as f32;
    let bd = bf - b0 as f32;

    let c000 = lut.table[r0][g0][b0];
    let c001 = lut.table[r0][g0][b0 + 1];
    let c010 = lut.table[r0][g0 + 1][b0];
    let c011 = lut.table[r0][g0 + 1][b0 + 1];
    let c100 = lut.table[r0 + 1][g0][b0];
    let c101 = lut.table[r0 + 1][g0][b0 + 1];
    let c110 = lut.table[r0 + 1][g0 + 1][b0];
    let c111 = lut.table[r0 + 1][g0 + 1][b0 + 1];

    let i = |a: u8, b: u8, c: u8, d: u8| -> u8 {
        let ab = a as f32 * (1.0 - rd) + c as f32 * rd;
        let cd = b as f32 * (1.0 - rd) + d as f32 * rd;
        (ab * (1.0 - gd) + cd * gd) as u8
    };

    Rgb([
        (i(c000[0], c001[0], c100[0], c101[0]) as f32 * (1.0 - bd)
            + i(c010[0], c011[0], c110[0], c111[0]) as f32 * bd)
            .clamp(0.0, 255.0) as u8,
        (i(c000[1], c001[1], c100[1], c101[1]) as f32 * (1.0 - bd)
            + i(c010[1], c011[1], c110[1], c111[1]) as f32 * bd)
            .clamp(0.0, 255.0) as u8,
        (i(c000[2], c001[2], c100[2], c101[2]) as f32 * (1.0 - bd)
            + i(c010[2], c011[2], c110[2], c111[2]) as f32 * bd)
            .clamp(0.0, 255.0) as u8,
    ])
}

fn apply_lut_grading(
    image: &mut RgbImage,
    lut: &MangaColorLUT,
    line_mask: &GrayImage,
    original: &GrayImage,
) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink > 0.85 {
            continue;
        }

        let mapped = lut_lookup(lut, pixel[0], pixel[1], pixel[2]);
        let orig_luma = original.get_pixel(x, y)[0] as f32;
        let new_luma =
            0.2126 * mapped[0] as f32 + 0.7152 * mapped[1] as f32 + 0.0722 * mapped[2] as f32;
        let anchor = 0.10;
        let diff = orig_luma - new_luma;

        pixel[0] = (mapped[0] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
        pixel[1] = (mapped[1] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
        pixel[2] = (mapped[2] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
    }
}

// ── Edge-Aware Color Smoothness ──────────────────────────────────────────
fn edge_aware_smooth(image: &mut RgbImage, line_mask: &GrayImage) {
    let (w, h) = image.dimensions();
    let src: Vec<Rgb<u8>> = image.pixels().copied().collect();
    let spatial_sigma = 2.5f32;
    let range_sigma = 35.0f32;
    let radius: i32 = 3;

    let spatial_w: Vec<f32> = (0..=radius)
        .map(|d| (-(d * d) as f32 / (2.0 * spatial_sigma * spatial_sigma)).exp())
        .collect();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let ink = line_mask.get_pixel(x, y)[0];

            if ink > 200 {
                continue;
            }

            let center = src[idx];
            let center_luma =
                0.2126 * center[0] as f32 + 0.7152 * center[1] as f32 + 0.0722 * center[2] as f32;
            let mut sum = [0.0f32; 3];
            let mut total_w = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    let n_id = (ny * w + nx) as usize;

                    let n_ink = line_mask.get_pixel(nx, ny)[0] as f32 / 255.0;
                    if n_ink > 0.85 {
                        continue;
                    }
                    let ink_w = (1.0 - n_ink).powi(3);

                    let n = src[n_id];
                    let n_luma = 0.2126 * n[0] as f32 + 0.7152 * n[1] as f32 + 0.0722 * n[2] as f32;

                    let dist = dx.abs().max(dy.abs()) as usize;
                    let sw = spatial_w[dist];

                    let r_diff = (center_luma - n_luma).abs();
                    let rw = (-r_diff * r_diff / (2.0 * range_sigma * range_sigma)).exp();

                    let weight = sw * rw * ink_w;
                    sum[0] += n[0] as f32 * weight;
                    sum[1] += n[1] as f32 * weight;
                    sum[2] += n[2] as f32 * weight;
                    total_w += weight;
                }
            }

            if total_w > 0.0 {
                image.put_pixel(
                    x,
                    y,
                    Rgb([
                        (sum[0] / total_w).clamp(0.0, 255.0) as u8,
                        (sum[1] / total_w).clamp(0.0, 255.0) as u8,
                        (sum[2] / total_w).clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
    }
}

/// Blend original grayscale back into screentone regions so the dot/line
/// patterns remain visible instead of being painted over with flat color.
pub fn preserve_screentones(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    tone_mask: &GrayImage,
    strength: f32,
    count: &mut u64,
) {
    let blend = strength.clamp(0.2, 1.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if tone_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let orig = original_gray.get_pixel(x, y)[0] as f32 / 255.0;

        // In screentone areas, blend 60-80% original value with colorized
        let t = blend * (0.6 + 0.2 * orig);
        let orig_val = (orig * 255.0) as u8;

        for c in 0..3 {
            let colored = pixel[c] as f32;
            pixel[c] = (colored * (1.0 - t) + orig_val as f32 * t).round() as u8;
        }
        *count += 1;
    }
}

/// Adaptive line restoration: stronger in dark ink areas, gentler near edges,
/// preserving colored details in thin-line regions.
pub fn restore_lines_adaptive(image: &mut RgbImage, line_mask: &GrayImage, strength: f32) {
    let strength = strength.clamp(0.0, 2.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink <= 0.01 {
            continue;
        }

        // Progressive darkening: heavier ink = more black, lighter ink = subtle darkening
        let factor = ink.powf(1.8) * strength;
        let keep = (1.0 - factor).clamp(0.0, 1.0);

        pixel[0] = (pixel[0] as f32 * keep).round() as u8;
        pixel[1] = (pixel[1] as f32 * keep).round() as u8;
        pixel[2] = (pixel[2] as f32 * keep).round() as u8;
    }
}

pub fn sharpen_lightly(image: &mut RgbImage, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 {
        return;
    }
    let blurred = gaussian_blur_3x3(image);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let b = blurred.get_pixel(x, y);
        for c in 0..3 {
            let v = pixel[c] as f32 + (pixel[c] as f32 - b[c] as f32) * amount;
            pixel[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn gaussian_blur_3x3(image: &RgbImage) -> RgbImage {
    let (w, h) = image.dimensions();
    let mut blurred = RgbImage::new(w, h);
    let kernel: [(i32, i32, f64); 9] = [
        (-1, -1, 1.0),
        (0, -1, 2.0),
        (1, -1, 1.0),
        (-1, 0, 2.0),
        (0, 0, 4.0),
        (1, 0, 2.0),
        (-1, 1, 1.0),
        (0, 1, 2.0),
        (1, 1, 1.0),
    ];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum = [0.0f64; 3];
            for (dx, dy, weight) in &kernel {
                let px = image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                for c in 0..3 {
                    sum[c] += px[c] as f64 * weight;
                }
            }
            blurred.put_pixel(
                x,
                y,
                Rgb([
                    (sum[0] / 16.0).round().clamp(0.0, 255.0) as u8,
                    (sum[1] / 16.0).round().clamp(0.0, 255.0) as u8,
                    (sum[2] / 16.0).round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    for y in 0..h {
        for x in 0..w {
            if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                blurred.put_pixel(x, y, *image.get_pixel(x, y));
            }
        }
    }
    blurred
}

fn blend(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    #[test]
    fn bubble_cleanup_whitens_bright_regions_and_preserves_text() {
        let mut image = RgbImage::from_pixel(2, 1, Rgb([120, 80, 60]));
        let mut original = GrayImage::from_pixel(2, 1, Luma([255]));
        original.put_pixel(1, 0, Luma([0]));
        let mask = GrayImage::from_pixel(2, 1, Luma([255]));
        let mut restored = 0;

        clean_speech_bubbles(&mut image, &original, &mask, &mut restored);

        assert_eq!(restored, 2);
        assert_eq!(image.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(image.get_pixel(1, 0).0, [120, 80, 60]);
    }
}
