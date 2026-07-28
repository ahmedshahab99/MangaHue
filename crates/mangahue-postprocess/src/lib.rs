mod region_renderer;

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
    #[serde(default)]
    pub chroma_pixels_completed: u64,
    #[serde(default)]
    pub material_pixels_refined: u64,
    #[serde(default)]
    pub material_regions: u64,
    #[serde(default)]
    pub seeded_material_regions: u64,
    #[serde(default)]
    pub inferred_material_regions: u64,
    #[serde(default)]
    pub region_pixels_rendered: u64,
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
        chroma_pixels_completed: 0,
        material_pixels_refined: 0,
        material_regions: 0,
        seeded_material_regions: 0,
        inferred_material_regions: 0,
        region_pixels_rendered: 0,
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
        apply_palette_harmony(&mut output, input.style, input.line_mask);
        complete_material_chroma(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            None,
            input.style,
            &mut report.chroma_pixels_completed,
            &mut report.material_pixels_refined,
        );
        let region_report = region_renderer::render_region_graph(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            input.style,
        );
        report.material_regions = region_report.regions;
        report.seeded_material_regions = region_report.seeded_regions;
        report.inferred_material_regions = region_report.inferred_regions;
        report.region_pixels_rendered = region_report.pixels_rendered;
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

pub fn reconstruct_full_resolution(colorized: &RgbImage, original_gray: &GrayImage) -> RgbImage {
    if colorized.dimensions() == original_gray.dimensions() {
        return colorized.clone();
    }

    let (width, height) = original_gray.dimensions();
    let mut output = image::imageops::resize(
        colorized,
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    );

    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let original = original_gray.get_pixel(x, y)[0] as f32;
        let left = original_gray.get_pixel(x.saturating_sub(1), y)[0] as f32;
        let right = original_gray.get_pixel((x + 1).min(width - 1), y)[0] as f32;
        let top = original_gray.get_pixel(x, y.saturating_sub(1))[0] as f32;
        let bottom = original_gray.get_pixel(x, (y + 1).min(height - 1))[0] as f32;
        let gradient = (right - left).abs() + (bottom - top).abs();
        let edge = ((gradient - 18.0) / 90.0).clamp(0.0, 1.0);
        let predicted_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        let target_luma = original * 0.58 + predicted_luma * 0.42;
        let tonal_distance = ((target_luma / 255.0 - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let desired_chroma_scale = (1.08 - tonal_distance * 0.16) * (1.0 - edge * 0.76);
        let mut gamut_scale = desired_chroma_scale;
        for channel in 0..3 {
            let chroma = pixel[channel] as f32 - predicted_luma;
            if chroma > 0.0 {
                gamut_scale = gamut_scale.min((255.0 - target_luma) / chroma);
            } else if chroma < 0.0 {
                gamut_scale = gamut_scale.min(target_luma / -chroma);
            }
        }
        let chroma_scale = gamut_scale.clamp(0.0, desired_chroma_scale);

        for channel in 0..3 {
            let chroma = pixel[channel] as f32 - predicted_luma;
            pixel[channel] = (target_luma + chroma * chroma_scale)
                .round()
                .clamp(0.0, 255.0) as u8;
        }

        let ink = ((112.0 - original) / 112.0).clamp(0.0, 1.0).powf(1.7);
        let ink_anchor = ink * 0.94;
        for channel in 0..3 {
            pixel[channel] = (pixel[channel] as f32 * (1.0 - ink_anchor) + original * ink_anchor)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    output
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
        let sat = style.saturation.clamp(0.7, 1.15);
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

fn apply_palette_harmony(image: &mut RgbImage, style: &StyleProfile, line_mask: &GrayImage) {
    if style.palette.is_empty() {
        return;
    }
    let palette = style
        .palette
        .iter()
        .map(|color| {
            let r = color.r as f32;
            let g = color.g as f32;
            let b = color.b as f32;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            (luma, b - luma, r - luma, r.max(g).max(b) - r.min(g).min(b))
        })
        .collect::<Vec<_>>();

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink > 0.72 {
            continue;
        }
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let cb = b - luma;
        let cr = r - luma;
        let saturation = r.max(g).max(b) - r.min(g).min(b);
        let source_is_neutral = saturation < 8.0;

        let target = palette
            .iter()
            .filter(|entry| (entry.3 < 10.0) == source_is_neutral)
            .min_by(|left, right| {
                let distance = |entry: &&(f32, f32, f32, f32)| {
                    let dl = (luma - entry.0) * 0.22;
                    let dcb = cb - entry.1;
                    let dcr = cr - entry.2;
                    dl * dl + dcb * dcb + dcr * dcr
                };
                distance(left).total_cmp(&distance(right))
            })
            .or_else(|| {
                palette.iter().min_by(|left, right| {
                    let distance = |entry: &&(f32, f32, f32, f32)| {
                        let dl = (luma - entry.0) * 0.22;
                        let dcb = cb - entry.1;
                        let dcr = cr - entry.2;
                        dl * dl + dcb * dcb + dcr * dcr
                    };
                    distance(left).total_cmp(&distance(right))
                })
            });
        let Some((_, target_cb, target_cr, _)) = target else {
            continue;
        };

        let strength = 0.16 * (1.0 - ink).powi(2);
        let harmonized_cb = cb + (target_cb - cb) * strength;
        let harmonized_cr = cr + (target_cr - cr) * strength;
        pixel[0] = (luma + harmonized_cr).round().clamp(0.0, 255.0) as u8;
        pixel[2] = (luma + harmonized_cb).round().clamp(0.0, 255.0) as u8;
        pixel[1] = ((luma - 0.2126 * pixel[0] as f32 - 0.0722 * pixel[2] as f32) / 0.7152)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

/// Complete sparse model chroma inside manga regions without changing geometry.
///
/// The model's confident colors are treated as seeds. Chroma diffuses through
/// similar grayscale pixels, while ink and strong luminance transitions act as
/// barriers. Remaining neutral material receives a restrained luminance-matched
/// palette prior. Original luminance is retained, so line art and screentones
/// remain available to later reconstruction stages.
pub fn complete_material_chroma(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    exclusion_mask: Option<&GrayImage>,
    style: &StyleProfile,
    completed: &mut u64,
    refined: &mut u64,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
    {
        return;
    }

    let len = width as usize * height as usize;
    let source_pixels = image.pixels().copied().collect::<Vec<_>>();
    let mut luma = vec![0.0f32; len];
    let mut red_chroma = vec![0.0f32; len];
    let mut blue_chroma = vec![0.0f32; len];
    let mut confidence = vec![0.0f32; len];
    let mut fixed_seed = vec![false; len];
    let mut blocked = vec![false; len];
    let mut gray = vec![0u8; len];

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = source_pixels[index];
            let y_value =
                0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
            let chroma =
                pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
            let ink = line_mask.get_pixel(x, y)[0];
            let excluded = exclusion_mask
                .map(|mask| mask.get_pixel(x, y)[0])
                .unwrap_or(0);
            let is_blocked = ink > 112 || excluded > 96;
            let seed_confidence = ((chroma as f32 - 5.0) / 24.0).clamp(0.0, 1.0);

            luma[index] = y_value;
            red_chroma[index] = pixel[0] as f32 - y_value;
            blue_chroma[index] = pixel[2] as f32 - y_value;
            confidence[index] = if is_blocked { 0.0 } else { seed_confidence };
            fixed_seed[index] = seed_confidence >= 0.48 && !is_blocked;
            blocked[index] = is_blocked;
            gray[index] = original_gray.get_pixel(x, y)[0];
        }
    }

    let similarity = (0..=255)
        .map(|difference| (-(difference as f32) / 20.0).exp())
        .collect::<Vec<_>>();
    let mut next_red = red_chroma.clone();
    let mut next_blue = blue_chroma.clone();
    let mut next_confidence = confidence.clone();
    const ITERATIONS: usize = 18;
    const NEIGHBORS: [(i32, i32); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];

    for _ in 0..ITERATIONS {
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if blocked[index] || fixed_seed[index] {
                    next_red[index] = red_chroma[index];
                    next_blue[index] = blue_chroma[index];
                    next_confidence[index] = confidence[index];
                    continue;
                }

                let mut red_sum = 0.0f32;
                let mut blue_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                let mut support_sum = 0.0f32;
                for (dx, dy) in NEIGHBORS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let neighbor = (ny as u32 * width + nx as u32) as usize;
                    if blocked[neighbor] || confidence[neighbor] <= 0.005 {
                        continue;
                    }
                    let difference = gray[index].abs_diff(gray[neighbor]) as usize;
                    let diagonal = if dx != 0 && dy != 0 { 0.707 } else { 1.0 };
                    let weight = confidence[neighbor] * similarity[difference] * diagonal;
                    red_sum += red_chroma[neighbor] * weight;
                    blue_sum += blue_chroma[neighbor] * weight;
                    weight_sum += weight;
                    support_sum += similarity[difference] * diagonal;
                }

                if weight_sum > 0.01 {
                    let proposal_red = red_sum / weight_sum;
                    let proposal_blue = blue_sum / weight_sum;
                    let proposal_confidence = (weight_sum / support_sum.max(0.01)).clamp(0.0, 1.0);
                    let spread =
                        (proposal_confidence * (1.0 - confidence[index]) * 0.58).clamp(0.0, 0.58);
                    next_red[index] =
                        red_chroma[index] + (proposal_red - red_chroma[index]) * spread;
                    next_blue[index] =
                        blue_chroma[index] + (proposal_blue - blue_chroma[index]) * spread;
                    next_confidence[index] = confidence[index].max(proposal_confidence * 0.82);
                } else {
                    next_red[index] = red_chroma[index];
                    next_blue[index] = blue_chroma[index];
                    next_confidence[index] = confidence[index];
                }
            }
        }
        std::mem::swap(&mut red_chroma, &mut next_red);
        std::mem::swap(&mut blue_chroma, &mut next_blue);
        std::mem::swap(&mut confidence, &mut next_confidence);
    }

    let palette = style
        .palette
        .iter()
        .map(|color| {
            let r = color.r as f32;
            let g = color.g as f32;
            let b = color.b as f32;
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            (y, r - y, b - y)
        })
        .collect::<Vec<_>>();

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if blocked[index] {
                continue;
            }

            let old_chroma = source_pixels[index][0]
                .max(source_pixels[index][1])
                .max(source_pixels[index][2])
                - source_pixels[index][0]
                    .min(source_pixels[index][1])
                    .min(source_pixels[index][2]);
            let propagated = confidence[index];
            let mut target_red = red_chroma[index];
            let mut target_blue = blue_chroma[index];

            if propagated < 0.22 && gray[index] > 32 && gray[index] < 247 {
                if let Some((_, palette_red, palette_blue)) =
                    palette.iter().min_by(|left, right| {
                        (luma[index] - left.0)
                            .abs()
                            .total_cmp(&(luma[index] - right.0).abs())
                    })
                {
                    let prior = if gray[index] < 92 {
                        0.18
                    } else if gray[index] < 188 {
                        0.12
                    } else {
                        0.075
                    };
                    target_red += (*palette_red - target_red) * prior;
                    target_blue += (*palette_blue - target_blue) * prior;
                    *refined += 1;
                }
            }

            let apply_strength = if fixed_seed[index] {
                1.0
            } else {
                (0.28 + propagated * 0.72).clamp(0.28, 0.92)
            };
            let red = rgb_from_luma_chroma(
                luma[index],
                target_red * apply_strength,
                target_blue * apply_strength,
            );
            image.put_pixel(x, y, red);

            let new_pixel = image.get_pixel(x, y);
            let new_chroma = new_pixel[0].max(new_pixel[1]).max(new_pixel[2])
                - new_pixel[0].min(new_pixel[1]).min(new_pixel[2]);
            if old_chroma < 10 && new_chroma >= 10 {
                *completed += 1;
            }
        }
    }

    refine_skin_and_material_light(image, original_gray, line_mask, exclusion_mask, style);
}

fn rgb_from_luma_chroma(luma: f32, red_chroma: f32, blue_chroma: f32) -> Rgb<u8> {
    let green_chroma = (-0.2126 * red_chroma - 0.0722 * blue_chroma) / 0.7152;
    let mut scale = 1.0f32;
    for chroma in [red_chroma, green_chroma, blue_chroma] {
        if chroma > 0.0 {
            scale = scale.min((255.0 - luma) / chroma);
        } else if chroma < 0.0 {
            scale = scale.min(luma / -chroma);
        }
    }
    let red = luma + red_chroma * scale.clamp(0.0, 1.0);
    let blue = luma + blue_chroma * scale.clamp(0.0, 1.0);
    let green = luma + green_chroma * scale.clamp(0.0, 1.0);
    Rgb([
        red.round().clamp(0.0, 255.0) as u8,
        green.round().clamp(0.0, 255.0) as u8,
        blue.round().clamp(0.0, 255.0) as u8,
    ])
}

fn refine_skin_and_material_light(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    exclusion_mask: Option<&GrayImage>,
    style: &StyleProfile,
) {
    let (width, height) = image.dimensions();
    let source = image.clone();
    for y in 0..height {
        for x in 0..width {
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let excluded = exclusion_mask
                .map(|mask| mask.get_pixel(x, y)[0])
                .unwrap_or(0);
            if ink > 0.44 || excluded > 96 {
                continue;
            }
            let pixel = source.get_pixel(x, y);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let y_value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let red_delta = r - g;
            let blue_delta = g - b;
            let skin = y_value > 72.0
                && y_value < 246.0
                && red_delta > 6.0
                && red_delta < 82.0
                && blue_delta > -4.0
                && blue_delta < 58.0;

            let original = original_gray.get_pixel(x, y)[0] as f32;
            let shadow = ((154.0 - original) / 154.0).clamp(0.0, 1.0);
            let highlight = ((original - 205.0) / 50.0).clamp(0.0, 1.0);
            let shade =
                -shadow * style.shadow_strength * 5.0 + highlight * style.highlight_strength * 2.5;
            let mut out = [r + shade, g + shade, b + shade];

            if skin {
                let warmth = (1.0 - shadow * 0.45) * (1.0 - ink) * 2.8;
                out[0] += warmth;
                out[1] += warmth * 0.28;
                out[2] -= warmth * 0.38;
            } else if y_value < 108.0 {
                out[0] += style.shadow_strength * 0.9;
                out[2] += style.shadow_strength * 1.4;
            }

            image.put_pixel(
                x,
                y,
                Rgb([
                    out[0].round().clamp(0.0, 255.0) as u8,
                    out[1].round().clamp(0.0, 255.0) as u8,
                    out[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
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

    let lerp = |left: f32, right: f32, amount: f32| left + (right - left) * amount;
    let channel = |index: usize| {
        let r0g0 = lerp(c000[index] as f32, c001[index] as f32, bd);
        let r0g1 = lerp(c010[index] as f32, c011[index] as f32, bd);
        let r1g0 = lerp(c100[index] as f32, c101[index] as f32, bd);
        let r1g1 = lerp(c110[index] as f32, c111[index] as f32, bd);
        let r0 = lerp(r0g0, r0g1, gd);
        let r1 = lerp(r1g0, r1g1, gd);
        lerp(r0, r1, rd).round().clamp(0.0, 255.0) as u8
    };

    Rgb([channel(0), channel(1), channel(2)])
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
        let original_luma = original_gray.get_pixel(x, y)[0] as f32;
        let colored_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        let tone_strength = blend * (0.52 + 0.16 * (1.0 - original_luma / 255.0));
        let target_luma =
            colored_luma + (original_luma - colored_luma) * tone_strength.clamp(0.0, 0.72);
        let chroma = [
            pixel[0] as f32 - colored_luma,
            pixel[1] as f32 - colored_luma,
            pixel[2] as f32 - colored_luma,
        ];
        let mut gamut_scale = 1.0f32;
        for value in chroma {
            if value > 0.0 {
                gamut_scale = gamut_scale.min((255.0 - target_luma) / value);
            } else if value < 0.0 {
                gamut_scale = gamut_scale.min(target_luma / -value);
            }
        }
        let gamut_scale = gamut_scale.clamp(0.0, 1.0);
        for channel in 0..3 {
            pixel[channel] = (target_luma + chroma[channel] * gamut_scale)
                .round()
                .clamp(0.0, 255.0) as u8;
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

    #[test]
    fn broad_protected_mask_does_not_erase_generated_color() {
        let colorized = RgbImage::from_pixel(3, 3, Rgb([210, 132, 104]));
        let original = GrayImage::from_pixel(3, 3, Luma([180]));
        let lines = GrayImage::from_pixel(3, 3, Luma([0]));
        let bubbles = GrayImage::from_pixel(3, 3, Luma([0]));
        let protected = GrayImage::from_pixel(3, 3, Luma([255]));
        let style = StyleProfile::default();
        let config = PostprocessSection {
            restore_lines: false,
            line_strength: 1.0,
            bubble_cleanup: true,
            bleeding_correction: false,
            palette_correction: false,
            final_sharpening: 0.0,
        };

        let (output, report) = finalize(PostprocessInput {
            colorized,
            original_grayscale: &original,
            line_mask: &lines,
            bubble_mask: &bubbles,
            tone_mask: None,
            protected_mask: Some(&protected),
            style: &style,
            config: &config,
        })
        .unwrap();

        assert_eq!(output.get_pixel(1, 1).0, [210, 132, 104]);
        assert_eq!(report.protected_pixels_restored, 0);
    }

    #[test]
    fn full_resolution_reconstruction_restores_dimensions_and_dark_ink() {
        let colorized = RgbImage::from_pixel(2, 2, Rgb([220, 150, 130]));
        let mut original = GrayImage::from_pixel(4, 4, Luma([230]));
        original.put_pixel(1, 1, Luma([0]));

        let restored = reconstruct_full_resolution(&colorized, &original);

        assert_eq!(restored.dimensions(), (4, 4));
        let ink = restored.get_pixel(1, 1);
        assert!(ink[0] < 20 && ink[1] < 20 && ink[2] < 20);
        let color = restored.get_pixel(3, 3);
        assert!(color[0] > color[1] && color[1] > color[2]);
    }

    #[test]
    fn full_resolution_reconstruction_suppresses_edge_chroma() {
        let colorized = RgbImage::from_pixel(2, 2, Rgb([230, 70, 40]));
        let mut original = GrayImage::from_pixel(4, 4, Luma([230]));
        for y in 0..4 {
            original.put_pixel(0, y, Luma([40]));
            original.put_pixel(1, y, Luma([40]));
        }

        let restored = reconstruct_full_resolution(&colorized, &original);
        let edge = restored.get_pixel(2, 2);
        let interior = restored.get_pixel(3, 2);
        let edge_chroma = edge.0.iter().max().unwrap() - edge.0.iter().min().unwrap();
        let interior_chroma = interior.0.iter().max().unwrap() - interior.0.iter().min().unwrap();

        assert!(edge_chroma < interior_chroma);
    }

    #[test]
    fn neutral_lut_is_continuous_and_preserves_channel_order() {
        let style = StyleProfile {
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
            ..StyleProfile::default()
        };
        let lut = MangaColorLUT::new(&style);

        let low = lut_lookup(&lut, 100, 140, 180);
        let high = lut_lookup(&lut, 101, 141, 181);

        assert!(low[0] < low[1] && low[1] < low[2]);
        for channel in 0..3 {
            assert!((high[channel] as i16 - low[channel] as i16).abs() <= 2);
        }
    }

    #[test]
    fn palette_harmony_preserves_neutral_pixels_and_guides_colored_pixels() {
        let style = StyleProfile {
            palette: vec![
                mangahue_core::ColorRgb::new(238, 199, 174),
                mangahue_core::ColorRgb::new(245, 245, 242),
            ],
            ..StyleProfile::default()
        };
        let mut image = RgbImage::from_pixel(2, 1, Rgb([210, 120, 150]));
        image.put_pixel(1, 0, Rgb([240, 240, 240]));
        let lines = GrayImage::from_pixel(2, 1, Luma([0]));

        apply_palette_harmony(&mut image, &style, &lines);

        let colored = image.get_pixel(0, 0);
        assert!(colored[1] > 120);
        assert!(colored[0] - colored[1] < 90);
        assert_eq!(image.get_pixel(1, 0).0, [240, 240, 240]);
    }

    #[test]
    fn screentone_restoration_preserves_chroma_while_recovering_luma() {
        let mut image = RgbImage::from_pixel(1, 1, Rgb([180, 80, 60]));
        let original = GrayImage::from_pixel(1, 1, Luma([180]));
        let tone_mask = GrayImage::from_pixel(1, 1, Luma([255]));
        let before_luma: f32 = 0.2126 * 180.0 + 0.7152 * 80.0 + 0.0722 * 60.0;
        let mut restored = 0;

        preserve_screentones(&mut image, &original, &tone_mask, 1.0, &mut restored);

        let pixel = image.get_pixel(0, 0);
        let after_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        let chroma = pixel.0.iter().max().unwrap() - pixel.0.iter().min().unwrap();
        assert_eq!(restored, 1);
        assert!((180.0f32 - after_luma).abs() < (180.0f32 - before_luma).abs());
        assert!(chroma >= 100);
    }

    #[test]
    fn chroma_completion_fills_region_but_respects_ink_wall() {
        let mut image = RgbImage::from_pixel(15, 5, Rgb([150, 150, 150]));
        image.put_pixel(2, 2, Rgb([205, 112, 88]));
        let original = GrayImage::from_pixel(15, 5, Luma([150]));
        let mut lines = GrayImage::from_pixel(15, 5, Luma([0]));
        for y in 0..5 {
            lines.put_pixel(7, y, Luma([255]));
        }
        let protected = GrayImage::from_pixel(15, 5, Luma([0]));
        let style = StyleProfile {
            palette: Vec::new(),
            ..StyleProfile::default()
        };
        let mut completed = 0;
        let mut refined = 0;

        complete_material_chroma(
            &mut image,
            &original,
            &lines,
            Some(&protected),
            &style,
            &mut completed,
            &mut refined,
        );

        let left = image.get_pixel(5, 2);
        let right = image.get_pixel(11, 2);
        assert!(left[0] > left[1] && left[1] > left[2]);
        assert_eq!(right.0, [150, 150, 150]);
        assert!(completed > 0);
        assert_eq!(refined, 0);
    }

    #[test]
    fn material_prior_adds_restrained_color_and_preserves_luminance() {
        let mut image = RgbImage::from_pixel(9, 9, Rgb([170, 170, 170]));
        let original = GrayImage::from_pixel(9, 9, Luma([170]));
        let lines = GrayImage::from_pixel(9, 9, Luma([0]));
        let protected = GrayImage::from_pixel(9, 9, Luma([0]));
        let style = StyleProfile {
            palette: vec![mangahue_core::ColorRgb::new(190, 145, 125)],
            ..StyleProfile::default()
        };
        let mut completed = 0;
        let mut refined = 0;

        complete_material_chroma(
            &mut image,
            &original,
            &lines,
            Some(&protected),
            &style,
            &mut completed,
            &mut refined,
        );

        let pixel = image.get_pixel(4, 4);
        let output_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        assert!(pixel[0] > pixel[2]);
        assert!((output_luma - 170.0).abs() < 8.0);
        assert!(refined > 0);
    }
}
