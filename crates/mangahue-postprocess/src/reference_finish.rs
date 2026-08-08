use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use mangahue_core::StyleProfile;

// Peach skin bases in linear-light (sRGB 240,200,172 / 196,144,116) used by
// the guaranteed face paint so faces match body skin even when the model
// outputs near-white.
const SKIN_LINEAR: [f32; 3] = [0.8715, 0.5774, 0.4125];
const SKIN_SHA_LINEAR: [f32; 3] = [0.5519, 0.2789, 0.1747];

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReferenceFinishReport {
    pub pixels_finished: u64,
    pub chroma_pixels_expanded: u64,
    pub shadow_pixels_shaped: u64,
    pub relit_pixels: u64,
    pub face_pixels_painted: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_reference_style(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    _protected_mask: Option<&GrayImage>,
    face_mask: Option<&GrayImage>,
    style: &StyleProfile,
    config: &PostprocessSection,
) -> ReferenceFinishReport {
    let (width, height) = image.dimensions();
    if width < 3
        || height < 3
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
        || bubble_mask.dimensions() != (width, height)
    {
        return ReferenceFinishReport::default();
    }

    let source = image.clone();
    let luma = source
        .pixels()
        .map(|pixel| linear_luma(to_linear(*pixel)))
        .collect::<Vec<_>>();
    let local = box_mean(&luma, width, height, 7);
    let scene = box_mean(&luma, width, height, 29);
    let ink_distance = config
        .semantic_relighting
        .then(|| distance_from_ink(line_mask, 18));
    let mut report = ReferenceFinishReport::default();

    // Specular ridge field: second-derivative of local luma highlights thin
    // convex ridges (hair tufts, cheekbones, fabric folds) — not flat fills.
    let specular = {
        let detail7 = luma
            .iter()
            .zip(local.iter())
            .map(|(l, m)| l - m)
            .collect::<Vec<f32>>();
        let detail_blur = box_mean(&detail7, width, height, 9);
        luma.iter()
            .enumerate()
            .map(|(i, _)| (detail7[i] - detail_blur[i]).clamp(-0.2, 0.2))
            .collect::<Vec<f32>>()
    };

    // Estimate the dominant light direction from the page's gradient field,
    // then build a directional AO field: ink casts shadows away from light.
    let (light_dir_x, light_dir_y) = if config.semantic_relighting {
        estimate_light_direction(&luma, width, height, line_mask, bubble_mask)
    } else {
        (-0.55_f32, -0.84_f32)
    };
    let directional_ao = config
        .semantic_relighting
        .then(|| {
            ink_distance.as_ref().map(|distance| {
                directional_ink_ao(distance, width, height, light_dir_x, light_dir_y, line_mask)
            })
        })
        .flatten();

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let bubble = bubble_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let exclusion = bubble;
            if ink > 0.72 || exclusion > 0.38 {
                continue;
            }

            let original = original_gray.get_pixel(x, y)[0] as f32 / 255.0;
            let rgb = to_linear(*source.get_pixel(x, y));
            let old_luma = luma[index];
            let max_channel = rgb[0].max(rgb[1]).max(rgb[2]);
            let min_channel = rgb[0].min(rgb[1]).min(rgb[2]);
            let chroma = max_channel - min_channel;
            // Paper detection: only TRUE paper (>=0.985 with near-zero
            // chroma) is skipped. Bright backgrounds and white materials
            // (0.92-0.985) are paintable so the page reads fully colored —
            // the soft-coverage tier colors those regions in the region
            // renderer, and this pass adds the warm lighting on top.
            let mut paper =
                smoothstep(0.985, 0.998, original) * (1.0 - smoothstep(0.018, 0.06, chroma));
            // Detected face regions are NEVER paper — they must be painted
            // regardless of brightness.
            if let Some(mask) = face_mask {
                if mask.dimensions() == (width, height) {
                    paper *= 1.0 - (mask.get_pixel(x, y)[0] as f32 / 255.0) * 0.98;
                }
            }
            let paintability =
                (1.0 - ink).powi(2) * (1.0 - exclusion).powi(3) * (1.0 - paper * 0.96);
            if paintability < 0.02 {
                continue;
            }

            let detail = (old_luma - local[index]).clamp(-0.16, 0.16);
            let broad_detail = (local[index] - scene[index]).clamp(-0.18, 0.18);
            let midtone = bell(old_luma, 0.52, 0.47);
            let dark_support = smoothstep(0.97, 0.20, original);
            let shadow_support = smoothstep(0.78, 0.18, original);
            let highlight_support = smoothstep(0.70, 0.96, original);

            let depth = config.midtone_depth
                * midtone
                * (0.38 + dark_support * 0.62)
                * (0.58 + (1.0 - scene[index]) * 0.42);
            let local_shape =
                detail * config.local_contrast * 1.7 + broad_detail * config.local_contrast * 0.72;
            let style_shadow = shadow_support * style.shadow_strength * 0.018;
            let style_highlight = highlight_support * style.highlight_strength * 0.008;
            let mut relighting = 0.0;
            if let Some(distance) = ink_distance.as_ref() {
                let edge_depth = 1.0 - smoothstep(1.5, 15.0, distance[index]);
                let x0 = x.saturating_sub(3);
                let x1 = (x + 3).min(width - 1);
                let y0 = y.saturating_sub(3);
                let y1 = (y + 3).min(height - 1);
                let gradient_x =
                    local[(y * width + x1) as usize] - local[(y * width + x0) as usize];
                let gradient_y =
                    local[(y1 * width + x) as usize] - local[(y0 * width + x) as usize];
                // Directional gradient along the estimated light direction
                // (light from light_dir, shading opposite).
                let directional =
                    (-gradient_x * light_dir_x - gradient_y * light_dir_y).clamp(-0.08, 0.08);
                let material_gate = (1.0 - paper) * bell(original, 0.55, 0.52);
                let mut ao = 0.0;
                if let Some(ao_field) = directional_ao.as_ref() {
                    // Ink casts shadow on the side facing away from light.
                    ao = ao_field[index] * shadow_support * 0.03;
                }
                relighting = config.relighting_strength
                    * material_gate
                    * (directional * 0.34 - edge_depth * shadow_support * 0.018 - ao);
            }
            let mut target_luma =
                (old_luma - depth - style_shadow + style_highlight + local_shape + relighting)
                    .clamp(0.0, 1.0);

            let neutral_gate = smoothstep(0.008, 0.035, chroma);
            let vivid_gate = smoothstep(0.015, 0.13, chroma);
            let chroma_room = bell(target_luma, 0.48, 0.52);
            // Deep expansion: midtones get the strongest lift (2.1-2.4x),
            // highlights damped by chroma_room, capped to stay in gamut.
            let expansion = (1.0
                + config.target_colorfulness
                    * neutral_gate
                    * chroma_room
                    * (0.55 + (1.0 - vivid_gate) * 0.35))
                .min(2.4);
            let mut centered = [
                (rgb[0] - old_luma) * expansion,
                (rgb[1] - old_luma) * expansion,
                (rgb[2] - old_luma) * expansion,
            ];

            let warm_material = centered[0] > centered[2] + 0.008;
            let cool_material = centered[2] > centered[0] + 0.008;
            let shadow_tint = shadow_support * style.shadow_strength * 0.008;
            // Skin subsurface scattering: shadows on skin go deep-warm amber,
            // the lit edge gets a warm bounce. Extended to DEEP shadows
            // (original > 0.22) so the darkest skin still warms.
            let skin_like = warm_material
                && original > 0.22
                && original < 0.80
                && chroma < 0.28
                && centered[0] < 0.16;
            if skin_like {
                let s = shadow_support;
                centered[0] += s * 0.045;
                centered[2] -= s * 0.016;
                centered[0] += smoothstep(0.0, 0.12, broad_detail) * 0.018;
            }
            if warm_material {
                centered[0] += highlight_support * style.highlight_strength * 0.014;
                centered[2] -= shadow_tint * 0.60;
            } else if cool_material {
                centered[0] += shadow_tint * 0.18;
                centered[2] += shadow_tint * 0.12;
            } else if neutral_gate > 0.15 {
                centered[0] += highlight_support * 0.004;
                centered[2] += shadow_tint * 0.08;
            }

            // Specular highlights: thin convex ridges brighten and desaturate
            // (white-ish gloss on hair/skin/folds). Gated to non-paper areas;
            // only STRONG ridges qualify so noisy midtones aren't flattened.
            let ridge = smoothstep(0.06, 0.14, specular[index]);
            if ridge > 0.01 && paper < 0.4 && chroma < 0.2 {
                let gloss = ridge * (1.0 - ink) * 0.04;
                target_luma = (target_luma + gloss).min(1.0);
                let gloss_chroma = ridge * 0.25;
                centered[0] *= 1.0 - gloss_chroma;
                centered[1] *= 1.0 - gloss_chroma;
                centered[2] *= 1.0 - gloss_chroma;
            }

            // GUARANTEED FACE PAINT: the model frequently outputs near-white
            // for bright faces. When a face box is known, inject a peach skin
            // base whose depth follows the page's own shading (bright → light
            // peach, shadowed → deep warm). The luma is also pulled toward
            // the skin base so gamut_map has headroom for chroma — a face
            // kept at 0.95 luma would clip its red channel back to white.
            if let Some(mask) = face_mask {
                if mask.dimensions() == (width, height) {
                    let fm = mask.get_pixel(x, y)[0] as f32 / 255.0;
                    if fm > 0.15 {
                        let depth = (1.0 - target_luma).clamp(0.0, 0.9);
                        let w = (depth / 0.35).clamp(0.0, 1.0);
                        let base = [
                            SKIN_LINEAR[0] * (1.0 - w) + SKIN_SHA_LINEAR[0] * w,
                            SKIN_LINEAR[1] * (1.0 - w) + SKIN_SHA_LINEAR[1] * w,
                            SKIN_LINEAR[2] * (1.0 - w) + SKIN_SHA_LINEAR[2] * w,
                        ];
                        let base_luma = 0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2];
                        let k = (fm * 0.92).min(0.92);
                        centered[0] = centered[0] * (1.0 - k) + (base[0] - base_luma) * k;
                        centered[1] = centered[1] * (1.0 - k) + (base[1] - base_luma) * k;
                        centered[2] = centered[2] * (1.0 - k) + (base[2] - base_luma) * k;
                        // Pull brightness toward the skin base while keeping
                        // 55% of the page's own shading structure.
                        let base_target = base_luma + (target_luma - base_luma) * 0.55;
                        target_luma = target_luma * (1.0 - k) + base_target * k;
                        report.face_pixels_painted += 1;
                    }
                }
            }

            let mapped = gamut_map(target_luma, centered);
            let blend = paintability * (0.82 + midtone * 0.18);
            let output = [
                rgb[0] + (mapped[0] - rgb[0]) * blend,
                rgb[1] + (mapped[1] - rgb[1]) * blend,
                rgb[2] + (mapped[2] - rgb[2]) * blend,
            ];
            *image.get_pixel_mut(x, y) = from_linear(output);
            report.pixels_finished += 1;
            if chroma > 0.012 && expansion > 1.08 {
                report.chroma_pixels_expanded += 1;
            }
            if depth + style_shadow > 0.012 {
                report.shadow_pixels_shaped += 1;
            }
            if relighting.abs() > 0.001 {
                report.relit_pixels += 1;
            }
        }
    }

    report
}

/// Estimate the dominant light direction from the page's gradient field.
/// Returns a unit-ish (x, y) vector pointing TOWARD the light source.
/// Falls back to a top-left light when the page has no clear direction.
fn estimate_light_direction(
    luma: &[f32],
    width: u32,
    height: u32,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
) -> (f32, f32) {
    let width_usize = width as usize;
    // 8-bin circular gradient histogram, weighted by magnitude.
    let mut hist = [0.0f32; 8];
    let mut total = 0.0f32;
    for y in 1..height as usize - 1 {
        for x in 1..width_usize - 1 {
            let ink = line_mask.get_pixel(x as u32, y as u32)[0];
            let bubble = bubble_mask.get_pixel(x as u32, y as u32)[0];
            if ink > 118 || bubble > 72 {
                continue;
            }
            let idx = y * width_usize + x;
            let gx = luma[idx + 1] - luma[idx - 1];
            let gy = luma[idx + width_usize] - luma[idx - width_usize];
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude < 0.004 {
                continue;
            }
            let theta = gy.atan2(gx);
            let bin = (((theta + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)) * 8.0)
                .floor() as usize
                % 8;
            hist[bin] += magnitude;
            total += magnitude;
        }
    }
    if total < 1.0 {
        return (-0.55, -0.84);
    }
    let (max_bin, max_value) = hist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap_or((2, &0.0));
    // Directional coherence: if no single direction dominates, use default.
    if max_value / total < 0.16 {
        return (-0.55, -0.84);
    }
    // Light comes from opposite the dominant gradient direction.
    let angle = ((max_bin as f32 + 4.0) % 8.0) * (2.0 * std::f32::consts::PI) / 8.0;
    let (sin, cos) = angle.sin_cos();
    (cos, sin)
}

/// Directional ink AO: ink casts a shadow on the side facing away from the
/// light. Computed from the distance field derivative along the light axis.
fn directional_ink_ao(
    distance: &[f32],
    width: u32,
    height: u32,
    light_x: f32,
    light_y: f32,
    _line_mask: &GrayImage,
) -> Vec<f32> {
    let width_usize = width as usize;
    let len = width_usize * height as usize;
    let mut ao = vec![0.0f32; len];
    let step = 6usize;
    // Step in integer lattice, normalized.
    let dx = (light_x * step as f32).round() as i32;
    let dy = (light_y * step as f32).round() as i32;
    let norm = ((dx * dx + dy * dy) as f32).sqrt().max(1.0);
    let sx = (dx as f32 / norm) as i32;
    let sy = (dy as f32 / norm) as i32;
    if sx == 0 && sy == 0 {
        return ao;
    }
    for y in 0..height as usize {
        for x in 0..width_usize {
            let fwd_x = x as i32 + sx;
            let fwd_y = y as i32 + sy;
            let back_x = x as i32 - sx;
            let back_y = y as i32 - sy;
            if fwd_x < 0 || fwd_y < 0 || back_x < 0 || back_y < 0 {
                continue;
            }
            let (fx, fy) = (fwd_x as usize, fwd_y as usize);
            let (bx, by) = (back_x as usize, back_y as usize);
            if fx >= width_usize
                || fy >= height as usize
                || bx >= width_usize
                || by >= height as usize
            {
                continue;
            }
            let idx = y * width_usize + x;
            // More ink ahead (away from light) than behind => shadow here.
            let diff = distance[fy * width_usize + fx] - distance[by * width_usize + bx];
            ao[idx] = (diff * 0.12).clamp(0.0, 1.0);
        }
    }
    ao
}

fn distance_from_ink(line_mask: &GrayImage, max_distance: u16) -> Vec<f32> {
    let (width, height) = line_mask.dimensions();
    let width_usize = width as usize;
    let mut distance = vec![max_distance; width_usize * height as usize];
    for y in 0..height as usize {
        for x in 0..width_usize {
            if line_mask.get_pixel(x as u32, y as u32)[0] >= 112 {
                distance[y * width_usize + x] = 0;
            } else {
                let mut best = max_distance;
                if x > 0 {
                    best = best.min(distance[y * width_usize + x - 1].saturating_add(1));
                }
                if y > 0 {
                    best = best.min(distance[(y - 1) * width_usize + x].saturating_add(1));
                }
                distance[y * width_usize + x] = best;
            }
        }
    }
    for y in (0..height as usize).rev() {
        for x in (0..width_usize).rev() {
            let index = y * width_usize + x;
            let mut best = distance[index];
            if x + 1 < width_usize {
                best = best.min(distance[index + 1].saturating_add(1));
            }
            if y + 1 < height as usize {
                best = best.min(distance[index + width_usize].saturating_add(1));
            }
            distance[index] = best;
        }
    }
    distance.into_iter().map(f32::from).collect()
}

fn box_mean(source: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let integral_width = width as usize + 1;
    let mut integral = vec![0.0_f64; integral_width * (height as usize + 1)];
    for y in 0..height as usize {
        let mut row_sum = 0.0_f64;
        for x in 0..width as usize {
            row_sum += source[y * width as usize + x] as f64;
            integral[(y + 1) * integral_width + x + 1] =
                integral[y * integral_width + x + 1] + row_sum;
        }
    }

    let mut output = vec![0.0_f32; source.len()];
    for y in 0..height {
        let top = y.saturating_sub(radius) as usize;
        let bottom = (y + radius + 1).min(height) as usize;
        for x in 0..width {
            let left = x.saturating_sub(radius) as usize;
            let right = (x + radius + 1).min(width) as usize;
            let sum = integral[bottom * integral_width + right]
                - integral[top * integral_width + right]
                - integral[bottom * integral_width + left]
                + integral[top * integral_width + left];
            let count = (right - left) * (bottom - top);
            output[(y * width + x) as usize] = (sum / count.max(1) as f64) as f32;
        }
    }
    output
}

fn to_linear(pixel: Rgb<u8>) -> [f32; 3] {
    [
        srgb_to_linear(pixel[0] as f32 / 255.0),
        srgb_to_linear(pixel[1] as f32 / 255.0),
        srgb_to_linear(pixel[2] as f32 / 255.0),
    ]
}

fn from_linear(rgb: [f32; 3]) -> Rgb<u8> {
    Rgb([
        (linear_to_srgb(rgb[0].clamp(0.0, 1.0)) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
        (linear_to_srgb(rgb[1].clamp(0.0, 1.0)) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
        (linear_to_srgb(rgb[2].clamp(0.0, 1.0)) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
    ])
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn gamut_map(luma: f32, chroma: [f32; 3]) -> [f32; 3] {
    let mut scale = 1.0_f32;
    for component in chroma {
        if component > 0.0 {
            scale = scale.min((1.0 - luma) / component);
        } else if component < 0.0 {
            scale = scale.min(luma / -component);
        }
    }
    let scale = scale.clamp(0.0, 1.0);
    if scale < 1.0 && luma > 0.85 {
        // Very bright pixels have almost no gamut headroom: crushing the
        // chroma would turn warm paper and lit backgrounds back to white.
        // Instead, gently pull the luma down (never below 0.86, so paper
        // stays paper-bright) until the color fits.
        let mut lowered = luma;
        for component in chroma {
            if component > 0.0 {
                lowered = lowered.min(1.0 - component);
            }
        }
        let lowered = lowered.clamp(0.86, 1.0);
        let mut scale2 = 1.0_f32;
        for component in chroma {
            if component > 0.0 {
                scale2 = scale2.min((1.0 - lowered) / component);
            } else if component < 0.0 {
                scale2 = scale2.min(lowered / -component);
            }
        }
        let scale2 = scale2.clamp(0.0, 1.0);
        if scale2 >= scale {
            return [
                lowered + chroma[0] * scale2,
                lowered + chroma[1] * scale2,
                lowered + chroma[2] * scale2,
            ];
        }
    }
    [
        luma + chroma[0] * scale,
        luma + chroma[1] * scale,
        luma + chroma[2] * scale,
    ]
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return f32::from(value >= edge1);
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bell(value: f32, center: f32, radius: f32) -> f32 {
    // Smooth quadratic falloff (no linear kink at the tent tip) for soft,
    // banding-free shading transitions.
    let d = ((value - center) / radius.max(0.001)).abs();
    (1.0 - d * d).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn config() -> PostprocessSection {
        PostprocessSection {
            restore_lines: true,
            line_strength: 1.0,
            bubble_cleanup: true,
            bleeding_correction: true,
            palette_correction: true,
            region_chroma_separation: 34.0,
            final_sharpening: 0.0,
            reference_finishing: true,
            target_colorfulness: 1.35,
            midtone_depth: 0.12,
            local_contrast: 0.18,
            semantic_relighting: true,
            relighting_strength: 0.4,
            natural_grading: true,
            reference_transfer: true,
            soft_coverage: true,
            muted_target_grade: false,
            muted_strength: 0.95,
            muted_hue_shift_deg: 16.0,
            muted_dark_pull: 0.92,
            soft_strength: 1.0,
            soft_confidence: 1.0,
            muted_dark_sat: 0.19,
            muted_mid_sat: 0.27,
            muted_bright_sat: 0.22,
            muted_paper_sat: 0.034,
        }
    }

    #[test]
    fn finishing_expands_supported_chroma_and_shapes_midtones() {
        let mut image = RgbImage::from_pixel(32, 32, Rgb([194, 171, 164]));
        let gray = GrayImage::from_pixel(32, 32, Luma([188]));
        let empty = GrayImage::from_pixel(32, 32, Luma([0]));
        let before = *image.get_pixel(16, 16);

        let report = finish_reference_style(
            &mut image,
            &gray,
            &empty,
            &empty,
            None,
            None,
            &StyleProfile::default(),
            &config(),
        );

        let after = image.get_pixel(16, 16);
        assert!(after[0].saturating_sub(after[2]) > before[0].saturating_sub(before[2]));
        assert!(linear_luma(to_linear(*after)) < linear_luma(to_linear(before)));
        assert!(report.chroma_pixels_expanded > 0);
        assert!(report.shadow_pixels_shaped > 0);
    }

    #[test]
    fn finishing_never_modifies_bubbles() {
        let original_pixel = Rgb([210, 168, 150]);
        let gray = GrayImage::from_pixel(12, 12, Luma([190]));
        let lines = GrayImage::from_pixel(12, 12, Luma([0]));
        let bubbles = GrayImage::from_pixel(12, 12, Luma([255]));
        let mut image = RgbImage::from_pixel(12, 12, original_pixel);
        finish_reference_style(
            &mut image,
            &gray,
            &lines,
            &bubbles,
            None,
            None,
            &StyleProfile::default(),
            &config(),
        );
        assert!(image.pixels().all(|pixel| *pixel == original_pixel));
    }

    #[test]
    fn finishing_preserves_strong_ink() {
        let original_pixel = Rgb([32, 28, 30]);
        let mut image = RgbImage::from_pixel(12, 12, original_pixel);
        let gray = GrayImage::from_pixel(12, 12, Luma([30]));
        let lines = GrayImage::from_pixel(12, 12, Luma([255]));
        let empty = GrayImage::from_pixel(12, 12, Luma([0]));

        finish_reference_style(
            &mut image,
            &gray,
            &lines,
            &empty,
            None,
            None,
            &StyleProfile::default(),
            &config(),
        );

        assert!(image.pixels().all(|pixel| *pixel == original_pixel));
    }

    #[test]
    fn relighting_adds_depth_near_ink_without_crossing_it() {
        let mut image = RgbImage::from_pixel(32, 32, Rgb([202, 168, 154]));
        let gray = GrayImage::from_pixel(32, 32, Luma([100]));
        let mut lines = GrayImage::from_pixel(32, 32, Luma([0]));
        for y in 0..32 {
            lines.put_pixel(8, y, Luma([255]));
        }
        let empty = GrayImage::from_pixel(32, 32, Luma([0]));
        let untouched_ink = *image.get_pixel(8, 16);

        let report = finish_reference_style(
            &mut image,
            &gray,
            &lines,
            &empty,
            None,
            None,
            &StyleProfile::default(),
            &config(),
        );

        assert_eq!(*image.get_pixel(8, 16), untouched_ink);
        assert!(report.relit_pixels > 0);
        assert_ne!(*image.get_pixel(10, 16), *image.get_pixel(24, 16));
    }
}
