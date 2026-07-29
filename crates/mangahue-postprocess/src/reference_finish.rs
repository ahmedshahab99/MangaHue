use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use mangahue_core::StyleProfile;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReferenceFinishReport {
    pub pixels_finished: u64,
    pub chroma_pixels_expanded: u64,
    pub shadow_pixels_shaped: u64,
}

pub(crate) fn finish_reference_style(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    _protected_mask: Option<&GrayImage>,
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
    let mut report = ReferenceFinishReport::default();

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
            let paper = smoothstep(0.91, 0.995, original) * (1.0 - smoothstep(0.018, 0.06, chroma));
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
                * (0.42 + dark_support * 0.58)
                * (0.58 + (1.0 - scene[index]) * 0.42);
            let local_shape =
                detail * config.local_contrast * 1.7 + broad_detail * config.local_contrast * 0.72;
            let style_shadow = shadow_support * style.shadow_strength * 0.018;
            let style_highlight = highlight_support * style.highlight_strength * 0.008;
            let target_luma =
                (old_luma - depth - style_shadow + style_highlight + local_shape).clamp(0.0, 1.0);

            let neutral_gate = smoothstep(0.008, 0.035, chroma);
            let vivid_gate = smoothstep(0.015, 0.13, chroma);
            let chroma_room = bell(target_luma, 0.48, 0.52);
            let expansion = 1.0
                + config.target_colorfulness
                    * neutral_gate
                    * chroma_room
                    * (0.42 + (1.0 - vivid_gate) * 0.34);
            let mut centered = [
                (rgb[0] - old_luma) * expansion,
                (rgb[1] - old_luma) * expansion,
                (rgb[2] - old_luma) * expansion,
            ];

            let warm_material = centered[0] > centered[2] + 0.008;
            let cool_material = centered[2] > centered[0] + 0.008;
            let shadow_tint = shadow_support * style.shadow_strength * 0.008;
            if warm_material {
                centered[0] += highlight_support * style.highlight_strength * 0.006;
                centered[2] -= shadow_tint * 0.58;
            } else if cool_material {
                centered[2] += shadow_tint * 0.48;
            } else if neutral_gate > 0.15 {
                centered[0] += highlight_support * 0.0025;
                centered[2] += shadow_tint * 0.28;
            }

            let mapped = gamut_map(target_luma, centered);
            let blend = paintability * (0.72 + midtone * 0.28);
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
        }
    }

    report
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
    (1.0 - ((value - center) / radius.max(0.001)).abs()).clamp(0.0, 1.0)
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
            final_sharpening: 0.0,
            reference_finishing: true,
            target_colorfulness: 1.35,
            midtone_depth: 0.12,
            local_contrast: 0.18,
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
            &StyleProfile::default(),
            &config(),
        );

        assert!(image.pixels().all(|pixel| *pixel == original_pixel));
    }
}
