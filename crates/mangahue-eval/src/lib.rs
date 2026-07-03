use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub line_preservation_score: f32,
    pub bubble_cleanliness_score: f32,
    pub protected_region_violation_score: f32,
    pub hint_adherence_score: f32,
    pub output_sharpness_estimate: f32,
}

pub fn evaluate_output(
    output: &RgbImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
) -> QualityReport {
    evaluate_output_with_guidance(output, line_mask, bubble_mask, None, None, None)
}

pub fn evaluate_output_with_guidance(
    output: &RgbImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: Option<&GrayImage>,
    hint_map: Option<&RgbImage>,
    hint_mask: Option<&GrayImage>,
) -> QualityReport {
    QualityReport {
        line_preservation_score: line_preservation_score(output, line_mask),
        bubble_cleanliness_score: bubble_cleanliness_score(output, bubble_mask),
        protected_region_violation_score: protected_region_violation_score(
            output,
            protected_mask.unwrap_or(bubble_mask),
        ),
        hint_adherence_score: match (hint_map, hint_mask) {
            (Some(hint_map), Some(hint_mask)) => hint_adherence_score(output, hint_map, hint_mask),
            _ => 1.0,
        },
        output_sharpness_estimate: output_sharpness_estimate(output),
    }
}

pub fn line_preservation_score(output: &RgbImage, line_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut dark = 0u32;
    for (x, y, mask) in line_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
        if luma < 48 {
            dark += 1;
        }
    }
    ratio(dark, samples)
}

pub fn bubble_cleanliness_score(output: &RgbImage, bubble_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut clean = 0u32;
    for (x, y, mask) in bubble_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let max_delta = pixel[0].abs_diff(pixel[1]).max(pixel[1].abs_diff(pixel[2]));
        if pixel[0] > 235 && pixel[1] > 235 && pixel[2] > 235 && max_delta < 8 {
            clean += 1;
        }
    }
    if samples == 0 {
        1.0
    } else {
        ratio(clean, samples)
    }
}

pub fn protected_region_violation_score(output: &RgbImage, protected_mask: &GrayImage) -> f32 {
    let mut samples = 0u32;
    let mut clean = 0u32;
    for (x, y, mask) in protected_mask.enumerate_pixels() {
        if mask[0] == 0 {
            continue;
        }
        samples += 1;
        let pixel = output.get_pixel(x, y);
        let chroma = pixel[0].abs_diff(pixel[1]).max(pixel[1].abs_diff(pixel[2]));
        let bright = pixel[0] > 230 && pixel[1] > 230 && pixel[2] > 230;
        let dark = pixel[0] < 36 && pixel[1] < 36 && pixel[2] < 36;
        if bright || dark || chroma < 8 {
            clean += 1;
        }
    }
    if samples == 0 {
        1.0
    } else {
        ratio(clean, samples)
    }
}

pub fn hint_adherence_score(output: &RgbImage, hint_map: &RgbImage, hint_mask: &GrayImage) -> f32 {
    if output.dimensions() != hint_map.dimensions() || output.dimensions() != hint_mask.dimensions()
    {
        return 0.0;
    }
    let mut weighted_error = 0.0f32;
    let mut total_weight = 0.0f32;
    for (x, y, mask) in hint_mask.enumerate_pixels() {
        let weight = mask[0] as f32 / 255.0;
        if weight <= 0.02 {
            continue;
        }
        let output = output.get_pixel(x, y);
        let hint = hint_map.get_pixel(x, y);
        let dr = output[0] as f32 - hint[0] as f32;
        let dg = output[1] as f32 - hint[1] as f32;
        let db = output[2] as f32 - hint[2] as f32;
        let error = (dr * dr + dg * dg + db * db).sqrt() / 441.67295;
        weighted_error += error * weight;
        total_weight += weight;
    }
    if total_weight == 0.0 {
        1.0
    } else {
        (1.0 - weighted_error / total_weight).clamp(0.0, 1.0)
    }
}

pub fn output_sharpness_estimate(output: &RgbImage) -> f32 {
    if output.width() < 3 || output.height() < 3 {
        return 0.0;
    }
    let mut total = 0f32;
    let mut samples = 0u32;
    for y in 1..output.height() - 1 {
        for x in 1..output.width() - 1 {
            let center = luma(output.get_pixel(x, y).0);
            let right = luma(output.get_pixel(x + 1, y).0);
            let down = luma(output.get_pixel(x, y + 1).0);
            total += (center - right).abs() + (center - down).abs();
            samples += 2;
        }
    }
    (total / samples.max(1) as f32 / 255.0).clamp(0.0, 1.0)
}

fn luma(pixel: [u8; 3]) -> f32 {
    0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32
}

fn ratio(count: u32, total: u32) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}
