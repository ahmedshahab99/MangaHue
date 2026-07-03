use anyhow::Result;
use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use mangahue_core::StyleProfile;

#[derive(Debug, Clone)]
pub struct PostprocessInput<'a> {
    pub colorized: RgbImage,
    pub original_grayscale: &'a GrayImage,
    pub line_mask: &'a GrayImage,
    pub bubble_mask: &'a GrayImage,
    pub style: &'a StyleProfile,
    pub config: &'a PostprocessSection,
}

pub fn finalize(input: PostprocessInput<'_>) -> Result<RgbImage> {
    let mut output = input.colorized;
    if input.config.palette_correction {
        apply_palette_curve(&mut output, input.style);
    }
    if input.config.bubble_cleanup {
        clean_speech_bubbles(&mut output, input.original_grayscale, input.bubble_mask);
    }
    if input.config.bleeding_correction {
        reduce_color_near_ink(&mut output, input.line_mask);
    }
    if input.config.restore_lines {
        restore_lines(
            &mut output,
            input.line_mask,
            input.config.line_strength * input.style.line_strength,
        );
    }
    if input.config.final_sharpening > 0.0 {
        sharpen_lightly(&mut output, input.config.final_sharpening);
    }
    Ok(output)
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
) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if bubble_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let original = original_gray.get_pixel(x, y)[0];
        if original < 105 {
            *pixel = Rgb([0, 0, 0]);
        } else {
            *pixel = Rgb([255, 255, 255]);
        }
    }
}

pub fn reduce_color_near_ink(image: &mut RgbImage, line_mask: &GrayImage) {
    let source = image.clone();
    if image.width() < 3 || image.height() < 3 {
        return;
    }
    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            if line_mask.get_pixel(x, y)[0] > 0 {
                continue;
            }
            let mut nearby_ink = 0;
            for ny in y - 1..=y + 1 {
                for nx in x - 1..=x + 1 {
                    if line_mask.get_pixel(nx, ny)[0] > 0 {
                        nearby_ink += 1;
                    }
                }
            }
            if nearby_ink == 0 {
                continue;
            }
            let pixel = source.get_pixel(x, y);
            let gray = ((pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3) as u8;
            let t = (nearby_ink as f32 / 8.0).clamp(0.0, 0.45);
            image.put_pixel(
                x,
                y,
                Rgb([
                    blend(pixel[0], gray, t),
                    blend(pixel[1], gray, t),
                    blend(pixel[2], gray, t),
                ]),
            );
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

pub fn sharpen_lightly(image: &mut RgbImage, amount: f32) {
    let amount = amount.clamp(0.0, 0.6);
    let source = image.clone();
    if image.width() < 3 || image.height() < 3 {
        return;
    }
    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            let center = source.get_pixel(x, y);
            let mut avg = [0u16; 3];
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                let pixel = source.get_pixel(nx, ny);
                avg[0] += pixel[0] as u16;
                avg[1] += pixel[1] as u16;
                avg[2] += pixel[2] as u16;
            }
            let mut next = [0u8; 3];
            for channel in 0..3 {
                let blur = avg[channel] as f32 / 4.0;
                let sharpened = center[channel] as f32 + (center[channel] as f32 - blur) * amount;
                next[channel] = sharpened.round().clamp(0.0, 255.0) as u8;
            }
            image.put_pixel(x, y, Rgb(next));
        }
    }
}

fn blend(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}
