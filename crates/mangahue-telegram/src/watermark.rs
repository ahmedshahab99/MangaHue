//! Page watermark: draws the manga title + page number on the delivered
//! image (bottom-right, white text with a black shadow, sized relative to
//! the page width). The font is embedded so the binary needs no system
//! fonts at runtime.

use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};

static FONT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/DejaVuSans-Bold.ttf");

/// Draw `title — p.{page}/{total}` at the bottom-right of the page.
pub fn draw_page_watermark(image: &mut RgbImage, title: &str, page: usize, total: usize) {
    let font = match FontArc::try_from_slice(FONT_BYTES) {
        Ok(font) => font,
        Err(_) => return, // never expected: embedded font
    };
    let width = image.width() as f32;
    if width < 64.0 {
        return;
    }
    let scale_px = (width / 60.0).clamp(16.0, 40.0);
    let scale = PxScale::from(scale_px);
    let text = if title.trim().is_empty() {
        format!("p.{page}/{total}")
    } else {
        format!("{title}  •  p.{page}/{total}")
    };
    let (tw, th) = text_size(scale, &font, &text);
    let pad = (scale_px * 0.9).round() as i32;
    let x = (image.width() as i32 - tw as i32 - pad).max(pad);
    let y = (image.height() as i32 - th as i32 - pad).max(pad);
    // Shadow, then text — readable on any background.
    draw_text_mut(image, Rgb([0, 0, 0]), x + 2, y + 2, scale, &font, &text);
    draw_text_mut(image, Rgb([255, 255, 255]), x, y, scale, &font, &text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watermark_renders_without_panicking_on_any_size() {
        for size in [(64, 64), (512, 800), (1280, 1827)] {
            let mut image = RgbImage::from_pixel(size.0, size.1, Rgb([200, 200, 200]));
            draw_page_watermark(&mut image, "Test Manga", 3, 20);
            draw_page_watermark(&mut image, "", 1, 1);
            // Some pixels must have changed (text drawn).
            assert!(image.pixels().any(|p| p[0] < 100 || p[0] > 240));
        }
    }

    #[test]
    fn watermark_respects_tiny_pages() {
        let mut image = RgbImage::from_pixel(32, 32, Rgb([200, 200, 200]));
        draw_page_watermark(&mut image, "T", 1, 1); // must not panic
    }
}
