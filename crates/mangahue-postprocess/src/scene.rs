use image::GrayImage;
use mangahue_core::ColorRgb;
use serde::{Deserialize, Serialize};

/// Scene environment classification. Every manga page has its own
/// lighting environment (day, sunset, night, overcast, indoor); the
/// colorization palette must adapt to it instead of applying one fixed
/// look. Classification is derived from the grayscale statistics of the
/// page itself — no external model needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneLighting {
    Day,
    Overcast,
    Sunset,
    Night,
    Indoor,
}

impl SceneLighting {
    pub fn label(&self) -> &'static str {
        match self {
            SceneLighting::Day => "daylight",
            SceneLighting::Overcast => "overcast",
            SceneLighting::Sunset => "sunset",
            SceneLighting::Night => "night",
            SceneLighting::Indoor => "indoor",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SceneProfile {
    pub lighting: SceneLighting,
    /// -1 (cool) .. +1 (warm): how strongly the scene leans warm.
    pub warmth: f32,
    /// 0..1 mean page brightness.
    pub brightness: f32,
    /// Gray std-dev (structure density / drama).
    pub contrast: f32,
}

/// Classify the page's lighting environment from its grayscale histogram.
pub fn analyze_scene(gray: &GrayImage) -> SceneProfile {
    let (width, height) = gray.dimensions();
    let total = (width as u64).saturating_mul(height as u64).max(1);
    let mut dark = 0u64;
    let mut bright = 0u64;
    let mut sum = 0u64;
    let mut sq = 0.0f64;
    for pixel in gray.pixels() {
        let v = pixel[0] as u32;
        sum += v as u64;
        sq += (v as f64) * (v as f64);
        if v < 70 {
            dark += 1;
        } else if v > 225 {
            bright += 1;
        }
    }
    let mean = sum as f64 / total as f64;
    let contrast = (sq / total as f64 - mean * mean).max(0.0).sqrt() as f32;
    let dark_ratio = dark as f64 / total as f64;
    let bright_ratio = bright as f64 / total as f64;
    let mean = mean as f32;

    let lighting = if mean < 105.0 && dark_ratio > 0.35 {
        SceneLighting::Night
    } else if mean < 105.0 {
        SceneLighting::Indoor
    } else if mean > 170.0 && contrast < 45.0 {
        SceneLighting::Overcast
    } else if mean > 160.0 && contrast >= 45.0 {
        SceneLighting::Day
    } else if mean >= 105.0 && mean <= 160.0 && contrast >= 55.0 {
        SceneLighting::Sunset
    } else if bright_ratio < 0.18 && mean <= 160.0 {
        SceneLighting::Night
    } else {
        SceneLighting::Indoor
    };

    let warmth = match lighting {
        SceneLighting::Night => -0.5,
        SceneLighting::Overcast => -0.2,
        SceneLighting::Day => 0.0,
        SceneLighting::Sunset => 0.7,
        SceneLighting::Indoor => 0.35,
    };
    SceneProfile {
        lighting,
        warmth,
        brightness: (mean / 255.0).clamp(0.0, 1.0),
        contrast,
    }
}

/// Re-weight the style palette for the scene. Colors whose luminance fits
/// the page and whose hue aligns with the scene's warmth rank highest; the
/// result always keeps one representative per major hue family (warm,
/// cool, neutral-light, neutral-dark) so the diversity director can
/// separate adjacent regions even in strongly warm or cool scenes.
pub fn scene_palette(palette: &[ColorRgb], scene: &SceneProfile, keep: usize) -> Vec<ColorRgb> {
    if palette.is_empty() {
        return Vec::new();
    }
    let keep = keep.max(4).min(20);
    let target_luma = scene.brightness * 255.0;
    let mut warm: Option<(f32, ColorRgb)> = None;
    let mut cool: Option<(f32, ColorRgb)> = None;
    let mut light_neutral: Option<(f32, ColorRgb)> = None;
    let mut dark_neutral: Option<(f32, ColorRgb)> = None;

    let mut scored: Vec<(f32, ColorRgb)> = palette
        .iter()
        .map(|color| {
            let luma = color_luma(*color);
            let chroma = color_chroma(*color);
            let warm_hue = if chroma < 18.0 {
                0.0
            } else {
                ((color.r as f32 - color.b as f32) / chroma).clamp(-1.0, 1.0)
            };
            let luma_fit = 1.0 - (luma - target_luma).abs() / 255.0;
            let hue_align = warm_hue * scene.warmth;
            let night_bonus = if matches!(scene.lighting, SceneLighting::Night) {
                (1.0 - luma / 255.0) * 0.9
            } else {
                0.0
            };
            let day_bonus = if matches!(scene.lighting, SceneLighting::Day) {
                luma / 255.0 * 0.3
            } else {
                0.0
            };
            let score =
                luma_fit * 0.55 + hue_align.clamp(-1.0, 1.0) * 0.35 + night_bonus + day_bonus;

            let family = if chroma < 18.0 {
                if luma > 190.0 {
                    &mut light_neutral
                } else {
                    &mut dark_neutral
                }
            } else if warm_hue > 0.15 {
                &mut warm
            } else {
                &mut cool
            };
            match family {
                Some((best, _)) if *best >= score => {}
                _ => *family = Some((score, *color)),
            }
            (score, *color)
        })
        .collect();

    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut chosen: Vec<ColorRgb> = Vec::with_capacity(keep);
    for (_, color) in scored {
        if chosen.len() >= keep {
            break;
        }
        if !chosen.contains(&color) {
            chosen.push(color);
        }
    }
    // Guarantee hue-family diversity for the region director.
    for family in [warm, cool, light_neutral, dark_neutral] {
        if let Some((_, color)) = family {
            if chosen.len() < keep && !chosen.contains(&color) {
                chosen.push(color);
            }
        }
    }
    chosen
}

fn color_luma(color: ColorRgb) -> f32 {
    0.2126 * color.r as f32 + 0.7152 * color.g as f32 + 0.0722 * color.b as f32
}

fn color_chroma(color: ColorRgb) -> f32 {
    color.r.max(color.g).max(color.b) as f32 - color.r.min(color.g).min(color.b) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn classifies_bright_high_contrast_page_as_day() {
        let mut gray = GrayImage::from_pixel(64, 64, Luma([240]));
        for y in 0..64 {
            for x in 0..64 {
                if (x + y) % 4 == 0 {
                    gray.put_pixel(x, y, Luma([30]));
                }
            }
        }
        let scene = analyze_scene(&gray);
        assert_eq!(scene.lighting, SceneLighting::Day);
    }

    #[test]
    fn classifies_dark_page_as_night() {
        let mut gray = GrayImage::from_pixel(64, 64, Luma([40]));
        for x in 0..64 {
            gray.put_pixel(x, 32, Luma([180]));
        }
        let scene = analyze_scene(&gray);
        assert_eq!(scene.lighting, SceneLighting::Night);
    }

    #[test]
    fn scene_palette_keeps_all_hue_families() {
        let palette = vec![
            ColorRgb::new(242, 179, 143), // warm
            ColorRgb::new(66, 103, 214),  // cool
            ColorRgb::new(240, 240, 240), // light neutral
            ColorRgb::new(40, 40, 44),    // dark neutral
        ];
        let scene = SceneProfile {
            lighting: SceneLighting::Night,
            warmth: -0.5,
            brightness: 0.2,
            contrast: 60.0,
        };
        let result = scene_palette(&palette, &scene, 4);
        assert!(result.len() >= 4, "got {}", result.len());
        assert!(result.contains(&ColorRgb::new(66, 103, 214)));
    }
}
