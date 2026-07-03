use anyhow::{Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_config::HintsSection;
use mangahue_core::{
    CharacterProfile, ColorHint, ColorRgb, HintSource, ImageSize, StyleProfile, UserHintFile,
};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct HintResult {
    pub hint_map: RgbImage,
    pub hint_mask: GrayImage,
    pub applied_hints: usize,
    pub style_hints_applied: usize,
    pub memory_hints_applied: usize,
    pub user_hints_applied: usize,
    pub notes: Vec<String>,
}

pub fn load_user_hints(path: impl AsRef<Path>) -> Result<UserHintFile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read hints {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse hints {}", path.display()))
}

pub fn generate_hints(
    size: ImageSize,
    user_hints: Option<&UserHintFile>,
    style: &StyleProfile,
    characters: &[CharacterProfile],
    config: &HintsSection,
) -> HintResult {
    let mut hint_map = RgbImage::from_pixel(size.width, size.height, Rgb([0, 0, 0]));
    let mut hint_mask = GrayImage::from_pixel(size.width, size.height, Luma([0]));
    let mut applied = 0usize;
    let mut style_hints_applied = 0usize;
    let mut memory_hints_applied = 0usize;
    let mut user_hints_applied = 0usize;
    let mut notes = Vec::new();

    if config.style_hints_enabled {
        style_hints_applied =
            apply_style_atmosphere(&mut hint_map, &mut hint_mask, style, config.hint_strength);
        applied += style_hints_applied;
    }

    if config.character_memory_enabled && !characters.is_empty() {
        notes.push(format!("loaded {} character profile(s)", characters.len()));
        if let Some(file) = user_hints {
            for hint in &file.hints {
                if let Some(memory_hint) = memory_hint_from_user_hint(hint, characters) {
                    rasterize_hint(
                        &mut hint_map,
                        &mut hint_mask,
                        &memory_hint,
                        config.hint_strength,
                    );
                    memory_hints_applied += 1;
                    applied += 1;
                }
            }
        }
    }

    if config.user_hints_enabled {
        if let Some(file) = user_hints {
            for hint in &file.hints {
                rasterize_hint(&mut hint_map, &mut hint_mask, hint, config.hint_strength);
                user_hints_applied += 1;
                applied += 1;
            }
        }
    }

    if config.hint_blur_radius > 0.0 {
        soften_mask_edges(&mut hint_mask, config.hint_blur_radius.round() as u32);
    }

    HintResult {
        hint_map,
        hint_mask,
        applied_hints: applied,
        style_hints_applied,
        memory_hints_applied,
        user_hints_applied,
        notes,
    }
}

pub fn rasterize_hint(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    hint: &ColorHint,
    global_strength: f32,
) {
    let radius = hint.radius.max(1);
    let radius_sq = (radius as i64) * (radius as i64);
    let min_x = hint.x.saturating_sub(radius);
    let min_y = hint.y.saturating_sub(radius);
    let max_x = (hint.x + radius + 1).min(hint_map.width());
    let max_y = (hint.y + radius + 1).min(hint_map.height());
    let strength = (hint.strength * global_strength).clamp(0.0, 1.0);
    let color = Rgb([hint.color.r, hint.color.g, hint.color.b]);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = x as i64 - hint.x as i64;
            let dy = y as i64 - hint.y as i64;
            let distance_sq = dx * dx + dy * dy;
            if distance_sq > radius_sq {
                continue;
            }
            let falloff = 1.0 - (distance_sq as f32 / radius_sq as f32).sqrt();
            let alpha = (falloff * strength).clamp(0.0, 1.0);
            let old = hint_map.get_pixel(x, y);
            let blended = [
                blend(old[0], color[0], alpha),
                blend(old[1], color[1], alpha),
                blend(old[2], color[2], alpha),
            ];
            hint_map.put_pixel(x, y, Rgb(blended));
            let mask_value = ((alpha * 255.0).round() as u8).max(hint_mask.get_pixel(x, y)[0]);
            hint_mask.put_pixel(x, y, Luma([mask_value]));
        }
    }
}

pub fn hint_from_character_label(
    x: u32,
    y: u32,
    radius: u32,
    label: &str,
    character: &CharacterProfile,
) -> Option<ColorHint> {
    let color = match label {
        "hair" => character.hair_color,
        "skin" => character.skin_color,
        "eyes" | "eye" => character.eye_color,
        "outfit" => character.outfit_colors.first().copied(),
        "accent" => character.accent_colors.first().copied(),
        _ => None,
    }?;
    Some(ColorHint {
        x,
        y,
        radius,
        color,
        label: Some(label.to_string()),
        strength: character.confidence.clamp(0.0, 1.0),
        source: HintSource::CharacterMemory,
    })
}

pub fn memory_hint_from_user_hint(
    user_hint: &ColorHint,
    characters: &[CharacterProfile],
) -> Option<ColorHint> {
    let label = user_hint.label.as_deref()?;
    let (character_selector, channel) = parse_character_label(label);
    let character = match character_selector {
        Some(selector) => characters
            .iter()
            .find(|character| character_matches(character, selector))?,
        None if characters.len() == 1 => &characters[0],
        None => return None,
    };
    let mut hint = hint_from_character_label(
        user_hint.x,
        user_hint.y,
        (user_hint.radius as f32 * 1.45).round().max(1.0) as u32,
        channel,
        character,
    )?;
    hint.strength = (hint.strength * 0.62).clamp(0.0, 0.82);
    hint.label = Some(format!("{}.{}", character.id, channel));
    Some(hint)
}

fn parse_character_label(label: &str) -> (Option<&str>, &str) {
    let label = label.trim();
    if let Some((character, channel)) = label.split_once(':') {
        return (Some(character.trim()), channel.trim());
    }
    if let Some((character, channel)) = label.split_once('.') {
        return (Some(character.trim()), channel.trim());
    }
    (None, label)
}

fn character_matches(character: &CharacterProfile, selector: &str) -> bool {
    character.id.eq_ignore_ascii_case(selector)
        || character.name.eq_ignore_ascii_case(selector)
        || character
            .aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(selector))
}

fn apply_style_atmosphere(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    style: &StyleProfile,
    global_strength: f32,
) -> usize {
    let Some(primary) = style.palette.first().copied() else {
        return 0;
    };
    let secondary = style
        .palette
        .get(1)
        .copied()
        .unwrap_or(ColorRgb::new(80, 100, 140));
    let width = hint_map.width();
    let height = hint_map.height();
    let radius = (width.min(height) / 12).max(12);
    let hints = [
        (width / 5, height / 5, primary),
        (width.saturating_mul(4) / 5, height / 5, secondary),
        (width / 5, height.saturating_mul(4) / 5, secondary),
        (
            width.saturating_mul(4) / 5,
            height.saturating_mul(4) / 5,
            primary,
        ),
    ];
    for (x, y, color) in hints {
        let hint = ColorHint {
            x,
            y,
            radius,
            color,
            label: Some("style_atmosphere".to_string()),
            strength: 0.18,
            source: HintSource::StyleProfile,
        };
        rasterize_hint(hint_map, hint_mask, &hint, global_strength);
    }
    4
}

fn soften_mask_edges(mask: &mut GrayImage, radius: u32) {
    if radius == 0 {
        return;
    }
    let source = mask.clone();
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let current = source.get_pixel(x, y)[0];
            if current == 255 {
                continue;
            }
            let min_x = x.saturating_sub(radius);
            let min_y = y.saturating_sub(radius);
            let max_x = (x + radius + 1).min(mask.width());
            let max_y = (y + radius + 1).min(mask.height());
            let mut best = current;
            for py in min_y..max_y {
                for px in min_x..max_x {
                    let value = source.get_pixel(px, py)[0];
                    if value > best {
                        let dx = px as f32 - x as f32;
                        let dy = py as f32 - y as f32;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let falloff = (1.0 - distance / radius as f32).clamp(0.0, 1.0);
                        best = best.max((value as f32 * falloff).round() as u8);
                    }
                }
            }
            mask.put_pixel(x, y, Luma([best]));
        }
    }
}

fn blend(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangahue_config::EngineConfig;

    #[test]
    fn rasterizes_user_hint() {
        let config = EngineConfig::default();
        let hint = ColorHint {
            x: 10,
            y: 10,
            radius: 4,
            color: ColorRgb::new(255, 0, 0),
            label: Some("hair".to_string()),
            strength: 1.0,
            source: HintSource::User,
        };
        let hints = UserHintFile { hints: vec![hint] };
        let result = generate_hints(
            ImageSize::new(32, 32),
            Some(&hints),
            &StyleProfile::default(),
            &[],
            &config.hints,
        );
        assert!(result.applied_hints >= 1);
        assert!(result.hint_mask.get_pixel(10, 10)[0] > 0);
    }

    #[test]
    fn character_memory_can_shadow_labeled_user_hint() {
        let character = CharacterProfile {
            id: "hero".to_string(),
            name: "Hero".to_string(),
            aliases: Vec::new(),
            hair_color: Some(ColorRgb::new(1, 2, 3)),
            skin_color: None,
            eye_color: None,
            outfit_colors: Vec::new(),
            accent_colors: Vec::new(),
            confidence: 0.9,
            reference_images: Vec::new(),
            notes: None,
        };
        let hint = ColorHint {
            x: 10,
            y: 10,
            radius: 4,
            color: ColorRgb::new(255, 0, 0),
            label: Some("hero.hair".to_string()),
            strength: 1.0,
            source: HintSource::User,
        };
        let memory = memory_hint_from_user_hint(&hint, &[character]).unwrap();
        assert_eq!(memory.color, ColorRgb::new(1, 2, 3));
        assert!(memory.radius > hint.radius);
    }
}
