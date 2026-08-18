use mangahue_core::ColorRgb;
use serde::{Deserialize, Serialize};

use crate::bus::ProposedColor;
use crate::PageContext;

/// The unified color plan produced by agent deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPlan {
    /// Adjustments to hint maps per region.
    pub hint_adjustments: Vec<HintAdjustment>,
    /// Target colors for post-inference correction.
    pub region_color_targets: Vec<RegionColorTarget>,
    /// Lighting assumptions for postprocessing.
    pub lighting_plan: LightingPlan,
    /// Overall page mood.
    pub mood: String,
    /// Overall time of day.
    pub time_of_day: String,
    /// Confidence in the plan.
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintAdjustment {
    pub region_id: String,
    pub semantic: String,
    pub color: ColorRgb,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionColorTarget {
    pub region_id: String,
    pub semantic: String,
    pub target_color: ColorRgb,
    pub tolerance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingPlan {
    pub direction: String,
    pub shadow_strength: f32,
    pub highlight_strength: f32,
    pub ambient_strength: f32,
}

impl Default for LightingPlan {
    fn default() -> Self {
        Self {
            direction: "top_right".to_string(),
            shadow_strength: 0.6,
            highlight_strength: 0.3,
            ambient_strength: 0.4,
        }
    }
}

impl ColorPlan {
    pub fn from_resolved(resolved: Vec<ProposedColor>, context: &PageContext) -> Self {
        let mut hint_adjustments = Vec::new();
        let mut region_color_targets = Vec::new();
        let mut total_confidence = 0.0;
        let mut count = 0;

        for color in &resolved {
            hint_adjustments.push(HintAdjustment {
                region_id: color.region_id.clone(),
                semantic: color.semantic.clone(),
                color: color.color,
                strength: color.confidence,
            });

            region_color_targets.push(RegionColorTarget {
                region_id: color.region_id.clone(),
                semantic: color.semantic.clone(),
                target_color: color.color,
                tolerance: 0.15,
            });

            total_confidence += color.confidence;
            count += 1;
        }

        if count > 0 {
            total_confidence /= count as f32;
        }

        // Extract mood and time from page_id hash for variety
        let hash: u32 = context.page_id.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let mood = match hash % 5 {
            0 => "calm".to_string(),
            1 => "tense".to_string(),
            2 => "action".to_string(),
            3 => "romantic".to_string(),
            _ => "neutral".to_string(),
        };
        let time_of_day = match hash % 3 {
            0 => "morning".to_string(),
            1 => "afternoon".to_string(),
            _ => "night".to_string(),
        };

        Self {
            hint_adjustments,
            region_color_targets,
            lighting_plan: LightingPlan::default(),
            mood,
            time_of_day,
            confidence: total_confidence,
        }
    }

    /// Apply this plan to existing hint maps and region locks.
    pub fn apply_to_hints(
        &self,
        hint_map: &mut image::RgbImage,
        hint_mask: &mut image::GrayImage,
    ) {
        let (w, h) = hint_map.dimensions();
        if w == 0 || h == 0 {
            return;
        }

        for adjustment in &self.hint_adjustments {
            // Parse region bounds from the adjustment
            // For now, apply color to a central area based on semantic
            let center_x = w / 2;
            let center_y = h / 2;
            let radius = w.min(h) / 8;

            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let x = center_x as i32 + dx;
                    let y = center_y as i32 + dy;
                    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                        let dist = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
                        if dist <= 1.0 {
                            let strength = (1.0 - dist) * adjustment.strength;
                            let pixel = hint_map.get_pixel_mut(x as u32, y as u32);
                            pixel[0] = (pixel[0] as f32 * (1.0 - strength)
                                + adjustment.color.r as f32 * strength) as u8;
                            pixel[1] = (pixel[1] as f32 * (1.0 - strength)
                                + adjustment.color.g as f32 * strength) as u8;
                            pixel[2] = (pixel[2] as f32 * (1.0 - strength)
                                + adjustment.color.b as f32 * strength) as u8;

                            let mask_pixel = hint_mask.get_pixel_mut(x as u32, y as u32);
                            let current = mask_pixel[0] as f32;
                            mask_pixel[0] =
                                (current.max(strength * 255.0)).min(255.0) as u8;
                        }
                    }
                }
            }
        }
    }
}
