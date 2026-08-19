use mangahue_core::ColorRgb;
use mangahue_hints::{
    RegionCoordinateSpace, RegionLock, RegionLockFile, RegionLockMode, RegionRun, RegionSemantic,
    RegionShape,
};
use serde::{Deserialize, Serialize};

use crate::conflict::Resolution;
use crate::PageContext;

/// The unified color plan produced by agent deliberation, bound to real
/// page geometry. Every adjustment references the exact structure-graph
/// region it belongs to so the engine can paint the right pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPlan {
    /// Per-region color adjustments with exact geometry.
    pub hint_adjustments: Vec<HintAdjustment>,
    /// Target colors for post-inference verification.
    pub region_color_targets: Vec<RegionColorTarget>,
    /// Lighting assumptions for postprocessing.
    pub lighting_plan: LightingPlan,
    /// Scene signal agreed by deliberation (SceneContext authority).
    pub scene: SceneSignal,
    /// Confidence in the plan.
    pub confidence: f32,
}

/// The page-level scene signal resolved during deliberation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneSignal {
    pub mood: String,
    pub time_of_day: String,
    pub lighting: String,
    pub palette_bias: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintAdjustment {
    pub region_id: String,
    pub semantic: String,
    pub color: ColorRgb,
    pub strength: f32,
    /// Exact source-resolution run-length mask when the node carries one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_runs: Vec<RegionRun>,
    /// Fallback normalized rectangle when no exact mask exists.
    pub normalized_bounds: mangahue_memory::NormalizedRegion,
    /// Source resolution the mask runs refer to.
    pub source_width: u32,
    pub source_height: u32,
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

fn semantic_to_region_semantic(semantic: &str) -> RegionSemantic {
    match semantic {
        "skin" => RegionSemantic::Skin,
        "hair" => RegionSemantic::Hair,
        "clothing" => RegionSemantic::Clothing,
        "background" => RegionSemantic::Background,
        "eyes" => RegionSemantic::Custom,
        _ => RegionSemantic::Custom,
    }
}

impl ColorPlan {
    /// Bind resolved colors to the real geometry of the structure graph.
    pub fn from_resolution(resolution: Resolution, context: &PageContext) -> Self {
        let mut hint_adjustments = Vec::new();
        let mut region_color_targets = Vec::new();
        let mut total_confidence = 0.0;
        let mut count = 0u32;

        for color in &resolution.colors {
            // Locate the node this region id belongs to.
            let node = context
                .structure_graph
                .nodes
                .iter()
                .find(|node| node.id == color.region_id);

            let (mask_runs, normalized_bounds) = match node {
                Some(node) => (node.mask_runs.clone(), node.normalized_bounds.clone()),
                None => (
                    Vec::new(),
                    mangahue_memory::NormalizedRegion {
                        x: 0.0,
                        y: 0.0,
                        width: 0.0,
                        height: 0.0,
                    },
                ),
            };
            // Skip proposals that do not map to any real page region.
            if mask_runs.is_empty()
                && (normalized_bounds.width <= 0.0 || normalized_bounds.height <= 0.0)
            {
                continue;
            }

            // The model stays the artist: agent paint strength is capped so
            // the hint remains guidance, not a flat repaint. Wider feather
            // keeps region borders soft. Lower strength lets the model's
            // creative output survive instead of forcing palette colors.
            let strength = color.confidence.clamp(0.15, 0.45);
            hint_adjustments.push(HintAdjustment {
                region_id: color.region_id.clone(),
                semantic: color.semantic.clone(),
                color: color.color,
                strength,
                mask_runs,
                normalized_bounds,
                source_width: context.structure_graph.size.width,
                source_height: context.structure_graph.size.height,
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

        // Lighting plan follows the scene signal.
        let lighting_plan = match resolution.scene.time_of_day.as_str() {
            "night" | "dusk" => LightingPlan {
                direction: "top_right".to_string(),
                shadow_strength: 0.85,
                highlight_strength: 0.15,
                ambient_strength: 0.35,
            },
            "indoor" => LightingPlan {
                direction: "top".to_string(),
                shadow_strength: 0.55,
                highlight_strength: 0.25,
                ambient_strength: 0.5,
            },
            "morning" | "afternoon" => LightingPlan {
                direction: "top_left".to_string(),
                shadow_strength: 0.45,
                highlight_strength: 0.35,
                ambient_strength: 0.55,
            },
            _ => LightingPlan::default(),
        };

        Self {
            hint_adjustments,
            region_color_targets,
            lighting_plan,
            scene: resolution.scene,
            confidence: total_confidence,
        }
    }

    /// Convert the plan into a region lock file so the engine's tested
    /// lock rasterizer paints the exact pixels the agents agreed on.
    /// Exact run-length masks are used where the structure graph has them;
    /// rectangular geometry is the fallback. Feathering keeps the paint
    /// from banding at region borders.
    pub fn to_region_lock_file(&self) -> RegionLockFile {
        let locks = self
            .hint_adjustments
            .iter()
            .map(|adjustment| {
                let region = if adjustment.mask_runs.is_empty() {
                    RegionShape::Rectangle {
                        x: adjustment.normalized_bounds.x,
                        y: adjustment.normalized_bounds.y,
                        width: adjustment.normalized_bounds.width,
                        height: adjustment.normalized_bounds.height,
                    }
                } else {
                    RegionShape::MaskRle {
                        source_width: adjustment.source_width,
                        source_height: adjustment.source_height,
                        runs: adjustment.mask_runs.clone(),
                    }
                };
                RegionLock {
                    id: format!("agent_{}", adjustment.region_id),
                    label: Some(format!(
                        "agent_{}",
                        adjustment.semantic
                    )),
                    semantic: semantic_to_region_semantic(&adjustment.semantic),
                    mode: RegionLockMode::Colorize,
                    region,
                    color: Some(adjustment.color),
                    strength: adjustment.strength,
                    feather_radius: 8.0,
                    enabled: true,
                }
            })
            .collect();
        RegionLockFile {
            version: 1,
            page_id: None,
            coordinate_space: RegionCoordinateSpace::Normalized,
            source_width: None,
            source_height: None,
            locks,
        }
    }
}
