use anyhow::Result;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ProposalTarget, ProposedColor};
use crate::config::AgentConfig;
use crate::PageContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeOfDay {
    Dawn,
    Morning,
    Noon,
    Afternoon,
    Dusk,
    Night,
    Indoor,
}

impl TimeOfDay {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dawn => "dawn",
            Self::Morning => "morning",
            Self::Noon => "noon",
            Self::Afternoon => "afternoon",
            Self::Dusk => "dusk",
            Self::Night => "night",
            Self::Indoor => "indoor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingStyle {
    DirectSun,
    Overcast,
    Artificial,
    Dramatic,
    Neon,
}

impl LightingStyle {
    pub fn label(self) -> &'static str {
        match self {
            Self::DirectSun => "direct_sun",
            Self::Overcast => "overcast",
            Self::Artificial => "artificial",
            Self::Dramatic => "dramatic",
            Self::Neon => "neon",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    Calm,
    Tense,
    Action,
    Romantic,
    Horror,
    Neutral,
}

impl Mood {
    pub fn label(self) -> &'static str {
        match self {
            Self::Calm => "calm",
            Self::Tense => "tense",
            Self::Action => "action",
            Self::Romantic => "romantic",
            Self::Horror => "horror",
            Self::Neutral => "neutral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteBias {
    Warm,
    Cool,
    Neutral,
    Desaturated,
    Vivid,
}

impl PaletteBias {
    pub fn label(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Cool => "cool",
            Self::Neutral => "neutral",
            Self::Desaturated => "desaturated",
            Self::Vivid => "vivid",
        }
    }
}

/// Perceptual scene analysis derived from real page luma measurements.
#[derive(Debug, Clone)]
pub struct SceneAnalysis {
    pub time_of_day: TimeOfDay,
    pub lighting: LightingStyle,
    pub mood: Mood,
    pub palette_bias: PaletteBias,
    /// Measured mean background luma (0..1).
    pub background_luma: f32,
    /// Measured background luma variance.
    pub background_variance: f32,
    /// Scene-adapted background color proposal.
    pub background_color: mangahue_core::ColorRgb,
    pub confidence: f32,
}

pub struct SceneContextAgent {
    id: AgentId,
    config: AgentConfig,
    authority: AgentAuthority,
}

impl SceneContextAgent {
    pub fn new(config: AgentConfig) -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(DecisionType::Palette, 0.20);
        weights.insert(DecisionType::Region, 0.10);
        weights.insert(DecisionType::Character, 0.05);
        weights.insert(DecisionType::Lighting, 0.35);
        weights.insert(DecisionType::Background, 0.20);
        weights.insert(DecisionType::Quality, 0.05);

        Self {
            id: AgentId("scene_context".to_string()),
            config,
            authority: AgentAuthority {
                role: AgentRole::SceneContext,
                weights,
            },
        }
    }

    /// Measure the page from the structure graph and grayscale: background
    /// luma decides time of day and lighting, luma variance decides mood,
    /// and the page's own brightness decides the palette bias. No hashes,
    /// no dice rolls - every decision is grounded in page pixels.
    fn analyze_scene(&self, context: &PageContext) -> SceneAnalysis {
        let background_nodes: Vec<_> = context
            .structure_graph
            .nodes
            .iter()
            .filter(|node| node.kind.semantic() == Some("background"))
            .collect();

        let (mean_luma, variance, _coverage, confidence) = if background_nodes.is_empty() {
            // No detected background regions: fall back to a whole-page
            // measurement with a strict stride budget.
            let (width, height) = context.grayscale.dimensions();
            let total = (width as u64).saturating_mul(height as u64);
            let step = (total / 16384).max(1);
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut n = 0u64;
            for (x, y, pixel) in context.grayscale.enumerate_pixels() {
                let index = y as u64 * width as u64 + x as u64;
                if !index.is_multiple_of(step) {
                    continue;
                }
                let luma = pixel[0] as f64 / 255.0;
                sum += luma;
                sum_sq += luma * luma;
                n += 1;
            }
            let mean = (sum / n.max(1) as f64) as f32;
            (
                mean,
                ((sum_sq / n.max(1) as f64) as f32 - mean * mean).max(0.0),
                1.0,
                0.35,
            )
        } else {
            let mut weighted_luma = 0.0f64;
            let mut weighted_var = 0.0f64;
            let mut total_weight = 0.0f64;
            let mut total_coverage = 0.0f32;
            for node in &background_nodes {
                let stats = context.region_luma_stats(node);
                if stats.samples < 4 {
                    continue;
                }
                let weight = stats.coverage as f64;
                weighted_luma += stats.mean_luma as f64 * weight;
                weighted_var += stats.variance as f64 * weight;
                total_weight += weight;
                total_coverage += stats.coverage;
            }
            if total_weight <= f64::EPSILON {
                (0.5, 0.01, 0.0, 0.4)
            } else {
                (
                    (weighted_luma / total_weight) as f32,
                    (weighted_var / total_weight) as f32,
                    total_coverage.min(1.0),
                    0.55 + 0.25 * total_coverage.min(1.0),
                )
            }
        };

        // Time of day from measured background luminance.
        let time_of_day = match mean_luma {
            l if l < 0.16 => TimeOfDay::Night,
            l if l < 0.28 => TimeOfDay::Dusk,
            l if l < 0.42 => TimeOfDay::Indoor,
            l if l < 0.52 => TimeOfDay::Dawn,
            l if l < 0.66 => TimeOfDay::Morning,
            l if l < 0.78 => TimeOfDay::Noon,
            _ => TimeOfDay::Afternoon,
        };

        // Lighting follows the time-of-day classification; high-variance
        // night scenes read as neon-lit.
        let lighting = match (time_of_day, variance) {
            (TimeOfDay::Night | TimeOfDay::Dusk, v) if v > 0.10 => LightingStyle::Neon,
            (TimeOfDay::Night, _) => LightingStyle::Dramatic,
            (TimeOfDay::Dusk, _) => LightingStyle::Dramatic,
            (TimeOfDay::Indoor, _) => LightingStyle::Artificial,
            (TimeOfDay::Morning | TimeOfDay::Afternoon, _) => LightingStyle::DirectSun,
            (TimeOfDay::Noon, _) => LightingStyle::DirectSun,
            _ => LightingStyle::Overcast,
        };

        // Mood from background contrast (variance): quiet backgrounds are
        // calm, high-variance backgrounds are dramatic, night scenes read
        // tense, and very quiet warm scenes read romantic.
        let mood = match (time_of_day, variance) {
            (TimeOfDay::Night | TimeOfDay::Dusk, v) if v > 0.06 => Mood::Horror,
            (TimeOfDay::Night | TimeOfDay::Dusk, _) => Mood::Tense,
            (TimeOfDay::Morning | TimeOfDay::Afternoon, v) if v < 0.02 => Mood::Romantic,
            (_, v) if v > 0.10 => Mood::Action,
            (_, v) if v > 0.045 => Mood::Neutral,
            (TimeOfDay::Morning | TimeOfDay::Afternoon, _) => Mood::Calm,
            _ => Mood::Neutral,
        };

        // Palette bias from the scene: night scenes cool down, dim indoor
        // scenes desaturate, bright daylight stays neutral, and warm
        // morning/afternoon light warms the palette. High-contrast night
        // scenes may go vivid (neon).
        let palette_bias = match time_of_day {
            TimeOfDay::Night | TimeOfDay::Dusk => {
                if variance > 0.10 {
                    PaletteBias::Vivid
                } else {
                    PaletteBias::Cool
                }
            }
            TimeOfDay::Indoor => PaletteBias::Desaturated,
            TimeOfDay::Morning | TimeOfDay::Afternoon => PaletteBias::Warm,
            TimeOfDay::Noon => PaletteBias::Neutral,
            _ => PaletteBias::Neutral,
        };

        // Background color proposal: tint a neutral gray by the measured
        // luminance, then shift hue by the palette bias.
        let base = (mean_luma * 255.0).clamp(8.0, 248.0) as u8;
        let (r, g, b) = match palette_bias {
            PaletteBias::Warm => (base.saturating_add(10), base.saturating_add(2), base.saturating_sub(8)),
            PaletteBias::Cool => (base.saturating_sub(8), base.saturating_add(2), base.saturating_add(12)),
            PaletteBias::Desaturated => (base, base, base),
            PaletteBias::Neutral => (base.saturating_add(2), base, base.saturating_sub(2)),
            PaletteBias::Vivid => (base.saturating_add(14), base, base.saturating_add(10)),
        };
        let background_color = mangahue_core::ColorRgb::new(r, g, b);

        SceneAnalysis {
            time_of_day,
            lighting,
            mood,
            palette_bias,
            background_luma: mean_luma,
            background_variance: variance,
            background_color,
            confidence: (confidence * self.config.authority_weight).clamp(0.0, 1.0),
        }
    }

    /// Adjust a region color for the scene lighting: night cools and
    /// darkens, warm afternoon light shifts toward gold, indoor light
    /// desaturates.
    fn adapt_color_for_scene(
        &self,
        color: mangahue_core::ColorRgb,
        semantic: &str,
        scene: &SceneAnalysis,
    ) -> mangahue_core::ColorRgb {
        if semantic == "background" || semantic == "effect" {
            return color;
        }
        let (dr, dg, db, desat) = match scene.time_of_day {
            TimeOfDay::Night | TimeOfDay::Dusk => (0, 0, 14, 0.10),
            TimeOfDay::Indoor => (0, 0, 0, 0.18),
            TimeOfDay::Morning | TimeOfDay::Afternoon => (12, 2, -8, 0.0),
            _ => (0, 0, 0, 0.0),
        };
        let mut r = (color.r as i16 + dr).clamp(0, 255) as u8;
        let mut g = (color.g as i16 + dg).clamp(0, 255) as u8;
        let mut b = (color.b as i16 + db).clamp(0, 255) as u8;
        if desat > 0.0 {
            let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
            r = (r as f32 + (luma - r as f32) * desat).round() as u8;
            g = (g as f32 + (luma - g as f32) * desat).round() as u8;
            b = (b as f32 + (luma - b as f32) * desat).round() as u8;
        }
        mangahue_core::ColorRgb::new(r, g, b)
    }
}

impl ColoringAgent for SceneContextAgent {
    fn id(&self) -> AgentId {
        self.id.clone()
    }

    fn name(&self) -> &str {
        "Scene Context"
    }

    fn authority(&self) -> &AgentAuthority {
        &self.authority
    }

    fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let scene = self.analyze_scene(context);

        let mut messages = vec![AgentMessage::ColorProposal {
            agent_id: self.id.clone(),
            target: ProposalTarget::PagePalette {
                mood: scene.mood.label().to_string(),
                time_of_day: scene.time_of_day.label().to_string(),
                lighting: scene.lighting.label().to_string(),
                palette_bias: scene.palette_bias.label().to_string(),
            },
            proposed_colors: Vec::new(),
            confidence: scene.confidence,
            reasoning: format!(
                "measured background luma {:.2} variance {:.3}: {} {} {} {}",
                scene.background_luma,
                scene.background_variance,
                scene.time_of_day.label(),
                scene.lighting.label(),
                scene.mood.label(),
                scene.palette_bias.label()
            ),
        }];

        // Scene-adapted background proposal per background node.
        let background_proposals: Vec<ProposedColor> = context
            .structure_graph
            .nodes
            .iter()
            .filter(|node| node.kind.semantic() == Some("background"))
            .map(|node| ProposedColor {
                region_id: node.id.clone(),
                color: scene.background_color,
                semantic: "background".to_string(),
                confidence: (scene.confidence * node.confidence)
                    .min(0.75)
                    .max(0.25),
            })
            .collect();
        if !background_proposals.is_empty() {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: ProposalTarget::Region {
                    region_id: "scene_backgrounds".to_string(),
                    semantic: "background".to_string(),
                },
                proposed_colors: background_proposals,
                confidence: scene.confidence,
                reasoning: format!(
                    "background painted from measured luma {:.2}",
                    scene.background_luma
                ),
            });
        }

        Ok(messages)
    }

    fn deliberate(
        &self,
        context: &PageContext,
        proposals: &[AgentMessage],
        _round: u32,
    ) -> Result<Vec<AgentMessage>> {
        let scene = self.analyze_scene(context);
        let mut responses = Vec::new();

        for proposal in proposals {
            if let AgentMessage::ColorProposal {
                agent_id,
                proposed_colors,
                ..
            } = proposal
            {
                if agent_id == &self.id {
                    continue;
                }
                // Vote on skin/hair/clothing proposals: the scene decides
                // whether the proposed brightness fits the environment,
                // and counters with a scene-adapted color when it does not.
                for color in proposed_colors {
                    let luma = 0.299 * color.color.r as f32
                        + 0.587 * color.color.g as f32
                        + 0.114 * color.color.b as f32;
                    let bright_ok = match scene.time_of_day {
                        TimeOfDay::Night | TimeOfDay::Dusk => luma < 220.0,
                        _ => luma < 250.0,
                    };
                    if bright_ok {
                        responses.push(AgentMessage::Vote {
                            agent_id: self.id.clone(),
                            proposal_id: color.region_id.clone(),
                            vote: crate::bus::VoteChoice::Approve,
                            weight: self.authority.weights[&DecisionType::Lighting],
                            counter_proposal: None,
                        });
                    } else {
                        // Too bright for this scene: counter with the
                        // scene-adapted variant instead of rejecting.
                        let adapted = self.adapt_color_for_scene(
                            color.color,
                            &color.semantic,
                            &scene,
                        );
                        responses.push(AgentMessage::Vote {
                            agent_id: self.id.clone(),
                            proposal_id: color.region_id.clone(),
                            vote: crate::bus::VoteChoice::Counter,
                            weight: self.authority.weights[&DecisionType::Lighting],
                            counter_proposal: Some(crate::bus::ProposedColor {
                                region_id: color.region_id.clone(),
                                color: adapted,
                                semantic: color.semantic.clone(),
                                confidence: color.confidence * 0.9,
                            }),
                        });
                    }
                }
            }
        }

        Ok(responses)
    }
}
