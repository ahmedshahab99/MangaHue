use anyhow::Result;
use mangahue_core::ColorRgb;
use mangahue_intelligence::StructureNodeKind;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ProposedColor};
use crate::config::AgentConfig;
use crate::PageContext;

#[derive(Debug, Clone)]
pub struct SceneAnalysis {
    pub time_of_day: TimeOfDay,
    pub lighting: LightingStyle,
    pub mood: Mood,
    pub palette_bias: PaletteBias,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeOfDay {
    Dawn,
    Morning,
    Noon,
    Afternoon,
    Dusk,
    Night,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingStyle {
    DirectSun,
    Overcast,
    Artificial,
    Dramatic,
    Neon,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteBias {
    Warm,
    Cool,
    Neutral,
    Desaturated,
    Vivid,
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

    fn analyze_scene(&self, context: &PageContext) -> SceneAnalysis {
        // Analyze luma distribution for time of day
        let graph = &context.structure_graph;
        let bg_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.kind.semantic() == Some("background"))
            .collect();

        let avg_confidence: f32 = if bg_nodes.is_empty() {
            0.5
        } else {
            bg_nodes.iter().map(|n| n.confidence).sum::<f32>() / bg_nodes.len() as f32
        };

        // Simple heuristic: high background confidence with many nodes = complex scene
        let node_count = graph.nodes.len();
        let character_count = graph
            .nodes
            .iter()
            .filter(|n| n.kind == StructureNodeKind::Character)
            .count();

        // Determine time of day from structure
        let time_of_day = if node_count > 20 {
            TimeOfDay::Night // Dense panels often = dramatic/night
        } else if character_count > 2 {
            TimeOfDay::Afternoon // Many characters = social scene
        } else {
            TimeOfDay::Noon // Default
        };

        // Determine mood from panel density
        let mood = if node_count > 15 {
            Mood::Tense
        } else if character_count <= 1 {
            Mood::Calm
        } else {
            Mood::Neutral
        };

        // Determine palette bias
        let palette_bias = match time_of_day {
            TimeOfDay::Night | TimeOfDay::Dusk => PaletteBias::Cool,
            TimeOfDay::Dawn | TimeOfDay::Morning => PaletteBias::Warm,
            TimeOfDay::Noon => PaletteBias::Neutral,
            TimeOfDay::Afternoon => PaletteBias::Warm,
        };

        SceneAnalysis {
            time_of_day,
            lighting: LightingStyle::Overcast, // Default
            mood,
            palette_bias,
            confidence: avg_confidence,
        }
    }

    fn adjust_colors_for_scene(
        &self,
        colors: Vec<ProposedColor>,
        scene: &SceneAnalysis,
    ) -> Vec<ProposedColor> {
        colors
            .into_iter()
            .map(|mut c| {
                let shift = match scene.palette_bias {
                    PaletteBias::Warm => (10, -5, -10),
                    PaletteBias::Cool => (-10, -5, 10),
                    PaletteBias::Desaturated => (-20, -20, -20),
                    PaletteBias::Vivid => (15, 15, 15),
                    PaletteBias::Neutral => (0, 0, 0),
                };

                c.color = ColorRgb::new(
                    (c.color.r as i16 + shift.0).clamp(0, 255) as u8,
                    (c.color.g as i16 + shift.1).clamp(0, 255) as u8,
                    (c.color.b as i16 + shift.2).clamp(0, 255) as u8,
                );
                c.confidence *= scene.confidence;
                c
            })
            .collect()
    }
}

#[async_trait::async_trait]
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

    async fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let scene = self.analyze_scene(context);

        // Get base proposals from structure
        let base_proposals: Vec<ProposedColor> = context
            .structure_graph
            .nodes
            .iter()
            .filter(|n| n.kind.semantic().is_some())
            .map(|n| ProposedColor {
                region_id: n.id.clone(),
                color: mangahue_core::ColorRgb::new(128, 128, 128),
                semantic: n.kind.semantic().unwrap_or("unknown").to_string(),
                confidence: n.confidence,
            })
            .collect();

        let adjusted = self.adjust_colors_for_scene(base_proposals, &scene);

        Ok(vec![AgentMessage::ColorProposal {
            agent_id: self.id.clone(),
            target: crate::bus::ProposalTarget::PagePalette {
                mood: format!("{:?}", scene.mood).to_lowercase(),
                time_of_day: format!("{:?}", scene.time_of_day).to_lowercase(),
            },
            proposed_colors: adjusted,
            confidence: scene.confidence,
            reasoning: format!(
                "Scene: {:?}, Lighting: {:?}, Mood: {:?}, Bias: {:?}",
                scene.time_of_day, scene.lighting, scene.mood, scene.palette_bias
            ),
        }])
    }

    async fn deliberate(
        &self,
        _context: &PageContext,
        _proposals: &[AgentMessage],
        _round: u32,
    ) -> Result<Vec<AgentMessage>> {
        // Scene agent mostly provides initial analysis, minimal deliberation
        Ok(Vec::new())
    }
}
