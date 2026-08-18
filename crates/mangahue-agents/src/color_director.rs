use anyhow::Result;
use mangahue_core::ColorRgb;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ConflictSeverity, ConflictType, ProposedColor, VoteChoice};
use crate::config::AgentConfig;
use crate::PageContext;

pub struct ColorDirector {
    id: AgentId,
    config: AgentConfig,
    authority: AgentAuthority,
}

impl ColorDirector {
    pub fn new(config: AgentConfig) -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(DecisionType::Palette, 0.40);
        weights.insert(DecisionType::Region, 0.20);
        weights.insert(DecisionType::Character, 0.15);
        weights.insert(DecisionType::Lighting, 0.10);
        weights.insert(DecisionType::Background, 0.10);
        weights.insert(DecisionType::Quality, 0.05);

        Self {
            id: AgentId("color_director".to_string()),
            config,
            authority: AgentAuthority {
                role: AgentRole::ColorDirector,
                weights,
            },
        }
    }

    fn suggest_palette(&self, context: &PageContext) -> Vec<ProposedColor> {
        let mut proposals = Vec::new();

        // Analyze structure graph for semantic regions
        for node in &context.structure_graph.nodes {
            if let Some(semantic) = node.kind.semantic() {
                let color = match semantic {
                    "skin" => ColorRgb::new(240, 200, 168), // Natural skin
                    "hair" => {
                        // Hair color varies by character
                        if node.character_id.is_some() {
                            ColorRgb::new(60, 45, 35) // Default dark hair
                        } else {
                            ColorRgb::new(80, 60, 45)
                        }
                    }
                    "clothing" => ColorRgb::new(70, 80, 120), // Default blue-ish
                    "background" => ColorRgb::new(180, 190, 200), // Light background
                    _ => continue,
                };

                proposals.push(ProposedColor {
                    region_id: node.id.clone(),
                    color,
                    semantic: semantic.to_string(),
                    confidence: node.confidence * self.config.authority_weight,
                });
            }
        }

        proposals
    }
}

#[async_trait::async_trait]
impl ColoringAgent for ColorDirector {
    fn id(&self) -> AgentId {
        self.id.clone()
    }

    fn name(&self) -> &str {
        "Color Director"
    }

    fn authority(&self) -> &AgentAuthority {
        &self.authority
    }

    async fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let proposals = self.suggest_palette(context);

        let mut messages = vec![AgentMessage::ColorProposal {
            agent_id: self.id.clone(),
            target: crate::bus::ProposalTarget::PagePalette {
                mood: "neutral".to_string(),
                time_of_day: "day".to_string(),
            },
            proposed_colors: proposals.clone(),
            confidence: 0.75,
            reasoning: "Base palette from semantic analysis".to_string(),
        }];

        // Also propose individual region colors
        for proposal in proposals {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: crate::bus::ProposalTarget::Region {
                    region_id: proposal.region_id.clone(),
                    semantic: proposal.semantic.clone(),
                },
                proposed_colors: vec![proposal],
                confidence: 0.70,
                reasoning: "Region color from semantic class".to_string(),
            });
        }

        Ok(messages)
    }

    async fn deliberate(
        &self,
        _context: &PageContext,
        proposals: &[AgentMessage],
        _round: u32,
    ) -> Result<Vec<AgentMessage>> {
        let mut responses = Vec::new();

        for proposal in proposals {
            if let AgentMessage::ColorProposal {
                agent_id,
                proposed_colors,
                ..
            } = proposal
            {
                // Don't review own proposals
                if agent_id == &self.id {
                    continue;
                }

                // Check if proposed colors make sense
                for color in proposed_colors {
                    let luma = (color.color.r as f32 * 0.299
                        + color.color.g as f32 * 0.587
                        + color.color.b as f32 * 0.114)
                        / 255.0;

                    // Flag overly bright skin tones
                    if color.semantic == "skin" && luma > 0.9 {
                        responses.push(AgentMessage::ConflictFlag {
                            agent_id: self.id.clone(),
                            conflict_type: ConflictType::PaletteDisagreement,
                            region_id: Some(color.region_id.clone()),
                            description: "Skin tone too bright, may look washed out".to_string(),
                            severity: ConflictSeverity::Medium,
                        });
                    }

                    // Approve reasonable colors
                    if (0.1..0.85).contains(&luma) || color.semantic == "background" {
                        responses.push(AgentMessage::Vote {
                            agent_id: self.id.clone(),
                            proposal_id: color.region_id.clone(),
                            vote: VoteChoice::Approve,
                            weight: self.authority.weights[&DecisionType::Palette],
                            counter_proposal: None,
                        });
                    }
                }
            }
        }

        Ok(responses)
    }
}
