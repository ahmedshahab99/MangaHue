use anyhow::Result;
use mangahue_core::ColorRgb;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ConflictSeverity, ConflictType, ProposedColor, ProposalTarget, VoteChoice};
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

    /// Resolve the artist palette slots: full palettes (>= 9) follow the
    /// documented contract 0-1 paper, 2-5 skin, 6-8 hair. Short palettes
    /// fall back to natural anime defaults instead of reading an arbitrary
    /// hue out of an unrelated slot.
    fn palette_slots(&self, context: &PageContext) -> (ColorRgb, ColorRgb, ColorRgb) {
        let skin = context
            .style_palette
            .get(2)
            .copied()
            .filter(|_| context.style_palette.len() >= 9)
            .unwrap_or(ColorRgb::new(242, 208, 182));
        let hair = context
            .style_palette
            .get(6)
            .copied()
            .filter(|_| context.style_palette.len() >= 9)
            .unwrap_or(ColorRgb::new(38, 28, 34));
        let eyes = context
            .style_palette
            .get(7)
            .copied()
            .filter(|_| context.style_palette.len() >= 9)
            .map(|c| ColorRgb::new(
                (c.r as u16 * 2 / 3) as u8,
                (c.g as u16 * 2 / 3) as u8,
                (c.b as u16 * 2 / 3) as u8,
            ))
            .unwrap_or(ColorRgb::new(46, 40, 54));
        (skin, hair, eyes)
    }

    /// Propose a canonical color per semantic node. The Color Bible is the
    /// supreme authority; without an entry the director falls back to the
    /// artist palette slots, and to natural defaults only when the palette
    /// is too short to carry a real slot.
    fn propose_node_color(&self, context: &PageContext, node: &mangahue_intelligence::StructureNode) -> Option<ProposedColor> {
        let semantic = node.kind.semantic()?;
        let (skin, hair, eyes) = self.palette_slots(context);

        let color = if let Some(bible) = &context.color_bible {
            let ctx = mangahue_memory::ColorContext {
                page_id: Some(&context.page_id),
                chapter_id: context.structure_graph.chapter_id.as_deref(),
                character_id: node.character_id.as_deref(),
                location_id: node.location_id.as_deref(),
                object_id: node.object_id.as_deref(),
                region_id: Some(&node.id),
                semantic: Some(semantic),
            };
            bible.resolve_color(&ctx).map(|entry| entry.color)
        } else {
            None
        };

        let color = match (semantic, color) {
            (_, Some(canonical)) => canonical,
            ("skin", None) => skin,
            ("hair", None) => hair,
            ("eyes", None) => eyes,
            ("clothing", None) => {
                // Clothing follows the hair family so characters read as
                // one designed palette, lightened for fabric contrast.
                ColorRgb::new(
                    (hair.r as u16 + 110).min(255) as u8,
                    (hair.g as u16 + 70).min(255) as u8,
                    (hair.b as u16 + 60).min(255) as u8,
                )
            }
            ("object", None) => ColorRgb::new(120, 96, 72),
            _ => return None,
        };

        Some(ProposedColor {
            region_id: node.id.clone(),
            color,
            semantic: semantic.to_string(),
            confidence: (node.confidence * self.config.authority_weight).clamp(0.25, 0.9),
        })
    }
}

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

    fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let proposals: Vec<ProposedColor> = context
            .structure_graph
            .nodes
            .iter()
            .filter_map(|node| self.propose_node_color(context, node))
            .collect();

        let mut messages = vec![AgentMessage::ColorProposal {
            agent_id: self.id.clone(),
            target: ProposalTarget::PagePalette {
                mood: "neutral".to_string(),
                time_of_day: "day".to_string(),
                lighting: "unknown".to_string(),
                palette_bias: "neutral".to_string(),
            },
            proposed_colors: proposals.clone(),
            confidence: 0.45,
            reasoning: "artist palette and Color Bible canonical colors".to_string(),
        }];

        for proposal in proposals {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: ProposalTarget::Region {
                    region_id: proposal.region_id.clone(),
                    semantic: proposal.semantic.clone(),
                },
                proposed_colors: vec![proposal],
                confidence: 0.6,
                reasoning: "director region color from palette slots".to_string(),
            });
        }

        Ok(messages)
    }

    fn deliberate(
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
                if agent_id == &self.id {
                    continue;
                }
                for color in proposed_colors {
                    let luma = (0.299 * color.color.r as f32
                        + 0.587 * color.color.g as f32
                        + 0.114 * color.color.b as f32)
                        / 255.0;

                    // Reject washed-out skin and white-washed characters.
                    if color.semantic == "skin" && luma > 0.92 {
                        responses.push(AgentMessage::ConflictFlag {
                            agent_id: self.id.clone(),
                            conflict_type: ConflictType::PaletteDisagreement,
                            region_id: Some(color.region_id.clone()),
                            description: "skin proposal is washed out".to_string(),
                            severity: ConflictSeverity::Medium,
                        });
                        continue;
                    }
                    // Accept materially plausible colors.
                    if luma > 0.06 && luma < 0.96 {
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
