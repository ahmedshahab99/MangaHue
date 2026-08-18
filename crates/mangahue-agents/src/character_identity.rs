use anyhow::Result;
use mangahue_brain::PaletteMemoryRole;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ConflictSeverity, ConflictType, ProposedColor, ProposalTarget, VoteChoice};
use crate::config::AgentConfig;
use crate::PageContext;

pub struct CharacterIdentityAgent {
    id: AgentId,
    config: AgentConfig,
    authority: AgentAuthority,
}

impl CharacterIdentityAgent {
    pub fn new(config: AgentConfig) -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(DecisionType::Palette, 0.15);
        weights.insert(DecisionType::Region, 0.10);
        weights.insert(DecisionType::Character, 0.40);
        weights.insert(DecisionType::Lighting, 0.05);
        weights.insert(DecisionType::Background, 0.05);
        weights.insert(DecisionType::Quality, 0.05);

        Self {
            id: AgentId("character_identity".to_string()),
            config,
            authority: AgentAuthority {
                role: AgentRole::CharacterIdentity,
                weights,
            },
        }
    }

    fn semantic_role(semantic: &str) -> Option<PaletteMemoryRole> {
        match semantic {
            "skin" => Some(PaletteMemoryRole::Skin),
            "hair" => Some(PaletteMemoryRole::Hair),
            "eyes" => Some(PaletteMemoryRole::Eyes),
            "clothing" => Some(PaletteMemoryRole::Outfit),
            "background" => Some(PaletteMemoryRole::Background),
            _ => None,
        }
    }

    /// Enforce canonical colors from the Color Bible (character-scoped,
    /// human-approved identity) and from the chapter palette memory
    /// (role-scoped canonical colors learned from previous pages).
    fn canonical_proposals(&self, context: &PageContext) -> Vec<ProposedColor> {
        let mut proposals = Vec::new();

        for node in &context.structure_graph.nodes {
            let Some(semantic) = node.kind.semantic() else {
                continue;
            };

            // 1. Color Bible: strongest authority, character-scoped.
            if let Some(bible) = &context.color_bible {
                let ctx = mangahue_memory::ColorContext {
                    page_id: Some(&context.page_id),
                    chapter_id: context.structure_graph.chapter_id.as_deref(),
                    character_id: node.character_id.as_deref(),
                    location_id: node.location_id.as_deref(),
                    object_id: node.object_id.as_deref(),
                    region_id: Some(&node.id),
                    semantic: Some(semantic),
                };
                if let Some(entry) = bible.resolve_color(&ctx) {
                    proposals.push(ProposedColor {
                        region_id: node.id.clone(),
                        color: entry.color,
                        semantic: semantic.to_string(),
                        confidence: (entry.confidence * node.confidence
                            * self.config.authority_weight)
                            .clamp(0.5, 0.98),
                    });
                    continue;
                }
            }

            // 2. Palette memory: role canonical colors from earlier pages.
            if let Some(role) = Self::semantic_role(semantic) {
                if let Some(entry) = context
                    .palette_memory
                    .iter()
                    .filter(|entry| entry.role == role && entry.locked)
                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                {
                    proposals.push(ProposedColor {
                        region_id: node.id.clone(),
                        color: entry.color,
                        semantic: semantic.to_string(),
                        confidence: (entry.confidence * node.confidence).clamp(0.35, 0.9),
                    });
                }
            }
        }

        proposals
    }

    /// Enforce cross-page consistency from previously colored pages.
    fn cross_page_proposals(&self, context: &PageContext) -> Vec<ProposedColor> {
        let mut proposals = Vec::new();
        for prev in &context.previous_page_colors {
            for node in &context.structure_graph.nodes {
                let Some(semantic) = node.kind.semantic() else {
                    continue;
                };
                if semantic != prev.semantic {
                    continue;
                }
                // Character-scoped previous color requires a character
                // match; role-scoped entries apply to any matching node.
                if let Some(prev_character) = prev.region_id.split('.').next() {
                    if prev_character.contains("character") {
                        if node.character_id.as_deref() != Some(prev_character) {
                            continue;
                        }
                    }
                }
                proposals.push(ProposedColor {
                    region_id: node.id.clone(),
                    color: prev.color,
                    semantic: semantic.to_string(),
                    confidence: (prev.confidence * 0.9).clamp(0.3, 0.85),
                });
            }
        }
        proposals
    }
}

impl ColoringAgent for CharacterIdentityAgent {
    fn id(&self) -> AgentId {
        self.id.clone()
    }

    fn name(&self) -> &str {
        "Character Identity"
    }

    fn authority(&self) -> &AgentAuthority {
        &self.authority
    }

    fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let mut messages = Vec::new();

        let canonical = self.canonical_proposals(context);
        if !canonical.is_empty() {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: ProposalTarget::PagePalette {
                    mood: "canonical".to_string(),
                    time_of_day: "any".to_string(),
                    lighting: "any".to_string(),
                    palette_bias: "canonical".to_string(),
                },
                proposed_colors: canonical,
                confidence: 0.92,
                reasoning: "Color Bible and palette memory canonical colors".to_string(),
            });
        }

        let cross_page = self.cross_page_proposals(context);
        if !cross_page.is_empty() {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: ProposalTarget::PagePalette {
                    mood: "canonical".to_string(),
                    time_of_day: "any".to_string(),
                    lighting: "any".to_string(),
                    palette_bias: "canonical".to_string(),
                },
                proposed_colors: cross_page,
                confidence: 0.8,
                reasoning: "cross-page character color consistency".to_string(),
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
                    // Character materials are identity-critical: any
                    // disagreement here is a consistency breach, and this
                    // agent owns the highest character authority.
                    if matches!(color.semantic.as_str(), "skin" | "hair" | "eyes") {
                        responses.push(AgentMessage::Vote {
                            agent_id: self.id.clone(),
                            proposal_id: color.region_id.clone(),
                            vote: VoteChoice::Approve,
                            weight: self.authority.weights[&DecisionType::Character],
                            counter_proposal: None,
                        });
                    } else if color.semantic == "clothing" && color.confidence < 0.3 {
                        responses.push(AgentMessage::ConflictFlag {
                            agent_id: self.id.clone(),
                            conflict_type: ConflictType::ConsistencyBreach,
                            region_id: Some(color.region_id.clone()),
                            description: "low-confidence outfit color may break character identity"
                                .to_string(),
                            severity: ConflictSeverity::Low,
                        });
                    }
                }
            }
        }

        Ok(responses)
    }
}
