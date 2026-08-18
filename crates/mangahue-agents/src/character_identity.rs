use anyhow::Result;

use crate::agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
use crate::bus::{AgentMessage, ProposedColor};
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

    fn enforce_canonical_colors(&self, context: &PageContext) -> Vec<ProposedColor> {
        let mut proposals = Vec::new();

        if let Some(ref bible) = context.color_bible {
            for node in &context.structure_graph.nodes {
                if let Some(character_id) = &node.character_id {
                    if let Some(semantic) = node.kind.semantic() {
                        // Try to resolve canonical color from bible
                        let ctx = mangahue_memory::ColorContext {
                            page_id: Some(&context.page_id),
                            chapter_id: None,
                            character_id: Some(character_id),
                            location_id: None,
                            object_id: None,
                            region_id: Some(&node.id),
                            semantic: Some(semantic),
                        };

                        if let Some(entry) = bible.resolve_color(&ctx) {
                            proposals.push(ProposedColor {
                                region_id: node.id.clone(),
                                color: entry.color,
                                semantic: semantic.to_string(),
                                confidence: entry.confidence * self.config.authority_weight,
                            });
                        }
                    }
                }
            }
        }

        proposals
    }

    fn check_cross_page_consistency(&self, context: &PageContext) -> Vec<AgentMessage> {
        let mut messages = Vec::new();

        // Check if current page characters match previous pages
        for prev in &context.previous_page_colors {
            for node in &context.structure_graph.nodes {
                if let Some(character_id) = &node.character_id {
                    if node.kind.semantic() == Some(&prev.semantic) {
                        // Same character, same semantic - should have same color
                        messages.push(AgentMessage::ColorProposal {
                            agent_id: self.id.clone(),
                            target: crate::bus::ProposalTarget::Character {
                                character_id: character_id.clone(),
                                component: prev.semantic.clone(),
                            },
                            proposed_colors: vec![ProposedColor {
                                region_id: node.id.clone(),
                                color: prev.color,
                                semantic: prev.semantic.clone(),
                                confidence: 0.85,
                            }],
                            confidence: 0.85,
                            reasoning: format!(
                                "Consistency with page {} for {}.{}",
                                prev.page_id, character_id, prev.semantic
                            ),
                        });
                    }
                }
            }
        }

        messages
    }
}

#[async_trait::async_trait]
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

    async fn analyze(&self, context: &PageContext) -> Result<Vec<AgentMessage>> {
        let mut messages = Vec::new();

        // Enforce canonical colors from Color Bible
        let canonical = self.enforce_canonical_colors(context);
        if !canonical.is_empty() {
            messages.push(AgentMessage::ColorProposal {
                agent_id: self.id.clone(),
                target: crate::bus::ProposalTarget::PagePalette {
                    mood: "canonical".to_string(),
                    time_of_day: "any".to_string(),
                },
                proposed_colors: canonical,
                confidence: 0.90,
                reasoning: "Canonical colors from Color Bible".to_string(),
            });
        }

        // Check cross-page consistency
        let consistency = self.check_cross_page_consistency(context);
        messages.extend(consistency);

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
                if agent_id == &self.id {
                    continue;
                }

                // Check if any proposal conflicts with character identity
                for color in proposed_colors {
                    if color.semantic == "skin" || color.semantic == "hair" {
                        // Character Identity agent has high authority for these
                        let weight = self.authority.weights[&DecisionType::Character];

                        if weight > 0.3 {
                            responses.push(crate::bus::AgentMessage::Vote {
                                agent_id: self.id.clone(),
                                proposal_id: color.region_id.clone(),
                                vote: crate::bus::VoteChoice::Approve,
                                weight,
                                counter_proposal: None,
                            });
                        }
                    }
                }
            }
        }

        Ok(responses)
    }
}
