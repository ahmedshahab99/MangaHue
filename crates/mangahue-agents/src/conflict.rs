use std::collections::HashMap;

use mangahue_core::ColorRgb;

use crate::agent::AgentId;
use crate::bus::{AgentMessage, ProposedColor, VoteChoice};
use crate::config::ConflictResolutionConfig;

pub struct ConflictResolver {
    config: ConflictResolutionConfig,
}

impl ConflictResolver {
    pub fn new(config: ConflictResolutionConfig) -> Self {
        Self { config }
    }

    pub fn resolve(&self, messages: &[AgentMessage]) -> Vec<ProposedColor> {
        // Collect all proposals with their votes
        let mut proposals: HashMap<String, Vec<ProposedColor>> = HashMap::new();
        let mut votes: HashMap<String, Vec<(AgentId, VoteChoice, f32)>> = HashMap::new();
        let conflicts: Vec<_> = messages
            .iter()
            .filter_map(|m| match m {
                AgentMessage::ConflictFlag {
                    region_id,
                    severity,
                    ..
                } => region_id.as_ref().map(|r| (r.clone(), severity)),
                _ => None,
            })
            .collect();

        for message in messages {
            match message {
                AgentMessage::ColorProposal {
                    target,
                    proposed_colors,
                    ..
                } => {
                    let key = match target {
                        crate::bus::ProposalTarget::Region { region_id, .. } => region_id.clone(),
                        crate::bus::ProposalTarget::Character {
                            character_id,
                            component,
                        } => format!("{}.{}", character_id, component),
                        crate::bus::ProposalTarget::PagePalette { .. } => "page".to_string(),
                    };
                    proposals
                        .entry(key.clone())
                        .or_default()
                        .extend(proposed_colors.clone());
                    votes.entry(key).or_default();
                }
                AgentMessage::Vote {
                    proposal_id,
                    vote,
                    weight,
                    agent_id,
                    ..
                } => {
                    votes
                        .entry(proposal_id.clone())
                        .or_default()
                        .push((agent_id.clone(), *vote, *weight));
                    proposals.entry(proposal_id.clone()).or_default();
                }
                _ => {}
            }
        }

        // Resolve each region
        let mut resolved = Vec::new();

        for (region_id, region_proposals) in &proposals {
            if region_proposals.is_empty() {
                continue;
            }

            // Check for conflicts
            let has_conflict = conflicts.iter().any(|(r, _)| r == region_id);

            if has_conflict && self.config.max_auto_corrections > 0 {
                // Use highest confidence proposal for conflicting regions
                if let Some(best) = region_proposals
                    .iter()
                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                {
                    resolved.push(best.clone());
                }
            } else {
                // Weighted average of all proposals
                let region_votes = votes.get(region_id).cloned().unwrap_or_default();

                if region_votes.is_empty() {
                    // No votes, use first proposal
                    if let Some(first) = region_proposals.first() {
                        resolved.push(first.clone());
                    }
                } else {
                    // Weighted blend
                    let total_weight: f32 = region_votes
                        .iter()
                        .filter(|(_, v, _)| matches!(v, VoteChoice::Approve))
                        .map(|(_, _, w)| w)
                        .sum();

                    if total_weight > 0.0 {
                        let mut r_sum = 0.0;
                        let mut g_sum = 0.0;
                        let mut b_sum = 0.0;

                        for proposal in region_proposals {
                            // Use total approve weight for blending
                            let vote_weight = total_weight;

                            if vote_weight > 0.0 {
                                let w = vote_weight * proposal.confidence;
                                r_sum += proposal.color.r as f32 * w;
                                g_sum += proposal.color.g as f32 * w;
                                b_sum += proposal.color.b as f32 * w;
                            }
                        }

                        let final_weight = total_weight.max(1.0);
                        resolved.push(ProposedColor {
                            region_id: region_id.clone(),
                            color: ColorRgb::new(
                                (r_sum / final_weight).round() as u8,
                                (g_sum / final_weight).round() as u8,
                                (b_sum / final_weight).round() as u8,
                            ),
                            semantic: region_proposals[0].semantic.clone(),
                            confidence: total_weight / region_votes.len() as f32,
                        });
                    }
                }
            }
        }

        resolved
    }
}
