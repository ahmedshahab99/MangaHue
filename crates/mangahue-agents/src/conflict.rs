use std::collections::HashMap;

use mangahue_core::ColorRgb;

use crate::agent::AgentId;
use crate::bus::{AgentMessage, ProposedColor, ProposalTarget, VoteChoice};
use crate::config::ConflictResolutionConfig;
use crate::plan::SceneSignal;

/// Outcome of conflict resolution: the agreed region colors plus the
/// highest-confidence page scene signal.
#[derive(Debug, Clone, Default)]
pub struct Resolution {
    pub colors: Vec<ProposedColor>,
    pub scene: SceneSignal,
}

pub struct ConflictResolver {
    config: ConflictResolutionConfig,
}

impl ConflictResolver {
    pub fn new(config: ConflictResolutionConfig) -> Self {
        Self { config }
    }

    pub fn resolve(&self, messages: &[AgentMessage]) -> Resolution {
        let mut proposals: HashMap<String, Vec<ProposedColor>> = HashMap::new();
        let mut votes: HashMap<String, Vec<(AgentId, VoteChoice, f32)>> = HashMap::new();
        let mut conflicts: HashMap<String, u8> = HashMap::new();
        let mut scene = SceneSignal::default();

        // Scene signal: the SceneContext agent's PagePalette proposal is
        // authoritative; take the highest-confidence one and remember the
        // confidence-weighted mood/time-of-day.
        let mut scene_confidence = -1.0f32;
        for message in messages {
            if let AgentMessage::ColorProposal {
                target: ProposalTarget::PagePalette {
                    mood,
                    time_of_day,
                    lighting,
                    palette_bias,
                },
                confidence,
                ..
            } = message
            {
                if *confidence > scene_confidence {
                    scene_confidence = *confidence;
                    scene.mood = mood.clone();
                    scene.time_of_day = time_of_day.clone();
                    scene.lighting = lighting.clone();
                    scene.palette_bias = palette_bias.clone();
                }
            }
        }
        scene.confidence = scene_confidence.max(0.0);

        for message in messages {
            match message {
                AgentMessage::ColorProposal {
                    target,
                    proposed_colors,
                    ..
                } => {
                    let key = match target {
                        ProposalTarget::Region { region_id, .. } => region_id.clone(),
                        ProposalTarget::Character {
                            character_id,
                            component,
                        } => format!("{character_id}.{component}"),
                        ProposalTarget::PagePalette { .. } => "page".to_string(),
                    };
                    proposals
                        .entry(key.clone())
                        .or_default()
                        .extend(proposed_colors.iter().cloned());
                    votes.entry(key).or_default();
                }
                AgentMessage::Vote {
                    proposal_id,
                    vote,
                    weight,
                    agent_id,
                    counter_proposal,
                } => {
                    votes
                        .entry(proposal_id.clone())
                        .or_default()
                        .push((agent_id.clone(), *vote, *weight));
                    // Counter proposals are alternatives for the same
                    // region: include them as candidates so the weighted
                    // blend can move toward the scene-adapted color.
                    if let Some(counter) = counter_proposal {
                        proposals
                            .entry(proposal_id.clone())
                            .or_default()
                            .push(counter.clone());
                    }
                    proposals.entry(proposal_id.clone()).or_default();
                }
                AgentMessage::ConflictFlag {
                    region_id, severity, ..
                } => {
                    if let Some(region_id) = region_id {
                        *conflicts.entry(region_id.clone()).or_insert(0) += match severity {
                            crate::bus::ConflictSeverity::Low => 1,
                            crate::bus::ConflictSeverity::Medium => 2,
                            crate::bus::ConflictSeverity::High => 3,
                            crate::bus::ConflictSeverity::Critical => 4,
                        };
                    }
                }
            }
        }

        let mut resolved = Vec::new();

        for (region_id, region_proposals) in &proposals {
            if region_proposals.is_empty() || region_id == "page" {
                continue;
            }

            let region_votes = votes.get(region_id).cloned().unwrap_or_default();
            let approve_weight: f32 = region_votes
                .iter()
                .filter(|(_, v, _)| matches!(v, VoteChoice::Approve))
                .map(|(_, _, w)| *w)
                .sum();
            let reject_weight: f32 = region_votes
                .iter()
                .filter(|(_, v, _)| matches!(v, VoteChoice::Reject))
                .map(|(_, _, w)| *w)
                .sum();

            let conflict_score = conflicts.get(region_id).copied().unwrap_or(0);

            // Rejected proposals are dropped unless the Color Bible or a
            // high-confidence canonical source stands behind them.
            let candidates: Vec<&ProposedColor> = region_proposals
                .iter()
                .filter(|p| {
                    reject_weight <= 0.0 || p.confidence > 0.8 || conflict_score == 0
                })
                .collect();

            if candidates.is_empty() {
                continue;
            }

            if conflict_score >= 2 && self.config.max_auto_corrections > 0 {
                // Conflicted region: highest-confidence candidate wins
                // instead of blending two incompatible colors.
                if let Some(best) = candidates
                    .iter()
                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                {
                    resolved.push((*best).clone());
                }
                continue;
            }

            if approve_weight > 0.0 {
                // Weighted blend of approved candidates.
                let mut r_sum = 0.0f64;
                let mut g_sum = 0.0f64;
                let mut b_sum = 0.0f64;
                let mut weight_sum = 0.0f64;
                for proposal in &candidates {
                    let w = (approve_weight * proposal.confidence) as f64;
                    r_sum += proposal.color.r as f64 * w;
                    g_sum += proposal.color.g as f64 * w;
                    b_sum += proposal.color.b as f64 * w;
                    weight_sum += w;
                }
                let total = (weight_sum * candidates.len() as f64).max(1.0);
                resolved.push(ProposedColor {
                    region_id: region_id.clone(),
                    color: ColorRgb::new(
                        (r_sum / total).round() as u8,
                        (g_sum / total).round() as u8,
                        (b_sum / total).round() as u8,
                    ),
                    semantic: candidates[0].semantic.clone(),
                    confidence: (approve_weight / (approve_weight + reject_weight + 0.25))
                        .clamp(0.2, 0.95),
                });
            } else {
                // No votes yet: keep the highest-confidence candidate.
                if let Some(best) = candidates
                    .iter()
                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                {
                    resolved.push((*best).clone());
                }
            }
        }

        Resolution {
            colors: resolved,
            scene,
        }
    }
}
