use mangahue_core::ColorRgb;
use serde::{Deserialize, Serialize};

use crate::agent::AgentId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessage {
    ColorProposal {
        agent_id: AgentId,
        target: ProposalTarget,
        proposed_colors: Vec<ProposedColor>,
        confidence: f32,
        reasoning: String,
    },
    Vote {
        agent_id: AgentId,
        proposal_id: String,
        vote: VoteChoice,
        weight: f32,
        counter_proposal: Option<ProposedColor>,
    },
    ConflictFlag {
        agent_id: AgentId,
        conflict_type: ConflictType,
        region_id: Option<String>,
        description: String,
        severity: ConflictSeverity,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedColor {
    pub region_id: String,
    pub color: ColorRgb,
    pub semantic: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalTarget {
    /// Page-level scene signal: mood, time of day, lighting, palette bias.
    /// The SceneContextAgent is the authority for this target; other agents
    /// may propose neutral defaults with lower confidence.
    PagePalette {
        mood: String,
        time_of_day: String,
        lighting: String,
        palette_bias: String,
    },
    Region {
        region_id: String,
        semantic: String,
    },
    Character {
        character_id: String,
        component: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteChoice {
    Approve,
    Reject,
    Counter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictType {
    RegionMismatch,
    PaletteDisagreement,
    ConsistencyBreach,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct AgentBus {
    messages: Vec<AgentMessage>,
    round: u32,
}

impl AgentBus {
    pub fn new(_max_rounds: u32) -> Self {
        Self {
            messages: Vec::new(),
            round: 0,
        }
    }

    pub fn broadcast(&mut self, messages: Vec<AgentMessage>) {
        self.messages.extend(messages);
        self.round += 1;
    }

    pub fn get_proposals(&self) -> Vec<AgentMessage> {
        self.messages
            .iter()
            .filter(|m| matches!(m, AgentMessage::ColorProposal { .. }))
            .cloned()
            .collect()
    }

    pub fn get_messages(&self) -> &[AgentMessage] {
        &self.messages
    }

    pub fn convergence_score(&self) -> f32 {
        let votes: Vec<_> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                AgentMessage::Vote { vote, weight, .. } => Some((*vote, *weight)),
                _ => None,
            })
            .collect();

        if votes.is_empty() {
            return 0.0;
        }

        let approve_weight: f32 = votes
            .iter()
            .filter(|(v, _)| matches!(v, VoteChoice::Approve))
            .map(|(_, w)| *w)
            .sum();

        let total_weight: f32 = votes.iter().map(|(_, w)| *w).sum();

        if total_weight <= 0.0 {
            0.0
        } else {
            approve_weight / total_weight
        }
    }
}
