use serde::{Deserialize, Serialize};

use crate::bus::AgentMessage;
use crate::PageContext;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    ColorDirector,
    SceneContext,
    CharacterIdentity,
    RegionAnalyst,
    ShadowHighlight,
    Background,
    MoodHarmony,
    QualityChecker,
    Palette,
}

/// Authority weights for different decision types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAuthority {
    pub role: AgentRole,
    pub weights: std::collections::HashMap<DecisionType, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    Palette,
    Region,
    Character,
    Lighting,
    Background,
    Quality,
}

/// Trait that all coloring agents must implement. Agents are pure,
/// deterministic, in-memory deliberators: no I/O, no async, no side
/// effects. The same input always produces the same messages.
pub trait ColoringAgent: Send + Sync {
    /// Unique identifier for this agent.
    fn id(&self) -> AgentId;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Agent's authority weights.
    fn authority(&self) -> &AgentAuthority;

    /// Analyze the page and produce initial proposals.
    fn analyze(&self, context: &PageContext) -> anyhow::Result<Vec<AgentMessage>>;

    /// Review other agents' proposals and vote/respond.
    fn deliberate(
        &self,
        context: &PageContext,
        proposals: &[AgentMessage],
        round: u32,
    ) -> anyhow::Result<Vec<AgentMessage>>;
}
