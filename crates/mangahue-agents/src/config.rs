use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSystemConfig {
    pub enabled: bool,
    pub max_deliberation_rounds: u32,
    pub convergence_threshold: f32,
    pub color_director: AgentConfig,
    pub scene_context: AgentConfig,
    pub character_identity: AgentConfig,
    pub conflict_resolution: ConflictResolutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub enabled: bool,
    pub authority_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionConfig {
    pub bible_override: bool,
    pub user_hint_override: bool,
    pub default_strategy: String,
    pub max_auto_corrections: usize,
}

impl Default for AgentSystemConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_deliberation_rounds: 3,
            convergence_threshold: 0.85,
            color_director: AgentConfig {
                enabled: true,
                authority_weight: 1.0,
            },
            scene_context: AgentConfig {
                enabled: true,
                authority_weight: 0.8,
            },
            character_identity: AgentConfig {
                enabled: true,
                authority_weight: 0.9,
            },
            conflict_resolution: ConflictResolutionConfig {
                bible_override: true,
                user_hint_override: true,
                default_strategy: "weighted_vote".to_string(),
                max_auto_corrections: 5,
            },
        }
    }
}
