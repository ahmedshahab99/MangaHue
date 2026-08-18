//! # MangaHue Multi-Agent Coloring System
//!
//! Specialized agents analyze, propose, debate, and converge on coloring decisions
//! before the pixel-level model runs. This adds deliberation to the pipeline.

mod agent;
mod bus;
mod color_director;
mod conflict;
mod plan;
mod scene_context;
mod character_identity;
mod config;

pub use agent::{ColoringAgent, AgentId, AgentRole};
pub use bus::{AgentBus, AgentMessage, ProposalTarget, VoteChoice};
pub use color_director::ColorDirector;
pub use conflict::ConflictResolver;
pub use plan::{ColorPlan, HintAdjustment, RegionColorTarget, LightingPlan};
pub use scene_context::SceneContextAgent;
pub use character_identity::CharacterIdentityAgent;
pub use config::{AgentConfig, AgentSystemConfig};

use anyhow::Result;
use mangahue_intelligence::StructureGraph;
use mangahue_memory::ColorBible;

/// Context passed to all agents for a single page.
#[derive(Clone)]
pub struct PageContext {
    pub page_id: String,
    pub input_path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
    pub structure_graph: StructureGraph,
    pub color_bible: Option<ColorBible>,
    pub previous_page_colors: Vec<PreviousPageColor>,
}

#[derive(Clone)]
pub struct PreviousPageColor {
    pub page_id: String,
    pub region_id: String,
    pub semantic: String,
    pub color: mangahue_core::ColorRgb,
}

/// The main agent system that coordinates deliberation.
pub struct AgentSystem {
    config: AgentSystemConfig,
    agents: Vec<Box<dyn ColoringAgent>>,
    resolver: ConflictResolver,
}

impl AgentSystem {
    pub fn new(config: AgentSystemConfig) -> Self {
        let mut agents: Vec<Box<dyn ColoringAgent>> = Vec::new();

        if config.color_director.enabled {
            agents.push(Box::new(ColorDirector::new(config.color_director.clone())));
        }
        if config.scene_context.enabled {
            agents.push(Box::new(SceneContextAgent::new(config.scene_context.clone())));
        }
        if config.character_identity.enabled {
            agents.push(Box::new(CharacterIdentityAgent::new(
                config.character_identity.clone(),
            )));
        }

        let resolver = ConflictResolver::new(config.conflict_resolution.clone());

        Self {
            config,
            agents,
            resolver,
        }
    }

    /// Run the full agent deliberation pipeline and produce a ColorPlan.
    pub async fn deliberate(&self, context: PageContext) -> Result<ColorPlan> {
        tracing::info!(
            page = %context.page_id,
            agents = self.agents.len(),
            "starting agent deliberation"
        );

        let mut bus = AgentBus::new(self.config.max_deliberation_rounds);

        // Phase 1: Analysis - each agent produces initial proposals
        for agent in &self.agents {
            let messages = agent.analyze(&context).await?;
            bus.broadcast(messages);
        }

        // Phase 2-3: Negotiation rounds
        for round in 0..self.config.max_deliberation_rounds {
            let proposals = bus.get_proposals();
            if proposals.is_empty() {
                break;
            }

            let convergence = bus.convergence_score();
            if convergence >= self.config.convergence_threshold {
                tracing::info!(
                    round,
                    convergence,
                    "early convergence reached"
                );
                break;
            }

            tracing::debug!(
                round,
                proposals = proposals.len(),
                convergence,
                "deliberation round"
            );

            for agent in &self.agents {
                let responses = agent.deliberate(&context, &proposals, round).await?;
                bus.broadcast(responses);
            }
        }

        // Phase 4: Resolution - resolve remaining conflicts
        let resolved = self.resolver.resolve(bus.get_messages());

        // Phase 5: Synthesize ColorPlan
        let plan = ColorPlan::from_resolved(resolved, &context);
        tracing::info!(
            page = %context.page_id,
            adjustments = plan.hint_adjustments.len(),
            "deliberation complete"
        );

        Ok(plan)
    }
}
