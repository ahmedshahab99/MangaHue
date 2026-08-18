//! # MangaHue Multi-Agent Coloring System
//!
//! Specialized agents analyze, propose, debate, and converge on coloring decisions
//! before the pixel-level model runs. This adds deliberation to the pipeline.

mod agent;
mod bus;
mod character_identity;
mod color_director;
mod conflict;
mod config;
mod plan;
mod scene_context;

pub use agent::{AgentAuthority, AgentId, AgentRole, ColoringAgent, DecisionType};
pub use bus::{
    AgentBus, AgentMessage, ConflictSeverity, ConflictType, ProposedColor, ProposalTarget, VoteChoice,
};
pub use character_identity::CharacterIdentityAgent;
pub use color_director::ColorDirector;
pub use conflict::{ConflictResolver, Resolution};
pub use config::{AgentConfig, AgentSystemConfig, ConflictResolutionConfig};
pub use plan::{
    ColorPlan, HintAdjustment, LightingPlan, RegionColorTarget, SceneSignal,
};
pub use scene_context::SceneContextAgent;

use anyhow::Result;
use mangahue_brain::PaletteMemoryEntry;
use mangahue_core::ColorRgb;
use mangahue_intelligence::{StructureGraph, StructureNode};
use mangahue_memory::ColorBible;
use std::path::PathBuf;

/// Luma statistics for one structure-graph region, measured from the page
/// grayscale. This is the perceptual input the agents reason over.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegionLumaStats {
    /// Mean normalized luma (0..1) inside the region, excluding near-ink pixels.
    pub mean_luma: f32,
    /// Luma variance inside the region.
    pub variance: f32,
    /// Fraction of the page covered by the region.
    pub coverage: f32,
    /// Normalized vertical centroid (0 = top, 1 = bottom).
    pub centroid_y: f32,
    /// Number of sampled pixels.
    pub samples: u32,
}

/// Context passed to all agents for a single page.
#[derive(Clone)]
pub struct PageContext {
    pub page_id: String,
    pub input_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub structure_graph: StructureGraph,
    pub color_bible: Option<ColorBible>,
    /// The scene-adapted artist palette (style palette after chapter accents
    /// and scene re-weighting). Slots >= 9 follow the documented contract:
    /// 0-1 paper, 2-5 skin, 6-8 hair.
    pub style_palette: Vec<ColorRgb>,
    /// Original grayscale page for perceptual region measurements.
    pub grayscale: image::GrayImage,
    /// Canonical chapter palette memory learned from previous pages.
    pub palette_memory: Vec<PaletteMemoryEntry>,
    /// Previous page canonical colors for cross-page consistency.
    pub previous_page_colors: Vec<PreviousPageColor>,
}

#[derive(Clone)]
pub struct PreviousPageColor {
    pub page_id: String,
    pub region_id: String,
    pub semantic: String,
    pub color: ColorRgb,
    pub confidence: f32,
}

impl PageContext {
    /// Measure luma statistics for a structure node from the page grayscale.
    /// Uses the node's exact run-length mask when available; otherwise the
    /// normalized bounding rectangle.
    pub fn region_luma_stats(&self, node: &StructureNode) -> RegionLumaStats {
        let (width, height) = self.grayscale.dimensions();
        if width == 0 || height == 0 {
            return RegionLumaStats::default();
        }
        let scale_x = width as f32 / self.width.max(1) as f32;
        let scale_y = height as f32 / self.height.max(1) as f32;

        let points: Vec<(u32, u32)> = if node.mask_runs.is_empty() {
            let x0 = (node.normalized_bounds.x * width as f32).floor().max(0.0) as u32;
            let y0 = (node.normalized_bounds.y * height as f32).floor().max(0.0) as u32;
            let x1 = ((node.normalized_bounds.x + node.normalized_bounds.width) * width as f32)
                .ceil()
                .min(width as f32) as u32;
            let y1 = ((node.normalized_bounds.y + node.normalized_bounds.height) * height as f32)
                .ceil()
                .min(height as f32) as u32;
            let area = (x1.saturating_sub(x0)) as u64 * (y1.saturating_sub(y0)) as u64;
            let stride = ((area as f32 / 4096.0).sqrt().floor() as u32).max(1);
            let mut out = Vec::new();
            for y in (y0..y1).step_by(stride as usize) {
                for x in (x0..x1).step_by(stride as usize) {
                    out.push((x, y));
                }
            }
            out
        } else {
            let mut out = Vec::new();
            for run in &node.mask_runs {
                let y = ((run.y as f32) * scale_y).floor().min(height.saturating_sub(1) as f32)
                    as u32;
                let x0 = ((run.x_start as f32) * scale_x).floor().max(0.0) as u32;
                let x1 = ((run.x_end as f32 + 1.0) * scale_x).ceil().min(width as f32) as u32;
                let len = x1.saturating_sub(x0);
                if len == 0 {
                    continue;
                }
                let stride = ((len as f32 / 96.0).ceil() as u32).max(1);
                let mut x = x0;
                while x < x1 {
                    out.push((x, y));
                    x += stride;
                }
            }
            out
        };

        let total = points.len() as f32;
        if total < 4.0 {
            return RegionLumaStats::default();
        }
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut samples = 0u32;
        for (x, y) in &points {
            let luma = self.grayscale.get_pixel(*x, *y)[0] as f64 / 255.0;
            sum += luma;
            sum_sq += luma * luma;
            sum_y += *y as f64 / height as f64;
            samples += 1;
        }
        let mean = (sum / total as f64) as f32;
        RegionLumaStats {
            mean_luma: mean,
            variance: ((sum_sq / total as f64) as f32 - mean * mean).max(0.0),
            coverage: points.len() as f32 / (width as f32 * height as f32).max(1.0),
            centroid_y: (sum_y / total as f64) as f32,
            samples,
        }
    }
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
    /// Purely synchronous and deterministic: zero network, zero runtime
    /// requirements, runs on the same CPU thread as the pipeline.
    pub fn deliberate(&self, context: PageContext) -> Result<ColorPlan> {
        tracing::info!(
            page = %context.page_id,
            agents = self.agents.len(),
            "starting agent deliberation"
        );

        let mut bus = AgentBus::new(self.config.max_deliberation_rounds);

        // Phase 1: Analysis - each agent produces initial proposals.
        for agent in &self.agents {
            let messages = agent.analyze(&context)?;
            bus.broadcast(messages);
        }

        // Phase 2-3: Negotiation rounds.
        for round in 0..self.config.max_deliberation_rounds {
            let proposals = bus.get_proposals();
            if proposals.is_empty() {
                break;
            }

            let convergence = bus.convergence_score();
            if convergence >= self.config.convergence_threshold {
                tracing::info!(round, convergence, "early convergence reached");
                break;
            }

            tracing::debug!(
                round,
                proposals = proposals.len(),
                convergence,
                "deliberation round"
            );

            for agent in &self.agents {
                let responses = agent.deliberate(&context, &proposals, round)?;
                bus.broadcast(responses);
            }
        }

        // Phase 4: Resolution - resolve remaining conflicts.
        let resolution = self.resolver.resolve(bus.get_messages());

        // Phase 5: Synthesize ColorPlan bound to real page geometry.
        let plan = ColorPlan::from_resolution(resolution, &context);
        tracing::info!(
            page = %context.page_id,
            adjustments = plan.hint_adjustments.len(),
            scene = %plan.scene.time_of_day,
            "deliberation complete"
        );

        Ok(plan)
    }
}
