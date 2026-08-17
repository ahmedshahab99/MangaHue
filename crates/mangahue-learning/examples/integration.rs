// Example integration of the Learning Agent with MangaHue's orchestrator.
//
// This shows how to hook the learning agent into the colorization pipeline
// so every page and every user correction feeds back into the system.

use mangahue_learning::{LearningAgent, LearningConfig};
use std::path::Path;

/// Initialize the learning agent for a project.
pub fn init_learning_agent(project_path: &Path) -> LearningAgent {
    let config = LearningConfig {
        min_corrections_for_patterns: 10,
        min_corrections_for_retrain: 100,
        max_events_in_memory: 500,
        research_trigger_threshold: 0.4,
        enabled: true,
    };
    LearningAgent::new(project_path.to_path_buf(), config)
}

/// Hook this into the colorization pipeline AFTER a page is colorized.
pub fn on_page_colorized(
    agent: &mut LearningAgent,
    page_id: &str,
    chapter_id: &str,
    quality_score: f32,
    saturation: f32,
    colorfulness: f32,
    model_name: &str,
) {
    if let Err(e) = agent.record_approval(
        page_id,
        chapter_id,
        quality_score,
        saturation,
        colorfulness,
        model_name,
    ) {
        tracing::warn!("Failed to record learning event: {}", e);
    }
}

#[allow(clippy::too_many_arguments)]
/// Hook this when a user corrects a color.
pub fn on_user_correction(
    agent: &mut LearningAgent,
    page_id: &str,
    chapter_id: &str,
    selector_key: &str,
    semantic_class: &str,
    predicted_color: [u8; 3],
    corrected_color: [u8; 3],
    region_area_px: u32,
    model_name: &str,
) {
    if let Err(e) = agent.record_correction(
        page_id,
        chapter_id,
        selector_key,
        semantic_class,
        predicted_color,
        corrected_color,
        region_area_px,
        model_name,
    ) {
        tracing::warn!("Failed to record correction: {}", e);
    }
}

/// Call at end of chapter to check for patterns.
pub fn on_chapter_complete(agent: &LearningAgent) {
    match agent.analyze_patterns() {
        Ok(analysis) => {
            tracing::info!(
                total_events = analysis.total_events,
                corrections = analysis.total_corrections,
                approvals = analysis.total_approvals,
                correction_rate = format!("{:.1}%", analysis.correction_rate * 100.0),
                trend = ?analysis.overall_trend,
                "Learning analysis complete"
            );
            for pattern in &analysis.patterns {
                tracing::info!(
                    class = %pattern.dominant_failing_class,
                    frequency = pattern.frequency,
                    priority = format!("{:.2}", pattern.priority),
                    fix = %pattern.suggested_fix,
                    "Pattern detected"
                );
            }
            if let Err(e) = agent.save_patterns(&analysis) {
                tracing::warn!("Failed to save patterns: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to analyze patterns: {}", e),
    }
}

/// Call periodically to get a learning decision.
pub fn get_learning_decision(agent: &LearningAgent) {
    match agent.decide() {
        Ok(decision) => tracing::info!(?decision, "Learning agent decision"),
        Err(e) => tracing::warn!("Failed to get learning decision: {}", e),
    }
}

fn main() {
    // Example usage
    let project = Path::new("my_manga_project");
    let mut agent = init_learning_agent(project);

    // Simulate colorizing pages
    on_page_colorized(&mut agent, "p1", "ch1", 0.9, 0.4, 12.0, "ddcolor-tiny");
    on_page_colorized(&mut agent, "p2", "ch1", 0.85, 0.35, 10.0, "ddcolor-tiny");

    // Simulate a user correction
    on_user_correction(
        &mut agent,
        "p1",
        "ch1",
        "skin:face1",
        "skin",
        [100, 80, 70],   // Model predicted gray-brown
        [255, 200, 180], // User corrected to warm peach
        5000,
        "ddcolor-tiny",
    );

    // Chapter complete — analyze patterns
    on_chapter_complete(&agent);

    // Check for learning decisions
    get_learning_decision(&agent);

    // Flush pending events
    agent.flush_events().unwrap();
}
