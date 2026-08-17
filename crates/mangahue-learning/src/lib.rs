//! Persistent learning agent for MangaHue.
//!
//! This crate implements a self-improving loop where every colorization
//! and user correction feeds back into the system to improve future output.
//!
//! # Architecture
//!
//! ```text
//! Page → Colorize → User Corrects → Learning Event Recorded
//!                                          ↓
//!                                   Pattern Detection
//!                                          ↓
//!                                   Learning Decision
//!                                    ↙       ↓       ↘
//!                            Update Hints  Retrain  Research
//!                                 ↓          ↓          ↓
//!                            Next Page Is Better
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LearningError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("memory: {0}")]
    Memory(String),
    #[error("no corrections to learn from")]
    NoCorrections,
}

pub type Result<T> = std::result::Result<T, LearningError>;

// ---------------------------------------------------------------------------
// Learning Events — what gets recorded for every page
// ---------------------------------------------------------------------------

/// A single observation from the colorization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub event_id: String,
    pub page_id: String,
    pub chapter_id: String,
    pub timestamp: DateTime<Utc>,
    pub kind: LearningEventKind,
    pub context: EventContext,
}

/// What happened on this page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningEventKind {
    /// Model produced output and user approved it (positive signal).
    Approved {
        quality_score: f32,
        saturation: f32,
        colorfulness: f32,
    },
    /// User corrected a specific region (learning signal).
    Correction {
        selector_key: String,
        semantic_class: String,
        predicted_color: [u8; 3],
        corrected_color: [u8; 3],
        color_distance: f32,
        region_area_px: u32,
    },
    /// User rejected the entire output (strong negative signal).
    Rejected { reason: String, quality_score: f32 },
}

/// Context around the event — what the model saw and decided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    /// Semantic class of the affected region (skin, hair, background, etc.)
    pub region_semantic: String,
    /// Character ID if the region belongs to a known character.
    pub character_id: Option<String>,
    /// Scene lighting estimate (day, night, indoor, etc.)
    pub scene_lighting: Option<String>,
    /// Time-of-day if detected.
    pub time_of_day: Option<String>,
    /// Quality metrics before the user correction.
    pub quality_before: QualitySnapshot,
    /// Model name that produced the output.
    pub model_name: String,
    /// Number of tiles this region spanned.
    pub tile_count: u32,
}

/// Quality metrics at the time of the event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySnapshot {
    pub mean_saturation: f32,
    pub colorfulness: f32,
    pub line_preservation: f32,
    pub hint_adherence: f32,
    pub bubble_cleanliness: f32,
}

// ---------------------------------------------------------------------------
// Pattern Detection — finding systematic errors
// ---------------------------------------------------------------------------

/// Aggregated pattern from analyzing multiple corrections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionPattern {
    /// What semantic class fails most often.
    pub dominant_failing_class: String,
    /// How many corrections involve this class.
    pub frequency: usize,
    /// Average color distance of corrections (bigger = worse).
    pub mean_color_distance: f32,
    /// Common wrong → correct color mappings.
    pub color_shifts: Vec<ColorShift>,
    /// Scenes where this pattern appears.
    pub scene_contexts: Vec<String>,
    /// Suggested fix description.
    pub suggested_fix: String,
    /// Priority score (0-1, higher = more impactful to fix).
    pub priority: f32,
}

/// A systematic color shift that keeps happening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorShift {
    pub from_color: [u8; 3],
    pub to_color: [u8; 3],
    pub count: usize,
    pub mean_distance: f32,
}

/// (predicted_color, corrected_color) -> frequency
pub type ColorShiftEntry = (([u8; 3], [u8; 3]), usize);

/// Pattern analysis result for a batch of corrections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub total_events: usize,
    pub total_corrections: usize,
    pub total_approvals: usize,
    pub total_rejections: usize,
    pub correction_rate: f32,
    pub patterns: Vec<CorrectionPattern>,
    pub top_failing_classes: Vec<(String, usize)>,
    pub overall_trend: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
}

// ---------------------------------------------------------------------------
// Learning Decisions — what the agent decides to do
// ---------------------------------------------------------------------------

/// A decision made by the learning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningDecision {
    /// Update hints/memory for immediate improvement.
    UpdateHints(HintUpdate),
    /// Update a character's color profile.
    UpdateCharacter(CharacterUpdate),
    /// Update palette rules for a scene type.
    UpdatePalette(PaletteUpdate),
    /// Trigger batch retraining when enough corrections accumulate.
    TriggerRetrain(RetrainTrigger),
    /// Log a research trigger when quality plateaus.
    TriggerResearch(ResearchTrigger),
    /// No action needed right now.
    NoOp,
}

/// Immediate hint adjustment based on correction patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintUpdate {
    pub target_semantic: String,
    pub adjustment_type: String,
    pub before: Option<[u8; 3]>,
    pub after: [u8; 3],
    pub confidence: f32,
    pub affected_pages: usize,
}

/// Character color profile update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterUpdate {
    pub character_id: String,
    pub field: String,
    pub old_value: [u8; 3],
    pub new_value: [u8; 3],
    pub confidence: f32,
}

/// Palette rule update for a scene type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteUpdate {
    pub scene_type: String,
    pub role: String,
    pub old_color: Option<[u8; 3]>,
    pub new_color: [u8; 3],
    pub confidence: f32,
}

/// Signal to trigger model retraining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainTrigger {
    pub reason: String,
    pub sample_count: usize,
    pub affected_styles: Vec<String>,
    pub priority: f32,
}

/// Signal that the agent hit a quality plateau and needs new techniques.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchTrigger {
    pub reason: String,
    pub plateau_metric: String,
    pub current_value: f32,
    pub target_value: f32,
}

// ---------------------------------------------------------------------------
// Learning Agent — the main orchestrator
// ---------------------------------------------------------------------------

/// Configuration for the learning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Minimum corrections before pattern detection runs.
    pub min_corrections_for_patterns: usize,
    /// Minimum corrections before retraining is considered.
    pub min_corrections_for_retrain: usize,
    /// Maximum events to keep in memory before flushing to disk.
    pub max_events_in_memory: usize,
    /// Correction rate threshold that triggers research (0-1).
    pub research_trigger_threshold: f32,
    /// Enable/disable automatic learning.
    pub enabled: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            min_corrections_for_patterns: 10,
            min_corrections_for_retrain: 100,
            max_events_in_memory: 500,
            research_trigger_threshold: 0.4,
            enabled: true,
        }
    }
}

/// The persistent learning agent.
pub struct LearningAgent {
    config: LearningConfig,
    #[allow(dead_code)]
    project_path: PathBuf,
    events: Vec<LearningEvent>,
    event_store_path: PathBuf,
    patterns_path: PathBuf,
}

impl LearningAgent {
    /// Create a new learning agent for a project.
    pub fn new(project_path: PathBuf, config: LearningConfig) -> Self {
        let agent_path = project_path.join(".mangahue").join("learning");
        Self {
            config,
            project_path,
            events: Vec::new(),
            event_store_path: agent_path.join("events.jsonl"),
            patterns_path: agent_path.join("patterns.json"),
        }
    }

    /// Record a learning event from a colorization result.
    pub fn record_event(&mut self, event: LearningEvent) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Append to in-memory buffer
        self.events.push(event);

        // Flush to disk if buffer is full
        if self.events.len() >= self.config.max_events_in_memory {
            self.flush_events()?;
        }

        Ok(())
    }

    /// Record that a user approved a colorized page.
    pub fn record_approval(
        &mut self,
        page_id: &str,
        chapter_id: &str,
        quality_score: f32,
        saturation: f32,
        colorfulness: f32,
        model_name: &str,
    ) -> Result<()> {
        let event = LearningEvent {
            event_id: uuid_simple(),
            page_id: page_id.to_string(),
            chapter_id: chapter_id.to_string(),
            timestamp: Utc::now(),
            kind: LearningEventKind::Approved {
                quality_score,
                saturation,
                colorfulness,
            },
            context: EventContext {
                region_semantic: "page".to_string(),
                character_id: None,
                scene_lighting: None,
                time_of_day: None,
                quality_before: QualitySnapshot {
                    mean_saturation: saturation,
                    colorfulness,
                    line_preservation: 1.0,
                    hint_adherence: 1.0,
                    bubble_cleanliness: 1.0,
                },
                model_name: model_name.to_string(),
                tile_count: 1,
            },
        };
        self.record_event(event)
    }

    /// Record that a user corrected a specific region.
    #[allow(clippy::too_many_arguments)]
    pub fn record_correction(
        &mut self,
        page_id: &str,
        chapter_id: &str,
        selector_key: &str,
        semantic_class: &str,
        predicted_color: [u8; 3],
        corrected_color: [u8; 3],
        region_area_px: u32,
        model_name: &str,
    ) -> Result<()> {
        let color_distance = color_distance(predicted_color, corrected_color);

        let event = LearningEvent {
            event_id: uuid_simple(),
            page_id: page_id.to_string(),
            chapter_id: chapter_id.to_string(),
            timestamp: Utc::now(),
            kind: LearningEventKind::Correction {
                selector_key: selector_key.to_string(),
                semantic_class: semantic_class.to_string(),
                predicted_color,
                corrected_color,
                color_distance,
                region_area_px,
            },
            context: EventContext {
                region_semantic: semantic_class.to_string(),
                character_id: None,
                scene_lighting: None,
                time_of_day: None,
                quality_before: QualitySnapshot {
                    mean_saturation: 0.0,
                    colorfulness: 0.0,
                    line_preservation: 0.0,
                    hint_adherence: 0.0,
                    bubble_cleanliness: 0.0,
                },
                model_name: model_name.to_string(),
                tile_count: 1,
            },
        };
        self.record_event(event)
    }

    /// Analyze accumulated corrections for patterns.
    pub fn analyze_patterns(&self) -> Result<PatternAnalysis> {
        let all_events = self.load_all_events()?;
        let corrections: Vec<&LearningEvent> = all_events
            .iter()
            .filter(|e| matches!(e.kind, LearningEventKind::Correction { .. }))
            .collect();
        let approvals: Vec<&LearningEvent> = all_events
            .iter()
            .filter(|e| matches!(e.kind, LearningEventKind::Approved { .. }))
            .collect();
        let rejections: Vec<&LearningEvent> = all_events
            .iter()
            .filter(|e| matches!(e.kind, LearningEventKind::Rejected { .. }))
            .collect();

        let total = all_events.len();
        let correction_rate = if total > 0 {
            corrections.len() as f32 / total as f32
        } else {
            0.0
        };

        // Group corrections by semantic class
        let mut class_corrections: HashMap<String, Vec<&LearningEvent>> = HashMap::new();
        for event in &corrections {
            if let LearningEventKind::Correction { semantic_class, .. } = &event.kind {
                class_corrections
                    .entry(semantic_class.clone())
                    .or_default()
                    .push(event);
            }
        }

        // Find top failing classes
        let mut class_counts: Vec<(String, usize)> = class_corrections
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        class_counts.sort_by_key(|k| std::cmp::Reverse(k.1));

        // Detect patterns for each failing class
        let mut patterns = Vec::new();
        for (class_name, class_events) in &class_corrections {
            if class_events.len() < 3 {
                continue; // Need at least 3 corrections to detect a pattern
            }

            let mut color_shifts: HashMap<([u8; 3], [u8; 3]), usize> = HashMap::new();
            let mut total_distance = 0.0f32;

            for event in class_events {
                if let LearningEventKind::Correction {
                    predicted_color,
                    corrected_color,
                    color_distance,
                    ..
                } = &event.kind
                {
                    total_distance += color_distance;
                    let key = (*predicted_color, *corrected_color);
                    *color_shifts.entry(key).or_insert(0) += 1;
                }
            }

            let mean_distance = total_distance / class_events.len() as f32;

            // Top color shifts
            let mut shifts: Vec<ColorShiftEntry> = color_shifts.into_iter().collect();
            shifts.sort_by_key(|b| std::cmp::Reverse(b.1));
            let top_shifts: Vec<ColorShift> = shifts
                .iter()
                .take(5)
                .map(|((from, to), count)| ColorShift {
                    from_color: *from,
                    to_color: *to,
                    count: *count,
                    mean_distance,
                })
                .collect();

            // Determine suggested fix
            let suggested_fix = match class_name.as_str() {
                "skin" => "Consider warmer shadow shifts and skin-specific palette".to_string(),
                "hair" => "Lock hair color in Color Bible for consistency".to_string(),
                "background" => {
                    "Use scene-aware palette modifiers (day/night, indoor/outdoor)".to_string()
                }
                "clothing" => "Update clothing color hints for this character".to_string(),
                _ => format!("Review {} color assignments", class_name),
            };

            // Priority: frequency * severity
            let priority =
                (class_events.len() as f32 / total.max(1) as f32) * (mean_distance / 100.0);

            patterns.push(CorrectionPattern {
                dominant_failing_class: class_name.clone(),
                frequency: class_events.len(),
                mean_color_distance: mean_distance,
                color_shifts: top_shifts,
                scene_contexts: Vec::new(),
                suggested_fix,
                priority: priority.min(1.0),
            });
        }

        patterns.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Determine overall trend
        let overall_trend = self.detect_trend(&all_events)?;

        Ok(PatternAnalysis {
            total_events: total,
            total_corrections: corrections.len(),
            total_approvals: approvals.len(),
            total_rejections: rejections.len(),
            correction_rate,
            patterns,
            top_failing_classes: class_counts.into_iter().take(5).collect(),
            overall_trend,
        })
    }

    /// Make a learning decision based on accumulated corrections.
    pub fn decide(&self) -> Result<LearningDecision> {
        let analysis = self.analyze_patterns()?;

        // Not enough data yet
        if analysis.total_corrections < self.config.min_corrections_for_patterns {
            return Ok(LearningDecision::NoOp);
        }

        // High correction rate → trigger research
        if analysis.correction_rate > self.config.research_trigger_threshold {
            if let Some(top_pattern) = analysis.patterns.first() {
                return Ok(LearningDecision::TriggerResearch(ResearchTrigger {
                    reason: format!(
                        "High correction rate ({:.0}%) on {} regions",
                        analysis.correction_rate * 100.0,
                        top_pattern.dominant_failing_class
                    ),
                    plateau_metric: "correction_rate".to_string(),
                    current_value: analysis.correction_rate,
                    target_value: 0.2, // Goal: less than 20% correction rate
                }));
            }
        }

        // Enough corrections for retraining
        if analysis.total_corrections >= self.config.min_corrections_for_retrain {
            return Ok(LearningDecision::TriggerRetrain(RetrainTrigger {
                reason: format!(
                    "{} corrections accumulated across {} event types",
                    analysis.total_corrections, analysis.total_events
                ),
                sample_count: analysis.total_corrections,
                affected_styles: analysis
                    .top_failing_classes
                    .iter()
                    .map(|(c, _)| c.clone())
                    .collect(),
                priority: 0.8,
            }));
        }

        // Otherwise, suggest hint updates for the top failing class
        if let Some(top_pattern) = analysis.patterns.first() {
            if let Some(shift) = top_pattern.color_shifts.first() {
                return Ok(LearningDecision::UpdateHints(HintUpdate {
                    target_semantic: top_pattern.dominant_failing_class.clone(),
                    adjustment_type: "average_correction".to_string(),
                    before: Some(shift.from_color),
                    after: shift.to_color,
                    confidence: 0.7,
                    affected_pages: top_pattern.frequency,
                }));
            }
        }

        Ok(LearningDecision::NoOp)
    }

    /// Save all pending events to disk.
    pub fn flush_events(&mut self) -> Result<()> {
        if self.events.is_empty() {
            return Ok(());
        }

        // Ensure directory exists
        if let Some(parent) = self.event_store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Append events to JSONL file
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.event_store_path)?;

        for event in &self.events {
            let line = serde_json::to_string(event)?;
            writeln!(file, "{}", line)?;
        }

        tracing::info!(
            count = self.events.len(),
            path = %self.event_store_path.display(),
            "Flushed learning events to disk"
        );

        self.events.clear();
        Ok(())
    }

    /// Load all events from disk.
    fn load_all_events(&self) -> Result<Vec<LearningEvent>> {
        let mut events = Vec::new();

        // Load from disk
        if self.event_store_path.exists() {
            use std::io::BufRead;
            let file = std::fs::File::open(&self.event_store_path)?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let event: LearningEvent = serde_json::from_str(&line)?;
                events.push(event);
            }
        }

        // Include in-memory events
        events.extend(self.events.iter().cloned());

        Ok(events)
    }

    /// Detect if the trend is improving, stable, or declining.
    fn detect_trend(&self, events: &[LearningEvent]) -> Result<TrendDirection> {
        if events.len() < 20 {
            return Ok(TrendDirection::Stable);
        }

        // Split events in half and compare correction rates
        let mid = events.len() / 2;
        let first_half = &events[..mid];
        let second_half = &events[mid..];

        let rate_first = correction_rate(first_half);
        let rate_second = correction_rate(second_half);

        let diff = rate_second - rate_first;

        if diff < -0.05 {
            Ok(TrendDirection::Improving)
        } else if diff > 0.05 {
            Ok(TrendDirection::Declining)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    /// Save patterns analysis to disk.
    pub fn save_patterns(&self, analysis: &PatternAnalysis) -> Result<()> {
        if let Some(parent) = self.patterns_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(analysis)?;
        std::fs::write(&self.patterns_path, json)?;
        Ok(())
    }

    /// Get summary statistics for display.
    pub fn summary(&self) -> Result<LearningSummary> {
        let events = self.load_all_events()?;
        let analysis = self.analyze_patterns()?;

        Ok(LearningSummary {
            total_events: events.len(),
            total_corrections: analysis.total_corrections,
            total_approvals: analysis.total_approvals,
            correction_rate: analysis.correction_rate,
            top_failing_class: analysis
                .top_failing_classes
                .first()
                .map(|(c, n)| (c.clone(), *n)),
            trend: analysis.overall_trend,
        })
    }
}

/// Summary for display in UI/Telegram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSummary {
    pub total_events: usize,
    pub total_corrections: usize,
    pub total_approvals: usize,
    pub correction_rate: f32,
    pub top_failing_class: Option<(String, usize)>,
    pub trend: TrendDirection,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn correction_rate(events: &[LearningEvent]) -> f32 {
    let total = events.len();
    if total == 0 {
        return 0.0;
    }
    let corrections = events
        .iter()
        .filter(|e| matches!(e.kind, LearningEventKind::Correction { .. }))
        .count();
    corrections as f32 / total as f32
}

/// Euclidean distance between two RGB colors.
fn color_distance(a: [u8; 3], b: [u8; 3]) -> f32 {
    let dr = a[0] as f32 - b[0] as f32;
    let dg = a[1] as f32 - b[1] as f32;
    let db = a[2] as f32 - b[2] as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Simple UUID-like string (no dependency needed).
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:016x}-{:04x}-{:04x}", t, t >> 16, t >> 32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent() -> LearningAgent {
        let dir = std::env::temp_dir().join("mangahue_learning_test");
        std::fs::create_dir_all(&dir).ok();
        LearningAgent::new(dir, LearningConfig::default())
    }

    #[test]
    fn record_approval_and_correction() {
        let mut agent = test_agent();
        agent
            .record_approval("page1", "ch1", 0.9, 0.4, 12.0, "ddcolor-tiny")
            .unwrap();
        agent
            .record_correction(
                "page1",
                "ch1",
                "skin:face1",
                "skin",
                [100, 80, 70],
                [255, 200, 180],
                5000,
                "ddcolor-tiny",
            )
            .unwrap();
        assert_eq!(agent.events.len(), 2);
    }

    #[test]
    fn pattern_detection_finds_failing_class() {
        let mut agent = test_agent();

        // Add many skin corrections to trigger pattern detection
        for i in 0..15 {
            agent
                .record_correction(
                    &format!("page{}", i),
                    "ch1",
                    &format!("skin:face{}", i),
                    "skin",
                    [100 + i as u8, 80, 70],
                    [255, 200, 180],
                    5000,
                    "ddcolor-tiny",
                )
                .unwrap();
        }

        let analysis = agent.analyze_patterns().unwrap();
        assert!(analysis.total_corrections >= 15);
        assert!(!analysis.patterns.is_empty());

        let skin_pattern = analysis
            .patterns
            .iter()
            .find(|p| p.dominant_failing_class == "skin");
        assert!(skin_pattern.is_some());
    }

    #[test]
    fn no_op_when_not_enough_data() {
        let agent = test_agent();
        let decision = agent.decide().unwrap();
        assert!(matches!(decision, LearningDecision::NoOp));
    }

    #[test]
    fn triggers_retrain_after_many_corrections() {
        let mut agent = test_agent();
        for i in 0..120 {
            agent
                .record_correction(
                    &format!("page{}", i),
                    "ch1",
                    &format!("region{}", i),
                    "skin",
                    [100, 80, 70],
                    [255, 200, 180],
                    5000,
                    "ddcolor-tiny",
                )
                .unwrap();
        }

        let decision = agent.decide().unwrap();
        assert!(matches!(decision, LearningDecision::TriggerRetrain(_)));
    }

    #[test]
    fn flush_and_reload() {
        let mut agent = test_agent();
        agent
            .record_approval("p1", "ch1", 0.9, 0.4, 12.0, "ddcolor-tiny")
            .unwrap();
        agent.flush_events().unwrap();

        let reloaded = LearningAgent::new(agent.project_path.clone(), LearningConfig::default());
        let events = reloaded.load_all_events().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn color_distance_calculation() {
        let a = [255, 200, 180];
        let b = [100, 80, 70];
        let d = color_distance(a, b);
        assert!(d > 200.0);
        assert!(d < 300.0);

        let d_same = color_distance([128, 128, 128], [128, 128, 128]);
        assert_eq!(d_same, 0.0);
    }

    #[test]
    fn trend_detection() {
        let agent = test_agent();
        let mut events = Vec::new();

        // First half: mostly corrections (bad)
        for i in 0..20 {
            events.push(LearningEvent {
                event_id: format!("e{}", i),
                page_id: format!("p{}", i),
                chapter_id: "ch1".to_string(),
                timestamp: Utc::now(),
                kind: LearningEventKind::Correction {
                    selector_key: "skin".to_string(),
                    semantic_class: "skin".to_string(),
                    predicted_color: [100, 80, 70],
                    corrected_color: [255, 200, 180],
                    color_distance: 150.0,
                    region_area_px: 5000,
                },
                context: EventContext {
                    region_semantic: "skin".to_string(),
                    character_id: None,
                    scene_lighting: None,
                    time_of_day: None,
                    quality_before: QualitySnapshot {
                        mean_saturation: 0.3,
                        colorfulness: 10.0,
                        line_preservation: 0.9,
                        hint_adherence: 0.7,
                        bubble_cleanliness: 1.0,
                    },
                    model_name: "ddcolor-tiny".to_string(),
                    tile_count: 1,
                },
            });
        }

        // Second half: mostly approvals (good)
        for i in 20..40 {
            events.push(LearningEvent {
                event_id: format!("e{}", i),
                page_id: format!("p{}", i),
                chapter_id: "ch1".to_string(),
                timestamp: Utc::now(),
                kind: LearningEventKind::Approved {
                    quality_score: 0.9,
                    saturation: 0.4,
                    colorfulness: 15.0,
                },
                context: EventContext {
                    region_semantic: "page".to_string(),
                    character_id: None,
                    scene_lighting: None,
                    time_of_day: None,
                    quality_before: QualitySnapshot {
                        mean_saturation: 0.4,
                        colorfulness: 15.0,
                        line_preservation: 0.95,
                        hint_adherence: 0.9,
                        bubble_cleanliness: 1.0,
                    },
                    model_name: "ddcolor-tiny".to_string(),
                    tile_count: 1,
                },
            });
        }

        let trend = agent.detect_trend(&events).unwrap();
        assert!(matches!(trend, TrendDirection::Improving));
    }
}
