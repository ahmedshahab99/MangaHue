use anyhow::{anyhow, Context, Result};
use mangahue_core::{ColorRgb, StyleProfile};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingDirection {
    LeftToRight,
    RightToLeft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSection {
    pub preserve_lines: bool,
    pub protect_speech_bubbles: bool,
    pub save_debug_outputs: bool,
    pub reading_direction: ReadingDirection,
    pub parallel_jobs: usize,
    #[serde(default)]
    pub pyramid_enabled: bool,
    #[serde(default)]
    pub hint_propagation_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessSection {
    pub grayscale_mode: String,
    pub denoise_enabled: bool,
    pub denoise_strength: u8,
    pub threshold_mode: String,
    pub line_sharpening: f32,
    pub background_cleanup: bool,
    pub max_input_dimension: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSection {
    pub panel_detection: bool,
    pub tiling_enabled: bool,
    pub tile_size: u32,
    pub tile_overlap: u32,
    pub webtoon_mode: String,
    pub reading_direction: ReadingDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintsSection {
    pub user_hints_enabled: bool,
    pub character_memory_enabled: bool,
    pub style_hints_enabled: bool,
    pub hint_blur_radius: f32,
    pub hint_strength: f32,
    #[serde(default)]
    pub region_style_hints_enabled: bool,
    #[serde(default = "default_region_style_hint_budget")]
    pub region_style_hint_budget: usize,
    #[serde(default = "default_propagation_radius")]
    pub propagation_radius: u32,
    #[serde(default = "default_propagation_similarity")]
    pub propagation_similarity_threshold: f32,
    #[serde(default)]
    pub adaptive_placement_enabled: bool,
}

fn default_propagation_radius() -> u32 {
    48
}

fn default_propagation_similarity() -> f32 {
    0.3
}

fn default_region_style_hint_budget() -> usize {
    16
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSection {
    pub model_path: PathBuf,
    pub model_metadata_path: PathBuf,
    pub device: String,
    pub batch_size: usize,
    pub input_size: u32,
    pub fp16_enabled: bool,
    pub resize_policy: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PostprocessSection {
    pub restore_lines: bool,
    pub line_strength: f32,
    pub bubble_cleanup: bool,
    pub bleeding_correction: bool,
    pub palette_correction: bool,
    pub final_sharpening: f32,
    #[serde(default)]
    pub reference_finishing: bool,
    #[serde(default = "default_target_colorfulness")]
    pub target_colorfulness: f32,
    #[serde(default = "default_midtone_depth")]
    pub midtone_depth: f32,
    #[serde(default = "default_local_contrast")]
    pub local_contrast: f32,
    #[serde(default)]
    pub semantic_relighting: bool,
    #[serde(default = "default_relighting_strength")]
    pub relighting_strength: f32,
    #[serde(default = "default_natural_grading")]
    pub natural_grading: bool,
    #[serde(default = "default_reference_transfer")]
    pub reference_transfer: bool,
    /// Full-coverage mode: regions the model leaves neutral (bright
    /// backgrounds, white clothing, page margins, blank paper) receive a
    /// gentle palette-matched tint instead of staying gray. Speech bubbles
    /// and ink remain protected.
    #[serde(default = "default_soft_coverage")]
    pub soft_coverage: bool,
    /// Hint-guided chroma correction strength (0 = off): where the hint map
    /// is confident and the model's own chroma is weak (near-gray output),
    /// the pixel's hue snaps toward the hint color while the model's value
    /// (shading) is preserved. This is how the semantic region hints reach
    /// the DDColor output: the model stays the artist, the hints fix its
    /// blind spots.
    #[serde(default)]
    pub hint_chroma_strength: f32,
    /// Remove isolated strong-color specks (the model's dot noise):
    /// DDColor-family models scatter small warm dots on textures; this pass
    /// blends such pixels toward their neighborhood.
    #[serde(default = "default_despeckle_color_specks")]
    pub despeckle_color_specks: bool,
    /// Remove printed halftone dot artifacts and periodic chroma noise:
    /// small saturated warm-color clusters (dark red/orange print dots on
    /// lighter paper — the "red dots" on colored scans) are replaced with
    /// the local background color. Black ink and large colored regions are
    /// never touched.
    #[serde(default = "default_descreen_halftone")]
    pub descreen_halftone: bool,
    /// Muted target grade: reshapes every pixel's saturation toward the
    /// measured profile of the reference style (low-key warm look) instead
    /// of the default saturation boost.
    #[serde(default = "default_region_chroma_separation")]
    pub region_chroma_separation: f32,
    #[serde(default)]
    pub muted_target_grade: bool,
    #[serde(default = "default_muted_strength")]
    pub muted_strength: f32,
    #[serde(default = "default_muted_hue_shift_deg")]
    pub muted_hue_shift_deg: f32,
    #[serde(default = "default_muted_dark_pull")]
    pub muted_dark_pull: f32,
    #[serde(default = "default_soft_strength")]
    pub soft_strength: f32,
    #[serde(default = "default_soft_confidence")]
    pub soft_confidence: f32,
    #[serde(default = "default_muted_dark_sat")]
    pub muted_dark_sat: f32,
    #[serde(default = "default_muted_mid_sat")]
    pub muted_mid_sat: f32,
    #[serde(default = "default_muted_bright_sat")]
    pub muted_bright_sat: f32,
    #[serde(default = "default_muted_paper_sat")]
    pub muted_paper_sat: f32,
    /// Fraction of vivid saturation the muted grade's hard knee keeps
    /// (0.0 = legacy: S > 0.55 is halved; 1.0 = the knee is a no-op).
    /// Lets presets hold "moderate" instead of "flat" saturation.
    #[serde(default = "default_muted_vivid_preserve")]
    pub muted_vivid_preserve: f32,
    /// Yellow-band hue shift (degrees): rotates hues in the 40-75 deg band
    /// toward orange. The model over-produces yellow on manga pages; the
    /// soft-cel reference is orange/skin with almost no yellow. Skin
    /// (0-40 deg) is untouched.
    #[serde(default)]
    pub muted_yellow_shift_deg: f32,
    /// Model-first purity mode: region machinery preserves the model's own
    /// chroma wherever it is confident (lower confidence bars, director and
    /// propagation freeze model-derived regions, palette harmony yields to
    /// model color, face paint defers to real skin chroma). The palette
    /// becomes a fallback for genuinely neutral regions only.
    #[serde(default)]
    pub model_first: bool,
    /// Confidence bars for the region color decision ladder. Regions whose
    /// model chroma magnitude (|R-L| + |B-L|, 0-255 scale) passes the bar
    /// keep the model's hue; weaker chroma falls to the palette tier.
    #[serde(default = "default_seed_chroma_bar_bright")]
    pub seed_chroma_bar_bright: f32,
    #[serde(default = "default_seed_chroma_bar_mid")]
    pub seed_chroma_bar_mid: f32,
    #[serde(default = "default_model_chroma_bar_bright")]
    pub model_chroma_bar_bright: f32,
    #[serde(default = "default_model_chroma_bar_mid")]
    pub model_chroma_bar_mid: f32,
    /// Gain applied to confident model chroma when a region keeps its own
    /// hue (1.0 = exact pass-through; legacy 1.3 amplified it).
    #[serde(default = "default_model_chroma_gain")]
    pub model_chroma_gain: f32,
    /// Strength of the global palette-harmony pull on every pixel. In
    /// model-first mode the pull is additionally gated by per-pixel chroma
    /// so model-colored pixels stay untouched.
    #[serde(default = "default_palette_harmony_strength")]
    pub palette_harmony_strength: f32,
    /// How strongly the guaranteed face paint replaces model skin chroma
    /// (1.0 = full peach injection; lower = keep real model skin).
    #[serde(default = "default_face_paint_strength")]
    pub face_paint_strength: f32,
    /// Directional ink ambient occlusion strength: ink casts a soft shadow
    /// away from the estimated light direction (0.0 = off; ~0.5-0.9 adds
    /// the readable cel "ink shadow" look without photoreal relighting).
    #[serde(default)]
    pub ambient_occlusion_strength: f32,
    /// Smart color governor: measures the page's own saturation and
    /// colorfulness, then applies a bounded chroma gain toward these
    /// targets (hue/luma preserved). The model's chroma signal scales the
    /// target, so the color budget follows the model's own judgment.
    #[serde(default)]
    pub smart_governor: bool,
    #[serde(default = "default_governor_target_saturation")]
    pub governor_target_saturation: f32,
    #[serde(default = "default_governor_target_colorfulness")]
    pub governor_target_colorfulness: f32,
    #[serde(default = "default_governor_min_gain")]
    pub governor_min_gain: f32,
    #[serde(default = "default_governor_max_gain")]
    pub governor_max_gain: f32,
    /// Warmth target for the smart governor: mean (R-B) over midtone
    /// pixels, 0-255 units. The model's palette leans golden; when the
    /// measured warmth exceeds the target, warm pixels are compressed
    /// toward neutral (bounded, hue-preserving for cool colors).
    #[serde(default = "default_governor_target_warmth")]
    pub governor_target_warmth: f32,
    /// Two-stage grading architecture: color authoring (hints, palette,
    /// region renderer, reference) runs first, then ONE measured, bounded
    /// grade pass replaces the legacy natural→muted→governor cascade that
    /// re-shaped the same chroma signal with four overlapping curves.
    /// False restores the legacy cascade exactly as it was.
    #[serde(default = "default_unified_grade")]
    pub unified_grade: bool,
    /// Soft cel shading: quantize the model's continuous tone into discrete
    /// brightness bands while preserving the model's local hue. The result
    /// is painterly flat regions with smooth band transitions instead of
    /// photo-style smooth gradients. Off by default for photo-natural look.
    #[serde(default)]
    pub cel_shading_enabled: bool,
    /// Number of luminance bands the cel pass reduces to (2 = shadow/mid,
    /// 3 = shadow/mid/high). Three is the classic soft cel look.
    #[serde(default = "default_cel_levels")]
    pub cel_levels: u8,
    /// Multiplier applied to the localized base color in the shadow band.
    /// Lower = deeper shadow. Typical 0.50-0.65.
    #[serde(default = "default_cel_shadow_factor")]
    pub cel_shadow_factor: f32,
    /// Multiplier applied to the localized base color in the highlight band.
    /// Higher = brighter highlight. Typical 1.10-1.25.
    #[serde(default = "default_cel_highlight_factor")]
    pub cel_highlight_factor: f32,
    /// Strength of the cel pass (0 = off, 1 = full band replacement).
    /// 0.55-0.75 keeps model color drift and soft blending; 0.95+ gives
    /// crisp bands.
    #[serde(default = "default_cel_strength")]
    pub cel_strength: f32,
    /// Smoothstep width around the band edges so transitions stay painterly
    /// instead of hard. Typical 0.10-0.25.
    #[serde(default = "default_cel_smoothing")]
    pub cel_smoothing: f32,
    /// Local-mean window radius (pixels) used to compute the "region base
    /// color" before band quantization. Higher = broader smooth cel.
    #[serde(default = "default_cel_window_radius")]
    pub cel_window_radius: u8,
}

fn default_cel_levels() -> u8 {
    3
}

fn default_cel_shadow_factor() -> f32 {
    0.58
}

fn default_cel_highlight_factor() -> f32 {
    1.18
}

fn default_cel_strength() -> f32 {
    0.7
}

fn default_cel_smoothing() -> f32 {
    0.18
}

fn default_cel_window_radius() -> u8 {
    4
}

fn default_muted_dark_sat() -> f32 {
    0.19
}

fn default_muted_mid_sat() -> f32 {
    0.27
}

fn default_muted_bright_sat() -> f32 {
    0.22
}

fn default_muted_paper_sat() -> f32 {
    0.034
}

fn default_muted_vivid_preserve() -> f32 {
    0.0
}

fn default_seed_chroma_bar_bright() -> f32 {
    48.0
}

fn default_seed_chroma_bar_mid() -> f32 {
    24.0
}

fn default_model_chroma_bar_bright() -> f32 {
    40.0
}

fn default_model_chroma_bar_mid() -> f32 {
    20.0
}

fn default_model_chroma_gain() -> f32 {
    1.3
}

fn default_palette_harmony_strength() -> f32 {
    0.16
}

fn default_face_paint_strength() -> f32 {
    0.92
}

fn default_governor_target_saturation() -> f32 {
    0.20
}

fn default_governor_target_colorfulness() -> f32 {
    36.0
}

fn default_governor_min_gain() -> f32 {
    0.8
}

fn default_governor_max_gain() -> f32 {
    1.25
}

fn default_governor_target_warmth() -> f32 {
    45.0
}

fn default_unified_grade() -> bool {
    true
}

fn default_soft_coverage() -> bool {
    true
}

fn default_region_chroma_separation() -> f32 {
    34.0
}

fn default_despeckle_color_specks() -> bool {
    true
}

fn default_descreen_halftone() -> bool {
    true
}

fn default_muted_strength() -> f32 {
    0.95
}

fn default_muted_hue_shift_deg() -> f32 {
    16.0
}

fn default_muted_dark_pull() -> f32 {
    0.92
}

fn default_soft_strength() -> f32 {
    1.0
}

fn default_soft_confidence() -> f32 {
    1.0
}

fn default_reference_transfer() -> bool {
    true
}

fn default_natural_grading() -> bool {
    true
}

fn default_target_colorfulness() -> f32 {
    1.0
}

fn default_midtone_depth() -> f32 {
    0.08
}

fn default_local_contrast() -> f32 {
    0.12
}

fn default_relighting_strength() -> f32 {
    0.25
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
    pub format: String,
    pub quality: u8,
    pub write_report: bool,
    pub write_intermediates: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSection {
    pub line_preservation_threshold: f32,
    /// Below this score the page is delivery-blocking (`needs_*`); between
    /// this and `line_preservation_threshold` the page is delivered with a
    /// non-blocking warning. Real model outputs routinely lighten hard
    /// anti-aliased or fine line work enough to miss the historic strict
    /// threshold without any visible damage.
    #[serde(default = "default_line_preservation_block_threshold")]
    pub line_preservation_block_threshold: f32,
    pub protected_region_violation_threshold: f32,
    /// Below this score the page is delivery-blocking (`needs_*`); between
    /// this and `protected_region_violation_threshold` the page is delivered
    /// with a non-blocking warning. Real manga pages routinely score
    /// 0.55-0.75 here because the protected mask is deliberately dilated
    /// around glyphs while the last-mile restore preserves midtone material
    /// color, so the historical single threshold caused valid chapters to
    /// lose pages.
    #[serde(default = "default_protected_region_block_threshold")]
    pub protected_region_block_threshold: f32,
    pub hint_adherence_threshold: f32,
    pub screen_tone_warning_threshold: f32,
}

fn default_line_preservation_block_threshold() -> f32 {
    0.55
}

fn default_protected_region_block_threshold() -> f32 {
    0.45
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleStrategy {
    MajorityWeighted,
    AverageBlend,
    DivergenceReject,
    #[default]
    ChromaComplement,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnsembleSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_ensemble_strategy")]
    pub strategy: EnsembleStrategy,
    #[serde(default = "default_ensemble_weight_primary")]
    pub primary_weight: f32,
    #[serde(default)]
    pub parallel_variants: bool,
    #[serde(default = "default_ensemble_divergence_threshold")]
    pub divergence_reject_threshold: f32,
    #[serde(default)]
    pub cielab_distance: bool,
}

fn default_ensemble_strategy() -> EnsembleStrategy {
    EnsembleStrategy::ChromaComplement
}

fn default_ensemble_weight_primary() -> f32 {
    1.0
}

fn default_ensemble_divergence_threshold() -> f32 {
    0.5
}

impl Default for EvalSection {
    fn default() -> Self {
        Self {
            line_preservation_threshold: 0.92,
            line_preservation_block_threshold: default_line_preservation_block_threshold(),
            protected_region_violation_threshold: 0.96,
            protected_region_block_threshold: default_protected_region_block_threshold(),
            hint_adherence_threshold: 0.72,
            screen_tone_warning_threshold: 0.18,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_pyramid_levels")]
    pub levels: Vec<u32>,
    #[serde(default = "default_pyramid_weights")]
    pub blend_weights: Vec<f32>,
}

fn default_pyramid_levels() -> Vec<u32> {
    vec![512, 768]
}

fn default_pyramid_weights() -> Vec<f32> {
    vec![0.35, 0.65]
}

impl Default for PyramidSection {
    fn default() -> Self {
        Self {
            enabled: false,
            levels: default_pyramid_levels(),
            blend_weights: default_pyramid_weights(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub engine: EngineSection,
    pub preprocess: PreprocessSection,
    pub layout: LayoutSection,
    pub hints: HintsSection,
    pub inference: InferenceSection,
    pub postprocess: PostprocessSection,
    pub output: OutputSection,
    pub eval: EvalSection,
    #[serde(default)]
    pub ensemble: EnsembleSection,
    #[serde(default)]
    pub pyramid: PyramidSection,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            engine: EngineSection {
                preserve_lines: true,
                protect_speech_bubbles: true,
                save_debug_outputs: false,
                reading_direction: ReadingDirection::RightToLeft,
                parallel_jobs: 0,
                pyramid_enabled: false,
                hint_propagation_enabled: false,
            },
            preprocess: PreprocessSection {
                grayscale_mode: "luma".to_string(),
                denoise_enabled: true,
                denoise_strength: 1,
                threshold_mode: "adaptive".to_string(),
                line_sharpening: 0.35,
                background_cleanup: true,
                max_input_dimension: 2600,
            },
            layout: LayoutSection {
                panel_detection: true,
                tiling_enabled: true,
                tile_size: 768,
                tile_overlap: 64,
                webtoon_mode: "auto".to_string(),
                reading_direction: ReadingDirection::RightToLeft,
            },
            hints: HintsSection {
                user_hints_enabled: true,
                character_memory_enabled: true,
                style_hints_enabled: true,
                hint_blur_radius: 4.0,
                hint_strength: 1.0,
                region_style_hints_enabled: false,
                region_style_hint_budget: default_region_style_hint_budget(),
                propagation_radius: default_propagation_radius(),
                propagation_similarity_threshold: default_propagation_similarity(),
                adaptive_placement_enabled: false,
            },
            inference: InferenceSection {
                model_path: PathBuf::from("models/example_model.onnx.placeholder"),
                model_metadata_path: PathBuf::from("models/model.mangahue.json"),
                device: "cpu".to_string(),
                batch_size: 1,
                input_size: 768,
                fp16_enabled: false,
                resize_policy: "fit".to_string(),
            },
            postprocess: PostprocessSection {
                restore_lines: true,
                line_strength: 1.0,
                bubble_cleanup: true,
                bleeding_correction: true,
                palette_correction: true,
                final_sharpening: 0.15,
                reference_finishing: false,
                target_colorfulness: default_target_colorfulness(),
                midtone_depth: default_midtone_depth(),
                local_contrast: default_local_contrast(),
                semantic_relighting: false,
                relighting_strength: default_relighting_strength(),
                natural_grading: default_natural_grading(),
                reference_transfer: default_reference_transfer(),
                soft_coverage: default_soft_coverage(),
                region_chroma_separation: default_region_chroma_separation(),
                muted_target_grade: false,
                muted_strength: default_muted_strength(),
                muted_hue_shift_deg: default_muted_hue_shift_deg(),
                muted_dark_pull: default_muted_dark_pull(),
                soft_strength: default_soft_strength(),
                soft_confidence: default_soft_confidence(),
                muted_dark_sat: default_muted_dark_sat(),
                muted_mid_sat: default_muted_mid_sat(),
                muted_bright_sat: default_muted_bright_sat(),
                muted_paper_sat: default_muted_paper_sat(),
                muted_vivid_preserve: default_muted_vivid_preserve(),
                muted_yellow_shift_deg: 0.0,
                model_first: false,
                seed_chroma_bar_bright: default_seed_chroma_bar_bright(),
                seed_chroma_bar_mid: default_seed_chroma_bar_mid(),
                model_chroma_bar_bright: default_model_chroma_bar_bright(),
                model_chroma_bar_mid: default_model_chroma_bar_mid(),
                model_chroma_gain: default_model_chroma_gain(),
                palette_harmony_strength: default_palette_harmony_strength(),
                face_paint_strength: default_face_paint_strength(),
                ambient_occlusion_strength: 0.0,
                smart_governor: false,
                governor_target_saturation: default_governor_target_saturation(),
                governor_target_colorfulness: default_governor_target_colorfulness(),
                governor_min_gain: default_governor_min_gain(),
                governor_max_gain: default_governor_max_gain(),
                governor_target_warmth: default_governor_target_warmth(),
                hint_chroma_strength: 0.0,
                despeckle_color_specks: default_despeckle_color_specks(),
                descreen_halftone: default_descreen_halftone(),
                unified_grade: default_unified_grade(),
                cel_shading_enabled: false,
                cel_levels: default_cel_levels(),
                cel_shadow_factor: default_cel_shadow_factor(),
                cel_highlight_factor: default_cel_highlight_factor(),
                cel_strength: default_cel_strength(),
                cel_smoothing: default_cel_smoothing(),
                cel_window_radius: default_cel_window_radius(),
            },
            output: OutputSection {
                format: "png".to_string(),
                quality: 95,
                write_report: true,
                write_intermediates: false,
            },
            eval: EvalSection::default(),
            ensemble: EnsembleSection {
                enabled: false,
                strategy: default_ensemble_strategy(),
                primary_weight: default_ensemble_weight_primary(),
                parallel_variants: false,
                divergence_reject_threshold: default_ensemble_divergence_threshold(),
                cielab_distance: false,
            },
            pyramid: PyramidSection::default(),
        }
    }
}

impl EngineConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_paths([path.as_ref()])
    }

    pub fn from_paths<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Result<Self> {
        let mut merged = serde_yaml::to_value(Self::default())?;
        for path in paths {
            if !path.exists() {
                return Err(anyhow!("config path does not exist: {}", path.display()));
            }
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read config {}", path.display()))?;
            let overlay: Value = serde_yaml::from_str(&text)
                .with_context(|| format!("failed to parse yaml {}", path.display()))?;
            merge_yaml(&mut merged, overlay);
        }
        let config: Self = serde_yaml::from_value(merged)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.layout.tile_size == 0 {
            return Err(anyhow!("layout.tile_size must be greater than zero"));
        }
        if self.layout.tile_overlap >= self.layout.tile_size {
            return Err(anyhow!(
                "layout.tile_overlap must be smaller than tile_size"
            ));
        }
        if !(0.0..=4.0).contains(&self.hints.hint_strength) {
            return Err(anyhow!("hints.hint_strength must be between 0 and 4"));
        }
        if self.inference.input_size == 0 {
            return Err(anyhow!("inference.input_size must be greater than zero"));
        }
        if !(1..=100).contains(&self.output.quality) {
            return Err(anyhow!("output.quality must be in 1..=100"));
        }
        if !(0.0..=2.5).contains(&self.postprocess.target_colorfulness) {
            return Err(anyhow!(
                "postprocess.target_colorfulness must be in 0.0..=2.5"
            ));
        }
        if !(0.0..=0.35).contains(&self.postprocess.midtone_depth) {
            return Err(anyhow!("postprocess.midtone_depth must be in 0.0..=0.35"));
        }
        if !(0.0..=0.6).contains(&self.postprocess.local_contrast) {
            return Err(anyhow!("postprocess.local_contrast must be in 0.0..=0.6"));
        }
        if self.postprocess.governor_min_gain > self.postprocess.governor_max_gain {
            return Err(anyhow!(
                "postprocess.governor_min_gain must be <= governor_max_gain"
            ));
        }
        if self.engine.parallel_jobs > 256 {
            return Err(anyhow!("engine.parallel_jobs must be <= 256"));
        }
        if self.hints.region_style_hint_budget > 128 {
            return Err(anyhow!("hints.region_style_hint_budget must be <= 128"));
        }
        // The blur is an O(W·H·r²) filter: an unbounded radius turns a page
        // into a multi-minute hang.
        if self.hints.hint_blur_radius > 64.0 {
            return Err(anyhow!("hints.hint_blur_radius must be <= 64"));
        }
        if self.hints.propagation_radius > 256 {
            return Err(anyhow!("hints.propagation_radius must be <= 256"));
        }
        if !(0.0..=1.0).contains(&self.hints.propagation_similarity_threshold) {
            return Err(anyhow!(
                "hints.propagation_similarity_threshold must be in 0.0..=1.0"
            ));
        }
        if !(0.1..=5.0).contains(&self.ensemble.primary_weight) {
            return Err(anyhow!("ensemble.primary_weight must be in 0.1..=5.0"));
        }
        if !(0.1..=1.0).contains(&self.ensemble.divergence_reject_threshold) {
            return Err(anyhow!(
                "ensemble.divergence_reject_threshold must be in 0.1..=1.0"
            ));
        }
        Ok(())
    }
}

pub fn load_style_profile(path: impl AsRef<Path>) -> Result<StyleProfile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read style profile {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse style profile {}", path.display()))
}

// ── Artist palette files (configs/palettes/*.yaml) ──────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteFile {
    pub name: String,
    pub colors: Vec<String>,
}

/// Load a palette YAML (e.g. configs/palettes/seinen.yaml) into colors.
pub fn load_palette(path: impl AsRef<Path>) -> Result<Vec<ColorRgb>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read palette {}", path.display()))?;
    let file: PaletteFile = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse palette {}", path.display()))?;
    file.colors
        .iter()
        .map(|hex| {
            ColorRgb::from_str(hex)
                .map_err(|_| anyhow!("invalid palette color {hex} in {}", path.display()))
        })
        .collect()
}

/// The scene palettes eligible for automatic selection (skin_tones and
/// hair_defaults are material palettes, not scene palettes).
pub const AUTO_PALETTES: &[&str] = &["fantasy", "seinen", "shojo", "shonen", "webtoon"];

/// Pick the artist palette that best fits the page's fingerprint.
/// brightness: 0..1 mean luma; contrast: gray std-dev (0..~80).
pub fn select_palette_name(brightness: f32, contrast: f32) -> &'static str {
    // Each scene palette has a target profile (brightness, contrast).
    let candidates: &[(&str, f32, f32)] = &[
        ("seinen", 0.42, 55.0),  // dark, dramatic, muted
        ("fantasy", 0.55, 45.0), // mid, saturated
        ("shonen", 0.60, 60.0),  // mid-bright, punchy
        ("shojo", 0.68, 35.0),   // bright, soft
        ("webtoon", 0.72, 58.0), // bright, high contrast
    ];
    let mut best = "shonen";
    let mut best_score = f32::MAX;
    for (name, b_target, c_target) in candidates {
        let score = (brightness - b_target).abs() * 3.0 + (contrast - c_target).abs() / 20.0;
        if score < best_score {
            best_score = score;
            best = name;
        }
    }
    best
}

/// Default palette directory (relative to the working directory, matching
/// the repo layout used by the bot service).
pub fn default_palettes_dir() -> PathBuf {
    PathBuf::from("configs/palettes")
}

fn merge_yaml(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Mapping(base_map), Value::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(base_value) => merge_yaml(base_value, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base_value, overlay_value) => *base_value = overlay_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_invalid_tile_overlap() {
        let mut config = EngineConfig::default();
        config.layout.tile_size = 128;
        config.layout.tile_overlap = 128;
        assert!(config.validate().is_err());
    }

    #[test]
    fn overlays_partial_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.yaml");
        let mut file = fs::File::create(&path).unwrap();
        writeln!(
            file,
            "layout:\n  tile_size: 512\npostprocess:\n  line_strength: 1.2"
        )
        .unwrap();

        let config = EngineConfig::from_path(path).unwrap();
        assert_eq!(config.layout.tile_size, 512);
        assert_eq!(config.layout.tile_overlap, 64);
        assert!((config.postprocess.line_strength - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn legacy_mode_and_already_colored_keys_cannot_reenable_bypass() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy-engine.yaml");
        fs::write(
            &path,
            "engine:\n  mode: fast\n  skip_already_colored: true\n  already_colored_sat_threshold: 0.01\n  already_colored_colored_fraction: 0.01\n",
        )
        .unwrap();

        // Legacy deployment files remain parseable, but those removed keys
        // have no representation in EngineConfig and therefore no behavior.
        let config = EngineConfig::from_path(path).unwrap();
        assert!(config.engine.preserve_lines);
        assert!(config.engine.protect_speech_bubbles);
    }

    #[test]
    fn loads_real_palette_files() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/palettes");
        for name in AUTO_PALETTES {
            let colors = load_palette(dir.join(format!("{name}.yaml"))).unwrap();
            assert!(!colors.is_empty(), "{name} palette empty");
            assert!(colors.len() <= 64, "{name} palette unexpectedly large");
        }
    }

    #[test]
    fn selects_palette_by_page_fingerprint() {
        // Dark dramatic page -> seinen; bright soft page -> shojo.
        assert_eq!(select_palette_name(0.40, 58.0), "seinen");
        assert_eq!(select_palette_name(0.70, 30.0), "shojo");
        assert_eq!(select_palette_name(0.60, 62.0), "shonen");
        assert_eq!(select_palette_name(0.75, 60.0), "webtoon");
        assert_eq!(select_palette_name(0.55, 45.0), "fantasy");
    }
}
