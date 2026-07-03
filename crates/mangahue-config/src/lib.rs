use anyhow::{anyhow, Context, Result};
use mangahue_core::StyleProfile;
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMode {
    Fast,
    Balanced,
    Quality,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingDirection {
    LeftToRight,
    RightToLeft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSection {
    pub mode: EngineMode,
    pub preserve_lines: bool,
    pub protect_speech_bubbles: bool,
    pub save_debug_outputs: bool,
    pub reading_direction: ReadingDirection,
    pub parallel_jobs: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessSection {
    pub restore_lines: bool,
    pub line_strength: f32,
    pub bubble_cleanup: bool,
    pub bleeding_correction: bool,
    pub palette_correction: bool,
    pub final_sharpening: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
    pub format: String,
    pub quality: u8,
    pub write_report: bool,
    pub write_intermediates: bool,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            engine: EngineSection {
                mode: EngineMode::Balanced,
                preserve_lines: true,
                protect_speech_bubbles: true,
                save_debug_outputs: false,
                reading_direction: ReadingDirection::RightToLeft,
                parallel_jobs: 0,
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
            },
            output: OutputSection {
                format: "png".to_string(),
                quality: 95,
                write_report: true,
                write_intermediates: false,
            },
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
        if self.engine.parallel_jobs > 256 {
            return Err(anyhow!("engine.parallel_jobs must be <= 256"));
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
}
