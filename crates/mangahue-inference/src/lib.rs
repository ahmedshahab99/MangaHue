use anyhow::{anyhow, Context, Result};
use image::{GrayImage, Rgb, RgbImage};
use mangahue_core::{ColorRgb, EngineError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub version: String,
    pub model_type: String,
    pub input_size: u32,
    pub input_channels: u32,
    pub output_channels: u32,
    pub input_range: String,
    pub output_range: String,
    pub expects_lineart_inverted: bool,
    pub supports_style_vector: bool,
}

impl ModelMetadata {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read model metadata {}", path.display()))?;
        let metadata: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse model metadata {}", path.display()))?;
        metadata.validate()?;
        Ok(metadata)
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_size == 0 {
            return Err(anyhow!("model input_size must be greater than zero"));
        }
        if !matches!(self.input_channels, 1 | 5) {
            return Err(anyhow!("model input_channels must be 1 or 5"));
        }
        if self.output_channels != 3 {
            return Err(anyhow!("model output_channels must be 3"));
        }
        if self.input_range != "0_1" || self.output_range != "0_1" {
            return Err(anyhow!("only 0_1 tensor range is currently supported"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub lineart: GrayImage,
    pub hint_map: RgbImage,
    pub hint_mask: GrayImage,
}

pub trait ColorizationBackend: Send + Sync {
    fn name(&self) -> &str;
    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError>;
}

#[derive(Debug, Clone)]
pub struct MockColorizationBackend {
    palette: Vec<ColorRgb>,
}

impl Default for MockColorizationBackend {
    fn default() -> Self {
        Self {
            palette: vec![
                ColorRgb::new(238, 177, 143),
                ColorRgb::new(82, 124, 206),
                ColorRgb::new(241, 205, 91),
                ColorRgb::new(87, 174, 132),
            ],
        }
    }
}

impl MockColorizationBackend {
    pub fn new(palette: Vec<ColorRgb>) -> Self {
        Self { palette }
    }
}

impl ColorizationBackend for MockColorizationBackend {
    fn name(&self) -> &str {
        "mock-hint-preview"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let width = input.lineart.width();
        let height = input.lineart.height();
        if input.hint_map.dimensions() != (width, height)
            || input.hint_mask.dimensions() != (width, height)
        {
            return Err(EngineError::DimensionMismatch(
                "hint maps must match line art tile dimensions".to_string(),
            ));
        }

        let mut output = RgbImage::new(width, height);
        let denom = (width + height).max(1) as f32;
        for y in 0..height {
            for x in 0..width {
                let line = input.lineart.get_pixel(x, y)[0] as f32 / 255.0;
                let palette_index = ((x / 96 + y / 96) as usize) % self.palette.len().max(1);
                let base = self
                    .palette
                    .get(palette_index)
                    .copied()
                    .unwrap_or(ColorRgb::new(190, 160, 140));
                let gradient = (x + y) as f32 / denom;
                let ambient = ColorRgb::new(
                    shift(base.r, 28.0 * (gradient - 0.5)),
                    shift(base.g, 18.0 * (0.5 - gradient)),
                    shift(base.b, 22.0 * (gradient - 0.25)),
                );
                let hint_strength = input.hint_mask.get_pixel(x, y)[0] as f32 / 255.0;
                let hint = input.hint_map.get_pixel(x, y);
                let hinted = ColorRgb::new(hint[0], hint[1], hint[2]);
                let mixed = ambient.mix(hinted, hint_strength);
                let paper_shadow = 0.72 + 0.28 * line;
                output.put_pixel(
                    x,
                    y,
                    Rgb([
                        (mixed.r as f32 * paper_shadow).round().clamp(0.0, 255.0) as u8,
                        (mixed.g as f32 * paper_shadow).round().clamp(0.0, 255.0) as u8,
                        (mixed.b as f32 * paper_shadow).round().clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct OnnxColorizationBackend {
    model_path: PathBuf,
    metadata: ModelMetadata,
    metadata_path: PathBuf,
    python: String,
}

impl OnnxColorizationBackend {
    pub fn new(model_path: impl AsRef<Path>, metadata_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(anyhow!(
                "model path does not exist: {}",
                model_path.display()
            ));
        }
        let metadata_path = metadata_path.as_ref();
        let metadata = ModelMetadata::load(metadata_path)?;
        Ok(Self {
            model_path: model_path.to_path_buf(),
            metadata_path: metadata_path.to_path_buf(),
            metadata,
            python: std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string()),
        })
    }
}

impl ColorizationBackend for OnnxColorizationBackend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        if input.hint_map.dimensions() != input.lineart.dimensions()
            || input.hint_mask.dimensions() != input.lineart.dimensions()
        {
            return Err(EngineError::DimensionMismatch(
                "ONNX input maps must share dimensions".to_string(),
            ));
        }
        let temp = tempfile::tempdir().map_err(|error| {
            EngineError::Inference(format!("failed to create temp dir: {error}"))
        })?;
        let lineart_path = temp.path().join("lineart.png");
        let hint_map_path = temp.path().join("hint_map.png");
        let hint_mask_path = temp.path().join("hint_mask.png");
        let output_path = temp.path().join("output.png");
        input.lineart.save(&lineart_path).map_err(|error| {
            EngineError::Inference(format!("failed to save lineart tile: {error}"))
        })?;
        input.hint_map.save(&hint_map_path).map_err(|error| {
            EngineError::Inference(format!("failed to save hint map tile: {error}"))
        })?;
        input.hint_mask.save(&hint_mask_path).map_err(|error| {
            EngineError::Inference(format!("failed to save hint mask tile: {error}"))
        })?;

        let status = Command::new(&self.python)
            .arg("-m")
            .arg("mangahue_train.run_onnx_tile")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--metadata")
            .arg(&self.metadata_path)
            .arg("--lineart")
            .arg(&lineart_path)
            .arg("--hint-map")
            .arg(&hint_map_path)
            .arg("--hint-mask")
            .arg(&hint_mask_path)
            .arg("--output")
            .arg(&output_path)
            .output()
            .map_err(|error| {
                EngineError::Inference(format!(
                    "failed to start Python ONNX backend with '{}': {error}. Install python package from ./python or set MANGAHUE_PYTHON.",
                    self.python
                ))
            })?;

        if !status.status.success() {
            let stderr = String::from_utf8_lossy(&status.stderr);
            let stdout = String::from_utf8_lossy(&status.stdout);
            return Err(EngineError::Inference(format!(
                "Python ONNX backend failed for {} ({} v{}): {}\n{}",
                self.model_path.display(),
                self.metadata.model_name,
                self.metadata.version,
                stderr.trim(),
                stdout.trim()
            )));
        }

        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| EngineError::Inference(format!("failed to read ONNX output: {error}")))
    }
}

fn shift(value: u8, delta: f32) -> u8 {
    (value as f32 + delta).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    #[test]
    fn mock_backend_respects_dimensions() {
        let lineart = GrayImage::from_pixel(16, 12, Luma([255]));
        let hint_map = RgbImage::from_pixel(16, 12, Rgb([0, 0, 0]));
        let hint_mask = GrayImage::from_pixel(16, 12, Luma([0]));
        let input = ModelInput {
            lineart,
            hint_map,
            hint_mask,
        };
        let output = MockColorizationBackend::default().colorize(&input).unwrap();
        assert_eq!(output.dimensions(), (16, 12));
    }
}
