use anyhow::{anyhow, Context, Result};
use image::{GrayImage, Rgb, RgbImage};
use mangahue_core::{ColorRgb, EngineError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SUPPORTED_CONTRACT_ID: &str = "mangahue.colorizer.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContractMetadata {
    pub contract_id: String,
    pub input_name: String,
    pub output_name: String,
    pub input_channels: Vec<String>,
    pub output_channels: Vec<String>,
    pub input_range: String,
    pub output_range: String,
    pub color_space: String,
    pub lineart_inverted: bool,
    pub dynamic_spatial_axes: bool,
    pub spatial_multiple: u32,
}

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
    #[serde(default)]
    pub contract: Option<ModelContractMetadata>,
    #[serde(default)]
    pub onnx_sha256: Option<String>,
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
        if let Some(contract) = &self.contract {
            let expected_inputs = ["lineart", "hint_r", "hint_g", "hint_b", "hint_mask"];
            let expected_outputs = ["srgb_r", "srgb_g", "srgb_b"];
            if contract.contract_id != SUPPORTED_CONTRACT_ID {
                return Err(anyhow!(
                    "unsupported model contract: {}",
                    contract.contract_id
                ));
            }
            if contract.input_name != "image_input" || contract.output_name != "color_output" {
                return Err(anyhow!(
                    "model tensor names do not match the runtime contract"
                ));
            }
            if contract
                .input_channels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != expected_inputs
                || contract
                    .output_channels
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    != expected_outputs
            {
                return Err(anyhow!(
                    "model channel order does not match the runtime contract"
                ));
            }
            if contract.color_space != "srgb" || contract.spatial_multiple == 0 {
                return Err(anyhow!("unsupported model color space or spatial multiple"));
            }
            if self.input_channels as usize != contract.input_channels.len()
                || self.output_channels as usize != contract.output_channels.len()
            {
                return Err(anyhow!(
                    "legacy metadata dimensions disagree with model contract"
                ));
            }
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
    #[allow(dead_code)]
    palette: Vec<ColorRgb>,
    ambient: ColorRgb,
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
            ambient: ColorRgb::new(185, 163, 147),
        }
    }
}

impl MockColorizationBackend {
    pub fn new(palette: Vec<ColorRgb>) -> Self {
        Self {
            palette,
            ambient: ColorRgb::new(185, 163, 147),
        }
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
        for y in 0..height {
            for x in 0..width {
                let line_value = input.lineart.get_pixel(x, y)[0] as f32;
                let line = line_value / 255.0;
                let hint_strength = input.hint_mask.get_pixel(x, y)[0] as f32 / 255.0;
                let hint = input.hint_map.get_pixel(x, y);
                let hinted = ColorRgb::new(hint[0], hint[1], hint[2]);
                let color = if hint_strength > 0.0 {
                    hinted
                } else {
                    self.ambient
                };
                let line_darken = if line_value < 50.0 { 0.75 } else { 1.0 };
                let paper_shadow = 0.72 + 0.28 * line;
                output.put_pixel(
                    x,
                    y,
                    Rgb([
                        (color.r as f32 * paper_shadow * line_darken)
                            .round()
                            .clamp(0.0, 255.0) as u8,
                        (color.g as f32 * paper_shadow * line_darken)
                            .round()
                            .clamp(0.0, 255.0) as u8,
                        (color.b as f32 * paper_shadow * line_darken)
                            .round()
                            .clamp(0.0, 255.0) as u8,
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
        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let status = std::process::Command::new(&python)
            .arg("--version")
            .output()
            .map_err(|e| {
                anyhow!(
                    "Python executable '{}' not found or not runnable: {}. Set MANGAHUE_PYTHON.",
                    python,
                    e
                )
            })?;
        if !status.status.success() {
            return Err(anyhow!(
                "Python executable '{}' failed version check",
                python
            ));
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            metadata_path: metadata_path.to_path_buf(),
            metadata,
            python,
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

#[cfg(feature = "native-onnx")]
use ort::{session::Session, value::Tensor};

#[cfg(feature = "native-onnx")]
use std::sync::Mutex;

#[cfg(feature = "native-onnx")]
#[derive(Debug)]
pub struct RustOnnxBackend {
    session: Mutex<Session>,
    metadata: ModelMetadata,
}

#[cfg(feature = "native-onnx")]
impl RustOnnxBackend {
    pub fn new(model_path: impl AsRef<Path>, metadata_path: impl AsRef<Path>) -> Result<Self> {
        let metadata = ModelMetadata::load(metadata_path)?;
        let model_path = model_path.as_ref();
        let session = Session::builder()
            .map_err(|e| anyhow!("failed to create ONNX session: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("failed to load ONNX model {}: {e}", model_path.display()))?;
        Ok(Self {
            session: Mutex::new(session),
            metadata,
        })
    }
}

#[cfg(feature = "native-onnx")]
impl ColorizationBackend for RustOnnxBackend {
    fn name(&self) -> &str {
        "rust-onnx"
    }
    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let h = input.lineart.height() as usize;
        let w = input.lineart.width() as usize;
        let channels = self.metadata.input_channels as usize;

        let mut tensor_data = Vec::with_capacity(1 * channels * h * w);
        for y in 0..h {
            for x in 0..w {
                tensor_data.push(input.lineart.get_pixel(x as u32, y as u32)[0] as f32 / 255.0);
            }
        }
        if channels == 5 {
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        tensor_data
                            .push(input.hint_map.get_pixel(x as u32, y as u32)[c] as f32 / 255.0);
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    tensor_data
                        .push(input.hint_mask.get_pixel(x as u32, y as u32)[0] as f32 / 255.0);
                }
            }
        }

        let array = ndarray::Array4::from_shape_vec((1, channels, h, w), tensor_data)
            .map_err(|e| EngineError::Inference(format!("tensor shape error: {e}")))?;
        let tensor = Tensor::from_array(array)
            .map_err(|e| EngineError::Inference(format!("failed to create tensor: {e}")))?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| EngineError::Inference(format!("session lock error: {e}")))?;
        let outputs = session
            .run(ort::inputs!["image_input" => tensor])
            .map_err(|e| EngineError::Inference(format!("ONNX inference failed: {e}")))?;
        let (_, output_data) = outputs["color_output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError::Inference(format!("output extraction failed: {e}")))?;

        let mut image = RgbImage::new(w as u32, h as u32);
        let plane_size = h * w;
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let r = (output_data[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (output_data[plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (output_data[2 * plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        Ok(image)
    }
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

    #[test]
    fn model_contract_rejects_reordered_channels() {
        let mut metadata = ModelMetadata {
            model_name: "test".into(),
            version: "1".into(),
            model_type: "hint_unet".into(),
            input_size: 768,
            input_channels: 5,
            output_channels: 3,
            input_range: "0_1".into(),
            output_range: "0_1".into(),
            expects_lineart_inverted: false,
            supports_style_vector: false,
            contract: Some(ModelContractMetadata {
                contract_id: SUPPORTED_CONTRACT_ID.into(),
                input_name: "image_input".into(),
                output_name: "color_output".into(),
                input_channels: ["lineart", "hint_r", "hint_g", "hint_b", "hint_mask"]
                    .map(str::to_string)
                    .to_vec(),
                output_channels: ["srgb_r", "srgb_g", "srgb_b"].map(str::to_string).to_vec(),
                input_range: "0_1".into(),
                output_range: "0_1".into(),
                color_space: "srgb".into(),
                lineart_inverted: false,
                dynamic_spatial_axes: true,
                spatial_multiple: 8,
            }),
            onnx_sha256: None,
        };
        assert!(metadata.validate().is_ok());
        metadata
            .contract
            .as_mut()
            .unwrap()
            .input_channels
            .swap(0, 1);
        assert!(metadata.validate().is_err());
    }
}
