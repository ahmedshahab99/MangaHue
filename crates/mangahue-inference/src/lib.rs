use anyhow::{anyhow, Context, Result};
use image::{GrayImage, Rgb, RgbImage};
use mangahue_core::{ColorRgb, EngineError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

const SUPPORTED_CONTRACT_ID: &str = "mangahue.colorizer.v1";
const DEFAULT_WORKER_TIMEOUT_SECONDS: u64 = 180;
const MIN_WORKER_TIMEOUT_SECONDS: u64 = 5;
const MAX_WORKER_TIMEOUT_SECONDS: u64 = 1_800;

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
            if contract.input_name.trim().is_empty() || contract.output_name.trim().is_empty() {
                return Err(anyhow!("model tensor names cannot be empty"));
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

    /// Whether this backend should receive the original grayscale page
    /// (with screentones/shading preserved) instead of the whitened
    /// `clean_lineart`. DDColor-style models colorize from real tone values
    /// and produce near-zero chroma on flattened white tiles.
    fn prefers_original_grayscale(&self) -> bool {
        false
    }
}

impl<T: ColorizationBackend + ?Sized> ColorizationBackend for Arc<T> {
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        self.as_ref().colorize(input)
    }

    fn prefers_original_grayscale(&self) -> bool {
        self.as_ref().prefers_original_grayscale()
    }
}

pub fn is_ddcolor_model_dir(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    // Accept both checkpoint naming conventions (.pt official, .bin HF-style).
    path.join("pytorch_model.pt").is_file()
        || path.join("pytorch_model.bin").is_file()
        || path.join("models").is_dir()
}

pub fn is_ddcolor_onnx_model(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.is_file()
        && path.extension().and_then(|value| value.to_str()) == Some("onnx")
        && path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains("ddcolor"))
}

pub fn is_reference_cgan_model_dir(path: impl AsRef<Path>) -> bool {
    path.as_ref().join("074000_gray.pt").is_file()
}

pub fn is_openvino_studio_model_dir(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.join("unet").join("openvino_model.xml").is_file()
        && path
            .join("vae_encoder")
            .join("openvino_model.xml")
            .is_file()
        && path
            .join("vae_decoder")
            .join("openvino_model.xml")
            .is_file()
}

pub fn is_anime_face_model_dir(path: impl AsRef<Path>) -> bool {
    path.as_ref().join("lbpcascade_animeface.xml").is_file()
}

mod anime_face;
pub use anime_face::AnimeFaceBackend;

#[derive(Debug, Clone)]
struct ReferenceCganWorkerLaunch {
    python: String,
    model_path: PathBuf,
    reference_path: PathBuf,
    worker_threads: usize,
}

impl ReferenceCganWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let child = Command::new(&self.python)
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.reference_cgan_worker")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--reference")
            .arg(&self.reference_path)
            .arg("--size")
            .arg("256")
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to launch reference cGAN worker with '{}'",
                    self.python
                )
            })?;
        let mut process =
            attach_worker_process(child, "reference-cgan-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "reference cGAN worker")?;
        Ok(process)
    }
}

/// Bound the shared-worker registries: when a registry grows past the cap,
/// evict workers referenced ONLY by the registry (strong_count == 1). A
/// worker in active use has strong_count >= 2 (registry + caller), so live
/// workers are never evicted. Each evicted worker releases its subprocess
/// and its ~1-2GB working set — without this, repeated configuration or
/// reference changes accumulated processes until OOM.
fn evict_idle_workers<T>(workers: &mut HashMap<String, Arc<T>>, cap: usize) {
    if workers.len() >= cap {
        workers.retain(|_, worker| Arc::strong_count(worker) > 1);
    }
}

#[derive(Debug)]
pub struct ReferenceCganBackend {
    launch: ReferenceCganWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

impl ReferenceCganBackend {
    pub fn new(model_dir: impl AsRef<Path>, reference_path: impl AsRef<Path>) -> Result<Self> {
        let requested_model = model_dir.as_ref();
        let model_dir = requested_model.canonicalize().with_context(|| {
            format!(
                "cannot resolve reference cGAN model directory {}",
                requested_model.display()
            )
        })?;
        if !is_reference_cgan_model_dir(&model_dir) {
            anyhow::bail!(
                "reference cGAN checkpoint missing in {}",
                model_dir.display()
            );
        }
        let requested_reference = reference_path.as_ref();
        let reference_path = requested_reference.canonicalize().with_context(|| {
            format!(
                "cannot resolve reference style image {}",
                requested_reference.display()
            )
        })?;
        let python = std::env::var("MANGAHUE_REFERENCE_PYTHON")
            .or_else(|_| std::env::var("MANGAHUE_PYTHON"))
            .unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let launch = ReferenceCganWorkerLaunch {
            python,
            model_path: model_dir.join("074000_gray.pt"),
            reference_path,
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(
        model_dir: impl AsRef<Path>,
        reference_path: impl AsRef<Path>,
    ) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<ReferenceCganBackend>>>> =
            OnceLock::new();
        let canonical_model = model_dir.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve reference cGAN model {}",
                model_dir.as_ref().display()
            )
        })?;
        let canonical_reference = reference_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve reference image {}",
                reference_path.as_ref().display()
            )
        })?;
        // The reference CONTENT is part of the key: replacing the reference
        // file at the same path used to be ignored (stale-reference bug) and
        // each distinct key spawned a never-evicted 1-2GB worker process.
        let reference_fingerprint = reference_content_fingerprint(&canonical_reference)?;
        let key = format!(
            "{}\x00{}\x00{}",
            canonical_model.display(),
            canonical_reference.display(),
            reference_fingerprint
        );
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("reference cGAN registry poisoned: {error}"))?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical_model, &canonical_reference)?);
        workers.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut workers, 4);
        Ok(worker)
    }
}

/// Dependency-free content fingerprint (FNV-1a 64) of the reference image.
/// Keeps the shared-worker key stable for identical content while changing
/// when the file is replaced with different pixels.
fn reference_content_fingerprint(path: &Path) -> Result<u64> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read reference image {}", path.display()))?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in &bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

impl ColorizationBackend for ReferenceCganBackend {
    fn name(&self) -> &str {
        "reference-cgan"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let temp = tempfile::tempdir()
            .map_err(|error| EngineError::Inference(format!("reference cGAN temp dir: {error}")))?;
        let lineart_path = temp.path().join("lineart.png");
        let output_path = temp.path().join("output.png");
        input
            .lineart
            .save(&lineart_path)
            .map_err(|error| EngineError::Inference(format!("reference cGAN save: {error}")))?;
        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "output": output_path,
        });
        let mut process = self
            .process
            .lock()
            .map_err(|error| EngineError::Inference(format!("reference cGAN lock: {error}")))?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "reference cGAN worker failed ({first_error}) and restart failed: {restart_error}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "reference cGAN worker failed after restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "reference cGAN failed: {}",
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error")
            )));
        }
        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| EngineError::Inference(format!("reference cGAN output: {error}")))
    }
}

#[derive(Debug, Clone)]
struct OpenVinoStudioWorkerLaunch {
    python: String,
    model_root: PathBuf,
    long_edge: u32,
    steps: u32,
    strength: f32,
    worker_threads: usize,
}

impl OpenVinoStudioWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let mut command = Command::new(&self.python);
        command
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.studio_openvino_worker")
            .arg("--model-root")
            .arg(&self.model_root)
            .arg("--long-edge")
            .arg(self.long_edge.to_string())
            .arg("--steps")
            .arg(self.steps.to_string())
            .arg("--strength")
            .arg(self.strength.to_string())
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .arg("--serve")
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        apply_worker_pythonpath(&mut command, &self.python);
        let child = command.spawn().with_context(|| {
            format!(
                "failed to launch OpenVINO Studio worker with '{}'",
                self.python
            )
        })?;
        let mut process =
            attach_worker_process(child, "openvino-studio-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "OpenVINO Studio worker")?;
        Ok(process)
    }
}

#[derive(Debug)]
pub struct OpenVinoStudioBackend {
    launch: OpenVinoStudioWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

impl OpenVinoStudioBackend {
    pub fn new(model_root: impl AsRef<Path>, input_size: u32) -> Result<Self> {
        let requested = model_root.as_ref();
        let model_root = requested.canonicalize().with_context(|| {
            format!(
                "cannot resolve OpenVINO Studio model {}",
                requested.display()
            )
        })?;
        if !is_openvino_studio_model_dir(&model_root) {
            anyhow::bail!(
                "OpenVINO Studio model directory is incomplete: {}",
                model_root.display()
            );
        }
        let python = std::env::var("MANGAHUE_STUDIO_PYTHON")
            .or_else(|_| std::env::var("MANGAHUE_PYTHON"))
            .unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let steps = std::env::var("MANGAHUE_STUDIO_STEPS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(4)
            .clamp(2, 12);
        let strength = std::env::var("MANGAHUE_STUDIO_STRENGTH")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.38)
            .clamp(0.05, 1.0);
        let long_edge = input_size.clamp(256, 768).next_multiple_of(64);
        let launch = OpenVinoStudioWorkerLaunch {
            python,
            model_root,
            long_edge,
            steps,
            strength,
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(model_root: impl AsRef<Path>, input_size: u32) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<OpenVinoStudioBackend>>>> =
            OnceLock::new();
        let canonical = model_root.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve OpenVINO Studio model {}",
                model_root.as_ref().display()
            )
        })?;
        let key = format!("{}|{}", canonical.display(), input_size);
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("OpenVINO Studio registry poisoned: {error}"))?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical, input_size)?);
        workers.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut workers, 1);
        Ok(worker)
    }
}

impl ColorizationBackend for OpenVinoStudioBackend {
    fn name(&self) -> &str {
        "openvino-lcm-studio"
    }

    fn prefers_original_grayscale(&self) -> bool {
        true
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let temp = tempfile::tempdir()
            .map_err(|error| EngineError::Inference(format!("Studio temp dir: {error}")))?;
        let lineart_path = temp.path().join("lineart.png");
        let output_path = temp.path().join("output.png");
        input
            .lineart
            .save(&lineart_path)
            .map_err(|error| EngineError::Inference(format!("Studio input save: {error}")))?;
        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "output": output_path,
        });
        let mut process = self
            .process
            .lock()
            .map_err(|error| EngineError::Inference(format!("Studio worker lock: {error}")))?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "Studio worker failed ({first_error}) and restart failed: {restart_error}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "Studio worker failed after restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "OpenVINO Studio failed: {}",
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error")
            )));
        }
        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| EngineError::Inference(format!("Studio output: {error}")))
    }
}

#[derive(Debug, Clone)]
struct DdColorOnnxWorkerLaunch {
    python: String,
    model_path: PathBuf,
    worker_threads: usize,
}

impl DdColorOnnxWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let mut command = Command::new(&self.python);
        command
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.ddcolor_onnx_worker")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        apply_worker_pythonpath(&mut command, &self.python);
        let child = command.spawn().with_context(|| {
            format!(
                "failed to launch DDColor ONNX worker with '{}'",
                self.python
            )
        })?;
        let mut process =
            attach_worker_process(child, "ddcolor-onnx-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "DDColor ONNX worker")?;
        Ok(process)
    }
}

#[derive(Debug)]
pub struct DdColorOnnxBackend {
    launch: DdColorOnnxWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

impl DdColorOnnxBackend {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let requested_path = model_path.as_ref();
        let model_path = requested_path.canonicalize().with_context(|| {
            format!(
                "cannot resolve DDColor ONNX model {}",
                requested_path.display()
            )
        })?;
        if !is_ddcolor_onnx_model(&model_path) {
            anyhow::bail!(
                "DDColor ONNX model name must contain 'ddcolor': {}",
                model_path.display()
            );
        }
        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let launch = DdColorOnnxWorkerLaunch {
            python,
            model_path,
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<DdColorOnnxBackend>>>> = OnceLock::new();
        let canonical = model_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve DDColor ONNX model {}",
                model_path.as_ref().display()
            )
        })?;
        let key = canonical.to_string_lossy().to_string();
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("DDColor ONNX registry poisoned: {error}"))?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical)?);
        workers.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut workers, 4);
        Ok(worker)
    }
}

impl ColorizationBackend for DdColorOnnxBackend {
    fn name(&self) -> &str {
        "ddcolor-tiny-onnx"
    }

    fn prefers_original_grayscale(&self) -> bool {
        true
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let temp = tempfile::tempdir()
            .map_err(|error| EngineError::Inference(format!("DDColor temp dir: {error}")))?;
        let lineart_path = temp.path().join("lineart.png");
        let output_path = temp.path().join("output.png");
        input
            .lineart
            .save(&lineart_path)
            .map_err(|error| EngineError::Inference(format!("DDColor save: {error}")))?;
        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "output": output_path,
        });
        let mut process = self
            .process
            .lock()
            .map_err(|error| EngineError::Inference(format!("DDColor lock: {error}")))?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "DDColor ONNX worker failed ({first_error}) and restart failed: {restart_error}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "DDColor ONNX worker failed after restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "DDColor ONNX failed: {}",
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error")
            )));
        }
        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| EngineError::Inference(format!("DDColor output: {error}")))
    }
}

#[derive(Debug, Clone)]
struct DdColorWorkerLaunch {
    python: String,
    model_path: PathBuf,
    device: String,
    input_size: u32,
    model_size: String,
    worker_threads: usize,
}

impl DdColorWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let mut command = Command::new(&self.python);
        command
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.ddcolor_worker")
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--device")
            .arg(&self.device)
            .arg("--input-size")
            .arg(self.input_size.to_string())
            .arg("--model-size")
            .arg(&self.model_size)
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2");
        apply_worker_pythonpath(&mut command, &self.python);
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to launch ddcolor worker with '{}'", self.python))?;
        let mut process =
            attach_worker_process(child, "ddcolor-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "ddcolor worker")?;
        Ok(process)
    }
}

/// Extend PYTHONPATH for worker subprocesses so mangahue_train and model
/// source repos (e.g. ddcolor_repo) resolve regardless of the current dir
/// or the interpreter's site-packages. Set MANGAHUE_PYTHONPATH (colon
/// separated) to add extra roots.
fn apply_worker_pythonpath(command: &mut Command, _python: &str) {
    let mut entries: Vec<String> = Vec::new();
    if let Ok(extra) = std::env::var("MANGAHUE_PYTHONPATH") {
        for part in extra.split(':') {
            if !part.is_empty() {
                entries.push(part.to_string());
            }
        }
    }
    if !entries.is_empty() {
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.is_empty() {
                entries.push(existing);
            }
        }
        command.env("PYTHONPATH", entries.join(":"));
    }
}

#[derive(Debug)]
pub struct DdColorBackend {
    launch: DdColorWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

impl DdColorBackend {
    pub fn new(
        model_dir: impl AsRef<Path>,
        device: &str,
        input_size: u32,
        model_size: &str,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_path = if model_dir.join("pytorch_model.pt").is_file() {
            model_dir.join("pytorch_model.pt")
        } else {
            let snapshots = model_dir.join("models");
            let mut found = None;
            if snapshots.is_dir() {
                for entry in std::fs::read_dir(&snapshots)
                    .with_context(|| format!("cannot read {}", snapshots.display()))?
                {
                    let entry = entry?;
                    let candidate = entry
                        .path()
                        .join("snapshots")
                        .join("master")
                        .join("pytorch_model.pt");
                    if candidate.is_file() {
                        found = Some(candidate);
                        break;
                    }
                }
            }
            found.context("ddcolor pytorch_model.pt not found in model directory")?
        };

        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let launch = DdColorWorkerLaunch {
            python,
            model_path,
            device: if device.eq_ignore_ascii_case("cuda") {
                "cuda".to_string()
            } else {
                "cpu".to_string()
            },
            input_size,
            model_size: model_size.to_string(),
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(
        model_dir: impl AsRef<Path>,
        device: &str,
        input_size: u32,
        model_size: &str,
    ) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<DdColorBackend>>>> = OnceLock::new();
        let canonical = model_dir.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve ddcolor dir {}",
                model_dir.as_ref().display()
            )
        })?;
        let key = format!(
            "{}|{}|{}|{}",
            canonical.display(),
            device.to_ascii_lowercase(),
            input_size,
            model_size
        );
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut w = workers
            .lock()
            .map_err(|e| anyhow!("ddcolor registry poisoned: {e}"))?;
        if let Some(worker) = w.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical, device, input_size, model_size)?);
        w.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut w, 4);
        Ok(worker)
    }
}

impl ColorizationBackend for DdColorBackend {
    fn name(&self) -> &str {
        "ddcolor"
    }

    fn prefers_original_grayscale(&self) -> bool {
        true
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let temp = tempfile::tempdir()
            .map_err(|e| EngineError::Inference(format!("ddcolor temp dir: {e}")))?;
        let lineart_path = temp.path().join("lineart.png");
        let output_path = temp.path().join("output.png");
        input
            .lineart
            .save(&lineart_path)
            .map_err(|e| EngineError::Inference(format!("ddcolor save: {e}")))?;
        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "output": output_path,
        });
        let mut process = self
            .process
            .lock()
            .map_err(|e| EngineError::Inference(format!("ddcolor lock: {e}")))?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(r) => r,
            Err(first) => {
                *process = self.launch.spawn().map_err(|restart| {
                    EngineError::Inference(format!(
                        "ddcolor worker failed ({first}) and restart failed: {restart}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second| {
                    EngineError::Inference(format!("ddcolor failed after restart: {second}"))
                })?
            }
        };
        if response.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "ddcolor failed: {}",
                response
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            )));
        }
        image::open(&output_path)
            .map(|i| i.to_rgb8())
            .map_err(|e| EngineError::Inference(format!("ddcolor output: {e}")))
    }
}

pub fn is_pretrained_model_dir(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.join("colorizator.py").is_file()
        && path.join("networks/generator.zip").is_file()
        && path.join("networks/extractor.pth").is_file()
        && path.join("denoising/models/net_rgb.pth").is_file()
}

#[derive(Debug)]
struct PretrainedWorkerProcess {
    child: Child,
    stdin: ChildStdin,
    responses: mpsc::Receiver<std::result::Result<String, String>>,
    timeout: Duration,
}

impl Drop for PretrainedWorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn worker_timeout_from(value: Option<&str>) -> Duration {
    let seconds = value
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_WORKER_TIMEOUT_SECONDS)
        .clamp(MIN_WORKER_TIMEOUT_SECONDS, MAX_WORKER_TIMEOUT_SECONDS);
    Duration::from_secs(seconds)
}

fn configured_worker_timeout() -> Duration {
    worker_timeout_from(
        std::env::var("MANGAHUE_MODEL_TIMEOUT_SECONDS")
            .ok()
            .as_deref(),
    )
}

fn attach_worker_process(
    mut child: Child,
    worker_name: &'static str,
    timeout: Duration,
) -> Result<PretrainedWorkerProcess> {
    let stdin = child
        .stdin
        .take()
        .with_context(|| format!("{worker_name} stdin unavailable"))?;
    let stdout = child
        .stdout
        .take()
        .with_context(|| format!("{worker_name} stdout unavailable"))?;
    let (sender, responses) = mpsc::sync_channel(1);
    thread::Builder::new()
        .name(format!("mangahue-{worker_name}-stdout"))
        .spawn(move || read_worker_stdout(stdout, sender))
        .with_context(|| format!("failed to start {worker_name} response reader"))?;
    Ok(PretrainedWorkerProcess {
        child,
        stdin,
        responses,
        timeout,
    })
}

fn read_worker_stdout(
    stdout: ChildStdout,
    sender: mpsc::SyncSender<std::result::Result<String, String>>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                let _ = sender.send(Err("worker stdout closed".to_string()));
                break;
            }
            Ok(_) => {
                if sender.send(Ok(line)).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = sender.send(Err(format!("failed to read worker response: {error}")));
                break;
            }
        }
    }
}

fn read_worker_response(
    process: &mut PretrainedWorkerProcess,
    phase: &str,
) -> std::result::Result<String, String> {
    match process.responses.recv_timeout(process.timeout) {
        Ok(response) => response,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = process.child.kill();
            Err(format!(
                "{phase} timed out after {} seconds; worker terminated",
                process.timeout.as_secs()
            ))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let status = process
                .child
                .try_wait()
                .ok()
                .flatten()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            Err(format!(
                "{phase} failed because worker response channel closed (status: {status})"
            ))
        }
    }
}

fn wait_for_worker_ready(process: &mut PretrainedWorkerProcess, worker_name: &str) -> Result<()> {
    let ready = read_worker_response(process, &format!("{worker_name} readiness handshake"))
        .map_err(anyhow::Error::msg)?;
    let handshake: serde_json::Value = serde_json::from_str(ready.trim())
        .with_context(|| format!("invalid {worker_name} handshake: {}", ready.trim()))?;
    if handshake.get("status").and_then(|value| value.as_str()) != Some("ready") {
        anyhow::bail!("{worker_name} did not become ready: {handshake}");
    }
    Ok(())
}

#[derive(Debug)]
pub struct PretrainedWorkerBackend {
    launch: PretrainedWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

#[derive(Debug, Clone)]
struct PretrainedWorkerLaunch {
    python: String,
    model_root: PathBuf,
    device: String,
    input_size: u32,
    worker_threads: usize,
}

impl PretrainedWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let child = Command::new(&self.python)
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.pretrained_worker")
            .arg("--model-root")
            .arg(&self.model_root)
            .arg("--device")
            .arg(&self.device)
            .arg("--size")
            .arg(self.input_size.to_string())
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to launch pretrained worker with '{}'; install MangaHue's Python package and model dependencies",
                    self.python
                )
            })?;
        let mut process =
            attach_worker_process(child, "pretrained-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "pretrained worker")?;
        Ok(process)
    }
}

fn exchange_worker_request(
    process: &mut PretrainedWorkerProcess,
    request: &serde_json::Value,
) -> std::result::Result<serde_json::Value, String> {
    serde_json::to_writer(&mut process.stdin, request)
        .map_err(|error| format!("failed to write pretrained request: {error}"))?;
    process
        .stdin
        .write_all(b"\n")
        .map_err(|error| format!("failed to terminate pretrained request: {error}"))?;
    process
        .stdin
        .flush()
        .map_err(|error| format!("failed to flush pretrained request: {error}"))?;
    let response_line = read_worker_response(process, "model inference")?;
    if response_line.is_empty() {
        return Err("pretrained worker exited without a response".to_string());
    }
    serde_json::from_str(response_line.trim())
        .map_err(|error| format!("invalid pretrained response: {error}: {response_line}"))
}

impl PretrainedWorkerBackend {
    pub fn new(model_root: impl AsRef<Path>, device: &str, input_size: u32) -> Result<Self> {
        let model_root = model_root.as_ref();
        if !is_pretrained_model_dir(model_root) {
            return Err(anyhow!(
                "pretrained model directory is incomplete: {}",
                model_root.display()
            ));
        }
        if input_size < 256 || !input_size.is_multiple_of(32) {
            return Err(anyhow!(
                "pretrained input size must be at least 256 and divisible by 32"
            ));
        }
        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let launch = PretrainedWorkerLaunch {
            python,
            model_root: model_root.to_path_buf(),
            device: if device.eq_ignore_ascii_case("cuda") {
                "cuda".to_string()
            } else {
                "cpu".to_string()
            },
            input_size,
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(
        model_root: impl AsRef<Path>,
        device: &str,
        input_size: u32,
    ) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<PretrainedWorkerBackend>>>> =
            OnceLock::new();
        let model_root = model_root.as_ref();
        let canonical = model_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve pretrained model directory {}",
                model_root.display()
            )
        })?;
        let key = format!(
            "{}|{}|{}",
            canonical.display(),
            device.to_ascii_lowercase(),
            input_size
        );
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("pretrained worker registry poisoned: {error}"))?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical, device, input_size)?);
        workers.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut workers, 4);
        Ok(worker)
    }
}

impl ColorizationBackend for PretrainedWorkerBackend {
    fn name(&self) -> &str {
        "pretrained-manga-colorization-v2"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        if input.hint_map.dimensions() != input.lineart.dimensions()
            || input.hint_mask.dimensions() != input.lineart.dimensions()
        {
            return Err(EngineError::DimensionMismatch(
                "pretrained input maps must share dimensions".to_string(),
            ));
        }
        let temp = tempfile::tempdir().map_err(|error| {
            EngineError::Inference(format!(
                "failed to create pretrained worker temp dir: {error}"
            ))
        })?;
        let lineart_path = temp.path().join("lineart.png");
        let hint_map_path = temp.path().join("hint_map.png");
        let hint_mask_path = temp.path().join("hint_mask.png");
        let output_path = temp.path().join("output.png");
        input.lineart.save(&lineart_path).map_err(|error| {
            EngineError::Inference(format!("failed to save pretrained lineart tile: {error}"))
        })?;
        input.hint_map.save(&hint_map_path).map_err(|error| {
            EngineError::Inference(format!("failed to save pretrained hint tile: {error}"))
        })?;
        input.hint_mask.save(&hint_mask_path).map_err(|error| {
            EngineError::Inference(format!("failed to save pretrained hint mask: {error}"))
        })?;

        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "hint_map": hint_map_path,
            "hint_mask": hint_mask_path,
            "output": output_path,
        });
        let mut process = self.process.lock().map_err(|error| {
            EngineError::Inference(format!("pretrained worker lock poisoned: {error}"))
        })?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "pretrained worker failed ({first_error}) and restart failed: {restart_error:#}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "pretrained worker failed after one restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "pretrained worker failed: {}",
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error")
            )));
        }
        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| {
                EngineError::Inference(format!("failed to read pretrained output: {error}"))
            })
    }
}

pub fn is_manga_ninja_model_dir(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.join("checkpoints")
        .join("StableDiffusion")
        .join("vae")
        .is_dir()
        && path.join("checkpoints").join("MangaNinjia").is_dir()
}

#[derive(Debug, Clone)]
struct MangaNinjaWorkerLaunch {
    python: String,
    model_root: PathBuf,
    device: String,
    denoise_steps: u32,
    worker_threads: usize,
}

impl MangaNinjaWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let child = Command::new(&self.python)
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.manga_ninja_worker")
            .arg("--model-root")
            .arg(&self.model_root)
            .arg("--device")
            .arg(&self.device)
            .arg("--denoise-steps")
            .arg(self.denoise_steps.to_string())
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to launch manga-ninja worker with '{}'; install MangaHue's Python package and model dependencies",
                    self.python
                )
            })?;
        let mut process =
            attach_worker_process(child, "manga-ninja-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "manga-ninja worker")?;
        Ok(process)
    }
}

#[derive(Debug)]
pub struct MangaNinjaBackend {
    launch: MangaNinjaWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

impl MangaNinjaBackend {
    pub fn new(model_root: impl AsRef<Path>, device: &str, denoise_steps: u32) -> Result<Self> {
        let model_root = model_root.as_ref();
        if !is_manga_ninja_model_dir(model_root) {
            return Err(anyhow!(
                "manga-ninja model directory is incomplete: {}",
                model_root.display()
            ));
        }
        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let launch = MangaNinjaWorkerLaunch {
            python,
            model_root: model_root.to_path_buf(),
            device: if device.eq_ignore_ascii_case("cuda") {
                "cuda".to_string()
            } else {
                "cpu".to_string()
            },
            denoise_steps,
            worker_threads,
        };
        let process = launch.spawn()?;
        Ok(Self {
            launch,
            process: Mutex::new(process),
        })
    }

    pub fn shared(
        model_root: impl AsRef<Path>,
        device: &str,
        denoise_steps: u32,
    ) -> Result<Arc<Self>> {
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<MangaNinjaBackend>>>> = OnceLock::new();
        let model_root = model_root.as_ref();
        let canonical = model_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve manga-ninja model directory {}",
                model_root.display()
            )
        })?;
        let key = format!(
            "{}|{}|{}",
            canonical.display(),
            device.to_ascii_lowercase(),
            denoise_steps
        );
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("manga-ninja worker registry poisoned: {error}"))?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical, device, denoise_steps)?);
        workers.insert(key, Arc::clone(&worker));
        evict_idle_workers(&mut workers, 4);
        Ok(worker)
    }
}

impl ColorizationBackend for MangaNinjaBackend {
    fn name(&self) -> &str {
        "manga-ninja"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        if input.hint_map.dimensions() != input.lineart.dimensions()
            || input.hint_mask.dimensions() != input.lineart.dimensions()
        {
            return Err(EngineError::DimensionMismatch(
                "manga-ninja input maps must share dimensions".to_string(),
            ));
        }
        let temp = tempfile::tempdir().map_err(|error| {
            EngineError::Inference(format!(
                "failed to create manga-ninja worker temp dir: {error}"
            ))
        })?;
        let lineart_path = temp.path().join("lineart.png");
        let hint_map_path = temp.path().join("hint_map.png");
        let hint_mask_path = temp.path().join("hint_mask.png");
        let output_path = temp.path().join("output.png");
        input.lineart.save(&lineart_path).map_err(|error| {
            EngineError::Inference(format!("failed to save manga-ninja lineart tile: {error}"))
        })?;
        input.hint_map.save(&hint_map_path).map_err(|error| {
            EngineError::Inference(format!("failed to save manga-ninja hint tile: {error}"))
        })?;
        input.hint_mask.save(&hint_mask_path).map_err(|error| {
            EngineError::Inference(format!("failed to save manga-ninja hint mask: {error}"))
        })?;

        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "hint_map": hint_map_path,
            "hint_mask": hint_mask_path,
            "output": output_path,
        });
        let mut process = self.process.lock().map_err(|error| {
            EngineError::Inference(format!("manga-ninja worker lock poisoned: {error}"))
        })?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "manga-ninja worker failed ({first_error}) and restart failed: {restart_error:#}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "manga-ninja worker failed after one restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "manga-ninja worker failed: {}",
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error")
            )));
        }
        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| {
                EngineError::Inference(format!("failed to read manga-ninja output: {error}"))
            })
    }
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
struct OnnxWorkerLaunch {
    model_path: PathBuf,
    metadata_path: PathBuf,
    python: String,
    worker_threads: usize,
}

impl OnnxWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let mut command = Command::new(&self.python);
        command
            .arg("-u")
            .arg("-m")
            .arg("mangahue_train.onnx_worker")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--metadata")
            .arg(&self.metadata_path)
            .arg("--threads")
            .arg(self.worker_threads.to_string())
            .env("OMP_NUM_THREADS", self.worker_threads.to_string())
            .env("MKL_NUM_THREADS", self.worker_threads.to_string())
            .env("MALLOC_ARENA_MAX", "2");
        // CRITICAL: without this, mangahue_train is never importable and the
        // hint-U-Net ONNX backend always failed to spawn its worker.
        apply_worker_pythonpath(&mut command, &self.python);
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to launch persistent ONNX worker with '{}'",
                    self.python
                )
            })?;
        let mut process = attach_worker_process(child, "onnx-worker", configured_worker_timeout())?;
        wait_for_worker_ready(&mut process, "ONNX worker")?;
        Ok(process)
    }
}

#[derive(Debug)]
struct OnnxWorkerState {
    launch: OnnxWorkerLaunch,
    process: Mutex<PretrainedWorkerProcess>,
}

#[derive(Debug, Clone)]
pub struct OnnxColorizationBackend {
    metadata: ModelMetadata,
    worker: Arc<OnnxWorkerState>,
}

impl OnnxColorizationBackend {
    pub fn new(model_path: impl AsRef<Path>, metadata_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        if !model_path.is_file() {
            return Err(anyhow!(
                "model path does not exist: {}",
                model_path.display()
            ));
        }
        let metadata_path = metadata_path.as_ref();
        let metadata = ModelMetadata::load(metadata_path)?;
        let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
        let worker_threads = std::env::var("MANGAHUE_MODEL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        let model_path = model_path
            .canonicalize()
            .with_context(|| format!("failed to resolve ONNX model {}", model_path.display()))?;
        let metadata_path = metadata_path.canonicalize().with_context(|| {
            format!(
                "failed to resolve ONNX metadata {}",
                metadata_path.display()
            )
        })?;
        let key = format!(
            "{}|{}|{}|{}",
            model_path.display(),
            metadata_path.display(),
            python,
            worker_threads
        );
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<OnnxWorkerState>>>> = OnceLock::new();
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|error| anyhow!("ONNX worker registry poisoned: {error}"))?;
        let worker = if let Some(worker) = workers.get(&key) {
            Arc::clone(worker)
        } else {
            let launch = OnnxWorkerLaunch {
                model_path,
                metadata_path,
                python,
                worker_threads,
            };
            let process = launch.spawn()?;
            let worker = Arc::new(OnnxWorkerState {
                launch,
                process: Mutex::new(process),
            });
            workers.insert(key, Arc::clone(&worker));
            evict_idle_workers(&mut workers, 4);
            worker
        };
        if metadata.input_channels != 1 && metadata.input_channels != 5 {
            anyhow::bail!(
                "persistent ONNX backend supports 1 or 5 input channels, got {}",
                metadata.input_channels
            );
        }
        Ok(Self { metadata, worker })
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

        let request = serde_json::json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "lineart": lineart_path,
            "hint_map": hint_map_path,
            "hint_mask": hint_mask_path,
            "output": output_path,
        });
        let mut process = self.worker.process.lock().map_err(|error| {
            EngineError::Inference(format!("ONNX worker lock poisoned: {error}"))
        })?;
        let response = match exchange_worker_request(&mut process, &request) {
            Ok(response) => response,
            Err(first_error) => {
                *process = self.worker.launch.spawn().map_err(|restart_error| {
                    EngineError::Inference(format!(
                        "ONNX worker failed ({first_error}) and restart failed: {restart_error:#}"
                    ))
                })?;
                exchange_worker_request(&mut process, &request).map_err(|second_error| {
                    EngineError::Inference(format!(
                        "ONNX worker failed after one restart: {second_error}"
                    ))
                })?
            }
        };
        if response.get("ok").and_then(|value| value.as_bool()) != Some(true) {
            return Err(EngineError::Inference(format!(
                "persistent ONNX backend failed ({} v{}): {}",
                self.metadata.model_name,
                self.metadata.version,
                response
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown ONNX worker error")
            )));
        }

        image::open(&output_path)
            .map(|image| image.to_rgb8())
            .map_err(|error| EngineError::Inference(format!("failed to read ONNX output: {error}")))
    }
}

#[cfg(feature = "native-onnx")]
use ort::{session::Session, value::Tensor};

/// Build an ONNX session tuned for the host: graph optimizations on and a
/// bounded intra-op thread pool (2 threads matches the 2-core VPS and the
/// previous Python worker's settings; override with MANGAHUE_MODEL_THREADS).
#[cfg(feature = "native-onnx")]
fn native_session_builder() -> Result<ort::session::builder::SessionBuilder> {
    // ort 2.0.0-rc.12's `load-dynamic` error path re-enters the API OnceLock
    // and deadlocks the process (pykeio/ort#560/#604) whenever dlopen fails —
    // which happens in long-lived processes with many shared libraries
    // loaded (pykeio/ort#579). Pre-loading the library ourselves first makes
    // ort's dlopen a no-op success, so the failure path is unreachable.
    preload_onnx_runtime()?;
    let threads = std::env::var("MANGAHUE_MODEL_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8);
    let builder = Session::builder()
        .map_err(|e| anyhow!("failed to create ONNX session: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
        .map_err(|e| anyhow!("failed to set ONNX graph optimization level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("failed to set ONNX intra-op threads: {e}"))?;
    // Optional CUDA execution provider (config flip for a future GPU host):
    // MANGAHUE_CUDA=1 routes inference through the GPU build of onnxruntime
    // with CPU fallback for unsupported ops. Unset (the current 2-core VPS)
    // means pure CPU — no behavioral change.
    let builder = if std::env::var("MANGAHUE_CUDA").as_deref() == Ok("1") {
        builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(|e| anyhow!("failed to enable CUDA execution provider: {e}"))?
    } else {
        builder
    };
    Ok(builder)
}

/// Pre-load the ONNX Runtime shared library so `ort`'s `load-dynamic` backend
/// never hits its rc.12 failure path (which re-enters the API `OnceLock` and
/// deadlocks the process — pykeio/ort#560/#604).
///
/// `ort` resolves the library by the same name we load here (OR\_DYLIB\_PATH
/// when set, otherwise the platform default). Loading it first — and leaking
/// the handle so it stays resident — makes ort's own `dlopen` find it already
/// mapped: it returns the existing handle without re-running constructors and
/// cannot fail. Idempotent; safe to call from every entry point.
///
/// The loaded library is also version-checked before ort ever initializes:
/// Windows ships an old in-box onnxruntime (1.17) in System32 for Windows ML,
/// and ort 2.0.0-rc.12 with `api-24` deadlocks on it instead of erroring.
/// We fail fast with an actionable error instead.
#[cfg(feature = "native-onnx")]
pub fn preload_onnx_runtime() -> Result<()> {
    static PRELOADED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    PRELOADED
        .get_or_init(|| {
            let library = std::env::var("ORT_DYLIB_PATH")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| {
                    if cfg!(target_os = "windows") {
                        "onnxruntime.dll".to_string()
                    } else if cfg!(target_os = "macos") {
                        "libonnxruntime.dylib".to_string()
                    } else {
                        "libonnxruntime.so".to_string()
                    }
                });
            match unsafe { libloading::Library::new(&library) } {
                Ok(handle) => {
                    if let Err(reason) = check_onnx_runtime_version(&handle) {
                        tracing::error!(
                            library = %library,
                            error = %reason,
                            "rejected ONNX Runtime: {reason} (set ORT_DYLIB_PATH to a recent \
                             onnxruntime.dll, e.g. .mangahue/python-runtime/Lib/site-packages/\
                             onnxruntime/capi/onnxruntime.dll)"
                        );
                        return Err(format!(
                            "incompatible ONNX Runtime ({library}): {reason}. \
                             Set ORT_DYLIB_PATH to a recent onnxruntime (>= 1.23), e.g. \
                             .mangahue/python-runtime/Lib/site-packages/onnxruntime/capi/\
                             onnxruntime.dll"
                        ));
                    }
                    // Leak on purpose: the library must stay mapped for the
                    // whole process lifetime so ort's dlopen reuses it.
                    std::mem::forget(handle);
                    tracing::info!(library = %library, "pre-loaded ONNX Runtime");
                    Ok(())
                }
                Err(error) => {
                    tracing::error!(
                        library = %library,
                        error = %error,
                        "failed to pre-load ONNX Runtime; model inference is unavailable"
                    );
                    Err(format!(
                        "failed to pre-load ONNX Runtime ({library}): {error}"
                    ))
                }
            }
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// The C API level this crate's `ort` build was compiled against
/// (`api-24` feature): ort asks the runtime for exactly this version.
#[cfg(feature = "native-onnx")]
const ORT_API_LEVEL: u32 = 24;

/// Verify the pre-loaded ONNX Runtime actually implements the C API level
/// `ort` will request (`api-24`). ort 2.0.0-rc.12's error path deadlocks
/// instead of reporting, so this check must run before ort initializes.
///
/// `OrtApiBase` member order changed across releases (1.19+ put the
/// versioned `GetApi` first; older and Windows' in-box builds put
/// `GetVersionString` first), so the members are probed layout-agnostically:
/// each is called as `GetApi(ORT_API_LEVEL)`, and a member that returns a
/// version STRING instead of a vtable is recognized and skipped.
#[cfg(feature = "native-onnx")]
fn check_onnx_runtime_version(handle: &libloading::Library) -> Result<(), String> {
    unsafe {
        let get_api_base: libloading::Symbol<unsafe extern "C" fn() -> *const std::ffi::c_void> =
            handle
                .get(b"OrtGetApiBase\0")
                .map_err(|e| format!("OrtGetApiBase not exported: {e}"))?;
        let base = get_api_base() as *const usize;
        if base.is_null() {
            return Err("OrtGetApiBase returned null".to_string());
        }
        let mut found_version: Option<String> = None;
        // Probe the first two members; all known layouts place the relevant
        // entry points there.
        for idx in 0..2usize {
            let member = *base.add(idx);
            if member == 0 {
                continue;
            }
            let probe: unsafe extern "C" fn(u32) -> *const std::ffi::c_void =
                std::mem::transmute(member);
            let result = probe(ORT_API_LEVEL);
            if result.is_null() {
                continue;
            }
            if looks_like_version_string(result) {
                // This member is GetVersionString, not GetApi: remember the
                // version for diagnostics and keep probing.
                found_version = std::ffi::CStr::from_ptr(result as *const std::ffi::c_char)
                    .to_string_lossy()
                    .into_owned()
                    .into();
                continue;
            }
            // The runtime implements the API level: report its version too
            // (best effort — newer layouts put GetVersionString later).
            if found_version.is_none() {
                for extra in idx + 1..=3usize {
                    let member = *base.add(extra);
                    if member == 0 {
                        continue;
                    }
                    let version_probe: unsafe extern "C" fn() -> *const std::ffi::c_char =
                        std::mem::transmute(member);
                    let vp = version_probe();
                    if !vp.is_null() && looks_like_version_string(vp as *const std::ffi::c_void) {
                        found_version = std::ffi::CStr::from_ptr(vp)
                            .to_string_lossy()
                            .into_owned()
                            .into();
                        break;
                    }
                }
            }
            tracing::info!(
                version = %found_version.as_deref().unwrap_or("unknown"),
                api_level = ORT_API_LEVEL,
                "ONNX Runtime version check passed"
            );
            return Ok(());
        }
        let version = found_version
            .map(|v| format!("onnxruntime {v} "))
            .unwrap_or_default();
        Err(format!(
            "{version}does not implement C API level {ORT_API_LEVEL} (ort 2.0.0-rc.12 \
             deadlocks on such runtimes; Windows' in-box System32 onnxruntime 1.17 is a \
             common offender). Set ORT_DYLIB_PATH to a recent onnxruntime >= 1.23"
        ))
    }
}

/// A vtable pointer vs. a version string, for `OrtApiBase` member probing:
/// version strings are printable digits/dots ending in a NUL within 16
/// bytes; vtable pointers contain non-printable bytes (function addresses)
/// almost immediately.
#[cfg(feature = "native-onnx")]
fn looks_like_version_string(ptr: *const std::ffi::c_void) -> bool {
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, 16) };
    let mut len = 0usize;
    for byte in bytes {
        if *byte == 0 {
            break;
        }
        if !byte.is_ascii_digit() && *byte != b'.' {
            return false;
        }
        len += 1;
    }
    len >= 3 // a bare "1." or "0." is not a real version
}

#[cfg(feature = "native-onnx")]
#[derive(Debug)]
pub struct RustOnnxBackend {
    session: Mutex<Session>,
    metadata: ModelMetadata,
    input_name: String,
    output_name: String,
}

#[cfg(feature = "native-onnx")]
fn apply_purple_correction(image: &mut RgbImage, lineart: &GrayImage) {
    let (width, height) = image.dimensions();
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            let purple = ((r + b) / 2.0 - g).clamp(0.0, 1.0);
            let midtone: f32 = if (luma > 0.15) && (luma < 0.85) {
                1.0
            } else {
                0.0
            };
            let kill = purple * midtone * 0.3;

            let mut new_r: f32 = r - kill * 0.5;
            let mut new_g: f32 = g;
            let mut new_b: f32 = b - kill;

            let gray = lineart.get_pixel(x, y)[0] as f32 / 255.0;
            if gray > 0.92 {
                new_r = 1.0;
                new_g = 1.0;
                new_b = 1.0;
            } else if gray < 0.08 {
                new_r = 0.0;
                new_g = 0.0;
                new_b = 0.0;
            }

            let p = Rgb([
                (new_r.clamp(0.0, 1.0) * 255.0).round() as u8,
                (new_g.clamp(0.0, 1.0) * 255.0).round() as u8,
                (new_b.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
            *image.get_pixel_mut(x, y) = p;
        }
    }
}

#[cfg(feature = "native-onnx")]
impl RustOnnxBackend {
    pub fn new(model_path: impl AsRef<Path>, metadata_path: impl AsRef<Path>) -> Result<Self> {
        let metadata = ModelMetadata::load(metadata_path)?;
        let model_path = model_path.as_ref();
        let mut builder = native_session_builder()?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("failed to load ONNX model {}: {e}", model_path.display()))?;
        let input_name = metadata
            .contract
            .as_ref()
            .map(|c| c.input_name.clone())
            .unwrap_or_else(|| "image_input".to_string());
        let output_name = metadata
            .contract
            .as_ref()
            .map(|c| c.output_name.clone())
            .unwrap_or_else(|| "color_output".to_string());
        Ok(Self {
            session: Mutex::new(session),
            metadata,
            input_name,
            output_name,
        })
    }

    /// Shared session keyed by canonical (model, metadata) paths so a
    /// chapter never reloads + re-optimizes the model for every page.
    pub fn shared(
        model_path: impl AsRef<Path>,
        metadata_path: impl AsRef<Path>,
    ) -> Result<Arc<Self>> {
        let model_path = model_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve ONNX model {}",
                model_path.as_ref().display()
            )
        })?;
        let metadata_path = metadata_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve model metadata {}",
                metadata_path.as_ref().display()
            )
        })?;
        let key = format!(
            "{}::{}\n{}",
            model_path.display(),
            metadata_path.display(),
            std::env::var("MANGAHUE_MODEL_THREADS").unwrap_or_default()
        );
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<RustOnnxBackend>>>> = OnceLock::new();
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|e| anyhow!("session registry lock poisoned: {e}"))?;
        if let Some(backend) = workers.get(&key) {
            return Ok(Arc::clone(backend));
        }
        // Bound the registry: keep at most 8 sessions, drop the oldest.
        if workers.len() >= 8 {
            if let Some(oldest) = workers.keys().next().cloned() {
                workers.remove(&oldest);
            }
        }
        let backend = Arc::new(RustOnnxBackend::new(&model_path, &metadata_path)?);
        workers.insert(key, Arc::clone(&backend));
        Ok(backend)
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
        // The model contract requires spatial dims that are multiples of 32
        // (skip-connection concat). Pad like the Python worker did: lineart
        // padded with white (paper), hints/mask with zero, then crop.
        let pad_h = (32 - h % 32) % 32;
        let pad_w = (32 - w % 32) % 32;
        let th = h + pad_h;
        let tw = w + pad_w;
        let plane = th * tw;

        let mut tensor_data = vec![0.0f32; channels * plane];
        for y in 0..th {
            for x in 0..tw {
                let value = if y < h && x < w {
                    input.lineart.get_pixel(x as u32, y as u32)[0] as f32 / 255.0
                } else {
                    1.0 // white paper border
                };
                tensor_data[y * tw + x] = value;
            }
        }
        if channels == 5 {
            for c in 0..3 {
                let base = (1 + c) * plane;
                for y in 0..h {
                    for x in 0..w {
                        tensor_data[base + y * tw + x] =
                            input.hint_map.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                    }
                }
            }
            let base = 4 * plane;
            for y in 0..h {
                for x in 0..w {
                    tensor_data[base + y * tw + x] =
                        input.hint_mask.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                }
            }
        }

        let array = ndarray::Array4::from_shape_vec((1, channels, th, tw), tensor_data)
            .map_err(|e| EngineError::Inference(format!("tensor shape error: {e}")))?;
        let tensor = Tensor::from_array(array)
            .map_err(|e| EngineError::Inference(format!("failed to create tensor: {e}")))?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| EngineError::Inference(format!("session lock error: {e}")))?;
        let in_name = &self.input_name;
        let out_name = &self.output_name;
        let outputs = session
            .run(ort::inputs![in_name.clone() => tensor])
            .map_err(|e| EngineError::Inference(format!("ONNX inference failed: {e}")))?;
        let (_, output_data) = outputs[out_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError::Inference(format!("output extraction failed: {e}")))?;

        let mut image = RgbImage::new(w as u32, h as u32);
        let plane_size = th * tw;
        for y in 0..h {
            for x in 0..w {
                let idx = y * tw + x;
                let r = (output_data[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (output_data[plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (output_data[2 * plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        apply_purple_correction(&mut image, &input.lineart);
        Ok(image)
    }
}

/// Native Rust DDColor backend — a faithful port of the Python
/// `ddcolor_onnx_worker.py` sidecar so DDColor ONNX models run entirely
/// inside the process: no Python runtime, no IPC, no extra working set.
///
/// Contract (verified against the shipped ddcolor-*.metadata.json files):
///   input  "input"  [1, 3, 512, 512]  neutral sRGB (gray via CIE L*)
///   output "output" [1, 2, 512, 512]  CIE L*a*b* a/b planes (tanh or scaled)
#[cfg(feature = "native-onnx")]
pub struct NativeDdColorOnnxBackend {
    session: Mutex<Session>,
    ab_scale: f32,
    input_name: String,
    output_name: String,
}

/// Whether the path names a Style2Paints V3 ONNX model (name contains
/// "style2paints"). Routed to [`Style2PaintsV3OnnxBackend`].
#[cfg(feature = "native-onnx")]
pub fn is_style2paints_onnx_model(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.is_file()
        && path.extension().and_then(|value| value.to_str()) == Some("onnx")
        && path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains("style2paints"))
}

/// Minimal contract reader for DDColor/Style2Paints-style model metadata
/// (`{ input: {name, shape, range}, output: {name, shape, range} }`).
#[cfg(feature = "native-onnx")]
#[derive(Debug, Clone, serde::Deserialize)]
struct OnnxIoSpec {
    name: String,
    #[serde(default)]
    shape: Vec<i64>,
    #[serde(default)]
    color_space: Option<String>,
    #[serde(default)]
    range: Option<String>,
}

#[cfg(feature = "native-onnx")]
#[derive(Debug, Clone, serde::Deserialize)]
struct OnnxModelContract {
    #[serde(default)]
    input: Option<OnnxIoSpec>,
    #[serde(default)]
    output: Option<OnnxIoSpec>,
}

/// Native Rust Style2Paints V3 backend.
///
/// Contract (measured at export; verified against the exported graph):
///   input  "input"  [1, 3, 512, 512]  grayscale replicated to RGB, 0..1
///   output "output" [1, 3, 512, 512]  full sRGB (line art preserved by the
///                                      model — NO Lab merge on this path)
/// The contract is read from `<model>.metadata.json` when present (the file
/// the exporter emits), with "input"/"output" + 512 + 0..1 as fallbacks.
#[cfg(feature = "native-onnx")]
pub struct Style2PaintsV3OnnxBackend {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    input_size: u32,
    input_channels: usize,
    // (value - offset) * scale -> 0..1, per the metadata "range" field.
    offset: f32,
    scale: f32,
}

#[cfg(feature = "native-onnx")]
impl Style2PaintsV3OnnxBackend {
    fn load_contract(model_path: &Path) -> OnnxModelContract {
        let meta_path = model_path.with_extension("metadata.json");
        std::fs::read(&meta_path)
            .ok()
            .and_then(|data| serde_json::from_slice(&data).ok())
            .unwrap_or_else(|| {
                tracing::warn!(
                    metadata = %meta_path.display(),
                    "Style2Paints contract missing; using defaults (input/output, 512, 0..1)"
                );
                OnnxModelContract {
                    input: None,
                    output: None,
                }
            })
    }

    fn range_params(range: Option<&str>) -> (f32, f32) {
        match range.unwrap_or("").trim() {
            "-1_1" | "-1..1" | "neg1_1" => (1.0, -1.0), // value = x*2-1
            _ => (1.0, 0.0),
        }
    }

    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let mut builder = native_session_builder()?;
        let session = builder.commit_from_file(model_path).map_err(|e| {
            anyhow!(
                "failed to load Style2Paints ONNX model {}: {e}",
                model_path.display()
            )
        })?;
        let contract = Self::load_contract(model_path);
        let input = contract.input.unwrap_or(OnnxIoSpec {
            name: "input".to_string(),
            shape: vec![1, 3, 512, 512],
            color_space: None,
            range: None,
        });
        let output = contract.output.unwrap_or(OnnxIoSpec {
            name: "output".to_string(),
            shape: vec![1, 3, 512, 512],
            color_space: None,
            range: None,
        });
        let input_size = input.shape.get(2).copied().unwrap_or(512).max(1) as u32;
        let input_channels = input.shape.get(1).copied().unwrap_or(3).max(1) as usize;
        let (scale, offset) = Self::range_params(input.range.as_deref());
        Ok(Self {
            session: Mutex::new(session),
            input_name: input.name,
            output_name: output.name,
            input_size,
            input_channels,
            scale,
            offset,
        })
    }

    /// Shared session keyed by the canonical model path (contract parsed
    /// once, session reused across pages).
    pub fn shared(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let model_path = model_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve Style2Paints ONNX model {}",
                model_path.as_ref().display()
            )
        })?;
        let key = model_path.display().to_string();
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<Style2PaintsV3OnnxBackend>>>> =
            OnceLock::new();
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|e| anyhow!("session registry lock poisoned: {e}"))?;
        if let Some(backend) = workers.get(&key) {
            return Ok(Arc::clone(backend));
        }
        if workers.len() >= 8 {
            if let Some(oldest) = workers.keys().next().cloned() {
                workers.remove(&oldest);
            }
        }
        let backend = Arc::new(Style2PaintsV3OnnxBackend::new(&model_path)?);
        workers.insert(key, Arc::clone(&backend));
        Ok(backend)
    }
}

#[cfg(feature = "native-onnx")]
impl ColorizationBackend for Style2PaintsV3OnnxBackend {
    fn name(&self) -> &str {
        "style2paints-v3-native-rust"
    }

    fn prefers_original_grayscale(&self) -> bool {
        true
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let (w, h) = input.lineart.dimensions();
        let gray = input.lineart.clone();
        let size = self.input_size;
        let resized =
            image::imageops::resize(&gray, size, size, image::imageops::FilterType::Lanczos3);
        let n = (size * size) as usize;
        let channels = self.input_channels;
        let mut tensor_data = vec![0.0f32; channels * n];
        for y in 0..size {
            for x in 0..size {
                let g = resized.get_pixel(x, y)[0] as f32 / 255.0;
                // (value - offset) * scale -> 0..1, then map to the contract range.
                let value = g * self.scale + self.offset;
                let idx = (y * size + x) as usize;
                for c in 0..channels {
                    tensor_data[c * n + idx] = value;
                }
            }
        }
        let array = ndarray::Array4::from_shape_vec(
            (1, channels, size as usize, size as usize),
            tensor_data,
        )
        .map_err(|e| EngineError::Inference(format!("tensor shape error: {e}")))?;
        let tensor = Tensor::from_array(array)
            .map_err(|e| EngineError::Inference(format!("failed to create tensor: {e}")))?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| EngineError::Inference(format!("session lock error: {e}")))?;
        let in_name = self.input_name.clone();
        let out_name = self.output_name.clone();
        let outputs = session
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| EngineError::Inference(format!("Style2Paints inference failed: {e}")))?;
        let (_, output_data) = outputs[out_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError::Inference(format!("output extraction failed: {e}")))?;

        // Output [1, C, S, S] full RGB (0..1 after the inverse range map).
        let out_channels = output_data.len() / n;
        let (inv_scale, inv_offset) = (1.0 / self.scale, -self.offset / self.scale);
        let mut model_rgb = RgbImage::new(size, size);
        for i in 0..n {
            let r = if out_channels >= 3 {
                ((output_data[i] * inv_scale + inv_offset).clamp(0.0, 1.0) * 255.0).round() as u8
            } else {
                0
            };
            let g = if out_channels >= 3 {
                ((output_data[n + i] * inv_scale + inv_offset).clamp(0.0, 1.0) * 255.0).round()
                    as u8
            } else {
                ((output_data[i] * inv_scale + inv_offset).clamp(0.0, 1.0) * 255.0).round() as u8
            };
            let b = if out_channels >= 3 {
                ((output_data[2 * n + i] * inv_scale + inv_offset).clamp(0.0, 1.0) * 255.0).round()
                    as u8
            } else {
                ((output_data[i] * inv_scale + inv_offset).clamp(0.0, 1.0) * 255.0).round() as u8
            };
            model_rgb.put_pixel(
                (i % size as usize) as u32,
                (i / size as usize) as u32,
                Rgb([r, g, b]),
            );
        }
        // Back to the page size (Lanczos3 keeps the a/b upscale crisp; the
        // engine's restore_lines pass re-encodes the ink afterwards).
        let out = image::imageops::resize(&model_rgb, w, h, image::imageops::FilterType::Lanczos3);
        Ok(out)
    }
}

#[cfg(feature = "native-onnx")]
const DDCOLOR_MODEL_SIZE: u32 = 512;

#[cfg(feature = "native-onnx")]
const CIE_DELTA: f32 = 6.0 / 29.0;

#[cfg(feature = "native-onnx")]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(feature = "native-onnx")]
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// CIE 1931 L* from an sRGB pixel, matching the Python worker exactly.
#[cfg(feature = "native-onnx")]
fn rgb_to_lab_l(r: f32, g: f32, b: f32) -> f32 {
    let lr = srgb_to_linear(r);
    let lg = srgb_to_linear(g);
    let lb = srgb_to_linear(b);
    let y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
    let delta = CIE_DELTA;
    let f = if y > delta * delta * delta {
        y.cbrt()
    } else {
        y / (3.0 * delta * delta) + 4.0 / 29.0
    };
    116.0 * f - 16.0
}

#[cfg(feature = "native-onnx")]
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let delta = CIE_DELTA;
    let inv = |t: f32| -> f32 {
        if t > delta {
            t * t * t
        } else {
            3.0 * delta * delta * (t - 4.0 / 29.0)
        }
    };
    let x = inv(fx) * 0.95047;
    let y = inv(fy);
    let z = inv(fz) * 1.08883;
    let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
    (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(bl))
}

/// Saturating warmup input: raw tanh graphs report |ab| <= ~1; graphs with
/// the scale baked in report far above 2. Mirrors the Python calibration.
#[cfg(feature = "native-onnx")]
fn calibrate_ab_scale(
    session: &mut Session,
    input_name: &str,
    output_name: &str,
) -> std::result::Result<f32, EngineError> {
    let n = (DDCOLOR_MODEL_SIZE * DDCOLOR_MODEL_SIZE) as usize;
    let mut pattern = vec![0.0f32; 3 * n];
    let half = (DDCOLOR_MODEL_SIZE / 2) as usize;
    // channel 0 (R): top half red; channel 1 (G): bottom half green
    for i in 0..(half * DDCOLOR_MODEL_SIZE as usize) {
        pattern[i] = 1.0;
    }
    for i in (half * DDCOLOR_MODEL_SIZE as usize)..n {
        pattern[n + i] = 1.0;
    }
    let array = ndarray::Array4::from_shape_vec(
        (
            1,
            3,
            DDCOLOR_MODEL_SIZE as usize,
            DDCOLOR_MODEL_SIZE as usize,
        ),
        pattern,
    )
    .map_err(|e| EngineError::Inference(format!("calibration tensor shape error: {e}")))?;
    let tensor = Tensor::from_array(array)
        .map_err(|e| EngineError::Inference(format!("failed to create calibration tensor: {e}")))?;
    let outputs = session
        .run(ort::inputs![input_name.to_string() => tensor])
        .map_err(|e| EngineError::Inference(format!("calibration inference failed: {e}")))?;
    let (_, data) = outputs[output_name]
        .try_extract_tensor::<f32>()
        .map_err(|e| {
            EngineError::Inference(format!("calibration output extraction failed: {e}"))
        })?;
    let peak = data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    Ok(if peak <= 2.0 { 110.0 } else { 1.0 })
}

#[cfg(feature = "native-onnx")]
impl NativeDdColorOnnxBackend {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let mut builder = native_session_builder()?;
        let session = builder.commit_from_file(model_path.as_ref()).map_err(|e| {
            anyhow!(
                "failed to load DDColor ONNX model {}: {e}",
                model_path.as_ref().display()
            )
        })?;
        let mut session = session;
        let input_name = "input".to_string();
        let output_name = "output".to_string();
        let ab_scale = calibrate_ab_scale(&mut session, &input_name, &output_name)?;
        Ok(Self {
            session: Mutex::new(session),
            ab_scale,
            input_name,
            output_name,
        })
    }

    /// Shared session keyed by the canonical model path (calibration runs
    /// once, not once per page).
    pub fn shared(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let model_path = model_path.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve DDColor ONNX model {}",
                model_path.as_ref().display()
            )
        })?;
        let key = model_path.display().to_string();
        static WORKERS: OnceLock<Mutex<HashMap<String, Arc<NativeDdColorOnnxBackend>>>> =
            OnceLock::new();
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut workers = workers
            .lock()
            .map_err(|e| anyhow!("session registry lock poisoned: {e}"))?;
        if let Some(backend) = workers.get(&key) {
            return Ok(Arc::clone(backend));
        }
        if workers.len() >= 8 {
            if let Some(oldest) = workers.keys().next().cloned() {
                workers.remove(&oldest);
            }
        }
        let backend = Arc::new(NativeDdColorOnnxBackend::new(&model_path)?);
        workers.insert(key, Arc::clone(&backend));
        Ok(backend)
    }
}

#[cfg(feature = "native-onnx")]
impl ColorizationBackend for NativeDdColorOnnxBackend {
    fn name(&self) -> &str {
        "ddcolor-native-rust"
    }

    fn prefers_original_grayscale(&self) -> bool {
        true
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let (w, h) = input.lineart.dimensions();
        let gray = input.lineart.clone();

        // Resize to the model's native 512x512 (Lanczos3 approximates the
        // Python worker's bicubic), then build the neutral-RGB tensor.
        let resized = image::imageops::resize(
            &gray,
            DDCOLOR_MODEL_SIZE,
            DDCOLOR_MODEL_SIZE,
            image::imageops::FilterType::Lanczos3,
        );
        let n = (DDCOLOR_MODEL_SIZE * DDCOLOR_MODEL_SIZE) as usize;
        let mut tensor_data = Vec::with_capacity(3 * n);
        for _ in 0..3 {
            tensor_data.extend((0..n).map(|_| 0.0f32));
        }
        for y in 0..DDCOLOR_MODEL_SIZE {
            for x in 0..DDCOLOR_MODEL_SIZE {
                let g = resized.get_pixel(x, y)[0] as f32 / 255.0;
                let l = rgb_to_lab_l(g, g, g);
                let (nr, ng, nb) = lab_to_rgb(l, 0.0, 0.0);
                let idx = (y * DDCOLOR_MODEL_SIZE + x) as usize;
                tensor_data[idx] = nr;
                tensor_data[n + idx] = ng;
                tensor_data[2 * n + idx] = nb;
            }
        }
        let array = ndarray::Array4::from_shape_vec(
            (
                1,
                3,
                DDCOLOR_MODEL_SIZE as usize,
                DDCOLOR_MODEL_SIZE as usize,
            ),
            tensor_data,
        )
        .map_err(|e| EngineError::Inference(format!("tensor shape error: {e}")))?;
        let tensor = Tensor::from_array(array)
            .map_err(|e| EngineError::Inference(format!("failed to create tensor: {e}")))?;
        // Run the model inside a tight scope: the session mutex is released
        // as soon as the raw output is copied out, so the (expensive) edge-
        // aware upscale and Lab merge can overlap across parallel jobs
        // instead of serializing on the session.
        let (a_plane, b_plane) = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| EngineError::Inference(format!("session lock error: {e}")))?;
            let in_name = self.input_name.clone();
            let out_name = self.output_name.clone();
            let outputs = session
                .run(ort::inputs![in_name => tensor])
                .map_err(|e| EngineError::Inference(format!("DDColor inference failed: {e}")))?;
            let (_, output_data) = outputs[out_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| EngineError::Inference(format!("output extraction failed: {e}")))?;

            // a/b planes at model resolution, scaled per the calibrated
            // convention, kept in f32 (quantizing to u8 here would cost
            // half a Lab unit of precision in smooth gradients).
            let scale = self.ab_scale;
            if std::env::var("MANGAHUE_DEBUG_DDCOLOR").is_ok() {
                let total = 2 * n;
                let mean_abs: f32 =
                    output_data.iter().take(total).map(|v| v.abs()).sum::<f32>() / total as f32;
                let min = output_data
                    .iter()
                    .take(total)
                    .fold(f32::MAX, |a, &b| a.min(b));
                let max = output_data
                    .iter()
                    .take(total)
                    .fold(f32::MIN, |a, &b| a.max(b));
                eprintln!(
                    "[ddcolor-debug] ab_scale={scale} mean|ab|={mean_abs:.3} min={min:.3} max={max:.3}"
                );
            }
            let plane = n;
            let mut a_plane = vec![0.0f32; n];
            let mut b_plane = vec![0.0f32; n];
            for i in 0..n {
                a_plane[i] = (output_data[i] * scale).clamp(-127.0, 127.0);
                b_plane[i] = (output_data[plane + i] * scale).clamp(-127.0, 127.0);
            }
            (a_plane, b_plane)
        };

        // Edge-aware chroma upscale: plain bilinear smears color across
        // ink boundaries (hair strands, folds). A joint-bilateral
        // upscale guided by the full-res grayscale makes chroma follow
        // the lines — sharper color, still 100% model-derived. Env
        // MANGAHUE_JBU_DISABLE=1 falls back to bilinear for A/B.
        let use_jbu = std::env::var("MANGAHUE_JBU_DISABLE").is_err();
        let a_resized: Vec<f32>;
        let b_resized: Vec<f32>;
        if use_jbu {
            a_resized = joint_bilateral_upscale_ab_f32(&a_plane, &gray, w, h);
            b_resized = joint_bilateral_upscale_ab_f32(&b_plane, &gray, w, h);
        } else {
            // Bilinear fallback on the f32 planes (index = y * w + x).
            a_resized = bilinear_upscale_ab_f32(&a_plane, &gray, w, h);
            b_resized = bilinear_upscale_ab_f32(&b_plane, &gray, w, h);
        }

        // Merge a/b with the ORIGINAL tile's L* and convert back to sRGB.
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let g = gray.get_pixel(x, y)[0] as f32 / 255.0;
                let l = rgb_to_lab_l(g, g, g);
                let a = a_resized[y as usize * w as usize + x as usize];
                let b = b_resized[y as usize * w as usize + x as usize];
                let (r, g2, bl) = lab_to_rgb(l, a, b);
                out.put_pixel(
                    x,
                    y,
                    Rgb([
                        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (g2.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (bl.clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]),
                );
            }
        }
        Ok(out)
    }
}

/// Joint-bilateral upscale of a 512x512 chroma plane (f32, -127..127) to the
/// full page size.
///
/// The spatial kernel is a small Gaussian in source space; the range kernel
/// compares the FULL-RES grayscale at the target pixel against the grayscale
/// at each source sample, so chroma never crosses strong ink edges. This
/// replaces the plain bilinear upscale (which bleeds color over lines) with
/// a purely model-derived, edge-aware reconstruction. Unlike the older u8
/// path, chroma stays in f32 so smooth gradients (skies, skin) keep their
/// precision; quantization happens only at the final RGB write.
fn joint_bilateral_upscale_ab_f32(
    src: &[f32],
    gray: &image::GrayImage,
    w: u32,
    h: u32,
) -> Vec<f32> {
    const RADIUS: i32 = 2;
    const SIGMA_SPACE_SQ: f32 = 1.4f32 * 1.4f32 * 2.0; // 2 * sigma^2, sigma ~1.4 src px
    const SIGMA_RANGE_SQ: f32 = 24.0f32 * 24.0f32 * 2.0; // 2 * sigma^2, sigma ~24 gray levels
                                                         // The DDColor model is fixed at 512x512; the constant lives behind a
                                                         // feature gate, so define the model side locally here.
    const SIDE: u32 = 512;
    let side = SIDE;
    let sw = side as f32;
    let sh = side as f32;
    let mut out = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let cx = x as f32 * sw / w as f32;
            let cy = y as f32 * sh / h as f32;
            let gx = (cx.round() as i32).clamp(0, side as i32 - 1);
            let gy = (cy.round() as i32).clamp(0, side as i32 - 1);
            let target_g = gray.get_pixel(x, y)[0] as f32;
            let mut sum = 0.0f32;
            let mut weight = 0.0f32;
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    let sx = (gx + dx).clamp(0, side as i32 - 1) as u32;
                    let sy = (gy + dy).clamp(0, side as i32 - 1) as u32;
                    let ddx = sx as f32 - cx;
                    let ddy = sy as f32 - cy;
                    let spatial = (-(ddx * ddx + ddy * ddy) / SIGMA_SPACE_SQ).exp();
                    let src_x = ((sx as f32 * w as f32 / sw).round() as u32).min(w - 1);
                    let src_y = ((sy as f32 * h as f32 / sh).round() as u32).min(h - 1);
                    let source_g = gray.get_pixel(src_x, src_y)[0] as f32;
                    let range = (-((target_g - source_g).powi(2)) / SIGMA_RANGE_SQ).exp();
                    let wgt = spatial * range;
                    sum += src[(sy * side + sx) as usize] * wgt;
                    weight += wgt;
                }
            }
            out[(y * w + x) as usize] = sum / weight.max(1e-6);
        }
    }
    out
}

/// Plain bilinear fallback for the f32 chroma planes (used when
/// MANGAHUE_JBU_DISABLE=1).
fn bilinear_upscale_ab_f32(src: &[f32], gray: &image::GrayImage, w: u32, h: u32) -> Vec<f32> {
    const SIDE: u32 = 512; // DDColor model side (see joint_bilateral_upscale_ab_f32)
    let side = SIDE;
    let sw = side as f32;
    let sh = side as f32;
    let mut out = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 * sw / w as f32 - 0.5;
            let fy = y as f32 * sh / h as f32 - 0.5;
            let x0 = fx.floor().max(0.0) as u32;
            let y0 = fy.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(side - 1);
            let y1 = (y0 + 1).min(side - 1);
            let tx = (fx - x0 as f32).clamp(0.0, 1.0);
            let ty = (fy - y0 as f32).clamp(0.0, 1.0);
            let p00 = src[(y0 * side + x0) as usize];
            let p10 = src[(y0 * side + x1) as usize];
            let p01 = src[(y1 * side + x0) as usize];
            let p11 = src[(y1 * side + x1) as usize];
            let top = p00 + (p10 - p00) * tx;
            let bottom = p01 + (p11 - p01) * tx;
            out[(y * w + x) as usize] = top + (bottom - top) * ty;
        }
    }
    let _ = gray;
    out
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

    #[test]
    fn detects_complete_pretrained_model_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("colorizator.py"), "").unwrap();
        fs::create_dir_all(dir.path().join("networks")).unwrap();
        fs::write(dir.path().join("networks/generator.zip"), "weights").unwrap();
        fs::write(dir.path().join("networks/extractor.pth"), "weights").unwrap();
        fs::create_dir_all(dir.path().join("denoising/models")).unwrap();
        fs::write(dir.path().join("denoising/models/net_rgb.pth"), "weights").unwrap();
        assert!(is_pretrained_model_dir(dir.path()));
    }

    #[test]
    fn detects_anime_face_palette_model_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_anime_face_model_dir(dir.path()));
        fs::write(dir.path().join("lbpcascade_animeface.xml"), "cascade").unwrap();
        assert!(is_anime_face_model_dir(dir.path()));
    }

    #[test]
    fn worker_timeout_is_bounded_and_has_a_safe_default() {
        assert_eq!(
            worker_timeout_from(None),
            Duration::from_secs(DEFAULT_WORKER_TIMEOUT_SECONDS)
        );
        assert_eq!(
            worker_timeout_from(Some("1")),
            Duration::from_secs(MIN_WORKER_TIMEOUT_SECONDS)
        );
        assert_eq!(
            worker_timeout_from(Some("99999")),
            Duration::from_secs(MAX_WORKER_TIMEOUT_SECONDS)
        );
        assert_eq!(worker_timeout_from(Some("45")), Duration::from_secs(45));
        assert_eq!(
            worker_timeout_from(Some("not-a-number")),
            Duration::from_secs(DEFAULT_WORKER_TIMEOUT_SECONDS)
        );
    }

    #[test]
    fn stalled_worker_is_terminated_at_deadline() {
        #[cfg(windows)]
        let child = Command::new("cmd")
            .args(["/C", "ping 127.0.0.1 -n 30 >NUL"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        #[cfg(unix)]
        let child = Command::new("sh")
            .args(["-c", "sleep 30"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let mut process =
            attach_worker_process(child, "deadline-test", Duration::from_millis(25)).unwrap();
        let error = read_worker_response(&mut process, "test inference").unwrap_err();
        assert!(error.contains("timed out"));
        assert!(process.child.wait().unwrap().code().unwrap_or(-1) != 0);
    }
}
