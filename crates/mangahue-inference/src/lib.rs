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
}

impl<T: ColorizationBackend + ?Sized> ColorizationBackend for Arc<T> {
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        self.as_ref().colorize(input)
    }
}

pub fn is_ddcolor_model_dir(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.join("pytorch_model.pt").is_file() || path.join("models").is_dir()
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

#[derive(Debug, Clone)]
struct DdColorOnnxWorkerLaunch {
    python: String,
    model_path: PathBuf,
    worker_threads: usize,
}

impl DdColorOnnxWorkerLaunch {
    fn spawn(&self) -> Result<PretrainedWorkerProcess> {
        let child = Command::new(&self.python)
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
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
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
        Ok(worker)
    }
}

impl ColorizationBackend for DdColorOnnxBackend {
    fn name(&self) -> &str {
        "ddcolor-tiny-onnx"
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
        let child = Command::new(&self.python)
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
            .env("MALLOC_ARENA_MAX", "2")
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
        Ok(worker)
    }
}

impl ColorizationBackend for DdColorBackend {
    fn name(&self) -> &str {
        "ddcolor"
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
        let child = Command::new(&self.python)
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
            .env("MALLOC_ARENA_MAX", "2")
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

#[cfg(feature = "native-onnx")]
#[derive(Debug)]
pub struct RustOnnxBackend {
    session: Mutex<Session>,
    metadata: ModelMetadata,
    input_name: String,
    output_name: String,
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

        let mut tensor_data = Vec::with_capacity(channels * h * w);
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
        let in_name = &self.input_name;
        let out_name = &self.output_name;
        let outputs = session
            .run(ort::inputs![in_name.clone() => tensor])
            .map_err(|e| EngineError::Inference(format!("ONNX inference failed: {e}")))?;
        let (_, output_data) = outputs[out_name.as_str()]
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
