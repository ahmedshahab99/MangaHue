use anyhow::{anyhow, Context, Result};
use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::extract::{Extension, Path as AxumPath, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{Stream, StreamExt};
use mangahue_brain::{
    build_palette_memory, load_palette_memory, write_palette_memory, write_palette_memory_style,
    ChapterBrainReport, PaletteMemoryBook, PaletteMemoryEdit, PaletteMemoryRequest,
};
use mangahue_chapter::{
    run_chapter_job, ChapterConsistencyRetry, ChapterJobRequest, ChapterSafetyGateConfig,
};
use mangahue_config::{load_style_profile, EngineConfig};
use mangahue_core::{BoundingBox, ColorRgb, ImageSize};
use mangahue_hints::{
    load_region_locks, rasterize_region_locks, validate_region_locks, RegionLockFile,
};
use mangahue_image::{
    export_chapter_archive, import_chapter_to_dir, load_image, save_gray, save_rgb,
    write_chapter_manifest, ChapterImportOptions, ChapterManifest,
};
use mangahue_inference::ModelMetadata;
use mangahue_ingest::{ingest_url, IngestReport, IngestRequest};
use mangahue_intelligence::{
    analyze_semantics, character_observations, curate_dataset, semantic_region_locks,
    track_characters, CuratedSample, CuratorCandidate, SemanticAnalysisConfig, SemanticClass,
    SemanticPageReport,
};
use mangahue_memory::{
    add_correction, create_character, create_project_directory, export_correction_hints,
    export_correction_training_manifest, export_corrections_jsonl, NewCorrection, ProjectMemory,
};
use mangahue_orchestrator::{ColorizeRequest, InspectRequest, MangaHueEngine, PageInspection};
use mangahue_production::{DurableJob, NewJob, ProductionStore, QuotaSnapshot, UserAccount};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, RwLock, Semaphore};
use tokio_stream::wrappers::BroadcastStream;
use tokio_util::io::ReaderStream;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiJobKind {
    #[default]
    Unknown,
    Page,
    Batch,
    Link,
    Chapter,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApiJobProgress {
    pub stage: String,
    pub completed: usize,
    pub total: usize,
    pub percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiColorizeRequest {
    pub request: ColorizeRequest,
    pub ads_token: Option<String>,
    pub premium_account_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBatchRequest {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub template: ColorizeRequest,
    #[serde(default = "default_continue_on_error")]
    pub continue_on_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiLinkIngestRequest {
    pub source_url: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_link_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_link_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiLinkColorizeRequest {
    pub source_url: String,
    pub output_dir: PathBuf,
    pub downloads_dir: Option<PathBuf>,
    pub template: ColorizeRequest,
    #[serde(default = "default_link_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_link_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    #[serde(default = "default_continue_on_error")]
    pub continue_on_error: bool,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChapterImportRequest {
    pub input_path: PathBuf,
    pub output_dir: PathBuf,
    pub manifest_path: Option<PathBuf>,
    #[serde(default = "default_chapter_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_chapter_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    #[serde(default = "default_true")]
    pub overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChapterExportRequest {
    pub input_dir: PathBuf,
    pub output_file: PathBuf,
    pub manifest_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChapterColorizeRequest {
    pub input_path: PathBuf,
    pub work_dir: PathBuf,
    pub output_dir: PathBuf,
    pub export_path: Option<PathBuf>,
    pub gallery_path: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub brain_report_path: Option<PathBuf>,
    pub locked_style_path: Option<PathBuf>,
    pub palette_memory_path: Option<PathBuf>,
    pub palette_memory_style_path: Option<PathBuf>,
    pub safety_report_path: Option<PathBuf>,
    pub template: ColorizeRequest,
    #[serde(default = "default_chapter_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_chapter_max_bytes_per_page")]
    pub max_bytes_per_page: u64,
    #[serde(default)]
    pub resume: bool,
    #[serde(default = "default_true")]
    pub safety_gate: bool,
    #[serde(default)]
    pub enforce_safety_gate: bool,
    #[serde(default)]
    pub consistency_retry: bool,
    #[serde(default = "default_retry_drift_threshold")]
    pub retry_drift_threshold: f32,
    #[serde(default = "default_retry_tone_threshold")]
    pub retry_tone_threshold: f32,
    #[serde(default = "default_retry_max_pages")]
    pub retry_max_pages: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiInspectRequest {
    pub request: InspectRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiJob {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
    #[serde(default)]
    pub kind: ApiJobKind,
    pub status: ApiJobStatus,
    #[serde(default)]
    pub progress: ApiJobProgress,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    pub result_url: Option<String>,
    pub report_url: Option<String>,
    pub output_path: Option<PathBuf>,
    pub error: Option<String>,
    pub result: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiJobStatus {
    Queued,
    Running,
    Complete,
    Cancelled,
    Failed,
}

impl ApiJob {
    fn new(
        id: String,
        kind: ApiJobKind,
        output_path: Option<PathBuf>,
        owner_id: Option<String>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            result_url: Some(format!("/api/jobs/{id}/result")),
            id,
            owner_id,
            kind,
            status: ApiJobStatus::Queued,
            progress: ApiJobProgress {
                stage: "queued".to_string(),
                completed: 0,
                total: 0,
                percent: 0.0,
            },
            created_at: now.clone(),
            updated_at: now,
            report_url: None,
            output_path,
            error: None,
            result: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiState {
    jobs: Arc<RwLock<HashMap<String, ApiJob>>>,
    pub jobs_dir: PathBuf,
    pub styles_dir: PathBuf,
    pub studio_dir: PathBuf,
    pub cancel_tokens: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    pub event_channels: Arc<RwLock<HashMap<String, broadcast::Sender<String>>>>,
    pub job_slots: Arc<Semaphore>,
    pub max_concurrent_jobs: usize,
    pub job_retention_hours: u64,
    pub default_model_path: PathBuf,
    pub default_config_path: PathBuf,
    pub production: ProductionStore,
    /// Root directory that /api/file and sandboxed job paths resolve under.
    pub file_root: PathBuf,
    /// Public mode: job inputs/outputs are confined to the jobs workspace
    /// and user-supplied model paths are rejected.
    pub sandbox_jobs: bool,
    /// Extra origins (besides loopback) allowed to call the API.
    pub allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiServerConfig {
    pub addr: SocketAddr,
    pub jobs_dir: PathBuf,
    pub styles_dir: PathBuf,
    pub studio_dir: PathBuf,
    pub production_db: PathBuf,
    pub public_api: bool,
    /// Root directory that /api/file and sandboxed job paths resolve under.
    pub file_root: PathBuf,
    /// Extra origins (besides loopback) allowed to call the API.
    pub allowed_origins: Vec<String>,
    #[serde(default = "default_max_concurrent_jobs")]
    pub max_concurrent_jobs: usize,
    #[serde(default = "default_job_retention_hours")]
    pub job_retention_hours: u64,
    #[serde(default = "default_model_path")]
    pub default_model_path: PathBuf,
    #[serde(default = "default_server_config_path")]
    pub default_config_path: PathBuf,
}

#[derive(Debug, Clone)]
struct RequestUser(UserAccount);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiDurableJobRequest {
    pub kind: String,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub priority: i64,
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiProjectCreateRequest {
    pub name: String,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCharacterCreateRequest {
    pub project: PathBuf,
    pub name: String,
    pub hair: Option<String>,
    pub eyes: Option<String>,
    pub skin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiProjectPathRequest {
    pub project: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCorrectionAddRequest {
    pub project: PathBuf,
    pub page: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub label: Option<String>,
    pub old: String,
    pub new: String,
    pub character: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCorrectionExportRequest {
    pub project: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCorrectionHintsExportRequest {
    pub project: PathBuf,
    pub page: String,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCorrectionTrainingManifestRequest {
    pub project: PathBuf,
    pub lineart_dir: PathBuf,
    pub color_dir: PathBuf,
    pub hints_dir: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPaletteMemoryCreateRequest {
    pub brain_report_path: PathBuf,
    pub output_path: PathBuf,
    pub style_output_path: Option<PathBuf>,
    pub base_style_path: Option<PathBuf>,
    pub id: Option<String>,
    pub name: Option<String>,
    #[serde(default = "default_palette_memory_max_entries")]
    pub max_entries: usize,
    #[serde(default = "default_palette_memory_min_confidence")]
    pub min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPaletteMemoryEditRequest {
    pub memory_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub style_output_path: Option<PathBuf>,
    pub base_style_path: Option<PathBuf>,
    #[serde(default)]
    pub edits: Vec<PaletteMemoryEdit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRegionLockPreviewRequest {
    pub input_path: PathBuf,
    pub region_locks_path: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRegionLockSaveRequest {
    pub output_path: PathBuf,
    pub document: RegionLockFile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSemanticAnalyzeRequest {
    pub input_path: PathBuf,
    pub report_output: Option<PathBuf>,
    pub locks_output: Option<PathBuf>,
    #[serde(default)]
    pub palette: HashMap<SemanticClass, ColorRgb>,
    #[serde(default)]
    pub config: SemanticAnalysisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCharacterTrackRequest {
    pub reports: Vec<PathBuf>,
    pub output: Option<PathBuf>,
    #[serde(default = "default_character_similarity")]
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiDatasetCurateRequest {
    pub candidates: Vec<CuratorCandidate>,
    pub brain_report: PathBuf,
    pub safety_report: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiFileQuery {
    pub path: PathBuf,
    pub job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorRequest {
    #[serde(default = "default_config_path")]
    pub config_path: PathBuf,
    #[serde(default = "default_styles_dir")]
    pub styles_dir: PathBuf,
    #[serde(default = "default_model_metadata_path")]
    pub metadata_path: PathBuf,
    #[serde(default = "default_jobs_dir")]
    pub jobs_dir: PathBuf,
    #[serde(default = "default_studio_dir")]
    pub studio_dir: PathBuf,
    #[serde(default = "default_sample_input_path")]
    pub sample_input: PathBuf,
    #[serde(default = "default_python_command")]
    pub python: String,
}

impl Default for DoctorRequest {
    fn default() -> Self {
        Self {
            config_path: default_config_path(),
            styles_dir: default_styles_dir(),
            metadata_path: default_model_metadata_path(),
            jobs_dir: default_jobs_dir(),
            studio_dir: default_studio_dir(),
            sample_input: default_sample_input_path(),
            python: default_python_command(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoctorStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorCheck {
    pub name: String,
    pub status: DoctorStatus,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub status: DoctorStatus,
    pub summary: String,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleSummary {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBatchSummary {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub processed: usize,
    pub failed: usize,
    pub pages: Vec<ApiBatchPage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBatchPage {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub status: ApiJobStatus,
    pub error: Option<String>,
    pub report: Option<mangahue_core::ProcessingReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiLinkColorizeSummary {
    pub source_url: String,
    pub downloads_dir: PathBuf,
    pub output_dir: PathBuf,
    pub archive_path: Option<PathBuf>,
    pub ingest: IngestReport,
    pub processed: usize,
    pub failed: usize,
    pub pages: Vec<ApiLinkColorizePage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiLinkColorizePage {
    pub source_url: String,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub status: ApiJobStatus,
    pub error: Option<String>,
    pub report: Option<mangahue_core::ProcessingReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub error: String,
}

#[derive(Debug, Clone)]
struct JobUpdate {
    status: ApiJobStatus,
    error: Option<String>,
    output_path: Option<PathBuf>,
    result: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct ApiError(StatusCode, anyhow::Error);

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error)
    }
}

impl ApiError {
    fn with_status(status: StatusCode, error: impl Into<anyhow::Error>) -> Self {
        Self(status, error.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let ApiError(status, error) = self;
        let message = if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %error, "API request failed");
            "internal server error".to_string()
        } else {
            error.to_string()
        };
        let body = Json(ApiErrorBody { error: message });
        (status, body).into_response()
    }
}

pub async fn serve(config: ApiServerConfig) -> Result<()> {
    std::fs::create_dir_all(&config.jobs_dir)
        .with_context(|| format!("failed to create {}", config.jobs_dir.display()))?;
    let recovered_jobs = recover_job_manifests(&config.jobs_dir)?;
    let default_model_path = config.default_model_path.canonicalize().with_context(|| {
        format!(
            "default model is unavailable at {}; download it before starting the API",
            config.default_model_path.display()
        )
    })?;
    let default_config_path = config.default_config_path.canonicalize().with_context(|| {
        format!(
            "default engine config is unavailable at {}",
            config.default_config_path.display()
        )
    })?;
    validate_default_inference_runtime(&default_model_path)?;
    let production = ProductionStore::open(&config.production_db)?;
    let file_root = config.file_root.clone();
    let allowed_origins = config.allowed_origins.clone();
    let sandbox_jobs = config.public_api;
    let state = ApiState {
        jobs: Arc::new(RwLock::new(recovered_jobs)),
        jobs_dir: config.jobs_dir,
        styles_dir: config.styles_dir,
        studio_dir: config.studio_dir,
        cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
        event_channels: Arc::new(RwLock::new(HashMap::new())),
        job_slots: Arc::new(Semaphore::new(config.max_concurrent_jobs.max(1))),
        max_concurrent_jobs: config.max_concurrent_jobs.max(1),
        job_retention_hours: config.job_retention_hours,
        default_model_path,
        default_config_path,
        production,
        file_root,
        sandbox_jobs,
        allowed_origins,
    };
    spawn_transient_job_retention(
        state.jobs.clone(),
        state.jobs_dir.clone(),
        state.job_retention_hours,
    );
    let app = router_with_mode(state, config.public_api);
    let listener = TcpListener::bind(config.addr).await?;
    info!("MangaHue API listening on http://{}", config.addr);
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn router(state: ApiState) -> Router {
    router_with_mode(state, false)
}

pub fn router_with_mode(state: ApiState, public_api: bool) -> Router {
    let studio_dir = state.studio_dir.clone();
    let public_routes = Router::new()
        .route("/health", get(health))
        .route(
            "/api/v1/jobs",
            get(list_durable_jobs).post(create_durable_job),
        )
        .route("/api/v1/jobs/:id", get(get_durable_job))
        .route("/api/v1/jobs/:id/cancel", post(cancel_durable_job))
        .route("/api/v1/quota", get(get_quota));

    // End-user webapp surfaces. In public mode every path field is sandboxed
    // under the jobs workspace (see the sandbox_* helpers), so an
    // authenticated caller can neither read nor write files outside the job
    // workspace and can never make the engine load arbitrary checkpoints.
    let user_routes = Router::new()
        .route("/api/styles", get(list_styles))
        .route("/api/file", get(get_local_file))
        .route(
            "/api/upload",
            post(upload_file).layer(DefaultBodyLimit::max(256 * 1024 * 1024)),
        )
        .route("/api/colorize", post(create_colorize_job))
        .route("/api/link/colorize", post(create_link_colorize_job))
        .route("/api/chapter/colorize", post(create_chapter_colorize_job))
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/:id", get(get_job))
        .route("/api/jobs/:id/cancel", post(cancel_job))
        .route("/api/jobs/:id/result", get(get_job_result))
        .route("/api/jobs/:id/events", get(job_events_sse));

    // Operator-only surfaces: engine internals, brain, projects and
    // corrections take host paths by design and are trusted-admin tools.
    let admin_routes = Router::new()
        .route("/metrics", get(metrics))
        .route(
            "/api/gpu/colorize",
            post(gpu_colorize).layer(DefaultBodyLimit::max(96 * 1024 * 1024)),
        )
        .route("/api/batch", post(create_batch_job))
        .route("/api/link/ingest", post(ingest_link))
        .route("/api/chapter/import", post(import_chapter))
        .route("/api/chapter/export", post(export_chapter))
        .route(
            "/api/doctor",
            get(run_doctor_default).post(run_doctor_custom),
        )
        .route("/api/inspect", post(inspect_page))
        .route(
            "/api/brain/palette-memory",
            post(create_palette_memory_from_report),
        )
        .route("/api/brain/palette-memory/edit", post(edit_palette_memory))
        .route(
            "/api/brain/region-locks/preview",
            post(preview_region_locks),
        )
        .route("/api/brain/region-locks/save", post(save_region_locks))
        .route("/api/brain/semantics", post(analyze_page_semantics))
        .route("/api/brain/characters/track", post(track_page_characters))
        .route("/api/brain/dataset/curate", post(curate_training_dataset))
        .route("/api/projects", post(create_project))
        .route("/api/projects/status", post(project_status))
        .route("/api/projects/export-corrections", post(export_corrections))
        .route(
            "/api/projects/export-hints",
            post(export_correction_hints_file),
        )
        .route(
            "/api/projects/export-training-manifest",
            post(export_training_manifest),
        )
        .route("/api/projects/characters", post(create_character_profile))
        .route(
            "/api/projects/:project_id/characters",
            post(create_character_profile_for_project),
        )
        .route("/api/corrections", post(add_project_correction));

    // Public static assets: the user webapp, operator Studio, WebGPU studio,
    // and locally installed models. The configured Studio directory now
    // points at the exported Next.js application.
    let web_index = studio_dir.join("index.html");
    let static_routes = Router::new()
        .nest_service(
            "/app",
            ServeDir::new(&studio_dir).not_found_service(ServeFile::new(&web_index)),
        )
        .nest_service(
            "/operator",
            ServeDir::new("web/studio").not_found_service(ServeFile::new("web/studio/index.html")),
        )
        .nest_service(
            "/gpu",
            ServeDir::new("web/gpu").not_found_service(ServeFile::new("web/gpu/index.html")),
        )
        .nest_service("/models", ServeDir::new("models"));

    let app = if public_api {
        // Public mode: everything except /health and static assets requires
        // a Bearer token; operator surfaces additionally require an admin
        // token. CORS admits loopback plus any configured origins.
        let user_authed = user_routes.route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_user,
        ));
        let admin_authed = admin_routes.route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_admin,
        ));
        public_routes
            .merge(static_routes)
            .merge(user_authed)
            .merge(admin_authed)
            .fallback_service(
                ServeDir::new(&studio_dir)
                    .not_found_service(ServeFile::new(studio_dir.join("index.html"))),
            )
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                loopback_cors,
            ))
    } else {
        // Local operator mode. The Studio is served from the same origin and
        // automatically attaches the guard header to every API call; foreign
        // web pages cannot send it (a custom header forces a CORS preflight,
        // which the loopback-only CORS policy rejects). This closes the
        // "any website you browse can drive the local API" and DNS-rebinding
        // windows while keeping the operator UX unchanged.
        let guarded_user = user_routes.route_layer(axum::middleware::from_fn(local_guard));
        let guarded_admin = admin_routes.route_layer(axum::middleware::from_fn(local_guard));
        public_routes
            .merge(guarded_user)
            .merge(guarded_admin)
            .merge(static_routes)
            .fallback_service(
                ServeDir::new(&studio_dir)
                    .not_found_service(ServeFile::new(studio_dir.join("index.html"))),
            )
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                loopback_cors,
            ))
    };
    app.layer(axum::middleware::from_fn(security_headers))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn security_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; base-uri 'self'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; img-src 'self' blob: data:; object-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'",
        ),
    );
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=(), payment=()"),
    );
    response
}

/// Rejects local-mode API calls that do not carry the guard header the
/// Studio attaches automatically. Custom headers force a CORS preflight,
/// which `loopback_cors` only grants to loopback origins — so cross-site
/// pages and DNS-rebinding hosts cannot drive the local API.
async fn local_guard(request: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let guarded = request
        .headers()
        .get("x-mangahue-local")
        .and_then(|value| value.to_str().ok())
        .map(|value| value == "1")
        .unwrap_or(false);
    if !guarded {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "local API requires the studio guard header"})),
        )
            .into_response();
    }
    next.run(request).await
}

/// Public-mode gate: any valid Bearer token is enough for the end-user
/// webapp surfaces.
async fn require_user(
    State(state): State<ApiState>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let user = match authenticated_user(&state, request.headers()) {
        Ok(user) => user,
        Err(error) => return error.into_response(),
    };
    request.extensions_mut().insert(RequestUser(user));
    next.run(request).await
}

/// Public-mode gate: operator surfaces additionally require an admin token.
async fn require_admin(
    State(state): State<ApiState>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let user = match authenticated_user(&state, request.headers()) {
        Ok(user) => user,
        Err(error) => return error.into_response(),
    };
    if user.tier != mangahue_production::UserTier::Admin {
        return ApiError::with_status(StatusCode::FORBIDDEN, anyhow!("admin token required"))
            .into_response();
    }
    request.extensions_mut().insert(RequestUser(user));
    next.run(request).await
}

/// CORS that only ever grants access to loopback origins (plus any origins
/// explicitly allowed by configuration), and OPTIONS preflight handling.
/// The Studio uses relative URLs, so its own origin is always the loopback
/// host that the operator is browsing.
async fn loopback_cors(
    State(state): State<ApiState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());
    let allowed = origin
        .as_deref()
        .map(|origin| {
            origin_is_loopback(origin)
                || state
                    .allowed_origins
                    .iter()
                    .any(|allowed| allowed == origin)
        })
        .unwrap_or(false);

    if request.method() == axum::http::Method::OPTIONS {
        if allowed {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::NO_CONTENT;
            apply_cors_headers(response.headers_mut(), origin.as_deref());
            return response;
        }
        return StatusCode::FORBIDDEN.into_response();
    }

    let mut response = next.run(request).await;
    if allowed {
        apply_cors_headers(response.headers_mut(), origin.as_deref());
    }
    response
}

fn apply_cors_headers(headers: &mut HeaderMap, origin: Option<&str>) {
    if let Some(origin) = origin {
        if let Ok(value) = HeaderValue::from_str(origin) {
            headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        }
    }
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type, x-mangahue-local, authorization"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static("600"),
    );
}

fn origin_is_loopback(origin: &str) -> bool {
    let Ok(url) = url::Url::parse(origin) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    if matches!(host, "localhost" | "localhost.localdomain") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.to_canonical().is_loopback())
        .unwrap_or(false)
}

async fn health(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "MangaHue Engine",
        "role": "manga color director",
        "jobs": {
            "running_capacity": state.max_concurrent_jobs,
            "available_slots": state.job_slots.available_permits(),
        },
        "model": {
            "ready": state.default_model_path.is_file(),
            "name": state.default_model_path.file_name().and_then(|name| name.to_str()),
        },
    }))
}

async fn metrics(State(state): State<ApiState>) -> Result<Response, ApiError> {
    let metrics = state.production.metrics()?;
    let body = format!(
        "# TYPE mangahue_users gauge\n\
         mangahue_users {}\n\
         # TYPE mangahue_jobs gauge\n\
         mangahue_jobs{{status=\"queued\"}} {}\n\
         mangahue_jobs{{status=\"running\"}} {}\n\
         mangahue_jobs{{status=\"complete\"}} {}\n\
         mangahue_jobs{{status=\"failed\"}} {}\n\
         mangahue_jobs{{status=\"dead_letter\"}} {}\n\
         # TYPE mangahue_artifacts gauge\n\
         mangahue_artifacts {}\n\
         mangahue_artifact_bytes {}\n\
         mangahue_pages_24h {}\n",
        metrics.users,
        metrics.queued_jobs,
        metrics.running_jobs,
        metrics.completed_jobs,
        metrics.failed_jobs,
        metrics.dead_letter_jobs,
        metrics.artifacts,
        metrics.artifact_bytes,
        metrics.pages_24h,
    );
    Ok(([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response())
}

async fn create_durable_job(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(mut payload): Json<ApiDurableJobRequest>,
) -> Result<Json<DurableJob>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    // Non-admin users cannot make the worker read/write arbitrary paths on
    // the host: every path field must stay relative (inside the worker's
    // working directory) and free of parent traversal. Admin tokens are
    // trusted operator credentials.
    if user.tier != mangahue_production::UserTier::Admin {
        if let Err(error) = sanitize_job_payload(&mut payload.payload) {
            return Err(ApiError::with_status(StatusCode::BAD_REQUEST, error));
        }
        payload.priority = payload.priority.clamp(0, 100);
        payload.max_attempts = payload.max_attempts.clamp(1, 5);
    }
    let job = state.production.enqueue(NewJob {
        user_id: user.id,
        kind: payload.kind,
        payload: payload.payload,
        priority: payload.priority,
        max_attempts: payload.max_attempts,
    })?;
    Ok(Json(job))
}

const PAYLOAD_PATH_FIELDS: &[&str] = &[
    "input_path",
    "output_path",
    "reference_image_path",
    "model_path",
    "cache_dir",
    "export_path",
    "downloads_dir",
    "input_dir",
    "output_dir",
    "manifest_path",
    "project_path",
    "config_path",
    "preset_path",
    "style_path",
    "user_hints_path",
    "debug_dir",
];

/// Rejects payloads whose path fields are absolute or escape the working
/// directory, so an authenticated non-admin caller cannot make the worker
/// read or write files outside the sandbox.
fn sanitize_job_payload(payload: &mut serde_json::Value) -> Result<()> {
    fn walk(value: &mut serde_json::Value) -> Result<()> {
        match value {
            serde_json::Value::Object(map) => {
                for (key, field) in map.iter_mut() {
                    if PAYLOAD_PATH_FIELDS.contains(&key.as_str()) {
                        if let Some(path) = field.as_str() {
                            if path.is_empty() {
                                continue;
                            }
                            let cleaned = path.replace('\\', "/");
                            if std::path::Path::new(path).is_absolute() || cleaned.contains("../") {
                                anyhow::bail!(
                                    "path field {key} must be relative to the job sandbox and must not contain '..': {path}"
                                );
                            }
                        }
                    } else {
                        walk(field)?;
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(payload)
}

/// Reject URLs that resolve to private, loopback or link-local space so the
/// API can never be used as a proxy into the host network or cloud metadata.
fn validate_public_source_url(source_url: &str) -> Result<()> {
    let url = url::Url::parse(source_url).with_context(|| "invalid source URL")?;
    mangahue_ingest::validate_public_url(&url)
}

/// Strip optional host-path knobs that only an operator should control.
/// Model/checkpoint paths are dropped outright: a remote caller must never
/// be able to make the engine torch.load an arbitrary file (pickle RCE).
fn sandbox_engine_paths(request: &mut ColorizeRequest) {
    request.model_path = None;
    request.reference_image_path = None;
    request.ensemble_model_paths.clear();
    request.config_path = None;
    request.preset_path = None;
    request.palette_path = None;
    request.draft_path = None;
    request.cache_dir = None;
    request.debug_dir = None;
    request.user_hints_path = None;
    request.project_path = None;
    request.region_locks_path = None;
    request.character_ids.clear();
    request.idempotency_key = None;
    request.use_mock_backend = false;
}

fn apply_engine_defaults(request: &mut ColorizeRequest, state: &ApiState) -> Result<()> {
    if state.sandbox_jobs {
        request.style_path = request
            .style_path
            .as_deref()
            .map(|path| resolve_trusted_style(&state.styles_dir, path))
            .transpose()?;
    }
    if request.model_path.is_none() {
        request.model_path = Some(state.default_model_path.clone());
    }
    if request.config_path.is_none() {
        request.config_path = Some(state.default_config_path.clone());
    }
    Ok(())
}

fn resolve_trusted_style(styles_dir: &Path, requested: &Path) -> Result<PathBuf> {
    let root = styles_dir
        .canonicalize()
        .with_context(|| format!("failed to resolve style directory {}", styles_dir.display()))?;
    let mut candidates = vec![requested.to_path_buf()];
    if !requested.is_absolute() {
        candidates.push(root.join(requested));
    }
    for candidate in candidates {
        if let Ok(resolved) = candidate.canonicalize() {
            if resolved.starts_with(&root)
                && resolved.extension().and_then(|value| value.to_str()) == Some("yaml")
                && resolved.is_file()
            {
                return Ok(resolved);
            }
        }
    }
    anyhow::bail!("style must be one of the server-provided YAML profiles")
}

/// Public-mode sandbox for a single-page colorize request: the input must
/// live inside the jobs workspace, the output is forced into it, and
/// user-supplied model/checkpoint paths are dropped.
fn sandbox_colorize_request(
    request: &mut ColorizeRequest,
    jobs_dir: &Path,
    id: &str,
    owner_id: Option<&str>,
) -> Result<()> {
    let input = request.input_path.clone();
    request.input_path = resolve_owned_upload(jobs_dir, &input, owner_id).with_context(|| {
        format!(
            "input_path must live under the job workspace: {}",
            input.display()
        )
    })?;
    let out_dir = jobs_dir.join("out");
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    request.output_path = out_dir.join(format!("{id}.png"));
    sandbox_engine_paths(request);
    Ok(())
}

fn sandbox_batch_request(
    request: &mut ApiBatchRequest,
    jobs_dir: &Path,
    id: &str,
    owner_id: Option<&str>,
) -> Result<()> {
    let base = jobs_dir.join("batch").join(id);
    std::fs::create_dir_all(&base)
        .with_context(|| format!("failed to create {}", base.display()))?;
    request.input_dir =
        resolve_owned_upload(jobs_dir, &request.input_dir, owner_id).with_context(|| {
            format!(
                "input_dir must live under the job workspace: {}",
                request.input_dir.display()
            )
        })?;
    request.output_dir = base.join("out");
    sandbox_engine_paths(&mut request.template);
    Ok(())
}

fn sandbox_link_request(
    request: &mut ApiLinkColorizeRequest,
    jobs_dir: &Path,
    id: &str,
) -> Result<()> {
    let base = jobs_dir.join("link").join(id);
    std::fs::create_dir_all(&base)
        .with_context(|| format!("failed to create {}", base.display()))?;
    request.output_dir = base.join("out");
    request.downloads_dir = Some(base.join("downloads"));
    sandbox_engine_paths(&mut request.template);
    Ok(())
}

fn sandbox_chapter_request(
    request: &mut ApiChapterColorizeRequest,
    jobs_dir: &Path,
    id: &str,
    owner_id: Option<&str>,
) -> Result<()> {
    let base = jobs_dir.join("chapter").join(id);
    std::fs::create_dir_all(&base)
        .with_context(|| format!("failed to create {}", base.display()))?;
    request.input_path = resolve_owned_upload(jobs_dir, &request.input_path, owner_id)
        .with_context(|| {
            format!(
                "input_path must live under the job workspace: {}",
                request.input_path.display()
            )
        })?;
    request.work_dir = base.join("work");
    request.output_dir = base.join("out");
    request.export_path = Some(base.join("chapter.cbz"));
    request.gallery_path = Some(base.join("gallery.html"));
    request.manifest_path = Some(base.join("manifest.json"));
    request.brain_report_path = Some(base.join("brain.json"));
    request.locked_style_path = Some(base.join("locked_style.json"));
    request.palette_memory_path = Some(base.join("palette_memory.json"));
    request.palette_memory_style_path = Some(base.join("palette_memory_style.json"));
    request.safety_report_path = Some(base.join("safety_gate.json"));
    sandbox_engine_paths(&mut request.template);
    Ok(())
}

async fn list_durable_jobs(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DurableJob>>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    Ok(Json(state.production.jobs_for_user(&user.id, 100)?))
}

async fn get_durable_job(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<DurableJob>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let job = state
        .production
        .job(&id)?
        .filter(|job| job.user_id == user.id || user.tier == mangahue_production::UserTier::Admin)
        .ok_or_else(|| ApiError::with_status(StatusCode::NOT_FOUND, anyhow!("job not found")))?;
    Ok(Json(job))
}

async fn cancel_durable_job(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let cancelled = state.production.cancel(&id, &user.id)?;
    Ok(Json(serde_json::json!({"id": id, "cancelled": cancelled})))
}

async fn get_quota(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<QuotaSnapshot>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    Ok(Json(state.production.quota_snapshot(&user.id)?))
}

fn authenticated_user(
    state: &ApiState,
    headers: &HeaderMap,
) -> std::result::Result<UserAccount, ApiError> {
    let value = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            ApiError::with_status(
                StatusCode::UNAUTHORIZED,
                anyhow!("missing Authorization header"),
            )
        })?;
    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .ok_or_else(|| {
            ApiError::with_status(
                StatusCode::UNAUTHORIZED,
                anyhow!("Authorization must use Bearer token"),
            )
        })?;
    let user = state
        .production
        .authenticate(token)
        .map_err(ApiError::from)?
        .ok_or_else(|| {
            ApiError::with_status(
                StatusCode::UNAUTHORIZED,
                anyhow!("invalid or expired API token"),
            )
        })?;
    let requests_per_minute = match user.tier {
        mangahue_production::UserTier::Free => 30,
        mangahue_production::UserTier::Creator => 120,
        mangahue_production::UserTier::Studio => 600,
        mangahue_production::UserTier::Admin => 10_000,
    };
    state
        .production
        .enforce_rate_limit(&user.id, requests_per_minute, 60)
        .map_err(|error| ApiError::with_status(StatusCode::TOO_MANY_REQUESTS, error))?;
    Ok(user)
}

fn request_owner(
    state: &ApiState,
    user: Option<&Extension<RequestUser>>,
) -> std::result::Result<Option<String>, ApiError> {
    if !state.sandbox_jobs {
        return Ok(None);
    }
    user.map(|user| Some(user.0 .0.id.clone())).ok_or_else(|| {
        ApiError::with_status(
            StatusCode::UNAUTHORIZED,
            anyhow!("authenticated user context is missing"),
        )
    })
}

fn job_visible_to(job: &ApiJob, user: Option<&Extension<RequestUser>>) -> bool {
    match user {
        None => job.owner_id.is_none(),
        Some(user) => {
            user.0 .0.tier == mangahue_production::UserTier::Admin
                || job.owner_id.as_deref() == Some(user.0 .0.id.as_str())
        }
    }
}

async fn authorized_job(
    state: &ApiState,
    id: &str,
    user: Option<&Extension<RequestUser>>,
) -> std::result::Result<ApiJob, ApiError> {
    let job = get_job_by_id(state, id).await?;
    if state.sandbox_jobs && !job_visible_to(&job, user) {
        return Err(ApiError::with_status(
            StatusCode::NOT_FOUND,
            anyhow!("job not found"),
        ));
    }
    Ok(job)
}

async fn reserve_web_job(
    state: &ApiState,
    owner_id: Option<&str>,
    requested_pages: usize,
    job_id: &str,
) -> std::result::Result<(), ApiError> {
    let Some(owner_id) = owner_id else {
        return Ok(());
    };
    let snapshot = state
        .production
        .enforce_job_quota(owner_id, requested_pages.max(1) as u64)
        .map_err(|error| ApiError::with_status(StatusCode::TOO_MANY_REQUESTS, error))?;
    let active_web_jobs = state
        .jobs
        .read()
        .await
        .values()
        .filter(|job| {
            job.owner_id.as_deref() == Some(owner_id)
                && matches!(job.status, ApiJobStatus::Queued | ApiJobStatus::Running)
        })
        .count() as u64;
    if active_web_jobs >= snapshot.policy.max_concurrent_jobs {
        return Err(ApiError::with_status(
            StatusCode::TOO_MANY_REQUESTS,
            anyhow!("concurrent job quota exhausted"),
        ));
    }
    state.production.record_usage(
        owner_id,
        "pages",
        requested_pages.max(1) as u64,
        Some(job_id),
    )?;
    state
        .production
        .record_usage(owner_id, "jobs", 1, Some(job_id))?;
    Ok(())
}

async fn run_doctor_default(State(state): State<ApiState>) -> Json<DoctorReport> {
    let request = DoctorRequest {
        styles_dir: state.styles_dir,
        jobs_dir: state.jobs_dir,
        studio_dir: state.studio_dir,
        ..DoctorRequest::default()
    };
    Json(run_doctor_report(request))
}

async fn run_doctor_custom(Json(payload): Json<DoctorRequest>) -> Json<DoctorReport> {
    // The `python` field is argv[0] of a spawned process: only accept known
    // Python interpreters so the endpoint can never launch arbitrary
    // binaries (and exfiltrate their stderr) from an unauthenticated call.
    let mut sanitized = payload;
    if !python_command_is_safe(&sanitized.python) {
        sanitized.python = default_python_command();
    }
    Json(run_doctor_report(sanitized))
}

fn python_command_is_safe(python: &str) -> bool {
    let python = python.trim();
    if python.is_empty() || python.contains(['/', '\\']) {
        return false;
    }
    // Plain interpreter names only (python, python3, python3.11, python.exe):
    // never accept a path, so argv[0] can only be a Python binary.
    let name = python.to_ascii_lowercase();
    let name = name.strip_suffix(".exe").unwrap_or(&name);
    name.starts_with("python")
}

pub fn run_doctor_report(request: DoctorRequest) -> DoctorReport {
    let checks = vec![
        check_engine_config(&request.config_path),
        check_styles_dir(&request.styles_dir),
        check_model_metadata(&request.metadata_path),
        check_python_sidecar(&request.python),
        check_jobs_dir(&request.jobs_dir),
        check_studio_dir(&request.studio_dir),
        check_sample_input(&request.sample_input),
    ];
    let status = aggregate_doctor_status(&checks);
    let summary = match status {
        DoctorStatus::Pass => "all platform checks passed".to_string(),
        DoctorStatus::Warn => "platform is usable with warnings".to_string(),
        DoctorStatus::Fail => "one or more required platform checks failed".to_string(),
    };
    DoctorReport {
        status,
        summary,
        checks,
    }
}

/// WebGPU flow: the client runs the model on ITS GPU (onnxruntime-web) and
/// uploads the colorized draft + the original page. The server finishes the
/// draft with the full protected-layer, region director and grading
/// postprocess, then returns the result file.
async fn gpu_colorize(
    State(state): State<ApiState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<serde_json::Value>, ApiError> {
    let id = Uuid::new_v4().to_string();
    let dir = state.jobs_dir.join(format!("gpu_{id}"));
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let mut page_path: Option<PathBuf> = None;
    let mut draft_path: Option<PathBuf> = None;
    let mut preset_path: Option<PathBuf> = None;
    let mut style_path: Option<PathBuf> = None;
    let mut palette_path: Option<PathBuf> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?
    {
        match field.name() {
            Some("page") => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?;
                if data.len() > 50 * 1024 * 1024 {
                    return Err(ApiError::with_status(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        anyhow!("page upload exceeds 50 MB"),
                    ));
                }
                let path = dir.join("page.jpg");
                std::fs::write(&path, &data)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                page_path = Some(path);
            }
            Some("draft") => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?;
                if data.len() > 50 * 1024 * 1024 {
                    return Err(ApiError::with_status(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        anyhow!("draft upload exceeds 50 MB"),
                    ));
                }
                let path = dir.join("draft.png");
                std::fs::write(&path, &data)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                draft_path = Some(path);
            }
            Some("preset") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?;
                if !text.trim().is_empty() {
                    preset_path = Some(PathBuf::from(text.trim()));
                }
            }
            Some("style") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?;
                if !text.trim().is_empty() {
                    style_path = Some(PathBuf::from(text.trim()));
                }
            }
            Some("palette") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(e)))?;
                if !text.trim().is_empty() {
                    palette_path = Some(PathBuf::from(text.trim()));
                }
            }
            _ => {}
        }
    }
    let (Some(page_path), Some(draft_path)) = (page_path, draft_path) else {
        return Err(ApiError::with_status(
            StatusCode::BAD_REQUEST,
            anyhow!("multipart requires 'page' and 'draft' image fields"),
        ));
    };
    let output_path = dir.join("result.png");
    let request = ColorizeRequest {
        input_path: page_path,
        output_path: output_path.clone(),
        draft_path: Some(draft_path),
        preset_path,
        style_path,
        palette_path,
        cache_dir: Some(state.jobs_dir.clone()),
        ..ColorizeRequest::default()
    };
    let result = MangaHueEngine::new().colorize(request)?;
    let file_root = state.file_root.clone();
    let relative = |path: &Path| path.strip_prefix(&file_root).unwrap_or(path).to_path_buf();
    let report_path = mangahue_image::sibling_with_suffix(&output_path, ".report", "json");
    Ok(Json(serde_json::json!({
        "id": id,
        "result_url": format!("/api/file?path={}", relative(&output_path).display()),
        "report_url": format!("/api/file?path={}", relative(&report_path).display()),
        "verdict": result.report.director_verdict,
    })))
}

async fn create_colorize_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    Json(mut payload): Json<ApiColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let owner_id = request_owner(&state, user.as_ref())?;
    let id = Uuid::new_v4().to_string();
    if state.sandbox_jobs {
        sandbox_colorize_request(
            &mut payload.request,
            &state.jobs_dir,
            &id,
            owner_id.as_deref(),
        )?;
    }
    apply_engine_defaults(&mut payload.request, &state)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    reserve_web_job(&state, owner_id.as_deref(), 1, &id).await?;
    let cancel_token = Arc::new(AtomicBool::new(false));
    let (event_tx, _) = broadcast::channel::<String>(32);
    state
        .cancel_tokens
        .write()
        .await
        .insert(id.clone(), cancel_token.clone());
    state
        .event_channels
        .write()
        .await
        .insert(id.clone(), event_tx.clone());

    payload.request.cancel_token = Some(cancel_token.clone());
    payload.request.cache_dir = Some(state.jobs_dir.clone());

    // Progressive rendering: create an unbounded mpsc channel so the
    // orchestrator's blocking task can push per-tile progress updates
    // without backpressure. A async forwarder task drains the receiver
    // and republishes each update as a JSON-encoded SSE payload into the
    // existing broadcast channel that `GET /api/jobs/:id/events` drinks.
    let (progress_tx, mut progress_rx) =
        tokio::sync::mpsc::unbounded_channel::<mangahue_orchestrator::ProgressUpdate>();
    payload.request.progress_sender = Some(progress_tx);

    let forward_tx = event_tx.clone();
    let progress_jobs = state.jobs.clone();
    let progress_id = id.clone();
    tokio::task::spawn(async move {
        while let Some(update) = progress_rx.recv().await {
            if let Some(job) = progress_jobs.write().await.get_mut(&progress_id) {
                if matches!(job.status, ApiJobStatus::Queued | ApiJobStatus::Running) {
                    job.progress = ApiJobProgress {
                        stage: update.stage.clone(),
                        completed: update.tiles_completed,
                        total: update.tiles_total,
                        percent: update.percent,
                    };
                    job.updated_at = chrono::Utc::now().to_rfc3339();
                }
            }
            let payload = serde_json::json!({
                "status": "progress",
                "stage": update.stage,
                "tiles_completed": update.tiles_completed,
                "tiles_total": update.tiles_total,
                "percent": update.percent,
                "tile_id": update.tile_id,
            })
            .to_string();
            let _ = forward_tx.send(payload);
        }
    });

    let job = ApiJob::new(
        id.clone(),
        ApiJobKind::Page,
        Some(payload.request.output_path.clone()),
        owner_id,
    );
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let _ = event_tx.send(r#"{"status":"queued","progress":0}"#.to_string());

    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let job_slots = state.job_slots.clone();
    let cleanup_cancel_tokens = state.cancel_tokens.clone();
    let cleanup_event_channels = state.event_channels.clone();
    let runtime = tokio::runtime::Handle::current();

    spawn_gated_blocking(job_slots, move || {
        let _ = event_tx.send(r#"{"status":"running","progress":0}"#.to_string());
        update_job_status_blocking(
            &runtime,
            &jobs,
            &id,
            &jobs_dir,
            JobUpdate {
                status: ApiJobStatus::Running,
                error: None,
                output_path: None,
                result: None,
            },
        );
        let result = MangaHueEngine::new().colorize(payload.request);
        match result {
            Ok(result) => {
                let output_path = result.output_path.clone();
                let _ = event_tx.send(r#"{"status":"complete","progress":100}"#.to_string());
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status: ApiJobStatus::Complete,
                        error: None,
                        output_path: Some(output_path),
                        result: Some(serde_json::to_value(result).unwrap_or_else(
                            |e| serde_json::json!({ "serialization_error": e.to_string() }),
                        )),
                    },
                );
            }
            Err(error) => {
                error!("job {id} failed: {error:#}");
                let cancelled = cancel_token.load(Ordering::Relaxed);
                let status = if cancelled {
                    ApiJobStatus::Cancelled
                } else {
                    ApiJobStatus::Failed
                };
                let _ = event_tx.send(
                    serde_json::json!({
                        "status": if cancelled { "cancelled" } else { "failed" },
                        "progress": 100,
                        "error": error.to_string(),
                    })
                    .to_string(),
                );
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status,
                        error: Some(error.to_string()),
                        output_path: None,
                        result: None,
                    },
                );
            }
        }
        cleanup_job_runtime_blocking(
            &runtime,
            &cleanup_cancel_tokens,
            &cleanup_event_channels,
            &id,
        );
    });

    Ok(Json(job))
}

async fn create_batch_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    Json(mut payload): Json<ApiBatchRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let owner_id = request_owner(&state, user.as_ref())?;
    let id = Uuid::new_v4().to_string();
    if state.sandbox_jobs {
        sandbox_batch_request(&mut payload, &state.jobs_dir, &id, owner_id.as_deref())?;
    }
    apply_engine_defaults(&mut payload.template, &state)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    let job = ApiJob::new(
        id.clone(),
        ApiJobKind::Batch,
        Some(payload.output_dir.clone()),
        owner_id,
    );
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let job_slots = state.job_slots.clone();
    let runtime = tokio::runtime::Handle::current();

    spawn_gated_blocking(job_slots, move || {
        update_job_status_blocking(
            &runtime,
            &jobs,
            &id,
            &jobs_dir,
            JobUpdate {
                status: ApiJobStatus::Running,
                error: None,
                output_path: None,
                result: None,
            },
        );
        match run_batch_payload(payload) {
            Ok(summary) => {
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status: ApiJobStatus::Complete,
                        error: None,
                        output_path: Some(summary.output_dir.clone()),
                        result: Some(serde_json::to_value(summary).unwrap_or_else(
                            |error| serde_json::json!({ "serialization_error": error.to_string() }),
                        )),
                    },
                );
            }
            Err(error) => {
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status: ApiJobStatus::Failed,
                        error: Some(error.to_string()),
                        output_path: None,
                        result: None,
                    },
                );
            }
        }
    });

    Ok(Json(job))
}

async fn ingest_link(
    State(state): State<ApiState>,
    Json(mut payload): Json<ApiLinkIngestRequest>,
) -> Result<Json<IngestReport>, ApiError> {
    validate_public_source_url(&payload.source_url)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    clamp_link_limits(&mut payload.max_pages, &mut payload.max_bytes_per_page);
    if state.sandbox_jobs {
        let base = state
            .jobs_dir
            .join("ingest")
            .join(Uuid::new_v4().to_string());
        std::fs::create_dir_all(&base)
            .with_context(|| format!("failed to create {}", base.display()))?;
        payload.output_dir = base.join("pages");
    }
    let report = tokio::task::spawn_blocking(move || {
        ingest_url(IngestRequest {
            source_url: payload.source_url,
            output_dir: payload.output_dir,
            max_pages: payload.max_pages,
            max_bytes_per_page: payload.max_bytes_per_page,
            user_agent: payload.user_agent,
            min_request_interval: std::time::Duration::from_millis(500),
            selectors: mangahue_ingest::default_selectors(),
        })
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(report))
}

/// Server-side ceilings for caller-supplied download limits. The values
/// also flow to `response.take(max_bytes)` in the ingest crate, where an
/// unbounded value would allow multi-GB downloads.
fn clamp_link_limits(max_pages: &mut usize, max_bytes_per_page: &mut u64) {
    *max_pages = (*max_pages).clamp(1, 500);
    *max_bytes_per_page = (*max_bytes_per_page).clamp(1, 64 * 1024 * 1024);
}

async fn create_link_colorize_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    Json(mut payload): Json<ApiLinkColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let owner_id = request_owner(&state, user.as_ref())?;
    validate_public_source_url(&payload.source_url)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    clamp_link_limits(&mut payload.max_pages, &mut payload.max_bytes_per_page);
    let id = Uuid::new_v4().to_string();
    if state.sandbox_jobs {
        sandbox_link_request(&mut payload, &state.jobs_dir, &id)?;
    }
    apply_engine_defaults(&mut payload.template, &state)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    reserve_web_job(&state, owner_id.as_deref(), payload.max_pages, &id).await?;
    let cancel_token = Arc::new(AtomicBool::new(false));
    let (event_tx, _) = broadcast::channel::<String>(64);
    state
        .cancel_tokens
        .write()
        .await
        .insert(id.clone(), cancel_token.clone());
    state
        .event_channels
        .write()
        .await
        .insert(id.clone(), event_tx.clone());
    payload.template.cancel_token = Some(cancel_token.clone());
    payload.template.cache_dir = Some(state.jobs_dir.clone());
    let job = ApiJob::new(
        id.clone(),
        ApiJobKind::Link,
        Some(payload.output_dir.clone()),
        owner_id,
    );
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let _ = event_tx.send(r#"{"status":"queued","stage":"queued","progress":0}"#.to_string());
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let job_slots = state.job_slots.clone();
    let cleanup_cancel_tokens = state.cancel_tokens.clone();
    let cleanup_event_channels = state.event_channels.clone();
    let runtime = tokio::runtime::Handle::current();

    spawn_gated_blocking(job_slots, move || {
        let _ =
            event_tx.send(r#"{"status":"running","stage":"downloading","progress":2}"#.to_string());
        update_job_status_blocking(
            &runtime,
            &jobs,
            &id,
            &jobs_dir,
            JobUpdate {
                status: ApiJobStatus::Running,
                error: None,
                output_path: None,
                result: None,
            },
        );
        match run_link_colorize_payload(
            payload,
            cancel_token.clone(),
            &event_tx,
            &runtime,
            &jobs,
            &id,
        ) {
            Ok(summary) => {
                let _ = event_tx
                    .send(r#"{"status":"complete","stage":"complete","progress":100}"#.to_string());
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status: ApiJobStatus::Complete,
                        error: None,
                        output_path: Some(summary.output_dir.clone()),
                        result: Some(serde_json::to_value(summary).unwrap_or_else(
                            |error| serde_json::json!({ "serialization_error": error.to_string() }),
                        )),
                    },
                );
            }
            Err(error) => {
                let cancelled = cancel_token.load(Ordering::Relaxed);
                let status = if cancelled {
                    ApiJobStatus::Cancelled
                } else {
                    ApiJobStatus::Failed
                };
                let _ = event_tx.send(
                    serde_json::json!({
                        "status": if cancelled { "cancelled" } else { "failed" },
                        "stage": if cancelled { "cancelled" } else { "failed" },
                        "error": error.to_string(),
                    })
                    .to_string(),
                );
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status,
                        error: Some(error.to_string()),
                        output_path: None,
                        result: None,
                    },
                );
            }
        }
        cleanup_job_runtime_blocking(
            &runtime,
            &cleanup_cancel_tokens,
            &cleanup_event_channels,
            &id,
        );
    });

    Ok(Json(job))
}

async fn import_chapter(
    Json(mut payload): Json<ApiChapterImportRequest>,
) -> Result<Json<ChapterManifest>, ApiError> {
    clamp_link_limits(&mut payload.max_pages, &mut payload.max_bytes_per_page);
    let manifest = tokio::task::spawn_blocking(move || {
        let manifest = import_chapter_to_dir(
            &payload.input_path,
            &payload.output_dir,
            ChapterImportOptions {
                max_pages: payload.max_pages,
                max_bytes_per_page: payload.max_bytes_per_page,
                overwrite: payload.overwrite,
            },
        )?;
        let manifest_path = payload
            .manifest_path
            .unwrap_or_else(|| payload.output_dir.join("mangahue_chapter_manifest.json"));
        write_chapter_manifest(&manifest, manifest_path)?;
        Ok::<_, anyhow::Error>(manifest)
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(manifest))
}

async fn export_chapter(
    Json(payload): Json<ApiChapterExportRequest>,
) -> Result<Json<ChapterManifest>, ApiError> {
    let manifest = tokio::task::spawn_blocking(move || {
        let manifest = export_chapter_archive(&payload.input_dir, &payload.output_file)?;
        if let Some(manifest_path) = payload.manifest_path {
            write_chapter_manifest(&manifest, manifest_path)?;
        }
        Ok::<_, anyhow::Error>(manifest)
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(manifest))
}

async fn create_chapter_colorize_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    Json(mut payload): Json<ApiChapterColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let owner_id = request_owner(&state, user.as_ref())?;
    clamp_link_limits(&mut payload.max_pages, &mut payload.max_bytes_per_page);
    let id = Uuid::new_v4().to_string();
    if state.sandbox_jobs {
        sandbox_chapter_request(&mut payload, &state.jobs_dir, &id, owner_id.as_deref())?;
    }
    apply_engine_defaults(&mut payload.template, &state)
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, error))?;
    reserve_web_job(&state, owner_id.as_deref(), payload.max_pages, &id).await?;
    let cancel_token = Arc::new(AtomicBool::new(false));
    let (event_tx, _) = broadcast::channel::<String>(64);
    state
        .cancel_tokens
        .write()
        .await
        .insert(id.clone(), cancel_token.clone());
    state
        .event_channels
        .write()
        .await
        .insert(id.clone(), event_tx.clone());
    payload.template.cancel_token = Some(cancel_token.clone());
    payload.template.cache_dir = Some(state.jobs_dir.clone());
    let (progress_tx, mut progress_rx) =
        tokio::sync::mpsc::unbounded_channel::<mangahue_orchestrator::ProgressUpdate>();
    payload.template.progress_sender = Some(progress_tx);

    let forward_tx = event_tx.clone();
    let progress_jobs = state.jobs.clone();
    let progress_id = id.clone();
    tokio::task::spawn(async move {
        while let Some(update) = progress_rx.recv().await {
            if let Some(job) = progress_jobs.write().await.get_mut(&progress_id) {
                if matches!(job.status, ApiJobStatus::Queued | ApiJobStatus::Running) {
                    job.progress = ApiJobProgress {
                        stage: update.stage.clone(),
                        completed: update.tiles_completed,
                        total: update.tiles_total,
                        percent: update.percent,
                    };
                    job.updated_at = chrono::Utc::now().to_rfc3339();
                }
            }
            let _ = forward_tx.send(
                serde_json::json!({
                    "status": "progress",
                    "stage": update.stage,
                    "progress": update.percent,
                    "completed": update.tiles_completed,
                    "total": update.tiles_total,
                })
                .to_string(),
            );
        }
    });
    let job = ApiJob::new(
        id.clone(),
        ApiJobKind::Chapter,
        Some(payload.output_dir.clone()),
        owner_id,
    );
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let _ = event_tx.send(r#"{"status":"queued","stage":"queued","progress":0}"#.to_string());
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let job_slots = state.job_slots.clone();
    let cleanup_cancel_tokens = state.cancel_tokens.clone();
    let cleanup_event_channels = state.event_channels.clone();
    let runtime = tokio::runtime::Handle::current();

    spawn_gated_blocking(job_slots, move || {
        let _ = event_tx
            .send(r#"{"status":"running","stage":"preparing_chapter","progress":1}"#.to_string());
        update_job_status_blocking(
            &runtime,
            &jobs,
            &id,
            &jobs_dir,
            JobUpdate {
                status: ApiJobStatus::Running,
                error: None,
                output_path: None,
                result: None,
            },
        );
        let result = run_chapter_job(ChapterJobRequest {
            input_path: payload.input_path,
            work_dir: payload.work_dir,
            output_dir: payload.output_dir,
            export_path: payload.export_path,
            gallery_path: payload.gallery_path,
            manifest_path: payload.manifest_path,
            brain_report_path: payload.brain_report_path,
            locked_style_path: payload.locked_style_path,
            palette_memory_path: payload.palette_memory_path,
            palette_memory_style_path: payload.palette_memory_style_path,
            safety_report_path: payload.safety_report_path,
            colorize_template: payload.template,
            import_options: ChapterImportOptions {
                max_pages: payload.max_pages,
                max_bytes_per_page: payload.max_bytes_per_page,
                overwrite: !payload.resume,
            },
            safety_gate: ChapterSafetyGateConfig {
                enabled: payload.safety_gate,
                enforce: payload.enforce_safety_gate,
                ..ChapterSafetyGateConfig::default()
            },
            consistency_retry: ChapterConsistencyRetry {
                enabled: payload.consistency_retry,
                palette_drift_threshold: payload.retry_drift_threshold,
                tone_drift_threshold: payload.retry_tone_threshold,
                max_pages: payload.retry_max_pages,
                max_rounds: mangahue_chapter::default_retry_max_rounds(),
            },
            resume: payload.resume,
        });
        match result {
            Ok(summary) => {
                let _ = event_tx
                    .send(r#"{"status":"complete","stage":"complete","progress":100}"#.to_string());
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status: ApiJobStatus::Complete,
                        error: None,
                        output_path: Some(summary.output_dir.clone()),
                        result: Some(serde_json::to_value(summary).unwrap_or_else(
                            |error| serde_json::json!({ "serialization_error": error.to_string() }),
                        )),
                    },
                );
            }
            Err(error) => {
                let cancelled = cancel_token.load(Ordering::Relaxed);
                let status = if cancelled {
                    ApiJobStatus::Cancelled
                } else {
                    ApiJobStatus::Failed
                };
                let _ = event_tx.send(
                    serde_json::json!({
                        "status": if cancelled { "cancelled" } else { "failed" },
                        "stage": if cancelled { "cancelled" } else { "failed" },
                        "error": error.to_string(),
                    })
                    .to_string(),
                );
                update_job_status_blocking(
                    &runtime,
                    &jobs,
                    &id,
                    &jobs_dir,
                    JobUpdate {
                        status,
                        error: Some(error.to_string()),
                        output_path: None,
                        result: None,
                    },
                );
            }
        }
        cleanup_job_runtime_blocking(
            &runtime,
            &cleanup_cancel_tokens,
            &cleanup_event_channels,
            &id,
        );
    });

    Ok(Json(job))
}

async fn inspect_page(
    Json(payload): Json<ApiInspectRequest>,
) -> Result<Json<PageInspection>, ApiError> {
    Ok(Json(MangaHueEngine::new().inspect_page(payload.request)?))
}

async fn get_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ApiJob>, ApiError> {
    let job = authorized_job(&state, &id, user.as_ref()).await?;
    Ok(Json(job))
}

async fn list_jobs(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
) -> Result<Json<Vec<ApiJob>>, ApiError> {
    let mut jobs: Vec<_> = state
        .jobs
        .read()
        .await
        .values()
        .filter(|job| !state.sandbox_jobs || job_visible_to(job, user.as_ref()))
        .cloned()
        .collect();
    jobs.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| b.id.cmp(&a.id))
    });
    jobs.truncate(250);
    Ok(Json(jobs))
}

const FILE_EXT_WHITELIST: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "json", "html", "cbz", "zip",
];

/// Serve a file from the API's file root (the jobs workspace in public
/// mode). Only image/report/archive types are served, so the endpoint can
/// never be used to exfiltrate credentials or other host files.
async fn get_local_file(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    Query(query): Query<ApiFileQuery>,
) -> Result<Response, ApiError> {
    let path = resolve_local_file(&state.file_root, &query.path)?;
    if state.sandbox_jobs {
        let job_id = query.job_id.as_deref().ok_or_else(|| {
            ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!("job_id is required"))
        })?;
        let job = authorized_job(&state, job_id, user.as_ref()).await?;
        if !job_references_path(&job, &path) {
            return Err(ApiError::with_status(
                StatusCode::FORBIDDEN,
                anyhow!("file is not attached to this job"),
            ));
        }
    }
    let allowed = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|ext| FILE_EXT_WHITELIST.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false);
    if !allowed {
        return Err(ApiError::with_status(
            StatusCode::FORBIDDEN,
            anyhow!("file type not served by the API"),
        ));
    }
    let file = tokio::fs::File::open(&path)
        .await
        .with_context(|| format!("failed to open {}", path.display()))?;
    let bytes = file
        .metadata()
        .await
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    let mut response = Response::new(Body::from_stream(ReaderStream::new(file)));
    let content_type = content_type_for(&path);
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    if let Ok(value) = HeaderValue::from_str(&bytes.to_string()) {
        response.headers_mut().insert(header::CONTENT_LENGTH, value);
    }
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=3600"),
    );
    Ok(response)
}

/// Upload a page or chapter archive into the job workspace. Returns the
/// path to pass to /api/colorize or /api/chapter/colorize.
async fn upload_file(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<serde_json::Value>, ApiError> {
    let owner_id = request_owner(&state, user.as_ref())?;
    let mut field = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(error)))?
        .ok_or_else(|| {
            ApiError::with_status(
                StatusCode::BAD_REQUEST,
                anyhow!("multipart field 'file' is required"),
            )
        })?;
    let file_name = field.file_name().map(|name| name.to_string());
    let ext = file_name
        .as_deref()
        .and_then(|name| Path::new(name).extension().and_then(|value| value.to_str()))
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| {
            matches!(
                value.as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff" | "zip" | "cbz"
            )
        })
        .ok_or_else(|| {
            ApiError::with_status(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                anyhow!("upload must be an image, ZIP, or CBZ file"),
            )
        })?;
    let dir = state
        .jobs_dir
        .join("uploads")
        .join(owner_id.as_deref().unwrap_or("local"));
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("failed to create {}", dir.display()))?;
    let name = format!("{}.{}", Uuid::new_v4(), ext);
    let path = dir.join(&name);
    let partial = dir.join(format!(".{name}.partial"));
    let mut output = tokio::fs::File::create(&partial)
        .await
        .with_context(|| format!("failed to create {}", partial.display()))?;
    let mut size = 0usize;
    let mut signature = Vec::with_capacity(16);
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|error| ApiError::with_status(StatusCode::BAD_REQUEST, anyhow!(error)))?
    {
        size = size.saturating_add(chunk.len());
        if size > 250 * 1024 * 1024 {
            drop(output);
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(ApiError::with_status(
                StatusCode::PAYLOAD_TOO_LARGE,
                anyhow!("upload exceeds 250 MB"),
            ));
        }
        if signature.len() < 16 {
            let take = (16 - signature.len()).min(chunk.len());
            signature.extend_from_slice(&chunk[..take]);
        }
        output
            .write_all(&chunk)
            .await
            .with_context(|| format!("failed to write {}", partial.display()))?;
    }
    output
        .flush()
        .await
        .with_context(|| format!("failed to flush {}", partial.display()))?;
    drop(output);
    if size == 0 || !upload_signature_matches(&ext, &signature) {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(ApiError::with_status(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            anyhow!("uploaded bytes do not match the file type"),
        ));
    }
    tokio::fs::rename(&partial, &path)
        .await
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    Ok(Json(serde_json::json!({
        "id": name,
        "path": path,
        "size": size,
        "kind": if matches!(ext.as_str(), "zip" | "cbz") { "chapter" } else { "page" },
    })))
}

fn upload_signature_matches(extension: &str, bytes: &[u8]) -> bool {
    match extension {
        "png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "jpg" | "jpeg" => bytes.starts_with(&[0xff, 0xd8, 0xff]),
        "webp" => bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP"),
        "bmp" => bytes.starts_with(b"BM"),
        "tif" | "tiff" => bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*"),
        "zip" | "cbz" => {
            bytes.starts_with(b"PK\x03\x04")
                || bytes.starts_with(b"PK\x05\x06")
                || bytes.starts_with(b"PK\x07\x08")
        }
        _ => false,
    }
}

async fn get_job_result(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job = authorized_job(&state, &id, user.as_ref()).await?;
    Ok(Json(serde_json::json!({
        "id": job.id,
        "status": job.status,
        "output_path": job.output_path,
        "result": job.result,
        "error": job.error
    })))
}

async fn list_styles(State(state): State<ApiState>) -> Result<Json<Vec<StyleSummary>>, ApiError> {
    let mut styles = Vec::new();
    if state.styles_dir.exists() {
        for entry in std::fs::read_dir(&state.styles_dir)
            .with_context(|| format!("failed to read {}", state.styles_dir.display()))?
        {
            let entry = entry.map_err(anyhow::Error::from)?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("yaml") {
                continue;
            }
            let style = load_style_profile(&path)?;
            styles.push(StyleSummary {
                id: style.id,
                name: style.name,
                path,
                notes: style.notes,
            });
        }
    }
    styles.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Json(styles))
}

async fn create_palette_memory_from_report(
    Json(payload): Json<ApiPaletteMemoryCreateRequest>,
) -> Result<Json<PaletteMemoryBook>, ApiError> {
    let report: ChapterBrainReport = serde_json::from_slice(
        &std::fs::read(&payload.brain_report_path)
            .with_context(|| format!("failed to read {}", payload.brain_report_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", payload.brain_report_path.display()))?;
    let memory = build_palette_memory(
        &report,
        PaletteMemoryRequest {
            id: payload.id,
            name: payload.name,
            source_report_path: Some(payload.brain_report_path),
            max_entries: payload.max_entries,
            min_confidence: payload.min_confidence,
            lock_entries: true,
        },
    );
    write_palette_memory(&memory, &payload.output_path)?;
    if let Some(style_output_path) = payload.style_output_path {
        let base_style = load_optional_style(payload.base_style_path.as_deref())?;
        write_palette_memory_style(&memory, &base_style, style_output_path)?;
    }
    Ok(Json(memory))
}

async fn edit_palette_memory(
    Json(payload): Json<ApiPaletteMemoryEditRequest>,
) -> Result<Json<PaletteMemoryBook>, ApiError> {
    let mut memory = load_palette_memory(&payload.memory_path)?;
    for edit in payload.edits {
        if !mangahue_brain::apply_palette_memory_edit(&mut memory, edit) {
            return Err(ApiError::with_status(
                StatusCode::NOT_FOUND,
                anyhow!("palette memory entry was not found"),
            ));
        }
    }
    let output_path = payload
        .output_path
        .clone()
        .unwrap_or_else(|| payload.memory_path.clone());
    write_palette_memory(&memory, &output_path)?;
    if let Some(style_output_path) = payload.style_output_path {
        let base_style = load_optional_style(payload.base_style_path.as_deref())?;
        write_palette_memory_style(&memory, &base_style, style_output_path)?;
    }
    Ok(Json(memory))
}

async fn preview_region_locks(
    Json(payload): Json<ApiRegionLockPreviewRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let page = load_image(&payload.input_path)?;
    let size = ImageSize::new(page.image.width(), page.image.height());
    let locks = load_region_locks(&payload.region_locks_path)?;
    let raster = rasterize_region_locks(&locks, size)?;
    std::fs::create_dir_all(&payload.output_dir)
        .with_context(|| format!("failed to create {}", payload.output_dir.display()))?;
    let color_map_path = payload.output_dir.join("region_color_map.png");
    let color_mask_path = payload.output_dir.join("region_color_mask.png");
    let preserve_mask_path = payload.output_dir.join("region_preserve_mask.png");
    let exclude_mask_path = payload.output_dir.join("region_exclude_mask.png");
    save_rgb(&color_map_path, &raster.color_map)?;
    save_gray(&color_mask_path, &raster.color_mask)?;
    save_gray(&preserve_mask_path, &raster.preserve_mask)?;
    save_gray(&exclude_mask_path, &raster.exclude_mask)?;
    Ok(Json(serde_json::json!({
        "page_id": locks.page_id,
        "size": size,
        "applied_locks": raster.applied_locks,
        "active_pixels": raster.active_pixels,
        "color_map_path": color_map_path,
        "color_mask_path": color_mask_path,
        "preserve_mask_path": preserve_mask_path,
        "exclude_mask_path": exclude_mask_path
    })))
}

async fn save_region_locks(
    Json(payload): Json<ApiRegionLockSaveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_region_locks(&payload.document)?;
    if let Some(parent) = payload.output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let document = serde_json::to_vec_pretty(&payload.document).map_err(anyhow::Error::from)?;
    std::fs::write(&payload.output_path, document).with_context(|| {
        format!(
            "failed to save region locks {}",
            payload.output_path.display()
        )
    })?;
    Ok(Json(serde_json::json!({
        "output_path": payload.output_path,
        "locks": payload.document.locks.len()
    })))
}

async fn analyze_page_semantics(
    Json(payload): Json<ApiSemanticAnalyzeRequest>,
) -> Result<Json<SemanticPageReport>, ApiError> {
    let report = analyze_semantics(&payload.input_path, &payload.config)?;
    if let Some(path) = payload.report_output.as_deref() {
        write_json_file(path, &report)?;
    }
    if let Some(path) = payload.locks_output.as_deref() {
        let locks = semantic_region_locks(&report, &payload.palette, &payload.config);
        write_json_file(path, &locks)?;
    }
    Ok(Json(report))
}

async fn track_page_characters(
    Json(payload): Json<ApiCharacterTrackRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut observations = Vec::new();
    for path in &payload.reports {
        let report: SemanticPageReport = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("failed to read semantic report {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse semantic report {}", path.display()))?;
        observations.extend(character_observations(&report));
    }
    let tracks = track_characters(&observations, payload.similarity_threshold.clamp(0.0, 1.0));
    let result = serde_json::json!({"observations": observations.len(), "tracks": tracks});
    if let Some(path) = payload.output.as_deref() {
        write_json_file(path, &result)?;
    }
    Ok(Json(result))
}

async fn curate_training_dataset(
    Json(payload): Json<ApiDatasetCurateRequest>,
) -> Result<Json<Vec<CuratedSample>>, ApiError> {
    let brain: ChapterBrainReport = read_json_file(&payload.brain_report)?;
    let safety: mangahue_brain::ChapterSafetyReport = read_json_file(&payload.safety_report)?;
    let samples = curate_dataset(payload.candidates, &brain, &safety);
    if let Some(path) = payload.output.as_deref() {
        write_json_file(path, &samples)?;
    }
    Ok(Json(samples))
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

async fn create_project(
    Json(payload): Json<ApiProjectCreateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let path = create_project_directory(&payload.output, &payload.name)?;
    Ok(Json(serde_json::json!({ "project_path": path })))
}

async fn project_status(
    Json(payload): Json<ApiProjectPathRequest>,
) -> Result<Json<ProjectMemory>, ApiError> {
    Ok(Json(ProjectMemory::load(payload.project)?))
}

async fn export_corrections(
    Json(payload): Json<ApiCorrectionExportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let count = export_corrections_jsonl(payload.project, &payload.output)?;
    Ok(Json(serde_json::json!({
        "output": payload.output,
        "corrections": count
    })))
}

async fn export_correction_hints_file(
    Json(payload): Json<ApiCorrectionHintsExportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let count = export_correction_hints(&payload.project, &payload.page, &payload.output)?;
    Ok(Json(serde_json::json!({
        "output": payload.output,
        "page": payload.page,
        "hints": count
    })))
}

async fn export_training_manifest(
    Json(payload): Json<ApiCorrectionTrainingManifestRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let summary = export_correction_training_manifest(
        payload.project,
        payload.lineart_dir,
        payload.color_dir,
        payload.hints_dir,
        payload.output,
    )?;
    Ok(Json(
        serde_json::to_value(summary).map_err(anyhow::Error::from)?,
    ))
}

async fn create_character_profile(
    Json(payload): Json<ApiCharacterCreateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hair = parse_color(payload.hair)?;
    let eyes = parse_color(payload.eyes)?;
    let skin = parse_color(payload.skin)?;
    let path = create_character(payload.project, &payload.name, hair, eyes, skin)?;
    Ok(Json(serde_json::json!({ "character_path": path })))
}

async fn create_character_profile_for_project(
    AxumPath(project_id): AxumPath<String>,
    Json(payload): Json<ApiCharacterCreateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hair = parse_color(payload.hair)?;
    let eyes = parse_color(payload.eyes)?;
    let skin = parse_color(payload.skin)?;
    let path = create_character(payload.project, &payload.name, hair, eyes, skin)?;
    Ok(Json(serde_json::json!({
        "project_id": project_id,
        "character_path": path
    })))
}

async fn add_project_correction(
    Json(payload): Json<ApiCorrectionAddRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    add_correction(
        &payload.project,
        NewCorrection {
            page: payload.page,
            region: BoundingBox::new(payload.x, payload.y, payload.width, payload.height),
            label: payload.label,
            old_color: payload
                .old
                .parse::<ColorRgb>()
                .map_err(anyhow::Error::from)?,
            new_color: payload
                .new
                .parse::<ColorRgb>()
                .map_err(anyhow::Error::from)?,
            character_id: payload.character,
        },
    )?;
    Ok(Json(serde_json::json!({
        "project": payload.project,
        "saved": true
    })))
}

fn parse_color(value: Option<String>) -> Result<Option<mangahue_core::ColorRgb>> {
    value
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn load_optional_style(path: Option<&Path>) -> Result<mangahue_core::StyleProfile> {
    match path {
        Some(path) => load_style_profile(path),
        None => Ok(mangahue_core::StyleProfile::default()),
    }
}

fn spawn_gated_blocking<F>(slots: Arc<Semaphore>, task: F)
where
    F: FnOnce() + Send + 'static,
{
    tokio::spawn(async move {
        let permit = match slots.acquire_owned().await {
            Ok(permit) => permit,
            Err(error) => {
                error!("job scheduler closed before work could start: {error}");
                return;
            }
        };
        if let Err(error) = tokio::task::spawn_blocking(task).await {
            error!("job worker task failed: {error}");
        }
        drop(permit);
    });
}

fn cleanup_job_runtime_blocking(
    runtime: &tokio::runtime::Handle,
    cancel_tokens: &Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    event_channels: &Arc<RwLock<HashMap<String, broadcast::Sender<String>>>>,
    id: &str,
) {
    runtime.block_on(async {
        cancel_tokens.write().await.remove(id);
        event_channels.write().await.remove(id);
    });
}

fn update_job_status_blocking(
    runtime: &tokio::runtime::Handle,
    jobs: &Arc<RwLock<HashMap<String, ApiJob>>>,
    id: &str,
    jobs_dir: &std::path::Path,
    update: JobUpdate,
) {
    runtime.block_on(async {
        let mut snapshot = None;
        if let Some(job) = jobs.write().await.get_mut(id) {
            job.status = update.status;
            job.updated_at = chrono::Utc::now().to_rfc3339();
            match update.status {
                ApiJobStatus::Queued => {}
                ApiJobStatus::Running => {
                    job.progress.stage = "running".to_string();
                }
                ApiJobStatus::Complete => {
                    job.progress.stage = "complete".to_string();
                    job.progress.percent = 100.0;
                    if job.progress.total > 0 {
                        job.progress.completed = job.progress.total;
                    }
                }
                ApiJobStatus::Cancelled => {
                    job.progress.stage = "cancelled".to_string();
                }
                ApiJobStatus::Failed => {
                    job.progress.stage = "failed".to_string();
                }
            }
            if let Some(error) = update.error {
                job.error = Some(error);
            }
            if let Some(output_path) = update.output_path {
                job.output_path = Some(output_path);
            }
            if let Some(result) = update.result {
                job.report_url = Some(format!("/api/jobs/{id}/result"));
                job.result = Some(result);
            }
            snapshot = Some(job.clone());
        }
        if let Some(job) = snapshot {
            if let Err(error) = persist_job(jobs_dir, &job) {
                error!("failed to persist job {}: {error:#}", job.id);
            }
        }
    });
}

fn update_job_progress_blocking(
    runtime: &tokio::runtime::Handle,
    jobs: &Arc<RwLock<HashMap<String, ApiJob>>>,
    id: &str,
    progress: ApiJobProgress,
) {
    runtime.block_on(async {
        if let Some(job) = jobs.write().await.get_mut(id) {
            job.progress = progress;
            job.updated_at = chrono::Utc::now().to_rfc3339();
        }
    });
}

async fn get_job_by_id(state: &ApiState, id: &str) -> Result<ApiJob> {
    if let Some(job) = state.jobs.read().await.get(id).cloned() {
        return Ok(job);
    }
    let job = read_job(&state.jobs_dir, id)?;
    state.jobs.write().await.insert(id.to_string(), job.clone());
    Ok(job)
}

fn safe_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

fn spawn_transient_job_retention(
    jobs: Arc<RwLock<HashMap<String, ApiJob>>>,
    jobs_dir: PathBuf,
    retention_hours: u64,
) {
    if retention_hours == 0 {
        info!("transient job retention is disabled");
        return;
    }
    tokio::spawn(async move {
        loop {
            match sweep_transient_jobs(&jobs, &jobs_dir, retention_hours).await {
                Ok(removed) if removed > 0 => {
                    info!(removed, "pruned expired transient jobs");
                }
                Ok(_) => {}
                Err(error) => warn!("transient job retention failed: {error:#}"),
            }
            tokio::time::sleep(std::time::Duration::from_secs(60 * 60)).await;
        }
    });
}

async fn sweep_transient_jobs(
    jobs: &Arc<RwLock<HashMap<String, ApiJob>>>,
    jobs_dir: &Path,
    retention_hours: u64,
) -> Result<usize> {
    let retention_seconds = retention_hours.saturating_mul(60 * 60);
    let cutoff = chrono::Utc::now()
        .timestamp()
        .saturating_sub(i64::try_from(retention_seconds).unwrap_or(i64::MAX));
    let expired = {
        let mut jobs = jobs.write().await;
        let expired_ids = jobs
            .values()
            .filter(|job| {
                matches!(
                    job.status,
                    ApiJobStatus::Complete | ApiJobStatus::Cancelled | ApiJobStatus::Failed
                ) && chrono::DateTime::parse_from_rfc3339(&job.updated_at)
                    .map(|updated| updated.timestamp() < cutoff)
                    .unwrap_or(false)
            })
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        expired_ids
            .into_iter()
            .filter_map(|id| jobs.remove(&id))
            .collect::<Vec<_>>()
    };
    let jobs_dir = jobs_dir.to_path_buf();
    let removed = expired.len();
    tokio::task::spawn_blocking(move || -> Result<()> {
        for job in expired {
            remove_expired_job_files(&jobs_dir, &job)?;
        }
        prune_old_uploads(
            &jobs_dir.join("uploads"),
            std::time::Duration::from_secs(retention_seconds),
        )?;
        Ok(())
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(removed)
}

fn remove_expired_job_files(jobs_dir: &Path, job: &ApiJob) -> Result<()> {
    if !safe_job_id(&job.id) {
        anyhow::bail!("refusing to remove files for unsafe job id {:?}", job.id);
    }
    remove_path_if_present(&jobs_dir.join(format!("{}.json", job.id)))?;
    for category in ["batch", "link", "chapter"] {
        remove_path_if_present(&jobs_dir.join(category).join(&job.id))?;
    }
    if let Some(output_path) = job.output_path.as_ref() {
        if output_path.starts_with(jobs_dir) && output_path != jobs_dir {
            remove_path_if_present(output_path)?;
        }
    }
    Ok(())
}

fn remove_path_if_present(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .with_context(|| format!("failed to remove expired path {}", path.display()))
}

fn prune_old_uploads(root: &Path, retention: std::time::Duration) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(root).min_depth(1).into_iter() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                warn!("could not inspect upload during retention: {error}");
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let old_enough = entry
            .metadata()
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| modified.elapsed().ok())
            .map(|age| age > retention)
            .unwrap_or(false);
        if old_enough {
            remove_path_if_present(entry.path())?;
        }
    }
    Ok(())
}

fn recover_job_manifests(jobs_dir: &Path) -> Result<HashMap<String, ApiJob>> {
    let mut jobs = HashMap::new();
    for entry in std::fs::read_dir(jobs_dir)
        .with_context(|| format!("failed to read {}", jobs_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                warn!(path = %path.display(), "could not read job manifest: {error}");
                continue;
            }
        };
        let value = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(value) => value,
            Err(error) => {
                warn!(path = %path.display(), "ignoring invalid JSON beside job manifests: {error}");
                continue;
            }
        };
        if value
            .get("id")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            continue;
        }
        let mut job = match serde_json::from_value::<ApiJob>(value) {
            Ok(job) => job,
            Err(error) => {
                warn!(path = %path.display(), "ignoring invalid job manifest: {error}");
                continue;
            }
        };
        if matches!(job.status, ApiJobStatus::Queued | ApiJobStatus::Running) {
            job.status = ApiJobStatus::Failed;
            job.progress.stage = "interrupted".to_string();
            job.error = Some("server restarted before this job completed".to_string());
            job.updated_at = chrono::Utc::now().to_rfc3339();
            persist_job(jobs_dir, &job)?;
        }
        jobs.insert(job.id.clone(), job);
    }
    Ok(jobs)
}

fn persist_job(jobs_dir: &std::path::Path, job: &ApiJob) -> Result<()> {
    if !safe_job_id(&job.id) {
        anyhow::bail!("refusing to persist job with unsafe id {:?}", job.id);
    }
    std::fs::create_dir_all(jobs_dir)
        .with_context(|| format!("failed to create {}", jobs_dir.display()))?;
    let path = jobs_dir.join(format!("{}.json", job.id));
    let temporary = jobs_dir.join(format!(".{}.{}.tmp", job.id, Uuid::new_v4()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(job)?)
        .with_context(|| format!("failed to write job manifest {}", temporary.display()))?;
    if let Err(first_error) = std::fs::rename(&temporary, &path) {
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to replace job manifest {}", path.display()))?;
            std::fs::rename(&temporary, &path).with_context(|| {
                format!(
                    "failed to finalize job manifest {} after rename error: {first_error}",
                    path.display()
                )
            })?;
        } else {
            return Err(first_error)
                .with_context(|| format!("failed to finalize job manifest {}", path.display()));
        }
    }
    Ok(())
}

fn read_job(jobs_dir: &std::path::Path, id: &str) -> Result<ApiJob> {
    // The id arrives URL-decoded from the path segment; an id like
    // "..%2F..%2Fsecret" must never reach the filesystem join.
    if !safe_job_id(id) {
        anyhow::bail!("invalid job id {:?}", id);
    }
    let path = jobs_dir.join(format!("{id}.json"));
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("unknown job id {id}; no manifest at {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to parse job manifest {}", path.display()))
}

fn run_batch_payload(payload: ApiBatchRequest) -> Result<ApiBatchSummary> {
    std::fs::create_dir_all(&payload.output_dir)
        .with_context(|| format!("failed to create {}", payload.output_dir.display()))?;
    let engine = MangaHueEngine::new();
    let mut processed = 0usize;
    let mut failed = 0usize;
    let mut pages = Vec::new();
    for entry in WalkDir::new(&payload.input_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let input_path = entry.path();
        if !is_image_path(input_path) {
            continue;
        }
        let output_path = payload
            .output_dir
            .join(input_path.file_name().unwrap_or_default())
            .with_extension("png");
        let mut request = payload.template.clone();
        request.input_path = input_path.to_path_buf();
        request.output_path = output_path.clone();
        match engine.colorize(request) {
            Ok(result) => {
                processed += 1;
                pages.push(ApiBatchPage {
                    input_path: input_path.to_path_buf(),
                    output_path,
                    status: ApiJobStatus::Complete,
                    error: None,
                    report: Some(result.report),
                });
            }
            Err(error) => {
                failed += 1;
                pages.push(ApiBatchPage {
                    input_path: input_path.to_path_buf(),
                    output_path,
                    status: ApiJobStatus::Failed,
                    error: Some(error.to_string()),
                    report: None,
                });
                if !payload.continue_on_error {
                    return Err(error);
                }
            }
        }
    }
    Ok(ApiBatchSummary {
        input_dir: payload.input_dir,
        output_dir: payload.output_dir,
        processed,
        failed,
        pages,
    })
}

fn run_link_colorize_payload(
    payload: ApiLinkColorizeRequest,
    cancel_token: Arc<AtomicBool>,
    event_tx: &broadcast::Sender<String>,
    runtime: &tokio::runtime::Handle,
    jobs: &Arc<RwLock<HashMap<String, ApiJob>>>,
    id: &str,
) -> Result<ApiLinkColorizeSummary> {
    std::fs::create_dir_all(&payload.output_dir)
        .with_context(|| format!("failed to create {}", payload.output_dir.display()))?;
    let downloads_root = payload
        .downloads_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".mangahue/link_downloads"));
    let downloads_dir = downloads_root.join(format!(
        "{}_{}",
        safe_url_slug(&payload.source_url),
        Uuid::new_v4()
    ));
    let ingest = ingest_url(IngestRequest {
        source_url: payload.source_url.clone(),
        output_dir: downloads_dir.clone(),
        max_pages: payload.max_pages,
        max_bytes_per_page: payload.max_bytes_per_page,
        user_agent: payload.user_agent,
        min_request_interval: std::time::Duration::from_millis(500),
        selectors: mangahue_ingest::default_selectors(),
    })?;
    if cancel_token.load(Ordering::Relaxed) {
        anyhow::bail!("job cancelled");
    }
    let engine = MangaHueEngine::new();
    let mut processed = 0usize;
    let mut failed = 0usize;
    let mut pages = Vec::new();

    let total = ingest.pages.len();
    let _ = event_tx.send(
        serde_json::json!({
            "status": "running",
            "stage": "colorizing",
            "progress": 5,
            "completed": 0,
            "total": total,
        })
        .to_string(),
    );

    for (page_index, page) in ingest.pages.iter().enumerate() {
        if cancel_token.load(Ordering::Relaxed) {
            anyhow::bail!("job cancelled");
        }
        let stem = page
            .local_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page");
        let output_path = payload.output_dir.join(format!("{stem}.png"));
        let mut request = payload.template.clone();
        request.input_path = page.local_path.clone();
        request.output_path = output_path.clone();
        request.cancel_token = Some(cancel_token.clone());
        match engine.colorize(request) {
            Ok(result) => {
                processed += 1;
                pages.push(ApiLinkColorizePage {
                    source_url: page.source_url.clone(),
                    input_path: page.local_path.clone(),
                    output_path,
                    status: ApiJobStatus::Complete,
                    error: None,
                    report: Some(result.report),
                });
            }
            Err(error) => {
                failed += 1;
                pages.push(ApiLinkColorizePage {
                    source_url: page.source_url.clone(),
                    input_path: page.local_path.clone(),
                    output_path,
                    status: ApiJobStatus::Failed,
                    error: Some(error.to_string()),
                    report: None,
                });
                if !payload.continue_on_error {
                    return Err(error);
                }
            }
        }
        let completed = page_index + 1;
        let percent = 5.0 + (completed as f32 / total.max(1) as f32) * 90.0;
        let progress = ApiJobProgress {
            stage: "colorizing".to_string(),
            completed,
            total,
            percent,
        };
        update_job_progress_blocking(runtime, jobs, id, progress.clone());
        let _ = event_tx.send(
            serde_json::json!({
                "status": "progress",
                "stage": progress.stage,
                "progress": progress.percent,
                "completed": progress.completed,
                "total": progress.total,
                "page": completed,
            })
            .to_string(),
        );
    }

    let archive_path = if processed > 0 {
        let archive_path = payload
            .output_dir
            .parent()
            .unwrap_or(&payload.output_dir)
            .join("mangahue-colored.cbz");
        export_chapter_archive(&payload.output_dir, &archive_path)?;
        Some(archive_path)
    } else {
        None
    };

    Ok(ApiLinkColorizeSummary {
        source_url: payload.source_url,
        downloads_dir,
        output_dir: payload.output_dir,
        archive_path,
        ingest,
        processed,
        failed,
        pages,
    })
}

fn default_continue_on_error() -> bool {
    true
}

fn default_max_concurrent_jobs() -> usize {
    1
}

fn default_job_retention_hours() -> u64 {
    7 * 24
}

fn default_model_path() -> PathBuf {
    PathBuf::from("models/ddcolor-tiny-fp16.onnx")
}

fn default_server_config_path() -> PathBuf {
    PathBuf::from("configs/engine.studio.yaml")
}

fn validate_default_inference_runtime(model_path: &Path) -> Result<()> {
    if !mangahue_inference::is_ddcolor_onnx_model(model_path) {
        return Ok(());
    }
    let python = std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string());
    let output = Command::new(&python)
        .args([
            "-c",
            "import numpy, onnxruntime, PIL, mangahue_train.ddcolor_onnx_worker",
        ])
        .output()
        .with_context(|| format!("failed to start inference Python runtime '{python}'"))?;
    if !output.status.success() {
        anyhow::bail!(
            "inference Python runtime is incomplete: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn default_true() -> bool {
    true
}

fn default_link_max_pages() -> usize {
    32
}

fn default_link_max_bytes_per_page() -> u64 {
    32 * 1024 * 1024
}

fn default_chapter_max_pages() -> usize {
    500
}

fn default_chapter_max_bytes_per_page() -> u64 {
    50 * 1024 * 1024
}

fn default_retry_drift_threshold() -> f32 {
    0.18
}

fn default_retry_tone_threshold() -> f32 {
    0.34
}

fn default_retry_max_pages() -> usize {
    24
}

fn default_max_attempts() -> u32 {
    3
}

fn default_character_similarity() -> f32 {
    0.92
}

fn default_palette_memory_max_entries() -> usize {
    12
}

fn default_palette_memory_min_confidence() -> f32 {
    0.18
}

fn default_config_path() -> PathBuf {
    PathBuf::from("configs/engine.default.yaml")
}

fn default_styles_dir() -> PathBuf {
    PathBuf::from("configs/styles")
}

fn default_model_metadata_path() -> PathBuf {
    PathBuf::from("models/model.mangahue.json")
}

fn default_jobs_dir() -> PathBuf {
    PathBuf::from(".mangahue/jobs")
}

fn default_studio_dir() -> PathBuf {
    PathBuf::from("web/app/out")
}

fn default_sample_input_path() -> PathBuf {
    PathBuf::from("examples/input/smoke_page.png")
}

fn default_python_command() -> String {
    std::env::var("MANGAHUE_PYTHON").unwrap_or_else(|_| "python".to_string())
}

fn check_engine_config(path: &Path) -> DoctorCheck {
    match EngineConfig::from_path(path) {
        Ok(_) => doctor_check(
            "engine_config",
            DoctorStatus::Pass,
            format!("loaded {}", path.display()),
            None,
        ),
        Err(error) => doctor_check(
            "engine_config",
            DoctorStatus::Fail,
            format!("failed to load {}: {error}", path.display()),
            None,
        ),
    }
}

fn check_styles_dir(dir: &Path) -> DoctorCheck {
    if !dir.exists() {
        return doctor_check(
            "styles",
            DoctorStatus::Fail,
            format!("styles directory does not exist: {}", dir.display()),
            None,
        );
    }
    let mut loaded = Vec::new();
    let mut failures = Vec::new();
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|value| value.to_str()) != Some("yaml") {
                    continue;
                }
                match load_style_profile(&path) {
                    Ok(style) => loaded.push(style.id),
                    Err(error) => failures.push(format!("{}: {error}", path.display())),
                }
            }
        }
        Err(error) => {
            return doctor_check(
                "styles",
                DoctorStatus::Fail,
                format!("failed to read {}: {error}", dir.display()),
                None,
            );
        }
    }
    if !failures.is_empty() {
        return doctor_check(
            "styles",
            DoctorStatus::Fail,
            format!("{} style profile(s) failed to parse", failures.len()),
            Some(serde_json::json!({ "failures": failures })),
        );
    }
    if loaded.is_empty() {
        return doctor_check(
            "styles",
            DoctorStatus::Warn,
            format!("no style profiles found in {}", dir.display()),
            None,
        );
    }
    doctor_check(
        "styles",
        DoctorStatus::Pass,
        format!("loaded {} style profile(s)", loaded.len()),
        Some(serde_json::json!({ "ids": loaded })),
    )
}

fn check_model_metadata(path: &Path) -> DoctorCheck {
    if !path.exists() {
        return doctor_check(
            "model_metadata",
            DoctorStatus::Warn,
            format!("model metadata is missing: {}", path.display()),
            None,
        );
    }
    match ModelMetadata::load(path) {
        Ok(metadata) => doctor_check(
            "model_metadata",
            DoctorStatus::Pass,
            format!(
                "loaded {} v{} ({})",
                metadata.model_name, metadata.version, metadata.model_type
            ),
            Some(serde_json::json!({
                "input_size": metadata.input_size,
                "input_channels": metadata.input_channels,
                "output_channels": metadata.output_channels
            })),
        ),
        Err(error) => doctor_check(
            "model_metadata",
            DoctorStatus::Fail,
            format!("failed to parse {}: {error}", path.display()),
            None,
        ),
    }
}

fn check_python_sidecar(python: &str) -> DoctorCheck {
    let mut command = Command::new(python);
    command
        .arg("-c")
        .arg("import mangahue_train; print('mangahue_train ok')");
    if Path::new("python").exists() {
        let mut paths = vec![PathBuf::from("python")];
        if let Some(current) = std::env::var_os("PYTHONPATH") {
            paths.extend(std::env::split_paths(&current));
        }
        if let Ok(pythonpath) = std::env::join_paths(paths) {
            command.env("PYTHONPATH", pythonpath);
        }
    }
    match command.output() {
        Ok(output) if output.status.success() => doctor_check(
            "python_sidecar",
            DoctorStatus::Pass,
            format!("{} can import mangahue_train", python),
            None,
        ),
        Ok(output) => doctor_check(
            "python_sidecar",
            DoctorStatus::Warn,
            format!(
                "{} could not import mangahue_train: {}{}",
                python,
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            ),
            None,
        ),
        Err(error) => doctor_check(
            "python_sidecar",
            DoctorStatus::Warn,
            format!("failed to start {python}: {error}"),
            None,
        ),
    }
}

fn check_jobs_dir(dir: &Path) -> DoctorCheck {
    match std::fs::create_dir_all(dir) {
        Ok(()) => doctor_check(
            "jobs_dir",
            DoctorStatus::Pass,
            format!("writable job directory: {}", dir.display()),
            None,
        ),
        Err(error) => doctor_check(
            "jobs_dir",
            DoctorStatus::Fail,
            format!("failed to create {}: {error}", dir.display()),
            None,
        ),
    }
}

fn check_studio_dir(dir: &Path) -> DoctorCheck {
    let index = dir.join("index.html");
    if index.exists() {
        doctor_check(
            "studio",
            DoctorStatus::Pass,
            format!("Studio entrypoint found at {}", index.display()),
            None,
        )
    } else {
        doctor_check(
            "studio",
            DoctorStatus::Warn,
            format!("Studio entrypoint missing: {}", index.display()),
            None,
        )
    }
}

fn check_sample_input(path: &Path) -> DoctorCheck {
    if path.exists() {
        doctor_check(
            "sample_input",
            DoctorStatus::Pass,
            format!("sample page found at {}", path.display()),
            None,
        )
    } else {
        doctor_check(
            "sample_input",
            DoctorStatus::Warn,
            format!("sample page missing: {}", path.display()),
            None,
        )
    }
}

fn doctor_check(
    name: impl Into<String>,
    status: DoctorStatus,
    message: impl Into<String>,
    details: Option<serde_json::Value>,
) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        status,
        message: message.into(),
        details,
    }
}

fn aggregate_doctor_status(checks: &[DoctorCheck]) -> DoctorStatus {
    if checks
        .iter()
        .any(|check| matches!(check.status, DoctorStatus::Fail))
    {
        DoctorStatus::Fail
    } else if checks
        .iter()
        .any(|check| matches!(check.status, DoctorStatus::Warn))
    {
        DoctorStatus::Warn
    } else {
        DoctorStatus::Pass
    }
}

fn resolve_local_file(root: &Path, path: &Path) -> Result<PathBuf> {
    let resolved = resolve_path_under(root, path)?;
    if !resolved.is_file() {
        anyhow::bail!("{} is not a file", resolved.display());
    }
    Ok(resolved)
}

fn resolve_path_under(root: &Path, path: &Path) -> Result<PathBuf> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let resolved = candidate
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", candidate.display()))?;
    if !resolved.starts_with(&root) {
        anyhow::bail!(
            "refusing to serve {} because it is outside {}",
            resolved.display(),
            root.display()
        );
    }
    Ok(resolved)
}

fn resolve_owned_upload(root: &Path, path: &Path, owner_id: Option<&str>) -> Result<PathBuf> {
    let resolved = resolve_path_under(root, path)?;
    if let Some(owner_id) = owner_id {
        let owner_root = root
            .join("uploads")
            .join(owner_id)
            .canonicalize()
            .with_context(|| "owned upload directory is missing")?;
        if !resolved.starts_with(&owner_root) {
            anyhow::bail!("input does not belong to the authenticated user");
        }
    }
    Ok(resolved)
}

fn job_references_path(job: &ApiJob, resolved: &Path) -> bool {
    fn same_file(candidate: &Path, resolved: &Path) -> bool {
        candidate
            .canonicalize()
            .map(|candidate| candidate == resolved)
            .unwrap_or(false)
    }

    fn value_references_path(value: &serde_json::Value, resolved: &Path) -> bool {
        match value {
            serde_json::Value::String(candidate) => same_file(Path::new(candidate), resolved),
            serde_json::Value::Array(values) => values
                .iter()
                .any(|value| value_references_path(value, resolved)),
            serde_json::Value::Object(values) => values
                .values()
                .any(|value| value_references_path(value, resolved)),
            _ => false,
        }
    }

    job.output_path
        .as_deref()
        .map(|path| same_file(path, resolved))
        .unwrap_or(false)
        || job
            .result
            .as_ref()
            .map(|value| value_references_path(value, resolved))
            .unwrap_or(false)
}

fn safe_url_slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = slug.trim_matches('_');
    if trimmed.is_empty() {
        "link".to_string()
    } else {
        trimmed.chars().take(64).collect()
    }
}

fn is_image_path(path: &std::path::Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "tif" | "tiff" | "bmp")
    )
}

fn content_type_for(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("cbz" | "zip") => "application/zip",
        Some("json") => "application/json",
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        _ => "application/octet-stream",
    }
}

pub fn planned_routes() -> &'static [&'static str] {
    &[
        "GET /health",
        "GET /api/doctor",
        "POST /api/doctor",
        "GET /api/file",
        "POST /api/inspect",
        "POST /api/colorize",
        "POST /api/batch",
        "POST /api/link/ingest",
        "POST /api/link/colorize",
        "POST /api/chapter/import",
        "POST /api/chapter/export",
        "POST /api/chapter/colorize",
        "POST /api/brain/palette-memory",
        "POST /api/brain/palette-memory/edit",
        "POST /api/brain/region-locks/preview",
        "GET /api/jobs",
        "GET /api/jobs/{id}",
        "GET /api/jobs/{id}/result",
        "POST /api/projects",
        "POST /api/projects/status",
        "POST /api/projects/export-corrections",
        "POST /api/projects/export-hints",
        "POST /api/projects/export-training-manifest",
        "POST /api/projects/characters",
        "POST /api/projects/{id}/characters",
        "POST /api/corrections",
        "GET /api/styles",
    ]
}

async fn job_events_sse(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, ApiError> {
    authorized_job(&state, &id, user.as_ref()).await?;
    let rx = {
        let channels = state.event_channels.read().await;
        match channels.get(&id).cloned() {
            Some(tx) => tx.subscribe(),
            None => {
                let (fake_tx, fake_rx) = broadcast::channel::<String>(1);
                let _ = fake_tx.send(r#"{"status":"not_found"}"#.to_string());
                fake_rx
            }
        }
    };
    let stream = BroadcastStream::new(rx).map(|msg| match msg {
        Ok(data) => Ok(Event::default().data(data)),
        Err(_) => Ok(Event::default().data(r#"{"status":"dropped"}"#)),
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn cancel_job(
    State(state): State<ApiState>,
    user: Option<Extension<RequestUser>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    authorized_job(&state, &id, user.as_ref()).await?;
    let token = state.cancel_tokens.read().await.get(&id).cloned();
    match token {
        Some(toggle) => {
            toggle.store(true, Ordering::Relaxed);
            if let Some(sender) = state.event_channels.read().await.get(&id) {
                let _ = sender.send(
                    r#"{"status":"cancelled","stage":"cancelled","progress":100}"#.to_string(),
                );
            }
            let snapshot = {
                let mut jobs = state.jobs.write().await;
                jobs.get_mut(&id).map(|job| {
                    job.status = ApiJobStatus::Cancelled;
                    job.progress.stage = "cancelled".to_string();
                    job.updated_at = chrono::Utc::now().to_rfc3339();
                    job.clone()
                })
            };
            if let Some(job) = snapshot {
                persist_job(&state.jobs_dir, &job)?;
            }
            Ok(Json(serde_json::json!({ "status": "cancelled", "id": id })))
        }
        None => Ok(Json(serde_json::json!({ "status": "not_found", "id": id }))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn job_manifest_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut job = ApiJob::new(
            "job_1".to_string(),
            ApiJobKind::Page,
            Some(PathBuf::from("out.png")),
            None,
        );
        job.status = ApiJobStatus::Complete;
        job.report_url = Some("/api/jobs/job_1/result".to_string());
        job.result = Some(serde_json::json!({ "ok": true }));
        persist_job(dir.path(), &job).unwrap();
        let loaded = read_job(dir.path(), "job_1").unwrap();
        assert_eq!(loaded.id, job.id);
        assert_eq!(loaded.status, ApiJobStatus::Complete);
        assert_eq!(loaded.result.unwrap()["ok"], true);
    }

    #[test]
    fn recovery_marks_interrupted_jobs_failed() {
        let dir = tempfile::tempdir().unwrap();
        let mut job = ApiJob::new("running_job".to_string(), ApiJobKind::Chapter, None, None);
        job.status = ApiJobStatus::Running;
        persist_job(dir.path(), &job).unwrap();

        let recovered = recover_job_manifests(dir.path()).unwrap();
        let recovered = recovered.get("running_job").unwrap();
        assert_eq!(recovered.status, ApiJobStatus::Failed);
        assert_eq!(recovered.progress.stage, "interrupted");
    }

    #[test]
    fn recovery_ignores_unrelated_json_reports() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("manifest.json"),
            serde_json::json!({ "pages": 12, "status": "complete" }).to_string(),
        )
        .unwrap();

        let recovered = recover_job_manifests(dir.path()).unwrap();
        assert!(recovered.is_empty());
    }

    #[tokio::test]
    async fn retention_removes_expired_terminal_jobs_and_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().join("link").join("expired");
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(output_dir.join("page.png"), b"old output").unwrap();
        let mut job = ApiJob::new(
            "expired".to_string(),
            ApiJobKind::Link,
            Some(output_dir.clone()),
            None,
        );
        job.status = ApiJobStatus::Complete;
        job.updated_at = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        persist_job(dir.path(), &job).unwrap();
        let jobs = Arc::new(RwLock::new(HashMap::from([(job.id.clone(), job)])));

        let removed = sweep_transient_jobs(&jobs, dir.path(), 1).await.unwrap();

        assert_eq!(removed, 1);
        assert!(jobs.read().await.is_empty());
        assert!(!dir.path().join("expired.json").exists());
        assert!(!output_dir.exists());
    }

    #[test]
    fn upload_signatures_reject_mismatched_extensions() {
        assert!(upload_signature_matches("png", b"\x89PNG\r\n\x1a\nrest"));
        assert!(upload_signature_matches("cbz", b"PK\x03\x04rest"));
        assert!(!upload_signature_matches("png", b"PK\x03\x04rest"));
        assert!(!upload_signature_matches("jpg", b"not an image"));
    }

    #[test]
    fn public_engine_policy_forces_trusted_model_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let styles_dir = dir.path().join("styles");
        std::fs::create_dir_all(&styles_dir).unwrap();
        let style = styles_dir.join("trusted.yaml");
        std::fs::write(&style, "id: trusted\nname: Trusted\n").unwrap();
        let state = ApiState {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            jobs_dir: dir.path().join("jobs"),
            styles_dir,
            studio_dir: dir.path().join("studio"),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
            event_channels: Arc::new(RwLock::new(HashMap::new())),
            job_slots: Arc::new(Semaphore::new(1)),
            max_concurrent_jobs: 1,
            job_retention_hours: 168,
            default_model_path: dir.path().join("trusted.onnx"),
            default_config_path: dir.path().join("trusted.yaml"),
            production: ProductionStore::open(dir.path().join("production.db")).unwrap(),
            file_root: dir.path().join("jobs"),
            sandbox_jobs: true,
            allowed_origins: vec![],
        };
        let mut request = ColorizeRequest {
            model_path: Some(PathBuf::from("attacker.pkl")),
            config_path: Some(PathBuf::from("attacker.yaml")),
            style_path: Some(style),
            use_mock_backend: true,
            ..ColorizeRequest::default()
        };

        sandbox_engine_paths(&mut request);
        apply_engine_defaults(&mut request, &state).unwrap();

        assert_eq!(
            request.model_path.as_deref(),
            Some(state.default_model_path.as_path())
        );
        assert_eq!(
            request.config_path.as_deref(),
            Some(state.default_config_path.as_path())
        );
        assert!(!request.use_mock_backend);
        assert!(request
            .style_path
            .unwrap()
            .starts_with(state.styles_dir.canonicalize().unwrap()));
    }

    #[test]
    fn job_visibility_is_owner_scoped() {
        let owned = ApiJob::new(
            "owned".to_string(),
            ApiJobKind::Page,
            None,
            Some("artist-1".to_string()),
        );
        let artist = Extension(RequestUser(UserAccount {
            id: "artist-2".to_string(),
            email: "artist@example.com".to_string(),
            display_name: "Artist".to_string(),
            tier: mangahue_production::UserTier::Free,
            active: true,
            created_at: 0,
        }));
        let admin = Extension(RequestUser(UserAccount {
            id: "admin".to_string(),
            email: "admin@example.com".to_string(),
            display_name: "Admin".to_string(),
            tier: mangahue_production::UserTier::Admin,
            active: true,
            created_at: 0,
        }));
        assert!(!job_visible_to(&owned, Some(&artist)));
        assert!(job_visible_to(&owned, Some(&admin)));
    }

    #[test]
    fn planned_routes_include_platform_surfaces() {
        let routes = planned_routes();
        assert!(routes.contains(&"GET /api/doctor"));
        assert!(routes.contains(&"POST /api/inspect"));
        assert!(routes.contains(&"POST /api/corrections"));
        assert!(routes.contains(&"POST /api/projects/export-corrections"));
        assert!(routes.contains(&"POST /api/projects/export-training-manifest"));
        assert!(routes.contains(&"POST /api/link/colorize"));
        assert!(routes.contains(&"POST /api/chapter/import"));
        assert!(routes.contains(&"POST /api/chapter/colorize"));
        assert!(routes.contains(&"POST /api/brain/palette-memory"));
        assert!(routes.contains(&"POST /api/brain/palette-memory/edit"));
        assert!(routes.contains(&"POST /api/brain/region-locks/preview"));
    }

    #[tokio::test]
    async fn public_router_exposes_health_but_hides_local_operator_routes() {
        let dir = tempfile::tempdir().unwrap();
        let state = ApiState {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            jobs_dir: dir.path().join("jobs"),
            styles_dir: dir.path().join("styles"),
            studio_dir: dir.path().join("studio"),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
            event_channels: Arc::new(RwLock::new(HashMap::new())),
            job_slots: Arc::new(Semaphore::new(1)),
            max_concurrent_jobs: 1,
            job_retention_hours: 168,
            default_model_path: dir.path().join("model.onnx"),
            default_config_path: dir.path().join("engine.yaml"),
            production: ProductionStore::open(dir.path().join("production.db")).unwrap(),
            file_root: dir.path().join("jobs"),
            sandbox_jobs: true,
            allowed_origins: vec![],
        };
        let app = router_with_mode(state, true);

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(
            health
                .headers()
                .get(header::X_CONTENT_TYPE_OPTIONS)
                .and_then(|value| value.to_str().ok()),
            Some("nosniff")
        );

        let unauthenticated = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

        // Operator and webapp surfaces exist in public mode but demand a
        // Bearer token; an anonymous caller gets 401 (or 404 for paths that
        // do not exist at all).
        for path in [
            "/api/file?path=Cargo.toml",
            "/api/doctor",
            "/api/colorize",
            "/metrics",
            "/api/jobs",
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{path} must not be anonymously accessible in public mode"
            );
        }

        let root = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(root.status(), StatusCode::NOT_FOUND);
    }
}
