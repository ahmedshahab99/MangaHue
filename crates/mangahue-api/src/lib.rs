use anyhow::{anyhow, Context, Result};
use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
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
use tokio::net::TcpListener;
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use uuid::Uuid;
use walkdir::WalkDir;

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
    pub status: ApiJobStatus,
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
    Failed,
}

#[derive(Debug, Clone)]
pub struct ApiState {
    jobs: Arc<RwLock<HashMap<String, ApiJob>>>,
    pub jobs_dir: PathBuf,
    pub styles_dir: PathBuf,
    pub studio_dir: PathBuf,
    pub cancel_tokens: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    pub event_channels: Arc<RwLock<HashMap<String, broadcast::Sender<String>>>>,
    pub production: ProductionStore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiServerConfig {
    pub addr: SocketAddr,
    pub jobs_dir: PathBuf,
    pub styles_dir: PathBuf,
    pub studio_dir: PathBuf,
    pub production_db: PathBuf,
    pub public_api: bool,
}

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
    let production = ProductionStore::open(&config.production_db)?;
    let state = ApiState {
        jobs: Arc::new(RwLock::new(HashMap::new())),
        jobs_dir: config.jobs_dir,
        styles_dir: config.styles_dir,
        studio_dir: config.studio_dir,
        cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
        event_channels: Arc::new(RwLock::new(HashMap::new())),
        production,
    };
    let app = router_with_mode(state, config.public_api);
    let listener = TcpListener::bind(config.addr).await?;
    info!("MangaHue API listening on http://{}", config.addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
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
    let app = if public_api {
        public_routes
    } else {
        public_routes
            .route("/metrics", get(metrics))
            .route(
                "/api/doctor",
                get(run_doctor_default).post(run_doctor_custom),
            )
            .route("/api/styles", get(list_styles))
            .route("/api/file", get(get_local_file))
            .route("/api/inspect", post(inspect_page))
            .route("/api/colorize", post(create_colorize_job))
            .route("/api/batch", post(create_batch_job))
            .route("/api/link/ingest", post(ingest_link))
            .route("/api/link/colorize", post(create_link_colorize_job))
            .route("/api/chapter/import", post(import_chapter))
            .route("/api/chapter/export", post(export_chapter))
            .route("/api/chapter/colorize", post(create_chapter_colorize_job))
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
            .route("/api/jobs", get(list_jobs))
            .route("/api/jobs/:id", get(get_job))
            .route("/api/jobs/:id/events", get(job_events_sse))
            .route("/api/jobs/:id/cancel", post(cancel_job))
            .route("/api/jobs/:id/result", get(get_job_result))
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
            .route("/api/corrections", post(add_project_correction))
            .fallback_service(
                ServeDir::new(&studio_dir)
                    .not_found_service(ServeFile::new(studio_dir.join("index.html"))),
            )
            .layer(CorsLayer::permissive())
    };
    app.layer(TraceLayer::new_for_http()).with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "MangaHue Engine",
        "role": "manga color director"
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
    Json(payload): Json<ApiDurableJobRequest>,
) -> Result<Json<DurableJob>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let job = state.production.enqueue(NewJob {
        user_id: user.id,
        kind: payload.kind,
        payload: payload.payload,
        priority: payload.priority,
        max_attempts: payload.max_attempts,
    })?;
    Ok(Json(job))
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
    Json(run_doctor_report(payload))
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

async fn create_colorize_job(
    State(state): State<ApiState>,
    Json(mut payload): Json<ApiColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let id = Uuid::new_v4().to_string();
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

    payload.request.cancel_token = Some(cancel_token);
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
    tokio::task::spawn(async move {
        while let Some(update) = progress_rx.recv().await {
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

    let job = ApiJob {
        id: id.clone(),
        status: ApiJobStatus::Queued,
        result_url: Some(format!("/api/jobs/{id}/result")),
        report_url: None,
        output_path: Some(payload.request.output_path.clone()),
        error: None,
        result: None,
    };
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let _ = event_tx.send(r#"{"status":"queued","progress":0}"#.to_string());

    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let runtime = tokio::runtime::Handle::current();

    tokio::task::spawn_blocking(move || {
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
                let _ = event_tx.send(format!(
                    r#"{{"status":"failed","progress":100,"error":"{}"}}"#,
                    error.to_string().replace('"', r#"\""#)
                ));
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

async fn create_batch_job(
    State(state): State<ApiState>,
    Json(payload): Json<ApiBatchRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let id = Uuid::new_v4().to_string();
    let job = ApiJob {
        id: id.clone(),
        status: ApiJobStatus::Queued,
        result_url: Some(format!("/api/jobs/{id}/result")),
        report_url: None,
        output_path: Some(payload.output_dir.clone()),
        error: None,
        result: None,
    };
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let runtime = tokio::runtime::Handle::current();

    tokio::task::spawn_blocking(move || {
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
    Json(payload): Json<ApiLinkIngestRequest>,
) -> Result<Json<IngestReport>, ApiError> {
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

async fn create_link_colorize_job(
    State(state): State<ApiState>,
    Json(payload): Json<ApiLinkColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let id = Uuid::new_v4().to_string();
    let job = ApiJob {
        id: id.clone(),
        status: ApiJobStatus::Queued,
        result_url: Some(format!("/api/jobs/{id}/result")),
        report_url: None,
        output_path: Some(payload.output_dir.clone()),
        error: None,
        result: None,
    };
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let runtime = tokio::runtime::Handle::current();

    tokio::task::spawn_blocking(move || {
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
        match run_link_colorize_payload(payload) {
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

async fn import_chapter(
    Json(payload): Json<ApiChapterImportRequest>,
) -> Result<Json<ChapterManifest>, ApiError> {
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
    Json(payload): Json<ApiChapterColorizeRequest>,
) -> Result<Json<ApiJob>, ApiError> {
    let id = Uuid::new_v4().to_string();
    let job = ApiJob {
        id: id.clone(),
        status: ApiJobStatus::Queued,
        result_url: Some(format!("/api/jobs/{id}/result")),
        report_url: None,
        output_path: Some(payload.output_dir.clone()),
        error: None,
        result: None,
    };
    state.jobs.write().await.insert(id.clone(), job.clone());
    persist_job(&state.jobs_dir, &job)?;
    let jobs = state.jobs.clone();
    let jobs_dir = state.jobs_dir.clone();
    let runtime = tokio::runtime::Handle::current();

    tokio::task::spawn_blocking(move || {
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
            },
            resume: payload.resume,
        });
        match result {
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

async fn inspect_page(
    Json(payload): Json<ApiInspectRequest>,
) -> Result<Json<PageInspection>, ApiError> {
    Ok(Json(MangaHueEngine::new().inspect_page(payload.request)?))
}

async fn get_job(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ApiJob>, ApiError> {
    let job = get_job_by_id(&state, &id).await?;
    Ok(Json(job))
}

async fn list_jobs(State(state): State<ApiState>) -> Result<Json<Vec<ApiJob>>, ApiError> {
    let mut jobs: HashMap<String, ApiJob> = HashMap::new();
    if state.jobs_dir.exists() {
        for entry in std::fs::read_dir(&state.jobs_dir)
            .with_context(|| format!("failed to read {}", state.jobs_dir.display()))?
        {
            let entry = entry.map_err(anyhow::Error::from)?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(job) = serde_json::from_str::<ApiJob>(&text) {
                    jobs.insert(job.id.clone(), job);
                }
            }
        }
    }
    for (id, job) in state.jobs.read().await.iter() {
        jobs.insert(id.clone(), job.clone());
    }
    let mut jobs: Vec<_> = jobs.into_values().collect();
    jobs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Json(jobs))
}

async fn get_local_file(Query(query): Query<ApiFileQuery>) -> Result<Response, ApiError> {
    let path = resolve_local_file(&query.path)?;
    let data = tokio::fs::read(&path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut response = Response::new(Body::from(data));
    let content_type = content_type_for(&path);
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    Ok(response)
}

async fn get_job_result(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job = get_job_by_id(&state, &id).await?;
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

async fn get_job_by_id(state: &ApiState, id: &str) -> Result<ApiJob> {
    if let Some(job) = state.jobs.read().await.get(id).cloned() {
        return Ok(job);
    }
    let job = read_job(&state.jobs_dir, id)?;
    state.jobs.write().await.insert(id.to_string(), job.clone());
    Ok(job)
}

fn persist_job(jobs_dir: &std::path::Path, job: &ApiJob) -> Result<()> {
    std::fs::create_dir_all(jobs_dir)
        .with_context(|| format!("failed to create {}", jobs_dir.display()))?;
    let path = jobs_dir.join(format!("{}.json", job.id));
    std::fs::write(&path, serde_json::to_string_pretty(job)?)
        .with_context(|| format!("failed to write job manifest {}", path.display()))
}

fn read_job(jobs_dir: &std::path::Path, id: &str) -> Result<ApiJob> {
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

fn run_link_colorize_payload(payload: ApiLinkColorizeRequest) -> Result<ApiLinkColorizeSummary> {
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
    let engine = MangaHueEngine::new();
    let mut processed = 0usize;
    let mut failed = 0usize;
    let mut pages = Vec::new();

    for page in &ingest.pages {
        let stem = page
            .local_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page");
        let output_path = payload.output_dir.join(format!("{stem}.png"));
        let mut request = payload.template.clone();
        request.input_path = page.local_path.clone();
        request.output_path = output_path.clone();
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
    }

    Ok(ApiLinkColorizeSummary {
        source_url: payload.source_url,
        downloads_dir,
        output_dir: payload.output_dir,
        ingest,
        processed,
        failed,
        pages,
    })
}

fn default_continue_on_error() -> bool {
    true
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
    PathBuf::from("web/studio")
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

fn resolve_local_file(path: &Path) -> Result<PathBuf> {
    let root = std::env::current_dir()?.canonicalize()?;
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
    if !resolved.is_file() {
        anyhow::bail!("{} is not a file", resolved.display());
    }
    Ok(resolved)
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
        Some("json") => "application/json",
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        _ => "application/octet-stream",
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    ctrl_c.await;
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
    AxumPath(id): AxumPath<String>,
) -> Sse<impl Stream<Item = anyhow::Result<Event>>> {
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
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn cancel_job(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let token = state.cancel_tokens.write().await.remove(&id);
    match token {
        Some(toggle) => {
            toggle.store(true, Ordering::Relaxed);
            let _ = state.event_channels.write().await.remove(&id);
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
        let job = ApiJob {
            id: "job_1".to_string(),
            status: ApiJobStatus::Complete,
            result_url: Some("/api/jobs/job_1/result".to_string()),
            report_url: Some("/api/jobs/job_1/result".to_string()),
            output_path: Some(PathBuf::from("out.png")),
            error: None,
            result: Some(serde_json::json!({ "ok": true })),
        };
        persist_job(dir.path(), &job).unwrap();
        let loaded = read_job(dir.path(), "job_1").unwrap();
        assert_eq!(loaded.id, job.id);
        assert_eq!(loaded.status, ApiJobStatus::Complete);
        assert_eq!(loaded.result.unwrap()["ok"], true);
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
            production: ProductionStore::open(dir.path().join("production.db")).unwrap(),
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

        for path in ["/api/file?path=Cargo.toml", "/api/doctor", "/metrics", "/"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{path} must not be exposed in public mode"
            );
        }
    }
}
