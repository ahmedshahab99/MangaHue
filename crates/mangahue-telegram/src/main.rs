use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use mangahue_export::{render_export, ExportFormat, ExportProfile, WatermarkMode};
use mangahue_ingest::{
    extract_manga_tags, extract_manga_urls, validate_public_url, SafePublicResolver,
};
use mangahue_orchestrator::{ColorizeRequest, MangaHueEngine, ProgressUpdate};
use reqwest::blocking::multipart::Form;
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod mood;
mod palette_feedback;
mod watermark;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

const DEFAULT_BATCH_SIZE: usize = 10;
const DEFAULT_DAILY_PAGE_LIMIT: usize = 30;
const DIRECT_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"];
/// Cap for user-supplied photos and URL images. 32 MP keeps the full-res
/// reconstruction buffers (five f32 planes) under ~0.7 GB on the 5.4 GB VPS
/// while still covering essentially every scanlation page.
const MAX_PHOTO_PIXELS: u64 = 32_000_000;

/// Max attempts for a Telegram API call before giving up on a transient
/// failure (5xx, 429, timeout, connection reset). Telegram intermittently
/// blocks this datacenter IP with 502s; without retries a single blip drops
/// the delivered page entirely (the engine had already colorized it).
const TELEGRAM_MAX_ATTEMPTS: usize = 3;
/// First retry delay in seconds; each retry doubles it (2s, 4s).
const TELEGRAM_RETRY_BASE_SECS: u64 = 2;

/// Telegram Bot API base URL. Defaults to api.telegram.org; the
/// `--telegram-api-base` flag / `MANGAHUE_TELEGRAM_API` env can point at a
/// Cloudflare Worker or regional proxy when Telegram blocks the VPS IP.
static TELEGRAM_API_BASE: OnceLock<String> = OnceLock::new();

fn delivery_quality_approved(verdict: Option<&str>) -> bool {
    matches!(
        verdict,
        Some(
            "production_ready_preview"
                | "production_ready_with_tone_training_warning"
                | "production_ready"
                | "already colored"
        )
    )
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeliveryFormat {
    Png,
    Jpeg,
}

const BUILTIN_STYLES: &[(&str, &str)] = &[
    ("natural_manga", "Natural Manga (default)"),
    ("realistic_comic", "Realistic Comic"),
    ("dark_seinen", "Dark Seinen"),
    ("watercolor_soft", "Watercolor Soft"),
    ("retro_90s_anime", "Retro 90s Anime"),
    // Creative mood rotation: the bot cycles these per job (unless the user
    // picked a style explicitly) so a channel never shows two identical-
    // looking pages. Skin hues are identical across all of them.
    ("creative_warm_sunset", "Creative Warm Sunset"),
    ("creative_cool_twilight", "Creative Cool Twilight"),
    ("creative_fresh_sage", "Creative Fresh Sage"),
    ("creative_soft_pastel", "Creative Soft Pastel"),
    ("creative_autumn_amber", "Creative Autumn Amber"),
    ("creative_ocean_teal", "Creative Ocean Teal"),
    ("creative_violet_dusk", "Creative Violet Dusk"),
    ("creative_warm_cocoa", "Creative Warm Cocoa"),
];

#[derive(Debug, Clone, Parser)]
#[command(
    name = "mangahue-telegram",
    about = "Telegram delivery bot for MangaHue link colorization"
)]
struct Args {
    #[arg(long)]
    token: Option<String>,
    #[arg(long, default_value = "TELEGRAM_BOT_TOKEN")]
    token_env: String,
    #[arg(
        long,
        env = "MANGAHUE_TELEGRAM_API",
        default_value = "https://api.telegram.org",
        help = "Telegram Bot API base URL; point at a Cloudflare Worker or regional proxy when the VPS IP is blocked"
    )]
    telegram_api_base: String,
    #[arg(long, default_value = ".mangahue/telegram")]
    work_dir: PathBuf,
    #[arg(
        long,
        help = "Ephemeral job workspace, ideally a bounded tmpfs such as /run/mangahue"
    )]
    scratch_dir: Option<PathBuf>,
    #[arg(
        long,
        default_value_t = 805_306_368,
        help = "Maximum bytes allowed in the ephemeral job workspace"
    )]
    scratch_limit_bytes: u64,
    #[arg(long, default_value_t = 100)]
    max_pages: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_bytes_per_page: u64,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, help = "Approved colored style or character reference image")]
    reference_image: Option<PathBuf>,
    #[arg(
        long = "ensemble-model",
        help = "Auxiliary color model; repeat to enable structure-preserving chroma fusion"
    )]
    ensemble_models: Vec<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long)]
    mock: bool,
    #[arg(long, help = "Enable multi-resolution pyramid inference")]
    pyramid: bool,
    #[arg(long, help = "Enable line-aware color-hint propagation")]
    hint_propagate: bool,
    #[arg(long, help = "Region lock file or per-page lock directory")]
    region_locks: Option<PathBuf>,
    #[arg(long, help = "Global user hints file (per-chat memory overrides)")]
    hints: Option<PathBuf>,
    #[arg(
        long,
        default_value = "configs/palettes",
        help = "Artist palette directory (configs/palettes/*.yaml)"
    )]
    palettes_dir: PathBuf,
    #[arg(
        long,
        help = "Per-chapter palette feedback: extract the first page's dominant hues and merge them as accent colors for the rest of the chapter (creative variety, model-derived)"
    )]
    palette_feedback: bool,
    #[arg(long, default_value_t = 1)]
    max_concurrent_jobs: usize,
    #[arg(
        long,
        default_value_t = 16,
        help = "Maximum waiting jobs across all users"
    )]
    max_queued_jobs: usize,
    #[arg(long, default_value_t = 1, help = "Maximum waiting jobs per user")]
    max_queued_jobs_per_chat: usize,
    #[arg(
        long,
        default_value_t = 24,
        help = "Hours to retain user inputs, references, delivered copies, and quality telemetry; 0 disables diagnostic retention"
    )]
    retention_hours: u64,
    #[arg(
        long,
        env = "MANGAHUE_ALLOWED_CHAT_IDS",
        value_delimiter = ',',
        help = "Invite-only Telegram chat IDs; empty allows all chats"
    )]
    allowed_chat_ids: Vec<i64>,
    #[arg(long, help = "Stop a chapter after the first failed page")]
    fail_fast: bool,
    #[arg(
        long,
        help = "Artifact cache directory: identical re-sent links are served instantly from cache"
    )]
    cache_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 30)]
    poll_timeout_seconds: u64,
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE, help = "Pages per media-group batch")]
    batch_size: usize,
    #[arg(long, value_enum, default_value_t = DeliveryFormat::Png)]
    delivery_format: DeliveryFormat,
    #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u8).range(80..=100))]
    jpeg_quality: u8,
    #[arg(
        long,
        default_value_t = DEFAULT_DAILY_PAGE_LIMIT,
        help = "Max pages per user per day; 0 disables the daily quota"
    )]
    daily_limit: usize,
    #[arg(
        long,
        help = "Allow standalone photo colorization (off by default: the launch product is chapter links; references via a photo captioned 'ref' always work)"
    )]
    photos_enabled: bool,
    #[arg(
        long,
        help = "Rotate creative mood styles per job (off by default: every chapter gets the stable professional look; users can still pick any style with /style)"
    )]
    creative_rotation: bool,
}

#[derive(Debug, Deserialize)]
struct TelegramEnvelope<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
    callback_query: Option<CallbackQuery>,
}

#[derive(Debug, Deserialize)]
struct Message {
    message_id: i64,
    chat: Chat,
    text: Option<String>,
    caption: Option<String>,
    photo: Option<Vec<PhotoSize>>,
    document: Option<Document>,
}

#[derive(Debug, Deserialize)]
struct Document {
    file_id: String,
    #[serde(rename = "file_name")]
    _file_name: Option<String>,
    mime_type: Option<String>,
    #[serde(default, rename = "file_size")]
    _file_size: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct PhotoSize {
    file_id: String,
    #[serde(rename = "width")]
    _width: i32,
    #[serde(rename = "height")]
    _height: i32,
    #[serde(default)]
    file_size: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct Chat {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct CallbackQuery {
    id: String,
    from: CallbackUser,
    message: Option<Message>,
    data: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CallbackUser {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct SentMessage {
    message_id: i64,
}

#[derive(Debug, Deserialize)]
struct BotIdentity {
    id: i64,
    username: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DailyUsage {
    date: String,
    pages: HashMap<i64, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerChatSettings {
    style_path: Option<PathBuf>,
    batch_size: usize,
    reference_image: Option<PathBuf>,
    /// Artist palette (configs/palettes/*.yaml) chosen per chat/series.
    palette_path: Option<PathBuf>,
    /// Auto-select the artist palette from the page fingerprint.
    auto_palette: bool,
}

/// A job waiting for a free worker slot (FIFO, shared across all chats).
#[derive(Debug, Clone, Serialize, Deserialize)]
enum QueuedJob {
    Link { url: String },
    Photo { file_id: String },
}

struct BotState {
    active_jobs: AtomicUsize,
    active_chats: Mutex<HashMap<i64, Arc<AtomicBool>>>,
    per_chat_settings: Mutex<HashMap<i64, PerChatSettings>>,
    /// Per-chat job counter driving the creative style rotation.
    job_counter: Mutex<HashMap<i64, u64>>,
    /// FIFO queue of jobs waiting for a worker slot (chat_id, job).
    queue: Mutex<VecDeque<(i64, QueuedJob)>>,
    usage_lock: Mutex<()>,
    usage_path: PathBuf,
    daily_limit: usize,
    max_queued_jobs: usize,
    max_queued_jobs_per_chat: usize,
    styles_dir: Option<PathBuf>,
    palettes_dir: PathBuf,
}

/// Lock that recovers from poisoning: a panic in a worker thread must never
/// wedge the whole bot (a poisoned lock would otherwise panic every later
/// acquirer and permanently leak chat slots).
fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn queue_admission_message(result: std::result::Result<usize, &'static str>) -> String {
    match result {
        Ok(position) => format!(
            "⏳ MangaHue is busy — you are #{position} in the queue. I'll start when a slot opens."
        ),
        Err(message) => message.to_string(),
    }
}

fn chat_is_allowed(args: &Args, chat_id: i64) -> bool {
    args.allowed_chat_ids.is_empty() || args.allowed_chat_ids.contains(&chat_id)
}

impl BotState {
    fn try_start(&self, chat_id: i64, limit: usize) -> Option<Arc<AtomicBool>> {
        let mut chats = lock_or_recover(&self.active_chats);
        if chats.contains_key(&chat_id) || self.active_jobs.load(Ordering::Acquire) >= limit.max(1)
        {
            return None;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        chats.insert(chat_id, Arc::clone(&cancel));
        self.active_jobs.fetch_add(1, Ordering::AcqRel);
        Some(cancel)
    }

    fn finish(&self, chat_id: i64) {
        lock_or_recover(&self.active_chats).remove(&chat_id);
        self.active_jobs.fetch_sub(1, Ordering::AcqRel);
    }

    /// Add a job to the bounded FIFO queue. Admission is intentionally strict
    /// on the CPU-only production host so one user cannot monopolize disk or
    /// keep an unbounded backlog alive.
    fn enqueue(&self, chat_id: i64, job: QueuedJob) -> std::result::Result<usize, &'static str> {
        let mut queue = lock_or_recover(&self.queue);
        if queue.len() >= self.max_queued_jobs {
            return Err("MangaHue is at capacity. Please try again after a current job finishes.");
        }
        let queued_for_chat = queue.iter().filter(|(cid, _)| *cid == chat_id).count();
        if queued_for_chat >= self.max_queued_jobs_per_chat {
            return Err(
                "You already have a job waiting. Please let it finish before adding another.",
            );
        }
        queue.push_back((chat_id, job));
        self.persist_queue_locked(&queue);
        Ok(queue.len())
    }

    fn queue_len(&self) -> usize {
        lock_or_recover(&self.queue).len()
    }

    fn queue_position(&self, chat_id: i64) -> Option<usize> {
        lock_or_recover(&self.queue)
            .iter()
            .position(|(cid, _)| *cid == chat_id)
    }

    /// Called when a worker slot frees up: start the next queued job.
    fn try_dispatch_next(
        self: &Arc<Self>,
        client: &Client,
        fetch_client: &Client,
        token: &str,
        args: &Args,
    ) {
        let next = {
            let mut queue = lock_or_recover(&self.queue);
            let mut candidate = None;
            // Bounded scan: a job whose owner is still active (chat sent a
            // new request while its previous job was running) cannot start
            // yet, but must never be dropped — defer it to the back and try
            // the next one. The scan is bounded by the queue length so a
            // queue of all-blocked jobs cannot spin forever.
            let max_scans = queue.len();
            for _ in 0..max_scans {
                let Some((cid, job)) = queue.pop_front() else {
                    break;
                };
                if lock_or_recover(&self.active_chats).contains_key(&cid) {
                    tracing::warn!(
                        chat_id = cid,
                        "dispatch deferring queued job: chat still has a running job"
                    );
                    queue.push_back((cid, job));
                    continue;
                }
                candidate = Some((cid, job));
                break;
            }
            candidate
        };
        // Reflect the pop (or deferral) on disk so a crash cannot resurrect
        // an already-dispatched job.
        self.persist_queue();
        let Some((chat_id, job)) = next else {
            return;
        };
        let Some(cancel_token) = self.try_start(chat_id, args.max_concurrent_jobs) else {
            // Pool still full (a race) — put the job back at the front and
            // keep the on-disk queue in sync (a crash between the pop and
            // this push_back would otherwise resurrect or drop the job).
            let mut queue = lock_or_recover(&self.queue);
            queue.push_front((chat_id, job));
            self.persist_queue_locked(&queue);
            return;
        };
        spawn_job(
            client.clone(),
            fetch_client.clone(),
            token.to_string(),
            args.clone(),
            Arc::clone(self),
            chat_id,
            job,
            cancel_token,
        );
    }

    fn is_active(&self, chat_id: i64) -> bool {
        lock_or_recover(&self.active_chats).contains_key(&chat_id)
    }

    fn cancel(&self, chat_id: i64) -> bool {
        let chats = lock_or_recover(&self.active_chats);
        let Some(cancel) = chats.get(&chat_id) else {
            return false;
        };
        cancel.store(true, Ordering::Release);
        true
    }

    fn get_settings(&self, chat_id: i64, default_batch: usize) -> PerChatSettings {
        let settings = lock_or_recover(&self.per_chat_settings);
        let mut result = settings.get(&chat_id).cloned().unwrap_or(PerChatSettings {
            style_path: None,
            batch_size: default_batch,
            reference_image: None,
            palette_path: None,
            auto_palette: false,
        });
        if result
            .reference_image
            .as_ref()
            .is_some_and(|path| !path.is_file())
        {
            result.reference_image = None;
        }
        result
    }

    fn set_reference_image(&self, chat_id: i64, path: PathBuf) {
        let mut settings = lock_or_recover(&self.per_chat_settings);
        let entry = settings.entry(chat_id).or_insert_with(|| PerChatSettings {
            style_path: None,
            batch_size: DEFAULT_BATCH_SIZE,
            reference_image: None,
            palette_path: None,
            auto_palette: false,
        });
        entry.reference_image = Some(path);
        self.persist_settings_locked(chat_id, &settings);
    }

    fn set_style(&self, chat_id: i64, path: Option<PathBuf>) {
        let mut settings = lock_or_recover(&self.per_chat_settings);
        let entry = settings.entry(chat_id).or_insert_with(|| PerChatSettings {
            style_path: None,
            batch_size: DEFAULT_BATCH_SIZE,
            reference_image: None,
            palette_path: None,
            auto_palette: false,
        });
        entry.style_path = path;
        self.persist_settings_locked(chat_id, &settings);
    }

    fn set_palette(&self, chat_id: i64, path: Option<PathBuf>, auto: bool) {
        let mut settings = lock_or_recover(&self.per_chat_settings);
        let entry = settings.entry(chat_id).or_insert_with(|| PerChatSettings {
            style_path: None,
            batch_size: DEFAULT_BATCH_SIZE,
            reference_image: None,
            palette_path: None,
            auto_palette: false,
        });
        entry.palette_path = path;
        entry.auto_palette = auto;
        self.persist_settings_locked(chat_id, &settings);
    }

    fn set_batch_size(&self, chat_id: i64, size: usize) {
        let mut settings = lock_or_recover(&self.per_chat_settings);
        let entry = settings.entry(chat_id).or_insert_with(|| PerChatSettings {
            style_path: None,
            batch_size: DEFAULT_BATCH_SIZE,
            reference_image: None,
            palette_path: None,
            auto_palette: false,
        });
        entry.batch_size = size.clamp(1, 10);
        self.persist_settings_locked(chat_id, &settings);
    }

    /// Settings live in `<work_dir>/settings/chat_<id>.json` so /style,
    /// /batch, /palette and reference choices survive restarts and deploys.
    fn settings_dir(&self) -> PathBuf {
        // usage_path = <work_dir>/usage/daily.json → parent(parent()) is the
        // work dir; the bot is always started with a work_dir.
        self.usage_path
            .parent()
            .and_then(|usage| usage.parent())
            .map(|work| work.join("settings"))
            .unwrap_or_else(|| PathBuf::from("settings"))
    }

    fn persist_settings_locked(
        &self,
        chat_id: i64,
        settings: &std::sync::MutexGuard<'_, HashMap<i64, PerChatSettings>>,
    ) {
        let Some(settings) = settings.get(&chat_id) else {
            return;
        };
        let dir = self.settings_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("chat_{chat_id}.json"));
        let tmp = path.with_extension("json.tmp");
        if let Err(error) =
            std::fs::write(&tmp, serde_json::to_string(settings).unwrap_or_default())
        {
            tracing::warn!(
                error = format!("{error:#}"),
                "failed to persist chat settings"
            );
        } else if let Err(error) = std::fs::rename(&tmp, &path) {
            tracing::warn!(
                error = format!("{error:#}"),
                "failed to commit chat settings"
            );
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Load usage, resetting the counter whenever the stored date is not
    /// today: the quota is DAILY, not cumulative.
    fn load_usage_locked(&self) -> DailyUsage {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let usage = match std::fs::read_to_string(&self.usage_path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_else(|_| DailyUsage {
                date: today.clone(),
                pages: HashMap::new(),
            }),
            Err(_) => DailyUsage {
                date: today.clone(),
                pages: HashMap::new(),
            },
        };
        if usage.date == today {
            usage
        } else {
            DailyUsage {
                date: today,
                pages: HashMap::new(),
            }
        }
    }

    fn write_usage_atomic(&self, json: &str) {
        let tmp = self.usage_path.with_extension("json.tmp");
        if let Err(error) = std::fs::write(&tmp, json) {
            tracing::warn!(error = format!("{error:#}"), "failed to persist usage file");
        } else if let Err(error) = std::fs::rename(&tmp, &self.usage_path) {
            // Windows refuses to rename over an existing file; best-effort
            // remove + rename keeps quota enforcement working everywhere.
            tracing::warn!(error = format!("{error:#}"), "failed to commit usage file");
            let _ = std::fs::remove_file(&tmp);
        }
    }

    fn record_pages(&self, chat_id: i64, count: usize) {
        // Serialize the read-modify-write so concurrent jobs cannot lose
        // updates (which would undercount and bypass the quota).
        let _guard = lock_or_recover(&self.usage_lock);
        let mut usage = self.load_usage_locked();
        *usage.pages.entry(chat_id).or_insert(0) += count;
        if let Ok(json) = serde_json::to_string(&usage) {
            self.write_usage_atomic(&json);
        }
    }

    fn remaining_pages_today(&self, chat_id: i64) -> usize {
        if self.daily_limit == 0 {
            return usize::MAX;
        }
        let _guard = lock_or_recover(&self.usage_lock);
        let usage = self.load_usage_locked();
        let used = usage.pages.get(&chat_id).copied().unwrap_or(0);
        self.daily_limit.saturating_sub(used)
    }

    fn pages_used_today(&self, chat_id: i64) -> usize {
        let _guard = lock_or_recover(&self.usage_lock);
        let usage = self.load_usage_locked();
        usage.pages.get(&chat_id).copied().unwrap_or(0)
    }

    /// /clean: forget this chat's daily usage (privacy + fresh quota).
    fn clear_usage(&self, chat_id: i64) {
        let _guard = lock_or_recover(&self.usage_lock);
        let mut usage = self.load_usage_locked();
        usage.pages.remove(&chat_id);
        if let Ok(json) = serde_json::to_string(&usage) {
            self.write_usage_atomic(&json);
        }
    }

    /// Remove a chat's queued jobs (used by /cancel when nothing is
    /// running). Returns how many were removed.
    fn dequeue(&self, chat_id: i64) -> usize {
        let mut queue = lock_or_recover(&self.queue);
        let before = queue.len();
        queue.retain(|(cid, _)| *cid != chat_id);
        let removed = before - queue.len();
        if removed > 0 {
            self.persist_queue_locked(&queue);
        }
        removed
    }

    /// Write the current queue to disk (outside the queue lock).
    fn persist_queue(&self) {
        let queue = lock_or_recover(&self.queue);
        self.persist_queue_locked(&queue);
    }

    /// The queue is persisted to `<work_dir>/settings/queue.json` so queued
    /// jobs survive restarts and deploys.
    fn persist_queue_locked(&self, queue: &std::sync::MutexGuard<'_, VecDeque<(i64, QueuedJob)>>) {
        let dir = self.settings_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("queue.json");
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string(&**queue).unwrap_or_else(|_| "[]".to_string());
        if let Err(error) = std::fs::write(&tmp, &json) {
            tracing::warn!(error = format!("{error:#}"), "failed to persist queue");
        } else if let Err(error) = std::fs::rename(&tmp, &path) {
            tracing::warn!(error = format!("{error:#}"), "failed to commit queue");
            let _ = std::fs::remove_file(&tmp);
        }
    }

    fn available_styles(&self) -> Vec<(String, String)> {
        let mut styles: Vec<(String, String)> = Vec::new();
        styles.extend(
            BUILTIN_STYLES
                .iter()
                .map(|&(id, name)| (id.to_string(), name.to_string())),
        );
        if let Some(ref dir) = self.styles_dir {
            if dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if matches!(
                            path.extension().and_then(|e| e.to_str()),
                            Some("json" | "yaml" | "yml")
                        ) {
                            let id = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let name = load_style_name(&path).unwrap_or_else(|| id.clone());
                            styles.push((id, name));
                        }
                    }
                }
            }
        }
        styles
    }

    fn resolve_style_path(
        &self,
        chat_id: i64,
        default_style: &Option<PathBuf>,
        job_override: Option<PathBuf>,
        creative_rotation: bool,
    ) -> Option<PathBuf> {
        let settings = lock_or_recover(&self.per_chat_settings);
        let explicit = settings.get(&chat_id).and_then(|s| s.style_path.clone());
        drop(settings);
        if explicit.is_some() {
            return explicit;
        }
        // Content-aware override (source tags / page mood) wins for THIS job
        // only — but only when it is a professional style, or when creative
        // rotation is explicitly enabled. Without the gate, genre tags like
        // "fantasy" inject violet-dusk palettes into every other chapter.
        if let Some(override_path) = job_override {
            if override_path.is_file() {
                let creative = override_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|id| id.starts_with("creative_"));
                if creative_rotation || !creative {
                    return Some(override_path);
                }
            }
        }
        // Creative rotation: only when explicitly enabled (--creative-rotation).
        // A channel then varies moods while skin stays natural; off by
        // default so the output is predictably professional.
        if creative_rotation {
            if let Some(creative) = self.next_creative_style(chat_id) {
                return Some(creative);
            }
        }
        default_style.clone().or_else(|| {
            self.styles_dir.as_ref().and_then(|dir| {
                let candidate = dir.join("anime_cel.json");
                if candidate.exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
        })
    }

    /// Deterministically cycle through the creative styles for a chat.
    /// Each job advances the counter, so consecutive jobs get consecutive
    /// moods; the cycle restarts after the last style.
    fn next_creative_style(&self, chat_id: i64) -> Option<PathBuf> {
        let creative_ids: Vec<&str> = BUILTIN_STYLES
            .iter()
            .map(|&(id, _)| id)
            .filter(|id| id.starts_with("creative_"))
            .collect();
        if creative_ids.is_empty() {
            return None;
        }
        let dir = self.styles_dir.as_ref()?;
        let mut counters = lock_or_recover(&self.job_counter);
        let counter = counters.entry(chat_id).or_insert(0);
        let index = (*counter % creative_ids.len() as u64) as usize;
        *counter = counter.wrapping_add(1);
        let id = creative_ids[index];
        for extension in ["yaml", "yml", "json"] {
            let candidate = dir.join(format!("{id}.{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }
}

fn load_style_name(path: &Path) -> Option<String> {
    let data = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = match path.extension().and_then(|value| value.to_str()) {
        Some("yaml" | "yml") => serde_yaml::from_str(&data).ok()?,
        _ => serde_json::from_str(&data).ok()?,
    };
    v.get("name")
        .and_then(|n| n.as_str())
        .map(|s| s.to_string())
}

/// MarkdownV2-safe " Mood: <style name>" suffix for the scan message.
/// Only the interpolated name is escaped; the literal markers stay literal.
fn mood_suffix(mood_name: &str) -> String {
    if mood_name.is_empty() {
        String::new()
    } else {
        format!(" Mood\\: *{}*\\.", md2_escape(mood_name))
    }
}

/// Escape a string for Telegram MarkdownV2 so interpolated user/artist
/// names can never break the message or inject formatting.
fn md2_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if matches!(
            ch,
            '_' | '*'
                | '['
                | ']'
                | '('
                | ')'
                | '~'
                | '`'
                | '>'
                | '#'
                | '+'
                | '-'
                | '='
                | '|'
                | '{'
                | '}'
                | '.'
                | '!'
                | '\\'
        ) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = tracing_subscriber::fmt()
        .with_env_filter("mangahue=info")
        .try_init();
    // Pre-load the ONNX Runtime shared library before anything else: ort
    // 2.0.0-rc.12 deadlocks on its load-dynamic failure path (pykeio/ort
    // #560/#604), which triggers in long-lived processes with many shared
    // libraries already mapped (#579). Loading the library here — while the
    // process is still clean — makes ort's dlopen a no-op success. Fail fast
    // instead of hanging on the first job.
    mangahue_inference::preload_onnx_runtime().with_context(|| {
        "ONNX Runtime pre-load failed; refusing to start a bot that would hang on every job"
    })?;
    let token = args
        .token
        .clone()
        .or_else(|| std::env::var(&args.token_env).ok())
        .with_context(|| {
            format!(
                "Telegram token missing; pass --token or set {}",
                args.token_env
            )
        })?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.poll_timeout_seconds + 30))
        .user_agent("MangaHueTelegram/0.2")
        .build()
        .context("failed to build Telegram HTTP client")?;
    let fetch_client = build_fetch_client()?;
    let _ = TELEGRAM_API_BASE.set(args.telegram_api_base.trim_end_matches('/').to_string());

    // Startup must survive a Telegram API outage: if getMe fails (502 block,
    // proxy down), keep retrying instead of exiting into a systemd crash loop.
    let identity = loop {
        match configure_bot(&client, &token) {
            Ok(identity) => break identity,
            Err(error) => {
                tracing::error!(
                    error = format!("{error:#}"),
                    "Telegram unreachable at startup; retrying in 30s"
                );
                std::thread::sleep(Duration::from_secs(30));
            }
        }
    };
    tracing::info!(
        bot_id = identity.id,
        username = identity.username.as_deref().unwrap_or("unknown"),
        "Telegram bot authenticated"
    );

    std::fs::create_dir_all(&args.work_dir).context("failed to create work_dir")?;
    std::fs::create_dir_all(args.work_dir.join("memory")).context("failed to create memory dir")?;
    std::fs::create_dir_all(args.work_dir.join("references"))
        .context("failed to create references dir")?;
    purge_expired_user_data(&args.work_dir, args.retention_hours)?;
    let scratch_dir = args.scratch_dir.as_ref().unwrap_or(&args.work_dir);
    std::fs::create_dir_all(scratch_dir).context("failed to create scratch_dir")?;
    if args.scratch_dir.is_some() {
        cleanup_stale_scratch(scratch_dir)?;
    }
    tracing::info!(
        scratch_dir = %scratch_dir.display(),
        scratch_limit_bytes = args.scratch_limit_bytes,
        "ephemeral job workspace ready"
    );
    let usage_dir = args.work_dir.join("usage");
    std::fs::create_dir_all(&usage_dir).context("failed to create usage dir")?;
    let usage_path = usage_dir.join("daily.json");

    let styles_dir = resolve_styles_dir(&args.style);

    if let Some(ref dir) = styles_dir {
        tracing::info!(styles_dir = %dir.display(), "style directory configured");
    } else {
        tracing::info!("no custom style directory; using built-in styles only");
    }

    let state = Arc::new(BotState {
        active_jobs: AtomicUsize::new(0),
        active_chats: Mutex::new(HashMap::new()),
        per_chat_settings: Mutex::new(HashMap::new()),
        job_counter: Mutex::new(HashMap::new()),
        queue: Mutex::new(VecDeque::new()),
        usage_lock: Mutex::new(()),
        usage_path,
        daily_limit: args.daily_limit,
        max_queued_jobs: args.max_queued_jobs.max(1),
        max_queued_jobs_per_chat: args.max_queued_jobs_per_chat.max(1),
        styles_dir,
        palettes_dir: args.palettes_dir.clone(),
    });

    // Re-hydrate per-chat reference photos after a restart so users keep
    // their saved style reference.
    restore_reference_photos(&state, &args.work_dir.join("references"));
    // Re-hydrate /style, /batch, /palette choices and queued jobs so a
    // deploy never resets user state or silently drops queued work.
    restore_chat_settings(&state);
    let restored_queue = restore_persisted_queue(&state);
    if restored_queue > 0 {
        tracing::info!(
            restored = restored_queue,
            "restored queued jobs after restart"
        );
        // The slots are free at boot: start the restored queue immediately.
        for _ in 0..args.max_concurrent_jobs {
            state.try_dispatch_next(&client, &fetch_client, &token, &args);
        }
    }

    let mut offset = None;
    let mut backoff = Duration::from_secs(1);
    let mut last_retention_purge = Instant::now();
    loop {
        if last_retention_purge.elapsed() >= Duration::from_secs(3_600) {
            if let Err(error) = purge_expired_user_data(&args.work_dir, args.retention_hours) {
                tracing::warn!(error = %error, "retention purge failed");
            }
            last_retention_purge = Instant::now();
        }
        match poll_once(&client, &fetch_client, &token, &args, &state, &mut offset) {
            Ok(()) => backoff = Duration::from_secs(1),
            Err(e) => {
                tracing::error!("poll error (will retry in {}s): {e}", backoff.as_secs());
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(60));
            }
        }
    }
}

/// HTTP client for fetching user-supplied content. Every redirect hop is
/// re-validated against the SSRF guard so a redirect to an internal host
/// (e.g. cloud metadata 169.254.169.254) is aborted.
fn build_fetch_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(45))
        // DNS-rebinding guard: the validating resolver applies to the
        // initial request and every redirect hop, and the connection uses
        // exactly the addresses it validated.
        .dns_resolver(std::sync::Arc::new(SafePublicResolver))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("too many redirects");
            }
            match validate_public_url(attempt.url()) {
                Ok(()) => attempt.follow(),
                Err(error) => attempt.error(error),
            }
        }))
        .build()
        .context("failed to build fetch client")
}

fn resolve_styles_dir(style_arg: &Option<PathBuf>) -> Option<PathBuf> {
    let path = style_arg.as_ref()?;
    if path.is_dir() {
        Some(path.clone())
    } else if path.is_file() {
        path.parent().map(|p| p.to_path_buf())
    } else {
        None
    }
}

fn poll_once(
    client: &Client,
    fetch_client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    offset: &mut Option<i64>,
) -> Result<()> {
    let updates = get_updates(client, token, *offset, args.poll_timeout_seconds)?;
    for update in updates {
        let update_id = update.update_id;
        // A failing update must never drop the rest of the batch: log and
        // continue, then advance the offset so Telegram stops redelivering.
        if let Err(error) = process_update(client, fetch_client, token, args, state, update) {
            tracing::error!("update {update_id} failed: {error:#}");
        }
        *offset = Some(update_id + 1);
    }
    Ok(())
}

fn process_update(
    client: &Client,
    fetch_client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    update: Update,
) -> Result<()> {
    if let Some(cb) = update.callback_query {
        let chat_id = cb.message.as_ref().map(|m| m.chat.id).unwrap_or(cb.from.id);
        if !chat_is_allowed(args, chat_id) {
            answer_callback_query(
                client,
                token,
                &cb.id,
                Some("MangaHue is currently invite-only"),
                Some(true),
            )?;
            return Ok(());
        }
        handle_callback(client, token, args, state, cb)?;
        return Ok(());
    }

    let Some(message) = update.message else {
        return Ok(());
    };

    let chat_id = message.chat.id;
    if !chat_is_allowed(args, chat_id) {
        send_message(
            client,
            token,
            chat_id,
            "MangaHue is currently in a small invite-only quality beta. Access will open after the release benchmark passes.",
            None,
        )?;
        return Ok(());
    }
    let caption = message
        .caption
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();

    // A photo/document message is COLORIZED by default; with a caption
    // starting with "ref" or "style" it sets the chat's reference look.
    // Must be checked BEFORE the text check: photos have text == None.
    let image_file_id: Option<String> = message
        .photo
        .as_ref()
        .and_then(|photos| photos.iter().max_by_key(|p| p.file_size.unwrap_or(0)))
        .map(|p| p.file_id.clone())
        .or_else(|| {
            message.document.as_ref().and_then(|doc| {
                let is_image = doc
                    .mime_type
                    .as_deref()
                    .map(|m| m.starts_with("image/"))
                    .unwrap_or(false);
                if is_image {
                    Some(doc.file_id.clone())
                } else {
                    None
                }
            })
        });

    if let Some(file_id) = image_file_id {
        if caption.starts_with("ref") || caption.starts_with("style") {
            match save_reference_photo(client, token, args, state, chat_id, &file_id) {
                Ok(_) => {
                    let _ = send_message(
                            client,
                            token,
                            chat_id,
                            "Reference style photo saved. Send a photo or manga link to colorize; pages will be graded to match this look. Use /resetref to clear it.",
                            None,
                        );
                }
                Err(error) => {
                    let _ = send_message(
                        client,
                        token,
                        chat_id,
                        &format!("Could not save reference photo: {error}"),
                        None,
                    );
                }
            }
            return Ok(());
        }

        // Colorize the image directly.
        // Launch mode is chapter-first: standalone photo colorization is gated
        // behind --photos-enabled (references captioned "ref"/"style" above
        // always work).
        if !args.photos_enabled {
            let _ = send_message(
                client,
                token,
                chat_id,
                "Single-photo colorization is coming soon. For now, send a comic chapter link and I will colorize the whole chapter for you!",
                None,
            );
            return Ok(());
        }
        if state.remaining_pages_today(chat_id) == 0 {
            send_message(
                    client,
                    token,
                    chat_id,
                    &format!(
                        "Daily limit of {} pages reached. Try again tomorrow or use /stats to see your usage.",
                        state.daily_limit
                    ),
                    None,
                )?;
            return Ok(());
        }
        // Fairness first: never jump ahead of an existing queue, and never
        // take a slot we might abandon — an abandoned slot is leaked forever
        // (active_jobs never decremented) and wedges the whole bot once all
        // slots are gone.
        if state.queue_len() > 0 {
            let message = queue_admission_message(state.enqueue(
                chat_id,
                QueuedJob::Photo {
                    file_id: file_id.clone(),
                },
            ));
            send_message(client, token, chat_id, &message, None)?;
            return Ok(());
        }
        let Some(cancel_token) = state.try_start(chat_id, args.max_concurrent_jobs) else {
            // A slot was taken between the queue check and now — queue
            // instead of blocking (never leak the slot we did not take).
            let message = queue_admission_message(state.enqueue(
                chat_id,
                QueuedJob::Photo {
                    file_id: file_id.clone(),
                },
            ));
            send_message(client, token, chat_id, &message, None)?;
            return Ok(());
        };
        let client = client.clone();
        let fetch_client = fetch_client.clone();
        let token = token.to_string();
        let args = args.clone();
        let state = Arc::clone(state);
        spawn_job(
            client,
            fetch_client,
            token,
            args,
            state,
            chat_id,
            QueuedJob::Photo {
                file_id: file_id.clone(),
            },
            cancel_token,
        );
        return Ok(());
    }

    let Some(text) = message.text.as_deref() else {
        if message.text.is_none() {
            send_message(
                client,
                token,
                message.chat.id,
                "Send an image, a manga chapter link, or an image URL and I will colorize it.",
                None,
            )?;
        }
        return Ok(());
    };

    let text = text.trim();

    if text.starts_with('/') {
        handle_command(
            client,
            token,
            args,
            state,
            chat_id,
            text,
            message.message_id,
        )?;
        return Ok(());
    }

    let url = match first_url(text) {
        Some(u) => u,
        None => {
            send_message(
                client,
                token,
                chat_id,
                "Send an authorized manga chapter or image URL. Use /help for available commands.",
                None,
            )?;
            return Ok(());
        }
    };

    let validation = validate_user_url(&url, args.photos_enabled);
    if let Some(warning) = validation.warning {
        send_message(client, token, chat_id, &warning, None)?;
    }
    if let Some(error) = validation.error {
        send_message(client, token, chat_id, &error, None)?;
        return Ok(());
    }

    let remaining = state.remaining_pages_today(chat_id);
    if remaining == 0 {
        send_message(
                client,
                token,
                chat_id,
                &format!(
                    "Daily limit of {} pages reached. Try again tomorrow or use /stats to see your usage.",
                    state.daily_limit
                ),
                None,
            )?;
        return Ok(());
    }

    // Fairness first: never jump ahead of an existing queue, and never
    // take a slot we might abandon (abandoned slots leak forever and wedge
    // the bot once all slots are gone).
    if state.queue_len() > 0 {
        let message =
            queue_admission_message(state.enqueue(chat_id, QueuedJob::Link { url: url.clone() }));
        send_message(client, token, chat_id, &message, None)?;
        return Ok(());
    }
    let Some(cancel_token) = state.try_start(chat_id, args.max_concurrent_jobs) else {
        // A slot was taken between the queue check and now — queue instead.
        let message =
            queue_admission_message(state.enqueue(chat_id, QueuedJob::Link { url: url.clone() }));
        send_message(client, token, chat_id, &message, None)?;
        return Ok(());
    };

    let client = client.clone();
    let fetch_client = fetch_client.clone();
    let token = token.to_string();
    let args = args.clone();
    let state = Arc::clone(state);
    spawn_job(
        client,
        fetch_client,
        token,
        args,
        state,
        chat_id,
        QueuedJob::Link { url: url.clone() },
        cancel_token,
    );
    Ok(())
}

/// Spawn a job (link or photo) on a worker thread; on completion, release
/// the chat slot and dispatch the next queued job. Errors are reported to
/// the user instead of being dropped.
#[allow(clippy::too_many_arguments)]
fn spawn_job(
    client: Client,
    fetch_client: Client,
    token: String,
    args: Args,
    state: Arc<BotState>,
    chat_id: i64,
    job: QueuedJob,
    cancel_token: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &job {
            QueuedJob::Link { url } => handle_link(
                &client,
                &fetch_client,
                &token,
                &args,
                &state,
                chat_id,
                url,
                cancel_token,
            ),
            QueuedJob::Photo { file_id } => handle_photo(
                &client,
                &token,
                &args,
                &state,
                chat_id,
                file_id,
                cancel_token,
            ),
        }));
        // Always release the chat slot, even on error or panic.
        state.finish(chat_id);
        // A slot freed up: start the next queued job (if any).
        state.try_dispatch_next(&client, &fetch_client, &token, &args);
        match result {
            Ok(Err(error)) => {
                let _ = send_message(
                    &client,
                    &token,
                    chat_id,
                    &format!("MangaHue could not finish this job: {error}"),
                    None,
                );
            }
            Err(error) => {
                let message = if let Some(payload) = error.downcast_ref::<&str>() {
                    (*payload).to_string()
                } else if let Some(payload) = error.downcast_ref::<String>() {
                    payload.clone()
                } else {
                    "internal error".to_string()
                };
                let _ = send_message(
                    &client,
                    &token,
                    chat_id,
                    &format!("MangaHue could not finish this job: {message}"),
                    None,
                );
            }
            Ok(Ok(())) => {}
        }
    });
}

fn handle_command(
    client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    chat_id: i64,
    text: &str,
    _message_id: i64,
) -> Result<()> {
    let cmd = text
        .split_whitespace()
        .next()
        .unwrap_or(text)
        .to_lowercase();
    let arg = text
        .split_once(char::is_whitespace)
        .map(|(_, rest)| rest.trim())
        .unwrap_or("")
        .to_string();

    match cmd.as_str() {
        "/start" => cmd_start(client, token, chat_id),
        "/help" => cmd_help(client, token, state, chat_id),
        "/status" => cmd_status(client, token, state, chat_id),
        "/queue" => cmd_queue(client, token, state, chat_id),
        "/cancel" => cmd_cancel(client, token, state, chat_id),
        "/style" => cmd_style(client, token, state, chat_id, &arg),
        "/palette" => cmd_palette(client, token, state, chat_id, &arg),
        "/batch" => cmd_batch(client, token, state, chat_id, &arg, args.batch_size),
        "/stats" => cmd_stats(client, token, state, chat_id),
        "/clean" => cmd_clean(client, token, state, args, chat_id),
        "/resetref" => cmd_reset_reference(client, token, args, state, chat_id),
        "/memory" => cmd_memory(client, token, args, chat_id),
        "/color" => cmd_color(client, token, args, chat_id, &arg),
        "/rate" => cmd_rate(client, token, args, chat_id, &arg),
        _ => {
            send_message(
                client,
                token,
                chat_id,
                "Unknown command. Use /help to see available commands.",
                None,
            )?;
            Ok(())
        }
    }
}

fn cmd_start(client: &Client, token: &str, chat_id: i64) -> Result<()> {
    let text = "Welcome to MangaHue. Send a photo, a manga chapter link, or an image URL and I will colorize it in soft cel-shaded anime style.\n\n\
                Use /help to learn about all available commands.";
    send_message(client, token, chat_id, text, None)?;
    Ok(())
}

fn cmd_reset_reference(
    client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    chat_id: i64,
) -> Result<()> {
    state.set_reference_image(chat_id, PathBuf::new());
    // Clear the reference on disk too, or it resurrects after a restart
    // (restore_reference_photos re-registers whatever is on disk).
    let ref_file = args
        .work_dir
        .join("references")
        .join(format!("chat_{chat_id}.jpg"));
    if let Err(error) = std::fs::remove_file(&ref_file) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(
                error = format!("{error:#}"),
                "failed to remove reference file"
            );
        }
    }
    send_message(
        client,
        token,
        chat_id,
        "Reference style photo cleared. Pages will use the default grading.",
        None,
    )?;
    Ok(())
}

fn cmd_memory(client: &Client, token: &str, args: &Args, chat_id: i64) -> Result<()> {
    let memory = args
        .work_dir
        .join("memory")
        .join(format!("chat_{chat_id}.hints.json"));
    let text = if memory.is_file() {
        let size = std::fs::metadata(&memory).map(|m| m.len()).unwrap_or(0);
        format!(
            "Your color memory is active ({size} bytes of learned hints). It is applied automatically to every page you colorize. Corrections will refine it over time."
        )
    } else {
        "No learned color memory yet. It builds automatically from your corrections over time."
            .to_string()
    };
    send_message(client, token, chat_id, &text, None)?;
    Ok(())
}

#[derive(Serialize)]
struct BetaFeedback<'a> {
    recorded_at: String,
    chat_id: i64,
    rating: u8,
    note: &'a str,
    release: &'a str,
}

fn cmd_rate(client: &Client, token: &str, args: &Args, chat_id: i64, arg: &str) -> Result<()> {
    let mut parts = arg.splitn(2, char::is_whitespace);
    let rating = parts.next().and_then(|value| value.parse::<u8>().ok());
    let Some(rating @ 1..=5) = rating else {
        send_message(
            client,
            token,
            chat_id,
            "Usage: /rate <1-5> [optional note]. Example: /rate 4 Great colors, but the hair changed.",
            None,
        )?;
        return Ok(());
    };
    let note: String = parts
        .next()
        .unwrap_or("")
        .trim()
        .chars()
        .take(500)
        .collect();
    let day = Utc::now().format("%Y-%m-%d");
    let directory = args.work_dir.join("feedback").join(day.to_string());
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    let record = BetaFeedback {
        recorded_at: Utc::now().to_rfc3339(),
        chat_id,
        rating,
        note: &note,
        release: env!("CARGO_PKG_VERSION"),
    };
    let path = directory.join("ratings.jsonl");
    use std::io::Write;
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    writeln!(output, "{}", serde_json::to_string(&record)?)?;
    send_message(
        client,
        token,
        chat_id,
        "Thank you. Your blind quality rating was recorded and will be included in the beta release review.",
        None,
    )?;
    Ok(())
}

/// /color <x> <y> <radius> #RRGGBB — add a color memory hint (normalized 0-1 coords).
fn cmd_color(client: &Client, token: &str, args: &Args, chat_id: i64, arg: &str) -> Result<()> {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() < 4 {
        send_message(
            client,
            token,
            chat_id,
            "Usage: /color <x> <y> <radius> <hex>  — e.g. /color 0.5 0.3 0.05 #f5c9a8 (coordinates are 0-1, relative to the page)",
            None,
        )?;
        return Ok(());
    }
    let x: f32 = parts[0].parse().unwrap_or(-1.0);
    let y: f32 = parts[1].parse().unwrap_or(-1.0);
    let radius: f32 = parts[2].parse().unwrap_or(-1.0);
    let hex = parts[3].trim_start_matches('#');
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&radius) {
        send_message(
            client,
            token,
            chat_id,
            "Coordinates and radius must be between 0 and 1.",
            None,
        )?;
        return Ok(());
    }
    let rgb = u32::from_str_radix(hex, 16);
    let rgb = match rgb {
        Ok(v) if v <= 0xFF_FFFF => v,
        _ => {
            send_message(
                client,
                token,
                chat_id,
                "Invalid hex color. Use #RRGGBB, e.g. #f5c9a8.",
                None,
            )?;
            return Ok(());
        }
    };
    let (r, g, b) = ((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8);

    let memory_path = args
        .work_dir
        .join("memory")
        .join(format!("chat_{chat_id}.hints.json"));
    let mut file: mangahue_core::UserHintFile = std::fs::read_to_string(&memory_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    file.hints.push(mangahue_core::ColorHint {
        x: (x * 1000.0) as u32,
        y: (y * 1000.0) as u32,
        radius: (radius * 1000.0) as u32,
        color: mangahue_core::ColorRgb { r, g, b },
        label: Some("user_memory".to_string()),
        strength: 0.9,
        source: mangahue_core::HintSource::User,
    });
    // The memory file is re-parsed for every page of every job: cap it so an
    // automated flood of /color commands cannot balloon it into multi-MB
    // parses per page. Newest hints are kept (they override older ones).
    const MAX_USER_HINTS: usize = 200;
    if file.hints.len() > MAX_USER_HINTS {
        let split = file.hints.len() - MAX_USER_HINTS;
        file.hints.drain(..split);
    }
    std::fs::write(&memory_path, serde_json::to_string_pretty(&file)?)
        .context("failed to save color memory")?;
    send_message(
        client,
        token,
        chat_id,
        &format!(
            "Color memory saved: #{:02X}{:02X}{:02X} at ({:.2}, {:.2}) r={:.2}. It applies to your future pages.",
            r, g, b, x, y, radius
        ),
        None,
    )?;
    Ok(())
}

/// Download a Telegram photo into the chat's reference file and register it.
fn save_reference_photo(
    client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    chat_id: i64,
    file_id: &str,
) -> Result<PathBuf> {
    let bytes = fetch_telegram_file(client, token, file_id)?;
    let refs_dir = args.work_dir.join("references");
    std::fs::create_dir_all(&refs_dir).context("failed to create references dir")?;
    let local_path = refs_dir.join(format!("chat_{chat_id}.jpg"));
    std::fs::write(&local_path, &bytes).context("failed to save reference photo")?;

    state.set_reference_image(chat_id, local_path.clone());
    Ok(local_path)
}

/// Download a Telegram file (photo/document) by file_id to bytes.
fn fetch_telegram_file(client: &Client, token: &str, file_id: &str) -> Result<Vec<u8>> {
    const MAX_TELEGRAM_FILE_BYTES: u64 = 64 * 1024 * 1024;
    #[derive(Debug, Deserialize)]
    struct FileEnvelope {
        ok: bool,
        result: Option<FileResult>,
        description: Option<String>,
    }
    #[derive(Debug, Deserialize)]
    struct FileResult {
        file_path: Option<String>,
    }

    let envelope: FileEnvelope = telegram_retry("getFile", || {
        // The inner let pins the Deserialize target so the trailing `?` has
        // an anchor inside the closure (tail-`?` chains confuse inference).
        let envelope: FileEnvelope = client
            .get(telegram_url(token, "getFile"))
            .query(&[("file_id", file_id)])
            .send()
            .context("failed to call getFile")?
            .error_for_status()
            .context("Telegram getFile failed")?
            .json()
            .context("failed to parse getFile response")?;
        Ok(envelope)
    })?;
    if !envelope.ok {
        anyhow::bail!(
            "Telegram getFile failed: {}",
            envelope.description.as_deref().unwrap_or("unknown error")
        );
    }
    let remote_path = envelope
        .result
        .and_then(|r| r.file_path)
        .ok_or_else(|| anyhow::anyhow!("Telegram did not return a file path"))?;

    // File downloads live under the same API base so a proxy covers them too.
    let url = format!("{}/file/bot{token}/{remote_path}", api_base());
    // Cap the body: an unbounded read lets a huge upload fill RAM before
    // any size check runs.
    let response = telegram_retry("telegram file download", || {
        client
            .get(&url)
            .send()
            .context("failed to download Telegram file")?
            .error_for_status()
            .context("Telegram file download failed")
    })?;
    use std::io::Read as _;
    let mut reader = response.take(MAX_TELEGRAM_FILE_BYTES + 1);
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .context("failed to read Telegram file body")?;
    if bytes.len() as u64 > MAX_TELEGRAM_FILE_BYTES {
        anyhow::bail!("Telegram file exceeds {MAX_TELEGRAM_FILE_BYTES} bytes");
    }
    Ok(bytes)
}

/// Colorize a single image sent as a photo/document and send the result back.
fn handle_photo(
    client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    chat_id: i64,
    file_id: &str,
    cancel_token: Arc<AtomicBool>,
) -> Result<()> {
    let job_id = Uuid::new_v4().to_string();
    let scratch_root = args.scratch_dir.as_ref().unwrap_or(&args.work_dir);
    let job_dir = scratch_root.join(chat_id.to_string()).join(&job_id);
    let _workspace = JobWorkspace::new(
        job_dir.clone(),
        args.work_dir.clone(),
        args.retention_hours > 0,
    );
    let downloads_dir = job_dir.join("downloads");
    let output_dir = job_dir.join("colored");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    std::fs::create_dir_all(&downloads_dir)
        .with_context(|| format!("failed to create {}", downloads_dir.display()))?;

    let cancel_kb = InlineKeyboardMarkup {
        inline_keyboard: vec![vec![
            InlineKeyboardButton::new("Cancel job").callback_data("cancel_job")
        ]],
    };
    let cancel_markup = serde_json::to_string(&cancel_kb).unwrap_or_default();

    send_chat_action(client, token, chat_id, "typing")?;
    let status_msg_id = send_message(
        client,
        token,
        chat_id,
        "Got it! Coloring your image… This takes about 3 minutes. You can cancel anytime.",
        Some(&cancel_markup),
    )?;

    if cancel_token.load(Ordering::Acquire) {
        let _ = edit_message(client, token, chat_id, status_msg_id, "Cancelled.");
        cleanup_dir(&job_dir);
        return Ok(());
    }

    // Download + validate the image.
    let bytes = fetch_telegram_file(client, token, file_id)?;
    if bytes.len() > args.max_bytes_per_page as usize {
        cleanup_dir(&job_dir);
        anyhow::bail!("image exceeds {} bytes", args.max_bytes_per_page);
    }
    let local_path = downloads_dir.join("image.jpg");
    std::fs::write(&local_path, &bytes).context("failed to save uploaded image")?;
    // Keep the INPUT image too (the exact bytes the user sent), so quality
    // complaints can be traced back to the source.
    if args.retention_hours > 0 {
        let _ = retain_input_copy(&local_path, &args.work_dir);
    }
    // Header-only dimension validation BEFORE decode: a small file with a
    // huge IHDR would otherwise allocate gigabytes at decode time.
    let (img_width, img_height) = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .context("uploaded file is not a readable image")?
        .into_dimensions()
        .context("uploaded file has invalid dimensions")?;
    if img_width > 50_000
        || img_height > 50_000
        || img_width as u64 * img_height as u64 > MAX_PHOTO_PIXELS
    {
        cleanup_dir(&job_dir);
        anyhow::bail!("image dimensions exceed security limits");
    }
    let image = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .context("uploaded file is not a readable image")?
        .decode()
        .context("uploaded file is not a readable image")?;
    if image.width() < 64 || image.height() < 64 {
        cleanup_dir(&job_dir);
        anyhow::bail!("that image is too small to colorize (minimum 64x64 px)");
    }

    let settings = state.get_settings(chat_id, args.batch_size);
    // Content-aware mood: pick the style from the photo's own lighting
    // (dark ink-heavy pages -> dark styles, bright airy pages -> pastels).
    let mood_override = mood::select_style_for_scene(&image.to_luma8())
        .and_then(|id| resolve_style_file(state, id));
    let style_path =
        state.resolve_style_path(chat_id, &args.style, mood_override, args.creative_rotation);
    let reference_image = settings
        .reference_image
        .clone()
        .filter(|p| !p.as_os_str().is_empty());

    let output_path = output_dir.join("image.png");
    let (progress_sender, mut progress_receiver) =
        tokio::sync::mpsc::unbounded_channel::<ProgressUpdate>();
    let progress_client = client.clone();
    let progress_token = token.to_string();
    let progress_msg_id2 = status_msg_id;
    let progress_chat = chat_id;
    let progress_markup = cancel_markup.clone();
    let progress_thread = std::thread::spawn(move || {
        let mut last_stage = String::new();
        let mut last_percent = -10i32;
        while let Some(update) = progress_receiver.blocking_recv() {
            let percent = update.percent.round() as i32;
            if update.stage != last_stage || percent >= last_percent + 10 || percent >= 100 {
                let text = format!(
                    "Coloring your image: {} ({}%)",
                    humanize_stage(&update.stage),
                    percent.clamp(0, 100)
                );
                let _ = edit_message_with_markup(
                    &progress_client,
                    &progress_token,
                    progress_chat,
                    progress_msg_id2,
                    &text,
                    Some(&progress_markup),
                );
                last_stage = update.stage;
                last_percent = percent;
            }
        }
    });

    let engine = MangaHueEngine::new();
    let result = engine.colorize(ColorizeRequest {
        input_path: local_path.clone(),
        output_path: output_path.clone(),
        model_path: args.model.clone(),
        reference_image_path: reference_image,
        enable_ensemble: !args.ensemble_models.is_empty(),
        ensemble_model_paths: args.ensemble_models.clone(),
        config_path: args.config.clone(),
        preset_path: args.preset.clone(),
        style_path: style_path.clone(),
        project_path: args.project.clone(),
        region_locks_path: None,
        user_hints_path: resolve_user_hints(args, chat_id, "image"),
        use_mock_backend: args.mock,
        pyramid_enabled: args.pyramid,
        hint_propagation_enabled: args.hint_propagate,
        palette_path: settings.palette_path.clone(),
        auto_palette: settings.auto_palette,
        cache_dir: args.cache_dir.clone(),
        cancel_token: Some(Arc::clone(&cancel_token)),
        progress_sender: Some(progress_sender),
        ..ColorizeRequest::default()
    });

    let _ = progress_thread.join();

    match result {
        Ok(colorize_result) => {
            if cancel_token.load(Ordering::Acquire) {
                let _ = edit_message(client, token, chat_id, status_msg_id, "Cancelled.");
                cleanup_dir(&job_dir);
                return Ok(());
            }
            let verdict = colorize_result.report.director_verdict.as_deref();
            if !delivery_quality_approved(verdict) {
                tracing::warn!(
                    verdict = verdict.unwrap_or("missing"),
                    "delivery blocked by quality gate"
                );
                let _ = edit_message_with_markup(
                    client,
                    token,
                    chat_id,
                    status_msg_id,
                    "This result did not pass MangaHue's quality checks, so it was not delivered or counted against your quota. Please try another scan while we review the detector.",
                    Some(r#"{"inline_keyboard":[]}"#),
                );
                cleanup_dir(&job_dir);
                return Ok(());
            }
            let caption = format!(
                "MangaHue | {}",
                colorize_result
                    .report
                    .director_verdict
                    .unwrap_or_else(|| "colored".to_string())
            );
            let delivery_path =
                prepare_delivery(&output_path, args.delivery_format, args.jpeg_quality)?;
            let kept = (args.retention_hours > 0)
                .then(|| retain_delivered_copy(&delivery_path, &args.work_dir, 1))
                .flatten();
            if let Some(kept_path) = &kept {
                tracing::info!(path = %kept_path.display(), "delivered copy kept");
            }
            // Cosmetic typing indicator: never abort the photo job on this.
            let _ = send_chat_action(client, token, chat_id, "upload_document");
            if let Err(e) = send_document(client, token, chat_id, &delivery_path, &caption) {
                tracing::error!(error = %e, "send failed");
                // Never report success without a delivered image, and never
                // charge quota for a page the user did not receive.
                let _ = edit_message_with_markup(
                    client,
                    token,
                    chat_id,
                    status_msg_id,
                    "The image was colorized but Telegram blocked the delivery (temporary API issue). It was not counted against your quota — please send it again.",
                    Some(r#"{"inline_keyboard":[]}"#),
                );
            } else {
                state.record_pages(chat_id, 1);
                let _ = edit_message_with_markup(
                    client,
                    token,
                    chat_id,
                    status_msg_id,
                    "Done! Send another photo or a manga chapter link to keep going.",
                    Some(r#"{"inline_keyboard":[]}"#),
                );
            }
            if delivery_path != output_path {
                let _ = std::fs::remove_file(&delivery_path);
            }
            let _ = std::fs::remove_file(&output_path);
            let _ = std::fs::remove_file(&local_path);
        }
        Err(error) => {
            if cancel_token.load(Ordering::Acquire) {
                let _ = edit_message_with_markup(
                    client,
                    token,
                    chat_id,
                    status_msg_id,
                    "Cancelled.",
                    Some(r#"{"inline_keyboard":[]}"#),
                );
            } else {
                let _ = edit_message_with_markup(
                    client,
                    token,
                    chat_id,
                    status_msg_id,
                    &format!("Could not colorize this image: {error}"),
                    Some(r#"{"inline_keyboard":[]}"#),
                );
            }
        }
    }
    cleanup_dir(&job_dir);
    Ok(())
}

fn cmd_help(client: &Client, token: &str, state: &Arc<BotState>, chat_id: i64) -> Result<()> {
    let access = if state.daily_limit == 0 {
        "Free access: no daily page quota\\. Server queue and chapter safety limits still apply\\."
            .to_string()
    } else {
        format!(
            "Daily quota: up to {} pages per user\\. Batch size is configurable via /batch\\.",
            state.daily_limit
        )
    };
    let text = format!(
        "*MangaHue Commands:*\n\n\
                /start \\- Start the bot and see instructions\n\
                /help \\- Show this help message\n\
                /status \\- Check if a job is running for this chat\n\
                /cancel \\- Cancel the currently running job\n\
                /style \\- List available color styles and choose one\n\
                /style _name_ \\- Set a specific style by ID\n\
                /batch \\- Show current batch size\n\
                /batch _N_ \\- Set batch size \\(1\\-10, default 10\\)\n\
                /stats \\- Show your daily page usage\n\
                /clean \\- Delete all your job data from the server\n\
                /resetref \\- Clear your reference style photo\n\
                /memory \\- Show learned correction memory\n\
                /color _x y radius hex_ \\- Add a correction for future pages\n\
                /rate _1\\-5 note_ \\- Record a blind beta quality rating\n\n\
                *How it works:*\n\
                Send a photo, a manga chapter URL, or a direct image link\\. The bot colorizes it in soft cel\\-shaded anime style and sends the result back\\.\n\n\
                *Optional reference style:*\n\
                Send a \\(colored\\) photo with caption \\`ref\\` to grade future pages toward that look\\. Use /resetref to clear it\\. Without it, pages get the default soft cel style\\.\n\n\
                *Supported URL types:*\n\
                \\- Manga reader chapter URLs \\(any site with page images\\)\n\
                \\- Direct image URLs \\(\\.png, \\.jpg, \\.webp, etc\\.\\)\n\n\
                *Access:*\n\
                {access}"
    );
    send_message(client, token, chat_id, &text, Some("MarkdownV2"))?;
    Ok(())
}

fn cmd_status(client: &Client, token: &str, state: &Arc<BotState>, chat_id: i64) -> Result<()> {
    let status = if state.is_active(chat_id) {
        "Your MangaHue chapter job is running. Use /cancel to stop it."
    } else {
        "No MangaHue job is active for this chat. Send a URL to start."
    };
    let queue_len = state.queue_len();
    let status = if queue_len > 0 {
        format!("{status}\n{queue_len} job(s) waiting in the queue.")
    } else {
        status.to_string()
    };
    send_message(client, token, chat_id, &status, None)?;
    Ok(())
}

/// /queue — report this chat's position in the FIFO job queue.
fn cmd_queue(client: &Client, token: &str, state: &Arc<BotState>, chat_id: i64) -> Result<()> {
    let message = match state.queue_position(chat_id) {
        // queue_position is 0-based; users count from 1.
        Some(position) => format!(
            "You are #{} in the queue ({} job(s) waiting total).",
            position + 1,
            state.queue_len()
        ),
        None => {
            if state.is_active(chat_id) {
                "Your job is running — no queue position.".to_string()
            } else {
                "You have no queued jobs. Send a URL to start one.".to_string()
            }
        }
    };
    send_message(client, token, chat_id, &message, None)?;
    Ok(())
}

fn cmd_cancel(client: &Client, token: &str, state: &Arc<BotState>, chat_id: i64) -> Result<()> {
    let cancelled_active = state.cancel(chat_id);
    let removed_queued = state.dequeue(chat_id);
    let status = match (cancelled_active, removed_queued) {
        (true, 0) => {
            "Cancelling the running job — it stops at the next safe checkpoint. \
             The queue keeps moving: your other queued jobs still run."
        }
        (false, 0) => "No active or queued MangaHue job to cancel.",
        (true, n) => &format!(
            "Cancelling the running job — it stops at the next safe checkpoint. \
                 Also removed your {n} queued job(s) for this chat. \
                 The bot keeps processing the queue."
        ),
        (false, n) => &format!(
            "Removed your {n} queued job(s) — nothing was running. \
                 Send another link or photo any time."
        ),
    };
    send_message(client, token, chat_id, status, None)?;
    Ok(())
}

/// /palette — choose the artist color palette for this chat (per-series
/// color memory). "auto" picks the palette from each page's fingerprint.
fn cmd_palette(
    client: &Client,
    token: &str,
    state: &Arc<BotState>,
    chat_id: i64,
    arg: &str,
) -> Result<()> {
    let arg = arg.trim();
    let dir = &state.palettes_dir;
    if !arg.is_empty() {
        match arg.to_ascii_lowercase().as_str() {
            "default" | "off" | "reset" => {
                state.set_palette(chat_id, None, false);
                send_message(
                    client,
                    token,
                    chat_id,
                    "Palette reset to the style default.",
                    None,
                )?;
                return Ok(());
            }
            "auto" => {
                state.set_palette(chat_id, None, true);
                send_message(
                    client,
                    token,
                    chat_id,
                    "Auto palette enabled: each page's fingerprint picks its artist palette (seinen/shonen/shojo/fantasy/webtoon).",
                    None,
                )?;
                return Ok(());
            }
            _ => {
                // Only plain artist-palette ids are accepted: anything with
                // separators could walk out of the palette directory.
                if !arg
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
                {
                    send_message(
                        client,
                        token,
                        chat_id,
                        "Palette names may only contain letters, digits, '-' and '_'. Use /palette to see available palettes.",
                        None,
                    )?;
                    return Ok(());
                }
                let candidate = dir.join(format!("{arg}.yaml"));
                if candidate.is_file() {
                    state.set_palette(chat_id, Some(candidate), false);
                    send_message(
                        client,
                        token,
                        chat_id,
                        &format!(
                            "Artist palette set to *{}* for this series.",
                            md2_escape(arg)
                        ),
                        Some("MarkdownV2"),
                    )?;
                    return Ok(());
                }
                send_message(
                    client,
                    token,
                    chat_id,
                    &format!("Palette '{arg}' not found. Use /palette to see available palettes."),
                    None,
                )?;
                return Ok(());
            }
        }
    }

    // List the artist palettes from the palette directory.
    let mut names: Vec<String> = Vec::new();
    if dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "yaml").unwrap_or(false) {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        names.push(stem.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    if names.is_empty() {
        send_message(
            client,
            token,
            chat_id,
            "No artist palettes configured.",
            None,
        )?;
        return Ok(());
    }
    let text = format!(
        "Available artist palettes (each series has its own color vision):\n{}\n\nUse /palette <name> to set it for this series, /palette auto for automatic selection, /palette default to reset.",
        names.iter().map(|n| format!("- {n}")).collect::<Vec<_>>().join("\n")
    );
    send_message(client, token, chat_id, &text, None)?;
    Ok(())
}

fn cmd_style(
    client: &Client,
    token: &str,
    state: &Arc<BotState>,
    chat_id: i64,
    arg: &str,
) -> Result<()> {
    let styles = state.available_styles();

    if styles.is_empty() {
        send_message(
            client,
            token,
            chat_id,
            "No custom styles configured. The bot is using the default Anime Cel style.",
            None,
        )?;
        return Ok(());
    }

    if !arg.is_empty() {
        if let Some((_id, name)) = styles.iter().find(|(id, _)| id == arg) {
            let path = resolve_style_file(state, arg);
            state.set_style(chat_id, path);
            send_message(
                client,
                token,
                chat_id,
                &format!("Style set to *{}*.", md2_escape(name)),
                Some("MarkdownV2"),
            )?;
            return Ok(());
        }
        send_message(
            client,
            token,
            chat_id,
            &format!("Style '{arg}' not found. Use /style to see available styles."),
            None,
        )?;
        return Ok(());
    }

    let mut rows: Vec<Vec<InlineKeyboardButton>> = Vec::new();
    for chunk in styles.chunks(2) {
        let row: Vec<InlineKeyboardButton> = chunk
            .iter()
            .map(|(id, name)| {
                InlineKeyboardButton::new(name).callback_data(format!("style:{}", id))
            })
            .collect();
        rows.push(row);
    }

    let settings = state.get_settings(chat_id, DEFAULT_BATCH_SIZE);
    let current = settings
        .style_path
        .as_ref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("anime_cel");

    send_message(
        client,
        token,
        chat_id,
        &format!("Choose a color style (current: *{}*):", current),
        Some("MarkdownV2"),
    )
    .map(|_| ())?;

    let reply_markup = InlineKeyboardMarkup {
        inline_keyboard: rows,
    };
    send_message(
        client,
        token,
        chat_id,
        "Select a style:",
        Some(&serde_json::to_string(&reply_markup).unwrap_or_default()),
    )?;

    Ok(())
}

fn resolve_style_file(state: &Arc<BotState>, style_id: &str) -> Option<PathBuf> {
    if let Some(ref dir) = state.styles_dir {
        for extension in ["yaml", "yml", "json"] {
            let candidate = dir.join(format!("{style_id}.{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn cmd_batch(
    client: &Client,
    token: &str,
    state: &Arc<BotState>,
    chat_id: i64,
    arg: &str,
    default_batch: usize,
) -> Result<()> {
    let settings = state.get_settings(chat_id, default_batch);

    if arg.is_empty() {
        send_message(
            client,
            token,
            chat_id,
            &format!(
                "Current batch size: *{}* pages. Use `/batch N` to change (1-10).",
                settings.batch_size
            ),
            Some("MarkdownV2"),
        )?;
        return Ok(());
    }

    match arg.parse::<usize>() {
        Ok(n) if (1..=10).contains(&n) => {
            state.set_batch_size(chat_id, n);
            send_message(
                client,
                token,
                chat_id,
                &format!("Batch size set to *{n}* pages.",),
                Some("MarkdownV2"),
            )?;
        }
        _ => {
            send_message(
                client,
                token,
                chat_id,
                "Batch size must be a number between 1 and 10.",
                None,
            )?;
        }
    }
    Ok(())
}

fn cmd_stats(client: &Client, token: &str, state: &Arc<BotState>, chat_id: i64) -> Result<()> {
    let used = state.pages_used_today(chat_id);
    let limit = state.daily_limit;
    if limit == 0 {
        send_message(
            client,
            token,
            chat_id,
            &format!(
                "Today's usage:\n\n*{used}* pages processed\nFree access is active with no daily page quota\\."
            ),
            Some("MarkdownV2"),
        )?;
        return Ok(());
    }

    let remaining = state.remaining_pages_today(chat_id);
    let bar = progress_bar(used, limit, 20);

    send_message(
        client,
        token,
        chat_id,
        &format!(
            "Today's usage:\n\n`{bar}` *{used}* / {limit} pages\n*{remaining}* remaining today",
        ),
        Some("MarkdownV2"),
    )?;
    Ok(())
}

fn cmd_clean(
    client: &Client,
    token: &str,
    state: &Arc<BotState>,
    args: &Args,
    chat_id: i64,
) -> Result<()> {
    if state.is_active(chat_id) {
        send_message(
            client,
            token,
            chat_id,
            "A job is currently running. Cancel it first with /cancel, then try /clean again.",
            None,
        )?;
        return Ok(());
    }

    let chat_dir = args.work_dir.join(chat_id.to_string());
    let ref_file = args
        .work_dir
        .join("references")
        .join(format!("chat_{chat_id}.jpg"));
    let mem_file = args
        .work_dir
        .join("memory")
        .join(format!("chat_{chat_id}.hints.json"));
    if chat_dir.exists() {
        match std::fs::remove_dir_all(&chat_dir) {
            Ok(()) => {}
            Err(e) => {
                send_message(
                    client,
                    token,
                    chat_id,
                    &format!("Failed to clean data: {e}"),
                    None,
                )?;
            }
        }
    }
    if ref_file.exists() {
        let _ = std::fs::remove_file(&ref_file);
    }
    if mem_file.exists() {
        let _ = std::fs::remove_file(&mem_file);
    }
    {
        let _ = lock_or_recover(&state.per_chat_settings).remove(&chat_id);
    }
    state.clear_usage(chat_id);
    send_message(
        client,
        token,
        chat_id,
        "All your job data, reference photo, and color memory have been cleaned up.",
        None,
    )?;
    Ok(())
}

fn handle_callback(
    client: &Client,
    token: &str,
    _args: &Args,
    state: &Arc<BotState>,
    cb: CallbackQuery,
) -> Result<()> {
    // Prefer the chat the button was attached to; in group chats cb.from.id
    // is the pressing user, not the chat.
    let chat_id = cb.message.as_ref().map(|m| m.chat.id).unwrap_or(cb.from.id);

    let Some(data) = cb.data.as_deref() else {
        answer_callback_query(client, token, &cb.id, None, None)?;
        return Ok(());
    };

    if data == "cancel_job" {
        let cancelled = state.cancel(chat_id);
        answer_callback_query(
            client,
            token,
            &cb.id,
            Some(if cancelled {
                "Cancellation requested"
            } else {
                "No active job"
            }),
            Some(false),
        )?;
        if let Some(ref msg) = cb.message {
            let _ = edit_message(
                client,
                token,
                chat_id,
                msg.message_id,
                if cancelled {
                    "Stopping safely and cleaning temporary files..."
                } else {
                    "No active MangaHue job."
                },
            );
        }
        return Ok(());
    }

    answer_callback_query(client, token, &cb.id, None, None)?;

    if data == "colorize_another" {
        send_message(
            client,
            token,
            chat_id,
            "Send another manga chapter or image URL.",
            None,
        )?;
        return Ok(());
    }

    if data == "retry_failed" {
        send_message(
            client,
            token,
            chat_id,
            "Retry not yet implemented. Send the URL again to re-process.",
            None,
        )?;
        return Ok(());
    }

    if let Some(style_id) = data.strip_prefix("style:") {
        let styles = state.available_styles();
        if let Some((_id, name)) = styles.iter().find(|(id, _)| id == style_id) {
            let path = resolve_style_file(state, style_id);
            state.set_style(chat_id, path);
            send_message(
                client,
                token,
                chat_id,
                &format!("Style set to *{}*.", md2_escape(name)),
                Some("MarkdownV2"),
            )?;
        }
        return Ok(());
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct InlineKeyboardMarkup {
    inline_keyboard: Vec<Vec<InlineKeyboardButton>>,
}

#[derive(Debug, Serialize)]
struct InlineKeyboardButton {
    text: String,
    callback_data: String,
}

impl InlineKeyboardButton {
    fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            callback_data: String::new(),
        }
    }

    fn callback_data(mut self, data: impl Into<String>) -> Self {
        self.callback_data = data.into();
        self
    }
}

struct UrlValidation {
    error: Option<String>,
    warning: Option<String>,
}

fn validate_user_url(url: &str, photos_enabled: bool) -> UrlValidation {
    let parsed = match url::Url::parse(url) {
        Ok(u) => u,
        Err(e) => {
            return UrlValidation {
                error: Some(format!("Invalid URL: {e}")),
                warning: None,
            }
        }
    };

    if !matches!(parsed.scheme(), "http" | "https") {
        return UrlValidation {
            error: Some("Only HTTP and HTTPS URLs are supported.".to_string()),
            warning: None,
        };
    }

    // Robust SSRF guard: DNS-resolve and reject private/reserved/loopback/
    // metadata hosts (127.0.0.2, [::1], 169.254.169.254, 172.16/12, ...).
    if let Err(e) = validate_public_url(&parsed) {
        return UrlValidation {
            error: Some(format!("Unsafe or invalid URL: {e}")),
            warning: None,
        };
    }

    if is_direct_image_url(&parsed) {
        if !photos_enabled {
            return UrlValidation {
                error: Some(
                    "Single-image colorization is coming soon. Send a comic chapter link and I will colorize the whole chapter!".to_string(),
                ),
                warning: None,
            };
        }
        return UrlValidation {
            error: None,
            warning: Some("Processing single image...".to_string()),
        };
    }

    UrlValidation {
        error: None,
        warning: None,
    }
}

fn is_direct_image_url(url: &url::Url) -> bool {
    let path = url.path().to_ascii_lowercase();
    DIRECT_IMAGE_EXTENSIONS
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

/// Madara sites (allporncomic.com, ilikecomix.com) offer a "paged" reader
/// via `?style=paged` that embeds only one page of the chapter. Drop that
/// parameter (and any other Madara reader params) so the continuous reader
/// — which embeds every page — is fetched instead.
fn normalize_madara_url(url: &url::Url) -> String {
    let is_madara = url.host_str().is_some_and(|host| {
        let host = host.to_ascii_lowercase();
        host.contains("allporncomic.com") || host.contains("ilikecomix.com")
    });
    if !is_madara || url.query().is_none() {
        return url.to_string();
    }
    let mut cleaned = url.clone();
    let mut pairs: Vec<(String, String)> = cleaned
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .filter(|(k, _)| k != "style")
        .collect();
    cleaned.set_query(None);
    if !pairs.is_empty() {
        let query = pairs
            .drain(..)
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        cleaned.set_query(Some(&query));
    }
    cleaned.to_string()
}

#[allow(clippy::too_many_arguments)]
fn handle_link(
    client: &Client,
    fetch_client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    chat_id: i64,
    url: &str,
    cancel_token: Arc<AtomicBool>,
) -> Result<()> {
    let job_id = Uuid::new_v4().to_string();
    let scratch_root = args.scratch_dir.as_ref().unwrap_or(&args.work_dir);
    let job_dir = scratch_root.join(chat_id.to_string()).join(&job_id);
    let _workspace = JobWorkspace::new(
        job_dir.clone(),
        args.work_dir.clone(),
        args.retention_hours > 0,
    );
    let downloads_dir = job_dir.join("downloads");
    let output_dir = job_dir.join("colored");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let source_url = url::Url::parse(url).with_context(|| format!("invalid URL: {url}"))?;
    // Madara sites offer a "paged" reader (?style=paged) that embeds only
    // ONE page of the chapter: normalize to the continuous reader, which
    // embeds every page, before fetching anything.
    let fetch_url = normalize_madara_url(&source_url);
    // Chapter title for the page watermark (best-effort; failures just mean
    // the watermark shows page numbers only).
    let job_title = fetch_page_title(fetch_client, &fetch_url).unwrap_or_default();
    if !job_title.is_empty() {
        tracing::info!(title = %job_title, "chapter title detected for watermark");
    }

    let settings = state.get_settings(chat_id, args.batch_size);
    let cancel_kb = InlineKeyboardMarkup {
        inline_keyboard: vec![vec![
            InlineKeyboardButton::new("Cancel job").callback_data("cancel_job")
        ]],
    };
    let cancel_markup = serde_json::to_string(&cancel_kb).unwrap_or_default();

    // Reference: only the chat's own saved reference photo (caption "ref").
    // No global default and no auto-cover fetch — pages colorize with the
    // soft cel style by default.
    let reference_image = settings
        .reference_image
        .clone()
        .filter(|p| !p.as_os_str().is_empty());

    send_chat_action(client, token, chat_id, "typing")?;
    let scan_msg_id = send_message(
        client,
        token,
        chat_id,
        "Scanning the complete chapter...",
        Some(&cancel_markup),
    )?;

    // The site may be Cloudflare-fronted (allporncomic.com challenges the
    // rustls TLS fingerprint): the challenge-safe fetch falls back to the
    // system curl, which passes. Uses the paged-reader-normalized URL.
    let page_html: Option<String> =
        mangahue_ingest::fetch_html_cloudflare_safe(fetch_client, &fetch_url, 16 * 1024 * 1024);

    let all_image_urls: Vec<String> = match &page_html {
        None => vec![fetch_url.clone()],
        Some(html) => {
            extract_manga_urls(fetch_client, &source_url, html, &source_url, args.max_pages)?
        }
    };

    // Content-aware mood: the source's genre tags pick the palette for this
    // chapter (dark/horror -> dark, romance -> pastel, fantasy -> vibrant).
    // Falls back to creative rotation when no tags are available.
    let tags = page_html
        .as_deref()
        .map(|html| extract_manga_tags(fetch_client, &source_url, html))
        .unwrap_or_default();
    let mood_style =
        mood::select_style_for_tags(&tags).and_then(|id| resolve_style_file(state, id));
    let mood_name = mood_style
        .as_deref()
        .and_then(load_style_name)
        .unwrap_or_default();
    let style_path =
        state.resolve_style_path(chat_id, &args.style, mood_style, args.creative_rotation);

    // Job diagnostics: URL, extracted page count and chosen style are logged
    // so "the output was rubbish" cases can be traced to extraction or
    // grading without asking the user for logs.
    tracing::info!(
        url = %url,
        pages = all_image_urls.len(),
        style = %style_path.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "default".into()),
        mood = %mood_name,
        tags = tags.len(),
        "chapter scan complete"
    );

    // SSRF guard: drop any extracted image URL that resolves to a private
    // or reserved host (attacker-controlled pages can point at internal
    // services or cloud metadata). Fail the image, not the whole job.
    let all_image_urls: Vec<String> = all_image_urls
        .into_iter()
        .filter(|u| {
            let parsed = match url::Url::parse(u) {
                Ok(p) => p,
                Err(_) => return false,
            };
            match validate_public_url(&parsed) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(url = %u, error = %e, "skipping unsafe image URL");
                    false
                }
            }
        })
        .collect();

    if all_image_urls.is_empty() {
        cleanup_dir(&job_dir);
        anyhow::bail!("no page images were found at that URL");
    }

    let total = all_image_urls.len();
    if cancel_token.load(Ordering::Acquire) {
        let _ = edit_message(client, token, chat_id, scan_msg_id, "Cancelled.");
        cleanup_dir(&job_dir);
        return Ok(());
    }
    let remaining = state.remaining_pages_today(chat_id);
    let effective_total = total.min(remaining);
    if effective_total == 0 {
        cleanup_dir(&job_dir);
        anyhow::bail!(
            "daily limit of {} pages reached; you already used {} today",
            state.daily_limit,
            state.pages_used_today(chat_id)
        );
    }

    let batch_size = settings.batch_size;
    let total_batches = effective_total.div_ceil(batch_size);

    let status_text = if remaining < total {
        // Never hide quota truncation: the user must know pages are skipped.
        format!(
            "Found all *{}* page\\(s\\)\\. Daily quota allows *{}* today\\. *{}* page\\(s\\) will be skipped\\.{}",
            total,
            effective_total,
            total - effective_total,
            mood_suffix(&mood_name)
        )
    } else {
        format!(
            "Found all *{}* page\\(s\\)\\. Coloring them one by one in chapter order\\.{}",
            effective_total,
            mood_suffix(&mood_name)
        )
    };
    let status_msg_id = send_message(client, token, chat_id, &status_text, Some("MarkdownV2"))?;
    let _ = edit_message(
        client,
        token,
        chat_id,
        scan_msg_id,
        "Chapter scan complete. One-page processing started.",
    );

    let progress_msg_id = send_message(
        client,
        token,
        chat_id,
        &format_progress(0, effective_total, 0, total_batches, None, 0),
        Some(&cancel_markup),
    )?;

    let engine = MangaHueEngine::new();
    // Shared mutable counters: the per-page closure and the retry pass both
    // update them (Rc<RefCell> so the closure's capture never fights the
    // outer reads for summary lines).
    #[derive(Default)]
    struct PageCounters {
        sent: usize,
        failed: usize,
        failed_pages: Vec<PathBuf>,
        page_times: Vec<f32>,
    }
    let counters = std::rc::Rc::new(std::cell::RefCell::new(PageCounters::default()));
    // Per-chapter palette feedback: the first successful page seeds the
    // accent palette the rest of the chapter grades toward (model-derived
    // variety; paper/skin/hair slots in the style stay untouched).
    let chapter_palette: std::rc::Rc<std::cell::RefCell<Option<Vec<mangahue_core::ColorRgb>>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    // Downloads that failed transiently (CDN hiccups) are retried once
    // after the main pass instead of being silently dropped.
    let mut retry_queue: Vec<(usize, String, PathBuf)> = Vec::new();
    let total_start = Instant::now();

    // Shared per-page pipeline (the caller owns the download): colorize,
    // deliver, report. Returns Ok(true) when the job was cancelled.
    let process_page = |position: usize,
                        local_path: &PathBuf,
                        batch_idx: usize|
     -> anyhow::Result<bool> {
        if cancel_token.load(Ordering::Acquire) {
            return Ok(true);
        }

        let page_start = Instant::now();
        let stem = local_path
            .file_stem()
            .and_then(|v| v.to_str())
            .unwrap_or("page");
        let output_path = output_dir.join(format!("{stem}.png"));

        let (progress_sender, mut progress_receiver) =
            tokio::sync::mpsc::unbounded_channel::<ProgressUpdate>();
        let progress_client = client.clone();
        let progress_token = token.to_string();
        let progress_msg_id2 = progress_msg_id;
        let progress_total = effective_total;
        let progress_position = position;
        let progress_chat = chat_id;
        let progress_markup = cancel_markup.clone();
        let progress_thread = std::thread::spawn(move || {
            let mut last_stage = String::new();
            let mut last_percent = -10i32;
            while let Some(update) = progress_receiver.blocking_recv() {
                let percent = update.percent.round() as i32;
                if update.stage != last_stage || percent >= last_percent + 10 || percent >= 100 {
                    let text = format!(
                        "Coloring page {}/{}: {} ({}%)",
                        progress_position + 1,
                        progress_total,
                        humanize_stage(&update.stage),
                        percent.clamp(0, 100)
                    );
                    let _ = edit_message_with_markup(
                        &progress_client,
                        &progress_token,
                        progress_chat,
                        progress_msg_id2,
                        &text,
                        Some(&progress_markup),
                    );
                    last_stage = update.stage;
                    last_percent = percent;
                }
            }
        });

        let result = engine.colorize(ColorizeRequest {
            input_path: local_path.clone(),
            output_path: output_path.clone(),
            model_path: args.model.clone(),
            reference_image_path: reference_image.clone(),
            enable_ensemble: !args.ensemble_models.is_empty(),
            ensemble_model_paths: args.ensemble_models.clone(),
            config_path: args.config.clone(),
            preset_path: args.preset.clone(),
            style_path: style_path.clone(),
            project_path: args.project.clone(),
            region_locks_path: resolve_region_locks(args.region_locks.as_deref(), stem),
            user_hints_path: resolve_user_hints(args, chat_id, stem),
            use_mock_backend: args.mock,
            pyramid_enabled: args.pyramid,
            hint_propagation_enabled: args.hint_propagate,
            palette_path: settings.palette_path.clone(),
            auto_palette: settings.auto_palette,
            palette_override: chapter_palette.borrow().clone(),
            cache_dir: args.cache_dir.clone(),
            cancel_token: Some(Arc::clone(&cancel_token)),
            progress_sender: Some(progress_sender),
            ..ColorizeRequest::default()
        });

        let _ = progress_thread.join();
        let elapsed = page_start.elapsed().as_secs_f32();
        let mut c = counters.borrow_mut();
        c.page_times.push(elapsed);

        match result {
            Ok(colorize_result) => {
                let quality_verdict = colorize_result.report.director_verdict.as_deref();
                if !delivery_quality_approved(quality_verdict) {
                    tracing::warn!(
                        page = position + 1,
                        verdict = quality_verdict.unwrap_or("missing"),
                        "chapter page blocked by quality gate"
                    );
                    c.failed += 1;
                    let _ = std::fs::remove_file(&output_path);
                    let _ = std::fs::remove_file(local_path);
                    return Ok(args.fail_fast);
                }
                // Chapter palette feedback: seed the accents from the first
                // colored page so later pages of this chapter share its
                // color universe (deterministic per job, model-derived).
                if args.palette_feedback && chapter_palette.borrow().is_none() {
                    if let Ok(image) = image::open(&output_path) {
                        let accents = palette_feedback::extract_accent_palette(&image.to_rgb8());
                        if !accents.is_empty() {
                            tracing::info!(
                                accents = accents.len(),
                                "chapter palette feedback: seeded accents from first page"
                            );
                            *chapter_palette.borrow_mut() = Some(accents);
                        }
                    }
                }
                if cancel_token.load(Ordering::Acquire) {
                    let _ = edit_message_with_markup(
                        client,
                        token,
                        chat_id,
                        progress_msg_id,
                        &format!("Cancelled after {} page(s).", c.sent),
                        Some(r#"{"inline_keyboard":[]}"#),
                    );
                    cleanup_dir(&job_dir);
                    return Ok(true);
                }
                let verdict = colorize_result
                    .report
                    .director_verdict
                    .clone()
                    .unwrap_or_else(|| "colored".to_string());
                // Compact page color metrics for the caption (saturation +
                // colorfulness from the color_metrics stage, e.g. "| s=19% c=35").
                let mut metrics = String::new();
                for stage in &colorize_result.report.stage_reports {
                    if stage.name == "color_metrics" {
                        let sat = stage.metrics.get("avg_saturation").copied().unwrap_or(0.0);
                        let cf = stage.metrics.get("colorfulness").copied().unwrap_or(0.0);
                        let w = stage.metrics.get("warmth").copied().unwrap_or(0.0);
                        if sat > 0.0 || cf > 0.0 {
                            metrics = format!(" | s={:.0}% c={:.0} w={:.0}", sat * 100.0, cf, w);
                        }
                        break;
                    }
                }
                // Caption: the manga name and the page number (the same
                // title is drawn into the image watermark). No detectable
                // title falls back to the product name.
                let caption = if job_title.is_empty() {
                    format!("MangaHue · page {}/{}", position + 1, effective_total)
                } else {
                    format!("{} · page {}/{}", job_title, position + 1, effective_total)
                };
                // Color metrics stay available for diagnostics (they used to
                // ride in the caption; the caption is user-facing now).
                tracing::info!(
                    page = position + 1,
                    total = effective_total,
                    sat = metrics,
                    verdict = %verdict,
                    "chapter page delivered"
                );
                // Page watermark: manga title + page number on the image.
                // Drawn on the engine's PNG before the single delivery
                // encode, so each page is JPEG-encoded exactly once.
                if let Ok(page_image) = image::open(&output_path) {
                    let mut rgb = page_image.to_rgb8();
                    watermark::draw_page_watermark(
                        &mut rgb,
                        &job_title,
                        position + 1,
                        effective_total,
                    );
                    // PNG round-trip is lossless; prepare_delivery below
                    // performs the single PNG->JPEG encode with the
                    // configured quality.
                    rgb.save(&output_path)?;
                }
                let delivery_path =
                    prepare_delivery(&output_path, args.delivery_format, args.jpeg_quality)?;
                // Keep a copy of the EXACT file the user receives under
                // <work_dir>/delivered/YYYY-MM-DD/ so quality complaints can
                // be traced to the actual delivered bytes (the scratch space
                // is wiped after the job).
                let kept = (args.retention_hours > 0)
                    .then(|| {
                        retain_delivered_copy(&delivery_path, &args.work_dir, position as u32 + 1)
                    })
                    .flatten();
                if let Some(kept_path) = &kept {
                    tracing::info!(page = position + 1, path = %kept_path.display(), "delivered copy kept");
                }
                ensure_scratch_budget(scratch_root, args.scratch_limit_bytes)?;
                // Cosmetic typing indicator: never abort the chapter on this.
                let _ = send_chat_action(client, token, chat_id, "upload_document");
                if let Err(e) = send_document(client, token, chat_id, &delivery_path, &caption) {
                    tracing::error!(error = %e, "send failed");
                    c.failed += 1;
                } else {
                    c.sent += 1;
                    // Quota is charged only for pages actually delivered:
                    // retrying failed pages must not double-charge.
                    state.record_pages(chat_id, 1);
                }
                if delivery_path != output_path {
                    let _ = std::fs::remove_file(&delivery_path);
                }
                let _ = std::fs::remove_file(&output_path);
                let _ = std::fs::remove_file(local_path);
            }
            Err(error) => {
                if cancel_token.load(Ordering::Acquire) {
                    let _ = edit_message_with_markup(
                        client,
                        token,
                        chat_id,
                        progress_msg_id,
                        "Cancelled.",
                        Some(r#"{"inline_keyboard":[]}"#),
                    );
                    cleanup_dir(&job_dir);
                    return Ok(true);
                }
                // Evidence for undecodable pages: file existence, size and
                // magic bytes of the exact file that failed, plus the full
                // error chain (Debug shows every context level).
                let file_evidence = std::fs::read(local_path).ok().map(|bytes| {
                    let magic = if bytes.len() >= 12 {
                        format!("{} {} {} {}", bytes[0], bytes[1], bytes[2], bytes[3])
                    } else {
                        format!("only {} bytes", bytes.len())
                    };
                    format!("exists size={} first_bytes=[{magic}]", bytes.len())
                });
                let file_evidence = file_evidence.unwrap_or_else(|| "file missing".to_string());
                tracing::warn!(
                    page = position + 1,
                    file = %file_evidence,
                    error = format!("{error:#}"),
                    "page colorization failed, continuing"
                );
                c.failed += 1;
                c.failed_pages.push(local_path.clone());
                if args.fail_fast {
                    cleanup_dir(&job_dir);
                    return Err(error);
                }
            }
        }

        let done = c.sent + c.failed;
        let eta = estimate_eta(&c.page_times, done, effective_total);
        let _ = edit_message_with_markup(
            client,
            token,
            chat_id,
            progress_msg_id,
            &format_progress(
                done,
                effective_total,
                batch_idx,
                total_batches,
                eta,
                c.failed,
            ),
            Some(&cancel_markup),
        );
        Ok(false)
    };

    for batch_idx in 0..total_batches {
        if cancel_token.load(Ordering::Acquire) {
            let _ = edit_message_with_markup(
                client,
                token,
                chat_id,
                progress_msg_id,
                "Cancelled.",
                Some(r#"{"inline_keyboard":[]}"#),
            );
            cleanup_dir(&job_dir);
            return Ok(());
        }

        let start = batch_idx * batch_size;
        let end = ((batch_idx + 1) * batch_size).min(effective_total);
        let batch_urls: Vec<(usize, String)> = all_image_urls[start..end]
            .iter()
            .enumerate()
            .map(|(i, u)| (start + i, u.clone()))
            .collect();

        // Download this batch; transient failures are queued for the retry pass.
        let mut batch_pages: Vec<(usize, PathBuf)> = Vec::new();
        for (pos, img_url) in &batch_urls {
            if cancel_token.load(Ordering::Acquire) {
                break;
            }
            let ext = image_extension_from_url(img_url);
            let local_path = downloads_dir.join(format!("page_{:03}.{}", pos + 1, ext));
            match download_image_url_telegram(
                fetch_client,
                img_url,
                source_url.as_str(),
                &local_path,
                args.max_bytes_per_page,
            ) {
                Ok(()) => {
                    ensure_scratch_budget(scratch_root, args.scratch_limit_bytes)?;
                    batch_pages.push((*pos, local_path));
                }
                Err(e) => {
                    tracing::warn!(url = %img_url, error = %e, "download failed, queued for retry");
                    retry_queue.push((*pos, img_url.clone(), local_path));
                }
            }
        }

        for (position, page) in &batch_pages {
            if process_page(*position, page, batch_idx)? {
                cleanup_dir(&job_dir);
                return Ok(());
            }
        }

        for (_, path) in &batch_pages {
            let _ = std::fs::remove_file(path);
        }
    }

    // Retry pass: give transiently failed downloads a second chance before
    // reporting them as failed.
    if !retry_queue.is_empty() && !cancel_token.load(Ordering::Acquire) {
        let mut recovered: Vec<(usize, PathBuf)> = Vec::new();
        for (pos, img_url, local_path) in &retry_queue {
            if cancel_token.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(Duration::from_secs(3));
            match download_image_url_telegram(
                fetch_client,
                img_url,
                source_url.as_str(),
                local_path,
                args.max_bytes_per_page,
            ) {
                Ok(()) => {
                    ensure_scratch_budget(scratch_root, args.scratch_limit_bytes)?;
                    recovered.push((*pos, local_path.clone()));
                }
                Err(e) => {
                    tracing::warn!(url = %img_url, error = %e, "retry download failed, skipping");
                    counters.borrow_mut().failed += 1;
                }
            }
        }
        for (position, page) in &recovered {
            if process_page(*position, page, total_batches)? {
                cleanup_dir(&job_dir);
                return Ok(());
            }
        }
        for (_, path) in &recovered {
            let _ = std::fs::remove_file(path);
        }
    }

    cleanup_dir(&job_dir);

    let elapsed = total_start.elapsed().as_secs_f32();
    let completion_kb = InlineKeyboardMarkup {
        inline_keyboard: vec![vec![
            InlineKeyboardButton::new("Colorize Another").callback_data("colorize_another")
        ]],
    };

    let mut final_text = format!(
        "Done in {:.0}s\\. Sent *{}* colored page\\(s\\)\\.",
        elapsed,
        counters.borrow().sent
    );
    if counters.borrow().failed > 0 {
        final_text.push_str(&format!(
            " *{}* page\\(s\\) failed\\. Send the URL again to retry failed pages\\.",
            counters.borrow().failed
        ));
    }

    let _ = edit_message(client, token, chat_id, status_msg_id, &final_text);

    send_message(
        client,
        token,
        chat_id,
        "What would you like to do next?",
        Some(&serde_json::to_string(&completion_kb).unwrap_or_default()),
    )?;

    let _ = edit_message_with_markup(
        client,
        token,
        chat_id,
        progress_msg_id,
        &format!(
            "Complete: *{}* sent, *{}* failed\\.",
            counters.borrow().sent,
            counters.borrow().failed
        ),
        Some(r#"{"inline_keyboard":[]}"#),
    );

    Ok(())
}

fn format_progress(
    done: usize,
    total: usize,
    batch_idx: usize,
    total_batches: usize,
    eta: Option<f32>,
    failed: usize,
) -> String {
    let pct = if total == 0 {
        0
    } else {
        ((done as f32 / total as f32) * 100.0).round() as usize
    };
    let bar = progress_bar(done, total, 16);
    // MarkdownV2-safe: the pipe and period must be escaped, the percent is
    // an integer so no decimal point appears.
    let mut text = format!(
        "`{bar}` {pct}%\nPages: {done}/{total} \\| Batch: {}/{}",
        (batch_idx + 1).min(total_batches),
        total_batches
    );
    if failed > 0 {
        text.push_str(&format!(" \\| Failed: {failed}"));
    }
    if let Some(e) = eta {
        if e > 0.0 {
            text.push_str(&format!(" \\| ETA: \\~{:.0}s", e));
        }
    }
    text
}

fn progress_bar(done: usize, total: usize, width: usize) -> String {
    if total == 0 {
        return "░".repeat(width);
    }
    let filled = ((done as f64 / total as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);
    format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
}

fn estimate_eta(times: &[f32], done: usize, total: usize) -> Option<f32> {
    if times.is_empty() || done >= total {
        return None;
    }
    let avg: f32 = times.iter().sum::<f32>() / times.len() as f32;
    let remaining = total.saturating_sub(done) as f32;
    Some(avg * remaining)
}

/// Append every `*.quality.json` report the engine wrote into `job_dir` to
/// `<work_dir>/quality/<YYYY-MM-DD>/<job>.jsonl` before the scratch space is
/// wiped. Keeps per-page quality data (line preservation, bubble cleanliness,
/// sharpness…) so tuning decisions are grounded in real telemetry instead of
/// anecdote.
fn retain_quality_telemetry(job_dir: &Path, work_dir: &Path) {
    let mut files = Vec::new();
    let mut pending = vec![job_dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".quality.json"))
            {
                files.push(path);
            }
        }
    }
    if files.is_empty() {
        return;
    }
    let day = Utc::now().format("%Y-%m-%d");
    let dir = work_dir.join("quality").join(day.to_string());
    let Ok(_) = std::fs::create_dir_all(&dir) else {
        return;
    };
    let log_path = dir.join(format!("{}.jsonl", Uuid::new_v4()));
    let Ok(mut log) = std::fs::File::create(&log_path) else {
        return;
    };
    use std::io::Write;
    for file in files {
        if let Ok(json) = std::fs::read_to_string(&file) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json) {
                let _ = writeln!(log, "{value}");
            }
        }
    }
}

fn cleanup_dir(path: &Path) {
    if path.exists() {
        let _ = std::fs::remove_dir_all(path);
    }
}

struct JobWorkspace {
    path: PathBuf,
    work_dir: PathBuf,
    retain_telemetry: bool,
}

impl JobWorkspace {
    fn new(path: PathBuf, work_dir: PathBuf, retain_telemetry: bool) -> Self {
        Self {
            path,
            work_dir,
            retain_telemetry,
        }
    }
}

impl Drop for JobWorkspace {
    fn drop(&mut self) {
        if self.retain_telemetry {
            retain_quality_telemetry(&self.path, &self.work_dir);
        }
        cleanup_dir(&self.path);
        if let Some(chat_dir) = self.path.parent() {
            let _ = std::fs::remove_dir(chat_dir);
        }
    }
}

fn cleanup_stale_scratch(root: &Path) -> Result<()> {
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("failed to inspect scratch directory {}", root.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            std::fs::remove_dir_all(&path)
                .with_context(|| format!("failed to remove stale workspace {}", path.display()))?;
        } else {
            std::fs::remove_file(&path).with_context(|| {
                format!("failed to remove stale scratch file {}", path.display())
            })?;
        }
    }
    Ok(())
}

/// Enforce a finite lifetime for every retained user image and diagnostic.
/// Only product-owned retention directories are traversed; configuration,
/// quota ledgers, and the executable release tree are never touched.
fn purge_expired_user_data(work_dir: &Path, retention_hours: u64) -> Result<()> {
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(retention_hours.saturating_mul(3_600)))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut roots = ["delivered", "quality", "references"]
        .into_iter()
        .map(|name| work_dir.join(name))
        .collect::<Vec<_>>();
    // Older releases retained job workspaces directly under
    // `<work_dir>/<chat-id>/<job-id>`. Numeric top-level directories are
    // user-owned legacy data, so include them without traversing named
    // configuration, quota, settings, feedback, or memory directories.
    if work_dir.is_dir() {
        for entry in std::fs::read_dir(work_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.parse::<i64>().is_ok())
            {
                roots.push(entry.path());
            }
        }
    }
    for root in roots {
        if !root.exists() {
            continue;
        }
        let mut pending = vec![root.clone()];
        let mut directories = Vec::new();
        while let Some(directory) = pending.pop() {
            directories.push(directory.clone());
            for entry in std::fs::read_dir(&directory)
                .with_context(|| format!("failed to inspect {}", directory.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                let metadata = entry.metadata()?;
                if metadata.is_dir() {
                    pending.push(path);
                } else if retention_hours == 0
                    || metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH) <= cutoff
                {
                    std::fs::remove_file(&path)
                        .with_context(|| format!("failed to purge {}", path.display()))?;
                }
            }
        }
        directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
        for directory in directories {
            if directory != root {
                let _ = std::fs::remove_dir(&directory);
            }
        }
    }
    Ok(())
}

fn directory_bytes(root: &Path) -> Result<u64> {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("failed to inspect {}", dir.display()))?
        {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                pending.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

fn ensure_scratch_budget(root: &Path, limit: u64) -> Result<()> {
    let used = directory_bytes(root)?;
    if used > limit {
        anyhow::bail!(
            "RAM workspace limit exceeded ({used} bytes used, {limit} allowed); reduce batch size"
        );
    }
    Ok(())
}

/// Copy the exact input image the user sent into `<work_dir>/delivered/`
/// (same folder as the delivered outputs) for source-side diagnosis.
fn retain_input_copy(input: &Path, work_dir: &Path) -> Option<PathBuf> {
    let day = Utc::now().format("%Y-%m-%d");
    let dir = work_dir.join("delivered").join(day.to_string());
    let Ok(_) = std::fs::create_dir_all(&dir) else {
        return None;
    };
    let target = dir.join(format!("input_{}.jpg", Utc::now().format("%H%M%S%3f")));
    std::fs::copy(input, &target).ok()?;
    Some(target)
}

/// Copy the exact delivery file (PNG or JPEG) into
/// `<work_dir>/delivered/YYYY-MM-DD/` before it is sent, so quality
/// complaints can be traced to the actual bytes the user received.
/// Returns the kept path (None when the copy failed — never fatal).
fn retain_delivered_copy(delivery: &Path, work_dir: &Path, page: u32) -> Option<PathBuf> {
    let day = Utc::now().format("%Y-%m-%d");
    let dir = work_dir.join("delivered").join(day.to_string());
    let Ok(_) = std::fs::create_dir_all(&dir) else {
        return None;
    };
    let stem = delivery
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("page");
    let ext = delivery
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin");
    let target = dir.join(format!("{stem}_p{page:03}.{ext}"));
    std::fs::copy(delivery, &target).ok()?;
    Some(target)
}

fn prepare_delivery(source: &Path, format: DeliveryFormat, jpeg_quality: u8) -> Result<PathBuf> {
    match format {
        DeliveryFormat::Png => Ok(source.to_path_buf()),
        DeliveryFormat::Jpeg => {
            let output = source.with_extension("jpg");
            let profile = ExportProfile {
                id: "telegram_hq".to_string(),
                max_dimension: u32::MAX,
                format: ExportFormat::Jpeg,
                quality: jpeg_quality,
                watermark: WatermarkMode::None,
                preserve_resolution: true,
            };
            render_export(source, &output, &profile, None)?;
            Ok(output)
        }
    }
}

fn download_image_url_telegram(
    client: &Client,
    url: &str,
    referer: &str,
    output: &Path,
    max_bytes: u64,
) -> Result<()> {
    // SSRF guard at the exact fetch point: reject private/reserved hosts
    // (attacker-controlled HTML can point image URLs at internal services).
    let parsed = url::Url::parse(url).with_context(|| format!("invalid image URL: {url}"))?;
    validate_public_url(&parsed).with_context(|| format!("unsafe image URL skipped: {url}"))?;

    // CDNs throttle burst scrapers (nhentai in particular): retry each
    // candidate host a few times with backoff. nhentai serves every image
    // on all four CDN hosts (i1..i4 + i) and throttles individual hosts
    // (i1 is a common offender), so the identical path is retried on the
    // sibling hosts before giving up.
    let mut candidates: Vec<String> = vec![url.to_string()];
    candidates.extend(alternate_nhentai_hosts(url));
    let mut last_error: Option<anyhow::Error> = None;
    for candidate in &candidates {
        let mut succeeded = false;
        for attempt in 0..3usize {
            let outcome = (|| -> Result<()> {
                let resp = client
                    .get(candidate)
                    .header(USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
                    .header(REFERER, referer)
                    .header(
                        ACCEPT,
                        // No image/avif: the image crate build has no avif
                        // decoder, and nhentai serves avif when asked for it.
                        "image/webp,image/png,image/jpeg,image/*;q=0.9,*/*;q=0.5",
                    )
                    .timeout(Duration::from_secs(60))
                    .send()
                    .with_context(|| format!("failed to fetch {candidate}"))?
                    .error_for_status()
                    .with_context(|| format!("server rejected {candidate}"))?;
                let content_type = resp
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let mut reader = resp.take(max_bytes + 1);
                let mut data = Vec::new();
                reader
                    .read_to_end(&mut data)
                    .with_context(|| format!("failed to read bytes from {candidate}"))?;
                if data.len() as u64 > max_bytes {
                    anyhow::bail!("image exceeds {max_bytes} bytes");
                }
                if !content_type.starts_with("image/") && !has_image_magic(&data) {
                    anyhow::bail!("URL returned non-image content ({content_type})");
                }
                // Header-only dimension validation BEFORE decode: a small
                // file with a huge IHDR would otherwise allocate gigabytes
                // at decode time (decode bomb).
                let (img_width, img_height) = image::ImageReader::new(std::io::Cursor::new(&data))
                    .with_guessed_format()
                    .with_context(|| {
                        format!(
                            "URL returned undecodable image content ({content_type}, {} bytes)",
                            data.len()
                        )
                    })?
                    .into_dimensions()
                    .with_context(|| {
                        format!(
                            "URL image has invalid dimensions ({content_type}, {} bytes)",
                            data.len()
                        )
                    })?;
                if img_width > 50_000
                    || img_height > 50_000
                    || img_width as u64 * img_height as u64 > MAX_PHOTO_PIXELS
                {
                    anyhow::bail!("image dimensions exceed security limits");
                }
                // Decode-verify BEFORE writing: throttled CDN hosts sometimes
                // return 200 with an HTML error body or an undecodable
                // variant (e.g. avif) under an image/* content type. Writing
                // such a file makes the whole colorize job fail later with a
                // confusing "failed to load" — retrying the sibling hosts
                // here instead usually gets the real image.
                image::load_from_memory(&data).with_context(|| {
                    format!(
                        "URL returned undecodable image content ({content_type}, {} bytes)",
                        data.len()
                    )
                })?;
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(output, &data)
                    .with_context(|| format!("failed to write {}", output.display()))?;
                Ok(())
            })();
            match outcome {
                Ok(()) => {
                    succeeded = true;
                    break;
                }
                Err(error) => {
                    last_error = Some(error);
                    if attempt < 2 {
                        std::thread::sleep(Duration::from_secs(1 + attempt as u64 * 2));
                    }
                }
            }
        }
        if succeeded {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("download failed after retries: {url}")))
}

/// nhentai serves every image on all CDN hosts (`i1..i4.nhentai.net` plus
/// the canonical `i.nhentai.net`). When one host throttles us, the same
/// path on a sibling host still works; return those siblings.
fn alternate_nhentai_hosts(url: &str) -> Vec<String> {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return Vec::new();
    };
    let Some(host) = parsed.host_str().map(str::to_string) else {
        return Vec::new();
    };
    let lower = host.to_ascii_lowercase();
    let Some(domain) = [".nhentai.net", ".nhentai.to"]
        .iter()
        .find(|d| lower.ends_with(**d))
    else {
        return Vec::new();
    };
    let prefix = &lower[..lower.len() - domain.len()]; // e.g. "i1" or "i"
    let Some(num_str) = prefix.strip_prefix('i') else {
        return Vec::new();
    };
    if num_str.is_empty() || !num_str.chars().all(|c| c.is_ascii_digit()) {
        return Vec::new();
    }
    let num: u32 = num_str.parse().unwrap_or(0);
    let mut siblings = Vec::new();
    for n in 1..=4u32 {
        if n != num {
            let _ = parsed.set_host(Some(&format!("i{n}{domain}")));
            siblings.push(parsed.to_string());
        }
    }
    let _ = parsed.set_host(Some(&format!("i{domain}")));
    siblings.push(parsed.to_string());
    siblings
}

fn has_image_magic(data: &[u8]) -> bool {
    data.starts_with(b"\xFF\xD8\xFF")
        || data.starts_with(b"\x89PNG\r\n\x1A\n")
        || data.starts_with(b"GIF87a")
        || data.starts_with(b"GIF89a")
        || data.starts_with(b"BM")
        || (data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP")
        || data.starts_with(b"II*\0")
        || data.starts_with(b"MM\0*")
}

fn image_extension_from_url(value: &str) -> &'static str {
    let path = url::Url::parse(value)
        .ok()
        .map(|url| url.path().to_string())
        .unwrap_or_else(|| value.split(['?', '#']).next().unwrap_or(value).to_string());
    match Path::new(&path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "png",
        Some("webp") => "webp",
        Some("gif") => "gif",
        Some("bmp") => "bmp",
        Some("tif") | Some("tiff") => "tiff",
        _ => "jpg",
    }
}

fn resolve_region_locks(root: Option<&Path>, page_stem: &str) -> Option<PathBuf> {
    let root = root?;
    if root.is_file() || !root.is_dir() {
        return Some(root.to_path_buf());
    }
    [
        format!("{page_stem}.region-locks.json"),
        format!("{page_stem}.regions.json"),
        format!("{page_stem}.json"),
    ]
    .into_iter()
    .map(|name| root.join(name))
    .find(|candidate| candidate.is_file())
}

/// Per-chat learned color memory. If the chat has a hints file from prior
/// corrections, prefer it; otherwise fall back to the global --hints file.
fn resolve_user_hints(args: &Args, chat_id: i64, _page_stem: &str) -> Option<PathBuf> {
    let chat_memory = args
        .work_dir
        .join("memory")
        .join(format!("chat_{chat_id}.hints.json"));
    if chat_memory.is_file() {
        return Some(chat_memory);
    }
    args.hints.clone()
}

/// After a restart, re-register saved reference photos (chat_{id}.jpg) so
/// the per-chat style reference survives process restarts.
fn restore_reference_photos(state: &BotState, references_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(references_dir) else {
        return;
    };
    let mut restored = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some(rest) = name_str.strip_prefix("chat_") else {
            continue;
        };
        let Some(chat_str) = rest.strip_suffix(".jpg") else {
            continue;
        };
        if let Ok(chat_id) = chat_str.parse::<i64>() {
            state.set_reference_image(chat_id, entry.path());
            restored += 1;
        }
    }
    if restored > 0 {
        tracing::info!(restored, "restored saved reference photos after restart");
    }
}

/// After a restart, reload per-chat settings (/style, /batch, /palette,
/// reference photo) from `<work_dir>/settings/chat_<id>.json`.
fn restore_chat_settings(state: &BotState) {
    let dir = state.settings_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut restored = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some(chat_str) = name_str
            .strip_prefix("chat_")
            .and_then(|rest| rest.strip_suffix(".json"))
        else {
            continue;
        };
        let Ok(chat_id) = chat_str.parse::<i64>() else {
            continue;
        };
        let Ok(data) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(settings) = serde_json::from_str::<PerChatSettings>(&data) else {
            continue;
        };
        // Never point at a reference file that is gone (the user may have
        // used /clean or /resetref while the bot was down).
        let settings = PerChatSettings {
            reference_image: settings.reference_image.filter(|p| p.is_file()),
            ..settings
        };
        lock_or_recover(&state.per_chat_settings).insert(chat_id, settings);
        restored += 1;
    }
    if restored > 0 {
        tracing::info!(restored, "restored per-chat settings after restart");
    }
}

/// After a restart, re-queue jobs that were waiting when the bot stopped.
/// Returns how many were restored.
fn restore_persisted_queue(state: &BotState) -> usize {
    let path = state.settings_dir().join("queue.json");
    let Ok(data) = std::fs::read_to_string(&path) else {
        return 0;
    };
    let Ok(queue) = serde_json::from_str::<Vec<(i64, QueuedJob)>>(&data) else {
        return 0;
    };
    if queue.is_empty() {
        return 0;
    }
    {
        let mut current = lock_or_recover(&state.queue);
        // Restore EVERY persisted item: with multiple jobs per chat allowed,
        // deduplicating by chat would silently drop legitimate queued work.
        for (chat_id, job) in queue {
            current.push_back((chat_id, job));
        }
        let restored = current.len();
        drop(current);
        // The queue file is now authoritative again (dedup may have shrunk
        // it, and dispatch will keep it updated from here on).
        state.persist_queue();
        restored
    }
}

/// Best-effort chapter title extraction from the page HTML `<title>` tag
/// (nhentai-style galleries put the manga name first). Failures return
/// None — the watermark then shows page numbers only.
fn fetch_page_title(client: &Client, url: &str) -> Option<String> {
    // Cloudflare-fronted sites challenge rustls: use the curl fallback too.
    let html = mangahue_ingest::fetch_html_cloudflare_safe(client, url, 1024 * 1024)?;
    let start = html.find("<title")?;
    let gt = html[start..].find('>')? + start + 1;
    let end = html[gt..].find("</title")? + gt;
    let mut title = html[gt..end].trim().to_string();
    // "Manga Title - Other stuff" → keep the first segment.
    if let Some((head, _)) = title.split_once(" - ") {
        title = head.trim().to_string();
    }
    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

fn humanize_stage(stage: &str) -> &str {
    match stage {
        "queued" => "queued",
        "preprocess" => "cleaning line art",
        "layout" => "analyzing panels",
        "hints" => "building color memory",
        "inference_started" | "inference_tile" => "colorizing",
        "inference_done" => "merging color layers",
        "postprocess" => "restoring ink and text",
        "done" => "complete",
        other => other,
    }
}

fn get_updates(
    client: &Client,
    token: &str,
    offset: Option<i64>,
    timeout: u64,
) -> Result<Vec<Update>> {
    telegram_retry("getUpdates", || {
        let url = telegram_url(token, "getUpdates");
        let mut body = json!({
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"]
        });
        if let Some(offset) = offset {
            body["offset"] = json!(offset);
        }
        let envelope: TelegramEnvelope<Vec<Update>> = client
            .post(url)
            .json(&body)
            .send()
            .context("failed to call getUpdates")?
            .error_for_status()
            .context("Telegram getUpdates failed")?
            .json()
            .context("failed to parse getUpdates response")?;
        telegram_result(envelope)
    })
}

fn configure_bot(client: &Client, token: &str) -> Result<BotIdentity> {
    let identity: TelegramEnvelope<BotIdentity> = telegram_retry("getMe", || {
        // Inner annotation: tail-`?` chains without a local type anchor
        // confuse the Deserialize inference inside the retry closure.
        let envelope: TelegramEnvelope<BotIdentity> = client
            .get(telegram_url(token, "getMe"))
            .send()
            .context("failed to call getMe")?
            .error_for_status()
            .context("Telegram getMe failed")?
            .json()
            .context("failed to parse getMe response")?;
        Ok(envelope)
    })?;
    let identity = telegram_result(identity)?;
    // Registering commands is best-effort: a failure here must never block
    // startup (the bot stays usable without the command menu).
    if let Err(error) = telegram_retry("setMyCommands", || {
        let commands: TelegramEnvelope<bool> = client
            .post(telegram_url(token, "setMyCommands"))
            .json(&json!({
                "commands": [
                    {"command": "start", "description": "Show MangaHue instructions"},
                    {"command": "help", "description": "Show all commands and usage"},
                    {"command": "status", "description": "Check the active chapter job"},
                    {"command": "cancel", "description": "Cancel the active chapter job"},
                    {"command": "style", "description": "List and choose color styles"},
                    {"command": "batch", "description": "Show or set batch size (1-10)"},
                    {"command": "stats", "description": "Show daily page usage"},
                    {"command": "clean", "description": "Delete all your job data"}
                ]
            }))
            .send()
            .context("failed to call setMyCommands")?
            .error_for_status()
            .context("Telegram setMyCommands failed")?
            .json()
            .context("failed to parse setMyCommands response")?;
        let _ = telegram_result(commands)?;
        Ok(())
    }) {
        tracing::warn!(
            error = format!("{error:#}"),
            "failed to register bot commands"
        );
    }
    Ok(identity)
}

fn send_message(
    client: &Client,
    token: &str,
    chat_id: i64,
    text: &str,
    parse_mode_or_markup: Option<&str>,
) -> Result<i64> {
    telegram_retry("sendMessage", || {
        let mut body = json!({
            "chat_id": chat_id,
            "text": text
        });

        if let Some(pm) = parse_mode_or_markup {
            if pm.starts_with('{') {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(pm) {
                    body["reply_markup"] = parsed;
                }
            } else {
                body["parse_mode"] = json!(pm);
            }
        }

        let envelope: TelegramEnvelope<SentMessage> = client
            .post(telegram_url(token, "sendMessage"))
            .json(&body)
            .send()
            .context("failed to call sendMessage")?
            .error_for_status()
            .context("Telegram sendMessage failed")?
            .json()
            .context("failed to parse sendMessage response")?;
        Ok(telegram_result(envelope)?.message_id)
    })
}

fn edit_message(
    client: &Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
) -> Result<()> {
    edit_message_with_markup(client, token, chat_id, message_id, text, None)
}

/// Like [edit_message] but re-applies an inline keyboard (e.g. the cancel
/// button) on every edit so it stays visible during the whole colorization.
fn edit_message_with_markup(
    client: &Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
    markup: Option<&str>,
) -> Result<()> {
    telegram_retry("editMessageText", || {
        let mut body = json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text
        });
        if text.contains('*') || text.contains('`') {
            body["parse_mode"] = json!("MarkdownV2");
        }
        if let Some(markup) = markup {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(markup) {
                body["reply_markup"] = parsed;
            }
        }

        let envelope: TelegramEnvelope<serde_json::Value> = client
            .post(telegram_url(token, "editMessageText"))
            .json(&body)
            .send()
            .context("failed to call editMessageText")?
            .error_for_status()
            .context("Telegram editMessageText failed")?
            .json()
            .context("failed to parse editMessageText response")?;
        let _ = telegram_result(envelope)?;
        Ok(())
    })
}

fn send_chat_action(client: &Client, token: &str, chat_id: i64, action: &str) -> Result<()> {
    telegram_retry("sendChatAction", || {
        let envelope: TelegramEnvelope<serde_json::Value> = client
            .post(telegram_url(token, "sendChatAction"))
            .json(&json!({
                "chat_id": chat_id,
                "action": action
            }))
            .send()
            .context("failed to call sendChatAction")?
            .error_for_status()
            .context("Telegram sendChatAction failed")?
            .json()
            .context("failed to parse sendChatAction response")?;
        let _ = telegram_result(envelope)?;
        Ok(())
    })
}

fn send_document(
    client: &Client,
    token: &str,
    chat_id: i64,
    path: &Path,
    caption: &str,
) -> Result<()> {
    // The multipart form is rebuilt on every attempt: a failed request
    // consumes it, and the file must be re-attached for the next try.
    telegram_retry("sendDocument", || {
        let form = Form::new()
            .text("chat_id", chat_id.to_string())
            .text("caption", caption.to_string())
            .file("document", path)
            .with_context(|| format!("failed to attach {}", path.display()))?;
        let envelope: TelegramEnvelope<serde_json::Value> = client
            .post(telegram_url(token, "sendDocument"))
            .multipart(form)
            .send()
            .context("failed to call sendDocument")?
            .error_for_status()
            .context("Telegram sendDocument failed")?
            .json()
            .context("failed to parse sendDocument response")?;
        let _ = telegram_result(envelope)?;
        Ok(())
    })
}

fn answer_callback_query(
    client: &Client,
    token: &str,
    callback_query_id: &str,
    text: Option<&str>,
    show_alert: Option<bool>,
) -> Result<()> {
    telegram_retry("answerCallbackQuery", || {
        let mut body = json!({
            "callback_query_id": callback_query_id
        });
        if let Some(t) = text {
            body["text"] = json!(t);
        }
        if let Some(a) = show_alert {
            body["show_alert"] = json!(a);
        }
        let envelope: TelegramEnvelope<serde_json::Value> = client
            .post(telegram_url(token, "answerCallbackQuery"))
            .json(&body)
            .send()
            .context("failed to call answerCallbackQuery")?
            .error_for_status()
            .context("Telegram answerCallbackQuery failed")?
            .json()
            .context("failed to parse answerCallbackQuery response")?;
        let _ = telegram_result(envelope)?;
        Ok(())
    })
}

/// A Telegram API error returned inside a JSON envelope (ok:false), kept
/// typed so the retry helper can tell transient wording ("too many requests")
/// from permanent rejections ("message is not modified").
#[derive(Debug)]
struct TelegramApiError {
    description: String,
}

impl std::fmt::Display for TelegramApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Telegram API error: {}", self.description)
    }
}

impl std::error::Error for TelegramApiError {}

fn telegram_result<T>(envelope: TelegramEnvelope<T>) -> Result<T> {
    if envelope.ok {
        envelope
            .result
            .context("Telegram response was ok but had no result")
    } else {
        Err(TelegramApiError {
            description: envelope
                .description
                .unwrap_or_else(|| "unknown error".to_string()),
        }
        .into())
    }
}

/// Retry a Telegram API call on transient failures. Permanent errors (4xx
/// validation, blocked users, "message is not modified") fail immediately;
/// everything else gets up to [TELEGRAM_MAX_ATTEMPTS] tries with exponential
/// backoff. This is what keeps a chapter alive through the 502/504 gateway
/// storms this bot has already survived.
fn telegram_retry<T>(label: &str, mut attempt: impl FnMut() -> Result<T>) -> Result<T> {
    let mut delay = Duration::from_secs(TELEGRAM_RETRY_BASE_SECS);
    let mut round = 1u32;
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(error)
                if round < TELEGRAM_MAX_ATTEMPTS as u32 && is_transient_telegram_error(&error) =>
            {
                tracing::warn!(
                    error = format!("{error:#}"),
                    attempt = round,
                    retry_in_secs = delay.as_secs(),
                    "{label} transient failure, retrying"
                );
                std::thread::sleep(delay);
                delay *= 2;
                round += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// True when a Telegram call failed in a way a retry can fix: HTTP 5xx and
/// 429 (the 502/504 gateway storms this bot has seen), timeouts, connection
/// errors, and "too many requests" style envelope errors.
fn is_transient_telegram_error(error: &anyhow::Error) -> bool {
    if let Some(request_error) = error.downcast_ref::<reqwest::Error>() {
        return request_error.is_timeout()
            || request_error.is_connect()
            || request_error.is_request()
            || request_error.is_status()
                && request_error
                    .status()
                    .is_some_and(|status| status.is_server_error() || status.as_u16() == 429);
    }
    if let Some(api) = error.downcast_ref::<TelegramApiError>() {
        let description = api.description.to_lowercase();
        return description.contains("too many requests")
            || description.contains("retry after")
            || description.contains("flood");
    }
    // Unknown error kinds are permanent by default: retrying wastes ~6s per
    // call and, on the polling thread, stalls every other user. Transient
    // HTTP/gateway failures are already classified above.
    false
}

/// The configured Bot API base URL (defaults to the official endpoint).
fn api_base() -> &'static str {
    TELEGRAM_API_BASE
        .get()
        .map(String::as_str)
        .unwrap_or("https://api.telegram.org")
}

fn telegram_url(token: &str, method: &str) -> String {
    telegram_url_at(api_base(), token, method)
}

fn telegram_url_at(base: &str, token: &str, method: &str) -> String {
    format!("{base}/bot{token}/{method}")
}

fn first_url(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|part| {
            part.trim_matches(|ch: char| {
                matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']' | '<' | '>' | ',')
            })
        })
        .find(|part| part.starts_with("http://") || part.starts_with("https://"))
        .map(|part| {
            part.trim_end_matches(['.', ',', ')', ']', '>', '!', '?'])
                .to_string()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_first_url_from_message() {
        assert_eq!(
            first_url("please color https://example.test/chapter/1."),
            Some("https://example.test/chapter/1".to_string())
        );
    }

    #[test]
    fn returns_none_when_message_has_no_url() {
        assert_eq!(first_url("hello mangahue"), None);
    }

    #[test]
    fn creative_rotation_cycles_deterministically_per_chat() {
        // Build a BotState with a styles dir containing the creative styles.
        let dir = tempfile::tempdir().unwrap();
        let creative_ids: Vec<&str> = BUILTIN_STYLES
            .iter()
            .map(|&(id, _)| id)
            .filter(|id| id.starts_with("creative_"))
            .collect();
        assert_eq!(creative_ids.len(), 8, "expected 8 creative styles");
        for id in &creative_ids {
            std::fs::write(dir.path().join(format!("{id}.yaml")), "id: {id}\n").unwrap();
        }
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: dir.path().join("usage.json"),
            daily_limit: 0,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: Some(dir.path().to_path_buf()),
            palettes_dir: dir.path().join("palettes"),
        };

        // Consecutive jobs cycle through all 8 moods, then repeat.
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..16 {
            let path = state.next_creative_style(42).unwrap();
            let file = path.file_stem().unwrap().to_str().unwrap().to_string();
            seen.push(file);
        }
        for i in 0..8 {
            assert_eq!(seen[i], creative_ids[i]);
            assert_eq!(seen[i + 8], seen[i], "cycle repeats after 8 jobs");
        }

        // A different chat starts its own cycle at the first mood.
        let first_other = state.next_creative_style(7).unwrap();
        assert_eq!(
            first_other.file_stem().unwrap().to_str().unwrap(),
            creative_ids[0]
        );
    }

    #[test]
    fn bot_state_bounds_jobs_and_supports_cancellation() {
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: PathBuf::from("test_usage.json"),
            daily_limit: 100,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: None,
            palettes_dir: PathBuf::from("configs/palettes"),
        };

        // No job can start when the pool is full.
        let first = state.try_start(10, 1).unwrap();
        assert!(state.try_start(10, 1).is_none());
        assert!(state.try_start(11, 1).is_none());
        assert!(state.is_active(10));
        assert!(state.cancel(10));
        assert!(first.load(Ordering::Acquire));
        state.finish(10);
        assert!(!state.is_active(10));
        assert!(state.try_start(11, 1).is_some());
    }

    #[test]
    fn queue_is_fifo_and_enforces_admission_limits() {
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: PathBuf::from("test_usage.json"),
            daily_limit: 100,
            max_queued_jobs: 4,
            max_queued_jobs_per_chat: 2,
            styles_dir: None,
            palettes_dir: PathBuf::from("configs/palettes"),
        };
        assert_eq!(state.enqueue(1, QueuedJob::Link { url: "a".into() }), Ok(1));
        assert_eq!(state.enqueue(2, QueuedJob::Link { url: "b".into() }), Ok(2));
        // The same chat may queue several links; each starts after the
        // previous one finishes (dispatch defers jobs of active chats).
        assert_eq!(state.enqueue(1, QueuedJob::Link { url: "c".into() }), Ok(3));
        assert!(state
            .enqueue(1, QueuedJob::Link { url: "d".into() })
            .is_err());
        assert_eq!(state.queue_len(), 3);
        // FIFO order: chat 1's second job sits behind chat 2's job.
        assert_eq!(state.queue_position(2), Some(1));
        assert_eq!(state.queue_position(9), None);
        // /cancel semantics: clearing a chat's jobs removes all of them.
        assert_eq!(state.dequeue(1), 2);
        assert_eq!(state.queue_len(), 1);
        assert_eq!(state.dequeue(1), 0);
    }

    #[test]
    fn delivery_gate_fails_closed() {
        assert!(delivery_quality_approved(Some("production_ready_preview")));
        assert!(delivery_quality_approved(Some(
            "production_ready_with_tone_training_warning"
        )));
        assert!(!delivery_quality_approved(None));
        assert!(!delivery_quality_approved(Some(
            "needs_protected_region_cleanup"
        )));
    }

    #[test]
    fn zero_retention_purges_user_images_only() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["delivered", "quality", "references"] {
            let path = dir.path().join(name).join("nested");
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("user-image.bin"), b"private").unwrap();
        }
        let usage = dir.path().join("usage").join("daily.json");
        std::fs::create_dir_all(usage.parent().unwrap()).unwrap();
        std::fs::write(&usage, b"{}").unwrap();

        purge_expired_user_data(dir.path(), 0).unwrap();

        assert!(!dir.path().join("delivered/nested/user-image.bin").exists());
        assert!(!dir.path().join("quality/nested/user-image.bin").exists());
        assert!(!dir.path().join("references/nested/user-image.bin").exists());
        assert!(usage.exists());
    }

    #[test]
    fn retention_purges_legacy_chat_workspaces_without_touching_named_state() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("12345").join("job-id").join("downloads");
        std::fs::create_dir_all(&legacy).unwrap();
        let uploaded = legacy.join("page.jpg");
        std::fs::write(&uploaded, b"user-image").unwrap();
        let settings = dir.path().join("settings").join("chat_12345.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(&settings, b"{}").unwrap();

        purge_expired_user_data(dir.path(), 0).unwrap();

        assert!(!uploaded.exists());
        assert!(settings.exists());
    }

    #[test]
    fn zero_daily_limit_means_unlimited_access() {
        let dir = tempfile::tempdir().unwrap();
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: dir.path().join("usage.json"),
            daily_limit: 0,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: None,
            palettes_dir: PathBuf::from("configs/palettes"),
        };

        // No daily limit means unlimited page usage.
        state.record_pages(42, 250);
        assert_eq!(state.pages_used_today(42), 250);
        assert_eq!(state.remaining_pages_today(42), usize::MAX);
    }

    #[test]
    fn resolves_page_specific_region_lock_files() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("page_001.region-locks.json");
        std::fs::write(&lock, "{}").unwrap();
        assert_eq!(
            resolve_region_locks(Some(dir.path()), "page_001").as_deref(),
            Some(lock.as_path())
        );
        assert!(resolve_region_locks(Some(dir.path()), "page_999").is_none());
    }

    #[test]
    fn loads_yaml_style_names_and_resolves_style_files() {
        let dir = tempfile::tempdir().unwrap();
        let style = dir.path().join("natural_manga.yaml");
        std::fs::write(&style, "id: natural_manga\nname: Natural Manga\n").unwrap();
        let state = Arc::new(BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: dir.path().join("usage.json"),
            daily_limit: 100,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: Some(dir.path().to_path_buf()),
            palettes_dir: dir.path().to_path_buf(),
        });

        assert_eq!(load_style_name(&style).as_deref(), Some("Natural Manga"));
        assert_eq!(
            resolve_style_file(&state, "natural_manga").as_deref(),
            Some(style.as_path())
        );
    }

    #[test]
    fn validates_direct_image_urls() {
        let v = validate_user_url("https://example.com/page.png", true);
        assert!(v.error.is_none());
        assert!(v.warning.is_some());

        // Photos disabled (launch mode): direct image URLs are refused.
        let v = validate_user_url("https://example.com/page.png", false);
        assert!(v.error.is_some());

        let v = validate_user_url("https://example.com/chapter/1", true);
        assert!(v.error.is_none());
        assert!(v.warning.is_none());

        let v = validate_user_url("http://127.0.0.1/page.png", true);
        assert!(v.error.is_some());
    }

    #[test]
    fn recognizes_downloaded_image_formats() {
        assert!(has_image_magic(b"\xFF\xD8\xFF\xE0jpeg"));
        assert!(has_image_magic(b"\x89PNG\r\n\x1A\nrest"));
        assert!(has_image_magic(b"RIFF\x00\x00\x00\x00WEBPrest"));
        assert!(!has_image_magic(b"<!doctype html>"));
    }

    #[test]
    fn derives_safe_image_extensions_from_urls() {
        assert_eq!(
            image_extension_from_url("https://cdn.test/page.webp?token=secret"),
            "webp"
        );
        assert_eq!(
            image_extension_from_url("https://cdn.test/render?id=10&format=png"),
            "jpg"
        );
        assert_eq!(
            image_extension_from_url("https://cdn.test/page.php?file=a.png"),
            "jpg"
        );
    }

    #[test]
    fn high_quality_delivery_keeps_master_and_writes_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("page.png");
        image::RgbImage::from_pixel(16, 16, image::Rgb([120, 80, 60]))
            .save(&source)
            .unwrap();

        let delivery = prepare_delivery(&source, DeliveryFormat::Jpeg, 100).unwrap();

        assert!(source.exists());
        assert_eq!(
            delivery.extension().and_then(|value| value.to_str()),
            Some("jpg")
        );
        assert_eq!(
            image::ImageReader::open(&delivery)
                .unwrap()
                .with_guessed_format()
                .unwrap()
                .format(),
            Some(image::ImageFormat::Jpeg)
        );
    }

    #[test]
    fn scratch_budget_counts_nested_files() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("chat").join("job");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("page.bin"), vec![0_u8; 128]).unwrap();

        assert!(ensure_scratch_budget(dir.path(), 128).is_ok());
        assert!(ensure_scratch_budget(dir.path(), 127).is_err());
    }

    #[test]
    fn formats_progress_bar_correctly() {
        assert_eq!(progress_bar(0, 10, 10), "░░░░░░░░░░");
        assert_eq!(progress_bar(5, 10, 10), "▓▓▓▓▓░░░░░");
        assert_eq!(progress_bar(10, 10, 10), "▓▓▓▓▓▓▓▓▓▓");
    }

    #[test]
    fn md2_escape_handles_all_specials() {
        assert_eq!(md2_escape("plain name"), "plain name");
        assert_eq!(md2_escape("a_b*c(d)"), "a\\_b\\*c\\(d\\)");
        assert_eq!(
            md2_escape("Natural Manga (default)"),
            "Natural Manga \\(default\\)"
        );
        assert_eq!(md2_escape("x~y.z!"), "x\\~y\\.z\\!");
    }

    #[test]
    fn madara_paged_url_is_normalized_to_continuous_reader() {
        let paged =
            url::Url::parse("https://ilikecomix.com/manga/c/some-comic/chapter-1/?style=paged&x=1")
                .unwrap();
        let normalized = normalize_madara_url(&paged);
        assert_eq!(
            normalized,
            "https://ilikecomix.com/manga/c/some-comic/chapter-1/?x=1"
        );
        // Non-Madara URLs are untouched.
        let other = url::Url::parse("https://nhentai.net/g/1/?style=paged").unwrap();
        assert_eq!(normalize_madara_url(&other), other.to_string());
        // No query at all: untouched.
        let plain = url::Url::parse("https://allporncomic.com/porncomic/x/y/").unwrap();
        assert_eq!(normalize_madara_url(&plain), plain.to_string());
    }

    #[test]
    fn dequeue_removes_only_the_chats_job() {
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: PathBuf::from("test_usage.json"),
            daily_limit: 0,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: None,
            palettes_dir: PathBuf::from("configs/palettes"),
        };
        let mut queue = lock_or_recover(&state.queue);
        queue.push_back((
            1,
            QueuedJob::Photo {
                file_id: "a".into(),
            },
        ));
        queue.push_back((
            2,
            QueuedJob::Photo {
                file_id: "b".into(),
            },
        ));
        drop(queue);

        assert_eq!(state.dequeue(1), 1);
        assert_eq!(state.dequeue(1), 0);
        assert_eq!(state.queue_len(), 1);
        assert_eq!(state.queue_position(2), Some(0));
    }

    #[test]
    fn slot_lifecycle_never_leaks_with_queue_fairness_order() {
        // Regression for the C1 slot leak: the photo/link paths now check
        // the queue BEFORE taking a slot, so a request that gets queued can
        // never hold a slot. Verify the two primitives together: a queued
        // chat consumes no slot, and finishing releases the slot for the
        // next job.
        let state = BotState {
            active_jobs: AtomicUsize::new(0),
            active_chats: Mutex::new(HashMap::new()),
            per_chat_settings: Mutex::new(HashMap::new()),
            job_counter: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            usage_lock: Mutex::new(()),
            usage_path: PathBuf::from("test_usage.json"),
            daily_limit: 0,
            max_queued_jobs: 16,
            max_queued_jobs_per_chat: 1,
            styles_dir: None,
            palettes_dir: PathBuf::from("configs/palettes"),
        };
        // Chat 1 takes the only slot in a 1-slot pool.
        let cancel = state.try_start(1, 1).unwrap();
        // Chat 2 is queued (as the fixed call sites do when the queue is
        // non-empty): it must not hold a slot.
        let _ = state.enqueue(
            2,
            QueuedJob::Photo {
                file_id: "b".into(),
            },
        );
        assert_eq!(state.active_jobs.load(Ordering::Acquire), 1);
        assert!(state.try_start(2, 1).is_none());
        // Job 1 finishes: slot freed, dispatch starts the queued job.
        state.finish(1);
        drop(cancel);
        assert_eq!(state.active_jobs.load(Ordering::Acquire), 0);
        let next_cancel = state.try_start(2, 1).unwrap();
        state.finish(2);
        drop(next_cancel);
        assert_eq!(state.active_jobs.load(Ordering::Acquire), 0);
    }
}
