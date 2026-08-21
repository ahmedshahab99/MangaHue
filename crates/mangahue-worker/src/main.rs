use anyhow::{Context, Result};
use clap::Parser;
use mangahue_chapter::{run_chapter_job, ChapterJobRequest};
use mangahue_orchestrator::{ColorizeRequest, MangaHueEngine};
use mangahue_production::{
    ArtifactRecord, DurableJob, LocalObjectStore, ObjectStore, ProductionStore,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "mangahue-worker", about = "Durable MangaHue inference worker")]
struct Args {
    #[arg(long, default_value = ".mangahue/production.db")]
    database: PathBuf,
    #[arg(long, default_value = ".mangahue/objects")]
    object_store: PathBuf,
    #[arg(long)]
    worker_id: Option<String>,
    #[arg(long, default_value_t = 120)]
    lease_seconds: i64,
    #[arg(long, default_value_t = 1_000)]
    poll_ms: u64,
    #[arg(long)]
    once: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("mangahue=info")
        .init();
    let args = Args::parse();
    let store = ProductionStore::open(&args.database)?;
    let objects = LocalObjectStore::new(&args.object_store)?;
    let worker_id = args
        .worker_id
        .unwrap_or_else(|| format!("worker-{}", Uuid::new_v4()));

    loop {
        match store.claim(&worker_id, args.lease_seconds)? {
            Some(job) => process_job(&store, &objects, &worker_id, args.lease_seconds, job),
            None if args.once => return Ok(()),
            None => std::thread::sleep(Duration::from_millis(args.poll_ms.max(50))),
        }
        if args.once {
            return Ok(());
        }
    }
}

fn process_job(
    store: &ProductionStore,
    objects: &LocalObjectStore,
    worker_id: &str,
    lease_seconds: i64,
    job: DurableJob,
) {
    tracing::info!(job_id = %job.id, kind = %job.kind, attempt = job.attempt, "claimed job");
    let stop = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU32::new(0f32.to_bits()));
    let heartbeat = spawn_heartbeat(
        store.clone(),
        job.id.clone(),
        worker_id.to_string(),
        lease_seconds,
        Arc::clone(&stop),
        Arc::clone(&cancelled),
        Arc::clone(&progress),
    );

    let result = execute_job(&job, objects, store, &cancelled, &progress);
    stop.store(true, Ordering::Release);
    let _ = heartbeat.join();

    match result {
        Ok(value) => {
            if let Err(error) = store.complete(&job.id, worker_id, &value) {
                tracing::error!(job_id = %job.id, "failed to complete job: {error:#}");
            }
        }
        Err(error) => {
            let retryable = !cancelled.load(Ordering::Acquire);
            if let Err(store_error) = store.fail(&job.id, worker_id, &error.to_string(), retryable)
            {
                tracing::error!(job_id = %job.id, "failed to record job failure: {store_error:#}");
            }
        }
    }
}

fn spawn_heartbeat(
    store: ProductionStore,
    job_id: String,
    worker_id: String,
    lease_seconds: i64,
    stop: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let interval = Duration::from_secs((lease_seconds.max(9) / 3) as u64);
        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(interval.min(Duration::from_secs(30)));
            if stop.load(Ordering::Acquire) {
                break;
            }
            let value = f32::from_bits(progress.load(Ordering::Acquire));
            match store.heartbeat(&job_id, &worker_id, lease_seconds, value) {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    cancelled.store(true, Ordering::Release);
                    break;
                }
            }
        }
    })
}

fn execute_job(
    job: &DurableJob,
    objects: &LocalObjectStore,
    store: &ProductionStore,
    cancelled: &Arc<AtomicBool>,
    progress: &Arc<AtomicU32>,
) -> Result<serde_json::Value> {
    match job.kind.as_str() {
        "colorize" => {
            let mut request: ColorizeRequest = serde_json::from_value(job.payload.clone())
                .context("invalid colorize job payload")?;
            request.cancel_token = Some(Arc::clone(cancelled));
            let (sender, mut receiver) = mangahue_progress_channel();
            request.progress_sender = Some(sender);
            let progress_value = Arc::clone(progress);
            let progress_thread = std::thread::spawn(move || {
                while let Some(update) = receiver.blocking_recv() {
                    progress_value.store(update.percent.to_bits(), Ordering::Release);
                }
            });
            let result = MangaHueEngine::new().colorize_for_delivery(request)?;
            let _ = progress_thread.join();
            let artifact = store_output(
                objects,
                store,
                job,
                &result.output_path,
                "colorized_page",
                "image/png",
            )?;
            store.record_usage(&job.user_id, "pages", 1, Some(&job.id))?;
            store.record_usage(&job.user_id, "output_bytes", artifact.bytes, Some(&job.id))?;
            Ok(serde_json::json!({"engine": result, "artifact": artifact}))
        }
        "chapter" => {
            let mut request: ChapterJobRequest = serde_json::from_value(job.payload.clone())
                .context("invalid chapter job payload")?;
            request.colorize_template.cancel_token = Some(Arc::clone(cancelled));
            let result = run_chapter_job(request)?;
            let pages = result.processed as u64;
            store.record_usage(&job.user_id, "pages", pages, Some(&job.id))?;
            let artifact = result
                .export_path
                .as_deref()
                .filter(|path| path.is_file())
                .map(|path| {
                    store_output(
                        objects,
                        store,
                        job,
                        path,
                        "chapter_archive",
                        "application/vnd.comicbook+zip",
                    )
                })
                .transpose()?;
            if let Some(artifact) = &artifact {
                store.record_usage(&job.user_id, "output_bytes", artifact.bytes, Some(&job.id))?;
            }
            Ok(serde_json::json!({"chapter": result, "artifact": artifact}))
        }
        other => anyhow::bail!("unsupported durable job kind {other}"),
    }
}

fn mangahue_progress_channel() -> (
    tokio::sync::mpsc::UnboundedSender<mangahue_orchestrator::ProgressUpdate>,
    tokio::sync::mpsc::UnboundedReceiver<mangahue_orchestrator::ProgressUpdate>,
) {
    tokio::sync::mpsc::unbounded_channel()
}

fn store_output(
    objects: &LocalObjectStore,
    store: &ProductionStore,
    job: &DurableJob,
    path: &Path,
    kind: &str,
    mime: &str,
) -> Result<ArtifactRecord> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read worker output {}", path.display()))?;
    let object = objects.put(&format!("users/{}/{}", job.user_id, kind), &bytes)?;
    let artifact = ArtifactRecord {
        id: Uuid::new_v4().to_string(),
        user_id: job.user_id.clone(),
        job_id: Some(job.id.clone()),
        kind: kind.to_string(),
        object_key: object.key,
        sha256: object.sha256,
        bytes: object.bytes,
        mime: mime.to_string(),
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
    };
    store.register_artifact(&artifact)?;
    Ok(artifact)
}
