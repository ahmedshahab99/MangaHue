use anyhow::{Context, Result};
use clap::Parser;
use mangahue_ingest::{ingest_url, IngestRequest};
use mangahue_orchestrator::{ColorizeRequest, MangaHueEngine, ProgressUpdate};
use reqwest::blocking::multipart::Form;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

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
    #[arg(long, default_value = ".mangahue/telegram")]
    work_dir: PathBuf,
    #[arg(long, default_value_t = 32)]
    max_pages: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_bytes_per_page: u64,
    #[arg(long)]
    model: Option<PathBuf>,
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
    #[arg(long, default_value_t = 2)]
    max_concurrent_jobs: usize,
    #[arg(long, help = "Stop a chapter after the first failed page")]
    fail_fast: bool,
    #[arg(long, default_value_t = 30)]
    poll_timeout_seconds: u64,
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
}

#[derive(Debug, Deserialize)]
struct Message {
    chat: Chat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Chat {
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

#[derive(Default)]
struct BotState {
    active_jobs: AtomicUsize,
    active_chats: Mutex<HashMap<i64, Arc<AtomicBool>>>,
}

impl BotState {
    fn try_start(&self, chat_id: i64, limit: usize) -> Option<Arc<AtomicBool>> {
        let mut chats = self.active_chats.lock().expect("active chat lock poisoned");
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
        self.active_chats
            .lock()
            .expect("active chat lock poisoned")
            .remove(&chat_id);
        self.active_jobs.fetch_sub(1, Ordering::AcqRel);
    }

    fn is_active(&self, chat_id: i64) -> bool {
        self.active_chats
            .lock()
            .expect("active chat lock poisoned")
            .contains_key(&chat_id)
    }

    fn cancel(&self, chat_id: i64) -> bool {
        let chats = self.active_chats.lock().expect("active chat lock poisoned");
        let Some(cancel) = chats.get(&chat_id) else {
            return false;
        };
        cancel.store(true, Ordering::Release);
        true
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = tracing_subscriber::fmt()
        .with_env_filter("mangahue=info")
        .try_init();
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
        .timeout(Duration::from_secs(args.poll_timeout_seconds + 15))
        .user_agent("MangaHueTelegram/0.1")
        .build()
        .context("failed to build Telegram HTTP client")?;
    let identity = configure_bot(&client, &token)?;
    tracing::info!(
        bot_id = identity.id,
        username = identity.username.as_deref().unwrap_or("unknown"),
        "Telegram bot authenticated"
    );

    let mut offset = None;
    let mut backoff = std::time::Duration::from_secs(1);
    let state = Arc::new(BotState::default());
    loop {
        match poll_once(&client, &token, &args, &state, &mut offset) {
            Ok(()) => backoff = std::time::Duration::from_secs(1),
            Err(e) => {
                tracing::error!("poll error (will retry in {}s): {e}", backoff.as_secs());
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
            }
        }
    }
}

fn poll_once(
    client: &Client,
    token: &str,
    args: &Args,
    state: &Arc<BotState>,
    offset: &mut Option<i64>,
) -> Result<()> {
    let updates = get_updates(client, token, *offset, args.poll_timeout_seconds)?;
    for update in updates {
        *offset = Some(update.update_id + 1);
        let Some(message) = update.message else {
            continue;
        };
        let Some(text) = message.text else {
            send_message(
                client,
                token,
                message.chat.id,
                "Send an authorized manga, webtoon, or image URL and I will colorize it.",
            )?;
            continue;
        };
        let chat_id = message.chat.id;
        if text.trim().starts_with("/start") {
            send_message(
                client,
                token,
                chat_id,
                "Send an authorized manga chapter or image URL. MangaHue will report progress and deliver each colored page as soon as it is ready.",
            )?;
            continue;
        }
        if text.trim().starts_with("/status") {
            let status = if state.is_active(chat_id) {
                "Your MangaHue chapter job is running."
            } else {
                "No MangaHue job is active for this chat."
            };
            send_message(client, token, chat_id, status)?;
            continue;
        }
        if text.trim().starts_with("/cancel") {
            let status = if state.cancel(chat_id) {
                "Cancellation requested. MangaHue will stop at the next safe checkpoint."
            } else {
                "No active MangaHue job to cancel."
            };
            send_message(client, token, chat_id, status)?;
            continue;
        }
        let Some(url) = first_url(&text) else {
            send_message(
                client,
                token,
                message.chat.id,
                "Send one authorized manga, webtoon, or image URL.",
            )?;
            continue;
        };
        let Some(cancel_token) = state.try_start(chat_id, args.max_concurrent_jobs) else {
            send_message(
                client,
                token,
                chat_id,
                "A MangaHue job is already active for this chat, or the worker pool is full.",
            )?;
            continue;
        };
        let client = client.clone();
        let token = token.to_string();
        let args = args.clone();
        let state = Arc::clone(state);
        std::thread::spawn(move || {
            if let Err(error) = handle_link(&client, &token, &args, chat_id, &url, cancel_token) {
                let _ = send_message(
                    &client,
                    &token,
                    chat_id,
                    &format!("MangaHue could not finish this link: {error}"),
                );
            }
            state.finish(chat_id);
        });
    }
    Ok(())
}

fn handle_link(
    client: &Client,
    token: &str,
    args: &Args,
    chat_id: i64,
    url: &str,
    cancel_token: Arc<AtomicBool>,
) -> Result<()> {
    let job_id = Uuid::new_v4().to_string();
    let job_dir = args.work_dir.join(chat_id.to_string()).join(&job_id);
    let downloads_dir = job_dir.join("downloads");
    let output_dir = job_dir.join("colored");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    send_message(
        client,
        token,
        chat_id,
        "MangaHue accepted the link. Downloading pages and preparing colorization.",
    )?;
    send_chat_action(client, token, chat_id, "typing")?;
    let ingest = ingest_url(IngestRequest {
        source_url: url.to_string(),
        output_dir: downloads_dir,
        max_pages: args.max_pages,
        max_bytes_per_page: args.max_bytes_per_page,
        min_request_interval: std::time::Duration::from_millis(500),
        selectors: mangahue_ingest::default_selectors(),
        user_agent: None,
    })?;
    if ingest.pages.is_empty() {
        anyhow::bail!("no page images were found");
    }

    send_message(
        client,
        token,
        chat_id,
        &format!("Found {} page image(s). Coloring now.", ingest.pages.len()),
    )?;
    let engine = MangaHueEngine::new();
    let mut sent = 0usize;
    let mut failed = 0usize;
    let total = ingest.pages.len();
    for (position, page) in ingest.pages.iter().enumerate() {
        if cancel_token.load(Ordering::Acquire) {
            send_message(client, token, chat_id, "MangaHue job cancelled.")?;
            return Ok(());
        }
        let stem = page
            .local_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page");
        let output_path = output_dir.join(format!("{stem}.png"));
        let status_message_id = send_message(
            client,
            token,
            chat_id,
            &format!("Coloring page {}/{}: queued", position + 1, total),
        )?;
        let (progress_sender, mut progress_receiver) =
            tokio::sync::mpsc::unbounded_channel::<ProgressUpdate>();
        let progress_client = client.clone();
        let progress_token = token.to_string();
        let progress_thread = std::thread::spawn(move || {
            let mut last_stage = String::new();
            let mut last_percent = -10i32;
            while let Some(update) = progress_receiver.blocking_recv() {
                let percent = update.percent.round() as i32;
                if update.stage != last_stage || percent >= last_percent + 10 || percent >= 100 {
                    let text = format!(
                        "Coloring page {}/{}: {} ({}%)",
                        position + 1,
                        total,
                        humanize_stage(&update.stage),
                        percent.clamp(0, 100)
                    );
                    let _ = edit_message(
                        &progress_client,
                        &progress_token,
                        chat_id,
                        status_message_id,
                        &text,
                    );
                    last_stage = update.stage;
                    last_percent = percent;
                }
            }
        });
        let result = engine.colorize(ColorizeRequest {
            input_path: page.local_path.clone(),
            output_path: output_path.clone(),
            model_path: args.model.clone(),
            config_path: args.config.clone(),
            preset_path: args.preset.clone(),
            style_path: args.style.clone(),
            project_path: args.project.clone(),
            region_locks_path: resolve_region_locks(args.region_locks.as_deref(), stem),
            use_mock_backend: args.mock,
            pyramid_enabled: args.pyramid,
            hint_propagation_enabled: args.hint_propagate,
            cancel_token: Some(Arc::clone(&cancel_token)),
            progress_sender: Some(progress_sender),
            ..ColorizeRequest::default()
        });
        let _ = progress_thread.join();
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                if cancel_token.load(Ordering::Acquire) {
                    send_message(client, token, chat_id, "MangaHue job cancelled.")?;
                    return Ok(());
                }
                failed += 1;
                let _ = edit_message(
                    client,
                    token,
                    chat_id,
                    status_message_id,
                    &format!("Page {}/{} failed: {error}", position + 1, total),
                );
                if args.fail_fast {
                    return Err(error);
                }
                continue;
            }
        };
        edit_message(
            client,
            token,
            chat_id,
            status_message_id,
            &format!("Page {}/{} complete. Uploading.", position + 1, total),
        )?;
        send_chat_action(client, token, chat_id, "upload_document")?;
        let caption = format!(
            "MangaHue page {}/{} | source {} | tiles {} | {}",
            position + 1,
            total,
            page.index,
            result.report.tiles_count,
            result
                .report
                .director_verdict
                .unwrap_or_else(|| "colored".to_string())
        );
        send_document(client, token, chat_id, &output_path, &caption)?;
        sent += 1;
    }

    send_message(
        client,
        token,
        chat_id,
        &format!("Done. Sent {sent} colored page(s); {failed} page(s) failed."),
    )?;
    Ok(())
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
    let url = telegram_url(token, "getUpdates");
    let mut body = json!({
        "timeout": timeout,
        "allowed_updates": ["message"]
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
}

fn configure_bot(client: &Client, token: &str) -> Result<BotIdentity> {
    let identity: TelegramEnvelope<BotIdentity> = client
        .get(telegram_url(token, "getMe"))
        .send()
        .context("failed to call getMe")?
        .error_for_status()
        .context("Telegram getMe failed")?
        .json()
        .context("failed to parse getMe response")?;
    let identity = telegram_result(identity)?;
    let commands: TelegramEnvelope<bool> = client
        .post(telegram_url(token, "setMyCommands"))
        .json(&json!({
            "commands": [
                {"command": "start", "description": "Show MangaHue instructions"},
                {"command": "status", "description": "Check the active chapter job"},
                {"command": "cancel", "description": "Cancel the active chapter job"}
            ]
        }))
        .send()
        .context("failed to call setMyCommands")?
        .error_for_status()
        .context("Telegram setMyCommands failed")?
        .json()
        .context("failed to parse setMyCommands response")?;
    let _ = telegram_result(commands)?;
    Ok(identity)
}

fn send_message(client: &Client, token: &str, chat_id: i64, text: &str) -> Result<i64> {
    let envelope: TelegramEnvelope<SentMessage> = client
        .post(telegram_url(token, "sendMessage"))
        .json(&json!({
            "chat_id": chat_id,
            "text": text
        }))
        .send()
        .context("failed to call sendMessage")?
        .error_for_status()
        .context("Telegram sendMessage failed")?
        .json()
        .context("failed to parse sendMessage response")?;
    Ok(telegram_result(envelope)?.message_id)
}

fn edit_message(
    client: &Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
) -> Result<()> {
    let envelope: TelegramEnvelope<serde_json::Value> = client
        .post(telegram_url(token, "editMessageText"))
        .json(&json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text
        }))
        .send()
        .context("failed to call editMessageText")?
        .error_for_status()
        .context("Telegram editMessageText failed")?
        .json()
        .context("failed to parse editMessageText response")?;
    let _ = telegram_result(envelope)?;
    Ok(())
}

fn send_chat_action(client: &Client, token: &str, chat_id: i64, action: &str) -> Result<()> {
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
}

fn send_document(
    client: &Client,
    token: &str,
    chat_id: i64,
    path: &Path,
    caption: &str,
) -> Result<()> {
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
}

fn telegram_result<T>(envelope: TelegramEnvelope<T>) -> Result<T> {
    if envelope.ok {
        envelope
            .result
            .context("Telegram response was ok but had no result")
    } else {
        anyhow::bail!(
            "Telegram API error: {}",
            envelope
                .description
                .unwrap_or_else(|| "unknown error".to_string())
        );
    }
}

fn telegram_url(token: &str, method: &str) -> String {
    format!("https://api.telegram.org/bot{token}/{method}")
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
    fn bot_state_bounds_jobs_and_supports_cancellation() {
        let state = BotState::default();
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
}
