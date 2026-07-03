use anyhow::{Context, Result};
use clap::Parser;
use mangahue_ingest::{ingest_url, IngestRequest};
use mangahue_orchestrator::{ColorizeRequest, MangaHueEngine};
use reqwest::blocking::multipart::Form;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Parser)]
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
    #[arg(long, default_value_t = 4)]
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

    let mut offset = None;
    let mut backoff = std::time::Duration::from_secs(1);
    loop {
        match poll_once(&client, &token, &args, &mut offset) {
            Ok(()) => backoff = std::time::Duration::from_secs(1),
            Err(e) => {
                tracing::error!("poll error (will retry in {}s): {e}", backoff.as_secs());
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
            }
        }
    }
}

fn poll_once(client: &Client, token: &str, args: &Args, offset: &mut Option<i64>) -> Result<()> {
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
        let Some(url) = first_url(&text) else {
            send_message(
                client,
                token,
                message.chat.id,
                "Send one authorized manga, webtoon, or image URL.",
            )?;
            continue;
        };
        if let Err(error) = handle_link(client, token, args, message.chat.id, &url) {
            let _ = send_message(
                client,
                token,
                message.chat.id,
                &format!("MangaHue could not finish this link: {error}"),
            );
        }
    }
    Ok(())
}

fn handle_link(client: &Client, token: &str, args: &Args, chat_id: i64, url: &str) -> Result<()> {
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
    for page in &ingest.pages {
        let stem = page
            .local_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page");
        let output_path = output_dir.join(format!("{stem}.png"));
        let result = engine.colorize(ColorizeRequest {
            input_path: page.local_path.clone(),
            output_path: output_path.clone(),
            model_path: args.model.clone(),
            config_path: args.config.clone(),
            preset_path: args.preset.clone(),
            style_path: args.style.clone(),
            project_path: args.project.clone(),
            use_mock_backend: args.mock,
            ..ColorizeRequest::default()
        })?;
        let caption = format!(
            "MangaHue page {} | tiles {} | {}",
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
        &format!("Done. Sent {sent} colored page(s)."),
    )?;
    Ok(())
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

fn send_message(client: &Client, token: &str, chat_id: i64, text: &str) -> Result<()> {
    let envelope: TelegramEnvelope<serde_json::Value> = client
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
}
