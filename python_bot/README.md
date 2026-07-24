# MangaHue

Local manga colorization bot for Telegram. Uses an ONNX inference model with
smart semantic colour hints to produce natural, beautiful manga colours.

## What It Does

- Accepts manga chapter URLs from supported sites
- Downloads pages, generates semantic colour hints, colorizes via ONNX
- Applies an 11-stage post-processing pipeline (white balance, screentone
  detection, ink restoration, colour grading, sharpening)
- Sends colorized pages back in batches on Telegram
- Tracks daily usage per user

## Requirements

- Python 3.10+
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- ONNX colorization model (download separately)

### Install

```bash
pip install -r requirements.txt
```

### ONNX Model

Download the manga colorization ONNX model and place it at the path set in
`MANGAHUE_MODEL_PATH` (default:
`<repo>/models/manga-onnx/manga-colorize-fp16.onnx`).

## Environment Variables

| Variable              | Default                        | Description               |
|-----------------------|--------------------------------|---------------------------|
| `TELEGRAM_BOT_TOKEN`  | *(required)*                   | Telegram bot token        |
| `MANGAHUE_MODEL_PATH` | `../models/manga-onnx/...onnx` | Path to ONNX model        |
| `MANGAHUE_WORK_DIR`   | `.mangahue/telegram`           | Temp / usage files        |
| `DAILY_PAGE_LIMIT`    | `0`                            | Max pages/user/day; `0` is unlimited |
| `DEFAULT_BATCH_SIZE`  | `5`                            | Pages per batch           |
| `MAX_CONCURRENT_JOBS` | `2`                            | Concurrent colorize jobs  |
| `MIN_SEND_INTERVAL`   | `3.1`                          | Seconds between sends     |
| `ADMIN_CHAT_ID`       | *(optional)*                   | Error notifications       |
| `HEALTH_PORT`         | *(optional)*                   | Enable HTTP health server |

## Running

```bash
cd python_bot
TELEGRAM_BOT_TOKEN=your_token python bot.py
```

If `HEALTH_PORT=8089` is set, a health endpoint will be available at
`http://localhost:8089/health` returning `{"status": "ok"}`.

## Supported Sites

- **nhentai.net** / **nhentai.to** — gallery pages (`/g/...`)
- **MangaDex.org** — chapter pages (`/chapter/...`)
- **Direct image URLs** — any URL ending in `.jpg`, `.png`, `.webp`, etc.
- **Generic pages** — `<img>` tags scraped from any HTML page

## Bot Commands

| Command    | Description                        |
|------------|------------------------------------|
| `/start`   | Welcome message                    |
| `/help`    | Show commands and daily limit      |
| `/status`  | Check if a job is running          |
| `/cancel`  | Stop the current colorization job  |
| `/batch N` | Set batch size (1--10)             |
| `/stats`   | Show daily usage progress          |
| `/clean`   | Remove temporary files             |

## Deployment (systemd)

```ini
[Unit]
Description=MangaHue Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mangahue
WorkingDirectory=/opt/MangaHue/python_bot
Environment=TELEGRAM_BOT_TOKEN=your_token_here
Environment=MANGAHUE_MODEL_PATH=/opt/MangaHue/models/manga-onnx/manga-colorize-fp16.onnx
Environment=HEALTH_PORT=8089
ExecStart=/opt/MangaHue/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mangahue
```
