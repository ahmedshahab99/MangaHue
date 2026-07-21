# MangaHue - Production Deployment Guide

## Architecture

```
Telegram User → Bot API → mangahue-telegram (Rust)
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
                 Ingest   Colorize   Deliver
              (scrape URL) (ONNX/     (per-page
               SSR-safe)   pretrained)  uploads)
                              │
                    ┌─────────┼──────────┐
                    │ Python worker      │
                    │ (manga-colorization│
                    │  v2 or MangaNinja) │
                    └────────────────────┘
```

## Two Backends Available

| Backend | Quality | Hardware | Speed | Model Size |
|---------|---------|----------|-------|------------|
| **manga-colorization-v2** (pretrained-small) | Good | CPU, 2GB+ RAM | ~10-15s/tile | ~140MB |
| **MangaNinja** (CVPR 2025) | Stunning | GPU, 12GB+ VRAM | ~5s/tile | ~12GB |

## Quick Start (VPS)

### 1. Prerequisites
- Ubuntu 22.04/24.04 VPS with **4GB+ RAM** (8GB recommended)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your private GitHub repo access token (for GHCR)

### 2. One-Command Setup
```bash
# Copy setup script to VPS
scp scripts/setup-vps.sh user@vps:/tmp/
ssh user@vps
sudo bash /tmp/setup-vps.sh
```

### 3. Upload Model
```bash
# From your local machine:
scp -r models/pretrained/manga-colorization-v2 user@vps:/opt/mangahue/models/pretrained/
```

### 4. Configure & Start
```bash
ssh user@vps
cd /opt/mangahue
nano .env   # Set TELEGRAM_BOT_TOKEN
docker compose up -d
docker compose logs -f
```

### 5. Test
Send a manga chapter URL to your bot on Telegram. It will:
1. Download page images from the URL
2. Colorize each page in batches of 10
3. Send colored pages as documents

## Manual Setup (without script)

### Windows (local testing)
```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_TOKEN"
cargo run -p mangahue-telegram -- --mock --work-dir .mangahue\telegram
```

### With real model
```powershell
cargo run -p mangahue-telegram -- `
  --model .\models\pretrained\manga-colorization-v2 `
  --preset .\configs\engine.pretrained-small.yaml `
  --hint-propagate --max-concurrent-jobs 1
```

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message |
| `/help` | Show all commands |
| `/status` | Check active job progress |
| `/cancel` | Cancel current job |
| `/style` | List available color styles |
| `/style <name>` | Set color style |
| `/batch` | Show current batch size |
| `/batch <N>` | Set batch size (1-10) |
| `/stats` | Show daily usage |
| `/clean` | Delete all job data |

Just send a manga chapter URL to start colorization.

## Docker Profiles

```bash
# Full setup (API + Worker + Telegram with PyTorch)
docker compose --profile telegram up -d

# Minimal setup (Telegram only, small model, no PyTorch in image)
docker compose --profile telegram-small up -d

# API + Worker only (no Telegram bot)
docker compose up -d
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | **Yes** | - | Bot token from @BotFather |
| `ADMIN_CHAT_ID` | No | - | Telegram ID for alerts |
| `DAILY_PAGE_LIMIT` | No | `100` | Max pages/user/day |
| `MANGAHUE_MODEL_THREADS` | No | `2` | Python thread count |
| `CLEANUP_AFTER_HOURS` | No | `24` | Auto-delete old job data |
| `RUST_LOG` | No | `mangahue=info` | Log level |

## Monitoring

```bash
# View logs
docker compose logs -f telegram

# Run healthcheck manually
TELEGRAM_BOT_TOKEN=... ADMIN_CHAT_ID=... bash scripts/healthcheck.sh

# Healthcheck runs every 5 minutes via cron (set by setup script)
```

## GitHub Actions CI/CD

Set these secrets in your GitHub repo:
- `VPS_HOST` - VPS IP address
- `VPS_USER` - SSH username
- `VPS_SSH_KEY` - Private SSH key
- (Optional) `VPS_PORT` - SSH port (default 22)

Push to `main` → auto-build → auto-deploy.

## Upgrading to MangaNinja (Premium)

1. Download MangaNinja weights from [HuggingFace](https://huggingface.co/Johanan0528/MangaNinjia)
2. Also download: Stable Diffusion v1.5, CLIP ViT-L/14, ControlNet lineart
3. Place in `models/manga-ninja/checkpoints/`:
```
checkpoints/
  StableDiffusion/   (SD v1.5)
  models/
    clip-vit-large-patch14/
    control_v11p_sd15_lineart/
    Annotators/
  MangaNinjia/
    denoising_unet.pth
    reference_unet.pth
    controlnet.pth
    point_net.pth
```
4. Run: `cargo run -p mangahue-telegram -- --model models/manga-ninja`

## File Structure After Setup

```
/opt/mangahue/
  .env                    # Environment config
  docker-compose.yml      # Production compose file
  models/
    pretrained/
      manga-colorization-v2/   # Small model (CPU)
    manga-ninja/               # Premium model (GPU, optional)
  configs/                # Engine presets & styles
  scripts/
    healthcheck.sh        # Monitoring script
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot not responding | Check `TELEGRAM_BOT_TOKEN` in `.env` |
| "No model found" | Verify model in `/opt/mangahue/models/pretrained/` |
| OOM errors | Increase swap: `fallocate -l 4G /swapfile` |
| Slow colorization | Reduce `MANGAHUE_MODEL_THREADS` to 1 |
| Disk full | Run `/clean` in bot or wait for cron cleanup |
| Container crashes | Check logs: `docker compose logs telegram` |
