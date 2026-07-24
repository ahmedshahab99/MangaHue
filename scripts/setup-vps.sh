#!/usr/bin/env bash
set -euo pipefail

# ── MangaHue VPS Setup ────────────────────────────────────────────────────
# Targets: Ubuntu 22.04 / 24.04 LTS
# Run as root or user with sudo:  sudo bash setup-vps.sh

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { printf "${GREEN}[+]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
err()  { printf "${RED}[x]${NC} %s\n" "$*"; exit 1; }

# ── Preflight ─────────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then err "Please run as root: sudo bash $0"; fi

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  ;;
  aarch64) ;;
  *)       warn "Untested architecture: $ARCH. Continuing anyway." ;;
esac

log "Starting MangaHue VPS setup on $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"

# ── 1. System packages ────────────────────────────────────────────────────
log "Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

log "Installing prerequisites..."
apt-get install -y -qq \
  curl ca-certificates gnupg lsb-release \
  ufw htop iotop ncdu unzip jq

# ── 2. Docker ─────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  log "Installing Docker..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
  log "Docker $(docker --version) installed."
else
  log "Docker already installed: $(docker --version)"
fi

# ── 3. Swap file (2 GB) ──────────────────────────────────────────────────
if [ ! -f /swapfile ]; then
  log "Creating 2 GB swap file..."
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  log "Swap enabled: $(free -h | awk '/^Swap:/ {print $2}')"
else
  log "Swap file already exists."
fi

# ── 4. UFW firewall ──────────────────────────────────────────────────────
log "Configuring UFW firewall..."
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
log "UFW status:"
ufw status verbose | sed 's/^/       /'

# ── 5. Docker log rotation ────────────────────────────────────────────────
log "Configuring Docker daemon log rotation..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
JSON

systemctl restart docker
log "Docker log rotation configured."

# ── 6. Host log rotation for /var/log ─────────────────────────────────────
log "Configuring system log rotation..."
cat > /etc/logrotate.d/mangahue <<'LOGROTATE'
/var/log/mangahue/*.log {
  daily
  rotate 7
  compress
  delaycompress
  missingok
  notifempty
  copytruncate
}
LOGROTATE
mkdir -p /var/log/mangahue

# ── 7. App directory ─────────────────────────────────────────────────────
APP_DIR="/opt/mangahue"
log "Creating app directory: $APP_DIR"
MODEL_DIR="$APP_DIR/models/manga-onnx"
MODEL_PATH="$MODEL_DIR/manga-colorize-fp16.onnx"
MODEL_URL="https://huggingface.co/Faridzar/manga-colorization-v2-onnx/resolve/main/manga-colorize-fp16.onnx"
MODEL_SHA256="39660d0047ea6f1a0ddee6aa89054997f95ea566f4d56ff762f66dbcf1a1a7ef"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ] || ! echo "$MODEL_SHA256  $MODEL_PATH" | sha256sum --check --status; then
  log "Downloading compact MangaHue ONNX model..."
  curl -fL --retry 3 "$MODEL_URL" -o "$MODEL_PATH.partial"
  echo "$MODEL_SHA256  $MODEL_PATH.partial" | sha256sum --check --status \
    || { rm -f "$MODEL_PATH.partial"; err "ONNX model checksum mismatch"; }
  mv "$MODEL_PATH.partial" "$MODEL_PATH"
fi

# Write .env from env var (optional) or create a placeholder
if [ ! -f "$APP_DIR/.env" ]; then
  cat > "$APP_DIR/.env" <<'ENVFILE'
TELEGRAM_BOT_TOKEN=__FILL_ME_IN__
REGISTRY=ghcr.io
OWNER=mangahue
TAG=latest
MANGAHUE_PYTHON=python
MANGAHUE_MODEL_THREADS=2
MANGAHUE_MODEL_TIMEOUT_SECONDS=180
RUST_LOG=mangahue=info,tower_http=info
DAILY_PAGE_LIMIT=100
TELEGRAM_MAX_PAGES=32
ADMIN_CHAT_ID=
CLEANUP_AFTER_HOURS=24
WORKER_ID=gpu-worker-1
API_PORT=7878
MANGAHUE_MODEL_DIR=/opt/mangahue/models
ENVFILE
  warn "Created $APP_DIR/.env — EDIT TELEGRAM_BOT_TOKEN before starting!"
fi

# ── 8. Pull compose file ─────────────────────────────────────────────────
log "Writing docker-compose.prod.yml..."
# Use the one from the repo or create inline
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
  # The user should copy docker-compose.prod.yml here, or we write a minimal one
  cat > "$APP_DIR/docker-compose.yml" <<'COMPOSE'
services:
  api:
    image: ${REGISTRY:-ghcr.io}/${OWNER:-mangahue}/mangahue:${TAG:-latest}
    pull_policy: always
    restart: unless-stopped
    ports:
      - "${API_PORT:-7878}:7878"
    environment:
      RUST_LOG: ${RUST_LOG:-mangahue=info,tower_http=info}
      MANGAHUE_PYTHON: ${MANGAHUE_PYTHON:-python}
    volumes:
      - mangahue-data:/data
      - ${MANGAHUE_MODEL_DIR:-./models}:/app/models:ro
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:7878/health"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M

  worker:
    image: ${REGISTRY:-ghcr.io}/${OWNER:-mangahue}/mangahue:${TAG:-latest}
    pull_policy: always
    restart: unless-stopped
    entrypoint: ["mangahue-worker"]
    command:
      - "--database"
      - "/data/production.db"
      - "--object-store"
      - "/data/objects"
      - "--worker-id"
      - "${WORKER_ID:-gpu-worker-1}"
    environment:
      RUST_LOG: ${RUST_LOG:-mangahue=info}
      MANGAHUE_MODEL_THREADS: ${MANGAHUE_MODEL_THREADS:-2}
      MANGAHUE_MODEL_TIMEOUT_SECONDS: ${MANGAHUE_MODEL_TIMEOUT_SECONDS:-180}
    volumes:
      - mangahue-data:/data
      - ${MANGAHUE_MODEL_DIR:-./models}:/app/models:ro
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "pgrep -f mangahue-worker || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4096M

  telegram-small:
    image: ${REGISTRY:-ghcr.io}/${OWNER:-mangahue}/mangahue:${SMALL_TAG:-small}
    pull_policy: always
    restart: unless-stopped
    entrypoint: ["mangahue-telegram"]
    command:
      - "--work-dir"
      - "/data/telegram"
      - "--max-pages"
      - "${TELEGRAM_MAX_PAGES:-32}"
      - "--max-concurrent-jobs"
      - "1"
      - "--model"
      - "/app/models/manga-onnx/manga-colorize-fp16.onnx"
      - "--preset"
      - "/app/configs/engine.telegram-cpu.yaml"
      - "--style"
      - "/app/configs/styles/studio_anime.yaml"
      - "--daily-limit"
      - "${DAILY_PAGE_LIMIT:-100}"
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:?required}
      RUST_LOG: ${RUST_LOG:-mangahue=info}
      MANGAHUE_MODEL_THREADS: ${MANGAHUE_MODEL_THREADS:-2}
      MANGAHUE_MODEL_TIMEOUT_SECONDS: ${MANGAHUE_MODEL_TIMEOUT_SECONDS:-180}
      CLEANUP_AFTER_HOURS: ${CLEANUP_AFTER_HOURS:-24}
      ADMIN_CHAT_ID: ${ADMIN_CHAT_ID:-}
    volumes:
      - mangahue-data:/data
      - ${MANGAHUE_MODEL_DIR:-./models}:/app/models:ro
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"
    healthcheck:
      test: ["CMD-SHELL", "pgrep -f mangahue-telegram || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1536M

volumes:
  mangahue-data:
COMPOSE
  log "Wrote $APP_DIR/docker-compose.yml"
fi

# ── 9. Pull images ────────────────────────────────────────────────────────
log "Logging into GHCR (if GHCR_TOKEN is set)..."
if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-mangahue}" --password-stdin
fi

log "Pulling Docker images..."
cd "$APP_DIR"
docker compose pull 2>/dev/null || docker compose pull 2>/dev/null || warn "Pull failed — will attempt on start."

# ── 10. Healthcheck script ────────────────────────────────────────────────
HEALTHSCRIPT="/opt/mangahue/scripts/healthcheck.sh"
log "Installing healthcheck script to $HEALTHSCRIPT"
mkdir -p "$(dirname "$HEALTHSCRIPT")"
cat > "$HEALTHSCRIPT" <<'HEALTHSCRIPT'
#!/usr/bin/env bash
# MangaHue healthcheck — polls docker and alerts via Telegram
set -euo pipefail

BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${ADMIN_CHAT_ID:-}"
APP_DIR="/opt/mangahue"
LOG_FILE="/var/log/mangahue/healthcheck.log"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG_FILE"
}

send_alert() {
  local msg="$1"
  if [ -n "$BOT_TOKEN" ] && [ -n "$CHAT_ID" ]; then
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
      -H "Content-Type: application/json" \
      -d "$(jq -nc --arg text "[MangaHue Alert] $msg" '{"chat_id": $CHAT_ID | tonumber, "text": $text}')" \
      >/dev/null 2>&1 || true
  fi
}

failures=0

# Check telegram bot container
if docker ps --format '{{.Names}}' | grep -q 'mangahue.*telegram'; then
  log "OK: telegram bot container is running."
else
  log "FAIL: telegram bot container is NOT running."
  send_alert "Telegram bot container is down on $(hostname)"
  failures=$((failures + 1))
fi

# Check disk usage (>90% triggers alert)
usage=$(df -h / | awk 'NR==2 {gsub(/%/,""); print $5}')
if [ "$usage" -gt 90 ]; then
  log "FAIL: disk usage at ${usage}%"
  send_alert "Disk usage critical: ${usage}% on $(hostname)"
  failures=$((failures + 1))
else
  log "OK: disk usage at ${usage}%"
fi

# Check Docker daemon
if docker info >/dev/null 2>&1; then
  log "OK: Docker daemon is responsive."
else
  log "FAIL: Docker daemon is unresponsive."
  send_alert "Docker daemon unresponsive on $(hostname)"
  failures=$((failures + 1))
fi

if [ "$failures" -eq 0 ]; then
  log "All checks passed."
else
  log "$failures check(s) failed."
fi
HEALTHSCRIPT

chmod +x "$HEALTHSCRIPT"

# ── 11. Cron jobs ────────────────────────────────────────────────────────
log "Setting up cron jobs..."
CRON_FILE="/etc/cron.d/mangahue"

# Daily cleanup cron
CLEANUP_CMD="docker exec \$(docker ps -q -f name=mangahue.*telegram | head -1) mangahue-cli cleanup --work-dir /data/telegram --max-age-hours \${CLEANUP_AFTER_HOURS:-24} 2>&1 | tee -a /var/log/mangahue/cleanup.log"

cat > "$CRON_FILE" <<'CRON'
# MangaHue cron jobs
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Healthcheck every 5 minutes
*/5 * * * * root /opt/mangahue/scripts/healthcheck.sh >> /var/log/mangahue/healthcheck.log 2>&1

# Daily cleanup at 3:00 AM UTC
0 3 * * * root docker exec $(docker ps -q -f name=mangahue | head -1) mangahue-cli cleanup --work-dir /data/telegram --max-age-hours ${CLEANUP_AFTER_HOURS:-24} >> /var/log/mangahue/cleanup.log 2>&1 || true
CRON

chmod 644 "$CRON_FILE"
log "Cron jobs installed."

# ── 12. Start services ───────────────────────────────────────────────────
log "Starting MangaHue services..."
cd "$APP_DIR"
if grep -Eq '^TELEGRAM_BOT_TOKEN=.+$' "$APP_DIR/.env" \
  && ! grep -q '^TELEGRAM_BOT_TOKEN=__FILL_ME_IN__$' "$APP_DIR/.env"; then
  docker compose --profile telegram-small up -d telegram-small
else
  warn "Telegram token is not configured; services were installed but not started."
fi

sleep 5
log "Container status:"
docker compose ps 2>/dev/null || docker compose ps

# ── 13. Done ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  MangaHue VPS setup complete!"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Edit $APP_DIR/.env and set:"
echo "     - TELEGRAM_BOT_TOKEN  (from @BotFather)"
echo "     - ADMIN_CHAT_ID       (your Telegram chat ID)"
echo ""
echo "  2. The compact ONNX model was downloaded and checksum-verified automatically."
echo ""
echo "  3. Restart services after .env changes:"
echo "     cd $APP_DIR && docker compose --profile telegram-small up -d telegram-small"
echo ""
echo "  4. View logs:"
echo "     cd $APP_DIR && docker compose logs -f --tail=50"
echo ""
echo "  5. Healthcheck logs:"
echo "     tail -f /var/log/mangahue/healthcheck.log"
echo ""
echo "  6. Telegram healthcheck (manual test):"
echo "     /opt/mangahue/scripts/healthcheck.sh"
echo ""
echo "  Firewall: UFW (SSH, 80, 443 open)"
echo "  Swap:     2 GB"
echo "  Logs:     rotated daily, kept 7 days"
echo "═══════════════════════════════════════════════════════════════"
