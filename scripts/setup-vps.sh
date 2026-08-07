#!/usr/bin/env bash
set -Eeuo pipefail

# Web-only deployment for Ubuntu 22.04/24.04. Run from the checked-out repo.
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo bash scripts/setup-vps.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOKEN_FILE="$APP_DIR/.mangahue-admin-token"
ADMIN_EMAIL="${MANGAHUE_ADMIN_EMAIL:-admin@mangahue.local}"
ADMIN_NAME="${MANGAHUE_ADMIN_NAME:-MangaHue Admin}"

log() { printf '[MangaHue] %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || return 1; }

if [[ ! -f "$APP_DIR/Dockerfile" || ! -f "$APP_DIR/docker-compose.yml" ]]; then
  echo "Run this script from a complete MangaHue checkout." >&2
  exit 2
fi

log "Installing host dependencies"
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg jq ufw

if ! need docker; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu %s stable\n' \
    "$(dpkg --print-architecture)" "${VERSION_CODENAME}" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

if [[ ! -f /swapfile ]]; then
  log "Creating a 2 GB emergency swap file"
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  printf '/swapfile none swap sw 0 0\n' >> /etc/fstab
fi

log "Configuring firewall without resetting existing rules"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null

if [[ ! -f "$APP_DIR/.env" ]]; then
  cat > "$APP_DIR/.env" <<'ENV'
MANGAHUE_BIND=127.0.0.1
API_PORT=7878
MANGAHUE_MAX_CONCURRENT_JOBS=1
MANGAHUE_JOB_RETENTION_HOURS=168
MANGAHUE_MODEL_THREADS=2
MANGAHUE_MODEL_TIMEOUT_SECONDS=900
MANGAHUE_API_MEMORY=4608m
MANGAHUE_API_CPUS=2.0
RUST_LOG=mangahue=info,tower_http=info
ENV
  chmod 600 "$APP_DIR/.env"
fi

mkdir -p "$APP_DIR/models"
cd "$APP_DIR"

log "Downloading and verifying the compact production color model"
bash scripts/download_ddcolor_tiny.sh "$APP_DIR/models/ddcolor-tiny-fp16.onnx"

log "Building the compact web runtime"
docker compose build api

if [[ ! -s "$TOKEN_FILE" ]]; then
  log "Creating the first administrator access token"
  bootstrap_output="$(
    docker compose run --rm --no-deps --entrypoint mangahue api \
      production bootstrap \
      --database /data/production.db \
      --email "$ADMIN_EMAIL" \
      --name "$ADMIN_NAME" \
      --tier admin
  )"
  admin_token="$(printf '%s\n' "$bootstrap_output" | sed -n 's/^api_token=//p' | tail -n 1)"
  if [[ -z "$admin_token" ]]; then
    echo "Administrator token creation failed." >&2
    exit 3
  fi
  printf '%s\n' "$admin_token" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi

log "Starting the web API"
docker compose up -d api

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${API_PORT:-7878}/health" >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${API_PORT:-7878}/health" | jq .

install -m 0755 "$APP_DIR/scripts/healthcheck.sh" /usr/local/bin/mangahue-healthcheck
cat > /etc/cron.d/mangahue-web <<CRON
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
*/5 * * * * root MANGAHUE_COMPOSE_DIR=$APP_DIR /usr/local/bin/mangahue-healthcheck >> /var/log/mangahue-healthcheck.log 2>&1
17 3 * * * root cd $APP_DIR && docker compose exec -T api mangahue production backup --database /data/production.db --output /data/backups/production-\$(date -u +\%Y\%m\%d).db >> /var/log/mangahue-backup.log 2>&1
29 3 * * * root cd $APP_DIR && docker compose exec -T api sh -c 'find /data/backups -type f -mtime +14 -delete; find /data/jobs -type f -name "*.partial" -mmin +60 -delete' >> /var/log/mangahue-cleanup.log 2>&1
CRON
chmod 644 /etc/cron.d/mangahue-web

log "Web runtime is healthy"
log "Administrator token: $TOKEN_FILE"
log "The API listens only on 127.0.0.1:${API_PORT:-7878}; add TLS reverse proxy after choosing a domain."
