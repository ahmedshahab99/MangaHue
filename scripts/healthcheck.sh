#!/usr/bin/env bash
# ── MangaHue Healthcheck ─────────────────────────────────────────────────
# Run from cron every 5 minutes. Alerts admin via Telegram if services are
# down or disk space is critical.
#
# Usage:
#   CRON: */5 * * * * /opt/mangahue/scripts/healthcheck.sh
#   MANUAL: TELEGRAM_BOT_TOKEN=... ADMIN_CHAT_ID=... ./scripts/healthcheck.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
# Use env vars or fall back to .env file if present
BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${ADMIN_CHAT_ID:-}"
LOG_DIR="${HEALTHCHECK_LOG_DIR:-/var/log/mangahue}"
LOG_FILE="$LOG_DIR/healthcheck.log"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-mangahue}"

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────
log() {
  printf "[%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

send_alert() {
  local message="$1"
  local host
  host="$(hostname -f 2>/dev/null || hostname)"

  if [ -z "$BOT_TOKEN" ] || [ -z "$CHAT_ID" ]; then
    log "SKIP alert (BOT_TOKEN or CHAT_ID not set): $message"
    return 0
  fi

  local payload
  payload=$(printf '{"chat_id":%s,"text":"%s"}' \
    "$CHAT_ID" \
    "[MangaHue Alert] ${host}: ${message}" \
  )

  local resp
  resp=$(curl -sS --max-time 10 -X POST \
    "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>&1) || true

  if echo "$resp" | grep -q '"ok":true'; then
    log "Alert sent: $message"
  else
    log "Alert FAILED ($resp): $message"
  fi
}

# ── Checks ────────────────────────────────────────────────────────────────
FAILURES=0

# 1. Telegram bot containers running?
log "check: telegram bot containers"

TELEGRAM_CONTAINERS=$(docker ps --format '{{.Names}}' 2>/dev/null | grep 'telegram' || true)
if [ -n "$TELEGRAM_CONTAINERS" ]; then
  while IFS= read -r c; do
    status=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo "unknown")
    uptime=$(docker inspect -f '{{.State.StartedAt}}' "$c" 2>/dev/null || echo "unknown")
    log "  $c: $status (since $uptime)"
  done <<< "$TELEGRAM_CONTAINERS"
else
  log "  FAIL: no telegram container found"
  send_alert "Telegram bot container is DOWN — no container running"
  FAILURES=$((FAILURES + 1))
fi

# 2. API container running?
log "check: API container"
API_CONTAINER=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'api|mangahue-api' || true)
if [ -n "$API_CONTAINER" ]; then
  status=$(docker inspect -f '{{.State.Status}}' "$API_CONTAINER" 2>/dev/null || echo "unknown")
  log "  $API_CONTAINER: $status"
else
  log "  WARN: API container not running (may be fine if not needed)"
fi

# 3. Worker container running?
log "check: worker container"
WORKER_CONTAINER=$(docker ps --format '{{.Names}}' 2>/dev/null | grep 'worker' || true)
if [ -n "$WORKER_CONTAINER" ]; then
  status=$(docker inspect -f '{{.State.Status}}' "$WORKER_CONTAINER" 2>/dev/null || echo "unknown")
  log "  $WORKER_CONTAINER: $status"
else
  log "  WARN: worker container not running"
fi

# 4. Disk space
log "check: disk space"

DISK_WARN=85
DISK_CRIT=95

df_output=$(df -h / /data 2>/dev/null || df -h /)
while IFS= read -r line; do
  [ -z "$line" ] && continue
  mount=$(echo "$line" | awk '{print $6}')
  usage_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
  avail=$(echo "$line" | awk '{print $4}')

  if [ "$mount" = "Mounted" ] || [ "$mount" = "on" ]; then continue; fi

  log "  $mount: ${usage_pct}% used (${avail} available)"

  if [ "$usage_pct" -ge "$DISK_CRIT" ]; then
    log "  FAIL: $mount critical at ${usage_pct}%"
    send_alert "Disk CRITICAL: $mount at ${usage_pct}% (${avail} free)"
    FAILURES=$((FAILURES + 1))
  elif [ "$usage_pct" -ge "$DISK_WARN" ]; then
    log "  WARN: $mount at ${usage_pct}%"
    send_alert "Disk WARNING: $mount at ${usage_pct}% (${avail} free)"
  fi
done <<< "$df_output"

# 5. Docker daemon health
log "check: Docker daemon"
if docker info >/dev/null 2>&1; then
  containers_total=$(docker ps -a --format '{{.ID}}' 2>/dev/null | wc -l)
  containers_running=$(docker ps --format '{{.ID}}' 2>/dev/null | wc -l)
  log "  Docker OK ($containers_running/$containers_total containers running)"
else
  log "  FAIL: Docker daemon unresponsive"
  send_alert "Docker daemon is DOWN"
  FAILURES=$((FAILURES + 1))
fi

# 6. Memory pressure
log "check: memory"
mem_line=$(free -m | awk '/^Mem:/ {printf "%.0f/%.0f MB (%.0f%%)", $3, $2, ($3/$2)*100}')
swap_line=$(free -m | awk '/^Swap:/ {printf "%.0f/%.0f MB (%.0f%%)", $3, $2, ($2>0 ? ($3/$2)*100 : 0)}')
log "  Mem: $mem_line"
log "  Swap: $swap_line"

# ── Summary ───────────────────────────────────────────────────────────────
if [ "$FAILURES" -eq 0 ]; then
  log "PASS: all healthchecks ok"
  exit 0
else
  log "DONE: $FAILURES check(s) failed"
  exit 1
fi
