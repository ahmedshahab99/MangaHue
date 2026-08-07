#!/usr/bin/env bash
set -Eeuo pipefail

COMPOSE_DIR="${MANGAHUE_COMPOSE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
API_PORT="${API_PORT:-7878}"
FAILURES=0

log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { log "FAIL: $*"; FAILURES=$((FAILURES + 1)); }

if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is unavailable"
else
  status="$(cd "$COMPOSE_DIR" && docker compose ps --status running --format json api 2>/dev/null || true)"
  [[ -n "$status" ]] || fail "API container is not running"
fi

health="$(curl -fsS --max-time 10 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null || true)"
if [[ -z "$health" ]]; then
  fail "API health endpoint is unavailable"
elif ! printf '%s' "$health" | jq -e '.status == "ok"' >/dev/null 2>&1; then
  fail "API returned an unhealthy response"
else
  log "API healthy: $health"
fi

disk_used="$(df -P "$COMPOSE_DIR" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
if [[ "$disk_used" -ge 95 ]]; then
  fail "disk usage is ${disk_used}%"
elif [[ "$disk_used" -ge 85 ]]; then
  log "WARN: disk usage is ${disk_used}%"
else
  log "Disk usage: ${disk_used}%"
fi

memory_available="$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
if [[ "$memory_available" -lt 256 ]]; then
  fail "only ${memory_available} MB RAM is available"
else
  log "Available RAM: ${memory_available} MB"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  log "$FAILURES health check(s) failed"
  exit 1
fi
log "All health checks passed"
