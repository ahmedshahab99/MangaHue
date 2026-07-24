#!/usr/bin/env bash
set -Eeuo pipefail

MODE="inspect"
EXPECTED_HOST=""
APP_DIR="/opt/mangahue"

usage() {
  cat <<'EOF'
Usage:
  cleanup-old-vps-bot.sh
  cleanup-old-vps-bot.sh --execute --confirm-host HOSTNAME [--app-dir PATH]

Inspection is the default and never changes the server. Execute mode stops and
removes MangaHue containers/services and terminates legacy MangaHue bot
processes. It preserves models, databases, job data, and .env for migration.
EOF
}

while (($#)); do
  case "$1" in
    --execute)
      MODE="execute"
      shift
      ;;
    --confirm-host)
      EXPECTED_HOST="${2:-}"
      shift 2
      ;;
    --app-dir)
      APP_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

HOST="$(hostname)"
echo "Host: $HOST"
echo "Mode: $MODE"
echo "Application directory: $APP_DIR"

echo
echo "MangaHue containers:"
if command -v docker >/dev/null 2>&1; then
  docker ps -a \
    --filter "name=mangahue" \
    --format '  {{.ID}}  {{.Names}}  {{.Status}}  {{.Image}}' || true
else
  echo "  Docker is not installed."
fi

echo
echo "MangaHue systemd services:"
if command -v systemctl >/dev/null 2>&1; then
  systemctl list-unit-files --type=service --no-legend 2>/dev/null \
    | awk '$1 ~ /^mangahue([-_.@].*)?\.service$/ {print "  " $1 "  " $2}' || true
else
  echo "  systemd is not available."
fi

echo
echo "Legacy MangaHue bot processes:"
pgrep -af '[/]mangahue-telegram|[M]angaHue.*python_bot[/]bot\.py' \
  | sed 's/^/  /' || true

if [[ "$MODE" != "execute" ]]; then
  echo
  echo "Inspection complete. No changes were made."
  exit 0
fi

if [[ -z "$EXPECTED_HOST" || "$EXPECTED_HOST" != "$HOST" ]]; then
  echo "Refusing cleanup: --confirm-host must exactly equal '$HOST'." >&2
  exit 3
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Execute mode must run as root." >&2
  exit 4
fi

echo
echo "Stopping known MangaHue systemd services..."
if command -v systemctl >/dev/null 2>&1; then
  mapfile -t SERVICES < <(
    systemctl list-unit-files --type=service --no-legend 2>/dev/null \
      | awk '$1 ~ /^mangahue([-_.@].*)?\.service$/ {print $1}'
  )
  for service in "${SERVICES[@]}"; do
    systemctl disable --now "$service"
  done
fi

echo "Stopping MangaHue Compose deployment..."
if command -v docker >/dev/null 2>&1 \
  && [[ -f "$APP_DIR/docker-compose.yml" ]]; then
  docker compose \
    --project-directory "$APP_DIR" \
    -f "$APP_DIR/docker-compose.yml" \
    --profile telegram \
    --profile telegram-small \
    down --remove-orphans
fi

echo "Terminating legacy standalone MangaHue bot processes..."
pkill -TERM -f '[/]mangahue-telegram' || true
pkill -TERM -f '[M]angaHue.*python_bot[/]bot\.py' || true
sleep 2

echo
echo "Cleanup complete. Persistent data and secrets were preserved for migration."
