#!/bin/bash
# MangaHue production healthcheck — runs hourly via mangahue-healthcheck.timer.
# Logs service/memory/disk status; restarts the bot if it died or hung
# (systemd Restart=on-failure already covers crashes; this covers a process
# that is alive but wedged, e.g. stuck on a stalled download).
SERVICE=mangahue-pro
LOG_TAG=mangahue-health

if ! systemctl is-active --quiet "$SERVICE"; then
    logger -t "$LOG_TAG" "CRITICAL: $SERVICE not active"
    systemctl restart "$SERVICE"
    sleep 3
    if systemctl is-active --quiet "$SERVICE"; then
        logger -t "$LOG_TAG" "RESTARTED $SERVICE ok"
    else
        logger -t "$LOG_TAG" "RESTART FAILED for $SERVICE"
    fi
    exit 0
fi

# Wedged check: the bot polls Telegram every ~30s; if it has not written a
# log line in 5 minutes it is hung. (journald timestamps are UTC.)
LAST=$(journalctl -u "$SERVICE" --since '5 min ago' --no-pager 2>/dev/null | wc -l)
if [ "$LAST" -eq 0 ]; then
    logger -t "$LOG_TAG" "WARN: $SERVICE silent for 5+ min, restarting"
    systemctl restart "$SERVICE"
fi

MEM=$(free -m | awk '/Mem:/{printf "%d", $3*100/$2}')
DISK=$(df / | awk 'NR==2{gsub("%","",$5); print $5}')
BOT_PID=$(systemctl show -p MainPID --value "$SERVICE")
logger -t "$LOG_TAG" "ok pid=${BOT_PID} mem=${MEM}% disk=${DISK}%"
