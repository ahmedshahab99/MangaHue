#!/bin/bash
TOKEN=$(grep -oP '(?<=TOKEN=).*|(?<=BOT_TOKEN=).*|(?<=TELEGRAM_BOT_TOKEN=).*' /opt/MangaHue-release/runtime/telegram.env | head -1)
echo "=== API health now (3 tries) ==="
for i in 1 2 3; do
  curl -s -m 15 -o /dev/null -w "try $i: HTTP %{http_code} in %{time_total}s\n" "https://api.telegram.org/bot$TOKEN/getMe" || echo "try $i: FAILED"
  sleep 2
done
echo "=== bot poll activity in last 90s (should be silent if healthy) ==="
journalctl -u mangahue-pro.service --since '90 sec ago' --no-pager 2>/dev/null | tail -3
echo "(end)"
echo "=== process still alive ==="
ps -p 1092583 -o pid,etime,rss,cmd --no-headers 2>/dev/null | head -c 200
