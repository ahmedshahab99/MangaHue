#!/usr/bin/env bash
set -Eeuo pipefail

release_sha="${1:-}"
if [[ ! "$release_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid release SHA" >&2
  exit 2
fi

upload_dir=/home/mangahue-deploy/upload
archive="$upload_dir/mangahue-${release_sha}.tar.gz"
checksum="$upload_dir/mangahue-${release_sha}.sha256"
release_dir="/opt/MangaHue-releases/${release_sha}"

test -f "$archive"
test -f "$checksum"
cd "$upload_dir"
sha256sum -c "$(basename "$checksum")"
install -d -m 0755 "$release_dir" /opt/MangaHue-releases
tar -xzf "$archive" -C "$release_dir"
test "$(cat "$release_dir/REVISION")" = "$release_sha"
test -x "$release_dir/bin/mangahue-telegram"

if ! getent group mangahue-pro >/dev/null; then
  groupadd --system mangahue-pro
fi
if ! id mangahue-pro >/dev/null 2>&1; then
  useradd --system --gid mangahue-pro --home-dir /var/lib/mangahue-pro --shell /usr/sbin/nologin mangahue-pro
fi
install -d -o mangahue-pro -g mangahue-pro -m 0750 /var/lib/mangahue-pro
chown -R mangahue-pro:mangahue-pro /var/lib/mangahue-pro
install -d -m 0750 -o root -g mangahue-pro /etc/mangahue
if [[ ! -s /etc/mangahue/telegram.env ]]; then
  install -m 0640 -o root -g mangahue-pro /opt/MangaHue-release/runtime/telegram.env /etc/mangahue/telegram.env
else
  chown root:mangahue-pro /etc/mangahue/telegram.env
  chmod 0640 /etc/mangahue/telegram.env
fi

previous="$(readlink -f /opt/MangaHue-current 2>/dev/null || true)"
ln -sfn "$release_dir" /opt/MangaHue-current.next
mv -Tf /opt/MangaHue-current.next /opt/MangaHue-current
install -m 0644 "$release_dir/configs/systemd/mangahue-pro.service" /etc/systemd/system/mangahue-pro.service
systemctl daemon-reload
if ! systemctl restart mangahue-pro.service || ! systemctl is-active --quiet mangahue-pro.service; then
  if [[ -n "$previous" && -d "$previous" ]]; then
    ln -sfn "$previous" /opt/MangaHue-current.next
    mv -Tf /opt/MangaHue-current.next /opt/MangaHue-current
    systemctl restart mangahue-pro.service
  fi
  exit 1
fi
test "$(readlink -f /opt/MangaHue-current)" = "$release_dir"
rm -f "$archive" "$checksum"
echo "promoted $release_sha"
