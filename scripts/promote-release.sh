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
staging_dir="/opt/MangaHue-releases/.${release_sha}.staging.$$"
unit_path=/etc/systemd/system/mangahue-pro.service
unit_backup="$(mktemp /tmp/mangahue-pro.service.XXXXXX)"
had_unit=false

cleanup() {
  rm -rf -- "$staging_dir"
  rm -f -- "$unit_backup"
}
trap cleanup EXIT

test -f "$archive"
test -f "$checksum"
cd "$upload_dir"
sha256sum -c "$(basename "$checksum")"
if [[ -e "$release_dir" ]]; then
  echo "immutable release already exists: $release_dir" >&2
  exit 3
fi
install -d -m 0755 /opt/MangaHue-releases
install -d -m 0755 "$staging_dir"
tar -xzf "$archive" -C "$staging_dir"
test "$(cat "$staging_dir/REVISION")" = "$release_sha"
test -x "$staging_dir/bin/mangahue-telegram"
test -r "$staging_dir/configs/systemd/mangahue-pro.service"
chmod -R a-w "$staging_dir"
mv "$staging_dir" "$release_dir"

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
  echo "missing required /etc/mangahue/telegram.env; refusing legacy mutable-release fallback" >&2
  exit 4
else
  chown root:mangahue-pro /etc/mangahue/telegram.env
  chmod 0640 /etc/mangahue/telegram.env
fi

previous="$(readlink -f /opt/MangaHue-current 2>/dev/null || true)"
if [[ -f "$unit_path" ]]; then
  cp -a "$unit_path" "$unit_backup"
  had_unit=true
fi
ln -sfn "$release_dir" /opt/MangaHue-current.next
mv -Tf /opt/MangaHue-current.next /opt/MangaHue-current
install -m 0644 "$release_dir/configs/systemd/mangahue-pro.service" "$unit_path"
systemctl daemon-reload
if ! systemctl restart mangahue-pro.service || ! sleep 10 || ! systemctl is-active --quiet mangahue-pro.service; then
  if [[ -n "$previous" && -d "$previous" ]]; then
    ln -sfn "$previous" /opt/MangaHue-current.next
    mv -Tf /opt/MangaHue-current.next /opt/MangaHue-current
  fi
  if [[ "$had_unit" == true ]]; then
    install -m 0644 "$unit_backup" "$unit_path"
  else
    rm -f -- "$unit_path"
  fi
  systemctl daemon-reload
  systemctl restart mangahue-pro.service
  exit 1
fi
test "$(readlink -f /opt/MangaHue-current)" = "$release_dir"
rm -f "$archive" "$checksum"
trap - EXIT
cleanup
echo "promoted $release_sha"
