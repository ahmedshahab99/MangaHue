# MangaHue Web App — Production Deployment (VPS)

Target host: **153.75.251.215** — 2 vCPU, ~5.4 GB RAM, Ubuntu (22.04/24.04).

This document deploys the MangaHue web app (Next.js static export) behind the
Rust API (`mangahue-api`, public mode) with an nginx HTTPS front. The Telegram
bot runs separately under `mangahue-pro.service` and is not affected by this
guide.

**Be honest about the hardware.** On this 2-core/5.4 GB VPS:

- Inference is CPU-only and takes roughly **20–70 s per page** with the tiny
  FP16 model (`ddcolor-tiny-fp16.onnx`), depending on page size and system load.
- The API supports **1–2 concurrent jobs** (`MANGAHUE_MAX_CONCURRENT_JOBS`).
  Two concurrent jobs on two cores will roughly double per-job latency, so
  default to 1 and raise to 2 only if throughput matters more than latency.
- The page-time SLO is therefore scoped as follows: control-plane latency
  (upload, job submit, status poll, download) is p95 **< 10 s**; full page
  colorization is a long-running async job with a realistic p95 target of
  **< 120 s/page** on this box.

---

## 1. Architecture

```
                        +----------------------------------------------------+
                        |                    VPS 153.75.251.215               |
                        |                    2 vCPU / 5.4 GB RAM              |
                        |                                                    |
  Browser  -----------> |  nginx :443 (TLS, Let's Encrypt)                   |
   (HTTPS)              |    |  proxy_pass http://127.0.0.1:8080             |
                        |    v                                               |
                        |  mangahue-api  (Rust, public mode, 127.0.0.1:8080) |
                        |    |                                                |
                        |    |-- /app       static webapp (web/app/out)      |
                        |    |-- /models    ONNX files (130 MB tiny model)   |
                        |    |-- /api/*     JSON endpoints (Bearer auth)     |
                        |    |-- /health    liveness                          |
                        |    v                                                |
                        |  Python sidecar  (/opt/mangahue/.venv/bin/python)  |
                        |    `python -m mangahue_train.run_onnx_tile`        |
                        |    ONNX Runtime CPU inference on the same host     |
                        |                                                    |
                        |  SQLite production DB  /var/lib/mangahue/production.db
                        |  Jobs workspace       /var/lib/mangahue/jobs        |
                        +----------------------------------------------------+
```

Route map of `mangahue-api` (verified in `crates/mangahue-api/src/lib.rs`):

| Path | Visibility in public mode |
|---|---|
| `GET /health` | Public (no auth) |
| `/app`, `/operator`, `/gpu`, `/models` | Public static assets |
| `/api/v1/jobs`, `/api/v1/quota` | Bearer token (durable job queue) |
| `/api/styles`, `/api/upload`, `/api/colorize`, `/api/jobs/*`, `/api/jobs/:id/events` (SSE), `/api/file` | Bearer token (transient in-process jobs) |
| `/api/doctor`, `/metrics`, `/api/projects/*`, brain/correction routes | Bearer token + **admin** tier only |

The webapp at `/app` is the primary user flow; `/gpu` (WebGPU studio) is
parked, and `/operator` is the internal operator studio — both are served but
not part of the user-facing deployment.

---

## 2. VPS prerequisites

### 2.1 System packages and swap

The full Rust workspace builds only on Linux. Install build dependencies, then
size swap: the box has 5.4 GB RAM and the Python sidecar plus Rust process can
spike near 4 GB during inference, so give it a 4 GB swap file (roughly
0.75–1x RAM; the bot unit already assumes swap exists).

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential pkg-config libssl-dev cmake \
    curl git sqlite3 python3 python3-venv python3-pip nginx \
    ufw fail2ban

# 4 GB swap (falls back to 2 GB if disk is tight)
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
sudo sysctl vm.swappiness=60
echo 'vm.swappiness=60' | sudo tee /etc/sysctl.d/99-mangahue.conf
```

Install Rust via rustup (the workspace pins `stable` in `rust-toolchain.toml`):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
```

### 2.2 Service user and directory layout

```bash
sudo useradd --system --home /opt/mangahue --shell /usr/sbin/nologin mangahue
sudo mkdir -p /opt/mangahue/bin /opt/mangahue/models /opt/mangahue/configs/styles \
             /opt/mangahue/web/app/out /opt/mangahue/python \
             /var/lib/mangahue/jobs /var/lib/mangahue/uploads
sudo chown -R mangahue:mangahue /opt/mangahue /var/lib/mangahue
```

### 2.3 Python venv for the inference sidecar

The Rust engine spawns the ONNX worker through the interpreter named by
`MANGAHUE_PYTHON` and expects the `mangahue_train` package importable via
`PYTHONPATH`. Only the inference path is needed in production:

```bash
sudo -u mangahue python3 -m venv /opt/mangahue/.venv
sudo -u mangahue /opt/mangahue/.venv/bin/pip install --upgrade pip
sudo -u mangahue /opt/mangahue/.venv/bin/pip install numpy onnxruntime pillow
```

> The full `python/pyproject.toml` (`torch`, `opencv`, `tqdm`, ...) is only
> needed for training/export on a workstation. Installing `torch` here would
> waste ~2 GB of the 5.4 GB budget — don't.

Copy the sidecar package source into place and verify the worker imports:

```bash
sudo rsync -a python/ /opt/mangahue/python/
sudo -u mangahue env PYTHONPATH=/opt/mangahue/python \
  /opt/mangahue/.venv/bin/python -c "import numpy, onnxruntime, PIL; print('sidecar ok')"
```

### 2.4 ONNX models

`models/` is **not in git** (see `.gitignore`: `*.onnx`, `models/pretrained/`).
Models ship as build artifacts. Current artifact sizes in the repo:

| File | Size |
|---|---|
| `models/ddcolor-tiny-fp16.onnx` | ~130 MB (default model) |
| `models/manga-colorize-fp16.onnx` | ~61 MB |
| `models/ddcolor-full-fp16.onnx` | ~456 MB (too heavy for this VPS) |
| `models/ddcolor-tiny.metadata.json` | sidecar metadata |

Place the tiny model + its metadata on the VPS and verify checksums:

```bash
# from your workstation or CI artifact store:
scp models/ddcolor-tiny-fp16.onnx models/ddcolor-tiny.metadata.json \
    user@153.75.251.215:/tmp/models/
ssh user@153.75.251.215
sudo cp /tmp/models/* /opt/mangahue/models/
sudo chown mangahue:mangahue /opt/mangahue/models/*
sha256sum /opt/mangahue/models/ddcolor-tiny-fp16.onnx   # compare with artifact manifest
```

### 2.5 Engine configs and styles

The default config (`configs/engine.studio.yaml`) and `configs/styles/` are
read at job time; copy them with the release:

```bash
sudo rsync -a configs/engine.studio.yaml configs/styles/ /opt/mangahue/configs/
sudo chown -R mangahue:mangahue /opt/mangahue/configs
```

---

## 3. Build & install

Build **only the needed binaries** on the VPS to avoid a 22-minute full
workspace build. Two binaries are required:

- `mangahue-api` — the runtime web API.
- `mangahue` — the ops CLI (`production bootstrap`, `production token`,
  `production backup`, `doctor`). Token issuance and backups live here, so it
  is part of the "needed" set even though it is not the long-running process.

```bash
git clone <your-repo-url> ~/mangahue-src
cd ~/mangahue-src
cargo build --release --bin mangahue-api --bin mangahue
```

(First build still compiles all transitive workspace crates once — expect
~15–22 min on 2 cores. Subsequent rebuilds with a warm `target/` are minutes.
Never run `cargo build --release` without `--bin` on this box.)

Install:

```bash
sudo cp target/release/mangahue-api target/release/mangahue /opt/mangahue/bin/
sudo chmod 755 /opt/mangahue/bin/mangahue-api /opt/mangahue/bin/mangahue
/opt/mangahue/bin/mangahue-api --help   # sanity check
/opt/mangahue/bin/mangahue serve --help # same server via the CLI, if preferred
```

### 3.1 Webapp static export

Build on any machine (or the VPS if Node is installed) and copy `web/app/out`:

```bash
cd web/app
npm ci
npm run build        # Next.js static export -> web/app/out (gitignored)
cd ../..
sudo rsync -a --delete web/app/out/ /opt/mangahue/web/app/out/
sudo chown -R mangahue:mangahue /opt/mangahue/web
```

### 3.2 Final layout

```
/opt/mangahue/
  bin/mangahue-api, bin/mangahue
  models/ddcolor-full-fp16.onnx, ddcolor-full.metadata.json   # studio tier
  models/ddcolor-tiny-fp16.onnx, ddcolor-tiny.metadata.json   # fast tier
  web/app/out/            # static webapp
  configs/engine.studio.yaml, configs/styles/
  python/                 # mangahue_train package
  .venv/                  # sidecar interpreter (numpy, onnxruntime, pillow)
/var/lib/mangahue/
  jobs/                   # job workspace (uploads, results, job JSON)
  production.db           # users + tokens + quotas (SQLite, WAL mode)
```

---

## 4. systemd unit (`mangahue-api`, public mode)

Create `/etc/systemd/system/mangahue-api.service`:

```ini
[Unit]
Description=MangaHue Engine Web API (public mode)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mangahue
Group=mangahue
WorkingDirectory=/opt/mangahue
Environment=MANGAHUE_PYTHON=/opt/mangahue/.venv/bin/python
Environment=PYTHONPATH=/opt/mangahue/python
Environment=MANGAHUE_DEFAULT_MODEL=/opt/mangahue/models/ddcolor-tiny-fp16.onnx
Environment=MANGAHUE_DEFAULT_CONFIG=/opt/mangahue/configs/engine.studio.yaml
Environment=MANGAHUE_MAX_CONCURRENT_JOBS=1
Environment=MANGAHUE_MODEL_THREADS=2
Environment=MANGAHUE_JOB_RETENTION_HOURS=168
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=MALLOC_ARENA_MAX=2
Environment=RUST_LOG=mangahue=info,tower_http=info
ExecStart=/opt/mangahue/bin/mangahue-api \
    --addr 127.0.0.1:8080 \
    --public-api \
    --jobs-dir /var/lib/mangahue/jobs \
    --styles-dir /opt/mangahue/configs/styles \
    --studio-dir /opt/mangahue/web/app/out \
    --production-db /var/lib/mangahue/production.db \
    --default-model /opt/mangahue/models/ddcolor-tiny-fp16.onnx \
    --default-config /opt/mangahue/configs/engine.studio.yaml \
    --cors-origin https://<domain>
Restart=on-failure
RestartSec=15
TimeoutStopSec=30
KillSignal=SIGTERM
LimitNOFILE=65536
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/var/lib/mangahue
MemoryHigh=3G
MemoryMax=4G
MemorySwapMax=1G

[Install]
WantedBy=multi-user.target
```

Notes on the settings:

- `--public-api` disables local file/operator surfaces, forces Bearer auth on
  all `/api/*` routes, and confines job paths to the jobs workspace
  (`sandbox_jobs = true`).
- `--cors-origin` is repeatable; loopback is always allowed. Add one flag per
  production origin (e.g. `https://<domain>` and `https://www.<domain>` if both
  are used).
- `MANGAHUE_JOB_RETENTION_HOURS=168` matches the CLI default; the API sweeps
  expired transient jobs automatically.
- `OMP_NUM_THREADS=2` / `MANGAHUE_MODEL_THREADS=2` match the 2 vCPU cores;
  `MALLOC_ARENA_MAX=2` caps glibc arena growth to keep RSS stable.
- Memory policy for 5.4 GB total: `MemoryHigh=3G` throttles under pressure,
  `MemoryMax=4G` hard-caps (swap-backed), leaving ~1.4 GB for nginx, sshd,
  cron, and the OS. If you raise `MANGAHUE_MAX_CONCURRENT_JOBS` to 2, expect
  ~2 x 1.5 GB peak from two sidecars — stay within the cap or reduce threads.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mangahue-api
sudo systemctl status mangahue-api --no-pager
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
```

### 4.1 Bootstrap the production DB and issue the first token

```bash
sudo -u mangahue /opt/mangahue/bin/mangahue production bootstrap \
    --database /var/lib/mangahue/production.db \
    --email ops@<domain> \
    --name "MangaHue Admin" \
    --tier admin
# prints: user_id=... and api_token=...  -> save BOTH securely (token shown once)

sudo -u mangahue /opt/mangahue/bin/mangahue production token \
    --database /var/lib/mangahue/production.db \
    --user <user-id> \
    --label webapp \
    --expires-days 365
```

The webapp users get tokens through the same `production token` flow
(see Section 10 for the issuance workflow).

---

## 5. nginx reverse proxy

Create `/etc/nginx/sites-available/mangahue`:

```nginx
# HTTP -> HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name <domain>;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name <domain>;

    ssl_certificate     /etc/letsencrypt/live/<domain>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<domain>/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    client_max_body_size 64M;

    # gzip for static assets (the Rust API serves /app as static files)
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/javascript application/json
               image/svg+xml application/wasm;

    # WebSocket / SSE: do not buffer the job event stream
    map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
    }

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # SSE event stream for job progress: no buffering, long timeout
    location ~ ^/api/jobs/[^/]+/events$ {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

> Note: the `map` directive must live in the `http{}` context, so put it in
> `/etc/nginx/nginx.conf` (or an included file) if nginx refuses it inside the
> server block. On Ubuntu it is safe in this file only if included from the
> http block — the standard layout is `conf.d/`; simplest is to add the `map`
> to `/etc/nginx/nginx.conf` before the `include` lines.

Enable and test:

```bash
sudo ln -s /etc/nginx/sites-available/mangahue /etc/nginx/sites-enabled/mangahue
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6. TLS with Let's Encrypt (certbot)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <domain> --redirect --agree-tos -m ops@<domain>
```

The `--redirect` flag rewrites the port-80 server block to the 301 above.
Auto-renewal is a systemd timer installed by the package:

```bash
sudo systemctl enable --now certbot.timer
sudo certbot renew --dry-run   # verify renewal works
```

`certbot --nginx` already reloads nginx after each successful renewal; if you
later issue certs with `certbot certonly`, add a deploy hook instead:
`certbot renew --deploy-hook "systemctl reload nginx"`.

---

## 7. Operations

### 7.1 Backup the production DB (cron)

The DB runs in WAL mode, so use the online-safe `sqlite3 .backup` (a consistent
snapshot even while the API is writing). Install this cron as root:

```bash
sudo crontab -e
```

```cron
# Daily 03:10 UTC: online backup, keep 30 days
10 3 * * * mkdir -p /var/backups/mangahue && sqlite3 /var/lib/mangahue/production.db ".backup '/var/backups/mangahue/production-$(date +\%F).db'" && find /var/backups/mangahue -name 'production-*.db' -mtime +30 -delete && find /var/backups/mangahue -name 'production-*.db' -type f -exec chmod 600 {} \;
```

Test the backup command manually before relying on it:

```bash
sudo mkdir -p /var/backups/mangahue
sudo sqlite3 /var/lib/mangahue/production.db ".backup '/var/backups/mangahue/production-test.db'"
sudo sqlite3 /var/backups/mangahue/production-test.db "SELECT count(*) FROM users;"
```

The CLI alternative (same mechanism, no sqlite3 dependency) is
`mangahue production backup --database /var/lib/mangahue/production.db
--output /var/backups/mangahue/production-$(date +%F).db`.

### 7.2 Job retention

- Transient job files: `MANGAHUE_JOB_RETENTION_HOURS` (default 168 = 7 days)
  in the systemd unit. Set to 0 to disable pruning; lower it if `/var/lib/mangahue`
  grows too fast.
- Durable queue artifacts: run the sweep on demand or via cron:
  `mangahue production retention --database /var/lib/mangahue/production.db
  --object-store /var/lib/mangahue/objects --limit 1000`.

### 7.3 Log rotation (journald)

Cap journal disk usage at 1 GB:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
sudo tee /etc/systemd/journald.conf.d/mangahue.conf > /dev/null <<'EOF'
[Journal]
SystemMaxUse=1G
MaxRetentionSec=30d
EOF
sudo systemctl restart systemd-journald
sudo journalctl --vacuum-size=1G
```

### 7.4 Monitoring

Health check cron (root), restarting the service if `/health` fails twice in a
row and logging the outage:

```bash
sudo tee /usr/local/sbin/mangahue-healthcheck.sh > /dev/null <<'EOF'
#!/bin/sh
if curl -fsS --max-time 15 http://127.0.0.1:8080/health > /dev/null; then
    echo "ok" > /var/lib/mangahue/health.state
    exit 0
fi
if [ -f /var/lib/mangahue/health.state ] && [ "$(cat /var/lib/mangahue/health.state)" = "ok" ]; then
    echo "down" > /var/lib/mangahue/health.state
    echo "$(date -Is) mangahue-api health check failed" >> /var/log/mangahue-health.log
    systemctl restart mangahue-api
fi
EOF
sudo chmod 755 /usr/local/sbin/mangahue-healthcheck.sh

sudo crontab -l > /tmp/mangahue-cron 2>/dev/null
echo "*/5 * * * * /usr/local/sbin/mangahue-healthcheck.sh" >> /tmp/mangahue-cron
sudo crontab /tmp/mangahue-cron
```

UptimeRobot-style external check: create an **HTTP(S)** monitor on
`https://<domain>/health` with a 30 s interval and 2 consecutive failure
threshold, plus a keyword alert for `"status": "ok"`.

### 7.5 SLO and capacity expectations

| Metric | Target | Reality on this VPS |
|---|---|---|
| Upload / job submit / poll / download (p95) | < 10 s | Achievable; these are cheap HTTP ops |
| Full page colorization (p95) | < 120 s | Typical 20–70 s/page; 2 concurrent jobs can push toward the upper end |
| Concurrent jobs | 1–2 | 1 by default; 2 only if latency is acceptable |
| Availability | 99.9% | Health check + `Restart=on-failure` covers restarts; model swap needs a maintenance window |

---

## 8. Release checklist

Run these in order on each release. Steps 1–4 can happen on a build machine;
steps 5+ happen on the VPS.

1. **Build the binaries** (on the VPS or a Linux builder with the repo):
   ```bash
   cd ~/mangahue-src && git pull --ff-only
   cargo build --release --bin mangahue-api --bin mangahue
   ```
2. **Run the API doctor** against the staged layout (it colorizes a sample
   page end-to-end, so this doubles as a pre-deploy inference check):
   ```bash
   sudo cp -r examples/input /opt/mangahue/examples/input   # stage smoke_page.png once
   sudo chown -R mangahue:mangahue /opt/mangahue/examples

   MANGAHUE_PYTHON=/opt/mangahue/.venv/bin/python \
   PYTHONPATH=/opt/mangahue/python \
   /opt/mangahue/bin/mangahue doctor \
       --config /opt/mangahue/configs/engine.studio.yaml \
       --styles-dir /opt/mangahue/configs/styles \
       --metadata /opt/mangahue/models/ddcolor-tiny.metadata.json \
       --jobs-dir /var/lib/mangahue/jobs \
       --studio-dir /opt/mangahue/web/app/out \
       --sample-input /opt/mangahue/examples/input/smoke_page.png
   ```
   Fix anything the report flags before continuing.
3. **Stage new artifacts**: copy `mangahue-api`, `mangahue`, and
   `web/app/out/` to a release dir (see Section 9 for the rollback layout).
4. **Smoke test with a token** (against the staging port or after swap;
   here shown against the live URL):
   ```bash
   TOKEN=<user-api-token>   # from `mangahue production token`

   # 4a. Upload a page
   curl -fsS -H "Authorization: Bearer $TOKEN" \
        -F "file=@test_page.jpg" https://<domain>/api/upload
   # -> {"id":"<uuid>.jpg","path":".../uploads/<user-id>/<uuid>.jpg",...,"kind":"page"}

   # 4b. Submit a colorize job (input_path is relative to the jobs workspace)
   curl -fsS -X POST -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"request":{"input_path":"uploads/<user-id>/<uuid>.jpg"}}' \
        https://<domain>/api/colorize
   # -> {"id":"<job-id>","status":"queued",...}

   # 4c. Poll until completed (SSE alternative: /api/jobs/<job-id>/events)
   curl -fsS -H "Authorization: Bearer $TOKEN" https://<domain>/api/jobs/<job-id>
   # -> {"status":"completed","output_path":"/var/lib/mangahue/jobs/out/<job-id>.png",...}

   # 4d. Download the result (public mode requires job_id)
   curl -fsS -H "Authorization: Bearer $TOKEN" \
        "https://<domain>/api/file?job_id=<job-id>&path=/var/lib/mangahue/jobs/out/<job-id>.png" \
        -o result.png
   file result.png   # must be a valid PNG
   ```
5. **Swap the binary and webapp**:
   ```bash
   sudo cp ~/release/mangahue-api /opt/mangahue/bin/mangahue-api
   sudo cp ~/release/mangahue     /opt/mangahue/bin/mangahue
   sudo rsync -a --delete ~/release/out/ /opt/mangahue/web/app/out/
   sudo systemctl restart mangahue-api
   ```
6. **Verify**:
   ```bash
   curl -fsS http://127.0.0.1:8080/health | python3 -m json.tool   # status ok
   curl -fsSI https://<domain>/app | head -n 5                     # 200, text/html
   curl -fsS -o /dev/null -w '%{http_code}\n' https://<domain>/models/ddcolor-tiny-fp16.onnx  # 200
   curl -fsS -H "Authorization: Bearer $TOKEN" https://<domain>/api/v1/quota   # quota JSON
   ```
7. If step 4 was done pre-swap, repeat a one-page colorize end-to-end against
   the live URL, then confirm `journalctl -u mangahue-api --since "-10 minutes"`
   shows a clean start and no worker errors.

---

## 9. Rollback plan

Keep the previous release fully restorable:

```bash
sudo mkdir -p /opt/mangahue/releases/prev-$(date +%F)
sudo cp /opt/mangahue/bin/mangahue-api /opt/mangahue/releases/prev-$(date +%F)/
sudo cp /opt/mangahue/bin/mangahue     /opt/mangahue/releases/prev-$(date +%F)/
sudo rsync -a /opt/mangahue/web/app/out/ /opt/mangahue/releases/prev-$(date +%F)/out/
sudo tar czf /opt/mangahue/releases/prev-$(date +%F).tgz /opt/mangahue/models
```

Also snapshot the production DB before any release that changes schema or
token logic (the daily cron backup covers this, but take an explicit one):

```bash
sudo sqlite3 /var/lib/mangahue/production.db \
  ".backup '/var/backups/mangahue/pre-release-$(date +%F).db'"
```

**Restore steps** (models rarely change; if they do, restore from the tarball):

```bash
# 1. Restore the previous binary + webapp
sudo cp /opt/mangahue/releases/prev-<date>/mangahue-api /opt/mangahue/bin/mangahue-api
sudo cp /opt/mangahue/releases/prev-<date>/mangahue     /opt/mangahue/bin/mangahue
sudo rsync -a --delete /opt/mangahue/releases/prev-<date>/out/ /opt/mangahue/web/app/out/

# 2. Restore models if they were part of the release
sudo tar xzf /opt/mangahue/releases/prev-<date>.tgz -C /opt/mangahue --strip-components=2

# 3. Restore the DB snapshot if the release wrote schema/token data
sudo cp /var/backups/mangahue/pre-release-<date>.db /var/lib/mangahue/production.db

# 4. Restart and verify
sudo systemctl restart mangahue-api
curl -fsS http://127.0.0.1:8080/health
curl -fsSI https://<domain>/app | head -n 5
```

Because `/var/lib/mangahue/jobs` is write-through data, do **not** restore it
during rollback unless a release corrupted outputs; in-flight jobs from the
bad release may be lost, which is acceptable.

---

## 10. Security notes

### 10.1 Firewall (ufw)

Allow only SSH and web traffic:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status verbose
```

### 10.2 SSH hardening

```bash
sudo ssh-keygen -t ed25519 -b 256 -f /root/.ssh/id_ed25519 -N ""   # or use an existing key
# append your public key to ~/.ssh/authorized_keys, then:
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

Keep `fail2ban` enabled (installed in Section 2) with defaults; it protects
`sshd` and can watch nginx logs:

```bash
sudo systemctl enable --now fail2ban
```

### 10.3 Token issuance workflow for users

1. Operator creates the account with `mangahue production bootstrap` (tier
   `admin` for operators) or reuses it for the first user.
2. For each user:
   ```bash
   sudo -u mangahue /opt/mangahue/bin/mangahue production token \
       --database /var/lib/mangahue/production.db \
       --user <user-id> --label <name-or-device> --expires-days 90
   ```
   The token prints once; deliver it over a secure channel. Tokens are hashed
   at rest (issued via `issue_token`, checked by `authenticate`), so a DB
   leak does not expose usable tokens.
3. Rotate/revoke on request — revocation support exists in the store
   (`revoke_token`); revoke by deleting the token row or issuing a fresh
   `--expires-days` short token for incident response.

### 10.4 Rate limits and quotas (enforced in the API)

`authenticated_user` enforces per-tier RPM via `enforce_rate_limit`:

| Tier | RPM | Daily pages | Daily jobs | Max pages/job | Max concurrent jobs |
|---|---|---|---|---|---|
| Free | 30 | 20 | 5 | 8 | 1 |
| Creator | 120 | (tier policy) | (tier policy) | (tier policy) | (tier policy) |
| Studio | 600 | (tier policy) | (tier policy) | (tier policy) | (tier policy) |
| Admin | 10,000 | — | — | — | — |

Additional enforcement: `enforce_job_quota` rejects jobs beyond the per-tier
daily page/job limits, and `MANGAHUE_MAX_CONCURRENT_JOBS` caps simultaneous
jobs globally (excess gets 429 "concurrent job quota exhausted").

### 10.5 CORS

Loopback is always allowed. In public mode, production origins must be listed
explicitly with `--cors-origin https://<domain>` (repeatable). Do not add
`*`; the browser will refuse to call the API from any origin not in the
allowlist.

### 10.6 Other hardenings in place

- `--public-api` sandboxes job paths to the jobs workspace; non-admin tokens
  cannot reference absolute paths or escape the workspace
  (`sanitize_job_payload`).
- `/api/doctor` and operator routes require the admin tier.
- Uploads are validated by magic bytes, capped at 250 MB (API) and 64 MB
  (nginx `client_max_body_size`).
- The systemd unit runs as the unprivileged `mangahue` user with
  `NoNewPrivileges`, `ProtectSystem=full`, `ProtectHome`, and write access
  limited to `/var/lib/mangahue`.
