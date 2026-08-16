# MangaHue Telegram Bot — Legacy Deploy & Operations Runbook

> **Retired production procedure:** this file is kept for historical and
> troubleshooting context only. Production deploys now use SHA-addressed
> immutable releases under `/opt/MangaHue-releases`, with
> `/opt/MangaHue-current` promoted only after CI gates pass. Follow
> `README-DEPLOY.md`; do not run the mutable-host deployment commands below.

Target host: **153.75.251.215** — 2 vCPU, ~5.4 GB RAM, Ubuntu (22.04/24.04).
Service: **`mangahue-pro.service`**. Binary: `/opt/MangaHue-release/bin/mangahue-telegram`.
Repo on the host: `/opt/MangaHue-release` (25-crate Cargo workspace).

> **Security note (read first):** the bot currently runs as **root** (`User=root`
> in the unit file, inherited from the old setup). This is accepted for now and
> is tracked as a hardening follow-up (dedicated system user, `chown` of
> `/var/lib/mangahue-pro`, `/opt/MangaHue-release/bin`, `runtime/telegram.env`,
> then `User=mangahue-pro` in the unit). Do not block a deploy on it — but do
> not forget it either.

**Inference is 100 % native Rust** (ONNX Runtime via the `ort` crate,
`load-dynamic`). No `MANGAHUE_PYTHON`, no ensemble workers, no Python in the
main path. The service uses the DDColor-FULL model
`/opt/MangaHue/models/ddcolor-full-fp16.onnx` (~435 MB FP16, native 512×512
transformer color decoder) with preset
`configs/engine.telegram-hq.yaml` (vibrant-natural grade: `target_colorfulness
1.55`, `muted_mid_sat 0.35`, `region_chroma_separation 50`) and style
`configs/styles/realistic_natural.yaml` (saturation 1.08, contrast 1.06).
Everything below verified against `configs/systemd/mangahue-pro.service` and
the workspace source. Rollback tier (manga-colorize + telegram-cpu + studio_anime)
is documented in the unit file's comments.

---

## 1. Pre-flight checklist

Run all of this **before** touching the running service. Order matters.

### 1.1 Disk, RAM, CPU

```sh
df -h /opt /var /run
free -m
nproc
```

- **Disk:** the release build needs roughly **2–4 GB** in `/opt/MangaHue-release/target/`
  for the two binaries below (`strip = "symbols"` keeps artifacts lean). A full
  25-crate workspace build needs more (~4–8 GB) and takes 20+ minutes — don't.
  Keep **≥ 5 GB free** on `/opt` before building.
- **RAM:** `cargo` + `rustc` during codegen can transiently use ~1–2 GB on this
  box. If `free -m` shows less than ~1.5 GB `available`, stop and close the
  other workload first.
- **`/run`:** the unit stages job scratch in `/run/mangahue-pro` (tmpfs, i.e.
  RAM-backed, capped at 768 MiB by `--scratch-limit-bytes 805306368`). Check
  `/run` has room, but the cap protects you.
- **CPU:** 2 cores. The build is single-binary, but it will still peg both
  cores — a quiet window helps (a colorize job during the build will be slow).

### 1.2 Build dependencies

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev cmake
```

`build-essential` (C linker) and `pkg-config`/`libssl-dev` are required by
transitive crates; `cmake` is listed because some native crates need it — if
the install fails on `cmake` alone, the build will tell you and it is safe to
retry without it.

### 1.3 Rust toolchain

```sh
command -v cargo && cargo --version && rustc --version
```

The repo pins `stable` via `rust-toolchain.toml`. If `cargo` is missing:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo --version
```

### 1.4 Get the code

```sh
cd /opt/MangaHue-release
git pull --ff-only
git --no-pager status
```

> Never run `git clean -fdx` here. Server-side artifacts (`bin/`,
> `runtime/telegram.env`) live in this tree and are **not** in `.gitignore`
> yet — `git clean` would delete them. Add `bin/` and `runtime/` to
> `.gitignore` as a follow-up.

If `/opt/MangaHue-release` is not a git checkout, rsync from the workstation
instead (exclude the heavy dirs — models stay under `/opt/MangaHue`):

```sh
rsync -av --delete \
  --exclude target --exclude .git --exclude models --exclude .venv \
  ./ root@153.75.251.215:/opt/MangaHue-release/
```

### 1.5 Verify model, config and env files

```sh
ls -lh /opt/MangaHue/models/ddcolor-full/ddcolor-full-fp16.onnx
ls -lh /opt/MangaHue/models/manga-onnx/manga-colorize-fp16.onnx
ls -lh /opt/MangaHue/models/ddcolor-tiny/ddcolor-tiny-fp16.onnx
ls -lh /opt/MangaHue-release/configs/engine.telegram-hq.yaml
ls -lh /opt/MangaHue-release/configs/styles/realistic_natural.yaml
ls -lh /opt/MangaHue-release/configs/model-contracts/manga-colorization-v2-onnx.json
ls -ld /opt/MangaHue-release/bin /opt/MangaHue-release/runtime
test -s /opt/MangaHue-release/runtime/telegram.env && echo "telegram.env OK" || echo "telegram.env MISSING"
```

- `ddcolor-full-fp16.onnx` — the **primary** model used by the service
  (~435 MB; verify the on-VPS size and that `ddcolor-full.metadata.json`
  sits next to it). If it is missing, ship it before starting the service.
- `manga-colorize-fp16.onnx` (~59 MB) — the rollback hint-U-Net used by the
  previous tier; keep it around for A/B benchmarks.
- `ddcolor-tiny-fp16.onnx` (~130 MB) also exists under `/opt/MangaHue/models/`
  — it is used by other presets/benchmarks, **not** by this service.
- `telegram.env` must exist and contain `TELEGRAM_BOT_TOKEN=...` (the unit
  loads it via `EnvironmentFile`). If it is missing, stop — the bot cannot
  start. Create it with `chmod 600`:
  ```sh
  printf 'TELEGRAM_BOT_TOKEN=%s\n' '<token-from-BotFather>' \
    | sudo tee /opt/MangaHue-release/runtime/telegram.env >/dev/null
  sudo chmod 600 /opt/MangaHue-release/runtime/telegram.env
  ```
- `bin/` does not exist in the repo — it is created on the server (§3.1).

---

## 2. Build

Build **only** the binaries you need. The workspace has 25 crates; a full
`cargo build --release` compiles everything and takes 20+ minutes on 2 cores.

```sh
cd /opt/MangaHue-release
cargo build --release --bin mangahue-telegram
```

That is the bot. Also build the ops CLI — it is needed for API token issuance
(`mangahue production bootstrap` / `production token`) and for offline
benchmarks (§6):

```sh
cargo build --release --bin mangahue
```

**Expected time:** first build of `mangahue-telegram` on 2 cores with
`lto = "thin"` + `codegen-units = 1`: roughly **15–25 minutes** (dependency
download included). `mangahue` shares most of the same dependency graph, so
the second build is a few minutes. Incremental rebuilds are much faster.
**Disk:** ~2–4 GB total in `target/`.

Verify both artifacts:

```sh
ls -lh target/release/mangahue-telegram target/release/mangahue
```

---

## 3. Install

### 3.1 Binary locations

```sh
install -d -m 0755 /opt/MangaHue-release/bin
install -m 0755 /opt/MangaHue-release/target/release/mangahue-telegram /opt/MangaHue-release/bin/mangahue-telegram
install -m 0755 /opt/MangaHue-release/target/release/mangahue /opt/MangaHue-release/bin/mangahue
ls -lh /opt/MangaHue-release/bin/
```

If a previous binary exists, keep it for rollback before overwriting (§8):

```sh
cp /opt/MangaHue-release/bin/mangahue-telegram /opt/MangaHue-release/bin/mangahue-telegram.bak
```

### 3.2 Configs and models

- **Configs** live in the release tree at `/opt/MangaHue-release/configs/`
  (`engine.telegram-cpu.yaml`, `styles/studio_anime.yaml`,
  `model-contracts/manga-colorization-v2-onnx.json`). `git pull` (§1.4) brings
  them; nothing else to do.
- **Models** live outside the release tree at `/opt/MangaHue/models/`
  (`manga-onnx/manga-colorize-fp16.onnx`, `ddcolor-tiny-fp16.onnx`). Verified
  in §1.5; if a file is missing, restore it from the rollback tarball (§8) or
  copy from the repo `models/` directory.
- **Web dirs:** none needed. The Telegram bot never reads `web/` (`web/app`,
  `web/studio`, `web/gpu` in the repo belong to the parked webapp, docs/17).
  Only `mangahue serve` (the webapp API, docs/17) needs `web/app/out`.
- **Work dir:** create the bot's state directory (the unit's
  `ReadWritePaths` allows exactly this):

```sh
install -d -m 0700 /var/lib/mangahue-pro /var/lib/mangahue-pro/telegram
```

---

## 4. Service install

### 4.1 Copy the unit file and enable

```sh
install -m 0644 /opt/MangaHue-release/configs/systemd/mangahue-pro.service /etc/systemd/system/mangahue-pro.service
cp /etc/systemd/system/mangahue-pro.service /etc/systemd/system/mangahue-pro.service.prev
systemctl daemon-reload
systemctl enable mangahue-pro
systemctl start mangahue-pro
systemctl status mangahue-pro --no-pager
```

### 4.2 Verify the ONNX Runtime library (do this before relying on the service)

The unit sets:

```
Environment=ORT_DYLIB_PATH=/opt/MangaHue/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so
```

Confirm that path actually exists on the host:

```sh
test -f /opt/MangaHue/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so && echo "ORT lib OK" || echo "ORT lib MISSING"
```

`ORT_DYLIB_PATH` can point **anywhere** readable — the loader only cares that
the file exists and is a valid `libonnxruntime.so`. The `ort` crate is pinned
to API 24 (ONNX Runtime ≥ 1.24), so the library must be at least that new.
Check the venv's version if you are in doubt:

```sh
/opt/MangaHue/.venv/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"
```

**Fallback A — system package:**

```sh
sudo apt-get update
sudo apt-get install -y libonnxruntime-dev
ldconfig -p | grep onnxruntime
```

Note: `libonnxruntime-dev` is **not** in the default Ubuntu 22.04/24.04
archives — this will usually fail. It is only an option if you have a PPA that
ships it.

**Fallback B — copy the `.so` from the venv (reliable, zero new packages):**

```sh
sudo cp /opt/MangaHue/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so /usr/local/lib/
sudo ldconfig
ldconfig -p | grep libonnxruntime
```

> **Trap (caused a production outage 2026-08-08):** never put the library in
> `/tmp` and symlink it into `/usr/local/lib`. The unit runs with
> `PrivateTmp=true`, which mounts a private empty `/tmp` — dlopen then fails
> (ENOENT on the symlink target) and ort 2.0.0-rc.12's `load-dynamic` error
> path deadlocks the process (pykeio/ort#560/#604). The real file must live
> in a permanent path (e.g. `/usr/local/lib`). The bot now pre-loads the
> library at startup (`pre-loaded ONNX Runtime` in the journal) and fails
> fast instead of hanging, but keep the library out of `/tmp` regardless —
> `/tmp` is also cleared on reboot.

If you use Fallback B, update the unit so it stops relying on the venv path:

```sh
sudo sed -i 's|^Environment=ORT_DYLIB_PATH=.*|Environment=ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so|' /etc/systemd/system/mangahue-pro.service
systemctl daemon-reload
systemctl restart mangahue-pro
```

(Or delete the `ORT_DYLIB_PATH` line entirely so the loader uses the default
search path.) The venv at `/opt/MangaHue/.venv` exists and provides the
library; it is also the home of the Python-side bits that are **not** loaded by
this service. Torch lives separately at `/opt/MangaHue-reference-runtime`
(still-Python reference-CGAN, not loaded by default).

### 4.3 What the unit does (context, no action needed)

- Runs `mangahue-telegram` with: `--work-dir /var/lib/mangahue-pro/telegram`,
  `--scratch-dir /run/mangahue-pro/jobs`, `--scratch-limit-bytes 805306368`,
  `--delivery-format jpeg --jpeg-quality 95`, `--max-pages 100`,
  `--max-concurrent-jobs 1`, `--batch-size 1`, `--daily-limit 300`, plus
  `--model /opt/MangaHue/models/ddcolor-full-fp16.onnx`,
  `--preset /opt/MangaHue-release/configs/engine.telegram-hq.yaml` and
  `--style /opt/MangaHue-release/configs/styles/realistic_natural.yaml`
  (rollback flags are in the unit file's comments).
- Threads pinned to 2 (`MANGAHUE_MODEL_THREADS=2`, `OMP_NUM_THREADS=2`,
  `MKL_NUM_THREADS=2`, `MALLOC_ARENA_MAX=2`).
- `Restart=on-failure` / `RestartSec=15`; cgroup memory backstop
  `MemoryHigh=4G`, `MemoryMax=5G`, `MemorySwapMax=512M`; sandboxing
  (`ProtectSystem=full`, `NoNewPrivileges`, `PrivateTmp`, `RuntimeDirectory`,
  `ReadWritePaths=/var/lib/mangahue-pro /run/mangahue-pro`).

---

## 5. First-run verification checklist

Do these in order after a deploy. Tick each box.

**1. Logs show a healthy startup**

```sh
journalctl -u mangahue-pro -n 100 --no-pager
```

Good — you should see lines like:

```
bot_id=123456789 username=YourBotName
scratch_dir=/run/mangahue-pro/jobs scratch_limit_bytes=805306368
ephemeral job workspace ready
```

Bad — stop here:

```
failed to load ONNX model ... libonnxruntime.so: cannot open shared object file
```

→ ORT resolution is broken; redo §4.2 (Fallback B is the least-effort fix).

```
poll error (will retry in 1s): <http error>
```

→ Telegram unreachable (outbound 443 blocked) or bad token in
`runtime/telegram.env`.

**2. `/status` via the bot**

In the chat: `/status` → expect
`No MangaHue job is active for this chat. Send a URL to start.`

**3. RAM before/after**

```sh
free -m
```

Record `free`/`available`. After the tests below, run `free -m` again and note
 the delta. Expect the idle bot at **~1–2 GB RSS** with DDColor-FULL loaded (no
torch); the previous manga-colorize tier idled near ~1 GB. While a job
runs, scratch staging adds up to ~768 MiB (tmpfs under `/run/mangahue-pro`).
The unit's `MemoryHigh=4G / MemoryMax=5G` backstop covers this.

**4. One real chapter colorize (timing)**

Send a real chapter URL (or a multi-page link the bot accepts). Expect
**20–70 s/page on 2 cores** for the previous manga-colorize tier; DDColor-FULL
runs at its native 512×512 so it is typically comparable or faster per page —
record whatever you actually see in the §6 baseline table. Watch live:

```sh
journalctl -u mangahue-pro -f
```

**5. Send a photo**

Send a photo as a direct image message; it must be colorized and delivered as
a JPEG page (`--delivery-format jpeg`, quality 95).

**6. Send a direct image URL**

Send a plain image URL as text; it must be ingested and colorized.

**7. Cancel a job**

Start a large job, then `/cancel` → expect
`Cancellation requested. MangaHue will stop at the next safe checkpoint.`
Check the job stops (`/status` → `No MangaHue job is active...`) and logs show
no further page deliveries.

---

## 6. Benchmark procedure

Use this before/after **any** performance-relevant change (model swap, ORT
upgrade, thread-count change, preset tweak) and record results below.

**Fixed page set.** Keep the same 3 pages forever (e.g. three real chapter
pages copied once):

```sh
mkdir -p /root/bench/pages /root/bench/out
# copy 3 fixed pages into /root/bench/pages ONCE (they must never change)
```

**Run.** From `/opt/MangaHue-release` (preset-relative paths resolve against
CWD), with the same model/preset/style and thread settings the service uses.

**A. Production tier — DDColor-FULL + HQ preset (this is what the service runs):**

```sh
export ORT_DYLIB_PATH=/opt/MangaHue/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so
export MANGAHUE_MODEL_THREADS=2
export OMP_NUM_THREADS=2
rm -rf /root/bench/out && mkdir -p /root/bench/out
cd /opt/MangaHue-release
time /opt/MangaHue-release/bin/mangahue benchmark /root/bench/pages \
  -o /root/bench/out \
  --model /opt/MangaHue/models/ddcolor-full-fp16.onnx \
  --preset /opt/MangaHue-release/configs/engine.telegram-hq.yaml \
  --style /opt/MangaHue-release/configs/styles/realistic_natural.yaml \
  --jobs 1 \
  --report /root/bench/report-full.json
cat /root/bench/report-full.json
```

**B. Rollback tier — manga-colorize + CPU preset (same pages):**

```sh
rm -rf /root/bench/out && mkdir -p /root/bench/out
cd /opt/MangaHue-release
time /opt/MangaHue-release/bin/mangahue benchmark /root/bench/pages \
  -o /root/bench/out \
  --model /opt/MangaHue/models/manga-onnx/manga-colorize-fp16.onnx \
  --preset /opt/MangaHue-release/configs/engine.telegram-cpu.yaml \
  --style /opt/MangaHue-release/configs/styles/studio_anime.yaml \
  --jobs 1 \
  --report /root/bench/report-small.json
cat /root/bench/report-small.json
```

- Run exactly **3 pages**, serial (`--jobs 1`), like the service
  (`--max-concurrent-jobs 1`).
- `time` gives total wall time; each `report.json` has the per-page breakdown
  plus the quality report (mean chroma / vivid pixels) — the A/B verdict
  (DDColor-FULL wins on color quality) comes from comparing those numbers and
  the rendered pages side by side.
- Do not run benchmarks while the bot is processing a job — 2 cores.
- The `export ORT_DYLIB_PATH=...` is only needed if you did not install the
  library system-wide (§4.2 Fallback B); the systemd service sets it itself.

**Baseline table — update every time you measure:**

| Date | Page set | Model | s/page (mean) | RAM peak (free -m delta) | Change being measured |
|---|---|---|---|---|---|
| 2026-08-08 | 3 fixed pages (test_images) | manga-colorize-fp16.onnx + telegram-cpu | **12.5** (37.6 s total) | **1.36 GB RSS** | native Rust baseline (rollback tier) |
| 2026-08-08 | 3 fixed pages (test_images) | ddcolor-full-fp16.onnx + telegram-hq | **17.4** (52.1 s total) | **3.45 GB RSS** | DDColor-FULL + HQ preset (production) |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

---

## 7. Operations

### 7.1 Log rotation (journald)

Cap the journal at 1 GB and restart journald:

```sh
sudo sed -i 's/^#SystemMaxUse=.*/SystemMaxUse=1G/' /etc/systemd/journald.conf
grep '^SystemMaxUse' /etc/systemd/journald.conf
sudo systemctl restart systemd-journald
```

Immediately reclaim space if the journal is already large:

```sh
sudo journalctl --vacuum-size=1G
```

### 7.2 Daily restart? No.

Do **not** schedule a daily restart. `Restart=on-failure` + `RestartSec=15`
already covers crashes, and a daily restart would kill in-flight jobs for no
benefit. The right tool is memory monitoring (§7.3): if RSS trends upward
job-over-job, investigate before it approaches `MemoryHigh=4G`.

### 7.3 Memory monitoring cron

Install the monitor script (appends a timestamped `free -m` + bot RSS
snapshot; RSS from `ps` is in KiB):

```sh
sudo tee /usr/local/sbin/mangahue-mem-mon.sh >/dev/null <<'EOF'
#!/bin/sh
# MangaHue bot memory snapshot (run by cron, see /etc/cron.d/mangahue-memory)
{
  echo "=== $(date -Is) ==="
  free -m
  ps -eo pid,rss,cmd | grep '[m]angahue-telegram'
} >> /var/log/mangahue-memory.log
EOF
sudo chmod 0755 /usr/local/sbin/mangahue-mem-mon.sh
```

Schedule it hourly (top of every hour, minute 5):

```sh
echo '5 * * * * root /usr/local/sbin/mangahue-mem-mon.sh' | sudo tee /etc/cron.d/mangahue-memory
```

Verify after one tick: `tail -n 20 /var/log/mangahue-memory.log`.

### 7.4 API token issuance (only if the webapp is ever re-enabled)

The webapp deploy is parked (docs/17), but when it resumes, tokens come from
the ops CLI you built in §2:

```sh
/opt/MangaHue-release/bin/mangahue production bootstrap \
  --database /var/lib/mangahue/production.db \
  --email ops@example.com \
  --name "MangaHue Admin" \
  --tier admin
```

→ prints `user_id=...` and `api_token=...` — the token is shown **once**, save
it. Subsequent tokens for users/operators:

```sh
/opt/MangaHue-release/bin/mangahue production token \
  --database /var/lib/mangahue/production.db \
  --user <user-id-or-email> \
  --label ops \
  --expires-days 30
```

See docs/17 §4.1 for the full webapp bootstrap flow.

### 7.5 Backup `/var/lib/mangahue-pro`

The bot's state (work dir, usage DB, per-chat state) lives here. Daily tar +
14-day retention via cron:

```sh
sudo tee /etc/cron.d/mangahue-pro-backup >/dev/null <<'EOF'
20 3 * * * root mkdir -p /var/backups/mangahue-pro && tar czf /var/backups/mangahue-pro/mangahue-pro-$(date +\%F).tar.gz -C /var/lib mangahue-pro && find /var/backups/mangahue-pro -name 'mangahue-pro-*.tar.gz' -mtime +14 -delete
EOF
```

Also snapshot the token file once (it is not in git — losing it means
re-issuing the token with BotFather):

```sh
sudo mkdir -p /root/backups
sudo cp -a /opt/MangaHue-release/runtime/telegram.env /root/backups/telegram.env
sudo chmod 600 /root/backups/telegram.env
```

Manual backup / restore:

```sh
sudo tar czf /var/backups/mangahue-pro/manual-$(date +%F-%H%M).tar.gz -C /var/lib mangahue-pro
# restore:
sudo tar xzf /var/backups/mangahue-pro/manual-<timestamp>.tar.gz -C /var/lib
sudo systemctl restart mangahue-pro
```

---

## 8. Rollback

Rollback = swap the binary and/or model and/or unit back. Keep the previous
artifacts **before** every change:

```sh
# previous binary (do this before overwriting, §3.1)
cp /opt/MangaHue-release/bin/mangahue-telegram /opt/MangaHue-release/bin/mangahue-telegram.bak

# previous unit file (do this before overwriting, §4.1)
cp /etc/systemd/system/mangahue-pro.service /etc/systemd/system/mangahue-pro.service.prev

# previous model tarball (keep one per model version; ~59 MB)
mkdir -p /var/backups/mangahue-pro/models
tar czf /var/backups/mangahue-pro/models/manga-colorize-fp16-$(date +%F).tar.gz \
  -C /opt/MangaHue/models/manga-onnx manga-colorize-fp16.onnx
```

**Roll back the binary:**

```sh
systemctl stop mangahue-pro
install -m 0755 /opt/MangaHue-release/bin/mangahue-telegram.bak /opt/MangaHue-release/bin/mangahue-telegram
systemctl start mangahue-pro
journalctl -u mangahue-pro -n 50 --no-pager
```

**Roll back the unit file:**

```sh
install -m 0644 /etc/systemd/system/mangahue-pro.service.prev /etc/systemd/system/mangahue-pro.service
systemctl daemon-reload
systemctl restart mangahue-pro
```

**Roll back the model:**

```sh
systemctl stop mangahue-pro
tar xzf /var/backups/mangahue-pro/models/manga-colorize-fp16-<date>.tar.gz \
  -C /opt/MangaHue/models/manga-onnx
systemctl start mangahue-pro
journalctl -u mangahue-pro -n 50 --no-pager
```

After any rollback, re-run §5 items 1–2 before telling users the bot is back.

---

## 9. Known limits & capacity

Be honest about this hardware: **2 cores / ~5.4 GB RAM**, CPU-only inference.

- **Concurrency:** max **1–2 concurrent jobs**. The unit runs
  `--max-concurrent-jobs 1` (default). Two concurrent jobs on two cores
  roughly double per-job latency, so stay at 1 unless throughput matters more
  than latency (see docs/17 SLO section).
- **Page time (fp16 hint-U-Net):** typical **20–70 s/page** on 2 cores
  depending on page size and load; the older fp16 benchmark measured
  **~133 s/page** worst case. Int8 export targets **40–60 s/page** once
  available; the mini-model roadmap (docs/16) targets 3–8 s/page.
- **Telegram flood limits:** the legacy Python bot paced sends with
  `MIN_SEND_INTERVAL=3.1 s`; the Rust bot has **no explicit send pacing**.
  Today that is fine — one document per page at 20–70 s spacing is far under
  Telegram's sustained per-chat limits. If page time ever drops to single
  digits (int8/mini model), add pacing before users start seeing HTTP 429
  flood errors.
- **Daily page limit:** `--daily-limit 0` = unlimited. Raise it when the bot
  goes public (`--daily-limit 20` is a sane start).
- **Reference-CGAN / torch:** still exists in the source tree and in
  `/opt/MangaHue-reference-runtime` (Python), but is **not loaded** by this
  preset. If a future preset re-enables Python backends, the RAM math below
  breaks — the old stack peaked at ~2.6 GB bot-side.
- **Runs as root** (see the note at the top of this document).

**RAM budget (~5.4 GB host):**

| Component | Budget |
|---|---|
| Bot RSS (single Rust process + libonnxruntime, no torch) | ~1 GB |
| Job scratch (tmpfs `/run/mangahue-pro/jobs`, capped) | up to 768 MiB while a job is active |
| OS + sshd + journald + misc | ~0.5–0.8 GB |
| Headroom / host reserve | ≥ 1 GB |
| **Peak with one active job** | **~2.5–2.8 GB — fits, but no slack for a second job + build at once** |

Backstops: cgroup `MemoryHigh=4G` / `MemoryMax=5G` / `MemorySwapMax=512M` in
the unit. Do not run a `cargo build --release` and a colorize job at the same
time on this box.

---

## 10. Next steps (roadmap links)

- **`docs/16_professional_mini_model_roadmap.md`** — the mini-model plan
  (≤ 35 MB int8 / ≤ 60 MB fp16, 3–8 s/page CPU target). **Optional**, high
  value: it would turn page time from minutes into seconds and shrink the RAM
  budget further. This is the recommended next project.
- **`docs/17_webapp_deploy_ready.md`** — the webapp deploy guide. **Parked.**
  If it is ever resumed: build `mangahue serve` + static `web/app/out`, then
  use §7.4 for token issuance.

Follow-ups tracked from this runbook: hardening to a non-root system user,
`bin/` + `runtime/` entries in `.gitignore`, send pacing if page time drops,
daily page limit before going public.
