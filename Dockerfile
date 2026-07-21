# syntax=docker/dockerfile:1
# ── Rust build stage ──────────────────────────────────────────────────────
FROM rust:1.96-bookworm AS rust-builder
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release \
    -p mangahue-api \
    -p mangahue-worker \
    -p mangahue-telegram \
    -p mangahue-cli

# ── Full runtime (PyTorch + training/eval scripts) ───────────────────────
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MANGAHUE_PYTHON=python \
    RUST_LOG=mangahue=info,tower_http=info

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mangahue \
    && useradd --no-log-init --create-home --uid 10001 --gid mangahue mangahue

WORKDIR /app

COPY --from=rust-builder /src/target/release/mangahue-api /usr/local/bin/mangahue-api
COPY --from=rust-builder /src/target/release/mangahue-worker /usr/local/bin/mangahue-worker
COPY --from=rust-builder /src/target/release/mangahue-telegram /usr/local/bin/mangahue-telegram
COPY --from=rust-builder /src/target/release/mangahue /usr/local/bin/mangahue

COPY python /app/python
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -e '/app/python[pretrained]'

COPY configs /app/configs
COPY models /app/models
COPY web /app/web

RUN mkdir -p /data/jobs /data/objects \
    && chown -R mangahue:mangahue /app /data \
    && chmod 550 /usr/local/bin/mangahue-* \
    && chmod 500 /app

USER mangahue
EXPOSE 7878

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://127.0.0.1:7878/health || exit 1

ENTRYPOINT ["mangahue-api"]
CMD ["--addr", "0.0.0.0:7878", "--jobs-dir", "/data/jobs", "--production-db", "/data/production.db", "--studio-dir", "/app/web/studio", "--styles-dir", "/app/configs/styles"]

# ── Small runtime (no PyTorch — models mounted at runtime) ────────────────
FROM python:3.12-slim-bookworm AS runtime-small
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MANGAHUE_PYTHON=python \
    RUST_LOG=mangahue=info,tower_http=info

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mangahue \
    && useradd --no-log-init --create-home --uid 10001 --gid mangahue mangahue

WORKDIR /app

COPY --from=rust-builder /src/target/release/mangahue-api /usr/local/bin/mangahue-api
COPY --from=rust-builder /src/target/release/mangahue-worker /usr/local/bin/mangahue-worker
COPY --from=rust-builder /src/target/release/mangahue-telegram /usr/local/bin/mangahue-telegram
COPY --from=rust-builder /src/target/release/mangahue /usr/local/bin/mangahue

COPY python /app/python
RUN pip install --no-cache-dir numpy pillow onnx onnxruntime tqdm

COPY configs /app/configs
COPY web /app/web

RUN mkdir -p /data/jobs /data/objects \
    && chown -R mangahue:mangahue /app /data \
    && chmod 550 /usr/local/bin/mangahue-* \
    && chmod 500 /app

USER mangahue
EXPOSE 7878

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://127.0.0.1:7878/health || exit 1

ENTRYPOINT ["mangahue-api"]
CMD ["--addr", "0.0.0.0:7878", "--jobs-dir", "/data/jobs", "--production-db", "/data/production.db", "--studio-dir", "/app/web/studio", "--styles-dir", "/app/configs/styles"]
