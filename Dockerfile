# syntax=docker/dockerfile:1
FROM rust:1.96-bookworm AS rust-builder
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --locked --release \
        -p mangahue-api \
        -p mangahue-worker \
        -p mangahue-cli \
        -p mangahue-telegram \
    && install -d /out \
    && install -m 0755 /src/target/release/mangahue-api /out/mangahue-api \
    && install -m 0755 /src/target/release/mangahue-worker /out/mangahue-worker \
    && install -m 0755 /src/target/release/mangahue /out/mangahue \
    && install -m 0755 /src/target/release/mangahue-telegram /out/mangahue-telegram

FROM debian:bookworm-slim AS runtime-small
ENV RUST_LOG=mangahue=info,tower_http=info
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mangahue \
    && useradd --no-log-init --create-home --uid 10001 --gid mangahue mangahue

WORKDIR /app
COPY --from=rust-builder /out/ /usr/local/bin/
COPY configs /app/configs

RUN install -d -o mangahue -g mangahue /data/jobs /data/objects /app/models \
    && chown -R mangahue:mangahue /app /data \
    && chmod 0550 /usr/local/bin/mangahue /usr/local/bin/mangahue-* \
    && chmod 0500 /app

USER mangahue
EXPOSE 7878
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://127.0.0.1:7878/health || exit 1
ENTRYPOINT ["mangahue-api"]
CMD ["--addr", "0.0.0.0:7878", "--jobs-dir", "/data/jobs", "--production-db", "/data/production.db", "--styles-dir", "/app/configs/styles", "--public-api"]

FROM runtime-small AS runtime
