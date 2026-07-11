FROM rust:1.96-bookworm AS rust-builder
WORKDIR /src
COPY . .
RUN cargo build --release -p mangahue-api -p mangahue-worker -p mangahue-telegram -p mangahue-cli

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 \
    MANGAHUE_PYTHON=python \
    RUST_LOG=mangahue=info,tower_http=info
RUN useradd --create-home --uid 10001 mangahue
WORKDIR /app
COPY python /app/python
RUN pip install --no-cache-dir -e /app/python
COPY --from=rust-builder /src/target/release/mangahue-api /usr/local/bin/mangahue-api
COPY --from=rust-builder /src/target/release/mangahue-worker /usr/local/bin/mangahue-worker
COPY --from=rust-builder /src/target/release/mangahue-telegram /usr/local/bin/mangahue-telegram
COPY --from=rust-builder /src/target/release/mangahue /usr/local/bin/mangahue
COPY configs /app/configs
COPY models /app/models
COPY web /app/web
RUN mkdir -p /data/jobs /data/objects && chown -R mangahue:mangahue /app /data
USER mangahue
EXPOSE 7878
ENTRYPOINT ["mangahue-api"]
CMD ["--addr", "0.0.0.0:7878", "--jobs-dir", "/data/jobs", "--production-db", "/data/production.db", "--studio-dir", "/app/web/studio", "--styles-dir", "/app/configs/styles"]
