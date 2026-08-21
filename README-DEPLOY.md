# MangaHue production deployment

Production is one Rust service, one pretrained model, and one immutable Git release.

## Runtime contract

- Service: `mangahue-pro.service`
- Binary: `/opt/MangaHue-current/bin/mangahue-telegram`
- Release: `/opt/MangaHue-current -> /opt/MangaHue-releases/<40-character-git-sha>`
- Model: `/opt/MangaHue/models/ddcolor-full/ddcolor-full-fp32.onnx`
- Model SHA-256: `08e7ae0362faddb4411a4412b5e017624da77f72d085f13862ea50fa4715a7e4`
- Config: `configs/engine.default.yaml`
- Style: `configs/styles/soft_cel_anime_v2.yaml`
- Inference: native Rust ONNX backend; no Python worker
- Concurrency: one job; 16 queued globally; one queued per chat
- Daily quota: disabled
- User-image retention: 24 hours

The Telegram binary validates the model filename and checksum at startup. The production config disables ensemble inference, pyramid inference, palette color authoring, agent deliberation, hint color authoring, learning, and cel-shading modes. Rust preprocessing and postprocessing remain because they protect ink, text, speech-bubble whites, and remove chroma artifacts.

## Release flow

1. Open a pull request into `main`.
2. `MangaHue CI` must pass formatting, Clippy with warnings denied, all Rust tests, Python compatibility tests, shell syntax, and the container smoke test.
3. Merging to `main` triggers `Immutable production release` for that exact successful SHA.
4. The workflow uploads a checksummed archive and runs `/usr/local/sbin/mangahue-promote <SHA>`.
5. Promotion atomically changes `/opt/MangaHue-current`, restarts systemd, and rolls back if the service is not healthy.

Do not deploy with the old mutable `vps_build.py`, `vps_deploy.py`, or hand-edited systemd workflow. A live service whose release directory is not a full Git SHA is configuration drift.

## Verification

```bash
readlink -f /opt/MangaHue-current
cat /opt/MangaHue-current/REVISION
systemctl is-active mangahue-pro.service
systemctl show mangahue-pro.service -p ExecStart -p MainPID -p NRestarts
sha256sum -c /opt/MangaHue-current/configs/model-checksums/ddcolor-full-fp32.sha256
journalctl -u mangahue-pro.service -n 100 --no-pager
```

The `ExecStart` line must contain only `ddcolor-full-fp32.onnx` and `engine.default.yaml`. A page is delivered only when inference completed, visible color coverage exists, the output changed, and the final verdict begins with `production_ready`. Failed pages are reported explicitly; original or unchanged pages are never passed through as colored output.
