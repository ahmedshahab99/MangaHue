# MangaHue models

MangaHue accepts one colorization artifact:

`ddcolor-full-fp32.onnx`

The file is not committed because of its size. Its required SHA-256 is pinned in Rust and in `configs/model-checksums/ddcolor-full-fp32.sha256`; startup fails closed when the name, size, or checksum is wrong.

Native Rust inference loads the bundled ONNX Runtime from `models/onnxruntime/`. The `models/anime-face/` LBP cascade is a deterministic face-region detector, not a colorization model.
