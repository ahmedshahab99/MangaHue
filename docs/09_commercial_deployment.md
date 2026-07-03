# Commercial Deployment

MangaHue is shaped for a free-with-ads plus premium model.

Free tier:

- Page/day limits.
- Queue-based processing.
- Standard resolution.
- Optional ad token validation at the API layer.

Premium tier:

- Higher limits.
- Batch jobs.
- Character-memory projects.
- Higher resolution.
- Faster queue priority.

Cost controls:

- Rust preprocessing and postprocessing remain CPU-efficient.
- ONNX model can start lightweight before advanced diffusion backends.
- Batch and job APIs are separated from model training.
- Correction memory creates training data from real product usage.

Deployment shape:

- Rust API service.
- Object storage for inputs/results.
- Queue backend for jobs.
- Database for projects, characters, corrections, and usage limits.
- ONNX runtime worker pool.

