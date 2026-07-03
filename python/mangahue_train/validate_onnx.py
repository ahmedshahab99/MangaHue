from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a MangaHue ONNX model contract")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    size = int(metadata["input_size"])
    channels = int(metadata["input_channels"])
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sample = np.zeros((1, channels, size, size), dtype=np.float32)
    output = session.run([output_name], {input_name: sample})[0]
    expected = (1, 3, size, size)
    if output.shape != expected:
        raise SystemExit(f"bad output shape: expected {expected}, got {output.shape}")
    if output.min() < -1e-4 or output.max() > 1.0001:
        raise SystemExit("output range is outside 0..1")
    print("ONNX contract OK")


if __name__ == "__main__":
    main()

